"""Fail-closed CPU contracts for a one-supervisor TP2/DP2 execution.

The production executor remains single-rank.  This module is an independent
adapter that freezes launch, routing, logical-run, and fresh-process retry
semantics without making distributed execution claimable.  Diagnostic callers
may exercise the state machine with fakes; formal entry checks source-owned
native-terminal capability before invoking a launcher and is blocked in the
current release.
"""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal, Protocol
from urllib.parse import urlsplit

from lightcone_spec.orchestration.industrial import IndustrialPhysicalAssignment
from lightcone_spec.orchestration.native_terminal import (
    canonical_json_bytes,
    canonical_sha256,
)
from lightcone_spec.orchestration.native_terminal_gang import (
    NativeTerminalGangAuthorityBlocked,
    NativeTerminalGangBinding,
    ReplicaRouteBinding,
    require_native_terminal_gang_release_capability,
)
from lightcone_spec.runtime.distributed import (
    RuntimeTopologyMode,
    VerifiedDistributedRuntimeGpuProof,
)
from lightcone_spec.telemetry.records import RunRecord

SERVING_GANG_EXECUTION_PROTOCOL_SHA256 = canonical_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_serving_gang_execution_protocol",
        "requirements": [
            "one_loopback_sglang_supervisor_not_one_server_per_rank",
            "exact_tp2_or_two_routed_tp1_rank_and_gpu_coverage",
            "assignment_bound_frontend_nccl_and_reserved_control_ports",
            "release_route_plan_injects_routed_dp_rank_not_caller",
            "logical_coordinator_run_record_with_true_tp_dp_world",
            "standalone_attempt_never_reuses_a_supervisor",
            "failure_poison_requires_exact_fresh_restart_lineage",
            "formal_source_capability_and_native_result_pointer_before_launch",
            "single_node_only",
        ],
    }
)

_METHODS = frozenset(
    {
        "target_only",
        "static",
        "tts",
        "l0",
        "onlinespec_ogd",
        "onlinespec_opt",
        "onlinespec_ens",
    }
)
_SHA_LENGTH = 64
_SAFE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:@+-"
)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(label: str, value: object) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or any(character not in _SAFE_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be one safe non-empty identity")
    return value


def _require_nonnegative_int(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(label: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strict_object(
    label: str, value: object, fields: frozenset[str]
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a JSON object")
    if set(value) != fields:
        missing = sorted(fields - set(value))
        unknown = sorted(set(value) - fields)
        raise ValueError(f"{label} fields differ: missing={missing}, unknown={unknown}")
    return value


def _strict_list(label: str, value: object) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON array")
    return value


def _validate_json(value: object) -> None:
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number is forbidden")
        return
    if type(value) is str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("unpaired JSON surrogate is forbidden")
        return
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("JSON object keys must be strings")
            _validate_json(item)
        return
    raise TypeError(f"unsupported strict JSON value {type(value).__name__}")


@dataclass(frozen=True)
class ServingGangPortPlan:
    frontend_http: int
    nccl_rendezvous: int
    reserved_control: int
    reserved_control_mode: Literal[
        "reserved_non_listening_until_native_control_contract"
    ] = "reserved_non_listening_until_native_control_contract"

    def __post_init__(self) -> None:
        ports = (self.frontend_http, self.nccl_rendezvous, self.reserved_control)
        if any(
            type(port) is not int or port < 1024 or port > 65_535 for port in ports
        ) or ports != tuple(sorted(set(ports))):
            raise ValueError("gang ports must be sorted unique values in [1024, 65535]")
        if self.reserved_control_mode != (
            "reserved_non_listening_until_native_control_contract"
        ):
            raise ValueError("gang reserved-control mode is unsupported")

    @property
    def ports(self) -> tuple[int, int, int]:
        return self.frontend_http, self.nccl_rendezvous, self.reserved_control

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "lightcone_serving_gang_port_plan",
            "frontend_http": self.frontend_http,
            "nccl_rendezvous": self.nccl_rendezvous,
            "reserved_control": self.reserved_control,
            "reserved_control_mode": self.reserved_control_mode,
        }

    @classmethod
    def from_dict(cls, value: object) -> ServingGangPortPlan:
        row = _strict_object(
            "serving gang port plan",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "frontend_http",
                    "nccl_rendezvous",
                    "reserved_control",
                    "reserved_control_mode",
                }
            ),
        )
        if (
            row["schema_version"] != 1
            or row["kind"] != "lightcone_serving_gang_port_plan"
        ):
            raise ValueError("serving gang port plan schema is unsupported")
        return cls(
            frontend_http=row["frontend_http"],
            nccl_rendezvous=row["nccl_rendezvous"],
            reserved_control=row["reserved_control"],
            reserved_control_mode=row["reserved_control_mode"],
        )


@dataclass(frozen=True)
class ServingGangRankLaunchBinding:
    global_rank: int
    tensor_parallel_rank: int
    data_parallel_rank: int
    local_rank: int
    gpu_uuid: str
    rank_config_sha256: str
    topology_receipt_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("launch global rank", self.global_rank),
            ("launch TP rank", self.tensor_parallel_rank),
            ("launch DP rank", self.data_parallel_rank),
            ("launch local rank", self.local_rank),
        ):
            _require_nonnegative_int(label, value)
        if self.local_rank != self.global_rank:
            raise ValueError("single-node launch local rank must equal global rank")
        if (
            type(self.gpu_uuid) is not str
            or not self.gpu_uuid.startswith("GPU-")
            or any(character not in _SAFE_CHARS for character in self.gpu_uuid)
        ):
            raise ValueError("launch rank requires one canonical physical GPU UUID")
        _require_sha256("launch rank config", self.rank_config_sha256)
        _require_sha256("launch topology receipt", self.topology_receipt_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "global_rank": self.global_rank,
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "data_parallel_rank": self.data_parallel_rank,
            "local_rank": self.local_rank,
            "gpu_uuid": self.gpu_uuid,
            "rank_config_sha256": self.rank_config_sha256,
            "topology_receipt_sha256": self.topology_receipt_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ServingGangRankLaunchBinding:
        row = _strict_object(
            "serving gang rank launch binding",
            value,
            frozenset(
                {
                    "global_rank",
                    "tensor_parallel_rank",
                    "data_parallel_rank",
                    "local_rank",
                    "gpu_uuid",
                    "rank_config_sha256",
                    "topology_receipt_sha256",
                }
            ),
        )
        return cls(**row)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _flag_value(argv: tuple[str, ...], flag: str) -> str:
    positions = tuple(index for index, item in enumerate(argv) if item == flag)
    if len(positions) != 1:
        raise ValueError(f"supervisor argv requires exactly one {flag}")
    index = positions[0]
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ValueError(f"supervisor argv {flag} lacks one value")
    return argv[index + 1]


@dataclass(frozen=True)
class ServingGangLaunch:
    mode: Literal["tp2_dp1", "tp1_dp2"]
    method: str
    run_id: str
    gang_binding_sha256: str
    topology_sha256: str
    topology_receipt_set_sha256: str
    rank_config_set_sha256: str
    physical_assignment_sha256: str
    scheduler_assignment_sha256: str
    host_id: str
    fixed_instance_gpu_count: int
    ports: ServingGangPortPlan
    ranks: tuple[ServingGangRankLaunchBinding, ...]
    base_url: str
    supervisor_argv: tuple[str, ...]
    supervisor_environment: tuple[tuple[str, str], ...]
    route_plan_sha256: str | None

    def __post_init__(self) -> None:
        if self.mode not in {"tp2_dp1", "tp1_dp2"}:
            raise ValueError("serving gang launch mode is unsupported")
        if self.method not in _METHODS:
            raise ValueError("serving gang launch method is unsupported")
        _require_text("serving gang run ID", self.run_id)
        for label, value in (
            ("gang binding", self.gang_binding_sha256),
            ("gang topology", self.topology_sha256),
            ("gang topology receipt set", self.topology_receipt_set_sha256),
            ("gang rank-config set", self.rank_config_set_sha256),
            ("gang physical assignment", self.physical_assignment_sha256),
            ("gang scheduler assignment", self.scheduler_assignment_sha256),
        ):
            _require_sha256(label, value)
        _require_text("serving gang host", self.host_id)
        _require_positive_int(
            "serving gang fixed-instance GPU count",
            self.fixed_instance_gpu_count,
        )
        if type(self.ports) is not ServingGangPortPlan:
            raise TypeError("serving gang launch requires an exact port plan")
        if (
            type(self.ranks) is not tuple
            or len(self.ranks) != 2
            or any(
                type(rank) is not ServingGangRankLaunchBinding for rank in self.ranks
            )
            or tuple(rank.global_rank for rank in self.ranks) != (0, 1)
            or len({rank.gpu_uuid for rank in self.ranks}) != 2
            or len({rank.rank_config_sha256 for rank in self.ranks}) != 2
            or len({rank.topology_receipt_sha256 for rank in self.ranks}) != 2
        ):
            raise ValueError(
                "serving gang ranks require exact unique world-size-2 coverage"
            )
        if self.rank_config_set_sha256 != canonical_sha256(
            [rank.rank_config_sha256 for rank in self.ranks]
        ):
            raise ValueError("serving gang rank-config set differs from its ranks")
        if self.topology_receipt_set_sha256 != canonical_sha256(
            [rank.topology_receipt_sha256 for rank in self.ranks]
        ):
            raise ValueError("serving gang topology receipt set differs from its ranks")
        expected_rank_pairs = (
            ((0, 0), (1, 0)) if self.mode == "tp2_dp1" else ((0, 0), (0, 1))
        )
        if (
            tuple(
                (rank.tensor_parallel_rank, rank.data_parallel_rank)
                for rank in self.ranks
            )
            != expected_rank_pairs
        ):
            raise ValueError("serving gang ranks differ from the declared TP/DP mode")
        if self.fixed_instance_gpu_count < len(self.ranks):
            raise ValueError("fixed-instance capacity does not cover the serving gang")
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port != self.ports.frontend_http
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "serving gang base URL must bind the loopback frontend port"
            )
        self._validate_supervisor()

    @property
    def tensor_parallel_size(self) -> int:
        return 2 if self.mode == "tp2_dp1" else 1

    @property
    def data_parallel_size(self) -> int:
        return 1 if self.mode == "tp2_dp1" else 2

    @property
    def world_size(self) -> int:
        return 2

    @property
    def runtime_topology_mode(self) -> RuntimeTopologyMode:
        return self.mode

    def _validate_supervisor(self) -> None:
        argv = self.supervisor_argv
        if (
            type(argv) is not tuple
            or len(argv) < 12
            or any(
                type(item) is not str or not item or "\n" in item or "\x00" in item
                for item in argv
            )
            or not argv[0].startswith("/")
            or argv[1:3] != ("-m", "lightcone_spec.sglang_bridge.launch")
            or argv.count("--") != 1
        ):
            raise ValueError("serving gang requires one registered supervisor argv")
        separator = argv.index("--")
        if separator < 5:
            raise ValueError("serving gang supervisor wrapper argv is incomplete")
        checkout = _flag_value(argv, "--checkout")
        if argv.index("--checkout") >= separator or not checkout.startswith("/"):
            raise ValueError("supervisor checkout must be one absolute wrapper input")
        for forbidden in ("--nnodes", "--node-rank", "--dist-init-addr"):
            if forbidden in argv:
                raise ValueError("multi-node supervisor flags remain blocked")
        expected = {
            "--host": "127.0.0.1",
            "--port": str(self.ports.frontend_http),
            "--nccl-port": str(self.ports.nccl_rendezvous),
            "--tp-size": str(self.tensor_parallel_size),
            "--dp-size": str(self.data_parallel_size),
        }
        if any(
            _flag_value(argv, flag) != value or argv.index(flag) <= separator
            for flag, value in expected.items()
        ):
            raise ValueError("supervisor argv differs from the gang topology/ports")
        if str(self.ports.reserved_control) in argv:
            raise ValueError("reserved control port cannot be advertised as live")
        if self.mode == "tp1_dp2":
            if (
                _flag_value(argv, "--load-balance-method") != "round_robin"
                or argv.index("--load-balance-method") <= separator
            ):
                raise ValueError("DP2 supervisor must use the pinned SGLang router")
            if self.route_plan_sha256 is None:
                raise ValueError("DP2 supervisor requires the release route plan")
            _require_sha256("serving gang route plan", self.route_plan_sha256)
        elif "--load-balance-method" in argv or self.route_plan_sha256 is not None:
            raise ValueError("TP2 supervisor cannot carry a DP route plan/router flag")
        expected_environment = (
            ("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
            (
                "CUDA_VISIBLE_DEVICES",
                ",".join(rank.gpu_uuid for rank in self.ranks),
            ),
        )
        if self.supervisor_environment != expected_environment:
            raise ValueError("supervisor environment differs from ordered rank GPUs")
        raise NativeTerminalGangAuthorityBlocked(
            "diagnostic_serving_gang_compile_authority_unavailable",
            "TP2/DP2 supervisor launch lacks per-gang compile/model content authority",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "lightcone_serving_gang_launch",
            "protocol_sha256": SERVING_GANG_EXECUTION_PROTOCOL_SHA256,
            "mode": self.mode,
            "method": self.method,
            "run_id": self.run_id,
            "gang_binding_sha256": self.gang_binding_sha256,
            "topology_sha256": self.topology_sha256,
            "topology_receipt_set_sha256": self.topology_receipt_set_sha256,
            "rank_config_set_sha256": self.rank_config_set_sha256,
            "physical_assignment_sha256": self.physical_assignment_sha256,
            "scheduler_assignment_sha256": self.scheduler_assignment_sha256,
            "host_id": self.host_id,
            "fixed_instance_gpu_count": self.fixed_instance_gpu_count,
            "ports": self.ports.to_dict(),
            "ranks": [rank.to_dict() for rank in self.ranks],
            "base_url": self.base_url,
            "supervisor_argv": list(self.supervisor_argv),
            "supervisor_environment": [
                list(item) for item in self.supervisor_environment
            ],
            "route_plan_sha256": self.route_plan_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ServingGangLaunch:
        row = _strict_object(
            "serving gang launch",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "protocol_sha256",
                    "mode",
                    "method",
                    "run_id",
                    "gang_binding_sha256",
                    "topology_sha256",
                    "topology_receipt_set_sha256",
                    "rank_config_set_sha256",
                    "physical_assignment_sha256",
                    "scheduler_assignment_sha256",
                    "host_id",
                    "fixed_instance_gpu_count",
                    "ports",
                    "ranks",
                    "base_url",
                    "supervisor_argv",
                    "supervisor_environment",
                    "route_plan_sha256",
                }
            ),
        )
        if (
            row["schema_version"] != 1
            or row["kind"] != "lightcone_serving_gang_launch"
            or row["protocol_sha256"] != SERVING_GANG_EXECUTION_PROTOCOL_SHA256
        ):
            raise ValueError("serving gang launch schema/protocol is unsupported")
        environment = tuple(
            tuple(_strict_list("supervisor environment pair", item))
            for item in _strict_list(
                "supervisor environment", row["supervisor_environment"]
            )
        )
        if any(len(item) != 2 for item in environment):
            raise ValueError("supervisor environment entries must be key/value pairs")
        return cls(
            mode=row["mode"],
            method=row["method"],
            run_id=row["run_id"],
            gang_binding_sha256=row["gang_binding_sha256"],
            topology_sha256=row["topology_sha256"],
            topology_receipt_set_sha256=row["topology_receipt_set_sha256"],
            rank_config_set_sha256=row["rank_config_set_sha256"],
            physical_assignment_sha256=row["physical_assignment_sha256"],
            scheduler_assignment_sha256=row["scheduler_assignment_sha256"],
            host_id=row["host_id"],
            fixed_instance_gpu_count=row["fixed_instance_gpu_count"],
            ports=ServingGangPortPlan.from_dict(row["ports"]),
            ranks=tuple(
                ServingGangRankLaunchBinding.from_dict(item)
                for item in _strict_list("serving gang ranks", row["ranks"])
            ),
            base_url=row["base_url"],
            supervisor_argv=tuple(
                _strict_list("supervisor argv", row["supervisor_argv"])
            ),
            supervisor_environment=environment,
            route_plan_sha256=row["route_plan_sha256"],
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def build_diagnostic_serving_gang_launch(
    *,
    gang_binding: NativeTerminalGangBinding,
    assignment: IndustrialPhysicalAssignment,
    supervisor_argv: Sequence[str],
) -> ServingGangLaunch:
    """Bind one diagnostic supervisor to exact scheduled resources."""

    if type(gang_binding) is not NativeTerminalGangBinding:
        raise TypeError("gang launch requires an exact native terminal gang binding")
    if type(assignment) is not IndustrialPhysicalAssignment:
        raise TypeError("gang launch requires an exact physical assignment")
    first = gang_binding.ranks[0]
    topology = (first.tensor_parallel_size, first.data_parallel_size)
    if gang_binding.world_size != 2 or topology not in {(2, 1), (1, 2)}:
        raise ValueError("diagnostic serving gang is restricted to TP2 or DP2")
    if (
        (assignment.tensor_parallel_size, assignment.data_parallel_size) != topology
        or assignment.gpu_uuids
        != tuple(uuid for group in assignment.rank_groups for uuid in group)
        or len(assignment.gpu_uuids) != gang_binding.world_size
        or len(assignment.ports) != 3
    ):
        raise ValueError("physical assignment differs from the gang topology")
    ports = ServingGangPortPlan(*assignment.ports)
    ranks = tuple(
        ServingGangRankLaunchBinding(
            global_rank=rank.global_rank,
            tensor_parallel_rank=rank.tensor_parallel_rank,
            data_parallel_rank=rank.data_parallel_rank,
            local_rank=rank.global_rank,
            gpu_uuid=assignment.gpu_uuids[rank.global_rank],
            rank_config_sha256=rank.run.rank_config_sha256,
            topology_receipt_sha256=rank.topology_receipt_sha256,
        )
        for rank in gang_binding.ranks
    )
    topology_receipt_set_sha256 = canonical_sha256(
        [rank.topology_receipt_sha256 for rank in gang_binding.ranks]
    )
    if (
        gang_binding.route_plan is not None
        and gang_binding.route_plan.topology_receipt_sha256
        != topology_receipt_set_sha256
    ):
        raise ValueError("gang route plan differs from topology receipt coverage")
    return ServingGangLaunch(
        mode="tp2_dp1" if topology == (2, 1) else "tp1_dp2",
        method=first.run.method,
        run_id=first.run.run_id,
        gang_binding_sha256=gang_binding.sha256,
        topology_sha256=gang_binding.topology_sha256,
        topology_receipt_set_sha256=topology_receipt_set_sha256,
        rank_config_set_sha256=gang_binding.rank_config_set_sha256,
        physical_assignment_sha256=assignment.sha256,
        scheduler_assignment_sha256=assignment.assignment_sha256,
        host_id=assignment.host_id,
        fixed_instance_gpu_count=assignment.fixed_instance_gpu_count,
        ports=ports,
        ranks=ranks,
        base_url=f"http://127.0.0.1:{ports.frontend_http}",
        supervisor_argv=tuple(supervisor_argv),
        supervisor_environment=(
            ("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
            ("CUDA_VISIBLE_DEVICES", ",".join(assignment.gpu_uuids)),
        ),
        route_plan_sha256=gang_binding.route_plan_sha256,
    )


@dataclass(frozen=True)
class RoutedServingRequestBody:
    request_id: str
    serving_gang_launch_sha256: str
    route_plan_sha256: str | None
    route_binding_sha256: str | None
    data_parallel_rank: int | None
    body_json: str

    def __post_init__(self) -> None:
        _require_text("routed serving request ID", self.request_id)
        _require_sha256("routed serving launch", self.serving_gang_launch_sha256)
        if self.route_plan_sha256 is None:
            if (
                self.route_binding_sha256 is not None
                or self.data_parallel_rank is not None
            ):
                raise ValueError("non-DP request cannot carry replica routing")
        else:
            _require_sha256("routed serving route plan", self.route_plan_sha256)
            _require_sha256("routed serving route binding", self.route_binding_sha256)
            _require_nonnegative_int(
                "routed serving data-parallel rank", self.data_parallel_rank
            )
        if type(self.body_json) is not str:
            raise TypeError("routed serving body must be canonical JSON text")
        value = json.loads(self.body_json)
        _validate_json(value)
        if (
            type(value) is not dict
            or canonical_json_bytes(value).decode() != self.body_json
        ):
            raise ValueError("routed serving body is not canonical strict JSON")
        if value.get("rid") != self.request_id:
            raise ValueError("routed serving body changed its request identity")
        if self.data_parallel_rank is None:
            if "routed_dp_rank" in value:
                raise ValueError("non-DP request cannot carry routed_dp_rank")
        elif value.get("routed_dp_rank") != self.data_parallel_rank:
            raise ValueError("routed serving body differs from the release route")
        if "data_parallel_rank" in value:
            raise ValueError("deprecated caller DP rank is forbidden")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_request_body(self) -> dict[str, object]:
        value = json.loads(self.body_json)
        if type(value) is not dict:  # pragma: no cover - constructor invariant
            raise TypeError("routed serving body stopped being an object")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "lightcone_routed_serving_request_body",
            "request_id": self.request_id,
            "serving_gang_launch_sha256": self.serving_gang_launch_sha256,
            "route_plan_sha256": self.route_plan_sha256,
            "route_binding_sha256": self.route_binding_sha256,
            "data_parallel_rank": self.data_parallel_rank,
            "body": self.to_request_body(),
        }


def inject_diagnostic_replica_route(
    *,
    launch: ServingGangLaunch,
    gang_binding: NativeTerminalGangBinding,
    request_id: str,
    request_body: Mapping[str, object],
) -> RoutedServingRequestBody:
    """Inject ``routed_dp_rank`` from the frozen plan, never caller choice."""

    if (
        type(launch) is not ServingGangLaunch
        or type(gang_binding) is not NativeTerminalGangBinding
    ):
        raise TypeError("route injection requires exact launch and gang bindings")
    if launch.gang_binding_sha256 != gang_binding.sha256:
        raise ValueError("route injection gang binding differs from launch")
    _require_text("route injection request ID", request_id)
    if not isinstance(request_body, Mapping):
        raise TypeError("route injection request body must be a mapping")
    body = dict(request_body)
    _validate_json(body)
    if body.get("rid") != request_id:
        raise ValueError("route injection requires exact body rid")
    if "routed_dp_rank" in body or "data_parallel_rank" in body:
        raise ValueError("caller-authored DP rank is forbidden")
    route: ReplicaRouteBinding | None = None
    if gang_binding.route_plan is not None:
        matches = tuple(
            item
            for item in gang_binding.route_plan.routes
            if item.request_id == request_id
        )
        if len(matches) != 1:
            raise ValueError(
                "request is absent or duplicated in the release route plan"
            )
        route = matches[0]
        body["routed_dp_rank"] = route.data_parallel_rank
    else:
        expected_request_ids = (
            gang_binding.ranks[0].run.warmup_request_ids
            + gang_binding.ranks[0].run.scored_request_ids
        )
        if expected_request_ids.count(request_id) != 1:
            raise ValueError(
                "request is absent or duplicated in the gang request contract"
            )
    result = RoutedServingRequestBody(
        request_id=request_id,
        serving_gang_launch_sha256=launch.sha256,
        route_plan_sha256=launch.route_plan_sha256,
        route_binding_sha256=(
            None if route is None else canonical_sha256(route.to_dict())
        ),
        data_parallel_rank=None if route is None else route.data_parallel_rank,
        body_json=canonical_json_bytes(body).decode(),
    )
    return result


@dataclass(frozen=True)
class GangRunRecordTopologySummary:
    run_record: RunRecord
    serving_gang_launch_sha256: str
    rank_config_set_sha256: str
    rank_binding_sha256s: tuple[str, ...]
    route_plan_sha256: str | None
    rank_semantics: Literal["logical_coordinator_observation_ranks_are_not_samples"] = (
        "logical_coordinator_observation_ranks_are_not_samples"
    )

    def __post_init__(self) -> None:
        if type(self.run_record) is not RunRecord:
            raise TypeError("gang run summary requires an exact RunRecord")
        _require_sha256("gang run launch", self.serving_gang_launch_sha256)
        _require_sha256("gang run rank-config set", self.rank_config_set_sha256)
        if (
            type(self.rank_binding_sha256s) is not tuple
            or len(self.rank_binding_sha256s) != 2
            or len(set(self.rank_binding_sha256s)) != 2
        ):
            raise ValueError("gang run summary requires two unique rank bindings")
        for digest in self.rank_binding_sha256s:
            _require_sha256("gang run rank binding", digest)
        if self.route_plan_sha256 is not None:
            _require_sha256("gang run route plan", self.route_plan_sha256)
        record = self.run_record
        if (
            record.tensor_parallel_size not in {1, 2}
            or record.data_parallel_size not in {1, 2}
            or record.tensor_parallel_size * record.data_parallel_size != 2
            or record.world_size != 2
            or record.rank != 0
            or record.rank_config_sha256 is None
            or record.topology_sha256 is None
        ):
            raise ValueError("RunRecord lacks truthful logical gang topology")
        if self.rank_semantics != (
            "logical_coordinator_observation_ranks_are_not_samples"
        ):
            raise ValueError("gang run rank semantics are unsupported")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "lightcone_gang_run_record_topology_summary",
            "run_record": asdict(self.run_record),
            "serving_gang_launch_sha256": self.serving_gang_launch_sha256,
            "rank_config_set_sha256": self.rank_config_set_sha256,
            "rank_binding_sha256s": list(self.rank_binding_sha256s),
            "route_plan_sha256": self.route_plan_sha256,
            "rank_semantics": self.rank_semantics,
        }


def bind_diagnostic_gang_run_record(
    *, record: RunRecord, launch: ServingGangLaunch
) -> GangRunRecordTopologySummary:
    """Make the existing logical RunRecord truthful for one gang observation."""

    if type(record) is not RunRecord or type(launch) is not ServingGangLaunch:
        raise TypeError("gang RunRecord binding requires exact record and launch")
    if record.run_id != launch.run_id:
        raise ValueError("RunRecord belongs to another serving gang run")
    if record.method != launch.method:
        raise ValueError("RunRecord method conflicts with the gang launch")
    expected: dict[str, object] = {
        "rank_config_sha256": launch.ranks[0].rank_config_sha256,
        "topology_sha256": launch.topology_sha256,
        "tensor_parallel_size": launch.tensor_parallel_size,
        "data_parallel_size": launch.data_parallel_size,
        "world_size": launch.world_size,
        "rank": 0,
    }
    for name, value in expected.items():
        current = getattr(record, name)
        if current is not None and current != value:
            raise ValueError(f"RunRecord {name} conflicts with the gang launch")
    bound = replace(record, **expected)
    return GangRunRecordTopologySummary(
        run_record=bound,
        serving_gang_launch_sha256=launch.sha256,
        rank_config_set_sha256=launch.rank_config_set_sha256,
        rank_binding_sha256s=tuple(rank.sha256 for rank in launch.ranks),
        route_plan_sha256=launch.route_plan_sha256,
    )


@dataclass(frozen=True)
class DiagnosticGangCompletion:
    serving_gang_launch_sha256: str
    terminal_aggregate_sha256: str
    run_record_topology_summary_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("diagnostic completion launch", self.serving_gang_launch_sha256)
        _require_sha256(
            "diagnostic completion terminal aggregate",
            self.terminal_aggregate_sha256,
        )
        _require_sha256(
            "diagnostic completion run summary",
            self.run_record_topology_summary_sha256,
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class FormalGangCompletion:
    """First-party all-rank terminal result returned by the native workload."""

    serving_gang_launch_sha256: str
    terminal_aggregate_sha256: str
    rank_terminal_sha256s: tuple[str, str]
    run_record_topology_summary_sha256: str
    gpu_proof_receipt_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("formal completion launch", self.serving_gang_launch_sha256),
            ("formal completion aggregate", self.terminal_aggregate_sha256),
            ("formal completion run summary", self.run_record_topology_summary_sha256),
            ("formal completion GPU proof", self.gpu_proof_receipt_sha256),
        ):
            _require_sha256(label, value)
        if (
            type(self.rank_terminal_sha256s) is not tuple
            or len(self.rank_terminal_sha256s) != 2
            or len(set(self.rank_terminal_sha256s)) != 2
        ):
            raise ValueError("formal completion requires two unique rank terminals")
        for value in self.rank_terminal_sha256s:
            _require_sha256("formal completion rank terminal", value)

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                **asdict(self),
                "rank_terminal_sha256s": list(self.rank_terminal_sha256s),
            }
        )


@dataclass(frozen=True)
class ServingGangAttemptReceipt:
    attempt_id: str
    serving_gang_launch_sha256: str
    previous_failed_attempt_sha256: str | None
    process_identity: str | None
    status: Literal["DIAGNOSTIC_COMPLETE", "TERMINAL_COMPLETE", "POISONED"]
    completion_sha256: str | None
    error_code: str | None
    restart_required: bool

    def __post_init__(self) -> None:
        _require_text("gang attempt ID", self.attempt_id)
        _require_sha256("gang attempt launch", self.serving_gang_launch_sha256)
        if self.previous_failed_attempt_sha256 is not None:
            _require_sha256(
                "gang previous failed attempt", self.previous_failed_attempt_sha256
            )
        if self.process_identity is not None:
            _require_text("gang supervisor process identity", self.process_identity)
        if self.status in {"DIAGNOSTIC_COMPLETE", "TERMINAL_COMPLETE"}:
            if self.process_identity is None:
                raise ValueError("completed diagnostic attempt requires a supervisor")
            _require_sha256("gang attempt completion", self.completion_sha256)
            if self.error_code is not None or self.restart_required:
                raise ValueError("completed diagnostic attempt cannot be poisoned")
        elif self.status == "POISONED":
            _require_text("gang attempt error code", self.error_code)
            if self.completion_sha256 is not None or not self.restart_required:
                raise ValueError("poisoned gang attempt must require a fresh restart")
        else:
            raise ValueError("gang attempt status is unsupported")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


class ServingGangAttemptFailed(RuntimeError):
    """Execution failed after producing one non-claimable poison receipt."""

    def __init__(self, receipt: ServingGangAttemptReceipt) -> None:
        self.receipt = receipt
        super().__init__(f"{receipt.error_code}: fresh serving-gang restart required")


class ServingGangSupervisorHandle(Protocol):
    @property
    def process_identity(self) -> str: ...

    async def wait_ready(self, timeout_s: float) -> None: ...

    async def terminate(self, timeout_s: float) -> None: ...


ServingGangSupervisorLauncher = Callable[
    [ServingGangLaunch], Awaitable[ServingGangSupervisorHandle]
]
DiagnosticGangWorkload = Callable[
    [ServingGangSupervisorHandle], Awaitable[DiagnosticGangCompletion]
]
FormalGangWorkload = Callable[
    [ServingGangSupervisorHandle], Awaitable[FormalGangCompletion]
]


class FreshProcessServingGangExecutor:
    """One-plan CPU state machine; retries always launch a new supervisor."""

    def __init__(
        self,
        launcher: ServingGangSupervisorLauncher,
        *,
        startup_timeout_s: float,
        shutdown_timeout_s: float,
    ) -> None:
        for label, value in (
            ("gang startup timeout", startup_timeout_s),
            ("gang shutdown timeout", shutdown_timeout_s),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{label} must be numeric")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        self._launcher = launcher
        self._startup_timeout_s = float(startup_timeout_s)
        self._shutdown_timeout_s = float(shutdown_timeout_s)
        self._last_receipt: ServingGangAttemptReceipt | None = None
        self._seen_attempt_ids: set[str] = set()
        self._seen_processes: set[str] = set()
        self._active = False

    @property
    def last_receipt(self) -> ServingGangAttemptReceipt | None:
        return self._last_receipt

    async def execute_fresh(
        self,
        *,
        launch: ServingGangLaunch,
        attempt_id: str,
        previous_failed_attempt_sha256: str | None,
        workload: DiagnosticGangWorkload,
    ) -> ServingGangAttemptReceipt:
        if self._active:
            raise RuntimeError("serving gang attempt is already active")
        _require_text("gang fresh attempt ID", attempt_id)
        if attempt_id in self._seen_attempt_ids:
            raise ValueError("serving gang attempt identity was reused")
        prior = self._last_receipt
        if prior is None:
            if previous_failed_attempt_sha256 is not None:
                raise ValueError("first gang attempt cannot claim retry lineage")
        elif prior.status != "POISONED":
            raise RuntimeError("completed standalone gang executor cannot be reused")
        elif previous_failed_attempt_sha256 != prior.sha256:
            raise ValueError("fresh gang restart lacks exact failed-attempt lineage")
        elif prior.serving_gang_launch_sha256 != launch.sha256:
            raise ValueError("fresh gang restart changed the serving-gang launch")
        self._seen_attempt_ids.add(attempt_id)
        self._active = True
        handle: ServingGangSupervisorHandle | None = None
        process_identity: str | None = None
        terminated = False
        try:
            handle = await self._launcher(launch)
            process_identity = _require_text(
                "gang supervisor process identity", handle.process_identity
            )
            if process_identity in self._seen_processes:
                raise RuntimeError("fresh restart reused an old supervisor process")
            self._seen_processes.add(process_identity)
            await handle.wait_ready(self._startup_timeout_s)
            completion = await workload(handle)
            if (
                type(completion) is not DiagnosticGangCompletion
                or completion.serving_gang_launch_sha256 != launch.sha256
            ):
                raise ValueError("diagnostic workload returned foreign completion")
            await handle.terminate(self._shutdown_timeout_s)
            terminated = True
            receipt = ServingGangAttemptReceipt(
                attempt_id=attempt_id,
                serving_gang_launch_sha256=launch.sha256,
                previous_failed_attempt_sha256=previous_failed_attempt_sha256,
                process_identity=process_identity,
                status="DIAGNOSTIC_COMPLETE",
                completion_sha256=completion.sha256,
                error_code=None,
                restart_required=False,
            )
            self._last_receipt = receipt
            return receipt
        except BaseException as error:
            termination_failed = False
            if handle is not None and not terminated:
                try:
                    await handle.terminate(self._shutdown_timeout_s)
                except BaseException:  # noqa: BLE001 - every cleanup failure poisons
                    termination_failed = True
            receipt = ServingGangAttemptReceipt(
                attempt_id=attempt_id,
                serving_gang_launch_sha256=launch.sha256,
                previous_failed_attempt_sha256=previous_failed_attempt_sha256,
                process_identity=process_identity,
                status="POISONED",
                completion_sha256=None,
                error_code=(
                    "gang_supervisor_termination_failed"
                    if termination_failed
                    else "gang_execution_failed"
                ),
                restart_required=True,
            )
            self._last_receipt = receipt
            raise ServingGangAttemptFailed(receipt) from error
        finally:
            self._active = False


def require_formal_serving_gang_launch(
    *,
    launch: ServingGangLaunch,
    claimed_capability_sha256: str,
    verified_gpu_proof: VerifiedDistributedRuntimeGpuProof | None = None,
) -> None:
    """Block before launch until source pin and native result pointer both exist."""

    if type(verified_gpu_proof) is not VerifiedDistributedRuntimeGpuProof:
        raise NativeTerminalGangAuthorityBlocked(
            "distributed_runtime_gpu_proof_unavailable",
            "formal gang launch lacks a verified root-signed GPU proof",
        )
    if (
        verified_gpu_proof.topology_mode != launch.runtime_topology_mode
        or verified_gpu_proof.topology_sha256 != launch.topology_sha256
        or verified_gpu_proof.gpu_uuids != tuple(rank.gpu_uuid for rank in launch.ranks)
    ):
        raise ValueError("distributed GPU proof belongs to another launch identity")
    require_native_terminal_gang_release_capability(
        topology_sha256=launch.topology_sha256,
        claimed_capability_sha256=claimed_capability_sha256,
    )


async def execute_formal_serving_gang(
    *,
    launch: ServingGangLaunch,
    claimed_capability_sha256: str,
    verified_gpu_proof: VerifiedDistributedRuntimeGpuProof | None = None,
    launcher: ServingGangSupervisorLauncher,
    attempt_id: str | None = None,
    previous_failed_attempt_sha256: str | None = None,
    workload: FormalGangWorkload | None = None,
    startup_timeout_s: float = 180.0,
    shutdown_timeout_s: float = 60.0,
) -> ServingGangAttemptReceipt:
    """Run one fresh proof-gated gang and poison every incomplete lifecycle."""

    require_formal_serving_gang_launch(
        launch=launch,
        claimed_capability_sha256=claimed_capability_sha256,
        verified_gpu_proof=verified_gpu_proof,
    )
    if attempt_id is None or workload is None:
        raise ValueError("formal gang execution requires an attempt ID and workload")
    _require_text("formal gang attempt ID", attempt_id)
    if previous_failed_attempt_sha256 is not None:
        _require_sha256(
            "formal previous failed attempt",
            previous_failed_attempt_sha256,
        )
    for label, value in (
        ("formal gang startup timeout", startup_timeout_s),
        ("formal gang shutdown timeout", shutdown_timeout_s),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{label} must be finite and positive")

    handle: ServingGangSupervisorHandle | None = None
    process_identity: str | None = None
    terminated = False
    try:
        handle = await launcher(launch)
        process_identity = _require_text(
            "formal gang supervisor process identity", handle.process_identity
        )
        await handle.wait_ready(float(startup_timeout_s))
        completion = await workload(handle)
        if type(completion) is not FormalGangCompletion:
            raise TypeError("formal gang workload returned a non-formal completion")
        if (
            completion.serving_gang_launch_sha256 != launch.sha256
            or completion.gpu_proof_receipt_sha256 != verified_gpu_proof.receipt_sha256
        ):
            raise ValueError("formal gang completion belongs to another authority")
        await handle.terminate(float(shutdown_timeout_s))
        terminated = True
        return ServingGangAttemptReceipt(
            attempt_id=attempt_id,
            serving_gang_launch_sha256=launch.sha256,
            previous_failed_attempt_sha256=previous_failed_attempt_sha256,
            process_identity=process_identity,
            status="TERMINAL_COMPLETE",
            completion_sha256=completion.sha256,
            error_code=None,
            restart_required=False,
        )
    except BaseException as error:
        termination_failed = False
        if handle is not None and not terminated:
            try:
                await handle.terminate(float(shutdown_timeout_s))
            except BaseException:  # noqa: BLE001 - cleanup failure poisons evidence
                termination_failed = True
        receipt = ServingGangAttemptReceipt(
            attempt_id=attempt_id,
            serving_gang_launch_sha256=launch.sha256,
            previous_failed_attempt_sha256=previous_failed_attempt_sha256,
            process_identity=process_identity,
            status="POISONED",
            completion_sha256=None,
            error_code=(
                "gang_supervisor_termination_failed"
                if termination_failed
                else "gang_execution_failed"
            ),
            restart_required=True,
        )
        raise ServingGangAttemptFailed(receipt) from error


__all__ = [
    "SERVING_GANG_EXECUTION_PROTOCOL_SHA256",
    "DiagnosticGangCompletion",
    "DiagnosticGangWorkload",
    "FormalGangCompletion",
    "FormalGangWorkload",
    "FreshProcessServingGangExecutor",
    "GangRunRecordTopologySummary",
    "RoutedServingRequestBody",
    "ServingGangAttemptFailed",
    "ServingGangAttemptReceipt",
    "ServingGangLaunch",
    "ServingGangPortPlan",
    "ServingGangRankLaunchBinding",
    "ServingGangSupervisorHandle",
    "ServingGangSupervisorLauncher",
    "bind_diagnostic_gang_run_record",
    "build_diagnostic_serving_gang_launch",
    "execute_formal_serving_gang",
    "inject_diagnostic_replica_route",
    "require_formal_serving_gang_launch",
]
