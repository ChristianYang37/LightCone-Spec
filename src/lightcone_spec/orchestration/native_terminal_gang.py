"""CPU contracts for rank-complete native terminal evidence.

This module deliberately does not unlock a distributed serving topology.  It
defines the strict host-side protocol, durable aggregate, and fail-closed
client needed before the pinned SGLang patch can expose first-party all-rank
receipts.  Formal consumers must pass the source-owned capability gate before
opening any evidence path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import stat
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeVar

from lightcone_spec.orchestration.formal_terminal_shards import (
    SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND,
    publish_scalable_native_terminal_artifact,
    reopen_scalable_native_terminal_artifact,
)
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalProvider,
    NativeTerminalRunBinding,
    TerminalRequestExpectation,
    ValidatedNativeTerminalEvidence,
    canonical_json_bytes,
    canonical_sha256,
    validate_native_terminal_artifact,
)
from lightcone_spec.runtime.attestation import TrustedAttesterPolicy
from lightcone_spec.runtime.distributed import (
    AdaptationCollectiveMode,
    CohortRouteIdentity,
    DistributedControlMode,
    ReplicaLocalRouter,
    RuntimeTopologyMode,
    TopologyReceiptSet,
    VerifiedDistributedRuntimeGpuProof,
    adaptation_collective_mode,
    distributed_control_mode,
    registered_runtime_topology_mode,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

NATIVE_TERMINAL_GANG_HOOK = (
    "sglang.schema_v3.content_bound_terminal_speculative_evidence.gang_v1"
)
NATIVE_TERMINAL_GANG_PROTOCOL_SHA256 = canonical_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_native_terminal_gang_protocol",
        "requirements": [
            "source_owned_distributed_capability_before_path_access",
            "global_rank_sorted_exact_world_coverage",
            "tp_rank_request_replication_and_dp_replica_partition",
            "native_actual_route_proof_for_dp2",
            "all_rank_begin_reset_finalize_or_poison",
            "rank_artifacts_published_before_aggregate",
            "strict_json_nofollow_single_link_stable_reopen_sidecar",
            "aggregate_is_only_completion_authority",
            "single_node_only",
        ],
    }
)

# This is the CPU-audited native producer identity, not GPU evidence.  Formal
# consumers additionally require an exact root-verified qualification token.
NATIVE_TERMINAL_GANG_RELEASE_CAPABILITY_SHA256 = canonical_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_native_terminal_gang_release_capability",
        "protocol_sha256": NATIVE_TERMINAL_GANG_PROTOCOL_SHA256,
        "sglang_upstream": "3312645a307453893a00778592f105581e3d1c3d",
        "supported_modes": ["tp2_dp1", "tp1_dp2"],
        "evidence_status": "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF",
    }
)

_SHA256_LENGTH = 64
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_SAFE_TEXT_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:@+-"
)
_T = TypeVar("_T")


class NativeTerminalGangAuthorityBlocked(RuntimeError):
    """Named fail-closed result for a missing first-party producer or pin."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def require_native_terminal_gang_release_capability(
    *, topology_sha256: str, claimed_capability_sha256: str
) -> str:
    """Resolve a source-owned distributed capability before path access."""

    _require_sha256("topology", topology_sha256)
    claimed = _require_sha256("claimed gang capability", claimed_capability_sha256)
    if claimed != NATIVE_TERMINAL_GANG_RELEASE_CAPABILITY_SHA256:
        raise NativeTerminalGangAuthorityBlocked(
            "native_terminal_gang_release_capability_unavailable",
            "the claim differs from the source-audited all-rank terminal producer",
        )
    return claimed


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
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
        or any(character not in _SAFE_TEXT_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be one safe non-empty identity")
    return value


def _require_positive_int(label: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
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


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _validate_json(value: object) -> None:
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("unpaired JSON surrogate is forbidden")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number is forbidden")
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


def _strict_json(body: bytes, *, label: str) -> object:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    _validate_json(value)
    return value


def _absolute_resolved(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or value.resolve(strict=False) != value:
        raise ValueError(f"{label} must be absolute, resolved, and symlink-free")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_file(path: Path, *, label: str) -> bytes:
    _absolute_resolved(path, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} is not a readable regular file") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size < 1
            or opened.st_size > _MAX_ARTIFACT_BYTES
        ):
            raise RuntimeError(f"{label} is not one supported single-link file")
        body = os.read(descriptor, opened.st_size + 1)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (
            len(body) != opened.st_size
            or _stat_identity(opened) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


def _publish_exclusive(path: Path, body: bytes, *, label: str) -> None:
    _absolute_resolved(path, label=label)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError(f"{label} parent must be an existing regular directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(body)
        written = 0
        while written < len(view):
            chunk = os.write(descriptor, view[written:])
            if chunk < 1:
                raise RuntimeError(f"{label} write made no progress")
            written += chunk
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path, *, label: str) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(opened.st_mode) or _stat_identity(opened) != _stat_identity(
            current
        ):
            raise RuntimeError(f"{label} directory identity changed")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class NativeTerminalRankBinding:
    run: NativeTerminalRunBinding
    topology_sha256: str
    topology_receipt_sha256: str
    global_rank: int
    tensor_parallel_rank: int
    data_parallel_rank: int
    tensor_parallel_size: int
    data_parallel_size: int
    world_size: int
    node_count: int
    node_rank: int

    def __post_init__(self) -> None:
        if type(self.run) is not NativeTerminalRunBinding:
            raise TypeError("rank terminal binding requires NativeTerminalRunBinding")
        self.run.validate()
        _require_sha256("rank topology", self.topology_sha256)
        _require_sha256("rank topology receipt", self.topology_receipt_sha256)
        for label, value in (
            ("global rank", self.global_rank),
            ("tensor-parallel rank", self.tensor_parallel_rank),
            ("data-parallel rank", self.data_parallel_rank),
        ):
            _require_nonnegative_int(label, value)
        for label, value in (
            ("tensor-parallel size", self.tensor_parallel_size),
            ("data-parallel size", self.data_parallel_size),
            ("world size", self.world_size),
            ("node count", self.node_count),
        ):
            _require_positive_int(label, value)
        _require_nonnegative_int("node rank", self.node_rank)
        if self.node_count != 1 or self.node_rank != 0:
            raise ValueError("native terminal gang is restricted to one local node")
        if self.world_size != self.tensor_parallel_size * self.data_parallel_size:
            raise ValueError("rank world size differs from TP*DP")
        registered_runtime_topology_mode(
            self.tensor_parallel_size,
            self.data_parallel_size,
            self.node_count,
        )
        if self.tensor_parallel_rank >= self.tensor_parallel_size:
            raise ValueError("tensor-parallel rank is outside the topology")
        if self.data_parallel_rank >= self.data_parallel_size:
            raise ValueError("data-parallel rank is outside the topology")
        expected = (
            self.data_parallel_rank * self.tensor_parallel_size
            + self.tensor_parallel_rank
        )
        if self.global_rank != expected:
            raise ValueError("global rank differs from DP-major TP rank order")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "lightcone_native_terminal_rank_binding",
            "run": _run_binding_to_dict(self.run),
            "topology_sha256": self.topology_sha256,
            "topology_receipt_sha256": self.topology_receipt_sha256,
            "global_rank": self.global_rank,
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "data_parallel_rank": self.data_parallel_rank,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "world_size": self.world_size,
            "node_count": self.node_count,
            "node_rank": self.node_rank,
        }

    @classmethod
    def from_dict(cls, value: object) -> NativeTerminalRankBinding:
        row = _strict_object(
            "rank terminal binding",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "run",
                    "topology_sha256",
                    "topology_receipt_sha256",
                    "global_rank",
                    "tensor_parallel_rank",
                    "data_parallel_rank",
                    "tensor_parallel_size",
                    "data_parallel_size",
                    "world_size",
                    "node_count",
                    "node_rank",
                }
            ),
        )
        if row["schema_version"] != 1 or row["kind"] != (
            "lightcone_native_terminal_rank_binding"
        ):
            raise ValueError("rank terminal binding schema is unsupported")
        return cls(
            run=_run_binding_from_dict(row["run"]),
            topology_sha256=row["topology_sha256"],
            topology_receipt_sha256=row["topology_receipt_sha256"],
            global_rank=row["global_rank"],
            tensor_parallel_rank=row["tensor_parallel_rank"],
            data_parallel_rank=row["data_parallel_rank"],
            tensor_parallel_size=row["tensor_parallel_size"],
            data_parallel_size=row["data_parallel_size"],
            world_size=row["world_size"],
            node_count=row["node_count"],
            node_rank=row["node_rank"],
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ReplicaRouteBinding:
    request_id: str
    cohort_identity_sha256: str
    data_parallel_rank: int
    rank_group: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_text("route request ID", self.request_id)
        _require_sha256("route cohort identity", self.cohort_identity_sha256)
        _require_nonnegative_int("route data-parallel rank", self.data_parallel_rank)
        if (
            type(self.rank_group) is not tuple
            or not self.rank_group
            or any(type(rank) is not int or rank < 0 for rank in self.rank_group)
            or self.rank_group != tuple(sorted(set(self.rank_group)))
        ):
            raise ValueError("route rank group must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "cohort_identity_sha256": self.cohort_identity_sha256,
            "data_parallel_rank": self.data_parallel_rank,
            "rank_group": list(self.rank_group),
        }

    @classmethod
    def from_dict(cls, value: object) -> ReplicaRouteBinding:
        row = _strict_object(
            "replica route binding",
            value,
            frozenset(
                {
                    "request_id",
                    "cohort_identity_sha256",
                    "data_parallel_rank",
                    "rank_group",
                }
            ),
        )
        return cls(
            request_id=row["request_id"],
            cohort_identity_sha256=row["cohort_identity_sha256"],
            data_parallel_rank=row["data_parallel_rank"],
            rank_group=tuple(_strict_list("route rank group", row["rank_group"])),
        )


@dataclass(frozen=True)
class ReplicaRoutePlan:
    topology_sha256: str
    topology_receipt_sha256: str
    router_id: str
    routes: tuple[ReplicaRouteBinding, ...]

    def __post_init__(self) -> None:
        _require_sha256("route topology", self.topology_sha256)
        _require_sha256("route topology receipt", self.topology_receipt_sha256)
        _require_text("route router ID", self.router_id)
        if (
            type(self.routes) is not tuple
            or not self.routes
            or any(type(route) is not ReplicaRouteBinding for route in self.routes)
        ):
            raise TypeError("route plan requires exact route bindings")
        request_ids = tuple(route.request_id for route in self.routes)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("route plan request IDs must be unique")
        cohort_ranks: dict[str, int] = {}
        for route in self.routes:
            expected_replica = int(route.cohort_identity_sha256[:16], 16) % 2
            if route.data_parallel_rank != expected_replica or route.rank_group != (
                expected_replica,
            ):
                raise ValueError(
                    "route plan differs from deterministic two-replica routing"
                )
            prior = cohort_ranks.setdefault(
                route.cohort_identity_sha256, route.data_parallel_rank
            )
            if prior != route.data_parallel_rank:
                raise ValueError("one cohort cannot route to multiple DP replicas")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "lightcone_native_terminal_replica_route_plan",
            "protocol_sha256": NATIVE_TERMINAL_GANG_PROTOCOL_SHA256,
            "topology_sha256": self.topology_sha256,
            "topology_receipt_sha256": self.topology_receipt_sha256,
            "router_id": self.router_id,
            "routes": [route.to_dict() for route in self.routes],
        }

    @classmethod
    def from_dict(cls, value: object) -> ReplicaRoutePlan:
        row = _strict_object(
            "replica route plan",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "protocol_sha256",
                    "topology_sha256",
                    "topology_receipt_sha256",
                    "router_id",
                    "routes",
                }
            ),
        )
        if (
            row["schema_version"] != 1
            or row["kind"] != "lightcone_native_terminal_replica_route_plan"
            or row["protocol_sha256"] != NATIVE_TERMINAL_GANG_PROTOCOL_SHA256
        ):
            raise ValueError("replica route plan schema/protocol is unsupported")
        return cls(
            topology_sha256=row["topology_sha256"],
            topology_receipt_sha256=row["topology_receipt_sha256"],
            router_id=row["router_id"],
            routes=tuple(
                ReplicaRouteBinding.from_dict(item)
                for item in _strict_list("replica routes", row["routes"])
            ),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def build_replica_route_plan(
    *,
    topology: TopologyReceiptSet,
    request_cohorts: Sequence[tuple[str, CohortRouteIdentity]],
) -> ReplicaRoutePlan:
    """Build the only supported caller-independent sticky DP route plan."""

    if type(topology) is not TopologyReceiptSet:
        raise TypeError("route plan requires an exact topology receipt set")
    if topology.data_parallel_size != 2 or topology.tensor_parallel_size != 1:
        raise ValueError("release route plan is restricted to two routed TP1 replicas")
    if not request_cohorts:
        raise ValueError("route plan requires request/cohort identities")
    reference = topology.receipts[0].topology
    router = ReplicaLocalRouter(topology)
    routes: list[ReplicaRouteBinding] = []
    seen: set[str] = set()
    for request_id, identity in request_cohorts:
        _require_text("route request ID", request_id)
        if request_id in seen:
            raise ValueError("route plan request IDs must be unique")
        seen.add(request_id)
        replica = router.route(identity)
        routes.append(
            ReplicaRouteBinding(
                request_id=request_id,
                cohort_identity_sha256=identity.sha256,
                data_parallel_rank=replica,
                rank_group=topology.tensor_parallel_group(replica),
            )
        )
    return ReplicaRoutePlan(
        topology_sha256=topology.topology_sha256,
        topology_receipt_sha256=topology.receipt_sha256,
        router_id=reference.router_id,
        routes=tuple(routes),
    )


@dataclass(frozen=True)
class NativeTerminalGangBinding:
    ranks: tuple[NativeTerminalRankBinding, ...]
    route_plan: ReplicaRoutePlan | None = None

    def __post_init__(self) -> None:
        if (
            type(self.ranks) is not tuple
            or not self.ranks
            or any(type(rank) is not NativeTerminalRankBinding for rank in self.ranks)
        ):
            raise TypeError("gang binding requires exact rank bindings")
        first = self.ranks[0]
        if (
            first.world_size not in {1, 2}
            or (first.world_size == 2)
            and (first.tensor_parallel_size, first.data_parallel_size)
            not in {(2, 1), (1, 2)}
        ):
            raise ValueError("gang topology is restricted to TP1, TP2, or DP2")
        if tuple(rank.global_rank for rank in self.ranks) != tuple(
            range(first.world_size)
        ):
            raise ValueError(
                "gang ranks must be global-rank sorted exact world coverage"
            )
        common = (
            first.run.run_id,
            first.run.run_nonce_sha256,
            first.run.execution_plan_sha256,
            first.run.attempt_id,
            first.run.session_id,
            first.run.session_epoch,
            first.run.previous_run_id,
            first.run.challenge_nonce_sha256,
            first.run.method,
            first.topology_sha256,
            first.tensor_parallel_size,
            first.data_parallel_size,
            first.world_size,
            first.node_count,
            first.node_rank,
        )
        if any(
            (
                rank.run.run_id,
                rank.run.run_nonce_sha256,
                rank.run.execution_plan_sha256,
                rank.run.attempt_id,
                rank.run.session_id,
                rank.run.session_epoch,
                rank.run.previous_run_id,
                rank.run.challenge_nonce_sha256,
                rank.run.method,
                rank.topology_sha256,
                rank.tensor_parallel_size,
                rank.data_parallel_size,
                rank.world_size,
                rank.node_count,
                rank.node_rank,
            )
            != common
            for rank in self.ranks
        ):
            raise ValueError("gang ranks disagree on common run/topology identity")
        rank_configs = tuple(rank.run.rank_config_sha256 for rank in self.ranks)
        topology_receipts = tuple(rank.topology_receipt_sha256 for rank in self.ranks)
        if len(set(rank_configs)) != len(rank_configs):
            raise ValueError("gang rank-config identities must be unique")
        if len(set(topology_receipts)) != len(topology_receipts):
            raise ValueError("gang topology receipt identities must be unique")
        for replica in range(first.data_parallel_size):
            group = tuple(
                rank for rank in self.ranks if rank.data_parallel_rank == replica
            )
            if tuple(rank.tensor_parallel_rank for rank in group) != tuple(
                range(first.tensor_parallel_size)
            ):
                raise ValueError("gang TP group coverage is incomplete")
            if (
                len({rank.run.warmup_request_ids for rank in group}) != 1
                or len({rank.run.scored_request_ids for rank in group}) != 1
            ):
                raise ValueError("TP ranks must bind identical request sequences")
        if first.data_parallel_size == 1:
            if self.route_plan is not None:
                raise ValueError("single-replica gang cannot carry a DP route plan")
        else:
            if first.tensor_parallel_size != 1 or first.data_parallel_size != 2:
                raise ValueError("gang contract only supports TP2 or two routed TP1")
            if type(self.route_plan) is not ReplicaRoutePlan:
                raise TypeError("DP2 gang requires an exact replica route plan")
            if (
                self.route_plan.topology_sha256 != first.topology_sha256
                or self.route_plan.topology_receipt_sha256
                != canonical_sha256(list(topology_receipts))
            ):
                raise ValueError("gang route plan differs from rank topology")
            expected = {
                replica: tuple(
                    route.request_id
                    for route in self.route_plan.routes
                    if route.data_parallel_rank == replica
                )
                for replica in range(first.data_parallel_size)
            }
            actual = {
                replica: (
                    next(
                        rank.run.warmup_request_ids
                        for rank in self.ranks
                        if rank.data_parallel_rank == replica
                    )
                    + next(
                        rank.run.scored_request_ids
                        for rank in self.ranks
                        if rank.data_parallel_rank == replica
                    )
                )
                for replica in range(first.data_parallel_size)
            }
            if actual != expected:
                raise ValueError("DP rank request coverage differs from route plan")

    @property
    def world_size(self) -> int:
        return self.ranks[0].world_size

    @property
    def topology_mode(self) -> RuntimeTopologyMode:
        first = self.ranks[0]
        return registered_runtime_topology_mode(
            first.tensor_parallel_size,
            first.data_parallel_size,
            first.node_count,
        )

    @property
    def distributed_control_mode(self) -> DistributedControlMode:
        return distributed_control_mode(self.topology_mode)

    @property
    def adaptation_collective_mode(self) -> AdaptationCollectiveMode:
        return adaptation_collective_mode(self.topology_mode)

    @property
    def topology_sha256(self) -> str:
        return self.ranks[0].topology_sha256

    @property
    def route_plan_sha256(self) -> str | None:
        return None if self.route_plan is None else self.route_plan.sha256

    @property
    def rank_config_set_sha256(self) -> str:
        return canonical_sha256([rank.run.rank_config_sha256 for rank in self.ranks])

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "lightcone_native_terminal_gang_binding",
            "protocol_sha256": NATIVE_TERMINAL_GANG_PROTOCOL_SHA256,
            "ranks": [rank.to_dict() for rank in self.ranks],
            "route_plan": (
                None if self.route_plan is None else self.route_plan.to_dict()
            ),
            "rank_config_set_sha256": self.rank_config_set_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> NativeTerminalGangBinding:
        row = _strict_object(
            "native terminal gang binding",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "protocol_sha256",
                    "ranks",
                    "route_plan",
                    "rank_config_set_sha256",
                }
            ),
        )
        if (
            row["schema_version"] != 1
            or row["kind"] != "lightcone_native_terminal_gang_binding"
            or row["protocol_sha256"] != NATIVE_TERMINAL_GANG_PROTOCOL_SHA256
        ):
            raise ValueError("native terminal gang binding schema is unsupported")
        result = cls(
            ranks=tuple(
                NativeTerminalRankBinding.from_dict(item)
                for item in _strict_list("gang ranks", row["ranks"])
            ),
            route_plan=(
                None
                if row["route_plan"] is None
                else ReplicaRoutePlan.from_dict(row["route_plan"])
            ),
        )
        if row["rank_config_set_sha256"] != result.rank_config_set_sha256:
            raise ValueError("gang rank-config-set digest differs")
        return result

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _run_binding_to_dict(binding: NativeTerminalRunBinding) -> dict[str, object]:
    binding.validate()
    return {
        "run_id": binding.run_id,
        "run_nonce_sha256": binding.run_nonce_sha256,
        "execution_plan_sha256": binding.execution_plan_sha256,
        "rank_config_sha256": binding.rank_config_sha256,
        "attempt_id": binding.attempt_id,
        "session_id": binding.session_id,
        "session_epoch": binding.session_epoch,
        "previous_run_id": binding.previous_run_id,
        "challenge_nonce_sha256": binding.challenge_nonce_sha256,
        "method": binding.method,
        "warmup_request_ids": list(binding.warmup_request_ids),
        "scored_request_ids": list(binding.scored_request_ids),
    }


def _run_binding_from_dict(value: object) -> NativeTerminalRunBinding:
    row = _strict_object(
        "native terminal run binding",
        value,
        frozenset(
            {
                "run_id",
                "run_nonce_sha256",
                "execution_plan_sha256",
                "rank_config_sha256",
                "attempt_id",
                "session_id",
                "session_epoch",
                "previous_run_id",
                "challenge_nonce_sha256",
                "method",
                "warmup_request_ids",
                "scored_request_ids",
            }
        ),
    )
    result = NativeTerminalRunBinding(
        run_id=row["run_id"],
        run_nonce_sha256=row["run_nonce_sha256"],
        execution_plan_sha256=row["execution_plan_sha256"],
        rank_config_sha256=row["rank_config_sha256"],
        attempt_id=row["attempt_id"],
        session_id=row["session_id"],
        session_epoch=row["session_epoch"],
        previous_run_id=row["previous_run_id"],
        challenge_nonce_sha256=row["challenge_nonce_sha256"],
        method=row["method"],
        warmup_request_ids=tuple(
            _strict_list("run warmup request IDs", row["warmup_request_ids"])
        ),
        scored_request_ids=tuple(
            _strict_list("run scored request IDs", row["scored_request_ids"])
        ),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class NativeTerminalRankTransition:
    phase: Literal["begin", "reset", "finalize"]
    binding_sha256: str
    global_rank: int
    server_process_id: int
    server_process_started_ns: int
    phase_receipt_sha256: str
    phase_sha256: str
    actual_data_parallel_rank: int | None
    native_route_proof_sha256: str | None

    def __post_init__(self) -> None:
        if self.phase not in {"begin", "reset", "finalize"}:
            raise ValueError("rank transition phase is unsupported")
        _require_sha256("rank transition binding", self.binding_sha256)
        _require_nonnegative_int("rank transition global rank", self.global_rank)
        _require_positive_int("rank transition process ID", self.server_process_id)
        _require_positive_int(
            "rank transition process start", self.server_process_started_ns
        )
        _require_sha256("rank transition phase digest", self.phase_sha256)
        _require_sha256(
            "rank transition native phase receipt", self.phase_receipt_sha256
        )
        if self.actual_data_parallel_rank is None:
            if self.native_route_proof_sha256 is not None:
                raise ValueError("route proof requires an actual DP rank")
        else:
            _require_nonnegative_int(
                "actual data-parallel rank", self.actual_data_parallel_rank
            )
            _require_sha256("native route proof", self.native_route_proof_sha256)

    @property
    def expected_phase_sha256(self) -> str:
        return canonical_sha256(
            {
                "phase": self.phase,
                "binding_sha256": self.binding_sha256,
                "global_rank": self.global_rank,
                "server_process_id": self.server_process_id,
                "server_process_started_ns": self.server_process_started_ns,
                "phase_receipt_sha256": self.phase_receipt_sha256,
                "actual_data_parallel_rank": self.actual_data_parallel_rank,
                "native_route_proof_sha256": self.native_route_proof_sha256,
            }
        )

    def validate_phase_sha256(self) -> None:
        if self.phase_sha256 != self.expected_phase_sha256:
            raise ValueError("rank transition phase digest differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "binding_sha256": self.binding_sha256,
            "global_rank": self.global_rank,
            "server_process_id": self.server_process_id,
            "server_process_started_ns": self.server_process_started_ns,
            "phase_receipt_sha256": self.phase_receipt_sha256,
            "phase_sha256": self.phase_sha256,
            "actual_data_parallel_rank": self.actual_data_parallel_rank,
            "native_route_proof_sha256": self.native_route_proof_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> NativeTerminalRankTransition:
        row = _strict_object(
            "native terminal rank transition",
            value,
            frozenset(
                {
                    "phase",
                    "binding_sha256",
                    "global_rank",
                    "server_process_id",
                    "server_process_started_ns",
                    "phase_receipt_sha256",
                    "phase_sha256",
                    "actual_data_parallel_rank",
                    "native_route_proof_sha256",
                }
            ),
        )
        return cls(**row)


@dataclass(frozen=True)
class NativeTerminalGangTransition:
    phase: Literal["begin", "reset", "finalize"]
    gang_binding_sha256: str
    ranks: tuple[NativeTerminalRankTransition, ...]

    def __post_init__(self) -> None:
        if self.phase not in {"begin", "reset", "finalize"}:
            raise ValueError("gang transition phase is unsupported")
        _require_sha256("gang transition binding", self.gang_binding_sha256)
        if (
            type(self.ranks) is not tuple
            or not self.ranks
            or any(
                type(rank) is not NativeTerminalRankTransition for rank in self.ranks
            )
        ):
            raise TypeError("gang transition requires exact rank transitions")
        if tuple(rank.global_rank for rank in self.ranks) != tuple(
            range(len(self.ranks))
        ):
            raise ValueError("gang transition ranks must be sorted exact coverage")
        for rank in self.ranks:
            rank.validate_phase_sha256()

    def validate(self, binding: NativeTerminalGangBinding) -> None:
        if self.phase not in {"begin", "reset", "finalize"}:
            raise ValueError("gang transition phase is unsupported")
        if self.gang_binding_sha256 != binding.sha256:
            raise ValueError("gang transition belongs to another binding")
        if tuple(rank.global_rank for rank in self.ranks) != tuple(
            range(binding.world_size)
        ):
            raise ValueError("gang transition lacks exact all-rank coverage")
        processes = tuple(
            (rank.server_process_id, rank.server_process_started_ns)
            for rank in self.ranks
        )
        if len(set(processes)) != len(processes):
            raise ValueError("gang transition process identities must be unique")
        for transition, expected in zip(self.ranks, binding.ranks, strict=True):
            transition.validate_phase_sha256()
            if (
                transition.phase != self.phase
                or transition.binding_sha256 != expected.sha256
            ):
                raise ValueError("gang transition rank binding/phase differs")
            if binding.route_plan is None:
                if transition.actual_data_parallel_rank is not None:
                    raise ValueError("non-DP transition cannot claim route proof")
            elif self.phase == "finalize":
                if (
                    transition.actual_data_parallel_rank != expected.data_parallel_rank
                    or transition.native_route_proof_sha256
                    != _expected_native_route_proof_sha256(binding, expected)
                ):
                    raise ValueError(
                        "DP transition lacks exact native actual-route proof"
                    )
            elif (
                transition.actual_data_parallel_rank is not None
                or transition.native_route_proof_sha256 is not None
            ):
                raise ValueError("DP route proof is only valid at finalization")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "lightcone_native_terminal_gang_transition",
            "phase": self.phase,
            "gang_binding_sha256": self.gang_binding_sha256,
            "ranks": [rank.to_dict() for rank in self.ranks],
        }

    @classmethod
    def from_dict(cls, value: object) -> NativeTerminalGangTransition:
        row = _strict_object(
            "native terminal gang transition",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "phase",
                    "gang_binding_sha256",
                    "ranks",
                }
            ),
        )
        if (
            row["schema_version"] != 1
            or row["kind"] != "lightcone_native_terminal_gang_transition"
        ):
            raise ValueError("native terminal gang transition schema is unsupported")
        return cls(
            phase=row["phase"],
            gang_binding_sha256=row["gang_binding_sha256"],
            ranks=tuple(
                NativeTerminalRankTransition.from_dict(item)
                for item in _strict_list("gang transition ranks", row["ranks"])
            ),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _expected_native_route_proof_sha256(
    binding: NativeTerminalGangBinding,
    rank: NativeTerminalRankBinding,
) -> str:
    if binding.route_plan is None:
        raise ValueError("native route proof requires a route plan")
    request_ids = rank.run.warmup_request_ids + rank.run.scored_request_ids
    return canonical_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_native_actual_replica_route_proof",
            "gang_binding_sha256": binding.sha256,
            "route_plan_sha256": binding.route_plan.sha256,
            "global_rank": rank.global_rank,
            "data_parallel_rank": rank.data_parallel_rank,
            "request_ids": list(request_ids),
        }
    )


def build_diagnostic_native_terminal_gang_transition(
    *,
    binding: NativeTerminalGangBinding,
    phase: Literal["begin", "reset", "finalize"],
    process_identities: Sequence[tuple[int, int]],
    phase_receipt_sha256s: Sequence[str],
) -> NativeTerminalGangTransition:
    """Build fake/CPU transitions; this is never formal native evidence."""

    if (
        len(process_identities) != binding.world_size
        or len(phase_receipt_sha256s) != binding.world_size
    ):
        raise ValueError("diagnostic transition lacks exact world coverage")
    rows: list[NativeTerminalRankTransition] = []
    for rank_binding, process, phase_receipt in zip(
        binding.ranks,
        process_identities,
        phase_receipt_sha256s,
        strict=True,
    ):
        _require_sha256("diagnostic phase receipt", phase_receipt)
        route_proof = (
            _expected_native_route_proof_sha256(binding, rank_binding)
            if binding.route_plan is not None and phase == "finalize"
            else None
        )
        seed = NativeTerminalRankTransition(
            phase=phase,
            binding_sha256=rank_binding.sha256,
            global_rank=rank_binding.global_rank,
            server_process_id=process[0],
            server_process_started_ns=process[1],
            phase_receipt_sha256=phase_receipt,
            phase_sha256="0" * 64,
            actual_data_parallel_rank=(
                rank_binding.data_parallel_rank
                if binding.route_plan is not None and phase == "finalize"
                else None
            ),
            native_route_proof_sha256=route_proof,
        )
        rows.append(
            NativeTerminalRankTransition(
                phase=seed.phase,
                binding_sha256=seed.binding_sha256,
                global_rank=seed.global_rank,
                server_process_id=seed.server_process_id,
                server_process_started_ns=seed.server_process_started_ns,
                phase_receipt_sha256=seed.phase_receipt_sha256,
                phase_sha256=seed.expected_phase_sha256,
                actual_data_parallel_rank=seed.actual_data_parallel_rank,
                native_route_proof_sha256=seed.native_route_proof_sha256,
            )
        )
    result = NativeTerminalGangTransition(
        phase=phase,
        gang_binding_sha256=binding.sha256,
        ranks=tuple(rows),
    )
    result.validate(binding)
    return result


class NativeTerminalGangTransport(Protocol):
    """Patch-native transport returning collective-produced all-rank receipts."""

    async def capability(self, binding: NativeTerminalGangBinding) -> object: ...

    async def begin_all(
        self, binding: NativeTerminalGangBinding
    ) -> NativeTerminalGangTransition: ...

    async def reset_all(
        self,
        binding: NativeTerminalGangBinding,
        warmup_requests: Mapping[int, Sequence[TerminalRequestExpectation]],
    ) -> NativeTerminalGangTransition: ...

    async def finalize_all(
        self,
        binding: NativeTerminalGangBinding,
        scored_requests: Mapping[int, Sequence[TerminalRequestExpectation]],
    ) -> tuple[
        NativeTerminalGangTransition, tuple[ValidatedNativeTerminalEvidence, ...]
    ]: ...


class NativeTerminalGangProvider:
    """Bounded all-rank lifecycle that permanently poisons on partial failure."""

    def __init__(
        self,
        transport: NativeTerminalGangTransport,
        *,
        timeout_s: float,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("gang terminal timeout must be finite and positive")
        self._transport = transport
        self._timeout_s = timeout_s
        self._binding: NativeTerminalGangBinding | None = None
        self._phase = "IDLE"
        self._last_finalized_run_id: str | None = None
        self._next_session_epoch = 1
        self._session_id: str | None = None
        self._seen_runs: set[str] = set()
        self._seen_attempts: set[str] = set()
        self._processes: tuple[tuple[int, int], ...] | None = None
        self._transitions: list[NativeTerminalGangTransition] = []

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def transitions(self) -> tuple[NativeTerminalGangTransition, ...]:
        return tuple(self._transitions)

    @staticmethod
    def _transition_processes(
        transition: NativeTerminalGangTransition,
    ) -> tuple[tuple[int, int], ...]:
        return tuple(
            (rank.server_process_id, rank.server_process_started_ns)
            for rank in transition.ranks
        )

    async def _bounded(self, call: Awaitable[_T]) -> _T:
        try:
            return await asyncio.wait_for(call, timeout=self._timeout_s)
        except BaseException:
            self._phase = "FAILED"
            raise

    async def begin(
        self, binding: NativeTerminalGangBinding
    ) -> NativeTerminalGangTransition:
        try:
            if self._phase not in {"IDLE", "FINALIZED"}:
                raise RuntimeError("gang terminal begin is illegal in this phase")
            run = binding.ranks[0].run
            if (
                run.session_epoch != self._next_session_epoch
                or run.previous_run_id != self._last_finalized_run_id
                or (self._session_id is not None and run.session_id != self._session_id)
                or run.run_id in self._seen_runs
                or run.attempt_id in self._seen_attempts
            ):
                raise ValueError("gang terminal session/run lineage is stale or reused")
            capability = await self._bounded(self._transport.capability(binding))
            _validate_diagnostic_capability(capability, binding=binding)
            result = await self._bounded(self._transport.begin_all(binding))
            result.validate(binding)
            if result.phase != "begin":
                raise ValueError("gang begin transport returned another phase")
            self._binding = binding
            self._processes = self._transition_processes(result)
            self._transitions = [result]
            self._session_id = run.session_id
            self._seen_runs.add(run.run_id)
            self._seen_attempts.add(run.attempt_id)
            self._phase = "WARMUP"
            return result
        except BaseException:
            self._phase = "FAILED"
            raise

    async def reset(
        self,
        *,
        warmup_requests: Mapping[int, Sequence[TerminalRequestExpectation]],
    ) -> NativeTerminalGangTransition:
        try:
            binding = self._require_phase("WARMUP")
            _validate_rank_request_mapping(binding, warmup_requests, warmup=True)
            result = await self._bounded(
                self._transport.reset_all(binding, warmup_requests)
            )
            result.validate(binding)
            if result.phase != "reset":
                raise ValueError("gang reset transport returned another phase")
            if self._transition_processes(result) != self._processes:
                raise ValueError("gang reset process identities changed")
            self._transitions.append(result)
            self._phase = "SCORED"
            return result
        except BaseException:
            self._phase = "FAILED"
            raise

    async def finalize(
        self,
        *,
        scored_requests: Mapping[int, Sequence[TerminalRequestExpectation]],
    ) -> tuple[
        NativeTerminalGangTransition, tuple[ValidatedNativeTerminalEvidence, ...]
    ]:
        try:
            binding = self._require_phase("SCORED")
            _validate_rank_request_mapping(binding, scored_requests, warmup=False)
            transition, evidence = await self._bounded(
                self._transport.finalize_all(binding, scored_requests)
            )
            transition.validate(binding)
            if transition.phase != "finalize":
                raise ValueError("gang finalize transport returned another phase")
            if self._transition_processes(transition) != self._processes:
                raise ValueError("gang finalize process identities changed")
            if len(evidence) != binding.world_size:
                raise ValueError("gang finalize lacks exact all-rank evidence")
            for rank, item in enumerate(evidence):
                if (
                    type(item) is not ValidatedNativeTerminalEvidence
                    or item.binding != binding.ranks[rank].run
                    or item.terminal_sha256
                    != transition.ranks[rank].phase_receipt_sha256
                    or item.begin_receipt.server_process_id
                    != transition.ranks[rank].server_process_id
                    or item.begin_receipt.server_process_started_ns
                    != transition.ranks[rank].server_process_started_ns
                ):
                    raise ValueError(
                        "gang finalize evidence rank/binding/process differs"
                    )
            self._transitions.append(transition)
            self._last_finalized_run_id = binding.ranks[0].run.run_id
            self._next_session_epoch += 1
            self._phase = "FINALIZED"
            return transition, evidence
        except BaseException:
            self._phase = "FAILED"
            raise

    def _require_phase(self, expected: str) -> NativeTerminalGangBinding:
        if self._phase != expected or self._binding is None:
            raise RuntimeError(f"gang terminal action requires phase {expected}")
        return self._binding


class SingleRankNativeTerminalGangTransport:
    """World-one adapter over the unchanged native terminal v1 provider."""

    def __init__(self, provider: NativeTerminalProvider) -> None:
        if type(provider) is not NativeTerminalProvider:
            raise TypeError("world-one adapter requires NativeTerminalProvider")
        self._provider = provider

    @staticmethod
    def _rank(binding: NativeTerminalGangBinding) -> NativeTerminalRankBinding:
        if binding.world_size != 1 or binding.route_plan is not None:
            raise ValueError("legacy terminal adapter is restricted to world size one")
        return binding.ranks[0]

    async def capability(self, binding: NativeTerminalGangBinding) -> object:
        rank = self._rank(binding)
        capability = await self._provider.capability(expected_method=rank.run.method)
        if (
            not capability.enabled
            or not capability.method_evidence_supported
            or not capability.topology_supported
        ):
            raise ValueError("legacy terminal capability is not ready")
        return {
            "schema_version": 1,
            "hook": NATIVE_TERMINAL_GANG_HOOK,
            "protocol_sha256": NATIVE_TERMINAL_GANG_PROTOCOL_SHA256,
            "topology_sha256": binding.topology_sha256,
            "world_size": 1,
            "native_all_rank_receipts": True,
            "native_actual_route_proof": False,
        }

    async def begin_all(
        self, binding: NativeTerminalGangBinding
    ) -> NativeTerminalGangTransition:
        rank = self._rank(binding)
        receipt = await self._provider.begin(rank.run)
        return build_diagnostic_native_terminal_gang_transition(
            binding=binding,
            phase="begin",
            process_identities=(
                (receipt.server_process_id, receipt.server_process_started_ns),
            ),
            phase_receipt_sha256s=(receipt.begin_sha256,),
        )

    async def reset_all(
        self,
        binding: NativeTerminalGangBinding,
        warmup_requests: Mapping[int, Sequence[TerminalRequestExpectation]],
    ) -> NativeTerminalGangTransition:
        self._rank(binding)
        receipt = await self._provider.reset(warmup_requests=warmup_requests[0])
        return build_diagnostic_native_terminal_gang_transition(
            binding=binding,
            phase="reset",
            process_identities=(
                (receipt.server_process_id, receipt.server_process_started_ns),
            ),
            phase_receipt_sha256s=(receipt.reset_sha256,),
        )

    async def finalize_all(
        self,
        binding: NativeTerminalGangBinding,
        scored_requests: Mapping[int, Sequence[TerminalRequestExpectation]],
    ) -> tuple[
        NativeTerminalGangTransition,
        tuple[ValidatedNativeTerminalEvidence, ...],
    ]:
        self._rank(binding)
        evidence = await self._provider.finalize(requests=scored_requests[0])
        transition = build_diagnostic_native_terminal_gang_transition(
            binding=binding,
            phase="finalize",
            process_identities=(
                (
                    evidence.begin_receipt.server_process_id,
                    evidence.begin_receipt.server_process_started_ns,
                ),
            ),
            phase_receipt_sha256s=(evidence.terminal_sha256,),
        )
        return transition, (evidence,)


def _validate_diagnostic_capability(
    value: object, *, binding: NativeTerminalGangBinding
) -> None:
    row = _strict_object(
        "diagnostic gang capability",
        value,
        frozenset(
            {
                "schema_version",
                "hook",
                "protocol_sha256",
                "topology_sha256",
                "world_size",
                "native_all_rank_receipts",
                "native_actual_route_proof",
            }
        ),
    )
    if (
        row["schema_version"] != 1
        or row["hook"] != NATIVE_TERMINAL_GANG_HOOK
        or row["protocol_sha256"] != NATIVE_TERMINAL_GANG_PROTOCOL_SHA256
        or row["topology_sha256"] != binding.topology_sha256
        or row["world_size"] != binding.world_size
        or row["native_all_rank_receipts"] is not True
    ):
        raise ValueError("diagnostic gang capability differs from the binding")
    route_proof = row["native_actual_route_proof"]
    if binding.route_plan is None:
        if type(route_proof) is not bool:
            raise TypeError("diagnostic route capability must be boolean")
    elif route_proof is not True:
        raise NativeTerminalGangAuthorityBlocked(
            "native_terminal_dp_actual_route_producer_unavailable",
            "DP2 requires first-party scheduler evidence of the actual routed replica",
        )


def _validate_rank_request_mapping(
    binding: NativeTerminalGangBinding,
    value: Mapping[int, Sequence[TerminalRequestExpectation]],
    *,
    warmup: bool,
) -> None:
    if type(value) is not dict or set(value) != set(range(binding.world_size)):
        raise ValueError("rank request map must cover every rank exactly")
    for rank_binding in binding.ranks:
        requests = tuple(value[rank_binding.global_rank])
        expected = (
            rank_binding.run.warmup_request_ids
            if warmup
            else rank_binding.run.scored_request_ids
        )
        if tuple(request.request_id for request in requests) != expected:
            raise ValueError("rank request map differs from bound request order")
        for request in requests:
            if type(request) is not TerminalRequestExpectation:
                raise TypeError("rank request map requires exact terminal expectations")
            request.validate()


@dataclass(frozen=True)
class NativeTerminalRankArtifactBinding:
    global_rank: int
    tensor_parallel_rank: int
    data_parallel_rank: int
    rank_binding_sha256: str
    rank_config_sha256: str
    server_process_id: int
    server_process_started_ns: int
    path: str
    sidecar_path: str
    size: int
    raw_sha256: str
    sidecar_file_sha256: str
    terminal_sha256: str
    trusted_attester_policy_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("rank artifact global rank", self.global_rank),
            ("rank artifact TP rank", self.tensor_parallel_rank),
            ("rank artifact DP rank", self.data_parallel_rank),
        ):
            _require_nonnegative_int(label, value)
        for label, value in (
            ("rank artifact binding", self.rank_binding_sha256),
            ("rank artifact config", self.rank_config_sha256),
            ("rank artifact raw", self.raw_sha256),
            ("rank artifact sidecar file", self.sidecar_file_sha256),
            ("rank artifact terminal", self.terminal_sha256),
            ("rank artifact trust policy", self.trusted_attester_policy_sha256),
        ):
            _require_sha256(label, value)
        _require_positive_int("rank artifact process ID", self.server_process_id)
        _require_positive_int(
            "rank artifact process start", self.server_process_started_ns
        )
        _require_positive_int("rank artifact size", self.size)
        path = _absolute_resolved(self.path, label="rank artifact path")
        sidecar = _absolute_resolved(
            self.sidecar_path, label="rank artifact sidecar path"
        )
        if sidecar != Path(f"{path}.sha256"):
            raise ValueError("rank artifact sidecar path is not exact")

    def to_dict(self) -> dict[str, object]:
        return {
            "global_rank": self.global_rank,
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "data_parallel_rank": self.data_parallel_rank,
            "rank_binding_sha256": self.rank_binding_sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "server_process_id": self.server_process_id,
            "server_process_started_ns": self.server_process_started_ns,
            "path": self.path,
            "sidecar_path": self.sidecar_path,
            "size": self.size,
            "raw_sha256": self.raw_sha256,
            "sidecar_file_sha256": self.sidecar_file_sha256,
            "terminal_sha256": self.terminal_sha256,
            "trusted_attester_policy_sha256": self.trusted_attester_policy_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> NativeTerminalRankArtifactBinding:
        row = _strict_object(
            "native terminal rank artifact binding",
            value,
            frozenset(
                {
                    "global_rank",
                    "tensor_parallel_rank",
                    "data_parallel_rank",
                    "rank_binding_sha256",
                    "rank_config_sha256",
                    "server_process_id",
                    "server_process_started_ns",
                    "path",
                    "sidecar_path",
                    "size",
                    "raw_sha256",
                    "sidecar_file_sha256",
                    "terminal_sha256",
                    "trusted_attester_policy_sha256",
                }
            ),
        )
        return cls(**row)


@dataclass(frozen=True)
class NativeTerminalAggregateReceipt:
    gang_binding: NativeTerminalGangBinding
    transitions: tuple[NativeTerminalGangTransition, ...]
    rank_artifacts: tuple[NativeTerminalRankArtifactBinding, ...]
    warmup_coverage_sha256: str
    scored_coverage_sha256: str

    def __post_init__(self) -> None:
        if type(self.gang_binding) is not NativeTerminalGangBinding:
            raise TypeError("aggregate requires an exact gang binding")
        if type(self.transitions) is not tuple or tuple(
            transition.phase for transition in self.transitions
        ) != ("begin", "reset", "finalize"):
            raise ValueError("aggregate requires ordered begin/reset/final transitions")
        for transition in self.transitions:
            transition.validate(self.gang_binding)
        process_sets = tuple(
            tuple(
                (rank.server_process_id, rank.server_process_started_ns)
                for rank in transition.ranks
            )
            for transition in self.transitions
        )
        if len(set(process_sets)) != 1:
            raise ValueError("aggregate rank process identities changed across phases")
        if type(self.rank_artifacts) is not tuple or tuple(
            row.global_rank for row in self.rank_artifacts
        ) != tuple(range(self.gang_binding.world_size)):
            raise ValueError(
                "aggregate rank artifacts lack exact sorted world coverage"
            )
        for row, binding in zip(
            self.rank_artifacts, self.gang_binding.ranks, strict=True
        ):
            if (
                row.tensor_parallel_rank != binding.tensor_parallel_rank
                or row.data_parallel_rank != binding.data_parallel_rank
                or row.rank_binding_sha256 != binding.sha256
                or row.rank_config_sha256 != binding.run.rank_config_sha256
                or row.server_process_id
                != self.transitions[-1].ranks[row.global_rank].server_process_id
                or row.server_process_started_ns
                != self.transitions[-1].ranks[row.global_rank].server_process_started_ns
            ):
                raise ValueError("aggregate rank artifact differs from rank binding")
        _require_sha256("aggregate warmup coverage", self.warmup_coverage_sha256)
        _require_sha256("aggregate scored coverage", self.scored_coverage_sha256)
        if self.warmup_coverage_sha256 != _coverage_sha256(
            self.gang_binding, warmup=True
        ) or self.scored_coverage_sha256 != _coverage_sha256(
            self.gang_binding, warmup=False
        ):
            raise ValueError("aggregate request coverage digests differ")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "lightcone_native_terminal_gang_aggregate",
            "hook": NATIVE_TERMINAL_GANG_HOOK,
            "protocol_sha256": NATIVE_TERMINAL_GANG_PROTOCOL_SHA256,
            "gang_binding": self.gang_binding.to_dict(),
            "transitions": [transition.to_dict() for transition in self.transitions],
            "rank_artifacts": [row.to_dict() for row in self.rank_artifacts],
            "warmup_coverage_sha256": self.warmup_coverage_sha256,
            "scored_coverage_sha256": self.scored_coverage_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> NativeTerminalAggregateReceipt:
        row = _strict_object(
            "native terminal gang aggregate",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "hook",
                    "protocol_sha256",
                    "gang_binding",
                    "transitions",
                    "rank_artifacts",
                    "warmup_coverage_sha256",
                    "scored_coverage_sha256",
                }
            ),
        )
        if (
            row["schema_version"] != 1
            or row["kind"] != "lightcone_native_terminal_gang_aggregate"
            or row["hook"] != NATIVE_TERMINAL_GANG_HOOK
            or row["protocol_sha256"] != NATIVE_TERMINAL_GANG_PROTOCOL_SHA256
        ):
            raise ValueError("native terminal gang aggregate schema is unsupported")
        return cls(
            gang_binding=NativeTerminalGangBinding.from_dict(row["gang_binding"]),
            transitions=tuple(
                NativeTerminalGangTransition.from_dict(item)
                for item in _strict_list("aggregate transitions", row["transitions"])
            ),
            rank_artifacts=tuple(
                NativeTerminalRankArtifactBinding.from_dict(item)
                for item in _strict_list(
                    "aggregate rank artifacts", row["rank_artifacts"]
                )
            ),
            warmup_coverage_sha256=row["warmup_coverage_sha256"],
            scored_coverage_sha256=row["scored_coverage_sha256"],
        )


def _coverage_sha256(binding: NativeTerminalGangBinding, *, warmup: bool) -> str:
    return canonical_sha256(
        [
            {
                "global_rank": rank.global_rank,
                "request_ids": list(
                    rank.run.warmup_request_ids
                    if warmup
                    else rank.run.scored_request_ids
                ),
            }
            for rank in binding.ranks
        ]
    )


def publish_native_terminal_gang_aggregate(
    *,
    output_path: str | Path,
    binding: NativeTerminalGangBinding,
    transitions: Sequence[NativeTerminalGangTransition],
    rank_artifacts: Sequence[ValidatedNativeTerminalEvidence],
    warmup_requests: Mapping[int, Sequence[TerminalRequestExpectation]],
) -> NativeTerminalAggregateReceipt:
    """Publish rank artifacts first and the claimable aggregate last."""

    output = _absolute_resolved(output_path, label="gang aggregate output")
    all_transitions = tuple(transitions)
    if tuple(transition.phase for transition in all_transitions) != (
        "begin",
        "reset",
        "finalize",
    ):
        raise ValueError("aggregate publication requires begin/reset/final transitions")
    for transition in all_transitions:
        transition.validate(binding)
    process_sets = tuple(
        tuple(
            (rank.server_process_id, rank.server_process_started_ns)
            for rank in transition.ranks
        )
        for transition in all_transitions
    )
    if len(set(process_sets)) != 1:
        raise ValueError("rank process identities changed across terminal phases")
    final_transition = all_transitions[-1]
    if len(rank_artifacts) != binding.world_size:
        raise ValueError("rank artifact publication lacks exact world coverage")
    _validate_rank_request_mapping(binding, warmup_requests, warmup=True)
    if output.exists() or Path(f"{output}.sha256").exists():
        raise FileExistsError("gang aggregate output already exists")
    rank_paths = tuple(
        output.with_name(f"{output.name}.rank{rank}.json")
        for rank in range(binding.world_size)
    )
    if any(path.exists() or Path(f"{path}.sha256").exists() for path in rank_paths):
        raise FileExistsError("gang rank artifact output already exists")
    for rank, evidence in enumerate(rank_artifacts):
        transition = final_transition.ranks[rank]
        if (
            type(evidence) is not ValidatedNativeTerminalEvidence
            or evidence.binding != binding.ranks[rank].run
            or evidence.terminal_sha256 != transition.phase_receipt_sha256
            or evidence.begin_receipt.server_process_id != transition.server_process_id
            or evidence.begin_receipt.server_process_started_ns
            != transition.server_process_started_ns
        ):
            raise ValueError("rank evidence differs from gang binding/transition")
    rows: list[NativeTerminalRankArtifactBinding] = []
    for rank, evidence in enumerate(rank_artifacts):
        expected = binding.ranks[rank]
        artifact = evidence.to_artifact(warmup_requests=warmup_requests[rank], rank=0)
        # v1 is internally rank-0-only.  The enclosing strict binding supplies
        # global rank identity until the native gang producer ships its own v2.
        body = canonical_json_bytes(artifact)
        rank_path = output.with_name(f"{output.name}.rank{rank}.json")
        sidecar_path = Path(f"{rank_path}.sha256")
        if len(body) > 1_500_000:
            publish_scalable_native_terminal_artifact(
                output_path=rank_path,
                legacy_artifact=artifact,
            )
            body = _read_regular_file(
                rank_path, label="sharded native terminal rank artifact"
            )
        else:
            _publish_exclusive(
                rank_path,
                body,
                label="native terminal rank artifact",
            )
        digest = hashlib.sha256(body).hexdigest()
        sidecar_body = f"{digest}\n".encode("ascii")
        _publish_exclusive(
            sidecar_path,
            sidecar_body,
            label="native terminal rank artifact sidecar",
        )
        rows.append(
            NativeTerminalRankArtifactBinding(
                global_rank=rank,
                tensor_parallel_rank=expected.tensor_parallel_rank,
                data_parallel_rank=expected.data_parallel_rank,
                rank_binding_sha256=expected.sha256,
                rank_config_sha256=expected.run.rank_config_sha256,
                server_process_id=(final_transition.ranks[rank].server_process_id),
                server_process_started_ns=(
                    final_transition.ranks[rank].server_process_started_ns
                ),
                path=str(rank_path),
                sidecar_path=str(sidecar_path),
                size=len(body),
                raw_sha256=digest,
                sidecar_file_sha256=hashlib.sha256(sidecar_body).hexdigest(),
                terminal_sha256=evidence.terminal_sha256,
                trusted_attester_policy_sha256=(
                    evidence.trusted_attester_policy_sha256
                ),
            )
        )
    _fsync_directory(output.parent, label="native terminal rank publication")
    receipt = NativeTerminalAggregateReceipt(
        gang_binding=binding,
        transitions=all_transitions,
        rank_artifacts=tuple(rows),
        warmup_coverage_sha256=_coverage_sha256(binding, warmup=True),
        scored_coverage_sha256=_coverage_sha256(binding, warmup=False),
    )
    body = canonical_json_bytes(receipt.to_dict())
    sidecar_body = f"{receipt.sha256}\n".encode("ascii")
    _publish_exclusive(output, body, label="native terminal gang aggregate")
    _publish_exclusive(
        Path(f"{output}.sha256"),
        sidecar_body,
        label="native terminal gang aggregate sidecar",
    )
    _fsync_directory(output.parent, label="native terminal aggregate publication")
    return receipt


def reopen_native_terminal_gang_aggregate_diagnostic(
    *,
    aggregate_path: str | Path,
    expected_aggregate_sha256: str,
    expected_binding: NativeTerminalGangBinding,
    trusted_attester_policy: TrustedAttesterPolicy,
    expected_warmup_requests: Mapping[int, Sequence[TerminalRequestExpectation]],
    expected_scored_requests: Mapping[int, Sequence[TerminalRequestExpectation]],
) -> NativeTerminalAggregateReceipt:
    """Reopen the complete aggregate without granting formal release authority."""

    if type(trusted_attester_policy) is not TrustedAttesterPolicy:
        raise TypeError("gang aggregate requires an exact trusted-attester policy")
    expected_aggregate = _require_sha256(
        "expected gang aggregate", expected_aggregate_sha256
    )
    _validate_rank_request_mapping(
        expected_binding, expected_warmup_requests, warmup=True
    )
    _validate_rank_request_mapping(
        expected_binding, expected_scored_requests, warmup=False
    )
    path = _absolute_resolved(aggregate_path, label="gang aggregate path")
    body = _read_regular_file(path, label="native terminal gang aggregate")
    sidecar = _read_regular_file(
        Path(f"{path}.sha256"), label="native terminal gang aggregate sidecar"
    )
    value = _strict_json(body, label="native terminal gang aggregate")
    receipt = NativeTerminalAggregateReceipt.from_dict(value)
    if (
        receipt.gang_binding != expected_binding
        or receipt.sha256 != expected_aggregate
        or sidecar != f"{receipt.sha256}\n".encode("ascii")
        or body != canonical_json_bytes(value)
    ):
        raise RuntimeError("native terminal gang aggregate identity changed")
    for row, rank_binding in zip(
        receipt.rank_artifacts, expected_binding.ranks, strict=True
    ):
        rank_body = _read_regular_file(
            Path(row.path), label="bound native terminal rank artifact"
        )
        rank_sidecar = _read_regular_file(
            Path(row.sidecar_path), label="bound native terminal rank artifact sidecar"
        )
        if (
            len(rank_body) != row.size
            or hashlib.sha256(rank_body).hexdigest() != row.raw_sha256
            or hashlib.sha256(rank_sidecar).hexdigest() != row.sidecar_file_sha256
            or rank_sidecar != f"{row.raw_sha256}\n".encode("ascii")
        ):
            raise RuntimeError("bound native terminal rank artifact changed")
        rank_value = _strict_json(rank_body, label="native terminal rank artifact")
        is_sharded = (
            type(rank_value) is dict
            and rank_value.get("schema_version") == 2
            and rank_value.get("artifact_kind") == SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND
        )
        if is_sharded:
            canonical_binding = CanonicalJsonProofBinding.bind(row.path)
            if canonical_binding.raw_sha256 != row.raw_sha256:
                raise RuntimeError("sharded native terminal rank binding changed")
            rank_value = reopen_scalable_native_terminal_artifact(rank_value)
        elif rank_body != canonical_json_bytes(rank_value):
            raise RuntimeError("native terminal rank artifact is not canonical JSON")
        evidence = validate_native_terminal_artifact(
            rank_value,
            trusted_attester_policy=trusted_attester_policy,
            expected_binding=rank_binding.run,
            expected_warmup_requests=expected_warmup_requests[row.global_rank],
            expected_scored_requests=expected_scored_requests[row.global_rank],
        )
        transition_receipts = tuple(
            transition.ranks[row.global_rank].phase_receipt_sha256
            for transition in receipt.transitions
        )
        if (
            evidence.terminal_sha256 != row.terminal_sha256
            or transition_receipts
            != (
                evidence.begin_receipt.begin_sha256,
                evidence.reset_receipt.reset_sha256,
                evidence.terminal_sha256,
            )
            or evidence.begin_receipt.server_process_id != row.server_process_id
            or evidence.begin_receipt.server_process_started_ns
            != row.server_process_started_ns
            or evidence.trusted_attester_policy_sha256
            != row.trusted_attester_policy_sha256
        ):
            raise RuntimeError("rank artifact terminal/process authority changed")
    return receipt


def require_native_terminal_gang_aggregate(
    *,
    aggregate_path: str | Path,
    expected_aggregate_sha256: str,
    expected_binding: NativeTerminalGangBinding,
    claimed_capability_sha256: str,
    verified_gpu_proof: VerifiedDistributedRuntimeGpuProof | None = None,
    trusted_attester_policy: TrustedAttesterPolicy,
    expected_warmup_requests: Mapping[int, Sequence[TerminalRequestExpectation]],
    expected_scored_requests: Mapping[int, Sequence[TerminalRequestExpectation]],
) -> NativeTerminalAggregateReceipt:
    """Formal entry: source-owned capability check precedes every path read."""

    if expected_binding.world_size > 1:
        if type(verified_gpu_proof) is not VerifiedDistributedRuntimeGpuProof:
            raise NativeTerminalGangAuthorityBlocked(
                "distributed_runtime_gpu_proof_unavailable",
                "formal gang evidence lacks a verified root-signed GPU proof",
            )
        if (
            verified_gpu_proof.topology_mode != expected_binding.topology_mode
            or verified_gpu_proof.topology_sha256 != expected_binding.topology_sha256
        ):
            raise ValueError("distributed GPU proof belongs to another topology")
    require_native_terminal_gang_release_capability(
        topology_sha256=expected_binding.topology_sha256,
        claimed_capability_sha256=claimed_capability_sha256,
    )
    if expected_binding.world_size > 1:
        return reopen_native_terminal_gang_aggregate_diagnostic(
            aggregate_path=aggregate_path,
            expected_aggregate_sha256=expected_aggregate_sha256,
            expected_binding=expected_binding,
            trusted_attester_policy=trusted_attester_policy,
            expected_warmup_requests=expected_warmup_requests,
            expected_scored_requests=expected_scored_requests,
        )
    raise NativeTerminalGangAuthorityBlocked(
        "native_terminal_gang_world1_uses_legacy_authority",
        "single-rank formal execution must retain the existing v1 authority",
    )


def publish_single_rank_legacy_aggregate(
    *,
    output_path: str | Path,
    rank_binding: NativeTerminalRankBinding,
    evidence: ValidatedNativeTerminalEvidence,
    warmup_requests: Sequence[TerminalRequestExpectation],
) -> NativeTerminalAggregateReceipt:
    """World-one adapter; it changes no v1 artifact wire semantics."""

    if rank_binding.world_size != 1:
        raise ValueError("legacy aggregate adapter is restricted to world size one")
    binding = NativeTerminalGangBinding(ranks=(rank_binding,))
    transitions = _legacy_transitions(binding, evidence)
    return publish_native_terminal_gang_aggregate(
        output_path=output_path,
        binding=binding,
        transitions=transitions,
        rank_artifacts=(evidence,),
        warmup_requests={0: tuple(warmup_requests)},
    )


def _legacy_transitions(
    binding: NativeTerminalGangBinding,
    evidence: ValidatedNativeTerminalEvidence,
) -> tuple[NativeTerminalGangTransition, ...]:
    process = (
        evidence.begin_receipt.server_process_id,
        evidence.begin_receipt.server_process_started_ns,
    )
    return tuple(
        build_diagnostic_native_terminal_gang_transition(
            binding=binding,
            phase=phase,
            process_identities=(process,),
            phase_receipt_sha256s=(digest,),
        )
        for phase, digest in (
            ("begin", evidence.begin_receipt.begin_sha256),
            ("reset", evidence.reset_receipt.reset_sha256),
            ("finalize", evidence.terminal_sha256),
        )
    )


__all__ = [
    "NATIVE_TERMINAL_GANG_HOOK",
    "NATIVE_TERMINAL_GANG_PROTOCOL_SHA256",
    "NATIVE_TERMINAL_GANG_RELEASE_CAPABILITY_SHA256",
    "NativeTerminalAggregateReceipt",
    "NativeTerminalGangAuthorityBlocked",
    "NativeTerminalGangBinding",
    "NativeTerminalGangProvider",
    "NativeTerminalGangTransition",
    "NativeTerminalGangTransport",
    "NativeTerminalRankArtifactBinding",
    "NativeTerminalRankBinding",
    "NativeTerminalRankTransition",
    "ReplicaRouteBinding",
    "ReplicaRoutePlan",
    "SingleRankNativeTerminalGangTransport",
    "build_diagnostic_native_terminal_gang_transition",
    "build_replica_route_plan",
    "publish_native_terminal_gang_aggregate",
    "publish_single_rank_legacy_aggregate",
    "reopen_native_terminal_gang_aggregate_diagnostic",
    "require_native_terminal_gang_aggregate",
    "require_native_terminal_gang_release_capability",
]
