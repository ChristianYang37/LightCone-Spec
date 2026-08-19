"""Physical evidence chain for a resident trusted-operator TP1 session.

The objects in this module are deliberately separate from the legacy
``UnsignedPinnedSglangServingRunReceipt``.  A scientific member trace does not
own the resident server process and therefore cannot truthfully claim a process
exit or an empty process group.  Those facts are recorded once, by the shared
close receipt, and a per-cell manifest may be published only after that receipt
has been reopened successfully.

This is trusted single-operator empirical evidence.  It never constructs a
``VerifiedNativeRuntimeGpuProof`` and never makes a formal ``MEASURED`` claim.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Literal, Protocol, Self
from urllib.parse import urlsplit

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.orchestration.formal_serving_session_group import (
    FormalServingSessionGroupPlan,
    FormalServingSessionGroupSpec,
)
from lightcone_spec.orchestration.formal_serving_session_group_worker import (
    FormalServingResidentFinalizedMemberResult,
    FormalServingResidentTracePhysicalResult,
    FormalServingSessionGroupPhysicalRuntime,
    FormalServingSessionMemberPhysicalResult,
    RevalidatedFormalServingSessionGroupExecution,
)
from lightcone_spec.orchestration.native_terminal import NativeTerminalRunBinding
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_SERVING_RESIDENT_PHYSICAL_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_serving_resident_physical_evidence",
        "scope": "prepared_ordinary_tp1_only",
        "process": "one_shared_launch_and_one_shared_close",
        "group_launch": (
            "path_bound_group_RunConfig_adaptation_argv_environment_and_port"
        ),
        "members": (
            "ordered_reset_boundary_then_complete_terminal_itl_lifecycle_junit_trace"
        ),
        "raw_file_bindings": "junit_and_shared_server_log_stdout_stderr",
        "source_chain": (
            "path_only_capability_initial_reset_warmup_clock_trace_terminal_close_"
            "manifest_reopened_from_shared_close"
        ),
        "terminal_projection": (
            "session_plan_sha256_epoch_and_group_bound_unique_run_attempt_ids"
        ),
        "publication": "per_cell_manifest_only_after_shared_close_and_pg_empty",
        "failure": "retain_prefix_evidence_then_fresh_process_remainder",
        "claim": "trusted_single_operator_empirical_no_signature",
        "formal_measured": False,
    }
)
_PROCESS_PATH_PLACEHOLDERS = {
    "--checkout": "<PATCHED_SGLANG_CHECKOUT>",
    "--compile-cache-plan": "<COMPILE_CACHE_PLAN>",
    "--run-config": "<GROUP_RUN_CONFIG>",
    "--model-path": "<TARGET_SNAPSHOT>",
    "--speculative-draft-model-path": "<DRAFTER_SNAPSHOT>",
    "--host": "<GROUP_HOST>",
    "--port": "<GROUP_PORT>",
    "--speculative-adaptation-config": "<GROUP_ADAPTATION_CONFIG>",
    "--speculative-adaptation-telemetry-path": "<GROUP_TELEMETRY_PATH>",
}


def _sha(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _text(label: str, value: object) -> str:
    if type(value) is not str or not value or any(c in value for c in "\n\r\x00"):
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


def _binding(label: str, value: object) -> CanonicalJsonProofBinding:
    if type(value) is not CanonicalJsonProofBinding:
        raise TypeError(f"{label} must be a canonical JSON binding")
    if CanonicalJsonProofBinding.bind(value.absolute_path) != value:
        raise ValueError(f"{label} changed after publication")
    return value


def _publish(path: str | Path, value: object) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _local_process_group_exists(process_group_id: int) -> bool:
    """Probe the local process group without signaling or mutating it."""

    if type(process_group_id) is not int or process_group_id < 1:
        raise ValueError("resident process group ID is invalid")
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _local_process_identity_matches(*, process_id: int, process_group_id: int) -> bool:
    if (
        type(process_id) is not int
        or process_id < 1
        or type(process_group_id) is not int
        or process_group_id < 1
    ):
        raise ValueError("resident local process identity is invalid")
    try:
        return os.getpgid(process_id) == process_group_id
    except ProcessLookupError:
        return False


def _validate_resident_launch_gpu_snapshots(
    *,
    plan: FormalServingSessionGroupPlan,
    before: CanonicalJsonProofBinding,
    ready: CanonicalJsonProofBinding,
    process_started_ns: int,
    process_group_id: int,
    server_ready_ns: int,
) -> None:
    from lightcone_spec.orchestration.live_sglang import (
        PinnedNvidiaSmiTool,
        validate_pinned_sglang_gpu_process_snapshot,
    )

    ready_raw = ready.reopen()
    if type(ready_raw) is not dict:
        raise TypeError("resident ready GPU snapshot is not an object")
    tool = PinnedNvidiaSmiTool.from_dict(ready_raw.get("nvidia_smi"))
    tool.revalidate()
    gpu_uuids = plan.assigned_gpu_uuids
    inventory_sha256 = plan.members[0].inventory_sha256
    before_raw = validate_pinned_sglang_gpu_process_snapshot(
        before,
        expected_tool=tool,
        expected_gpu_uuids=gpu_uuids,
        expected_inventory_sha256=inventory_sha256,
        expected_phase="before",
    )
    ready_validated = validate_pinned_sglang_gpu_process_snapshot(
        ready,
        expected_tool=tool,
        expected_gpu_uuids=gpu_uuids,
        expected_inventory_sha256=inventory_sha256,
        expected_phase="ready",
        expected_server_process_group_ids=(process_group_id,),
    )
    if not (
        before_raw["captured_ns"]
        <= process_started_ns
        <= ready_validated["captured_ns"]
        <= server_ready_ns
    ):
        raise ValueError("resident launch GPU snapshot lifecycle differs")


def _validate_resident_after_gpu_snapshot(
    *,
    plan: FormalServingSessionGroupPlan,
    launch: FormalServingResidentSharedLaunchReceipt,
    after: CanonicalJsonProofBinding,
    process_group_empty_checked_ns: int,
    evidence_flush_completed_ns: int,
) -> None:
    from lightcone_spec.orchestration.live_sglang import (
        PinnedNvidiaSmiTool,
        validate_pinned_sglang_gpu_process_snapshot,
    )

    ready_raw = launch.ready_gpu_snapshot.reopen()
    if type(ready_raw) is not dict:
        raise TypeError("resident ready GPU snapshot is not an object")
    tool = PinnedNvidiaSmiTool.from_dict(ready_raw.get("nvidia_smi"))
    tool.revalidate()
    after_raw = validate_pinned_sglang_gpu_process_snapshot(
        after,
        expected_tool=tool,
        expected_gpu_uuids=plan.assigned_gpu_uuids,
        expected_inventory_sha256=plan.members[0].inventory_sha256,
        expected_phase="after",
    )
    if not (
        process_group_empty_checked_ns
        <= after_raw["captured_ns"]
        <= evidence_flush_completed_ns
    ):
        raise ValueError("resident after GPU snapshot lifecycle differs")


def _terminal_binding_to_dict(value: NativeTerminalRunBinding) -> dict[str, object]:
    value.validate()
    return {
        "run_id": value.run_id,
        "run_nonce_sha256": value.run_nonce_sha256,
        "execution_plan_sha256": value.execution_plan_sha256,
        "rank_config_sha256": value.rank_config_sha256,
        "attempt_id": value.attempt_id,
        "session_id": value.session_id,
        "session_epoch": value.session_epoch,
        "previous_run_id": value.previous_run_id,
        "challenge_nonce_sha256": value.challenge_nonce_sha256,
        "method": value.method,
        "warmup_request_ids": list(value.warmup_request_ids),
        "scored_request_ids": list(value.scored_request_ids),
    }


def _terminal_binding_from_dict(value: object) -> NativeTerminalRunBinding:
    fields = {
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
    if type(value) is not dict or set(value) != fields:
        raise ValueError("resident terminal binding fields differ")
    row = dict(value)
    warmup = row.pop("warmup_request_ids")
    scored = row.pop("scored_request_ids")
    if type(warmup) is not list or type(scored) is not list:
        raise TypeError("resident terminal request IDs must be arrays")
    result = NativeTerminalRunBinding(
        **row,
        warmup_request_ids=tuple(warmup),
        scored_request_ids=tuple(scored),
    )
    result.validate()
    return result


def _resident_actual_argv_process_projection(
    argv: tuple[str, ...], *, plan: FormalServingSessionGroupPlan
) -> tuple[str, ...]:
    key = plan.normalized_process_key
    if key is None or not argv:
        raise ValueError("resident actual argv lacks its process key")
    projected = list(argv)
    for flag, replacement in _PROCESS_PATH_PLACEHOLDERS.items():
        positions = tuple(
            index for index, value in enumerate(projected) if value == flag
        )
        if len(positions) > 1 or (positions and positions[0] + 1 >= len(projected)):
            raise ValueError(f"resident actual argv differs at {flag}")
        if positions:
            projected[positions[0] + 1] = replacement
    for flag, replacement in (
        ("--run-config-sha256", key.process_run_config_sha256),
        ("--compile-cache-plan-sha256", key.compile_cache_process_sha256),
    ):
        positions = tuple(
            index for index, value in enumerate(projected) if value == flag
        )
        if len(positions) != 1 or positions[0] + 1 >= len(projected):
            raise ValueError(f"resident actual argv lacks {flag}")
        projected[positions[0] + 1] = replacement
    result = tuple(projected)
    if (
        result != key.normalized_server_argv
        or content_sha256({"argv": list(result)}) != key.normalized_server_argv_sha256
    ):
        raise ValueError("resident actual argv leaves normalized process semantics")
    return result


def _receipt_path(root: Path, name: str) -> Path:
    if not root.is_absolute() or root != root.resolve(strict=False):
        raise ValueError("resident evidence root must be absolute and normalized")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root / name


def formal_serving_resident_session_binding_sha256(
    *,
    group_plan: CanonicalJsonProofBinding,
    reset_authority: CanonicalJsonProofBinding,
    process_key_sha256: str,
    server_process_id: int,
    server_process_group_id: int,
    server_process_started_ns: int,
) -> str:
    """Derive the immutable identity of one concrete resident process."""

    _binding("resident group plan", group_plan)
    _binding("resident reset authority", reset_authority)
    _sha("resident process key", process_key_sha256)
    if (
        type(server_process_id) is not int
        or server_process_id < 1
        or type(server_process_group_id) is not int
        or server_process_group_id < 1
        or type(server_process_started_ns) is not int
        or server_process_started_ns < 1
    ):
        raise ValueError("resident process identity is invalid")
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_serving_resident_session_binding",
            "group_plan_sha256": group_plan.semantic_sha256,
            "reset_authority_sha256": reset_authority.semantic_sha256,
            "process_key_sha256": process_key_sha256,
            "server_process_id": server_process_id,
            "server_process_group_id": server_process_group_id,
            "server_process_started_ns": server_process_started_ns,
        }
    )


def effective_formal_serving_resident_terminal_binding(
    *, plan: FormalServingSessionGroupPlan, member_index: int
) -> NativeTerminalRunBinding:
    """Project a registered cell binding onto the shared session deterministically."""

    if (
        type(plan) is not FormalServingSessionGroupPlan
        or plan.execution_mode != "shared_session_tp1"
        or type(member_index) is not int
        or not 0 <= member_index < len(plan.members)
    ):
        raise ValueError("resident terminal binding projection is out of scope")
    member = plan.members[member_index]
    registered = member.run_plan.reopen()
    if type(registered) is not dict:
        raise TypeError("resident registered run plan must be an object")
    raw_binding = registered.get("native_terminal_binding")
    binding = _terminal_binding_from_dict(raw_binding)

    def projected_id(*, kind: str, index: int, value: str) -> str:
        return (
            f"resident:{plan.group_id[:24]}:{index + 1}:{kind}:"
            f"{content_sha256({'registered': value})[:16]}"
        )

    prior_run_id = None
    if member_index:
        previous = _terminal_binding_from_dict(
            plan.members[member_index - 1]
            .run_plan.reopen()
            .get("native_terminal_binding")
        )
        prior_run_id = projected_id(
            kind="run",
            index=member_index - 1,
            value=previous.run_id,
        )
    result = replace(
        binding,
        # The source-owned session contract names its plan by this canonical
        # SHA-256.  Using a display namespace here made the otherwise-valid
        # resident projection impossible to execute through session-live.
        session_id=plan.session_plan_sha256,
        session_epoch=member_index + 1,
        previous_run_id=prior_run_id,
        run_id=projected_id(
            kind="run",
            index=member_index,
            value=binding.run_id,
        ),
        attempt_id=projected_id(
            kind="attempt",
            index=member_index,
            value=binding.attempt_id,
        ),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class FormalServingResidentSharedLaunchReceipt:
    schema_version: Literal[1]
    kind: Literal["formal_serving_resident_shared_launch_receipt"]
    protocol_sha256: str
    group_plan: CanonicalJsonProofBinding
    reset_authority: CanonicalJsonProofBinding
    group_launch_authority: CanonicalJsonProofBinding
    normalized_process_key_sha256: str
    group_session_binding_sha256: str
    gpu_uuid: str
    server_process_id: int
    server_process_group_id: int
    server_process_started_ns: int
    server_ready_ns: int
    actual_server_argv: tuple[str, ...]
    actual_server_argv_sha256: str
    base_url: str
    before_gpu_snapshot: CanonicalJsonProofBinding
    ready_gpu_snapshot: CanonicalJsonProofBinding
    server_log_path: str
    server_stdout_path: str
    server_stderr_path: str
    evidence_level: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_resident_shared_launch_receipt"
            or self.protocol_sha256 != FORMAL_SERVING_RESIDENT_PHYSICAL_PROTOCOL_SHA256
            or self.evidence_level != "trusted_single_operator_empirical_no_signature"
            or self.formal_measured is not False
        ):
            raise ValueError("resident shared launch schema differs")
        _binding("resident launch group plan", self.group_plan)
        _binding("resident launch authority", self.reset_authority)
        _binding("resident group-scoped launch authority", self.group_launch_authority)
        _binding("resident before GPU snapshot", self.before_gpu_snapshot)
        _binding("resident ready GPU snapshot", self.ready_gpu_snapshot)
        _sha("resident process key", self.normalized_process_key_sha256)
        _sha("resident group session", self.group_session_binding_sha256)
        _text("resident GPU UUID", self.gpu_uuid)
        parsed_base_url = urlsplit(_text("resident base URL", self.base_url))
        if (
            type(self.actual_server_argv) is not tuple
            or not self.actual_server_argv
            or any(
                type(item) is not str or not item for item in self.actual_server_argv
            )
            or content_sha256({"argv": list(self.actual_server_argv)})
            != self.actual_server_argv_sha256
        ):
            raise ValueError("resident actual server argv differs")
        for label, value in (
            ("PID", self.server_process_id),
            ("PGID", self.server_process_group_id),
            ("start", self.server_process_started_ns),
            ("ready", self.server_ready_ns),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"resident server {label} is invalid")
        if self.server_ready_ns < self.server_process_started_ns:
            raise ValueError("resident server became ready before it started")
        if (
            parsed_base_url.scheme != "http"
            or parsed_base_url.hostname not in {"127.0.0.1", "localhost"}
            or parsed_base_url.port is None
            or parsed_base_url.username is not None
            or parsed_base_url.password is not None
            or parsed_base_url.path not in {"", "/"}
            or parsed_base_url.query
            or parsed_base_url.fragment
        ):
            raise ValueError("resident base URL is not exact loopback HTTP")
        argv_values: dict[str, str] = {}
        for flag in ("--host", "--port"):
            positions = tuple(
                index
                for index, value in enumerate(self.actual_server_argv)
                if value == flag
            )
            if len(positions) != 1 or positions[0] + 1 >= len(self.actual_server_argv):
                raise ValueError(f"resident actual argv lacks one {flag}")
            argv_values[flag] = self.actual_server_argv[positions[0] + 1]
        if argv_values["--host"] != parsed_base_url.hostname or argv_values[
            "--port"
        ] != str(parsed_base_url.port):
            raise ValueError("resident base URL differs from actual server argv")
        for label, value in (
            ("server log", self.server_log_path),
            ("stdout", self.server_stdout_path),
            ("stderr", self.server_stderr_path),
        ):
            path = Path(value)
            if not path.is_absolute() or path != path.resolve(strict=False):
                raise ValueError(f"resident {label} path differs")
        expected = formal_serving_resident_session_binding_sha256(
            group_plan=self.group_plan,
            reset_authority=self.reset_authority,
            process_key_sha256=self.normalized_process_key_sha256,
            server_process_id=self.server_process_id,
            server_process_group_id=self.server_process_group_id,
            server_process_started_ns=self.server_process_started_ns,
        )
        if self.group_session_binding_sha256 != expected:
            raise ValueError("resident group session binding differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = asdict(self)
        value["group_plan"] = self.group_plan.to_dict()
        value["reset_authority"] = self.reset_authority.to_dict()
        value["group_launch_authority"] = self.group_launch_authority.to_dict()
        value["before_gpu_snapshot"] = self.before_gpu_snapshot.to_dict()
        value["ready_gpu_snapshot"] = self.ready_gpu_snapshot.to_dict()
        value["actual_server_argv"] = list(self.actual_server_argv)
        if include_sha256:
            value["receipt_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "receipt_sha256",
        }:
            raise ValueError("resident shared launch fields differ")
        row = dict(value)
        declared = _sha("resident shared launch", row.pop("receipt_sha256"))
        for name in (
            "group_plan",
            "reset_authority",
            "group_launch_authority",
            "before_gpu_snapshot",
            "ready_gpu_snapshot",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["actual_server_argv"] = tuple(row["actual_server_argv"])
        result = cls(**row)  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("resident shared launch digest differs")
        return result


@dataclass(frozen=True)
class FormalServingResidentResetBoundaryReceipt:
    schema_version: Literal[1]
    kind: Literal["formal_serving_resident_reset_boundary_receipt"]
    protocol_sha256: str
    group_plan: CanonicalJsonProofBinding
    shared_launch: CanonicalJsonProofBinding
    reset_authority_sha256: str
    group_session_binding_sha256: str
    process_id: int
    process_group_id: int
    session_epoch: int
    prior_materialized_cell_id: str | None
    next_materialized_cell_id: str
    source_reset_receipt: CanonicalJsonProofBinding
    reset_started_ns: int
    reset_finished_ns: int
    all_reset_complete: Literal[True]
    request_queue_empty: Literal[True]
    optimizer_state_reset: Literal[True]
    adaptation_state_reset: Literal[True]
    candidate_state_reset: Literal[True]
    cache_policy_restored: Literal[True]
    terminal_writer_flushed: Literal[True]
    previous_requests_fully_terminal: Literal[True]
    hbm_allocated_bytes: int
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_resident_reset_boundary_receipt"
            or self.protocol_sha256 != FORMAL_SERVING_RESIDENT_PHYSICAL_PROTOCOL_SHA256
            or self.formal_measured is not False
            or any(
                value is not True
                for value in (
                    self.all_reset_complete,
                    self.request_queue_empty,
                    self.optimizer_state_reset,
                    self.adaptation_state_reset,
                    self.candidate_state_reset,
                    self.cache_policy_restored,
                    self.terminal_writer_flushed,
                    self.previous_requests_fully_terminal,
                )
            )
        ):
            raise ValueError("resident reset boundary is not clean")
        for name, value in (
            ("group plan", self.group_plan),
            ("shared launch", self.shared_launch),
            ("source reset", self.source_reset_receipt),
        ):
            _binding(f"resident reset {name}", value)
        _sha("resident reset authority", self.reset_authority_sha256)
        _sha("resident reset session", self.group_session_binding_sha256)
        _text("resident reset next cell", self.next_materialized_cell_id)
        if self.prior_materialized_cell_id is not None:
            _text("resident reset prior cell", self.prior_materialized_cell_id)
        if (
            type(self.process_id) is not int
            or self.process_id < 1
            or type(self.process_group_id) is not int
            or self.process_group_id < 1
            or type(self.session_epoch) is not int
            or self.session_epoch < 1
            or type(self.reset_started_ns) is not int
            or type(self.reset_finished_ns) is not int
            or self.reset_started_ns < 1
            or self.reset_finished_ns < self.reset_started_ns
            or type(self.hbm_allocated_bytes) is not int
            or self.hbm_allocated_bytes < 0
        ):
            raise ValueError("resident reset lifecycle differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = asdict(self)
        for name in ("group_plan", "shared_launch", "source_reset_receipt"):
            value[name] = getattr(self, name).to_dict()
        if include_sha256:
            value["receipt_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "receipt_sha256",
        }:
            raise ValueError("resident reset boundary fields differ")
        row = dict(value)
        declared = _sha("resident reset boundary", row.pop("receipt_sha256"))
        for name in ("group_plan", "shared_launch", "source_reset_receipt"):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        result = cls(**row)  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("resident reset boundary digest differs")
        return result


@dataclass(frozen=True)
class FormalServingResidentTraceReceipt:
    schema_version: Literal[1]
    kind: Literal["formal_serving_resident_trace_receipt"]
    protocol_sha256: str
    group_plan: CanonicalJsonProofBinding
    shared_launch: CanonicalJsonProofBinding
    reset_boundary: CanonicalJsonProofBinding
    member_run_plan: CanonicalJsonProofBinding
    group_session_binding_sha256: str
    materialized_cell_id: str
    member_index: int
    session_epoch: int
    process_id: int
    process_group_id: int
    effective_terminal_binding: NativeTerminalRunBinding
    raw_terminal: CanonicalJsonProofBinding
    native_itl: CanonicalJsonProofBinding
    client_lifecycle: CanonicalJsonProofBinding
    junit: EvidenceFileBinding
    trace_lifecycle: CanonicalJsonProofBinding
    trace_started_ns: int
    scored_started_ns: int
    trace_finished_ns: int
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_resident_trace_receipt"
            or self.protocol_sha256 != FORMAL_SERVING_RESIDENT_PHYSICAL_PROTOCOL_SHA256
            or self.formal_measured is not False
        ):
            raise ValueError("resident trace schema differs")
        for name, value in (
            ("group plan", self.group_plan),
            ("shared launch", self.shared_launch),
            ("reset boundary", self.reset_boundary),
            ("member run plan", self.member_run_plan),
            ("raw terminal", self.raw_terminal),
            ("native ITL", self.native_itl),
            ("client lifecycle", self.client_lifecycle),
            ("trace lifecycle", self.trace_lifecycle),
        ):
            _binding(f"resident trace {name}", value)
        if type(self.junit) is not EvidenceFileBinding:
            raise TypeError("resident trace JUnit binding type differs")
        self.junit.reopen(label="resident trace JUnit")
        _sha("resident trace session", self.group_session_binding_sha256)
        _text("resident trace cell", self.materialized_cell_id)
        if type(self.effective_terminal_binding) is not NativeTerminalRunBinding:
            raise TypeError("resident trace terminal binding type differs")
        self.effective_terminal_binding.validate()
        if (
            type(self.member_index) is not int
            or self.member_index < 0
            or self.session_epoch != self.member_index + 1
            or type(self.process_id) is not int
            or self.process_id < 1
            or type(self.process_group_id) is not int
            or self.process_group_id < 1
            or any(
                type(value) is not int or value < 1
                for value in (
                    self.trace_started_ns,
                    self.scored_started_ns,
                    self.trace_finished_ns,
                )
            )
            or not (
                self.trace_started_ns
                <= self.scored_started_ns
                <= self.trace_finished_ns
            )
        ):
            raise ValueError("resident trace lifecycle differs")

    @property
    def run_binding_sha256(self) -> str:
        return content_sha256(self.effective_terminal_binding.begin_payload())

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "group_plan",
            "shared_launch",
            "reset_boundary",
            "member_run_plan",
            "raw_terminal",
            "native_itl",
            "client_lifecycle",
            "trace_lifecycle",
        ):
            value[name] = getattr(self, name).to_dict()
        value["junit"] = self.junit.to_dict()
        value["effective_terminal_binding"] = _terminal_binding_to_dict(
            self.effective_terminal_binding
        )
        if include_sha256:
            value["receipt_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "receipt_sha256",
        }:
            raise ValueError("resident trace receipt fields differ")
        row = dict(value)
        declared = _sha("resident trace receipt", row.pop("receipt_sha256"))
        for name in (
            "group_plan",
            "shared_launch",
            "reset_boundary",
            "member_run_plan",
            "raw_terminal",
            "native_itl",
            "client_lifecycle",
            "trace_lifecycle",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["junit"] = EvidenceFileBinding.from_dict(
            row["junit"], label="resident trace JUnit"
        )
        row["effective_terminal_binding"] = _terminal_binding_from_dict(
            row["effective_terminal_binding"]
        )
        result = cls(**row)  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("resident trace receipt digest differs")
        return result


@dataclass(frozen=True)
class FormalServingResidentSharedCloseReceipt:
    schema_version: Literal[1]
    kind: Literal["formal_serving_resident_shared_close_receipt"]
    protocol_sha256: str
    group_plan: CanonicalJsonProofBinding
    shared_launch: CanonicalJsonProofBinding
    group_session_binding_sha256: str
    source_close_receipt: CanonicalJsonProofBinding | None
    member_trace_receipts: tuple[CanonicalJsonProofBinding, ...]
    gpu_uuid: str
    server_process_id: int
    server_process_group_id: int
    server_process_started_ns: int
    close_started_ns: int
    process_exited_ns: int
    process_exit_code: int
    process_group_empty: Literal[True]
    process_group_empty_checked_ns: int
    evidence_flush_completed_ns: int
    cleanup_kind: Literal[
        "source_close_sigterm", "forced_sigterm", "already_exited_clean"
    ]
    after_gpu_snapshot: CanonicalJsonProofBinding
    server_log: EvidenceFileBinding
    server_stdout: EvidenceFileBinding
    server_stderr: EvidenceFileBinding
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_resident_shared_close_receipt"
            or self.protocol_sha256 != FORMAL_SERVING_RESIDENT_PHYSICAL_PROTOCOL_SHA256
            or self.process_group_empty is not True
            or self.formal_measured is not False
        ):
            raise ValueError("resident shared close schema differs")
        for name, value in (
            ("group plan", self.group_plan),
            ("shared launch", self.shared_launch),
            ("after GPU", self.after_gpu_snapshot),
            *(
                ()
                if self.source_close_receipt is None
                else (("source close", self.source_close_receipt),)
            ),
            *(
                (f"member trace {index}", item)
                for index, item in enumerate(self.member_trace_receipts)
            ),
        ):
            _binding(f"resident close {name}", value)
        for name, value in (
            ("server log", self.server_log),
            ("stdout", self.server_stdout),
            ("stderr", self.server_stderr),
        ):
            if type(value) is not EvidenceFileBinding:
                raise TypeError(f"resident close {name} binding type differs")
            value.reopen(label=f"resident close {name}")
        _sha("resident close session", self.group_session_binding_sha256)
        _text("resident close GPU", self.gpu_uuid)
        if type(self.member_trace_receipts) is not tuple or len(
            {item.semantic_sha256 for item in self.member_trace_receipts}
        ) != len(self.member_trace_receipts):
            raise ValueError("resident close trace coverage is not canonical")
        values = (
            self.server_process_started_ns,
            self.close_started_ns,
            self.process_exited_ns,
            self.process_group_empty_checked_ns,
            self.evidence_flush_completed_ns,
        )
        if (
            type(self.server_process_id) is not int
            or self.server_process_id < 1
            or type(self.server_process_group_id) is not int
            or self.server_process_group_id < 1
            or type(self.process_exit_code) is not int
            or any(type(value) is not int or value < 1 for value in values)
            or values != tuple(sorted(values))
        ):
            raise ValueError("resident shared close lifecycle differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "group_plan",
            "shared_launch",
            "after_gpu_snapshot",
        ):
            value[name] = getattr(self, name).to_dict()
        for name in ("server_log", "server_stdout", "server_stderr"):
            value[name] = getattr(self, name).to_dict()
        value["source_close_receipt"] = (
            None
            if self.source_close_receipt is None
            else self.source_close_receipt.to_dict()
        )
        value["member_trace_receipts"] = [
            item.to_dict() for item in self.member_trace_receipts
        ]
        if include_sha256:
            value["receipt_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "receipt_sha256",
        }:
            raise ValueError("resident shared close fields differ")
        row = dict(value)
        declared = _sha("resident shared close", row.pop("receipt_sha256"))
        for name in (
            "group_plan",
            "shared_launch",
            "after_gpu_snapshot",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        for name, label in (
            ("server_log", "resident close server log"),
            ("server_stdout", "resident close stdout"),
            ("server_stderr", "resident close stderr"),
        ):
            row[name] = EvidenceFileBinding.from_dict(row[name], label=label)
        if row["source_close_receipt"] is not None:
            row["source_close_receipt"] = CanonicalJsonProofBinding.from_dict(
                row["source_close_receipt"]
            )
        row["member_trace_receipts"] = tuple(
            CanonicalJsonProofBinding.from_dict(item)
            for item in row["member_trace_receipts"]
        )
        result = cls(**row)  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("resident shared close digest differs")
        return result


def publish_formal_serving_resident_shared_launch_receipt(
    *, receipt: FormalServingResidentSharedLaunchReceipt, output_path: str | Path
) -> CanonicalJsonProofBinding:
    binding = _publish(output_path, receipt.to_dict())
    if (
        revalidate_formal_serving_resident_shared_launch_receipt(output_path)[1]
        != receipt
    ):
        raise RuntimeError("resident shared launch changed during publication")
    return binding


def revalidate_formal_serving_resident_shared_launch_receipt(
    path: str | Path,
) -> tuple[CanonicalJsonProofBinding, FormalServingResidentSharedLaunchReceipt]:
    binding = CanonicalJsonProofBinding.bind(path)
    receipt = FormalServingResidentSharedLaunchReceipt.from_dict(binding.reopen())
    plan = FormalServingSessionGroupPlan.from_dict(receipt.group_plan.reopen())
    from lightcone_spec.experiments.formal_single_operator_session_reset import (
        revalidate_trusted_empirical_tp1_session_reset_authority,
    )

    authority_binding, authority = (
        revalidate_trusted_empirical_tp1_session_reset_authority(
            receipt.reset_authority.absolute_path
        )
    )
    if (
        plan.execution_mode != "shared_session_tp1"
        or authority_binding != receipt.reset_authority
        or plan.reset_authority_sha256 != authority.sha256
        or plan.normalized_process_key is None
        or plan.normalized_process_key.sha256 != receipt.normalized_process_key_sha256
        or plan.assigned_gpu_uuids != (receipt.gpu_uuid,)
    ):
        raise ValueError("resident shared launch leaves its group plan")
    _resident_actual_argv_process_projection(receipt.actual_server_argv, plan=plan)
    from lightcone_spec.orchestration.formal_serving_session_group_launch import (
        revalidate_formal_serving_resident_group_launch_authority,
    )

    group_launch = revalidate_formal_serving_resident_group_launch_authority(
        receipt.group_launch_authority.absolute_path
    )
    if (
        group_launch.binding != receipt.group_launch_authority
        or group_launch.authority.group_plan != receipt.group_plan
        or group_launch.authority.actual_server_argv != receipt.actual_server_argv
        or group_launch.authority.port != urlsplit(receipt.base_url).port
    ):
        raise ValueError("resident shared launch leaves group-scoped launch authority")
    _validate_resident_launch_gpu_snapshots(
        plan=plan,
        before=receipt.before_gpu_snapshot,
        ready=receipt.ready_gpu_snapshot,
        process_started_ns=receipt.server_process_started_ns,
        process_group_id=receipt.server_process_group_id,
        server_ready_ns=receipt.server_ready_ns,
    )
    return binding, receipt


def publish_formal_serving_resident_reset_boundary_receipt(
    *, receipt: FormalServingResidentResetBoundaryReceipt, output_path: str | Path
) -> CanonicalJsonProofBinding:
    binding = _publish(output_path, receipt.to_dict())
    revalidate_formal_serving_resident_reset_boundary_receipt(output_path)
    return binding


def revalidate_formal_serving_resident_reset_boundary_receipt(
    path: str | Path,
) -> tuple[CanonicalJsonProofBinding, FormalServingResidentResetBoundaryReceipt]:
    binding = CanonicalJsonProofBinding.bind(path)
    receipt = FormalServingResidentResetBoundaryReceipt.from_dict(binding.reopen())
    _launch_binding, launch = revalidate_formal_serving_resident_shared_launch_receipt(
        receipt.shared_launch.absolute_path
    )
    plan = FormalServingSessionGroupPlan.from_dict(receipt.group_plan.reopen())
    if (
        receipt.group_plan != launch.group_plan
        or receipt.group_session_binding_sha256 != launch.group_session_binding_sha256
        or receipt.reset_authority_sha256 != plan.reset_authority_sha256
        or (receipt.process_id, receipt.process_group_id)
        != (launch.server_process_id, launch.server_process_group_id)
        or receipt.session_epoch > len(plan.members)
    ):
        raise ValueError("resident reset leaves its shared launch")
    index = receipt.session_epoch - 1
    member = plan.members[index]
    expected_prior = (
        None if index == 0 else plan.members[index - 1].materialized_cell_id
    )
    if (
        receipt.next_materialized_cell_id != member.materialized_cell_id
        or receipt.prior_materialized_cell_id != expected_prior
    ):
        raise ValueError("resident reset breaks member order")
    return binding, receipt


def publish_formal_serving_resident_trace_receipt(
    *, receipt: FormalServingResidentTraceReceipt, output_path: str | Path
) -> CanonicalJsonProofBinding:
    binding = _publish(output_path, receipt.to_dict())
    revalidate_formal_serving_resident_trace_receipt(output_path)
    return binding


def revalidate_formal_serving_resident_trace_receipt(
    path: str | Path,
) -> tuple[CanonicalJsonProofBinding, FormalServingResidentTraceReceipt]:
    binding = CanonicalJsonProofBinding.bind(path)
    receipt = FormalServingResidentTraceReceipt.from_dict(binding.reopen())
    _launch_binding, launch = revalidate_formal_serving_resident_shared_launch_receipt(
        receipt.shared_launch.absolute_path
    )
    _reset_binding, reset = revalidate_formal_serving_resident_reset_boundary_receipt(
        receipt.reset_boundary.absolute_path
    )
    plan = FormalServingSessionGroupPlan.from_dict(receipt.group_plan.reopen())
    if not 0 <= receipt.member_index < len(plan.members):
        raise ValueError("resident trace member index leaves group")
    member = plan.members[receipt.member_index]
    expected_binding = effective_formal_serving_resident_terminal_binding(
        plan=plan, member_index=receipt.member_index
    )
    if (
        receipt.group_plan != launch.group_plan
        or receipt.group_plan != reset.group_plan
        or receipt.group_session_binding_sha256 != launch.group_session_binding_sha256
        or receipt.materialized_cell_id != member.materialized_cell_id
        or receipt.member_run_plan != member.run_plan
        or receipt.session_epoch != reset.session_epoch
        or (receipt.process_id, receipt.process_group_id)
        != (launch.server_process_id, launch.server_process_group_id)
        or receipt.effective_terminal_binding != expected_binding
        or launch.server_ready_ns > reset.reset_started_ns
        or reset.reset_finished_ns > receipt.trace_started_ns
    ):
        raise ValueError("resident trace leaves its member/reset/session")
    return binding, receipt


def publish_formal_serving_resident_shared_close_receipt(
    *, receipt: FormalServingResidentSharedCloseReceipt, output_path: str | Path
) -> CanonicalJsonProofBinding:
    binding = _publish(output_path, receipt.to_dict())
    revalidate_formal_serving_resident_shared_close_receipt(output_path)
    return binding


def revalidate_formal_serving_resident_shared_close_receipt(
    path: str | Path,
) -> tuple[CanonicalJsonProofBinding, FormalServingResidentSharedCloseReceipt]:
    binding = CanonicalJsonProofBinding.bind(path)
    receipt = FormalServingResidentSharedCloseReceipt.from_dict(binding.reopen())
    _launch_binding, launch = revalidate_formal_serving_resident_shared_launch_receipt(
        receipt.shared_launch.absolute_path
    )
    plan = FormalServingSessionGroupPlan.from_dict(receipt.group_plan.reopen())
    traces = tuple(
        revalidate_formal_serving_resident_trace_receipt(item.absolute_path)[1]
        for item in receipt.member_trace_receipts
    )
    if (
        receipt.group_plan != launch.group_plan
        or receipt.group_session_binding_sha256 != launch.group_session_binding_sha256
        or receipt.gpu_uuid != launch.gpu_uuid
        or (
            receipt.server_process_id,
            receipt.server_process_group_id,
            receipt.server_process_started_ns,
        )
        != (
            launch.server_process_id,
            launch.server_process_group_id,
            launch.server_process_started_ns,
        )
        or tuple(row.member_index for row in traces) != tuple(range(len(traces)))
        or len(traces) > len(plan.members)
        or any(
            row.group_session_binding_sha256 != receipt.group_session_binding_sha256
            for row in traces
        )
        or (traces and receipt.close_started_ns < traces[-1].trace_finished_ns)
        or any(
            current.trace_finished_ns > following.trace_started_ns
            for current, following in pairwise(traces)
        )
        or (
            receipt.server_log.absolute_path,
            receipt.server_stdout.absolute_path,
            receipt.server_stderr.absolute_path,
        )
        != (
            launch.server_log_path,
            launch.server_stdout_path,
            launch.server_stderr_path,
        )
    ):
        raise ValueError("resident close leaves its launch/ordered trace prefix")
    if (
        receipt.cleanup_kind in {"source_close_sigterm", "already_exited_clean"}
        and receipt.source_close_receipt is None
    ):
        raise ValueError("normal resident close lacks source close receipt")
    if (
        receipt.cleanup_kind == "forced_sigterm"
        and receipt.source_close_receipt is not None
    ):
        raise ValueError("forced resident close carries a source-chain claim")
    if receipt.source_close_receipt is not None:
        from lightcone_spec.orchestration.formal_serving_session_source_chain import (
            revalidate_formal_serving_resident_source_chain,
        )

        source_chain = revalidate_formal_serving_resident_source_chain(
            receipt.source_close_receipt
        )
        source_process = source_chain.capability.process_identity.rsplit(":", 1)
        if (
            source_chain.session_plan_sha256 != plan.session_plan_sha256
            or source_process != ["scheduler", str(receipt.server_process_id)]
            or source_chain.capability.process_started_ns
            != receipt.server_process_started_ns
            or source_chain.execution_plan_sha256s
            != tuple(
                row.effective_terminal_binding.execution_plan_sha256 for row in traces
            )
            or tuple(epoch.terminal_artifact for epoch in source_chain.epochs)
            != tuple(row.raw_terminal for row in traces)
        ):
            raise ValueError(
                "resident source chain leaves the shared process/ordered traces"
            )
    _validate_resident_after_gpu_snapshot(
        plan=plan,
        launch=launch,
        after=receipt.after_gpu_snapshot,
        process_group_empty_checked_ns=receipt.process_group_empty_checked_ns,
        evidence_flush_completed_ns=receipt.evidence_flush_completed_ns,
    )
    return binding, receipt


@dataclass(frozen=True)
class FormalServingResidentResetEvidence:
    """Driver-owned result of one source reset; no caller-authored digest."""

    source_reset_receipt: CanonicalJsonProofBinding
    reset_started_ns: int
    reset_finished_ns: int
    hbm_allocated_bytes: int
    request_queue_empty: bool
    optimizer_state_reset: bool
    adaptation_state_reset: bool
    candidate_state_reset: bool
    cache_policy_restored: bool
    terminal_writer_flushed: bool
    previous_requests_fully_terminal: bool

    def __post_init__(self) -> None:
        _binding("resident driver source reset", self.source_reset_receipt)
        if (
            type(self.reset_started_ns) is not int
            or type(self.reset_finished_ns) is not int
            or self.reset_started_ns < 1
            or self.reset_finished_ns < self.reset_started_ns
            or type(self.hbm_allocated_bytes) is not int
            or self.hbm_allocated_bytes < 0
        ):
            raise ValueError("resident driver reset lifecycle differs")
        for name, value in (
            ("request queue", self.request_queue_empty),
            ("optimizer state", self.optimizer_state_reset),
            ("adaptation state", self.adaptation_state_reset),
            ("candidate state", self.candidate_state_reset),
            ("cache policy", self.cache_policy_restored),
            ("terminal writer", self.terminal_writer_flushed),
            ("prior requests", self.previous_requests_fully_terminal),
        ):
            if type(value) is not bool:
                raise TypeError(f"resident driver reset {name} flag is not boolean")


@dataclass(frozen=True)
class FormalServingResidentTraceEvidence:
    effective_terminal_binding: NativeTerminalRunBinding
    raw_terminal: CanonicalJsonProofBinding
    native_itl: CanonicalJsonProofBinding
    client_lifecycle: CanonicalJsonProofBinding
    junit: EvidenceFileBinding
    trace_lifecycle: CanonicalJsonProofBinding
    trace_started_ns: int
    scored_started_ns: int
    trace_finished_ns: int

    def __post_init__(self) -> None:
        if type(self.effective_terminal_binding) is not NativeTerminalRunBinding:
            raise TypeError("resident driver trace terminal binding differs")
        self.effective_terminal_binding.validate()
        for name, value in (
            ("raw terminal", self.raw_terminal),
            ("native ITL", self.native_itl),
            ("client lifecycle", self.client_lifecycle),
            ("trace lifecycle", self.trace_lifecycle),
        ):
            _binding(f"resident driver trace {name}", value)
        if type(self.junit) is not EvidenceFileBinding:
            raise TypeError("resident driver trace JUnit binding differs")
        self.junit.reopen(label="resident driver trace JUnit")
        if any(
            type(value) is not int or value < 1
            for value in (
                self.trace_started_ns,
                self.scored_started_ns,
                self.trace_finished_ns,
            )
        ) or not (
            self.trace_started_ns <= self.scored_started_ns <= self.trace_finished_ns
        ):
            raise ValueError("resident driver trace lifecycle differs")


@dataclass(frozen=True)
class FormalServingResidentCloseEvidence:
    source_close_receipt: CanonicalJsonProofBinding | None
    server_process_id: int
    server_process_group_id: int
    close_started_ns: int
    process_exited_ns: int
    process_exit_code: int
    process_group_empty: bool
    process_group_empty_checked_ns: int
    evidence_flush_completed_ns: int
    cleanup_kind: Literal[
        "source_close_sigterm", "forced_sigterm", "already_exited_clean"
    ]
    after_gpu_snapshot: CanonicalJsonProofBinding
    server_log: EvidenceFileBinding
    server_stdout: EvidenceFileBinding
    server_stderr: EvidenceFileBinding

    def __post_init__(self) -> None:
        if self.source_close_receipt is not None:
            _binding("resident driver source close", self.source_close_receipt)
        _binding("resident driver after GPU", self.after_gpu_snapshot)
        for name, value in (
            ("server log", self.server_log),
            ("stdout", self.server_stdout),
            ("stderr", self.server_stderr),
        ):
            if type(value) is not EvidenceFileBinding:
                raise TypeError(f"resident driver close {name} binding differs")
            value.reopen(label=f"resident driver close {name}")
        if (
            type(self.server_process_id) is not int
            or self.server_process_id < 1
            or type(self.server_process_group_id) is not int
            or self.server_process_group_id < 1
            or type(self.process_exit_code) is not int
            or type(self.process_group_empty) is not bool
            or self.cleanup_kind
            not in {"source_close_sigterm", "forced_sigterm", "already_exited_clean"}
            or any(
                type(value) is not int or value < 1
                for value in (
                    self.close_started_ns,
                    self.process_exited_ns,
                    self.process_group_empty_checked_ns,
                    self.evidence_flush_completed_ns,
                )
            )
            or not (
                self.close_started_ns
                <= self.process_exited_ns
                <= self.process_group_empty_checked_ns
                <= self.evidence_flush_completed_ns
            )
        ):
            raise ValueError("resident driver close lifecycle differs")


class FormalServingResidentProcessDriver(Protocol):
    """Owns the one real process, HTTP pool, resets, traces, and termination."""

    @property
    def process_id(self) -> int: ...
    @property
    def process_group_id(self) -> int: ...
    @property
    def process_started_ns(self) -> int: ...
    @property
    def ready_ns(self) -> int: ...
    @property
    def actual_server_argv(self) -> tuple[str, ...]: ...
    @property
    def group_launch_authority(self) -> CanonicalJsonProofBinding: ...
    @property
    def base_url(self) -> str: ...
    @property
    def before_gpu_snapshot(self) -> CanonicalJsonProofBinding: ...
    @property
    def ready_gpu_snapshot(self) -> CanonicalJsonProofBinding: ...
    @property
    def server_log_path(self) -> str: ...
    @property
    def server_stdout_path(self) -> str: ...
    @property
    def server_stderr_path(self) -> str: ...

    async def reset_member(
        self, *, member: FormalServingSessionGroupSpec, member_index: int
    ) -> FormalServingResidentResetEvidence: ...

    async def execute_trace(
        self,
        *,
        member: FormalServingSessionGroupSpec,
        member_index: int,
        effective_terminal_binding: NativeTerminalRunBinding,
    ) -> FormalServingResidentTraceEvidence: ...

    async def close_session(
        self, *, force: bool
    ) -> FormalServingResidentCloseEvidence: ...


class FormalServingResidentProcessFactory(Protocol):
    async def launch(
        self,
        *,
        execution: RevalidatedFormalServingSessionGroupExecution,
        evidence_root: Path,
    ) -> FormalServingResidentProcessDriver: ...

    async def execute_fresh_member(
        self, *, member: FormalServingSessionGroupSpec, fallback_reason: str
    ) -> FormalServingSessionMemberPhysicalResult: ...


class FormalServingResidentSharedSessionHandle:
    """Evidence-publishing adapter around a single process-owning driver."""

    def __init__(
        self,
        *,
        execution: RevalidatedFormalServingSessionGroupExecution,
        driver: FormalServingResidentProcessDriver,
        evidence_root: Path,
        repository_root: Path,
    ) -> None:
        self._execution = execution
        self._driver = driver
        self._root = evidence_root
        self._repository_root = repository_root.resolve()
        if not (self._repository_root / ".git").exists():
            raise ValueError("resident repository root is not a Git checkout")
        self._trace_bindings: list[CanonicalJsonProofBinding] = []
        self._closed: CanonicalJsonProofBinding | None = None
        key = execution.plan.normalized_process_key
        if key is None:
            raise ValueError("resident shared session lacks a process key")
        if not _local_process_identity_matches(
            process_id=driver.process_id,
            process_group_id=driver.process_group_id,
        ):
            raise RuntimeError("resident driver PID does not own its claimed PGID")
        session_sha = formal_serving_resident_session_binding_sha256(
            group_plan=execution.plan_binding,
            reset_authority=execution.authority_binding,
            process_key_sha256=key.sha256,
            server_process_id=driver.process_id,
            server_process_group_id=driver.process_group_id,
            server_process_started_ns=driver.process_started_ns,
        )
        launch = FormalServingResidentSharedLaunchReceipt(
            schema_version=1,
            kind="formal_serving_resident_shared_launch_receipt",
            protocol_sha256=FORMAL_SERVING_RESIDENT_PHYSICAL_PROTOCOL_SHA256,
            group_plan=execution.plan_binding,
            reset_authority=execution.authority_binding,
            group_launch_authority=driver.group_launch_authority,
            normalized_process_key_sha256=key.sha256,
            group_session_binding_sha256=session_sha,
            gpu_uuid=execution.plan.assigned_gpu_uuids[0],
            server_process_id=driver.process_id,
            server_process_group_id=driver.process_group_id,
            server_process_started_ns=driver.process_started_ns,
            server_ready_ns=driver.ready_ns,
            actual_server_argv=driver.actual_server_argv,
            actual_server_argv_sha256=content_sha256(
                {"argv": list(driver.actual_server_argv)}
            ),
            base_url=driver.base_url,
            before_gpu_snapshot=driver.before_gpu_snapshot,
            ready_gpu_snapshot=driver.ready_gpu_snapshot,
            server_log_path=driver.server_log_path,
            server_stdout_path=driver.server_stdout_path,
            server_stderr_path=driver.server_stderr_path,
            evidence_level="trusted_single_operator_empirical_no_signature",
            formal_measured=False,
        )
        self._launch_binding = publish_formal_serving_resident_shared_launch_receipt(
            receipt=launch,
            output_path=_receipt_path(evidence_root, "shared-launch.json"),
        )
        self._launch = launch

    @property
    def process_id(self) -> int:
        return self._driver.process_id

    @property
    def process_group_id(self) -> int:
        return self._driver.process_group_id

    @property
    def process_started_ns(self) -> int:
        return self._driver.process_started_ns

    @property
    def shared_launch_binding(self) -> CanonicalJsonProofBinding:
        return self._launch_binding

    async def reset_for_member(
        self,
        *,
        session_plan_sha256: str,
        reset_authority_sha256: str,
        prior_member: FormalServingSessionGroupSpec | None,
        next_member: FormalServingSessionGroupSpec,
        session_epoch: int,
    ) -> CanonicalJsonProofBinding:
        if self._closed is not None:
            raise RuntimeError("resident session is already closed")
        if (
            session_plan_sha256 != self._execution.plan.session_plan_sha256
            or reset_authority_sha256 != self._execution.authority.sha256
            or session_epoch != len(self._trace_bindings) + 1
            or self._execution.plan.members[session_epoch - 1] != next_member
        ):
            raise ValueError("resident reset call leaves its group plan")
        expected_prior = (
            None
            if session_epoch == 1
            else self._execution.plan.members[session_epoch - 2]
        )
        if prior_member != expected_prior:
            raise ValueError("resident reset prior member differs")
        evidence = await self._driver.reset_member(
            member=next_member, member_index=session_epoch - 1
        )
        if type(evidence) is not FormalServingResidentResetEvidence:
            raise TypeError("resident driver reset evidence differs")
        if not all(
            (
                evidence.request_queue_empty,
                evidence.optimizer_state_reset,
                evidence.adaptation_state_reset,
                evidence.candidate_state_reset,
                evidence.cache_policy_restored,
                evidence.terminal_writer_flushed,
                evidence.previous_requests_fully_terminal,
            )
        ):
            raise RuntimeError("resident reset gate failed")
        receipt = FormalServingResidentResetBoundaryReceipt(
            schema_version=1,
            kind="formal_serving_resident_reset_boundary_receipt",
            protocol_sha256=FORMAL_SERVING_RESIDENT_PHYSICAL_PROTOCOL_SHA256,
            group_plan=self._execution.plan_binding,
            shared_launch=self._launch_binding,
            reset_authority_sha256=self._execution.authority.sha256,
            group_session_binding_sha256=self._launch.group_session_binding_sha256,
            process_id=self.process_id,
            process_group_id=self._driver.process_group_id,
            session_epoch=session_epoch,
            prior_materialized_cell_id=(
                None if prior_member is None else prior_member.materialized_cell_id
            ),
            next_materialized_cell_id=next_member.materialized_cell_id,
            source_reset_receipt=evidence.source_reset_receipt,
            reset_started_ns=evidence.reset_started_ns,
            reset_finished_ns=evidence.reset_finished_ns,
            all_reset_complete=True,
            request_queue_empty=evidence.request_queue_empty,
            optimizer_state_reset=evidence.optimizer_state_reset,
            adaptation_state_reset=evidence.adaptation_state_reset,
            candidate_state_reset=evidence.candidate_state_reset,
            cache_policy_restored=evidence.cache_policy_restored,
            terminal_writer_flushed=evidence.terminal_writer_flushed,
            previous_requests_fully_terminal=(
                evidence.previous_requests_fully_terminal
            ),
            hbm_allocated_bytes=evidence.hbm_allocated_bytes,
            formal_measured=False,
        )
        return publish_formal_serving_resident_reset_boundary_receipt(
            receipt=receipt,
            output_path=_receipt_path(self._root, f"reset-{session_epoch:04d}.json"),
        )

    async def execute_member(
        self, *, member: FormalServingSessionGroupSpec, session_epoch: int
    ) -> FormalServingResidentTracePhysicalResult:
        if session_epoch != len(self._trace_bindings) + 1:
            raise ValueError("resident trace epoch differs")
        reset_path = _receipt_path(self._root, f"reset-{session_epoch:04d}.json")
        reset_binding = CanonicalJsonProofBinding.bind(reset_path)
        effective = effective_formal_serving_resident_terminal_binding(
            plan=self._execution.plan, member_index=session_epoch - 1
        )
        evidence = await self._driver.execute_trace(
            member=member,
            member_index=session_epoch - 1,
            effective_terminal_binding=effective,
        )
        if (
            type(evidence) is not FormalServingResidentTraceEvidence
            or evidence.effective_terminal_binding != effective
        ):
            raise TypeError("resident driver trace evidence differs")
        receipt = FormalServingResidentTraceReceipt(
            schema_version=1,
            kind="formal_serving_resident_trace_receipt",
            protocol_sha256=FORMAL_SERVING_RESIDENT_PHYSICAL_PROTOCOL_SHA256,
            group_plan=self._execution.plan_binding,
            shared_launch=self._launch_binding,
            reset_boundary=reset_binding,
            member_run_plan=member.run_plan,
            group_session_binding_sha256=self._launch.group_session_binding_sha256,
            materialized_cell_id=member.materialized_cell_id,
            member_index=session_epoch - 1,
            session_epoch=session_epoch,
            process_id=self.process_id,
            process_group_id=self._driver.process_group_id,
            effective_terminal_binding=effective,
            raw_terminal=evidence.raw_terminal,
            native_itl=evidence.native_itl,
            client_lifecycle=evidence.client_lifecycle,
            junit=evidence.junit,
            trace_lifecycle=evidence.trace_lifecycle,
            trace_started_ns=evidence.trace_started_ns,
            scored_started_ns=evidence.scored_started_ns,
            trace_finished_ns=evidence.trace_finished_ns,
            formal_measured=False,
        )
        registered_plan = member.run_plan.reopen()
        if (
            type(registered_plan) is not dict
            or type(registered_plan.get("live_run_receipt_output_path")) is not str
        ):
            raise ValueError("resident member plan lacks its live receipt path")
        binding = publish_formal_serving_resident_trace_receipt(
            receipt=receipt,
            output_path=str(registered_plan["live_run_receipt_output_path"]),
        )
        self._trace_bindings.append(binding)
        return FormalServingResidentTracePhysicalResult(
            process_id=self.process_id,
            started_ns=receipt.trace_started_ns,
            finished_ns=receipt.trace_finished_ns,
            trace_receipt=binding,
        )

    async def _close(self, *, force: bool) -> CanonicalJsonProofBinding:
        if self._closed is not None:
            return self._closed
        evidence = await self._driver.close_session(force=force)
        if type(evidence) is not FormalServingResidentCloseEvidence:
            raise TypeError("resident driver close evidence differs")
        if force and evidence.cleanup_kind == "source_close_sigterm":
            raise ValueError("forced resident close claimed normal source close")
        if (
            (evidence.server_process_id, evidence.server_process_group_id)
            != (self.process_id, self._driver.process_group_id)
            or evidence.process_group_empty is not True
            or _local_process_group_exists(evidence.server_process_group_id)
        ):
            raise RuntimeError("resident process group is not proven empty")
        receipt = FormalServingResidentSharedCloseReceipt(
            schema_version=1,
            kind="formal_serving_resident_shared_close_receipt",
            protocol_sha256=FORMAL_SERVING_RESIDENT_PHYSICAL_PROTOCOL_SHA256,
            group_plan=self._execution.plan_binding,
            shared_launch=self._launch_binding,
            group_session_binding_sha256=self._launch.group_session_binding_sha256,
            source_close_receipt=evidence.source_close_receipt,
            member_trace_receipts=tuple(self._trace_bindings),
            gpu_uuid=self._launch.gpu_uuid,
            server_process_id=self.process_id,
            server_process_group_id=self._driver.process_group_id,
            server_process_started_ns=self._driver.process_started_ns,
            close_started_ns=evidence.close_started_ns,
            process_exited_ns=evidence.process_exited_ns,
            process_exit_code=evidence.process_exit_code,
            process_group_empty=evidence.process_group_empty,
            process_group_empty_checked_ns=evidence.process_group_empty_checked_ns,
            evidence_flush_completed_ns=evidence.evidence_flush_completed_ns,
            cleanup_kind=evidence.cleanup_kind,
            after_gpu_snapshot=evidence.after_gpu_snapshot,
            server_log=evidence.server_log,
            server_stdout=evidence.server_stdout,
            server_stderr=evidence.server_stderr,
            formal_measured=False,
        )
        self._closed = publish_formal_serving_resident_shared_close_receipt(
            receipt=receipt,
            output_path=_receipt_path(self._root, "shared-close.json"),
        )
        return self._closed

    async def close(self) -> CanonicalJsonProofBinding:
        return await self._close(force=False)

    async def force_close(self) -> CanonicalJsonProofBinding:
        return await self._close(force=True)

    async def finalize_resident_member(
        self,
        *,
        member: FormalServingSessionGroupSpec,
        trace: FormalServingResidentTracePhysicalResult,
        shared_close_receipt: CanonicalJsonProofBinding,
    ) -> FormalServingResidentFinalizedMemberResult:
        if self._closed != shared_close_receipt:
            raise ValueError("resident finalizer did not receive the sealed close")
        _close_binding, close = revalidate_formal_serving_resident_shared_close_receipt(
            shared_close_receipt.absolute_path
        )
        _trace_binding, trace_receipt = (
            revalidate_formal_serving_resident_trace_receipt(
                trace.trace_receipt.absolute_path
            )
        )
        if (
            trace.trace_receipt not in close.member_trace_receipts
            or trace_receipt.member_run_plan != member.run_plan
        ):
            raise ValueError("resident finalizer trace is outside shared close")
        from lightcone_spec.runtime.formal_single_operator import (
            finalize_formal_single_operator_resident_run,
        )

        manifest = finalize_formal_single_operator_resident_run(
            repository_root=self._repository_root,
            run_plan_path=member.run_plan.absolute_path,
            group_plan_path=self._execution.plan_binding.absolute_path,
            reset_authority_path=self._execution.authority_binding.absolute_path,
            shared_launch_path=self._launch_binding.absolute_path,
            reset_boundary_path=trace_receipt.reset_boundary.absolute_path,
            trace_receipt_path=trace.trace_receipt.absolute_path,
            shared_close_path=shared_close_receipt.absolute_path,
        )
        pointer = CanonicalJsonProofBinding.bind(
            Path(manifest.run_directory) / "formal-single-operator-manifest.json"
        )
        return FormalServingResidentFinalizedMemberResult(
            process_id=self.process_id,
            started_ns=trace.started_ns,
            finished_ns=trace.finished_ns,
            result_pointer=pointer,
        )


@dataclass(frozen=True)
class FormalServingResidentActiveProcessTarget:
    """One live resident server identity published before any member trace."""

    process_id: int
    process_group_id: int
    process_started_ns: int
    shared_launch: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            type(self.process_id) is not int
            or self.process_id < 1
            or type(self.process_group_id) is not int
            or self.process_group_id < 1
            or type(self.process_started_ns) is not int
            or self.process_started_ns < 1
            or type(self.shared_launch) is not CanonicalJsonProofBinding
            or CanonicalJsonProofBinding.bind(self.shared_launch.absolute_path)
            != self.shared_launch
        ):
            raise ValueError("resident active process target differs")


class FormalServingResidentPhysicalRuntime(FormalServingSessionGroupPhysicalRuntime):
    """Worker runtime whose factory owns the real server process."""

    def __init__(
        self,
        *,
        factory: FormalServingResidentProcessFactory,
        evidence_root: str | Path,
        repository_root: str | Path,
        active_target_publisher: (
            Callable[[FormalServingResidentActiveProcessTarget], None] | None
        ) = None,
    ) -> None:
        self._factory = factory
        self._evidence_root = Path(evidence_root).resolve()
        self._repository_root = Path(repository_root).resolve()
        self._active_handle: FormalServingResidentSharedSessionHandle | None = None
        if active_target_publisher is not None and not callable(
            active_target_publisher
        ):
            raise TypeError("resident active-target publisher must be callable")
        self._active_target_publisher = active_target_publisher

    async def start_shared_session(
        self, *, execution: RevalidatedFormalServingSessionGroupExecution
    ) -> FormalServingResidentSharedSessionHandle:
        group_root = self._evidence_root / execution.plan.group_id
        driver = await self._factory.launch(
            execution=execution, evidence_root=group_root
        )
        handle = FormalServingResidentSharedSessionHandle(
            execution=execution,
            driver=driver,
            evidence_root=group_root,
            repository_root=self._repository_root,
        )
        self._active_handle = handle
        publisher = self._active_target_publisher
        if publisher is not None:
            target = FormalServingResidentActiveProcessTarget(
                process_id=handle.process_id,
                process_group_id=handle.process_group_id,
                process_started_ns=handle.process_started_ns,
                shared_launch=handle.shared_launch_binding,
            )
            try:
                publisher(target)
            except BaseException as error:
                try:
                    await handle.force_close()
                except BaseException as cleanup_error:  # noqa: BLE001
                    error.add_note(
                        "resident target publication cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise
        return handle

    async def force_close_active(self) -> CanonicalJsonProofBinding | None:
        """Best-effort close for a cancelled production group worker."""

        handle = self._active_handle
        if handle is None:
            return None
        return await handle.force_close()

    async def execute_fresh_member(
        self, *, member: FormalServingSessionGroupSpec, fallback_reason: str
    ) -> FormalServingSessionMemberPhysicalResult:
        return await self._factory.execute_fresh_member(
            member=member, fallback_reason=fallback_reason
        )


__all__ = (
    "FORMAL_SERVING_RESIDENT_PHYSICAL_PROTOCOL_SHA256",
    "FormalServingResidentActiveProcessTarget",
    "FormalServingResidentCloseEvidence",
    "FormalServingResidentPhysicalRuntime",
    "FormalServingResidentProcessDriver",
    "FormalServingResidentProcessFactory",
    "FormalServingResidentResetBoundaryReceipt",
    "FormalServingResidentResetEvidence",
    "FormalServingResidentSharedCloseReceipt",
    "FormalServingResidentSharedLaunchReceipt",
    "FormalServingResidentSharedSessionHandle",
    "FormalServingResidentTraceEvidence",
    "FormalServingResidentTraceReceipt",
    "effective_formal_serving_resident_terminal_binding",
    "formal_serving_resident_session_binding_sha256",
    "publish_formal_serving_resident_reset_boundary_receipt",
    "publish_formal_serving_resident_shared_close_receipt",
    "publish_formal_serving_resident_shared_launch_receipt",
    "publish_formal_serving_resident_trace_receipt",
    "revalidate_formal_serving_resident_reset_boundary_receipt",
    "revalidate_formal_serving_resident_shared_close_receipt",
    "revalidate_formal_serving_resident_shared_launch_receipt",
    "revalidate_formal_serving_resident_trace_receipt",
)
