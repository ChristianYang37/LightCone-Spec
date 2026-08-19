"""Bounded trusted-operator executor for one TP1 serving session group.

This module consumes a path-bound group plan plus the path-bound unsigned
empirical reset authority.  It publishes one immutable artifact per scientific
cell.  If a reset boundary fails, the already completed shared-session prefix
is preserved and every unstarted member is executed with a fresh process.

The physical serving implementation is supplied through the narrow runtime
protocol below; DAG/operator wiring remains outside this module.  This module
never accepts or constructs ``VerifiedNativeRuntimeGpuProof`` and every output
remains ``formal_measured=False``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Protocol, Self

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_single_operator_session_reset import (
    TrustedEmpiricalTp1SessionResetAuthority,
    revalidate_trusted_empirical_tp1_session_reset_authority,
)
from lightcone_spec.orchestration.formal_serving_session_group import (
    FormalServingSessionGroupPlan,
    FormalServingSessionGroupSpec,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_serving_session_group_execution",
        "input": "path_bound_deep_plan_and_trusted_empirical_reset_authority",
        "bounds": "plan_member_count_and_estimated_duration",
        "publication": "one_no_replace_atomic_artifact_per_cell_attempt",
        "reset_failure": (
            "preserve_completed_prefix_force_close_failed_session_and_execute_"
            "unstarted_remainder_fresh"
        ),
        "typed_failure": (
            "infrastructure_scientific_unsafe_oom_candidate_exactness_or_"
            "failure_diagnostic"
        ),
        "scientific_failure": "retain_failed_cell_and_run_only_remainder_fresh",
        "claim": "trusted_single_operator_empirical_no_signature",
        "formal_measured": False,
    }
)

type FormalServingSessionFailureClass = Literal[
    "INFRASTRUCTURE",
    "SCIENTIFIC",
    "UNSAFE",
    "OOM_CANDIDATE",
    "EXACTNESS",
    "FAILURE_DIAGNOSTIC",
]
_FAILURE_CLASSES = frozenset(
    {
        "INFRASTRUCTURE",
        "SCIENTIFIC",
        "UNSAFE",
        "OOM_CANDIDATE",
        "EXACTNESS",
        "FAILURE_DIAGNOSTIC",
    }
)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute_path(label: str, value: object) -> Path:
    if type(value) is not str:
        raise TypeError(f"{label} must be a path string")
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or path == Path(path.anchor)
    ):
        raise ValueError(f"{label} must be an absolute normalized non-root path")
    return path


@dataclass(frozen=True)
class FormalServingSessionGroupExecutionSpec:
    """Path-only instruction for one already bounded shared-session plan."""

    schema_version: Literal[1]
    kind: Literal["formal_serving_session_group_execution_spec"]
    protocol_sha256: str
    group_plan_path: str
    reset_authority_path: str
    output_directory: str
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_session_group_execution_spec"
            or self.protocol_sha256
            != FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256
            or self.formal_measured is not False
        ):
            raise ValueError("formal serving session execution spec differs")
        paths = (
            self.group_plan_path,
            self.reset_authority_path,
            self.output_directory,
        )
        for index, value in enumerate(paths):
            _absolute_path(f"formal serving session execution path {index}", value)
        if len(set(paths)) != len(paths):
            raise ValueError("formal serving session execution paths alias")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("formal serving session execution spec fields differ")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RevalidatedFormalServingSessionGroupExecution:
    spec_binding: CanonicalJsonProofBinding
    spec: FormalServingSessionGroupExecutionSpec
    plan_binding: CanonicalJsonProofBinding
    plan: FormalServingSessionGroupPlan
    authority_binding: CanonicalJsonProofBinding
    authority: TrustedEmpiricalTp1SessionResetAuthority


def publish_formal_serving_session_group_execution_spec(
    *, spec: FormalServingSessionGroupExecutionSpec, output_path: str | Path
) -> CanonicalJsonProofBinding:
    if type(spec) is not FormalServingSessionGroupExecutionSpec:
        raise TypeError("formal serving session execution spec type differs")
    publish_canonical_json_no_replace(output_path, spec.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    if (
        binding.semantic_sha256 != spec.sha256
        or FormalServingSessionGroupExecutionSpec.from_dict(binding.reopen()) != spec
    ):
        raise RuntimeError("formal serving session execution spec changed")
    return binding


def revalidate_formal_serving_session_group_execution(
    spec_path: str | Path,
) -> RevalidatedFormalServingSessionGroupExecution:
    spec_binding = CanonicalJsonProofBinding.bind(spec_path)
    spec = FormalServingSessionGroupExecutionSpec.from_dict(spec_binding.reopen())
    if spec_binding.semantic_sha256 != spec.sha256:
        raise ValueError("formal serving session execution spec identity differs")
    plan_binding = CanonicalJsonProofBinding.bind(spec.group_plan_path)
    plan = FormalServingSessionGroupPlan.from_dict(plan_binding.reopen())
    authority_binding, authority = (
        revalidate_trusted_empirical_tp1_session_reset_authority(
            spec.reset_authority_path
        )
    )
    if (
        plan.execution_mode != "shared_session_tp1"
        or plan.reset_authority_sha256 != authority.sha256
        or len(plan.members) < 2
        or len(plan.members) > plan.max_member_count
        or sum(item.estimated_duration_seconds for item in plan.members)
        > plan.max_estimated_duration_seconds
    ):
        raise ValueError("formal serving session execution plan is not shared/bounded")
    assert plan.normalized_process_key is not None
    for member in plan.members:
        if not authority.matches(
            protocol_lock_sha256=member.protocol_lock_sha256,
            source_snapshot_sha256=member.source_snapshot_sha256,
            patched_sglang_tree=plan.normalized_process_key.patched_sglang_tree,
            inventory_sha256=member.inventory_sha256,
            gpu_uuid=member.assigned_gpu_uuids[0],
            backend=member.backend,
            method_family=member.method_family,
        ):
            raise ValueError("formal serving session member leaves reset authority")
    return RevalidatedFormalServingSessionGroupExecution(
        spec_binding=spec_binding,
        spec=spec,
        plan_binding=plan_binding,
        plan=plan,
        authority_binding=authority_binding,
        authority=authority,
    )


@dataclass(frozen=True)
class FormalServingSessionMemberPhysicalResult:
    """One raw serving result returned by the eventual physical adapter."""

    status: Literal["COMPLETE", "FAILED"]
    process_id: int
    started_ns: int
    finished_ns: int
    exit_code: int
    result_pointer: CanonicalJsonProofBinding | None
    failure_code: str | None

    def validate(self) -> None:
        if self.status not in {"COMPLETE", "FAILED"}:
            raise ValueError("formal serving member physical status differs")
        if (
            type(self.process_id) is not int
            or self.process_id < 1
            or type(self.started_ns) is not int
            or self.started_ns < 0
            or type(self.finished_ns) is not int
            or self.finished_ns < self.started_ns
            or type(self.exit_code) is not int
        ):
            raise ValueError("formal serving member physical lifecycle differs")
        if self.status == "COMPLETE":
            if (
                self.exit_code != 0
                or type(self.result_pointer) is not CanonicalJsonProofBinding
                or self.failure_code is not None
            ):
                raise ValueError("formal serving complete physical result differs")
            rebound = CanonicalJsonProofBinding.bind(self.result_pointer.absolute_path)
            if rebound != self.result_pointer:
                raise ValueError("formal serving physical result pointer changed")
        elif (
            self.exit_code == 0
            or self.result_pointer is not None
            or type(self.failure_code) is not str
            or not self.failure_code
            or "\n" in self.failure_code
        ):
            raise ValueError("formal serving failed physical result differs")


@dataclass(frozen=True)
class FormalServingResidentTracePhysicalResult:
    """One completed resident trace that is not yet a cell manifest.

    A resident trace deliberately has no process exit code or process-group
    empty claim.  The shared-session handle may turn it into the ordinary
    ``FormalServingSessionMemberPhysicalResult`` only after a bound shared
    close receipt proves that the owning process group is gone.
    """

    process_id: int
    started_ns: int
    finished_ns: int
    trace_receipt: CanonicalJsonProofBinding

    def validate(self) -> None:
        if (
            type(self.process_id) is not int
            or self.process_id < 1
            or type(self.started_ns) is not int
            or self.started_ns < 0
            or type(self.finished_ns) is not int
            or self.finished_ns < self.started_ns
            or type(self.trace_receipt) is not CanonicalJsonProofBinding
            or CanonicalJsonProofBinding.bind(self.trace_receipt.absolute_path)
            != self.trace_receipt
        ):
            raise ValueError("formal serving resident trace result differs")


@dataclass(frozen=True)
class FormalServingResidentFinalizedMemberResult:
    """A sealed resident cell with no per-cell process-exit claim."""

    process_id: int
    started_ns: int
    finished_ns: int
    result_pointer: CanonicalJsonProofBinding

    def validate(self) -> None:
        if (
            type(self.process_id) is not int
            or self.process_id < 1
            or type(self.started_ns) is not int
            or self.started_ns < 0
            or type(self.finished_ns) is not int
            or self.finished_ns < self.started_ns
            or type(self.result_pointer) is not CanonicalJsonProofBinding
            or CanonicalJsonProofBinding.bind(self.result_pointer.absolute_path)
            != self.result_pointer
        ):
            raise ValueError("formal serving finalized resident member differs")


class FormalServingSharedSessionHandle(Protocol):
    @property
    def process_id(self) -> int: ...

    async def reset_for_member(
        self,
        *,
        session_plan_sha256: str,
        reset_authority_sha256: str,
        prior_member: FormalServingSessionGroupSpec | None,
        next_member: FormalServingSessionGroupSpec,
        session_epoch: int,
    ) -> CanonicalJsonProofBinding: ...

    async def execute_member(
        self,
        *,
        member: FormalServingSessionGroupSpec,
        session_epoch: int,
    ) -> (
        FormalServingSessionMemberPhysicalResult
        | FormalServingResidentTracePhysicalResult
    ): ...

    async def close(self) -> CanonicalJsonProofBinding | None: ...

    async def force_close(self) -> CanonicalJsonProofBinding | None: ...

    async def finalize_resident_member(
        self,
        *,
        member: FormalServingSessionGroupSpec,
        trace: FormalServingResidentTracePhysicalResult,
        shared_close_receipt: CanonicalJsonProofBinding,
    ) -> FormalServingResidentFinalizedMemberResult: ...


class FormalServingSessionGroupPhysicalRuntime(Protocol):
    async def start_shared_session(
        self,
        *,
        execution: RevalidatedFormalServingSessionGroupExecution,
    ) -> FormalServingSharedSessionHandle: ...

    async def execute_fresh_member(
        self,
        *,
        member: FormalServingSessionGroupSpec,
        fallback_reason: str,
    ) -> FormalServingSessionMemberPhysicalResult: ...


class FormalServingSessionResetFailed(RuntimeError):
    """Reset failure carrying an optional already-published raw receipt."""

    def __init__(
        self,
        message: str,
        *,
        evidence: CanonicalJsonProofBinding | None = None,
    ) -> None:
        self.evidence = evidence
        super().__init__(message)


class FormalServingSessionClassifiedFailure(RuntimeError):
    """Physical failure whose retry semantics are explicit and immutable."""

    def __init__(
        self,
        failure_class: FormalServingSessionFailureClass,
        failure_code: str,
    ) -> None:
        if failure_class not in _FAILURE_CLASSES:
            raise ValueError("formal serving classified failure class differs")
        if type(failure_code) is not str or not failure_code or "\n" in failure_code:
            raise ValueError("formal serving classified failure code differs")
        self.failure_class = failure_class
        self.failure_code = failure_code
        super().__init__(failure_code)


def formal_serving_session_failure_class(
    error: BaseException,
) -> FormalServingSessionFailureClass:
    """Classify only known infrastructure failures automatically.

    Scientific/unsafe/exactness producers should raise the typed exception.
    Unknown exceptions are retained as scientific evidence rather than being
    retried blindly.  OS, timeout, connection, server-process and transport
    failures are safe to retry through the fresh-process path.
    """

    declared = getattr(error, "failure_class", None)
    if declared in _FAILURE_CLASSES:
        return declared  # type: ignore[return-value]
    if isinstance(error, (OSError, ConnectionError, TimeoutError)):
        return "INFRASTRUCTURE"
    class_name = type(error).__name__.lower()
    if "transport" in class_name or "serverprocess" in class_name:
        return "INFRASTRUCTURE"
    if class_name == "pinnedsglangservingrunerror":
        reason = str(getattr(error, "reason_code", "")).lower()
        if "exact" in reason:
            return "EXACTNESS"
        if "unsafe" in reason or "nonfinite" in reason:
            return "UNSAFE"
        if "oom" in reason:
            return "OOM_CANDIDATE"
        return "INFRASTRUCTURE"
    return "SCIENTIFIC"


def _failure_class_from_code(value: str | None) -> FormalServingSessionFailureClass:
    if type(value) is str:
        prefix = value.split(":", 1)[0]
        if prefix in _FAILURE_CLASSES:
            return prefix  # type: ignore[return-value]
    return "SCIENTIFIC"


def _failure_code_from_error(error: BaseException, *, prefix: str) -> str:
    declared = getattr(error, "failure_code", None)
    if type(declared) is str and declared and "\n" not in declared:
        return declared
    return f"{prefix}:{type(error).__name__}"


@dataclass(frozen=True)
class FormalServingSessionGroupCellArtifact:
    schema_version: Literal[1]
    kind: Literal["formal_serving_session_group_cell_artifact"]
    protocol_sha256: str
    group_id: str
    materialized_cell_id: str
    attempt: int
    member_index: int
    execution_mode: Literal["shared_session_tp1", "fresh_process_fallback", "failed"]
    status: Literal["COMPLETE", "FAILED"]
    process_id: int | None
    session_epoch: int | None
    reset_boundary: CanonicalJsonProofBinding | None
    result_pointer: CanonicalJsonProofBinding | None
    fallback_reason: str | None
    failure_class: FormalServingSessionFailureClass | None
    failure_code: str | None
    started_ns: int | None
    finished_ns: int | None
    exit_code: int | None
    evidence_level: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_session_group_cell_artifact"
            or self.protocol_sha256
            != FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256
            or self.execution_mode
            not in {"shared_session_tp1", "fresh_process_fallback", "failed"}
            or self.status not in {"COMPLETE", "FAILED"}
            or self.evidence_level != "trusted_single_operator_empirical_no_signature"
            or self.formal_measured is not False
        ):
            raise ValueError("formal serving session cell artifact differs")
        for label, value in (
            ("group", self.group_id),
            ("cell", self.materialized_cell_id),
        ):
            _require_sha256(f"formal serving session cell {label}", value)
        if (
            type(self.attempt) is not int
            or self.attempt < 1
            or type(self.member_index) is not int
            or self.member_index < 0
        ):
            raise ValueError("formal serving session cell position differs")
        if self.status == "COMPLETE":
            if (
                type(self.process_id) is not int
                or self.process_id < 1
                or type(self.result_pointer) is not CanonicalJsonProofBinding
                or self.failure_class is not None
                or self.failure_code is not None
            ):
                raise ValueError("formal serving COMPLETE cell artifact differs")
            if (
                self.execution_mode == "shared_session_tp1"
                and self.exit_code is not None
            ) or (
                self.execution_mode == "fresh_process_fallback" and self.exit_code != 0
            ):
                raise ValueError("formal serving COMPLETE exit ownership differs")
        elif (
            self.execution_mode != "failed"
            or self.result_pointer is not None
            or self.failure_class not in _FAILURE_CLASSES
            or type(self.failure_code) is not str
            or not self.failure_code
            or self.exit_code == 0
        ):
            raise ValueError("formal serving FAILED cell artifact differs")
        if self.execution_mode == "shared_session_tp1":
            if (
                type(self.reset_boundary) is not CanonicalJsonProofBinding
                or type(self.session_epoch) is not int
                or self.session_epoch < 1
                or self.fallback_reason is not None
            ):
                raise ValueError("formal serving shared cell reset binding differs")
        elif self.reset_boundary is not None or self.session_epoch is not None:
            raise ValueError("formal serving non-shared cell carries session state")
        if self.execution_mode == "fresh_process_fallback" and (
            type(self.fallback_reason) is not str or not self.fallback_reason
        ):
            raise ValueError("formal serving fresh cell lacks fallback reason")
        if self.execution_mode == "failed" and self.fallback_reason is not None:
            raise ValueError("formal serving failed cell has ambiguous fallback")
        lifecycle = (self.started_ns, self.finished_ns)
        if self.status == "COMPLETE" and (
            type(lifecycle[0]) is not int
            or type(lifecycle[1]) is not int
            or lifecycle[1] < lifecycle[0]
        ):
            raise ValueError("formal serving cell lifecycle differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = asdict(self)
        value["reset_boundary"] = (
            None if self.reset_boundary is None else self.reset_boundary.to_dict()
        )
        value["result_pointer"] = (
            None if self.result_pointer is None else self.result_pointer.to_dict()
        )
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "artifact_sha256",
        }:
            raise ValueError("formal serving session cell artifact fields differ")
        row = dict(value)
        declared = _require_sha256(
            "formal serving session cell artifact", row.pop("artifact_sha256")
        )
        for name in ("reset_boundary", "result_pointer"):
            if row[name] is not None:
                row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        result = cls(**row)  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("formal serving session cell artifact digest differs")
        return result


@dataclass(frozen=True)
class FormalServingSessionGroupExecutionResult:
    schema_version: Literal[1]
    kind: Literal["formal_serving_session_group_execution_result"]
    protocol_sha256: str
    group_id: str
    status: Literal["COMPLETE", "PARTIAL"]
    execution_spec: CanonicalJsonProofBinding
    group_plan: CanonicalJsonProofBinding
    reset_authority: CanonicalJsonProofBinding
    cell_artifacts: tuple[CanonicalJsonProofBinding, ...]
    shared_completed: int
    fresh_fallback_completed: int
    failed: int
    fallback_reason: str | None
    fallback_evidence: CanonicalJsonProofBinding | None
    evidence_level: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_session_group_execution_result"
            or self.protocol_sha256
            != FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256
            or self.status not in {"COMPLETE", "PARTIAL"}
            or self.evidence_level != "trusted_single_operator_empirical_no_signature"
            or self.formal_measured is not False
        ):
            raise ValueError("formal serving session group result differs")
        _require_sha256("formal serving session result group", self.group_id)
        if any(
            type(item) is not CanonicalJsonProofBinding
            for item in (
                self.execution_spec,
                self.group_plan,
                self.reset_authority,
                *self.cell_artifacts,
            )
        ):
            raise TypeError("formal serving session result bindings differ")
        counts = (
            self.shared_completed,
            self.fresh_fallback_completed,
            self.failed,
        )
        if any(type(item) is not int or item < 0 for item in counts) or sum(
            counts
        ) != len(self.cell_artifacts):
            raise ValueError("formal serving session result coverage differs")
        if self.status != ("COMPLETE" if self.failed == 0 else "PARTIAL"):
            raise ValueError("formal serving session result status/count differs")
        if (self.fallback_reason is None) != (self.fallback_evidence is None):
            raise ValueError("formal serving session fallback evidence is ambiguous")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "execution_spec": self.execution_spec.to_dict(),
            "group_plan": self.group_plan.to_dict(),
            "reset_authority": self.reset_authority.to_dict(),
            "cell_artifacts": [item.to_dict() for item in self.cell_artifacts],
            "fallback_evidence": (
                None
                if self.fallback_evidence is None
                else self.fallback_evidence.to_dict()
            ),
        }
        if include_sha256:
            value["result_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "result_sha256",
        }:
            raise ValueError("formal serving session group result fields differ")
        row = dict(value)
        declared = _require_sha256(
            "formal serving session group result", row.pop("result_sha256")
        )
        rows = row.pop("cell_artifacts")
        if type(rows) is not list:
            raise TypeError("formal serving session cell bindings must be an array")
        for name in ("execution_spec", "group_plan", "reset_authority"):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        fallback = row.pop("fallback_evidence")
        result = cls(
            **row,
            cell_artifacts=tuple(
                CanonicalJsonProofBinding.from_dict(item) for item in rows
            ),
            fallback_evidence=(
                None
                if fallback is None
                else CanonicalJsonProofBinding.from_dict(fallback)
            ),
        )  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("formal serving session group result digest differs")
        return result


def _validate_reset_boundary(
    binding: CanonicalJsonProofBinding,
    *,
    execution: RevalidatedFormalServingSessionGroupExecution,
    process_id: int,
    prior_member: FormalServingSessionGroupSpec | None,
    next_member: FormalServingSessionGroupSpec,
    session_epoch: int,
) -> None:
    rebound = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if rebound != binding:
        raise ValueError("formal serving reset boundary changed")
    value = binding.reopen()
    if type(value) is dict and value.get("kind") == (
        "formal_serving_resident_reset_boundary_receipt"
    ):
        from lightcone_spec.orchestration.formal_serving_session_group_physical import (
            revalidate_formal_serving_resident_reset_boundary_receipt,
        )

        _receipt_binding, receipt = (
            revalidate_formal_serving_resident_reset_boundary_receipt(
                binding.absolute_path
            )
        )
        if (
            receipt.group_plan != execution.plan_binding
            or receipt.reset_authority_sha256 != execution.authority.sha256
            or receipt.process_id != process_id
            or receipt.session_epoch != session_epoch
            or receipt.prior_materialized_cell_id
            != (None if prior_member is None else prior_member.materialized_cell_id)
            or receipt.next_materialized_cell_id != next_member.materialized_cell_id
            or not receipt.all_reset_complete
            or not receipt.request_queue_empty
            or not receipt.adaptation_state_reset
            or not receipt.candidate_state_reset
            or not receipt.cache_policy_restored
            or not receipt.terminal_writer_flushed
        ):
            raise ValueError("formal serving resident reset boundary did not pass")
        return
    if type(value) is not dict or set(value) != {
        "schema_version",
        "kind",
        "session_plan_sha256",
        "reset_authority_sha256",
        "process_id",
        "session_epoch",
        "prior_materialized_cell_id",
        "next_materialized_cell_id",
        "all_reset_complete",
        "request_queue_empty",
        "terminal_writer_flushed",
    }:
        raise ValueError("formal serving reset boundary fields differ")
    if value != {
        "schema_version": 1,
        "kind": "formal_serving_session_reset_boundary",
        "session_plan_sha256": execution.plan.session_plan_sha256,
        "reset_authority_sha256": execution.authority.sha256,
        "process_id": process_id,
        "session_epoch": session_epoch,
        "prior_materialized_cell_id": (
            None if prior_member is None else prior_member.materialized_cell_id
        ),
        "next_materialized_cell_id": next_member.materialized_cell_id,
        "all_reset_complete": True,
        "request_queue_empty": True,
        "terminal_writer_flushed": True,
    }:
        raise ValueError("formal serving reset boundary did not pass")


def _publish_fallback_event(
    *,
    output: Path,
    execution: RevalidatedFormalServingSessionGroupExecution,
    failed_member_index: int,
    reason: str,
    completed_prefix_count: int,
    external_evidence: CanonicalJsonProofBinding | None,
) -> CanonicalJsonProofBinding:
    event_path = output / "fallback-event.json"
    publish_canonical_json_no_replace(
        event_path,
        {
            "schema_version": 1,
            "kind": "formal_serving_session_group_fallback_event",
            "protocol_sha256": (FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256),
            "group_id": execution.plan.group_id,
            "failed_member_index": failed_member_index,
            "failed_materialized_cell_id": (
                execution.plan.members[failed_member_index].materialized_cell_id
            ),
            "reason": reason,
            "external_evidence": (
                None if external_evidence is None else external_evidence.to_dict()
            ),
            "completed_prefix_count": completed_prefix_count,
            "formal_measured": False,
        },
    )
    return CanonicalJsonProofBinding.bind(event_path)


def _cell_artifact_path(output: Path, member: FormalServingSessionGroupSpec) -> Path:
    return output / (
        f"cell-{member.materialized_cell_id}-attempt-{member.attempt}.json"
    )


def _publish_cell_artifact(
    path: Path, value: FormalServingSessionGroupCellArtifact
) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path, value.to_dict())
    binding = CanonicalJsonProofBinding.bind(path)
    if FormalServingSessionGroupCellArtifact.from_dict(binding.reopen()) != value:
        raise RuntimeError("formal serving session cell artifact changed")
    return binding


def _artifact_from_physical(
    *,
    execution: RevalidatedFormalServingSessionGroupExecution,
    member: FormalServingSessionGroupSpec,
    index: int,
    mode: Literal["shared_session_tp1", "fresh_process_fallback"],
    physical: (
        FormalServingSessionMemberPhysicalResult
        | FormalServingResidentFinalizedMemberResult
    ),
    reset_boundary: CanonicalJsonProofBinding | None,
    session_epoch: int | None,
    fallback_reason: str | None,
) -> FormalServingSessionGroupCellArtifact:
    physical.validate()
    resident = type(physical) is FormalServingResidentFinalizedMemberResult
    status: Literal["COMPLETE", "FAILED"] = "COMPLETE" if resident else physical.status
    return FormalServingSessionGroupCellArtifact(
        schema_version=1,
        kind="formal_serving_session_group_cell_artifact",
        protocol_sha256=FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
        group_id=execution.plan.group_id,
        materialized_cell_id=member.materialized_cell_id,
        attempt=member.attempt,
        member_index=index,
        execution_mode=(mode if status == "COMPLETE" else "failed"),
        status=status,
        process_id=physical.process_id,
        session_epoch=(
            session_epoch
            if status == "COMPLETE" and mode == "shared_session_tp1"
            else None
        ),
        reset_boundary=(
            reset_boundary
            if status == "COMPLETE" and mode == "shared_session_tp1"
            else None
        ),
        result_pointer=physical.result_pointer,
        fallback_reason=(
            fallback_reason
            if status == "COMPLETE" and mode == "fresh_process_fallback"
            else None
        ),
        failure_code=(None if resident else physical.failure_code),
        failure_class=(
            None
            if status == "COMPLETE"
            else _failure_class_from_code(physical.failure_code)
        ),
        started_ns=physical.started_ns,
        finished_ns=physical.finished_ns,
        exit_code=(None if mode == "shared_session_tp1" else physical.exit_code),
        evidence_level="trusted_single_operator_empirical_no_signature",
        formal_measured=False,
    )


def _failed_artifact(
    *,
    execution: RevalidatedFormalServingSessionGroupExecution,
    member: FormalServingSessionGroupSpec,
    index: int,
    failure_class: FormalServingSessionFailureClass,
    failure_code: str,
) -> FormalServingSessionGroupCellArtifact:
    return FormalServingSessionGroupCellArtifact(
        schema_version=1,
        kind="formal_serving_session_group_cell_artifact",
        protocol_sha256=FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
        group_id=execution.plan.group_id,
        materialized_cell_id=member.materialized_cell_id,
        attempt=member.attempt,
        member_index=index,
        execution_mode="failed",
        status="FAILED",
        process_id=None,
        session_epoch=None,
        reset_boundary=None,
        result_pointer=None,
        fallback_reason=None,
        failure_class=failure_class,
        failure_code=failure_code,
        started_ns=None,
        finished_ns=None,
        exit_code=-1,
        evidence_level="trusted_single_operator_empirical_no_signature",
        formal_measured=False,
    )


async def execute_formal_serving_session_group(
    *,
    execution_spec_path: str | Path,
    runtime: FormalServingSessionGroupPhysicalRuntime,
) -> FormalServingSessionGroupExecutionResult:
    """Run a bounded shared group and seal cells only after session close.

    Legacy physical adapters may still return an already-finalized ordinary
    member result.  Resident adapters return a trace-only result; those traces
    stay unpublished at the cell layer until ``close``/``force_close`` returns
    one immutable receipt and ``finalize_resident_member`` derives the actual
    per-cell manifest from it.
    """

    execution = revalidate_formal_serving_session_group_execution(execution_spec_path)
    output = Path(execution.spec.output_directory)
    if output.exists():
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(
                "formal serving session output must be absent or empty"
            )
    else:
        output.mkdir(parents=False, mode=0o700)
    members = execution.plan.members
    artifacts_by_index: dict[int, CanonicalJsonProofBinding] = {}
    pending_shared: list[
        tuple[
            int,
            FormalServingSessionGroupSpec,
            int,
            CanonicalJsonProofBinding,
            FormalServingSessionMemberPhysicalResult
            | FormalServingResidentTracePhysicalResult,
        ]
    ] = []
    shared_completed = 0
    fresh_completed = 0
    failed = 0
    fallback_reason: str | None = None
    fallback_evidence: CanonicalJsonProofBinding | None = None
    handle: FormalServingSharedSessionHandle | None = None
    shared_close_receipt: CanonicalJsonProofBinding | None = None
    remainder_start = len(members)
    shared_failed: tuple[int, FormalServingSessionGroupCellArtifact] | None = None

    try:
        handle = await runtime.start_shared_session(execution=execution)
        process_id = handle.process_id
        if type(process_id) is not int or process_id < 1:
            raise ValueError("formal serving shared process PID is invalid")
    except Exception as error:  # noqa: BLE001 - start failure selects fresh
        cleanup_suffix = ""
        if handle is not None:
            try:
                await handle.force_close()
            except Exception as cleanup_error:  # noqa: BLE001 - retain both
                cleanup_suffix = (
                    ";force_close_failed:"
                    f"{type(cleanup_error).__name__}:{cleanup_error}"
                )
        fallback_reason = (
            f"session_start_failed:{type(error).__name__}:{error}{cleanup_suffix}"
        )
        fallback_evidence = _publish_fallback_event(
            output=output,
            execution=execution,
            failed_member_index=0,
            reason=fallback_reason,
            completed_prefix_count=0,
            external_evidence=None,
        )
        remainder_start = 0
    else:
        prior: FormalServingSessionGroupSpec | None = None
        for index, member in enumerate(members):
            epoch = index + 1
            try:
                reset = await handle.reset_for_member(
                    session_plan_sha256=execution.plan.session_plan_sha256,
                    reset_authority_sha256=execution.authority.sha256,
                    prior_member=prior,
                    next_member=member,
                    session_epoch=epoch,
                )
                if type(reset) is not CanonicalJsonProofBinding:
                    raise TypeError("formal serving reset did not return a binding")
                _validate_reset_boundary(
                    reset,
                    execution=execution,
                    process_id=process_id,
                    prior_member=prior,
                    next_member=member,
                    session_epoch=epoch,
                )
            except Exception as error:  # noqa: BLE001 - reset selects fresh
                fallback_reason = f"session_reset_failed:{type(error).__name__}:{error}"
                external = (
                    error.evidence
                    if isinstance(error, FormalServingSessionResetFailed)
                    else None
                )
                if (
                    external is not None
                    and CanonicalJsonProofBinding.bind(external.absolute_path)
                    != external
                ):
                    raise ValueError("formal serving reset failure evidence changed")
                fallback_evidence = _publish_fallback_event(
                    output=output,
                    execution=execution,
                    failed_member_index=index,
                    reason=fallback_reason,
                    completed_prefix_count=len(pending_shared),
                    external_evidence=external,
                )
                try:
                    shared_close_receipt = await handle.force_close()
                except Exception as close_error:  # noqa: BLE001 - fail closed
                    fallback_reason += (
                        ";force_close_failed:"
                        f"{type(close_error).__name__}:{close_error}"
                    )
                remainder_start = index
                break
            try:
                physical = await handle.execute_member(
                    member=member,
                    session_epoch=epoch,
                )
                if type(physical) not in {
                    FormalServingSessionMemberPhysicalResult,
                    FormalServingResidentTracePhysicalResult,
                }:
                    raise TypeError("formal serving shared member result type differs")
                physical.validate()
                if physical.process_id != process_id:
                    raise ValueError(
                        "formal serving shared member changed server process"
                    )
                if (
                    type(physical) is FormalServingSessionMemberPhysicalResult
                    and physical.status != "COMPLETE"
                ):
                    failure_code = (
                        physical.failure_code
                        or "SCIENTIFIC:shared_member_failed_without_code"
                    )
                    raise FormalServingSessionClassifiedFailure(
                        _failure_class_from_code(failure_code),
                        failure_code,
                    )
                pending_shared.append((index, member, epoch, reset, physical))
                prior = member
                continue
            except Exception as error:  # noqa: BLE001 - retain cell failure
                shared_failed = (
                    index,
                    _failed_artifact(
                        execution=execution,
                        member=member,
                        index=index,
                        failure_class=formal_serving_session_failure_class(error),
                        failure_code=_failure_code_from_error(
                            error, prefix="shared_member_failed"
                        ),
                    ),
                )
                fallback_reason = "shared_member_failed_remainder_fresh"
                fallback_evidence = _publish_fallback_event(
                    output=output,
                    execution=execution,
                    failed_member_index=index,
                    reason=fallback_reason,
                    completed_prefix_count=len(pending_shared),
                    external_evidence=None,
                )
                try:
                    shared_close_receipt = await handle.force_close()
                except Exception as close_error:  # noqa: BLE001 - fail closed
                    fallback_reason += (
                        ";force_close_failed:"
                        f"{type(close_error).__name__}:{close_error}"
                    )
                remainder_start = index + 1
                break
        else:
            try:
                shared_close_receipt = await handle.close()
            except Exception as close_error:  # noqa: BLE001 - close then force
                close_reason = (
                    f"shared_session_close_failed:{type(close_error).__name__}:"
                    f"{close_error}"
                )
                try:
                    shared_close_receipt = await handle.force_close()
                except Exception as force_error:  # noqa: BLE001 - retain both
                    close_reason += (
                        ";force_close_failed:"
                        f"{type(force_error).__name__}:{force_error}"
                    )
                fallback_reason = close_reason
                fallback_evidence = _publish_fallback_event(
                    output=output,
                    execution=execution,
                    failed_member_index=max(len(members) - 1, 0),
                    reason=close_reason,
                    completed_prefix_count=len(pending_shared),
                    external_evidence=shared_close_receipt,
                )

        for index, member, epoch, reset, pending in pending_shared:
            try:
                if type(pending) is FormalServingResidentTracePhysicalResult:
                    if type(shared_close_receipt) is not CanonicalJsonProofBinding:
                        raise RuntimeError("resident_trace_lacks_shared_close_receipt")
                    physical = await handle.finalize_resident_member(
                        member=member,
                        trace=pending,
                        shared_close_receipt=shared_close_receipt,
                    )
                else:
                    physical = pending
                artifact = _artifact_from_physical(
                    execution=execution,
                    member=member,
                    index=index,
                    mode="shared_session_tp1",
                    physical=physical,
                    reset_boundary=reset,
                    session_epoch=epoch,
                    fallback_reason=None,
                )
            except Exception as error:  # noqa: BLE001 - no duplicate trace
                artifact = _failed_artifact(
                    execution=execution,
                    member=member,
                    index=index,
                    failure_class=formal_serving_session_failure_class(error),
                    failure_code=_failure_code_from_error(
                        error, prefix="resident_member_finalize_failed"
                    ),
                )
            artifacts_by_index[index] = _publish_cell_artifact(
                _cell_artifact_path(output, member), artifact
            )
            if artifact.status == "COMPLETE":
                shared_completed += 1
            else:
                failed += 1

        if shared_failed is not None:
            index, artifact = shared_failed
            member = members[index]
            artifacts_by_index[index] = _publish_cell_artifact(
                _cell_artifact_path(output, member), artifact
            )
            failed += 1

    if remainder_start < len(members):
        assert fallback_reason is not None
        for index in range(remainder_start, len(members)):
            member = members[index]
            try:
                physical = await runtime.execute_fresh_member(
                    member=member,
                    fallback_reason=fallback_reason,
                )
                artifact = _artifact_from_physical(
                    execution=execution,
                    member=member,
                    index=index,
                    mode="fresh_process_fallback",
                    physical=physical,
                    reset_boundary=None,
                    session_epoch=None,
                    fallback_reason=fallback_reason,
                )
            except Exception as error:  # noqa: BLE001 - retain fresh failure
                artifact = _failed_artifact(
                    execution=execution,
                    member=member,
                    index=index,
                    failure_class=formal_serving_session_failure_class(error),
                    failure_code=_failure_code_from_error(
                        error, prefix="fresh_fallback_failed"
                    ),
                )
            if index in artifacts_by_index:
                raise RuntimeError("formal serving session attempted a duplicate cell")
            artifacts_by_index[index] = _publish_cell_artifact(
                _cell_artifact_path(output, member), artifact
            )
            if artifact.status == "COMPLETE":
                fresh_completed += 1
            else:
                failed += 1
    if set(artifacts_by_index) != set(range(len(members))):
        raise RuntimeError("formal serving session execution lost cell coverage")
    artifacts = [artifacts_by_index[index] for index in range(len(members))]
    result = FormalServingSessionGroupExecutionResult(
        schema_version=1,
        kind="formal_serving_session_group_execution_result",
        protocol_sha256=FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
        group_id=execution.plan.group_id,
        status=("COMPLETE" if failed == 0 else "PARTIAL"),
        execution_spec=execution.spec_binding,
        group_plan=execution.plan_binding,
        reset_authority=execution.authority_binding,
        cell_artifacts=tuple(artifacts),
        shared_completed=shared_completed,
        fresh_fallback_completed=fresh_completed,
        failed=failed,
        fallback_reason=fallback_reason,
        fallback_evidence=fallback_evidence,
        evidence_level="trusted_single_operator_empirical_no_signature",
        formal_measured=False,
    )
    result_path = output / "result.json"
    publish_canonical_json_no_replace(result_path, result.to_dict())
    rebound = FormalServingSessionGroupExecutionResult.from_dict(
        CanonicalJsonProofBinding.bind(result_path).reopen()
    )
    if rebound != result:
        raise RuntimeError("formal serving session group result changed")
    return result


__all__ = (
    "FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256",
    "FormalServingResidentFinalizedMemberResult",
    "FormalServingResidentTracePhysicalResult",
    "FormalServingSessionClassifiedFailure",
    "FormalServingSessionFailureClass",
    "FormalServingSessionGroupCellArtifact",
    "FormalServingSessionGroupExecutionResult",
    "FormalServingSessionGroupExecutionSpec",
    "FormalServingSessionGroupPhysicalRuntime",
    "FormalServingSessionMemberPhysicalResult",
    "FormalServingSessionResetFailed",
    "FormalServingSharedSessionHandle",
    "RevalidatedFormalServingSessionGroupExecution",
    "execute_formal_serving_session_group",
    "formal_serving_session_failure_class",
    "publish_formal_serving_session_group_execution_spec",
    "revalidate_formal_serving_session_group_execution",
)
