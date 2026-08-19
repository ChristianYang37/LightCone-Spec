"""Conjunctive terminal authority for the formal preflight stage.

Activation answers whether a cell may be dispatched.  It is not evidence that
the cell reached a terminal state.  This module deliberately keeps those two
facts separate and reduces an exact, registry-derived terminal row for every
mandatory preflight cell.  Missing ranks, failures, errors, and skips all keep
the aggregate ``BLOCKED``; no successful subset can promote the stage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from lightcone_spec.experiments.preflight_interference import (
    validate_formal_preflight_interference_proof_artifact,
)
from lightcone_spec.experiments.registry import (
    CellStatus,
    ExperimentCell,
    ExperimentRegistry,
    WorkloadClass,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.stage_activation import (
    RegistryStageActivationArtifact,
    RegistryStageDispositionStatus,
    materialize_pointer_preflight_stage_activation,
    verify_pointer_preflight_stage_activation,
    verify_registry_stage_activation,
)
from lightcone_spec.runtime.compile_runner import (
    CompileAssignmentPlan,
    CompileControlVerificationReceipt,
    CompileResultPointer,
)
from lightcone_spec.runtime.distributed import (
    validate_distributed_runtime_gpu_proof_artifact,
)
from lightcone_spec.runtime.native_qualification_runner import (
    NativeRuntimeQualificationAssignment,
    NativeRuntimeQualificationObservation,
    NativeRuntimeQualificationResultPointer,
)
from lightcone_spec.runtime.preflight_runner import (
    ExactnessPreflightAssignment,
    ExactnessPreflightResultPointer,
    ExactnessPreflightTerminal,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding
from lightcone_spec.runtime.readiness import (
    NativeRuntimeGpuProofArtifact,
    validate_native_runtime_gpu_proof_artifact,
)
from lightcone_spec.runtime.session_reset_runner import (
    SessionResetQualificationAssignment,
    SessionResetQualificationProofPointer,
    SessionResetQualificationResultPointer,
)

if TYPE_CHECKING:
    from lightcone_spec.experiments.formal_protocol import (
        TtsL0CandidateStateCoverage,
    )
    from lightcone_spec.experiments.stage_materialization import (
        StageCoverageReceipt,
        StageMaterializationReceipt,
    )
    from lightcone_spec.orchestration.execution_bundle import BoundJsonSource


def _bound_json_source_type() -> type[BoundJsonSource]:
    """Load the execution-bundle codec without forming a package import cycle.

    ``orchestration.__init__`` imports the serving executor, which imports the
    experiments package.  Eagerly importing ``execution_bundle`` while that
    package is still initializing therefore makes the formal preflight module
    depend on a partially initialized executor.  Source bindings are only
    needed while parsing or materializing evidence, so the import belongs at
    that narrow runtime boundary.
    """

    from lightcone_spec.orchestration.execution_bundle import BoundJsonSource

    return BoundJsonSource


PreflightTerminalKind = Literal["compile", "exactness", "interference"]
PreflightTerminalStatus = Literal["PASSED", "FAILED", "ERROR", "SKIPPED"]

PREFLIGHT_REQUIRED_QUALIFICATION_SUITES = (
    "chronobelief_gpu_parity",
    "dspark_tp1",
    "dspark_tp2",
    "dspark_dp2",
    "native_hot_path_tp1",
    "nextn_tp1",
    "nextn_tp2",
    "session_reset_tp1",
    "tp1_dp2",
    "tp2_dp1",
)
_PREFLIGHT_NATIVE_QUALIFICATION_SUITES = frozenset(
    {
        "chronobelief_gpu_parity",
        "dspark_tp1",
        "dspark_tp2",
        "dspark_dp2",
        "native_hot_path_tp1",
        "nextn_tp1",
        "nextn_tp2",
    }
)
_PREFLIGHT_DISTRIBUTED_QUALIFICATION_SUITES = frozenset({"tp1_dp2", "tp2_dp1"})

PREFLIGHT_COVERAGE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_preflight_terminal_coverage_reducer",
        "activation": "exact_verified_registry_stage_activation",
        "mandatory": "every_non_na_preflight_cell",
        "terminal_kinds": ("compile", "exactness", "interference"),
        "rank_coverage": "every_expected_rank_terminal",
        "complete": "every_cell_passed_zero_failure_error_skip",
        "partial_success": "BLOCKED",
        "serialized_summary_is_not_authority": True,
    }
)
PREFLIGHT_POINTER_COVERAGE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 4,
        "kind": "formal_preflight_pointer_terminal_coverage_reducer",
        "compile": "deep_reopened_formal_schema3_compile_result_pointer",
        "exactness": (
            "deep_reopened_formal_schema4_qualified_exactness_pointer_junit_"
            "two_rank_native_terminals_rank_aggregate_control_and_reservation_"
            "plus_exact_eight_suite_specific_native_distributed_session_proofs"
        ),
        "interference": (
            "deep_reopened_dynamic_eight_row_native_result_plus_native_itl_proof_"
            "aggregate_with_local_interference_control_atomic_reservation_slo_"
            "qualified_goodput_and_paired_pass_reduction"
        ),
        "activation": "specialized_pointer_stage_activation",
        "coverage": "exact_ten_registry_cells_zero_skip",
        "serialized_summary_is_not_authority": True,
    }
)
PREFLIGHT_STAGE_COVERAGE_BRIDGE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_preflight_pointer_to_stage_coverage_bridge",
        "materialization": "exact_protocol_lock_derived_ten_cell_preflight",
        "terminal_mapping": "registry_cell_id_dimension_to_exact_pointer_terminal",
        "candidate_state": (
            "exact_tts_l0_candidate_coverage_plus_two_deep_reopened_external_"
            "control_replay_proofs"
        ),
        "complete": "ten_complete_zero_failure_error_skip",
        "activation": "pointer_activation_sha256_bound_into_each_terminal_receipt",
        "caller_dispositions_forbidden": True,
    }
)


class PreflightCoverageBlocked(RuntimeError):
    """Raised when a terminal aggregate cannot authorize formal work."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"formal preflight is BLOCKED: {reason_code}")


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_nonnegative(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _strict_object(
    label: str,
    value: object,
    fields: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a JSON object with string keys")
    if set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return value


def _preflight_terminal_kind(cell: ExperimentCell) -> PreflightTerminalKind:
    if cell.resources.workload_class is WorkloadClass.COMPILE:
        return "compile"
    if cell.identity.task == "exactness_memory_telemetry_preflight":
        return "exactness"
    if cell.identity.task == "simultaneous_single_gpu_interference":
        return "interference"
    raise ValueError("mandatory preflight cell has no registered terminal kind")


def preflight_coverage_control_lineage_sha256(
    *,
    activation_sha256: str,
    runtime_sha256: str,
    split_sha256: str,
    inventory_sha256: str,
    raw_completed_cells_sha256: str,
) -> str:
    """Canonical control-subject lineage for the all-cell rank aggregate."""

    for label, digest in (
        ("activation", activation_sha256),
        ("runtime", runtime_sha256),
        ("split", split_sha256),
        ("inventory", inventory_sha256),
        ("raw completed cells", raw_completed_cells_sha256),
    ):
        _require_sha256(f"preflight control {label}", digest)
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_preflight_coverage_control_lineage",
            "activation_sha256": activation_sha256,
            "runtime_sha256": runtime_sha256,
            "split_sha256": split_sha256,
            "inventory_sha256": inventory_sha256,
            "raw_completed_cells_sha256": raw_completed_cells_sha256,
        }
    )


@dataclass(frozen=True)
class PreflightCellTerminal:
    """Content identity and outcome of one mandatory preflight assignment."""

    cell_id: str
    terminal_kind: PreflightTerminalKind
    terminal_authority_sha256: str
    status: PreflightTerminalStatus
    expected_rank_count: int
    terminal_rank_count: int
    failure_count: int
    error_count: int
    skip_count: int

    def __post_init__(self) -> None:
        _require_sha256("preflight terminal cell", self.cell_id)
        _require_sha256("preflight terminal authority", self.terminal_authority_sha256)
        if self.terminal_kind not in {"compile", "exactness", "interference"}:
            raise ValueError("preflight terminal kind is unsupported")
        if self.status not in {"PASSED", "FAILED", "ERROR", "SKIPPED"}:
            raise ValueError("preflight terminal status is unsupported")
        if type(self.expected_rank_count) is not int or self.expected_rank_count < 1:
            raise ValueError("preflight expected rank count must be positive")
        if (
            type(self.terminal_rank_count) is not int
            or self.terminal_rank_count < 0
            or self.terminal_rank_count > self.expected_rank_count
        ):
            raise ValueError("preflight terminal rank count is invalid")
        for name in ("failure_count", "error_count", "skip_count"):
            _require_nonnegative(f"preflight {name}", getattr(self, name))
        counters = (self.failure_count, self.error_count, self.skip_count)
        if self.status == "PASSED" and (
            any(counters) or self.terminal_rank_count != self.expected_rank_count
        ):
            raise ValueError("passed preflight terminal is not clean and rank-complete")
        if self.status == "FAILED" and self.failure_count < 1:
            raise ValueError("failed preflight terminal lacks a failure")
        if self.status == "ERROR" and self.error_count < 1:
            raise ValueError("errored preflight terminal lacks an error")
        if self.status == "SKIPPED" and self.skip_count < 1:
            raise ValueError("skipped preflight terminal lacks a skip")

    @property
    def passed(self) -> bool:
        return self.status == "PASSED"

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "terminal_kind": self.terminal_kind,
            "terminal_authority_sha256": self.terminal_authority_sha256,
            "status": self.status,
            "expected_rank_count": self.expected_rank_count,
            "terminal_rank_count": self.terminal_rank_count,
            "failure_count": self.failure_count,
            "error_count": self.error_count,
            "skip_count": self.skip_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "preflight terminal",
            value,
            frozenset(
                {
                    "cell_id",
                    "terminal_kind",
                    "terminal_authority_sha256",
                    "status",
                    "expected_rank_count",
                    "terminal_rank_count",
                    "failure_count",
                    "error_count",
                    "skip_count",
                }
            ),
        )
        return cls(**row)


@dataclass(frozen=True)
class PreflightQualificationProofSource:
    """One suite-specific raw result plus its locally controlled proof.

    The result pointer owns the assignment, live observation, exact JUnit and
    clean GPU-boundary snapshots.  The proof pointer/artifact owns the local
    root-authorized control and replay reservation.  Keeping both path-bound
    prevents a valid proof from being relabelled as another suite or launch.
    """

    suite_id: str
    result_pointer: CanonicalJsonProofBinding
    proof_artifact: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            type(self.suite_id) is not str
            or self.suite_id not in PREFLIGHT_REQUIRED_QUALIFICATION_SUITES
        ):
            raise ValueError("preflight qualification suite is unsupported")
        for label, binding in (
            ("result pointer", self.result_pointer),
            ("proof artifact", self.proof_artifact),
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError(f"preflight qualification {label} is not path-bound")
            binding.__post_init__()

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_id": self.suite_id,
            "result_pointer": self.result_pointer.to_dict(),
            "proof_artifact": self.proof_artifact.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "preflight qualification proof source",
            value,
            frozenset({"suite_id", "result_pointer", "proof_artifact"}),
        )
        return cls(
            suite_id=row["suite_id"],
            result_pointer=CanonicalJsonProofBinding.from_dict(row["result_pointer"]),
            proof_artifact=CanonicalJsonProofBinding.from_dict(row["proof_artifact"]),
        )

    @classmethod
    def bind(
        cls,
        *,
        suite_id: str,
        result_pointer_path: str | Path,
        proof_artifact_path: str | Path,
    ) -> Self:
        return cls(
            suite_id=suite_id,
            result_pointer=CanonicalJsonProofBinding.bind(result_pointer_path),
            proof_artifact=CanonicalJsonProofBinding.bind(proof_artifact_path),
        )


@dataclass(frozen=True)
class PreflightExecutionSourceAuthority:
    """Path-bound raw authority for all ten mandatory preflight terminals."""

    schema_version: int
    kind: Literal["formal_preflight_execution_source_authority"]
    registry_sha256: str
    dispatch_activation_sha256: str
    runtime_sha256: str
    split_sha256: str
    inventory_sha256: str
    release_root_manifest_sha256: str
    compile_result: BoundJsonSource
    exactness_result: BoundJsonSource
    interference_proof_artifact: CanonicalJsonProofBinding
    qualification_proofs: tuple[PreflightQualificationProofSource, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 3 or self.kind != (
            "formal_preflight_execution_source_authority"
        ):
            raise ValueError("preflight execution source schema is unsupported")
        for label, digest in (
            ("registry", self.registry_sha256),
            ("dispatch activation", self.dispatch_activation_sha256),
            ("runtime", self.runtime_sha256),
            ("split", self.split_sha256),
            ("inventory", self.inventory_sha256),
            ("release root manifest", self.release_root_manifest_sha256),
        ):
            _require_sha256(f"preflight source {label}", digest)
        for label, source in (
            ("compile result", self.compile_result),
            ("exactness result", self.exactness_result),
        ):
            if type(source) is not _bound_json_source_type():
                raise TypeError(f"preflight {label} source binding is invalid")
        if type(self.interference_proof_artifact) is not CanonicalJsonProofBinding:
            raise TypeError("preflight interference proof binding is invalid")
        if (
            type(self.qualification_proofs) is not tuple
            or any(
                type(row) is not PreflightQualificationProofSource
                for row in self.qualification_proofs
            )
            or tuple(row.suite_id for row in self.qualification_proofs)
            != PREFLIGHT_REQUIRED_QUALIFICATION_SUITES
        ):
            raise ValueError(
                "preflight qualification proof coverage is not the exact ten core suites"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "registry_sha256": self.registry_sha256,
            "dispatch_activation_sha256": self.dispatch_activation_sha256,
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "inventory_sha256": self.inventory_sha256,
            "release_root_manifest_sha256": self.release_root_manifest_sha256,
            "compile_result": self.compile_result.to_dict(),
            "exactness_result": self.exactness_result.to_dict(),
            "interference_proof_artifact": (self.interference_proof_artifact.to_dict()),
            "qualification_proofs": [
                row.to_dict() for row in self.qualification_proofs
            ],
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        bound_json_source = _bound_json_source_type()
        row = _strict_object(
            "preflight execution source authority",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "registry_sha256",
                    "dispatch_activation_sha256",
                    "runtime_sha256",
                    "split_sha256",
                    "inventory_sha256",
                    "release_root_manifest_sha256",
                    "compile_result",
                    "exactness_result",
                    "interference_proof_artifact",
                    "qualification_proofs",
                }
            ),
        )
        raw_qualification_proofs = row.pop("qualification_proofs")
        if type(raw_qualification_proofs) is not list:
            raise TypeError("preflight qualification proofs must be an array")
        return cls(
            compile_result=bound_json_source.from_dict(row.pop("compile_result")),
            exactness_result=bound_json_source.from_dict(row.pop("exactness_result")),
            interference_proof_artifact=CanonicalJsonProofBinding.from_dict(
                row.pop("interference_proof_artifact")
            ),
            qualification_proofs=tuple(
                PreflightQualificationProofSource.from_dict(value)
                for value in raw_qualification_proofs
            ),
            **row,
        )

    @classmethod
    def bind(
        cls,
        *,
        registry: ExperimentRegistry,
        dispatch_activation_sha256: str,
        runtime_sha256: str,
        split_sha256: str,
        release_root_manifest_sha256: str,
        compile_result_path: str | Path,
        exactness_result_path: str | Path,
        interference_proof_artifact_path: str | Path,
        qualification_proof_paths: dict[str, tuple[str | Path, str | Path]],
        now_ns: int,
    ) -> Self:
        bound_json_source = _bound_json_source_type()
        if type(registry) is not ExperimentRegistry:
            raise TypeError("preflight source binding requires an exact registry")
        compile_pointer = CompileResultPointer.load(compile_result_path)
        exactness_pointer = ExactnessPreflightResultPointer.load(exactness_result_path)
        exactness_assignment = ExactnessPreflightAssignment.load(
            exactness_pointer.assignment.absolute_path
        )
        interference_binding = CanonicalJsonProofBinding.bind(
            interference_proof_artifact_path
        )
        interference = validate_formal_preflight_interference_proof_artifact(
            interference_binding.absolute_path,
            registry=registry,
            expected_activation_sha256=_require_sha256(
                "preflight dispatch activation", dispatch_activation_sha256
            ),
            expected_runtime_sha256=runtime_sha256,
            expected_split_sha256=split_sha256,
            expected_inventory_sha256=exactness_assignment.inventory_sha256,
            now_ns=now_ns,
        )
        if type(qualification_proof_paths) is not dict or set(
            qualification_proof_paths
        ) != set(PREFLIGHT_REQUIRED_QUALIFICATION_SUITES):
            raise ValueError("preflight qualification proof path coverage is not exact")
        if any(
            type(paths) is not tuple
            or len(paths) != 2
            or any(not isinstance(path, (str, Path)) for path in paths)
            for paths in qualification_proof_paths.values()
        ):
            raise TypeError("preflight qualification proof paths are malformed")
        qualification_proofs = tuple(
            PreflightQualificationProofSource.bind(
                suite_id=suite_id,
                result_pointer_path=qualification_proof_paths[suite_id][0],
                proof_artifact_path=qualification_proof_paths[suite_id][1],
            )
            for suite_id in PREFLIGHT_REQUIRED_QUALIFICATION_SUITES
        )
        source = cls(
            schema_version=3,
            kind="formal_preflight_execution_source_authority",
            registry_sha256=registry.sha256,
            dispatch_activation_sha256=dispatch_activation_sha256,
            runtime_sha256=_require_sha256("preflight source runtime", runtime_sha256),
            split_sha256=_require_sha256("preflight source split", split_sha256),
            inventory_sha256=exactness_assignment.inventory_sha256,
            release_root_manifest_sha256=_require_sha256(
                "preflight source release root", release_root_manifest_sha256
            ),
            compile_result=bound_json_source.bind(
                compile_result_path, semantic_sha256=compile_pointer.sha256
            ),
            exactness_result=bound_json_source.bind(
                exactness_result_path, semantic_sha256=exactness_pointer.sha256
            ),
            interference_proof_artifact=interference_binding,
            qualification_proofs=qualification_proofs,
        )
        if interference.artifact_sha256 != interference_binding.semantic_sha256:
            raise ValueError("preflight interference proof semantic identity differs")
        source.revalidate(registry, now_ns=now_ns)
        return source

    def _revalidate_qualification_proofs(
        self,
        *,
        exactness_pointer: ExactnessPreflightResultPointer,
        exactness_assignment: ExactnessPreflightAssignment,
        now_ns: int,
    ) -> None:
        """Deep-open the ten core suite proofs required by later formal stages."""

        for source in self.qualification_proofs:
            if (
                CanonicalJsonProofBinding.bind(source.result_pointer.absolute_path)
                != source.result_pointer
                or CanonicalJsonProofBinding.bind(source.proof_artifact.absolute_path)
                != source.proof_artifact
            ):
                raise ValueError("preflight qualification source file changed")

            if source.suite_id == "session_reset_tp1":
                result = SessionResetQualificationResultPointer.load(
                    source.result_pointer.absolute_path
                )
                if result.sha256 != source.result_pointer.semantic_sha256:
                    raise ValueError("session reset result pointer identity changed")
                assignment = SessionResetQualificationAssignment.from_dict(
                    result.assignment.reopen()
                )
                pointer = SessionResetQualificationProofPointer.from_dict(
                    source.proof_artifact.reopen()
                )
                if (
                    pointer.sha256 != source.proof_artifact.semantic_sha256
                    or pointer.result_pointer != source.result_pointer
                ):
                    raise ValueError("session reset durable proof pointer changed")
                verified = pointer.revalidate(now_ns=now_ns)
                artifact = NativeRuntimeGpuProofArtifact.from_dict(
                    pointer.gpu_proof_artifact.reopen()
                )
                if (
                    verified.suite_id != "session_reset_tp1"
                    or verified.assignment_sha256 != assignment.sha256
                    or verified.source_identity_sha256
                    != assignment.source_identity_sha256
                    or verified.topology_sha256 != assignment.topology_sha256
                    or verified.inventory_sha256 != assignment.inventory_sha256
                    or verified.hardware_envelope_sha256
                    != assignment.hardware_envelope_sha256
                    or verified.gpu_uuids != (assignment.gpu_uuid,)
                    or artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
                    != self.release_root_manifest_sha256
                ):
                    raise ValueError("session reset proof identity differs")
                proof_registry_sha256 = assignment.registry_sha256
                proof_runtime_sha256 = assignment.runtime_sha256
                proof_inventory_sha256 = assignment.inventory_sha256
                proof_hardware_sha256 = assignment.hardware_envelope_sha256
                proof_gpu_uuids = (assignment.gpu_uuid,)
            else:
                result = NativeRuntimeQualificationResultPointer.load(
                    source.result_pointer.absolute_path
                )
                if result.sha256 != source.result_pointer.semantic_sha256:
                    raise ValueError(
                        "native qualification result pointer identity changed"
                    )
                assignment = NativeRuntimeQualificationAssignment.from_dict(
                    result.assignment.reopen()
                )
                observation = NativeRuntimeQualificationObservation.from_dict(
                    result.live_observation.reopen()
                )
                observation.validate_assignment(assignment)
                if assignment.suite_id != source.suite_id:
                    raise ValueError("preflight qualification suite was relabelled")
                if source.suite_id in _PREFLIGHT_NATIVE_QUALIFICATION_SUITES:
                    validate_native_runtime_gpu_proof_artifact(
                        source.proof_artifact.absolute_path,
                        expected_suite_id=source.suite_id,  # type: ignore[arg-type]
                        expected_topology_sha256=assignment.topology_sha256,
                        expected_source_identity_sha256=(
                            assignment.source_identity_sha256
                        ),
                        expected_inventory_sha256=assignment.inventory_sha256,
                        expected_gpu_uuids=assignment.gpu_uuids,
                        expected_hardware_envelope_sha256=(
                            assignment.hardware_envelope_sha256
                        ),
                        expected_assignment_sha256=assignment.sha256,
                        expected_qualification_observation_sha256=(observation.sha256),
                        expected_root_manifest_sha256=(
                            self.release_root_manifest_sha256
                        ),
                        now_ns=now_ns,
                    )
                elif source.suite_id in (_PREFLIGHT_DISTRIBUTED_QUALIFICATION_SUITES):
                    base = assignment.base_exactness_result_pointer
                    if (
                        base is None
                        or base.semantic_sha256 != exactness_pointer.sha256
                        or base != CanonicalJsonProofBinding.bind(base.absolute_path)
                    ):
                        raise ValueError(
                            "distributed proof uses another base exactness result"
                        )
                    validate_distributed_runtime_gpu_proof_artifact(
                        source.proof_artifact.absolute_path,
                        expected_topology_mode=source.suite_id,  # type: ignore[arg-type]
                        expected_topology_sha256=assignment.topology_sha256,
                        expected_source_identity_sha256=(
                            assignment.source_identity_sha256
                        ),
                        expected_inventory_sha256=assignment.inventory_sha256,
                        expected_gpu_uuids=assignment.gpu_uuids,  # type: ignore[arg-type]
                        expected_hardware_envelope_sha256=(
                            assignment.hardware_envelope_sha256
                        ),
                        expected_assignment_sha256=assignment.sha256,
                        expected_qualification_observation_sha256=(observation.sha256),
                        expected_base_exactness_result_pointer_sha256=(
                            base.semantic_sha256
                        ),
                        expected_root_manifest_sha256=(
                            self.release_root_manifest_sha256
                        ),
                        now_ns=now_ns,
                    )
                else:  # pragma: no cover - constructor closes the suite universe
                    raise AssertionError("unsupported preflight qualification suite")
                proof_registry_sha256 = assignment.registry_sha256
                proof_runtime_sha256 = assignment.runtime_sha256
                proof_inventory_sha256 = assignment.inventory_sha256
                proof_hardware_sha256 = assignment.hardware_envelope_sha256
                proof_gpu_uuids = assignment.gpu_uuids

            if (
                proof_registry_sha256 != self.registry_sha256
                or proof_runtime_sha256 != self.runtime_sha256
                or proof_inventory_sha256 != self.inventory_sha256
                or proof_hardware_sha256
                != exactness_assignment.hardware_envelope_sha256
                or not proof_gpu_uuids
                or not set(proof_gpu_uuids).issubset(exactness_assignment.gpu_uuids)
            ):
                raise ValueError(
                    "preflight qualification proof differs from exactness identity"
                )

    def revalidate(
        self,
        registry: ExperimentRegistry,
        *,
        now_ns: int | None = None,
    ) -> tuple[PreflightCellTerminal, ...]:
        """Deep-reopen every raw pointer and derive—not accept—terminal rows."""

        verification_ns = time.time_ns() if now_ns is None else now_ns
        if type(verification_ns) is not int or verification_ns < 0:
            raise ValueError("preflight verification time is invalid")

        if type(registry) is not ExperimentRegistry or (
            registry.sha256 != self.registry_sha256
        ):
            raise ValueError("preflight execution source belongs to another registry")
        compile_value = self.compile_result.load()
        compile_pointer = CompileResultPointer.load(self.compile_result.path)
        if (
            compile_pointer.sha256 != self.compile_result.semantic_sha256
            or content_sha256(compile_value) != compile_pointer.sha256
            or compile_pointer.schema_version != 3
            or compile_pointer.formal_execution_authorized is not True
            or compile_pointer.assignment_plan_source is None
            or compile_pointer.control_verification_receipt is None
        ):
            raise ValueError("preflight compile pointer is not formal authority")
        compile_plan = CompileAssignmentPlan.load(
            compile_pointer.assignment_plan_source.absolute_path
        )
        compile_assignment, _cache, _prewarm, compile_launch = compile_plan.revalidate()
        compile_control = CompileControlVerificationReceipt.load(
            compile_pointer.control_verification_receipt.absolute_path
        )
        compile_cells = tuple(
            cell
            for cell in registry.cells_for("preflight")
            if cell.resources.workload_class is WorkloadClass.COMPILE
        )
        if len(compile_cells) != 1:
            raise ValueError("preflight registry compile cardinality changed")
        compile_cell = compile_cells[0]
        if (
            compile_assignment.cell_id != compile_cell.cell_id
            or compile_assignment.registry_sha256 != self.registry_sha256
            or compile_assignment.runtime_sha256 != self.runtime_sha256
            or compile_assignment.split_sha256 != self.split_sha256
            or compile_assignment.inventory_sha256 != self.inventory_sha256
            or compile_assignment.gpu_uuids != compile_cell.resources.gpu_uuids
            or compile_launch.gpu_uuids != compile_assignment.gpu_uuids
            or compile_control.assignment_plan_sha256 != compile_plan.sha256
            or compile_control.inventory_sha256 != self.inventory_sha256
        ):
            raise ValueError("preflight compile authority lineage differs")

        exactness_value = self.exactness_result.load()
        exactness_pointer = ExactnessPreflightResultPointer.load(
            self.exactness_result.path
        )
        if (
            exactness_pointer.sha256 != self.exactness_result.semantic_sha256
            or content_sha256(exactness_value) != exactness_pointer.sha256
            or exactness_pointer.schema_version != 4
            or exactness_pointer.control_verification_receipt is None
            or exactness_pointer.qualification_proof_artifact is None
        ):
            raise ValueError("preflight exactness pointer is not formal authority")
        exactness_assignment = ExactnessPreflightAssignment.load(
            exactness_pointer.assignment.absolute_path
        )
        exactness_terminal = ExactnessPreflightTerminal.load(
            exactness_pointer.terminal.absolute_path
        )
        exactness_cells = tuple(
            cell
            for cell in registry.cells_for("preflight")
            if cell.identity.task == "exactness_memory_telemetry_preflight"
        )
        if len(exactness_cells) != 1:
            raise ValueError("preflight registry exactness cardinality changed")
        exactness_cell = exactness_cells[0]
        if (
            exactness_assignment.cell_id != exactness_cell.cell_id
            or exactness_assignment.registry_sha256 != self.registry_sha256
            or exactness_assignment.runtime_sha256 != self.runtime_sha256
            or exactness_assignment.split_sha256 != self.split_sha256
            or exactness_assignment.inventory_sha256 != self.inventory_sha256
            or exactness_assignment.gpu_uuids != exactness_cell.resources.gpu_uuids
            or exactness_terminal.status != "PASSED"
        ):
            raise ValueError("preflight exactness authority lineage/outcome differs")

        self._revalidate_qualification_proofs(
            exactness_pointer=exactness_pointer,
            exactness_assignment=exactness_assignment,
            now_ns=verification_ns,
        )

        interference = validate_formal_preflight_interference_proof_artifact(
            self.interference_proof_artifact.absolute_path,
            registry=registry,
            expected_activation_sha256=self.dispatch_activation_sha256,
            expected_runtime_sha256=self.runtime_sha256,
            expected_split_sha256=self.split_sha256,
            expected_inventory_sha256=self.inventory_sha256,
            now_ns=verification_ns,
        )
        if (
            interference.artifact_sha256
            != self.interference_proof_artifact.semantic_sha256
        ):
            raise ValueError("preflight interference proof identity differs")
        expected_interference = {
            cell.cell_id: cell
            for cell in registry.cells_for("preflight")
            if cell.identity.task == "simultaneous_single_gpu_interference"
        }
        interference_rows: list[PreflightCellTerminal] = []
        seen: set[str] = set()
        for proof_row in interference.rows:
            cell_id = proof_row.registry_cell_id
            cell = expected_interference.get(cell_id)
            if cell is None or cell_id in seen:
                raise ValueError("preflight interference terminal coverage differs")
            if (
                (proof_row.gpu_uuid,) != cell.resources.gpu_uuids
                or proof_row.observation.completed_requests
                != len(proof_row.observation.request_ids)
                or any(value != 0 for _, value in proof_row.observation.safety_counters)
            ):
                raise ValueError("preflight interference terminal lineage differs")
            seen.add(cell_id)
            interference_rows.append(
                PreflightCellTerminal(
                    cell_id=cell_id,
                    terminal_kind="interference",
                    terminal_authority_sha256=proof_row.sha256,
                    status=("PASSED" if interference.status == "PASSED" else "FAILED"),
                    expected_rank_count=1,
                    terminal_rank_count=1,
                    failure_count=0 if interference.status == "PASSED" else 1,
                    error_count=0,
                    skip_count=0,
                )
            )
        if seen != set(expected_interference) or len(seen) != 8:
            raise ValueError("preflight requires exactly eight interference terminals")
        return tuple(
            sorted(
                (
                    PreflightCellTerminal(
                        cell_id=compile_cell.cell_id,
                        terminal_kind="compile",
                        terminal_authority_sha256=compile_pointer.sha256,
                        status="PASSED",
                        expected_rank_count=len(compile_cell.resources.gpu_uuids),
                        terminal_rank_count=len(compile_assignment.gpu_uuids),
                        failure_count=0,
                        error_count=0,
                        skip_count=0,
                    ),
                    PreflightCellTerminal(
                        cell_id=exactness_cell.cell_id,
                        terminal_kind="exactness",
                        terminal_authority_sha256=exactness_pointer.sha256,
                        status="PASSED",
                        expected_rank_count=len(exactness_cell.resources.gpu_uuids),
                        terminal_rank_count=exactness_terminal.terminal_rank_count,
                        failure_count=exactness_terminal.tests_failed,
                        error_count=exactness_terminal.tests_errored,
                        skip_count=exactness_terminal.tests_skipped,
                    ),
                    *interference_rows,
                ),
                key=lambda row: row.cell_id,
            )
        )


@dataclass(frozen=True)
class PreflightCoverageReceipt:
    """Exact all-cell terminal reduction for one preflight activation."""

    schema_version: int
    kind: Literal["formal_preflight_coverage_receipt"]
    protocol_sha256: str
    registry_sha256: str
    activation_sha256: str
    runtime_sha256: str
    split_sha256: str
    status: Literal["COMPLETE", "BLOCKED"]
    terminals: tuple[PreflightCellTerminal, ...]
    source_authority: PreflightExecutionSourceAuthority | None = None

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version not in {1, 2}
            or self.kind != "formal_preflight_coverage_receipt"
        ):
            raise ValueError("preflight coverage receipt schema is unsupported")
        expected_protocol = (
            PREFLIGHT_POINTER_COVERAGE_PROTOCOL_SHA256
            if self.schema_version == 2
            else PREFLIGHT_COVERAGE_PROTOCOL_SHA256
        )
        if self.protocol_sha256 != expected_protocol:
            raise ValueError("preflight coverage receipt uses another reducer")
        if self.schema_version == 1 and self.source_authority is not None:
            raise ValueError("diagnostic preflight coverage cannot claim raw authority")
        if (
            self.schema_version == 2
            and type(self.source_authority) is not PreflightExecutionSourceAuthority
        ):
            raise TypeError("formal preflight coverage lacks raw source authority")
        for label, digest in (
            ("registry", self.registry_sha256),
            ("activation", self.activation_sha256),
            ("runtime", self.runtime_sha256),
            ("split", self.split_sha256),
        ):
            _require_sha256(f"preflight coverage {label}", digest)
        if not self.terminals or any(
            type(row) is not PreflightCellTerminal for row in self.terminals
        ):
            raise TypeError("preflight coverage requires exact terminal rows")
        cell_ids = tuple(row.cell_id for row in self.terminals)
        if cell_ids != tuple(sorted(set(cell_ids))):
            raise ValueError("preflight terminals must be cell-sorted and unique")
        expected_status = (
            "COMPLETE" if all(row.passed for row in self.terminals) else "BLOCKED"
        )
        if self.status != expected_status:
            raise ValueError("preflight aggregate status differs from terminal rows")

    def _payload(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "registry_sha256": self.registry_sha256,
            "activation_sha256": self.activation_sha256,
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "status": self.status,
            "terminals": [row.to_dict() for row in self.terminals],
        }
        if self.schema_version == 2:
            if self.source_authority is None:
                raise AssertionError("formal preflight coverage lost source authority")
            value["source_authority"] = self.source_authority.to_dict()
        return value

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"receipt_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("preflight coverage receipt must be a JSON object")
        schema_version = value.get("schema_version")
        fields = {
            "receipt_sha256",
            "schema_version",
            "kind",
            "protocol_sha256",
            "registry_sha256",
            "activation_sha256",
            "runtime_sha256",
            "split_sha256",
            "status",
            "terminals",
        }
        if schema_version == 2:
            fields.add("source_authority")
        row = _strict_object(
            "preflight coverage receipt",
            value,
            frozenset(fields),
        )
        raw_terminals = row.pop("terminals")
        declared = row.pop("receipt_sha256")
        raw_source = row.pop("source_authority", None)
        if type(raw_terminals) is not list:
            raise TypeError("preflight terminals must be a JSON array")
        receipt = cls(
            terminals=tuple(
                PreflightCellTerminal.from_dict(item) for item in raw_terminals
            ),
            source_authority=(
                None
                if raw_source is None
                else PreflightExecutionSourceAuthority.from_dict(raw_source)
            ),
            **row,
        )
        if declared != receipt.sha256:
            raise ValueError("preflight coverage receipt SHA-256 mismatch")
        return receipt


@dataclass(frozen=True)
class PreflightSealControlBinding:
    """Receipt-bound summary of the verified dynamic control decision."""

    schema_version: int
    kind: Literal["formal_preflight_seal_control_binding"]
    status: Literal["SEALED"]
    registry_sha256: str
    activation_sha256: str
    runtime_sha256: str
    split_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    raw_completed_cells_sha256: str
    coverage_receipt_sha256: str
    coverage_attestation_sha256: str
    capacity_gate_sha256: str
    capacity_attestation_sha256: str
    deployment_policy_authorization_sha256: str
    trust_bundle_sha256: str
    trusted_attester_policy_sha256: str
    replay_reservation_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "formal_preflight_seal_control_binding"
            or self.status != "SEALED"
        ):
            raise ValueError("preflight seal control binding schema is unsupported")
        for label, digest in (
            ("registry", self.registry_sha256),
            ("activation", self.activation_sha256),
            ("runtime", self.runtime_sha256),
            ("split", self.split_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("raw completed cells", self.raw_completed_cells_sha256),
            ("coverage receipt", self.coverage_receipt_sha256),
            ("coverage attestation", self.coverage_attestation_sha256),
            ("capacity gate", self.capacity_gate_sha256),
            ("capacity attestation", self.capacity_attestation_sha256),
            (
                "deployment policy authorization",
                self.deployment_policy_authorization_sha256,
            ),
            ("trust bundle", self.trust_bundle_sha256),
            ("trusted attester policy", self.trusted_attester_policy_sha256),
            ("replay reservation", self.replay_reservation_sha256),
        ):
            _require_sha256(f"preflight seal {label}", digest)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "status": self.status,
            "registry_sha256": self.registry_sha256,
            "activation_sha256": self.activation_sha256,
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "inventory_sha256": self.inventory_sha256,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "raw_completed_cells_sha256": self.raw_completed_cells_sha256,
            "coverage_receipt_sha256": self.coverage_receipt_sha256,
            "coverage_attestation_sha256": self.coverage_attestation_sha256,
            "capacity_gate_sha256": self.capacity_gate_sha256,
            "capacity_attestation_sha256": self.capacity_attestation_sha256,
            "deployment_policy_authorization_sha256": (
                self.deployment_policy_authorization_sha256
            ),
            "trust_bundle_sha256": self.trust_bundle_sha256,
            "trusted_attester_policy_sha256": (self.trusted_attester_policy_sha256),
            "replay_reservation_sha256": self.replay_reservation_sha256,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"binding_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = frozenset(
            {
                "binding_sha256",
                "schema_version",
                "kind",
                "status",
                "registry_sha256",
                "activation_sha256",
                "runtime_sha256",
                "split_sha256",
                "inventory_sha256",
                "hardware_envelope_sha256",
                "raw_completed_cells_sha256",
                "coverage_receipt_sha256",
                "coverage_attestation_sha256",
                "capacity_gate_sha256",
                "capacity_attestation_sha256",
                "deployment_policy_authorization_sha256",
                "trust_bundle_sha256",
                "trusted_attester_policy_sha256",
                "replay_reservation_sha256",
            }
        )
        row = _strict_object("preflight seal control binding", value, fields)
        declared = row.pop("binding_sha256")
        binding = cls(**row)
        if declared != binding.sha256:
            raise ValueError("preflight seal control binding SHA-256 mismatch")
        return binding


def _coverage_from_verified_activation(
    registry: ExperimentRegistry,
    activation: RegistryStageActivationArtifact,
    terminals: tuple[PreflightCellTerminal, ...],
    *,
    source_authority: PreflightExecutionSourceAuthority | None,
) -> PreflightCoverageReceipt:
    if activation.experiment != "preflight":
        raise ValueError("preflight coverage received another stage activation")
    if activation.status != "AVAILABLE":
        raise PreflightCoverageBlocked("mandatory_preflight_activation_incomplete")
    if any(
        row.status is RegistryStageDispositionStatus.BLOCKED
        for row in activation.dispositions
    ):
        raise PreflightCoverageBlocked("mandatory_preflight_activation_incomplete")
    mandatory_cells = tuple(
        cell
        for cell in registry.cells_for("preflight")
        if cell.status is not CellStatus.NOT_APPLICABLE
    )
    terminal_by_cell = {row.cell_id: row for row in terminals}
    if len(terminal_by_cell) != len(terminals) or set(terminal_by_cell) != {
        cell.cell_id for cell in mandatory_cells
    }:
        raise ValueError("preflight terminal coverage differs from mandatory cells")
    for cell in mandatory_cells:
        terminal = terminal_by_cell[cell.cell_id]
        if terminal.terminal_kind != _preflight_terminal_kind(cell):
            raise ValueError("preflight terminal kind differs from registry cell")
        if terminal.expected_rank_count != len(cell.resources.gpu_uuids):
            raise ValueError("preflight terminal rank count differs from registry cell")
    ordered = tuple(sorted(terminals, key=lambda row: row.cell_id))
    formal = source_authority is not None
    return PreflightCoverageReceipt(
        schema_version=2 if formal else 1,
        kind="formal_preflight_coverage_receipt",
        protocol_sha256=(
            PREFLIGHT_POINTER_COVERAGE_PROTOCOL_SHA256
            if formal
            else PREFLIGHT_COVERAGE_PROTOCOL_SHA256
        ),
        registry_sha256=registry.sha256,
        activation_sha256=activation.sha256,
        runtime_sha256=activation.runtime_sha256,
        split_sha256=activation.split_sha256,
        status="COMPLETE" if all(row.passed for row in ordered) else "BLOCKED",
        terminals=ordered,
        source_authority=source_authority,
    )


def materialize_preflight_coverage(
    registry: ExperimentRegistry,
    activation: RegistryStageActivationArtifact,
    terminals: tuple[PreflightCellTerminal, ...],
) -> PreflightCoverageReceipt:
    """Legacy diagnostic reducer; never sufficient to seal formal preflight."""

    if type(registry) is not ExperimentRegistry:
        raise TypeError("preflight coverage requires an exact registry")
    if type(activation) is not RegistryStageActivationArtifact:
        raise TypeError("preflight coverage requires an exact activation")
    verify_registry_stage_activation(registry, activation)
    return _coverage_from_verified_activation(
        registry,
        activation,
        terminals,
        source_authority=None,
    )


def materialize_pointer_preflight_coverage(
    registry: ExperimentRegistry,
    source_authority: PreflightExecutionSourceAuthority,
) -> tuple[RegistryStageActivationArtifact, PreflightCoverageReceipt]:
    """Deep-reopen three typed sources and reduce the exact ten-cell receipt."""

    if type(registry) is not ExperimentRegistry:
        raise TypeError("pointer preflight coverage requires an exact registry")
    if type(source_authority) is not PreflightExecutionSourceAuthority:
        raise TypeError("pointer preflight coverage requires an exact source")
    terminals = source_authority.revalidate(registry)
    activation = materialize_pointer_preflight_stage_activation(
        registry,
        runtime_sha256=source_authority.runtime_sha256,
        split_sha256=source_authority.split_sha256,
        source_authority_sha256=source_authority.sha256,
    )
    verify_pointer_preflight_stage_activation(
        registry,
        activation,
        source_authority_sha256=source_authority.sha256,
    )
    receipt = _coverage_from_verified_activation(
        registry,
        activation,
        terminals,
        source_authority=source_authority,
    )
    return activation, receipt


def verify_preflight_coverage(
    registry: ExperimentRegistry,
    activation: RegistryStageActivationArtifact,
    receipt: PreflightCoverageReceipt,
) -> None:
    if type(receipt) is not PreflightCoverageReceipt:
        raise TypeError("preflight verifier requires an exact coverage receipt")
    if receipt.schema_version == 2:
        if receipt.source_authority is None:
            raise AssertionError("formal preflight receipt lost its raw source")
        terminals = receipt.source_authority.revalidate(registry)
        verify_pointer_preflight_stage_activation(
            registry,
            activation,
            source_authority_sha256=receipt.source_authority.sha256,
        )
        expected = _coverage_from_verified_activation(
            registry,
            activation,
            terminals,
            source_authority=receipt.source_authority,
        )
    else:
        expected = materialize_preflight_coverage(
            registry,
            activation,
            receipt.terminals,
        )
    if receipt != expected:
        raise ValueError("preflight coverage is not the exact reducer output")


def require_complete_preflight_coverage(receipt: PreflightCoverageReceipt) -> None:
    if type(receipt) is not PreflightCoverageReceipt:
        raise TypeError("preflight gate requires an exact coverage receipt")
    if receipt.schema_version != 2 or receipt.status != "COMPLETE":
        raise PreflightCoverageBlocked("mandatory_preflight_terminal_not_clean")


def materialize_formal_preflight_stage_coverage(
    materialization: StageMaterializationReceipt,
    pointer_coverage: PreflightCoverageReceipt,
    *,
    candidate_state_coverage: TtsL0CandidateStateCoverage,
    candidate_replay_proof_paths: tuple[str | Path, str | Path],
    now_ns: int,
) -> StageCoverageReceipt:
    """Derive the only signable preflight ``StageCoverageReceipt``.

    The generic formal registry signs :class:`StageCoverageReceipt`, whereas
    the GPU preflight reducer deliberately emits the richer pointer coverage
    above.  This bridge is therefore a verifier, not a caller-supplied
    disposition adapter: it reopens the two TTS/L0 candidate replay proofs,
    maps every materialized cell to its exact registry terminal, and binds the
    pointer activation into every terminal receipt identity.
    """

    from lightcone_spec.experiments.formal_protocol import (
        TtsL0CandidateStateCoverage,
    )
    from lightcone_spec.experiments.stage_materialization import (
        StageCellDisposition,
        StageCoverageReceipt,
        StageMaterializationReceipt,
        materialize_preflight,
    )
    from lightcone_spec.orchestration.native_terminal import (
        validate_candidate_state_replay_proof_artifact,
    )

    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("preflight stage coverage requires an exact materialization")
    if type(pointer_coverage) is not PreflightCoverageReceipt:
        raise TypeError("preflight stage coverage requires exact pointer coverage")
    if type(candidate_state_coverage) is not TtsL0CandidateStateCoverage:
        raise TypeError("preflight stage coverage requires exact candidate coverage")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("preflight stage coverage verification time is invalid")
    require_complete_preflight_coverage(pointer_coverage)
    source = pointer_coverage.source_authority
    if source is None:  # pragma: no cover - require_complete closes this branch
        raise AssertionError("complete preflight pointer coverage lost its source")
    expected_materialization = materialize_preflight(
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        gpu_hours=materialization.gpu_hours,
    )
    if materialization != expected_materialization:
        raise ValueError("preflight stage materialization is not the exact prefix")
    registry = build_industrial_registry()
    if (
        materialization.source_decision_sha256 != registry.sha256
        or pointer_coverage.registry_sha256 != registry.sha256
    ):
        raise ValueError("preflight materialization uses another formal registry")
    _activation, expected_pointer_coverage = materialize_pointer_preflight_coverage(
        registry,
        source,
    )
    if pointer_coverage != expected_pointer_coverage:
        raise ValueError("preflight pointer coverage is not the exact deep reducer")

    if (
        type(candidate_replay_proof_paths) is not tuple
        or len(candidate_replay_proof_paths) != 2
        or any(
            not isinstance(path, (str, Path)) for path in candidate_replay_proof_paths
        )
    ):
        raise TypeError("preflight candidate replay proofs must be an exact pair")
    supplied_paths = tuple(Path(path) for path in candidate_replay_proof_paths)
    if len(set(supplied_paths)) != 2 or any(
        not path.is_absolute() or path != path.resolve(strict=False)
        for path in supplied_paths
    ):
        raise ValueError(
            "preflight candidate replay proof paths are not distinct/canonical"
        )
    canonical_paths = tuple(str(path) for path in supplied_paths)
    pointers = tuple(
        validate_candidate_state_replay_proof_artifact(
            path,
            expected_inventory_sha256=source.inventory_sha256,
            expected_registry_sha256=source.registry_sha256,
            expected_root_manifest_sha256=source.release_root_manifest_sha256,
            now_ns=now_ns,
        )
        for path in canonical_paths
    )
    if {pointer.method for pointer in pointers} != {"tts", "l0"}:
        raise ValueError("preflight candidate replay proofs are not exact TTS/L0")
    candidate_state_coverage.validate_native_replay_pointers(pointers)

    terminal_by_registry_cell = {row.cell_id: row for row in pointer_coverage.terminals}
    if len(terminal_by_registry_cell) != 10:
        raise ValueError("preflight pointer terminal coverage is not exact ten")
    materialized_registry_ids: list[str] = []
    dispositions: list[StageCellDisposition] = []
    for cell in materialization.cells:
        registry_cell_id = dict(cell.dimensions).get("registry_cell_id")
        if type(registry_cell_id) is not str:
            raise ValueError("preflight materialized cell lost its registry identity")
        terminal = terminal_by_registry_cell.get(registry_cell_id)
        if terminal is None or not terminal.passed:
            raise ValueError(
                "preflight materialized cell lacks a clean pointer terminal"
            )
        materialized_registry_ids.append(registry_cell_id)
        terminal_receipt_sha256 = content_sha256(
            {
                "schema_version": 1,
                "kind": "formal_preflight_materialized_terminal_receipt",
                "protocol_sha256": PREFLIGHT_STAGE_COVERAGE_BRIDGE_PROTOCOL_SHA256,
                "protocol_lock_sha256": materialization.protocol_lock_sha256,
                "materialization_receipt_sha256": materialization.sha256,
                "pointer_coverage_sha256": pointer_coverage.sha256,
                "pointer_activation_sha256": pointer_coverage.activation_sha256,
                "source_authority_sha256": source.sha256,
                "materialized_cell_id": cell.cell_id,
                "registry_cell_id": registry_cell_id,
                "terminal": terminal.to_dict(),
            }
        )
        dispositions.append(
            StageCellDisposition(
                stage="preflight",
                cell_id=cell.cell_id,
                status="COMPLETE",
                reason_code="pointer_terminal_complete",
                terminal_receipt_sha256=terminal_receipt_sha256,
            )
        )
    if len(set(materialized_registry_ids)) != 10 or set(
        materialized_registry_ids
    ) != set(terminal_by_registry_cell):
        raise ValueError("preflight materialization/pointer terminal mapping differs")
    result = StageCoverageReceipt(
        schema_version=2,
        stage="preflight",
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(sorted(dispositions, key=lambda row: row.cell_id)),
        tts_l0_candidate_state_coverages=(candidate_state_coverage,),
    )
    result.validate_against(materialization)
    return result


__all__ = [
    "PREFLIGHT_COVERAGE_PROTOCOL_SHA256",
    "PREFLIGHT_POINTER_COVERAGE_PROTOCOL_SHA256",
    "PREFLIGHT_REQUIRED_QUALIFICATION_SUITES",
    "PREFLIGHT_STAGE_COVERAGE_BRIDGE_PROTOCOL_SHA256",
    "PreflightCellTerminal",
    "PreflightCoverageBlocked",
    "PreflightCoverageReceipt",
    "PreflightExecutionSourceAuthority",
    "PreflightQualificationProofSource",
    "PreflightSealControlBinding",
    "materialize_formal_preflight_stage_coverage",
    "materialize_pointer_preflight_coverage",
    "materialize_preflight_coverage",
    "preflight_coverage_control_lineage_sha256",
    "require_complete_preflight_coverage",
    "verify_preflight_coverage",
]
