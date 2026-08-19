"""Verifier-owned execution identity for staged E5 failure diagnostics.

The serving execution plan and the fault assignment are deliberately distinct.
One serving cell may establish the process/model/topology identity, while each
fault scenario receives a dedicated assignment commitment that also binds its
cohort count and run nonce.  This module does not execute or attest a fault;
the source-owned actuator consumes the private-sealed value at the host-local
execution boundary and publishes a separately controlled terminal proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from lightcone_spec.experiments.failure_actuator import (
    FAILURE_ACTUATOR_PROTOCOL_SHA256,
)
from lightcone_spec.experiments.formal_protocol import (
    FormalRuntimeAuthorityManifest,
    ProtocolLock,
    content_sha256,
)
from lightcone_spec.experiments.formal_stage_execution import (
    FormalServingExecutionRebuildInput,
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.stage_materialization import (
    E5_BACKENDS,
    E5_COHORT_COUNTS,
    E5_FAILURE_DIAGNOSTIC_PROTOCOL_SHA256,
    E5_FAILURES,
    E5_TOPOLOGIES,
    MaterializedCell,
    StageMaterializationReceipt,
    default_e5_failure_diagnostic_authority,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

if TYPE_CHECKING:
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FormalSingleOperatorJsonBinding,
    )

FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_e5_failure_execution_binding_protocol",
        "serving_plan": "sealed_formal_serving_execution_plan_sha256",
        "fault_assignment": (
            "cell_plus_serving_binding_plus_scenario_plus_topology_plus_cohort_"
            "inventory_plus_run_nonce"
        ),
        "scope": "one_correctness_only_diagnostic_attempt",
        "materialized_matrix": "11_scenarios_x_2_backends_x_3_topologies_x_4_K",
    }
)
FORMAL_SINGLE_OPERATOR_E5_FAILURE_EXECUTION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e5_failure_execution_protocol",
        "source": "exact_e5_final_execution_source_and_prepared_launch_entry",
        "matrix": "11_failures_x_2_backends_x_3_topologies_x_4_cohorts",
        "placement": "prepared_topology_and_ordered_gpu_uuids",
        "attempt": "attempt_0_one_shot_no_retry",
        "terminal": "correctness_only_pass_fail_without_confidence_interval",
    }
)


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _stable_binding(
    binding: CanonicalJsonProofBinding,
    *,
    label: str,
) -> CanonicalJsonProofBinding:
    if type(binding) is not CanonicalJsonProofBinding:
        raise TypeError(f"{label} is not path-bound")
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError(f"{label} changed")
    return binding


@dataclass(frozen=True)
class FormalFailureExecutionSubject:
    """Deterministic but non-authorizing identity for one fault attempt."""

    schema_version: int
    protocol_lock_sha256: str
    formal_runtime_authority_manifest_sha256: str
    materialization_receipt_sha256: str
    materialized_cell_id: str
    serving_execution_binding_sha256: str
    serving_execution_plan_sha256: str
    serving_rank_config_sha256: str
    assignment_sha256: str
    inventory_sha256: str
    registry_sha256: str
    backend: str
    topology: str
    scenario: str
    cohort_count: int
    run_nonce_sha256: str
    failure_actuator_authority_sha256: str
    failure_reducer_authority_sha256: str
    correctness_only: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("formal failure execution schema is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            (
                "runtime authority manifest",
                self.formal_runtime_authority_manifest_sha256,
            ),
            ("materialization", self.materialization_receipt_sha256),
            ("materialized cell", self.materialized_cell_id),
            ("serving binding", self.serving_execution_binding_sha256),
            ("serving plan", self.serving_execution_plan_sha256),
            ("serving rank config", self.serving_rank_config_sha256),
            ("failure assignment", self.assignment_sha256),
            ("inventory", self.inventory_sha256),
            ("registry", self.registry_sha256),
            ("run nonce", self.run_nonce_sha256),
            ("failure actuator authority", self.failure_actuator_authority_sha256),
            ("failure reducer authority", self.failure_reducer_authority_sha256),
        ):
            _sha256(f"formal failure {label}", digest)
        if (
            self.backend not in E5_BACKENDS
            or self.topology not in E5_TOPOLOGIES
            or self.scenario not in E5_FAILURES
            or self.cohort_count not in E5_COHORT_COUNTS
        ):
            raise ValueError("formal failure execution lies outside the 264-row matrix")
        if self.correctness_only is not True:
            raise ValueError("formal failure execution must remain correctness-only")
        expected_assignment = content_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_formal_e5_failure_assignment",
                "protocol_sha256": (FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256),
                "materialized_cell_id": self.materialized_cell_id,
                "serving_execution_binding_sha256": (
                    self.serving_execution_binding_sha256
                ),
                "serving_execution_plan_sha256": (self.serving_execution_plan_sha256),
                "scenario": self.scenario,
                "backend": self.backend,
                "topology": self.topology,
                "cohort_count": self.cohort_count,
                "inventory_sha256": self.inventory_sha256,
                "run_nonce_sha256": self.run_nonce_sha256,
            }
        )
        if self.assignment_sha256 != expected_assignment:
            raise ValueError("formal failure assignment identity is not canonical")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("formal failure execution subject fields differ")
        return cls(**value)  # type: ignore[arg-type]


_VERIFIED_FORMAL_FAILURE_EXECUTION_SEAL = object()


@dataclass(frozen=True, init=False)
class VerifiedFormalFailureExecutionBinding:
    """Private-sealed token accepted by the formal failure actuator bridge."""

    subject: FormalFailureExecutionSubject
    serving_execution: VerifiedFormalServingExecutionBinding
    _construction_seal: object

    def __init__(
        self,
        *,
        subject: FormalFailureExecutionSubject,
        serving_execution: VerifiedFormalServingExecutionBinding,
        _construction_seal: object,
    ) -> None:
        if _construction_seal is not _VERIFIED_FORMAL_FAILURE_EXECUTION_SEAL:
            raise TypeError("formal failure binding is verifier-constructed only")
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "serving_execution", serving_execution)
        object.__setattr__(self, "_construction_seal", _construction_seal)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "protocol_sha256": FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256,
                "subject_sha256": self.subject.sha256,
                "serving_execution_binding_sha256": self.serving_execution.sha256,
            }
        )


def _require_exact_e5_final_failure_materialization(
    materialization: StageMaterializationReceipt,
    *,
    protocol_lock_sha256: str,
) -> tuple[object, tuple[MaterializedCell, ...]]:
    """Validate the final-only E5 matrix before deriving any assignment."""

    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("formal failure binding requires exact materialization")
    if (
        materialization.stage != "E5"
        or materialization.protocol_lock_sha256 != protocol_lock_sha256
        or materialization.materialization_rule
        != ("450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics")
    ):
        raise ValueError("formal failure binding requires the final-only E5 matrix")
    authority = default_e5_failure_diagnostic_authority()
    failure_cells = tuple(
        row
        for row in materialization.cells
        if row.task == "deterministic_failure_injection"
    )
    headline_cells = tuple(
        row
        for row in materialization.cells
        if row.task == "production_slo_power_prefix"
    )
    observed_members = {
        dict(row.dimensions).get("failure_member_id") for row in failure_cells
    }
    block_values = {dict(row.dimensions).get("block") for row in headline_cells}
    if (
        len(failure_cells) != 264
        or observed_members != {row.member_id for row in authority.members}
        or len(headline_cells) + len(failure_cells) != len(materialization.cells)
        or any(type(block) is not int for block in block_values)
        or not 12 <= len(block_values) <= 20
        or block_values != set(range(4, 4 + len(block_values)))
        or len(headline_cells) != 450 * len(block_values)
        or any(
            dict(row.dimensions).get("block_phase") != "final" for row in headline_cells
        )
    ):
        raise ValueError(
            "formal failure binding requires exact 450N final plus 264 matrix"
        )
    return authority, failure_cells


def bind_formal_failure_execution(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    materialized_cell_id: str,
    serving_execution: VerifiedFormalServingExecutionBinding,
) -> VerifiedFormalFailureExecutionBinding:
    """Rebuild one exact E5 diagnostic assignment from sealed serving identity."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("formal failure binding requires exact ProtocolLock")
    if type(formal_runtime_authority_manifest) is not FormalRuntimeAuthorityManifest:
        raise TypeError("formal failure binding requires exact runtime authority")
    if (
        formal_runtime_authority_manifest.sha256
        != protocol_lock.formal_runtime_authority_manifest_sha256
    ):
        raise ValueError("formal failure runtime authority differs from ProtocolLock")
    actuator = formal_runtime_authority_manifest.member("failure_actuator")
    reducer = formal_runtime_authority_manifest.member("e5_failure_reducer")
    if (
        actuator.protocol_sha256 != FAILURE_ACTUATOR_PROTOCOL_SHA256
        or reducer.protocol_sha256 != E5_FAILURE_DIAGNOSTIC_PROTOCOL_SHA256
    ):
        raise ValueError("formal failure source protocols differ from ProtocolLock")
    authority, _failure_cells = _require_exact_e5_final_failure_materialization(
        materialization,
        protocol_lock_sha256=protocol_lock.sha256,
    )
    cells = tuple(
        row for row in materialization.cells if row.cell_id == materialized_cell_id
    )
    if len(cells) != 1:
        raise ValueError("formal failure binding requires one exact materialized cell")
    cell = cells[0]
    dimensions = dict(cell.dimensions)
    if (
        cell.method_role != "LightCone"
        or cell.task != "deterministic_failure_injection"
        or cell.publication_policy != "diagnostic_only"
        or dimensions.get("diagnostic_only") != "true"
        or dimensions.get("failure_authority_sha256") != authority.sha256
    ):
        raise ValueError("formal failure cell is not an authorized diagnostic row")
    backend = cell.backend
    topology = dimensions.get("topology")
    scenario = dimensions.get("failure")
    cohort_count = dimensions.get("cohort_count")
    if (
        type(backend) is not str
        or type(topology) is not str
        or type(scenario) is not str
        or type(cohort_count) is not int
    ):
        raise TypeError("formal failure cell dimensions are not exact")
    verified_serving = require_verified_formal_serving_execution_binding(
        serving_execution
    )
    serving_subject = verified_serving.subject
    if (
        serving_subject.protocol_lock_sha256 != protocol_lock.sha256
        or serving_subject.materialization_receipt_sha256 != materialization.sha256
        or serving_subject.materialized_cell_id != cell.cell_id
        or serving_subject.stage != "E5"
        or serving_subject.method != "l0"
        or serving_subject.topology_mode != topology
        or verified_serving.run_config.model.algorithm != backend
    ):
        raise ValueError(
            "formal failure serving execution differs from diagnostic cell"
        )
    assignment_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_e5_failure_assignment",
            "protocol_sha256": FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256,
            "materialized_cell_id": cell.cell_id,
            "serving_execution_binding_sha256": verified_serving.sha256,
            "serving_execution_plan_sha256": serving_subject.execution_plan_sha256,
            "scenario": scenario,
            "backend": backend,
            "topology": topology,
            "cohort_count": cohort_count,
            "inventory_sha256": serving_subject.inventory_sha256,
            "run_nonce_sha256": (serving_subject.execution_identity.run_nonce_sha256),
        }
    )
    subject = FormalFailureExecutionSubject(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        formal_runtime_authority_manifest_sha256=(
            formal_runtime_authority_manifest.sha256
        ),
        materialization_receipt_sha256=materialization.sha256,
        materialized_cell_id=cell.cell_id,
        serving_execution_binding_sha256=verified_serving.sha256,
        serving_execution_plan_sha256=serving_subject.execution_plan_sha256,
        serving_rank_config_sha256=serving_subject.rank_config_sha256,
        assignment_sha256=assignment_sha256,
        inventory_sha256=serving_subject.inventory_sha256,
        registry_sha256=protocol_lock.registry_sha256,
        backend=backend,
        topology=topology,
        scenario=scenario,
        cohort_count=cohort_count,
        run_nonce_sha256=serving_subject.execution_identity.run_nonce_sha256,
        failure_actuator_authority_sha256=actuator.sha256,
        failure_reducer_authority_sha256=reducer.sha256,
        correctness_only=True,
    )
    return VerifiedFormalFailureExecutionBinding(
        subject=subject,
        serving_execution=verified_serving,
        _construction_seal=_VERIFIED_FORMAL_FAILURE_EXECUTION_SEAL,
    )


def require_verified_formal_failure_execution_binding(
    value: object,
) -> VerifiedFormalFailureExecutionBinding:
    if (
        type(value) is not VerifiedFormalFailureExecutionBinding
        or value._construction_seal is not _VERIFIED_FORMAL_FAILURE_EXECUTION_SEAL
        or type(value.subject) is not FormalFailureExecutionSubject
        or type(value.serving_execution) is not VerifiedFormalServingExecutionBinding
    ):
        raise TypeError("formal failure execution requires a sealed binding")
    return value


@dataclass(frozen=True)
class FormalSingleOperatorE5FailureExecutionDescriptor:
    """Public trusted-mode identity for one exact E5 one-shot diagnostic."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_e5_failure_execution_descriptor"]
    protocol_sha256: str
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    prepared_launch_bundle: CanonicalJsonProofBinding
    prepared_launch_bundle_sha256: str
    prepared_launch_entry_sha256: str
    runtime_authority_manifest: CanonicalJsonProofBinding
    materialization: FormalSingleOperatorJsonBinding
    materialization_sha256: str
    inventory: CanonicalJsonProofBinding
    compile_launch_manifest: CanonicalJsonProofBinding
    request_schedule_receipt: CanonicalJsonProofBinding
    execution_binding_sha256: str
    subject_sha256: str
    failure_subject: FormalFailureExecutionSubject
    expected_failure_execution_binding_sha256: str
    gpu_uuids: tuple[str, ...]
    attempt_id: Literal["attempt-0"]
    retry_allowance: Literal[0]
    exclusive_timing: Literal[True]
    private_output_root: str

    def __post_init__(self) -> None:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            FormalSingleOperatorJsonBinding,
        )

        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e5_failure_execution_descriptor"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_E5_FAILURE_EXECUTION_PROTOCOL_SHA256
            or self.attempt_id != "attempt-0"
            or self.retry_allowance != 0
            or self.exclusive_timing is not True
            or type(self.failure_subject) is not FormalFailureExecutionSubject
        ):
            raise ValueError("single-operator E5 failure descriptor differs")
        for label, binding in (
            ("execution source", self.execution_source),
            ("prepared launch bundle", self.prepared_launch_bundle),
            ("runtime authority manifest", self.runtime_authority_manifest),
            ("inventory", self.inventory),
            ("compile launch", self.compile_launch_manifest),
            ("request schedule", self.request_schedule_receipt),
        ):
            _stable_binding(binding, label=f"single-operator E5 {label}")
        if type(self.materialization) is not FormalSingleOperatorJsonBinding:
            raise TypeError("single-operator E5 materialization is not path-bound")
        if (
            FormalSingleOperatorJsonBinding.bind(
                self.materialization.absolute_path,
                label="single-operator E5 materialization",
            )
            != self.materialization
        ):
            raise ValueError("single-operator E5 materialization changed")
        for label, digest in (
            ("execution source", self.execution_source_sha256),
            ("prepared launch bundle", self.prepared_launch_bundle_sha256),
            ("prepared launch entry", self.prepared_launch_entry_sha256),
            ("materialization", self.materialization_sha256),
            ("execution binding", self.execution_binding_sha256),
            ("serving subject", self.subject_sha256),
            (
                "failure execution binding",
                self.expected_failure_execution_binding_sha256,
            ),
        ):
            _sha256(f"single-operator E5 {label}", digest)
        expected_gpus = 1 if self.failure_subject.topology == "tp1_dp1" else 2
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != expected_gpus
            or len(set(self.gpu_uuids)) != expected_gpus
            or any(type(value) is not str or not value for value in self.gpu_uuids)
        ):
            raise ValueError("single-operator E5 GPU placement differs")
        root = Path(self.private_output_root)
        if (
            not root.is_absolute()
            or root != root.resolve(strict=False)
            or not root.is_dir()
            or root.is_symlink()
        ):
            raise ValueError("single-operator E5 private output root differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    @property
    def subject(self) -> FormalFailureExecutionSubject:
        return self.failure_subject

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "execution_source": self.execution_source.to_dict(),
            "execution_source_sha256": self.execution_source_sha256,
            "prepared_launch_bundle": self.prepared_launch_bundle.to_dict(),
            "prepared_launch_bundle_sha256": self.prepared_launch_bundle_sha256,
            "prepared_launch_entry_sha256": self.prepared_launch_entry_sha256,
            "runtime_authority_manifest": self.runtime_authority_manifest.to_dict(),
            "materialization": self.materialization.to_dict(),
            "materialization_sha256": self.materialization_sha256,
            "inventory": self.inventory.to_dict(),
            "compile_launch_manifest": self.compile_launch_manifest.to_dict(),
            "request_schedule_receipt": self.request_schedule_receipt.to_dict(),
            "execution_binding_sha256": self.execution_binding_sha256,
            "subject_sha256": self.subject_sha256,
            "failure_subject": self.failure_subject.to_dict(),
            "expected_failure_execution_binding_sha256": (
                self.expected_failure_execution_binding_sha256
            ),
            "gpu_uuids": list(self.gpu_uuids),
            "attempt_id": self.attempt_id,
            "retry_allowance": self.retry_allowance,
            "exclusive_timing": self.exclusive_timing,
            "private_output_root": self.private_output_root,
        }
        if include_sha256:
            value["descriptor_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            FormalSingleOperatorJsonBinding,
        )

        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "descriptor_sha256",
        }:
            raise ValueError("single-operator E5 failure descriptor fields differ")
        row = dict(value)
        declared = _sha256(
            "single-operator E5 failure descriptor",
            row.pop("descriptor_sha256"),
        )
        for name in (
            "execution_source",
            "prepared_launch_bundle",
            "runtime_authority_manifest",
            "inventory",
            "compile_launch_manifest",
            "request_schedule_receipt",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["materialization"] = FormalSingleOperatorJsonBinding.from_dict(
            row["materialization"]
        )
        row["failure_subject"] = FormalFailureExecutionSubject.from_dict(
            row["failure_subject"]
        )
        gpu_uuids = row.pop("gpu_uuids")
        if type(gpu_uuids) is not list:
            raise TypeError("single-operator E5 GPU UUIDs must be an array")
        descriptor = cls(**row, gpu_uuids=tuple(gpu_uuids))  # type: ignore[arg-type]
        if descriptor.sha256 != declared:
            raise ValueError("single-operator E5 failure descriptor digest differs")
        return descriptor


def current_formal_failure_execution_binding_sha256(
    subject: FormalFailureExecutionSubject,
) -> str:
    """Return the legacy-compatible binding digest for a public subject."""

    if type(subject) is not FormalFailureExecutionSubject:
        raise TypeError("current formal failure subject is not exact")
    subject.__post_init__()
    return content_sha256(
        {
            "protocol_sha256": FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256,
            "subject_sha256": subject.sha256,
            "serving_execution_binding_sha256": (
                subject.serving_execution_binding_sha256
            ),
        }
    )


def formal_single_operator_e5_failure_native_identities(
    *,
    prepared_launch_bundle_sha256: str,
    prepared_launch_entry_sha256: str,
    compile_launch_manifest_sha256: str,
    request_schedule_sha256: str,
    topology_mode: str,
    gpu_uuids: tuple[str, ...],
) -> tuple[str, str, str]:
    """Derive the fixed attempt-0 native identities used by plan and fault."""

    for label, value in (
        ("prepared bundle", prepared_launch_bundle_sha256),
        ("prepared entry", prepared_launch_entry_sha256),
        ("compile launch", compile_launch_manifest_sha256),
        ("request schedule", request_schedule_sha256),
    ):
        _sha256(f"single-operator E5 native {label}", value)
    run_nonce = content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_single_operator_e5_failure_run_nonce",
            "prepared_launch_bundle_sha256": prepared_launch_bundle_sha256,
            "prepared_launch_entry_sha256": prepared_launch_entry_sha256,
            "attempt": 0,
        }
    )
    execution_plan = content_sha256(
        {
            "compile_launch_manifest_sha256": compile_launch_manifest_sha256,
            "request_schedule_sha256": request_schedule_sha256,
        }
    )
    rank_config = content_sha256(
        {
            "topology_mode": topology_mode,
            "gpu_uuids": list(gpu_uuids),
        }
    )
    return run_nonce, execution_plan, rank_config


def _current_failure_subject(
    *,
    protocol_lock: ProtocolLock,
    runtime_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
    execution_binding_sha256: str,
    serving_execution_plan_sha256: str,
    serving_rank_config_sha256: str,
    run_nonce_sha256: str,
    inventory_sha256: str,
) -> FormalFailureExecutionSubject:
    authority, _failure_cells = _require_exact_e5_final_failure_materialization(
        materialization,
        protocol_lock_sha256=protocol_lock.sha256,
    )
    dimensions = dict(cell.dimensions)
    topology = dimensions.get("topology")
    scenario = dimensions.get("failure")
    cohort_count = dimensions.get("cohort_count")
    if (
        cell.method_role != "LightCone"
        or cell.task != "deterministic_failure_injection"
        or cell.publication_policy != "diagnostic_only"
        or dimensions.get("diagnostic_only") != "true"
        or dimensions.get("failure_authority_sha256") != authority.sha256
        or type(topology) is not str
        or type(scenario) is not str
        or type(cohort_count) is not int
    ):
        raise ValueError("single-operator E5 cell is not an authorized failure row")
    if (
        runtime_manifest.sha256
        != protocol_lock.formal_runtime_authority_manifest_sha256
    ):
        raise ValueError("single-operator E5 runtime authority differs from lock")
    actuator = runtime_manifest.member("failure_actuator")
    reducer = runtime_manifest.member("e5_failure_reducer")
    if (
        actuator.protocol_sha256 != FAILURE_ACTUATOR_PROTOCOL_SHA256
        or reducer.protocol_sha256 != E5_FAILURE_DIAGNOSTIC_PROTOCOL_SHA256
    ):
        raise ValueError("single-operator E5 runtime protocols differ")
    assignment = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_e5_failure_assignment",
            "protocol_sha256": FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256,
            "materialized_cell_id": cell.cell_id,
            "serving_execution_binding_sha256": execution_binding_sha256,
            "serving_execution_plan_sha256": serving_execution_plan_sha256,
            "scenario": scenario,
            "backend": cell.backend,
            "topology": topology,
            "cohort_count": cohort_count,
            "inventory_sha256": inventory_sha256,
            "run_nonce_sha256": run_nonce_sha256,
        }
    )
    return FormalFailureExecutionSubject(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
        materialization_receipt_sha256=materialization.sha256,
        materialized_cell_id=cell.cell_id,
        serving_execution_binding_sha256=execution_binding_sha256,
        serving_execution_plan_sha256=serving_execution_plan_sha256,
        serving_rank_config_sha256=serving_rank_config_sha256,
        assignment_sha256=assignment,
        inventory_sha256=inventory_sha256,
        registry_sha256=protocol_lock.registry_sha256,
        backend=cell.backend,
        topology=topology,
        scenario=scenario,
        cohort_count=cohort_count,
        run_nonce_sha256=run_nonce_sha256,
        failure_actuator_authority_sha256=actuator.sha256,
        failure_reducer_authority_sha256=reducer.sha256,
        correctness_only=True,
    )


def materialize_formal_single_operator_e5_failure_execution_descriptor(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    prepared_launch_bundle_path: str | Path,
    repository_root: str | Path,
    private_output_root: str | Path,
    current_ns: int,
) -> FormalSingleOperatorE5FailureExecutionDescriptor:
    """Publish one replayable public descriptor from the current E5 source."""

    from lightcone_spec.experiments.formal_registry import (
        formal_runtime_authority_manifest_to_dict,
        protocol_lock_from_dict,
        stage_materialization_receipt_from_dict,
    )
    from lightcone_spec.experiments.formal_runtime_manifest import (
        build_source_formal_runtime_authority_manifest,
    )
    from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
        formal_single_operator_prepared_execution_identities,
        revalidate_formal_single_operator_prepared_launch_bundle,
    )
    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        route_formal_single_operator_cell,
    )
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRequestScheduleReceipt,
    )
    from lightcone_spec.runtime.proof_artifact import (
        publish_canonical_json_no_replace,
    )

    validated = revalidate_formal_single_operator_prepared_launch_bundle(
        execution_source_path=execution_source_path,
        prepared_launch_bundle_path=prepared_launch_bundle_path,
        materialized_cell_id=materialized_cell_id,
        current_ns=current_ns,
    )
    source, cell, route = route_formal_single_operator_cell(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
    )
    entry = validated.entry(cell.cell_id)
    if (
        source.node != "e5_final"
        or route.physical_kind != "e5_failure"
        or entry.physical_kind != "e5_failure"
        or entry.request_schedule_receipt is None
    ):
        raise ValueError("single-operator E5 failure route differs")
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen()
    )
    protocol_lock = protocol_lock_from_dict(source.protocol_lock_source.reopen())
    _require_exact_e5_final_failure_materialization(
        materialization,
        protocol_lock_sha256=protocol_lock.sha256,
    )
    runtime = build_source_formal_runtime_authority_manifest(repository_root)
    if runtime.sha256 != source.runtime_authority_manifest_sha256:
        raise ValueError("single-operator E5 source/runtime identity differs")
    root = Path(private_output_root).resolve(strict=False)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("single-operator E5 output root differs")
    runtime_path = root / "formal-runtime-authority-manifest.json"
    publish_canonical_json_no_replace(
        runtime_path,
        formal_runtime_authority_manifest_to_dict(runtime),
    )
    runtime_binding = CanonicalJsonProofBinding.bind(runtime_path)
    schedule = FormalServingRequestScheduleReceipt.from_dict(
        entry.request_schedule_receipt.reopen()
    )
    execution_binding_sha256, subject_sha256 = (
        formal_single_operator_prepared_execution_identities(
            bundle=validated.bundle,
            entry=entry,
        )
    )
    run_nonce, execution_plan, rank_config = (
        formal_single_operator_e5_failure_native_identities(
            prepared_launch_bundle_sha256=validated.bundle.sha256,
            prepared_launch_entry_sha256=entry.sha256,
            compile_launch_manifest_sha256=(
                entry.compile_launch_manifest.semantic_sha256
            ),
            request_schedule_sha256=schedule.sha256,
            topology_mode=entry.topology_mode,
            gpu_uuids=entry.gpu_uuids,
        )
    )
    subject = _current_failure_subject(
        protocol_lock=protocol_lock,
        runtime_manifest=runtime,
        materialization=materialization,
        cell=cell,
        execution_binding_sha256=execution_binding_sha256,
        serving_execution_plan_sha256=execution_plan,
        serving_rank_config_sha256=rank_config,
        run_nonce_sha256=run_nonce,
        inventory_sha256=validated.inventory.sha256,
    )
    descriptor = FormalSingleOperatorE5FailureExecutionDescriptor(
        schema_version=1,
        kind="formal_single_operator_e5_failure_execution_descriptor",
        protocol_sha256=(FORMAL_SINGLE_OPERATOR_E5_FAILURE_EXECUTION_PROTOCOL_SHA256),
        execution_source=CanonicalJsonProofBinding.bind(execution_source_path),
        execution_source_sha256=source.sha256,
        prepared_launch_bundle=CanonicalJsonProofBinding.bind(
            prepared_launch_bundle_path
        ),
        prepared_launch_bundle_sha256=validated.bundle.sha256,
        prepared_launch_entry_sha256=entry.sha256,
        runtime_authority_manifest=runtime_binding,
        materialization=source.materialization_source,
        materialization_sha256=materialization.sha256,
        inventory=validated.bundle.inventory,
        compile_launch_manifest=entry.compile_launch_manifest,
        request_schedule_receipt=entry.request_schedule_receipt,
        execution_binding_sha256=execution_binding_sha256,
        subject_sha256=subject_sha256,
        failure_subject=subject,
        expected_failure_execution_binding_sha256=(
            current_formal_failure_execution_binding_sha256(subject)
        ),
        gpu_uuids=entry.gpu_uuids,
        attempt_id="attempt-0",
        retry_allowance=0,
        exclusive_timing=True,
        private_output_root=str(root),
    )
    output = root / "formal-single-operator-e5-failure-execution.json"
    publish_canonical_json_no_replace(output, descriptor.to_dict())
    return revalidate_formal_single_operator_e5_failure_execution_descriptor(
        output,
        current_ns=current_ns,
    )


def revalidate_formal_single_operator_e5_failure_execution_descriptor(
    path: str | Path,
    *,
    current_ns: int,
) -> FormalSingleOperatorE5FailureExecutionDescriptor:
    """Deep-replay a current public descriptor and its exact prepared row."""

    from lightcone_spec.experiments.formal_registry import (
        formal_runtime_authority_manifest_from_dict,
        protocol_lock_from_dict,
        stage_materialization_receipt_from_dict,
    )
    from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
        formal_single_operator_prepared_execution_identities,
        revalidate_formal_single_operator_prepared_launch_bundle,
    )
    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        route_formal_single_operator_cell,
    )
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRequestScheduleReceipt,
    )

    binding = CanonicalJsonProofBinding.bind(path)
    descriptor = FormalSingleOperatorE5FailureExecutionDescriptor.from_dict(
        binding.reopen()
    )
    validated = revalidate_formal_single_operator_prepared_launch_bundle(
        execution_source_path=descriptor.execution_source.absolute_path,
        prepared_launch_bundle_path=(descriptor.prepared_launch_bundle.absolute_path),
        materialized_cell_id=descriptor.failure_subject.materialized_cell_id,
        current_ns=current_ns,
    )
    source, cell, route = route_formal_single_operator_cell(
        execution_source_path=descriptor.execution_source.absolute_path,
        materialized_cell_id=descriptor.failure_subject.materialized_cell_id,
    )
    entry = validated.entry(cell.cell_id)
    materialization = stage_materialization_receipt_from_dict(
        descriptor.materialization.reopen(label="single-operator E5 materialization")
    )
    protocol_lock = protocol_lock_from_dict(source.protocol_lock_source.reopen())
    runtime = formal_runtime_authority_manifest_from_dict(
        descriptor.runtime_authority_manifest.reopen()
    )
    schedule = FormalServingRequestScheduleReceipt.from_dict(
        descriptor.request_schedule_receipt.reopen()
    )
    execution_binding_sha256, subject_sha256 = (
        formal_single_operator_prepared_execution_identities(
            bundle=validated.bundle,
            entry=entry,
        )
    )
    run_nonce, execution_plan, rank_config = (
        formal_single_operator_e5_failure_native_identities(
            prepared_launch_bundle_sha256=validated.bundle.sha256,
            prepared_launch_entry_sha256=entry.sha256,
            compile_launch_manifest_sha256=(
                entry.compile_launch_manifest.semantic_sha256
            ),
            request_schedule_sha256=schedule.sha256,
            topology_mode=entry.topology_mode,
            gpu_uuids=entry.gpu_uuids,
        )
    )
    expected_subject = _current_failure_subject(
        protocol_lock=protocol_lock,
        runtime_manifest=runtime,
        materialization=materialization,
        cell=cell,
        execution_binding_sha256=execution_binding_sha256,
        serving_execution_plan_sha256=execution_plan,
        serving_rank_config_sha256=rank_config,
        run_nonce_sha256=run_nonce,
        inventory_sha256=validated.inventory.sha256,
    )
    expected_path = (
        Path(descriptor.private_output_root)
        / "formal-single-operator-e5-failure-execution.json"
    )
    if (
        Path(binding.absolute_path) != expected_path
        or binding.semantic_sha256 != descriptor.sha256
        or source.node != "e5_final"
        or route.physical_kind != "e5_failure"
        or entry.physical_kind != "e5_failure"
        or descriptor.execution_source_sha256 != source.sha256
        or descriptor.prepared_launch_bundle_sha256 != validated.bundle.sha256
        or descriptor.prepared_launch_entry_sha256 != entry.sha256
        or descriptor.materialization != source.materialization_source
        or descriptor.materialization_sha256 != materialization.sha256
        or descriptor.inventory != validated.bundle.inventory
        or descriptor.compile_launch_manifest != entry.compile_launch_manifest
        or descriptor.request_schedule_receipt != entry.request_schedule_receipt
        or descriptor.execution_binding_sha256 != execution_binding_sha256
        or descriptor.subject_sha256 != subject_sha256
        or descriptor.failure_subject != expected_subject
        or descriptor.expected_failure_execution_binding_sha256
        != current_formal_failure_execution_binding_sha256(expected_subject)
        or descriptor.gpu_uuids != entry.gpu_uuids
    ):
        raise ValueError("single-operator E5 failure descriptor lineage differs")
    return descriptor


@dataclass(frozen=True)
class FormalFailureExecutionRebuildInput:
    """Public durable descriptor for one verifier-private failure token."""

    schema_version: Literal[1]
    kind: Literal["formal_failure_execution_rebuild_input"]
    protocol_sha256: str
    subject: FormalFailureExecutionSubject
    serving_execution_rebuild_input_sha256: str
    expected_failure_execution_binding_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_failure_execution_rebuild_input"
            or self.protocol_sha256 != FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256
        ):
            raise ValueError("formal failure rebuild descriptor is unsupported")
        if type(self.subject) is not FormalFailureExecutionSubject:
            raise TypeError("formal failure rebuild subject is not exact")
        _sha256(
            "formal failure serving rebuild input",
            self.serving_execution_rebuild_input_sha256,
        )
        _sha256(
            "formal failure expected binding",
            self.expected_failure_execution_binding_sha256,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "subject": self.subject.to_dict(),
            "serving_execution_rebuild_input_sha256": (
                self.serving_execution_rebuild_input_sha256
            ),
            "expected_failure_execution_binding_sha256": (
                self.expected_failure_execution_binding_sha256
            ),
        }
        if include_sha256:
            value["rebuild_input_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "rebuild_input_sha256",
        }:
            raise ValueError("formal failure rebuild descriptor fields differ")
        row = dict(value)
        declared = _sha256(
            "formal failure rebuild descriptor", row.pop("rebuild_input_sha256")
        )
        row["subject"] = FormalFailureExecutionSubject.from_dict(row["subject"])
        descriptor = cls(**row)  # type: ignore[arg-type]
        if descriptor.sha256 != declared:
            raise ValueError("formal failure rebuild descriptor digest differs")
        return descriptor


def bind_formal_failure_execution_rebuild_input(
    binding: VerifiedFormalFailureExecutionBinding,
    *,
    serving_execution_rebuild_input: FormalServingExecutionRebuildInput,
) -> FormalFailureExecutionRebuildInput:
    verified = require_verified_formal_failure_execution_binding(binding)
    if type(serving_execution_rebuild_input) is not (
        FormalServingExecutionRebuildInput
    ):
        raise TypeError("formal failure rebuild requires exact serving descriptor")
    if (
        serving_execution_rebuild_input.execution_binding_sha256
        != verified.serving_execution.sha256
    ):
        raise ValueError("formal failure serving rebuild descriptor differs")
    return FormalFailureExecutionRebuildInput(
        schema_version=1,
        kind="formal_failure_execution_rebuild_input",
        protocol_sha256=FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256,
        subject=verified.subject,
        serving_execution_rebuild_input_sha256=(serving_execution_rebuild_input.sha256),
        expected_failure_execution_binding_sha256=verified.sha256,
    )


def rebuild_formal_failure_execution_binding(
    descriptor: FormalFailureExecutionRebuildInput,
    *,
    serving_execution_rebuild_input: FormalServingExecutionRebuildInput,
    serving_execution: VerifiedFormalServingExecutionBinding,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
) -> VerifiedFormalFailureExecutionBinding:
    """Recompute the assignment after the serving token was publicly rebuilt."""

    if type(descriptor) is not FormalFailureExecutionRebuildInput:
        raise TypeError("formal failure rebuild requires exact typed descriptor")
    descriptor.__post_init__()
    if type(serving_execution_rebuild_input) is not (
        FormalServingExecutionRebuildInput
    ) or (
        serving_execution_rebuild_input.sha256
        != descriptor.serving_execution_rebuild_input_sha256
    ):
        raise ValueError("formal failure rebuild serving descriptor differs")
    rebuilt = bind_formal_failure_execution(
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=materialization,
        materialized_cell_id=descriptor.subject.materialized_cell_id,
        serving_execution=serving_execution,
    )
    if (
        rebuilt.subject != descriptor.subject
        or rebuilt.sha256 != descriptor.expected_failure_execution_binding_sha256
    ):
        raise ValueError("formal failure execution rebuild result differs")
    return rebuilt


__all__ = [
    "FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_E5_FAILURE_EXECUTION_PROTOCOL_SHA256",
    "FormalFailureExecutionRebuildInput",
    "FormalFailureExecutionSubject",
    "FormalSingleOperatorE5FailureExecutionDescriptor",
    "VerifiedFormalFailureExecutionBinding",
    "bind_formal_failure_execution",
    "bind_formal_failure_execution_rebuild_input",
    "current_formal_failure_execution_binding_sha256",
    "formal_single_operator_e5_failure_native_identities",
    "materialize_formal_single_operator_e5_failure_execution_descriptor",
    "rebuild_formal_failure_execution_binding",
    "require_verified_formal_failure_execution_binding",
    "revalidate_formal_single_operator_e5_failure_execution_descriptor",
]
