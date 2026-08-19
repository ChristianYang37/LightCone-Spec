"""Fail-closed bridge from signed stage rows to first-party GPU dispatch.

The signed materialization registry deliberately does not authorize execution.
This module is the narrow host-side boundary that joins it to the existing raw
budget, capacity, inventory, interference, and scheduler authorities.  It does
not deserialize an "authorized" token: the only authorized value is returned
after all raw authorities have been replayed and fresh dynamic control
challenges have been atomically reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_protocol import (
    ProtocolLock,
    content_sha256,
)
from lightcone_spec.experiments.formal_registry import (
    FormalRegistryManifest,
    FormalRegistryVerificationReceipt,
    formal_registry_verification_receipt_from_dict,
    formal_registry_verification_receipt_to_dict,
    signed_stage_materialization_from_dict,
    signed_stage_materialization_to_dict,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAssignment,
    GpuDispatchExecutionContext,
    GpuDispatchPlan,
    GpuDispatchWave,
    GpuInventory,
    validate_dispatch_plan_for_execution,
)
from lightcone_spec.experiments.planning import BudgetPlan
from lightcone_spec.experiments.planning_artifacts import (
    budget_plan_from_dict,
    budget_plan_to_dict,
)
from lightcone_spec.experiments.registry import (
    ExperimentCell,
    ExperimentRegistry,
    build_industrial_registry,
)
from lightcone_spec.experiments.stage_activation import (
    RegistryStageActivationArtifact,
    registry_stage_activation_from_dict,
    registry_stage_activation_to_dict,
    verify_registry_stage_activation,
)
from lightcone_spec.experiments.stage_capacity import (
    STAGE_CAPACITY_GATE_PROTOCOL_SHA256,
    StageCapacityGate,
    StageCapacitySchedule,
    bind_stage_capacity_schedule,
    revalidate_stage_capacity_gate_sources,
    stage_capacity_control_lineage_sha256,
)
from lightcone_spec.experiments.stage_materialization import (
    SignedStageMaterializationReceipt,
    StageMaterializationReceipt,
    materialize_preflight,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    VerifiedControlArtifact,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

PreflightRunnerKind = Literal[
    "first_party_compile",
    "first_party_exactness",
    "first_party_interference",
]


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _runner_for_cell(cell: ExperimentCell) -> PreflightRunnerKind:
    if cell.identity.task == "environment_and_patch_preflight":
        return "first_party_compile"
    if cell.identity.task == "exactness_memory_telemetry_preflight":
        return "first_party_exactness"
    if cell.identity.task == "simultaneous_single_gpu_interference":
        return "first_party_interference"
    raise ValueError("formal preflight cell has no first-party runner")


@dataclass(frozen=True)
class FormalPreflightExecutionBinding:
    """One exact materialized row bound to its scheduled physical assignment."""

    materialized_cell_id: str
    registry_cell_id: str
    runner_kind: PreflightRunnerKind
    work_item_sha256: str
    assignment_sha256: str
    experiment_budget_sha256: str
    source_authority_bindings: tuple[tuple[str, str], ...]
    cell: ExperimentCell
    assignment: GpuAssignment
    gpu_uuids: tuple[str, ...]
    rank_groups: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        for label, digest in (
            ("materialized cell", self.materialized_cell_id),
            ("registry cell", self.registry_cell_id),
            ("work item", self.work_item_sha256),
            ("physical assignment", self.assignment_sha256),
            ("experiment budget", self.experiment_budget_sha256),
        ):
            _require_sha256(f"formal preflight {label}", digest)
        if self.runner_kind not in {
            "first_party_compile",
            "first_party_exactness",
            "first_party_interference",
        }:
            raise ValueError("formal preflight runner is unsupported")
        names = {name for name, _ in self.source_authority_bindings}
        allowed_names = {
            frozenset(
                {
                    "compile_qualification",
                    "exactness_qualification",
                    "native_runtime_qualification",
                    "trusted_single_operator_content_bundle",
                }
            ),
            frozenset(
                {
                    "burstgpt_shape",
                    "compile_qualification",
                    "exactness_qualification",
                    "formal_workload_e0",
                    "formal_workload_e3a",
                    "native_runtime_qualification",
                    "offline_release_trust_root",
                    "prepared_model_content",
                }
            ),
        }
        if (
            self.source_authority_bindings
            != tuple(sorted(self.source_authority_bindings))
            or len(names) != len(self.source_authority_bindings)
            or frozenset(names) not in allowed_names
        ):
            raise ValueError("formal preflight source-authority binding is not exact")
        for name, digest in self.source_authority_bindings:
            if type(name) is not str or not name:
                raise ValueError("formal preflight source-authority name is invalid")
            _require_sha256(f"formal preflight source authority {name}", digest)
        if type(self.cell) is not ExperimentCell:
            raise TypeError("formal preflight binding requires an exact typed cell")
        if type(self.assignment) is not GpuAssignment:
            raise TypeError(
                "formal preflight binding requires an exact typed assignment"
            )
        if (
            self.registry_cell_id != self.cell.cell_id
            or self.assignment.work_item.cell != self.cell
            or self.work_item_sha256 != self.assignment.work_item.sha256
            or self.assignment_sha256 != self.assignment.sha256
            or self.gpu_uuids != self.assignment.gpu_uuids
            or self.rank_groups != self.assignment.rank_groups
        ):
            raise ValueError("formal preflight typed assignment binding differs")
        if (
            not self.gpu_uuids
            or len(self.gpu_uuids) != len(set(self.gpu_uuids))
            or tuple(uuid for group in self.rank_groups for uuid in group)
            != self.gpu_uuids
        ):
            raise ValueError("formal preflight assignment rank binding is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "materialized_cell_id": self.materialized_cell_id,
            "registry_cell_id": self.registry_cell_id,
            "runner_kind": self.runner_kind,
            "work_item_sha256": self.work_item_sha256,
            "assignment_sha256": self.assignment_sha256,
            "experiment_budget_sha256": self.experiment_budget_sha256,
            "source_authority_bindings": [
                {"name": name, "artifact_sha256": digest}
                for name, digest in self.source_authority_bindings
            ],
            "assignment": self.assignment.to_dict(),
            "gpu_uuids": list(self.gpu_uuids),
            "rank_groups": [list(group) for group in self.rank_groups],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "materialized_cell_id",
            "registry_cell_id",
            "runner_kind",
            "work_item_sha256",
            "assignment_sha256",
            "experiment_budget_sha256",
            "source_authority_bindings",
            "assignment",
            "gpu_uuids",
            "rank_groups",
        }:
            raise ValueError("formal preflight execution binding fields differ")
        raw_sources = value["source_authority_bindings"]
        raw_gpus = value["gpu_uuids"]
        raw_groups = value["rank_groups"]
        if (
            type(raw_sources) is not list
            or type(raw_gpus) is not list
            or type(raw_groups) is not list
        ):
            raise TypeError("formal preflight execution binding arrays are invalid")
        sources = []
        for raw_source in raw_sources:
            if type(raw_source) is not dict or set(raw_source) != {
                "name",
                "artifact_sha256",
            }:
                raise ValueError("formal preflight source binding fields differ")
            sources.append((raw_source["name"], raw_source["artifact_sha256"]))
        if any(type(group) is not list for group in raw_groups):
            raise TypeError("formal preflight rank groups are not arrays")
        assignment = GpuAssignment.from_dict(value["assignment"])
        result = cls(
            materialized_cell_id=value["materialized_cell_id"],
            registry_cell_id=value["registry_cell_id"],
            runner_kind=value["runner_kind"],
            work_item_sha256=value["work_item_sha256"],
            assignment_sha256=value["assignment_sha256"],
            experiment_budget_sha256=value["experiment_budget_sha256"],
            source_authority_bindings=tuple(sources),
            cell=assignment.work_item.cell,
            assignment=assignment,
            gpu_uuids=tuple(raw_gpus),
            rank_groups=tuple(tuple(group) for group in raw_groups),
        )
        return result


def _preflight_execution_bindings(
    materialization: StageMaterializationReceipt,
    *,
    protocol_lock: ProtocolLock,
    registry: ExperimentRegistry,
    dispatch_plan: GpuDispatchPlan,
) -> tuple[FormalPreflightExecutionBinding, ...]:
    """Bind all and only the ten signed preflight cells to scheduler rows."""

    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("formal dispatch requires an exact materialization")
    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("formal dispatch requires an exact ProtocolLock")
    if type(registry) is not ExperimentRegistry:
        raise TypeError("formal dispatch requires an exact registry")
    if type(dispatch_plan) is not GpuDispatchPlan:
        raise TypeError("formal dispatch requires an exact GPU dispatch plan")
    if (
        materialization.stage != "preflight"
        or materialization.expected_cell_count != 10
    ):
        raise ValueError("formal preflight dispatch requires exactly ten signed rows")
    if materialization.protocol_lock_sha256 != protocol_lock.sha256:
        raise ValueError("formal preflight materialization belongs to another lock")
    if registry.materialization_mode != "signed_staged":
        raise ValueError("legacy diagnostic registry cannot authorize formal dispatch")

    registered = {cell.cell_id: cell for cell in registry.cells_for("preflight")}
    materialized_by_registry: dict[str, str] = {}
    for cell in materialization.cells:
        dimensions = dict(cell.dimensions)
        if set(dimensions) - {
            "block",
            "concurrency",
            "context",
            "learning_rate",
            "regime",
            "registry_cell_id",
            "width",
        }:
            raise ValueError("formal preflight materialization has foreign dimensions")
        registry_cell_id = dimensions.get("registry_cell_id")
        _require_sha256("formal preflight registry-cell link", registry_cell_id)
        if registry_cell_id in materialized_by_registry:
            raise ValueError("formal preflight repeats a registry-cell link")
        materialized_by_registry[registry_cell_id] = cell.cell_id
    if set(materialized_by_registry) != set(registered):
        raise ValueError(
            "formal preflight materialization does not cover exact registry"
        )

    scheduled = tuple(
        assignment for wave in dispatch_plan.waves for assignment in wave.assignments
    )
    assignment_by_cell = {
        assignment.work_item.item_id: assignment for assignment in scheduled
    }
    if (
        len(assignment_by_cell) != len(scheduled)
        or set(assignment_by_cell) != set(registered)
        or dispatch_plan.completed_cell_ids
    ):
        raise ValueError(
            "formal preflight dispatch does not schedule all ten fresh cells"
        )
    budget_by_cell = dict(dispatch_plan.budget_sha256_by_cell)
    if set(budget_by_cell) != set(registered):
        raise ValueError("formal preflight dispatch lacks exact budget coverage")

    bindings = []
    for registry_cell_id, cell in registered.items():
        assignment = assignment_by_cell[registry_cell_id]
        if type(assignment) is not GpuAssignment:
            raise TypeError("formal preflight requires exact physical assignments")
        if assignment.work_item.cell != cell:
            raise ValueError("formal preflight assignment changes a registry cell")
        bindings.append(
            FormalPreflightExecutionBinding(
                materialized_cell_id=materialized_by_registry[registry_cell_id],
                registry_cell_id=registry_cell_id,
                runner_kind=_runner_for_cell(cell),
                work_item_sha256=assignment.work_item.sha256,
                assignment_sha256=assignment.sha256,
                experiment_budget_sha256=budget_by_cell[registry_cell_id],
                source_authority_bindings=(
                    protocol_lock.preflight_source_authority_bindings
                ),
                cell=cell,
                assignment=assignment,
                gpu_uuids=assignment.gpu_uuids,
                rank_groups=assignment.rank_groups,
            )
        )
    bindings = sorted(bindings, key=lambda row: row.registry_cell_id)
    counts = {
        runner: sum(row.runner_kind == runner for row in bindings)
        for runner in (
            "first_party_compile",
            "first_party_exactness",
            "first_party_interference",
        )
    }
    if counts != {
        "first_party_compile": 1,
        "first_party_exactness": 1,
        "first_party_interference": 8,
    }:
        raise ValueError("formal preflight runner coverage is not 1+1+8")
    return tuple(bindings)


@dataclass(frozen=True)
class FormalPreflightDispatchSubject:
    """Deterministic artifact that the final dispatch control must authorize."""

    schema_version: int
    kind: Literal["lightcone_formal_preflight_dispatch_subject"]
    manifest_sha256: str
    signed_materialization_sha256: str
    inventory_sha256: str
    dispatch_context_sha256: str
    dispatch_plan_sha256: str
    budget_plan_sha256: str
    capacity_schedule_sha256: str
    capacity_gate_sha256: str
    capacity_control_envelope_sha256: str
    execution_bindings: tuple[FormalPreflightExecutionBinding, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "lightcone_formal_preflight_dispatch_subject"
        ):
            raise ValueError("formal preflight dispatch subject schema is unsupported")
        for label, digest in (
            ("manifest", self.manifest_sha256),
            ("signed materialization", self.signed_materialization_sha256),
            ("inventory", self.inventory_sha256),
            ("dispatch context", self.dispatch_context_sha256),
            ("dispatch plan", self.dispatch_plan_sha256),
            ("budget plan", self.budget_plan_sha256),
            ("capacity schedule", self.capacity_schedule_sha256),
            ("capacity gate", self.capacity_gate_sha256),
            ("capacity control", self.capacity_control_envelope_sha256),
        ):
            _require_sha256(f"formal preflight dispatch {label}", digest)
        if (
            len(self.execution_bindings) != 10
            or self.execution_bindings
            != tuple(
                sorted(self.execution_bindings, key=lambda row: row.registry_cell_id)
            )
            or len({row.registry_cell_id for row in self.execution_bindings}) != 10
        ):
            raise ValueError("formal preflight dispatch bindings are not exact")

    @cached_property
    def lineage_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_formal_preflight_dispatch_lineage",
                "manifest_sha256": self.manifest_sha256,
                "inventory_sha256": self.inventory_sha256,
                "dispatch_context_sha256": self.dispatch_context_sha256,
                "dispatch_plan_sha256": self.dispatch_plan_sha256,
                "budget_plan_sha256": self.budget_plan_sha256,
                "capacity_schedule_sha256": self.capacity_schedule_sha256,
                "capacity_gate_sha256": self.capacity_gate_sha256,
                "capacity_control_envelope_sha256": (
                    self.capacity_control_envelope_sha256
                ),
            }
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_sha256": self.sha256,
            "schema_version": self.schema_version,
            "kind": self.kind,
            "manifest_sha256": self.manifest_sha256,
            "signed_materialization_sha256": self.signed_materialization_sha256,
            "inventory_sha256": self.inventory_sha256,
            "dispatch_context_sha256": self.dispatch_context_sha256,
            "dispatch_plan_sha256": self.dispatch_plan_sha256,
            "budget_plan_sha256": self.budget_plan_sha256,
            "capacity_schedule_sha256": self.capacity_schedule_sha256,
            "capacity_gate_sha256": self.capacity_gate_sha256,
            "capacity_control_envelope_sha256": (self.capacity_control_envelope_sha256),
            "execution_bindings": [row.to_dict() for row in self.execution_bindings],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "subject_sha256",
            "schema_version",
            "kind",
            "manifest_sha256",
            "signed_materialization_sha256",
            "inventory_sha256",
            "dispatch_context_sha256",
            "dispatch_plan_sha256",
            "budget_plan_sha256",
            "capacity_schedule_sha256",
            "capacity_gate_sha256",
            "capacity_control_envelope_sha256",
            "execution_bindings",
        }:
            raise ValueError("formal preflight dispatch subject fields differ")
        rows = value["execution_bindings"]
        if type(rows) is not list:
            raise TypeError("formal preflight execution bindings are not an array")
        subject = cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            manifest_sha256=value["manifest_sha256"],
            signed_materialization_sha256=value["signed_materialization_sha256"],
            inventory_sha256=value["inventory_sha256"],
            dispatch_context_sha256=value["dispatch_context_sha256"],
            dispatch_plan_sha256=value["dispatch_plan_sha256"],
            budget_plan_sha256=value["budget_plan_sha256"],
            capacity_schedule_sha256=value["capacity_schedule_sha256"],
            capacity_gate_sha256=value["capacity_gate_sha256"],
            capacity_control_envelope_sha256=value["capacity_control_envelope_sha256"],
            execution_bindings=tuple(
                FormalPreflightExecutionBinding.from_dict(row) for row in rows
            ),
        )
        if value["subject_sha256"] != subject.sha256:
            raise ValueError("formal preflight dispatch subject digest differs")
        return subject


_VERIFIED_FORMAL_PREFLIGHT_DISPATCH_SEAL = object()
_DURABLE_FORMAL_PREFLIGHT_CONTEXT_SEAL = object()


@dataclass(frozen=True, init=False)
class DurableFormalPreflightDispatchContext:
    """Minimal verifier-owned context restored from a durable dispatch receipt.

    GPU execution consumers need the exact registry, inventory and activation,
    while the signed subject binds the complete original execution-context
    identity.  Callers cannot construct this view directly.
    """

    registry: ExperimentRegistry
    inventory: GpuInventory
    activation_artifact: RegistryStageActivationArtifact
    budget_plan: BudgetPlan
    authority_sha256: str
    _construction_seal: object

    def __init__(
        self,
        *,
        registry: ExperimentRegistry,
        inventory: GpuInventory,
        activation_artifact: RegistryStageActivationArtifact,
        budget_plan: BudgetPlan,
        authority_sha256: str,
        _construction_seal: object,
    ) -> None:
        if _construction_seal is not _DURABLE_FORMAL_PREFLIGHT_CONTEXT_SEAL:
            raise TypeError("durable preflight context is verifier-constructed only")
        _require_sha256("durable preflight context", authority_sha256)
        if type(registry) is not ExperimentRegistry:
            raise TypeError("durable preflight context requires an exact registry")
        if type(inventory) is not GpuInventory:
            raise TypeError("durable preflight context requires an exact inventory")
        if type(activation_artifact) is not RegistryStageActivationArtifact:
            raise TypeError("durable preflight context requires an exact activation")
        if type(budget_plan) is not BudgetPlan:
            raise TypeError("durable preflight context requires an exact BudgetPlan")
        budget_plan.require_ready()
        for name, value in (
            ("registry", registry),
            ("inventory", inventory),
            ("activation_artifact", activation_artifact),
            ("budget_plan", budget_plan),
            ("authority_sha256", authority_sha256),
            ("_construction_seal", _construction_seal),
        ):
            object.__setattr__(self, name, value)

    @property
    def sha256(self) -> str:
        return self.authority_sha256


@dataclass(frozen=True, init=False)
class VerifiedFormalPreflightDispatch:
    """Sealed in-memory execution authority; callers cannot construct it."""

    manifest: FormalRegistryManifest
    protocol_lock: ProtocolLock
    subject: FormalPreflightDispatchSubject
    capacity_control: VerifiedControlArtifact
    dispatch_control: VerifiedControlArtifact
    dispatch_context: (
        GpuDispatchExecutionContext | DurableFormalPreflightDispatchContext
    )
    dispatch_plan: GpuDispatchPlan
    challenge_reservation_sha256: str
    _construction_seal: object

    def __init__(
        self,
        *,
        manifest: FormalRegistryManifest,
        protocol_lock: ProtocolLock,
        subject: FormalPreflightDispatchSubject,
        capacity_control: VerifiedControlArtifact,
        dispatch_control: VerifiedControlArtifact,
        dispatch_context: (
            GpuDispatchExecutionContext | DurableFormalPreflightDispatchContext
        ),
        dispatch_plan: GpuDispatchPlan,
        challenge_reservation_sha256: str,
        _construction_seal: object,
    ) -> None:
        if _construction_seal is not _VERIFIED_FORMAL_PREFLIGHT_DISPATCH_SEAL:
            raise TypeError(
                "formal preflight dispatch authority is verifier-constructed only"
            )
        _require_sha256(
            "formal preflight full challenge reservation",
            challenge_reservation_sha256,
        )
        for name, value in (
            ("manifest", manifest),
            ("protocol_lock", protocol_lock),
            ("subject", subject),
            ("capacity_control", capacity_control),
            ("dispatch_control", dispatch_control),
            ("dispatch_context", dispatch_context),
            ("dispatch_plan", dispatch_plan),
            ("challenge_reservation_sha256", challenge_reservation_sha256),
            ("_construction_seal", _construction_seal),
        ):
            object.__setattr__(self, name, value)

    @property
    def formal_dispatch_authorized(self) -> bool:
        require_verified_formal_preflight_dispatch(self)
        return True

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "manifest_sha256": self.manifest.sha256,
                "protocol_lock_sha256": self.protocol_lock.sha256,
                "subject_sha256": self.subject.sha256,
                "capacity_control_envelope_sha256": (
                    self.capacity_control.envelope_sha256
                ),
                "dispatch_control_envelope_sha256": (
                    self.dispatch_control.envelope_sha256
                ),
                "dispatch_context_sha256": self.dispatch_context.sha256,
                "dispatch_plan_sha256": self.dispatch_plan.sha256,
                "challenge_reservation_sha256": self.challenge_reservation_sha256,
            }
        )


def require_verified_formal_preflight_dispatch(
    value: object,
) -> VerifiedFormalPreflightDispatch:
    """Reject lookalikes and directly constructed dispatch capabilities."""

    if (
        type(value) is not VerifiedFormalPreflightDispatch
        or value._construction_seal is not _VERIFIED_FORMAL_PREFLIGHT_DISPATCH_SEAL
        or type(value.protocol_lock) is not ProtocolLock
        or value.protocol_lock.sha256 != value.manifest.protocol_lock_sha256
        or type(value.dispatch_context)
        not in {
            GpuDispatchExecutionContext,
            DurableFormalPreflightDispatchContext,
        }
        or (
            type(value.dispatch_context) is DurableFormalPreflightDispatchContext
            and value.dispatch_context._construction_seal
            is not _DURABLE_FORMAL_PREFLIGHT_CONTEXT_SEAL
        )
    ):
        raise TypeError("formal execution requires an exact sealed dispatch authority")
    return value


def _dispatch_plan_from_dict_structural(value: object) -> GpuDispatchPlan:
    """Decode a signed-subject plan without treating it as scheduler authority."""

    if type(value) is not dict:
        raise TypeError("formal preflight dispatch plan must be a JSON object")
    raw_budget = value.get("budget_sha256_by_cell")
    raw_waves = value.get("waves")
    raw_completed = value.get("completed_cell_ids")
    if (
        type(raw_budget) is not list
        or type(raw_waves) is not list
        or type(raw_completed) is not list
    ):
        raise TypeError("formal preflight dispatch plan arrays are invalid")
    budget_rows: list[tuple[str, str]] = []
    for raw in raw_budget:
        if type(raw) is not dict or set(raw) != {
            "cell_id",
            "experiment_budget_sha256",
        }:
            raise ValueError("formal preflight budget binding fields differ")
        budget_rows.append((raw["cell_id"], raw["experiment_budget_sha256"]))
    plan = GpuDispatchPlan(
        schema_version=value.get("schema_version"),
        registry_sha256=value.get("registry_sha256"),
        inventory_sha256=value.get("inventory_sha256"),
        receipts_sha256=value.get("receipts_sha256"),
        interference_envelope_sha256=value.get("interference_envelope_sha256"),
        budget_sha256_by_cell=tuple(budget_rows),
        seed=value.get("seed"),
        waves=tuple(GpuDispatchWave.from_dict(row) for row in raw_waves),
        completed_cell_ids=tuple(raw_completed),
    )
    if plan.to_dict() != value:
        raise ValueError("formal preflight dispatch plan summary differs from content")
    return plan


@dataclass(frozen=True)
class FormalPreflightDispatchReceipt:
    """Durable replay-bound form of one previously verified dispatch token.

    The receipt does not re-consume challenges.  It replays both signatures at
    the immutable reservation time, reopens the capacity source authority, and
    reconstructs the exact ten runner bindings before returning a private-sealed
    in-memory capability.
    """

    schema_version: int
    kind: Literal["lightcone_formal_preflight_dispatch_receipt"]
    verified_ns: int
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_materialization: SignedStageMaterializationReceipt
    inventory: GpuInventory
    activation: RegistryStageActivationArtifact
    dispatch_context_sha256: str
    budget_plan: BudgetPlan
    dispatch_plan: GpuDispatchPlan
    capacity_schedule: StageCapacitySchedule
    capacity_gate: StageCapacityGate
    capacity_control_attestation: ControlArtifactAttestation
    dispatch_control_attestation: ControlArtifactAttestation
    subject: FormalPreflightDispatchSubject
    reservation: ChallengeReplayReservationBinding

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.kind != (
            "lightcone_formal_preflight_dispatch_receipt"
        ):
            raise ValueError("formal preflight dispatch receipt schema is unsupported")
        if type(self.verified_ns) is not int or self.verified_ns < 1:
            raise ValueError("formal preflight dispatch receipt time is invalid")
        _require_sha256(
            "formal preflight dispatch context", self.dispatch_context_sha256
        )
        for expected_type, value, label in (
            (
                FormalRegistryVerificationReceipt,
                self.registry_verification_receipt,
                "registry verification receipt",
            ),
            (
                SignedStageMaterializationReceipt,
                self.signed_materialization,
                "materialization",
            ),
            (GpuInventory, self.inventory, "inventory"),
            (RegistryStageActivationArtifact, self.activation, "activation"),
            (BudgetPlan, self.budget_plan, "budget plan"),
            (GpuDispatchPlan, self.dispatch_plan, "dispatch plan"),
            (StageCapacitySchedule, self.capacity_schedule, "capacity schedule"),
            (StageCapacityGate, self.capacity_gate, "capacity gate"),
            (
                ControlArtifactAttestation,
                self.capacity_control_attestation,
                "capacity control",
            ),
            (
                ControlArtifactAttestation,
                self.dispatch_control_attestation,
                "dispatch control",
            ),
            (FormalPreflightDispatchSubject, self.subject, "dispatch subject"),
            (ChallengeReplayReservationBinding, self.reservation, "reservation"),
        ):
            if type(value) is not expected_type:
                raise TypeError(
                    f"formal preflight dispatch receipt {label} is not exact"
                )
        if self.verified_ns != self.reservation.reserved_ns:
            raise ValueError("formal preflight receipt time differs from reservation")
        self.budget_plan.require_ready()
        if (
            self.budget_plan.sha256 != self.capacity_schedule.budget_plan_sha256
            or self.budget_plan.sha256 != self.subject.budget_plan_sha256
        ):
            raise ValueError("formal preflight receipt BudgetPlan lineage differs")

    @property
    def sha256(self) -> str:
        return content_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "verified_ns": self.verified_ns,
            "registry_verification_receipt": (
                formal_registry_verification_receipt_to_dict(
                    self.registry_verification_receipt
                )
            ),
            "signed_materialization": signed_stage_materialization_to_dict(
                self.signed_materialization
            ),
            "inventory": self.inventory.to_dict(),
            "activation": registry_stage_activation_to_dict(self.activation),
            "dispatch_context_sha256": self.dispatch_context_sha256,
            "budget_plan": budget_plan_to_dict(self.budget_plan),
            "dispatch_plan": self.dispatch_plan.to_dict(),
            "capacity_schedule": self.capacity_schedule.to_dict(),
            "capacity_gate": self.capacity_gate.to_dict(),
            "capacity_control_attestation": (
                self.capacity_control_attestation.to_dict()
            ),
            "dispatch_control_attestation": self.dispatch_control_attestation.to_dict(),
            "subject": self.subject.to_dict(),
            "reservation": self.reservation.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {"receipt_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = frozenset(
            {
                "receipt_sha256",
                "schema_version",
                "kind",
                "verified_ns",
                "registry_verification_receipt",
                "signed_materialization",
                "inventory",
                "activation",
                "dispatch_context_sha256",
                "budget_plan",
                "dispatch_plan",
                "capacity_schedule",
                "capacity_gate",
                "capacity_control_attestation",
                "dispatch_control_attestation",
                "subject",
                "reservation",
            }
        )
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal preflight dispatch receipt fields differ")
        receipt = cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            verified_ns=value["verified_ns"],
            registry_verification_receipt=(
                formal_registry_verification_receipt_from_dict(
                    value["registry_verification_receipt"]
                )
            ),
            signed_materialization=signed_stage_materialization_from_dict(
                value["signed_materialization"]
            ),
            inventory=GpuInventory.from_dict(value["inventory"]),
            activation=registry_stage_activation_from_dict(value["activation"]),
            dispatch_context_sha256=value["dispatch_context_sha256"],
            budget_plan=budget_plan_from_dict(value["budget_plan"]),
            dispatch_plan=_dispatch_plan_from_dict_structural(value["dispatch_plan"]),
            capacity_schedule=StageCapacitySchedule.from_dict(
                value["capacity_schedule"]
            ),
            capacity_gate=StageCapacityGate.from_dict(value["capacity_gate"]),
            capacity_control_attestation=ControlArtifactAttestation.from_dict(
                value["capacity_control_attestation"]
            ),
            dispatch_control_attestation=ControlArtifactAttestation.from_dict(
                value["dispatch_control_attestation"]
            ),
            subject=FormalPreflightDispatchSubject.from_dict(value["subject"]),
            reservation=ChallengeReplayReservationBinding.from_dict(
                value["reservation"]
            ),
        )
        if value["receipt_sha256"] != receipt.sha256:
            raise ValueError("formal preflight dispatch receipt digest differs")
        return receipt

    def revalidate(self, *, current_ns: int) -> VerifiedFormalPreflightDispatch:
        if type(current_ns) is not int or current_ns < self.verified_ns:
            raise ValueError("formal preflight receipt current time is invalid")
        self.reservation.revalidate()
        manifest = self.registry_verification_receipt.revalidate(
            current_ns=self.verified_ns
        )
        registry = build_industrial_registry()
        if (
            manifest.registry_sha256 != registry.sha256
            or self.inventory.sha256 != self.subject.inventory_sha256
            or self.activation.registry_sha256 != registry.sha256
            or self.activation.experiment != "preflight"
            or self.activation.status != "AVAILABLE"
            or len(self.activation.activated_cell_ids) != 10
        ):
            raise ValueError("formal preflight durable raw authority differs")
        verify_registry_stage_activation(registry, self.activation)
        self.budget_plan.require_ready()
        if (
            self.registry_verification_receipt.cumulative_signed_materializations
            != (self.signed_materialization,)
            or self.registry_verification_receipt.cumulative_signed_coverage
            or self.budget_plan.sha256 != self.capacity_schedule.budget_plan_sha256
            or self.budget_plan.registry_sha256 != registry.sha256
            or self.budget_plan.inventory.sha256
            != self.capacity_schedule.budget_inventory_sha256
        ):
            raise ValueError("formal preflight durable registry prefix differs")
        protocol_lock = self.registry_verification_receipt.signed_protocol_lock.payload
        materialization = self.signed_materialization.payload
        if materialization != materialize_preflight(
            protocol_lock_sha256=protocol_lock.sha256,
            gpu_hours=materialization.gpu_hours,
        ):
            raise ValueError("formal preflight durable materialization is not exact")
        expected_bindings = _preflight_execution_bindings(
            materialization,
            protocol_lock=protocol_lock,
            registry=registry,
            dispatch_plan=self.dispatch_plan,
        )
        if self.subject != FormalPreflightDispatchSubject(
            schema_version=1,
            kind="lightcone_formal_preflight_dispatch_subject",
            manifest_sha256=manifest.sha256,
            signed_materialization_sha256=self.signed_materialization.sha256,
            inventory_sha256=self.inventory.sha256,
            dispatch_context_sha256=self.dispatch_context_sha256,
            dispatch_plan_sha256=self.dispatch_plan.sha256,
            budget_plan_sha256=self.capacity_schedule.budget_plan_sha256,
            capacity_schedule_sha256=self.capacity_schedule.sha256,
            capacity_gate_sha256=self.capacity_gate.sha256,
            capacity_control_envelope_sha256=(self.capacity_control_attestation.sha256),
            execution_bindings=expected_bindings,
        ):
            raise ValueError("formal preflight durable subject differs")
        if (
            self.capacity_gate.schedule_sha256 != self.capacity_schedule.sha256
            or self.capacity_gate.dispatch_plan_sha256 != self.dispatch_plan.sha256
            or self.capacity_gate.status != "AVAILABLE"
            or self.capacity_gate.mode != "SIGNED_STAGE_ENVELOPE"
        ):
            raise ValueError("formal preflight durable capacity differs")
        revalidate_stage_capacity_gate_sources(
            registry,
            self.capacity_gate,
            schedule=self.capacity_schedule,
            now_ns=self.verified_ns,
        )
        capacity_lineage = stage_capacity_control_lineage_sha256(
            activation_sha256=materialization.sha256,
            inventory_sha256=self.inventory.sha256,
            gate=self.capacity_gate,
        )
        _require_control_subject(
            self.capacity_control_attestation,
            artifact_type="capacity",
            artifact_sha256=self.capacity_gate.sha256,
            protocol_sha256=STAGE_CAPACITY_GATE_PROTOCOL_SHA256,
            registry_sha256=registry.sha256,
            lineage_sha256=capacity_lineage,
        )
        _require_control_subject(
            self.dispatch_control_attestation,
            artifact_type="dispatch",
            artifact_sha256=self.subject.sha256,
            protocol_sha256=protocol_lock.sha256,
            registry_sha256=registry.sha256,
            lineage_sha256=self.subject.lineage_sha256,
        )
        verified = tuple(
            verify_release_control_artifact_attestation(
                control,
                expected_inventory_sha256=self.inventory.sha256,
                now_ns=self.verified_ns,
                consumed_challenge_sha256s=(),
            )
            for control in (
                self.capacity_control_attestation,
                self.dispatch_control_attestation,
            )
        )
        expected_reservation = control_challenge_reservation_sha256(
            verified,
            reserved_ns=self.verified_ns,
        )
        if (
            expected_reservation != self.reservation.reservation_sha256
            or self.reservation.challenge_sha256s
            != tuple(
                sorted(
                    {
                        *(row.challenge_sha256 for row in verified),
                        *(row.deployment_policy_challenge_sha256 for row in verified),
                    }
                )
            )
        ):
            raise ValueError("formal preflight durable reservation differs")
        verified_by_envelope = {row.envelope_sha256: row for row in verified}
        context = DurableFormalPreflightDispatchContext(
            registry=registry,
            inventory=self.inventory,
            activation_artifact=self.activation,
            budget_plan=self.budget_plan,
            authority_sha256=self.dispatch_context_sha256,
            _construction_seal=_DURABLE_FORMAL_PREFLIGHT_CONTEXT_SEAL,
        )
        return VerifiedFormalPreflightDispatch(
            manifest=manifest,
            protocol_lock=protocol_lock,
            subject=self.subject,
            capacity_control=verified_by_envelope[
                self.capacity_control_attestation.sha256
            ],
            dispatch_control=verified_by_envelope[
                self.dispatch_control_attestation.sha256
            ],
            dispatch_context=context,
            dispatch_plan=self.dispatch_plan,
            challenge_reservation_sha256=expected_reservation,
            _construction_seal=_VERIFIED_FORMAL_PREFLIGHT_DISPATCH_SEAL,
        )


def publish_formal_preflight_dispatch_receipt(
    dispatch: VerifiedFormalPreflightDispatch,
    *,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_materialization: SignedStageMaterializationReceipt,
    capacity_schedule: StageCapacitySchedule,
    capacity_gate: StageCapacityGate,
    capacity_control_attestation: ControlArtifactAttestation,
    dispatch_control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    verified_ns: int,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish the sole durable boundary after successful atomic authorization."""

    token = require_verified_formal_preflight_dispatch(dispatch)
    if type(token.dispatch_context) is not GpuDispatchExecutionContext:
        raise TypeError("a durable dispatch receipt cannot mint another receipt")
    reservation = replay_store.bind_reservation(token.challenge_reservation_sha256)
    receipt = FormalPreflightDispatchReceipt(
        schema_version=2,
        kind="lightcone_formal_preflight_dispatch_receipt",
        verified_ns=verified_ns,
        registry_verification_receipt=registry_verification_receipt,
        signed_materialization=signed_materialization,
        inventory=token.dispatch_context.inventory,
        activation=token.dispatch_context.activation_artifact,
        dispatch_context_sha256=token.dispatch_context.sha256,
        budget_plan=token.dispatch_context.budget_plan,
        dispatch_plan=token.dispatch_plan,
        capacity_schedule=capacity_schedule,
        capacity_gate=capacity_gate,
        capacity_control_attestation=capacity_control_attestation,
        dispatch_control_attestation=dispatch_control_attestation,
        subject=token.subject,
        reservation=reservation,
    )
    reloaded = receipt.revalidate(current_ns=verified_ns)
    if reloaded.sha256 != token.sha256:
        raise RuntimeError("durable dispatch receipt changed the sealed token")
    publish_canonical_json_no_replace(output_path, receipt.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    if FormalPreflightDispatchReceipt.from_dict(binding.reopen()) != receipt:
        raise RuntimeError("written formal preflight dispatch receipt changed")
    return binding


def load_formal_preflight_dispatch_receipt(
    path: str | Path,
    *,
    current_ns: int,
) -> VerifiedFormalPreflightDispatch:
    binding = CanonicalJsonProofBinding.bind(path)
    receipt = FormalPreflightDispatchReceipt.from_dict(binding.reopen())
    return receipt.revalidate(current_ns=current_ns)


def _require_control_subject(
    control: ControlArtifactAttestation,
    *,
    artifact_type: str,
    artifact_sha256: str,
    protocol_sha256: str,
    registry_sha256: str,
    lineage_sha256: str,
) -> None:
    if type(control) is not ControlArtifactAttestation:
        raise TypeError("formal dispatch requires an exact dynamic control")
    subject = control.subject
    if (
        subject.artifact_type != artifact_type
        or subject.artifact_sha256 != artifact_sha256
        or subject.protocol_sha256 != protocol_sha256
        or subject.registry_sha256 != registry_sha256
        or subject.lineage_sha256 != lineage_sha256
    ):
        raise ValueError("formal dispatch control subject differs from exact artifact")


def authorize_formal_preflight_dispatch(
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    *,
    signed_materialization: SignedStageMaterializationReceipt,
    capacity_control_attestation: ControlArtifactAttestation,
    dispatch_control_attestation: ControlArtifactAttestation,
    dispatch_context: GpuDispatchExecutionContext,
    dispatch_plan: GpuDispatchPlan,
    capacity_schedule: StageCapacitySchedule,
    capacity_gate: StageCapacityGate,
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> VerifiedFormalPreflightDispatch:
    """Authorize the exact ten-cell preflight against every raw authority."""

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("formal preflight requires a durable registry verification")
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    signed_protocol_lock = registry_verification_receipt.signed_protocol_lock
    if type(signed_materialization) is not SignedStageMaterializationReceipt:
        raise TypeError("formal preflight requires an exact signed materialization")
    if type(dispatch_context) is not GpuDispatchExecutionContext:
        raise TypeError("formal preflight requires an execution-safe dispatch context")
    if type(dispatch_plan) is not GpuDispatchPlan:
        raise TypeError("formal preflight requires an exact dispatch plan")
    if type(capacity_schedule) is not StageCapacitySchedule:
        raise TypeError("formal preflight requires an exact capacity schedule")
    if type(capacity_gate) is not StageCapacityGate:
        raise TypeError("formal preflight requires an exact capacity gate")
    registry = build_industrial_registry()
    if dispatch_context.registry != registry:
        raise ValueError("formal preflight dispatch context uses another registry")
    inventory_sha256 = dispatch_context.inventory.sha256
    if (
        registry_verification_receipt.inventory_sha256 != inventory_sha256
        or registry_verification_receipt.cumulative_signed_materializations
        != (signed_materialization,)
        or registry_verification_receipt.cumulative_signed_coverage
        or manifest.prior_registry_verification_receipt_sha256 is None
    ):
        raise ValueError(
            "formal preflight dispatch requires the exact durable materialized prefix"
        )
    materialization = signed_materialization.payload
    expected_materialization = materialize_preflight(
        protocol_lock_sha256=signed_protocol_lock.payload.sha256,
        gpu_hours=materialization.gpu_hours,
    )
    if materialization != expected_materialization:
        raise ValueError("formal preflight materialization is not first-party exact")
    if manifest.status != "MATERIALIZED_PENDING_COVERAGE":
        raise ValueError("preflight cannot already claim terminal coverage at dispatch")

    validate_dispatch_plan_for_execution(
        dispatch_plan,
        execution_context=dispatch_context,
    )
    budget_plan = dispatch_context.budget_plan
    stage_cell_ids = tuple(
        sorted(cell.cell_id for cell in registry.cells_for("preflight"))
    )
    if (
        budget_plan.status != "READY"
        or budget_plan.activated_cell_ids != stage_cell_ids
        or dispatch_plan.completed_cell_ids
    ):
        raise ValueError(
            "formal preflight budget/dispatch set is not fresh and complete"
        )
    expected_schedule = bind_stage_capacity_schedule(
        registry,
        experiment="preflight",
        activated_cell_ids=stage_cell_ids,
        dispatch_plan=dispatch_plan,
        budget_plan=budget_plan,
    )
    if capacity_schedule != expected_schedule:
        raise ValueError("formal preflight capacity schedule is not first-party exact")
    if (
        capacity_gate.schema_version != 3
        or capacity_gate.experiment != "preflight"
        or capacity_gate.registry_sha256 != registry.sha256
        or capacity_gate.activated_cell_ids != stage_cell_ids
        or capacity_gate.mode != "SIGNED_STAGE_ENVELOPE"
        or capacity_gate.status != "AVAILABLE"
        or capacity_gate.schedule_sha256 != capacity_schedule.sha256
        or capacity_gate.dispatch_plan_sha256 != dispatch_plan.sha256
        or capacity_gate.budget_plan_sha256 != budget_plan.sha256
    ):
        raise ValueError(
            "formal preflight capacity gate is not AVAILABLE for exact plan"
        )
    revalidate_stage_capacity_gate_sources(
        registry,
        capacity_gate,
        schedule=capacity_schedule,
        now_ns=now_ns,
    )

    capacity_lineage = stage_capacity_control_lineage_sha256(
        activation_sha256=materialization.sha256,
        inventory_sha256=inventory_sha256,
        gate=capacity_gate,
    )
    _require_control_subject(
        capacity_control_attestation,
        artifact_type="capacity",
        artifact_sha256=capacity_gate.sha256,
        protocol_sha256=STAGE_CAPACITY_GATE_PROTOCOL_SHA256,
        registry_sha256=registry.sha256,
        lineage_sha256=capacity_lineage,
    )
    if (
        capacity_control_attestation.deployment_policy_authorization.root_manifest_sha256
        != signed_protocol_lock.payload.offline_release_trust_root_sha256
        or capacity_control_attestation.trusted_attester_policy_sha256
        != manifest.trusted_attester_policy_sha256
    ):
        raise ValueError("capacity and formal registry controls use different policies")

    bindings = _preflight_execution_bindings(
        materialization,
        protocol_lock=signed_protocol_lock.payload,
        registry=registry,
        dispatch_plan=dispatch_plan,
    )
    subject = FormalPreflightDispatchSubject(
        schema_version=1,
        kind="lightcone_formal_preflight_dispatch_subject",
        manifest_sha256=manifest.sha256,
        signed_materialization_sha256=signed_materialization.sha256,
        inventory_sha256=inventory_sha256,
        dispatch_context_sha256=dispatch_context.sha256,
        dispatch_plan_sha256=dispatch_plan.sha256,
        budget_plan_sha256=budget_plan.sha256,
        capacity_schedule_sha256=capacity_schedule.sha256,
        capacity_gate_sha256=capacity_gate.sha256,
        capacity_control_envelope_sha256=capacity_control_attestation.sha256,
        execution_bindings=bindings,
    )
    _require_control_subject(
        dispatch_control_attestation,
        artifact_type="dispatch",
        artifact_sha256=subject.sha256,
        protocol_sha256=signed_protocol_lock.payload.sha256,
        registry_sha256=registry.sha256,
        lineage_sha256=subject.lineage_sha256,
    )
    if (
        dispatch_control_attestation.deployment_policy_authorization.sha256
        != capacity_control_attestation.deployment_policy_authorization.sha256
        or dispatch_control_attestation.trusted_attester_policy_sha256
        != manifest.trusted_attester_policy_sha256
    ):
        raise ValueError("dispatch and formal registry controls use different policies")
    all_controls = (capacity_control_attestation, dispatch_control_attestation)
    verified = verify_and_reserve_release_control_artifact_attestations(
        all_controls,
        expected_inventory_sha256=inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )
    verified_by_envelope = {row.envelope_sha256: row for row in verified}
    verified_capacity = verified_by_envelope[capacity_control_attestation.sha256]
    verified_dispatch = verified_by_envelope[dispatch_control_attestation.sha256]
    reservation_sha256 = control_challenge_reservation_sha256(
        verified,
        reserved_ns=now_ns,
    )
    return VerifiedFormalPreflightDispatch(
        manifest=manifest,
        protocol_lock=signed_protocol_lock.payload,
        subject=subject,
        capacity_control=verified_capacity,
        dispatch_control=verified_dispatch,
        dispatch_context=dispatch_context,
        dispatch_plan=dispatch_plan,
        challenge_reservation_sha256=reservation_sha256,
        _construction_seal=_VERIFIED_FORMAL_PREFLIGHT_DISPATCH_SEAL,
    )


__all__ = [
    "FormalPreflightDispatchReceipt",
    "FormalPreflightDispatchSubject",
    "FormalPreflightExecutionBinding",
    "PreflightRunnerKind",
    "VerifiedFormalPreflightDispatch",
    "authorize_formal_preflight_dispatch",
    "load_formal_preflight_dispatch_receipt",
    "publish_formal_preflight_dispatch_receipt",
    "require_verified_formal_preflight_dispatch",
]
