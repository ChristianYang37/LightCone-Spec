"""Fail-closed capacity and GPU-hour admission for formal GPU launches.

The physical serving runners intentionally know how to start a process but do
not decide whether a process may be started.  This module is that decision
boundary.  It joins one exact materialization cell and run plan to fresh raw
capacity observations, a release-controlled capacity gate, and exactly one of
the two registered GPU-hour authorities:

* an AVAILABLE durable stage GPU-hour receipt; or
* the deterministic minimum-pilot bootstrap budget for an uncovered stratum.

Every accepted admission is path-bound, replay-bound, and single-use.  The
consumer claim is written with no-replace semantics before any GPU snapshot,
port reservation, output allocation, or child process creation.
"""

from __future__ import annotations

import fcntl
import math
import os
import stat
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from lightcone_spec.experiments.capacity_authority import (
    UnsignedCapacitySourceReplay,
    replay_unsigned_capacity_source_manifest,
)
from lightcone_spec.experiments.formal_gpu_hour_registry import (
    FormalStageGpuHourVerificationReceipt,
)
from lightcone_spec.experiments.formal_protocol import (
    FORMAL_STAGE_DAG,
    content_sha256,
)
from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_from_dict,
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.formal_stage_execution import (
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.gpu_hour_authority import (
    StagedProspectiveGpuHourSourceManifest,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_capacity import (
    STAGE_CAPACITY_MINIMUM_SAFETY_MARGIN_BYTES,
    STAGE_CAPACITY_SAFETY_MARGIN_BPS,
    StageCapacitySchedule,
)
from lightcone_spec.experiments.stage_materialization import (
    MaterializedCell,
    StageMaterializationReceipt,
)
from lightcone_spec.orchestration.formal_dynamic_dispatch import (
    FormalDynamicStageCapacitySchedule,
    FormalPilotDynamicDispatchSchedule,
    revalidate_formal_dynamic_stage_capacity_schedule,
)
from lightcone_spec.orchestration.formal_e5_dynamic_dispatch import (
    FormalE5OneShotStageCapacitySchedule,
    revalidate_formal_e5_one_shot_stage_capacity_schedule,
)
from lightcone_spec.orchestration.formal_e5_launch_budget import (
    FormalE5OneShotBudgetVerificationReceipt,
    FormalE5OneShotLaunchBudget,
    revalidate_formal_e5_one_shot_budget_verification,
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

FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE = 1
FORMAL_PILOT_CELL_HARD_TIMEOUT_NS = 3_600_000_000_000
FORMAL_PILOT_BUDGET_RESERVE_BPS = 2_000
FORMAL_PILOT_BUDGET_MAXIMUM_LIFETIME_NS = 86_400_000_000_000
NANOSECONDS_PER_GPU_HOUR = 3_600_000_000_000

FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "lightcone_formal_stage_capacity_protocol",
        "materialization": "path_bound_exact_formal_cells",
        "schedule": (
            "source_owned_dynamic_materialized_cell_wave_from_typed_launch_caps"
        ),
        "retained_evidence": "all_materialized_cells_times_retry_plus_one",
        "transient": "exact_launch_wave_model_staging_plus_compile_overlay",
        "raw_source": "fresh_deep_replayed_capacity_manifest",
        "authorization": "release_capacity_control_plus_atomic_replay",
    }
)
FORMAL_PILOT_LAUNCH_BUDGET_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_pilot_launch_budget_protocol",
        "authority": (
            "blocked_staged_prospective_source_exact_minimum_stratum_pilots_"
            "plus_available_preflight_budget"
        ),
        "timeout_ns": FORMAL_PILOT_CELL_HARD_TIMEOUT_NS,
        "retry_allowance": FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE,
        "provider_reservation": ("exact_per_cell_fixed_two_gpu_instance_upper_bound"),
        "reserve_basis_points": FORMAL_PILOT_BUDGET_RESERVE_BPS,
        "full_stage_projection": "forbidden",
        "authorization": "release_rank_aggregate_control_and_single_use",
    }
)
FORMAL_PILOT_BUDGET_VERIFICATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_pilot_budget_verification_protocol",
        "control": "budget_challenge_reserved_exactly_once",
        "launches": "fresh_capacity_control_plus_append_only_cell_attempt_ledger",
        "retry_allowance": FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE,
    }
)
FORMAL_STAGE_LAUNCH_ADMISSION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_stage_launch_admission_protocol",
        "capacity": FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256,
        "budget_union": (
            "available_stage_gpu_hour_receipt",
            "minimum_pilot_bootstrap_durable_budget_verification",
            "registered_e5_one_shot_durable_budget_verification",
        ),
        "binding": (
            "materialization_cell_plan_topology_inventory_registry_runtime_root"
        ),
        "single_use": "atomic_no_replace_pre_allocation_consumer_claim",
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


def _positive_int(label: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _strict(label: str, value: object, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _materialization(binding: CanonicalJsonProofBinding) -> StageMaterializationReceipt:
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("formal launch materialization binding changed")
    receipt = stage_materialization_receipt_from_dict(binding.reopen())
    if receipt.sha256 != binding.semantic_sha256:
        raise ValueError("formal launch materialization semantic digest differs")
    return receipt


def _materialization_cell(
    materialization: StageMaterializationReceipt, cell_id: str
) -> MaterializedCell:
    cells = tuple(row for row in materialization.cells if row.cell_id == cell_id)
    if len(cells) != 1:
        raise ValueError("formal launch cell is outside the materialization")
    return cells[0]


def _formal_cell_gpu_count(cell: MaterializedCell) -> int:
    dimensions = dict(cell.dimensions)
    topology = dimensions.get("topology")
    if topology in {"tp2_dp1", "tp1_dp2", "two_replica_tp1_dp2"}:
        return 2
    if topology in {"tp1_dp1", None}:
        registry_cell_id = dimensions.get("registry_cell_id")
        if registry_cell_id is not None:
            source = tuple(
                row
                for row in build_industrial_registry().cells_for(cell.stage)
                if row.cell_id == registry_cell_id
            )
            if len(source) != 1:
                raise ValueError("pilot cell registry source is not exact")
            return source[0].resources.gpu_count
        return 1
    raise ValueError("pilot cell topology is not registered")


@dataclass(frozen=True)
class FormalStageCapacitySchedule:
    """One exact formal launch wave plus all-cell retained-evidence sizing."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_stage_capacity_schedule"]
    protocol_sha256: str
    stage: str
    protocol_lock_sha256: str
    runtime_authority_manifest_sha256: str
    registry_sha256: str
    materialization: CanonicalJsonProofBinding
    materialization_receipt_sha256: str
    stage_capacity_schedule: CanonicalJsonProofBinding
    stage_capacity_schedule_sha256: str
    activated_cell_ids: tuple[str, ...]
    materialized_cell_id: str
    execution_binding_sha256: str
    execution_plan_sha256: str
    run_plan: CanonicalJsonProofBinding
    run_plan_sha256: str
    topology_mode: str
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    wave_index: int
    wave_cell_ids: tuple[str, ...]
    provider_inventory_gpu_count: Literal[2]
    provider_reserved_gpu_count: Literal[1, 2]
    retry_allowance: int
    capacity_source_manifest: CanonicalJsonProofBinding
    capacity_envelope_sha256: str
    budget_inventory_sha256: str
    capacity_captured_at_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_stage_capacity_schedule"
            or self.protocol_sha256 != FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256
            or self.stage not in FORMAL_STAGE_DAG
        ):
            raise ValueError("formal stage capacity schedule schema is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime authority", self.runtime_authority_manifest_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("stage capacity schedule", self.stage_capacity_schedule_sha256),
            ("materialized cell", self.materialized_cell_id),
            ("execution binding", self.execution_binding_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("run plan", self.run_plan_sha256),
            ("inventory", self.inventory_sha256),
            ("capacity envelope", self.capacity_envelope_sha256),
            ("budget inventory", self.budget_inventory_sha256),
        ):
            _sha256(f"formal capacity schedule {label}", digest)
        if (
            type(self.materialization) is not CanonicalJsonProofBinding
            or type(self.stage_capacity_schedule) is not CanonicalJsonProofBinding
            or type(self.run_plan) is not CanonicalJsonProofBinding
            or type(self.capacity_source_manifest) is not CanonicalJsonProofBinding
        ):
            raise TypeError("formal capacity schedule sources must be path-bound")
        if (
            not self.activated_cell_ids
            or self.activated_cell_ids != tuple(sorted(set(self.activated_cell_ids)))
            or self.materialized_cell_id not in self.activated_cell_ids
            or type(self.wave_index) is not int
            or self.wave_index < 0
            or self.retry_allowance != FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE
        ):
            raise ValueError("formal capacity schedule cell/wave coverage differs")
        if self.topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
            raise ValueError("formal capacity schedule topology is unsupported")
        expected_gpus = 1 if self.topology_mode == "tp1_dp1" else 2
        if (
            len(self.gpu_uuids) != expected_gpus
            or len(set(self.gpu_uuids)) != expected_gpus
        ):
            raise ValueError("formal capacity schedule GPU gang differs")
        if (
            self.provider_inventory_gpu_count != 2
            or self.materialized_cell_id not in self.wave_cell_ids
            or self.wave_cell_ids != tuple(sorted(set(self.wave_cell_ids)))
            or not set(self.wave_cell_ids) <= set(self.activated_cell_ids)
            or not 1 <= len(self.wave_cell_ids) <= 2
            or (
                self.topology_mode == "tp1_dp1"
                and self.provider_reserved_gpu_count
                != self.provider_inventory_gpu_count // len(self.wave_cell_ids)
            )
            or (
                self.topology_mode != "tp1_dp1"
                and (
                    self.wave_cell_ids != (self.materialized_cell_id,)
                    or self.provider_reserved_gpu_count != 2
                )
            )
        ):
            raise ValueError("formal capacity provider reservation wave differs")
        _nonnegative_int(
            "formal capacity schedule capture time", self.capacity_captured_at_ns
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "stage": self.stage,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "runtime_authority_manifest_sha256": (
                self.runtime_authority_manifest_sha256
            ),
            "registry_sha256": self.registry_sha256,
            "materialization": self.materialization.to_dict(),
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "stage_capacity_schedule": self.stage_capacity_schedule.to_dict(),
            "stage_capacity_schedule_sha256": (self.stage_capacity_schedule_sha256),
            "activated_cell_ids": list(self.activated_cell_ids),
            "materialized_cell_id": self.materialized_cell_id,
            "execution_binding_sha256": self.execution_binding_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "run_plan": self.run_plan.to_dict(),
            "run_plan_sha256": self.run_plan_sha256,
            "topology_mode": self.topology_mode,
            "inventory_sha256": self.inventory_sha256,
            "gpu_uuids": list(self.gpu_uuids),
            "wave_index": self.wave_index,
            "wave_cell_ids": list(self.wave_cell_ids),
            "provider_inventory_gpu_count": self.provider_inventory_gpu_count,
            "provider_reserved_gpu_count": self.provider_reserved_gpu_count,
            "retry_allowance": self.retry_allowance,
            "capacity_source_manifest": self.capacity_source_manifest.to_dict(),
            "capacity_envelope_sha256": self.capacity_envelope_sha256,
            "budget_inventory_sha256": self.budget_inventory_sha256,
            "capacity_captured_at_ns": self.capacity_captured_at_ns,
            "schedule_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal stage capacity schedule",
            value,
            {*cls.__dataclass_fields__, "schedule_sha256"},
        )
        declared = _sha256("formal stage capacity schedule", row.pop("schedule_sha256"))
        for name in ("activated_cell_ids", "gpu_uuids", "wave_cell_ids"):
            raw = row[name]
            if type(raw) is not list or any(type(item) is not str for item in raw):
                raise TypeError(f"formal capacity schedule {name} must be an array")
            row[name] = tuple(raw)
        for name in (
            "materialization",
            "stage_capacity_schedule",
            "run_plan",
            "capacity_source_manifest",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        schedule = cls(**row)
        if schedule.sha256 != declared:
            raise ValueError("formal stage capacity schedule digest differs")
        return schedule


def _replay_capacity(
    schedule: FormalStageCapacitySchedule, *, now_ns: int
) -> UnsignedCapacitySourceReplay:
    if (
        CanonicalJsonProofBinding.bind(schedule.stage_capacity_schedule.absolute_path)
        != schedule.stage_capacity_schedule
    ):
        raise ValueError("formal first-party stage capacity schedule changed")
    stage_schedule = StageCapacitySchedule.from_dict(
        schedule.stage_capacity_schedule.reopen()
    )
    matching_waves = tuple(
        row
        for row in stage_schedule.waves
        if schedule.materialized_cell_id in row.cell_ids
    )
    retry_by_cell = {row.cell_id: row.retry_allowance for row in stage_schedule.retries}
    if (
        stage_schedule.sha256 != schedule.stage_capacity_schedule_sha256
        or stage_schedule.registry_sha256 != schedule.registry_sha256
        or stage_schedule.experiment != schedule.stage
        or stage_schedule.activated_cell_ids != schedule.activated_cell_ids
        or stage_schedule.gpu_inventory_sha256 != schedule.inventory_sha256
        or stage_schedule.capacity_envelope_sha256 != schedule.capacity_envelope_sha256
        or len(matching_waves) != 1
        or matching_waves[0].wave_index != schedule.wave_index
        or matching_waves[0].cell_ids != schedule.wave_cell_ids
        or retry_by_cell
        != {
            cell_id: FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE
            for cell_id in schedule.activated_cell_ids
        }
    ):
        raise ValueError("formal first-party stage capacity schedule differs")
    if (
        CanonicalJsonProofBinding.bind(schedule.capacity_source_manifest.absolute_path)
        != schedule.capacity_source_manifest
    ):
        raise ValueError("formal capacity raw source manifest changed")
    replay = replay_unsigned_capacity_source_manifest(
        schedule.capacity_source_manifest.absolute_path,
        expected_registry_sha256=schedule.registry_sha256,
        now_ns=now_ns,
    )
    requirements = tuple(
        row.cell_id for row in replay.capacity_envelope.cell_requirements
    )
    if (
        replay.capacity_envelope.sha256 != schedule.capacity_envelope_sha256
        or replay.budget_inventory.sha256 != schedule.budget_inventory_sha256
        or replay.gpu_inventory.sha256 != schedule.inventory_sha256
        or replay.captured_at_ns != schedule.capacity_captured_at_ns
        or requirements != schedule.activated_cell_ids
    ):
        raise ValueError("formal capacity raw replay differs from the schedule")
    return replay


def _load_capacity_schedule(
    path: str | Path,
) -> (
    FormalStageCapacitySchedule
    | FormalDynamicStageCapacitySchedule
    | FormalE5OneShotStageCapacitySchedule
):
    binding = CanonicalJsonProofBinding.bind(path)
    value = binding.reopen()
    if type(value) is not dict:
        raise TypeError("formal capacity schedule must be a JSON object")
    kind = value.get("kind")
    if kind == "lightcone_formal_dynamic_stage_capacity_schedule":
        schedule = FormalDynamicStageCapacitySchedule.from_dict(value)
    elif kind == "lightcone_formal_e5_one_shot_stage_capacity_schedule":
        schedule = FormalE5OneShotStageCapacitySchedule.from_dict(value)
    elif kind == "lightcone_formal_stage_capacity_schedule":
        schedule = FormalStageCapacitySchedule.from_dict(value)
    else:
        raise ValueError("formal capacity schedule kind is unsupported")
    if schedule.sha256 != binding.semantic_sha256:
        raise ValueError("formal capacity schedule semantic binding differs")
    return schedule


def _replay_capacity_schedule(
    schedule: (
        FormalStageCapacitySchedule
        | FormalDynamicStageCapacitySchedule
        | FormalE5OneShotStageCapacitySchedule
    ),
    *,
    now_ns: int,
) -> UnsignedCapacitySourceReplay:
    if type(schedule) is FormalDynamicStageCapacitySchedule:
        return revalidate_formal_dynamic_stage_capacity_schedule(
            schedule, current_ns=now_ns
        )
    if type(schedule) is FormalE5OneShotStageCapacitySchedule:
        return revalidate_formal_e5_one_shot_stage_capacity_schedule(
            schedule,
            current_ns=now_ns,
        )
    if type(schedule) is FormalStageCapacitySchedule:
        return _replay_capacity(schedule, now_ns=now_ns)
    raise TypeError("formal capacity schedule type is unsupported")


def materialize_formal_stage_capacity_schedule(
    *,
    execution_binding: VerifiedFormalServingExecutionBinding,
    run_plan_path: str | Path,
    materialization_path: str | Path,
    stage_capacity_schedule_path: str | Path,
    capacity_source_manifest_path: str | Path,
    output_path: str | Path,
    now_ns: int,
) -> FormalStageCapacitySchedule:
    """Publish the only launch schedule accepted by the formal capacity gate."""

    from lightcone_spec.orchestration.formal_physical_dispatch import (
        load_formal_serving_run_plan,
    )

    verified = require_verified_formal_serving_execution_binding(execution_binding)
    plan = load_formal_serving_run_plan(
        run_plan_path,
        execution_binding=verified,
        verified_nextn_tp2_authority=verified.verified_nextn_tp2_authority,
    )
    materialization_binding = CanonicalJsonProofBinding.bind(materialization_path)
    materialization = _materialization(materialization_binding)
    stage_schedule_binding = CanonicalJsonProofBinding.bind(
        stage_capacity_schedule_path
    )
    stage_schedule = StageCapacitySchedule.from_dict(stage_schedule_binding.reopen())
    subject = verified.subject
    _materialization_cell(materialization, subject.materialized_cell_id)
    if (
        materialization.sha256 != subject.materialization_receipt_sha256
        or materialization.stage != subject.stage
        or plan.materialized_cell_id != subject.materialized_cell_id
        or plan.execution_binding_sha256 != verified.sha256
    ):
        raise ValueError("formal capacity schedule immutable execution differs")
    source_binding = CanonicalJsonProofBinding.bind(capacity_source_manifest_path)
    replay = replay_unsigned_capacity_source_manifest(
        source_binding.absolute_path,
        expected_registry_sha256=subject.execution_identity.registry_sha256,
        now_ns=now_ns,
    )
    cell_ids = tuple(row.cell_id for row in materialization.cells)
    matching_waves = tuple(
        wave
        for wave in stage_schedule.waves
        if subject.materialized_cell_id in wave.cell_ids
    )
    if len(matching_waves) != 1:
        raise ValueError("formal capacity source schedule lacks exact cell wave")
    wave = matching_waves[0]
    materialized_cells = {row.cell_id: row for row in materialization.cells}
    if (
        stage_schedule.registry_sha256 != subject.execution_identity.registry_sha256
        or stage_schedule.experiment != subject.stage
        or stage_schedule.activated_cell_ids != cell_ids
        or stage_schedule.gpu_inventory_sha256 != subject.inventory_sha256
        or stage_schedule.capacity_envelope_sha256 != replay.capacity_envelope.sha256
        or tuple(row.cell_id for row in stage_schedule.retries) != cell_ids
        or any(
            row.retry_allowance != FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE
            for row in stage_schedule.retries
        )
        or len(wave.cell_ids) > 2
        or any(
            _formal_cell_gpu_count(materialized_cells[cell_id]) != 1
            for cell_id in wave.cell_ids
            if len(wave.cell_ids) == 2
        )
        or (
            subject.topology_mode != "tp1_dp1"
            and wave.cell_ids != (subject.materialized_cell_id,)
        )
    ):
        raise ValueError("formal capacity first-party stage wave differs")
    if (
        tuple(row.cell_id for row in replay.capacity_envelope.cell_requirements)
        != cell_ids
        or replay.gpu_inventory.sha256 != subject.inventory_sha256
    ):
        raise ValueError(
            "formal capacity sizing does not exactly cover materialization"
        )
    plan_binding = CanonicalJsonProofBinding.bind(
        run_plan_path, semantic_sha256=plan.sha256
    )
    schedule = FormalStageCapacitySchedule(
        schema_version=1,
        kind="lightcone_formal_stage_capacity_schedule",
        protocol_sha256=FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256,
        stage=subject.stage,
        protocol_lock_sha256=subject.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(
            subject.formal_runtime_authority_manifest_sha256
        ),
        registry_sha256=subject.execution_identity.registry_sha256,
        materialization=materialization_binding,
        materialization_receipt_sha256=materialization.sha256,
        stage_capacity_schedule=stage_schedule_binding,
        stage_capacity_schedule_sha256=stage_schedule.sha256,
        activated_cell_ids=cell_ids,
        materialized_cell_id=subject.materialized_cell_id,
        execution_binding_sha256=verified.sha256,
        execution_plan_sha256=subject.execution_plan_sha256,
        run_plan=plan_binding,
        run_plan_sha256=plan.sha256,
        topology_mode=subject.topology_mode,
        inventory_sha256=subject.inventory_sha256,
        gpu_uuids=subject.gpu_uuids,
        wave_index=wave.wave_index,
        wave_cell_ids=wave.cell_ids,
        provider_inventory_gpu_count=2,
        provider_reserved_gpu_count=(
            2 if subject.topology_mode != "tp1_dp1" else 2 // len(wave.cell_ids)
        ),
        retry_allowance=FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE,
        capacity_source_manifest=source_binding,
        capacity_envelope_sha256=replay.capacity_envelope.sha256,
        budget_inventory_sha256=replay.budget_inventory.sha256,
        capacity_captured_at_ns=replay.captured_at_ns,
    )
    _replay_capacity(schedule, now_ns=now_ns)
    publish_canonical_json_no_replace(output_path, schedule.to_dict())
    reopened = FormalStageCapacitySchedule.from_dict(
        CanonicalJsonProofBinding.bind(output_path).reopen()
    )
    if reopened != schedule:
        raise RuntimeError("formal capacity schedule changed during publication")
    return schedule


@dataclass(frozen=True)
class FormalStageCapacityGate:
    schema_version: Literal[3]
    kind: Literal["lightcone_formal_stage_capacity_gate"]
    protocol_sha256: str
    stage: str
    materialization_receipt_sha256: str
    materialized_cell_id: str
    schedule_sha256: str
    inventory_sha256: str
    status: Literal["AVAILABLE", "BLOCKED"]
    reason_code: str
    observed_free_bytes: int
    retained_evidence_bytes: int
    maximum_concurrent_transient_bytes: int
    high_water_bytes: int
    safety_margin_bytes: int
    required_free_bytes: int
    capacity_source_replay_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 3
            or self.kind != "lightcone_formal_stage_capacity_gate"
            or self.protocol_sha256 != FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256
            or self.stage not in FORMAL_STAGE_DAG
        ):
            raise ValueError("formal stage capacity gate schema is unsupported")
        for label, digest in (
            ("materialization", self.materialization_receipt_sha256),
            ("cell", self.materialized_cell_id),
            ("schedule", self.schedule_sha256),
            ("inventory", self.inventory_sha256),
            ("source replay", self.capacity_source_replay_sha256),
        ):
            _sha256(f"formal capacity gate {label}", digest)
        for label, value in (
            ("observed", self.observed_free_bytes),
            ("retained", self.retained_evidence_bytes),
            ("transient", self.maximum_concurrent_transient_bytes),
            ("high water", self.high_water_bytes),
            ("safety margin", self.safety_margin_bytes),
            ("required", self.required_free_bytes),
        ):
            _nonnegative_int(f"formal capacity gate {label}", value)
        expected_status = (
            "AVAILABLE"
            if self.observed_free_bytes >= self.required_free_bytes
            else "BLOCKED"
        )
        expected_reason = (
            "formal_stage_capacity_verified"
            if expected_status == "AVAILABLE"
            else "formal_stage_capacity_insufficient"
        )
        if (
            self.status != expected_status
            or self.reason_code != expected_reason
            or self.high_water_bytes
            != self.retained_evidence_bytes + self.maximum_concurrent_transient_bytes
            or self.required_free_bytes
            != self.high_water_bytes + self.safety_margin_bytes
        ):
            raise ValueError("formal stage capacity gate arithmetic/status differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {**self.__dict__, "gate_sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal stage capacity gate",
            value,
            {*cls.__dataclass_fields__, "gate_sha256"},
        )
        declared = _sha256("formal stage capacity gate", row.pop("gate_sha256"))
        gate = cls(**row)
        if gate.sha256 != declared:
            raise ValueError("formal stage capacity gate digest differs")
        return gate


def _derive_gate(
    schedule: (
        FormalStageCapacitySchedule
        | FormalDynamicStageCapacitySchedule
        | FormalE5OneShotStageCapacitySchedule
    ),
    replay: UnsignedCapacitySourceReplay,
) -> FormalStageCapacityGate:
    requirements = {
        row.cell_id: row for row in replay.capacity_envelope.cell_requirements
    }
    retained = sum(
        requirements[cell_id].maximum_evidence_bytes * (schedule.retry_allowance + 1)
        for cell_id in schedule.activated_cell_ids
    )
    transient = sum(
        requirements[cell_id].model_staging_bytes
        + requirements[cell_id].compile_overlay_bytes
        for cell_id in schedule.wave_cell_ids
    )
    high_water = retained + transient
    safety = max(
        STAGE_CAPACITY_MINIMUM_SAFETY_MARGIN_BYTES,
        (high_water * STAGE_CAPACITY_SAFETY_MARGIN_BPS + 9_999) // 10_000,
    )
    required = high_water + safety
    observed = replay.capacity_envelope.effective_host_bytes
    available = observed >= required
    return FormalStageCapacityGate(
        schema_version=3,
        kind="lightcone_formal_stage_capacity_gate",
        protocol_sha256=FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256,
        stage=schedule.stage,
        materialization_receipt_sha256=schedule.materialization_receipt_sha256,
        materialized_cell_id=schedule.materialized_cell_id,
        schedule_sha256=schedule.sha256,
        inventory_sha256=schedule.inventory_sha256,
        status="AVAILABLE" if available else "BLOCKED",
        reason_code=(
            "formal_stage_capacity_verified"
            if available
            else "formal_stage_capacity_insufficient"
        ),
        observed_free_bytes=observed,
        retained_evidence_bytes=retained,
        maximum_concurrent_transient_bytes=transient,
        high_water_bytes=high_water,
        safety_margin_bytes=safety,
        required_free_bytes=required,
        capacity_source_replay_sha256=replay.sha256,
    )


def materialize_formal_stage_capacity_gate(
    *,
    schedule_path: str | Path,
    output_path: str | Path,
    now_ns: int,
) -> FormalStageCapacityGate:
    schedule = _load_capacity_schedule(schedule_path)
    replay = _replay_capacity_schedule(schedule, now_ns=now_ns)
    gate = _derive_gate(schedule, replay)
    publish_canonical_json_no_replace(output_path, gate.to_dict())
    if (
        FormalStageCapacityGate.from_dict(
            CanonicalJsonProofBinding.bind(output_path).reopen()
        )
        != gate
    ):
        raise RuntimeError("formal stage capacity gate changed during publication")
    return gate


def revalidate_formal_stage_capacity_gate(
    *,
    schedule: (
        FormalStageCapacitySchedule
        | FormalDynamicStageCapacitySchedule
        | FormalE5OneShotStageCapacitySchedule
    ),
    gate: FormalStageCapacityGate,
    now_ns: int,
) -> UnsignedCapacitySourceReplay:
    replay = _replay_capacity_schedule(schedule, now_ns=now_ns)
    expected = _derive_gate(schedule, replay)
    if gate != expected:
        raise ValueError("formal stage capacity gate differs from fresh raw replay")
    return replay


def formal_stage_capacity_control_lineage_sha256(
    *,
    schedule: (
        FormalStageCapacitySchedule
        | FormalDynamicStageCapacitySchedule
        | FormalE5OneShotStageCapacitySchedule
    ),
    gate: FormalStageCapacityGate,
) -> str:
    if type(schedule) is FormalStageCapacitySchedule:
        source_schedule = schedule.stage_capacity_schedule.to_dict()
        source_schedule_sha256 = schedule.stage_capacity_schedule_sha256
    elif type(schedule) is FormalDynamicStageCapacitySchedule:
        source_schedule = schedule.dynamic_dispatch_schedule.to_dict()
        source_schedule_sha256 = schedule.dynamic_dispatch_schedule_sha256
    else:
        source_schedule = schedule.dispatch_schedule.to_dict()
        source_schedule_sha256 = schedule.dispatch_schedule_sha256
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_stage_capacity_control_lineage",
            "materialization_receipt_sha256": (schedule.materialization_receipt_sha256),
            "materialized_cell_id": schedule.materialized_cell_id,
            "stage_capacity_schedule": source_schedule,
            "stage_capacity_schedule_sha256": source_schedule_sha256,
            "execution_binding_sha256": schedule.execution_binding_sha256,
            "execution_plan_sha256": schedule.execution_plan_sha256,
            "run_plan_sha256": schedule.run_plan_sha256,
            "schedule_sha256": schedule.sha256,
            "gate_sha256": gate.sha256,
            "inventory_sha256": schedule.inventory_sha256,
            "topology_mode": schedule.topology_mode,
            "gpu_uuids": schedule.gpu_uuids,
            "wave_index": schedule.wave_index,
            "wave_cell_ids": schedule.wave_cell_ids,
            "provider_inventory_gpu_count": (schedule.provider_inventory_gpu_count),
            "provider_reserved_gpu_count": schedule.provider_reserved_gpu_count,
            "capacity_source_manifest": schedule.capacity_source_manifest.to_dict(),
        }
    )


@dataclass(frozen=True)
class FormalPilotLaunchBudget:
    """Control-signable hard cap for only the deterministic minimum pilots."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_pilot_launch_budget"]
    protocol_sha256: str
    protocol_lock_sha256: str
    runtime_authority_manifest_sha256: str
    stage: str
    phase: Literal["minimum_stratum_pilot"]
    prospective_source_manifest: CanonicalJsonProofBinding
    prospective_source_manifest_sha256: str
    materialization: CanonicalJsonProofBinding
    materialization_receipt_sha256: str
    minimum_pilot_cell_ids: tuple[str, ...]
    per_cell_hard_timeout_ns: int
    retry_allowance: int
    gpu_count_by_cell: tuple[tuple[str, int], ...]
    provider_reserved_gpu_count_by_cell: tuple[tuple[str, int], ...]
    maximum_compute_gpu_ns: int
    maximum_reserved_gpu_ns: int
    preflight_budget_receipt: CanonicalJsonProofBinding
    preflight_budget_receipt_sha256: str
    issued_ns: int
    expires_ns: int
    launch_nonce_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_pilot_launch_budget"
            or self.protocol_sha256 != FORMAL_PILOT_LAUNCH_BUDGET_PROTOCOL_SHA256
            or self.stage not in FORMAL_STAGE_DAG
            or self.phase != "minimum_stratum_pilot"
        ):
            raise ValueError("formal pilot launch budget schema is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime authority", self.runtime_authority_manifest_sha256),
            ("prospective source", self.prospective_source_manifest_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("preflight budget", self.preflight_budget_receipt_sha256),
            ("launch nonce", self.launch_nonce_sha256),
        ):
            _sha256(f"formal pilot budget {label}", digest)
        for value in (
            self.prospective_source_manifest,
            self.materialization,
            self.preflight_budget_receipt,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("formal pilot budget sources must be path-bound")
        if (
            not self.minimum_pilot_cell_ids
            or self.minimum_pilot_cell_ids
            != tuple(sorted(set(self.minimum_pilot_cell_ids)))
            or tuple(cell_id for cell_id, _count in self.gpu_count_by_cell)
            != self.minimum_pilot_cell_ids
            or any(count not in {1, 2} for _cell_id, count in self.gpu_count_by_cell)
            or tuple(
                cell_id for cell_id, _count in self.provider_reserved_gpu_count_by_cell
            )
            != self.minimum_pilot_cell_ids
            or any(
                count != 2
                for _cell_id, count in self.provider_reserved_gpu_count_by_cell
            )
        ):
            raise ValueError("formal pilot budget cell/GPU set is not canonical")
        if self.per_cell_hard_timeout_ns != FORMAL_PILOT_CELL_HARD_TIMEOUT_NS:
            raise ValueError("formal pilot budget timeout differs from protocol")
        if self.retry_allowance != FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE:
            raise ValueError(
                "formal pilot budget retry allowance differs from protocol"
            )
        _positive_int("formal pilot budget compute cap", self.maximum_compute_gpu_ns)
        _positive_int("formal pilot budget reserved cap", self.maximum_reserved_gpu_ns)
        if (
            type(self.issued_ns) is not int
            or self.issued_ns < 1
            or type(self.expires_ns) is not int
            or not self.issued_ns < self.expires_ns
            or self.expires_ns - self.issued_ns
            > FORMAL_PILOT_BUDGET_MAXIMUM_LIFETIME_NS
        ):
            raise ValueError("formal pilot budget lifetime is invalid")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "prospective_source_manifest": self.prospective_source_manifest.to_dict(),
            "materialization": self.materialization.to_dict(),
            "minimum_pilot_cell_ids": list(self.minimum_pilot_cell_ids),
            "gpu_count_by_cell": [list(row) for row in self.gpu_count_by_cell],
            "provider_reserved_gpu_count_by_cell": [
                list(row) for row in self.provider_reserved_gpu_count_by_cell
            ],
            "preflight_budget_receipt": self.preflight_budget_receipt.to_dict(),
            "budget_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal pilot launch budget",
            value,
            {*cls.__dataclass_fields__, "budget_sha256"},
        )
        declared = _sha256("formal pilot launch budget", row.pop("budget_sha256"))
        raw_cells = row["minimum_pilot_cell_ids"]
        raw_gpu = row["gpu_count_by_cell"]
        raw_provider_gpu = row["provider_reserved_gpu_count_by_cell"]
        if (
            type(raw_cells) is not list
            or type(raw_gpu) is not list
            or type(raw_provider_gpu) is not list
        ):
            raise TypeError("formal pilot budget cell sets must be arrays")
        row["minimum_pilot_cell_ids"] = tuple(raw_cells)
        gpu_rows: list[tuple[str, int]] = []
        for item in raw_gpu:
            if type(item) is not list or len(item) != 2:
                raise TypeError("formal pilot GPU row must be a pair")
            gpu_rows.append((item[0], item[1]))
        row["gpu_count_by_cell"] = tuple(gpu_rows)
        provider_gpu_rows: list[tuple[str, int]] = []
        for item in raw_provider_gpu:
            if type(item) is not list or len(item) != 2:
                raise TypeError("formal pilot provider GPU row must be a pair")
            provider_gpu_rows.append((item[0], item[1]))
        row["provider_reserved_gpu_count_by_cell"] = tuple(provider_gpu_rows)
        for name in (
            "prospective_source_manifest",
            "materialization",
            "preflight_budget_receipt",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        budget = cls(**row)
        if budget.sha256 != declared:
            raise ValueError("formal pilot launch budget digest differs")
        return budget


def revalidate_formal_pilot_launch_budget(
    budget: FormalPilotLaunchBudget, *, current_ns: int
) -> tuple[
    StagedProspectiveGpuHourSourceManifest, FormalStageGpuHourVerificationReceipt
]:
    """Deep-rebuild a BLOCKED staged source; never promote it to full-stage."""

    from lightcone_spec.experiments.gpu_hour_authority import (
        _derive_staged_prospective_gpu_hour_source,
        _reopen_completed_staged_source,
        _require_runtime_authority,
    )

    budget.__post_init__()
    if (
        type(current_ns) is not int
        or not budget.issued_ns <= current_ns <= budget.expires_ns
    ):
        raise ValueError("formal pilot launch budget is not currently fresh")
    for binding in (
        budget.prospective_source_manifest,
        budget.materialization,
        budget.preflight_budget_receipt,
    ):
        if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
            raise ValueError("formal pilot launch budget path binding changed")
    preflight = FormalStageGpuHourVerificationReceipt.from_dict(
        budget.preflight_budget_receipt.reopen()
    )
    preflight.revalidate(current_ns=current_ns)
    if (
        preflight.sha256 != budget.preflight_budget_receipt_sha256
        or preflight.stage != "preflight"
    ):
        raise ValueError("formal pilot launch lacks exact AVAILABLE preflight budget")
    lock = preflight.registry_receipt.signed_protocol_lock.payload
    runtime = preflight.formal_runtime_authority_manifest
    materialization = _materialization(budget.materialization)
    persisted = StagedProspectiveGpuHourSourceManifest.from_dict(
        budget.prospective_source_manifest.reopen()
    )
    if (
        budget.protocol_lock_sha256 != lock.sha256
        or budget.runtime_authority_manifest_sha256 != runtime.sha256
        or budget.materialization_receipt_sha256 != materialization.sha256
        or budget.prospective_source_manifest_sha256 != persisted.sha256
        or materialization.stage != budget.stage
        or persisted.stage != budget.stage
        or persisted.status != "BLOCKED"
        or persisted.protocol_lock_sha256 != lock.sha256
        or persisted.materialization_receipt_sha256 != materialization.sha256
        or persisted.inventory_sha256 != preflight.inventory.sha256
    ):
        raise ValueError("formal pilot launch immutable prospective lineage differs")
    member_sha256 = _require_runtime_authority(lock, runtime)
    completed_binding = None
    completed_source = None
    if persisted.completed_source_manifest is not None:
        completed_binding, completed_source = _reopen_completed_staged_source(
            source_manifest_path=(persisted.completed_source_manifest.absolute_path),
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime,
            materialization=materialization,
            inventory=preflight.inventory,
            now_ns=current_ns,
        )
    expected = _derive_staged_prospective_gpu_hour_source(
        protocol_lock=lock,
        runtime_authority_member_sha256=member_sha256,
        materialization=materialization,
        inventory=preflight.inventory,
        completed_source_binding=completed_binding,
        completed_source=completed_source,
    )
    if (
        expected != persisted
        or budget.minimum_pilot_cell_ids != expected.minimum_pilot_cell_ids
    ):
        raise ValueError("formal pilot minimum cell set differs from source rebuild")
    cells = {row.cell_id: row for row in materialization.cells}
    expected_gpu = tuple(
        (cell_id, _formal_cell_gpu_count(cells[cell_id]))
        for cell_id in expected.minimum_pilot_cell_ids
    )
    expected_provider_gpu = tuple(
        (cell_id, 2) for cell_id in expected.minimum_pilot_cell_ids
    )
    compute = (budget.retry_allowance + 1) * sum(
        FORMAL_PILOT_CELL_HARD_TIMEOUT_NS * count for _cell_id, count in expected_gpu
    )
    provider_process_reserve = (budget.retry_allowance + 1) * sum(
        FORMAL_PILOT_CELL_HARD_TIMEOUT_NS * count
        for _cell_id, count in expected_provider_gpu
    )
    reserved = (
        provider_process_reserve * (10_000 + FORMAL_PILOT_BUDGET_RESERVE_BPS) + 9_999
    ) // 10_000
    nonce = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_pilot_launch_nonce",
            "prospective_source_manifest_sha256": expected.sha256,
            "preflight_budget_receipt_sha256": preflight.sha256,
            "minimum_pilot_cell_ids": expected.minimum_pilot_cell_ids,
            "retry_allowance": FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE,
            "issued_ns": budget.issued_ns,
            "expires_ns": budget.expires_ns,
        }
    )
    if (
        budget.gpu_count_by_cell != expected_gpu
        or budget.provider_reserved_gpu_count_by_cell != expected_provider_gpu
        or budget.maximum_compute_gpu_ns != compute
        or budget.maximum_reserved_gpu_ns != reserved
        or budget.launch_nonce_sha256 != nonce
    ):
        raise ValueError("formal pilot launch hard cap differs from source protocol")
    return persisted, preflight


def formal_pilot_launch_budget_control_lineage_sha256(
    budget: FormalPilotLaunchBudget,
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_pilot_launch_budget_control_lineage",
            "budget_sha256": budget.sha256,
            "prospective_source_manifest": (
                budget.prospective_source_manifest.to_dict()
            ),
            "materialization": budget.materialization.to_dict(),
            "preflight_budget_receipt": budget.preflight_budget_receipt.to_dict(),
            "launch_nonce_sha256": budget.launch_nonce_sha256,
        }
    )


def materialize_formal_pilot_launch_budget(
    *,
    blocked_prospective_source_manifest_path: str | Path,
    preflight_budget_receipt_path: str | Path,
    issued_ns: int,
    expires_ns: int,
    output_path: str | Path,
) -> FormalPilotLaunchBudget:
    """Publish a hard cap derived only from the BLOCKED source and preflight.

    The target materialization is recovered from the registry prefix embedded
    in the AVAILABLE preflight receipt and is published beside the budget as a
    canonical, no-replace replay member.  No caller supplies cell IDs, gang
    sizes, durations, caps, or a nonce.
    """

    _positive_int("formal pilot budget issue time", issued_ns)
    _positive_int("formal pilot budget expiry time", expires_ns)
    if (
        not issued_ns < expires_ns
        or expires_ns - issued_ns > FORMAL_PILOT_BUDGET_MAXIMUM_LIFETIME_NS
    ):
        raise ValueError("formal pilot budget requested lifetime is invalid")
    source_binding = CanonicalJsonProofBinding.bind(
        blocked_prospective_source_manifest_path
    )
    source = StagedProspectiveGpuHourSourceManifest.from_dict(source_binding.reopen())
    if source.status != "BLOCKED":
        raise ValueError("formal pilot budget requires a BLOCKED staged source")
    preflight_binding = CanonicalJsonProofBinding.bind(preflight_budget_receipt_path)
    preflight = FormalStageGpuHourVerificationReceipt.from_dict(
        preflight_binding.reopen()
    )
    preflight.revalidate(current_ns=issued_ns)
    if preflight.stage != "preflight":
        raise ValueError("formal pilot budget requires AVAILABLE preflight authority")
    candidates = tuple(
        row.payload
        for row in preflight.registry_receipt.cumulative_signed_materializations
        if row.payload.sha256 == source.materialization_receipt_sha256
    )
    if len(candidates) != 1:
        raise ValueError(
            "preflight budget registry prefix lacks exact pilot materialization"
        )
    materialization = candidates[0]
    destination = Path(output_path)
    if not destination.is_absolute() or destination != destination.resolve(
        strict=False
    ):
        raise ValueError("formal pilot budget output must be absolute and resolved")
    materialization_path = destination.with_name(
        f"formal-pilot-materialization-{materialization.sha256}.json"
    )
    publish_canonical_json_no_replace(
        materialization_path,
        stage_materialization_receipt_to_dict(materialization),
    )
    materialization_binding = CanonicalJsonProofBinding.bind(materialization_path)
    cells = {row.cell_id: row for row in materialization.cells}
    gpu_count_by_cell = tuple(
        (cell_id, _formal_cell_gpu_count(cells[cell_id]))
        for cell_id in source.minimum_pilot_cell_ids
    )
    provider_reserved_gpu_count_by_cell = tuple(
        (cell_id, 2) for cell_id in source.minimum_pilot_cell_ids
    )
    maximum_compute_gpu_ns = (FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE + 1) * sum(
        FORMAL_PILOT_CELL_HARD_TIMEOUT_NS * count
        for _cell_id, count in gpu_count_by_cell
    )
    provider_process_reserve_gpu_ns = (FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE + 1) * sum(
        FORMAL_PILOT_CELL_HARD_TIMEOUT_NS * count
        for _cell_id, count in provider_reserved_gpu_count_by_cell
    )
    maximum_reserved_gpu_ns = (
        provider_process_reserve_gpu_ns * (10_000 + FORMAL_PILOT_BUDGET_RESERVE_BPS)
        + 9_999
    ) // 10_000
    launch_nonce_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_pilot_launch_nonce",
            "prospective_source_manifest_sha256": source.sha256,
            "preflight_budget_receipt_sha256": preflight.sha256,
            "minimum_pilot_cell_ids": source.minimum_pilot_cell_ids,
            "retry_allowance": FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE,
            "issued_ns": issued_ns,
            "expires_ns": expires_ns,
        }
    )
    lock = preflight.registry_receipt.signed_protocol_lock.payload
    budget = FormalPilotLaunchBudget(
        schema_version=1,
        kind="lightcone_formal_pilot_launch_budget",
        protocol_sha256=FORMAL_PILOT_LAUNCH_BUDGET_PROTOCOL_SHA256,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_manifest_sha256=(
            preflight.formal_runtime_authority_manifest.sha256
        ),
        stage=source.stage,
        phase="minimum_stratum_pilot",
        prospective_source_manifest=source_binding,
        prospective_source_manifest_sha256=source.sha256,
        materialization=materialization_binding,
        materialization_receipt_sha256=materialization.sha256,
        minimum_pilot_cell_ids=source.minimum_pilot_cell_ids,
        per_cell_hard_timeout_ns=FORMAL_PILOT_CELL_HARD_TIMEOUT_NS,
        retry_allowance=FORMAL_STAGE_CAPACITY_RETRY_ALLOWANCE,
        gpu_count_by_cell=gpu_count_by_cell,
        provider_reserved_gpu_count_by_cell=(provider_reserved_gpu_count_by_cell),
        maximum_compute_gpu_ns=maximum_compute_gpu_ns,
        maximum_reserved_gpu_ns=maximum_reserved_gpu_ns,
        preflight_budget_receipt=preflight_binding,
        preflight_budget_receipt_sha256=preflight.sha256,
        issued_ns=issued_ns,
        expires_ns=expires_ns,
        launch_nonce_sha256=launch_nonce_sha256,
    )
    revalidate_formal_pilot_launch_budget(budget, current_ns=issued_ns)
    publish_canonical_json_no_replace(destination, budget.to_dict())
    reopened = FormalPilotLaunchBudget.from_dict(
        CanonicalJsonProofBinding.bind(destination).reopen()
    )
    if reopened != budget:
        raise RuntimeError("formal pilot budget changed during publication")
    return budget


@dataclass(frozen=True)
class FormalPilotBudgetVerificationReceipt:
    """Durable one-time verification of the stage-level pilot budget control."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_pilot_budget_verification_receipt"]
    protocol_sha256: str
    pilot_launch_budget: CanonicalJsonProofBinding
    pilot_launch_budget_sha256: str
    pilot_budget_control: ControlArtifactAttestation
    reservation: ChallengeReplayReservationBinding
    verified_ns: int
    registry_sha256: str
    inventory_sha256: str
    root_manifest_sha256: str
    trusted_attester_policy_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_pilot_budget_verification_receipt"
            or self.protocol_sha256 != FORMAL_PILOT_BUDGET_VERIFICATION_PROTOCOL_SHA256
        ):
            raise ValueError("formal pilot budget verification schema differs")
        if type(self.pilot_launch_budget) is not CanonicalJsonProofBinding:
            raise TypeError("formal pilot verification budget must be path-bound")
        if type(self.pilot_budget_control) is not ControlArtifactAttestation:
            raise TypeError("formal pilot verification control is not exact")
        if type(self.reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("formal pilot verification reservation is not exact")
        for label, digest in (
            ("pilot budget", self.pilot_launch_budget_sha256),
            ("registry", self.registry_sha256),
            ("inventory", self.inventory_sha256),
            ("release root", self.root_manifest_sha256),
            ("attester policy", self.trusted_attester_policy_sha256),
        ):
            _sha256(f"formal pilot verification {label}", digest)
        _positive_int("formal pilot verification time", self.verified_ns)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "pilot_launch_budget": self.pilot_launch_budget.to_dict(),
            "pilot_budget_control": self.pilot_budget_control.to_dict(),
            "reservation": self.reservation.to_dict(),
            "receipt_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal pilot budget verification receipt",
            value,
            {*cls.__dataclass_fields__, "receipt_sha256"},
        )
        declared = _sha256(
            "formal pilot budget verification receipt",
            row.pop("receipt_sha256"),
        )
        row["pilot_launch_budget"] = CanonicalJsonProofBinding.from_dict(
            row["pilot_launch_budget"]
        )
        row["pilot_budget_control"] = ControlArtifactAttestation.from_dict(
            row["pilot_budget_control"]
        )
        row["reservation"] = ChallengeReplayReservationBinding.from_dict(
            row["reservation"]
        )
        receipt = cls(**row)
        if receipt.sha256 != declared:
            raise ValueError("formal pilot budget verification digest differs")
        return receipt


def _pilot_verification_output_path(
    replay_store: ChallengeReplayStore,
    *,
    budget_sha256: str,
) -> Path:
    root = Path(replay_store.root)
    return (
        root.parent
        / "formal-pilot-budget-verification-receipts"
        / budget_sha256
        / "receipt.json"
    )


def verify_and_publish_formal_pilot_budget_receipt(
    *,
    pilot_launch_budget_path: str | Path,
    pilot_budget_control: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> CanonicalJsonProofBinding:
    """Reserve one budget challenge once, then publish its durable receipt."""

    budget_binding = CanonicalJsonProofBinding.bind(pilot_launch_budget_path)
    budget = FormalPilotLaunchBudget.from_dict(budget_binding.reopen())
    _source, preflight = revalidate_formal_pilot_launch_budget(
        budget, current_ns=now_ns
    )
    lock = preflight.registry_receipt.signed_protocol_lock.payload
    policy_sha256 = preflight.registry_receipt.trusted_release_policy(
        current_ns=now_ns
    ).sha256
    if Path(replay_store.root) != Path(preflight.reservation.path).parent:
        raise ValueError("formal pilot verification uses another replay store")
    _require_control_subject(
        pilot_budget_control,
        artifact_type="rank_aggregate",
        artifact_sha256=budget.sha256,
        protocol_sha256=FORMAL_PILOT_LAUNCH_BUDGET_PROTOCOL_SHA256,
        registry_sha256=lock.registry_sha256,
        lineage_sha256=formal_pilot_launch_budget_control_lineage_sha256(budget),
    )
    if (
        pilot_budget_control.deployment_policy_authorization.root_manifest_sha256
        != lock.offline_release_trust_root_sha256
        or pilot_budget_control.trusted_attester_policy_sha256 != policy_sha256
    ):
        raise ValueError("formal pilot verification release policy differs")
    verified = verify_and_reserve_release_control_artifact_attestations(
        (pilot_budget_control,),
        expected_inventory_sha256=preflight.inventory.sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified, reserved_ns=now_ns
    )
    receipt = FormalPilotBudgetVerificationReceipt(
        schema_version=1,
        kind="lightcone_formal_pilot_budget_verification_receipt",
        protocol_sha256=FORMAL_PILOT_BUDGET_VERIFICATION_PROTOCOL_SHA256,
        pilot_launch_budget=budget_binding,
        pilot_launch_budget_sha256=budget.sha256,
        pilot_budget_control=pilot_budget_control,
        reservation=replay_store.bind_reservation(reservation_sha256),
        verified_ns=now_ns,
        registry_sha256=lock.registry_sha256,
        inventory_sha256=preflight.inventory.sha256,
        root_manifest_sha256=lock.offline_release_trust_root_sha256,
        trusted_attester_policy_sha256=policy_sha256,
    )
    output = _pilot_verification_output_path(replay_store, budget_sha256=budget.sha256)
    _ensure_private_ledger_directory(output.parent)
    publish_canonical_json_no_replace(output, receipt.to_dict())
    binding = CanonicalJsonProofBinding.bind(output)
    revalidate_formal_pilot_budget_verification_receipt(binding, current_ns=now_ns)
    return binding


def revalidate_formal_pilot_budget_verification_receipt(
    binding: CanonicalJsonProofBinding,
    *,
    current_ns: int,
) -> tuple[
    FormalPilotBudgetVerificationReceipt,
    FormalPilotLaunchBudget,
    FormalStageGpuHourVerificationReceipt,
]:
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("formal pilot verification receipt changed")
    receipt = FormalPilotBudgetVerificationReceipt.from_dict(binding.reopen())
    budget = FormalPilotLaunchBudget.from_dict(receipt.pilot_launch_budget.reopen())
    _source, preflight = revalidate_formal_pilot_launch_budget(
        budget, current_ns=current_ns
    )
    lock = preflight.registry_receipt.signed_protocol_lock.payload
    policy_sha256 = preflight.registry_receipt.trusted_release_policy(
        current_ns=current_ns
    ).sha256
    replay_store = ChallengeReplayStore(str(Path(receipt.reservation.path).parent))
    expected_path = _pilot_verification_output_path(
        replay_store, budget_sha256=budget.sha256
    )
    _require_control_subject(
        receipt.pilot_budget_control,
        artifact_type="rank_aggregate",
        artifact_sha256=budget.sha256,
        protocol_sha256=FORMAL_PILOT_LAUNCH_BUDGET_PROTOCOL_SHA256,
        registry_sha256=lock.registry_sha256,
        lineage_sha256=formal_pilot_launch_budget_control_lineage_sha256(budget),
    )
    verified = verify_release_control_artifact_attestation(
        receipt.pilot_budget_control,
        expected_inventory_sha256=preflight.inventory.sha256,
        now_ns=receipt.verified_ns,
        consumed_challenge_sha256s=(),
    )
    expected_challenges = tuple(
        sorted(
            {
                verified.challenge_sha256,
                verified.deployment_policy_challenge_sha256,
            }
        )
    )
    if (
        receipt.pilot_launch_budget
        != CanonicalJsonProofBinding.bind(receipt.pilot_launch_budget.absolute_path)
        or receipt.pilot_launch_budget_sha256 != budget.sha256
        or receipt.registry_sha256 != lock.registry_sha256
        or receipt.inventory_sha256 != preflight.inventory.sha256
        or receipt.root_manifest_sha256 != lock.offline_release_trust_root_sha256
        or receipt.trusted_attester_policy_sha256 != policy_sha256
        or receipt.pilot_budget_control.deployment_policy_authorization.root_manifest_sha256
        != receipt.root_manifest_sha256
        or receipt.pilot_budget_control.trusted_attester_policy_sha256 != policy_sha256
        or receipt.reservation.revalidate() != expected_challenges
        or receipt.reservation.reservation_sha256
        != control_challenge_reservation_sha256(
            (verified,), reserved_ns=receipt.verified_ns
        )
        or Path(binding.absolute_path) != expected_path
    ):
        raise ValueError("formal pilot verification durable lineage differs")
    return receipt, budget, preflight


@dataclass(frozen=True)
class _FormalStageBudgetLimits:
    mode: Literal[
        "available_stage_gpu_hour",
        "minimum_pilot_bootstrap",
        "registered_e5_one_shot",
    ]
    authority_sha256: str
    authority_reservation_sha256: str
    control_replay_root: str
    ledger_parent: str
    stage: str
    materialization_receipt_sha256: str
    cell_ids: tuple[str, ...]
    gpu_count_by_cell: tuple[tuple[str, int], ...]
    provider_reserved_gpu_count_cap_by_cell: tuple[tuple[str, int], ...]
    allowed_attempts_per_cell: int
    hard_timeout_ns_by_cell: tuple[tuple[str, int], ...]
    provider_wave_hard_timeout_ns_by_cell: tuple[tuple[str, int], ...]
    compute_charge_gpu_ns_by_cell: tuple[tuple[str, int], ...]
    reserved_charge_gpu_ns_by_cell: tuple[tuple[str, int], ...]
    hard_timeout_derivation_sha256: str
    maximum_compute_gpu_ns: int
    maximum_reserved_gpu_ns: int

    def __post_init__(self) -> None:
        for label, digest in (
            ("authority", self.authority_sha256),
            ("authority reservation", self.authority_reservation_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("hard timeout derivation", self.hard_timeout_derivation_sha256),
        ):
            _sha256(f"formal budget limits {label}", digest)
        for label, value in (
            ("control replay root", self.control_replay_root),
            ("ledger parent", self.ledger_parent),
        ):
            parent = Path(value)
            if not parent.is_absolute() or parent != parent.resolve(strict=False):
                raise ValueError(f"formal budget {label} is not absolute")
        if Path(self.control_replay_root) == Path(self.ledger_parent):
            raise ValueError("formal budget ledger must not pollute replay store")
        if (
            self.stage not in FORMAL_STAGE_DAG
            or not self.cell_ids
            or self.cell_ids != tuple(sorted(set(self.cell_ids)))
            or tuple(row[0] for row in self.gpu_count_by_cell) != self.cell_ids
            or any(row[1] not in {1, 2} for row in self.gpu_count_by_cell)
            or tuple(row[0] for row in self.provider_reserved_gpu_count_cap_by_cell)
            != self.cell_ids
            or any(
                row[1] not in {1, 2}
                for row in self.provider_reserved_gpu_count_cap_by_cell
            )
            or tuple(row[0] for row in self.hard_timeout_ns_by_cell) != self.cell_ids
            or any(row[1] < 1 for row in self.hard_timeout_ns_by_cell)
            or tuple(row[0] for row in self.provider_wave_hard_timeout_ns_by_cell)
            != self.cell_ids
            or any(row[1] < 1 for row in self.provider_wave_hard_timeout_ns_by_cell)
            or tuple(row[0] for row in self.compute_charge_gpu_ns_by_cell)
            != self.cell_ids
            or tuple(row[0] for row in self.reserved_charge_gpu_ns_by_cell)
            != self.cell_ids
            or any(row[1] < 1 for row in self.compute_charge_gpu_ns_by_cell)
            or any(row[1] < 1 for row in self.reserved_charge_gpu_ns_by_cell)
        ):
            raise ValueError("formal budget cell universe differs")
        gpu_count = dict(self.gpu_count_by_cell)
        provider_count = dict(self.provider_reserved_gpu_count_cap_by_cell)
        timeout = dict(self.hard_timeout_ns_by_cell)
        provider_timeout = dict(self.provider_wave_hard_timeout_ns_by_cell)
        if any(
            dict(self.compute_charge_gpu_ns_by_cell)[cell_id]
            != timeout[cell_id] * gpu_count[cell_id]
            or dict(self.reserved_charge_gpu_ns_by_cell)[cell_id]
            != provider_timeout[cell_id] * provider_count[cell_id]
            or timeout[cell_id] > provider_timeout[cell_id]
            for cell_id in self.cell_ids
        ):
            raise ValueError(
                "formal budget typed cell charges differ from process caps"
            )
        _positive_int("formal budget attempts", self.allowed_attempts_per_cell)
        _positive_int("formal budget compute cap", self.maximum_compute_gpu_ns)
        _positive_int("formal budget reserved cap", self.maximum_reserved_gpu_ns)
        if self.maximum_reserved_gpu_ns < self.maximum_compute_gpu_ns:
            raise ValueError("formal budget reserved cap is below compute cap")

    @cached_property
    def ledger_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_formal_stage_budget_consumption_ledger",
                "mode": self.mode,
                "authority_sha256": self.authority_sha256,
                "authority_reservation_sha256": (self.authority_reservation_sha256),
                "control_replay_root": self.control_replay_root,
                "ledger_parent": self.ledger_parent,
                "stage": self.stage,
                "materialization_receipt_sha256": (self.materialization_receipt_sha256),
                "cell_ids": self.cell_ids,
                "gpu_count_by_cell": self.gpu_count_by_cell,
                "provider_reserved_gpu_count_cap_by_cell": (
                    self.provider_reserved_gpu_count_cap_by_cell
                ),
                "allowed_attempts_per_cell": self.allowed_attempts_per_cell,
                "hard_timeout_ns_by_cell": self.hard_timeout_ns_by_cell,
                "provider_wave_hard_timeout_ns_by_cell": (
                    self.provider_wave_hard_timeout_ns_by_cell
                ),
                "compute_charge_gpu_ns_by_cell": (self.compute_charge_gpu_ns_by_cell),
                "reserved_charge_gpu_ns_by_cell": (self.reserved_charge_gpu_ns_by_cell),
                "hard_timeout_derivation_sha256": (self.hard_timeout_derivation_sha256),
                "maximum_compute_gpu_ns": self.maximum_compute_gpu_ns,
                "maximum_reserved_gpu_ns": self.maximum_reserved_gpu_ns,
                "protocol_sha256": (FORMAL_STAGE_LAUNCH_ADMISSION_PROTOCOL_SHA256),
            }
        )

    @property
    def ledger_directory(self) -> Path:
        return (
            Path(self.ledger_parent)
            / "formal-stage-budget-ledgers"
            / self.ledger_sha256
        )

    def hard_timeout_ns(self, materialized_cell_id: str) -> int:
        try:
            return dict(self.hard_timeout_ns_by_cell)[materialized_cell_id]
        except KeyError as error:
            raise ValueError("formal budget lacks a per-cell timeout") from error


@dataclass(frozen=True)
class FormalStageBudgetConsumption:
    """One append-only reservation of potential GPU time for a launch."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_stage_budget_consumption"]
    protocol_sha256: str
    ledger_sha256: str
    budget_mode: str
    budget_authority_sha256: str
    budget_authority_reservation_sha256: str
    stage: str
    materialization_receipt_sha256: str
    materialized_cell_id: str
    attempt_index: int
    global_index: int
    allowed_attempts_per_cell: int
    execution_binding_sha256: str
    execution_plan_sha256: str
    run_plan_sha256: str
    capacity_schedule_sha256: str
    capacity_gate_sha256: str
    capacity_control_sha256: str
    topology_mode: str
    gpu_count: int
    provider_reserved_gpu_count: int
    hard_timeout_ns: int
    provider_wave_hard_timeout_ns: int
    hard_timeout_derivation_sha256: str
    compute_charge_gpu_ns: int
    reserved_charge_gpu_ns: int
    cumulative_compute_gpu_ns: int
    cumulative_reserved_gpu_ns: int
    maximum_compute_gpu_ns: int
    maximum_reserved_gpu_ns: int
    reserved_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_stage_budget_consumption"
            or self.protocol_sha256 != FORMAL_STAGE_LAUNCH_ADMISSION_PROTOCOL_SHA256
            or self.stage not in FORMAL_STAGE_DAG
        ):
            raise ValueError("formal budget consumption schema is unsupported")
        for label, digest in (
            ("ledger", self.ledger_sha256),
            ("budget authority", self.budget_authority_sha256),
            ("budget reservation", self.budget_authority_reservation_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("cell", self.materialized_cell_id),
            ("execution binding", self.execution_binding_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("run plan", self.run_plan_sha256),
            ("capacity schedule", self.capacity_schedule_sha256),
            ("capacity gate", self.capacity_gate_sha256),
            ("capacity control", self.capacity_control_sha256),
            ("hard timeout derivation", self.hard_timeout_derivation_sha256),
        ):
            _sha256(f"formal budget consumption {label}", digest)
        if self.budget_mode not in {
            "available_stage_gpu_hour",
            "minimum_pilot_bootstrap",
            "registered_e5_one_shot",
        }:
            raise ValueError("formal budget consumption mode is unsupported")
        for label, value in (
            ("attempt index", self.attempt_index),
            ("global index", self.global_index),
        ):
            _nonnegative_int(f"formal budget consumption {label}", value)
        _positive_int(
            "formal budget consumption allowed attempts",
            self.allowed_attempts_per_cell,
        )
        if self.attempt_index >= self.allowed_attempts_per_cell:
            raise ValueError("formal budget consumption exceeds per-cell attempts")
        if self.topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
            raise ValueError("formal budget consumption topology is unsupported")
        if self.gpu_count not in {1, 2}:
            raise ValueError("formal budget consumption gang size is invalid")
        if self.provider_reserved_gpu_count not in {1, 2}:
            raise ValueError("formal budget provider reservation size is invalid")
        for label, value in (
            ("hard timeout", self.hard_timeout_ns),
            ("provider wave hard timeout", self.provider_wave_hard_timeout_ns),
            ("compute charge", self.compute_charge_gpu_ns),
            ("reserved charge", self.reserved_charge_gpu_ns),
            ("cumulative compute", self.cumulative_compute_gpu_ns),
            ("cumulative reserved", self.cumulative_reserved_gpu_ns),
            ("maximum compute", self.maximum_compute_gpu_ns),
            ("maximum reserved", self.maximum_reserved_gpu_ns),
            ("reservation time", self.reserved_ns),
        ):
            _positive_int(f"formal budget consumption {label}", value)
        if (
            self.compute_charge_gpu_ns != self.hard_timeout_ns * self.gpu_count
            or self.reserved_charge_gpu_ns
            != self.provider_wave_hard_timeout_ns * self.provider_reserved_gpu_count
            or self.hard_timeout_ns > self.provider_wave_hard_timeout_ns
            or self.cumulative_compute_gpu_ns > self.maximum_compute_gpu_ns
            or self.cumulative_reserved_gpu_ns > self.maximum_reserved_gpu_ns
        ):
            raise ValueError("formal budget consumption arithmetic exceeds authority")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {**self.__dict__, "consumption_sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal stage budget consumption",
            value,
            {*cls.__dataclass_fields__, "consumption_sha256"},
        )
        declared = _sha256(
            "formal stage budget consumption", row.pop("consumption_sha256")
        )
        result = cls(**row)
        if result.sha256 != declared:
            raise ValueError("formal stage budget consumption digest differs")
        return result


@dataclass(frozen=True)
class FormalStageLaunchConsumption:
    """The one-shot, pre-allocation consumption of one launch admission."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_stage_launch_consumption"]
    admission_sha256: str
    materialization_receipt_sha256: str
    materialized_cell_id: str
    execution_plan_sha256: str
    run_plan_sha256: str
    reservation_sha256: str
    consumed_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_stage_launch_consumption"
        ):
            raise ValueError("formal launch consumption schema differs")
        for label, digest in (
            ("admission", self.admission_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("cell", self.materialized_cell_id),
            ("execution plan", self.execution_plan_sha256),
            ("run plan", self.run_plan_sha256),
            ("reservation", self.reservation_sha256),
        ):
            _sha256(f"formal launch consumption {label}", digest)
        _positive_int("formal launch consumption time", self.consumed_ns)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {**self.__dict__, "consumption_sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal stage launch consumption",
            value,
            {*cls.__dataclass_fields__, "consumption_sha256"},
        )
        declared = _sha256(
            "formal stage launch consumption", row.pop("consumption_sha256")
        )
        result = cls(**row)
        if result.sha256 != declared:
            raise ValueError("formal stage launch consumption digest differs")
        return result


def _ensure_private_ledger_directory(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.mkdir(mode=0o700, exist_ok=True)
    for item in (path.parent, path):
        status = item.lstat()
        if (
            not stat.S_ISDIR(status.st_mode)
            or item.is_symlink()
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) & 0o077
        ):
            raise ValueError("formal budget ledger directory is unsafe")


def _ledger_entries(
    limits: _FormalStageBudgetLimits,
) -> tuple[tuple[CanonicalJsonProofBinding, FormalStageBudgetConsumption], ...]:
    directory = limits.ledger_directory
    rows: list[tuple[CanonicalJsonProofBinding, FormalStageBudgetConsumption]] = []
    inodes: set[tuple[int, int]] = set()
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == ".lock":
            continue
        if not path.name.startswith("entry-") or not path.name.endswith(".json"):
            raise ValueError("formal budget ledger contains an unknown member")
        status = path.lstat()
        if (
            not stat.S_ISREG(status.st_mode)
            or path.is_symlink()
            or status.st_nlink != 1
            or (status.st_dev, status.st_ino) in inodes
        ):
            raise ValueError("formal budget ledger member is unsafe or aliased")
        inodes.add((status.st_dev, status.st_ino))
        binding = CanonicalJsonProofBinding.bind(path)
        entry = FormalStageBudgetConsumption.from_dict(binding.reopen())
        if (
            entry.ledger_sha256 != limits.ledger_sha256
            or entry.global_index != len(rows)
            or path.name != f"entry-{entry.global_index:08d}.json"
            or entry.compute_charge_gpu_ns
            != dict(limits.compute_charge_gpu_ns_by_cell)[entry.materialized_cell_id]
            or entry.reserved_charge_gpu_ns
            != dict(limits.reserved_charge_gpu_ns_by_cell)[entry.materialized_cell_id]
        ):
            raise ValueError("formal budget ledger sequence/identity differs")
        rows.append((binding, entry))
    return tuple(rows)


def _next_stage_budget_attempt(
    *,
    limits: _FormalStageBudgetLimits,
    materialized_cell_id: str,
) -> int:
    """Observe the next append index used by a cell-scoped control subject.

    The append operation rechecks this value while holding the exclusive
    ledger lock.  A concurrent winner can therefore only make this admission
    fail closed after its unique control challenge was reserved; it cannot
    cause two entries to claim one attempt.
    """

    if materialized_cell_id not in limits.cell_ids:
        raise ValueError("formal budget does not cover requested attempt")
    directory = limits.ledger_directory
    _ensure_private_ledger_directory(directory)
    lock_path = directory / ".lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        attempt = sum(
            entry.materialized_cell_id == materialized_cell_id
            for _binding, entry in _ledger_entries(limits)
        )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    if attempt >= limits.allowed_attempts_per_cell:
        raise ValueError("formal budget per-cell attempt cap is exhausted")
    return attempt


def _reserve_stage_budget_consumption(
    *,
    limits: _FormalStageBudgetLimits,
    schedule: FormalStageCapacitySchedule,
    gate: FormalStageCapacityGate,
    capacity_control: ControlArtifactAttestation,
    reserved_ns: int,
    expected_attempt_index: int | None = None,
) -> CanonicalJsonProofBinding:
    if schedule.materialized_cell_id not in limits.cell_ids:
        raise ValueError("formal budget does not cover this materialized cell")
    expected_gpus = dict(limits.gpu_count_by_cell)[schedule.materialized_cell_id]
    provider_gpu_cap = dict(limits.provider_reserved_gpu_count_cap_by_cell)[
        schedule.materialized_cell_id
    ]
    actual_gpus = len(schedule.gpu_uuids)
    if actual_gpus != expected_gpus:
        raise ValueError("formal budget gang cap differs from exact run plan")
    if schedule.provider_reserved_gpu_count != provider_gpu_cap:
        raise ValueError("formal budget provider reservation differs from its cell cap")
    directory = limits.ledger_directory
    _ensure_private_ledger_directory(directory)
    lock_path = directory / ".lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise ValueError("formal budget ledger lock is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        rows = _ledger_entries(limits)
        existing = tuple(entry for _binding, entry in rows)
        if any(
            row.execution_binding_sha256 == schedule.execution_binding_sha256
            or row.capacity_control_sha256 == capacity_control.sha256
            for row in existing
        ):
            raise ValueError("formal budget launch attempt was already reserved")
        cell_attempts = sum(
            row.materialized_cell_id == schedule.materialized_cell_id
            for row in existing
        )
        if cell_attempts >= limits.allowed_attempts_per_cell:
            raise ValueError("formal budget per-cell attempt cap is exhausted")
        if (
            expected_attempt_index is not None
            and cell_attempts != expected_attempt_index
        ):
            raise ValueError("formal budget cell attempt changed before reservation")
        hard_timeout_ns = limits.hard_timeout_ns(schedule.materialized_cell_id)
        provider_wave_hard_timeout_ns = dict(
            limits.provider_wave_hard_timeout_ns_by_cell
        )[schedule.materialized_cell_id]
        compute_charge = dict(limits.compute_charge_gpu_ns_by_cell)[
            schedule.materialized_cell_id
        ]
        reserved_charge = dict(limits.reserved_charge_gpu_ns_by_cell)[
            schedule.materialized_cell_id
        ]
        cumulative_compute = (
            sum(row.compute_charge_gpu_ns for row in existing) + compute_charge
        )
        cumulative_reserved = (
            sum(row.reserved_charge_gpu_ns for row in existing) + reserved_charge
        )
        if (
            cumulative_compute > limits.maximum_compute_gpu_ns
            or cumulative_reserved > limits.maximum_reserved_gpu_ns
        ):
            raise ValueError("formal stage GPU-hour budget is exhausted")
        entry = FormalStageBudgetConsumption(
            schema_version=1,
            kind="lightcone_formal_stage_budget_consumption",
            protocol_sha256=FORMAL_STAGE_LAUNCH_ADMISSION_PROTOCOL_SHA256,
            ledger_sha256=limits.ledger_sha256,
            budget_mode=limits.mode,
            budget_authority_sha256=limits.authority_sha256,
            budget_authority_reservation_sha256=(limits.authority_reservation_sha256),
            stage=limits.stage,
            materialization_receipt_sha256=(limits.materialization_receipt_sha256),
            materialized_cell_id=schedule.materialized_cell_id,
            attempt_index=cell_attempts,
            global_index=len(existing),
            allowed_attempts_per_cell=limits.allowed_attempts_per_cell,
            execution_binding_sha256=schedule.execution_binding_sha256,
            execution_plan_sha256=schedule.execution_plan_sha256,
            run_plan_sha256=schedule.run_plan_sha256,
            capacity_schedule_sha256=schedule.sha256,
            capacity_gate_sha256=gate.sha256,
            capacity_control_sha256=capacity_control.sha256,
            topology_mode=schedule.topology_mode,
            gpu_count=actual_gpus,
            provider_reserved_gpu_count=schedule.provider_reserved_gpu_count,
            hard_timeout_ns=hard_timeout_ns,
            provider_wave_hard_timeout_ns=provider_wave_hard_timeout_ns,
            hard_timeout_derivation_sha256=(limits.hard_timeout_derivation_sha256),
            compute_charge_gpu_ns=compute_charge,
            reserved_charge_gpu_ns=reserved_charge,
            cumulative_compute_gpu_ns=cumulative_compute,
            cumulative_reserved_gpu_ns=cumulative_reserved,
            maximum_compute_gpu_ns=limits.maximum_compute_gpu_ns,
            maximum_reserved_gpu_ns=limits.maximum_reserved_gpu_ns,
            reserved_ns=reserved_ns,
        )
        path = directory / f"entry-{entry.global_index:08d}.json"
        publish_canonical_json_no_replace(path, entry.to_dict())
        binding = CanonicalJsonProofBinding.bind(path)
        if FormalStageBudgetConsumption.from_dict(binding.reopen()) != entry:
            raise RuntimeError("formal budget ledger entry changed during append")
        return binding
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_stage_budget_consumption(
    binding: CanonicalJsonProofBinding,
    *,
    limits: _FormalStageBudgetLimits,
    schedule: FormalStageCapacitySchedule,
    gate: FormalStageCapacityGate,
    capacity_control: ControlArtifactAttestation,
) -> FormalStageBudgetConsumption:
    expected_directory = limits.ledger_directory
    path = Path(binding.absolute_path)
    if path.parent != expected_directory:
        raise ValueError("formal budget ledger was copied to another root")
    _ensure_private_ledger_directory(expected_directory)
    lock_path = expected_directory / ".lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | (getattr(os, "O_NOFOLLOW", 0)),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        rows = _ledger_entries(limits)
        matches = tuple(entry for current, entry in rows if current == binding)
        if len(matches) != 1:
            raise ValueError("formal budget consumption is absent from exact ledger")
        entry = matches[0]
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    if (
        entry.execution_binding_sha256 != schedule.execution_binding_sha256
        or entry.execution_plan_sha256 != schedule.execution_plan_sha256
        or entry.run_plan_sha256 != schedule.run_plan_sha256
        or entry.capacity_schedule_sha256 != schedule.sha256
        or entry.capacity_gate_sha256 != gate.sha256
        or entry.capacity_control_sha256 != capacity_control.sha256
        or entry.materialized_cell_id != schedule.materialized_cell_id
        or entry.gpu_count != len(schedule.gpu_uuids)
        or entry.provider_reserved_gpu_count != schedule.provider_reserved_gpu_count
        or entry.hard_timeout_ns
        != limits.hard_timeout_ns(schedule.materialized_cell_id)
        or entry.provider_wave_hard_timeout_ns
        != dict(limits.provider_wave_hard_timeout_ns_by_cell)[
            schedule.materialized_cell_id
        ]
        or entry.hard_timeout_derivation_sha256 != limits.hard_timeout_derivation_sha256
        or entry.maximum_compute_gpu_ns != limits.maximum_compute_gpu_ns
        or entry.maximum_reserved_gpu_ns != limits.maximum_reserved_gpu_ns
    ):
        raise ValueError("formal budget consumption differs from launch admission")
    return entry


@dataclass(frozen=True)
class FormalStageLaunchAdmission:
    schema_version: Literal[1]
    kind: Literal["lightcone_formal_stage_launch_admission"]
    protocol_sha256: str
    verified_ns: int
    stage: str
    registry_sha256: str
    protocol_lock_sha256: str
    root_manifest_sha256: str
    runtime_authority_manifest_sha256: str
    materialization_receipt_sha256: str
    materialized_cell_id: str
    execution_binding_sha256: str
    execution_plan_sha256: str
    run_plan_sha256: str
    topology_mode: str
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    capacity_schedule: CanonicalJsonProofBinding
    capacity_gate: CanonicalJsonProofBinding
    capacity_control: ControlArtifactAttestation
    budget_mode: Literal[
        "available_stage_gpu_hour",
        "minimum_pilot_bootstrap",
        "registered_e5_one_shot",
    ]
    stage_gpu_hour_receipt: CanonicalJsonProofBinding | None
    pilot_launch_budget: CanonicalJsonProofBinding | None
    pilot_budget_verification_receipt: CanonicalJsonProofBinding | None
    e5_one_shot_launch_budget: CanonicalJsonProofBinding | None
    e5_one_shot_budget_verification_receipt: CanonicalJsonProofBinding | None
    failure_execution_binding_sha256: str | None
    reservation: ChallengeReplayReservationBinding
    budget_consumption: CanonicalJsonProofBinding
    hard_timeout_ns: int
    provider_wave_hard_timeout_ns: int
    consumption_path: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_stage_launch_admission"
            or self.protocol_sha256 != FORMAL_STAGE_LAUNCH_ADMISSION_PROTOCOL_SHA256
            or self.stage not in FORMAL_STAGE_DAG
        ):
            raise ValueError("formal stage launch admission schema is unsupported")
        _positive_int("formal launch verification time", self.verified_ns)
        for label, digest in (
            ("registry", self.registry_sha256),
            ("protocol lock", self.protocol_lock_sha256),
            ("root manifest", self.root_manifest_sha256),
            ("runtime authority", self.runtime_authority_manifest_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("cell", self.materialized_cell_id),
            ("execution binding", self.execution_binding_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("run plan", self.run_plan_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _sha256(f"formal launch admission {label}", digest)
        if (
            type(self.capacity_schedule) is not CanonicalJsonProofBinding
            or type(self.capacity_gate) is not CanonicalJsonProofBinding
        ):
            raise TypeError("formal launch capacity artifacts must be path-bound")
        if (
            type(self.capacity_control) is not ControlArtifactAttestation
            or type(self.reservation) is not ChallengeReplayReservationBinding
        ):
            raise TypeError("formal launch control/reservation is not exact")
        if type(self.budget_consumption) is not CanonicalJsonProofBinding:
            raise TypeError("formal launch budget consumption is not path-bound")
        if self.budget_mode == "available_stage_gpu_hour":
            if (
                type(self.stage_gpu_hour_receipt) is not CanonicalJsonProofBinding
                or self.pilot_launch_budget is not None
                or self.pilot_budget_verification_receipt is not None
                or self.e5_one_shot_launch_budget is not None
                or self.e5_one_shot_budget_verification_receipt is not None
                or self.failure_execution_binding_sha256 is not None
            ):
                raise ValueError("formal full-stage launch budget union is not exact")
        elif self.budget_mode == "minimum_pilot_bootstrap":
            if (
                self.stage_gpu_hour_receipt is not None
                or type(self.pilot_launch_budget) is not CanonicalJsonProofBinding
                or type(self.pilot_budget_verification_receipt)
                is not CanonicalJsonProofBinding
                or self.e5_one_shot_launch_budget is not None
                or self.e5_one_shot_budget_verification_receipt is not None
                or self.failure_execution_binding_sha256 is not None
            ):
                raise ValueError("formal pilot launch budget union is not exact")
        elif self.budget_mode == "registered_e5_one_shot":
            if (
                self.stage != "E5"
                or self.stage_gpu_hour_receipt is not None
                or self.pilot_launch_budget is not None
                or self.pilot_budget_verification_receipt is not None
                or type(self.e5_one_shot_launch_budget) is not CanonicalJsonProofBinding
                or type(self.e5_one_shot_budget_verification_receipt)
                is not CanonicalJsonProofBinding
                or self.failure_execution_binding_sha256 is None
            ):
                raise ValueError("formal E5 one-shot launch budget union is not exact")
            _sha256(
                "formal E5 one-shot failure execution",
                self.failure_execution_binding_sha256,
            )
        else:
            raise ValueError("formal launch budget mode is unsupported")
        _positive_int("formal launch hard timeout", self.hard_timeout_ns)
        _positive_int(
            "formal launch provider-wave hard timeout",
            self.provider_wave_hard_timeout_ns,
        )
        if self.hard_timeout_ns > self.provider_wave_hard_timeout_ns:
            raise ValueError("formal launch process timeout exceeds provider wave")
        path = Path(self.consumption_path)
        if (
            not path.is_absolute()
            or path != path.resolve(strict=False)
            or path.name != "formal-stage-launch-consumed.json"
        ):
            raise ValueError("formal launch consumption path is not canonical")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "gpu_uuids": list(self.gpu_uuids),
            "capacity_schedule": self.capacity_schedule.to_dict(),
            "capacity_gate": self.capacity_gate.to_dict(),
            "capacity_control": self.capacity_control.to_dict(),
            "stage_gpu_hour_receipt": (
                None
                if self.stage_gpu_hour_receipt is None
                else self.stage_gpu_hour_receipt.to_dict()
            ),
            "pilot_launch_budget": (
                None
                if self.pilot_launch_budget is None
                else self.pilot_launch_budget.to_dict()
            ),
            "pilot_budget_verification_receipt": (
                None
                if self.pilot_budget_verification_receipt is None
                else self.pilot_budget_verification_receipt.to_dict()
            ),
            "e5_one_shot_launch_budget": (
                None
                if self.e5_one_shot_launch_budget is None
                else self.e5_one_shot_launch_budget.to_dict()
            ),
            "e5_one_shot_budget_verification_receipt": (
                None
                if self.e5_one_shot_budget_verification_receipt is None
                else self.e5_one_shot_budget_verification_receipt.to_dict()
            ),
            "reservation": self.reservation.to_dict(),
            "budget_consumption": self.budget_consumption.to_dict(),
            "admission_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal stage launch admission",
            value,
            {*cls.__dataclass_fields__, "admission_sha256"},
        )
        declared = _sha256("formal stage launch admission", row.pop("admission_sha256"))
        raw_gpus = row["gpu_uuids"]
        if type(raw_gpus) is not list:
            raise TypeError("formal launch GPU UUIDs must be an array")
        row["gpu_uuids"] = tuple(raw_gpus)
        for name in ("capacity_schedule", "capacity_gate"):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        for name in (
            "stage_gpu_hour_receipt",
            "pilot_launch_budget",
            "pilot_budget_verification_receipt",
            "e5_one_shot_launch_budget",
            "e5_one_shot_budget_verification_receipt",
        ):
            if row[name] is not None:
                row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["capacity_control"] = ControlArtifactAttestation.from_dict(
            row["capacity_control"]
        )
        row["reservation"] = ChallengeReplayReservationBinding.from_dict(
            row["reservation"]
        )
        row["budget_consumption"] = CanonicalJsonProofBinding.from_dict(
            row["budget_consumption"]
        )
        admission = cls(**row)
        if admission.sha256 != declared:
            raise ValueError("formal stage launch admission digest differs")
        return admission


def _rebuild_expected_admission(
    *,
    plan: object,
    schedule_binding: CanonicalJsonProofBinding,
    schedule: FormalStageCapacitySchedule,
    gate_binding: CanonicalJsonProofBinding,
    capacity_control: ControlArtifactAttestation,
    root_manifest_sha256: str,
    budget_mode: Literal[
        "available_stage_gpu_hour",
        "minimum_pilot_bootstrap",
        "registered_e5_one_shot",
    ],
    stage_gpu_hour_receipt: CanonicalJsonProofBinding | None,
    pilot_launch_budget: CanonicalJsonProofBinding | None,
    pilot_budget_verification_receipt: CanonicalJsonProofBinding | None,
    e5_one_shot_launch_budget: CanonicalJsonProofBinding | None = None,
    e5_one_shot_budget_verification_receipt: CanonicalJsonProofBinding | None = None,
    failure_execution_binding_sha256: str | None = None,
    reservation: ChallengeReplayReservationBinding,
    budget_consumption: CanonicalJsonProofBinding,
    hard_timeout_ns: int,
    provider_wave_hard_timeout_ns: int,
) -> FormalStageLaunchAdmission:
    """Rebuild every admission field from opened authorities.

    ``plan`` is intentionally duck-typed here to avoid an import cycle with
    the physical dispatcher.  Both callers first reconstruct the strict
    ``FormalServingRunPlan`` codec before invoking this helper.
    """

    return FormalStageLaunchAdmission(
        schema_version=1,
        kind="lightcone_formal_stage_launch_admission",
        protocol_sha256=FORMAL_STAGE_LAUNCH_ADMISSION_PROTOCOL_SHA256,
        verified_ns=reservation.reserved_ns,
        stage=schedule.stage,
        registry_sha256=schedule.registry_sha256,
        protocol_lock_sha256=schedule.protocol_lock_sha256,
        root_manifest_sha256=root_manifest_sha256,
        runtime_authority_manifest_sha256=(schedule.runtime_authority_manifest_sha256),
        materialization_receipt_sha256=(schedule.materialization_receipt_sha256),
        materialized_cell_id=schedule.materialized_cell_id,
        execution_binding_sha256=schedule.execution_binding_sha256,
        execution_plan_sha256=schedule.execution_plan_sha256,
        run_plan_sha256=schedule.run_plan_sha256,
        topology_mode=schedule.topology_mode,
        inventory_sha256=schedule.inventory_sha256,
        gpu_uuids=schedule.gpu_uuids,
        capacity_schedule=schedule_binding,
        capacity_gate=gate_binding,
        capacity_control=capacity_control,
        budget_mode=budget_mode,
        stage_gpu_hour_receipt=stage_gpu_hour_receipt,
        pilot_launch_budget=pilot_launch_budget,
        pilot_budget_verification_receipt=(pilot_budget_verification_receipt),
        e5_one_shot_launch_budget=e5_one_shot_launch_budget,
        e5_one_shot_budget_verification_receipt=(
            e5_one_shot_budget_verification_receipt
        ),
        failure_execution_binding_sha256=failure_execution_binding_sha256,
        reservation=reservation,
        budget_consumption=budget_consumption,
        hard_timeout_ns=hard_timeout_ns,
        provider_wave_hard_timeout_ns=provider_wave_hard_timeout_ns,
        consumption_path=str(
            Path(plan.private_output_root) / "formal-stage-launch-consumed.json"
        ),
    )


def _require_expected_admission(
    artifact: FormalStageLaunchAdmission,
    expected: FormalStageLaunchAdmission,
    *,
    label: str,
) -> None:
    if artifact != expected:
        raise ValueError(f"{label} differs from deterministic rebuild")


_VALIDATED_FORMAL_STAGE_LAUNCH_ADMISSION_SEAL = object()


@dataclass(frozen=True, init=False)
class ValidatedFormalStageLaunchAdmission:
    artifact: FormalStageLaunchAdmission
    capacity_control: VerifiedControlArtifact
    pilot_budget_verification_receipt: FormalPilotBudgetVerificationReceipt | None
    e5_one_shot_budget_verification_receipt: (
        FormalE5OneShotBudgetVerificationReceipt | None
    )
    _construction_seal: object

    def __init__(
        self,
        *,
        artifact: FormalStageLaunchAdmission,
        capacity_control: VerifiedControlArtifact,
        pilot_budget_verification_receipt: (
            FormalPilotBudgetVerificationReceipt | None
        ),
        e5_one_shot_budget_verification_receipt: (
            FormalE5OneShotBudgetVerificationReceipt | None
        ),
        _construction_seal: object,
    ) -> None:
        if _construction_seal is not _VALIDATED_FORMAL_STAGE_LAUNCH_ADMISSION_SEAL:
            raise TypeError("formal stage launch admission is verifier-constructed")
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "capacity_control", capacity_control)
        object.__setattr__(
            self,
            "pilot_budget_verification_receipt",
            pilot_budget_verification_receipt,
        )
        object.__setattr__(
            self,
            "e5_one_shot_budget_verification_receipt",
            e5_one_shot_budget_verification_receipt,
        )
        object.__setattr__(self, "_construction_seal", _construction_seal)


def _require_control_subject(
    control: ControlArtifactAttestation,
    *,
    artifact_type: str,
    artifact_sha256: str,
    protocol_sha256: str,
    registry_sha256: str,
    lineage_sha256: str,
) -> None:
    subject = control.subject
    if (
        subject.artifact_type != artifact_type
        or subject.artifact_sha256 != artifact_sha256
        or subject.protocol_sha256 != protocol_sha256
        or subject.registry_sha256 != registry_sha256
        or subject.lineage_sha256 != lineage_sha256
    ):
        raise ValueError("formal launch control subject differs from artifact")


def _load_schedule_gate(
    schedule_path: str | Path,
    gate_path: str | Path,
    *,
    now_ns: int,
) -> tuple[
    CanonicalJsonProofBinding,
    FormalDynamicStageCapacitySchedule | FormalE5OneShotStageCapacitySchedule,
    CanonicalJsonProofBinding,
    FormalStageCapacityGate,
]:
    schedule_binding = CanonicalJsonProofBinding.bind(schedule_path)
    schedule = _load_capacity_schedule(schedule_path)
    if type(schedule) not in {
        FormalDynamicStageCapacitySchedule,
        FormalE5OneShotStageCapacitySchedule,
    }:
        raise ValueError("formal launch rejects legacy registry-ID capacity schedules")
    gate_binding = CanonicalJsonProofBinding.bind(gate_path)
    gate = FormalStageCapacityGate.from_dict(gate_binding.reopen())
    if (
        schedule.sha256 != schedule_binding.semantic_sha256
        or gate.sha256 != gate_binding.semantic_sha256
    ):
        raise ValueError("formal launch capacity semantic binding differs")
    revalidate_formal_stage_capacity_gate(
        schedule=schedule,
        gate=gate,
        now_ns=now_ns,
    )
    if gate.status != "AVAILABLE":
        raise ValueError("formal launch capacity gate is not AVAILABLE")
    return schedule_binding, schedule, gate_binding, gate


def _budget_ledger_parent(reservation: ChallengeReplayReservationBinding) -> str:
    replay_root = Path(reservation.path).parent
    return str(replay_root.parent / "formal-stage-budget-consumption")


def _full_budget_root(
    binding: CanonicalJsonProofBinding,
    *,
    schedule: FormalStageCapacitySchedule,
    current_ns: int,
) -> tuple[
    FormalStageGpuHourVerificationReceipt,
    str,
    str,
    _FormalStageBudgetLimits,
]:
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("formal launch stage GPU-hour receipt changed")
    receipt = FormalStageGpuHourVerificationReceipt.from_dict(binding.reopen())
    source = receipt.revalidate(current_ns=current_ns)
    lock = receipt.registry_receipt.signed_protocol_lock.payload
    materializations = tuple(
        row.payload
        for row in receipt.registry_receipt.cumulative_signed_materializations
        if row.payload.sha256 == receipt.materialization_receipt_sha256
    )
    if len(materializations) != 1:
        raise ValueError("formal launch GPU-hour materialization is not exact")
    materialization = materializations[0]
    if (
        receipt.stage != schedule.stage
        or receipt.materialization_receipt_sha256
        != schedule.materialization_receipt_sha256
        or materialization.sha256 != schedule.materialization_receipt_sha256
        or schedule.materialized_cell_id
        not in {row.cell_id for row in materialization.cells}
        or receipt.inventory.sha256 != schedule.inventory_sha256
        or lock.sha256 != schedule.protocol_lock_sha256
        or lock.registry_sha256 != schedule.registry_sha256
        or receipt.formal_runtime_authority_manifest.sha256
        != schedule.runtime_authority_manifest_sha256
    ):
        raise ValueError("formal launch AVAILABLE GPU-hour receipt lineage differs")
    from lightcone_spec.experiments.gpu_hour_authority import (
        derive_and_validate_formal_launch_cap_schedule,
    )

    pilot_materialization = None
    if receipt.prospective_pilot_materialization is not None:
        pilot_materialization = stage_materialization_receipt_from_dict(
            receipt.prospective_pilot_materialization.reopen()
        )
    cap_schedule = derive_and_validate_formal_launch_cap_schedule(
        source,
        materialization,
        pilot_materialization=pilot_materialization,
    )
    launchable_caps = tuple(
        row for row in cap_schedule.cell_caps if row.disposition == "LAUNCHABLE"
    )
    if not launchable_caps or schedule.materialized_cell_id not in {
        row.materialized_cell_id for row in launchable_caps
    }:
        raise ValueError(
            "formal launch cell is not launchable in the typed cap schedule"
        )
    selected_cap = cap_schedule.cap_for(schedule.materialized_cell_id)
    if type(schedule) is FormalDynamicStageCapacitySchedule and (
        schedule.stage_gpu_hour_receipt != binding
        or schedule.launch_cap_schedule_sha256 != cap_schedule.sha256
        or schedule.process_hard_timeout_ns
        != selected_cap.process_hard_timeout_ns_per_attempt
        or schedule.provider_wave_hard_timeout_ns
        != selected_cap.provider_wave_hard_timeout_ns_per_attempt
        or schedule.maximum_compute_gpu_ns_per_attempt
        != selected_cap.maximum_compute_gpu_ns_per_attempt
        or schedule.maximum_provider_reserved_gpu_ns_per_attempt
        != selected_cap.maximum_provider_reserved_gpu_ns_per_attempt
    ):
        raise ValueError("formal launch dynamic schedule differs from typed caps")
    gpu_counts = tuple(
        (row.materialized_cell_id, row.gpu_count) for row in launchable_caps
    )
    estimate = receipt.signed_envelope.payload.estimate
    assert estimate.compute_gpu_hours is not None
    assert estimate.reserved_gpu_hours is not None
    assert estimate.retry_reserve_gpu_hours is not None
    base_compute = math.ceil(estimate.compute_gpu_hours * NANOSECONDS_PER_GPU_HOUR)
    retry_compute = math.ceil(
        estimate.retry_reserve_gpu_hours * NANOSECONDS_PER_GPU_HOUR
    )
    signed_maximum_compute = base_compute + retry_compute
    signed_maximum_reserved = math.ceil(
        estimate.reserved_gpu_hours * NANOSECONDS_PER_GPU_HOUR
    )
    cell_ids = tuple(cell_id for cell_id, _count in gpu_counts)
    hard_timeout_by_cell = tuple(
        (row.materialized_cell_id, row.process_hard_timeout_ns_per_attempt)
        for row in launchable_caps
    )
    provider_wave_timeout_by_cell = tuple(
        (
            row.materialized_cell_id,
            row.provider_wave_hard_timeout_ns_per_attempt,
        )
        for row in launchable_caps
    )
    maximum_compute = (
        cap_schedule.launchable_compute_gpu_ns + cap_schedule.retry_reserve_gpu_ns
    )
    maximum_reserved = (
        cap_schedule.launchable_provider_reserved_gpu_ns
        + cap_schedule.retry_reserve_gpu_ns
    )
    if (
        maximum_compute > signed_maximum_compute
        or maximum_reserved > signed_maximum_reserved
    ):
        raise ValueError("typed launch caps exceed the signed GPU-hour envelope")
    limits = _FormalStageBudgetLimits(
        mode="available_stage_gpu_hour",
        authority_sha256=receipt.sha256,
        authority_reservation_sha256=receipt.reservation.reservation_sha256,
        control_replay_root=str(Path(receipt.reservation.path).parent),
        ledger_parent=_budget_ledger_parent(receipt.reservation),
        stage=receipt.stage,
        materialization_receipt_sha256=receipt.materialization_receipt_sha256,
        cell_ids=cell_ids,
        gpu_count_by_cell=gpu_counts,
        provider_reserved_gpu_count_cap_by_cell=tuple(
            (row.materialized_cell_id, row.provider_reserved_gpu_count)
            for row in launchable_caps
        ),
        allowed_attempts_per_cell=launchable_caps[0].allowed_attempts,
        hard_timeout_ns_by_cell=hard_timeout_by_cell,
        provider_wave_hard_timeout_ns_by_cell=(provider_wave_timeout_by_cell),
        compute_charge_gpu_ns_by_cell=tuple(
            (
                row.materialized_cell_id,
                row.maximum_compute_gpu_ns_per_attempt,
            )
            for row in launchable_caps
        ),
        reserved_charge_gpu_ns_by_cell=tuple(
            (
                row.materialized_cell_id,
                row.maximum_provider_reserved_gpu_ns_per_attempt,
            )
            for row in launchable_caps
        ),
        hard_timeout_derivation_sha256=cap_schedule.sha256,
        maximum_compute_gpu_ns=maximum_compute,
        maximum_reserved_gpu_ns=maximum_reserved,
    )
    return (
        receipt,
        lock.offline_release_trust_root_sha256,
        receipt.registry_receipt.trusted_release_policy(current_ns=current_ns).sha256,
        limits,
    )


def _pilot_budget_limits(
    pilot: FormalPilotLaunchBudget,
    verification: FormalPilotBudgetVerificationReceipt,
) -> _FormalStageBudgetLimits:
    return _FormalStageBudgetLimits(
        mode="minimum_pilot_bootstrap",
        authority_sha256=pilot.sha256,
        authority_reservation_sha256=verification.reservation.reservation_sha256,
        control_replay_root=str(Path(verification.reservation.path).parent),
        ledger_parent=_budget_ledger_parent(verification.reservation),
        stage=pilot.stage,
        materialization_receipt_sha256=pilot.materialization_receipt_sha256,
        cell_ids=pilot.minimum_pilot_cell_ids,
        gpu_count_by_cell=pilot.gpu_count_by_cell,
        provider_reserved_gpu_count_cap_by_cell=(
            pilot.provider_reserved_gpu_count_by_cell
        ),
        allowed_attempts_per_cell=pilot.retry_allowance + 1,
        hard_timeout_ns_by_cell=tuple(
            (cell_id, pilot.per_cell_hard_timeout_ns)
            for cell_id in pilot.minimum_pilot_cell_ids
        ),
        provider_wave_hard_timeout_ns_by_cell=tuple(
            (cell_id, pilot.per_cell_hard_timeout_ns)
            for cell_id in pilot.minimum_pilot_cell_ids
        ),
        compute_charge_gpu_ns_by_cell=tuple(
            (
                cell_id,
                pilot.per_cell_hard_timeout_ns * gpu_count,
            )
            for cell_id, gpu_count in pilot.gpu_count_by_cell
        ),
        reserved_charge_gpu_ns_by_cell=tuple(
            (
                cell_id,
                pilot.per_cell_hard_timeout_ns * provider_count,
            )
            for cell_id, provider_count in (pilot.provider_reserved_gpu_count_by_cell)
        ),
        hard_timeout_derivation_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_formal_pilot_timeout_authority",
                "pilot_budget_sha256": pilot.sha256,
                "per_cell_hard_timeout_ns": pilot.per_cell_hard_timeout_ns,
                "retry_allowance": pilot.retry_allowance,
            }
        ),
        maximum_compute_gpu_ns=pilot.maximum_compute_gpu_ns,
        maximum_reserved_gpu_ns=pilot.maximum_reserved_gpu_ns,
    )


def _require_pilot_dynamic_capacity_schedule(
    *,
    schedule: FormalDynamicStageCapacitySchedule,
    pilot: FormalPilotLaunchBudget,
    verification: FormalPilotBudgetVerificationReceipt,
    verification_binding: CanonicalJsonProofBinding,
) -> None:
    dispatch = schedule._dynamic_dispatch
    if type(dispatch) is not FormalPilotDynamicDispatchSchedule:
        raise ValueError("minimum pilot launch requires pilot dispatch authority")
    cell_id = schedule.materialized_cell_id
    gpu_count = dict(pilot.gpu_count_by_cell).get(cell_id)
    provider_count = dict(pilot.provider_reserved_gpu_count_by_cell).get(cell_id)
    if (
        dispatch.pilot_budget_verification_receipt != verification_binding
        or dispatch.pilot_budget_verification_receipt_sha256 != verification.sha256
        or dispatch.pilot_launch_budget != verification.pilot_launch_budget
        or dispatch.pilot_launch_budget_sha256 != pilot.sha256
        or dispatch.launch_nonce_sha256 != pilot.launch_nonce_sha256
        or dispatch.materialization_receipt_sha256
        != pilot.materialization_receipt_sha256
        or dispatch.activated_cell_ids != pilot.minimum_pilot_cell_ids
        or schedule.process_hard_timeout_ns != pilot.per_cell_hard_timeout_ns
        or schedule.provider_wave_hard_timeout_ns != pilot.per_cell_hard_timeout_ns
        or schedule.wave_cell_ids != (cell_id,)
        or schedule.provider_reserved_gpu_count != provider_count
        or schedule.maximum_compute_gpu_ns_per_attempt
        != pilot.per_cell_hard_timeout_ns * gpu_count
        or schedule.maximum_provider_reserved_gpu_ns_per_attempt
        != pilot.per_cell_hard_timeout_ns * provider_count
        or schedule.retry_allowance != pilot.retry_allowance
    ):
        raise ValueError("minimum pilot capacity schedule differs from hard budget")


def _e5_one_shot_budget_limits(
    budget: FormalE5OneShotLaunchBudget,
    verification: FormalE5OneShotBudgetVerificationReceipt,
) -> _FormalStageBudgetLimits:
    caps = budget.cell_caps
    return _FormalStageBudgetLimits(
        mode="registered_e5_one_shot",
        authority_sha256=budget.sha256,
        authority_reservation_sha256=verification.reservation.reservation_sha256,
        control_replay_root=str(Path(verification.reservation.path).parent),
        ledger_parent=_budget_ledger_parent(verification.reservation),
        stage="E5",
        materialization_receipt_sha256=budget.materialization_receipt_sha256,
        cell_ids=tuple(row.materialized_cell_id for row in caps),
        gpu_count_by_cell=tuple(
            (row.materialized_cell_id, row.gpu_count) for row in caps
        ),
        provider_reserved_gpu_count_cap_by_cell=tuple(
            (row.materialized_cell_id, row.provider_reserved_gpu_count) for row in caps
        ),
        allowed_attempts_per_cell=1,
        hard_timeout_ns_by_cell=tuple(
            (row.materialized_cell_id, row.process_hard_timeout_ns) for row in caps
        ),
        provider_wave_hard_timeout_ns_by_cell=tuple(
            (row.materialized_cell_id, row.provider_wave_hard_timeout_ns)
            for row in caps
        ),
        compute_charge_gpu_ns_by_cell=tuple(
            (row.materialized_cell_id, row.maximum_compute_gpu_ns) for row in caps
        ),
        reserved_charge_gpu_ns_by_cell=tuple(
            (row.materialized_cell_id, row.maximum_provider_reserved_gpu_ns)
            for row in caps
        ),
        hard_timeout_derivation_sha256=budget.sha256,
        maximum_compute_gpu_ns=budget.maximum_compute_gpu_ns,
        maximum_reserved_gpu_ns=budget.maximum_provider_reserved_gpu_ns,
    )


def authorize_formal_stage_launch(
    *,
    execution_binding: VerifiedFormalServingExecutionBinding,
    run_plan_path: str | Path,
    capacity_schedule_path: str | Path,
    capacity_gate_path: str | Path,
    capacity_control: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    output_path: str | Path,
    stage_gpu_hour_receipt_path: str | Path | None = None,
    pilot_budget_verification_receipt_path: str | Path | None = None,
    e5_one_shot_budget_verification_receipt_path: str | Path | None = None,
    failure_execution_binding: object | None = None,
) -> FormalStageLaunchAdmission:
    """Verify and reserve one exact pre-allocation launch decision."""

    from lightcone_spec.orchestration.formal_physical_dispatch import (
        load_formal_serving_run_plan,
    )

    verified = require_verified_formal_serving_execution_binding(execution_binding)
    plan = load_formal_serving_run_plan(
        run_plan_path,
        execution_binding=verified,
        verified_nextn_tp2_authority=verified.verified_nextn_tp2_authority,
    )
    schedule_binding, schedule, gate_binding, gate = _load_schedule_gate(
        capacity_schedule_path,
        capacity_gate_path,
        now_ns=now_ns,
    )
    subject = verified.subject
    plan_binding = CanonicalJsonProofBinding.bind(
        run_plan_path, semantic_sha256=plan.sha256
    )
    if (
        schedule.execution_binding_sha256 != verified.sha256
        or schedule.execution_plan_sha256 != subject.execution_plan_sha256
        or schedule.run_plan != plan_binding
        or schedule.run_plan_sha256 != plan.sha256
        or schedule.materialized_cell_id != subject.materialized_cell_id
        or schedule.inventory_sha256 != subject.inventory_sha256
        or schedule.topology_mode != subject.topology_mode
        or schedule.gpu_uuids != subject.gpu_uuids
    ):
        raise ValueError("formal launch schedule differs from sealed plan")
    capacity_lineage = formal_stage_capacity_control_lineage_sha256(
        schedule=schedule, gate=gate
    )
    _require_control_subject(
        capacity_control,
        artifact_type="capacity",
        artifact_sha256=gate.sha256,
        protocol_sha256=FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256,
        registry_sha256=schedule.registry_sha256,
        lineage_sha256=capacity_lineage,
    )

    supplied = (
        stage_gpu_hour_receipt_path is not None,
        pilot_budget_verification_receipt_path is not None,
        e5_one_shot_budget_verification_receipt_path is not None,
    )
    if sum(supplied) != 1:
        raise ValueError("formal launch requires exactly one GPU-hour budget mode")
    if (
        e5_one_shot_budget_verification_receipt_path is None
        and type(schedule) is not FormalDynamicStageCapacitySchedule
    ) or (
        e5_one_shot_budget_verification_receipt_path is not None
        and type(schedule) is not FormalE5OneShotStageCapacitySchedule
    ):
        raise ValueError("formal launch capacity schedule differs from budget mode")
    full_binding = None
    pilot_binding = None
    pilot_verification_binding = None
    e5_budget_binding = None
    e5_verification_binding = None
    failure_execution_binding_sha256 = None
    controls = (capacity_control,)
    expected_attempt_index = None
    if stage_gpu_hour_receipt_path is not None:
        full_binding = CanonicalJsonProofBinding.bind(stage_gpu_hour_receipt_path)
        _receipt, root_sha256, policy_sha256, budget_limits = _full_budget_root(
            full_binding,
            schedule=schedule,
            current_ns=now_ns,
        )
        mode = "available_stage_gpu_hour"
        hard_timeout_ns = budget_limits.hard_timeout_ns(schedule.materialized_cell_id)
    elif pilot_budget_verification_receipt_path is not None:
        pilot_verification_binding = CanonicalJsonProofBinding.bind(
            pilot_budget_verification_receipt_path
        )
        verification, pilot, preflight = (
            revalidate_formal_pilot_budget_verification_receipt(
                pilot_verification_binding,
                current_ns=now_ns,
            )
        )
        _require_pilot_dynamic_capacity_schedule(
            schedule=schedule,
            pilot=pilot,
            verification=verification,
            verification_binding=pilot_verification_binding,
        )
        pilot_binding = verification.pilot_launch_budget
        lock = preflight.registry_receipt.signed_protocol_lock.payload
        if (
            schedule.stage != pilot.stage
            or schedule.materialization_receipt_sha256
            != pilot.materialization_receipt_sha256
            or schedule.materialized_cell_id not in pilot.minimum_pilot_cell_ids
            or schedule.inventory_sha256 != preflight.inventory.sha256
            or schedule.protocol_lock_sha256 != pilot.protocol_lock_sha256
            or schedule.runtime_authority_manifest_sha256
            != pilot.runtime_authority_manifest_sha256
        ):
            raise ValueError("formal launch cell is outside minimum pilot budget")
        root_sha256 = lock.offline_release_trust_root_sha256
        policy_sha256 = preflight.registry_receipt.trusted_release_policy(
            current_ns=now_ns
        ).sha256
        mode = "minimum_pilot_bootstrap"
        hard_timeout_ns = pilot.per_cell_hard_timeout_ns
        budget_limits = _pilot_budget_limits(pilot, verification)
        expected_attempt_index = _next_stage_budget_attempt(
            limits=budget_limits,
            materialized_cell_id=schedule.materialized_cell_id,
        )
    else:
        from lightcone_spec.experiments.formal_failure_execution import (
            require_verified_formal_failure_execution_binding,
        )

        e5_verification_binding = CanonicalJsonProofBinding.bind(
            e5_one_shot_budget_verification_receipt_path
        )
        e5_verification, e5_budget, e5_registry = (
            revalidate_formal_e5_one_shot_budget_verification(
                e5_verification_binding,
                current_ns=now_ns,
            )
        )
        failure = require_verified_formal_failure_execution_binding(
            failure_execution_binding
        )
        cap = e5_budget.cap_for(schedule.materialized_cell_id)
        if (
            failure.serving_execution.sha256 != verified.sha256
            or failure.subject.materialized_cell_id != schedule.materialized_cell_id
            or failure.subject.materialization_receipt_sha256
            != e5_budget.materialization_receipt_sha256
            or failure.subject.protocol_lock_sha256 != e5_budget.protocol_lock_sha256
            or failure.subject.formal_runtime_authority_manifest_sha256
            != e5_budget.runtime_authority_manifest_sha256
            or failure.subject.inventory_sha256 != e5_budget.inventory_sha256
            or failure.subject.registry_sha256 != e5_budget.registry_sha256
            or failure.subject.topology != cap.topology_mode
            or failure.subject.backend != cap.backend
            or failure.subject.scenario != cap.scenario
            or failure.subject.cohort_count != cap.cohort_count
            or schedule.stage != "E5"
            or schedule.materialization_receipt_sha256
            != e5_budget.materialization_receipt_sha256
            or schedule.protocol_lock_sha256 != e5_budget.protocol_lock_sha256
            or schedule.runtime_authority_manifest_sha256
            != e5_budget.runtime_authority_manifest_sha256
            or schedule.registry_sha256 != e5_budget.registry_sha256
            or schedule.inventory_sha256 != e5_budget.inventory_sha256
            or schedule.topology_mode != cap.topology_mode
            or len(schedule.gpu_uuids) != cap.gpu_count
            or schedule.provider_reserved_gpu_count != cap.provider_reserved_gpu_count
            or schedule.failure_execution_binding_sha256 != failure.sha256
        ):
            raise ValueError("formal launch differs from registered E5 one-shot cap")
        e5_budget_binding = e5_verification.budget
        lock = e5_registry.signed_protocol_lock.payload
        root_sha256 = lock.offline_release_trust_root_sha256
        policy_sha256 = e5_registry.trusted_release_policy(current_ns=now_ns).sha256
        mode = "registered_e5_one_shot"
        hard_timeout_ns = cap.process_hard_timeout_ns
        budget_limits = _e5_one_shot_budget_limits(
            e5_budget,
            e5_verification,
        )
        failure_execution_binding_sha256 = failure.sha256
        expected_attempt_index = _next_stage_budget_attempt(
            limits=budget_limits,
            materialized_cell_id=schedule.materialized_cell_id,
        )
    if Path(replay_store.root) != Path(budget_limits.control_replay_root):
        raise ValueError(
            "formal launch replay store differs from budget control authority"
        )
    if any(
        control.deployment_policy_authorization.root_manifest_sha256 != root_sha256
        or control.trusted_attester_policy_sha256 != policy_sha256
        for control in controls
    ):
        raise ValueError("formal launch controls use another release policy")
    verified_controls = verify_and_reserve_release_control_artifact_attestations(
        controls,
        expected_inventory_sha256=schedule.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified_controls, reserved_ns=now_ns
    )
    reservation = replay_store.bind_reservation(reservation_sha256)
    verified_by_envelope = {row.envelope_sha256: row for row in verified_controls}
    if capacity_control.sha256 not in verified_by_envelope:
        raise AssertionError("formal launch capacity control was not reserved")
    budget_consumption = _reserve_stage_budget_consumption(
        limits=budget_limits,
        schedule=schedule,
        gate=gate,
        capacity_control=capacity_control,
        reserved_ns=now_ns,
        expected_attempt_index=expected_attempt_index,
    )
    expected_admission_path = (
        Path(plan.private_output_root) / "formal-stage-launch-admission.json"
    )
    if Path(output_path) != expected_admission_path:
        raise ValueError(
            "formal launch admission output is outside the sealed run root"
        )
    admission = _rebuild_expected_admission(
        plan=plan,
        schedule_binding=schedule_binding,
        schedule=schedule,
        gate_binding=gate_binding,
        capacity_control=capacity_control,
        root_manifest_sha256=root_sha256,
        budget_mode=mode,
        stage_gpu_hour_receipt=full_binding,
        pilot_launch_budget=pilot_binding,
        pilot_budget_verification_receipt=pilot_verification_binding,
        e5_one_shot_launch_budget=e5_budget_binding,
        e5_one_shot_budget_verification_receipt=e5_verification_binding,
        failure_execution_binding_sha256=failure_execution_binding_sha256,
        reservation=reservation,
        budget_consumption=budget_consumption,
        hard_timeout_ns=hard_timeout_ns,
        provider_wave_hard_timeout_ns=dict(
            budget_limits.provider_wave_hard_timeout_ns_by_cell
        )[schedule.materialized_cell_id],
    )
    publish_canonical_json_no_replace(output_path, admission.to_dict())
    reopened = FormalStageLaunchAdmission.from_dict(
        CanonicalJsonProofBinding.bind(output_path).reopen()
    )
    if reopened != admission:
        raise RuntimeError("formal launch admission changed during publication")
    return admission


def validate_formal_stage_launch_admission(
    path: str | Path,
    *,
    execution_binding: VerifiedFormalServingExecutionBinding,
    run_plan_path: str | Path,
    current_ns: int,
) -> ValidatedFormalStageLaunchAdmission:
    """Deep-reopen a durable admission immediately before process allocation."""

    from lightcone_spec.orchestration.formal_physical_dispatch import (
        load_formal_serving_run_plan,
    )

    artifact_binding = CanonicalJsonProofBinding.bind(path)
    artifact = FormalStageLaunchAdmission.from_dict(artifact_binding.reopen())
    verified = require_verified_formal_serving_execution_binding(execution_binding)
    plan = load_formal_serving_run_plan(
        run_plan_path,
        execution_binding=verified,
        verified_nextn_tp2_authority=verified.verified_nextn_tp2_authority,
    )
    schedule_binding, schedule, gate_binding, gate = _load_schedule_gate(
        artifact.capacity_schedule.absolute_path,
        artifact.capacity_gate.absolute_path,
        now_ns=current_ns,
    )
    subject = verified.subject
    if (
        schedule_binding != artifact.capacity_schedule
        or gate_binding != artifact.capacity_gate
        or artifact.registry_sha256 != schedule.registry_sha256
        or artifact.protocol_lock_sha256 != schedule.protocol_lock_sha256
        or artifact.runtime_authority_manifest_sha256
        != schedule.runtime_authority_manifest_sha256
        or artifact.stage != schedule.stage
        or artifact.materialization_receipt_sha256
        != schedule.materialization_receipt_sha256
        or artifact.materialized_cell_id != schedule.materialized_cell_id
        or artifact.execution_binding_sha256 != schedule.execution_binding_sha256
        or artifact.execution_plan_sha256 != schedule.execution_plan_sha256
        or artifact.run_plan_sha256 != schedule.run_plan_sha256
        or artifact.topology_mode != schedule.topology_mode
        or artifact.inventory_sha256 != schedule.inventory_sha256
        or artifact.gpu_uuids != schedule.gpu_uuids
        or artifact.execution_binding_sha256 != verified.sha256
        or artifact.execution_plan_sha256 != subject.execution_plan_sha256
        or artifact.run_plan_sha256 != plan.sha256
        or artifact.materialized_cell_id != subject.materialized_cell_id
        or artifact.materialization_receipt_sha256
        != subject.materialization_receipt_sha256
        or artifact.stage != subject.stage
        or artifact.registry_sha256 != subject.execution_identity.registry_sha256
        or artifact.protocol_lock_sha256 != subject.protocol_lock_sha256
        or artifact.runtime_authority_manifest_sha256
        != subject.formal_runtime_authority_manifest_sha256
        or artifact.inventory_sha256 != subject.inventory_sha256
        or artifact.topology_mode != subject.topology_mode
        or artifact.gpu_uuids != subject.gpu_uuids
        or schedule.run_plan
        != CanonicalJsonProofBinding.bind(run_plan_path, semantic_sha256=plan.sha256)
        or Path(path)
        != Path(plan.private_output_root) / "formal-stage-launch-admission.json"
        or artifact.consumption_path
        != str(Path(plan.private_output_root) / "formal-stage-launch-consumed.json")
    ):
        raise ValueError("formal launch admission differs from exact execution")
    _require_control_subject(
        artifact.capacity_control,
        artifact_type="capacity",
        artifact_sha256=gate.sha256,
        protocol_sha256=FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256,
        registry_sha256=artifact.registry_sha256,
        lineage_sha256=formal_stage_capacity_control_lineage_sha256(
            schedule=schedule, gate=gate
        ),
    )
    controls = (artifact.capacity_control,)
    pilot_verification: FormalPilotBudgetVerificationReceipt | None = None
    e5_verification: FormalE5OneShotBudgetVerificationReceipt | None = None
    if (
        artifact.budget_mode == "registered_e5_one_shot"
        and type(schedule) is not FormalE5OneShotStageCapacitySchedule
    ) or (
        artifact.budget_mode != "registered_e5_one_shot"
        and type(schedule) is not FormalDynamicStageCapacitySchedule
    ):
        raise ValueError("formal launch durable capacity/budget mode differs")
    if artifact.budget_mode == "available_stage_gpu_hour":
        assert artifact.stage_gpu_hour_receipt is not None
        _receipt, root_sha256, policy_sha256, budget_limits = _full_budget_root(
            artifact.stage_gpu_hour_receipt,
            schedule=schedule,
            current_ns=current_ns,
        )
    elif artifact.budget_mode == "minimum_pilot_bootstrap":
        assert artifact.pilot_launch_budget is not None
        assert artifact.pilot_budget_verification_receipt is not None
        if (
            CanonicalJsonProofBinding.bind(artifact.pilot_launch_budget.absolute_path)
            != artifact.pilot_launch_budget
        ):
            raise ValueError("formal pilot launch budget binding changed")
        pilot_verification, pilot, preflight = (
            revalidate_formal_pilot_budget_verification_receipt(
                artifact.pilot_budget_verification_receipt,
                current_ns=current_ns,
            )
        )
        _require_pilot_dynamic_capacity_schedule(
            schedule=schedule,
            pilot=pilot,
            verification=pilot_verification,
            verification_binding=artifact.pilot_budget_verification_receipt,
        )
        if artifact.pilot_launch_budget != pilot_verification.pilot_launch_budget:
            raise ValueError("formal pilot verification budget binding differs")
        if (
            artifact.materialized_cell_id not in pilot.minimum_pilot_cell_ids
            or artifact.hard_timeout_ns != pilot.per_cell_hard_timeout_ns
        ):
            raise ValueError("formal launch admission exceeds pilot budget")
        lock = preflight.registry_receipt.signed_protocol_lock.payload
        root_sha256 = lock.offline_release_trust_root_sha256
        policy_sha256 = preflight.registry_receipt.trusted_release_policy(
            current_ns=current_ns
        ).sha256
        budget_limits = _pilot_budget_limits(pilot, pilot_verification)
    else:
        assert artifact.e5_one_shot_launch_budget is not None
        assert artifact.e5_one_shot_budget_verification_receipt is not None
        assert artifact.failure_execution_binding_sha256 is not None
        e5_verification, e5_budget, e5_registry = (
            revalidate_formal_e5_one_shot_budget_verification(
                artifact.e5_one_shot_budget_verification_receipt,
                current_ns=current_ns,
            )
        )
        cap = e5_budget.cap_for(artifact.materialized_cell_id)
        if (
            artifact.e5_one_shot_launch_budget != e5_verification.budget
            or artifact.stage != "E5"
            or artifact.materialization_receipt_sha256
            != e5_budget.materialization_receipt_sha256
            or artifact.protocol_lock_sha256 != e5_budget.protocol_lock_sha256
            or artifact.runtime_authority_manifest_sha256
            != e5_budget.runtime_authority_manifest_sha256
            or artifact.registry_sha256 != e5_budget.registry_sha256
            or artifact.inventory_sha256 != e5_budget.inventory_sha256
            or artifact.topology_mode != cap.topology_mode
            or len(artifact.gpu_uuids) != cap.gpu_count
            or schedule.failure_execution_binding_sha256
            != artifact.failure_execution_binding_sha256
        ):
            raise ValueError("formal launch admission exceeds E5 one-shot budget")
        lock = e5_registry.signed_protocol_lock.payload
        root_sha256 = lock.offline_release_trust_root_sha256
        policy_sha256 = e5_registry.trusted_release_policy(current_ns=current_ns).sha256
        budget_limits = _e5_one_shot_budget_limits(
            e5_budget,
            e5_verification,
        )
    if (
        artifact.root_manifest_sha256 != root_sha256
        or artifact.hard_timeout_ns
        != budget_limits.hard_timeout_ns(artifact.materialized_cell_id)
        or artifact.provider_wave_hard_timeout_ns
        != dict(budget_limits.provider_wave_hard_timeout_ns_by_cell)[
            artifact.materialized_cell_id
        ]
        or any(
            control.deployment_policy_authorization.root_manifest_sha256 != root_sha256
            or control.trusted_attester_policy_sha256 != policy_sha256
            for control in controls
        )
    ):
        raise ValueError("formal launch durable control policy differs")
    verified_controls = tuple(
        verify_release_control_artifact_attestation(
            control,
            expected_inventory_sha256=artifact.inventory_sha256,
            now_ns=artifact.verified_ns,
            consumed_challenge_sha256s=(),
        )
        for control in controls
    )
    expected_reservation = control_challenge_reservation_sha256(
        verified_controls,
        reserved_ns=artifact.verified_ns,
    )
    reserved = artifact.reservation.revalidate()
    expected_challenges = tuple(
        sorted(
            {
                *(row.challenge_sha256 for row in verified_controls),
                *(row.deployment_policy_challenge_sha256 for row in verified_controls),
            }
        )
    )
    expected_reservation_binding = ChallengeReplayStore(
        budget_limits.control_replay_root
    ).bind_reservation(expected_reservation)
    if (
        artifact.reservation.reservation_sha256 != expected_reservation
        or artifact.reservation != expected_reservation_binding
        or reserved != expected_challenges
    ):
        raise ValueError("formal launch durable replay reservation differs")
    _validate_stage_budget_consumption(
        artifact.budget_consumption,
        limits=budget_limits,
        schedule=schedule,
        gate=gate,
        capacity_control=artifact.capacity_control,
    )
    expected_artifact = _rebuild_expected_admission(
        plan=plan,
        schedule_binding=schedule_binding,
        schedule=schedule,
        gate_binding=gate_binding,
        capacity_control=artifact.capacity_control,
        root_manifest_sha256=root_sha256,
        budget_mode=artifact.budget_mode,
        stage_gpu_hour_receipt=artifact.stage_gpu_hour_receipt,
        pilot_launch_budget=(
            None
            if pilot_verification is None
            else pilot_verification.pilot_launch_budget
        ),
        pilot_budget_verification_receipt=(
            None
            if pilot_verification is None
            else artifact.pilot_budget_verification_receipt
        ),
        e5_one_shot_launch_budget=(
            None if e5_verification is None else e5_verification.budget
        ),
        e5_one_shot_budget_verification_receipt=(
            None
            if e5_verification is None
            else artifact.e5_one_shot_budget_verification_receipt
        ),
        failure_execution_binding_sha256=(
            None
            if e5_verification is None
            else artifact.failure_execution_binding_sha256
        ),
        reservation=expected_reservation_binding,
        budget_consumption=artifact.budget_consumption,
        hard_timeout_ns=budget_limits.hard_timeout_ns(schedule.materialized_cell_id),
        provider_wave_hard_timeout_ns=dict(
            budget_limits.provider_wave_hard_timeout_ns_by_cell
        )[schedule.materialized_cell_id],
    )
    _require_expected_admission(
        artifact,
        expected_artifact,
        label="formal launch admission",
    )
    by_envelope = {row.envelope_sha256: row for row in verified_controls}
    return ValidatedFormalStageLaunchAdmission(
        artifact=artifact,
        capacity_control=by_envelope[artifact.capacity_control.sha256],
        pilot_budget_verification_receipt=pilot_verification,
        e5_one_shot_budget_verification_receipt=e5_verification,
        _construction_seal=_VALIDATED_FORMAL_STAGE_LAUNCH_ADMISSION_SEAL,
    )


def validate_formal_stage_launch_evidence_lineage(
    *,
    admission: CanonicalJsonProofBinding,
    launch_consumption: CanonicalJsonProofBinding,
    budget_consumption: CanonicalJsonProofBinding,
    run_plan_path: str | Path,
    current_ns: int,
) -> FormalStageLaunchAdmission:
    """Deep-open persisted admission evidence after the physical process exits.

    Unlike the pre-allocation validator, this verifier intentionally consumes
    only durable artifacts.  It lets offline terminal, ITL, lifecycle, and E5
    proof builders reject evidence from an unadmitted or foreign launch without
    retaining an in-memory verifier capability from the execution process.
    """

    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRunPlan,
    )

    if type(current_ns) is not int or current_ns < 1:
        raise ValueError("formal launch evidence validation time is invalid")
    rebound_admission = CanonicalJsonProofBinding.bind(admission.absolute_path)
    rebound_launch = CanonicalJsonProofBinding.bind(launch_consumption.absolute_path)
    rebound_budget = CanonicalJsonProofBinding.bind(budget_consumption.absolute_path)
    if (
        rebound_admission != admission
        or rebound_launch != launch_consumption
        or rebound_budget != budget_consumption
    ):
        raise ValueError("formal launch evidence path identity changed")
    artifact = FormalStageLaunchAdmission.from_dict(admission.reopen())
    if artifact.sha256 != admission.semantic_sha256:
        raise ValueError("formal launch admission semantic identity changed")
    consumption = FormalStageLaunchConsumption.from_dict(launch_consumption.reopen())
    if consumption.consumed_ns > current_ns:
        raise ValueError("formal launch consumption is in the future")
    evidence_ns = consumption.consumed_ns
    plan_binding = CanonicalJsonProofBinding.bind(run_plan_path)
    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    if plan.sha256 != plan_binding.semantic_sha256:
        raise ValueError("formal launch run-plan semantic identity changed")
    schedule_binding, schedule, gate_binding, gate = _load_schedule_gate(
        artifact.capacity_schedule.absolute_path,
        artifact.capacity_gate.absolute_path,
        now_ns=evidence_ns,
    )
    if (
        schedule_binding != artifact.capacity_schedule
        or gate_binding != artifact.capacity_gate
        or artifact.run_plan_sha256 != plan.sha256
        or schedule.run_plan != plan_binding
        or schedule.run_plan_sha256 != plan.sha256
        or artifact.execution_binding_sha256 != plan.execution_binding_sha256
        or schedule.execution_binding_sha256 != plan.execution_binding_sha256
        or artifact.execution_plan_sha256
        != plan.native_terminal_binding.execution_plan_sha256
        or schedule.execution_plan_sha256
        != plan.native_terminal_binding.execution_plan_sha256
        or artifact.materialized_cell_id != plan.materialized_cell_id
        or schedule.materialized_cell_id != plan.materialized_cell_id
        or artifact.stage != plan.stage
        or schedule.stage != plan.stage
        or artifact.registry_sha256 != schedule.registry_sha256
        or artifact.protocol_lock_sha256 != schedule.protocol_lock_sha256
        or artifact.runtime_authority_manifest_sha256
        != schedule.runtime_authority_manifest_sha256
        or artifact.inventory_sha256 != plan.inventory_sha256
        or schedule.inventory_sha256 != plan.inventory_sha256
        or artifact.topology_mode != plan.topology_mode
        or schedule.topology_mode != plan.topology_mode
        or artifact.gpu_uuids != plan.gpu_uuids
        or schedule.gpu_uuids != plan.gpu_uuids
        or artifact.materialization_receipt_sha256
        != schedule.materialization_receipt_sha256
        or Path(admission.absolute_path)
        != Path(plan.private_output_root) / "formal-stage-launch-admission.json"
        or artifact.consumption_path
        != str(Path(plan.private_output_root) / "formal-stage-launch-consumed.json")
    ):
        raise ValueError("formal launch evidence differs from exact run plan")
    _require_control_subject(
        artifact.capacity_control,
        artifact_type="capacity",
        artifact_sha256=gate.sha256,
        protocol_sha256=FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256,
        registry_sha256=artifact.registry_sha256,
        lineage_sha256=formal_stage_capacity_control_lineage_sha256(
            schedule=schedule, gate=gate
        ),
    )
    controls = (artifact.capacity_control,)
    if (
        artifact.budget_mode == "registered_e5_one_shot"
        and type(schedule) is not FormalE5OneShotStageCapacitySchedule
    ) or (
        artifact.budget_mode != "registered_e5_one_shot"
        and type(schedule) is not FormalDynamicStageCapacitySchedule
    ):
        raise ValueError("formal launch evidence capacity/budget mode differs")
    if artifact.budget_mode == "available_stage_gpu_hour":
        if artifact.stage_gpu_hour_receipt is None:
            raise ValueError("formal launch lost its stage GPU-hour receipt")
        _receipt, root_sha256, policy_sha256, limits = _full_budget_root(
            artifact.stage_gpu_hour_receipt,
            schedule=schedule,
            current_ns=evidence_ns,
        )
    elif artifact.budget_mode == "minimum_pilot_bootstrap":
        if (
            artifact.pilot_launch_budget is None
            or artifact.pilot_budget_verification_receipt is None
        ):
            raise ValueError("formal launch lost its pilot budget authority")
        verification, pilot, preflight = (
            revalidate_formal_pilot_budget_verification_receipt(
                artifact.pilot_budget_verification_receipt,
                current_ns=evidence_ns,
            )
        )
        _require_pilot_dynamic_capacity_schedule(
            schedule=schedule,
            pilot=pilot,
            verification=verification,
            verification_binding=artifact.pilot_budget_verification_receipt,
        )
        if (
            artifact.pilot_launch_budget != verification.pilot_launch_budget
            or artifact.materialized_cell_id not in pilot.minimum_pilot_cell_ids
        ):
            raise ValueError("formal launch is outside persisted pilot budget")
        lock = preflight.registry_receipt.signed_protocol_lock.payload
        root_sha256 = lock.offline_release_trust_root_sha256
        policy_sha256 = preflight.registry_receipt.trusted_release_policy(
            current_ns=evidence_ns
        ).sha256
        limits = _pilot_budget_limits(pilot, verification)
    elif artifact.budget_mode == "registered_e5_one_shot":
        if (
            artifact.e5_one_shot_launch_budget is None
            or artifact.e5_one_shot_budget_verification_receipt is None
            or artifact.failure_execution_binding_sha256 is None
        ):
            raise ValueError("formal launch lost its E5 one-shot authority")
        e5_verification, e5_budget, e5_registry = (
            revalidate_formal_e5_one_shot_budget_verification(
                artifact.e5_one_shot_budget_verification_receipt,
                current_ns=evidence_ns,
            )
        )
        cap = e5_budget.cap_for(artifact.materialized_cell_id)
        if (
            artifact.e5_one_shot_launch_budget != e5_verification.budget
            or artifact.stage != "E5"
            or artifact.materialization_receipt_sha256
            != e5_budget.materialization_receipt_sha256
            or artifact.protocol_lock_sha256 != e5_budget.protocol_lock_sha256
            or artifact.runtime_authority_manifest_sha256
            != e5_budget.runtime_authority_manifest_sha256
            or artifact.registry_sha256 != e5_budget.registry_sha256
            or artifact.inventory_sha256 != e5_budget.inventory_sha256
            or artifact.topology_mode != cap.topology_mode
            or len(artifact.gpu_uuids) != cap.gpu_count
            or schedule.failure_execution_binding_sha256
            != artifact.failure_execution_binding_sha256
        ):
            raise ValueError("formal launch is outside persisted E5 one-shot budget")
        lock = e5_registry.signed_protocol_lock.payload
        root_sha256 = lock.offline_release_trust_root_sha256
        policy_sha256 = e5_registry.trusted_release_policy(
            current_ns=evidence_ns
        ).sha256
        limits = _e5_one_shot_budget_limits(e5_budget, e5_verification)
    else:  # pragma: no cover - closed by the strict codec
        raise ValueError("formal launch budget mode is unsupported")
    if (
        artifact.root_manifest_sha256 != root_sha256
        or artifact.hard_timeout_ns
        != limits.hard_timeout_ns(artifact.materialized_cell_id)
        or artifact.provider_wave_hard_timeout_ns
        != dict(limits.provider_wave_hard_timeout_ns_by_cell)[
            artifact.materialized_cell_id
        ]
    ):
        raise ValueError("formal launch release root/timeout differs")
    verified_controls = tuple(
        verify_release_control_artifact_attestation(
            control,
            expected_inventory_sha256=artifact.inventory_sha256,
            now_ns=artifact.verified_ns,
            consumed_challenge_sha256s=(),
        )
        for control in controls
    )
    if any(
        control.deployment_policy_authorization.root_manifest_sha256 != root_sha256
        or control.trusted_attester_policy_sha256 != policy_sha256
        for control in controls
    ):
        raise ValueError("formal launch control policy differs")
    expected_reservation = control_challenge_reservation_sha256(
        verified_controls, reserved_ns=artifact.verified_ns
    )
    expected_challenges = tuple(
        sorted(
            {
                *(row.challenge_sha256 for row in verified_controls),
                *(row.deployment_policy_challenge_sha256 for row in verified_controls),
            }
        )
    )
    expected_reservation_binding = ChallengeReplayStore(
        limits.control_replay_root
    ).bind_reservation(expected_reservation)
    if (
        artifact.reservation.reservation_sha256 != expected_reservation
        or artifact.reservation != expected_reservation_binding
        or artifact.reservation.revalidate() != expected_challenges
    ):
        raise ValueError("formal launch persisted replay reservation differs")
    if artifact.budget_consumption != budget_consumption:
        raise ValueError("formal launch budget-consumption binding differs")
    _validate_stage_budget_consumption(
        budget_consumption,
        limits=limits,
        schedule=schedule,
        gate=gate,
        capacity_control=artifact.capacity_control,
    )
    expected_artifact = _rebuild_expected_admission(
        plan=plan,
        schedule_binding=schedule_binding,
        schedule=schedule,
        gate_binding=gate_binding,
        capacity_control=artifact.capacity_control,
        root_manifest_sha256=root_sha256,
        budget_mode=artifact.budget_mode,
        stage_gpu_hour_receipt=artifact.stage_gpu_hour_receipt,
        pilot_launch_budget=(
            artifact.pilot_launch_budget
            if artifact.budget_mode == "minimum_pilot_bootstrap"
            else None
        ),
        pilot_budget_verification_receipt=(
            artifact.pilot_budget_verification_receipt
            if artifact.budget_mode == "minimum_pilot_bootstrap"
            else None
        ),
        e5_one_shot_launch_budget=(
            artifact.e5_one_shot_launch_budget
            if artifact.budget_mode == "registered_e5_one_shot"
            else None
        ),
        e5_one_shot_budget_verification_receipt=(
            artifact.e5_one_shot_budget_verification_receipt
            if artifact.budget_mode == "registered_e5_one_shot"
            else None
        ),
        failure_execution_binding_sha256=(
            artifact.failure_execution_binding_sha256
            if artifact.budget_mode == "registered_e5_one_shot"
            else None
        ),
        reservation=expected_reservation_binding,
        budget_consumption=budget_consumption,
        hard_timeout_ns=limits.hard_timeout_ns(schedule.materialized_cell_id),
        provider_wave_hard_timeout_ns=dict(
            limits.provider_wave_hard_timeout_ns_by_cell
        )[schedule.materialized_cell_id],
    )
    _require_expected_admission(
        artifact,
        expected_artifact,
        label="formal launch evidence",
    )
    if (
        launch_consumption.absolute_path != artifact.consumption_path
        or launch_consumption.semantic_sha256 != consumption.sha256
        or consumption.admission_sha256 != artifact.sha256
        or consumption.materialization_receipt_sha256
        != artifact.materialization_receipt_sha256
        or consumption.materialized_cell_id != artifact.materialized_cell_id
        or consumption.execution_plan_sha256 != artifact.execution_plan_sha256
        or consumption.run_plan_sha256 != artifact.run_plan_sha256
        or consumption.reservation_sha256 != artifact.reservation.reservation_sha256
        or consumption.consumed_ns < artifact.verified_ns
    ):
        raise ValueError("formal launch one-shot consumption lineage differs")
    return artifact


def consume_formal_stage_launch_admission(
    admission: ValidatedFormalStageLaunchAdmission,
    *,
    consumed_ns: int,
) -> CanonicalJsonProofBinding:
    """Commit the one-shot launch claim before any allocation can occur."""

    if (
        type(admission) is not ValidatedFormalStageLaunchAdmission
        or admission._construction_seal
        is not _VALIDATED_FORMAL_STAGE_LAUNCH_ADMISSION_SEAL
    ):
        raise TypeError("formal launch consumption requires verified admission")
    if type(consumed_ns) is not int or consumed_ns < admission.artifact.verified_ns:
        raise ValueError("formal launch consumption time is invalid")
    artifact = admission.artifact
    consumption = FormalStageLaunchConsumption(
        schema_version=1,
        kind="lightcone_formal_stage_launch_consumption",
        admission_sha256=artifact.sha256,
        materialization_receipt_sha256=artifact.materialization_receipt_sha256,
        materialized_cell_id=artifact.materialized_cell_id,
        execution_plan_sha256=artifact.execution_plan_sha256,
        run_plan_sha256=artifact.run_plan_sha256,
        reservation_sha256=artifact.reservation.reservation_sha256,
        consumed_ns=consumed_ns,
    )
    publish_canonical_json_no_replace(
        artifact.consumption_path,
        consumption.to_dict(),
    )
    binding = CanonicalJsonProofBinding.bind(
        artifact.consumption_path, semantic_sha256=consumption.sha256
    )
    if FormalStageLaunchConsumption.from_dict(binding.reopen()) != consumption:
        raise RuntimeError("formal launch consumption changed during publication")
    return binding


__all__ = [
    "FORMAL_PILOT_BUDGET_VERIFICATION_PROTOCOL_SHA256",
    "FORMAL_PILOT_CELL_HARD_TIMEOUT_NS",
    "FORMAL_PILOT_LAUNCH_BUDGET_PROTOCOL_SHA256",
    "FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256",
    "FORMAL_STAGE_LAUNCH_ADMISSION_PROTOCOL_SHA256",
    "FormalPilotBudgetVerificationReceipt",
    "FormalPilotLaunchBudget",
    "FormalStageCapacityGate",
    "FormalStageCapacitySchedule",
    "FormalStageLaunchAdmission",
    "FormalStageLaunchConsumption",
    "ValidatedFormalStageLaunchAdmission",
    "authorize_formal_stage_launch",
    "consume_formal_stage_launch_admission",
    "formal_pilot_launch_budget_control_lineage_sha256",
    "formal_stage_capacity_control_lineage_sha256",
    "materialize_formal_stage_capacity_gate",
    "materialize_formal_stage_capacity_schedule",
    "revalidate_formal_pilot_budget_verification_receipt",
    "revalidate_formal_pilot_launch_budget",
    "revalidate_formal_stage_capacity_gate",
    "validate_formal_stage_launch_admission",
    "validate_formal_stage_launch_evidence_lineage",
    "verify_and_publish_formal_pilot_budget_receipt",
]
