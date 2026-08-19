"""Source-owned physical waves for dynamically materialized formal cells.

The historical industrial scheduler addresses static ``ExperimentRegistry``
cell IDs.  Formal stages address hashes of ``MaterializedCell`` values instead,
so relabeling the historical schedule is not an admissible bridge.  This module
builds a new, path-bound dispatch schedule from the signed GPU-hour source and
its exact ``FormalLaunchCapSchedule``.  In particular, the cap schedule owns
the wave groups; an operator cannot pair two TP1 cells merely to improve
throughput.
"""

from __future__ import annotations

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
from lightcone_spec.experiments.formal_protocol import FORMAL_STAGE_DAG, content_sha256
from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_stage_execution import (
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.gpu_hour_authority import (
    FormalLaunchCapSchedule,
    derive_and_validate_formal_launch_cap_schedule,
)
from lightcone_spec.experiments.stage_materialization import (
    StageMaterializationReceipt,
)
from lightcone_spec.orchestration.formal_physical_dispatch import (
    FormalServingRunPlan,
    load_formal_serving_run_plan,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_DYNAMIC_DISPATCH_PORT_BASE = 31_000
FORMAL_DYNAMIC_DISPATCH_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_dynamic_dispatch_protocol",
        "cell_identity": "exact_signed_materialized_cell_not_registry_alias",
        "wave_authority": "formal_launch_cap_schedule_wave_group_sha256",
        "placement": "canonical_inventory_order_and_topology",
        "ports": "code_owned_per_wave_31000_31001",
        "pair": ("exact_two_tp1_distinct_gpu_port_private_root_same_cap_and_wave"),
        "preconsumed": "never_scheduled",
        "sources": "path_bound_materialization_gpu_hour_receipt_and_cap_schedule",
    }
)
FORMAL_DYNAMIC_STAGE_CAPACITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_dynamic_stage_capacity_protocol",
        "dispatch": FORMAL_DYNAMIC_DISPATCH_PROTOCOL_SHA256,
        "capacity": "fresh_raw_source_exact_dynamic_materialized_cells",
        "projection": "one_exact_work_item_and_its_source_owned_wave",
    }
)
FORMAL_PILOT_DYNAMIC_DISPATCH_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_pilot_dynamic_dispatch_protocol",
        "authority": (
            "durably_verified_minimum_pilot_budget_and_exact_materialization"
        ),
        "coverage": "exact_minimum_pilot_cell_ids_only",
        "waves": "one_source_owned_singleton_wave_per_pilot_cell",
        "placement": "canonical_inventory_order_and_topology",
        "ports": "code_owned_31000",
        "caps": ("pilot_process_timeout_times_gang_and_provider_reserved_counts"),
        "attempts": "registered_retry_allowance_plus_one",
        "full_stage_projection": "forbidden",
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


def _strict(label: str, value: object, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _materialization(
    binding: CanonicalJsonProofBinding,
) -> StageMaterializationReceipt:
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("formal dynamic materialization binding changed")
    receipt = stage_materialization_receipt_from_dict(binding.reopen())
    if receipt.sha256 != binding.semantic_sha256:
        raise ValueError("formal dynamic materialization semantic identity differs")
    return receipt


def _pilot_materialization(
    receipt: FormalStageGpuHourVerificationReceipt,
) -> StageMaterializationReceipt | None:
    binding = receipt.prospective_pilot_materialization
    if binding is None:
        return None
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("formal dynamic pilot materialization binding changed")
    result = stage_materialization_receipt_from_dict(binding.reopen())
    if result.sha256 != binding.semantic_sha256:
        raise ValueError("formal dynamic pilot materialization identity differs")
    return result


def _load_stage_budget(
    binding: CanonicalJsonProofBinding,
    *,
    materialization: StageMaterializationReceipt,
    current_ns: int,
) -> tuple[
    FormalStageGpuHourVerificationReceipt,
    FormalLaunchCapSchedule,
]:
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("formal dynamic GPU-hour receipt binding changed")
    receipt = FormalStageGpuHourVerificationReceipt.from_dict(binding.reopen())
    source = receipt.revalidate(current_ns=current_ns)
    registry_receipt = receipt.registry_receipt
    lock = registry_receipt.signed_protocol_lock.payload
    registered = tuple(
        row.payload
        for row in registry_receipt.cumulative_signed_materializations
        if row.payload.sha256 == materialization.sha256
    )
    if (
        receipt.sha256 != binding.semantic_sha256
        or receipt.materialization_receipt_sha256 != materialization.sha256
        or len(registered) != 1
        or registered[0] != materialization
        or receipt.stage != materialization.stage
        or receipt.signed_envelope.payload.protocol_lock_sha256 != lock.sha256
    ):
        raise ValueError("formal dynamic GPU-hour/materialization lineage differs")
    schedule = derive_and_validate_formal_launch_cap_schedule(
        source,
        materialization,
        pilot_materialization=_pilot_materialization(receipt),
    )
    if (
        schedule.materialization_receipt_sha256 != materialization.sha256
        or schedule.protocol_lock_sha256 != lock.sha256
        or schedule.inventory_sha256 != receipt.inventory.sha256
    ):
        raise ValueError("formal dynamic launch cap lineage differs")
    return receipt, schedule


def materialize_formal_launch_cap_schedule(
    *,
    materialization_path: str | Path,
    stage_gpu_hour_receipt_path: str | Path,
    output_path: str | Path,
    current_ns: int,
) -> FormalLaunchCapSchedule:
    """Deep-build and atomically publish the only normal-stage launch caps."""

    materialization_binding = CanonicalJsonProofBinding.bind(materialization_path)
    materialization = _materialization(materialization_binding)
    budget_binding = CanonicalJsonProofBinding.bind(stage_gpu_hour_receipt_path)
    _receipt, schedule = _load_stage_budget(
        budget_binding,
        materialization=materialization,
        current_ns=current_ns,
    )
    publish_canonical_json_no_replace(output_path, schedule.to_dict())
    reopened = FormalLaunchCapSchedule.from_dict(
        CanonicalJsonProofBinding.bind(output_path).reopen()
    )
    if reopened != schedule:
        raise RuntimeError("formal launch cap schedule changed during publication")
    return schedule


@dataclass(frozen=True)
class FormalDynamicDispatchWorkItem:
    materialized_cell_id: str
    execution_binding_sha256: str
    execution_plan_sha256: str
    run_plan: CanonicalJsonProofBinding
    run_plan_sha256: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    gpu_uuids: tuple[str, ...]
    rank_groups: tuple[tuple[str, ...], ...]
    localhost_port: int
    private_output_root: str
    wave_group_sha256: str
    process_hard_timeout_ns: int
    provider_wave_hard_timeout_ns: int
    provider_reserved_gpu_count: Literal[1, 2]
    maximum_compute_gpu_ns_per_attempt: int
    maximum_provider_reserved_gpu_ns_per_attempt: int
    allowed_attempts: Literal[2]

    def __post_init__(self) -> None:
        for label, digest in (
            ("cell", self.materialized_cell_id),
            ("execution binding", self.execution_binding_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("run plan", self.run_plan_sha256),
            ("wave group", self.wave_group_sha256),
        ):
            _sha256(f"formal dynamic work item {label}", digest)
        if type(self.run_plan) is not CanonicalJsonProofBinding:
            raise TypeError("formal dynamic work item run plan is not path-bound")
        expected_gpus = 1 if self.topology_mode == "tp1_dp1" else 2
        expected_groups = (
            ((self.gpu_uuids[0],),)
            if self.topology_mode == "tp1_dp1"
            else (
                (self.gpu_uuids,)
                if self.topology_mode == "tp2_dp1"
                else tuple((uuid,) for uuid in self.gpu_uuids)
            )
        )
        root = Path(self.private_output_root)
        if (
            len(self.gpu_uuids) != expected_gpus
            or len(set(self.gpu_uuids)) != expected_gpus
            or self.rank_groups != expected_groups
            or type(self.localhost_port) is not int
            or self.localhost_port
            not in {
                FORMAL_DYNAMIC_DISPATCH_PORT_BASE,
                FORMAL_DYNAMIC_DISPATCH_PORT_BASE + 1,
            }
            or not root.is_absolute()
            or root != root.resolve(strict=False)
            or self.provider_reserved_gpu_count not in {1, 2}
            or self.allowed_attempts != 2
        ):
            raise ValueError("formal dynamic work item placement differs")
        for label, value in (
            ("process timeout", self.process_hard_timeout_ns),
            ("provider timeout", self.provider_wave_hard_timeout_ns),
            ("compute cap", self.maximum_compute_gpu_ns_per_attempt),
            ("provider cap", self.maximum_provider_reserved_gpu_ns_per_attempt),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"formal dynamic work item {label} is invalid")
        if (
            self.process_hard_timeout_ns > self.provider_wave_hard_timeout_ns
            or self.maximum_compute_gpu_ns_per_attempt
            != self.process_hard_timeout_ns * expected_gpus
            or self.maximum_provider_reserved_gpu_ns_per_attempt
            != self.provider_wave_hard_timeout_ns * self.provider_reserved_gpu_count
        ):
            raise ValueError("formal dynamic work item cap arithmetic differs")

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "run_plan": self.run_plan.to_dict(),
            "gpu_uuids": list(self.gpu_uuids),
            "rank_groups": [list(group) for group in self.rank_groups],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal dynamic dispatch work item",
            value,
            set(cls.__dataclass_fields__),
        )
        for field in ("gpu_uuids", "rank_groups"):
            raw = row[field]
            if type(raw) is not list:
                raise TypeError(f"formal dynamic work item {field} is not an array")
        row["gpu_uuids"] = tuple(row["gpu_uuids"])
        row["rank_groups"] = tuple(tuple(group) for group in row["rank_groups"])
        row["run_plan"] = CanonicalJsonProofBinding.from_dict(row["run_plan"])
        return cls(**row)


@dataclass(frozen=True)
class FormalDynamicDispatchSchedule:
    schema_version: Literal[1]
    kind: Literal["lightcone_formal_dynamic_dispatch_schedule"]
    protocol_sha256: str
    stage: str
    protocol_lock_sha256: str
    runtime_authority_manifest_sha256: str
    registry_sha256: str
    materialization: CanonicalJsonProofBinding
    materialization_receipt_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    stage_gpu_hour_receipt: CanonicalJsonProofBinding
    stage_gpu_hour_receipt_sha256: str
    launch_cap_schedule: CanonicalJsonProofBinding
    launch_cap_schedule_sha256: str
    activated_cell_ids: tuple[str, ...]
    work_items: tuple[FormalDynamicDispatchWorkItem, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_dynamic_dispatch_schedule"
            or self.protocol_sha256 != FORMAL_DYNAMIC_DISPATCH_PROTOCOL_SHA256
            or self.stage not in FORMAL_STAGE_DAG
        ):
            raise ValueError("formal dynamic dispatch schema is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime", self.runtime_authority_manifest_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware", self.hardware_envelope_sha256),
            ("GPU-hour receipt", self.stage_gpu_hour_receipt_sha256),
            ("launch cap", self.launch_cap_schedule_sha256),
        ):
            _sha256(f"formal dynamic dispatch {label}", digest)
        if any(
            type(value) is not CanonicalJsonProofBinding
            for value in (
                self.materialization,
                self.stage_gpu_hour_receipt,
                self.launch_cap_schedule,
            )
        ):
            raise TypeError("formal dynamic dispatch sources are not path-bound")
        work_ids = tuple(row.materialized_cell_id for row in self.work_items)
        if (
            not self.work_items
            or work_ids != tuple(sorted(set(work_ids)))
            or self.activated_cell_ids != work_ids
        ):
            raise ValueError("formal dynamic dispatch cell coverage differs")
        groups: dict[str, list[FormalDynamicDispatchWorkItem]] = {}
        for row in self.work_items:
            groups.setdefault(row.wave_group_sha256, []).append(row)
        for rows in groups.values():
            ordered = tuple(sorted(rows, key=lambda row: row.materialized_cell_id))
            if len(ordered) not in {1, 2}:
                raise ValueError("formal dynamic dispatch wave cardinality differs")
            if len(ordered) == 2 and (
                any(row.topology_mode != "tp1_dp1" for row in ordered)
                or len({row.process_hard_timeout_ns for row in ordered}) != 1
                or len({row.provider_wave_hard_timeout_ns for row in ordered}) != 1
                or {row.localhost_port for row in ordered}
                != {
                    FORMAL_DYNAMIC_DISPATCH_PORT_BASE,
                    FORMAL_DYNAMIC_DISPATCH_PORT_BASE + 1,
                }
                or len({row.private_output_root for row in ordered}) != 2
                or sum(row.provider_reserved_gpu_count for row in ordered) != 2
            ):
                raise ValueError("formal dynamic paired TP1 wave differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "materialization": self.materialization.to_dict(),
            "stage_gpu_hour_receipt": self.stage_gpu_hour_receipt.to_dict(),
            "launch_cap_schedule": self.launch_cap_schedule.to_dict(),
            "activated_cell_ids": list(self.activated_cell_ids),
            "work_items": [row.to_dict() for row in self.work_items],
            "schedule_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal dynamic dispatch schedule",
            value,
            {*cls.__dataclass_fields__, "schedule_sha256"},
        )
        declared = _sha256(
            "formal dynamic dispatch schedule", row.pop("schedule_sha256")
        )
        raw_cells = row.pop("activated_cell_ids")
        raw_items = row.pop("work_items")
        if type(raw_cells) is not list or type(raw_items) is not list:
            raise TypeError("formal dynamic dispatch collections are not arrays")
        for field in (
            "materialization",
            "stage_gpu_hour_receipt",
            "launch_cap_schedule",
        ):
            row[field] = CanonicalJsonProofBinding.from_dict(row[field])
        schedule = cls(
            **row,
            activated_cell_ids=tuple(raw_cells),
            work_items=tuple(
                FormalDynamicDispatchWorkItem.from_dict(item) for item in raw_items
            ),
        )
        if schedule.sha256 != declared:
            raise ValueError("formal dynamic dispatch digest differs")
        return schedule

    def work_item(self, materialized_cell_id: str) -> FormalDynamicDispatchWorkItem:
        rows = tuple(
            row
            for row in self.work_items
            if row.materialized_cell_id == materialized_cell_id
        )
        if len(rows) != 1:
            raise ValueError("formal dynamic dispatch lacks exact work item")
        return rows[0]


@dataclass(frozen=True)
class FormalPilotDynamicDispatchSchedule:
    """Source-owned physical placement for the exact minimum pilot set.

    This is deliberately not a ``FormalLaunchCapSchedule``.  A blocked
    prospective source has no full-stage estimate yet; its only launch
    authority is the already reserved ``FormalPilotLaunchBudget``.  Every
    pilot is therefore an independent two-GPU-provider wave even when the
    process itself is TP1.
    """

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_pilot_dynamic_dispatch_schedule"]
    protocol_sha256: str
    stage: str
    protocol_lock_sha256: str
    runtime_authority_manifest_sha256: str
    registry_sha256: str
    materialization: CanonicalJsonProofBinding
    materialization_receipt_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    pilot_budget_verification_receipt: CanonicalJsonProofBinding
    pilot_budget_verification_receipt_sha256: str
    pilot_launch_budget: CanonicalJsonProofBinding
    pilot_launch_budget_sha256: str
    launch_nonce_sha256: str
    activated_cell_ids: tuple[str, ...]
    work_items: tuple[FormalDynamicDispatchWorkItem, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_pilot_dynamic_dispatch_schedule"
            or self.protocol_sha256 != FORMAL_PILOT_DYNAMIC_DISPATCH_PROTOCOL_SHA256
            or self.stage not in FORMAL_STAGE_DAG
        ):
            raise ValueError("formal pilot dynamic dispatch schema differs")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime", self.runtime_authority_manifest_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware", self.hardware_envelope_sha256),
            (
                "budget verification",
                self.pilot_budget_verification_receipt_sha256,
            ),
            ("pilot budget", self.pilot_launch_budget_sha256),
            ("launch nonce", self.launch_nonce_sha256),
        ):
            _sha256(f"formal pilot dynamic dispatch {label}", digest)
        if any(
            type(value) is not CanonicalJsonProofBinding
            for value in (
                self.materialization,
                self.pilot_budget_verification_receipt,
                self.pilot_launch_budget,
            )
        ):
            raise TypeError("formal pilot dispatch sources are not path-bound")
        work_ids = tuple(row.materialized_cell_id for row in self.work_items)
        if (
            not self.work_items
            or work_ids != tuple(sorted(set(work_ids)))
            or self.activated_cell_ids != work_ids
            or any(
                row.wave_group_sha256
                != _pilot_wave_group_sha256(
                    verification_receipt_sha256=(
                        self.pilot_budget_verification_receipt_sha256
                    ),
                    launch_nonce_sha256=self.launch_nonce_sha256,
                    materialized_cell_id=row.materialized_cell_id,
                    canonical_index=index,
                )
                for index, row in enumerate(self.work_items)
            )
            or any(row.provider_reserved_gpu_count != 2 for row in self.work_items)
            or len({row.wave_group_sha256 for row in self.work_items})
            != len(self.work_items)
        ):
            raise ValueError("formal pilot dispatch coverage/waves differ")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "materialization": self.materialization.to_dict(),
            "pilot_budget_verification_receipt": (
                self.pilot_budget_verification_receipt.to_dict()
            ),
            "pilot_launch_budget": self.pilot_launch_budget.to_dict(),
            "activated_cell_ids": list(self.activated_cell_ids),
            "work_items": [row.to_dict() for row in self.work_items],
            "schedule_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal pilot dynamic dispatch schedule",
            value,
            {*cls.__dataclass_fields__, "schedule_sha256"},
        )
        declared = _sha256(
            "formal pilot dynamic dispatch schedule", row.pop("schedule_sha256")
        )
        raw_cells = row.pop("activated_cell_ids")
        raw_items = row.pop("work_items")
        if type(raw_cells) is not list or type(raw_items) is not list:
            raise TypeError("formal pilot dispatch collections are not arrays")
        for field in (
            "materialization",
            "pilot_budget_verification_receipt",
            "pilot_launch_budget",
        ):
            row[field] = CanonicalJsonProofBinding.from_dict(row[field])
        schedule = cls(
            **row,
            activated_cell_ids=tuple(raw_cells),
            work_items=tuple(
                FormalDynamicDispatchWorkItem.from_dict(item) for item in raw_items
            ),
        )
        if schedule.sha256 != declared:
            raise ValueError("formal pilot dynamic dispatch digest differs")
        return schedule

    def work_item(self, materialized_cell_id: str) -> FormalDynamicDispatchWorkItem:
        rows = tuple(
            row
            for row in self.work_items
            if row.materialized_cell_id == materialized_cell_id
        )
        if len(rows) != 1:
            raise ValueError("formal pilot dispatch lacks exact work item")
        return rows[0]


def _pilot_wave_group_sha256(
    *,
    verification_receipt_sha256: str,
    launch_nonce_sha256: str,
    materialized_cell_id: str,
    canonical_index: int,
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_pilot_singleton_wave",
            "pilot_budget_verification_receipt_sha256": (verification_receipt_sha256),
            "launch_nonce_sha256": launch_nonce_sha256,
            "materialized_cell_id": materialized_cell_id,
            "canonical_index": canonical_index,
        }
    )


def _canonical_inventory_gpu_uuids_from_inventory(inventory: object) -> tuple[str, str]:
    devices = tuple(sorted(inventory.devices, key=lambda row: row.uuid))
    if len(devices) != 2 or any(not row.ready for row in devices):
        raise ValueError(
            "formal dynamic dispatch requires exact ready two-GPU inventory"
        )
    if len({row.hardware_envelope_sha256 for row in devices}) != 1:
        raise ValueError("formal dynamic dispatch inventory is not homogeneous")
    return devices[0].uuid, devices[1].uuid


def _canonical_inventory_gpu_uuids(
    receipt: FormalStageGpuHourVerificationReceipt,
) -> tuple[str, str]:
    return _canonical_inventory_gpu_uuids_from_inventory(receipt.inventory)


def _expected_placement(
    *,
    cell_id: str,
    wave_cells: tuple[str, ...],
    topology_mode: str,
    inventory_gpus: tuple[str, str],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], int]:
    if topology_mode == "tp1_dp1":
        index = wave_cells.index(cell_id)
        gpus = (inventory_gpus[index],)
        return gpus, (gpus,), FORMAL_DYNAMIC_DISPATCH_PORT_BASE + index
    if wave_cells != (cell_id,):
        raise ValueError("formal dynamic distributed work must be a singleton wave")
    if topology_mode == "tp2_dp1":
        return inventory_gpus, (inventory_gpus,), FORMAL_DYNAMIC_DISPATCH_PORT_BASE
    if topology_mode == "tp1_dp2":
        return (
            inventory_gpus,
            ((inventory_gpus[0],), (inventory_gpus[1],)),
            FORMAL_DYNAMIC_DISPATCH_PORT_BASE,
        )
    raise ValueError("formal dynamic topology is unsupported")


def _work_item_from_plan(
    *,
    plan: FormalServingRunPlan,
    plan_binding: CanonicalJsonProofBinding,
    cap_schedule: FormalLaunchCapSchedule,
    inventory_gpus: tuple[str, str],
) -> FormalDynamicDispatchWorkItem:
    cap = cap_schedule.cap_for(plan.materialized_cell_id)
    if cap.disposition != "LAUNCHABLE":
        raise ValueError("formal dynamic dispatch cannot schedule preconsumed work")
    wave_cells = tuple(
        row.materialized_cell_id
        for row in cap_schedule.cell_caps
        if row.wave_group_sha256 == cap.wave_group_sha256
        and row.disposition == "LAUNCHABLE"
    )
    expected_gpus, rank_groups, expected_port = _expected_placement(
        cell_id=plan.materialized_cell_id,
        wave_cells=wave_cells,
        topology_mode=plan.topology_mode,
        inventory_gpus=inventory_gpus,
    )
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    if (
        plan.gpu_uuids != expected_gpus
        or launch.gpu_uuids != expected_gpus
        or launch.localhost_port != expected_port
        or cap.gpu_count != len(expected_gpus)
    ):
        raise ValueError("formal dynamic plan GPU/port differs from source placement")
    return FormalDynamicDispatchWorkItem(
        materialized_cell_id=plan.materialized_cell_id,
        execution_binding_sha256=plan.execution_binding_sha256,
        execution_plan_sha256=plan.native_terminal_binding.execution_plan_sha256,
        run_plan=plan_binding,
        run_plan_sha256=plan.sha256,
        topology_mode=plan.topology_mode,
        gpu_uuids=plan.gpu_uuids,
        rank_groups=rank_groups,
        localhost_port=launch.localhost_port,
        private_output_root=plan.private_output_root,
        wave_group_sha256=cap.wave_group_sha256,
        process_hard_timeout_ns=cap.process_hard_timeout_ns_per_attempt,
        provider_wave_hard_timeout_ns=(cap.provider_wave_hard_timeout_ns_per_attempt),
        provider_reserved_gpu_count=cap.provider_reserved_gpu_count,
        maximum_compute_gpu_ns_per_attempt=(cap.maximum_compute_gpu_ns_per_attempt),
        maximum_provider_reserved_gpu_ns_per_attempt=(
            cap.maximum_provider_reserved_gpu_ns_per_attempt
        ),
        allowed_attempts=cap.allowed_attempts,
    )


def revalidate_formal_dynamic_dispatch_schedule(
    schedule: FormalDynamicDispatchSchedule,
    *,
    current_ns: int,
) -> FormalDynamicDispatchSchedule:
    """Deep-open every durable source without requiring private capability tokens."""

    if type(schedule) is not FormalDynamicDispatchSchedule:
        raise TypeError("formal dynamic dispatch revalidator requires exact schedule")
    materialization = _materialization(schedule.materialization)
    receipt, rebuilt_caps = _load_stage_budget(
        schedule.stage_gpu_hour_receipt,
        materialization=materialization,
        current_ns=current_ns,
    )
    if (
        CanonicalJsonProofBinding.bind(schedule.launch_cap_schedule.absolute_path)
        != schedule.launch_cap_schedule
    ):
        raise ValueError("formal dynamic launch cap binding changed")
    persisted_caps = FormalLaunchCapSchedule.from_dict(
        schedule.launch_cap_schedule.reopen()
    )
    lock = receipt.registry_receipt.signed_protocol_lock.payload
    if (
        persisted_caps != rebuilt_caps
        or persisted_caps.sha256 != schedule.launch_cap_schedule_sha256
        or schedule.stage != materialization.stage
        or schedule.protocol_lock_sha256 != lock.sha256
        or schedule.runtime_authority_manifest_sha256
        != receipt.formal_runtime_authority_manifest.sha256
        or schedule.registry_sha256 != lock.registry_sha256
        or schedule.materialization_receipt_sha256 != materialization.sha256
        or schedule.inventory_sha256 != receipt.inventory.sha256
        or schedule.hardware_envelope_sha256 != rebuilt_caps.hardware_envelope_sha256
        or schedule.stage_gpu_hour_receipt_sha256 != receipt.sha256
        or schedule.activated_cell_ids != rebuilt_caps.launchable_cell_ids
    ):
        raise ValueError("formal dynamic dispatch source lineage differs")
    inventory_gpus = _canonical_inventory_gpu_uuids(receipt)
    rebuilt_items = []
    for row in schedule.work_items:
        if CanonicalJsonProofBinding.bind(row.run_plan.absolute_path) != row.run_plan:
            raise ValueError("formal dynamic run plan binding changed")
        plan = FormalServingRunPlan.from_dict(row.run_plan.reopen())
        if plan.sha256 != row.run_plan.semantic_sha256:
            raise ValueError("formal dynamic run plan identity changed")
        rebuilt_items.append(
            _work_item_from_plan(
                plan=plan,
                plan_binding=row.run_plan,
                cap_schedule=rebuilt_caps,
                inventory_gpus=inventory_gpus,
            )
        )
    rebuilt = FormalDynamicDispatchSchedule(
        schema_version=1,
        kind="lightcone_formal_dynamic_dispatch_schedule",
        protocol_sha256=FORMAL_DYNAMIC_DISPATCH_PROTOCOL_SHA256,
        stage=materialization.stage,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_manifest_sha256=(
            receipt.formal_runtime_authority_manifest.sha256
        ),
        registry_sha256=lock.registry_sha256,
        materialization=schedule.materialization,
        materialization_receipt_sha256=materialization.sha256,
        inventory_sha256=receipt.inventory.sha256,
        hardware_envelope_sha256=rebuilt_caps.hardware_envelope_sha256,
        stage_gpu_hour_receipt=schedule.stage_gpu_hour_receipt,
        stage_gpu_hour_receipt_sha256=receipt.sha256,
        launch_cap_schedule=schedule.launch_cap_schedule,
        launch_cap_schedule_sha256=rebuilt_caps.sha256,
        activated_cell_ids=rebuilt_caps.launchable_cell_ids,
        work_items=tuple(
            sorted(rebuilt_items, key=lambda item: item.materialized_cell_id)
        ),
    )
    if rebuilt != schedule:
        raise ValueError("formal dynamic dispatch differs from deterministic rebuild")
    return schedule


@dataclass(frozen=True)
class FormalDynamicStageCapacitySchedule:
    """One cell projection of a complete dynamic dispatch/capacity schedule."""

    schema_version: Literal[2]
    kind: Literal["lightcone_formal_dynamic_stage_capacity_schedule"]
    protocol_sha256: str
    stage: str
    protocol_lock_sha256: str
    runtime_authority_manifest_sha256: str
    registry_sha256: str
    materialization_receipt_sha256: str
    dynamic_dispatch_schedule: CanonicalJsonProofBinding
    dynamic_dispatch_schedule_sha256: str
    activated_cell_ids: tuple[str, ...]
    materialized_cell_id: str
    execution_binding_sha256: str
    execution_plan_sha256: str
    run_plan: CanonicalJsonProofBinding
    run_plan_sha256: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    rank_groups: tuple[tuple[str, ...], ...]
    localhost_port: int
    wave_index: int
    wave_group_sha256: str
    wave_cell_ids: tuple[str, ...]
    provider_inventory_gpu_count: Literal[2]
    provider_reserved_gpu_count: Literal[1, 2]
    retry_allowance: Literal[1]
    process_hard_timeout_ns: int
    provider_wave_hard_timeout_ns: int
    maximum_compute_gpu_ns_per_attempt: int
    maximum_provider_reserved_gpu_ns_per_attempt: int
    capacity_source_manifest: CanonicalJsonProofBinding
    capacity_envelope_sha256: str
    budget_inventory_sha256: str
    capacity_captured_at_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.kind != "lightcone_formal_dynamic_stage_capacity_schedule"
            or self.protocol_sha256 != FORMAL_DYNAMIC_STAGE_CAPACITY_PROTOCOL_SHA256
            or self.stage not in FORMAL_STAGE_DAG
        ):
            raise ValueError("formal dynamic capacity schedule schema differs")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime", self.runtime_authority_manifest_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("dynamic dispatch", self.dynamic_dispatch_schedule_sha256),
            ("cell", self.materialized_cell_id),
            ("execution binding", self.execution_binding_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("run plan", self.run_plan_sha256),
            ("inventory", self.inventory_sha256),
            ("wave", self.wave_group_sha256),
            ("capacity envelope", self.capacity_envelope_sha256),
            ("budget inventory", self.budget_inventory_sha256),
        ):
            _sha256(f"formal dynamic capacity {label}", digest)
        if any(
            type(value) is not CanonicalJsonProofBinding
            for value in (
                self.dynamic_dispatch_schedule,
                self.run_plan,
                self.capacity_source_manifest,
            )
        ):
            raise TypeError("formal dynamic capacity sources are not path-bound")
        if (
            self.activated_cell_ids != tuple(sorted(set(self.activated_cell_ids)))
            or self.materialized_cell_id not in self.activated_cell_ids
            or self.wave_cell_ids != tuple(sorted(set(self.wave_cell_ids)))
            or self.materialized_cell_id not in self.wave_cell_ids
            or self.provider_inventory_gpu_count != 2
            or self.retry_allowance != 1
            or type(self.wave_index) is not int
            or self.wave_index < 0
            or type(self.capacity_captured_at_ns) is not int
            or self.capacity_captured_at_ns < 0
        ):
            raise ValueError("formal dynamic capacity wave/cell coverage differs")
        work = FormalDynamicDispatchWorkItem(
            materialized_cell_id=self.materialized_cell_id,
            execution_binding_sha256=self.execution_binding_sha256,
            execution_plan_sha256=self.execution_plan_sha256,
            run_plan=self.run_plan,
            run_plan_sha256=self.run_plan_sha256,
            topology_mode=self.topology_mode,
            gpu_uuids=self.gpu_uuids,
            rank_groups=self.rank_groups,
            localhost_port=self.localhost_port,
            private_output_root=str(Path(self.run_plan.absolute_path).parent),
            wave_group_sha256=self.wave_group_sha256,
            process_hard_timeout_ns=self.process_hard_timeout_ns,
            provider_wave_hard_timeout_ns=self.provider_wave_hard_timeout_ns,
            provider_reserved_gpu_count=self.provider_reserved_gpu_count,
            maximum_compute_gpu_ns_per_attempt=(
                self.maximum_compute_gpu_ns_per_attempt
            ),
            maximum_provider_reserved_gpu_ns_per_attempt=(
                self.maximum_provider_reserved_gpu_ns_per_attempt
            ),
            allowed_attempts=2,
        )
        work.__post_init__()

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    @property
    def _dynamic_dispatch(
        self,
    ) -> FormalDynamicDispatchSchedule | FormalPilotDynamicDispatchSchedule:
        value = self.dynamic_dispatch_schedule.reopen()
        if type(value) is not dict:
            raise TypeError("formal dynamic dispatch source is not an object")
        if value.get("kind") == "lightcone_formal_dynamic_dispatch_schedule":
            return FormalDynamicDispatchSchedule.from_dict(value)
        if value.get("kind") == "lightcone_formal_pilot_dynamic_dispatch_schedule":
            return FormalPilotDynamicDispatchSchedule.from_dict(value)
        raise ValueError("formal dynamic dispatch source kind differs")

    @property
    def stage_gpu_hour_receipt(self) -> CanonicalJsonProofBinding:
        dispatch = self._dynamic_dispatch
        if type(dispatch) is not FormalDynamicDispatchSchedule:
            raise ValueError("pilot dispatch has no full-stage GPU-hour receipt")
        return dispatch.stage_gpu_hour_receipt

    @property
    def launch_cap_schedule_sha256(self) -> str:
        dispatch = self._dynamic_dispatch
        if type(dispatch) is not FormalDynamicDispatchSchedule:
            raise ValueError("pilot dispatch has no full-stage launch cap schedule")
        return dispatch.launch_cap_schedule_sha256

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "dynamic_dispatch_schedule": self.dynamic_dispatch_schedule.to_dict(),
            "activated_cell_ids": list(self.activated_cell_ids),
            "run_plan": self.run_plan.to_dict(),
            "gpu_uuids": list(self.gpu_uuids),
            "rank_groups": [list(group) for group in self.rank_groups],
            "wave_cell_ids": list(self.wave_cell_ids),
            "capacity_source_manifest": self.capacity_source_manifest.to_dict(),
            "schedule_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal dynamic stage capacity schedule",
            value,
            {*cls.__dataclass_fields__, "schedule_sha256"},
        )
        declared = _sha256(
            "formal dynamic stage capacity schedule", row.pop("schedule_sha256")
        )
        for field in (
            "activated_cell_ids",
            "gpu_uuids",
            "rank_groups",
            "wave_cell_ids",
        ):
            if type(row[field]) is not list:
                raise TypeError(f"formal dynamic capacity {field} is not an array")
        row["activated_cell_ids"] = tuple(row["activated_cell_ids"])
        row["gpu_uuids"] = tuple(row["gpu_uuids"])
        row["rank_groups"] = tuple(tuple(group) for group in row["rank_groups"])
        row["wave_cell_ids"] = tuple(row["wave_cell_ids"])
        for field in (
            "dynamic_dispatch_schedule",
            "run_plan",
            "capacity_source_manifest",
        ):
            row[field] = CanonicalJsonProofBinding.from_dict(row[field])
        schedule = cls(**row)
        if schedule.sha256 != declared:
            raise ValueError("formal dynamic stage capacity digest differs")
        return schedule


def _replay_dynamic_capacity(
    schedule: FormalDynamicStageCapacitySchedule,
    *,
    current_ns: int,
) -> UnsignedCapacitySourceReplay:
    replay = replay_unsigned_capacity_source_manifest(
        schedule.capacity_source_manifest.absolute_path,
        expected_registry_sha256=schedule.registry_sha256,
        now_ns=current_ns,
    )
    requirement_ids = tuple(
        row.cell_id for row in replay.capacity_envelope.cell_requirements
    )
    if (
        replay.capacity_envelope.sha256 != schedule.capacity_envelope_sha256
        or replay.budget_inventory.sha256 != schedule.budget_inventory_sha256
        or replay.gpu_inventory.sha256 != schedule.inventory_sha256
        or replay.captured_at_ns != schedule.capacity_captured_at_ns
        or requirement_ids != schedule.activated_cell_ids
    ):
        raise ValueError("formal dynamic capacity raw source differs")
    return replay


def revalidate_formal_dynamic_stage_capacity_schedule(
    schedule: FormalDynamicStageCapacitySchedule,
    *,
    current_ns: int,
) -> UnsignedCapacitySourceReplay:
    if (
        CanonicalJsonProofBinding.bind(schedule.dynamic_dispatch_schedule.absolute_path)
        != schedule.dynamic_dispatch_schedule
    ):
        raise ValueError("formal dynamic dispatch schedule binding changed")
    dispatch = schedule._dynamic_dispatch
    if type(dispatch) is FormalDynamicDispatchSchedule:
        revalidate_formal_dynamic_dispatch_schedule(dispatch, current_ns=current_ns)
    else:
        revalidate_formal_pilot_dynamic_dispatch_schedule(
            dispatch, current_ns=current_ns
        )
    work = dispatch.work_item(schedule.materialized_cell_id)
    groups = tuple(sorted({row.wave_group_sha256 for row in dispatch.work_items}))
    wave_cells = tuple(
        row.materialized_cell_id
        for row in dispatch.work_items
        if row.wave_group_sha256 == work.wave_group_sha256
    )
    rebuilt = FormalDynamicStageCapacitySchedule(
        schema_version=2,
        kind="lightcone_formal_dynamic_stage_capacity_schedule",
        protocol_sha256=FORMAL_DYNAMIC_STAGE_CAPACITY_PROTOCOL_SHA256,
        stage=dispatch.stage,
        protocol_lock_sha256=dispatch.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(dispatch.runtime_authority_manifest_sha256),
        registry_sha256=dispatch.registry_sha256,
        materialization_receipt_sha256=(dispatch.materialization_receipt_sha256),
        dynamic_dispatch_schedule=schedule.dynamic_dispatch_schedule,
        dynamic_dispatch_schedule_sha256=dispatch.sha256,
        activated_cell_ids=dispatch.activated_cell_ids,
        materialized_cell_id=work.materialized_cell_id,
        execution_binding_sha256=work.execution_binding_sha256,
        execution_plan_sha256=work.execution_plan_sha256,
        run_plan=work.run_plan,
        run_plan_sha256=work.run_plan_sha256,
        topology_mode=work.topology_mode,
        inventory_sha256=dispatch.inventory_sha256,
        gpu_uuids=work.gpu_uuids,
        rank_groups=work.rank_groups,
        localhost_port=work.localhost_port,
        wave_index=groups.index(work.wave_group_sha256),
        wave_group_sha256=work.wave_group_sha256,
        wave_cell_ids=wave_cells,
        provider_inventory_gpu_count=2,
        provider_reserved_gpu_count=work.provider_reserved_gpu_count,
        retry_allowance=1,
        process_hard_timeout_ns=work.process_hard_timeout_ns,
        provider_wave_hard_timeout_ns=work.provider_wave_hard_timeout_ns,
        maximum_compute_gpu_ns_per_attempt=(work.maximum_compute_gpu_ns_per_attempt),
        maximum_provider_reserved_gpu_ns_per_attempt=(
            work.maximum_provider_reserved_gpu_ns_per_attempt
        ),
        capacity_source_manifest=schedule.capacity_source_manifest,
        capacity_envelope_sha256=schedule.capacity_envelope_sha256,
        budget_inventory_sha256=schedule.budget_inventory_sha256,
        capacity_captured_at_ns=schedule.capacity_captured_at_ns,
    )
    if rebuilt != schedule:
        raise ValueError("formal dynamic capacity differs from deterministic rebuild")
    return _replay_dynamic_capacity(schedule, current_ns=current_ns)


def materialize_formal_dynamic_stage_capacity_schedule(
    *,
    dynamic_dispatch_schedule_path: str | Path,
    materialized_cell_id: str,
    capacity_source_manifest_path: str | Path,
    output_path: str | Path,
    current_ns: int,
) -> FormalDynamicStageCapacitySchedule:
    """Project one immutable dynamic wave into the capacity admission input."""

    dispatch_binding = CanonicalJsonProofBinding.bind(dynamic_dispatch_schedule_path)
    value = dispatch_binding.reopen()
    if type(value) is not dict:
        raise TypeError("formal dynamic dispatch artifact is not an object")
    if value.get("kind") == "lightcone_formal_dynamic_dispatch_schedule":
        dispatch: FormalDynamicDispatchSchedule | FormalPilotDynamicDispatchSchedule
        dispatch = FormalDynamicDispatchSchedule.from_dict(value)
        revalidate_formal_dynamic_dispatch_schedule(dispatch, current_ns=current_ns)
    elif value.get("kind") == "lightcone_formal_pilot_dynamic_dispatch_schedule":
        dispatch = FormalPilotDynamicDispatchSchedule.from_dict(value)
        revalidate_formal_pilot_dynamic_dispatch_schedule(
            dispatch, current_ns=current_ns
        )
    else:
        raise ValueError("formal dynamic dispatch artifact kind differs")
    work = dispatch.work_item(materialized_cell_id)
    groups = tuple(sorted({row.wave_group_sha256 for row in dispatch.work_items}))
    wave_cells = tuple(
        row.materialized_cell_id
        for row in dispatch.work_items
        if row.wave_group_sha256 == work.wave_group_sha256
    )
    source_binding = CanonicalJsonProofBinding.bind(capacity_source_manifest_path)
    replay = replay_unsigned_capacity_source_manifest(
        source_binding.absolute_path,
        expected_registry_sha256=dispatch.registry_sha256,
        now_ns=current_ns,
    )
    schedule = FormalDynamicStageCapacitySchedule(
        schema_version=2,
        kind="lightcone_formal_dynamic_stage_capacity_schedule",
        protocol_sha256=FORMAL_DYNAMIC_STAGE_CAPACITY_PROTOCOL_SHA256,
        stage=dispatch.stage,
        protocol_lock_sha256=dispatch.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(dispatch.runtime_authority_manifest_sha256),
        registry_sha256=dispatch.registry_sha256,
        materialization_receipt_sha256=dispatch.materialization_receipt_sha256,
        dynamic_dispatch_schedule=dispatch_binding,
        dynamic_dispatch_schedule_sha256=dispatch.sha256,
        activated_cell_ids=dispatch.activated_cell_ids,
        materialized_cell_id=work.materialized_cell_id,
        execution_binding_sha256=work.execution_binding_sha256,
        execution_plan_sha256=work.execution_plan_sha256,
        run_plan=work.run_plan,
        run_plan_sha256=work.run_plan_sha256,
        topology_mode=work.topology_mode,
        inventory_sha256=dispatch.inventory_sha256,
        gpu_uuids=work.gpu_uuids,
        rank_groups=work.rank_groups,
        localhost_port=work.localhost_port,
        wave_index=groups.index(work.wave_group_sha256),
        wave_group_sha256=work.wave_group_sha256,
        wave_cell_ids=wave_cells,
        provider_inventory_gpu_count=2,
        provider_reserved_gpu_count=work.provider_reserved_gpu_count,
        retry_allowance=1,
        process_hard_timeout_ns=work.process_hard_timeout_ns,
        provider_wave_hard_timeout_ns=work.provider_wave_hard_timeout_ns,
        maximum_compute_gpu_ns_per_attempt=(work.maximum_compute_gpu_ns_per_attempt),
        maximum_provider_reserved_gpu_ns_per_attempt=(
            work.maximum_provider_reserved_gpu_ns_per_attempt
        ),
        capacity_source_manifest=source_binding,
        capacity_envelope_sha256=replay.capacity_envelope.sha256,
        budget_inventory_sha256=replay.budget_inventory.sha256,
        capacity_captured_at_ns=replay.captured_at_ns,
    )
    _replay_dynamic_capacity(schedule, current_ns=current_ns)
    publish_canonical_json_no_replace(output_path, schedule.to_dict())
    reopened = FormalDynamicStageCapacitySchedule.from_dict(
        CanonicalJsonProofBinding.bind(output_path).reopen()
    )
    if reopened != schedule:
        raise RuntimeError("formal dynamic capacity changed during publication")
    return schedule


def materialize_formal_dynamic_dispatch_schedule(
    *,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    run_plan_paths: tuple[str | Path, ...],
    materialization_path: str | Path,
    stage_gpu_hour_receipt_path: str | Path,
    launch_cap_schedule_path: str | Path,
    output_path: str | Path,
    current_ns: int,
) -> FormalDynamicDispatchSchedule:
    """Build all launchable work items; GPU, rank and port are not arguments."""

    if (
        type(execution_bindings) is not tuple
        or type(run_plan_paths) is not tuple
        or len(execution_bindings) != len(run_plan_paths)
        or not execution_bindings
    ):
        raise ValueError("formal dynamic dispatch execution inputs are not exact")
    materialization_binding = CanonicalJsonProofBinding.bind(materialization_path)
    materialization = _materialization(materialization_binding)
    budget_binding = CanonicalJsonProofBinding.bind(stage_gpu_hour_receipt_path)
    receipt, cap_schedule = _load_stage_budget(
        budget_binding,
        materialization=materialization,
        current_ns=current_ns,
    )
    cap_binding = CanonicalJsonProofBinding.bind(launch_cap_schedule_path)
    persisted_caps = FormalLaunchCapSchedule.from_dict(cap_binding.reopen())
    if (
        persisted_caps != cap_schedule
        or cap_binding.semantic_sha256 != cap_schedule.sha256
    ):
        raise ValueError("formal dynamic dispatch launch cap artifact differs")
    inventory_gpus = _canonical_inventory_gpu_uuids(receipt)
    items = []
    for token, plan_path in zip(execution_bindings, run_plan_paths, strict=True):
        verified = require_verified_formal_serving_execution_binding(token)
        plan = load_formal_serving_run_plan(
            plan_path,
            execution_binding=verified,
            verified_nextn_tp2_authority=verified.verified_nextn_tp2_authority,
        )
        binding = CanonicalJsonProofBinding.bind(plan_path, semantic_sha256=plan.sha256)
        items.append(
            _work_item_from_plan(
                plan=plan,
                plan_binding=binding,
                cap_schedule=cap_schedule,
                inventory_gpus=inventory_gpus,
            )
        )
    lock = receipt.registry_receipt.signed_protocol_lock.payload
    schedule = FormalDynamicDispatchSchedule(
        schema_version=1,
        kind="lightcone_formal_dynamic_dispatch_schedule",
        protocol_sha256=FORMAL_DYNAMIC_DISPATCH_PROTOCOL_SHA256,
        stage=materialization.stage,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_manifest_sha256=(
            receipt.formal_runtime_authority_manifest.sha256
        ),
        registry_sha256=lock.registry_sha256,
        materialization=materialization_binding,
        materialization_receipt_sha256=materialization.sha256,
        inventory_sha256=receipt.inventory.sha256,
        hardware_envelope_sha256=cap_schedule.hardware_envelope_sha256,
        stage_gpu_hour_receipt=budget_binding,
        stage_gpu_hour_receipt_sha256=receipt.sha256,
        launch_cap_schedule=cap_binding,
        launch_cap_schedule_sha256=cap_schedule.sha256,
        activated_cell_ids=cap_schedule.launchable_cell_ids,
        work_items=tuple(sorted(items, key=lambda item: item.materialized_cell_id)),
    )
    if tuple(row.materialized_cell_id for row in schedule.work_items) != (
        schedule.activated_cell_ids
    ):
        raise ValueError("formal dynamic dispatch does not cover every launchable cell")
    publish_canonical_json_no_replace(output_path, schedule.to_dict())
    reopened = FormalDynamicDispatchSchedule.from_dict(
        CanonicalJsonProofBinding.bind(output_path).reopen()
    )
    if reopened != schedule:
        raise RuntimeError("formal dynamic dispatch changed during publication")
    revalidate_formal_dynamic_dispatch_schedule(schedule, current_ns=current_ns)
    return schedule


def _load_pilot_dispatch_budget(
    binding: CanonicalJsonProofBinding,
    *,
    current_ns: int,
) -> tuple[
    object, object, FormalStageGpuHourVerificationReceipt, StageMaterializationReceipt
]:
    # Imported lazily because formal_launch_admission imports the schedule
    # codecs from this module.  The durable revalidator remains callback-free.
    from lightcone_spec.orchestration.formal_launch_admission import (
        revalidate_formal_pilot_budget_verification_receipt,
    )

    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("formal pilot dispatch verification receipt changed")
    verification, budget, preflight = (
        revalidate_formal_pilot_budget_verification_receipt(
            binding,
            current_ns=current_ns,
        )
    )
    if verification.sha256 != binding.semantic_sha256:
        raise ValueError("formal pilot dispatch verification identity differs")
    materialization = _materialization(budget.materialization)
    if (
        CanonicalJsonProofBinding.bind(verification.pilot_launch_budget.absolute_path)
        != verification.pilot_launch_budget
        or budget.materialization_receipt_sha256 != materialization.sha256
        or budget.minimum_pilot_cell_ids
        != tuple(sorted(set(budget.minimum_pilot_cell_ids)))
    ):
        raise ValueError("formal pilot dispatch budget lineage differs")
    return verification, budget, preflight, materialization


def _pilot_work_item_from_plan(
    *,
    plan: FormalServingRunPlan,
    plan_binding: CanonicalJsonProofBinding,
    budget: object,
    preflight: FormalStageGpuHourVerificationReceipt,
    verification_receipt_sha256: str,
    canonical_index: int,
) -> FormalDynamicDispatchWorkItem:
    cell_id = plan.materialized_cell_id
    if cell_id not in budget.minimum_pilot_cell_ids:
        raise ValueError("formal pilot plan is outside the minimum pilot set")
    gpu_count = dict(budget.gpu_count_by_cell)[cell_id]
    provider_count = dict(budget.provider_reserved_gpu_count_by_cell)[cell_id]
    inventory_gpus = _canonical_inventory_gpu_uuids_from_inventory(preflight.inventory)
    expected_gpus, rank_groups, expected_port = _expected_placement(
        cell_id=cell_id,
        wave_cells=(cell_id,),
        topology_mode=plan.topology_mode,
        inventory_gpus=inventory_gpus,
    )
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    if (
        plan.stage != budget.stage
        or plan.inventory_sha256 != preflight.inventory.sha256
        or plan.gpu_uuids != expected_gpus
        or launch.gpu_uuids != expected_gpus
        or launch.localhost_port != expected_port
        or gpu_count != len(expected_gpus)
        or provider_count != 2
        or budget.retry_allowance + 1 != 2
    ):
        raise ValueError("formal pilot plan placement/cap differs from budget")
    wave = _pilot_wave_group_sha256(
        verification_receipt_sha256=verification_receipt_sha256,
        launch_nonce_sha256=budget.launch_nonce_sha256,
        materialized_cell_id=cell_id,
        canonical_index=canonical_index,
    )
    return FormalDynamicDispatchWorkItem(
        materialized_cell_id=cell_id,
        execution_binding_sha256=plan.execution_binding_sha256,
        execution_plan_sha256=plan.native_terminal_binding.execution_plan_sha256,
        run_plan=plan_binding,
        run_plan_sha256=plan.sha256,
        topology_mode=plan.topology_mode,
        gpu_uuids=expected_gpus,
        rank_groups=rank_groups,
        localhost_port=expected_port,
        private_output_root=plan.private_output_root,
        wave_group_sha256=wave,
        process_hard_timeout_ns=budget.per_cell_hard_timeout_ns,
        provider_wave_hard_timeout_ns=budget.per_cell_hard_timeout_ns,
        provider_reserved_gpu_count=provider_count,
        maximum_compute_gpu_ns_per_attempt=(
            budget.per_cell_hard_timeout_ns * gpu_count
        ),
        maximum_provider_reserved_gpu_ns_per_attempt=(
            budget.per_cell_hard_timeout_ns * provider_count
        ),
        allowed_attempts=budget.retry_allowance + 1,
    )


def revalidate_formal_pilot_dynamic_dispatch_schedule(
    schedule: FormalPilotDynamicDispatchSchedule,
    *,
    current_ns: int,
) -> FormalPilotDynamicDispatchSchedule:
    if type(schedule) is not FormalPilotDynamicDispatchSchedule:
        raise TypeError("formal pilot dispatch requires its exact schedule type")
    verification, budget, preflight, materialization = _load_pilot_dispatch_budget(
        schedule.pilot_budget_verification_receipt,
        current_ns=current_ns,
    )
    lock = preflight.registry_receipt.signed_protocol_lock.payload
    devices = tuple(sorted(preflight.inventory.devices, key=lambda row: row.uuid))
    hardware = {row.hardware_envelope_sha256 for row in devices}
    if (
        len(hardware) != 1
        or schedule.stage != budget.stage
        or schedule.protocol_lock_sha256 != budget.protocol_lock_sha256
        or schedule.runtime_authority_manifest_sha256
        != budget.runtime_authority_manifest_sha256
        or schedule.registry_sha256 != lock.registry_sha256
        or schedule.materialization != budget.materialization
        or schedule.materialization_receipt_sha256 != materialization.sha256
        or schedule.inventory_sha256 != preflight.inventory.sha256
        or schedule.hardware_envelope_sha256 != next(iter(hardware))
        or schedule.pilot_budget_verification_receipt_sha256 != verification.sha256
        or schedule.pilot_launch_budget != verification.pilot_launch_budget
        or schedule.pilot_launch_budget_sha256 != budget.sha256
        or schedule.launch_nonce_sha256 != budget.launch_nonce_sha256
        or schedule.activated_cell_ids != budget.minimum_pilot_cell_ids
    ):
        raise ValueError("formal pilot dispatch immutable lineage differs")
    by_cell = {row.materialized_cell_id: row for row in schedule.work_items}
    rebuilt_items = []
    for index, cell_id in enumerate(budget.minimum_pilot_cell_ids):
        row = by_cell.get(cell_id)
        if row is None:
            raise ValueError("formal pilot dispatch lacks a minimum pilot")
        if CanonicalJsonProofBinding.bind(row.run_plan.absolute_path) != row.run_plan:
            raise ValueError("formal pilot run plan binding changed")
        plan = FormalServingRunPlan.from_dict(row.run_plan.reopen())
        if plan.sha256 != row.run_plan.semantic_sha256:
            raise ValueError("formal pilot run plan identity changed")
        rebuilt_items.append(
            _pilot_work_item_from_plan(
                plan=plan,
                plan_binding=row.run_plan,
                budget=budget,
                preflight=preflight,
                verification_receipt_sha256=verification.sha256,
                canonical_index=index,
            )
        )
    rebuilt = FormalPilotDynamicDispatchSchedule(
        schema_version=1,
        kind="lightcone_formal_pilot_dynamic_dispatch_schedule",
        protocol_sha256=FORMAL_PILOT_DYNAMIC_DISPATCH_PROTOCOL_SHA256,
        stage=budget.stage,
        protocol_lock_sha256=budget.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(budget.runtime_authority_manifest_sha256),
        registry_sha256=lock.registry_sha256,
        materialization=budget.materialization,
        materialization_receipt_sha256=materialization.sha256,
        inventory_sha256=preflight.inventory.sha256,
        hardware_envelope_sha256=next(iter(hardware)),
        pilot_budget_verification_receipt=(schedule.pilot_budget_verification_receipt),
        pilot_budget_verification_receipt_sha256=verification.sha256,
        pilot_launch_budget=verification.pilot_launch_budget,
        pilot_launch_budget_sha256=budget.sha256,
        launch_nonce_sha256=budget.launch_nonce_sha256,
        activated_cell_ids=budget.minimum_pilot_cell_ids,
        work_items=tuple(rebuilt_items),
    )
    if rebuilt != schedule:
        raise ValueError("formal pilot dispatch differs from deterministic rebuild")
    return schedule


def materialize_formal_pilot_dynamic_dispatch_schedule(
    *,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    run_plan_paths: tuple[str | Path, ...],
    pilot_budget_verification_receipt_path: str | Path,
    output_path: str | Path,
    current_ns: int,
) -> FormalPilotDynamicDispatchSchedule:
    """Publish exactly the verified minimum pilots; no IDs/caps/ports are inputs."""

    if (
        type(execution_bindings) is not tuple
        or type(run_plan_paths) is not tuple
        or not execution_bindings
        or len(execution_bindings) != len(run_plan_paths)
    ):
        raise ValueError("formal pilot dispatch execution inputs are not exact")
    verification_binding = CanonicalJsonProofBinding.bind(
        pilot_budget_verification_receipt_path
    )
    verification, budget, preflight, materialization = _load_pilot_dispatch_budget(
        verification_binding,
        current_ns=current_ns,
    )
    paths_by_cell: dict[
        str, tuple[FormalServingRunPlan, CanonicalJsonProofBinding]
    ] = {}
    for token, plan_path in zip(execution_bindings, run_plan_paths, strict=True):
        verified = require_verified_formal_serving_execution_binding(token)
        subject = verified.subject
        if (
            subject.stage != budget.stage
            or subject.materialization_receipt_sha256 != materialization.sha256
            or subject.materialized_cell_id not in budget.minimum_pilot_cell_ids
            or subject.protocol_lock_sha256 != budget.protocol_lock_sha256
            or subject.formal_runtime_authority_manifest_sha256
            != budget.runtime_authority_manifest_sha256
            or subject.inventory_sha256 != preflight.inventory.sha256
        ):
            raise ValueError("formal pilot execution binding differs from budget")
        plan = load_formal_serving_run_plan(
            plan_path,
            execution_binding=verified,
            verified_nextn_tp2_authority=verified.verified_nextn_tp2_authority,
        )
        binding = CanonicalJsonProofBinding.bind(plan_path, semantic_sha256=plan.sha256)
        if plan.materialized_cell_id in paths_by_cell:
            raise ValueError("formal pilot dispatch has duplicate run plans")
        paths_by_cell[plan.materialized_cell_id] = (plan, binding)
    if tuple(sorted(paths_by_cell)) != budget.minimum_pilot_cell_ids:
        raise ValueError("formal pilot dispatch does not cover exact minimum pilots")
    items = tuple(
        _pilot_work_item_from_plan(
            plan=paths_by_cell[cell_id][0],
            plan_binding=paths_by_cell[cell_id][1],
            budget=budget,
            preflight=preflight,
            verification_receipt_sha256=verification.sha256,
            canonical_index=index,
        )
        for index, cell_id in enumerate(budget.minimum_pilot_cell_ids)
    )
    lock = preflight.registry_receipt.signed_protocol_lock.payload
    devices = tuple(sorted(preflight.inventory.devices, key=lambda row: row.uuid))
    hardware = {row.hardware_envelope_sha256 for row in devices}
    if len(hardware) != 1:
        raise ValueError("formal pilot inventory is not homogeneous")
    schedule = FormalPilotDynamicDispatchSchedule(
        schema_version=1,
        kind="lightcone_formal_pilot_dynamic_dispatch_schedule",
        protocol_sha256=FORMAL_PILOT_DYNAMIC_DISPATCH_PROTOCOL_SHA256,
        stage=budget.stage,
        protocol_lock_sha256=budget.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(budget.runtime_authority_manifest_sha256),
        registry_sha256=lock.registry_sha256,
        materialization=budget.materialization,
        materialization_receipt_sha256=materialization.sha256,
        inventory_sha256=preflight.inventory.sha256,
        hardware_envelope_sha256=next(iter(hardware)),
        pilot_budget_verification_receipt=verification_binding,
        pilot_budget_verification_receipt_sha256=verification.sha256,
        pilot_launch_budget=verification.pilot_launch_budget,
        pilot_launch_budget_sha256=budget.sha256,
        launch_nonce_sha256=budget.launch_nonce_sha256,
        activated_cell_ids=budget.minimum_pilot_cell_ids,
        work_items=items,
    )
    publish_canonical_json_no_replace(output_path, schedule.to_dict())
    reopened = FormalPilotDynamicDispatchSchedule.from_dict(
        CanonicalJsonProofBinding.bind(output_path).reopen()
    )
    if reopened != schedule:
        raise RuntimeError("formal pilot dynamic dispatch changed during publication")
    revalidate_formal_pilot_dynamic_dispatch_schedule(
        schedule,
        current_ns=current_ns,
    )
    return schedule


__all__ = [
    "FORMAL_DYNAMIC_DISPATCH_PORT_BASE",
    "FORMAL_DYNAMIC_DISPATCH_PROTOCOL_SHA256",
    "FORMAL_DYNAMIC_STAGE_CAPACITY_PROTOCOL_SHA256",
    "FORMAL_PILOT_DYNAMIC_DISPATCH_PROTOCOL_SHA256",
    "FormalDynamicDispatchSchedule",
    "FormalDynamicDispatchWorkItem",
    "FormalDynamicStageCapacitySchedule",
    "FormalPilotDynamicDispatchSchedule",
    "materialize_formal_dynamic_dispatch_schedule",
    "materialize_formal_dynamic_stage_capacity_schedule",
    "materialize_formal_launch_cap_schedule",
    "materialize_formal_pilot_dynamic_dispatch_schedule",
    "revalidate_formal_dynamic_dispatch_schedule",
    "revalidate_formal_dynamic_stage_capacity_schedule",
    "revalidate_formal_pilot_dynamic_dispatch_schedule",
]
