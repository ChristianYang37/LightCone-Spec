"""Deterministic singleton dispatch for E5 one-shot failure diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from lightcone_spec.experiments.capacity_authority import (
    UnsignedCapacitySourceReplay,
    replay_unsigned_capacity_source_manifest,
)
from lightcone_spec.experiments.formal_failure_execution import (
    FormalFailureExecutionRebuildInput,
)
from lightcone_spec.experiments.formal_gpu_hour_registry import (
    FormalStageGpuHourVerificationReceipt,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.orchestration.formal_e5_launch_budget import (
    FormalE5OneShotLaunchBudget,
    revalidate_formal_e5_one_shot_budget_verification,
)
from lightcone_spec.orchestration.formal_physical_dispatch import (
    FormalServingRunPlan,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_E5_ONE_SHOT_PORT = 31_000
FORMAL_E5_ONE_SHOT_DYNAMIC_DISPATCH_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_e5_one_shot_dynamic_dispatch_protocol",
        "cells": "exact_registered_264_failure_rows",
        "waves": "canonical_singleton_sequence_no_overlap",
        "placement": "topology_and_sorted_inventory_index",
        "port": FORMAL_E5_ONE_SHOT_PORT,
        "inputs": "durable_failure_rebuild_descriptors_and_run_plans",
        "caller_gpu_port_timeout_wave": "forbidden",
    }
)
FORMAL_E5_ONE_SHOT_CAPACITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_e5_one_shot_capacity_protocol",
        "dispatch": FORMAL_E5_ONE_SHOT_DYNAMIC_DISPATCH_PROTOCOL_SHA256,
        "raw_capacity": "exact_264_materialized_ids",
        "attempts": 1,
    }
)


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _strict(label: str, value: object, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _rank_groups(
    topology: str,
    gpu_uuids: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    if topology == "tp1_dp1" and len(gpu_uuids) == 1:
        return (gpu_uuids,)
    if topology == "tp2_dp1" and len(gpu_uuids) == 2:
        return (gpu_uuids,)
    if topology == "tp1_dp2" and len(gpu_uuids) == 2:
        return tuple((uuid,) for uuid in gpu_uuids)
    raise ValueError("formal E5 one-shot topology/GPU placement differs")


@dataclass(frozen=True)
class FormalE5OneShotDispatchWorkItem:
    materialized_cell_id: str
    failure_execution_rebuild_input: CanonicalJsonProofBinding
    failure_execution_rebuild_input_sha256: str
    failure_execution_binding_sha256: str
    serving_execution_binding_sha256: str
    execution_plan_sha256: str
    run_plan: CanonicalJsonProofBinding
    run_plan_sha256: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    gpu_uuids: tuple[str, ...]
    rank_groups: tuple[tuple[str, ...], ...]
    localhost_port: Literal[31000]
    private_output_root: str
    wave_group_sha256: str
    process_hard_timeout_ns: int
    provider_wave_hard_timeout_ns: int
    provider_reserved_gpu_count: Literal[2]
    maximum_compute_gpu_ns: int
    maximum_provider_reserved_gpu_ns: int
    allowed_attempts: Literal[1]

    def __post_init__(self) -> None:
        for label, value in (
            ("cell", self.materialized_cell_id),
            ("failure descriptor", self.failure_execution_rebuild_input_sha256),
            ("failure binding", self.failure_execution_binding_sha256),
            ("serving binding", self.serving_execution_binding_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("run plan", self.run_plan_sha256),
            ("wave", self.wave_group_sha256),
        ):
            _sha256(f"formal E5 one-shot work {label}", value)
        if (
            type(self.failure_execution_rebuild_input) is not CanonicalJsonProofBinding
            or type(self.run_plan) is not CanonicalJsonProofBinding
        ):
            raise TypeError("formal E5 one-shot work sources are not path-bound")
        root = Path(self.private_output_root)
        if (
            self.rank_groups != _rank_groups(self.topology_mode, self.gpu_uuids)
            or self.localhost_port != FORMAL_E5_ONE_SHOT_PORT
            or not root.is_absolute()
            or root != root.resolve(strict=False)
            or self.provider_reserved_gpu_count != 2
            or self.allowed_attempts != 1
            or self.process_hard_timeout_ns < 1
            or self.provider_wave_hard_timeout_ns < self.process_hard_timeout_ns
            or self.maximum_compute_gpu_ns
            != self.process_hard_timeout_ns * len(self.gpu_uuids)
            or self.maximum_provider_reserved_gpu_ns
            != self.provider_wave_hard_timeout_ns * 2
        ):
            raise ValueError("formal E5 one-shot work placement/cap differs")

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "failure_execution_rebuild_input": (
                self.failure_execution_rebuild_input.to_dict()
            ),
            "run_plan": self.run_plan.to_dict(),
            "gpu_uuids": list(self.gpu_uuids),
            "rank_groups": [list(group) for group in self.rank_groups],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal E5 one-shot dispatch work",
            value,
            set(cls.__dataclass_fields__),
        )
        row["failure_execution_rebuild_input"] = CanonicalJsonProofBinding.from_dict(
            row["failure_execution_rebuild_input"]
        )
        row["run_plan"] = CanonicalJsonProofBinding.from_dict(row["run_plan"])
        for name in ("gpu_uuids", "rank_groups"):
            if type(row[name]) is not list:
                raise TypeError(f"formal E5 one-shot {name} is not an array")
        row["gpu_uuids"] = tuple(row["gpu_uuids"])
        row["rank_groups"] = tuple(tuple(group) for group in row["rank_groups"])
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalE5OneShotDispatchSchedule:
    schema_version: Literal[1]
    kind: Literal["lightcone_formal_e5_one_shot_dispatch_schedule"]
    protocol_sha256: str
    budget_verification: CanonicalJsonProofBinding
    budget_verification_sha256: str
    budget_sha256: str
    protocol_lock_sha256: str
    runtime_authority_manifest_sha256: str
    registry_sha256: str
    materialization_receipt_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    activated_cell_ids: tuple[str, ...]
    work_items: tuple[FormalE5OneShotDispatchWorkItem, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_e5_one_shot_dispatch_schedule"
            or self.protocol_sha256
            != FORMAL_E5_ONE_SHOT_DYNAMIC_DISPATCH_PROTOCOL_SHA256
        ):
            raise ValueError("formal E5 one-shot dispatch schema differs")
        for label, value in (
            ("budget verification", self.budget_verification_sha256),
            ("budget", self.budget_sha256),
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime", self.runtime_authority_manifest_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware", self.hardware_envelope_sha256),
        ):
            _sha256(f"formal E5 one-shot dispatch {label}", value)
        ids = tuple(row.materialized_cell_id for row in self.work_items)
        roots = tuple(row.private_output_root for row in self.work_items)
        if (
            type(self.budget_verification) is not CanonicalJsonProofBinding
            or len(self.work_items) != 264
            or ids != tuple(sorted(set(ids)))
            or self.activated_cell_ids != ids
            or len(set(roots)) != 264
            or len({row.wave_group_sha256 for row in self.work_items}) != 264
        ):
            raise ValueError("formal E5 one-shot dispatch coverage differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "budget_verification": self.budget_verification.to_dict(),
            "activated_cell_ids": list(self.activated_cell_ids),
            "work_items": [row.to_dict() for row in self.work_items],
            "schedule_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal E5 one-shot dispatch schedule",
            value,
            {*cls.__dataclass_fields__, "schedule_sha256"},
        )
        declared = _sha256(
            "formal E5 one-shot dispatch schedule",
            row.pop("schedule_sha256"),
        )
        row["budget_verification"] = CanonicalJsonProofBinding.from_dict(
            row["budget_verification"]
        )
        raw_ids = row.pop("activated_cell_ids")
        raw_items = row.pop("work_items")
        if type(raw_ids) is not list or type(raw_items) is not list:
            raise TypeError("formal E5 one-shot dispatch collections differ")
        schedule = cls(
            **row,
            activated_cell_ids=tuple(raw_ids),
            work_items=tuple(
                FormalE5OneShotDispatchWorkItem.from_dict(item) for item in raw_items
            ),
        )
        if schedule.sha256 != declared:
            raise ValueError("formal E5 one-shot dispatch digest differs")
        return schedule

    def work_item(self, cell_id: str) -> FormalE5OneShotDispatchWorkItem:
        rows = tuple(
            row for row in self.work_items if row.materialized_cell_id == cell_id
        )
        if len(rows) != 1:
            raise ValueError("formal E5 one-shot dispatch lacks exact cell")
        return rows[0]


def _inventory_gpu_uuids(
    preflight: FormalStageGpuHourVerificationReceipt,
) -> tuple[str, str]:
    devices = tuple(sorted(preflight.inventory.devices, key=lambda row: row.uuid))
    if (
        len(devices) != 2
        or any(not row.ready for row in devices)
        or len({row.hardware_envelope_sha256 for row in devices}) != 1
    ):
        raise ValueError("formal E5 one-shot dispatch requires homogeneous two GPUs")
    return devices[0].uuid, devices[1].uuid


def _work_item(
    *,
    index: int,
    descriptor_binding: CanonicalJsonProofBinding,
    descriptor: FormalFailureExecutionRebuildInput,
    plan_binding: CanonicalJsonProofBinding,
    plan: FormalServingRunPlan,
    budget: FormalE5OneShotLaunchBudget,
    inventory_gpus: tuple[str, str],
) -> FormalE5OneShotDispatchWorkItem:
    subject = descriptor.subject
    cap = budget.cap_for(subject.materialized_cell_id)
    expected_gpus = (
        (inventory_gpus[index % 2],)
        if cap.topology_mode == "tp1_dp1"
        else inventory_gpus
    )
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    if (
        subject.materialization_receipt_sha256 != budget.materialization_receipt_sha256
        or subject.protocol_lock_sha256 != budget.protocol_lock_sha256
        or subject.formal_runtime_authority_manifest_sha256
        != budget.runtime_authority_manifest_sha256
        or subject.registry_sha256 != budget.registry_sha256
        or subject.inventory_sha256 != budget.inventory_sha256
        or subject.backend != cap.backend
        or subject.topology != cap.topology_mode
        or subject.scenario != cap.scenario
        or subject.cohort_count != cap.cohort_count
        or plan.stage != "E5"
        or plan.method != "l0"
        or plan.materialized_cell_id != subject.materialized_cell_id
        or plan.execution_binding_sha256 != subject.serving_execution_binding_sha256
        or plan.native_terminal_binding.execution_plan_sha256
        != subject.serving_execution_plan_sha256
        or plan.native_terminal_binding.rank_config_sha256
        != subject.serving_rank_config_sha256
        or plan.native_terminal_binding.run_nonce_sha256 != subject.run_nonce_sha256
        or plan.inventory_sha256 != budget.inventory_sha256
        or plan.topology_mode != cap.topology_mode
        or plan.gpu_uuids != expected_gpus
        or launch.gpu_uuids != expected_gpus
        or launch.localhost_port != FORMAL_E5_ONE_SHOT_PORT
        or launch.inventory_sha256 != budget.inventory_sha256
        or Path(plan.private_output_root) != Path(plan_binding.absolute_path).parent
    ):
        raise ValueError("formal E5 one-shot run plan differs from source placement")
    wave = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_e5_one_shot_singleton_wave",
            "budget_sha256": budget.sha256,
            "wave_index": index,
            "materialized_cell_id": subject.materialized_cell_id,
        }
    )
    return FormalE5OneShotDispatchWorkItem(
        materialized_cell_id=subject.materialized_cell_id,
        failure_execution_rebuild_input=descriptor_binding,
        failure_execution_rebuild_input_sha256=descriptor.sha256,
        failure_execution_binding_sha256=(
            descriptor.expected_failure_execution_binding_sha256
        ),
        serving_execution_binding_sha256=subject.serving_execution_binding_sha256,
        execution_plan_sha256=subject.serving_execution_plan_sha256,
        run_plan=plan_binding,
        run_plan_sha256=plan.sha256,
        topology_mode=cap.topology_mode,
        gpu_uuids=expected_gpus,
        rank_groups=_rank_groups(cap.topology_mode, expected_gpus),
        localhost_port=FORMAL_E5_ONE_SHOT_PORT,
        private_output_root=plan.private_output_root,
        wave_group_sha256=wave,
        process_hard_timeout_ns=cap.process_hard_timeout_ns,
        provider_wave_hard_timeout_ns=cap.provider_wave_hard_timeout_ns,
        provider_reserved_gpu_count=2,
        maximum_compute_gpu_ns=cap.maximum_compute_gpu_ns,
        maximum_provider_reserved_gpu_ns=cap.maximum_provider_reserved_gpu_ns,
        allowed_attempts=1,
    )


def revalidate_formal_e5_one_shot_dispatch_schedule(
    schedule: FormalE5OneShotDispatchSchedule,
    *,
    current_ns: int,
) -> FormalE5OneShotDispatchSchedule:
    if (
        CanonicalJsonProofBinding.bind(schedule.budget_verification.absolute_path)
        != schedule.budget_verification
    ):
        raise ValueError("formal E5 one-shot dispatch budget binding changed")
    verification, budget, _registry = revalidate_formal_e5_one_shot_budget_verification(
        schedule.budget_verification,
        current_ns=current_ns,
    )
    preflight = FormalStageGpuHourVerificationReceipt.from_dict(
        budget.preflight_budget_receipt.reopen()
    )
    inventory_gpus = _inventory_gpu_uuids(preflight)
    rebuilt_items = []
    for index, row in enumerate(schedule.work_items):
        if (
            CanonicalJsonProofBinding.bind(
                row.failure_execution_rebuild_input.absolute_path
            )
            != row.failure_execution_rebuild_input
            or CanonicalJsonProofBinding.bind(row.run_plan.absolute_path)
            != row.run_plan
        ):
            raise ValueError("formal E5 one-shot dispatch source changed")
        descriptor = FormalFailureExecutionRebuildInput.from_dict(
            row.failure_execution_rebuild_input.reopen()
        )
        plan = FormalServingRunPlan.from_dict(row.run_plan.reopen())
        rebuilt_items.append(
            _work_item(
                index=index,
                descriptor_binding=row.failure_execution_rebuild_input,
                descriptor=descriptor,
                plan_binding=row.run_plan,
                plan=plan,
                budget=budget,
                inventory_gpus=inventory_gpus,
            )
        )
    hardware = preflight.inventory.devices[0].hardware_envelope_sha256
    rebuilt = FormalE5OneShotDispatchSchedule(
        schema_version=1,
        kind="lightcone_formal_e5_one_shot_dispatch_schedule",
        protocol_sha256=(FORMAL_E5_ONE_SHOT_DYNAMIC_DISPATCH_PROTOCOL_SHA256),
        budget_verification=schedule.budget_verification,
        budget_verification_sha256=verification.sha256,
        budget_sha256=budget.sha256,
        protocol_lock_sha256=budget.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(budget.runtime_authority_manifest_sha256),
        registry_sha256=budget.registry_sha256,
        materialization_receipt_sha256=budget.materialization_receipt_sha256,
        inventory_sha256=budget.inventory_sha256,
        hardware_envelope_sha256=hardware,
        activated_cell_ids=tuple(row.materialized_cell_id for row in rebuilt_items),
        work_items=tuple(rebuilt_items),
    )
    if rebuilt != schedule:
        raise ValueError(
            "formal E5 one-shot dispatch differs from deterministic rebuild"
        )
    return schedule


def materialize_formal_e5_one_shot_dispatch_schedule(
    *,
    budget_verification_receipt_path: str | Path,
    failure_execution_rebuild_input_paths: tuple[str | Path, ...],
    run_plan_paths: tuple[str | Path, ...],
    output_path: str | Path,
    current_ns: int,
) -> FormalE5OneShotDispatchSchedule:
    if (
        type(failure_execution_rebuild_input_paths) is not tuple
        or type(run_plan_paths) is not tuple
        or len(failure_execution_rebuild_input_paths) != 264
        or len(run_plan_paths) != 264
    ):
        raise ValueError("formal E5 one-shot dispatch requires exact 264 paths")
    verification_binding = CanonicalJsonProofBinding.bind(
        budget_verification_receipt_path
    )
    verification, budget, _registry = revalidate_formal_e5_one_shot_budget_verification(
        verification_binding,
        current_ns=current_ns,
    )
    preflight = FormalStageGpuHourVerificationReceipt.from_dict(
        budget.preflight_budget_receipt.reopen()
    )
    inventory_gpus = _inventory_gpu_uuids(preflight)
    sources = []
    for descriptor_path, plan_path in zip(
        failure_execution_rebuild_input_paths,
        run_plan_paths,
        strict=True,
    ):
        descriptor_binding = CanonicalJsonProofBinding.bind(descriptor_path)
        descriptor = FormalFailureExecutionRebuildInput.from_dict(
            descriptor_binding.reopen()
        )
        plan_binding = CanonicalJsonProofBinding.bind(plan_path)
        plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
        sources.append((descriptor_binding, descriptor, plan_binding, plan))
    sources.sort(key=lambda row: row[1].subject.materialized_cell_id)
    items = tuple(
        _work_item(
            index=index,
            descriptor_binding=descriptor_binding,
            descriptor=descriptor,
            plan_binding=plan_binding,
            plan=plan,
            budget=budget,
            inventory_gpus=inventory_gpus,
        )
        for index, (descriptor_binding, descriptor, plan_binding, plan) in enumerate(
            sources
        )
    )
    schedule = FormalE5OneShotDispatchSchedule(
        schema_version=1,
        kind="lightcone_formal_e5_one_shot_dispatch_schedule",
        protocol_sha256=FORMAL_E5_ONE_SHOT_DYNAMIC_DISPATCH_PROTOCOL_SHA256,
        budget_verification=verification_binding,
        budget_verification_sha256=verification.sha256,
        budget_sha256=budget.sha256,
        protocol_lock_sha256=budget.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(budget.runtime_authority_manifest_sha256),
        registry_sha256=budget.registry_sha256,
        materialization_receipt_sha256=budget.materialization_receipt_sha256,
        inventory_sha256=budget.inventory_sha256,
        hardware_envelope_sha256=(
            preflight.inventory.devices[0].hardware_envelope_sha256
        ),
        activated_cell_ids=tuple(row.materialized_cell_id for row in items),
        work_items=items,
    )
    publish_canonical_json_no_replace(output_path, schedule.to_dict())
    reopened = FormalE5OneShotDispatchSchedule.from_dict(
        CanonicalJsonProofBinding.bind(output_path).reopen()
    )
    if reopened != schedule:
        raise RuntimeError("formal E5 one-shot dispatch changed during publication")
    revalidate_formal_e5_one_shot_dispatch_schedule(schedule, current_ns=current_ns)
    return schedule


@dataclass(frozen=True)
class FormalE5OneShotStageCapacitySchedule:
    schema_version: Literal[1]
    kind: Literal["lightcone_formal_e5_one_shot_stage_capacity_schedule"]
    protocol_sha256: str
    stage: Literal["E5"]
    protocol_lock_sha256: str
    runtime_authority_manifest_sha256: str
    registry_sha256: str
    materialization_receipt_sha256: str
    dispatch_schedule: CanonicalJsonProofBinding
    dispatch_schedule_sha256: str
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
    localhost_port: Literal[31000]
    wave_index: int
    wave_group_sha256: str
    wave_cell_ids: tuple[str, ...]
    provider_inventory_gpu_count: Literal[2]
    provider_reserved_gpu_count: Literal[2]
    retry_allowance: Literal[0]
    process_hard_timeout_ns: int
    provider_wave_hard_timeout_ns: int
    maximum_compute_gpu_ns_per_attempt: int
    maximum_provider_reserved_gpu_ns_per_attempt: int
    failure_execution_rebuild_input: CanonicalJsonProofBinding
    failure_execution_binding_sha256: str
    capacity_source_manifest: CanonicalJsonProofBinding
    capacity_envelope_sha256: str
    budget_inventory_sha256: str
    capacity_captured_at_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_e5_one_shot_stage_capacity_schedule"
            or self.protocol_sha256 != FORMAL_E5_ONE_SHOT_CAPACITY_PROTOCOL_SHA256
            or self.stage != "E5"
        ):
            raise ValueError("formal E5 one-shot capacity schema differs")
        for label, value in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime", self.runtime_authority_manifest_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("dispatch", self.dispatch_schedule_sha256),
            ("cell", self.materialized_cell_id),
            ("execution binding", self.execution_binding_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("run plan", self.run_plan_sha256),
            ("inventory", self.inventory_sha256),
            ("wave", self.wave_group_sha256),
            ("failure binding", self.failure_execution_binding_sha256),
            ("capacity", self.capacity_envelope_sha256),
            ("budget inventory", self.budget_inventory_sha256),
        ):
            _sha256(f"formal E5 one-shot capacity {label}", value)
        if any(
            type(value) is not CanonicalJsonProofBinding
            for value in (
                self.dispatch_schedule,
                self.run_plan,
                self.failure_execution_rebuild_input,
                self.capacity_source_manifest,
            )
        ):
            raise TypeError("formal E5 one-shot capacity sources are not path-bound")
        if (
            self.materialized_cell_id not in self.activated_cell_ids
            or self.wave_cell_ids != (self.materialized_cell_id,)
            or self.provider_inventory_gpu_count != 2
            or self.provider_reserved_gpu_count != 2
            or self.retry_allowance != 0
            or type(self.wave_index) is not int
            or not 0 <= self.wave_index < 264
            or self.rank_groups != _rank_groups(self.topology_mode, self.gpu_uuids)
            or self.localhost_port != FORMAL_E5_ONE_SHOT_PORT
            or self.maximum_compute_gpu_ns_per_attempt
            != self.process_hard_timeout_ns * len(self.gpu_uuids)
            or self.maximum_provider_reserved_gpu_ns_per_attempt
            != self.provider_wave_hard_timeout_ns * 2
        ):
            raise ValueError("formal E5 one-shot capacity placement differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "dispatch_schedule": self.dispatch_schedule.to_dict(),
            "activated_cell_ids": list(self.activated_cell_ids),
            "run_plan": self.run_plan.to_dict(),
            "gpu_uuids": list(self.gpu_uuids),
            "rank_groups": [list(group) for group in self.rank_groups],
            "wave_cell_ids": list(self.wave_cell_ids),
            "failure_execution_rebuild_input": (
                self.failure_execution_rebuild_input.to_dict()
            ),
            "capacity_source_manifest": self.capacity_source_manifest.to_dict(),
            "schedule_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal E5 one-shot capacity schedule",
            value,
            {*cls.__dataclass_fields__, "schedule_sha256"},
        )
        declared = _sha256(
            "formal E5 one-shot capacity schedule",
            row.pop("schedule_sha256"),
        )
        for name in (
            "dispatch_schedule",
            "run_plan",
            "failure_execution_rebuild_input",
            "capacity_source_manifest",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        for name in ("activated_cell_ids", "gpu_uuids", "rank_groups", "wave_cell_ids"):
            if type(row[name]) is not list:
                raise TypeError(f"formal E5 one-shot capacity {name} is not an array")
        row["activated_cell_ids"] = tuple(row["activated_cell_ids"])
        row["gpu_uuids"] = tuple(row["gpu_uuids"])
        row["rank_groups"] = tuple(tuple(group) for group in row["rank_groups"])
        row["wave_cell_ids"] = tuple(row["wave_cell_ids"])
        schedule = cls(**row)  # type: ignore[arg-type]
        if schedule.sha256 != declared:
            raise ValueError("formal E5 one-shot capacity digest differs")
        return schedule

    @property
    def budget_verification(self) -> CanonicalJsonProofBinding:
        dispatch = FormalE5OneShotDispatchSchedule.from_dict(
            self.dispatch_schedule.reopen()
        )
        return dispatch.budget_verification


def _replay_capacity(
    schedule: FormalE5OneShotStageCapacitySchedule,
    *,
    current_ns: int,
) -> UnsignedCapacitySourceReplay:
    replay = replay_unsigned_capacity_source_manifest(
        schedule.capacity_source_manifest.absolute_path,
        expected_registry_sha256=schedule.registry_sha256,
        now_ns=current_ns,
    )
    if (
        replay.capacity_envelope.sha256 != schedule.capacity_envelope_sha256
        or replay.budget_inventory.sha256 != schedule.budget_inventory_sha256
        or replay.gpu_inventory.sha256 != schedule.inventory_sha256
        or replay.captured_at_ns != schedule.capacity_captured_at_ns
        or tuple(row.cell_id for row in replay.capacity_envelope.cell_requirements)
        != schedule.activated_cell_ids
    ):
        raise ValueError("formal E5 one-shot raw capacity differs")
    return replay


def revalidate_formal_e5_one_shot_stage_capacity_schedule(
    schedule: FormalE5OneShotStageCapacitySchedule,
    *,
    current_ns: int,
) -> UnsignedCapacitySourceReplay:
    if (
        CanonicalJsonProofBinding.bind(schedule.dispatch_schedule.absolute_path)
        != schedule.dispatch_schedule
    ):
        raise ValueError("formal E5 one-shot capacity dispatch changed")
    dispatch = FormalE5OneShotDispatchSchedule.from_dict(
        schedule.dispatch_schedule.reopen()
    )
    revalidate_formal_e5_one_shot_dispatch_schedule(
        dispatch,
        current_ns=current_ns,
    )
    work = dispatch.work_item(schedule.materialized_cell_id)
    index = dispatch.activated_cell_ids.index(schedule.materialized_cell_id)
    rebuilt = FormalE5OneShotStageCapacitySchedule(
        schema_version=1,
        kind="lightcone_formal_e5_one_shot_stage_capacity_schedule",
        protocol_sha256=FORMAL_E5_ONE_SHOT_CAPACITY_PROTOCOL_SHA256,
        stage="E5",
        protocol_lock_sha256=dispatch.protocol_lock_sha256,
        runtime_authority_manifest_sha256=dispatch.runtime_authority_manifest_sha256,
        registry_sha256=dispatch.registry_sha256,
        materialization_receipt_sha256=dispatch.materialization_receipt_sha256,
        dispatch_schedule=schedule.dispatch_schedule,
        dispatch_schedule_sha256=dispatch.sha256,
        activated_cell_ids=dispatch.activated_cell_ids,
        materialized_cell_id=work.materialized_cell_id,
        execution_binding_sha256=work.serving_execution_binding_sha256,
        execution_plan_sha256=work.execution_plan_sha256,
        run_plan=work.run_plan,
        run_plan_sha256=work.run_plan_sha256,
        topology_mode=work.topology_mode,
        inventory_sha256=dispatch.inventory_sha256,
        gpu_uuids=work.gpu_uuids,
        rank_groups=work.rank_groups,
        localhost_port=work.localhost_port,
        wave_index=index,
        wave_group_sha256=work.wave_group_sha256,
        wave_cell_ids=(work.materialized_cell_id,),
        provider_inventory_gpu_count=2,
        provider_reserved_gpu_count=2,
        retry_allowance=0,
        process_hard_timeout_ns=work.process_hard_timeout_ns,
        provider_wave_hard_timeout_ns=work.provider_wave_hard_timeout_ns,
        maximum_compute_gpu_ns_per_attempt=work.maximum_compute_gpu_ns,
        maximum_provider_reserved_gpu_ns_per_attempt=(
            work.maximum_provider_reserved_gpu_ns
        ),
        failure_execution_rebuild_input=work.failure_execution_rebuild_input,
        failure_execution_binding_sha256=work.failure_execution_binding_sha256,
        capacity_source_manifest=schedule.capacity_source_manifest,
        capacity_envelope_sha256=schedule.capacity_envelope_sha256,
        budget_inventory_sha256=schedule.budget_inventory_sha256,
        capacity_captured_at_ns=schedule.capacity_captured_at_ns,
    )
    if rebuilt != schedule:
        raise ValueError(
            "formal E5 one-shot capacity differs from deterministic rebuild"
        )
    return _replay_capacity(schedule, current_ns=current_ns)


def materialize_formal_e5_one_shot_stage_capacity_schedule(
    *,
    dispatch_schedule_path: str | Path,
    materialized_cell_id: str,
    capacity_source_manifest_path: str | Path,
    output_path: str | Path,
    current_ns: int,
) -> FormalE5OneShotStageCapacitySchedule:
    dispatch_binding = CanonicalJsonProofBinding.bind(dispatch_schedule_path)
    dispatch = FormalE5OneShotDispatchSchedule.from_dict(dispatch_binding.reopen())
    revalidate_formal_e5_one_shot_dispatch_schedule(dispatch, current_ns=current_ns)
    work = dispatch.work_item(materialized_cell_id)
    source_binding = CanonicalJsonProofBinding.bind(capacity_source_manifest_path)
    replay = replay_unsigned_capacity_source_manifest(
        source_binding.absolute_path,
        expected_registry_sha256=dispatch.registry_sha256,
        now_ns=current_ns,
    )
    schedule = FormalE5OneShotStageCapacitySchedule(
        schema_version=1,
        kind="lightcone_formal_e5_one_shot_stage_capacity_schedule",
        protocol_sha256=FORMAL_E5_ONE_SHOT_CAPACITY_PROTOCOL_SHA256,
        stage="E5",
        protocol_lock_sha256=dispatch.protocol_lock_sha256,
        runtime_authority_manifest_sha256=dispatch.runtime_authority_manifest_sha256,
        registry_sha256=dispatch.registry_sha256,
        materialization_receipt_sha256=dispatch.materialization_receipt_sha256,
        dispatch_schedule=dispatch_binding,
        dispatch_schedule_sha256=dispatch.sha256,
        activated_cell_ids=dispatch.activated_cell_ids,
        materialized_cell_id=work.materialized_cell_id,
        execution_binding_sha256=work.serving_execution_binding_sha256,
        execution_plan_sha256=work.execution_plan_sha256,
        run_plan=work.run_plan,
        run_plan_sha256=work.run_plan_sha256,
        topology_mode=work.topology_mode,
        inventory_sha256=dispatch.inventory_sha256,
        gpu_uuids=work.gpu_uuids,
        rank_groups=work.rank_groups,
        localhost_port=work.localhost_port,
        wave_index=dispatch.activated_cell_ids.index(materialized_cell_id),
        wave_group_sha256=work.wave_group_sha256,
        wave_cell_ids=(materialized_cell_id,),
        provider_inventory_gpu_count=2,
        provider_reserved_gpu_count=2,
        retry_allowance=0,
        process_hard_timeout_ns=work.process_hard_timeout_ns,
        provider_wave_hard_timeout_ns=work.provider_wave_hard_timeout_ns,
        maximum_compute_gpu_ns_per_attempt=work.maximum_compute_gpu_ns,
        maximum_provider_reserved_gpu_ns_per_attempt=(
            work.maximum_provider_reserved_gpu_ns
        ),
        failure_execution_rebuild_input=work.failure_execution_rebuild_input,
        failure_execution_binding_sha256=work.failure_execution_binding_sha256,
        capacity_source_manifest=source_binding,
        capacity_envelope_sha256=replay.capacity_envelope.sha256,
        budget_inventory_sha256=replay.budget_inventory.sha256,
        capacity_captured_at_ns=replay.captured_at_ns,
    )
    _replay_capacity(schedule, current_ns=current_ns)
    publish_canonical_json_no_replace(output_path, schedule.to_dict())
    reopened = FormalE5OneShotStageCapacitySchedule.from_dict(
        CanonicalJsonProofBinding.bind(output_path).reopen()
    )
    if reopened != schedule:
        raise RuntimeError("formal E5 one-shot capacity changed during publication")
    revalidate_formal_e5_one_shot_stage_capacity_schedule(
        schedule,
        current_ns=current_ns,
    )
    return schedule


__all__ = [
    "FORMAL_E5_ONE_SHOT_CAPACITY_PROTOCOL_SHA256",
    "FORMAL_E5_ONE_SHOT_DYNAMIC_DISPATCH_PROTOCOL_SHA256",
    "FORMAL_E5_ONE_SHOT_PORT",
    "FormalE5OneShotDispatchSchedule",
    "FormalE5OneShotDispatchWorkItem",
    "FormalE5OneShotStageCapacitySchedule",
    "materialize_formal_e5_one_shot_dispatch_schedule",
    "materialize_formal_e5_one_shot_stage_capacity_schedule",
    "revalidate_formal_e5_one_shot_dispatch_schedule",
    "revalidate_formal_e5_one_shot_stage_capacity_schedule",
]
