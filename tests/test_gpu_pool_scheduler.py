from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

import lightcone_spec.experiments.gpu_pool as gpu_pool_module
import lightcone_spec.orchestration.execution_bundle as execution_bundle_module
from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.completion_authority import (
    AssignmentTerminalAuthority,
    AssignmentTerminalBinding,
)
from lightcone_spec.experiments.gpu_pool import (
    ArtifactSidecar,
    AssignmentExecutionReceipt,
    AssignmentExecutionStatus,
    CapabilityRejectionError,
    DispatchExecutionPhase,
    DispatchScheduleReceipt,
    GangShape,
    GpuAssignment,
    GpuAvailability,
    GpuDevice,
    GpuDispatchExecutionContext,
    GpuDispatchPlan,
    GpuDispatchPlanningContext,
    GpuDispatchWave,
    GpuInventory,
    GpuPoolScheduler,
    GpuTopologyGroup,
    HomogeneousDeviceConstraint,
    InterferenceEnvelope,
    InterferenceRule,
    PoolWorkItem,
    execute_dispatch_plan,
    registry_pool_work_item,
    supported_pool_size,
    validate_dispatch_resume,
)
from lightcone_spec.experiments.interference_authority import (
    materialize_interference_calibration_bootstrap_authority,
)
from lightcone_spec.experiments.planning import (
    CONFIRMATION_FAMILY_POWER_REDUCER_PROTOCOL_SHA256,
    BudgetJobKind,
    ConfirmationFamilyPowerPlan,
    ConfirmationFamilyPowerReductionArtifact,
    ExpectedMaximumCount,
    ExperimentBudget,
    P99AnchorStatus,
    RawEvidenceRunBinding,
    ScenarioMilliseconds,
    derive_confirmation_family,
    family_pilot_block_id,
    materialize_confirmation_pilots,
    materialize_confirmation_prefix,
)
from lightcone_spec.experiments.registry import (
    CORE_METHODS,
    FINAL_BLOCKS,
    PILOT_BLOCKS,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    WorkloadClass,
    build_industrial_registry,
    content_sha256,
    serving_cell_rejection_reason,
)
from lightcone_spec.experiments.stage_activation import (
    RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE,
    RELEASE_DOWNLOAD_ASSIGNMENT_CONTRACT_UNAVAILABLE,
    RegistryStageActivationArtifact,
    RegistryStageDispositionStatus,
    materialize_registry_stage_activation,
    release_dispatch_rejection_reason,
)
from lightcone_spec.experiments.statistics import PilotBlock, preregister_power_sizing
from lightcone_spec.orchestration.execution_bundle import (
    DispatchAttemptJournal,
    ExecutionBundleBlockedError,
    publish_dispatch_schedule_receipt,
)


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_industrial_registry()


def _inventory(
    size: int,
    *,
    reserved: frozenset[int] = frozenset(),
    paired_topology: bool = False,
    source: str = "inventory",
) -> GpuInventory:
    if paired_topology and size % 2:
        raise ValueError("paired topology needs an even inventory")
    if paired_topology:
        group_specs = tuple(
            (
                f"pair-{index // 2:02d}",
                (f"GPU-{index:03d}", f"GPU-{index + 1:03d}"),
            )
            for index in range(0, size, 2)
        )
    else:
        group_specs = (("all", tuple(f"GPU-{index:03d}" for index in range(size))),)
    allowed_by_uuid: dict[str, list[str]] = {
        f"GPU-{index:03d}": [] for index in range(size)
    }
    for group_id, uuids in group_specs:
        for uuid in uuids:
            allowed_by_uuid[uuid].append(group_id)
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-{index:03d}",
            host_id="host-a",
            model="H100-SXM",
            memory_bytes=80 * 1024**3,
            compute_capability=(9, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root=f"root-{index // 4}",
            numa_node=index // 4,
            interconnects=("NVLink4", "PCIe5"),
            peer_access_class="NVSwitch",
            clock_policy="locked-1980MHz",
            power_limit_watts=700.0,
            thermal_limit_celsius=83.0,
            availability=GpuAvailability.READY,
            reserved_processes=(f"pid-{index}",) if index in reserved else (),
            allowed_topology_groups=tuple(sorted(allowed_by_uuid[f"GPU-{index:03d}"])),
        )
        for index in range(size)
    )
    groups = tuple(
        GpuTopologyGroup(
            group_id=group_id,
            host_id="host-a",
            gpu_uuids=uuids,
            fabric="NVSwitch" if not paired_topology else "NVLink",
            bandwidth_class="high",
        )
        for group_id, uuids in sorted(group_specs)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=groups,
        source_receipt_sha256=content_sha256({"source": source, "size": size}),
    )


def _target_tuning_cells(registry: ExperimentRegistry, count: int):
    cells = tuple(
        cell
        for cell in registry.cells_for("E3a")
        if cell.identity.method == "target_only"
        and serving_cell_rejection_reason(cell) is None
    )
    assert len(cells) >= count
    return cells[:count]


def _envelope_for(
    inventory: GpuInventory,
    item: PoolWorkItem,
    cardinalities: tuple[int, ...],
) -> InterferenceEnvelope:
    rules = tuple(
        sorted(
            (
                InterferenceRule.for_claim(
                    device=inventory.devices[0],
                    claim=item.claim,
                    simultaneous_jobs=count,
                    evidence_sha256=content_sha256(
                        {"calibration": count, "inventory": inventory.sha256}
                    ),
                )
                for count in cardinalities
            ),
            key=lambda rule: rule.key,
        )
    )
    return InterferenceEnvelope(
        schema_version=1,
        rules=rules,
        source_receipt_sha256=content_sha256(
            {"rules": tuple(rule.sha256 for rule in rules)}
        ),
    )


def _scheduler(
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    envelope: InterferenceEnvelope,
) -> GpuPoolScheduler:
    return GpuPoolScheduler(
        registry=registry,
        inventory=inventory,
        interference_envelope=envelope,
        port_start=31_000,
        port_end=31_999,
        seed=20260811,
    )


def _receipts_through(
    registry: ExperimentRegistry, experiment: str
) -> tuple[ExperimentReceipt, ...]:
    receipts: list[ExperimentReceipt] = []
    for definition in registry.definitions:
        receipt = registry.make_receipt(
            definition.name,
            {
                output: content_sha256(
                    {"experiment": definition.name, "output": output}
                )
                for output in definition.locked_outputs
            },
            runtime_sha256=content_sha256(
                {"runtime": definition.name, "kind": "scheduler-test"}
            ),
            split_sha256=content_sha256(
                {"split": definition.name, "kind": "scheduler-test"}
            ),
            completed_cells_sha256=content_sha256(
                {"completed": definition.name, "kind": "scheduler-test"}
            ),
            dependencies=tuple(receipts),
        )
        receipts.append(receipt)
        if definition.name == experiment:
            return tuple(receipts)
    raise AssertionError(f"unknown experiment {experiment}")


def _milliseconds(value: int) -> ScenarioMilliseconds:
    return ScenarioMilliseconds(value, value, value)


def _budget(
    cell: ExperimentCell,
    *,
    fixed_instance_gpu_count: int,
) -> ExperimentBudget:
    component = _milliseconds(100)
    zero = _milliseconds(0)
    wall_ms = 1_700
    gpu_ms = _milliseconds(wall_ms * cell.resources.gpu_count)
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=cell.resources.workload_class,
        job_kind=BudgetJobKind.STANDARD,
        startup_model_load=component,
        compile_jit_graph_prewarm=component,
        excluded_warmup=component,
        excluded_warmup_requests=ExpectedMaximumCount(8, 10),
        scored_arrival=_milliseconds(1_000),
        request_deadline=_milliseconds(5_000),
        drain=component,
        reset_finalization=component,
        evidence_flush_shutdown=component,
        output_tokens=ExpectedMaximumCount(1_000, 2_000),
        minimum_completed_requests=64,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=zero,
        failure_injection=zero,
        retry=component,
        retry_allowance=1,
        profiler=zero,
        download_compile_reservation=zero,
        gpu_count=cell.resources.gpu_count,
        topology=cell.identity.topology,
        reserved_gpu_ms=gpu_ms,
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=_milliseconds(wall_ms * fixed_instance_gpu_count),
    )


def _e5_family(registry: ExperimentRegistry, *, concurrency: int):
    cell = next(
        cell
        for cell in registry.cells_for("E5")
        if cell.identity.backend == "DFLASH"
        and cell.identity.task == "production_crossover"
        and cell.identity.method == "static"
        and cell.identity.arrival == "closed_loop"
        and cell.identity.concurrency == concurrency
        and cell.identity.block == PILOT_BLOCKS[0]
    )
    return derive_confirmation_family(
        registry,
        cell_id=cell.cell_id,
        runtime_sha256=content_sha256({"family": concurrency, "kind": "runtime"}),
        split_sha256=content_sha256({"family": concurrency, "kind": "split"}),
        trace_sha256=content_sha256({"family": concurrency, "kind": "trace"}),
        sampling_sha256=content_sha256({"family": concurrency, "kind": "sampling"}),
        hardware_envelope_sha256=content_sha256(
            {"family": concurrency, "kind": "hardware"}
        ),
    )


def _power_sizing(family):
    multipliers = (0.99, 1.01, 1.00, 1.02)
    return preregister_power_sizing(
        tuple(
            PilotBlock(
                block_id=family_pilot_block_id(family, block),
                static_goodput=100.0,
                tts_goodput=101.0,
                l0_goodput=103.0 * multiplier,
            )
            for block, multiplier in zip(PILOT_BLOCKS, multipliers, strict=True)
        )
    )


def _family_power_reduction_from_sizing(
    registry: ExperimentRegistry,
    family,
    pilot_activation,
    sizing,
    *,
    evidence_label: str,
):
    evidence_sha256 = content_sha256(
        {"family": family.sha256, "evidence": evidence_label}
    )
    if sizing.status == "READY":
        status = "POWERED"
        selected = sizing.selected_final_blocks
        assert selected is not None
        prefix = FINAL_BLOCKS[:selected]
        reason = "registered_family_power_target_met"
    else:
        status = "UNDERPOWERED"
        selected = None
        prefix = ()
        reason = "registered_family_underpowered"
    plan = ConfirmationFamilyPowerPlan(
        schema_version=1,
        family=family,
        pilot_activation_sha256=pilot_activation.sha256,
        completed_pilot_cells_sha256=content_sha256(
            tuple(sorted(pilot_activation.activated_cell_ids))
        ),
        pilot_evidence_sha256=evidence_sha256,
        power_sizing=sizing,
        status=status,
        selected_final_blocks=selected,
        selected_final_prefix=prefix,
        reason_code=reason,
        selection_state="sealed_before_confirmation_unblinding",
    )

    cells = {cell.cell_id: cell for cell in registry.cells}

    def binding_sha256(kind: str, cell_id: str) -> str:
        return content_sha256(
            {"family": family.sha256, "kind": kind, "cell_id": cell_id}
        )

    run_bindings = tuple(
        RawEvidenceRunBinding(
            schema_version=1,
            cell_id=cell_id,
            experiment=family.experiment,
            method=cells[cell_id].identity.method,
            scientific_unit=f"excluded_pilot_{cells[cell_id].identity.block}",
            config_sha256=binding_sha256("config", cell_id),
            rank_config_sha256s=(binding_sha256("rank-config", cell_id),),
            run_id=f"family-power-test-{cell_id}",
            rank_count=1,
            model_pair=family.model,
            runtime_sha256=family.runtime_sha256,
            split_sha256=family.split_sha256,
            corpus_sha256=binding_sha256("corpus", cell_id),
            arrival_trace_sha256=family.trace_sha256,
            request_ids_sha256=binding_sha256("request-ids", cell_id),
            sampling_profile_sha256=family.sampling_sha256,
            model_lock_sha256=binding_sha256("model-lock", cell_id),
            patched_sglang_tree=PINNED_SGLANG_TREE,
            run_nonce_sha256=binding_sha256("run-nonce", cell_id),
            topology_sha256=binding_sha256("topology", cell_id),
            experiment_budget_sha256=binding_sha256("budget-plan", cell_id),
            physical_gpu_uuids=(f"GPU-family-test-{cell_id}",),
            terminal_receipt_sha256s=(binding_sha256("terminal", cell_id),),
            hardware_receipt_sha256=binding_sha256("hardware", cell_id),
            budget_observation_sha256=binding_sha256("budget-observation", cell_id),
        )
        for cell_id in sorted(pilot_activation.activated_cell_ids)
    )

    return ConfirmationFamilyPowerReductionArtifact(
        schema_version=2,
        plan=plan,
        inventory_sha256=content_sha256({"inventory": "family-test"}),
        inventory_source_receipt_sha256=content_sha256(
            {"inventory": "family-test-source"}
        ),
        fixed_instance_gpu_count=2,
        inventory_host_id="family-test-host",
        raw_evidence_manifest_sha256=evidence_sha256,
        terminal_receipt_sha256s=tuple(
            sorted(
                digest
                for binding in run_bindings
                for digest in binding.terminal_receipt_sha256s
            )
        ),
        hardware_receipt_sha256s=tuple(
            sorted(binding.hardware_receipt_sha256 for binding in run_bindings)
        ),
        budget_observation_sha256s=tuple(
            sorted(binding.budget_observation_sha256 for binding in run_bindings)
        ),
        run_bindings=run_bindings,
        reducer_protocol_sha256=(CONFIRMATION_FAMILY_POWER_REDUCER_PROTOCOL_SHA256),
        data_source="excluded_pilots_only",
        confirmation_data_visible=False,
    )


def _family_power_reduction(
    registry: ExperimentRegistry,
    family,
    pilot_activation,
    *,
    selected_final_blocks: int,
):
    sizing = _power_sizing(family)
    if selected_final_blocks == 20:
        sizing = replace(
            sizing,
            selected_final_blocks=20,
            power_grid=tuple(
                replace(cell, power=0.79 if cell.final_blocks < 20 else 0.80)
                for cell in sizing.power_grid
            ),
        )
    else:
        assert selected_final_blocks == 12
        assert sizing.selected_final_blocks == 12
    return _family_power_reduction_from_sizing(
        registry,
        family,
        pilot_activation,
        sizing,
        evidence_label=f"selected-{selected_final_blocks}",
    )


def _target_ids(
    registry: ExperimentRegistry, activated_cell_ids: tuple[str, ...]
) -> set[str]:
    cells = {cell.cell_id: cell for cell in registry.cells}
    return {
        cell_id
        for cell_id in activated_cell_ids
        if cells[cell_id].identity.method == "target_only"
        and serving_cell_rejection_reason(cells[cell_id]) is None
    }


def _budgets_for(
    registry: ExperimentRegistry,
    cell_ids: set[str],
    *,
    fixed_instance_gpu_count: int,
):
    cells = {cell.cell_id: cell for cell in registry.cells}
    return {
        cell_id: _budget(
            cells[cell_id],
            fixed_instance_gpu_count=fixed_instance_gpu_count,
        )
        for cell_id in cell_ids
    }


def _planned_cell_ids(plan: GpuDispatchPlan) -> set[str]:
    return {
        assignment.work_item.item_id
        for wave in plan.waves
        for assignment in wave.assignments
    }


def _diagnostic_budget_bindings(
    items: tuple[PoolWorkItem, ...],
) -> dict[str, str]:
    return {
        item.item_id: content_sha256(
            {"diagnostic_budget_for": item.item_id, "claim": item.claim.sha256}
        )
        for item in items
    }


def _e3a_execution_context(
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    envelope: InterferenceEnvelope,
) -> GpuDispatchPlanningContext:
    cells = tuple(
        cell
        for cell in registry.cells_for("E3a")
        if GpuPoolScheduler._dispatchable(cell)
    )
    budgets = tuple(
        sorted(
            (
                _budget(
                    cell,
                    fixed_instance_gpu_count=len(inventory.devices),
                )
                for cell in cells
            ),
            key=lambda budget: budget.cell_id,
        )
    )
    receipts = _receipts_through(registry, "preflight")
    activation = materialize_registry_stage_activation(
        registry,
        experiment="E3a",
        dependency_receipts=receipts,
        runtime_sha256=content_sha256("e3a-scheduler-runtime"),
        split_sha256=content_sha256("e3a-scheduler-split"),
    )
    return GpuDispatchPlanningContext(
        registry=registry,
        inventory=inventory,
        interference_envelope=envelope,
        budgets=budgets,
        receipts=receipts,
        activation_artifact=activation,
        port_start=31_000,
        port_end=31_999,
        seed=20260811,
    )


async def _execute_planning_dispatch_for_engine_test(
    plan: GpuDispatchPlan,
    *,
    execution_context: GpuDispatchPlanningContext,
    runner,
    resume_receipt: DispatchScheduleReceipt | None = None,
    stop_after_wave_index: int | None = None,
) -> DispatchScheduleReceipt:
    """Exercise the post-authority engine without weakening production gates.

    Formal execution requires path-bound budget/completion authorities that the
    scheduler unit tests intentionally do not fabricate.  Replaying the exact
    planning context here preserves plan-forgery coverage while a local patch
    bypasses only the production context type gate for the duration of one
    downstream state-machine test.
    """

    gpu_pool_module.validate_dispatch_plan_for_planning(
        plan,
        planning_context=execution_context,
    )

    def validate_replayed_plan(candidate, *, execution_context: object) -> None:
        if execution_context is not execution_context_value:
            raise AssertionError("engine test changed its planning context")
        gpu_pool_module.validate_dispatch_plan_for_planning(
            candidate,
            planning_context=execution_context_value,
        )

    execution_context_value = execution_context
    with patch.object(
        gpu_pool_module,
        "validate_dispatch_plan_for_execution",
        validate_replayed_plan,
    ):
        return await execute_dispatch_plan(
            plan,
            execution_context=execution_context,  # type: ignore[arg-type]
            runner=runner,
            resume_receipt=resume_receipt,
            stop_after_wave_index=stop_after_wave_index,
        )


def _validate_planning_resume_for_engine_test(
    plan: GpuDispatchPlan,
    receipt: DispatchScheduleReceipt,
    *,
    execution_context: GpuDispatchPlanningContext,
) -> None:
    """Reach receipt-structure gates after an exact planning replay."""

    def validate_replayed_plan(candidate, *, execution_context: object) -> None:
        if execution_context is not execution_context_value:
            raise AssertionError("resume test changed its planning context")
        gpu_pool_module.validate_dispatch_plan_for_planning(
            candidate,
            planning_context=execution_context_value,
        )

    execution_context_value = execution_context
    with patch.object(
        gpu_pool_module,
        "validate_dispatch_plan_for_execution",
        validate_replayed_plan,
    ):
        validate_dispatch_resume(
            plan,
            receipt,
            execution_context=execution_context,  # type: ignore[arg-type]
        )


def _restore_planning_schedule_receipt_for_engine_test(
    value: object,
    *,
    sidecar: ArtifactSidecar,
    plan: GpuDispatchPlan,
    execution_context: GpuDispatchPlanningContext,
) -> DispatchScheduleReceipt:
    """Round-trip a receipt while retaining exact planning-plan replay."""

    def validate_replayed_plan(candidate, *, execution_context: object) -> None:
        if execution_context is not execution_context_value:
            raise AssertionError("receipt test changed its planning context")
        gpu_pool_module.validate_dispatch_plan_for_planning(
            candidate,
            planning_context=execution_context_value,
        )

    execution_context_value = execution_context
    with patch.object(
        gpu_pool_module,
        "validate_dispatch_plan_for_execution",
        validate_replayed_plan,
    ):
        return DispatchScheduleReceipt.from_dict(
            value,
            sidecar=sidecar,
            plan=plan,
            execution_context=execution_context,  # type: ignore[arg-type]
        )


def test_generic_stage_activation_is_replayed_before_scheduling(
    registry: ExperimentRegistry,
) -> None:
    context = _e3a_execution_context(
        registry,
        _inventory(2),
        InterferenceEnvelope.serial(
            source_receipt_sha256=content_sha256("generic-activation-replay")
        ),
    )
    artifact = context.activation_artifact
    assert isinstance(artifact, RegistryStageActivationArtifact)
    blocked_index = next(
        index
        for index, row in enumerate(artifact.dispositions)
        if row.status is RegistryStageDispositionStatus.BLOCKED
    )
    edited_rows = list(artifact.dispositions)
    edited_rows[blocked_index] = replace(
        edited_rows[blocked_index],
        reason_code="caller_edited_dispatch_disposition",
    )
    edited = replace(artifact, dispositions=tuple(edited_rows))

    with pytest.raises(ValueError, match="exact reducer-generated"):
        replace(context, activation_artifact=edited).issue_plan()


def test_bare_completed_cells_are_planning_only_and_cannot_skip_the_runner(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(2)
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("completed-planning-only")
    )
    base = _e3a_execution_context(registry, inventory, envelope)
    completed_cell_id = base.issue_plan().waves[0].assignments[0].work_item.item_id
    planning = GpuDispatchPlanningContext(
        registry=registry,
        inventory=inventory,
        interference_envelope=envelope,
        budgets=tuple(
            budget for budget in base.budgets if budget.cell_id != completed_cell_id
        ),
        receipts=base.receipts,
        completed_cell_ids=(completed_cell_id,),
        activation_artifact=base.activation_artifact,
        port_start=base.port_start,
        port_end=base.port_end,
        seed=base.seed,
    )
    plan = planning.issue_plan()
    assert (
        GpuDispatchPlan.from_dict(
            plan.to_dict(),
            planning_context=planning,
        )
        == plan
    )
    calls: list[str] = []

    async def runner(assignment):
        calls.append(assignment.assignment_id)
        return content_sha256("must-not-run-from-planning-context")

    with pytest.raises(TypeError, match="GpuDispatchExecutionContext"):
        asyncio.run(
            execute_dispatch_plan(
                plan,
                execution_context=planning,  # type: ignore[arg-type]
                runner=runner,
            )
        )
    with pytest.raises(TypeError, match="budget_plan"):
        GpuDispatchExecutionContext(
            registry=registry,
            inventory=inventory,
            interference_envelope=envelope,
            budgets=planning.budgets,
            receipts=planning.receipts,
            completed_cell_ids=planning.completed_cell_ids,
            activation_artifact=planning.activation_artifact,
            port_start=planning.port_start,
            port_end=planning.port_end,
            seed=planning.seed,
        )
    assert calls == []


def test_one_confirmation_family_materializes_only_its_pilots(
    registry: ExperimentRegistry,
) -> None:
    family = _e5_family(registry, concurrency=1)
    pilots = materialize_confirmation_pilots(registry, family)
    expected = _target_ids(registry, pilots.activated_cell_ids)
    assert len(pilots.activated_cell_ids) == len(PILOT_BLOCKS) * len(CORE_METHODS)
    assert len(expected) == len(PILOT_BLOCKS)
    inventory = _inventory(4)
    scheduler = _scheduler(
        registry,
        inventory,
        InterferenceEnvelope.serial(
            source_receipt_sha256=content_sha256("family-pilot-serial")
        ),
    )

    plan = scheduler.schedule(
        budgets_by_cell_id=_budgets_for(
            registry,
            expected,
            fixed_instance_gpu_count=len(inventory.devices),
        ),
        receipts=_receipts_through(registry, "E1a"),
        family_activations=(pilots,),
    )

    assert _planned_cell_ids(plan) == expected
    assert plan.scientific_budget_bound
    assert set(dict(plan.budget_sha256_by_cell)) == expected
    assert all(
        assignment.work_item.cell.identity.method == "target_only"
        for wave in plan.waves
        for assignment in wave.assignments
    )


@pytest.mark.parametrize("selected_final_blocks", (12, 20))
def test_family_final_prefix_is_exact_and_unrelated_family_does_not_block(
    registry: ExperimentRegistry,
    selected_final_blocks: int,
) -> None:
    family = _e5_family(registry, concurrency=1)
    pilots = materialize_confirmation_pilots(registry, family)
    power = _family_power_reduction(
        registry,
        family,
        pilots,
        selected_final_blocks=selected_final_blocks,
    )
    final = materialize_confirmation_prefix(
        registry,
        family=family,
        reduction=power,
        pilot_activation=pilots,
    )
    unrelated_family = _e5_family(registry, concurrency=2)
    unrelated_pilots = materialize_confirmation_pilots(registry, unrelated_family)
    expected_final = _target_ids(registry, final.activated_cell_ids)
    expected_unrelated_pilots = _target_ids(
        registry, unrelated_pilots.activated_cell_ids
    )
    expected = expected_final | expected_unrelated_pilots
    inventory = _inventory(8)
    scheduler = _scheduler(
        registry,
        inventory,
        InterferenceEnvelope.serial(
            source_receipt_sha256=content_sha256(
                {"family-final": selected_final_blocks}
            )
        ),
    )

    plan = scheduler.schedule(
        budgets_by_cell_id=_budgets_for(
            registry,
            expected,
            fixed_instance_gpu_count=len(inventory.devices),
        ),
        receipts=_receipts_through(registry, "E1a"),
        completed_cell_ids=pilots.activated_cell_ids,
        family_activations=(pilots, final, unrelated_pilots),
        family_power_reductions=(power,),
    )

    assert len(expected_final) == selected_final_blocks
    assert _planned_cell_ids(plan) == expected
    final_blocks = {
        assignment.work_item.cell.identity.block
        for wave in plan.waves
        for assignment in wave.assignments
        if assignment.work_item.item_id in expected_final
    }
    assert final_blocks == set(FINAL_BLOCKS[:selected_final_blocks])


def test_underpowered_family_activates_zero_final_cells(
    registry: ExperimentRegistry,
) -> None:
    family = _e5_family(registry, concurrency=1)
    pilots = materialize_confirmation_pilots(registry, family)
    sizing = _power_sizing(family)
    underpowered_sizing = replace(
        sizing,
        status="UNDERPOWERED",
        selected_final_blocks=None,
        power_grid=tuple(replace(cell, power=0.79) for cell in sizing.power_grid),
    )
    power = _family_power_reduction_from_sizing(
        registry,
        family,
        pilots,
        underpowered_sizing,
        evidence_label="underpowered-family-evidence",
    )
    final = materialize_confirmation_prefix(
        registry,
        family=family,
        reduction=power,
        pilot_activation=pilots,
    )
    assert power.status == "UNDERPOWERED"
    assert final.activated_cell_ids == ()
    scheduler = _scheduler(
        registry,
        _inventory(2),
        InterferenceEnvelope.serial(
            source_receipt_sha256=content_sha256("underpowered-serial")
        ),
    )

    dispatch = scheduler.schedule(
        budgets_by_cell_id={},
        receipts=_receipts_through(registry, "E1a"),
        completed_cell_ids=pilots.activated_cell_ids,
        family_activations=(pilots, final),
        family_power_reductions=(power,),
    )

    assert dispatch.waves == ()


def test_confirmation_family_artifacts_reject_missing_duplicate_and_forged_inputs(
    registry: ExperimentRegistry,
) -> None:
    family = _e5_family(registry, concurrency=1)
    pilots = materialize_confirmation_pilots(registry, family)
    power = _family_power_reduction(registry, family, pilots, selected_final_blocks=12)
    final = materialize_confirmation_prefix(
        registry,
        family=family,
        reduction=power,
        pilot_activation=pilots,
    )
    other_family = _e5_family(registry, concurrency=2)
    other_pilots = materialize_confirmation_pilots(registry, other_family)
    other_power = _family_power_reduction(
        registry, other_family, other_pilots, selected_final_blocks=12
    )
    scheduler = _scheduler(
        registry,
        _inventory(2),
        InterferenceEnvelope.serial(
            source_receipt_sha256=content_sha256("family-rejection-serial")
        ),
    )
    receipts = _receipts_through(registry, "E1a")

    with pytest.raises(ValueError, match="reducer-generated family activations"):
        scheduler.schedule(budgets_by_cell_id={}, receipts=receipts)
    with pytest.raises(ValueError, match="matching pilot artifact"):
        scheduler.schedule(
            budgets_by_cell_id={},
            receipts=receipts,
            family_activations=(final,),
            family_power_reductions=(power,),
        )
    with pytest.raises(ValueError, match="exact family power reduction"):
        scheduler.schedule(
            budgets_by_cell_id={},
            receipts=receipts,
            completed_cell_ids=pilots.activated_cell_ids,
            family_activations=(pilots, final),
        )
    with pytest.raises(ValueError, match="duplicate confirmation family activation"):
        scheduler.schedule(
            budgets_by_cell_id={},
            receipts=receipts,
            family_activations=(pilots, pilots),
        )
    with pytest.raises(
        ValueError, match="duplicate confirmation family power reduction"
    ):
        scheduler.schedule(
            budgets_by_cell_id={},
            receipts=receipts,
            completed_cell_ids=pilots.activated_cell_ids,
            family_activations=(pilots, final),
            family_power_reductions=(power, power),
        )
    with pytest.raises(ValueError, match="prior completion"):
        scheduler.schedule(
            budgets_by_cell_id={},
            receipts=receipts,
            family_activations=(pilots, final),
            family_power_reductions=(power,),
        )
    forged = replace(final, power_plan_sha256="0" * 64)
    with pytest.raises(
        ValueError, match="not reducer-generated from its power reduction"
    ):
        scheduler.schedule(
            budgets_by_cell_id={},
            receipts=receipts,
            completed_cell_ids=pilots.activated_cell_ids,
            family_activations=(pilots, forged),
            family_power_reductions=(power,),
        )
    forged_power = replace(
        power,
        plan=replace(power.plan, reason_code="caller_forged_power_decision"),
    )
    with pytest.raises(ValueError, match="not reducer-generated from its power"):
        scheduler.schedule(
            budgets_by_cell_id={},
            receipts=receipts,
            completed_cell_ids=pilots.activated_cell_ids,
            family_activations=(pilots, final),
            family_power_reductions=(forged_power,),
        )
    with pytest.raises(ValueError, match="exact family power reduction"):
        scheduler.schedule(
            budgets_by_cell_id={},
            receipts=receipts,
            completed_cell_ids=pilots.activated_cell_ids,
            family_activations=(pilots, final),
            family_power_reductions=(other_power,),
        )
    with pytest.raises(ValueError, match="do not match the ready stage"):
        scheduler.schedule(
            budgets_by_cell_id={},
            receipts=_receipts_through(registry, "preflight"),
            family_activations=(pilots,),
        )


@pytest.mark.parametrize("size", (1, 2, 4, 8, 16))
def test_required_pool_sizes_fill_one_deterministic_wave(
    registry: ExperimentRegistry, size: int
) -> None:
    inventory = _inventory(size)
    items = tuple(
        registry_pool_work_item(cell, estimated_duration_seconds=1.0)
        for cell in _target_tuning_cells(registry, size)
    )
    envelope = _envelope_for(
        inventory,
        items[0],
        tuple(range(2, size + 1)),
    )
    scheduler = _scheduler(registry, inventory, envelope)
    receipts_sha256 = content_sha256({"receipts": "none"})

    plan = scheduler.schedule_work_items(
        tuple(reversed(items)),
        receipts_sha256=receipts_sha256,
        budget_sha256_by_cell=_diagnostic_budget_bindings(items),
    )
    repeated = scheduler.schedule_work_items(
        items,
        receipts_sha256=receipts_sha256,
        budget_sha256_by_cell=_diagnostic_budget_bindings(items),
    )

    assert supported_pool_size(size)
    assert len(plan.waves) == 1
    assert len(plan.waves[0].assignments) == size
    assert plan.sha256 == repeated.sha256
    assert plan.estimated_wall_seconds == 1.0
    assert plan.estimated_gpu_seconds == float(size)
    assert plan.estimated_gpu_hours == pytest.approx(size / 3600.0)
    assert {
        uuid
        for assignment in plan.waves[0].assignments
        for uuid in assignment.gpu_uuids
    } == {device.uuid for device in inventory.devices}
    assert (
        len(
            {
                port
                for assignment in plan.waves[0].assignments
                for port in assignment.ports
            }
        )
        == size
    )


def test_registered_interference_calibration_freezes_isolated_and_paired_waves(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(2)
    activation = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=content_sha256("calibration-runtime"),
        split_sha256=content_sha256("calibration-split"),
    )
    bootstrap = materialize_interference_calibration_bootstrap_authority(
        registry,
        inventory,
        activation,
    )
    cells_by_id = {cell.cell_id: cell for cell in registry.cells}
    items = tuple(
        registry_pool_work_item(cells_by_id[cell_id], estimated_duration_seconds=1.0)
        for cell_id in activation.activated_cell_ids
    )

    plan = _scheduler(
        registry,
        inventory,
        bootstrap.bootstrap_envelope,
    ).schedule_work_items(
        items,
        receipts_sha256=content_sha256("calibration-receipts"),
        budget_sha256_by_cell=_diagnostic_budget_bindings(items),
    )

    assert sorted(len(wave.assignments) for wave in plan.waves) == [1, 1, 1, 1, 2, 2]
    paired = tuple(wave for wave in plan.waves if len(wave.assignments) == 2)
    assert len(paired) == 2
    assert {
        tuple(
            sorted(
                assignment.work_item.cell.identity.block
                for assignment in wave.assignments
            )
        )
        for wave in paired
    } == {(0, 0), (1, 1)}
    assert all(
        str(assignment.work_item.cell.identity.variant).startswith("concurrent_slot_")
        for wave in paired
        for assignment in wave.assignments
    )
    assert all(
        len(wave.assignments) == 1
        for wave in plan.waves
        if any(
            str(assignment.work_item.cell.identity.variant).startswith("isolated_slot_")
            for assignment in wave.assignments
        )
    )


@pytest.mark.parametrize("size", (1, 2, 4, 8, 16))
def test_digest_only_runner_cannot_complete_required_pool_sizes(
    registry: ExperimentRegistry,
    size: int,
) -> None:
    inventory = _inventory(size)
    first = registry_pool_work_item(
        _target_tuning_cells(registry, 1)[0], estimated_duration_seconds=1.0
    )
    envelope = _envelope_for(
        inventory,
        first,
        tuple(range(2, size + 1)),
    )
    context = _e3a_execution_context(registry, inventory, envelope)
    plan = context.issue_plan()
    calls: list[str] = []

    async def runner(assignment):
        calls.append(assignment.assignment_id)
        await asyncio.sleep(0)
        return content_sha256({"terminal": assignment.assignment_id})

    receipt = asyncio.run(
        _execute_planning_dispatch_for_engine_test(
            plan,
            execution_context=context,
            runner=runner,
        )
    )

    assert receipt.phase is DispatchExecutionPhase.FAILED
    assert set(calls) == {
        assignment.assignment_id for assignment in plan.waves[0].assignments
    }
    assert all(
        row.status is AssignmentExecutionStatus.FAILED
        and row.terminal_receipt_sha256 is None
        and row.terminal_binding is None
        and row.failure_sha256 is not None
        for row in receipt.wave_receipts[0].assignment_receipts
    )
    assert receipt.fixed_instance_gpu_count == size
    assert receipt.fixed_instance_actual_billed_gpu_ns == (
        sum(finish - start for start, finish in receipt.active_intervals_monotonic_ns)
        * size
    )


def test_scheduler_accepts_positive_non_gate_pool_size(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(3)
    items = tuple(
        registry_pool_work_item(cell, estimated_duration_seconds=1.0)
        for cell in _target_tuning_cells(registry, 3)
    )
    scheduler = _scheduler(
        registry,
        inventory,
        _envelope_for(inventory, items[0], (2, 3)),
    )

    plan = scheduler.schedule_work_items(
        items, receipts_sha256=content_sha256("receipts")
    )

    assert not supported_pool_size(3)
    assert len(plan.waves) == 1
    assert len(plan.waves[0].assignments) == 3


def test_two_way_calibration_never_authorizes_eight_way_execution(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(8)
    items = tuple(
        registry_pool_work_item(cell, estimated_duration_seconds=1.0)
        for cell in _target_tuning_cells(registry, 8)
    )
    scheduler = _scheduler(
        registry,
        inventory,
        _envelope_for(inventory, items[0], (2,)),
    )

    plan = scheduler.schedule_work_items(
        items, receipts_sha256=content_sha256("receipts")
    )

    assert len(plan.waves) == 4
    assert all(len(wave.assignments) == 2 for wave in plan.waves)
    assert max(len(wave.assignments) for wave in plan.waves) == 2


def test_topology_aware_tp_gang_is_atomic_and_cannot_span_groups(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(4, paired_topology=True)
    preflight = next(
        cell
        for cell in registry.cells_for("preflight")
        if cell.identity.method == "target_only"
    )
    # Exercise the generic TP placement primitive without treating the blocked
    # COMPILE declaration as executable authority.
    topology_cell = replace(
        preflight,
        resources=replace(
            preflight.resources,
            workload_class=WorkloadClass.CORRECTNESS,
        ),
        identity=replace(
            preflight.identity,
            experiment="E4",
            task="topology_placement_primitive",
            backend="NONE",
            context=4096,
            concurrency=1,
            topology="tp2_dp1",
        ),
    )
    topology_registry = replace(
        registry,
        cells=tuple(
            topology_cell if cell.cell_id == preflight.cell_id else cell
            for cell in registry.cells
        ),
    )
    item = registry_pool_work_item(topology_cell, estimated_duration_seconds=1.0)
    scheduler = _scheduler(
        topology_registry,
        inventory,
        InterferenceEnvelope.serial(source_receipt_sha256=content_sha256("serial")),
    )

    plan = scheduler.schedule_work_items(
        (item,), receipts_sha256=content_sha256("receipts")
    )
    assignment = plan.waves[0].assignments[0]

    assert item.claim.gang_shape == GangShape(2, 1)
    assert assignment.rank_groups == (assignment.gpu_uuids,)
    assert set(assignment.gpu_uuids) in (
        {"GPU-000", "GPU-001"},
        {"GPU-002", "GPU-003"},
    )

    impossible = replace(
        item,
        claim=replace(
            item.claim,
            exact_gpu_uuids=("GPU-001", "GPU-002"),
        ),
    )
    with pytest.raises(CapabilityRejectionError, match="no ready capability"):
        scheduler.schedule_work_items(
            (impossible,), receipts_sha256=content_sha256("receipts")
        )

    one_gpu = _inventory(1)
    with pytest.raises(CapabilityRejectionError, match="no ready capability"):
        _scheduler(
            topology_registry,
            one_gpu,
            InterferenceEnvelope.serial(
                source_receipt_sha256=content_sha256("serial-one")
            ),
        ).schedule_work_items((item,), receipts_sha256=content_sha256("receipts"))


def test_readiness_capability_and_foreign_claims_fail_closed(
    registry: ExperimentRegistry,
) -> None:
    cell = _target_tuning_cells(registry, 1)[0]
    item = registry_pool_work_item(cell, estimated_duration_seconds=1.0)
    serial = InterferenceEnvelope.serial(source_receipt_sha256=content_sha256("serial"))

    reserved_scheduler = _scheduler(
        registry, _inventory(1, reserved=frozenset({0})), serial
    )
    with pytest.raises(CapabilityRejectionError, match="no ready capability"):
        reserved_scheduler.schedule_work_items(
            (item,), receipts_sha256=content_sha256("receipts")
        )

    capability_item = replace(
        item,
        claim=replace(
            item.claim,
            homogeneous=HomogeneousDeviceConstraint(
                model="H100-SXM",
                minimum_memory_bytes=81 * 1024**3,
                minimum_compute_capability=(9, 0),
                allowed_peer_access_classes=("NVSwitch",),
            ),
        ),
    )
    with pytest.raises(CapabilityRejectionError, match="no ready capability"):
        _scheduler(registry, _inventory(1), serial).schedule_work_items(
            (capability_item,), receipts_sha256=content_sha256("receipts")
        )

    forged_cell = replace(cell, reason="caller changed registered evidence state")
    with pytest.raises(ValueError, match="registered cell declaration"):
        _scheduler(registry, _inventory(1), serial).schedule_work_items(
            (replace(item, cell=forged_cell),),
            receipts_sha256=content_sha256("receipts"),
        )


def test_affinity_groups_stay_on_one_gpu_and_balance_across_pool(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(8)
    cells = _target_tuning_cells(registry, 16)
    items = tuple(
        replace(
            registry_pool_work_item(cell, estimated_duration_seconds=1.0),
            affinity_key=content_sha256({"group": index // 2}),
        )
        for index, cell in enumerate(cells)
    )
    serial = InterferenceEnvelope.serial(source_receipt_sha256=content_sha256("serial"))
    plan = _scheduler(registry, inventory, serial).schedule_work_items(
        items, receipts_sha256=content_sha256("receipts")
    )
    assignments = tuple(
        assignment for wave in plan.waves for assignment in wave.assignments
    )
    by_affinity: dict[str, set[str]] = {}
    for assignment in assignments:
        key = assignment.work_item.affinity_key
        assert key is not None
        by_affinity.setdefault(key, set()).update(assignment.gpu_uuids)

    assert all(len(uuids) == 1 for uuids in by_affinity.values())
    counts = Counter(assignment.gpu_uuids[0] for assignment in assignments)
    assert set(counts) == {device.uuid for device in inventory.devices}
    assert max(counts.values()) - min(counts.values()) == 0


def test_nonserving_cells_without_terminal_contracts_never_dispatch(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(4)
    tuning = registry_pool_work_item(
        _target_tuning_cells(registry, 1)[0], estimated_duration_seconds=1.0
    )
    preflight = next(
        registry_pool_work_item(cell, estimated_duration_seconds=1.0)
        for cell in registry.cells_for("preflight")
        if cell.identity.method == "target_only"
    )
    download = next(
        registry_pool_work_item(cell, estimated_duration_seconds=1.0)
        for cell in registry.cells_for("E6")
        if cell.resources.workload_class is WorkloadClass.DOWNLOAD
    )
    assert release_dispatch_rejection_reason(preflight.cell) == (
        RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE
    )
    assert release_dispatch_rejection_reason(download.cell) == (
        RELEASE_DOWNLOAD_ASSIGNMENT_CONTRACT_UNAVAILABLE
    )
    rules = tuple(
        sorted(
            (
                InterferenceRule.for_claim(
                    device=inventory.devices[0],
                    claim=tuning.claim,
                    simultaneous_jobs=2,
                    evidence_sha256=content_sha256("tuning-two"),
                ),
            ),
            key=lambda row: row.key,
        )
    )
    envelope = InterferenceEnvelope(
        schema_version=1,
        rules=rules,
        source_receipt_sha256=content_sha256("envelope"),
    )
    for nonserving in (preflight, download):
        with pytest.raises(ValueError, match="non-executable"):
            _scheduler(registry, inventory, envelope).schedule_work_items(
                (tuning, nonserving),
                receipts_sha256=content_sha256("receipts"),
            )

    blocked_profile = next(
        registry_pool_work_item(cell, estimated_duration_seconds=1.0)
        for cell in registry.cells_for("E4")
        if cell.resources.workload_class is WorkloadClass.PROFILE
    )
    with pytest.raises(ValueError, match="non-executable"):
        _scheduler(registry, inventory, envelope).schedule_work_items(
            (blocked_profile,), receipts_sha256=content_sha256("receipts")
        )


def test_ports_cache_evidence_and_costs_are_exact(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(2)
    cells = _target_tuning_cells(registry, 2)
    items = (
        registry_pool_work_item(cells[0], estimated_duration_seconds=10.0),
        registry_pool_work_item(cells[1], estimated_duration_seconds=20.0),
    )
    scheduler = _scheduler(
        registry,
        inventory,
        _envelope_for(inventory, items[0], (2,)),
    )
    plan = scheduler.schedule_work_items(
        items, receipts_sha256=content_sha256("receipts")
    )
    wave = plan.waves[0]

    assert wave.estimated_wall_seconds == 20.0
    assert wave.estimated_gpu_seconds == 30.0
    assert plan.estimated_gpu_hours == pytest.approx(30.0 / 3600.0)
    assert len({row.ports for row in wave.assignments}) == 2
    assert len({row.work_item.claim.cache_root for row in wave.assignments}) == 2
    assert len({row.work_item.claim.evidence_root for row in wave.assignments}) == 2


def test_forged_static_plan_is_rejected_before_runner_or_contextual_parse(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("forged-static-serial")
    )
    context = _e3a_execution_context(registry, inventory, envelope)
    valid = context.issue_plan()
    static_cell = next(
        cell
        for cell in registry.cells_for("E3a")
        if cell.identity.method == "static"
        and serving_cell_rejection_reason(cell) is None
    )
    static_item = registry_pool_work_item(static_cell, estimated_duration_seconds=1.0)
    original = valid.waves[0].assignments[0]
    forged_assignment = GpuAssignment(
        work_item=static_item,
        gpu_uuids=original.gpu_uuids,
        rank_groups=original.rank_groups,
        ports=original.ports,
    )
    forged_wave = GpuDispatchWave(
        wave_index=0,
        assignments=(forged_assignment,),
        interference_envelope_sha256=envelope.sha256,
    )
    forged = replace(
        valid,
        waves=(forged_wave,),
        budget_sha256_by_cell=(
            (static_item.item_id, content_sha256("forged-static-budget")),
        ),
    )
    calls: list[str] = []

    async def runner(assignment):
        calls.append(assignment.assignment_id)
        return content_sha256("must-not-run")

    with pytest.raises(ValueError, match="not the exact plan"):
        GpuDispatchPlan.from_dict(
            forged.to_dict(),
            planning_context=context,
        )
    with pytest.raises(ValueError, match="not the exact plan"):
        asyncio.run(
            _execute_planning_dispatch_for_engine_test(
                forged,
                execution_context=context,
                runner=runner,
            )
        )
    assert calls == []


def test_canonical_e5_target_cannot_bypass_ready_stage_dag(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("forged-e5-serial")
    )
    context = _e3a_execution_context(registry, inventory, envelope)
    e5_cell = next(
        cell
        for cell in registry.cells_for("E5")
        if GpuPoolScheduler._dispatchable(cell)
    )
    item = registry_pool_work_item(e5_cell, estimated_duration_seconds=1.0)
    assignment = GpuAssignment(
        work_item=item,
        gpu_uuids=(inventory.devices[0].uuid,),
        rank_groups=((inventory.devices[0].uuid,),),
        ports=(31_000,),
    )
    wave = GpuDispatchWave(
        wave_index=0,
        assignments=(assignment,),
        interference_envelope_sha256=envelope.sha256,
    )
    forged = GpuDispatchPlan(
        schema_version=1,
        registry_sha256=registry.sha256,
        inventory_sha256=inventory.sha256,
        receipts_sha256=content_sha256("forged-ready-stage-receipts"),
        interference_envelope_sha256=envelope.sha256,
        budget_sha256_by_cell=((e5_cell.cell_id, content_sha256("forged-e5-budget")),),
        seed=20260811,
        waves=(wave,),
        completed_cell_ids=(),
    )
    calls: list[str] = []

    async def runner(candidate):
        calls.append(candidate.assignment_id)
        return content_sha256("must-not-run")

    with pytest.raises(ValueError, match="not the exact plan"):
        asyncio.run(
            _execute_planning_dispatch_for_engine_test(
                forged,
                execution_context=context,
                runner=runner,
            )
        )
    assert calls == []


def test_forged_noncanonical_claim_is_rejected_before_runner(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("forged-claim-serial")
    )
    context = _e3a_execution_context(registry, inventory, envelope)
    valid = context.issue_plan()
    original = valid.waves[0].assignments[0]
    forged_item = replace(
        original.work_item,
        claim=replace(
            original.work_item.claim,
            homogeneous=HomogeneousDeviceConstraint(model="H100-SXM"),
        ),
    )
    forged_assignment = replace(original, work_item=forged_item)
    forged_wave = replace(valid.waves[0], assignments=(forged_assignment,))
    forged = replace(valid, waves=(forged_wave, *valid.waves[1:]))
    calls: list[str] = []

    async def runner(assignment):
        calls.append(assignment.assignment_id)
        return content_sha256("must-not-run")

    with pytest.raises(ValueError, match="not the exact plan"):
        asyncio.run(
            _execute_planning_dispatch_for_engine_test(
                forged,
                execution_context=context,
                runner=runner,
            )
        )
    assert calls == []


def test_forged_unready_capability_assignment_is_rejected_before_runner(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(2, reserved=frozenset({1}))
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("forged-capability-serial")
    )
    context = _e3a_execution_context(registry, inventory, envelope)
    valid = context.issue_plan()
    original = valid.waves[0].assignments[0]
    forged_assignment = replace(
        original,
        gpu_uuids=("GPU-001",),
        rank_groups=(("GPU-001",),),
    )
    forged_wave = replace(valid.waves[0], assignments=(forged_assignment,))
    forged = replace(valid, waves=(forged_wave, *valid.waves[1:]))
    calls: list[str] = []

    async def runner(assignment):
        calls.append(assignment.assignment_id)
        return content_sha256("must-not-run")

    with pytest.raises(ValueError, match="not the exact plan"):
        asyncio.run(
            _execute_planning_dispatch_for_engine_test(
                forged,
                execution_context=context,
                runner=runner,
            )
        )
    assert calls == []


def test_forged_concurrent_wave_recomputes_interference_before_runner(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(2)
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("forged-wave-serial")
    )
    context = _e3a_execution_context(registry, inventory, envelope)
    valid = context.issue_plan()
    assert len(valid.waves) >= 2
    first = valid.waves[0].assignments[0]
    second = replace(
        valid.waves[1].assignments[0],
        ports=(first.ports[0] + 1,),
    )
    forged_wave = GpuDispatchWave(
        wave_index=0,
        assignments=(first, second),
        interference_envelope_sha256=envelope.sha256,
    )
    remaining = tuple(
        replace(wave, wave_index=index)
        for index, wave in enumerate(valid.waves[2:], start=1)
    )
    forged = replace(valid, waves=(forged_wave, *remaining))
    calls: list[str] = []

    async def runner(assignment):
        calls.append(assignment.assignment_id)
        return content_sha256("must-not-run")

    with pytest.raises(ValueError, match="not the exact plan"):
        asyncio.run(
            _execute_planning_dispatch_for_engine_test(
                forged,
                execution_context=context,
                runner=runner,
            )
        )
    assert calls == []


def test_digest_siblings_and_rehashed_failure_resume_fail_closed(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(2)
    first_item = registry_pool_work_item(
        _target_tuning_cells(registry, 1)[0], estimated_duration_seconds=1.0
    )
    envelope = _envelope_for(inventory, first_item, (2,))
    context = _e3a_execution_context(registry, inventory, envelope)
    plan = context.issue_plan()
    assert len(plan.waves[0].assignments) == 2
    failed_id = plan.waves[0].assignments[0].assignment_id
    first_calls: list[str] = []

    async def first_runner(assignment):
        first_calls.append(assignment.assignment_id)
        if assignment.assignment_id == failed_id:
            raise RuntimeError("synthetic sibling failure")
        return content_sha256({"terminal": assignment.assignment_id})

    failed = asyncio.run(
        _execute_planning_dispatch_for_engine_test(
            plan,
            execution_context=context,
            runner=first_runner,
        )
    )
    assert failed.phase is DispatchExecutionPhase.FAILED
    assert not failed.wave_receipts[-1].partial_sibling_failure
    assert set(first_calls) == {
        assignment.assignment_id for assignment in plan.waves[0].assignments
    }
    assert {
        receipt.status for receipt in failed.wave_receipts[-1].assignment_receipts
    } == {AssignmentExecutionStatus.FAILED}
    assert all(
        row.terminal_receipt_sha256 is None and row.terminal_binding is None
        for row in failed.wave_receipts[-1].assignment_receipts
    )
    failed_attempt = next(
        row
        for row in failed.wave_receipts[-1].assignment_receipts
        if row.assignment_sha256 == failed_id
    )
    assert failed_attempt.attempt == 1
    assert failed_attempt.failure_sha256 is not None
    assert len(failed_attempt.attempt_intervals_monotonic_ns) == 1
    assert failed_attempt.attributed_gpu_ns >= 0
    assert failed_attempt.attributed_fixed_instance_gpu_ns == (
        failed_attempt.attributed_gpu_ns * len(inventory.devices)
    )

    resume_calls: list[str] = []

    async def resume_runner(assignment):
        resume_calls.append(assignment.assignment_id)
        return content_sha256({"terminal": assignment.assignment_id, "retry": 2})

    with pytest.raises(ValueError, match="append-only attempt/cost authority"):
        asyncio.run(
            _execute_planning_dispatch_for_engine_test(
                plan,
                execution_context=context,
                runner=resume_runner,
                resume_receipt=failed,
            )
        )
    assert resume_calls == []

    forged_rows = tuple(
        replace(
            row,
            attempt_intervals_monotonic_ns=((0, 0),),
            attributed_gpu_ns=0,
            attributed_fixed_instance_gpu_ns=0,
        )
        for row in failed.wave_receipts[-1].assignment_receipts
    )
    forged_wave = replace(
        failed.wave_receipts[-1],
        assignment_receipts=forged_rows,
        active_intervals_monotonic_ns=((0, 0),),
        fixed_instance_actual_billed_gpu_ns=0,
        per_assignment_attributed_gpu_ns=0,
        per_assignment_attributed_fixed_instance_gpu_ns=0,
    )
    forged = replace(
        failed,
        wave_receipts=(forged_wave,),
        active_intervals_monotonic_ns=((0, 0),),
        fixed_instance_actual_billed_gpu_ns=0,
        per_assignment_attributed_gpu_ns=0,
        per_assignment_attributed_fixed_instance_gpu_ns=0,
    )
    assert forged.sha256 != failed.sha256
    restored = _restore_planning_schedule_receipt_for_engine_test(
        json.loads(json.dumps(forged.to_dict())),
        sidecar=forged.sidecar(),
        plan=plan,
        execution_context=context,
    )
    assert restored == forged
    with pytest.raises(ValueError, match="append-only attempt/cost authority"):
        _validate_planning_resume_for_engine_test(
            plan,
            restored,
            execution_context=context,
        )

    other_inventory = _inventory(2, source="other-inventory")
    other_envelope = _envelope_for(other_inventory, first_item, (2,))
    foreign_context = _e3a_execution_context(registry, other_inventory, other_envelope)
    foreign_plan = foreign_context.issue_plan()
    with pytest.raises(ValueError, match="another dispatch plan"):
        _validate_planning_resume_for_engine_test(
            foreign_plan,
            failed,
            execution_context=foreign_context,
        )


def test_forged_inner_plan_wave_receipt_cannot_skip_runner(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("inner-receipt-forgery")
    )
    context = _e3a_execution_context(registry, inventory, envelope)
    plan = context.issue_plan()

    async def initial_runner(assignment):
        return content_sha256({"terminal": assignment.assignment_id})

    complete = asyncio.run(
        _execute_planning_dispatch_for_engine_test(
            plan,
            execution_context=context,
            runner=initial_runner,
        )
    )
    first_wave = complete.wave_receipts[0]
    forged_inner = replace(
        first_wave.assignment_receipts[0],
        plan_sha256=content_sha256("foreign-plan"),
        wave_sha256=content_sha256("foreign-wave"),
    )
    forged_wave = replace(
        first_wave,
        assignment_receipts=(
            forged_inner,
            *first_wave.assignment_receipts[1:],
        ),
    )
    forged = replace(
        complete,
        wave_receipts=(forged_wave, *complete.wave_receipts[1:]),
    )
    calls: list[str] = []

    async def resume_runner(assignment):
        calls.append(assignment.assignment_id)
        return content_sha256("must-not-run")

    with pytest.raises(ValueError, match="plan/wave mismatch"):
        asyncio.run(
            _execute_planning_dispatch_for_engine_test(
                plan,
                execution_context=context,
                runner=resume_runner,
                resume_receipt=forged,
            )
        )
    assert calls == []


def test_unbound_low_level_diagnostic_plan_cannot_execute(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    item = registry_pool_work_item(
        _target_tuning_cells(registry, 1)[0], estimated_duration_seconds=1.0
    )
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("unbound-budget-serial")
    )
    context = _e3a_execution_context(registry, inventory, envelope)
    plan = _scheduler(
        registry,
        inventory,
        envelope,
    ).schedule_work_items((item,), receipts_sha256=content_sha256("receipts"))

    async def runner(_assignment):
        raise AssertionError("an unbound diagnostic plan must not dispatch")

    with pytest.raises(ValueError, match="ExperimentBudget bindings"):
        asyncio.run(
            execute_dispatch_plan(
                plan,
                execution_context=context,  # type: ignore[arg-type]
                runner=runner,
            )
        )


def test_dispatch_cancellation_is_not_relabelled_as_a_failed_measurement(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("cancelled-dispatch")
    )
    context = _e3a_execution_context(registry, inventory, envelope)
    plan = context.issue_plan()

    async def runner(_assignment):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            _execute_planning_dispatch_for_engine_test(
                plan,
                execution_context=context,
                runner=runner,
            )
        )


def test_failure_stops_before_later_frozen_wave(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(2)
    items = tuple(
        registry_pool_work_item(cell, estimated_duration_seconds=1.0)
        for cell in _target_tuning_cells(registry, 4)
    )
    envelope = _envelope_for(inventory, items[0], (2,))
    context = _e3a_execution_context(registry, inventory, envelope)
    plan = context.issue_plan()
    assert len(plan.waves) > 1
    first_wave_ids = {
        assignment.assignment_id for assignment in plan.waves[0].assignments
    }
    calls: list[str] = []

    async def runner(assignment):
        calls.append(assignment.assignment_id)
        raise RuntimeError("wave failure")

    receipt = asyncio.run(
        _execute_planning_dispatch_for_engine_test(
            plan,
            execution_context=context,
            runner=runner,
        )
    )

    assert receipt.phase is DispatchExecutionPhase.FAILED
    assert len(receipt.wave_receipts) == 1
    assert set(calls) == first_wave_ids


def test_successful_wave_prefix_is_a_durable_running_receipt(
    registry: ExperimentRegistry,
    tmp_path: Path,
) -> None:
    inventory = _inventory(1)
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("durable-wave-prefix")
    )
    context = _e3a_execution_context(registry, inventory, envelope)
    plan = context.issue_plan()
    assert len(plan.waves) > 1

    def synthetic_success(**kwargs):
        assignment = kwargs["assignment"]
        budget = kwargs["budget"]
        wave = kwargs["wave"]
        evidence_path = str(
            (tmp_path / f"{assignment.assignment_id}.parquet").resolve()
        )
        binding = AssignmentTerminalBinding(
            authority_sha256=content_sha256(
                {"assignment": assignment.assignment_id, "authority": "raw-test"}
            ),
            cell_id=assignment.work_item.item_id,
            assignment_sha256=assignment.assignment_id,
            budget_sha256=budget.sha256,
            inventory_sha256=inventory.sha256,
            physical_gpu_uuids=assignment.gpu_uuids,
            execution_plan_sha256=content_sha256("synthetic-execution-plan"),
            dispatch_plan_sha256=plan.sha256,
            run_id=f"synthetic-{assignment.assignment_id[:16]}",
            run_nonce_sha256=content_sha256("synthetic-run-nonce"),
            terminal_receipt_path=str((tmp_path / "terminal.json").resolve()),
            terminal_receipt_sha256=content_sha256("terminal"),
            budget_observation_path=str((tmp_path / "budget.json").resolve()),
            budget_observation_sha256=content_sha256("budget"),
            budget_observation_sidecar_path=str(
                (tmp_path / "budget.json.sha256").resolve()
            ),
            budget_observation_sidecar_sha256=content_sha256("budget-sidecar"),
            native_terminal_artifact_path=str((tmp_path / "native.json").resolve()),
            native_terminal_raw_sha256=content_sha256("native-raw"),
            native_terminal_sha256=content_sha256("native"),
            trusted_attester_policy_sha256=content_sha256("trusted-policy"),
            evidence_file_paths=(evidence_path,),
            evidence_file_sha256s=(content_sha256("evidence"),),
        )
        return AssignmentExecutionReceipt(
            plan_sha256=plan.sha256,
            wave_sha256=wave.sha256,
            assignment_sha256=assignment.assignment_id,
            budget_sha256=budget.sha256,
            attempt=kwargs["attempt"],
            status=AssignmentExecutionStatus.SUCCEEDED,
            terminal_receipt_sha256=binding.sha256,
            terminal_binding=binding,
            failure_sha256=None,
            prior_attempt_receipt_sha256=None,
            gpu_count=len(assignment.gpu_uuids),
            fixed_instance_gpu_count=len(inventory.devices),
            attempt_intervals_monotonic_ns=((1, 2),),
            attributed_gpu_ns=len(assignment.gpu_uuids),
            attributed_fixed_instance_gpu_ns=len(inventory.devices),
        )

    wave = plan.waves[0]
    assignment_receipts = tuple(
        synthetic_success(
            assignment=assignment,
            budget=context.budgets_by_cell_id[assignment.work_item.item_id],
            wave=wave,
            attempt=1,
        )
        for assignment in wave.assignments
    )
    wave_receipt = gpu_pool_module._make_wave_execution_receipt(
        plan=plan,
        wave=wave,
        assignment_receipts=assignment_receipts,
        execution_context=context,
    )
    receipt = gpu_pool_module._make_schedule_receipt(
        plan=plan,
        phase=DispatchExecutionPhase.RUNNING,
        wave_receipts=(wave_receipt,),
        execution_context=context,
        prior_schedule_receipt_sha256=None,
    )

    assert receipt.phase is DispatchExecutionPhase.RUNNING
    assert len(receipt.wave_receipts) == 1
    serialized = json.loads(json.dumps(receipt.to_dict()))
    assert serialized["phase"] == DispatchExecutionPhase.RUNNING.value
    assert serialized["wave_receipt_sha256"] == [wave_receipt.sha256]
    assert receipt.sidecar().artifact_sha256 == receipt.sha256


def test_inventory_and_interference_digests_are_canonical() -> None:
    inventory = _inventory(2)
    assert inventory.sha256 == _inventory(2).sha256
    with pytest.raises(ValueError, match="sorted by UUID"):
        replace(inventory, devices=tuple(reversed(inventory.devices)))
    with pytest.raises(ValueError, match="duplicate rule keys"):
        rule = InterferenceRule(
            hardware_envelope_sha256=inventory.devices[0].hardware_envelope_sha256,
            workload_class=WorkloadClass.TUNING,
            co_run_signature="same",
            simultaneous_jobs=2,
            gang_shape="tp1_dp1",
            load_thermal_power_envelope="locked",
            contention_class="locked",
            evidence_sha256=content_sha256("evidence"),
        )
        InterferenceEnvelope(
            schema_version=1,
            rules=(rule, rule),
            source_receipt_sha256=content_sha256("source"),
        )


def test_empty_planning_plan_cannot_mint_an_execution_receipt(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("serial")
    )
    context = GpuDispatchPlanningContext(
        registry=registry,
        inventory=inventory,
        interference_envelope=envelope,
        budgets=(),
        receipts=_receipts_through(registry, "E0"),
        port_start=31_000,
        port_end=31_999,
    )
    plan = context.issue_plan()

    async def runner(_assignment):
        raise AssertionError("empty plan must not invoke its runner")

    with pytest.raises(TypeError, match="GpuDispatchExecutionContext"):
        asyncio.run(
            execute_dispatch_plan(
                plan,
                execution_context=context,  # type: ignore[arg-type]
                runner=runner,
            )
        )


def test_cli_artifacts_round_trip_full_content_and_reject_tampering(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(2)
    first_item = registry_pool_work_item(
        _target_tuning_cells(registry, 1)[0], estimated_duration_seconds=1.0
    )
    envelope = _envelope_for(inventory, first_item, (2,))
    context = _e3a_execution_context(registry, inventory, envelope)
    plan = context.issue_plan()

    inventory_json = json.loads(json.dumps(inventory.to_dict()))
    inventory_copy = GpuInventory.from_dict(
        inventory_json, sidecar=inventory.sidecar().to_dict()
    )
    assert inventory_copy == inventory
    assert ArtifactSidecar.from_dict(inventory.sidecar().to_dict()) == (
        inventory.sidecar()
    )

    envelope_json = json.loads(json.dumps(envelope.to_dict()))
    assert (
        InterferenceEnvelope.from_dict(envelope_json, sidecar=envelope.sidecar())
        == envelope
    )

    plan_json = json.loads(json.dumps(plan.to_dict()))
    assignment_json = plan_json["waves"][0]["assignments"][0]
    assert set(assignment_json) == {
        "work_item",
        "work_item_sha256",
        "gpu_uuids",
        "rank_groups",
        "ports",
    }
    assert "cell" in assignment_json["work_item"]
    assert "claim" in assignment_json["work_item"]
    restored_plan = GpuDispatchPlan.from_dict(
        plan_json,
        sidecar=plan.sidecar().to_dict(),
        planning_context=context,
    )
    assert restored_plan == plan
    assert restored_plan.sha256 == plan.sha256

    async def runner(assignment):
        return content_sha256({"terminal": assignment.assignment_id})

    receipt = asyncio.run(
        _execute_planning_dispatch_for_engine_test(
            plan,
            execution_context=context,
            runner=runner,
        )
    )
    receipt_json = json.loads(json.dumps(receipt.to_dict()))
    restored_receipt = _restore_planning_schedule_receipt_for_engine_test(
        receipt_json,
        sidecar=receipt.sidecar(),
        plan=restored_plan,
        execution_context=context,
    )
    assert restored_receipt == receipt

    unknown = copy.deepcopy(plan_json)
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match=r"unknown=\['unexpected'\]"):
        GpuDispatchPlan.from_dict(
            unknown,
            planning_context=context,
        )

    stale_sidecar = replace(plan.sidecar(), artifact_sha256="0" * 64)
    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        GpuDispatchPlan.from_dict(
            plan_json,
            sidecar=stale_sidecar,
            planning_context=context,
        )

    tampered = copy.deepcopy(plan_json)
    tampered["waves"][0]["assignments"][0]["ports"][0] += 1
    with pytest.raises(ValueError, match="assignment SHA-256 list mismatch"):
        GpuDispatchPlan.from_dict(
            tampered,
            planning_context=context,
        )


def _journal_terminal_binding(
    tmp_path: Path,
    *,
    plan: GpuDispatchPlan,
    context: GpuDispatchPlanningContext,
    assignment: GpuAssignment,
    budget: ExperimentBudget,
    label: str,
) -> AssignmentTerminalBinding:
    evidence_path = str((tmp_path / f"{label}.evidence.parquet").resolve())
    return AssignmentTerminalBinding(
        authority_sha256=content_sha256({"authority": label}),
        cell_id=assignment.work_item.item_id,
        assignment_sha256=assignment.assignment_id,
        budget_sha256=budget.sha256,
        inventory_sha256=context.inventory.sha256,
        physical_gpu_uuids=assignment.gpu_uuids,
        execution_plan_sha256=content_sha256({"execution-plan": label}),
        dispatch_plan_sha256=plan.sha256,
        run_id=f"journal-{label}",
        run_nonce_sha256=content_sha256({"nonce": label}),
        terminal_receipt_path=str((tmp_path / f"{label}.terminal.json").resolve()),
        terminal_receipt_sha256=content_sha256({"terminal": label}),
        budget_observation_path=str((tmp_path / f"{label}.budget.json").resolve()),
        budget_observation_sha256=content_sha256({"budget": label}),
        budget_observation_sidecar_path=str(
            (tmp_path / f"{label}.budget.json.sha256").resolve()
        ),
        budget_observation_sidecar_sha256=content_sha256({"budget-sidecar": label}),
        native_terminal_artifact_path=str(
            (tmp_path / f"{label}.native.json").resolve()
        ),
        native_terminal_raw_sha256=content_sha256({"native-raw": label}),
        native_terminal_sha256=content_sha256({"native": label}),
        trusted_attester_policy_sha256=content_sha256({"policy": label}),
        evidence_file_paths=(evidence_path,),
        evidence_file_sha256s=(content_sha256({"evidence": label}),),
    )


def _journal_assignment_receipt(
    *,
    plan: GpuDispatchPlan,
    wave: GpuDispatchWave,
    assignment: GpuAssignment,
    budget: ExperimentBudget,
    attempt: int,
    started_ns: int,
    prior: AssignmentExecutionReceipt | None,
    fixed_instance_gpu_count: int,
    terminal_binding: AssignmentTerminalBinding | None,
) -> AssignmentExecutionReceipt:
    finished_ns = started_ns + 1
    intervals = (() if prior is None else prior.attempt_intervals_monotonic_ns) + (
        (started_ns, finished_ns),
    )
    elapsed_ns = sum(finish - start for start, finish in intervals)
    succeeded = terminal_binding is not None
    return AssignmentExecutionReceipt(
        plan_sha256=plan.sha256,
        wave_sha256=wave.sha256,
        assignment_sha256=assignment.assignment_id,
        budget_sha256=budget.sha256,
        attempt=attempt,
        status=(
            AssignmentExecutionStatus.SUCCEEDED
            if succeeded
            else AssignmentExecutionStatus.FAILED
        ),
        terminal_receipt_sha256=(
            None if terminal_binding is None else terminal_binding.sha256
        ),
        terminal_binding=terminal_binding,
        failure_sha256=(
            None
            if succeeded
            else content_sha256(
                {"assignment": assignment.assignment_id, "attempt": attempt}
            )
        ),
        prior_attempt_receipt_sha256=None if prior is None else prior.sha256,
        gpu_count=len(assignment.gpu_uuids),
        fixed_instance_gpu_count=fixed_instance_gpu_count,
        attempt_intervals_monotonic_ns=intervals,
        attributed_gpu_ns=elapsed_ns * len(assignment.gpu_uuids),
        attributed_fixed_instance_gpu_ns=elapsed_ns * fixed_instance_gpu_count,
    )


def test_append_only_journal_recovers_partial_sibling_and_retries_only_failure(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(2)
    first_item = registry_pool_work_item(
        _target_tuning_cells(registry, 1)[0], estimated_duration_seconds=1.0
    )
    context = _e3a_execution_context(
        registry,
        inventory,
        _envelope_for(inventory, first_item, (2,)),
    )
    plan = context.issue_plan()
    wave = plan.waves[0]
    assert len(wave.assignments) == 2
    budgets = context.budgets_by_cell_id
    attempts = tuple(
        (assignment, 1, budgets[assignment.work_item.item_id], None)
        for assignment in wave.assignments
    )
    journal_root = tmp_path / "attempt-journal"
    journal = DispatchAttemptJournal.open_or_create(
        journal_root,
        plan=plan,
        execution_context=context,  # type: ignore[arg-type]
    )

    async def first_wave() -> None:
        await journal.begin_wave_attempts(
            plan=plan,
            wave=wave,
            attempts=attempts,
            prior_schedule_receipt_sha256=None,
        )
        tokens = []
        for assignment, attempt, budget, prior in attempts:
            tokens.append(
                await journal.begin_attempt(
                    plan=plan,
                    wave=wave,
                    assignment=assignment,
                    attempt=attempt,
                    budget=budget,
                    fixed_instance_gpu_count=len(inventory.devices),
                    prior_attempt_receipt=prior,
                    prior_schedule_receipt_sha256=None,
                )
            )
        success_assignment, failed_assignment = wave.assignments
        success_budget = budgets[success_assignment.work_item.item_id]
        failed_budget = budgets[failed_assignment.work_item.item_id]
        success = _journal_assignment_receipt(
            plan=plan,
            wave=wave,
            assignment=success_assignment,
            budget=success_budget,
            attempt=1,
            started_ns=tokens[0].started_monotonic_ns,
            prior=None,
            fixed_instance_gpu_count=len(inventory.devices),
            terminal_binding=_journal_terminal_binding(
                tmp_path,
                plan=plan,
                context=context,
                assignment=success_assignment,
                budget=success_budget,
                label="sibling-success",
            ),
        )
        failure = _journal_assignment_receipt(
            plan=plan,
            wave=wave,
            assignment=failed_assignment,
            budget=failed_budget,
            attempt=1,
            started_ns=tokens[1].started_monotonic_ns,
            prior=None,
            fixed_instance_gpu_count=len(inventory.devices),
            terminal_binding=None,
        )
        await journal.finish_attempt(token=tokens[0], receipt=success)
        await journal.finish_attempt(token=tokens[1], receipt=failure)

    asyncio.run(first_wave())
    failed_snapshot = journal.replay()
    failed_snapshot.require_complete_cost_authority()
    assert failed_snapshot.receipt is not None
    assert failed_snapshot.receipt.phase is DispatchExecutionPhase.FAILED
    assert failed_snapshot.receipt.wave_receipts[-1].partial_sibling_failure
    original_binding = failed_snapshot.binding
    assert original_binding is not None
    envelope_path = tmp_path / "failed-wave-receipt.json"
    publish_dispatch_schedule_receipt(
        envelope_path,
        failed_snapshot.receipt,
        attempt_journal=original_binding,
    )
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["schema_version"] == 2
    assert envelope["attempt_journal"] == original_binding.to_dict()

    # Coordinator crash after all FINISH rows but before the schedule envelope.
    reopened = DispatchAttemptJournal.open_or_create(
        journal_root,
        plan=plan,
        execution_context=context,  # type: ignore[arg-type]
    )
    assert reopened.replay().receipt == failed_snapshot.receipt
    failed_row = next(
        row
        for row in failed_snapshot.latest_assignment_receipts
        if row.status is AssignmentExecutionStatus.FAILED
    )
    failed_assignment = next(
        row
        for row in wave.assignments
        if row.assignment_id == failed_row.assignment_sha256
    )
    failed_budget = budgets[failed_assignment.work_item.item_id]

    async def retry_failure_only() -> None:
        retry_attempts = ((failed_assignment, 2, failed_budget, failed_row),)
        await reopened.begin_wave_attempts(
            plan=plan,
            wave=wave,
            attempts=retry_attempts,
            prior_schedule_receipt_sha256=failed_snapshot.receipt.sha256,
        )
        token = await reopened.begin_attempt(
            plan=plan,
            wave=wave,
            assignment=failed_assignment,
            attempt=2,
            budget=failed_budget,
            fixed_instance_gpu_count=len(inventory.devices),
            prior_attempt_receipt=failed_row,
            prior_schedule_receipt_sha256=failed_snapshot.receipt.sha256,
        )
        succeeded = _journal_assignment_receipt(
            plan=plan,
            wave=wave,
            assignment=failed_assignment,
            budget=failed_budget,
            attempt=2,
            started_ns=token.started_monotonic_ns,
            prior=failed_row,
            fixed_instance_gpu_count=len(inventory.devices),
            terminal_binding=_journal_terminal_binding(
                tmp_path,
                plan=plan,
                context=context,
                assignment=failed_assignment,
                budget=failed_budget,
                label="sibling-retry-success",
            ),
        )
        await reopened.finish_attempt(token=token, receipt=succeeded)

    asyncio.run(retry_failure_only())
    recovered = reopened.replay()
    recovered.require_complete_cost_authority()
    assert recovered.receipt is not None
    assert recovered.receipt.wave_receipts[0].succeeded
    retried = next(
        row
        for row in recovered.latest_assignment_receipts
        if row.assignment_sha256 == failed_assignment.assignment_id
    )
    assert retried.attempt == 2
    assert len(retried.attempt_intervals_monotonic_ns) == 2
    assert retried.prior_attempt_receipt_sha256 == failed_row.sha256

    # The older receipt remains a valid immutable prefix after append-only retry.
    DispatchAttemptJournal.open_or_create(
        journal_root,
        plan=plan,
        execution_context=context,  # type: ignore[arg-type]
        expected_prefix=original_binding,
    )

    final_binding = recovered.binding
    assert final_binding is not None
    event_paths = tuple(sorted((journal_root / "events").iterdir()))
    first_event = event_paths[0]
    first_body = first_event.read_bytes()
    first_event.chmod(0o600)
    first_event.write_bytes(first_body + b" ")
    first_event.chmod(0o400)
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="event_digest_mismatch",
    ):
        reopened.replay()
    first_event.chmod(0o600)
    first_event.write_bytes(first_body)
    first_event.chmod(0o400)

    deleted_body = first_event.read_bytes()
    first_event.unlink()
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="sequence_gap",
    ):
        reopened.replay()
    first_event.write_bytes(deleted_body)
    first_event.chmod(0o400)

    outside = tmp_path / "symlink-event-backup.json"
    first_event.replace(outside)
    first_event.symlink_to(outside)
    with pytest.raises((RuntimeError, ValueError)):
        reopened.replay()
    first_event.unlink()
    outside.replace(first_event)

    # Coordinated row+file rehash still cannot cross the immutable envelope
    # head that was published before the mutation.
    last_event = tuple(sorted((journal_root / "events").iterdir()))[-1]
    last_value = json.loads(last_event.read_text(encoding="utf-8"))
    binding_value = last_value["payload"]["assignment_receipt"]["terminal_binding"]
    binding_value["run_id"] = "jointly-rehashed-run"
    last_value["payload"]["assignment_receipt"]["terminal_receipt_sha256"] = (
        content_sha256(binding_value)
    )
    rewritten_body = (
        json.dumps(
            last_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        + b"\n"
    )
    rewritten_digest = hashlib.sha256(rewritten_body).hexdigest()
    rewritten_path = last_event.with_name(
        f"{int(last_event.name.split('.')[0]):012d}.{rewritten_digest}.json"
    )
    last_event.unlink()
    rewritten_path.write_bytes(rewritten_body)
    rewritten_path.chmod(0o400)
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="receipt_prefix_mismatch",
    ):
        DispatchAttemptJournal.open_or_create(
            journal_root,
            plan=plan,
            execution_context=context,  # type: ignore[arg-type]
            expected_prefix=final_binding,
        )


def test_append_only_journal_blocks_unfinished_intent_and_foreign_plan(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    context = _e3a_execution_context(
        registry,
        inventory,
        InterferenceEnvelope.serial(
            source_receipt_sha256=content_sha256("journal-unfinished")
        ),
    )
    plan = context.issue_plan()
    wave = plan.waves[0]
    assignment = wave.assignments[0]
    budget = context.budgets_by_cell_id[assignment.work_item.item_id]
    root = tmp_path / "unfinished-journal"
    journal = DispatchAttemptJournal.open_or_create(
        root,
        plan=plan,
        execution_context=context,  # type: ignore[arg-type]
    )

    async def leave_intent() -> None:
        await journal.begin_wave_attempts(
            plan=plan,
            wave=wave,
            attempts=((assignment, 1, budget, None),),
            prior_schedule_receipt_sha256=None,
        )

    asyncio.run(leave_intent())
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_attempt_intent_without_finish_cost_unresolved",
    ):
        journal.replay().require_complete_cost_authority()
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="manifest_identity_mismatch",
    ):
        DispatchAttemptJournal.open_or_create(
            root,
            plan=replace(plan, seed=plan.seed + 1),
            execution_context=context,  # type: ignore[arg-type]
        )


def test_append_only_journal_enforces_retry_allowance(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    context = _e3a_execution_context(
        registry,
        inventory,
        InterferenceEnvelope.serial(
            source_receipt_sha256=content_sha256("journal-retry-limit")
        ),
    )
    plan = context.issue_plan()
    wave = plan.waves[0]
    assignment = wave.assignments[0]
    budget = context.budgets_by_cell_id[assignment.work_item.item_id]
    assert budget.retry_allowance == 1
    journal = DispatchAttemptJournal.open_or_create(
        tmp_path / "retry-limit-journal",
        plan=plan,
        execution_context=context,  # type: ignore[arg-type]
    )

    async def fail_attempt(
        attempt: int,
        prior: AssignmentExecutionReceipt | None,
        prior_schedule_sha256: str | None,
    ) -> AssignmentExecutionReceipt:
        attempts = ((assignment, attempt, budget, prior),)
        await journal.begin_wave_attempts(
            plan=plan,
            wave=wave,
            attempts=attempts,
            prior_schedule_receipt_sha256=prior_schedule_sha256,
        )
        token = await journal.begin_attempt(
            plan=plan,
            wave=wave,
            assignment=assignment,
            attempt=attempt,
            budget=budget,
            fixed_instance_gpu_count=1,
            prior_attempt_receipt=prior,
            prior_schedule_receipt_sha256=prior_schedule_sha256,
        )
        receipt = _journal_assignment_receipt(
            plan=plan,
            wave=wave,
            assignment=assignment,
            budget=budget,
            attempt=attempt,
            started_ns=token.started_monotonic_ns,
            prior=prior,
            fixed_instance_gpu_count=1,
            terminal_binding=None,
        )
        await journal.finish_attempt(token=token, receipt=receipt)
        return receipt

    first = asyncio.run(fail_attempt(1, None, None))
    first_schedule = journal.replay().receipt
    assert first_schedule is not None
    asyncio.run(fail_attempt(2, first, first_schedule.sha256))
    second_snapshot = journal.replay()
    second_schedule = second_snapshot.receipt
    assert second_schedule is not None
    assert second_snapshot.replay_authority is not None

    class ResumeContext:
        resume_terminal_authorities = ()

        def __getattr__(self, name):
            return getattr(context, name)

    resume_context = ResumeContext()
    with patch.object(
        gpu_pool_module,
        "validate_dispatch_plan_for_execution",
        lambda candidate, *, execution_context: None,
    ):
        validate_dispatch_resume(
            plan,
            second_schedule,
            execution_context=resume_context,  # type: ignore[arg-type]
            attempt_journal_replay=second_snapshot.replay_authority,
        )
        with pytest.raises(ValueError, match="journal replay differs"):
            validate_dispatch_resume(
                plan,
                replace(
                    second_schedule,
                    prior_schedule_receipt_sha256=content_sha256(
                        "caller-rehashed-schedule"
                    ),
                ),
                execution_context=resume_context,  # type: ignore[arg-type]
                attempt_journal_replay=second_snapshot.replay_authority,
            )
    with pytest.raises(ValueError, match="retry would exceed"):
        asyncio.run(
            journal.begin_wave_attempts(
                plan=plan,
                wave=wave,
                attempts=(
                    (
                        assignment,
                        3,
                        budget,
                        journal.replay().latest_assignment_receipts[0],
                    ),
                ),
                prior_schedule_receipt_sha256=second_schedule.sha256,
            )
        )


def test_execute_dispatch_plan_uses_raw_journal_for_failed_resume(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    context = _e3a_execution_context(
        registry,
        inventory,
        InterferenceEnvelope.serial(
            source_receipt_sha256=content_sha256("journal-engine-resume")
        ),
    )
    plan = context.issue_plan()
    journal = DispatchAttemptJournal.open_or_create(
        tmp_path / "engine-journal",
        plan=plan,
        execution_context=context,  # type: ignore[arg-type]
    )

    class EngineContext:
        resume_terminal_authorities = ()

        def __getattr__(self, name):
            return getattr(context, name)

    engine_context = EngineContext()
    calls: list[str] = []

    async def failing_runner(assignment):
        calls.append(assignment.assignment_id)
        raise RuntimeError("journal-bound synthetic failure")

    with patch.object(
        gpu_pool_module,
        "validate_dispatch_plan_for_execution",
        lambda candidate, *, execution_context: None,
    ):
        first = asyncio.run(
            execute_dispatch_plan(
                plan,
                execution_context=engine_context,  # type: ignore[arg-type]
                runner=failing_runner,
                attempt_journal=journal,
                stop_after_wave_index=0,
            )
        )
        first_snapshot = journal.replay()
        assert first_snapshot.receipt == first
        assert first_snapshot.replay_authority is not None
        second = asyncio.run(
            execute_dispatch_plan(
                plan,
                execution_context=engine_context,  # type: ignore[arg-type]
                runner=failing_runner,
                resume_receipt=first,
                attempt_journal=journal,
                attempt_journal_replay=first_snapshot.replay_authority,
                stop_after_wave_index=0,
            )
        )
        second_snapshot = journal.replay()
        assert second_snapshot.receipt == second
        assert second.wave_receipts[-1].assignment_receipts[0].attempt == 2
        assert (
            len(
                second.wave_receipts[-1]
                .assignment_receipts[0]
                .attempt_intervals_monotonic_ns
            )
            == 2
        )
        assert second_snapshot.replay_authority is not None
        with pytest.raises(ValueError, match="retry would exceed"):
            asyncio.run(
                execute_dispatch_plan(
                    plan,
                    execution_context=engine_context,  # type: ignore[arg-type]
                    runner=failing_runner,
                    resume_receipt=second,
                    attempt_journal=journal,
                    attempt_journal_replay=second_snapshot.replay_authority,
                    stop_after_wave_index=0,
                )
            )
    assert len(calls) == 2


def test_execute_dispatch_plan_journal_preserves_true_partial_sibling(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(2)
    first_item = registry_pool_work_item(
        _target_tuning_cells(registry, 1)[0], estimated_duration_seconds=1.0
    )
    context = _e3a_execution_context(
        registry,
        inventory,
        _envelope_for(inventory, first_item, (2,)),
    )
    plan = context.issue_plan()
    wave = plan.waves[0]
    assert len(wave.assignments) == 2
    budgets = context.budgets_by_cell_id
    bindings = {
        assignment.assignment_id: _journal_terminal_binding(
            tmp_path,
            plan=plan,
            context=context,
            assignment=assignment,
            budget=budgets[assignment.work_item.item_id],
            label=f"engine-partial-{index}",
        )
        for index, assignment in enumerate(wave.assignments)
    }
    exact_terminal = object.__new__(AssignmentTerminalAuthority)

    def revalidate_terminal(
        _authority,
        *,
        registry,
        inventory,
        assignment_sha256,
        budget_sha256,
        physical_gpu_uuids,
    ):
        assert registry == context.registry
        assert inventory == context.inventory
        binding = bindings[assignment_sha256]
        assert binding.budget_sha256 == budget_sha256
        assert binding.physical_gpu_uuids == physical_gpu_uuids
        return binding

    class ResumeAuthority:
        def __init__(self, binding: AssignmentTerminalBinding) -> None:
            self.binding = binding
            self.sha256 = binding.authority_sha256

        def revalidate(self, **kwargs):
            assert kwargs["assignment_sha256"] == self.binding.assignment_sha256
            assert kwargs["budget_sha256"] == self.binding.budget_sha256
            assert kwargs["physical_gpu_uuids"] == self.binding.physical_gpu_uuids
            return self.binding

    class EngineContext:
        def __init__(self, resume_authorities=()) -> None:
            self.resume_terminal_authorities = resume_authorities

        def __getattr__(self, name):
            return getattr(context, name)

    failed_id = wave.assignments[-1].assignment_id
    first_calls: list[str] = []

    async def partial_runner(assignment):
        first_calls.append(assignment.assignment_id)
        if assignment.assignment_id == failed_id:
            raise RuntimeError("synthetic true sibling failure")
        return exact_terminal

    root = tmp_path / "engine-partial-journal"
    journal = DispatchAttemptJournal.open_or_create(
        root,
        plan=plan,
        execution_context=context,  # type: ignore[arg-type]
    )
    with (
        patch.object(
            gpu_pool_module,
            "validate_dispatch_plan_for_execution",
            lambda candidate, *, execution_context: None,
        ),
        patch.object(
            AssignmentTerminalAuthority,
            "revalidate",
            revalidate_terminal,
        ),
    ):
        failed = asyncio.run(
            execute_dispatch_plan(
                plan,
                execution_context=EngineContext(),  # type: ignore[arg-type]
                runner=partial_runner,
                attempt_journal=journal,
                stop_after_wave_index=0,
            )
        )
        failed_snapshot = journal.replay()
        assert failed_snapshot.receipt == failed
        assert failed.phase is DispatchExecutionPhase.FAILED
        assert failed.wave_receipts[-1].partial_sibling_failure
        succeeded_row = next(
            row
            for row in failed.wave_receipts[-1].assignment_receipts
            if row.status is AssignmentExecutionStatus.SUCCEEDED
        )
        assert set(first_calls) == {
            assignment.assignment_id for assignment in wave.assignments
        }
        assert failed_snapshot.binding is not None
        assert failed_snapshot.replay_authority is not None

        # Simulate a new coordinator process: reconstruct exclusively from the
        # raw journal and the successful row's reopened terminal authority.
        reopened = DispatchAttemptJournal.open_or_create(
            root,
            plan=plan,
            execution_context=context,  # type: ignore[arg-type]
            expected_prefix=failed_snapshot.binding,
        )
        replay = reopened.replay()
        assert replay.receipt == failed
        assert replay.replay_authority is not None
        retry_calls: list[str] = []

        async def retry_runner(assignment):
            retry_calls.append(assignment.assignment_id)
            return exact_terminal

        recovered = asyncio.run(
            execute_dispatch_plan(
                plan,
                execution_context=EngineContext(
                    (ResumeAuthority(bindings[succeeded_row.assignment_sha256]),)
                ),  # type: ignore[arg-type]
                runner=retry_runner,
                resume_receipt=replay.receipt,
                attempt_journal=reopened,
                attempt_journal_replay=replay.replay_authority,
                stop_after_wave_index=0,
            )
        )

    assert retry_calls == [failed_id]
    assert recovered.wave_receipts[0].succeeded
    retried_row = next(
        row
        for row in recovered.wave_receipts[0].assignment_receipts
        if row.assignment_sha256 == failed_id
    )
    assert retried_row.attempt == 2
    assert len(retried_row.attempt_intervals_monotonic_ns) == 2
    assert reopened.replay().receipt == recovered


def test_execute_dispatch_plan_chains_each_journaled_wave_receipt(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    context = _e3a_execution_context(
        registry,
        inventory,
        InterferenceEnvelope.serial(
            source_receipt_sha256=content_sha256("journal-multi-wave-chain")
        ),
    )
    plan = context.issue_plan()
    assert len(plan.waves) > 1
    budgets = context.budgets_by_cell_id
    selected_assignments = tuple(
        assignment for wave in plan.waves[:2] for assignment in wave.assignments
    )
    bindings = {
        assignment.assignment_id: _journal_terminal_binding(
            tmp_path,
            plan=plan,
            context=context,
            assignment=assignment,
            budget=budgets[assignment.work_item.item_id],
            label=f"engine-chain-{index}",
        )
        for index, assignment in enumerate(selected_assignments)
    }
    exact_terminal = object.__new__(AssignmentTerminalAuthority)

    def revalidate_terminal(
        _authority,
        *,
        registry,
        inventory,
        assignment_sha256,
        budget_sha256,
        physical_gpu_uuids,
    ):
        assert registry == context.registry
        assert inventory == context.inventory
        binding = bindings[assignment_sha256]
        assert binding.budget_sha256 == budget_sha256
        assert binding.physical_gpu_uuids == physical_gpu_uuids
        return binding

    class EngineContext:
        resume_terminal_authorities = ()

        def __getattr__(self, name):
            return getattr(context, name)

    async def successful_runner(_assignment):
        return exact_terminal

    journal = DispatchAttemptJournal.open_or_create(
        tmp_path / "engine-multi-wave-journal",
        plan=plan,
        execution_context=context,  # type: ignore[arg-type]
    )
    with (
        patch.object(
            gpu_pool_module,
            "validate_dispatch_plan_for_execution",
            lambda candidate, *, execution_context: None,
        ),
        patch.object(
            AssignmentTerminalAuthority,
            "revalidate",
            revalidate_terminal,
        ),
    ):
        receipt = asyncio.run(
            execute_dispatch_plan(
                plan,
                execution_context=EngineContext(),  # type: ignore[arg-type]
                runner=successful_runner,
                attempt_journal=journal,
                stop_after_wave_index=1,
            )
        )

    snapshot = journal.replay()
    assert snapshot.receipt == receipt
    assert len(receipt.wave_receipts) == 2
    wave_events = []
    for path in sorted((journal.root / "events").iterdir()):
        event = json.loads(path.read_text(encoding="utf-8"))
        if event["event_type"] == "WAVE":
            wave_events.append(event)
    assert len(wave_events) == 2
    assert wave_events[0]["payload"]["prior_schedule_receipt_sha256"] is None
    second_sequence = wave_events[1]["sequence"]
    prefix = journal.replay(event_count=second_sequence)
    assert prefix.receipt is not None
    assert (
        wave_events[1]["payload"]["prior_schedule_receipt_sha256"]
        == prefix.receipt.sha256
    )
    assert receipt.prior_schedule_receipt_sha256 == prefix.receipt.sha256


@pytest.mark.parametrize("crash_point", ("root", "events", "manifest_prefix"))
def test_attempt_journal_initialization_safe_prefix_is_idempotent(
    tmp_path: Path,
    registry: ExperimentRegistry,
    crash_point: str,
) -> None:
    inventory = _inventory(1)
    context = _e3a_execution_context(
        registry,
        inventory,
        InterferenceEnvelope.serial(
            source_receipt_sha256=content_sha256({"journal-init-crash": crash_point})
        ),
    )
    plan = context.issue_plan()
    root = tmp_path / f"journal-init-{crash_point}"
    root.mkdir(mode=0o700)
    if crash_point in {"events", "manifest_prefix"}:
        (root / "events").mkdir(mode=0o700)
    if crash_point == "manifest_prefix":
        manifest = {
            "schema_version": 1,
            "kind": "industrial_dispatch_attempt_journal",
            "protocol_sha256": DispatchAttemptJournal._PROTOCOL_SHA256,
            "journal_path": str(root),
            "events_path": str(root / "events"),
            "plan_sha256": plan.sha256,
            "execution_context_sha256": context.sha256,
            "inventory_sha256": inventory.sha256,
            "fixed_instance_gpu_count": 1,
        }
        body = execution_bundle_module._canonical_bytes(manifest) + b"\n"
        (root / "manifest.json").write_bytes(body[: len(body) // 2])
        (root / "manifest.json").chmod(0o400)

    journal = DispatchAttemptJournal.open_or_create(
        root,
        plan=plan,
        execution_context=context,  # type: ignore[arg-type]
    )
    snapshot = journal.replay()
    assert snapshot.receipt is None
    assert snapshot.event_sha256s == ()
    assert tuple(sorted(path.name for path in root.iterdir())) == (
        "events",
        "manifest.json",
    )
    assert (
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))["plan_sha256"]
        == plan.sha256
    )
