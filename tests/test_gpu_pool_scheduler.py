from __future__ import annotations

import asyncio
import copy
import json
from collections import Counter
from dataclasses import replace

import pytest

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.gpu_pool import (
    ArtifactSidecar,
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
from lightcone_spec.experiments.statistics import PilotBlock, preregister_power_sizing


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
) -> GpuDispatchExecutionContext:
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
    return GpuDispatchExecutionContext(
        registry=registry,
        inventory=inventory,
        interference_envelope=envelope,
        budgets=budgets,
        receipts=_receipts_through(registry, "preflight"),
        port_start=31_000,
        port_end=31_999,
        seed=20260811,
    )


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
    with pytest.raises(ValueError, match="cannot trust bare completed cell IDs"):
        GpuDispatchExecutionContext(
            registry=registry,
            inventory=inventory,
            interference_envelope=envelope,
            budgets=planning.budgets,
            receipts=planning.receipts,
            completed_cell_ids=planning.completed_cell_ids,
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


@pytest.mark.parametrize("size", (1, 2, 4, 8, 16))
def test_scheduler_issued_context_executes_on_required_pool_sizes(
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
        execute_dispatch_plan(
            plan,
            execution_context=context,
            runner=runner,
        )
    )

    assert receipt.phase is DispatchExecutionPhase.COMPLETE
    assert len(calls) == len(context.budgets)
    assert receipt.fixed_instance_gpu_count == size
    assert receipt.fixed_instance_actual_billed_gpu_ns == (
        sum(finish - start for start, finish in receipt.active_intervals_monotonic_ns)
        * size
    )
    if size > 1:
        assert receipt.per_assignment_attributed_fixed_instance_gpu_ns > (
            receipt.fixed_instance_actual_billed_gpu_ns
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
    item = registry_pool_work_item(preflight, estimated_duration_seconds=1.0)
    scheduler = _scheduler(
        registry,
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
            registry,
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


def test_compile_is_host_exclusive_and_blocked_methods_never_dispatch(
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
    plan = _scheduler(registry, inventory, envelope).schedule_work_items(
        (tuning, preflight), receipts_sha256=content_sha256("receipts")
    )

    compile_waves = tuple(
        wave
        for wave in plan.waves
        if any(
            assignment.work_item.claim.workload_class is WorkloadClass.COMPILE
            for assignment in wave.assignments
        )
    )
    assert len(compile_waves) == 1
    assert len(compile_waves[0].assignments) == 1

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
            execute_dispatch_plan(
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
            execute_dispatch_plan(
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
            execute_dispatch_plan(
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
            execute_dispatch_plan(
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
            execute_dispatch_plan(
                forged,
                execution_context=context,
                runner=runner,
            )
        )
    assert calls == []


def test_partial_sibling_failure_is_receipted_but_untrusted_resume_is_blocked(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(2)
    first_item = registry_pool_work_item(
        _target_tuning_cells(registry, 1)[0], estimated_duration_seconds=1.0
    )
    envelope = _envelope_for(inventory, first_item, (2,))
    context = _e3a_execution_context(registry, inventory, envelope)
    plan = context.issue_plan()
    assert len(plan.waves[-1].assignments) == 2
    failed_id = plan.waves[-1].assignments[0].assignment_id
    first_calls: list[str] = []

    async def first_runner(assignment):
        first_calls.append(assignment.assignment_id)
        if assignment.assignment_id == failed_id:
            raise RuntimeError("synthetic sibling failure")
        return content_sha256({"terminal": assignment.assignment_id})

    failed = asyncio.run(
        execute_dispatch_plan(
            plan,
            execution_context=context,
            runner=first_runner,
        )
    )
    assert failed.phase is DispatchExecutionPhase.FAILED
    assert failed.wave_receipts[-1].partial_sibling_failure
    assert len(first_calls) == len(context.budgets)
    assert {
        receipt.status for receipt in failed.wave_receipts[-1].assignment_receipts
    } == {AssignmentExecutionStatus.SUCCEEDED, AssignmentExecutionStatus.FAILED}
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

    with pytest.raises(ValueError, match="durable terminal artifact store"):
        asyncio.run(
            execute_dispatch_plan(
                plan,
                execution_context=context,
                runner=resume_runner,
                resume_receipt=failed,
            )
        )
    assert resume_calls == []

    other_inventory = _inventory(2, source="other-inventory")
    other_envelope = _envelope_for(other_inventory, first_item, (2,))
    foreign_context = _e3a_execution_context(registry, other_inventory, other_envelope)
    foreign_plan = foreign_context.issue_plan()
    with pytest.raises(ValueError, match="another dispatch plan"):
        validate_dispatch_resume(
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
        execute_dispatch_plan(
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
            execute_dispatch_plan(
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
                execution_context=context,
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
            execute_dispatch_plan(
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
        execute_dispatch_plan(
            plan,
            execution_context=context,
            runner=runner,
        )
    )

    assert receipt.phase is DispatchExecutionPhase.FAILED
    assert len(receipt.wave_receipts) == 1
    assert set(calls) == first_wave_ids


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


def test_empty_frozen_plan_has_a_complete_noop_receipt(
    registry: ExperimentRegistry,
) -> None:
    inventory = _inventory(1)
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("serial")
    )
    context = GpuDispatchExecutionContext(
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

    receipt = asyncio.run(
        execute_dispatch_plan(
            plan,
            execution_context=context,
            runner=runner,
        )
    )

    assert receipt.phase is DispatchExecutionPhase.COMPLETE
    assert receipt.wave_receipts == ()
    with pytest.raises(ValueError, match="durable terminal artifact store"):
        validate_dispatch_resume(
            plan,
            receipt,
            execution_context=context,
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
        execute_dispatch_plan(
            plan,
            execution_context=context,
            runner=runner,
        )
    )
    receipt_json = json.loads(json.dumps(receipt.to_dict()))
    restored_receipt = DispatchScheduleReceipt.from_dict(
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
