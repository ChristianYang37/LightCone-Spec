from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.cli.main import _validate_e2_final_seal_authority, _write_json
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.planning import (
    _E2_PROMOTION_MINIMA,
    E1_COMMON_LOAD_AUTHORITY_UNREGISTERED_REASON,
    E2_HALVING_PROTOCOL_SHA256,
    E2_PROMOTION_MINIMA_AUTHORITY_SHA256,
    BudgetInventoryIdentity,
    BudgetJobKind,
    BudgetObservationReceipt,
    ConfirmationFamilyPowerReductionArtifact,
    DispositionStatus,
    E1GeometryIdentity,
    E1ParetoArtifact,
    E2CandidateEvaluation,
    E2CandidateIdentity,
    E2StageEvidenceArtifact,
    E2StageReductionArtifact,
    E2SurvivorReceipt,
    EvidenceAliasCandidate,
    EvidenceAliasReceipt,
    ExactScenarioHours,
    ExecutionSemanticsIdentity,
    ExpectedMaximumCount,
    ExperimentBudget,
    P99AnchorStatus,
    PresentationAxis,
    RawEvidenceRunBinding,
    ScenarioMilliseconds,
    SealedE3aSelection,
    _reduce_e2_successive_halving,
    build_evidence_dependence_map,
    derive_confirmation_family,
    estimate_industrial_budget,
    family_pilot_block_id,
    materialize_confirmation_pilots,
    materialize_confirmation_prefix,
    materialize_e2_final_recipe,
    reduce_e1_activation,
    reduce_e2_activation,
    reduce_e2_successive_halving,
    seal_confirmation_family_power,
    verify_e1_activation,
    verify_e2_activation,
)
from lightcone_spec.experiments.planning_artifacts import (
    e2_final_recipe_artifact_to_dict,
)
from lightcone_spec.experiments.registry import (
    CONFIRMATION_METHOD_ROLES,
    FINAL_BLOCKS,
    PILOT_BLOCKS,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    LockedOutput,
    WorkloadClass,
    build_industrial_registry,
    content_sha256,
    scientific_role_for_cell,
)
from lightcone_spec.experiments.statistics import (
    PilotBlock,
    preregister_power_sizing,
)


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_industrial_registry(
        gpu_uuids=("GPU-aaaaaaaa", "GPU-bbbbbbbb"),
        base_port=24000,
        cache_root="runtime-cache/planning-test",
        evidence_root="artifacts/planning-test",
    )


def _sha(label: str) -> str:
    return content_sha256({"test": label})


def _gpu_inventory() -> GpuInventory:
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-planning-{index}",
            host_id="host-a",
            model="A100",
            memory_bytes=80_000_000_000,
            compute_capability=(8, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root="root-0",
            numa_node=0,
            interconnects=("nvlink",),
            peer_access_class="nvlink",
            clock_policy="locked",
            power_limit_watts=300.0,
            thermal_limit_celsius=85.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=("planning-nvlink-group",),
        )
        for index in range(2)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id="planning-nvlink-group",
                host_id="host-a",
                gpu_uuids=tuple(device.uuid for device in devices),
                fabric="nvlink",
                bandwidth_class="high",
            ),
        ),
        source_receipt_sha256=_sha("planning-inventory-source"),
    )


def _raw_binding(
    label: str,
    *,
    cell_id: str,
    experiment: str,
    runtime_sha256: str,
    split_sha256: str,
    scientific_unit: str,
    method: str = "tts",
    scientific_role: str | None = None,
) -> RawEvidenceRunBinding:
    role = method if scientific_role is None else scientific_role
    return RawEvidenceRunBinding(
        schema_version=3,
        cell_id=cell_id,
        experiment=experiment,
        method=method,
        scientific_role=role,
        scientific_unit=scientific_unit,
        config_sha256=cell_id,
        rank_config_sha256s=(_sha(f"{label}-rank"),),
        run_id=f"run-{label}",
        rank_count=1,
        model_pair="target-model",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        corpus_sha256=_sha(f"{label}-corpus"),
        arrival_trace_sha256=_sha(f"{label}-arrival"),
        request_ids_sha256=_sha(f"{label}-requests"),
        sampling_profile_sha256=_sha(f"{label}-sampling"),
        model_lock_sha256=_sha(f"{label}-model"),
        patched_sglang_tree="a" * 40,
        run_nonce_sha256=_sha(f"{label}-nonce"),
        topology_sha256=_sha(f"{label}-topology"),
        experiment_budget_sha256=_sha(f"{label}-budget-plan"),
        physical_gpu_uuids=(f"GPU-{label}",),
        terminal_receipt_sha256s=(_sha(f"{label}-terminal"),),
        hardware_receipt_sha256=_sha(f"{label}-hardware"),
        budget_observation_sha256=_sha(f"{label}-budget"),
        execution_plan_sha256=_sha(f"{label}-execution-plan"),
        execution_split_sha256=_sha(f"{label}-execution-split"),
    )


def _direct_receipt(
    registry: ExperimentRegistry,
    experiment: str,
    *,
    outputs: dict[str, str],
    runtime_sha256: str,
    split_sha256: str,
) -> ExperimentReceipt:
    definition = registry.definition(experiment)
    assert set(outputs) == set(definition.locked_outputs)
    return ExperimentReceipt(
        experiment=experiment,
        registry_sha256=registry.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        completed_cells_sha256=_sha(f"{experiment}-completed"),
        dependency_receipts=tuple(
            LockedOutput(name=name, content_sha256=_sha(f"dependency-{name}"))
            for name in definition.dependencies
        ),
        outputs=tuple(
            LockedOutput(name=name, content_sha256=outputs[name])
            for name in sorted(outputs)
        ),
    )


def _e3a_selection_and_receipt(
    registry: ExperimentRegistry,
    *,
    width: int = 8,
    concurrency: int = 4,
) -> tuple[SealedE3aSelection, ExperimentReceipt]:
    selection = SealedE3aSelection(
        schema_version=1,
        registry_sha256=registry.sha256,
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("e3a-split"),
        width=width,
        concurrency=concurrency,
        reducer_evidence_sha256=_sha("e3a-evidence"),
    )
    outputs = {
        name: _sha(f"E3a-{name}") for name in registry.definition("E3a").locked_outputs
    }
    outputs["matched_width"] = selection.matched_width_output_sha256
    outputs["e1_reference_load"] = selection.reference_load_output_sha256
    receipt = _direct_receipt(
        registry,
        "E3a",
        outputs=outputs,
        runtime_sha256=selection.runtime_sha256,
        split_sha256=selection.split_sha256,
    )
    return selection, receipt


def _milliseconds(value: int) -> ScenarioMilliseconds:
    return ScenarioMilliseconds(value, value, value)


def _budget(cell: ExperimentCell, *, scored_ms: int = 1_000) -> ExperimentBudget:
    component = _milliseconds(100)
    zero = _milliseconds(0)
    wall_ms = 7 * 100 + scored_ms
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
        scored_arrival=_milliseconds(scored_ms),
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
        fixed_instance_billed_gpu_ms=_milliseconds(wall_ms * 2),
    )


def test_experiment_budget_exact_arithmetic_and_observation_delta(
    registry: ExperimentRegistry,
) -> None:
    cells = tuple(registry.cells_for("E1")[:2])
    budgets = tuple(_budget(cell) for cell in cells)
    inventory = BudgetInventoryIdentity(
        schema_version=1,
        host_sha256=_sha("host"),
        gpu_uuids=registry.gpu_uuids,
        topology_sha256=_sha("topology"),
    )
    report = estimate_industrial_budget(
        registry,
        activated_cell_ids=tuple(cell.cell_id for cell in cells),
        activation_sha256=_sha("activation"),
        budgets=budgets,
        inventory=inventory,
    )

    assert report.cells == 2
    assert report.gpu_cell_units == 2
    assert report.compute_gpu_ms.registered == 3_400
    assert report.compute_gpu_hours == ExactScenarioHours(3_400, 3_400, 3_400)
    assert report.estimated_wall_ms is None
    assert report.schedule_fixed_instance_billed_gpu_ms is None
    assert report.unresolved_assumptions == (
        (
            "exact_inventory_schedule_unresolved:"
            "full_inventory_and_interference_required"
        ),
    )
    assert report.output_tokens == ExpectedMaximumCount(2_000, 4_000)
    assert report.excluded_warmup_requests == ExpectedMaximumCount(16, 20)
    assert report.minimum_completed_requests == 128
    assert report.retry_allowance == 2
    assert sum(group.cells for group in report.groups) == 2
    assert (
        report.sha256
        == estimate_industrial_budget(
            registry,
            activated_cell_ids=tuple(reversed(tuple(cell.cell_id for cell in cells))),
            activation_sha256=_sha("activation"),
            budgets=tuple(reversed(budgets)),
            inventory=inventory,
        ).sha256
    )

    with pytest.raises(ValueError, match="coverage mismatch"):
        estimate_industrial_budget(
            registry,
            activated_cell_ids=tuple(cell.cell_id for cell in cells),
            activation_sha256=_sha("activation"),
            budgets=budgets[:1],
            inventory=inventory,
        )

    observed = BudgetObservationReceipt(
        schema_version=1,
        budget=budgets[0],
        observed_component_ms=tuple(
            (name, getattr(budgets[0], name).registered + 1)
            for name in (
                "startup_model_load",
                "compile_jit_graph_prewarm",
                "excluded_warmup",
                "scored_arrival",
                "drain",
                "reset_finalization",
                "evidence_flush_shutdown",
                "soak",
                "failure_injection",
                "retry",
                "profiler",
                "download_compile_reservation",
            )
        ),
        measured_gpu_ms=1_712,
        fixed_instance_billed_gpu_ms=3_424,
        terminal_evidence_sha256=_sha("terminal"),
    )
    assert observed.registered_wall_delta_ms == 12
    assert observed.registered_gpu_delta_ms == 12
    assert observed.registered_billed_delta_ms == 24


def test_budget_missing_job_specific_duration_and_inventory_are_fail_closed(
    registry: ExperimentRegistry,
) -> None:
    cell = next(
        cell
        for cell in registry.cells_for("E4")
        if cell.resources.workload_class is WorkloadClass.PROFILE
    )
    budget = _budget(cell)
    with pytest.raises(ValueError, match="profiler jobs require"):
        replace(budget, job_kind=BudgetJobKind.PROFILER)
    with pytest.raises(ValueError, match="only for preregistered p99-anchor"):
        replace(budget, p99_anchor_status=P99AnchorStatus.LOCKED)
    with pytest.raises(ValueError, match="at least 10,000"):
        replace(
            budget,
            job_kind=BudgetJobKind.P99_ANCHOR,
            p99_anchor_status=P99AnchorStatus.LOCKED,
        )
    p99_budget = replace(
        budget,
        job_kind=BudgetJobKind.P99_ANCHOR,
        p99_anchor_status=P99AnchorStatus.LOCKED,
        minimum_completed_requests=10_000,
    )
    assert p99_budget.minimum_completed_requests == 10_000

    inventory = BudgetInventoryIdentity(
        schema_version=1,
        host_sha256=_sha("one-host"),
        gpu_uuids=("GPU-only",),
        topology_sha256=_sha("one-topology"),
    )
    report = estimate_industrial_budget(
        registry,
        activated_cell_ids=(cell.cell_id,),
        activation_sha256=_sha("gang-activation"),
        budgets=(budget,),
        inventory=inventory,
    )
    assert report.estimated_wall_ms is None
    assert report.unresolved_assumptions == (
        f"cell_requires_2_gpus_but_inventory_has_1:{cell.cell_id}",
    )


def test_e1_reducer_materializes_exact_single_68_cell_slice(
    registry: ExperimentRegistry,
) -> None:
    selection, receipt = _e3a_selection_and_receipt(registry)
    artifact = reduce_e1_activation(
        registry,
        e3a_receipt=receipt,
        selection=selection,
    )

    active = {
        cell.cell_id: cell
        for cell in registry.cells_for("E1")
        if cell.cell_id in set(artifact.plan.activated_cell_ids)
    }
    assert len(registry.cells_for("E1")) == 1_428
    tag = f"width={selection.width}:concurrency={selection.concurrency}"
    selected_slice = tuple(
        cell for cell in registry.cells_for("E1") if tag in cell.identity.variant
    )
    assert len(selected_slice) == 68
    assert Counter(
        scientific_role_for_cell(registry, cell) for cell in selected_slice
    ) == {
        "target_only": 1,
        "static": 1,
        "tts": 1,
        "l0_naive": 1,
        "lc_candidate": 64,
    }
    assert set(active) == {cell.cell_id for cell in selected_slice if cell.runnable}
    assert len(artifact.dispositions) == 1_428
    disposition_by_id = {row.cell_id: row for row in artifact.dispositions}
    assert all(
        disposition_by_id[cell.cell_id].status is not DispositionStatus.DEFERRED
        for cell in selected_slice
    )
    frozen_anchor_rows = tuple(
        disposition_by_id[cell.cell_id]
        for cell in selected_slice
        if scientific_role_for_cell(registry, cell) in {"tts", "l0_naive"}
    )
    assert len(frozen_anchor_rows) == 2
    assert all(row.status is DispositionStatus.BLOCKED for row in frozen_anchor_rows)
    assert all(
        row.reason_code == "tts_official_recipe_unavailable"
        for row in frozen_anchor_rows
    )
    assert {
        row.cell_id
        for row in artifact.dispositions
        if row.status is DispositionStatus.DEFERRED
    } == {
        cell.cell_id
        for cell in registry.cells_for("E1")
        if tag not in cell.identity.variant and cell.runnable
    }

    forged = replace(selection, concurrency=8)
    with pytest.raises(ValueError, match="selected load artifact"):
        reduce_e1_activation(registry, e3a_receipt=receipt, selection=forged)
    forged_plan = replace(
        artifact.plan, source_selection_sha256=_sha("forged-e1-selection")
    )
    forged_artifact = replace(artifact, plan=forged_plan)
    with pytest.raises(ValueError, match="not the exact reducer-generated"):
        verify_e1_activation(
            registry,
            e3a_receipt=receipt,
            selection=selection,
            artifact=forged_artifact,
        )


def _e1_pareto_and_receipt(
    registry: ExperimentRegistry,
) -> tuple[E1ParetoArtifact, ExperimentReceipt]:
    cell = next(
        cell
        for cell in registry.cells_for("E2")
        if scientific_role_for_cell(registry, cell) == "lc_candidate"
        and cell.identity.optimizer != "chronobelief"
        and "halving_stage=0:" in cell.identity.variant
    )
    geometry = E1GeometryIdentity.from_cell(cell, registry=registry)
    pareto = E1ParetoArtifact(
        schema_version=2,
        registry_sha256=registry.sha256,
        runtime_sha256=_sha("e2-runtime"),
        split_sha256=_sha("e1-selection-split"),
        e1_activation_sha256=_sha("e1-activation"),
        reducer_evidence_sha256=_sha("e1-pareto-evidence"),
        surviving_geometries=(geometry,),
        selection_state="sealed_before_e2_unblinding",
    )
    outputs = {
        "dflash_pareto_set": pareto.sha256,
        "common_downstream_load": _sha("untrusted-common-load"),
    }
    receipt = _direct_receipt(
        registry,
        "E1",
        outputs=outputs,
        runtime_sha256=pareto.runtime_sha256,
        split_sha256=pareto.split_sha256,
    )
    return pareto, receipt


@pytest.mark.parametrize("wrong_schema", (True, 1.0))
def test_e1_pareto_schema_version_is_an_exact_integer(
    registry: ExperimentRegistry,
    wrong_schema: object,
) -> None:
    pareto, _ = _e1_pareto_and_receipt(registry)
    with pytest.raises(ValueError, match="Pareto schema version 2"):
        replace(pareto, schema_version=wrong_schema)


def _require_executable_e2_recipe_authority(registry: ExperimentRegistry) -> None:
    if not any(
        cell.runnable
        for cell in registry.cells_for("E2")
        if cell.identity.method in {"tts", "l0"}
    ):
        pytest.skip(
            "E2 positive halving path awaits source-owned optimizer/stride/width values"
        )


def _active_e2_candidates(
    registry: ExperimentRegistry, cell_ids: tuple[str, ...]
) -> dict[str, E2CandidateIdentity]:
    known = {cell.cell_id: cell for cell in registry.cells_for("E2")}
    result: dict[str, E2CandidateIdentity] = {}
    for cell_id in cell_ids:
        cell = known[cell_id]
        if scientific_role_for_cell(registry, cell) != "lc_candidate":
            continue
        candidate = E2CandidateIdentity.from_cell(cell, registry=registry)
        result[candidate.sha256] = candidate
    return result


def _e2_evaluations(
    candidates: dict[str, E2CandidateIdentity],
) -> tuple[E2CandidateEvaluation, ...]:
    return tuple(
        E2CandidateEvaluation(
            candidate_id=candidate_id,
            evidence_sha256=_sha(f"candidate-evidence-{candidate_id}"),
            safety_passed=True,
            confidence_pareto=True,
            lc_vs_tts_goodput_ratio=1.0 + index / 10_000,
            lc_vs_tts_confidence_lower_goodput_ratio=1.0 + index / 20_000,
            lc_vs_static_goodput_ratio=1.1 + index / 10_000,
            lc_vs_static_confidence_lower_goodput_ratio=1.05 + index / 20_000,
            hbm_bytes=1_000 + index,
            p99_itl_us=10_000 + index,
            exposed_update_us=100 + index,
            minimum_launched_updates=1,
            minimum_published_updates=1,
            safety_reason_codes=(),
        )
        for index, candidate_id in enumerate(sorted(candidates))
    )


def _stage_evidence(
    activation,
    evaluations: tuple[E2CandidateEvaluation, ...],
    *,
    registry: ExperimentRegistry,
    prior_stage_reduction_sha256: str | None = None,
    inventory: GpuInventory | None = None,
) -> E2StageEvidenceArtifact:
    stage_index = int(activation.plan.activation_round.removeprefix("halving_"))
    known = {cell.cell_id: cell for cell in registry.cells_for("E2")}
    excluded = tuple(
        cell_id
        for cell_id in activation.plan.activated_cell_ids
        if scientific_role_for_cell(registry, known[cell_id]) == "l0_naive"
    )
    completed = tuple(
        cell_id
        for cell_id in activation.plan.activated_cell_ids
        if cell_id not in excluded
    )
    bindings = tuple(
        _raw_binding(
            f"e2-{stage_index}-{index}",
            cell_id=cell_id,
            experiment="E2",
            runtime_sha256=activation.plan.runtime_sha256,
            split_sha256=activation.plan.split_sha256,
            scientific_unit=f"halving_{stage_index}",
            method=known[cell_id].identity.method,
            scientific_role=scientific_role_for_cell(registry, known[cell_id]),
        )
        for index, cell_id in enumerate(completed)
    )
    return E2StageEvidenceArtifact(
        schema_version=5,
        registry_sha256=activation.plan.registry_sha256,
        runtime_sha256=activation.plan.runtime_sha256,
        split_sha256=activation.plan.split_sha256,
        inventory_sha256=(_sha("inventory") if inventory is None else inventory.sha256),
        inventory_source_receipt_sha256=(
            _sha("inventory-source")
            if inventory is None
            else inventory.source_receipt_sha256
        ),
        fixed_instance_gpu_count=(2 if inventory is None else len(inventory.devices)),
        inventory_host_id=("host-a" if inventory is None else inventory.host_ids[0]),
        activation_sha256=activation.sha256,
        stage_index=stage_index,
        prior_stage_reduction_sha256=prior_stage_reduction_sha256,
        raw_evidence_manifest_sha256=_sha(f"raw-evidence-{stage_index}"),
        excluded_mechanism_anchor_cell_ids=excluded,
        completed_cell_ids=completed,
        terminal_receipt_sha256s=tuple(
            sorted(
                digest
                for binding in bindings
                for digest in binding.terminal_receipt_sha256s
            )
        ),
        hardware_receipt_sha256s=tuple(
            sorted(binding.hardware_receipt_sha256 for binding in bindings)
        ),
        budget_observation_sha256s=tuple(
            sorted(binding.budget_observation_sha256 for binding in bindings)
        ),
        run_bindings=bindings,
        evaluations=evaluations,
        reducer_protocol_sha256=E2_HALVING_PROTOCOL_SHA256,
        data_source="tuning_only",
        confirmation_data_visible=False,
    )


def test_e2_successive_halving_materializes_only_sealed_survivors(
    registry: ExperimentRegistry,
) -> None:
    _require_executable_e2_recipe_authority(registry)
    pareto, receipt = _e1_pareto_and_receipt(registry)
    stage_zero = reduce_e2_activation(
        registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=0,
    )
    candidates = _active_e2_candidates(registry, stage_zero.plan.activated_cell_ids)
    assert len(stage_zero.plan.activated_cell_ids) == 4 + len(candidates)
    assert all(
        1
        == sum(
            1
            for cell in registry.cells_for("E2")
            if cell.cell_id in stage_zero.plan.activated_cell_ids
            and scientific_role_for_cell(registry, cell) == "lc_candidate"
            and E2CandidateIdentity.from_cell(cell, registry=registry).sha256
            == candidate_id
        )
        for candidate_id in candidates
    )
    stage_zero_reduction = _reduce_e2_successive_halving(
        stage_zero,
        registry=registry,
        stage_evidence=_stage_evidence(
            stage_zero, _e2_evaluations(candidates), registry=registry
        ),
    )
    survivor_receipt = stage_zero_reduction.survivor_receipt
    assert survivor_receipt.status == "SURVIVORS"
    assert 0 < len(survivor_receipt.survivor_candidate_ids) < len(candidates)

    stage_one = reduce_e2_activation(
        registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=1,
        prior_reduction=stage_zero_reduction,
    )
    assert set(
        _active_e2_candidates(registry, stage_one.plan.activated_cell_ids)
    ) == set(survivor_receipt.survivor_candidate_ids)
    completed_prior = {
        row.cell_id
        for row in stage_one.dispositions
        if row.status is DispositionStatus.COMPLETED_PRIOR_ROUND
    }
    assert completed_prior == set(stage_zero.plan.activated_cell_ids)
    verify_e2_activation(
        registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=1,
        prior_reduction=stage_zero_reduction,
        artifact=stage_one,
    )
    forged_stage_one = replace(
        stage_one,
        plan=replace(
            stage_one.plan,
            source_selection_sha256=_sha("forged-stage-one-selection"),
        ),
    )
    with pytest.raises(ValueError, match="not the exact reducer-generated"):
        verify_e2_activation(
            registry,
            e1_receipt=receipt,
            pareto=pareto,
            stage_index=1,
            prior_reduction=stage_zero_reduction,
            artifact=forged_stage_one,
        )

    with pytest.raises(ValueError, match="bound to its raw reduction"):
        replace(
            stage_zero_reduction,
            survivor_receipt=replace(survivor_receipt, split_sha256=_sha("wrong")),
        )
    with pytest.raises(ValueError, match="raw terminal-evidence reducer"):
        reduce_e2_successive_halving(
            stage_zero,
            registry=registry,
            stage_evidence=_stage_evidence(
                stage_zero,
                _e2_evaluations(candidates),
                registry=registry,
            ),
        )
    with pytest.raises(ValueError, match="tuning-only"):
        replace(
            _stage_evidence(stage_zero, _e2_evaluations(candidates), registry=registry),
            confirmation_data_visible=True,
        )


def test_e2_family_floor_blocks_unsafe_family_and_never_promotes_it(
    registry: ExperimentRegistry,
) -> None:
    _require_executable_e2_recipe_authority(registry)
    pareto, receipt = _e1_pareto_and_receipt(registry)
    activation = reduce_e2_activation(
        registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=0,
    )
    candidates = _active_e2_candidates(registry, activation.plan.activated_cell_ids)
    evaluations = list(_e2_evaluations(candidates))
    blocked_family = next(iter(candidates.values())).family
    evaluations = [
        replace(
            row,
            safety_passed=False,
            confidence_pareto=False,
            safety_reason_codes=("nonfinite_update",),
        )
        if candidates[row.candidate_id].family == blocked_family
        else row
        for row in evaluations
    ]
    reduction = _reduce_e2_successive_halving(
        activation,
        registry=registry,
        stage_evidence=_stage_evidence(
            activation, tuple(evaluations), registry=registry
        ),
    )
    survivors = reduction.survivor_receipt
    assert survivors.status == "BLOCKED"
    assert survivors.survivor_candidate_ids == ()
    with pytest.raises(ValueError, match="wrong lineage or round"):
        reduce_e2_activation(
            registry,
            e1_receipt=receipt,
            pareto=pareto,
            stage_index=1,
            prior_reduction=reduction,
        )


def _final_e2_reduction(
    registry: ExperimentRegistry, *, inventory: GpuInventory | None = None
):
    _require_executable_e2_recipe_authority(registry)
    pareto, receipt = _e1_pareto_and_receipt(registry)
    reductions = []
    prior = None
    for stage_index in range(4):
        activation = reduce_e2_activation(
            registry,
            e1_receipt=receipt,
            pareto=pareto,
            stage_index=stage_index,
            prior_reduction=prior,
        )
        candidates = _active_e2_candidates(registry, activation.plan.activated_cell_ids)
        prior = _reduce_e2_successive_halving(
            activation,
            registry=registry,
            stage_evidence=_stage_evidence(
                activation,
                _e2_evaluations(candidates),
                registry=registry,
                prior_stage_reduction_sha256=(None if prior is None else prior.sha256),
                inventory=inventory,
            ),
        )
        reductions.append(prior)
    return tuple(reductions)


def test_e2_final_recipe_is_materialized_only_from_halving_3_raw_reduction(
    registry: ExperimentRegistry,
) -> None:
    reductions = _final_e2_reduction(registry)
    final = reductions[-1]
    assert final.survivor_receipt.status == "FINAL_RECIPE"
    recipe = materialize_e2_final_recipe(registry, final)
    assert recipe.final_stage_reduction_sha256 == final.sha256
    assert recipe.source_activation_sha256 == final.activation.sha256
    assert recipe.candidate_id == final.survivor_receipt.final_recipe_candidate_id
    assert recipe.candidate.sha256 == recipe.candidate_id
    with pytest.raises(ValueError, match="halving_3"):
        materialize_e2_final_recipe(registry, reductions[0])


def test_e2_final_seal_revalidates_recipe_and_raw_completion_receipts(
    registry: ExperimentRegistry,
    tmp_path: Path,
) -> None:
    inventory = _gpu_inventory()
    final = _final_e2_reduction(registry, inventory=inventory)[-1]
    evidence = final.stage_evidence
    recipe = materialize_e2_final_recipe(registry, final)
    recipe_path = tmp_path / "dflash-recipe.json"
    _write_json(recipe_path, e2_final_recipe_artifact_to_dict(recipe))
    completed_path = tmp_path / "completed.json"
    _write_json(
        completed_path,
        {
            "rows": [
                {
                    "status": "MEASURED",
                    "terminal_receipt_sha256": terminal,
                    "budget_observation_sha256": budget,
                }
                for terminal, budget in zip(
                    evidence.terminal_receipt_sha256s,
                    evidence.budget_observation_sha256s,
                    strict=True,
                )
            ]
        },
    )
    authority = {
        "registry": registry,
        "reduction": final,
        "inventory": inventory,
        "runtime_sha256": final.runtime_sha256,
        "split_sha256": final.split_sha256,
        "direct_dependency_receipt_sha256": (
            final.activation.plan.dependency_receipt_sha256
        ),
        "completed_cell_ids": final.survivor_receipt.completed_stage_cell_ids,
    }
    _validate_e2_final_seal_authority(
        **authority,
        completed_cells_path=str(completed_path),
        locked_output_paths={"dflash_recipe": str(recipe_path)},
    )

    wrong_candidate = replace(
        recipe.candidate,
        learning_rate=recipe.candidate.learning_rate * 2,
    )
    wrong_recipe = replace(
        recipe,
        candidate_id=wrong_candidate.sha256,
        candidate=wrong_candidate,
    )
    wrong_recipe_path = tmp_path / "wrong-dflash-recipe.json"
    _write_json(
        wrong_recipe_path,
        e2_final_recipe_artifact_to_dict(wrong_recipe),
    )
    with pytest.raises(ValueError, match="raw halving_3 final candidate"):
        _validate_e2_final_seal_authority(
            **authority,
            completed_cells_path=str(completed_path),
            locked_output_paths={"dflash_recipe": str(wrong_recipe_path)},
        )

    wrong_completed_path = tmp_path / "wrong-completed.json"
    wrong_rows = [
        {
            "status": "MEASURED",
            "terminal_receipt_sha256": terminal,
            "budget_observation_sha256": budget,
        }
        for terminal, budget in zip(
            evidence.terminal_receipt_sha256s,
            evidence.budget_observation_sha256s,
            strict=True,
        )
    ]
    wrong_rows[0]["terminal_receipt_sha256"] = _sha("foreign-terminal")
    _write_json(wrong_completed_path, {"rows": wrong_rows})
    with pytest.raises(ValueError, match="differs from the sealing inputs"):
        _validate_e2_final_seal_authority(
            **authority,
            completed_cells_path=str(wrong_completed_path),
            locked_output_paths={"dflash_recipe": str(recipe_path)},
        )


def test_e2_activation_blocks_before_minima_on_missing_common_load_authority(
    registry: ExperimentRegistry,
) -> None:
    pareto, receipt = _e1_pareto_and_receipt(registry)
    activation = reduce_e2_activation(
        registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=0,
    )
    assert activation.plan.status == "BLOCKED"
    assert activation.plan.activated_cell_ids == ()
    assert activation.plan.reason_code == E1_COMMON_LOAD_AUTHORITY_UNREGISTERED_REASON
    assert activation.plan.blocked_cell_ids
    assert E1_COMMON_LOAD_AUTHORITY_UNREGISTERED_REASON in {
        row.reason_code
        for row in activation.dispositions
        if row.status is DispositionStatus.BLOCKED
    }
    stage_zero_cells = tuple(
        cell
        for cell in registry.cells_for("E2")
        if "halving_stage=0:" in cell.identity.variant
    )
    assert Counter(
        scientific_role_for_cell(registry, cell) for cell in stage_zero_cells
    ) == {
        "target_only": 1,
        "static": 1,
        "tts": 1,
        "l0_naive": 1,
        "lc_candidate": 2_976,
    }
    dispositions = {row.cell_id: row for row in activation.dispositions}
    frozen_anchor_rows = tuple(
        dispositions[cell.cell_id]
        for cell in stage_zero_cells
        if scientific_role_for_cell(registry, cell) in {"tts", "l0_naive"}
    )
    assert all(row.status is DispositionStatus.BLOCKED for row in frozen_anchor_rows)
    assert all(
        row.reason_code == "tts_official_recipe_unavailable"
        for row in frozen_anchor_rows
    )

    # Common-load authority is the earlier unresolved dependency; promotion
    # minima cannot become the observable blocker until that reducer exists.
    assert len(_E2_PROMOTION_MINIMA) == 4
    assert all(
        row.stage_index == stage_index
        and row.minimum_launched_updates_per_adapted_method is None
        and row.minimum_published_updates_per_adapted_method is None
        and row.blocker_reason_code == "e2_promotion_minima_unregistered"
        for stage_index, row in enumerate(_E2_PROMOTION_MINIMA)
    )
    assert len(E2_PROMOTION_MINIMA_AUTHORITY_SHA256) == 64


def test_confirmation_templates_fill_the_lightcone_slot_but_cannot_activate(
    registry: ExperimentRegistry,
) -> None:
    family = _family(registry, context=1_024)
    pilots = materialize_confirmation_pilots(registry, family)
    cells_by_id = {cell.cell_id: cell for cell in registry.cells_for("E3b")}

    # The template is a structural member of every five-method paired block,
    # while the registry keeps it separate from the reportable LightCone role.
    for block in PILOT_BLOCKS:
        roles = {
            (
                "lightcone"
                if scientific_role_for_cell(registry, cells_by_id[row.cell_id])
                == "lightcone_template"
                else scientific_role_for_cell(registry, cells_by_id[row.cell_id])
            )
            for row in pilots.dispositions
            if cells_by_id[row.cell_id].identity.block == block
        }
        assert roles == set(CONFIRMATION_METHOD_ROLES)

    template_rows = tuple(
        row
        for row in pilots.dispositions
        if scientific_role_for_cell(registry, cells_by_id[row.cell_id])
        == "lightcone_template"
    )
    assert template_rows
    assert all(row.status is DispositionStatus.BLOCKED for row in template_rows)
    assert all(
        row.reason_code == "sealed_e2_recipe_receipt_required" for row in template_rows
    )
    assert not pilots.activated_cell_ids


def test_e2_common_load_block_precedes_untrusted_later_stage_cargo(
    registry: ExperimentRegistry,
) -> None:
    pareto, receipt = _e1_pareto_and_receipt(registry)
    with pytest.raises(ValueError, match="wrong lineage or round"):
        reduce_e2_activation(
            registry,
            e1_receipt=receipt,
            pareto=pareto,
            stage_index=3,
            prior_reduction=object(),  # type: ignore[arg-type]
        )


def test_e2_common_load_block_rebuilds_mutated_pareto_before_use(
    registry: ExperimentRegistry,
) -> None:
    pareto, receipt = _e1_pareto_and_receipt(registry)
    _ = pareto.sha256
    object.__setattr__(pareto, "schema_version", 1)
    with pytest.raises(ValueError, match="Pareto schema version 2"):
        reduce_e2_activation(
            registry,
            e1_receipt=receipt,
            pareto=pareto,
            stage_index=0,
        )


@pytest.mark.parametrize(
    ("stage_index", "status"),
    ((0, "SURVIVORS"), (1, "SURVIVORS"), (2, "SURVIVORS"), (3, "FINAL_RECIPE")),
)
def test_e2_unregistered_promotion_minima_reject_forged_survivor_receipt(
    stage_index: int,
    status: str,
) -> None:
    candidate_id = _sha(f"forged-e2-survivor-{stage_index}")
    completed_cell_id = _sha(f"forged-e2-completed-cell-{stage_index}")
    with pytest.raises(
        ValueError,
        match="unregistered E2 promotion minima categorically forbid survivors",
    ):
        E2SurvivorReceipt(
            schema_version=2,
            registry_sha256=_sha("e2-registry"),
            runtime_sha256=_sha("e2-runtime"),
            split_sha256=_sha("e2-split"),
            halving_protocol_sha256=E2_HALVING_PROTOCOL_SHA256,
            stage_index=stage_index,
            source_activation_sha256=_sha("e2-activation"),
            prior_stage_reduction_sha256=(
                None if stage_index == 0 else _sha(f"e2-prior-stage-{stage_index}")
            ),
            completed_cells_sha256=content_sha256((completed_cell_id,)),
            completed_stage_cell_ids=(completed_cell_id,),
            completed_lineage_cell_ids=(completed_cell_id,),
            tuning_evidence_sha256=_sha("e2-tuning-evidence"),
            source_candidate_ids=(candidate_id,),
            survivor_candidate_ids=(candidate_id,),
            final_recipe_candidate_id=(
                candidate_id if status == "FINAL_RECIPE" else None
            ),
            status=status,
            reason_code=(
                "registered_final_recipe_locked"
                if status == "FINAL_RECIPE"
                else "registered_quarter_retention_with_family_floor"
            ),
            selection_state="sealed_before_next_stage_unblinding",
        )


def test_e2_final_materializer_replays_minima_after_memory_mutation(
    registry: ExperimentRegistry,
) -> None:
    candidate = next(
        cell
        for cell in registry.cells_for("E2")
        if scientific_role_for_cell(registry, cell) == "lc_candidate"
        and "halving_stage=3:" in cell.identity.variant
    )
    candidate_id = _sha("mutated-final-candidate")
    receipt = E2SurvivorReceipt(
        schema_version=2,
        registry_sha256=registry.sha256,
        runtime_sha256=_sha("mutated-final-runtime"),
        split_sha256=_sha("mutated-final-split"),
        halving_protocol_sha256=E2_HALVING_PROTOCOL_SHA256,
        stage_index=3,
        source_activation_sha256=_sha("mutated-final-activation"),
        prior_stage_reduction_sha256=_sha("mutated-final-prior"),
        completed_cells_sha256=content_sha256((candidate.cell_id,)),
        completed_stage_cell_ids=(candidate.cell_id,),
        completed_lineage_cell_ids=(candidate.cell_id,),
        tuning_evidence_sha256=_sha("mutated-final-evidence"),
        source_candidate_ids=(candidate_id,),
        survivor_candidate_ids=(),
        final_recipe_candidate_id=None,
        status="BLOCKED",
        reason_code="e2_promotion_minima_unregistered",
        selection_state="sealed_before_next_stage_unblinding",
    )
    object.__setattr__(receipt, "status", "FINAL_RECIPE")
    object.__setattr__(receipt, "survivor_candidate_ids", (candidate_id,))
    object.__setattr__(receipt, "final_recipe_candidate_id", candidate_id)
    object.__setattr__(receipt, "reason_code", "registered_final_recipe_locked")
    fake_reduction = object.__new__(E2StageReductionArtifact)
    object.__setattr__(fake_reduction, "schema_version", 1)
    object.__setattr__(fake_reduction, "survivor_receipt", receipt)
    with pytest.raises(ValueError, match="e2_promotion_minima_unregistered"):
        materialize_e2_final_recipe(registry, fake_reduction)


def _family(registry: ExperimentRegistry, *, context: int):
    cell = next(
        cell
        for cell in registry.cells_for("E3b")
        if cell.identity.context == context and cell.identity.method == "static"
    )
    return derive_confirmation_family(
        registry,
        cell_id=cell.cell_id,
        runtime_sha256=_sha("confirmation-runtime"),
        split_sha256=_sha("confirmation-split"),
        trace_sha256=_sha("confirmation-trace"),
        sampling_sha256=_sha("confirmation-sampling"),
        hardware_envelope_sha256=_sha("confirmation-hardware"),
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


def _summary_only_family_reduction(
    *,
    registry: ExperimentRegistry,
    family,
    pilots,
    power_sizing,
    evidence_label: str,
) -> ConfirmationFamilyPowerReductionArtifact:
    """Stop until a path-replayed E2 seal can bind the LightCone pilot rows."""

    del registry, family, pilots, power_sizing, evidence_label
    pytest.skip(
        "formal confirmation planning requires a path-replayed E2 LightCone "
        "seal, which is unavailable in this release"
    )


def test_family_level_pilots_power_and_exact_final_prefix(
    registry: ExperimentRegistry,
) -> None:
    family = _family(registry, context=1_024)
    pilots = materialize_confirmation_pilots(registry, family)
    if not pilots.activated_cell_ids:
        pytest.skip("confirmation pilots await frozen TTS and sealed LightCone recipes")
    assert len(pilots.activated_cell_ids) == 4 * len(CONFIRMATION_METHOD_ROLES) == 20
    assert {
        cell.identity.block
        for cell in registry.cells_for("E3b")
        if cell.cell_id in pilots.activated_cell_ids
    } == set(PILOT_BLOCKS)

    reduction = _summary_only_family_reduction(
        registry=registry,
        family=family,
        pilots=pilots,
        power_sizing=_power_sizing(family),
        evidence_label="family-pilot-evidence",
    )
    plan = reduction.plan
    assert plan.status == "POWERED"
    assert plan.selected_final_blocks == 12
    final = materialize_confirmation_prefix(
        registry,
        family=family,
        reduction=reduction,
        pilot_activation=pilots,
    )
    assert len(final.activated_cell_ids) == 12 * len(CONFIRMATION_METHOD_ROLES) == 60
    assert {
        cell.identity.block
        for cell in registry.cells_for("E3b")
        if cell.cell_id in final.activated_cell_ids
    } == set(FINAL_BLOCKS[:12])
    assert sum(
        row.status is DispositionStatus.DEFERRED for row in final.dispositions
    ) == 8 * len(CONFIRMATION_METHOD_ROLES)

    other_family = _family(registry, context=2_048)
    with pytest.raises(ValueError, match="raw terminal-evidence reducer"):
        seal_confirmation_family_power(
            registry=registry,
            family=other_family,
            pilot_activation=pilots,
            completed_pilot_cell_ids=pilots.activated_cell_ids,
            pilot_evidence_sha256=_sha("cross-family-evidence"),
            power_sizing=_power_sizing(family),
            confirmation_data_visible=False,
        )
    with pytest.raises(ValueError, match="differs from power sizing"):
        replace(plan, selected_final_blocks=20, selected_final_prefix=FINAL_BLOCKS)

    sizing_20 = replace(
        _power_sizing(family),
        selected_final_blocks=20,
        power_grid=tuple(
            replace(cell, power=0.79 if cell.final_blocks < 20 else 0.80)
            for cell in _power_sizing(family).power_grid
        ),
    )
    reduction_20 = _summary_only_family_reduction(
        registry=registry,
        family=family,
        pilots=pilots,
        power_sizing=sizing_20,
        evidence_label="family-pilot-evidence-20",
    )
    plan_20 = reduction_20.plan
    final_20 = materialize_confirmation_prefix(
        registry,
        family=family,
        reduction=reduction_20,
        pilot_activation=pilots,
    )
    assert plan_20.selected_final_blocks == 20
    assert (
        len(final_20.activated_cell_ids) == 20 * len(CONFIRMATION_METHOD_ROLES) == 100
    )
    assert not any(
        row.status is DispositionStatus.DEFERRED for row in final_20.dispositions
    )


def _semantics(**changes: object) -> ExecutionSemanticsIdentity:
    values: dict[str, object] = {
        "target_model": "Qwen/Qwen3-8B",
        "target_revision": "revision-1",
        "runtime_sha256": _sha("alias-runtime"),
        "patched_tree_identity": PINNED_SGLANG_TREE,
        "sampling_sha256": _sha("alias-sampling"),
        "seed": 7,
        "request_corpus_sha256": _sha("alias-corpus"),
        "arrival_trace_sha256": _sha("alias-trace"),
        "maximum_context_tokens": 40_928,
        "maximum_output_tokens": 256,
        "hardware_envelope_sha256": _sha("alias-hardware"),
        "topology": "tp1_dp1",
        "rank_layout_sha256": _sha("alias-ranks"),
        "method": "target_only",
        "method_implementation_sha256": _sha("target-only-implementation"),
        "server_config_sha256": _sha("alias-server"),
        "evidence_schema": "schema_v3",
        "output_token_contract_sha256": _sha("alias-output"),
        "timing_contract_sha256": _sha("alias-timing"),
    }
    values.update(changes)
    return ExecutionSemanticsIdentity(**values)  # type: ignore[arg-type]


def _alias_candidate(
    label: str, backend_label: str, semantics: ExecutionSemanticsIdentity
) -> EvidenceAliasCandidate:
    return EvidenceAliasCandidate(
        cell_id=_sha(f"alias-cell-{label}"),
        semantics=semantics,
        presentation_axes=(
            PresentationAxis("analysis_panel", "core"),
            PresentationAxis("backend_label", backend_label),
        ),
    )


def test_evidence_alias_is_exact_and_analysis_preserves_dependence() -> None:
    semantics = _semantics()
    source = _alias_candidate("source", "DFLASH", semantics)
    target = _alias_candidate("target", "DSPARK", semantics)
    alias = EvidenceAliasReceipt(
        schema_version=1,
        source=source,
        target=target,
        source_evidence_sha256=_sha("source-terminal-evidence"),
        removed_presentation_axis="backend_label",
        reason_code="target_only_backend_label_only",
        analysis_state="sealed_before_analysis",
    )
    with pytest.raises(TypeError, match="first-party"):
        build_evidence_dependence_map(
            direct_observation_cell_ids=(source.cell_id,),
            aliases=(alias,),  # type: ignore[arg-type]
        )

    changed = _alias_candidate(
        "changed",
        "DSPARK",
        _semantics(arrival_trace_sha256=_sha("different-arrival")),
    )
    with pytest.raises(ValueError, match="not content-identical"):
        EvidenceAliasReceipt(
            schema_version=1,
            source=source,
            target=changed,
            source_evidence_sha256=_sha("source-terminal-evidence"),
            removed_presentation_axis="backend_label",
            reason_code="target_only_backend_label_only",
            analysis_state="sealed_before_analysis",
        )
    with pytest.raises(ValueError, match="only Target-only"):
        _semantics(method="static")
    with pytest.raises(TypeError, match="first-party"):
        build_evidence_dependence_map(
            direct_observation_cell_ids=(source.cell_id, target.cell_id),
            aliases=(alias,),  # type: ignore[arg-type]
        )
