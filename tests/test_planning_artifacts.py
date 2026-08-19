from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace
from functools import cache

import pytest

from lightcone_spec.experiments.planning import (
    E2_HALVING_PROTOCOL_SHA256,
    AnalysisDependenceUnit,
    BudgetGroupTotal,
    BudgetInventoryIdentity,
    BudgetJobKind,
    CellDisposition,
    ConfirmationFamilyIdentity,
    ConfirmationFamilyPowerPlan,
    DispositionStatus,
    E1GeometryIdentity,
    E1ParetoArtifact,
    E2CandidateEvaluation,
    E2CandidateIdentity,
    E2FinalRecipeArtifact,
    E2StageEvidenceArtifact,
    E2StageReductionArtifact,
    E2SurvivorReceipt,
    EvidenceAliasCandidate,
    EvidenceAliasReceipt,
    EvidenceDependenceMap,
    ExactScenarioHours,
    ExecutionSemanticsIdentity,
    ExpectedMaximumCount,
    ExperimentBudget,
    FamilyActivationArtifact,
    IndustrialBudgetReport,
    P99AnchorStatus,
    PresentationAxis,
    RawEvidenceRunBinding,
    ReducerActivationArtifact,
    ScenarioMilliseconds,
    SealedE3aSelection,
    family_pilot_block_id,
)
from lightcone_spec.experiments.planning_artifacts import (
    PlanningArtifactSidecar,
    budget_inventory_identity_from_dict,
    budget_inventory_identity_to_dict,
    confirmation_family_identity_from_dict,
    confirmation_family_identity_to_dict,
    confirmation_family_power_plan_from_dict,
    confirmation_family_power_plan_to_dict,
    e1_pareto_artifact_from_dict,
    e1_pareto_artifact_to_dict,
    e2_final_recipe_artifact_from_dict,
    e2_final_recipe_artifact_to_dict,
    e2_stage_evidence_artifact_from_dict,
    e2_stage_evidence_artifact_to_dict,
    e2_stage_reduction_artifact_from_dict,
    e2_stage_reduction_artifact_to_dict,
    e2_survivor_receipt_from_dict,
    e2_survivor_receipt_to_dict,
    evidence_alias_receipt_from_dict,
    evidence_alias_receipt_to_dict,
    evidence_dependence_map_from_dict,
    evidence_dependence_map_to_dict,
    experiment_budget_from_dict,
    experiment_budget_sequence_from_dict,
    experiment_budget_sequence_to_dict,
    experiment_budget_to_dict,
    family_activation_artifact_from_dict,
    family_activation_artifact_to_dict,
    industrial_budget_report_from_dict,
    industrial_budget_report_to_dict,
    reducer_activation_artifact_from_dict,
    reducer_activation_artifact_to_dict,
    sealed_e3a_selection_from_dict,
    sealed_e3a_selection_to_dict,
)
from lightcone_spec.experiments.registry import (
    CONFIRMATION_METHOD_ROLES,
    FINAL_BLOCKS,
    PILOT_BLOCKS,
    StageActivationPlan,
    WorkloadClass,
    content_sha256,
    scientific_role_for_cell,
)
from lightcone_spec.experiments.registry import (
    build_legacy_industrial_registry as build_industrial_registry,
)
from lightcone_spec.experiments.statistics import (
    MAXIMUM_FINAL_BLOCKS,
    MINIMUM_FINAL_BLOCKS,
    PRIMARY_CONTRASTS,
    PRIMARY_FAMILY_ALPHA,
    PRIMARY_MINIMUM_RELATIVE_EFFECT,
    PRIMARY_TARGET_POWER,
    ContrastPower,
    PowerSizingPlan,
)

_COMPONENTS = (
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


def _sha(label: str) -> str:
    return content_sha256({"test": label})


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


def test_raw_run_binding_rejects_unsealed_lightcone_label() -> None:
    with pytest.raises(ValueError, match="path-replayed E2 seal"):
        _raw_binding(
            "unsealed-lightcone",
            cell_id=_sha("unsealed-lightcone-cell"),
            experiment="E3b",
            runtime_sha256=_sha("runtime"),
            split_sha256=_sha("split"),
            scientific_unit="final_0",
            method="l0",
            scientific_role="lightcone",
        )


def _ms(value: int) -> ScenarioMilliseconds:
    return ScenarioMilliseconds(value, value, value)


def _budget(label: str) -> ExperimentBudget:
    zero = _ms(0)
    return ExperimentBudget(
        schema_version=1,
        cell_id=_sha(f"cell-{label}"),
        experiment="E1",
        method="static",
        workload_class=WorkloadClass.TUNING,
        job_kind=BudgetJobKind.STANDARD,
        startup_model_load=_ms(10),
        compile_jit_graph_prewarm=_ms(20),
        excluded_warmup=_ms(30),
        excluded_warmup_requests=ExpectedMaximumCount(2, 3),
        scored_arrival=_ms(40),
        request_deadline=_ms(100),
        drain=_ms(5),
        reset_finalization=_ms(5),
        evidence_flush_shutdown=_ms(5),
        output_tokens=ExpectedMaximumCount(40, 60),
        minimum_completed_requests=1,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=zero,
        failure_injection=zero,
        retry=_ms(5),
        retry_allowance=1,
        profiler=zero,
        download_compile_reservation=zero,
        gpu_count=1,
        topology="tp1_dp1",
        reserved_gpu_ms=_ms(120),
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=_ms(120),
    )


def _inventory() -> BudgetInventoryIdentity:
    return BudgetInventoryIdentity(
        schema_version=1,
        host_sha256=_sha("host"),
        gpu_uuids=("GPU-test",),
        topology_sha256=_sha("topology"),
    )


def _budget_report() -> IndustrialBudgetReport:
    budget = _budget("report")
    components = tuple((name, getattr(budget, name)) for name in _COMPONENTS)
    statuses = (
        ("not_required", 0),
        ("required_unresolved", 0),
        ("locked", 1),
    )
    group = BudgetGroupTotal(
        experiment=budget.experiment,
        method=budget.method,
        workload_class=budget.workload_class.value,
        topology=budget.topology,
        cells=1,
        gpu_cell_units=1,
        component_ms=components,
        request_deadline_ms=budget.request_deadline,
        excluded_warmup_requests=budget.excluded_warmup_requests,
        output_tokens=budget.output_tokens,
        minimum_completed_requests=budget.minimum_completed_requests,
        retry_allowance=budget.retry_allowance,
        p99_anchor_status_counts=statuses,
        compute_gpu_ms=budget.compute_gpu_ms,
        compute_gpu_hours=ExactScenarioHours.from_milliseconds(budget.compute_gpu_ms),
        reserved_gpu_ms=budget.reserved_gpu_ms,
        reserved_gpu_hours=ExactScenarioHours.from_milliseconds(budget.reserved_gpu_ms),
        fixed_instance_billed_gpu_ms=budget.fixed_instance_billed_gpu_ms,
        fixed_instance_billed_gpu_hours=ExactScenarioHours.from_milliseconds(
            budget.fixed_instance_billed_gpu_ms
        ),
    )
    return IndustrialBudgetReport(
        schema_version=1,
        registry_sha256=_sha("registry"),
        activation_sha256=_sha("activation"),
        inventory=_inventory(),
        budget_sha256s=(budget.sha256,),
        groups=(group,),
        cells=1,
        gpu_cell_units=1,
        component_ms=components,
        request_deadline_ms=budget.request_deadline,
        excluded_warmup_requests=budget.excluded_warmup_requests,
        output_tokens=budget.output_tokens,
        minimum_completed_requests=budget.minimum_completed_requests,
        retry_allowance=budget.retry_allowance,
        p99_anchor_status_counts=statuses,
        compute_gpu_ms=budget.compute_gpu_ms,
        compute_gpu_hours=ExactScenarioHours.from_milliseconds(budget.compute_gpu_ms),
        reserved_gpu_ms=budget.reserved_gpu_ms,
        reserved_gpu_hours=ExactScenarioHours.from_milliseconds(budget.reserved_gpu_ms),
        fixed_instance_billed_gpu_ms=budget.fixed_instance_billed_gpu_ms,
        fixed_instance_billed_gpu_hours=ExactScenarioHours.from_milliseconds(
            budget.fixed_instance_billed_gpu_ms
        ),
        estimated_wall_ms=budget.wall_time,
        estimated_wall_hours=ExactScenarioHours.from_milliseconds(budget.wall_time),
        schedule_fixed_instance_billed_gpu_ms=budget.wall_time,
        schedule_fixed_instance_billed_gpu_hours=ExactScenarioHours.from_milliseconds(
            budget.wall_time
        ),
        unresolved_assumptions=(),
        scheduler_gpu_inventory_sha256=_sha("scheduler-inventory"),
        interference_envelope_sha256=_sha("interference-envelope"),
    )


def _selection() -> SealedE3aSelection:
    return SealedE3aSelection(
        schema_version=1,
        registry_sha256=_sha("registry"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        width=8,
        concurrency=4,
        reducer_evidence_sha256=_sha("selection-evidence"),
    )


def _activation() -> ReducerActivationArtifact:
    statuses = (
        DispositionStatus.ACTIVATED,
        DispositionStatus.BLOCKED,
        DispositionStatus.NOT_APPLICABLE,
        DispositionStatus.DEFERRED,
        DispositionStatus.COMPLETED_PRIOR_ROUND,
    )
    rows = tuple(
        sorted(
            [
                CellDisposition(
                    cell_id=_sha(f"activation-{status.value}"),
                    status=status,
                    reason_code=f"reason_{index}",
                )
                for index, status in enumerate(statuses)
            ]
            + [
                CellDisposition(
                    cell_id=_sha(f"activation-activated-{index}"),
                    status=DispositionStatus.ACTIVATED,
                    reason_code="additional_e2_candidate_member",
                )
                for index in range(1, 5)
            ],
            key=lambda row: row.cell_id,
        )
    )
    by_status = {
        status: tuple(row.cell_id for row in rows if row.status is status)
        for status in statuses
    }
    plan = StageActivationPlan(
        registry_sha256=_sha("registry"),
        experiment="E2",
        dependency_receipt_sha256=_sha("dependency"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        source_selection_sha256=_sha("source-selection"),
        activation_round="halving_0",
        status="AVAILABLE",
        activated_cell_ids=by_status[DispositionStatus.ACTIVATED],
        not_applicable_cell_ids=tuple(
            sorted(
                by_status[DispositionStatus.NOT_APPLICABLE]
                + by_status[DispositionStatus.COMPLETED_PRIOR_ROUND]
            )
        ),
        blocked_cell_ids=by_status[DispositionStatus.BLOCKED],
        deferred_cell_ids=by_status[DispositionStatus.DEFERRED],
        reason_code="successive_halving_reducer_activation",
    )
    return ReducerActivationArtifact(
        schema_version=1,
        plan=plan,
        reducer_protocol_sha256=E2_HALVING_PROTOCOL_SHA256,
        dispositions=rows,
    )


def _pareto() -> E1ParetoArtifact:
    return E1ParetoArtifact(
        schema_version=2,
        registry_sha256=_sha("registry"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        e1_activation_sha256=_sha("e1-activation"),
        reducer_evidence_sha256=_sha("e1-evidence"),
        surviving_geometries=(
            E1GeometryIdentity(
                scope="native_heads",
                parameterization="lora",
                rank=8,
                alpha_over_rank=1.0,
            ),
        ),
        selection_state="sealed_before_e2_unblinding",
    )


def _e2_stage_evidence() -> E2StageEvidenceArtifact:
    activation = _activation()
    completed = activation.plan.activated_cell_ids[:4]
    method_roles = (
        ("target_only", "target_only"),
        ("static", "static"),
        ("tts", "tts"),
        ("l0", "lc_candidate"),
    )
    bindings = tuple(
        _raw_binding(
            f"e2-{index}",
            cell_id=cell_id,
            experiment="E2",
            runtime_sha256=activation.plan.runtime_sha256,
            split_sha256=activation.plan.split_sha256,
            scientific_unit="halving_0",
            method=method,
            scientific_role=role,
        )
        for index, (cell_id, (method, role)) in enumerate(
            zip(completed, method_roles, strict=True)
        )
    )
    evaluation = E2CandidateEvaluation(
        candidate_id=_sha("candidate"),
        evidence_sha256=_sha("candidate-evidence"),
        safety_passed=True,
        confidence_pareto=True,
        lc_vs_tts_goodput_ratio=1.05,
        lc_vs_tts_confidence_lower_goodput_ratio=1.04,
        lc_vs_static_goodput_ratio=1.06,
        lc_vs_static_confidence_lower_goodput_ratio=1.03,
        hbm_bytes=100,
        p99_itl_us=200,
        exposed_update_us=300,
        minimum_launched_updates=1,
        minimum_published_updates=1,
        safety_reason_codes=(),
    )
    return E2StageEvidenceArtifact(
        schema_version=5,
        registry_sha256=_sha("registry"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        inventory_sha256=_sha("inventory"),
        inventory_source_receipt_sha256=_sha("inventory-source"),
        fixed_instance_gpu_count=2,
        inventory_host_id="host-a",
        activation_sha256=activation.sha256,
        stage_index=0,
        prior_stage_reduction_sha256=None,
        raw_evidence_manifest_sha256=_sha("raw-evidence"),
        excluded_mechanism_anchor_cell_ids=(activation.plan.activated_cell_ids[-1],),
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
        evaluations=(evaluation,),
        reducer_protocol_sha256=E2_HALVING_PROTOCOL_SHA256,
        data_source="tuning_only",
        confirmation_data_visible=False,
    )


def _e2_survivor() -> E2SurvivorReceipt:
    evidence = _e2_stage_evidence()
    stage_cells = evidence.completed_cell_ids
    candidate = _sha("candidate")
    return E2SurvivorReceipt(
        schema_version=2,
        registry_sha256=_sha("registry"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        halving_protocol_sha256=E2_HALVING_PROTOCOL_SHA256,
        stage_index=0,
        source_activation_sha256=_activation().sha256,
        prior_stage_reduction_sha256=None,
        completed_cells_sha256=content_sha256(stage_cells),
        completed_stage_cell_ids=stage_cells,
        completed_lineage_cell_ids=stage_cells,
        tuning_evidence_sha256=evidence.sha256,
        source_candidate_ids=(candidate,),
        survivor_candidate_ids=(),
        final_recipe_candidate_id=None,
        status="BLOCKED",
        reason_code="e2_promotion_minima_unregistered",
        selection_state="sealed_before_next_stage_unblinding",
    )


def _e2_reduction() -> E2StageReductionArtifact:
    return E2StageReductionArtifact(
        schema_version=1,
        activation=_activation(),
        stage_evidence=_e2_stage_evidence(),
        survivor_receipt=_e2_survivor(),
    )


@cache
def _e2_final_recipe() -> E2FinalRecipeArtifact:
    registry = build_industrial_registry()
    cell = next(
        cell
        for cell in registry.cells_for("E2")
        if scientific_role_for_cell(registry, cell) == "lc_candidate"
        and cell.identity.learning_rate is not None
    )
    candidate = E2CandidateIdentity.from_cell(cell, registry=registry)
    recipe = registry.adaptation_recipe_for_cell(cell)
    return E2FinalRecipeArtifact(
        schema_version=3,
        registry_sha256=registry.sha256,
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        final_stage_reduction_sha256=_sha("final-reduction"),
        source_activation_sha256=_sha("final-activation"),
        candidate_id=candidate.sha256,
        candidate=candidate,
        recipe_sha256=recipe.sha256,
        recipe=recipe,
        selection_state="locked_from_raw_halving_3",
    )


def _family() -> ConfirmationFamilyIdentity:
    return ConfirmationFamilyIdentity(
        schema_version=1,
        registry_sha256=_sha("registry"),
        experiment="E3b",
        model="target-model",
        backend="DFLASH",
        task="summarization",
        context=4096,
        regime="medium",
        arrival="poisson",
        load_arrival_sha256=_sha("load-arrival"),
        width_panel="matched",
        topology="tp1_dp1",
        cohort_family="paired",
        cohort_count=1,
        method_family=CONFIRMATION_METHOD_ROLES,
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        trace_sha256=_sha("trace"),
        sampling_sha256=_sha("sampling"),
        hardware_envelope_sha256=_sha("hardware"),
    )


def _family_activation() -> FamilyActivationArtifact:
    family = _family()
    return FamilyActivationArtifact(
        schema_version=1,
        family=family,
        activation_round="excluded_pilots",
        power_plan_sha256=None,
        dispositions=(
            CellDisposition(
                cell_id=_sha("family-pilot-cell"),
                status=DispositionStatus.ACTIVATED,
                reason_code="family_excluded_pilot",
            ),
        ),
    )


def _family_power() -> ConfirmationFamilyPowerPlan:
    family = _family()
    selected = MINIMUM_FINAL_BLOCKS
    power_sizing = PowerSizingPlan(
        status="READY",
        pilot_block_ids=tuple(
            family_pilot_block_id(family, block) for block in PILOT_BLOCKS
        ),
        selected_final_blocks=selected,
        minimum_final_blocks=MINIMUM_FINAL_BLOCKS,
        maximum_final_blocks=MAXIMUM_FINAL_BLOCKS,
        target_power=PRIMARY_TARGET_POWER,
        family_alpha=PRIMARY_FAMILY_ALPHA,
        adjusted_alpha=PRIMARY_FAMILY_ALPHA / len(PRIMARY_CONTRASTS),
        minimum_relative_effect=PRIMARY_MINIMUM_RELATIVE_EFFECT,
        minimum_log_effect=math.log1p(PRIMARY_MINIMUM_RELATIVE_EFFECT),
        pilot_log_standard_deviations=tuple(
            (contrast, 0.1) for contrast in PRIMARY_CONTRASTS
        ),
        power_grid=tuple(
            ContrastPower(contrast=contrast, final_blocks=block, power=0.9)
            for block in range(MINIMUM_FINAL_BLOCKS, MAXIMUM_FINAL_BLOCKS + 1)
            for contrast in PRIMARY_CONTRASTS
        ),
    )
    return ConfirmationFamilyPowerPlan(
        schema_version=1,
        family=family,
        pilot_activation_sha256=_sha("pilot-activation"),
        completed_pilot_cells_sha256=content_sha256(
            tuple(sorted(_sha(f"family-cell-{index}") for index in range(20)))
        ),
        pilot_evidence_sha256=_sha("pilot-evidence"),
        power_sizing=power_sizing,
        status="POWERED",
        selected_final_blocks=selected,
        selected_final_prefix=FINAL_BLOCKS[:selected],
        reason_code="registered_family_power_target_met",
        selection_state="sealed_before_confirmation_unblinding",
    )


def _alias() -> EvidenceAliasReceipt:
    semantics = ExecutionSemanticsIdentity(
        target_model="target-model",
        target_revision="revision",
        runtime_sha256=_sha("runtime"),
        patched_tree_identity="patched-tree",
        sampling_sha256=_sha("sampling"),
        seed=7,
        request_corpus_sha256=_sha("corpus"),
        arrival_trace_sha256=_sha("arrival"),
        maximum_context_tokens=4096,
        maximum_output_tokens=256,
        hardware_envelope_sha256=_sha("hardware"),
        topology="tp1_dp1",
        rank_layout_sha256=_sha("rank-layout"),
        method="target_only",
        method_implementation_sha256=_sha("implementation"),
        server_config_sha256=_sha("server-config"),
        evidence_schema="schema.v1",
        output_token_contract_sha256=_sha("token-contract"),
        timing_contract_sha256=_sha("timing-contract"),
    )
    source = EvidenceAliasCandidate(
        cell_id=_sha("alias-source"),
        semantics=semantics,
        presentation_axes=(
            PresentationAxis("analysis_panel", "primary"),
            PresentationAxis("backend_label", "DFLASH"),
        ),
    )
    target = EvidenceAliasCandidate(
        cell_id=_sha("alias-target"),
        semantics=semantics,
        presentation_axes=(
            PresentationAxis("analysis_panel", "primary"),
            PresentationAxis("backend_label", "NONE"),
        ),
    )
    return EvidenceAliasReceipt(
        schema_version=1,
        source=source,
        target=target,
        source_evidence_sha256=_sha("source-evidence"),
        removed_presentation_axis="backend_label",
        reason_code="target_only_backend_label_only",
        analysis_state="sealed_before_analysis",
    )


def _dependence_map() -> EvidenceDependenceMap:
    alias = _alias()
    unit = AnalysisDependenceUnit(
        unit_sha256=alias.dependence_unit_sha256,
        source_cell_id=alias.source.cell_id,
        member_cell_ids=tuple(sorted((alias.source.cell_id, alias.target.cell_id))),
    )
    return EvidenceDependenceMap(
        schema_version=1,
        units=(unit,),
    )


def test_every_cli_planning_artifact_round_trips_as_json() -> None:
    cases = (
        (_budget("single"), experiment_budget_to_dict, experiment_budget_from_dict),
        (
            _inventory(),
            budget_inventory_identity_to_dict,
            budget_inventory_identity_from_dict,
        ),
        (
            _budget_report(),
            industrial_budget_report_to_dict,
            industrial_budget_report_from_dict,
        ),
        (
            _selection(),
            sealed_e3a_selection_to_dict,
            sealed_e3a_selection_from_dict,
        ),
        (
            _activation(),
            reducer_activation_artifact_to_dict,
            reducer_activation_artifact_from_dict,
        ),
        (_pareto(), e1_pareto_artifact_to_dict, e1_pareto_artifact_from_dict),
        (
            _e2_final_recipe(),
            e2_final_recipe_artifact_to_dict,
            e2_final_recipe_artifact_from_dict,
        ),
        (
            _e2_stage_evidence(),
            e2_stage_evidence_artifact_to_dict,
            e2_stage_evidence_artifact_from_dict,
        ),
        (
            _e2_survivor(),
            e2_survivor_receipt_to_dict,
            e2_survivor_receipt_from_dict,
        ),
        (
            _e2_reduction(),
            e2_stage_reduction_artifact_to_dict,
            e2_stage_reduction_artifact_from_dict,
        ),
        (
            _family(),
            confirmation_family_identity_to_dict,
            confirmation_family_identity_from_dict,
        ),
        (
            _family_activation(),
            family_activation_artifact_to_dict,
            family_activation_artifact_from_dict,
        ),
        (
            _family_power(),
            confirmation_family_power_plan_to_dict,
            confirmation_family_power_plan_from_dict,
        ),
        (_alias(), evidence_alias_receipt_to_dict, evidence_alias_receipt_from_dict),
        (
            _dependence_map(),
            evidence_dependence_map_to_dict,
            evidence_dependence_map_from_dict,
        ),
    )
    for artifact, encode, decode in cases:
        wire = encode(artifact)
        parsed = json.loads(
            json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        sidecar = PlanningArtifactSidecar(
            schema_version=1,
            artifact_kind=wire["artifact_kind"],
            artifact_sha256=wire["artifact_sha256"],
        )
        assert decode(parsed, sidecar=sidecar.to_dict()) == artifact


def test_budget_sequence_is_cell_sorted_content_bound_and_strict() -> None:
    first, second = _budget("first"), _budget("second")
    wire = experiment_budget_sequence_to_dict((second, first))
    decoded = experiment_budget_sequence_from_dict(wire)
    assert tuple(row.cell_id for row in decoded) == tuple(
        sorted((first.cell_id, second.cell_id))
    )
    assert wire == experiment_budget_sequence_to_dict((first, second))

    reversed_wire = deepcopy(wire)
    reversed_wire["budgets"].reverse()
    with pytest.raises(ValueError, match="cell-sorted"):
        experiment_budget_sequence_from_dict(reversed_wire)

    duplicate_wire = experiment_budget_sequence_to_dict((first,))
    duplicate_wire["budgets"].append(deepcopy(duplicate_wire["budgets"][0]))
    with pytest.raises(ValueError, match="cell-sorted and unique"):
        experiment_budget_sequence_from_dict(duplicate_wire)


def test_unknown_missing_enum_and_scalar_type_confusion_fail_closed() -> None:
    selection_wire = sealed_e3a_selection_to_dict(_selection())
    unknown = deepcopy(selection_wire)
    unknown["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown"):
        sealed_e3a_selection_from_dict(unknown)

    missing = deepcopy(selection_wire)
    del missing["split_sha256"]
    with pytest.raises(ValueError, match="missing"):
        sealed_e3a_selection_from_dict(missing)

    bool_width = deepcopy(selection_wire)
    bool_width["width"] = True
    with pytest.raises(TypeError, match="JSON integer"):
        sealed_e3a_selection_from_dict(bool_width)

    budget_wire = experiment_budget_to_dict(_budget("types"))
    enum_object = deepcopy(budget_wire)
    enum_object["workload_class"] = WorkloadClass.TUNING
    with pytest.raises(TypeError, match="JSON text"):
        experiment_budget_from_dict(enum_object)

    integer_float = e1_pareto_artifact_to_dict(_pareto())
    integer_float["surviving_geometries"][0]["alpha_over_rank"] = 1
    with pytest.raises(TypeError, match="floating-point"):
        e1_pareto_artifact_from_dict(integer_float)

    legacy_pareto = e1_pareto_artifact_to_dict(_pareto())
    legacy_pareto["schema_version"] = 1
    legacy_pareto["common_load_sha256"] = _sha("legacy-minted-common-load")
    with pytest.raises(ValueError, match="fields differ"):
        e1_pareto_artifact_from_dict(legacy_pareto)

    wrong_schema_pareto = e1_pareto_artifact_to_dict(_pareto())
    wrong_schema_pareto["schema_version"] = 1
    with pytest.raises(ValueError, match="schema version 2"):
        e1_pareto_artifact_from_dict(wrong_schema_pareto)

    e2_evidence = e2_stage_evidence_artifact_to_dict(_e2_stage_evidence())
    del e2_evidence["evaluations"][0]["minimum_launched_updates"]
    with pytest.raises(ValueError, match="fields differ"):
        e2_stage_evidence_artifact_from_dict(e2_evidence)

    forged_counts = e2_stage_evidence_artifact_to_dict(_e2_stage_evidence())
    forged_counts["evaluations"][0]["minimum_launched_updates"] = 0
    with pytest.raises(ValueError, match="cannot exceed launched"):
        e2_stage_evidence_artifact_from_dict(forged_counts)


def test_nonfinite_nested_numbers_and_redundant_digests_fail_closed() -> None:
    evidence_wire = e2_stage_evidence_artifact_to_dict(_e2_stage_evidence())
    evidence_wire["evaluations"][0]["lc_vs_tts_confidence_lower_goodput_ratio"] = float(
        "inf"
    )
    with pytest.raises(ValueError, match="finite"):
        e2_stage_evidence_artifact_from_dict(evidence_wire)

    selection_wire = sealed_e3a_selection_to_dict(_selection())
    selection_wire["artifact_sha256"] = _sha("wrong-artifact")
    with pytest.raises(ValueError, match="redundant artifact SHA-256 mismatch"):
        sealed_e3a_selection_from_dict(selection_wire)

    survivor_wire = e2_survivor_receipt_to_dict(_e2_survivor())
    survivor_wire["completed_cells_sha256"] = _sha("wrong-completed-cells")
    with pytest.raises(ValueError, match="completed-cell digest"):
        e2_survivor_receipt_from_dict(survivor_wire)


def test_e2_final_recipe_rejects_bare_summary_and_forged_candidate() -> None:
    with pytest.raises(ValueError, match="fields differ"):
        e2_final_recipe_artifact_from_dict(
            {
                "artifact_kind": "e2_final_recipe_artifact",
                "artifact_sha256": _sha("bare-summary"),
                "candidate_id": _sha("caller-candidate"),
            }
        )
    wire = e2_final_recipe_artifact_to_dict(_e2_final_recipe())
    wire["candidate"]["learning_rate"] = 0.002
    with pytest.raises(ValueError, match="candidate identity"):
        e2_final_recipe_artifact_from_dict(wire)


def test_e2_reduction_replays_minima_after_nested_receipt_memory_mutation() -> None:
    survivor = _e2_survivor()
    object.__setattr__(survivor, "status", "SURVIVORS")
    object.__setattr__(
        survivor, "survivor_candidate_ids", survivor.source_candidate_ids
    )
    object.__setattr__(
        survivor,
        "reason_code",
        "registered_quarter_retention_with_family_floor",
    )
    with pytest.raises(ValueError, match="e2_promotion_minima_unregistered"):
        replace(_e2_reduction(), survivor_receipt=survivor)

    blocked_with_cargo = _e2_survivor()
    object.__setattr__(
        blocked_with_cargo,
        "survivor_candidate_ids",
        blocked_with_cargo.source_candidate_ids,
    )
    with pytest.raises(ValueError, match="blocked E2 receipts cannot promote"):
        replace(_e2_reduction(), survivor_receipt=blocked_with_cargo)


def test_nested_activation_and_budget_report_structure_is_revalidated() -> None:
    activation_wire = reducer_activation_artifact_to_dict(_activation())
    activation_wire["plan"]["activated_cell_ids"] = list(
        reversed(activation_wire["plan"]["activated_cell_ids"])
    )
    # The one-element fixture remains sorted, so use the blocked ID as an extra
    # value while preserving disjointness at the wire boundary.
    activation_wire["plan"]["activated_cell_ids"].append(_sha("out-of-order-activated"))
    with pytest.raises(ValueError, match="activated dispositions"):
        reducer_activation_artifact_from_dict(activation_wire)

    report_wire = industrial_budget_report_to_dict(_budget_report())
    report_wire["groups"][0]["compute_gpu_hours"][
        "registered_millisecond_numerator"
    ] += 1
    report_wire["groups"][0]["compute_gpu_hours"][
        "quota_envelope_millisecond_numerator"
    ] += 1
    with pytest.raises(ValueError, match="exact-hour numerator mismatch"):
        industrial_budget_report_from_dict(report_wire)


def test_sidecars_bind_both_kind_and_artifact_identity() -> None:
    selection_wire = sealed_e3a_selection_to_dict(_selection())
    wrong_kind = PlanningArtifactSidecar(
        schema_version=1,
        artifact_kind="e1_pareto_artifact",
        artifact_sha256=selection_wire["artifact_sha256"],
    )
    with pytest.raises(ValueError, match="sidecar kind mismatch"):
        sealed_e3a_selection_from_dict(selection_wire, sidecar=wrong_kind)

    wrong_digest = PlanningArtifactSidecar(
        schema_version=1,
        artifact_kind="sealed_e3a_selection",
        artifact_sha256=_sha("wrong-sidecar"),
    )
    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        sealed_e3a_selection_from_dict(selection_wire, sidecar=wrong_digest)

    malformed = wrong_digest.to_dict()
    malformed["unknown"] = "field"
    with pytest.raises(ValueError, match="unknown"):
        sealed_e3a_selection_from_dict(selection_wire, sidecar=malformed)


def test_dependence_map_rejects_non_digest_member_identities() -> None:
    valid = _dependence_map()
    unit = valid.units[0]
    corrupt = EvidenceDependenceMap(
        schema_version=1,
        units=(
            AnalysisDependenceUnit(
                unit_sha256=unit.unit_sha256,
                source_cell_id=unit.source_cell_id,
                member_cell_ids=tuple(sorted((unit.source_cell_id, "zz-not-a-sha"))),
            ),
        ),
    )
    with pytest.raises(ValueError, match="member cell_id"):
        evidence_dependence_map_to_dict(corrupt)
