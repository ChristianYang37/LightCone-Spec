from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from lightcone_spec.experiments.industrial_analysis import (
    E3bLongContextStageArtifact,
    E3bNamedLongContextReduction,
    IndustrialReducerArtifact,
    IndustrialRunBinding,
    MethodReduction,
)
from lightcone_spec.experiments.long_context_analysis import (
    E3B_CONTEXT_GRID,
    E3B_LONG_CONTEXT_PROTOCOL_SHA256,
    E3bCrossoverOutcome,
    E3bCrossoverReduction,
    E3bCurvePoint,
    E3bIntervalEstimate,
    E3bLongContextAnalysisPlan,
    E3bLongContextReduction,
    E3bMethod,
    E3bMetric,
    E3bReductionStatus,
)
from lightcone_spec.experiments.planning import (
    CONFIRMATION_FAMILY_POWER_REDUCER_PROTOCOL_SHA256,
    ConfirmationFamilyIdentity,
    ConfirmationFamilyPowerPlan,
    ConfirmationFamilyPowerReductionArtifact,
    RawEvidenceRunBinding,
    family_pilot_block_id,
)
from lightcone_spec.experiments.production_output import (
    OutputStatus,
    build_production_output_artifact,
    production_output_artifact_from_json_bytes,
)
from lightcone_spec.experiments.registry import (
    CORE_METHODS,
    FINAL_BLOCKS,
    PILOT_BLOCKS,
    content_sha256,
)
from lightcone_spec.experiments.runtime_metrics import export_formal_runtime_metrics
from lightcone_spec.experiments.statistics import (
    MAXIMUM_FINAL_BLOCKS,
    MINIMUM_FINAL_BLOCKS,
    PRIMARY_CONTRASTS,
    PRIMARY_FAMILY_ALPHA,
    PRIMARY_MINIMUM_RELATIVE_EFFECT,
    PRIMARY_TARGET_POWER,
    ContrastPower,
    MultiplicityDecision,
    P99ClaimGuard,
    PairedBcaContrast,
    PowerSizingPlan,
    SloRequest,
    account_slo,
)


def _sha(label: str) -> str:
    return content_sha256({"production-output-test": label})


def _family() -> ConfirmationFamilyIdentity:
    return ConfirmationFamilyIdentity(
        schema_version=1,
        registry_sha256=_sha("registry"),
        experiment="E3b",
        model="target-model",
        backend="DFLASH",
        task="code",
        context=4096,
        regime="long_input",
        arrival="concurrency_1",
        load_arrival_sha256=_sha("load-arrival"),
        width_panel="matched",
        topology="tp1_dp1",
        cohort_family="none",
        cohort_count=1,
        method_family=CORE_METHODS,
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        trace_sha256=_sha("trace"),
        sampling_sha256=_sha("sampling"),
        hardware_envelope_sha256=_sha("hardware-envelope"),
    )


def _raw_binding(
    *,
    family: ConfirmationFamilyIdentity,
    cell_id: str,
    index: int,
) -> RawEvidenceRunBinding:
    method = CORE_METHODS[index % len(CORE_METHODS)]
    block = PILOT_BLOCKS[index // len(CORE_METHODS)]
    return RawEvidenceRunBinding(
        schema_version=1,
        cell_id=cell_id,
        experiment=family.experiment,
        method=method,
        scientific_unit=f"excluded_pilot_{block}",
        config_sha256=cell_id,
        rank_config_sha256s=(_sha(f"rank-{index}"),),
        run_id=f"pilot-run-{index}",
        rank_count=1,
        model_pair=family.model,
        runtime_sha256=family.runtime_sha256,
        split_sha256=family.split_sha256,
        corpus_sha256=_sha(f"corpus-{index}"),
        arrival_trace_sha256=_sha(f"arrival-{index}"),
        request_ids_sha256=_sha(f"requests-{index}"),
        sampling_profile_sha256=family.sampling_sha256,
        model_lock_sha256=_sha(f"model-{index}"),
        patched_sglang_tree="a" * 40,
        run_nonce_sha256=_sha(f"nonce-{index}"),
        topology_sha256=_sha(f"topology-{index}"),
        experiment_budget_sha256=_sha(f"budget-plan-{index}"),
        physical_gpu_uuids=(f"GPU-{index % 2}",),
        terminal_receipt_sha256s=(_sha(f"terminal-{index}"),),
        hardware_receipt_sha256=_sha(f"hardware-{index}"),
        budget_observation_sha256=_sha(f"budget-observation-{index}"),
    )


def _power_reduction(
    *,
    underpowered: bool = False,
) -> ConfirmationFamilyPowerReductionArtifact:
    family = _family()
    cell_ids = tuple(
        sorted(
            _sha(f"pilot-cell-{index}")
            for index in range(len(PILOT_BLOCKS) * len(CORE_METHODS))
        )
    )
    bindings = tuple(
        _raw_binding(family=family, cell_id=cell_id, index=index)
        for index, cell_id in enumerate(cell_ids)
    )
    power = 0.50 if underpowered else 0.90
    selected = None if underpowered else MINIMUM_FINAL_BLOCKS
    sizing = PowerSizingPlan(
        status="UNDERPOWERED" if underpowered else "READY",
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
            (contrast, 0.10) for contrast in PRIMARY_CONTRASTS
        ),
        power_grid=tuple(
            ContrastPower(
                contrast=contrast,
                final_blocks=blocks,
                power=power,
            )
            for blocks in range(MINIMUM_FINAL_BLOCKS, MAXIMUM_FINAL_BLOCKS + 1)
            for contrast in PRIMARY_CONTRASTS
        ),
    )
    evidence_sha256 = _sha("pilot-evidence")
    plan = ConfirmationFamilyPowerPlan(
        schema_version=1,
        family=family,
        pilot_activation_sha256=_sha("pilot-activation"),
        completed_pilot_cells_sha256=content_sha256(cell_ids),
        pilot_evidence_sha256=evidence_sha256,
        power_sizing=sizing,
        status="UNDERPOWERED" if underpowered else "POWERED",
        selected_final_blocks=selected,
        selected_final_prefix=() if selected is None else FINAL_BLOCKS[:selected],
        reason_code=(
            "registered_family_underpowered"
            if underpowered
            else "registered_family_power_target_met"
        ),
        selection_state="sealed_before_confirmation_unblinding",
    )
    return ConfirmationFamilyPowerReductionArtifact(
        schema_version=2,
        plan=plan,
        inventory_sha256=_sha("inventory"),
        inventory_source_receipt_sha256=_sha("inventory-source"),
        fixed_instance_gpu_count=2,
        inventory_host_id="production-output-test-host",
        raw_evidence_manifest_sha256=evidence_sha256,
        terminal_receipt_sha256s=tuple(
            sorted(
                receipt
                for binding in bindings
                for receipt in binding.terminal_receipt_sha256s
            )
        ),
        hardware_receipt_sha256s=tuple(
            sorted(binding.hardware_receipt_sha256 for binding in bindings)
        ),
        budget_observation_sha256s=tuple(
            sorted(binding.budget_observation_sha256 for binding in bindings)
        ),
        run_bindings=bindings,
        reducer_protocol_sha256=CONFIRMATION_FAMILY_POWER_REDUCER_PROTOCOL_SHA256,
        data_source="excluded_pilots_only",
        confirmation_data_visible=False,
    )


def _observed_e3b_stage(
    *,
    registry_sha256: str,
) -> E3bLongContextStageArtifact:
    final_blocks = FINAL_BLOCKS[:MINIMUM_FINAL_BLOCKS]
    family_sha256 = _sha("e3b-context-family")
    interval = E3bIntervalEstimate(
        estimate=123.456,
        lower=120.0,
        upper=126.0,
    )
    points = tuple(
        E3bCurvePoint(
            context_tokens=context,
            candidate_fitted_metric=interval,
            baseline_fitted_metric=interval,
            candidate_elasticity=interval,
            baseline_elasticity=interval,
            paired_elasticity_difference=interval,
            candidate_curvature=interval,
            baseline_curvature=interval,
            paired_curvature_difference=interval,
        )
        for context in E3B_CONTEXT_GRID
    )
    identities = tuple(
        sorted(
            (
                (E3bMetric.ACCEPTED_LENGTH, E3bMethod.L0, E3bMethod.STATIC),
                (
                    E3bMetric.ACCEPTED_LENGTH,
                    E3bMethod.L0,
                    E3bMethod.TARGET_ONLY,
                ),
                (E3bMetric.ACCEPTED_LENGTH, E3bMethod.L0, E3bMethod.TTS),
                (
                    E3bMetric.COMMITTED_TOKEN_GOODPUT,
                    E3bMethod.L0,
                    E3bMethod.STATIC,
                ),
                (
                    E3bMetric.COMMITTED_TOKEN_GOODPUT,
                    E3bMethod.L0,
                    E3bMethod.TARGET_ONLY,
                ),
            ),
            key=lambda value: ":".join(item.value for item in value),
        )
    )
    named: list[E3bNamedLongContextReduction] = []
    for index, (metric, candidate, baseline) in enumerate(identities):
        plan = E3bLongContextAnalysisPlan(
            schema_version=1,
            protocol_sha256=E3B_LONG_CONTEXT_PROTOCOL_SHA256,
            family_sha256=family_sha256,
            metric=metric,
            candidate_method=candidate,
            baseline_method=baseline,
            final_block_ids=final_blocks,
            bootstrap_repetitions=100,
            bootstrap_seed=7,
        )
        named.append(
            E3bNamedLongContextReduction(
                metric=metric,
                candidate_method=candidate,
                baseline_method=baseline,
                reduction=E3bLongContextReduction(
                    schema_version=1,
                    status=E3bReductionStatus.OBSERVED,
                    reason_code="e3b_reduction_observed",
                    protocol_sha256=E3B_LONG_CONTEXT_PROTOCOL_SHA256,
                    plan_sha256=plan.sha256,
                    observations_sha256=_sha(f"e3b-observations-{index}"),
                    curve_points=points,
                    crossover=E3bCrossoverReduction(
                        outcome=E3bCrossoverOutcome.CROSSOVER,
                        reason_code="first_registered_crossover_observed",
                        first_bracket_tokens=(8192, 16384),
                        root_tokens=12000.0,
                        root_interval_tokens=(11000.0, 13000.0),
                    ),
                    bootstrap_repetitions_completed=100,
                ),
            )
        )
    return E3bLongContextStageArtifact(
        schema_version=1,
        status="UNRESOLVED",
        evidence_level="RAW_DIAGNOSTIC_OBSERVED_UNATTESTED",
        reasons=("gpu_attestation:missing",),
        registry_sha256=registry_sha256,
        protocol_sha256=E3B_LONG_CONTEXT_PROTOCOL_SHA256,
        context_family_sha256=family_sha256,
        raw_family_input_sha256s=tuple(
            sorted(_sha(f"raw-family-{index}") for index in range(8))
        ),
        family_reduction_sha256s=tuple(
            sorted(_sha(f"family-reduction-{index}") for index in range(8))
        ),
        final_block_ids=final_blocks,
        bootstrap_repetitions=100,
        bootstrap_seed=7,
        reductions=tuple(named),
    )


def _industrial_reduction(
    power: ConfirmationFamilyPowerReductionArtifact,
) -> IndustrialReducerArtifact:
    run_id = "production-output-formal-run"
    runtime_metrics = export_formal_runtime_metrics(
        None,
        expected_run_ids=(run_id,),
    )
    run_binding = IndustrialRunBinding(
        block=FINAL_BLOCKS[0],
        method="l0",
        cell_id=_sha("industrial-cell"),
        config_sha256=_sha("industrial-config"),
        rank_config_sha256s=(_sha("industrial-rank"),),
        run_id=run_id,
        rank_count=1,
        model_pair=power.family.model,
        corpus_sha256=_sha("industrial-corpus"),
        arrival_trace_sha256=power.family.trace_sha256,
        request_ids_sha256=_sha("industrial-requests"),
        sampling_profile_sha256=power.family.sampling_sha256,
        model_lock_sha256=_sha("industrial-model-lock"),
        patched_sglang_tree="a" * 40,
        run_nonce_sha256=_sha("industrial-nonce"),
        topology_sha256=_sha("industrial-topology"),
        experiment_budget_sha256=_sha("industrial-budget-plan"),
        inventory_sha256=power.inventory_sha256,
        inventory_source_receipt_sha256=power.inventory_source_receipt_sha256,
        fixed_instance_gpu_count=power.fixed_instance_gpu_count,
        physical_host_id=power.inventory_host_id,
        gpu_uuids=("GPU-0",),
        terminal_receipt_sha256s=(_sha("industrial-terminal"),),
        hardware_receipt_sha256=_sha("industrial-hardware"),
        budget_observation_sha256=_sha("industrial-budget-observation"),
    )
    slo = account_slo(
        (
            SloRequest(
                request_id="request-1",
                prompt_bucket="short",
                eligible=True,
                completed=True,
                error=False,
                ttft_ms=10.0,
                within_request_p99_itl_ms=10.0,
            ),
        )
    )
    methods = tuple(
        MethodReduction(
            method=method,
            block_ids=("final-unit",),
            mean_output_goodput_tps=999.0,
            mean_slo_qualified_goodput_tps=998.0,
            slo=slo,
            aggregate_latency_p99=P99ClaimGuard(
                anchor_id=f"anchor-{method}",
                completed_requests=10_000 if method == "l0" else 9_999,
                observed_p99_ms=87.0 if method == "l0" else None,
                minimum_completions=10_000,
                status="CLAIMABLE" if method == "l0" else "UNRESOLVED",
            ),
        )
        for method in CORE_METHODS
    )
    contrasts = tuple(
        PairedBcaContrast(
            name=name,
            block_ids=("final-1", "final-2"),
            mean_log_ratio=0.10,
            mean_relative_gain=0.105,
            ci_lower_relative_gain=0.01,
            ci_upper_relative_gain=0.20,
            raw_p_value=0.01,
            confidence=0.95,
        )
        for name in PRIMARY_CONTRASTS
    )
    holm = tuple(
        MultiplicityDecision(
            name=name,
            raw_p_value=0.01,
            adjusted_p_value=0.02,
            rejected=True,
            procedure="holm",
        )
        for name in PRIMARY_CONTRASTS
    )
    return IndustrialReducerArtifact(
        status="UNRESOLVED",
        gpu_evidence="UNMEASURED",
        reasons=("gpu_attestation:missing",),
        registry_sha256=power.family.registry_sha256,
        experiment=power.family.experiment,
        runtime_sha256=power.family.runtime_sha256,
        split_sha256=power.family.split_sha256,
        inventory_sha256=power.inventory_sha256,
        inventory_source_receipt_sha256=power.inventory_source_receipt_sha256,
        fixed_instance_gpu_count=power.fixed_instance_gpu_count,
        inventory_host_id=power.inventory_host_id,
        confirmation_family_sha256=power.family.sha256,
        pilot_activation_sha256=power.plan.pilot_activation_sha256,
        final_activation_sha256=_sha("final-activation"),
        confirmation_plan_sha256=power.sha256,
        evidence_dependence_map_sha256=None,
        evidence_alias_reduction_sha256s=(),
        patched_sglang_tree="a" * 40,
        model_lock_sha256=_sha("industrial-model-lock"),
        hardware_envelope_sha256=power.family.hardware_envelope_sha256,
        gpu_attestation_sha256=None,
        doctor_report_sha256=None,
        pilot_evidence_sha256=power.plan.pilot_evidence_sha256,
        completed_pilot_cells_sha256=power.plan.completed_pilot_cells_sha256,
        terminal_receipt_sha256s=run_binding.terminal_receipt_sha256s,
        qualification_lock_sha256s=(_sha("qualification"),),
        hardware_receipt_sha256s=(run_binding.hardware_receipt_sha256,),
        budget_observation_sha256s=(run_binding.budget_observation_sha256,),
        run_bindings=(run_binding,),
        runtime_metrics=runtime_metrics,
        power_plan=power.plan.power_sizing,
        hardware_validity=(),
        methods=methods,
        primary_contrasts=contrasts,
        holm_family=holm,
        bootstrap_hooks=(("hierarchical_block_request", ("block", "request")),),
    )


def test_missing_production_sources_emit_canonical_named_blocks() -> None:
    artifact = build_production_output_artifact(
        e3b_stage=None,
        industrial_reduction=None,
        family_power_reduction=None,
    )

    assert artifact.status is OutputStatus.BLOCKED
    assert artifact.blocker_codes == (
        "confirmation_family_power_reduction_artifact_missing",
        "e3b_long_context_stage_artifact_missing",
        "industrial_reducer_artifact_missing",
    )
    value = artifact.to_dict()
    assert value["figures"][0]["status"] == "BLOCKED"
    assert len(value["figures"][0]["panels"]) == 5
    assert value["tables"][0]["scientific_status"] == "MISSING"
    assert value["tables"][0]["rows"] == []
    assert len(value["tables"][1]["p99_rows"]) == len(CORE_METHODS)
    assert len(value["tables"][1]["primary_rows"]) == len(PRIMARY_CONTRASTS)
    assert artifact.canonical_json_bytes() == artifact.canonical_json_bytes()
    assert json.loads(artifact.canonical_json_bytes()) == value


def test_unattested_typed_reductions_keep_all_publication_measurements_null() -> None:
    power = _power_reduction()
    industrial = _industrial_reduction(power)
    e3b = _observed_e3b_stage(registry_sha256=power.family.registry_sha256)

    artifact = build_production_output_artifact(
        e3b_stage=e3b,
        industrial_reduction=industrial,
        family_power_reduction=power,
    )
    value = artifact.to_dict()

    assert artifact.blocker_codes == (
        "e3b_formal_production_status_unresolved",
        "industrial_formal_production_status_unresolved",
    )
    figure = value["figures"][0]
    assert figure["source_evidence_level"] == ("RAW_DIAGNOSTIC_OBSERVED_UNATTESTED")
    assert all(
        panel["source_reduction_status"] == "OBSERVED" for panel in figure["panels"]
    )
    assert all(
        point["values"] is None
        for panel in figure["panels"]
        for point in panel["points"]
    )
    assert all(panel["crossover"]["root_tokens"] is None for panel in figure["panels"])

    power_table, claim_table = value["tables"]
    assert power_table["status"] == "READY"
    assert power_table["scientific_status"] == "POWERED"
    assert power_table["evidence_role"] == "preregistered_power_planning_only"
    assert power_table["formal_result_eligible"] is False
    assert len(power_table["rows"]) == len(PRIMARY_CONTRASTS) * 9
    l0_p99 = next(row for row in claim_table["p99_rows"] if row["method"] == "l0")
    assert l0_p99["request_count_gate_status"] == "CLAIMABLE"
    assert l0_p99["minimum_completions"] == 10_000
    assert l0_p99["completed_requests"] is None
    assert l0_p99["observed_p99_ms"] is None
    assert all(
        row["mean_relative_gain"] is None
        and row["ci_lower_relative_gain"] is None
        and row["adjusted_p_value"] is None
        for row in claim_table["primary_rows"]
    )
    assert b"123.456" not in artifact.canonical_json_bytes()
    assert b"12000.0" not in artifact.canonical_json_bytes()
    assert (
        production_output_artifact_from_json_bytes(
            artifact.canonical_json_bytes(),
            e3b_stage=e3b,
            industrial_reduction=industrial,
            family_power_reduction=power,
        ).to_dict()
        == artifact.to_dict()
    )


def test_underpowered_power_artifact_is_visible_and_blocks_completion() -> None:
    power = _power_reduction(underpowered=True)
    artifact = build_production_output_artifact(
        e3b_stage=None,
        industrial_reduction=None,
        family_power_reduction=power,
    )

    assert "confirmation_family_underpowered" in artifact.blocker_codes
    power_table = artifact.to_dict()["tables"][0]
    assert power_table["status"] == "BLOCKED"
    assert power_table["scientific_status"] == "UNDERPOWERED"
    assert power_table["selected_final_blocks"] is None
    assert {row["power"] for row in power_table["rows"]} == {0.50}


def test_output_rejects_foreign_or_summary_like_sources() -> None:
    power = _power_reduction()
    industrial = _industrial_reduction(power)
    with pytest.raises(TypeError, match="exact industrial reducer"):
        build_production_output_artifact(
            e3b_stage=None,
            industrial_reduction=industrial.to_dict(),  # type: ignore[arg-type]
            family_power_reduction=power,
        )
    with pytest.raises(ValueError, match="sources differ"):
        build_production_output_artifact(
            e3b_stage=None,
            industrial_reduction=replace(
                industrial,
                registry_sha256=_sha("foreign-registry"),
            ),
            family_power_reduction=power,
        )
    with pytest.raises(TypeError, match="method reductions must be exact"):
        build_production_output_artifact(
            e3b_stage=None,
            industrial_reduction=replace(
                industrial,
                methods=(object(),),  # type: ignore[arg-type]
            ),
            family_power_reduction=None,
        )
    forged_methods = tuple(
        replace(
            value,
            aggregate_latency_p99=replace(
                value.aggregate_latency_p99,
                minimum_completions=1,
            ),
        )
        if value.method == "l0"
        else value
        for value in industrial.methods
    )
    with pytest.raises(ValueError, match="p99 claim guard is not canonical"):
        build_production_output_artifact(
            e3b_stage=None,
            industrial_reduction=replace(industrial, methods=forged_methods),
            family_power_reduction=None,
        )


def test_strict_codec_rejects_duplicate_nonfinite_and_noncanonical_json() -> None:
    artifact = build_production_output_artifact(
        e3b_stage=None,
        industrial_reduction=None,
        family_power_reduction=None,
    )
    body = artifact.canonical_json_bytes()
    reopened = production_output_artifact_from_json_bytes(
        body,
        e3b_stage=None,
        industrial_reduction=None,
        family_power_reduction=None,
    )
    assert reopened.to_dict() == artifact.to_dict()
    assert reopened.sha256 == artifact.sha256

    with pytest.raises(ValueError, match="duplicate keys"):
        production_output_artifact_from_json_bytes(
            b'{"schema_version":1,"schema_version":1}\n',
            e3b_stage=None,
            industrial_reduction=None,
            family_power_reduction=None,
        )
    with pytest.raises(ValueError, match="non-finite or invalid"):
        production_output_artifact_from_json_bytes(
            b'{"value":NaN}\n',
            e3b_stage=None,
            industrial_reduction=None,
            family_power_reduction=None,
        )
    with pytest.raises(ValueError, match="not canonical JSON"):
        production_output_artifact_from_json_bytes(
            b" " + body,
            e3b_stage=None,
            industrial_reduction=None,
            family_power_reduction=None,
        )


def test_joint_rehash_cannot_promote_or_leak_diagnostic_e3b_values() -> None:
    power = _power_reduction()
    artifact = build_production_output_artifact(
        e3b_stage=_observed_e3b_stage(registry_sha256=power.family.registry_sha256),
        industrial_reduction=None,
        family_power_reduction=power,
    )
    payload = artifact.figure.to_dict()
    payload["status"] = "READY"
    payload["source_sha256"] = "b" * 64
    payload["reason_codes"] = []
    payload["source_stage_status"] = "UNRESOLVED"
    payload["panels"][0]["points"][0]["values"] = {
        "candidate_fitted_metric": {
            "estimate": 123.456,
            "lower": 120.0,
            "upper": 126.0,
        }
    }
    forged_body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    with pytest.raises(ValueError, match="must remain BLOCKED"):
        replace(
            artifact.figure,
            status=OutputStatus.READY,
            source_sha256="b" * 64,
            reason_codes=(),
            canonical_payload=forged_body,
        )


def test_codec_replays_typed_source_against_rehashed_power_payload() -> None:
    power = _power_reduction()
    artifact = build_production_output_artifact(
        e3b_stage=None,
        industrial_reduction=None,
        family_power_reduction=power,
    )
    value = artifact.to_dict()
    value["tables"][0]["rows"][0]["power"] = 0.123
    forged_body = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with pytest.raises(
        ValueError,
        match="registered grid|exact typed source replay",
    ):
        production_output_artifact_from_json_bytes(
            forged_body,
            e3b_stage=None,
            industrial_reduction=None,
            family_power_reduction=power,
        )


def test_foreign_source_object_fails_with_stable_type_error() -> None:
    artifact = build_production_output_artifact(
        e3b_stage=None,
        industrial_reduction=None,
        family_power_reduction=None,
    )
    with pytest.raises(TypeError, match="source binding must be exact"):
        replace(
            artifact,
            sources=(object(), *artifact.sources[1:]),  # type: ignore[arg-type]
        )


def test_cached_typed_source_mutation_cannot_reuse_a_sealed_digest() -> None:
    power = _power_reduction()
    _ = power.sha256
    for row in power.plan.power_sizing.power_grid:
        object.__setattr__(row, "power", 0.91)
    with pytest.raises(ValueError, match="family-power source changed"):
        build_production_output_artifact(
            e3b_stage=None,
            industrial_reduction=None,
            family_power_reduction=power,
        )

    e3b = _observed_e3b_stage(registry_sha256=_sha("e3b-registry"))
    _ = e3b.sha256
    interval = e3b.reductions[0].reduction.curve_points[0].candidate_fitted_metric  # type: ignore[index,union-attr]
    object.__setattr__(interval, "estimate", 124.0)
    with pytest.raises(ValueError, match="E3b reduction changed"):
        build_production_output_artifact(
            e3b_stage=e3b,
            industrial_reduction=None,
            family_power_reduction=None,
        )


def test_present_sources_require_one_registry_and_registered_claim_semantics() -> None:
    power = _power_reduction()
    industrial = _industrial_reduction(power)
    foreign_e3b = _observed_e3b_stage(registry_sha256=_sha("foreign-registry"))
    with pytest.raises(ValueError, match="sources differ in registry identity"):
        build_production_output_artifact(
            e3b_stage=foreign_e3b,
            industrial_reduction=industrial,
            family_power_reduction=None,
        )

    forged = replace(
        industrial,
        primary_contrasts=tuple(
            replace(value, independent_unit="request")
            for value in industrial.primary_contrasts
        ),
    )
    with pytest.raises(ValueError, match="contrast semantics"):
        build_production_output_artifact(
            e3b_stage=None,
            industrial_reduction=forged,
            family_power_reduction=None,
        )


def test_missing_reducer_cannot_expose_source_derived_claim_metadata() -> None:
    artifact = build_production_output_artifact(
        e3b_stage=None,
        industrial_reduction=None,
        family_power_reduction=None,
    )
    claim = artifact.tables[1]
    payload = claim.to_dict()
    row = payload["p99_rows"][0]
    row["request_count_gate_status"] = "CLAIMABLE"
    row["anchor_id"] = "forged-anchor"
    row["minimum_completions"] = 10_000
    forged_body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    with pytest.raises(ValueError, match="missing reducer cannot expose"):
        replace(claim, canonical_payload=forged_body)
