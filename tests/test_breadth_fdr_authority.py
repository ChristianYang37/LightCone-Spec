from __future__ import annotations

import json
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments.breadth_fdr_authority import (
    E0_BREADTH_FALSE_DISCOVERY_RATE,
    E0_BREADTH_LEGACY_DIAGNOSTIC_PROTOCOL_SHA256,
    E0_BREADTH_RAW_SOURCE_MISSING_REASON,
    E0BreadthFdrAuthorityBlocked,
    E0BreadthHypothesis,
    bind_e0_breadth_fdr_authority,
    formal_e0_breadth_fdr_receipt_from_dict,
    formal_e0_breadth_fdr_receipt_to_dict,
    reduce_e0_breadth_fdr,
    reduce_formal_e0_breadth_fdr_from_projection,
    registered_e0_breadth_hypotheses,
    registered_formal_e0_breadth_hypotheses,
    revalidate_e0_breadth_fdr_authority,
)
from lightcone_spec.experiments.e0_authority_artifact import (
    E0_FINAL_COMPLETION_PROTOCOL_SHA256,
    E0_FINAL_SLO_GOODPUT_POLICY_SHA256,
    E0FinalAnalysisCell,
    E0FinalAnalysisProjection,
    E0FinalCellCompletion,
    E0FinalCompletionReceipt,
    _e0_slo_goodput_inputs,
)
from lightcone_spec.experiments.formal_protocol import (
    E0_METHOD_ROLES as FORMAL_E0_METHOD_ROLES,
)
from lightcone_spec.experiments.registry import (
    E0_BACKENDS,
    E0_MODELS,
    E0_TASKS,
    ExperimentRegistry,
    build_industrial_registry,
    build_legacy_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.stage_materialization import (
    E0CompatibilityDecision,
)
from lightcone_spec.experiments.statistics import benjamini_hochberg


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_legacy_industrial_registry()


@pytest.fixture(scope="module")
def hypotheses(
    registry: ExperimentRegistry,
) -> tuple[E0BreadthHypothesis, ...]:
    return registered_e0_breadth_hypotheses(registry)


def _raw_row(hypothesis: E0BreadthHypothesis, p_value: float) -> dict[str, object]:
    numerator = content_sha256(
        {"role": "numerator_terminal", "hypothesis_id": hypothesis.hypothesis_id}
    )
    denominator = content_sha256(
        {"role": "denominator_terminal", "hypothesis_id": hypothesis.hypothesis_id}
    )
    artifact = content_sha256(
        {
            "schema_version": 1,
            "hypothesis_id": hypothesis.hypothesis_id,
            "numerator_terminal_sha256": numerator,
            "denominator_terminal_sha256": denominator,
            "raw_p_value": float(p_value),
        }
    )
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "numerator_terminal_sha256": numerator,
        "denominator_terminal_sha256": denominator,
        "contrast_artifact_sha256": artifact,
        "raw_p_value": p_value,
    }


def _raw_document(
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
    p_value: Callable[[E0BreadthHypothesis, int], float] | None = None,
) -> dict[str, object]:
    value_for = p_value or (lambda _hypothesis, _index: 0.5)
    rows = {
        hypothesis.hypothesis_id: _raw_row(
            hypothesis,
            value_for(hypothesis, index),
        )
        for index, hypothesis in enumerate(hypotheses)
    }
    return {
        "schema_version": 1,
        "kind": "e0_breadth_raw_p_values",
        "registry_sha256": registry.sha256,
        "protocol_sha256": E0_BREADTH_LEGACY_DIAGNOSTIC_PROTOCOL_SHA256,
        "false_discovery_rate": E0_BREADTH_FALSE_DISCOVERY_RATE,
        "hypotheses_sha256": content_sha256(
            [hypothesis.to_dict() for hypothesis in hypotheses]
        ),
        "families": [
            {
                "family_id": family_id,
                "p_values": [
                    rows[hypothesis.hypothesis_id]
                    for hypothesis in hypotheses
                    if hypothesis.family_id == family_id
                ],
            }
            for family_id in (
                "e0_core_breadth",
                "e0_isolated_onlinespec_breadth",
            )
        ],
    }


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path.resolve()


def test_registered_universe_is_exact_and_registry_owned(
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
) -> None:
    assert len(hypotheses) == 756
    assert len({hypothesis.hypothesis_id for hypothesis in hypotheses}) == 756
    assert tuple(hypothesis.hypothesis_id for hypothesis in hypotheses) == tuple(
        sorted(hypothesis.hypothesis_id for hypothesis in hypotheses)
    )
    assert {hypothesis.model for hypothesis in hypotheses} == set(E0_MODELS)
    assert {hypothesis.backend for hypothesis in hypotheses} == set(E0_BACKENDS)
    assert {hypothesis.task for hypothesis in hypotheses} == set(E0_TASKS)

    core = tuple(
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.family_id == "e0_core_breadth"
    )
    online = tuple(
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.family_id == "e0_isolated_onlinespec_breadth"
    )
    assert len(core) == 432
    assert {hypothesis.contrast for hypothesis in core} == {
        "lightcone_vs_tts",
        "lightcone_vs_static",
        "l0_naive_vs_tts",
        "lightcone_vs_l0_naive",
    }
    assert len(online) == 324
    assert {hypothesis.contrast for hypothesis in online} == {
        "onlinespec_ogd_vs_static",
        "onlinespec_opt_vs_static",
        "onlinespec_ens_vs_static",
    }

    cells = {cell.cell_id: cell for cell in registry.cells_for("E0")}
    for hypothesis in hypotheses:
        assert cells[hypothesis.numerator_cell_id].sha256 == (
            hypothesis.numerator_cell_sha256
        )
        assert cells[hypothesis.denominator_cell_id].sha256 == (
            hypothesis.denominator_cell_sha256
        )
        assert cells[hypothesis.numerator_cell_id].identity.variant.startswith(
            "compatibility_template:role="
        )
        assert cells[hypothesis.denominator_cell_id].identity.variant.startswith(
            "compatibility_template:role="
        )


def test_complete_source_applies_existing_bh_separately_to_fixed_families(
    tmp_path: Path,
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
) -> None:
    first_by_family = {
        "e0_core_breadth": next(
            value for value in hypotheses if value.family_id == "e0_core_breadth"
        ).hypothesis_id,
        "e0_isolated_onlinespec_breadth": next(
            value
            for value in hypotheses
            if value.family_id == "e0_isolated_onlinespec_breadth"
        ).hypothesis_id,
    }

    def p_value(hypothesis: E0BreadthHypothesis, _index: int) -> float:
        return 1e-8 if hypothesis.hypothesis_id in first_by_family.values() else 0.5

    path = _write_json(
        tmp_path / "e0-p-values.json",
        _raw_document(registry, hypotheses, p_value),
    )
    authority = bind_e0_breadth_fdr_authority(registry, path)
    reduction = reduce_e0_breadth_fdr(registry, authority)

    assert len(authority.p_values) == len(reduction.decisions) == 756
    assert reduction.primary_family_eligible is False
    assert reduction.formal_execution_authorized is False
    assert reduction.families == (
        "e0_core_breadth",
        "e0_isolated_onlinespec_breadth",
    )
    assert sum(decision.rejected for decision in reduction.decisions) == 2
    decision_by_id = {
        decision.hypothesis_id: decision for decision in reduction.decisions
    }
    raw_by_id = {value.hypothesis_id: value for value in authority.p_values}
    for family_id, expected_size in (
        ("e0_core_breadth", 432),
        ("e0_isolated_onlinespec_breadth", 324),
    ):
        family = tuple(
            hypothesis for hypothesis in hypotheses if hypothesis.family_id == family_id
        )
        expected = benjamini_hochberg(
            {
                hypothesis.hypothesis_id: raw_by_id[
                    hypothesis.hypothesis_id
                ].raw_p_value
                for hypothesis in family
            },
            false_discovery_rate=0.05,
        )
        assert len(expected) == expected_size
        for adjusted in expected:
            decision = decision_by_id[adjusted.name]
            assert decision.q_value == adjusted.adjusted_p_value
            assert decision.rejected == adjusted.rejected
            assert decision.procedure == "benjamini-hochberg"


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda document: document["families"][0]["p_values"].pop(),
            "coverage",
        ),
        (
            lambda document: document["families"].reverse(),
            "reordered or regrouped",
        ),
        (
            lambda document: document["families"].pop(),
            "registered families",
        ),
        (
            lambda document: document.__setitem__("false_discovery_rate", 0.1),
            "registered protocol",
        ),
        (
            lambda document: document["families"][0]["p_values"][0].__setitem__(
                "hypothesis_id", "0" * 64
            ),
            "source identity|coverage",
        ),
        (
            lambda document: document["families"][0]["p_values"][0].__setitem__(
                "raw_p_value", 1.1
            ),
            "finite probability",
        ),
    ],
)
def test_subset_regroup_foreign_and_invalid_values_fail_closed(
    tmp_path: Path,
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    document = _raw_document(registry, hypotheses)
    mutation(document)
    path = _write_json(tmp_path / "mutated.json", document)
    with pytest.raises(ValueError, match=message):
        bind_e0_breadth_fdr_authority(registry, path)


def test_duplicate_json_keys_and_nonfinite_json_fail_closed(
    tmp_path: Path,
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
) -> None:
    document = _raw_document(registry, hypotheses)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"))

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        encoded.replace('"families":', '"families":[],"families":', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicates key"):
        bind_e0_breadth_fdr_authority(registry, duplicate.resolve())

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(encoded.replace('"raw_p_value":0.5', '"raw_p_value":NaN', 1))
    with pytest.raises(ValueError, match="non-finite"):
        bind_e0_breadth_fdr_authority(registry, nonfinite.resolve())


def test_source_identities_are_required_and_consistent(
    tmp_path: Path,
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
) -> None:
    same_terminal = _raw_document(registry, hypotheses)
    row = same_terminal["families"][0]["p_values"][0]
    row["denominator_terminal_sha256"] = row["numerator_terminal_sha256"]
    path = _write_json(tmp_path / "same-terminal.json", same_terminal)
    with pytest.raises(ValueError, match="sources must differ"):
        bind_e0_breadth_fdr_authority(registry, path)

    wrong_artifact = _raw_document(registry, hypotheses)
    wrong_artifact["families"][0]["p_values"][0]["contrast_artifact_sha256"] = "f" * 64
    path = _write_json(tmp_path / "wrong-artifact.json", wrong_artifact)
    with pytest.raises(ValueError, match="source identity"):
        bind_e0_breadth_fdr_authority(registry, path)


def test_missing_symlink_foreign_registry_and_revalidation_tamper_fail_closed(
    tmp_path: Path,
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
) -> None:
    with pytest.raises(E0BreadthFdrAuthorityBlocked) as missing:
        bind_e0_breadth_fdr_authority(registry, tmp_path / "missing.json")
    assert missing.value.reason == E0_BREADTH_RAW_SOURCE_MISSING_REASON

    path = _write_json(
        tmp_path / "raw.json",
        _raw_document(registry, hypotheses),
    )
    link = tmp_path / "raw-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        bind_e0_breadth_fdr_authority(registry, link.absolute())

    authority = bind_e0_breadth_fdr_authority(registry, path)
    foreign_registry = build_legacy_industrial_registry(seed=20260812)
    with pytest.raises(ValueError, match="registered protocol"):
        bind_e0_breadth_fdr_authority(foreign_registry, path)

    document = _raw_document(registry, hypotheses)
    hypothesis = hypotheses[0]
    replacement = _raw_row(hypothesis, 0.25)
    for family in document["families"]:
        for index, row in enumerate(family["p_values"]):
            if row["hypothesis_id"] == hypothesis.hypothesis_id:
                family["p_values"][index] = replacement
    _write_json(path, document)
    with pytest.raises(ValueError, match="changed during revalidation"):
        revalidate_e0_breadth_fdr_authority(registry, authority)


def _formal_projection(registry: ExperimentRegistry) -> E0FinalAnalysisProjection:
    combinations = [
        (model, backend, task)
        for model in E0_MODELS
        for backend in E0_BACKENDS
        for task in E0_TASKS
    ]
    valid_key = combinations[0]
    decisions = tuple(
        sorted(
            (
                E0CompatibilityDecision(
                    model=model,
                    backend=backend,
                    task=task,
                    disposition=(
                        "VALID" if (model, backend, task) == valid_key else "N/A"
                    ),
                    reason_code=(
                        "compatible"
                        if (model, backend, task) == valid_key
                        else "official_interface_incompatible"
                    ),
                    interface_sha256=content_sha256(
                        {"interface": [model, backend, task]}
                    ),
                    task_native_workload_sha256=content_sha256(
                        {"workload": [model, backend, task]}
                    ),
                )
                for model, backend, task in combinations
            ),
            key=lambda row: row.decision_id,
        )
    )
    valid = next(row for row in decisions if row.disposition == "VALID")
    analysis_rows = []
    completion_rows = []
    for block in range(4, 16):
        for role_index, role in enumerate(FORMAL_E0_METHOD_ROLES):
            for load in ("concurrency_one", "common_slo_load"):
                cell_id = content_sha256(
                    {"formal-e0-cell": [block, role, load, valid.decision_id]}
                )
                terminal = content_sha256({"terminal": cell_id})
                analysis_rows.append(
                    E0FinalAnalysisCell(
                        materialized_cell_id=cell_id,
                        compatibility_decision_id=valid.decision_id,
                        model=valid.model,
                        backend=valid.backend,
                        task=valid.task,
                        method_role=role,
                        block=block,
                        load=load,
                        terminal_receipt_sha256=terminal,
                        request_identity_sha256=content_sha256(
                            {"request-identity": [block, load]}
                        ),
                        completed_output_tokens=1_000 + role_index * 10,
                        slo_goodput_numerator_tokens=(
                            0 if role == "TTS" else 900 + role_index * 10
                        ),
                        scored_window_ns=1_000_000_000,
                        slo_accounting_sha256=content_sha256(
                            {"slo-accounting": cell_id}
                        ),
                        slo_policy_sha256=E0_FINAL_SLO_GOODPUT_POLICY_SHA256,
                    )
                )
                completion_rows.append(
                    E0FinalCellCompletion(
                        materialized_cell_id=cell_id,
                        execution_binding_sha256=content_sha256({"execution": cell_id}),
                        terminal_receipt_sha256=terminal,
                        native_result_proof_semantic_sha256=content_sha256(
                            {"native": cell_id}
                        ),
                        stage_itl_proof_semantic_sha256=content_sha256(
                            {"itl": cell_id}
                        ),
                    )
                )
    completion = E0FinalCompletionReceipt(
        schema_version=1,
        protocol_lock_sha256=content_sha256("protocol-lock"),
        registry_sha256=registry.sha256,
        prior_registry_verification_receipt_sha256=content_sha256("prior"),
        current_registry_verification_receipt_sha256=content_sha256("current"),
        materialization_receipt_sha256=content_sha256("materialization"),
        coverage_receipt_sha256=content_sha256("coverage"),
        stage_source_binding_sha256=content_sha256("stage-source"),
        evidence_manifest_sha256=content_sha256("evidence"),
        inventory_sha256=content_sha256("inventory"),
        rebuild_artifact_sha256=content_sha256("rebuild"),
        selected_final_prefix=tuple(range(4, 16)),
        valid_compatibility_count=1,
        cells=tuple(sorted(completion_rows, key=lambda row: row.materialized_cell_id)),
        protocol_sha256=E0_FINAL_COMPLETION_PROTOCOL_SHA256,
    )
    return E0FinalAnalysisProjection(
        schema_version=1,
        completion_receipt=completion,
        compatibility_decisions=decisions,
        cells=tuple(sorted(analysis_rows, key=lambda row: row.materialized_cell_id)),
    )


def _all_na_projection(registry: ExperimentRegistry) -> E0FinalAnalysisProjection:
    decisions = tuple(
        sorted(
            (
                E0CompatibilityDecision(
                    model=model,
                    backend=backend,
                    task=task,
                    disposition="N/A",
                    reason_code="official_interface_incompatible",
                    interface_sha256=content_sha256(
                        {"all-na-interface": [model, backend, task]}
                    ),
                    task_native_workload_sha256=content_sha256(
                        {"all-na-workload": [model, backend, task]}
                    ),
                )
                for model in E0_MODELS
                for backend in E0_BACKENDS
                for task in E0_TASKS
            ),
            key=lambda row: row.decision_id,
        )
    )
    completion = E0FinalCompletionReceipt(
        schema_version=1,
        protocol_lock_sha256=content_sha256("all-na-protocol-lock"),
        registry_sha256=registry.sha256,
        prior_registry_verification_receipt_sha256=content_sha256("all-na-prior"),
        current_registry_verification_receipt_sha256=content_sha256("all-na-current"),
        materialization_receipt_sha256=content_sha256("all-na-materialization"),
        coverage_receipt_sha256=content_sha256("all-na-coverage"),
        stage_source_binding_sha256=content_sha256("all-na-stage-source"),
        evidence_manifest_sha256=content_sha256("all-na-evidence"),
        inventory_sha256=content_sha256("all-na-inventory"),
        rebuild_artifact_sha256=content_sha256("all-na-rebuild"),
        selected_final_prefix=(),
        valid_compatibility_count=0,
        cells=(),
        protocol_sha256=E0_FINAL_COMPLETION_PROTOCOL_SHA256,
    )
    return E0FinalAnalysisProjection(
        schema_version=1,
        completion_receipt=completion,
        compatibility_decisions=decisions,
        cells=(),
    )


def test_formal_fdr_uses_signed_staged_registry_and_explicit_na_exclusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_industrial_registry()
    assert registry.materialization_mode == "signed_staged"
    assert not registry.cells_for("E0")
    projection = _formal_projection(registry)
    monkeypatch.setattr(
        "lightcone_spec.experiments.breadth_fdr_authority."
        "E0_BREADTH_BOOTSTRAP_REPETITIONS",
        100,
    )
    universe = registered_formal_e0_breadth_hypotheses(projection)
    receipt = reduce_formal_e0_breadth_fdr_from_projection(registry, projection)
    assert len(universe) == len(receipt.hypotheses) == 756
    assert len(receipt.decisions) == 7
    assert sum(row.status == "EXCLUDED_NA" for row in receipt.hypotheses) == 749
    assert all(
        row.exclusion_reason == "official_interface_incompatible"
        for row in receipt.hypotheses
        if row.status == "EXCLUDED_NA"
    )
    assert receipt.formal_result_authorized is True
    assert receipt.primary_family_eligible is False

    encoded = formal_e0_breadth_fdr_receipt_to_dict(receipt)
    assert formal_e0_breadth_fdr_receipt_from_dict(encoded) == receipt
    encoded["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest differs"):
        formal_e0_breadth_fdr_receipt_from_dict(encoded)


def test_formal_fdr_all_na_is_complete_without_fake_blocks_or_tests() -> None:
    registry = build_industrial_registry()
    projection = _all_na_projection(registry)
    receipt = reduce_formal_e0_breadth_fdr_from_projection(registry, projection)
    assert projection.completion_receipt.selected_final_prefix == ()
    assert projection.completion_receipt.valid_compatibility_count == 0
    assert projection.cells == ()
    assert len(receipt.hypotheses) == 756
    assert all(row.status == "EXCLUDED_NA" for row in receipt.hypotheses)
    assert receipt.decisions == ()
    assert receipt.formal_result_authorized is True

    foreign_cell = _formal_projection(registry).cells[0]
    with pytest.raises(ValueError, match="cell universe is not exact"):
        E0FinalAnalysisProjection(
            schema_version=1,
            completion_receipt=projection.completion_receipt,
            compatibility_decisions=projection.compatibility_decisions,
            cells=(foreign_cell,),
        )
    values = {
        **{
            name: getattr(projection.completion_receipt, name)
            for name in E0FinalCompletionReceipt.__dataclass_fields__
        },
        "selected_final_prefix": tuple(range(4, 16)),
    }
    with pytest.raises(ValueError, match="cardinality/prefix"):
        E0FinalCompletionReceipt(**values)


def test_e0_slo_goodput_excludes_slow_request_and_uses_scored_window() -> None:
    fast = SimpleNamespace(
        request_id="fast",
        input_token_ids=(1,),
        output_token_ids=(2, 3),
        output_tokens=2,
        p99_itl_ns=Fraction(1_000_000),
    )
    slow = SimpleNamespace(
        request_id="slow",
        input_token_ids=(1,),
        output_token_ids=(2, 3, 4),
        output_tokens=3,
        p99_itl_ns=Fraction(101_000_000),
    )
    timing = SimpleNamespace(
        requests=(
            SimpleNamespace(
                request_id="fast",
                request_started_ns=100,
                token_observed_ns=(1_000_100, 2_000_100),
                request_terminal_ns=1_000_000_100,
            ),
            SimpleNamespace(
                request_id="slow",
                request_started_ns=200,
                token_observed_ns=(1_000_200, 102_000_200, 203_000_200),
                request_terminal_ns=2_000_000_200,
            ),
        )
    )
    qualified_tokens, scored_window_ns, accounting = _e0_slo_goodput_inputs(
        SimpleNamespace(metrics=(fast, slow), timing=timing)
    )
    assert qualified_tokens == 2
    assert scored_window_ns == 2_000_000_100
    assert len(accounting) == 64


def test_e0_all_slo_failures_are_measured_zero_but_missing_timing_is_rejected() -> None:
    metric = SimpleNamespace(
        request_id="slow",
        input_token_ids=(1,),
        output_token_ids=(2,),
        output_tokens=1,
        p99_itl_ns=Fraction(101_000_000),
    )
    timing = SimpleNamespace(
        requests=(
            SimpleNamespace(
                request_id="slow",
                request_started_ns=1,
                token_observed_ns=(101_000_001,),
                request_terminal_ns=200_000_001,
            ),
        )
    )
    qualified_tokens, scored_window_ns, _ = _e0_slo_goodput_inputs(
        SimpleNamespace(metrics=(metric,), timing=timing)
    )
    assert qualified_tokens == 0
    assert scored_window_ns == 200_000_000
    with pytest.raises(ValueError, match="coverage differs"):
        _e0_slo_goodput_inputs(
            SimpleNamespace(metrics=(metric,), timing=SimpleNamespace(requests=()))
        )


def test_e0_analysis_rejects_foreign_policy_window_and_old_throughput_shape() -> None:
    values = {
        "materialized_cell_id": content_sha256("cell"),
        "compatibility_decision_id": content_sha256("decision"),
        "model": E0_MODELS[0],
        "backend": E0_BACKENDS[0],
        "task": E0_TASKS[0],
        "method_role": "Static",
        "block": 4,
        "load": "common_slo_load",
        "terminal_receipt_sha256": content_sha256("terminal"),
        "request_identity_sha256": content_sha256("requests"),
        "completed_output_tokens": 10,
        "slo_goodput_numerator_tokens": 8,
        "scored_window_ns": 1,
        "slo_accounting_sha256": content_sha256("slo"),
        "slo_policy_sha256": E0_FINAL_SLO_GOODPUT_POLICY_SHA256,
    }
    E0FinalAnalysisCell(**values)
    with pytest.raises(ValueError, match="fields are not exact"):
        E0FinalAnalysisCell(**{**values, "scored_window_ns": 0})
    with pytest.raises(ValueError, match="fields are not exact"):
        E0FinalAnalysisCell(**{**values, "slo_policy_sha256": "0" * 64})
    with pytest.raises(TypeError, match="unexpected keyword"):
        E0FinalAnalysisCell(
            **{
                key: value
                for key, value in values.items()
                if key
                not in {
                    "slo_goodput_numerator_tokens",
                    "scored_window_ns",
                    "slo_accounting_sha256",
                    "slo_policy_sha256",
                }
            },
            summed_request_latency_ns=1,
        )
