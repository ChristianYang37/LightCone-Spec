import gzip
import json
import math

import pytest
import torch

import lightcone_spec.runner as runner
from lightcone_spec.metrics import (
    SAFETY_COUNTERS,
    benjamini_hochberg,
    block_bootstrap_interval,
    committed_goodput,
    hierarchical_request_interval,
    holm_decisions,
    normalize_attempt_semantics,
    paired_bca_interval,
    paired_block_statistics,
    paired_relative_bca_interval,
    per_user_generation_speed,
    summarize_attempts,
    validate_scientific_metrics,
)
from lightcone_spec.runner import _confirmatory_holm, _natural_spline_fit
from lightcone_spec.state import StateStore


def test_goodput_bootstrap_and_holm():
    assert committed_goodput(200, 4.0) == 50.0
    estimate, low, high = paired_bca_interval([2, 3, 4, 5], [1, 1, 2, 3], resamples=500, seed=0)
    assert low <= estimate <= high
    relative = paired_relative_bca_interval(
        [101, 102, 103, 104], [100, 100, 100, 100], resamples=500
    )
    assert relative[1] <= relative[0] <= relative[2]
    assert holm_decisions([0.001, 0.02, 0.8]) == (True, True, False)


def test_native_per_user_speed_and_historical_tts_concurrency(tmp_path):
    requests = [
        {"native_token_timestamps_ns": [1_000_000_000, 1_010_000_000, 1_020_000_000]},
        {"native_token_timestamps_ns": [2_000_000_000, 2_020_000_000]},
    ]
    assert per_user_generation_speed(requests) == pytest.approx(75.0)
    with gzip.open(tmp_path / "requests.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in requests:
            stream.write(json.dumps(row) + "\n")
    config, metrics = normalize_attempt_semantics(
        {"method": "tts", "load": "c2", "parameters": {}},
        {"goodput": 200.0, "per_user_generation_speed": 100.0},
        tmp_path,
    )
    assert config["parameters"]["declared_concurrency"] == 2
    assert config["parameters"]["dispatcher_concurrency"] == 1
    assert metrics["effective_load"] == "c1"
    assert metrics["per_user_generation_speed"] == pytest.approx(75.0)
    raw_metrics = {"goodput": 200.0, "per_user_generation_speed": 100.0}
    (tmp_path / "config.json").write_text(
        json.dumps({"method": "tts", "load": "c2", "parameters": {}})
    )
    (tmp_path / "metrics.json").write_text(json.dumps(raw_metrics))
    summary = summarize_attempts([tmp_path], tmp_path / "summary")
    assert summary.iloc[0]["per_user_generation_speed"] == pytest.approx(75.0)
    assert json.loads((tmp_path / "metrics.json").read_text()) == raw_metrics


def test_tts_source_policy_kl_has_zero_one_step_gradient():
    source_logits = torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64)
    teacher = torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64)
    gradients = []
    for coefficient in (0.0, 0.1, 1.0, 10.0):
        logits = source_logits.clone().requires_grad_(True)
        source = torch.softmax(source_logits, dim=-1)
        target = torch.softmax(teacher, dim=-1)
        log_q = torch.log_softmax(logits, dim=-1)
        distillation = torch.sum(target * (torch.log(target) - log_q))
        proximal = torch.sum(source * (torch.log(source) - log_q))
        (distillation + coefficient * proximal).backward()
        gradients.append(logits.grad.clone())
    assert all(
        torch.allclose(gradients[0], gradient, atol=1e-14, rtol=0) for gradient in gradients[1:]
    )


def test_dspark_confidence_selection_aggregates_threshold_segments(monkeypatch):
    rows = []
    for weight in (0.05, 0.1, 0.25, 0.5, 1.0):
        for threshold in range(10):
            rows.append(
                (
                    {
                        "parameters": {
                            "workload": "confidence_calibration",
                            "confidence_loss_weight": weight,
                            "confidence_threshold": threshold / 10,
                        }
                    },
                    {
                        "slo_pass": True,
                        "feasible": True,
                        **{counter: 0 for counter in SAFETY_COUNTERS},
                        "confidence_brier": abs(weight - 0.25) + 0.1,
                        "confidence_ece": abs(weight - 0.25) + 0.05,
                        "goodput": 100.0,
                        "peak_hbm_bytes": 10,
                        "confidence_probabilities": [0.2, 0.8],
                        "confidence_outcomes": [0.0, 1.0],
                    },
                )
            )
    monkeypatch.setattr(runner, "_metric_rows", lambda state, node: rows)
    weight = runner._select_confidence_weight(object())
    temperature = runner._select_confidence_temperature(object(), weight)
    assert weight == 0.25
    assert 0.25 <= temperature <= 4.0


def test_tts_recipe_groups_confirmation_stimuli(monkeypatch):
    rows = []
    for block in range(4):
        rows.append(
            (
                {
                    "parameters": {
                        "learning_rate": 1e-4,
                        "stride": 50,
                        "workload": "tts_calibration_confirmation",
                        "confirmation_block": block,
                        "stimulus_id": f"confirmation-block-{block}",
                    }
                },
                {
                    "goodput": 100.0 + block,
                    "peak_hbm_bytes": 10,
                    "itl_p99_ms": 5.0,
                },
            )
        )

    monkeypatch.setattr(
        runner,
        "_metric_rows",
        lambda state, node: rows if node == "TTS-Cal-confirmation" else [],
    )
    recipe = runner._select_tts_recipe(None)
    assert recipe == {"learning_rate": 1e-4, "stride": 50}


def test_e2_ranking_uses_static_goodput_and_tts_native_user_speed(monkeypatch):
    rows = [
        ({"method": "static", "parameters": {}}, {"goodput": 100.0}),
        (
            {"method": "tts", "parameters": {}},
            {"goodput": 10.0, "per_user_generation_speed": 100.0},
        ),
        (
            {"method": "lightcone_candidate", "parameters": {"name": "aggregate"}},
            {
                "goodput": 120.0,
                "per_user_generation_speed": 80.0,
                "peak_hbm_bytes": 1,
                "itl_p99_ms": 1.0,
                "exposed_update_ms": 1.0,
            },
        ),
        (
            {"method": "lightcone_candidate", "parameters": {"name": "balanced"}},
            {
                "goodput": 90.0,
                "per_user_generation_speed": 110.0,
                "peak_hbm_bytes": 1,
                "itl_p99_ms": 1.0,
                "exposed_update_ms": 1.0,
            },
        ),
    ]
    monkeypatch.setattr(runner, "_metric_rows", lambda state, node: rows)
    assert runner._rank_e2_candidates(object(), "E2-r0", 1) == [{"name": "balanced"}]


def test_e2_audit_preserves_existing_order_when_scientific_set_matches(tmp_path):
    state = StateStore(tmp_path)
    state.set_selection("e2_round_0", [{"rank": 1}, {"rank": 8}])
    corrected = [{"rank": 8, "metric_semantics": "per_request_native_v2"}, {"rank": 1}]
    assert runner._preserve_or_audit_e2_selection(
        state, "E2-r0", "e2_round_0", corrected
    ) == [{"rank": 1}, {"rank": 8}]
    audit = json.loads((tmp_path / "stages/E2-r0/concurrency_metric_audit.json").read_text())
    assert audit["same_scientific_set"] is True
    assert audit["action"] == "preserved_existing_order"


def test_e5_frontier_rejects_nominal_concurrency_speed_fallback(monkeypatch):
    class State:
        def selection(self, name, default=None):
            return "static" if name == "e5_operational_baseline" else default

    monkeypatch.setattr(
        runner,
        "_metric_rows",
        lambda state, node: [
            (
                {
                    "method": "static",
                    "backend": "DFLASH",
                    "block": 0,
                    "load": "closed_loop_c2",
                },
                {"goodput": 200.0},
            )
        ],
    )
    with pytest.raises(runner.ScientificFailure, match="native per-request"):
        runner._e5_frontier_statistic(State())


def test_final_stage_requires_completed_pilot():
    class State:
        def stage_status(self, node):
            return {"preflight": "completed", "E5-pilot": "skipped"}.get(node)

        def selection(self, name, default=None):
            return default

    assert runner._dependency_reason(None, State(), "E5-final") == "E5-pilot did not complete"


def test_safety_metrics_reject_numerical_failures():
    metrics = {
        "committed_tokens": 100,
        "duration_seconds": 1.0,
        "goodput": 100.0,
        "peak_hbm_bytes": 1,
        "kv_capacity": 1,
        "itl_p99_ms": 1.0,
        **{name: int(name == "nonfinite_updates") for name in SAFETY_COUNTERS},
    }
    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_scientific_metrics(metrics)
    with pytest.raises(ValueError):
        committed_goodput(1, math.nan)


def test_fdr_and_time_block_bootstrap():
    assert benjamini_hochberg([0.001, 0.02, 0.9]) == (True, True, False)
    point, low, high = block_bootstrap_interval([10, 11, 12, 13], resamples=500)
    assert low <= point <= high
    point, low, high = hierarchical_request_interval(
        {block: [(10 + block, 1.0), (20 + block, 2.0)] for block in range(4)},
        resamples=500,
    )
    assert low <= point <= high


def test_final_block_statistics_are_paired():
    rows = []
    for block in range(4, 16):
        for method, goodput in (("static", 100.0), ("lightcone", 104.0 + block / 100)):
            rows.append(
                (
                    {
                        "method": method,
                        "model": "m",
                        "backend": "DFLASH",
                        "task": "t",
                        "context": 40928,
                        "load": "c4",
                        "block": block,
                        "parameters": {"regime": "long"},
                    },
                    {"goodput": goodput, "request_count": 10},
                )
            )
    result = paired_block_statistics(rows)
    focal = next(row for row in result if row["baseline"] == "static")
    assert focal["candidate"] == "lightcone"
    assert focal["blocks"] == list(range(4, 16))
    assert focal["ci95_relative_low"] > 0
    assert focal["reducer"] == "paired_log_goodput_bca"
    assert focal["holm_reject"] is None


def test_pairing_separates_effective_concurrency():
    rows = []
    for block in range(4):
        for method, effective_load in (("static", "c2"), ("lightcone", "c1")):
            rows.append(
                (
                    {
                        "method": method,
                        "model": "m",
                        "backend": "DFLASH",
                        "task": "t",
                        "context": 40928,
                        "load": "c2",
                        "block": block,
                        "parameters": {
                            "regime": "long",
                            "effective_load": effective_load,
                        },
                    },
                    {"goodput": 100.0, "request_count": 1},
                )
            )
    assert paired_block_statistics(rows) == []


def test_holm_combines_only_the_three_preregistered_hypotheses(tmp_path):
    state = StateStore(tmp_path)
    path = tmp_path / "stages" / "E3b-final"
    path.mkdir(parents=True)
    rows = [
        {
            "candidate": "lightcone",
            "baseline": baseline,
            "workload": "primary_long_history",
            "context": 32768,
            "p_value": p_value,
        }
        for baseline, p_value in (("tts", 0.001), ("operational_baseline", 0.02))
    ]
    (path / "statistics.json").write_text(json.dumps(rows))
    combined = _confirmatory_holm(
        state,
        {
            "hypothesis": "H3",
            "candidate": "lightcone",
            "baseline": "operational_baseline",
            "metric": "maximum_slo_feasible_rate",
            "p_value": 0.8,
        },
    )
    assert [row["hypothesis"] for row in combined] == ["H1", "H2", "H3"]
    assert [row["holm_reject"] for row in combined] == [True, True, False]


def test_pairing_uses_block_stimulus_but_ignores_runtime_backend_and_width():
    rows = []
    for block in range(4):
        stimulus = f"shared-block-{block}"
        for method, backend, width, goodput in (
            ("target_only", "NONE", None, 100.0),
            ("lightcone", "DFLASH", 16, 104.0 + block / 100),
        ):
            rows.append(
                (
                    {
                        "method": method,
                        "model": "m",
                        "backend": backend,
                        "task": "t",
                        "context": 40928,
                        "load": "c4",
                        "width": width,
                        "block": block,
                        "parameters": {
                            "comparison_backend": "DFLASH",
                            "width_panel": "deployment_optimal",
                            "stimulus_id": stimulus,
                        },
                    },
                    {"goodput": goodput, "request_count": 10},
                )
            )
    result = paired_block_statistics(rows)
    assert [(row["candidate"], row["baseline"]) for row in result] == [("lightcone", "target_only")]


def test_natural_spline_uses_fixed_interior_knots_and_natural_boundaries():
    contexts = pytest.importorskip("numpy").array(
        [1024, 2048, 4096, 8192, 16384, 24576, 32768, 40928], dtype=float
    )
    x = pytest.importorskip("numpy").log(contexts)
    y = 0.2 * x + 0.01 * x**2
    fitted, elasticity, curvature = _natural_spline_fit(x, y, x)
    assert len(fitted) == len(contexts)
    assert len(elasticity) == len(contexts)
    assert abs(curvature[0]) < 1e-8
    assert abs(curvature[-1]) < 1e-8


def test_context_spline_ignores_infeasible_rows_without_goodput(monkeypatch):
    config = {
        "method": "static",
        "context": 40928,
        "load": "c64",
        "parameters": {"regime": "long_input_short_output"},
    }
    monkeypatch.setattr(runner, "_metric_rows", lambda state, node: [(config, {"feasible": False})])

    class EmptyState:
        def completed_attempt_dirs(self, node):
            return ()

    assert runner._context_splines(EmptyState(), "E3a") == []


def test_context_spline_skips_unbracketed_fixed_knots(monkeypatch):
    rows = [
        (
            {
                "method": "static",
                "context": context,
                "load": "c1",
                "parameters": {"regime": "long_input_short_output"},
            },
            {"goodput": 10.0},
        )
        for context in (4096, 16384, 32768, 40928)
    ]
    monkeypatch.setattr(runner, "_metric_rows", lambda state, node: rows)

    class EmptyState:
        def completed_attempt_dirs(self, node):
            return ()

    assert runner._context_splines(EmptyState(), "E3a") == []


def test_attempt_summary_serializes_mixed_nested_parquet_columns(tmp_path):
    pytest.importorskip("pyarrow")
    attempts = []
    for index, layout in enumerate(([["weight", [4, 4]]], "N/A")):
        directory = tmp_path / f"attempt-{index:02d}"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"method": "static"}))
        (directory / "metrics.json").write_text(json.dumps({"parameter_layout": layout}))
        attempts.append(directory)
    output = tmp_path / "summary"
    summarize_attempts(attempts, output)
    assert (output / "summary.csv").is_file()
    assert (output / "summary.parquet").is_file()
