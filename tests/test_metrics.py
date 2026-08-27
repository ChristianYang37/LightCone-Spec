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
    paired_bca_interval,
    paired_block_statistics,
    paired_relative_bca_interval,
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
