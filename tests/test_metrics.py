import math

import pytest

from lightcone_spec.metrics import (
    SAFETY_COUNTERS,
    benjamini_hochberg,
    block_bootstrap_interval,
    choose_final_blocks,
    committed_goodput,
    hierarchical_request_interval,
    holm_decisions,
    paired_bca_interval,
    paired_block_statistics,
    paired_relative_bca_interval,
    validate_scientific_metrics,
)
from lightcone_spec.runner import _natural_spline_fit


def test_goodput_bootstrap_holm_and_power():
    assert committed_goodput(200, 4.0) == 50.0
    estimate, low, high = paired_bca_interval([2, 3, 4, 5], [1, 1, 2, 3], resamples=500, seed=0)
    assert low <= estimate <= high
    relative = paired_relative_bca_interval(
        [101, 102, 103, 104], [100, 100, 100, 100], resamples=500
    )
    assert relative[1] <= relative[0] <= relative[2]
    assert holm_decisions([0.001, 0.02, 0.8]) == (True, True, False)
    assert choose_final_blocks([0.04, 0.04, 0.04, 0.04]) == 12


def test_safety_metrics_reject_numerical_failures():
    metrics = {
        "committed_tokens": 100,
        "duration_seconds": 1.0,
        "goodput": 100.0,
        "peak_hbm_bytes": 1,
        "kv_capacity": 1,
        "exactness_violations": 1,
        "itl_p99_ms": 1.0,
        **{name: 0 for name in SAFETY_COUNTERS if name != "exactness_violations"},
    }
    with pytest.raises(RuntimeError, match="exactness"):
        validate_scientific_metrics(metrics)
    with pytest.raises(ValueError):
        committed_goodput(1, math.nan)


def test_fdr_and_time_block_bootstrap():
    assert benjamini_hochberg([0.001, 0.02, 0.9]) == (True, True, False)
    point, low, high = block_bootstrap_interval([10, 11, 12, 13], resamples=500)
    assert low <= point <= high
    point, low, high = hierarchical_request_interval(
        {
            block: [(10 + block, 1.0), (20 + block, 2.0)]
            for block in range(4)
        },
        resamples=500,
    )
    assert low <= point <= high


def test_final_block_statistics_are_paired_and_corrected():
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
    assert len(result) == 1
    assert result[0]["candidate"] == "lightcone"
    assert result[0]["blocks"] == list(range(4, 16))
    assert result[0]["ci95_relative_low"] > 0
    assert result[0]["reducer"] == "paired_log_goodput_bca"
    assert result[0]["holm_reject"] is True


def test_holm_only_marks_two_confirmatory_comparisons():
    rows = []
    for block in range(4, 16):
        for method, goodput, slope in (
            ("static", 100.0, 0.01),
            ("tts", 101.0, 0.02),
            ("l0_naive", 102.0, 0.03),
            ("lightcone", 105.0, 0.05),
        ):
            rows.append(({
                "method": method, "model": "m", "backend": "DFLASH",
                "task": "t", "context": 4096, "load": "c1", "block": block,
                "parameters": {},
            }, {"goodput": goodput + block * slope, "request_count": 10}))
    result = paired_block_statistics(rows)
    corrected = {(row["candidate"], row["baseline"]) for row in result if row["holm_reject"] is not None}
    assert corrected == {("lightcone", "static"), ("lightcone", "tts")}


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
    assert [(row["candidate"], row["baseline"]) for row in result] == [
        ("lightcone", "target_only")
    ]


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
