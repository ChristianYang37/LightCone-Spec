import math

import pytest

from lightcone_spec.metrics import (
    SAFETY_COUNTERS,
    benjamini_hochberg,
    block_bootstrap_interval,
    choose_final_blocks,
    committed_goodput,
    holm_decisions,
    paired_bca_interval,
    paired_block_statistics,
    validate_scientific_metrics,
)


def test_goodput_bootstrap_holm_and_power():
    assert committed_goodput(200, 4.0) == 50.0
    estimate, low, high = paired_bca_interval([2, 3, 4, 5], [1, 1, 2, 3], resamples=500, seed=0)
    assert low <= estimate <= high
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
                    {"goodput": goodput},
                )
            )
    result = paired_block_statistics(rows)
    assert len(result) == 1
    assert result[0]["candidate"] == "lightcone"
    assert result[0]["blocks"] == list(range(4, 16))
    assert result[0]["ci95_low"] > 0
    assert result[0]["holm_reject"] is True
