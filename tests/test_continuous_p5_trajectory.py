from __future__ import annotations

import pytest


def test_continuous_prefix_windows_accepts_one_contiguous_trajectory():
    from lightcone_spec.cli.main import _continuous_prefix_windows

    assert _continuous_prefix_windows(
        {
            "p5_continuous_prefix_windows": [
                [128, 10_240],
                [10_240, 20_480],
                [20_480, 30_720],
                [30_720, 40_960],
            ]
        }
    ) == (
        (128, 10_240),
        (10_240, 20_480),
        (20_480, 30_720),
        (30_720, 40_960),
    )


@pytest.mark.parametrize(
    "windows",
    (
        [],
        [[-1, 10]],
        [[0, 0]],
        [[0, 10], [11, 20]],
        [[0, 10], [9, 20]],
        [[0, 10], [10, 9]],
        [[0, True]],
        [[0, 10.0]],
        [[0, 10, 20]],
    ),
)
def test_continuous_prefix_windows_rejects_ambiguous_bucketing(windows):
    from lightcone_spec.cli.main import _continuous_prefix_windows
    from lightcone_spec.exit_codes import ArtifactValidationFailure

    with pytest.raises(ArtifactValidationFailure):
        _continuous_prefix_windows(
            {"p5_continuous_prefix_windows": windows}
        )


def test_continuous_prefix_windows_is_opt_in():
    from lightcone_spec.cli.main import _continuous_prefix_windows

    assert _continuous_prefix_windows({}) == ()


def test_p5_stride_comes_from_run_manifest_with_visible_legacy_default():
    from lightcone_spec.cli.main import _p5_update_stride

    assert _p5_update_stride({"stride": 1}) == 1
    assert _p5_update_stride({"stride": 4}) == 4
    assert _p5_update_stride({"update_stride": 8}) == 8
    assert _p5_update_stride({}) == 4


@pytest.mark.parametrize("stride", (0, -1, True, 1.5, "4"))
def test_p5_stride_rejects_ambiguous_manifest_values(stride):
    from lightcone_spec.cli.main import _p5_update_stride
    from lightcone_spec.exit_codes import ArtifactValidationFailure

    with pytest.raises(ArtifactValidationFailure):
        _p5_update_stride({"stride": stride})


def _round(
    method: str,
    *,
    start: int,
    end: int,
    accepted: int,
    prompt: str = "prompt-0",
) -> dict:
    return {
        "method": method,
        "model_pair": "qwen3_4b_dflash16",
        "weight_update_mode": "lora",
        "dataset": "math500",
        "lifecycle": "stream",
        "offered_concurrency": 1,
        "context_length": end,
        "request_id": f"{method}-{prompt}-{end}",
        "prompt_cluster": prompt,
        "seed": 0,
        "prefix_len_before": start,
        "draft_tokens": 4,
        "verify_len": 5,
        "accepted_drafts": accepted,
        "committed_per_verify": accepted + 1,
        "target_calls": 1,
        "draft_cuda_us": 1.0,
        "verify_cuda_us": 1.0,
        "accept_cuda_us": 1.0,
        "batch_size": 1,
        "version_canary_ok": True,
        "trajectory_kind": "continuous_prefix",
        "initial_prefix_len": 128,
        "prefix_window_start": start,
        "prefix_window_end": end,
        "benchmark_repetitions": 1,
    }


def test_continuous_lcag_excludes_the_pre_4k_window():
    import pandas as pd

    from lightcone_spec.statistics.tables import long_context_acceptance_table

    rows = []
    for method, early, long in (("static", 1, 1), ("tts", 4, 1)):
        rows.extend(
            (
                _round(method, start=128, end=4096, accepted=early),
                _round(method, start=4096, end=10240, accepted=long),
            )
        )
    table = long_context_acceptance_table(pd.DataFrame(rows), b=50)
    tts = table[table["method"] == "tts"]

    assert tts.loc[tts["context_length"] == 4096, "acceptance_gain_vs_baseline"].iloc[0] == 3
    assert set(tts["lcag"]) == {0.0}


def test_one_prompt_cannot_confirm_benefit_or_pass_scientific_gate():
    import pandas as pd

    from lightcone_spec.cli.main import _p5_scientific_sample_pass
    from lightcone_spec.statistics.tables import long_context_acceptance_table

    rows = []
    for method, accepted in (("static", 1), ("tts", 2)):
        rows.extend(
            (
                _round(method, start=128, end=4096, accepted=accepted),
                _round(method, start=4096, end=10240, accepted=accepted),
            )
        )
    table = long_context_acceptance_table(pd.DataFrame(rows), b=50)
    tts = table[table["method"] == "tts"]

    assert set(tts["gain_prompt_clusters"]) == {1}
    assert set(tts["benefit_onset_status"]) == {"candidate"}
    assert not _p5_scientific_sample_pass(1, 5)
    assert not _p5_scientific_sample_pass(2, 1)
    assert not _p5_scientific_sample_pass(2, 2)
    assert _p5_scientific_sample_pass(2, 5)


def test_continuous_shape_onset_and_window_dominance_keep_interval_semantics():
    import pandas as pd

    from lightcone_spec.cli.main import _p5_window_dominance
    from lightcone_spec.statistics.tables import (
        acceptance_elasticity_table,
        long_context_acceptance_table,
    )

    rows = []
    for prompt in ("prompt-0", "prompt-1"):
        for method, accepted in (("static", 1), ("tts", 2)):
            rows.extend(
                (
                    _round(
                        method,
                        start=128,
                        end=4096,
                        accepted=accepted,
                        prompt=prompt,
                    ),
                    _round(
                        method,
                        start=4096,
                        end=10240,
                        accepted=accepted,
                        prompt=prompt,
                    ),
                )
            )
    rounds = pd.DataFrame(rows)
    table = long_context_acceptance_table(rounds, b=50)
    tts = table[table["method"] == "tts"]

    assert set(tts["benefit_onset_status"]) == {"confirmed"}
    assert tts["benefit_onset_context"].isna().all()
    assert set(tts["benefit_onset_window_start"]) == {128}
    assert set(tts["benefit_onset_window_end"]) == {4096}
    passed, failures = _p5_window_dominance(
        tts, continuous_trajectory=True
    )
    assert passed
    assert failures == []

    shape = acceptance_elasticity_table(rounds, b=50)
    assert set(shape["shape_semantics"]) == {"window_average_shape_proxy"}

    failed = tts.copy()
    failed.loc[
        failed["prefix_window_start"] == 4096, "acceptance_gain_ci_low"
    ] = 0.0
    passed, failures = _p5_window_dominance(
        failed, continuous_trajectory=True
    )
    assert not passed
    assert failures == [
        {
            "prefix_window_start": 4096,
            "prefix_window_end": 10240,
            "acceptance_gain_ci_low": 0.0,
            "paired_prompt_clusters": 2,
            "reasons": ["acceptance_gain_ci_low_not_positive"],
        }
    ]
