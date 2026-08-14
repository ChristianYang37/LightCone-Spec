from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lightcone_spec.experiments.statistics import (
    MAXIMUM_FINAL_BLOCKS,
    MINIMUM_FINAL_BLOCKS,
    P99_MINIMUM_COMPLETIONS,
    PRIMARY_CONTRASTS,
    SECONDARY_CONTRASTS,
    BootstrapInterval,
    HardwareBlockObservation,
    HardwareEnvelope,
    PairedBcaContrast,
    PilotBlock,
    SloRequest,
    account_slo,
    benjamini_hochberg,
    guard_p99_claim,
    hierarchical_block_request_bootstrap,
    holm_primary_contrasts,
    paired_bca_contrast,
    preregister_power_sizing,
    time_block_bootstrap,
    validate_final_block_ids,
    validate_hardware_block,
    validate_hardware_blocks,
)


def pilot_blocks(*, noisy: bool = False) -> tuple[PilotBlock, ...]:
    multipliers = (0.90, 1.10, 0.85, 1.15) if noisy else (0.99, 1.01, 1.00, 1.02)
    return tuple(
        PilotBlock(
            block_id=f"pilot-{index}",
            static_goodput=100.0,
            tts_goodput=101.0,
            lightcone_goodput=103.0 * multiplier,
        )
        for index, multiplier in enumerate(multipliers)
    )


def test_power_sizing_uses_four_excluded_pilots_and_fixed_range() -> None:
    plan = preregister_power_sizing(pilot_blocks())
    assert plan.status == "READY"
    assert plan.selected_final_blocks == MINIMUM_FINAL_BLOCKS == 12
    assert plan.maximum_final_blocks == MAXIMUM_FINAL_BLOCKS == 20
    assert plan.adjusted_alpha == pytest.approx(0.025)
    assert {cell.contrast for cell in plan.power_grid} == set(PRIMARY_CONTRASTS)
    assert len(plan.power_grid) == 2 * (20 - 12 + 1)
    assert all(plan.power(name, 12) >= 0.80 for name in PRIMARY_CONTRASTS)

    final_ids = tuple(f"final-{index}" for index in range(12))
    assert validate_final_block_ids(plan, final_ids) == final_ids
    with pytest.raises(ValueError, match="excluded"):
        validate_final_block_ids(plan, ("pilot-0", *final_ids[1:]))
    with pytest.raises(ValueError, match="count"):
        validate_final_block_ids(plan, final_ids[:-1])


def test_power_sizing_marks_twenty_blocks_underpowered_without_extension() -> None:
    plan = preregister_power_sizing(pilot_blocks(noisy=True))
    assert plan.status == "UNDERPOWERED"
    assert plan.selected_final_blocks is None
    assert any(plan.power(name, 20) < 0.80 for name in PRIMARY_CONTRASTS)
    with pytest.raises(ValueError, match="UNDERPOWERED"):
        validate_final_block_ids(plan, tuple(f"final-{index}" for index in range(20)))


@pytest.mark.parametrize(
    "blocks, message",
    [
        (pilot_blocks()[:-1], "exactly four"),
        (
            tuple(replace(block, block_id="same") for block in pilot_blocks()),
            "unique",
        ),
        (
            tuple(replace(block, lightcone_goodput=100.0) for block in pilot_blocks()),
            "positive finite variance",
        ),
    ],
)
def test_power_sizing_fails_closed_on_invalid_pilot_evidence(
    blocks: tuple[PilotBlock, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        preregister_power_sizing(blocks)
    with pytest.raises(ValueError, match="fixed at 12--20"):
        preregister_power_sizing(pilot_blocks(), maximum_final_blocks=21)
    with pytest.raises(ValueError, match="preregistered and fixed"):
        preregister_power_sizing(pilot_blocks(), target_power=0.79)


def paired_values(ratio: float) -> dict[str, tuple[float, float]]:
    deviations = (-0.004, 0.003, -0.002, 0.005, 0.0, 0.002) * 2
    return {
        f"block-{index}": (100.0 * (ratio + deviation), 100.0)
        for index, deviation in enumerate(deviations)
    }


def test_paired_bca_is_block_paired_and_holm_family_is_exact() -> None:
    assert SECONDARY_CONTRASTS == (
        "l0_naive_vs_tts",
        "lightcone_vs_l0_naive",
    )
    lightcone_tts = paired_bca_contrast(
        "lightcone_vs_tts",
        paired_values(1.04),
        repetitions=2_000,
        seed=7,
    )
    lightcone_static = paired_bca_contrast(
        "lightcone_vs_static",
        paired_values(1.03),
        repetitions=2_000,
        seed=8,
    )
    assert lightcone_tts.independent_unit == "paired_block"
    assert lightcone_tts.mean_relative_gain == pytest.approx(0.040663, abs=1e-6)
    assert lightcone_tts.ci_lower_relative_gain > 0.0
    decisions = holm_primary_contrasts(
        {
            "lightcone_vs_tts": lightcone_tts,
            "lightcone_vs_static": lightcone_static,
        }
    )
    assert tuple(decision.name for decision in decisions) == PRIMARY_CONTRASTS
    assert all(decision.procedure == "holm" for decision in decisions)
    assert all(decision.rejected for decision in decisions)
    assert all(
        decision.adjusted_p_value >= decision.raw_p_value for decision in decisions
    )
    with pytest.raises(ValueError, match="LightCone--TTS and LightCone--Static"):
        holm_primary_contrasts({"lightcone_vs_tts": lightcone_tts})


def test_paired_bca_rejects_missing_or_nonpositive_pairs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        paired_bca_contrast("x", {"one": (1.0, 1.0)})
    with pytest.raises(ValueError, match="finite and positive"):
        paired_bca_contrast("x", {"one": (1.0, 1.0), "two": (0.0, 1.0)})
    with pytest.raises(ValueError, match="fixed at 95%"):
        paired_bca_contrast(
            "x",
            {"one": (1.0, 1.0), "two": (1.1, 1.0)},
            confidence=0.90,
        )


def test_holm_monotonicity_and_bh_fdr_adjustment() -> None:
    template = PairedBcaContrast(
        name="lightcone_vs_tts",
        block_ids=("a", "b"),
        mean_log_ratio=0.1,
        mean_relative_gain=0.1,
        ci_lower_relative_gain=0.01,
        ci_upper_relative_gain=0.2,
        raw_p_value=0.01,
        confidence=0.95,
    )
    primary = holm_primary_contrasts(
        {
            "lightcone_vs_tts": template,
            "lightcone_vs_static": replace(
                template,
                name="lightcone_vs_static",
                raw_p_value=0.04,
            ),
        }
    )
    assert [decision.adjusted_p_value for decision in primary] == pytest.approx(
        [0.02, 0.04]
    )
    breadth = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.20})
    by_name = {decision.name: decision for decision in breadth}
    assert by_name["a"].adjusted_p_value == pytest.approx(0.004)
    assert by_name["b"].adjusted_p_value == pytest.approx(0.04)
    assert by_name["c"].adjusted_p_value == pytest.approx(0.0533333333)
    assert by_name["d"].adjusted_p_value == pytest.approx(0.20)
    assert {name for name, decision in by_name.items() if decision.rejected} == {
        "a",
        "b",
    }
    with pytest.raises(ValueError, match="finite probabilities"):
        benjamini_hochberg({"missing": float("nan")})


def vector_summary(rows: np.ndarray) -> np.ndarray:
    return np.asarray([np.mean(rows), np.quantile(rows, 0.75)])


def test_hierarchical_and_time_block_bootstraps_preserve_registered_units() -> None:
    rows = {
        "block-a": np.asarray([1.0, 2.0, 3.0]),
        "block-b": np.asarray([4.0, 5.0]),
        "block-c": np.asarray([7.0, 8.0, 9.0, 10.0]),
    }
    hierarchical = hierarchical_block_request_bootstrap(
        rows,
        vector_summary,
        repetitions=500,
        seed=19,
    )
    repeated = hierarchical_block_request_bootstrap(
        rows,
        vector_summary,
        repetitions=500,
        seed=19,
    )
    assert isinstance(hierarchical, BootstrapInterval)
    assert hierarchical == repeated
    assert hierarchical.independent_units == ("block", "request")
    assert hierarchical.estimate == pytest.approx((49 / 9, 8.0))
    assert all(
        lower <= estimate <= upper
        for lower, estimate, upper in zip(
            hierarchical.ci_lower,
            hierarchical.estimate,
            hierarchical.ci_upper,
            strict=True,
        )
    )

    tail = time_block_bootstrap(
        rows,
        lambda values: float(np.quantile(values, 0.99)),
        repetitions=500,
        seed=20,
    )
    assert tail.independent_units == ("time_block",)
    assert tail.estimate == pytest.approx((9.92,))


def test_bootstraps_reject_missing_rows_and_unstable_statistics() -> None:
    with pytest.raises(ValueError, match="observed rows"):
        hierarchical_block_request_bootstrap(
            {"a": np.asarray([]), "b": np.asarray([1.0])},
            np.mean,
            repetitions=100,
        )
    with pytest.raises(ValueError, match="finite"):
        time_block_bootstrap(
            {"a": np.asarray([float("nan")]), "b": np.asarray([1.0])},
            np.mean,
            repetitions=100,
        )

    calls = 0

    def changing_shape(values: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(1 if calls == 1 else 2)

    with pytest.raises(ValueError, match="fixed finite vector"):
        time_block_bootstrap(
            {"a": np.asarray([1.0]), "b": np.asarray([2.0])},
            changing_shape,
            repetitions=100,
        )


def test_p99_claim_guard_requires_ten_thousand_completed_at_anchor() -> None:
    unresolved = guard_p99_claim(
        "anchor-c64",
        completed_requests=P99_MINIMUM_COMPLETIONS - 1,
        observed_p99_ms=None,
        minimum_completions=P99_MINIMUM_COMPLETIONS,
        preregistered_anchor_locked=True,
    )
    assert unresolved.status == "UNRESOLVED"
    assert not unresolved.claimable
    assert unresolved.observed_p99_ms is None
    claimable = guard_p99_claim(
        "anchor-c64",
        completed_requests=P99_MINIMUM_COMPLETIONS,
        observed_p99_ms=87.0,
        minimum_completions=P99_MINIMUM_COMPLETIONS,
        preregistered_anchor_locked=True,
    )
    assert claimable.claimable
    with pytest.raises(ValueError, match="observed p99"):
        guard_p99_claim(
            "anchor-c64",
            completed_requests=P99_MINIMUM_COMPLETIONS,
            observed_p99_ms=None,
            minimum_completions=P99_MINIMUM_COMPLETIONS,
            preregistered_anchor_locked=True,
        )

    custom_minimum = P99_MINIMUM_COMPLETIONS + 2_000
    custom = guard_p99_claim(
        "anchor-custom",
        completed_requests=P99_MINIMUM_COMPLETIONS,
        observed_p99_ms=None,
        minimum_completions=custom_minimum,
        preregistered_anchor_locked=True,
    )
    assert custom.status == "UNRESOLVED"
    assert custom.minimum_completions == custom_minimum
    with pytest.raises(ValueError, match="cannot expose"):
        guard_p99_claim(
            "anchor-unregistered",
            completed_requests=custom_minimum,
            observed_p99_ms=87.0,
            minimum_completions=custom_minimum,
            preregistered_anchor_locked=False,
        )
    with pytest.raises(ValueError, match="positive completion minimum"):
        guard_p99_claim(
            "anchor-zero",
            completed_requests=0,
            observed_p99_ms=None,
            minimum_completions=0,
            preregistered_anchor_locked=True,
        )


def slo_rows(count: int = 1_000) -> list[SloRequest]:
    buckets = ("short", "medium", "long")
    limits = {"short": 2_000.0, "medium": 5_000.0, "long": 10_000.0}
    return [
        SloRequest(
            request_id=f"request-{index}",
            prompt_bucket=bucket,
            eligible=True,
            completed=True,
            error=False,
            ttft_ms=limits[bucket],
            within_request_p99_itl_ms=100.0,
        )
        for index in range(count)
        for bucket in (buckets[index % len(buckets)],)
    ]


def test_slo_accounting_uses_fixed_eligible_denominator_and_exact_boundaries() -> None:
    rows = slo_rows()
    rows.append(
        SloRequest(
            request_id="ineligible",
            prompt_bucket="short",
            eligible=False,
            completed=False,
            error=True,
            ttft_ms=None,
            within_request_p99_itl_ms=None,
        )
    )
    accounting = account_slo(rows)
    assert accounting.passed
    assert accounting.eligible_requests == 1_000
    assert accounting.qualification_rate == 1.0
    assert accounting.error_rate == 0.0
    assert accounting.completion_rate == 1.0

    ten_slow = slo_rows()
    for index in range(10):
        ten_slow[index] = replace(ten_slow[index], ttft_ms=None)
    assert account_slo(ten_slow).passed
    ten_slow[10] = replace(ten_slow[10], ttft_ms=None)
    failed = account_slo(ten_slow)
    assert not failed.passed
    assert failed.qualification_rate == pytest.approx(0.989)


def test_slo_accounting_enforces_error_and_completion_rates_without_imputation() -> (
    None
):
    rows = slo_rows()
    rows[0] = replace(rows[0], error=True)
    assert account_slo(rows).passed
    rows[1] = replace(rows[1], error=True)
    assert account_slo(rows).error_rate == pytest.approx(0.002)
    assert not account_slo(rows).passed

    incomplete = slo_rows()
    incomplete[0] = replace(
        incomplete[0],
        completed=False,
        ttft_ms=None,
        within_request_p99_itl_ms=None,
    )
    assert account_slo(incomplete).passed
    incomplete[1] = replace(
        incomplete[1],
        completed=False,
        ttft_ms=None,
        within_request_p99_itl_ms=None,
    )
    assert account_slo(incomplete).completion_rate == pytest.approx(0.998)
    assert not account_slo(incomplete).passed

    with pytest.raises(ValueError, match="unique"):
        account_slo([rows[0], rows[0]])
    with pytest.raises(ValueError, match="eligible"):
        account_slo([replace(rows[0], eligible=False)])


def hardware_envelope() -> HardwareEnvelope:
    return HardwareEnvelope(
        gpu_clock_mhz_min=1_500.0,
        gpu_clock_mhz_max=2_100.0,
        memory_clock_mhz_min=1_000.0,
        memory_clock_mhz_max=1_500.0,
        temperature_c_max=80.0,
        power_watts_min=200.0,
        power_watts_max=600.0,
        power_state="P0",
        allowed_throttling_reasons=("idle",),
        allowed_background_processes=("nvidia-persistenced",),
    )


def hardware_observation(block_id: str = "block-0") -> HardwareBlockObservation:
    return HardwareBlockObservation(
        block_id=block_id,
        gpu_clock_mhz=1_800.0,
        memory_clock_mhz=1_200.0,
        temperature_c=70.0,
        power_watts=450.0,
        power_state="P0",
        throttling_reasons=(),
        background_processes=("nvidia-persistenced",),
    )


def test_hardware_envelope_invalidates_missing_and_out_of_range_blocks() -> None:
    valid = validate_hardware_block(hardware_envelope(), hardware_observation())
    assert valid.valid
    assert valid.reasons == ()

    invalid = validate_hardware_block(
        hardware_envelope(),
        replace(
            hardware_observation(),
            gpu_clock_mhz=1_400.0,
            temperature_c=None,
            power_state="P2",
            throttling_reasons=("thermal",),
            background_processes=("unknown-worker",),
        ),
    )
    assert invalid.status == "INVALIDATED"
    assert set(invalid.reasons) == {
        "gpu_clock_mhz:below_min",
        "temperature_c:missing",
        "power_state:mismatch",
        "throttling_reason:unexpected:thermal",
        "background_process:unexpected:unknown-worker",
    }


def test_hardware_envelope_rejects_duplicate_or_invalid_registration() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_hardware_blocks(
            hardware_envelope(),
            (hardware_observation(), hardware_observation()),
        )
    with pytest.raises(ValueError, match="clock envelope"):
        validate_hardware_block(
            replace(hardware_envelope(), gpu_clock_mhz_min=float("nan")),
            hardware_observation(),
        )
