from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import pytest

from lightcone_spec.experiments import long_context_analysis as analysis
from lightcone_spec.experiments.long_context_analysis import (
    E3B_CONTEXT_GRID,
    E3B_INTERIOR_KNOTS,
    E3B_LONG_CONTEXT_PROTOCOL_SHA256,
    E3bCrossoverOutcome,
    E3bLongContextAnalysisPlan,
    E3bMethod,
    E3bMetric,
    E3bObservationDisposition,
    E3bPairedRequestObservation,
    E3bReductionStatus,
    reduce_e3b_long_context_pair,
)
from lightcone_spec.experiments.registry import FINAL_BLOCKS, content_sha256

MetricFunction = Callable[[int, int, int], float]


def _plan(
    *,
    metric: E3bMetric = E3bMetric.ACCEPTED_LENGTH,
    seed: int = 20260813,
    repetitions: int = 100,
    final_blocks: int = 12,
) -> E3bLongContextAnalysisPlan:
    return E3bLongContextAnalysisPlan(
        schema_version=1,
        protocol_sha256=E3B_LONG_CONTEXT_PROTOCOL_SHA256,
        family_sha256=content_sha256(
            {
                "experiment": "E3b",
                "regime": "long_input_short_output",
                "load": "common_load",
                "width_panel": "matched",
            }
        ),
        metric=metric,
        candidate_method=E3bMethod.L0,
        baseline_method=E3bMethod.STATIC,
        final_block_ids=FINAL_BLOCKS[:final_blocks],
        bootstrap_repetitions=repetitions,
        bootstrap_seed=seed,
    )


def _rows(
    plan: E3bLongContextAnalysisPlan,
    candidate: MetricFunction,
    baseline: MetricFunction,
    *,
    requests: int = 1,
) -> tuple[E3bPairedRequestObservation, ...]:
    if plan.metric is E3bMetric.COMMITTED_TOKEN_GOODPUT:
        raise ValueError("goodput tests must retain raw timestamps")
    return tuple(
        E3bPairedRequestObservation(
            block_id=block_id,
            context_tokens=context,
            request_id=f"request-{request_index:03d}",
            disposition=E3bObservationDisposition.OBSERVED,
            candidate_numerator=candidate(block_id, context, request_index),
            candidate_denominator=1.0,
            baseline_numerator=baseline(block_id, context, request_index),
            baseline_denominator=1.0,
            source_sha256=content_sha256(
                {
                    "block": block_id,
                    "context": context,
                    "request": request_index,
                }
            ),
        )
        for block_id in plan.final_block_ids
        for context in E3B_CONTEXT_GRID
        for request_index in range(requests)
    )


def _goodput_rows(
    plan: E3bLongContextAnalysisPlan,
    candidate: MetricFunction,
    baseline: MetricFunction,
    *,
    requests: int = 1,
) -> tuple[E3bPairedRequestObservation, ...]:
    assert plan.metric is E3bMetric.COMMITTED_TOKEN_GOODPUT
    token_scale = 100_000_000
    return tuple(
        E3bPairedRequestObservation(
            block_id=block_id,
            context_tokens=context,
            request_id=f"request-{request_index:03d}",
            disposition=E3bObservationDisposition.OBSERVED,
            candidate_numerator=None,
            candidate_denominator=None,
            baseline_numerator=None,
            baseline_denominator=None,
            source_sha256=content_sha256(
                {
                    "block": block_id,
                    "context": context,
                    "request": request_index,
                    "metric": "goodput",
                }
            ),
            candidate_completed_tokens=round(
                token_scale * candidate(block_id, context, request_index)
            ),
            candidate_arrival_ns=0,
            candidate_completed_ns=1_000_000_000,
            baseline_completed_tokens=round(
                token_scale * baseline(block_id, context, request_index)
            ),
            baseline_arrival_ns=0,
            baseline_completed_ns=1_000_000_000,
        )
        for block_id in plan.final_block_ids
        for context in E3B_CONTEXT_GRID
        for request_index in range(requests)
    )


def _power(exponent: float) -> MetricFunction:
    return lambda _block, context, _request: (context / 1024.0) ** (-exponent)


def _assert_null_numeric_reduction(result: object) -> None:
    assert isinstance(result, analysis.E3bLongContextReduction)
    assert result.status is E3bReductionStatus.UNRESOLVED
    assert result.curve_points is None
    assert result.bootstrap_repetitions_completed is None
    assert result.crossover.first_bracket_tokens is None
    assert result.crossover.root_tokens is None
    assert result.crossover.root_interval_tokens is None


def test_protocol_binds_exact_grid_knots_signs_and_final_prefix() -> None:
    plan = _plan()

    assert E3B_CONTEXT_GRID == (
        1024,
        2048,
        4096,
        8192,
        16384,
        24576,
        32768,
        40928,
    )
    assert E3B_INTERIOR_KNOTS == (4096, 16384, 32768)
    assert plan.final_block_ids == FINAL_BLOCKS[:12]
    assert plan.sha256 == content_sha256(plan)

    with pytest.raises(ValueError, match="registered context grid"):
        replace(plan, context_grid_tokens=plan.context_grid_tokens[:-1])
    with pytest.raises(ValueError, match="fixed at 4K, 16K, and 32K"):
        replace(plan, interior_knots_tokens=(4096, 8192, 32768))
    with pytest.raises(ValueError, match="registered protocol"):
        replace(plan, protocol_sha256="0" * 64)
    with pytest.raises(ValueError, match="exact registered prefix"):
        replace(plan, final_block_ids=tuple(reversed(plan.final_block_ids)))
    with pytest.raises(ValueError, match="12--20"):
        replace(plan, final_block_ids=FINAL_BLOCKS[:11])
    with pytest.raises(ValueError, match="at least 100"):
        replace(plan, bootstrap_repetitions=99)


def test_power_law_recovers_registered_elasticity_and_curvature_signs() -> None:
    plan = _plan()
    rows = _rows(plan, _power(0.5), _power(0.8))

    result = reduce_e3b_long_context_pair(plan, rows)

    assert result.status is E3bReductionStatus.OBSERVED
    assert result.reason_code == "e3b_registered_long_context_reduced"
    assert result.bootstrap_repetitions_completed == plan.bootstrap_repetitions
    assert result.observations_sha256 is not None
    assert result.crossover.outcome is E3bCrossoverOutcome.NOT_APPLICABLE
    assert result.crossover.root_tokens is None
    assert result.curve_points is not None
    assert tuple(point.context_tokens for point in result.curve_points) == (
        E3B_CONTEXT_GRID
    )
    for point in result.curve_points:
        assert point.candidate_elasticity.estimate == pytest.approx(0.5, abs=1e-12)
        assert point.baseline_elasticity.estimate == pytest.approx(0.8, abs=1e-12)
        assert point.paired_elasticity_difference.estimate == pytest.approx(
            -0.3, abs=1e-12
        )
        assert point.candidate_curvature.estimate == pytest.approx(0.0, abs=1e-12)
        assert point.baseline_curvature.estimate == pytest.approx(0.0, abs=1e-12)


def test_natural_boundary_is_zero_and_interior_curvature_is_analytic() -> None:
    plan = _plan()

    def curved(_block: int, context: int, _request: int) -> float:
        log_context = math.log(context / 1024.0)
        return math.exp(-0.1 * log_context - 0.04 * log_context**2)

    result = reduce_e3b_long_context_pair(plan, _rows(plan, curved, _power(0.1)))

    assert result.status is E3bReductionStatus.OBSERVED
    assert result.curve_points is not None
    assert result.curve_points[0].candidate_curvature.estimate == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.curve_points[-1].candidate_curvature.estimate == pytest.approx(
        0.0, abs=1e-12
    )
    assert all(
        point.candidate_curvature.estimate > 0.0 for point in result.curve_points[1:-1]
    )


def test_request_contributions_are_ratio_of_sums_not_mean_of_ratios() -> None:
    plan = _plan()
    rows = list(_rows(plan, _power(0.2), _power(0.2), requests=2))
    for index, row in enumerate(rows):
        if row.request_id == "request-000":
            rows[index] = replace(
                row,
                candidate_numerator=9.0,
                candidate_denominator=1.0,
                baseline_numerator=2.0,
                baseline_denominator=2.0,
            )
        else:
            rows[index] = replace(
                row,
                candidate_numerator=1.0,
                candidate_denominator=9.0,
                baseline_numerator=2.0,
                baseline_denominator=2.0,
            )

    result = reduce_e3b_long_context_pair(plan, rows)

    assert result.status is E3bReductionStatus.OBSERVED
    assert result.curve_points is not None
    for point in result.curve_points:
        assert point.candidate_fitted_metric.estimate == pytest.approx(1.0)
        assert point.baseline_fitted_metric.estimate == pytest.approx(1.0)


def test_goodput_request_resample_uses_multiplicity_and_raw_time_extrema() -> None:
    plan = _plan(metric=E3bMetric.COMMITTED_TOKEN_GOODPUT)
    template = _goodput_rows(
        plan,
        lambda _block, _context, _request: 1.0,
        lambda _block, _context, _request: 1.0,
    )[0]
    first = replace(
        template,
        request_id="first",
        candidate_completed_tokens=10,
        candidate_arrival_ns=0,
        candidate_completed_ns=10,
        source_sha256=content_sha256({"request": "first"}),
    )
    second = replace(
        template,
        request_id="second",
        candidate_completed_tokens=30,
        candidate_arrival_ns=20,
        candidate_completed_ns=40,
        source_sha256=content_sha256({"request": "second"}),
    )

    one_copy = analysis._block_metric(plan, (first,), candidate=True)
    duplicated = analysis._block_metric(plan, (first, first), candidate=True)
    mixed = analysis._block_metric(plan, (first, first, second), candidate=True)

    assert one_copy == pytest.approx(10 / (10 / 1_000_000_000))
    # A duplicate contributes its tokens twice but does not create a second
    # timestamp timeline, so the original min/max remain 0 and 10.
    assert duplicated == pytest.approx(20 / (10 / 1_000_000_000))
    assert mixed == pytest.approx(50 / (40 / 1_000_000_000))


def test_goodput_reports_first_measured_crossover_bracket_and_root_interval() -> None:
    plan = _plan(metric=E3bMetric.COMMITTED_TOKEN_GOODPUT)

    def candidate(_block: int, context: int, _request: int) -> float:
        return 10.0 * (context / 12000.0) ** (-0.3)

    result = reduce_e3b_long_context_pair(
        plan,
        _goodput_rows(plan, candidate, lambda _block, _context, _request: 10.0),
    )

    assert result.status is E3bReductionStatus.OBSERVED
    assert result.crossover.outcome is E3bCrossoverOutcome.CROSSOVER
    assert result.crossover.first_bracket_tokens == (8192, 16384)
    assert result.crossover.root_tokens == pytest.approx(12000.0, rel=1e-10)
    assert result.crossover.root_interval_tokens is not None
    assert result.crossover.root_interval_tokens[0] == pytest.approx(12000.0, rel=1e-10)
    assert result.crossover.root_interval_tokens[1] == pytest.approx(12000.0, rel=1e-10)


def test_goodput_rejects_summary_ratio_and_invalid_raw_timing() -> None:
    plan = _plan(metric=E3bMetric.COMMITTED_TOKEN_GOODPUT)
    rows = _goodput_rows(plan, _power(0.2), _power(0.3))

    summary_ratio = reduce_e3b_long_context_pair(
        plan,
        (replace(rows[0], candidate_numerator=1.0), *rows[1:]),
    )
    invalid_timing = reduce_e3b_long_context_pair(
        plan,
        (replace(rows[0], candidate_completed_ns=-1), *rows[1:]),
    )

    _assert_null_numeric_reduction(summary_ratio)
    _assert_null_numeric_reduction(invalid_timing)
    assert summary_ratio.reason_code == "e3b_goodput_observation_payload_conflict"
    assert invalid_timing.reason_code == "e3b_goodput_raw_timing_invalid"


def test_no_crossover_through_40928_is_distinct_from_hbm_infeasible() -> None:
    plan = _plan(metric=E3bMetric.COMMITTED_TOKEN_GOODPUT)
    rows = _goodput_rows(
        plan,
        _power(0.1),
        lambda _block, context, _request: 0.5 * (context / 1024) ** -0.1,
    )

    no_crossover = reduce_e3b_long_context_pair(plan, rows)

    assert no_crossover.status is E3bReductionStatus.OBSERVED
    assert (
        no_crossover.crossover.outcome is E3bCrossoverOutcome.NO_CROSSOVER_THROUGH_LIMIT
    )
    assert no_crossover.crossover.first_bracket_tokens is None

    marker_index = next(
        index
        for index, row in enumerate(rows)
        if row.block_id == plan.final_block_ids[0] and row.context_tokens == 40928
    )
    marker = replace(
        rows[marker_index],
        disposition=E3bObservationDisposition.CANDIDATE_HBM_INFEASIBLE,
        candidate_numerator=None,
        candidate_denominator=None,
        baseline_numerator=None,
        baseline_denominator=None,
        candidate_completed_tokens=None,
        candidate_arrival_ns=None,
        candidate_completed_ns=None,
        baseline_completed_tokens=None,
        baseline_arrival_ns=None,
        baseline_completed_ns=None,
    )
    hbm_rows = (*rows[:marker_index], marker, *rows[marker_index + 1 :])

    infeasible = reduce_e3b_long_context_pair(plan, hbm_rows)

    _assert_null_numeric_reduction(infeasible)
    assert infeasible.crossover.outcome is E3bCrossoverOutcome.HBM_INFEASIBLE
    assert infeasible.crossover.infeasible_contexts == (40928,)
    assert infeasible.crossover.infeasible_methods == (E3bMethod.L0,)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    (
        (
            lambda rows, _plan: rows[1:],
            "e3b_context_grid_coverage_inexact",
        ),
        (
            lambda rows, _plan: (*rows, rows[0]),
            "e3b_paired_request_identity_duplicated",
        ),
        (
            lambda rows, _plan: (
                replace(rows[0], candidate_numerator=float("nan")),
                *rows[1:],
            ),
            "e3b_observation_nonfinite",
        ),
        (
            lambda rows, _plan: tuple(
                replace(row, candidate_numerator=0.0) for row in rows
            ),
            "e3b_metric_nonpositive",
        ),
        (
            lambda rows, plan: (
                *rows,
                replace(
                    rows[0],
                    block_id=plan.final_block_ids[-1] + 1,
                    source_sha256=content_sha256({"extra": "block"}),
                ),
            ),
            "e3b_final_block_coverage_inexact",
        ),
    ),
)
def test_missing_duplicate_nonfinite_and_nonpositive_inputs_are_null_unresolved(
    mutate: Callable[
        [
            tuple[E3bPairedRequestObservation, ...],
            E3bLongContextAnalysisPlan,
        ],
        tuple[E3bPairedRequestObservation, ...],
    ],
    reason: str,
) -> None:
    plan = _plan()
    rows = _rows(plan, _power(0.2), _power(0.3))

    result = reduce_e3b_long_context_pair(plan, mutate(rows, plan))

    _assert_null_numeric_reduction(result)
    assert result.reason_code == reason
    if reason == "e3b_observation_nonfinite":
        assert result.observations_sha256 is None


def test_rank_deficiency_is_named_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    rows = _rows(plan, _power(0.2), _power(0.3))
    real_design = analysis._natural_design_matrix

    def rank_deficient(context_tokens: object, *, derivative_order: int):
        design = real_design(context_tokens, derivative_order=derivative_order)
        if derivative_order == 0:
            design[:, 1:] = design[:, :1]
        return design

    monkeypatch.setattr(analysis, "_natural_design_matrix", rank_deficient)

    result = reduce_e3b_long_context_pair(plan, rows)

    _assert_null_numeric_reduction(result)
    assert result.reason_code == "e3b_spline_rank_insufficient"


def test_hierarchical_bootstrap_refits_every_replicate_and_is_seed_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(seed=73)

    def candidate(block: int, context: int, request: int) -> float:
        return (context / 1024.0) ** (-0.4) * (
            1.0 + 0.004 * (block - FINAL_BLOCKS[0]) + 0.01 * request
        )

    def baseline(block: int, context: int, request: int) -> float:
        return (context / 1024.0) ** (-0.6) * (
            1.0 + 0.002 * (block - FINAL_BLOCKS[0]) + 0.005 * request
        )

    rows = _rows(plan, candidate, baseline, requests=3)
    real_fit = analysis._fit_registered_natural_log_spline
    fit_calls = 0

    def counting_fit(values: object):
        nonlocal fit_calls
        fit_calls += 1
        return real_fit(values)

    monkeypatch.setattr(
        analysis,
        "_fit_registered_natural_log_spline",
        counting_fit,
    )

    result = reduce_e3b_long_context_pair(plan, rows)

    assert result.status is E3bReductionStatus.OBSERVED
    assert fit_calls == 2 * (plan.bootstrap_repetitions + 1)
    assert result.plan_sha256 == plan.sha256
    assert result.curve_points is not None
    assert (
        result.curve_points[3].candidate_fitted_metric.upper
        > result.curve_points[3].candidate_fitted_metric.lower
    )
    repeated = reduce_e3b_long_context_pair(plan, rows)
    assert repeated == result
    assert repeated.sha256 == result.sha256
    assert result.sha256 == content_sha256(result.to_dict())
    assert fit_calls == 4 * (plan.bootstrap_repetitions + 1)
    assert replace(plan, bootstrap_seed=74).sha256 != plan.sha256
