"""Fail-closed E3b long-context spline and crossover reduction.

The reducer in this module is deliberately independent of execution and raw
artifact loading.  A future formal caller must first reopen the schema-v4
completion and native-terminal authority, then materialize paired request
contributions for exactly one registered confirmation family.  This module
only performs the preregistered numerical reduction.

Missing coverage, non-finite observations, non-positive fitted metrics, and
rank- or conditioning-deficient spline systems become typed ``UNRESOLVED``
results.  They are never represented by a numeric zero.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from functools import cached_property
from itertools import pairwise

import numpy as np

from lightcone_spec.experiments.registry import (
    CONTEXT_GRID,
    FINAL_BLOCKS,
    LONG_CONTEXT_ANCHORS,
    content_sha256,
)
from lightcone_spec.experiments.statistics import (
    MAXIMUM_FINAL_BLOCKS,
    MINIMUM_FINAL_BLOCKS,
    REGISTERED_CONFIDENCE,
)

E3B_CONTEXT_GRID = CONTEXT_GRID
E3B_INTERIOR_KNOTS = LONG_CONTEXT_ANCHORS[:-1]
E3B_MAXIMUM_CONTEXT_TOKENS = CONTEXT_GRID[-1]

_MINIMUM_BOOTSTRAP_REPETITIONS = 100
_MAXIMUM_DESIGN_CONDITION_NUMBER = 1.0e10
_ROOT_LOG_DIFFERENCE_TOLERANCE = 1.0e-12
_ROOT_LOG_CONTEXT_TOLERANCE = 1.0e-12
_ROOT_MAXIMUM_ITERATIONS = 128

# This digest is the reducer-owned rendering of EXPERIMENT_PROTOCOL.md section
# 11 and section 16 plus Part I, Phase II sections 11--12.  Keeping every
# scientific constant in the payload prevents a caller from silently changing
# a sign, knot, context point, resampling unit, or crossover interpretation.
E3B_LONG_CONTEXT_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e3b_registered_long_context_spline_protocol",
        "context_grid_tokens": E3B_CONTEXT_GRID,
        "interior_knots_tokens": E3B_INTERIOR_KNOTS,
        "maximum_context_tokens": E3B_MAXIMUM_CONTEXT_TOKENS,
        "response": "log_positive_metric",
        "predictor": "log_context_tokens",
        "spline": "ordinary_least_squares_natural_cubic_regression_spline",
        "boundary_knots_tokens": (
            E3B_CONTEXT_GRID[0],
            E3B_CONTEXT_GRID[-1],
        ),
        "smoothing_selection": "forbidden_on_confirmation",
        "elasticity": "-d_log_metric/d_log_context",
        "curvature": "-d2_log_metric/d_log_context2",
        "point_aggregation": (
            "paired_ratio_of_sums_within_block_then_equal_mean_across_blocks"
        ),
        "final_block_count": (MINIMUM_FINAL_BLOCKS, MAXIMUM_FINAL_BLOCKS),
        "final_block_identity": "exact_registered_final_prefix",
        "bootstrap": "paired_hierarchical_block_then_request_refit_same_basis",
        "bootstrap_rng": "numpy_pcg64_unsigned_seed_bound_to_plan",
        "minimum_bootstrap_repetitions": _MINIMUM_BOOTSTRAP_REPETITIONS,
        "confidence": REGISTERED_CONFIDENCE,
        "interval": "hierarchical_bootstrap_percentile_linear_quantile",
        "crossover_metric": "committed_token_goodput",
        "crossover": (
            "first_measured_grid_bracket_where_paired_fitted_goodput_changes_sign"
        ),
        "crossover_root": "bisection_in_log_context_within_first_bracket",
        "crossover_bootstrap": "same_first_bracket_required_for_root_interval",
        "no_crossover": "no_change_through_40928_in_point_and_bootstrap_refits",
        "hbm_infeasible": "distinct_terminal_outcome",
        "missing_or_nonfinite": "unresolved_with_null_numeric_payload",
        "maximum_design_condition_number": _MAXIMUM_DESIGN_CONDITION_NUMBER,
        "root_log_difference_tolerance": _ROOT_LOG_DIFFERENCE_TOLERANCE,
        "root_log_context_tolerance": _ROOT_LOG_CONTEXT_TOLERANCE,
        "root_maximum_iterations": _ROOT_MAXIMUM_ITERATIONS,
    }
)


class E3bMetric(str, Enum):
    """Positive E3b outcomes for which log-context derivatives are defined."""

    ACCEPTED_LENGTH = "accepted_length"
    COMMITTED_TOKEN_GOODPUT = "committed_token_goodput"
    SLO_QUALIFIED_GOODPUT = "slo_qualified_goodput"
    TARGET_CALLS_PER_OUTPUT_TOKEN = "target_calls_per_output_token"
    POSITION_WISE_SURVIVAL = "position_wise_survival"


class E3bMethod(str, Enum):
    TARGET_ONLY = "target_only"
    STATIC = "static"
    TTS = "tts"
    L0 = "l0"


class E3bObservationDisposition(str, Enum):
    OBSERVED = "OBSERVED"
    CANDIDATE_HBM_INFEASIBLE = "CANDIDATE_HBM_INFEASIBLE"
    BASELINE_HBM_INFEASIBLE = "BASELINE_HBM_INFEASIBLE"
    BOTH_HBM_INFEASIBLE = "BOTH_HBM_INFEASIBLE"


class E3bReductionStatus(str, Enum):
    OBSERVED = "OBSERVED"
    UNRESOLVED = "UNRESOLVED"


class E3bCrossoverOutcome(str, Enum):
    CROSSOVER = "CROSSOVER"
    NO_CROSSOVER_THROUGH_LIMIT = "NO_CROSSOVER_THROUGH_LIMIT"
    HBM_INFEASIBLE = "HBM_INFEASIBLE"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "N/A"


@dataclass(frozen=True)
class E3bLongContextAnalysisPlan:
    """Immutable numerical plan sealed before confirmation is unblinded."""

    schema_version: int
    protocol_sha256: str
    family_sha256: str
    metric: E3bMetric
    candidate_method: E3bMethod
    baseline_method: E3bMethod
    final_block_ids: tuple[int, ...]
    bootstrap_repetitions: int
    bootstrap_seed: int
    context_grid_tokens: tuple[int, ...] = E3B_CONTEXT_GRID
    interior_knots_tokens: tuple[int, ...] = E3B_INTERIOR_KNOTS
    confidence: float = REGISTERED_CONFIDENCE

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("E3b analysis plan schema is unsupported")
        if self.protocol_sha256 != E3B_LONG_CONTEXT_PROTOCOL_SHA256:
            raise ValueError("E3b analysis plan changes the registered protocol")
        if not _is_sha256(self.family_sha256):
            raise ValueError("E3b family identity must be a lower-case SHA-256")
        if not isinstance(self.metric, E3bMetric):
            raise TypeError("E3b metric must be an E3bMetric")
        if not isinstance(self.candidate_method, E3bMethod) or not isinstance(
            self.baseline_method, E3bMethod
        ):
            raise TypeError("E3b methods must be E3bMethod values")
        if self.candidate_method not in {E3bMethod.TTS, E3bMethod.L0}:
            raise ValueError("E3b candidate must be TTS or L0")
        if self.baseline_method not in {
            E3bMethod.TARGET_ONLY,
            E3bMethod.STATIC,
            E3bMethod.TTS,
        }:
            raise ValueError("E3b baseline is not registered")
        if self.candidate_method == self.baseline_method:
            raise ValueError("E3b candidate and baseline must differ")
        if self.context_grid_tokens != E3B_CONTEXT_GRID:
            raise ValueError("E3b analysis requires the exact registered context grid")
        if self.interior_knots_tokens != E3B_INTERIOR_KNOTS:
            raise ValueError("E3b spline knots are fixed at 4K, 16K, and 32K")
        if self.confidence != REGISTERED_CONFIDENCE:
            raise ValueError("E3b intervals are fixed at 95% confidence")
        if not (
            MINIMUM_FINAL_BLOCKS <= len(self.final_block_ids) <= MAXIMUM_FINAL_BLOCKS
        ):
            raise ValueError("E3b requires exactly 12--20 final blocks")
        if any(type(block_id) is not int for block_id in self.final_block_ids):
            raise TypeError("E3b final block IDs must be integers")
        expected_prefix = FINAL_BLOCKS[: len(self.final_block_ids)]
        if self.final_block_ids != expected_prefix:
            raise ValueError("E3b final blocks must be the exact registered prefix")
        if (
            type(self.bootstrap_repetitions) is not int
            or self.bootstrap_repetitions < _MINIMUM_BOOTSTRAP_REPETITIONS
        ):
            raise ValueError("E3b bootstrap requires at least 100 refits")
        if type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed < 2**64:
            raise ValueError("E3b bootstrap seed must be an unsigned 64-bit integer")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E3bPairedRequestObservation:
    """One paired request contribution at a registered block/context cell.

    Metrics are reduced as a ratio of sums within each block, then blocks are
    equally weighted.  A mean-valued outcome uses denominator ``1`` for each
    contribution.  Keeping numerator and denominator separate avoids averaging
    caller-precomputed ratios.

    Numeric validation intentionally happens in the reducer so a corrupt or
    non-finite raw row produces ``UNRESOLVED`` rather than escaping as an
    exception or being coerced to zero.
    """

    block_id: int
    context_tokens: int
    request_id: str
    disposition: E3bObservationDisposition
    candidate_numerator: float | int | None
    candidate_denominator: float | int | None
    baseline_numerator: float | int | None
    baseline_denominator: float | int | None
    source_sha256: str


@dataclass(frozen=True)
class E3bIntervalEstimate:
    estimate: float
    lower: float
    upper: float
    confidence: float = REGISTERED_CONFIDENCE

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) for value in (self.estimate, self.lower, self.upper)
        ):
            raise ValueError("E3b interval values must be finite")
        if self.lower > self.upper:
            raise ValueError("E3b interval bounds are reversed")
        if self.confidence != REGISTERED_CONFIDENCE:
            raise ValueError("E3b intervals are fixed at 95% confidence")


@dataclass(frozen=True)
class E3bCurvePoint:
    context_tokens: int
    candidate_fitted_metric: E3bIntervalEstimate
    baseline_fitted_metric: E3bIntervalEstimate
    candidate_elasticity: E3bIntervalEstimate
    baseline_elasticity: E3bIntervalEstimate
    paired_elasticity_difference: E3bIntervalEstimate
    candidate_curvature: E3bIntervalEstimate
    baseline_curvature: E3bIntervalEstimate
    paired_curvature_difference: E3bIntervalEstimate


@dataclass(frozen=True)
class E3bCrossoverReduction:
    outcome: E3bCrossoverOutcome
    reason_code: str
    first_bracket_tokens: tuple[int, int] | None = None
    root_tokens: float | None = None
    root_interval_tokens: tuple[float, float] | None = None
    infeasible_contexts: tuple[int, ...] = ()
    infeasible_methods: tuple[E3bMethod, ...] = ()

    def __post_init__(self) -> None:
        if not _is_reason(self.reason_code):
            raise ValueError("E3b crossover reason must be a stable reason code")
        numeric_present = (
            self.first_bracket_tokens is not None
            or self.root_tokens is not None
            or self.root_interval_tokens is not None
        )
        if self.outcome is E3bCrossoverOutcome.CROSSOVER:
            if (
                self.first_bracket_tokens is None
                or self.root_tokens is None
                or self.root_interval_tokens is None
            ):
                raise ValueError(
                    "observed crossover requires bracket, root, and interval"
                )
            lower_context, upper_context = self.first_bracket_tokens
            lower_root, upper_root = self.root_interval_tokens
            if not (
                lower_context <= self.root_tokens <= upper_context
                and math.isfinite(self.root_tokens)
                and math.isfinite(lower_root)
                and math.isfinite(upper_root)
                and lower_context <= lower_root <= upper_root <= upper_context
            ):
                raise ValueError("E3b crossover root is outside its measured bracket")
            if self.infeasible_contexts or self.infeasible_methods:
                raise ValueError("observed crossover cannot also be HBM infeasible")
        elif numeric_present:
            raise ValueError("non-crossover outcomes must carry null numeric payloads")
        if self.outcome is E3bCrossoverOutcome.HBM_INFEASIBLE:
            if not self.infeasible_contexts or not self.infeasible_methods:
                raise ValueError("HBM infeasibility requires contexts and methods")
        elif self.infeasible_contexts or self.infeasible_methods:
            raise ValueError("only HBM infeasibility may name infeasible cells")


@dataclass(frozen=True)
class E3bLongContextReduction:
    schema_version: int
    status: E3bReductionStatus
    reason_code: str
    protocol_sha256: str
    plan_sha256: str
    observations_sha256: str | None
    curve_points: tuple[E3bCurvePoint, ...] | None
    crossover: E3bCrossoverReduction
    bootstrap_repetitions_completed: int | None
    resampling_unit: str = "paired_block_then_paired_request"
    interval_method: str = "hierarchical_bootstrap_percentile_95"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("E3b reduction schema is unsupported")
        if self.protocol_sha256 != E3B_LONG_CONTEXT_PROTOCOL_SHA256:
            raise ValueError("E3b reduction uses another protocol")
        if not _is_sha256(self.plan_sha256):
            raise ValueError("E3b reduction plan SHA is invalid")
        if self.observations_sha256 is not None and not _is_sha256(
            self.observations_sha256
        ):
            raise ValueError("E3b observation SHA is invalid")
        if not _is_reason(self.reason_code):
            raise ValueError("E3b reduction reason must be a stable reason code")
        if self.status is E3bReductionStatus.OBSERVED:
            if (
                self.curve_points is None
                or len(self.curve_points) != len(E3B_CONTEXT_GRID)
                or self.bootstrap_repetitions_completed is None
                or self.observations_sha256 is None
            ):
                raise ValueError("observed E3b reduction is incomplete")
            if any(type(point) is not E3bCurvePoint for point in self.curve_points):
                raise TypeError("observed E3b curve contains a foreign point type")
            if tuple(point.context_tokens for point in self.curve_points) != (
                E3B_CONTEXT_GRID
            ):
                raise ValueError("observed E3b curve changes the registered grid")
        elif self.status is E3bReductionStatus.UNRESOLVED:
            if (
                self.curve_points is not None
                or self.bootstrap_repetitions_completed is not None
            ):
                raise ValueError("unresolved E3b reduction must have null estimates")
        else:
            raise ValueError("invalid E3b reduction status")
        if self.resampling_unit != "paired_block_then_paired_request":
            raise ValueError("E3b resampling unit is fixed")
        if self.interval_method != "hierarchical_bootstrap_percentile_95":
            raise ValueError("E3b interval method is fixed")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class _SplineFit:
    coefficients: np.ndarray
    condition_number: float


@dataclass(frozen=True)
class _SplineEvaluation:
    metric: np.ndarray
    elasticity: np.ndarray
    curvature: np.ndarray


@dataclass(frozen=True)
class _PointCrossover:
    outcome: E3bCrossoverOutcome
    bracket: tuple[int, int] | None
    root_tokens: float | None
    reason_code: str


class _UnresolvedEvidenceError(ValueError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_reason(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value[0].isalnum()
        and all(
            character.islower() or character.isdigit() or character in "_.-"
            for character in value
        )
    )


def _observation_dict(value: E3bPairedRequestObservation) -> dict[str, object]:
    row = asdict(value)
    disposition = value.disposition
    row["disposition"] = (
        disposition.value
        if isinstance(disposition, E3bObservationDisposition)
        else disposition
    )
    return row


def _observation_sha256(
    observations: Sequence[E3bPairedRequestObservation],
) -> str | None:
    try:
        rows = sorted(
            (_observation_dict(value) for value in observations),
            key=lambda row: (
                str(row["block_id"]),
                str(row["context_tokens"]),
                str(row["request_id"]),
                str(row["source_sha256"]),
            ),
        )
        return content_sha256(rows)
    except (TypeError, ValueError):
        return None


def _unresolved(
    plan: E3bLongContextAnalysisPlan,
    observations: Sequence[E3bPairedRequestObservation],
    reason_code: str,
    *,
    crossover: E3bCrossoverReduction | None = None,
) -> E3bLongContextReduction:
    return E3bLongContextReduction(
        schema_version=1,
        status=E3bReductionStatus.UNRESOLVED,
        reason_code=reason_code,
        protocol_sha256=E3B_LONG_CONTEXT_PROTOCOL_SHA256,
        plan_sha256=plan.sha256,
        observations_sha256=_observation_sha256(observations),
        curve_points=None,
        crossover=crossover
        or E3bCrossoverReduction(
            outcome=E3bCrossoverOutcome.UNRESOLVED,
            reason_code=reason_code,
        ),
        bootstrap_repetitions_completed=None,
    )


def _validate_numeric(value: object, *, numerator: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _UnresolvedEvidenceError("e3b_observation_numeric_missing")
    checked = float(value)
    if not math.isfinite(checked):
        raise _UnresolvedEvidenceError("e3b_observation_nonfinite")
    if numerator:
        if checked < 0.0:
            raise _UnresolvedEvidenceError("e3b_observation_negative_numerator")
    elif checked <= 0.0:
        raise _UnresolvedEvidenceError("e3b_observation_nonpositive_denominator")
    return checked


def _validate_observations(
    plan: E3bLongContextAnalysisPlan,
    observations: tuple[E3bPairedRequestObservation, ...],
) -> tuple[
    dict[tuple[int, int], tuple[E3bPairedRequestObservation, ...]],
    tuple[int, ...],
    tuple[E3bMethod, ...],
]:
    if not observations:
        raise _UnresolvedEvidenceError("e3b_observations_missing")
    if any(type(value) is not E3bPairedRequestObservation for value in observations):
        raise _UnresolvedEvidenceError("e3b_observation_type_invalid")

    grouped: dict[tuple[int, int], list[E3bPairedRequestObservation]] = defaultdict(
        list
    )
    for value in observations:
        if type(value.block_id) is not int or type(value.context_tokens) is not int:
            raise _UnresolvedEvidenceError("e3b_observation_identity_invalid")
        if (
            type(value.request_id) is not str
            or not value.request_id
            or "\n" in value.request_id
        ):
            raise _UnresolvedEvidenceError("e3b_request_identity_invalid")
        if not _is_sha256(value.source_sha256):
            raise _UnresolvedEvidenceError("e3b_observation_source_invalid")
        if not isinstance(value.disposition, E3bObservationDisposition):
            raise _UnresolvedEvidenceError("e3b_observation_disposition_invalid")
        grouped[(value.block_id, value.context_tokens)].append(value)

    observed_blocks = {block_id for block_id, _ in grouped}
    if observed_blocks != set(plan.final_block_ids):
        raise _UnresolvedEvidenceError("e3b_final_block_coverage_inexact")
    expected_keys = {
        (block_id, context)
        for block_id in plan.final_block_ids
        for context in E3B_CONTEXT_GRID
    }
    if set(grouped) != expected_keys:
        raise _UnresolvedEvidenceError("e3b_context_grid_coverage_inexact")

    normalized: dict[tuple[int, int], tuple[E3bPairedRequestObservation, ...]] = {}
    infeasible_contexts: set[int] = set()
    infeasible_methods: set[E3bMethod] = set()
    numeric_fields = (
        "candidate_numerator",
        "candidate_denominator",
        "baseline_numerator",
        "baseline_denominator",
    )
    for key in sorted(grouped):
        rows = grouped[key]
        request_ids = [value.request_id for value in rows]
        if len(request_ids) != len(set(request_ids)):
            raise _UnresolvedEvidenceError("e3b_paired_request_identity_duplicated")
        dispositions = {value.disposition for value in rows}
        if dispositions == {E3bObservationDisposition.OBSERVED}:
            for value in rows:
                _validate_numeric(value.candidate_numerator, numerator=True)
                _validate_numeric(value.candidate_denominator, numerator=False)
                _validate_numeric(value.baseline_numerator, numerator=True)
                _validate_numeric(value.baseline_denominator, numerator=False)
        elif len(rows) == 1 and len(dispositions) == 1:
            marker = rows[0]
            if marker.disposition is E3bObservationDisposition.OBSERVED:
                raise _UnresolvedEvidenceError("e3b_observation_group_invalid")
            if any(getattr(marker, field) is not None for field in numeric_fields):
                raise _UnresolvedEvidenceError("e3b_hbm_marker_has_numeric_payload")
            infeasible_contexts.add(marker.context_tokens)
            if marker.disposition in {
                E3bObservationDisposition.CANDIDATE_HBM_INFEASIBLE,
                E3bObservationDisposition.BOTH_HBM_INFEASIBLE,
            }:
                infeasible_methods.add(plan.candidate_method)
            if marker.disposition in {
                E3bObservationDisposition.BASELINE_HBM_INFEASIBLE,
                E3bObservationDisposition.BOTH_HBM_INFEASIBLE,
            }:
                infeasible_methods.add(plan.baseline_method)
        else:
            raise _UnresolvedEvidenceError("e3b_observed_and_hbm_rows_conflict")
        normalized[key] = tuple(
            sorted(rows, key=lambda value: (value.request_id, value.source_sha256))
        )
    return (
        normalized,
        tuple(sorted(infeasible_contexts)),
        tuple(sorted(infeasible_methods, key=lambda method: method.value)),
    )


def _positive_part_power(values: np.ndarray, knot: float, power: int) -> np.ndarray:
    return np.maximum(values - knot, 0.0) ** power


def _natural_design_matrix(
    context_tokens: np.ndarray, *, derivative_order: int
) -> np.ndarray:
    if derivative_order not in {0, 1, 2}:
        raise ValueError("natural spline derivative order must be 0, 1, or 2")
    contexts = np.asarray(context_tokens, dtype=np.float64)
    if contexts.ndim != 1 or contexts.size == 0:
        raise _UnresolvedEvidenceError("e3b_spline_context_shape_invalid")
    if not np.isfinite(contexts).all() or np.any(contexts <= 0.0):
        raise _UnresolvedEvidenceError("e3b_spline_context_nonfinite")
    lower = math.log(E3B_CONTEXT_GRID[0])
    upper = math.log(E3B_CONTEXT_GRID[-1])
    span = upper - lower
    z = (np.log(contexts) - lower) / span
    total_knots = np.asarray(
        (E3B_CONTEXT_GRID[0], *E3B_INTERIOR_KNOTS, E3B_CONTEXT_GRID[-1]),
        dtype=np.float64,
    )
    knots = (np.log(total_knots) - lower) / span
    upper_knot = float(knots[-1])
    anchor_knot = float(knots[-2])

    if derivative_order == 0:
        columns: list[np.ndarray] = [np.ones_like(z), z]
        power = 3
        factor = 1.0
    elif derivative_order == 1:
        columns = [np.zeros_like(z), np.ones_like(z)]
        power = 2
        factor = 3.0
    else:
        columns = [np.zeros_like(z), np.zeros_like(z)]
        power = 1
        factor = 6.0

    def d_basis(knot: float) -> np.ndarray:
        numerator = factor * (
            _positive_part_power(z, knot, power)
            - _positive_part_power(z, upper_knot, power)
        )
        return numerator / (upper_knot - knot)

    anchor = d_basis(anchor_knot)
    for knot in knots[:-2]:
        columns.append(d_basis(float(knot)) - anchor)
    design = np.column_stack(columns)
    if not np.isfinite(design).all():
        raise _UnresolvedEvidenceError("e3b_spline_design_nonfinite")
    return design


def _fit_registered_natural_log_spline(values: np.ndarray) -> _SplineFit:
    checked = np.asarray(values, dtype=np.float64)
    if checked.shape != (len(E3B_CONTEXT_GRID),):
        raise _UnresolvedEvidenceError("e3b_metric_grid_incomplete")
    if not np.isfinite(checked).all():
        raise _UnresolvedEvidenceError("e3b_metric_nonfinite")
    if np.any(checked <= 0.0):
        raise _UnresolvedEvidenceError("e3b_metric_nonpositive")
    design = _natural_design_matrix(
        np.asarray(E3B_CONTEXT_GRID, dtype=np.float64), derivative_order=0
    )
    try:
        coefficients, _, rank, singular_values = np.linalg.lstsq(
            design,
            np.log(checked),
            rcond=None,
        )
    except np.linalg.LinAlgError as error:
        raise _UnresolvedEvidenceError("e3b_spline_linear_algebra_failed") from error
    if rank != design.shape[1] or singular_values.size != design.shape[1]:
        raise _UnresolvedEvidenceError("e3b_spline_rank_insufficient")
    smallest = float(singular_values[-1])
    if smallest <= np.finfo(np.float64).tiny:
        raise _UnresolvedEvidenceError("e3b_spline_rank_insufficient")
    condition_number = float(singular_values[0] / smallest)
    if (
        not math.isfinite(condition_number)
        or condition_number > _MAXIMUM_DESIGN_CONDITION_NUMBER
    ):
        raise _UnresolvedEvidenceError("e3b_spline_numerically_unstable")
    if not np.isfinite(coefficients).all():
        raise _UnresolvedEvidenceError("e3b_spline_coefficients_nonfinite")
    return _SplineFit(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        condition_number=condition_number,
    )


def _evaluate_spline(
    fit: _SplineFit, contexts: np.ndarray | None = None
) -> _SplineEvaluation:
    evaluated_contexts = np.asarray(
        E3B_CONTEXT_GRID if contexts is None else contexts,
        dtype=np.float64,
    )
    design = _natural_design_matrix(evaluated_contexts, derivative_order=0)
    first = _natural_design_matrix(evaluated_contexts, derivative_order=1)
    second = _natural_design_matrix(evaluated_contexts, derivative_order=2)
    span = math.log(E3B_CONTEXT_GRID[-1]) - math.log(E3B_CONTEXT_GRID[0])
    log_metric = design @ fit.coefficients
    with np.errstate(over="ignore", invalid="ignore"):
        metric = np.exp(log_metric)
    elasticity = -(first @ fit.coefficients) / span
    curvature = -(second @ fit.coefficients) / (span * span)
    if not all(
        np.isfinite(values).all() for values in (metric, elasticity, curvature)
    ) or np.any(metric <= 0.0):
        raise _UnresolvedEvidenceError("e3b_spline_evaluation_nonfinite")
    return _SplineEvaluation(
        metric=np.asarray(metric),
        elasticity=np.asarray(elasticity),
        curvature=np.asarray(curvature),
    )


def _ratio_of_sums(
    rows: Sequence[E3bPairedRequestObservation], *, candidate: bool
) -> float:
    numerator_name = "candidate_numerator" if candidate else "baseline_numerator"
    denominator_name = "candidate_denominator" if candidate else "baseline_denominator"
    numerator = math.fsum(float(getattr(value, numerator_name)) for value in rows)
    denominator = math.fsum(float(getattr(value, denominator_name)) for value in rows)
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise _UnresolvedEvidenceError("e3b_aggregate_nonfinite")
    if denominator <= 0.0:
        raise _UnresolvedEvidenceError("e3b_aggregate_nonpositive_denominator")
    result = numerator / denominator
    if not math.isfinite(result) or result <= 0.0:
        raise _UnresolvedEvidenceError("e3b_metric_nonpositive")
    return result


def _aggregate_context_curves(
    plan: E3bLongContextAnalysisPlan,
    grouped: dict[tuple[int, int], tuple[E3bPairedRequestObservation, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    candidate_values: list[float] = []
    baseline_values: list[float] = []
    for context in E3B_CONTEXT_GRID:
        candidate_blocks = [
            _ratio_of_sums(grouped[(block_id, context)], candidate=True)
            for block_id in plan.final_block_ids
        ]
        baseline_blocks = [
            _ratio_of_sums(grouped[(block_id, context)], candidate=False)
            for block_id in plan.final_block_ids
        ]
        candidate_values.append(math.fsum(candidate_blocks) / len(candidate_blocks))
        baseline_values.append(math.fsum(baseline_blocks) / len(baseline_blocks))
    return np.asarray(candidate_values), np.asarray(baseline_values)


def _bootstrap_context_curves(
    plan: E3bLongContextAnalysisPlan,
    grouped: dict[tuple[int, int], tuple[E3bPairedRequestObservation, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    randomizer = np.random.Generator(np.random.PCG64(plan.bootstrap_seed))
    block_ids = plan.final_block_ids
    candidate = np.empty(
        (plan.bootstrap_repetitions, len(E3B_CONTEXT_GRID)), dtype=np.float64
    )
    baseline = np.empty_like(candidate)
    for repetition in range(plan.bootstrap_repetitions):
        sampled_positions = randomizer.integers(0, len(block_ids), size=len(block_ids))
        sampled_blocks = tuple(
            block_ids[int(position)] for position in sampled_positions
        )
        for context_index, context in enumerate(E3B_CONTEXT_GRID):
            candidate_blocks: list[float] = []
            baseline_blocks: list[float] = []
            for block_id in sampled_blocks:
                rows = grouped[(block_id, context)]
                request_positions = randomizer.integers(0, len(rows), size=len(rows))
                sampled_rows = tuple(
                    rows[int(position)] for position in request_positions
                )
                candidate_blocks.append(_ratio_of_sums(sampled_rows, candidate=True))
                baseline_blocks.append(_ratio_of_sums(sampled_rows, candidate=False))
            candidate[repetition, context_index] = math.fsum(candidate_blocks) / len(
                candidate_blocks
            )
            baseline[repetition, context_index] = math.fsum(baseline_blocks) / len(
                baseline_blocks
            )
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise _UnresolvedEvidenceError("e3b_bootstrap_aggregate_nonfinite")
    return candidate, baseline


def _log_difference(
    fit_candidate: _SplineFit, fit_baseline: _SplineFit, token: float
) -> float:
    design = _natural_design_matrix(
        np.asarray([token], dtype=np.float64), derivative_order=0
    )
    value = float(
        (design @ (fit_candidate.coefficients - fit_baseline.coefficients))[0]
    )
    if not math.isfinite(value):
        raise _UnresolvedEvidenceError("e3b_crossover_difference_nonfinite")
    return value


def _bisect_root(
    fit_candidate: _SplineFit,
    fit_baseline: _SplineFit,
    bracket: tuple[int, int],
) -> float:
    left = math.log(bracket[0])
    right = math.log(bracket[1])

    def value(log_context: float) -> float:
        return _log_difference(fit_candidate, fit_baseline, math.exp(log_context))

    left_value = value(left)
    right_value = value(right)
    if abs(left_value) <= _ROOT_LOG_DIFFERENCE_TOLERANCE:
        return float(bracket[0])
    if abs(right_value) <= _ROOT_LOG_DIFFERENCE_TOLERANCE:
        return float(bracket[1])
    if left_value * right_value >= 0.0:
        raise _UnresolvedEvidenceError("e3b_crossover_root_not_bracketed")
    for _ in range(_ROOT_MAXIMUM_ITERATIONS):
        middle = (left + right) / 2.0
        middle_value = value(middle)
        if (
            abs(middle_value) <= _ROOT_LOG_DIFFERENCE_TOLERANCE
            or right - left <= _ROOT_LOG_CONTEXT_TOLERANCE
        ):
            root = math.exp(middle)
            if not math.isfinite(root):
                raise _UnresolvedEvidenceError("e3b_crossover_root_nonfinite")
            return root
        if left_value * middle_value < 0.0:
            right = middle
            right_value = middle_value
        else:
            left = middle
            left_value = middle_value
    raise _UnresolvedEvidenceError("e3b_crossover_root_did_not_converge")


def _point_crossover(
    fit_candidate: _SplineFit, fit_baseline: _SplineFit
) -> _PointCrossover:
    differences = np.asarray(
        [
            _log_difference(fit_candidate, fit_baseline, float(context))
            for context in E3B_CONTEXT_GRID
        ]
    )
    if np.all(np.abs(differences) <= _ROOT_LOG_DIFFERENCE_TOLERANCE):
        return _PointCrossover(
            outcome=E3bCrossoverOutcome.UNRESOLVED,
            bracket=None,
            root_tokens=None,
            reason_code="e3b_paired_goodput_curves_identical",
        )
    for index, (left, right) in enumerate(pairwise(differences)):
        if (
            abs(left) <= _ROOT_LOG_DIFFERENCE_TOLERANCE
            or abs(right) <= _ROOT_LOG_DIFFERENCE_TOLERANCE
            or left * right < 0.0
        ):
            bracket = (E3B_CONTEXT_GRID[index], E3B_CONTEXT_GRID[index + 1])
            return _PointCrossover(
                outcome=E3bCrossoverOutcome.CROSSOVER,
                bracket=bracket,
                root_tokens=_bisect_root(fit_candidate, fit_baseline, bracket),
                reason_code="e3b_first_registered_goodput_crossover",
            )
    return _PointCrossover(
        outcome=E3bCrossoverOutcome.NO_CROSSOVER_THROUGH_LIMIT,
        bracket=None,
        root_tokens=None,
        reason_code="e3b_no_crossover_through_40928",
    )


def _interval(estimate: float, bootstrap: np.ndarray) -> E3bIntervalEstimate:
    values = np.asarray(bootstrap, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise _UnresolvedEvidenceError("e3b_bootstrap_interval_nonfinite")
    alpha = (1.0 - REGISTERED_CONFIDENCE) / 2.0
    lower, upper = np.quantile(
        values,
        (alpha, 1.0 - alpha),
        method="linear",
    )
    return E3bIntervalEstimate(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
    )


def _reduce_crossover(
    plan: E3bLongContextAnalysisPlan,
    point_candidate: _SplineFit,
    point_baseline: _SplineFit,
    bootstrap_fits: Sequence[tuple[_SplineFit, _SplineFit]],
) -> E3bCrossoverReduction:
    if plan.metric is not E3bMetric.COMMITTED_TOKEN_GOODPUT:
        return E3bCrossoverReduction(
            outcome=E3bCrossoverOutcome.NOT_APPLICABLE,
            reason_code="e3b_crossover_requires_committed_token_goodput",
        )
    point = _point_crossover(point_candidate, point_baseline)
    bootstrap = tuple(
        _point_crossover(candidate, baseline) for candidate, baseline in bootstrap_fits
    )
    if point.outcome is E3bCrossoverOutcome.CROSSOVER:
        stable = all(
            value.outcome is E3bCrossoverOutcome.CROSSOVER
            and value.bracket == point.bracket
            and value.root_tokens is not None
            for value in bootstrap
        )
        if not stable:
            return E3bCrossoverReduction(
                outcome=E3bCrossoverOutcome.UNRESOLVED,
                reason_code="e3b_bootstrap_crossover_bracket_unstable",
            )
        roots = np.asarray([float(value.root_tokens) for value in bootstrap])
        alpha = (1.0 - REGISTERED_CONFIDENCE) / 2.0
        lower, upper = np.quantile(
            roots,
            (alpha, 1.0 - alpha),
            method="linear",
        )
        assert point.bracket is not None
        assert point.root_tokens is not None
        return E3bCrossoverReduction(
            outcome=E3bCrossoverOutcome.CROSSOVER,
            reason_code=point.reason_code,
            first_bracket_tokens=point.bracket,
            root_tokens=point.root_tokens,
            root_interval_tokens=(float(lower), float(upper)),
        )
    if point.outcome is E3bCrossoverOutcome.NO_CROSSOVER_THROUGH_LIMIT:
        if all(
            value.outcome is E3bCrossoverOutcome.NO_CROSSOVER_THROUGH_LIMIT
            for value in bootstrap
        ):
            return E3bCrossoverReduction(
                outcome=E3bCrossoverOutcome.NO_CROSSOVER_THROUGH_LIMIT,
                reason_code=point.reason_code,
            )
        return E3bCrossoverReduction(
            outcome=E3bCrossoverOutcome.UNRESOLVED,
            reason_code="e3b_bootstrap_no_crossover_classification_unstable",
        )
    return E3bCrossoverReduction(
        outcome=E3bCrossoverOutcome.UNRESOLVED,
        reason_code=point.reason_code,
    )


def reduce_e3b_long_context_pair(
    plan: E3bLongContextAnalysisPlan,
    observations: Sequence[E3bPairedRequestObservation],
) -> E3bLongContextReduction:
    """Reduce one exact E3b candidate/baseline family.

    The function has no ambient random state: the seed and replicate count are
    part of ``plan.sha256``.  Each replicate resamples registered final blocks,
    then paired request contributions within every sampled block/context cell,
    re-aggregates the metric, and refits both natural splines.
    """

    if type(plan) is not E3bLongContextAnalysisPlan:
        raise TypeError("E3b reduction requires an E3bLongContextAnalysisPlan")
    values = tuple(observations)
    try:
        grouped, infeasible_contexts, infeasible_methods = _validate_observations(
            plan, values
        )
        if infeasible_contexts:
            return _unresolved(
                plan,
                values,
                "e3b_registered_pair_hbm_infeasible",
                crossover=E3bCrossoverReduction(
                    outcome=E3bCrossoverOutcome.HBM_INFEASIBLE,
                    reason_code="e3b_registered_pair_hbm_infeasible",
                    infeasible_contexts=infeasible_contexts,
                    infeasible_methods=infeasible_methods,
                ),
            )

        candidate_values, baseline_values = _aggregate_context_curves(plan, grouped)
        point_candidate = _fit_registered_natural_log_spline(candidate_values)
        point_baseline = _fit_registered_natural_log_spline(baseline_values)
        point_candidate_evaluation = _evaluate_spline(point_candidate)
        point_baseline_evaluation = _evaluate_spline(point_baseline)

        bootstrap_candidate_values, bootstrap_baseline_values = (
            _bootstrap_context_curves(plan, grouped)
        )
        bootstrap_candidate_metric = np.empty_like(bootstrap_candidate_values)
        bootstrap_baseline_metric = np.empty_like(bootstrap_baseline_values)
        bootstrap_candidate_elasticity = np.empty_like(bootstrap_candidate_values)
        bootstrap_baseline_elasticity = np.empty_like(bootstrap_baseline_values)
        bootstrap_candidate_curvature = np.empty_like(bootstrap_candidate_values)
        bootstrap_baseline_curvature = np.empty_like(bootstrap_baseline_values)
        bootstrap_fits: list[tuple[_SplineFit, _SplineFit]] = []
        for repetition in range(plan.bootstrap_repetitions):
            candidate_fit = _fit_registered_natural_log_spline(
                bootstrap_candidate_values[repetition]
            )
            baseline_fit = _fit_registered_natural_log_spline(
                bootstrap_baseline_values[repetition]
            )
            candidate_evaluation = _evaluate_spline(candidate_fit)
            baseline_evaluation = _evaluate_spline(baseline_fit)
            bootstrap_candidate_metric[repetition] = candidate_evaluation.metric
            bootstrap_baseline_metric[repetition] = baseline_evaluation.metric
            bootstrap_candidate_elasticity[repetition] = candidate_evaluation.elasticity
            bootstrap_baseline_elasticity[repetition] = baseline_evaluation.elasticity
            bootstrap_candidate_curvature[repetition] = candidate_evaluation.curvature
            bootstrap_baseline_curvature[repetition] = baseline_evaluation.curvature
            bootstrap_fits.append((candidate_fit, baseline_fit))

        curve_points = tuple(
            E3bCurvePoint(
                context_tokens=context,
                candidate_fitted_metric=_interval(
                    point_candidate_evaluation.metric[index],
                    bootstrap_candidate_metric[:, index],
                ),
                baseline_fitted_metric=_interval(
                    point_baseline_evaluation.metric[index],
                    bootstrap_baseline_metric[:, index],
                ),
                candidate_elasticity=_interval(
                    point_candidate_evaluation.elasticity[index],
                    bootstrap_candidate_elasticity[:, index],
                ),
                baseline_elasticity=_interval(
                    point_baseline_evaluation.elasticity[index],
                    bootstrap_baseline_elasticity[:, index],
                ),
                paired_elasticity_difference=_interval(
                    point_candidate_evaluation.elasticity[index]
                    - point_baseline_evaluation.elasticity[index],
                    bootstrap_candidate_elasticity[:, index]
                    - bootstrap_baseline_elasticity[:, index],
                ),
                candidate_curvature=_interval(
                    point_candidate_evaluation.curvature[index],
                    bootstrap_candidate_curvature[:, index],
                ),
                baseline_curvature=_interval(
                    point_baseline_evaluation.curvature[index],
                    bootstrap_baseline_curvature[:, index],
                ),
                paired_curvature_difference=_interval(
                    point_candidate_evaluation.curvature[index]
                    - point_baseline_evaluation.curvature[index],
                    bootstrap_candidate_curvature[:, index]
                    - bootstrap_baseline_curvature[:, index],
                ),
            )
            for index, context in enumerate(E3B_CONTEXT_GRID)
        )
        crossover = _reduce_crossover(
            plan,
            point_candidate,
            point_baseline,
            bootstrap_fits,
        )
    except _UnresolvedEvidenceError as error:
        return _unresolved(plan, values, error.reason_code)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return _unresolved(plan, values, "e3b_numerical_reduction_failed")

    return E3bLongContextReduction(
        schema_version=1,
        status=E3bReductionStatus.OBSERVED,
        reason_code="e3b_registered_long_context_reduced",
        protocol_sha256=E3B_LONG_CONTEXT_PROTOCOL_SHA256,
        plan_sha256=plan.sha256,
        observations_sha256=_observation_sha256(values),
        curve_points=curve_points,
        crossover=crossover,
        bootstrap_repetitions_completed=plan.bootstrap_repetitions,
    )
