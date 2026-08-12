"""Paired repetition-block inference and fail-closed analysis contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.stats import nct, norm, t

from lightcone_spec.experiments.data import DFLASH_SAFE_CONTEXT_LIMIT

PILOT_BLOCK_COUNT = 4
MINIMUM_FINAL_BLOCKS = 12
MAXIMUM_FINAL_BLOCKS = 20
PRIMARY_CONTRASTS = ("l0_vs_static", "l0_vs_tts")
PRIMARY_FAMILY_ALPHA = 0.05
PRIMARY_MINIMUM_RELATIVE_EFFECT = 0.03
PRIMARY_TARGET_POWER = 0.80
REGISTERED_CONFIDENCE = 0.95
P99_MINIMUM_COMPLETIONS = 10_000
TTFT_LIMIT_MS: Mapping[str, float] = MappingProxyType(
    {
        "short": 2_000.0,
        "medium": 5_000.0,
        "long": 10_000.0,
    }
)
WITHIN_REQUEST_P99_ITL_LIMIT_MS = 100.0
MINIMUM_QUALIFICATION_RATE = 0.99
MAXIMUM_ERROR_RATE = 0.001
MINIMUM_COMPLETION_RATE = 0.999


def bca_mean_interval(
    cluster_values: dict[str, np.ndarray],
    *,
    confidence: float = 0.95,
    repetitions: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Cluster BCa interval for the mean paired effect."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if len(cluster_values) < 2:
        raise ValueError("BCa requires at least two independent clusters")
    if repetitions < 100:
        raise ValueError("bootstrap repetitions are too small")
    clusters = [
        np.asarray(cluster_values[key], dtype=np.float64)
        for key in sorted(cluster_values)
    ]
    if any(values.size == 0 or not np.isfinite(values).all() for values in clusters):
        raise ValueError("cluster effects must be finite and non-empty")
    cluster_means = np.asarray([values.mean() for values in clusters])
    estimate = float(cluster_means.mean())
    randomizer = np.random.default_rng(seed)
    indices = randomizer.integers(0, len(clusters), size=(repetitions, len(clusters)))
    bootstrap = cluster_means[indices].mean(axis=1)
    less = np.count_nonzero(bootstrap < estimate)
    equal = np.count_nonzero(bootstrap == estimate)
    probability = (less + 0.5 * equal) / repetitions
    probability = np.clip(
        probability,
        1.0 / (2 * repetitions),
        1 - 1.0 / (2 * repetitions),
    )
    bias = norm.ppf(probability)
    jackknife = np.asarray(
        [np.delete(cluster_means, index).mean() for index in range(len(clusters))]
    )
    centered = jackknife.mean() - jackknife
    denominator = 6.0 * np.power(np.square(centered).sum(), 1.5)
    acceleration = (
        float(np.power(centered, 3).sum() / denominator)
        if denominator > np.finfo(np.float64).tiny
        else 0.0
    )
    alpha = (1.0 - confidence) / 2.0
    normal_quantiles = norm.ppf(np.asarray([alpha, 1.0 - alpha]))
    divisor = 1.0 - acceleration * (bias + normal_quantiles)
    divisor = np.where(
        np.abs(divisor) < np.finfo(np.float64).eps,
        np.copysign(np.finfo(np.float64).eps, divisor),
        divisor,
    )
    adjusted = norm.cdf(bias + (bias + normal_quantiles) / divisor)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    lower, upper = np.quantile(bootstrap, adjusted)
    return estimate, float(lower), float(upper)


@dataclass(frozen=True)
class PilotBlock:
    """One excluded pilot block with all registered primary methods."""

    block_id: str
    static_goodput: float
    tts_goodput: float
    l0_goodput: float


@dataclass(frozen=True)
class ContrastPower:
    """Power of one registered contrast at one prospective block count."""

    contrast: str
    final_blocks: int
    power: float


@dataclass(frozen=True)
class PowerSizingPlan:
    """A power decision frozen from exactly four excluded pilot blocks."""

    status: str
    pilot_block_ids: tuple[str, ...]
    selected_final_blocks: int | None
    minimum_final_blocks: int
    maximum_final_blocks: int
    target_power: float
    family_alpha: float
    adjusted_alpha: float
    minimum_relative_effect: float
    minimum_log_effect: float
    pilot_log_standard_deviations: tuple[tuple[str, float], ...]
    power_grid: tuple[ContrastPower, ...]

    @property
    def underpowered(self) -> bool:
        return self.status == "UNDERPOWERED"

    def power(self, contrast: str, final_blocks: int) -> float:
        """Return a registered grid value without interpolating missing cells."""
        matches = [
            cell.power
            for cell in self.power_grid
            if cell.contrast == contrast and cell.final_blocks == final_blocks
        ]
        if len(matches) != 1:
            raise ValueError("requested power cell is not registered")
        return matches[0]


def _finite_positive(value: float, *, field: str) -> float:
    checked = float(value)
    if not np.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return checked


def _paired_t_power(
    *,
    final_blocks: int,
    log_effect: float,
    pilot_standard_deviation: float,
    alpha: float,
) -> float:
    degrees_of_freedom = final_blocks - 1
    noncentrality = log_effect * np.sqrt(final_blocks) / pilot_standard_deviation
    critical = t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)
    lower_tail = nct.sf(critical, degrees_of_freedom, -noncentrality)
    upper_tail = nct.sf(critical, degrees_of_freedom, noncentrality)
    power = float(lower_tail + upper_tail)
    if not np.isfinite(power) or not 0.0 <= power <= 1.0:
        raise ValueError("power calculation produced an invalid probability")
    return power


def preregister_power_sizing(
    pilot_blocks: Sequence[PilotBlock],
    *,
    family_alpha: float = PRIMARY_FAMILY_ALPHA,
    minimum_relative_effect: float = PRIMARY_MINIMUM_RELATIVE_EFFECT,
    target_power: float = PRIMARY_TARGET_POWER,
    minimum_final_blocks: int = MINIMUM_FINAL_BLOCKS,
    maximum_final_blocks: int = MAXIMUM_FINAL_BLOCKS,
) -> PowerSizingPlan:
    """Fix 12--20 final blocks from four excluded paired pilot blocks.

    The calculation uses the first Holm threshold for the two registered
    primary contrasts.  A result that cannot reach the target by the maximum
    is explicit ``UNDERPOWERED``; it is never rounded up to a passing plan.
    """
    blocks = tuple(pilot_blocks)
    if len(blocks) != PILOT_BLOCK_COUNT:
        raise ValueError("power sizing requires exactly four pilot blocks")
    block_ids = tuple(block.block_id for block in blocks)
    if any(not isinstance(block_id, str) or not block_id for block_id in block_ids):
        raise ValueError("pilot block IDs must be non-empty")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("pilot block IDs must be unique")
    if not 0.0 < family_alpha < 1.0:
        raise ValueError("family alpha must be in (0, 1)")
    if not 0.0 < minimum_relative_effect < 1.0:
        raise ValueError("minimum relative effect must be in (0, 1)")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target power must be in (0, 1)")
    if (
        family_alpha != PRIMARY_FAMILY_ALPHA
        or minimum_relative_effect != PRIMARY_MINIMUM_RELATIVE_EFFECT
        or target_power != PRIMARY_TARGET_POWER
    ):
        raise ValueError(
            "power alpha, 3% effect, and 80% target are preregistered and fixed"
        )
    if (
        minimum_final_blocks != MINIMUM_FINAL_BLOCKS
        or maximum_final_blocks != MAXIMUM_FINAL_BLOCKS
    ):
        raise ValueError("the registered final block range is fixed at 12--20")

    effects: dict[str, list[float]] = {contrast: [] for contrast in PRIMARY_CONTRASTS}
    for block in blocks:
        static = _finite_positive(
            block.static_goodput,
            field="pilot Static goodput",
        )
        tts = _finite_positive(block.tts_goodput, field="pilot TTS goodput")
        l0 = _finite_positive(block.l0_goodput, field="pilot L0 goodput")
        effects["l0_vs_static"].append(float(np.log(l0 / static)))
        effects["l0_vs_tts"].append(float(np.log(l0 / tts)))

    standard_deviations: dict[str, float] = {}
    for contrast, values in effects.items():
        deviation = float(np.std(np.asarray(values), ddof=1))
        if not np.isfinite(deviation) or deviation <= 0.0:
            raise ValueError(
                f"{contrast} pilot log effects need positive finite variance"
            )
        standard_deviations[contrast] = deviation

    adjusted_alpha = family_alpha / len(PRIMARY_CONTRASTS)
    log_effect = float(np.log1p(minimum_relative_effect))
    grid = tuple(
        ContrastPower(
            contrast=contrast,
            final_blocks=final_blocks,
            power=_paired_t_power(
                final_blocks=final_blocks,
                log_effect=log_effect,
                pilot_standard_deviation=standard_deviations[contrast],
                alpha=adjusted_alpha,
            ),
        )
        for final_blocks in range(minimum_final_blocks, maximum_final_blocks + 1)
        for contrast in PRIMARY_CONTRASTS
    )
    selected = next(
        (
            final_blocks
            for final_blocks in range(
                minimum_final_blocks,
                maximum_final_blocks + 1,
            )
            if all(
                next(
                    cell.power
                    for cell in grid
                    if cell.final_blocks == final_blocks and cell.contrast == contrast
                )
                >= target_power
                for contrast in PRIMARY_CONTRASTS
            )
        ),
        None,
    )
    return PowerSizingPlan(
        status="READY" if selected is not None else "UNDERPOWERED",
        pilot_block_ids=block_ids,
        selected_final_blocks=selected,
        minimum_final_blocks=minimum_final_blocks,
        maximum_final_blocks=maximum_final_blocks,
        target_power=target_power,
        family_alpha=family_alpha,
        adjusted_alpha=adjusted_alpha,
        minimum_relative_effect=minimum_relative_effect,
        minimum_log_effect=log_effect,
        pilot_log_standard_deviations=tuple(
            (contrast, standard_deviations[contrast]) for contrast in PRIMARY_CONTRASTS
        ),
        power_grid=grid,
    )


def validate_final_block_ids(
    plan: PowerSizingPlan,
    final_block_ids: Sequence[str],
) -> tuple[str, ...]:
    """Bind the frozen block count and prove pilots are excluded."""
    if plan.underpowered or plan.selected_final_blocks is None:
        raise ValueError("UNDERPOWERED plans cannot start confirmation")
    block_ids = tuple(final_block_ids)
    if len(block_ids) != plan.selected_final_blocks:
        raise ValueError("final block count does not match the power plan")
    if any(not isinstance(block_id, str) or not block_id for block_id in block_ids):
        raise ValueError("final block IDs must be non-empty")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("final block IDs must be unique")
    if set(block_ids) & set(plan.pilot_block_ids):
        raise ValueError("pilot blocks must be excluded from confirmation")
    return block_ids


@dataclass(frozen=True)
class PairedBcaContrast:
    """A paired log-ratio contrast whose independent unit is one block."""

    name: str
    block_ids: tuple[str, ...]
    mean_log_ratio: float
    mean_relative_gain: float
    ci_lower_relative_gain: float
    ci_upper_relative_gain: float
    raw_p_value: float
    confidence: float
    independent_unit: str = "paired_block"


def paired_bca_contrast(
    name: str,
    paired_goodput: Mapping[str, tuple[float, float]],
    *,
    confidence: float = 0.95,
    repetitions: int = 10_000,
    seed: int = 0,
) -> PairedBcaContrast:
    """Estimate numerator/denominator goodput as a paired log contrast."""
    if confidence != REGISTERED_CONFIDENCE:
        raise ValueError("registered analysis intervals are fixed at 95%")
    if not isinstance(name, str) or not name:
        raise ValueError("contrast name must be non-empty")
    if len(paired_goodput) < 2:
        raise ValueError("paired inference requires at least two blocks")
    unsorted_block_ids = tuple(paired_goodput)
    if any(
        not isinstance(block_id, str) or not block_id for block_id in unsorted_block_ids
    ):
        raise ValueError("paired block IDs must be non-empty")
    block_ids = tuple(sorted(unsorted_block_ids))
    log_effects: dict[str, np.ndarray] = {}
    for block_id in block_ids:
        pair = paired_goodput[block_id]
        if len(pair) != 2:
            raise ValueError("each paired block requires numerator and denominator")
        numerator = _finite_positive(pair[0], field="numerator goodput")
        denominator = _finite_positive(pair[1], field="denominator goodput")
        log_effects[block_id] = np.asarray([np.log(numerator / denominator)])
    estimate, lower, upper = bca_mean_interval(
        log_effects,
        confidence=confidence,
        repetitions=repetitions,
        seed=seed,
    )
    values = np.asarray([log_effects[key][0] for key in block_ids])
    sample_standard_deviation = float(np.std(values, ddof=1))
    if sample_standard_deviation <= np.finfo(np.float64).tiny:
        raw_p_value = 1.0 if abs(estimate) <= np.finfo(np.float64).eps else 0.0
    else:
        statistic = estimate / (sample_standard_deviation / np.sqrt(values.size))
        raw_p_value = float(2.0 * t.sf(abs(statistic), values.size - 1))
    if not np.isfinite(raw_p_value) or not 0.0 <= raw_p_value <= 1.0:
        raise ValueError("paired contrast produced an invalid p-value")
    return PairedBcaContrast(
        name=name,
        block_ids=block_ids,
        mean_log_ratio=estimate,
        mean_relative_gain=float(np.expm1(estimate)),
        ci_lower_relative_gain=float(np.expm1(lower)),
        ci_upper_relative_gain=float(np.expm1(upper)),
        raw_p_value=raw_p_value,
        confidence=confidence,
    )


@dataclass(frozen=True)
class MultiplicityDecision:
    """One hypothesis after a named family-wise or FDR adjustment."""

    name: str
    raw_p_value: float
    adjusted_p_value: float
    rejected: bool
    procedure: str


def _validate_p_values(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        raise ValueError("a multiplicity family cannot be empty")
    checked: dict[str, float] = {}
    for name, raw_value in p_values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("hypothesis names must be non-empty")
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be finite probabilities")
        checked[name] = value
    return checked


def holm_primary_contrasts(
    contrasts: Mapping[str, PairedBcaContrast],
    *,
    alpha: float = PRIMARY_FAMILY_ALPHA,
) -> tuple[MultiplicityDecision, ...]:
    """Holm-adjust exactly L0--Static and L0--TTS."""
    if set(contrasts) != set(PRIMARY_CONTRASTS):
        raise ValueError("primary family requires L0--Static and L0--TTS")
    if any(name != contrast.name for name, contrast in contrasts.items()):
        raise ValueError("primary contrast keys and names must agree")
    if not 0.0 < alpha < 1.0:
        raise ValueError("family alpha must be in (0, 1)")
    if alpha != PRIMARY_FAMILY_ALPHA:
        raise ValueError("primary family alpha is preregistered and fixed")
    p_values = _validate_p_values(
        {name: contrast.raw_p_value for name, contrast in contrasts.items()}
    )
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    family_size = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * p_values[name])
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = running_maximum
    return tuple(
        MultiplicityDecision(
            name=name,
            raw_p_value=p_values[name],
            adjusted_p_value=adjusted[name],
            rejected=adjusted[name] <= alpha,
            procedure="holm",
        )
        for name in PRIMARY_CONTRASTS
    )


def benjamini_hochberg(
    p_values: Mapping[str, float],
    *,
    false_discovery_rate: float = 0.05,
) -> tuple[MultiplicityDecision, ...]:
    """Adjust one explicitly supplied secondary breadth family with BH FDR."""
    checked = _validate_p_values(p_values)
    if not 0.0 < false_discovery_rate < 1.0:
        raise ValueError("false discovery rate must be in (0, 1)")
    ordered = sorted(checked, key=lambda name: (checked[name], name))
    family_size = len(ordered)
    adjusted: dict[str, float] = {}
    running_minimum = 1.0
    for rank in range(family_size, 0, -1):
        name = ordered[rank - 1]
        candidate = min(1.0, checked[name] * family_size / rank)
        running_minimum = min(running_minimum, candidate)
        adjusted[name] = running_minimum
    return tuple(
        MultiplicityDecision(
            name=name,
            raw_p_value=checked[name],
            adjusted_p_value=adjusted[name],
            rejected=adjusted[name] <= false_discovery_rate,
            procedure="benjamini-hochberg",
        )
        for name in sorted(checked)
    )


@dataclass(frozen=True)
class BootstrapInterval:
    """A vector-valued percentile interval with named resampling units."""

    estimate: tuple[float, ...]
    ci_lower: tuple[float, ...]
    ci_upper: tuple[float, ...]
    confidence: float
    repetitions: int
    independent_units: tuple[str, ...]


BootstrapStatistic = Callable[[np.ndarray], float | np.ndarray]


def _validated_cluster_rows(
    cluster_rows: Mapping[str, np.ndarray],
    *,
    unit: str,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    if len(cluster_rows) < 2:
        raise ValueError(f"{unit} bootstrap requires at least two blocks")
    unsorted_keys = tuple(cluster_rows)
    if any(not isinstance(key, str) or not key for key in unsorted_keys):
        raise ValueError(f"{unit} block IDs must be non-empty")
    keys = tuple(sorted(unsorted_keys))
    arrays = tuple(np.asarray(cluster_rows[key], dtype=np.float64) for key in keys)
    if any(array.ndim not in (1, 2) or array.shape[0] == 0 for array in arrays):
        raise ValueError(f"{unit} blocks must contain observed rows")
    reference = arrays[0]
    if any(
        array.ndim != reference.ndim or array.shape[1:] != reference.shape[1:]
        for array in arrays
    ):
        raise ValueError(f"{unit} block row shapes must agree")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{unit} block rows must be finite")
    return keys, arrays


def _statistic_vector(
    statistic: BootstrapStatistic,
    rows: np.ndarray,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    result = np.asarray(statistic(rows), dtype=np.float64)
    if result.ndim == 0:
        result = result.reshape(1)
    if (
        result.ndim != 1
        or result.size == 0
        or not np.isfinite(result).all()
        or (expected_size is not None and result.size != expected_size)
    ):
        raise ValueError("bootstrap statistic must return a fixed finite vector")
    return result


def _cluster_bootstrap(
    cluster_rows: Mapping[str, np.ndarray],
    statistic: BootstrapStatistic,
    *,
    within_cluster: bool,
    confidence: float,
    repetitions: int,
    seed: int,
    unit: str,
) -> BootstrapInterval:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if confidence != REGISTERED_CONFIDENCE:
        raise ValueError("registered analysis intervals are fixed at 95%")
    if repetitions < 100:
        raise ValueError("bootstrap repetitions are too small")
    if not isinstance(seed, int):
        raise TypeError("bootstrap seed must be an integer")
    _, arrays = _validated_cluster_rows(cluster_rows, unit=unit)
    estimate = _statistic_vector(statistic, np.concatenate(arrays, axis=0))
    randomizer = np.random.default_rng(seed)
    samples = np.empty((repetitions, estimate.size), dtype=np.float64)
    for repetition in range(repetitions):
        block_indices = randomizer.integers(0, len(arrays), size=len(arrays))
        sampled_blocks: list[np.ndarray] = []
        for block_index in block_indices:
            block = arrays[int(block_index)]
            if within_cluster:
                request_indices = randomizer.integers(
                    0,
                    block.shape[0],
                    size=block.shape[0],
                )
                block = block[request_indices]
            sampled_blocks.append(block)
        samples[repetition] = _statistic_vector(
            statistic,
            np.concatenate(sampled_blocks, axis=0),
            expected_size=estimate.size,
        )
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha), axis=0)
    return BootstrapInterval(
        estimate=tuple(float(value) for value in estimate),
        ci_lower=tuple(float(value) for value in lower),
        ci_upper=tuple(float(value) for value in upper),
        confidence=confidence,
        repetitions=repetitions,
        independent_units=(unit, "request") if within_cluster else (unit,),
    )


def hierarchical_block_request_bootstrap(
    block_request_rows: Mapping[str, np.ndarray],
    statistic: BootstrapStatistic,
    *,
    confidence: float = 0.95,
    repetitions: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Resample independent blocks, then requests within sampled blocks."""
    return _cluster_bootstrap(
        block_request_rows,
        statistic,
        within_cluster=True,
        confidence=confidence,
        repetitions=repetitions,
        seed=seed,
        unit="block",
    )


def time_block_bootstrap(
    time_block_rows: Mapping[str, np.ndarray],
    statistic: BootstrapStatistic,
    *,
    confidence: float = 0.95,
    repetitions: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Resample whole time blocks while preserving within-block tail dependence."""
    return _cluster_bootstrap(
        time_block_rows,
        statistic,
        within_cluster=False,
        confidence=confidence,
        repetitions=repetitions,
        seed=seed,
        unit="time_block",
    )


@dataclass(frozen=True)
class P99ClaimGuard:
    """Whether an observed anchor p99 has enough completed requests to claim."""

    anchor_id: str
    completed_requests: int
    observed_p99_ms: float | None
    minimum_completions: int
    status: str

    @property
    def claimable(self) -> bool:
        return self.status == "CLAIMABLE"


def guard_p99_claim(
    anchor_id: str,
    *,
    completed_requests: int,
    observed_p99_ms: float | None,
    minimum_completions: int,
    preregistered_anchor_locked: bool,
) -> P99ClaimGuard:
    """Gate one p99 only with its explicit preregistered anchor authority."""
    if not isinstance(anchor_id, str) or not anchor_id:
        raise ValueError("anchor ID must be non-empty")
    if (
        not isinstance(completed_requests, int)
        or isinstance(completed_requests, bool)
        or completed_requests < 0
    ):
        raise ValueError("completed request count must be a non-negative integer")
    if (
        not isinstance(minimum_completions, int)
        or isinstance(minimum_completions, bool)
        or minimum_completions < 0
    ):
        raise ValueError("minimum completions must be a non-negative integer")
    if not isinstance(preregistered_anchor_locked, bool):
        raise TypeError("preregistered anchor lock must be boolean")
    if preregistered_anchor_locked and minimum_completions == 0:
        raise ValueError("locked p99 anchors require a positive completion minimum")
    if observed_p99_ms is not None:
        observed_p99_ms = float(observed_p99_ms)
        if not np.isfinite(observed_p99_ms) or observed_p99_ms < 0.0:
            raise ValueError("observed p99 must be finite and non-negative")
    eligible = preregistered_anchor_locked and completed_requests >= minimum_completions
    if eligible and observed_p99_ms is None:
        raise ValueError("claimable anchors require an observed p99")
    if not eligible and observed_p99_ms is not None:
        raise ValueError("unresolved p99 anchors cannot expose an observed p99")
    return P99ClaimGuard(
        anchor_id=anchor_id,
        completed_requests=completed_requests,
        observed_p99_ms=observed_p99_ms,
        minimum_completions=minimum_completions,
        status="CLAIMABLE" if eligible else "UNRESOLVED",
    )


@dataclass(frozen=True)
class SloRequest:
    """One request with eligibility frozen before adaptation."""

    request_id: str
    prompt_bucket: str
    eligible: bool
    completed: bool
    error: bool
    ttft_ms: float | None
    within_request_p99_itl_ms: float | None


@dataclass(frozen=True)
class SloAccounting:
    """Qualification, error, and completion accounting over eligible requests."""

    status: str
    eligible_requests: int
    qualified_requests: int
    error_requests: int
    completed_requests: int
    qualification_rate: float
    error_rate: float
    completion_rate: float
    ttft_limits_ms: tuple[tuple[str, float], ...]
    within_request_p99_itl_limit_ms: float

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def _optional_nonnegative_metric(value: float | None, *, field: str) -> float | None:
    if value is None:
        return None
    checked = float(value)
    if not np.isfinite(checked) or checked < 0.0:
        raise ValueError(f"{field} must be finite and non-negative when observed")
    return checked


def account_slo(requests: Sequence[SloRequest]) -> SloAccounting:
    """Apply the registered request-level production SLO without imputation."""
    rows = tuple(requests)
    if not rows:
        raise ValueError("SLO accounting requires request rows")
    request_ids = tuple(row.request_id for row in rows)
    if any(
        not isinstance(request_id, str) or not request_id for request_id in request_ids
    ):
        raise ValueError("request IDs must be non-empty")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request IDs must be unique")

    eligible_rows: list[tuple[SloRequest, float | None, float | None]] = []
    for row in rows:
        if row.prompt_bucket not in TTFT_LIMIT_MS:
            raise ValueError("prompt bucket must be short, medium, or long")
        if not all(
            isinstance(value, bool)
            for value in (row.eligible, row.completed, row.error)
        ):
            raise ValueError("eligibility, completion, and error must be boolean")
        ttft = _optional_nonnegative_metric(row.ttft_ms, field="TTFT")
        itl = _optional_nonnegative_metric(
            row.within_request_p99_itl_ms,
            field="within-request p99 ITL",
        )
        if row.eligible:
            eligible_rows.append((row, ttft, itl))
    if not eligible_rows:
        raise ValueError("SLO accounting requires eligible requests")

    qualified = 0
    errors = 0
    completed = 0
    for row, ttft, itl in eligible_rows:
        errors += int(row.error)
        completed += int(row.completed)
        qualifies = (
            row.completed
            and not row.error
            and ttft is not None
            and itl is not None
            and ttft <= TTFT_LIMIT_MS[row.prompt_bucket]
            and itl <= WITHIN_REQUEST_P99_ITL_LIMIT_MS
        )
        qualified += int(qualifies)
    eligible_count = len(eligible_rows)
    qualification_rate = qualified / eligible_count
    error_rate = errors / eligible_count
    completion_rate = completed / eligible_count
    passed = (
        qualified * 100 >= eligible_count * 99
        and errors * 1_000 <= eligible_count
        and completed * 1_000 >= eligible_count * 999
    )
    return SloAccounting(
        status="PASS" if passed else "FAIL",
        eligible_requests=eligible_count,
        qualified_requests=qualified,
        error_requests=errors,
        completed_requests=completed,
        qualification_rate=qualification_rate,
        error_rate=error_rate,
        completion_rate=completion_rate,
        ttft_limits_ms=tuple(TTFT_LIMIT_MS.items()),
        within_request_p99_itl_limit_ms=WITHIN_REQUEST_P99_ITL_LIMIT_MS,
    )


@dataclass(frozen=True)
class HardwareEnvelope:
    """Registered per-block operating bounds."""

    gpu_clock_mhz_min: float
    gpu_clock_mhz_max: float
    memory_clock_mhz_min: float
    memory_clock_mhz_max: float
    temperature_c_max: float
    power_watts_min: float
    power_watts_max: float
    power_state: str
    allowed_throttling_reasons: tuple[str, ...] = ()
    allowed_background_processes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HardwareBlockObservation:
    """Observed block environment; ``None`` means evidence is missing."""

    block_id: str
    gpu_clock_mhz: float | None
    memory_clock_mhz: float | None
    temperature_c: float | None
    power_watts: float | None
    power_state: str | None
    throttling_reasons: tuple[str, ...] | None
    background_processes: tuple[str, ...] | None


@dataclass(frozen=True)
class HardwareBlockValidity:
    """A block is either valid or explicitly invalidated with reasons."""

    block_id: str
    status: str
    reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.status == "VALID"


def _validate_hardware_envelope(envelope: HardwareEnvelope) -> None:
    ranges = (
        (
            "gpu clock",
            envelope.gpu_clock_mhz_min,
            envelope.gpu_clock_mhz_max,
        ),
        (
            "memory clock",
            envelope.memory_clock_mhz_min,
            envelope.memory_clock_mhz_max,
        ),
        ("power", envelope.power_watts_min, envelope.power_watts_max),
    )
    for name, lower, upper in ranges:
        if not np.isfinite([lower, upper]).all() or lower <= 0.0 or upper < lower:
            raise ValueError(f"hardware {name} envelope is invalid")
    if not np.isfinite(envelope.temperature_c_max) or envelope.temperature_c_max <= 0:
        raise ValueError("hardware temperature envelope is invalid")
    if not isinstance(envelope.power_state, str) or not envelope.power_state:
        raise ValueError("registered power state must be non-empty")
    if any(
        not isinstance(value, str) or not value
        for value in (
            *envelope.allowed_throttling_reasons,
            *envelope.allowed_background_processes,
        )
    ):
        raise ValueError("allowed hardware labels must be non-empty strings")
    if len(set(envelope.allowed_throttling_reasons)) != len(
        envelope.allowed_throttling_reasons
    ):
        raise ValueError("allowed throttling reasons must be unique")
    if len(set(envelope.allowed_background_processes)) != len(
        envelope.allowed_background_processes
    ):
        raise ValueError("allowed background processes must be unique")


def validate_hardware_block(
    envelope: HardwareEnvelope,
    observation: HardwareBlockObservation,
) -> HardwareBlockValidity:
    """Invalidate a block with missing or out-of-envelope hardware evidence."""
    _validate_hardware_envelope(envelope)
    if not isinstance(observation.block_id, str) or not observation.block_id:
        raise ValueError("hardware block ID must be non-empty")
    for name, labels in (
        ("throttling reasons", observation.throttling_reasons),
        ("background processes", observation.background_processes),
    ):
        if labels is not None and any(
            not isinstance(value, str) or not value for value in labels
        ):
            raise ValueError(f"observed {name} must be non-empty strings")
        if labels is not None and len(set(labels)) != len(labels):
            raise ValueError(f"observed {name} must be unique")
    reasons: list[str] = []
    measurements = (
        (
            "gpu_clock_mhz",
            observation.gpu_clock_mhz,
            envelope.gpu_clock_mhz_min,
            envelope.gpu_clock_mhz_max,
        ),
        (
            "memory_clock_mhz",
            observation.memory_clock_mhz,
            envelope.memory_clock_mhz_min,
            envelope.memory_clock_mhz_max,
        ),
        (
            "power_watts",
            observation.power_watts,
            envelope.power_watts_min,
            envelope.power_watts_max,
        ),
    )
    for name, raw_value, lower, upper in measurements:
        if raw_value is None:
            reasons.append(f"{name}:missing")
            continue
        value = float(raw_value)
        if not np.isfinite(value):
            reasons.append(f"{name}:nonfinite")
        elif value < lower:
            reasons.append(f"{name}:below_min")
        elif value > upper:
            reasons.append(f"{name}:above_max")
    if observation.temperature_c is None:
        reasons.append("temperature_c:missing")
    else:
        temperature = float(observation.temperature_c)
        if not np.isfinite(temperature):
            reasons.append("temperature_c:nonfinite")
        elif temperature > envelope.temperature_c_max:
            reasons.append("temperature_c:above_max")
    if observation.power_state is None:
        reasons.append("power_state:missing")
    elif observation.power_state != envelope.power_state:
        reasons.append("power_state:mismatch")
    if observation.throttling_reasons is None:
        reasons.append("throttling_reasons:missing")
    else:
        unexpected_throttling = set(observation.throttling_reasons) - set(
            envelope.allowed_throttling_reasons
        )
        reasons.extend(
            f"throttling_reason:unexpected:{reason}"
            for reason in sorted(unexpected_throttling)
        )
    if observation.background_processes is None:
        reasons.append("background_processes:missing")
    else:
        unexpected_processes = set(observation.background_processes) - set(
            envelope.allowed_background_processes
        )
        reasons.extend(
            f"background_process:unexpected:{process}"
            for process in sorted(unexpected_processes)
        )
    reasons_tuple = tuple(reasons)
    return HardwareBlockValidity(
        block_id=observation.block_id,
        status="INVALIDATED" if reasons_tuple else "VALID",
        reasons=reasons_tuple,
    )


def validate_hardware_blocks(
    envelope: HardwareEnvelope,
    observations: Sequence[HardwareBlockObservation],
) -> tuple[HardwareBlockValidity, ...]:
    """Validate a complete named set while rejecting duplicate block evidence."""
    rows = tuple(observations)
    if not rows:
        raise ValueError("hardware validation requires block observations")
    block_ids = tuple(row.block_id for row in rows)
    if any(not isinstance(block_id, str) or not block_id for block_id in block_ids):
        raise ValueError("hardware block IDs must be non-empty")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("hardware block observations must be unique")
    return tuple(validate_hardware_block(envelope, row) for row in rows)


@dataclass(frozen=True)
class MethodSpeedGate:
    method: str
    mean_speedup: float
    ci_lower: float
    ci_upper: float
    safety_pass: bool
    acceleration_pass: bool

    @property
    def passed(self) -> bool:
        return self.safety_pass and self.acceleration_pass


@dataclass(frozen=True)
class PairwiseSpeedGate:
    numerator_method: str
    denominator_method: str
    mean_speedup: float
    ci_lower: float
    ci_upper: float
    no_worse_pass: bool

    @property
    def passed(self) -> bool:
        return self.no_worse_pass


@dataclass(frozen=True)
class SpeedGate:
    status: str
    tts: MethodSpeedGate
    l0: MethodSpeedGate
    l0_vs_tts: PairwiseSpeedGate
    gpu_evidence: str
    evidence_sha256: str | None

    @property
    def passed(self) -> bool:
        return (
            self.gpu_evidence == "MEASURED"
            and self.evidence_sha256 is not None
            and self.tts.passed
            and self.l0.passed
            and self.l0_vs_tts.passed
        )


def _validate_coverage(
    rows: list[dict],
) -> tuple[dict[int, dict[str, dict]], dict[int, dict[str, dict]]]:
    required_methods = {"static", "tts", "l0"}
    if {str(row["method"]) for row in rows} != required_methods:
        raise ValueError("formal gate requires exactly Static, TTS, and L0")
    filtered = [row for row in rows if str(row.get("region")) == "long_region"]
    by_key: dict[int, dict[str, dict]] = {}
    for row in filtered:
        key = int(row["repetition_block"])
        method_rows = by_key.setdefault(key, {})
        method = str(row["method"])
        if method in method_rows:
            raise ValueError("duplicate paired performance row")
        method_rows[method] = row
    if not by_key or any(set(group) != required_methods for group in by_key.values()):
        raise ValueError("paired long-region coverage is incomplete")
    prompt_batches = {
        str(row["prompt_id"]) for group in by_key.values() for row in group.values()
    }
    blocks = set(by_key)
    if len(prompt_batches) != 1 or not next(iter(prompt_batches)).startswith("batch-"):
        raise ValueError("formal gate requires one jointly timed confirmation batch")
    if blocks != set(range(8)):
        raise ValueError("formal gate requires eight independent blocks")
    concurrencies = {
        int(row["concurrency"]) for group in by_key.values() for row in group.values()
    }
    if len(concurrencies) != 1 or next(iter(concurrencies)) < 1:
        raise ValueError("formal methods must share one positive load")
    long_ends = {
        int(row["generated_bucket_end"])
        for group in by_key.values()
        for row in group.values()
    }
    if (
        len(long_ends) != 1
        or not 16384 < next(iter(long_ends)) < DFLASH_SAFE_CONTEXT_LIMIT
    ):
        raise ValueError("long-region bounds must be shared generated-token positions")
    for group in by_key.values():
        at_risk = {int(row["at_risk_requests"]) for row in group.values()}
        output = {int(row["output_tokens"]) for row in group.values()}
        starts = {int(row["generated_bucket_start"]) for row in group.values()}
        ends = {int(row["generated_bucket_end"]) for row in group.values()}
        if (
            len(at_risk) != 1
            or len(output) != 1
            or min(at_risk | output) < 1
            or starts != {16384}
            or ends != long_ends
        ):
            raise ValueError("paired methods do not share the same at-risk sample")
    full_rows = [row for row in rows if str(row.get("region")) == "full_trajectory"]
    full_by_block: dict[int, dict[str, dict]] = {}
    for row in full_rows:
        block = int(row["repetition_block"])
        method = str(row["method"])
        methods = full_by_block.setdefault(block, {})
        if method in methods:
            raise ValueError("duplicate run-scope performance row")
        methods[method] = row
    if set(full_by_block) != set(range(8)) or any(
        set(methods) != required_methods for methods in full_by_block.values()
    ):
        raise ValueError("run-scope safety coverage is incomplete")
    if any(
        str(row.get("prompt_id")) not in prompt_batches
        or int(row.get("concurrency", -1)) not in concurrencies
        for methods in full_by_block.values()
        for row in methods.values()
    ):
        raise ValueError("run-scope evidence is not bound to the timed batch")
    return by_key, full_by_block


def evaluate_speed_gate(
    rows: list[dict],
    *,
    minimum_speedup: float = 0.03,
    seed: int = 0,
    gpu_evidence: str = "UNMEASURED",
    evidence_sha256: str | None = None,
) -> SpeedGate:
    """Evaluate 16K-to-limit paired goodput with strict safety counters."""
    if minimum_speedup < 0:
        raise ValueError("minimum speedup cannot be negative")
    if gpu_evidence not in {"UNMEASURED", "MEASURED"}:
        raise ValueError("unknown GPU evidence state")
    if gpu_evidence == "MEASURED" or evidence_sha256 is not None:
        raise ValueError("trusted hardware attestation is unavailable in this release")
    by_key, full_by_block = _validate_coverage(rows)
    gates: dict[str, MethodSpeedGate] = {}
    for method in ("tts", "l0"):
        clusters: dict[str, list[float]] = {}
        safety_pass = True
        for block, group in by_key.items():
            static_goodput = float(group["static"]["decode_goodput_tps"])
            method_goodput = float(group[method]["decode_goodput_tps"])
            if (
                static_goodput <= 0
                or method_goodput <= 0
                or not np.isfinite([static_goodput, method_goodput]).all()
            ):
                raise ValueError("goodput must be finite and positive")
            clusters[str(block)] = [method_goodput / static_goodput - 1.0]
        for block, group in full_by_block.items():
            for field in (
                "exactness_violations",
                "version_mismatches",
                "fallbacks",
                "nonfinite_updates",
                "oom_events",
                "retractions",
            ):
                safety_pass &= int(group[method][field]) == 0
                safety_pass &= int(group["static"][field]) == 0
            safety_pass &= int(group[method]["updates_launched"]) > 0
            safety_pass &= int(group[method]["updates_published"]) > 0
        estimate, lower, upper = bca_mean_interval(
            {cluster: np.asarray(values) for cluster, values in clusters.items()},
            seed=seed + (0 if method == "tts" else 1),
        )
        gates[method] = MethodSpeedGate(
            method=method,
            mean_speedup=estimate,
            ci_lower=lower,
            ci_upper=upper,
            safety_pass=safety_pass,
            acceleration_pass=estimate >= minimum_speedup and lower > 0.0,
        )
    l0_vs_tts_clusters = {
        str(block): np.asarray(
            [
                float(group["l0"]["decode_goodput_tps"])
                / float(group["tts"]["decode_goodput_tps"])
                - 1.0
            ]
        )
        for block, group in by_key.items()
    }
    l0_estimate, l0_lower, l0_upper = bca_mean_interval(
        l0_vs_tts_clusters,
        seed=seed + 2,
    )
    l0_vs_tts = PairwiseSpeedGate(
        numerator_method="l0",
        denominator_method="tts",
        mean_speedup=l0_estimate,
        ci_lower=l0_lower,
        ci_upper=l0_upper,
        no_worse_pass=l0_estimate >= 0.0 and l0_lower >= 0.0,
    )
    return SpeedGate(
        status="UNMEASURED",
        tts=gates["tts"],
        l0=gates["l0"],
        l0_vs_tts=l0_vs_tts,
        gpu_evidence="UNMEASURED",
        evidence_sha256=None,
    )
