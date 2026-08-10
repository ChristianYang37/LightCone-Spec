"""Paired repetition-block inference and the formal GPU speed gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from lightcone_spec.experiments.data import DFLASH_SAFE_CONTEXT_LIMIT


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
    if any(
        values.size == 0 or not np.isfinite(values).all()
        for values in clusters
    ):
        raise ValueError("cluster effects must be finite and non-empty")
    cluster_means = np.asarray([values.mean() for values in clusters])
    estimate = float(cluster_means.mean())
    randomizer = np.random.default_rng(seed)
    indices = randomizer.integers(
        0, len(clusters), size=(repetitions, len(clusters))
    )
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
        [
            np.delete(cluster_means, index).mean()
            for index in range(len(clusters))
        ]
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
    naive_async: MethodSpeedGate
    l0_vs_tts: PairwiseSpeedGate
    gpu_evidence: str
    evidence_sha256: str | None

    @property
    def passed(self) -> bool:
        return (
            self.gpu_evidence == "MEASURED"
            and self.evidence_sha256 is not None
            and self.tts.passed
            and self.naive_async.passed
            and self.l0_vs_tts.passed
        )


def _validate_coverage(
    rows: list[dict],
) -> tuple[dict[int, dict[str, dict]], dict[int, dict[str, dict]]]:
    required_methods = {"static", "tts", "naive_async"}
    if {str(row["method"]) for row in rows} != required_methods:
        raise ValueError("formal gate requires exactly Static, TTS, and L0")
    filtered = [
        row
        for row in rows
        if str(row.get("region")) == "long_region"
    ]
    by_key: dict[int, dict[str, dict]] = {}
    for row in filtered:
        key = int(row["repetition_block"])
        method_rows = by_key.setdefault(key, {})
        method = str(row["method"])
        if method in method_rows:
            raise ValueError("duplicate paired performance row")
        method_rows[method] = row
    if not by_key or any(
        set(group) != required_methods for group in by_key.values()
    ):
        raise ValueError("paired long-region coverage is incomplete")
    prompt_batches = {
        str(row["prompt_id"])
        for group in by_key.values()
        for row in group.values()
    }
    blocks = set(by_key)
    if len(prompt_batches) != 1 or not next(iter(prompt_batches)).startswith("batch-"):
        raise ValueError("formal gate requires one jointly timed confirmation batch")
    if blocks != set(range(8)):
        raise ValueError("formal gate requires eight independent blocks")
    concurrencies = {
        int(row["concurrency"])
        for group in by_key.values()
        for row in group.values()
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
        raise ValueError(
            "long-region bounds must be shared generated-token positions"
        )
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
    if (gpu_evidence == "MEASURED") != (evidence_sha256 is not None):
        raise ValueError("MEASURED evidence requires exactly one attestation")
    by_key, full_by_block = _validate_coverage(rows)
    gates: dict[str, MethodSpeedGate] = {}
    for method in ("tts", "naive_async"):
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
            {
                cluster: np.asarray(values)
                for cluster, values in clusters.items()
            },
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
                float(group["naive_async"]["decode_goodput_tps"])
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
        numerator_method="naive_async",
        denominator_method="tts",
        mean_speedup=l0_estimate,
        ci_lower=l0_lower,
        ci_upper=l0_upper,
        no_worse_pass=l0_estimate >= 0.0 and l0_lower >= 0.0,
    )
    statistical_pass = (
        all(gate.passed for gate in gates.values()) and l0_vs_tts.passed
    )
    if gpu_evidence == "UNMEASURED":
        status = "UNMEASURED"
    elif statistical_pass:
        status = "PASS"
    else:
        status = "BLOCKED"
    return SpeedGate(
        status=status,
        tts=gates["tts"],
        naive_async=gates["naive_async"],
        l0_vs_tts=l0_vs_tts,
        gpu_evidence=gpu_evidence,
        evidence_sha256=evidence_sha256,
    )
