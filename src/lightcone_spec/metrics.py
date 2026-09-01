"""Scientific metrics, paired uncertainty, and stage summaries."""

from __future__ import annotations

import gzip
import itertools
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

SAFETY_COUNTERS = (
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "stale_publications",
)


def committed_goodput(committed_tokens: int, duration_seconds: float) -> float:
    if committed_tokens < 0 or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("goodput requires non-negative tokens and positive finite time")
    return committed_tokens / duration_seconds


def per_user_generation_speed(requests: Iterable[dict[str, object]]) -> float | None:
    """Return the mean native decode speed across completed requests."""
    speeds = []
    for request in requests:
        timestamps = request.get("native_token_timestamps_ns")
        if not isinstance(timestamps, list) or len(timestamps) < 2:
            continue
        elapsed_ns = int(timestamps[-1]) - int(timestamps[0])
        if elapsed_ns > 0:
            speeds.append((len(timestamps) - 1) * 1_000_000_000 / elapsed_ns)
    return float(np.mean(speeds)) if speeds else None


def _load_concurrency(load: object) -> int:
    value = str(load or "")
    if value.startswith("closed_loop_c"):
        suffix = value.removeprefix("closed_loop_c")
        return int(suffix) if suffix.isdigit() else 1
    if value.startswith("c") and value[1:].isdigit():
        return int(value[1:])
    return 1


def _attempt_requests(directory: Path) -> list[dict[str, object]]:
    compressed = directory / "requests.jsonl.gz"
    plain = directory / "requests.jsonl"
    if compressed.is_file():
        with gzip.open(compressed, "rt", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if plain.is_file():
        return [
            json.loads(line)
            for line in plain.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return []


def _capacity_cell(config: dict[str, object]) -> bool:
    parameters = config.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    node = str(parameters.get("source_node", config.get("node", "")))
    if node.endswith("-segments"):
        node = node[: -len("-segments")]
    return node in {"E1-common-load", "E3a", "E6-common-load"} or node.startswith(
        "E5"
    )


def derive_feasibility_semantics(
    config: dict[str, object], metrics: dict[str, object]
) -> dict[str, object]:
    """Separate runtime correctness, capacity, and report-only latency SLOs."""

    outcomes = metrics.get("request_outcomes")
    outcomes = outcomes if isinstance(outcomes, dict) else {}
    offered = int(outcomes.get("offered", metrics.get("request_count", 0)) or 0)
    completed = int(outcomes.get("completed", metrics.get("request_count", 0)) or 0)
    incomplete = sum(
        int(outcomes.get(name, 0) or 0)
        for name in ("error", "timed_out", "cancelled", "unfinished")
    )
    scientific_outcome = str(metrics.get("scientific_outcome", "completed"))
    safety_clean = all(int(metrics.get(counter, 0) or 0) == 0 for counter in SAFETY_COUNTERS)
    hard_feasible = (
        metrics.get("status") != "failed"
        and scientific_outcome not in {"blocked", "infeasible", "rejected"}
        and metrics.get("compatible") is not False
        and safety_clean
        and incomplete == 0
        and (offered == 0 or completed == offered)
    )
    capacity = _capacity_cell(config)
    return {
        "hard_feasible": hard_feasible,
        "capacity_feasible": hard_feasible if capacity else "N/A",
        "slo_semantics": "report_only_v2",
    }


def normalize_attempt_semantics(
    config: dict[str, object], metrics: dict[str, object], directory: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Derive concurrency and per-user metrics without rewriting raw attempts."""
    normalized_config = dict(config)
    parameters = dict(normalized_config.get("parameters") or {})
    declared = _load_concurrency(normalized_config.get("load"))
    request_scoped = normalized_config.get("method") in {"tts", "l0_naive"}
    burst_trace = parameters.get("registered_load") == "burstgpt_shape"
    dispatcher = 1 if request_scoped else (256 if burst_trace else declared)
    parameters.update(
        declared_concurrency=declared,
        dispatcher_concurrency=dispatcher,
        effective_load=f"c{dispatcher}",
        metric_semantics="per_request_native_v2",
    )
    normalized_config["parameters"] = parameters

    normalized_metrics = dict(metrics)
    speed = per_user_generation_speed(_attempt_requests(directory))
    normalized_metrics.update(
        declared_concurrency=declared,
        dispatcher_concurrency=dispatcher,
        effective_load=f"c{dispatcher}",
        metric_semantics="per_request_native_v2",
        per_user_generation_speed=speed if speed is not None else "N/A",
    )
    normalized_metrics.update(derive_feasibility_semantics(normalized_config, normalized_metrics))
    return normalized_config, normalized_metrics


def validate_scientific_metrics(metrics: dict[str, object]) -> None:
    required = (
        "committed_tokens",
        "duration_seconds",
        "goodput",
        "peak_hbm_bytes",
        "kv_capacity",
        "itl_p99_ms",
        *SAFETY_COUNTERS,
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(f"metrics are missing {missing}")
    numeric = [
        value
        for value in metrics.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError("metrics contain a non-finite number")
    for counter in SAFETY_COUNTERS:
        value = metrics[counter]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{counter} must be a non-negative integer")
        if value:
            raise RuntimeError(f"scientific safety failure: {counter}={value}")


def benjamini_hochberg(p_values: Iterable[float], alpha: float = 0.05) -> tuple[bool, ...]:
    values = tuple(float(value) for value in p_values)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("p-values must be in [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    cutoff = -1
    for rank, index in enumerate(order, start=1):
        if values[index] <= alpha * rank / len(values):
            cutoff = rank
    decisions = [False] * len(values)
    if cutoff > 0:
        for index in order[:cutoff]:
            decisions[index] = True
    return tuple(decisions)


def paired_bca_interval(
    candidate: Iterable[float],
    baseline: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    left = np.asarray(tuple(candidate), dtype=np.float64)
    right = np.asarray(tuple(baseline), dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 3:
        raise ValueError("paired BCa requires equal one-dimensional samples of size at least three")
    if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("paired samples must be finite")
    differences = left - right
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    boot = np.mean(differences[indices], axis=1)
    less = np.mean(boot < observed)
    z0 = norm.ppf(np.clip(less, 1 / (2 * resamples), 1 - 1 / (2 * resamples)))
    jack = np.asarray([np.mean(np.delete(differences, index)) for index in range(differences.size)])
    centered = np.mean(jack) - jack
    denominator = 6 * np.sum(centered**2) ** 1.5
    acceleration = 0.0 if denominator == 0 else float(np.sum(centered**3) / denominator)
    alpha = (1 - confidence) / 2
    adjusted = []
    for probability in (alpha, 1 - alpha):
        z = norm.ppf(probability)
        adjusted.append(norm.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z))))
    low, high = np.quantile(boot, np.clip(adjusted, 0, 1))
    return observed, float(low), float(high)


def paired_relative_bca_interval(
    candidate: Iterable[float],
    baseline: Iterable[float],
    *,
    resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    left = np.asarray(tuple(candidate), dtype=np.float64)
    right = np.asarray(tuple(baseline), dtype=np.float64)
    if np.any(right <= 0):
        raise ValueError("relative BCa requires positive baselines")
    return paired_bca_interval(
        left / right - 1.0,
        np.zeros_like(right),
        resamples=resamples,
        seed=seed,
    )


def holm_decisions(p_values: Iterable[float], alpha: float = 0.05) -> tuple[bool, ...]:
    values = tuple(float(value) for value in p_values)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("p-values must be in [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    decisions = [False] * len(values)
    active = True
    for rank, index in enumerate(order):
        threshold = alpha / (len(values) - rank)
        if active and values[index] <= threshold:
            decisions[index] = True
        else:
            active = False
    return tuple(decisions)


def block_bootstrap_interval(
    values: Iterable[float], *, resamples: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    rows = np.asarray(tuple(values), dtype=np.float64)
    if rows.size < 3 or np.any(~np.isfinite(rows)):
        raise ValueError("block bootstrap requires at least three finite blocks")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, rows.size, size=(resamples, rows.size))
    draws = np.mean(rows[indexes], axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(np.mean(rows)), float(low), float(high)


def hierarchical_request_interval(
    blocks: dict[int, Iterable[tuple[int, float]]],
    *,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    if len(blocks) < 3:
        raise ValueError("hierarchical bootstrap requires at least three blocks")
    samples = []
    for rows in blocks.values():
        array = np.asarray(tuple(rows), dtype=np.float64)
        if (
            array.ndim != 2
            or array.shape[0] == 0
            or array.shape[1] != 2
            or np.any(~np.isfinite(array))
            or np.any(array[:, 0] < 0)
            or np.any(array[:, 1] <= 0)
        ):
            raise ValueError("request rows require non-negative tokens and positive time")
        samples.append(array)
    rng = np.random.default_rng(seed)
    within = np.empty((resamples, len(samples)), dtype=np.float64)
    points = []
    for block_index, rows in enumerate(samples):
        points.append(float(np.sum(rows[:, 0]) / np.max(rows[:, 1])))
        indexes = rng.integers(0, len(rows), size=(resamples, len(rows)))
        draws = rows[indexes]
        within[:, block_index] = np.sum(draws[:, :, 0], axis=1) / np.max(draws[:, :, 1], axis=1)
    block_indexes = rng.integers(0, len(samples), size=(resamples, len(samples)))
    draws = np.mean(within[np.arange(resamples)[:, None], block_indexes], axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(np.mean(points)), float(low), float(high)


def summarize_attempts(attempt_dirs: Iterable[Path], output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for directory in attempt_dirs:
        metrics_path = directory / "metrics.json"
        config_path = directory / "config.json"
        if not metrics_path.is_file() or not config_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config, metrics = normalize_attempt_semantics(config, metrics, directory)
        rows.append(
            {
                **config,
                **metrics,
                "attempt": directory.name,
                "attempt_dir": str(directory),
                "reducer": "attempt_summary_v2",
            }
        )
    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "summary.csv", index=False)
    parquet = frame.copy()
    for name in parquet.columns:
        if parquet[name].dtype != object:
            continue
        parquet[name] = parquet[name].map(
            lambda value: (
                value
                if value is None or isinstance(value, str)
                else json.dumps(value, sort_keys=True)
            )
        )
    parquet.to_parquet(output_root / "summary.parquet", index=False)
    return frame


def paired_block_statistics(
    rows: Iterable[tuple[dict[str, object], dict[str, object]]],
) -> list[dict[str, object]]:
    axes = (
        "regime",
        "width_panel",
        "topology",
        "cohorts",
        "popularity",
        "traffic",
        "workload",
        "registered_load",
        "effective_load",
    )
    groups: dict[str, dict[str, dict[int, tuple[float, int, dict[str, object]]]]] = {}
    descriptions: dict[str, dict[str, object]] = {}
    for config, metrics in rows:
        block = config.get("block")
        method = config.get("method")
        if not isinstance(block, int) or not isinstance(method, str):
            continue
        parameters = config.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        condition = {name: config.get(name) for name in ("model", "task", "context", "load")}
        condition["comparison_backend"] = parameters.get(
            "comparison_backend", config.get("backend")
        )
        condition.update({name: parameters.get(name) for name in axes})
        key = json.dumps(condition, sort_keys=True)
        descriptions[key] = condition
        source = {
            "job_id": config.get("job_id"),
            "attempt": metrics.get("source_attempt"),
            "attempt_dir": metrics.get("source_attempt_dir"),
        }
        method_rows = groups.setdefault(key, {}).setdefault(method, {})
        if block in method_rows:
            raise ValueError(f"duplicate paired row for {method} block {block}")
        method_rows[block] = (
            float(metrics["goodput"]),
            int(metrics.get("request_count", 0)),
            {**source, "stimulus_id": parameters.get("stimulus_id")},
        )
    comparisons = (
        ("lightcone", "target_only"),
        ("lightcone", "static"),
        ("lightcone", "tts"),
        ("lightcone", "operational_baseline"),
        ("l0_naive", "tts"),
        ("lightcone", "l0_naive"),
        ("onlinespec_ogd", "static"),
        ("onlinespec_opt", "static"),
        ("onlinespec_ens", "static"),
    )
    results: list[dict[str, object]] = []
    for key, methods in groups.items():
        if {"target_only", "static"} <= methods.keys():
            shared = methods["target_only"].keys() & methods["static"].keys()
            methods["operational_baseline"] = {
                block: max(
                    (methods["target_only"][block], methods["static"][block]),
                    key=lambda row: row[0],
                )
                for block in shared
            }
        for candidate_name, baseline_name in comparisons:
            if candidate_name not in methods or baseline_name not in methods:
                continue
            blocks = sorted(methods[candidate_name].keys() & methods[baseline_name].keys())
            if len(blocks) < 3:
                continue
            candidate = [methods[candidate_name][block][0] for block in blocks]
            baseline = [methods[baseline_name][block][0] for block in blocks]
            if any(
                methods[candidate_name][block][2]["stimulus_id"]
                != methods[baseline_name][block][2]["stimulus_id"]
                for block in blocks
            ):
                raise ValueError(f"paired methods received different stimuli for {key}")
            if any(value <= 0 for value in (*candidate, *baseline)):
                continue
            log_ratios = np.log(candidate) - np.log(baseline)
            estimate, low, high = paired_bca_interval(log_ratios, np.zeros_like(log_ratios))
            observed = float(np.mean(log_ratios))
            signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(log_ratios))))
            p_value = float(
                (np.sum(np.mean(signs * log_ratios, axis=1) >= observed) + 1) / (len(signs) + 1)
            )
            results.append(
                {
                    **descriptions[key],
                    "candidate": candidate_name,
                    "baseline": baseline_name,
                    "blocks": blocks,
                    "pairing_key": key,
                    "paired_unit_keys": [
                        json.dumps({**descriptions[key], "block": block}, sort_keys=True)
                        for block in blocks
                    ],
                    "request_counts": {
                        candidate_name: sum(methods[candidate_name][block][1] for block in blocks),
                        baseline_name: sum(methods[baseline_name][block][1] for block in blocks),
                    },
                    "source_attempts": {
                        candidate_name: [methods[candidate_name][block][2] for block in blocks],
                        baseline_name: [methods[baseline_name][block][2] for block in blocks],
                    },
                    "reducer": "paired_log_goodput_bca",
                    "mean_log_ratio": estimate,
                    "relative_effect": math.exp(estimate) - 1.0,
                    "ci95_relative_low": math.exp(low) - 1.0,
                    "ci95_relative_high": math.exp(high) - 1.0,
                    "p_value": p_value,
                }
            )
    for row in results:
        row["holm_reject"] = None
    return results
