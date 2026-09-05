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
from scipy.stats import norm, t

SAFETY_COUNTERS = (
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "stale_publications",
)


def position_acceptance_metrics(rounds: object) -> dict[str, object]:
    """Separate prefix survival from acceptance conditional on reaching a position.

    verify_len counts exposed draft positions, not the target bonus token.
    Missing exposure is not a rejection; unobserved conditional rates are None.
    """
    empty = {
        name: "N/A"
        for name in (
            "position_prefix_survival",
            "position_conditional_survival",
            "position_conditional_acceptance",
            "position_exposure_counts",
            "position_reached_counts",
        )
    }
    if not isinstance(rounds, list) or not rounds:
        return empty
    pairs = []
    for row in rounds:
        if not isinstance(row, dict):
            return empty
        accepted, verified = row.get("accepted_drafts"), row.get("verify_len")
        if not isinstance(accepted, list) or not isinstance(verified, list):
            return empty
        if len(accepted) != len(verified):
            return empty
        for a, v in zip(accepted, verified, strict=True):
            if type(a) is not int or type(v) is not int or not 0 <= a <= v:
                return empty
            pairs.append((a, v))
    if not pairs:
        return empty
    prefix, conditional, exposure, reached = [], [], [], []
    for position in range(1, max(v for _, v in pairs) + 1):
        total = sum(v >= position for _, v in pairs)
        denominator = sum(v >= position and a >= position - 1 for a, v in pairs)
        numerator = sum(v >= position and a >= position for a, v in pairs)
        prefix.append(numerator / total if total else None)
        conditional.append(numerator / denominator if denominator else None)
        exposure.append(total)
        reached.append(denominator)
    return {
        "position_prefix_survival": prefix,
        "position_conditional_survival": conditional,
        "position_conditional_acceptance": conditional,
        "position_exposure_counts": exposure,
        "position_reached_counts": reached,
        "position_metric_semantics": "prefix_and_conditional_v2",
    }


def historical_position_rounds(config: dict, rounds: object) -> object:
    """Read legacy bonus-inclusive telemetry without modifying raw evidence.

    Old DSpark dense and compact counters had different meanings. A gamma+1
    observation identifies the compact run; otherwise do not guess its layout.
    """
    if not isinstance(rounds, list) or not rounds:
        return rounds
    if all(row.get("verify_len_semantics") == "draft_positions_v2" for row in rounds):
        return rounds
    backend = config.get("backend")
    if backend not in {"EAGLE3", "DSPARK"}:
        return rounds
    if backend == "DSPARK" and not any(
        value == 8 for row in rounds for value in row.get("verify_len", [])
    ):
        return None
    return [
        {
            **row,
            "verify_len": [value - 1 for value in row["verify_len"]],
            "verify_len_semantics": "draft_positions_v2",
        }
        for row in rounds
    ]


def mechanism_position_summary(bins: list[dict], requests: list[dict]) -> list[dict]:
    """Within-one-block position diagnostics; requests are not replicate blocks."""
    by_request = {request["request_id"]: request for request in requests}
    if len(by_request) != len(requests):
        raise ValueError("duplicate mechanism request IDs")
    grouped = {}
    observed = set()
    for row in bins:
        rid, start, end = row["request_id"], row["position_start"], row["position_end"]
        if rid not in by_request or (rid, start) in observed:
            raise ValueError("unknown or duplicate mechanism request bin")
        observed.add((rid, start))
        bucket = grouped.setdefault(
            (start, end),
            {
                "position_start": start,
                "position_end": end,
                "request_ids": set(),
                "target_calls": 0,
                "accepted_drafts": 0,
                "committed_tokens": 0,
                "updates_published": 0,
                "exposed_draft_positions": 0,
                "target_entropy_sum": 0.0,
                "draft_top1_ce_sum": 0.0,
                "positions": {},
                "native_itl_ms": [],
                "natural_stops": 0,
            },
        )
        bucket["request_ids"].add(rid)
        for name in (
            "target_calls",
            "accepted_drafts",
            "committed_tokens",
            "updates_published",
            "exposed_draft_positions",
            "target_entropy_sum",
            "draft_top1_ce_sum",
        ):
            bucket[name] += row[name]
        for position, counts in row["positions"].items():
            destination = bucket["positions"].setdefault(
                position, {"exposed": 0, "reached": 0, "accepted": 0}
            )
            for name in destination:
                destination[name] += counts[name]
        stamps = by_request[rid]["native_token_timestamps_ns"]
        if any(right < left for left, right in zip(stamps, stamps[1:])):
            raise ValueError("non-monotone native mechanism timestamps")
        bucket["native_itl_ms"].extend(
            (stamps[index] - stamps[index - 1]) / 1e6
            for index in range(max(start, 1), min(end, len(stamps)))
        )
        bucket["natural_stops"] += int(
            row["request_ending"] == "natural_stop" and start <= len(stamps) - 1 < end
        )
    output = []
    for _, bucket in sorted(grouped.items()):
        count = bucket["exposed_draft_positions"]
        calls = bucket["target_calls"]
        for counts in bucket["positions"].values():
            counts["prefix_survival"] = (
                counts["accepted"] / counts["exposed"] if counts["exposed"] else None
            )
            counts["conditional_acceptance"] = (
                counts["accepted"] / counts["reached"] if counts["reached"] else None
            )
        times = bucket.pop("native_itl_ms")
        bucket.update(
            effective_requests=len(bucket.pop("request_ids")),
            target_entropy=bucket["target_entropy_sum"] / count if count else None,
            draft_top1_ce=bucket["draft_top1_ce_sum"] / count if count else None,
            accepted_drafts_per_target_call=bucket["accepted_drafts"] / calls if calls else None,
            committed_tokens_per_target_call=bucket["committed_tokens"] / calls if calls else None,
            native_itl_samples=len(times),
            native_p50_itl_ms=float(np.quantile(times, 0.5)) if times else None,
            native_p99_itl_ms=float(np.quantile(times, 0.99)) if times else None,
            statistical_unit="within_one_independent_block",
        )
        output.append(bucket)
    return output


def four_block_log_ratio_statistics(
    candidate: dict[int, float],
    baseline: dict[int, float],
) -> dict[str, object]:
    """Four independent paired blocks, never four request partitions."""
    if set(candidate) != set(range(4)) or set(baseline) != set(range(4)):
        raise ValueError("four-block comparison requires exactly blocks 0,1,2,3 on both sides")
    left = np.asarray([candidate[b] for b in range(4)], dtype=float)
    right = np.asarray([baseline[b] for b in range(4)], dtype=float)
    if (
        np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
        or np.any(left <= 0)
        or np.any(right <= 0)
    ):
        raise ValueError("log-ratio comparison requires positive finite block metrics")
    differences = np.log(left) - np.log(right)
    mean = float(np.mean(differences))
    radius = float(t.ppf(0.975, df=3) * np.std(differences, ddof=1) / 2)
    signs = np.asarray(list(itertools.product((-1, 1), repeat=4)))
    p_value = float(np.mean(np.abs(np.mean(signs * differences, axis=1)) >= abs(mean) - 1e-14))
    return {
        "blocks": list(range(4)),
        "candidate": left.tolist(),
        "baseline": right.tolist(),
        "log_ratios": differences.tolist(),
        "geometric_mean_ratio": math.exp(mean),
        "ratio_ci95": [math.exp(mean - radius), math.exp(mean + radius)],
        "ci_method": "paired_block_log_ratio_student_t",
        "degrees_of_freedom": 3,
        "assumption": "independent approximately normal block log ratios",
        "exact_sign_flip_p": p_value,
        "minimum_two_sided_p": 0.125,
        "statistical_unit": "independent_clean_server_paired_block",
    }


def four_block_panel_statistics(
    rows: Iterable[tuple[dict[str, object], dict[str, object]]],
    *,
    metric: str = "goodput",
) -> list[dict[str, object]]:
    """Keep incomplete/negative panels visible without inventing four-block CIs."""
    groups: dict[str, dict[str, dict[int, tuple[dict, dict]]]] = {}
    for config, metrics in rows:
        parameters = config.get("parameters", {})
        if parameters.get("statistical_unit") != "independent_clean_server_paired_block":
            continue
        condition = {
            name: config.get(name) for name in ("node", "model", "backend", "task", "load", "width")
        }
        condition["metric"] = metric
        condition.update(
            {
                name: parameters.get(name)
                for name in (
                    "pairing_key",
                    "topology",
                    "draft_key",
                    "regime",
                    "position_start",
                    "position_end",
                )
            }
        )
        key = json.dumps(condition, sort_keys=True)
        block = config.get("block")
        if type(block) is not int or block not in range(4):
            raise ValueError("four-block evidence has an invalid block index")
        method = str(config["method"])
        values = groups.setdefault(key, {}).setdefault(method, {})
        if block in values:
            raise ValueError(f"duplicate four-block evidence: {key}, {method}, {block}")
        values[block] = (config, metrics)
    results = []
    for key, methods in groups.items():
        for candidate, baseline in (
            ("tts", "static"),
            ("lightcone", "static"),
            ("lightcone", "tts"),
        ):
            left, right = methods.get(candidate, {}), methods.get(baseline, {})
            result = {
                **json.loads(key),
                "candidate_method": candidate,
                "baseline_method": baseline,
                "status": "incomplete_or_infeasible",
                "statistical_unit": "independent_clean_server_paired_block",
                "raw_blocks": {
                    method: [
                        {
                            "block": block,
                            "job_id": config.get("job_id"),
                            "sampling_seed": config["parameters"].get("sampling_seed"),
                            metric: metrics.get(metric),
                            "effective_requests": metrics.get("effective_requests"),
                            "natural_stops": metrics.get("natural_stops"),
                            "hard_feasible": metrics.get("hard_feasible", False),
                            "scientific_outcome": metrics.get("scientific_outcome"),
                            "attempt_dir": metrics.get("source_attempt_dir"),
                        }
                        for block, (config, metrics) in sorted(values.items())
                    ]
                    for method, values in ((candidate, left), (baseline, right))
                },
            }
            if set(left) == set(right) == set(range(4)):
                seeds = []
                for block in range(4):
                    a, b = left[block][0]["parameters"], right[block][0]["parameters"]
                    if a.get("sampling_seed") != b.get("sampling_seed"):
                        raise ValueError("paired four-block methods have different seeds")
                    seeds.append(a.get("sampling_seed"))
                if None in seeds or len(set(seeds)) != 4:
                    raise ValueError("four clean-server blocks require four distinct frozen seeds")
                if all(
                    metrics.get("hard_feasible") is True
                    for values in (left, right)
                    for _, metrics in values.values()
                ):
                    values = [
                        metrics.get(metric)
                        for group in (left, right)
                        for _, metrics in group.values()
                    ]
                    if any(
                        not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0
                        for value in values
                    ):
                        result["status"] = "undefined_log_ratio"
                        results.append(result)
                        continue
                    result.update(
                        four_block_log_ratio_statistics(
                            {b: float(left[b][1][metric]) for b in range(4)},
                            {b: float(right[b][1][metric]) for b in range(4)},
                        )
                    )
                    result["status"] = "measured"
            results.append(result)
    return results


def four_block_mechanism_statistics(rows):
    """Four independent run points per position bin; no request pseudoreplication."""
    expanded, names = [], set()
    for config, metrics in rows:
        for bucket in metrics.get("mechanism_position_summary", []):
            item = {
                **config,
                "parameters": {
                    **config.get("parameters", {}),
                    "position_start": bucket["position_start"],
                    "position_end": bucket["position_end"],
                },
            }
            point = {
                **bucket,
                "hard_feasible": metrics.get("hard_feasible", False),
                "source_attempt_dir": metrics.get("source_attempt_dir"),
            }
            fields = {
                "accepted_drafts_per_target_call",
                "committed_tokens_per_target_call",
                "target_entropy",
                "draft_top1_ce",
                "native_p50_itl_ms",
                "native_p99_itl_ms",
            }
            for position, counts in bucket.get("positions", {}).items():
                for quantity in ("prefix_survival", "conditional_acceptance"):
                    field = f"position_{position}_{quantity}"
                    point[field] = counts.get(quantity)
                    fields.add(field)
            names.update(fields)
            expanded.append((item, point))
    return [
        result
        for name in sorted(names)
        for result in four_block_panel_statistics(expanded, metric=name)
    ]


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
    return node in {"E1-common-load", "E3a", "E6-common-load"} or node.startswith("E5")


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
    if "rounds" in normalized_metrics:
        rounds = historical_position_rounds(normalized_config, normalized_metrics["rounds"])
        normalized_metrics.update(position_acceptance_metrics(rounds))
        if rounds is None:
            normalized_metrics["position_metric_semantics"] = "unidentified_legacy_verify_layout"
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
    evidence = []
    for directory in attempt_dirs:
        metrics_path = directory / "metrics.json"
        config_path = directory / "config.json"
        if not metrics_path.is_file() or not config_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config, metrics = normalize_attempt_semantics(config, metrics, directory)
        evidence.append((config, {**metrics, "source_attempt_dir": str(directory)}))
    return summarize_metric_rows(evidence, output_root)


def summarize_metric_rows(evidence, output_root: Path) -> pd.DataFrame:
    """Write derived tables from the same replacement-filtered rows as inference."""
    four_block_rows = list(evidence)
    rows = []
    for config, metrics in four_block_rows:
        directory = metrics.get("source_attempt_dir")
        rows.append({
            **config, **metrics,
            "attempt": Path(directory).name if directory else None,
            "attempt_dir": directory, "reducer": "logical_cell_summary_v3",
        })
    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "summary.csv", index=False)
    (output_root / "four_block_statistics.json").write_text(
        json.dumps(four_block_panel_statistics(four_block_rows), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_root / "mechanism_four_block_statistics.json").write_text(
        json.dumps(four_block_mechanism_statistics(four_block_rows), indent=2, allow_nan=False),
        encoding="utf-8",
    )
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
        goodput = metrics.get("goodput")
        if (metrics.get("hard_feasible") is False
                or not isinstance(goodput, (int, float)) or not math.isfinite(goodput)
                or goodput <= 0):
            # Infeasible outcomes remain in the raw summary; they cannot enter
            # a log-ratio or manufacture a paired observation.
            continue
        parameters = config.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        if parameters.get("statistical_unit") == "independent_clean_server_paired_block":
            # These panels use the explicit four-block t reducer, not legacy BCa.
            continue
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
