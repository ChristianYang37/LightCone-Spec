"""Main/side table assembly (spec 14, 16.4).

- request-reset and streaming tables are strictly separate (never merged
  averages, never shared bootstrap clusters, never one headline speedup);
- default speedups are against Static on the same model / hardware / requests /
  sampling; an explicitly selected alternative baseline is recorded verbatim;
- accepted_drafts and committed_per_verify are reported side by side.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from lightcone_spec.config.schema import (
    METHOD_REPORT_NAMES,
    MODEL_PAIRS,
    canonical_weight_update_mode,
)
from lightcone_spec.statistics.bootstrap import (
    BootstrapResult,
    cluster_bca,
    paired_difference_by_cluster,
)


# Historical P5 manifests used stride 4 before derived tables carried the
# field explicitly.  New analysis reads the authoritative run manifest; this
# fallback applies only to legacy/direct DataFrames with no stride column.
DEFAULT_P5_UPDATE_STRIDE = 4
P5_IDENTITY_COLUMNS = (
    "model_pair",
    "weight_update_mode",
    "update_stride",
    "dataset",
    "lifecycle",
    "offered_concurrency",
)


def unit_key(row: dict) -> str:
    """prompt-cluster :: seed pairing key; clusters are base prompt ids
    (all seeds of one prompt share a cluster)."""
    return f"{row['prompt_id_hash']}::seed{row['seed']}"


_REPEAT_SUFFIX = re.compile(r":repeat-(\d+)$")
_RUNTIME_PROMPT = re.compile(
    r"^lightcone-g([0-9a-f]{64})(?:-p([0-9a-f]{64}))?-"
)


def _aggregate_run_throughput(group: pd.DataFrame, wall_column: str) -> float:
    """Aggregate one run without treating its request rows as replicates."""
    if group.empty:
        return float("nan")
    if (group["offered_concurrency"] == 1).all():
        return float(
            group["output_tokens"].sum()
            / max(group[wall_column].sum(), 1e-9)
        )
    # Concurrent SGLang artifacts copy the run-level goodput to every
    # request row.  Selecting one value preserves the true sample size.
    metric = "decode_tps" if wall_column == "decode_wall_s" else "e2e_tps"
    return float(group[metric].iloc[0])


def _run_throughputs(group: pd.DataFrame, wall_column: str) -> dict[str, float]:
    """Recover independent run/repetition throughput values."""
    labels = group["prompt_id_hash"].astype(str).str.extract(
        _REPEAT_SUFFIX, expand=False
    )
    if labels.isna().all():
        return {"single": _aggregate_run_throughput(group, wall_column)}
    if labels.isna().any():
        raise ValueError("throughput rows mix repeated and unrepeated prompt ids")
    output = {}
    for repetition, repeated in group.assign(_repeat=labels).groupby("_repeat"):
        output[str(repetition)] = _aggregate_run_throughput(repeated, wall_column)
    return output


def method_table(
    summaries: pd.DataFrame,
    lifecycle: str,
    baseline_method: str = "static",
    b: int = 5000,
) -> pd.DataFrame:
    """One table per lifecycle, paired against ``baseline_method``.

    ``speedup_vs_static`` remains an exact compatibility alias when the
    selected baseline is Static.  Non-Static analyses expose only the
    semantically correct generic field.
    """
    df = summaries[
        (summaries["lifecycle"] == lifecycle)
        & (summaries["status"] == "complete_valid")
    ].copy()
    if df.empty:
        return pd.DataFrame()
    rows = []
    base = df[df["method"] == baseline_method]
    base_runs = _run_throughputs(base, "decode_wall_s")
    for method, group in df.groupby("method"):
        run_tps = _run_throughputs(group, "decode_wall_s")
        speedups = {
            k: run_tps[k] / base_runs[k]
            for k in run_tps
            if k in base_runs and base_runs[k] > 0
        }
        if speedups:
            vals = np.asarray(list(speedups.values()))
            speedup_mean = float(vals.mean())
            speedup_n_clusters = len(speedups)
            if speedup_n_clusters >= 2:
                boot = cluster_bca(
                    vals, np.asarray(list(speedups)), np.mean, b=b
                )
                lo, hi = boot.ci_low, boot.ci_high
            else:
                # A single run supports a point estimate, never a confidence
                # interval.  NaNs make the missing replication explicit.
                lo = hi = float("nan")
        else:
            speedup_mean = lo = hi = float("nan")
            speedup_n_clusters = 0
        run_e2e = _run_throughputs(group, "e2e_wall_s")
        record = {
            "method": method,
            "report_name": METHOD_REPORT_NAMES.get(method, method),
            "lifecycle": lifecycle,
            "baseline_method": baseline_method,
            "n_requests": len(group),
            "decode_tps_mean": float(np.mean(list(run_tps.values()))),
            "e2e_tps_mean": float(np.mean(list(run_e2e.values()))),
            "speedup_vs_baseline": speedup_mean,
            "speedup_ci_low": lo,
            "speedup_ci_high": hi,
            "speedup_n_clusters": speedup_n_clusters,
            "mean_accepted_drafts": float(group["mean_accepted_drafts"].mean()),
            "mean_committed_per_verify": float(
                group["mean_committed_per_verify"].mean()
            ),
            "target_calls_per_output_token": float(
                group["target_calls_per_output_token"].mean()
            ),
            "quality_mean": float(group["quality_value"].dropna().mean())
            if group["quality_value"].notna().any()
            else float("nan"),
            "p95_round_ms": float(group["p95_round_ms"].mean()),
            "p50_itl_ms": float(group["p50_itl_ms"].mean()),
            "p95_itl_ms": float(group["p95_itl_ms"].mean()),
            "p99_itl_ms": float(group["p99_itl_ms"].mean()),
            "version_mismatch_total": int(group["version_mismatch_count"].sum()),
        }
        if baseline_method == "static":
            record["speedup_vs_static"] = record["speedup_vs_baseline"]
        rows.append(record)
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def paired_method_delta(
    summaries: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = "decode_tps",
    lifecycle: str = "request",
    b: int = 5000,
):
    df = summaries[
        (summaries["lifecycle"] == lifecycle)
        & (summaries["status"] == "complete_valid")
    ]
    if metric in {"decode_tps", "goodput_tps"}:
        a_vals = _run_throughputs(
            df[df["method"] == method_a], "decode_wall_s"
        )
        b_vals = _run_throughputs(
            df[df["method"] == method_b], "decode_wall_s"
        )
        paired = sorted(set(a_vals) & set(b_vals))
        if not paired:
            raise ValueError(f"no paired runs between {method_a} and {method_b}")
        diffs = np.asarray([a_vals[key] - b_vals[key] for key in paired])
        if len(paired) == 1:
            return BootstrapResult(
                float(diffs.mean()),
                float("nan"),
                float("nan"),
                0,
                1,
                method="insufficient_runs",
            )
        return cluster_bca(diffs, np.asarray(paired), np.mean, b=b)

    a_vals = {
        unit_key(r): r[metric]
        for _, r in df[df["method"] == method_a].iterrows()
    }
    b_vals = {
        unit_key(r): r[metric]
        for _, r in df[df["method"] == method_b].iterrows()
    }
    diffs, clusters = paired_difference_by_cluster(a_vals, b_vals)
    if len(diffs) == 0:
        raise ValueError(f"no paired units between {method_a} and {method_b}")
    return cluster_bca(diffs, clusters, np.mean, b=b)


def select_load_profiles(
    summaries: pd.DataFrame,
    *,
    itl_slo_ms: float,
    saturation_gain_threshold: float = 0.03,
) -> list[dict]:
    """Select throughput/SLO profiles and reject an unbounded load search.

    A throughput winner at the largest tested concurrency is not evidence of
    saturation when the final geometric step still adds material goodput.  In
    that case the result remains useful, but it is explicitly marked for a
    higher-concurrency follow-up instead of being presented as the MFU ceiling.
    """
    if not 0.0 <= float(saturation_gain_threshold) < 1.0:
        raise ValueError("saturation_gain_threshold must be in [0, 1)")
    valid = summaries[summaries["status"] == "complete_valid"].copy()
    required = {"offered_concurrency", "p99_itl_ms", "decode_tps"}
    if valid.empty or not required.issubset(valid.columns):
        return []
    group_keys = ["model_pair_id", "method", "dataset", "lifecycle"]
    if "weight_update_mode" in valid.columns:
        # A mode is part of the algorithm identity.  Pooling LoRA/full/residual
        # can select a concurrency that no individual configuration achieves.
        group_keys.insert(2, "weight_update_mode")
    if "update_stride" in valid.columns:
        # P5 stride screens share every other algorithm label.  Treating
        # stride as a load-profile identity prevents a synthetic winner made
        # by averaging several update schedules.
        insert_at = group_keys.index("dataset")
        group_keys.insert(insert_at, "update_stride")
    output: list[dict] = []
    for key, group in valid.groupby(group_keys, dropna=False):
        aggregations = {
            "goodput_tps": ("decode_tps", "mean"),
            "p99_itl_ms": ("p99_itl_ms", "mean"),
            "requests": ("request_id", "count"),
        }
        for name, reducer in (
            ("estimated_mfu", "mean"),
            ("decode_batch_fill_ratio", "mean"),
            ("nvml_gpu_utilization_mean", "mean"),
            ("kv_retracted_requests", "max"),
            ("peak_queue_requests", "max"),
            ("peak_hbm_bytes", "max"),
        ):
            if name in group:
                aggregations[name] = (name, reducer)
        candidates = (
            group.groupby("offered_concurrency", as_index=False)
            .agg(**aggregations)
            .sort_values("offered_concurrency")
        )
        if candidates.empty:
            continue
        max_concurrency = int(candidates["offered_concurrency"].max())
        safe_candidates = candidates
        if "kv_retracted_requests" in candidates:
            safe_candidates = candidates[
                candidates["kv_retracted_requests"] == 0
            ]
        capacity_limited = bool(
            safe_candidates.empty
            or int(safe_candidates["offered_concurrency"].max())
            < max_concurrency
        )
        selection_candidates = (
            safe_candidates if not safe_candidates.empty else candidates
        )
        throughput = selection_candidates.sort_values(
            ["goodput_tps", "offered_concurrency"],
            ascending=[False, True],
        ).iloc[0]
        at_search_boundary = (
            int(throughput["offered_concurrency"]) == max_concurrency
        )
        last_step_gain = None
        if len(selection_candidates) >= 2:
            previous, boundary = selection_candidates.iloc[-2], selection_candidates.iloc[-1]
            previous_goodput = float(previous["goodput_tps"])
            last_step_gain = (
                float(boundary["goodput_tps"]) - previous_goodput
            ) / max(previous_goodput, 1e-12)
        saturation_confirmed = bool(
            len(selection_candidates) >= 2
            and not capacity_limited
            and (
                not at_search_boundary
                or float(last_step_gain) <= float(saturation_gain_threshold)
            )
        )
        recommended_next = (
            max_concurrency * 2
            if not saturation_confirmed and not capacity_limited
            else None
        )
        within_slo = selection_candidates[
            selection_candidates["p99_itl_ms"] <= itl_slo_ms
        ]
        latency = (
            within_slo.sort_values(
                ["goodput_tps", "offered_concurrency"],
                ascending=[False, True],
            ).iloc[0]
            if not within_slo.empty
            else selection_candidates.sort_values(
                ["p99_itl_ms", "offered_concurrency"],
                ascending=[True, True],
            ).iloc[0]
        )
        identity = dict(zip(group_keys, key))
        for profile, row, slo_met in (
            ("throughput", throughput, throughput["p99_itl_ms"] <= itl_slo_ms),
            ("latency_slo", latency, latency["p99_itl_ms"] <= itl_slo_ms),
        ):
            output.append(
                {
                    **identity,
                    "profile": profile,
                    "selected_concurrency": int(row["offered_concurrency"]),
                    "goodput_tps": float(row["goodput_tps"]),
                    "p99_itl_ms": float(row["p99_itl_ms"]),
                    "itl_slo_ms": float(itl_slo_ms),
                    "slo_met": bool(slo_met),
                    "requests": int(row["requests"]),
                    "max_tested_concurrency": max_concurrency,
                    "throughput_at_search_boundary": at_search_boundary,
                    "last_step_goodput_gain_fraction": last_step_gain,
                    "saturation_gain_threshold": float(
                        saturation_gain_threshold
                    ),
                    "saturation_confirmed": saturation_confirmed,
                    "capacity_limited": capacity_limited,
                    "recommended_next_concurrency": recommended_next,
                    "estimated_mfu": (
                        None
                        if "estimated_mfu" not in row or pd.isna(row["estimated_mfu"])
                        else float(row["estimated_mfu"])
                    ),
                    "decode_batch_fill_ratio": (
                        None
                        if "decode_batch_fill_ratio" not in row
                        or pd.isna(row["decode_batch_fill_ratio"])
                        else float(row["decode_batch_fill_ratio"])
                    ),
                    "nvml_gpu_utilization_mean": (
                        None
                        if "nvml_gpu_utilization_mean" not in row
                        or pd.isna(row["nvml_gpu_utilization_mean"])
                        else float(row["nvml_gpu_utilization_mean"])
                    ),
                    "kv_retracted_requests": int(
                        row.get("kv_retracted_requests", 0) or 0
                    ),
                }
            )
    return output


_P5_STATIC_SCOPE_COLUMNS = (
    "model_pair",
    "dataset",
    "lifecycle",
    "offered_concurrency",
    "context_length",
    "trajectory_kind",
)


def _expand_static_identities(
    frame: pd.DataFrame, identity_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Copy one semantic Static sample into each observed adaptive identity."""

    required = {"model_pair", "method", *identity_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"P5 static expansion lacks columns: {missing}")
    scope_columns = [
        name for name in _P5_STATIC_SCOPE_COLUMNS if name in frame.columns
    ]
    expanded = []
    for _, scope_rows in frame.groupby(scope_columns, dropna=False):
        adaptive = scope_rows[scope_rows["method"] != "static"]
        static_rows = scope_rows[scope_rows["method"] == "static"]
        identities = adaptive[list(identity_columns)].drop_duplicates()
        expanded.append(adaptive)
        if identities.empty or static_rows.empty:
            if not static_rows.empty:
                expanded.append(static_rows)
            continue

        # Re-expanding an already labelled baseline must be idempotent.  Drop
        # only rows that differ solely in the semantically irrelevant Static
        # labels; prompt/repetition/run evidence remains distinct.
        evidence_columns = [
            name for name in static_rows.columns if name not in identity_columns
        ]
        static_rows = static_rows.drop_duplicates(subset=evidence_columns)
        for identity in identities.itertuples(index=False, name=None):
            copy = static_rows.copy()
            for name, value in zip(identity_columns, identity):
                copy[name] = value
            expanded.append(copy)
    return (
        pd.concat(expanded, ignore_index=True)
        if expanded
        else frame.copy().reset_index(drop=True)
    )


def expand_static_weight_update_modes(frame: pd.DataFrame) -> pd.DataFrame:
    """Label one mode-agnostic Static measurement for each adapted mode.

    Static does not load adaptation parameters, so run-manifest deliberately
    keeps its unit unchanged under ``--weight-update-mode``.  P5 pairing does
    include the mode label, however.  Expand only within one model pair and
    retain the original Static label when no adapted mode is present.  The
    same helper is used for algorithmic rounds and performance summaries so
    acceptance and throughput cannot disagree about their baseline identity.
    """
    return _expand_static_identities(frame, ("weight_update_mode",))


def expand_static_p5_identities(frame: pd.DataFrame) -> pd.DataFrame:
    """Pair Static once with every observed ``(mode, update_stride)``.

    Expanding the tuple jointly is important: independent mode and stride
    expansion would create an unobserved Cartesian product and could count the
    same Static prompt more than once in paired bootstrap samples.
    """

    return _expand_static_identities(
        frame, ("weight_update_mode", "update_stride")
    )


def _p5_rounds(
    rounds: pd.DataFrame, *, expand_static: bool = True
) -> pd.DataFrame:
    """Normalize raw P5 JSONL rows without changing the P0--P4 schema."""
    df = rounds.copy()
    if "prefix_len_before" not in df:
        df["prefix_len_before"] = df.get("prefix_pos_before", 0)
    if "verify_len" not in df:
        df["verify_len"] = df["draft_tokens"] + 1
    if "batch_size" not in df:
        df["batch_size"] = 1
    if "offered_concurrency" not in df:
        df["offered_concurrency"] = 1
    if "update_stride" not in df:
        # Directly consumed historical tables predate this column.  The
        # historical P5 suite used stride 4; keeping it numeric makes the
        # compatibility identity visible in every output.
        df["update_stride"] = DEFAULT_P5_UPDATE_STRIDE
    stride = pd.to_numeric(df["update_stride"], errors="coerce")
    if stride.isna().any() or (stride < 1).any() or (stride % 1 != 0).any():
        raise ValueError("P5 update_stride must contain positive integers")
    df["update_stride"] = stride.astype(int)
    if "context_length" not in df:
        raise ValueError("P5 rounds require their manifest context_length")
    if "model_pair" not in df:
        # Old single-pair artifacts predate this identity column.  Keeping a
        # visible sentinel preserves readability without allowing a new
        # cross-backend analysis to merge real pairs accidentally.
        df["model_pair"] = "unspecified"
    if "weight_update_mode" not in df:
        df["weight_update_mode"] = df.get(
            "trainable_scope", "output_residual"
        )
    df["weight_update_mode"] = df["weight_update_mode"].map(
        canonical_weight_update_mode
    )

    if expand_static:
        df = expand_static_p5_identities(df)

    def prompt_cluster(request_id: str) -> str:
        match = _RUNTIME_PROMPT.match(str(request_id))
        return (
            (match.group(2) or match.group(1))
            if match
            else str(request_id).split(":repeat-", 1)[0]
        )

    if "prompt_cluster" not in df:
        df["prompt_cluster"] = df["request_id"].map(prompt_cluster)
    if "round_count" not in df:
        df["round_count"] = 1
    count = df["round_count"].astype(float).clip(lower=1.0)
    if "accepted_sum" not in df:
        df["accepted_sum"] = df["accepted_drafts"].astype(float)
    if "committed_sum" not in df:
        df["committed_sum"] = df["committed_per_verify"].astype(float)
    verified = (df["verify_len"].astype(float) - 1.0).clip(lower=0.0)
    if "verified_sum" not in df:
        df["verified_sum"] = verified
    if "waste_sum" not in df:
        df["waste_sum"] = (verified - df["accepted_drafts"]).clip(lower=0.0)
    if "target_calls_sum" not in df:
        df["target_calls_sum"] = df["target_calls"].astype(float)
    if "semantic_round_count" not in df:
        censored = df.get(
            "algorithmic_censored", pd.Series(False, index=df.index)
        ).astype(bool)
        semantic = (~censored).astype(float)
        df["semantic_round_count"] = count * semantic
        df["semantic_accepted_sum"] = df["accepted_sum"] * semantic
        df["semantic_committed_sum"] = df["committed_sum"] * semantic
        df["semantic_verified_sum"] = df["verified_sum"] * semantic
        df["semantic_waste_sum"] = df["waste_sum"] * semantic
        df["semantic_target_calls_sum"] = df["target_calls_sum"] * semantic
        df["algorithmic_censored_count"] = count * censored.astype(float)
    for name in ("draft_cuda_us", "verify_cuda_us", "accept_cuda_us", "batch_size"):
        total = f"{name}_sum"
        if total not in df:
            df[total] = df[name].astype(float)
    if "signal_prep_cuda_us_sum" not in df:
        if "signal_prep_cuda_us" in df:
            signal = pd.to_numeric(df["signal_prep_cuda_us"], errors="coerce")
            df["signal_prep_cuda_us_sum"] = signal.fillna(0.0) * count
            df["signal_prep_timed_rounds"] = signal.notna().astype(float) * count
            df["signal_prep_unknown_rounds"] = signal.isna().astype(float) * count
        else:
            df["signal_prep_cuda_us_sum"] = 0.0
            df["signal_prep_timed_rounds"] = 0.0
            df["signal_prep_unknown_rounds"] = count
    if "signal_prep_timed_rounds" not in df:
        df["signal_prep_timed_rounds"] = 0.0
    if "signal_prep_unknown_rounds" not in df:
        df["signal_prep_unknown_rounds"] = count - df[
            "signal_prep_timed_rounds"
        ].astype(float)
    for name in (
        "side_queue_cuda_us",
        "candidate_cuda_us",
        "backward_cuda_us",
        "optimizer_cuda_us",
        "controller_cuda_us",
        "publish_cuda_us",
        "barrier_wait_cpu_us",
    ):
        total = f"{name}_sum"
        if total not in df:
            df[total] = 0.0
    if "update_count" not in df:
        df["update_count"] = 0
    if "update_cost_evidence_complete" not in df:
        # Historical round-only inputs do not prove that absent update work
        # was zero.  Preserve that uncertainty in the derived cost table.
        df["update_cost_evidence_complete"] = False
    if "batch_reciprocal_sum" not in df:
        # Raw telemetry contributes one row per request and scheduler step.
        # For historical request-aggregated rows this is the best available
        # reconstruction; new artifacts carry the exact reciprocal sum.
        batch = df["batch_size"].astype(float).clip(lower=1.0)
        df["batch_reciprocal_sum"] = count / batch
    if "version_mismatch_count" not in df:
        df["version_mismatch_count"] = (~df["version_canary_ok"]).astype(int)
    if "observed_prefix_min" not in df:
        df["observed_prefix_min"] = df["prefix_len_before"]
    if "observed_prefix_max" not in df:
        df["observed_prefix_max"] = df["prefix_len_before"]
    df["accepted_drafts"] = df["accepted_sum"] / count
    df["committed_per_verify"] = df["committed_sum"] / count
    df["verified_drafts"] = df["verified_sum"] / count
    return df


def _boot(values, clusters, *, b: int) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    if not len(values):
        return (float("nan"),) * 3 + (0,)
    result = cluster_bca(values, clusters, np.mean, b=b)
    return result.estimate, result.ci_low, result.ci_high, result.n_clusters


def _prompt_acceptance(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            [
                "method",
                *P5_IDENTITY_COLUMNS,
                "context_length",
                "prompt_cluster",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            accepted_sum=("semantic_accepted_sum", "sum"),
            round_count=("semantic_round_count", "sum"),
        )
    )
    grouped["acceptance"] = grouped["accepted_sum"] / grouped[
        "round_count"
    ].replace(0, np.nan)
    return grouped


def p5_prompt_acceptance_table(rounds: pd.DataFrame) -> pd.DataFrame:
    """Expose the prompt-level P5 statistic used by every paired bootstrap.

    This is a derived, compact evidence layer rather than raw hot-path
    telemetry.  Keeping it public lets receipt-bound comparisons reuse the
    exact same prompt grouping as the headline table without reverse
    engineering prompt values from already aggregated confidence intervals.
    """

    # Preserve the physical Static stride in this attested evidence layer.
    # Headline tables expand Static only when evaluating a declared adaptive
    # identity; a cross-stride comparator must instead be able to prove that
    # it selected the original stride-1 baseline explicitly.
    normalized = _p5_rounds(rounds, expand_static=False)
    table = _prompt_acceptance(normalized)
    if "benchmark_repetitions" in normalized:
        keys = [
            "method",
            *P5_IDENTITY_COLUMNS,
            "context_length",
            "prompt_cluster",
            "seed",
        ]
        repetitions = normalized.assign(
            benchmark_repetitions=pd.to_numeric(
                normalized["benchmark_repetitions"], errors="coerce"
            )
        ).groupby(keys, as_index=False, dropna=False).agg(
            benchmark_repetitions=("benchmark_repetitions", "min"),
            benchmark_repetitions_max=("benchmark_repetitions", "max"),
        )
        if repetitions[
            ["benchmark_repetitions", "benchmark_repetitions_max"]
        ].isna().any().any() or not (
            repetitions["benchmark_repetitions"]
            == repetitions["benchmark_repetitions_max"]
        ).all():
            raise ValueError("P5 prompt group has conflicting repetitions")
        table = table.merge(
            repetitions.drop(columns="benchmark_repetitions_max"),
            on=keys,
            how="left",
            validate="one_to_one",
        )
    return table.sort_values(
        [
            "method",
            *P5_IDENTITY_COLUMNS,
            "context_length",
            "prompt_cluster",
            "seed",
        ]
    ).reset_index(drop=True)


def paired_cross_stride_acceptance_table(
    prompt_acceptance: pd.DataFrame,
    *,
    candidate_method: str,
    candidate_stride: int,
    baseline_method: str,
    baseline_stride: int,
    b: int = 5000,
) -> pd.DataFrame:
    """Prompt-paired BCa acceptance gain across two explicit stride cells.

    ``update_stride`` is intentionally removed from the pairing identity only
    after both selected values have been fixed.  Every other P5 identity and
    every prompt/seed cell must match exactly; missing evidence fails closed
    instead of being silently intersected.
    """

    for name, value in (
        ("candidate_stride", candidate_stride),
        ("baseline_stride", baseline_stride),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer")
        if int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(b, bool) or not isinstance(b, int) or b < 1:
        raise ValueError("b must be a positive integer")
    if (candidate_method, int(candidate_stride)) == (
        baseline_method,
        int(baseline_stride),
    ):
        raise ValueError("candidate and baseline identities must differ")

    cross_identity = tuple(
        name for name in P5_IDENTITY_COLUMNS if name != "update_stride"
    )
    required = {
        "method",
        *P5_IDENTITY_COLUMNS,
        "context_length",
        "prompt_cluster",
        "seed",
        "acceptance",
    }
    missing = sorted(required - set(prompt_acceptance.columns))
    if missing:
        raise ValueError(
            "P5 prompt acceptance table lacks required columns: "
            + ", ".join(missing)
        )
    frame = prompt_acceptance.copy()
    stride = pd.to_numeric(frame["update_stride"], errors="coerce")
    if stride.isna().any() or (stride < 1).any() or (stride % 1 != 0).any():
        raise ValueError("P5 update_stride must contain positive integers")
    frame["update_stride"] = stride.astype(int)
    frame["acceptance"] = pd.to_numeric(frame["acceptance"], errors="coerce")

    candidate = frame[
        (frame["method"] == candidate_method)
        & (frame["update_stride"] == int(candidate_stride))
    ].copy()
    baseline = frame[
        (frame["method"] == baseline_method)
        & (frame["update_stride"] == int(baseline_stride))
    ].copy()
    if candidate.empty or baseline.empty:
        raise ValueError(
            "selected candidate/baseline stride has no prompt-level evidence"
        )
    evidence_columns = [
        *cross_identity,
        "context_length",
        "prompt_cluster",
        "seed",
        "acceptance",
    ]
    if candidate[evidence_columns].isna().any().any() or baseline[
        evidence_columns
    ].isna().any().any():
        raise ValueError("cross-stride prompt evidence contains null values")

    group_columns = [*cross_identity, "context_length"]
    prompt_columns = [*group_columns, "prompt_cluster", "seed"]
    if candidate.duplicated(prompt_columns).any():
        raise ValueError("candidate prompt/seed evidence is duplicated")
    if baseline.duplicated(prompt_columns).any():
        raise ValueError("baseline prompt/seed evidence is duplicated")

    def grouped(frame: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
        return {
            key if isinstance(key, tuple) else (key,): group
            for key, group in frame.groupby(group_columns, dropna=False)
        }

    candidate_groups = grouped(candidate)
    baseline_groups = grouped(baseline)
    if set(candidate_groups) != set(baseline_groups):
        missing_baseline = sorted(set(candidate_groups) - set(baseline_groups))
        missing_candidate = sorted(set(baseline_groups) - set(candidate_groups))
        raise ValueError(
            "cross-stride identity coverage differs: "
            f"missing_baseline={missing_baseline}, "
            f"missing_candidate={missing_candidate}"
        )

    rows = []
    for identity in sorted(candidate_groups):
        left = candidate_groups[identity]
        right = baseline_groups[identity]
        left_values = {
            (str(row.prompt_cluster), int(row.seed)): float(row.acceptance)
            for row in left.itertuples()
        }
        right_values = {
            (str(row.prompt_cluster), int(row.seed)): float(row.acceptance)
            for row in right.itertuples()
        }
        if set(left_values) != set(right_values):
            raise ValueError(
                "cross-stride prompt/seed coverage differs for identity "
                f"{identity}"
            )
        paired = sorted(left_values)
        gains = np.asarray(
            [left_values[key] - right_values[key] for key in paired],
            dtype=np.float64,
        )
        clusters = np.asarray([key[0] for key in paired])
        result = cluster_bca(gains, clusters, np.mean, b=b)
        rows.append(
            {
                "candidate_method": candidate_method,
                "candidate_update_stride": int(candidate_stride),
                "baseline_method": baseline_method,
                "baseline_update_stride": int(baseline_stride),
                **dict(zip(group_columns, identity)),
                "candidate_acceptance": float(
                    np.mean([left_values[key] for key in paired])
                ),
                "baseline_acceptance": float(
                    np.mean([right_values[key] for key in paired])
                ),
                "acceptance_gain": result.estimate,
                "acceptance_gain_ci_low": result.ci_low,
                "acceptance_gain_ci_high": result.ci_high,
                "paired_prompt_seed_cells": len(paired),
                "paired_prompt_clusters": result.n_clusters,
                "bootstrap_replicates": result.b,
                "bootstrap_method": result.method,
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _p5_report_name(method: str, model_pair: str) -> str:
    pair = MODEL_PAIRS.get(str(model_pair))
    if pair is None:
        return METHOD_REPORT_NAMES.get(method, method)
    algorithm = {
        "DSPARK": "DSpark",
        "DFLASH": "DFlash",
        "EAGLE": "EAGLE",
        "EAGLE3": "EAGLE3",
    }.get(pair["speculative_algorithm"], pair["speculative_algorithm"])
    if method == "static":
        return f"{algorithm}-Static"
    if method == "tts":
        return f"TTS-{algorithm}"
    return f"{METHOD_REPORT_NAMES.get(method, method)}-{algorithm}"


def long_context_acceptance_table(
    rounds: pd.DataFrame,
    *,
    baseline_method: str = "static",
    long_context_min: int = 4096,
    b: int = 5000,
) -> pd.DataFrame:
    """P5 survival-weighted acceptance, paired gain, LCAG and onset.

    ``mean(accepted_drafts)`` equals ``sum_k P(K >= k)`` directly, so the
    headline cannot be improved merely by asking the target to verify fewer
    draft positions.  The ordinary accepted/verified ratio remains auxiliary.
    """
    df = _p5_rounds(rounds)
    keys = [*P5_IDENTITY_COLUMNS, "context_length"]
    prompt = _prompt_acceptance(df)
    prompt_lookup = {}
    for key, group in prompt.groupby(["method", *keys], dropna=False):
        prompt_lookup[key] = {
            (str(row.prompt_cluster), int(row.seed)): float(row.acceptance)
            for row in group.itertuples()
        }

    gamma = int(df["draft_tokens"].max()) if len(df) else 0
    aggregated_censored = (df["round_count"] > 1) & (
        df["algorithmic_censored_count"] > 0
    )
    if aggregated_censored.any():
        missing_survival = [
            f"semantic_survival_count_k{k}"
            for k in range(1, gamma + 1)
            if f"semantic_survival_count_k{k}" not in df
        ]
        if missing_survival:
            raise ValueError(
                "censored aggregate lacks semantic survival counts: "
                + ", ".join(missing_survival)
            )
    rows = []
    for (method, *identity), group in df.groupby(
        ["method", *keys], dropna=False
    ):
        mask = prompt["method"] == method
        for name, value in zip(keys, identity):
            mask &= prompt[name] == value
        pgroup = prompt[mask]
        estimate, lo, hi, n_clusters = _boot(
            pgroup["acceptance"], pgroup["prompt_cluster"], b=b
        )
        base = prompt_lookup.get((baseline_method, *identity), {})
        current = prompt_lookup.get((method, *identity), {})
        paired = sorted(set(base) & set(current))
        gain = [current[key] - base[key] for key in paired]
        gain_clusters = [key[0] for key in paired]
        gain_est, gain_lo, gain_hi, gain_n = _boot(gain, gain_clusters, b=b)
        physical_rounds = float(group["round_count"].sum())
        rounds = float(group["semantic_round_count"].sum())
        accepted_sum = float(group["semantic_accepted_sum"].sum())
        verified_sum = float(group["semantic_verified_sum"].sum())
        scheduler_steps = float(group["batch_reciprocal_sum"].sum())
        request_weighted_batch = float(group["batch_size_sum"].sum()) / max(
            physical_rounds, 1.0
        )
        signal_unknown_rounds = int(group["signal_prep_unknown_rounds"].sum())
        signal_cost_complete = signal_unknown_rounds == 0
        update_cost_complete = bool(
            group["update_cost_evidence_complete"].astype(bool).all()
        )
        adaptation_cost_complete = signal_cost_complete and update_cost_complete

        def update_component(name: str) -> float:
            if not update_cost_complete:
                return float("nan")
            return float(group[f"{name}_sum"].sum()) / max(physical_rounds, 1.0)

        signal_prep_cuda_us = (
            float(group["signal_prep_cuda_us_sum"].sum())
            / max(physical_rounds, 1.0)
            if signal_cost_complete
            else float("nan")
        )
        candidate_cuda_us = update_component("candidate_cuda_us")
        controller_cuda_us = update_component("controller_cuda_us")
        publish_cuda_us = update_component("publish_cuda_us")
        adaptation_cuda_us = (
            signal_prep_cuda_us
            + candidate_cuda_us
            + controller_cuda_us
            + publish_cuda_us
            if adaptation_cost_complete
            else float("nan")
        )
        record = {
            "method": method,
            "report_name": _p5_report_name(method, identity[0]),
            "baseline_method": baseline_method,
            **dict(zip(keys, identity)),
            "rounds": int(rounds),
            "physical_rounds": int(physical_rounds),
            "algorithmic_censored_rounds": int(
                group["algorithmic_censored_count"].sum()
            ),
            "prompt_clusters": int(pgroup["prompt_cluster"].nunique()),
            "survival_weighted_accepted_prefix": estimate,
            "acceptance_ci_low": lo,
            "acceptance_ci_high": hi,
            "acceptance_gain_vs_baseline": gain_est,
            "acceptance_gain_ci_low": gain_lo,
            "acceptance_gain_ci_high": gain_hi,
            "gain_prompt_clusters": gain_n,
            "accepted_drafts_per_verify": accepted_sum / max(rounds, 1.0),
            "committed_tokens_per_verify": float(
                group["semantic_committed_sum"].sum()
            )
            / max(rounds, 1.0),
            "verified_drafts_per_verify": verified_sum / max(rounds, 1.0),
            "ordinary_acceptance_rate": float(
                accepted_sum / max(verified_sum, 1e-9)
            ),
            "verification_waste": float(group["semantic_waste_sum"].sum())
            / max(rounds, 1.0),
            "target_calls_per_output_token": float(
                group["semantic_target_calls_sum"].sum()
                / max(group["semantic_committed_sum"].sum(), 1)
            ),
            "draft_cuda_us": float(group["draft_cuda_us_sum"].sum())
            / max(physical_rounds, 1.0),
            "verify_cuda_us": float(group["verify_cuda_us_sum"].sum())
            / max(physical_rounds, 1.0),
            "accept_cuda_us": float(group["accept_cuda_us_sum"].sum())
            / max(physical_rounds, 1.0),
            "signal_prep_cuda_us": signal_prep_cuda_us,
            # candidate_cuda_us is inclusive side-stream candidate time;
            # backward/optimizer are diagnostic subsets and are not added to
            # adaptation_cuda_us a second time.
            "candidate_cuda_us": candidate_cuda_us,
            "backward_cuda_us": update_component("backward_cuda_us"),
            "optimizer_cuda_us": update_component("optimizer_cuda_us"),
            "controller_cuda_us": controller_cuda_us,
            "publish_cuda_us": publish_cuda_us,
            "side_queue_cuda_us": update_component("side_queue_cuda_us"),
            "barrier_wait_cpu_us": update_component("barrier_wait_cpu_us"),
            "adaptation_cuda_us": adaptation_cuda_us,
            "adaptation_update_count": int(group["update_count"].sum()),
            "adaptation_cost_complete": adaptation_cost_complete,
            "adaptation_cost_semantics": (
                "component_cuda_event_span_sum_not_critical_path"
            ),
            "signal_prep_unknown_rounds": signal_unknown_rounds,
            "mean_batch_size": physical_rounds / max(scheduler_steps, 1e-9),
            "request_weighted_mean_batch_size": request_weighted_batch,
            "decode_step_count_estimate": scheduler_steps,
            "version_mismatch_count": int(group["version_mismatch_count"].sum()),
            "observed_prefix_min": int(group["observed_prefix_min"].min()),
            "observed_prefix_max": int(group["observed_prefix_max"].max()),
        }
        if baseline_method == "static":
            record["acceptance_gain_vs_static"] = record[
                "acceptance_gain_vs_baseline"
            ]
        previous = 1.0
        for k in range(1, gamma + 1):
            semantic_column = f"semantic_survival_count_k{k}"
            column = f"survival_count_k{k}"
            survival = (
                float(group[semantic_column].sum()) / max(rounds, 1.0)
                if semantic_column in group
                else float(group[column].sum()) / max(rounds, 1.0)
                if column in group
                and not bool(group["algorithmic_censored_count"].sum())
                else float(
                    sum(
                        count if value >= k else 0
                        for value, count in zip(
                            group["accepted_drafts"],
                            group["semantic_round_count"],
                        )
                    )
                    / max(rounds, 1.0)
                )
            )
            record[f"survival_k{k}"] = survival
            record[f"conditional_accept_k{k}"] = (
                survival / previous if previous > 0 else float("nan")
            )
            previous = survival
        for name in (
            "trajectory_kind",
            "initial_prefix_len",
            "prefix_window_start",
            "prefix_window_end",
            "benchmark_repetitions",
        ):
            if name not in group:
                continue
            values = group[name].dropna().unique()
            if len(values) > 1:
                raise ValueError(
                    f"P5 group has conflicting {name} values: {values.tolist()}"
                )
            record[name] = values[0] if len(values) else None
        rows.append(record)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    group_keys = ["method", *P5_IDENTITY_COLUMNS]
    # A continuous trajectory's context coordinate is an interval, not its
    # right endpoint.  In particular, [128, 4096) must never enter the L>=4K
    # headline merely because it is displayed at 4096.  Independent checkpoint
    # runs retain the historical pointwise filter.
    if "trajectory_kind" in df and "prefix_window_start" in df:
        continuous = df["trajectory_kind"].astype(str).eq("continuous_prefix")
        window_start = pd.to_numeric(df["prefix_window_start"], errors="coerce")
        long_mask = (
            continuous & window_start.ge(long_context_min)
        ) | (
            ~continuous & df["context_length"].ge(long_context_min)
        )
    else:
        long_mask = df["context_length"].ge(long_context_min)
    long_prompt = _prompt_acceptance(df[long_mask])

    lcag = {}
    for key, method_group in long_prompt.groupby(group_keys, dropna=False):
        method = key[0]
        if method == baseline_method:
            lcag[key] = (0.0, 0.0, 0.0, 0)
            continue
        values = []
        clusters = []
        for (cluster, seed), curve in method_group.groupby(
            ["prompt_cluster", "seed"]
        ):
            deltas = []
            for row in curve.itertuples():
                identity = tuple(getattr(row, name) for name in keys)
                base = prompt_lookup.get((baseline_method, *identity), {})
                if (str(cluster), int(seed)) in base:
                    deltas.append(row.acceptance - base[(str(cluster), int(seed))])
            if deltas:
                values.append(float(np.mean(deltas)))
                clusters.append(str(cluster))
        lcag[key] = _boot(values, clusters, b=b)

    for key, indices in table.groupby(group_keys, dropna=False).groups.items():
        value, lo, hi, n = lcag.get(key, (float("nan"),) * 3 + (0,))
        table.loc[indices, "lcag"] = value
        table.loc[indices, "lcag_ci_low"] = lo
        table.loc[indices, "lcag_ci_high"] = hi
        table.loc[indices, "lcag_prompt_clusters"] = n
        ordered = table.loc[indices].sort_values("context_length")
        positive = ordered["acceptance_gain_vs_baseline"] > 0
        # A one-cluster bootstrap is degenerate (CI == point estimate).  It is
        # useful calibration evidence but cannot confirm an onset.
        sufficiently_paired = ordered["gain_prompt_clusters"] >= 2
        confirmed = sufficiently_paired & (
            ordered["acceptance_gain_ci_low"] > 0
        ) & sufficiently_paired.shift(-1, fill_value=False) & (
            ordered["acceptance_gain_ci_low"].shift(-1) > 0
        )
        selected = None
        if confirmed.any():
            selected = ordered.loc[confirmed].iloc[0]
            status = "confirmed"
        elif positive.any():
            selected = ordered.loc[positive].iloc[0]
            status = "candidate"
        else:
            status = "none"
        trajectory_kinds = (
            set(ordered["trajectory_kind"].dropna().astype(str))
            if "trajectory_kind" in ordered
            else set()
        )
        continuous = trajectory_kinds == {"continuous_prefix"}
        # A continuous bucket is an interval average.  Returning its right
        # endpoint as a point onset would overstate the measurement precision.
        # Keep the historical point field only for independent checkpoints and
        # expose the measured interval explicitly for continuous trajectories.
        onset_context = np.nan
        onset_window_start = np.nan
        onset_window_end = np.nan
        if selected is not None:
            if continuous:
                onset_window_start = int(selected["prefix_window_start"])
                onset_window_end = int(selected["prefix_window_end"])
            else:
                onset_context = int(selected["context_length"])
        table.loc[indices, "benefit_onset_context"] = onset_context
        table.loc[indices, "benefit_onset_window_start"] = onset_window_start
        table.loc[indices, "benefit_onset_window_end"] = onset_window_end
        table.loc[indices, "benefit_onset_status"] = status
    return table.sort_values([*group_keys, "context_length"]).reset_index(drop=True)


def acceptance_elasticity_table(
    rounds: pd.DataFrame,
    *,
    baseline_method: str = "static",
    b: int = 5000,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Prompt-paired log-length elasticity and curvature with BCa CIs."""
    columns = [
        "metric",
        "method",
        "baseline_method",
        *P5_IDENTITY_COLUMNS,
        "context_left",
        "context_center",
        "context_right",
        "estimate",
        "ci_low",
        "ci_high",
        "delta_vs_baseline",
        "delta_ci_low",
        "delta_ci_high",
        "prompt_clusters",
        "paired_prompt_clusters",
        "shape_semantics",
    ]
    normalized = _p5_rounds(rounds)
    trajectory_kinds = (
        set(normalized["trajectory_kind"].dropna().astype(str))
        if "trajectory_kind" in normalized
        else set()
    )
    shape_semantics = (
        "window_average_shape_proxy"
        if trajectory_kinds == {"continuous_prefix"}
        else "pointwise_context_shape"
    )
    prompt = _prompt_acceptance(normalized)
    identity_keys = list(P5_IDENTITY_COLUMNS)
    curves = {}
    for key, group in prompt.groupby(
        ["method", *identity_keys, "prompt_cluster", "seed"], dropna=False
    ):
        curves[key] = {
            int(row.context_length): float(row.acceptance)
            for row in group.itertuples()
        }

    def elasticity(curve, left, right):
        return -(
            np.log(max(curve[right], epsilon)) - np.log(max(curve[left], epsilon))
        ) / (np.log(right) - np.log(left))

    output = []
    identities = prompt[["method", *identity_keys]].drop_duplicates()
    for identity in identities.itertuples(index=False, name=None):
        identity_record = dict(zip(["method", *identity_keys], identity))
        method = identity_record["method"]
        selected = {
            key: curve
            for key, curve in curves.items()
            if key[: len(identity)] == identity
        }
        lengths = sorted({length for curve in selected.values() for length in curve})
        interval_values = {}
        for left, right in zip(lengths, lengths[1:]):
            vals, clusters, prompt_values = [], [], {}
            for key, curve in selected.items():
                if left in curve and right in curve:
                    value = float(elasticity(curve, left, right))
                    prompt_key = (str(key[-2]), int(key[-1]))
                    vals.append(value); clusters.append(prompt_key[0]); prompt_values[prompt_key] = value
            estimate, lo, hi, n = _boot(vals, clusters, b=b)
            interval_values[(left, right)] = prompt_values
            base_values = {}
            for key, curve in curves.items():
                if (
                    key[0] == baseline_method
                    and key[1 : len(identity)] == identity[1:]
                    and left in curve
                    and right in curve
                ):
                    base_values[(str(key[-2]), int(key[-1]))] = float(
                        elasticity(curve, left, right)
                    )
            paired = sorted(set(prompt_values) & set(base_values))
            delta = [prompt_values[key] - base_values[key] for key in paired]
            d_est, d_lo, d_hi, d_n = _boot(delta, [key[0] for key in paired], b=b)
            output.append({
                "metric": "elasticity",
                **identity_record,
                "baseline_method": baseline_method,
                "context_left": left,
                "context_center": float(np.sqrt(left * right)),
                "context_right": right,
                "estimate": estimate,
                "ci_low": lo,
                "ci_high": hi,
                "delta_vs_baseline": d_est,
                "delta_ci_low": d_lo,
                "delta_ci_high": d_hi,
                "prompt_clusters": n,
                "paired_prompt_clusters": d_n,
                "shape_semantics": shape_semantics,
            })

        for left, center, right in zip(lengths, lengths[1:], lengths[2:]):
            first = interval_values.get((left, center), {})
            second = interval_values.get((center, right), {})
            common = sorted(set(first) & set(second))
            denom = np.log(np.sqrt(center * right)) - np.log(np.sqrt(left * center))
            values = {key: (second[key] - first[key]) / denom for key in common}
            estimate, lo, hi, n = _boot(list(values.values()), [key[0] for key in values], b=b)
            base_first = {}
            base_second = {}
            for key, curve in curves.items():
                if (
                    key[0] == baseline_method
                    and key[1 : len(identity)] == identity[1:]
                ):
                    prompt_key = (str(key[-2]), int(key[-1]))
                    if left in curve and center in curve:
                        base_first[prompt_key] = float(elasticity(curve, left, center))
                    if center in curve and right in curve:
                        base_second[prompt_key] = float(elasticity(curve, center, right))
            paired = sorted(set(values) & set(base_first) & set(base_second))
            delta = [
                values[key] - (base_second[key] - base_first[key]) / denom
                for key in paired
            ]
            d_est, d_lo, d_hi, d_n = _boot(delta, [key[0] for key in paired], b=b)
            output.append({
                "metric": "curvature",
                **identity_record,
                "baseline_method": baseline_method,
                "context_left": left,
                "context_center": center,
                "context_right": right,
                "estimate": estimate,
                "ci_low": lo,
                "ci_high": hi,
                "delta_vs_baseline": d_est,
                "delta_ci_low": d_lo,
                "delta_ci_high": d_hi,
                "prompt_clusters": n,
                "paired_prompt_clusters": d_n,
                "shape_semantics": shape_semantics,
            })
    if not output:
        if baseline_method == "static":
            columns.append("delta_vs_static")
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(output, columns=columns)
    if baseline_method == "static":
        table["delta_vs_static"] = table["delta_vs_baseline"]
    return table.sort_values(
        ["metric", "method", *identity_keys, "context_center"]
    ).reset_index(drop=True)
