"""Standard figures: predictor comparison, delay x drift utility map,
speedup table plot, collapse plots for P0-A (spec 13.1, 16.4).

All figures render headless (Agg backend) and save deterministic PNGs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def predictor_comparison_figure(
    evaluations: dict[str, dict], path: str | Path
) -> Path:
    names = list(evaluations)
    maes = [evaluations[n]["mae"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, maes, color="#4878a8")
    ax.set_ylabel("held-out utility MAE")
    ax.set_title("Clock predictor comparison (same model class)")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, path)


def delay_drift_utility_figure(
    delays: np.ndarray,
    drifts: np.ndarray,
    utilities: np.ndarray,
    path: str | Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(delays, drifts, c=utilities, cmap="coolwarm_r", s=18)
    fig.colorbar(sc, ax=ax, label="future utility U(8)")
    ax.set_xlabel("round delay")
    ax.set_ylabel("trajectory drift (rho)")
    ax.set_title("Delay x drift utility map")
    return _save(fig, path)


def speedup_figure(table_rows: list[dict], path: str | Path) -> Path:
    names = [r["report_name"] for r in table_rows]
    speedup_key = (
        "speedup_vs_baseline"
        if table_rows and "speedup_vs_baseline" in table_rows[0]
        else "speedup_vs_static"
    )
    speedups = [r[speedup_key] for r in table_rows]
    lows = [r[speedup_key] - r["speedup_ci_low"] for r in table_rows]
    highs = [r["speedup_ci_high"] - r[speedup_key] for r in table_rows]
    baseline = table_rows[0].get("baseline_method", "static") if table_rows else "static"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(
        names, speedups, yerr=[lows, highs], fmt="o", capsize=4, color="#a84848"
    )
    ax.axhline(1.0, ls="--", color="grey")
    ax.set_ylabel(f"decode TPS speedup vs {baseline}")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, path)


def long_context_acceptance_figure(table, path: str | Path) -> Path:
    """P5 A(L) and committed-token curves; facets stay in the data labels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    identities = [
        name
        for name in (
            "model_pair",
            "weight_update_mode",
            "update_stride",
            "method",
        )
        if name in table
    ]
    for identity, group in table.groupby(identities, dropna=False):
        if not isinstance(identity, tuple):
            identity = (identity,)
        curve = group.groupby("context_length", as_index=False).agg(
            acceptance=("survival_weighted_accepted_prefix", "mean"),
            committed=("committed_tokens_per_verify", "mean"),
        )
        label = ":".join(str(value) for value in identity)
        axes[0].plot(curve.context_length, curve.acceptance, marker="o", label=label)
        axes[1].plot(curve.context_length, curve.committed, marker="o", label=label)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("prefix length before proposal")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("survival-weighted accepted prefix A(L)")
    axes[1].set_ylabel("committed tokens / target verification")
    axes[0].legend(fontsize=7)
    return _save(fig, path)


def acceptance_shape_figure(table, path: str | Path) -> Path:
    """P5 elasticity and curvature with prompt-cluster BCa intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False)
    for metric, ax in (("elasticity", axes[0]), ("curvature", axes[1])):
        selected = table[table["metric"] == metric]
        has_positive_context = False
        identities = [
            name
            for name in (
                "model_pair",
                "weight_update_mode",
                "update_stride",
                "method",
            )
            if name in selected
        ]
        for identity, group in selected.groupby(identities, dropna=False):
            if not isinstance(identity, tuple):
                identity = (identity,)
            curve = group.groupby("context_center", as_index=False).agg(
                estimate=("estimate", "mean"),
                low=("ci_low", "mean"),
                high=("ci_high", "mean"),
            )
            curve = curve[
                (curve.context_center > 0)
                & np.isfinite(curve.context_center)
                & np.isfinite(curve.estimate)
                & np.isfinite(curve.low)
                & np.isfinite(curve.high)
            ]
            if curve.empty:
                continue
            has_positive_context = True
            label = ":".join(str(value) for value in identity)
            ax.plot(curve.context_center, curve.estimate, marker="o", label=label)
            ax.fill_between(
                curve.context_center, curve.low, curve.high, alpha=0.12
            )
        if has_positive_context:
            ax.set_xscale("log", base=2)
        else:
            ax.text(
                0.5,
                0.5,
                "insufficient context buckets",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel("context length")
        ax.set_ylabel("E(L)" if metric == "elasticity" else "C(L)")
        ax.grid(alpha=0.2)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7)
    return _save(fig, path)


def acceptance_cost_pareto_figure(table, path: str | Path) -> Path:
    """Acceptance gain versus real throughput, colored by CUDA cost."""
    gain_column = (
        "acceptance_gain_vs_baseline"
        if "acceptance_gain_vs_baseline" in table
        else "acceptance_gain_vs_static"
    )
    speedup_column = (
        "throughput_speedup_vs_baseline"
        if "throughput_speedup_vs_baseline" in table
        else "throughput_speedup_vs_static"
    )
    required = {gain_column, speedup_column, "round_cuda_us"}
    fig, ax = plt.subplots(figsize=(7, 5))
    baseline = (
        str(table["baseline_method"].iloc[0])
        if "baseline_method" in table and len(table)
        else "static"
    )
    if required.issubset(table.columns):
        identities = [
            name
            for name in (
                "model_pair",
                "weight_update_mode",
                "update_stride",
                "method",
            )
            if name in table
        ]
        for identity, group in table.groupby(identities, dropna=False):
            if not isinstance(identity, tuple):
                identity = (identity,)
            method = identity[-1]
            if method == baseline:
                continue
            group = group[np.isfinite(group["round_cuda_us"])]
            if group.empty:
                continue
            label = ":".join(str(value) for value in identity)
            sc = ax.scatter(
                group[gain_column],
                group[speedup_column],
                c=group.round_cuda_us,
                s=32,
                alpha=0.8,
                label=label,
                cmap="viridis",
            )
        if "sc" in locals():
            fig.colorbar(
                sc,
                ax=ax,
                label="decode + adaptation CUDA-event span us / verify",
            )
    ax.axhline(1.0, ls="--", color="grey")
    ax.axvline(0.0, ls="--", color="grey")
    ax.set_xlabel(f"accepted drafts / verify gain vs {baseline}")
    ax.set_ylabel(f"decode goodput speedup vs {baseline}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    return _save(fig, path)


def collapse_figure(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    path: str | Path,
    title: str = "P0-A collapse plot",
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, (x, y) in curves.items():
        ax.plot(x, y, marker="o", ms=3, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    return _save(fig, path)
