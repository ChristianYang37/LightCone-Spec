#!/usr/bin/env python3
"""Render three standalone evidence groups only from completed preview exports.

New contributions: LicenseRef-LightCone-Source-Available-1.0. See LICENSE.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--style-helper", type=Path,
                        help="Optional local ai-paper-figures-tables figstyle.py; not redistributed")
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    if evidence.get("expected_cells") != 56 or evidence["counts"].get("UNMEASURED", 0):
        raise RuntimeError("all 56 terminal outcomes are required before rendering release figures")
    if args.style_helper:
        import sys
        spec = importlib.util.spec_from_file_location("preview_figstyle", args.style_helper)
        style = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = style
        spec.loader.exec_module(style)
        style.set_publication_style()
    else:
        plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 10,
                             "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42,
                             "savefig.pad_inches": .02, "legend.frameon": False})
    args.output.mkdir(parents=True, exist_ok=True)
    labels = {"static": "Static", "tts": "Full TTS", "lightcone": "LightCone"}
    colors = {"static": "#D55E00", "tts": "#009E73", "lightcone": "#0072B2"}
    captions = []
    for panel, metrics in (
        ("long_generation", ("accepted_drafts_per_target_call", "goodput")),
        ("serving", ("goodput", "per_user_generation_speed", "accepted_drafts_per_target_call")),
        ("burstgpt", ("goodput", "ttft_p50_ms", "itl_p99_ms")),
    ):
        all_rows = [row for row in evidence["rows"] if row["panel"] == panel]
        conditions = sorted({row["task"] if panel == "long_generation" else row["load"] for row in all_rows},
                            key=lambda x: int(x.removeprefix("closed_loop_c")) if x.startswith("closed_loop_c") else 0)
        ncols = len(metrics) + int(panel == "serving")
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 2.8), squeeze=False)
        for ax, metric in zip(axes[0][:len(metrics)], metrics, strict=True):
            for method in sorted({row["method"] for row in all_rows}):
                centers, lows, highs = [], [], []
                for index, condition in enumerate(conditions):
                    group = [row for row in all_rows if row["method"] == method and
                             (row["task"] if panel == "long_generation" else row["load"]) == condition]
                    values = [row["metrics"].get(metric) for row in group if row["status"] == "measured"]
                    values = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
                    ax.scatter([index] * len(values), values, color=colors[method], s=13, alpha=.7)
                    if len(values) == 4:
                        center = np.mean(values)
                        radius = t.ppf(.975, 3) * np.std(values, ddof=1) / 2
                        centers.append(center)
                        lows.append(center-radius)
                        highs.append(center+radius)
                    else:
                        centers.append(np.nan)
                        lows.append(np.nan)
                        highs.append(np.nan)
                ax.plot(range(len(conditions)), centers, color=colors[method], marker="o", label=labels[method])
                ax.fill_between(range(len(conditions)), lows, highs, color=colors[method], alpha=.2)
            ax.set_xticks(range(len(conditions)), [v.replace("closed_loop_", "") for v in conditions])
            ax.set_ylabel({"goodput": "Committed tokens/s", "accepted_drafts_per_target_call": "AL (bonus excluded)",
                           "per_user_generation_speed": "Native per-user tokens/s", "ttft_p50_ms": "Median TTFT (ms)",
                           "itl_p99_ms": "p99 ITL (ms)"}[metric])
            ax.margins(y=.35)
            ax.legend(loc="upper right")
        if panel == "serving":
            ax = axes[0][-1]
            for method in ("static", "lightcone"):
                centers = []
                for condition in conditions:
                    pairs = [(row["metrics"].get("goodput"), row["metrics"].get("per_user_generation_speed"))
                             for row in all_rows if row["method"] == method
                             and row["load"] == condition and row["status"] == "measured"]
                    pairs = [p for p in pairs if all(isinstance(v, (int, float)) and np.isfinite(v) for v in p)]
                    if pairs:
                        a = np.asarray(pairs)
                        ax.scatter(a[:, 0], a[:, 1], color=colors[method], alpha=.7, s=13)
                    if len(pairs) == 4:
                        center = a.mean(axis=0)
                        radius = t.ppf(.975, 3) * a.std(axis=0, ddof=1) / 2
                        ax.errorbar(*center, xerr=radius[0], yerr=radius[1], color=colors[method], capsize=2)
                        ax.annotate(condition.removeprefix("closed_loop_"), center, fontsize=8,
                                    xytext=(4, 4), textcoords="offset points")
                        centers.append(center)
                if centers:
                    centers = np.asarray(centers)
                    ax.plot(centers[:, 0], centers[:, 1], color=colors[method], label=labels[method])
            ax.set_xlabel("Aggregate committed tokens/s")
            ax.set_ylabel("Native per-user tokens/s")
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / f"{panel}.pdf", bbox_inches="tight")
        fig.savefig(args.output / f"{panel}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        missing = [row["job_id"] for row in all_rows if row["status"] != "measured"]
        captions.append(f"{panel}: independent block points and mean t95% intervals (n=4, df=3). "
                        f"Unavailable cells: {missing or 'none'}. No interval for incomplete groups.")
    # Lossless allowlisted table, including all unavailable rows.
    with (args.output / "cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("job_id", "panel", "task", "method", "load", "block", "status", "metrics"))
        writer.writeheader()
        for row in evidence["rows"]:
            writer.writerow({key: json.dumps(row[key]) if key == "metrics" else row[key] for key in writer.fieldnames})
    (args.output / "captions.txt").write_text("\n\n".join(captions) + "\n\nStandalone figures; final-size visual QA required before release.\n")


if __name__ == "__main__":
    main()
