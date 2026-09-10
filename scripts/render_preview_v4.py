"""Vector plots from complete v4 evidence; no synthetic replacement for missing data."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


def interval(values):
    """Four independent block means, never request-level pseudo-replicates."""
    if len(values) != 4 or not np.isfinite(values).all():
        return None
    center = np.mean(values)
    radius = t.ppf(.975, 3) * np.std(values, ddof=1) / 2
    return center, center-radius, center+radius


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads((args.report / "preview.json").read_text())
    detail = json.loads((args.report / "details.json").read_text())
    if report["version"] != 4:
        raise ValueError("v4 only")
    output = args.report / "figures"
    output.mkdir(exist_ok=True)
    # Final-size standalone style, adapted from the figures/tables skill helper.
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 10, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": .02})
    families = {name: [] for name in ("efficiency", "long_trajectories", "cohort_gain", "dspark_serving", "memory_update")}
    def save(fig, family, key, caption):
        name = family + "-" + hashlib.sha256(key.encode()).hexdigest()[:12]
        fig.tight_layout()
        for extension in ("pdf", "svg"):
            fig.savefig(output / f"{name}.{extension}")
        plt.close(fig)
        families[family].append({"file": name, "condition": key, "caption": caption})
    groups = defaultdict(list)
    for row in report["rows"]:
        if row["status"] == "measured":
            key = "|".join(str(row[k]) for k in ("model", "panel", "task", "load", "flow_order", "topology"))
            groups[key].append(row)
    for key, rows in groups.items():
        variants = sorted({r["variant"] for r in rows})
        for family, metrics in (("efficiency", (("goodput", "Complete-window throughput (tok/s)"),
                  ("al_with_bonus", "Delivered draft + bonus / verify call"))),
                ("memory_update", (("peak_hbm_bytes", "Maximum rank allocator peak (GiB)"),
                  ("updates_published", "Published updates (count)")))):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            any_data = False
            for ax, (metric, label) in zip(axes, metrics, strict=True):
                for i, variant in enumerate(variants):
                    samples = {r["block"]: (r["metrics"] or {}).get(metric) for r in rows if r["variant"] == variant}
                    if set(samples) != set(range(4)) or any(type(v) not in (int, float) for v in samples.values()):
                        continue
                    values = np.array([samples[b] for b in range(4)], dtype=float)
                    if metric == "peak_hbm_bytes":
                        values /= 1024**3
                    bounds = interval(values)
                    if bounds is None:
                        continue
                    center, lower, upper = bounds
                    color = "#0072B2" if variant.startswith("lightcone:") else "#6B7280"
                    ax.scatter(i+np.linspace(-.12, .12, 4), values, color=color, s=14, alpha=.6)
                    ax.errorbar(i, center, yerr=[[center-lower], [upper-center]], fmt="o", color=color, capsize=4)
                    any_data = True
                ax.set_xticks(range(len(variants)), variants, rotation=25, ha="right")
                ax.set_ylabel(label)
            if any_data:
                save(fig, family, key, "All four raw block points and mean t95 interval (df=3); no pooled promotional multiplier. Missing metrics remain in the full table.")
            else:
                plt.close(fig)
    curves = defaultdict(list)
    for row in detail["cohort_curves"]:
        curves[(row["flow_order"], row["variant"])].append(row)
    for (order, variant), rows in curves.items():
        if len(rows) != 4 or {r["block"] for r in rows} != set(range(4)):
            continue
        values = np.array([[p["gain_seconds"] for p in r["points"]] for r in rows])
        center = values.mean(axis=0)
        radius = t.ppf(.975, 3)*values.std(axis=0, ddof=1)/2
        fig, ax = plt.subplots(figsize=(8, 3.8))
        x = np.arange(1, 49)
        for raw in values:
            ax.plot(x, raw, color="#0072B2", alpha=.3, linewidth=.7)
        ax.plot(x, center, "o-", color="#0072B2", markersize=2)
        ax.fill_between(x, center-radius, center+radius, color="#0072B2", alpha=.2)
        ax.axhline(0, color="#6B7280", linewidth=.8)
        ax.set(xlabel="Request ordinal from cold state", ylabel="Cumulative Static − LightCone time (s)")
        save(fig, "cohort_gain", order+variant, "Whole cold stream; four raw block curves and pointwise t95 interval, not a simultaneous band. Reset/endpoint costs included; loading listed separately.")
    lookup = {r["job_id"]: r for r in report["rows"]}
    grouped = defaultdict(lambda: defaultdict(list))
    for point in detail["long_generation_trajectories"]:
        row = lookup[point["job_id"]]
        key = "|".join(str(row[k]) for k in ("model", "task", "topology", "variant"))
        grouped[key][(point["position"], row["block"])].append(point["decode_tok_s"])
    for key, bins in grouped.items():
        measured = []
        for position in sorted({p for p, _ in bins}):
            if all((position, block) in bins for block in range(4)):
                measured.append((position, interval([np.mean(bins[position, block]) for block in range(4)])))
        if not measured:
            continue
        x = [p for p, _ in measured]
        values = np.array([b for _, b in measured])
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(x, values[:, 0], "o-", color="#0072B2")
        ax.fill_between(x, values[:, 1], values[:, 2], color="#0072B2", alpha=.2)
        ax.set(xlabel="Generated-token bin start", ylabel="Native within-bin decode speed (tok/s)")
        save(fig, "long_trajectories", key, "Surviving requests averaged within each block, then four-block mean/t95. EOS changes the population; counts and EOS are in details.json. Not a causal matched-length effect.")
    serving = defaultdict(list)
    for row in report["rows"]:
        if row["panel"] == "serving" and row["status"] == "measured":
            serving[(row["model"], row["topology"], row["variant"])].append(row)
    for key, rows in serving.items():
        points = []
        for load in ("closed_loop_c1", "closed_loop_c8", "closed_loop_c32"):
            matched = [r for r in rows if r["load"] == load]
            if len(matched) != 4:
                continue
            x = [r["metrics"].get("per_user_generation_speed") for r in matched]
            y = [r["metrics"].get("goodput") for r in matched]
            if any(type(v) not in (int, float) for v in x+y):
                continue
            points.append((load, interval(x), interval(y)))
        if not points:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for load, (x, xl, xu), (y, yl, yu) in points:
            ax.errorbar(x, y, xerr=[[x-xl], [xu-x]], yerr=[[y-yl], [yu-y]], fmt="o", color="#0072B2", capsize=4)
            ax.annotate(load.removeprefix("closed_loop_"), (x, y), xytext=(7, 7), textcoords="offset points")
        ax.set(xlabel="Native per-user generation speed (tok/s)", ylabel="Complete-window throughput (tok/s)")
        save(fig, "dspark_serving", "|".join(key), "All measured registered concurrency points; marginal four-block t95 intervals. Not a hand-selected frontier; BurstGPT stays in the complete table.")
    manifest = {"scope": "GitHub preview, not paper integration", "families": families,
        "status": {k: "generated_requires_visual_review" if v else "UNMEASURED" for k, v in families.items()},
        "sources": {name: hashlib.sha256((args.report / name).read_bytes()).hexdigest() for name in ("preview.json", "details.json")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: len(v) for k, v in families.items()}))


if __name__ == "__main__":
    main()
