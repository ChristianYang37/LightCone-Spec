"""Complete v4 evidence, block-level effects and disclosed scene selection."""

import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

FIELDS = ("goodput", "duration_seconds", "session_startup_seconds", "request_count",
          "committed_tokens", "accepted_drafts", "target_calls", "verified_drafts",
          "per_user_generation_speed", "itl_p99_ms", "ttft_p50_ms", "ttft_p99_ms",
          "updates_published", "resolved_stride", "peak_hbm_bytes", "nvml_peak_hbm_bytes",
          "allocated_peak_hbm_bytes", "reserved_peak_hbm_bytes", "rank_memory",
          "memory_budget", "kv_capacity", "measured_update_peak_bytes",
          "request_outcomes", "scientific_outcome", "hard_feasible", "capacity_feasible", "capacity_reason",
          "delivered_bonus_tokens", "delivered_draft_tokens", "delivered_verification_calls",
          "cell_setup_seconds", "cell_tail_seconds", "cell_wall_seconds")


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def paired_effect(points):
    complete = len(points) == 4 and {p["block"] for p in points} == set(range(4))
    logs = [math.log(p["candidate"] / p["baseline"]) for p in points]
    mean = float(np.mean(logs)) if complete else None
    radius = float(t.ppf(.975, 3) * np.std(logs, ddof=1) / 2) if complete else None
    return {"status": "measured" if complete else "UNMEASURED", "n_blocks": len(points),
            "raw_points": points, "paired_log_ratios": logs,
            "ratio": math.exp(mean) if complete else None,
            "ratio_ci95": [math.exp(mean - radius), math.exp(mean + radius)] if complete else None}


def scene_ranking(effects):
    """Rank whole scenes, including negative results; never select individual takes."""
    candidates = [e for e in effects if e["metric"] == "goodput" and e["status"] == "measured"
                  and e["baseline"] == "static:DFLASH:static" and e["candidate"].startswith("lightcone:")
                  and e["panel"] in {"long_generation", "qwen38_transfer", "cohort"}]
    candidates += [e for e in effects if e["metric"] == "goodput" and e["status"] == "measured"
                   and e["baseline"] == "static:DSPARK:static" and e["panel"] in {"serving", "burstgpt"}]
    groups = defaultdict(list)
    for effect in candidates:
        if effect["panel"] == "cohort" and effect["candidate"] != "lightcone:DFLASH:cohort":
            continue
        groups["cohort_appendix" if effect["panel"] == "cohort" else effect["model"]].append(effect)
    return {key: {"selected_scene": ranked[0]["scene_id"], "candidates": ranked,
                  "positive_gain": ranked[0]["ratio"] > 1,
                  "disclosure": "Selected effect case from formal results, not average performance; c1 does not establish c8 gain",
                  "take_rule": "one valid take per method; technical failures retained; slow takes never retried"}
            for key, entries in groups.items()
            if (ranked := sorted(entries, key=lambda e: (-e["ratio"], e["scene_id"]))) }


def cumulative_gain(static_rows, candidate_rows, *, static_setup=0., candidate_setup=0.):
    """Ordered, paired cold flow: include all requests and separately charged setup."""
    if len(static_rows) != 48 or len(candidate_rows) != 48:
        raise ValueError("cohort gain needs the full 48-request stream")
    value, points = static_setup - candidate_setup, []
    for index, (a, b) in enumerate(zip(static_rows, candidate_rows, strict=True)):
        if (a["source"], a["problem_id"]) != (b["source"], b["problem_id"]):
            raise ValueError("cohort request identity/order mismatch")
        if not all(positive(r["wall_seconds"]) for r in (a, b)):
            raise ValueError("invalid cohort request wall time")
        value += a["wall_seconds"] - b["wall_seconds"]
        points.append({"n": index + 1, "gain_seconds": value,
                       "domain": a["source"], "problem_id": a["problem_id"]})
    first = next((p["n"] for p in points if p["gain_seconds"] > 0), None)
    return {"points": points, "first_payback_request": first, "final_net_seconds": value,
            "turned_negative_after_payback": any(p["gain_seconds"] < 0 and p["n"] > first for p in points) if first else False,
            "setup_included_seconds": {"static": static_setup, "candidate": candidate_setup}}


def summary(evidence, expected_jobs, output):
    evidence = list(evidence)
    expected = {j.job_id: j for j in expected_jobs}
    measured, rows, scenes = {}, [], defaultdict(dict)
    for config, metrics in evidence:
        identity = config.get("job_id")
        if identity not in expected:
            continue
        if identity in measured:
            raise ValueError("duplicate v4 attempt evidence")
        job, params = expected[identity], config["parameters"]
        for key in ("preview_revision", "preview_state_scope", "stride", "pairing_key", "topology", "frozen_recipe"):
            if params.get(key) != job.parameters.get(key):
                raise ValueError(f"v4 evidence mismatch: {key}")
        m = {key: metrics.get(key) for key in FIELDS}
        m["execution_policy"] = params.get("execution_policy")
        m["effective_load"] = metrics.get("effective_load")
        m["memory_budget_policy"] = params.get("memory_budget_policy", "fixed_reserve_v1")
        calls, accepted, bonus = m["delivered_verification_calls"], m["delivered_draft_tokens"], m["delivered_bonus_tokens"]
        m["al_draft_only"] = accepted / calls if job.method != "target_only" and positive(calls) and isinstance(accepted, (int, float)) else None
        # Missing bonus evidence is not assumed to be one per verification.
        m["al_with_bonus"] = (accepted + bonus) / calls if m["al_draft_only"] is not None and isinstance(bonus, (int, float)) else None
        measured[identity] = m
    for identity, job in expected.items():
        p, m = job.parameters, measured.get(identity)
        variant = f"{job.method}:{job.backend}:{p['state_variant']}"
        condition = (job.model, p["preview_panel"], job.task, job.load, p.get("flow_order", ""), p["topology"])
        row = {"job_id": identity, "model": job.model, "panel": p["preview_panel"], "task": job.task,
               "load": job.load, "flow_order": p.get("flow_order"), "topology": p["topology"],
               "block": job.block, "variant": variant, "method_label": p["method_label"],
               "state_scope": p["preview_state_scope"], "metrics": m,
               "status": "UNMEASURED" if m is None else "measured" if m["hard_feasible"] is True else "unavailable"}
        rows.append(row)
        if row["status"] == "measured":
            scenes[condition][(variant, job.block)] = m
    effects = []
    for condition, measurements in sorted(scenes.items()):
        model, panel, task, load, order, topology = condition
        variants = sorted({variant for variant, _ in measurements})
        for candidate in (v for v in variants if v.startswith("lightcone:")):
            for baseline in (v for v in variants if not v.startswith("lightcone:") or (panel == "cohort" and v.endswith(":request") and v != candidate)):
                for metric in ("goodput", "al_draft_only", "al_with_bonus", "per_user_generation_speed"):
                    points = []
                    for block in range(4):
                        a, b = measurements.get((candidate, block), {}), measurements.get((baseline, block), {})
                        environment = ("execution_policy", "effective_load", "memory_budget_policy")
                        if any(a.get(k) is None or a.get(k) != b.get(k) for k in environment):
                            continue
                        if positive(a.get(metric)) and positive(b.get(metric)):
                            points.append({"block": block, "candidate": a[metric], "baseline": b[metric]})
                    effects.append({"scene_id": "|".join(map(str, condition)), "model": model,
                                    "panel": panel, "task": task, "load": load, "order": order, "topology": topology,
                                    "candidate": candidate, "baseline": baseline, "metric": metric, **paired_effect(points)})
    result = {"version": 4, "expected_cells": len(expected), "counts": dict(Counter(r["status"] for r in rows)),
              "rows": rows, "effects": effects, "selected_scenes": scene_ranking(effects),
              "uncertainty": "Four paired blocks; log-ratio t interval df=3; approximate normality; no request pseudoreplication"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "preview.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    export_details(result, evidence, output)
    return result


def export_details(result, evidence, output):
    """Compact public counts and full-flow curves; no prompts or local paths."""
    from .preview_v4_qa import read_rows
    records = {c["job_id"]: m for c, m in evidence}
    cohorts, trajectories, hashes = [], [], {}
    rows = {r["job_id"]: r for r in result["rows"]}
    costs = {}
    for identity, metrics in records.items():
        if identity not in rows or rows[identity]["status"] != "measured":
            continue
        directory = metrics.get("source_attempt_dir")
        if not directory:
            continue
        directory = Path(directory)
        path = directory / "preview_request_costs.jsonl.gz"
        if path.is_file():
            costs[identity] = read_rows(path)
            hashes[f"{identity}/request_costs"] = hashlib.sha256(path.read_bytes()).hexdigest()
        path = directory / "requests.jsonl.gz"
        if rows[identity]["panel"] not in {"long_generation", "qwen38_transfer"} or not path.is_file():
            continue
        hashes[f"{identity}/requests"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for index, request in enumerate(read_rows(path)):
            timestamps = request.get("native_token_timestamps_ns", [])
            if len(timestamps) != request.get("completion_tokens"):
                raise ValueError("trajectory lacks complete native timestamps")
            for start in range(0, len(timestamps), 1024):
                end = min(start + 1024, len(timestamps))
                seconds = (timestamps[end - 1] - timestamps[start]) / 1e9
                if seconds > 0:
                    trajectories.append({"job_id": identity, "request_index": index,
                        "position": start, "tokens_observed": end-start,
                        "decode_tok_s": (end-start-1)/seconds,
                        "completion_tokens": len(timestamps), "finish_reason": request.get("stop_reason")})
    for identity, row in rows.items():
        if row["panel"] != "cohort" or not row["variant"].startswith("lightcone:") or identity not in costs:
            continue
        baseline = next((r for r in rows.values() if r["panel"] == "cohort"
            and r["variant"] == "static:DFLASH:static" and r["block"] == row["block"]
            and r["flow_order"] == row["flow_order"] and r["topology"] == row["topology"]), None)
        if baseline is None or baseline["job_id"] not in costs:
            continue
        cohorts.append({"job_id": identity, "baseline_job_id": baseline["job_id"],
            "block": row["block"], "flow_order": row["flow_order"], "variant": row["variant"],
            **cumulative_gain(costs[baseline["job_id"]], costs[identity]),
            "cost_scope": "cold measurement flow including endpoint/reset work; model/session loading published separately"})
    detail = {"version": 4, "cohort_curves": cohorts, "long_generation_trajectories": trajectories,
              "raw_evidence_sha256": hashes,
              "trajectory_uncertainty": "requests are nested within four blocks; no request-level run error bars"}
    (output / "details.json").write_text(json.dumps(detail, indent=2, allow_nan=False))
    fields = ["job_id", "model", "panel", "task", "load", "flow_order", "topology", "block", "variant", "status",
              "goodput", "al_draft_only", "al_with_bonus", "per_user_generation_speed", "peak_hbm_bytes", "updates_published"]
    with (output / "cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: (r.get(key) if key in r else (r["metrics"] or {}).get(key)) for key in fields}
                         for r in result["rows"])
    content = ["<!doctype html><meta charset='utf-8'><title>Preview v4 complete evidence</title>",
               "<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse}td,th{padding:.4rem;border:1px solid #bbb}pre{white-space:pre-wrap}</style>",
               "<h1>Preview v4 — all 180 registered cells</h1>",
               "<p>Selected videos are effect cases, not average performance. Negative and unavailable results remain visible.</p>",
               f"<pre>{html.escape(json.dumps(result['counts']))}</pre><p>{html.escape(result['uncertainty'])}</p>",
               "<p><a href='cells.csv'>All raw block points (CSV)</a> · <a href='preview.json'>Effects and full scene ranking</a> · <a href='details.json'>Cold-flow and trajectory evidence</a></p>",
               "<table><tr>" + "".join(f"<th>{k}</th>" for k in fields) + "</tr>"]
    for row in result["rows"]:
        values = [row.get(k) if k in row else (row["metrics"] or {}).get(k) for k in fields]
        content.append("<tr>" + "".join(f"<td>{html.escape(str(v) if v is not None else 'UNMEASURED/N/A')}</td>" for v in values) + "</tr>")
    content.append("</table>")
    (output / "index.html").write_text("\n".join(content))
