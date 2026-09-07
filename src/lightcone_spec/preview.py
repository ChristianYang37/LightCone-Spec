"""Registered, append-only four-block preview; never a replacement for main effects.

New contributions: LicenseRef-LightCone-Source-Available-1.0. See LICENSE.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

from .protocol import Job
from .scheduling import logical_unit_key

PREVIEW_NODES = ("E3b-preview-v1", "E5-preview-v1")
VIDEO_METHODS = (
    ("NONE", "target_only", "Target-only"),
    ("EAGLE3", "static", "Static EAGLE3"),
    ("DFLASH", "static", "Static DFlash"),
    ("DSPARK", "static", "Native DSpark"),
    ("DFLASH", "tts_lora_batched", "TTS-LoRA-Batched (DFlash)"),
    ("DFLASH", "lightcone", "LightCone (DFlash)"),
)


def preview_jobs(manifest: dict) -> tuple[Job, ...]:
    """All mutable selections and prompt/trace decisions are frozen once upstream."""
    jobs = []
    for block in range(4):
        conditions = [
            ("long_generation", task, "c1", ("static", "tts", "lightcone"))
            for task in ("LiveCodeBench", "MATH-500")
        ] + [
            ("serving", "LiveCodeBench", f"closed_loop_c{c}", ("static", "lightcone"))
            for c in (1, 8, 32)
        ] + [("burstgpt", "LiveCodeBench", "burstgpt_shape", ("static", "lightcone"))]
        for panel, task, load, methods in conditions:
            order = list(methods)
            random.Random(f"preview-v1:{panel}:{task}:{load}:{block}").shuffle(order)
            long = panel == "long_generation"
            backend = "DFLASH" if long else "DSPARK"
            for method in order:
                recipe_name = "tts_recipe" if method == "tts" else (
                    "dspark_recipe" if backend == "DSPARK" else "lightcone_recipe"
                )
                params = {
                    "panel": "preview_v1", "preview_panel": panel,
                    "pairing_key": f"preview-v1:{panel}:{task}:{load}:{block}",
                    "sampling_seed": block, "topology": "tp1_dp1",
                    "comparison_backend": backend, "stride": 10,
                    "memory_budget_policy": "fixed_reserve_v1",
                    "frozen_recipe": manifest[recipe_name] if method != "static" else None,
                    "verification": "native_scheduler",
                    "execution_request_count": 8 if long else (
                        manifest["trace_request_count"] if panel == "burstgpt" else 32
                    ),
                    "preview_prompt_records": manifest["prompts"][task][:8 if long else 32],
                    "preview_prompt_offset": manifest["trace_offset"] if panel == "burstgpt" else 0,
                    "regime": "mechanism_native_prompt" if long else "long_input_short_output",
                    "generation_tokens": 32768 if long else manifest["serving_output_tokens"],
                    "respect_eos": long,
                    "workload": "preview_formal_supplement",
                    # Normal performance counters only; diagnostic entropy/CE adds overhead.
                }
                if panel == "burstgpt":
                    params.update(
                        registered_load="burstgpt_shape", arrival_trace="BurstGPT",
                        preview_anchor=manifest["trace_anchor"],
                        preview_trace=manifest["trace"],
                    )
                jobs.append(Job(
                    job_id=f"preview-v1__{panel}__{task}__{load}__b{block}__{method}",
                    node=PREVIEW_NODES[0 if long else 1], ordinal=len(jobs),
                    method=method, model="Qwen/Qwen3-8B", backend=backend,
                    task=task, context=40928, load=load, block=block, gpu_count=1,
                    width=manifest["dflash_width"] if long else manifest["dspark_width"],
                    parameters=params,
                ))
    assert len(jobs) == 56 and len({job.job_id for job in jobs}) == 56
    return tuple(jobs)


def held_out_pool(records, calibration, count: int) -> list[dict]:
    """No outcome-dependent selection; exclude matching IDs AND exact prompt text."""
    texts = {row["prompt"].strip() for row in calibration}
    ids = {(str(row.get("source", "")), str(row.get("problem_id", "")))
           for row in calibration if row.get("problem_id")}
    chosen, seen = [], set()
    rows = sorted(records, key=lambda row: (str(row.get("problem_id", "")), row["prompt"]))
    random.Random(0).shuffle(rows)
    for row in rows:
        text = row["prompt"].strip()
        identity = (str(row.get("source", "")), str(row.get("problem_id", "")))
        if text in texts or text in seen or identity in ids:
            continue
        chosen.append(dict(row))
        seen.add(text)
        if len(chosen) == count:
            return chosen
    raise ValueError(f"preview needs {count} distinct held-out prompts; found {len(chosen)}")


def preview_summary(evidence, expected_jobs, output: Path) -> dict:
    """Allowlisted public counters; raw prompt text and local paths never exported."""
    expected = {job.job_id: job for job in expected_jobs}
    measured = {}
    fields = (
        "goodput", "per_user_generation_speed", "accepted_drafts_per_target_call",
        "accepted_drafts", "target_calls", "verified_drafts", "committed_tokens",
        "duration_seconds", "session_startup_seconds", "request_count", "itl_p99_ms", "ttft_p50_ms", "ttft_p99_ms",
        "hard_feasible", "capacity_feasible", "scientific_outcome", "request_outcomes",
        "updates_published", "resolved_stride", "peak_hbm_bytes",
    )
    for config, metrics in evidence:
        if config.get("job_id") not in expected or config.get("parameters", {}).get("panel") != "preview_v1":
            continue
        identity = config["job_id"]
        if identity in measured:
            raise ValueError(f"duplicate preview evidence: {identity}")
        measured[identity] = {key: metrics.get(key) for key in fields}
        calls, accepted, verified = (metrics.get(key) for key in (
            "target_calls", "accepted_drafts", "verified_drafts",
        ))
        if isinstance(calls, (int, float)) and calls > 0 and isinstance(accepted, (int, float)):
            measured[identity]["accepted_drafts_per_target_call"] = accepted / calls
        measured[identity]["accepted_verified_ratio"] = (
            accepted / verified if isinstance(accepted, (int, float))
            and isinstance(verified, (int, float)) and verified > 0 else None
        )
        measured[identity]["execution_policy"] = config.get("parameters", {}).get("execution_policy")
        measured[identity]["effective_load"] = metrics.get("effective_load")
    rows, groups = [], defaultdict(dict)
    for identity, job in expected.items():
        m = measured.get(identity)
        row = {
            "job_id": identity, "panel": job.parameters["preview_panel"], "task": job.task,
            "method": job.method, "backend": job.backend, "load": job.load, "block": job.block,
            "status": "UNMEASURED" if m is None else (
                "measured" if m.get("hard_feasible") is True else "unavailable"
            ), "metrics": m,
        }
        rows.append(row)
        if row["status"] == "measured":
            groups[(row["panel"], job.task, job.load)][(job.method, job.block)] = m
    effects = []
    for condition, data in sorted(groups.items()):
        for baseline, metric in itertools.product(
            ("static", "tts") if condition[0] == "long_generation" else ("static",),
            ("goodput", "accepted_drafts_per_target_call", "per_user_generation_speed"),
        ):
            ratios = []
            for block in range(4):
                a, b = data.get(("lightcone", block), {}), data.get((baseline, block), {})
                x, y = a.get(metric), b.get(metric)
                if (a.get("execution_policy") != b.get("execution_policy")
                        or a.get("effective_load") != b.get("effective_load")):
                    continue
                if all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (x, y)):
                    ratios.append(math.log(x / y))
            complete = len(ratios) == 4
            mean = float(np.mean(ratios)) if complete else None
            radius = float(t.ppf(.975, 3) * np.std(ratios, ddof=1) / 2) if complete else None
            effects.append({
                "condition": list(condition), "baseline": baseline, "metric": metric,
                "n_blocks": len(ratios), "status": "measured" if complete else "UNMEASURED",
                "paired_log_ratios": ratios,
                "ratio": math.exp(mean) if complete else None,
                "gain_percent": 100 * math.expm1(mean) if complete else None,
                "ratio_ci95": [math.exp(mean-radius), math.exp(mean+radius)] if complete else None,
            })
    result = {
        "panel": "preview_v1", "expected_cells": len(expected),
        "counts": dict(Counter(row["status"] for row in rows)), "rows": rows, "effects": effects,
        "uncertainty": "four independent paired blocks; log-ratio Student t, df=3; approximate normality",
        "AL_definition": "accepted draft tokens / target verification calls; bonus token excluded",
        "scope": "formal small-budget supplement; not pooled with primary confirmatory effects",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "preview.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def preview_eta(evidence, remaining_jobs, *, repetitions=10000) -> dict:
    """Cell-time resampling, frozen whole-pair units; never extrapolate missing panels."""
    def key(job, policy):
        return (job.backend, job.method, job.task, job.load, job.parameters["preview_panel"],
                job.parameters.get("memory_budget_policy"), policy)

    pools = defaultdict(list)
    for config, metrics in evidence:
        if config.get("parameters", {}).get("panel") != "preview_v1" or metrics.get("hard_feasible") is not True:
            continue
        duration, startup = metrics.get("duration_seconds"), metrics.get("session_startup_seconds")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (duration, startup)):
            continue
        job = Job(**{name: config[name] for name in Job.__dataclass_fields__ if name in config})
        pools[key(job, job.parameters.get("execution_policy"))].append(duration + startup)
    units, missing = defaultdict(list), Counter()
    for job in remaining_jobs:
        units[logical_unit_key(job)].append(job)
    rng = np.random.default_rng(0)
    clocks = np.zeros((2, repetitions))
    samples = np.arange(repetitions)
    priced = 0
    for jobs in units.values():
        cost = np.zeros(repetitions)
        for job in jobs:
            pool = pools.get(key(job, "automatic_units_v3"))
            if not pool:
                missing[str(key(job, "automatic_units_v3"))] += 1
                continue
            cost += rng.choice(pool, repetitions)
            priced += 1
        gpu = clocks.argmin(axis=0)
        clocks[gpu, samples] += cost
    totals = clocks.max(axis=0)
    return {
        "remaining_leaves": len(remaining_jobs), "priced_leaves": priced,
        "missing_strata": dict(missing), "repetitions": repetitions,
        "status": "UNMEASURED" if missing else "estimated",
        "p50_seconds": None if missing else float(np.quantile(totals, .5)),
        "p90_seconds": None if missing else float(np.quantile(totals, .9)),
        "priced_subset_p50_seconds": float(np.quantile(totals, .5)),
        "assumptions": "remaining cells cost a complete matched cell; clean starts; paired-unit TP1 scheduling; no video/QA",
    }
