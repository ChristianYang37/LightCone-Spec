"""Registered, append-only four-block preview; never a replacement for main effects.

New contributions: LicenseRef-LightCone-Source-Available-1.0. See LICENSE.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import t

from .protocol import Job
from .scheduling import logical_unit_key

PREVIEW_NODES = ("E3b-preview-v1", "E5-preview-v1", "Qwen38-preview-v1")
QWEN38_MODEL = "Qwen/Qwen3.8-27B"
QWEN38_CHECKPOINTS = {
    "target": {"repo": QWEN38_MODEL, "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"},
    "NEXTN": {"repo": QWEN38_MODEL, "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"},
    "DSPARK": {"repo": "RadixArk/Qwen3.8-27B-DSpark", "revision": "b9a5dbdf03bc999c6c73c426b19c2d9041cea393"},
    "DFLASH": {"repo": "incoai/Qwen3.8-27B-DFlash2", "revision": "dedf8df68adfb1afeaf7b7480c0a0243108177b4"},
}
QWEN38_METHODS = (
    ("NONE", "target_only", "Target-only"),
    ("NEXTN", "static", "Native MTP"),
    ("DSPARK", "static", "Community DSpark"),
    ("DFLASH", "static", "Community DFlash2"),
    ("DFLASH", "tts", "Full TTS (DFlash2 transfer)"),
    ("DFLASH", "lightcone", "LightCone (DFlash2 transfer)"),
)
QWEN38_VIDEO_METHODS = tuple(
    (backend, "tts_lora_batched", "TTS-LoRA-Batched (DFlash2)") if method == "tts"
    else (backend, method, label) for backend, method, label in QWEN38_METHODS
)
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
    topologies = manifest.get("comparison_topologies", {})
    for i, job in enumerate(jobs):
        group = "dflash_long" if job.backend == "DFLASH" else "dspark_serving"
        tp = topologies.get(group, 1)
        if type(tp) is not int or tp not in (1, 2):
            raise ValueError("preview comparison TP must be 1 or 2")
        if tp == 2:
            params = {**job.parameters, "topology": "tp2_dp1", "replaces_job_id": job.job_id,
                      "replacement_reason": "matched comparison topology TP2"}
            if job.load == "burstgpt_shape":
                anchor = manifest.get("tp2_trace_anchor")
                if not anchor or anchor.get("topology") != "tp2_dp1" or not anchor.get("source_job_ids"):
                    raise ValueError("TP2 BurstGPT requires a measured topology-matched anchor")
                params["preview_anchor"] = anchor
            jobs[i] = replace(job, job_id=job.job_id + "__tp2", gpu_count=2, parameters=params)
    if manifest.get("qwen38") is not None:
        jobs.extend(qwen38_jobs(manifest))
    return tuple(jobs)


def select_common_tp(rows: list[dict], expected_cases: set[str]) -> int:
    """Only a complete, correct matched acceptance panel can freeze a TP."""
    if not expected_cases:
        raise ValueError("empty topology acceptance panel")
    for tp in (1, 2):
        selected = [r for r in rows if r.get("tp") == tp]
        by_case = {r["case"]: r for r in selected}
        if len(by_case) != len(selected):
            raise ValueError("duplicate topology acceptance case")
        if set(by_case) != expected_cases:
            continue
        if all(r.get("correct") is True and r.get("full_workload") is True
               and r.get("reset_verified") is True
               and r.get("gpu_binding_verified") is True for r in selected):
            return tp
    raise ValueError("no fully validated common TP; preserve capacity/error evidence")


def qwen38_jobs(manifest: dict) -> tuple[Job, ...]:
    spec = manifest["qwen38"]
    for name in ("tts_recipe", "lightcone_recipe"):
        recipe = manifest[name]
        if recipe.get("stride") != 10 or (name == "tts_recipe" and recipe.get("lr") != 1e-4):
            raise ValueError("27B transfer must retain frozen S10 and TTS lr=1e-4")
    tp = spec["tp"]
    if type(tp) is not int or tp not in (1, 2):
        raise ValueError("Qwen3.8 requires common TP1 or TP2")
    checkpoints = spec["checkpoints"]
    expected = {"target": QWEN38_MODEL, "NEXTN": QWEN38_MODEL,
                "DFLASH": "incoai/Qwen3.8-27B-DFlash2",
                "DSPARK": "RadixArk/Qwen3.8-27B-DSpark"}
    for key, repo in expected.items():
        record = checkpoints[key]
        revision = record.get("revision", "")
        if record.get("repo") != repo or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Qwen3.8 checkpoint needs an exact repository and revision")
    if checkpoints["target"] != checkpoints["NEXTN"]:
        raise ValueError("native MTP must use the same target checkpoint revision")
    records = spec["prompts"]
    if len(records) != 8 or len({r["prompt"] for r in records}) != 8:
        raise ValueError("Qwen3.8 needs eight distinct frozen held-out prompts")
    jobs = []
    for block in range(4):
        methods = list(QWEN38_METHODS)
        random.Random(f"qwen38-preview-v1:{block}").shuffle(methods)
        for backend, method, label in methods:
            recipe = manifest["tts_recipe" if method == "tts" else "lightcone_recipe"] if method in {"tts", "lightcone"} else None
            jobs.append(Job(
                job_id=f"qwen38-preview-v1__{backend}__{method}__b{block}__tp{tp}",
                node=PREVIEW_NODES[2], ordinal=56 + len(jobs), model=QWEN38_MODEL,
                backend=backend, method=method, task="LiveCodeBench", context=17408,
                load="c1", block=block, gpu_count=tp,
                width=None if backend == "NONE" else (4 if backend == "NEXTN" else 8),
                parameters={
                    "panel": "preview_v1", "preview_panel": "qwen38_transfer", "method_label": label,
                    "pairing_key": f"qwen38-preview-v1:{block}", "comparison_backend": "Qwen38-matched",
                    "topology": f"tp{tp}_dp1", "memory_budget_policy": "method_peak_v1",
                    "coverage_runtime": True, "checkpoint_provenance": checkpoints,
                    "sampling_seed": block, "stride": 10, "frozen_recipe": recipe,
                    "execution_request_count": 8, "preview_prompt_records": records,
                    "regime": "preview_constructed_chat", "generation_tokens": 1024,
                    "input_tokens": 16384, "respect_eos": True, "enable_thinking": False,
                    "temperature": 1.0, "verification": "native_scheduler",
                    "workload": "preview_formal_supplement",
                },
            ))
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
        measured[identity]["topology"] = config.get("parameters", {}).get("topology", "tp1_dp1")
        measured[identity]["memory_budget_policy"] = config.get("parameters", {}).get("memory_budget_policy", "fixed_reserve_v1")
    rows, groups = [], defaultdict(dict)
    for identity, job in expected.items():
        m = measured.get(identity)
        row = {
            "job_id": identity, "panel": job.parameters["preview_panel"], "task": job.task,
            "method": job.method, "backend": job.backend, "load": job.load, "block": job.block,
            "model": job.model, "topology": job.parameters.get("topology", "tp1_dp1"),
            "method_label": job.parameters.get("method_label", job.method),
            "status": "UNMEASURED" if m is None else (
                "measured" if m.get("hard_feasible") is True else "unavailable"
            ), "metrics": m,
        }
        rows.append(row)
        if row["status"] == "measured":
            if m["topology"] != row["topology"]:
                raise ValueError("preview evidence topology differs from frozen comparison")
            groups[(row["panel"], job.task, job.load, job.model)][(job.method, job.backend, job.block)] = m
    effects = []
    for condition, data in sorted(groups.items()):
        backend = "DFLASH" if condition[0] in {"long_generation", "qwen38_transfer"} else "DSPARK"
        baselines = (("static", backend), ("tts", backend)) if condition[0] == "long_generation" else (("static", backend),)
        if condition[0] == "qwen38_transfer":
            baselines = tuple((method, b) for b, method, _ in QWEN38_METHODS if method != "lightcone")
        for (baseline, baseline_backend), metric in itertools.product(
            baselines,
            ("goodput", "accepted_drafts_per_target_call", "per_user_generation_speed"),
        ):
            ratios = []
            for block in range(4):
                a, b = data.get(("lightcone", backend, block), {}), data.get((baseline, baseline_backend, block), {})
                x, y = a.get(metric), b.get(metric)
                if (a.get("execution_policy") != b.get("execution_policy")
                        or a.get("effective_load") != b.get("effective_load")
                        or a.get("topology") != b.get("topology")
                        or a.get("memory_budget_policy") != b.get("memory_budget_policy")):
                    continue
                if all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (x, y)):
                    ratios.append(math.log(x / y))
            complete = len(ratios) == 4
            mean = float(np.mean(ratios)) if complete else None
            radius = float(t.ppf(.975, 3) * np.std(ratios, ddof=1) / 2) if complete else None
            effects.append({
                "condition": list(condition), "baseline": baseline, "metric": metric,
                "baseline_backend": baseline_backend,
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
        return (job.model, job.backend, job.method, job.task, job.load, job.context,
                job.parameters.get("generation_tokens"), job.parameters.get("topology", "tp1_dp1"),
                job.parameters["preview_panel"],
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
        if any(j.parameters.get("topology") == "tp2_dp1" for j in jobs):
            clocks[:] = clocks.max(axis=0) + cost
        else:
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
        "assumptions": "matched model/topology/budget/execution strata; whole paired units; TP2 reserves both GPUs; no video/QA",
    }
