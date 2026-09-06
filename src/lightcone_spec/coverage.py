"""Append-only coverage repair plans, separate from source/diagnostic additions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .protocol import Job, materialize, mechanism_jobs, memory_budget_policy, source_coverage_jobs

COMPATIBILITY_NODE = "coverage-compatibility-v1"
DENSE_14B = "Qwen/Qwen3-14B"


def request_budget_eta(jobs, evidence, *, request_counts, resources, seed=0, repetitions=10000):
    """No cross-output/topology extrapolation or old-small-cell duration reuse.

    Request-linear extrapolation is labelled explicitly; unknown strata keep
    the total ETA unmeasured rather than inheriting unrelated global timings.
    Cells are supplied in execution order. A resource count of two also denotes
    a TP1 cell requiring clean dual-device isolation.
    """
    def stratum(job, policy):
        p = job.parameters
        return (job.model, job.backend, job.method, p.get("topology", "tp1_dp1"),
                job.load, p.get("generation_tokens"), p.get("panel"),
                p.get("regime"), bool(p.get("respect_eos")), policy)

    pools = {}
    for item, metrics in evidence:
        job = Job(**item) if isinstance(item, dict) else item
        p = job.parameters
        if (p.get("excluded_from_analysis") or p.get("probe")
                or metrics.get("hard_feasible") is not True):
            continue
        duration, count = metrics.get("duration_seconds"), metrics.get("request_count")
        startup = metrics.get("session_startup_seconds")
        if not all(isinstance(value, (int, float)) and np.isfinite(value)
                   for value in (duration, count, startup)):
            continue
        if duration <= 0 or count <= 0 or startup < 0:
            continue
        pools.setdefault(stratum(job, metrics.get("memory_budget_policy", "fixed_reserve_v1")),
                         []).append((duration / count, startup))
    rng = np.random.default_rng(seed)
    clocks = np.zeros((2, repetitions), dtype=float)
    affinities, missing, priced = {}, {}, 0
    samples = np.arange(repetitions)
    for job in jobs:
        key = stratum(job, memory_budget_policy(job))
        pool = pools.get(key)
        if not pool:
            missing[str(key)] = missing.get(str(key), 0) + 1
            continue
        values = np.asarray(pool)[rng.integers(len(pool), size=repetitions)]
        # Charging a clean server per cell is conservative for reused sessions.
        cost = values[:, 0] * request_counts[job.job_id] + values[:, 1]
        if resources[job.job_id] == 2:
            clocks[:] = clocks.max(axis=0) + cost
        else:
            affinity = (job.node, job.block) if job.block is not None else None
            gpu = affinities.get(affinity) if affinity is not None else None
            if gpu is None:
                gpu = clocks.argmin(axis=0)
                if affinity is not None:
                    affinities[affinity] = gpu
            clocks[gpu, samples] += cost
        priced += 1
    totals = clocks.max(axis=0)
    return {
        "status": "UNMEASURED" if missing else "estimated",
        "remaining_leaves": len(jobs), "priced_leaves": priced,
        "missing_strata": missing, "registered_requests": sum(request_counts.values()),
        "p50_seconds": None if missing else float(np.quantile(totals, .5)),
        "p90_seconds": None if missing else float(np.quantile(totals, .9)),
        "priced_subset_p50_seconds": float(np.quantile(totals, .5)),
        "bootstrap_repetitions": repetitions,
        "basis": "matched output/load/topology/panel; request-linear cost; clean server per cell",
        "scope_note": "unpriced work is omitted only from the explicitly labelled priced subset",
    }


def gpu_acceptance_jobs() -> tuple[Job, ...]:
    """Excluded, fixed short budgets; never overwrite a registered formal row."""
    jobs = []
    for original in materialize("E0-tune"):
        if not original.parameters.get("pair_calibration"):
            continue
        dense = original.model == DENSE_14B
        phase = "dense14" if dense else ("gemma" if original.model == "Gemma4-12B" else "qwen")
        jobs.append(
            replace(
                original,
                job_id=f"qa-coverage__{original.job_id}",
                task="MATH-500",
                context=40960,
                load="c1",
                block=None,
                gpu_count=2 if dense else 1,
                parameters={
                    **original.parameters,
                    "coverage_runtime": True,
                    "topology": "tp2_dp1" if dense else "tp1_dp1",
                    "qa_phase": phase,
                    "regime": "short_input_long_generation",
                    "generation_tokens": 512,
                    "execution_request_count": 2,
                    "minimum_updates": 2,
                    "temperature": 0.0,
                    "excluded_from_analysis": True,
                    "clean_server_per_cell": True,
                },
            )
        )
    dense_target = replace(
        jobs[0],
        job_id="qa-coverage-dense14-target",
        method="target_only",
        model=DENSE_14B,
        backend="NONE",
        gpu_count=2,
        parameters={**jobs[0].parameters, "qa_phase": "dense14", "topology": "tp2_dp1"},
    )
    jobs.append(dense_target)
    for original in tuple(jobs):
        if (
            original.model == "Gemma4-12B"
            and original.backend == "EAGLE3"
            and original.method in {"static", "lightcone"}
        ):
            jobs.append(
                replace(
                    original,
                    job_id=original.job_id + "-tp2",
                    gpu_count=2,
                    parameters={**original.parameters, "topology": "tp2_dp1"},
                )
            )
    for node_jobs in (source_coverage_jobs(), mechanism_jobs()):
        original = next(
            job
            for job in node_jobs
            if job.model == "Qwen/Qwen3-8B"
            and job.backend == "DFLASH"
            and job.method == "lightcone"
            and job.task == "AIME-2025"
        )
        jobs.append(
            replace(
                original,
                job_id=f"qa-coverage__{original.job_id}",
                block=None,
                parameters={
                    **original.parameters,
                    "qa_phase": "panels",
                    "execution_request_count": 2,
                    "generation_tokens": 512,
                    "position_bin_tokens": 128,
                    "excluded_from_analysis": True,
                    "statistical_unit": "excluded_runtime_acceptance",
                },
            )
        )
    return tuple(jobs)


def memory_pressure_job(case: str) -> Job:
    """Excluded stress cases, separate from the 41 short acceptance cases."""
    if case not in {"long_tts", "long_lightcone", "long_gemma_lightcone", "retraction", "tp2"}:
        raise ValueError(case)
    method = "tts" if case == "long_tts" else "lightcone"
    model = "Gemma4-12B" if case == "long_gemma_lightcone" else "Qwen/Qwen3-8B"
    source = next(j for j in gpu_acceptance_jobs()
                  if j.model == model and j.backend == "DFLASH" and j.method == method)
    return replace(
        source, job_id=("qa-memory-retraction-native-hook-v1" if case == "retraction" else f"qa-memory-{case}"),
        load="c8" if case == "retraction" else "c1",
        gpu_count=2 if case == "tp2" else 1,
        parameters={
            **source.parameters, "memory_pressure_case": case,
            "topology": "tp2_dp1" if case == "tp2" else "tp1_dp1",
            "generation_tokens": 8192 if case == "retraction" else 32768,
            "execution_request_count": 8 if case == "retraction" else 2,
            "respect_eos": False,
            **({"qa_kv_token_cap": 41024, "qa_native_retraction_interval": 500}
               if case == "retraction" else {}),
        },
    )


def compatibility_replacements(rows: list[tuple[Job, dict[str, Any]]]) -> tuple[Job, ...]:
    """Replace proven registration faults per method, not an entire backend.

    A genuine TP1 capacity result remains valid at TP1; the requested dense
    transfer is a new TP2 measurement with an explicit link to that result.
    Runtime-error text is evidence for a retry, never evidence of feasibility.
    """
    output = []
    for job, metrics in rows:
        if not job.parameters.get("pair_calibration"):
            continue
        reason = None
        error = str(metrics.get("error", ""))
        if job.model == DENSE_14B and job.parameters.get("topology", "tp1_dp1") != "tp2_dp1":
            reason = "dense_14b_move_to_tp2"
        elif any(
            text in error
            for text in (
                "is not a registered model",
                "Cannot find model module",
                "require the base DFlashDraftModel",
            )
        ):
            reason = "official_draft_registration_or_adapter"
        elif job.backend == "DSPARK" and job.method in {"tts", "lightcone"}:
            # Legacy native teacher indexing mixed RID bonus rows and masked
            # with verify lengths including that bonus. These are not valid
            # updated-model measurements even when inference finished.
            reason = "native_teacher_rid_and_owned_mask"
        if reason is None:
            continue
        dense = job.model == DENSE_14B
        output.append(
            replace(
                job,
                job_id=f"{COMPATIBILITY_NODE}__{job.job_id}",
                node=COMPATIBILITY_NODE,
                gpu_count=2 if dense else 1,
                parameters={
                    **job.parameters,
                    "source_node": "E0-tune",
                    "replaces_job_id": job.job_id,
                    "reconciliation_kind": reason,
                    "coverage_runtime": True,
                    "topology": "tp2_dp1" if dense else "tp1_dp1",
                    "evidence_owner": "E6" if dense else "E0",
                    "panel": "dense_14b_transfer" if dense else "coverage_compatibility",
                },
            )
        )
    return tuple(output)


def method_feasibility(rows: list[tuple[Job, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """One auditable outcome per model/backend/method/topology; no pair pruning."""
    output = {}
    for job, metrics in rows:
        if not job.parameters.get("pair_calibration"):
            continue
        key = "|".join(
            (job.model, job.backend, job.method, job.parameters.get("topology", "tp1_dp1"))
        )
        if key in output:
            raise ValueError(f"duplicate logical calibration evidence: {key}")
        output[key] = {
            "hard_feasible": metrics.get("hard_feasible") is True,
            "scientific_outcome": metrics.get("scientific_outcome"),
            "source_job_id": job.job_id,
            "source_attempt_dir": metrics.get("source_attempt_dir"),
        }
    return output


def dense_14b_transfer(job: Job) -> Job:
    """E6-owned copy of an E0 scientific cell; its workload and seed are unchanged."""
    if job.model != DENSE_14B or not job.node.startswith("E0"):
        raise ValueError("dense migration applies only to the registered E0 Qwen3-14B cells")
    if job.method not in {"target_only", "static", "tts", "lightcone"}:
        raise ValueError("dense migration does not extend the registered method surface")
    return replace(
        job,
        job_id=f"dense-14b-transfer-v1__{job.job_id}",
        node="dense-14b-transfer-v1",
        gpu_count=2,
        parameters={
            **job.parameters,
            "source_node": job.node,
            "evidence_owner": "E6",
            "panel": "dense_14b_transfer",
            "topology": "tp2_dp1",
            "replaces_job_id": job.job_id,
            "coverage_runtime": True,
            "reconciliation_kind": "dense_14b_move_to_tp2",
        },
    )
