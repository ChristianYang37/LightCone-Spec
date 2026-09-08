"""Excluded calibration audit only; never freezes stride or starts the formal DAG."""

import argparse
import functools
import hashlib
import json
import os
import random
import signal
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path

from validate_preview_group import RankCompleteProcess, qa_manifest

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.data import load_prompt_pool
from lightcone_spec.metrics import SAFETY_COUNTERS
from lightcone_spec.preview import preview_jobs
from lightcone_spec.preview_continuation import continuation_lock
from lightcone_spec.runner import _execute_cell, _selection_for_job
from lightcone_spec.server import apply_runner_affinity
from lightcone_spec.state import StateStore
from lightcone_spec.stride_audit import audit_job, calibration_split, stage_jobs
from lightcone_spec.timing_audit import TimingRecorder, summarize_timing, write_timing_report


class AuditedProcess(RankCompleteProcess):
    """CPU request spans include network wait, never mislabel it as GPU work."""

    def configure(self, job, selection):
        with self.cpu_recorder.span("server_configure"):
            client = super().configure(job, selection)
        if not getattr(client, "_cpu_audited", False):
            for method in ("reset", "tokenize", "server_info", "run_batch", "run_scheduled", "run_timed"):
                if not hasattr(client, method):
                    continue
                original = getattr(client, method)

                def wrap(function, label):
                    @functools.wraps(function)
                    def call(*args, **kwargs):
                        with self.cpu_recorder.span(label):
                            return function(*args, **kwargs)
                    return call

                setattr(client, method, wrap(original, f"client_{method}"))
            client._cpu_audited = True
        return client


def freeze(path, value):
    encoded = json.dumps(value, sort_keys=True, indent=2)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(encoded):
            raise RuntimeError(f"immutable audit input changed: {path.name}")
    else:
        path.write_text(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("baseline", "coarse", "refine", "confirmation"), default="baseline")
    parser.add_argument("--strides", nargs="*", type=int)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--deep-trace", action="store_true", help="Excluded CPU/CUDA activity trace; not a performance sample")
    args = parser.parse_args()
    if args.deep_trace and args.phase != "baseline":
        raise ValueError("deep timeline cannot contaminate stride selection/confirmation")
    original = ExperimentConfig.load(args.config)
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    local_diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=repo, text=True)
    # Record local changes rather than requiring a clean checkout to do research.
    if local_diff:
        revision += "-local-" + hashlib.sha256(local_diff.encode()).hexdigest()[:12]
    with sqlite3.connect(f"file:{original.run_dir / 'state.sqlite'}?mode=ro", uri=True) as source:
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("formal SQLite integrity failed")
        selections = {n: json.loads(v) for n, v in source.execute("SELECT name,value_json FROM selections")}
    manifest, _ = qa_manifest(selections)
    template = next(j for j in preview_jobs(manifest) if j.backend == "DFLASH" and j.method == "lightcone"
                    and j.model == "Qwen/Qwen3-8B" and j.block == 0)
    excluded = [r for rows in manifest["prompts"].values() for r in rows]
    split = calibration_split(load_prompt_pool(original.datasets["CalibrationMix"]), excluded)
    if args.phase == "baseline":
        jobs = []
        # Same four calibration prompts and seed for every method/mode in a block.
        # Full/off pairs quantify observer overhead; both keep native telemetry.
        for domain in split:
            for block in (0, 1):
                unit = [audit_job(template, split, phase="baseline", domain=domain, method=method,
                         stride=stride, block=block, implementation=revision, timing_mode=mode)
                        for method, stride in (("static", 1), ("lightcone", 1), ("lightcone", 10))
                        for mode in ("full", "off")]
                random.Random(f"timing-baseline:{domain}:{block}").shuffle(unit)
                jobs.extend(unit)
        if args.deep_trace:
            jobs = [replace(j, job_id=j.job_id + "__deep", parameters={**j.parameters, "deep_trace": True})
                    for j in jobs if j.parameters["timing_mode"] == "full"]
        jobs = tuple(replace(j, ordinal=i) for i, j in enumerate(jobs))
    else:
        jobs = stage_jobs(template, split, phase=args.phase, implementation=revision, strides=args.strides)
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output / "implementation.json", {"revision": revision, "local_diff": local_diff})
    freeze(args.output / "calibration-split.json", split)
    freeze(args.output / "jobs.json", [j.to_dict() for j in jobs])
    print(json.dumps({"phase": args.phase, "excluded_cells": len(jobs), "tp": 2,
                      "request_count": 8 if args.phase == "confirmation" else 4,
                      "output_limit": 32768, "formal_mutation": False, "eta": "UNMEASURED"}), flush=True)
    if args.plan_only:
        return
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    # The shared lease also excludes the existing continuation controller.
    with continuation_lock(original.run_dir / "preview-continuation.lock"):
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if processes.strip():
            raise RuntimeError("GPU still has owners; do not overlap excluded TP2 audit")
        config = replace(original, results_root=args.output, run_name="excluded")
        state = StateStore(config.run_dir)
        state.recover_interrupted()
        for name, value in selections.items():
            state.set_selection(name, value)
        state.add_internal_jobs(jobs)
        apply_runner_affinity(original.gpu_ids, config.run_dir / "numa-affinity.json")
        count = 0
        for job in jobs:
            if stopping or (args.max_cells is not None and count >= args.max_cells):
                break
            status = state.job_status(job.job_id)
            if status == "completed":
                continue
            if status != "pending":
                raise RuntimeError(f"audit failed cell needs diagnosis, not automatic retry: {job.job_id}")
            server_dir = args.output / "servers" / job.job_id / f"attempt-{state.next_attempt(job.job_id):02d}"
            server_dir.mkdir(parents=True, exist_ok=False)
            os.environ.pop("LIGHTCONE_EXCLUDED_VERIFY_TRACE", None)
            os.environ["LIGHTCONE_TIMING_AUDIT"] = json.dumps({"output_directory": str(server_dir.resolve()),
                                                            "mode": job.parameters["timing_mode"], "deep_trace": args.deep_trace})
            os.environ["PYTHONPATH"] = os.pathsep.join((str(repo / "scripts" / "timing_hooks"), str(repo / "src")))
            selection = _selection_for_job(state, job)
            process = AuditedProcess(config, job, gpus=original.gpu_ids[:2], port=config.server.base_port + 60,
                                          output_dir=server_dir, selection=selection)
            process.cpu_recorder = TimingRecorder(rank=-1)
            try:
                with process.cpu_recorder.span("cell_with_startup_and_teardown"):
                    with process:
                        _execute_cell(config, state, job, gpus=original.gpu_ids[:2], selection=selection, server=process)
                (server_dir / "client-timing.json").write_text(json.dumps(process.cpu_recorder.snapshot()))
                directory = state.completed_attempt_dir(job.job_id)
                if directory is None:
                    raise RuntimeError("audit cell failed; inspect preserved attempt")
                metrics = json.loads((directory / "metrics.json").read_text())
                ranks = metrics.get("rank_local_after", [])
                if (not metrics.get("hard_feasible") or len(ranks) != 2
                        or any(r.get(k) != 0 for r in ranks for k in SAFETY_COUNTERS)):
                    raise RuntimeError("audit correctness/requests failed; no performance acceptance")
                snapshots = [json.loads((server_dir / f"rank-{rank}-timing.json").read_text()) for rank in (0, 1)]
                if any(s["job_id"] != job.job_id for s in snapshots):
                    raise RuntimeError("timing window identity mismatch")
                if job.parameters["timing_mode"] == "full":
                    report = summarize_timing(snapshots, expected_ranks=(0, 1), wall_seconds=metrics["duration_seconds"],
                                              tokens=metrics["committed_tokens"], updates=metrics.get("updates_published"))
                    write_timing_report(report, directory / "timing")
                print(json.dumps({"job": job.job_id, "goodput": metrics["goodput"], "status": "completed"}), flush=True)
                count += 1
            except BaseException as error:
                (server_dir / "audit-failure.json").write_text(json.dumps({"error": str(error), "formal_acceptance": False}))
                raise


if __name__ == "__main__":
    main()
