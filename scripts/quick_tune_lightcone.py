"""Excluded 30-second tuning; never freezes recipes or runs the formal DAG."""

import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

from audit_lightcone_timing import freeze
from validate_preview_group import RankCompleteProcess, qa_manifest

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.data import load_prompt_pool
from lightcone_spec.metrics import SAFETY_COUNTERS
from lightcone_spec.preview import preview_jobs
from lightcone_spec.preview_continuation import continuation_lock
from lightcone_spec.quick_tuning import (
    WindowEvidence,
    WindowInterrupted,
    paired_decision,
    run_window,
)
from lightcone_spec.runner import _cell_inputs, _selection_for_job, _speed_metrics
from lightcone_spec.server import apply_runner_affinity, server_session_key
from lightcone_spec.state import StateStore
from lightcone_spec.stride_audit import ALL_STRIDES, audit_job, calibration_split


def safe_metrics(client, *, adaptive, reset=False):
    measured = _speed_metrics(client.server_info(), "tp2_dp1")
    ranks = measured["rank_local"]
    if len(ranks) != 2 or any(r.get(k) != 0 for r in ranks for k in SAFETY_COUNTERS):
        raise RuntimeError("quick-window rank-local safety failed")
    if adaptive:
        if len({r.get("active_version") for r in ranks}) != 1:
            raise RuntimeError("TP ranks disagree on active version")
        if any(r.get("disabled_reason") is not None for r in ranks):
            raise RuntimeError("adaptive runtime disabled")
        if reset and any(r.get("active_version") != 0 or r.get("round") != 0
                         or r.get("updates_published") != 0 for r in ranks):
            raise RuntimeError("adapter/reset state not clear")
        if not reset and any(r.get("updates_published", 0) < 1 for r in ranks):
            raise RuntimeError("adaptive short window published no update")
    if reset and any(r.get("committed_tokens") != 0 for r in ranks):
        raise RuntimeError("request counters not reset")
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("baseline", "compare", "stride"), default="baseline")
    parser.add_argument("--stride", type=int, choices=ALL_STRIDES, default=1)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if (args.phase == "compare") != (args.candidate_config is not None):
        raise ValueError("compare requires exactly one separately verified candidate runtime config")
    original = ExperimentConfig.load(args.config)
    variants = {"old": original}
    if args.candidate_config:
        variants["new"] = ExperimentConfig.load(args.candidate_config)
        old, new = asdict(original), asdict(variants["new"])
        for payload in (old, new):
            payload.pop("sglang_root")
            payload.pop("source")  # YAML file locations necessarily differ.
        if old != new:
            raise ValueError("A/B may differ only in verified runtime, not scientific configuration")
    with sqlite3.connect(f"file:{original.run_dir / 'state.sqlite'}?mode=ro", uri=True) as db:
        if db.execute("pragma integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("formal integrity failed")
        selections = {n: json.loads(v) for n, v in db.execute("select name,value_json from selections")}
    manifest, _ = qa_manifest(selections)
    template = next(j for j in preview_jobs(manifest) if j.backend == "DFLASH" and j.method == "lightcone"
                    and j.model == "Qwen/Qwen3-8B" and j.block == 0)
    split = calibration_split(load_prompt_pool(original.datasets["CalibrationMix"]),
                              [r for rows in manifest["prompts"].values() for r in rows])
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    configs = {}
    for variant, config in variants.items():
        runtime = config.sglang_root
        configs[variant] = {"path": str(runtime), "marker": (runtime / ".lightcone-spec-patched").read_text().strip()}
    registration = {"version": 1, "phase": args.phase, "stride": args.stride,
                    "code": revision, "runtimes": configs, "window_seconds": 30, "warmup_seconds": 10,
                    "tp": 2, "concurrency": 1, "formal_acceptance": False,
                    "local_diff": subprocess.check_output(["git", "diff", "HEAD"], cwd=repo, text=True)}
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output / "registration.json", registration)
    freeze(args.output / "calibration-split.json", split)
    print(json.dumps(registration), flush=True)
    if args.plan_only:
        return
    stopping = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    with continuation_lock(original.run_dir / "preview-continuation.lock"):
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPU has owners; cannot overlap TP2 diagnostic")
        state = StateStore(args.output / "excluded")
        for name, value in selections.items():
            state.set_selection(name, value)
        config = replace(original, results_root=args.output, run_name="excluded")
        apply_runner_affinity(config.gpu_ids, config.run_dir / "numa-affinity.json")
        rows = []
        process, active_runtime = None, None
        try:
            for repeat in range(3 if args.phase == "compare" else 1):
                cases = ([(v, "lightcone", args.stride) for v in (("old", "new") if repeat % 2 == 0 else ("new", "old"))]
                         if args.phase == "compare" else
                         [("old", "static", 1)] + [("old", "lightcone", s) for s in
                         ((1, 10) if args.phase == "baseline" else ALL_STRIDES)])
                for variant, method, stride in cases:
                    for domain in split:
                        identity = f"quick-v1__{args.phase}__{domain}__{method}__s{stride}__r{repeat}__{variant}"
                        logical = args.output / identity
                        if (logical / "result.json").exists():
                            row = json.loads((logical / "result.json").read_text())
                            if row.get("status") not in ("budget_end", "interrupted"):
                                raise RuntimeError("failed window needs diagnosis, not automatic retry")
                            if row["status"] == "budget_end":
                                rows.append(row)
                                continue
                        if stopping.is_set():
                            raise RuntimeError("quick tuning interrupted")
                        if shutil.disk_usage(args.output).free < 12 * 1024**3:
                            raise RuntimeError("less than 12 GiB at window boundary")
                        logical.mkdir(exist_ok=True)
                        directory = logical / f"attempt-{len(list(logical.glob('attempt-*'))) + 1:02d}"
                        directory.mkdir()
                        job = audit_job(template, split, phase="optimization", domain=domain, method=method,
                                        stride=stride, block=repeat, implementation=revision)
                        job = replace(job, job_id=identity, parameters={**job.parameters, "clean_server_per_cell": False})
                        selection = _selection_for_job(state, job)
                        current = replace(variants[variant], results_root=args.output, run_name="excluded")
                        runtime = str(current.sglang_root)
                        if (process is None or runtime != active_runtime
                                or process.session_key != server_session_key(job, selection)):
                            if process is not None:
                                process.stop()
                            server_dir = directory / "server"
                            server_dir.mkdir()
                            os.environ["LIGHTCONE_TIMING_AUDIT"] = json.dumps({"output_directory": str(server_dir.resolve()), "mode": "off"})
                            os.environ.pop("LIGHTCONE_EXCLUDED_VERIFY_TRACE", None)
                            os.environ["PYTHONPATH"] = str(repo / "scripts/timing_hooks") + ":" + str(repo / "src")
                            process = RankCompleteProcess(current, job, gpus=current.gpu_ids[:2],
                                port=current.server.base_port + 60, output_dir=server_dir, selection=selection)
                            active_runtime = runtime
                        started = time.perf_counter()
                        client = process.configure(job, selection)
                        load_seconds = time.perf_counter() - started
                        prepared = time.perf_counter()
                        prompts, budget, metadata = _cell_inputs(current, state, client, job)
                        input_seconds = time.perf_counter() - prepared
                        freeze(directory / "inputs.json", {"job": job.to_dict(), "prompts": prompts, "metadata": metadata})

                        def cleanup(deadline):
                            after = safe_metrics(client, adaptive=method == "lightcone")
                            client.reset(timeout_seconds=max(.001, deadline-time.perf_counter()))
                            reset = safe_metrics(client, adaptive=method == "lightcone", reset=True)
                            return {"after": after, "reset": reset}

                        result = {"status": "failed", "formal_acceptance": False}
                        try:
                            for phase, seconds in (("warmup", 10), ("measure", 30)):
                                safe_metrics(client, adaptive=method == "lightcone", reset=True)
                                evidence = WindowEvidence(seconds)
                                try:
                                    payload = run_window(client, prompts, seconds=seconds, seed=repeat,
                                        prefix=f"{identity}-{phase}", max_new_tokens=budget,
                                        temperature=float(job.parameters.get("temperature", 0.0)),
                                        reset_and_verify=cleanup, evidence=evidence, stop=stopping)
                                finally:
                                    (directory / f"{phase}-events.json").write_text(json.dumps(evidence.report()))
                                (directory / f"{phase}.json").write_text(json.dumps(payload))
                            result = {k: v for k, v in payload.items() if k != "events"}
                            result.update(domain=domain, repeat=repeat, variant=variant, method=method,
                                stride=stride, job_id=identity, configure_seconds=load_seconds,
                                input_prepare_seconds=input_seconds,
                                runtime=configs[variant], execution_gpu_ids=current.gpu_ids[:2],
                                comparison_key={"domain": domain, "repeat": repeat, "stride": stride,
                                                "prompts": prompts, "output_limit": budget, "tp": 2, "c": 1,
                                                "window": 30, "recipe": selection})
                        except BaseException as error:
                            if isinstance(error, WindowInterrupted):
                                result["status"] = "interrupted"
                            result["error"] = f"{type(error).__name__}: {error}"
                            raise
                        finally:
                            (directory / "result.json").write_text(json.dumps(result))
                            (logical / "result.json").write_text(json.dumps(result))
                        rows.append(result)
                        print(json.dumps({k: result[k] for k in ("job_id", "status", "window_goodput", "decode_window_speed")}), flush=True)
                        (args.output / "progress.json").write_text(json.dumps({"completed_windows": len(rows), "last": identity}))
                if args.phase == "compare":
                    decision = paired_decision(rows)
                    (args.output / "decision.json").write_text(json.dumps(decision, indent=2))
                    print(json.dumps(decision), flush=True)
                    if decision["decision"] not in {"repeat", "repeat_promising"}:
                        break
        finally:
            if process is not None:
                process.stop()
        (args.output / "completed.json").write_text(json.dumps({"windows": len(rows), "formal_acceptance": False}))


if __name__ == "__main__":
    main()
