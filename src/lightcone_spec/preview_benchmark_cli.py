"""Independent preview-benchmark CLI. Never starts the formal DAG or powers GPUs."""

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

from .preview_benchmark import (
    DOMAINS,
    VERSION,
    cached_gate,
    calibrate,
    cases,
    construct_inputs,
    digest,
    freeze,
    sample_sets,
    validate_call_provenance,
    validate_engine_recovery,
    write_report,
)


def prompt_records(value):
    if isinstance(value, dict):
        if isinstance(value.get("prompt"), str):
            yield value
        for child in value.values():
            yield from prompt_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from prompt_records(child)


def checkpoint_inventory(config):
    """Cheap identity guard over the already locally frozen checkpoint files."""
    targets = {"target": config.model_path("Qwen/Qwen3-8B"),
               "drafter": config.draft_path("Qwen/Qwen3-8B", "DFLASH")}
    return {kind: [{"file": str(p.relative_to(root)), "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns}
                   for p in sorted(root.rglob("*")) if p.is_file()
                   and p.suffix in (".json", ".safetensors", ".model", ".txt")]
            for kind, root in targets.items()}


def prepare(args):
    from .config import ExperimentConfig
    from .data import load_prompt_pool
    from .runner import _native_tokenizer
    config = ExperimentConfig.load(args.config)
    pools = {name: load_prompt_pool(config.datasets[name]) for names in DOMAINS.values() for name in names}
    with sqlite3.connect(f"file:{config.run_dir / 'state.sqlite'}?mode=ro", uri=True) as db:
        if db.execute("pragma integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("formal SQLite integrity failed")
        selections = {n: json.loads(v) for n, v in db.execute("select name,value_json from selections")}
    exclusions = list(prompt_records(selections))
    for path in args.exclusions:
        exclusions.extend(prompt_records(json.loads(path.read_text())))
    if not exclusions:
        raise ValueError("existing preview/tuning exclusion evidence required")
    model = "Qwen/Qwen3-8B"
    tokenizer = _native_tokenizer(config.models[model])
    splits = sample_sets(pools, exclusions)
    environment = json.loads(args.environment.read_text())
    required = {"model_revision", "drafter_revision", "runtime", "gpu", "tp", "dtype", "budget", "width", "concurrency"}
    if not required <= environment.keys() or any(environment[k] is None for k in required):
        raise ValueError("missing model/runtime/hardware provenance")
    if environment["tp"] != 2 or environment["concurrency"] != 1:
        raise ValueError("benchmark is matched TP2/c1")
    environment.update(benchmark=VERSION, template=digest(tokenizer.chat_template),
                       tokenizer=digest(tokenizer.get_vocab()),
                       logical_context_tokens=40960, engine_context_tokens=40962,
                       engine_reserved_context_slots=2,
                       checkpoint_inventory=checkpoint_inventory(config),
                       sources={name: digest(list(pool)) for name, pool in pools.items()})
    manifest = {"version": VERSION, "environment": environment,
                "inputs": construct_inputs(tokenizer, splits), "splits": splits,
                "exclusion_digest": digest(exclusions), "generation_tokens": 4096,
                "ignore_eos": True, "temperature": 1, "non_thinking": True}
    cases(manifest, "calibrate")
    cases(manifest, "run")
    freeze(args.output / "manifest.json", manifest)
    print(json.dumps({"status": "prepared", "calibration_calls": 120, "evaluation_calls": 360,
                      "manifest": digest(manifest)}), flush=True)


def benchmark_job(case, manifest, gate):
    from .protocol import Job
    recipe = {"optimizer": "chronobelief", "parameterization": "lora", "rank": 8,
              "scope": "last1", "learning_rate": .001, "schedule": "constant", "stride": 10}
    params = {"context_benchmark_v1": True, "excluded_from_analysis": True,
              "topology": "tp2_dp1", "generation_tokens": case["output_tokens"],
              "execution_request_count": 1, "stride": 10, "prefix_reuse": False,
              "memory_budget_policy": "fixed_reserve_v1", "temperature": 1.0,
              "clean_server_per_cell": False, **recipe}
    if case["mode"] == "gated_s10":
        params["context_gate_v1"] = gate
    return Job(case["id"], "preview-context-benchmark-v1", 0,
               "static" if case["mode"] == "static" else "lightcone", "Qwen/Qwen3-8B", "DFLASH",
               case["domain"], context=40960, width=manifest["environment"]["width"], load="c1",
               gpu_count=2, parameters=params)


def read_results(output):
    rows = []
    for path in sorted(output.glob("calls/*/result.json")):
        row = json.loads(path.read_text())
        if row.get("status") != "completed":
            raise RuntimeError(f"unreviewed failed/interrupted call: {path.parent.name}")
        rows.append(row)
    return rows


def execute(args):
    import shutil

    from .config import ExperimentConfig
    from .metrics import SAFETY_COUNTERS
    from .preview_continuation import continuation_lock
    from .recording import recording_nvml_peaks
    from .runner import _speed_metrics
    from .server import GpuSampler, apply_runner_affinity, server_session_key
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts"))
    from validate_preview_group import RankCompleteProcess

    original = ExperimentConfig.load(args.config)
    manifest = json.loads((args.output / "manifest.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("wrong benchmark manifest")
    env = manifest["environment"]
    if (env.get("logical_context_tokens"), env.get("engine_context_tokens"),
            env.get("engine_reserved_context_slots")) != (40960, 40962, 2):
        raise RuntimeError("benchmark lacks verified engine context headroom; preserve old evidence and review migration")
    if checkpoint_inventory(original) != env["checkpoint_inventory"]:
        raise RuntimeError("checkpoint files changed after preparation")
    budget = {"policy": "fixed_reserve_v1", "adaptation_reserve_mb": original.server.adaptation_reserve_mb,
              "mem_fraction_static": original.server.mem_fraction_static}
    if env["budget"] != budget:
        raise RuntimeError("configured memory budget differs from frozen benchmark")
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
                                       "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if hardware != env["gpu"] or len(original.gpu_ids) != 2:
        raise RuntimeError("GPU environment differs from frozen benchmark")
    marker = (original.sglang_root / ".lightcone-spec-patched").read_text().strip()
    if marker != env["runtime"]:
        raise RuntimeError("runtime differs from calibration/benchmark environment")
    calibration_path = args.output / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    gate, cache_status = cached_gate(calibration, env)
    if args.command == "run" and cache_status == "uncalibrated":
        raise RuntimeError("run requires completed independent calibration")
    pending = cases(manifest, "calibrate" if args.command == "calibrate" else "run")
    if args.command == "qa":
        # Six excluded requests. Threshold 4096 tests crossing vs initially active.
        source = next(r for r in manifest["inputs"] if r["split"] == "calibration" and r["bucket"] == 2)
        pending = [{**source, "id": f"qa-{mode}-{length}", "mode": mode, "input_ids": source["input_ids"][-length:],
                    "input_tokens": length, "output_tokens": 256, "split": "qa"}
                   for mode in ("static", "always_s10", "gated_s10") for length in (4080, 8192)]
        gate = {"threshold": 4096, "max_context": 40960}
        # Short QA cannot detect the native context-boundary output clipping.
        boundary = next(r for r in manifest["inputs"] if r["split"] == "calibration" and r["bucket"] == 9)
        pending.append({**boundary, "id": "qa-static-full-40k-boundary", "mode": "static",
                        "output_tokens": 4096, "split": "qa"})
    # Group mode within each length window to reuse loaded servers; rotate mode order by window.
    pending.sort(key=lambda c: (c["bucket"], (("static", "always_s10", "gated_s10").index(c["mode"])-c["bucket"]) % 3, c["sample"]))
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    provenance = {"commit": commit, "manifest": digest(manifest), "environment": env}
    recovery_path = args.output / "engine-context-recovery.json"
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        validate_engine_recovery(manifest, recovery)
        provenance["engine_context_recovery_v1"] = recovery
        recovered = {r["id"]: r for r in read_results(args.output)}
        for identity in recovery["row_digests"]:
            if identity not in recovered:
                raise RuntimeError("recovery source call missing; do not repeat valid evidence")
            validate_call_provenance(recovered[identity], provenance)
    freeze(args.output / "provenance.json", provenance)
    with continuation_lock(original.run_dir / "preview-continuation.lock"):
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPU owned by another process")
        with sqlite3.connect(f"file:{original.run_dir / 'state.sqlite'}?mode=ro", uri=True) as db:
            if db.execute("select 1 from jobs where status='running' limit 1").fetchone():
                raise RuntimeError("formal runner has active claims")
        config = replace(original, results_root=args.output, run_name="isolated-benchmark")
        apply_runner_affinity(config.gpu_ids, args.output / "numa-affinity.json")
        process = None
        try:
            for case in pending:
                directory = args.output / ("qa" if args.command == "qa" else "calls") / case["id"]
                if (directory / "result.json").exists():
                    prior = json.loads((directory / "result.json").read_text())
                    if prior.get("status") != "completed":
                        raise RuntimeError("existing call failed or has different provenance; explicit review required")
                    validate_call_provenance(prior, provenance)
                    continue
                if directory.exists():
                    raise RuntimeError("incomplete attempt retained; explicit review required")
                if stopping.is_set() or shutil.disk_usage(args.output).free < 12 * 1024**3:
                    raise RuntimeError("stopped at request boundary")
                directory.mkdir(parents=True)
                job = benchmark_job(case, manifest, gate)
                selection = dict(job.parameters) if job.method != "static" else None
                try:
                    start = time.perf_counter()
                    if process is None or process.session_key != server_session_key(job, selection):
                        if process is not None:
                            process.stop()
                        server_dir = directory / "server"
                        server_dir.mkdir()
                        os.environ["LIGHTCONE_EXCLUDED_VERIFY_TRACE"] = json.dumps({
                            "output_directory": str(server_dir), "rank_metrics": True, "trace_verify": False})
                        os.environ["PYTHONPATH"] = str(repo / "scripts/preview_verify_trace") + ":" + str(repo / "src")
                        process = RankCompleteProcess(config, job, gpus=config.gpu_ids[:2],
                            port=config.server.base_port + 80, output_dir=server_dir, selection=selection)
                    client = process.configure(job, selection)
                    setup_seconds = time.perf_counter() - start
                    before = _speed_metrics(client.server_info(), "tp2_dp1")
                    ranks = before["rank_local"]
                    if len(ranks) != 2 or any(r.get(k, 0) != 0 for r in ranks for k in (*SAFETY_COUNTERS, "updates_published")):
                        raise RuntimeError("reset safety/updates not clear on both ranks")
                    sampler = GpuSampler(config.gpu_ids[:2], directory / "gpu.csv")
                    sampler.start()
                    try:
                        requests, duration = client.run_batch((case["input_ids"],), max_new_tokens=case["output_tokens"],
                            seed=case["seed"], temperature=1., request_id_prefix=case["id"], ignore_eos=True)
                        after = _speed_metrics(client.server_info(), "tp2_dp1")
                    finally:
                        sampler.stop()
                    freeze(directory / "requests.json", [r.to_dict() for r in requests])
                    freeze(directory / "telemetry.json", {"before": before, "after": after})
                    r = requests[0]
                    ranks = after["rank_local"]
                    if (len(ranks) != 2 or any(rank.get(k, 0) != 0 for rank in ranks for k in SAFETY_COUNTERS)
                            or r.completion_tokens != case["output_tokens"] or r.input_tokens != case["input_tokens"]):
                        raise RuntimeError("request completion/input/safety failed")
                    if job.method != "static" and len({rank.get("active_version") for rank in ranks}) != 1:
                        raise RuntimeError("TP active versions disagree")
                    calls = [a["target_calls"] - b["target_calls"] for a, b in zip(ranks, before["rank_local"], strict=True)]
                    if min(calls) <= 0 or len(set(calls)) != 1:
                        raise RuntimeError("TP target-call accounting inconsistent")
                    updates = after.get("updates_published", 0) - before.get("updates_published", 0)
                    if case["mode"] == "always_s10" and updates <= 0:
                        raise RuntimeError("always-on LightCone produced no update")
                    if case["mode"] == "gated_s10" and gate["threshold"] is None and updates != 0:
                        raise RuntimeError("no-trigger gate unexpectedly updated")
                    if case["mode"] == "gated_s10":
                        activations = [rank.get("context_gate_activations") for rank in ranks]
                        contexts = [rank.get("context_gate_activation_context") for rank in ranks]
                        if len(set(activations)) != 1 or len(set(contexts)) != 1:
                            raise RuntimeError("TP context gate decisions disagree")
                        if args.command == "qa" and (activations != [1, 1] or updates <= 0
                                or any(c is None or c < gate["threshold"] for c in contexts)):
                            raise RuntimeError("QA gate failed activation/publication")
                    stamps = r.native_token_timestamps_ns
                    if len(stamps) != r.completion_tokens or stamps[-1] <= stamps[0]:
                        raise RuntimeError("native timestamps unavailable")
                    result = {k: case[k] for k in ("id", "mode", "split", "sample", "domain", "bucket", "input_tokens", "output_tokens")}
                    result.update(status="completed", provenance=provenance, throughput=r.completion_tokens/duration,
                        decode_speed=(r.completion_tokens-1)*1e9/(stamps[-1]-stamps[0]),
                        al=(r.completion_tokens-1)/calls[0],
                        al_scope="delivered verify drafts plus bonus; excludes the initial prefill token",
                        prefill_generated_tokens=1, delivered_verify_tokens=r.completion_tokens-1,
                        duration_seconds=duration, setup_reset_seconds=setup_seconds, prefill_ttft_ms=r.ttft_ms,
                        native_first_token_ns=stamps[0], native_last_token_ns=stamps[-1],
                        itl_p99_ms=float(__import__("numpy").quantile(r.inter_token_ms, .99)),
                        updates_published=updates, target_calls=calls[0], execution_gpu_ids=list(config.gpu_ids[:2]),
                        context_gate=[{k: rank.get(k) for k in ("context_gate_activations",
                            "context_gate_activation_context", "context_gate_blocked_rounds")} for rank in ranks],
                        rank_memory=[{k: rank.get(k) for k in ("peak_hbm_bytes", "peak_hbm_reserved_bytes", "kv_token_capacity", "resident_bytes", "peak_bytes")} for rank in ranks],
                        **recording_nvml_peaks(directory / "gpu.csv", config.gpu_ids[:2]))
                    freeze(directory / "result.json", result)
                    print(json.dumps({k: result[k] for k in ("id", "throughput", "al", "updates_published")}), flush=True)
                    if args.command != "qa":
                        write_report(args.output / "report", read_results(args.output), provenance)
                except BaseException as error:
                    freeze(directory / "failure.json", {"status": "failed", "error": str(error), "case": case["id"]})
                    raise
        finally:
            if process is not None:
                process.stop()
    if args.command == "calibrate":
        freeze(calibration_path, calibrate(read_results(args.output), env))
    elif args.command == "run":
        write_report(args.output / "report", read_results(args.output), provenance, complete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "calibrate", "run", "report", "qa"):
        child = commands.add_parser(name)
        child.add_argument("--output", type=Path, required=True)
        if name != "report":
            child.add_argument("--config", type=Path, required=True)
        if name == "prepare":
            child.add_argument("--environment", type=Path, required=True)
            child.add_argument("--exclusions", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "report":
        write_report(args.output / "report", read_results(args.output),
                     json.loads((args.output / "provenance.json").read_text()), complete=True)
    else:
        execute(args)


if __name__ == "__main__":
    main()
