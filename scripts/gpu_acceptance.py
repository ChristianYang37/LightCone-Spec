#!/usr/bin/env python3
"""Manual GPU gates kept outside the four-command paper interface."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.data import load_prompts
from lightcone_spec.metrics import SAFETY_COUNTERS, committed_goodput
from lightcone_spec.protocol import Job
from lightcone_spec.runner import (
    _run_request_scoped,
    _speed_metrics,
    _validate_committed_tokens,
)
from lightcone_spec.server import ReplicaServerProcess, ServerProcess, StickyReplicaClient


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_token_accounting(results, committed: int, budget: int) -> None:
    if any(result.completion_tokens != budget for result in results):
        raise RuntimeError("generation did not honor the output-token budget")
    _validate_committed_tokens(results, committed)


def _job(
    ordinal: int,
    method: str,
    backend: str,
    *,
    block: int,
    model: str = "Qwen/Qwen3-8B",
    tp2: bool = False,
    dp2: bool = False,
) -> Job:
    if tp2 and dp2:
        raise ValueError("TP2 and DP2 are separate acceptance cases")
    parameters: dict[str, object] = {
        "regime": "short_input_long_generation",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 0.0,
        "parameterization": "full",
        "scope": "all",
        "stride": 10,
    }
    if tp2:
        parameters["topology"] = "tp2_dp1"
    if dp2:
        parameters["topology"] = "two_replica_tp1_dp2"
    return Job(
        job_id=f"gpu-acceptance-{ordinal:03d}-{method}",
        node="E6-acceptance" if backend == "NEXTN" else "gpu-acceptance",
        ordinal=ordinal,
        method=method,
        model=model,
        backend=backend,
        task="controlled_baseline",
        context=40928,
        load="c8",
        width=None if method == "target_only" else 16,
        block=block,
        gpu_count=2 if tp2 or dp2 or backend == "NEXTN" else 1,
        parameters=parameters,
    )


def _measure(
    config: ExperimentConfig,
    job: Job,
    output: Path,
    *,
    max_new_tokens: int,
    exactness_tokens: int = 0,
) -> dict[str, object]:
    gpus = config.gpu_ids if job.gpu_count == 2 else (config.gpu_ids[0],)
    port = config.server.base_port + (2 if job.gpu_count == 2 else 0)
    output.mkdir(parents=True, exist_ok=True)
    process_type = (
        ReplicaServerProcess
        if job.parameters.get("topology") == "two_replica_tp1_dp2"
        else ServerProcess
    )
    process = process_type(
        config,
        job,
        gpus=gpus,
        port=port,
        output_dir=output,
        selection=None,
    )
    with process as client:
        raw = load_prompts(
            config.dataset_path("controlled_baseline"),
            limit=16,
            split="tuning",
            offset=(job.block or 0) * 16,
        )
        prompts = tuple(tuple(client.tokenize(prompt)[-128:]) for prompt in raw)
        if isinstance(client, StickyReplicaClient):
            client.warmup(prompts[0], max_new_tokens=16, seed=0)
        else:
            client.run_batch(prompts[:1], max_new_tokens=16, seed=0)
            client.reset()
        exactness_trajectory = None
        if exactness_tokens:
            exactness, _ = client.run_batch(
                prompts[:1],
                max_new_tokens=exactness_tokens,
                seed=job.block or 0,
                request_id_prefix="exactness",
            )
            exactness_trajectory = list(exactness[0].output_ids)
            client.reset()
        topology = str(job.parameters.get("topology", "tp1_dp1"))
        before = _speed_metrics(client.server_info(), topology)
        if isinstance(client, StickyReplicaClient):
            routing_keys = tuple(f"cohort-{index % 4}" for index in range(len(prompts)))
            run = client.run_scheduled(
                prompts,
                (0.0,) * len(prompts),
                max_new_tokens=max_new_tokens,
                seed=job.block or 0,
                routing_keys=routing_keys,
                max_in_flight=8,
                deadline_seconds=config.server.request_timeout_seconds,
                drain_seconds=config.server.request_timeout_seconds,
            )
            results, elapsed = run.results, run.elapsed_seconds
            if len(run.outcomes) != len(prompts) or any(
                outcome.status != "completed" for outcome in run.outcomes
            ):
                raise RuntimeError("DP2 acceptance did not complete every request")
        elif job.method in {"tts", "l0_naive"}:
            results, elapsed = _run_request_scoped(
                client,
                prompts,
                max_new_tokens,
                job.block or 0,
                request_prefix=f"acceptance-{job.method}",
            )
        else:
            results, elapsed = client.run_batch(
                prompts,
                max_new_tokens=max_new_tokens,
                seed=job.block or 0,
            )
        after = _speed_metrics(client.server_info(), topology)
    intervals = [value for result in results for value in result.inter_token_ms]
    committed = int(after["committed_tokens"]) - int(before["committed_tokens"])
    _validate_token_accounting(results, committed, max_new_tokens)
    counters = {
        name: int(after[name]) - int(before[name]) for name in SAFETY_COUNTERS
    }
    row: dict[str, object] = {
        "block": job.block,
        "method": job.method,
        "backend": job.backend,
        "goodput": committed_goodput(committed, elapsed),
        "p99_itl_ms": float(np.quantile(intervals, 0.99)),
        "peak_hbm_bytes": int(after["peak_hbm_bytes"]),
        "committed_tokens": committed,
        "trajectories": [list(result.output_ids) for result in results],
        "exactness_trajectory": exactness_trajectory,
        "counters": counters,
        "rank_local": after["rank_local"],
        "rank_aggregates": after["rank_aggregates"],
    }
    if any(counters.values()):
        raise RuntimeError(f"{job.method} reported nonzero safety counters: {counters}")
    if job.method not in {"target_only", "static"} and int(after["updates_published"]) < 1:
        raise RuntimeError(f"{job.method} did not publish an update")
    if job.backend == "DSPARK":
        for name in ("confidence_brier", "confidence_ece"):
            value = after.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"DSpark did not report finite {name}")
    _write(output / "acceptance.json", row)
    return row


def benchmark(args: argparse.Namespace) -> None:
    config = ExperimentConfig.load(args.config)
    config.validate_local_paths()
    output = args.output
    rows = []
    methods = (("static", "DFLASH"), ("tts", "DFLASH"), ("l0_naive", "DFLASH"))
    for block in range(3):
        block_rows = []
        for ordinal, (method, backend) in enumerate(methods, start=block * len(methods)):
            job = _job(ordinal, method, backend, block=block)
            block_rows.append(
                _measure(
                    config,
                    job,
                    output / f"block-{block}" / method,
                    max_new_tokens=args.max_new_tokens,
                )
            )
        baseline = block_rows[0]["trajectories"]
        if any(row["trajectories"] != baseline for row in block_rows[1:]):
            raise RuntimeError(f"block {block} token trajectories differ")
        rows.extend(block_rows)
    _write(output / "benchmark.json", rows)


def smoke(args: argparse.Namespace) -> None:
    config = ExperimentConfig.load(args.config)
    config.validate_local_paths()
    cases = (
        _job(0, "target_only", "NONE", block=0),
        _job(1, "static", "DFLASH", block=0),
        _job(2, "tts", "DFLASH", block=0),
        _job(3, "l0_naive", "DFLASH", block=0),
        _job(4, "lightcone", "DFLASH", block=0),
        _job(5, "onlinespec_ogd", "DFLASH", block=0),
        _job(6, "lightcone", "DFLASH", block=0, tp2=True),
        _job(7, "lightcone", "DFLASH", block=0, dp2=True),
        _job(8, "lightcone", "DSPARK", block=0),
        _job(
            9,
            "lightcone",
            "NEXTN",
            block=0,
            model="Qwen/Qwen3.6-35B-A3B",
            tp2=True,
        ),
        _job(
            10,
            "lightcone",
            "NEXTN",
            block=0,
            model="Qwen/Qwen3.5-122B-A10B-FP8",
            tp2=True,
        ),
    )
    rows = []
    for job in cases:
        rows.append(
            _measure(
                config,
                job,
                args.output / f"{job.ordinal:02d}-{job.method}-{job.backend.lower()}",
                max_new_tokens=args.max_new_tokens,
                exactness_tokens=args.max_new_tokens,
            )
        )
        time.sleep(1)
    baseline = rows[0]["exactness_trajectory"]
    for row in rows[1:6]:
        if row["exactness_trajectory"] != baseline:
            raise RuntimeError(f"{row['method']} smoke trajectory differs from Target-only")
    _write(args.output / "smoke.json", rows)


def compare(args: argparse.Namespace) -> None:
    donor = json.loads(args.donor.read_text(encoding="utf-8"))
    rebuild = json.loads(args.rebuild.read_text(encoding="utf-8"))
    failures = []
    for method in ("static", "tts", "l0_naive"):
        old = [row for row in donor if row["method"] == method]
        new = [row for row in rebuild if row["method"] == method]
        if len(old) != 3 or len(new) != 3:
            failures.append(f"{method}: expected three donor and rebuild blocks")
            continue
        old_goodput = statistics.median(float(row["goodput"]) for row in old)
        new_goodput = statistics.median(float(row["goodput"]) for row in new)
        old_itl = statistics.median(float(row["p99_itl_ms"]) for row in old)
        new_itl = statistics.median(float(row["p99_itl_ms"]) for row in new)
        old_hbm = max(int(row["peak_hbm_bytes"]) for row in old)
        new_hbm = max(int(row["peak_hbm_bytes"]) for row in new)
        if new_goodput < old_goodput * 0.97:
            failures.append(f"{method}: goodput regressed by more than 3%")
        if new_itl > old_itl * 1.05:
            failures.append(f"{method}: p99 ITL regressed by more than 5%")
        if new_hbm > old_hbm + 512 * 1024 * 1024:
            failures.append(f"{method}: peak HBM increased by more than 512 MiB")
        old_by_block = {int(row["block"]): row["trajectories"] for row in old}
        new_by_block = {int(row["block"]): row["trajectories"] for row in new}
        if old_by_block != new_by_block:
            failures.append(f"{method}: donor/rebuild token trajectories differ")
        for label, rows in (("donor", old), ("rebuild", new)):
            if any(any(row["counters"].values()) for row in rows):
                failures.append(f"{method}: {label} has nonzero safety counters")
    report = {"passed": not failures, "failures": failures}
    _write(args.output, report)
    if failures:
        raise SystemExit("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler, default_tokens in (
        ("benchmark", benchmark, 4096),
        ("smoke", smoke, 128),
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--max-new-tokens", type=int, default=default_tokens)
        command.set_defaults(handler=handler)
    command = commands.add_parser("compare")
    command.add_argument("--donor", type=Path, required=True)
    command.add_argument("--rebuild", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(handler=compare)
    args = parser.parse_args()
    if hasattr(args, "max_new_tokens") and not 1 <= args.max_new_tokens <= 40800:
        parser.error("--max-new-tokens must be in 1..40800")
    args.handler(args)


if __name__ == "__main__":
    main()
