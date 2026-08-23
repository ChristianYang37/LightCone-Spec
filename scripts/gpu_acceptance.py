#!/usr/bin/env python3
"""Manual GPU gates kept outside the four-command paper interface."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pickle
import signal
import statistics
import subprocess
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.data import load_prompts
from lightcone_spec.metrics import SAFETY_COUNTERS, committed_goodput
from lightcone_spec.protocol import Job
from lightcone_spec.runner import (
    _speed_metrics,
    _validate_committed_tokens,
    _validate_greedy_verify_counts,
)
from lightcone_spec.server import (
    GpuSampler,
    ReplicaServerProcess,
    ServerProcess,
    StickyReplicaClient,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _post_json(url: str, value: object, timeout: float) -> object:
    request = urllib.request.Request(
        url,
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _wait_health(url: str, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"adapter-batching server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError("adapter-batching server did not become healthy")


def _adapter_tensors(model_path: Path, adapter_index: int):
    import torch
    from transformers import AutoConfig

    model = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text = model.get_text_config() if hasattr(model, "get_text_config") else model
    layers = int(text.num_hidden_layers)
    hidden = int(text.hidden_size)
    head_dim = int(getattr(text, "head_dim", hidden // text.num_attention_heads))
    q_out = int(text.num_attention_heads) * head_dim
    v_out = int(text.num_key_value_heads) * head_dim
    generator = torch.Generator().manual_seed(adapter_index)
    tensors = {}
    for layer in range(layers):
        prefix = f"base_model.model.model.layers.{layer}.self_attn"
        for name, output in (("q_proj", q_out), ("v_proj", v_out)):
            a = torch.randn((1, hidden), generator=generator, dtype=torch.float32)
            a.div_(math.sqrt(hidden))
            b = torch.zeros((output, 1), dtype=torch.float32)
            if adapter_index:
                b.fill_(adapter_index * 0.01)
            tensors[f"{prefix}.{name}.lora_A.weight"] = a
            tensors[f"{prefix}.{name}.lora_B.weight"] = b
    return tensors


def _portable_tensor_payload(tensors: object) -> str:
    """Serialize CPU tensors as bytes instead of multiprocessing file descriptors."""
    payload = pickle.dumps(tensors, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(payload).decode("ascii")


def _adapter_observation(row: dict[str, object]) -> dict[str, object]:
    meta = row.get("meta_info")
    if not isinstance(meta, dict):
        raise RuntimeError("adapter-batching response has no meta_info")
    raw_logprobs = meta.get("output_token_logprobs")
    if not isinstance(raw_logprobs, list):
        raise RuntimeError("adapter-batching response has no output logprobs")
    events = meta.get("native_token_timestamp_events")
    if not isinstance(events, list):
        raise RuntimeError("adapter-batching response has no native timestamps")
    return {
        "rid": str(meta["id"]),
        "output_ids": [int(value) for value in row["output_ids"]],
        "logprobs": [float(value[0]) for value in raw_logprobs],
        "timestamps_ns": [int(value["committed_ns"]) for value in events],
    }


def _observations_match(left: dict[str, object], right: dict[str, object]) -> bool:
    if left["output_ids"] != right["output_ids"]:
        return False
    a = np.asarray(left["logprobs"], dtype=np.float64)
    b = np.asarray(right["logprobs"], dtype=np.float64)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=0.02, rtol=0.02))


def _adapter_generate(
    base_url: str,
    prompt: str,
    adapters: tuple[str | None, ...],
    label: str,
    max_new_tokens: int,
    timeout: float,
) -> tuple[list[dict[str, object]], float]:
    rids = [f"{label}-{index:02d}" for index in range(len(adapters))]
    body = {
        "text": [prompt] * len(adapters),
        "lora_path": list(adapters),
        "rid": rids,
        "sampling_params": [
            {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
                "sampling_seed": 0,
            }
            for _ in adapters
        ],
        "return_logprob": True,
        "return_native_token_timestamps": True,
        "stream": False,
    }
    started = time.perf_counter()
    payload = _post_json(base_url + "/generate", body, timeout)
    elapsed = time.perf_counter() - started
    rows = payload if isinstance(payload, list) else [payload]
    if len(rows) != len(adapters) or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("adapter-batching response has the wrong batch size")
    by_rid = {str(row["meta_info"]["id"]): row for row in rows}
    if len(by_rid) != len(rids) or set(by_rid) != set(rids):
        raise RuntimeError("adapter-batching response has wrong request ownership")
    return [by_rid[rid] for rid in rids], elapsed


def _gpu_memory_mb(gpu: int) -> float:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip().splitlines()[0])


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
    if backend == "DSPARK":
        parameters["scope"] = "last5_native_heads"
    if backend == "NEXTN":
        parameters["parameterization"] = "lora"
        parameters["rank"] = 1
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


def _native_exactness(
    config: ExperimentConfig,
    job: Job,
    output: Path,
    tokens: int,
) -> tuple[list[int], dict[str, int]]:
    exact_job = replace(
        job,
        method="static",
        parameters={
            **job.parameters,
            "deterministic_exactness": True,
            "exactness_bootstrap": True,
        },
    )
    gpus = config.gpu_ids if exact_job.gpu_count == 2 else (config.gpu_ids[0],)
    port = config.server.base_port + (2 if exact_job.gpu_count == 2 else 0)
    output.mkdir(parents=True, exist_ok=True)
    process_type = (
        ReplicaServerProcess
        if exact_job.parameters.get("topology") == "two_replica_tp1_dp2"
        else ServerProcess
    )
    process = process_type(
        config,
        exact_job,
        gpus=gpus,
        port=port,
        output_dir=output,
        selection=None,
    )
    with process as client:
        raw = load_prompts(
            config.dataset_path("controlled_baseline"),
            limit=1,
            split="tuning",
            offset=(job.block or 0) * 16,
        )
        prompt = tuple(client.tokenize(raw[0])[-128:])
        client.run_batch((prompt,), max_new_tokens=16, seed=0, request_id_prefix="warmup")
        client.reset()
        topology = str(exact_job.parameters.get("topology", "tp1_dp1"))
        before = _speed_metrics(client.server_info(), topology)
        results, _ = client.run_batch(
            (prompt,),
            max_new_tokens=tokens,
            seed=job.block or 0,
            request_id_prefix="exactness",
        )
        after = _speed_metrics(client.server_info(), topology)
    committed = int(after["committed_tokens"]) - int(before["committed_tokens"])
    _validate_token_accounting(results, committed, tokens)
    checked = int(after.get("greedy_token_checks", 0)) - int(before.get("greedy_token_checks", 0))
    mismatched = int(after.get("greedy_token_mismatches", 0)) - int(
        before.get("greedy_token_mismatches", 0)
    )
    evidence = {
        "committed_tokens": committed,
        "greedy_token_checks": checked,
        "greedy_token_mismatches": mismatched,
        **_validate_greedy_verify_counts(committed, checked, mismatched),
    }
    return list(results[0].output_ids), evidence


def _measure(
    config: ExperimentConfig,
    job: Job,
    output: Path,
    *,
    max_new_tokens: int,
    exactness_tokens: int = 0,
) -> dict[str, object]:
    dflash_exactness = bool(exactness_tokens and job.backend == "DFLASH")
    separate_exactness = bool(
        dflash_exactness and job.method not in {"target_only", "static"}
    )
    if dflash_exactness and not separate_exactness:
        job = replace(
            job,
            parameters={**job.parameters, "deterministic_exactness": True},
        )
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
    exactness_trajectory = None
    exactness_evidence = None
    sticky_routing = None
    if separate_exactness:
        exactness_trajectory, exactness_evidence = _native_exactness(
            config, job, output / "exactness-bootstrap", exactness_tokens
        )
    with process as client:
        raw = load_prompts(
            config.dataset_path("controlled_baseline"),
            limit=8 if isinstance(client, StickyReplicaClient) else 16,
            split="tuning",
            offset=(job.block or 0) * 16,
        )
        prompts = tuple(tuple(client.tokenize(prompt)[-128:]) for prompt in raw)
        if isinstance(client, StickyReplicaClient):
            client.warmup(prompts[0], max_new_tokens=16, seed=0)
        else:
            client.run_batch(
                prompts[:1],
                max_new_tokens=16,
                seed=0,
                request_id_prefix="warmup",
            )
            client.reset()
        if dflash_exactness and not separate_exactness:
            exactness_before = _speed_metrics(
                client.server_info(), str(job.parameters.get("topology", "tp1_dp1"))
            )
            exactness, _ = client.run_batch(
                prompts[:1],
                max_new_tokens=exactness_tokens,
                seed=job.block or 0,
                request_id_prefix="exactness",
            )
            exactness_trajectory = list(exactness[0].output_ids)
            exactness_after = _speed_metrics(
                client.server_info(), str(job.parameters.get("topology", "tp1_dp1"))
            )
            if job.backend == "DFLASH":
                committed_exactness = int(exactness_after["committed_tokens"]) - int(
                    exactness_before["committed_tokens"]
                )
                checked = int(exactness_after.get("greedy_token_checks", 0)) - int(
                    exactness_before.get("greedy_token_checks", 0)
                )
                mismatched = int(exactness_after.get("greedy_token_mismatches", 0)) - int(
                    exactness_before.get("greedy_token_mismatches", 0)
                )
                exactness_evidence = {
                    "committed_tokens": committed_exactness,
                    "greedy_token_checks": checked,
                    "greedy_token_mismatches": mismatched,
                    **_validate_greedy_verify_counts(committed_exactness, checked, mismatched),
                }
            client.reset()
        topology = str(job.parameters.get("topology", "tp1_dp1"))
        before = _speed_metrics(client.server_info(), topology)
        if isinstance(client, StickyReplicaClient):
            routing_keys = tuple(f"cohort-{index % 4:04d}" for index in range(len(prompts)))
            routing_rows = [
                {
                    "request_id": f"scheduled-{index:05d}",
                    "routing_key": key,
                    "replica_index": client.replica_index(key),
                }
                for index, key in enumerate(routing_keys)
            ]
            sticky_routing = {
                "requests": len(routing_rows),
                "cohorts": len(set(routing_keys)),
                "replicas_used": sorted({row["replica_index"] for row in routing_rows}),
            }
            _write(output / "sticky-routing.json", routing_rows)
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
            _write(
                output / "request-outcomes.json",
                [outcome.to_dict() for outcome in run.outcomes],
            )
            if len(run.outcomes) != len(prompts) or any(
                outcome.status != "completed" for outcome in run.outcomes
            ):
                raise RuntimeError("DP2 acceptance did not complete every request")
        elif job.method in {"tts", "l0_naive"}:
            run = client.run_bounded(
                prompts,
                max_new_tokens=max_new_tokens,
                seed=job.block or 0,
                request_ids=tuple(
                    f"acceptance-{job.method}-{index:05d}"
                    for index in range(len(prompts))
                ),
                max_in_flight=8,
                deadline_seconds=config.server.request_timeout_seconds,
            )
            results, elapsed = run.results, run.elapsed_seconds
            _write(
                output / "request-outcomes.json",
                [outcome.to_dict() for outcome in run.outcomes],
            )
            if len(run.outcomes) != len(prompts) or any(
                outcome.status != "completed" for outcome in run.outcomes
            ):
                raise RuntimeError(f"{job.method} acceptance did not complete every request")
        else:
            results, elapsed = client.run_batch(
                prompts,
                max_new_tokens=max_new_tokens,
                seed=job.block or 0,
                request_id_prefix=f"acceptance-{job.method}",
            )
        after = _speed_metrics(client.server_info(), topology)
    intervals = [value for result in results for value in result.inter_token_ms]
    committed = int(after["committed_tokens"]) - int(before["committed_tokens"])
    _write(
        output / "raw.json",
        {
            "results": [result.to_dict() for result in results],
            "exactness_trajectory": exactness_trajectory,
            "exactness_evidence": exactness_evidence,
            "before": before,
            "after": after,
            "elapsed_seconds": elapsed,
        },
    )
    _validate_token_accounting(results, committed, max_new_tokens)
    counters = {name: int(after[name]) - int(before[name]) for name in SAFETY_COUNTERS}
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
        "exactness_evidence": exactness_evidence,
        "counters": counters,
        "rank_local": after["rank_local"],
        "rank_aggregates": after["rank_aggregates"],
    }
    if sticky_routing is not None:
        row["sticky_routing"] = sticky_routing
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
    methods = (
        ("target_only", "NONE"),
        ("static", "DFLASH"),
        ("tts", "DFLASH"),
        ("l0_naive", "DFLASH"),
        ("lightcone", "DFLASH"),
        ("onlinespec_ogd", "DFLASH"),
    )
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
        rows.extend(block_rows)
    _write(output / "benchmark.json", rows)


def adapter_batching(args: argparse.Namespace) -> None:
    config = ExperimentConfig.load(args.config)
    config.validate_local_paths()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{config.server.host}:{config.server.base_port + 8}"
    project = Path(__file__).parents[1]
    command = [
        str(config.server.python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(config.model_path("Qwen/Qwen3-8B")),
        "--host",
        config.server.host,
        "--port",
        str(config.server.base_port + 8),
        "--tp-size",
        "1",
        "--context-length",
        "4096",
        "--max-running-requests",
        "8",
        "--mem-fraction-static",
        str(min(config.server.mem_fraction_static, 0.80)),
        "--enable-lora",
        "--max-lora-rank",
        "1",
        "--lora-target-modules",
        "q_proj",
        "v_proj",
        "--max-loras-per-batch",
        "9",
        "--lora-backend",
        "torch_native",
        "--disable-cuda-graph",
        "--enable-deterministic-inference",
        "--disable-radix-cache",
        "--skip-server-warmup",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids[0])
    pythonpath = f"{project / 'src'}:{config.sglang_root / 'python'}"
    if env.get("PYTHONPATH"):
        pythonpath += ":" + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    log = (output / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )
    sampler = GpuSampler((config.gpu_ids[0],), output / "gpu.csv")
    try:
        _wait_health(base_url, process, config.server.startup_timeout_seconds)
        sampler.start()
        names = tuple(f"excluded-adapter-{index}" for index in range(8))
        adapter_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        }
        for index, name in enumerate(names):
            serialized = _portable_tensor_payload(
                _adapter_tensors(config.model_path("Qwen/Qwen3-8B"), index)
            )
            loaded = _post_json(
                base_url + "/load_lora_adapter_from_tensors",
                {
                    "lora_name": name,
                    "config_dict": adapter_config,
                    "serialized_named_tensors": [serialized],
                    "pinned": True,
                },
                config.server.request_timeout_seconds,
            )
            if not isinstance(loaded, dict) or not loaded.get("success"):
                raise RuntimeError(f"failed to load {name}: {loaded}")

        prompt = load_prompts(
            config.dataset_path("controlled_baseline"), limit=1, split="tuning"
        )[0]
        base_rows, _ = _adapter_generate(
            base_url,
            prompt,
            (None,),
            "base",
            args.max_new_tokens,
            config.server.request_timeout_seconds,
        )
        base = _adapter_observation(base_rows[0])
        solo = {}
        for name in names:
            rows, _ = _adapter_generate(
                base_url,
                prompt,
                (name,),
                f"solo-{name}",
                args.max_new_tokens,
                config.server.request_timeout_seconds,
            )
            solo[name] = _adapter_observation(rows[0])
        if not _observations_match(base, solo[names[0]]):
            raise RuntimeError("zero-delta adapter changed the base output")
        if len({tuple(solo[name]["logprobs"]) for name in names[1:]}) != len(names) - 1:
            raise RuntimeError("nonzero adapters were not distinguishable")

        blocks = []
        for count in (1, 2, 4, 8):
            active = names[:count]
            repeated_rows, _ = _adapter_generate(
                base_url,
                prompt,
                (names[0],) * count,
                f"repeated-{count}",
                args.max_new_tokens,
                config.server.request_timeout_seconds,
            )
            repeated = [_adapter_observation(row) for row in repeated_rows]
            if any(not _observations_match(row, solo[names[0]]) for row in repeated):
                _write(
                    output / "failure.json",
                    {
                        "kind": "batch_shape_mismatch",
                        "adapter_count": count,
                        "solo": solo[names[0]],
                        "batched": repeated,
                    },
                )
                raise RuntimeError("repeated adapter batch changed a request result")
            for block in range(3):
                rows, elapsed = _adapter_generate(
                    base_url,
                    prompt,
                    active,
                    f"mixed-{count}-{block}",
                    args.max_new_tokens,
                    config.server.request_timeout_seconds,
                )
                observations = [_adapter_observation(row) for row in rows]
                mismatches = [
                    {
                        "adapter": name,
                        "solo": solo[name],
                        "mixed": observation,
                    }
                    for name, observation in zip(active, observations, strict=True)
                    if not _observations_match(observation, solo[name])
                ]
                if mismatches:
                    _write(
                        output / "failure.json",
                        {
                            "kind": "mixed_adapter_mismatch",
                            "adapter_count": count,
                            "block": block,
                            "mismatches": mismatches,
                        },
                    )
                    raise RuntimeError("mixed adapter batch changed a request result")
                intervals = [
                    (right - left) / 1_000_000
                    for observation in observations
                    for left, right in zip(
                        observation["timestamps_ns"], observation["timestamps_ns"][1:]
                    )
                ]
                tokens = sum(len(observation["output_ids"]) for observation in observations)
                blocks.append(
                    {
                        "adapter_count": count,
                        "block": block,
                        "goodput": committed_goodput(tokens, elapsed),
                        "p99_itl_ms": float(np.quantile(intervals, 0.99)),
                        "hbm_used_mb": _gpu_memory_mb(config.gpu_ids[0]),
                        "elapsed_seconds": elapsed,
                        "observations": observations,
                    }
                )
        _write(
            output / "excluded-adapter-batching.json",
            {
                "registered_paper_experiment": False,
                "lora_backend": "torch_native",
                "base": base,
                "solo": solo,
                "blocks": blocks,
            },
        )
    finally:
        sampler.stop()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        log.close()


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
    selected = set(args.cases or ())
    names = (
        "target",
        "static",
        "tts",
        "l0",
        "lightcone",
        "onlinespec",
        "tp2",
        "dp2",
        "dspark",
        "nextn35",
        "nextn122",
    )
    rows = []
    for name, job in zip(names, cases, strict=True):
        if selected and name not in selected:
            continue
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
    diagnostic = []
    target = next((row for row in rows if row["method"] == "target_only"), None)
    if target is not None:
        for row in rows:
            diagnostic.append(
                {
                    "method": row["method"],
                    "backend": row["backend"],
                    "cross_kernel_trajectory_equal": (
                        row["exactness_trajectory"] == target["exactness_trajectory"]
                    ),
                }
            )
    _write(args.output / "smoke.json", {"rows": rows, "diagnostic": diagnostic})


def nextn(args: argparse.Namespace) -> None:
    config = ExperimentConfig.load(args.config)
    config.validate_local_paths()
    models = {
        "35b": "Qwen/Qwen3.6-35B-A3B",
        "122b": "Qwen/Qwen3.5-122B-A10B-FP8",
    }
    method = "static" if args.mode == "static" else "lightcone"
    job = _job(
        0,
        method,
        "NEXTN",
        block=0,
        model=models[args.model],
        tp2=True,
    )
    parameters = dict(job.parameters)
    parameters["parameterization"] = "full" if args.mode == "full" else "lora"
    if args.mode == "static":
        parameters.pop("rank", None)
    job = replace(job, load="c1" if args.mode == "full" else "c8", parameters=parameters)
    row = _measure(
        config,
        job,
        args.output,
        max_new_tokens=args.max_new_tokens,
    )
    _write(args.output / "nextn.json", {"model": args.model, "mode": args.mode, "row": row})


def compare(args: argparse.Namespace) -> None:
    donor = json.loads(args.donor.read_text(encoding="utf-8"))
    rebuild = json.loads(args.rebuild.read_text(encoding="utf-8"))
    failures = []
    for method in (
        "target_only",
        "static",
        "tts",
        "l0_naive",
        "lightcone",
        "onlinespec_ogd",
    ):
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
        ("adapter-batching", adapter_batching, 128),
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--max-new-tokens", type=int, default=default_tokens)
        if name == "smoke":
            command.add_argument(
                "--cases",
                nargs="+",
                choices=(
                    "target",
                    "static",
                    "tts",
                    "l0",
                    "lightcone",
                    "onlinespec",
                    "tp2",
                    "dp2",
                    "dspark",
                    "nextn35",
                    "nextn122",
                ),
            )
        command.set_defaults(handler=handler)
    command = commands.add_parser("compare")
    command.add_argument("--donor", type=Path, required=True)
    command.add_argument("--rebuild", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(handler=compare)
    command = commands.add_parser("nextn")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--model", choices=("35b", "122b"), required=True)
    command.add_argument("--mode", choices=("static", "lora", "full"), required=True)
    command.add_argument("--max-new-tokens", type=int, default=128)
    command.set_defaults(handler=nextn)
    args = parser.parse_args()
    if hasattr(args, "max_new_tokens") and not 1 <= args.max_new_tokens <= 40800:
        parser.error("--max-new-tokens must be in 1..40800")
    args.handler(args)


if __name__ == "__main__":
    main()
