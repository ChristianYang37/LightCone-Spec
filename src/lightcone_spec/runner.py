"""Stage-by-stage execution of the complete paper protocol."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

from .client import GenerationResult
from .config import ExperimentConfig
from .data import (
    load_arrival_offsets,
    load_arrival_trace,
    load_prompt_pool,
    load_prompt_records,
    load_tts_calibration,
)
from .metrics import (
    SAFETY_COUNTERS,
    benjamini_hochberg,
    block_bootstrap_interval,
    choose_final_blocks,
    committed_goodput,
    hierarchical_request_interval,
    paired_block_statistics,
    paired_relative_bca_interval,
    summarize_attempts,
    validate_scientific_metrics,
)
from .protocol import PAPER_NODES, Job, e2_candidates, materialize
from .server import (
    ReplicaServerProcess,
    ServerProcess,
    adaptation_payload,
    server_session_key,
)
from .state import StateStore


class ScientificFailure(RuntimeError):
    """A measured cell violated a scientific correctness requirement."""


PARTIAL_REDUCTION_NODES = {
    "E3a",
    "TTS-Cal",
    "E1",
    "E2-r0",
    "E2-r1",
    "E2-r2",
    "E2-r3",
    "E3b-pilot",
    "E1a",
    "E5-pilot",
    "E0-tune",
}


def _speed_metrics(server_info: dict[str, object], topology: str) -> dict[str, Any]:
    states = server_info.get("internal_states")
    rank_states = states if isinstance(states, list) and states else [server_info.get("internal_state", server_info)]
    rank_metrics: list[dict[str, Any]] = []
    for state in rank_states:
        if not isinstance(state, dict):
            raise ScientificFailure("SGLang server state is malformed")
        metrics = state.get("speed_study_metrics")
        if not isinstance(metrics, dict):
            raise ScientificFailure("patched SGLang did not expose speed-study metrics")
        adaptation = state.get("speculative_adaptation_info_record")
        if isinstance(adaptation, dict):
            online = adaptation.get("online_adaptation")
            if isinstance(online, dict):
                metrics = {**metrics, **online}
        rank_metrics.append(metrics)
    required = {
        "committed_tokens",
        "peak_hbm_bytes",
        "peak_hbm_reserved_bytes",
        "kv_token_capacity",
        *SAFETY_COUNTERS,
    }
    for rank, metrics in enumerate(rank_metrics):
        missing = sorted(required - metrics.keys())
        if missing:
            raise ScientificFailure(f"SGLang rank {rank} metrics are missing {missing}")
    combined: dict[str, Any] = {}
    keys = set().union(*(row.keys() for row in rank_metrics))
    max_fields = {
        "peak_hbm_bytes",
        "peak_hbm_reserved_bytes",
        "exposed_update_ms",
        *SAFETY_COUNTERS,
    }
    min_fields = {"kv_token_capacity"}
    mean_fields = {"batch_fill", "queue_occupancy"}
    for key in keys:
        values = [row[key] for row in rank_metrics if key in row]
        if not values:
            continue
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            if key in max_fields:
                combined[key] = max(values)
            elif key in min_fields:
                combined[key] = min(values)
            elif key in mean_fields:
                combined[key] = sum(values) / len(values)
            elif topology != "two_replica_tp1_dp2":
                combined[key] = max(values)
            else:
                combined[key] = sum(values)
        elif all(isinstance(value, bool) for value in values):
            combined[key] = all(values)
        else:
            combined[key] = values[0]
    combined["rank_local"] = rank_metrics
    combined["rank_aggregates"] = {
        key: {
            "sum": sum(values),
            "max": max(values),
            "min": min(values),
        }
        for key in keys
        if (
            values := [
                row[key]
                for row in rank_metrics
                if key in row
                and isinstance(row[key], (int, float))
                and not isinstance(row[key], bool)
            ]
        )
        and len(values) == len(rank_metrics)
    }
    return combined


def _peak_hbm_from_csv(path: Path, offset: int = 0) -> int:
    if not path.is_file():
        return 0
    peak = 0.0
    with path.open("rb") as stream:
        stream.seek(offset)
        lines = stream.read().decode("utf-8").splitlines()
    for line in lines:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 3:
            try:
                peak = max(peak, float(fields[2]))
            except ValueError:
                continue
    return int(peak * 1024 * 1024)


def _task_for_data(config: ExperimentConfig, job: Job) -> str:
    if job.task not in config.datasets:
        raise ScientificFailure(f"job requires an explicit dataset path for {job.task}")
    return job.task


def _prompt_offset(job: Job, limit: int) -> int:
    condition = "|".join(
        str(value)
        for value in (job.model, job.task, job.context, job.load, job.block)
        if value is not None
    )
    condition += "|" + "|".join(
        f"{name}={job.parameters[name]}"
        for name in ("regime", "width_panel", "topology", "cohorts", "popularity")
        if name in job.parameters
    )
    return sum((index + 1) * ord(character) for index, character in enumerate(condition)) * limit


def _request_count(config: ExperimentConfig, job: Job) -> int:
    if job.node == "TTS-Cal":
        return 76
    concurrency = 1
    load = job.load or ""
    if load.startswith("c") and load[1:].isdigit():
        concurrency = int(load[1:])
    elif load.startswith("closed_loop_c"):
        concurrency = int(load.removeprefix("closed_loop_c"))
    count = max(config.server.requests_per_cell, concurrency)
    if job.node == "E5-final" and load == "saturation_soak":
        # Twelve final blocks still exceed the registered 10,000-request p99 gate.
        count = max(count, 840)
    return count


def _fit_prompt(tokens: tuple[int, ...], filler: tuple[int, ...], length: int) -> tuple[int, ...]:
    if length < 1:
        raise ScientificFailure("a context cell has no room for a prompt")
    if len(tokens) >= length:
        return tokens[-length:]
    needed = length - len(tokens)
    if len(filler) < needed:
        raise ScientificFailure(
            f"dataset-native context has {len(filler)} tokens; {needed} required"
        )
    return filler[:needed] + tokens


def _cell_inputs(
    config: ExperimentConfig,
    client,
    job: Job,
) -> tuple[tuple[str | tuple[int, ...], ...], int, dict[str, object]]:
    count = _request_count(config, job)
    dataset_key = _task_for_data(config, job)
    metadata: dict[str, object] = {"dataset": dataset_key}
    if job.node == "TTS-Cal":
        prompts, holdout = load_tts_calibration(config.dataset_path(dataset_key))
        metadata["holdout_problem_ids"] = holdout
    else:
        records = load_prompt_records(
            config.dataset_path(dataset_key),
            limit=count,
            selection_seed=_prompt_offset(job, count),
        )
        prompts = tuple(str(row["prompt"]) for row in records)
        metadata["examples"] = records
    if job.context is None:
        return prompts, config.server.max_new_tokens, metadata
    tokenized = tuple(client.tokenize(prompt) for prompt in prompts)
    if job.node == "TTS-Cal":
        pool_prompts = prompts
    else:
        pool_prompts = tuple(
            str(row["prompt"])
            for row in load_prompt_pool(config.dataset_path(dataset_key))
        )
    filler = tuple(
        token for prompt in pool_prompts for token in client.tokenize(prompt + "\n")
    )
    regime = str(job.parameters.get("regime", "long_input_short_output"))
    if regime == "short_input_long_generation":
        inputs = tuple(tokens[-min(len(tokens), 128) :] for tokens in tokenized)
        max_new_tokens = max(1, job.context - max(len(tokens) for tokens in inputs))
    else:
        max_new_tokens = min(256, config.server.max_new_tokens)
        prompt_length = max(1, job.context - max_new_tokens)
        if regime == "multi_turn_shared_prefix":
            shared_length = prompt_length // 2
            shared = (filler * ((shared_length + len(filler) - 1) // len(filler)))[
                :shared_length
            ]
            inputs = tuple(
                shared + _fit_prompt(tokens, filler, prompt_length - shared_length)
                for tokens in tokenized
            )
        else:
            inputs = tuple(_fit_prompt(tokens, filler, prompt_length) for tokens in tokenized)
    if job.load == "burstgpt_shape":
        trace_path = config.datasets.get("BurstGPT")
        if trace_path is None:
            raise ScientificFailure("BurstGPT row lacks its explicit trace")
        _, lengths = load_arrival_trace(
            trace_path,
            limit=count,
            offset=_prompt_offset(job, count),
        )
        inputs = tuple(
            _fit_prompt(tokens, filler, min(input_length, job.context - 1))
            for tokens, (input_length, _) in zip(tokenized, lengths, strict=True)
        )
        max_new_tokens = tuple(
            min(output_length, job.context - len(tokens))
            for tokens, (_, output_length) in zip(inputs, lengths, strict=True)
        )
        metadata["trace_lengths"] = lengths
    metadata.update(
        regime=regime,
        prompt_tokens=[len(tokens) for tokens in inputs],
        max_new_tokens=max_new_tokens,
    )
    return inputs, max_new_tokens, metadata


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _e5_reference(state: StateStore, job: Job) -> tuple[float, int]:
    by_method: dict[str, list[tuple[float, int]]] = {}
    for config, metrics in _metric_rows(state, job.node):
        load = config.get("load")
        if (
            config.get("block") != job.block
            or config.get("backend") != job.backend
            or config.get("method") not in {"target_only", "static"}
            or not isinstance(load, str)
            or not load.startswith("closed_loop_c")
            or metrics.get("slo_pass") is not True
        ):
            continue
        concurrency = int(load.removeprefix("closed_loop_c"))
        by_method.setdefault(config["method"], []).append(
            (float(metrics["request_rate"]), concurrency)
        )
    if set(by_method) != {"target_only", "static"}:
        raise ScientificFailure(
            f"{job.node} lacks SLO-feasible Target-only/Static load anchors"
        )
    per_method = [max(rows) for rows in by_method.values()]
    return min(per_method, key=lambda row: row[0])


def _runtime_job(state: StateStore, job: Job) -> Job:
    if job.node == "E1" and job.load is None:
        capacity = state.selection("e3a", None)
        if not isinstance(capacity, dict) or not isinstance(capacity.get("load"), str):
            raise ScientificFailure("E1 cell lacks the frozen common load")
        job = replace(job, load=capacity["load"])
    if job.node.startswith("E4") and job.load in {"low", "moderate", "saturation"}:
        capacity = state.selection("e3a", None)
        if not isinstance(capacity, dict) or not isinstance(capacity.get("load"), str):
            raise ScientificFailure("E4 cell lacks the E3a load selection")
        maximum = int(capacity["load"].removeprefix("c"))
        concurrency = {
            "low": 1,
            "moderate": max(1, maximum // 2),
            "saturation": maximum,
        }[job.load]
        return replace(
            job,
            load=f"c{concurrency}",
            parameters={**job.parameters, "registered_load": job.load},
        )
    if job.parameters.get("width_panel") in {"matched", "deployment_optimal"}:
        capacity = state.selection("e3a", None)
        if not isinstance(capacity, dict):
            raise ScientificFailure("width-panel cell lacks the E3a width selection")
        width = None
        if job.method != "target_only":
            if job.parameters["width_panel"] == "deployment_optimal":
                widths = state.selection("deployment_widths", None)
                if not isinstance(widths, dict) or job.method not in widths:
                    raise ScientificFailure(
                        f"deployment width is not frozen for {job.method}"
                    )
                width = int(widths[job.method])
            else:
                width = int(capacity.get("width", 16))
        job = replace(job, width=width)
    if job.load not in {"common_load", "common_slo_load"}:
        if job.node.startswith("E5") and isinstance(job.load, str) and (
            job.load.startswith("lambda_")
            or job.load in {"immediate_burst", "burstgpt_shape", "moderate_soak", "saturation_soak", "overload_soak", "saturation"}
        ):
            _, concurrency = _e5_reference(state, job)
            return replace(
                job,
                load=f"c{concurrency}",
                parameters={**job.parameters, "registered_load": job.load},
            )
        return job
    if job.node.startswith("E6"):
        loads = state.selection("e6_common_loads", None)
        if not isinstance(loads, dict) or not isinstance(loads.get(job.model), str):
            raise ScientificFailure(f"E6 lacks a feasible common load for {job.model}")
        return replace(
            job,
            load=loads[job.model],
            parameters={**job.parameters, "registered_load": job.load},
        )
    capacity = state.selection("e3a", None)
    if not isinstance(capacity, dict) or not isinstance(capacity.get("load"), str):
        raise ScientificFailure("common-load cell lacks a feasible E3a selection")
    width = job.width
    if job.method != "target_only" and width is None:
        width = int(capacity.get("width", 16))
    return replace(
        job,
        load=capacity["load"],
        width=width,
        parameters={**job.parameters, "registered_load": job.load},
    )


def _arrival_offsets(
    config: ExperimentConfig,
    state: StateStore,
    original_job: Job,
    runtime_job: Job,
    count: int,
) -> tuple[float, ...] | None:
    registered = runtime_job.parameters.get("registered_load")
    workload = original_job.parameters.get("workload")
    if workload == "topology_cohort_capacity":
        return (0.0,) * count
    if workload != "production_crossover" or not isinstance(registered, str):
        return None
    reference_rate, _ = _e5_reference(state, original_job)
    if registered == "immediate_burst":
        return (0.0,) * count
    if registered == "burstgpt_shape":
        trace = config.datasets.get("BurstGPT")
        if trace is None:
            raise ScientificFailure("BurstGPT workload-shape row lacks its local trace")
        return load_arrival_offsets(
            trace,
            limit=count,
            offset=_prompt_offset(original_job, count),
        )
    factors = {
        "moderate_soak": 0.5,
        "saturation_soak": 1.0,
        "overload_soak": 1.25,
    }
    if registered.startswith("lambda_"):
        factor = float(registered.removeprefix("lambda_"))
    else:
        factor = factors.get(registered)
    if factor is None:
        return None
    rate = reference_rate * factor
    if rate <= 0:
        raise ScientificFailure("open-loop reference rate is not positive")
    rng = np.random.default_rng(
        config.protocol.seed + (original_job.block or 0) * 1000
    )
    gaps = rng.exponential(1.0 / rate, size=max(0, count - 1))
    return tuple(np.concatenate(([0.0], np.cumsum(gaps))).tolist())


def _routing_keys(config: ExperimentConfig, job: Job, count: int) -> tuple[str, ...] | None:
    cohorts = job.parameters.get("cohorts")
    if not isinstance(cohorts, int) or cohorts < 1:
        return None
    if job.parameters.get("popularity") == "zipf":
        rng = np.random.default_rng(config.protocol.seed + (job.block or 0))
        indexes = np.minimum(rng.zipf(1.2, size=count) - 1, cohorts - 1)
    else:
        indexes = np.arange(count) % cohorts
    return tuple(f"cohort-{int(index):04d}" for index in indexes)


def _slo_metrics(results: Iterable[Any]) -> tuple[float, bool]:
    rows = tuple(results)
    passed = 0
    for result in rows:
        ttft_limit = 2000 if result.input_tokens < 4096 else 5000 if result.input_tokens < 16384 else 10000
        itl_p99 = float(np.quantile(result.inter_token_ms or (0.0,), 0.99))
        passed += result.ttft_ms <= ttft_limit and itl_p99 <= 100
    pass_rate = passed / len(rows)
    return pass_rate, pass_rate >= 0.99


def _exercise_request_fault(client, failure: str, prompt, max_new_tokens: int) -> bool:
    if failure == "cancellation":
        request_id = "fault-cancellation"
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                client.run_batch,
                (prompt,),
                max_new_tokens=max(32, min(max_new_tokens, 256)),
                seed=0,
                request_ids=(request_id,),
            )
            time.sleep(0.1)
            client.abort(request_id)
            try:
                future.result()
            except Exception:
                pass
        return True
    elif failure == "duplicate_retry":
        request_id = "fault-duplicate-retry"
        for _ in range(2):
            client.run_batch(
                (prompt,),
                max_new_tokens=min(max_new_tokens, 8),
                seed=0,
                request_ids=(request_id,),
            )
        return True
    return True


def _fault_action_passed(
    failure: str, request_action_passed: bool, metrics: dict[str, Any]
) -> bool:
    if failure == "nonfinite_candidate":
        return metrics["nonfinite_updates"] > 0
    if failure == "oom_candidate":
        return metrics["oom_events"] > 0
    if failure in {"telemetry_backpressure", "disk_quota"}:
        return metrics["fallbacks"] > 0
    if failure == "queue_saturation":
        outcomes = metrics["request_outcomes"]
        return outcomes["offered"] == 256 and outcomes["unfinished"] == 0
    return request_action_passed


def _run_multi_turn(
    client, prompts, max_new_tokens: int, seed: int, *, request_scoped: bool
):
    histories = [tuple(prompt) for prompt in prompts]
    turns_by_request: list[list[GenerationResult]] = [[] for _ in prompts]
    elapsed = 0.0
    remaining = max_new_tokens
    for turn in range(4):
        turns_left = 4 - turn
        budget = max(1, remaining // turns_left)
        if request_scoped:
            turn_results, turn_elapsed = _run_request_scoped(
                client, histories, budget, seed + turn
            )
        else:
            turn_results, turn_elapsed = client.run_batch(
                histories,
                max_new_tokens=budget,
                seed=seed + turn,
            )
        elapsed += turn_elapsed
        for index, result in enumerate(turn_results):
            turns_by_request[index].append(result)
        histories = [
            (*history, *result.output_ids)
            for history, result in zip(histories, turn_results, strict=True)
        ]
        remaining -= budget
    results = []
    for index, rows in enumerate(turns_by_request):
        timestamps = tuple(
            value for row in rows for value in row.native_token_timestamps_ns
        )
        results.append(
            GenerationResult(
                request_id=f"multi-turn-{index:05d}",
                input_tokens=rows[0].input_tokens,
                completion_tokens=sum(row.completion_tokens for row in rows),
                ttft_ms=rows[0].ttft_ms,
                inter_token_ms=tuple(
                    (right - left) / 1_000_000
                    for left, right in zip(timestamps, timestamps[1:])
                ),
                elapsed_seconds=sum(row.elapsed_seconds for row in rows),
                stop_reason=rows[-1].stop_reason,
                output_ids=tuple(value for row in rows for value in row.output_ids),
                output_text="".join(row.output_text for row in rows),
                native_token_timestamps_ns=timestamps,
            )
        )
    return tuple(results), elapsed


def _run_request_scoped(
    client, prompts, max_new_tokens: int, seed: int, *, same_seed: bool = False
):
    started = time.perf_counter()
    results = []
    for index, prompt in enumerate(prompts):
        rows, _ = client.run_batch(
            (prompt,),
            max_new_tokens=max_new_tokens,
            seed=seed if same_seed else seed + index,
            request_ids=(f"request-scoped-{index:05d}",),
        )
        results.append(rows[0])
    return tuple(results), time.perf_counter() - started


def _answer_value(text: object) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].replace(" ", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def _canonical_accuracy(
    task: str, results: tuple[GenerationResult, ...], workload: dict[str, object]
) -> tuple[float | None, str]:
    if task not in {"GSM8K", "MATH-500", "AIME-2025"}:
        return None, "N/A: no local official scorer"
    examples = workload.get("examples")
    if not isinstance(examples, tuple) or len(examples) != len(results):
        return None, "N/A: scorer metadata unavailable"
    pairs = [
        (_answer_value(result.output_text), _answer_value(example.get("reference")))
        for result, example in zip(results, examples, strict=True)
        if isinstance(example, dict)
    ]
    if len(pairs) != len(results) or any(None in pair for pair in pairs):
        return None, "N/A: canonical answer unavailable"
    return sum(left == right for left, right in pairs) / len(pairs), "exact_final_answer"


def _execute_cell(
    config: ExperimentConfig,
    state: StateStore,
    job: Job,
    *,
    gpus: tuple[int, ...],
    selection: dict[str, Any] | None,
    server: ServerProcess,
) -> None:
    while True:
        next_attempt = state.next_attempt(job.job_id)
        output_dir = config.run_dir / "jobs" / job.job_id / f"attempt-{next_attempt:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        attempt = state.start(job, gpus, output_dir)
        _write_json(output_dir / "config.json", job.to_dict())
        (output_dir / "requests.jsonl").write_text("", encoding="utf-8")
        (output_dir / "cycles.jsonl").write_text("", encoding="utf-8")
        (output_dir / "server.log").write_text("", encoding="utf-8")
        (output_dir / "gpu.csv").write_text(
            "timestamp,index,memory_used_mb,power_w,temperature_c,sm_clock_mhz\n",
            encoding="utf-8",
        )
        session_files = {
            "server.log": server.output_dir / "server.log",
            "cycles.jsonl": server.output_dir / "cycles.jsonl",
            "gpu.csv": server.output_dir / "gpu.csv",
        }
        session_offsets = {
            name: path.stat().st_size if path.exists() else 0
            for name, path in session_files.items()
        }
        if server.process is not None:
            (output_dir / "server.pid").write_text(
                f"{server.process.pid}\n", encoding="utf-8"
            )
        offered = 0
        try:
            runtime_job = _runtime_job(state, job)
            raw_config = runtime_job.to_dict()
            raw_config["adaptation"] = adaptation_payload(runtime_job, selection)
            _write_json(output_dir / "config.json", raw_config)
            client = server.configure(runtime_job, selection)
            prompts, max_new_tokens, workload = _cell_inputs(config, client, job)
            offered = len(prompts)
            if runtime_job.parameters.get("controlled_replay") or runtime_job.parameters.get(
                "controlled_pair_baseline"
            ):
                prompts = (prompts[0], prompts[0])
                workload["controlled_pair"] = {
                    "same_teacher_rows": True,
                    "same_seed": config.protocol.seed + (job.block or 0),
                }
                if runtime_job.parameters.get("controlled_replay"):
                    workload["controlled_pair"]["candidate_policies"] = ["tts", "l0"]
            _write_json(output_dir / "workload.json", workload)
            static_interface_passed = False
            if runtime_job.parameters.get("probe"):
                probe_budget = (
                    max_new_tokens[0]
                    if isinstance(max_new_tokens, tuple)
                    else max_new_tokens
                )
                static_rows, _ = client.run_batch(
                    prompts[:1],
                    max_new_tokens=min(8, probe_budget),
                    seed=config.protocol.seed,
                )
                _write_json(
                    output_dir / "static_probe.json",
                    static_rows[0].to_dict(),
                )
                static_interface_passed = True
                runtime_job = replace(runtime_job, method="lightcone_candidate")
                client = server.restart_for(runtime_job, selection)
                if server.process is not None:
                    (output_dir / "server.pid").write_text(
                        f"{server.process.pid}\n", encoding="utf-8"
                    )
                raw_config["adaptive_probe"] = runtime_job.to_dict()
                raw_config["adaptation"] = adaptation_payload(runtime_job, selection)
                _write_json(output_dir / "config.json", raw_config)
            if config.server.warmup_requests:
                warmup_budget = (
                    max_new_tokens[0]
                    if isinstance(max_new_tokens, tuple)
                    else max_new_tokens
                )
                client.run_batch(
                    prompts[: config.server.warmup_requests],
                    max_new_tokens=min(16, warmup_budget),
                    seed=config.protocol.seed,
                )
                client.reset()
            failure = runtime_job.parameters.get("failure")
            if isinstance(failure, str):
                client = server.inject_failure(failure)
                if not client.health():
                    raise RuntimeError(f"server did not recover from {failure}")
                fault_action_passed = _exercise_request_fault(
                    client,
                    failure,
                    prompts[0],
                    max(max_new_tokens)
                    if isinstance(max_new_tokens, tuple)
                    else max_new_tokens,
                )
            topology = str(runtime_job.parameters.get("topology", "tp1_dp1"))
            before = _speed_metrics(client.server_info(), topology)
            arrivals = _arrival_offsets(config, state, job, runtime_job, len(prompts))
            if runtime_job.parameters.get("regime") == "multi_turn_shared_prefix":
                if arrivals is not None:
                    raise ScientificFailure("multi-turn rows cannot use an open-loop trace")
                results, elapsed = _run_multi_turn(
                    client,
                    prompts,
                    max_new_tokens,
                    config.protocol.seed + (job.block or 0),
                    request_scoped=runtime_job.method in {"tts", "l0_naive"},
                )
            elif arrivals is None:
                seed = config.protocol.seed + (job.block or 0)
                serial = runtime_job.method in {"tts", "l0_naive"} or (
                    runtime_job.node == "E1a"
                    and runtime_job.parameters.get("verification") == "fixed_budget"
                )
                if serial:
                    results, elapsed = _run_request_scoped(
                        client,
                        prompts,
                        max_new_tokens,
                        seed,
                        same_seed=bool(runtime_job.parameters.get("controlled_replay")),
                    )
                else:
                    results, elapsed = client.run_batch(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                    )
            else:
                results, elapsed = client.run_scheduled(
                    prompts,
                    arrivals,
                    max_new_tokens=max_new_tokens,
                    seed=config.protocol.seed + (job.block or 0),
                    routing_keys=_routing_keys(config, runtime_job, len(prompts)),
                )
            after = _speed_metrics(client.server_info(), topology)
            stochastic_rows: list[dict[str, int]] = []
            if runtime_job.parameters.get("distribution_check"):
                for sample in range(16):
                    sampled, _ = client.run_batch(
                        prompts,
                        max_new_tokens=1,
                        seed=config.protocol.seed + 10000 + sample * len(prompts),
                        temperature=0.8,
                    )
                    stochastic_rows.extend(
                        {
                            "request_index": index,
                            "sample": sample,
                            "token_id": int(result.output_ids[0]),
                        }
                        for index, result in enumerate(sampled)
                    )
            request_rows = [result.to_dict() for result in results]
            with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as stream:
                for row in request_rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            if stochastic_rows:
                with (output_dir / "stochastic.jsonl").open("w", encoding="utf-8") as stream:
                    for row in stochastic_rows:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
            cycles = output_dir / "cycles.jsonl"
            if not cycles.exists():
                cycles.write_text("", encoding="utf-8")
            committed = int(after["committed_tokens"]) - int(before["committed_tokens"])
            peak_hbm = int(after["peak_hbm_bytes"])
            reserved_hbm = int(after["peak_hbm_reserved_bytes"])
            nvml_hbm = _peak_hbm_from_csv(
                server.output_dir / "gpu.csv", session_offsets["gpu.csv"]
            )
            kv_capacity = after.get("kv_token_capacity")
            native_intervals = [
                value for result in results for value in result.inter_token_ms
            ]
            native_itl = (
                float(np.quantile(native_intervals, 0.99))
                if native_intervals
                else 0.0
            )
            if peak_hbm <= 0 or not isinstance(kv_capacity, (int, float)) or kv_capacity <= 0:
                raise ScientificFailure("patched runtime did not report positive HBM and KV capacity")
            if (
                not isinstance(native_itl, (int, float))
                or isinstance(native_itl, bool)
                or not math.isfinite(float(native_itl))
                or native_itl < 0
            ):
                raise ScientificFailure("patched runtime did not report native p99 ITL")
            metrics: dict[str, Any] = {
                "committed_tokens": committed,
                "duration_seconds": elapsed,
                "goodput": committed_goodput(committed, elapsed),
                "peak_hbm_bytes": peak_hbm,
                "allocated_peak_hbm_bytes": peak_hbm,
                "reserved_peak_hbm_bytes": reserved_hbm,
                "nvml_peak_hbm_bytes": nvml_hbm,
                "kv_capacity": int(kv_capacity),
                "request_count": len(results),
                "request_outcomes": {
                    "offered": len(prompts),
                    "admitted": len(results),
                    "completed": len(results),
                    "error": 0,
                    "timeout": 0,
                    "cancelled": 0,
                    "unfinished": 0,
                },
                "output_tokens": sum(result.completion_tokens for result in results),
                "ttft_p50_ms": float(np.median([result.ttft_ms for result in results])),
                "itl_p99_ms": float(native_itl),
                "rank_local_before": before["rank_local"],
                "rank_local_after": after["rank_local"],
                "rank_aggregates_before": before["rank_aggregates"],
                "rank_aggregates_after": after["rank_aggregates"],
            }
            accuracy, scorer = _canonical_accuracy(job.task, results, workload)
            metrics["canonical_accuracy"] = accuracy
            metrics["scorer"] = scorer
            if runtime_job.parameters.get("probe"):
                metrics["compatible"] = True
                metrics["static_interface_passed"] = static_interface_passed
                metrics["adaptive_interface_passed"] = True
            if job.node == "E1a":
                metrics["fixed_verification_budget"] = (
                    8 if job.parameters.get("verification") == "fixed_budget" else None
                )
                metrics["confidence_loss_weight"] = 0.1
            slo_pass_rate, slo_pass = _slo_metrics(results)
            metrics["request_rate"] = len(results) / elapsed
            metrics["slo_pass_rate"] = slo_pass_rate
            metrics["slo_pass"] = slo_pass
            if arrivals is not None:
                metrics["offered_rate"] = (
                    (len(arrivals) - 1) / arrivals[-1]
                    if len(arrivals) > 1 and arrivals[-1] > 0
                    else None
                )
            for counter in SAFETY_COUNTERS:
                metrics[counter] = int(after[counter]) - int(before[counter])
            for name in (
                "target_calls",
                "accepted_drafts",
                "verified_drafts",
                "verification_waste",
                "updates_launched",
                "updates_published",
                "exposed_update_ms",
                "batch_fill",
                "queue_occupancy",
            ):
                if name in after:
                    value = after[name]
                    metrics[name] = value - before.get(name, 0) if isinstance(value, (int, float)) else value
            if isinstance(runtime_job.parameters.get("failure"), str):
                failure_name = runtime_job.parameters["failure"]
                metrics["failure_injected"] = failure_name
                metrics["recovery_health_passed"] = True
                metrics["expected_action_passed"] = _fault_action_passed(
                    failure_name, fault_action_passed, metrics
                )
            if (
                job.parameters.get("workload") != "failure_injection"
                and runtime_job.method
                in {"tts", "l0_naive", "lightcone", "lightcone_candidate"}
                and metrics.get("updates_published", 0)
                < int(job.parameters.get("minimum_updates", 1))
            ):
                raise ScientificFailure("adaptive cell did not publish the required updates")
            if runtime_job.parameters.get("controlled_replay"):
                if after.get("controlled_candidate_equal") is not True:
                    raise ScientificFailure("controlled replay changed the staged candidate")
                metrics["controlled_candidate_equal"] = True
            if job.parameters.get("workload") != "failure_injection":
                validate_scientific_metrics(metrics)
            elif not (
                metrics.get("recovery_health_passed")
                and metrics.get("expected_action_passed")
            ):
                raise ScientificFailure("fault diagnostic did not complete its expected action")
            _write_json(output_dir / "metrics.json", metrics)
            state.complete(job.job_id, attempt)
            return
        except ScientificFailure as error:
            if job.parameters.get("probe"):
                _write_json(
                    output_dir / "metrics.json",
                    {
                        "compatible": False,
                        "error": str(error),
                        "request_outcomes": {
                            "offered": offered,
                            "completed": 0,
                            "error": int(offered > 0),
                            "unfinished": max(0, offered - 1),
                        },
                    },
                )
                state.complete(job.job_id, attempt)
                return
            state.fail(job.job_id, attempt, str(error), retry=False)
            _write_json(
                output_dir / "metrics.json",
                {
                    "status": "failed",
                    "error": str(error),
                    "request_outcomes": {
                        "offered": offered,
                        "completed": 0,
                        "error": int(offered > 0),
                        "unfinished": max(0, offered - 1),
                    },
                },
            )
            return
        except Exception as error:
            retry = (
                _retryable_process_error(error)
                and state.failed_attempts(job.job_id)
                < config.protocol.max_process_retries
            )
            _write_json(
                output_dir / "metrics.json",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "retry_scheduled": retry,
                    "request_outcomes": {
                        "offered": offered,
                        "completed": 0,
                        "error": int(offered > 0),
                        "unfinished": max(0, offered - 1),
                    },
                },
            )
            state.fail(job.job_id, attempt, f"{type(error).__name__}: {error}", retry=retry)
            if not retry:
                return
            server.restart()
        finally:
            for name, source in session_files.items():
                if not source.exists():
                    continue
                with source.open("rb") as stream:
                    stream.seek(session_offsets[name])
                    payload = stream.read()
                if name == "gpu.csv":
                    with (output_dir / name).open("ab") as stream:
                        stream.write(payload)
                else:
                    (output_dir / name).write_bytes(payload)


def _selection_for_job(state: StateStore, job: Job) -> dict[str, Any] | None:
    if job.method in {"tts", "l0_naive"}:
        return state.selection("tts_recipe", {})
    if job.method == "lightcone":
        name = "dspark_recipe" if job.backend == "DSPARK" else "lightcone_recipe"
        return state.selection(name, {})
    if job.node == "E1a" and job.method == "lightcone_candidate":
        return state.selection("lightcone_recipe", {})
    return None


def _retryable_process_error(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return False
    if isinstance(
        error,
        (TimeoutError, ConnectionError, urllib.error.URLError, subprocess.SubprocessError),
    ):
        return True
    return isinstance(error, RuntimeError) and any(
        phrase in str(error)
        for phrase in (
            "SGLang exited during startup",
            "server did not recover",
            "stream did not return a complete result",
        )
    )


def _assigned_gpu(config: ExperimentConfig, job: Job) -> int:
    gpu_index = job.parameters.get("gpu_index")
    if isinstance(gpu_index, int) and 0 <= gpu_index < len(config.gpu_ids):
        return config.gpu_ids[gpu_index]
    if job.block is not None:
        return config.gpu_ids[job.block % 2]
    return config.gpu_ids[job.ordinal % 2]


def _e5_execution_phases(
    jobs: tuple[Job, ...],
) -> tuple[tuple[Job, ...], tuple[Job, ...]]:
    anchors = tuple(
        job
        for job in jobs
        if job.method in {"target_only", "static"}
        and isinstance(job.load, str)
        and job.load.startswith("closed_loop_c")
    )
    anchor_ids = {job.job_id for job in anchors}
    return anchors, tuple(job for job in jobs if job.job_id not in anchor_ids)


def _run_pending_jobs(
    config: ExperimentConfig,
    state: StateStore,
    node: str,
    stop_event: threading.Event,
    pending: tuple[Job, ...],
) -> None:
    exclusive = tuple(job for job in pending if job.gpu_count == 2)
    singles = tuple(job for job in pending if job.gpu_count == 1)

    def run_sessions(
        jobs: Iterable[Job], *, gpus: tuple[int, ...], port: int, label: str
    ) -> None:
        grouped: dict[
            tuple[object, ...], list[tuple[Job, Job, dict[str, Any] | None]]
        ] = {}
        for job in jobs:
            runtime_job = _runtime_job(state, job)
            selection = _selection_for_job(state, job)
            probe = job.job_id if job.parameters.get("adaptive_probe") else None
            key = (job.block, probe, *server_session_key(runtime_job, selection))
            grouped.setdefault(key, []).append((job, runtime_job, selection))
        keys = list(grouped)
        rng = np.random.default_rng(config.protocol.seed + sum(gpus) + len(node))
        rng.shuffle(keys)
        for key in keys:
            if stop_event.is_set():
                return
            rows = grouped[key]
            first_job, first_runtime, first_selection = rows[0]
            block = "none" if first_job.block is None else f"{first_job.block:02d}"
            session_dir = (
                config.run_dir
                / "sessions"
                / node
                / label
                / f"block-{block}-from-job-{first_job.ordinal:06d}"
            )
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "cycles.jsonl").touch()
            startup_attempt = 0
            while True:
                process_type = (
                    ReplicaServerProcess
                    if first_runtime.parameters.get("topology")
                    == "two_replica_tp1_dp2"
                    else ServerProcess
                )
                process = process_type(
                    config,
                    first_runtime,
                    gpus=gpus,
                    port=port,
                    output_dir=session_dir,
                    selection=first_selection,
                )
                try:
                    with process:
                        for job, _, selection in rows:
                            if stop_event.is_set():
                                return
                            _execute_cell(
                                config,
                                state,
                                job,
                                gpus=gpus,
                                selection=selection,
                                server=process,
                            )
                    break
                except Exception as error:
                    retry = (
                        _retryable_process_error(error)
                        and startup_attempt < config.protocol.max_process_retries
                    )
                    startup_attempt += 1
                    _write_json(
                        session_dir / f"startup-{startup_attempt:02d}.json",
                        {
                            "error": f"{type(error).__name__}: {error}",
                            "retry_scheduled": retry,
                        },
                    )
                    if not retry:
                        raise

    run_sessions(
        exclusive,
        gpus=config.gpu_ids,
        port=config.server.base_port + 2,
        label="gpu-both",
    )
    if node == "preflight":
        isolated = [job for job in singles if job.parameters.get("mode") == "isolated"]
        for job in isolated:
            gpu = _assigned_gpu(config, job)
            run_sessions(
                (job,),
                gpus=(gpu,),
                port=config.server.base_port + config.gpu_ids.index(gpu),
                label=f"isolated-gpu-{gpu}",
            )
        concurrent = [job for job in singles if job.parameters.get("mode") == "concurrent"]
        for block in sorted({job.block for job in concurrent}):
            rows = [job for job in concurrent if job.block == block]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = []
                for job in rows:
                    gpu = _assigned_gpu(config, job)
                    futures.append(
                        pool.submit(
                            run_sessions,
                            (job,),
                            gpus=(gpu,),
                            port=config.server.base_port + config.gpu_ids.index(gpu),
                            label=f"concurrent-gpu-{gpu}",
                        )
                    )
                for future in futures:
                    future.result()
        return
    queues = {
        gpu: tuple(job for job in singles if _assigned_gpu(config, job) == gpu)
        for gpu in config.gpu_ids
    }

    def worker(gpu: int, jobs: Iterable[Job]) -> None:
        port = config.server.base_port + config.gpu_ids.index(gpu)
        run_sessions(jobs, gpus=(gpu,), port=port, label=f"gpu-{gpu}")

    headline = node in {"E3b-final", "E5-final", "E6-final", "E0-final"}
    calibration = state.selection("headline_parallel", {"enabled": False})
    workers = 2 if not headline or calibration.get("enabled") else 1
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lightcone-gpu") as pool:
        futures = [pool.submit(worker, gpu, jobs) for gpu, jobs in queues.items()]
        for future in futures:
            future.result()


def _run_node_jobs(
    config: ExperimentConfig,
    state: StateStore,
    node: str,
    stop_event: threading.Event,
) -> None:
    pending = state.pending_jobs(node)
    if node.startswith("E5"):
        anchors, _ = _e5_execution_phases(pending)
        if anchors:
            _run_pending_jobs(config, state, node, stop_event, anchors)
        if stop_event.is_set():
            return
        pending = state.pending_jobs(node)
    if node == "E6-pilot":
        interfaces = tuple(
            job for job in pending if job.parameters.get("interface_fit")
        )
        if interfaces:
            _run_pending_jobs(config, state, node, stop_event, interfaces)
            rows = [
                (item, metrics)
                for item, metrics in _metric_rows(state, node)
                if item.get("parameters", {}).get("interface_fit")
                and metrics.get("slo_pass") is True
            ]
            loads = {item["model"]: str(item["load"]) for item, _ in rows}
            if len(loads) == 2:
                state.set_selection("e6_common_loads", loads)
        if stop_event.is_set():
            return
        pending = state.pending_jobs(node)
    _run_pending_jobs(config, state, node, stop_event, pending)


def _metric_rows(state: StateStore, node: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for directory in state.completed_attempt_dirs(node):
        config_path, metrics_path = directory / "config.json", directory / "metrics.json"
        if config_path.is_file() and metrics_path.is_file():
            rows.append((json.loads(config_path.read_text()), json.loads(metrics_path.read_text())))
    return rows


def _trajectory_group(config: dict[str, Any]) -> tuple[Any, ...]:
    parameters = config.get("parameters", {})
    return (
        config.get("model"),
        config.get("task"),
        config.get("context"),
        config.get("load"),
        config.get("block"),
        *(parameters.get(name) for name in ("regime", "width_panel", "topology", "cohorts", "popularity")),
    )


def _check_greedy_trajectories(state: StateStore, node: str) -> None:
    groups: dict[tuple[Any, ...], list[tuple[str, tuple[tuple[int, ...], ...]]]] = {}
    for directory in state.completed_attempt_dirs(node):
        config_path, requests_path = directory / "config.json", directory / "requests.jsonl"
        if not config_path.is_file() or not requests_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("parameters", {}).get("probe"):
            continue
        trajectories = tuple(
            tuple(json.loads(line)["output_ids"])
            for line in requests_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        groups.setdefault(_trajectory_group(config), []).append((config["method"], trajectories))
    for group, rows in groups.items():
        baselines = [trajectory for method, trajectory in rows if method == "target_only"]
        if not baselines:
            continue
        baseline = baselines[0]
        mismatches = [method for method, trajectory in rows if trajectory != baseline]
        if mismatches:
            raise ScientificFailure(f"{node} greedy token trajectory mismatch for {group}: {mismatches}")


def _categorical_distance(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> float:
    categories = set(left) | set(right)
    left_raw, right_raw = Counter(left), Counter(right)
    left_counts = {item: left_raw[item] / len(left) for item in categories}
    right_counts = {item: right_raw[item] / len(right) for item in categories}
    return sum((left_counts[item] - right_counts[item]) ** 2 for item in categories)


def _check_stochastic_exactness(state: StateStore, node: str) -> None:
    rows: dict[str, list[tuple[int, int]]] = {}
    for directory in state.completed_attempt_dirs(node):
        config_path, samples_path = directory / "config.json", directory / "stochastic.jsonl"
        if not config_path.is_file() or not samples_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows[config["method"]] = [
            (int(row["request_index"]), int(row["token_id"])) for row in samples
        ]
    target = rows.get("target_only")
    candidate = rows.get("l0_naive")
    if not target or not candidate or len(target) != len(candidate):
        raise ScientificFailure("preflight lacks paired stochastic samples")
    observed = _categorical_distance(target, candidate)
    combined = np.asarray(target + candidate, dtype=np.int64)
    rng = np.random.default_rng(0)
    exceed = 0
    for _ in range(999):
        permutation = rng.permutation(len(combined))
        left = [tuple(row) for row in combined[permutation[: len(target)]].tolist()]
        right = [tuple(row) for row in combined[permutation[len(target) :]].tolist()]
        exceed += _categorical_distance(left, right) >= observed
    p_value = (exceed + 1) / 1000
    if p_value < 0.01:
        raise ScientificFailure(
            f"preflight stochastic distributions differ (permutation p={p_value:.4f})"
        )


def _rank_candidates(state: StateStore, node: str, keep: int) -> list[dict[str, Any]]:
    candidates = [
        (float(metrics["goodput"]), config["parameters"])
        for config, metrics in _metric_rows(state, node)
        if config["method"] in {"lightcone_candidate", "lightcone"}
    ]
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in candidates[:keep]]


def _rank_e1_geometries(state: StateStore) -> list[dict[str, Any]]:
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for config, metrics in _metric_rows(state, "E1"):
        if config["method"] != "lightcone_candidate":
            continue
        parameters = {
            name: value
            for name, value in config["parameters"].items()
            if name not in {"optimizer", "fixed_role"}
        }
        key = json.dumps(parameters, sort_keys=True)
        row = grouped.setdefault(key, (parameters, []))
        row[1].append(
            {
                "goodput": float(metrics["goodput"]),
                "hbm": float(metrics["peak_hbm_bytes"]),
                "itl": float(metrics["itl_p99_ms"]),
                "optimizer": config["parameters"].get("optimizer"),
            }
        )
    points = [
        (
            parameters,
            float(np.mean([row["goodput"] for row in values])),
            float(np.max([row["hbm"] for row in values])),
            float(np.max([row["itl"] for row in values])),
        )
        for parameters, values in grouped.values()
        if {row["optimizer"] for row in values} == {"adamw", "sgdm"}
    ]
    pareto = [
        row
        for row in points
        if not any(
            other[1] >= row[1]
            and other[2] <= row[2]
            and other[3] <= row[3]
            and other[1:] != row[1:]
            for other in points
        )
    ]
    pareto.sort(key=lambda row: (-row[1], row[2], row[3]))
    return [parameters for parameters, _, _, _ in pareto[:32]]


def _rank_e2_candidates(state: StateStore, node: str, keep: int) -> list[dict[str, Any]]:
    rows = _metric_rows(state, node)
    baselines = {
        config["method"]: float(metrics["goodput"])
        for config, metrics in rows
        if config["method"] in {"static", "tts"}
    }
    if set(baselines) != {"static", "tts"}:
        return []
    candidates = []
    for config, metrics in rows:
        if config["method"] != "lightcone_candidate":
            continue
        goodput = float(metrics["goodput"])
        objective = min(
            goodput / baselines["static"], goodput / baselines["tts"]
        )
        candidates.append(
            (
                -objective,
                int(metrics["peak_hbm_bytes"]),
                float(metrics["itl_p99_ms"]),
                float(metrics.get("exposed_update_ms", math.inf)),
                {
                    name: value
                    for name, value in config["parameters"].items()
                    if name not in {"round", "minimum_updates", "fixed_role"}
                },
            )
        )
    candidates.sort(key=lambda row: row[:-1])
    return [row[-1] for row in candidates[:keep]]


def _context_splines(state: StateStore, node: str) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[int, list[float]]] = {}
    for config, metrics in _metric_rows(state, node):
        context = config.get("context")
        if not isinstance(context, int):
            continue
        parameters = config["parameters"]
        key = (
            config["method"],
            parameters.get("regime"),
            config.get("load"),
            parameters.get("width_panel"),
        )
        grouped.setdefault(key, {}).setdefault(context, []).append(
            float(metrics["goodput"])
        )
    request_groups: dict[
        tuple[object, ...], dict[int, dict[int, list[tuple[int, float]]]]
    ] = {}
    for directory in state.completed_attempt_dirs(node):
        config_path = directory / "config.json"
        requests_path = directory / "requests.jsonl"
        if not config_path.is_file() or not requests_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        context, block = config.get("context"), config.get("block")
        if not isinstance(context, int) or not isinstance(block, int):
            continue
        parameters = config["parameters"]
        key = (
            config["method"],
            parameters.get("regime"),
            config.get("load"),
            parameters.get("width_panel"),
        )
        rows = []
        for line in requests_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                request = json.loads(line)
                rows.append(
                    (
                        int(request["completion_tokens"]),
                        float(request["elapsed_seconds"]),
                    )
                )
        request_groups.setdefault(key, {}).setdefault(context, {})[block] = rows
    result = []
    for key, values in grouped.items():
        if len(values) < 4:
            continue
        contexts = np.asarray(sorted(values), dtype=np.float64)
        goodput = np.asarray(
            [np.mean(values[int(context)]) for context in contexts], dtype=np.float64
        )
        fixed = np.asarray([4096, 16384, 32768], dtype=np.float64)
        if not all(int(value) in values for value in fixed):
            continue
        fixed_goodput = np.asarray(
            [np.mean(values[int(context)]) for context in fixed], dtype=np.float64
        )
        spline = CubicSpline(np.log(fixed), np.log(fixed_goodput))
        request_intervals = {}
        for context in fixed.astype(int):
            blocks = request_groups.get(key, {}).get(int(context), {})
            if len(blocks) >= 3:
                point, low, high = hierarchical_request_interval(blocks)
                request_intervals[str(context)] = {
                    "point": point,
                    "ci95_low": low,
                    "ci95_high": high,
                }
        result.append(
            {
                "method": key[0],
                "regime": key[1],
                "load": key[2],
                "width_panel": key[3],
                "contexts": contexts.astype(int).tolist(),
                "goodput": goodput.tolist(),
                "spline_knots": fixed.astype(int).tolist(),
                "hierarchical_request_goodput": request_intervals,
                "elasticity": spline(np.log(contexts), 1).tolist(),
                "curvature": spline(np.log(contexts), 2).tolist(),
            }
        )
    return result


def _e5_tail_statistics(state: StateStore, node: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[int, list[float]]] = {}
    request_counts: Counter[tuple[str, str, str]] = Counter()
    for directory in state.completed_attempt_dirs(node):
        config_path = directory / "config.json"
        requests_path = directory / "requests.jsonl"
        if not config_path.is_file() or not requests_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        parameters = config.get("parameters", {})
        load = parameters.get("registered_load", config.get("load"))
        key = (config["method"], config["backend"], str(load))
        for line in requests_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            request = json.loads(line)
            request_counts[key] += 1
            stamps = request.get("native_token_timestamps_ns", [])
            intervals = request.get("inter_token_ms", [])
            if stamps and intervals:
                time_block = int(stamps[0]) // 10_000_000_000
                grouped.setdefault(key, {}).setdefault(time_block, []).extend(intervals)
    result = []
    for (method, backend, load), blocks in grouped.items():
        request_count = request_counts[(method, backend, load)]
        values = [float(np.quantile(rows, 0.99)) for rows in blocks.values() if rows]
        resolved = request_count >= 10_000 and len(values) >= 3
        estimate = block_bootstrap_interval(values) if resolved else (None, None, None)
        result.append(
            {
                "method": method,
                "backend": backend,
                "load": load,
                "request_count": request_count,
                "time_block_count": len(values),
                "resolved": resolved,
                "p99_point_ms": estimate[0],
                "p99_ci95_low_ms": estimate[1],
                "p99_ci95_high_ms": estimate[2],
            }
        )
    return result


def _select_valid_e0(state: StateStore) -> list[tuple[str, str, str]]:
    dspark_ready = state.selection("dspark_recipe", None) is not None
    return [
        (item["model"], item["backend"], item["task"])
        for item, metrics in _metric_rows(state, "E0-tune")
        if item["parameters"].get("probe")
        and metrics.get("compatible", True)
        and (item["backend"] != "DSPARK" or dspark_ready)
    ]


def _select_e0_recipes(state: StateStore) -> dict[str, dict[str, Any]]:
    winners: dict[str, tuple[float, dict[str, Any]]] = {}
    for config, metrics in _metric_rows(state, "E0-tune"):
        method = config["method"]
        if method not in {"onlinespec_ogd", "onlinespec_opt", "onlinespec_ens"}:
            continue
        key = "|".join(
            (config["model"], config["backend"], config["task"], method)
        )
        parameters = {
            name: value
            for name, value in config["parameters"].items()
            if name != "tuning_index"
        }
        score = float(metrics["goodput"])
        if key not in winners or score > winners[key][0]:
            winners[key] = (score, parameters)
    return {key: parameters for key, (_, parameters) in winners.items()}


def _select_e4_screen(state: StateStore) -> dict[str, tuple[object, object]]:
    groups: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for config, metrics in _metric_rows(state, "E4-screen"):
        row = config["parameters"].get("screen_row")
        if isinstance(row, int):
            groups.setdefault(row, []).append((config, metrics))
    complete = [rows for rows in groups.values() if len(rows) == 6]
    if len(complete) != 8:
        raise ScientificFailure("E4 screen lacks eight complete six-stratum rows")
    winner = max(
        complete,
        key=lambda rows: (
            min(float(metrics["goodput"]) for _, metrics in rows),
            -max(int(metrics["peak_hbm_bytes"]) for _, metrics in rows),
            -max(float(metrics["itl_p99_ms"]) for _, metrics in rows),
        ),
    )[0][0]["parameters"]
    return {
        "stride": (1, 5) if winner["stride"] == 1 else (30, 50),
        "microbatch": (1, 2) if winner["microbatch"] == 1 else (4, 8),
        "coalescing": (1, 2) if winner["coalescing"] == 1 else (4, 8),
        "stream_priority": ("default", "high"),
    }


def _pilot_final_blocks(state: StateStore, node: str) -> int | None:
    by_block: dict[int, dict[str, list[float]]] = {}
    metric_rows = _metric_rows(state, node)
    row_counts = Counter(
        config.get("block")
        for config, _ in metric_rows
        if isinstance(config.get("block"), int)
    )
    expected = {
        "E3b-pilot": 480,
        "E5-pilot": 450,
        "E6-pilot": 60,
        "E0-pilot": 16 * len(state.selection("valid_e0", [])),
    }[node]
    if any(row_counts.get(block, 0) != expected for block in range(4)):
        return None
    for config, metrics in metric_rows:
        block = config.get("block")
        method = config.get("method")
        if isinstance(block, int) and method in {"static", "lightcone"}:
            by_block.setdefault(block, {}).setdefault(method, []).append(float(metrics["goodput"]))
    differences = [
        math.log(float(np.mean(rows["lightcone"]))) - math.log(float(np.mean(rows["static"])))
        for block, rows in sorted(by_block.items())
        if block in {0, 1, 2, 3} and {"lightcone", "static"} <= rows.keys()
    ]
    if len(differences) != 4:
        return None
    return choose_final_blocks(differences)


def _reduce_node(config: ExperimentConfig, state: StateStore, node: str) -> None:
    summary_dir = config.run_dir / "stages" / node
    summarize_attempts(state.completed_attempt_dirs(node), summary_dir)
    if node.endswith("-final"):
        statistics = paired_block_statistics(_metric_rows(state, node))
        _write_json(
            summary_dir / "statistics.json",
            statistics,
        )
        if node == "E0-final":
            secondary = [
                row
                for row in statistics
                if (row["candidate"], row["baseline"])
                not in {("lightcone", "static"), ("lightcone", "tts")}
            ]
            decisions = benjamini_hochberg(
                [float(row["p_value"]) for row in secondary]
            ) if secondary else ()
            _write_json(
                summary_dir / "breadth_fdr.json",
                [
                    {**row, "fdr_reject": decision}
                    for row, decision in zip(secondary, decisions, strict=True)
                ],
            )
    if node in {"E3a", "E3b-pilot", "E3b-final"}:
        _write_json(summary_dir / "context_splines.json", _context_splines(state, node))
    if node in {"E5-pilot", "E5-final"}:
        _write_json(
            summary_dir / "tail_latency.json", _e5_tail_statistics(state, node)
        )
    if node in {"preflight", "E3a", "E1", "E3b-pilot", "E3b-final", "E5-pilot", "E5-final", "E6-pilot", "E6-final", "E0-pilot", "E0-final"}:
        _check_greedy_trajectories(state, node)
    if node == "preflight":
        _check_stochastic_exactness(state, node)
        timings: dict[tuple[int, int], dict[str, dict[str, float]]] = {}
        for item, metrics in _metric_rows(state, node):
            mode = item["parameters"].get("mode")
            gpu = item["parameters"].get("gpu_index")
            block = item.get("block")
            if (
                mode in {"isolated", "concurrent"}
                and isinstance(gpu, int)
                and isinstance(block, int)
            ):
                timings.setdefault((gpu, block), {})[mode] = {
                    "goodput": float(metrics["goodput"]),
                    "itl": float(metrics["itl_p99_ms"]),
                }
        pairs = [
            rows
            for rows in timings.values()
            if {"isolated", "concurrent"} <= rows.keys()
        ]
        calibrations = {
            metric: paired_relative_bca_interval(
                [rows["concurrent"][metric] for rows in pairs],
                [rows["isolated"][metric] for rows in pairs],
            )
            for metric in ("goodput", "itl")
        } if len(pairs) >= 3 else {}
        state.set_selection(
            "headline_parallel",
            {
                "enabled": len(pairs) == 4
                and all(
                    abs(point) <= 0.01 and low <= 0 <= high
                    for point, low, high in calibrations.values()
                ),
                "paired_relative_bca": calibrations,
            },
        )
    if node == "E3a":
        rows = _metric_rows(state, node)
        anchors = [
            (item, metrics)
            for item, metrics in rows
            if item.get("context") == 40928
        ]
        feasible_loads: list[str] = []
        for load in {str(item["load"]) for item, _ in anchors}:
            target_regimes = {
                item["parameters"].get("regime")
                for item, _ in anchors
                if item["method"] == "target_only" and item["load"] == load
            }
            static_regimes = {
                item["parameters"].get("regime")
                for item, _ in anchors
                if item["method"] == "static"
                and item["load"] == load
                and item.get("width") == 16
            }
            if len(target_regimes) == len(static_regimes) == 3:
                feasible_loads.append(load)
        if not feasible_loads:
            raise ScientificFailure(
                "E3a found no common 40,928-token Target-only/Static load"
            )
        common_load = max(feasible_loads, key=lambda value: int(value.removeprefix("c")))
        width_scores = {
            width: float(
                np.mean(
                    [
                        metrics["goodput"]
                        for item, metrics in anchors
                        if item["method"] == "static"
                        and item["load"] == common_load
                        and item.get("width") == width
                    ]
                )
            )
            for width in (4, 8, 16)
            if len(
                {
                    item["parameters"].get("regime")
                    for item, _ in anchors
                    if item["method"] == "static"
                    and item["load"] == common_load
                    and item.get("width") == width
                }
            )
            == 3
        }
        selected_width = max(width_scores, key=width_scores.get, default=16)
        state.set_selection(
            "e3a", {"width": selected_width, "load": common_load}
        )
        state.set_selection(
            "deployment_widths",
            {
                method: selected_width
                for method in ("static", "tts", "l0_naive", "lightcone")
            },
        )
    elif node == "TTS-Cal":
        winners = _rank_candidates(state, node, 1)
        if not winners:
            rows = _metric_rows(state, node)
            if not rows:
                raise ScientificFailure("TTS-Cal produced no feasible recipe")
            winners = [max(rows, key=lambda pair: pair[1]["goodput"])[0]["parameters"]]
        state.set_selection("tts_recipe", winners[0])
    elif node == "E1":
        winners = _rank_e1_geometries(state)
        if not winners:
            state.set_selection("E1_no_feasible_candidate", True)
        else:
            state.set_selection("e1_geometries", winners)
    elif node == "E4-screen":
        state.set_selection("e4_neighborhoods", _select_e4_screen(state))
    elif node.startswith("E2-r"):
        round_index = int(node[-1])
        keep = (840, 210, 53, 1)[round_index]
        winners = _rank_e2_candidates(state, node, keep)
        if len(winners) != keep:
            state.set_selection(f"{node}_no_feasible_candidate", True)
        else:
            state.set_selection("lightcone_recipe" if round_index == 3 else f"e2_round_{round_index}", winners[0] if round_index == 3 else winners)
    elif node == "E3b-pilot" and config.protocol.final_blocks is None:
        selected = _pilot_final_blocks(state, node)
        if selected is None:
            state.set_selection(f"{node}_underpowered", True)
        else:
            state.set_selection("global_final_blocks", selected)
    elif node == "E1a":
        winners = _rank_candidates(state, node, 1)
        if winners:
            state.set_selection("dspark_recipe", winners[0])
    elif node == "E0-tune":
        valid = _select_valid_e0(state)
        recipes = _select_e0_recipes(state)
        state.set_selection("valid_e0", valid)
        if len(recipes) == 3 * len(valid):
            state.set_selection("e0_recipes", recipes)
        else:
            state.set_selection("E0_incomplete_onlinespec_tuning", True)


def _cleanup_interrupted_servers(run_dir: Path) -> None:
    pid_paths = list(run_dir.glob("jobs/*/attempt-*/server.pid"))
    pid_paths.extend(run_dir.glob("sessions/**/server.pid"))
    for pid_path in pid_paths:
        if (pid_path.parent / "server.stopped").exists():
            continue
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            command = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            if "sglang.launch_server" in command:
                os.killpg(pid, signal.SIGTERM)
        except (OSError, ValueError):
            continue


def _save_or_validate_run_config(config: ExperimentConfig) -> None:
    saved = config.run_dir / "paper.yaml"
    normalized = config.normalized()
    if saved.exists():
        if yaml.safe_load(saved.read_text(encoding="utf-8")) != normalized:
            raise RuntimeError("run directory belongs to a different experiment config")
        return
    saved.write_text(yaml.safe_dump(normalized, sort_keys=True), encoding="utf-8")


def _node_final_blocks(config: ExperimentConfig, state: StateStore, node: str) -> int:
    if config.protocol.final_blocks is not None:
        return config.protocol.final_blocks
    if node in {"E3b-final", "E5-final", "E6-final", "E0-final"}:
        return int(state.selection("global_final_blocks", 12))
    return 12


def _dependency_reason(config: ExperimentConfig, state: StateStore, node: str) -> str | None:
    if node != "preflight" and state.stage_status("preflight") != "completed":
        return "preflight did not complete"
    requirements = {
        "E1": ("tts_recipe", "e3a"),
        "E2-r0": ("e1_geometries",),
        "E2-r1": ("e2_round_0",),
        "E2-r2": ("e2_round_1",),
        "E2-r3": ("e2_round_2",),
        "E4-screen": ("lightcone_recipe",),
        "E4-local": ("lightcone_recipe", "e4_neighborhoods"),
        "E4-profile": ("lightcone_recipe",),
        "E3b-pilot": ("e3a", "lightcone_recipe"),
        "E1a": ("lightcone_recipe",),
        "E5-pilot": ("lightcone_recipe", "dspark_recipe"),
        "E6-pilot": ("e3a", "lightcone_recipe"),
        "E0-tune": ("lightcone_recipe",),
        "E0-pilot": ("e3a", "valid_e0", "e0_recipes"),
    }
    for selection in requirements.get(node, ()):
        if state.selection(selection, None) is None:
            return f"required selection {selection} is unavailable"
    if (
        node in {"E3b-final", "E5-final", "E6-final", "E0-final"}
        and config.protocol.final_blocks is None
        and state.selection("global_final_blocks", None) is None
    ):
        return "E3b-pilot did not select global N in 12-20"
    return None


class PaperRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.state = StateStore(config.run_dir)
        self.stop_event = threading.Event()

    def _signal(self, signum, frame) -> None:
        self.stop_event.set()

    def run(self) -> None:
        self.config.validate_local_paths()
        _cleanup_interrupted_servers(self.config.run_dir)
        self.state.recover_interrupted()
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        _save_or_validate_run_config(self.config)
        old_term = signal.signal(signal.SIGTERM, self._signal)
        old_int = signal.signal(signal.SIGINT, self._signal)
        try:
            nodes = list(PAPER_NODES)
            if self.config.protocol.start_stage:
                nodes = nodes[nodes.index(self.config.protocol.start_stage) :]
            if self.config.protocol.end_stage:
                nodes = nodes[: nodes.index(self.config.protocol.end_stage) + 1]
            for node in nodes:
                if self.stop_event.is_set():
                    break
                final_blocks = _node_final_blocks(self.config, self.state, node)
                valid_e0 = self.state.selection("valid_e0", None)
                e2_rows = None
                if node == "E2-r0":
                    e2_rows = e2_candidates(self.state.selection("e1_geometries", None))
                elif node.startswith("E2-r"):
                    e2_rows = self.state.selection(f"e2_round_{int(node[-1]) - 1}", None)
                if node == "E0-tune" and valid_e0 is None:
                    probe_jobs = materialize(node, valid_e0=[])
                    self.state.add_jobs(node, probe_jobs)
                    dependency_reason = _dependency_reason(self.config, self.state, node)
                    if dependency_reason:
                        self.state.skip_pending(node, dependency_reason)
                        continue
                    _run_node_jobs(
                        self.config, self.state, node, self.stop_event
                    )
                    if self.stop_event.is_set():
                        break
                    if self.state.status_counts(node).get("failed"):
                        self.state.finish_stage(node)
                        self.state.set_selection(f"{node}_failed", True)
                        continue
                    valid_e0 = _select_valid_e0(self.state)
                    self.state.set_selection("valid_e0", valid_e0)
                jobs = materialize(
                    node,
                    final_blocks=final_blocks,
                    valid_e0=valid_e0,
                    e2_rows=e2_rows,
                    e0_recipes=self.state.selection("e0_recipes", None),
                    e4_neighborhoods=self.state.selection("e4_neighborhoods", None),
                )
                self.state.add_jobs(node, jobs)
                dependency_reason = _dependency_reason(self.config, self.state, node)
                if dependency_reason:
                    self.state.skip_pending(node, dependency_reason)
                    continue
                _run_node_jobs(self.config, self.state, node, self.stop_event)
                if self.stop_event.is_set():
                    break
                if self.state.finish_stage(
                    node, allow_failed=node in PARTIAL_REDUCTION_NODES
                ) != "completed":
                    self.state.set_selection(f"{node}_failed", True)
                    continue
                try:
                    _reduce_node(self.config, self.state, node)
                except ScientificFailure as error:
                    self.state.mark_stage_failed(node)
                    self.state.set_selection(f"{node}_failed", str(error))
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
