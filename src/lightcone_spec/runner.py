"""Stage-by-stage execution of the complete paper protocol."""

from __future__ import annotations

import gzip
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
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.interpolate import BSpline
from scipy.linalg import null_space

from .client import GenerationResult, ScheduledRun
from .config import ExperimentConfig
from .data import (
    load_arrival_offsets,
    load_arrival_trace,
    load_calibration_mix,
    load_prompt_pool,
    load_prompt_records,
)
from .metrics import (
    SAFETY_COUNTERS,
    benjamini_hochberg,
    block_bootstrap_interval,
    choose_final_blocks,
    committed_goodput,
    paired_block_statistics,
    paired_relative_bca_interval,
    summarize_attempts,
    validate_scientific_metrics,
)
from .protocol import (
    CONFIDENCE_WEIGHTS,
    E1_REFERENCE_LOAD,
    E5_DRAIN_SECONDS,
    E5_HEADLINE_SECONDS,
    E5_P99_EXTENSION_REQUESTS,
    E5_P99_MIN_COMPLETED,
    E5_REQUEST_DEADLINE_SECONDS,
    E5_SOAK_SECONDS,
    E5_WARMUP_SECONDS,
    PAPER_NODES,
    TTS_GENERATION_TOKENS,
    Job,
    e2_candidates,
    materialize,
)
from .server import (
    ReplicaServerProcess,
    ServerProcess,
    adaptation_payload,
    server_session_key,
)
from .state import StateStore


class ScientificFailure(RuntimeError):
    """A measured cell violated a scientific correctness requirement."""


CANDIDATE_METHODS = {
    "lightcone_candidate",
    "onlinespec_candidate",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
}


def _screening_job(job: Job) -> bool:
    return job.node in {"E3a", "E1-common-load", "E6-interface", "E6-common-load"}


def _capacity_infeasible(
    error: Exception,
    server_log: Path | None = None,
    offset: int = 0,
) -> bool:
    message = str(error)
    if server_log is not None and server_log.exists():
        with server_log.open("rb") as stream:
            stream.seek(offset)
            message += stream.read().decode("utf-8", errors="replace")
    return isinstance(error, MemoryError) or re.search(
        r"out of memory|\b(?:cuda )?oom\b|adaptation peak .* exceeds pre-KV reserve",
        message,
        flags=re.IGNORECASE,
    ) is not None


def _exactness_bootstrap(job: Job) -> Job:
    if not job.parameters.get("deterministic_verify"):
        return job
    return replace(
        job,
        method="static",
        parameters={
            **job.parameters,
            "controlled_replay": False,
            "deterministic_exactness": True,
            "exactness_bootstrap": True,
        },
    )


def _validate_committed_tokens(results: Sequence[GenerationResult], committed: int) -> int:
    output_tokens = sum(result.completion_tokens for result in results)
    if committed != output_tokens:
        raise ScientificFailure(
            f"runtime committed {committed} tokens for {output_tokens} output tokens"
        )
    return output_tokens


def _validate_greedy_verify_counts(
    committed: int,
    checked: int,
    mismatched: int,
    *,
    allowed_unverified: int = 1,
) -> dict[str, int]:
    missing_output_tokens = max(committed - checked, 0)
    extra_checked_tokens = max(checked - committed, 0)
    if missing_output_tokens > allowed_unverified or mismatched:
        raise ScientificFailure(
            "deterministic verification did not match every speculative output token "
            f"to its same-logit target argmax ({checked} checked, "
            f"{missing_output_tokens} unverified prefill, "
            f"{extra_checked_tokens} checked beyond streamed output, "
            f"{mismatched} mismatched)"
        )
    return {
        "unverified_prefill_tokens": missing_output_tokens,
        "extra_checked_tokens": extra_checked_tokens,
    }


def _speed_metrics(
    server_info: dict[str, object],
    topology: str,
    *,
    unmeasured: tuple[str, ...] = (),
) -> dict[str, Any]:
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
        for name in unmeasured:
            metrics.setdefault(name, None)
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


def _request_scope_released(server_info: dict[str, object]) -> bool:
    states = server_info.get("internal_states")
    rank_states = (
        states
        if isinstance(states, list) and states
        else [server_info.get("internal_state", server_info)]
    )
    found = False
    for state in rank_states:
        if not isinstance(state, dict):
            continue
        record = state.get("speculative_adaptation_info_record")
        online = record.get("online_adaptation") if isinstance(record, dict) else None
        if not isinstance(online, dict):
            continue
        found = True
        if online.get("active_request_id") is not None:
            return False
    if not found:
        raise ScientificFailure("SGLang did not expose request-scope state")
    return True


def _wait_request_scope_release(client, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not _request_scope_released(client.server_info()):
        if time.monotonic() >= deadline:
            raise ScientificFailure("request-scoped adaptation did not reach terminal state")
        time.sleep(0.01)


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


def _energy_from_csv(path: Path, offset: int = 0) -> float | None:
    if not path.is_file():
        return None
    by_gpu: dict[str, list[float]] = {}
    with path.open("rb") as stream:
        stream.seek(offset)
        lines = stream.read().decode("utf-8").splitlines()
    for line in lines:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 7 or fields[6] == "N/A":
            continue
        try:
            by_gpu.setdefault(fields[1], []).append(float(fields[6]))
        except ValueError:
            continue
    if not by_gpu or any(len(values) < 2 for values in by_gpu.values()):
        return None
    return sum(max(values) - min(values) for values in by_gpu.values())


def _position_survival(rounds: object) -> list[float] | str:
    if not isinstance(rounds, list):
        return "N/A"
    accepted = [
        int(value)
        for row in rounds
        if isinstance(row, dict) and isinstance(row.get("accepted_drafts"), list)
        for value in row["accepted_drafts"]
    ]
    verified = [
        int(value)
        for row in rounds
        if isinstance(row, dict) and isinstance(row.get("verify_len"), list)
        for value in row["verify_len"]
    ]
    if not accepted or len(accepted) != len(verified):
        return "N/A"
    width = max(verified, default=0)
    return [
        sum(a >= position for a, v in zip(accepted, verified, strict=True) if v >= position)
        / sum(v >= position for v in verified)
        for position in range(1, width + 1)
        if any(v >= position for v in verified)
    ]


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


def _stimulus_id(job: Job) -> str:
    values = [job.node, f"block-{job.block if job.block is not None else 'none'}", job.task]
    for name, value in (
        ("context", job.context),
        ("load", job.parameters.get("registered_load", job.load)),
        ("regime", job.parameters.get("regime")),
        ("panel", job.parameters.get("width_panel")),
        ("traffic", job.parameters.get("traffic")),
        ("topology", job.parameters.get("topology")),
        ("cohorts", job.parameters.get("cohorts")),
        ("popularity", job.parameters.get("popularity")),
    ):
        if value is not None:
            values.append(f"{name}-{value}")
    return "__".join(str(value) for value in values)


def _poisson_offsets(rate: float, duration: float, seed: int) -> tuple[float, ...]:
    rng = np.random.default_rng(seed)
    offsets = [0.0]
    while True:
        value = offsets[-1] + float(rng.exponential(1.0 / rate))
        if value > duration:
            return tuple(offsets)
        offsets.append(value)


def _e5_poisson_offsets(
    config: ExperimentConfig, state: StateStore, job: Job
) -> tuple[float, ...] | None:
    registered = job.parameters.get("registered_load", job.load)
    if not job.node.startswith("E5") or not isinstance(registered, str):
        return None
    if job.parameters.get("p99_extension"):
        rate = float(job.parameters["arrival_rate"])
        rng = np.random.default_rng(config.protocol.seed + (job.block or 0) * 1000)
        gaps = rng.exponential(1.0 / rate, size=E5_P99_EXTENSION_REQUESTS - 1)
        return tuple(np.concatenate(([0.0], np.cumsum(gaps))).tolist())
    if registered.startswith("lambda_"):
        factor = float(registered.removeprefix("lambda_"))
        duration = E5_HEADLINE_SECONDS
    else:
        factors = {
            "moderate_soak": 0.75,
            "saturation_soak": 1.0,
            "overload_soak": 1.25,
        }
        factor = factors.get(registered)
        duration = E5_SOAK_SECONDS
    if factor is None:
        return None
    reference_rate, _ = _e5_reference(state, job)
    return _poisson_offsets(
        reference_rate * factor,
        duration,
        config.protocol.seed + (job.block or 0) * 1000,
    )


def _request_count(config: ExperimentConfig, state: StateStore, job: Job) -> int:
    if job.node == "TTS-Cal":
        return 76
    if job.parameters.get("p99_extension"):
        return E5_P99_EXTENSION_REQUESTS
    if job.parameters.get("failure") == "queue_saturation":
        return 512
    concurrency = 1
    load = job.load or ""
    if load.startswith("c") and load[1:].isdigit():
        concurrency = int(load[1:])
    elif load.startswith("closed_loop_c"):
        concurrency = int(load.removeprefix("closed_loop_c"))
    offsets = _e5_poisson_offsets(config, state, job)
    return len(offsets) if offsets is not None else max(config.server.requests_per_cell, concurrency)


def _cell_concurrency(job: Job) -> int:
    load = job.load or ""
    if load.startswith("closed_loop_c"):
        return int(load.removeprefix("closed_loop_c"))
    if load.startswith("c") and load[1:].isdigit():
        return int(load[1:])
    return 1


def _uses_request_scope(job: Job) -> bool:
    return job.method in {"tts", "l0_naive"}


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
    state: StateStore,
    client,
    job: Job,
) -> tuple[tuple[str | tuple[int, ...], ...], int, dict[str, object]]:
    count = _request_count(config, state, job)
    dataset_key = _task_for_data(config, job)
    metadata: dict[str, object] = {"dataset": dataset_key}
    if job.node == "TTS-Cal":
        prompts = load_calibration_mix(config.dataset_path(dataset_key))
    else:
        records = load_prompt_records(
            config.dataset_path(dataset_key),
            limit=count,
            selection_seed=_prompt_offset(job, count),
            allow_repeat=job.node.startswith("E5"),
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
        available = job.context - max(len(tokens) for tokens in inputs)
        requested = int(job.parameters.get("generation_tokens", available))
        max_new_tokens = max(1, min(requested, available))
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


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _request_metrics(result: GenerationResult) -> dict[str, object]:
    return {
        "request_id": result.request_id,
        "input_tokens": result.input_tokens,
        "completion_tokens": result.completion_tokens,
        "ttft_ms": result.ttft_ms,
        "inter_token_ms": list(result.inter_token_ms),
        "elapsed_seconds": result.elapsed_seconds,
        "stop_reason": result.stop_reason,
        "native_token_timestamps_ns": list(result.native_token_timestamps_ns),
        "stop_details": result.stop_details,
    }


def _jsonl_path(directory: Path, name: str) -> Path:
    compressed = directory / f"{name}.gz"
    return compressed if compressed.is_file() else directory / name


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


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
    return max(per_method, key=lambda row: row[0])


def _runtime_job(state: StateStore, job: Job) -> Job:
    if job.node == "E1" and job.load is None:
        job = replace(
            job,
            load=E1_REFERENCE_LOAD,
            parameters={**job.parameters, "registered_load": "reference_load"},
        )
    if job.node.startswith("E2-r") and isinstance(job.load, str):
        common = state.selection("e1_common_load", None)
        if not isinstance(common, str):
            raise ScientificFailure("E2 cell lacks E1 measured common load")
        registered = int(job.load.removeprefix("c"))
        maximum = int(common.removeprefix("c"))
        effective_load = f"c{min(registered, maximum)}"
        job = replace(
            job,
            load=effective_load,
            parameters={
                **job.parameters,
                "registered_load": f"c{registered}",
                "effective_load": effective_load,
            },
        )
    if job.node.startswith("E4") and job.load in {"low", "moderate", "saturation"}:
        common = state.selection("e1_common_load", None)
        if not isinstance(common, str):
            raise ScientificFailure("E4 cell lacks E1 measured common load")
        maximum = int(common.removeprefix("c"))
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
    common = state.selection("e1_common_load", None)
    if (
        not isinstance(capacity, dict)
        or not isinstance(common, str)
    ):
        raise ScientificFailure("common-load cell lacks E1/E3a selections")
    width = job.width
    if job.method != "target_only" and width is None:
        width = int(capacity.get("width", 16))
    return replace(
        job,
        load=common,
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
    if original_job.parameters.get("failure") == "queue_saturation":
        return (0.0,) * count
    if workload == "topology_cohort_capacity":
        return (0.0,) * count
    if workload != "production_crossover" or not isinstance(registered, str):
        return None
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
    offsets = _e5_poisson_offsets(config, state, original_job)
    if offsets is None or len(offsets) != count:
        raise ScientificFailure("open-loop arrival trace is incomplete")
    return offsets


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


def _slo_metrics(
    results: Iterable[Any], outcomes: Iterable[dict[str, object]]
) -> tuple[float, bool]:
    rows = tuple(results)
    outcome_rows = tuple(outcomes)
    passed = 0
    for result in rows:
        ttft_limit = 2000 if result.input_tokens < 4096 else 5000 if result.input_tokens < 16384 else 10000
        itl_p99 = float(np.quantile(result.inter_token_ms or (0.0,), 0.99))
        passed += result.ttft_ms <= ttft_limit and itl_p99 <= 100
    offered = len(outcome_rows)
    completed = sum(row["status"] == "completed" for row in outcome_rows)
    errors = sum(
        row["status"] in {"error", "timed_out", "cancelled", "unfinished"}
        for row in outcome_rows
    )
    pass_rate = passed / offered if offered else 0.0
    completion_rate = completed / offered if offered else 0.0
    error_rate = errors / offered if offered else 1.0
    return pass_rate, (
        pass_rate >= 0.99 and error_rate <= 0.001 and completion_rate >= 0.999
    )


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
                return True
        return False
    elif failure == "duplicate_retry":
        request_id = "fault-duplicate-retry"
        trajectories = []
        for _ in range(2):
            rows, _ = client.run_batch(
                (prompt,),
                max_new_tokens=min(max_new_tokens, 8),
                seed=0,
                request_ids=(request_id,),
            )
            trajectories.append(rows[0].output_ids)
        return trajectories[0] == trajectories[1]
    return failure in {
        "nonfinite_candidate",
        "oom_candidate",
        "telemetry_backpressure",
        "disk_quota",
        "slow_rank",
        "communicator_failure",
        "replica_drain",
        "replica_restart",
        "queue_saturation",
    }


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
        return (
            outcomes["offered"] == 512
            and outcomes["admitted"] <= 256
            and outcomes["unfinished"] >= 256
        )
    if failure in {"slow_rank", "communicator_failure", "replica_drain", "replica_restart"}:
        return request_action_passed and metrics.get("recovery_health_passed") is True
    return request_action_passed


def _run_multi_turn(
    client,
    prompts,
    max_new_tokens: int,
    seed: int,
    *,
    request_scoped: bool,
    max_in_flight: int,
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
                client,
                histories,
                budget,
                seed + turn,
                request_prefix=f"multi-turn-{turn}",
            )
        else:
            scheduled = client.run_bounded(
                histories,
                max_new_tokens=budget,
                seed=seed + turn,
                request_ids=tuple(
                    f"multi-turn-{turn}-{index:05d}"
                    for index in range(len(histories))
                ),
                max_in_flight=max_in_flight,
            )
            turn_results, turn_elapsed = (
                scheduled.results,
                scheduled.elapsed_seconds,
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
    client,
    prompts,
    max_new_tokens: int,
    seed: int,
    *,
    same_seed: bool = False,
    request_prefix: str = "request-scoped",
    temperature: float = 0.0,
):
    started = time.perf_counter()
    results = []
    for index, prompt in enumerate(prompts):
        rows, _ = client.run_batch(
            (prompt,),
            max_new_tokens=max_new_tokens,
            seed=seed if same_seed else seed + index,
            temperature=temperature,
            request_ids=(f"{request_prefix}-{index:05d}",),
        )
        results.append(rows[0])
    return tuple(results), time.perf_counter() - started


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
        _write_jsonl(output_dir / "requests.jsonl.gz", ())
        session_files = {
            "server.log": server.output_dir / "server.log",
            "cycles.jsonl": server.output_dir / "cycles.jsonl",
            "gpu.csv": server.output_dir / "gpu.csv",
        }
        session_offsets = {
            name: path.stat().st_size if path.exists() else 0
            for name, path in session_files.items()
        }
        if isinstance(server, ReplicaServerProcess):
            for name in ("server.log", "cycles.jsonl", "gpu.csv"):
                key = f"replica-1/{name}"
                session_files[key] = server.output_dir / "replica-1" / name
                source = session_files[key]
                session_offsets[key] = source.stat().st_size if source.exists() else 0
        if server.process is not None:
            (output_dir / "server.pid").write_text(
                f"{server.process.pid}\n", encoding="utf-8"
            )
        offered = 0
        try:
            runtime_job = _runtime_job(state, job)
            raw_config = runtime_job.to_dict()
            raw_config["parameters"]["stimulus_id"] = _stimulus_id(runtime_job)
            raw_config["adaptation"] = adaptation_payload(runtime_job, selection)
            _write_json(output_dir / "config.json", raw_config)
            bootstrap_job = _exactness_bootstrap(runtime_job)
            client = server.configure(bootstrap_job, selection)
            prompts, max_new_tokens, workload = _cell_inputs(
                config, state, client, job
            )
            offered = len(prompts)
            exactness_rows: list[dict[str, object]] = []
            exactness_evidence: dict[str, object] | None = None
            if bootstrap_job is not runtime_job:
                pair_seed = config.protocol.seed + (job.block or 0)
                bootstrap_topology = str(
                    bootstrap_job.parameters.get("topology", "tp1_dp1")
                )
                bootstrap_before = _speed_metrics(
                    client.server_info(), bootstrap_topology
                )
                verified, _ = client.run_batch(
                    prompts[:4],
                    max_new_tokens=max_new_tokens,
                    seed=pair_seed,
                    request_id_prefix="controlled-speculative-verify",
                )
                bootstrap_after = _speed_metrics(
                    client.server_info(), bootstrap_topology
                )
                bootstrap_committed = int(bootstrap_after["committed_tokens"]) - int(
                    bootstrap_before["committed_tokens"]
                )
                _validate_committed_tokens(verified, bootstrap_committed)
                bootstrap_safety = {
                    counter: int(bootstrap_after[counter])
                    - int(bootstrap_before[counter])
                    for counter in SAFETY_COUNTERS
                }
                if any(bootstrap_safety.values()):
                    raise ScientificFailure(
                        f"deterministic verification raised safety counters: {bootstrap_safety}"
                    )
                checked_tokens = int(bootstrap_after.get("greedy_token_checks", 0)) - int(
                    bootstrap_before.get("greedy_token_checks", 0)
                )
                mismatched_tokens = int(
                    bootstrap_after.get("greedy_token_mismatches", 0)
                ) - int(bootstrap_before.get("greedy_token_mismatches", 0))
                coverage = _validate_greedy_verify_counts(
                    bootstrap_committed,
                    checked_tokens,
                    mismatched_tokens,
                    allowed_unverified=len(verified),
                )
                exactness_evidence = {
                    "mode": "deterministic_verification_kernel",
                    "committed_tokens": bootstrap_committed,
                    "greedy_token_checks": checked_tokens,
                    "greedy_token_mismatches": mismatched_tokens,
                    **coverage,
                    "safety_counters": bootstrap_safety,
                }
                exactness_rows.extend(
                    {
                        "policy": "speculative_verify",
                        "exactness_scope": "deterministic_verification_kernel",
                        **result.to_dict(),
                    }
                    for result in verified
                )
                runtime_job = replace(
                    runtime_job,
                    parameters={
                        key: value
                        for key, value in runtime_job.parameters.items()
                        if key != "deterministic_verify"
                    },
                )
                client = server.restart_for(runtime_job, selection)
                raw_config["exactness_bootstrap"] = bootstrap_job.to_dict()
                raw_config["runtime"] = runtime_job.to_dict()
                raw_config["adaptation"] = adaptation_payload(runtime_job, selection)
                _write_json(output_dir / "config.json", raw_config)
                if server.process is not None:
                    (output_dir / "server.pid").write_text(
                        f"{server.process.pid}\n", encoding="utf-8"
                    )
            if runtime_job.parameters.get("controlled_replay") or runtime_job.parameters.get(
                "controlled_pair_baseline"
            ):
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
                    request_id_prefix=f"{job.job_id}-probe",
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
            if config.server.warmup_requests and not runtime_job.parameters.get(
                "controlled_replay"
            ):
                warmup_budget = (
                    max_new_tokens[0]
                    if isinstance(max_new_tokens, tuple)
                    else max_new_tokens
                )
                warmup_deadline = (
                    time.monotonic() + E5_WARMUP_SECONDS
                    if job.node.startswith("E5")
                    else None
                )
                warmup_index = 0
                while True:
                    if hasattr(client, "warmup"):
                        client.warmup(
                            prompts[0],
                            max_new_tokens=min(16, warmup_budget),
                            seed=config.protocol.seed,
                        )
                    else:
                        client.run_batch(
                            prompts[: config.server.warmup_requests],
                            max_new_tokens=min(16, warmup_budget),
                            seed=config.protocol.seed,
                            request_id_prefix=f"{job.job_id}-warmup-{warmup_index}",
                        )
                        client.reset()
                    warmup_index += 1
                    if warmup_deadline is None or time.monotonic() >= warmup_deadline:
                        break
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
            profiler = runtime_job.parameters.get("profiler")
            if profiler in {"nvtx", "nsys", "ncu"}:
                client.start_profile(
                    output_dir=str(output_dir / "torch-profile"),
                    cuda_range=profiler in {"nsys", "ncu"},
                )
            before = _speed_metrics(client.server_info(), topology)
            arrivals = _arrival_offsets(config, state, job, runtime_job, len(prompts))
            scheduled: ScheduledRun | None = None
            concurrency = _cell_concurrency(runtime_job)
            request_scoped = _uses_request_scope(runtime_job)
            controlled_rows: list[dict[str, object]] = list(exactness_rows)
            if runtime_job.parameters.get("controlled_replay"):
                pair_seed = config.protocol.seed + (job.block or 0)
                capture = replace(
                    runtime_job,
                    method="tts",
                    parameters={
                        **runtime_job.parameters,
                        "controlled_candidate_role": "capture",
                    },
                )
                client = server.configure(capture, state.selection("tts_recipe", {}))
                tts_results, tts_elapsed = _run_request_scoped(
                    client,
                    prompts[:4],
                    max_new_tokens,
                    pair_seed,
                    same_seed=True,
                    request_prefix="controlled-tts",
                )
                compare = replace(
                    runtime_job,
                    method="l0_naive",
                    parameters={
                        **runtime_job.parameters,
                        "controlled_candidate_role": "compare",
                    },
                )
                client = server.configure(compare, state.selection("tts_recipe", {}))
                l0_results, l0_elapsed = _run_request_scoped(
                    client,
                    prompts[:4],
                    max_new_tokens,
                    pair_seed,
                    same_seed=True,
                    request_prefix="controlled-l0",
                )
                results = (*tts_results, *l0_results)
                elapsed = tts_elapsed + l0_elapsed
                controlled_rows.extend(
                    {"policy": "tts", **result.to_dict()} for result in tts_results
                )
                controlled_rows.extend(
                    {"policy": "l0_naive", **result.to_dict()}
                    for result in l0_results
                )
            elif runtime_job.parameters.get("regime") == "multi_turn_shared_prefix":
                if arrivals is not None:
                    raise ScientificFailure("multi-turn rows cannot use an open-loop trace")
                results, elapsed = _run_multi_turn(
                    client,
                    prompts,
                    max_new_tokens,
                    config.protocol.seed + (job.block or 0),
                    request_scoped=request_scoped,
                    max_in_flight=concurrency,
                )
            elif arrivals is None:
                seed = config.protocol.seed + (job.block or 0)
                if runtime_job.load and runtime_job.load.startswith("closed_loop_c"):
                    scheduled = client.run_closed_loop(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                        routing_keys=_routing_keys(config, runtime_job, len(prompts)),
                        max_in_flight=1 if request_scoped else concurrency,
                        duration_seconds=E5_HEADLINE_SECONDS,
                        deadline_seconds=E5_REQUEST_DEADLINE_SECONDS,
                        request_id_prefix=f"{job.job_id}-closed-loop",
                    )
                    results, elapsed = scheduled.results, scheduled.elapsed_seconds
                    if any(
                        outcome.status == "error" for outcome in scheduled.outcomes
                    ):
                        raise RuntimeError("closed-loop request failed")
                elif request_scoped:
                    results, elapsed = _run_request_scoped(
                        client,
                        prompts,
                        max_new_tokens,
                        seed,
                        same_seed=bool(runtime_job.parameters.get("controlled_replay")),
                        request_prefix=f"{job.job_id}-measure",
                    )
                else:
                    scheduled = client.run_bounded(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                        routing_keys=_routing_keys(config, runtime_job, len(prompts)),
                        request_ids=tuple(
                            f"{job.job_id}-measure-{index:05d}"
                            for index in range(len(prompts))
                        ),
                        max_in_flight=concurrency,
                        deadline_seconds=E5_REQUEST_DEADLINE_SECONDS,
                    )
                    results, elapsed = scheduled.results, scheduled.elapsed_seconds
            else:
                scheduled = client.run_scheduled(
                    prompts,
                    arrivals,
                    max_new_tokens=max_new_tokens,
                    seed=config.protocol.seed + (job.block or 0),
                    routing_keys=_routing_keys(config, runtime_job, len(prompts)),
                    request_ids=tuple(
                        f"{job.job_id}-scheduled-{index:05d}"
                        for index in range(len(prompts))
                    ),
                    max_in_flight=1 if request_scoped else 256,
                    deadline_seconds=E5_REQUEST_DEADLINE_SECONDS,
                    drain_seconds=E5_DRAIN_SECONDS,
                )
                results, elapsed = scheduled.results, scheduled.elapsed_seconds
            if profiler in {"nvtx", "nsys", "ncu"}:
                client.stop_profile()
            if request_scoped:
                _wait_request_scope_release(client)
            after = _speed_metrics(client.server_info(), topology)
            if runtime_job.parameters.get("controlled_pair_baseline"):
                pair_seed = config.protocol.seed + (job.block or 0)
                controlled, _ = client.run_batch(
                    prompts[:4],
                    max_new_tokens=max_new_tokens,
                    seed=pair_seed,
                    request_id_prefix="controlled-target",
                )
                controlled_rows = [
                    {"policy": "target_only", **result.to_dict()}
                    for result in controlled
                ]
            request_rows = [_request_metrics(result) for result in results]
            outcome_rows = (
                [outcome.to_dict() for outcome in scheduled.outcomes]
                if scheduled is not None
                else [
                    {
                        "request_id": result.request_id,
                        "status": "completed",
                        "offered_ns": 0,
                        "admitted_ns": 0,
                        "finished_ns": 0,
                        "error": None,
                    }
                    for result in results
                ]
            )
            _write_jsonl(output_dir / "requests.jsonl.gz", request_rows)
            _write_jsonl(output_dir / "request_outcomes.jsonl.gz", outcome_rows)
            if controlled_rows:
                _write_jsonl(output_dir / "controlled.jsonl.gz", controlled_rows)
            committed = int(after["committed_tokens"]) - int(before["committed_tokens"])
            output_tokens = _validate_committed_tokens(results, committed)
            peak_hbm = int(after["peak_hbm_bytes"])
            reserved_hbm = int(after["peak_hbm_reserved_bytes"])
            nvml_hbm = _peak_hbm_from_csv(
                server.output_dir / "gpu.csv", session_offsets["gpu.csv"]
            )
            energy_mj = _energy_from_csv(
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
                    "offered": len(outcome_rows),
                    "admitted": sum(row["admitted_ns"] is not None for row in outcome_rows),
                    "completed": sum(row["status"] == "completed" for row in outcome_rows),
                    "error": sum(row["status"] == "error" for row in outcome_rows),
                    "timed_out": sum(row["status"] == "timed_out" for row in outcome_rows),
                    "cancelled": sum(row["status"] == "cancelled" for row in outcome_rows),
                    "unfinished": sum(row["status"] == "unfinished" for row in outcome_rows),
                },
                "output_tokens": output_tokens,
                "ttft_p50_ms": float(np.median([result.ttft_ms for result in results])),
                "itl_p99_ms": float(native_itl),
                "rank_local_before": before["rank_local"],
                "rank_local_after": after["rank_local"],
                "rank_aggregates_before": before["rank_aggregates"],
                "rank_aggregates_after": after["rank_aggregates"],
                "energy_mj": energy_mj if energy_mj is not None else "N/A",
                "tokens_per_joule": (
                    committed / (energy_mj / 1000.0)
                    if energy_mj is not None and energy_mj > 0
                    else "N/A"
                ),
                "slo_requests_per_gpu_hour": (
                    sum(row["status"] == "completed" for row in outcome_rows)
                    / (elapsed * len(gpus) / 3600.0)
                ),
                "executed_flops": "N/A",
                "hbm_bytes_per_committed_token": "N/A",
            }
            if exactness_evidence is not None:
                metrics["exactness_bootstrap"] = exactness_evidence
            if runtime_job.parameters.get("probe"):
                metrics["compatible"] = True
                metrics["static_interface_passed"] = static_interface_passed
                metrics["adaptive_interface_passed"] = True
            if job.node == "E1a":
                metrics["fixed_verification_budget"] = (
                    8 if job.parameters.get("verification") == "fixed_budget" else None
                )
                metrics["confidence_loss_weight"] = float(
                    (selection or {}).get("confidence_loss_weight", 0.1)
                )
            slo_pass_rate, slo_pass = _slo_metrics(results, outcome_rows)
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
                "confidence_brier",
                "confidence_ece",
                "resident_bytes",
                "peak_bytes",
                "optimizer_bytes",
                "trainable_parameters",
                "memory_ledger",
                "parameter_layout",
                "graph_replay_hit_rate",
                "main_side_overlap_ratio",
                "max_batch_size",
                "max_queue_occupancy",
                "timings_ms",
                "updates",
                "rounds",
                "teacher_row_acquisitions",
                "active_version",
            ):
                if name in after:
                    value = after[name]
                    metrics[name] = (
                        value
                        if name
                        in {
                            "confidence_brier",
                            "confidence_ece",
                            "graph_replay_hit_rate",
                            "main_side_overlap_ratio",
                            "memory_ledger",
                            "parameter_layout",
                            "timings_ms",
                            "updates",
                            "rounds",
                            "teacher_row_acquisitions",
                        }
                        else value - before.get(name, 0)
                        if isinstance(value, (int, float))
                        else value
                    )
            survival = _position_survival(metrics.get("rounds"))
            metrics["position_conditional_survival"] = survival
            metrics["effective_target_token_batch"] = (
                float(sum(survival)) if isinstance(survival, list) else "N/A"
            )
            metrics["total_variation"] = "N/A"
            metrics["target_entropy"] = "N/A"
            metrics["target_top_token_draft_cross_entropy"] = "N/A"
            metrics["directional_condition_frequency"] = "N/A"
            metrics["projected_net_drift"] = "N/A"
            metrics["residual_magnitude"] = "N/A"
            metrics["theorem_bound_slack"] = "N/A"
            active_versions = after["rank_aggregates"].get("active_version")
            metrics["publication_skew"] = (
                active_versions["max"] - active_versions["min"]
                if isinstance(active_versions, dict)
                else "N/A"
            )
            for name in (
                "prefix_cache_hit_rate",
                "tenant_fairness",
                "collective_bytes",
                "collective_time_ms",
                "collective_wait_ms",
                "slowest_rank_exposure_ms",
                "straggler_time_ms",
                "precision",
            ):
                metrics[name] = after.get(name, "N/A")
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
                if after.get("controlled_candidate_compared") is not True:
                    raise ScientificFailure("controlled replay did not compare a candidate")
                if after.get("controlled_candidate_equal") is not True:
                    raise ScientificFailure("controlled replay changed the staged candidate")
                metrics["controlled_candidate_compared"] = True
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
            if (
                job.method in CANDIDATE_METHODS
                or job.parameters.get("interface_fit")
                or _screening_job(job)
            ):
                _write_json(
                    output_dir / "metrics.json",
                    {
                        "scientific_outcome": "rejected",
                        "feasible": False,
                        "error": str(error),
                        "request_outcomes": {
                            "offered": offered,
                            "admitted": 0,
                            "completed": 0,
                            "error": 0,
                            "timed_out": 0,
                            "cancelled": 0,
                            "unfinished": offered,
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
            if _screening_job(job) and _capacity_infeasible(
                error,
                session_files["server.log"],
                session_offsets["server.log"],
            ):
                _write_json(
                    output_dir / "metrics.json",
                    {
                        "scientific_outcome": "infeasible",
                        "feasible": False,
                        "error": f"{type(error).__name__}: {error}",
                        "request_outcomes": {
                            "offered": offered,
                            "admitted": 0,
                            "completed": 0,
                            "error": 0,
                            "timed_out": 0,
                            "cancelled": 0,
                            "unfinished": offered,
                        },
                    },
                )
                state.complete(job.job_id, attempt)
                server.restart()
                return
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
                target = output_dir / f"{name}.gz"
                target.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith("gpu.csv"):
                    header = (
                        b"timestamp,index,memory_used_mb,gpu_util_pct,"
                        b"memory_util_pct,power_w,energy_mj,temperature_c,"
                        b"sm_clock_mhz,pstate,throttle\n"
                    )
                    if payload.startswith(b"timestamp,"):
                        header = b""
                    target.write_bytes(gzip.compress(header + payload))
                else:
                    target.write_bytes(gzip.compress(payload))


def _selection_for_job(state: StateStore, job: Job) -> dict[str, Any] | None:
    if job.method in {"tts", "l0_naive"}:
        return state.selection("tts_recipe", {})
    if job.method == "lightcone":
        name = "dspark_recipe" if job.backend == "DSPARK" else "lightcone_recipe"
        selected = dict(state.selection(name, {}))
        if job.node.startswith("E6"):
            selected.update(parameterization="lora", rank=8, scope="all")
        return selected
    if job.node.startswith("E1a") and job.method == "lightcone_candidate":
        selected = dict(state.selection("lightcone_recipe", {}))
        weight = state.selection("dspark_confidence_weight", None)
        if isinstance(weight, (int, float)):
            selected["confidence_loss_weight"] = float(weight)
        return selected
    return None


def _retryable_process_error(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return 500 <= error.code < 600
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


def _gpu_pairs(config: ExperimentConfig) -> tuple[tuple[int, int], ...]:
    return tuple(zip(config.gpu_ids[::2], config.gpu_ids[1::2], strict=True))


def _assigned_pair(config: ExperimentConfig, job: Job) -> tuple[int, int]:
    pairs = _gpu_pairs(config)
    if job.node == "preflight":
        return pairs[0]
    index = job.block if job.block is not None else job.ordinal
    return pairs[index % len(pairs)]


def _assigned_gpu(config: ExperimentConfig, job: Job) -> int:
    gpu_index = job.parameters.get("gpu_index")
    if isinstance(gpu_index, int) and 0 <= gpu_index < len(config.gpu_ids):
        return config.gpu_ids[gpu_index]
    if job.block is not None:
        return config.gpu_ids[job.block % len(config.gpu_ids)]
    return config.gpu_ids[job.ordinal % len(config.gpu_ids)]


def _resource_port(config: ExperimentConfig, gpus: tuple[int, ...]) -> int:
    if len(gpus) == 1:
        return config.server.base_port + config.gpu_ids.index(gpus[0])
    return (
        config.server.base_port
        + len(config.gpu_ids)
        + 2 * _gpu_pairs(config).index((gpus[0], gpus[1]))
    )


def _job_gpus(config: ExperimentConfig, job: Job) -> tuple[int, ...]:
    if job.gpu_count == 2:
        return _assigned_pair(config, job)
    return (_assigned_gpu(config, job),)


def _pair_interference_jobs(config: ExperimentConfig) -> tuple[Job, ...]:
    rows = []
    for pair_index in range(len(_gpu_pairs(config))):
        for mode in ("isolated", "concurrent"):
            ordinal = 2 * pair_index + (mode == "concurrent")
            rows.append(
                Job(
                    job_id=f"GPU-pair-interference__{ordinal:06d}__{mode}-pair-{pair_index}",
                    node="GPU-pair-interference",
                    ordinal=ordinal,
                    method="static",
                    model="Qwen/Qwen3-8B",
                    backend="DFLASH",
                    task="controlled_baseline",
                    context=4096,
                    load="c8",
                    width=8,
                    block=pair_index,
                    gpu_count=2,
                    parameters={
                        "mode": mode,
                        "topology": "tp2_dp1",
                        "workload": "pair_interference",
                    },
                )
            )
    return tuple(rows)


def _select_pair_parallelism(state: StateStore, node: str) -> dict[str, Any]:
    timings: dict[int, dict[str, dict[str, float]]] = {}
    for item, metrics in _metric_rows(state, node):
        mode = item["parameters"].get("mode")
        block = item.get("block")
        if mode in {"isolated", "concurrent"} and isinstance(block, int):
            timings.setdefault(block, {})[mode] = {
                "goodput": float(metrics["goodput"]),
                "itl": float(metrics["itl_p99_ms"]),
            }
    pairs = [
        rows for rows in timings.values() if {"isolated", "concurrent"} <= rows.keys()
    ]
    intervals = {}
    if len(pairs) >= 3:
        for metric in ("goodput", "itl"):
            candidate = [rows["concurrent"][metric] for rows in pairs]
            baseline = [rows["isolated"][metric] for rows in pairs]
            intervals[metric] = paired_relative_bca_interval(candidate, baseline)
    return {
        "enabled": len(pairs) == len(timings) >= 3
        and all(
            abs(point) <= 0.01 and low <= 0 <= high
            for point, low, high in intervals.values()
        ),
        "paired_relative_bca": intervals,
    }


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


def _require_internal_jobs(state: StateStore, node: str) -> None:
    counts = state.status_counts(node)
    if counts.get("failed") or counts.get("pending") or counts.get("running"):
        raise ScientificFailure(f"internal substage {node} did not complete")


def _complete_infeasible_startup(
    state: StateStore,
    job: Job,
    run_dir: Path,
    gpus: tuple[int, ...],
    error: Exception,
) -> None:
    attempt_number = state.next_attempt(job.job_id)
    output_dir = run_dir / "jobs" / job.job_id / f"attempt-{attempt_number:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt = state.start(job, gpus, output_dir)
    _write_json(output_dir / "config.json", job.to_dict())
    _write_json(
        output_dir / "metrics.json",
        {
            "scientific_outcome": "infeasible",
            "feasible": False,
            "error": f"{type(error).__name__}: {error}",
            "request_outcomes": {
                "offered": 0,
                "admitted": 0,
                "completed": 0,
                "error": 0,
                "timed_out": 0,
                "cancelled": 0,
                "unfinished": 0,
            },
        },
    )
    state.complete(job.job_id, attempt)


def _confidence_calibration_jobs(state: StateStore) -> tuple[Job, ...]:
    common = state.selection("e1_common_load", None)
    if not isinstance(common, str):
        raise ScientificFailure("E1a confidence calibration lacks E1 common load")
    return tuple(
        Job(
            job_id=f"e1a-confidence-weight-{str(weight).replace('.', 'p')}",
            node="E1a-confidence-calibration",
            ordinal=index,
            method="lightcone_candidate",
            model="Qwen/Qwen3-8B",
            backend="DSPARK",
            task="CalibrationMix",
            context=40928,
            load=common,
            width=16,
            parameters={
                "scope": "last5_native_heads",
                "parameterization": "full",
                "rank": None,
                "verification": "native_scheduler",
                "confidence_loss_weight": weight,
                "regime": "short_input_long_generation",
                "generation_tokens": TTS_GENERATION_TOKENS,
                "workload": "excluded_confidence_calibration",
            },
        )
        for index, weight in enumerate(CONFIDENCE_WEIGHTS)
    )


def _e1_load_jobs(state: StateStore) -> tuple[Job, ...]:
    geometries = _rank_e1_geometries(state)
    if not geometries:
        raise ScientificFailure("E1 reference screen has no two-anchor Pareto geometry")
    rows: list[Job] = []
    ordinal = 0
    for concurrency in (1, 2, 4, 8, 16, 32, 64):
        load = f"c{concurrency}"
        for method, backend in (("target_only", "NONE"), ("static", "DFLASH")):
            rows.append(
                Job(
                    f"e1-load-{load}-{method}",
                    "E1-common-load",
                    ordinal,
                    method,
                    "Qwen/Qwen3-8B",
                    backend,
                    "CalibrationMix",
                    context=40928,
                    load=load,
                    width=None if method == "target_only" else 16,
                    parameters={
                        "regime": "short_input_long_generation",
                        "generation_tokens": TTS_GENERATION_TOKENS,
                        "workload": "excluded_common_load_probe",
                    },
                )
            )
            ordinal += 1
        for geometry_index, geometry in enumerate(geometries):
            for optimizer in ("adamw", "sgdm"):
                rows.append(
                    Job(
                        f"e1-load-{load}-g{geometry_index:02d}-{optimizer}",
                        "E1-common-load",
                        ordinal,
                        "lightcone_candidate",
                        "Qwen/Qwen3-8B",
                        "DFLASH",
                        "CalibrationMix",
                        context=40928,
                        load=load,
                        width=16,
                        parameters={
                            **geometry,
                            "optimizer": optimizer,
                            "regime": "short_input_long_generation",
                            "generation_tokens": TTS_GENERATION_TOKENS,
                            "workload": "excluded_common_load_probe",
                        },
                    )
                )
                ordinal += 1
    return tuple(rows)


def _select_e1_common_load(state: StateStore, expected_per_load: int) -> str:
    counts = Counter(
        str(config["load"])
        for config, metrics in _metric_rows(state, "E1-common-load")
        if metrics.get("slo_pass") is True
        and metrics.get("feasible") is not False
        and all(metrics.get(counter, 0) == 0 for counter in SAFETY_COUNTERS)
    )
    feasible = [
        load
        for load, count in counts.items()
        if count == expected_per_load
    ]
    if not feasible:
        raise ScientificFailure("E1 found no common safe adaptation load")
    return max(feasible, key=lambda value: int(value.removeprefix("c")))


def _safe_screen_row(metrics: dict[str, Any]) -> bool:
    return (
        metrics.get("feasible") is not False
        and metrics.get("slo_pass") is True
        and isinstance(metrics.get("peak_hbm_bytes"), (int, float))
        and float(metrics["peak_hbm_bytes"]) > 0
        and isinstance(metrics.get("kv_capacity"), (int, float))
        and float(metrics["kv_capacity"]) > 0
        and all(metrics.get(counter, 0) == 0 for counter in SAFETY_COUNTERS)
    )


def _select_confidence_weight(state: StateStore) -> float:
    candidates = []
    for config, metrics in _metric_rows(state, "E1a-confidence-calibration"):
        brier = metrics.get("confidence_brier")
        ece = metrics.get("confidence_ece")
        required = (
            brier,
            ece,
            metrics.get("goodput"),
            metrics.get("peak_hbm_bytes"),
        )
        if (
            metrics.get("slo_pass") is not True
            or metrics.get("feasible") is False
            or any(metrics.get(counter) != 0 for counter in SAFETY_COUNTERS)
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in required
            )
        ):
            continue
        candidates.append(
            (
                float(brier),
                float(ece),
                -float(metrics["goodput"]),
                int(metrics["peak_hbm_bytes"]),
                float(config["parameters"]["confidence_loss_weight"]),
            )
        )
    if len(candidates) != len(CONFIDENCE_WEIGHTS):
        raise ScientificFailure("DSpark confidence calibration is incomplete")
    return min(candidates)[-1]


def _deployment_width_jobs(state: StateStore) -> tuple[Job, ...]:
    common = state.selection("e1_common_load", None)
    if not isinstance(common, str):
        raise ScientificFailure("deployment width tuning lacks E1 common load")
    tasks = {
        "long_input_short_output": "CalibrationMix",
        "short_input_long_generation": "CalibrationMix",
        "multi_turn_shared_prefix": "CalibrationMix",
    }
    rows = []
    ordinal = 0
    for method in ("static", "tts", "l0_naive", "lightcone"):
        for width in (4, 8, 16):
            for regime, task in tasks.items():
                rows.append(
                    Job(
                        f"e3-width-{method}-{width}-{regime}",
                        "E3-width-calibration",
                        ordinal,
                        method,
                        "Qwen/Qwen3-8B",
                        "DFLASH",
                        task,
                        context=40928,
                        load=common,
                        width=width,
                        parameters={
                            "regime": regime,
                            "workload": "excluded_deployment_width_tuning",
                        },
                    )
                )
                ordinal += 1
    return tuple(rows)


def _select_deployment_widths(state: StateStore) -> dict[str, int]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for config, metrics in _metric_rows(state, "E3-width-calibration"):
        if metrics.get("slo_pass") is True:
            groups.setdefault((config["method"], int(config["width"])), []).append(
                metrics
            )
    selected = {}
    for method in ("static", "tts", "l0_naive", "lightcone"):
        candidates = []
        for width in (4, 8, 16):
            rows = groups.get((method, width), [])
            if len(rows) == 3:
                candidates.append(
                    (
                        -float(np.mean([row["goodput"] for row in rows])),
                        max(int(row["peak_hbm_bytes"]) for row in rows),
                        width,
                    )
                )
        if not candidates:
            raise ScientificFailure(f"deployment width tuning failed for {method}")
        selected[method] = min(candidates)[-1]
    return selected


def _e6_load_jobs() -> tuple[Job, ...]:
    models = ("Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-122B-A10B-FP8")
    roles = ("target_only", "static", "tts", "l0_naive", "lightcone")
    rows = []
    ordinal = 0
    for model in models:
        for concurrency in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            for role in roles:
                rows.append(
                    Job(
                        f"e6-load-{model.rsplit('/', 1)[-1]}-c{concurrency}-{role}",
                        "E6-common-load",
                        ordinal,
                        role,
                        model,
                        "NONE" if role == "target_only" else "NEXTN",
                        "LiveCodeBench",
                        context=40928,
                        load=f"c{concurrency}",
                        width=None if role == "target_only" else 16,
                        gpu_count=2,
                        parameters={"workload": "excluded_e6_common_load_probe"},
                    )
                )
                ordinal += 1
    return tuple(rows)


def _e6_interface_jobs(parents: tuple[Job, ...]) -> tuple[Job, ...]:
    rows = []
    for parent in parents:
        for mode in ("lora", "full"):
            rows.append(
                replace(
                    parent,
                    job_id=f"{parent.job_id}-{mode}",
                    node="E6-interface",
                    ordinal=len(rows),
                    parameters={
                        **parent.parameters,
                        "interface_fit": True,
                        "interface_parent": parent.job_id,
                        "parameterization": mode,
                        "rank": 8 if mode == "lora" else None,
                        "scope": "all",
                        "minimum_updates": 1,
                    },
                )
            )
    return tuple(rows)


def _e6_role_supported(role: str, modes: set[str]) -> bool:
    if role == "lightcone":
        return "lora" in modes
    if role in {"tts", "l0_naive"}:
        return "full" in modes
    return True


def _complete_e6_interface_rows(
    config: ExperimentConfig,
    state: StateStore,
    parents: tuple[Job, ...],
) -> dict[str, set[str]]:
    components: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = {}
    for raw_config, metrics in _metric_rows(state, "E6-interface"):
        parameters = raw_config.get("parameters", {})
        parent = parameters.get("interface_parent")
        mode = parameters.get("parameterization")
        if isinstance(parent, str) and mode in {"lora", "full"}:
            components.setdefault(parent, {})[str(mode)] = (raw_config, metrics)
    capabilities: dict[str, set[str]] = {}
    pending = {job.job_id for job in state.pending_jobs("E6-pilot")}
    for parent in parents:
        modes = components.get(parent.job_id, {})
        if set(modes) != {"lora", "full"}:
            raise ScientificFailure(f"{parent.job_id} lacks LoRA/Full interface results")
        supported = {
            mode
            for mode, (_, metrics) in modes.items()
            if metrics.get("feasible") is not False
        }
        capabilities[parent.model] = supported
        feasible = supported == {"lora", "full"}
        if parent.job_id not in pending:
            continue
        attempt_number = state.next_attempt(parent.job_id)
        output_dir = (
            config.run_dir
            / "jobs"
            / parent.job_id
            / f"attempt-{attempt_number:02d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        attempt = state.start(parent, _job_gpus(config, parent), output_dir)
        _write_json(output_dir / "config.json", parent.to_dict())
        _write_json(
            output_dir / "metrics.json",
            {
                "scientific_outcome": "completed" if feasible else "blocked",
                "feasible": feasible,
                "slo_pass": feasible,
                "components": {
                    mode: {
                        "feasible": metrics.get("feasible") is not False,
                        "error": metrics.get("error"),
                        "source_attempt": metrics.get("source_attempt"),
                        "source_attempt_dir": metrics.get("source_attempt_dir"),
                    }
                    for mode, (_, metrics) in modes.items()
                },
                "request_outcomes": {
                    "offered": sum(
                        int(row[1].get("request_outcomes", {}).get("offered", 0))
                        for row in modes.values()
                    ),
                    "completed": sum(
                        int(row[1].get("request_outcomes", {}).get("completed", 0))
                        for row in modes.values()
                    ),
                },
            },
        )
        state.complete(parent.job_id, attempt)
    return capabilities


def _select_e6_common_loads(
    state: StateStore, capabilities: dict[str, set[str]]
) -> dict[str, str]:
    counts: Counter[tuple[str, str]] = Counter()
    for config, metrics in _metric_rows(state, "E6-common-load"):
        if (
            metrics.get("slo_pass") is True
            and metrics.get("feasible") is not False
            and all(metrics.get(counter, 0) == 0 for counter in SAFETY_COUNTERS)
        ):
            counts[(str(config["model"]), str(config["load"]))] += 1
    selected = {}
    for model in {key[0] for key in counts}:
        required = sum(
            _e6_role_supported(role, capabilities.get(model, set()))
            for role in ("target_only", "static", "tts", "l0_naive", "lightcone")
        )
        loads = [
            load
            for (candidate, load), count in counts.items()
            if candidate == model and count == required
        ]
        if loads:
            selected[model] = max(
                loads, key=lambda value: int(value.removeprefix("c"))
            )
    return selected


def _e5_p99_jobs(state: StateStore) -> tuple[Job, ...]:
    winners: dict[tuple[int, str, str], tuple[float, dict[str, Any], dict[str, Any]]] = {}
    for config, metrics in _metric_rows(state, "E5-final"):
        parameters = config.get("parameters", {})
        registered = parameters.get("registered_load")
        if (
            not isinstance(config.get("block"), int)
            or not isinstance(registered, str)
            or not registered.startswith("lambda_")
            or metrics.get("slo_pass") is not True
            or not isinstance(metrics.get("offered_rate"), (int, float))
        ):
            continue
        key = (int(config["block"]), str(config["backend"]), str(config["method"]))
        rate = float(metrics["offered_rate"])
        if key not in winners or rate > winners[key][0]:
            winners[key] = (rate, config, metrics)
    rows = []
    for ordinal, ((block, backend, method), (rate, config, _)) in enumerate(
        sorted(winners.items())
    ):
        rows.append(
            Job(
                f"e5-p99-b{block:02d}-{backend.lower()}-{method}",
                "E5-p99-extension",
                ordinal,
                method,
                str(config["model"]),
                backend,
                str(config["task"]),
                context=int(config["context"]),
                load=str(config["load"]),
                width=config.get("width"),
                block=block,
                gpu_count=2,
                parameters={
                    **config["parameters"],
                    "registered_load": config["parameters"]["registered_load"],
                    "arrival_rate": rate,
                    "p99_extension": True,
                    "workload": "production_crossover",
                },
            )
        )
    return tuple(rows)


def _run_pending_jobs(
    config: ExperimentConfig,
    state: StateStore,
    node: str,
    stop_event: threading.Event,
    pending: tuple[Job, ...],
) -> None:
    exclusive = tuple(job for job in pending if job.gpu_count == 2)
    singles = tuple(job for job in pending if job.gpu_count == 1)
    node_failed = threading.Event()

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
            process_job = _exactness_bootstrap(runtime_job)
            key = (job.block, probe, *server_session_key(process_job, selection))
            grouped.setdefault(key, []).append((job, runtime_job, selection))
        keys = []
        blocks = sorted({key[0] for key in grouped}, key=lambda value: (-1 if value is None else value))
        for block in blocks:
            block_keys = [key for key in grouped if key[0] == block]
            rng = np.random.default_rng(
                config.protocol.seed + sum(gpus) + len(node) + int(block or 0)
            )
            rng.shuffle(block_keys)
            keys.extend(block_keys)
        for key in keys:
            if stop_event.is_set() or node_failed.is_set():
                return
            rows = grouped[key]
            first_job, first_runtime, first_selection = rows[0]
            first_process_job = _exactness_bootstrap(first_runtime)
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
                    first_process_job,
                    gpus=gpus,
                    port=port,
                    output_dir=session_dir,
                    selection=first_selection,
                )
                try:
                    with process:
                        for job, _, selection in rows:
                            if stop_event.is_set() or node_failed.is_set():
                                return
                            _execute_cell(
                                config,
                                state,
                                job,
                                gpus=gpus,
                                selection=selection,
                                server=process,
                            )
                            if state.status_counts(node).get("failed"):
                                node_failed.set()
                                return
                    break
                except Exception as error:
                    if _screening_job(first_job) and _capacity_infeasible(error):
                        for job, _, _ in rows:
                            _complete_infeasible_startup(
                                state, job, config.run_dir, gpus, error
                            )
                        break
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

    headline = node in {"E3b-final", "E5-final", "E6-final", "E0-final"}
    calibration = state.selection("headline_parallel", {"enabled": False})
    exclusive_queues = {
        pair: tuple(job for job in exclusive if _assigned_pair(config, job) == pair)
        for pair in _gpu_pairs(config)
    }
    exclusive_queues = {
        pair: jobs for pair, jobs in exclusive_queues.items() if jobs
    }
    exclusive_workers = (
        len(exclusive_queues)
        if not headline or calibration.get("enabled")
        else min(1, len(exclusive_queues))
    )
    if exclusive_workers:
        with ThreadPoolExecutor(
            max_workers=exclusive_workers,
            thread_name_prefix="lightcone-pair",
        ) as pool:
            futures = [
                pool.submit(
                    run_sessions,
                    jobs,
                    gpus=pair,
                    port=_resource_port(config, pair),
                    label=f"gpu-pair-{pair[0]}-{pair[1]}",
                )
                for pair, jobs in exclusive_queues.items()
            ]
            for future in futures:
                future.result()
    if node == "preflight":
        isolated = [job for job in singles if job.parameters.get("mode") == "isolated"]
        for job in isolated:
            gpu = _assigned_gpu(config, job)
            run_sessions(
                (job,),
                gpus=(gpu,),
                port=_resource_port(config, (gpu,)),
                label=f"isolated-gpu-{gpu}",
            )
        concurrent = [job for job in singles if job.parameters.get("mode") == "concurrent"]
        for block in sorted({job.block for job in concurrent}):
            rows = [job for job in concurrent if job.block == block]
            with ThreadPoolExecutor(max_workers=len(rows)) as pool:
                futures = []
                for job in rows:
                    gpu = _assigned_gpu(config, job)
                    futures.append(
                        pool.submit(
                            run_sessions,
                            (job,),
                            gpus=(gpu,),
                            port=_resource_port(config, (gpu,)),
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
        port = _resource_port(config, (gpu,))
        run_sessions(jobs, gpus=(gpu,), port=port, label=f"gpu-{gpu}")

    queues = {gpu: jobs for gpu, jobs in queues.items() if jobs}
    workers = len(queues) if not headline or calibration.get("enabled") else min(1, len(queues))
    if workers:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="lightcone-gpu"
        ) as pool:
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
    if (
        node == "preflight"
        and len(_gpu_pairs(config)) > 1
        and state.selection("pair_interference_complete", None) is None
    ):
        calibration = _pair_interference_jobs(config)
        state.add_internal_jobs(calibration)
        for job in state.pending_jobs("GPU-pair-interference"):
            if job.parameters.get("mode") == "isolated":
                _run_pending_jobs(
                    config,
                    state,
                    "GPU-pair-interference",
                    stop_event,
                    (job,),
                )
        concurrent = tuple(
            job
            for job in state.pending_jobs("GPU-pair-interference")
            if job.parameters.get("mode") == "concurrent"
        )
        _run_pending_jobs(
            config,
            state,
            "GPU-pair-interference",
            stop_event,
            concurrent,
        )
        if stop_event.is_set():
            return
        _require_internal_jobs(state, "GPU-pair-interference")
        state.set_selection(
            "headline_parallel",
            _select_pair_parallelism(state, "GPU-pair-interference"),
        )
        state.set_selection("pair_interference_complete", True)
    if node == "E0-tune":
        for job in pending:
            if not job.parameters.get("probe") or config.has_exact_draft(
                job.model, job.backend
            ):
                continue
            attempt_number = state.next_attempt(job.job_id)
            output_dir = (
                config.run_dir
                / "jobs"
                / job.job_id
                / f"attempt-{attempt_number:02d}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            attempt = state.start(job, _job_gpus(config, job), output_dir)
            _write_json(output_dir / "config.json", job.to_dict())
            _write_json(
                output_dir / "metrics.json",
                {
                    "scientific_outcome": "N/A",
                    "compatible": False,
                    "reason": "exact target-drafter pair is not configured",
                    "request_outcomes": {
                        "offered": 0,
                        "admitted": 0,
                        "completed": 0,
                        "error": 0,
                        "timed_out": 0,
                        "cancelled": 0,
                        "unfinished": 0,
                    },
                },
            )
            state.complete(job.job_id, attempt)
        pending = state.pending_jobs(node)
    if node == "E3b-pilot" and state.selection(
        "deployment_widths_tuned", None
    ) is None:
        width_jobs = _deployment_width_jobs(state)
        state.add_internal_jobs(width_jobs)
        _run_pending_jobs(
            config,
            state,
            "E3-width-calibration",
            stop_event,
            state.pending_jobs("E3-width-calibration"),
        )
        if stop_event.is_set():
            return
        _require_internal_jobs(state, "E3-width-calibration")
        widths = _select_deployment_widths(state)
        state.set_selection("deployment_widths", widths)
        state.set_selection("deployment_widths_tuned", True)
        pending = state.pending_jobs(node)
    if node == "E1a" and state.selection("dspark_confidence_weight", None) is None:
        calibration = _confidence_calibration_jobs(state)
        state.add_internal_jobs(calibration)
        _run_pending_jobs(
            config,
            state,
            "E1a-confidence-calibration",
            stop_event,
            state.pending_jobs("E1a-confidence-calibration"),
        )
        if stop_event.is_set():
            return
        _require_internal_jobs(state, "E1a-confidence-calibration")
        state.set_selection(
            "dspark_confidence_weight", _select_confidence_weight(state)
        )
        pending = state.pending_jobs(node)
    if node.startswith("E5"):
        anchors, _ = _e5_execution_phases(pending)
        if anchors:
            _run_pending_jobs(config, state, node, stop_event, anchors)
        if stop_event.is_set():
            return
        pending = state.pending_jobs(node)
    if node == "E6-pilot" and state.selection("e6_common_loads", None) is None:
        interfaces = tuple(
            job for job in state.jobs(node) if job.parameters.get("interface_fit")
        )
        if interfaces:
            state.add_internal_jobs(_e6_interface_jobs(interfaces))
            _run_pending_jobs(
                config,
                state,
                "E6-interface",
                stop_event,
                state.pending_jobs("E6-interface"),
            )
            if stop_event.is_set():
                return
            _require_internal_jobs(state, "E6-interface")
            capabilities = _complete_e6_interface_rows(
                config, state, interfaces
            )
            probes = tuple(
                job
                for job in _e6_load_jobs()
                if job.model in capabilities
                and _e6_role_supported(job.method, capabilities[job.model])
            )
            state.add_internal_jobs(probes)
            _run_pending_jobs(
                config,
                state,
                "E6-common-load",
                stop_event,
                state.pending_jobs("E6-common-load"),
            )
            _require_internal_jobs(state, "E6-common-load")
            state.set_selection(
                "e6_capabilities",
                {model: sorted(modes) for model, modes in capabilities.items()},
            )
            state.set_selection(
                "e6_common_loads", _select_e6_common_loads(state, capabilities)
            )
    if node in {"E6-pilot", "E6-final"}:
        if stop_event.is_set():
            return
        pending = state.pending_jobs(node)
        loads = state.selection("e6_common_loads", {})
        capabilities = {
            model: set(modes)
            for model, modes in state.selection("e6_capabilities", {}).items()
        }
        for job in pending:
            if not _e6_role_supported(job.method, capabilities.get(job.model, set())):
                state.skip_job(job.job_id, "required NEXTN update mode is infeasible")
            elif job.model not in loads:
                state.skip_job(job.job_id, "model has no feasible TP2 NEXTN common load")
        pending = state.pending_jobs(node)
    _run_pending_jobs(config, state, node, stop_event, pending)
    if node == "E5-final" and state.selection("e5_p99_extension_complete", None) is None:
        extensions = _e5_p99_jobs(state)
        state.add_internal_jobs(extensions)
        _run_pending_jobs(
            config,
            state,
            "E5-p99-extension",
            stop_event,
            state.pending_jobs("E5-p99-extension"),
        )
        if not stop_event.is_set():
            _require_internal_jobs(state, "E5-p99-extension")
            state.set_selection("e5_p99_extension_complete", True)
    if node == "E1" and state.selection("e1_common_load", None) is None:
        probes = _e1_load_jobs(state)
        state.add_internal_jobs(probes)
        _run_pending_jobs(
            config,
            state,
            "E1-common-load",
            stop_event,
            state.pending_jobs("E1-common-load"),
        )
        if stop_event.is_set():
            return
        _require_internal_jobs(state, "E1-common-load")
        per_load = len(probes) // 7
        state.set_selection(
            "e1_common_load", _select_e1_common_load(state, per_load)
        )
        state.set_selection("e1_geometries", _rank_e1_geometries(state))


def _metric_rows(state: StateStore, node: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for directory in state.completed_attempt_dirs(node):
        config_path, metrics_path = directory / "config.json", directory / "metrics.json"
        if config_path.is_file() and metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            metrics["source_attempt"] = int(
                directory.name.removeprefix("attempt-")
            )
            metrics["source_attempt_dir"] = str(directory)
            rows.append((json.loads(config_path.read_text()), metrics))
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
    groups: dict[
        tuple[Any, ...], dict[str, list[tuple[int, ...]]]
    ] = {}
    preflight_policies: set[str] = set()
    for directory in state.completed_attempt_dirs(node):
        config_path = directory / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("parameters", {}).get("probe"):
            continue
        if node == "preflight":
            controlled_path = _jsonl_path(directory, "controlled.jsonl")
            if not controlled_path.is_file():
                continue
            for row in _read_jsonl(controlled_path):
                preflight_policies.add(str(row["policy"]))
                policy = str(row["policy"])
                groups.setdefault(_trajectory_group(config), {}).setdefault(
                    policy, []
                ).append(tuple(row["output_ids"]))
            continue
    if node == "preflight":
        required = {"target_only", "speculative_verify", "tts", "l0_naive"}
        if preflight_policies != required:
            raise ScientificFailure("preflight lacks target/verify/TTS/L0 trajectories")
    diagnostics: list[dict[str, object]] = []
    for group, policies in groups.items():
        baseline = tuple(policies.get("target_only", ()))
        if not baseline:
            continue
        for method, method_rows in policies.items():
            if method == "target_only":
                continue
            trajectory = tuple(method_rows)
            mismatch_count = 0
            first_mismatch = None
            request_count = max(len(baseline), len(trajectory))
            for request_index in range(request_count):
                left = baseline[request_index] if request_index < len(baseline) else ()
                right = trajectory[request_index] if request_index < len(trajectory) else ()
                token_count = max(len(left), len(right))
                for token_index in range(token_count):
                    left_token = left[token_index] if token_index < len(left) else None
                    right_token = right[token_index] if token_index < len(right) else None
                    if left_token != right_token:
                        mismatch_count += 1
                        if first_mismatch is None:
                            first_mismatch = {
                                "request_index": request_index,
                                "token_index": token_index,
                                "target_token": left_token,
                                "method_token": right_token,
                            }
            diagnostics.append(
                {
                    "group": list(group),
                    "method": method,
                    "equal": mismatch_count == 0,
                    "mismatch_count": mismatch_count,
                    "first_mismatch": first_mismatch,
                }
            )
    diagnostic_path = state.run_dir / "stages" / node / "greedy_trajectory_diagnostics.json"
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        diagnostic_path,
        {
            "interpretation": (
                "excluded implementation smoke; it is not a benchmark-quality result"
            ),
            "comparisons": diagnostics,
        },
    )


def _rank_candidates(state: StateStore, node: str, keep: int) -> list[dict[str, Any]]:
    candidates = [
        (
            -float(metrics["goodput"]),
            int(metrics["peak_hbm_bytes"]),
            float(metrics["itl_p99_ms"]),
            float(metrics.get("confidence_brier", 0.0)),
            float(metrics.get("confidence_ece", 0.0)),
            config["parameters"],
        )
        for config, metrics in _metric_rows(state, node)
        if config["method"] in {"lightcone_candidate", "lightcone"}
        and metrics.get("feasible") is not False
    ]
    candidates.sort(key=lambda item: item[:-1])
    return [row[-1] for row in candidates[:keep]]


def _rank_e1_geometries(state: StateStore) -> list[dict[str, Any]]:
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for config, metrics in _metric_rows(state, "E1"):
        if config["method"] != "lightcone_candidate":
            continue
        if metrics.get("feasible") is False:
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
        if metrics.get("feasible") is False:
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


def _natural_spline_fit(
    x: np.ndarray, y: np.ndarray, evaluation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = float(x.min()), float(x.max())
    interior = np.log(np.asarray([4096, 16384, 32768], dtype=np.float64))
    if not left < interior[0] < interior[1] < interior[2] < right:
        raise ValueError("natural spline data do not bracket the fixed knots")
    knots = np.asarray([left] * 4 + interior.tolist() + [right] * 4)
    coefficients = np.eye(len(knots) - 4)

    def basis(points: np.ndarray, derivative: int = 0) -> np.ndarray:
        return np.column_stack(
            [
                BSpline(knots, row, 3)(points, nu=derivative)
                for row in coefficients
            ]
        )

    constraints = np.vstack(
        (basis(np.asarray([left]), 2), basis(np.asarray([right]), 2))
    )
    natural = null_space(constraints)
    design = basis(x) @ natural
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return tuple(
        basis(evaluation, derivative) @ natural @ beta
        for derivative in (0, 1, 2)
    )


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
        tuple[object, ...],
        dict[int, dict[int, tuple[list[tuple[int, float]], float]]],
    ] = {}
    sources: dict[tuple[object, ...], list[dict[str, object]]] = {}
    request_counts: Counter[tuple[object, ...]] = Counter()
    for directory in state.completed_attempt_dirs(node):
        config_path = directory / "config.json"
        metrics_path = directory / "metrics.json"
        requests_path = _jsonl_path(directory, "requests.jsonl")
        if (
            not config_path.is_file()
            or not metrics_path.is_file()
            or not requests_path.is_file()
        ):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        duration = metrics.get("duration_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            continue
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
        for request in _read_jsonl(requests_path):
            rows.append(
                (
                    int(request["completion_tokens"]),
                    float(request["elapsed_seconds"]),
                )
            )
        if rows:
            request_groups.setdefault(key, {}).setdefault(context, {})[block] = (
                rows,
                float(duration),
            )
            request_counts[key] += len(rows)
            sources.setdefault(key, []).append(
                {
                    "job_id": config.get("job_id"),
                    "attempt": int(directory.name.removeprefix("attempt-")),
                    "attempt_dir": str(directory),
                }
            )
    result = []
    for key, values in grouped.items():
        if len(values) < 4:
            continue
        contexts = np.asarray(sorted(values), dtype=np.float64)
        goodput = np.asarray(
            [np.mean(values[int(context)]) for context in contexts], dtype=np.float64
        )
        if np.any(goodput <= 0):
            continue
        fixed = np.asarray([4096, 16384, 32768], dtype=np.float64)
        if not all(int(value) in values for value in fixed):
            continue
        log_contexts = np.log(contexts)
        fitted, elasticity, curvature = _natural_spline_fit(
            log_contexts, np.log(goodput), log_contexts
        )
        by_context = request_groups.get(key, {})
        common_blocks = set.intersection(
            *(set(by_context.get(int(context), {})) for context in contexts)
        ) if len(contexts) else set()
        bootstrap_elasticity = []
        bootstrap_curvature = []
        bootstrap_fitted = []
        if len(common_blocks) >= 3:
            block_ids = np.asarray(sorted(common_blocks))
            rng = np.random.default_rng(0)
            for _ in range(1000):
                sampled_blocks = rng.choice(block_ids, size=len(block_ids), replace=True)
                sampled_goodput = []
                for context in contexts.astype(int):
                    points = []
                    for block in sampled_blocks:
                        raw_rows, duration = by_context[context][int(block)]
                        rows = np.asarray(raw_rows, dtype=np.float64)
                        indexes = rng.integers(0, len(rows), size=len(rows))
                        draw = rows[indexes]
                        points.append(float(np.sum(draw[:, 0]) / duration))
                    sampled_goodput.append(float(np.mean(points)))
                draw_fitted, draw_elasticity, draw_curvature = _natural_spline_fit(
                    log_contexts,
                    np.log(np.asarray(sampled_goodput)),
                    log_contexts,
                )
                bootstrap_fitted.append(draw_fitted)
                bootstrap_elasticity.append(draw_elasticity)
                bootstrap_curvature.append(draw_curvature)
        elasticity_ci = (
            np.quantile(np.asarray(bootstrap_elasticity), (0.025, 0.975), axis=0)
            if bootstrap_elasticity
            else None
        )
        curvature_ci = (
            np.quantile(np.asarray(bootstrap_curvature), (0.025, 0.975), axis=0)
            if bootstrap_curvature
            else None
        )
        result.append(
            {
                "method": key[0],
                "regime": key[1],
                "load": key[2],
                "width_panel": key[3],
                "contexts": contexts.astype(int).tolist(),
                "goodput": goodput.tolist(),
                "spline_knots": fixed.astype(int).tolist(),
                "fitted_goodput": np.exp(fitted).tolist(),
                "elasticity": elasticity.tolist(),
                "elasticity_ci95": None if elasticity_ci is None else elasticity_ci.tolist(),
                "curvature": curvature.tolist(),
                "curvature_ci95": None if curvature_ci is None else curvature_ci.tolist(),
                "bootstrap_refits": len(bootstrap_elasticity),
                "request_count": request_counts[key],
                "source_attempts": sources.get(key, []),
                "reducer": "natural_log_context_spline_block_request_bootstrap",
                "_bootstrap_log_fitted": [row.tolist() for row in bootstrap_fitted],
            }
        )
    return result


def _first_crossover(contexts: list[int], differences: np.ndarray) -> float | None:
    for index in range(len(contexts) - 1):
        left, right = float(differences[index]), float(differences[index + 1])
        if left == 0:
            return float(contexts[index])
        if left * right < 0:
            x0, x1 = math.log(contexts[index]), math.log(contexts[index + 1])
            return math.exp(x0 - left * (x1 - x0) / (right - left))
    return None


def _context_crossover_statistics(
    splines: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped = {
        (
            row["method"],
            row["regime"],
            row["load"],
            row["width_panel"],
        ): row
        for row in splines
    }
    result = []
    conditions = {(key[1], key[2], key[3]) for key in grouped}
    for regime, load, panel in conditions:
        for baseline in ("static", "tts"):
            candidate = grouped.get(("lightcone", regime, load, panel))
            reference = grouped.get((baseline, regime, load, panel))
            if candidate is None or reference is None:
                continue
            contexts = list(candidate["contexts"])
            if contexts != reference["contexts"]:
                continue
            differences = np.log(candidate["fitted_goodput"]) - np.log(
                reference["fitted_goodput"]
            )
            point = _first_crossover(contexts, differences)
            candidate_draws = candidate["_bootstrap_log_fitted"]
            reference_draws = reference["_bootstrap_log_fitted"]
            roots = []
            if len(candidate_draws) == len(reference_draws):
                roots = [
                    root
                    for left, right in zip(candidate_draws, reference_draws, strict=True)
                    if (
                        root := _first_crossover(
                            contexts, np.asarray(left) - np.asarray(right)
                        )
                    )
                    is not None
                ]
            interval = np.quantile(roots, (0.025, 0.975)) if len(roots) >= 100 else None
            result.append(
                {
                    "candidate": "lightcone",
                    "baseline": baseline,
                    "regime": regime,
                    "load": load,
                    "width_panel": panel,
                    "first_crossover_context": point,
                    "ci95_low": None if interval is None else float(interval[0]),
                    "ci95_high": None if interval is None else float(interval[1]),
                    "outcome": "no_crossover_through_40928" if point is None else "crossover",
                    "request_count": int(candidate["request_count"])
                    + int(reference["request_count"]),
                    "pairing_key": json.dumps(
                        {
                            "regime": regime,
                            "load": load,
                            "width_panel": panel,
                            "candidate": "lightcone",
                            "baseline": baseline,
                        },
                        sort_keys=True,
                    ),
                    "source_attempts": [
                        *candidate["source_attempts"],
                        *reference["source_attempts"],
                    ],
                    "reducer": "natural_spline_crossover_bootstrap",
                }
            )
    return result


def _e5_tail_statistics(state: StateStore, node: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[int, list[float]]] = {}
    offered_counts: Counter[tuple[str, str, str]] = Counter()
    completed_counts: Counter[tuple[str, str, str]] = Counter()
    sources: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    directories = list(state.completed_attempt_dirs(node))
    if node == "E5-final":
        directories.extend(state.completed_attempt_dirs("E5-p99-extension"))
    for directory in directories:
        config_path = directory / "config.json"
        requests_path = _jsonl_path(directory, "requests.jsonl")
        outcomes_path = _jsonl_path(directory, "request_outcomes.jsonl")
        if not config_path.is_file() or not requests_path.is_file() or not outcomes_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        parameters = config.get("parameters", {})
        load = parameters.get("registered_load", config.get("load"))
        key = (config["method"], config["backend"], str(load))
        outcome_rows = _read_jsonl(outcomes_path)
        offered_counts[key] += len(outcome_rows)
        completed_counts[key] += sum(row.get("status") == "completed" for row in outcome_rows)
        sources.setdefault(key, []).append(
            {
                "job_id": config.get("job_id"),
                "attempt": int(directory.name.removeprefix("attempt-")),
                "attempt_dir": str(directory),
            }
        )
        for request in _read_jsonl(requests_path):
            stamps = request.get("native_token_timestamps_ns", [])
            intervals = request.get("inter_token_ms", [])
            if stamps and intervals:
                time_block = int(stamps[0]) // 10_000_000_000
                grouped.setdefault(key, {}).setdefault(time_block, []).extend(intervals)
    result = []
    for (method, backend, load), blocks in grouped.items():
        offered = offered_counts[(method, backend, load)]
        completed = completed_counts[(method, backend, load)]
        values = [float(np.quantile(rows, 0.99)) for rows in blocks.values() if rows]
        resolved = completed >= E5_P99_MIN_COMPLETED and len(values) >= 3
        estimate = block_bootstrap_interval(values) if resolved else (None, None, None)
        result.append(
            {
                "method": method,
                "backend": backend,
                "load": load,
                "offered_requests": offered,
                "completed_requests": completed,
                "required_completed_requests": E5_P99_MIN_COMPLETED,
                "extension_offered_requests": E5_P99_EXTENSION_REQUESTS,
                "time_block_count": len(values),
                "resolved": resolved,
                "request_count": completed,
                "pairing_key": json.dumps(
                    {"method": method, "backend": backend, "load": load},
                    sort_keys=True,
                ),
                "source_attempts": sources[(method, backend, load)],
                "reducer": "time_block_p99_bootstrap",
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
    winners: dict[str, tuple[tuple[float, int, float], dict[str, Any]]] = {}
    for config, metrics in _metric_rows(state, "E0-tune"):
        method = config["method"]
        if method not in {"onlinespec_ogd", "onlinespec_opt", "onlinespec_ens"}:
            continue
        if metrics.get("feasible") is False or metrics.get("slo_pass") is not True:
            continue
        key = "|".join(
            (config["model"], config["backend"], config["task"], method)
        )
        parameters = {
            name: value
            for name, value in config["parameters"].items()
            if name != "tuning_index"
        }
        score = (
            -float(metrics["goodput"]),
            int(metrics["peak_hbm_bytes"]),
            float(metrics["itl_p99_ms"]),
        )
        if key not in winners or score < winners[key][0]:
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
        if isinstance(block, int) and method in {"static", "tts", "lightcone"}:
            by_block.setdefault(block, {}).setdefault(method, []).append(float(metrics["goodput"]))
    required = []
    for baseline in ("static", "tts"):
        differences = [
            math.log(float(np.mean(rows["lightcone"])))
            - math.log(float(np.mean(rows[baseline])))
            for block, rows in sorted(by_block.items())
            if block in {0, 1, 2, 3} and {"lightcone", baseline} <= rows.keys()
        ]
        if len(differences) != 4:
            return None
        selected = choose_final_blocks(differences)
        if selected is None:
            return None
        required.append(selected)
    return max(required)


def _mechanism_summary(
    state: StateStore, node: str
) -> list[dict[str, object]]:
    fields = (
        "position_conditional_survival",
        "total_variation",
        "target_entropy",
        "target_top_token_draft_cross_entropy",
        "effective_target_token_batch",
        "target_calls",
        "verification_waste",
        "updates",
        "memory_ledger",
        "graph_replay_hit_rate",
        "main_side_overlap_ratio",
        "prefix_cache_hit_rate",
        "tenant_fairness",
        "collective_bytes",
        "collective_time_ms",
        "collective_wait_ms",
        "publication_skew",
        "straggler_time_ms",
        "precision",
        "executed_flops",
        "hbm_bytes_per_committed_token",
        "tokens_per_joule",
        "slo_requests_per_gpu_hour",
    )
    return [
        {
            "job_id": config["job_id"],
            "attempt": metrics.get("source_attempt"),
            "attempt_dir": metrics.get("source_attempt_dir"),
            **{name: metrics.get(name, "N/A") for name in fields},
        }
        for config, metrics in _metric_rows(state, node)
    ]


def _reduce_node(config: ExperimentConfig, state: StateStore, node: str) -> None:
    summary_dir = config.run_dir / "stages" / node
    summarize_attempts(state.completed_attempt_dirs(node), summary_dir)
    if node != "preflight":
        _write_json(
            summary_dir / "mechanism.json", _mechanism_summary(state, node)
        )
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
        splines = _context_splines(state, node)
        _write_json(
            summary_dir / "context_crossovers.json",
            _context_crossover_statistics(splines),
        )
        _write_json(
            summary_dir / "context_splines.json",
            [
                {key: value for key, value in row.items() if not key.startswith("_")}
                for row in splines
            ],
        )
    if node in {"E5-pilot", "E5-final"}:
        _write_json(
            summary_dir / "tail_latency.json", _e5_tail_statistics(state, node)
        )
    if node == "preflight":
        _check_greedy_trajectories(state, node)
    if node == "preflight":
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
        calibrations = {}
        if len(pairs) >= 3:
            for metric in ("goodput", "itl"):
                candidate = [rows["concurrent"][metric] for rows in pairs]
                baseline = [rows["isolated"][metric] for rows in pairs]
                if all(left == right == 0 for left, right in zip(candidate, baseline, strict=True)):
                    calibrations[metric] = (0.0, 0.0, 0.0)
                elif any(value <= 0 for value in baseline):
                    calibrations[metric] = (2.0, 2.0, 2.0)
                else:
                    calibrations[metric] = paired_relative_bca_interval(
                        candidate, baseline
                    )
        if len(_gpu_pairs(config)) == 1:
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
                for item, metrics in anchors
                if item["method"] == "target_only" and item["load"] == load
                and _safe_screen_row(metrics)
            }
            static_regimes = {
                item["parameters"].get("regime")
                for item, metrics in anchors
                if item["method"] == "static"
                and item["load"] == load
                and item.get("width") == 16
                and _safe_screen_row(metrics)
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
                        and _safe_screen_row(metrics)
                    ]
                )
            )
            for width in (4, 8, 16)
            if len(
                {
                    item["parameters"].get("regime")
                    for item, metrics in anchors
                    if item["method"] == "static"
                    and item["load"] == common_load
                    and item.get("width") == width
                    and _safe_screen_row(metrics)
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
            {"static": selected_width},
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
        candidate_count = sum(
            item["method"] == "lightcone_candidate"
            for item, _ in _metric_rows(state, node)
        )
        keep = (
            max(math.ceil(candidate_count / 4), 21)
            if round_index < 3
            else 1
        )
        winners = _rank_e2_candidates(state, node, keep)
        state.set_selection(
            f"{node}_counts",
            {
                "entered": candidate_count,
                "rejected": candidate_count - len(winners),
                "retained": len(winners),
            },
        )
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


def _save_environment(config: ExperimentConfig) -> None:
    path = config.run_dir / "environment.json"
    if path.exists():
        return
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(config.sglang_root / "python"), str(Path(__file__).parents[1]))
    )
    runtime = subprocess.run(
        [
            str(config.server.python),
            "-c",
            "import json,platform,torch,sglang; print(json.dumps({"
            "'python':platform.python_version(),'torch':torch.__version__,"
            "'cuda':torch.version.cuda,'sglang':getattr(sglang,'__version__','unknown')}))",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    versions = json.loads(runtime.stdout) if runtime.returncode == 0 else {
        "runtime_error": runtime.stderr.strip()
    }
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    profilers = {}
    for name, executable in config.profiler_tools.items():
        result = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        profilers[name] = (result.stdout or result.stderr).splitlines()[:1]
    _write_json(
        path,
        {
            "runtime": versions,
            "gpus": gpu.stdout.splitlines() if gpu.returncode == 0 else [],
            "profilers": profilers,
        },
    )


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
        _save_environment(self.config)
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
                    try:
                        _run_node_jobs(
                            self.config, self.state, node, self.stop_event
                        )
                    except ScientificFailure as error:
                        self.state.skip_pending(node, str(error))
                        self.state.mark_stage_failed(node)
                        self.state.set_selection(f"{node}_failed", str(error))
                        continue
                    if self.stop_event.is_set():
                        break
                    if self.state.status_counts(node).get("failed"):
                        self.state.skip_pending(node, "stage has a terminal failed job")
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
                try:
                    _run_node_jobs(self.config, self.state, node, self.stop_event)
                except ScientificFailure as error:
                    self.state.skip_pending(node, str(error))
                    self.state.mark_stage_failed(node)
                    self.state.set_selection(f"{node}_failed", str(error))
                    continue
                if self.stop_event.is_set():
                    break
                if self.state.status_counts(node).get("failed"):
                    self.state.skip_pending(node, "stage has a terminal failed job")
                if self.state.finish_stage(node) != "completed":
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
