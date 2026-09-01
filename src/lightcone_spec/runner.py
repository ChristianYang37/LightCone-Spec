"""Stage-by-stage execution of the complete paper protocol."""

from __future__ import annotations

import gzip
import itertools
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
    committed_goodput,
    holm_decisions,
    normalize_attempt_semantics,
    paired_bca_interval,
    paired_block_statistics,
    paired_relative_bca_interval,
    per_user_generation_speed,
    summarize_attempts,
    validate_scientific_metrics,
)
from .protocol import (
    CONFIDENCE_WEIGHTS,
    E0_ONLINESPEC_METHODS,
    E0_ONLINESPEC_RECIPES,
    E1_REFERENCE_LOAD,
    E5_DRAIN_SECONDS,
    E5_HEADLINE_SECONDS,
    E5_REQUEST_DEADLINE_SECONDS,
    E5_WARMUP_SECONDS,
    FORMAL_ADAPTATION_STRIDE,
    GEOMETRY_GENERATION_TOKENS,
    MAX_E2_GEOMETRIES,
    PAPER_NODES,
    PRIMARY_BLOCKS,
    Job,
    e2_candidates,
    materialize,
    uses_formal_adaptation_stride,
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


class RunnerInterrupted(RuntimeError):
    """A user or runner cancellation that must resume from pending."""


CANDIDATE_METHODS = {
    "lightcone_candidate",
    "onlinespec_candidate",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
}


def _resume_materialization(
    state: StateStore, node: str, planned: tuple[Job, ...]
) -> tuple[Job, ...]:
    """Keep a completed stage's immutable rows when selections later change.

    Selection audits can add narrowly scoped dependency-repair jobs without
    authorizing a completed paper stage to be rematerialized.  Reducers still
    need to run on resume, so return the stored rows instead of skipping the
    stage entirely.  Pending or reopened stages continue to use the current
    protocol materialization.
    """
    existing = state.jobs(node)
    if existing and state.stage_status(node) == "completed":
        return existing
    return planned


def _upgrade_legacy_e0_materialization(
    state: StateStore, planned: tuple[Job, ...]
) -> tuple[Job, ...] | None:
    """Supersede an already-materialized 236-row grid without deleting evidence."""

    existing = state.jobs("E0-tune")
    if len(existing) <= 12:
        return None
    if tuple(job.to_dict() for job in existing[:12]) != tuple(
        job.to_dict() for job in planned[:12]
    ):
        raise RuntimeError("legacy E0 compatibility probes do not match the frozen protocol")
    current_ids = {job.job_id for job in planned}
    obsolete = tuple(job for job in existing[12:] if job.job_id not in current_ids)
    state.supersede_jobs(
        tuple(job.job_id for job in obsolete),
        "superseded by frozen OnlineSPEC source-transfer protocol",
    )
    exclusions = set(state.selection("formal_evidence_exclusions", []))
    exclusions.update(
        job.job_id
        for job in obsolete
        if state.completed_attempt_dir(job.job_id) is not None
    )
    state.set_selection("formal_evidence_exclusions", sorted(exclusions))
    state.add_internal_jobs(planned[12:], storage_node="E0-tune")
    state.set_selection("formal_e0_source_transfer_upgrade_version", 1)
    return state.jobs("E0-tune")


def _records_scientific_rejection(job: Job) -> bool:
    """Return whether a measured rejection is a terminal scientific outcome.

    Reconciliation jobs replace already-completed evidence.  A scientifically
    rejected replacement must remain auditable, but it must not abort the
    sibling replacement queue or leave the other GPU idle.
    """
    return (
        job.node == "S10-reconciliation"
        or job.node == "bugfix-reconciliation-v1"
        or job.node == "E3-width-calibration"
        or job.method in CANDIDATE_METHODS
        or bool(job.parameters.get("interface_fit"))
        or _screening_job(job)
    )


def _screening_job(job: Job) -> bool:
    node = str(job.parameters.get("source_node", job.node))
    if node.endswith("-segments"):
        node = node[: -len("-segments")]
    return node in {
        "E3a",
        "TTS-Cal",
        "E1-common-load",
        "E6-interface",
        "E6-common-load",
        "TTS-batched-calibration",
    }


def _scientific_rejection(
    metrics: dict[str, Any] | None, offered: int, error: Exception
) -> dict[str, Any]:
    rejected = dict(metrics or {})
    rejected.update(
        {
            "scientific_outcome": "rejected",
            "feasible": False,
            "error": str(error),
        }
    )
    rejected.setdefault(
        "request_outcomes",
        {
            "offered": offered,
            "admitted": 0,
            "completed": 0,
            "error": 0,
            "timed_out": 0,
            "cancelled": 0,
            "unfinished": offered,
        },
    )
    return rejected


def _validate_measured_metrics(metrics: dict[str, Any]) -> None:
    try:
        validate_scientific_metrics(metrics)
    except RuntimeError as error:
        raise ScientificFailure(str(error)) from error


def _all_jobs_completed(counts: dict[str, int]) -> bool:
    return counts.get("completed", 0) > 0 and not any(
        counts.get(status, 0) for status in ("pending", "running", "failed")
    )


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
    return (
        isinstance(error, MemoryError)
        or re.search(
            r"out of memory|\b(?:cuda )?oom\b|adaptation peak .* exceeds pre-KV reserve|"
            r"leave no GPU memory for the KV cache",
            message,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _screening_incomplete_classification(rows: Sequence[dict[str, Any]]) -> str:
    statuses = {str(row.get("status")) for row in rows}
    if "cancelled" in statuses:
        return "interrupted"
    if "error" in statuses:
        return "runtime_failure"
    if statuses <= {"timed_out", "unfinished"}:
        return "scientific_infeasible"
    return "runtime_failure"


def _incomplete_scientific_outcome(
    job: Job, rows: Sequence[dict[str, Any]]
) -> str | None:
    """Map pure registered-load timeouts to an auditable scientific outcome."""

    if _screening_incomplete_classification(rows) != "scientific_infeasible":
        return None
    if _screening_job(job):
        return "infeasible"
    if _records_scientific_rejection(job):
        return "rejected"
    return None


def _schedule_exhausted_updates(
    metrics: dict[str, Any], adaptation: dict[str, Any] | None
) -> int | None:
    if not isinstance(adaptation, dict):
        return None
    optimizer = adaptation.get("optimizer")
    if not isinstance(optimizer, dict):
        return None
    horizon = optimizer.get("schedule_total_published_updates")
    if not isinstance(horizon, int):
        return None
    return max(0, int(metrics.get("updates_published", 0)) - horizon)


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
    rank_states = (
        states
        if isinstance(states, list) and states
        else [server_info.get("internal_state", server_info)]
    )
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


def _request_count(config: ExperimentConfig, state: StateStore, job: Job) -> int:
    if job.node == "TTS-Cal" or job.parameters.get("workload") == "tts_stride10_confirmation":
        return 19
    concurrency = 1
    load = job.load or ""
    if load.startswith("c") and load[1:].isdigit():
        concurrency = int(load[1:])
    elif load.startswith("closed_loop_c"):
        concurrency = int(load.removeprefix("closed_loop_c"))
    return max(config.server.requests_per_cell, concurrency)


def _cell_concurrency(job: Job) -> int:
    load = job.load or ""
    if load.startswith("closed_loop_c"):
        return int(load.removeprefix("closed_loop_c"))
    if load.startswith("c") and load[1:].isdigit():
        return int(load[1:])
    return 1


def _uses_request_scope(job: Job) -> bool:
    return job.method in {"tts", "l0_naive"}


def _dispatcher_concurrency(job: Job) -> int:
    if _uses_request_scope(job):
        return 1
    if job.parameters.get("registered_load") == "burstgpt_shape":
        return 256
    return _cell_concurrency(job)


def _fit_prompt(tokens: tuple[int, ...], filler: tuple[int, ...], length: int) -> tuple[int, ...]:
    if length < 1:
        raise ScientificFailure("a context cell has no room for a prompt")
    if len(tokens) >= length:
        return tokens[-length:]
    needed = length - len(tokens)
    if not filler:
        raise ScientificFailure("a long-context cell has an empty workload prompt pool")
    repeated = filler * math.ceil(needed / len(filler))
    return repeated[:needed] + tokens


def _cell_inputs(
    config: ExperimentConfig,
    state: StateStore,
    client,
    job: Job,
) -> tuple[tuple[str | tuple[int, ...], ...], int, dict[str, object]]:
    count = _request_count(config, state, job)
    dataset_key = _task_for_data(config, job)
    metadata: dict[str, object] = {"dataset": dataset_key}
    tts_calibration_window = (
        job.node == "TTS-Cal"
        or job.parameters.get("workload") == "tts_stride10_confirmation"
    )
    if tts_calibration_window:
        calibration = load_calibration_mix(config.dataset_path(dataset_key))
        start = int(job.block or 0) * 19
        prompts = calibration[start : start + 19]
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
    if tts_calibration_window:
        pool_prompts = prompts
    else:
        pool_prompts = tuple(
            str(row["prompt"]) for row in load_prompt_pool(config.dataset_path(dataset_key))
        )
    filler = tuple(token for prompt in pool_prompts for token in client.tokenize(prompt + "\n"))
    regime = str(job.parameters.get("regime", "long_input_short_output"))
    if regime == "short_input_long_generation":
        inputs = tuple(tokens[-min(len(tokens), 128) :] for tokens in tokenized)
        available = job.context - max(len(tokens) for tokens in inputs)
        requested = int(job.parameters.get("generation_tokens", available))
        max_new_tokens = max(1, min(requested, available))
    else:
        max_new_tokens = min(256, config.server.max_new_tokens)
        prompt_length = max(1, job.context - max_new_tokens)
        if not filler:
            raise ScientificFailure("a long-context cell has an empty workload prompt pool")
        metadata["context_construction"] = "repeated_workload_prefix"
        metadata["prefix_repetitions"] = math.ceil(prompt_length / len(filler))
        if regime == "multi_turn_shared_prefix":
            shared_length = prompt_length // 2
            shared = (filler * ((shared_length + len(filler) - 1) // len(filler)))[:shared_length]
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


def _trajectory_checkpoint_metrics(
    results: Iterable[GenerationResult], checkpoints: Iterable[int]
) -> list[dict[str, float | int]]:
    rows = tuple(results)
    output = []
    for checkpoint in checkpoints:
        timestamps = [
            result.native_token_timestamps_ns[:checkpoint]
            for result in rows
            if len(result.native_token_timestamps_ns) >= checkpoint
        ]
        if len(timestamps) != len(rows) or not timestamps:
            continue
        start = min(row[0] for row in timestamps)
        end = max(row[-1] for row in timestamps)
        duration = (end - start) / 1e9
        if duration <= 0:
            continue
        intervals = [
            (right - left) / 1e6
            for row in timestamps
            for left, right in zip(row, row[1:], strict=False)
        ]
        output.append(
            {
                "generation_tokens": checkpoint,
                "committed_tokens": checkpoint * len(rows),
                "decode_seconds": duration,
                "goodput": checkpoint * len(rows) / duration,
                "itl_p99_ms": float(np.quantile(intervals, 0.99)),
                "request_count": len(rows),
            }
        )
    return output


def _jsonl_path(directory: Path, name: str) -> Path:
    compressed = directory / f"{name}.gz"
    return compressed if compressed.is_file() else directory / name


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _e5_reference(state: StateStore, job: Job) -> tuple[float, int]:
    by_method: dict[str, list[tuple[float, int]]] = {}
    rows = _metric_rows(state, job.node)
    rows.extend(_metric_rows(state, f"{job.node}-segments"))
    for config, metrics in rows:
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
        raise ScientificFailure(f"{job.node} lacks SLO-feasible Target-only/Static load anchors")
    per_method = [max(rows) for rows in by_method.values()]
    return max(per_method, key=lambda row: row[0])


def _runtime_job(config: ExperimentConfig, state: StateStore, job: Job) -> Job:
    job = replace(
        job,
        parameters={
            **job.parameters,
            "registered_request_count": _request_count(config, state, job),
        },
    )
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
                    raise ScientificFailure(f"deployment width is not frozen for {job.method}")
                width = int(widths[job.method])
            else:
                width = int(capacity.get("width", 16))
        job = replace(job, width=width)
    if job.load not in {"common_load", "common_slo_load"}:
        if (
            job.node.startswith("E5")
            and isinstance(job.load, str)
            and (
                job.load == "burstgpt_shape"
            )
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
    if not isinstance(capacity, dict) or not isinstance(common, str):
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
    if not original_job.node.startswith("E5") or not isinstance(registered, str):
        return None
    if registered == "burstgpt_shape":
        trace = config.datasets.get("BurstGPT")
        if trace is None:
            raise ScientificFailure("BurstGPT workload-shape row lacks its local trace")
        return load_arrival_offsets(
            trace,
            limit=count,
            offset=_prompt_offset(original_job, count),
        )
    return None


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
        ttft_limit = (
            2000 if result.input_tokens < 4096 else 5000 if result.input_tokens < 16384 else 10000
        )
        itl_p99 = float(np.quantile(result.inter_token_ms or (0.0,), 0.99))
        passed += result.ttft_ms <= ttft_limit and itl_p99 <= 100
    offered = len(outcome_rows)
    completed = sum(row["status"] == "completed" for row in outcome_rows)
    errors = sum(
        row["status"] in {"error", "timed_out", "cancelled", "unfinished"} for row in outcome_rows
    )
    pass_rate = passed / offered if offered else 0.0
    completion_rate = completed / offered if offered else 0.0
    error_rate = errors / offered if offered else 1.0
    return pass_rate, (pass_rate >= 0.99 and error_rate <= 0.001 and completion_rate >= 0.999)


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
    if request_scoped:
        started = time.perf_counter()
        results = []
        for index, prompt in enumerate(prompts):
            history = tuple(prompt)
            rows: list[GenerationResult] = []
            remaining = max_new_tokens
            try:
                for turn in range(4):
                    turns_left = 4 - turn
                    budget = max(1, remaining // turns_left)
                    turn_rows, _ = client.run_batch(
                        (history,),
                        max_new_tokens=budget,
                        seed=seed + turn,
                        request_ids=(
                            f"multi-turn-{index:05d}::turn-{turn}::of-4",
                        ),
                    )
                    result = turn_rows[0]
                    rows.append(result)
                    history = (*history, *result.output_ids)
                    remaining -= budget
            except Exception:
                # A final-turn terminal hook normally restores request-scoped
                # source weights. An interrupted conversation needs an
                # explicit reset so it cannot leak state into the next cell.
                client.reset()
                raise
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
                        value for row in rows for value in row.inter_token_ms
                    ),
                    elapsed_seconds=sum(row.elapsed_seconds for row in rows),
                    stop_reason=rows[-1].stop_reason,
                    output_ids=tuple(value for row in rows for value in row.output_ids),
                    output_text="".join(row.output_text for row in rows),
                    native_token_timestamps_ns=timestamps,
                )
            )
        return tuple(results), time.perf_counter() - started

    histories = [tuple(prompt) for prompt in prompts]
    turns_by_request: list[list[GenerationResult]] = [[] for _ in prompts]
    elapsed = 0.0
    remaining = max_new_tokens
    for turn in range(4):
        turns_left = 4 - turn
        budget = max(1, remaining // turns_left)
        scheduled = client.run_bounded(
            histories,
            max_new_tokens=budget,
            seed=seed + turn,
            request_ids=tuple(
                f"multi-turn-{turn}-{index:05d}" for index in range(len(histories))
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
        timestamps = tuple(value for row in rows for value in row.native_token_timestamps_ns)
        results.append(
            GenerationResult(
                request_id=f"multi-turn-{index:05d}",
                input_tokens=rows[0].input_tokens,
                completion_tokens=sum(row.completion_tokens for row in rows),
                ttft_ms=rows[0].ttft_ms,
                inter_token_ms=tuple(value for row in rows for value in row.inter_token_ms),
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
            (output_dir / "server.pid").write_text(f"{server.process.pid}\n", encoding="utf-8")
        offered = 0
        metrics: dict[str, Any] | None = None
        try:
            runtime_job = _runtime_job(config, state, job)
            declared_concurrency = _cell_concurrency(runtime_job)
            dispatcher_concurrency = _dispatcher_concurrency(runtime_job)
            runtime_job = replace(
                runtime_job,
                parameters={
                    **runtime_job.parameters,
                    "declared_concurrency": declared_concurrency,
                    "dispatcher_concurrency": dispatcher_concurrency,
                    "effective_load": f"c{dispatcher_concurrency}",
                    "metric_semantics": "per_request_native_v2",
                },
            )
            raw_config = runtime_job.to_dict()
            raw_config["parameters"]["stimulus_id"] = _stimulus_id(runtime_job)
            raw_config["adaptation"] = adaptation_payload(runtime_job, selection)
            _write_json(output_dir / "config.json", raw_config)
            bootstrap_job = _exactness_bootstrap(runtime_job)
            client = server.configure(bootstrap_job, selection)
            prompts, max_new_tokens, workload = _cell_inputs(config, state, client, job)
            offered = len(prompts)
            exactness_rows: list[dict[str, object]] = []
            exactness_evidence: dict[str, object] | None = None
            if bootstrap_job is not runtime_job:
                pair_seed = config.protocol.seed + (job.block or 0)
                bootstrap_topology = str(bootstrap_job.parameters.get("topology", "tp1_dp1"))
                bootstrap_before = _speed_metrics(client.server_info(), bootstrap_topology)
                verified, _ = client.run_batch(
                    prompts[:4],
                    max_new_tokens=max_new_tokens,
                    seed=pair_seed,
                    request_id_prefix="controlled-speculative-verify",
                )
                bootstrap_after = _speed_metrics(client.server_info(), bootstrap_topology)
                bootstrap_committed = int(bootstrap_after["committed_tokens"]) - int(
                    bootstrap_before["committed_tokens"]
                )
                _validate_committed_tokens(verified, bootstrap_committed)
                bootstrap_safety = {
                    counter: int(bootstrap_after[counter]) - int(bootstrap_before[counter])
                    for counter in SAFETY_COUNTERS
                }
                if any(bootstrap_safety.values()):
                    raise ScientificFailure(
                        f"deterministic verification raised safety counters: {bootstrap_safety}"
                    )
                checked_tokens = int(bootstrap_after.get("greedy_token_checks", 0)) - int(
                    bootstrap_before.get("greedy_token_checks", 0)
                )
                mismatched_tokens = int(bootstrap_after.get("greedy_token_mismatches", 0)) - int(
                    bootstrap_before.get("greedy_token_mismatches", 0)
                )
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
                    max_new_tokens[0] if isinstance(max_new_tokens, tuple) else max_new_tokens
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
                    max_new_tokens[0] if isinstance(max_new_tokens, tuple) else max_new_tokens
                )
                warmup_deadline = (
                    time.monotonic() + E5_WARMUP_SECONDS if job.node.startswith("E5") else None
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
                    max(max_new_tokens) if isinstance(max_new_tokens, tuple) else max_new_tokens,
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
            request_scoped = _uses_request_scope(runtime_job)
            temperature = float(runtime_job.parameters.get("temperature", 0.0))
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
                    {"policy": "l0_naive", **result.to_dict()} for result in l0_results
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
                    max_in_flight=dispatcher_concurrency,
                )
            elif arrivals is None:
                seed = config.protocol.seed + (job.block or 0)
                if runtime_job.load and runtime_job.load.startswith("closed_loop_c"):
                    scheduled = client.run_closed_loop(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                        temperature=temperature,
                        routing_keys=_routing_keys(config, runtime_job, len(prompts)),
                        max_in_flight=dispatcher_concurrency,
                        duration_seconds=E5_HEADLINE_SECONDS,
                        deadline_seconds=E5_REQUEST_DEADLINE_SECONDS,
                        request_id_prefix=f"{job.job_id}-closed-loop",
                    )
                    results, elapsed = scheduled.results, scheduled.elapsed_seconds
                    if any(outcome.status == "error" for outcome in scheduled.outcomes):
                        raise RuntimeError("closed-loop request failed")
                elif request_scoped:
                    results, elapsed = _run_request_scoped(
                        client,
                        prompts,
                        max_new_tokens,
                        seed,
                        same_seed=bool(runtime_job.parameters.get("controlled_replay")),
                        request_prefix=f"{job.job_id}-measure",
                        temperature=temperature,
                    )
                else:
                    scheduled = client.run_bounded(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                        temperature=temperature,
                        routing_keys=_routing_keys(config, runtime_job, len(prompts)),
                        request_ids=tuple(
                            f"{job.job_id}-measure-{index:05d}" for index in range(len(prompts))
                        ),
                        max_in_flight=dispatcher_concurrency,
                        deadline_seconds=E5_REQUEST_DEADLINE_SECONDS,
                    )
                    results, elapsed = scheduled.results, scheduled.elapsed_seconds
            else:
                scheduled = client.run_scheduled(
                    prompts,
                    arrivals,
                    max_new_tokens=max_new_tokens,
                    seed=config.protocol.seed + (job.block or 0),
                    temperature=temperature,
                    routing_keys=_routing_keys(config, runtime_job, len(prompts)),
                    request_ids=tuple(
                        f"{job.job_id}-scheduled-{index:05d}" for index in range(len(prompts))
                    ),
                    max_in_flight=dispatcher_concurrency,
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
                    {"policy": "target_only", **result.to_dict()} for result in controlled
                ]
            request_rows = [_request_metrics(result) for result in results]
            measured_user_speed = per_user_generation_speed(request_rows)
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
            incomplete = [row for row in outcome_rows if row["status"] != "completed"]
            if incomplete:
                classification = _screening_incomplete_classification(incomplete)
                if classification == "interrupted":
                    raise RunnerInterrupted("screening request was cancelled")
                if classification == "runtime_failure":
                    raise RuntimeError(
                        "measured request failed at runtime: "
                        + "; ".join(
                            str(row.get("error") or row["status"])
                            for row in incomplete
                        )
                    )
                scientific_outcome = _incomplete_scientific_outcome(job, incomplete)
                if scientific_outcome is None:
                    raise RuntimeError(
                        f"{len(incomplete)} requests did not complete in a measured cell"
                    )
                counters = {
                    name: int(after[name]) - int(before[name]) for name in SAFETY_COUNTERS
                }
                _write_json(
                    output_dir / "metrics.json",
                    {
                        "scientific_outcome": scientific_outcome,
                        "feasible": False,
                        "slo_pass": False,
                        "error": f"{len(incomplete)} requests did not complete at registered load",
                        "duration_seconds": elapsed,
                        "committed_tokens": committed,
                        "output_tokens": sum(result.completion_tokens for result in results),
                        "request_count": len(results),
                        "request_outcomes": {
                            "offered": len(outcome_rows),
                            "admitted": sum(
                                row["admitted_ns"] is not None for row in outcome_rows
                            ),
                            "completed": sum(
                                row["status"] == "completed" for row in outcome_rows
                            ),
                            "error": sum(row["status"] == "error" for row in outcome_rows),
                            "timed_out": sum(
                                row["status"] == "timed_out" for row in outcome_rows
                            ),
                            "cancelled": sum(
                                row["status"] == "cancelled" for row in outcome_rows
                            ),
                            "unfinished": sum(
                                row["status"] == "unfinished" for row in outcome_rows
                            ),
                        },
                        **counters,
                    },
                )
                client.reset()
                state.complete(job.job_id, attempt)
                return
            output_tokens = _validate_committed_tokens(results, committed)
            peak_hbm = int(after["peak_hbm_bytes"])
            reserved_hbm = int(after["peak_hbm_reserved_bytes"])
            nvml_hbm = _peak_hbm_from_csv(server.output_dir / "gpu.csv", session_offsets["gpu.csv"])
            energy_mj = _energy_from_csv(server.output_dir / "gpu.csv", session_offsets["gpu.csv"])
            kv_capacity = after.get("kv_token_capacity")
            native_intervals = [value for result in results for value in result.inter_token_ms]
            native_itl = float(np.quantile(native_intervals, 0.99)) if native_intervals else 0.0
            if peak_hbm <= 0 or not isinstance(kv_capacity, (int, float)) or kv_capacity <= 0:
                raise ScientificFailure(
                    "patched runtime did not report positive HBM and KV capacity"
                )
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
                "per_user_generation_speed": (
                    measured_user_speed if measured_user_speed is not None else "N/A"
                ),
                "declared_concurrency": declared_concurrency,
                "dispatcher_concurrency": dispatcher_concurrency,
                "effective_load": f"c{dispatcher_concurrency}",
                "metric_semantics": "per_request_native_v2",
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
            checkpoints = runtime_job.parameters.get("generation_checkpoints")
            if isinstance(checkpoints, (list, tuple)):
                metrics["trajectory_checkpoints"] = _trajectory_checkpoint_metrics(
                    results, (int(value) for value in checkpoints)
                )
            if exactness_evidence is not None:
                metrics["exactness_bootstrap"] = exactness_evidence
            if runtime_job.parameters.get("probe"):
                metrics["compatible"] = True
                metrics["static_interface_passed"] = static_interface_passed
                metrics["adaptive_interface_passed"] = True
            if str(job.parameters.get("source_node", job.node)) == "E1a":
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
                "confidence_probabilities",
                "confidence_outcomes",
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
                            "confidence_probabilities",
                            "confidence_outcomes",
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
            adaptation = raw_config.get("adaptation")
            metrics["resolved_stride"] = (
                adaptation.get("stride") if isinstance(adaptation, dict) else None
            )
            if (
                uses_formal_adaptation_stride(runtime_job)
                and metrics["resolved_stride"] != FORMAL_ADAPTATION_STRIDE
            ):
                raise ScientificFailure("formal adaptive cell did not resolve to stride S=10")
            exhausted = _schedule_exhausted_updates(metrics, adaptation)
            if exhausted is not None:
                metrics["schedule_horizon_basis"] = "registered_max_output_tokens"
                metrics["schedule_exhausted_updates"] = exhausted
                if exhausted:
                    raise ScientificFailure("registered cosine schedule horizon was exceeded")
            if (
                uses_formal_adaptation_stride(runtime_job)
                and runtime_job.parameters.get("regime") == "multi_turn_shared_prefix"
                and int(metrics.get("target_calls", 0)) < FORMAL_ADAPTATION_STRIDE
            ):
                raise ScientificFailure(
                    "formal multi-turn segment produced fewer than ten speculation rounds"
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
                in {
                    "tts",
                    "tts_lora_batched",
                    "l0_naive",
                    "lightcone",
                    "lightcone_candidate",
                }
                and metrics.get("updates_published", 0)
                < int(job.parameters.get("minimum_updates", 1))
            ):
                raise ScientificFailure("adaptive cell did not publish the required updates")
            if job.parameters.get("workload") == "confidence_calibration":
                probabilities = metrics.get("confidence_probabilities")
                outcomes = metrics.get("confidence_outcomes")
                if (
                    not isinstance(probabilities, list)
                    or not probabilities
                    or not isinstance(outcomes, list)
                    or len(probabilities) != len(outcomes)
                ):
                    raise ScientificFailure(
                        "confidence calibration produced incomplete probability/outcome telemetry"
                    )
            if runtime_job.parameters.get("controlled_replay"):
                if after.get("controlled_candidate_compared") is not True:
                    raise ScientificFailure("controlled replay did not compare a candidate")
                if after.get("controlled_candidate_equal") is not True:
                    raise ScientificFailure("controlled replay changed the staged candidate")
                metrics["controlled_candidate_compared"] = True
                metrics["controlled_candidate_equal"] = True
            if job.parameters.get("workload") != "failure_injection":
                _validate_measured_metrics(metrics)
            elif not (
                metrics.get("recovery_health_passed") and metrics.get("expected_action_passed")
            ):
                raise ScientificFailure("fault diagnostic did not complete its expected action")
            _write_json(output_dir / "metrics.json", metrics)
            state.complete(job.job_id, attempt)
            return
        except RunnerInterrupted as error:
            _write_json(
                output_dir / "metrics.json",
                {
                    "status": "interrupted",
                    "error": str(error),
                    "request_outcomes": {
                        "offered": offered,
                        "completed": 0,
                        "error": 0,
                        "cancelled": int(offered > 0),
                        "unfinished": max(0, offered - 1),
                    },
                },
            )
            state.interrupt(job.job_id, attempt, str(error))
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
            if _records_scientific_rejection(job):
                _write_json(
                    output_dir / "metrics.json",
                    _scientific_rejection(metrics, offered, error),
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
                and state.failed_attempts(job.job_id) < config.protocol.max_process_retries
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
        return _formalize_recipe(dict(state.selection("tts_recipe", {})))
    if job.method == "tts_lora_batched":
        return _formalize_recipe(dict(state.selection("tts_batched_geometry", {})))
    if job.method == "lightcone":
        name = "dspark_recipe" if job.backend == "DSPARK" else "lightcone_recipe"
        selected = dict(state.selection(name, {}))
        if job.node.startswith("E6"):
            selected.update(parameterization="lora", rank=8, scope="all")
        return _formalize_recipe(selected)
    if job.node.startswith("E1a") and job.method == "lightcone_candidate":
        finalists = state.selection("e1a_finalists", [])
        slot = job.parameters.get("finalist_slot")
        selected = (
            dict(finalists[int(slot)])
            if isinstance(slot, int) and slot < len(finalists)
            else dict(state.selection("lightcone_recipe", {}))
        )
        weight = state.selection("dspark_confidence_weight", None)
        if isinstance(weight, (int, float)):
            selected["confidence_loss_weight"] = float(weight)
        temperature = state.selection("dspark_confidence_temperature", None)
        if isinstance(temperature, (int, float)):
            selected["confidence_temperature"] = float(temperature)
        return _formalize_recipe(selected)
    if job.method in E0_ONLINESPEC_METHODS:
        if job.parameters.get("recipe_validation"):
            return None
        recipes = state.selection("e0_recipes", {})
        key = "|".join((job.model, job.backend, job.method))
        selected = recipes.get(key) if isinstance(recipes, dict) else None
        if not isinstance(selected, dict):
            raise ScientificFailure(f"formal E0 job lacks validated recipe {key}")
        return _formalize_recipe(dict(selected))
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
            "screening request failed at runtime",
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
    if job.parameters.get("workload") == "tts_calibration_screen":
        return config.gpu_ids[job.ordinal % len(config.gpu_ids)]
    if job.block is not None:
        return config.gpu_ids[job.block % len(config.gpu_ids)]
    if job.parameters.get("parent_job_id"):
        return config.gpu_ids[(job.ordinal // 1000) % len(config.gpu_ids)]
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


def _single_gpu_queues(
    config: ExperimentConfig, jobs: tuple[Job, ...]
) -> dict[int, tuple[Job, ...]]:
    if jobs and all(
        job.parameters.get("workload") == "tts_calibration_screen" for job in jobs
    ):
        queues: dict[int, list[Job]] = {gpu: [] for gpu in config.gpu_ids}
        work = {gpu: 0.0 for gpu in config.gpu_ids}
        ordered = sorted(
            jobs,
            key=lambda job: (
                -float(job.parameters["generation_tokens"])
                / float(job.parameters["stride"]),
                job.ordinal,
            ),
        )
        for job in ordered:
            gpu = min(config.gpu_ids, key=lambda value: (work[value], value))
            queues[gpu].append(job)
            work[gpu] += float(job.parameters["generation_tokens"]) / float(
                job.parameters["stride"]
            )
        return {gpu: tuple(rows) for gpu, rows in queues.items() if rows}
    groups: dict[tuple[str, object], list[Job]] = {}
    for job in jobs:
        parent = job.parameters.get("parent_job_id")
        if parent is not None:
            key = ("parent", parent)
        elif job.block is not None:
            key = ("block", job.block)
        else:
            key = ("job", job.job_id)
        groups.setdefault(key, []).append(job)

    queues = {gpu: [] for gpu in config.gpu_ids}
    work = {gpu: 0.0 for gpu in config.gpu_ids}
    ordered = sorted(
        groups.values(),
        key=lambda rows: (
            -sum(float(row.parameters.get("generation_tokens", 256)) for row in rows),
            min(row.ordinal for row in rows),
        ),
    )
    for rows in ordered:
        gpu = min(config.gpu_ids, key=lambda value: (work[value], value))
        queues[gpu].extend(rows)
        work[gpu] += sum(
            float(row.parameters.get("generation_tokens", 256)) for row in rows
        )
    return {gpu: tuple(rows) for gpu, rows in queues.items() if rows}


def _session_pool_eligible(job: Job) -> bool:
    """Return whether a bundled cell is independent and safe to steal."""

    if (
        job.gpu_count != 1
        or job.block is not None
        or job.parameters.get("parent_job_id") is None
        or job.parameters.get("topology", "tp1_dp1") != "tp1_dp1"
    ):
        return False
    if any(
        job.parameters.get(name)
        for name in (
            "profiler",
            "failure",
            "adaptive_probe",
            "controlled_replay",
            "controlled_pair_baseline",
        )
    ):
        return False
    return (
        job.parameters.get("workload")
        in {"confidence_calibration", "excluded_deployment_width_tuning"}
        or job.parameters.get("reconciliation_kind")
        == "screening_runtime_error_classification"
    )


class _SessionCellPool:
    """Thread-safe work pool with deterministic session-local preference."""

    def __init__(
        self,
        entries: Iterable[tuple[Job, tuple[object, ...], float]],
    ):
        self._pending = list(entries)
        self._lock = threading.Lock()

    def claim(self, preferred_key: tuple[object, ...] | None = None) -> Job | None:
        with self._lock:
            if not self._pending:
                return None
            candidates = [
                index
                for index, (_, key, _) in enumerate(self._pending)
                if preferred_key is not None and key == preferred_key
            ]
            if not candidates:
                candidates = list(range(len(self._pending)))
            index = min(
                candidates,
                key=lambda value: (
                    -self._pending[value][2],
                    self._pending[value][0].ordinal,
                    self._pending[value][0].job_id,
                ),
            )
            job, _, _ = self._pending.pop(index)
            return job

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)


def _session_pool_work(job: Job) -> float:
    return float(job.parameters.get("generation_tokens", 256))


def _run_session_cell_pool(
    config: ExperimentConfig,
    state: StateStore,
    node: str,
    stop_event: threading.Event,
    node_failed: threading.Event,
    jobs: tuple[Job, ...],
) -> None:
    entries = []
    for job in jobs:
        runtime_job = _runtime_job(config, state, job)
        selection = _selection_for_job(state, job)
        process_job = _exactness_bootstrap(runtime_job)
        entries.append(
            (
                job,
                server_session_key(process_job, selection),
                _session_pool_work(job),
            )
        )
    pool = _SessionCellPool(entries)

    def worker(gpu: int) -> None:
        first = pool.claim()
        if first is None:
            return
        first_runtime = _runtime_job(config, state, first)
        first_selection = _selection_for_job(state, first)
        first_process_job = _exactness_bootstrap(first_runtime)
        session_dir = config.run_dir / "sessions" / node / f"cell-pool-gpu-{gpu}"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "cycles.jsonl").touch()
        process = ServerProcess(
            config,
            first_process_job,
            gpus=(gpu,),
            port=_resource_port(config, (gpu,)),
            output_dir=session_dir,
            selection=first_selection,
        )
        job = first
        try:
            while not stop_event.is_set() and not node_failed.is_set():
                selection = _selection_for_job(state, job)
                _execute_cell(
                    config,
                    state,
                    job,
                    gpus=(gpu,),
                    selection=selection,
                    server=process,
                )
                if state.status_counts(node).get("failed"):
                    node_failed.set()
                    return
                if stop_event.is_set() or node_failed.is_set():
                    return
                job = pool.claim(process.session_key)
                if job is None:
                    return
        finally:
            process.stop()

    worker_count = min(len(config.gpu_ids), len(jobs))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="lightcone-cell-pool",
    ) as executor:
        futures = [executor.submit(worker, gpu) for gpu in config.gpu_ids[:worker_count]]
        for future in futures:
            future.result()


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
    pairs = [rows for rows in timings.values() if {"isolated", "concurrent"} <= rows.keys()]
    intervals = {}
    if len(pairs) >= 3:
        for metric in ("goodput", "itl"):
            candidate = [rows["concurrent"][metric] for rows in pairs]
            baseline = [rows["isolated"][metric] for rows in pairs]
            intervals[metric] = paired_relative_bca_interval(candidate, baseline)
    return {
        "enabled": len(pairs) == len(timings) >= 3
        and all(abs(point) <= 0.01 and low <= 0 <= high for point, low, high in intervals.values()),
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


def _ncu_permission_block_reason(
    executable: Path, python: Path, gpu: int
) -> str | None:
    """Probe Nsight Compute counter access before launching a formal server."""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    result = subprocess.run(
        [
            str(executable),
            "--target-processes",
            "all",
            "--profile-from-start",
            "on",
            "--metrics",
            "gpu__time_duration.sum",
            str(python),
            "-c",
            (
                "import torch; "
                "x=torch.ones(1,device='cuda'); "
                "(x+1).sum().item(); torch.cuda.synchronize()"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=environment,
    )
    output = f"{result.stdout}\n{result.stderr}"
    if "ERR_NVGPUCTRPERM" in output:
        return "Nsight Compute counters blocked by provider (ERR_NVGPUCTRPERM)"
    if result.returncode != 0:
        tail = " ".join(output.splitlines()[-3:]).strip()
        raise RuntimeError(f"Nsight Compute capability probe failed: {tail}")
    return None


def _complete_blocked_profiler(
    state: StateStore,
    job: Job,
    run_dir: Path,
    gpus: tuple[int, ...],
    reason: str,
) -> None:
    attempt_number = state.next_attempt(job.job_id)
    output_dir = run_dir / "jobs" / job.job_id / f"attempt-{attempt_number:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt = state.start(job, gpus, output_dir)
    _write_json(output_dir / "config.json", job.to_dict())
    _write_json(
        output_dir / "metrics.json",
        {
            "scientific_outcome": "blocked",
            "feasible": False,
            "profiler": job.parameters.get("profiler"),
            "blocked_reason": reason,
            "error": reason,
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


def _e1_load_jobs(state: StateStore) -> tuple[Job, ...]:
    geometries = _rank_e1_geometries(state)
    if not geometries:
        raise ScientificFailure("E1 reference screen has no two-anchor Pareto geometry")
    segments = [{"load": f"c{value}"} for value in (1, 2, 4, 8, 16, 32, 64)]
    rows: list[Job] = []
    for method, backend in (("target_only", "NONE"), ("static", "DFLASH")):
        rows.append(
            Job(
                f"e1-load-{method}",
                "E1-common-load",
                len(rows),
                method,
                "Qwen/Qwen3-8B",
                backend,
                "CalibrationMix",
                context=40928,
                load="c1",
                width=None if method == "target_only" else 16,
                parameters={
                    "regime": "long_input_short_output",
                    "segments": segments,
                    "workload": "excluded_common_load_probe",
                },
            )
        )
    for geometry_index, geometry in enumerate(geometries):
        for optimizer in ("adamw", "sgdm"):
            rows.append(
                Job(
                    f"e1-load-g{geometry_index:02d}-{optimizer}",
                    "E1-common-load",
                    len(rows),
                    "lightcone_candidate",
                    "Qwen/Qwen3-8B",
                    "DFLASH",
                    "CalibrationMix",
                    context=40928,
                    load="c1",
                    width=16,
                    parameters={
                        **geometry,
                        "optimizer": optimizer,
                        "regime": "long_input_short_output",
                        "segments": segments,
                        "workload": "excluded_common_load_probe",
                    },
                )
            )
    return tuple(rows)


def _select_e1_common_load(state: StateStore, expected_per_load: int) -> str:
    counts = Counter(
        str(config["load"])
        for config, metrics in _metric_rows(state, "E1-common-load")
        if metrics.get("slo_pass") is True
        and metrics.get("feasible") is not False
        and all(metrics.get(counter, 0) == 0 for counter in SAFETY_COUNTERS)
    )
    feasible = [load for load, count in counts.items() if count == expected_per_load]
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
    grouped: dict[float, list[tuple[float, float, float, int]]] = {}
    for config, metrics in _metric_rows(state, "E1a"):
        if config.get("parameters", {}).get("workload") != "confidence_calibration":
            continue
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
        weight = float(config["parameters"]["confidence_loss_weight"])
        grouped.setdefault(weight, []).append(
            (
                float(brier),
                float(ece),
                -float(metrics["goodput"]),
                int(metrics["peak_hbm_bytes"]),
            )
        )
    if set(grouped) != set(CONFIDENCE_WEIGHTS) or any(
        len(rows) != 10 for rows in grouped.values()
    ):
        raise ScientificFailure("DSpark confidence calibration is incomplete")
    candidates = [
        (
            float(np.mean([row[0] for row in rows])),
            float(np.mean([row[1] for row in rows])),
            float(np.mean([row[2] for row in rows])),
            max(row[3] for row in rows),
            weight,
        )
        for weight, rows in grouped.items()
    ]
    return min(candidates)[-1]


def _select_confidence_temperature(state: StateStore, weight: float) -> float:
    probabilities: list[float] = []
    outcomes: list[float] = []
    for config, metrics in _metric_rows(state, "E1a"):
        parameters = config.get("parameters", {})
        if (
            parameters.get("workload") != "confidence_calibration"
            or float(parameters.get("confidence_loss_weight", -1)) != weight
        ):
            continue
        probabilities.extend(metrics.get("confidence_probabilities", []))
        outcomes.extend(metrics.get("confidence_outcomes", []))
    if not probabilities or len(probabilities) != len(outcomes):
        raise ScientificFailure("DSpark confidence outcomes are incomplete")
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    target = np.asarray(outcomes, dtype=np.float64)
    logits = np.log(probability / (1.0 - probability))
    candidates = np.linspace(0.25, 4.0, 151)
    losses = []
    for temperature in candidates:
        calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
        losses.append(float(np.mean((calibrated - target) ** 2)))
    return float(candidates[int(np.argmin(losses))])


def _tts_batched_calibration_jobs(state: StateStore) -> tuple[Job, ...]:
    geometries = [
        dict(row)
        for row in state.selection("e1_geometries", [])
        if row.get("parameterization") == "lora"
    ][:4]
    return tuple(
        Job(
            job_id=f"tts-batched-geometry-{index:02d}",
            node="TTS-batched-calibration",
            ordinal=index,
            method="tts_lora_batched",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="CalibrationMix",
            context=40928,
            load="c8",
            width=16,
            parameters={
                **geometry,
                "generation_tokens": 512,
                "regime": "short_input_long_generation",
                "workload": "excluded_tts_batched_geometry",
            },
        )
        for index, geometry in enumerate(geometries)
    )


def _select_tts_batched_geometry(state: StateStore) -> dict[str, Any]:
    candidates = []
    for config, metrics in _metric_rows(state, "TTS-batched-calibration"):
        if metrics.get("feasible") is False:
            continue
        if any(int(metrics.get(counter, 0)) for counter in SAFETY_COUNTERS):
            continue
        candidates.append(
            (
                int(metrics.get("adapter_crosstalk", 0)),
                -float(metrics.get("goodput", 0.0)),
                int(metrics.get("peak_hbm_bytes", 0)),
                config["parameters"],
            )
        )
    if not candidates:
        raise ScientificFailure("per-request LoRA calibration found no feasible geometry")
    return dict(min(candidates, key=lambda row: row[:-1])[-1])


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
    for method in ("static", "tts", "l0_naive", "lightcone"):
        for width in (4, 8, 16):
            segments = [
                {
                    "task": task,
                    "regime": regime,
                    **(
                        {"generation_tokens": GEOMETRY_GENERATION_TOKENS}
                        if regime == "short_input_long_generation"
                        else {}
                    ),
                }
                for regime, task in tasks.items()
            ]
            rows.append(
                Job(
                    f"e3-width-{method}-{width}",
                    "E3-width-calibration",
                    len(rows),
                    method,
                    "Qwen/Qwen3-8B",
                    "DFLASH",
                    "CalibrationMix",
                    context=40928,
                    load=common,
                    width=width,
                    parameters={
                        "segments": segments,
                        "workload": "excluded_deployment_width_tuning",
                    },
                )
            )
    return tuple(rows)


def _select_deployment_widths(state: StateStore) -> dict[str, int]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for config, metrics in _metric_rows(state, "E3-width-calibration"):
        if metrics.get("slo_pass") is True:
            groups.setdefault((config["method"], int(config["width"])), []).append(metrics)
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
    segments = [{"load": f"c{value}"} for value in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
    for model in models:
        for role in roles:
            rows.append(
                Job(
                    f"e6-load-{model.rsplit('/', 1)[-1]}-{role}",
                    "E6-common-load",
                    len(rows),
                    role,
                    model,
                    "NONE" if role == "target_only" else "NEXTN",
                    "LiveCodeBench",
                    context=40928,
                    load="c1",
                    width=None if role == "target_only" else 16,
                    gpu_count=2,
                    parameters={
                        "segments": segments,
                        "workload": "excluded_e6_common_load_probe",
                    },
                )
            )
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
            mode for mode, (_, metrics) in modes.items() if metrics.get("feasible") is not False
        }
        capabilities[parent.model] = supported
        feasible = supported == {"lora", "full"}
        if parent.job_id not in pending:
            continue
        attempt_number = state.next_attempt(parent.job_id)
        output_dir = config.run_dir / "jobs" / parent.job_id / f"attempt-{attempt_number:02d}"
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


def _select_e6_common_loads(state: StateStore, capabilities: dict[str, set[str]]) -> dict[str, str]:
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
            selected[model] = max(loads, key=lambda value: int(value.removeprefix("c")))
    return selected


_JOB_FIELDS = {
    "method",
    "model",
    "backend",
    "task",
    "context",
    "load",
    "width",
    "block",
    "gpu_count",
}


def _segment_jobs(parent: Job) -> tuple[Job, ...]:
    raw_segments = parent.parameters.get("segments")
    if not isinstance(raw_segments, list):
        return ()
    base_parameters = {key: value for key, value in parent.parameters.items() if key != "segments"}
    capacities = []
    for row in raw_segments:
        load = row.get("load", parent.load)
        if isinstance(load, str) and load.startswith("c") and load[1:].isdigit():
            capacities.append(int(load[1:]))
        elif isinstance(load, str) and load.startswith("closed_loop_c"):
            capacities.append(int(load.removeprefix("closed_loop_c")))
    if capacities:
        base_parameters["server_capacity"] = max(capacities)
    children = []
    for index, raw in enumerate(raw_segments):
        segment = dict(raw)
        fields = {key: segment.pop(key) for key in tuple(segment) if key in _JOB_FIELDS}
        children.append(
            replace(
                parent,
                job_id=f"{parent.job_id}__segment-{index:03d}",
                ordinal=parent.ordinal * 1000 + index,
                parameters={
                    **base_parameters,
                    **segment,
                    "parent_job_id": parent.job_id,
                    "segment_index": index,
                },
                **fields,
            )
        )
    return tuple(children)


def _complete_segment_parent(
    config: ExperimentConfig,
    state: StateStore,
    parent: Job,
    children: tuple[Job, ...],
) -> None:
    if parent.job_id not in {job.job_id for job in state.pending_jobs(parent.node)}:
        return
    rows = []
    for child in children:
        directory = state.completed_attempt_dir(child.job_id)
        if directory is None:
            raise ScientificFailure(f"bundle {parent.job_id} has an incomplete segment")
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        child_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "segment_index": child.parameters["segment_index"],
                "config": child_config,
                "metrics": metrics,
                "attempt_dir": str(directory),
            }
        )
    attempt_number = state.next_attempt(parent.job_id)
    output_dir = config.run_dir / "jobs" / parent.job_id / f"attempt-{attempt_number:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt = state.start(parent, _job_gpus(config, parent), output_dir)
    outcomes = Counter()
    for row in rows:
        outcomes.update(row["metrics"].get("request_outcomes", {}))
    committed = sum(int(row["metrics"].get("committed_tokens", 0)) for row in rows)
    duration = sum(float(row["metrics"].get("duration_seconds", 0.0)) for row in rows)
    feasible = all(row["metrics"].get("feasible") is not False for row in rows)
    metrics = {
        "scientific_outcome": "completed" if feasible else "rejected",
        "segments": rows,
        "segment_count": len(rows),
        "committed_tokens": committed,
        "duration_seconds": duration,
        "goodput": committed_goodput(committed, duration) if duration > 0 else 0.0,
        "peak_hbm_bytes": max(int(row["metrics"].get("peak_hbm_bytes", 0)) for row in rows),
        "request_count": sum(int(row["metrics"].get("request_count", 0)) for row in rows),
        "request_outcomes": dict(outcomes),
        "feasible": feasible,
        "slo_pass": all(row["metrics"].get("slo_pass") is not False for row in rows),
    }
    for counter in SAFETY_COUNTERS:
        metrics[counter] = sum(int(row["metrics"].get(counter, 0)) for row in rows)
    _write_json(output_dir / "config.json", parent.to_dict())
    _write_json(output_dir / "metrics.json", metrics)
    state.complete(parent.job_id, attempt)


def _run_pending_jobs(
    config: ExperimentConfig,
    state: StateStore,
    node: str,
    stop_event: threading.Event,
    pending: tuple[Job, ...],
) -> None:
    bundled = tuple(job for job in pending if isinstance(job.parameters.get("segments"), list))
    if bundled:
        storage_node = f"{node}-segments"
        children_by_parent = {job.job_id: _segment_jobs(job) for job in bundled}
        children = tuple(child for rows in children_by_parent.values() for child in rows)
        state.add_internal_jobs(children, storage_node=storage_node)
        wanted = {job.job_id for job in children}
        child_pending = tuple(
            job for job in state.pending_jobs(storage_node) if job.job_id in wanted
        )
        if node.startswith("E5"):
            anchors = tuple(
                job
                for job in child_pending
                if job.method in {"target_only", "static"}
                and isinstance(job.load, str)
                and job.load.startswith("closed_loop_c")
            )
            if anchors:
                _run_pending_jobs(config, state, node, stop_event, anchors)
            child_pending = tuple(
                job for job in state.pending_jobs(storage_node) if job.job_id in wanted
            )
        if child_pending and not stop_event.is_set():
            _run_pending_jobs(config, state, node, stop_event, child_pending)
        if stop_event.is_set():
            return
        for parent in bundled:
            _complete_segment_parent(config, state, parent, children_by_parent[parent.job_id])
        pending = tuple(job for job in pending if job not in bundled)
        if not pending:
            return
    exclusive = tuple(job for job in pending if job.gpu_count == 2)
    singles = tuple(job for job in pending if job.gpu_count == 1)
    node_failed = threading.Event()
    pooled_singles = tuple(job for job in singles if _session_pool_eligible(job))
    singles = tuple(job for job in singles if job not in pooled_singles)

    def run_sessions(jobs: Iterable[Job], *, gpus: tuple[int, ...], port: int, label: str) -> None:
        grouped: dict[tuple[object, ...], list[tuple[Job, Job, dict[str, Any] | None]]] = {}
        for job in jobs:
            runtime_job = _runtime_job(config, state, job)
            selection = _selection_for_job(state, job)
            probe = job.job_id if job.parameters.get("adaptive_probe") else None
            process_job = _exactness_bootstrap(runtime_job)
            key = (job.block, probe, *server_session_key(process_job, selection))
            grouped.setdefault(key, []).append((job, runtime_job, selection))
        keys = []
        blocks = sorted(
            {key[0] for key in grouped}, key=lambda value: -1 if value is None else value
        )
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
            if (
                first_job.node == "E4-profile"
                and first_job.parameters.get("profiler") == "ncu"
            ):
                reason = _ncu_permission_block_reason(
                    config.profiler_tools["ncu"], config.server.python, gpus[0]
                )
                if reason is not None:
                    for job, _, _ in rows:
                        _complete_blocked_profiler(
                            state, job, config.run_dir, gpus, reason
                        )
                    continue
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
                    if first_runtime.parameters.get("topology") == "two_replica_tp1_dp2"
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
                            _complete_infeasible_startup(state, job, config.run_dir, gpus, error)
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
    exclusive_queues = {pair: jobs for pair, jobs in exclusive_queues.items() if jobs}
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
    if pooled_singles:
        _run_session_cell_pool(
            config,
            state,
            node,
            stop_event,
            node_failed,
            pooled_singles,
        )
        if stop_event.is_set() or node_failed.is_set():
            return
    queues = _single_gpu_queues(config, singles)

    def worker(gpu: int, jobs: Iterable[Job]) -> None:
        port = _resource_port(config, (gpu,))
        run_sessions(jobs, gpus=(gpu,), port=port, label=f"gpu-{gpu}")

    workers = len(queues) if not headline or calibration.get("enabled") else min(1, len(queues))
    if workers:
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
            if not job.parameters.get("probe") or config.has_exact_draft(job.model, job.backend):
                continue
            attempt_number = state.next_attempt(job.job_id)
            output_dir = config.run_dir / "jobs" / job.job_id / f"attempt-{attempt_number:02d}"
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
    if node == "E3b-pilot" and state.selection("deployment_widths_tuned", None) is None:
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
        calibration = tuple(
            job for job in pending if job.parameters.get("workload") == "confidence_calibration"
        )
        _run_pending_jobs(
            config,
            state,
            node,
            stop_event,
            calibration,
        )
        if stop_event.is_set():
            return
        weight = _select_confidence_weight(state)
        state.set_selection("dspark_confidence_weight", weight)
        state.set_selection(
            "dspark_confidence_temperature",
            _select_confidence_temperature(state, weight),
        )
        pending = state.pending_jobs(node)
    if node == "E1a" and state.selection("e1a_finalists", None) is None:
        confirmation = tuple(
            job for job in pending if isinstance(job.parameters.get("finalist_slot"), int)
        )
        screen = tuple(
            job
            for job in pending
            if job not in confirmation
            and job.parameters.get("workload") != "confidence_calibration"
        )
        _run_pending_jobs(config, state, node, stop_event, screen)
        if stop_event.is_set():
            return
        finalists = _rank_candidates(state, node, 4)
        if len(finalists) != 4:
            raise ScientificFailure("E1a did not produce four confirmation finalists")
        state.set_selection("e1a_finalists", finalists)
        pending = tuple(
            job
            for job in state.pending_jobs(node)
            if isinstance(job.parameters.get("finalist_slot"), int)
        )
    if node == "E5-pilot" and state.selection("tts_batched_geometry", None) is None:
        calibration = _tts_batched_calibration_jobs(state)
        if not calibration:
            raise ScientificFailure("E1 produced no LoRA geometry for TTS batching")
        state.add_internal_jobs(calibration)
        _run_pending_jobs(
            config,
            state,
            "TTS-batched-calibration",
            stop_event,
            state.pending_jobs("TTS-batched-calibration"),
        )
        if stop_event.is_set():
            return
        _require_internal_jobs(state, "TTS-batched-calibration")
        state.set_selection("tts_batched_geometry", _select_tts_batched_geometry(state))
        pending = state.pending_jobs(node)
    if node.startswith("E5"):
        anchors, _ = _e5_execution_phases(pending)
        if anchors:
            _run_pending_jobs(config, state, node, stop_event, anchors)
        if stop_event.is_set():
            return
        pending = state.pending_jobs(node)
    if node == "E6-pilot" and state.selection("e6_common_loads", None) is None:
        interfaces = tuple(job for job in state.jobs(node) if job.parameters.get("interface_fit"))
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
            capabilities = _complete_e6_interface_rows(config, state, interfaces)
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
            state.set_selection("e6_common_loads", _select_e6_common_loads(state, capabilities))
    if node in {"E6-pilot", "E6-final"}:
        if stop_event.is_set():
            return
        pending = state.pending_jobs(node)
        loads = state.selection("e6_common_loads", {})
        capabilities = {
            model: set(modes) for model, modes in state.selection("e6_capabilities", {}).items()
        }
        for job in pending:
            if not _e6_role_supported(job.method, capabilities.get(job.model, set())):
                state.skip_job(job.job_id, "required NEXTN update mode is infeasible")
            elif job.model not in loads:
                state.skip_job(job.job_id, "model has no feasible TP2 NEXTN common load")
        pending = state.pending_jobs(node)
    _run_pending_jobs(config, state, node, stop_event, pending)
    if stop_event.is_set():
        return
    if node == "TTS-Cal" and state.selection("tts_confirmation_complete", None) is None:
        if not _all_jobs_completed(state.status_counts(node)):
            return
        confirmations = _tts_confirmation_jobs(state)
        state.add_internal_jobs(confirmations)
        _run_pending_jobs(
            config,
            state,
            "TTS-Cal-confirmation",
            stop_event,
            state.pending_jobs("TTS-Cal-confirmation"),
        )
        if stop_event.is_set():
            return
        _require_internal_jobs(state, "TTS-Cal-confirmation")
        state.set_selection("tts_recipe", _select_tts_recipe(state))
        state.set_selection("tts_confirmation_complete", True)
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
        state.set_selection("e1_common_load", _select_e1_common_load(state, len(probes)))
        state.set_selection("e1_geometries", _rank_e1_geometries(state))


def _metric_rows(state: StateStore, node: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    excluded = set(state.selection("formal_evidence_exclusions", []))
    candidates = list(state.completed_attempt_rows(node))
    if node in PAPER_NODES:
        for job_id, directory in state.completed_attempt_rows():
            config_path = directory / "config.json"
            if not config_path.is_file():
                continue
            config = json.loads(config_path.read_text())
            if config.get("parameters", {}).get("source_node") == node:
                candidates.append((job_id, directory))
    for job_id, directory in candidates:
        if job_id in excluded:
            continue
        config_path, metrics_path = directory / "config.json", directory / "metrics.json"
        if config_path.is_file() and metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            metrics["source_attempt"] = int(directory.name.removeprefix("attempt-"))
            metrics["source_attempt_dir"] = str(directory)
            config = json.loads(config_path.read_text())
            if config.get("parameters", {}).get("source_node") == node:
                config = dict(config)
                config["node"] = node
                config["parameters"] = {
                    key: value
                    for key, value in config["parameters"].items()
                    if key
                    not in {
                        "source_node",
                        "replaces_job_id",
                        "reconciliation_kind",
                    }
                }
            segments = metrics.get("segments")
            if isinstance(segments, list):
                for segment in segments:
                    segment_config = dict(segment["config"])
                    if str(segment_config.get("job_id")) in excluded:
                        continue
                    segment_metrics = dict(segment["metrics"])
                    segment_config, segment_metrics = normalize_attempt_semantics(
                        segment_config,
                        segment_metrics,
                        Path(segment["attempt_dir"]),
                    )
                    segment_metrics["source_attempt"] = metrics["source_attempt"]
                    segment_metrics["source_attempt_dir"] = segment["attempt_dir"]
                    rows.append((segment_config, segment_metrics))
            else:
                config, metrics = normalize_attempt_semantics(config, metrics, directory)
                rows.append((config, metrics))
    return rows


def _trajectory_group(config: dict[str, Any]) -> tuple[Any, ...]:
    parameters = config.get("parameters", {})
    return (
        config.get("model"),
        config.get("task"),
        config.get("context"),
        config.get("load"),
        config.get("block"),
        *(
            parameters.get(name)
            for name in ("regime", "width_panel", "topology", "cohorts", "popularity")
        ),
    )


def _check_greedy_trajectories(state: StateStore, node: str) -> None:
    groups: dict[tuple[Any, ...], dict[str, list[tuple[int, ...]]]] = {}
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
                groups.setdefault(_trajectory_group(config), {}).setdefault(policy, []).append(
                    tuple(row["output_ids"])
                )
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
        and config.get("parameters", {}).get("workload")
        not in {"confidence_calibration", "dspark_finalist_confirmation"}
    ]
    candidates.sort(key=lambda item: item[:-1])
    return [row[-1] for row in candidates[:keep]]


def _select_tts_recipe(state: StateStore) -> dict[str, Any]:
    groups: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    rows = _metric_rows(state, "TTS-Cal")
    rows.extend(_metric_rows(state, "TTS-Cal-confirmation"))
    for config, metrics in rows:
        if metrics.get("feasible") is False:
            continue
        parameters = {
            key: value
            for key, value in config["parameters"].items()
            if key not in {"workload", "confirmation_block", "stimulus_id"}
        }
        key = json.dumps(parameters, sort_keys=True)
        groups.setdefault(key, (parameters, []))[1].append(metrics)
    candidates = [
        (
            -float(np.mean([row["goodput"] for row in rows])),
            int(np.max([row["peak_hbm_bytes"] for row in rows])),
            float(np.max([row["itl_p99_ms"] for row in rows])),
            parameters,
        )
        for parameters, rows in groups.values()
        if len(rows) >= 4
    ]
    if not candidates:
        raise ScientificFailure("TTS-Cal produced no complete four-window recipe")
    return min(candidates, key=lambda row: row[:-1])[-1]


def _job_from_metric_config(config: dict[str, Any]) -> Job:
    return Job(**{name: config[name] for name in Job.__dataclass_fields__})


def _tts_confirmation_jobs(state: StateStore) -> tuple[Job, ...]:
    candidates = []
    for config, metrics in _metric_rows(state, "TTS-Cal"):
        if metrics.get("feasible") is False:
            continue
        candidates.append(
            (
                -float(metrics["goodput"]),
                int(metrics["peak_hbm_bytes"]),
                float(metrics["itl_p99_ms"]),
                config,
            )
        )
    rows = []
    for finalist, (_, _, _, config) in enumerate(sorted(candidates)[:9]):
        source = _job_from_metric_config(config)
        for block in range(4):
            rows.append(
                replace(
                    source,
                    job_id=f"tts-confirm-{finalist:02d}-block-{block:02d}",
                    node="TTS-Cal-confirmation",
                    ordinal=len(rows),
                    block=block,
                    parameters={
                        **source.parameters,
                        "workload": "tts_calibration_confirmation",
                        "confirmation_block": block,
                    },
                )
            )
    return tuple(rows)


def _tts_s10_confirmation_jobs(state: StateStore) -> tuple[Job, ...]:
    sources = {
        float(job.parameters["learning_rate"]): job
        for job in state.jobs("TTS-Cal")
        if job.method == "tts"
        and int(job.parameters.get("stride", -1)) == FORMAL_ADAPTATION_STRIDE
        and float(job.parameters.get("learning_rate", -1.0)) in {3e-5, 1e-4}
    }
    if set(sources) != {3e-5, 1e-4}:
        raise ScientificFailure("TTS-Cal has no S=10 source rows for both confirmation LRs")
    rows: list[Job] = []
    for learning_rate in (3e-5, 1e-4):
        source = sources[learning_rate]
        for block in range(4):
            rows.append(
                replace(
                    source,
                    job_id=f"tts-s10-lr-{learning_rate:.0e}-block-{block:02d}",
                    node="TTS-S10-confirmation",
                    ordinal=len(rows),
                    block=block,
                    parameters={
                        **source.parameters,
                        "learning_rate": learning_rate,
                        "stride": FORMAL_ADAPTATION_STRIDE,
                        "workload": "tts_stride10_confirmation",
                        "confirmation_block": block,
                    },
                )
            )
    return tuple(rows)


def _select_tts_s10_recipe(state: StateStore) -> dict[str, Any]:
    grouped: dict[float, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for config, metrics in _metric_rows(state, "TTS-S10-confirmation"):
        learning_rate = float(config["parameters"]["learning_rate"])
        grouped.setdefault(learning_rate, []).append((config, metrics))
    candidates: list[dict[str, Any]] = []
    for learning_rate in (3e-5, 1e-4):
        rows = grouped.get(learning_rate, [])
        blocks = {int(config.get("block", -1)) for config, _ in rows}
        valid = (
            blocks == set(range(4))
            and len(rows) == 4
            and all(metrics.get("feasible") is not False for _, metrics in rows)
            and all(int(metrics.get("updates_published", 0)) >= 1 for _, metrics in rows)
            and all(
                all(int(metrics.get(counter, 0)) == 0 for counter in SAFETY_COUNTERS)
                for _, metrics in rows
            )
        )
        if not valid:
            continue
        goodputs = np.asarray([float(metrics["goodput"]) for _, metrics in rows])
        if np.any(~np.isfinite(goodputs)) or np.any(goodputs <= 0):
            continue
        accepted_per_call = [
            float(metrics.get("accepted_drafts", 0.0))
            / max(float(metrics.get("target_calls", 0.0)), 1.0)
            for _, metrics in rows
        ]
        candidates.append(
            {
                "learning_rate": learning_rate,
                "geometric_mean_goodput": float(np.exp(np.mean(np.log(goodputs)))),
                "accepted_drafts_per_target_call": float(np.mean(accepted_per_call)),
                "p99_itl_ms": float(max(metrics["itl_p99_ms"] for _, metrics in rows)),
            }
        )
    if not candidates:
        raise ScientificFailure("S=10 TTS confirmation produced no valid four-block recipe")
    candidates.sort(key=lambda row: row["geometric_mean_goodput"], reverse=True)
    selected = candidates[0]
    if len(candidates) == 2:
        best, other = candidates
        relative_gap = (
            best["geometric_mean_goodput"] / other["geometric_mean_goodput"] - 1.0
        )
        if relative_gap <= 0.01:
            selected = min(
                candidates,
                key=lambda row: (
                    -row["accepted_drafts_per_target_call"],
                    row["p99_itl_ms"],
                    row["learning_rate"],
                ),
            )
    audit = {
        "formal_stride": FORMAL_ADAPTATION_STRIDE,
        "blocks": 4,
        "decision_rule": (
            "paired geometric-mean goodput; within one percent, higher accepted "
            "drafts per target call then lower p99 ITL"
        ),
        "candidates": candidates,
        "selected_learning_rate": selected["learning_rate"],
    }
    path = state.run_dir / "stages" / "TTS-S10-confirmation" / "selection_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, audit)
    return {
        "learning_rate": selected["learning_rate"],
        "stride": FORMAL_ADAPTATION_STRIDE,
        "optimizer": "adam",
        "parameterization": "full",
        "scope": "all",
    }


def _is_reconstruction_repair_source(job: Job) -> bool:
    parameters = job.parameters
    if job.node == "E1":
        return (
            job.method == "lightcone_candidate"
            and parameters.get("optimizer") in {"adamw", "sgdm"}
            and parameters.get("parameterization") == "lora"
            and parameters.get("rank") == 16
            and parameters.get("scope") in {"last5", "all"}
        )
    return (
        job.node == "E2-r1"
        and job.method == "lightcone_candidate"
        and parameters.get("optimizer") == "nag"
        and float(parameters.get("learning_rate", -1.0)) == 3e-5
        and parameters.get("parameterization") == "lora"
        and parameters.get("rank") == 1
        and parameters.get("scope") == "last1"
        and parameters.get("schedule") == "constant"
    )


def _s10_reconciliation_jobs(state: StateStore) -> tuple[Job, ...]:
    sources: list[tuple[Job, str]] = []
    sources.extend(
        (job, "formal_stride")
        for job in state.jobs("E1")
        if job.method in {"tts", "l0_naive"}
    )
    for node in ("E2-r0", "E2-r1", "E2-r2", "E2-r3"):
        sources.extend(
            (job, "formal_stride")
            for job in state.jobs(node)
            if job.method in {"tts", "l0_naive"}
        )
    sources.extend(
        (job, "formal_stride")
        for job in state.jobs("E4-screen")
        if job.method == "tts" and job.parameters.get("workload") == "tts_update_steps"
    )
    for node in ("E1", "E2-r1"):
        sources.extend(
            (job, "masked_logit_reconstruction")
            for job in state.jobs(node)
            if _is_reconstruction_repair_source(job)
        )
    unique: dict[str, tuple[Job, str]] = {job.job_id: (job, kind) for job, kind in sources}
    if len(unique) != 19:
        raise ScientificFailure(
            f"formal S=10 reconciliation expected 19 completed source jobs, found {len(unique)}"
        )
    rows: list[Job] = []
    for source, kind in sorted(unique.values(), key=lambda row: row[0].job_id):
        rows.append(
            replace(
                source,
                job_id=f"s10-repair__{source.job_id}",
                node="S10-reconciliation",
                ordinal=len(rows),
                parameters={
                    **source.parameters,
                    "source_node": source.node,
                    "replaces_job_id": source.job_id,
                    "reconciliation_kind": kind,
                },
            )
        )
    return tuple(rows)


def _formalize_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    return {**recipe, "stride": FORMAL_ADAPTATION_STRIDE}


def _enforce_formal_recipe_selections(state: StateStore) -> None:
    for name in ("tts_recipe", "lightcone_recipe", "dspark_recipe", "tts_batched_geometry"):
        value = state.selection(name, None)
        if isinstance(value, dict):
            state.set_selection(name, _formalize_recipe(value))


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
    return [parameters for parameters, _, _, _ in pareto[:MAX_E2_GEOMETRIES]]


def _rank_e2_candidates(state: StateStore, node: str, keep: int) -> list[dict[str, Any]]:
    rows = _metric_rows(state, node)
    baselines = {
        config["method"]: metrics
        for config, metrics in rows
        if config["method"] in {"static", "tts"}
    }
    if set(baselines) != {"static", "tts"}:
        return []
    static_goodput = float(baselines["static"]["goodput"])
    tts_user_speed = baselines["tts"].get("per_user_generation_speed")
    if not isinstance(tts_user_speed, (int, float)) or tts_user_speed <= 0:
        return []
    candidates: dict[str, tuple[Any, ...]] = {}
    for config, metrics in rows:
        if config["method"] != "lightcone_candidate":
            continue
        if metrics.get("feasible") is False:
            continue
        goodput = float(metrics["goodput"])
        user_speed = metrics.get("per_user_generation_speed")
        if not isinstance(user_speed, (int, float)) or user_speed <= 0:
            continue
        objective = min(
            goodput / static_goodput,
            float(user_speed) / float(tts_user_speed),
        )
        recipe = {
            name: value
            for name, value in config["parameters"].items()
            if name
            not in {
                "round",
                "minimum_updates",
                "fixed_role",
                "stride",
            }
        }
        row = (
                -objective,
                int(metrics["peak_hbm_bytes"]),
                float(metrics["itl_p99_ms"]),
                float(metrics.get("exposed_update_ms", math.inf)),
                recipe,
            )
        key = _e2_recipe_key(recipe)
        if key not in candidates or row[:-1] < candidates[key][:-1]:
            candidates[key] = row
    ordered = sorted(candidates.values(), key=lambda row: row[:-1])
    return [row[-1] for row in ordered[:keep]]


def _e2_keep_count(candidate_count: int, feasible_count: int, round_index: int) -> int:
    """Apply the registered halving floor without inventing infeasible finalists."""

    if candidate_count < 0 or feasible_count < 0 or feasible_count > candidate_count:
        raise ValueError("invalid E2 candidate cardinality")
    if feasible_count == 0:
        return 0
    requested = 1 if round_index == 3 else max(math.ceil(candidate_count / 4), 21)
    return min(requested, feasible_count)


_E2_RECIPE_PARAMETER_KEYS = (
    "parameterization",
    "rank",
    "scope",
    "optimizer",
    "learning_rate",
    "schedule",
)


def _e2_recipe_key(recipe: dict[str, Any]) -> str:
    return json.dumps(
        {
            key: recipe[key]
            for key in _E2_RECIPE_PARAMETER_KEYS
            if key in recipe
        },
        sort_keys=True,
    )


def _preserve_or_audit_e2_selection(
    state: StateStore,
    node: str,
    selection_name: str,
    winners: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = state.selection(selection_name, None)
    if not isinstance(existing, list):
        return winners
    same_set = {_e2_recipe_key(row) for row in existing} == {
        _e2_recipe_key(row) for row in winners
    }
    path = state.run_dir / "stages" / node / "concurrency_metric_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "metric_semantics": "per_request_native_v2",
            "previous_finalists": len(existing),
            "corrected_finalists": len(winners),
            "same_scientific_set": same_set,
            "action": "preserved_existing_order" if same_set else "selection_changed",
        },
    )
    if same_set:
        return existing
    return winners


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
            [BSpline(knots, row, 3)(points, nu=derivative) for row in coefficients]
        )

    constraints = np.vstack((basis(np.asarray([left]), 2), basis(np.asarray([right]), 2)))
    natural = null_space(constraints)
    design = basis(x) @ natural
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return tuple(basis(evaluation, derivative) @ natural @ beta for derivative in (0, 1, 2))


def _context_splines(state: StateStore, node: str) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[int, list[float]]] = {}
    for config, metrics in _metric_rows(state, node):
        if metrics.get("feasible") is False:
            continue
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
        checkpoints = metrics.get("trajectory_checkpoints")
        points = (
            [
                (int(row["generation_tokens"]), float(row["goodput"]))
                for row in checkpoints
                if isinstance(row.get("goodput"), (int, float)) and row["goodput"] > 0
            ]
            if parameters.get("regime") == "short_input_long_generation"
            and isinstance(checkpoints, list)
            else (
                [(context, float(metrics["goodput"]))]
                if isinstance(metrics.get("goodput"), (int, float))
                and metrics["goodput"] > 0
                else []
            )
        )
        for length, goodput in points:
            grouped.setdefault(key, {}).setdefault(length, []).append(goodput)
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
        if not config_path.is_file() or not metrics_path.is_file() or not requests_path.is_file():
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
        requests = _read_jsonl(requests_path)
        checkpoint_rows = metrics.get("trajectory_checkpoints")
        lengths = (
            [int(row["generation_tokens"]) for row in checkpoint_rows]
            if parameters.get("regime") == "short_input_long_generation"
            and isinstance(checkpoint_rows, list)
            else [context]
        )
        for length in lengths:
            rows = []
            starts = []
            ends = []
            for request in requests:
                timestamps = request.get("native_token_timestamps_ns", [])
                if length != context and len(timestamps) >= length:
                    starts.append(int(timestamps[0]))
                    ends.append(int(timestamps[length - 1]))
                    rows.append((length, (ends[-1] - starts[-1]) / 1e9))
                elif length == context:
                    rows.append(
                        (
                            int(request["completion_tokens"]),
                            float(request["elapsed_seconds"]),
                        )
                    )
            if rows:
                block_duration = (max(ends) - min(starts)) / 1e9 if starts else float(duration)
                request_groups.setdefault(key, {}).setdefault(length, {})[block] = (
                    rows,
                    block_duration,
                )
                request_counts[key] += len(rows)
        if requests:
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
        if not contexts[0] < fixed[0] or not fixed[-1] < contexts[-1]:
            continue
        log_contexts = np.log(contexts)
        fitted, elasticity, curvature = _natural_spline_fit(
            log_contexts, np.log(goodput), log_contexts
        )
        by_context = request_groups.get(key, {})
        common_blocks = (
            set.intersection(*(set(by_context.get(int(context), {})) for context in contexts))
            if len(contexts)
            else set()
        )
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
                "length_axis": (
                    "generated_tokens"
                    if key[1] == "short_input_long_generation"
                    else "total_context_tokens"
                ),
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
            differences = np.log(candidate["fitted_goodput"]) - np.log(reference["fitted_goodput"])
            point = _first_crossover(contexts, differences)
            candidate_draws = candidate["_bootstrap_log_fitted"]
            reference_draws = reference["_bootstrap_log_fitted"]
            roots = []
            if len(candidate_draws) == len(reference_draws):
                roots = [
                    root
                    for left, right in zip(candidate_draws, reference_draws, strict=True)
                    if (root := _first_crossover(contexts, np.asarray(left) - np.asarray(right)))
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
        resolved = len(values) >= 3
        estimate = block_bootstrap_interval(values) if resolved else (None, None, None)
        result.append(
            {
                "method": method,
                "backend": backend,
                "load": load,
                "offered_requests": offered,
                "completed_requests": completed,
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


def _exact_sign_flip(log_ratios: np.ndarray) -> float:
    observed = float(np.mean(log_ratios))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(log_ratios))))
    return float((np.sum(np.mean(signs * log_ratios, axis=1) >= observed) + 1) / (len(signs) + 1))


def _frontier_area(points: list[tuple[float, float]]) -> float | None:
    rows = sorted((speed, throughput) for speed, throughput in points if speed > 0 and throughput > 0)
    if len(rows) < 2:
        return None
    x = np.log([row[0] for row in rows])
    y = np.log([row[1] for row in rows])
    return float(np.trapezoid(y, x))


def _e5_frontier_statistic(state: StateStore) -> dict[str, object] | None:
    baseline_method = state.selection("e5_operational_baseline", None)
    if baseline_method not in {"target_only", "static"}:
        return None
    points: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for config, metrics in _metric_rows(state, "E5-final"):
        if config.get("backend") not in {"NONE", "DFLASH"}:
            continue
        block = config.get("block")
        method = config.get("method")
        if not isinstance(block, int) or method not in {baseline_method, "lightcone"}:
            continue
        load = str(config.get("load", ""))
        if not load.startswith("closed_loop_c"):
            continue
        throughput = float(metrics.get("goodput", 0.0))
        speed_value = metrics.get("per_user_generation_speed")
        if not isinstance(speed_value, (int, float)) or speed_value <= 0:
            raise ScientificFailure("E5 frontier lacks native per-request generation speed")
        speed = float(speed_value)
        points.setdefault((block, str(method)), []).append((speed, throughput))
    candidate = []
    baseline = []
    for block in PRIMARY_BLOCKS:
        left = _frontier_area(points.get((block, "lightcone"), []))
        right = _frontier_area(points.get((block, baseline_method), []))
        if left is None or right is None:
            return None
        candidate.append(left)
        baseline.append(right)
    log_ratios = np.asarray(candidate) - np.asarray(baseline)
    estimate, low, high = paired_bca_interval(log_ratios, np.zeros_like(log_ratios))
    return {
        "hypothesis": "H3",
        "candidate": "lightcone",
        "baseline": "operational_baseline",
        "metric": "throughput_interactivity_frontier_log_auc",
        "blocks": list(PRIMARY_BLOCKS),
        "operational_baseline_method": baseline_method,
        "mean_log_ratio": estimate,
        "relative_effect": math.exp(estimate) - 1.0,
        "ci95_relative_low": math.exp(low) - 1.0,
        "ci95_relative_high": math.exp(high) - 1.0,
        "p_value": _exact_sign_flip(log_ratios),
        "reducer": "paired_log_log_frontier_auc_bca",
    }


def _confirmatory_holm(
    state: StateStore, e5_frontier: dict[str, object] | None
) -> list[dict[str, object]]:
    path = state.run_dir / "stages" / "E3b-final" / "statistics.json"
    if not path.is_file() or e5_frontier is None:
        return []
    e3 = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for hypothesis, baseline in (("H1", "tts"), ("H2", "operational_baseline")):
        matches = [
            row
            for row in e3
            if row.get("candidate") == "lightcone"
            and row.get("baseline") == baseline
            and row.get("workload") == "primary_long_history"
            and row.get("context") == 32768
        ]
        if len(matches) != 1:
            return []
        rows.append({**matches[0], "hypothesis": hypothesis})
    rows.append(e5_frontier)
    decisions = holm_decisions([float(row["p_value"]) for row in rows])
    return [{**row, "holm_reject": decision} for row, decision in zip(rows, decisions, strict=True)]


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
    recipes: dict[str, dict[str, Any]] = {}
    observed: set[str] = set()
    for config, metrics in _metric_rows(state, "E0-tune"):
        method = config["method"]
        if method not in E0_ONLINESPEC_METHODS or not config["parameters"].get(
            "recipe_validation"
        ):
            continue
        observed.add(method)
        expected = E0_ONLINESPEC_RECIPES[method]
        actual = {name: config["parameters"].get(name) for name in expected}
        if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ScientificFailure(f"E0 {method} source-transfer recipe drifted")
        if metrics.get("feasible") is False or metrics.get("slo_pass") is not True:
            continue
        key = "|".join((config["model"], config["backend"], method))
        recipes[key] = dict(expected)
    if observed != set(E0_ONLINESPEC_METHODS):
        missing = sorted(set(E0_ONLINESPEC_METHODS) - observed)
        raise ScientificFailure(f"E0 lacks source-transfer validation for {missing}")
    return recipes


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


def _mechanism_summary(state: StateStore, node: str) -> list[dict[str, object]]:
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
        _write_json(summary_dir / "mechanism.json", _mechanism_summary(state, node))
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
            decisions = (
                benjamini_hochberg([float(row["p_value"]) for row in secondary])
                if secondary
                else ()
            )
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
        _write_json(summary_dir / "tail_latency.json", _e5_tail_statistics(state, node))
    if node == "E5-final":
        frontier = _e5_frontier_statistic(state)
        _write_json(summary_dir / "serving_frontier.json", frontier)
        _write_json(
            summary_dir / "confirmatory_holm.json",
            _confirmatory_holm(state, frontier),
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
        pairs = [rows for rows in timings.values() if {"isolated", "concurrent"} <= rows.keys()]
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
                    calibrations[metric] = paired_relative_bca_interval(candidate, baseline)
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
        anchors = [(item, metrics) for item, metrics in rows if item.get("context") == 40928]
        feasible_loads: list[str] = []
        for load in {str(item["load"]) for item, _ in anchors}:
            target_regimes = {
                item["parameters"].get("regime")
                for item, metrics in anchors
                if item["method"] == "target_only"
                and item["load"] == load
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
            raise ScientificFailure("E3a found no common 40,928-token Target-only/Static load")
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
        state.set_selection("e3a", {"width": selected_width, "load": common_load})
        state.set_selection(
            "deployment_widths",
            {"static": selected_width},
        )
    elif node == "TTS-Cal":
        state.set_selection("tts_recipe", _select_tts_recipe(state))
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
        candidate_count = len(
            {
                _e2_recipe_key(item["parameters"])
                for item, _ in _metric_rows(state, node)
                if item["method"] == "lightcone_candidate"
            }
        )
        feasible = _rank_e2_candidates(state, node, candidate_count)
        keep = _e2_keep_count(candidate_count, len(feasible), round_index)
        winners = feasible[:keep]
        selection_name = (
            "lightcone_recipe" if round_index == 3 else f"e2_round_{round_index}"
        )
        winners = _preserve_or_audit_e2_selection(
            state,
            node,
            selection_name,
            winners,
        )
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
            state.set_selection(
                selection_name,
                winners[0] if round_index == 3 else winners,
            )
    elif node == "E1a":
        winners = _rank_candidates(state, node, 1)
        if winners:
            state.set_selection("dspark_recipe", winners[0])
    elif node == "E5-pilot":
        scores: dict[str, float] = {}
        for config, metrics in _metric_rows(state, node):
            method = str(config["method"])
            if method not in {"target_only", "static"}:
                continue
            if str(config.get("load", "")).startswith("closed_loop_c"):
                scores.setdefault(method, 0.0)
                scores[method] += float(metrics.get("goodput", 0.0))
        if scores:
            state.set_selection("e5_operational_baseline", max(scores, key=scores.get))
    elif node == "E0-tune":
        valid = _select_valid_e0(state)
        recipes = _select_e0_recipes(state)
        state.set_selection("valid_e0", valid)
        state.set_selection("e0_recipes", recipes)
        feasible = {key.rsplit("|", 1)[-1] for key in recipes}
        state.set_selection(
            "E0_infeasible_onlinespec_methods",
            sorted(set(E0_ONLINESPEC_METHODS) - feasible),
        )


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
        previous = yaml.safe_load(saved.read_text(encoding="utf-8"))
        old_paths = dict(previous["paths"])
        new_paths = dict(normalized["paths"])
        old_datasets = dict(old_paths.pop("datasets"))
        new_datasets = dict(new_paths.pop("datasets"))
        old_paths.pop("sglang_root")
        new_paths.pop("sglang_root")
        same_existing_datasets = all(
            new_datasets.get(name) == value for name, value in old_datasets.items()
        )
        previous = {**previous, "paths": old_paths}
        current = {**normalized, "paths": new_paths}
        if previous != current or not same_existing_datasets:
            raise RuntimeError("run directory belongs to a different experiment config")
        saved.write_text(yaml.safe_dump(normalized, sort_keys=True), encoding="utf-8")
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
    versions = (
        json.loads(runtime.stdout)
        if runtime.returncode == 0
        else {"runtime_error": runtime.stderr.strip()}
    )
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


def _dependency_reason(config: ExperimentConfig, state: StateStore, node: str) -> str | None:
    if node != "preflight" and state.stage_status("preflight") != "completed":
        return "preflight did not complete"
    stage_requirements = {
        "E3b-final": "E3b-pilot",
        "E5-final": "E5-pilot",
        "E6-final": "E6-pilot",
        "E0-final": "E0-pilot",
    }
    required_stage = stage_requirements.get(node)
    if required_stage and state.stage_status(required_stage) != "completed":
        return f"{required_stage} did not complete"
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
    return None


def _e2_missing_dependency_jobs(
    state: StateStore,
    node: str,
    recipes: list[dict[str, Any]],
) -> tuple[Job, ...]:
    existing = {
        _e2_recipe_key(config["parameters"])
        for config, _ in _metric_rows(state, node)
        if config["method"] == "lightcone_candidate"
    }
    planned = materialize(node, e2_rows=recipes)
    rows: list[Job] = []
    for job in planned:
        if job.method != "lightcone_candidate":
            continue
        if _e2_recipe_key(job.parameters) in existing:
            continue
        learning_rate = format(float(job.parameters["learning_rate"]), ".12g")
        learning_rate = learning_rate.replace("+", "").replace("-", "m").replace(".", "p")
        recipe_identity = "__".join(
            (
                f"{job.parameters['parameterization']}-r{job.parameters['rank']}",
                str(job.parameters["scope"]),
                str(job.parameters["optimizer"]),
                f"lr-{learning_rate}",
                str(job.parameters["schedule"]),
            )
        )
        rows.append(
            replace(
                job,
                # Ordinals are positions within a selected candidate set and can
                # legitimately refer to a different recipe after the preceding
                # round is re-audited.  Keep the internal evidence identity tied
                # to the scientific recipe so a resumed audit can add the new
                # dependency without mutating or colliding with the old attempt.
                job_id=f"s10-e2-dependency-v2__{node}__{recipe_identity}",
                node="S10-e2-dependency-repair",
                ordinal=job.ordinal,
                parameters={
                    **job.parameters,
                    "source_node": node,
                    "reconciliation_kind": "e2_dependency_closure",
                },
            )
        )
    return tuple(rows)


def _skip_satisfied_e2_dependency_jobs(state: StateStore, node: str) -> int:
    """Retire pending closure work already backed by equivalent valid evidence."""

    existing = {
        _e2_recipe_key(config["parameters"])
        for config, _ in _metric_rows(state, node)
        if config["method"] == "lightcone_candidate"
    }
    skipped = 0
    for job in state.pending_jobs("S10-e2-dependency-repair"):
        if job.parameters.get("source_node") != node:
            continue
        if _e2_recipe_key(job.parameters) not in existing:
            continue
        state.skip_job(job.job_id, "dependency satisfied by equivalent completed evidence")
        skipped += 1
    return skipped


def _exclude_redundant_e2_dependency_jobs(state: StateStore, node: str) -> int:
    """Exclude v2 closure reruns when older equivalent evidence already exists."""

    canonical: set[str] = set()
    versioned: list[tuple[str, str]] = []
    for job_id, directory in state.completed_attempt_rows():
        config_path = directory / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        parameters = dict(config.get("parameters", {}))
        if parameters.get("source_node", config.get("node")) != node:
            continue
        if config.get("method") != "lightcone_candidate":
            continue
        recipe_key = _e2_recipe_key(parameters)
        if job_id.startswith("s10-e2-dependency-v2__"):
            versioned.append((job_id, recipe_key))
        else:
            canonical.add(recipe_key)
    redundant = {job_id for job_id, recipe_key in versioned if recipe_key in canonical}
    exclusions = set(state.selection("formal_evidence_exclusions", []))
    exclusions.update(redundant)
    state.set_selection("formal_evidence_exclusions", sorted(exclusions))
    return len(redundant)


def _bugfix_replacement(source: Job, *, ordinal: int, reason: str) -> Job:
    return replace(
        source,
        job_id=f"bugfix-v1__{source.job_id}",
        node="bugfix-reconciliation-v1",
        ordinal=ordinal,
        parameters={
            **source.parameters,
            "source_node": source.node,
            "replaces_job_id": source.job_id,
            "reconciliation_kind": reason,
        },
    )


def _bugfix_reconciliation_jobs(state: StateStore) -> tuple[Job, ...]:
    """Materialize only evidence known to be contaminated by the four fixed bugs."""

    sources: dict[str, tuple[Job, str]] = {}
    for job in state.jobs("E2-r0"):
        if job.method != "lightcone_candidate":
            continue
        optimizer = str(job.parameters.get("optimizer"))
        schedule = str(job.parameters.get("schedule"))
        if optimizer == "muon" or schedule == "cosine_to_zero":
            sources[job.job_id] = (job, "e2_optimizer_or_cosine_horizon")

    for job in state.jobs("E1a"):
        if job.parameters.get("workload") == "confidence_calibration":
            sources[job.job_id] = (
                replace(
                    job,
                    parameters={
                        **job.parameters,
                        "scope": "last1_native_heads",
                        "parameterization": "full",
                        "regime": "short_input_long_generation",
                        "generation_tokens": GEOMETRY_GENERATION_TOKENS,
                        "stride": FORMAL_ADAPTATION_STRIDE,
                    },
                ),
                "e1a_native_confidence_calibration",
            )

    e3a_prefix = "E3a__000111__static__Qwen-Qwen3-8B__DFLASH__MATH-500__segment-"
    for job in state.jobs("E3a-segments"):
        if job.job_id in {f"{e3a_prefix}000", f"{e3a_prefix}001"}:
            sources[job.job_id] = (job, "screening_runtime_error_classification")

    tts_ordinals = {8, 24, 56}
    for job in state.jobs("TTS-Cal"):
        if job.ordinal in tts_ordinals:
            sources[job.job_id] = (job, "pre_reconstruction_stride1")

    counts = Counter(reason for _, reason in sources.values())
    expected = {
        "e2_optimizer_or_cosine_horizon": 90,
        "e1a_native_confidence_calibration": 5,
        "screening_runtime_error_classification": 2,
        "pre_reconstruction_stride1": 3,
    }
    if counts != expected:
        raise ScientificFailure(
            f"bugfix reconciliation source mismatch: found {dict(counts)}, expected {expected}"
        )
    return tuple(
        _bugfix_replacement(source, ordinal=index, reason=reason)
        for index, (source, reason) in enumerate(
            sorted(sources.values(), key=lambda row: row[0].job_id)
        )
    )


def _set_e2_expected_evidence(
    state: StateStore,
    node: str,
    recipes: list[dict[str, Any]],
) -> int:
    """Exclude stale downstream finalists while preserving all raw attempts."""

    expected = {_e2_recipe_key(recipe) for recipe in recipes}
    previous = set(state.selection("bugfix_v1_e2_obsolete_ids", []))
    exclusions = set(state.selection("formal_evidence_exclusions", [])) - previous
    obsolete: set[str] = set()
    for job_id, directory in state.completed_attempt_rows():
        path = directory / "config.json"
        if not path.is_file():
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        parameters = dict(config.get("parameters", {}))
        effective_node = str(parameters.get("source_node", config.get("node")))
        if effective_node != node or config.get("method") != "lightcone_candidate":
            continue
        for name in ("source_node", "replaces_job_id", "reconciliation_kind"):
            parameters.pop(name, None)
        if _e2_recipe_key(parameters) not in expected:
            obsolete.add(job_id)
    exclusions.update(obsolete)
    state.set_selection("bugfix_v1_e2_obsolete_ids", sorted(obsolete))
    state.set_selection("formal_evidence_exclusions", sorted(exclusions))
    return len(obsolete)


def _audit_e2_after_s10(
    config: ExperimentConfig,
    state: StateStore,
    stop_event: threading.Event,
) -> None:
    timeout_repair = state.selection("formal_registered_load_timeout_repair", None)
    if timeout_repair is None:
        requeued = state.retry_failed_errors(
            "S10-e2-dependency-repair",
            "requests did not complete in a measured cell",
            reason="registered-load timeout classification repair",
        )
        state.set_selection(
            "formal_registered_load_timeout_repair",
            {"version": 1, "requeued": requeued},
        )
    geometries = _rank_e1_geometries(state)
    if not geometries:
        raise ScientificFailure("S=10 E1 reconciliation produced no Pareto geometry")
    state.set_selection("e1_geometries", geometries)
    audit_rows: list[dict[str, Any]] = []
    previous = geometries
    for round_index in range(4):
        node = f"E2-r{round_index}"
        obsolete = 0
        redundant_dependencies = 0
        satisfied_dependencies = 0
        if round_index > 0:
            obsolete = _set_e2_expected_evidence(state, node, previous)
            redundant_dependencies = _exclude_redundant_e2_dependency_jobs(state, node)
            satisfied_dependencies = _skip_satisfied_e2_dependency_jobs(state, node)
            missing = _e2_missing_dependency_jobs(state, node, previous)
            if missing:
                state.add_internal_jobs(missing, storage_node="S10-e2-dependency-repair")
                _run_pending_jobs(
                    config,
                    state,
                    "S10-e2-dependency-repair",
                    stop_event,
                    state.pending_jobs("S10-e2-dependency-repair"),
                )
                if stop_event.is_set():
                    return
                _require_internal_jobs(state, "S10-e2-dependency-repair")
        candidate_count = len(
            {
                _e2_recipe_key(config_row["parameters"])
                for config_row, _ in _metric_rows(state, node)
                if config_row["method"] == "lightcone_candidate"
            }
        )
        feasible = _rank_e2_candidates(state, node, candidate_count)
        keep = _e2_keep_count(candidate_count, len(feasible), round_index)
        winners = feasible[:keep]
        if not winners:
            raise ScientificFailure(f"{node} S=10 audit found no feasible candidates")
        if len(winners) != keep:
            raise ScientificFailure(
                f"{node} S=10 audit retained {len(winners)} candidates, expected {keep}"
            )
        selection_name = "lightcone_recipe" if round_index == 3 else f"e2_round_{round_index}"
        existing = state.selection(selection_name, None)
        existing_rows = [existing] if isinstance(existing, dict) else existing or []
        same_set = {_e2_recipe_key(row) for row in existing_rows} == {
            _e2_recipe_key(row) for row in winners
        }
        selected_rows = existing_rows if same_set else winners
        selected_rows = [_formalize_recipe(dict(row)) for row in selected_rows]
        state.set_selection(
            selection_name,
            selected_rows[0] if round_index == 3 else selected_rows,
        )
        state.set_selection(
            f"{node}_counts",
            {
                "entered": candidate_count,
                "rejected": candidate_count - len(winners),
                "retained": len(winners),
            },
        )
        audit_rows.append(
            {
                "node": node,
                "previous_finalists": len(existing_rows),
                "corrected_finalists": len(winners),
                "same_scientific_set": same_set,
                "dependency_jobs_added": len(missing) if round_index > 0 else 0,
                "redundant_dependency_jobs_excluded": redundant_dependencies,
                "satisfied_dependency_jobs_skipped": satisfied_dependencies,
                "obsolete_evidence_excluded": obsolete,
            }
        )
        previous = selected_rows
    path = state.run_dir / "stages" / "S10-reconciliation" / "e2_finalist_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "formal_stride": FORMAL_ADAPTATION_STRIDE,
            "rounds": audit_rows,
        },
    )


def _run_formal_s10_reconciliation(
    config: ExperimentConfig,
    state: StateStore,
    stop_event: threading.Event,
) -> None:
    _enforce_formal_recipe_selections(state)
    if state.selection("formal_s10_reconciliation_complete", False):
        return
    prerequisites = (
        "TTS-Cal",
        "E1",
        "E2-r0",
        "E2-r1",
        "E2-r2",
        "E2-r3",
        "E4-screen",
    )
    if any(state.stage_status(node) != "completed" for node in prerequisites):
        return
    if state.stage_status("E4-profile") not in {"completed", "failed"}:
        return

    confirmations = _tts_s10_confirmation_jobs(state)
    state.add_internal_jobs(confirmations)
    _run_pending_jobs(
        config,
        state,
        "TTS-S10-confirmation",
        stop_event,
        state.pending_jobs("TTS-S10-confirmation"),
    )
    if stop_event.is_set():
        return
    _require_internal_jobs(state, "TTS-S10-confirmation")
    state.set_selection("tts_recipe", _select_tts_s10_recipe(state))

    repairs = _s10_reconciliation_jobs(state)
    excluded = set(state.selection("formal_evidence_exclusions", []))
    excluded.update(str(job.parameters["replaces_job_id"]) for job in repairs)
    state.set_selection("formal_evidence_exclusions", sorted(excluded))
    state.add_internal_jobs(repairs)
    _run_pending_jobs(
        config,
        state,
        "S10-reconciliation",
        stop_event,
        state.pending_jobs("S10-reconciliation"),
    )
    if stop_event.is_set():
        return
    _require_internal_jobs(state, "S10-reconciliation")
    _audit_e2_after_s10(config, state, stop_event)
    if stop_event.is_set():
        return

    e4_profile_retries = state.retry_failed("E4-profile")
    width_retries = state.retry_failed("E3-width-calibration")
    width_segment_retries = state.retry_failed("E3-width-calibration-segments")
    reopened = state.reopen_skipped(
        (
            "E3b-pilot",
            "E3b-final",
            "E1a",
            "E5-pilot",
            "E5-final",
            "E6-pilot",
            "E6-final",
            "E0-tune",
            "E0-pilot",
            "E0-final",
        )
    )
    _enforce_formal_recipe_selections(state)
    audit_path = state.run_dir / "stages" / "S10-reconciliation" / "reconciliation.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        audit_path,
        {
            "formal_stride": FORMAL_ADAPTATION_STRIDE,
            "tts_confirmation_jobs": len(confirmations),
            "replacement_jobs": len(repairs),
            "excluded_source_jobs": len(excluded),
            "e4_profile_retries": e4_profile_retries,
            "width_calibration_retries": width_retries,
            "width_calibration_segment_retries": width_segment_retries,
            "future_jobs_reopened": reopened,
        },
    )
    state.set_selection("formal_s10_reconciliation_complete", True)


def _repair_completed_s10_downstream_resume(state: StateStore) -> None:
    """Apply the bundled-segment resume fix to an already reconciled run."""

    if not state.selection("formal_s10_reconciliation_complete", False):
        return
    if state.selection("formal_s10_downstream_resume_version", 0) >= 2:
        return
    e4_profile_retries = state.retry_failed("E4-profile")
    width_retries = state.retry_failed("E3-width-calibration")
    width_segment_retries = state.retry_failed("E3-width-calibration-segments")
    reopened = state.reopen_skipped(
        (
            "E3b-pilot",
            "E3b-final",
            "E1a",
            "E5-pilot",
            "E5-final",
            "E6-pilot",
            "E6-final",
            "E0-tune",
            "E0-pilot",
            "E0-final",
        )
    )
    audit_path = (
        state.run_dir
        / "stages"
        / "S10-reconciliation"
        / "downstream-resume-v2.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        audit_path,
        {
            "e4_profile_retries": e4_profile_retries,
            "width_calibration_retries": width_retries,
            "width_calibration_segment_retries": width_segment_retries,
            "future_jobs_reopened": reopened,
        },
    )
    state.set_selection("formal_s10_downstream_resume_version", 2)


def _run_recipe_change_replacements(
    config: ExperimentConfig,
    state: StateStore,
    stop_event: threading.Event,
) -> int:
    """Replace only completed evidence whose runtime recipe is selected globally."""

    exclusions = set(state.selection("formal_evidence_exclusions", []))

    def replacement(source: Job, *, ordinal: int, group: str) -> Job:
        exclusions.add(source.job_id)
        return replace(
            source,
            job_id=f"bugfix-recipe-v1__{source.job_id}",
            node=group,
            ordinal=ordinal,
            parameters={
                **source.parameters,
                "source_node": source.node,
                "replaces_job_id": source.job_id,
                "reconciliation_kind": "changed_lightcone_recipe",
            },
        )

    screen_sources = tuple(
        job for job in state.jobs("E4-screen") if job.method == "lightcone"
    )
    if len(screen_sources) != 48:
        raise ScientificFailure(
            f"recipe closure expected 48 E4-screen rows, found {len(screen_sources)}"
        )
    screen = tuple(
        replacement(job, ordinal=index, group="bugfix-recipe-v1-screen")
        for index, job in enumerate(screen_sources)
    )
    state.set_selection("formal_evidence_exclusions", sorted(exclusions))
    state.add_internal_jobs(screen, storage_node="bugfix-recipe-v1-screen")
    _run_pending_jobs(
        config,
        state,
        "bugfix-recipe-v1-screen",
        stop_event,
        state.pending_jobs("bugfix-recipe-v1-screen"),
    )
    if stop_event.is_set():
        return 0
    _require_internal_jobs(state, "bugfix-recipe-v1-screen")
    _reduce_node(config, state, "E4-screen")

    local_sources = materialize(
        "E4-local",
        e4_neighborhoods=state.selection("e4_neighborhoods", None),
    )
    old_local = state.jobs("E4-local")
    if len(local_sources) != 168 or len(old_local) != 168:
        raise ScientificFailure("recipe closure expected 168 E4-local rows")
    exclusions.update(job.job_id for job in old_local)
    local = tuple(
        replacement(job, ordinal=index, group="bugfix-recipe-v1-local")
        for index, job in enumerate(local_sources)
    )

    profile_sources = tuple(
        job
        for job in state.jobs("E4-profile")
        if job.parameters.get("profiler") != "ncu"
    )
    if len(profile_sources) != 2:
        raise ScientificFailure(
            f"recipe closure expected 2 non-NCU profile rows, found {len(profile_sources)}"
        )
    profile = tuple(
        replacement(job, ordinal=index, group="bugfix-recipe-v1-profile")
        for index, job in enumerate(profile_sources)
    )

    width_sources = tuple(
        job for job in _deployment_width_jobs(state) if job.method == "lightcone"
    )
    if len(width_sources) != 3:
        raise ScientificFailure("recipe closure expected three LightCone width parents")
    width = tuple(
        replacement(job, ordinal=index, group="bugfix-recipe-v1-width")
        for index, job in enumerate(width_sources)
    )
    state.set_selection("formal_evidence_exclusions", sorted(exclusions))

    groups = (
        ("bugfix-recipe-v1-local", local),
        ("bugfix-recipe-v1-profile", profile),
        ("bugfix-recipe-v1-width", width),
    )
    for group, jobs in groups:
        state.add_internal_jobs(jobs, storage_node=group)
        _run_pending_jobs(config, state, group, stop_event, state.pending_jobs(group))
        if stop_event.is_set():
            return 0
        _require_internal_jobs(state, group)
    widths = _select_deployment_widths(state)
    state.set_selection("deployment_widths", widths)
    state.set_selection("deployment_widths_tuned", True)
    return 48 + 168 + 2 + 9


def _run_bugfix_reconciliation_v1(
    config: ExperimentConfig,
    state: StateStore,
    stop_event: threading.Event,
) -> None:
    """Repair the E1a/E2/screening/width evidence without rewriting raw attempts."""

    if state.selection("formal_bugfix_reconciliation_version", 0) >= 1:
        return
    if state.selection("formal_s10_downstream_resume_version", 0) < 2:
        return
    confidence_jobs = tuple(
        job
        for job in state.jobs("E1a")
        if job.parameters.get("workload") == "confidence_calibration"
    )
    if len(confidence_jobs) != 5:
        return

    old_e3a = state.selection("e3a", None)
    old_recipe = state.selection("lightcone_recipe", None)
    repairs = _bugfix_reconciliation_jobs(state)
    exclusions = set(state.selection("formal_evidence_exclusions", []))
    exclusions.update(str(job.parameters["replaces_job_id"]) for job in repairs)
    state.set_selection("formal_evidence_exclusions", sorted(exclusions))
    state.add_internal_jobs(repairs, storage_node="bugfix-reconciliation-v1")
    _run_pending_jobs(
        config,
        state,
        "bugfix-reconciliation-v1",
        stop_event,
        state.pending_jobs("bugfix-reconciliation-v1"),
    )
    if stop_event.is_set():
        return
    _require_internal_jobs(state, "bugfix-reconciliation-v1")

    width_segment_retries = int(state.selection("bugfix_v1_width_retries", 0))
    if not width_segment_retries:
        width_segment_retries = state.retry_failed("E3-width-calibration-segments")
        if width_segment_retries != 7:
            raise ScientificFailure(
                f"bugfix reconciliation expected 7 width retries, found {width_segment_retries}"
            )
        state.set_selection("bugfix_v1_width_retries", width_segment_retries)
    _run_pending_jobs(
        config,
        state,
        "E3-width-calibration",
        stop_event,
        state.pending_jobs("E3-width-calibration-segments"),
    )
    if stop_event.is_set():
        return
    _require_internal_jobs(state, "E3-width-calibration-segments")
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

    _reduce_node(config, state, "E3a")
    new_e3a = state.selection("e3a", None)
    if new_e3a != old_e3a:
        raise ScientificFailure(
            f"E3a selection changed from {old_e3a} to {new_e3a}; broader closure required"
        )

    _audit_e2_after_s10(config, state, stop_event)
    if stop_event.is_set():
        return
    new_recipe = state.selection("lightcone_recipe", None)
    recipe_changed = (
        isinstance(old_recipe, dict)
        and isinstance(new_recipe, dict)
        and _e2_recipe_key(old_recipe) != _e2_recipe_key(new_recipe)
    )
    conditional_cells = (
        _run_recipe_change_replacements(config, state, stop_event)
        if recipe_changed
        else 0
    )
    if stop_event.is_set():
        return

    reopened = state.reopen_skipped(
        ("E3b-pilot", "E3b-final", "E1a", "E5-pilot", "E5-final")
    )
    audit_path = (
        state.run_dir
        / "stages"
        / "bugfix-reconciliation-v1"
        / "reconciliation.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        audit_path,
        {
            "direct_replacement_jobs": len(repairs),
            "direct_gpu_cells": 145,
            "width_retry_cells": width_segment_retries,
            "total_direct_gpu_cells": 145 + width_segment_retries,
            "e3a_selection_unchanged": new_e3a == old_e3a,
            "lightcone_recipe_changed": recipe_changed,
            "conditional_recipe_cells": conditional_cells,
            "downstream_jobs_reopened": reopened,
        },
    )
    state.set_selection("formal_bugfix_reconciliation_version", 1)


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
                _run_formal_s10_reconciliation(
                    self.config,
                    self.state,
                    self.stop_event,
                )
                _repair_completed_s10_downstream_resume(self.state)
                _run_bugfix_reconciliation_v1(
                    self.config,
                    self.state,
                    self.stop_event,
                )
                if self.stop_event.is_set():
                    break
                valid_e0 = self.state.selection("valid_e0", None)
                e2_rows = None
                if node == "E2-r0":
                    e2_rows = e2_candidates(self.state.selection("e1_geometries", None))
                elif node.startswith("E2-r"):
                    e2_rows = self.state.selection(f"e2_round_{int(node[-1]) - 1}", None)
                if node == "E0-tune" and valid_e0 is None:
                    probe_jobs = tuple(
                        job for job in materialize(node) if job.parameters.get("probe")
                    )
                    self.state.add_jobs(node, probe_jobs)
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
                        self.state.finish_stage(node)
                        self.state.set_selection(f"{node}_failed", True)
                        continue
                    valid_e0 = _select_valid_e0(self.state)
                    self.state.set_selection("valid_e0", valid_e0)
                jobs = _resume_materialization(
                    self.state,
                    node,
                    materialize(
                        node,
                        valid_e0=valid_e0,
                        e2_rows=e2_rows,
                        e0_recipes=self.state.selection("e0_recipes", None),
                        e4_neighborhoods=self.state.selection(
                            "e4_neighborhoods", None
                        ),
                    ),
                )
                upgraded = (
                    _upgrade_legacy_e0_materialization(self.state, jobs)
                    if node == "E0-tune"
                    else None
                )
                if upgraded is None:
                    self.state.add_jobs(node, jobs)
                else:
                    jobs = upgraded
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
