"""GPU-ready confirmation driver with independent, attested measurements."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lightcone_spec.experiments.data import (
    DFLASH_SAFE_CONTEXT_LIMIT,
    GENERATED_TOKEN_BUCKETS,
    LongContinuationAdapter,
    PromptSample,
    sample_set_sha256,
)
from lightcone_spec.experiments.evidence import evidence_files_sha256
from lightcone_spec.experiments.protocol import paired_blocks
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.selection import LossPoint, SliceMeasurement
from lightcone_spec.sglang_bridge.client import (
    GenerationResult,
    MethodRun,
    ServerSnapshot,
    SGLangHTTPClient,
    independent_method_run,
)
from lightcone_spec.telemetry import (
    EvidenceWriter,
    PerformanceRecord,
    RequestRecord,
    RoundRecord,
    RunRecord,
    UpdateRecord,
    load_completed_evidence,
)

_FORMAL_METHODS = {"static", "tts", "naive_async"}
_ONLINE_SPEC_METHODS = {
    "static",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
}
_EVIDENCE_METHODS = _FORMAL_METHODS | _ONLINE_SPEC_METHODS
_SAFETY_COUNTERS = (
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
)
_UPDATE_COUNTERS = ("updates_launched", "updates_published")
_TIMING_LANES = ("training", "optimizer", "merge", "publish", "barrier")


def _run_id(
    manifest_sha256: str,
    block: int,
    prompt_id: str,
    method: str,
    namespace: str = "confirmation",
) -> str:
    if not namespace:
        raise ValueError("run namespace must be non-empty")
    value = f"{namespace}:{manifest_sha256}:{block}:{prompt_id}:{method}"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _output_sha256(result: GenerationResult) -> str:
    text = result.response.get("text")
    if not isinstance(text, str):
        raise TypeError("final SGLang response lacks generated text for exactness")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _output_set_sha256(rows: list[tuple[str, tuple[str, ...]]]) -> str:
    body = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(body).hexdigest()


def _payloads(
    sample: PromptSample,
    *,
    method: str,
    block: int,
    concurrency: int,
    max_new_tokens: int,
    sampling_profile: SamplingProfile,
    request_namespace: str | None = None,
) -> tuple[dict, ...]:
    if method not in _EVIDENCE_METHODS:
        raise ValueError("unknown measured method")
    if request_namespace is not None and not request_namespace:
        raise ValueError("request namespace must be non-empty when provided")
    namespace = f"-{request_namespace}" if request_namespace is not None else ""
    return tuple(
        {
            "rid": (
                f"{sample.sample_id}{namespace}-b{block}-{method}-r{replica}"
            ),
            "text": sample.prompt,
            "sampling_params": sampling_profile.parameters(
                seed=sample.seed + replica,
                max_new_tokens=max_new_tokens,
            ),
        }
        for replica in range(concurrency)
    )


def _batch_prompt_id(samples: tuple[PromptSample, ...]) -> str:
    """Return the evidence identity for one jointly timed prompt batch."""
    if not samples:
        raise ValueError("a measured prompt batch cannot be empty")
    identifiers = tuple(sample.sample_id for sample in samples)
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("measured prompt IDs must be non-empty and unique")
    return f"batch-{sample_set_sha256(samples)[:24]}"


def _batched_payloads(
    samples: tuple[PromptSample, ...],
    *,
    budgets: dict[str, tuple[int, int]],
    method: str,
    block: int,
    concurrency: int,
    sampling_profile: SamplingProfile,
    fill_concurrency: bool,
) -> tuple[tuple[dict, ...], dict[str, tuple[PromptSample, int, int]]]:
    """Build one queue of distinct prompts, filling only undersized screens.

    Confirmation and natural-task runs submit each prompt exactly once. Load
    and early tuning stages may contain fewer prompts than the registered
    concurrency, so those stages repeat the complete prompt set round-robin
    only until the selected load is full. The returned request map binds every
    response to its prompt and context-safe budget without parsing request IDs.
    """
    if method not in _EVIDENCE_METHODS:
        raise ValueError("unknown measured method")
    if concurrency < 1:
        raise ValueError("batch concurrency must be positive")
    _batch_prompt_id(samples)
    expected_ids = {sample.sample_id for sample in samples}
    if set(budgets) != expected_ids:
        raise ValueError("prompt budgets do not match the measured batch")
    request_count = max(len(samples), concurrency) if fill_concurrency else len(samples)
    payloads: list[dict] = []
    assignments: dict[str, tuple[PromptSample, int, int]] = {}
    for index in range(request_count):
        sample = samples[index % len(samples)]
        replica = index // len(samples)
        expected_input_tokens, max_new_tokens = budgets[sample.sample_id]
        request_id = f"{sample.sample_id}-b{block}-{method}-r{replica}"
        if request_id in assignments:
            raise AssertionError("batched request identities are not unique")
        payloads.append(
            {
                "rid": request_id,
                "text": sample.prompt,
                "sampling_params": sampling_profile.parameters(
                    seed=sample.seed + replica,
                    max_new_tokens=max_new_tokens,
                ),
            }
        )
        assignments[request_id] = (
            sample,
            expected_input_tokens,
            max_new_tokens,
        )
    return tuple(payloads), assignments


def _group_results(
    results: tuple[GenerationResult, ...],
    assignments: dict[str, tuple[PromptSample, int, int]],
) -> dict[str, tuple[GenerationResult, ...]]:
    """Validate result coverage and group replicas by original prompt."""
    if len(results) != len(assignments):
        raise RuntimeError("measured batch result coverage is incomplete")
    observed = {result.request_id for result in results}
    if len(observed) != len(results) or observed != set(assignments):
        raise RuntimeError("measured batch returned unknown or duplicate requests")
    grouped: dict[str, list[GenerationResult]] = {}
    for result in results:
        sample, _, _ = assignments[result.request_id]
        grouped.setdefault(sample.sample_id, []).append(result)
    return {
        prompt_id: tuple(sorted(rows, key=lambda row: row.request_id))
        for prompt_id, rows in grouped.items()
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _region(
    results: tuple[GenerationResult, ...],
    *,
    start: int,
    end: int,
) -> tuple[int, int, float, list[float]] | None:
    if start < 0 or end <= start:
        raise ValueError("token region must be a non-empty half-open interval")
    at_risk = tuple(result for result in results if result.completion_tokens > start)
    if not at_risk:
        return None
    output_tokens = 0
    spans: list[tuple[float, float]] = []
    intervals: list[float] = []
    for result in at_risk:
        stop = min(result.completion_tokens, end)
        count = stop - start
        if count < 1:
            continue
        output_tokens += count
        # inter_token_ms[i] is the transition from token i to token i + 1.
        request_intervals = result.inter_token_ms[start : stop - 1]
        intervals.extend(request_intervals)
        if request_intervals:
            spans.append(
                (
                    result.token_arrival_ms[start],
                    result.token_arrival_ms[stop - 1],
                )
            )
    if output_tokens < 1 or not spans or not intervals:
        return None
    spans.sort()
    active_ms = 0.0
    active_start, active_end = spans[0]
    for span_start, span_end in spans[1:]:
        if span_start > active_end:
            active_ms += active_end - active_start
            active_start, active_end = span_start, span_end
        else:
            active_end = max(active_end, span_end)
    active_ms += active_end - active_start
    elapsed_s = active_ms / 1000.0
    if elapsed_s <= 0:
        raise RuntimeError(
            "decode region has no measurable arrival interval; increase its size"
        )
    return len(at_risk), output_tokens, elapsed_s, intervals


def _prompt_budgets(
    client: SGLangHTTPClient,
    samples: tuple[PromptSample, ...],
    *,
    safe_context_limit: int,
    minimum_generation_tokens: int = 1,
) -> dict[str, tuple[int, int]]:
    if minimum_generation_tokens < 1:
        raise ValueError("minimum generation budget must be positive")
    prompts = tuple(sample.prompt for sample in samples)
    counts, reported_limit = client.tokenize_prompts(prompts)
    if reported_limit < safe_context_limit:
        raise RuntimeError(
            f"tokenizer limit {reported_limit} is below the registered safe "
            f"limit {safe_context_limit}"
        )
    budgets = {
        sample.sample_id: (count, safe_context_limit - count)
        for sample, count in zip(samples, counts, strict=True)
    }
    if any(
        generation < minimum_generation_tokens for _, generation in budgets.values()
    ):
        raise RuntimeError(
            "controlled prompts leave insufficient context-safe generation budget"
        )
    return budgets


def _adaptation_fields(
    method: str,
    snapshot: ServerSnapshot,
    expected_adaptation_sha256: str | None,
) -> tuple[dict, tuple[dict, ...], tuple[dict, ...]]:
    if method == "static":
        if expected_adaptation_sha256 is not None:
            raise RuntimeError("Static cannot have an adaptation config identity")
        if snapshot.adaptation is not None:
            raise RuntimeError("Static returned adaptation diagnostics")
        return (
            {
                "optimizer_bytes": 0,
                "adaptation_memory_ledger": None,
                "trainable_parameters": 0,
                "training_cuda_ms": None,
                "optimizer_cuda_ms": None,
                "merge_cuda_ms": None,
                "publish_cuda_ms": None,
                "barrier_cuda_ms": None,
                "exposed_update_ms": None,
                "main_side_overlap_ratio": None,
                **{field: 0 for field in _UPDATE_COUNTERS},
                **{field: 0 for field in _SAFETY_COUNTERS},
                "oom_events": snapshot.oom_events,
                "retractions": snapshot.retractions,
            },
            (),
            (),
        )
    if (
        expected_adaptation_sha256 is None
        or len(expected_adaptation_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_adaptation_sha256)
    ):
        raise RuntimeError("adapted run lacks an expected config identity")
    diagnostics = snapshot.adaptation
    if not isinstance(diagnostics, dict) or diagnostics.get("schema_version") != 2:
        raise RuntimeError("adapted run lacks schema-v2 diagnostics")
    counters = diagnostics.get("counters")
    timings = diagnostics.get("timings_ms")
    updates = diagnostics.get("updates")
    rounds = diagnostics.get("rounds")
    if not isinstance(counters, dict) or not isinstance(timings, dict):
        raise RuntimeError(  # noqa: TRY004 - malformed remote evidence
            "adaptation counters or timings are missing"
        )
    if not isinstance(updates, list):
        raise RuntimeError(  # noqa: TRY004 - malformed remote evidence
            "adaptation update evidence is missing"
        )
    if not isinstance(rounds, list) or not rounds:
        raise RuntimeError("adaptation round evidence is missing")
    missing_counters = set(_SAFETY_COUNTERS + _UPDATE_COUNTERS) - set(counters)
    missing_timings = set(_TIMING_LANES) - set(timings)
    if missing_counters or missing_timings:
        raise RuntimeError(
            "adaptation evidence is incomplete: "
            f"counters={sorted(missing_counters)}, "
            f"timings={sorted(missing_timings)}"
        )
    if int(counters.get("target_calls", -1)) != snapshot.target_calls:
        raise RuntimeError("scheduler and adaptation target-call counts disagree")
    required = {
        "adaptation_config_sha256",
        "cohort_sha256",
        "kv_segments",
        "parameter_layout_sha256",
        "optimizer_bytes",
        "trainable_parameters",
        "memory_ledger",
        "resident_bytes",
        "peak_bytes",
        "exposed_update_ms",
        "main_side_overlap_ratio",
    }
    if not required <= set(diagnostics):
        raise RuntimeError("adaptation memory or overlap evidence is incomplete")
    if diagnostics["adaptation_config_sha256"] != expected_adaptation_sha256:
        raise RuntimeError("runtime adaptation config identity mismatch")
    if not isinstance(diagnostics["kv_segments"], dict):
        raise RuntimeError(  # noqa: TRY004 - malformed remote evidence
            "adaptation KV-version evidence is malformed"
        )
    layout = str(diagnostics["parameter_layout_sha256"])
    if len(layout) != 64 or any(char not in "0123456789abcdef" for char in layout):
        raise RuntimeError("adaptation parameter layout identity is invalid")
    cohort = str(diagnostics["cohort_sha256"])
    if len(cohort) != 64 or any(char not in "0123456789abcdef" for char in cohort):
        raise RuntimeError("adaptation cohort identity is invalid")
    ledger = diagnostics["memory_ledger"]
    ledger_keys = {
        "active_or_base_bytes",
        "master_fp32_bytes",
        "gradient_bytes",
        "first_moment_bytes",
        "second_moment_bytes",
        "online_state_bytes",
        "optimizer_metadata_bytes",
        "staging_bytes",
        "training_activation_bytes",
        "kv_gather_scratch_bytes",
        "candidate_scratch_bytes",
        "graph_buffer_bytes",
        "telemetry_bytes",
        "resident_bytes",
        "optimizer_bytes",
        "peak_bytes",
    }
    if (
        not isinstance(ledger, dict)
        or set(ledger) != ledger_keys
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in ledger.values()
        )
    ):
        raise RuntimeError("adaptation memory ledger is incomplete")
    if (
        ledger["resident_bytes"] != int(diagnostics["resident_bytes"])
        or ledger["peak_bytes"] != int(diagnostics["peak_bytes"])
        or ledger["optimizer_bytes"] != int(diagnostics["optimizer_bytes"])
    ):
        raise RuntimeError("adaptation memory ledger totals disagree")
    resident_categories = sum(
        ledger[name]
        for name in (
            "active_or_base_bytes",
            "master_fp32_bytes",
            "first_moment_bytes",
            "second_moment_bytes",
            "online_state_bytes",
            "optimizer_metadata_bytes",
            "staging_bytes",
            "graph_buffer_bytes",
            "telemetry_bytes",
        )
    )
    scratch_categories = sum(
        ledger[name]
        for name in (
            "gradient_bytes",
            "training_activation_bytes",
            "kv_gather_scratch_bytes",
            "candidate_scratch_bytes",
        )
    )
    optimizer_categories = sum(
        ledger[name]
        for name in (
            "master_fp32_bytes",
            "first_moment_bytes",
            "second_moment_bytes",
            "online_state_bytes",
            "optimizer_metadata_bytes",
        )
    )
    if (
        resident_categories != ledger["resident_bytes"]
        or resident_categories + scratch_categories != ledger["peak_bytes"]
        or optimizer_categories != ledger["optimizer_bytes"]
    ):
        raise RuntimeError("adaptation memory ledger categories do not sum")
    fields = {
        "optimizer_bytes": int(diagnostics["optimizer_bytes"]),
        "adaptation_memory_ledger": json.dumps(
            ledger, sort_keys=True, separators=(",", ":")
        ),
        "trainable_parameters": int(diagnostics["trainable_parameters"]),
        **{f"{lane}_cuda_ms": float(timings[lane]) for lane in _TIMING_LANES},
        "exposed_update_ms": float(diagnostics["exposed_update_ms"]),
        "main_side_overlap_ratio": float(diagnostics["main_side_overlap_ratio"]),
        **{field: int(counters[field]) for field in _SAFETY_COUNTERS},
        **{field: int(counters[field]) for field in _UPDATE_COUNTERS},
    }
    fields["oom_events"] += snapshot.oom_events
    fields["retractions"] = max(fields["retractions"], snapshot.retractions)
    numeric_fields = tuple(
        float(value)
        for key, value in fields.items()
        if key not in {"main_side_overlap_ratio", "adaptation_memory_ledger"}
        and value is not None
    )
    if not all(math.isfinite(value) and value >= 0 for value in numeric_fields):
        raise RuntimeError(
            "adaptation counters and timings must be finite and non-negative"
        )
    overlap = float(fields["main_side_overlap_ratio"])
    if not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
        raise RuntimeError("adaptation overlap ratio is outside [0, 1]")
    if int(fields["updates_launched"]) != len(updates):
        raise RuntimeError("update counter and trace coverage disagree")
    published = sum(update.get("status") == "published" for update in updates)
    if not published <= int(fields["updates_published"]) <= len(updates):
        raise RuntimeError("published update counter and trace coverage disagree")
    return fields, tuple(updates), tuple(rounds)


def _performance_record(
    *,
    run_id: str,
    prompt_id: str,
    method: str,
    block: int,
    concurrency: int,
    region_name: str,
    region_start: int,
    region_end: int,
    results: tuple[GenerationResult, ...],
    snapshot: ServerSnapshot,
    adaptation: dict,
    run_scope_metrics: bool,
) -> PerformanceRecord | None:
    measured = _region(results, start=region_start, end=region_end)
    if measured is None:
        return None
    at_risk, output_tokens, elapsed_s, intervals = measured
    target_calls = snapshot.target_calls if run_scope_metrics else None
    accepted = snapshot.accepted_drafts if run_scope_metrics else None
    committed = snapshot.committed_tokens if run_scope_metrics else None
    verified = snapshot.verified_drafts if run_scope_metrics else None
    if target_calls is not None and target_calls < 1:
        raise RuntimeError("run-scope performance requires target calls")
    return PerformanceRecord(
        run_id=run_id,
        prompt_id=prompt_id,
        method=method,
        repetition_block=block,
        region=region_name,
        concurrency=concurrency,
        generated_bucket_start=region_start,
        generated_bucket_end=region_end,
        at_risk_requests=at_risk,
        output_tokens=output_tokens,
        elapsed_s=elapsed_s,
        decode_goodput_tps=len(intervals) / elapsed_s,
        itl_p50_ms=_percentile(intervals, 0.50),
        itl_p95_ms=_percentile(intervals, 0.95),
        itl_p99_ms=_percentile(intervals, 0.99),
        survival_weighted_accepted_prefix=(
            accepted / target_calls if run_scope_metrics else None
        ),
        accepted_drafts_per_verify=(
            accepted / target_calls if run_scope_metrics else None
        ),
        committed_tokens_per_verify=(
            committed / target_calls if run_scope_metrics else None
        ),
        verified_drafts_per_verify=(
            verified / target_calls if run_scope_metrics else None
        ),
        verification_waste=(
            snapshot.verification_waste / verified
            if run_scope_metrics and verified
            else (0.0 if run_scope_metrics else None)
        ),
        target_calls_per_output_token=(
            target_calls / output_tokens if run_scope_metrics else None
        ),
        batch_fill=snapshot.batch_fill if run_scope_metrics else None,
        queue_occupancy=(snapshot.queue_occupancy if run_scope_metrics else None),
        gpu_busy=None,
        sm_utilization=None,
        dram_utilization=None,
        target_estimated_mfu=None,
        peak_hbm_bytes=(snapshot.peak_hbm_bytes if run_scope_metrics else None),
        kv_bytes=(snapshot.kv_bytes if run_scope_metrics else None),
        optimizer_bytes=(
            int(adaptation["optimizer_bytes"]) if run_scope_metrics else None
        ),
        adaptation_memory_ledger=(
            adaptation["adaptation_memory_ledger"] if run_scope_metrics else None
        ),
        trainable_parameters=(
            int(adaptation["trainable_parameters"]) if run_scope_metrics else None
        ),
        training_cuda_ms=(adaptation["training_cuda_ms"] if run_scope_metrics else None),
        optimizer_cuda_ms=(
            adaptation["optimizer_cuda_ms"] if run_scope_metrics else None
        ),
        merge_cuda_ms=(adaptation["merge_cuda_ms"] if run_scope_metrics else None),
        publish_cuda_ms=(
            adaptation["publish_cuda_ms"] if run_scope_metrics else None
        ),
        barrier_cuda_ms=(
            adaptation["barrier_cuda_ms"] if run_scope_metrics else None
        ),
        exposed_update_ms=(
            adaptation["exposed_update_ms"] if run_scope_metrics else None
        ),
        main_side_overlap_ratio=(
            adaptation["main_side_overlap_ratio"] if run_scope_metrics else None
        ),
        graph_replay_hit_rate=(
            snapshot.graph_replay_hit_rate if run_scope_metrics else None
        ),
        updates_launched=(
            int(adaptation["updates_launched"]) if run_scope_metrics else None
        ),
        updates_published=(
            int(adaptation["updates_published"]) if run_scope_metrics else None
        ),
        exactness_violations=(
            int(adaptation["exactness_violations"]) if run_scope_metrics else None
        ),
        version_mismatches=(
            int(adaptation["version_mismatches"]) if run_scope_metrics else None
        ),
        fallbacks=(int(adaptation["fallbacks"]) if run_scope_metrics else None),
        nonfinite_updates=(
            int(adaptation["nonfinite_updates"]) if run_scope_metrics else None
        ),
        oom_events=(int(adaptation["oom_events"]) if run_scope_metrics else None),
        retractions=(
            int(adaptation["retractions"]) if run_scope_metrics else None
        ),
    )


def _write_updates(
    writer: EvidenceWriter,
    *,
    run_id: str,
    method: str,
    diagnostics: dict | None,
    updates: tuple[dict, ...],
) -> None:
    if method not in _EVIDENCE_METHODS:
        raise RuntimeError("update trace belongs to an unknown method")
    if method == "static":
        if diagnostics is not None or updates:
            raise RuntimeError("Static cannot contain update evidence")
        return
    if diagnostics is None or not updates:
        raise RuntimeError("adapted method requires non-empty update evidence")
    cohort = str(diagnostics["cohort_sha256"])
    layout = str(diagnostics["parameter_layout_sha256"])
    trainable = int(diagnostics["trainable_parameters"])
    for index, update in enumerate(updates):
        required = {
            "source_round",
            "source_version",
            "optimizer_step",
            "published_version",
            "status",
            "loss",
            "gradient_norm",
            "reconstruction_ok",
            "reconstruction_max_abs",
            "supervision_nonempty",
        }
        if not isinstance(update, dict) or not required <= set(update):
            raise RuntimeError("update trace is incomplete")
        request_ids = update.get("request_ids")
        prefix_lens = update.get("prefix_len_before")
        if (
            not isinstance(request_ids, list)
            or not request_ids
            or any(not isinstance(value, str) or not value for value in request_ids)
            or not isinstance(prefix_lens, list)
            or len(prefix_lens) != len(request_ids)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in prefix_lens
            )
        ):
            raise RuntimeError("update trace lacks request-level prefix lengths")
        loss = float(update["loss"])
        if not math.isfinite(loss) or loss < -1e-6:
            raise RuntimeError("update trace loss must be finite and non-negative")
        optimizer_step = update["optimizer_step"]
        if (
            not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or optimizer_step < 1
        ):
            raise RuntimeError("update trace optimizer step must be positive")
        gradient_norm = float(update["gradient_norm"])
        if not math.isfinite(gradient_norm) or gradient_norm < 0:
            raise RuntimeError(
                "update trace gradient norm must be finite and non-negative"
            )
        reconstruction_ok = update["reconstruction_ok"]
        supervision_nonempty = update["supervision_nonempty"]
        if not isinstance(reconstruction_ok, bool) or not isinstance(
            supervision_nonempty, bool
        ):
            raise TypeError("update reconstruction flags must be boolean")
        reconstruction_max_abs = float(update["reconstruction_max_abs"])
        if not math.isfinite(reconstruction_max_abs) or reconstruction_max_abs < 0:
            raise RuntimeError(
                "update reconstruction error must be finite and non-negative"
            )
        optional_reconstruction = {
            name: update.get(name)
            for name in (
                "reconstruction_relative_rms",
                "reconstruction_top1_match",
                "reconstruction_mean_kl",
            )
        }
        if any(
            value is not None and (not math.isfinite(float(value)) or float(value) < 0)
            for value in optional_reconstruction.values()
        ) or (
            optional_reconstruction["reconstruction_top1_match"] is not None
            and float(optional_reconstruction["reconstruction_top1_match"]) > 1
        ):
            raise RuntimeError("update reconstruction diagnostics are invalid")
        online_scalars = {
            name: update.get(name)
            for name in (
                "online_hint_error",
                "online_ensemble_entropy",
                "online_effective_experts",
            )
        }
        if any(
            value is not None and (not math.isfinite(float(value)) or float(value) < 0)
            for value in online_scalars.values()
        ):
            raise RuntimeError("OnlineSPEC update diagnostics are invalid")
        probabilities = update.get("online_expert_probabilities")
        cumulative_losses = update.get("online_cumulative_losses")
        expert_gradient_norms = update.get("online_expert_gradient_norms")
        if (
            len(
                {
                    probabilities is None,
                    cumulative_losses is None,
                    expert_gradient_norms is None,
                }
            )
            != 1
        ):
            raise RuntimeError("OnlineSPEC ensemble diagnostics are incomplete")
        if probabilities is not None and (
            not isinstance(probabilities, list)
            or not isinstance(cumulative_losses, list)
            or not isinstance(expert_gradient_norms, list)
            or len(probabilities) < 2
            or len(probabilities) != len(cumulative_losses)
            or len(probabilities) != len(expert_gradient_norms)
            or not all(
                math.isfinite(float(value)) and float(value) >= 0
                for value in (
                    *probabilities,
                    *cumulative_losses,
                    *expert_gradient_norms,
                )
            )
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
        ):
            raise RuntimeError("OnlineSPEC ensemble evidence is invalid")
        online_values = (
            *online_scalars.values(),
            probabilities,
            cumulative_losses,
            expert_gradient_norms,
        )
        if method == "onlinespec_opt":
            if online_scalars["online_hint_error"] is None or any(
                value is not None
                for value in (
                    online_scalars["online_ensemble_entropy"],
                    online_scalars["online_effective_experts"],
                    probabilities,
                    cumulative_losses,
                    expert_gradient_norms,
                )
            ):
                raise RuntimeError("optimistic OnlineSPEC diagnostics are incomplete")
        elif method == "onlinespec_ens":
            if (
                online_scalars["online_hint_error"] is not None
                or online_scalars["online_ensemble_entropy"] is None
                or online_scalars["online_effective_experts"] is None
                or probabilities is None
            ):
                raise RuntimeError("OnlineSPEC ensemble diagnostics are incomplete")
        elif method in {"tts", "naive_async", "onlinespec_ogd"} and any(
            value is not None for value in online_values
        ):
            raise RuntimeError("method emitted foreign OnlineSPEC diagnostics")
        writer.write(
            UpdateRecord(
                run_id=run_id,
                cohort_sha256=cohort,
                parameter_layout_sha256=layout,
                update_index=index,
                request_ids=json.dumps(request_ids, separators=(",", ":")),
                prefix_len_before=json.dumps(prefix_lens, separators=(",", ":")),
                prefix_len_min=min(prefix_lens),
                prefix_len_max=max(prefix_lens),
                prefix_len_mean=sum(prefix_lens) / len(prefix_lens),
                source_round=int(update["source_round"]),
                source_version=int(update["source_version"]),
                optimizer_step=optimizer_step,
                published_version=(
                    None
                    if update["published_version"] is None
                    else int(update["published_version"])
                ),
                candidate_status=str(update["status"]),
                loss=loss,
                gradient_norm=gradient_norm,
                reconstruction_ok=reconstruction_ok,
                reconstruction_max_abs=reconstruction_max_abs,
                reconstruction_relative_rms=(
                    None
                    if optional_reconstruction["reconstruction_relative_rms"] is None
                    else float(optional_reconstruction["reconstruction_relative_rms"])
                ),
                reconstruction_top1_match=(
                    None
                    if optional_reconstruction["reconstruction_top1_match"] is None
                    else float(optional_reconstruction["reconstruction_top1_match"])
                ),
                reconstruction_mean_kl=(
                    None
                    if optional_reconstruction["reconstruction_mean_kl"] is None
                    else float(optional_reconstruction["reconstruction_mean_kl"])
                ),
                supervision_nonempty=supervision_nonempty,
                trainable_parameters=trainable,
                training_cuda_ms=None,
                optimizer_cuda_ms=None,
                merge_cuda_ms=None,
                publish_cuda_ms=None,
                barrier_cuda_ms=None,
                exposed_update_ms=None,
                overlap_ratio=None,
                online_hint_error=(
                    None
                    if online_scalars["online_hint_error"] is None
                    else float(online_scalars["online_hint_error"])
                ),
                online_ensemble_entropy=(
                    None
                    if online_scalars["online_ensemble_entropy"] is None
                    else float(online_scalars["online_ensemble_entropy"])
                ),
                online_effective_experts=(
                    None
                    if online_scalars["online_effective_experts"] is None
                    else float(online_scalars["online_effective_experts"])
                ),
                online_expert_probabilities=(
                    None
                    if probabilities is None
                    else json.dumps(probabilities, separators=(",", ":"))
                ),
                online_cumulative_losses=(
                    None
                    if cumulative_losses is None
                    else json.dumps(cumulative_losses, separators=(",", ":"))
                ),
                online_expert_gradient_norms=(
                    None
                    if expert_gradient_norms is None
                    else json.dumps(expert_gradient_norms, separators=(",", ":"))
                ),
            )
        )


def _round_records(
    *,
    run_id: str,
    diagnostics: dict | None,
    results: tuple[GenerationResult, ...],
    rounds: tuple[dict, ...],
) -> tuple[RoundRecord, ...]:
    if diagnostics is None:
        if rounds:
            raise RuntimeError("Static cannot contain round adaptation evidence")
        return ()
    inputs = {result.request_id: result.input_tokens for result in results}
    completions = {result.request_id: result.completion_tokens for result in results}
    if len(inputs) != len(results):
        raise RuntimeError("request identities are not unique within a run")
    histories: dict[str, list[dict[str, int]]] = {}
    flat: list[tuple[int, int, str, int, int, int, int]] = []
    seen_rounds: set[int] = set()
    for trace in rounds:
        required = {
            "round_index",
            "source_version",
            "request_ids",
            "prefix_len_before",
            "verify_len",
            "accepted_drafts",
            "committed_tokens",
        }
        if not isinstance(trace, dict) or set(trace) != required:
            raise RuntimeError("round trace fields are incomplete")
        round_index = int(trace["round_index"])
        source_version = int(trace["source_version"])
        if round_index < 1 or source_version < 0 or round_index in seen_rounds:
            raise RuntimeError("round trace identity is invalid or duplicated")
        seen_rounds.add(round_index)
        columns = tuple(
            trace[name] for name in required - {"round_index", "source_version"}
        )
        if any(not isinstance(column, list) for column in columns):
            raise RuntimeError("round trace columns must be arrays")
        request_ids = trace["request_ids"]
        count = len(request_ids)
        if count < 1 or any(len(column) != count for column in columns):
            raise RuntimeError("round trace columns have inconsistent lengths")
        for request_id, prefix, verified, accepted, committed in zip(
            request_ids,
            trace["prefix_len_before"],
            trace["verify_len"],
            trace["accepted_drafts"],
            trace["committed_tokens"],
            strict=True,
        ):
            flat.append(
                (
                    round_index,
                    source_version,
                    str(request_id),
                    int(prefix),
                    int(verified),
                    int(accepted),
                    int(committed),
                )
            )
    records: list[RoundRecord] = []
    seen_cells: set[tuple[int, str]] = set()
    for (
        round_index,
        source_version,
        request_id,
        prefix,
        verified,
        accepted,
        committed,
    ) in sorted(flat):
        cell = (round_index, request_id)
        if cell in seen_cells or request_id not in inputs:
            raise RuntimeError("round trace references a duplicate or unknown request")
        seen_cells.add(cell)
        history = histories.get(request_id)
        if history is None:
            if prefix != inputs[request_id]:
                raise RuntimeError(
                    "first round prefix does not match the tokenized prompt"
                )
            history = [
                {
                    "start": 0,
                    "end": prefix,
                    "source_version": source_version,
                }
            ]
            histories[request_id] = history
        elif prefix != history[-1]["end"]:
            raise RuntimeError("round prefix does not continue the recorded KV history")
        request_end = inputs[request_id] + completions[request_id]
        closes_request_without_bonus = (
            committed == accepted
            and committed > 0
            and prefix + committed == request_end
        )
        if (
            verified < 0
            or accepted < 0
            or accepted > verified
            or committed < 0
            or (
                committed != accepted + 1
                and not closes_request_without_bonus
                and not (committed == 0 and accepted == 0)
            )
        ):
            raise RuntimeError("round speculative counts are inconsistent")
        generated_before = prefix - inputs[request_id]
        if generated_before < 0:
            raise RuntimeError("round prefix precedes the tokenized prompt")
        records.append(
            RoundRecord(
                run_id=run_id,
                request_id=request_id,
                round_index=round_index,
                generated_tokens_before=generated_before,
                prefix_len_before=prefix,
                verify_len=verified,
                accepted_drafts=accepted,
                committed_tokens=committed,
                target_calls=1,
                proposal_source_version=source_version,
                kv_source_versions=json.dumps(
                    history, sort_keys=True, separators=(",", ":")
                ),
            )
        )
        end = prefix + committed
        if committed > 0:
            if history[-1]["source_version"] == source_version:
                history[-1] = {**history[-1], "end": end}
            else:
                history.append(
                    {
                        "start": prefix,
                        "end": end,
                        "source_version": source_version,
                    }
                )
    if {record.request_id for record in records} != set(inputs):
        raise RuntimeError("round traces do not cover every completed request")
    for request_id, history in histories.items():
        expected_end = inputs[request_id] + completions[request_id]
        if history[-1]["end"] != expected_end:
            raise RuntimeError("round commits do not reconstruct request output length")
    runtime_segments = diagnostics["kv_segments"]
    if set(runtime_segments) != set(inputs):
        raise RuntimeError("KV-version evidence does not cover completed requests")
    for request_id, expected in histories.items():
        actual = runtime_segments[request_id]
        if not isinstance(actual, list) or not actual:
            raise RuntimeError("KV-version segment list is empty")
        clipped = []
        limit = expected[-1]["end"]
        for segment in actual:
            if not isinstance(segment, dict) or set(segment) != {
                "start",
                "end",
                "source_version",
            }:
                raise RuntimeError("KV-version segment fields are malformed")
            start = int(segment["start"])
            end = min(int(segment["end"]), limit)
            if start >= limit:
                break
            clipped.append(
                {
                    "start": start,
                    "end": end,
                    "source_version": int(segment["source_version"]),
                }
            )
        if clipped != expected:
            raise RuntimeError("round and KV-version evidence disagree")
    return tuple(records)


def _warmup(
    client: SGLangHTTPClient,
    *,
    method: str,
    concurrency: int,
    adaptation_group_id: str,
    sampling_profile: SamplingProfile,
) -> None:
    sample = LongContinuationAdapter().window("load")[0]
    independent_method_run(
        client,
        method=method,
        payloads=_payloads(
            sample,
            method=method,
            block=-1,
            concurrency=concurrency,
            max_new_tokens=64,
            sampling_profile=sampling_profile,
            request_namespace="warmup",
        ),
        concurrency=concurrency,
        adaptation_group_id=(None if method == "static" else adaptation_group_id),
    )


def measure_controlled_slice(
    *,
    client: SGLangHTTPClient,
    method: str,
    samples: tuple[PromptSample, ...],
    phase: str,
    stage: int,
    candidate_id: str | None,
    manifest_sha256: str,
    config_sha256: str,
    model_lock_sha256: str,
    adaptation_config_sha256: str | None,
    sampling_profile: SamplingProfile,
    context_limit: int,
    concurrency: int,
    adaptation_group_id: str,
    warmup: bool = True,
) -> SliceMeasurement:
    """Measure one pre-confirmation slice at the highest registered batch."""
    allowed = {
        "static_load_screen": {"static"},
        "shared_config_tuning": _FORMAL_METHODS,
        "onlinespec_tuning": _ONLINE_SPEC_METHODS,
    }
    if phase not in allowed:
        raise ValueError("controlled slice has an invalid phase")
    if method not in allowed[phase]:
        raise ValueError("method is outside the controlled study")
    if not samples or concurrency < 1:
        raise ValueError("controlled slice needs prompts and positive concurrency")
    budgets = _prompt_budgets(
        client,
        samples,
        safe_context_limit=context_limit,
    )
    if warmup:
        _warmup(
            client,
            method=method,
            concurrency=concurrency,
            adaptation_group_id=adaptation_group_id,
            sampling_profile=sampling_profile,
        )
    payloads, assignments = _batched_payloads(
        samples,
        budgets=budgets,
        method=method,
        block=stage,
        concurrency=concurrency,
        sampling_profile=sampling_profile,
        fill_concurrency=True,
    )
    run = independent_method_run(
        client,
        method=method,
        payloads=payloads,
        concurrency=concurrency,
        adaptation_group_id=(None if method == "static" else adaptation_group_id),
    )
    if run.after.kv_token_capacity < concurrency * context_limit:
        raise RuntimeError(
            "KV token capacity cannot sustain the registered load/context cell"
        )
    grouped = _group_results(run.results, assignments)
    for result in run.results:
        _, expected_input_tokens, max_new_tokens = assignments[result.request_id]
        if (
            result.input_tokens != expected_input_tokens
            or result.completion_tokens != max_new_tokens
            or result.stop_reason != "length"
        ):
            raise RuntimeError("controlled slice did not reach its context limit")
    output_trajectories = [
        (
            sample.sample_id,
            tuple(_output_sha256(result) for result in grouped[sample.sample_id]),
        )
        for sample in samples
    ]
    measured = _region(
        run.results,
        start=0,
        end=max(
            max_new_tokens for _, _, max_new_tokens in assignments.values()
        ),
    )
    if measured is None:
        raise RuntimeError("controlled slice produced no decode interval")
    _, _, decode_elapsed_s, intervals = measured
    transition_count = len(intervals)
    fields, updates, _rounds = _adaptation_fields(
        method, run.after, adaptation_config_sha256
    )
    loss_points: list[LossPoint] = []
    for update in updates:
        prefixes = update.get("prefix_len_before")
        if not isinstance(prefixes, list) or not prefixes:
            raise RuntimeError("update loss is not bound to real prefix lengths")
        resolved = tuple(int(value) for value in prefixes)
        loss_points.append(
            LossPoint(
                prefix_len_min=min(resolved),
                prefix_len_max=max(resolved),
                prefix_len_mean=sum(resolved) / len(resolved),
                loss=float(update["loss"]),
            )
        )
    if transition_count < 1 or decode_elapsed_s <= 0:
        raise RuntimeError("controlled slice has no measurable decode work")
    measurement = SliceMeasurement(
        schema_version=2,
        phase=phase,
        stage=stage,
        method=method,
        candidate_id=candidate_id,
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
        model_lock_sha256=model_lock_sha256,
        sampling_profile_sha256=sampling_profile.sha256,
        window_sha256=sample_set_sha256(samples),
        output_set_sha256=_output_set_sha256(output_trajectories),
        prompt_count=len(samples),
        context_limit=context_limit,
        concurrency=concurrency,
        decode_goodput_tps=transition_count / decode_elapsed_s,
        itl_p99_ms=_percentile(intervals, 0.99),
        peak_hbm_bytes=run.after.peak_hbm_bytes,
        kv_bytes=run.after.kv_bytes,
        kv_token_capacity=run.after.kv_token_capacity,
        optimizer_bytes=int(fields["optimizer_bytes"]),
        trainable_parameters=int(fields["trainable_parameters"]),
        exposed_update_ms=float(fields["exposed_update_ms"] or 0.0),
        updates_launched=int(fields["updates_launched"]),
        updates_published=int(fields["updates_published"]),
        exactness_violations=int(fields["exactness_violations"]),
        version_mismatches=int(fields["version_mismatches"]),
        fallbacks=int(fields["fallbacks"]),
        nonfinite_updates=int(fields["nonfinite_updates"]),
        oom_events=int(fields["oom_events"]),
        retractions=int(fields["retractions"]),
        loss_points=tuple(loss_points),
    )
    measurement.validate()
    return measurement


def _earlier_slices(
    *,
    method: str,
    block: int,
    schedule_seed: int,
    study_methods: tuple[str, ...] = ("static", "tts", "naive_async"),
) -> tuple[tuple[int, str], ...]:
    jobs = tuple(
        (entry.block, scheduled_method)
        for entry in paired_blocks(schedule_seed, study_methods)
        for scheduled_method in entry.method_order
    )
    target = (block, method)
    if target not in jobs:
        raise ValueError("confirmation slice is outside the registered schedule")
    return jobs[: jobs.index(target)]


def _assert_prior_slices_complete(
    output_root: str | Path,
    *,
    manifest_sha256: str,
    method: str,
    block: int,
    schedule_seed: int,
    samples: tuple[PromptSample, ...],
    study_methods: tuple[str, ...] = ("static", "tts", "naive_async"),
    namespace: str = "confirmation",
) -> None:
    batch_prompt_id = _batch_prompt_id(samples)
    for earlier_block, earlier_method in _earlier_slices(
        method=method,
        block=block,
        schedule_seed=schedule_seed,
        study_methods=study_methods,
    ):
        run_id = _run_id(
            manifest_sha256,
            earlier_block,
            batch_prompt_id,
            earlier_method,
            namespace=namespace,
        )
        if load_completed_evidence(output_root, run_id=run_id, rank=0) is None:
            raise RuntimeError(
                "confirmation slices must follow the registered randomized "
                f"order; missing predecessor {earlier_block}/{earlier_method}"
            )


def run_confirmation_slice(
    *,
    client: SGLangHTTPClient,
    method: str,
    block: int,
    manifest_sha256: str,
    config_sha256: str,
    adaptation_config_sha256: str | None,
    output_root: str | Path,
    concurrency: int,
    safe_context_limit: int,
    adaptation_group_id: str,
    schedule_seed: int,
    sampling_profile: SamplingProfile,
    model_pair: str = "qwen3_8b_dflash16",
    warmup: bool = True,
    study_methods: tuple[str, ...] = ("static", "tts", "naive_async"),
    namespace: str = "confirmation",
) -> tuple[Path, ...]:
    """Run one exclusive-device method slice in registered random order."""
    if frozenset(study_methods) not in {
        frozenset(_FORMAL_METHODS),
        frozenset(_ONLINE_SPEC_METHODS),
    }:
        raise ValueError("unknown paired study method set")
    if method not in study_methods:
        raise ValueError("method is outside the paired study")
    if block not in range(8):
        raise ValueError("formal confirmation block must be in [0, 8)")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if safe_context_limit != DFLASH_SAFE_CONTEXT_LIMIT:
        raise ValueError("formal confirmation must end at the registered safe limit")
    if not adaptation_group_id:
        raise ValueError("formal adapted runs require a cohort group")
    samples = LongContinuationAdapter().window("confirm")
    budgets = _prompt_budgets(
        client,
        samples,
        safe_context_limit=safe_context_limit,
        minimum_generation_tokens=32769,
    )
    _assert_prior_slices_complete(
        output_root,
        manifest_sha256=manifest_sha256,
        method=method,
        block=block,
        schedule_seed=schedule_seed,
        samples=samples,
        study_methods=study_methods,
        namespace=namespace,
    )
    if warmup:
        _warmup(
            client,
            method=method,
            concurrency=concurrency,
            adaptation_group_id=adaptation_group_id,
            sampling_profile=sampling_profile,
        )
    batch_prompt_id = _batch_prompt_id(samples)
    run_id = _run_id(
        manifest_sha256,
        block,
        batch_prompt_id,
        method,
        namespace=namespace,
    )
    completed = load_completed_evidence(output_root, run_id=run_id, rank=0)
    if completed is not None:
        run_rows = pq.read_table(completed["run"]).to_pylist()
        request_rows = pq.read_table(completed["request"]).to_pylist()
        performance_rows = pq.read_table(completed["performance"]).to_pylist()
        expected = {
            "manifest_sha256": manifest_sha256,
            "config_sha256": config_sha256,
            "method": method,
            "repetition_block": block,
        }
        if len(run_rows) != 1 or any(
            run_rows[0].get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError(f"completed run identity mismatch for {run_id}")
        expected_prompt_ids = {sample.sample_id for sample in samples}
        if (
            len(request_rows) != len(samples)
            or {str(row.get("prompt_id")) for row in request_rows}
            != expected_prompt_ids
            or any(
                row.get("method") != method
                or int(row.get("repetition_block", -1)) != block
                or int(row.get("concurrency", -1)) != concurrency
                for row in request_rows
            )
        ):
            raise RuntimeError(f"completed request identity mismatch for {run_id}")
        batch_rows = [
            row
            for row in performance_rows
            if row.get("prompt_id") == batch_prompt_id
        ]
        if (
            {str(row.get("region")) for row in batch_rows}
            != {"generated_bucket", "long_region", "full_trajectory"}
            or any(
                int(row.get("concurrency", -1)) != concurrency
                for row in batch_rows
            )
        ):
            raise RuntimeError(f"completed performance identity mismatch for {run_id}")
        return tuple(completed.values())

    payloads, assignments = _batched_payloads(
        samples,
        budgets=budgets,
        method=method,
        block=block,
        concurrency=concurrency,
        sampling_profile=sampling_profile,
        fill_concurrency=False,
    )
    started_ns = time.time_ns()
    run: MethodRun = independent_method_run(
        client,
        method=method,
        payloads=payloads,
        concurrency=concurrency,
        adaptation_group_id=(None if method == "static" else adaptation_group_id),
    )
    completed_ns = time.time_ns()
    grouped = _group_results(run.results, assignments)
    for result in run.results:
        _, expected_input_tokens, max_new_tokens = assignments[result.request_id]
        if (
            result.input_tokens != expected_input_tokens
            or result.completion_tokens != max_new_tokens
            or result.stop_reason != "length"
        ):
            raise RuntimeError(
                "controlled run did not reach its exact context-safe limit"
            )
    adaptation, updates, rounds = _adaptation_fields(
        method, run.after, adaptation_config_sha256
    )
    with EvidenceWriter(output_root, run_id=run_id, rank=0) as writer:
        writer.write(
            RunRecord(
                run_id=run_id,
                manifest_sha256=manifest_sha256,
                config_sha256=config_sha256,
                method=method,
                model_pair=model_pair,
                repetition_block=block,
                started_ns=started_ns,
                completed_ns=completed_ns,
                status="complete",
            )
        )
        for result in run.results:
            sample, _, _ = assignments[result.request_id]
            writer.write(
                RequestRecord(
                    run_id=run_id,
                    request_id=result.request_id,
                    prompt_id=sample.sample_id,
                    method=method,
                    repetition_block=block,
                    concurrency=concurrency,
                    input_tokens=result.input_tokens,
                    output_tokens=result.completion_tokens,
                    output_sha256=_output_sha256(result),
                    ttft_ms=result.ttft_ms,
                    finished=result.stop_reason is not None,
                    stop_reason=result.stop_reason,
                )
            )
        for round_record in _round_records(
            run_id=run_id,
            diagnostics=run.after.adaptation,
            results=run.results,
            rounds=rounds,
        ):
            writer.write(round_record)
        longest = max(result.completion_tokens for result in run.results)
        for start, end in GENERATED_TOKEN_BUCKETS:
            if start >= longest:
                continue
            batch_row = _performance_record(
                run_id=run_id,
                prompt_id=batch_prompt_id,
                method=method,
                block=block,
                concurrency=concurrency,
                region_name="generated_bucket",
                region_start=start,
                region_end=min(end, longest),
                results=run.results,
                snapshot=run.after,
                adaptation=adaptation,
                run_scope_metrics=False,
            )
            if batch_row is not None:
                writer.write(batch_row)
            for sample in samples:
                result = grouped[sample.sample_id][0]
                if min(end, result.completion_tokens) - start < 2:
                    continue
                request_row = _performance_record(
                    run_id=run_id,
                    prompt_id=sample.sample_id,
                    method=method,
                    block=block,
                    concurrency=concurrency,
                    region_name="request_generated_bucket",
                    region_start=start,
                    region_end=min(end, result.completion_tokens),
                    results=(result,),
                    snapshot=run.after,
                    adaptation=adaptation,
                    run_scope_metrics=False,
                )
                if request_row is not None:
                    writer.write(request_row)
        long_region = _performance_record(
            run_id=run_id,
            prompt_id=batch_prompt_id,
            method=method,
            block=block,
            concurrency=concurrency,
            region_name="long_region",
            region_start=16384,
            region_end=longest,
            results=run.results,
            snapshot=run.after,
            adaptation=adaptation,
            run_scope_metrics=False,
        )
        if long_region is None:
            raise RuntimeError("completed run has no measurable long region")
        writer.write(long_region)
        full = _performance_record(
            run_id=run_id,
            prompt_id=batch_prompt_id,
            method=method,
            block=block,
            concurrency=concurrency,
            region_name="full_trajectory",
            region_start=0,
            region_end=longest,
            results=run.results,
            snapshot=run.after,
            adaptation=adaptation,
            run_scope_metrics=True,
        )
        if full is None:
            raise RuntimeError("completed run has no performance row")
        writer.write(full)
        for sample in samples:
            result = grouped[sample.sample_id][0]
            if result.completion_tokens < 2:
                continue
            request_full = _performance_record(
                run_id=run_id,
                prompt_id=sample.sample_id,
                method=method,
                block=block,
                concurrency=concurrency,
                region_name="request_full_trajectory",
                region_start=0,
                region_end=result.completion_tokens,
                results=(result,),
                snapshot=run.after,
                adaptation=adaptation,
                run_scope_metrics=False,
            )
            if request_full is None:
                raise RuntimeError("completed request has no performance trajectory")
            writer.write(request_full)
        _write_updates(
            writer,
            run_id=run_id,
            method=method,
            diagnostics=run.after.adaptation,
            updates=updates,
        )
        return tuple(writer.close().values())


def _collect_paired_performance(
    *,
    evidence_root: str | Path,
    manifest_sha256: str,
    config_sha256: dict[str, str],
    concurrency: int,
    methods: tuple[str, ...],
    namespace: str,
) -> tuple[tuple[Path, ...], str]:
    """Collect one completed, identity-matched shard per paired study cell."""
    if set(config_sha256) != set(methods):
        raise ValueError("collector requires one config identity per method")
    samples = LongContinuationAdapter().window("confirm")
    batch_prompt_id = _batch_prompt_id(samples)
    expected_prompt_ids = {sample.sample_id for sample in samples}
    performance: list[Path] = []
    all_evidence: list[Path] = []
    for block in range(8):
        paired_outputs: dict[str, dict[str, str]] = {}
        for method in methods:
            run_id = _run_id(
                manifest_sha256,
                block,
                batch_prompt_id,
                method,
                namespace=namespace,
            )
            completed = load_completed_evidence(
                evidence_root,
                run_id=run_id,
                rank=0,
            )
            if completed is None:
                raise RuntimeError(f"confirmation evidence is incomplete: {run_id}")
            run_rows = pq.read_table(completed["run"]).to_pylist()
            request_rows = pq.read_table(completed["request"]).to_pylist()
            rows = pq.read_table(completed["performance"]).to_pylist()
            if len(run_rows) != 1 or any(
                run_rows[0].get(key) != value
                for key, value in {
                    "manifest_sha256": manifest_sha256,
                    "config_sha256": config_sha256[method],
                    "method": method,
                    "repetition_block": block,
                }.items()
            ):
                raise RuntimeError(f"confirmation run identity mismatch: {run_id}")
            batch_rows = [
                row for row in rows if row.get("prompt_id") == batch_prompt_id
            ]
            if (
                not batch_rows
                or {str(row.get("region")) for row in batch_rows}
                != {"generated_bucket", "long_region", "full_trajectory"}
                or any(
                    int(row.get("concurrency", -1)) != concurrency
                    for row in batch_rows
                )
            ):
                raise RuntimeError(
                    f"confirmation performance identity mismatch: {run_id}"
                )
            if (
                len(request_rows) != len(samples)
                or {str(row.get("prompt_id")) for row in request_rows}
                != expected_prompt_ids
                or any(
                    row.get("method") != method
                    or int(row.get("repetition_block", -1)) != block
                    or int(row.get("concurrency", -1)) != concurrency
                    or not isinstance(row.get("output_sha256"), str)
                    or len(str(row["output_sha256"])) != 64
                    for row in request_rows
                )
            ):
                raise RuntimeError(
                    f"confirmation request identity mismatch: {run_id}"
                )
            outputs = {
                str(row["prompt_id"]): str(row["output_sha256"])
                for row in request_rows
            }
            if len(outputs) != len(samples):
                raise RuntimeError("confirmation prompt outputs are duplicated")
            paired_outputs[method] = outputs
            performance.append(completed["performance"])
            all_evidence.extend(completed.values())
        reference = paired_outputs[methods[0]]
        if any(outputs != reference for outputs in paired_outputs.values()):
            raise RuntimeError(
                "paired greedy methods produced different output trajectories: "
                f"block={block}"
            )
    return tuple(performance), evidence_files_sha256(all_evidence)


def collect_confirmation_performance(
    *,
    evidence_root: str | Path,
    manifest_sha256: str,
    config_sha256: dict[str, str],
    concurrency: int,
) -> tuple[tuple[Path, ...], str]:
    """Collect exactly one completed, identity-matched formal shard per cell."""
    return _collect_paired_performance(
        evidence_root=evidence_root,
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
        concurrency=concurrency,
        methods=("static", "tts", "naive_async"),
        namespace="confirmation",
    )


def run_onlinespec_confirmation_slice(**kwargs) -> tuple[Path, ...]:
    """Run one clean-room OnlineSPEC comparison slice on confirmation data."""
    return run_confirmation_slice(
        **kwargs,
        study_methods=(
            "static",
            "onlinespec_ogd",
            "onlinespec_opt",
            "onlinespec_ens",
        ),
        namespace="onlinespec_confirmation",
    )


def collect_onlinespec_performance(
    *,
    evidence_root: str | Path,
    manifest_sha256: str,
    config_sha256: dict[str, str],
    concurrency: int,
) -> tuple[tuple[Path, ...], str]:
    return _collect_paired_performance(
        evidence_root=evidence_root,
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
        concurrency=concurrency,
        methods=(
            "static",
            "onlinespec_ogd",
            "onlinespec_opt",
            "onlinespec_ens",
        ),
        namespace="onlinespec_confirmation",
    )


def run_natural_replication_slice(
    *,
    client: SGLangHTTPClient,
    method: str,
    dataset_name: str,
    samples: tuple[PromptSample, ...],
    manifest_sha256: str,
    config_sha256: str,
    adaptation_config_sha256: str | None,
    output_root: str | Path,
    concurrency: int,
    safe_context_limit: int,
    adaptation_group_id: str,
    sampling_profile: SamplingProfile,
    model_pair: str = "qwen3_8b_dflash16",
    warmup: bool = True,
) -> tuple[Path, ...]:
    """Run one natural-EOS side-table slice; never enters the formal gate."""
    if method not in _FORMAL_METHODS or dataset_name not in {
        "livecodebench",
        "math500",
    }:
        raise ValueError("natural replication identity is invalid")
    if sampling_profile.purpose != "natural" or sampling_profile.ignore_eos:
        raise ValueError("natural replication requires its EOS-enabled profile")
    if len(samples) != 32:
        raise ValueError("natural side tables require exactly 32 locked prompts")
    budgets = _prompt_budgets(
        client,
        samples,
        safe_context_limit=safe_context_limit,
    )
    if warmup:
        _warmup(
            client,
            method=method,
            concurrency=concurrency,
            adaptation_group_id=adaptation_group_id,
            sampling_profile=sampling_profile,
        )
    namespace = f"natural-{dataset_name}"
    batch_prompt_id = _batch_prompt_id(samples)
    run_id = _run_id(
        manifest_sha256,
        0,
        batch_prompt_id,
        method,
        namespace=namespace,
    )
    completed = load_completed_evidence(output_root, run_id=run_id, rank=0)
    if completed is not None:
        run_rows = pq.read_table(completed["run"]).to_pylist()
        request_rows = pq.read_table(completed["request"]).to_pylist()
        performance_rows = pq.read_table(completed["performance"]).to_pylist()
        expected = {
            "manifest_sha256": manifest_sha256,
            "config_sha256": config_sha256,
            "method": method,
            "repetition_block": 0,
        }
        if len(run_rows) != 1 or any(
            run_rows[0].get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError(f"completed natural run identity mismatch for {run_id}")
        if (
            len(request_rows) != len(samples)
            or {str(row.get("prompt_id")) for row in request_rows}
            != {sample.sample_id for sample in samples}
            or not any(
                row.get("prompt_id") == batch_prompt_id
                and str(row.get("region", "")).startswith("natural")
                for row in performance_rows
            )
            or any(
                int(row.get("concurrency", -1)) != concurrency
                for row in performance_rows
            )
        ):
            raise RuntimeError(f"completed natural evidence mismatch for {run_id}")
        return tuple(completed.values())

    payloads, assignments = _batched_payloads(
        samples,
        budgets=budgets,
        method=method,
        block=0,
        concurrency=concurrency,
        sampling_profile=sampling_profile,
        fill_concurrency=False,
    )
    started_ns = time.time_ns()
    run = independent_method_run(
        client,
        method=method,
        payloads=payloads,
        concurrency=concurrency,
        adaptation_group_id=(None if method == "static" else adaptation_group_id),
    )
    completed_ns = time.time_ns()
    grouped = _group_results(run.results, assignments)
    for result in run.results:
        _, expected_input_tokens, max_new_tokens = assignments[result.request_id]
        if (
            result.input_tokens != expected_input_tokens
            or result.completion_tokens > max_new_tokens
            or result.stop_reason is None
        ):
            raise RuntimeError("natural run violated its context or terminal contract")
    adaptation, updates, rounds = _adaptation_fields(
        method, run.after, adaptation_config_sha256
    )
    with EvidenceWriter(output_root, run_id=run_id, rank=0) as writer:
        writer.write(
            RunRecord(
                run_id=run_id,
                manifest_sha256=manifest_sha256,
                config_sha256=config_sha256,
                method=method,
                model_pair=model_pair,
                repetition_block=0,
                started_ns=started_ns,
                completed_ns=completed_ns,
                status="complete",
            )
        )
        for result in run.results:
            sample, _, _ = assignments[result.request_id]
            writer.write(
                RequestRecord(
                    run_id=run_id,
                    request_id=result.request_id,
                    prompt_id=sample.sample_id,
                    method=method,
                    repetition_block=0,
                    concurrency=concurrency,
                    input_tokens=result.input_tokens,
                    output_tokens=result.completion_tokens,
                    output_sha256=_output_sha256(result),
                    ttft_ms=result.ttft_ms,
                    finished=True,
                    stop_reason=result.stop_reason,
                )
            )
        for round_record in _round_records(
            run_id=run_id,
            diagnostics=run.after.adaptation,
            results=run.results,
            rounds=rounds,
        ):
            writer.write(round_record)
        longest = max(result.completion_tokens for result in run.results)
        for start, end in GENERATED_TOKEN_BUCKETS:
            if start >= longest:
                continue
            batch_row = _performance_record(
                run_id=run_id,
                prompt_id=batch_prompt_id,
                method=method,
                block=0,
                concurrency=concurrency,
                region_name=f"natural:{dataset_name}",
                region_start=start,
                region_end=min(end, longest),
                results=run.results,
                snapshot=run.after,
                adaptation=adaptation,
                run_scope_metrics=False,
            )
            if batch_row is not None:
                writer.write(batch_row)
            for sample in samples:
                result = grouped[sample.sample_id][0]
                if min(end, result.completion_tokens) - start < 2:
                    continue
                request_row = _performance_record(
                    run_id=run_id,
                    prompt_id=sample.sample_id,
                    method=method,
                    block=0,
                    concurrency=concurrency,
                    region_name=f"natural_request:{dataset_name}",
                    region_start=start,
                    region_end=min(end, result.completion_tokens),
                    results=(result,),
                    snapshot=run.after,
                    adaptation=adaptation,
                    run_scope_metrics=False,
                )
                if request_row is not None:
                    writer.write(request_row)
        full = _performance_record(
            run_id=run_id,
            prompt_id=batch_prompt_id,
            method=method,
            block=0,
            concurrency=concurrency,
            region_name=f"natural_full:{dataset_name}",
            region_start=0,
            region_end=longest,
            results=run.results,
            snapshot=run.after,
            adaptation=adaptation,
            run_scope_metrics=True,
        )
        if full is None:
            raise RuntimeError("natural run has no measurable decode trajectory")
        writer.write(full)
        for sample in samples:
            result = grouped[sample.sample_id][0]
            if result.completion_tokens < 2:
                continue
            request_full = _performance_record(
                run_id=run_id,
                prompt_id=sample.sample_id,
                method=method,
                block=0,
                concurrency=concurrency,
                region_name=f"natural_request_full:{dataset_name}",
                region_start=0,
                region_end=result.completion_tokens,
                results=(result,),
                snapshot=run.after,
                adaptation=adaptation,
                run_scope_metrics=False,
            )
            if request_full is not None:
                writer.write(request_full)
        _write_updates(
            writer,
            run_id=run_id,
            method=method,
            diagnostics=run.after.adaptation,
            updates=updates,
        )
        return tuple(writer.close().values())
