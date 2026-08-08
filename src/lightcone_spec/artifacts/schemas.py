"""Parquet table schemas (spec 11.2-11.7). Field names and types are
normative; the validator rejects any table missing required fields."""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = 1

# Additive telemetry columns may be absent from schema-v1 artifacts created
# before the column was introduced.  New writers always materialize them (as
# null when the producer does not supply a value), while the validator uses
# this allow-list to keep historical artifacts readable without weakening the
# required-column contract for the original schema.
SCHEMA_COMPAT_OPTIONAL_FIELDS = {
    "rounds": frozenset(
        {
            "prefix_len_before",
            "verify_len",
            "batch_size",
            "offered_concurrency",
            "round_wall_us",
            "prefix_feature_exact",
            "algorithmic_censored",
            "cache_policy",
            "proposal_weight_version",
            "kv_version_min",
            "kv_version_max",
            "kv_append_version",
            "cache_version_canary_ok",
            "signal_prep_cuda_us",
        }
    ),
    "updates": frozenset(
        {
            "candidate_batch_size",
            "backward_cuda_us",
            "optimizer_cuda_us",
            "source_training_loss",
            "source_expected_accepted_prefix",
            "source_prefix_len",
            "gradient_weight_version",
            "gradient_kv_version_min",
            "gradient_kv_version_max",
            "gradient_version_canary_ok",
        }
    ),
    "system_samples": frozenset(
        {
            "sample_source",
            "activity_provenance",
            "contention_provenance",
            "sync_provenance",
        }
    ),
}

COMMON_FIELDS = [
    pa.field("schema_version", pa.int32()),
    pa.field("run_id", pa.string()),
    pa.field("unit_id", pa.string()),
    pa.field("request_id", pa.string()),
    pa.field("stream_id", pa.string(), nullable=True),
    pa.field("tenant_id_hash", pa.string()),
    pa.field("model_pair_id", pa.string()),
    pa.field("method", pa.string()),
    pa.field("dataset", pa.string()),
    pa.field("seed", pa.int32()),
    pa.field("lifecycle", pa.string()),
]

ROUNDS_SCHEMA = pa.schema(
    COMMON_FIELDS
    + [
        pa.field("round_id", pa.int64()),
        pa.field("prefix_pos_before", pa.int64()),
        pa.field("prefix_pos_after", pa.int64()),
        pa.field("active_version", pa.int64()),
        pa.field("proposal_version", pa.int64()),
        pa.field("draft_tokens", pa.int32()),
        pa.field("accepted_drafts", pa.int32()),
        pa.field("committed_per_verify", pa.int32()),
        pa.field("target_calls", pa.int32()),
        pa.field("draft_cpu_us", pa.float64()),
        pa.field("draft_cuda_us", pa.float64()),
        pa.field("verify_cpu_us", pa.float64()),
        pa.field("verify_cuda_us", pa.float64()),
        pa.field("accept_cuda_us", pa.float64()),
        pa.field("target_topk_token_ids", pa.list_(pa.int32())),
        pa.field("target_topk_probs", pa.list_(pa.float32())),
        pa.field("target_other_mass", pa.float32()),
        pa.field("proposal_topk_token_ids", pa.list_(pa.int32())),
        pa.field("proposal_topk_probs", pa.list_(pa.float32())),
        pa.field("proposal_other_mass", pa.float32()),
        pa.field("hidden_proj", pa.list_(pa.float32())),
        pa.field("event_sketch", pa.list_(pa.float32())),
        pa.field("endpoint_from_previous", pa.float32()),
        pa.field("rng_substream_id", pa.string()),
        pa.field("version_canary_ok", pa.bool_()),
        # Exact P5/load evidence was historically retained only in raw JSONL.
        # Keep these nullable so older/non-SGLang producers remain valid while
        # canonical Parquet artifacts preserve every supplied value.
        pa.field("prefix_len_before", pa.int64(), nullable=True),
        pa.field("verify_len", pa.int32(), nullable=True),
        pa.field("batch_size", pa.int32(), nullable=True),
        pa.field("offered_concurrency", pa.int32(), nullable=True),
        pa.field("round_wall_us", pa.float64(), nullable=True),
        pa.field("prefix_feature_exact", pa.bool_(), nullable=True),
        pa.field("algorithmic_censored", pa.bool_(), nullable=True),
        # Reserved for parameter-dependent draft-cache runtimes. Tail-only
        # runs leave these null; future frozen-old-KV producers can populate
        # them without another artifact schema migration.
        pa.field("cache_policy", pa.string(), nullable=True),
        pa.field("proposal_weight_version", pa.int64(), nullable=True),
        pa.field("kv_version_min", pa.int64(), nullable=True),
        pa.field("kv_version_max", pa.int64(), nullable=True),
        pa.field("kv_append_version", pa.int64(), nullable=True),
        pa.field("cache_version_canary_ok", pa.bool_(), nullable=True),
        # Nullable by design: older backends/artifacts did not time teacher
        # signal preparation.  Analysis preserves that state as unknown rather
        # than treating missing main-stream work as zero cost.
        pa.field("signal_prep_cuda_us", pa.float64(), nullable=True),
    ]
)

UPDATES_SCHEMA = pa.schema(
    COMMON_FIELDS
    + [
        pa.field("update_id", pa.string()),
        pa.field("source_round", pa.int64()),
        pa.field("apply_round", pa.int64(), nullable=True),
        pa.field("exposure_round", pa.int64(), nullable=True),
        pa.field("source_version", pa.int64()),
        pa.field("source_training_loss", pa.float64(), nullable=True),
        pa.field(
            "source_expected_accepted_prefix", pa.float64(), nullable=True
        ),
        pa.field("source_prefix_len", pa.int64(), nullable=True),
        pa.field("active_version_at_arrival", pa.int64()),
        pa.field("staging_version", pa.int64()),
        pa.field("published_version", pa.int64(), nullable=True),
        pa.field("delay_rounds", pa.int32()),
        pa.field("delay_tokens", pa.int64()),
        pa.field("delay_wall_us", pa.float64()),
        pa.field("delay_versions", pa.int32()),
        pa.field("snapshot_ts_us", pa.float64()),
        pa.field("teacher_ts_us", pa.float64(), nullable=True),
        pa.field("launch_ts_us", pa.float64(), nullable=True),
        pa.field("done_ts_us", pa.float64(), nullable=True),
        pa.field("commit_ts_us", pa.float64(), nullable=True),
        pa.field("exposure_ts_us", pa.float64(), nullable=True),
        pa.field("launch_event_id", pa.string()),
        pa.field("done_event_id", pa.string()),
        pa.field("commit_event_id", pa.string(), nullable=True),
        pa.field("grad_norm", pa.float64()),
        pa.field("grad_clip_scale", pa.float64()),
        pa.field("grad_sketch", pa.list_(pa.float32())),
        pa.field("candidate_delta_norm", pa.float64()),
        pa.field("side_queue_cuda_us", pa.float64()),
        pa.field("candidate_cuda_us", pa.float64()),
        # Physical candidate launches may be shared by several requests.  The
        # CUDA timing columns are amortized per request; this fanout is needed
        # to reconstruct launch efficiency without consulting raw JSONL.
        pa.field("candidate_batch_size", pa.int32(), nullable=True),
        pa.field("backward_cuda_us", pa.float64(), nullable=True),
        pa.field("optimizer_cuda_us", pa.float64(), nullable=True),
        pa.field("barrier_wait_cpu_us", pa.float64()),
        pa.field("publish_cuda_us", pa.float64()),
        pa.field("optimizer_step", pa.int32()),
        pa.field("numerical_ok", pa.bool_()),
        pa.field("failure_reason", pa.string(), nullable=True),
        # Optional version provenance for differentiable draft-backbone
        # updates. Existing tail optimizers leave these fields null.
        pa.field("gradient_weight_version", pa.int64(), nullable=True),
        pa.field("gradient_kv_version_min", pa.int64(), nullable=True),
        pa.field("gradient_kv_version_max", pa.int64(), nullable=True),
        pa.field("gradient_version_canary_ok", pa.bool_(), nullable=True),
    ]
)

DECISIONS_SCHEMA = pa.schema(
    COMMON_FIELDS
    + [
        pa.field("update_id", pa.string()),
        pa.field("rho_path", pa.float64()),
        pa.field("endpoint_distance", pa.float64()),
        pa.field("parameter_displacement", pa.float64()),
        pa.field("predicted_utility", pa.float64(), nullable=True),
        pa.field("predicted_mismatch", pa.float64(), nullable=True),
        pa.field("predicted_harm_probability", pa.float64(), nullable=True),
        pa.field("threshold", pa.float64(), nullable=True),
        pa.field("decision", pa.string()),
        pa.field("damping_factor", pa.float64()),
        pa.field("transport_rank", pa.int32(), nullable=True),
        pa.field("parameter_comp_norm", pa.float64(), nullable=True),
        pa.field("state_transport_norm", pa.float64(), nullable=True),
        pa.field("random_transport", pa.bool_()),
        pa.field("controller_cpu_us", pa.float64()),
        pa.field("controller_cuda_us", pa.float64()),
    ]
)

DECISION_ENUM = (
    "apply",
    "discard",
    "discard_noop_publish",
    "damp",
    "transport",
    "version_conflict",
)

SYSTEM_SAMPLES_SCHEMA = pa.schema(
    COMMON_FIELDS
    + [
        pa.field("timestamp_us", pa.float64()),
        pa.field("gpu_index", pa.int32()),
        pa.field("hbm_used_bytes", pa.int64()),
        pa.field("sm_occupancy", pa.float32(), nullable=True),
        pa.field("gpu_utilization", pa.float32()),
        pa.field("power_watts", pa.float32()),
        pa.field("energy_joules_delta", pa.float64()),
        # NVML does not observe CUDA stream activity or host synchronizations.
        # These values are nullable and their provenance is explicit so an
        # absence of evidence can never be interpreted as a measured zero.
        pa.field("main_stream_active", pa.bool_(), nullable=True),
        pa.field("side_stream_active", pa.bool_(), nullable=True),
        pa.field("stream_contention_class", pa.string(), nullable=True),
        pa.field("sync_us_delta", pa.float64(), nullable=True),
        pa.field("sample_source", pa.string(), nullable=True),
        pa.field("activity_provenance", pa.string(), nullable=True),
        pa.field("contention_provenance", pa.string(), nullable=True),
        pa.field("sync_provenance", pa.string(), nullable=True),
    ]
)

REQUEST_SUMMARY_SCHEMA = pa.schema(
    COMMON_FIELDS
    + [
        pa.field("prompt_id_hash", pa.string()),
        pa.field("task_type", pa.string()),
        pa.field("output_tokens", pa.int64()),
        pa.field("quality_metric_name", pa.string()),
        pa.field("quality_value", pa.float64(), nullable=True),
        pa.field("decode_wall_s", pa.float64()),
        pa.field("e2e_wall_s", pa.float64()),
        pa.field("decode_tps", pa.float64()),
        pa.field("e2e_tps", pa.float64()),
        pa.field("goodput_tps", pa.float64()),
        pa.field("offered_concurrency", pa.int32()),
        pa.field("ttft_ms", pa.float64(), nullable=True),
        pa.field("queue_ms", pa.float64(), nullable=True),
        pa.field("estimated_perf_scope", pa.string()),
        pa.field("estimated_tflops_per_gpu", pa.float64(), nullable=True),
        pa.field("estimated_mfu", pa.float64(), nullable=True),
        pa.field("estimated_read_gbps_per_gpu", pa.float64(), nullable=True),
        pa.field("estimated_write_gbps_per_gpu", pa.float64(), nullable=True),
        pa.field("peak_tflops_per_gpu", pa.float64(), nullable=True),
        pa.field("decode_step_count", pa.int64(), nullable=True),
        pa.field("decode_batch_size_step_mean", pa.float64(), nullable=True),
        pa.field("decode_batch_size_time_mean", pa.float64(), nullable=True),
        pa.field("decode_batch_size_std", pa.float64(), nullable=True),
        pa.field("decode_batch_fill_ratio", pa.float64(), nullable=True),
        pa.field("decode_scheduler_span_s", pa.float64(), nullable=True),
        pa.field(
            "decode_generated_tps_scheduler_span", pa.float64(), nullable=True
        ),
        pa.field("prefill_uncached_tokens", pa.int64(), nullable=True),
        pa.field("prefill_busy_s", pa.float64(), nullable=True),
        pa.field("nvml_gpu_utilization_mean", pa.float64(), nullable=True),
        pa.field("nvml_gpu_utilization_p10", pa.float64(), nullable=True),
        pa.field("nvml_gpu_utilization_p90", pa.float64(), nullable=True),
        pa.field("nvml_gpu_busy_fraction_90", pa.float64(), nullable=True),
        pa.field("adaptation_fallback_count", pa.int32()),
        pa.field("kv_retracted_requests", pa.int64()),
        pa.field("peak_running_requests", pa.int32()),
        pa.field("peak_queue_requests", pa.int32()),
        pa.field("model_weight_hbm_bytes", pa.int64()),
        pa.field("kv_cache_hbm_bytes", pa.int64()),
        pa.field("cuda_graph_hbm_bytes", pa.int64()),
        pa.field("kv_token_capacity", pa.int64()),
        pa.field("adaptation_fixed_bytes", pa.int64()),
        pa.field("adaptation_reserve_bytes", pa.int64()),
        pa.field("mean_accepted_drafts", pa.float64()),
        pa.field("mean_committed_per_verify", pa.float64()),
        pa.field("target_calls_per_output_token", pa.float64()),
        pa.field("p50_round_ms", pa.float64()),
        pa.field("p95_round_ms", pa.float64()),
        pa.field("p99_round_ms", pa.float64()),
        pa.field("p50_itl_ms", pa.float64()),
        pa.field("p95_itl_ms", pa.float64()),
        pa.field("p99_itl_ms", pa.float64()),
        pa.field("energy_per_token_j", pa.float64(), nullable=True),
        pa.field("peak_hbm_bytes", pa.int64()),
        pa.field("version_mismatch_count", pa.int32()),
        pa.field("status", pa.string()),
    ]
)

TABLES = {
    "rounds": ROUNDS_SCHEMA,
    "updates": UPDATES_SCHEMA,
    "decisions": DECISIONS_SCHEMA,
    "system_samples": SYSTEM_SAMPLES_SCHEMA,
    "request_summary": REQUEST_SUMMARY_SCHEMA,
}

UNIT_STATUSES = (
    "complete_valid",
    "failed_exactness",
    "failed_runtime",
    "resource_skip",
    "missing",
    "invalid_artifact",
)
