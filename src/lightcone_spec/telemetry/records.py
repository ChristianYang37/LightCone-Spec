"""Five normalized evidence tables used by the speed study."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    manifest_sha256: str
    config_sha256: str
    method: str
    model_pair: str
    repetition_block: int
    started_ns: int
    completed_ns: int | None
    status: str
    # Schema-v3 industrial runs bind the durable evidence shard to the exact
    # registry cell and every immutable input used to execute it.  These fields
    # remain optional so historical schema-v2 evidence stays readable and the
    # legacy speed-study runner keeps its existing wire contract.
    industrial_cell_id: str | None = None
    rank_config_sha256: str | None = None
    runtime_sha256: str | None = None
    split_sha256: str | None = None
    corpus_sha256: str | None = None
    arrival_trace_sha256: str | None = None
    request_ids_sha256: str | None = None
    sampling_profile_sha256: str | None = None
    model_lock_sha256: str | None = None
    patched_sglang_tree: str | None = None
    run_nonce_sha256: str | None = None
    topology_sha256: str | None = None
    tensor_parallel_size: int | None = None
    data_parallel_size: int | None = None
    world_size: int | None = None
    rank: int | None = None
    expected_request_rows: int | None = None
    expected_round_rows: int | None = None
    expected_update_rows: int | None = None
    expected_performance_rows: int | None = None
    workload_contract: str | None = None
    preflight_attestation_sha256: str | None = None


@dataclass(frozen=True)
class RequestRecord:
    run_id: str
    request_id: str
    prompt_id: str
    method: str
    repetition_block: int
    concurrency: int
    input_tokens: int
    output_tokens: int
    output_sha256: str
    ttft_ms: float | None
    finished: bool
    stop_reason: str | None
    # Claim-grade exactness uses ordered token IDs. ``output_sha256`` is the
    # canonical token-ID digest whenever these fields are present; decoded text
    # alone is deliberately insufficient because decoding is not injective.
    output_token_ids: str | None = None
    output_token_ids_sha256: str | None = None
    # Schema-v3 records preserve the exact terminal classification.  The
    # historical ``finished`` bit remains for schema-v2 readers, but cannot
    # distinguish rejection, deadline, cancellation, or drain-time unfinished
    # work and must not be used to impute those outcomes.
    outcome_status: str | None = None
    arrival_ns: int | None = None
    queue_enter_ns: int | None = None
    admitted_ns: int | None = None
    first_token_ns: int | None = None
    completed_ns: int | None = None
    token_timestamps_ns: str | None = None
    inter_token_ms: str | None = None
    token_timing_coverage: float | None = None
    coalesced_intervals: int | None = None
    admission_code: str | None = None
    cancellation_code: str | None = None
    error_code: str | None = None
    retry_of_request_id: str | None = None
    retry_attempt: int | None = None
    cohort_sha256: str | None = None
    route_id: str | None = None


@dataclass(frozen=True)
class RoundRecord:
    run_id: str
    request_id: str
    round_index: int
    generated_tokens_before: int
    prefix_len_before: int
    verify_len: int
    accepted_drafts: int
    committed_tokens: int
    target_calls: int
    proposal_source_version: int
    kv_source_versions: str
    draft_width: int | None = None
    confidence: float | None = None
    brier_score: float | None = None
    expected_calibration_error: float | None = None
    graph_replay_hit: bool | None = None
    graph_bucket: str | None = None
    graph_fallback: bool | None = None
    interval_union_ms: float | None = None
    interval_overlap_ms: float | None = None
    target_executed_flops: float | None = None
    draft_executed_flops: float | None = None
    rejected_flops: float | None = None
    executed_hbm_bytes: int | None = None


@dataclass(frozen=True)
class UpdateRecord:
    run_id: str
    cohort_sha256: str
    parameter_layout_sha256: str
    update_index: int
    request_ids: str
    prefix_len_before: str
    prefix_len_min: int
    prefix_len_max: int
    prefix_len_mean: float
    source_round: int
    source_version: int
    optimizer_step: int
    published_version: int | None
    candidate_status: str
    loss: float
    gradient_norm: float
    reconstruction_ok: bool
    reconstruction_max_abs: float
    reconstruction_relative_rms: float | None
    reconstruction_top1_match: float | None
    reconstruction_mean_kl: float | None
    supervision_nonempty: bool
    trainable_parameters: int
    training_cuda_ms: float | None
    optimizer_cuda_ms: float | None
    merge_cuda_ms: float | None
    publish_cuda_ms: float | None
    barrier_cuda_ms: float | None
    exposed_update_ms: float | None
    overlap_ratio: float | None
    online_hint_error: float | None
    online_ensemble_entropy: float | None
    online_effective_experts: float | None
    online_expert_probabilities: str | None
    online_cumulative_losses: str | None
    online_expert_gradient_norms: str | None
    cohort_epoch: int | None = None
    route_id: str | None = None
    retry_identity: str | None = None
    discard_reason: str | None = None
    candidate_age_ms: float | None = None
    graph_fallback: bool | None = None
    reserved_bytes: int | None = None
    collective_type: str | None = None
    collective_bytes: int | None = None
    collective_duration_ms: float | None = None
    collective_exposed_wait_ms: float | None = None
    collective_overlap_ratio: float | None = None
    collective_algorithm: str | None = None
    collective_topology: str | None = None
    collective_slowest_rank_ms: float | None = None
    exactness_violation: bool | None = None
    stale_candidate: bool | None = None
    nonfinite_candidate: bool | None = None
    oom_candidate: bool | None = None
    retracted_candidate: bool | None = None


@dataclass(frozen=True)
class PerformanceRecord:
    run_id: str
    prompt_id: str
    method: str
    repetition_block: int
    region: str
    concurrency: int
    generated_bucket_start: int
    generated_bucket_end: int
    at_risk_requests: int
    output_tokens: int
    elapsed_s: float
    decode_goodput_tps: float
    itl_p50_ms: float | None
    itl_p95_ms: float | None
    itl_p99_ms: float | None
    survival_weighted_accepted_prefix: float | None
    accepted_drafts_per_verify: float | None
    committed_tokens_per_verify: float | None
    verified_drafts_per_verify: float | None
    verification_waste: float | None
    target_calls_per_output_token: float | None
    batch_fill: float | None
    queue_occupancy: float | None
    gpu_busy: float | None
    sm_utilization: float | None
    dram_utilization: float | None
    target_estimated_mfu: float | None
    peak_hbm_bytes: int | None
    kv_bytes: int | None
    optimizer_bytes: int | None
    adaptation_memory_ledger: str | None
    trainable_parameters: int | None
    training_cuda_ms: float | None
    optimizer_cuda_ms: float | None
    merge_cuda_ms: float | None
    publish_cuda_ms: float | None
    barrier_cuda_ms: float | None
    exposed_update_ms: float | None
    main_side_overlap_ratio: float | None
    graph_replay_hit_rate: float | None
    updates_launched: int | None
    updates_published: int | None
    exactness_violations: int | None
    version_mismatches: int | None
    fallbacks: int | None
    nonfinite_updates: int | None
    oom_events: int | None
    retractions: int | None
    queue_count: int | None = None
    active_requests: int | None = None
    allocated_hbm_bytes: int | None = None
    reserved_hbm_bytes: int | None = None
    nvml_process_hbm_bytes: int | None = None
    nvml_global_hbm_bytes: int | None = None
    fragmentation_margin_bytes: int | None = None
    collective_type: str | None = None
    collective_bytes: int | None = None
    collective_duration_ms: float | None = None
    collective_exposed_wait_ms: float | None = None
    collective_overlap_ratio: float | None = None
    collective_algorithm: str | None = None
    collective_topology: str | None = None
    collective_slowest_rank_ms: float | None = None
    executed_flops: float | None = None
    committed_useful_flops: float | None = None
    precision_normalized_executed_mfu: float | None = None
    target_equivalent_useful_utilization: float | None = None
    executed_hbm_bytes: int | None = None
    executed_flops_per_committed_token: float | None = None
    hbm_bytes_per_committed_token: float | None = None
    power_watts: float | None = None
    energy_joules: float | None = None
    output_tokens_per_joule: float | None = None
    slo_qualified_requests_per_gpu_hour: float | None = None
    slo_qualified_useful_token_goodput: float | None = None
    gpu_clock_mhz: float | None = None
    memory_clock_mhz: float | None = None
    temperature_c: float | None = None
    throttling_reasons: str | None = None
    admission_rejections: int | None = None
    timeouts: int | None = None
    cancellations: int | None = None
    offered_requests: int | None = None
    admitted_requests: int | None = None
    completed_requests: int | None = None
    unfinished_requests: int | None = None
    communicator_failures: int | None = None
    evidence_backpressure_events: int | None = None
    evidence_dropped_rows: int | None = None
