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
    ttft_ms: float
    finished: bool
    stop_reason: str | None


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
    published_version: int | None
    candidate_status: str
    loss: float
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
    itl_p50_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
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
    peak_hbm_bytes: int
    kv_bytes: int
    optimizer_bytes: int
    adaptation_memory_ledger: str | None
    trainable_parameters: int
    training_cuda_ms: float | None
    optimizer_cuda_ms: float | None
    merge_cuda_ms: float | None
    publish_cuda_ms: float | None
    barrier_cuda_ms: float | None
    exposed_update_ms: float | None
    main_side_overlap_ratio: float | None
    graph_replay_hit_rate: float | None
    updates_launched: int
    updates_published: int
    exactness_violations: int
    version_mismatches: int
    fallbacks: int
    nonfinite_updates: int
    oom_events: int
    retractions: int
