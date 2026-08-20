"""Strict host client for the pinned SGLang terminal-evidence hook.

The server owns the evidence.  This module only constructs nonce-bound control
messages and validates the exact JSON returned by the admin endpoints.  In
particular, object attributes, command-line flags, and caller-authored
capability dictionaries never substitute for a capability response obtained
through :class:`AsyncNativeTerminalAdminTransport`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from lightcone_spec.orchestration.formal_terminal_shards import (
    publish_scalable_native_terminal_artifact,
    publish_scalable_unsigned_native_itl_bundle,
    reopen_scalable_native_terminal_artifact,
    reopen_scalable_unsigned_native_itl_bundle,
)
from lightcone_spec.runtime.attestation import (
    NO_TRUSTED_ATTESTERS,
    TrustedAttesterPolicy,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    VerifiedControlArtifact,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

if TYPE_CHECKING:
    from lightcone_spec.experiments.serving import BoundServingRequest

LEGACY_NATIVE_TERMINAL_EVIDENCE_HOOK = (
    "sglang.schema_v3.content_bound_terminal_speculative_evidence.v1"
)
NATIVE_TERMINAL_EVIDENCE_HOOK = (
    "sglang.schema_v3.content_bound_terminal_speculative_evidence.v2"
)
PINNED_SGLANG_UPSTREAM_COMMIT = "3312645a307453893a00778592f105581e3d1c3d"
PINNED_SGLANG_PATCH_SHA256 = (
    "0c4db4f8798645c0ba65e97031030fb5e891d15f63cd75105fc1e1656c1a2874"
)
PINNED_SGLANG_TREE = "bb6371242e82592d1b8a2f5f4ba6d0630d8365cb"

CAPABILITY_PATH = "/v1/lightcone-spec/terminal-evidence/capability"
TERMINAL_EVIDENCE_PATH = "/v1/lightcone-spec/terminal-evidence"
LEGACY_NATIVE_TERMINAL_ARTIFACT_KIND = "native_terminal_evidence_bundle_v1"
NATIVE_TERMINAL_ARTIFACT_KIND = "native_terminal_evidence_bundle_v2"
_ZERO_SHA256 = "0" * 64
REQUEST_SOURCE_POINT_RESET_PROTOCOL_CANONICAL_JSON = (
    b'{"acquired_receipt":{"adaptation_state_acquired":true,'
    b'"five_reset_predicates":true,"reset_required":true,'
    b'"state_untouched":false},'
    b'"admission":"serialized_native_scheduler_v1",'
    b'"archive":{"canonical_payload":["schema_version",'
    b'"previous_archive_sha256","request_epoch","request_id",'
    b'"updates","rounds"],"initial_sha256":"64_zeroes",'
    b'"order":"native_append_order"},'
    b'"candidate":"discard_before_restore",'
    b'"coverage":"every_server_submitted_completed_or_aborted_request",'
    b'"evidence_key":["request_epoch","request_id","source_round",'
    b'"source_version"],"epochs":"acquired_contiguous_from_1_unacquired_0",'
    b'"failure":"sticky_disable_and_terminal_fail",'
    b'"noop_receipt":{"adaptation_state_acquired":false,'
    b'"five_reset_predicates":null,"reset_required":false,'
    b'"state_untouched":true},'
    b'"optimizer":"restore_initial_zero_moments",'
    b'"phases":["warmup","scored"],"reset_scope":"request",'
    b'"terminal_boundary":{"round":"exact_max_archived_or_zero",'
    b'"version":"exact_max_published_or_zero"},'
    b'"schema_version":2,"terminal_paths":["completed","aborted"],'
    b'"zero_evidence_acquired_terminal":true}'
)
REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256 = hashlib.sha256(
    REQUEST_SOURCE_POINT_RESET_PROTOCOL_CANONICAL_JSON
).hexdigest()

LEGACY_NATIVE_TERMINAL_EVIDENCE_FIELDS = (
    "schema_version",
    "hook",
    "run_id",
    "run_nonce_sha256",
    "execution_plan_sha256",
    "rank_config_sha256",
    "server_process_id",
    "server_process_started_ns",
    "attempt_id",
    "session_id",
    "session_epoch",
    "previous_run_id",
    "challenge_nonce_sha256",
    "method",
    "expected_request_ids",
    "reset_receipt_sha256",
    "request_round_rows",
    "update_rows",
    "performance_counters",
    "historical_kv_source_versions",
    "final_state",
    "completion_marker",
    "terminal_sha256",
    "attestation",
)
NATIVE_TERMINAL_EVIDENCE_FIELDS = (
    *LEGACY_NATIVE_TERMINAL_EVIDENCE_FIELDS[:14],
    "reset_scope",
    "request_admission_policy",
    "request_source_point_reset_protocol_sha256",
    "runtime_trust_mode",
    "formal_measurement",
    *LEGACY_NATIVE_TERMINAL_EVIDENCE_FIELDS[14:20],
    "request_source_point_resets",
    *LEGACY_NATIVE_TERMINAL_EVIDENCE_FIELDS[20:],
)
_CANDIDATE_METHODS = frozenset({"tts", "l0"})
_ADAPTIVE_METHODS = frozenset(
    {
        "tts",
        "l0",
        "onlinespec_ogd",
        "onlinespec_opt",
        "onlinespec_ens",
    }
)
SUPPORTED_METHODS = frozenset(
    {
        "target_only",
        "static",
        "tts",
        "l0",
        "onlinespec_ogd",
        "onlinespec_opt",
        "onlinespec_ens",
    }
)
_ORDERED_SUPPORTED_METHODS = tuple(sorted(SUPPORTED_METHODS))
_RUNTIME_TRUST_IDENTITIES = frozenset(
    {
        ("release_verified_signature", True),
        ("qualification_empirical_no_signature", False),
        ("trusted_single_operator_empirical_no_signature", False),
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,511}\Z")
_LOWER_HEX = re.compile(r"(?:[0-9a-f]{2})+\Z")

_CAPABILITY_KEYS = {
    "schema_version",
    "hook",
    "required_fields",
    "supported_methods",
    "enabled",
    "active_method",
    "method_evidence_supported",
    "topology_supported",
    "trusted_attester_configured",
    "reset_scope",
    "request_admission_policy",
    "request_source_point_reset_protocol_sha256",
    "runtime_trust_mode",
    "formal_measurement",
}
_LEGACY_IDENTITY_KEYS = {
    "run_id",
    "run_nonce_sha256",
    "execution_plan_sha256",
    "rank_config_sha256",
    "attempt_id",
    "session_id",
    "session_epoch",
    "previous_run_id",
    "challenge_nonce_sha256",
    "method",
}
_IDENTITY_KEYS = {
    *_LEGACY_IDENTITY_KEYS,
    "reset_scope",
    "request_admission_policy",
    "runtime_trust_mode",
    "formal_measurement",
}
_BEGIN_RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "hook",
    *_IDENTITY_KEYS,
    "server_process_id",
    "server_process_started_ns",
    "reset_generation",
    "request_source_point_reset_protocol_sha256",
    "prior_state_sha256",
    "reset_state_sha256",
    "warmup_request_ids_sha256",
    "scored_request_ids_sha256",
    "begin_sha256",
}
_LEGACY_BEGIN_RECEIPT_KEYS = _BEGIN_RECEIPT_KEYS - {
    "reset_scope",
    "request_admission_policy",
    "request_source_point_reset_protocol_sha256",
    "runtime_trust_mode",
    "formal_measurement",
}
_RESET_RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "hook",
    *_IDENTITY_KEYS,
    "server_process_id",
    "server_process_started_ns",
    "begin_sha256",
    "reset_generation",
    "request_source_point_reset_protocol_sha256",
    "prior_trace_run_id",
    "next_trace_run_id",
    "warmup_request_rows_sha256",
    "warmup_performance_sha256",
    "discarded_native_sha256",
    "warmup_state_sha256",
    "reset_state_sha256",
    "expected_scored_request_ids_sha256",
    "completion_event_generation",
    "warmup_request_rows",
    "warmup_round_rows",
    "warmup_update_rows",
    "warmup_historical_kv_source_versions",
    "warmup_request_source_point_resets",
    "warmup_performance_counters",
    "warmup_state",
    "reset_state",
    "reset_sha256",
}
_LEGACY_RESET_RECEIPT_KEYS = _RESET_RECEIPT_KEYS - {
    "reset_scope",
    "request_admission_policy",
    "request_source_point_reset_protocol_sha256",
    "runtime_trust_mode",
    "formal_measurement",
    "warmup_request_rows",
    "warmup_round_rows",
    "warmup_update_rows",
    "warmup_historical_kv_source_versions",
    "warmup_request_source_point_resets",
    "warmup_performance_counters",
    "warmup_state",
    "reset_state",
}
_SERVER_REQUEST_KEYS = {
    "request_id",
    "terminal_source",
    "input_tokens",
    "input_token_ids_sha256",
    "output_tokens",
    "ordered_output_token_ids",
    "output_token_ids_sha256",
    "terminal_status",
    "terminal_reason",
    "request_sha256",
}
_CLIENT_TERMINAL_KEYS = {
    "schema_version",
    "hook",
    "run_id",
    "run_nonce_sha256",
    "execution_plan_sha256",
    "rank_config_sha256",
    "request_id",
    "terminal_status",
    "terminal_reason",
    "client_terminal_sha256",
}
_RECONCILED_REQUEST_KEYS = {
    *_CLIENT_TERMINAL_KEYS,
    "terminal_source",
    "output_tokens",
    "request_sha256",
}
_LEGACY_ROUND_KEYS = {
    "request_id",
    "round_index",
    "proposal_source_version",
    "prefix_len_before",
    "verify_len",
    "accepted_drafts",
    "committed_tokens",
    "target_calls",
    "historical_kv_source_versions",
    "round_sha256",
}
_ROUND_KEYS = {
    *(_LEGACY_ROUND_KEYS - {"historical_kv_source_versions"}),
    "historical_kv_source_versions_sha256",
    "reset_scope",
    "request_epoch",
    "request_reset_receipt_sha256",
}
_LEGACY_UPDATE_KEYS = {
    "update_index",
    "cohort_sha256",
    "cohort_epoch",
    "parameter_layout_sha256",
    "source_round",
    "source_version",
    "request_ids",
    "prefix_len_before",
    "optimizer_step",
    "published_version",
    "status",
    "loss",
    "gradient_norm",
    "reconstruction_ok",
    "reconstruction_max_abs",
    "reconstruction_relative_rms",
    "reconstruction_top1_match",
    "supervision_nonempty",
    "reconstruction_mean_kl",
    "online_hint_error",
    "online_ensemble_entropy",
    "online_effective_experts",
    "online_expert_probabilities",
    "online_cumulative_losses",
    "online_expert_gradient_norms",
    "source_state_sha256",
    "candidate_bytes_sha256",
    "optimizer_state_bytes_sha256",
    "proposal_evidence_sha256",
    "update_sha256",
}
_UPDATE_KEYS = {
    *_LEGACY_UPDATE_KEYS,
    "effective_learning_rate",
    "schedule_valid",
    "intrinsic_ready_round",
    "extra_logical_delay",
    "publication_round",
    "reset_scope",
    "request_epoch",
    "request_reset_receipt_sha256",
}
_REQUEST_ARCHIVE_UPDATE_STRIP_KEYS = {
    "update_index",
    "reset_scope",
    "request_reset_receipt_sha256",
    "cohort_sha256",
    "cohort_epoch",
    "parameter_layout_sha256",
    "update_sha256",
}
_REQUEST_SOURCE_POINT_RESET_KEYS = {
    "schema_version",
    "reset_scope",
    "request_admission_policy",
    "protocol_sha256",
    "final_archive_sha256",
    "receipts",
}
_REQUEST_SOURCE_POINT_RESET_RECEIPT_KEYS = {
    "request_id",
    "request_epoch",
    "terminal_outcome",
    "terminal_round",
    "terminal_version",
    "adaptation_state_acquired",
    "reset_required",
    "state_untouched",
    "source_point_identity_sha256",
    "master_reset",
    "optimizer_reset",
    "inference_reset",
    "captured_state_empty",
    "runtime_reset",
    "sticky_disabled_reason",
    "evidence_archive_sha256",
    "archived_update_count",
    "archived_round_count",
    "previous_receipt_sha256",
    "protocol_sha256",
    "receipt_sha256",
}
_PERFORMANCE_KEYS = {
    "target_calls",
    "accepted_drafts",
    "committed_tokens",
    "verified_drafts",
    "survival_weighted_accepted_prefix",
    "accepted_drafts_per_verify",
    "committed_tokens_per_verify",
    "verified_drafts_per_verify",
    "verification_waste",
    "target_calls_per_output_token",
    "batch_fill",
    "queue_occupancy",
    "gpu_busy",
    "sm_utilization",
    "dram_utilization",
    "target_estimated_mfu",
    "peak_hbm_bytes",
    "kv_bytes",
    "kv_token_capacity",
    "optimizer_bytes",
    "adaptation_memory_ledger",
    "trainable_parameters",
    "training_cuda_ms",
    "optimizer_cuda_ms",
    "merge_cuda_ms",
    "publish_cuda_ms",
    "barrier_cuda_ms",
    "exposed_update_ms",
    "main_side_overlap_ratio",
    "graph_replay_hit_rate",
    "updates_launched",
    "updates_published",
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "communicator_failures",
    "collective_type",
    "collective_bytes",
    "collective_duration_ms",
    "collective_exposed_wait_ms",
    "collective_overlap_ratio",
}
_PERFORMANCE_INTEGER_KEYS = {
    "target_calls",
    "accepted_drafts",
    "committed_tokens",
    "verified_drafts",
    "peak_hbm_bytes",
    "kv_bytes",
    "kv_token_capacity",
    "optimizer_bytes",
    "trainable_parameters",
    "updates_launched",
    "updates_published",
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "communicator_failures",
    "collective_bytes",
}
_PERFORMANCE_OVERRIDE_KEYS = (
    "survival_weighted_accepted_prefix",
    "accepted_drafts_per_verify",
    "committed_tokens_per_verify",
    "verified_drafts_per_verify",
    "verification_waste",
    "target_calls_per_output_token",
    "batch_fill",
    "queue_occupancy",
    "gpu_busy",
    "sm_utilization",
    "dram_utilization",
    "target_estimated_mfu",
    "peak_hbm_bytes",
    "kv_bytes",
    "optimizer_bytes",
    "adaptation_memory_ledger",
    "trainable_parameters",
    "training_cuda_ms",
    "optimizer_cuda_ms",
    "merge_cuda_ms",
    "publish_cuda_ms",
    "barrier_cuda_ms",
    "exposed_update_ms",
    "main_side_overlap_ratio",
    "graph_replay_hit_rate",
    "updates_launched",
    "updates_published",
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "communicator_failures",
    "collective_type",
    "collective_bytes",
    "collective_duration_ms",
    "collective_exposed_wait_ms",
    "collective_overlap_ratio",
)
_ADAPTATION_MEMORY_LEDGER_KEYS = {
    "active_or_base_bytes",
    "master_fp32_bytes",
    "first_moment_bytes",
    "second_moment_bytes",
    "online_state_bytes",
    "optimizer_metadata_bytes",
    "gradient_bytes",
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
_LEGACY_STATE_KEYS = {
    "schema_version",
    "scheduler_idle",
    "active_requests",
    "queued_requests",
    "request_pool_active_slots",
    "allocator_current_hbm_bytes",
    "allocator_reserved_hbm_bytes",
    "allocator_peak_hbm_bytes",
    "kv_token_capacity",
    "kv_available_tokens",
    "kv_state_sha256",
    "rng_state_sha256",
    "adapter_state_sha256",
    "adapter_reset_verified",
    "adapter_active_version",
    "adapter_epoch",
    "optimizer_generation",
    "telemetry_generation",
    "completion_event_generation",
    "completion_event_complete",
}
_STATE_KEYS = {
    *_LEGACY_STATE_KEYS,
    "adapter_reset_scope",
    "adapter_request_admission_policy",
    "adapter_request_source_point_reset_protocol_sha256",
    "adapter_runtime_trust_mode",
    "adapter_formal_measurement",
    "adapter_active_request_id",
    "adapter_request_epoch",
    "adapter_source_round",
}
_ATTESTATION_KEYS = {
    "schema_version",
    "status",
    "challenge_nonce_sha256",
    "message_sha256",
    "attester_id",
    "trust_domain",
    "signature_hex",
}
_LEGACY_TERMINAL_KEYS = set(LEGACY_NATIVE_TERMINAL_EVIDENCE_FIELDS)
_TERMINAL_KEYS = set(NATIVE_TERMINAL_EVIDENCE_FIELDS)
_ARTIFACT_KEYS = {
    "schema_version",
    "artifact_kind",
    "run_id",
    "rank",
    "trusted_attester_policy_sha256",
    "begin_sha256",
    "reset_sha256",
    "terminal_sha256",
    "binding",
    "warmup_requests",
    "scored_requests",
    "begin",
    "reset",
    "terminal",
}
_LEGACY_ARTIFACT_BINDING_KEYS = {
    "run_id",
    "run_nonce_sha256",
    "execution_plan_sha256",
    "rank_config_sha256",
    "attempt_id",
    "session_id",
    "session_epoch",
    "previous_run_id",
    "challenge_nonce_sha256",
    "method",
    "warmup_request_ids",
    "scored_request_ids",
}
_ARTIFACT_BINDING_KEYS = {
    *_LEGACY_ARTIFACT_BINDING_KEYS,
    "reset_scope",
    "request_admission_policy",
    "runtime_trust_mode",
    "formal_measurement",
}
_ARTIFACT_REQUEST_KEYS = {
    "request_id",
    "input_token_ids",
    "output_token_ids",
    "terminal_status",
    "terminal_reason",
    "submitted_to_server",
}


class AsyncNativeTerminalAdminTransport(Protocol):
    """Authenticated JSON transport for the two native admin endpoints."""

    async def get_json(self, path: str, /) -> object: ...

    async def post_json(
        self,
        path: str,
        body: Mapping[str, object],
        /,
    ) -> object: ...


# Deprecated import compatibility only.  Provider construction deliberately
# accepts no caller-supplied verifier; release trust is rooted exclusively in
# ``TrustedAttesterPolicy``.
SignatureVerifier = Callable[[bytes, bytes], bool]


def canonical_json_bytes(value: object) -> bytes:
    """Return the exact canonical JSON encoding used by the pinned hook."""

    _validate_strict_json(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_json_text(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _validate_strict_json(value: object, field: str = "payload") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_strict_json(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{field} contains a non-string JSON key")
        for key, item in value.items():
            _validate_strict_json(item, f"{field}.{key}")
        return
    raise ValueError(f"{field} is not strict JSON")


NATIVE_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256 = canonical_sha256(
    {
        "schema_version": 1,
        "kind": "native_terminal_external_control_protocol",
        "remote_producer": "canonical_unsigned_or_genuine_hardware_signed_terminal",
        "local_authority": "root_authorized_non_serving_terminal_control",
        "content_binding": [
            "canonical_raw_sha256",
            "semantic_artifact_sha256",
            "native_terminal_sha256",
            "inventory_sha256",
            "registry_sha256",
            "run_and_attempt_lineage",
        ],
        "replay": "atomic_release_control_reservation",
    }
)


def _exact_object(value: object, keys: set[str], field: str) -> dict[str, object]:
    _validate_strict_json(value, field)
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} fields are incomplete or unknown")
    return dict(value)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _safe_id(value: object, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be a safe non-empty identity")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be an integer at least {minimum}")
    return value


def _number(
    value: object,
    field: str,
    *,
    nullable: bool = False,
    minimum: float | None = None,
) -> int | float | None:
    if value is None and nullable:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be boolean")
    return value


def _token_ids(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must contain ordered token IDs")
    tokens = tuple(value)
    if any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in tokens
    ):
        raise ValueError(f"{field} contains an invalid token ID")
    return tokens


def _wire_reset_identity(binding: NativeTerminalRunBinding) -> tuple[str, str]:
    """Map the typed host identity to the only accepted native wire pair."""

    if binding.reset_scope is None and binding.request_admission_policy is None:
        return "none", "allocation_free"
    if binding.reset_scope == "request" and binding.request_admission_policy == (
        "serialized_native_scheduler_v1"
    ):
        return binding.reset_scope, binding.request_admission_policy
    if binding.reset_scope == "cohort" and binding.request_admission_policy == (
        "cohort_batching_v1"
    ):
        return binding.reset_scope, binding.request_admission_policy
    raise ValueError("native terminal reset identity is not an exact supported pair")


def _validate_runtime_trust_identity(
    *,
    method: str,
    runtime_trust_mode: object,
    formal_measurement: object,
    field: str,
) -> tuple[str | None, bool | None]:
    """Validate the source-owned runtime trust and measurement pair."""

    if (runtime_trust_mode, formal_measurement) == (None, None):
        return None, None
    if method in {"target_only", "static"}:
        raise ValueError(f"{field} allocation-free runtime trust must be null")
    if (runtime_trust_mode, formal_measurement) not in _RUNTIME_TRUST_IDENTITIES:
        raise ValueError(f"{field} runtime trust/formal measurement pair differs")
    assert isinstance(runtime_trust_mode, str)
    assert isinstance(formal_measurement, bool)
    return runtime_trust_mode, formal_measurement


def _validate_wire_reset_protocol(
    *,
    method: str,
    reset_scope: object,
    request_admission_policy: object,
    protocol_sha256: object,
    field: str,
) -> str | None:
    """Validate the source-owned reset identity against the pinned protocol."""

    if method in {"target_only", "static"}:
        if (
            reset_scope != "none"
            or request_admission_policy != "allocation_free"
            or protocol_sha256 is not None
        ):
            raise ValueError(f"{field} allocation-free reset identity differs")
        return None
    if method == "tts":
        expected_pairs = {("request", "serialized_native_scheduler_v1")}
    elif method.startswith("onlinespec_"):
        expected_pairs = {("cohort", "cohort_batching_v1")}
    elif method == "l0":
        expected_pairs = {
            ("request", "serialized_native_scheduler_v1"),
            ("cohort", "cohort_batching_v1"),
        }
    else:  # pragma: no cover - callers validate the method first
        raise ValueError(f"{field} method is unsupported")
    if (reset_scope, request_admission_policy) not in expected_pairs:
        raise ValueError(f"{field} reset scope/admission identity differs")
    if protocol_sha256 != REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256:
        raise ValueError(f"{field} request reset protocol differs from the pin")
    return REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256


def _identity_values(
    binding: NativeTerminalRunBinding,
    *,
    legacy: bool = False,
) -> dict[str, object]:
    values: dict[str, object] = {
        "run_id": binding.run_id,
        "run_nonce_sha256": binding.run_nonce_sha256,
        "execution_plan_sha256": binding.execution_plan_sha256,
        "rank_config_sha256": binding.rank_config_sha256,
        "attempt_id": binding.attempt_id,
        "session_id": binding.session_id,
        "session_epoch": binding.session_epoch,
        "previous_run_id": binding.previous_run_id,
        "challenge_nonce_sha256": binding.challenge_nonce_sha256,
        "method": binding.method,
    }
    if not legacy:
        reset_scope, request_admission_policy = _wire_reset_identity(binding)
        runtime_trust_mode, formal_measurement = _validate_runtime_trust_identity(
            method=binding.method,
            runtime_trust_mode=binding.runtime_trust_mode,
            formal_measurement=binding.formal_measurement,
            field="native terminal binding",
        )
        values.update(
            {
                "reset_scope": reset_scope,
                "request_admission_policy": request_admission_policy,
                "runtime_trust_mode": runtime_trust_mode,
                "formal_measurement": formal_measurement,
            }
        )
    return values


def _validate_bound_identity(
    value: Mapping[str, object],
    binding: NativeTerminalRunBinding,
    field: str,
    *,
    legacy: bool = False,
) -> None:
    expected = _identity_values(binding, legacy=legacy)
    if any(
        value.get(name) != expected_value for name, expected_value in expected.items()
    ):
        raise ValueError(f"{field} run/session identity mismatch")


@dataclass(frozen=True)
class NativeTerminalRunBinding:
    run_id: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    attempt_id: str
    session_id: str
    session_epoch: int
    previous_run_id: str | None
    challenge_nonce_sha256: str
    method: str
    reset_scope: str | None
    request_admission_policy: str | None
    runtime_trust_mode: str | None
    formal_measurement: bool | None
    warmup_request_ids: tuple[str, ...]
    scored_request_ids: tuple[str, ...]

    def validate(self) -> None:
        for name in ("run_id", "attempt_id", "session_id"):
            _safe_id(getattr(self, name), name)
        _safe_id(self.previous_run_id, "previous_run_id", nullable=True)
        for name in (
            "run_nonce_sha256",
            "execution_plan_sha256",
            "rank_config_sha256",
            "challenge_nonce_sha256",
        ):
            _sha256(getattr(self, name), name)
        _integer(self.session_epoch, "session_epoch", minimum=1)
        if self.method not in SUPPORTED_METHODS:
            raise ValueError("terminal method is unsupported")
        reset_scope, request_admission_policy = _wire_reset_identity(self)
        runtime_trust_mode, formal_measurement = _validate_runtime_trust_identity(
            method=self.method,
            runtime_trust_mode=self.runtime_trust_mode,
            formal_measurement=self.formal_measurement,
            field="native terminal binding",
        )
        if self.method in {"target_only", "static"}:
            if (reset_scope, request_admission_policy) != (
                "none",
                "allocation_free",
            ) or (runtime_trust_mode, formal_measurement) != (None, None):
                raise ValueError("allocation-free terminal has adaptation reset state")
        elif self.method == "tts":
            if (reset_scope, request_admission_policy) != (
                "request",
                "serialized_native_scheduler_v1",
            ):
                raise ValueError("TTS terminal requires request-scoped reset identity")
        elif self.method.startswith("onlinespec_"):
            if (reset_scope, request_admission_policy) != (
                "cohort",
                "cohort_batching_v1",
            ):
                raise ValueError("OnlineSPEC terminal requires cohort reset identity")
        elif (reset_scope, request_admission_policy) not in {
            ("request", "serialized_native_scheduler_v1"),
            ("cohort", "cohort_batching_v1"),
        }:
            raise ValueError("L0 terminal reset identity is incomplete")
        if not self.scored_request_ids:
            raise ValueError("terminal binding requires scored request IDs")
        for field, request_ids in (
            ("warmup_request_ids", self.warmup_request_ids),
            ("scored_request_ids", self.scored_request_ids),
        ):
            for request_id in request_ids:
                _safe_id(request_id, field)
            if len(request_ids) != len(set(request_ids)):
                raise ValueError(f"{field} contains duplicate request IDs")
        if set(self.warmup_request_ids) & set(self.scored_request_ids):
            raise ValueError("warmup and scored request IDs must be disjoint")

    def begin_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": 2,
            "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
            **_identity_values(self),
            "warmup_request_ids": list(self.warmup_request_ids),
            "scored_request_ids": list(self.scored_request_ids),
        }


def _validate_legacy_run_binding(binding: NativeTerminalRunBinding) -> None:
    """Validate a read-only v1 binding whose reset identity was never emitted."""

    if any(
        value is not None
        for value in (
            binding.reset_scope,
            binding.request_admission_policy,
            binding.runtime_trust_mode,
            binding.formal_measurement,
        )
    ):
        raise ValueError("legacy terminal binding cannot invent reset identity")
    for name in ("run_id", "attempt_id", "session_id"):
        _safe_id(getattr(binding, name), name)
    _safe_id(binding.previous_run_id, "previous_run_id", nullable=True)
    for name in (
        "run_nonce_sha256",
        "execution_plan_sha256",
        "rank_config_sha256",
        "challenge_nonce_sha256",
    ):
        _sha256(getattr(binding, name), name)
    _integer(binding.session_epoch, "session_epoch", minimum=1)
    if binding.method not in SUPPORTED_METHODS or not binding.scored_request_ids:
        raise ValueError("legacy terminal method/request identity is unsupported")
    for field, request_ids in (
        ("warmup_request_ids", binding.warmup_request_ids),
        ("scored_request_ids", binding.scored_request_ids),
    ):
        for request_id in request_ids:
            _safe_id(request_id, field)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError(f"{field} contains duplicate request IDs")
    if set(binding.warmup_request_ids) & set(binding.scored_request_ids):
        raise ValueError("warmup and scored request IDs must be disjoint")


@dataclass(frozen=True)
class TerminalRequestExpectation:
    """Caller knowledge used to verify one server or non-submitted terminal row."""

    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...] | None
    terminal_status: str
    terminal_reason: str
    submitted_to_server: bool

    def validate(self) -> None:
        _safe_id(self.request_id, "request_id")
        _token_ids(self.input_token_ids, "input_token_ids")
        _safe_id(self.terminal_reason, "terminal_reason")
        if not isinstance(self.submitted_to_server, bool):
            raise TypeError("submitted_to_server must be boolean")
        if self.submitted_to_server:
            if self.terminal_status not in {"completed", "aborted"}:
                raise ValueError("submitted request has an invalid server status")
            if self.output_token_ids is None:
                raise ValueError("submitted request requires exact output token IDs")
            _token_ids(self.output_token_ids, "output_token_ids")
        else:
            if self.terminal_status not in {
                "rejected",
                "cancelled",
                "timed_out",
                "unfinished",
            }:
                raise ValueError("non-submitted request has an invalid client status")
            if self.output_token_ids is not None:
                raise ValueError("non-submitted terminal rows cannot carry output IDs")


@dataclass(frozen=True)
class UnsignedNativeServingPhaseResult:
    """Untrusted result of executing one exact input-only serving phase.

    The remote producer has no release key.  It returns exact terminal rows
    only after generation and preserves each first-party native ITL pointer as
    canonical JSON.  Validation here prevents input/output mixups; formal
    authority still requires the durable local external-control proof chain.
    """

    phase: str
    requests: tuple[TerminalRequestExpectation, ...]
    native_result_pointer_json: tuple[str, ...]
    client_lifecycle_rows: tuple[dict[str, object], ...] = ()

    def validate(
        self,
        *,
        expected_phase: str,
        bound_requests: tuple[BoundServingRequest, ...],
    ) -> tuple[dict[str, object], ...]:
        if self.phase != expected_phase or expected_phase not in {"warmup", "scored"}:
            raise ValueError("unsigned serving phase identity differs")
        if type(self.requests) is not tuple:
            raise TypeError("unsigned serving phase requires exact terminal rows")
        if tuple(row.request_id for row in self.requests) != tuple(
            row.request_id for row in bound_requests
        ):
            raise ValueError("unsigned serving terminal order differs from inputs")
        if self.client_lifecycle_rows and (
            type(self.client_lifecycle_rows) is not tuple
            or any(type(row) is not dict for row in self.client_lifecycle_rows)
            or tuple(row.get("request_id") for row in self.client_lifecycle_rows)
            != tuple(row.request_id for row in bound_requests)
        ):
            raise ValueError("unsigned serving client lifecycle coverage differs")
        for bound, observed in zip(bound_requests, self.requests, strict=True):
            if type(observed) is not TerminalRequestExpectation:
                raise TypeError("unsigned serving terminal rows must be typed")
            observed.validate()
            if (
                observed.request_id != bound.request_id
                or observed.input_token_ids != bound.input_token_ids
            ):
                raise ValueError("unsigned serving terminal input identity changed")
        pointers = tuple(
            _validate_unsigned_native_result_pointer(value)
            for value in self.native_result_pointer_json
        )
        by_request = {str(value["request_id"]): value for value in pointers}
        if len(by_request) != len(pointers):
            raise ValueError("unsigned native ITL pointers contain duplicates")
        expected_pointer_ids = {
            row.request_id
            for row in self.requests
            if row.submitted_to_server and row.terminal_status == "completed"
        }
        if set(by_request) != expected_pointer_ids:
            raise ValueError("unsigned native ITL pointer coverage is incomplete")
        for row in self.requests:
            pointer = by_request.get(row.request_id)
            if pointer is None:
                continue
            events = pointer["events"]
            assert isinstance(events, list)
            if (
                pointer["terminal_status"] != row.terminal_status
                or row.output_token_ids is None
                or tuple(event["token_id"] for event in events) != row.output_token_ids
            ):
                raise ValueError(
                    "unsigned native ITL pointer differs from terminal output"
                )
        return pointers


@dataclass(frozen=True)
class UnsignedNativeLifecycleEvents:
    """Source-owned monotonic phase edges for one unsigned lifecycle."""

    begin_started_ns: int
    begin_finished_ns: int
    warmup_started_ns: int
    warmup_finished_ns: int
    reset_started_ns: int
    reset_finished_ns: int
    scored_started_ns: int
    scored_finished_ns: int
    finalize_started_ns: int
    finalize_finished_ns: int
    terminal_published_ns: int
    itl_pointer_published_ns: int

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if any(type(value) is not int or value < 1 for value in values) or values != (
            tuple(sorted(values))
        ):
            raise ValueError("unsigned native lifecycle timestamps are not ordered")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class UnsignedNativeTerminalCollection:
    """Path-bound remote outputs; neither binding is formal authority."""

    terminal_artifact: CanonicalJsonProofBinding
    native_itl_pointer_artifact: CanonicalJsonProofBinding
    lifecycle_events: UnsignedNativeLifecycleEvents | None = None

    def __post_init__(self) -> None:
        if (
            type(self.terminal_artifact) is not CanonicalJsonProofBinding
            or type(self.native_itl_pointer_artifact) is not CanonicalJsonProofBinding
            or (
                self.lifecycle_events is not None
                and type(self.lifecycle_events) is not UnsignedNativeLifecycleEvents
            )
        ):
            raise TypeError("unsigned native collection requires exact bindings")


@dataclass(frozen=True)
class UnsignedNativeItlTokenEvent:
    token_index: int
    token_id: int
    observed_ns: int


_VALIDATED_UNSIGNED_ITL_BUNDLE_SENTINEL = object()


@dataclass(frozen=True, init=False)
class ValidatedUnsignedNativeItlPointer:
    """One first-party pointer reopened from the path-bound unsigned bundle."""

    request_id: str
    request_started_ns: int
    request_terminal_ns: int
    terminal_status: str
    terminal_reason: str
    events: tuple[UnsignedNativeItlTokenEvent, ...]
    result_pointer_sha256: str

    def __init__(
        self,
        *,
        request_id: str,
        request_started_ns: int,
        request_terminal_ns: int,
        terminal_status: str,
        terminal_reason: str,
        events: tuple[UnsignedNativeItlTokenEvent, ...],
        result_pointer_sha256: str,
        _verification_tag: object = None,
    ) -> None:
        if _verification_tag is not _VALIDATED_UNSIGNED_ITL_BUNDLE_SENTINEL:
            raise TypeError("unsigned native ITL pointer requires path validation")
        for name, value in locals().copy().items():
            if name not in {"self", "_verification_tag"}:
                object.__setattr__(self, name, value)


@dataclass(frozen=True, init=False)
class ValidatedUnsignedNativeItlPointerBundle:
    """Verifier-owned view of a collector bundle; never formal authority."""

    artifact_raw_sha256: str
    artifact_semantic_sha256: str
    run_binding_sha256: str
    terminal_artifact_raw_sha256: str
    terminal_artifact_semantic_sha256: str
    scored_request_inputs_sha256: str
    pointers: tuple[ValidatedUnsignedNativeItlPointer, ...]

    def __init__(
        self,
        *,
        artifact_raw_sha256: str,
        artifact_semantic_sha256: str,
        run_binding_sha256: str,
        terminal_artifact_raw_sha256: str,
        terminal_artifact_semantic_sha256: str,
        scored_request_inputs_sha256: str,
        pointers: tuple[ValidatedUnsignedNativeItlPointer, ...],
        _verification_tag: object = None,
    ) -> None:
        if _verification_tag is not _VALIDATED_UNSIGNED_ITL_BUNDLE_SENTINEL:
            raise TypeError("unsigned native ITL bundle requires path validation")
        for name, value in locals().copy().items():
            if name not in {"self", "_verification_tag"}:
                object.__setattr__(self, name, value)


@dataclass(frozen=True)
class NativeTerminalCapability:
    active_method: str
    enabled: bool
    method_evidence_supported: bool
    topology_supported: bool
    trusted_attester_configured: bool
    required_fields: tuple[str, ...]
    supported_methods: tuple[str, ...]
    reset_scope: str
    request_admission_policy: str
    request_source_point_reset_protocol_sha256: str | None
    runtime_trust_mode: str | None
    formal_measurement: bool | None
    raw_json: str


@dataclass(frozen=True)
class NativeTerminalBeginReceipt:
    binding: NativeTerminalRunBinding
    server_process_id: int
    server_process_started_ns: int
    reset_generation: int
    request_source_point_reset_protocol_sha256: str | None
    runtime_trust_mode: str | None
    formal_measurement: bool | None
    begin_sha256: str
    raw_json: str


@dataclass(frozen=True)
class NativeTerminalResetReceipt:
    binding: NativeTerminalRunBinding
    server_process_id: int
    server_process_started_ns: int
    reset_generation: int
    completion_event_generation: int
    request_source_point_reset_protocol_sha256: str | None
    runtime_trust_mode: str | None
    formal_measurement: bool | None
    reset_sha256: str
    raw_json: str
    warmup_requests: tuple[TerminalRequestExpectation, ...] = ()
    warmup_request_source_point_resets: NativeRequestSourcePointResets | None = None


@dataclass(frozen=True)
class NativeRequestSourcePointResetReceipt:
    request_id: str
    request_epoch: int
    terminal_outcome: str
    terminal_round: int
    terminal_version: int
    adaptation_state_acquired: bool
    reset_required: bool
    state_untouched: bool
    source_point_identity_sha256: str
    master_reset: bool | None
    optimizer_reset: bool | None
    inference_reset: bool | None
    captured_state_empty: bool | None
    runtime_reset: bool | None
    sticky_disabled_reason: str | None
    evidence_archive_sha256: str
    archived_update_count: int
    archived_round_count: int
    previous_receipt_sha256: str
    protocol_sha256: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class NativeRequestSourcePointResets:
    reset_scope: str
    request_admission_policy: str
    protocol_sha256: str | None
    final_archive_sha256: str
    receipts: tuple[NativeRequestSourcePointResetReceipt, ...]

    @property
    def receipt_by_sha256(self) -> dict[str, NativeRequestSourcePointResetReceipt]:
        return {row.receipt_sha256: row for row in self.receipts}

    @property
    def receipt_by_request_id(self) -> dict[str, NativeRequestSourcePointResetReceipt]:
        return {row.request_id: row for row in self.receipts}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reset_scope": self.reset_scope,
            "request_admission_policy": self.request_admission_policy,
            "protocol_sha256": self.protocol_sha256,
            "final_archive_sha256": self.final_archive_sha256,
            "receipts": [row.to_dict() for row in self.receipts],
        }


@dataclass(frozen=True)
class NativeTerminalAttestation:
    status: str
    challenge_nonce_sha256: str
    message_sha256: str
    attester_id: str | None
    trust_domain: str | None
    signature_hex: str | None
    key_id: str | None
    public_key_sha256: str | None
    trusted_attester_policy_sha256: str
    trusted: bool


_VALIDATED_NATIVE_TERMINAL_SENTINEL = object()


@dataclass(frozen=True, init=False)
class ValidatedNativeTerminalEvidence:
    """Immutable validated terminal envelope.

    ``raw_json`` is canonical and callers receive a new object from
    :meth:`to_dict`, so post-validation mutation cannot alter the receipt.
    """

    binding: NativeTerminalRunBinding
    begin_receipt: NativeTerminalBeginReceipt
    reset_receipt: NativeTerminalResetReceipt
    requests: tuple[TerminalRequestExpectation, ...]
    attestation: NativeTerminalAttestation
    terminal_sha256: str
    raw_json: str
    external_control_binding_sha256: str | None
    external_control_envelope_sha256: str | None
    external_control_reservation_sha256: str | None
    external_control_trusted_policy_sha256: str | None

    def __init__(
        self,
        *,
        binding: NativeTerminalRunBinding,
        begin_receipt: NativeTerminalBeginReceipt,
        reset_receipt: NativeTerminalResetReceipt,
        requests: tuple[TerminalRequestExpectation, ...],
        attestation: NativeTerminalAttestation,
        terminal_sha256: str,
        raw_json: str,
        _verification_tag: object,
        external_control_binding_sha256: str | None = None,
        external_control_envelope_sha256: str | None = None,
        external_control_reservation_sha256: str | None = None,
        external_control_trusted_policy_sha256: str | None = None,
    ) -> None:
        if _verification_tag is not _VALIDATED_NATIVE_TERMINAL_SENTINEL:
            raise TypeError(
                "validated native terminal evidence requires first-party validation"
            )
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "begin_receipt", begin_receipt)
        object.__setattr__(self, "reset_receipt", reset_receipt)
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "attestation", attestation)
        object.__setattr__(self, "terminal_sha256", terminal_sha256)
        object.__setattr__(self, "raw_json", raw_json)
        control_fields = (
            external_control_binding_sha256,
            external_control_envelope_sha256,
            external_control_reservation_sha256,
            external_control_trusted_policy_sha256,
        )
        if any(value is None for value in control_fields) and any(
            value is not None for value in control_fields
        ):
            raise ValueError("native terminal external control fields are atomic")
        for label, value in zip(
            (
                "native terminal control binding",
                "native terminal control envelope",
                "native terminal control reservation",
                "native terminal control policy",
            ),
            control_fields,
            strict=True,
        ):
            if value is not None:
                _sha256(value, label)
        object.__setattr__(
            self,
            "external_control_binding_sha256",
            external_control_binding_sha256,
        )
        object.__setattr__(
            self,
            "external_control_envelope_sha256",
            external_control_envelope_sha256,
        )
        object.__setattr__(
            self,
            "external_control_reservation_sha256",
            external_control_reservation_sha256,
        )
        object.__setattr__(
            self,
            "external_control_trusted_policy_sha256",
            external_control_trusted_policy_sha256,
        )

    @property
    def trusted_attestation(self) -> bool:
        return self.attestation.trusted or (
            self.external_control_envelope_sha256 is not None
        )

    @property
    def authority_kind(self) -> str:
        if self.external_control_envelope_sha256 is not None:
            return "external_release_control"
        if self.attestation.trusted:
            return "native_hardware_attestation"
        return "untrusted_raw_terminal"

    @property
    def trusted_attester_policy_sha256(self) -> str:
        return (
            self.external_control_trusted_policy_sha256
            or self.attestation.trusted_attester_policy_sha256
        )

    def to_dict(self) -> dict[str, object]:
        value = json.loads(self.raw_json)
        if not isinstance(value, dict):  # pragma: no cover - construction invariant
            raise TypeError("validated terminal JSON stopped being an object")
        return value

    @property
    def terminal_schema_version(self) -> int:
        value = self.to_dict().get("schema_version")
        if type(value) is not int:
            raise TypeError("validated terminal schema version is malformed")
        return value

    @property
    def request_source_point_resets(self) -> NativeRequestSourcePointResets | None:
        if self.terminal_schema_version == 1:
            return None
        value = self.to_dict().get("request_source_point_resets")
        if type(value) is not dict:  # validated current terminal invariant
            raise TypeError("validated request reset evidence is malformed")
        return _request_source_point_resets_from_validated(value)

    def to_artifact(
        self,
        *,
        warmup_requests: Sequence[TerminalRequestExpectation],
        rank: int = 0,
    ) -> dict[str, object]:
        """Return one canonicalizable bundle for durable terminal publication."""

        if self.terminal_schema_version != 2:
            raise RuntimeError("legacy native terminal evidence is read-only")
        warmup = _validate_request_expectations(
            warmup_requests,
            expected_ids=self.binding.warmup_request_ids,
            warmup=True,
        )
        scored = _validate_request_expectations(
            self.requests,
            expected_ids=self.binding.scored_request_ids,
            warmup=False,
        )
        artifact: dict[str, object] = {
            "schema_version": 2,
            "artifact_kind": NATIVE_TERMINAL_ARTIFACT_KIND,
            "run_id": self.binding.run_id,
            "rank": rank,
            "trusted_attester_policy_sha256": (self.trusted_attester_policy_sha256),
            "begin_sha256": self.begin_receipt.begin_sha256,
            "reset_sha256": self.reset_receipt.reset_sha256,
            "terminal_sha256": self.terminal_sha256,
            "binding": _binding_artifact(self.binding),
            "warmup_requests": [
                _request_expectation_artifact(request) for request in warmup
            ],
            "scored_requests": [
                _request_expectation_artifact(request) for request in scored
            ],
            "begin": json.loads(self.begin_receipt.raw_json),
            "reset": json.loads(self.reset_receipt.raw_json),
            "terminal": self.to_dict(),
        }
        _validate_strict_json(artifact, "native terminal artifact")
        return artifact

    def to_native_evidence_batch(self):
        """Convert exact native rows without imputing absent per-update timing.

        Imports are local so importing this module never creates an executor
        cycle.  Aggregate CUDA timing remains in performance overrides; it is
        deliberately not copied into individual update rows.
        """

        from lightcone_spec.orchestration.executor import NativeEvidenceBatch
        from lightcone_spec.telemetry.records import RoundRecord, UpdateRecord

        envelope = self.to_dict()
        request_round_rows = envelope["request_round_rows"]
        if not isinstance(request_round_rows, dict):  # validated above
            raise TypeError("validated request/round container is malformed")
        performance = envelope["performance_counters"]
        if not isinstance(performance, dict):
            raise TypeError("validated performance container is malformed")
        input_lengths = {
            request.request_id: len(request.input_token_ids)
            for request in self.requests
        }
        current = self.terminal_schema_version == 2
        historical_kv = envelope["historical_kv_source_versions"]
        if not isinstance(historical_kv, dict):  # validated above
            raise TypeError("validated historical KV container is malformed")
        rounds = tuple(
            RoundRecord(
                run_id=self.binding.run_id,
                request_id=str(row["request_id"]),
                round_index=int(row["round_index"]),
                generated_tokens_before=(
                    int(row["prefix_len_before"])
                    - input_lengths[str(row["request_id"])]
                ),
                prefix_len_before=int(row["prefix_len_before"]),
                verify_len=int(row["verify_len"]),
                accepted_drafts=int(row["accepted_drafts"]),
                committed_tokens=int(row["committed_tokens"]),
                target_calls=int(row["target_calls"]),
                proposal_source_version=int(row["proposal_source_version"]),
                kv_source_versions=_canonical_json_text(
                    historical_kv[str(row["request_id"])]
                    if current
                    else row["historical_kv_source_versions"]
                ),
                reset_scope=(str(row["reset_scope"]) if current else None),
                request_epoch=(int(row["request_epoch"]) if current else None),
                request_reset_receipt_sha256=(
                    None
                    if not current or row["request_reset_receipt_sha256"] is None
                    else str(row["request_reset_receipt_sha256"])
                ),
                historical_kv_source_versions_sha256=(
                    str(row["historical_kv_source_versions_sha256"])
                    if current
                    else None
                ),
            )
            for row in request_round_rows["rounds"]
        )
        trainable_parameters = performance["trainable_parameters"]
        updates = tuple(
            UpdateRecord(
                run_id=self.binding.run_id,
                cohort_sha256=str(row["cohort_sha256"]),
                parameter_layout_sha256=str(row["parameter_layout_sha256"]),
                update_index=int(row["update_index"]),
                request_ids=_canonical_json_text(row["request_ids"]),
                prefix_len_before=_canonical_json_text(row["prefix_len_before"]),
                prefix_len_min=min(row["prefix_len_before"]),
                prefix_len_max=max(row["prefix_len_before"]),
                prefix_len_mean=(
                    sum(row["prefix_len_before"]) / len(row["prefix_len_before"])
                ),
                source_round=int(row["source_round"]),
                source_version=int(row["source_version"]),
                optimizer_step=int(row["optimizer_step"]),
                published_version=(
                    None
                    if row["published_version"] is None
                    else int(row["published_version"])
                ),
                candidate_status=str(row["status"]),
                loss=float(row["loss"]),
                gradient_norm=float(row["gradient_norm"]),
                reconstruction_ok=bool(row["reconstruction_ok"]),
                reconstruction_max_abs=float(row["reconstruction_max_abs"]),
                reconstruction_relative_rms=(
                    None
                    if row["reconstruction_relative_rms"] is None
                    else float(row["reconstruction_relative_rms"])
                ),
                reconstruction_top1_match=(
                    None
                    if row["reconstruction_top1_match"] is None
                    else float(row["reconstruction_top1_match"])
                ),
                reconstruction_mean_kl=(
                    None
                    if row["reconstruction_mean_kl"] is None
                    else float(row["reconstruction_mean_kl"])
                ),
                supervision_nonempty=bool(row["supervision_nonempty"]),
                trainable_parameters=int(trainable_parameters),
                training_cuda_ms=None,
                optimizer_cuda_ms=None,
                merge_cuda_ms=None,
                publish_cuda_ms=None,
                barrier_cuda_ms=None,
                exposed_update_ms=None,
                overlap_ratio=None,
                online_hint_error=(
                    None
                    if row["online_hint_error"] is None
                    else float(row["online_hint_error"])
                ),
                online_ensemble_entropy=(
                    None
                    if row["online_ensemble_entropy"] is None
                    else float(row["online_ensemble_entropy"])
                ),
                online_effective_experts=(
                    None
                    if row["online_effective_experts"] is None
                    else float(row["online_effective_experts"])
                ),
                online_expert_probabilities=(
                    None
                    if row["online_expert_probabilities"] is None
                    else _canonical_json_text(row["online_expert_probabilities"])
                ),
                online_cumulative_losses=(
                    None
                    if row["online_cumulative_losses"] is None
                    else _canonical_json_text(row["online_cumulative_losses"])
                ),
                online_expert_gradient_norms=(
                    None
                    if row["online_expert_gradient_norms"] is None
                    else _canonical_json_text(row["online_expert_gradient_norms"])
                ),
                source_state_sha256=(
                    None
                    if row["source_state_sha256"] is None
                    else str(row["source_state_sha256"])
                ),
                candidate_bytes_sha256=(
                    None
                    if row["candidate_bytes_sha256"] is None
                    else str(row["candidate_bytes_sha256"])
                ),
                optimizer_state_bytes_sha256=(
                    None
                    if row["optimizer_state_bytes_sha256"] is None
                    else str(row["optimizer_state_bytes_sha256"])
                ),
                proposal_evidence_sha256=(
                    None
                    if row["proposal_evidence_sha256"] is None
                    else str(row["proposal_evidence_sha256"])
                ),
                cohort_epoch=int(row["cohort_epoch"]),
                exactness_violation=row["status"] == "reconstruction_mismatch",
                stale_candidate=row["status"] == "version_conflict",
                nonfinite_candidate=row["status"] == "nonfinite_update",
                reset_scope=(str(row["reset_scope"]) if current else None),
                request_epoch=(int(row["request_epoch"]) if current else None),
                request_reset_receipt_sha256=(
                    None
                    if not current or row["request_reset_receipt_sha256"] is None
                    else str(row["request_reset_receipt_sha256"])
                ),
                effective_learning_rate=(
                    float(row["effective_learning_rate"]) if current else None
                ),
                schedule_valid=(bool(row["schedule_valid"]) if current else None),
                intrinsic_ready_round=(
                    int(row["intrinsic_ready_round"])
                    if current and row["intrinsic_ready_round"] is not None
                    else None
                ),
                extra_logical_delay=(
                    int(row["extra_logical_delay"]) if current else None
                ),
                publication_round=(
                    int(row["publication_round"])
                    if current and row["publication_round"] is not None
                    else None
                ),
            )
            for row in envelope["update_rows"]
        )
        overrides = []
        for key in _PERFORMANCE_OVERRIDE_KEYS:
            value = performance[key]
            if key == "adaptation_memory_ledger" and value is not None:
                value = value if isinstance(value, str) else _canonical_json_text(value)
            overrides.append((key, value))
        batch = NativeEvidenceBatch(
            rounds=rounds,
            updates=updates,
            performance_overrides=tuple(overrides),
        )
        batch.validate(run_id=self.binding.run_id, method=self.binding.method)
        return batch


@dataclass(frozen=True)
class NativeTerminalExternalControlBinding:
    """Typed content identity signed only after the raw terminal is pulled.

    The remote SGLang process never receives the offline release key.  It emits
    the canonical terminal bundle, while this binding commits both its exact
    canonical bytes and its semantic object identity before a local control
    attestation can authorize downstream use.
    """

    schema_version: int
    kind: str
    canonical_raw_sha256: str
    semantic_artifact_sha256: str
    terminal_sha256: str
    run_id: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    attempt_id: str
    session_id: str
    session_epoch: int
    method: str
    reset_scope: str | None
    request_admission_policy: str | None
    runtime_trust_mode: str | None
    formal_measurement: bool | None
    inventory_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.kind != (
            "native_terminal_external_control_binding"
        ):
            raise ValueError("native terminal control binding schema is unsupported")
        for label, value in (
            ("terminal canonical raw", self.canonical_raw_sha256),
            ("terminal semantic artifact", self.semantic_artifact_sha256),
            ("terminal semantic receipt", self.terminal_sha256),
            ("terminal run nonce", self.run_nonce_sha256),
            ("terminal execution plan", self.execution_plan_sha256),
            ("terminal rank config", self.rank_config_sha256),
            ("terminal inventory", self.inventory_sha256),
        ):
            _sha256(value, label)
        for label, value in (
            ("terminal run", self.run_id),
            ("terminal attempt", self.attempt_id),
            ("terminal session", self.session_id),
            ("terminal method", self.method),
        ):
            _safe_id(value, label)
        _integer(self.session_epoch, "terminal session epoch", minimum=1)
        if self.method not in SUPPORTED_METHODS:
            raise ValueError("native terminal control method is unsupported")
        probe = NativeTerminalRunBinding(
            run_id=self.run_id,
            run_nonce_sha256=self.run_nonce_sha256,
            execution_plan_sha256=self.execution_plan_sha256,
            rank_config_sha256=self.rank_config_sha256,
            attempt_id=self.attempt_id,
            session_id=self.session_id,
            session_epoch=self.session_epoch,
            previous_run_id=None,
            challenge_nonce_sha256=_ZERO_SHA256,
            method=self.method,
            reset_scope=self.reset_scope,
            request_admission_policy=self.request_admission_policy,
            runtime_trust_mode=self.runtime_trust_mode,
            formal_measurement=self.formal_measurement,
            warmup_request_ids=(),
            scored_request_ids=("identity-probe",),
        )
        probe.validate()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "canonical_raw_sha256": self.canonical_raw_sha256,
            "semantic_artifact_sha256": self.semantic_artifact_sha256,
            "terminal_sha256": self.terminal_sha256,
            "run_id": self.run_id,
            "run_nonce_sha256": self.run_nonce_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "attempt_id": self.attempt_id,
            "session_id": self.session_id,
            "session_epoch": self.session_epoch,
            "method": self.method,
            "reset_scope": self.reset_scope,
            "request_admission_policy": self.request_admission_policy,
            "runtime_trust_mode": self.runtime_trust_mode,
            "formal_measurement": self.formal_measurement,
            "inventory_sha256": self.inventory_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def lineage_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema_version": 2,
                "kind": "native_terminal_external_control_lineage",
                "terminal_binding_sha256": self.sha256,
                "run_id": self.run_id,
                "run_nonce_sha256": self.run_nonce_sha256,
                "execution_plan_sha256": self.execution_plan_sha256,
                "rank_config_sha256": self.rank_config_sha256,
                "attempt_id": self.attempt_id,
                "session_id": self.session_id,
                "session_epoch": self.session_epoch,
                "method": self.method,
                "reset_scope": self.reset_scope,
                "request_admission_policy": self.request_admission_policy,
                "terminal_sha256": self.terminal_sha256,
                "inventory_sha256": self.inventory_sha256,
            }
        )


_PREPARED_NATIVE_TERMINAL_CONTROL_SENTINEL = object()


@dataclass(frozen=True, init=False)
class PreparedNativeTerminalExternalControl:
    """Structurally checked batch row with no replay-store mutation."""

    evidence: ValidatedNativeTerminalEvidence
    binding: NativeTerminalExternalControlBinding
    control_attestation: ControlArtifactAttestation
    expected_inventory_sha256: str
    expected_registry_sha256: str

    def __init__(
        self,
        *,
        evidence: ValidatedNativeTerminalEvidence,
        binding: NativeTerminalExternalControlBinding,
        control_attestation: ControlArtifactAttestation,
        expected_inventory_sha256: str,
        expected_registry_sha256: str,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _PREPARED_NATIVE_TERMINAL_CONTROL_SENTINEL:
            raise TypeError("prepared native terminal control requires validation")
        if type(evidence) is not ValidatedNativeTerminalEvidence:
            raise TypeError("prepared native terminal control evidence is not exact")
        if type(binding) is not NativeTerminalExternalControlBinding:
            raise TypeError("prepared native terminal control binding is not exact")
        if type(control_attestation) is not ControlArtifactAttestation:
            raise TypeError("prepared native terminal control envelope is not exact")
        _sha256(expected_inventory_sha256, "prepared terminal inventory")
        _sha256(expected_registry_sha256, "prepared terminal registry")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "control_attestation", control_attestation)
        object.__setattr__(self, "expected_inventory_sha256", expected_inventory_sha256)
        object.__setattr__(self, "expected_registry_sha256", expected_registry_sha256)


_CANDIDATE_STATE_POINTER_SENTINEL = object()


@dataclass(frozen=True, init=False)
class CandidateStateByteIdentity:
    """One source-owned mechanism-replay update byte identity."""

    update_index: int
    reset_scope: str
    request_epoch: int
    request_reset_receipt_sha256: str | None
    source_round: int
    source_version: int
    request_ids: tuple[str, ...]
    source_state_sha256: str
    candidate_bytes_sha256: str
    optimizer_state_bytes_sha256: str
    proposal_evidence_sha256: str
    update_sha256: str

    def __init__(
        self,
        *,
        update_index: int,
        reset_scope: str,
        request_epoch: int,
        request_reset_receipt_sha256: str | None,
        source_round: int,
        source_version: int,
        request_ids: tuple[str, ...],
        source_state_sha256: str,
        candidate_bytes_sha256: str,
        optimizer_state_bytes_sha256: str,
        proposal_evidence_sha256: str,
        update_sha256: str,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _CANDIDATE_STATE_POINTER_SENTINEL:
            raise TypeError(
                "candidate-state byte identity requires validated native evidence"
            )
        for name, value in (
            ("source_state_sha256", source_state_sha256),
            ("candidate_bytes_sha256", candidate_bytes_sha256),
            ("optimizer_state_bytes_sha256", optimizer_state_bytes_sha256),
            ("proposal_evidence_sha256", proposal_evidence_sha256),
            ("update_sha256", update_sha256),
        ):
            _sha256(value, name)
        if not request_ids or len(request_ids) != len(set(request_ids)):
            raise ValueError("candidate-state request coverage is not unique")
        for request_id in request_ids:
            _safe_id(request_id, "candidate_state.request_id")
        for name, value, minimum in (
            ("update_index", update_index, 0),
            ("request_epoch", request_epoch, 0),
            ("source_round", source_round, 1),
            ("source_version", source_version, 0),
        ):
            _integer(value, name, minimum=minimum)
        object.__setattr__(self, "update_index", update_index)
        if reset_scope not in {"request", "cohort"}:
            raise ValueError("candidate-state reset scope is invalid")
        if reset_scope == "request":
            _sha256(
                request_reset_receipt_sha256,
                "candidate-state request reset receipt",
            )
            if request_epoch < 1:
                raise ValueError("candidate-state request epoch is invalid")
        elif request_epoch != 0 or request_reset_receipt_sha256 is not None:
            raise ValueError("cohort candidate-state carries request reset identity")
        object.__setattr__(self, "reset_scope", reset_scope)
        object.__setattr__(self, "request_epoch", request_epoch)
        object.__setattr__(
            self,
            "request_reset_receipt_sha256",
            request_reset_receipt_sha256,
        )
        object.__setattr__(self, "source_round", source_round)
        object.__setattr__(self, "source_version", source_version)
        object.__setattr__(self, "request_ids", request_ids)
        object.__setattr__(self, "source_state_sha256", source_state_sha256)
        object.__setattr__(self, "candidate_bytes_sha256", candidate_bytes_sha256)
        object.__setattr__(
            self, "optimizer_state_bytes_sha256", optimizer_state_bytes_sha256
        )
        object.__setattr__(self, "proposal_evidence_sha256", proposal_evidence_sha256)
        object.__setattr__(self, "update_sha256", update_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "update_index": self.update_index,
            "reset_scope": self.reset_scope,
            "request_epoch": self.request_epoch,
            "request_reset_receipt_sha256": self.request_reset_receipt_sha256,
            "source_round": self.source_round,
            "source_version": self.source_version,
            "request_ids": list(self.request_ids),
            "source_state_sha256": self.source_state_sha256,
            "candidate_bytes_sha256": self.candidate_bytes_sha256,
            "optimizer_state_bytes_sha256": self.optimizer_state_bytes_sha256,
            "proposal_evidence_sha256": self.proposal_evidence_sha256,
            "update_sha256": self.update_sha256,
        }


@dataclass(frozen=True, init=False)
class CandidateStateReplayPointer:
    """Deep-replay pointer derived from one trusted first-party terminal."""

    schema_version: int
    kind: str
    run_id: str
    method: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    attempt_id: str
    terminal_sha256: str
    attestation_message_sha256: str
    trusted_attester_policy_sha256: str
    authority_kind: str
    external_control_binding_sha256: str | None
    external_control_envelope_sha256: str | None
    external_control_reservation_sha256: str | None
    updates: tuple[CandidateStateByteIdentity, ...]

    def __init__(
        self,
        *,
        evidence: ValidatedNativeTerminalEvidence,
        updates: tuple[CandidateStateByteIdentity, ...],
        _verification_tag: object,
    ) -> None:
        if (
            _verification_tag is not _CANDIDATE_STATE_POINTER_SENTINEL
            or type(evidence) is not ValidatedNativeTerminalEvidence
            or not evidence.trusted_attestation
        ):
            raise TypeError(
                "candidate-state replay pointer requires trusted native evidence"
            )
        if evidence.binding.method not in _CANDIDATE_METHODS:
            raise ValueError("candidate-state replay requires TTS or L0 evidence")
        if not updates or any(
            type(value) is not CandidateStateByteIdentity for value in updates
        ):
            raise ValueError("candidate-state replay lacks exact update coverage")
        object.__setattr__(self, "schema_version", 1)
        object.__setattr__(self, "kind", "native_candidate_state_replay_pointer")
        object.__setattr__(self, "run_id", evidence.binding.run_id)
        object.__setattr__(self, "method", evidence.binding.method)
        object.__setattr__(self, "run_nonce_sha256", evidence.binding.run_nonce_sha256)
        object.__setattr__(
            self, "execution_plan_sha256", evidence.binding.execution_plan_sha256
        )
        object.__setattr__(
            self, "rank_config_sha256", evidence.binding.rank_config_sha256
        )
        object.__setattr__(self, "attempt_id", evidence.binding.attempt_id)
        object.__setattr__(self, "terminal_sha256", evidence.terminal_sha256)
        object.__setattr__(
            self,
            "attestation_message_sha256",
            evidence.attestation.message_sha256,
        )
        object.__setattr__(
            self,
            "trusted_attester_policy_sha256",
            evidence.trusted_attester_policy_sha256,
        )
        object.__setattr__(self, "authority_kind", evidence.authority_kind)
        object.__setattr__(
            self,
            "external_control_binding_sha256",
            evidence.external_control_binding_sha256,
        )
        object.__setattr__(
            self,
            "external_control_envelope_sha256",
            evidence.external_control_envelope_sha256,
        )
        object.__setattr__(
            self,
            "external_control_reservation_sha256",
            evidence.external_control_reservation_sha256,
        )
        object.__setattr__(self, "updates", updates)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "method": self.method,
            "run_nonce_sha256": self.run_nonce_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "attempt_id": self.attempt_id,
            "terminal_sha256": self.terminal_sha256,
            "attestation_message_sha256": self.attestation_message_sha256,
            "trusted_attester_policy_sha256": self.trusted_attester_policy_sha256,
            "authority_kind": self.authority_kind,
            "external_control_binding_sha256": (self.external_control_binding_sha256),
            "external_control_envelope_sha256": (self.external_control_envelope_sha256),
            "external_control_reservation_sha256": (
                self.external_control_reservation_sha256
            ),
            "updates": [value.to_dict() for value in self.updates],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def semantic_commitment_dict(self) -> dict[str, object]:
        """Return the reservation-independent scientific replay identity."""

        value = self.to_dict()
        value.pop("external_control_reservation_sha256")
        value["kind"] = "native_candidate_state_replay_commitment"
        return value

    @property
    def semantic_commitment_sha256(self) -> str:
        return canonical_sha256(self.semantic_commitment_dict())


@dataclass(frozen=True)
class CandidateStateReplayProjection:
    """Non-authorizing pre-reservation projection for atomic coverage checks."""

    schema_version: int
    kind: str
    terminal_binding_sha256: str
    control_envelope_sha256: str
    pointer_commitment: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "native_candidate_state_replay_projection_untrusted"
        ):
            raise ValueError("candidate-state projection schema is unsupported")
        for label, value in (
            ("projection terminal binding", self.terminal_binding_sha256),
            ("projection control envelope", self.control_envelope_sha256),
        ):
            _sha256(value, label)
        if (
            type(self.pointer_commitment) is not dict
            or "external_control_reservation_sha256" in self.pointer_commitment
            or self.pointer_commitment.get("kind")
            != "native_candidate_state_replay_commitment"
        ):
            raise TypeError(
                "candidate-state projection requires a reservation-free commitment"
            )

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "schema_version": self.schema_version,
                "kind": self.kind,
                "terminal_binding_sha256": self.terminal_binding_sha256,
                "control_envelope_sha256": self.control_envelope_sha256,
                "pointer_commitment": self.pointer_commitment,
            }
        )

    @property
    def pointer_commitment_sha256(self) -> str:
        return canonical_sha256(self.pointer_commitment)


@dataclass(frozen=True)
class CandidateStateReplayProofArtifact:
    """Durable external-control proof for one candidate-state replay pointer.

    The raw serving terminal is produced unsigned on the GPU host.  A local
    release signer authorizes its exact content binding and atomically
    reserves the terminal nonce together with both control challenges.  This
    artifact stores only public verification material; reopening it never
    reserves or consumes a challenge a second time.
    """

    schema_version: int
    kind: str
    raw_terminal: CanonicalJsonProofBinding
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding
    expected_inventory_sha256: str
    expected_registry_sha256: str
    expected_root_manifest_sha256: str
    pointer: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "native_candidate_state_replay_proof_artifact"
        ):
            raise ValueError("candidate-state proof artifact schema is unsupported")
        if type(self.raw_terminal) is not CanonicalJsonProofBinding:
            raise TypeError("candidate-state proof requires one raw terminal binding")
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("candidate-state proof requires one exact control envelope")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("candidate-state proof requires one replay reservation")
        _sha256(
            self.expected_inventory_sha256,
            "candidate-state proof inventory",
        )
        _sha256(
            self.expected_registry_sha256,
            "candidate-state proof registry",
        )
        _sha256(
            self.expected_root_manifest_sha256,
            "candidate-state proof release root",
        )
        # ``pointer_sha256`` is deliberately not embedded because that would
        # make its identity recursive.  The exact derived pointer is compared
        # during deep revalidation below.
        if type(self.pointer) is not dict or set(self.pointer) != {
            "schema_version",
            "kind",
            "run_id",
            "method",
            "run_nonce_sha256",
            "execution_plan_sha256",
            "rank_config_sha256",
            "attempt_id",
            "terminal_sha256",
            "attestation_message_sha256",
            "trusted_attester_policy_sha256",
            "authority_kind",
            "external_control_binding_sha256",
            "external_control_envelope_sha256",
            "external_control_reservation_sha256",
            "updates",
        }:
            raise ValueError("candidate-state proof pointer fields differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "raw_terminal": self.raw_terminal.to_dict(),
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
            "expected_inventory_sha256": self.expected_inventory_sha256,
            "expected_registry_sha256": self.expected_registry_sha256,
            "expected_root_manifest_sha256": self.expected_root_manifest_sha256,
            "pointer": self.pointer,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> CandidateStateReplayProofArtifact:
        raw = _exact_object(
            value,
            {
                "schema_version",
                "kind",
                "raw_terminal",
                "control_attestation",
                "replay_reservation",
                "expected_inventory_sha256",
                "expected_registry_sha256",
                "expected_root_manifest_sha256",
                "pointer",
            },
            "candidate-state proof artifact",
        )
        pointer = raw.pop("pointer")
        if type(pointer) is not dict:
            raise TypeError("candidate-state proof pointer must be an object")
        raw_terminal = CanonicalJsonProofBinding.from_dict(raw.pop("raw_terminal"))
        control_attestation = ControlArtifactAttestation.from_dict(
            raw.pop("control_attestation")
        )
        replay_reservation = ChallengeReplayReservationBinding.from_dict(
            raw.pop("replay_reservation")
        )
        return cls(
            **raw,
            raw_terminal=raw_terminal,
            control_attestation=control_attestation,
            replay_reservation=replay_reservation,
            pointer=pointer,
        )

    def revalidate(self, *, now_ns: int) -> CandidateStateReplayPointer:
        """Deep-reopen every authority without mutating the replay ledger."""

        controlled = _revalidate_controlled_native_terminal_proof(
            raw_terminal=self.raw_terminal,
            control_attestation=self.control_attestation,
            replay_reservation=self.replay_reservation,
            expected_inventory_sha256=self.expected_inventory_sha256,
            expected_registry_sha256=self.expected_registry_sha256,
            expected_root_manifest_sha256=self.expected_root_manifest_sha256,
            now_ns=now_ns,
            field="candidate-state proof",
        )
        pointer = derive_candidate_state_replay_pointer(controlled)
        if pointer.to_dict() != self.pointer:
            raise ValueError("candidate-state proof derived pointer changed")
        return pointer


_NATIVE_TERMINAL_RESULT_SENTINEL = object()


@dataclass(frozen=True, init=False)
class NativeTerminalRequestResult:
    """One exact request outcome projected from controlled native evidence."""

    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...] | None
    terminal_status: str
    terminal_reason: str
    submitted_to_server: bool
    request_sha256: str

    def __init__(
        self,
        *,
        expectation: TerminalRequestExpectation,
        request_row: Mapping[str, object],
        _verification_tag: object,
    ) -> None:
        if (
            _verification_tag is not _NATIVE_TERMINAL_RESULT_SENTINEL
            or type(expectation) is not TerminalRequestExpectation
        ):
            raise TypeError("native request result requires validated evidence")
        expectation.validate()
        if request_row.get("request_id") != expectation.request_id:
            raise RuntimeError("native request result row identity differs")
        request_sha256 = _sha256(
            request_row.get("request_sha256"), "native request result digest"
        )
        object.__setattr__(self, "request_id", expectation.request_id)
        object.__setattr__(self, "input_token_ids", expectation.input_token_ids)
        object.__setattr__(self, "output_token_ids", expectation.output_token_ids)
        object.__setattr__(self, "terminal_status", expectation.terminal_status)
        object.__setattr__(self, "terminal_reason", expectation.terminal_reason)
        object.__setattr__(self, "submitted_to_server", expectation.submitted_to_server)
        object.__setattr__(self, "request_sha256", request_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "input_token_ids": list(self.input_token_ids),
            "output_token_ids": (
                None if self.output_token_ids is None else list(self.output_token_ids)
            ),
            "terminal_status": self.terminal_status,
            "terminal_reason": self.terminal_reason,
            "submitted_to_server": self.submitted_to_server,
            "request_sha256": self.request_sha256,
        }


@dataclass(frozen=True, init=False)
class NativeTerminalUpdateResult:
    """Verifier-sealed update row projected from the validated native terminal."""

    update_index: int
    reset_scope: str
    request_epoch: int
    request_reset_receipt_sha256: str | None
    status: str
    published_version: int | None
    reconstruction_ok: bool
    source_round: int
    source_version: int
    optimizer_step: int
    cohort_sha256: str
    parameter_layout_sha256: str
    request_ids: tuple[str, ...]
    source_state_sha256: str | None
    candidate_bytes_sha256: str | None
    optimizer_state_bytes_sha256: str | None
    proposal_evidence_sha256: str | None
    update_sha256: str

    def __init__(
        self,
        *,
        update_row: Mapping[str, object],
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _NATIVE_TERMINAL_RESULT_SENTINEL:
            raise TypeError("native update result requires validated evidence")
        requests = update_row.get("request_ids")
        if not isinstance(requests, list):  # validated terminal invariant
            raise TypeError("native update result request IDs are malformed")
        values = {
            "update_index": int(update_row["update_index"]),
            "reset_scope": str(update_row["reset_scope"]),
            "request_epoch": int(update_row["request_epoch"]),
            "request_reset_receipt_sha256": update_row["request_reset_receipt_sha256"],
            "status": str(update_row["status"]),
            "published_version": (
                None
                if update_row["published_version"] is None
                else int(update_row["published_version"])
            ),
            "reconstruction_ok": bool(update_row["reconstruction_ok"]),
            "source_round": int(update_row["source_round"]),
            "source_version": int(update_row["source_version"]),
            "optimizer_step": int(update_row["optimizer_step"]),
            "cohort_sha256": str(update_row["cohort_sha256"]),
            "parameter_layout_sha256": str(update_row["parameter_layout_sha256"]),
            "request_ids": tuple(str(value) for value in requests),
            "source_state_sha256": update_row["source_state_sha256"],
            "candidate_bytes_sha256": update_row["candidate_bytes_sha256"],
            "optimizer_state_bytes_sha256": (
                update_row["optimizer_state_bytes_sha256"]
            ),
            "proposal_evidence_sha256": update_row["proposal_evidence_sha256"],
            "update_sha256": str(update_row["update_sha256"]),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, object]:
        return {
            "update_index": self.update_index,
            "reset_scope": self.reset_scope,
            "request_epoch": self.request_epoch,
            "request_reset_receipt_sha256": self.request_reset_receipt_sha256,
            "status": self.status,
            "published_version": self.published_version,
            "reconstruction_ok": self.reconstruction_ok,
            "source_round": self.source_round,
            "source_version": self.source_version,
            "optimizer_step": self.optimizer_step,
            "cohort_sha256": self.cohort_sha256,
            "parameter_layout_sha256": self.parameter_layout_sha256,
            "request_ids": list(self.request_ids),
            "source_state_sha256": self.source_state_sha256,
            "candidate_bytes_sha256": self.candidate_bytes_sha256,
            "optimizer_state_bytes_sha256": self.optimizer_state_bytes_sha256,
            "proposal_evidence_sha256": self.proposal_evidence_sha256,
            "update_sha256": self.update_sha256,
        }


@dataclass(frozen=True, init=False)
class NativeTerminalResultProjection:
    """Typed performance/safety projection from one controlled raw terminal.

    This projection intentionally contains no goodput or ITL statistic.  The
    native terminal has exact token/count/state evidence but no client arrival
    or completion timestamps; those statistics require their independent
    path-bound client-timestamp authority.
    """

    schema_version: int
    kind: str
    run_id: str
    method: str
    reset_scope: str | None
    request_admission_policy: str | None
    request_source_point_reset_protocol_sha256: str | None
    runtime_trust_mode: str | None
    formal_measurement: bool | None
    request_evidence_archive_sha256: str
    request_source_point_reset_receipt_count: int
    request_source_point_resets_sha256: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    attempt_id: str
    terminal_sha256: str
    authority_kind: str
    external_control_binding_sha256: str
    external_control_envelope_sha256: str
    external_control_reservation_sha256: str
    requests: tuple[NativeTerminalRequestResult, ...]
    updates: tuple[NativeTerminalUpdateResult, ...]
    scored_request_ids: tuple[str, ...]
    output_token_count: int
    request_rows_sha256: str
    round_rows_sha256: str
    update_rows_sha256: str
    performance_counters_sha256: str
    final_state_sha256: str
    performance_counters_json: str

    def __init__(
        self,
        *,
        evidence: ValidatedNativeTerminalEvidence,
        _verification_tag: object,
    ) -> None:
        if (
            _verification_tag is not _NATIVE_TERMINAL_RESULT_SENTINEL
            or type(evidence) is not ValidatedNativeTerminalEvidence
            or evidence.authority_kind != "external_release_control"
        ):
            raise TypeError(
                "native terminal result requires externally controlled evidence"
            )
        if evidence.terminal_schema_version != 2:
            raise RuntimeError("legacy terminal cannot authorize a current result")
        control_values = (
            evidence.external_control_binding_sha256,
            evidence.external_control_envelope_sha256,
            evidence.external_control_reservation_sha256,
        )
        if any(value is None for value in control_values):
            raise TypeError("native terminal result lacks external control identity")
        envelope = evidence.to_dict()
        request_round_rows = _exact_object(
            envelope["request_round_rows"],
            {"requests", "rounds"},
            "native terminal result request/round rows",
        )
        requests = request_round_rows["requests"]
        rounds = request_round_rows["rounds"]
        updates = envelope["update_rows"]
        performance = envelope["performance_counters"]
        final_state = envelope["final_state"]
        request_resets = evidence.request_source_point_resets
        if request_resets is None:  # current schema invariant
            raise RuntimeError("native terminal result lacks request reset identity")
        if (
            not isinstance(requests, list)
            or not isinstance(rounds, list)
            or not isinstance(updates, list)
            or not isinstance(performance, dict)
            or not isinstance(final_state, dict)
        ):
            raise TypeError("native terminal result containers are malformed")
        output_token_count = sum(
            _integer(row.get("output_tokens"), "result.output_tokens")
            for row in requests
            if isinstance(row, dict)
        )
        if len(requests) != len(evidence.requests):
            raise RuntimeError("native terminal result request coverage changed")
        request_results = tuple(
            NativeTerminalRequestResult(
                expectation=expectation,
                request_row=request_row,
                _verification_tag=_NATIVE_TERMINAL_RESULT_SENTINEL,
            )
            for expectation, request_row in zip(
                evidence.requests,
                requests,
                strict=True,
            )
        )
        update_results = tuple(
            NativeTerminalUpdateResult(
                update_row=row,
                _verification_tag=_NATIVE_TERMINAL_RESULT_SENTINEL,
            )
            for row in updates
        )
        object.__setattr__(self, "schema_version", 1)
        object.__setattr__(self, "kind", "native_terminal_result_projection")
        object.__setattr__(self, "run_id", evidence.binding.run_id)
        object.__setattr__(self, "method", evidence.binding.method)
        object.__setattr__(self, "reset_scope", evidence.binding.reset_scope)
        object.__setattr__(
            self,
            "request_admission_policy",
            evidence.binding.request_admission_policy,
        )
        object.__setattr__(
            self,
            "request_source_point_reset_protocol_sha256",
            request_resets.protocol_sha256,
        )
        object.__setattr__(
            self, "runtime_trust_mode", evidence.binding.runtime_trust_mode
        )
        object.__setattr__(
            self, "formal_measurement", evidence.binding.formal_measurement
        )
        object.__setattr__(
            self,
            "request_evidence_archive_sha256",
            request_resets.final_archive_sha256,
        )
        object.__setattr__(
            self,
            "request_source_point_reset_receipt_count",
            len(request_resets.receipts),
        )
        object.__setattr__(
            self,
            "request_source_point_resets_sha256",
            canonical_sha256(request_resets.to_dict()),
        )
        object.__setattr__(self, "run_nonce_sha256", evidence.binding.run_nonce_sha256)
        object.__setattr__(
            self, "execution_plan_sha256", evidence.binding.execution_plan_sha256
        )
        object.__setattr__(
            self, "rank_config_sha256", evidence.binding.rank_config_sha256
        )
        object.__setattr__(self, "attempt_id", evidence.binding.attempt_id)
        object.__setattr__(self, "terminal_sha256", evidence.terminal_sha256)
        object.__setattr__(self, "authority_kind", evidence.authority_kind)
        object.__setattr__(self, "external_control_binding_sha256", control_values[0])
        object.__setattr__(self, "external_control_envelope_sha256", control_values[1])
        object.__setattr__(
            self, "external_control_reservation_sha256", control_values[2]
        )
        object.__setattr__(self, "requests", request_results)
        object.__setattr__(self, "updates", update_results)
        object.__setattr__(
            self, "scored_request_ids", evidence.binding.scored_request_ids
        )
        object.__setattr__(self, "output_token_count", output_token_count)
        object.__setattr__(self, "request_rows_sha256", canonical_sha256(requests))
        object.__setattr__(self, "round_rows_sha256", canonical_sha256(rounds))
        object.__setattr__(self, "update_rows_sha256", canonical_sha256(updates))
        object.__setattr__(
            self, "performance_counters_sha256", canonical_sha256(performance)
        )
        object.__setattr__(self, "final_state_sha256", canonical_sha256(final_state))
        object.__setattr__(
            self, "performance_counters_json", _canonical_json_text(performance)
        )

    @property
    def performance_counters(self) -> dict[str, object]:
        value = json.loads(self.performance_counters_json)
        if not isinstance(value, dict):  # pragma: no cover - constructor invariant
            raise TypeError("native result performance stopped being an object")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "method": self.method,
            "reset_scope": self.reset_scope,
            "request_admission_policy": self.request_admission_policy,
            "request_source_point_reset_protocol_sha256": (
                self.request_source_point_reset_protocol_sha256
            ),
            "runtime_trust_mode": self.runtime_trust_mode,
            "formal_measurement": self.formal_measurement,
            "request_evidence_archive_sha256": self.request_evidence_archive_sha256,
            "request_source_point_reset_receipt_count": (
                self.request_source_point_reset_receipt_count
            ),
            "request_source_point_resets_sha256": (
                self.request_source_point_resets_sha256
            ),
            "run_nonce_sha256": self.run_nonce_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "attempt_id": self.attempt_id,
            "terminal_sha256": self.terminal_sha256,
            "authority_kind": self.authority_kind,
            "external_control_binding_sha256": (self.external_control_binding_sha256),
            "external_control_envelope_sha256": (self.external_control_envelope_sha256),
            "external_control_reservation_sha256": (
                self.external_control_reservation_sha256
            ),
            "requests": [request.to_dict() for request in self.requests],
            "updates": [update.to_dict() for update in self.updates],
            "scored_request_ids": list(self.scored_request_ids),
            "output_token_count": self.output_token_count,
            "request_rows_sha256": self.request_rows_sha256,
            "round_rows_sha256": self.round_rows_sha256,
            "update_rows_sha256": self.update_rows_sha256,
            "performance_counters_sha256": self.performance_counters_sha256,
            "final_state_sha256": self.final_state_sha256,
            "performance_counters": self.performance_counters,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class NativeTerminalResultProofArtifact:
    """Durable external-control proof for native performance/safety rows."""

    schema_version: int
    kind: str
    raw_terminal: CanonicalJsonProofBinding
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding
    expected_inventory_sha256: str
    expected_registry_sha256: str
    expected_root_manifest_sha256: str
    result: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "native_terminal_result_proof_artifact"
        ):
            raise ValueError("native result proof artifact schema is unsupported")
        if type(self.raw_terminal) is not CanonicalJsonProofBinding:
            raise TypeError("native result proof requires one raw terminal binding")
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("native result proof requires one exact control envelope")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("native result proof requires one replay reservation")
        for label, value in (
            ("native result proof inventory", self.expected_inventory_sha256),
            ("native result proof registry", self.expected_registry_sha256),
            ("native result proof release root", self.expected_root_manifest_sha256),
        ):
            _sha256(value, label)
        if (
            type(self.result) is not dict
            or self.result.get("kind") != "native_terminal_result_projection"
        ):
            raise TypeError("native result proof projection is malformed")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "raw_terminal": self.raw_terminal.to_dict(),
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
            "expected_inventory_sha256": self.expected_inventory_sha256,
            "expected_registry_sha256": self.expected_registry_sha256,
            "expected_root_manifest_sha256": self.expected_root_manifest_sha256,
            "result": self.result,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> NativeTerminalResultProofArtifact:
        raw = _exact_object(
            value,
            {
                "schema_version",
                "kind",
                "raw_terminal",
                "control_attestation",
                "replay_reservation",
                "expected_inventory_sha256",
                "expected_registry_sha256",
                "expected_root_manifest_sha256",
                "result",
            },
            "native result proof artifact",
        )
        result = raw.pop("result")
        if type(result) is not dict:
            raise TypeError("native result proof projection must be an object")
        raw_terminal = CanonicalJsonProofBinding.from_dict(raw.pop("raw_terminal"))
        control_attestation = ControlArtifactAttestation.from_dict(
            raw.pop("control_attestation")
        )
        replay_reservation = ChallengeReplayReservationBinding.from_dict(
            raw.pop("replay_reservation")
        )
        return cls(
            **raw,
            raw_terminal=raw_terminal,
            control_attestation=control_attestation,
            replay_reservation=replay_reservation,
            result=result,
        )

    def revalidate(self, *, now_ns: int) -> NativeTerminalResultProjection:
        evidence = _revalidate_controlled_native_terminal_proof(
            raw_terminal=self.raw_terminal,
            control_attestation=self.control_attestation,
            replay_reservation=self.replay_reservation,
            expected_inventory_sha256=self.expected_inventory_sha256,
            expected_registry_sha256=self.expected_registry_sha256,
            expected_root_manifest_sha256=self.expected_root_manifest_sha256,
            now_ns=now_ns,
            field="native result proof",
        )
        result = derive_native_terminal_result_projection(evidence)
        if result.to_dict() != self.result:
            raise ValueError("native result proof derived projection changed")
        return result


def _revalidate_controlled_native_terminal_proof(
    *,
    raw_terminal: CanonicalJsonProofBinding,
    control_attestation: ControlArtifactAttestation,
    replay_reservation: ChallengeReplayReservationBinding,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    field: str,
) -> ValidatedNativeTerminalEvidence:
    """Deep-reopen a durable external-control proof without replay mutation."""

    if type(now_ns) is not int or now_ns < replay_reservation.reserved_ns:
        raise ValueError(f"{field} time precedes reservation")
    if (
        control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError(f"{field} release root differs")
    prepared = prepare_native_terminal_external_control(
        raw_terminal.reopen(),
        control_attestation=control_attestation,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )
    reserved = replay_reservation.revalidate()
    verified = verify_release_control_artifact_attestation(
        control_attestation,
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=replay_reservation.reserved_ns,
        consumed_challenge_sha256s=(),
    )
    required_challenges = {
        verified.challenge_sha256,
        verified.deployment_policy_challenge_sha256,
        prepared.evidence.binding.run_nonce_sha256,
    }
    if not required_challenges.issubset(set(reserved)):
        raise ValueError(f"{field} reservation is incomplete")
    if (
        verified.artifact_type != "non_serving_terminal"
        or verified.artifact_sha256 != prepared.binding.sha256
        or verified.envelope_sha256 != control_attestation.sha256
    ):
        raise ValueError(f"{field} control identity differs")
    raw_evidence = prepared.evidence
    return ValidatedNativeTerminalEvidence(
        binding=raw_evidence.binding,
        begin_receipt=raw_evidence.begin_receipt,
        reset_receipt=raw_evidence.reset_receipt,
        requests=raw_evidence.requests,
        attestation=raw_evidence.attestation,
        terminal_sha256=raw_evidence.terminal_sha256,
        raw_json=raw_evidence.raw_json,
        external_control_binding_sha256=prepared.binding.sha256,
        external_control_envelope_sha256=verified.envelope_sha256,
        external_control_reservation_sha256=(replay_reservation.reservation_sha256),
        external_control_trusted_policy_sha256=(
            verified.trusted_attester_policy_sha256
        ),
        _verification_tag=_VALIDATED_NATIVE_TERMINAL_SENTINEL,
    )


def derive_native_terminal_result_projection(
    evidence: ValidatedNativeTerminalEvidence,
) -> NativeTerminalResultProjection:
    """Project exact native performance/safety rows from controlled evidence."""

    if type(evidence) is not ValidatedNativeTerminalEvidence:
        raise TypeError("native terminal result requires exact native evidence")
    return NativeTerminalResultProjection(
        evidence=evidence,
        _verification_tag=_NATIVE_TERMINAL_RESULT_SENTINEL,
    )


def derive_native_terminal_result_projection_from_verified_formal_control(
    value: object,
    *,
    expected_binding: NativeTerminalRunBinding,
    expected_inventory_sha256: str,
    formal_control_binding_sha256: str,
    verified_control: VerifiedControlArtifact,
    replay_reservation: ChallengeReplayReservationBinding,
) -> NativeTerminalResultProjection:
    """Project a native TP1 terminal under a verifier-owned formal wrapper.

    The formal wrapper may bind more evidence than the native terminal alone
    (launch admission, spend ledger, process lifecycle).  This helper accepts
    only an already verified control and replay reservation, reopens the raw
    native terminal under the empty remote trust policy, and carries the exact
    formal control identity into the standard native result projection.
    """

    if type(verified_control) is not VerifiedControlArtifact:
        raise TypeError("formal native projection requires verified control")
    if type(replay_reservation) is not ChallengeReplayReservationBinding:
        raise TypeError("formal native projection requires replay reservation")
    _sha256(formal_control_binding_sha256, "formal native control binding")
    _sha256(expected_inventory_sha256, "formal native expected inventory")
    if (
        verified_control.artifact_type != "non_serving_terminal"
        or verified_control.artifact_sha256 != formal_control_binding_sha256
    ):
        raise ValueError("formal native control does not authorize this binding")
    raw = validate_native_terminal_artifact(
        value,
        trusted_attester_policy=NO_TRUSTED_ATTESTERS,
        expected_binding=expected_binding,
    )
    if raw.binding.run_nonce_sha256 != expected_binding.run_nonce_sha256:
        raise ValueError("formal native terminal run nonce differs")
    reserved = replay_reservation.revalidate()
    required = {
        expected_binding.run_nonce_sha256,
        verified_control.challenge_sha256,
        verified_control.deployment_policy_challenge_sha256,
    }
    if not required.issubset(set(reserved)):
        raise ValueError("formal native terminal replay reservation is incomplete")
    controlled = ValidatedNativeTerminalEvidence(
        binding=raw.binding,
        begin_receipt=raw.begin_receipt,
        reset_receipt=raw.reset_receipt,
        requests=raw.requests,
        attestation=raw.attestation,
        terminal_sha256=raw.terminal_sha256,
        raw_json=raw.raw_json,
        external_control_binding_sha256=formal_control_binding_sha256,
        external_control_envelope_sha256=verified_control.envelope_sha256,
        external_control_reservation_sha256=replay_reservation.reservation_sha256,
        external_control_trusted_policy_sha256=(
            verified_control.trusted_attester_policy_sha256
        ),
        _verification_tag=_VALIDATED_NATIVE_TERMINAL_SENTINEL,
    )
    return derive_native_terminal_result_projection(controlled)


def derive_candidate_state_replay_pointer(
    evidence: ValidatedNativeTerminalEvidence,
) -> CandidateStateReplayPointer:
    """Derive formal replay provenance from a validated signed terminal only."""

    if type(evidence) is not ValidatedNativeTerminalEvidence:
        raise TypeError("candidate-state replay requires exact native evidence")
    if not evidence.trusted_attestation:
        raise RuntimeError("candidate-state replay requires trusted attestation")
    if evidence.terminal_schema_version != 2:
        raise RuntimeError("legacy terminal cannot authorize candidate-state replay")
    envelope = evidence.to_dict()
    raw_updates = envelope.get("update_rows")
    if not isinstance(raw_updates, list) or not raw_updates:
        raise RuntimeError("candidate-state replay lacks update rows")
    updates: list[CandidateStateByteIdentity] = []
    for raw in raw_updates:
        if not isinstance(raw, dict):  # validated evidence invariant
            raise TypeError("candidate-state replay update is malformed")
        replay_fields = (
            raw["source_state_sha256"],
            raw["candidate_bytes_sha256"],
            raw["optimizer_state_bytes_sha256"],
            raw["proposal_evidence_sha256"],
        )
        if any(value is None for value in replay_fields):
            raise RuntimeError(
                "candidate-state replay requires profile byte identities"
            )
        updates.append(
            CandidateStateByteIdentity(
                update_index=int(raw["update_index"]),
                reset_scope=str(raw["reset_scope"]),
                request_epoch=int(raw["request_epoch"]),
                request_reset_receipt_sha256=raw["request_reset_receipt_sha256"],
                source_round=int(raw["source_round"]),
                source_version=int(raw["source_version"]),
                request_ids=tuple(str(value) for value in raw["request_ids"]),
                source_state_sha256=str(replay_fields[0]),
                candidate_bytes_sha256=str(replay_fields[1]),
                optimizer_state_bytes_sha256=str(replay_fields[2]),
                proposal_evidence_sha256=str(replay_fields[3]),
                update_sha256=str(raw["update_sha256"]),
                _verification_tag=_CANDIDATE_STATE_POINTER_SENTINEL,
            )
        )
    return CandidateStateReplayPointer(
        evidence=evidence,
        updates=tuple(updates),
        _verification_tag=_CANDIDATE_STATE_POINTER_SENTINEL,
    )


def _controlled_evidence_for_reservation(
    prepared: PreparedNativeTerminalExternalControl,
    verified_control: VerifiedControlArtifact,
    reservation_sha256: str,
) -> ValidatedNativeTerminalEvidence:
    if type(prepared) is not PreparedNativeTerminalExternalControl:
        raise TypeError("controlled terminal finalization requires one prepared row")
    if type(verified_control) is not VerifiedControlArtifact:
        raise TypeError("controlled terminal finalization requires verified control")
    _sha256(reservation_sha256, "controlled terminal reservation")
    if (
        verified_control.artifact_type != "non_serving_terminal"
        or verified_control.artifact_sha256 != prepared.binding.sha256
        or verified_control.envelope_sha256 != prepared.control_attestation.sha256
        or verified_control.trusted_attester_policy_sha256
        != prepared.control_attestation.trusted_attester_policy_sha256
    ):
        raise ValueError("controlled terminal verified identity differs")
    evidence = prepared.evidence
    return ValidatedNativeTerminalEvidence(
        binding=evidence.binding,
        begin_receipt=evidence.begin_receipt,
        reset_receipt=evidence.reset_receipt,
        requests=evidence.requests,
        attestation=evidence.attestation,
        terminal_sha256=evidence.terminal_sha256,
        raw_json=evidence.raw_json,
        external_control_binding_sha256=prepared.binding.sha256,
        external_control_envelope_sha256=verified_control.envelope_sha256,
        external_control_reservation_sha256=reservation_sha256,
        external_control_trusted_policy_sha256=(
            verified_control.trusted_attester_policy_sha256
        ),
        _verification_tag=_VALIDATED_NATIVE_TERMINAL_SENTINEL,
    )


def project_prepared_candidate_state_replay_pointer(
    prepared: PreparedNativeTerminalExternalControl,
    *,
    verified_control: VerifiedControlArtifact,
) -> CandidateStateReplayProjection:
    """Project rows before one wider atomic registry reservation.

    The return type is intentionally not a ``CandidateStateReplayPointer`` and
    cannot authorize coverage or materialization.  It exists only so a caller
    can validate the complete candidate-coverage structure before committing a
    single reservation shared with its registry controls.
    """

    evidence = _controlled_evidence_for_reservation(
        prepared,
        verified_control,
        "0" * 64,
    )
    pointer = derive_candidate_state_replay_pointer(evidence)
    return CandidateStateReplayProjection(
        schema_version=2,
        kind="native_candidate_state_replay_projection_untrusted",
        terminal_binding_sha256=prepared.binding.sha256,
        control_envelope_sha256=verified_control.envelope_sha256,
        pointer_commitment=pointer.semantic_commitment_dict(),
    )


def finalize_prepared_native_terminal_external_controls(
    prepared: tuple[PreparedNativeTerminalExternalControl, ...],
    *,
    verified_controls: tuple[VerifiedControlArtifact, ...],
    replay_reservation: ChallengeReplayReservationBinding,
) -> tuple[ValidatedNativeTerminalEvidence, ...]:
    """Finalize a caller-owned atomic reservation without reserving again."""

    if (
        type(prepared) is not tuple
        or not prepared
        or any(
            type(row) is not PreparedNativeTerminalExternalControl for row in prepared
        )
        or type(verified_controls) is not tuple
        or len(verified_controls) != len(prepared)
        or any(type(row) is not VerifiedControlArtifact for row in verified_controls)
    ):
        raise TypeError("controlled terminal finalization requires exact tuples")
    if type(replay_reservation) is not ChallengeReplayReservationBinding:
        raise TypeError("controlled terminal finalization requires replay binding")
    reserved = set(replay_reservation.revalidate())
    results: list[ValidatedNativeTerminalEvidence] = []
    for row, supplied_verified in zip(prepared, verified_controls, strict=True):
        reverified = verify_release_control_artifact_attestation(
            row.control_attestation,
            expected_inventory_sha256=row.expected_inventory_sha256,
            now_ns=replay_reservation.reserved_ns,
            consumed_challenge_sha256s=(),
        )
        if reverified != supplied_verified:
            raise ValueError("controlled terminal verified row changed")
        required = {
            reverified.challenge_sha256,
            reverified.deployment_policy_challenge_sha256,
            row.evidence.binding.run_nonce_sha256,
        }
        if not required.issubset(reserved):
            raise ValueError("controlled terminal replay reservation is incomplete")
        results.append(
            _controlled_evidence_for_reservation(
                row,
                reverified,
                replay_reservation.reservation_sha256,
            )
        )
    return tuple(results)


def finalize_prepared_candidate_state_replay_pointers(
    prepared: tuple[PreparedNativeTerminalExternalControl, ...],
    *,
    verified_controls: tuple[VerifiedControlArtifact, ...],
    replay_reservation: ChallengeReplayReservationBinding,
) -> tuple[CandidateStateReplayPointer, ...]:
    """Return sealed pointers after a unified reservation is durable."""

    evidences = finalize_prepared_native_terminal_external_controls(
        prepared,
        verified_controls=verified_controls,
        replay_reservation=replay_reservation,
    )
    return tuple(derive_candidate_state_replay_pointer(row) for row in evidences)


def validate_candidate_state_replay_pointer_artifact(
    value: object,
    *,
    trusted_attester_policy: TrustedAttesterPolicy,
) -> CandidateStateReplayPointer:
    """Deep-reopen a durable native terminal before deriving replay identity."""

    evidence = validate_native_terminal_artifact(
        value,
        trusted_attester_policy=trusted_attester_policy,
    )
    return derive_candidate_state_replay_pointer(evidence)


def _binding_artifact(binding: NativeTerminalRunBinding) -> dict[str, object]:
    binding.validate()
    return {
        **_identity_values(binding),
        "warmup_request_ids": list(binding.warmup_request_ids),
        "scored_request_ids": list(binding.scored_request_ids),
    }


def _binding_from_artifact(
    value: object,
    *,
    legacy: bool = False,
) -> NativeTerminalRunBinding:
    raw = _exact_object(
        value,
        _LEGACY_ARTIFACT_BINDING_KEYS if legacy else _ARTIFACT_BINDING_KEYS,
        "terminal artifact binding",
    )
    warmup = raw.pop("warmup_request_ids")
    scored = raw.pop("scored_request_ids")
    if not isinstance(warmup, list) or not isinstance(scored, list):
        raise TypeError("terminal artifact request IDs must be JSON lists")
    if legacy:
        raw.update(
            {
                "reset_scope": None,
                "request_admission_policy": None,
                "runtime_trust_mode": None,
                "formal_measurement": None,
            }
        )
    else:
        reset_scope = raw.get("reset_scope")
        request_admission_policy = raw.get("request_admission_policy")
        if (reset_scope, request_admission_policy) == (
            "none",
            "allocation_free",
        ):
            raw["reset_scope"] = None
            raw["request_admission_policy"] = None
    binding = NativeTerminalRunBinding(
        **raw,
        warmup_request_ids=tuple(warmup),
        scored_request_ids=tuple(scored),
    )
    if legacy:
        _validate_legacy_run_binding(binding)
    else:
        binding.validate()
    return binding


def _request_expectation_artifact(
    request: TerminalRequestExpectation,
) -> dict[str, object]:
    request.validate()
    return {
        "request_id": request.request_id,
        "input_token_ids": list(request.input_token_ids),
        "output_token_ids": (
            None if request.output_token_ids is None else list(request.output_token_ids)
        ),
        "terminal_status": request.terminal_status,
        "terminal_reason": request.terminal_reason,
        "submitted_to_server": request.submitted_to_server,
    }


def _request_expectation_from_artifact(value: object) -> TerminalRequestExpectation:
    raw = _exact_object(value, _ARTIFACT_REQUEST_KEYS, "terminal artifact request")
    inputs = _token_ids(raw.pop("input_token_ids"), "artifact input_token_ids")
    raw_outputs = raw.pop("output_token_ids")
    outputs = (
        None
        if raw_outputs is None
        else _token_ids(raw_outputs, "artifact output_token_ids")
    )
    request = TerminalRequestExpectation(
        **raw,
        input_token_ids=inputs,
        output_token_ids=outputs,
    )
    request.validate()
    return request


def validate_native_terminal_artifact(
    value: object,
    *,
    trusted_attester_policy: TrustedAttesterPolicy,
    expected_binding: NativeTerminalRunBinding | None = None,
    expected_warmup_requests: Sequence[TerminalRequestExpectation] | None = None,
    expected_scored_requests: Sequence[TerminalRequestExpectation] | None = None,
) -> ValidatedNativeTerminalEvidence:
    """Revalidate a durable begin/reset/final bundle and its release signature."""

    if type(trusted_attester_policy) is not TrustedAttesterPolicy:
        raise TypeError("terminal artifact requires an exact release policy")
    trusted_attester_policy.validate()
    direct_current = (
        type(value) is dict
        and value.get("schema_version") == 2
        and value.get("artifact_kind") == NATIVE_TERMINAL_ARTIFACT_KIND
    )
    reopened = (
        dict(value)
        if direct_current
        else reopen_scalable_native_terminal_artifact(value)
    )
    raw = _exact_object(reopened, _ARTIFACT_KEYS, "native terminal artifact")
    legacy = (
        raw["schema_version"] == 1
        and raw["artifact_kind"] == LEGACY_NATIVE_TERMINAL_ARTIFACT_KIND
    )
    current = (
        raw["schema_version"] == 2
        and raw["artifact_kind"] == NATIVE_TERMINAL_ARTIFACT_KIND
    )
    if (not legacy and not current) or raw["rank"] != 0:
        raise ValueError("native terminal artifact schema/rank is unsupported")
    policy_sha256 = _sha256(
        raw["trusted_attester_policy_sha256"],
        "trusted_attester_policy_sha256",
    )
    if policy_sha256 != trusted_attester_policy.sha256:
        raise ValueError("native terminal artifact uses another release policy")
    binding = _binding_from_artifact(raw["binding"], legacy=legacy)
    if raw["run_id"] != binding.run_id:
        raise ValueError("native terminal artifact changed its run identity")
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("native terminal artifact changed its expected binding")
    warmup_values = raw["warmup_requests"]
    scored_values = raw["scored_requests"]
    if not isinstance(warmup_values, list) or not isinstance(scored_values, list):
        raise TypeError("native terminal artifact request coverage is malformed")
    warmup = _validate_request_expectations(
        tuple(_request_expectation_from_artifact(row) for row in warmup_values),
        expected_ids=binding.warmup_request_ids,
        warmup=True,
    )
    scored = _validate_request_expectations(
        tuple(_request_expectation_from_artifact(row) for row in scored_values),
        expected_ids=binding.scored_request_ids,
        warmup=False,
    )
    if expected_warmup_requests is not None and warmup != tuple(
        expected_warmup_requests
    ):
        raise ValueError("native terminal artifact changed warmup expectations")
    if expected_scored_requests is not None and scored != tuple(
        expected_scored_requests
    ):
        raise ValueError("native terminal artifact changed scored expectations")
    begin_raw = _exact_object(
        raw["begin"],
        _LEGACY_BEGIN_RECEIPT_KEYS if legacy else _BEGIN_RECEIPT_KEYS,
        "artifact begin receipt",
    )
    begin_generation = _integer(
        begin_raw["reset_generation"], "artifact begin reset_generation", minimum=1
    )
    begin = _validate_begin_receipt(
        begin_raw,
        binding=binding,
        prior_reset_generation=begin_generation - 1,
        prior_process=None,
        legacy=legacy,
    )
    reset = _validate_reset_receipt(
        raw["reset"],
        begin=begin,
        warmup_requests=warmup,
        legacy=legacy,
    )
    terminal = _validate_terminal(
        raw["terminal"],
        begin=begin,
        reset=reset,
        requests=scored,
        trusted_attester_policy=trusted_attester_policy,
        legacy=legacy,
    )
    if (
        _sha256(raw["begin_sha256"], "artifact begin_sha256") != begin.begin_sha256
        or _sha256(raw["reset_sha256"], "artifact reset_sha256") != reset.reset_sha256
        or _sha256(raw["terminal_sha256"], "artifact terminal_sha256")
        != terminal.terminal_sha256
    ):
        raise ValueError("native terminal artifact digest binding is inconsistent")
    return terminal


def build_native_terminal_external_control_binding(
    value: object,
    *,
    trusted_attester_policy: TrustedAttesterPolicy,
    inventory_sha256: str,
    expected_binding: NativeTerminalRunBinding | None = None,
    expected_warmup_requests: Sequence[TerminalRequestExpectation] | None = None,
    expected_scored_requests: Sequence[TerminalRequestExpectation] | None = None,
) -> NativeTerminalExternalControlBinding:
    """Deep-validate pulled raw evidence and derive the offline signing subject."""

    _sha256(inventory_sha256, "native terminal external-control inventory")
    evidence = validate_native_terminal_artifact(
        value,
        trusted_attester_policy=trusted_attester_policy,
        expected_binding=expected_binding,
        expected_warmup_requests=expected_warmup_requests,
        expected_scored_requests=expected_scored_requests,
    )
    if evidence.terminal_schema_version != 2:
        raise RuntimeError("legacy terminal is read-only and cannot be controlled")
    canonical_body = canonical_json_bytes(value) + b"\n"
    binding = evidence.binding
    return NativeTerminalExternalControlBinding(
        schema_version=2,
        kind="native_terminal_external_control_binding",
        canonical_raw_sha256=hashlib.sha256(canonical_body).hexdigest(),
        semantic_artifact_sha256=canonical_sha256(value),
        terminal_sha256=evidence.terminal_sha256,
        run_id=binding.run_id,
        run_nonce_sha256=binding.run_nonce_sha256,
        execution_plan_sha256=binding.execution_plan_sha256,
        rank_config_sha256=binding.rank_config_sha256,
        attempt_id=binding.attempt_id,
        session_id=binding.session_id,
        session_epoch=binding.session_epoch,
        method=binding.method,
        reset_scope=binding.reset_scope,
        request_admission_policy=binding.request_admission_policy,
        runtime_trust_mode=binding.runtime_trust_mode,
        formal_measurement=binding.formal_measurement,
        inventory_sha256=inventory_sha256,
    )


def prepare_native_terminal_external_control(
    value: object,
    *,
    control_attestation: ControlArtifactAttestation,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_binding: NativeTerminalRunBinding | None = None,
    expected_warmup_requests: Sequence[TerminalRequestExpectation] | None = None,
    expected_scored_requests: Sequence[TerminalRequestExpectation] | None = None,
) -> PreparedNativeTerminalExternalControl:
    """Deep-check raw evidence and its subject without consuming a nonce."""

    if type(control_attestation) is not ControlArtifactAttestation:
        raise TypeError("native terminal requires an exact external control")
    _sha256(expected_inventory_sha256, "native terminal expected inventory")
    _sha256(expected_registry_sha256, "native terminal expected registry")
    control_attestation.__post_init__()
    # Formal remote collection is deliberately unsigned: the offline/root and
    # signer keys never enter the serving host.  Reopen that raw artifact
    # under the source-owned empty policy, then authorize it independently via
    # the dynamic local control below.  Using the signer policy here would
    # make every first-party unsigned collection impossible to qualify.
    policy = NO_TRUSTED_ATTESTERS
    evidence = validate_native_terminal_artifact(
        value,
        trusted_attester_policy=policy,
        expected_binding=expected_binding,
        expected_warmup_requests=expected_warmup_requests,
        expected_scored_requests=expected_scored_requests,
    )
    control_binding = build_native_terminal_external_control_binding(
        value,
        trusted_attester_policy=policy,
        inventory_sha256=expected_inventory_sha256,
        expected_binding=expected_binding,
        expected_warmup_requests=expected_warmup_requests,
        expected_scored_requests=expected_scored_requests,
    )
    subject = control_attestation.subject
    if (
        subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != control_binding.sha256
        or subject.protocol_sha256 != NATIVE_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256
        or subject.registry_sha256 != expected_registry_sha256
        or subject.lineage_sha256 != control_binding.lineage_sha256
    ):
        raise ValueError("native terminal external control subject is not exact")
    return PreparedNativeTerminalExternalControl(
        evidence=evidence,
        binding=control_binding,
        control_attestation=control_attestation,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        _verification_tag=_PREPARED_NATIVE_TERMINAL_CONTROL_SENTINEL,
    )


def validate_native_terminal_artifacts_with_external_controls(
    prepared: tuple[PreparedNativeTerminalExternalControl, ...],
    *,
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> tuple[ValidatedNativeTerminalEvidence, ...]:
    """Verify a complete terminal batch and reserve all challenges atomically."""

    if (
        type(prepared) is not tuple
        or not prepared
        or any(
            type(row) is not PreparedNativeTerminalExternalControl for row in prepared
        )
    ):
        raise TypeError("native terminal control batch requires exact prepared rows")
    if type(replay_store) is not ChallengeReplayStore:
        raise TypeError("native terminal control batch requires the replay store")
    inventories = {row.expected_inventory_sha256 for row in prepared}
    if len(inventories) != 1:
        raise ValueError("native terminal control batch spans multiple inventories")
    binding_sha256s = tuple(row.binding.sha256 for row in prepared)
    envelope_sha256s = tuple(row.control_attestation.sha256 for row in prepared)
    run_attempts = tuple(
        (row.evidence.binding.run_id, row.evidence.binding.attempt_id)
        for row in prepared
    )
    if (
        len(set(binding_sha256s)) != len(binding_sha256s)
        or len(set(envelope_sha256s)) != len(envelope_sha256s)
        or len(set(run_attempts)) != len(run_attempts)
    ):
        raise ValueError("native terminal control batch contains duplicate evidence")
    controls = tuple(row.control_attestation for row in prepared)
    run_nonces = tuple(row.evidence.binding.run_nonce_sha256 for row in prepared)
    if len(set(run_nonces)) != len(run_nonces):
        raise ValueError("native terminal control batch reuses one run nonce")
    verified_rows = verify_and_reserve_release_control_artifact_attestations(
        controls,
        expected_inventory_sha256=next(iter(inventories)),
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=run_nonces,
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified_rows,
        reserved_ns=now_ns,
        additional_challenge_sha256s=run_nonces,
    )
    reservation = replay_store.bind_reservation(reservation_sha256)
    return finalize_prepared_native_terminal_external_controls(
        prepared,
        verified_controls=verified_rows,
        replay_reservation=reservation,
    )


def validate_native_terminal_artifact_with_external_control(
    value: object,
    *,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    expected_binding: NativeTerminalRunBinding | None = None,
    expected_warmup_requests: Sequence[TerminalRequestExpectation] | None = None,
    expected_scored_requests: Sequence[TerminalRequestExpectation] | None = None,
) -> ValidatedNativeTerminalEvidence:
    """Authorize one pulled terminal with local root-controlled evidence.

    The control key is intentionally absent from the remote process.  Only the
    canonical terminal returned by the native producer is accepted, and the
    deployment/control challenges are reserved atomically before the sealed
    evidence object becomes trusted.
    """

    _sha256(expected_root_manifest_sha256, "native terminal expected release root")
    if (
        control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("native terminal external control uses another release root")
    prepared = prepare_native_terminal_external_control(
        value,
        control_attestation=control_attestation,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_binding=expected_binding,
        expected_warmup_requests=expected_warmup_requests,
        expected_scored_requests=expected_scored_requests,
    )
    return validate_native_terminal_artifacts_with_external_controls(
        (prepared,),
        replay_store=replay_store,
        now_ns=now_ns,
    )[0]


def validate_controlled_candidate_state_replay_pointer_artifact(
    value: object,
    *,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    expected_binding: NativeTerminalRunBinding | None = None,
) -> CandidateStateReplayPointer:
    """Deep-reopen, externally authorize, then seal candidate-state rows."""

    evidence = validate_native_terminal_artifact_with_external_control(
        value,
        control_attestation=control_attestation,
        replay_store=replay_store,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
        expected_binding=expected_binding,
    )
    return derive_candidate_state_replay_pointer(evidence)


def _reserve_native_terminal_proof_batch(
    raw_terminal_artifact_paths: tuple[str, ...],
    *,
    control_attestations: tuple[ControlArtifactAttestation, ...],
    replay_store: ChallengeReplayStore,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_paths: tuple[str, ...],
    expected_bindings: tuple[NativeTerminalRunBinding | None, ...],
    field: str,
) -> tuple[
    tuple[CanonicalJsonProofBinding, ...],
    tuple[ValidatedNativeTerminalEvidence, ...],
    ChallengeReplayReservationBinding,
]:
    """Validate a complete proof batch, then reserve its authority once."""

    rows = (
        raw_terminal_artifact_paths,
        control_attestations,
        proof_artifact_paths,
        expected_bindings,
    )
    if (
        any(type(row) is not tuple for row in rows)
        or not raw_terminal_artifact_paths
        or len(raw_terminal_artifact_paths) > 512
        or any(len(row) != len(raw_terminal_artifact_paths) for row in rows[1:])
        or any(
            type(value) is not ControlArtifactAttestation
            for value in control_attestations
        )
        or any(
            value is not None and type(value) is not NativeTerminalRunBinding
            for value in expected_bindings
        )
    ):
        raise TypeError(f"{field} batch inputs are not exact")
    _sha256(expected_inventory_sha256, f"{field} expected inventory")
    _sha256(expected_registry_sha256, f"{field} expected registry")
    _sha256(expected_root_manifest_sha256, f"{field} expected release root")
    if len(set(raw_terminal_artifact_paths)) != len(raw_terminal_artifact_paths) or len(
        set(proof_artifact_paths)
    ) != len(proof_artifact_paths):
        raise ValueError(f"{field} batch paths must be unique")
    for path_value in proof_artifact_paths:
        path = Path(path_value)
        if (
            not path.is_absolute()
            or Path(os.path.abspath(path)) != path
            or path.exists()
        ):
            raise ValueError(f"{field} output must be a new absolute path")
    raw_bindings = tuple(
        CanonicalJsonProofBinding.bind(path) for path in raw_terminal_artifact_paths
    )
    prepared = tuple(
        prepare_native_terminal_external_control(
            raw_binding.reopen(),
            control_attestation=control,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_registry_sha256=expected_registry_sha256,
            expected_binding=expected_binding,
        )
        for raw_binding, control, expected_binding in zip(
            raw_bindings,
            control_attestations,
            expected_bindings,
            strict=True,
        )
    )
    if any(
        control.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
        for control in control_attestations
    ):
        raise ValueError(f"{field} batch uses another release root")
    evidences = validate_native_terminal_artifacts_with_external_controls(
        prepared,
        replay_store=replay_store,
        now_ns=now_ns,
    )
    reservation_sha256s = {
        evidence.external_control_reservation_sha256 for evidence in evidences
    }
    if len(reservation_sha256s) != 1 or None in reservation_sha256s:
        raise RuntimeError(f"{field} batch lacks one reservation")
    reservation_sha256 = next(iter(reservation_sha256s))
    assert reservation_sha256 is not None
    return (
        raw_bindings,
        evidences,
        replay_store.bind_reservation(reservation_sha256),
    )


def publish_candidate_state_replay_proof_artifact(
    raw_terminal_artifact_path: str,
    *,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str,
    expected_binding: NativeTerminalRunBinding | None = None,
) -> CanonicalJsonProofBinding:
    """Trust-lift one pulled raw terminal and publish a durable replay proof.

    This is the only producer for the external-control proof artifact.  The
    raw terminal must already be an immutable canonical file in a safe local
    evidence directory.  The replay reservation is committed once here; later
    registry or coverage validation uses :func:`validate_candidate_state_replay_proof_artifact`
    and never mutates the replay store.
    """

    return publish_candidate_state_replay_proof_artifacts(
        (raw_terminal_artifact_path,),
        control_attestations=(control_attestation,),
        replay_store=replay_store,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
        proof_artifact_paths=(proof_artifact_path,),
        expected_bindings=(expected_binding,),
    )[0]


def publish_candidate_state_replay_proof_artifacts(
    raw_terminal_artifact_paths: tuple[str, ...],
    *,
    control_attestations: tuple[ControlArtifactAttestation, ...],
    replay_store: ChallengeReplayStore,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_paths: tuple[str, ...],
    expected_bindings: tuple[NativeTerminalRunBinding | None, ...],
) -> tuple[CanonicalJsonProofBinding, ...]:
    """Publish one exact terminal batch under a shared atomic reservation.

    All raw terminals, control subjects, and output-path conflicts are checked
    before the replay ledger changes.  Signature verification and reservation
    are one locked transaction.  If a later filesystem publication fails, the
    reservation remains consumed and the function returns no bindings; formal
    coverage must require the complete returned batch and one shared
    reservation SHA, so an orphaned partial file remains non-materializable.
    """

    raw_bindings, evidences, reservation = _reserve_native_terminal_proof_batch(
        raw_terminal_artifact_paths,
        control_attestations=control_attestations,
        replay_store=replay_store,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
        proof_artifact_paths=proof_artifact_paths,
        expected_bindings=expected_bindings,
        field="candidate-state proof",
    )
    bindings: list[CanonicalJsonProofBinding] = []
    try:
        for raw_binding, control, evidence, output_path in zip(
            raw_bindings,
            control_attestations,
            evidences,
            proof_artifact_paths,
            strict=True,
        ):
            pointer = derive_candidate_state_replay_pointer(evidence)
            artifact = CandidateStateReplayProofArtifact(
                schema_version=1,
                kind="native_candidate_state_replay_proof_artifact",
                raw_terminal=raw_binding,
                control_attestation=control,
                replay_reservation=reservation,
                expected_inventory_sha256=expected_inventory_sha256,
                expected_registry_sha256=expected_registry_sha256,
                expected_root_manifest_sha256=expected_root_manifest_sha256,
                pointer=pointer.to_dict(),
            )
            publish_canonical_json_no_replace(output_path, artifact.to_dict())
            bindings.append(
                CanonicalJsonProofBinding.bind(
                    output_path,
                    semantic_sha256=artifact.sha256,
                )
            )
    except Exception as error:
        raise RuntimeError(
            "candidate-state proof batch publication failed after reservation; "
            "discard every partial output and issue new controls"
        ) from error
    return tuple(bindings)


def validate_candidate_state_replay_proof_artifact(
    proof_artifact_path: str,
    *,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> CandidateStateReplayPointer:
    """Deep-reopen a durable proof without consuming its challenges again."""

    _sha256(expected_inventory_sha256, "candidate-state expected inventory")
    _sha256(expected_registry_sha256, "candidate-state expected registry")
    _sha256(expected_root_manifest_sha256, "candidate-state expected release root")
    binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = CandidateStateReplayProofArtifact.from_dict(binding.reopen())
    if (
        binding.semantic_sha256 != artifact.sha256
        or artifact.expected_inventory_sha256 != expected_inventory_sha256
        or artifact.expected_registry_sha256 != expected_registry_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
    ):
        raise ValueError("candidate-state proof file identity differs")
    return artifact.revalidate(now_ns=now_ns)


def publish_native_terminal_result_proof_artifact(
    raw_terminal_artifact_path: str,
    *,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str,
    expected_binding: NativeTerminalRunBinding,
) -> CanonicalJsonProofBinding:
    """Publish one durable native performance/safety result proof."""

    return publish_native_terminal_result_proof_artifacts(
        (raw_terminal_artifact_path,),
        control_attestations=(control_attestation,),
        replay_store=replay_store,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
        proof_artifact_paths=(proof_artifact_path,),
        expected_bindings=(expected_binding,),
    )[0]


def publish_native_terminal_result_proof_artifacts(
    raw_terminal_artifact_paths: tuple[str, ...],
    *,
    control_attestations: tuple[ControlArtifactAttestation, ...],
    replay_store: ChallengeReplayStore,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_paths: tuple[str, ...],
    expected_bindings: tuple[NativeTerminalRunBinding, ...],
) -> tuple[CanonicalJsonProofBinding, ...]:
    """Publish a result-proof batch under one atomic replay reservation."""

    if type(expected_bindings) is not tuple or any(
        type(value) is not NativeTerminalRunBinding for value in expected_bindings
    ):
        raise TypeError("native result proof requires exact expected bindings")
    raw_bindings, evidences, reservation = _reserve_native_terminal_proof_batch(
        raw_terminal_artifact_paths,
        control_attestations=control_attestations,
        replay_store=replay_store,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
        proof_artifact_paths=proof_artifact_paths,
        expected_bindings=expected_bindings,
        field="native result proof",
    )
    bindings: list[CanonicalJsonProofBinding] = []
    try:
        for raw_binding, control, evidence, output_path in zip(
            raw_bindings,
            control_attestations,
            evidences,
            proof_artifact_paths,
            strict=True,
        ):
            result = derive_native_terminal_result_projection(evidence)
            artifact = NativeTerminalResultProofArtifact(
                schema_version=1,
                kind="native_terminal_result_proof_artifact",
                raw_terminal=raw_binding,
                control_attestation=control,
                replay_reservation=reservation,
                expected_inventory_sha256=expected_inventory_sha256,
                expected_registry_sha256=expected_registry_sha256,
                expected_root_manifest_sha256=expected_root_manifest_sha256,
                result=result.to_dict(),
            )
            publish_canonical_json_no_replace(output_path, artifact.to_dict())
            bindings.append(
                CanonicalJsonProofBinding.bind(
                    output_path,
                    semantic_sha256=artifact.sha256,
                )
            )
    except Exception as error:
        raise RuntimeError(
            "native result proof batch publication failed after reservation; "
            "discard every partial output and issue new controls"
        ) from error
    return tuple(bindings)


def validate_native_terminal_result_proof_artifact(
    proof_artifact_path: str,
    *,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_execution_plan_sha256: str,
    expected_rank_config_sha256: str,
    expected_run_id: str,
    expected_run_nonce_sha256: str,
    expected_attempt_id: str,
    expected_method: str,
    now_ns: int,
) -> NativeTerminalResultProjection:
    """Deep-reopen one result proof and bind it to formal execution identity."""

    for label, value in (
        ("native result expected inventory", expected_inventory_sha256),
        ("native result expected registry", expected_registry_sha256),
        ("native result expected release root", expected_root_manifest_sha256),
        ("native result expected execution plan", expected_execution_plan_sha256),
        ("native result expected rank config", expected_rank_config_sha256),
        ("native result expected run nonce", expected_run_nonce_sha256),
    ):
        _sha256(value, label)
    _safe_id(expected_run_id, "native result expected run")
    _safe_id(expected_attempt_id, "native result expected attempt")
    if expected_method not in SUPPORTED_METHODS:
        raise ValueError("native result expected method is unsupported")
    binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = NativeTerminalResultProofArtifact.from_dict(binding.reopen())
    if (
        binding.semantic_sha256 != artifact.sha256
        or artifact.expected_inventory_sha256 != expected_inventory_sha256
        or artifact.expected_registry_sha256 != expected_registry_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
    ):
        raise ValueError("native result proof file identity differs")
    result = artifact.revalidate(now_ns=now_ns)
    if (
        result.execution_plan_sha256 != expected_execution_plan_sha256
        or result.rank_config_sha256 != expected_rank_config_sha256
        or result.run_id != expected_run_id
        or result.run_nonce_sha256 != expected_run_nonce_sha256
        or result.attempt_id != expected_attempt_id
        or result.method != expected_method
    ):
        raise ValueError("native result proof formal execution identity differs")
    return result


def _validate_capability(
    value: object, *, expected_method: str
) -> NativeTerminalCapability:
    raw = _exact_object(value, _CAPABILITY_KEYS, "terminal capability")
    if raw["schema_version"] != 2 or raw["hook"] != NATIVE_TERMINAL_EVIDENCE_HOOK:
        raise ValueError("terminal capability schema/hook mismatch")
    required = raw["required_fields"]
    methods = raw["supported_methods"]
    if (
        not isinstance(required, list)
        or tuple(required) != NATIVE_TERMINAL_EVIDENCE_FIELDS
    ):
        raise ValueError("terminal capability required fields differ from the hook")
    if not isinstance(methods, list) or tuple(methods) != _ORDERED_SUPPORTED_METHODS:
        raise ValueError("terminal capability supported methods differ from the hook")
    for name in (
        "enabled",
        "method_evidence_supported",
        "topology_supported",
        "trusted_attester_configured",
    ):
        _boolean(raw[name], f"capability.{name}")
    active_method = raw["active_method"]
    if active_method not in SUPPORTED_METHODS or active_method != expected_method:
        raise ValueError("terminal capability active method mismatch")
    protocol_sha256 = _validate_wire_reset_protocol(
        method=str(active_method),
        reset_scope=raw["reset_scope"],
        request_admission_policy=raw["request_admission_policy"],
        protocol_sha256=raw["request_source_point_reset_protocol_sha256"],
        field="terminal capability",
    )
    runtime_trust_mode, formal_measurement = _validate_runtime_trust_identity(
        method=str(active_method),
        runtime_trust_mode=raw["runtime_trust_mode"],
        formal_measurement=raw["formal_measurement"],
        field="terminal capability",
    )
    if raw["enabled"] and not (
        raw["method_evidence_supported"] and raw["topology_supported"]
    ):
        raise ValueError("terminal capability enabled state is internally inconsistent")
    if not raw["enabled"]:
        raise RuntimeError("native terminal evidence is not enabled")
    return NativeTerminalCapability(
        active_method=active_method,
        enabled=True,
        method_evidence_supported=bool(raw["method_evidence_supported"]),
        topology_supported=bool(raw["topology_supported"]),
        trusted_attester_configured=bool(raw["trusted_attester_configured"]),
        required_fields=tuple(required),
        supported_methods=tuple(methods),
        reset_scope=str(raw["reset_scope"]),
        request_admission_policy=str(raw["request_admission_policy"]),
        request_source_point_reset_protocol_sha256=protocol_sha256,
        runtime_trust_mode=runtime_trust_mode,
        formal_measurement=formal_measurement,
        raw_json=_canonical_json_text(raw),
    )


def _validate_begin_receipt(
    value: object,
    *,
    binding: NativeTerminalRunBinding,
    prior_reset_generation: int,
    prior_process: tuple[int, int] | None,
    legacy: bool = False,
    expected_protocol_sha256: str | None = None,
) -> NativeTerminalBeginReceipt:
    raw = _exact_object(
        value,
        _LEGACY_BEGIN_RECEIPT_KEYS if legacy else _BEGIN_RECEIPT_KEYS,
        "terminal begin receipt",
    )
    expected_hook = (
        LEGACY_NATIVE_TERMINAL_EVIDENCE_HOOK
        if legacy
        else NATIVE_TERMINAL_EVIDENCE_HOOK
    )
    if (
        raw["schema_version"] != (1 if legacy else 2)
        or raw["kind"] != "lightcone_terminal_begin_receipt"
        or raw["hook"] != expected_hook
    ):
        raise ValueError("terminal begin receipt schema/hook mismatch")
    _validate_bound_identity(raw, binding, "terminal begin receipt", legacy=legacy)
    protocol_sha256 = None
    runtime_trust_mode: str | None = None
    formal_measurement: bool | None = None
    if not legacy:
        reset_scope, request_admission_policy = _wire_reset_identity(binding)
        protocol_sha256 = _validate_wire_reset_protocol(
            method=binding.method,
            reset_scope=reset_scope,
            request_admission_policy=request_admission_policy,
            protocol_sha256=raw["request_source_point_reset_protocol_sha256"],
            field="terminal begin receipt",
        )
        if (
            expected_protocol_sha256 is not None
            and protocol_sha256 != expected_protocol_sha256
        ):
            raise ValueError("terminal begin protocol differs from capability")
        runtime_trust_mode, formal_measurement = _validate_runtime_trust_identity(
            method=binding.method,
            runtime_trust_mode=raw["runtime_trust_mode"],
            formal_measurement=raw["formal_measurement"],
            field="terminal begin receipt",
        )
    process_id = _integer(raw["server_process_id"], "server_process_id", minimum=1)
    process_started = _integer(
        raw["server_process_started_ns"], "server_process_started_ns", minimum=1
    )
    if prior_process is not None and (process_id, process_started) != prior_process:
        raise RuntimeError("terminal server process changed inside a session")
    generation = _integer(raw["reset_generation"], "reset_generation", minimum=1)
    if generation != prior_reset_generation + 1:
        raise ValueError("terminal begin reset generation is stale or skipped")
    for name in (
        "prior_state_sha256",
        "reset_state_sha256",
        "warmup_request_ids_sha256",
        "scored_request_ids_sha256",
        "begin_sha256",
    ):
        _sha256(raw[name], name)
    if raw["warmup_request_ids_sha256"] != canonical_sha256(
        list(binding.warmup_request_ids)
    ) or raw["scored_request_ids_sha256"] != canonical_sha256(
        list(binding.scored_request_ids)
    ):
        raise ValueError("terminal begin request-set digest mismatch")
    unsigned = dict(raw)
    digest = unsigned.pop("begin_sha256")
    if canonical_sha256(unsigned) != digest:
        raise ValueError("terminal begin receipt content digest mismatch")
    return NativeTerminalBeginReceipt(
        binding=binding,
        server_process_id=process_id,
        server_process_started_ns=process_started,
        reset_generation=generation,
        request_source_point_reset_protocol_sha256=protocol_sha256,
        runtime_trust_mode=runtime_trust_mode,
        formal_measurement=formal_measurement,
        begin_sha256=str(digest),
        raw_json=_canonical_json_text(raw),
    )


def _server_request_row(request: TerminalRequestExpectation) -> dict[str, object]:
    request.validate()
    if not request.submitted_to_server or request.output_token_ids is None:
        raise ValueError("server row requires a submitted request")
    inputs = list(request.input_token_ids)
    outputs = list(request.output_token_ids)
    row: dict[str, object] = {
        "request_id": request.request_id,
        "terminal_source": "server",
        "input_tokens": len(inputs),
        "input_token_ids_sha256": canonical_sha256(inputs),
        "output_tokens": len(outputs),
        "ordered_output_token_ids": outputs,
        "output_token_ids_sha256": canonical_sha256(outputs),
        "terminal_status": request.terminal_status,
        "terminal_reason": request.terminal_reason,
    }
    row["request_sha256"] = canonical_sha256(row)
    return row


def _client_terminal_row(
    binding: NativeTerminalRunBinding,
    request: TerminalRequestExpectation,
) -> dict[str, object]:
    request.validate()
    if request.submitted_to_server:
        raise ValueError("client reconciliation is only for non-submitted requests")
    row: dict[str, object] = {
        "schema_version": 1,
        "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
        "run_id": binding.run_id,
        "run_nonce_sha256": binding.run_nonce_sha256,
        "execution_plan_sha256": binding.execution_plan_sha256,
        "rank_config_sha256": binding.rank_config_sha256,
        "request_id": request.request_id,
        "terminal_status": request.terminal_status,
        "terminal_reason": request.terminal_reason,
    }
    row["client_terminal_sha256"] = canonical_sha256(row)
    return row


def _reconciled_request_row(
    binding: NativeTerminalRunBinding,
    request: TerminalRequestExpectation,
) -> dict[str, object]:
    row = _client_terminal_row(binding, request)
    row["terminal_source"] = "client_reconciliation"
    row["output_tokens"] = 0
    row["request_sha256"] = canonical_sha256(row)
    return row


def _validate_request_expectations(
    requests: Sequence[TerminalRequestExpectation],
    *,
    expected_ids: tuple[str, ...],
    warmup: bool,
) -> tuple[TerminalRequestExpectation, ...]:
    values = tuple(requests)
    for request in values:
        if not isinstance(request, TerminalRequestExpectation):
            raise TypeError("terminal request expectations are typed records")
        request.validate()
    if tuple(request.request_id for request in values) != expected_ids:
        raise ValueError("terminal request expectations differ from planned order")
    if warmup and any(
        not request.submitted_to_server or request.terminal_status != "completed"
        for request in values
    ):
        raise RuntimeError("warmup must be fully server-observed and completed")
    return values


def _validate_reset_receipt(
    value: object,
    *,
    begin: NativeTerminalBeginReceipt,
    warmup_requests: tuple[TerminalRequestExpectation, ...],
    legacy: bool = False,
) -> NativeTerminalResetReceipt:
    raw = _exact_object(
        value,
        _LEGACY_RESET_RECEIPT_KEYS if legacy else _RESET_RECEIPT_KEYS,
        "terminal reset receipt",
    )
    binding = begin.binding
    expected_hook = (
        LEGACY_NATIVE_TERMINAL_EVIDENCE_HOOK
        if legacy
        else NATIVE_TERMINAL_EVIDENCE_HOOK
    )
    if (
        raw["schema_version"] != (1 if legacy else 2)
        or raw["kind"] != "lightcone_terminal_reset_receipt"
        or raw["hook"] != expected_hook
    ):
        raise ValueError("terminal reset receipt schema/hook mismatch")
    _validate_bound_identity(raw, binding, "terminal reset receipt", legacy=legacy)
    protocol_sha256 = None
    runtime_trust_mode: str | None = None
    formal_measurement: bool | None = None
    if not legacy:
        reset_scope, request_admission_policy = _wire_reset_identity(binding)
        protocol_sha256 = _validate_wire_reset_protocol(
            method=binding.method,
            reset_scope=reset_scope,
            request_admission_policy=request_admission_policy,
            protocol_sha256=raw["request_source_point_reset_protocol_sha256"],
            field="terminal reset receipt",
        )
        if protocol_sha256 != begin.request_source_point_reset_protocol_sha256:
            raise ValueError("terminal reset protocol differs from begin")
        runtime_trust_mode, formal_measurement = _validate_runtime_trust_identity(
            method=binding.method,
            runtime_trust_mode=raw["runtime_trust_mode"],
            formal_measurement=raw["formal_measurement"],
            field="terminal reset receipt",
        )
        if (
            runtime_trust_mode != begin.runtime_trust_mode
            or formal_measurement is not begin.formal_measurement
        ):
            raise ValueError("terminal reset runtime trust differs from begin")
    if (
        raw["server_process_id"] != begin.server_process_id
        or raw["server_process_started_ns"] != begin.server_process_started_ns
        or raw["begin_sha256"] != begin.begin_sha256
    ):
        raise ValueError("terminal reset receipt process/begin identity mismatch")
    generation = _integer(raw["reset_generation"], "reset_generation", minimum=1)
    if generation != begin.reset_generation + 1:
        raise ValueError("terminal reset generation is stale or skipped")
    if (
        raw["prior_trace_run_id"] != binding.previous_run_id
        or raw["next_trace_run_id"] != binding.run_id
    ):
        raise ValueError("terminal reset trace lineage mismatch")
    for name in (
        "warmup_request_rows_sha256",
        "warmup_performance_sha256",
        "discarded_native_sha256",
        "warmup_state_sha256",
        "reset_state_sha256",
        "expected_scored_request_ids_sha256",
        "reset_sha256",
    ):
        _sha256(raw[name], name)
    expected_warmup_rows = [_server_request_row(request) for request in warmup_requests]
    if raw["warmup_request_rows_sha256"] != canonical_sha256(expected_warmup_rows):
        raise ValueError("terminal reset warmup request/token digest mismatch")
    if raw["expected_scored_request_ids_sha256"] != canonical_sha256(
        list(binding.scored_request_ids)
    ):
        raise ValueError("terminal reset scored request-set digest mismatch")
    completion_generation = _integer(
        raw["completion_event_generation"],
        "completion_event_generation",
        minimum=1,
    )
    warmup_resets: NativeRequestSourcePointResets | None = None
    if not legacy:
        warmup_rows = _validate_request_rows(
            raw["warmup_request_rows"],
            binding=binding,
            requests=warmup_requests,
        )
        if warmup_rows != expected_warmup_rows:
            raise ValueError("terminal reset embedded warmup requests differ")
        warmup_performance = _validate_performance(
            raw["warmup_performance_counters"],
            method=binding.method,
            output_tokens=sum(int(row["output_tokens"]) for row in warmup_rows),
        )
        if raw["warmup_performance_sha256"] != canonical_sha256(warmup_performance):
            raise ValueError("terminal reset embedded warmup performance differs")
        warmup_resets = _validate_request_source_point_resets(
            raw["warmup_request_source_point_resets"],
            binding=binding,
            requests=warmup_requests,
        )
        warmup_rounds, _ = _validate_rounds_and_kv(
            raw["warmup_round_rows"],
            raw["warmup_historical_kv_source_versions"],
            binding=binding,
            requests=warmup_requests,
            performance=warmup_performance,
            request_resets=warmup_resets,
        )
        warmup_updates = _validate_updates(
            raw["warmup_update_rows"],
            binding=binding,
            rounds=warmup_rounds,
            performance=warmup_performance,
            request_resets=warmup_resets,
        )
        _validate_request_reset_row_coverage(
            warmup_resets,
            requests=warmup_requests,
            rounds=warmup_rounds,
            updates=warmup_updates,
        )
        warmup_state = _validate_current_state_snapshot(
            raw["warmup_state"],
            binding=binding,
            field="terminal reset warmup state",
            clean=False,
        )
        reset_state = _validate_current_state_snapshot(
            raw["reset_state"],
            binding=binding,
            field="terminal reset clean state",
            clean=True,
        )
        if raw["warmup_state_sha256"] != canonical_sha256(warmup_state) or raw[
            "reset_state_sha256"
        ] != canonical_sha256(reset_state):
            raise ValueError("terminal reset embedded state digest differs")
        if (
            warmup_state["allocator_peak_hbm_bytes"]
            != warmup_performance["peak_hbm_bytes"]
            or reset_state["completion_event_generation"] != completion_generation
            or reset_state["completion_event_generation"]
            <= warmup_state["completion_event_generation"]
        ):
            raise RuntimeError("terminal reset embedded state transition differs")
        if (
            reset_state["adapter_request_epoch"] != 0
            or reset_state["adapter_active_request_id"] is not None
        ):
            raise RuntimeError("terminal reset retained warmup request ownership")
    unsigned = dict(raw)
    digest = unsigned.pop("reset_sha256")
    if canonical_sha256(unsigned) != digest:
        raise ValueError("terminal reset receipt content digest mismatch")
    return NativeTerminalResetReceipt(
        binding=binding,
        server_process_id=begin.server_process_id,
        server_process_started_ns=begin.server_process_started_ns,
        reset_generation=generation,
        completion_event_generation=completion_generation,
        request_source_point_reset_protocol_sha256=protocol_sha256,
        runtime_trust_mode=runtime_trust_mode,
        formal_measurement=formal_measurement,
        reset_sha256=str(digest),
        raw_json=_canonical_json_text(raw),
        warmup_requests=warmup_requests,
        warmup_request_source_point_resets=warmup_resets,
    )


def _request_source_point_resets_from_validated(
    value: Mapping[str, object],
) -> NativeRequestSourcePointResets:
    receipts_value = value["receipts"]
    assert isinstance(receipts_value, list)
    receipts = tuple(
        NativeRequestSourcePointResetReceipt(**row)  # type: ignore[arg-type]
        for row in receipts_value
    )
    return NativeRequestSourcePointResets(
        reset_scope=str(value["reset_scope"]),
        request_admission_policy=str(value["request_admission_policy"]),
        protocol_sha256=(
            None if value["protocol_sha256"] is None else str(value["protocol_sha256"])
        ),
        final_archive_sha256=str(value["final_archive_sha256"]),
        receipts=receipts,
    )


def _validate_request_source_point_resets(
    value: object,
    *,
    binding: NativeTerminalRunBinding,
    requests: tuple[TerminalRequestExpectation, ...],
) -> NativeRequestSourcePointResets:
    raw = _exact_object(
        value,
        _REQUEST_SOURCE_POINT_RESET_KEYS,
        "request source-point resets",
    )
    if raw["schema_version"] != 1:
        raise ValueError("request source-point reset schema is unsupported")
    reset_scope, request_admission_policy = _wire_reset_identity(binding)
    if (
        raw["reset_scope"] != reset_scope
        or raw["request_admission_policy"] != request_admission_policy
    ):
        raise ValueError("request source-point reset identity differs from binding")
    final_archive = _sha256(raw["final_archive_sha256"], "request reset final archive")
    receipt_values = raw["receipts"]
    if not isinstance(receipt_values, list):
        raise TypeError("request source-point reset receipts must be an array")
    if reset_scope == "none":
        if (
            raw["protocol_sha256"] is not None
            or receipt_values
            or final_archive != (_ZERO_SHA256)
        ):
            raise RuntimeError("allocation-free terminal reports request reset state")
        return _request_source_point_resets_from_validated(raw)
    protocol = _validate_wire_reset_protocol(
        method=binding.method,
        reset_scope=reset_scope,
        request_admission_policy=request_admission_policy,
        protocol_sha256=raw["protocol_sha256"],
        field="request source-point resets",
    )
    assert protocol is not None
    if reset_scope == "cohort":
        if receipt_values or final_archive != _ZERO_SHA256:
            raise RuntimeError("cohort-scoped terminal reports per-request resets")
        return _request_source_point_resets_from_validated(raw)

    submitted = tuple(request for request in requests if request.submitted_to_server)
    expected_by_id = {request.request_id: request for request in submitted}
    if len(receipt_values) != len(expected_by_id):
        raise RuntimeError("request reset receipt coverage differs from submission")
    normalized: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    next_acquired_epoch = 1
    previous_receipt = _ZERO_SHA256
    previous_archive = _ZERO_SHA256
    source_point_identity: str | None = None
    for receipt_value in receipt_values:
        receipt = _exact_object(
            receipt_value,
            _REQUEST_SOURCE_POINT_RESET_RECEIPT_KEYS,
            "request source-point reset receipt",
        )
        request_id = _safe_id(receipt["request_id"], "request reset request_id")
        if request_id not in expected_by_id or request_id in seen_ids:
            raise RuntimeError("request reset receipt is duplicated or non-submitted")
        seen_ids.add(str(request_id))
        expectation = expected_by_id[str(request_id)]
        if receipt["terminal_outcome"] != expectation.terminal_status or receipt[
            "terminal_outcome"
        ] not in {"completed", "aborted"}:
            raise RuntimeError("request reset receipt terminal outcome differs")
        request_epoch = _integer(
            receipt["request_epoch"], "request reset epoch", minimum=0
        )
        terminal_round = _integer(
            receipt["terminal_round"], "request reset terminal round", minimum=0
        )
        terminal_version = _integer(
            receipt["terminal_version"], "request reset terminal version", minimum=0
        )
        archived_update_count = _integer(
            receipt["archived_update_count"],
            "request reset archived update count",
            minimum=0,
        )
        archived_round_count = _integer(
            receipt["archived_round_count"],
            "request reset archived round count",
            minimum=0,
        )
        acquired = _boolean(
            receipt["adaptation_state_acquired"], "request reset acquired"
        )
        reset_required = _boolean(receipt["reset_required"], "request reset required")
        state_untouched = _boolean(
            receipt["state_untouched"], "request reset state untouched"
        )
        predicates = tuple(
            receipt[name]
            for name in (
                "master_reset",
                "optimizer_reset",
                "inference_reset",
                "captured_state_empty",
                "runtime_reset",
            )
        )
        sticky_reason = receipt["sticky_disabled_reason"]
        if sticky_reason is not None and (
            not isinstance(sticky_reason, str) or not sticky_reason
        ):
            raise ValueError("request reset sticky reason is malformed")
        source_point = _sha256(
            receipt["source_point_identity_sha256"],
            "request reset source-point identity",
        )
        if source_point_identity is None:
            source_point_identity = source_point
        elif source_point != source_point_identity:
            raise RuntimeError("request reset source-point identity changed")
        evidence_archive = _sha256(
            receipt["evidence_archive_sha256"], "request reset evidence archive"
        )
        if receipt["previous_receipt_sha256"] != previous_receipt:
            raise RuntimeError("request reset receipt chain is not contiguous")
        _sha256(receipt["previous_receipt_sha256"], "previous request reset receipt")
        if receipt["protocol_sha256"] != protocol:
            raise ValueError("request reset receipt protocol differs")
        declared = _sha256(receipt["receipt_sha256"], "request reset receipt")
        unsigned = dict(receipt)
        unsigned.pop("receipt_sha256")
        if canonical_sha256(unsigned) != declared:
            raise ValueError("request reset receipt content digest differs")
        if acquired:
            if (
                not reset_required
                or state_untouched
                or request_epoch != next_acquired_epoch
                or any(value is not True for value in predicates)
            ):
                raise RuntimeError("acquired request lacks a required reset epoch")
            next_acquired_epoch += 1
        elif (
            reset_required
            or not state_untouched
            or any(value is not None for value in predicates)
            or request_epoch != 0
            or receipt["terminal_outcome"] != "aborted"
            or terminal_round != 0
            or terminal_version != 0
            or archived_update_count != 0
            or archived_round_count != 0
            or evidence_archive != previous_archive
        ):
            raise RuntimeError("unacquired submitted abort reset receipt is invalid")
        previous_receipt = declared
        previous_archive = evidence_archive
        normalized.append(receipt)
    if set(expected_by_id) != seen_ids:
        raise RuntimeError("request reset receipt submission coverage is incomplete")
    if final_archive != previous_archive:
        raise RuntimeError("request reset final archive differs from receipt chain")
    raw["receipts"] = normalized
    return _request_source_point_resets_from_validated(raw)


def _validate_request_rows(
    value: object,
    *,
    binding: NativeTerminalRunBinding,
    requests: tuple[TerminalRequestExpectation, ...],
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("terminal request rows must be a JSON list")
    if len(value) != len(requests):
        raise RuntimeError("terminal request coverage is incomplete")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_value, expected in zip(value, requests, strict=True):
        if (
            not expected.submitted_to_server
            and isinstance(raw_value, dict)
            and raw_value.get("terminal_source") == "server"
        ):
            raise RuntimeError(
                "server-observed row cannot override client non-submission"
            )
        keys = (
            _SERVER_REQUEST_KEYS
            if expected.submitted_to_server
            else _RECONCILED_REQUEST_KEYS
        )
        raw = _exact_object(raw_value, keys, "terminal request row")
        request_id = _safe_id(raw["request_id"], "request_id")
        if request_id != expected.request_id or request_id in seen:
            raise RuntimeError(
                "terminal request order/identity is duplicated or foreign"
            )
        seen.add(str(request_id))
        digest = _sha256(raw["request_sha256"], "request_sha256")
        unsigned = dict(raw)
        unsigned.pop("request_sha256")
        if canonical_sha256(unsigned) != digest:
            raise ValueError("terminal request content digest mismatch")
        expected_row = (
            _server_request_row(expected)
            if expected.submitted_to_server
            else _reconciled_request_row(binding, expected)
        )
        if raw != expected_row:
            raise ValueError(
                "terminal request tokens/status differ from caller evidence"
            )
        normalized.append(raw)
    # Scored IDs are non-empty by binding invariant; a source-owned run may
    # legitimately configure no warmup requests.
    if not seen and requests:  # pragma: no cover - scored IDs are non-empty
        raise RuntimeError("terminal request evidence is empty")
    return normalized


def _validate_performance(
    value: object,
    *,
    method: str,
    output_tokens: int,
) -> dict[str, object]:
    raw = _exact_object(value, _PERFORMANCE_KEYS, "performance_counters")
    for field in _PERFORMANCE_INTEGER_KEYS:
        nullable = field in {
            "accepted_drafts",
            "committed_tokens",
            "verified_drafts",
            "optimizer_bytes",
            "trainable_parameters",
            "updates_launched",
            "updates_published",
            "exactness_violations",
            "version_mismatches",
            "fallbacks",
            "nonfinite_updates",
        }
        if raw[field] is None and nullable:
            continue
        _integer(raw[field], f"performance.{field}")
    numeric_fields = (
        _PERFORMANCE_KEYS
        - _PERFORMANCE_INTEGER_KEYS
        - {
            "adaptation_memory_ledger",
            "collective_type",
        }
    )
    for field in numeric_fields:
        _number(raw[field], f"performance.{field}", nullable=True)
    if raw["collective_type"] != "none_tp1":
        raise RuntimeError("terminal performance is not the supported TP1 path")
    if (
        raw["collective_bytes"] != 0
        or float(raw["collective_duration_ms"]) != 0.0
        or float(raw["collective_exposed_wait_ms"]) != 0.0
        or raw["collective_overlap_ratio"] is not None
    ):
        raise RuntimeError("TP1 terminal evidence reports a fabricated collective")
    target_calls = int(raw["target_calls"])
    expected_target_ratio = target_calls / output_tokens if output_tokens else None
    _require_ratio(
        raw["target_calls_per_output_token"],
        expected_target_ratio,
        "target_calls_per_output_token",
    )
    allocation_zero = (
        "optimizer_bytes",
        "trainable_parameters",
        "updates_launched",
        "updates_published",
    )
    allocation_null = (
        "adaptation_memory_ledger",
        "training_cuda_ms",
        "optimizer_cuda_ms",
        "merge_cuda_ms",
        "publish_cuda_ms",
        "barrier_cuda_ms",
        "exposed_update_ms",
        "main_side_overlap_ratio",
    )
    safety = (
        "exactness_violations",
        "version_mismatches",
        "fallbacks",
        "nonfinite_updates",
    )
    speculative_counts = (
        "accepted_drafts",
        "committed_tokens",
        "verified_drafts",
    )
    if method in {"target_only", "static"} and (
        any(raw[field] != 0 for field in allocation_zero)
        or any(raw[field] is not None for field in allocation_null)
    ):
        raise RuntimeError("allocation-free performance reports adaptation state")
    if method == "target_only":
        if target_calls != output_tokens:
            raise RuntimeError("Target-only target-call accounting is not exact")
        if any(raw[field] is not None for field in (*speculative_counts, *safety)):
            raise RuntimeError("Target-only performance reports speculative state")
        for field in (
            "accepted_drafts_per_verify",
            "committed_tokens_per_verify",
            "verified_drafts_per_verify",
            "verification_waste",
        ):
            if raw[field] is not None:
                raise RuntimeError("Target-only performance reports speculative ratios")
    else:
        for field in (*speculative_counts, *safety):
            _integer(raw[field], f"performance.{field}")
        accepted = int(raw["accepted_drafts"])
        committed = int(raw["committed_tokens"])
        verified = int(raw["verified_drafts"])
        if accepted > verified or committed < accepted:
            raise RuntimeError("speculative aggregate counts are inconsistent")
        if accepted != max(committed - target_calls, 0):
            raise RuntimeError("accepted/committed aggregate closure is inconsistent")
        _require_ratio(
            raw["accepted_drafts_per_verify"],
            accepted / target_calls if target_calls else None,
            "accepted_drafts_per_verify",
        )
        _require_ratio(
            raw["committed_tokens_per_verify"],
            committed / target_calls if target_calls else None,
            "committed_tokens_per_verify",
        )
        _require_ratio(
            raw["verified_drafts_per_verify"],
            verified / target_calls if target_calls else None,
            "verified_drafts_per_verify",
        )
        if raw["verification_waste"] != verified - accepted:
            raise RuntimeError("verification-waste aggregate closure is inconsistent")
    if method in _ADAPTIVE_METHODS:
        for field in (
            "optimizer_bytes",
            "trainable_parameters",
            "updates_launched",
            "updates_published",
        ):
            _integer(raw[field], f"performance.{field}")
        if raw["adaptation_memory_ledger"] is None:
            raise RuntimeError("adapted performance lacks its memory ledger")
        ledger = _exact_object(
            raw["adaptation_memory_ledger"],
            _ADAPTATION_MEMORY_LEDGER_KEYS,
            "adaptation memory ledger",
        )
        for field in _ADAPTATION_MEMORY_LEDGER_KEYS:
            _integer(ledger[field], f"adaptation_memory_ledger.{field}")
        resident_fields = (
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
        scratch_fields = (
            "gradient_bytes",
            "training_activation_bytes",
            "kv_gather_scratch_bytes",
            "candidate_scratch_bytes",
        )
        optimizer_fields = (
            "master_fp32_bytes",
            "first_moment_bytes",
            "second_moment_bytes",
            "online_state_bytes",
            "optimizer_metadata_bytes",
        )
        if ledger["resident_bytes"] != sum(ledger[name] for name in resident_fields):
            raise RuntimeError("adaptation resident memory ledger does not sum")
        if ledger["peak_bytes"] != ledger["resident_bytes"] + sum(
            ledger[name] for name in scratch_fields
        ):
            raise RuntimeError("adaptation peak memory ledger does not sum")
        if (
            ledger["optimizer_bytes"] != sum(ledger[name] for name in optimizer_fields)
            or ledger["optimizer_bytes"] != raw["optimizer_bytes"]
        ):
            raise RuntimeError("adaptation optimizer memory ledger does not close")
        if ledger["peak_bytes"] > raw["peak_hbm_bytes"]:
            raise RuntimeError("adaptation peak memory exceeds observed peak HBM")
    return raw


def _require_ratio(value: object, expected: float | None, field: str) -> None:
    if expected is None:
        if value is not None:
            raise RuntimeError(f"{field} must be null without a denominator")
        return
    actual = _number(value, field)
    if actual is None or not math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RuntimeError(f"{field} disagrees with exact aggregate counts")


def _validate_current_state_snapshot(
    value: object,
    *,
    binding: NativeTerminalRunBinding,
    field: str,
    clean: bool,
) -> dict[str, object]:
    raw = _exact_object(value, _STATE_KEYS, field)
    if raw["schema_version"] != 2:
        raise ValueError(f"{field} schema is unsupported")
    for integer_field in (
        "active_requests",
        "queued_requests",
        "request_pool_active_slots",
        "allocator_current_hbm_bytes",
        "allocator_reserved_hbm_bytes",
        "allocator_peak_hbm_bytes",
        "kv_token_capacity",
        "kv_available_tokens",
        "adapter_active_version",
        "adapter_epoch",
        "optimizer_generation",
        "telemetry_generation",
        "completion_event_generation",
        "adapter_request_epoch",
        "adapter_source_round",
    ):
        _integer(raw[integer_field], f"{field}.{integer_field}")
    for digest_field in ("kv_state_sha256", "rng_state_sha256"):
        _sha256(raw[digest_field], f"{field}.{digest_field}")
    if raw["adapter_state_sha256"] is not None:
        _sha256(raw["adapter_state_sha256"], f"{field}.adapter_state_sha256")
    _safe_id(
        raw["adapter_active_request_id"],
        f"{field}.adapter_active_request_id",
        nullable=True,
    )
    reset_scope, request_admission_policy = _wire_reset_identity(binding)
    _validate_wire_reset_protocol(
        method=binding.method,
        reset_scope=raw["adapter_reset_scope"],
        request_admission_policy=raw["adapter_request_admission_policy"],
        protocol_sha256=raw["adapter_request_source_point_reset_protocol_sha256"],
        field=field,
    )
    runtime_trust_mode, formal_measurement = _validate_runtime_trust_identity(
        method=binding.method,
        runtime_trust_mode=raw["adapter_runtime_trust_mode"],
        formal_measurement=raw["adapter_formal_measurement"],
        field=field,
    )
    if (
        raw["adapter_reset_scope"] != reset_scope
        or raw["adapter_request_admission_policy"] != request_admission_policy
        or runtime_trust_mode != binding.runtime_trust_mode
        or formal_measurement is not binding.formal_measurement
    ):
        raise RuntimeError(f"{field} reset/runtime trust identity differs")
    for boolean_field in (
        "scheduler_idle",
        "adapter_reset_verified",
        "completion_event_complete",
    ):
        _boolean(raw[boolean_field], f"{field}.{boolean_field}")
    if raw["allocator_peak_hbm_bytes"] < raw["allocator_current_hbm_bytes"]:
        raise ValueError(f"{field} allocator peak is below current allocation")
    if raw["kv_available_tokens"] > raw["kv_token_capacity"]:
        raise ValueError(f"{field} KV availability exceeds capacity")
    if clean and (
        not raw["scheduler_idle"]
        or raw["active_requests"] != 0
        or raw["queued_requests"] != 0
        or raw["request_pool_active_slots"] != 0
        or raw["kv_available_tokens"] != raw["kv_token_capacity"]
        or raw["adapter_active_version"] != 0
        or raw["adapter_source_round"] != 0
        or raw["adapter_active_request_id"] is not None
        or raw["optimizer_generation"] != 0
        or not raw["adapter_reset_verified"]
        or not raw["completion_event_complete"]
    ):
        raise RuntimeError(f"{field} is not a clean server state")
    return raw


def _validate_state(
    value: object,
    *,
    binding: NativeTerminalRunBinding,
    reset: NativeTerminalResetReceipt,
    performance: Mapping[str, object],
    request_resets: NativeRequestSourcePointResets | None,
    legacy: bool,
) -> dict[str, object]:
    if legacy:
        raw = _exact_object(value, _LEGACY_STATE_KEYS, "terminal final state")
        if raw["schema_version"] != 1:
            raise ValueError("terminal final-state schema is unsupported")
        for integer_field in (
            "active_requests",
            "queued_requests",
            "request_pool_active_slots",
            "allocator_current_hbm_bytes",
            "allocator_reserved_hbm_bytes",
            "allocator_peak_hbm_bytes",
            "kv_token_capacity",
            "kv_available_tokens",
            "adapter_active_version",
            "adapter_epoch",
            "optimizer_generation",
            "telemetry_generation",
            "completion_event_generation",
        ):
            _integer(raw[integer_field], f"final_state.{integer_field}")
        for digest_field in (
            "kv_state_sha256",
            "rng_state_sha256",
            "adapter_state_sha256",
        ):
            _sha256(raw[digest_field], f"final_state.{digest_field}")
        for boolean_field in (
            "scheduler_idle",
            "adapter_reset_verified",
            "completion_event_complete",
        ):
            _boolean(raw[boolean_field], f"final_state.{boolean_field}")
    else:
        raw = _validate_current_state_snapshot(
            value,
            binding=binding,
            field="terminal final state",
            clean=False,
        )
    if (
        not raw["scheduler_idle"]
        or raw["active_requests"] != 0
        or raw["queued_requests"] != 0
        or raw["request_pool_active_slots"] != 0
        or not raw["completion_event_complete"]
    ):
        raise RuntimeError("terminal final state is not drained and synchronized")
    if raw["allocator_peak_hbm_bytes"] != performance["peak_hbm_bytes"]:
        raise RuntimeError(
            "terminal allocator peak differs from performance peak HBM evidence"
        )
    if raw["completion_event_generation"] <= reset.completion_event_generation:
        raise ValueError("terminal completion event does not follow reset")
    method = binding.method
    if method in {"target_only", "static"} and (
        raw["adapter_active_version"] != 0
        or raw["adapter_epoch"] != 0
        or raw["optimizer_generation"] != 0
        or not raw["adapter_reset_verified"]
    ):
        raise RuntimeError("allocation-free final state reports adaptation mutation")
    if method in _ADAPTIVE_METHODS:
        if not legacy and binding.reset_scope == "request":
            if request_resets is None:
                raise RuntimeError("request-scoped final state lacks reset evidence")
            acquired_epoch = max(
                (
                    receipt.request_epoch
                    for receipt in request_resets.receipts
                    if receipt.adaptation_state_acquired
                ),
                default=0,
            )
            if (
                raw["adapter_active_version"] != 0
                or raw["optimizer_generation"] != 0
                or not raw["adapter_reset_verified"]
                or raw["adapter_active_request_id"] is not None
                or raw["adapter_request_epoch"] != acquired_epoch
                or raw["adapter_source_round"] != 0
            ):
                raise RuntimeError(
                    "request-scoped final adaptation state is not at source point"
                )
        elif not legacy and (
            raw["adapter_active_request_id"] is not None
            or raw["adapter_request_epoch"] != 0
        ):
            raise RuntimeError("cohort-scoped final state reports request ownership")
        elif (
            raw["adapter_active_version"] != performance["updates_published"]
            or raw["optimizer_generation"] != performance["updates_published"]
        ):
            raise RuntimeError("adapted final versions disagree with published updates")
    return raw


def _validate_rounds_and_kv(
    rounds_value: object,
    kv_value: object,
    *,
    binding: NativeTerminalRunBinding,
    requests: tuple[TerminalRequestExpectation, ...],
    performance: Mapping[str, object] | None,
    request_resets: NativeRequestSourcePointResets | None = None,
    legacy: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not isinstance(rounds_value, list):
        raise TypeError("terminal round rows must be a JSON list")
    if not isinstance(kv_value, dict):
        raise TypeError("terminal historical KV map must be a JSON object")
    _validate_strict_json(kv_value, "historical_kv_source_versions")
    if binding.method in {"target_only", "static"}:
        if rounds_value or kv_value:
            raise RuntimeError("allocation-free evidence contains detailed rounds/KV")
        return [], {}
    kv_segment_sha256s = {
        str(request_id): canonical_sha256(segments)
        for request_id, segments in kv_value.items()
    }
    expectation = {request.request_id: request for request in requests}
    server_ids = {
        request.request_id for request in requests if request.submitted_to_server
    }
    rows: list[dict[str, object]] = []
    identities: set[tuple[int, str, int]] = set()
    by_request: dict[str, list[dict[str, object]]] = {}
    for value in rounds_value:
        raw = _exact_object(
            value,
            _LEGACY_ROUND_KEYS if legacy else _ROUND_KEYS,
            "terminal round row",
        )
        request_id = _safe_id(raw["request_id"], "round.request_id")
        if request_id not in server_ids:
            raise RuntimeError("terminal round references a non-submitted request")
        round_index = _integer(raw["round_index"], "round.round_index", minimum=1)
        request_epoch = 0
        if not legacy:
            request_epoch = _integer(
                raw["request_epoch"], "round.request_epoch", minimum=0
            )
        identity = (request_epoch, str(request_id), round_index)
        if identity in identities:
            raise RuntimeError("terminal round identity is duplicated")
        identities.add(identity)
        if not legacy:
            assert request_resets is not None
            row_scope = raw["reset_scope"]
            receipt_sha256 = raw["request_reset_receipt_sha256"]
            if row_scope != request_resets.reset_scope:
                raise RuntimeError("terminal round reset scope differs")
            if row_scope == "request":
                receipt = request_resets.receipt_by_sha256.get(str(receipt_sha256))
                if (
                    receipt is None
                    or receipt.request_id != request_id
                    or not receipt.adaptation_state_acquired
                    or receipt.request_epoch != request_epoch
                ):
                    raise RuntimeError("terminal round lacks its exact request reset")
            elif request_epoch != 0 or receipt_sha256 is not None:
                raise RuntimeError("cohort round carries request reset identity")
        for field in (
            "proposal_source_version",
            "prefix_len_before",
            "verify_len",
            "accepted_drafts",
            "committed_tokens",
            "target_calls",
        ):
            _integer(raw[field], f"round.{field}")
        if raw["target_calls"] != 1 or raw["accepted_drafts"] > raw["verify_len"]:
            raise RuntimeError("terminal round speculative counts are inconsistent")
        committed = int(raw["committed_tokens"])
        accepted = int(raw["accepted_drafts"])
        if committed not in {0, accepted, accepted + 1}:
            raise RuntimeError("terminal round commit/accept closure is inconsistent")
        if legacy:
            if not isinstance(raw["historical_kv_source_versions"], list):
                raise TypeError("terminal round historical KV value is malformed")
            if raw["historical_kv_source_versions"] != kv_value.get(request_id):
                raise RuntimeError(
                    "terminal round historical KV differs from terminal map"
                )
        elif raw["historical_kv_source_versions_sha256"] != kv_segment_sha256s.get(
            str(request_id)
        ):
            raise RuntimeError("terminal round historical KV digest differs from map")
        digest = _sha256(raw["round_sha256"], "round_sha256")
        unsigned = dict(raw)
        unsigned.pop("round_sha256")
        if canonical_sha256(unsigned) != digest:
            raise ValueError("terminal round content digest mismatch")
        rows.append(raw)
        by_request.setdefault(str(request_id), []).append(raw)
    if set(kv_value) != set(by_request):
        raise RuntimeError("historical KV request coverage differs from rounds")
    for request_id, request_rows in by_request.items():
        request_rows.sort(key=lambda row: int(row["round_index"]))
        if tuple(int(row["round_index"]) for row in request_rows) != tuple(
            range(1, len(request_rows) + 1)
        ):
            raise RuntimeError("terminal request round indexes are not contiguous")
        if (
            not legacy
            and request_resets is not None
            and (request_resets.reset_scope == "request")
        ):
            versions = tuple(
                int(row["proposal_source_version"]) for row in request_rows
            )
            if versions[0] != 0 or any(
                right < left or right > left + 1
                for left, right in itertools.pairwise(versions)
            ):
                raise RuntimeError("request-scoped proposal versions did not restart")
        request = expectation[request_id]
        if request.output_token_ids is None:  # submitted invariant
            raise RuntimeError("submitted request lost output token IDs")
        prefix = len(request.input_token_ids)
        for row in request_rows:
            if row["prefix_len_before"] != prefix:
                raise RuntimeError("terminal round prefixes are not contiguous")
            prefix += int(row["committed_tokens"])
        if prefix != len(request.input_token_ids) + len(request.output_token_ids):
            raise RuntimeError("terminal rounds do not reconstruct exact output tokens")
        segments = kv_value[request_id]
        if not isinstance(segments, list) or not segments:
            raise ValueError("historical KV request has no segments")
        previous_end = 0
        previous_version = -1
        for segment in segments:
            segment = _exact_object(
                segment,
                {"start", "end", "source_version"},
                "historical KV segment",
            )
            start = _integer(segment["start"], "kv.start")
            end = _integer(segment["end"], "kv.end", minimum=1)
            version = _integer(segment["source_version"], "kv.source_version")
            if start != previous_end or end <= start or version < previous_version:
                raise RuntimeError(
                    "historical KV segments are not contiguous/versioned"
                )
            previous_end = end
            previous_version = version
        if previous_end != prefix:
            raise RuntimeError("historical KV does not cover the reconstructed request")
    if performance is not None:
        for performance_field, round_field in (
            ("target_calls", "target_calls"),
            ("accepted_drafts", "accepted_drafts"),
            ("committed_tokens", "committed_tokens"),
            ("verified_drafts", "verify_len"),
        ):
            observed = sum(int(row[round_field]) for row in rows)
            if (legacy and performance[performance_field] != observed) or (
                not legacy and performance[performance_field] < observed
            ):
                raise RuntimeError(
                    "terminal round aggregates disagree with performance"
                )
    return rows, dict(kv_value)


def _validate_updates(
    value: object,
    *,
    binding: NativeTerminalRunBinding,
    rounds: list[dict[str, object]],
    performance: Mapping[str, object] | None,
    request_resets: NativeRequestSourcePointResets | None = None,
    legacy: bool = False,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("terminal update rows must be a JSON list")
    if binding.method in {"target_only", "static"}:
        if value:
            raise RuntimeError("allocation-free evidence contains update rows")
        return []
    round_index = {
        (str(row["request_id"]), int(row["round_index"])): row for row in rounds
    }
    rows: list[dict[str, object]] = []
    allowed_status = {
        "published",
        "version_conflict",
        "request_aborted",
        "request_completed_before_publication",
        "no_supervision",
        "nonfinite_update",
        "reconstruction_mismatch",
    }
    seen_source_identities: set[tuple[int, int, int]] = set()
    optimizer_steps_by_epoch: dict[int, list[int]] = {}
    for expected_index, value_row in enumerate(value):
        raw = _exact_object(
            value_row,
            _LEGACY_UPDATE_KEYS if legacy else _UPDATE_KEYS,
            "terminal update row",
        )
        if raw["update_index"] != expected_index:
            raise RuntimeError("terminal update indexes are not contiguous")
        _sha256(raw["cohort_sha256"], "update.cohort_sha256")
        _sha256(raw["parameter_layout_sha256"], "update.parameter_layout_sha256")
        for field, minimum in (
            ("cohort_epoch", 0),
            ("source_round", 1),
            ("source_version", 0),
            ("optimizer_step", 1),
        ):
            _integer(raw[field], f"update.{field}", minimum=minimum)
        request_ids = raw["request_ids"]
        prefixes = raw["prefix_len_before"]
        if (
            not isinstance(request_ids, list)
            or not request_ids
            or not isinstance(prefixes, list)
            or len(prefixes) != len(request_ids)
        ):
            raise ValueError("terminal update request/prefix coverage is malformed")
        if len(request_ids) != len(set(request_ids)):
            raise RuntimeError("terminal update request IDs are duplicated")
        request_epoch = 0
        if not legacy:
            assert request_resets is not None
            request_epoch = _integer(
                raw["request_epoch"], "update.request_epoch", minimum=0
            )
            receipt_sha256 = raw["request_reset_receipt_sha256"]
            if raw["reset_scope"] != request_resets.reset_scope:
                raise RuntimeError("terminal update reset scope differs")
            if request_resets.reset_scope == "request":
                if len(request_ids) != 1:
                    raise RuntimeError("request-scoped update is not request-exclusive")
                receipt = request_resets.receipt_by_sha256.get(str(receipt_sha256))
                if (
                    receipt is None
                    or receipt.request_id != request_ids[0]
                    or not receipt.adaptation_state_acquired
                    or receipt.request_epoch != request_epoch
                ):
                    raise RuntimeError("terminal update lacks its exact request reset")
                optimizer_steps_by_epoch.setdefault(request_epoch, []).append(
                    int(raw["optimizer_step"])
                )
            elif request_epoch != 0 or receipt_sha256 is not None:
                raise RuntimeError("cohort update carries request reset identity")
        for request_id, prefix in zip(request_ids, prefixes, strict=True):
            _safe_id(request_id, "update.request_id")
            _integer(prefix, "update.prefix_len_before")
            source = round_index.get((request_id, int(raw["source_round"])))
            if (
                source is None
                or source["prefix_len_before"] != prefix
                or source["proposal_source_version"] != raw["source_version"]
            ):
                raise RuntimeError("terminal update lacks its exact source round")
        status = _safe_id(raw["status"], "update.status")
        if status not in allowed_status:
            raise RuntimeError("terminal update status is incomplete or pending")
        published = raw["published_version"]
        if published is not None:
            _integer(published, "update.published_version", minimum=1)
        if (status == "published") != (published is not None):
            raise RuntimeError("terminal update publication identity is inconsistent")
        if not legacy:
            _number(
                raw["effective_learning_rate"],
                "update.effective_learning_rate",
                minimum=0,
            )
            _boolean(raw["schedule_valid"], "update.schedule_valid")
            intrinsic_ready_round = raw["intrinsic_ready_round"]
            if intrinsic_ready_round is not None:
                _integer(
                    intrinsic_ready_round,
                    "update.intrinsic_ready_round",
                    minimum=0,
                )
            logical_delay = _integer(
                raw["extra_logical_delay"],
                "update.extra_logical_delay",
                minimum=0,
            )
            publication_round = raw["publication_round"]
            if publication_round is not None:
                _integer(
                    publication_round,
                    "update.publication_round",
                    minimum=0,
                )
            if status == "published" and (
                not raw["schedule_valid"]
                or intrinsic_ready_round is None
                or publication_round is None
                or publication_round < intrinsic_ready_round + logical_delay
            ):
                raise RuntimeError("published update violates its logical schedule")
        for field in ("reconstruction_ok", "supervision_nonempty"):
            _boolean(raw[field], f"update.{field}")
        _number(raw["loss"], "update.loss", minimum=-1e-6)
        _number(raw["gradient_norm"], "update.gradient_norm", minimum=0)
        _number(
            raw["reconstruction_max_abs"],
            "update.reconstruction_max_abs",
            minimum=0,
        )
        for field in (
            "reconstruction_relative_rms",
            "reconstruction_top1_match",
            "reconstruction_mean_kl",
            "online_hint_error",
            "online_ensemble_entropy",
            "online_effective_experts",
        ):
            _number(raw[field], f"update.{field}", nullable=True, minimum=0)
        replay_digest_fields = (
            "source_state_sha256",
            "candidate_bytes_sha256",
            "optimizer_state_bytes_sha256",
            "proposal_evidence_sha256",
        )
        replay_digests = tuple(raw[field] for field in replay_digest_fields)
        if any(value is None for value in replay_digests) and any(
            value is not None for value in replay_digests
        ):
            raise ValueError("terminal mechanism-replay byte digests must be atomic")
        for field, digest_value in zip(
            replay_digest_fields, replay_digests, strict=True
        ):
            if digest_value is not None:
                _sha256(digest_value, f"update.{field}")
        source_identity = (
            request_epoch,
            int(raw["source_round"]),
            int(raw["source_version"]),
        )
        if source_identity in seen_source_identities:
            raise RuntimeError(
                "terminal update duplicates a source-round/version identity"
            )
        seen_source_identities.add(source_identity)
        if (
            raw["reconstruction_top1_match"] is not None
            and raw["reconstruction_top1_match"] > 1
        ):
            raise ValueError("terminal reconstruction top-1 match exceeds one")
        online_fields = (
            "online_hint_error",
            "online_ensemble_entropy",
            "online_effective_experts",
            "online_expert_probabilities",
            "online_cumulative_losses",
            "online_expert_gradient_norms",
        )
        if any(raw[field] is not None for field in online_fields):
            raise RuntimeError("TTS/L0 update contains foreign OnlineSPEC diagnostics")
        digest = _sha256(raw["update_sha256"], "update_sha256")
        unsigned = dict(raw)
        unsigned.pop("update_sha256")
        if canonical_sha256(unsigned) != digest:
            raise ValueError("terminal update content digest mismatch")
        rows.append(raw)
    for request_epoch, optimizer_steps in optimizer_steps_by_epoch.items():
        if optimizer_steps != list(range(1, len(optimizer_steps) + 1)):
            raise RuntimeError(
                f"request epoch {request_epoch} optimizer steps did not restart"
            )
    if performance is not None:
        if performance["updates_launched"] != len(rows):
            raise RuntimeError("terminal launched-update aggregate differs from rows")
        published_count = sum(row["status"] == "published" for row in rows)
        if performance["updates_published"] != published_count:
            raise RuntimeError("terminal published-update aggregate differs from rows")
        lower_bounds = {
            "version_mismatches": sum(
                row["status"] == "version_conflict" for row in rows
            ),
            "nonfinite_updates": sum(
                row["status"] == "nonfinite_update" for row in rows
            ),
            "exactness_violations": sum(
                row["status"] == "reconstruction_mismatch" for row in rows
            ),
            "fallbacks": sum(
                row["status"] == "reconstruction_mismatch" for row in rows
            ),
        }
        if any(performance[field] < minimum for field, minimum in lower_bounds.items()):
            raise RuntimeError("terminal safety aggregate undercounts update rows")
    return rows


def _validate_request_reset_row_coverage(
    resets: NativeRequestSourcePointResets,
    *,
    requests: tuple[TerminalRequestExpectation, ...],
    rounds: list[dict[str, object]],
    updates: list[dict[str, object]],
) -> None:
    if resets.reset_scope != "request":
        return
    submitted_request_ids = {
        row.request_id for row in requests if row.submitted_to_server
    }
    rounds_by_receipt: dict[str, list[dict[str, object]]] = {}
    updates_by_receipt: dict[str, list[dict[str, object]]] = {}
    for row in rounds:
        rounds_by_receipt.setdefault(
            str(row["request_reset_receipt_sha256"]), []
        ).append(row)
    for row in updates:
        updates_by_receipt.setdefault(
            str(row["request_reset_receipt_sha256"]), []
        ).append(row)
    known_receipts = resets.receipt_by_sha256
    if not set(rounds_by_receipt) <= set(known_receipts) or not set(
        updates_by_receipt
    ) <= set(known_receipts):
        raise RuntimeError("terminal rows reference an unknown request reset receipt")
    archive_sha256 = _ZERO_SHA256
    for receipt in resets.receipts:
        if receipt.request_id not in submitted_request_ids:
            raise RuntimeError("request reset receipt has a foreign submitted request")
        receipt_rounds = rounds_by_receipt.get(receipt.receipt_sha256, [])
        receipt_updates = updates_by_receipt.get(receipt.receipt_sha256, [])
        if receipt.archived_round_count != len(
            receipt_rounds
        ) or receipt.archived_update_count != len(receipt_updates):
            raise RuntimeError("request reset archived row counts differ")
        terminal_round = max(
            (int(row["round_index"]) for row in receipt_rounds), default=0
        )
        terminal_version = max(
            (
                int(row["published_version"])
                for row in receipt_updates
                if row["published_version"] is not None
            ),
            default=0,
        )
        if (
            terminal_round != receipt.terminal_round
            or terminal_version != receipt.terminal_version
        ):
            raise RuntimeError("request reset rows differ from their terminal boundary")
        if not receipt.adaptation_state_acquired:
            if receipt_rounds or receipt_updates:
                raise RuntimeError("unacquired request carries archived rows")
            if receipt.evidence_archive_sha256 != archive_sha256:
                raise RuntimeError("unacquired request changed the evidence archive")
            continue
        if [int(row["round_index"]) for row in receipt_rounds] != list(
            range(1, len(receipt_rounds) + 1)
        ):
            raise RuntimeError("request archive round order is not native append order")
        archived_updates = [
            {
                key: value
                for key, value in row.items()
                if key not in _REQUEST_ARCHIVE_UPDATE_STRIP_KEYS
            }
            for row in receipt_updates
        ]
        archived_rounds = [
            {
                "request_epoch": receipt.request_epoch,
                "round_index": row["round_index"],
                "source_version": row["proposal_source_version"],
                "request_ids": [receipt.request_id],
                "prefix_len_before": [row["prefix_len_before"]],
                "verify_len": [row["verify_len"]],
                "accepted_drafts": [row["accepted_drafts"]],
                "committed_tokens": [row["committed_tokens"]],
            }
            for row in receipt_rounds
        ]
        archive_sha256 = canonical_sha256(
            {
                "schema_version": 1,
                "previous_archive_sha256": archive_sha256,
                "request_epoch": receipt.request_epoch,
                "request_id": receipt.request_id,
                "updates": archived_updates,
                "rounds": archived_rounds,
            }
        )
        if receipt.evidence_archive_sha256 != archive_sha256:
            raise RuntimeError("request evidence archive replay digest differs")
    if resets.final_archive_sha256 != archive_sha256:
        raise RuntimeError("final request evidence archive replay differs")


def _attestation_message(envelope: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "lightcone_terminal_attestation_challenge",
        "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
        "challenge_nonce_sha256": envelope["challenge_nonce_sha256"],
        "terminal_sha256": envelope["terminal_sha256"],
        "run_id": envelope["run_id"],
        "run_nonce_sha256": envelope["run_nonce_sha256"],
        "server_process_id": envelope["server_process_id"],
        "server_process_started_ns": envelope["server_process_started_ns"],
        "session_id": envelope["session_id"],
        "session_epoch": envelope["session_epoch"],
        "attempt_id": envelope["attempt_id"],
    }


def _validate_attestation(
    value: object,
    *,
    envelope: Mapping[str, object],
    trusted_attester_policy: TrustedAttesterPolicy,
) -> NativeTerminalAttestation:
    trusted_attester_policy.validate()
    policy_sha256 = trusted_attester_policy.sha256
    raw = _exact_object(value, _ATTESTATION_KEYS, "terminal attestation")
    if raw["schema_version"] != 1:
        raise ValueError("terminal attestation schema is unsupported")
    challenge = _sha256(
        raw["challenge_nonce_sha256"], "attestation.challenge_nonce_sha256"
    )
    message_digest = _sha256(raw["message_sha256"], "attestation.message_sha256")
    if challenge != envelope["challenge_nonce_sha256"]:
        raise ValueError("terminal attestation challenge nonce mismatch")
    message = _attestation_message(envelope)
    if message_digest != canonical_sha256(message):
        raise ValueError("terminal attestation message digest mismatch")
    status = raw["status"]
    if status == "UNAVAILABLE":
        if any(
            raw[field] is not None
            for field in ("attester_id", "trust_domain", "signature_hex")
        ):
            raise ValueError("UNAVAILABLE terminal attestation carries signer material")
        return NativeTerminalAttestation(
            status=status,
            challenge_nonce_sha256=challenge,
            message_sha256=message_digest,
            attester_id=None,
            trust_domain=None,
            signature_hex=None,
            key_id=None,
            public_key_sha256=None,
            trusted_attester_policy_sha256=policy_sha256,
            trusted=False,
        )
    if status != "SIGNED":
        raise ValueError("terminal attestation status is unsupported")
    attester_id = _safe_id(raw["attester_id"], "attestation.attester_id")
    trust_domain = raw["trust_domain"]
    signature_hex = raw["signature_hex"]
    if trust_domain not in {"hardware", "test"}:
        raise ValueError("terminal attestation trust domain is unsupported")
    if (
        not isinstance(signature_hex, str)
        or len(signature_hex) != 128
        or _LOWER_HEX.fullmatch(signature_hex) is None
    ):
        raise ValueError("terminal attestation signature is not canonical hex")
    key_id: str | None = None
    public_key_sha256: str | None = None
    trusted = False
    if trust_domain == "hardware" and trusted_attester_policy.allows_terminal_attester(
        str(attester_id)
    ):
        key_id, public_key_sha256 = trusted_attester_policy.verify_terminal_signature(
            attester_id=str(attester_id),
            trust_domain=str(trust_domain),
            message=canonical_json_bytes(message),
            signature_hex=signature_hex,
        )
        trusted = True
    return NativeTerminalAttestation(
        status=status,
        challenge_nonce_sha256=challenge,
        message_sha256=message_digest,
        attester_id=str(attester_id),
        trust_domain=str(trust_domain),
        signature_hex=signature_hex,
        key_id=key_id,
        public_key_sha256=public_key_sha256,
        trusted_attester_policy_sha256=policy_sha256,
        trusted=trusted,
    )


def _validate_terminal(
    value: object,
    *,
    begin: NativeTerminalBeginReceipt,
    reset: NativeTerminalResetReceipt,
    requests: tuple[TerminalRequestExpectation, ...],
    trusted_attester_policy: TrustedAttesterPolicy,
    legacy: bool = False,
) -> ValidatedNativeTerminalEvidence:
    raw = _exact_object(
        value,
        _LEGACY_TERMINAL_KEYS if legacy else _TERMINAL_KEYS,
        "terminal evidence",
    )
    binding = begin.binding
    expected_schema = 1 if legacy else 2
    expected_hook = (
        LEGACY_NATIVE_TERMINAL_EVIDENCE_HOOK
        if legacy
        else NATIVE_TERMINAL_EVIDENCE_HOOK
    )
    if raw["schema_version"] != expected_schema or raw["hook"] != expected_hook:
        raise ValueError("terminal evidence schema/hook mismatch")
    _validate_bound_identity(raw, binding, "terminal evidence", legacy=legacy)
    if not legacy:
        reset_scope, request_admission_policy = _wire_reset_identity(binding)
        terminal_protocol = _validate_wire_reset_protocol(
            method=binding.method,
            reset_scope=reset_scope,
            request_admission_policy=request_admission_policy,
            protocol_sha256=raw["request_source_point_reset_protocol_sha256"],
            field="terminal evidence",
        )
        if terminal_protocol != reset.request_source_point_reset_protocol_sha256:
            raise ValueError("terminal request reset protocol differs from reset")
    if (
        raw["server_process_id"] != begin.server_process_id
        or raw["server_process_started_ns"] != begin.server_process_started_ns
        or raw["reset_receipt_sha256"] != reset.reset_sha256
    ):
        raise ValueError("terminal evidence process/reset identity mismatch")
    if raw["expected_request_ids"] != list(binding.scored_request_ids):
        raise RuntimeError("terminal expected request order differs from the plan")
    if raw["completion_marker"] != "TERMINAL_COMPLETE":
        raise RuntimeError("terminal completion marker is missing")
    terminal_digest = _sha256(raw["terminal_sha256"], "terminal_sha256")
    unsigned = dict(raw)
    unsigned.pop("terminal_sha256")
    unsigned.pop("attestation")
    if canonical_sha256(unsigned) != terminal_digest:
        raise ValueError("terminal evidence content digest mismatch")
    request_round = _exact_object(
        raw["request_round_rows"],
        {"requests", "rounds"},
        "request_round_rows",
    )
    request_rows = _validate_request_rows(
        request_round["requests"], binding=binding, requests=requests
    )
    output_tokens = sum(int(row["output_tokens"]) for row in request_rows)
    performance = _validate_performance(
        raw["performance_counters"],
        method=binding.method,
        output_tokens=output_tokens,
    )
    request_resets = (
        None
        if legacy
        else _validate_request_source_point_resets(
            raw["request_source_point_resets"],
            binding=binding,
            requests=requests,
        )
    )
    rounds, _ = _validate_rounds_and_kv(
        request_round["rounds"],
        raw["historical_kv_source_versions"],
        binding=binding,
        requests=requests,
        performance=performance,
        request_resets=request_resets,
        legacy=legacy,
    )
    updates = _validate_updates(
        raw["update_rows"],
        binding=binding,
        rounds=rounds,
        performance=performance,
        request_resets=request_resets,
        legacy=legacy,
    )
    if request_resets is not None:
        _validate_request_reset_row_coverage(
            request_resets,
            requests=requests,
            rounds=rounds,
            updates=updates,
        )
    _validate_state(
        raw["final_state"],
        binding=binding,
        reset=reset,
        performance=performance,
        request_resets=request_resets,
        legacy=legacy,
    )
    attestation = _validate_attestation(
        raw["attestation"],
        envelope=raw,
        trusted_attester_policy=trusted_attester_policy,
    )
    return ValidatedNativeTerminalEvidence(
        binding=binding,
        begin_receipt=begin,
        reset_receipt=reset,
        requests=requests,
        attestation=attestation,
        terminal_sha256=terminal_digest,
        raw_json=_canonical_json_text(raw),
        _verification_tag=_VALIDATED_NATIVE_TERMINAL_SENTINEL,
    )


class NativeTerminalProvider:
    """Fail-closed begin/reset/finalize client for one pinned server process."""

    # Pin metadata is useful to the root integration, but is never treated as a
    # capability receipt.  ``begin`` always performs the admin GET below.
    native_evidence_hook = NATIVE_TERMINAL_EVIDENCE_HOOK
    patched_sglang_tree = PINNED_SGLANG_TREE
    supported_methods = SUPPORTED_METHODS

    def __init__(
        self,
        transport: AsyncNativeTerminalAdminTransport,
        *,
        trusted_attester_policy: TrustedAttesterPolicy = NO_TRUSTED_ATTESTERS,
    ) -> None:
        if type(trusted_attester_policy) is not TrustedAttesterPolicy:
            raise TypeError("native terminal trust requires an exact release policy")
        trusted_attester_policy.validate()
        self._transport = transport
        self._trusted_attester_policy = trusted_attester_policy
        self._trusted_attester_policy_sha256 = trusted_attester_policy.sha256
        self._phase = "IDLE"
        self._binding: NativeTerminalRunBinding | None = None
        self._begin: NativeTerminalBeginReceipt | None = None
        self._reset: NativeTerminalResetReceipt | None = None
        self._process: tuple[int, int] | None = None
        self._reset_generation = 0
        self._next_session_epoch = 1
        self._last_finalized_run_id: str | None = None
        self._session_id: str | None = None
        self._seen_runs: set[str] = set()
        self._seen_attempts: set[str] = set()

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def trusted_attester_policy(self) -> TrustedAttesterPolicy:
        if self._trusted_attester_policy.sha256 != self._trusted_attester_policy_sha256:
            raise RuntimeError("native terminal release policy identity changed")
        return self._trusted_attester_policy

    @property
    def trusted_attester_policy_sha256(self) -> str:
        _ = self.trusted_attester_policy
        return self._trusted_attester_policy_sha256

    async def capability(self, *, expected_method: str) -> NativeTerminalCapability:
        if self._phase == "FAILED":
            raise RuntimeError("native terminal client is fail-closed")
        value = await self._transport.get_json(CAPABILITY_PATH)
        return _validate_capability(value, expected_method=expected_method)

    async def begin(
        self, binding: NativeTerminalRunBinding
    ) -> NativeTerminalBeginReceipt:
        try:
            if self._phase not in {"IDLE", "FINALIZED"}:
                raise RuntimeError("native terminal begin is illegal in this phase")
            binding.validate()
            if (
                binding.session_epoch != self._next_session_epoch
                or binding.previous_run_id != self._last_finalized_run_id
            ):
                raise ValueError("native terminal session lineage is stale or skipped")
            if self._session_id is not None and binding.session_id != self._session_id:
                raise ValueError("native terminal session identity changed")
            if (
                binding.run_id in self._seen_runs
                or binding.attempt_id in self._seen_attempts
            ):
                raise ValueError("native terminal run/attempt identity was reused")
            capability = await self.capability(expected_method=binding.method)
            reset_scope, request_admission_policy = _wire_reset_identity(binding)
            if (
                capability.reset_scope != reset_scope
                or capability.request_admission_policy != request_admission_policy
                or capability.runtime_trust_mode != binding.runtime_trust_mode
                or capability.formal_measurement is not binding.formal_measurement
            ):
                raise ValueError(
                    "native capability differs from the run reset/runtime identity"
                )
            response = await self._transport.post_json(
                TERMINAL_EVIDENCE_PATH,
                {"action": "begin", "payload": binding.begin_payload()},
            )
            receipt = _validate_begin_receipt(
                response,
                binding=binding,
                prior_reset_generation=self._reset_generation,
                prior_process=self._process,
                expected_protocol_sha256=(
                    capability.request_source_point_reset_protocol_sha256
                ),
            )
            self._binding = binding
            self._begin = receipt
            self._reset = None
            self._process = (
                receipt.server_process_id,
                receipt.server_process_started_ns,
            )
            self._reset_generation = receipt.reset_generation
            self._session_id = binding.session_id
            self._seen_runs.add(binding.run_id)
            self._seen_attempts.add(binding.attempt_id)
            self._phase = "WARMUP"
            return receipt
        except BaseException:
            self._phase = "FAILED"
            raise

    async def reset(
        self,
        *,
        warmup_requests: Sequence[TerminalRequestExpectation] = (),
    ) -> NativeTerminalResetReceipt:
        try:
            if self._phase != "WARMUP" or self._binding is None or self._begin is None:
                raise RuntimeError("native terminal reset is illegal in this phase")
            warmup = _validate_request_expectations(
                warmup_requests,
                expected_ids=self._binding.warmup_request_ids,
                warmup=True,
            )
            response = await self._transport.post_json(
                TERMINAL_EVIDENCE_PATH,
                {
                    "action": "reset",
                    "payload": {
                        "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
                        "run_id": self._binding.run_id,
                        "begin_sha256": self._begin.begin_sha256,
                    },
                },
            )
            receipt = _validate_reset_receipt(
                response,
                begin=self._begin,
                warmup_requests=warmup,
            )
            self._reset = receipt
            self._reset_generation = receipt.reset_generation
            self._phase = "SCORED"
            return receipt
        except BaseException:
            self._phase = "FAILED"
            raise

    async def finalize(
        self,
        *,
        requests: Sequence[TerminalRequestExpectation],
    ) -> ValidatedNativeTerminalEvidence:
        try:
            if (
                self._phase != "SCORED"
                or self._binding is None
                or self._begin is None
                or self._reset is None
            ):
                raise RuntimeError("native terminal finalize is illegal in this phase")
            expected = _validate_request_expectations(
                requests,
                expected_ids=self._binding.scored_request_ids,
                warmup=False,
            )
            client_rows = [
                _client_terminal_row(self._binding, request)
                for request in expected
                if not request.submitted_to_server
            ]
            response = await self._transport.post_json(
                TERMINAL_EVIDENCE_PATH,
                {
                    "action": "finalize",
                    "payload": {
                        "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
                        "run_id": self._binding.run_id,
                        "reset_sha256": self._reset.reset_sha256,
                        "client_terminal_rows": client_rows,
                    },
                },
            )
            result = _validate_terminal(
                response,
                begin=self._begin,
                reset=self._reset,
                requests=expected,
                trusted_attester_policy=self.trusted_attester_policy,
            )
            self._last_finalized_run_id = self._binding.run_id
            self._next_session_epoch += 1
            self._phase = "FINALIZED"
            return result
        except BaseException:
            self._phase = "FAILED"
            raise


def _validate_unsigned_native_result_pointer(value: str) -> dict[str, object]:
    if type(value) is not str:
        raise TypeError("unsigned native ITL pointer must be canonical JSON text")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("unsigned native ITL pointer is not valid JSON") from error
    expected_fields = {
        "schema_version",
        "kind",
        "hook",
        "semantics",
        "release_status",
        "request_id",
        "request_started_ns",
        "request_terminal_ns",
        "terminal_status",
        "terminal_reason",
        "events",
        "result_pointer_sha256",
    }
    row = _exact_object(parsed, expected_fields, "unsigned native ITL pointer")
    if value != _canonical_json_text(row):
        raise ValueError("unsigned native ITL pointer is not canonical JSON")
    digest = _sha256(row["result_pointer_sha256"], "unsigned native ITL result pointer")
    unsigned = dict(row)
    unsigned.pop("result_pointer_sha256")
    if canonical_sha256(unsigned) != digest:
        raise ValueError("unsigned native ITL pointer content digest mismatch")
    if (
        row["schema_version"] != 1
        or row["kind"] != "sglang_native_itl_result_pointer"
        or row["hook"] != "sglang.schema_v3.native_per_token_timestamp.v2"
        or row["semantics"] != "scheduler_committed_token_at_result_processor_v1"
        or row["release_status"] != "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF"
        or row["terminal_status"] != "completed"
    ):
        raise ValueError("unsigned native ITL pointer release identity differs")
    _safe_id(row["request_id"], "unsigned native ITL request")
    _safe_id(row["terminal_reason"], "unsigned native ITL terminal reason")
    started_ns = _integer(
        row["request_started_ns"], "unsigned native ITL request start"
    )
    terminal_ns = _integer(
        row["request_terminal_ns"], "unsigned native ITL request terminal"
    )
    if terminal_ns < started_ns:
        raise ValueError("unsigned native ITL terminal precedes request start")
    raw_events = row["events"]
    if type(raw_events) is not list or not raw_events:
        raise ValueError("unsigned native ITL pointer has no token coverage")
    previous_ns = started_ns
    for index, event_value in enumerate(raw_events):
        event = _exact_object(
            event_value,
            {"token_index", "token_id", "observed_ns"},
            "unsigned native ITL event",
        )
        if _integer(event["token_index"], "unsigned native ITL token index") != index:
            raise ValueError("unsigned native ITL token indices are not contiguous")
        _integer(event["token_id"], "unsigned native ITL token ID")
        observed_ns = _integer(event["observed_ns"], "unsigned native ITL observation")
        if observed_ns <= previous_ns or observed_ns > terminal_ns:
            raise ValueError("unsigned native ITL observations are not ordered")
        previous_ns = observed_ns
    return row


def validate_unsigned_native_itl_pointer_bundle(
    pointer_artifact: str | Path | CanonicalJsonProofBinding,
    *,
    expected_binding: NativeTerminalRunBinding,
    expected_terminal_artifact: CanonicalJsonProofBinding,
    expected_scored_request_inputs_sha256: str,
    expected_terminal_output_tokens: Mapping[str, tuple[int, ...]],
) -> ValidatedUnsignedNativeItlPointerBundle:
    """Deep-reopen the collector's ITL bundle under exact serving lineage.

    The expected input digest must come from the sealed request materialization;
    output token tuples must come from the independently validated native
    terminal.  The returned type is verifier-constructed but remains unsigned
    input to the later local external-control proof.
    """

    if type(expected_binding) is not NativeTerminalRunBinding:
        raise TypeError("unsigned native ITL bundle requires exact run binding")
    expected_binding.validate()
    if type(expected_terminal_artifact) is not CanonicalJsonProofBinding:
        raise TypeError("unsigned native ITL bundle requires terminal binding")
    expected_terminal_artifact.__post_init__()
    _sha256(
        expected_scored_request_inputs_sha256,
        "unsigned native ITL scored request inputs",
    )
    if type(expected_terminal_output_tokens) is not dict:
        raise TypeError("unsigned native ITL terminal outputs must be an exact map")
    expected_pointer_ids = tuple(
        request_id
        for request_id in expected_binding.scored_request_ids
        if request_id in expected_terminal_output_tokens
    )
    if tuple(expected_terminal_output_tokens) != expected_pointer_ids:
        raise ValueError("unsigned native ITL terminal output coverage differs")
    for request_id, token_ids in expected_terminal_output_tokens.items():
        _safe_id(request_id, "unsigned native ITL expected request")
        _token_ids(token_ids, "unsigned native ITL expected output tokens")
    if type(pointer_artifact) is CanonicalJsonProofBinding:
        artifact = pointer_artifact
        artifact.__post_init__()
    elif isinstance(pointer_artifact, (str, Path)):
        artifact = CanonicalJsonProofBinding.bind(pointer_artifact)
    else:
        raise TypeError("unsigned native ITL bundle requires path-bound artifact")
    raw = _exact_object(
        reopen_scalable_unsigned_native_itl_bundle(artifact.reopen()),
        {
            "schema_version",
            "kind",
            "run_binding_sha256",
            "terminal_artifact_raw_sha256",
            "terminal_artifact_semantic_sha256",
            "scored_request_inputs_sha256",
            "native_result_pointers",
        },
        "unsigned native ITL pointer bundle",
    )
    expected_run_binding_sha256 = canonical_sha256(expected_binding.begin_payload())
    if (
        raw["schema_version"] != 1
        or raw["kind"] != "unsigned_native_itl_result_pointer_bundle"
        or raw["run_binding_sha256"] != expected_run_binding_sha256
        or raw["terminal_artifact_raw_sha256"] != expected_terminal_artifact.raw_sha256
        or raw["terminal_artifact_semantic_sha256"]
        != expected_terminal_artifact.semantic_sha256
        or raw["scored_request_inputs_sha256"] != expected_scored_request_inputs_sha256
    ):
        raise ValueError("unsigned native ITL pointer bundle lineage differs")
    pointer_values = raw["native_result_pointers"]
    if type(pointer_values) is not list:
        raise ValueError("unsigned native ITL pointer bundle pointers are malformed")
    pointers: list[ValidatedUnsignedNativeItlPointer] = []
    for pointer_value in pointer_values:
        if type(pointer_value) is not dict:
            raise TypeError("unsigned native ITL pointer must be an object")
        row = _validate_unsigned_native_result_pointer(
            _canonical_json_text(pointer_value)
        )
        request_id = str(row["request_id"])
        raw_events = row["events"]
        assert isinstance(raw_events, list)
        events = tuple(
            UnsignedNativeItlTokenEvent(
                token_index=int(event["token_index"]),
                token_id=int(event["token_id"]),
                observed_ns=int(event["observed_ns"]),
            )
            for event in raw_events
        )
        if tuple(event.token_id for event in events) != (
            expected_terminal_output_tokens.get(request_id)
        ):
            raise ValueError("unsigned native ITL pointer differs from terminal tokens")
        pointers.append(
            ValidatedUnsignedNativeItlPointer(
                request_id=request_id,
                request_started_ns=int(row["request_started_ns"]),
                request_terminal_ns=int(row["request_terminal_ns"]),
                terminal_status=str(row["terminal_status"]),
                terminal_reason=str(row["terminal_reason"]),
                events=events,
                result_pointer_sha256=str(row["result_pointer_sha256"]),
                _verification_tag=_VALIDATED_UNSIGNED_ITL_BUNDLE_SENTINEL,
            )
        )
    if tuple(pointer.request_id for pointer in pointers) != expected_pointer_ids:
        raise ValueError("unsigned native ITL pointer order/coverage differs")
    return ValidatedUnsignedNativeItlPointerBundle(
        artifact_raw_sha256=artifact.raw_sha256,
        artifact_semantic_sha256=artifact.semantic_sha256,
        run_binding_sha256=expected_run_binding_sha256,
        terminal_artifact_raw_sha256=expected_terminal_artifact.raw_sha256,
        terminal_artifact_semantic_sha256=expected_terminal_artifact.semantic_sha256,
        scored_request_inputs_sha256=expected_scored_request_inputs_sha256,
        pointers=tuple(pointers),
        _verification_tag=_VALIDATED_UNSIGNED_ITL_BUNDLE_SENTINEL,
    )


def _validate_bound_serving_phase(
    requests: Sequence[BoundServingRequest],
    *,
    expected_ids: tuple[str, ...],
) -> tuple[BoundServingRequest, ...]:
    # Local import avoids the experiments -> orchestration package-init cycle.
    from lightcone_spec.experiments.serving import BoundServingRequest

    values = tuple(requests)
    if tuple(request.request_id for request in values) != expected_ids:
        raise ValueError("unsigned serving inputs differ from terminal binding")
    for request in values:
        if type(request) is not BoundServingRequest:
            raise TypeError("unsigned serving inputs must be exact bound requests")
        request.validate()
    return values


async def collect_unsigned_native_terminal_artifact(
    transport: AsyncNativeTerminalAdminTransport,
    *,
    binding: NativeTerminalRunBinding,
    warmup_requests: Sequence[BoundServingRequest],
    scored_requests: Sequence[BoundServingRequest],
    execute_requests: Callable[
        [str, tuple[BoundServingRequest, ...]],
        Awaitable[UnsignedNativeServingPhaseResult],
    ],
    output_path: str,
    native_itl_pointer_output_path: str,
    expected_server_process_id: int | None = None,
) -> UnsignedNativeTerminalCollection:
    """Run one first-party lifecycle and publish an unsigned raw terminal.

    This is the remote half of the formal evidence DAG.  It intentionally uses
    ``NO_TRUSTED_ATTESTERS`` and therefore cannot mint an execution/completion
    authority.  After stable pull, the local verifier must consume the result
    through the external-control durable proof APIs in this module.
    """

    if not callable(execute_requests):
        raise TypeError("unsigned native collection requires a request executor")
    binding.validate()
    warmup = _validate_bound_serving_phase(
        warmup_requests,
        expected_ids=binding.warmup_request_ids,
    )
    scored = _validate_bound_serving_phase(
        scored_requests,
        expected_ids=binding.scored_request_ids,
    )
    provider = NativeTerminalProvider(
        transport,
        trusted_attester_policy=NO_TRUSTED_ATTESTERS,
    )
    begin_started_ns = time.monotonic_ns()
    begin = await provider.begin(binding)
    begin_finished_ns = time.monotonic_ns()
    if expected_server_process_id is not None and (
        type(expected_server_process_id) is not int
        or expected_server_process_id < 1
        or begin.server_process_id != expected_server_process_id
    ):
        raise RuntimeError(
            "unsigned native lifecycle reached an unexpected server process"
        )
    warmup_started_ns = time.monotonic_ns()
    warmup_result = await execute_requests("warmup", warmup)
    warmup_finished_ns = time.monotonic_ns()
    if type(warmup_result) is not UnsignedNativeServingPhaseResult:
        raise TypeError("unsigned warmup executor returned an untyped result")
    warmup_result.validate(expected_phase="warmup", bound_requests=warmup)
    reset_started_ns = time.monotonic_ns()
    await provider.reset(warmup_requests=warmup_result.requests)
    reset_finished_ns = time.monotonic_ns()
    scored_started_ns = time.monotonic_ns()
    scored_result = await execute_requests("scored", scored)
    scored_finished_ns = time.monotonic_ns()
    if type(scored_result) is not UnsignedNativeServingPhaseResult:
        raise TypeError("unsigned scored executor returned an untyped result")
    scored_pointers = scored_result.validate(
        expected_phase="scored", bound_requests=scored
    )
    finalize_started_ns = time.monotonic_ns()
    evidence = await provider.finalize(requests=scored_result.requests)
    finalize_finished_ns = time.monotonic_ns()
    if evidence.authority_kind != "untrusted_raw_terminal":
        raise RuntimeError("unsigned native collection unexpectedly gained authority")
    terminal_binding = publish_scalable_native_terminal_artifact(
        output_path=output_path,
        legacy_artifact=evidence.to_artifact(warmup_requests=warmup_result.requests),
    )
    terminal_published_ns = time.monotonic_ns()
    itl_binding = publish_scalable_unsigned_native_itl_bundle(
        output_path=native_itl_pointer_output_path,
        legacy_bundle={
            "schema_version": 1,
            "kind": "unsigned_native_itl_result_pointer_bundle",
            "run_binding_sha256": canonical_sha256(binding.begin_payload()),
            "terminal_artifact_raw_sha256": terminal_binding.raw_sha256,
            "terminal_artifact_semantic_sha256": terminal_binding.semantic_sha256,
            "scored_request_inputs_sha256": canonical_sha256(
                [request.sha256 for request in scored]
            ),
            "native_result_pointers": [dict(value) for value in scored_pointers],
        },
    )
    itl_pointer_published_ns = time.monotonic_ns()
    return UnsignedNativeTerminalCollection(
        terminal_artifact=terminal_binding,
        native_itl_pointer_artifact=itl_binding,
        lifecycle_events=UnsignedNativeLifecycleEvents(
            begin_started_ns=begin_started_ns,
            begin_finished_ns=begin_finished_ns,
            warmup_started_ns=warmup_started_ns,
            warmup_finished_ns=warmup_finished_ns,
            reset_started_ns=reset_started_ns,
            reset_finished_ns=reset_finished_ns,
            scored_started_ns=scored_started_ns,
            scored_finished_ns=scored_finished_ns,
            finalize_started_ns=finalize_started_ns,
            finalize_finished_ns=finalize_finished_ns,
            terminal_published_ns=terminal_published_ns,
            itl_pointer_published_ns=itl_pointer_published_ns,
        ),
    )
