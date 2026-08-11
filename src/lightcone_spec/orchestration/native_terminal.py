"""Strict host client for the pinned SGLang terminal-evidence hook.

The server owns the evidence.  This module only constructs nonce-bound control
messages and validates the exact JSON returned by the admin endpoints.  In
particular, object attributes, command-line flags, and caller-authored
capability dictionaries never substitute for a capability response obtained
through :class:`AsyncNativeTerminalAdminTransport`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

NATIVE_TERMINAL_EVIDENCE_HOOK = (
    "sglang.schema_v3.content_bound_terminal_speculative_evidence.v1"
)
PINNED_SGLANG_UPSTREAM_COMMIT = "3312645a307453893a00778592f105581e3d1c3d"
PINNED_SGLANG_PATCH_SHA256 = (
    "c29324de3f5893d2d140829d93a1c069002093216c39144f0d6c19d23710ff08"
)
PINNED_SGLANG_TREE = "2810ac94ed225aa78b4256ded56e78890a4a590f"

CAPABILITY_PATH = "/v1/lightcone-spec/terminal-evidence/capability"
TERMINAL_EVIDENCE_PATH = "/v1/lightcone-spec/terminal-evidence"

NATIVE_TERMINAL_EVIDENCE_FIELDS = (
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
SUPPORTED_METHODS = frozenset({"target_only", "static", "tts", "l0"})
_ORDERED_SUPPORTED_METHODS = ("l0", "static", "target_only", "tts")

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
}
_IDENTITY_KEYS = {
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
_BEGIN_RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "hook",
    *_IDENTITY_KEYS,
    "server_process_id",
    "server_process_started_ns",
    "reset_generation",
    "prior_state_sha256",
    "reset_state_sha256",
    "warmup_request_ids_sha256",
    "scored_request_ids_sha256",
    "begin_sha256",
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
    "prior_trace_run_id",
    "next_trace_run_id",
    "warmup_request_rows_sha256",
    "warmup_performance_sha256",
    "discarded_native_sha256",
    "warmup_state_sha256",
    "reset_state_sha256",
    "expected_scored_request_ids_sha256",
    "completion_event_generation",
    "reset_sha256",
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
_ROUND_KEYS = {
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
_UPDATE_KEYS = {
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
    "update_sha256",
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
_STATE_KEYS = {
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
_ATTESTATION_KEYS = {
    "schema_version",
    "status",
    "challenge_nonce_sha256",
    "message_sha256",
    "attester_id",
    "trust_domain",
    "signature_hex",
}
_TERMINAL_KEYS = set(NATIVE_TERMINAL_EVIDENCE_FIELDS)


class AsyncNativeTerminalAdminTransport(Protocol):
    """Authenticated JSON transport for the two native admin endpoints."""

    async def get_json(self, path: str, /) -> object: ...

    async def post_json(
        self,
        path: str,
        body: Mapping[str, object],
        /,
    ) -> object: ...


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


def _identity_values(binding: NativeTerminalRunBinding) -> dict[str, object]:
    return {
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


def _validate_bound_identity(
    value: Mapping[str, object], binding: NativeTerminalRunBinding, field: str
) -> None:
    expected = _identity_values(binding)
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
            "schema_version": 1,
            "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
            **_identity_values(self),
            "warmup_request_ids": list(self.warmup_request_ids),
            "scored_request_ids": list(self.scored_request_ids),
        }


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
            if self.terminal_status not in {"rejected", "cancelled", "timed_out"}:
                raise ValueError("non-submitted request has an invalid client status")
            if self.output_token_ids is not None:
                raise ValueError("non-submitted terminal rows cannot carry output IDs")


@dataclass(frozen=True)
class NativeTerminalCapability:
    active_method: str
    enabled: bool
    method_evidence_supported: bool
    topology_supported: bool
    trusted_attester_configured: bool
    required_fields: tuple[str, ...]
    supported_methods: tuple[str, ...]
    raw_json: str


@dataclass(frozen=True)
class NativeTerminalBeginReceipt:
    binding: NativeTerminalRunBinding
    server_process_id: int
    server_process_started_ns: int
    reset_generation: int
    begin_sha256: str
    raw_json: str


@dataclass(frozen=True)
class NativeTerminalResetReceipt:
    binding: NativeTerminalRunBinding
    server_process_id: int
    server_process_started_ns: int
    reset_generation: int
    completion_event_generation: int
    reset_sha256: str
    raw_json: str


@dataclass(frozen=True)
class NativeTerminalAttestation:
    status: str
    challenge_nonce_sha256: str
    message_sha256: str
    attester_id: str | None
    trust_domain: str | None
    signature_hex: str | None
    trusted: bool


@dataclass(frozen=True)
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

    @property
    def trusted_attestation(self) -> bool:
        return self.attestation.trusted

    def to_dict(self) -> dict[str, object]:
        value = json.loads(self.raw_json)
        if not isinstance(value, dict):  # pragma: no cover - construction invariant
            raise TypeError("validated terminal JSON stopped being an object")
        return value

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
                    row["historical_kv_source_versions"]
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
                cohort_epoch=int(row["cohort_epoch"]),
                exactness_violation=row["status"] == "reconstruction_mismatch",
                stale_candidate=row["status"] == "version_conflict",
                nonfinite_candidate=row["status"] == "nonfinite_update",
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


def _validate_capability(
    value: object, *, expected_method: str
) -> NativeTerminalCapability:
    raw = _exact_object(value, _CAPABILITY_KEYS, "terminal capability")
    if raw["schema_version"] != 1 or raw["hook"] != NATIVE_TERMINAL_EVIDENCE_HOOK:
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
        raw_json=_canonical_json_text(raw),
    )


def _validate_begin_receipt(
    value: object,
    *,
    binding: NativeTerminalRunBinding,
    prior_reset_generation: int,
    prior_process: tuple[int, int] | None,
) -> NativeTerminalBeginReceipt:
    raw = _exact_object(value, _BEGIN_RECEIPT_KEYS, "terminal begin receipt")
    if (
        raw["schema_version"] != 1
        or raw["kind"] != "lightcone_terminal_begin_receipt"
        or raw["hook"] != NATIVE_TERMINAL_EVIDENCE_HOOK
    ):
        raise ValueError("terminal begin receipt schema/hook mismatch")
    _validate_bound_identity(raw, binding, "terminal begin receipt")
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
) -> NativeTerminalResetReceipt:
    raw = _exact_object(value, _RESET_RECEIPT_KEYS, "terminal reset receipt")
    binding = begin.binding
    if (
        raw["schema_version"] != 1
        or raw["kind"] != "lightcone_terminal_reset_receipt"
        or raw["hook"] != NATIVE_TERMINAL_EVIDENCE_HOOK
    ):
        raise ValueError("terminal reset receipt schema/hook mismatch")
    _validate_bound_identity(raw, binding, "terminal reset receipt")
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
        reset_sha256=str(digest),
        raw_json=_canonical_json_text(raw),
    )


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
    if not seen:  # pragma: no cover - scored IDs are non-empty
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
    if method in {"tts", "l0"}:
        for field in (
            "optimizer_bytes",
            "trainable_parameters",
            "updates_launched",
            "updates_published",
        ):
            _integer(raw[field], f"performance.{field}")
        if raw["adaptation_memory_ledger"] is None:
            raise RuntimeError("adapted performance lacks its memory ledger")
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


def _validate_state(
    value: object,
    *,
    method: str,
    reset: NativeTerminalResetReceipt,
    performance: Mapping[str, object],
) -> dict[str, object]:
    raw = _exact_object(value, _STATE_KEYS, "terminal final state")
    if raw["schema_version"] != 1:
        raise ValueError("terminal final-state schema is unsupported")
    for field in (
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
        _integer(raw[field], f"final_state.{field}")
    for field in ("kv_state_sha256", "rng_state_sha256", "adapter_state_sha256"):
        _sha256(raw[field], f"final_state.{field}")
    for field in (
        "scheduler_idle",
        "adapter_reset_verified",
        "completion_event_complete",
    ):
        _boolean(raw[field], f"final_state.{field}")
    if (
        not raw["scheduler_idle"]
        or raw["active_requests"] != 0
        or raw["queued_requests"] != 0
        or not raw["completion_event_complete"]
    ):
        raise RuntimeError("terminal final state is not drained and synchronized")
    if raw["allocator_peak_hbm_bytes"] < raw["allocator_current_hbm_bytes"]:
        raise ValueError("terminal allocator peak is below current allocation")
    if raw["kv_available_tokens"] > raw["kv_token_capacity"]:
        raise ValueError("terminal KV availability exceeds capacity")
    if raw["completion_event_generation"] <= reset.completion_event_generation:
        raise ValueError("terminal completion event does not follow reset")
    if method in {"target_only", "static"} and (
        raw["adapter_active_version"] != 0
        or raw["adapter_epoch"] != 0
        or raw["optimizer_generation"] != 0
        or not raw["adapter_reset_verified"]
    ):
        raise RuntimeError("allocation-free final state reports adaptation mutation")
    if method in {"tts", "l0"} and (
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
    performance: Mapping[str, object],
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
    expectation = {request.request_id: request for request in requests}
    server_ids = {
        request.request_id for request in requests if request.submitted_to_server
    }
    rows: list[dict[str, object]] = []
    identities: set[tuple[str, int]] = set()
    by_request: dict[str, list[dict[str, object]]] = {}
    for value in rounds_value:
        raw = _exact_object(value, _ROUND_KEYS, "terminal round row")
        request_id = _safe_id(raw["request_id"], "round.request_id")
        if request_id not in server_ids:
            raise RuntimeError("terminal round references a non-submitted request")
        round_index = _integer(raw["round_index"], "round.round_index", minimum=1)
        identity = (str(request_id), round_index)
        if identity in identities:
            raise RuntimeError("terminal round identity is duplicated")
        identities.add(identity)
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
        if not isinstance(raw["historical_kv_source_versions"], list):
            raise TypeError("terminal round historical KV value is malformed")
        if raw["historical_kv_source_versions"] != kv_value.get(request_id):
            raise RuntimeError("terminal round historical KV differs from terminal map")
        digest = _sha256(raw["round_sha256"], "round_sha256")
        unsigned = dict(raw)
        unsigned.pop("round_sha256")
        if canonical_sha256(unsigned) != digest:
            raise ValueError("terminal round content digest mismatch")
        rows.append(raw)
        by_request.setdefault(str(request_id), []).append(raw)
    completed = {
        request.request_id
        for request in requests
        if request.submitted_to_server and request.terminal_status == "completed"
    }
    if not completed.issubset(by_request):
        raise RuntimeError("completed adapted request lacks round evidence")
    if set(kv_value) != set(by_request):
        raise RuntimeError("historical KV request coverage differs from rounds")
    for request_id, request_rows in by_request.items():
        request_rows.sort(key=lambda row: int(row["round_index"]))
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
    for performance_field, round_field in (
        ("target_calls", "target_calls"),
        ("accepted_drafts", "accepted_drafts"),
        ("committed_tokens", "committed_tokens"),
        ("verified_drafts", "verify_len"),
    ):
        if performance[performance_field] != sum(int(row[round_field]) for row in rows):
            raise RuntimeError("terminal round aggregates disagree with performance")
    return rows, dict(kv_value)


def _validate_updates(
    value: object,
    *,
    binding: NativeTerminalRunBinding,
    rounds: list[dict[str, object]],
    performance: Mapping[str, object],
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("terminal update rows must be a JSON list")
    if binding.method in {"target_only", "static"}:
        if value:
            raise RuntimeError("allocation-free evidence contains update rows")
        return []
    if not value:
        raise RuntimeError("adapted evidence requires update rows")
    round_index = {
        (str(row["request_id"]), int(row["round_index"])): row for row in rounds
    }
    rows: list[dict[str, object]] = []
    allowed_status = {
        "published",
        "version_conflict",
        "request_aborted",
        "no_supervision",
        "nonfinite_update",
        "reconstruction_mismatch",
    }
    for expected_index, value_row in enumerate(value):
        raw = _exact_object(value_row, _UPDATE_KEYS, "terminal update row")
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
    if performance["updates_launched"] != len(rows):
        raise RuntimeError("terminal launched-update aggregate differs from rows")
    published_count = sum(row["status"] == "published" for row in rows)
    if performance["updates_published"] != published_count:
        raise RuntimeError("terminal published-update aggregate differs from rows")
    lower_bounds = {
        "version_mismatches": sum(row["status"] == "version_conflict" for row in rows),
        "nonfinite_updates": sum(row["status"] == "nonfinite_update" for row in rows),
        "exactness_violations": sum(
            row["status"] == "reconstruction_mismatch" for row in rows
        ),
        "fallbacks": sum(row["status"] == "reconstruction_mismatch" for row in rows),
    }
    if any(performance[field] < minimum for field, minimum in lower_bounds.items()):
        raise RuntimeError("terminal safety aggregate undercounts update rows")
    return rows


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
    verifiers: Mapping[str, SignatureVerifier],
) -> NativeTerminalAttestation:
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
        or _LOWER_HEX.fullmatch(signature_hex) is None
    ):
        raise ValueError("terminal attestation signature is not canonical hex")
    signature = bytes.fromhex(signature_hex)
    release_identity = trust_domain == "hardware" and not str(
        attester_id
    ).lower().startswith(("test", "fixture", "cpu"))
    verifier = verifiers.get(str(attester_id)) if release_identity else None
    trusted = False
    if verifier is not None:
        try:
            trusted = verifier(canonical_json_bytes(message), signature) is True
        except Exception as error:
            raise ValueError("terminal attestation verifier failed") from error
        if not trusted:
            raise ValueError("terminal attestation signature verification failed")
    return NativeTerminalAttestation(
        status=status,
        challenge_nonce_sha256=challenge,
        message_sha256=message_digest,
        attester_id=str(attester_id),
        trust_domain=str(trust_domain),
        signature_hex=signature_hex,
        trusted=trusted,
    )


def _validate_terminal(
    value: object,
    *,
    begin: NativeTerminalBeginReceipt,
    reset: NativeTerminalResetReceipt,
    requests: tuple[TerminalRequestExpectation, ...],
    verifiers: Mapping[str, SignatureVerifier],
) -> ValidatedNativeTerminalEvidence:
    raw = _exact_object(value, _TERMINAL_KEYS, "terminal evidence")
    binding = begin.binding
    if raw["schema_version"] != 1 or raw["hook"] != NATIVE_TERMINAL_EVIDENCE_HOOK:
        raise ValueError("terminal evidence schema/hook mismatch")
    _validate_bound_identity(raw, binding, "terminal evidence")
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
    rounds, _ = _validate_rounds_and_kv(
        request_round["rounds"],
        raw["historical_kv_source_versions"],
        binding=binding,
        requests=requests,
        performance=performance,
    )
    _validate_updates(
        raw["update_rows"],
        binding=binding,
        rounds=rounds,
        performance=performance,
    )
    _validate_state(
        raw["final_state"],
        method=binding.method,
        reset=reset,
        performance=performance,
    )
    attestation = _validate_attestation(
        raw["attestation"], envelope=raw, verifiers=verifiers
    )
    return ValidatedNativeTerminalEvidence(
        binding=binding,
        begin_receipt=begin,
        reset_receipt=reset,
        requests=requests,
        attestation=attestation,
        terminal_sha256=terminal_digest,
        raw_json=_canonical_json_text(raw),
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
        trusted_hardware_verifiers: Mapping[str, SignatureVerifier] | None = None,
    ) -> None:
        self._transport = transport
        self._verifiers = dict(trusted_hardware_verifiers or {})
        for attester_id, verifier in self._verifiers.items():
            _safe_id(attester_id, "trusted attester ID")
            if not callable(verifier):
                raise TypeError("trusted hardware verifier must be callable")
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
            await self.capability(expected_method=binding.method)
            response = await self._transport.post_json(
                TERMINAL_EVIDENCE_PATH,
                {"action": "begin", "payload": binding.begin_payload()},
            )
            receipt = _validate_begin_receipt(
                response,
                binding=binding,
                prior_reset_generation=self._reset_generation,
                prior_process=self._process,
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
                verifiers=self._verifiers,
            )
            self._last_finalized_run_id = self._binding.run_id
            self._next_session_epoch += 1
            self._phase = "FINALIZED"
            return result
        except BaseException:
            self._phase = "FAILED"
            raise
