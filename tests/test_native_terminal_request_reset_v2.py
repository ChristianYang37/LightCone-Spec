from __future__ import annotations

import copy
import hashlib

import pytest

from lightcone_spec.orchestration import native_terminal
from lightcone_spec.runtime.attestation import NO_TRUSTED_ATTESTERS

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
ZERO_SHA256 = "0" * 64


def _binding(
    *,
    warmup_ids: tuple[str, ...] = ("warm-0",),
    scored_ids: tuple[str, ...] = ("score-0",),
    rank_config_sha256: str = SHA_C,
    runtime_trust_mode: str | None = None,
    formal_measurement: bool | None = None,
) -> native_terminal.NativeTerminalRunBinding:
    return native_terminal.NativeTerminalRunBinding(
        run_id="run-reset-v2",
        run_nonce_sha256=SHA_A,
        execution_plan_sha256=SHA_B,
        rank_config_sha256=rank_config_sha256,
        attempt_id="attempt-reset-v2",
        session_id="session-reset-v2",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=SHA_D,
        method="tts",
        reset_scope="request",
        request_admission_policy="serialized_native_scheduler_v1",
        runtime_trust_mode=runtime_trust_mode,
        formal_measurement=formal_measurement,
        warmup_request_ids=warmup_ids,
        scored_request_ids=scored_ids,
    )


def _request(request_id: str) -> native_terminal.TerminalRequestExpectation:
    return native_terminal.TerminalRequestExpectation(
        request_id=request_id,
        input_token_ids=(11, 12),
        output_token_ids=(13,),
        terminal_status="completed",
        terminal_reason="FINISH_LENGTH",
        submitted_to_server=True,
    )


def _identity(
    binding: native_terminal.NativeTerminalRunBinding,
) -> dict[str, object]:
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
        "reset_scope": "request",
        "request_admission_policy": "serialized_native_scheduler_v1",
        "runtime_trust_mode": binding.runtime_trust_mode,
        "formal_measurement": binding.formal_measurement,
    }


def _performance(
    *,
    output_tokens: int,
    target_calls: int = 0,
    accepted_drafts: int = 0,
    committed_tokens: int = 0,
    verified_drafts: int = 0,
) -> dict[str, object]:
    denominator = target_calls or None
    return {
        "target_calls": target_calls,
        "accepted_drafts": accepted_drafts,
        "committed_tokens": committed_tokens,
        "verified_drafts": verified_drafts,
        "survival_weighted_accepted_prefix": None,
        "accepted_drafts_per_verify": (
            accepted_drafts / denominator if denominator is not None else None
        ),
        "committed_tokens_per_verify": (
            committed_tokens / denominator if denominator is not None else None
        ),
        "verified_drafts_per_verify": (
            verified_drafts / denominator if denominator is not None else None
        ),
        "verification_waste": verified_drafts - accepted_drafts,
        "target_calls_per_output_token": target_calls / output_tokens,
        "batch_fill": 1.0,
        "queue_occupancy": 0.0,
        "gpu_busy": None,
        "sm_utilization": None,
        "dram_utilization": None,
        "target_estimated_mfu": None,
        "peak_hbm_bytes": 3072,
        "kv_bytes": 4096,
        "kv_token_capacity": 1024,
        "optimizer_bytes": 128,
        "adaptation_memory_ledger": {
            "active_or_base_bytes": 32,
            "master_fp32_bytes": 32,
            "first_moment_bytes": 32,
            "second_moment_bytes": 64,
            "online_state_bytes": 0,
            "optimizer_metadata_bytes": 0,
            "gradient_bytes": 32,
            "staging_bytes": 16,
            "training_activation_bytes": 32,
            "kv_gather_scratch_bytes": 16,
            "candidate_scratch_bytes": 64,
            "graph_buffer_bytes": 0,
            "telemetry_bytes": 0,
            "resident_bytes": 176,
            "optimizer_bytes": 128,
            "peak_bytes": 320,
        },
        "trainable_parameters": 4,
        "training_cuda_ms": 0.0,
        "optimizer_cuda_ms": 0.0,
        "merge_cuda_ms": 0.0,
        "publish_cuda_ms": 0.0,
        "barrier_cuda_ms": 0.0,
        "exposed_update_ms": 0.0,
        "main_side_overlap_ratio": 0.0,
        "graph_replay_hit_rate": 1.0,
        "updates_launched": 0,
        "updates_published": 0,
        "exactness_violations": 0,
        "version_mismatches": 0,
        "fallbacks": 0,
        "nonfinite_updates": 0,
        "oom_events": 0,
        "retractions": 0,
        "communicator_failures": 0,
        "collective_type": "none_tp1",
        "collective_bytes": 0,
        "collective_duration_ms": 0.0,
        "collective_exposed_wait_ms": 0.0,
        "collective_overlap_ratio": None,
    }


def _state(
    *,
    request_epoch: int,
    generation: int,
    runtime_trust_mode: str | None = None,
    formal_measurement: bool | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "scheduler_idle": True,
        "active_requests": 0,
        "queued_requests": 0,
        "request_pool_active_slots": 0,
        "allocator_current_hbm_bytes": 32,
        "allocator_reserved_hbm_bytes": 64,
        "allocator_peak_hbm_bytes": 3072,
        "kv_token_capacity": 1024,
        "kv_available_tokens": 1024,
        "kv_state_sha256": SHA_A,
        "rng_state_sha256": SHA_B,
        "adapter_state_sha256": SHA_C,
        "adapter_reset_verified": True,
        "adapter_reset_scope": "request",
        "adapter_request_admission_policy": "serialized_native_scheduler_v1",
        "adapter_request_source_point_reset_protocol_sha256": (
            native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
        ),
        "adapter_runtime_trust_mode": runtime_trust_mode,
        "adapter_formal_measurement": formal_measurement,
        "adapter_active_request_id": None,
        "adapter_request_epoch": request_epoch,
        "adapter_source_round": 0,
        "adapter_active_version": 0,
        "adapter_epoch": 1,
        "optimizer_generation": 0,
        "telemetry_generation": 1,
        "completion_event_generation": generation,
        "completion_event_complete": True,
    }


def _zero_row_resets(request_ids: tuple[str, ...]) -> dict[str, object]:
    archive_sha256 = ZERO_SHA256
    previous_receipt_sha256 = ZERO_SHA256
    receipts: list[dict[str, object]] = []
    for request_epoch, request_id in enumerate(request_ids, start=1):
        archive_sha256 = native_terminal.canonical_sha256(
            {
                "schema_version": 1,
                "previous_archive_sha256": archive_sha256,
                "request_epoch": request_epoch,
                "request_id": request_id,
                "updates": [],
                "rounds": [],
            }
        )
        receipt: dict[str, object] = {
            "request_id": request_id,
            "request_epoch": request_epoch,
            "terminal_outcome": "completed",
            "terminal_round": 0,
            "terminal_version": 0,
            "adaptation_state_acquired": True,
            "reset_required": True,
            "state_untouched": False,
            "source_point_identity_sha256": SHA_A,
            "master_reset": True,
            "optimizer_reset": True,
            "inference_reset": True,
            "captured_state_empty": True,
            "runtime_reset": True,
            "sticky_disabled_reason": None,
            "evidence_archive_sha256": archive_sha256,
            "archived_update_count": 0,
            "archived_round_count": 0,
            "previous_receipt_sha256": previous_receipt_sha256,
            "protocol_sha256": (
                native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
            ),
        }
        receipt["receipt_sha256"] = native_terminal.canonical_sha256(receipt)
        previous_receipt_sha256 = str(receipt["receipt_sha256"])
        receipts.append(receipt)
    return {
        "schema_version": 1,
        "reset_scope": "request",
        "request_admission_policy": "serialized_native_scheduler_v1",
        "protocol_sha256": (native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256),
        "final_archive_sha256": archive_sha256,
        "receipts": receipts,
    }


def _one_completed_round(
    request: native_terminal.TerminalRequestExpectation,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    request_epoch = 1
    native_round = {
        "request_epoch": request_epoch,
        "round_index": 1,
        "source_version": 0,
        "request_ids": [request.request_id],
        "prefix_len_before": [2],
        "verify_len": [1],
        "accepted_drafts": [0],
        "committed_tokens": [1],
    }
    archive_sha256 = native_terminal.canonical_sha256(
        {
            "schema_version": 1,
            "previous_archive_sha256": ZERO_SHA256,
            "request_epoch": request_epoch,
            "request_id": request.request_id,
            "updates": [],
            "rounds": [native_round],
        }
    )
    receipt: dict[str, object] = {
        "request_id": request.request_id,
        "request_epoch": request_epoch,
        "terminal_outcome": "completed",
        # The terminal boundary is the exact maximum archived native row;
        # zero-row acquired requests therefore use the natural 0/0 boundary.
        "terminal_round": 1,
        "terminal_version": 0,
        "adaptation_state_acquired": True,
        "reset_required": True,
        "state_untouched": False,
        "source_point_identity_sha256": SHA_A,
        "master_reset": True,
        "optimizer_reset": True,
        "inference_reset": True,
        "captured_state_empty": True,
        "runtime_reset": True,
        "sticky_disabled_reason": None,
        "evidence_archive_sha256": archive_sha256,
        "archived_update_count": 0,
        "archived_round_count": 1,
        "previous_receipt_sha256": ZERO_SHA256,
        "protocol_sha256": native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256,
    }
    receipt["receipt_sha256"] = native_terminal.canonical_sha256(receipt)
    kv = {
        request.request_id: [
            {"start": 0, "end": 3, "source_version": 0},
        ]
    }
    round_row: dict[str, object] = {
        "reset_scope": "request",
        "request_epoch": request_epoch,
        "request_reset_receipt_sha256": receipt["receipt_sha256"],
        "request_id": request.request_id,
        "round_index": 1,
        "proposal_source_version": 0,
        "prefix_len_before": 2,
        "verify_len": 1,
        "accepted_drafts": 0,
        "committed_tokens": 1,
        "target_calls": 1,
        "historical_kv_source_versions_sha256": native_terminal.canonical_sha256(
            kv[request.request_id]
        ),
    }
    round_row["round_sha256"] = native_terminal.canonical_sha256(round_row)
    resets = {
        "schema_version": 1,
        "reset_scope": "request",
        "request_admission_policy": "serialized_native_scheduler_v1",
        "protocol_sha256": (native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256),
        "final_archive_sha256": archive_sha256,
        "receipts": [receipt],
    }
    return resets, [round_row], kv


def _begin(
    binding: native_terminal.NativeTerminalRunBinding,
) -> native_terminal.NativeTerminalBeginReceipt:
    raw: dict[str, object] = {
        "schema_version": 2,
        "kind": "lightcone_terminal_begin_receipt",
        "hook": native_terminal.NATIVE_TERMINAL_EVIDENCE_HOOK,
        **_identity(binding),
        "server_process_id": 1234,
        "server_process_started_ns": 1_000_000,
        "reset_generation": 1,
        "request_source_point_reset_protocol_sha256": (
            native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
        ),
        "prior_state_sha256": SHA_A,
        "reset_state_sha256": SHA_B,
        "warmup_request_ids_sha256": native_terminal.canonical_sha256(
            list(binding.warmup_request_ids)
        ),
        "scored_request_ids_sha256": native_terminal.canonical_sha256(
            list(binding.scored_request_ids)
        ),
    }
    raw["begin_sha256"] = native_terminal.canonical_sha256(raw)
    return native_terminal._validate_begin_receipt(
        raw,
        binding=binding,
        prior_reset_generation=0,
        prior_process=None,
    )


def _reset_payload(
    begin: native_terminal.NativeTerminalBeginReceipt,
    warmup: tuple[native_terminal.TerminalRequestExpectation, ...],
    *,
    with_completed_round: bool = False,
) -> dict[str, object]:
    if with_completed_round:
        warmup_resets, warmup_rounds, warmup_kv = _one_completed_round(warmup[0])
        performance = _performance(
            output_tokens=1,
            target_calls=1,
            committed_tokens=1,
            verified_drafts=1,
        )
    else:
        warmup_resets = _zero_row_resets(tuple(row.request_id for row in warmup))
        warmup_rounds = []
        warmup_kv = {}
        performance = _performance(output_tokens=len(warmup))
    warmup_rows = [native_terminal._server_request_row(row) for row in warmup]
    warmup_state = _state(
        request_epoch=len(warmup),
        generation=3,
        runtime_trust_mode=begin.binding.runtime_trust_mode,
        formal_measurement=begin.binding.formal_measurement,
    )
    reset_state = _state(
        request_epoch=0,
        generation=4,
        runtime_trust_mode=begin.binding.runtime_trust_mode,
        formal_measurement=begin.binding.formal_measurement,
    )
    raw: dict[str, object] = {
        "schema_version": 2,
        "kind": "lightcone_terminal_reset_receipt",
        "hook": native_terminal.NATIVE_TERMINAL_EVIDENCE_HOOK,
        **_identity(begin.binding),
        "server_process_id": begin.server_process_id,
        "server_process_started_ns": begin.server_process_started_ns,
        "begin_sha256": begin.begin_sha256,
        "reset_generation": 2,
        "request_source_point_reset_protocol_sha256": (
            native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
        ),
        "prior_trace_run_id": None,
        "next_trace_run_id": begin.binding.run_id,
        "warmup_request_rows_sha256": native_terminal.canonical_sha256(warmup_rows),
        "warmup_performance_sha256": native_terminal.canonical_sha256(performance),
        "discarded_native_sha256": SHA_D,
        "warmup_state_sha256": native_terminal.canonical_sha256(warmup_state),
        "reset_state_sha256": native_terminal.canonical_sha256(reset_state),
        "expected_scored_request_ids_sha256": native_terminal.canonical_sha256(
            list(begin.binding.scored_request_ids)
        ),
        "completion_event_generation": 4,
        "warmup_request_rows": warmup_rows,
        "warmup_round_rows": warmup_rounds,
        "warmup_update_rows": [],
        "warmup_historical_kv_source_versions": warmup_kv,
        "warmup_request_source_point_resets": warmup_resets,
        "warmup_performance_counters": performance,
        "warmup_state": warmup_state,
        "reset_state": reset_state,
    }
    raw["reset_sha256"] = native_terminal.canonical_sha256(raw)
    return raw


def _validated_reset(
    begin: native_terminal.NativeTerminalBeginReceipt,
    warmup: tuple[native_terminal.TerminalRequestExpectation, ...],
) -> native_terminal.NativeTerminalResetReceipt:
    return native_terminal._validate_reset_receipt(
        _reset_payload(begin, warmup),
        begin=begin,
        warmup_requests=warmup,
    )


def validated_zero_row_terminal(
    *,
    warmup_ids: tuple[str, ...] = ("warm-0",),
    scored_ids: tuple[str, ...] = ("score-0",),
    rank_config_sha256: str = SHA_C,
    runtime_trust_mode: str | None = None,
    formal_measurement: bool | None = None,
) -> tuple[
    native_terminal.ValidatedNativeTerminalEvidence,
    tuple[native_terminal.TerminalRequestExpectation, ...],
    tuple[native_terminal.TerminalRequestExpectation, ...],
]:
    """Build one fully deep-validated current fixture for cross-layer tests."""

    binding = _binding(
        warmup_ids=warmup_ids,
        scored_ids=scored_ids,
        rank_config_sha256=rank_config_sha256,
        runtime_trust_mode=runtime_trust_mode,
        formal_measurement=formal_measurement,
    )
    warmup = tuple(_request(request_id) for request_id in warmup_ids)
    scored = tuple(_request(request_id) for request_id in scored_ids)
    begin = _begin(binding)
    reset = _validated_reset(begin, warmup)
    request_rows = [native_terminal._server_request_row(row) for row in scored]
    terminal: dict[str, object] = {
        "schema_version": 2,
        "hook": native_terminal.NATIVE_TERMINAL_EVIDENCE_HOOK,
        **_identity(binding),
        "request_source_point_reset_protocol_sha256": (
            native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
        ),
        "server_process_id": begin.server_process_id,
        "server_process_started_ns": begin.server_process_started_ns,
        "expected_request_ids": list(scored_ids),
        "reset_receipt_sha256": reset.reset_sha256,
        "request_round_rows": {"requests": request_rows, "rounds": []},
        "update_rows": [],
        "performance_counters": _performance(output_tokens=len(scored)),
        "historical_kv_source_versions": {},
        "request_source_point_resets": _zero_row_resets(scored_ids),
        "final_state": _state(
            request_epoch=len(scored),
            generation=5,
            runtime_trust_mode=binding.runtime_trust_mode,
            formal_measurement=binding.formal_measurement,
        ),
        "completion_marker": "TERMINAL_COMPLETE",
    }
    terminal["terminal_sha256"] = native_terminal.canonical_sha256(terminal)
    terminal["attestation"] = {
        "schema_version": 1,
        "status": "UNAVAILABLE",
        "challenge_nonce_sha256": binding.challenge_nonce_sha256,
        "message_sha256": native_terminal.canonical_sha256(
            native_terminal._attestation_message(terminal)
        ),
        "attester_id": None,
        "trust_domain": None,
        "signature_hex": None,
    }
    return (
        native_terminal._validate_terminal(
            terminal,
            begin=begin,
            reset=reset,
            requests=scored,
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
        ),
        warmup,
        scored,
    )


def test_request_reset_protocol_pin_matches_native_0008() -> None:
    expected_literal = (
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
    assert native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_CANONICAL_JSON == (
        expected_literal
    )
    assert len(expected_literal) == 1055
    assert native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256 == (
        "95f9ae5c444f26c249475d72a5a64f9e02b8e27cc9e96b7c305fbaafc1020166"
    )
    assert hashlib.sha256(expected_literal).hexdigest() == (
        native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
    )
    assert native_terminal.NATIVE_TERMINAL_EVIDENCE_FIELDS == (
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
        "reset_scope",
        "request_admission_policy",
        "request_source_point_reset_protocol_sha256",
        "runtime_trust_mode",
        "formal_measurement",
        "expected_request_ids",
        "reset_receipt_sha256",
        "request_round_rows",
        "update_rows",
        "performance_counters",
        "historical_kv_source_versions",
        "request_source_point_resets",
        "final_state",
        "completion_marker",
        "terminal_sha256",
        "attestation",
    )


def test_schema2_tp1_accepts_warmup_and_scored_zero_row_acquired_receipts() -> None:
    binding = _binding()
    warmup = (_request("warm-0"),)
    scored = (_request("score-0"),)
    begin = _begin(binding)
    reset = _validated_reset(begin, warmup)
    scored_resets = _zero_row_resets(("score-0",))
    request_rows = [native_terminal._server_request_row(scored[0])]
    performance = _performance(output_tokens=1)
    terminal: dict[str, object] = {
        "schema_version": 2,
        "hook": native_terminal.NATIVE_TERMINAL_EVIDENCE_HOOK,
        **_identity(binding),
        "request_source_point_reset_protocol_sha256": (
            native_terminal.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
        ),
        "server_process_id": begin.server_process_id,
        "server_process_started_ns": begin.server_process_started_ns,
        "expected_request_ids": ["score-0"],
        "reset_receipt_sha256": reset.reset_sha256,
        "request_round_rows": {"requests": request_rows, "rounds": []},
        "update_rows": [],
        "performance_counters": performance,
        "historical_kv_source_versions": {},
        "request_source_point_resets": scored_resets,
        "final_state": _state(request_epoch=1, generation=5),
        "completion_marker": "TERMINAL_COMPLETE",
    }
    terminal["terminal_sha256"] = native_terminal.canonical_sha256(terminal)
    message = native_terminal._attestation_message(terminal)
    terminal["attestation"] = {
        "schema_version": 1,
        "status": "UNAVAILABLE",
        "challenge_nonce_sha256": binding.challenge_nonce_sha256,
        "message_sha256": native_terminal.canonical_sha256(message),
        "attester_id": None,
        "trust_domain": None,
        "signature_hex": None,
    }

    result = native_terminal._validate_terminal(
        terminal,
        begin=begin,
        reset=reset,
        requests=scored,
        trusted_attester_policy=NO_TRUSTED_ATTESTERS,
    )

    assert result.terminal_schema_version == 2
    assert reset.warmup_request_source_point_resets is not None
    assert (
        reset.warmup_request_source_point_resets.receipts[0].archived_round_count == 0
    )
    assert result.request_source_point_resets is not None
    assert result.request_source_point_resets.receipts[0].archived_update_count == 0
    native_batch = result.to_native_evidence_batch()
    assert native_batch.rounds == ()
    assert native_batch.updates == ()


def test_request_reset_receipts_cover_only_server_submitted_mixed_lifecycle() -> None:
    binding = _binding(
        warmup_ids=(),
        scored_ids=("score-server", "score-client"),
    )
    requests = (
        _request("score-server"),
        native_terminal.TerminalRequestExpectation(
            request_id="score-client",
            input_token_ids=(21, 22),
            output_token_ids=None,
            terminal_status="cancelled",
            terminal_reason="pre_admission_cancelled",
            submitted_to_server=False,
        ),
    )

    validated = native_terminal._validate_request_source_point_resets(
        _zero_row_resets(("score-server",)),
        binding=binding,
        requests=requests,
    )
    assert tuple(receipt.request_id for receipt in validated.receipts) == (
        "score-server",
    )

    with pytest.raises(RuntimeError, match="coverage differs from submission"):
        native_terminal._validate_request_source_point_resets(
            _zero_row_resets(("score-server", "score-client")),
            binding=binding,
            requests=requests,
        )


def test_schema2_runtime_trust_pair_is_bound_through_terminal_state() -> None:
    evidence, _warmup, _scored = validated_zero_row_terminal(
        runtime_trust_mode="trusted_single_operator_empirical_no_signature",
        formal_measurement=False,
    )

    assert evidence.binding.runtime_trust_mode == (
        "trusted_single_operator_empirical_no_signature"
    )
    assert evidence.binding.formal_measurement is False
    assert evidence.to_dict()["final_state"]["adapter_runtime_trust_mode"] == (
        evidence.binding.runtime_trust_mode
    )

    with pytest.raises(ValueError, match="runtime trust/formal measurement pair"):
        _binding(
            runtime_trust_mode="release_verified_signature",
            formal_measurement=False,
        ).validate()


def test_warmup_round_rehash_without_archive_update_is_rejected() -> None:
    binding = _binding()
    warmup = (_request("warm-0"),)
    begin = _begin(binding)
    raw = _reset_payload(begin, warmup, with_completed_round=True)
    round_row = raw["warmup_round_rows"][0]
    assert isinstance(round_row, dict)
    round_row["verify_len"] = 2
    round_row["round_sha256"] = native_terminal.canonical_sha256(
        {key: value for key, value in round_row.items() if key != "round_sha256"}
    )
    performance = raw["warmup_performance_counters"]
    assert isinstance(performance, dict)
    performance["verified_drafts"] = 2
    performance["verified_drafts_per_verify"] = 2.0
    performance["verification_waste"] = 2
    raw["warmup_performance_sha256"] = native_terminal.canonical_sha256(performance)
    raw["reset_sha256"] = native_terminal.canonical_sha256(
        {key: value for key, value in raw.items() if key != "reset_sha256"}
    )

    with pytest.raises(RuntimeError, match="archive replay"):
        native_terminal._validate_reset_receipt(
            raw,
            begin=begin,
            warmup_requests=warmup,
        )


def test_resigned_terminal_boundary_cannot_exceed_archived_rows() -> None:
    binding = _binding()
    warmup = (_request("warm-0"),)
    begin = _begin(binding)
    raw = _reset_payload(begin, warmup, with_completed_round=True)
    resets = raw["warmup_request_source_point_resets"]
    assert isinstance(resets, dict)
    receipts = resets["receipts"]
    assert isinstance(receipts, list) and len(receipts) == 1
    receipt = receipts[0]
    assert isinstance(receipt, dict)
    receipt["terminal_round"] = 2
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = native_terminal.canonical_sha256(receipt)
    rounds = raw["warmup_round_rows"]
    assert isinstance(rounds, list) and len(rounds) == 1
    round_row = rounds[0]
    assert isinstance(round_row, dict)
    round_row["request_reset_receipt_sha256"] = receipt["receipt_sha256"]
    round_row.pop("round_sha256")
    round_row["round_sha256"] = native_terminal.canonical_sha256(round_row)
    raw["reset_sha256"] = native_terminal.canonical_sha256(
        {key: value for key, value in raw.items() if key != "reset_sha256"}
    )

    with pytest.raises(RuntimeError, match="terminal boundary"):
        native_terminal._validate_reset_receipt(
            raw,
            begin=begin,
            warmup_requests=warmup,
        )


def test_reordered_acquired_epochs_fail_after_receipt_chain_is_resigned() -> None:
    binding = _binding(warmup_ids=("warm-0", "warm-1"))
    requests = (_request("warm-0"), _request("warm-1"))
    resets = _zero_row_resets(("warm-0", "warm-1"))
    original = resets["receipts"]
    assert isinstance(original, list)
    reordered = [copy.deepcopy(original[1]), copy.deepcopy(original[0])]
    archive_sha256 = ZERO_SHA256
    previous_receipt_sha256 = ZERO_SHA256
    for receipt in reordered:
        assert isinstance(receipt, dict)
        archive_sha256 = native_terminal.canonical_sha256(
            {
                "schema_version": 1,
                "previous_archive_sha256": archive_sha256,
                "request_epoch": receipt["request_epoch"],
                "request_id": receipt["request_id"],
                "updates": [],
                "rounds": [],
            }
        )
        receipt["evidence_archive_sha256"] = archive_sha256
        receipt["previous_receipt_sha256"] = previous_receipt_sha256
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = native_terminal.canonical_sha256(receipt)
        previous_receipt_sha256 = str(receipt["receipt_sha256"])
    resets["receipts"] = reordered
    resets["final_archive_sha256"] = archive_sha256

    with pytest.raises(RuntimeError, match="required reset epoch"):
        native_terminal._validate_request_source_point_resets(
            resets,
            binding=binding,
            requests=requests,
        )


def test_request_archive_validator_handles_11k_zero_row_receipts() -> None:
    request_ids = tuple(f"request-{index}" for index in range(11_000))
    binding = _binding(warmup_ids=request_ids)
    requests = tuple(_request(request_id) for request_id in request_ids)
    parsed = native_terminal._validate_request_source_point_resets(
        _zero_row_resets(request_ids),
        binding=binding,
        requests=requests,
    )

    native_terminal._validate_request_reset_row_coverage(
        parsed,
        requests=requests,
        rounds=[],
        updates=[],
    )
    assert len(parsed.receipts) == 11_000
