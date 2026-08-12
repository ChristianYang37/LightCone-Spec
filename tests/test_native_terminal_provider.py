from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.orchestration.executor import (
    TRUSTED_NATIVE_ATTESTER_UNAVAILABLE_REASON,
    native_evidence_preflight,
)
from lightcone_spec.orchestration.native_terminal import (
    CAPABILITY_PATH,
    NATIVE_TERMINAL_EVIDENCE_FIELDS,
    NATIVE_TERMINAL_EVIDENCE_HOOK,
    PINNED_SGLANG_PATCH_SHA256,
    PINNED_SGLANG_TREE,
    TERMINAL_EVIDENCE_PATH,
    NativeTerminalProvider,
    NativeTerminalRunBinding,
    TerminalRequestExpectation,
    canonical_json_bytes,
    canonical_sha256,
    validate_native_terminal_artifact,
)
from lightcone_spec.runtime.attestation import TrustedAttesterPolicy
from lightcone_spec.telemetry.writer import EvidenceWriter

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _release_policy(
    private_key: Ed25519PrivateKey,
    *,
    attester_id: str = "prod-hsm-1",
    key_id: str = "release-key-1",
) -> TrustedAttesterPolicy:
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    policy = TrustedAttesterPolicy(
        policy_id="release-terminal-attesters-v1",
        trusted_attesters=((attester_id, key_id, public_key_sha256),),
        public_keys=((public_key_sha256, base64.b64encode(public_key).decode()),),
    )
    policy.validate()
    return policy


def _binding(
    *,
    method: str = "static",
    warmup: tuple[str, ...] = ("warm-0",),
    scored: tuple[str, ...] = ("score-0",),
) -> NativeTerminalRunBinding:
    return NativeTerminalRunBinding(
        run_id="run-0",
        run_nonce_sha256=SHA_A,
        execution_plan_sha256=SHA_B,
        rank_config_sha256=SHA_C,
        attempt_id="attempt-0",
        session_id="session-0",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=SHA_D,
        method=method,
        warmup_request_ids=warmup,
        scored_request_ids=scored,
    )


def _server_request(
    request_id: str,
    *,
    inputs: tuple[int, ...],
    outputs: tuple[int, ...],
    status: str = "completed",
    reason: str = "FINISH_LENGTH",
) -> TerminalRequestExpectation:
    return TerminalRequestExpectation(
        request_id=request_id,
        input_token_ids=inputs,
        output_token_ids=outputs,
        terminal_status=status,
        terminal_reason=reason,
        submitted_to_server=True,
    )


def _client_request(
    request_id: str,
    *,
    inputs: tuple[int, ...],
    status: str = "rejected",
    reason: str = "lane_deadline",
) -> TerminalRequestExpectation:
    return TerminalRequestExpectation(
        request_id=request_id,
        input_token_ids=inputs,
        output_token_ids=None,
        terminal_status=status,
        terminal_reason=reason,
        submitted_to_server=False,
    )


def _identity(binding: NativeTerminalRunBinding) -> dict[str, object]:
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


def _server_row(request: TerminalRequestExpectation) -> dict[str, object]:
    assert request.output_token_ids is not None
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


def _state(*, method: str, generation: int, published: int = 0) -> dict[str, object]:
    adapted = method in {"tts", "l0"}
    return {
        "schema_version": 1,
        "scheduler_idle": True,
        "active_requests": 0,
        "queued_requests": 0,
        "request_pool_active_slots": 0,
        "allocator_current_hbm_bytes": 32,
        "allocator_reserved_hbm_bytes": 64,
        "allocator_peak_hbm_bytes": 3072,
        "kv_token_capacity": 1024,
        "kv_available_tokens": 1000 if adapted else 1024,
        "kv_state_sha256": SHA_A,
        "rng_state_sha256": SHA_B,
        "adapter_state_sha256": SHA_C,
        "adapter_reset_verified": True,
        "adapter_active_version": published if adapted else 0,
        "adapter_epoch": 1 if adapted else 0,
        "optimizer_generation": published if adapted else 0,
        "telemetry_generation": 1 if adapted else 0,
        "completion_event_generation": generation,
        "completion_event_complete": True,
    }


def _performance(
    *,
    method: str,
    output_tokens: int,
    target_calls: int | None = None,
) -> dict[str, object]:
    target_calls = (
        output_tokens
        if target_calls is None and method == "target_only"
        else 1
        if target_calls is None
        else target_calls
    )
    speculative = method != "target_only"
    accepted = 1 if speculative else None
    committed = 2 if speculative else None
    verified = 1 if speculative else None
    adapted = method in {"tts", "l0"}
    return {
        "target_calls": target_calls,
        "accepted_drafts": accepted,
        "committed_tokens": committed,
        "verified_drafts": verified,
        "survival_weighted_accepted_prefix": None,
        "accepted_drafts_per_verify": accepted / target_calls if speculative else None,
        "committed_tokens_per_verify": (
            committed / target_calls if speculative else None
        ),
        "verified_drafts_per_verify": (
            verified / target_calls if speculative else None
        ),
        "verification_waste": verified - accepted if speculative else None,
        "target_calls_per_output_token": (
            target_calls / output_tokens if output_tokens else None
        ),
        "batch_fill": 1.0,
        "queue_occupancy": 0.0,
        "gpu_busy": None,
        "sm_utilization": None,
        "dram_utilization": None,
        "target_estimated_mfu": None,
        "peak_hbm_bytes": 3072,
        "kv_bytes": 4096,
        "kv_token_capacity": 1024,
        "optimizer_bytes": 128 if adapted else 0,
        "adaptation_memory_ledger": (
            {
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
            }
            if adapted
            else None
        ),
        "trainable_parameters": 4 if adapted else 0,
        "training_cuda_ms": 0.4 if adapted else None,
        "optimizer_cuda_ms": 0.2 if adapted else None,
        "merge_cuda_ms": 0.1 if adapted else None,
        "publish_cuda_ms": 0.1 if adapted else None,
        "barrier_cuda_ms": 0.0 if adapted else None,
        "exposed_update_ms": 0.1 if adapted else None,
        "main_side_overlap_ratio": 0.8 if adapted else None,
        "graph_replay_hit_rate": 1.0,
        "updates_launched": 1 if adapted else 0,
        "updates_published": 1 if adapted else 0,
        "exactness_violations": 0 if speculative else None,
        "version_mismatches": 0 if speculative else None,
        "fallbacks": 0 if speculative else None,
        "nonfinite_updates": 0 if speculative else None,
        "oom_events": 0,
        "retractions": 0,
        "communicator_failures": 0,
        "collective_type": "none_tp1",
        "collective_bytes": 0,
        "collective_duration_ms": 0.0,
        "collective_exposed_wait_ms": 0.0,
        "collective_overlap_ratio": None,
    }


def _round_and_update(
    request: TerminalRequestExpectation,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    assert request.output_token_ids is not None
    prefix = len(request.input_token_ids)
    end = prefix + len(request.output_token_ids)
    kv = {
        request.request_id: [
            {"start": 0, "end": end, "source_version": 0},
        ]
    }
    round_row: dict[str, object] = {
        "request_id": request.request_id,
        "round_index": 1,
        "proposal_source_version": 0,
        "prefix_len_before": prefix,
        "verify_len": 1,
        "accepted_drafts": 1,
        "committed_tokens": 2,
        "target_calls": 1,
        "historical_kv_source_versions": kv[request.request_id],
    }
    round_row["round_sha256"] = canonical_sha256(round_row)
    update: dict[str, object] = {
        "update_index": 0,
        "cohort_sha256": SHA_E,
        "cohort_epoch": 1,
        "parameter_layout_sha256": SHA_F,
        "source_round": 1,
        "source_version": 0,
        "request_ids": [request.request_id],
        "prefix_len_before": [prefix],
        "optimizer_step": 1,
        "published_version": 1,
        "status": "published",
        "loss": 0.25,
        "gradient_norm": 0.5,
        "reconstruction_ok": True,
        "reconstruction_max_abs": 0.0,
        "reconstruction_relative_rms": 0.0,
        "reconstruction_top1_match": 1.0,
        "supervision_nonempty": True,
        "reconstruction_mean_kl": 0.0,
        "online_hint_error": None,
        "online_ensemble_entropy": None,
        "online_effective_experts": None,
        "online_expert_probabilities": None,
        "online_cumulative_losses": None,
        "online_expert_gradient_norms": None,
    }
    update["update_sha256"] = canonical_sha256(update)
    return [round_row], [update], kv


class FakeAdminTransport:
    def __init__(
        self,
        *,
        binding: NativeTerminalRunBinding,
        warmup: tuple[TerminalRequestExpectation, ...],
        scored: tuple[TerminalRequestExpectation, ...],
        capability_mutator: Callable[[dict[str, object]], None] | None = None,
        terminal_mutator: Callable[[dict[str, object]], None] | None = None,
        override_client_with_server: bool = False,
        attester_id: str | None = None,
        trust_domain: str | None = None,
        signing_key: Ed25519PrivateKey | None = None,
    ) -> None:
        self.binding = binding
        self.warmup = warmup
        self.scored = scored
        self.capability_mutator = capability_mutator
        self.terminal_mutator = terminal_mutator
        self.override_client_with_server = override_client_with_server
        self.attester_id = attester_id
        self.trust_domain = trust_domain
        self.signing_key = signing_key
        self.get_paths: list[str] = []
        self.posts: list[tuple[str, dict[str, object]]] = []
        self.begin_receipt: dict[str, object] | None = None
        self.reset_receipt: dict[str, object] | None = None
        # Deliberately forge caller-visible attributes.  The provider must never
        # consult them as a capability receipt.
        self.native_evidence_hook = NATIVE_TERMINAL_EVIDENCE_HOOK
        self.patched_sglang_tree = PINNED_SGLANG_TREE
        self.supported_methods = frozenset({binding.method})

    async def get_json(self, path: str) -> object:
        self.get_paths.append(path)
        value: dict[str, object] = {
            "schema_version": 1,
            "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
            "required_fields": list(NATIVE_TERMINAL_EVIDENCE_FIELDS),
            "supported_methods": ["l0", "static", "target_only", "tts"],
            "enabled": True,
            "active_method": self.binding.method,
            "method_evidence_supported": True,
            "topology_supported": True,
            "trusted_attester_configured": self.attester_id is not None,
        }
        if self.capability_mutator is not None:
            self.capability_mutator(value)
        return value

    async def post_json(self, path: str, body: Mapping[str, object]) -> object:
        copied = copy.deepcopy(dict(body))
        self.posts.append((path, copied))
        action = copied["action"]
        payload = copied["payload"]
        assert isinstance(payload, dict)
        if action == "begin":
            self.begin_receipt = self._begin(payload)
            return self.begin_receipt
        if action == "reset":
            self.reset_receipt = self._reset(payload)
            return self.reset_receipt
        if action == "finalize":
            terminal = self._terminal(payload)
            if self.terminal_mutator is not None:
                self.terminal_mutator(terminal)
            return terminal
        raise AssertionError(f"unexpected action {action}")

    def _begin(self, payload: dict[str, object]) -> dict[str, object]:
        assert payload == self.binding.begin_payload()
        value: dict[str, object] = {
            "schema_version": 1,
            "kind": "lightcone_terminal_begin_receipt",
            "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
            **_identity(self.binding),
            "server_process_id": 1234,
            "server_process_started_ns": 1_000_000,
            "reset_generation": 1,
            "prior_state_sha256": SHA_A,
            "reset_state_sha256": SHA_B,
            "warmup_request_ids_sha256": canonical_sha256(
                list(self.binding.warmup_request_ids)
            ),
            "scored_request_ids_sha256": canonical_sha256(
                list(self.binding.scored_request_ids)
            ),
        }
        value["begin_sha256"] = canonical_sha256(value)
        return value

    def _reset(self, payload: dict[str, object]) -> dict[str, object]:
        assert self.begin_receipt is not None
        assert payload == {
            "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
            "run_id": self.binding.run_id,
            "begin_sha256": self.begin_receipt["begin_sha256"],
        }
        value: dict[str, object] = {
            "schema_version": 1,
            "kind": "lightcone_terminal_reset_receipt",
            "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
            **_identity(self.binding),
            "server_process_id": 1234,
            "server_process_started_ns": 1_000_000,
            "begin_sha256": self.begin_receipt["begin_sha256"],
            "reset_generation": 2,
            "prior_trace_run_id": self.binding.previous_run_id,
            "next_trace_run_id": self.binding.run_id,
            "warmup_request_rows_sha256": canonical_sha256(
                [_server_row(request) for request in self.warmup]
            ),
            "warmup_performance_sha256": SHA_C,
            "discarded_native_sha256": SHA_D,
            "warmup_state_sha256": SHA_E,
            "reset_state_sha256": SHA_F,
            "expected_scored_request_ids_sha256": canonical_sha256(
                list(self.binding.scored_request_ids)
            ),
            "completion_event_generation": 4,
        }
        value["reset_sha256"] = canonical_sha256(value)
        return value

    def _terminal(self, payload: dict[str, object]) -> dict[str, object]:
        assert self.reset_receipt is not None
        assert set(payload) == {
            "hook",
            "run_id",
            "reset_sha256",
            "client_terminal_rows",
        }
        assert payload["hook"] == NATIVE_TERMINAL_EVIDENCE_HOOK
        assert payload["run_id"] == self.binding.run_id
        assert payload["reset_sha256"] == self.reset_receipt["reset_sha256"]
        client_rows = payload["client_terminal_rows"]
        assert isinstance(client_rows, list)
        by_client_id = {row["request_id"]: row for row in client_rows}
        request_rows: list[dict[str, object]] = []
        for request in self.scored:
            if request.submitted_to_server:
                request_rows.append(_server_row(request))
                continue
            if self.override_client_with_server:
                forged = _server_request(
                    request.request_id,
                    inputs=request.input_token_ids,
                    outputs=(),
                    status="aborted",
                    reason="FINISH_ABORT",
                )
                request_rows.append(_server_row(forged))
                continue
            row = copy.deepcopy(by_client_id[request.request_id])
            assert set(row) == {
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
            row["terminal_source"] = "client_reconciliation"
            row["output_tokens"] = 0
            row["request_sha256"] = canonical_sha256(row)
            request_rows.append(row)
        output_tokens = sum(int(row["output_tokens"]) for row in request_rows)
        if self.binding.method in {"tts", "l0"}:
            rounds, updates, kv = _round_and_update(self.scored[0])
        else:
            rounds, updates, kv = [], [], {}
        performance = _performance(
            method=self.binding.method,
            output_tokens=output_tokens,
        )
        value: dict[str, object] = {
            "schema_version": 1,
            "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
            **_identity(self.binding),
            "server_process_id": 1234,
            "server_process_started_ns": 1_000_000,
            "expected_request_ids": list(self.binding.scored_request_ids),
            "reset_receipt_sha256": self.reset_receipt["reset_sha256"],
            "request_round_rows": {"requests": request_rows, "rounds": rounds},
            "update_rows": updates,
            "performance_counters": performance,
            "historical_kv_source_versions": kv,
            "final_state": _state(
                method=self.binding.method,
                generation=5,
                published=1 if self.binding.method in {"tts", "l0"} else 0,
            ),
            "completion_marker": "TERMINAL_COMPLETE",
        }
        value["terminal_sha256"] = canonical_sha256(value)
        message = {
            "schema_version": 1,
            "kind": "lightcone_terminal_attestation_challenge",
            "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
            "challenge_nonce_sha256": self.binding.challenge_nonce_sha256,
            "terminal_sha256": value["terminal_sha256"],
            "run_id": self.binding.run_id,
            "run_nonce_sha256": self.binding.run_nonce_sha256,
            "server_process_id": 1234,
            "server_process_started_ns": 1_000_000,
            "session_id": self.binding.session_id,
            "session_epoch": self.binding.session_epoch,
            "attempt_id": self.binding.attempt_id,
        }
        if self.attester_id is None:
            value["attestation"] = {
                "schema_version": 1,
                "status": "UNAVAILABLE",
                "challenge_nonce_sha256": self.binding.challenge_nonce_sha256,
                "message_sha256": canonical_sha256(message),
                "attester_id": None,
                "trust_domain": None,
                "signature_hex": None,
            }
        else:
            assert self.signing_key is not None
            value["attestation"] = {
                "schema_version": 1,
                "status": "SIGNED",
                "challenge_nonce_sha256": self.binding.challenge_nonce_sha256,
                "message_sha256": canonical_sha256(message),
                "attester_id": self.attester_id,
                "trust_domain": self.trust_domain,
                "signature_hex": self.signing_key.sign(
                    canonical_json_bytes(message)
                ).hex(),
            }
        return value


async def _run(
    transport: FakeAdminTransport,
    *,
    provider: NativeTerminalProvider | None = None,
):
    provider = provider or NativeTerminalProvider(transport)
    begin = await provider.begin(transport.binding)
    reset = await provider.reset(warmup_requests=transport.warmup)
    terminal = await provider.finalize(requests=transport.scored)
    return provider, begin, reset, terminal


def _rehash_terminal(value: dict[str, object]) -> None:
    unsigned = dict(value)
    unsigned.pop("terminal_sha256", None)
    unsigned.pop("attestation", None)
    value["terminal_sha256"] = canonical_sha256(unsigned)
    message = {
        "schema_version": 1,
        "kind": "lightcone_terminal_attestation_challenge",
        "hook": NATIVE_TERMINAL_EVIDENCE_HOOK,
        "challenge_nonce_sha256": value["challenge_nonce_sha256"],
        "terminal_sha256": value["terminal_sha256"],
        "run_id": value["run_id"],
        "run_nonce_sha256": value["run_nonce_sha256"],
        "server_process_id": value["server_process_id"],
        "server_process_started_ns": value["server_process_started_ns"],
        "session_id": value["session_id"],
        "session_epoch": value["session_epoch"],
        "attempt_id": value["attempt_id"],
    }
    attestation = value["attestation"]
    assert isinstance(attestation, dict)
    attestation["message_sha256"] = canonical_sha256(message)


def test_static_protocol_is_exact_and_unavailable_attestation_is_untrusted() -> None:
    binding = _binding()
    warmup = (_server_request("warm-0", inputs=(1,), outputs=(2,)),)
    scored = (_server_request("score-0", inputs=(3, 4), outputs=(5, 6)),)
    transport = FakeAdminTransport(binding=binding, warmup=warmup, scored=scored)

    provider, begin, reset, terminal = asyncio.run(_run(transport))

    assert provider.phase == "FINALIZED"
    assert transport.get_paths == [CAPABILITY_PATH]
    assert [path for path, _ in transport.posts] == [TERMINAL_EVIDENCE_PATH] * 3
    assert [body["action"] for _, body in transport.posts] == [
        "begin",
        "reset",
        "finalize",
    ]
    assert begin.begin_sha256 == transport.begin_receipt["begin_sha256"]
    assert reset.reset_sha256 == transport.reset_receipt["reset_sha256"]
    assert terminal.terminal_sha256 == terminal.to_dict()["terminal_sha256"]
    assert terminal.attestation.status == "UNAVAILABLE"
    assert terminal.trusted_attestation is False
    batch = terminal.to_native_evidence_batch()
    assert batch.rounds == ()
    assert batch.updates == ()
    assert dict(batch.performance_overrides)["exactness_violations"] == 0


def test_non_submitted_reconciliation_has_no_output_identity() -> None:
    binding = _binding(warmup=(), scored=("score-0", "score-1"))
    scored = (
        _server_request("score-0", inputs=(1,), outputs=(2,)),
        _client_request(
            "score-1", inputs=(3,), status="timed_out", reason="lane_timeout"
        ),
    )
    transport = FakeAdminTransport(binding=binding, warmup=(), scored=scored)

    _, _, _, terminal = asyncio.run(_run(transport))

    finalize = transport.posts[-1][1]
    payload = finalize["payload"]
    assert isinstance(payload, dict)
    client_rows = payload["client_terminal_rows"]
    assert isinstance(client_rows, list) and len(client_rows) == 1
    assert "output_tokens" not in client_rows[0]
    assert "ordered_output_token_ids" not in client_rows[0]
    response_rows = terminal.to_dict()["request_round_rows"]["requests"]
    assert response_rows[1]["terminal_source"] == "client_reconciliation"
    assert response_rows[1]["output_tokens"] == 0


def test_tts_round_update_kv_closure_converts_without_bucket_imputation() -> None:
    binding = _binding(method="tts", warmup=())
    scored = (_server_request("score-0", inputs=(1, 2), outputs=(3, 4)),)
    transport = FakeAdminTransport(binding=binding, warmup=(), scored=scored)

    _, _, _, terminal = asyncio.run(_run(transport))

    batch = terminal.to_native_evidence_batch()
    assert len(batch.rounds) == 1
    assert batch.rounds[0].generated_tokens_before == 0
    assert batch.rounds[0].kv_source_versions == (
        '[{"end":4,"source_version":0,"start":0}]'
    )
    assert len(batch.updates) == 1
    assert batch.updates[0].candidate_status == "published"
    assert batch.updates[0].training_cuda_ms is None
    assert dict(batch.performance_overrides)["training_cuda_ms"] == 0.4


@pytest.mark.parametrize("location", ["capability", "terminal", "request"])
def test_unknown_fields_fail_closed(location: str) -> None:
    binding = _binding(warmup=())
    scored = (_server_request("score-0", inputs=(1,), outputs=(2, 3)),)

    def capability_mutator(value: dict[str, object]) -> None:
        if location == "capability":
            value["caller_claim"] = True

    def terminal_mutator(value: dict[str, object]) -> None:
        if location == "terminal":
            value["caller_claim"] = True
        elif location == "request":
            request_round = value["request_round_rows"]
            assert isinstance(request_round, dict)
            request_round["requests"][0]["caller_claim"] = True
            _rehash_terminal(value)

    transport = FakeAdminTransport(
        binding=binding,
        warmup=(),
        scored=scored,
        capability_mutator=capability_mutator,
        terminal_mutator=terminal_mutator,
    )
    provider = NativeTerminalProvider(transport)

    with pytest.raises(ValueError, match="fields|capability"):
        asyncio.run(_run(transport, provider=provider))
    assert provider.phase == "FAILED"


@pytest.mark.parametrize("location", ["request", "terminal"])
def test_content_digest_tamper_fails_closed(location: str) -> None:
    binding = _binding(warmup=())
    scored = (_server_request("score-0", inputs=(1,), outputs=(2, 3)),)

    def terminal_mutator(value: dict[str, object]) -> None:
        if location == "request":
            request_round = value["request_round_rows"]
            assert isinstance(request_round, dict)
            request_round["requests"][0]["request_sha256"] = SHA_A
            _rehash_terminal(value)
        else:
            value["terminal_sha256"] = SHA_A

    transport = FakeAdminTransport(
        binding=binding,
        warmup=(),
        scored=scored,
        terminal_mutator=terminal_mutator,
    )
    provider = NativeTerminalProvider(transport)

    with pytest.raises(ValueError, match="content digest"):
        asyncio.run(_run(transport, provider=provider))
    assert provider.phase == "FAILED"


def test_server_row_cannot_override_non_submitted_terminal() -> None:
    binding = _binding(warmup=())
    scored = (_client_request("score-0", inputs=(1,), status="cancelled"),)
    transport = FakeAdminTransport(
        binding=binding,
        warmup=(),
        scored=scored,
        override_client_with_server=True,
    )
    provider = NativeTerminalProvider(transport)

    with pytest.raises(RuntimeError, match="cannot override"):
        asyncio.run(_run(transport, provider=provider))
    assert provider.phase == "FAILED"


def test_missing_static_safety_counter_cannot_finalize() -> None:
    binding = _binding(warmup=())
    scored = (_server_request("score-0", inputs=(1,), outputs=(2, 3)),)

    def terminal_mutator(value: dict[str, object]) -> None:
        performance = value["performance_counters"]
        assert isinstance(performance, dict)
        performance["fallbacks"] = None
        _rehash_terminal(value)

    transport = FakeAdminTransport(
        binding=binding,
        warmup=(),
        scored=scored,
        terminal_mutator=terminal_mutator,
    )
    provider = NativeTerminalProvider(transport)

    with pytest.raises(ValueError, match="fallbacks"):
        asyncio.run(_run(transport, provider=provider))
    assert provider.phase == "FAILED"


@pytest.mark.parametrize("mismatch", ["live_request_slot", "peak_hbm"])
def test_terminal_final_state_must_close_request_pool_and_peak_hbm(
    mismatch: str,
) -> None:
    binding = _binding(warmup=())
    scored = (_server_request("score-0", inputs=(1,), outputs=(2, 3)),)

    def terminal_mutator(value: dict[str, object]) -> None:
        final_state = value["final_state"]
        assert isinstance(final_state, dict)
        if mismatch == "live_request_slot":
            final_state["request_pool_active_slots"] = 1
        else:
            final_state["allocator_peak_hbm_bytes"] = 97
        _rehash_terminal(value)

    transport = FakeAdminTransport(
        binding=binding,
        warmup=(),
        scored=scored,
        terminal_mutator=terminal_mutator,
    )
    provider = NativeTerminalProvider(transport)

    with pytest.raises(RuntimeError, match="drained|peak HBM"):
        asyncio.run(_run(transport, provider=provider))
    assert provider.phase == "FAILED"


@pytest.mark.parametrize("field", ["resident_bytes", "optimizer_bytes", "peak_bytes"])
def test_adaptation_memory_ledger_must_close_before_finalize(field: str) -> None:
    binding = _binding(method="tts", warmup=())
    scored = (_server_request("score-0", inputs=(1,), outputs=(2, 3)),)

    def terminal_mutator(value: dict[str, object]) -> None:
        performance = value["performance_counters"]
        assert isinstance(performance, dict)
        ledger = performance["adaptation_memory_ledger"]
        assert isinstance(ledger, dict)
        ledger[field] = int(ledger[field]) + 1
        _rehash_terminal(value)

    transport = FakeAdminTransport(
        binding=binding,
        warmup=(),
        scored=scored,
        terminal_mutator=terminal_mutator,
    )
    provider = NativeTerminalProvider(transport)

    with pytest.raises(RuntimeError, match="memory ledger"):
        asyncio.run(_run(transport, provider=provider))
    assert provider.phase == "FAILED"


def test_forged_transport_attributes_do_not_replace_capability_get() -> None:
    binding = _binding(warmup=())
    scored = (_server_request("score-0", inputs=(1,), outputs=(2, 3)),)

    def forge(value: dict[str, object]) -> None:
        value["hook"] = "forged.hook"

    transport = FakeAdminTransport(
        binding=binding,
        warmup=(),
        scored=scored,
        capability_mutator=forge,
    )
    provider = NativeTerminalProvider(transport)

    with pytest.raises(ValueError, match="schema/hook"):
        asyncio.run(provider.begin(binding))
    assert transport.get_paths == [CAPABILITY_PATH]
    assert transport.posts == []
    assert provider.phase == "FAILED"


def test_warmup_token_digest_is_verified_before_scored_reset() -> None:
    binding = _binding()
    real_warmup = (_server_request("warm-0", inputs=(1,), outputs=(2,)),)
    caller_warmup = (_server_request("warm-0", inputs=(1,), outputs=(99,)),)
    scored = (_server_request("score-0", inputs=(3,), outputs=(4, 5)),)
    transport = FakeAdminTransport(
        binding=binding,
        warmup=real_warmup,
        scored=scored,
    )
    provider = NativeTerminalProvider(transport)

    async def run() -> None:
        await provider.begin(binding)
        await provider.reset(warmup_requests=caller_warmup)

    with pytest.raises(ValueError, match="warmup request/token digest"):
        asyncio.run(run())
    assert provider.phase == "FAILED"


def test_release_owned_ed25519_allowlist_is_the_only_positive_trust_path() -> None:
    private_key = Ed25519PrivateKey.generate()
    policy = _release_policy(private_key)
    binding = _binding(warmup=())
    scored = (_server_request("score-0", inputs=(1,), outputs=(2, 3)),)
    transport = FakeAdminTransport(
        binding=binding,
        warmup=(),
        scored=scored,
        attester_id="prod-hsm-1",
        trust_domain="hardware",
        signing_key=private_key,
    )
    provider = NativeTerminalProvider(
        transport,
        trusted_attester_policy=policy,
    )

    _, _, _, terminal = asyncio.run(_run(transport, provider=provider))

    assert terminal.trusted_attestation is True
    assert terminal.attestation.key_id == "release-key-1"
    assert terminal.attestation.public_key_sha256 == policy.trusted_attesters[0][2]
    assert terminal.trusted_attester_policy_sha256 == policy.sha256


def test_caller_release_policy_cannot_unlock_speculative_preflight() -> None:
    private_key = Ed25519PrivateKey.generate()
    policy = _release_policy(private_key)
    binding = _binding(method="l0", warmup=())
    scored = (_server_request("score-0", inputs=(1,), outputs=(2, 3)),)
    transport = FakeAdminTransport(binding=binding, warmup=(), scored=scored)
    provider = NativeTerminalProvider(
        transport,
        trusted_attester_policy=policy,
    )
    plan = SimpleNamespace(
        runtime_plan=SimpleNamespace(
            rank_configs=(SimpleNamespace(method="l0"),),
        ),
        trusted_attester_policy=policy,
    )

    preflight = native_evidence_preflight(plan, provider)

    assert preflight.status == "BLOCKED"
    assert preflight.reason_code == TRUSTED_NATIVE_ATTESTER_UNAVAILABLE_REASON
    assert preflight.missing_hook is None


@pytest.mark.parametrize(
    ("attester_id", "trust_domain"),
    [
        ("cpu-fixture-1", "hardware"),
        ("test-key-1", "test"),
    ],
)
def test_valid_but_nonrelease_signer_identity_remains_untrusted(
    attester_id: str,
    trust_domain: str,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    policy = _release_policy(private_key)
    binding = _binding(warmup=())
    scored = (_server_request("score-0", inputs=(1,), outputs=(2, 3)),)
    transport = FakeAdminTransport(
        binding=binding,
        warmup=(),
        scored=scored,
        attester_id=attester_id,
        trust_domain=trust_domain,
        signing_key=private_key,
    )
    provider = NativeTerminalProvider(
        transport,
        trusted_attester_policy=policy,
    )
    _, _, _, terminal = asyncio.run(_run(transport, provider=provider))

    assert terminal.trusted_attestation is False
    assert terminal.attestation.key_id is None
    assert terminal.attestation.public_key_sha256 is None


def test_caller_verifier_keyword_cannot_create_a_trust_root() -> None:
    binding = _binding(warmup=())
    scored = (_server_request("score-0", inputs=(1,), outputs=(2, 3)),)
    transport = FakeAdminTransport(binding=binding, warmup=(), scored=scored)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        NativeTerminalProvider(
            transport,
            trusted_hardware_verifiers={"prod-hsm-1": lambda *_: True},
        )


def test_artifact_roundtrip_reverifies_identity_policy_and_signature() -> None:
    private_key = Ed25519PrivateKey.generate()
    policy = _release_policy(private_key)
    binding = _binding()
    warmup = (_server_request("warm-0", inputs=(1,), outputs=(2,)),)
    scored = (_server_request("score-0", inputs=(3, 4), outputs=(5, 6)),)
    transport = FakeAdminTransport(
        binding=binding,
        warmup=warmup,
        scored=scored,
        attester_id="prod-hsm-1",
        trust_domain="hardware",
        signing_key=private_key,
    )
    provider = NativeTerminalProvider(
        transport,
        trusted_attester_policy=policy,
    )
    _, _, _, terminal = asyncio.run(_run(transport, provider=provider))
    artifact = terminal.to_artifact(warmup_requests=warmup)
    persisted = json.loads(canonical_json_bytes(artifact))

    replayed = validate_native_terminal_artifact(
        persisted,
        trusted_attester_policy=policy,
        expected_binding=binding,
        expected_warmup_requests=warmup,
        expected_scored_requests=scored,
    )

    assert replayed.terminal_sha256 == terminal.terminal_sha256
    assert replayed.trusted_attestation is True

    tampered = copy.deepcopy(persisted)
    terminal_row = tampered["terminal"]
    assert isinstance(terminal_row, dict)
    attestation = terminal_row["attestation"]
    assert isinstance(attestation, dict)
    signature = bytearray.fromhex(str(attestation["signature_hex"]))
    signature[0] ^= 1
    attestation["signature_hex"] = signature.hex()
    with pytest.raises(ValueError, match="Ed25519 signature is invalid"):
        validate_native_terminal_artifact(
            tampered,
            trusted_attester_policy=policy,
            expected_binding=binding,
            expected_warmup_requests=warmup,
            expected_scored_requests=scored,
        )

    another_policy = _release_policy(Ed25519PrivateKey.generate())
    with pytest.raises(ValueError, match="another release policy"):
        validate_native_terminal_artifact(
            persisted,
            trusted_attester_policy=another_policy,
        )


def test_writer_exclusively_persists_the_canonical_native_bundle(
    tmp_path: Path,
) -> None:
    binding = _binding(method="target_only", warmup=())
    scored = (_server_request("score-0", inputs=(1,), outputs=(2, 3)),)
    transport = FakeAdminTransport(
        binding=binding,
        warmup=(),
        scored=scored,
    )
    _, _, _, terminal = asyncio.run(_run(transport))
    writer = EvidenceWriter(tmp_path, run_id=binding.run_id, rank=0)

    artifact_binding = writer.persist_native_terminal_artifact(
        terminal.to_artifact(warmup_requests=())
    )

    artifact_path = tmp_path / str(artifact_binding["path"])
    body = artifact_path.read_bytes()
    assert artifact_path.is_file() and not artifact_path.is_symlink()
    assert artifact_binding["size"] == len(body)
    assert artifact_binding["raw_sha256"] == hashlib.sha256(body).hexdigest()
    assert artifact_binding["terminal_sha256"] == terminal.terminal_sha256
    assert artifact_binding["trusted_attester_policy_sha256"] == (
        terminal.trusted_attester_policy_sha256
    )
    with pytest.raises(RuntimeError, match="already persisted"):
        writer.persist_native_terminal_artifact(
            terminal.to_artifact(warmup_requests=())
        )
    writer.abort(reason="persistence-only fixture")


def test_final_hook_patch_and_tree_are_exactly_pinned() -> None:
    assert NATIVE_TERMINAL_EVIDENCE_HOOK == (
        "sglang.schema_v3.content_bound_terminal_speculative_evidence.v1"
    )
    assert PINNED_SGLANG_PATCH_SHA256 == (
        "907c8ecc8fdf970a585c359d902777cca3bd08b11ae2342a68cf33016f5272f4"
    )
    assert PINNED_SGLANG_TREE == "c6070bbf97711a01dc9ab01a0e9b3ee3c2d48cb4"
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
