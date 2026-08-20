from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import lightcone_spec.experiments as _experiments  # noqa: F401
from lightcone_spec.orchestration import formal_terminal_result

_RESET_FIXTURE_PATH = Path(__file__).with_name(
    "test_native_terminal_request_reset_v2.py"
)
_RESET_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_lightcone_test_native_terminal_request_reset_v2_for_result",
    _RESET_FIXTURE_PATH,
)
assert _RESET_FIXTURE_SPEC is not None and _RESET_FIXTURE_SPEC.loader is not None
_RESET_FIXTURE_MODULE = importlib.util.module_from_spec(_RESET_FIXTURE_SPEC)
_RESET_FIXTURE_SPEC.loader.exec_module(_RESET_FIXTURE_MODULE)


def _current_native_state():
    evidence, _warmup, scored = _RESET_FIXTURE_MODULE.validated_zero_row_terminal()
    terminal = evidence.to_dict()
    resets = terminal["request_source_point_resets"]
    assert isinstance(resets, dict)
    adaptation = {
        "schema_version": 3,
        "adaptation_config_sha256": "a" * 64,
        "optimizer_schedule": {},
        "schedule_total_published_updates": 0,
        "extra_logical_delay": 0,
        "adaptation_microbatch_size": 1,
        "adaptation_publication_coalescing": 1,
        "adaptation_stream_priority": 0,
        "teacher_row_policy": "request_local",
        "reset_scope": "request",
        "request_admission_policy": "serialized_native_scheduler_v1",
        "request_source_point_reset_protocol_sha256": resets["protocol_sha256"],
        "quota_shadow_protocol_sha256": "b" * 64,
        "tts_fixed_boundary_protocol_sha256": "c" * 64,
        "enabled": True,
        "disabled_reason": None,
        "cohort_sha256": "d" * 64,
        "cohort_epoch": 0,
        "request_epoch": 1,
        "active_request_id": None,
        "request_reset_in_progress": False,
        "request_reset_failed": False,
        "request_evidence_archive_sha256": resets["final_archive_sha256"],
        "slot_generation": 1,
        "active_version": 0,
        "round": 0,
        "resident_bytes": 176,
        "peak_bytes": 320,
        "reserved_hbm_budget_bytes": 4096,
        "peak_hbm_bytes": 3072,
        "optimizer_bytes": 128,
        "trainable_parameters": 4,
        "memory_ledger": {},
        "parameter_layout_sha256": "e" * 64,
        "runtime_authority": None,
        "tp2_last_publication_receipt": None,
        "dp2_replica_state": None,
        "batch_fill": 1.0,
        "max_batch_size": 1,
        "queue_occupancy": 0.0,
        "max_queue_occupancy": 0,
        "graph_replay_hit_rate": 1.0,
        "native_commit_event_protocol_sha256": "f" * 64,
        "hot_path_blocking_d2h_count": 0,
        "hot_path_host_synchronize_count": 0,
        "exposed_update_ms": 0.0,
        "main_side_overlap_ratio": 0.0,
        "counters": {},
        "timings_ms": {},
        "updates": [],
        "teacher_row_acquisitions": [],
        "rounds": [],
        "kv_segments": {},
        "request_source_point_reset_receipts": resets["receipts"],
        "fixed_inference_address_sha256": "1" * 64,
        "fixed_staging_address_sha256": "2" * 64,
        "online_spec_state": None,
    }
    native_state = {
        "scheduler": terminal["final_state"],
        "round_rows": terminal["request_round_rows"]["rounds"],
        "update_rows": terminal["update_rows"],
        "performance_counters": terminal["performance_counters"],
        "historical_kv_source_versions": terminal["historical_kv_source_versions"],
        "request_source_point_resets": resets,
        "adaptation": adaptation,
    }
    return evidence, scored, native_state


def _current_warmup_gang():
    evidence, scored, native_state = _current_native_state()
    request = scored[0]
    cohort_sha256 = "8" * 64
    schedule_rows = (
        SimpleNamespace(
            phase="warmup",
            routed_dp_rank=0,
            request=SimpleNamespace(
                request_id=request.request_id,
                cohort_sha256=cohort_sha256,
                input_token_ids=request.input_token_ids,
            ),
        ),
    )
    plan = SimpleNamespace(
        topology_mode="tp2_dp1",
        gpu_uuids=("GPU-0", "GPU-1"),
        native_terminal_binding=evidence.binding,
        method=evidence.binding.method,
    )
    full_schedule = {
        "warmup": [
            {
                "request_id": request.request_id,
                "cohort_sha256": cohort_sha256,
                "routed_dp_rank": 0,
            }
        ],
        "scored": [],
    }
    local_routes = full_schedule["warmup"]
    sticky_routes = [(cohort_sha256, 0)]
    events_sha256 = "9" * 64
    native_by_id = {
        request.request_id: {
            "request_id": request.request_id,
            "input_token_ids": list(request.input_token_ids),
            "output_token_ids": list(request.output_token_ids or ()),
            "terminal_status": request.terminal_status,
            "terminal_reason": request.terminal_reason,
            "submitted_to_server": True,
        }
    }
    pointer_by_id = {request.request_id: ("a" * 64, events_sha256)}
    request_terminal = {
        "request_id": request.request_id,
        "input_token_ids": list(request.input_token_ids),
        "output_token_ids": list(request.output_token_ids or ()),
        "native_itl_semantics": "scheduler_committed_token_at_result_processor_v1",
        "native_itl_event_count": len(request.output_token_ids or ()),
        "native_itl_events_sha256": events_sha256,
        "terminal_status": request.terminal_status,
        "terminal_reason": request.terminal_reason,
    }
    identity = formal_terminal_result._current_gang_binding_identity(evidence.binding)
    rank_terminals = []
    for rank, gpu_uuid in enumerate(plan.gpu_uuids):
        rank_value = {
            "schema_version": 2,
            "kind": "sglang_formal_gang_rank_terminal",
            "hook": "sglang.lightcone_formal_gang_serving.v1",
            "protocol_sha256": formal_terminal_result.FORMAL_GANG_SERVING_PROTOCOL_SHA256,
            "topology": plan.topology_mode,
            "rank": rank,
            "world_size": 2,
            "gpu_uuid": gpu_uuid,
            "execution_plan_sha256": evidence.binding.execution_plan_sha256,
            "rank_config_sha256": evidence.binding.rank_config_sha256,
            "run_nonce_sha256": evidence.binding.run_nonce_sha256,
            "method": evidence.binding.method,
            "reset_scope": identity[0],
            "request_admission_policy": identity[1],
            "request_source_point_reset_protocol_sha256": identity[2],
            "runtime_trust_mode": identity[3],
            "formal_measurement": identity[4],
            "phase": "warmup",
            "full_schedule_sha256": formal_terminal_result.content_sha256(
                full_schedule
            ),
            "local_request_routes_sha256": formal_terminal_result.content_sha256(
                local_routes
            ),
            "sticky_cohort_routes_sha256": formal_terminal_result.content_sha256(
                sticky_routes
            ),
            "expected_request_ids_sha256": formal_terminal_result.content_sha256(
                [request.request_id]
            ),
            "request_terminals": [copy.deepcopy(request_terminal)],
            "request_terminal_sha256s": [
                formal_terminal_result.content_sha256(request_terminal)
            ],
            "native_state": copy.deepcopy(native_state),
            "native_state_sha256": formal_terminal_result.content_sha256(native_state),
            "client_lifecycle_sha256": None,
            "non_submitted_request_ids_sha256": None,
            "status": "COMPLETE",
            "reason_code": None,
        }
        rank_terminals.append(
            {
                **rank_value,
                "terminal_sha256": formal_terminal_result.content_sha256(rank_value),
            }
        )
    aggregate_value = {
        "schema_version": 2,
        "kind": "sglang_formal_gang_all_rank_terminal",
        "hook": "sglang.lightcone_formal_gang_serving.v1",
        "protocol_sha256": formal_terminal_result.FORMAL_GANG_SERVING_PROTOCOL_SHA256,
        "topology": plan.topology_mode,
        "world_size": 2,
        "action": "formal_gang_reset",
        "reset_scope": identity[0],
        "request_admission_policy": identity[1],
        "request_source_point_reset_protocol_sha256": identity[2],
        "runtime_trust_mode": identity[3],
        "formal_measurement": identity[4],
        "decision": "COMMITTED",
        "published_ranks": [0, 1],
        "reason_code": None,
        "cross_replica_gradient_collective": False,
        "rank_terminals": rank_terminals,
        "rank_reset_sha256s": ["b" * 64, "c" * 64],
        "rank_terminal_sha256s": [row["terminal_sha256"] for row in rank_terminals],
    }
    gang = {
        **aggregate_value,
        "aggregate_sha256": formal_terminal_result.content_sha256(aggregate_value),
    }
    return gang, plan, schedule_rows, native_by_id, pointer_by_id


def _resign_current_warmup_gang(gang: dict[str, object], rank: int) -> None:
    rank_row = gang["rank_terminals"][rank]
    rank_row["native_state_sha256"] = formal_terminal_result.content_sha256(
        rank_row["native_state"]
    )
    unsigned_rank = dict(rank_row)
    unsigned_rank.pop("terminal_sha256")
    rank_row["terminal_sha256"] = formal_terminal_result.content_sha256(unsigned_rank)
    gang["rank_terminal_sha256s"][rank] = rank_row["terminal_sha256"]
    unsigned_gang = dict(gang)
    unsigned_gang.pop("aggregate_sha256")
    gang["aggregate_sha256"] = formal_terminal_result.content_sha256(unsigned_gang)


def test_formal_gang_native_state_replays_current_request_reset_archive() -> None:
    evidence, scored, native_state = _current_native_state()
    resets = formal_terminal_result._validate_current_gang_native_state(
        native_state,
        binding=evidence.binding,
        requests=scored,
    )
    assert tuple(row.request_id for row in resets.receipts) == ("score-0",)


def test_formal_gang_native_state_rejects_reset_envelope_diagnostics_drift() -> None:
    evidence, scored, native_state = _current_native_state()
    changed = copy.deepcopy(native_state)
    changed["adaptation"]["request_source_point_reset_receipts"] = []
    with pytest.raises(RuntimeError, match="adaptation/reset identity"):
        formal_terminal_result._validate_current_gang_native_state(
            changed,
            binding=evidence.binding,
            requests=scored,
        )


def test_formal_gang_native_state_rejects_missing_submitted_reset_receipt() -> None:
    evidence, scored, native_state = _current_native_state()
    changed = copy.deepcopy(native_state)
    changed["request_source_point_resets"]["receipts"] = []
    with pytest.raises(RuntimeError, match="receipt coverage"):
        formal_terminal_result._validate_current_gang_native_state(
            changed,
            binding=evidence.binding,
            requests=scored,
        )


def test_formal_gang_warmup_reset_deep_replays_all_rank_native_state() -> None:
    gang, plan, schedule_rows, native_by_id, pointer_by_id = _current_warmup_gang()
    assert formal_terminal_result._validate_current_gang_warmup_reset(
        gang,
        plan=plan,
        schedule_rows=schedule_rows,
        native_by_id=native_by_id,
        pointer_by_id=pointer_by_id,
    ) == formal_terminal_result.content_sha256(None)


def test_formal_gang_warmup_reset_rejects_self_signed_missing_receipt() -> None:
    gang, plan, schedule_rows, native_by_id, pointer_by_id = _current_warmup_gang()
    for rank in range(2):
        state = gang["rank_terminals"][rank]["native_state"]
        state["request_source_point_resets"]["receipts"] = []
        state["request_source_point_resets"]["final_archive_sha256"] = "0" * 64
        state["adaptation"]["request_source_point_reset_receipts"] = []
        state["adaptation"]["request_evidence_archive_sha256"] = "0" * 64
        state["adaptation"]["request_epoch"] = 0
        state["scheduler"]["adapter_request_epoch"] = 0
        _resign_current_warmup_gang(gang, rank)
    with pytest.raises(RuntimeError, match="receipt coverage"):
        formal_terminal_result._validate_current_gang_warmup_reset(
            gang,
            plan=plan,
            schedule_rows=schedule_rows,
            native_by_id=native_by_id,
            pointer_by_id=pointer_by_id,
        )


def test_formal_gang_warmup_tp2_rejects_locally_valid_logical_reset_drift() -> None:
    gang, plan, schedule_rows, native_by_id, pointer_by_id = _current_warmup_gang()
    state = gang["rank_terminals"][1]["native_state"]
    receipt = state["request_source_point_resets"]["receipts"][0]
    receipt["source_point_identity_sha256"] = "4" * 64
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = formal_terminal_result.content_sha256(unsigned_receipt)
    state["adaptation"]["request_source_point_reset_receipts"] = copy.deepcopy(
        state["request_source_point_resets"]["receipts"]
    )
    _resign_current_warmup_gang(gang, 1)
    with pytest.raises(RuntimeError, match="TP2 logical resets differ"):
        formal_terminal_result._validate_current_gang_warmup_reset(
            gang,
            plan=plan,
            schedule_rows=schedule_rows,
            native_by_id=native_by_id,
            pointer_by_id=pointer_by_id,
        )


def test_trusted_runtime_authority_accepts_distinct_role_lineages() -> None:
    evidence, _warmup, _scored = _RESET_FIXTURE_MODULE.validated_zero_row_terminal(
        runtime_trust_mode="trusted_single_operator_empirical_no_signature",
        formal_measurement=False,
    )
    authority = {
        "proof_sha256": "1" * 64,
        "source_identity_sha256": "2" * 64,
        "topology_mode": "tp2_dp1",
        "topology_sha256": "3" * 64,
        "gpu_uuids": ["GPU-0", "GPU-1"],
        "backend_capabilities": {"native_adaptation": True},
        "trust_mode": "trusted_single_operator_empirical_no_signature",
        "formal_measurement": False,
        "qualification_only": False,
        "trusted_authority_kind": "preflight_qualification",
        "trusted_authority_source_sha256": "4" * 64,
        "trusted_consumer_identity_sha256": "5" * 64,
        "trusted_evidence_sha256s": ["6" * 64, "7" * 64],
        "trusted_role_lineages": [
            {
                "role": "distributed",
                "source_suite_id": "distributed-suite",
                "source_capability_sha256": "8" * 64,
                "role_source_identity_sha256": "9" * 64,
            },
            {
                "role": "native",
                "source_suite_id": "native-suite",
                "source_capability_sha256": "a" * 64,
                "role_source_identity_sha256": "9" * 64,
            },
        ],
    }

    assert (
        formal_terminal_result._validate_current_runtime_authority(
            authority,
            binding=evidence.binding,
        )
        == authority
    )

    missing_role = copy.deepcopy(authority)
    missing_role["trusted_role_lineages"] = missing_role["trusted_role_lineages"][1:]
    with pytest.raises(RuntimeError, match="role lineages"):
        formal_terminal_result._validate_current_runtime_authority(
            missing_role,
            binding=evidence.binding,
        )
