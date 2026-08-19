from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightcone_spec.orchestration.formal_terminal_shards import (
    SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND,
    SHARDED_UNSIGNED_NATIVE_ITL_BUNDLE_KIND,
    publish_scalable_client_request_lifecycle,
    publish_scalable_formal_gang_itl_bundle,
    publish_scalable_formal_gang_request_terminal,
    publish_scalable_formal_gang_terminal,
    publish_scalable_native_terminal_artifact,
    publish_scalable_unsigned_native_itl_bundle,
    reopen_scalable_client_request_lifecycle,
    reopen_scalable_formal_gang_itl_bundle,
    reopen_scalable_formal_gang_request_terminal,
    reopen_scalable_formal_gang_terminal,
    reopen_scalable_native_terminal_artifact,
    reopen_scalable_unsigned_native_itl_bundle,
)


def _large_terminal_artifact(request_count: int = 10_000) -> dict[str, object]:
    request_ids = [f"request-{ordinal:05d}" for ordinal in range(request_count)]
    expectations = [
        {
            "request_id": request_id,
            "input_token_ids": [ordinal, ordinal + 1, ordinal + 2],
            "output_token_ids": [ordinal + 3, ordinal + 4],
            "terminal_status": "completed",
            "terminal_reason": "finished",
            "submitted_to_server": True,
        }
        for ordinal, request_id in enumerate(request_ids)
    ]
    request_rows = [
        {
            "request_id": request_id,
            "terminal_source": "server",
            "input_tokens": 3,
            "input_token_ids_sha256": "a" * 64,
            "output_tokens": 2,
            "ordered_output_token_ids": [ordinal + 3, ordinal + 4],
            "output_token_ids_sha256": "b" * 64,
            "terminal_status": "completed",
            "terminal_reason": "finished",
            "request_sha256": "c" * 64,
        }
        for ordinal, request_id in enumerate(request_ids)
    ]
    terminal = {
        "schema_version": 1,
        "hook": "hook",
        "run_id": "run",
        "run_nonce_sha256": "0" * 64,
        "execution_plan_sha256": "1" * 64,
        "rank_config_sha256": "2" * 64,
        "server_process_id": 1,
        "server_process_started_ns": 1,
        "attempt_id": "attempt-0",
        "session_id": "session",
        "session_epoch": 1,
        "previous_run_id": None,
        "challenge_nonce_sha256": "3" * 64,
        "method": "target_only",
        "expected_request_ids": request_ids,
        "reset_receipt_sha256": "4" * 64,
        "request_round_rows": {"requests": request_rows, "rounds": []},
        "update_rows": [],
        "performance_counters": {"target_calls": request_count * 2},
        "historical_kv_source_versions": {},
        "final_state": {"scheduler_idle": True},
        "completion_marker": "TERMINAL_COMPLETE",
        "terminal_sha256": "5" * 64,
        "attestation": {"status": "UNSIGNED"},
    }
    return {
        "schema_version": 1,
        "artifact_kind": "native_terminal_evidence_bundle_v1",
        "run_id": "run",
        "rank": 0,
        "trusted_attester_policy_sha256": "6" * 64,
        "begin_sha256": "7" * 64,
        "reset_sha256": "4" * 64,
        "terminal_sha256": "5" * 64,
        "binding": {
            "run_id": "run",
            "warmup_request_ids": [],
            "scored_request_ids": request_ids,
        },
        "warmup_requests": [],
        "scored_requests": expectations,
        "begin": {"kind": "begin"},
        "reset": {"kind": "reset"},
        "terminal": terminal,
    }


def _large_itl_bundle(event_count: int = 40_928) -> dict[str, object]:
    events = [
        {"token_index": index, "token_id": index % 32_000, "observed_ns": index + 2}
        for index in range(event_count)
    ]
    return {
        "schema_version": 1,
        "kind": "unsigned_native_itl_result_pointer_bundle",
        "run_binding_sha256": "0" * 64,
        "terminal_artifact_raw_sha256": "1" * 64,
        "terminal_artifact_semantic_sha256": "2" * 64,
        "scored_request_inputs_sha256": "3" * 64,
        "native_result_pointers": [
            {
                "schema_version": 1,
                "kind": "sglang_native_itl_result_pointer",
                "hook": "hook",
                "semantics": "native",
                "release_status": "pending",
                "request_id": "request-00000",
                "request_started_ns": 1,
                "request_terminal_ns": event_count + 2,
                "terminal_status": "completed",
                "terminal_reason": "finished",
                "events": events,
                "result_pointer_sha256": "4" * 64,
            }
        ],
    }


def _assert_bounded_json_tree(root: Path) -> None:
    paths = tuple(root.rglob("*.json"))
    assert paths
    assert max(path.stat().st_size for path in paths) < 2_000_000


def test_small_terminal_container_preserves_legacy_schema_and_bytes(
    tmp_path: Path,
) -> None:
    artifact = _large_terminal_artifact(request_count=1)
    output = tmp_path / "terminal.json"
    binding = publish_scalable_native_terminal_artifact(
        output_path=output,
        legacy_artifact=artifact,
    )
    assert binding.reopen() == artifact
    assert reopen_scalable_native_terminal_artifact(binding.reopen()) == artifact


def test_ten_thousand_request_terminal_is_sharded_and_deep_reopened(
    tmp_path: Path,
) -> None:
    artifact = _large_terminal_artifact()
    output = tmp_path / "terminal.json"
    binding = publish_scalable_native_terminal_artifact(
        output_path=output,
        legacy_artifact=artifact,
    )
    header = binding.reopen()
    assert header["artifact_kind"] == SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND
    assert reopen_scalable_native_terminal_artifact(header) == artifact
    _assert_bounded_json_tree(tmp_path)

    first_index = header["sequence_indexes"]["scored_requests"]
    first_shard = Path(first_index["absolute_path"]).parent / "shard-000000.json"
    mutated = json.loads(first_shard.read_text())
    mutated["rows"][0]["terminal_reason"] = "changed"
    first_shard.write_text(
        json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises((RuntimeError, ValueError), match="changed|binding|shard"):
        reopen_scalable_native_terminal_artifact(header)


def test_one_long_request_itl_events_use_nested_shards(tmp_path: Path) -> None:
    bundle = _large_itl_bundle()
    output = tmp_path / "itl.json"
    binding = publish_scalable_unsigned_native_itl_bundle(
        output_path=output,
        legacy_bundle=bundle,
    )
    header = binding.reopen()
    assert header["kind"] == SHARDED_UNSIGNED_NATIVE_ITL_BUNDLE_KIND
    assert reopen_scalable_unsigned_native_itl_bundle(header) == bundle
    _assert_bounded_json_tree(tmp_path)


def test_ten_thousand_short_itl_pointers_do_not_create_per_request_directories(
    tmp_path: Path,
) -> None:
    prototype = _large_itl_bundle(event_count=2)
    pointers = []
    for ordinal in range(10_000):
        pointer = dict(prototype["native_result_pointers"][0])
        pointer["request_id"] = f"request-{ordinal:05d}"
        pointers.append(pointer)
    bundle = {**prototype, "native_result_pointers": pointers}
    output = tmp_path / "itl.json"
    binding = publish_scalable_unsigned_native_itl_bundle(
        output_path=output,
        legacy_bundle=bundle,
    )
    assert reopen_scalable_unsigned_native_itl_bundle(binding.reopen()) == bundle
    event_directories = tuple(tmp_path.rglob("events-*"))
    assert event_directories == ()
    assert len(tuple(tmp_path.rglob("*.json"))) < 100
    _assert_bounded_json_tree(tmp_path)


def test_ten_thousand_client_lifecycles_roundtrip_no_replace_and_detect_tamper(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "request_id": f"request-{ordinal:05d}",
            "phase": "warmup" if ordinal < 8 else "scored",
            "scheduled_arrival_us": 0,
            "offered": True,
            "offered_at_us": ordinal,
            "admitted_at_us": ordinal,
            "effective_deadline_us": ordinal + 1_000_000,
            "cancellation_at_us": None,
            "terminal_at_us": ordinal + 1,
            "outcome_status": "completed",
            "outcome_code": "completed",
            "submitted_to_server": True,
            "native_terminal_status": "completed",
            "native_result_pointer_sha256": f"{ordinal % 16:x}" * 64,
        }
        for ordinal in range(10_000)
    ]
    output = (tmp_path / "client-lifecycle.json").resolve()
    binding = publish_scalable_client_request_lifecycle(
        output_path=output,
        run_binding_sha256="a" * 64,
        execution_policy_sha256="b" * 64,
        rows=rows,
    )
    assert (
        reopen_scalable_client_request_lifecycle(
            binding,
            expected_run_binding_sha256="a" * 64,
            expected_execution_policy_sha256="b" * 64,
        )
        == rows
    )
    _assert_bounded_json_tree(tmp_path)

    with pytest.raises(FileExistsError):
        publish_scalable_client_request_lifecycle(
            output_path=output,
            run_binding_sha256="a" * 64,
            execution_policy_sha256="b" * 64,
            rows=rows,
        )

    header = binding.reopen()
    index_path = Path(header["rows_index"]["absolute_path"])
    first_shard = index_path.parent / "shard-000000.json"
    mutated = json.loads(first_shard.read_text())
    mutated["rows"][0]["outcome_code"] = "tampered"
    first_shard.write_text(
        json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises((RuntimeError, ValueError), match="changed|binding|shard"):
        reopen_scalable_client_request_lifecycle(binding)


def test_ten_thousand_request_gang_terminal_pointer_and_rank_rows_roundtrip(
    tmp_path: Path,
) -> None:
    request_count = 10_000
    request_rows = [
        {
            "request_id": f"request-{ordinal:05d}",
            "input_token_ids": [ordinal, ordinal + 1],
            "output_token_ids": [ordinal + 2, ordinal + 3],
            "terminal_status": "completed",
            "terminal_reason": "finished",
        }
        for ordinal in range(request_count)
    ]
    request_terminal = {
        "schema_version": 1,
        "kind": "unsigned_formal_gang_request_terminal",
        "protocol_sha256": "0" * 64,
        "formal_execution_authorized": False,
        "plan_sha256": "1" * 64,
        "formal_launch_admission": {"path": "admission"},
        "formal_launch_consumption": {"path": "consumption"},
        "budget_consumption": {"path": "budget"},
        "capability_sha256": "2" * 64,
        "begin_sha256": "3" * 64,
        "reset_sha256": "4" * 64,
        "finalize_sha256": "5" * 64,
        "warmup_requests": request_rows[:8],
        "scored_requests": request_rows,
    }
    request_output = tmp_path / "gang-request-terminal.json"
    request_binding = publish_scalable_formal_gang_request_terminal(
        output_path=request_output,
        legacy_terminal=request_terminal,
    )
    assert (
        reopen_scalable_formal_gang_request_terminal(request_binding.reopen())
        == request_terminal
    )

    prototype = _large_itl_bundle(event_count=2)["native_result_pointers"][0]
    pointers = []
    for ordinal in range(request_count):
        pointer = dict(prototype)
        pointer["request_id"] = f"request-{ordinal:05d}"
        pointers.append(pointer)
    gang_itl = {
        "schema_version": 1,
        "kind": "unsigned_formal_gang_native_itl_pointer_bundle",
        "protocol_sha256": "0" * 64,
        "formal_execution_authorized": False,
        "plan_sha256": "1" * 64,
        "formal_launch_admission": {"path": "admission"},
        "formal_launch_consumption": {"path": "consumption"},
        "budget_consumption": {"path": "budget"},
        "warmup_pointers": pointers[:8],
        "scored_pointers": pointers,
    }
    itl_output = tmp_path / "gang-itl.json"
    itl_binding = publish_scalable_formal_gang_itl_bundle(
        output_path=itl_output,
        legacy_bundle=gang_itl,
    )
    assert reopen_scalable_formal_gang_itl_bundle(itl_binding.reopen()) == gang_itl

    rank_terminals = []
    for rank in range(2):
        local = request_rows[rank::2]
        rank_terminals.append(
            {
                "schema_version": 1,
                "kind": "sglang_formal_gang_rank_terminal",
                "rank": rank,
                "request_terminals": local,
                "request_terminal_sha256s": ["6" * 64 for _row in local],
                "native_state": {
                    "scheduler": {"idle": True},
                    "round_rows": [],
                    "update_rows": [],
                    "performance_counters": {"target_calls": len(local) * 2},
                    "historical_kv_source_versions": {},
                    "adaptation": None,
                },
            }
        )
    gang_terminal = {
        "schema_version": 1,
        "kind": "sglang_formal_gang_all_rank_terminal",
        "rank_terminals": rank_terminals,
        "rank_terminal_sha256s": ["7" * 64, "8" * 64],
        "aggregate_sha256": "9" * 64,
    }
    gang_output = tmp_path / "gang-terminal.json"
    gang_binding = publish_scalable_formal_gang_terminal(
        output_path=gang_output,
        legacy_terminal=gang_terminal,
    )
    assert reopen_scalable_formal_gang_terminal(gang_binding.reopen()) == gang_terminal
    assert not tuple(tmp_path.rglob("warmup-events-*"))
    assert not tuple(tmp_path.rglob("scored-events-*"))
    _assert_bounded_json_tree(tmp_path)
