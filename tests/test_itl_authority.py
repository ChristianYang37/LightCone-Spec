from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments import itl_authority
from lightcone_spec.experiments.industrial_analysis import _request_metric
from lightcone_spec.experiments.itl_authority import (
    ITL_COALESCED_CHUNK_UNPROVEN_REASON,
    ITL_CPU_CONTRACT_ONLY_REASON,
    ITL_FIRST_PARTY_RESULT_POINTER_UNAVAILABLE_REASON,
    ITL_RAW_RECEIPT_MISSING_REASON,
    ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256,
    ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON,
    ItlRequestExpectation,
    ItlTimestampAuthorityBlocked,
    ReleaseItlTimestampProducer,
    assess_serving_chunks_for_formal_itl,
    bind_itl_timestamp_authority,
    evaluate_e2_itl_timestamp_activation,
    load_path_bound_itl_timestamp_authority,
    reject_cpu_contract_only_itl_metadata,
    release_e2_itl_timestamp_plan,
    replay_e2_itl_timestamp_plan,
    require_e2_itl_timestamp_prelaunch,
)
from lightcone_spec.experiments.load import TokenChunkTiming
from lightcone_spec.experiments.registry import build_industrial_registry


def _registry_and_cell():
    registry = build_industrial_registry()
    cell = next(
        row
        for row in registry.cells_for("E2")
        if row.identity.method == "tts" and row.identity.optimizer == "adamw"
    )
    return registry, cell


def _producer(mode: str) -> ReleaseItlTimestampProducer:
    hook = {
        "native_per_token_timestamp_hook": (
            "sglang.schema_v3.native_per_token_timestamp.v1"
        ),
        "sse_one_token_per_frame": (
            "sglang.benchmark.serving.raw_sse_frame_observation.v1"
        ),
    }[mode]
    return ReleaseItlTimestampProducer(
        producer_id=f"release-{mode}",
        source_mode=mode,
        hook_id=hook,
        producer_version_sha256="a" * 64,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        clock="monotonic_ns",
        protocol_sha256=ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256,
    )


def _ready_plan(monkeypatch: pytest.MonkeyPatch, mode: str):
    producer = _producer(mode)
    monkeypatch.setattr(
        itl_authority,
        "RELEASE_ITL_TIMESTAMP_PRODUCERS",
        (producer,),
    )
    registry, cell = _registry_and_cell()
    plan = release_e2_itl_timestamp_plan(registry, cell)
    return plan, producer


def _receipt(plan, producer) -> dict[str, object]:
    common = {
        "request_id": "request-1",
        "request_started_ns": 100,
        "request_terminal_ns": 180,
        "output_token_ids": [10, 11, 12],
    }
    if producer.source_mode == "native_per_token_timestamp_hook":
        common["token_events"] = [
            {"token_index": 0, "token_id": 10, "observed_ns": 110},
            {"token_index": 1, "token_id": 11, "observed_ns": 130},
            {"token_index": 2, "token_id": 12, "observed_ns": 170},
        ]
    else:
        common["raw_sse_frames"] = [
            {"frame_index": 0, "new_token_ids": [10], "observed_ns": 110},
            {"frame_index": 1, "new_token_ids": [11], "observed_ns": 130},
            {"frame_index": 2, "new_token_ids": [12], "observed_ns": 170},
        ]
    return {
        "schema_version": 1,
        "kind": "cpu_test_itl_timestamp_raw_receipt",
        "plan_sha256": plan.sha256,
        "producer_id": producer.producer_id,
        "producer_version_sha256": producer.producer_version_sha256,
        "source_mode": producer.source_mode,
        "hook_id": producer.hook_id,
        "clock": "monotonic_ns",
        "complete": True,
        "requests": [common],
    }


def _expected() -> tuple[ItlRequestExpectation, ...]:
    return (
        ItlRequestExpectation(
            request_id="request-1",
            output_token_ids=(10, 11, 12),
            terminal_status="completed",
        ),
    )


def _write(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def test_current_release_blocks_e2_before_raw_path_is_inspected(tmp_path: Path) -> None:
    registry, cell = _registry_and_cell()
    plan = release_e2_itl_timestamp_plan(registry, cell)
    activation = evaluate_e2_itl_timestamp_activation(plan)

    assert activation.status == "BLOCKED"
    assert activation.reason_code == ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON
    assert activation.producer_sha256 is None
    with pytest.raises(ItlTimestampAuthorityBlocked) as error:
        require_e2_itl_timestamp_prelaunch(plan)
    assert error.value.reason == ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON

    missing = tmp_path / "must-not-be-created" / "raw-itl.json"
    with pytest.raises(ItlTimestampAuthorityBlocked) as error:
        bind_itl_timestamp_authority(plan, missing, expected_requests=_expected())
    assert error.value.reason == ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON
    assert not missing.parent.exists()

    with pytest.raises(ItlTimestampAuthorityBlocked) as error:
        load_path_bound_itl_timestamp_authority(
            missing,
            registry=registry,
            cell=cell,
            expected_requests=_expected(),
        )
    assert error.value.reason == ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON
    assert not missing.parent.exists()


def test_receipt_reader_rejects_immediate_parent_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = (tmp_path / "authority-parent").resolve()
    parent.mkdir()
    receipt = _write(parent / "receipt.json", {"receipt": "original"}).resolve()
    moved = (tmp_path / "authority-parent-moved").resolve()
    attacker = (tmp_path / "attacker-parent").resolve()
    attacker.mkdir()
    _write(attacker / "receipt.json", {"receipt": "attacker"})
    original_fstat = itl_authority.os.fstat
    swapped = False

    def swap_after_directory_open(descriptor: int):
        nonlocal swapped
        metadata = original_fstat(descriptor)
        if not swapped and os.path.samestat(metadata, parent.stat()):
            swapped = True
            parent.rename(moved)
            attacker.rename(parent)
        return metadata

    monkeypatch.setattr(itl_authority.os, "fstat", swap_after_directory_open)
    with pytest.raises(ValueError, match="parent changed before read"):
        itl_authority._read_stable_receipt(receipt)

    assert swapped


def test_release_plan_replay_rejects_a_caller_supplied_producer() -> None:
    registry, cell = _registry_and_cell()
    plan = release_e2_itl_timestamp_plan(registry, cell)

    assert replay_e2_itl_timestamp_plan(registry, cell, plan.to_dict()) == plan
    forged = plan.to_dict()
    forged["producer"] = _producer("native_per_token_timestamp_hook").to_dict()
    with pytest.raises(ValueError, match="source-owned replay"):
        replay_e2_itl_timestamp_plan(registry, cell, forged)


def test_producer_allowlist_alone_cannot_mint_formal_itl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = _producer("native_per_token_timestamp_hook")
    monkeypatch.setattr(
        itl_authority,
        "RELEASE_ITL_TIMESTAMP_PRODUCERS",
        (producer,),
    )
    registry, cell = _registry_and_cell()
    plan = release_e2_itl_timestamp_plan(registry, cell)
    activation = evaluate_e2_itl_timestamp_activation(plan)
    assert activation.status == "BLOCKED"
    assert activation.reason_code == ITL_FIRST_PARTY_RESULT_POINTER_UNAVAILABLE_REASON
    missing = tmp_path / "must-not-read" / "forged.json"
    with pytest.raises(ItlTimestampAuthorityBlocked) as error:
        bind_itl_timestamp_authority(plan, missing, expected_requests=_expected())
    assert error.value.reason == ITL_FIRST_PARTY_RESULT_POINTER_UNAVAILABLE_REASON
    assert not missing.parent.exists()


def test_terminal_expectation_has_no_cross_clock_timestamp_fields() -> None:
    expectation = _expected()[0]
    assert expectation.to_dict() == {
        "request_id": "request-1",
        "output_token_ids": [10, 11, 12],
        "terminal_status": "completed",
    }
    forged = expectation.to_dict()
    forged["request_started_ns"] = 100
    forged["request_terminal_ns"] = 180
    with pytest.raises(ValueError, match="fields differ"):
        ItlRequestExpectation.from_dict(forged)


def test_external_itl_schema_versions_reject_boolean_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, cell = _registry_and_cell()
    plan = release_e2_itl_timestamp_plan(registry, cell)
    with pytest.raises(ValueError, match="plan schema is unsupported"):
        replace(plan, schema_version=True)

    ready_plan, producer = _ready_plan(monkeypatch, "native_per_token_timestamp_hook")
    receipt = _receipt(ready_plan, producer)
    receipt["schema_version"] = True
    path = _write(tmp_path / "bool-schema.json", receipt).resolve()
    with pytest.raises(ValueError, match="differs from its release plan"):
        itl_authority._parse_itl_timestamp_receipt_for_cpu_test(
            ready_plan, path, expected_requests=_expected()
        )


def test_cpu_contract_only_events_cannot_enter_formal_itl_or_p99() -> None:
    metadata = {
        "native_token_timestamp_hook": (
            "sglang.schema_v3.native_per_token_timestamp.v1"
        ),
        "native_token_timestamp_semantics": (
            "cpu_committed_token_observed_at_streamer_v1"
        ),
        "native_token_timestamp_release_status": "CPU_CONTRACT_ONLY",
        "native_token_timestamp_events": [
            {"token_index": 0, "token_id": 10, "observed_ns": 100},
            {"token_index": 1, "token_id": 11, "observed_ns": 101},
        ],
    }
    with pytest.raises(ItlTimestampAuthorityBlocked) as error:
        reject_cpu_contract_only_itl_metadata(metadata)
    assert error.value.reason == ITL_CPU_CONTRACT_ONLY_REASON


def test_serving_chunks_never_average_a_coalesced_gap() -> None:
    coalesced = (
        TokenChunkTiming(
            request_id="request-1",
            first_token_index=0,
            token_count=3,
            chunk_observed_at_us=900,
            per_token_observed_at_us=None,
        ),
    )
    assert (
        assess_serving_chunks_for_formal_itl(
            request_id="request-1",
            output_tokens=3,
            chunks=coalesced,
        )
        == ITL_COALESCED_CHUNK_UNPROVEN_REASON
    )

    exact_looking = tuple(
        TokenChunkTiming("request-1", index, 1, observed)
        for index, observed in enumerate((100, 200, 300))
    )
    assert (
        assess_serving_chunks_for_formal_itl(
            request_id="request-1",
            output_tokens=3,
            chunks=exact_looking,
        )
        == ITL_RAW_RECEIPT_MISSING_REASON
    )


def test_native_hook_receipt_preserves_exact_token_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, producer = _ready_plan(monkeypatch, "native_per_token_timestamp_hook")
    path = _write(tmp_path / "native-itl.json", _receipt(plan, producer)).resolve()

    requests = itl_authority._parse_itl_timestamp_receipt_for_cpu_test(
        plan, path, expected_requests=_expected()
    )
    request = requests[0]
    assert request.output_token_ids == (10, 11, 12)
    assert request.token_observed_ns == (110, 130, 170)
    assert request.inter_token_ns == (20, 40)


def test_formal_request_metric_uses_only_authority_token_timestamps() -> None:
    metric = _request_metric(
        {
            "request_id": "request-1",
            "arrival_ns": 100,
            "completed_ns": 100_000_100,
            "first_token_ns": 110,
            "ttft_ms": 0.00001,
            "output_tokens": 3,
            "finished": True,
            "outcome_status": "completed",
            # Deliberately impossible diagnostic summaries must not enter E2.
            "inter_token_ms": [999_999.0],
            "token_timestamps_ns": [110, 111],
        },
        authoritative_token_timestamps_ns=(110, 20_000_110, 60_000_110),
    )

    assert metric.within_request_p99_itl_ms == pytest.approx(39.8)


def test_raw_receipt_must_cover_the_complete_terminal_request_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, producer = _ready_plan(monkeypatch, "native_per_token_timestamp_hook")
    path = _write(tmp_path / "native-itl.json", _receipt(plan, producer)).resolve()
    expected = _expected() + (
        ItlRequestExpectation(
            request_id="request-2",
            output_token_ids=(20, 21),
            terminal_status="completed",
        ),
    )

    with pytest.raises(ValueError, match="omits an expected request"):
        itl_authority._parse_itl_timestamp_receipt_for_cpu_test(
            plan, path, expected_requests=expected
        )


def test_sse_receipt_requires_exactly_one_new_token_per_raw_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, producer = _ready_plan(monkeypatch, "sse_one_token_per_frame")
    receipt = _receipt(plan, producer)
    path = _write(tmp_path / "sse-itl.json", receipt).resolve()
    requests = itl_authority._parse_itl_timestamp_receipt_for_cpu_test(
        plan, path, expected_requests=_expected()
    )
    assert requests[0].inter_token_ns == (20, 40)

    coalesced = deepcopy(receipt)
    request = coalesced["requests"][0]
    request["raw_sse_frames"] = [
        {"frame_index": 0, "new_token_ids": [10, 11], "observed_ns": 130},
        {"frame_index": 1, "new_token_ids": [12], "observed_ns": 170},
    ]
    _write(path, coalesced)
    with pytest.raises(ItlTimestampAuthorityBlocked) as error:
        itl_authority._parse_itl_timestamp_receipt_for_cpu_test(
            plan, path, expected_requests=_expected()
        )
    assert error.value.reason == ITL_COALESCED_CHUNK_UNPROVEN_REASON


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_event", "coverage is incomplete"),
        ("wrong_token", "differ from ordered output tokens"),
        ("equal_timestamp", "strictly increasing"),
        ("outside_lifetime", "outside the request lifetime"),
    ),
)
def test_native_raw_receipt_rejects_incomplete_or_invented_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    plan, producer = _ready_plan(monkeypatch, "native_per_token_timestamp_hook")
    receipt = _receipt(plan, producer)
    request = receipt["requests"][0]
    events = request["token_events"]
    if mutation == "missing_event":
        events.pop()
    elif mutation == "wrong_token":
        events[1]["token_id"] = 99
    elif mutation == "equal_timestamp":
        events[1]["observed_ns"] = 110
    else:
        events[-1]["observed_ns"] = 181
    path = _write(tmp_path / f"bad-{mutation}.json", receipt).resolve()

    with pytest.raises(ValueError, match=message):
        itl_authority._parse_itl_timestamp_receipt_for_cpu_test(
            plan, path, expected_requests=_expected()
        )


def test_cpu_parser_rejects_a_raw_path_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, producer = _ready_plan(monkeypatch, "native_per_token_timestamp_hook")
    receipt = _receipt(plan, producer)
    path = _write(tmp_path / "native-itl.json", receipt).resolve()
    link = tmp_path / "native-itl-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="resolved and non-symlink"):
        itl_authority._parse_itl_timestamp_receipt_for_cpu_test(
            plan, link, expected_requests=_expected()
        )


def test_foreign_cell_and_ambiguous_release_producers_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_industrial_registry()
    foreign = registry.cells_for("E1")[0]
    with pytest.raises(ValueError, match="foreign to the E2 registry"):
        release_e2_itl_timestamp_plan(registry, foreign)

    monkeypatch.setattr(
        itl_authority,
        "RELEASE_ITL_TIMESTAMP_PRODUCERS",
        (
            _producer("native_per_token_timestamp_hook"),
            _producer("sse_one_token_per_frame"),
        ),
    )
    with pytest.raises(RuntimeError, match="allowlist is ambiguous"):
        release_e2_itl_timestamp_plan(registry, registry.cells_for("E2")[0])
