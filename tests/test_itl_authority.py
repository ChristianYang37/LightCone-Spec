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
    ITL_DYNAMIC_GPU_PROOF_UNAVAILABLE_REASON,
    ITL_RAW_RECEIPT_MISSING_REASON,
    ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256,
    ItlRequestExpectation,
    ItlTimestampAuthorityBlocked,
    ReleaseItlTimestampProducer,
    StageItlExecutionIdentity,
    StageItlTimestampAuthority,
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
from lightcone_spec.experiments.registry import (
    build_legacy_industrial_registry as build_industrial_registry,
)
from lightcone_spec.experiments.registry import (
    scientific_role_for_cell,
)


def _registry_and_cell():
    registry = build_industrial_registry()
    cell = next(
        row
        for row in registry.cells_for("E2")
        if scientific_role_for_cell(registry, row) == "lc_candidate"
        and row.identity.optimizer == "adamw"
    )
    return registry, cell


def _producer(mode: str) -> ReleaseItlTimestampProducer:
    hook = {
        "native_per_token_timestamp_hook": (
            "sglang.schema_v3.native_per_token_timestamp.v2"
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
    assert activation.reason_code == ITL_DYNAMIC_GPU_PROOF_UNAVAILABLE_REASON
    assert activation.producer_sha256 is None
    with pytest.raises(ItlTimestampAuthorityBlocked) as error:
        require_e2_itl_timestamp_prelaunch(plan)
    assert error.value.reason == ITL_DYNAMIC_GPU_PROOF_UNAVAILABLE_REASON

    missing = tmp_path / "must-not-be-created" / "raw-itl.json"
    with pytest.raises(ItlTimestampAuthorityBlocked) as error:
        bind_itl_timestamp_authority(plan, missing, expected_requests=_expected())
    assert error.value.reason == ITL_DYNAMIC_GPU_PROOF_UNAVAILABLE_REASON
    assert not missing.parent.exists()

    with pytest.raises(ItlTimestampAuthorityBlocked) as error:
        load_path_bound_itl_timestamp_authority(
            missing,
            registry=registry,
            cell=cell,
            expected_requests=_expected(),
        )
    assert error.value.reason == ITL_DYNAMIC_GPU_PROOF_UNAVAILABLE_REASON
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
    assert activation.reason_code == ITL_DYNAMIC_GPU_PROOF_UNAVAILABLE_REASON
    missing = tmp_path / "must-not-read" / "forged.json"
    with pytest.raises(ItlTimestampAuthorityBlocked) as error:
        bind_itl_timestamp_authority(plan, missing, expected_requests=_expected())
    assert error.value.reason == ITL_DYNAMIC_GPU_PROOF_UNAVAILABLE_REASON
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


def test_stage_itl_identity_binds_materialized_execution_and_rejects_forgery() -> None:
    identity = StageItlExecutionIdentity(
        schema_version=1,
        kind="stage_itl_execution_identity",
        materialized_cell_id="1" * 64,
        inventory_sha256="2" * 64,
        registry_sha256="3" * 64,
        execution_plan_sha256="4" * 64,
        rank_config_sha256="5" * 64,
        run_id="run-e1-0",
        run_nonce_sha256="6" * 64,
        attempt_id="attempt-e1-0",
        method="l0",
        runtime_trust_mode=None,
        formal_measurement=None,
    )
    assert StageItlExecutionIdentity.from_dict(identity.to_dict()) == identity
    assert identity.sha256 == itl_authority.content_sha256(identity.to_dict())
    forged = identity.to_dict()
    forged["materialized_cell_id"] = "not-a-digest"
    with pytest.raises(ValueError, match="materialized cell"):
        StageItlExecutionIdentity.from_dict(forged)

    for runtime_trust_mode, formal_measurement in (
        (None, False),
        ("release_verified_signature", False),
        ("release_verified_signature", 1),
        ("qualification_empirical_no_signature", True),
    ):
        invalid_trust = {
            **identity.to_dict(),
            "runtime_trust_mode": runtime_trust_mode,
            "formal_measurement": formal_measurement,
        }
        with pytest.raises(ValueError, match="runtime trust identity"):
            StageItlExecutionIdentity.from_dict(invalid_trust)

    allocation_free = {
        **identity.to_dict(),
        "method": "static",
        "runtime_trust_mode": "release_verified_signature",
        "formal_measurement": True,
    }
    with pytest.raises(ValueError, match="allocation-free"):
        StageItlExecutionIdentity.from_dict(allocation_free)


@pytest.mark.parametrize(
    "method",
    ("onlinespec_ogd", "onlinespec_opt", "onlinespec_ens"),
)
def test_stage_itl_identity_accepts_only_registered_onlinespec_methods(
    method: str,
) -> None:
    value = StageItlExecutionIdentity(
        schema_version=1,
        kind="stage_itl_execution_identity",
        materialized_cell_id="1" * 64,
        inventory_sha256="2" * 64,
        registry_sha256="3" * 64,
        execution_plan_sha256="4" * 64,
        rank_config_sha256="5" * 64,
        run_id=f"run-{method}",
        run_nonce_sha256="6" * 64,
        attempt_id=f"attempt-{method}",
        method=method,  # type: ignore[arg-type]
        runtime_trust_mode=None,
        formal_measurement=None,
    )
    assert StageItlExecutionIdentity.from_dict(value.to_dict()) == value

    forged = value.to_dict()
    forged["method"] = "onlinespec_unregistered"
    with pytest.raises(ValueError, match="method is unsupported"):
        StageItlExecutionIdentity.from_dict(forged)


def test_stage_itl_authority_cannot_be_constructed_from_caller_timestamps() -> None:
    identity = StageItlExecutionIdentity(
        schema_version=1,
        kind="stage_itl_execution_identity",
        materialized_cell_id="1" * 64,
        inventory_sha256="2" * 64,
        registry_sha256="3" * 64,
        execution_plan_sha256="4" * 64,
        rank_config_sha256="5" * 64,
        run_id="run-e1-0",
        run_nonce_sha256="6" * 64,
        attempt_id="attempt-e1-0",
        method="l0",
        runtime_trust_mode=None,
        formal_measurement=None,
    )
    with pytest.raises(TypeError, match="first-party validation"):
        StageItlTimestampAuthority(
            execution_identity=identity,
            raw_receipt=None,  # type: ignore[arg-type]
            native_result_proof=None,  # type: ignore[arg-type]
            native_gpu_proof=None,  # type: ignore[arg-type]
            native_gpu_verified_proof_sha256="7" * 64,
            producer_sha256="8" * 64,
            control_binding_sha256="9" * 64,
            control_envelope_sha256="a" * 64,
            replay_reservation_sha256="b" * 64,
            expectations_sha256="c" * 64,
            native_result_pointer_sha256s=("d" * 64,),
            requests=(),
            _verification_tag=object(),
        )


def test_stage_itl_source_derives_integer_measurement_inputs_only() -> None:
    source = Path(itl_authority.__file__).read_text(encoding="utf-8")
    for required in (
        '"arrival_ns": row.request_started_ns',
        '"first_token_ns": row.token_observed_ns[0]',
        '"completion_ns": row.request_terminal_ns',
        '"native_per_token_observed_ns": list(row.token_observed_ns)',
        "throughput_numerator_tokens",
        "throughput_window_ns",
        "p99_itl_input_ns",
        "validate_formal_terminal_result_proof_artifact",
        "NativeRuntimeGpuProofArtifact.from_dict",
        "verify_and_reserve_release_control_artifact_attestations",
    ):
        assert required in source
    assert "caller_goodput" not in source


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
