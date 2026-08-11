from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from pathlib import Path

import pytest

import lightcone_spec.orchestration.session as session_module
from lightcone_spec.orchestration.session import (
    SHARED_SESSION_UNAVAILABLE_REASON,
    IndustrialResetReceipt,
    IndustrialServerSessionKey,
    IndustrialServerSessionPlan,
    IndustrialSessionOpenReceipt,
    SessionBoundaryState,
    SessionExecutionBinding,
    SharedSessionUnavailableError,
)


def _key(**updates: object) -> IndustrialServerSessionKey:
    values: dict[str, object] = {
        "patched_sglang_tree": "a" * 40,
        "capability_receipt_sha256": "b" * 64,
        "rank_config_sha256": "c" * 64,
        "model_lock_sha256": "d" * 64,
        "parameter_plan_sha256": None,
        "topology_sha256": "e" * 64,
        "gpu_uuids": ("GPU-a",),
        "method": "target_only",
        "backend": "DFLASH",
        "dtype": "bfloat16",
        "precision": "bf16",
        "context_limit": 40960,
        "graph_buckets": (1, 2, 4, 8),
        "max_running_requests": 8,
        "memory_fraction": "0.8",
        "hbm_reservation_bytes": 0,
        "telemetry_mode": "headline",
        "compile_cache_receipt_sha256": "f" * 64,
        "port_router_sha256": "1" * 64,
        "server_argv_sha256": "2" * 64,
    }
    values.update(updates)
    return IndustrialServerSessionKey(**values)  # type: ignore[arg-type]


def _plan(*, fault: bool = False) -> IndustrialServerSessionPlan:
    value = IndustrialServerSessionPlan(
        session_key=_key(),
        execution_plan_sha256s=("3" * 64, "4" * 64),
        method="target_only",
        block=2,
        fault_injection=fault,
    )
    value.validate()
    return value


def _state(**updates: object) -> SessionBoundaryState:
    values: dict[str, object] = {
        "process_identity": "pid-123-start-456",
        "session_epoch": 7,
        "reset_generation": 0,
        "active_requests": 0,
        "queued_requests": 0,
        "rng_sha256": "5" * 64,
        "inference_weights_sha256": "6" * 64,
        "fp32_master_sha256": None,
        "optimizer_state_sha256": None,
        "candidate_buffers_sha256": None,
        "scheduler_state_sha256": "7" * 64,
        "kv_state_sha256": "8" * 64,
        "telemetry_state_sha256": "9" * 64,
        "adapter_version": 0,
        "optimizer_generation": 0,
        "allocator_allocated_bytes": 100,
        "allocator_reserved_bytes": 200,
        "hbm_state_sha256": "a" * 64,
        "completion_event_sha256": "b" * 64,
    }
    values.update(updates)
    return SessionBoundaryState(**values)  # type: ignore[arg-type]


def test_session_key_binds_every_server_affecting_identity() -> None:
    baseline = _key()
    baseline.validate()
    for name, value in (
        ("compile_cache_receipt_sha256", "0" * 64),
        ("server_argv_sha256", "3" * 64),
        ("gpu_uuids", ("GPU-b",)),
        ("context_limit", 32768),
        ("max_running_requests", 16),
        ("hbm_reservation_bytes", 1024),
        ("telemetry_mode", "profile"),
    ):
        assert _key(**{name: value}).sha256 != baseline.sha256


def test_session_cannot_cross_block_method_or_fault_boundary() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="fault injection"):
        replace(plan, fault_injection=True).validate()
    with pytest.raises(ValueError, match="method"):
        replace(plan, method="static").validate()


def test_session_execution_binding_has_no_caller_authored_timing() -> None:
    field_names = {field.name for field in fields(SessionExecutionBinding)}
    assert "pre_trace_reset_observed_ms" not in field_names
    assert "execution_plan_sha256" in field_names

    binding = SessionExecutionBinding(
        session_plan_sha256="1" * 64,
        session_open_receipt_sha256="2" * 64,
        reset_receipt_sha256="3" * 64,
        execution_plan_sha256="4" * 64,
        session_epoch=7,
        native_session_id="native-session",
        native_trace_epoch=1,
        native_previous_run_id=None,
    )
    binding.validate()
    with pytest.raises(ValueError, match="execution_plan_sha256"):
        replace(binding, execution_plan_sha256="wrong-plan").validate()


def test_reset_receipt_restores_exact_open_state_and_order() -> None:
    plan = _plan()
    clean = _state()
    opened = IndustrialSessionOpenReceipt(
        session_plan_sha256=plan.sha256,
        process_identity=clean.process_identity,
        process_started_ns=1000,
        session_epoch=clean.session_epoch,
        clean_state_sha256=clean.clean_state_sha256,
        native_capability_receipt_sha256="c" * 64,
    )
    before = replace(
        clean,
        active_requests=2,
        queued_requests=1,
        reset_generation=4,
        kv_state_sha256="d" * 64,
    )
    after = replace(clean, reset_generation=5)
    receipt = IndustrialResetReceipt.create(
        session_plan=plan,
        open_receipt=opened,
        prior_execution_plan_sha256=None,
        next_execution_plan_sha256=plan.execution_plan_sha256s[0],
        before=before,
        after=after,
        reset_duration_ms=2.5,
    )
    receipt.validate(session_plan=plan, open_receipt=opened)

    with pytest.raises(ValueError, match="ordered trace"):
        IndustrialResetReceipt.create(
            session_plan=plan,
            open_receipt=opened,
            prior_execution_plan_sha256=None,
            next_execution_plan_sha256=plan.execution_plan_sha256s[1],
            before=before,
            after=after,
            reset_duration_ms=2.5,
        )


def test_reset_rejects_live_requests_and_state_drift() -> None:
    plan = _plan()
    clean = _state()
    opened = IndustrialSessionOpenReceipt(
        session_plan_sha256=plan.sha256,
        process_identity=clean.process_identity,
        process_started_ns=1000,
        session_epoch=clean.session_epoch,
        clean_state_sha256=clean.clean_state_sha256,
        native_capability_receipt_sha256="c" * 64,
    )
    before = replace(clean, reset_generation=1, active_requests=1)
    for after, message in (
        (replace(clean, reset_generation=2, active_requests=1), "zero live"),
        (
            replace(clean, reset_generation=2, inference_weights_sha256="e" * 64),
            "clean state",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            IndustrialResetReceipt.create(
                session_plan=plan,
                open_receipt=opened,
                prior_execution_plan_sha256=None,
                next_execution_plan_sha256=plan.execution_plan_sha256s[0],
                before=before,
                after=after,
                reset_duration_ms=1.0,
            )


def test_direct_trace_execution_is_blocked_before_evidence_root_mutation(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(
        SharedSessionUnavailableError,
        match=SHARED_SESSION_UNAVAILABLE_REASON,
    ):
        asyncio.run(
            session_module.execute_trace_in_session(
                object(),  # type: ignore[arg-type]
                output_root=output_root,
                run_nonce_sha256="a" * 64,
            )
        )
    assert not output_root.exists()


def test_forged_private_lifecycle_cannot_claim_session_timing() -> None:
    lifecycle = session_module._SessionExecutionLifecycle(object())
    with pytest.raises(
        SharedSessionUnavailableError,
        match=SHARED_SESSION_UNAVAILABLE_REASON,
    ):
        lifecycle.claim_startup_interval_ns(execution_plan_sha256="b" * 64)
