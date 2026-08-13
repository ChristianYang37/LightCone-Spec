from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy

import pytest

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.orchestration.session_reuse_authority import (
    CONNECTION_ACCOUNTING_BLOCK_REASON,
    FRESH_PROCESS_FALLBACK_MODE,
    GPU_RESET_SEMANTICS,
    OFFICIAL_ALL_RESET_PRODUCER_AVAILABLE,
    OFFICIAL_RESET_STATE_PRODUCER_AVAILABLE,
    RESET_STATE_FIELDS,
    SESSION_REUSE_BLOCK_REASON,
    SOURCE_OWNED_SESSION_HOOK,
    audit_source_owned_reuse_contract,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _bind(value: dict[str, object], field: str) -> dict[str, object]:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return {**value, field: hashlib.sha256(body).hexdigest()}


class _FakeSourceRuntime:
    def __init__(
        self,
        *,
        session_plan_sha256: str,
        traces: tuple[str, ...],
        adapted: bool = True,
        fault_reset_supported: bool = False,
        continuous_connection_accounting: bool = True,
    ) -> None:
        self.session_plan_sha256 = session_plan_sha256
        self.traces = traces
        self.adapted = adapted
        self.fault_reset_supported = fault_reset_supported
        self.continuous_connection_accounting = continuous_connection_accounting
        self.process_identity = "native-process-17"
        self.session_epoch = 2
        self.reset_generation = 0
        self.clock_generation = 0
        self.connections_created = 1
        self.connections_closed = 0
        self.submitted_requests = 0
        self.reused_requests = 0
        self.abort_requests = 0
        self.last_warmup: dict[str, object] | None = None
        self.last_clock: dict[str, object] | None = None
        self.close_calls = 0
        self.force_close_calls = 0
        self.initial_receipt_sha256: str | None = None
        self.mutate_capability = None
        self.mutate_initial = None
        self.mutate_reset = None
        self.mutate_after = None
        self.mutate_warmup = None
        self.mutate_clock = None
        self.mutate_trace = None
        self.mutate_close = None

    def _accounting(self) -> dict[str, int]:
        return {
            "generation": 4,
            "connections_created": self.connections_created,
            "connections_closed": self.connections_closed,
            "submitted_requests": self.submitted_requests,
            "reused_requests": self.reused_requests,
            "abort_requests": self.abort_requests,
        }

    def _state(self, *, generation: int, clean: bool) -> dict[str, object]:
        adaptation = {
            "fp32_master_sha256": _sha("initial-master"),
            "optimizer_moments_sha256": _sha("zero-moments"),
            "candidate_buffers_sha256": _sha("zero-candidate"),
            "cohort_state_sha256": _sha("empty-cohorts"),
        }
        if not self.adapted:
            adaptation = {name: None for name in adaptation}
        return {
            "process_identity": self.process_identity,
            "session_epoch": self.session_epoch,
            "reset_generation": generation,
            "active_requests": 0 if clean else 2,
            "queued_requests": 0 if clean else 1,
            "request_kv_entries": 0 if clean else 8,
            "prefix_entries": 0,
            "prefix_policy": "clear",
            "registered_prefix_sha256": None,
            "rng_sha256": _sha("seed-locked"),
            "counters_sha256": _sha("zero-counters"),
            "cuda_peaks_sha256": _sha("zero-peaks"),
            "scheduler_statistics_sha256": _sha("zero-scheduler"),
            "telemetry_sha256": _sha("zero-telemetry"),
            "inference_weights_sha256": _sha("initial-weights"),
            **adaptation,
            "adapter_version": 0 if clean else 3,
            "optimizer_generation": 0 if clean else 3,
            "update_counter": 0 if clean else 3,
            "allocator_allocated_bytes": 10_000,
            "allocator_reserved_bytes": 20_000,
            "hbm_state_sha256": _sha("initial-hbm"),
            "completion_event_generation": generation,
            "completion_event_complete": clean,
            "completion_event_sha256": _sha(f"event-{generation}"),
            "connection_accounting": self._accounting(),
        }

    async def capability(self, *, session_plan_sha256: str) -> object:
        assert session_plan_sha256 == self.session_plan_sha256
        value: dict[str, object] = {
            "schema_version": 1,
            "hook": SOURCE_OWNED_SESSION_HOOK,
            "producer": "native_server",
            "patched_sglang_tree": PINNED_SGLANG_TREE,
            "session_plan_sha256": self.session_plan_sha256,
            "process_identity": self.process_identity,
            "process_started_ns": 100,
            "session_epoch": self.session_epoch,
            "adapted_method": self.adapted,
            "reset_state_fields": list(RESET_STATE_FIELDS),
            "continuous_connection_accounting": self.continuous_connection_accounting,
            "fault_reset_supported": self.fault_reset_supported,
            "gpu_reset_semantics": GPU_RESET_SEMANTICS,
            "fallback_mode": FRESH_PROCESS_FALLBACK_MODE,
        }
        if self.mutate_capability is not None:
            self.mutate_capability(value)
        return _bind(value, "capability_sha256")

    async def initial_state(self, *, capability_sha256: str) -> object:
        assert len(capability_sha256) == 64
        value: dict[str, object] = {
            "schema_version": 1,
            "hook": SOURCE_OWNED_SESSION_HOOK,
            "capability_sha256": capability_sha256,
            "session_plan_sha256": self.session_plan_sha256,
            "process_identity": self.process_identity,
            "session_epoch": self.session_epoch,
            "state": self._state(generation=0, clean=True),
        }
        if self.mutate_initial is not None:
            self.mutate_initial(value)
        receipt = _bind(value, "initial_state_receipt_sha256")
        self.initial_receipt_sha256 = receipt["initial_state_receipt_sha256"]  # type: ignore[assignment]
        return receipt

    async def reset_boundary(
        self,
        *,
        capability_sha256: str,
        prior_execution_plan_sha256: str | None,
        next_execution_plan_sha256: str,
    ) -> object:
        before = self._state(generation=self.reset_generation, clean=False)
        self.reset_generation += 1
        after = self._state(generation=self.reset_generation, clean=True)
        if self.mutate_after is not None:
            self.mutate_after(after)
        value: dict[str, object] = {
            "schema_version": 1,
            "hook": SOURCE_OWNED_SESSION_HOOK,
            "capability_sha256": capability_sha256,
            "initial_state_receipt_sha256": self.initial_receipt_sha256,
            "session_plan_sha256": self.session_plan_sha256,
            "process_identity": self.process_identity,
            "session_epoch": self.session_epoch,
            "prior_execution_plan_sha256": prior_execution_plan_sha256,
            "next_execution_plan_sha256": next_execution_plan_sha256,
            "before": before,
            "after": after,
            "reset_duration_ns": 500,
        }
        if self.mutate_reset is not None:
            self.mutate_reset(value)
        return _bind(value, "reset_receipt_sha256")

    async def excluded_warmup(self, *, execution_plan_sha256: str) -> object:
        self.submitted_requests += 1
        self.reused_requests += 1
        value: dict[str, object] = {
            "schema_version": 1,
            "execution_plan_sha256": execution_plan_sha256,
            "request_pool_sha256": _sha(f"warmup-pool-{execution_plan_sha256}"),
            "excluded": True,
            "started_ns": 1_000 * (self.clock_generation + 1),
            "completed_ns": 1_000 * (self.clock_generation + 1) + 100,
            "connection_accounting": self._accounting(),
        }
        if self.mutate_warmup is not None:
            self.mutate_warmup(value)
        self.last_warmup = _bind(value, "warmup_receipt_sha256")
        return self.last_warmup

    async def start_scored_clock(self, *, execution_plan_sha256: str) -> object:
        assert self.last_warmup is not None
        self.clock_generation += 1
        value: dict[str, object] = {
            "schema_version": 1,
            "execution_plan_sha256": execution_plan_sha256,
            "warmup_receipt_sha256": self.last_warmup["warmup_receipt_sha256"],
            "clock_generation": self.clock_generation,
            "scored_started_ns": self.last_warmup["completed_ns"],
        }
        if self.mutate_clock is not None:
            self.mutate_clock(value)
        self.last_clock = _bind(value, "clock_receipt_sha256")
        return self.last_clock

    async def finish_trace(self, *, execution_plan_sha256: str) -> object:
        assert self.last_clock is not None
        self.submitted_requests += 2
        self.reused_requests += 2
        value: dict[str, object] = {
            "schema_version": 1,
            "execution_plan_sha256": execution_plan_sha256,
            "clock_receipt_sha256": self.last_clock["clock_receipt_sha256"],
            "terminal_receipt_sha256": _sha(f"terminal-{execution_plan_sha256}"),
            "aborted": False,
            "connection_accounting": self._accounting(),
        }
        if self.mutate_trace is not None:
            self.mutate_trace(value)
        return _bind(value, "trace_receipt_sha256")

    async def close(self, *, capability_sha256: str) -> object:
        assert len(capability_sha256) == 64
        self.close_calls += 1
        self.connections_closed = self.connections_created
        value: dict[str, object] = {
            "schema_version": 1,
            "process_identity": self.process_identity,
            "closed": True,
            "connection_accounting": self._accounting(),
        }
        if self.mutate_close is not None:
            self.mutate_close(value)
        return _bind(value, "close_receipt_sha256")

    async def force_close(self) -> None:
        self.force_close_calls += 1


def _audit(runtime: _FakeSourceRuntime, *, fault_injection: bool = False):
    return asyncio.run(
        audit_source_owned_reuse_contract(
            session_plan_sha256=runtime.session_plan_sha256,
            execution_plan_sha256s=runtime.traces,
            runtime=runtime,
            fault_injection=fault_injection,
        )
    )


def test_complete_source_owned_lifecycle_stays_cpu_only_and_formally_blocked() -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"), _sha("trace-2")),
    )
    result = _audit(runtime)
    assert OFFICIAL_RESET_STATE_PRODUCER_AVAILABLE
    assert not OFFICIAL_ALL_RESET_PRODUCER_AVAILABLE
    assert result.status == "CPU_CONTRACT_ONLY"
    assert result.reason == SESSION_REUSE_BLOCK_REASON
    assert not result.reuse_authorized
    assert result.fallback_mode == FRESH_PROCESS_FALLBACK_MODE
    assert len(result.reset_receipt_sha256s) == 2
    assert result.initial_state_receipt_sha256 is not None
    assert len(result.warmup_receipt_sha256s) == 2
    assert len(result.clock_receipt_sha256s) == 2
    assert len(result.trace_receipt_sha256s) == 2
    assert result.close_receipt_sha256 is not None
    assert runtime.close_calls == 1
    assert runtime.clock_generation == 2


def test_capability_cannot_promote_pending_gpu_semantics() -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"),),
    )
    runtime.mutate_capability = lambda value: value.__setitem__(
        "gpu_reset_semantics", "MEASURED"
    )
    result = _audit(runtime)
    assert result.status == "FRESH_PROCESS_REQUIRED"
    assert "not release-pending" in result.reason
    assert not result.reuse_authorized
    assert runtime.close_calls == 0
    assert runtime.force_close_calls == 1


def test_native_partial_producer_names_missing_connection_authority() -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"),),
        continuous_connection_accounting=False,
    )
    result = _audit(runtime)
    assert result.status == "FRESH_PROCESS_REQUIRED"
    assert CONNECTION_ACCOUNTING_BLOCK_REASON in result.reason
    assert result.initial_state_receipt_sha256 is None
    assert runtime.force_close_calls == 1
    assert not result.reuse_authorized


def test_initial_state_receipt_is_content_bound_to_capability() -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"),),
    )
    runtime.mutate_initial = lambda value: value.__setitem__(
        "capability_sha256", _sha("foreign-capability")
    )
    result = _audit(runtime)
    assert result.status == "FRESH_PROCESS_REQUIRED"
    assert "another capability" in result.reason
    assert runtime.force_close_calls == 1


@pytest.mark.parametrize(
    "mutation",
    ("capability", "initial", "reset", "warmup", "clock", "trace", "close"),
)
def test_boolean_schema_versions_are_rejected(mutation: str) -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"),),
    )
    setattr(
        runtime,
        f"mutate_{mutation}",
        lambda value: value.__setitem__("schema_version", True),
    )
    result = _audit(runtime)
    assert result.status == "FRESH_PROCESS_REQUIRED"
    assert "schema is unsupported" in result.reason
    assert not result.reuse_authorized


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("active_requests", 1, "drain all live"),
        ("queued_requests", 1, "drain all live"),
        ("request_kv_entries", 1, "request KV"),
        ("prefix_entries", 1, "prefix state"),
        ("inference_weights_sha256", _sha("mutated"), "clean state"),
        ("optimizer_generation", 1, "initial generations"),
        ("cohort_state_sha256", _sha("dirty-cohort"), "clean state"),
        ("completion_event_complete", False, "completion event"),
    ),
)
def test_incomplete_native_reset_closes_and_requires_fresh_process(
    field: str,
    value: object,
    message: str,
) -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"),),
    )
    runtime.mutate_after = lambda state: state.__setitem__(field, value)
    result = _audit(runtime)
    assert result.status == "FRESH_PROCESS_REQUIRED"
    assert message in result.reason
    assert result.close_receipt_sha256 is not None
    assert runtime.close_calls == 1


def test_reset_generation_cannot_skip_the_source_owned_chain() -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"),),
    )

    def skip_generation(value: dict[str, object]) -> None:
        before = value["before"]
        after = value["after"]
        assert isinstance(before, dict) and isinstance(after, dict)
        before["reset_generation"] = 7
        after["reset_generation"] = 8

    runtime.mutate_reset = skip_generation
    result = _audit(runtime)
    assert result.status == "FRESH_PROCESS_REQUIRED"
    assert "pre-reset generation" in result.reason
    assert runtime.close_calls == 1


def test_client_style_or_incomplete_capability_cannot_unlock_native_reuse() -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"),),
    )
    runtime.mutate_capability = lambda value: value.__setitem__(
        "producer", "client_counter_wrapper"
    )
    result = _audit(runtime)
    assert result.status == "FRESH_PROCESS_REQUIRED"
    assert "not source-owned" in result.reason
    assert result.capability_sha256 is None
    assert not result.reuse_authorized


@pytest.mark.parametrize("mutation", ("warmup", "clock"))
def test_every_trace_requires_excluded_warmup_then_new_scored_clock(
    mutation: str,
) -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"),),
    )
    if mutation == "warmup":
        runtime.mutate_warmup = lambda value: value.__setitem__("excluded", False)
    else:
        runtime.mutate_clock = lambda value: value.__setitem__("clock_generation", 7)
    result = _audit(runtime)
    assert result.status == "FRESH_PROCESS_REQUIRED"
    assert runtime.close_calls == 1


def test_abort_closes_source_and_never_reuses_the_process() -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"), _sha("trace-2")),
    )

    def abort(value: dict[str, object]) -> None:
        value["aborted"] = True
        runtime.abort_requests = 1
        value["connection_accounting"] = runtime._accounting()

    runtime.mutate_trace = abort
    runtime.mutate_close = lambda value: value.__setitem__("closed", False)
    result = _audit(runtime)
    assert result.status == "FRESH_PROCESS_REQUIRED"
    assert "source_trace_aborted" in result.reason
    assert "shared_session_close_failed" in result.reason
    assert len(result.trace_receipt_sha256s) == 1
    assert runtime.close_calls == 1


def test_connection_accounting_must_be_continuous_and_source_closed() -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"),),
    )

    def move_backwards(value: dict[str, object]) -> None:
        accounting = deepcopy(value["connection_accounting"])
        assert isinstance(accounting, dict)
        accounting["submitted_requests"] = 0
        accounting["reused_requests"] = 0
        value["connection_accounting"] = accounting

    runtime.mutate_trace = move_backwards
    result = _audit(runtime)
    assert result.status == "FRESH_PROCESS_REQUIRED"
    assert "connection accounting moved backwards" in result.reason
    assert runtime.close_calls == 1


def test_fault_injection_defaults_to_fresh_process() -> None:
    runtime = _FakeSourceRuntime(
        session_plan_sha256=_sha("session-plan"),
        traces=(_sha("trace-1"),),
    )
    result = _audit(runtime, fault_injection=True)
    assert result.status == "FRESH_PROCESS_REQUIRED"
    assert "fault_injection_requires_fresh_process" in result.reason
    assert runtime.close_calls == 1
