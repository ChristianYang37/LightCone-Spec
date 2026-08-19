from __future__ import annotations

import asyncio
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalProvider,
    NativeTerminalRunBinding,
    TerminalRequestExpectation,
    canonical_json_bytes,
    canonical_sha256,
)
from lightcone_spec.orchestration.session_live_runtime import (
    SessionLivePinnedBenchTransport,
    SessionLiveStepBinding,
    SessionLiveTraceInput,
    run_session_live_contract,
)
from lightcone_spec.runtime import readiness as readiness_module
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256,
    NATIVE_RUNTIME_QUALIFICATION_TESTS,
    NATIVE_RUNTIME_RELEASE_CAPABILITY,
    NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
    NativeRuntimeGpuProofReceipt,
    VerifiedNativeRuntimeGpuProof,
)

_TERMINAL_FIXTURE_PATH = Path(__file__).with_name("test_native_terminal_provider.py")
_TERMINAL_SPEC = importlib.util.spec_from_file_location(
    "_live_terminal_fixture", _TERMINAL_FIXTURE_PATH
)
assert _TERMINAL_SPEC is not None and _TERMINAL_SPEC.loader is not None
_TERMINAL = importlib.util.module_from_spec(_TERMINAL_SPEC)
_TERMINAL_SPEC.loader.exec_module(_TERMINAL)

_SOURCE_FIXTURE_PATH = Path(__file__).with_name("test_session_reuse_authority.py")
_SOURCE_SPEC = importlib.util.spec_from_file_location(
    "_live_source_fixture", _SOURCE_FIXTURE_PATH
)
assert _SOURCE_SPEC is not None and _SOURCE_SPEC.loader is not None
_SOURCE = importlib.util.module_from_spec(_SOURCE_SPEC)
_SOURCE_SPEC.loader.exec_module(_SOURCE)


def _sha(label: str) -> str:
    return canonical_sha256(label)


def _request(request_id: str, token: int) -> TerminalRequestExpectation:
    return TerminalRequestExpectation(
        request_id=request_id,
        input_token_ids=(token,),
        output_token_ids=(token + 1,),
        terminal_status="completed",
        terminal_reason="FINISH_LENGTH",
        submitted_to_server=True,
    )


class _Driver:
    def __init__(self, *, warmup, scored, events: list[str]) -> None:
        self.warmup = warmup
        self.scored = scored
        self.events = events

    async def run_warmup(self, **_kwargs):
        self.events.append("run_warmup")
        return self.warmup

    async def run_scored(self, **_kwargs):
        self.events.append("run_scored")
        return self.scored


class _CancelledDriver(_Driver):
    async def run_warmup(self, **_kwargs):
        self.events.append("run_warmup")
        raise asyncio.CancelledError


class _ProcessOwner:
    def __init__(self) -> None:
        self.close_calls = 0
        self.force_close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

    async def force_close(self) -> None:
        self.force_close_calls += 1


class _Response:
    status = 200

    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self):
        if hasattr(self.value, "__await__"):
            self.value = await self.value
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self, **_kwargs):
        return self.value


class _Session:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.trace_configs: list[object] = []
        self.closed = False

    def get(self, *, url, headers):
        return _Response(self.backend.get(urlsplit(url).path, headers))

    def post(self, *, url, json, headers):
        return _Response(self.backend.post(urlsplit(url).path, json, headers))


class _Backend:
    def __init__(self, *, binding, warmup, scored) -> None:
        self.native = _TERMINAL.FakeAdminTransport(
            binding=binding,
            warmup=warmup,
            scored=scored,
        )
        self.source = _SOURCE._FakeSourceRuntime(
            session_plan_sha256=binding.session_id,
            traces=(binding.execution_plan_sha256,),
            adapted=False,
        )
        self.calls: list[str] = []
        self.malformed_clock = False
        self.drop_combined_source = False
        self.break_begin_binding = False
        self.break_reset_binding = False

    async def get(self, path: str, _headers: object) -> object:
        self.calls.append(path)
        return await self.native.get_json(path)

    async def post(
        self, path: str, body: dict[str, object], _headers: object
    ) -> object:
        self.calls.append(path)
        if path.endswith("/capability") and "session-reset" in path:
            return await self.source.capability(
                session_plan_sha256=body["session_plan_sha256"],
                execution_plan_sha256s=tuple(body["execution_plan_sha256s"]),
            )
        if path.endswith("/initial-state"):
            return await self.source.initial_state(
                capability_sha256=body["capability_sha256"]
            )
        if path.endswith("/session-reset"):
            return await self.source.reset_boundary(
                capability_sha256=body["capability_sha256"],
                prior_execution_plan_sha256=body["prior_execution_plan_sha256"],
                next_execution_plan_sha256=body["next_execution_plan_sha256"],
            )
        if path.endswith("/trace/begin"):
            terminal = await self.native.post_json(
                "/v1/lightcone-spec/terminal-evidence",
                {"action": "begin", "payload": body["terminal_payload"]},
            )
            source = {
                "schema_version": 1,
                "hook": _SOURCE.SOURCE_OWNED_SESSION_HOOK,
                "capability_sha256": body["capability_sha256"],
                "execution_plan_sha256": self.native.binding.execution_plan_sha256,
                "begin_sha256": terminal["begin_sha256"],
            }
            if self.break_begin_binding:
                source["begin_sha256"] = _sha("another-native-begin")
            source["session_begin_sha256"] = canonical_sha256(source)
            return self._combined(terminal, source)
        if path.endswith("/trace/reset"):
            terminal = await self.native.post_json(
                "/v1/lightcone-spec/terminal-evidence",
                {"action": "reset", "payload": body["terminal_payload"]},
            )
            plan = self.native.binding.execution_plan_sha256
            warmup = await self.source.excluded_warmup(execution_plan_sha256=plan)
            clock = await self.source.start_scored_clock(execution_plan_sha256=plan)
            clock["native_reset_sha256"] = terminal["reset_sha256"]
            clock["clock_receipt_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in clock.items()
                    if key != "clock_receipt_sha256"
                }
            )
            self.source.clock_receipt_sha256s[-1] = clock["clock_receipt_sha256"]
            if self.malformed_clock:
                clock["clock_generation"] = 9
                clock["clock_receipt_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in clock.items()
                        if key != "clock_receipt_sha256"
                    }
                )
            if self.break_reset_binding:
                clock["native_reset_sha256"] = _sha("another-native-reset")
                clock["clock_receipt_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in clock.items()
                        if key != "clock_receipt_sha256"
                    }
                )
                self.source.clock_receipt_sha256s[-1] = clock["clock_receipt_sha256"]
            source = {
                "schema_version": 1,
                "hook": _SOURCE.SOURCE_OWNED_SESSION_HOOK,
                "capability_sha256": body["capability_sha256"],
                "warmup_receipt": warmup,
                "clock_receipt": clock,
            }
            source["trace_ready_sha256"] = canonical_sha256(source)
            return self._combined(terminal, source)
        if path.endswith("/trace/finalize"):
            terminal = await self.native.post_json(
                "/v1/lightcone-spec/terminal-evidence",
                {"action": "finalize", "payload": body["terminal_payload"]},
            )
            plan = self.native.binding.execution_plan_sha256
            source = await self.source.finish_trace(execution_plan_sha256=plan)
            source["terminal_receipt_sha256"] = terminal["terminal_sha256"]
            source.pop("trace_receipt_sha256")
            source["trace_receipt_sha256"] = canonical_sha256(source)
            self.source.trace_receipt_sha256s[-1] = source["trace_receipt_sha256"]
            self.source.terminal_receipt_sha256s[-1] = terminal["terminal_sha256"]
            return self._combined(terminal, source)
        if path.endswith("/close-terminal"):
            return await self.source.close(capability_sha256=body["capability_sha256"])
        raise AssertionError(f"unexpected path {path}")

    def _combined(self, terminal: object, source: object) -> object:
        value = {
            "schema_version": 1,
            "terminal": terminal,
            "source_owned_session": source,
        }
        if self.drop_combined_source:
            value.pop("source_owned_session")
        return value


class _TraceConfig:
    def __init__(self) -> None:
        self.on_connection_create_end = []
        self.on_connection_reuseconn = []

    def freeze(self) -> None:
        return None


def _resources(
    *,
    malformed_clock: bool = False,
    drop_source: bool = False,
    evidence_sink=None,
):
    plan = _sha("plan")
    trace = _sha("trace")
    warmup = (_request("warm-0", 1),)
    scored = (_request("score-0", 3),)
    binding = NativeTerminalRunBinding(
        run_id="run-0",
        run_nonce_sha256=_sha("nonce"),
        execution_plan_sha256=trace,
        rank_config_sha256=_sha("rank"),
        attempt_id="attempt-0",
        session_id=plan,
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=_sha("challenge"),
        method="static",
        warmup_request_ids=("warm-0",),
        scored_request_ids=("score-0",),
    )
    backend = _Backend(binding=binding, warmup=warmup, scored=scored)
    backend.malformed_clock = malformed_clock
    backend.drop_combined_source = drop_source
    session = _Session(backend)

    async def open_session(*, total_timeout_s: float = 6 * 60 * 60):
        assert total_timeout_s > 0
        return session

    async def close_session(client_session) -> None:
        assert client_session is session
        session.closed = True

    async def request(
        request_func_input, pbar=None, *, client_session=None, timeout_s=None
    ):
        raise AssertionError("driver fixture does not call raw generate")

    async def abort(request_id, base_url, *, client_session=None, timeout_s=None):
        raise AssertionError("driver fixture does not abort")

    transport = SessionLivePinnedBenchTransport(
        evidence_sink=evidence_sink,
        request_type=SimpleNamespace,
        request_callable=request,
        abort_callable=abort,
        open_session_callable=open_session,
        close_session_callable=close_session,
        set_global_args=lambda _value: None,
        trace_config_factory=_TraceConfig,
        module_identity="sglang.benchmark.serving.async_request_sglang_generate",
    )
    provider = NativeTerminalProvider(transport)
    events: list[str] = []
    driver = _Driver(warmup=warmup, scored=scored, events=events)
    owner = _ProcessOwner()
    return (
        plan,
        SessionLiveTraceInput(binding=binding, driver=driver),
        transport,
        provider,
        owner,
        backend,
        events,
    )


def _session_gpu_proof() -> VerifiedNativeRuntimeGpuProof:
    capability = NATIVE_RUNTIME_RELEASE_CAPABILITY
    receipt = NativeRuntimeGpuProofReceipt(
        schema_version=1,
        kind="lightcone_native_runtime_gpu_proof",
        suite_id="session_reset_tp1",
        topology_mode="tp1_dp1",
        topology_sha256=_sha("session-topology"),
        runner_protocol_sha256=NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[
            "session_reset_tp1"
        ],
        assignment_sha256=_sha("session-assignment"),
        qualification_observation_sha256=_sha("session-observation"),
        source_capability_sha256=capability.sha256,
        pinned_sglang_commit=capability.pinned_sglang_commit,
        patched_sglang_tree=capability.patched_sglang_tree,
        semantic_patch_sha256=capability.semantic_patch_sha256,
        run_nonce_sha256=_sha("session-qualification-nonce"),
        qualification_authority_sha256=(NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256),
        source_identity_sha256=_sha("session-source"),
        inventory_sha256=_sha("session-inventory"),
        gpu_uuids=("GPU-A",),
        hardware_envelope_sha256=_sha("session-hardware"),
        junit_xml_sha256=_sha("session-junit"),
        test_names=NATIVE_RUNTIME_QUALIFICATION_TESTS["session_reset_tp1"],
        tests_collected=8,
        tests_passed=8,
        tests_failed=0,
        tests_errored=0,
        tests_skipped=0,
    )
    return VerifiedNativeRuntimeGpuProof(
        receipt=receipt,
        receipt_raw_sha256=_sha("session-raw-receipt"),
        trusted_policy_sha256=_sha("session-policy"),
        challenge_sha256=_sha("session-control-challenge"),
        control_envelope_sha256=_sha("session-control-envelope"),
        challenge_reservation_sha256=_sha("session-reservation"),
        _verification_tag=readiness_module._VERIFIED_NATIVE_GPU_PROOF_SENTINEL,
    )


def _run(resources, *, verified_gpu_proof=None):
    plan, trace, transport, provider, owner, backend, events = resources
    result = asyncio.run(
        run_session_live_contract(
            session_plan_sha256=plan,
            traces=(trace,),
            base_url="http://127.0.0.1:30000",
            request_timeout_s=1.0,
            abort_timeout_s=1.0,
            transport=transport,
            provider=provider,
            process_owner=owner,
            verified_gpu_proof=verified_gpu_proof,
        )
    )
    return result, owner, backend, events


def test_live_consumer_uses_one_pool_one_provider_and_exact_order() -> None:
    result, owner, backend, events = _run(_resources())

    assert result.audit.status == "CPU_CONTRACT_ONLY"
    assert result.reuse_authorized is False
    assert len(result.native_terminals) == 1
    assert owner.close_calls == 1 and owner.force_close_calls == 0
    assert events == ["run_warmup", "run_scored"]
    assert [step.step for step in result.steps] == [
        "session_capability",
        "session_initial_state",
        "session_reset_boundary",
        "native_terminal_capability",
        "atomic_trace_begin",
        "atomic_trace_reset",
        "atomic_trace_finalize",
        "session_close_terminal",
    ]
    assert backend.calls == [
        "/v1/lightcone-spec/session-reset/capability",
        "/v1/lightcone-spec/session-reset/initial-state",
        "/v1/lightcone-spec/session-reset",
        "/v1/lightcone-spec/terminal-evidence/capability",
        "/v1/lightcone-spec/session-reset/trace/begin",
        "/v1/lightcone-spec/session-reset/trace/reset",
        "/v1/lightcone-spec/session-reset/trace/finalize",
        "/v1/lightcone-spec/session-reset/close-terminal",
    ]


def test_exact_session_gpu_proof_authorizes_reuse_without_weakening_close() -> None:
    proof = _session_gpu_proof()
    result, owner, _backend, events = _run(
        _resources(),
        verified_gpu_proof=proof,
    )

    assert result.audit.status == "GPU_VERIFIED"
    assert result.reuse_authorized
    assert result.verified_gpu_proof_sha256 == proof.sha256
    assert owner.close_calls == 1 and owner.force_close_calls == 0
    assert events == ["run_warmup", "run_scored"]


def test_gpu_verified_result_still_deep_validates_retained_receipt_chain() -> None:
    result, _owner, _backend, _events = _run(
        _resources(),
        verified_gpu_proof=_session_gpu_proof(),
    )
    steps = list(result.steps)
    index = next(
        index for index, step in enumerate(steps) if step.step == "session_capability"
    )
    payload = json.loads(steps[index].raw_json)
    payload["session_plan_sha256"] = _sha("replaced-session-plan")
    steps[index] = SessionLiveStepBinding.capture(
        step="session_capability",
        value=payload,
    )

    with pytest.raises(ValueError, match="not content-bound|raw authority changed"):
        replace(result, steps=tuple(steps)).validate()


def test_malformed_clock_blocks_before_any_scored_request_and_force_closes() -> None:
    result, owner, _backend, events = _run(_resources(malformed_clock=True))

    assert result.audit.status == "FRESH_PROCESS_REQUIRED"
    assert events == ["run_warmup"]
    assert owner.close_calls == 0 and owner.force_close_calls == 1
    assert result.transport_closed and result.process_force_closed


def test_incomplete_combined_response_closes_pool_and_force_closes() -> None:
    result, owner, _backend, events = _run(_resources(drop_source=True))

    assert result.audit.status == "FRESH_PROCESS_REQUIRED"
    assert events == []
    assert owner.close_calls == 0 and owner.force_close_calls == 1
    assert result.transport_closed and result.process_force_closed


def test_stale_pool_or_provider_is_rejected_before_network_mutation() -> None:
    plan, trace, transport, provider, owner, backend, _events = _resources()
    provider._seen_runs.add("old-run")

    with pytest.raises(ValueError, match="stale native terminal"):
        asyncio.run(
            run_session_live_contract(
                session_plan_sha256=plan,
                traces=(trace,),
                base_url="http://127.0.0.1:30000",
                request_timeout_s=1.0,
                abort_timeout_s=1.0,
                transport=transport,
                provider=provider,
                process_owner=owner,
            )
        )
    assert backend.calls == []
    assert transport._session is None
    assert owner.close_calls == owner.force_close_calls == 0


@pytest.mark.parametrize("broken_binding", ["begin", "reset"])
def test_composite_source_receipt_must_bind_same_native_transition(
    broken_binding: str,
) -> None:
    resources = _resources()
    backend = resources[5]
    if broken_binding == "begin":
        backend.break_begin_binding = True
    else:
        backend.break_reset_binding = True

    result, owner, _backend, events = _run(resources)

    assert result.audit.status == "FRESH_PROCESS_REQUIRED"
    assert events == ([] if broken_binding == "begin" else ["run_warmup"])
    assert owner.close_calls == 0 and owner.force_close_calls == 1
    assert result.transport_closed and result.process_force_closed


def test_cancellation_closes_pool_and_force_closes_owned_process() -> None:
    plan, trace, transport, provider, owner, _backend, events = _resources()
    trace = replace(
        trace,
        driver=_CancelledDriver(warmup=(), scored=(), events=events),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_session_live_contract(
                session_plan_sha256=plan,
                traces=(trace,),
                base_url="http://127.0.0.1:30000",
                request_timeout_s=1.0,
                abort_timeout_s=1.0,
                transport=transport,
                provider=provider,
                process_owner=owner,
            )
        )

    assert events == ["run_warmup"]
    assert transport._session is None
    assert owner.close_calls == 0 and owner.force_close_calls == 1


def test_result_rejects_reordered_or_replace_and_rehashed_raw_steps() -> None:
    result, _owner, _backend, _events = _run(_resources())

    with pytest.raises(ValueError, match="step chain"):
        replace(result, steps=tuple(reversed(result.steps))).validate()

    capability_step = result.steps[0]
    value = json.loads(capability_step.raw_json)
    value["capability_sha256"] = _sha("replaced-capability")
    replacement = replace(
        capability_step,
        raw_json=canonical_json_bytes(value).decode("utf-8"),
        content_sha256=canonical_sha256(value),
    )
    with pytest.raises(ValueError, match="content-bound|raw authority changed"):
        replace(result, steps=(replacement, *result.steps[1:])).validate()
