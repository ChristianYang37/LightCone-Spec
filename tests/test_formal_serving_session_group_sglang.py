from __future__ import annotations

import asyncio
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from lightcone_spec.orchestration.executor import RegisteredServingExecutionPolicy
from lightcone_spec.orchestration.formal_serving_session_group_physical import (
    effective_formal_serving_resident_terminal_binding,
)
from lightcone_spec.orchestration.formal_serving_session_group_sglang import (
    PinnedSglangResidentProcessDriver,
    _PreparedResidentMember,
)
from lightcone_spec.orchestration.formal_serving_session_source_chain import (
    revalidate_formal_serving_resident_source_chain,
)
from lightcone_spec.orchestration.native_terminal import (
    TerminalRequestExpectation,
    UnsignedNativeServingPhaseResult,
    canonical_json_bytes,
    canonical_sha256,
)
from lightcone_spec.orchestration.session_live_runtime import (
    SessionLivePinnedBenchTransport,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_RESIDENT_FIXTURE_PATH = Path(__file__).with_name(
    "test_formal_serving_session_group_resident.py"
)
_RESIDENT_SPEC = importlib.util.spec_from_file_location(
    "_resident_sglang_group_fixture", _RESIDENT_FIXTURE_PATH
)
assert _RESIDENT_SPEC is not None and _RESIDENT_SPEC.loader is not None
_RESIDENT = importlib.util.module_from_spec(_RESIDENT_SPEC)
_RESIDENT_SPEC.loader.exec_module(_RESIDENT)

_TERMINAL_FIXTURE_PATH = Path(__file__).with_name("test_native_terminal_provider.py")
_TERMINAL_SPEC = importlib.util.spec_from_file_location(
    "_resident_sglang_terminal_fixture", _TERMINAL_FIXTURE_PATH
)
assert _TERMINAL_SPEC is not None and _TERMINAL_SPEC.loader is not None
_TERMINAL = importlib.util.module_from_spec(_TERMINAL_SPEC)
_TERMINAL_SPEC.loader.exec_module(_TERMINAL)

_SOURCE_FIXTURE_PATH = Path(__file__).with_name("test_session_reuse_authority.py")
_SOURCE_SPEC = importlib.util.spec_from_file_location(
    "_resident_sglang_source_fixture", _SOURCE_FIXTURE_PATH
)
assert _SOURCE_SPEC is not None and _SOURCE_SPEC.loader is not None
_SOURCE = importlib.util.module_from_spec(_SOURCE_SPEC)
_SOURCE_SPEC.loader.exec_module(_SOURCE)


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
    def __init__(self, backend: _Backend) -> None:
        self.backend = backend
        self.trace_configs: list[object] = []
        self.closed = False

    def get(self, *, url, headers):
        return _Response(self.backend.get(urlsplit(url).path, headers))

    def post(self, *, url, json, headers):
        return _Response(self.backend.post(urlsplit(url).path, json, headers))


class _Backend:
    def __init__(self, *, session_plan_sha256: str, traces) -> None:
        self.natives = {
            binding.execution_plan_sha256: _TERMINAL.FakeAdminTransport(
                binding=binding,
                warmup=warmup,
                scored=scored,
            )
            for binding, warmup, scored in traces
        }
        self.native_indexes = {
            execution_plan_sha256: index
            for index, execution_plan_sha256 in enumerate(self.natives)
        }
        self.source = _SOURCE._FakeSourceRuntime(
            session_plan_sha256=session_plan_sha256,
            traces=tuple(self.natives),
            adapted=False,
        )
        self.source.process_identity = "scheduler:1234"
        self.source.mutate_capability = lambda value: value.__setitem__(
            "process_started_ns", 1_000_000
        )
        self.current = None
        self.calls: list[str] = []

    async def get(self, path: str, _headers: object) -> object:
        self.calls.append(path)
        return await next(iter(self.natives.values())).get_json(path)

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
            payload = body["terminal_payload"]
            assert isinstance(payload, dict)
            self.current = self.natives[payload["execution_plan_sha256"]]
            terminal = await self.current.post_json(
                "/v1/lightcone-spec/terminal-evidence",
                {"action": "begin", "payload": payload},
            )
            terminal["reset_generation"] = (
                2 * self.native_indexes[self.current.binding.execution_plan_sha256] + 1
            )
            terminal.pop("begin_sha256")
            terminal["begin_sha256"] = canonical_sha256(terminal)
            self.current.begin_receipt = terminal
            source = {
                "schema_version": 1,
                "hook": _SOURCE.SOURCE_OWNED_SESSION_HOOK,
                "capability_sha256": body["capability_sha256"],
                "execution_plan_sha256": self.current.binding.execution_plan_sha256,
                "begin_sha256": terminal["begin_sha256"],
            }
            source["session_begin_sha256"] = canonical_sha256(source)
            return self._combined(terminal, source)
        if path.endswith("/trace/reset"):
            assert self.current is not None
            terminal = await self.current.post_json(
                "/v1/lightcone-spec/terminal-evidence",
                {"action": "reset", "payload": body["terminal_payload"]},
            )
            terminal["reset_generation"] = (
                2 * self.native_indexes[self.current.binding.execution_plan_sha256] + 2
            )
            terminal.pop("reset_sha256")
            terminal["reset_sha256"] = canonical_sha256(terminal)
            self.current.reset_receipt = terminal
            plan = self.current.binding.execution_plan_sha256
            warmup = await self.source.excluded_warmup(execution_plan_sha256=plan)
            clock = await self.source.start_scored_clock(execution_plan_sha256=plan)
            clock["native_reset_sha256"] = terminal["reset_sha256"]
            clock.pop("clock_receipt_sha256")
            clock["clock_receipt_sha256"] = canonical_sha256(clock)
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
            assert self.current is not None
            terminal = await self.current.post_json(
                "/v1/lightcone-spec/terminal-evidence",
                {"action": "finalize", "payload": body["terminal_payload"]},
            )
            plan = self.current.binding.execution_plan_sha256
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

    @staticmethod
    def _combined(terminal: object, source: object) -> dict[str, object]:
        return {
            "schema_version": 1,
            "terminal": terminal,
            "source_owned_session": source,
        }


class _TraceConfig:
    def __init__(self) -> None:
        self.on_connection_create_end = []
        self.on_connection_reuseconn = []

    def freeze(self) -> None:
        return None


class _Process:
    pid = 1234

    def __init__(self) -> None:
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


def _publish(path: Path, value: object) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _itl_pointer(expectation: TerminalRequestExpectation) -> str:
    assert expectation.output_token_ids is not None
    started_ns = time.monotonic_ns()
    events = [
        {
            "token_index": index,
            "token_id": token_id,
            "observed_ns": started_ns + index + 1,
        }
        for index, token_id in enumerate(expectation.output_token_ids)
    ]
    value = {
        "schema_version": 1,
        "kind": "sglang_native_itl_result_pointer",
        "hook": "sglang.schema_v3.native_per_token_timestamp.v2",
        "semantics": "scheduler_committed_token_at_result_processor_v1",
        "release_status": "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF",
        "request_id": expectation.request_id,
        "request_started_ns": started_ns,
        "request_terminal_ns": started_ns + len(events) + 1,
        "terminal_status": expectation.terminal_status,
        "terminal_reason": "length",
        "events": events,
    }
    value["result_pointer_sha256"] = canonical_sha256(value)
    return canonical_json_bytes(value).decode("utf-8")


def _resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    execution = _RESIDENT._execution(tmp_path, monkeypatch, member_count=2)
    policy = RegisteredServingExecutionPolicy(
        schema_version=1,
        kind="registered_serving_execution_policy",
        source_kind="scheduled",
        warmup_duration_us=1,
        arrival_duration_us=1,
        request_deadline_us=1_000_000,
        drain_duration_us=1,
        max_concurrency=1,
        complete_closed_loop_pool=False,
    )
    prepared = []
    trace_rows = []
    for index, member in enumerate(execution.plan.members):
        root = Path(member.output_directory)
        binding = effective_formal_serving_resident_terminal_binding(
            plan=execution.plan, member_index=index
        )
        warmup_request = _TERMINAL._bound_request(
            binding.warmup_request_ids[0],
            inputs=(index + 1,),
            requested_output_tokens=1,
            ordinal=0,
        )
        scored_request = _TERMINAL._bound_request(
            binding.scored_request_ids[0],
            inputs=(index + 2,),
            requested_output_tokens=2,
            ordinal=1,
        )
        warmup_expectation = _TERMINAL._server_request(
            warmup_request.request_id,
            inputs=warmup_request.input_token_ids,
            outputs=(index + 10,),
        )
        scored_expectation = _TERMINAL._server_request(
            scored_request.request_id,
            inputs=scored_request.input_token_ids,
            outputs=(index + 20, index + 21),
        )
        trace_rows.append((binding, (warmup_expectation,), (scored_expectation,)))
        plan = SimpleNamespace(
            materialized_cell_id=member.materialized_cell_id,
            private_output_root=str(root),
            terminal_output_path=str(root / "terminal.json"),
            native_itl_pointer_output_path=str(root / "itl.json"),
            lifecycle_timing_output_path=str(root / "lifecycle.json"),
            junit_output_path=str(root / "junit.xml"),
            topology_mode="tp1_dp1",
        )
        launch = SimpleNamespace(
            server_argv=("python", "-m", "sglang.launch_server", "--port", "28000"),
            localhost_port=28_000,
            gpu_uuids=execution.plan.assigned_gpu_uuids,
            inventory_sha256=member.inventory_sha256,
        )
        config = SimpleNamespace(
            runtime=SimpleNamespace(max_running_requests=1),
            model=SimpleNamespace(target="fixture-model", algorithm="dflash"),
        )
        prepared.append(
            _PreparedResidentMember(
                spec=member,
                plan=plan,
                launch=launch,
                config=config,
                schedule=SimpleNamespace(),
                warmup=(warmup_request,),
                scored=(scored_request,),
                execution_policy=policy,
                timeout_seconds=2.0,
            )
        )
    backend = _Backend(
        session_plan_sha256=execution.plan.session_plan_sha256,
        traces=tuple(trace_rows),
    )
    session = _Session(backend)
    opened = 0
    closed = 0

    async def open_session(*, total_timeout_s: float = 6 * 60 * 60):
        nonlocal opened
        assert total_timeout_s > 0
        opened += 1
        return session

    async def close_session(client_session) -> None:
        nonlocal closed
        assert client_session is session
        session.closed = True
        closed += 1

    async def unused_request(
        request_func_input,
        pbar=None,
        *,
        client_session=None,
        timeout_s=None,
    ):
        raise AssertionError("resident test phases bypass raw request callable")

    async def unused_abort(
        request_id,
        base_url,
        *,
        client_session=None,
        timeout_s=None,
    ):
        raise AssertionError("resident test phases never abort")

    transport = SessionLivePinnedBenchTransport(
        request_type=SimpleNamespace,
        request_callable=unused_request,
        abort_callable=unused_abort,
        open_session_callable=open_session,
        close_session_callable=close_session,
        set_global_args=lambda _value: None,
        trace_config_factory=_TraceConfig,
        module_identity="sglang.benchmark.serving.async_request_sglang_generate",
    )

    async def execute_phase(phase, requests, **_kwargs):
        by_id = {
            row.request_id: row
            for _binding, warmup, scored in trace_rows
            for row in (warmup if phase == "warmup" else scored)
        }
        expectations = tuple(by_id[row.request_id] for row in requests)
        return UnsignedNativeServingPhaseResult(
            phase=phase,
            requests=expectations,
            native_result_pointer_json=tuple(_itl_pointer(row) for row in expectations),
            client_lifecycle_rows=tuple(
                {"phase": phase, "request_id": row.request_id} for row in expectations
            ),
        )

    async def observe_policy(**_kwargs):
        return "{}", canonical_sha256({})

    monkeypatch.setattr(
        "lightcone_spec.orchestration.formal_serving_session_group_sglang."
        "_execute_source_owned_phase",
        execute_phase,
    )
    monkeypatch.setattr(
        "lightcone_spec.orchestration.formal_serving_session_group_sglang."
        "_observe_live_server_execution_policy",
        observe_policy,
    )
    process = _Process()

    def terminate(_process):
        assert _process is process
        process.returncode = -15
        now = time.monotonic_ns()
        return -15, "sigterm_clean", now

    monkeypatch.setattr(
        "lightcone_spec.orchestration.formal_serving_session_group_sglang."
        "_terminate_process_group",
        terminate,
    )
    monkeypatch.setattr(
        "lightcone_spec.orchestration.formal_serving_session_group_sglang.os.killpg",
        lambda _pid, _signal: None,
    )

    def snapshot(**kwargs):
        return _publish(Path(kwargs["output_path"]), {"phase": kwargs["phase"]})

    monkeypatch.setattr(
        "lightcone_spec.orchestration.formal_serving_session_group_sglang."
        "_capture_gpu_process_snapshot",
        snapshot,
    )
    evidence_root = (tmp_path / "concrete-evidence").resolve()
    evidence_root.mkdir()
    before = _publish(evidence_root / "before.json", {"phase": "before"})
    ready = _publish(evidence_root / "ready.json", {"phase": "ready"})
    streams = tuple(
        (evidence_root / name).open("xb", buffering=0)
        for name in ("server.log", "server.stdout", "server.stderr")
    )
    for stream in streams:
        stream.write(b"resident concrete fixture\n")
    group_launch_binding = _publish(
        evidence_root / "group-launch-authority.json",
        {"kind": "resident-concrete-group-launch-fixture"},
    )
    group_launch = SimpleNamespace(
        binding=group_launch_binding,
        authority=SimpleNamespace(
            actual_server_argv=prepared[0].launch.server_argv,
            host="127.0.0.1",
            port=prepared[0].launch.localhost_port,
        ),
        run_config=prepared[0].config,
        source_launch=prepared[0].launch,
    )
    return SimpleNamespace(
        execution=execution,
        prepared=tuple(prepared),
        backend=backend,
        session=session,
        opened=lambda: opened,
        closed=lambda: closed,
        transport=transport,
        process=process,
        evidence_root=evidence_root,
        before=before,
        ready=ready,
        streams=streams,
        group_launch=group_launch,
    )


def test_concrete_driver_runs_two_traces_on_one_process_and_http_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = _resources(tmp_path, monkeypatch)

    async def run():
        driver = PinnedSglangResidentProcessDriver(
            execution=resources.execution,
            prepared=resources.prepared,
            evidence_root=resources.evidence_root,
            nvidia_smi_tool=SimpleNamespace(),
            process=resources.process,
            group_launch=resources.group_launch,
            transport=resources.transport,
            before_gpu_snapshot=resources.before,
            ready_gpu_snapshot=resources.ready,
            server_ready_ns=time.monotonic_ns(),
            server_log=resources.streams[0],
            server_stdout=resources.streams[1],
            server_stderr=resources.streams[2],
        )
        await driver.wait_initial_reset_ready()
        results = []
        for index, member in enumerate(resources.execution.plan.members):
            results.append(await driver.reset_member(member=member, member_index=index))
            results.append(
                await driver.execute_trace(
                    member=member,
                    member_index=index,
                    effective_terminal_binding=(
                        effective_formal_serving_resident_terminal_binding(
                            plan=resources.execution.plan,
                            member_index=index,
                        )
                    ),
                )
            )
        close = await driver.close_session(force=False)
        return driver, results, close

    driver, results, close = asyncio.run(run())
    assert driver.process_id == close.server_process_id == 1234
    assert resources.opened() == resources.closed() == 1
    assert resources.session.closed
    assert len({id(driver.transport), id(driver.provider)}) == 2
    assert len(results) == 4
    assert close.source_close_receipt is not None
    chain = revalidate_formal_serving_resident_source_chain(
        close.source_close_receipt.absolute_path
    )
    assert len(chain.epochs) == 2
    assert chain.capability.process_identity == "scheduler:1234"
    assert tuple(row.terminal.binding.session_epoch for row in chain.epochs) == (1, 2)
    assert resources.backend.source.close_calls == 1


def test_second_reset_failure_preserves_first_trace_and_has_no_close_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = _resources(tmp_path, monkeypatch)

    def fail_second_reset(state: dict[str, object]) -> None:
        if state["reset_generation"] == 2:
            state["active_requests"] = 1

    resources.backend.source.mutate_after = fail_second_reset

    async def run():
        driver = PinnedSglangResidentProcessDriver(
            execution=resources.execution,
            prepared=resources.prepared,
            evidence_root=resources.evidence_root,
            nvidia_smi_tool=SimpleNamespace(),
            process=resources.process,
            group_launch=resources.group_launch,
            transport=resources.transport,
            before_gpu_snapshot=resources.before,
            ready_gpu_snapshot=resources.ready,
            server_ready_ns=time.monotonic_ns(),
            server_log=resources.streams[0],
            server_stdout=resources.streams[1],
            server_stderr=resources.streams[2],
        )
        await driver.wait_initial_reset_ready()
        first = resources.execution.plan.members[0]
        await driver.reset_member(member=first, member_index=0)
        trace = await driver.execute_trace(
            member=first,
            member_index=0,
            effective_terminal_binding=effective_formal_serving_resident_terminal_binding(
                plan=resources.execution.plan, member_index=0
            ),
        )
        with pytest.raises(RuntimeError, match="ended before the requested boundary"):
            await driver.reset_member(
                member=resources.execution.plan.members[1], member_index=1
            )
        close = await driver.close_session(force=True)
        return trace, close

    trace, close = asyncio.run(run())
    assert trace.raw_terminal.reopen()["terminal"]
    assert close.cleanup_kind == "forced_sigterm"
    assert close.source_close_receipt is None
    assert not (
        resources.evidence_root / "source-chain" / "resident-source-chain-manifest.json"
    ).exists()
    assert resources.opened() == resources.closed() == 1
    # The source attempts a terminal close, but its incomplete chain is
    # rejected and therefore never becomes resident close authority.
    assert resources.backend.source.close_calls == 1
    assert resources.process.returncode == -15
