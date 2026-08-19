from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.orchestration import (
    formal_serving_session_group_physical as resident_physical,
)
from lightcone_spec.orchestration.formal_serving_session_group import (
    partition_formal_serving_session_groups,
)
from lightcone_spec.orchestration.formal_serving_session_group_physical import (
    FormalServingResidentCloseEvidence,
    FormalServingResidentResetEvidence,
    FormalServingResidentSharedSessionHandle,
    FormalServingResidentTraceEvidence,
    effective_formal_serving_resident_terminal_binding,
    revalidate_formal_serving_resident_reset_boundary_receipt,
    revalidate_formal_serving_resident_shared_close_receipt,
    revalidate_formal_serving_resident_shared_launch_receipt,
    revalidate_formal_serving_resident_trace_receipt,
)
from lightcone_spec.orchestration.formal_serving_session_group_worker import (
    FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
    FormalServingResidentFinalizedMemberResult,
    FormalServingSessionClassifiedFailure,
    FormalServingSessionGroupCellArtifact,
    FormalServingSessionGroupExecutionSpec,
    FormalServingSessionMemberPhysicalResult,
    execute_formal_serving_session_group,
    formal_serving_session_failure_class,
    revalidate_formal_serving_session_group_execution,
)
from lightcone_spec.orchestration.live_sglang import (
    PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
)
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalRunBinding,
    canonical_sha256,
)
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_FIXTURE_PATH = Path(__file__).with_name("test_formal_serving_session_group.py")
_SPEC = importlib.util.spec_from_file_location("_resident_group_fixture", _FIXTURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GROUP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GROUP)
_REPOSITORY_ROOT = Path(__file__).parents[1].resolve()


def _sha(value: str) -> str:
    return canonical_sha256(value)


def _publish(path: Path, value: object) -> CanonicalJsonProofBinding:
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _raw_binding(path: Path, value: object, *, label: str) -> EvidenceFileBinding:
    binding = _publish(path, value)
    return EvidenceFileBinding.bind(Path(binding.absolute_path), label=label)


def _patch_test_source_chain(
    monkeypatch: pytest.MonkeyPatch,
    *,
    execution,
    driver: _Driver,
) -> None:
    """Keep the fake process focused while production still deep-reopens a chain."""

    from lightcone_spec.orchestration import formal_serving_session_source_chain

    def revalidate(_path: str | Path) -> SimpleNamespace:
        traces = tuple(
            revalidate_formal_serving_resident_trace_receipt(
                member.run_plan.reopen()["live_run_receipt_output_path"]
            )[1]
            for member in execution.plan.members[: len(driver.trace_calls)]
        )
        return SimpleNamespace(
            session_plan_sha256=execution.plan.session_plan_sha256,
            capability=SimpleNamespace(
                process_identity=f"scheduler:{driver.process_id}",
                process_started_ns=driver.process_started_ns,
            ),
            execution_plan_sha256s=tuple(
                trace.effective_terminal_binding.execution_plan_sha256
                for trace in traces
            ),
            epochs=tuple(
                SimpleNamespace(terminal_artifact=trace.raw_terminal)
                for trace in traces
            ),
        )

    monkeypatch.setattr(
        formal_serving_session_source_chain,
        "revalidate_formal_serving_resident_source_chain",
        revalidate,
    )


def _native(index: int) -> NativeTerminalRunBinding:
    value = NativeTerminalRunBinding(
        run_id=f"resident-run-{index}",
        run_nonce_sha256=_sha(f"nonce-{index}"),
        execution_plan_sha256=_sha(f"execution-{index}"),
        rank_config_sha256=_sha(f"rank-{index}"),
        attempt_id="attempt-1",
        session_id=f"registered-cell-{index}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=_sha(f"challenge-{index}"),
        method="static",
        warmup_request_ids=(f"warm-{index}",),
        scored_request_ids=(f"score-{index}",),
    )
    value.validate()
    return value


def _actual_server_argv(execution) -> tuple[str, ...]:
    assert execution.plan.normalized_process_key is not None
    return tuple(
        "127.0.0.1"
        if value == "<GROUP_HOST>"
        else "28000"
        if value == "<GROUP_PORT>"
        else value
        for value in execution.plan.normalized_process_key.normalized_server_argv
    )


def _native_dict(value: NativeTerminalRunBinding) -> dict[str, object]:
    return {
        "run_id": value.run_id,
        "run_nonce_sha256": value.run_nonce_sha256,
        "execution_plan_sha256": value.execution_plan_sha256,
        "rank_config_sha256": value.rank_config_sha256,
        "attempt_id": value.attempt_id,
        "session_id": value.session_id,
        "session_epoch": value.session_epoch,
        "previous_run_id": value.previous_run_id,
        "challenge_nonce_sha256": value.challenge_nonce_sha256,
        "method": value.method,
        "warmup_request_ids": list(value.warmup_request_ids),
        "scored_request_ids": list(value.scored_request_ids),
    }


def _execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    member_count: int,
):
    from lightcone_spec.orchestration import formal_serving_session_group_launch

    def revalidate_test_group_launch(path: str | Path) -> SimpleNamespace:
        binding = CanonicalJsonProofBinding.bind(path)
        value = binding.reopen()
        assert isinstance(value, dict)
        return SimpleNamespace(
            binding=binding,
            authority=SimpleNamespace(
                group_plan=CanonicalJsonProofBinding.from_dict(value["group_plan"]),
                actual_server_argv=tuple(value["actual_server_argv"]),
                port=value["port"],
            ),
        )

    monkeypatch.setattr(
        formal_serving_session_group_launch,
        "revalidate_formal_serving_resident_group_launch_authority",
        revalidate_test_group_launch,
    )
    monkeypatch.setattr(
        resident_physical,
        "_local_process_identity_matches",
        lambda **_identity: True,
    )
    authority_binding, authority = _GROUP._published_authority(
        tmp_path / "authority",
        monkeypatch,
        method_family="static",
    )
    specs = []
    for index in range(member_count):
        config = _GROUP._config(label=f"resident-{index}")
        launch = _GROUP._producer_generated_launch(
            tmp_path,
            label=f"resident-{index}",
            config=config,
            port=28000 + index,
        )
        spec = _GROUP._group_spec(
            tmp_path,
            index=500 + index,
            config=config,
            launch=launch,
            method_family="static",
            source_snapshot_sha256=authority.source_snapshot_sha256,
            protocol_lock_sha256=authority.protocol_lock_sha256,
            inventory_sha256=authority.inventory_sha256,
        )
        run_root = (tmp_path / f"member-{index}").resolve()
        run_root.mkdir()
        plan = _publish(
            run_root / "formal-serving-run-plan.json",
            {
                "native_terminal_binding": _native_dict(_native(index)),
                "live_run_receipt_output_path": str(
                    run_root / "unsigned-live-run-receipt.json"
                ),
            },
        )
        specs.append(replace(spec, run_plan=plan, output_directory=str(run_root)))
    plans = partition_formal_serving_session_groups(
        specs,
        reset_authorities=(authority_binding,),
        max_member_count=member_count,
        max_estimated_duration_seconds=float(member_count * 20),
    )
    assert len(plans) == 1 and plans[0].execution_mode == "shared_session_tp1"
    plan_path = (tmp_path / "group-plan.json").resolve()
    plan_binding = _publish(plan_path, plans[0].to_dict())
    execution_spec = FormalServingSessionGroupExecutionSpec(
        schema_version=1,
        kind="formal_serving_session_group_execution_spec",
        protocol_sha256=FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
        group_plan_path=str(plan_path),
        reset_authority_path=authority_binding.absolute_path,
        output_directory=str((tmp_path / "execution-output").resolve()),
        formal_measured=False,
    )
    spec_path = (tmp_path / "execution-spec.json").resolve()
    spec_binding = _publish(spec_path, execution_spec.to_dict())
    execution = revalidate_formal_serving_session_group_execution(spec_path)
    assert execution.plan_binding == plan_binding
    assert execution.spec_binding == spec_binding
    return execution


class _Driver:
    def __init__(
        self,
        root: Path,
        *,
        group_plan: CanonicalJsonProofBinding,
        inventory_sha256: str,
        gpu_uuid: str,
        orphan_on_close: bool = False,
        fail_reset_index: int | None = None,
        fail_trace_index: int | None = None,
        trace_error: Exception | None = None,
        actual_server_argv: tuple[str, ...] | None = None,
    ) -> None:
        self.root = root
        self.group_plan = group_plan
        self.orphan_on_close = orphan_on_close
        self.fail_reset_index = fail_reset_index
        self.fail_trace_index = fail_trace_index
        self.trace_error = trace_error
        self._actual_server_argv = actual_server_argv
        self._group_launch_authority: CanonicalJsonProofBinding | None = None
        self.inventory_sha256 = inventory_sha256
        self.gpu_uuid = gpu_uuid
        self.reset_calls: list[int] = []
        self.trace_calls: list[int] = []
        self.close_calls: list[bool] = []
        self._nvidia_smi = EvidenceFileBinding.bind(
            Path(__file__).resolve(), label="resident test nvidia-smi identity"
        )
        self._before = self._gpu_snapshot(
            root / "before.json", phase="before", captured_ns=90
        )
        self._ready = self._gpu_snapshot(
            root / "ready.json", phase="ready", captured_ns=120
        )

    def _gpu_snapshot(
        self, path: Path, *, phase: str, captured_ns: int
    ) -> CanonicalJsonProofBinding:
        ready = phase == "ready"
        return _publish(
            path,
            {
                "schema_version": 1,
                "kind": "unsigned_pinned_sglang_gpu_process_snapshot",
                "protocol_sha256": PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
                "phase": phase,
                "captured_ns": captured_ns,
                "inventory_sha256": self.inventory_sha256,
                "gpu_uuids": [self.gpu_uuid],
                "server_process_group_ids": [self.process_group_id] if ready else None,
                "nvidia_smi": {
                    "executable_path": self._nvidia_smi.absolute_path,
                    "executable_raw_sha256": self._nvidia_smi.raw_sha256,
                    "executable_size": self._nvidia_smi.size,
                },
                "gpu_rows": [
                    {
                        "uuid": self.gpu_uuid,
                        "name": "resident test GPU",
                        "memory_used_mib": 1_024 if ready else 0,
                    }
                ],
                "compute_process_rows": (
                    [
                        {
                            "gpu_uuid": self.gpu_uuid,
                            "pid": self.process_id,
                            "process_group_id": self.process_group_id,
                            "used_gpu_memory_mib": 1_024,
                        }
                    ]
                    if ready
                    else []
                ),
            },
        )

    @property
    def process_id(self) -> int:
        return 7001

    @property
    def process_group_id(self) -> int:
        return 7001

    @property
    def process_started_ns(self) -> int:
        return 100

    @property
    def ready_ns(self) -> int:
        return 120

    @property
    def actual_server_argv(self) -> tuple[str, ...]:
        if self._actual_server_argv is None:
            raise RuntimeError("test resident driver lacks actual argv")
        return self._actual_server_argv

    @property
    def group_launch_authority(self) -> CanonicalJsonProofBinding:
        if self._group_launch_authority is None:
            self._group_launch_authority = _publish(
                self.root / "test-group-launch-authority.json",
                {
                    "group_plan": self.group_plan.to_dict(),
                    "actual_server_argv": list(self.actual_server_argv),
                    "port": 28_000,
                },
            )
        return self._group_launch_authority

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:28000"

    @property
    def before_gpu_snapshot(self) -> CanonicalJsonProofBinding:
        return self._before

    @property
    def ready_gpu_snapshot(self) -> CanonicalJsonProofBinding:
        return self._ready

    @property
    def server_log_path(self) -> str:
        return str((self.root / "server.log.json").resolve())

    @property
    def server_stdout_path(self) -> str:
        return str((self.root / "stdout.json").resolve())

    @property
    def server_stderr_path(self) -> str:
        return str((self.root / "stderr.json").resolve())

    async def reset_member(self, *, member, member_index):
        self.reset_calls.append(member_index)
        if member_index == self.fail_reset_index:
            raise RuntimeError("injected resident reset failure")
        return FormalServingResidentResetEvidence(
            source_reset_receipt=_publish(
                self.root / f"source-reset-{member_index}.json",
                {"cell": member.materialized_cell_id, "epoch": member_index + 1},
            ),
            reset_started_ns=200 + member_index * 100,
            reset_finished_ns=210 + member_index * 100,
            hbm_allocated_bytes=1_000,
            request_queue_empty=True,
            optimizer_state_reset=True,
            adaptation_state_reset=True,
            candidate_state_reset=True,
            cache_policy_restored=True,
            terminal_writer_flushed=True,
            previous_requests_fully_terminal=True,
        )

    async def execute_trace(self, *, member, member_index, effective_terminal_binding):
        self.trace_calls.append(member_index)
        if member_index == self.fail_trace_index:
            raise self.trace_error or RuntimeError(
                "injected resident scientific failure"
            )
        trace_root = Path(member.output_directory)
        return FormalServingResidentTraceEvidence(
            effective_terminal_binding=effective_terminal_binding,
            raw_terminal=_publish(
                trace_root / "terminal.json", {"member": member_index}
            ),
            native_itl=_publish(trace_root / "itl.json", {"member": member_index}),
            client_lifecycle=_publish(
                trace_root / "client.json", {"member": member_index}
            ),
            junit=_raw_binding(
                trace_root / "junit.xml",
                {"member": member_index},
                label="resident test JUnit",
            ),
            trace_lifecycle=_publish(
                trace_root / "lifecycle.json", {"member": member_index}
            ),
            trace_started_ns=220 + member_index * 100,
            scored_started_ns=230 + member_index * 100,
            trace_finished_ns=250 + member_index * 100,
        )

    async def close_session(self, *, force: bool):
        self.close_calls.append(force)
        if self.orphan_on_close:
            raise RuntimeError("process group still alive")
        close_started_ns = 1_000 + len(self.trace_calls) * 100
        source = None
        if not force:
            source = _publish(self.root / "source-close.json", {"closed": True})
        after = self._gpu_snapshot(
            self.root / "after.json",
            phase="after",
            captured_ns=close_started_ns + 25,
        )
        log = _raw_binding(
            self.root / "server.log.json", {"sealed": True}, label="server log"
        )
        stdout = _raw_binding(
            self.root / "stdout.json", {"sealed": True}, label="server stdout"
        )
        stderr = _raw_binding(
            self.root / "stderr.json", {"sealed": True}, label="server stderr"
        )
        return FormalServingResidentCloseEvidence(
            source_close_receipt=source,
            server_process_id=self.process_id,
            server_process_group_id=self.process_group_id,
            close_started_ns=close_started_ns,
            process_exited_ns=close_started_ns + 10,
            process_exit_code=-15,
            process_group_empty=True,
            process_group_empty_checked_ns=close_started_ns + 20,
            evidence_flush_completed_ns=close_started_ns + 30,
            cleanup_kind=("forced_sigterm" if force else "source_close_sigterm"),
            after_gpu_snapshot=after,
            server_log=log,
            server_stdout=stdout,
            server_stderr=stderr,
        )


@pytest.mark.parametrize("member_count", (2, 32))
def test_resident_physical_chain_uses_one_process_and_seals_ordered_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_count: int,
) -> None:
    execution = _execution(tmp_path, monkeypatch, member_count=member_count)
    evidence_root = (tmp_path / "physical").resolve()
    evidence_root.mkdir()
    assert execution.plan.normalized_process_key is not None
    driver = _Driver(
        evidence_root,
        group_plan=execution.plan_binding,
        inventory_sha256=execution.plan.members[0].inventory_sha256,
        gpu_uuid=execution.plan.assigned_gpu_uuids[0],
        actual_server_argv=_actual_server_argv(execution),
    )
    _patch_test_source_chain(
        monkeypatch,
        execution=execution,
        driver=driver,
    )
    handle = FormalServingResidentSharedSessionHandle(
        execution=execution,
        driver=driver,
        evidence_root=evidence_root,
        repository_root=_REPOSITORY_ROOT,
    )

    async def run():
        traces = []
        prior = None
        for index, member in enumerate(execution.plan.members):
            reset = await handle.reset_for_member(
                session_plan_sha256=execution.plan.session_plan_sha256,
                reset_authority_sha256=execution.authority.sha256,
                prior_member=prior,
                next_member=member,
                session_epoch=index + 1,
            )
            traces.append(
                (
                    reset,
                    await handle.execute_member(member=member, session_epoch=index + 1),
                )
            )
            prior = member
        with pytest.raises(ValueError, match="sealed close"):
            await handle.finalize_resident_member(
                member=execution.plan.members[0],
                trace=traces[0][1],
                shared_close_receipt=traces[0][0],
            )
        close = await handle.close()
        return traces, close

    traces, close_binding = asyncio.run(run())
    launch_binding, launch = revalidate_formal_serving_resident_shared_launch_receipt(
        evidence_root / "shared-launch.json"
    )
    close_rebound, close = revalidate_formal_serving_resident_shared_close_receipt(
        close_binding.absolute_path
    )
    assert close_rebound == close_binding
    assert close.shared_launch == launch_binding
    assert close.server_process_id == launch.server_process_id == 7001
    assert close.process_group_empty is True
    assert len(close.member_trace_receipts) == member_count
    assert driver.reset_calls == driver.trace_calls == list(range(member_count))
    for index, (reset_binding, trace_result) in enumerate(traces):
        _reset_binding, reset = (
            revalidate_formal_serving_resident_reset_boundary_receipt(
                reset_binding.absolute_path
            )
        )
        _trace_binding, trace = revalidate_formal_serving_resident_trace_receipt(
            trace_result.trace_receipt.absolute_path
        )
        expected = effective_formal_serving_resident_terminal_binding(
            plan=execution.plan, member_index=index
        )
        assert reset.session_epoch == trace.session_epoch == index + 1
        assert trace.effective_terminal_binding == expected
        assert trace.process_id == 7001


def test_orphan_process_group_never_publishes_a_shared_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution(tmp_path, monkeypatch, member_count=2)
    evidence_root = (tmp_path / "orphan").resolve()
    evidence_root.mkdir()
    handle = FormalServingResidentSharedSessionHandle(
        execution=execution,
        driver=_Driver(
            evidence_root,
            group_plan=execution.plan_binding,
            inventory_sha256=execution.plan.members[0].inventory_sha256,
            gpu_uuid=execution.plan.assigned_gpu_uuids[0],
            orphan_on_close=True,
            actual_server_argv=_actual_server_argv(execution),
        ),
        evidence_root=evidence_root,
        repository_root=_REPOSITORY_ROOT,
    )

    async def run() -> None:
        with pytest.raises(RuntimeError, match="still alive"):
            await handle.force_close()

    asyncio.run(run())
    assert not (evidence_root / "shared-close.json").exists()


def test_driver_cannot_replace_reset_booleans_or_pg_empty_with_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution(tmp_path, monkeypatch, member_count=2)
    evidence_root = (tmp_path / "claimed-empty").resolve()
    evidence_root.mkdir()
    driver = _Driver(
        evidence_root,
        group_plan=execution.plan_binding,
        inventory_sha256=execution.plan.members[0].inventory_sha256,
        gpu_uuid=execution.plan.assigned_gpu_uuids[0],
        actual_server_argv=_actual_server_argv(execution),
    )
    reset = asyncio.run(
        driver.reset_member(member=execution.plan.members[0], member_index=0)
    )
    with pytest.raises(TypeError, match="not boolean"):
        replace(reset, request_queue_empty=1)

    handle = FormalServingResidentSharedSessionHandle(
        execution=execution,
        driver=driver,
        evidence_root=evidence_root,
        repository_root=_REPOSITORY_ROOT,
    )
    monkeypatch.setattr(
        resident_physical,
        "_local_process_group_exists",
        lambda _process_group_id: True,
    )
    with pytest.raises(RuntimeError, match="not proven empty"):
        asyncio.run(handle.force_close())
    assert not (evidence_root / "shared-close.json").exists()


class _ManifestlessHandle:
    """Keep worker tests independent from the full prepared-run fixture."""

    def __init__(self, wrapped: FormalServingResidentSharedSessionHandle, root: Path):
        self.wrapped = wrapped
        self.root = root
        self.closed: CanonicalJsonProofBinding | None = None

    @property
    def process_id(self) -> int:
        return self.wrapped.process_id

    async def reset_for_member(self, **kwargs):
        return await self.wrapped.reset_for_member(**kwargs)

    async def execute_member(self, **kwargs):
        return await self.wrapped.execute_member(**kwargs)

    async def close(self):
        self.closed = await self.wrapped.close()
        return self.closed

    async def force_close(self):
        self.closed = await self.wrapped.force_close()
        return self.closed

    async def finalize_resident_member(self, *, member, trace, shared_close_receipt):
        assert self.closed == shared_close_receipt
        pointer = _publish(
            self.root / f"manifest-{member.materialized_cell_id}.json",
            {
                "kind": "test-resident-manifest",
                "cell": member.materialized_cell_id,
                "trace": trace.trace_receipt.to_dict(),
                "shared_close": shared_close_receipt.to_dict(),
            },
        )
        return FormalServingResidentFinalizedMemberResult(
            process_id=self.process_id,
            started_ns=trace.started_ns,
            finished_ns=trace.finished_ns,
            result_pointer=pointer,
        )


class _WorkerRuntime:
    def __init__(self, execution, root: Path, driver: _Driver) -> None:
        assert execution.plan.normalized_process_key is not None
        driver._actual_server_argv = _actual_server_argv(execution)
        wrapped = FormalServingResidentSharedSessionHandle(
            execution=execution,
            driver=driver,
            evidence_root=root / "resident",
            repository_root=_REPOSITORY_ROOT,
        )
        self.handle = _ManifestlessHandle(wrapped, root / "manifests")
        self.fresh: list[str] = []
        self.root = root

    async def start_shared_session(self, *, execution):
        return self.handle

    async def execute_fresh_member(self, *, member, fallback_reason):
        self.fresh.append(member.materialized_cell_id)
        pointer = _publish(
            self.root / "fresh" / f"{member.materialized_cell_id}.json",
            {"cell": member.materialized_cell_id, "reason": fallback_reason},
        )
        return FormalServingSessionMemberPhysicalResult(
            status="COMPLETE",
            process_id=8_000 + len(self.fresh),
            started_ns=2_000 + len(self.fresh) * 10,
            finished_ns=2_005 + len(self.fresh) * 10,
            exit_code=0,
            result_pointer=pointer,
            failure_code=None,
        )


@pytest.mark.parametrize(
    ("fail_reset", "fail_trace", "expected_modes", "expected_status"),
    (
        (
            1,
            None,
            ("shared_session_tp1", "fresh_process_fallback", "fresh_process_fallback"),
            "COMPLETE",
        ),
        (
            None,
            1,
            ("shared_session_tp1", "failed", "fresh_process_fallback"),
            "PARTIAL",
        ),
    ),
)
def test_worker_seals_resident_prefix_then_uses_fresh_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_reset: int | None,
    fail_trace: int | None,
    expected_modes: tuple[str, ...],
    expected_status: str,
) -> None:
    execution = _execution(tmp_path, monkeypatch, member_count=3)
    runtime_root = (tmp_path / "worker-runtime").resolve()
    runtime_root.mkdir()
    driver = _Driver(
        runtime_root,
        group_plan=execution.plan_binding,
        inventory_sha256=execution.plan.members[0].inventory_sha256,
        gpu_uuid=execution.plan.assigned_gpu_uuids[0],
        fail_reset_index=fail_reset,
        fail_trace_index=fail_trace,
    )
    runtime = _WorkerRuntime(execution, runtime_root, driver)
    result = asyncio.run(
        execute_formal_serving_session_group(
            execution_spec_path=execution.spec_binding.absolute_path,
            runtime=runtime,
        )
    )
    artifacts = tuple(
        FormalServingSessionGroupCellArtifact.from_dict(item.reopen())
        for item in result.cell_artifacts
    )
    assert result.status == expected_status
    assert tuple(item.execution_mode for item in artifacts) == expected_modes
    assert driver.close_calls == [True]
    assert runtime.handle.closed is not None
    assert all(
        item.result_pointer is None
        or item.result_pointer.reopen().get("shared_close")
        == runtime.handle.closed.to_dict()
        or item.execution_mode == "fresh_process_fallback"
        for item in artifacts
    )
    with pytest.raises(FileExistsError):
        asyncio.run(
            execute_formal_serving_session_group(
                execution_spec_path=execution.spec_binding.absolute_path,
                runtime=runtime,
            )
        )


@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (OSError("resident server disappeared"), "INFRASTRUCTURE"),
        (ConnectionError("resident transport closed"), "INFRASTRUCTURE"),
        (
            FormalServingSessionClassifiedFailure(
                "EXACTNESS", "TOKEN_TRAJECTORY_MISMATCH"
            ),
            "EXACTNESS",
        ),
        (
            FormalServingSessionClassifiedFailure("UNSAFE", "NONFINITE_CANDIDATE"),
            "UNSAFE",
        ),
        (RuntimeError("untyped scientific failure"), "SCIENTIFIC"),
    ),
)
def test_resident_failure_classification_retries_only_infrastructure(
    error: Exception,
    expected: str,
) -> None:
    assert formal_serving_session_failure_class(error) == expected


def test_resident_transport_failure_is_typed_for_fresh_retry_without_regroup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution(tmp_path, monkeypatch, member_count=3)
    runtime_root = (tmp_path / "worker-runtime-infrastructure").resolve()
    runtime_root.mkdir()
    driver = _Driver(
        runtime_root,
        group_plan=execution.plan_binding,
        inventory_sha256=execution.plan.members[0].inventory_sha256,
        gpu_uuid=execution.plan.assigned_gpu_uuids[0],
        fail_trace_index=1,
        trace_error=ConnectionError("resident HTTP pool closed"),
    )
    runtime = _WorkerRuntime(execution, runtime_root, driver)

    result = asyncio.run(
        execute_formal_serving_session_group(
            execution_spec_path=execution.spec_binding.absolute_path,
            runtime=runtime,
        )
    )
    artifacts = tuple(
        FormalServingSessionGroupCellArtifact.from_dict(item.reopen())
        for item in result.cell_artifacts
    )

    assert result.status == "PARTIAL"
    assert artifacts[0].status == "COMPLETE"
    assert artifacts[1].status == "FAILED"
    assert artifacts[1].failure_class == "INFRASTRUCTURE"
    assert artifacts[2].execution_mode == "fresh_process_fallback"
    assert artifacts[2].status == "COMPLETE"
