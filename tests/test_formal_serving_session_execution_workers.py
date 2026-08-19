from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.orchestration import (
    formal_single_operator_session_reset_physical as physical,
)
from lightcone_spec.orchestration.formal_serving_session_group import (
    partition_formal_serving_session_groups,
)
from lightcone_spec.orchestration.formal_serving_session_group_worker import (
    FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
    FormalServingSessionGroupCellArtifact,
    FormalServingSessionGroupExecutionResult,
    FormalServingSessionGroupExecutionSpec,
    FormalServingSessionMemberPhysicalResult,
    FormalServingSessionResetFailed,
    execute_formal_serving_session_group,
    publish_formal_serving_session_group_execution_spec,
)
from lightcone_spec.orchestration.formal_single_operator_session_reset_physical import (
    TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PHYSICAL_PLAN_PROTOCOL_SHA256,
    TrustedEmpiricalTp1SessionResetLiveResources,
    TrustedEmpiricalTp1SessionResetPhysicalPlan,
    execute_trusted_empirical_tp1_session_reset_qualification,
    publish_trusted_empirical_tp1_session_reset_physical_plan,
)
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalRunBinding,
    TerminalRequestExpectation,
    canonical_sha256,
)
from lightcone_spec.orchestration.session_live_runtime import (
    SessionLiveStepBinding,
    SessionLiveTraceInput,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_GROUP_FIXTURE_PATH = Path(__file__).with_name("test_formal_serving_session_group.py")
_GROUP_SPEC = importlib.util.spec_from_file_location(
    "_formal_session_group_fixture", _GROUP_FIXTURE_PATH
)
assert _GROUP_SPEC is not None and _GROUP_SPEC.loader is not None
_GROUP = importlib.util.module_from_spec(_GROUP_SPEC)
_GROUP_SPEC.loader.exec_module(_GROUP)

_LIVE_FIXTURE_PATH = Path(__file__).with_name("test_session_live_runtime.py")
_LIVE_SPEC = importlib.util.spec_from_file_location(
    "_formal_session_live_fixture", _LIVE_FIXTURE_PATH
)
assert _LIVE_SPEC is not None and _LIVE_SPEC.loader is not None
_LIVE = importlib.util.module_from_spec(_LIVE_SPEC)
_LIVE_SPEC.loader.exec_module(_LIVE)


def _sha(label: str) -> str:
    return canonical_sha256(label)


def _publish(path: Path, value: object) -> CanonicalJsonProofBinding:
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _two_trace_inputs(plan) -> tuple[SessionLiveTraceInput, SessionLiveTraceInput]:
    traces = []
    prior = None
    for index, execution_plan in enumerate(plan.execution_plan_sha256s):
        warmup = (
            TerminalRequestExpectation(
                request_id=f"warm-{index}",
                input_token_ids=(1,),
                output_token_ids=(2,),
                terminal_status="completed",
                terminal_reason="FINISH_LENGTH",
                submitted_to_server=True,
            ),
        )
        scored = (
            TerminalRequestExpectation(
                request_id=f"score-{index}",
                input_token_ids=(3,),
                output_token_ids=(4, 5),
                terminal_status="completed",
                terminal_reason="FINISH_LENGTH",
                submitted_to_server=True,
            ),
        )
        binding = NativeTerminalRunBinding(
            run_id=f"qualification-run-{index}",
            run_nonce_sha256=_sha(f"nonce-{index}"),
            execution_plan_sha256=execution_plan,
            rank_config_sha256=_sha(f"rank-{index}"),
            attempt_id=f"qualification-attempt-{index}",
            session_id=plan.qualification_run_id,
            session_epoch=index + 1,
            previous_run_id=prior,
            challenge_nonce_sha256=_sha(f"challenge-{index}"),
            method=plan.traces[index].method,
            warmup_request_ids=(f"warm-{index}",),
            scored_request_ids=(f"score-{index}",),
        )
        traces.append(
            SessionLiveTraceInput(
                binding=binding,
                driver=_LIVE._Driver(warmup=warmup, scored=scored, events=[]),
            )
        )
        prior = binding.run_id
    return traces[0], traces[1]


class _QualificationRuntime:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.resources = None

    async def launch(
        self,
        *,
        plan,
        evidence_sink,
        native_timestamp_evidence_paths,
    ):
        if self.fail:
            raise RuntimeError("injected qualification failure")
        traces = _two_trace_inputs(plan)
        for index, (trace, path) in enumerate(
            zip(traces, native_timestamp_evidence_paths, strict=True)
        ):
            request = trace.driver.scored[0]
            publish_canonical_json_no_replace(
                path,
                {
                    "schema_version": 1,
                    "kind": ("trusted_empirical_tp1_session_reset_native_timestamps"),
                    "qualification_run_id": plan.qualification_run_id,
                    "execution_plan_sha256": trace.binding.execution_plan_sha256,
                    "requests": [
                        {
                            "request_id": request.request_id,
                            "output_token_ids": list(request.output_token_ids),
                            "native_token_timestamps_ns": [110 + index, 120 + index],
                            "request_started_ns": 100,
                            "request_terminal_ns": 130,
                        }
                    ],
                },
            )
        _plan, _trace, transport, provider, owner, _backend, _events = _LIVE._resources(
            evidence_sink=evidence_sink
        )
        self.resources = TrustedEmpiricalTp1SessionResetLiveResources(
            server_pid=1234,
            base_url="http://127.0.0.1:21000",
            transport=transport,
            provider=provider,
            process_owner=owner,
            traces=traces,
            native_timestamp_evidence_paths=native_timestamp_evidence_paths,
        )
        return self.resources


def _clean_state(*, generation: int) -> dict[str, object]:
    return {
        "process_identity": "server-1234",
        "session_epoch": 2,
        "reset_generation": generation,
        "active_requests": 0,
        "queued_requests": 0,
        "request_kv_entries": 0,
        "prefix_entries": 0,
        "prefix_policy": "clear",
        "registered_prefix_sha256": None,
        "rng_sha256": _sha("rng"),
        "counters_sha256": _sha("counters"),
        "cuda_peaks_sha256": _sha("peaks"),
        "scheduler_statistics_sha256": _sha("scheduler"),
        "telemetry_sha256": _sha("telemetry"),
        "inference_weights_sha256": _sha("weights"),
        "fp32_master_sha256": None,
        "optimizer_moments_sha256": None,
        "candidate_buffers_sha256": None,
        "adapter_version": 0,
        "optimizer_generation": 0,
        "cohort_state_sha256": None,
        "update_counter": 0,
        "allocator_allocated_bytes": 10_000,
        "allocator_reserved_bytes": 20_000,
        "hbm_state_sha256": _sha("hbm"),
        "completion_event_generation": generation,
        "completion_event_complete": True,
        "completion_event_sha256": _sha(f"event-{generation}"),
        "connection_accounting": {
            "process_id": 1234,
            "generation": 1,
            "connections_created": 1,
            "connections_closed": 0,
            "connections_current": 1,
        },
    }


def _terminal(trace: SessionLiveTraceInput):
    return SimpleNamespace(
        binding=trace.binding,
        requests=tuple(trace.driver.scored),
        terminal_sha256=_sha(f"terminal-{trace.binding.run_id}"),
        begin_receipt=SimpleNamespace(
            server_process_id=1234,
            server_process_started_ns=1_000_000,
        ),
        reset_receipt=SimpleNamespace(
            server_process_id=1234,
            server_process_started_ns=1_000_000,
        ),
    )


async def _fake_live_contract(**kwargs):
    assert kwargs["verified_gpu_proof"] is None
    resources = kwargs["transport"]
    traces = tuple(kwargs["traces"])
    steps = (
        SessionLiveStepBinding.capture(
            step="session_initial_state",
            value={"state": _clean_state(generation=0)},
        ),
        SessionLiveStepBinding.capture(
            step="session_reset_boundary",
            execution_plan_sha256=traces[0].binding.execution_plan_sha256,
            value={"after": _clean_state(generation=1)},
        ),
        SessionLiveStepBinding.capture(
            step="session_reset_boundary",
            execution_plan_sha256=traces[1].binding.execution_plan_sha256,
            value={"after": _clean_state(generation=2)},
        ),
    )
    audit = SimpleNamespace(
        status="CPU_CONTRACT_ONLY",
        reuse_authorized=False,
        reset_receipt_sha256s=(_sha("reset-1"), _sha("reset-2")),
        sha256=_sha("audit"),
    )
    result = SimpleNamespace(
        audit=audit,
        reuse_authorized=False,
        native_terminals=tuple(_terminal(trace) for trace in traces),
        steps=steps,
        transport_closed=True,
        process_closed=True,
        process_force_closed=False,
    )
    for step in steps:
        resources._evidence_sink.record_step(step)
    resources._evidence_sink.finalize(result)
    return result


def _physical_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    authority_binding, authority = _GROUP._published_authority(
        tmp_path / "authority-source",
        monkeypatch,
        method_family="static",
    )
    configs = (_GROUP._config(label="qual-a"), _GROUP._config(label="qual-b"))
    launches = tuple(
        _GROUP._producer_generated_launch(
            tmp_path,
            label=f"qual-{index}",
            config=config,
            port=25000 + index,
        )
        for index, config in enumerate(configs)
    )
    monkeypatch.setattr(
        CompileLaunchManifest,
        "sha256",
        property(lambda self: _sha(f"test-trusted-launch:{self.server_argv_sha256}")),
    )
    schedule = _sha("qualification-shared-schedule")
    specs = tuple(
        replace(
            _GROUP._group_spec(
                tmp_path,
                index=100 + index,
                config=config,
                launch=launch,
                method_family="static",
                source_snapshot_sha256=authority.source_snapshot_sha256,
                protocol_lock_sha256=authority.protocol_lock_sha256,
                inventory_sha256=authority.inventory_sha256,
            ),
            compile_launch_manifest_sha256=launch.sha256,
            request_schedule_sha256=schedule,
        )
        for index, (config, launch) in enumerate(zip(configs, launches, strict=True))
    )
    spec_paths = []
    launch_paths = []
    launch_by_path = {}
    for index, (spec, launch) in enumerate(zip(specs, launches, strict=True)):
        spec_path = (tmp_path / f"qualification-trace-{index}.json").resolve()
        _publish(spec_path, spec.to_dict())
        spec_paths.append(str(spec_path))
        launch_path = (tmp_path / f"qualification-launch-{index}.json").resolve()
        _publish(launch_path, {"fixture_launch": index})
        launch_paths.append(str(launch_path))
        launch_by_path[str(launch_path)] = launch
    monkeypatch.setattr(
        CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, path: launch_by_path[str(path)]),
    )
    qualification = authority.qualification_spec.reopen()
    plan = TrustedEmpiricalTp1SessionResetPhysicalPlan(
        schema_version=1,
        kind="trusted_empirical_tp1_session_reset_physical_plan",
        protocol_sha256=(
            TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PHYSICAL_PLAN_PROTOCOL_SHA256
        ),
        protocol_lock_path=qualification["protocol_lock_path"],
        content_bundle_path=qualification["content_bundle_path"],
        inventory_path=qualification["inventory_path"],
        trace_spec_paths=(spec_paths[0], spec_paths[1]),
        compile_launch_manifest_paths=(launch_paths[0], launch_paths[1]),
        output_directory=str((tmp_path / "qualification-output").resolve()),
        request_timeout_seconds=30.0,
        abort_timeout_seconds=5.0,
        hbm_allowed_growth_bytes=0,
        formal_measured=False,
    )
    plan_path = (tmp_path / "physical-plan.json").resolve()
    publish_trusted_empirical_tp1_session_reset_physical_plan(
        plan=plan, output_path=plan_path
    )
    return plan_path, authority_binding, authority, specs, launches


def test_two_trace_physical_qualification_publishes_unsigned_authority_only_after_8_of_8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _old_binding, _old_authority, _specs, _launches = _physical_plan(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(physical, "run_session_live_contract", _fake_live_contract)
    result = asyncio.run(
        execute_trusted_empirical_tp1_session_reset_qualification(
            physical_plan_path=plan_path,
            runtime=_QualificationRuntime(),
        )
    )

    assert result.status == "PASS"
    assert result.formal_measured is False
    assert result.authority is not None
    authority = physical.revalidate_trusted_empirical_tp1_session_reset_authority(
        result.authority.absolute_path
    )[1]
    assert authority.tests_collected == authority.tests_passed == 8
    assert authority.test_names == (
        "same_server_process_identity",
        "native_session_epoch_lineage",
        "exact_output_token_trajectory",
        "request_queue_empty_after_trace",
        "optimizer_candidate_and_adaptation_state_reset",
        "registered_cache_policy_restored",
        "terminal_writer_fully_flushed",
        "hbm_returns_without_monotonic_growth",
    )
    assert authority.formal_measured is False


def test_physical_qualification_failure_retains_raw_evidence_without_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _old_binding, _old_authority, _specs, _launches = _physical_plan(
        tmp_path, monkeypatch
    )
    result = asyncio.run(
        execute_trusted_empirical_tp1_session_reset_qualification(
            physical_plan_path=plan_path,
            runtime=_QualificationRuntime(fail=True),
        )
    )

    assert result.status == "FAIL"
    assert result.authority is None
    assert result.failure_terminal is not None
    result.raw_terminal.reopen(label="trusted empirical reset failed raw")
    assert result.raw_terminal.size > 0
    assert not (
        Path(result.plan.absolute_path).parent
        / "qualification-output"
        / "authority.json"
    ).exists()


class _GroupHandle:
    def __init__(self, output: Path, *, fail_reset_index: int) -> None:
        self.output = output
        self.fail_reset_index = fail_reset_index
        self.reset_calls = 0
        self.shared_calls = []
        self.close_calls = 0
        self.force_close_calls = 0

    @property
    def process_id(self) -> int:
        return 777

    async def reset_for_member(
        self,
        *,
        session_plan_sha256,
        reset_authority_sha256,
        prior_member,
        next_member,
        session_epoch,
    ):
        index = self.reset_calls
        self.reset_calls += 1
        if index == self.fail_reset_index:
            evidence = _publish(
                self.output / f"reset-failed-{index}.json",
                {"kind": "injected-reset-failure", "index": index},
            )
            raise FormalServingSessionResetFailed(
                "injected reset failure", evidence=evidence
            )
        return _publish(
            self.output / f"reset-{index}.json",
            {
                "schema_version": 1,
                "kind": "formal_serving_session_reset_boundary",
                "session_plan_sha256": session_plan_sha256,
                "reset_authority_sha256": reset_authority_sha256,
                "process_id": 777,
                "session_epoch": session_epoch,
                "prior_materialized_cell_id": (
                    None if prior_member is None else prior_member.materialized_cell_id
                ),
                "next_materialized_cell_id": next_member.materialized_cell_id,
                "all_reset_complete": True,
                "request_queue_empty": True,
                "terminal_writer_flushed": True,
            },
        )

    async def execute_member(self, *, member, session_epoch):
        self.shared_calls.append((member.materialized_cell_id, session_epoch))
        pointer = _publish(
            self.output / f"shared-result-{session_epoch}.json",
            {"cell": member.materialized_cell_id, "mode": "shared"},
        )
        return FormalServingSessionMemberPhysicalResult(
            status="COMPLETE",
            process_id=777,
            started_ns=100 * session_epoch,
            finished_ns=100 * session_epoch + 10,
            exit_code=0,
            result_pointer=pointer,
            failure_code=None,
        )

    async def close(self) -> None:
        self.close_calls += 1

    async def force_close(self) -> None:
        self.force_close_calls += 1


class _GroupRuntime:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.handle = _GroupHandle(output, fail_reset_index=1)
        self.fresh_calls = []

    async def start_shared_session(self, *, execution):
        assert execution.authority.formal_measured is False
        return self.handle

    async def execute_fresh_member(self, *, member, fallback_reason):
        self.fresh_calls.append((member.materialized_cell_id, fallback_reason))
        index = len(self.fresh_calls)
        pointer = _publish(
            self.output / f"fresh-result-{index}.json",
            {"cell": member.materialized_cell_id, "mode": "fresh"},
        )
        return FormalServingSessionMemberPhysicalResult(
            status="COMPLETE",
            process_id=900 + index,
            started_ns=1000 * index,
            finished_ns=1000 * index + 10,
            exit_code=0,
            result_pointer=pointer,
            failure_code=None,
        )


def test_reset_failure_preserves_shared_prefix_and_runs_unstarted_remainder_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_binding, authority = _GROUP._published_authority(
        tmp_path / "group-authority",
        monkeypatch,
        method_family="static",
    )
    specs = []
    for index in range(3):
        config = _GROUP._config(label=f"group-{index}")
        launch = _GROUP._producer_generated_launch(
            tmp_path,
            label=f"group-{index}",
            config=config,
            port=26000 + index,
        )
        specs.append(
            _GROUP._group_spec(
                tmp_path,
                index=200 + index,
                config=config,
                launch=launch,
                method_family="static",
                source_snapshot_sha256=authority.source_snapshot_sha256,
                protocol_lock_sha256=authority.protocol_lock_sha256,
                inventory_sha256=authority.inventory_sha256,
            )
        )
    plan = partition_formal_serving_session_groups(
        specs,
        reset_authorities=(authority_binding,),
        max_member_count=3,
        max_estimated_duration_seconds=100.0,
    )[0]
    assert len(plan.members) == 3
    plan_path = (tmp_path / "group-plan.json").resolve()
    _publish(plan_path, plan.to_dict())
    execution_output = (tmp_path / "group-execution").resolve()
    spec = FormalServingSessionGroupExecutionSpec(
        schema_version=1,
        kind="formal_serving_session_group_execution_spec",
        protocol_sha256=FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
        group_plan_path=str(plan_path),
        reset_authority_path=authority_binding.absolute_path,
        output_directory=str(execution_output),
        formal_measured=False,
    )
    spec_path = (tmp_path / "group-execution-spec.json").resolve()
    publish_formal_serving_session_group_execution_spec(
        spec=spec, output_path=spec_path
    )
    runtime_output = (tmp_path / "group-runtime").resolve()
    runtime_output.mkdir()
    runtime = _GroupRuntime(runtime_output)
    result = asyncio.run(
        execute_formal_serving_session_group(
            execution_spec_path=spec_path,
            runtime=runtime,
        )
    )

    assert result.status == "COMPLETE"
    assert result.shared_completed == 1
    assert result.fresh_fallback_completed == 2
    assert result.failed == 0
    assert runtime.handle.force_close_calls == 1
    assert runtime.handle.close_calls == 0
    assert len(runtime.handle.shared_calls) == 1
    assert len(runtime.fresh_calls) == 2
    artifacts = tuple(
        FormalServingSessionGroupCellArtifact.from_dict(binding.reopen())
        for binding in result.cell_artifacts
    )
    assert tuple(item.execution_mode for item in artifacts) == (
        "shared_session_tp1",
        "fresh_process_fallback",
        "fresh_process_fallback",
    )
    assert artifacts[0].process_id == 777
    assert artifacts[0].reset_boundary is not None
    assert all(item.reset_boundary is None for item in artifacts[1:])
    result_path = execution_output / "result.json"
    assert (
        FormalServingSessionGroupExecutionResult.from_dict(
            CanonicalJsonProofBinding.bind(result_path).reopen()
        )
        == result
    )
