from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from test_industrial_executor import _execution_fixture, _FakeTransport

import lightcone_spec.orchestration.executor as executor_module
from lightcone_spec.orchestration.executor import IndustrialExecutionResult
from lightcone_spec.orchestration.native_terminal import NativeTerminalProvider
from lightcone_spec.orchestration.session import (
    SHARED_SESSION_FALLBACK_MODE,
    SHARED_SESSION_UNAVAILABLE_REASON,
    IndustrialServerSessionPlan,
    execute_industrial_server_session,
)


def _session_plan(plans) -> IndustrialServerSessionPlan:
    return IndustrialServerSessionPlan.create(
        plans,
        capability_receipt_sha256="a" * 64,
        compile_cache_receipt_sha256="b" * 64,
        dtype="bfloat16",
        precision="bf16",
        graph_buckets=(1,),
        hbm_reservation_bytes=0,
    )


def test_block_fallback_executes_every_trace_as_a_fresh_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _execution_fixture(tmp_path, request_count=1).plan
    second = replace(first, abort_grace_s=first.abort_grace_s + 1)
    second.validate()
    plans = (first, second)
    session_plan = _session_plan(plans)
    calls: list[tuple[str, int, int]] = []
    resources: list[tuple[_FakeTransport, NativeTerminalProvider]] = []

    def resources_for_plan(plan):
        transport = _FakeTransport(plan=plan)
        provider = NativeTerminalProvider(
            transport,
            trusted_attester_policy=plan.trusted_attester_policy,
        )
        resources.append((transport, provider))
        return transport, provider

    async def fake_execute(
        plan,
        *,
        output_root,
        run_nonce_sha256,
        launch_server,
        transport,
        native_evidence,
    ) -> IndustrialExecutionResult:
        del output_root, launch_server, run_nonce_sha256
        calls.append((plan.sha256, id(transport), id(native_evidence)))
        return IndustrialExecutionResult(
            run_id=f"fresh-{len(calls)}",
            execution_plan_sha256=plan.sha256,
            experiment_budget_sha256=plan.budget.sha256,
            rank_config_sha256=plan.rank_config_sha256,
            topology_sha256=plan.topology_sha256,
            resumed=False,
            terminal_receipt=f"terminal-{len(calls)}.json",
            terminal_receipt_sha256=f"{len(calls):064x}",
            budget_observation=f"observation-{len(calls)}.json",
            budget_observation_sidecar=f"observation-{len(calls)}.sha256",
            budget_observation_sha256=f"{len(calls) + 2:064x}",
            evidence_files=(),
            accounting=None,
        )

    monkeypatch.setattr(executor_module, "execute_industrial_plan", fake_execute)

    async def forbidden_launch(_server):
        raise AssertionError("the fake standalone executor owns this boundary")

    result = asyncio.run(
        execute_industrial_server_session(
            session_plan,
            plans,
            output_roots=(
                first.runtime_plan.cell.resources.evidence_root,
                second.runtime_plan.cell.resources.evidence_root,
            ),
            run_nonce_sha256s=("1" * 64, "2" * 64),
            launch_server=forbidden_launch,
            resources_for_plan=resources_for_plan,
        )
    )

    assert result.execution_mode == SHARED_SESSION_FALLBACK_MODE
    assert result.fallback_reason == SHARED_SESSION_UNAVAILABLE_REASON
    assert tuple(row.execution_plan_sha256 for row in result.executions) == tuple(
        plan.sha256 for plan in plans
    )
    assert len({transport_id for _, transport_id, _ in calls}) == 2
    assert len({provider_id for _, _, provider_id in calls}) == 2
    assert len(resources) == 2
    assert len(result.sha256) == 64


def test_block_fallback_rejects_reused_resources_before_first_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _execution_fixture(tmp_path, request_count=1).plan
    second = replace(first, abort_grace_s=first.abort_grace_s + 1)
    plans = (first, second)
    session_plan = _session_plan(plans)
    transport = _FakeTransport(plan=first)
    provider = NativeTerminalProvider(
        transport,
        trusted_attester_policy=first.trusted_attester_policy,
    )
    executions = 0

    async def forbidden_execute(*args, **kwargs):
        nonlocal executions
        executions += 1
        raise AssertionError("reused resources reached execution")

    monkeypatch.setattr(executor_module, "execute_industrial_plan", forbidden_execute)

    async def forbidden_launch(_server):
        raise AssertionError("reused resources reached launch")

    with pytest.raises(ValueError, match="unique per-trace resources"):
        asyncio.run(
            execute_industrial_server_session(
                session_plan,
                plans,
                output_roots=(
                    first.runtime_plan.cell.resources.evidence_root,
                    second.runtime_plan.cell.resources.evidence_root,
                ),
                run_nonce_sha256s=("3" * 64, "4" * 64),
                launch_server=forbidden_launch,
                resources_for_plan=lambda _plan: (transport, provider),
            )
        )
    assert executions == 0
