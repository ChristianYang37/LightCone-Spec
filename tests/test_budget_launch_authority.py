from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_budget_materialization_authority import (
    _AuthorityFixture,
    _build_authority_fixture,
)

from lightcone_spec.experiments.budget_authority import (
    DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON,
    BudgetMaterializationBlockedError,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuDispatchExecutionContext,
    execute_dispatch_plan,
)
from lightcone_spec.experiments.planning import BudgetPlan, ScenarioMilliseconds


@pytest.fixture(scope="module")
def launch_fixture(tmp_path_factory: pytest.TempPathFactory) -> _AuthorityFixture:
    return _build_authority_fixture(tmp_path_factory.mktemp("budget-launch-authority"))


def test_unresolved_budget_is_raw_bound_and_cannot_reach_runner(
    launch_fixture: _AuthorityFixture,
) -> None:
    fixture = launch_fixture
    summary = fixture.execution.authority_dict()
    assert summary["budget_plan_sha256"] == fixture.plan.sha256
    assert summary["capacity_authority_sha256"] == (
        fixture.plan.capacity_authority.sha256
    )
    assert summary["budget_materialization_authority_sha256"] == (
        fixture.authority.sha256
    )
    dispatch = fixture.planning.issue_plan()
    calls: list[str] = []

    async def runner(assignment):
        calls.append(assignment.assignment_id)
        raise AssertionError("runner must not be reached")

    with pytest.raises(BudgetMaterializationBlockedError) as blocked:
        asyncio.run(
            execute_dispatch_plan(
                dispatch,
                execution_context=fixture.execution,
                runner=runner,
            )
        )
    assert blocked.value.reason_code == (
        DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON
    )
    assert calls == []


def test_capacity_ready_stub_cannot_bypass_raw_budget_or_dependency_authority(
    launch_fixture: _AuthorityFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = launch_fixture
    ready_calls: list[str] = []

    def caller_stub(_plan: BudgetPlan):
        ready_calls.append("called")
        return fixture.plan.budgets

    monkeypatch.setattr(BudgetPlan, "require_ready", caller_stub)
    with pytest.raises(BudgetMaterializationBlockedError) as blocked:
        fixture.execution.issue_plan()
    assert blocked.value.reason_code == (
        DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON
    )
    assert ready_calls == []


def test_context_rejects_budget_rows_and_jointly_rehashed_plan_summary(
    launch_fixture: _AuthorityFixture,
) -> None:
    fixture = launch_fixture
    common = {
        "registry": fixture.registry,
        "inventory": fixture.inventory,
        "interference_envelope": fixture.planning.interference_envelope,
        "budgets": fixture.plan.diagnostic_budgets,
        "receipts": fixture.activation.dependency_receipts,
        "activation_artifact": fixture.activation,
        "budget_materialization_authority": fixture.authority,
    }
    first_budget = fixture.plan.diagnostic_budgets[0]
    with pytest.raises(ValueError, match="exact BudgetPlan rows"):
        GpuDispatchExecutionContext(
            **{
                **common,
                "budgets": (
                    replace(
                        first_budget,
                        minimum_completed_requests=(
                            first_budget.minimum_completed_requests + 1
                        ),
                    ),
                    *fixture.plan.diagnostic_budgets[1:],
                ),
            },
            budget_plan=fixture.plan,
        )
    with pytest.raises(ValueError, match="exact BudgetPlan rows"):
        GpuDispatchExecutionContext(
            **{
                **common,
                "budgets": fixture.plan.diagnostic_budgets[:-1],
            },
            budget_plan=fixture.plan,
        )
    with pytest.raises(ValueError, match="raw activation authority"):
        GpuDispatchExecutionContext(
            **{
                **common,
                "receipts": (),
            },
            budget_plan=fixture.plan,
        )

    policy = fixture.plan.policy
    duration = policy.job_policies[0].startup_model_load
    forged_policy = replace(
        policy,
        job_policies=(
            replace(
                policy.job_policies[0],
                startup_model_load=ScenarioMilliseconds(
                    duration.optimistic + 1,
                    duration.registered + 1,
                    duration.quota_envelope + 1,
                ),
            ),
            *policy.job_policies[1:],
        ),
    )
    with pytest.raises(ValueError, match="first-party raw rematerialization"):
        GpuDispatchExecutionContext(
            **common,
            budget_plan=replace(fixture.plan, policy=forged_policy),
        )


def test_stage_aggregate_cannot_be_reused_as_an_execution_activation(
    launch_fixture: _AuthorityFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import budget_authority

    monkeypatch.setattr(
        budget_authority,
        "replay_budget_activation_authority",
        lambda _binding: SimpleNamespace(
            stage_family_authorities=(object(),),
            auxiliary_authority=None,
        ),
    )
    with pytest.raises(ValueError, match="completion authority, not an execution"):
        launch_fixture.execution._budget_activation_replay()
