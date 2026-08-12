from __future__ import annotations

from types import ModuleType

from lightcone_spec import experiments, orchestration
from lightcone_spec.experiments import (
    completion_authority,
    gpu_pool,
    industrial_analysis,
    planning,
    planning_artifacts,
)
from lightcone_spec.orchestration import (
    execution_bundle,
    executor,
    industrial,
    native_terminal,
    session,
)


def _assert_exact_exports(module: ModuleType, expected: dict[str, object]) -> None:
    exported = module.__dict__["__all__"]
    assert len(exported) == len(set(exported))
    for name, value in expected.items():
        assert name in exported
        assert getattr(module, name) is value


def test_experiment_public_api_exports_industrial_planning_and_pool() -> None:
    expected = {
        name: getattr(planning_artifacts, name) for name in planning_artifacts.__all__
    }
    expected.update(
        {
            "AssignmentRunner": gpu_pool.AssignmentRunner,
            "AssignmentTerminalAuthority": (
                completion_authority.AssignmentTerminalAuthority
            ),
            "CompletedCellAuthority": completion_authority.CompletedCellAuthority,
            "BudgetGroupTotal": planning.BudgetGroupTotal,
            "DispatchExecutionState": gpu_pool.DispatchExecutionState,
            "GpuPoolScheduler": gpu_pool.GpuPoolScheduler,
            "reduce_confirmation_family_power": (
                industrial_analysis.reduce_confirmation_family_power
            ),
            "reduce_e2_stage_from_raw": industrial_analysis.reduce_e2_stage_from_raw,
        }
    )

    _assert_exact_exports(experiments, expected)
    assert "ConfirmationBlockPlan" not in experiments.__all__
    assert not hasattr(experiments, "ConfirmationBlockPlan")


def test_orchestration_public_api_exports_assignment_and_native_hook() -> None:
    expected = {
        "AsyncNativeTerminalAdminTransport": (
            native_terminal.AsyncNativeTerminalAdminTransport
        ),
        "IndustrialPhysicalAssignment": industrial.IndustrialPhysicalAssignment,
        "IndustrialExecutionTerminalBinding": (
            executor.IndustrialExecutionTerminalBinding
        ),
        "IndustrialServerBlockResult": session.IndustrialServerBlockResult,
        "NativeTerminalAttestation": native_terminal.NativeTerminalAttestation,
        "NativeTerminalBeginReceipt": native_terminal.NativeTerminalBeginReceipt,
        "NativeTerminalCapability": native_terminal.NativeTerminalCapability,
        "NativeTerminalProvider": native_terminal.NativeTerminalProvider,
        "NativeTerminalResetReceipt": native_terminal.NativeTerminalResetReceipt,
        "NativeTerminalRunBinding": native_terminal.NativeTerminalRunBinding,
        "TerminalRequestExpectation": native_terminal.TerminalRequestExpectation,
        "ValidatedNativeTerminalEvidence": (
            native_terminal.ValidatedNativeTerminalEvidence
        ),
        "bind_industrial_gpu_assignment": industrial.bind_industrial_gpu_assignment,
        "execute_industrial_fresh_process_fallback": (
            session.execute_industrial_fresh_process_fallback
        ),
        "execute_industrial_server_session": (
            session.execute_industrial_server_session
        ),
        "render_assigned_industrial_cell_runtime_plan": (
            industrial.render_assigned_industrial_cell_runtime_plan
        ),
        "revalidate_industrial_execution_result": (
            executor.revalidate_industrial_execution_result
        ),
    }

    _assert_exact_exports(orchestration, expected)
    assert "IndustrialAssignmentExecutionBundle" not in orchestration.__all__
    assert "execute_dispatch_wave_bundles" not in orchestration.__all__
    assert execution_bundle.IndustrialAssignmentExecutionBundle is not None
    assert (
        orchestration.NativeEvidenceProvider is native_terminal.NativeTerminalProvider
    )
    for unsupported_session_api in (
        "OpenIndustrialServerSession",
        "SessionBoundaryRuntime",
        "SessionExecutionBinding",
        "close_server_session",
        "execute_trace_in_session",
        "finalize_trace",
        "open_server_session",
        "reset_and_attest_trace_boundary",
    ):
        assert unsupported_session_api not in orchestration.__all__
        assert not hasattr(orchestration, unsupported_session_api)
