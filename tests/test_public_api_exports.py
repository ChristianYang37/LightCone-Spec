from __future__ import annotations

from types import ModuleType

from lightcone_spec import experiments, orchestration
from lightcone_spec.experiments import (
    completion_authority,
    formal_gpu_hour_proof,
    formal_preflight_coverage,
    formal_preflight_execution,
    formal_single_operator_early_execution,
    formal_single_operator_gpu_hours,
    formal_single_operator_profiler,
    formal_single_operator_stages,
    formal_slo_metrics,
    gpu_fleet,
    gpu_hour_authority,
    gpu_pool,
    industrial_analysis,
    long_context_analysis,
    planning,
    planning_artifacts,
    preflight_authority,
)
from lightcone_spec.orchestration import (
    execution_bundle,
    executor,
    industrial,
    native_terminal,
    remote_dispatch,
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
            "GpuFleetInventory": gpu_fleet.GpuFleetInventory,
            "GpuFleetScheduler": gpu_fleet.GpuFleetScheduler,
            "GpuFleetDispatchPlan": gpu_fleet.GpuFleetDispatchPlan,
            "StagedGpuHourStratum": gpu_hour_authority.StagedGpuHourStratum,
            "StagedProspectiveGpuHourCost": (
                gpu_hour_authority.StagedProspectiveGpuHourCost
            ),
            "StagedProspectiveGpuHourSourceManifest": (
                gpu_hour_authority.StagedProspectiveGpuHourSourceManifest
            ),
            "HostInventoryBinding": gpu_fleet.HostInventoryBinding,
            "HostExecutionBinding": gpu_fleet.HostExecutionBinding,
            "FleetWaveReceipt": gpu_fleet.FleetWaveReceipt,
            "STAGED_PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256": (
                gpu_hour_authority.STAGED_PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256
            ),
            "assemble_gpu_fleet_inventory": (gpu_fleet.assemble_gpu_fleet_inventory),
            "reduce_confirmation_family_power": (
                industrial_analysis.reduce_confirmation_family_power
            ),
            "reduce_e2_stage_from_raw": industrial_analysis.reduce_e2_stage_from_raw,
            "E3bLongContextAnalysisPlan": (
                long_context_analysis.E3bLongContextAnalysisPlan
            ),
            "E3bLongContextRawFamilyInput": (
                industrial_analysis.E3bLongContextRawFamilyInput
            ),
            "E3bLongContextStageArtifact": (
                industrial_analysis.E3bLongContextStageArtifact
            ),
            "reduce_e3b_long_context_pair": (
                long_context_analysis.reduce_e3b_long_context_pair
            ),
            "reduce_e3b_long_context_from_raw": (
                industrial_analysis.reduce_e3b_long_context_from_raw
            ),
            "PREFLIGHT_STAGE_COVERAGE_BRIDGE_PROTOCOL_SHA256": (
                preflight_authority.PREFLIGHT_STAGE_COVERAGE_BRIDGE_PROTOCOL_SHA256
            ),
            "materialize_formal_preflight_stage_coverage": (
                preflight_authority.materialize_formal_preflight_stage_coverage
            ),
            "FormalPreflightStageCoverageProofArtifact": (
                formal_preflight_coverage.FormalPreflightStageCoverageProofArtifact
            ),
            "FormalStageGpuHourEnvelopeProofArtifact": (
                formal_gpu_hour_proof.FormalStageGpuHourEnvelopeProofArtifact
            ),
            "bind_formal_stage_gpu_hour_envelope_proof_artifact": (
                formal_gpu_hour_proof.bind_formal_stage_gpu_hour_envelope_proof_artifact
            ),
            "publish_formal_stage_gpu_hour_envelope_proof_artifact": (
                formal_gpu_hour_proof.publish_formal_stage_gpu_hour_envelope_proof_artifact
            ),
            "revalidate_formal_stage_gpu_hour_envelope_proof_artifact": (
                formal_gpu_hour_proof.revalidate_formal_stage_gpu_hour_envelope_proof_artifact
            ),
            "publish_formal_preflight_stage_coverage_proof_artifact": (
                formal_preflight_coverage.publish_formal_preflight_stage_coverage_proof_artifact
            ),
            "materialize_staged_prospective_gpu_hour_envelope": (
                gpu_hour_authority.materialize_staged_prospective_gpu_hour_envelope
            ),
            "qualify_formal_preflight_interference_locally": (
                formal_preflight_execution.qualify_formal_preflight_interference_locally
            ),
            "revalidate_persisted_staged_prospective_gpu_hour_source_manifest": (
                gpu_hour_authority.revalidate_persisted_staged_prospective_gpu_hour_source_manifest
            ),
            "revalidate_formal_preflight_stage_coverage_proof_artifact": (
                formal_preflight_coverage.revalidate_formal_preflight_stage_coverage_proof_artifact
            ),
        }
    )
    expected.update(
        {name: getattr(formal_slo_metrics, name) for name in formal_slo_metrics.__all__}
    )
    expected.update(
        {
            name: getattr(formal_single_operator_early_execution, name)
            for name in formal_single_operator_early_execution.__all__
        }
    )
    expected.update(
        {
            name: getattr(formal_single_operator_gpu_hours, name)
            for name in formal_single_operator_gpu_hours.__all__
        }
    )
    expected.update(
        {
            name: getattr(formal_single_operator_profiler, name)
            for name in formal_single_operator_profiler.__all__
        }
    )
    expected.update(
        {
            name: getattr(formal_single_operator_stages, name)
            for name in formal_single_operator_stages.__all__
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
        "RemoteFleetWaveReceipt": remote_dispatch.RemoteFleetWaveReceipt,
        "RemoteHostExecutionBinding": (remote_dispatch.RemoteHostExecutionBinding),
        "RemoteHostWaveRequest": remote_dispatch.RemoteHostWaveRequest,
        "SshHostRoute": remote_dispatch.SshHostRoute,
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
        "execute_fleet_wave": remote_dispatch.execute_fleet_wave,
        "execute_host_local_wave_request": (
            remote_dispatch.execute_host_local_wave_request
        ),
        "execute_remote_host_wave": remote_dispatch.execute_remote_host_wave,
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
