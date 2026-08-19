"""Lazy public exports for the orchestration package.

Keeping this package initializer import-free is a runtime requirement. Formal
workers are started with ``python -m lightcone_spec.orchestration.<worker>``;
eagerly importing the serving stack here can create a cold-start cycle before
the selected worker module is reached.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "executor",
        (
            "MISSING_NATIVE_EVIDENCE_REASON",
            "TRUSTED_NATIVE_ATTESTER_UNAVAILABLE_REASON",
            "ArtifactBinding",
            "AsyncSubprocessServerHandle",
            "ExecutionClock",
            "IndustrialExecutionPlan",
            "IndustrialExecutionResult",
            "IndustrialExecutionTerminalBinding",
            "NativeEvidenceBatch",
            "NativeEvidencePreflight",
            "NativeEvidenceProvider",
            "NativeEvidenceUnavailableError",
            "ServerHandle",
            "ServerLauncher",
            "build_industrial_execution_plan",
            "execute_industrial_plan",
            "industrial_execution_split_contract",
            "industrial_run_id",
            "launch_server_subprocess",
            "native_evidence_preflight",
            "render_industrial_execution_plan",
            "revalidate_industrial_execution_result",
        ),
    ),
    (
        "formal_failure_physical",
        (
            "FORMAL_E5_FAILURE_LIFECYCLE_CONTROL_PROTOCOL_SHA256",
            "FORMAL_E5_FAILURE_PHYSICAL_PROTOCOL_SHA256",
            "FormalE5FailureLifecycleProofArtifact",
            "FormalE5FailureLifecycleRawReceipt",
            "ValidatedUnsignedFormalE5FailureRun",
            "build_formal_e5_failure_lifecycle_control_subject",
            "execute_formal_e5_failure_run_plan",
            "publish_formal_e5_failure_lifecycle_proof_artifact",
            "validate_formal_e5_failure_lifecycle_proof_artifact",
            "validate_formal_e5_failure_lifecycle_raw_receipt",
        ),
    ),
    (
        "formal_physical_dispatch",
        (
            "FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256",
            "FORMAL_SINGLE_OPERATOR_EXECUTION_REBUILD_SOURCE_PROTOCOL_SHA256",
            "FormalPhysicalDispatchError",
            "FormalServingRequestScheduleReceipt",
            "FormalServingRequestScheduleRow",
            "FormalServingRequestScheduleSource",
            "FormalServingRequestScheduleSourceRow",
            "FormalServingRunPlan",
            "FormalSingleOperatorExecutionRebuildSource",
            "ValidatedUnsignedFormalGangServingRun",
            "execute_formal_distributed_serving_run_plan",
            "execute_formal_serving_run_plan",
            "execute_formal_single_operator_serving_run_plan",
            "execute_formal_tp1_serving_run_plan",
            "load_formal_serving_run_plan",
            "materialize_formal_serving_request_schedule",
            "materialize_formal_serving_run_plan",
            "materialize_formal_single_operator_serving_run_plan",
            "rebuild_formal_single_operator_execution_binding_from_plan",
            "revalidate_formal_serving_run_plan",
            "revalidate_formal_single_operator_execution_rebuild_source",
        ),
    ),
    (
        "formal_serving_lift",
        (
            "FormalDistributedLifecycleTimingProofArtifact",
            "build_formal_serving_itl_control_subject",
            "build_formal_serving_lifecycle_control_subject",
            "build_formal_serving_terminal_control_subject",
            "formal_serving_stage_itl_execution_identity",
            "formal_serving_stage_itl_gpu_proof_path",
            "publish_formal_serving_itl_proof",
            "publish_formal_serving_itl_raw_receipt",
            "publish_formal_serving_lifecycle_proof",
            "publish_formal_serving_terminal_proof",
            "validate_formal_distributed_lifecycle_timing_proof_artifact",
        ),
    ),
    (
        "formal_terminal_result",
        (
            "FORMAL_DISTRIBUTED_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256",
            "FORMAL_SINGLE_OPERATOR_PREFLIGHT_TP1_RAW_TERMINAL_PROTOCOL_SHA256",
            "FormalDistributedTerminalExternalControlBinding",
            "FormalDistributedTerminalRequestResult",
            "FormalDistributedTerminalResultProjection",
            "FormalDistributedTerminalResultProofArtifact",
            "FormalDistributedTerminalUpdateResult",
            "FormalSingleOperatorPreflightRawRequestResult",
            "FormalSingleOperatorPreflightRawTerminalProjection",
            "FormalSingleOperatorPreflightTp1RawTerminalProofArtifact",
            "build_formal_distributed_terminal_control_subject",
            "build_formal_distributed_terminal_external_control_binding",
            "build_formal_terminal_control_subject",
            "build_formal_tp1_terminal_control_subject",
            "formal_distributed_scored_native_itl_pointers",
            "formal_scored_native_itl_pointers",
            "publish_formal_distributed_terminal_result_proof_artifact",
            "publish_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact",
            "publish_formal_terminal_result_proof_artifact",
            "validate_formal_distributed_terminal_result_proof_artifact",
            "validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact",
            "validate_formal_terminal_result_proof_artifact",
        ),
    ),
    (
        "industrial",
        (
            "IndustrialPhysicalAssignment",
            "IndustrialRuntimePlan",
            "bind_industrial_gpu_assignment",
            "render_assigned_industrial_cell_runtime_plan",
            "render_industrial_cell_runtime_plan",
        ),
    ),
    ("manifest", ("PreliminarySpeedStudyManifest",)),
    (
        "native_terminal",
        (
            "CAPABILITY_PATH",
            "NATIVE_TERMINAL_EVIDENCE_FIELDS",
            "NATIVE_TERMINAL_EVIDENCE_HOOK",
            "NATIVE_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256",
            "PINNED_SGLANG_PATCH_SHA256",
            "PINNED_SGLANG_TREE",
            "PINNED_SGLANG_UPSTREAM_COMMIT",
            "SUPPORTED_METHODS",
            "TERMINAL_EVIDENCE_PATH",
            "AsyncNativeTerminalAdminTransport",
            "CandidateStateReplayPointer",
            "CandidateStateReplayProjection",
            "CandidateStateReplayProofArtifact",
            "NativeTerminalAttestation",
            "NativeTerminalBeginReceipt",
            "NativeTerminalCapability",
            "NativeTerminalExternalControlBinding",
            "NativeTerminalProvider",
            "NativeTerminalRequestResult",
            "NativeTerminalResetReceipt",
            "NativeTerminalResultProjection",
            "NativeTerminalResultProofArtifact",
            "NativeTerminalRunBinding",
            "NativeTerminalUpdateResult",
            "PreparedNativeTerminalExternalControl",
            "SignatureVerifier",
            "TerminalRequestExpectation",
            "UnsignedNativeItlTokenEvent",
            "UnsignedNativeServingPhaseResult",
            "UnsignedNativeTerminalCollection",
            "ValidatedNativeTerminalEvidence",
            "ValidatedUnsignedNativeItlPointer",
            "ValidatedUnsignedNativeItlPointerBundle",
            "build_native_terminal_external_control_binding",
            "canonical_json_bytes",
            "canonical_sha256",
            "collect_unsigned_native_terminal_artifact",
            "derive_candidate_state_replay_pointer",
            "derive_native_terminal_result_projection",
            "finalize_prepared_candidate_state_replay_pointers",
            "finalize_prepared_native_terminal_external_controls",
            "prepare_native_terminal_external_control",
            "project_prepared_candidate_state_replay_pointer",
            "publish_candidate_state_replay_proof_artifact",
            "publish_candidate_state_replay_proof_artifacts",
            "publish_native_terminal_result_proof_artifact",
            "publish_native_terminal_result_proof_artifacts",
            "validate_candidate_state_replay_proof_artifact",
            "validate_controlled_candidate_state_replay_pointer_artifact",
            "validate_native_terminal_artifact_with_external_control",
            "validate_native_terminal_artifacts_with_external_controls",
            "validate_native_terminal_result_proof_artifact",
            "validate_unsigned_native_itl_pointer_bundle",
        ),
    ),
    (
        "remote_dispatch",
        (
            "AsyncioSshTransport",
            "AsyncSshTransport",
            "CrossHostCollectivesUnvalidated",
            "FleetWaveOutcome",
            "HostAssignmentBinding",
            "RemoteFleetWaveReceipt",
            "RemoteHostExecutionBinding",
            "RemoteHostWaveRequest",
            "RemoteHostWaveResponse",
            "RemoteHostWaveResult",
            "RemoteTransportOutcome",
            "RemoteWorkerStatus",
            "SshHostRoute",
            "SshOutputLimitExceeded",
            "SshProcessResult",
            "SshTransportTimedOut",
            "build_ssh_argv",
            "decode_remote_host_wave_request",
            "execute_fleet_wave",
            "execute_host_local_wave_request",
            "execute_remote_host_wave",
        ),
    ),
    (
        "runtime",
        (
            "ServerLaunch",
            "render_onlinespec_runtime_plan",
            "render_onlinespec_tuning_runtime_plan",
            "render_replication_runtime_plan",
            "render_runtime_plan",
            "render_static_load_runtime_plan",
            "render_target_only_runtime_plan",
            "render_tuning_runtime_plan",
        ),
    ),
    (
        "session",
        (
            "SHARED_SESSION_FALLBACK_MODE",
            "SHARED_SESSION_UNAVAILABLE_REASON",
            "IndustrialResetReceipt",
            "IndustrialServerBlockResult",
            "IndustrialServerSessionKey",
            "IndustrialServerSessionPlan",
            "IndustrialSessionOpenReceipt",
            "IndustrialSessionTraceReceipt",
            "SessionBoundaryState",
            "SharedSessionUnavailableError",
            "execute_industrial_fresh_process_fallback",
            "execute_industrial_server_session",
        ),
    ),
)

__all__ = [
    name
    for _module_name, _module_exports in _LAZY_EXPORT_GROUPS
    for name in _module_exports
]


def _lazy_export_map() -> dict[str, str]:
    exports: dict[str, str] = {}
    for module_name, names in _LAZY_EXPORT_GROUPS:
        for name in names:
            if name in exports:
                raise RuntimeError(f"duplicate orchestration export: {name}")
            exports[name] = f"{__name__}.{module_name}"
    if len(exports) != len(__all__):
        raise RuntimeError("orchestration lazy exports differ from __all__")
    return exports


_LAZY_EXPORTS = _lazy_export_map()


def __getattr__(name: str) -> Any:
    """Resolve one historical public symbol without importing unrelated stacks."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
