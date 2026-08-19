"""Fail-closed CLI for the Static/TTS/L0 speed study."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

import pyarrow as pa
import pyarrow.parquet as pq

from lightcone_spec import (
    PINNED_SGLANG_COMMIT,
    PINNED_SGLANG_PATCH_COUNT,
    PINNED_SGLANG_TREE,
)
from lightcone_spec.cli.formal_single_operator import (
    add_formal_single_operator_parser,
    handle_formal_single_operator_command,
)
from lightcone_spec.config import RunConfig, load_run_config, run_config_sha256
from lightcone_spec.doctor import (
    _require_project_runtime_source_identity,
    format_doctor,
)
from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.experiments.breadth_fdr_authority import (
    formal_e0_breadth_fdr_receipt_to_dict,
    reduce_formal_e0_breadth_fdr_from_artifact,
    signed_formal_e0_breadth_fdr_from_dict,
    signed_formal_e0_breadth_fdr_to_dict,
)
from lightcone_spec.experiments.budget_authority import (
    bind_budget_materialization_authority,
)
from lightcone_spec.experiments.capacity_authority import bind_capacity_authority
from lightcone_spec.experiments.data import (
    DFLASH_MODEL_CONTEXT_LIMIT,
    LongContinuationAdapter,
    load_natural_prompts,
    sample_set_sha256,
)
from lightcone_spec.experiments.e0_authority_artifact import (
    E0ExecutionRebuildShard,
    E0FinalResultRebuildArtifact,
    E0FormalRegistryAuthorityArtifact,
    E5FailureExecutionRebuildShard,
    E6RecursiveSourceDagArtifact,
    e0_final_completion_receipt_to_dict,
    load_e0_formal_registry_authority_artifact_index,
    load_e0_formal_registry_authority_bundle,
    publish_e0_execution_rebuild_shard,
    publish_e0_final_result_rebuild_artifact,
    publish_e0_formal_registry_authority_artifact,
    publish_e5_failure_execution_rebuild_shard,
    publish_e6_recursive_source_dag_artifact,
    reduce_e0_final_completion_from_artifact,
    signed_e0_final_completion_from_dict,
    signed_e0_final_completion_to_dict,
    signed_e1a_verification_to_dict,
    signed_e3b_confirmation_to_dict,
    signed_e5_confirmation_to_dict,
    signed_e6_confirmation_to_dict,
    signed_e6_model_compatibility_to_dict,
)
from lightcone_spec.experiments.e3a_staged_selection_proof import (
    bind_formal_e3a_staged_selection_proof_artifact,
    publish_formal_e3a_staged_selection_proof_artifact,
    revalidate_formal_e3a_staged_selection_proof_artifact,
)
from lightcone_spec.experiments.e4_stage_authority import (
    e4_profiler_completion_receipt_to_dict,
    reduce_e4_profiler_completion_from_registry,
    signed_e4_profiler_completion_from_dict,
    signed_e4_profiler_completion_to_dict,
)
from lightcone_spec.experiments.evidence import (
    GreedyTargetReference,
    evidence_files_sha256,
)
from lightcone_spec.experiments.formal_downstream_prefix import (
    FORMAL_DOWNSTREAM_MATERIALIZATION_ORDER,
    build_formal_downstream_completed_prefix_artifact,
    build_formal_downstream_materialization_proof_artifact,
    build_formal_downstream_pilot_precoverage_artifact,
    build_formal_downstream_reduction_proof_artifact,
    publish_formal_downstream_completed_prefix_artifact,
    publish_formal_downstream_materialization_proof_artifact,
    publish_formal_downstream_pilot_precoverage_artifact,
    publish_formal_downstream_reduction_proof_artifact,
    rebuild_formal_downstream_completed_prefix,
    rebuild_formal_downstream_materialization_proof,
    rebuild_formal_downstream_pilot_precoverage,
    rebuild_formal_downstream_reduction_proof,
)
from lightcone_spec.experiments.formal_gpu_hour_proof import (
    bind_formal_stage_gpu_hour_envelope_proof_artifact,
    publish_formal_stage_gpu_hour_envelope_proof_artifact,
)
from lightcone_spec.experiments.formal_gpu_hour_registry import (
    FormalStageGpuHourVerificationReceipt,
    aggregate_formal_study_gpu_hours,
    reserve_formal_stage_gpu_hour_verification_receipt,
)
from lightcone_spec.experiments.formal_initial_stage_proof import (
    bind_formal_initial_stage_materialization_proof_artifact,
    publish_formal_initial_stage_materialization_proof_artifact,
)
from lightcone_spec.experiments.formal_method_authority import (
    build_source_chronobelief_authority_artifact,
    build_source_tts_calibration_authority_artifact,
    load_chronobelief_authority_artifact,
    load_tts_calibration_authority_artifact,
    publish_chronobelief_authority_artifact,
    publish_tts_calibration_authority_artifact,
)
from lightcone_spec.experiments.formal_protocol import (
    FORMAL_STAGE_DAG,
    ProtocolLock,
    code_owned_qualification_source_identities,
)
from lightcone_spec.experiments.formal_protocol_lock_proof import (
    bind_formal_protocol_lock_source_proof_artifact,
    publish_formal_protocol_lock_git_snapshot,
    publish_formal_protocol_lock_source_proof_artifact,
    revalidate_formal_protocol_lock_source_proof_artifact,
)
from lightcone_spec.experiments.formal_registry import (
    assemble_and_reserve_formal_registry_manifest,
    extend_formal_registry_verification_receipt,
    formal_runtime_authority_manifest_from_dict,
    gpu_hour_estimate_from_dict,
    gpu_hour_estimate_to_dict,
    protocol_lock_to_dict,
    publish_formal_runtime_authority_manifest,
    reserve_formal_registry_verification_receipt,
    signed_e0_compatibility_from_dict,
    signed_e0_compatibility_to_dict,
    signed_e0_onlinespec_tuning_seal_to_dict,
    signed_e0_power_prefix_to_dict,
    signed_e1_survivor_selection_from_dict,
    signed_e1_survivor_selection_to_dict,
    signed_e2_staged_selection_from_dict,
    signed_e2_staged_selection_to_dict,
    signed_e3a_staged_selection_from_dict,
    signed_e3a_staged_selection_to_dict,
    signed_e3b_power_prefix_from_dict,
    signed_e3b_power_prefix_to_dict,
    signed_e4_stage_selection_from_dict,
    signed_e4_stage_selection_to_dict,
    signed_e5_anchor_selection_from_dict,
    signed_e5_power_and_anchor_from_dict,
    signed_e5_power_and_anchor_to_dict,
    signed_e6_power_prefix_from_dict,
    signed_e6_power_prefix_to_dict,
    signed_pilot_duration_from_dict,
    signed_protocol_lock_from_dict,
    signed_stage_coverage_from_dict,
    signed_stage_gpu_hour_from_dict,
    signed_stage_materialization_from_dict,
    signed_tts_calibration_seal_from_dict,
    signed_tts_calibration_seal_to_dict,
    stage_coverage_receipt_from_dict,
    stage_coverage_receipt_to_dict,
    stage_gpu_hour_envelope_from_dict,
    stage_gpu_hour_envelope_to_dict,
    stage_materialization_receipt_from_dict,
    stage_materialization_receipt_to_dict,
    tts_calibration_authority_from_dict,
    tts_l0_candidate_state_coverage_from_dict,
)
from lightcone_spec.experiments.formal_registry_layers import (
    FORMAL_REGISTRY_LAYER_ARTIFACT_KIND,
    bind_formal_registry_layer_artifact,
    load_formal_registry_verification_receipt_path,
    load_formal_signed_coverage_path,
    load_formal_signed_materialization_path,
    publish_formal_registry_layer_artifact,
    publish_formal_registry_replay_proof_shards,
)
from lightcone_spec.experiments.formal_runtime_manifest import (
    build_source_formal_runtime_authority_manifest,
)
from lightcone_spec.experiments.formal_stage_coverage_portable import (
    bind_formal_portable_stage_coverage_proof_artifact,
    publish_formal_portable_stage_coverage_proof_artifact,
    revalidate_portable_formal_stage_coverage_proof_artifact,
)
from lightcone_spec.experiments.formal_stage_execution import (
    FormalServingExecutionRebuildInput,
    FormalStageSourceRebuildInput,
    build_source_e1_recipe_anchor_authority_artifact,
    load_e1_recipe_anchor_authority_artifact,
    publish_e1_recipe_anchor_authority_artifact,
    publish_formal_stage_source_rebuild_input,
)
from lightcone_spec.experiments.formal_stage_prefix import (
    FORMAL_STAGE_EXECUTION_REBUILD_SHARD_KIND,
    FORMAL_STAGE_PREFIX_ORDER,
    FormalStageExecutionRebuildShard,
    bind_formal_stage_prefix_artifact,
    load_and_rebuild_formal_stage_prefix,
    materialize_next_formal_stage_from_prefix,
    publish_formal_stage_execution_rebuild_shard,
    publish_formal_stage_prefix_artifact,
    reduce_formal_stage_prefix,
    verify_signed_formal_stage_prefix_result,
)
from lightcone_spec.experiments.gpu_fleet import (
    GpuFleetInventory,
    HostInventoryBinding,
    assemble_gpu_fleet_inventory,
)
from lightcone_spec.experiments.gpu_hour_authority import (
    materialize_prospective_stage_gpu_hour_envelope,
    materialize_staged_prospective_gpu_hour_envelope,
    verify_registered_prospective_gpu_hour_authority,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuDispatchPlan,
    GpuDispatchPlanningContext,
    GpuInventory,
    InterferenceEnvelope,
)
from lightcone_spec.experiments.industrial_analysis import (
    _DISABLED_SESSION_RUN_FIELDS,
    BoundArtifact,
    E3bLongContextRawFamilyInput,
    IndustrialBlockEvidence,
    IndustrialCellEvidence,
    RawEvidenceAliasManifest,
    _validate_allocation_free_performance,
    _validate_disabled_session_run_fields,
    raw_evidence_alias_manifest_from_dict,
    reduce_confirmation_family_power,
    reduce_e2_stage_from_raw,
    reduce_e3b_long_context_from_raw,
    reduce_evidence_alias,
    reduce_industrial_schema_v3,
)
from lightcone_spec.experiments.interference_authority import (
    InterferenceCalibrationBlockedError,
    materialize_interference_calibration_bootstrap_authority,
    require_release_interference_attester,
)
from lightcone_spec.experiments.inventory import (
    build_serial_interference_envelope,
    collect_gpu_inventory,
)
from lightcone_spec.experiments.onlinespec import (
    ONLINE_SPEC_METHODS,
    ONLINE_SPEC_STUDY_METHODS,
    ONLINE_SPEC_TUNING_STAGES,
    OnlineSpecCandidate,
    OnlineSpecManifest,
    OnlineSpecSelection,
    OnlineSpecTuningMeasurement,
    compare_onlinespec,
    onlinespec_candidates,
    onlinespec_tuning_stage,
    reduce_onlinespec_tuning_stage,
    select_onlinespec,
    select_onlinespec_heldout_anchor,
    verify_onlinespec_source_checkout,
)
from lightcone_spec.experiments.planning import (
    BudgetDispositionStatus,
    BudgetObservationReceipt,
    BudgetPlan,
    ConfirmationFamilyPowerReductionArtifact,
    DispositionStatus,
    EvidenceDependenceMap,
    FamilyActivationArtifact,
    budget_inventory_identity_from_gpu_inventory,
    build_evidence_dependence_map,
    estimate_industrial_budget,
    materialize_confirmation_pilots,
    materialize_confirmation_prefix,
    materialize_e2_final_recipe,
    materialize_industrial_budgets,
    reduce_e1_activation,
    reduce_e2_activation,
)
from lightcone_spec.experiments.planning_artifacts import (
    budget_load_binding_from_dict,
    budget_materialization_authority_binding_from_dict,
    budget_materialization_authority_binding_to_dict,
    budget_plan_from_dict,
    budget_plan_to_dict,
    budget_policy_from_dict,
    capacity_envelope_from_dict,
    confirmation_family_identity_from_dict,
    confirmation_family_power_reduction_artifact_to_dict,
    e1_pareto_artifact_from_dict,
    e2_final_recipe_artifact_from_dict,
    e2_stage_reduction_artifact_to_dict,
    evidence_alias_reduction_artifact_from_dict,
    evidence_alias_reduction_artifact_to_dict,
    evidence_dependence_map_from_dict,
    evidence_dependence_map_to_dict,
    experiment_budget_from_dict,
    family_activation_artifact_from_dict,
    family_activation_artifact_to_dict,
    industrial_budget_report_to_dict,
    reducer_activation_artifact_from_dict,
    reducer_activation_artifact_to_dict,
    sealed_e3a_selection_from_dict,
)
from lightcone_spec.experiments.preflight_authority import (
    PREFLIGHT_COVERAGE_PROTOCOL_SHA256,
    PreflightCoverageReceipt,
    PreflightExecutionSourceAuthority,
    PreflightSealControlBinding,
    materialize_pointer_preflight_coverage,
    preflight_coverage_control_lineage_sha256,
    require_complete_preflight_coverage,
    verify_preflight_coverage,
)
from lightcone_spec.experiments.protocol import (
    DFLASH_LOSS_POSITION_DECAY,
    HISTORICAL_EVIDENCE_CLASSIFICATION,
    TUNING_STAGES,
    assert_confirmation_slice_config,
    assert_historical_matched_recipe_diagnostic_configs,
    confirmation_blocks,
    onlinespec_blocks,
    select_static_load,
    tuning_candidates,
    tuning_stage,
)
from lightcone_spec.experiments.registry import (
    ExperimentReceipt,
    ExperimentRegistry,
    LockedOutput,
    build_industrial_registry,
    build_legacy_industrial_registry,
)
from lightcone_spec.experiments.runner import (
    collect_onlinespec_performance,
    collect_preliminary_confirmation_performance,
    measure_onlinespec_controlled_slice,
    measure_preliminary_controlled_slice,
    run_onlinespec_confirmation_slice,
    run_preliminary_confirmation_slice,
    run_preliminary_greedy_target_reference,
    run_preliminary_natural_replication_slice,
)
from lightcone_spec.experiments.runtime_metrics import (
    RuntimeMetricsAuthority,
    bind_compile_runtime_metrics,
    bind_fresh_process_runtime_metrics_from_terminal_receipts,
    bind_native_runtime_metrics,
    build_runtime_metrics_authority,
    reduce_runtime_metrics,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.selection import (
    CandidateMeasurement,
    SelectionArtifact,
    SliceMeasurement,
    reduce_tuning_stage,
    select_heldout_anchor,
    select_shared_config,
)
from lightcone_spec.experiments.stage_activation import (
    RegistryStageActivationArtifact,
    RegistryStageDispositionStatus,
    is_serving_interference_calibration_cell,
    materialize_registry_stage_activation,
    registry_stage_activation_from_dict,
    registry_stage_activation_to_dict,
    verify_pointer_preflight_stage_activation,
    verify_registry_stage_activation,
)
from lightcone_spec.experiments.stage_capacity import (
    STAGE_CAPACITY_GATE_PROTOCOL_SHA256,
    StageCapacityGate,
    StageCapacitySchedule,
    materialize_stage_capacity_gate_from_raw_sources,
    revalidate_stage_capacity_gate_sources,
    stage_capacity_control_lineage_sha256,
)
from lightcone_spec.experiments.stage_materialization import (
    E1Geometry,
    E2CandidateRecipe,
    FormalGpuHourAuthorityBlocked,
    GpuHourEstimate,
    StageCellDisposition,
    StageCoverageReceipt,
    default_e2_recipe_grid_authority,
    materialize_e0_from_signed_compatibility,
    materialize_e1_first_slice,
    materialize_e1a,
    materialize_e2_round,
    materialize_e3a,
    materialize_e3b,
    materialize_e3b_excluded_pilots,
    materialize_e4_profiler,
    materialize_e4_strength2_screen,
    materialize_e4_winner_neighborhood,
    materialize_e5,
    materialize_e6,
    materialize_preflight,
    materialize_tts_calibration,
    reduce_stage_gpu_hour_envelope_from_signed_pilots,
)
from lightcone_spec.experiments.statistics import (
    HardwareEnvelope,
    evaluate_speed_gate,
)
from lightcone_spec.experiments.tts_calibration_authority import (
    build_formal_tts_calibration_reduction_proof_artifact,
    publish_formal_tts_calibration_reduction_proof_artifact,
    revalidate_formal_tts_calibration_reduction_proof_artifact,
)
from lightcone_spec.experiments.workload_authority import (
    FORMAL_WORKLOAD_PROTOCOLS,
    bind_authorized_formal_workload_authority,
    formal_workload_authority_cli_artifact,
    formal_workload_authority_from_cli_artifact,
    revalidate_authorized_formal_workload_authority,
)
from lightcone_spec.locking import ModelLock, prepare_models, resolve_model_lock
from lightcone_spec.orchestration import (
    PreliminarySpeedStudyManifest,
    render_onlinespec_runtime_plan,
    render_onlinespec_tuning_runtime_plan,
    render_replication_runtime_plan,
    render_runtime_plan,
    render_static_load_runtime_plan,
    render_target_only_runtime_plan,
    render_tuning_runtime_plan,
)
from lightcone_spec.orchestration.execution_bundle import (
    ExecutionBundleBlockedError,
    InterferenceCalibrationExecutionAuthority,
    execute_dispatch_wave_bundles,
)
from lightcone_spec.orchestration.industrial import IndustrialPhysicalAssignment
from lightcone_spec.orchestration.manifest import (
    PRELIMINARY_DIAGNOSTIC_ONLY,
    PRELIMINARY_SPEED_STUDY_MANIFEST_KIND,
)
from lightcone_spec.orchestration.remote_dispatch import (
    MAX_REQUEST_BYTES,
    execute_host_local_wave_request,
)
from lightcone_spec.runtime.content_authorization import (
    ContentJsonArtifactBinding,
    ContentVerificationReceipt,
    VerifiedDatasetContentRelease,
    VerifiedPreparedModelContentRelease,
    VerifiedReleaseWorkloadSources,
    derive_stage_content_verification_receipt,
    verify_and_reserve_content_authorizations,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.scientific_signing import (
    rebuild_scientific_signed_proof_wrapper,
)
from lightcone_spec.runtime.scientific_source_validation import (
    PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES,
    publish_scientific_source_validation_artifact,
)
from lightcone_spec.sglang_bridge import (
    SGLangHTTPClient,
    sglang_adaptation_sha256,
    verify_patched_checkout,
)
from lightcone_spec.telemetry import OUTPUT_HASH_FORMAT, load_completed_evidence


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is not a readable regular file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        body = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            body.extend(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
            or not stat.S_ISREG(current.st_mode)
        ):
            raise ValueError(f"{label} changed while it was read: {path}")
        return bytes(body)
    finally:
        os.close(descriptor)


def _publish_immutable_bytes(path: Path, body: bytes, *, label: str) -> None:
    try:
        existing = _read_regular_bytes(path, label=label)
    except ValueError:
        if path.exists() or path.is_symlink():
            raise
    else:
        if existing != body:
            raise ValueError(f"refusing to overwrite immutable artifact {path}")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if _read_regular_bytes(path, label=label) != body:
                raise ValueError(f"refusing to overwrite immutable artifact {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: str | Path, value: object) -> None:
    output = Path(os.path.abspath(os.fspath(path)))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or output.parent.resolve() != output.parent:
        raise ValueError("artifact parent must be an existing resolved directory")
    body = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = _canonical_sha256(value)
    sidecar = Path(f"{output}.sha256")
    _publish_immutable_bytes(output, body, label="JSON artifact")
    _publish_immutable_bytes(
        sidecar,
        f"{digest}\n".encode("ascii"),
        label="JSON artifact sidecar",
    )
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _canonical_sha256(value: object) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(body).hexdigest()


def _strict_json_bytes(body: bytes, *, label: str) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"{label} contains non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error

    def reject_nonfinite(item: object) -> None:
        if type(item) is float and not math.isfinite(item):
            raise ValueError(f"{label} contains a non-finite JSON number")
        if type(item) is dict:
            for nested in item.values():
                reject_nonfinite(nested)
        elif type(item) is list:
            for nested in item:
                reject_nonfinite(nested)

    reject_nonfinite(value)
    return value


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_request_output_identity(row: dict[str, object]) -> None:
    if row.get("output_hash_format") != OUTPUT_HASH_FORMAT:
        raise ValueError("industrial request has a wrong output hash format")
    serialized = row.get("output_token_ids")
    if not isinstance(serialized, str):
        raise TypeError("industrial request lacks ordered output token IDs")
    try:
        token_ids = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("industrial request has malformed output token IDs") from exc
    if (
        not isinstance(token_ids, list)
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in token_ids
        )
        or json.dumps(token_ids, separators=(",", ":")) != serialized
    ):
        raise ValueError("industrial request has non-canonical output token IDs")
    output_tokens = row.get("output_tokens")
    if (
        not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or len(token_ids) != output_tokens
    ):
        raise ValueError("industrial request token IDs do not cover its output")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if (
        row.get("output_token_ids_sha256") != digest
        or row.get("output_sha256") != digest
    ):
        raise ValueError("industrial request token-ID digest is inconsistent")


def _file_sha256(path: str | Path) -> str:
    source = Path(os.path.abspath(os.fspath(path)))
    return hashlib.sha256(
        _read_regular_bytes(source, label="content-bound artifact")
    ).hexdigest()


def _load_bound_json(path: str | Path) -> object:
    source = Path(os.path.abspath(os.fspath(path)))
    value = _strict_json_bytes(
        _read_regular_bytes(source, label="JSON artifact"),
        label="JSON artifact",
    )
    sidecar = Path(f"{source}.sha256")
    try:
        sidecar_value = _read_regular_bytes(
            sidecar, label="JSON artifact sidecar"
        ).decode("ascii")
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            f"JSON artifact sidecar is missing or invalid: {source}"
        ) from error
    if sidecar_value != f"{_canonical_sha256(value)}\n":
        raise ValueError(f"JSON artifact sidecar is missing or invalid: {source}")
    return value


def _load_formal_registry_receipt_path(
    path: str | Path,
    *,
    now_ns: int,
):
    """Load a bounded schema-5 proof-replay registry layer."""

    source = Path(os.path.abspath(os.fspath(path)))
    proof = CanonicalJsonProofBinding.bind(source)
    value = proof.reopen()
    if value.get("kind") == FORMAL_REGISTRY_LAYER_ARTIFACT_KIND:
        return load_formal_registry_verification_receipt_path(
            proof.absolute_path,
            now_ns=now_ns,
        )
    return load_formal_registry_verification_receipt_path(
        proof.absolute_path,
        now_ns=now_ns,
    )


def _load_formal_scientific_signed_path(
    path: str | Path,
    *,
    artifact_type: str,
    decoder,
    now_ns: int,
):
    """Load a compact reducer wrapper; ProtocolLock has no raw fallback."""

    binding = CanonicalJsonProofBinding.bind(path)
    value = binding.reopen()
    if value.get("kind") == "lightcone_scientific_signed_proof_wrapper":
        signed = rebuild_scientific_signed_proof_wrapper(
            binding.absolute_path,
            now_ns=now_ns,
        )
    elif artifact_type == "protocol-lock":
        raise ValueError("formal ProtocolLock requires a proof-replay wrapper")
    else:
        signed = decoder(value)
    if CanonicalJsonProofBinding.bind(path) != binding:
        raise RuntimeError("formal scientific signed source changed")
    return signed


def _artifact_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.suffix.lower() == ".json":
        return _canonical_sha256(_load_bound_json(source))
    return _file_sha256(source)


def _load_bound_run_config(path: str | Path) -> RunConfig:
    source = Path(path)
    config = load_run_config(source)
    sidecar = Path(f"{source}.sha256")
    if not sidecar.is_file() or sidecar.read_text(
        encoding="utf-8"
    ).strip() != run_config_sha256(config):
        raise ValueError(f"run-config sidecar is missing or invalid: {source}")
    return config


_INDUSTRIAL_REGISTRY_GENERATOR = (
    "lightcone_spec.experiments.registry.build_industrial_registry:signed-staged-v1"
)
_LEGACY_INDUSTRIAL_REGISTRY_GENERATOR = (
    "lightcone_spec.experiments.registry.build_legacy_industrial_registry:v3"
)
# No hardware-rooted/provider signing identity is registered in this source
# release.  Self-authored JSON plus hashes proves content consistency, not GPU
# provenance, and therefore cannot mint a claim-bearing receipt.
_TRUSTED_HARDWARE_ATTESTER_ID: str | None = None


def _trusted_attester_unavailable(label: str) -> RuntimeError:
    return RuntimeError(f"{label} is BLOCKED: trusted_hardware_attester_unavailable")


def _industrial_registry_artifact(
    registry: ExperimentRegistry,
    *,
    base_port: int,
    cache_root: str,
    evidence_root: str,
    seed: int,
) -> dict:
    return {
        "schema_version": 3,
        "generator": (
            _LEGACY_INDUSTRIAL_REGISTRY_GENERATOR
            if registry.materialization_mode == "legacy_diagnostic"
            else _INDUSTRIAL_REGISTRY_GENERATOR
        ),
        "parameters": {
            "logical_gpu_slots": list(registry.gpu_uuids),
            "base_port": base_port,
            "cache_root": cache_root,
            "evidence_root": evidence_root,
            "seed": seed,
        },
        "registry_sha256": registry.sha256,
        "registry": registry.to_dict(),
    }


def _load_industrial_registry(path: str | Path) -> ExperimentRegistry:
    value = _load_bound_json(path)
    if not isinstance(value, dict):
        raise TypeError("industrial registry artifact must be an object")
    if set(value) != {
        "schema_version",
        "generator",
        "parameters",
        "registry_sha256",
        "registry",
    }:
        raise ValueError("industrial registry artifact fields do not match schema")
    generator = value.get("generator")
    if value.get("schema_version") != 3 or generator not in {
        _INDUSTRIAL_REGISTRY_GENERATOR,
        _LEGACY_INDUSTRIAL_REGISTRY_GENERATOR,
    }:
        raise ValueError("industrial registry generator identity mismatch")
    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        raise TypeError("industrial registry parameters are missing")
    expected_parameters = {
        "logical_gpu_slots",
        "base_port",
        "cache_root",
        "evidence_root",
        "seed",
    }
    if set(parameters) != expected_parameters:
        raise ValueError("industrial registry parameter fields do not match schema")
    logical_gpu_slots = parameters.get("logical_gpu_slots")
    if (
        not isinstance(logical_gpu_slots, list)
        or not logical_gpu_slots
        or len(set(logical_gpu_slots)) != len(logical_gpu_slots)
        or not all(
            isinstance(item, str)
            and item.strip()
            and "\n" not in item
            and "\r" not in item
            for item in logical_gpu_slots
        )
    ):
        raise ValueError("industrial registry requires unique logical rank slots")
    if (
        type(parameters["base_port"]) is not int
        or type(parameters["seed"]) is not int
        or type(parameters["cache_root"]) is not str
        or not parameters["cache_root"]
        or type(parameters["evidence_root"]) is not str
        or not parameters["evidence_root"]
    ):
        raise TypeError("industrial registry parameter types do not match schema")
    registry_builder = (
        build_legacy_industrial_registry
        if generator == _LEGACY_INDUSTRIAL_REGISTRY_GENERATOR
        else build_industrial_registry
    )
    registry = registry_builder(
        gpu_uuids=tuple(logical_gpu_slots),
        base_port=parameters["base_port"],
        cache_root=parameters["cache_root"],
        evidence_root=parameters["evidence_root"],
        seed=parameters["seed"],
    )
    if value.get("registry_sha256") != registry.sha256:
        raise ValueError("industrial registry content digest mismatch")
    if value.get("registry") != registry.to_dict():
        raise ValueError(
            "industrial registry declarations were edited after generation"
        )
    return registry


def _load_gpu_inventory(path: str | Path) -> GpuInventory:
    value = _load_bound_json(path)
    inventory = GpuInventory.from_dict(value)
    if inventory.sha256 != _canonical_sha256(inventory.to_dict()):
        raise ValueError("GPU inventory canonical identity mismatch")
    return inventory


def _load_interference_envelope(path: str | Path) -> InterferenceEnvelope:
    value = _load_bound_json(path)
    envelope = InterferenceEnvelope.from_dict(value)
    if envelope.sha256 != _canonical_sha256(envelope.to_dict()):
        raise ValueError("interference envelope canonical identity mismatch")
    return envelope


def _load_gpu_fleet_inventory(path: str | Path) -> GpuFleetInventory:
    value = _load_bound_json(path)
    fleet = GpuFleetInventory.from_dict(value)
    if fleet.sha256 != _canonical_sha256(fleet.to_dict()):
        raise ValueError("GPU fleet inventory canonical identity mismatch")
    return fleet


def _load_budget_plan(path: str | Path) -> BudgetPlan:
    return budget_plan_from_dict(_load_bound_json(path))


def _load_budget_materialization_inputs(
    *,
    policy_path: str | Path,
    load_binding_paths: list[str],
    capacity_envelope_path: str | Path,
    capacity_manifest_path: str | Path | None = None,
    capacity_verification_receipt_path: str | Path | None = None,
):
    policy = budget_policy_from_dict(_load_bound_json(policy_path))
    bindings = tuple(
        budget_load_binding_from_dict(_load_bound_json(path))
        for path in load_binding_paths
    )
    if len({binding.cell_id for binding in bindings}) != len(bindings):
        raise ValueError("duplicate budget load binding")
    capacity = capacity_envelope_from_dict(_load_bound_json(capacity_envelope_path))
    if (capacity_manifest_path is None) != (capacity_verification_receipt_path is None):
        raise ValueError(
            "capacity manifest and verification receipt must be supplied together"
        )
    capacity_authority = (
        None
        if capacity_manifest_path is None
        else bind_capacity_authority(
            capacity_manifest_path,
            capacity_verification_receipt_path,
        )
    )
    return policy, bindings, capacity, capacity_authority


def _load_budget_activation_bundle(
    *,
    activation_paths: list[str],
    family_activation_paths: list[str],
    family_power_plan_paths: list[str],
):
    activations = tuple(
        artifact
        for path in activation_paths
        if (artifact := _load_stage_activation_plan(path)) is not None
    )
    if len({artifact.sha256 for artifact in activations}) != len(activations):
        raise ValueError("duplicate reducer activation artifact")
    family_activations = _load_family_activations(family_activation_paths)
    family_power_reductions = _load_family_power_reductions(family_power_plan_paths)
    if not activations and not family_activations:
        raise RuntimeError(
            "industrial budget materialization is BLOCKED: "
            "reducer_owned_activation_manifest_missing; no bound reducer-owned "
            "activation manifest was supplied"
        )
    return activations, family_activations, family_power_reductions


def _rematerialize_budget_plan(
    *,
    registry: ExperimentRegistry,
    gpu_inventory: GpuInventory,
    declared_plan_path: str | Path,
    activation_paths: list[str],
    family_activation_paths: list[str],
    family_power_plan_paths: list[str],
    policy_path: str | Path,
    load_binding_paths: list[str],
    capacity_envelope_path: str | Path,
    capacity_manifest_path: str | Path | None = None,
    capacity_verification_receipt_path: str | Path | None = None,
    require_ready: bool = True,
) -> tuple[BudgetPlan, tuple, tuple, tuple]:
    """Rebuild one plan from raw authority and reject serialized-plan trust."""

    activations, family_activations, family_power_reductions = (
        _load_budget_activation_bundle(
            activation_paths=activation_paths,
            family_activation_paths=family_activation_paths,
            family_power_plan_paths=family_power_plan_paths,
        )
    )
    (
        policy,
        load_bindings,
        capacity_envelope,
        capacity_authority,
    ) = _load_budget_materialization_inputs(
        policy_path=policy_path,
        load_binding_paths=load_binding_paths,
        capacity_envelope_path=capacity_envelope_path,
        capacity_manifest_path=capacity_manifest_path,
        capacity_verification_receipt_path=(capacity_verification_receipt_path),
    )
    rematerialized = materialize_industrial_budgets(
        registry,
        activations=activations,
        family_activations=family_activations,
        family_power_reductions=family_power_reductions,
        load_bindings=load_bindings,
        policy=policy,
        inventory=budget_inventory_identity_from_gpu_inventory(gpu_inventory),
        capacity_envelope=capacity_envelope,
        capacity_authority=capacity_authority,
    )
    declared = _load_budget_plan(declared_plan_path)
    if declared != rematerialized or declared.sha256 != rematerialized.sha256:
        raise ValueError(
            "serialized BudgetPlan differs from first-party rematerialization"
        )
    if require_ready:
        declared.require_ready()
    return (
        declared,
        activations,
        family_activations,
        family_power_reductions,
    )


def _industrial_receipt_from_value(value: object) -> ExperimentReceipt:
    if not isinstance(value, dict):
        raise TypeError("industrial dependency receipt must be an object")
    expected = {
        "experiment",
        "registry_sha256",
        "runtime_sha256",
        "split_sha256",
        "completed_cells_sha256",
        "dependency_receipts",
        "outputs",
        "selection_state",
    }
    if set(value) != expected:
        raise ValueError("industrial dependency receipt fields do not match schema")

    def locked_rows(name: str) -> tuple[LockedOutput, ...]:
        rows = value.get(name)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise TypeError(f"industrial receipt {name} must be a list")
        if any(set(row) != {"name", "content_sha256"} for row in rows):
            raise ValueError(f"industrial receipt {name} row fields differ")
        return tuple(
            LockedOutput(
                name=row["name"],
                content_sha256=row["content_sha256"],
            )
            for row in rows
        )

    return ExperimentReceipt(
        experiment=value["experiment"],
        registry_sha256=value["registry_sha256"],
        runtime_sha256=value["runtime_sha256"],
        split_sha256=value["split_sha256"],
        completed_cells_sha256=value["completed_cells_sha256"],
        dependency_receipts=locked_rows("dependency_receipts"),
        outputs=locked_rows("outputs"),
        selection_state=value["selection_state"],
    )


def _load_industrial_receipts(paths: list[str]) -> tuple[ExperimentReceipt, ...]:
    if paths and _TRUSTED_HARDWARE_ATTESTER_ID is None:
        raise _trusted_attester_unavailable("industrial dependency receipts")
    return tuple(
        _industrial_receipt_from_value(_load_bound_json(path)) for path in paths
    )


def _load_registry_stage_activation_manifest(
    path: str | Path,
) -> RegistryStageActivationArtifact:
    value = _load_bound_json(path)
    expected = {
        "schema_version",
        "kind",
        "registry_artifact",
        "experiment",
        "runtime_artifact",
        "split_artifact",
        "dependency_receipts",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("registry-stage activation manifest fields differ")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("kind") != ("industrial_registry_stage_activation_manifest")
    ):
        raise ValueError("registry-stage activation manifest identity mismatch")

    def artifact_path(name: str) -> str:
        candidate = value.get(name)
        if (
            not isinstance(candidate, str)
            or not candidate.strip()
            or "\n" in candidate
            or "\r" in candidate
        ):
            raise ValueError(f"registry-stage manifest {name} is invalid")
        return candidate

    experiment = value.get("experiment")
    if (
        not isinstance(experiment, str)
        or not experiment.strip()
        or "\n" in experiment
        or "\r" in experiment
    ):
        raise ValueError("registry-stage manifest experiment is invalid")
    dependency_paths = value.get("dependency_receipts")
    if (
        not isinstance(dependency_paths, list)
        or any(
            not isinstance(item, str)
            or not item.strip()
            or "\n" in item
            or "\r" in item
            for item in dependency_paths
        )
        or len(dependency_paths) != len(set(dependency_paths))
    ):
        raise ValueError(
            "registry-stage dependency receipt paths must be unique strings"
        )
    registry = _load_industrial_registry(artifact_path("registry_artifact"))
    artifact = materialize_registry_stage_activation(
        registry,
        experiment=experiment,
        dependency_receipts=_load_industrial_receipts(dependency_paths),
        runtime_sha256=_artifact_sha256(artifact_path("runtime_artifact")),
        split_sha256=_artifact_sha256(artifact_path("split_artifact")),
    )
    verify_registry_stage_activation(registry, artifact)
    return artifact


def _load_preflight_pointer_activation_manifest(
    path: str | Path,
) -> RegistryStageActivationArtifact:
    value = _load_bound_json(path)
    expected = {
        "schema_version",
        "kind",
        "registry_artifact",
        "source_authority",
        "activation",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("preflight pointer activation manifest fields differ")
    if value.get("schema_version") != 1 or value.get("kind") != (
        "formal_preflight_pointer_activation_manifest"
    ):
        raise ValueError("preflight pointer activation manifest identity differs")
    registry_path = value.get("registry_artifact")
    source_path = value.get("source_authority")
    if not isinstance(registry_path, str) or not isinstance(source_path, str):
        raise TypeError("preflight pointer manifest paths must be strings")
    registry = _load_industrial_registry(registry_path)
    source = PreflightExecutionSourceAuthority.from_dict(_load_bound_json(source_path))
    source.revalidate(registry)
    activation = registry_stage_activation_from_dict(value.get("activation"))
    verify_pointer_preflight_stage_activation(
        registry,
        activation,
        source_authority_sha256=source.sha256,
    )
    return activation


def _load_stage_activation_plan(path: str | None):
    if path is None:
        return None
    value = _load_bound_json(path)
    if (
        isinstance(value, dict)
        and value.get("kind") == "industrial_registry_stage_activation_manifest"
    ):
        return _load_registry_stage_activation_manifest(path)
    if (
        isinstance(value, dict)
        and value.get("kind") == "formal_preflight_pointer_activation_manifest"
    ):
        return _load_preflight_pointer_activation_manifest(path)
    if (
        isinstance(value, dict)
        and value.get("artifact_kind") == "registry_stage_activation"
    ):
        raise ValueError(
            "registry-stage consumers require the bound raw activation manifest"
        )
    if (
        isinstance(value, dict)
        and value.get("kind") == "industrial_e2_activation_manifest"
    ):
        registry, receipt, pareto, stage_index, prior = _load_e2_activation_manifest(
            path
        )
        return reduce_e2_activation(
            registry,
            e1_receipt=receipt,
            pareto=pareto,
            stage_index=stage_index,
            prior_reduction=prior,
        )
    artifact = reducer_activation_artifact_from_dict(value)
    if artifact.plan.experiment == "E2":
        raise ValueError("E2 consumers require a bound raw activation manifest")
    return artifact


def _load_family_activations(paths: list[str]):
    artifacts = tuple(
        family_activation_artifact_from_dict(_load_bound_json(path)) for path in paths
    )
    if len({artifact.sha256 for artifact in artifacts}) != len(artifacts):
        raise ValueError("duplicate confirmation family activation artifact")
    return artifacts


def _load_family_power_reductions(paths: list[str]):
    reductions = tuple(
        reduce_confirmation_family_power(
            registry=registry,
            pilot_activation=pilot,
            blocks=blocks,
            hardware_envelope=envelope,
            inventory=inventory,
            confirmation_data_visible=False,
        )
        for registry, pilot, inventory, envelope, blocks in (
            _load_family_power_manifest(path) for path in paths
        )
    )
    if len({reduction.sha256 for reduction in reductions}) != len(reductions):
        raise ValueError("duplicate confirmation family power artifact")
    return reductions


def _validate_family_artifact_bundle(
    registry: ExperimentRegistry,
    *,
    activations,
    power_reductions,
) -> tuple[str, ...]:
    """Validate full pilot/final lineage and return the current selected round."""

    by_round = {}
    for artifact in activations:
        family = artifact.family
        if family.registry_sha256 != registry.sha256:
            raise ValueError("confirmation activation belongs to another registry")
        key = (family.sha256, artifact.activation_round)
        if key in by_round:
            raise ValueError("duplicate confirmation family activation round")
        by_round[key] = artifact
    by_family_power = {}
    for reduction in power_reductions:
        if reduction.family.registry_sha256 != registry.sha256:
            raise ValueError("confirmation power reduction belongs to another registry")
        if reduction.family.sha256 in by_family_power:
            raise ValueError("duplicate confirmation family power reduction")
        by_family_power[reduction.family.sha256] = reduction

    pilots = {
        family_sha256: artifact
        for (family_sha256, round_name), artifact in by_round.items()
        if round_name == "excluded_pilots"
    }
    finals = {
        family_sha256: artifact
        for (family_sha256, round_name), artifact in by_round.items()
        if round_name == "final_prefix"
    }
    if set(finals) - pilots.keys():
        raise ValueError("family final prefix lacks its pilot activation")
    if set(finals) != set(by_family_power):
        raise ValueError(
            "family power reductions must exactly cover final-prefix activations"
        )

    selected: list[str] = []
    for family_sha256, pilot in sorted(pilots.items()):
        expected_pilot = materialize_confirmation_pilots(registry, pilot.family)
        if pilot != expected_pilot:
            raise ValueError("family pilot activation is not reducer-generated")
        final = finals.get(family_sha256)
        if final is None:
            selected.extend(pilot.activated_cell_ids)
            continue
        reduction = by_family_power[family_sha256]
        expected_final = materialize_confirmation_prefix(
            registry,
            family=pilot.family,
            reduction=reduction,
            pilot_activation=pilot,
        )
        if final != expected_final:
            raise ValueError("family final activation is not reducer-generated")
        selected.extend(final.activated_cell_ids)
    if len(selected) != len(set(selected)):
        raise ValueError("confirmation families select overlapping cells")
    return tuple(sorted(selected))


_BUDGET_OBSERVATION_COMPONENTS = (
    "startup_model_load",
    "compile_jit_graph_prewarm",
    "excluded_warmup",
    "scored_arrival",
    "drain",
    "reset_finalization",
    "evidence_flush_shutdown",
    "soak",
    "failure_injection",
    "retry",
    "profiler",
    "download_compile_reservation",
)
_BUDGET_OBSERVATION_KIND = "industrial_budget_observation_receipt_v1"
_RESERVED_GANG_MEASUREMENT = "exclusive_reserved_gang_wall_ms_x_gpu_count"
_WHOLE_INSTANCE_BILLING = "whole_inventory_wall_clock_v1"


def _industrial_physical_assignment_from_dict(
    value: object,
) -> IndustrialPhysicalAssignment:
    expected = {
        "schema_version",
        "kind",
        "inventory_sha256",
        "inventory_source_receipt_sha256",
        "dispatch_plan_sha256",
        "experiment_budget_sha256",
        "budget_plan_sha256",
        "capacity_authority_sha256",
        "budget_materialization_authority_sha256",
        "assignment_sha256",
        "work_item_sha256",
        "gpu_uuids",
        "rank_groups",
        "ports",
        "gang_shape",
        "fixed_instance_gpu_count",
        "fixed_instance_billing_semantics",
        "host_id",
        "topology_group_ids",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("physical assignment receipt fields differ from schema")
    if value.get("schema_version") != 3 or value.get("kind") != (
        "industrial_physical_assignment"
    ):
        raise ValueError("physical assignment receipt identity mismatch")
    if value.get("fixed_instance_billing_semantics") != _WHOLE_INSTANCE_BILLING:
        raise ValueError("physical assignment billing semantics mismatch")
    gang = value.get("gang_shape")
    if not isinstance(gang, dict) or set(gang) != {
        "tensor_parallel_size",
        "data_parallel_size",
    }:
        raise ValueError("physical assignment gang shape is incomplete")
    gpu_uuids = value.get("gpu_uuids")
    rank_groups = value.get("rank_groups")
    ports = value.get("ports")
    topology_group_ids = value.get("topology_group_ids")
    if (
        not isinstance(gpu_uuids, list)
        or not isinstance(rank_groups, list)
        or not all(isinstance(group, list) for group in rank_groups)
        or not isinstance(ports, list)
        or not isinstance(topology_group_ids, list)
        or not all(isinstance(group, list) for group in topology_group_ids)
    ):
        raise TypeError("physical assignment receipt arrays are malformed")
    try:
        assignment = IndustrialPhysicalAssignment(
            inventory_sha256=value["inventory_sha256"],
            inventory_source_receipt_sha256=value["inventory_source_receipt_sha256"],
            dispatch_plan_sha256=value["dispatch_plan_sha256"],
            experiment_budget_sha256=value["experiment_budget_sha256"],
            budget_plan_sha256=value["budget_plan_sha256"],
            capacity_authority_sha256=value["capacity_authority_sha256"],
            budget_materialization_authority_sha256=value[
                "budget_materialization_authority_sha256"
            ],
            assignment_sha256=value["assignment_sha256"],
            work_item_sha256=value["work_item_sha256"],
            gpu_uuids=tuple(gpu_uuids),
            rank_groups=tuple(tuple(group) for group in rank_groups),
            ports=tuple(ports),
            tensor_parallel_size=gang["tensor_parallel_size"],
            data_parallel_size=gang["data_parallel_size"],
            fixed_instance_gpu_count=value["fixed_instance_gpu_count"],
            host_id=value["host_id"],
            topology_group_ids=tuple(tuple(group) for group in topology_group_ids),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("physical assignment receipt is invalid") from error
    if assignment.to_dict() != value:
        raise ValueError("physical assignment receipt is not canonical")
    return assignment


def _validate_assignment_inventory_authority(
    assignment: IndustrialPhysicalAssignment,
    inventory: GpuInventory,
) -> None:
    """Recompute assignment host, UUID, topology, and billing authority."""

    if len(inventory.host_ids) != 1:
        raise ValueError("formal completion requires one whole-instance GPU host")
    host_id = inventory.host_ids[0]
    if (
        assignment.inventory_sha256 != inventory.sha256
        or assignment.inventory_source_receipt_sha256 != inventory.source_receipt_sha256
        or assignment.fixed_instance_gpu_count != len(inventory.devices)
        or assignment.host_id != host_id
    ):
        raise ValueError("physical assignment differs from the bound GPU inventory")
    devices = {device.uuid: device for device in inventory.devices}
    if any(uuid not in devices for uuid in assignment.gpu_uuids) or any(
        devices[uuid].host_id != host_id for uuid in assignment.gpu_uuids
    ):
        raise ValueError("physical assignment names a foreign GPU or host")
    groups = {group.group_id: group for group in inventory.topology_groups}
    for rank_group, group_ids in zip(
        assignment.rank_groups,
        assignment.topology_group_ids,
        strict=True,
    ):
        if assignment.tensor_parallel_size == 1:
            if group_ids:
                raise ValueError("TP1 assignment cannot claim topology groups")
            continue
        rank_set = set(rank_group)
        if any(
            group_id not in groups
            or groups[group_id].host_id != host_id
            or not rank_set <= set(groups[group_id].gpu_uuids)
            or any(
                group_id not in devices[uuid].allowed_topology_groups
                for uuid in rank_group
            )
            for group_id in group_ids
        ):
            raise ValueError("physical TP assignment lacks bound topology authority")


def _industrial_completion_activation_contract(
    registry: ExperimentRegistry,
    *,
    experiment: str,
    runtime_sha256: str,
    split_sha256: str,
    direct_dependency_receipt_sha256: str | None,
    activation_artifact,
    family_activations,
    family_power_reductions,
    require_stage_sealable: bool = False,
) -> tuple[tuple[str, ...], dict[str, dict[str, str]], dict[str, object]]:
    """Return the exact materialized set and immutable stage dispositions."""

    stage_cells = {cell.cell_id: cell for cell in registry.cells_for(experiment)}
    family_rows = tuple(family_activations)
    power_rows = tuple(family_power_reductions)
    dispositions: dict[str, dict[str, str]] = {}
    requires_dependency = bool(registry.definition(experiment).dependencies)
    if requires_dependency and not _is_lower_sha256(direct_dependency_receipt_sha256):
        raise ValueError("stage completion lacks its direct dependency receipt")
    if not requires_dependency and direct_dependency_receipt_sha256 is not None:
        raise ValueError("root stage cannot claim a dependency receipt")

    if experiment in {"E1", "E2"}:
        if activation_artifact is None:
            raise ValueError(f"{experiment} completion requires an activation artifact")
        if family_rows or power_rows:
            raise ValueError("family artifacts do not belong to this stage")
        plan = activation_artifact.plan
        if (
            plan.registry_sha256 != registry.sha256
            or plan.experiment != experiment
            or plan.runtime_sha256 != runtime_sha256
            or plan.split_sha256 != split_sha256
        ):
            raise ValueError("stage activation runtime/split identity mismatch")
        if plan.dependency_receipt_sha256 != direct_dependency_receipt_sha256:
            raise ValueError("stage activation dependency lineage mismatch")
        activation_round = plan.activation_round
        if experiment == "E1" and activation_round != "e3a_locked_reference":
            raise ValueError("E1 completion requires its locked reference activation")
        if experiment == "E2" and activation_round not in {
            "halving_0",
            "halving_1",
            "halving_2",
            "halving_3",
        }:
            raise ValueError("E2 completion has an unknown halving activation")
        if (
            require_stage_sealable
            and experiment == "E2"
            and activation_round != "halving_3"
        ):
            raise ValueError("E2 cannot seal before the final halving activation")
        disposition_rows = activation_artifact.dispositions
    elif experiment in {"E3b", "E5"}:
        if activation_artifact is not None:
            raise ValueError(
                "stage activation artifact does not belong to confirmation"
            )
        if not family_rows:
            raise ValueError(
                "confirmation completion requires family activation artifacts"
            )
        _validate_family_artifact_bundle(
            registry,
            activations=family_rows,
            power_reductions=power_rows,
        )
        by_round = {
            (artifact.family.sha256, artifact.activation_round): artifact
            for artifact in family_rows
        }
        family_ids = {artifact.family.sha256 for artifact in family_rows}
        final_family_ids = {
            family_sha256
            for family_sha256, round_name in by_round
            if round_name == "final_prefix"
        }
        if require_stage_sealable and final_family_ids != family_ids:
            raise ValueError(
                "confirmation stage cannot seal from pilot-only activations"
            )
        activation_round = (
            "final_prefix"
            if final_family_ids == family_ids
            else "excluded_pilots"
            if not final_family_ids
            else "family_incremental"
        )
        current = []
        for family_sha256 in sorted(family_ids):
            final = by_round.get((family_sha256, "final_prefix"))
            pilot = by_round.get((family_sha256, "excluded_pilots"))
            artifact = final if final is not None else pilot
            if artifact is None:  # pragma: no cover - family_ids construction
                raise RuntimeError("confirmation family lost its validated activation")
            if (
                artifact.family.experiment != experiment
                or artifact.family.runtime_sha256 != runtime_sha256
                or artifact.family.split_sha256 != split_sha256
            ):
                raise ValueError("family activation runtime/split identity mismatch")
            current.extend(artifact.dispositions)
        disposition_rows = tuple(current)
    else:
        if family_rows or power_rows:
            raise ValueError("family artifacts do not belong to this stage")
        if not isinstance(activation_artifact, RegistryStageActivationArtifact):
            raise ValueError(
                "generic stage completion requires reducer-owned registry activation"
            )
        verify_registry_stage_activation(registry, activation_artifact)
        if (
            activation_artifact.experiment != experiment
            or activation_artifact.runtime_sha256 != runtime_sha256
            or activation_artifact.split_sha256 != split_sha256
            or activation_artifact.direct_dependency_receipt_sha256
            != direct_dependency_receipt_sha256
        ):
            raise ValueError(
                "registry-stage activation runtime/split/dependency identity mismatch"
            )
        if require_stage_sealable and activation_artifact.status != "AVAILABLE":
            raise ValueError(
                "generic stage cannot seal without an AVAILABLE registry activation"
            )
        status_map = {
            RegistryStageDispositionStatus.ACTIVATED: DispositionStatus.ACTIVATED,
            RegistryStageDispositionStatus.BLOCKED: DispositionStatus.BLOCKED,
            RegistryStageDispositionStatus.NOT_APPLICABLE: (
                DispositionStatus.NOT_APPLICABLE
            ),
        }
        disposition_rows = tuple(
            {
                "cell_id": row.cell_id,
                "status": status_map[row.status],
                "reason_code": row.reason_code,
            }
            for row in activation_artifact.dispositions
        )
        activation_round = activation_artifact.activation_round

    for row in disposition_rows:
        if isinstance(row, dict):
            cell_id = row["cell_id"]
            status = row["status"]
            reason_code = row["reason_code"]
        else:
            cell_id = row.cell_id
            status = row.status
            reason_code = row.reason_code
        if cell_id not in stage_cells or cell_id in dispositions:
            raise ValueError("activation has an unknown or duplicate disposition")
        if not isinstance(status, DispositionStatus):
            raise TypeError("activation disposition status is not canonical")
        cell = stage_cells[cell_id]
        if status is DispositionStatus.ACTIVATED and not cell.runnable:
            raise ValueError("activation promotes a registry-blocked cell")
        if not cell.runnable:
            expected_status = DispositionStatus(cell.status.value)
            if status is not expected_status or reason_code != cell.reason_code:
                raise ValueError("activation changes an immutable registry disposition")
        dispositions[cell_id] = {
            "cell_id": cell_id,
            "status": status.value,
            "reason_code": reason_code,
        }
    if set(dispositions) != set(stage_cells):
        raise ValueError("activation must disposition every stage template exactly")
    activated = tuple(
        sorted(
            cell_id
            for cell_id, row in dispositions.items()
            if row["status"] == DispositionStatus.ACTIVATED.value
        )
    )
    encoded_dispositions = [dispositions[cell_id] for cell_id in sorted(dispositions)]
    binding = {
        "schema_version": 1,
        "kind": "industrial_stage_activation_binding",
        "stage_activation_sha256": (
            None if activation_artifact is None else activation_artifact.sha256
        ),
        "family_activation_sha256s": sorted(
            artifact.sha256 for artifact in family_rows
        ),
        "family_power_reduction_sha256s": sorted(
            reduction.sha256 for reduction in power_rows
        ),
        "direct_dependency_receipt_sha256": direct_dependency_receipt_sha256,
        "activation_round": activation_round,
        "dispositions_sha256": _canonical_sha256(encoded_dispositions),
    }
    return activated, dispositions, binding


def _validate_budget_observation_receipt(
    path: object,
    *,
    expected_sha256: object,
    experiment_budget_sha256: str,
    prepared_receipt_sha256: object,
    cell,
    evidence_root: Path,
    fixed_instance_gpu_count: int,
) -> None:
    if not _is_lower_sha256(prepared_receipt_sha256):
        raise ValueError("serving completion lacks its prepared receipt binding")
    if not isinstance(path, str) or not path:
        raise ValueError("serving completion lacks a budget observation path")
    source = Path(path)
    sidecar = Path(f"{source}.sha256")
    resolved_root = evidence_root.resolve()
    if (
        source.is_symlink()
        or not source.is_file()
        or sidecar.is_symlink()
        or not sidecar.is_file()
        or resolved_root not in source.resolve().parents
    ):
        raise ValueError("budget observation must be bound inside the evidence root")
    if (
        not _is_lower_sha256(expected_sha256)
        or sidecar.read_text(encoding="utf-8") != f"{expected_sha256}\n"
    ):
        raise ValueError("budget observation semantic sidecar mismatch")
    try:
        artifact = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("budget observation is not valid JSON") from error
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "experiment_budget_sha256",
        "budget_observation_sha256",
        "budget",
        "observed_component_ms",
        "measured_gpu_ms",
        "fixed_instance_billed_gpu_ms",
        "terminal_evidence_sha256",
        "observed_wall_ms",
        "registered_wall_delta_ms",
        "registered_gpu_delta_ms",
        "registered_billed_delta_ms",
        "gpu_measurement_semantics",
        "fixed_instance_billing_semantics",
    }
    if not isinstance(artifact, dict) or set(artifact) != expected_fields:
        raise ValueError("budget observation fields differ from schema")
    budget_value = artifact.get("budget")
    if not isinstance(budget_value, dict):
        raise TypeError("budget observation lacks an ExperimentBudget")
    try:
        budget = experiment_budget_from_dict(
            {
                "artifact_kind": "experiment_budget",
                "artifact_sha256": experiment_budget_sha256,
                **budget_value,
            }
        )
    except (TypeError, ValueError) as error:
        raise ValueError("budget observation contains a forged budget") from error
    if (
        budget.sha256 != experiment_budget_sha256
        or budget.cell_id != cell.cell_id
        or budget.experiment != cell.identity.experiment
        or budget.method != cell.identity.method
        or budget.workload_class is not cell.resources.workload_class
        or budget.gpu_count != cell.resources.gpu_count
        or budget.topology != cell.identity.topology
        or budget.measured_gpu_ms is not None
    ):
        raise ValueError("budget observation belongs to another registry cell")
    rows = artifact.get("observed_component_ms")
    if (
        not isinstance(rows, list)
        or tuple(row[0] for row in rows if isinstance(row, list) and len(row) == 2)
        != _BUDGET_OBSERVATION_COMPONENTS
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not isinstance(row[1], int)
            or isinstance(row[1], bool)
            or row[1] < 0
            for row in rows
        )
    ):
        raise ValueError("budget observation component coverage is invalid")
    try:
        observation = BudgetObservationReceipt(
            schema_version=artifact["schema_version"],
            budget=budget,
            observed_component_ms=tuple((row[0], row[1]) for row in rows),
            measured_gpu_ms=artifact["measured_gpu_ms"],
            fixed_instance_billed_gpu_ms=artifact["fixed_instance_billed_gpu_ms"],
            terminal_evidence_sha256=artifact["terminal_evidence_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("budget observation violates its accounting schema") from error
    expected_artifact = {
        "schema_version": 1,
        "artifact_kind": _BUDGET_OBSERVATION_KIND,
        "experiment_budget_sha256": budget.sha256,
        "budget_observation_sha256": observation.sha256,
        "budget": asdict(budget),
        "observed_component_ms": [
            list(row) for row in observation.observed_component_ms
        ],
        "measured_gpu_ms": observation.measured_gpu_ms,
        "fixed_instance_billed_gpu_ms": observation.fixed_instance_billed_gpu_ms,
        "terminal_evidence_sha256": observation.terminal_evidence_sha256,
        "observed_wall_ms": observation.observed_wall_ms,
        "registered_wall_delta_ms": observation.registered_wall_delta_ms,
        "registered_gpu_delta_ms": observation.registered_gpu_delta_ms,
        "registered_billed_delta_ms": observation.registered_billed_delta_ms,
        "gpu_measurement_semantics": _RESERVED_GANG_MEASUREMENT,
        "fixed_instance_billing_semantics": _WHOLE_INSTANCE_BILLING,
    }
    if (
        artifact != expected_artifact
        or expected_sha256 != observation.sha256
        or observation.terminal_evidence_sha256 != prepared_receipt_sha256
        or observation.measured_gpu_ms
        != observation.observed_wall_ms * budget.gpu_count
        or observation.fixed_instance_billed_gpu_ms
        != observation.observed_wall_ms * fixed_instance_gpu_count
    ):
        raise ValueError("budget observation content binding is invalid")


def _analysis_manifest_path(
    manifest_path: Path,
    value: object,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} path must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _analysis_file_binding(
    manifest_path: Path,
    value: object,
    *,
    label: str,
) -> BoundArtifact:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} binding fields do not match schema")
    sha256 = value.get("sha256")
    if not _is_lower_sha256(sha256):
        raise ValueError(f"{label} digest must be lower-case SHA-256")
    return BoundArtifact(
        path=_analysis_manifest_path(manifest_path, value.get("path"), label=label),
        sha256=sha256,
    )


def _analysis_bound_json_path(
    manifest_path: Path,
    value: object,
    *,
    label: str,
) -> Path:
    binding = _analysis_file_binding(manifest_path, value, label=label)
    path = binding.path
    sidecar = Path(f"{path}.sha256")
    if (
        path.is_symlink()
        or not path.is_file()
        or sidecar.is_symlink()
        or not sidecar.is_file()
    ):
        raise ValueError(f"{label} must be a regular bound JSON artifact")
    payload = _load_bound_json(path)
    if _canonical_sha256(payload) != binding.sha256:
        raise ValueError(f"{label} manifest digest mismatch")
    return path


def _runtime_metrics_source_path(
    manifest_path: Path,
    value: object,
    *,
    label: str,
) -> Path:
    """Resolve one raw source path; metric values are never accepted here."""

    return _analysis_manifest_path(manifest_path, value, label=label)


def _load_runtime_metrics_authority_manifest(
    analysis_manifest_path: Path,
    binding: object,
) -> RuntimeMetricsAuthority | None:
    """Build and replay formal runtime authority from path-only raw sources."""

    if binding is None:
        return None
    manifest_path = _analysis_bound_json_path(
        analysis_manifest_path,
        binding,
        label="runtime metrics raw-source manifest",
    )
    value = _load_bound_json(manifest_path)
    expected = {
        "schema_version",
        "kind",
        "compile_sources",
        "fresh_process_sources",
        "native_sources",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("runtime metrics raw-source manifest fields differ")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "runtime_metrics_raw_source_manifest"
    ):
        raise ValueError("runtime metrics raw-source manifest identity differs")
    compile_rows = value.get("compile_sources")
    fresh_rows = value.get("fresh_process_sources")
    native_rows = value.get("native_sources")
    if any(type(rows) is not list for rows in (compile_rows, fresh_rows, native_rows)):
        raise TypeError("runtime metrics raw-source groups must be JSON arrays")

    declared_paths: set[Path] = set()

    def source_path(raw: object, *, label: str) -> Path:
        path = _runtime_metrics_source_path(manifest_path, raw, label=label)
        canonical = path.resolve(strict=False)
        if canonical in declared_paths:
            raise ValueError("runtime metrics raw-source manifest duplicates a path")
        declared_paths.add(canonical)
        return path

    compile_sources = []
    for index, row in enumerate(compile_rows):
        row_expected = {
            "plan",
            "attempt",
            "result_receipt",
            "subject_id",
        }
        if not isinstance(row, dict) or set(row) != row_expected:
            raise ValueError("runtime metrics compile source fields differ")
        subject_id = row.get("subject_id")
        if subject_id is not None and (
            not isinstance(subject_id, str)
            or not subject_id
            or "\n" in subject_id
            or "\r" in subject_id
        ):
            raise ValueError("runtime metrics compile subject_id is invalid")
        compile_sources.append(
            bind_compile_runtime_metrics(
                plan_path=source_path(
                    row.get("plan"),
                    label=f"runtime compile plan {index}",
                ),
                attempt_path=source_path(
                    row.get("attempt"),
                    label=f"runtime compile attempt {index}",
                ),
                result_receipt_path=source_path(
                    row.get("result_receipt"),
                    label=f"runtime compile result {index}",
                ),
                subject_id=subject_id,
            )
        )

    fresh_sources = []
    for index, row in enumerate(fresh_rows):
        if not isinstance(row, dict) or set(row) != {
            "session_plan_sha256",
            "terminal_receipts",
        }:
            raise ValueError("runtime metrics fresh-process source fields differ")
        terminal_rows = row.get("terminal_receipts")
        if type(terminal_rows) is not list or not terminal_rows:
            raise ValueError(
                "runtime metrics fresh-process source needs terminal paths"
            )
        terminals = tuple(
            source_path(
                raw,
                label=f"runtime fresh terminal {index}:{terminal_index}",
            )
            for terminal_index, raw in enumerate(terminal_rows)
        )
        fresh_sources.append(
            bind_fresh_process_runtime_metrics_from_terminal_receipts(
                session_plan_sha256=row.get("session_plan_sha256"),
                terminal_receipt_paths=terminals,
            )
        )

    native_sources = []
    for index, row in enumerate(native_rows):
        if not isinstance(row, dict) or set(row) != {"artifact"}:
            raise ValueError("runtime metrics native source fields differ")
        native_sources.append(
            bind_native_runtime_metrics(
                source_path(
                    row.get("artifact"),
                    label=f"runtime native terminal {index}",
                )
            )
        )
    authority = build_runtime_metrics_authority(
        compile_sources=tuple(compile_sources),
        fresh_process_sources=tuple(fresh_sources),
        native_sources=tuple(native_sources),
    )
    # Reject corrupt, foreign-schema, or changed raw evidence before entering
    # the formal analyzer. The analyzer replays it again at export time.
    reduction = reduce_runtime_metrics(authority)
    reduction.validate_against(authority)
    return authority


def _analysis_hardware_envelope(value: object) -> HardwareEnvelope:
    expected = {
        "gpu_clock_mhz_min",
        "gpu_clock_mhz_max",
        "memory_clock_mhz_min",
        "memory_clock_mhz_max",
        "temperature_c_max",
        "power_watts_min",
        "power_watts_max",
        "power_state",
        "allowed_throttling_reasons",
        "allowed_background_processes",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("hardware envelope fields do not match schema")
    throttling = value.get("allowed_throttling_reasons")
    processes = value.get("allowed_background_processes")
    if not isinstance(throttling, list) or not isinstance(processes, list):
        raise TypeError("hardware envelope allowlists must be JSON arrays")
    return HardwareEnvelope(
        gpu_clock_mhz_min=value["gpu_clock_mhz_min"],
        gpu_clock_mhz_max=value["gpu_clock_mhz_max"],
        memory_clock_mhz_min=value["memory_clock_mhz_min"],
        memory_clock_mhz_max=value["memory_clock_mhz_max"],
        temperature_c_max=value["temperature_c_max"],
        power_watts_min=value["power_watts_min"],
        power_watts_max=value["power_watts_max"],
        power_state=value["power_state"],
        allowed_throttling_reasons=tuple(throttling),
        allowed_background_processes=tuple(processes),
    )


def _analysis_blocks(
    manifest_path: Path,
    value: object,
) -> tuple[IndustrialBlockEvidence, ...]:
    if type(value) is not list or not value:
        raise ValueError("industrial analysis manifest requires block evidence")
    blocks: list[IndustrialBlockEvidence] = []
    for raw_block in value:
        if not isinstance(raw_block, dict) or set(raw_block) != {
            "block",
            "qualification_lock",
            "cells",
        }:
            raise ValueError("industrial analysis block fields do not match schema")
        block = raw_block.get("block")
        raw_cells = raw_block.get("cells")
        if (
            not isinstance(block, int)
            or isinstance(block, bool)
            or not isinstance(raw_cells, list)
            or not raw_cells
        ):
            raise ValueError("industrial analysis block identity is invalid")
        cells: list[IndustrialCellEvidence] = []
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, dict) or set(raw_cell) != {
                "cell_id",
                "terminal_receipts",
                "hardware_receipt",
                "budget_observation",
                "completion_contract",
            }:
                raise ValueError("industrial analysis cell fields do not match schema")
            terminal = raw_cell.get("terminal_receipts")
            if not isinstance(terminal, list) or not terminal:
                raise ValueError("industrial analysis cell requires rank receipts")
            cells.append(
                IndustrialCellEvidence(
                    cell_id=raw_cell.get("cell_id"),
                    terminal_receipts=tuple(
                        _analysis_file_binding(
                            manifest_path,
                            receipt,
                            label="terminal receipt",
                        )
                        for receipt in terminal
                    ),
                    hardware_receipt=_analysis_file_binding(
                        manifest_path,
                        raw_cell.get("hardware_receipt"),
                        label="hardware receipt",
                    ),
                    budget_observation=_analysis_file_binding(
                        manifest_path,
                        raw_cell.get("budget_observation"),
                        label="budget observation",
                    ),
                    completion_contract=_analysis_file_binding(
                        manifest_path,
                        raw_cell.get("completion_contract"),
                        label="schema-v4 completion contract",
                    ),
                )
            )
        blocks.append(
            IndustrialBlockEvidence(
                block=block,
                cells=tuple(cells),
                qualification_lock=_analysis_file_binding(
                    manifest_path,
                    raw_block.get("qualification_lock"),
                    label="qualification lock",
                ),
            )
        )
    return tuple(blocks)


def _analysis_cells(
    manifest_path: Path,
    value: object,
) -> tuple[IndustrialCellEvidence, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("E2 raw manifest requires cell evidence")
    cells: list[IndustrialCellEvidence] = []
    for raw_cell in value:
        if type(raw_cell) is not dict or set(raw_cell) != {
            "cell_id",
            "terminal_receipts",
            "hardware_receipt",
            "budget_observation",
            "completion_contract",
            "itl_timestamp_authority_path",
        }:
            raise ValueError("E2 raw cell fields do not match schema")
        terminal = raw_cell.get("terminal_receipts")
        if type(terminal) is not list or not terminal:
            raise ValueError("E2 raw cell requires terminal rank receipts")
        authority_path = raw_cell.get("itl_timestamp_authority_path")
        if type(authority_path) is not str:
            raise TypeError("E2 ITL timestamp authority path must be an exact string")
        cells.append(
            IndustrialCellEvidence(
                cell_id=raw_cell.get("cell_id"),
                terminal_receipts=tuple(
                    _analysis_file_binding(
                        manifest_path,
                        receipt,
                        label="E2 terminal receipt",
                    )
                    for receipt in terminal
                ),
                hardware_receipt=_analysis_file_binding(
                    manifest_path,
                    raw_cell.get("hardware_receipt"),
                    label="E2 hardware receipt",
                ),
                budget_observation=_analysis_file_binding(
                    manifest_path,
                    raw_cell.get("budget_observation"),
                    label="E2 budget observation",
                ),
                completion_contract=_analysis_file_binding(
                    manifest_path,
                    raw_cell.get("completion_contract"),
                    label="E2 schema-v4 completion contract",
                ),
                itl_timestamp_authority_path=Path(
                    os.path.abspath(
                        _analysis_manifest_path(
                            manifest_path,
                            authority_path,
                            label="E2 ITL timestamp authority",
                        )
                    )
                ),
            )
        )
    return tuple(cells)


def _load_e2_stage_manifest(
    path: str | Path,
    *,
    _seen: frozenset[Path] = frozenset(),
):
    manifest_path = Path(path).resolve()
    if manifest_path in _seen:
        raise ValueError("E2 prior-stage manifests contain a cycle")
    value = _load_bound_json(manifest_path)
    expected = {
        "schema_version",
        "kind",
        "registry_artifact",
        "e1_receipt",
        "pareto",
        "stage_index",
        "prior_stage_manifest",
        "gpu_inventory",
        "hardware_envelope",
        "cells",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("E2 raw stage manifest fields do not match schema")
    stage_index = value.get("stage_index")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 3
        or value.get("kind") != "industrial_e2_stage_reduction_manifest"
        or type(stage_index) is not int
    ):
        raise ValueError("E2 raw stage manifest identity is invalid")
    registry_path = _analysis_bound_json_path(
        manifest_path,
        value.get("registry_artifact"),
        label="E2 registry artifact",
    )
    receipt_path = _analysis_bound_json_path(
        manifest_path,
        value.get("e1_receipt"),
        label="E1 receipt",
    )
    pareto_path = _analysis_bound_json_path(
        manifest_path,
        value.get("pareto"),
        label="E1 Pareto artifact",
    )
    registry = _load_industrial_registry(registry_path)
    inventory_path = _analysis_bound_json_path(
        manifest_path,
        value.get("gpu_inventory"),
        label="E2 GPU inventory",
    )
    inventory = _load_gpu_inventory(inventory_path)
    receipt = _single_industrial_receipt(receipt_path, experiment="E1")
    pareto = e1_pareto_artifact_from_dict(_load_bound_json(pareto_path))
    prior_value = value.get("prior_stage_manifest")
    if stage_index == 0:
        if prior_value is not None:
            raise ValueError("E2 stage zero cannot name prior raw evidence")
        prior = None
    else:
        prior_path = _analysis_bound_json_path(
            manifest_path,
            prior_value,
            label="prior E2 raw stage manifest",
        )
        prior = _load_e2_stage_manifest(
            prior_path,
            _seen=_seen | {manifest_path},
        )
        if prior.stage_index != stage_index - 1:
            raise ValueError("E2 prior raw stage is not the immediate predecessor")
    return reduce_e2_stage_from_raw(
        registry=registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=stage_index,
        cells=_analysis_cells(manifest_path, value.get("cells")),
        hardware_envelope=_analysis_hardware_envelope(value.get("hardware_envelope")),
        inventory=inventory,
        prior_stage_reduction=prior,
        confirmation_data_visible=False,
    )


def _load_e2_activation_manifest(path: str | Path):
    manifest_path = Path(path)
    value = _load_bound_json(manifest_path)
    expected = {
        "schema_version",
        "kind",
        "registry_artifact",
        "e1_receipt",
        "pareto",
        "stage_index",
        "prior_stage_manifest",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("E2 activation manifest fields do not match schema")
    stage_index = value.get("stage_index")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "industrial_e2_activation_manifest"
        or not isinstance(stage_index, int)
        or isinstance(stage_index, bool)
    ):
        raise ValueError("E2 activation manifest identity is invalid")
    registry_path = _analysis_bound_json_path(
        manifest_path,
        value.get("registry_artifact"),
        label="E2 registry artifact",
    )
    receipt_path = _analysis_bound_json_path(
        manifest_path,
        value.get("e1_receipt"),
        label="E1 receipt",
    )
    pareto_path = _analysis_bound_json_path(
        manifest_path,
        value.get("pareto"),
        label="E1 Pareto artifact",
    )
    registry = _load_industrial_registry(registry_path)
    receipt = _single_industrial_receipt(receipt_path, experiment="E1")
    pareto = e1_pareto_artifact_from_dict(_load_bound_json(pareto_path))
    prior_value = value.get("prior_stage_manifest")
    if stage_index == 0:
        if prior_value is not None:
            raise ValueError("E2 stage zero cannot name prior raw evidence")
        prior = None
    else:
        prior_path = _analysis_bound_json_path(
            manifest_path,
            prior_value,
            label="prior E2 raw stage manifest",
        )
        prior = _load_e2_stage_manifest(prior_path)
        if prior.stage_index != stage_index - 1:
            raise ValueError("E2 prior raw stage is not the immediate predecessor")
    return registry, receipt, pareto, stage_index, prior


def _load_family_power_manifest(
    path: str | Path,
) -> tuple[
    ExperimentRegistry,
    FamilyActivationArtifact,
    GpuInventory,
    HardwareEnvelope,
    tuple[IndustrialBlockEvidence, ...],
]:
    manifest_path = Path(path)
    value = _load_bound_json(manifest_path)
    expected = {
        "schema_version",
        "kind",
        "registry_artifact",
        "pilot_activation",
        "gpu_inventory",
        "hardware_envelope",
        "blocks",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("family power manifest fields do not match schema")
    if (
        value.get("schema_version") != 2
        or value.get("kind") != "industrial_family_power_manifest"
    ):
        raise ValueError("family power manifest identity mismatch")
    registry_path = _analysis_bound_json_path(
        manifest_path,
        value.get("registry_artifact"),
        label="registry artifact",
    )
    pilot_path = _analysis_bound_json_path(
        manifest_path,
        value.get("pilot_activation"),
        label="pilot activation",
    )
    registry = _load_industrial_registry(registry_path)
    pilot = family_activation_artifact_from_dict(_load_bound_json(pilot_path))
    inventory_path = _analysis_bound_json_path(
        manifest_path,
        value.get("gpu_inventory"),
        label="family GPU inventory",
    )
    inventory = _load_gpu_inventory(inventory_path)
    envelope = _analysis_hardware_envelope(value.get("hardware_envelope"))
    return (
        registry,
        pilot,
        inventory,
        envelope,
        _analysis_blocks(manifest_path, value["blocks"]),
    )


def _load_industrial_analysis_manifest(
    path: str | Path,
) -> tuple[
    ExperimentRegistry,
    FamilyActivationArtifact,
    FamilyActivationArtifact,
    ConfirmationFamilyPowerReductionArtifact,
    GpuInventory,
    HardwareEnvelope,
    tuple[IndustrialBlockEvidence, ...],
    tuple[RawEvidenceAliasManifest, ...],
    EvidenceDependenceMap | None,
    BoundArtifact | None,
    BoundArtifact | None,
    int,
    int,
    RuntimeMetricsAuthority | None,
]:
    manifest_path = Path(path)
    manifest_sidecar = Path(f"{manifest_path}.sha256")
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_sidecar.is_symlink()
        or not manifest_sidecar.is_file()
    ):
        raise ValueError("industrial analysis manifest must be a regular bound file")
    value = _load_bound_json(manifest_path)
    if isinstance(value, dict) and (
        value.get("kind") == PRELIMINARY_SPEED_STUDY_MANIFEST_KIND
        or value.get("evidence_scope") == PRELIMINARY_DIAGNOSTIC_ONLY
        or value.get("name") == "static-tts-l0-speed-study"
    ):
        raise ValueError(
            "PRELIMINARY_DIAGNOSTIC_ONLY manifests cannot enter industrial analysis"
        )
    expected = {
        "schema_version",
        "kind",
        "registry_artifact",
        "pilot_activation",
        "final_activation",
        "confirmation_power_manifest",
        "gpu_inventory",
        "evidence_alias_manifests",
        "evidence_dependence_map",
        "gpu_attestation",
        "doctor_report",
        "hardware_envelope",
        "bootstrap",
        "blocks",
    }
    allowed_fields = (expected, expected | {"runtime_metrics_manifest"})
    if not isinstance(value, dict) or set(value) not in allowed_fields:
        raise ValueError("industrial analysis manifest fields do not match schema")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 3
        or value.get("kind") != "industrial_analysis_manifest"
    ):
        raise ValueError("industrial analysis manifest identity mismatch")

    registry_path = _analysis_bound_json_path(
        manifest_path,
        value.get("registry_artifact"),
        label="registry artifact",
    )
    pilot_path = _analysis_bound_json_path(
        manifest_path,
        value.get("pilot_activation"),
        label="pilot activation",
    )
    final_path = _analysis_bound_json_path(
        manifest_path,
        value.get("final_activation"),
        label="final activation",
    )
    power_manifest_path = _analysis_bound_json_path(
        manifest_path,
        value.get("confirmation_power_manifest"),
        label="confirmation family power manifest",
    )
    registry = _load_industrial_registry(registry_path)
    pilot = family_activation_artifact_from_dict(_load_bound_json(pilot_path))
    final = family_activation_artifact_from_dict(_load_bound_json(final_path))
    inventory_path = _analysis_bound_json_path(
        manifest_path,
        value.get("gpu_inventory"),
        label="analysis GPU inventory",
    )
    inventory = _load_gpu_inventory(inventory_path)
    raw_alias_bindings = value.get("evidence_alias_manifests")
    if type(raw_alias_bindings) is not list:
        raise ValueError("industrial analysis alias manifests must be a JSON array")
    alias_manifests = tuple(
        raw_evidence_alias_manifest_from_dict(
            _load_bound_json(
                _analysis_bound_json_path(
                    manifest_path,
                    binding,
                    label=f"raw evidence alias manifest {index}",
                )
            )
        )
        for index, binding in enumerate(raw_alias_bindings)
    )
    if len({manifest.sha256 for manifest in alias_manifests}) != len(alias_manifests):
        raise ValueError("duplicate raw evidence alias manifest")
    raw_dependence = value.get("evidence_dependence_map")
    dependence = None
    if raw_dependence is not None:
        dependence_path = _analysis_bound_json_path(
            manifest_path,
            raw_dependence,
            label="evidence dependence map",
        )
        dependence = evidence_dependence_map_from_dict(
            _load_bound_json(dependence_path)
        )
    raw_attestation = value.get("gpu_attestation")
    raw_doctor = value.get("doctor_report")
    if (raw_attestation is None) != (raw_doctor is None):
        raise ValueError(
            "industrial GPU attestation and doctor report must be supplied together"
        )
    gpu_attestation = (
        None
        if raw_attestation is None
        else _analysis_file_binding(
            manifest_path,
            raw_attestation,
            label="industrial GPU attestation",
        )
    )
    doctor_report = (
        None
        if raw_doctor is None
        else _analysis_file_binding(
            manifest_path,
            raw_doctor,
            label="doctor report",
        )
    )
    envelope = _analysis_hardware_envelope(value.get("hardware_envelope"))
    power_registry, power_pilot, power_inventory, power_envelope, power_blocks = (
        _load_family_power_manifest(power_manifest_path)
    )
    if (
        power_registry.sha256 != registry.sha256
        or power_pilot != pilot
        or power_inventory != inventory
        or power_envelope != envelope
    ):
        raise ValueError(
            "analysis family-power manifest differs from registry/pilot/hardware"
        )
    plan = reduce_confirmation_family_power(
        registry=power_registry,
        pilot_activation=power_pilot,
        blocks=power_blocks,
        hardware_envelope=power_envelope,
        inventory=power_inventory,
        confirmation_data_visible=False,
    )

    bootstrap = value.get("bootstrap")
    if not isinstance(bootstrap, dict) or set(bootstrap) != {
        "repetitions",
        "seed",
    }:
        raise ValueError("industrial analysis bootstrap fields do not match schema")
    repetitions = bootstrap.get("repetitions")
    seed = bootstrap.get("seed")
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < 100
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("industrial analysis bootstrap values are invalid")

    blocks = _analysis_blocks(manifest_path, value.get("blocks"))
    runtime_metrics_authority = _load_runtime_metrics_authority_manifest(
        manifest_path,
        value.get("runtime_metrics_manifest"),
    )
    return (
        registry,
        pilot,
        final,
        plan,
        inventory,
        envelope,
        blocks,
        alias_manifests,
        dependence,
        gpu_attestation,
        doctor_report,
        repetitions,
        seed,
        runtime_metrics_authority,
    )


def _load_e3b_long_context_analysis_manifest(
    path: str | Path,
) -> tuple[
    ExperimentRegistry,
    tuple[E3bLongContextRawFamilyInput, ...],
    GpuInventory,
    HardwareEnvelope,
    int,
    int,
]:
    """Reopen path-bound per-family manifests without accepting metric rows."""

    manifest_path = Path(path)
    manifest_sidecar = Path(f"{manifest_path}.sha256")
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_sidecar.is_symlink()
        or not manifest_sidecar.is_file()
    ):
        raise ValueError("E3b long-context manifest must be a regular bound file")
    value = _load_bound_json(manifest_path)
    expected = {
        "schema_version",
        "kind",
        "family_manifests",
        "bootstrap",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("E3b long-context manifest fields do not match schema")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "industrial_e3b_long_context_analysis_manifest"
    ):
        raise ValueError("E3b long-context manifest identity mismatch")
    raw_families = value.get("family_manifests")
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError("E3b long-context manifest requires raw family manifests")
    family_paths = tuple(
        Path(
            os.path.abspath(
                os.fspath(
                    _analysis_bound_json_path(
                        manifest_path,
                        binding,
                        label=f"E3b raw family manifest {index}",
                    )
                )
            )
        )
        for index, binding in enumerate(raw_families)
    )
    if len(set(family_paths)) != len(family_paths):
        raise ValueError("E3b long-context manifest duplicates a raw family path")
    bootstrap = value.get("bootstrap")
    if not isinstance(bootstrap, dict) or set(bootstrap) != {
        "repetitions",
        "seed",
    }:
        raise ValueError("E3b long-context bootstrap fields do not match schema")
    repetitions = bootstrap.get("repetitions")
    seed = bootstrap.get("seed")
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < 100
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed < 2**64
    ):
        raise ValueError("E3b long-context bootstrap values are invalid")

    loaded = tuple(_load_industrial_analysis_manifest(path) for path in family_paths)
    registry = loaded[0][0]
    inventory = loaded[0][4]
    envelope = loaded[0][5]
    if any(
        row[0].sha256 != registry.sha256
        or row[4] != inventory
        or row[5] != envelope
        or row[11] != repetitions
        or row[12] != seed
        for row in loaded
    ):
        raise ValueError(
            "E3b raw families differ in registry, inventory, hardware, or bootstrap"
        )
    families = tuple(
        E3bLongContextRawFamilyInput(
            pilot_activation=row[1],
            final_activation=row[2],
            confirmation_reduction=row[3],
            blocks=row[6],
            evidence_alias_manifests=row[7],
            evidence_dependence_map=row[8],
            gpu_attestation=row[9],
            doctor_report=row[10],
        )
        for row in loaded
    )
    return registry, families, inventory, envelope, repetitions, seed


def _parse_locked_outputs(values: list[str]) -> dict[str, str]:
    return {
        name: _artifact_sha256(path)
        for name, path in _parse_locked_output_paths(values).items()
    }


def _parse_locked_output_paths(values: list[str]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in outputs:
            raise ValueError("locked outputs must be unique NAME=PATH values")
        outputs[name] = path
    return outputs


def _static_load_rows(
    value: object,
    *,
    manifest: PreliminarySpeedStudyManifest,
) -> list[dict]:
    if not isinstance(value, dict):
        raise TypeError("Static load screen must be a schema-v2 terminal artifact")
    expected = {
        "schema_version": 2,
        "phase": "static_load_screen",
        "manifest_sha256": manifest.sha256,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "window_sha256": manifest.controlled_window_hashes["load"],
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError("Static load-screen artifact identity mismatch")
    model_lock_sha256 = value.get("model_lock_sha256")
    if not _is_lower_sha256(model_lock_sha256):
        raise ValueError("Static load-screen model-lock identity is invalid")
    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError("Static load-screen artifact lacks measurement rows")
    return rows


def _preliminary_table_metadata(
    *,
    manifest: PreliminarySpeedStudyManifest,
    selection: SelectionArtifact,
    model_lock: ModelLock,
    config_sha256: dict[str, str],
    source_evidence_sha256: str,
    target_reference_sha256: str,
) -> dict[bytes, bytes]:
    return {
        b"lightcone_schema_version": b"2",
        b"lightcone_manifest_kind": PRELIMINARY_SPEED_STUDY_MANIFEST_KIND.encode(),
        b"lightcone_evidence_scope": PRELIMINARY_DIAGNOSTIC_ONLY.encode(),
        b"lightcone_formal_execution_authorized": b"false",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": model_lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": (
            manifest.sampling_profile_sha256.encode()
        ),
        b"lightcone_execution_policy_sha256": (
            manifest.execution_policy_sha256.encode()
        ),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_config_set_sha256": _canonical_sha256(config_sha256).encode(),
        b"lightcone_source_evidence_sha256": source_evidence_sha256.encode(),
        b"lightcone_target_reference_sha256": target_reference_sha256.encode(),
    }


def _load_preliminary_table(
    path: str | Path,
    *,
    manifest: PreliminarySpeedStudyManifest,
    selection: SelectionArtifact,
    model_lock: ModelLock,
    target_reference: GreedyTargetReference,
) -> pa.Table:
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    expected = {
        b"lightcone_schema_version": b"2",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": model_lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": (
            manifest.sampling_profile_sha256.encode()
        ),
        b"lightcone_execution_policy_sha256": (
            manifest.execution_policy_sha256.encode()
        ),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_target_reference_sha256": target_reference.sha256.encode(),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("preliminary speed table identity metadata mismatch")
    scope_metadata = {
        b"lightcone_manifest_kind": PRELIMINARY_SPEED_STUDY_MANIFEST_KIND.encode(),
        b"lightcone_evidence_scope": PRELIMINARY_DIAGNOSTIC_ONLY.encode(),
        b"lightcone_formal_execution_authorized": b"false",
    }
    present_scope_fields = set(scope_metadata) & set(metadata)
    if present_scope_fields and any(
        metadata.get(key) != value for key, value in scope_metadata.items()
    ):
        raise ValueError("preliminary speed table scope metadata is invalid")
    for key in (
        b"lightcone_config_set_sha256",
        b"lightcone_source_evidence_sha256",
    ):
        value = metadata.get(key, b"").decode(errors="ignore")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("preliminary speed table evidence metadata is invalid")
    return table


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lightcone-spec")
    commands = parser.add_subparsers(dest="command", required=True)
    add_formal_single_operator_parser(commands)

    doctor = commands.add_parser("doctor")
    doctor.add_argument(
        "--project-root",
        help="clean LightCone-Spec checkout containing the runtime manifest",
    )
    doctor.add_argument(
        "--sglang-root",
        help="separate clean checkout with the complete pinned SGLang patch set",
    )
    doctor.add_argument(
        "--path",
        help="legacy shorthand that uses one path for both roots (not ready)",
    )
    doctor.add_argument(
        "--stage-capacity-gate",
        help=(
            "path-bound schema-3 preflight capacity gate; requires the exact "
            "stage schedule"
        ),
    )
    doctor.add_argument(
        "--stage-capacity-schedule",
        help="path-bound schedule used to rederive the supplied capacity gate",
    )
    doctor.add_argument(
        "--stage-capacity-attestation",
        help="root-authorized dynamic capacity control for the exact gate",
    )
    doctor.add_argument(
        "--stage-capacity-activation-sha256",
        help="exact preflight activation identity bound by the capacity control",
    )
    doctor.add_argument(
        "--stage-capacity-now-ns",
        type=int,
        help="verification time for the path-bound raw capacity receipts",
    )

    validate = commands.add_parser("validate-config")
    validate.add_argument("config")

    build = commands.add_parser("build-preliminary-speed-study")
    build.add_argument("--output", required=True)

    bind_workload = commands.add_parser(
        "bind-formal-workload-authority", allow_abbrev=False
    )
    bind_workload.add_argument(
        "--workload",
        choices=tuple(sorted(FORMAL_WORKLOAD_PROTOCOLS)),
        required=True,
    )
    bind_workload.add_argument("--source", required=True)
    bind_workload.add_argument("--content-verification-receipt", required=True)
    bind_workload.add_argument("--now-ns", type=int, required=True)
    bind_workload.add_argument("--output", required=True)

    revalidate_workload = commands.add_parser(
        "revalidate-formal-workload-authority", allow_abbrev=False
    )
    revalidate_workload.add_argument("--authority", required=True)
    revalidate_workload.add_argument("--content-verification-receipt", required=True)
    revalidate_workload.add_argument("--now-ns", type=int, required=True)

    verify_content = commands.add_parser(
        "verify-content-authorizations", allow_abbrev=False
    )
    verify_content.add_argument("--workload-authorization", required=True)
    verify_content.add_argument("--prepared-model-authorization", required=True)
    verify_content.add_argument("--burstgpt-authorization", required=True)
    verify_content.add_argument("--e0-dataset-authorization", required=True)
    verify_content.add_argument(
        "--content-artifact",
        action="append",
        required=True,
        metavar="ARTIFACT_ID=ABSOLUTE_PATH",
    )
    verify_content.add_argument("--replay-store", required=True)
    verify_content.add_argument("--now-ns", type=int, required=True)
    verify_content.add_argument("--output", required=True)

    scope_content = commands.add_parser(
        "scope-content-verification-receipt", allow_abbrev=False
    )
    scope_content.add_argument("--master-receipt", required=True)
    scope_content.add_argument(
        "--stage",
        choices=tuple(stage for stage in FORMAL_STAGE_DAG if stage != "preflight"),
        required=True,
    )
    scope_content.add_argument("--now-ns", type=int, required=True)
    scope_content.add_argument("--output", required=True)

    burstgpt_shape = commands.add_parser(
        "publish-burstgpt-shape-authority", allow_abbrev=False
    )
    burstgpt_shape.add_argument("--content-verification-receipt", required=True)
    burstgpt_shape.add_argument("--now-ns", type=int, required=True)
    burstgpt_shape.add_argument("--output", required=True)

    build_industrial = commands.add_parser("build-industrial-registry")
    build_industrial.add_argument(
        "--logical-gpu-slot",
        nargs="+",
        default=("logical-rank-slot-0", "logical-rank-slot-1"),
        metavar="SLOT",
        help=(
            "one or more stable logical rank slots; physical GPU UUIDs come only from "
            "the inventory and frozen dispatch assignment"
        ),
    )
    build_industrial.add_argument("--base-port", type=int, default=24000)
    build_industrial.add_argument("--cache-root", default="runtime-cache/industrial")
    build_industrial.add_argument("--evidence-root", default="artifacts/industrial")
    build_industrial.add_argument("--seed", type=int, default=20260811)
    build_industrial.add_argument(
        "--legacy-diagnostic",
        action="store_true",
        help=("emit the eager schema-v3 historical matrix; never formal-executable"),
    )
    build_industrial.add_argument("--output", required=True)

    publish_tts_authority = commands.add_parser(
        "publish-tts-calibration-source-authority", allow_abbrev=False
    )
    publish_tts_authority.add_argument("--paper-pdf", required=True)
    publish_tts_authority.add_argument("--paper-source", required=True)
    publish_tts_authority.add_argument("--tuning-window", required=True)
    publish_tts_authority.add_argument("--trainable-plan-authority", required=True)
    publish_tts_authority.add_argument("--drafter-native-loss", required=True)
    publish_tts_authority.add_argument("--output", required=True)

    publish_chronobelief_authority = commands.add_parser(
        "publish-chronobelief-source-authority", allow_abbrev=False
    )
    publish_chronobelief_authority.add_argument("--paper-pdf", required=True)
    publish_chronobelief_authority.add_argument("--tex-source", required=True)
    publish_chronobelief_authority.add_argument("--output", required=True)

    publish_e1_anchor_authority = commands.add_parser(
        "publish-e1-recipe-anchor-authority", allow_abbrev=False
    )
    publish_e1_anchor_authority.add_argument(
        "--trainable-plan-authority", required=True
    )
    publish_e1_anchor_authority.add_argument("--output", required=True)

    publish_lock_git_snapshot = commands.add_parser(
        "publish-formal-protocol-lock-git-snapshot",
        allow_abbrev=False,
    )
    publish_lock_git_snapshot.add_argument("--project-root", required=True)
    publish_lock_git_snapshot.add_argument("--chunk-output-directory", required=True)
    publish_lock_git_snapshot.add_argument("--output", required=True)

    publish_lock_source_proof = commands.add_parser(
        "publish-formal-protocol-lock-source-proof",
        allow_abbrev=False,
    )
    publish_lock_source_proof.add_argument("--protocol-id", required=True)
    publish_lock_source_proof.add_argument("--git-snapshot", required=True)
    publish_lock_source_proof.add_argument(
        "--patch-manifest-relative-path", required=True
    )
    publish_lock_source_proof.add_argument(
        "--english-protocol-relative-path", required=True
    )
    publish_lock_source_proof.add_argument(
        "--chinese-protocol-relative-path", required=True
    )
    publish_lock_source_proof.add_argument(
        "--formal-runtime-authority-manifest", required=True
    )
    publish_lock_source_proof.add_argument("--tts-calibration-authority", required=True)
    publish_lock_source_proof.add_argument("--chronobelief-authority", required=True)
    publish_lock_source_proof.add_argument(
        "--e1-recipe-anchor-authority", required=True
    )
    publish_lock_source_proof.add_argument(
        "--content-verification-receipt", required=True
    )
    publish_lock_source_proof.add_argument("--burstgpt-shape-authority", required=True)
    publish_lock_source_proof.add_argument("--now-ns", type=int, required=True)
    publish_lock_source_proof.add_argument("--output", required=True)

    create_lock = commands.add_parser("create-protocol-lock", allow_abbrev=False)
    create_lock.add_argument("--protocol-id", required=True)
    create_lock.add_argument("--project-root", required=True)
    create_lock.add_argument("--code-git-head")
    create_lock.add_argument("--code-git-tree")
    create_lock.add_argument("--patch-manifest", required=True)
    create_lock.add_argument("--patch-manifest-sha256")
    create_lock.add_argument("--registry-sha256")
    create_lock.add_argument("--english-protocol", required=True)
    create_lock.add_argument("--english-protocol-sha256")
    create_lock.add_argument("--chinese-protocol", required=True)
    create_lock.add_argument("--chinese-protocol-sha256")
    create_lock.add_argument("--tts-calibration-authority", required=True)
    create_lock.add_argument("--chronobelief-authority", required=True)
    create_lock.add_argument("--e1-recipe-anchor-authority", required=True)
    create_lock.add_argument("--formal-runtime-authority-manifest", required=True)
    create_lock.add_argument("--content-verification-receipt", required=True)
    create_lock.add_argument("--content-verification-now-ns", type=int, required=True)
    create_lock.add_argument("--burstgpt-shape-authority", required=True)
    create_lock.add_argument("--output", required=True)

    verify_lock = commands.add_parser("verify-signed-protocol-lock", allow_abbrev=False)
    verify_lock.add_argument("--signed-lock", required=True)
    verify_lock.add_argument("--control-attestation", required=True)
    verify_lock.add_argument("--inventory-sha256", required=True)
    verify_lock.add_argument("--control-replay-store", required=True)
    verify_lock.add_argument("--now-ns", type=int, required=True)
    verify_lock.add_argument("--output", required=True)

    gpu_hours = commands.add_parser("create-gpu-hour-envelope", allow_abbrev=False)
    gpu_hours.add_argument("--output", required=True)

    reduce_gpu_hours = commands.add_parser(
        "reduce-stage-gpu-hour-envelope", allow_abbrev=False
    )
    reduce_gpu_hours.add_argument("--signed-pilot-receipt", required=True)
    reduce_gpu_hours.add_argument("--control-attestation", required=True)
    reduce_gpu_hours.add_argument("--inventory-sha256", required=True)
    reduce_gpu_hours.add_argument("--control-replay-store", required=True)
    reduce_gpu_hours.add_argument("--protocol-lock-sha256", required=True)
    reduce_gpu_hours.add_argument("--materialization-receipt-sha256", required=True)
    reduce_gpu_hours.add_argument("--schedule-sha256", required=True)
    reduce_gpu_hours.add_argument("--now-ns", type=int, required=True)
    reduce_gpu_hours.add_argument("--output", required=True)

    preflight_gpu_hours = commands.add_parser(
        "materialize-preflight-gpu-hour-envelope",
        allow_abbrev=False,
    )
    preflight_gpu_hours.add_argument("--dispatch-receipt", required=True)
    preflight_gpu_hours.add_argument("--remote-raw-receipt", required=True)
    preflight_gpu_hours.add_argument("--source-authority", required=True)
    preflight_gpu_hours.add_argument("--activation", required=True)
    preflight_gpu_hours.add_argument("--coverage", required=True)
    preflight_gpu_hours.add_argument("--stage-coverage", required=True)
    preflight_gpu_hours.add_argument(
        "--interference-lifecycle-proof",
        action="append",
        required=True,
        metavar="CELL_ID=PATH",
    )
    preflight_gpu_hours.add_argument(
        "--formal-runtime-authority-manifest", required=True
    )
    preflight_gpu_hours.add_argument("--source-output", required=True)
    preflight_gpu_hours.add_argument("--now-ns", type=int, required=True)
    preflight_gpu_hours.add_argument("--output", required=True)

    prospective_gpu_hours = commands.add_parser(
        "materialize-prospective-stage-gpu-hours", allow_abbrev=False
    )
    prospective_gpu_hours.add_argument(
        "--stage", choices=("E3b", "E5", "E6", "E0"), required=True
    )
    prospective_gpu_hours.add_argument("--registry-verification-receipt", required=True)
    prospective_gpu_hours.add_argument("--pilot-materialization", required=True)
    prospective_gpu_hours.add_argument("--pilot-envelope", required=True)
    prospective_gpu_hours.add_argument("--pilot-source-manifest", required=True)
    prospective_gpu_hours.add_argument(
        "--formal-runtime-authority-manifest", required=True
    )
    prospective_gpu_hours.add_argument("--inventory", required=True)
    prospective_gpu_hours.add_argument("--one-shot-source-manifest")
    prospective_gpu_hours.add_argument("--source-output", required=True)
    prospective_gpu_hours.add_argument("--now-ns", type=int, required=True)
    prospective_gpu_hours.add_argument("--output", required=True)

    staged_prospective_gpu_hours = commands.add_parser(
        "materialize-staged-prospective-gpu-hours", allow_abbrev=False
    )
    staged_prospective_gpu_hours.add_argument(
        "--stage",
        choices=("E3a", "TTS-Cal", "E1", "E2", "E4", "E1a"),
        required=True,
    )
    staged_prospective_gpu_hours.add_argument("--materialization-sha256", required=True)
    staged_prospective_gpu_hours.add_argument(
        "--registry-verification-receipt", required=True
    )
    staged_prospective_gpu_hours.add_argument("--completed-source-manifest")
    staged_prospective_gpu_hours.add_argument(
        "--formal-runtime-authority-manifest", required=True
    )
    staged_prospective_gpu_hours.add_argument("--inventory", required=True)
    staged_prospective_gpu_hours.add_argument("--source-output", required=True)
    staged_prospective_gpu_hours.add_argument("--now-ns", type=int, required=True)
    staged_prospective_gpu_hours.add_argument("--output", required=True)

    publish_gpu_hour_proof = commands.add_parser(
        "publish-formal-stage-gpu-hour-envelope-proof", allow_abbrev=False
    )
    publish_gpu_hour_proof.add_argument("--protocol-lock", required=True)
    publish_gpu_hour_proof.add_argument(
        "--formal-runtime-authority-manifest", required=True
    )
    publish_gpu_hour_proof.add_argument("--registry-layer", required=True)
    publish_gpu_hour_proof.add_argument("--inventory", required=True)
    publish_gpu_hour_proof.add_argument("--final-materialization", required=True)
    publish_gpu_hour_proof.add_argument("--pilot-materialization")
    publish_gpu_hour_proof.add_argument("--gpu-hour-source-manifest", required=True)
    publish_gpu_hour_proof.add_argument("--envelope", required=True)
    publish_gpu_hour_proof.add_argument("--preflight-coverage-proof")
    publish_gpu_hour_proof.add_argument("--now-ns", type=int, required=True)
    publish_gpu_hour_proof.add_argument("--output", required=True)

    publish_initial_materialization_proof = commands.add_parser(
        "publish-formal-initial-stage-materialization-proof",
        allow_abbrev=False,
    )
    publish_initial_materialization_proof.add_argument(
        "--phase",
        choices=("preflight", "e3a", "tts_calibration", "e1"),
        required=True,
    )
    publish_initial_materialization_proof.add_argument(
        "--registry-layer", required=True
    )
    publish_initial_materialization_proof.add_argument("--tts-calibration-authority")
    publish_initial_materialization_proof.add_argument(
        "--now-ns", type=int, required=True
    )
    publish_initial_materialization_proof.add_argument("--output", required=True)

    publish_downstream_materialization_proof = commands.add_parser(
        "publish-formal-downstream-materialization-proof",
        allow_abbrev=False,
    )
    publish_downstream_materialization_proof.add_argument(
        "--phase",
        choices=FORMAL_DOWNSTREAM_MATERIALIZATION_ORDER,
        required=True,
    )
    publish_downstream_materialization_proof.add_argument(
        "--registry-layer", required=True
    )
    publish_downstream_materialization_proof.add_argument(
        "--immediate-predecessor", required=True
    )
    publish_downstream_materialization_proof.add_argument(
        "--now-ns", type=int, required=True
    )
    publish_downstream_materialization_proof.add_argument("--output", required=True)

    publish_downstream_pilot_precoverage = commands.add_parser(
        "publish-formal-downstream-pilot-precoverage",
        allow_abbrev=False,
    )
    publish_downstream_pilot_precoverage.add_argument(
        "--phase",
        choices=("e3b_pilot", "e5_pilot", "e6_pilot", "e0_pilot"),
        required=True,
    )
    publish_downstream_pilot_precoverage.add_argument(
        "--materialization-proof", required=True
    )
    publish_downstream_pilot_precoverage.add_argument(
        "--signed-materialization", required=True
    )
    publish_downstream_pilot_precoverage.add_argument(
        "--now-ns", type=int, required=True
    )
    publish_downstream_pilot_precoverage.add_argument("--output", required=True)

    publish_portable_stage_coverage = commands.add_parser(
        "publish-formal-portable-stage-coverage-proof",
        allow_abbrev=False,
    )
    publish_portable_stage_coverage.add_argument("--coverage-proof", required=True)
    publish_portable_stage_coverage.add_argument("--registry-layer", required=True)
    publish_portable_stage_coverage.add_argument("--prior-prefix")
    publish_portable_stage_coverage.add_argument("--e1-recipe-anchor-authority")
    publish_portable_stage_coverage.add_argument("--downstream-pilot-precoverage")
    publish_portable_stage_coverage.add_argument("--now-ns", type=int, required=True)
    publish_portable_stage_coverage.add_argument("--output", required=True)

    publish_downstream_reduction_proof = commands.add_parser(
        "publish-formal-downstream-reduction-proof",
        allow_abbrev=False,
    )
    publish_downstream_reduction_proof.add_argument(
        "--phase",
        choices=FORMAL_DOWNSTREAM_MATERIALIZATION_ORDER,
        required=True,
    )
    publish_downstream_reduction_proof.add_argument(
        "--materialization-proof", required=True
    )
    publish_downstream_reduction_proof.add_argument(
        "--portable-coverage-proof", required=True
    )
    publish_downstream_reduction_proof.add_argument("--now-ns", type=int, required=True)
    publish_downstream_reduction_proof.add_argument("--output", required=True)

    publish_downstream_completed_prefix = commands.add_parser(
        "publish-formal-downstream-completed-prefix",
        allow_abbrev=False,
    )
    publish_downstream_completed_prefix.add_argument(
        "--phase",
        choices=FORMAL_DOWNSTREAM_MATERIALIZATION_ORDER,
        required=True,
    )
    publish_downstream_completed_prefix.add_argument("--reduction-proof", required=True)
    publish_downstream_completed_prefix.add_argument("--signed-result", required=True)
    publish_downstream_completed_prefix.add_argument(
        "--now-ns", type=int, required=True
    )
    publish_downstream_completed_prefix.add_argument("--output", required=True)

    publish_e3a_selection_proof = commands.add_parser(
        "publish-formal-e3a-staged-selection-proof",
        allow_abbrev=False,
    )
    publish_e3a_selection_proof.add_argument("--coverage-proof", required=True)
    publish_e3a_selection_proof.add_argument("--registry-layer", required=True)
    publish_e3a_selection_proof.add_argument("--now-ns", type=int, required=True)
    publish_e3a_selection_proof.add_argument("--output", required=True)

    reserve_formal_gpu_hours = commands.add_parser(
        "reserve-formal-stage-gpu-hours", allow_abbrev=False
    )
    reserve_formal_gpu_hours.add_argument(
        "--registry-verification-receipt", required=True
    )
    reserve_formal_gpu_hours.add_argument("--signed-envelope", required=True)
    reserve_formal_gpu_hours.add_argument("--source-manifest", required=True)
    reserve_formal_gpu_hours.add_argument("--prospective-pilot-materialization")
    reserve_formal_gpu_hours.add_argument(
        "--formal-runtime-authority-manifest", required=True
    )
    reserve_formal_gpu_hours.add_argument("--inventory", required=True)
    reserve_formal_gpu_hours.add_argument("--control-attestation", required=True)
    reserve_formal_gpu_hours.add_argument("--control-replay-store", required=True)
    reserve_formal_gpu_hours.add_argument("--now-ns", type=int, required=True)
    reserve_formal_gpu_hours.add_argument("--output", required=True)

    aggregate_formal_gpu_hours = commands.add_parser(
        "aggregate-formal-study-gpu-hours", allow_abbrev=False
    )
    aggregate_formal_gpu_hours.add_argument(
        "--registry-verification-receipt", required=True
    )
    aggregate_formal_gpu_hours.add_argument(
        "--stage-receipt", action="append", required=True
    )
    aggregate_formal_gpu_hours.add_argument("--now-ns", type=int, required=True)
    aggregate_formal_gpu_hours.add_argument("--allow-partial", action="store_true")
    aggregate_formal_gpu_hours.add_argument("--output", required=True)

    create_materialization = commands.add_parser(
        "create-stage-materialization-receipt", allow_abbrev=False
    )
    create_materialization.add_argument("--request", required=True)
    create_materialization.add_argument("--control-attestation")
    create_materialization.add_argument("--inventory-sha256")
    create_materialization.add_argument("--control-replay-store")
    create_materialization.add_argument("--now-ns", type=int)
    create_materialization.add_argument("--output", required=True)

    verify_materialization = commands.add_parser(
        "verify-signed-stage-materialization", allow_abbrev=False
    )
    verify_materialization.add_argument("--signed-receipt", required=True)
    verify_materialization.add_argument("--control-attestation", required=True)
    verify_materialization.add_argument("--inventory-sha256", required=True)
    verify_materialization.add_argument("--control-replay-store", required=True)
    verify_materialization.add_argument("--now-ns", type=int, required=True)
    verify_materialization.add_argument("--output", required=True)

    create_coverage = commands.add_parser(
        "create-stage-coverage-receipt", allow_abbrev=False
    )
    create_coverage.add_argument("--materialization", required=True)
    create_coverage.add_argument("--dispositions", required=True)
    create_coverage.add_argument(
        "--tts-l0-candidate-state-coverage",
        action="append",
        default=[],
    )
    create_coverage.add_argument("--output", required=True)

    verify_coverage = commands.add_parser(
        "verify-signed-stage-coverage", allow_abbrev=False
    )
    verify_coverage.add_argument("--signed-receipt", required=True)
    verify_coverage.add_argument("--materialization", required=True)
    verify_coverage.add_argument("--control-attestation", required=True)
    verify_coverage.add_argument("--inventory-sha256", required=True)
    verify_coverage.add_argument("--control-replay-store", required=True)
    verify_coverage.add_argument("--now-ns", type=int, required=True)
    verify_coverage.add_argument("--output", required=True)

    publish_runtime_manifest = commands.add_parser(
        "publish-formal-runtime-authority-manifest", allow_abbrev=False
    )
    publish_runtime_manifest.add_argument("--repository-root", required=True)
    publish_runtime_manifest.add_argument("--output", required=True)

    publish_formal_rebuild = commands.add_parser(
        "publish-formal-rebuild-artifact", allow_abbrev=False
    )
    publish_formal_rebuild.add_argument(
        "--artifact-kind",
        choices=(
            "stage-source",
            "serving-shard",
            "failure-shard",
            "e6-recursive-dag",
            "e0-aggregate",
            "e0-final-result",
        ),
        required=True,
    )
    publish_formal_rebuild.add_argument("--input", required=True)
    publish_formal_rebuild.add_argument("--output", required=True)

    publish_tts_reduction = commands.add_parser(
        "publish-formal-tts-calibration-reduction-proof", allow_abbrev=False
    )
    publish_tts_reduction.add_argument("--portable-coverage-proof", required=True)
    publish_tts_reduction.add_argument("--hardware-envelope", required=True)
    publish_tts_reduction.add_argument("--replay-reservation", required=True)
    publish_tts_reduction.add_argument("--runtime-sha256", required=True)
    publish_tts_reduction.add_argument("--split-sha256", required=True)
    publish_tts_reduction.add_argument("--now-ns", type=int, required=True)
    publish_tts_reduction.add_argument("--output", required=True)

    publish_formal_stage_shard = commands.add_parser(
        "publish-formal-stage-execution-shard", allow_abbrev=False
    )
    publish_formal_stage_shard.add_argument(
        "--phase",
        choices=(
            "e1_selection",
            "e2_round0",
            "e2_round1",
            "e2_round2",
            "e2_round3",
            "e4_screen",
            "e4_local",
        ),
        required=True,
    )
    publish_formal_stage_shard.add_argument("--materialization", required=True)
    publish_formal_stage_shard.add_argument("--stage-source-rebuild")
    publish_formal_stage_shard.add_argument(
        "--execution-rebuild-input", action="append", required=True
    )
    publish_formal_stage_shard.add_argument("--output", required=True)

    publish_formal_stage_prefix = commands.add_parser(
        "publish-formal-stage-prefix", allow_abbrev=False
    )
    publish_formal_stage_prefix.add_argument(
        "--phase", choices=FORMAL_STAGE_PREFIX_ORDER, required=True
    )
    publish_formal_stage_prefix.add_argument(
        "--registry-verification-receipt", required=True
    )
    publish_formal_stage_prefix.add_argument("--formal-runtime-authority-manifest")
    publish_formal_stage_prefix.add_argument("--inventory")
    publish_formal_stage_prefix.add_argument("--materialization")
    publish_formal_stage_prefix.add_argument("--coverage")
    publish_formal_stage_prefix.add_argument("--coverage-proof", required=True)
    publish_formal_stage_prefix.add_argument("--stage-source-rebuild")
    publish_formal_stage_prefix.add_argument(
        "--execution-rebuild-shard", action="append", default=[]
    )
    publish_formal_stage_prefix.add_argument("--e1-recipe-anchor-authority")
    publish_formal_stage_prefix.add_argument("--prior-prefix")
    publish_formal_stage_prefix.add_argument("--now-ns", type=int, required=True)
    publish_formal_stage_prefix.add_argument("--output", required=True)

    publish_scientific_source = commands.add_parser(
        "publish-scientific-source-validation", allow_abbrev=False
    )
    publish_scientific_source.add_argument(
        "--artifact-type",
        choices=tuple(sorted(PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES)),
        required=True,
    )
    publish_scientific_source.add_argument(
        "--proof-bundle",
        dest="proof_bundle",
        required=True,
    )
    publish_scientific_source.add_argument("--proof-entry")
    publish_scientific_source.add_argument("--now-ns", type=int, required=True)
    publish_scientific_source.add_argument("--output", required=True)

    formal_stage_operation = commands.add_parser(
        "formal-stage-operation", allow_abbrev=False
    )
    formal_stage_operation.add_argument(
        "--operation", choices=("materialize", "reduce", "sign"), required=True
    )
    formal_stage_operation.add_argument(
        "--stage",
        choices=("E3a", "TTS-Cal", "E1", "E2", "E4", "E3b", "E1a", "E5", "E6", "E0"),
        required=True,
    )
    formal_stage_operation.add_argument("--phase", required=True)
    formal_stage_operation.add_argument(
        "--registry-verification-receipt", required=True
    )
    formal_stage_operation.add_argument(
        "--stage-prefix-artifact",
        help=(
            "current-only path-bound E1/E2/E4 proof prefix; mandatory for "
            "sequential materialize/reduce/sign"
        ),
    )
    formal_stage_operation.add_argument("--e0-authority-bundle")
    formal_stage_operation.add_argument(
        "--e0-materialization",
        help="exact offline-signed E0 final materialization wrapper",
    )
    formal_stage_operation.add_argument(
        "--result-rebuild-artifact",
        help="typed proof-rebuild artifact consumed by a stage result reducer",
    )
    formal_stage_operation.add_argument(
        "--signed-stage-result",
        help="offline-signed result wrapper to deep-verify against the reducer",
    )
    formal_stage_operation.add_argument(
        "--signed-e0-fdr-result",
        help=(
            "offline-signed proof-derived E0 breadth FDR wrapper; required with "
            "E0 final sign"
        ),
    )
    formal_stage_operation.add_argument(
        "--tts-calibration-authority",
        help="typed TTS calibration authority required to materialize TTS-Cal",
    )
    formal_stage_operation.add_argument("--now-ns", type=int, required=True)
    formal_stage_operation.add_argument("--output", required=True)

    assemble_formal = commands.add_parser(
        "assemble-formal-registry-manifest", allow_abbrev=False
    )
    assemble_formal.add_argument("--signed-protocol-lock", required=True)
    assemble_formal.add_argument(
        "--signed-materialization", action="append", default=[]
    )
    assemble_formal.add_argument("--signed-coverage", action="append", default=[])
    assemble_formal.add_argument(
        "--tts-calibration-authority", action="append", default=[]
    )
    assemble_formal.add_argument(
        "--signed-tts-calibration-seal", action="append", default=[]
    )
    assemble_formal.add_argument(
        "--signed-e3b-power-prefix", action="append", default=[]
    )
    assemble_formal.add_argument(
        "--signed-e5-power-and-anchor-prefix", action="append", default=[]
    )
    assemble_formal.add_argument(
        "--signed-e6-power-prefix", action="append", default=[]
    )
    assemble_formal.add_argument(
        "--control-attestation", action="append", required=True
    )
    assemble_formal.add_argument(
        "--candidate-state-replay-proof-artifact",
        action="append",
        default=[],
        help=(
            "repeat for every durable, externally controlled TTS/L0 replay proof "
            "referenced by stage coverage"
        ),
    )
    assemble_formal.add_argument("--inventory-sha256", required=True)
    assemble_formal.add_argument("--control-replay-store", required=True)
    assemble_formal.add_argument("--now-ns", type=int, required=True)
    assemble_formal.add_argument("--output", required=True)

    reserve_formal_registry = commands.add_parser(
        "reserve-formal-registry-verification",
        allow_abbrev=False,
    )
    reserve_formal_registry.add_argument("--signed-protocol-lock", required=True)
    reserve_formal_registry.add_argument("--control-attestation", required=True)
    reserve_formal_registry.add_argument("--inventory-sha256", required=True)
    reserve_formal_registry.add_argument("--control-replay-store", required=True)
    reserve_formal_registry.add_argument("--now-ns", type=int, required=True)
    reserve_formal_registry.add_argument("--output", required=True)

    extend_formal_registry = commands.add_parser(
        "extend-formal-registry-verification",
        allow_abbrev=False,
    )
    extend_formal_registry.add_argument("--prior-receipt", required=True)
    extend_formal_registry.add_argument(
        "--signed-materialization", action="append", default=[]
    )
    extend_formal_registry.add_argument(
        "--signed-coverage", action="append", default=[]
    )
    extend_formal_registry.add_argument(
        "--control-attestation", action="append", required=True
    )
    extend_formal_registry.add_argument(
        "--tts-calibration-authority", action="append", default=[]
    )
    extend_formal_registry.add_argument(
        "--signed-tts-calibration-seal", action="append", default=[]
    )
    extend_formal_registry.add_argument(
        "--tts-calibration-reduction-proof",
        action="append",
        default=[],
        help=(
            "path-bound 288-cell raw reduction proof; required one-for-one "
            "with every appended signed TTS calibration seal"
        ),
    )
    extend_formal_registry.add_argument(
        "--e3a-staged-selection-proof",
        action="append",
        default=[],
        help=(
            "path-bound exact 360-row reducer proof; required one-for-one "
            "with every appended signed E3a staged selection"
        ),
    )
    extend_formal_registry.add_argument(
        "--signed-e3a-staged-selection", action="append", default=[]
    )
    extend_formal_registry.add_argument(
        "--signed-e1-survivor-selection", action="append", default=[]
    )
    extend_formal_registry.add_argument(
        "--signed-e2-staged-selection", action="append", default=[]
    )
    extend_formal_registry.add_argument(
        "--signed-e4-stage-selection", action="append", default=[]
    )
    extend_formal_registry.add_argument(
        "--formal-stage-prefix-artifact",
        action="append",
        default=[],
        help=(
            "mandatory path-bound proof prefix for every E1/E2/E4 "
            "coverage/selection append"
        ),
    )
    extend_formal_registry.add_argument(
        "--signed-e3b-power-prefix", action="append", default=[]
    )
    extend_formal_registry.add_argument(
        "--signed-e5-power-and-anchor-prefix", action="append", default=[]
    )
    extend_formal_registry.add_argument(
        "--signed-e6-power-prefix", action="append", default=[]
    )
    extend_formal_registry.add_argument(
        "--e0-authority-bundle",
        action="append",
        default=[],
        help=(
            "complete path-bound E0 typed authority bundle; digest-only inputs "
            "are never accepted"
        ),
    )
    extend_formal_registry.add_argument(
        "--candidate-state-replay-proof-artifact",
        action="append",
        default=[],
    )
    extend_formal_registry.add_argument("--control-replay-store", required=True)
    extend_formal_registry.add_argument("--now-ns", type=int, required=True)
    extend_formal_registry.add_argument("--output", required=True)

    verify_formal_registry = commands.add_parser(
        "verify-formal-registry-verification",
        allow_abbrev=False,
    )
    verify_formal_registry.add_argument("--receipt", required=True)
    verify_formal_registry.add_argument("--now-ns", type=int, required=True)
    verify_formal_registry.add_argument("--output", required=True)

    authorize_preflight = commands.add_parser(
        "authorize-formal-preflight-dispatch",
        allow_abbrev=False,
    )
    authorize_preflight.add_argument("--registry-verification-receipt", required=True)
    authorize_preflight.add_argument("--signed-materialization", required=True)
    authorize_preflight.add_argument("--capacity-control", required=True)
    authorize_preflight.add_argument("--dispatch-control", required=True)
    authorize_preflight.add_argument("--inventory", required=True)
    authorize_preflight.add_argument("--stage-activation", required=True)
    authorize_preflight.add_argument("--budget-plan", required=True)
    authorize_preflight.add_argument("--budget-authority", required=True)
    authorize_preflight.add_argument("--dispatch-plan", required=True)
    authorize_preflight.add_argument("--capacity-schedule", required=True)
    authorize_preflight.add_argument("--capacity-gate", required=True)
    authorize_preflight.add_argument("--control-replay-store", required=True)
    authorize_preflight.add_argument("--now-ns", type=int, required=True)
    authorize_preflight.add_argument("--output", required=True)

    materialize_preflight_caps = commands.add_parser(
        "materialize-formal-preflight-launch-cap-schedule",
        allow_abbrev=False,
    )
    materialize_preflight_caps.add_argument("--dispatch-receipt", required=True)
    materialize_preflight_caps.add_argument("--now-ns", type=int, required=True)
    materialize_preflight_caps.add_argument("--output", required=True)

    execute_preflight_raw = commands.add_parser(
        "execute-formal-preflight-raw",
        allow_abbrev=False,
    )
    execute_preflight_raw.add_argument("--dispatch-receipt", required=True)
    execute_preflight_raw.add_argument("--launch-cap-schedule", required=True)
    execute_preflight_raw.add_argument("--compile-assignment-plan", required=True)
    execute_preflight_raw.add_argument(
        "--prepared-content-verification-receipt", required=True
    )
    execute_preflight_raw.add_argument("--compile-control", required=True)
    execute_preflight_raw.add_argument("--exactness-assignment", required=True)
    execute_preflight_raw.add_argument("--exactness-control", required=True)
    execute_preflight_raw.add_argument(
        "--interference-execution-manifest", required=True
    )
    execute_preflight_raw.add_argument("--nvidia-smi", required=True)
    execute_preflight_raw.add_argument("--control-replay-store", required=True)
    execute_preflight_raw.add_argument("--evidence-root", required=True)
    execute_preflight_raw.add_argument("--now-ns", type=int, required=True)
    execute_preflight_raw.add_argument("--output", required=True)

    qualify_preflight_exactness = commands.add_parser(
        "qualify-formal-preflight-exactness",
        allow_abbrev=False,
    )
    qualify_preflight_exactness.add_argument("--dispatch-receipt", required=True)
    qualify_preflight_exactness.add_argument("--remote-raw-receipt", required=True)
    qualify_preflight_exactness.add_argument("--rank-aggregate-control", required=True)
    qualify_preflight_exactness.add_argument("--control-replay-store", required=True)
    qualify_preflight_exactness.add_argument("--now-ns", type=int, required=True)
    qualify_preflight_exactness.add_argument("--proof-output", required=True)
    qualify_preflight_exactness.add_argument("--qualified-output", required=True)

    qualify_preflight_interference = commands.add_parser(
        "qualify-formal-preflight-interference",
        allow_abbrev=False,
    )
    qualify_preflight_interference.add_argument("--dispatch-receipt", required=True)
    qualify_preflight_interference.add_argument("--remote-raw-receipt", required=True)
    qualify_preflight_interference.add_argument(
        "--native-result-proof",
        action="append",
        required=True,
        metavar="CELL_ID=PATH",
    )
    qualify_preflight_interference.add_argument(
        "--native-itl-proof",
        action="append",
        required=True,
        metavar="CELL_ID=PATH",
    )
    qualify_preflight_interference.add_argument("--aggregate-control", required=True)
    qualify_preflight_interference.add_argument("--control-replay-store", required=True)
    qualify_preflight_interference.add_argument("--now-ns", type=int, required=True)
    qualify_preflight_interference.add_argument("--output", required=True)

    finalize_preflight = commands.add_parser(
        "finalize-formal-preflight-evidence",
        allow_abbrev=False,
    )
    finalize_preflight.add_argument("--dispatch-receipt", required=True)
    finalize_preflight.add_argument("--remote-raw-receipt", required=True)
    finalize_preflight.add_argument("--exactness-result", required=True)
    finalize_preflight.add_argument("--interference-proof", required=True)
    finalize_preflight.add_argument(
        "--qualification-proof",
        action="append",
        required=True,
        metavar="SUITE=RESULT_POINTER,PROOF_ARTIFACT",
    )
    finalize_preflight.add_argument("--candidate-state-coverage", required=True)
    finalize_preflight.add_argument(
        "--candidate-replay-proof",
        action="append",
        required=True,
        metavar="PATH",
    )
    finalize_preflight.add_argument("--now-ns", type=int, required=True)
    finalize_preflight.add_argument("--source-output", required=True)
    finalize_preflight.add_argument("--activation-output", required=True)
    finalize_preflight.add_argument("--coverage-output", required=True)
    finalize_preflight.add_argument("--stage-coverage-output", required=True)

    collect_inventory = commands.add_parser("collect-gpu-inventory")
    collect_inventory.add_argument("--challenge-nonce-sha256", required=True)
    collect_inventory.add_argument("--receipt-output", required=True)
    collect_inventory.add_argument("--output", required=True)

    assemble_fleet = commands.add_parser(
        "assemble-gpu-fleet-inventory",
        allow_abbrev=False,
    )
    assemble_fleet.add_argument(
        "--inventory",
        action="append",
        required=True,
        metavar="PATH",
        help=("repeat once per host with a content-bound single-host GPU inventory"),
    )
    assemble_fleet.add_argument(
        "--interference-envelope",
        action="append",
        required=True,
        metavar="PATH",
        help="repeat in the same host order as --inventory",
    )
    assemble_fleet.add_argument("--output", required=True)

    build_interference = commands.add_parser("build-interference-envelope")
    build_interference.add_argument("--inventory", required=True)
    build_interference.add_argument("--receipt-output", required=True)
    build_interference.add_argument("--output", required=True)

    bootstrap_interference = commands.add_parser(
        "materialize-interference-calibration-bootstrap"
    )
    bootstrap_interference.add_argument("--registry", required=True)
    bootstrap_interference.add_argument("--activation-manifest", required=True)
    bootstrap_interference.add_argument("--inventory", required=True)
    bootstrap_interference.add_argument("--receipt-output", required=True)
    bootstrap_interference.add_argument("--output", required=True)

    reduce_interference = commands.add_parser("reduce-interference-calibration")
    reduce_interference.add_argument("--authority", required=True)
    reduce_interference.add_argument("--envelope-output", required=True)
    reduce_interference.add_argument("--output", required=True)

    materialize_capacity = commands.add_parser(
        "materialize-stage-capacity-gate",
        allow_abbrev=False,
    )
    materialize_capacity.add_argument("--registry", required=True)
    materialize_capacity.add_argument("--capacity-source-manifest", required=True)
    materialize_capacity.add_argument("--stage-schedule", required=True)
    materialize_capacity.add_argument("--now-ns", type=int, required=True)
    materialize_capacity.add_argument("--output", required=True)

    seal_industrial = commands.add_parser("seal-industrial-stage")
    seal_industrial.add_argument("--registry", required=True)
    seal_industrial.add_argument("--experiment", required=True)
    seal_industrial.add_argument("--runtime-artifact", required=True)
    seal_industrial.add_argument("--split-artifact", required=True)
    seal_industrial.add_argument("--completed-cells", required=True)
    seal_industrial.add_argument("--inventory", required=True)
    seal_industrial.add_argument("--e2-final-stage-manifest")
    seal_industrial.add_argument("--interference-calibration-authority")
    seal_industrial.add_argument("--preflight-coverage-receipt")
    seal_industrial.add_argument("--preflight-coverage-attestation")
    seal_industrial.add_argument("--stage-capacity-gate")
    seal_industrial.add_argument("--stage-capacity-attestation")
    seal_industrial.add_argument("--control-replay-store")
    seal_industrial.add_argument("--preflight-control-binding-output")
    seal_industrial.add_argument("--activation-plan")
    seal_industrial.add_argument("--family-activation", action="append", default=[])
    seal_industrial.add_argument("--family-power-plan", action="append", default=[])
    seal_industrial.add_argument("--dependency-receipt", action="append", default=[])
    seal_industrial.add_argument("--locked-output", action="append", required=True)
    seal_industrial.add_argument("--output", required=True)

    plan_industrial = commands.add_parser("plan-industrial-dispatch")
    plan_industrial.add_argument("--registry", required=True)
    plan_industrial.add_argument("--inventory", required=True)
    plan_industrial.add_argument("--interference-envelope", required=True)
    plan_industrial.add_argument("--budget-plan", required=True)
    plan_industrial.add_argument("--budget-policy", required=True)
    plan_industrial.add_argument("--budget-load-binding", action="append", default=[])
    plan_industrial.add_argument("--capacity-envelope", required=True)
    plan_industrial.add_argument("--capacity-manifest")
    plan_industrial.add_argument("--capacity-verification-receipt")
    plan_industrial.add_argument("--receipt", action="append", default=[])
    plan_industrial.add_argument("--completed-cells")
    plan_industrial.add_argument("--completed-e2-stage-manifest")
    plan_industrial.add_argument("--activation-plan", action="append", default=[])
    plan_industrial.add_argument("--family-activation", action="append", default=[])
    plan_industrial.add_argument("--family-power-plan", action="append", default=[])
    plan_industrial.add_argument("--output", required=True)

    materialize_dispatch = commands.add_parser(
        "materialize-dispatch-execution-bundles",
        allow_abbrev=False,
    )
    materialize_dispatch.add_argument("--request", required=True)
    materialize_dispatch.add_argument("--output-directory", required=True)

    execute_wave = commands.add_parser("execute-dispatch-wave", allow_abbrev=False)
    execute_wave.add_argument("--host-request-stdin", action="store_true")
    execute_wave.add_argument("--materialization-manifest")
    execute_wave.add_argument("--wave-index", type=int)
    execute_wave.add_argument("--resume-receipt")
    execute_wave.add_argument("--receipt-output")

    materialize_budget = commands.add_parser("materialize-industrial-budgets")
    materialize_budget.add_argument("--registry", required=True)
    materialize_budget.add_argument("--activation-plan", action="append", default=[])
    materialize_budget.add_argument("--family-activation", action="append", default=[])
    materialize_budget.add_argument("--family-power-plan", action="append", default=[])
    materialize_budget.add_argument("--budget-policy", required=True)
    materialize_budget.add_argument(
        "--budget-load-binding", action="append", default=[]
    )
    materialize_budget.add_argument("--capacity-envelope", required=True)
    materialize_budget.add_argument("--capacity-manifest")
    materialize_budget.add_argument("--capacity-verification-receipt")
    materialize_budget.add_argument("--inventory", required=True)
    materialize_budget.add_argument("--output", required=True)

    bind_budget_authority = commands.add_parser(
        "bind-industrial-budget-authority", allow_abbrev=False
    )
    bind_budget_authority.add_argument("--activation-manifest", required=True)
    bind_budget_authority.add_argument("--budget-policy", required=True)
    bind_budget_authority.add_argument(
        "--budget-load-binding", action="append", default=[]
    )
    bind_budget_authority.add_argument("--capacity-envelope", required=True)
    bind_budget_authority.add_argument("--capacity-manifest", required=True)
    bind_budget_authority.add_argument("--capacity-verification-receipt", required=True)
    bind_budget_authority.add_argument("--budget-plan", required=True)
    bind_budget_authority.add_argument("--output", required=True)

    materialize_stage = commands.add_parser("materialize-stage-activation")
    materialize_stage.add_argument("--manifest", required=True)
    materialize_stage.add_argument("--output", required=True)

    pointer_preflight = commands.add_parser(
        "materialize-preflight-pointer-coverage", allow_abbrev=False
    )
    pointer_preflight.add_argument("--registry", required=True)
    pointer_preflight.add_argument("--runtime-artifact", required=True)
    pointer_preflight.add_argument("--split-artifact", required=True)
    pointer_preflight.add_argument("--compile-result", required=True)
    pointer_preflight.add_argument("--exactness-result", required=True)
    pointer_preflight.add_argument("--interference-authority", required=True)
    pointer_preflight.add_argument("--source-output", required=True)
    pointer_preflight.add_argument("--activation-output", required=True)
    pointer_preflight.add_argument("--coverage-output", required=True)

    estimate_budget = commands.add_parser("estimate-industrial-budget")
    estimate_budget.add_argument("--registry", required=True)
    estimate_budget.add_argument("--activation-plan", action="append", default=[])
    estimate_budget.add_argument("--family-activation", action="append", default=[])
    estimate_budget.add_argument("--family-power-plan", action="append", default=[])
    estimate_budget.add_argument("--inventory", required=True)
    estimate_budget.add_argument("--interference-envelope", required=True)
    estimate_budget.add_argument("--budget-plan", required=True)
    estimate_budget.add_argument("--budget-policy", required=True)
    estimate_budget.add_argument("--budget-load-binding", action="append", default=[])
    estimate_budget.add_argument("--capacity-envelope", required=True)
    estimate_budget.add_argument("--capacity-manifest")
    estimate_budget.add_argument("--capacity-verification-receipt")
    estimate_budget.add_argument("--output", required=True)

    reduce_e1 = commands.add_parser("reduce-e1-activation")
    reduce_e1.add_argument("--registry", required=True)
    reduce_e1.add_argument("--e3a-receipt", required=True)
    reduce_e1.add_argument("--selection", required=True)
    reduce_e1.add_argument("--output", required=True)

    reduce_e2 = commands.add_parser("reduce-e2-activation")
    reduce_e2.add_argument("--manifest", required=True)
    reduce_e2.add_argument("--output", required=True)

    halve_e2 = commands.add_parser("reduce-e2-successive-halving")
    halve_e2.add_argument("--manifest", required=True)
    halve_e2.add_argument("--output", required=True)

    family_pilots = commands.add_parser("materialize-confirmation-pilots")
    family_pilots.add_argument("--registry", required=True)
    family_pilots.add_argument("--family", required=True)
    family_pilots.add_argument("--output", required=True)

    family_power = commands.add_parser("reduce-confirmation-family-power")
    family_power.add_argument("--manifest", required=True)
    family_power.add_argument("--output", required=True)

    family_prefix = commands.add_parser("materialize-confirmation-prefix")
    family_prefix.add_argument("--power-manifest", required=True)
    family_prefix.add_argument("--output", required=True)

    validate_alias = commands.add_parser("validate-evidence-alias", allow_abbrev=False)
    validate_alias.add_argument("--manifest", required=True)
    validate_alias.add_argument("--registry", required=True)
    validate_alias.add_argument("--inventory", required=True)
    validate_alias.add_argument("--hardware-envelope", required=True)
    validate_alias.add_argument("--output", required=True)

    dependence = commands.add_parser(
        "build-evidence-dependence-map", allow_abbrev=False
    )
    dependence.add_argument("--direct-map", required=True)
    dependence.add_argument("--alias-reduction", action="append", default=[])
    dependence.add_argument("--output", required=True)

    analyze_industrial = commands.add_parser("analyze-industrial")
    analyze_industrial.add_argument("--manifest", required=True)
    analyze_industrial.add_argument("--output", required=True)

    analyze_e3b = commands.add_parser("analyze-e3b-long-context")
    analyze_e3b.add_argument("--manifest", required=True)
    analyze_e3b.add_argument("--output", required=True)

    build_online = commands.add_parser("build-onlinespec-study")
    build_online.add_argument("--output", required=True)

    verify_online_source = commands.add_parser("verify-onlinespec-source")
    verify_online_source.add_argument("--checkout", required=True)
    verify_online_source.add_argument("--audit", required=True)
    verify_online_source.add_argument("--output", required=True)

    list_online = commands.add_parser("list-onlinespec-candidates")
    list_online.add_argument("--output", required=True)

    select_online = commands.add_parser("select-onlinespec-config")
    select_online.add_argument("--measurements", required=True)
    select_online.add_argument("--manifest", required=True)
    select_online.add_argument("--model-lock", required=True)
    select_online.add_argument("--sampling-profile", required=True)
    select_online.add_argument("--core-selection", required=True)
    select_online.add_argument("--output", required=True)

    select_online_anchor = commands.add_parser("select-onlinespec-anchor-config")
    select_online_anchor.add_argument("--measurements", nargs=4, required=True)
    select_online_anchor.add_argument("--candidate-ids", nargs=3, required=True)
    select_online_anchor.add_argument("--manifest", required=True)
    select_online_anchor.add_argument("--model-lock", required=True)
    select_online_anchor.add_argument("--sampling-profile", required=True)
    select_online_anchor.add_argument("--core-selection", required=True)
    select_online_anchor.add_argument("--output", required=True)

    lock = commands.add_parser("lock-models")
    lock.add_argument("--output", required=True)
    lock.add_argument("models", nargs="+")

    prepare = commands.add_parser("prepare-models")
    prepare.add_argument("--lockfile", required=True)
    prepare.add_argument("--model-cache", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--offline", action="store_true")

    select = commands.add_parser("select-preliminary-speed-config")
    select.add_argument("--measurements", required=True)
    select.add_argument("--static-load-screen", required=True)
    select.add_argument("--manifest", required=True)
    select.add_argument("--model-lock", required=True)
    select.add_argument("--sampling-profile", required=True)
    select.add_argument("--output", required=True)

    select_anchor = commands.add_parser("select-preliminary-anchor-config")
    select_anchor.add_argument("--measurements", nargs=3, required=True)
    select_anchor.add_argument("--candidate-id", required=True)
    select_anchor.add_argument("--static-load-screen", required=True)
    select_anchor.add_argument("--manifest", required=True)
    select_anchor.add_argument("--model-lock", required=True)
    select_anchor.add_argument("--sampling-profile", required=True)
    select_anchor.add_argument("--output", required=True)

    render = commands.add_parser("render-preliminary-runtime")
    render.add_argument("--selection", required=True)
    render.add_argument("--model-lock", required=True)
    render.add_argument("--model-roots", required=True)
    render.add_argument("--sglang-checkout", required=True)
    render.add_argument("--sampling-profile", required=True)
    render.add_argument("--compile-cache-plan", required=True)
    render.add_argument("--adaptation-group-id", required=True)
    render.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render.add_argument("--mem-fraction-static", type=float, required=True)
    render.add_argument("--output-root", required=True)
    render.add_argument("--host", default="127.0.0.1")
    render.add_argument("--first-port", type=int, default=30000)

    render_online = commands.add_parser("render-onlinespec-runtime")
    render_online.add_argument("--selection", required=True)
    render_online.add_argument("--model-lock", required=True)
    render_online.add_argument("--model-roots", required=True)
    render_online.add_argument("--sglang-checkout", required=True)
    render_online.add_argument("--sampling-profile", required=True)
    render_online.add_argument("--compile-cache-plan", required=True)
    render_online.add_argument("--adaptation-group-id", required=True)
    render_online.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render_online.add_argument("--mem-fraction-static", type=float, required=True)
    render_online.add_argument("--output-root", required=True)
    render_online.add_argument("--host", default="127.0.0.1")
    render_online.add_argument("--first-port", type=int, default=30000)

    render_online_tune = commands.add_parser("render-onlinespec-tuning-runtime")
    render_online_tune.add_argument("--candidate-id", required=True)
    render_online_tune.add_argument("--concurrency", type=int, required=True)
    render_online_tune.add_argument("--model-lock", required=True)
    render_online_tune.add_argument("--model-roots", required=True)
    render_online_tune.add_argument("--sglang-checkout", required=True)
    render_online_tune.add_argument("--sampling-profile", required=True)
    render_online_tune.add_argument("--compile-cache-plan", required=True)
    render_online_tune.add_argument("--adaptation-group-id", required=True)
    render_online_tune.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render_online_tune.add_argument("--mem-fraction-static", type=float, required=True)
    render_online_tune.add_argument("--output-root", required=True)
    render_online_tune.add_argument("--host", default="127.0.0.1")
    render_online_tune.add_argument("--first-port", type=int, default=30000)

    render_static = commands.add_parser("render-preliminary-static-load-runtime")
    render_static.add_argument("--concurrency", type=int, required=True)
    render_static.add_argument("--model-lock", required=True)
    render_static.add_argument("--model-roots", required=True)
    render_static.add_argument("--sglang-checkout", required=True)
    render_static.add_argument("--sampling-profile", required=True)
    render_static.add_argument("--compile-cache-plan", required=True)
    render_static.add_argument("--mem-fraction-static", type=float, required=True)
    render_static.add_argument("--output-root", required=True)
    render_static.add_argument("--host", default="127.0.0.1")
    render_static.add_argument("--first-port", type=int, default=30000)

    render_target = commands.add_parser("render-preliminary-target-only-runtime")
    render_target.add_argument("--concurrency", type=int, required=True)
    render_target.add_argument("--model-lock", required=True)
    render_target.add_argument("--model-roots", required=True)
    render_target.add_argument("--sglang-checkout", required=True)
    render_target.add_argument("--sampling-profile", required=True)
    render_target.add_argument("--compile-cache-plan", required=True)
    render_target.add_argument("--gpu-uuid", required=True)
    render_target.add_argument("--mem-fraction-static", type=float, required=True)
    render_target.add_argument("--output-root", required=True)
    render_target.add_argument("--host", default="127.0.0.1")
    render_target.add_argument("--first-port", type=int, default=30000)

    render_tuning = commands.add_parser("render-preliminary-tuning-runtime")
    render_tuning.add_argument("--candidate-id", required=True)
    render_tuning.add_argument("--concurrency", type=int, required=True)
    render_tuning.add_argument("--model-lock", required=True)
    render_tuning.add_argument("--model-roots", required=True)
    render_tuning.add_argument("--sglang-checkout", required=True)
    render_tuning.add_argument("--sampling-profile", required=True)
    render_tuning.add_argument("--compile-cache-plan", required=True)
    render_tuning.add_argument("--adaptation-group-id", required=True)
    render_tuning.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render_tuning.add_argument("--mem-fraction-static", type=float, required=True)
    render_tuning.add_argument("--output-root", required=True)
    render_tuning.add_argument("--host", default="127.0.0.1")
    render_tuning.add_argument("--first-port", type=int, default=30000)

    replication = commands.add_parser("render-preliminary-replication-runtime")
    replication.add_argument("--phase", choices=("natural", "profile"), required=True)
    replication.add_argument("--selection", required=True)
    replication.add_argument("--model-lock", required=True)
    replication.add_argument("--model-roots", required=True)
    replication.add_argument("--sglang-checkout", required=True)
    replication.add_argument("--sampling-profile", required=True)
    replication.add_argument("--compile-cache-plan", required=True)
    replication.add_argument("--adaptation-group-id", required=True)
    replication.add_argument("--adaptation-reserve-mb", type=int, required=True)
    replication.add_argument("--mem-fraction-static", type=float, required=True)
    replication.add_argument("--output-root", required=True)
    replication.add_argument("--host", default="127.0.0.1")
    replication.add_argument("--first-port", type=int, default=30000)

    candidates = commands.add_parser("list-preliminary-tuning-candidates")
    candidates.add_argument("--output", required=True)

    controlled = commands.add_parser("run-preliminary-controlled-slice")
    controlled.add_argument("--manifest", required=True)
    controlled.add_argument("--model-lock", required=True)
    controlled.add_argument("--sampling-profile", required=True)
    controlled.add_argument("--config", required=True)
    controlled.add_argument("--url", required=True)
    controlled.add_argument("--phase", choices=("static-load", "tune"), required=True)
    controlled.add_argument("--method", choices=("static", "tts", "l0"), required=True)
    controlled.add_argument("--candidate-id")
    controlled.add_argument("--stage", type=int, default=-1)
    controlled.add_argument("--concurrency", type=int, required=True)
    controlled.add_argument("--output", required=True)
    controlled.add_argument("--no-warmup", action="store_true")

    controlled_online = commands.add_parser("run-onlinespec-tuning-slice")
    controlled_online.add_argument("--manifest", required=True)
    controlled_online.add_argument("--model-lock", required=True)
    controlled_online.add_argument("--sampling-profile", required=True)
    controlled_online.add_argument("--config", required=True)
    controlled_online.add_argument("--url", required=True)
    controlled_online.add_argument(
        "--method", choices=ONLINE_SPEC_STUDY_METHODS, required=True
    )
    controlled_online.add_argument("--candidate-id")
    controlled_online.add_argument("--stage", type=int, required=True)
    controlled_online.add_argument("--concurrency", type=int, required=True)
    controlled_online.add_argument("--output", required=True)
    controlled_online.add_argument("--no-warmup", action="store_true")

    natural = commands.add_parser("run-preliminary-natural-slice")
    natural.add_argument("--manifest", required=True)
    natural.add_argument("--selection", required=True)
    natural.add_argument("--model-lock", required=True)
    natural.add_argument("--sampling-profile", required=True)
    natural.add_argument("--config", required=True)
    natural.add_argument("--url", required=True)
    natural.add_argument("--method", choices=("static", "tts", "l0"), required=True)
    natural.add_argument(
        "--dataset", choices=("livecodebench", "math500"), required=True
    )
    natural.add_argument("--dataset-revision", required=True)
    natural.add_argument("--split", required=True)
    natural.add_argument("--output-root", required=True)
    natural.add_argument("--no-warmup", action="store_true")

    profiler = commands.add_parser("build-preliminary-profiler-plan")
    profiler.add_argument("--launch-plan", required=True)
    profiler.add_argument("--method", choices=("static", "tts", "l0"), required=True)
    profiler.add_argument("--trace-root", required=True)
    profiler.add_argument("--output", required=True)
    profiler.add_argument("workload_argv", nargs=argparse.REMAINDER)

    load_collect = commands.add_parser("collect-preliminary-static-load-screen")
    load_collect.add_argument("--manifest", required=True)
    load_collect.add_argument("--measurements", nargs="+", required=True)
    load_collect.add_argument("--output", required=True)

    advance = commands.add_parser("advance-preliminary-tuning-stage")
    advance.add_argument("--manifest", required=True)
    advance.add_argument("--stage", type=int, required=True)
    advance.add_argument("--measurements", nargs="+", required=True)
    advance.add_argument("--active-set")
    advance.add_argument("--output", required=True)

    advance_online = commands.add_parser("advance-onlinespec-tuning-stage")
    advance_online.add_argument("--manifest", required=True)
    advance_online.add_argument("--stage", type=int, required=True)
    advance_online.add_argument("--measurements", nargs="+", required=True)
    advance_online.add_argument("--active-set")
    advance_online.add_argument("--output", required=True)

    run = commands.add_parser("run-preliminary-confirmation")
    run.add_argument("--manifest", required=True)
    run.add_argument("--selection", required=True)
    run.add_argument("--model-lock", required=True)
    run.add_argument("--sampling-profile", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--url", required=True)
    run.add_argument("--method", choices=("static", "tts", "l0"), required=True)
    run.add_argument("--block", type=int, required=True)
    run.add_argument("--adaptation-group-id", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--no-warmup", action="store_true")

    run_online = commands.add_parser("run-onlinespec-confirmation")
    run_online.add_argument("--manifest", required=True)
    run_online.add_argument("--selection", required=True)
    run_online.add_argument("--model-lock", required=True)
    run_online.add_argument("--sampling-profile", required=True)
    run_online.add_argument("--config", required=True)
    run_online.add_argument("--url", required=True)
    run_online.add_argument(
        "--method", choices=ONLINE_SPEC_STUDY_METHODS, required=True
    )
    run_online.add_argument("--block", type=int, required=True)
    run_online.add_argument("--adaptation-group-id", required=True)
    run_online.add_argument("--output-root", required=True)
    run_online.add_argument("--no-warmup", action="store_true")

    target_reference = commands.add_parser("run-preliminary-target-reference")
    target_reference.add_argument("--manifest", required=True)
    target_reference.add_argument("--model-lock", required=True)
    target_reference.add_argument("--sampling-profile", required=True)
    target_reference.add_argument("--url", required=True)
    target_reference.add_argument("--concurrency", type=int, required=True)
    target_reference.add_argument("--doctor-json", required=True)
    target_reference.add_argument("--output", required=True)
    target_reference.add_argument("--no-warmup", action="store_true")

    collect = commands.add_parser("collect-preliminary-speed-study")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--selection", required=True)
    collect.add_argument("--model-lock", required=True)
    collect.add_argument("--static-config", required=True)
    collect.add_argument("--tts-config", required=True)
    collect.add_argument("--l0-config", required=True)
    collect.add_argument("--evidence-root", required=True)
    collect.add_argument("--target-reference", required=True)
    collect.add_argument("--output", required=True)

    collect_online = commands.add_parser("collect-onlinespec-study")
    collect_online.add_argument("--manifest", required=True)
    collect_online.add_argument("--selection", required=True)
    collect_online.add_argument("--model-lock", required=True)
    collect_online.add_argument("--static-config", required=True)
    collect_online.add_argument("--ogd-config", required=True)
    collect_online.add_argument("--opt-config", required=True)
    collect_online.add_argument("--ens-config", required=True)
    collect_online.add_argument("--evidence-root", required=True)
    collect_online.add_argument("--target-reference", required=True)
    collect_online.add_argument("--output", required=True)

    queue = commands.add_parser("build-preliminary-confirmation-queue")
    queue.add_argument("--manifest", required=True)
    queue.add_argument("--selection", required=True)
    queue.add_argument("--model-lock", required=True)
    queue.add_argument("--sampling-profile", required=True)
    queue.add_argument("--launch-plan", required=True)
    queue.add_argument("--evidence-root", required=True)
    queue.add_argument("--output", required=True)

    queue_online = commands.add_parser("build-onlinespec-queue")
    queue_online.add_argument("--manifest", required=True)
    queue_online.add_argument("--selection", required=True)
    queue_online.add_argument("--model-lock", required=True)
    queue_online.add_argument("--sampling-profile", required=True)
    queue_online.add_argument("--launch-plan", required=True)
    queue_online.add_argument("--evidence-root", required=True)
    queue_online.add_argument("--output", required=True)

    attest = commands.add_parser("attest-preliminary-speed-study")
    attest.add_argument("--manifest", required=True)
    attest.add_argument("--output", required=True)

    attest_online = commands.add_parser("attest-onlinespec-study")
    attest_online.add_argument("--manifest", required=True)
    attest_online.add_argument("--output", required=True)

    analyze = commands.add_parser("analyze-preliminary-speed-study")
    analyze.add_argument("--performance", required=True)
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--selection", required=True)
    analyze.add_argument("--model-lock", required=True)
    analyze.add_argument("--target-reference", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--bootstrap-seed", type=int, default=0)

    analyze_online = commands.add_parser("analyze-onlinespec-study")
    analyze_online.add_argument("--performance", required=True)
    analyze_online.add_argument("--manifest", required=True)
    analyze_online.add_argument("--selection", required=True)
    analyze_online.add_argument("--model-lock", required=True)
    analyze_online.add_argument("--target-reference", required=True)
    analyze_online.add_argument("--output", required=True)
    analyze_online.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def _select(args: argparse.Namespace) -> int:
    manifest = PreliminarySpeedStudyManifest.load(args.manifest)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("sampling profile belongs to a different manifest")
    tuning_artifact = _load_bound_json(args.measurements)
    load_artifact = _load_bound_json(args.static_load_screen)
    load_rows = _static_load_rows(load_artifact, manifest=manifest)
    measurements_value = tuning_artifact
    if isinstance(tuning_artifact, dict):
        if (
            tuning_artifact.get("schema_version") != 2
            or tuning_artifact.get("phase") != "shared_config_tuning"
            or tuning_artifact.get("manifest_sha256") != manifest.sha256
            or tuning_artifact.get("stage") != len(TUNING_STAGES) - 1
            or tuning_artifact.get("next_stage") is not None
            or not _is_lower_sha256(tuning_artifact.get("prior_stage_sha256"))
        ):
            raise ValueError("selection requires the terminal tuning-stage artifact")
        measurements_value = tuning_artifact.get("candidate_measurements")
    else:
        raise TypeError("selection requires a terminal tuning-stage artifact")
    if not isinstance(measurements_value, list):
        raise TypeError("terminal tuning measurements must be a JSON array")
    lock = ModelLock.load(args.model_lock)
    selected_concurrency = select_static_load(
        load_rows,
        required_context_limit=manifest.safe_context_limit,
    )
    if (
        tuning_artifact.get("model_lock_sha256") != lock.sha256
        or load_artifact.get("model_lock_sha256") != lock.sha256
    ):
        raise ValueError("selection inputs belong to a different model lock")
    if (
        tuning_artifact.get("sampling_profile_sha256") != sampling.sha256
        or tuning_artifact.get("execution_policy_sha256")
        != manifest.execution_policy_sha256
        or tuning_artifact.get("window_sha256")
        != manifest.controlled_window_hashes["tune"]
        or tuning_artifact.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
        or tuning_artifact.get("concurrency") != selected_concurrency
    ):
        raise ValueError(
            "terminal tuning artifact is not bound to this study and selected load"
        )
    measurements = [CandidateMeasurement(**row) for row in measurements_value]
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    artifact = select_shared_config(
        measurements,
        candidates=grid,
        selected_concurrency=selected_concurrency,
        manifest_sha256=manifest.sha256,
        sampling_profile_sha256=sampling.sha256,
        tuning_grid_sha256=manifest.tuning_grid_sha256,
        load_screen_sha256=_canonical_sha256(load_artifact),
        tuning_window_sha256=LongContinuationAdapter().window_sha256("tune"),
        model_lock_sha256=lock.sha256,
        tuning_evidence_sha256=_canonical_sha256(tuning_artifact),
    )
    artifact.write(args.output)
    print(artifact.sha256)
    return 0


def _select_anchor(args: argparse.Namespace) -> int:
    manifest = PreliminarySpeedStudyManifest.load(args.manifest)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("sampling profile belongs to a different manifest")
    lock = ModelLock.load(args.model_lock)
    load_artifact = _load_bound_json(args.static_load_screen)
    load_rows = _static_load_rows(load_artifact, manifest=manifest)
    selected_concurrency = select_static_load(
        load_rows,
        required_context_limit=manifest.safe_context_limit,
    )
    if load_artifact.get("model_lock_sha256") != lock.sha256:
        raise ValueError("Static load screen belongs to a different model lock")
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    candidate = grid.get(args.candidate_id)
    if candidate is None:
        raise ValueError("anchor candidate is outside the registered tuning grid")
    measurements = [SliceMeasurement.load(path) for path in args.measurements]
    expected_count, expected_context = tuning_stage(len(TUNING_STAGES) - 1)
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:expected_count]
    )
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.model_lock_sha256 != lock.sha256
        or row.sampling_profile_sha256 != sampling.sha256
        or row.window_sha256 != expected_window
        or row.prompt_count != expected_count
        or row.context_limit != expected_context
        for row in measurements
    ):
        raise ValueError("anchor measurement identity is not terminal tuning evidence")
    evidence = _canonical_sha256(
        {
            "selection_protocol": "heldout_anchor",
            "candidate_id": candidate.candidate_id,
            "measurement_sha256": sorted(row.sha256 for row in measurements),
        }
    )
    artifact = select_heldout_anchor(
        measurements,
        candidate=candidate,
        selected_concurrency=selected_concurrency,
        manifest_sha256=manifest.sha256,
        sampling_profile_sha256=sampling.sha256,
        tuning_grid_sha256=manifest.tuning_grid_sha256,
        load_screen_sha256=_canonical_sha256(load_artifact),
        tuning_window_sha256=manifest.controlled_window_hashes["tune"],
        model_lock_sha256=lock.sha256,
        tuning_evidence_sha256=evidence,
    )
    artifact.write(args.output)
    print(artifact.sha256)
    return 0


def _select_onlinespec(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    core_selection = SelectionArtifact.load(args.core_selection)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("OnlineSPEC manifest uses another sampling profile")
    if (
        core_selection.manifest_sha256 != PreliminarySpeedStudyManifest.default().sha256
        or core_selection.model_lock_sha256 != lock.sha256
        or core_selection.sampling_profile_sha256 != sampling.sha256
    ):
        raise ValueError(
            "OnlineSPEC selection requires the registered core Static load"
        )
    value = _load_bound_json(args.measurements)
    if (
        value.get("schema_version") != 2
        or value.get("phase") != "onlinespec_tuning"
        or value.get("manifest_sha256") != manifest.sha256
        or value.get("model_lock_sha256") != lock.sha256
        or value.get("sampling_profile_sha256") != sampling.sha256
        or value.get("execution_policy_sha256") != manifest.execution_policy_sha256
        or value.get("window_sha256") != manifest.tuning_window_sha256
        or value.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
        or value.get("stage") != len(ONLINE_SPEC_TUNING_STAGES) - 1
        or value.get("next_stage") is not None
        or not _is_lower_sha256(value.get("prior_stage_sha256"))
        or value.get("concurrency") != core_selection.selected_concurrency
    ):
        raise ValueError("OnlineSPEC tuning artifact identity mismatch")
    raw_rows = value.get("measurements")
    if not isinstance(raw_rows, list):
        raise TypeError("OnlineSPEC tuning artifact lacks measurements")
    candidates = {
        candidate.candidate_id: candidate for candidate in onlinespec_candidates()
    }
    selection = select_onlinespec(
        [OnlineSpecTuningMeasurement(**row) for row in raw_rows],
        candidates=candidates,
        selected_concurrency=core_selection.selected_concurrency,
        manifest_sha256=manifest.sha256,
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=sampling.sha256,
        reference_core_selection_sha256=core_selection.sha256,
        tuning_evidence_sha256=_canonical_sha256(value),
    )
    selection.write(args.output)
    print(selection.sha256)
    return 0


def _select_onlinespec_anchor(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    core_selection = SelectionArtifact.load(args.core_selection)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("OnlineSPEC manifest uses another sampling profile")
    if (
        core_selection.manifest_sha256 != PreliminarySpeedStudyManifest.default().sha256
        or core_selection.model_lock_sha256 != lock.sha256
        or core_selection.sampling_profile_sha256 != sampling.sha256
    ):
        raise ValueError(
            "OnlineSPEC selection requires the registered core Static load"
        )
    candidate_ids = tuple(args.candidate_ids)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("OnlineSPEC anchor candidate identities must be unique")
    registered = {
        candidate.candidate_id: candidate for candidate in onlinespec_candidates()
    }
    try:
        candidates = {
            candidate_id: registered[candidate_id] for candidate_id in candidate_ids
        }
    except KeyError as exc:
        raise ValueError(
            "OnlineSPEC anchor is outside the registered tuning grid"
        ) from exc
    if {candidate.method for candidate in candidates.values()} != set(
        ONLINE_SPEC_METHODS
    ):
        raise ValueError("OnlineSPEC anchor requires one candidate per learner")
    measurements = tuple(SliceMeasurement.load(path) for path in args.measurements)
    expected_count, expected_context = onlinespec_tuning_stage(
        len(ONLINE_SPEC_TUNING_STAGES) - 1
    )
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:expected_count]
    )
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.model_lock_sha256 != lock.sha256
        or row.sampling_profile_sha256 != sampling.sha256
        or row.window_sha256 != expected_window
        or row.prompt_count != expected_count
        or row.context_limit != expected_context
        for row in measurements
    ):
        raise ValueError(
            "OnlineSPEC anchor measurement identity is not terminal tuning evidence"
        )
    evidence = _canonical_sha256(
        {
            "selection_protocol": "heldout_anchor",
            "candidate_ids": sorted(candidate_ids),
            "measurement_sha256": sorted(row.sha256 for row in measurements),
        }
    )
    selection = select_onlinespec_heldout_anchor(
        measurements,
        candidates=candidates,
        selected_concurrency=core_selection.selected_concurrency,
        manifest_sha256=manifest.sha256,
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=sampling.sha256,
        reference_core_selection_sha256=core_selection.sha256,
        tuning_evidence_sha256=evidence,
    )
    selection.write(args.output)
    print(selection.sha256)
    return 0


def _assert_onlinespec_study(
    manifest: OnlineSpecManifest,
    selection: OnlineSpecSelection,
    lock: ModelLock,
    sampling: SamplingProfile | None = None,
) -> None:
    if (
        selection.manifest_sha256 != manifest.sha256
        or selection.model_lock_sha256 != lock.sha256
        or selection.sampling_profile_sha256 != manifest.sampling_profile_sha256
        or (
            sampling is not None
            and selection.sampling_profile_sha256 != sampling.sha256
        )
    ):
        raise ValueError("OnlineSPEC study identities do not match")


def _study_inputs(
    args: argparse.Namespace,
) -> tuple[PreliminarySpeedStudyManifest, ModelLock, SamplingProfile]:
    manifest = PreliminarySpeedStudyManifest.load(args.manifest)
    model_lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("sampling profile belongs to a different manifest")
    return manifest, model_lock, sampling


def _run_controlled_slice(args: argparse.Namespace) -> int:
    manifest, model_lock, sampling = _study_inputs(args)
    config = _load_bound_run_config(args.config)
    _assert_locked_config(
        config,
        model_lock=model_lock,
        sampling_profile=sampling,
    )
    if config.method != args.method:
        raise ValueError("run config is bound to another method")
    if config.runtime.max_running_requests < args.concurrency:
        raise ValueError("slice concurrency exceeds the runtime admission limit")
    adapter = LongContinuationAdapter()
    if args.phase == "static-load":
        if args.method != "static" or args.candidate_id is not None or args.stage != -1:
            raise ValueError(
                "Static load screen accepts only an unadapted stage -1 slice"
            )
        if args.concurrency not in manifest.concurrency_grid:
            raise ValueError("Static load concurrency is outside the registered grid")
        samples = adapter.window("load")
        phase = "static_load_screen"
        context_limit = manifest.load_screen_context_limit
        candidate_id = None
    else:
        prompt_count, context_limit = tuning_stage(args.stage)
        samples = adapter.window("tune")[:prompt_count]
        phase = "shared_config_tuning"
        candidate_id = args.candidate_id
        if args.method == "static":
            if candidate_id is not None:
                raise ValueError("Static tuning baseline has no candidate ID")
        else:
            grid = {
                candidate.candidate_id: candidate for candidate in tuning_candidates()
            }
            candidate = grid.get(candidate_id or "")
            if candidate is None:
                raise ValueError("adapted tuning slice has an unknown candidate ID")
            assert_confirmation_slice_config(
                config,
                method=args.method,
                selected_candidate=candidate,
                selected_concurrency=args.concurrency,
            )
    if config.model.max_context_length < context_limit:
        raise ValueError("slice context exceeds the locked model configuration")
    group = (
        "static-preselection"
        if config.adaptation is None
        else config.adaptation.adaptation_group_id
    )
    measurement = measure_preliminary_controlled_slice(
        preliminary_manifest=manifest,
        client=SGLangHTTPClient(args.url),
        method=args.method,
        samples=samples,
        phase=phase,
        stage=args.stage,
        candidate_id=candidate_id,
        config_sha256=run_config_sha256(config),
        model_lock_sha256=model_lock.sha256,
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        sampling_profile=sampling,
        context_limit=context_limit,
        concurrency=args.concurrency,
        adaptation_group_id=group,
        warmup=not args.no_warmup,
    )
    measurement.write(args.output)
    print(measurement.sha256)
    return 0


def _collect_static_load(args: argparse.Namespace) -> int:
    manifest = PreliminarySpeedStudyManifest.load(args.manifest)
    measurements = [SliceMeasurement.load(path) for path in args.measurements]
    expected_window = LongContinuationAdapter().window_sha256("load")
    if len(measurements) != len(manifest.concurrency_grid):
        raise ValueError("Static load screen has incomplete concurrency coverage")
    rows = []
    model_locks = {row.model_lock_sha256 for row in measurements}
    if len(model_locks) != 1:
        raise ValueError("Static load screen mixes model-lock identities")
    for row in measurements:
        if (
            row.phase != "static_load_screen"
            or row.stage != -1
            or row.method != "static"
            or row.manifest_sha256 != manifest.sha256
            or row.window_sha256 != expected_window
            or row.prompt_count != 8
            or row.context_limit != manifest.load_screen_context_limit
            or row.sampling_profile_sha256 != manifest.sampling_profile_sha256
        ):
            raise ValueError("Static load slice identity mismatch")
        rows.append(
            {
                "concurrency": row.concurrency,
                "decode_goodput_tps": row.decode_goodput_tps,
                "itl_p99_ms": row.itl_p99_ms,
                "oom_events": row.oom_events,
                "retractions": row.retractions,
                "kv_token_capacity": row.kv_token_capacity,
                "measurement_sha256": row.sha256,
            }
        )
    rows.sort(key=lambda value: int(value["concurrency"]))
    select_static_load(
        rows,
        required_context_limit=manifest.safe_context_limit,
    )
    artifact = {
        "schema_version": 2,
        "phase": "static_load_screen",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": next(iter(model_locks)),
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "window_sha256": expected_window,
        "rows": rows,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _advance_tuning(args: argparse.Namespace) -> int:
    manifest = PreliminarySpeedStudyManifest.load(args.manifest)
    tuning_stage(args.stage)
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    prior = None
    if args.stage == 0:
        if args.active_set:
            raise ValueError("stage zero must start from the complete registered grid")
        active = tuple(sorted(grid))
    else:
        if not args.active_set:
            raise ValueError("later tuning stages require the prior survivor artifact")
        prior = _load_bound_json(args.active_set)
        if (
            not isinstance(prior, dict)
            or prior.get("schema_version") != 2
            or prior.get("phase") != "shared_config_tuning"
            or prior.get("manifest_sha256") != manifest.sha256
            or prior.get("next_stage") != args.stage
            or prior.get("stage") != args.stage - 1
            or not isinstance(prior.get("survivors"), list)
            or prior.get("sampling_profile_sha256") != manifest.sampling_profile_sha256
            or prior.get("execution_policy_sha256") != manifest.execution_policy_sha256
            or prior.get("window_sha256") != manifest.controlled_window_hashes["tune"]
            or prior.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
            or (args.stage == 1 and prior.get("prior_stage_sha256") is not None)
            or (
                args.stage > 1 and not _is_lower_sha256(prior.get("prior_stage_sha256"))
            )
        ):
            raise ValueError("prior tuning survivor artifact is invalid")
        active = tuple(str(value) for value in prior["survivors"])
    measurements = [SliceMeasurement.load(path) for path in args.measurements]
    model_locks = {row.model_lock_sha256 for row in measurements}
    if len(model_locks) != 1:
        raise ValueError("tuning stage mixes model-lock identities")
    model_lock_sha256 = next(iter(model_locks))
    if prior is not None and prior.get("model_lock_sha256") != model_lock_sha256:
        raise ValueError(
            "tuning stage uses a different model lock than its predecessor"
        )
    expected_count, _ = tuning_stage(args.stage)
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:expected_count]
    )
    concurrencies = {row.concurrency for row in measurements}
    if len(concurrencies) != 1:
        raise ValueError("tuning stage mixes runtime concurrency")
    concurrency = next(iter(concurrencies))
    if prior is not None and prior.get("concurrency") != concurrency:
        raise ValueError("tuning stage changes the selected runtime load")
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.window_sha256 != expected_window
        or row.sampling_profile_sha256 != manifest.sampling_profile_sha256
        for row in measurements
    ):
        raise ValueError("tuning measurements use another manifest or prompt window")
    survivors, candidate_rows = reduce_tuning_stage(
        measurements,
        candidates=grid,
        active_candidate_ids=active,
        stage=args.stage,
    )
    next_stage = args.stage + 1 if args.stage + 1 < len(TUNING_STAGES) else None
    artifact = {
        "schema_version": 2,
        "phase": "shared_config_tuning",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": model_lock_sha256,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "window_sha256": manifest.controlled_window_hashes["tune"],
        "tuning_grid_sha256": manifest.tuning_grid_sha256,
        "concurrency": concurrency,
        "stage": args.stage,
        "next_stage": next_stage,
        "prior_stage_sha256": (None if prior is None else _canonical_sha256(prior)),
        "active_candidates": list(active),
        "survivors": list(survivors),
        "measurement_sha256": sorted(row.sha256 for row in measurements),
        "candidate_measurements": [asdict(row) for row in candidate_rows],
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _list_tuning_candidates(args: argparse.Namespace) -> int:
    rows = [
        {**asdict(candidate), "candidate_id": candidate.candidate_id}
        for candidate in tuning_candidates()
    ]
    _write_json(args.output, rows)
    print(_canonical_sha256(rows))
    return 0


def _list_onlinespec_candidates(args: argparse.Namespace) -> int:
    rows = [
        {**asdict(candidate), "candidate_id": candidate.candidate_id}
        for candidate in onlinespec_candidates()
    ]
    _write_json(args.output, rows)
    print(_canonical_sha256(rows))
    return 0


def _confirmation_inputs(
    args: argparse.Namespace,
) -> tuple[
    PreliminarySpeedStudyManifest,
    SelectionArtifact,
    ModelLock,
    SamplingProfile,
]:
    manifest = PreliminarySpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to a different model lock")
    if selection.tuning_window_sha256 != manifest.controlled_window_hashes["tune"]:
        raise ValueError("selection artifact belongs to a different tuning window")
    sampling_profile = SamplingProfile.load(args.sampling_profile)
    if (
        sampling_profile.sha256 != manifest.sampling_profile_sha256
        or selection.sampling_profile_sha256 != sampling_profile.sha256
        or selection.tuning_grid_sha256 != manifest.tuning_grid_sha256
    ):
        raise ValueError("sampling profile belongs to a different manifest")
    return manifest, selection, model_lock, sampling_profile


def _assert_selection_study(
    selection: SelectionArtifact, manifest: PreliminarySpeedStudyManifest
) -> None:
    if (
        selection.manifest_sha256 != manifest.sha256
        or selection.tuning_grid_sha256 != manifest.tuning_grid_sha256
        or selection.sampling_profile_sha256 != manifest.sampling_profile_sha256
        or selection.tuning_window_sha256 != manifest.controlled_window_hashes["tune"]
    ):
        raise ValueError("selection artifact belongs to a different speed study")


def _assert_locked_config(
    config: RunConfig,
    *,
    model_lock: ModelLock,
    sampling_profile: SamplingProfile,
) -> None:
    if config.runtime.sampling_profile_sha256 != sampling_profile.sha256:
        raise ValueError("run config does not match the sampling profile")
    if config.runtime.execution_policy_sha256 != ControlledExecutionPolicy().sha256:
        raise ValueError("run config does not match the registered execution policy")
    locked = {model.model_id: model.revision for model in model_lock.models}
    pair = config.model
    if (
        locked.get(pair.target) != pair.target_revision
        or locked.get(pair.drafter) != pair.drafter_revision
    ):
        raise ValueError("run config does not match the immutable model lock")


def _load_target_reference(
    path: str | Path,
    *,
    model_lock: ModelLock,
    sampling_profile_sha256: str,
    concurrency: int,
) -> GreedyTargetReference:
    revisions = {model.model_id: model.revision for model in model_lock.models}
    target_revision = revisions.get("Qwen/Qwen3-8B")
    if target_revision is None:
        raise ValueError("model lock lacks the formal Qwen3-8B target")
    reference = GreedyTargetReference.load(path)
    reference.verify_study(
        model_lock_sha256=model_lock.sha256,
        target_revision=target_revision,
        sampling_profile_sha256=sampling_profile_sha256,
        execution_policy_sha256=ControlledExecutionPolicy().sha256,
        window_sha256=LongContinuationAdapter().window_sha256("confirm"),
        concurrency=concurrency,
    )
    return reference


def _load_patched_gpu_doctor(path: str | Path, *, purpose: str) -> dict:
    hardware = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(hardware, dict):
        raise TypeError(f"{purpose} doctor evidence is not an object")
    commands = hardware.get("commands")
    if not isinstance(commands, dict):
        raise TypeError(f"{purpose} doctor commands are malformed")
    nvidia = commands.get("nvidia_smi")
    source_tree = hardware.get("source_tree")
    if not isinstance(nvidia, str) or not nvidia.strip():
        raise ValueError(f"{purpose} requires a successful nvidia-smi report")
    if (
        not isinstance(source_tree, dict)
        or source_tree.get("is_git_checkout") is not True
        or source_tree.get("tree") != PINNED_SGLANG_TREE
        or source_tree.get("dirty") is not False
        or source_tree.get("pinned_ancestor") is not True
        or source_tree.get("patch_commits") != PINNED_SGLANG_PATCH_COUNT
    ):
        raise ValueError(f"{purpose} requires the exact clean patched checkout")
    return hardware


def _run_confirmation(args: argparse.Namespace) -> int:
    manifest, selection, model_lock, sampling_profile = _confirmation_inputs(args)
    config = _load_bound_run_config(args.config)
    _assert_locked_config(
        config,
        model_lock=model_lock,
        sampling_profile=sampling_profile,
    )
    assert_confirmation_slice_config(
        config,
        method=args.method,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    if (
        config.adaptation is not None
        and config.adaptation.adaptation_group_id != args.adaptation_group_id
    ):
        raise ValueError("run argument and config adaptation groups differ")
    written = run_preliminary_confirmation_slice(
        preliminary_manifest=manifest,
        client=SGLangHTTPClient(args.url),
        method=args.method,
        block=args.block,
        config_sha256=run_config_sha256(config),
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        output_root=args.output_root,
        concurrency=selection.selected_concurrency,
        safe_context_limit=manifest.safe_context_limit,
        adaptation_group_id=args.adaptation_group_id,
        schedule_seed=manifest.confirmation_schedule_seed,
        sampling_profile=sampling_profile,
        model_pair=manifest.model_pair,
        warmup=not args.no_warmup,
    )
    if not written:
        raise RuntimeError("confirmation slice produced no completed evidence")
    print(
        f"{PRELIMINARY_DIAGNOSTIC_ONLY} "
        f"{args.block}/{args.method}: {len(written)} diagnostic files"
    )
    return 0


def _assert_onlinespec_candidate_config(
    config: RunConfig,
    *,
    method: str,
    candidate: OnlineSpecCandidate | None,
    concurrency: int,
) -> None:
    model = config.model
    runtime = config.runtime
    if (
        model.key != "qwen3_8b_dflash16"
        or model.target != "Qwen/Qwen3-8B"
        or model.drafter != "z-lab/Qwen3-8B-DFlash-b16"
        or model.algorithm != "DFLASH"
        or model.max_context_length != DFLASH_MODEL_CONTEXT_LIMIT
        or model.draft_depth != 15
        or runtime.speculative_num_draft_tokens != 16
        or runtime.telemetry_detail != "headline"
    ):
        raise ValueError("OnlineSPEC run config is outside the registered DFlash study")
    if config.method != method or runtime.max_running_requests != concurrency:
        raise ValueError("OnlineSPEC run config method or load mismatch")
    if method == "static":
        if candidate is not None:
            raise ValueError("OnlineSPEC Static reference has no candidate")
        if config.adaptation is not None or config.online_spec is not None:
            raise ValueError("OnlineSPEC Static reference allocated adaptation state")
        return
    adaptation = config.adaptation
    learner = config.online_spec
    if (
        candidate is None
        or candidate.method != method
        or adaptation is None
        or learner is None
    ):
        raise ValueError("OnlineSPEC run config is incomplete")
    actual = {
        "weight_update_mode": adaptation.weight_update_mode,
        "parameter_scope": adaptation.parameter_scope,
        "learning_rate": adaptation.optimizer.learning_rate,
        "rank": adaptation.rank,
        "stride": adaptation.stride,
        "projection_radius": learner.projection_radius,
        "additional_learning_rates": learner.additional_learning_rates,
        "hedge_learning_rate": learner.hedge_learning_rate,
        "grad_clip": adaptation.optimizer.grad_clip,
    }
    expected = {
        key: value for key, value in asdict(candidate).items() if key not in {"method"}
    }
    if (
        actual != expected
        or adaptation.optimizer.name != "sgd"
        or not math.isclose(
            adaptation.loss_position_decay,
            DFLASH_LOSS_POSITION_DECAY,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("OnlineSPEC run config does not match its tuning selection")


def _assert_onlinespec_config(
    config: RunConfig,
    *,
    method: str,
    selection: OnlineSpecSelection,
) -> None:
    candidate = next(
        (candidate for candidate in selection.selected if candidate.method == method),
        None,
    )
    _assert_onlinespec_candidate_config(
        config,
        method=method,
        candidate=candidate,
        concurrency=selection.selected_concurrency,
    )


def _run_onlinespec_tuning_slice(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("OnlineSPEC tuning uses another sampling profile")
    config = _load_bound_run_config(args.config)
    _assert_locked_config(config, model_lock=lock, sampling_profile=sampling)
    prompt_count, context_limit = onlinespec_tuning_stage(args.stage)
    samples = LongContinuationAdapter().window("tune")[:prompt_count]
    candidate = None
    if args.method == "static":
        if args.candidate_id is not None:
            raise ValueError("OnlineSPEC Static tuning has no candidate ID")
    else:
        grid = {row.candidate_id: row for row in onlinespec_candidates()}
        candidate = grid.get(args.candidate_id or "")
        if candidate is None:
            raise ValueError("OnlineSPEC tuning candidate is not registered")
    _assert_onlinespec_candidate_config(
        config,
        method=args.method,
        candidate=candidate,
        concurrency=args.concurrency,
    )
    if config.model.max_context_length < context_limit:
        raise ValueError("OnlineSPEC tuning exceeds the locked model context")
    group = (
        "onlinespec-static-tuning"
        if config.adaptation is None
        else config.adaptation.adaptation_group_id
    )
    measurement = measure_onlinespec_controlled_slice(
        onlinespec_manifest=manifest,
        client=SGLangHTTPClient(args.url),
        method=args.method,
        samples=samples,
        phase="onlinespec_tuning",
        stage=args.stage,
        candidate_id=(None if candidate is None else candidate.candidate_id),
        config_sha256=run_config_sha256(config),
        model_lock_sha256=lock.sha256,
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        sampling_profile=sampling,
        context_limit=context_limit,
        concurrency=args.concurrency,
        adaptation_group_id=group,
        warmup=not args.no_warmup,
    )
    measurement.write(args.output)
    print(measurement.sha256)
    return 0


def _advance_onlinespec_tuning(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    onlinespec_tuning_stage(args.stage)
    grid = {candidate.candidate_id: candidate for candidate in onlinespec_candidates()}
    prior = None
    if args.stage == 0:
        if args.active_set:
            raise ValueError("stage zero starts from the complete OnlineSPEC grid")
        active = tuple(sorted(grid))
    else:
        if not args.active_set:
            raise ValueError("later OnlineSPEC stages require survivor evidence")
        prior = _load_bound_json(args.active_set)
        if (
            prior.get("schema_version") != 2
            or prior.get("phase") != "onlinespec_tuning"
            or prior.get("manifest_sha256") != manifest.sha256
            or prior.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
            or prior.get("window_sha256") != manifest.tuning_window_sha256
            or prior.get("sampling_profile_sha256") != manifest.sampling_profile_sha256
            or prior.get("execution_policy_sha256") != manifest.execution_policy_sha256
            or prior.get("stage") != args.stage - 1
            or prior.get("next_stage") != args.stage
            or not isinstance(prior.get("survivors"), list)
            or (args.stage == 1 and prior.get("prior_stage_sha256") is not None)
            or (
                args.stage > 1 and not _is_lower_sha256(prior.get("prior_stage_sha256"))
            )
        ):
            raise ValueError("prior OnlineSPEC tuning artifact is invalid")
        active = tuple(str(value) for value in prior["survivors"])
    measurements = tuple(SliceMeasurement.load(path) for path in args.measurements)
    model_locks = {row.model_lock_sha256 for row in measurements}
    concurrencies = {row.concurrency for row in measurements}
    if len(model_locks) != 1 or len(concurrencies) != 1:
        raise ValueError("OnlineSPEC stage mixes model locks or runtime loads")
    model_lock_sha256 = next(iter(model_locks))
    concurrency = next(iter(concurrencies))
    if prior is not None and (
        prior.get("model_lock_sha256") != model_lock_sha256
        or prior.get("concurrency") != concurrency
    ):
        raise ValueError("OnlineSPEC tuning changed its model lock or load")
    prompt_count, _ = onlinespec_tuning_stage(args.stage)
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:prompt_count]
    )
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.sampling_profile_sha256 != manifest.sampling_profile_sha256
        or row.window_sha256 != expected_window
        for row in measurements
    ):
        raise ValueError("OnlineSPEC measurements use another registered input")
    survivors, reduced = reduce_onlinespec_tuning_stage(
        measurements,
        candidates=grid,
        active_candidate_ids=active,
        stage=args.stage,
    )
    next_stage = (
        args.stage + 1 if args.stage + 1 < len(ONLINE_SPEC_TUNING_STAGES) else None
    )
    artifact = {
        "schema_version": 2,
        "phase": "onlinespec_tuning",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": model_lock_sha256,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "tuning_grid_sha256": manifest.tuning_grid_sha256,
        "window_sha256": manifest.tuning_window_sha256,
        "measurement_window_sha256": expected_window,
        "concurrency": concurrency,
        "stage": args.stage,
        "next_stage": next_stage,
        "prior_stage_sha256": (None if prior is None else _canonical_sha256(prior)),
        "active_candidates": list(active),
        "survivors": list(survivors),
        "measurement_sha256": sorted(row.sha256 for row in measurements),
        "measurements": [asdict(row) for row in reduced],
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _run_onlinespec_confirmation(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    _assert_onlinespec_study(manifest, selection, lock, sampling)
    config = _load_bound_run_config(args.config)
    _assert_locked_config(config, model_lock=lock, sampling_profile=sampling)
    _assert_onlinespec_config(config, method=args.method, selection=selection)
    if (
        config.adaptation is not None
        and config.adaptation.adaptation_group_id != args.adaptation_group_id
    ):
        raise ValueError("OnlineSPEC adaptation group mismatch")
    written = run_onlinespec_confirmation_slice(
        onlinespec_manifest=manifest,
        client=SGLangHTTPClient(args.url),
        method=args.method,
        block=args.block,
        config_sha256=run_config_sha256(config),
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        output_root=args.output_root,
        concurrency=selection.selected_concurrency,
        safe_context_limit=manifest.safe_context_limit,
        adaptation_group_id=args.adaptation_group_id,
        schedule_seed=manifest.confirmation_schedule_seed,
        sampling_profile=sampling,
        warmup=not args.no_warmup,
    )
    if not written:
        raise RuntimeError("OnlineSPEC slice produced no completed evidence")
    print(f"completed OnlineSPEC {args.block}/{args.method}: {len(written)} files")
    return 0


def _run_target_reference(args: argparse.Namespace) -> int:
    manifest = PreliminarySpeedStudyManifest.load(args.manifest)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    hardware = _load_patched_gpu_doctor(
        args.doctor_json,
        purpose="target reference",
    )
    revisions = {model.model_id: model.revision for model in lock.models}
    target_revision = revisions.get("Qwen/Qwen3-8B")
    if target_revision is None:
        raise ValueError("model lock lacks the preliminary Qwen3-8B target")
    artifact = run_preliminary_greedy_target_reference(
        preliminary_manifest=manifest,
        client=SGLangHTTPClient(args.url),
        model_lock_sha256=lock.sha256,
        target_revision=target_revision,
        hardware_sha256=_canonical_sha256(hardware),
        concurrency=args.concurrency,
        sampling_profile=sampling,
        warmup=not args.no_warmup,
    )
    artifact.write(args.output)
    print(artifact.sha256)
    return 0


def _confirmation_configs(args: argparse.Namespace) -> dict:
    return {
        "static": _load_bound_run_config(args.static_config),
        "tts": _load_bound_run_config(args.tts_config),
        "l0": _load_bound_run_config(args.l0_config),
    }


def _concat_evidence_tables(paths: tuple[Path, ...]) -> pa.Table:
    """Concatenate evidence while promoting only all-null inferred columns."""
    tables = [pq.read_table(path) for path in paths]
    if not tables:
        raise ValueError("evidence table set cannot be empty")
    column_names = tables[0].column_names
    if any(table.column_names != column_names for table in tables[1:]):
        raise ValueError("evidence tables have different columns")
    for index, name in enumerate(column_names):
        concrete_types: list[pa.DataType] = []
        for table in tables:
            data_type = table.schema.field(index).type
            if not pa.types.is_null(data_type) and data_type not in concrete_types:
                concrete_types.append(data_type)
        if len(concrete_types) > 1:
            rendered = ", ".join(str(value) for value in concrete_types)
            raise ValueError(
                f"evidence column {name!r} has incompatible types: {rendered}"
            )
    return pa.concat_tables(tables, promote_options="default")


def _collect_speed_study(args: argparse.Namespace) -> int:
    manifest = PreliminarySpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to a different model lock")
    configs = _confirmation_configs(args)
    for config in configs.values():
        if config.runtime.sampling_profile_sha256 != manifest.sampling_profile_sha256:
            raise ValueError("run config does not match the manifest sampling profile")
        if config.runtime.execution_policy_sha256 != manifest.execution_policy_sha256:
            raise ValueError("run config does not match the execution policy")
        locked = {model.model_id: model.revision for model in model_lock.models}
        if (
            locked.get(config.model.target) != config.model.target_revision
            or locked.get(config.model.drafter) != config.model.drafter_revision
        ):
            raise ValueError("run config does not match the immutable model lock")
    assert_historical_matched_recipe_diagnostic_configs(
        configs,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=model_lock,
        sampling_profile_sha256=manifest.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    target_revision = next(
        model.revision
        for model in model_lock.models
        if model.model_id == "Qwen/Qwen3-8B"
    )
    performance, source_evidence_sha256 = collect_preliminary_confirmation_performance(
        preliminary_manifest=manifest,
        evidence_root=args.evidence_root,
        config_sha256={
            method: run_config_sha256(config) for method, config in configs.items()
        },
        concurrency=selection.selected_concurrency,
        target_reference=target_reference,
        model_lock_sha256=model_lock.sha256,
        sampling_profile_sha256=manifest.sampling_profile_sha256,
        execution_policy_sha256=manifest.execution_policy_sha256,
        target_revision=target_revision,
    )
    table = _concat_evidence_tables(performance)
    table = table.replace_schema_metadata(
        _preliminary_table_metadata(
            manifest=manifest,
            selection=selection,
            model_lock=model_lock,
            config_sha256={
                method: run_config_sha256(config) for method, config in configs.items()
            },
            source_evidence_sha256=source_evidence_sha256,
            target_reference_sha256=target_reference.sha256,
        )
    )
    output = Path(args.output)
    if output.exists():
        existing = _load_preliminary_table(
            output,
            manifest=manifest,
            selection=selection,
            model_lock=model_lock,
            target_reference=target_reference,
        )
        if existing.schema.metadata == table.schema.metadata and existing.equals(table):
            print(output)
            return 0
        raise RuntimeError("existing speed-study table differs from evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, output)
    print(output)
    return 0


def _collect_onlinespec_study(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    _assert_onlinespec_study(manifest, selection, lock)
    configs = {
        "static": _load_bound_run_config(args.static_config),
        "onlinespec_ogd": _load_bound_run_config(args.ogd_config),
        "onlinespec_opt": _load_bound_run_config(args.opt_config),
        "onlinespec_ens": _load_bound_run_config(args.ens_config),
    }
    locked = {model.model_id: model.revision for model in lock.models}
    for method, config in configs.items():
        if config.runtime.sampling_profile_sha256 != selection.sampling_profile_sha256:
            raise ValueError("OnlineSPEC config sampling identity mismatch")
        if config.runtime.execution_policy_sha256 != manifest.execution_policy_sha256:
            raise ValueError("OnlineSPEC config execution-policy identity mismatch")
        if (
            locked.get(config.model.target) != config.model.target_revision
            or locked.get(config.model.drafter) != config.model.drafter_revision
        ):
            raise ValueError("OnlineSPEC config model-lock identity mismatch")
        _assert_onlinespec_config(config, method=method, selection=selection)
    config_hashes = {
        method: run_config_sha256(config) for method, config in configs.items()
    }
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=lock,
        sampling_profile_sha256=selection.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    target_revision = next(
        model.revision for model in lock.models if model.model_id == "Qwen/Qwen3-8B"
    )
    performance, evidence_sha256 = collect_onlinespec_performance(
        onlinespec_manifest=manifest,
        evidence_root=args.evidence_root,
        config_sha256=config_hashes,
        concurrency=selection.selected_concurrency,
        target_reference=target_reference,
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=selection.sampling_profile_sha256,
        execution_policy_sha256=manifest.execution_policy_sha256,
        target_revision=target_revision,
    )
    table = _concat_evidence_tables(performance)
    metadata = {
        b"lightcone_schema_version": b"2",
        b"lightcone_study": b"onlinespec-clean-room-baseline",
        b"lightcone_manifest_kind": manifest.kind.encode(),
        b"lightcone_evidence_scope": manifest.evidence_scope.encode(),
        b"lightcone_formal_execution_authorized": b"false",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": selection.sampling_profile_sha256.encode(),
        b"lightcone_execution_policy_sha256": (
            manifest.execution_policy_sha256.encode()
        ),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_config_set_sha256": _canonical_sha256(config_hashes).encode(),
        b"lightcone_source_evidence_sha256": evidence_sha256.encode(),
        b"lightcone_target_reference_sha256": target_reference.sha256.encode(),
    }
    table = table.replace_schema_metadata(metadata)
    output = Path(args.output)
    if output.exists():
        existing = pq.read_table(output)
        if existing.schema.metadata == metadata and existing.equals(table):
            print(output)
            return 0
        raise RuntimeError("existing OnlineSPEC table differs from evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, output)
    print(output)
    return 0


def _render_runtime(args: argparse.Namespace) -> int:
    selection = SelectionArtifact.load(args.selection)
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    launches = render_runtime_plan(
        output_root=args.output_root,
        selection=selection,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        compile_cache_plan_path=args.compile_cache_plan,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    print(Path(args.output_root).resolve() / "launch-plan.json")
    if len(launches) != 3:
        raise AssertionError("runtime renderer did not create three method slices")
    if len({launch.base_url for launch in launches}) != 1 or not all(
        launch.exclusive_device for launch in launches
    ):
        raise AssertionError("formal method slices must share one exclusive endpoint")
    return 0


def _render_onlinespec_runtime(args: argparse.Namespace) -> int:
    selection = OnlineSpecSelection.load(args.selection)
    launches = render_onlinespec_runtime_plan(
        output_root=args.output_root,
        selection=selection,
        model_lock=ModelLock.load(args.model_lock),
        model_roots=_load_bound_json(args.model_roots),
        sampling_profile=SamplingProfile.load(args.sampling_profile),
        sglang_checkout=args.sglang_checkout,
        compile_cache_plan_path=args.compile_cache_plan,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if [launch.method for launch in launches] != list(ONLINE_SPEC_STUDY_METHODS):
        raise AssertionError("OnlineSPEC runtime plan has incomplete method coverage")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_onlinespec_tuning_runtime(args: argparse.Namespace) -> int:
    grid = {candidate.candidate_id: candidate for candidate in onlinespec_candidates()}
    candidate = grid.get(args.candidate_id)
    if candidate is None:
        raise ValueError("OnlineSPEC candidate ID is outside the registered grid")
    launches = render_onlinespec_tuning_runtime_plan(
        output_root=args.output_root,
        candidate=candidate,
        concurrency=args.concurrency,
        model_lock=ModelLock.load(args.model_lock),
        model_roots=_load_bound_json(args.model_roots),
        sampling_profile=SamplingProfile.load(args.sampling_profile),
        sglang_checkout=args.sglang_checkout,
        compile_cache_plan_path=args.compile_cache_plan,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if [launch.method for launch in launches] != ["static", candidate.method]:
        raise AssertionError("OnlineSPEC tuning plan is not paired to Static")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_static_load_runtime(args: argparse.Namespace) -> int:
    if args.concurrency not in {1, 2, 4, 8, 16, 32, 48}:
        raise ValueError("Static concurrency is outside the registered load grid")
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.purpose != "controlled":
        raise ValueError("Static load screen requires the controlled sampling profile")
    launches = render_static_load_runtime_plan(
        output_root=args.output_root,
        concurrency=args.concurrency,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        compile_cache_plan_path=args.compile_cache_plan,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if (
        len(launches) != 1
        or launches[0].method != "static"
        or launches[0].adaptation_config is not None
        or launches[0].telemetry_path is not None
        or "--speculative-adaptation-config" in launches[0].argv
    ):
        raise AssertionError("Static load renderer allocated an adaptation path")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_target_only_runtime(args: argparse.Namespace) -> int:
    if args.concurrency < 1:
        raise ValueError("Target-only concurrency must be positive")
    launches = render_target_only_runtime_plan(
        output_root=args.output_root,
        concurrency=args.concurrency,
        gpu_uuid=args.gpu_uuid,
        model_lock=ModelLock.load(args.model_lock),
        model_roots=_load_bound_json(args.model_roots),
        sampling_profile=SamplingProfile.load(args.sampling_profile),
        sglang_checkout=args.sglang_checkout,
        compile_cache_plan_path=args.compile_cache_plan,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    launch = launches[0]
    if (
        launch.method != "target_only"
        or launch.adaptation_config is not None
        or launch.telemetry_path is not None
        or "--speculative-algorithm" in launch.argv
        or "--speculative-draft-model-path" in launch.argv
        or any("draft" in argument for argument in launch.argv)
        or any("adaptation" in argument for argument in launch.argv)
    ):
        raise AssertionError("Target-only renderer allocated speculative state")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_tuning_runtime(args: argparse.Namespace) -> int:
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    candidate = grid.get(args.candidate_id)
    if candidate is None:
        raise ValueError("candidate ID is outside the registered tuning grid")
    if args.concurrency not in {1, 2, 4, 8, 16, 32, 48}:
        raise ValueError("tuning concurrency is outside the registered load grid")
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    launches = render_tuning_runtime_plan(
        output_root=args.output_root,
        candidate=candidate,
        concurrency=args.concurrency,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        compile_cache_plan_path=args.compile_cache_plan,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if [launch.method for launch in launches] != ["tts", "l0"] or len(
        {launch.base_url for launch in launches}
    ) != 1:
        raise AssertionError("tuning runtime must contain two adapted slices")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_replication_runtime(args: argparse.Namespace) -> int:
    selection = SelectionArtifact.load(args.selection)
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    phase = (
        "natural_task_replication"
        if args.phase == "natural"
        else "independent_profiler"
    )
    if phase == "natural_task_replication" and sampling.purpose != "natural":
        raise ValueError("natural runtime requires an EOS-enabled sampling profile")
    if phase == "independent_profiler" and sampling.purpose != "controlled":
        raise ValueError("profiler runtime requires the controlled sampling profile")
    launches = render_replication_runtime_plan(
        output_root=args.output_root,
        selection=selection,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        compile_cache_plan_path=args.compile_cache_plan,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        phase=phase,
        host=args.host,
        first_port=args.first_port,
    )
    if len(launches) != 3 or not all(launch.exclusive_device for launch in launches):
        raise AssertionError("replication runtime slices are not exclusive")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _run_natural_slice(args: argparse.Namespace) -> int:
    manifest = PreliminarySpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to another model lock")
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.purpose != "natural" or sampling.ignore_eos:
        raise ValueError("natural side table requires the EOS-enabled profile")
    config = _load_bound_run_config(args.config)
    _assert_locked_config(
        config,
        model_lock=model_lock,
        sampling_profile=sampling,
    )
    assert_confirmation_slice_config(
        config,
        method=args.method,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    samples = load_natural_prompts(
        args.dataset,
        revision=args.dataset_revision,
        split=args.split,
        limit=32,
    )
    adaptation_group_id = (
        "natural-static"
        if config.adaptation is None
        else config.adaptation.adaptation_group_id
    )
    paths = run_preliminary_natural_replication_slice(
        preliminary_manifest=manifest,
        client=SGLangHTTPClient(args.url),
        method=args.method,
        dataset_name=args.dataset,
        samples=samples,
        config_sha256=run_config_sha256(config),
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        output_root=args.output_root,
        concurrency=selection.selected_concurrency,
        safe_context_limit=manifest.safe_context_limit,
        adaptation_group_id=adaptation_group_id,
        sampling_profile=sampling,
        model_pair=manifest.model_pair,
        warmup=not args.no_warmup,
    )
    print(f"completed natural {args.dataset}/{args.method}: {len(paths)} files")
    return 0


def _build_profiler_plan(args: argparse.Namespace) -> int:
    source = _load_bound_json(args.launch_plan)
    if (
        source.get("schema_version") != 2
        or source.get("phase") != "independent_profiler"
        or source.get("execution_mode") != "sequential_exclusive_device"
        or source.get("patched_sglang_tree") != PINNED_SGLANG_TREE
        or source.get("execution_policy_sha256") != ControlledExecutionPolicy().sha256
    ):
        raise ValueError("profiler requires an independent-profiler launch plan")
    verify_patched_checkout(str(source.get("sglang_checkout", "")))
    servers = source.get("servers")
    if not isinstance(servers, list):
        raise TypeError("profiler launch plan lacks server slices")
    matching = [row for row in servers if row.get("method") == args.method]
    if len(matching) != 1 or matching[0].get("exclusive_device") is not True:
        raise ValueError("profiler method slice is missing or not exclusive")
    workload = list(args.workload_argv)
    if workload and workload[0] == "--":
        workload = workload[1:]
    if not workload or not all(isinstance(value, str) and value for value in workload):
        raise ValueError("profiler plan requires an explicit workload argv after --")
    trace_root = Path(args.trace_root).resolve()
    server_argv = matching[0].get("argv")
    if not isinstance(server_argv, list) or not server_argv:
        raise ValueError("profiler server argv is missing")
    artifact = {
        "schema_version": 2,
        "phase": "independent_profiler",
        "method": args.method,
        "launch_plan_sha256": _canonical_sha256(source),
        "exclusive_device": True,
        "profile_launch_argv": [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--force-overwrite=false",
            "--output",
            str(trace_root / args.method),
            *server_argv,
        ],
        "workload_argv": workload,
        "device_monitor_argv": [
            "nvidia-smi",
            "dmon",
            "-s",
            "puctm",
            "-o",
            "DT",
        ],
        "headline_evidence_forbidden": True,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _build_confirmation_queue(args: argparse.Namespace) -> int:
    manifest, selection, model_lock, sampling = _confirmation_inputs(args)
    plan_path = Path(args.launch_plan).resolve()
    plan = _load_bound_json(plan_path)
    if (
        plan.get("schema_version") != 2
        or plan.get("execution_mode") != "sequential_exclusive_device"
        or plan.get("selection_sha256") != selection.sha256
        or plan.get("model_lock_sha256") != model_lock.sha256
        or plan.get("sampling_profile_sha256") != sampling.sha256
        or plan.get("execution_policy_sha256") != manifest.execution_policy_sha256
        or plan.get("patched_sglang_tree") != PINNED_SGLANG_TREE
    ):
        raise ValueError("launch plan identity does not match the speed study")
    verify_patched_checkout(str(plan.get("sglang_checkout", "")))
    rows = plan.get("servers")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("launch plan must contain exactly three method slices")
    servers: dict[str, dict] = {}
    configs: dict[str, RunConfig] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("launch plan server entry is not an object")
        method = row.get("method")
        if method not in {"static", "tts", "l0"} or method in servers:
            raise ValueError("launch plan has invalid or duplicate methods")
        if row.get("exclusive_device") is not True:
            raise ValueError("every formal slice must own the GPU exclusively")
        argv = row.get("argv")
        if not isinstance(argv, list) or not all(
            isinstance(value, str) and value for value in argv
        ):
            raise ValueError("launch argv must be a non-empty string vector")
        config_path = Path(str(row.get("run_config"))).resolve()
        config = _load_bound_run_config(config_path)
        _assert_locked_config(
            config,
            model_lock=model_lock,
            sampling_profile=sampling,
        )
        servers[method] = {**row, "run_config": str(config_path)}
        configs[method] = config
    if len({str(row["base_url"]) for row in servers.values()}) != 1:
        raise ValueError("sequential method slices must reuse one endpoint")
    assert_historical_matched_recipe_diagnostic_configs(
        configs,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    adaptation = configs["tts"].adaptation
    if adaptation is None:
        raise AssertionError("TTS launch lacks adaptation identity")
    common = [
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--selection",
        str(Path(args.selection).resolve()),
        "--model-lock",
        str(Path(args.model_lock).resolve()),
        "--sampling-profile",
        str(Path(args.sampling_profile).resolve()),
        "--adaptation-group-id",
        adaptation.adaptation_group_id,
        "--output-root",
        str(Path(args.evidence_root).resolve()),
    ]
    jobs: list[dict] = []
    ordinal = 0
    for block in confirmation_blocks(manifest.confirmation_schedule_seed):
        for method in block.method_order:
            server = servers[method]
            jobs.append(
                {
                    "ordinal": ordinal,
                    "block": block.block,
                    "method": method,
                    "launch_argv": server["argv"],
                    "run_argv": [
                        "lightcone-spec",
                        "run-preliminary-confirmation",
                        *common,
                        "--config",
                        server["run_config"],
                        "--url",
                        str(server["base_url"]),
                        "--method",
                        method,
                        "--block",
                        str(block.block),
                    ],
                    "requires_clean_server_start": True,
                    "requires_server_exit_after": True,
                }
            )
            ordinal += 1
    artifact = {
        "schema_version": 2,
        "execution_mode": "sequential_exclusive_device",
        "manifest_sha256": manifest.sha256,
        "selection_sha256": selection.sha256,
        "model_lock_sha256": model_lock.sha256,
        "sampling_profile_sha256": sampling.sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "patched_sglang_tree": PINNED_SGLANG_TREE,
        "launch_plan_sha256": _canonical_sha256(plan),
        "schedule_seed": manifest.confirmation_schedule_seed,
        "jobs": jobs,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _build_onlinespec_queue(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    _assert_onlinespec_study(manifest, selection, lock, sampling)
    plan = _load_bound_json(args.launch_plan)
    if (
        plan.get("schema_version") != 2
        or plan.get("phase") != "onlinespec_paired_confirmation"
        or plan.get("selection_sha256") != selection.sha256
        or plan.get("model_lock_sha256") != lock.sha256
        or plan.get("sampling_profile_sha256") != sampling.sha256
        or plan.get("execution_policy_sha256") != manifest.execution_policy_sha256
        or plan.get("patched_sglang_tree") != PINNED_SGLANG_TREE
    ):
        raise ValueError("OnlineSPEC launch plan identity mismatch")
    verify_patched_checkout(str(plan.get("sglang_checkout", "")))
    raw_servers = plan.get("servers")
    if not isinstance(raw_servers, list) or len(raw_servers) != 4:
        raise ValueError("OnlineSPEC launch plan requires four method slices")
    servers = {}
    for row in raw_servers:
        if (
            not isinstance(row, dict)
            or row.get("method") not in ONLINE_SPEC_STUDY_METHODS
        ):
            raise ValueError("OnlineSPEC launch plan contains an invalid server")
        method = str(row["method"])
        if method in servers or row.get("exclusive_device") is not True:
            raise ValueError("OnlineSPEC servers must be unique and exclusive")
        config = _load_bound_run_config(row["run_config"])
        _assert_locked_config(config, model_lock=lock, sampling_profile=sampling)
        _assert_onlinespec_config(config, method=method, selection=selection)
        servers[method] = {**row, "config": config}
    if (
        set(servers) != set(ONLINE_SPEC_STUDY_METHODS)
        or len({str(row["base_url"]) for row in servers.values()}) != 1
    ):
        raise ValueError("OnlineSPEC servers do not share one exclusive endpoint")
    group_ids = {
        row["config"].adaptation.adaptation_group_id
        for method, row in servers.items()
        if method != "static"
    }
    if len(group_ids) != 1:
        raise ValueError("OnlineSPEC configs do not share one cohort group")
    common = [
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--selection",
        str(Path(args.selection).resolve()),
        "--model-lock",
        str(Path(args.model_lock).resolve()),
        "--sampling-profile",
        str(Path(args.sampling_profile).resolve()),
        "--adaptation-group-id",
        next(iter(group_ids)),
        "--output-root",
        str(Path(args.evidence_root).resolve()),
    ]
    jobs = []
    ordinal = 0
    for block in onlinespec_blocks(manifest.confirmation_schedule_seed):
        for method in block.method_order:
            server = servers[method]
            jobs.append(
                {
                    "ordinal": ordinal,
                    "block": block.block,
                    "method": method,
                    "launch_argv": server["argv"],
                    "run_argv": [
                        "lightcone-spec",
                        "run-onlinespec-confirmation",
                        *common,
                        "--config",
                        str(Path(server["run_config"]).resolve()),
                        "--url",
                        str(server["base_url"]),
                        "--method",
                        method,
                        "--block",
                        str(block.block),
                    ],
                    "requires_clean_server_start": True,
                    "requires_server_exit_after": True,
                }
            )
            ordinal += 1
    artifact = {
        "schema_version": 2,
        "execution_mode": "sequential_exclusive_device",
        "study": "onlinespec-clean-room-baseline",
        "manifest_sha256": manifest.sha256,
        "selection_sha256": selection.sha256,
        "model_lock_sha256": lock.sha256,
        "sampling_profile_sha256": sampling.sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "patched_sglang_tree": PINNED_SGLANG_TREE,
        "launch_plan_sha256": _canonical_sha256(plan),
        "schedule_seed": manifest.confirmation_schedule_seed,
        "jobs": jobs,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


_ATTESTATION_DOCTOR_CHECKS = frozenset(
    {
        "compatibility_manifest",
        "project_patch_binding",
        "project_sglang_roots_distinct",
        "project_source_tree",
        "project_runtime_source",
        "python",
        "linux_host",
        "torch",
        "triton",
        "flashinfer",
        "flashinfer_cuda_flavor",
        "cuda_build",
        "cuda_toolkit",
        "torch_cuda_visibility",
        "driver",
        "gpu_count",
        "gpu_identity",
        "gpu_memory",
        "gpu_topology",
        "compiler",
        "disk",
        "network",
        "sglang_commit_lineage",
        "sglang_tree",
        "sglang_import",
    }
)


def _validate_attestation_doctor(hardware: object, *, label: str) -> dict:
    def reject(reason: str) -> NoReturn:
        raise ValueError(
            f"{label} attestation requires a complete PASS doctor report: {reason}"
        )

    if not isinstance(hardware, dict) or hardware.get("schema_version") != 2:
        reject("doctor schema-v2 object is required")
    readiness = hardware.get("readiness")
    compatibility = hardware.get("compatibility")
    if hardware.get("status") != "PASS":
        reject("top-level readiness is not PASS")
    if not isinstance(readiness, dict) or readiness.get("status") != "PASS":
        reject("readiness.status is not PASS")
    if not isinstance(compatibility, dict) or compatibility.get("status") != "PASS":
        reject("compatibility.status is not PASS")

    runtime_manifest = hardware.get("runtime_manifest")
    if (
        not isinstance(runtime_manifest, dict)
        or runtime_manifest.get("valid") is not True
    ):
        reject("runtime compatibility manifest is not valid")
    manifest_sha256 = runtime_manifest.get("sha256")
    if (
        not _is_lower_sha256(manifest_sha256)
        or runtime_manifest.get("sidecar_sha256") != manifest_sha256
        or runtime_manifest.get("error") is not None
        or compatibility.get("manifest_sha256") != manifest_sha256
    ):
        reject("runtime compatibility manifest digests are inconsistent")

    checks = hardware.get("checks")
    if not isinstance(checks, dict):
        reject("doctor checks are missing")
    missing_checks = sorted(_ATTESTATION_DOCTOR_CHECKS - checks.keys())
    if missing_checks:
        reject(f"required doctor checks are missing: {', '.join(missing_checks)}")
    if any(
        not isinstance(check, dict) or check.get("status") != "PASS"
        for check in checks.values()
    ):
        reject("every doctor check must be PASS")
    if (
        readiness.get("pass_count") != len(checks)
        or readiness.get("fail_count") != 0
        or readiness.get("unknown_count") != 0
    ):
        reject("doctor readiness counters do not match its checks")

    roots = hardware.get("roots")
    if not isinstance(roots, dict):
        reject("project and patched-SGLang roots are missing")
    project_root = roots.get("project")
    sglang_root = roots.get("patched_sglang")
    if (
        roots.get("distinct") is not True
        or not isinstance(project_root, str)
        or not project_root
        or not isinstance(sglang_root, str)
        or not sglang_root
        or project_root == sglang_root
    ):
        reject("project and patched-SGLang roots must be distinct")
    try:
        _require_project_runtime_source_identity(
            hardware,
            executing_file=Path(__file__),
        )
    except (TypeError, ValueError):
        reject("LightCone source identity is stale or incomplete")

    source_tree = hardware.get("source_tree")
    source_head = source_tree.get("head") if isinstance(source_tree, dict) else None
    if (
        not isinstance(source_tree, dict)
        or source_tree.get("path") != sglang_root
        or source_tree.get("is_git_checkout") is not True
        or source_tree.get("root_matches_toplevel") is not True
        or not isinstance(source_head, str)
        or len(source_head) != 40
        or any(char not in "0123456789abcdef" for char in source_head)
        or source_tree.get("tree") != PINNED_SGLANG_TREE
        or source_tree.get("dirty") is not False
        or source_tree.get("pinned_ancestor") is not True
        or source_tree.get("patch_commits") != PINNED_SGLANG_PATCH_COUNT
    ):
        reject("patched SGLang source-tree identity is not exact")
    if (
        compatibility.get("sglang_commit") != PINNED_SGLANG_COMMIT
        or compatibility.get("sglang_tree") != PINNED_SGLANG_TREE
        or compatibility.get("patch_count") != PINNED_SGLANG_PATCH_COUNT
        or compatibility.get("python_supported") is not True
        or compatibility.get("single_node_only") is not True
        or compatibility.get("multi_node_supported") is not False
    ):
        reject("compatibility identities do not match the release")

    commands = hardware.get("commands")
    nvidia = commands.get("nvidia_smi") if isinstance(commands, dict) else None
    gpu = hardware.get("gpu")
    if (
        not isinstance(nvidia, str)
        or not nvidia.strip()
        or not isinstance(gpu, dict)
        or gpu.get("gpu_pool_visible") is not True
        or not isinstance(gpu.get("visible_gpu_count"), int)
        or isinstance(gpu.get("visible_gpu_count"), bool)
        or gpu["visible_gpu_count"] < 1
    ):
        reject("successful same-host GPU-pool nvidia-smi evidence is required")
    return hardware


def _attest(args: argparse.Namespace) -> int:
    """Categorically refuse to turn legacy diagnostics into attestation."""

    manifest = PreliminarySpeedStudyManifest.load(args.manifest)
    decision = {
        "schema_version": 1,
        "kind": "preliminary_speed_study_attestation_decision",
        "status": PRELIMINARY_DIAGNOSTIC_ONLY,
        "reason_code": "legacy_speed_study_has_no_formal_attestation_path",
        "manifest_kind": manifest.kind,
        "manifest_sha256": manifest.sha256,
        "formal_execution_authorized": False,
        "industrial_evidence_receipt": None,
    }
    _write_json(args.output, decision)
    print(_canonical_sha256(decision))
    return 42


def _onlinespec_table(
    path: str | Path,
    *,
    manifest: OnlineSpecManifest,
    selection: OnlineSpecSelection,
    lock: ModelLock,
    target_reference: GreedyTargetReference,
) -> pa.Table:
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    expected = {
        b"lightcone_schema_version": b"2",
        b"lightcone_study": b"onlinespec-clean-room-baseline",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": selection.sampling_profile_sha256.encode(),
        b"lightcone_execution_policy_sha256": (
            manifest.execution_policy_sha256.encode()
        ),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_target_reference_sha256": target_reference.sha256.encode(),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("OnlineSPEC table identity metadata mismatch")
    scope_metadata = {
        b"lightcone_manifest_kind": manifest.kind.encode(),
        b"lightcone_evidence_scope": manifest.evidence_scope.encode(),
        b"lightcone_formal_execution_authorized": b"false",
    }
    present_scope_fields = set(scope_metadata) & set(metadata)
    if present_scope_fields and any(
        metadata.get(key) != value for key, value in scope_metadata.items()
    ):
        raise ValueError("OnlineSPEC table scope metadata is invalid")
    return table


def _attest_onlinespec(args: argparse.Namespace) -> int:
    """Refuse to mint claim authority for the separate comparison workflow."""

    manifest = OnlineSpecManifest.load(args.manifest)
    decision = {
        "schema_version": 1,
        "kind": "onlinespec_diagnostic_attestation_decision",
        "status": PRELIMINARY_DIAGNOSTIC_ONLY,
        "reason_code": "onlinespec_comparison_has_no_formal_attestation_path",
        "manifest_sha256": manifest.sha256,
        "formal_execution_authorized": False,
        "core_speed_gate_affected": False,
    }
    _write_json(args.output, decision)
    print(_canonical_sha256(decision))
    return 42


def _preliminary_method_statistics(value: object) -> dict[str, object]:
    """Copy only non-authoritative method diagnostics from a reducer row."""

    raw = value if isinstance(value, dict) else asdict(value)
    fields = (
        "method",
        "mean_speedup",
        "ci_lower",
        "ci_upper",
        "safety_pass",
        "acceleration_pass",
    )
    if not isinstance(raw, dict) or any(field not in raw for field in fields):
        raise TypeError("preliminary method statistics are incomplete")
    return {field: raw[field] for field in fields}


def _preliminary_gate_statistics(gate: object) -> dict[str, object]:
    """Whitelist diagnostic fields and categorically erase claim authority."""

    raw = asdict(gate)
    pair_fields = (
        "numerator_method",
        "denominator_method",
        "mean_speedup",
        "ci_lower",
        "ci_upper",
        "no_worse_pass",
    )
    pair = raw.get("l0_vs_tts")
    if not isinstance(pair, dict) or any(field not in pair for field in pair_fields):
        raise TypeError("preliminary pairwise statistics are incomplete")
    return {
        "status": PRELIMINARY_DIAGNOSTIC_ONLY,
        "evidence_classification": HISTORICAL_EVIDENCE_CLASSIFICATION,
        "gpu_evidence": PRELIMINARY_DIAGNOSTIC_ONLY,
        "evidence_sha256": None,
        "tts": _preliminary_method_statistics(raw.get("tts")),
        "l0": _preliminary_method_statistics(raw.get("l0")),
        "l0_vs_tts": {field: pair[field] for field in pair_fields},
    }


def _analyze(args: argparse.Namespace) -> int:
    manifest = PreliminarySpeedStudyManifest.load(args.manifest)
    if getattr(args, "attestation", None):
        raise ValueError(
            "preliminary analysis cannot consume any attestation; use "
            "analyze-industrial with raw industrial authority"
        )
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to a different model lock")
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=model_lock,
        sampling_profile_sha256=manifest.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    table = _load_preliminary_table(
        args.performance,
        manifest=manifest,
        selection=selection,
        model_lock=model_lock,
        target_reference=target_reference,
    )
    gate = evaluate_speed_gate(
        table.to_pylist(),
        seed=args.bootstrap_seed,
        gpu_evidence="UNMEASURED",
        evidence_sha256=None,
    )
    diagnostic_statistics = _preliminary_gate_statistics(gate)
    _write_json(
        args.output,
        {
            "schema_version": 1,
            "kind": "preliminary_speed_study_analysis",
            "status": PRELIMINARY_DIAGNOSTIC_ONLY,
            "manifest_kind": manifest.kind,
            "manifest_sha256": manifest.sha256,
            "formal_execution_authorized": False,
            "evidence_classification": HISTORICAL_EVIDENCE_CLASSIFICATION,
            "industrial_evidence_receipt": None,
            "diagnostic_statistics": diagnostic_statistics,
            "selection_protocol": selection.selection_protocol,
            "optimized_grid_claim": (
                selection.selection_protocol == "successive_halving"
            ),
        },
    )
    return 42


def _analyze_onlinespec(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    if getattr(args, "attestation", None):
        raise ValueError(
            "OnlineSPEC comparison analysis cannot consume attestation or enter "
            "the core industrial gate"
        )
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    _assert_onlinespec_study(manifest, selection, lock)
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=lock,
        sampling_profile_sha256=selection.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    table = _onlinespec_table(
        args.performance,
        manifest=manifest,
        selection=selection,
        lock=lock,
        target_reference=target_reference,
    )
    comparisons = compare_onlinespec(table.to_pylist(), seed=args.bootstrap_seed)
    safety_pass = all(comparison.safety_pass for comparison in comparisons)
    acceleration_pass = any(comparison.acceleration_pass for comparison in comparisons)
    _write_json(
        args.output,
        {
            "schema_version": 2,
            "study": "onlinespec-clean-room-baseline",
            "gpu_evidence": PRELIMINARY_DIAGNOSTIC_ONLY,
            "status": PRELIMINARY_DIAGNOSTIC_ONLY,
            "attestation_sha256": None,
            "formal_execution_authorized": False,
            "industrial_evidence_receipt": None,
            "core_speed_gate_affected": False,
            "safety_pass": safety_pass,
            "at_least_one_acceleration_pass": acceleration_pass,
            "passing_methods": [
                comparison.method for comparison in comparisons if comparison.passed
            ],
            "selection_protocol": selection.selection_protocol,
            "optimized_grid_claim": (
                selection.selection_protocol == "successive_halving"
            ),
            "comparisons": [_preliminary_method_statistics(row) for row in comparisons],
        },
    )
    return 42


def _formal_request(
    value: object,
    *,
    kind: str,
    fields: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError("formal materialization request must be a JSON object")
    if value.get("kind") != kind or set(value) != fields | {"kind"}:
        raise ValueError(f"{kind} materialization request fields differ from schema")
    return dict(value)


def _formal_single_control_lineage_sha256(
    *,
    signed_artifact_sha256: str,
    protocol_lock_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_single_control_lineage",
            "signed_artifact_sha256": signed_artifact_sha256,
            "protocol_lock_sha256": protocol_lock_sha256,
            "registry_sha256": build_industrial_registry().sha256,
        }
    )


def _candidate_dynamic_formal_policy(
    control: ControlArtifactAttestation,
):
    authorization = control.deployment_policy_authorization
    policy = authorization.bundle.trusted_attester_policy
    if (
        control.trust_bundle_sha256 != authorization.bundle.sha256
        or control.trusted_attester_policy_sha256 != policy.sha256
    ):
        raise ValueError("formal control differs from its deployment policy")
    return policy, policy.sha256


def _reserve_single_formal_control(
    control: ControlArtifactAttestation,
    *,
    signed_artifact_sha256: str,
    inner_challenge_sha256: str,
    protocol_lock_sha256: str,
    expected_artifact_type: str,
    inventory_sha256: str,
    replay_store_path: str,
    now_ns: int,
) -> str:
    registry_sha256 = build_industrial_registry().sha256
    subject = control.subject
    if (
        subject.artifact_type != expected_artifact_type
        or subject.artifact_sha256 != signed_artifact_sha256
        or subject.protocol_sha256 != protocol_lock_sha256
        or subject.registry_sha256 != registry_sha256
        or subject.lineage_sha256
        != _formal_single_control_lineage_sha256(
            signed_artifact_sha256=signed_artifact_sha256,
            protocol_lock_sha256=protocol_lock_sha256,
        )
    ):
        raise ValueError("formal single-control subject differs from signed authority")
    verified = verify_and_reserve_release_control_artifact_attestations(
        (control,),
        expected_inventory_sha256=inventory_sha256,
        now_ns=now_ns,
        replay_store=ChallengeReplayStore(replay_store_path),
        additional_challenge_sha256s=(inner_challenge_sha256,),
    )
    if verified[0].artifact_sha256 != signed_artifact_sha256:
        raise RuntimeError("verified formal control differs from signed authority")
    return control_challenge_reservation_sha256(
        verified,
        additional_challenge_sha256s=(inner_challenge_sha256,),
        reserved_ns=now_ns,
    )


def _verify_single_formal_control_diagnostic(
    control: ControlArtifactAttestation,
    *,
    signed_artifact_sha256: str,
    protocol_lock_sha256: str,
    expected_artifact_type: str,
    inventory_sha256: str,
    now_ns: int,
):
    """Verify one control without consuming its formal execution challenge."""

    registry_sha256 = build_industrial_registry().sha256
    subject = control.subject
    if (
        subject.artifact_type != expected_artifact_type
        or subject.artifact_sha256 != signed_artifact_sha256
        or subject.protocol_sha256 != protocol_lock_sha256
        or subject.registry_sha256 != registry_sha256
        or subject.lineage_sha256
        != _formal_single_control_lineage_sha256(
            signed_artifact_sha256=signed_artifact_sha256,
            protocol_lock_sha256=protocol_lock_sha256,
        )
    ):
        raise ValueError("formal single-control subject differs from signed authority")
    verified = verify_release_control_artifact_attestation(
        control,
        expected_inventory_sha256=inventory_sha256,
        now_ns=now_ns,
        consumed_challenge_sha256s=(),
    )
    if verified.artifact_sha256 != signed_artifact_sha256:
        raise RuntimeError("verified formal control differs from signed authority")
    return verified


def _formal_clean_git_identity(project_root: str | Path) -> tuple[Path, str, str]:
    root = Path(os.path.abspath(os.fspath(project_root)))
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("formal project root must be an existing resolved directory")

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ValueError("formal project root is not a readable Git checkout")
        return completed.stdout.strip()

    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("formal ProtocolLock requires a clean Git worktree")
    head = git("rev-parse", "--verify", "HEAD")
    tree = git("rev-parse", "--verify", "HEAD^{tree}")
    for label, value in (("HEAD", head), ("tree", tree)):
        if len(value) != 40 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"formal Git {label} is not an exact object ID")
    return root, head, tree


def _formal_source_file_sha256(
    project_root: Path,
    source_path: str | Path,
    *,
    label: str,
) -> str:
    source = Path(os.path.abspath(os.fspath(source_path)))
    if source.is_symlink() or source.resolve() != source:
        raise ValueError(f"formal {label} source path must be resolved")
    try:
        source.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            f"formal {label} source must be inside project root"
        ) from error
    return hashlib.sha256(_read_regular_bytes(source, label=label)).hexdigest()


def _require_expected_identity(label: str, expected: str | None, observed: str) -> None:
    if expected is not None and expected != observed:
        raise ValueError(f"formal {label} differs from reopened source identity")


def _publish_tts_calibration_source_authority(args: argparse.Namespace) -> int:
    artifact = build_source_tts_calibration_authority_artifact(
        paper_pdf_path=Path(os.path.abspath(args.paper_pdf)),
        paper_source_path=Path(os.path.abspath(args.paper_source)),
        tuning_window_path=Path(os.path.abspath(args.tuning_window)),
        trainable_plan_authority_path=Path(
            os.path.abspath(args.trainable_plan_authority)
        ),
        drafter_native_loss_path=Path(os.path.abspath(args.drafter_native_loss)),
    )
    publish_tts_calibration_authority_artifact(
        artifact,
        Path(os.path.abspath(args.output)),
    )
    print(artifact.authority.sha256)
    return 0


def _publish_chronobelief_source_authority(args: argparse.Namespace) -> int:
    artifact = build_source_chronobelief_authority_artifact(
        paper_pdf_path=Path(os.path.abspath(args.paper_pdf)),
        tex_source_path=Path(os.path.abspath(args.tex_source)),
    )
    publish_chronobelief_authority_artifact(
        artifact,
        Path(os.path.abspath(args.output)),
    )
    print(artifact.authority.sha256)
    return 0


def _publish_e1_recipe_anchor_authority(args: argparse.Namespace) -> int:
    artifact = build_source_e1_recipe_anchor_authority_artifact(
        Path(os.path.abspath(args.trainable_plan_authority))
    )
    publish_e1_recipe_anchor_authority_artifact(
        artifact,
        Path(os.path.abspath(args.output)),
    )
    print(artifact.authority.sha256)
    return 0


def _publish_formal_protocol_lock_git_snapshot(args: argparse.Namespace) -> int:
    binding = publish_formal_protocol_lock_git_snapshot(
        project_root=Path(os.path.abspath(args.project_root)),
        chunk_output_directory=Path(os.path.abspath(args.chunk_output_directory)),
        index_output_path=Path(os.path.abspath(args.output)),
    )
    print(binding.semantic_sha256)
    return 0


def _publish_formal_protocol_lock_source_proof(args: argparse.Namespace) -> int:
    artifact = bind_formal_protocol_lock_source_proof_artifact(
        protocol_id=args.protocol_id,
        git_snapshot_path=Path(os.path.abspath(args.git_snapshot)),
        patch_manifest_relative_path=args.patch_manifest_relative_path,
        english_protocol_relative_path=args.english_protocol_relative_path,
        chinese_protocol_relative_path=args.chinese_protocol_relative_path,
        runtime_authority_path=Path(
            os.path.abspath(args.formal_runtime_authority_manifest)
        ),
        tts_calibration_authority_path=Path(
            os.path.abspath(args.tts_calibration_authority)
        ),
        chronobelief_authority_path=Path(os.path.abspath(args.chronobelief_authority)),
        e1_recipe_anchor_authority_path=Path(
            os.path.abspath(args.e1_recipe_anchor_authority)
        ),
        content_verification_receipt_path=Path(
            os.path.abspath(args.content_verification_receipt)
        ),
        burstgpt_shape_authority_path=Path(
            os.path.abspath(args.burstgpt_shape_authority)
        ),
        now_ns=args.now_ns,
    )
    binding = publish_formal_protocol_lock_source_proof_artifact(
        artifact,
        Path(os.path.abspath(args.output)),
    )
    lock = revalidate_formal_protocol_lock_source_proof_artifact(
        binding.absolute_path,
        now_ns=args.now_ns,
    )
    print(lock.sha256)
    return 0


def _create_protocol_lock(args: argparse.Namespace) -> int:
    project_root, code_git_head, code_git_tree = _formal_clean_git_identity(
        args.project_root
    )
    patch_manifest_sha256 = _formal_source_file_sha256(
        project_root, args.patch_manifest, label="patch manifest"
    )
    english_protocol_sha256 = _formal_source_file_sha256(
        project_root, args.english_protocol, label="English protocol"
    )
    chinese_protocol_sha256 = _formal_source_file_sha256(
        project_root, args.chinese_protocol, label="Chinese protocol"
    )
    runtime_manifest_path = Path(
        os.path.abspath(args.formal_runtime_authority_manifest)
    )
    try:
        runtime_manifest_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            "formal runtime authority manifest must be inside project root"
        ) from error
    runtime_binding_before = CanonicalJsonProofBinding.bind(str(runtime_manifest_path))
    runtime_authority_manifest = formal_runtime_authority_manifest_from_dict(
        runtime_binding_before.reopen()
    )
    source_runtime_authority_manifest = build_source_formal_runtime_authority_manifest(
        project_root
    )
    if runtime_authority_manifest != source_runtime_authority_manifest:
        raise ValueError(
            "formal runtime authority manifest differs from source-owned rebuild"
        )
    if CanonicalJsonProofBinding.bind(str(runtime_manifest_path)) != (
        runtime_binding_before
    ):
        raise RuntimeError(
            "formal runtime authority changed while ProtocolLock was built"
        )
    tts_artifact = load_tts_calibration_authority_artifact(
        Path(os.path.abspath(args.tts_calibration_authority))
    )
    chronobelief_artifact = load_chronobelief_authority_artifact(
        Path(os.path.abspath(args.chronobelief_authority))
    )
    e1_anchor_artifact = load_e1_recipe_anchor_authority_artifact(
        Path(os.path.abspath(args.e1_recipe_anchor_authority))
    )
    e2_grid = default_e2_recipe_grid_authority()
    qualification_identities = code_owned_qualification_source_identities()
    native_qualification = qualification_identities["native_runtime"]
    compile_qualification = qualification_identities["compile"]
    exactness_qualification = qualification_identities["exactness"]
    content_receipt, verified_content = _load_content_verification_receipt(
        args.content_verification_receipt,
        now_ns=args.content_verification_now_ns,
    )
    verified_content = content_receipt.revalidate_formal_scope(
        current_ns=args.content_verification_now_ns
    )
    prepared_content = tuple(
        row
        for row in verified_content
        if type(row) is VerifiedPreparedModelContentRelease
    )
    workload_content = tuple(
        row for row in verified_content if type(row) is VerifiedReleaseWorkloadSources
    )
    e0_content = tuple(
        row
        for row in verified_content
        if type(row) is VerifiedDatasetContentRelease
        and row.authority_domain == "e0_task_native"
    )
    if len(prepared_content) != 1 or len(workload_content) != 1 or len(e0_content) != 1:
        raise ValueError("ProtocolLock content master lacks exact formal authorities")
    root_manifest_sha256s = {
        row.authorization.root_manifest_sha256 for row in verified_content
    }
    if len(root_manifest_sha256s) != 1:
        raise ValueError("ProtocolLock content authorities use different roots")
    from lightcone_spec.runtime.preflight_runner import (
        BurstGptShapeAuthority,
        derive_burstgpt_shape_authority_from_content_receipt,
    )

    burstgpt_shape_authority = BurstGptShapeAuthority.from_dict(
        CanonicalJsonProofBinding.bind(args.burstgpt_shape_authority).reopen()
    )
    derived_burstgpt_shape = derive_burstgpt_shape_authority_from_content_receipt(
        content_receipt,
        current_ns=args.content_verification_now_ns,
    )
    if burstgpt_shape_authority != derived_burstgpt_shape:
        raise ValueError("ProtocolLock BurstGPT shape differs from signed content")
    registry_sha256 = build_industrial_registry().sha256
    for label, expected, observed in (
        ("Git HEAD", args.code_git_head, code_git_head),
        ("Git tree", args.code_git_tree, code_git_tree),
        ("patch manifest", args.patch_manifest_sha256, patch_manifest_sha256),
        ("registry", args.registry_sha256, registry_sha256),
        (
            "English protocol",
            args.english_protocol_sha256,
            english_protocol_sha256,
        ),
        (
            "Chinese protocol",
            args.chinese_protocol_sha256,
            chinese_protocol_sha256,
        ),
    ):
        _require_expected_identity(label, expected, observed)
    root_after, head_after, tree_after = _formal_clean_git_identity(project_root)
    if (
        root_after != project_root
        or head_after != code_git_head
        or tree_after != code_git_tree
    ):
        raise RuntimeError(
            "formal source checkout changed while ProtocolLock was built"
        )
    if (
        load_tts_calibration_authority_artifact(
            Path(os.path.abspath(args.tts_calibration_authority))
        )
        != tts_artifact
        or load_chronobelief_authority_artifact(
            Path(os.path.abspath(args.chronobelief_authority))
        )
        != chronobelief_artifact
        or load_e1_recipe_anchor_authority_artifact(
            Path(os.path.abspath(args.e1_recipe_anchor_authority))
        )
        != e1_anchor_artifact
    ):
        raise RuntimeError("formal method authority changed while lock was built")
    lock = ProtocolLock(
        schema_version=4,
        protocol_id=args.protocol_id,
        code_git_head=code_git_head,
        code_git_tree=code_git_tree,
        patch_manifest_sha256=patch_manifest_sha256,
        registry_sha256=registry_sha256,
        english_protocol_sha256=english_protocol_sha256,
        chinese_protocol_sha256=chinese_protocol_sha256,
        tts_calibration_authority_sha256=tts_artifact.authority.sha256,
        chronobelief_authority_sha256=chronobelief_artifact.authority.sha256,
        e1_recipe_anchor_authority_sha256=e1_anchor_artifact.authority.sha256,
        e2_recipe_grid_authority_sha256=e2_grid.sha256,
        formal_runtime_authority_manifest_sha256=(runtime_authority_manifest.sha256),
        offline_release_trust_root_sha256=next(iter(root_manifest_sha256s)),
        prepared_model_content_authorization_sha256=(
            prepared_content[0].authorization_sha256
        ),
        formal_workload_e3a_authorization_sha256=(
            workload_content[0].authorization_sha256
        ),
        formal_workload_e0_authorization_sha256=(e0_content[0].authorization_sha256),
        burstgpt_shape_authorization_sha256=burstgpt_shape_authority.sha256,
        native_runtime_qualification_protocol_sha256=native_qualification[0],
        native_runtime_qualification_runner_sha256=native_qualification[1],
        native_runtime_qualification_test_set_sha256=native_qualification[2],
        compile_qualification_protocol_sha256=compile_qualification[0],
        compile_qualification_runner_sha256=compile_qualification[1],
        compile_qualification_test_set_sha256=compile_qualification[2],
        exactness_qualification_protocol_sha256=exactness_qualification[0],
        exactness_qualification_runner_sha256=exactness_qualification[1],
        exactness_qualification_test_set_sha256=exactness_qualification[2],
    )
    _write_json(
        args.output,
        {
            "schema_version": 1,
            "kind": "unsigned_protocol_lock_payload",
            "payload": protocol_lock_to_dict(lock),
            "payload_sha256": lock.sha256,
            "formal_dispatch_authorized": False,
        },
    )
    print(lock.sha256)
    return 0


def _verify_signed_protocol_lock(args: argparse.Namespace) -> int:
    signed = signed_protocol_lock_from_dict(_load_bound_json(args.signed_lock))
    control = ControlArtifactAttestation.from_dict(
        _load_bound_json(args.control_attestation)
    )
    policy, policy_sha256 = _candidate_dynamic_formal_policy(control)
    payload = signed.verify(
        policy=policy,
        expected_policy_sha256=policy_sha256,
        now_ns=args.now_ns,
    )
    if payload.registry_sha256 != build_industrial_registry().sha256:
        raise ValueError("ProtocolLock does not bind the staged registry")
    verified_control = _verify_single_formal_control_diagnostic(
        control,
        signed_artifact_sha256=signed.sha256,
        protocol_lock_sha256=payload.sha256,
        expected_artifact_type="dispatch",
        inventory_sha256=args.inventory_sha256,
        now_ns=args.now_ns,
    )
    _write_json(
        args.output,
        {
            "schema_version": 1,
            "kind": "verified_protocol_lock_receipt",
            "payload_sha256": payload.sha256,
            "signed_protocol_lock_sha256": signed.sha256,
            "trusted_attester_policy_sha256": policy_sha256,
            "control_envelope_sha256": verified_control.envelope_sha256,
            "challenge_reservation_sha256": None,
            "verification_mode": "diagnostic_only_non_consuming",
            "formal_dispatch_authorized": False,
        },
    )
    print(payload.sha256)
    return 0


def _create_gpu_hour_envelope(args: argparse.Namespace) -> int:
    estimate = GpuHourEstimate.unmeasured()
    _write_json(args.output, gpu_hour_estimate_to_dict(estimate))
    print(_canonical_sha256(gpu_hour_estimate_to_dict(estimate)))
    return 0


def _reduce_stage_gpu_hour_envelope(args: argparse.Namespace) -> int:
    signed = signed_pilot_duration_from_dict(
        _load_bound_json(args.signed_pilot_receipt)
    )
    control = ControlArtifactAttestation.from_dict(
        _load_bound_json(args.control_attestation)
    )
    policy, policy_sha256 = _candidate_dynamic_formal_policy(control)
    try:
        envelope = reduce_stage_gpu_hour_envelope_from_signed_pilots(
            signed,
            policy=policy,
            expected_policy_sha256=policy_sha256,
            protocol_lock_sha256=args.protocol_lock_sha256,
            materialization_receipt_sha256=args.materialization_receipt_sha256,
            schedule_sha256=args.schedule_sha256,
            now_ns=args.now_ns,
        )
    except FormalGpuHourAuthorityBlocked as error:
        _write_json(
            args.output,
            {
                "schema_version": 1,
                "kind": "formal_stage_gpu_hour_envelope_blocked",
                "status": "BLOCKED",
                "reason_code": str(error),
                "formal_dispatch_authorized": False,
            },
        )
        print(str(error))
        return 42
    reservation_sha256 = _reserve_single_formal_control(
        control,
        signed_artifact_sha256=signed.sha256,
        inner_challenge_sha256=signed.challenge.sha256,
        protocol_lock_sha256=args.protocol_lock_sha256,
        expected_artifact_type="capacity",
        inventory_sha256=args.inventory_sha256,
        replay_store_path=args.control_replay_store,
        now_ns=args.now_ns,
    )
    _write_json(
        args.output,
        {
            **stage_gpu_hour_envelope_to_dict(envelope),
            "challenge_reservation_sha256": reservation_sha256,
            "formal_dispatch_authorized": False,
        },
    )
    print(envelope.sha256)
    return 0


def _parse_preflight_gpu_hour_lifecycle_proofs(
    values: list[str],
) -> tuple[object, ...]:
    """Decode exactly eight unique CELL_ID=PATH interference proof joins."""

    from lightcone_spec.experiments.gpu_hour_authority import (
        PreflightGpuHourLifecycleProofInput,
    )

    parsed: dict[str, str] = {}
    for value in values:
        cell_id, separator, path_text = value.partition("=")
        path = Path(path_text)
        if (
            separator != "="
            or not cell_id
            or not path_text
            or cell_id in parsed
            or not path.is_absolute()
            or path != path.resolve(strict=False)
        ):
            raise ValueError(
                "preflight GPU-hour lifecycle proof must be unique CELL_ID=PATH"
            )
        parsed[cell_id] = str(path)
    if len(parsed) != 8 or len(set(parsed.values())) != 8:
        raise ValueError(
            "preflight GPU-hour lifecycle proofs must cover exact eight cells"
        )
    return tuple(
        PreflightGpuHourLifecycleProofInput(
            materialized_cell_id=cell_id,
            lifecycle_proof_artifact_path=path,
        )
        for cell_id, path in sorted(parsed.items())
    )


def _materialize_preflight_gpu_hour_envelope_cli(args: argparse.Namespace) -> int:
    """Deep-reopen typed 1+1+8 timing and publish a signable envelope."""

    from lightcone_spec.experiments.formal_dispatch import (
        FormalPreflightDispatchReceipt,
    )
    from lightcone_spec.experiments.formal_preflight_execution import (
        FormalPreflightFinalEvidence,
        FormalPreflightRemoteRawEvidenceReceipt,
    )
    from lightcone_spec.experiments.gpu_hour_authority import (
        materialize_preflight_gpu_hour_envelope,
    )
    from lightcone_spec.experiments.preflight_authority import (
        PreflightCoverageReceipt,
        PreflightExecutionSourceAuthority,
    )
    from lightcone_spec.runtime.proof_artifact import (
        CanonicalJsonProofBinding,
        publish_canonical_json_no_replace,
    )

    source_output = Path(args.source_output)
    envelope_output = Path(args.output)
    if (
        not source_output.is_absolute()
        or source_output != source_output.resolve(strict=False)
        or not envelope_output.is_absolute()
        or envelope_output != envelope_output.resolve(strict=False)
        or source_output == envelope_output
    ):
        raise ValueError(
            "preflight GPU-hour outputs must be distinct absolute normalized paths"
        )
    lifecycle_proofs = _parse_preflight_gpu_hour_lifecycle_proofs(
        args.interference_lifecycle_proof
    )
    dispatch_binding = CanonicalJsonProofBinding.bind(args.dispatch_receipt)
    durable_dispatch = FormalPreflightDispatchReceipt.from_dict(
        dispatch_binding.reopen()
    )
    token = durable_dispatch.revalidate(current_ns=args.now_ns)
    if CanonicalJsonProofBinding.bind(args.dispatch_receipt) != dispatch_binding:
        raise ValueError("preflight GPU-hour dispatch receipt changed")
    runtime_binding = CanonicalJsonProofBinding.bind(
        args.formal_runtime_authority_manifest
    )
    runtime_manifest = formal_runtime_authority_manifest_from_dict(
        runtime_binding.reopen()
    )
    remote_raw_receipt = CanonicalJsonProofBinding.bind(args.remote_raw_receipt)
    remote_raw_payload = remote_raw_receipt.reopen()
    remote_raw = FormalPreflightRemoteRawEvidenceReceipt.from_dict(remote_raw_payload)
    remote_raw.revalidate(token)
    if remote_raw_receipt.semantic_sha256 != remote_raw.sha256:
        raise ValueError("preflight GPU-hour raw receipt identity differs")
    evidence = FormalPreflightFinalEvidence(
        remote_raw_receipt=remote_raw_receipt,
        source_authority=PreflightExecutionSourceAuthority.from_dict(
            CanonicalJsonProofBinding.bind(args.source_authority).reopen()
        ),
        activation=registry_stage_activation_from_dict(
            CanonicalJsonProofBinding.bind(args.activation).reopen()
        ),
        coverage=PreflightCoverageReceipt.from_dict(
            CanonicalJsonProofBinding.bind(args.coverage).reopen()
        ),
        materialization=durable_dispatch.signed_materialization.payload,
        stage_coverage=stage_coverage_receipt_from_dict(
            CanonicalJsonProofBinding.bind(args.stage_coverage).reopen()
        ),
    )
    if (
        token.protocol_lock
        != durable_dispatch.registry_verification_receipt.signed_protocol_lock.payload
        or evidence.materialization != durable_dispatch.signed_materialization.payload
        or evidence.activation != durable_dispatch.activation
        or evidence.source_authority.inventory_sha256
        != token.dispatch_context.inventory.sha256
    ):
        raise ValueError("preflight GPU-hour sealed dispatch identity differs")
    envelope = materialize_preflight_gpu_hour_envelope(
        protocol_lock=token.protocol_lock,
        formal_runtime_authority_manifest=runtime_manifest,
        final_evidence=evidence,
        inventory=token.dispatch_context.inventory,
        interference_lifecycle_proof_inputs=lifecycle_proofs,
        source_manifest_output_path=str(source_output),
        now_ns=args.now_ns,
    )
    publish_canonical_json_no_replace(
        envelope_output,
        stage_gpu_hour_envelope_to_dict(envelope),
    )
    reopened = stage_gpu_hour_envelope_from_dict(
        CanonicalJsonProofBinding.bind(envelope_output).reopen()
    )
    if reopened != envelope:
        raise RuntimeError("preflight GPU-hour envelope changed during publication")
    print(envelope.sha256)
    return 0


def _reserve_formal_stage_gpu_hours(args: argparse.Namespace) -> int:
    """Deep-reopen lifecycle sources and consume their stage budget once."""

    receipt = reserve_formal_stage_gpu_hour_verification_receipt(
        registry_receipt=_load_formal_registry_receipt_path(
            args.registry_verification_receipt,
            now_ns=args.now_ns,
        ),
        registry_receipt_path=args.registry_verification_receipt,
        signed_envelope=signed_stage_gpu_hour_from_dict(
            _load_bound_json(args.signed_envelope)
        ),
        source_manifest_path=args.source_manifest,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest_from_dict(
            _load_bound_json(args.formal_runtime_authority_manifest)
        ),
        inventory=_load_gpu_inventory(args.inventory),
        control_attestation=ControlArtifactAttestation.from_dict(
            _load_bound_json(args.control_attestation)
        ),
        replay_store=ChallengeReplayStore(args.control_replay_store),
        now_ns=args.now_ns,
        prospective_pilot_materialization_path=(args.prospective_pilot_materialization),
    )
    _write_json(args.output, receipt.to_dict())
    print(receipt.sha256)
    return 0


def _publish_formal_stage_gpu_hour_envelope_proof(
    args: argparse.Namespace,
) -> int:
    """Publish a deep-replayed source proof for one signable envelope."""

    artifact = bind_formal_stage_gpu_hour_envelope_proof_artifact(
        protocol_lock_path=args.protocol_lock,
        runtime_authority_path=args.formal_runtime_authority_manifest,
        registry_layer_path=args.registry_layer,
        inventory_path=args.inventory,
        final_materialization_path=args.final_materialization,
        pilot_materialization_path=args.pilot_materialization,
        gpu_hour_source_manifest_path=args.gpu_hour_source_manifest,
        envelope_path=args.envelope,
        preflight_coverage_proof_path=args.preflight_coverage_proof,
        now_ns=args.now_ns,
    )
    publish_formal_stage_gpu_hour_envelope_proof_artifact(
        artifact,
        args.output,
    )
    print(artifact.sha256)
    return 0


def _publish_formal_initial_stage_materialization_proof(
    args: argparse.Namespace,
) -> int:
    """Publish a replayable preflight/E3a/TTS-Cal/E1 materializer proof."""

    artifact = bind_formal_initial_stage_materialization_proof_artifact(
        phase=args.phase,
        registry_layer_path=args.registry_layer,
        tts_calibration_authority_path=args.tts_calibration_authority,
        now_ns=args.now_ns,
    )
    publish_formal_initial_stage_materialization_proof_artifact(
        artifact,
        args.output,
    )
    print(artifact.sha256)
    return 0


def _publish_formal_downstream_materialization_proof(
    args: argparse.Namespace,
) -> int:
    """Publish a current-only post-E4 typed materializer proof."""

    artifact = build_formal_downstream_materialization_proof_artifact(
        phase=args.phase,
        registry_layer_path=args.registry_layer,
        immediate_predecessor_path=args.immediate_predecessor,
        now_ns=args.now_ns,
    )
    binding = publish_formal_downstream_materialization_proof_artifact(
        artifact,
        args.output,
        now_ns=args.now_ns,
    )
    rebuilt = rebuild_formal_downstream_materialization_proof(
        binding.absolute_path,
        now_ns=args.now_ns,
    )
    if rebuilt.artifact != artifact:
        raise RuntimeError("formal downstream proof changed after publication")
    print(binding.semantic_sha256)
    return 0


def _publish_formal_downstream_pilot_precoverage(args: argparse.Namespace) -> int:
    """Publish the proof-replayed signed pilot bridge before coverage."""

    artifact = build_formal_downstream_pilot_precoverage_artifact(
        phase=args.phase,
        materialization_proof_path=args.materialization_proof,
        signed_materialization_path=args.signed_materialization,
        now_ns=args.now_ns,
    )
    binding = publish_formal_downstream_pilot_precoverage_artifact(
        artifact,
        args.output,
        now_ns=args.now_ns,
    )
    rebuilt = rebuild_formal_downstream_pilot_precoverage(
        binding.absolute_path,
        now_ns=args.now_ns,
    )
    if rebuilt.artifact != artifact:
        raise RuntimeError("formal pilot precoverage changed after publication")
    print(binding.semantic_sha256)
    return 0


def _publish_formal_portable_stage_coverage(args: argparse.Namespace) -> int:
    """Publish one zero-caller portable stage-coverage proof graph."""

    artifact = bind_formal_portable_stage_coverage_proof_artifact(
        args.coverage_proof,
        registry_layer_path=args.registry_layer,
        prior_prefix_path=args.prior_prefix,
        e1_recipe_anchor_authority_path=args.e1_recipe_anchor_authority,
        downstream_pilot_precoverage_path=args.downstream_pilot_precoverage,
        now_ns=args.now_ns,
    )
    binding = publish_formal_portable_stage_coverage_proof_artifact(
        artifact,
        args.output,
        now_ns=args.now_ns,
    )
    rebuilt = revalidate_portable_formal_stage_coverage_proof_artifact(
        binding.absolute_path,
        now_ns=args.now_ns,
    )
    if rebuilt.coverage.sha256 != artifact.coverage_receipt_sha256:
        raise RuntimeError("portable stage coverage changed after publication")
    print(binding.semantic_sha256)
    return 0


def _publish_formal_downstream_reduction_proof(args: argparse.Namespace) -> int:
    """Publish one proof-derived post-E4 reducer root."""

    artifact = build_formal_downstream_reduction_proof_artifact(
        phase=args.phase,
        materialization_proof_path=args.materialization_proof,
        portable_coverage_proof_path=args.portable_coverage_proof,
        now_ns=args.now_ns,
    )
    binding = publish_formal_downstream_reduction_proof_artifact(
        artifact,
        args.output,
        now_ns=args.now_ns,
    )
    rebuilt = rebuild_formal_downstream_reduction_proof(
        binding.absolute_path,
        now_ns=args.now_ns,
    )
    if rebuilt.artifact != artifact:
        raise RuntimeError("formal downstream reduction changed after publication")
    print(binding.semantic_sha256)
    return 0


def _publish_formal_downstream_completed_prefix(args: argparse.Namespace) -> int:
    """Publish one signed, reducer-replayed downstream prefix node."""

    artifact = build_formal_downstream_completed_prefix_artifact(
        phase=args.phase,
        reduction_proof_path=args.reduction_proof,
        signed_result_path=args.signed_result,
        now_ns=args.now_ns,
    )
    binding = publish_formal_downstream_completed_prefix_artifact(
        artifact,
        args.output,
        now_ns=args.now_ns,
    )
    rebuilt = rebuild_formal_downstream_completed_prefix(
        binding.absolute_path,
        now_ns=args.now_ns,
    )
    if rebuilt.artifact != artifact:
        raise RuntimeError(
            "formal downstream completed prefix changed after publication"
        )
    print(binding.semantic_sha256)
    return 0


def _publish_formal_e3a_staged_selection_proof(
    args: argparse.Namespace,
) -> int:
    """Publish the proof-replayed exact 360-row E3a selection root."""

    artifact = bind_formal_e3a_staged_selection_proof_artifact(
        coverage_proof_path=args.coverage_proof,
        registry_layer_path=args.registry_layer,
        now_ns=args.now_ns,
    )
    publish_formal_e3a_staged_selection_proof_artifact(
        artifact,
        args.output,
    )
    print(artifact.sha256)
    return 0


def _materialize_prospective_stage_gpu_hours(args: argparse.Namespace) -> int:
    """Create a source-bound powered stage estimate without duration scalars."""

    registry_receipt = _load_formal_registry_receipt_path(
        args.registry_verification_receipt,
        now_ns=args.now_ns,
    )
    final_materializations = tuple(
        row.payload
        for row in registry_receipt.cumulative_signed_materializations
        if row.payload.stage == args.stage
    )
    if len(final_materializations) != 1:
        raise ValueError(
            "prospective GPU-hour CLI requires one exact registered final stage"
        )
    final_materialization = final_materializations[0]
    pilot_materialization = stage_materialization_receipt_from_dict(
        _load_bound_json(args.pilot_materialization)
    )
    authority = verify_registered_prospective_gpu_hour_authority(
        registry_receipt=registry_receipt,
        pilot_materialization=pilot_materialization,
        final_materialization=final_materialization,
        current_ns=args.now_ns,
    )
    _source, envelope = materialize_prospective_stage_gpu_hour_envelope(
        authority=authority,
        protocol_lock=registry_receipt.signed_protocol_lock.payload,
        formal_runtime_authority_manifest=(
            formal_runtime_authority_manifest_from_dict(
                _load_bound_json(args.formal_runtime_authority_manifest)
            )
        ),
        pilot_materialization=pilot_materialization,
        pilot_envelope=stage_gpu_hour_envelope_from_dict(
            _load_bound_json(args.pilot_envelope)
        ),
        pilot_source_manifest_path=args.pilot_source_manifest,
        final_materialization=final_materialization,
        inventory=_load_gpu_inventory(args.inventory),
        prospective_source_manifest_output_path=args.source_output,
        now_ns=args.now_ns,
        existing_one_shot_source_manifest_path=(args.one_shot_source_manifest),
    )
    _write_json(args.output, stage_gpu_hour_envelope_to_dict(envelope))
    print(envelope.sha256)
    return 0


def _materialize_staged_prospective_gpu_hours(args: argparse.Namespace) -> int:
    """Publish honest early-stage totals or a minimum-pilot BLOCKED report."""

    source_output = Path(args.source_output)
    result_output = Path(args.output)
    if (
        not source_output.is_absolute()
        or source_output != source_output.resolve(strict=False)
        or not result_output.is_absolute()
        or result_output != result_output.resolve(strict=False)
        or source_output == result_output
    ):
        raise ValueError(
            "staged prospective GPU-hour outputs must be distinct absolute paths"
        )
    registry_receipt = _load_formal_registry_receipt_path(
        args.registry_verification_receipt,
        now_ns=args.now_ns,
    )
    candidates = tuple(
        row.payload
        for row in registry_receipt.cumulative_signed_materializations
        if row.payload.stage == args.stage
        and row.payload.sha256 == args.materialization_sha256
    )
    if len(candidates) != 1:
        raise ValueError(
            "staged prospective GPU-hour CLI requires one exact registered "
            "materialization"
        )
    manifest, envelope = materialize_staged_prospective_gpu_hour_envelope(
        protocol_lock=registry_receipt.signed_protocol_lock.payload,
        formal_runtime_authority_manifest=(
            formal_runtime_authority_manifest_from_dict(
                _load_bound_json(args.formal_runtime_authority_manifest)
            )
        ),
        materialization=candidates[0],
        inventory=_load_gpu_inventory(args.inventory),
        completed_source_manifest_path=args.completed_source_manifest,
        source_manifest_output_path=str(source_output),
        now_ns=args.now_ns,
    )
    if envelope is None:
        _write_json(result_output, manifest.to_dict())
        print(manifest.sha256)
        return 42
    _write_json(result_output, stage_gpu_hour_envelope_to_dict(envelope))
    print(envelope.sha256)
    return 0


def _aggregate_formal_study_gpu_hours(args: argparse.Namespace) -> int:
    """Publish study totals only from durable, source-reopened stage receipts."""

    estimate = aggregate_formal_study_gpu_hours(
        registry_receipt=_load_formal_registry_receipt_path(
            args.registry_verification_receipt,
            now_ns=args.now_ns,
        ),
        stage_receipts=tuple(
            FormalStageGpuHourVerificationReceipt.from_dict(_load_bound_json(path))
            for path in args.stage_receipt
        ),
        current_ns=args.now_ns,
        require_complete=not args.allow_partial,
    )
    _write_json(args.output, estimate.to_dict())
    print(estimate.derivation_sha256)
    return 0 if estimate.status == "COMPLETE" else 42


def _materialize_formal_request(
    value: object,
    *,
    now_ns: int | None,
    nested_policy=None,
    nested_policy_sha256: str | None = None,
):
    if type(value) is not dict or type(value.get("kind")) is not str:
        raise TypeError("formal materialization request requires a kind")
    kind = value["kind"]
    common = {"protocol_lock_sha256", "gpu_hours"}
    if kind == "preflight":
        row = _formal_request(value, kind=kind, fields=frozenset(common))
        return materialize_preflight(
            protocol_lock_sha256=row["protocol_lock_sha256"],
            gpu_hours=gpu_hour_estimate_from_dict(row["gpu_hours"]),
        )
    if kind == "E3a":
        raise ValueError(
            "E3a JSON summaries are non-authorizing; formal E3a materialization "
            "requires the durable registry receipt and exact signed preflight "
            "coverage API"
        )
    if kind == "TTS-Cal":
        raise ValueError(
            "TTS-Cal JSON summaries are non-authorizing; formal TTS-Cal "
            "materialization requires the durable signed E3a six-output source "
            "and exact TTS authority API"
        )
    if kind == "E1":
        raise ValueError(
            "E1 JSON summaries are non-authorizing; formal E1 materialization "
            "requires the path-reopened signed E3a selection and signed TTS-Cal "
            "seal API"
        )
    if kind in {
        "E2",
        "E4-screen",
        "E4-local",
        "E4-profile",
        "E3b",
        "E1a",
        "E5",
        "E6",
        "E0",
    }:
        raise ValueError(
            f"{kind} JSON summaries are non-authorizing; formal materialization "
            "requires the stage-specific path-reopened signed selection, power, "
            "and upstream coverage authority API"
        )
    if kind == "E2":
        row = _formal_request(
            value,
            kind=kind,
            fields=frozenset(
                common
                | {
                    "upstream_receipt_sha256",
                    "source_selection_sha256",
                    "grid_authority_sha256",
                    "geometries",
                    "round_index",
                    "model",
                    "frozen_tts_recipe_sha256",
                    "candidate_recipes",
                    "prior_round_materialization",
                }
            ),
        )
        grid = default_e2_recipe_grid_authority()
        if row["grid_authority_sha256"] != grid.sha256:
            raise ValueError("E2 request differs from the registered recipe grid")
        if type(row["geometries"]) is not list:
            raise TypeError("E2 geometries must be a JSON array")
        geometries = tuple(E1Geometry(**item) for item in row["geometries"])
        raw_candidates = row["candidate_recipes"]
        if raw_candidates is None:
            candidates = None
        else:
            if type(raw_candidates) is not list:
                raise TypeError("E2 candidate recipes must be a JSON array")
            candidates = tuple(
                E2CandidateRecipe(
                    geometry=E1Geometry(**item["geometry"]),
                    optimizer=item["optimizer"],
                    schedule=item["schedule"],
                    learning_rate=item["learning_rate"],
                    optimizer_recipe_authority_sha256=item[
                        "optimizer_recipe_authority_sha256"
                    ],
                )
                for item in raw_candidates
            )
        raw_prior = row["prior_round_materialization"]
        prior = (
            None
            if raw_prior is None
            else stage_materialization_receipt_from_dict(raw_prior)
        )
        return materialize_e2_round(
            protocol_lock_sha256=row["protocol_lock_sha256"],
            upstream_receipt_sha256=row["upstream_receipt_sha256"],
            source_selection_sha256=row["source_selection_sha256"],
            grid=grid,
            geometries=geometries,
            round_index=row["round_index"],
            model=row["model"],
            frozen_tts_recipe_sha256=row["frozen_tts_recipe_sha256"],
            candidate_recipes=candidates,
            prior_round_materialization=prior,
            gpu_hours=gpu_hour_estimate_from_dict(row["gpu_hours"]),
        )
    if kind == "E4-screen":
        row = _formal_request(
            value,
            kind=kind,
            fields=frozenset(
                common
                | {
                    "upstream_e2_receipt_sha256",
                    "source_decision_sha256",
                    "model",
                    "lightcone_recipe_sha256",
                }
            ),
        )
        return materialize_e4_strength2_screen(
            **{name: row[name] for name in row if name not in {"kind", "gpu_hours"}},
            gpu_hours=gpu_hour_estimate_from_dict(row["gpu_hours"]),
        )
    if kind == "E4-local":
        row = _formal_request(
            value,
            kind=kind,
            fields=frozenset(
                common
                | {
                    "upstream_screen_receipt_sha256",
                    "winner_decision_sha256",
                    "model",
                    "lightcone_recipe_sha256",
                    "factor_neighborhoods",
                }
            ),
        )
        if type(row["factor_neighborhoods"]) is not list:
            raise TypeError("E4 factor neighborhoods must be a JSON array")
        neighborhoods = tuple(tuple(item) for item in row["factor_neighborhoods"])
        return materialize_e4_winner_neighborhood(
            protocol_lock_sha256=row["protocol_lock_sha256"],
            upstream_screen_receipt_sha256=row["upstream_screen_receipt_sha256"],
            winner_decision_sha256=row["winner_decision_sha256"],
            model=row["model"],
            lightcone_recipe_sha256=row["lightcone_recipe_sha256"],
            factor_neighborhoods=neighborhoods,
            gpu_hours=gpu_hour_estimate_from_dict(row["gpu_hours"]),
        )
    if kind == "E4-profile":
        row = _formal_request(
            value,
            kind=kind,
            fields=frozenset(
                common
                | {
                    "upstream_local_receipt_sha256",
                    "selected_configuration_sha256",
                    "model",
                    "lightcone_recipe_sha256",
                }
            ),
        )
        return materialize_e4_profiler(
            **{name: row[name] for name in row if name not in {"kind", "gpu_hours"}},
            gpu_hours=gpu_hour_estimate_from_dict(row["gpu_hours"]),
        )
    if kind in {"E3b", "E1a", "E5", "E6"}:
        fields_by_kind = {
            "E3b": {
                "upstream_receipt_sha256",
                "source_decision_sha256",
                "model",
                "frozen_tts_recipe_sha256",
                "lightcone_recipe_sha256",
                "final_blocks",
            },
            "E1a": {
                "upstream_receipt_sha256",
                "source_decision_sha256",
                "model",
                "lightcone_recipe_sha256",
            },
            "E5": {
                "upstream_e1a_receipt_sha256",
                "power_prefix_decision_sha256",
                "model",
                "frozen_tts_recipe_sha256",
                "lightcone_recipe_sha256",
                "final_blocks",
                "signed_anchor_selection",
            },
            "E6": {
                "upstream_receipt_sha256",
                "source_decision_sha256",
                "frozen_tts_recipe_sha256",
                "lightcone_recipe_sha256",
                "final_blocks",
            },
        }
        row = _formal_request(
            value, kind=kind, fields=frozenset(common | fields_by_kind[kind])
        )
        if kind == "E5":
            if nested_policy is None or nested_policy_sha256 is None:
                raise ValueError("E5 requires a dynamically authorized anchor policy")
            return materialize_e5(
                protocol_lock_sha256=row["protocol_lock_sha256"],
                upstream_e1a_receipt_sha256=row["upstream_e1a_receipt_sha256"],
                power_prefix_decision_sha256=row["power_prefix_decision_sha256"],
                model=row["model"],
                frozen_tts_recipe_sha256=row["frozen_tts_recipe_sha256"],
                lightcone_recipe_sha256=row["lightcone_recipe_sha256"],
                final_blocks=row["final_blocks"],
                signed_anchor_selection=signed_e5_anchor_selection_from_dict(
                    row["signed_anchor_selection"]
                ),
                anchor_policy=nested_policy,
                expected_anchor_policy_sha256=nested_policy_sha256,
                now_ns=now_ns,
                gpu_hours=gpu_hour_estimate_from_dict(row["gpu_hours"]),
            )
        function = {
            "E3b": materialize_e3b,
            "E1a": materialize_e1a,
            "E6": materialize_e6,
        }[kind]
        return function(
            **{name: row[name] for name in row if name not in {"kind", "gpu_hours"}},
            gpu_hours=gpu_hour_estimate_from_dict(row["gpu_hours"]),
        )
    if kind == "E0":
        row = _formal_request(
            value,
            kind=kind,
            fields=frozenset(
                common
                | {
                    "signed_compatibility",
                    "source_decision_sha256",
                    "frozen_tts_recipe_sha256",
                    "lightcone_recipe_sha256",
                    "online_spec_recipe_sha256s",
                    "final_blocks",
                }
            ),
        )
        if (
            row["protocol_lock_sha256"]
            != row["signed_compatibility"]["payload"]["protocol_lock_sha256"]
        ):
            raise ValueError("E0 request ProtocolLock differs from compatibility")
        if type(row["online_spec_recipe_sha256s"]) is not list:
            raise TypeError("E0 OnlineSPEC recipes must be a JSON array")
        if nested_policy is None or nested_policy_sha256 is None:
            raise ValueError(
                "E0 requires a dynamically authorized compatibility policy"
            )
        return materialize_e0_from_signed_compatibility(
            signed_e0_compatibility_from_dict(row["signed_compatibility"]),
            policy=nested_policy,
            expected_policy_sha256=nested_policy_sha256,
            now_ns=now_ns,
            source_decision_sha256=row["source_decision_sha256"],
            frozen_tts_recipe_sha256=row["frozen_tts_recipe_sha256"],
            lightcone_recipe_sha256=row["lightcone_recipe_sha256"],
            online_spec_recipe_sha256s=tuple(
                tuple(item) for item in row["online_spec_recipe_sha256s"]
            ),
            final_blocks=row["final_blocks"],
            gpu_hours=gpu_hour_estimate_from_dict(row["gpu_hours"]),
        )
    raise ValueError("formal materialization request names an unknown stage/wave")


def _create_stage_materialization(args: argparse.Namespace) -> int:
    request = _load_bound_json(args.request)
    if type(request) is not dict or type(request.get("kind")) is not str:
        raise TypeError("formal materialization request requires a kind")
    nested_signed = None
    control = None
    policy = None
    policy_sha256 = None
    if request["kind"] in {"E5", "E0"}:
        if not all(
            (
                args.control_attestation,
                args.inventory_sha256,
                args.control_replay_store,
                args.now_ns is not None,
            )
        ):
            raise ValueError("signed nested authority requires dynamic control inputs")
        nested_signed = (
            signed_e5_anchor_selection_from_dict(request["signed_anchor_selection"])
            if request["kind"] == "E5"
            else signed_e0_compatibility_from_dict(request["signed_compatibility"])
        )
        control = ControlArtifactAttestation.from_dict(
            _load_bound_json(args.control_attestation)
        )
        policy, policy_sha256 = _candidate_dynamic_formal_policy(control)
    receipt = _materialize_formal_request(
        request,
        now_ns=args.now_ns,
        nested_policy=policy,
        nested_policy_sha256=policy_sha256,
    )
    if nested_signed is not None and control is not None:
        _verify_single_formal_control_diagnostic(
            control,
            signed_artifact_sha256=nested_signed.sha256,
            protocol_lock_sha256=receipt.protocol_lock_sha256,
            expected_artifact_type="dispatch",
            inventory_sha256=args.inventory_sha256,
            now_ns=args.now_ns,
        )
    _write_json(args.output, stage_materialization_receipt_to_dict(receipt))
    print(receipt.sha256)
    return 0


def _verify_signed_stage_materialization(args: argparse.Namespace) -> int:
    signed = signed_stage_materialization_from_dict(
        _load_bound_json(args.signed_receipt)
    )
    control = ControlArtifactAttestation.from_dict(
        _load_bound_json(args.control_attestation)
    )
    policy, policy_sha256 = _candidate_dynamic_formal_policy(control)
    payload = signed.verify(
        policy=policy,
        expected_policy_sha256=policy_sha256,
        now_ns=args.now_ns,
    )
    verified_control = _verify_single_formal_control_diagnostic(
        control,
        signed_artifact_sha256=signed.sha256,
        protocol_lock_sha256=payload.protocol_lock_sha256,
        expected_artifact_type="dispatch",
        inventory_sha256=args.inventory_sha256,
        now_ns=args.now_ns,
    )
    _write_json(
        args.output,
        {
            "schema_version": 1,
            "kind": "verified_stage_materialization_receipt",
            "stage": payload.stage,
            "payload_sha256": payload.sha256,
            "signed_receipt_sha256": signed.sha256,
            "trusted_attester_policy_sha256": policy_sha256,
            "control_envelope_sha256": verified_control.envelope_sha256,
            "challenge_reservation_sha256": None,
            "verification_mode": "diagnostic_only_non_consuming",
            "formal_dispatch_authorized": False,
        },
    )
    print(payload.sha256)
    return 0


def _create_stage_coverage(args: argparse.Namespace) -> int:
    materialization = stage_materialization_receipt_from_dict(
        _load_bound_json(args.materialization)
    )
    if materialization.stage in FORMAL_STAGE_DAG:
        raise ValueError(
            "formal DAG coverage is reducer-owned; use the stage-specific "
            "proof-derived coverage reducer"
        )
    raw_dispositions = _load_bound_json(args.dispositions)
    if type(raw_dispositions) is not list:
        raise TypeError("coverage dispositions must be a JSON array")
    dispositions = tuple(StageCellDisposition(**row) for row in raw_dispositions)
    receipt = StageCoverageReceipt(
        schema_version=2,
        stage=materialization.stage,
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(sorted(dispositions, key=lambda row: row.cell_id)),
        tts_l0_candidate_state_coverages=tuple(
            sorted(
                (
                    tts_l0_candidate_state_coverage_from_dict(_load_bound_json(path))
                    for path in args.tts_l0_candidate_state_coverage
                ),
                key=lambda row: row.pair_id,
            )
        ),
    )
    receipt.validate_against(materialization)
    _write_json(args.output, stage_coverage_receipt_to_dict(receipt))
    print(receipt.sha256)
    return 0


def _verify_signed_stage_coverage(args: argparse.Namespace) -> int:
    materialization = stage_materialization_receipt_from_dict(
        _load_bound_json(args.materialization)
    )
    signed = signed_stage_coverage_from_dict(_load_bound_json(args.signed_receipt))
    control = ControlArtifactAttestation.from_dict(
        _load_bound_json(args.control_attestation)
    )
    policy, policy_sha256 = _candidate_dynamic_formal_policy(control)
    payload = signed.verify(
        materialization=materialization,
        policy=policy,
        expected_policy_sha256=policy_sha256,
        now_ns=args.now_ns,
    )
    verified_control = _verify_single_formal_control_diagnostic(
        control,
        signed_artifact_sha256=signed.sha256,
        protocol_lock_sha256=payload.protocol_lock_sha256,
        expected_artifact_type="rank_aggregate",
        inventory_sha256=args.inventory_sha256,
        now_ns=args.now_ns,
    )
    _write_json(
        args.output,
        {
            "schema_version": 1,
            "kind": "verified_stage_coverage_receipt",
            "stage": payload.stage,
            "payload_sha256": payload.sha256,
            "signed_receipt_sha256": signed.sha256,
            "trusted_attester_policy_sha256": policy_sha256,
            "control_envelope_sha256": verified_control.envelope_sha256,
            "challenge_reservation_sha256": None,
            "verification_mode": "diagnostic_only_non_consuming",
            "formal_dispatch_authorized": False,
        },
    )
    print(payload.sha256)
    return 0


def _publish_formal_rebuild_artifact(args: argparse.Namespace) -> int:
    """Strictly decode and immutably publish one closed rebuild artifact."""

    value = _load_bound_json(args.input)
    handlers = {
        "stage-source": (
            FormalStageSourceRebuildInput.from_dict,
            publish_formal_stage_source_rebuild_input,
        ),
        "serving-shard": (
            E0ExecutionRebuildShard.from_dict,
            publish_e0_execution_rebuild_shard,
        ),
        "failure-shard": (
            E5FailureExecutionRebuildShard.from_dict,
            publish_e5_failure_execution_rebuild_shard,
        ),
        "e6-recursive-dag": (
            E6RecursiveSourceDagArtifact.from_dict,
            publish_e6_recursive_source_dag_artifact,
        ),
        "e0-aggregate": (
            E0FormalRegistryAuthorityArtifact.from_dict,
            publish_e0_formal_registry_authority_artifact,
        ),
        "e0-final-result": (
            E0FinalResultRebuildArtifact.from_dict,
            publish_e0_final_result_rebuild_artifact,
        ),
    }
    decoder, publisher = handlers[args.artifact_kind]
    artifact = decoder(value)
    binding = publisher(artifact, Path(args.output).resolve())
    print(binding.semantic_sha256)
    return 0


def _publish_formal_tts_calibration_reduction_proof(
    args: argparse.Namespace,
) -> int:
    """Re-reduce all 288 TTS controls and publish the unique winner proof."""

    reservation = ChallengeReplayReservationBinding.from_dict(
        _load_bound_json(args.replay_reservation)
    )
    artifact = build_formal_tts_calibration_reduction_proof_artifact(
        portable_coverage_proof_path=args.portable_coverage_proof,
        hardware_envelope_source_path=args.hardware_envelope,
        replay_reservation=reservation,
        runtime_sha256=args.runtime_sha256,
        split_sha256=args.split_sha256,
        now_ns=args.now_ns,
    )
    binding = publish_formal_tts_calibration_reduction_proof_artifact(
        artifact,
        args.output,
        now_ns=args.now_ns,
    )
    rebuilt, seal = revalidate_formal_tts_calibration_reduction_proof_artifact(
        binding.absolute_path,
        now_ns=args.now_ns,
    )
    if (
        rebuilt.sha256 != artifact.expected_reduction_sha256
        or seal.sha256 != artifact.expected_seal_payload_sha256
    ):
        raise AssertionError("TTS reduction proof changed after publication")
    print(binding.semantic_sha256)
    return 0


def _publish_formal_stage_execution_shard(args: argparse.Namespace) -> int:
    materialization = stage_materialization_receipt_from_dict(
        _load_bound_json(args.materialization)
    )
    stage_source = (
        None
        if args.stage_source_rebuild is None
        else FormalStageSourceRebuildInput.from_dict(
            _load_bound_json(args.stage_source_rebuild)
        )
    )
    shard = FormalStageExecutionRebuildShard(
        schema_version=1,
        kind=FORMAL_STAGE_EXECUTION_REBUILD_SHARD_KIND,
        phase=args.phase,
        materialization_receipt_sha256=materialization.sha256,
        stage_source_rebuild_input_sha256=(
            None if stage_source is None else stage_source.sha256
        ),
        descriptors=tuple(
            FormalServingExecutionRebuildInput.from_dict(_load_bound_json(path))
            for path in args.execution_rebuild_input
        ),
    )
    binding = publish_formal_stage_execution_rebuild_shard(shard, args.output)
    print(binding.semantic_sha256)
    return 0


def _publish_formal_stage_prefix(args: argparse.Namespace) -> int:
    artifact = bind_formal_stage_prefix_artifact(
        phase=args.phase,
        registry_verification_receipt_path=args.registry_verification_receipt,
        formal_runtime_authority_manifest_path=(args.formal_runtime_authority_manifest),
        inventory_path=args.inventory,
        materialization_path=args.materialization,
        coverage_path=args.coverage,
        coverage_proof_path=args.coverage_proof,
        stage_source_rebuild_path=args.stage_source_rebuild,
        execution_rebuild_shard_paths=tuple(args.execution_rebuild_shard),
        e1_recipe_anchor_authority_path=args.e1_recipe_anchor_authority,
        prior_prefix_path=args.prior_prefix,
    )
    binding = publish_formal_stage_prefix_artifact(artifact, args.output)
    rebuilt = load_and_rebuild_formal_stage_prefix(
        binding.absolute_path,
        now_ns=args.now_ns,
    )
    if rebuilt.artifact != artifact:
        raise RuntimeError("published formal stage prefix changed while rebuilt")
    print(binding.semantic_sha256)
    return 0


def _publish_scientific_source_validation(args: argparse.Namespace) -> int:
    binding = publish_scientific_source_validation_artifact(
        artifact_type=args.artifact_type,
        proof_bundle_path=args.proof_bundle,
        proof_entry_remote_absolute_path=args.proof_entry,
        now_ns=args.now_ns,
        output_path=args.output,
    )
    print(binding.semantic_sha256)
    return 0


def _publish_formal_runtime_authority_manifest(args: argparse.Namespace) -> int:
    """Derive the closed runtime authority from source and publish O_EXCL."""

    manifest = build_source_formal_runtime_authority_manifest(
        Path(args.repository_root)
    )
    binding = publish_formal_runtime_authority_manifest(args.output, manifest)
    reopened = formal_runtime_authority_manifest_from_dict(binding.reopen())
    if reopened != manifest:
        raise RuntimeError("formal runtime authority changed after CLI publication")
    print(manifest.sha256)
    return 0


def _one_formal_operator_row(rows, *, label: str, predicate):
    selected = tuple(row for row in rows if predicate(row))
    if len(selected) != 1:
        raise ValueError(f"formal stage operator requires one exact {label}")
    return selected[0]


def _formal_operator_e2_round(materialization) -> int:
    rounds = {dict(cell.dimensions).get("round") for cell in materialization.cells}
    if len(rounds) != 1:
        raise ValueError("formal stage operator E2 materialization is not one round")
    round_index = next(iter(rounds))
    if type(round_index) is not int or round_index not in range(4):
        raise ValueError("formal stage operator E2 round is invalid")
    return round_index


_FORMAL_OPERATOR_E4_RULE_BY_PHASE = {
    "screen": "strength2_8_rows_x_3_loads_x_2_traffic",
    "local": "winner_neighborhood_2pow4_x_3_loads_x_2_traffic",
    "profiler": "three_profiler_only_rows_separate_from_headline",
}

_FORMAL_OPERATOR_PHASES = {
    "E3a": frozenset({"selection"}),
    "TTS-Cal": frozenset({"calibration"}),
    "E1": frozenset({"selection"}),
    "E2": frozenset({"round0", "round1", "round2", "round3"}),
    "E4": frozenset(_FORMAL_OPERATOR_E4_RULE_BY_PHASE),
    "E3b": frozenset({"pilot", "final"}),
    "E1a": frozenset({"verification"}),
    "E5": frozenset({"pilot", "final"}),
    "E6": frozenset({"compatibility", "pilot", "final"}),
    "E0": frozenset({"compatibility", "tuning", "pilot", "final"}),
}


def _require_formal_operator_phase(*, stage: str, phase: str) -> None:
    if phase not in _FORMAL_OPERATOR_PHASES[stage]:
        allowed = ",".join(sorted(_FORMAL_OPERATOR_PHASES[stage]))
        raise ValueError(
            f"formal stage operator phase is unsupported for {stage}; expected {allowed}"
        )


def _reopen_formal_operator_dag(
    artifact: E0FormalRegistryAuthorityArtifact,
    *,
    registry_receipt,
) -> E6RecursiveSourceDagArtifact:
    binding = artifact.e6_recursive_source_dag_source
    before = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if before != binding:
        raise ValueError("formal stage operator recursive DAG path identity changed")
    dag = E6RecursiveSourceDagArtifact.from_dict(before.reopen())
    after = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if (
        after != before
        or dag.protocol_lock_sha256 != artifact.protocol_lock_sha256
        or dag.registry_verification_receipt_sha256 != registry_receipt.sha256
    ):
        raise ValueError("formal stage operator recursive DAG lineage differs")
    return dag


def _load_formal_operator_e0_context(
    args: argparse.Namespace,
    *,
    registry_receipt,
):
    if args.e0_authority_bundle is None or args.e0_materialization is None:
        raise ValueError(
            "formal downstream operator requires both E0 aggregate and exact "
            "offline-signed E0 materialization"
        )
    signed_e0_materialization = signed_stage_materialization_from_dict(
        _load_bound_json(args.e0_materialization)
    )
    if signed_e0_materialization.payload.stage != "E0":
        raise ValueError(
            "formal downstream operator E0 materialization has wrong stage"
        )
    artifact = load_e0_formal_registry_authority_artifact_index(
        args.e0_authority_bundle,
        registry_verification_receipt=registry_receipt,
        materialization=signed_e0_materialization.payload,
    )
    bundle = load_e0_formal_registry_authority_bundle(
        args.e0_authority_bundle,
        registry_verification_receipt=registry_receipt,
        materialization=signed_e0_materialization.payload,
        now_ns=args.now_ns,
    )
    return (
        signed_e0_materialization,
        artifact,
        _reopen_formal_operator_dag(artifact, registry_receipt=registry_receipt),
        bundle,
    )


def _reopen_formal_operator_materialization(binding: CanonicalJsonProofBinding):
    before = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if before != binding:
        raise ValueError("formal stage operator materialization path identity changed")
    materialization = stage_materialization_receipt_from_dict(before.reopen())
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != before:
        raise RuntimeError("formal stage operator materialization changed while read")
    return materialization


def _formal_operator_upstream_pair(registry_receipt, *, stage: str):
    materializations = tuple(
        row.payload
        for row in registry_receipt.cumulative_signed_materializations
        if row.payload.stage == stage
    )
    if len(materializations) != 1:
        raise ValueError(f"formal operator requires one exact {stage} materialization")
    materialization = materializations[0]
    coverages = tuple(
        row.payload
        for row in registry_receipt.cumulative_signed_coverage
        if row.payload.stage == stage
        and row.payload.materialization_receipt_sha256 == materialization.sha256
    )
    if len(coverages) != 1:
        raise ValueError(f"formal operator requires one exact {stage} coverage")
    return materialization, coverages[0]


def _formal_operator_create_upstream_materialization(
    args: argparse.Namespace,
    *,
    registry_receipt,
):
    """Create the early DAG stages directly from the durable typed prefix."""

    unmeasured = GpuHourEstimate.unmeasured()
    if args.stage == "E3a" and args.phase == "selection":
        protocol_lock = registry_receipt.signed_protocol_lock.payload
        preflight, preflight_coverage = _formal_operator_upstream_pair(
            registry_receipt, stage="preflight"
        )
        return materialize_e3a(
            registry_verification_receipt=registry_receipt,
            protocol_lock=protocol_lock,
            preflight_materialization=preflight,
            preflight_coverage=preflight_coverage,
            now_ns=args.now_ns,
            gpu_hours=unmeasured,
        )
    if args.stage == "TTS-Cal" and args.phase == "calibration":
        protocol_lock = registry_receipt.signed_protocol_lock.payload
        if args.tts_calibration_authority is None:
            raise ValueError(
                "TTS-Cal materialization requires its typed calibration authority"
            )
        e3a, e3a_coverage = _formal_operator_upstream_pair(
            registry_receipt, stage="E3a"
        )
        authority = tts_calibration_authority_from_dict(
            _load_bound_json(args.tts_calibration_authority)
        )
        return materialize_tts_calibration(
            registry_verification_receipt=registry_receipt,
            protocol_lock=protocol_lock,
            tts_calibration_authority=authority,
            e3a_materialization=e3a,
            e3a_coverage=e3a_coverage,
            now_ns=args.now_ns,
            gpu_hours=unmeasured,
        )
    if args.stage == "E1" and args.phase == "selection":
        protocol_lock = registry_receipt.signed_protocol_lock.payload
        e3a, e3a_coverage = _formal_operator_upstream_pair(
            registry_receipt, stage="E3a"
        )
        calibration, calibration_coverage = _formal_operator_upstream_pair(
            registry_receipt, stage="TTS-Cal"
        )
        return materialize_e1_first_slice(
            registry_verification_receipt=registry_receipt,
            protocol_lock=protocol_lock,
            tts_calibration_materialization=calibration,
            tts_calibration_coverage=calibration_coverage,
            e3a_materialization=e3a,
            e3a_coverage=e3a_coverage,
            now_ns=args.now_ns,
            gpu_hours=unmeasured,
        )
    return None


def _formal_operator_materialization(
    *,
    stage: str,
    phase: str,
    registry_receipt,
    signed_e0_materialization,
    artifact: E0FormalRegistryAuthorityArtifact | None,
    dag: E6RecursiveSourceDagArtifact | None,
):
    signed_rows = registry_receipt.cumulative_signed_materializations
    if stage == "E0" and phase == "final":
        if signed_e0_materialization is None:
            raise ValueError("formal E0 final materialization wrapper is missing")
        return signed_e0_materialization.payload
    if stage == "E0" and phase in {"tuning", "pilot"}:
        if artifact is None:
            raise ValueError("formal E0 aggregate is missing")
        binding = (
            artifact.e0_tuning_materialization_source
            if phase == "tuning"
            else artifact.e0_pilot_materialization_source
        )
        return _reopen_formal_operator_materialization(binding)
    if stage in {"E3b", "E5", "E6"} and phase == "pilot":
        if dag is None:
            raise ValueError("formal recursive DAG is missing")
        node_id = {"E3b": "e3b_pilot", "E5": "e5_pilot", "E6": "e6_pilot"}[stage]
        node = _one_formal_operator_row(
            dag.nodes,
            label=f"{node_id} DAG node",
            predicate=lambda row: row.node_id == node_id,
        )
        return _reopen_formal_operator_materialization(node.materialization_source)
    if phase in {"compatibility"}:
        raise ValueError(f"formal {stage} compatibility is not a materialization phase")
    if stage == "E2":
        round_index = int(phase.removeprefix("round"))
        signed = _one_formal_operator_row(
            signed_rows,
            label=f"E2 round {round_index} materialization",
            predicate=lambda row: (
                row.payload.stage == "E2"
                and _formal_operator_e2_round(row.payload) == round_index
            ),
        )
        return signed.payload
    if stage == "E4":
        signed = _one_formal_operator_row(
            signed_rows,
            label=f"E4 {phase} materialization",
            predicate=lambda row: (
                row.payload.stage == "E4"
                and row.payload.materialization_rule
                == _FORMAL_OPERATOR_E4_RULE_BY_PHASE[phase]
            ),
        )
        return signed.payload
    signed = _one_formal_operator_row(
        signed_rows,
        label=f"{stage} materialization",
        predicate=lambda row: row.payload.stage == stage,
    )
    return signed.payload


def _formal_operator_signed_sources(
    *,
    stage: str,
    phase: str,
    registry_receipt,
    artifact: E0FormalRegistryAuthorityArtifact | None,
    dag: E6RecursiveSourceDagArtifact | None,
):
    if stage == "E3a":
        row = _one_formal_operator_row(
            registry_receipt.cumulative_signed_e3a_staged_selections,
            label="signed E3a selection",
            predicate=lambda _row: True,
        )
        return ((row, signed_e3a_staged_selection_to_dict),)
    if stage == "TTS-Cal":
        row = _one_formal_operator_row(
            registry_receipt.cumulative_signed_tts_calibration_seals,
            label="signed TTS calibration seal",
            predicate=lambda _row: True,
        )
        return ((row, signed_tts_calibration_seal_to_dict),)
    if stage == "E1":
        row = _one_formal_operator_row(
            registry_receipt.cumulative_signed_e1_survivor_selections,
            label="signed E1 survivor selection",
            predicate=lambda _row: True,
        )
        return ((row, signed_e1_survivor_selection_to_dict),)
    if stage == "E2":
        round_index = int(phase.removeprefix("round"))
        row = _one_formal_operator_row(
            registry_receipt.cumulative_signed_e2_staged_selections,
            label=f"signed E2 round {round_index} selection",
            predicate=lambda item: item.payload.round_index == round_index,
        )
        return ((row, signed_e2_staged_selection_to_dict),)
    if stage == "E4":
        if phase == "profiler":
            raise ValueError("formal E4 profiler has no registered reducer/sign result")
        row = _one_formal_operator_row(
            registry_receipt.cumulative_signed_e4_stage_selections,
            label=f"signed E4 {phase} selection",
            predicate=lambda item: item.payload.phase == phase,
        )
        return ((row, signed_e4_stage_selection_to_dict),)
    if stage == "E3b" and phase == "pilot":
        row = _one_formal_operator_row(
            registry_receipt.cumulative_signed_e3b_power_prefixes,
            label="signed E3b power prefix",
            predicate=lambda _row: True,
        )
        return ((row, signed_e3b_power_prefix_to_dict),)
    if stage == "E5" and phase == "pilot":
        row = _one_formal_operator_row(
            registry_receipt.cumulative_signed_e5_power_and_anchor_prefixes,
            label="signed E5 power/anchor prefix",
            predicate=lambda _row: True,
        )
        return ((row, signed_e5_power_and_anchor_to_dict),)
    if stage == "E6" and phase == "pilot":
        row = _one_formal_operator_row(
            registry_receipt.cumulative_signed_e6_power_prefixes,
            label="signed E6 power prefix",
            predicate=lambda _row: True,
        )
        return ((row, signed_e6_power_prefix_to_dict),)
    if artifact is None or dag is None:
        raise ValueError("formal downstream signed source requires the E0 aggregate")
    if stage == "E3b" and phase == "final":
        return ((dag.signed_e3b_confirmation, signed_e3b_confirmation_to_dict),)
    if stage == "E1a":
        return ((dag.signed_e1a_verification, signed_e1a_verification_to_dict),)
    if stage == "E5" and phase == "final":
        return ((dag.signed_e5_confirmation, signed_e5_confirmation_to_dict),)
    if stage == "E6" and phase == "compatibility":
        return (
            (
                artifact.signed_e6_model_compatibility,
                signed_e6_model_compatibility_to_dict,
            ),
        )
    if stage == "E6" and phase == "final":
        return ((artifact.signed_e6_confirmation, signed_e6_confirmation_to_dict),)
    if stage == "E0" and phase == "compatibility":
        return ((artifact.signed_e0_compatibility, signed_e0_compatibility_to_dict),)
    if stage == "E0" and phase == "tuning":
        return tuple(
            (row, signed_e0_onlinespec_tuning_seal_to_dict)
            for row in artifact.signed_e0_tuning_seals
        )
    if stage == "E0" and phase == "pilot":
        return ((artifact.signed_e0_power_prefix, signed_e0_power_prefix_to_dict),)
    if stage == "E0" and phase == "final":
        raise ValueError("formal E0 final result requires its proof-rebuild artifact")
    raise AssertionError((stage, phase))


def _formal_operator_reduce_completion(
    args: argparse.Namespace,
    *,
    registry_receipt,
):
    """Run one proof-derived completion reducer, never a result replay."""

    if args.stage == "E4" and args.phase == "profiler":
        materialization = _formal_operator_materialization(
            stage=args.stage,
            phase=args.phase,
            registry_receipt=registry_receipt,
            signed_e0_materialization=None,
            artifact=None,
            dag=None,
        )
        receipt = reduce_e4_profiler_completion_from_registry(
            registry_verification_receipt=registry_receipt,
            materialization=materialization,
            now_ns=args.now_ns,
        )
        return receipt, e4_profiler_completion_receipt_to_dict
    if args.stage == "E0" and args.phase == "final":
        if args.result_rebuild_artifact is None:
            raise ValueError(
                "formal E0 final reducer requires a proof-rebuild artifact"
            )
        receipt = reduce_e0_final_completion_from_artifact(
            args.result_rebuild_artifact,
            now_ns=args.now_ns,
        )
        if receipt.current_registry_verification_receipt_sha256 != (
            registry_receipt.sha256
        ):
            raise ValueError("formal E0 final reducer used a foreign registry prefix")
        return receipt, e0_final_completion_receipt_to_dict
    return None


def _formal_operator_verify_completion_signature(
    args: argparse.Namespace,
    *,
    registry_receipt,
):
    """Deep-reduce then verify one existing offline-signed completion."""

    if args.signed_stage_result is None:
        raise ValueError("formal completion sign verification requires signed result")
    policy = registry_receipt.trusted_release_policy(current_ns=args.now_ns)
    expected_policy_sha256 = policy.sha256
    if args.stage == "E4" and args.phase == "profiler":
        materialization = _formal_operator_materialization(
            stage=args.stage,
            phase=args.phase,
            registry_receipt=registry_receipt,
            signed_e0_materialization=None,
            artifact=None,
            dag=None,
        )
        signed = signed_e4_profiler_completion_from_dict(
            _load_bound_json(args.signed_stage_result)
        )
        signed.verify(
            registry_verification_receipt=registry_receipt,
            materialization=materialization,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=args.now_ns,
        )
        return signed, signed_e4_profiler_completion_to_dict
    if args.stage == "E0" and args.phase == "final":
        if args.result_rebuild_artifact is None:
            raise ValueError(
                "formal E0 final signature verification requires proof-rebuild artifact"
            )
        signed = signed_e0_final_completion_from_dict(
            _load_bound_json(args.signed_stage_result)
        )
        payload = signed.verify(
            rebuild_artifact_path=args.result_rebuild_artifact,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=args.now_ns,
        )
        if payload.current_registry_verification_receipt_sha256 != (
            registry_receipt.sha256
        ):
            raise ValueError(
                "formal E0 final signed result used a foreign registry prefix"
            )
        return signed, signed_e0_final_completion_to_dict
    return None


def _formal_prefix_requested_operation(
    args: argparse.Namespace,
    *,
    registry_receipt,
):
    """Replay one sequential E1/E2/E4 operation from its current-only prefix."""

    path = getattr(args, "stage_prefix_artifact", None)
    if path is None:
        return None
    prefix = load_and_rebuild_formal_stage_prefix(path, now_ns=args.now_ns)
    current = {
        "e1_selection": ("E1", "selection"),
        "e2_round0": ("E2", "round0"),
        "e2_round1": ("E2", "round1"),
        "e2_round2": ("E2", "round2"),
        "e2_round3": ("E2", "round3"),
        "e4_screen": ("E4", "screen"),
        "e4_local": ("E4", "local"),
        "e4_profiler": ("E4", "profiler"),
    }[prefix.artifact.phase]
    if args.operation == "materialize":
        successor = {
            "e1_selection": ("E2", "round0"),
            "e2_round0": ("E2", "round1"),
            "e2_round1": ("E2", "round2"),
            "e2_round2": ("E2", "round3"),
            "e2_round3": ("E4", "screen"),
            "e4_screen": ("E4", "local"),
            "e4_local": ("E4", "profiler"),
            "e4_profiler": ("E3b", "pilot"),
        }.get(prefix.artifact.phase)
        if successor is None or (args.stage, args.phase) != successor:
            raise ValueError("formal stage prefix is not the requested DAG predecessor")
        if prefix.artifact.phase == "e4_profiler":
            prior = prefix.prior
            if prior is None or prior.artifact.phase != "e4_local":
                raise ValueError("formal E3b pilot lacks its immediate E4 local prefix")
            signed_e4 = _one_formal_operator_row(
                registry_receipt.cumulative_signed_e4_stage_selections,
                label="signed E4 local selection",
                predicate=lambda row: (
                    row.payload.phase == "local"
                    and row.payload.materialization_receipt_sha256
                    == prior.materialization.sha256
                    and row.payload.coverage_receipt_sha256 == prior.coverage.sha256
                ),
            )
            tts = _one_formal_operator_row(
                registry_receipt.cumulative_tts_calibration_authorities,
                label="frozen TTS authority",
                predicate=lambda _row: True,
            )
            signed_tts = _one_formal_operator_row(
                registry_receipt.cumulative_signed_tts_calibration_seals,
                label="signed frozen TTS seal",
                predicate=lambda _row: True,
            )
            if prior.evidence_manifest is None:
                raise ValueError("formal E3b pilot lacks proof-derived E4 evidence")
            materialization = materialize_e3b_excluded_pilots(
                registry_verification_receipt=registry_receipt,
                signed_e4_final_selection=signed_e4,
                local_materialization=prior.materialization,
                local_coverage=prior.coverage,
                local_evidence_manifest=prior.evidence_manifest,
                local_execution_bindings=prior.execution_bindings,
                profiler_materialization=prefix.materialization,
                profiler_coverage=prefix.coverage,
                tts_calibration_authority=tts,
                signed_tts_calibration_seal=signed_tts,
                now_ns=args.now_ns,
            )
        else:
            materialization = materialize_next_formal_stage_from_prefix(
                prefix,
                registry_verification_receipt=registry_receipt,
                now_ns=args.now_ns,
            )
        return "materialization", materialization
    if (args.stage, args.phase) != current:
        raise ValueError("formal stage prefix is not the requested reducer phase")
    if prefix.artifact.phase == "e4_profiler":
        raise ValueError(
            "formal E4 profiler completion is reduced only after registry append"
        )
    if registry_receipt.sha256 != prefix.registry_verification_receipt.sha256:
        raise ValueError("formal reducer registry differs from its bound prefix")
    if args.operation == "reduce":
        return "reduction", reduce_formal_stage_prefix(prefix, now_ns=args.now_ns)
    if args.signed_stage_result is None:
        raise ValueError("formal prefix sign verification requires signed result")
    if prefix.artifact.phase == "e1_selection":
        artifact_type = "e1-survivor-selection"
        decoder = signed_e1_survivor_selection_from_dict
    elif prefix.artifact.phase.startswith("e2_round"):
        artifact_type = "e2-staged-selection"
        decoder = signed_e2_staged_selection_from_dict
    else:
        artifact_type = "e4-stage-selection"
        decoder = signed_e4_stage_selection_from_dict
    signed = _load_formal_scientific_signed_path(
        args.signed_stage_result,
        artifact_type=artifact_type,
        decoder=decoder,
        now_ns=args.now_ns,
    )
    verify_signed_formal_stage_prefix_result(prefix, signed, now_ns=args.now_ns)
    return "signature", signed


def _formal_prefix_payload_json(value: object) -> dict[str, object]:
    normalized = json.loads(json.dumps(asdict(value), sort_keys=True))
    if type(normalized) is not dict:
        raise TypeError("formal prefix reducer payload is not a JSON object")
    return normalized


def _formal_stage_operation(args: argparse.Namespace) -> int:
    """Replay one closed stage operation from durable public signed inputs.

    ``sign`` validates and exports an existing offline-signed wrapper.  It never
    loads a private key or mints a new scientific authority.
    """

    _require_formal_operator_phase(stage=args.stage, phase=args.phase)
    registry_receipt = _load_formal_registry_receipt_path(
        args.registry_verification_receipt,
        now_ns=args.now_ns,
    )
    completion_operation = (args.stage, args.phase) in {
        ("E4", "profiler"),
        ("E0", "final"),
    } and args.operation in {"reduce", "sign"}
    needs_e0_context = (
        args.stage in {"E3b", "E1a", "E5", "E6", "E0"}
        and getattr(args, "stage_prefix_artifact", None) is None
        and not (args.stage == "E0" and args.phase == "final" and completion_operation)
    )
    signed_e0_materialization = artifact = dag = authority_bundle = None
    if needs_e0_context or args.e0_authority_bundle is not None:
        (
            signed_e0_materialization,
            artifact,
            dag,
            authority_bundle,
        ) = _load_formal_operator_e0_context(
            args,
            registry_receipt=registry_receipt,
        )
    elif args.e0_materialization is not None:
        raise ValueError("E0 materialization cannot be supplied without its aggregate")

    artifacts: list[dict[str, object]] = []
    recursively_reduced = authority_bundle is not None
    prefix_operation = _formal_prefix_requested_operation(
        args,
        registry_receipt=registry_receipt,
    )
    created_upstream = (
        _formal_operator_create_upstream_materialization(
            args,
            registry_receipt=registry_receipt,
        )
        if args.operation == "materialize"
        else None
    )
    if (
        args.operation == "materialize"
        and not recursively_reduced
        and created_upstream is None
        and prefix_operation is None
    ):
        raise ValueError(
            "formal materialization is NON_OPERATOR_BLOCKED without a typed "
            "source-rebuild authority"
        )
    if args.operation == "materialize":
        materialization = (
            prefix_operation[1]
            if prefix_operation is not None
            else created_upstream
            if created_upstream is not None
            else _formal_operator_materialization(
                stage=args.stage,
                phase=args.phase,
                registry_receipt=registry_receipt,
                signed_e0_materialization=signed_e0_materialization,
                artifact=artifact,
                dag=dag,
            )
        )
        artifacts.append(
            {
                "artifact_kind": "stage_materialization_receipt",
                "artifact_sha256": materialization.sha256,
                "value": stage_materialization_receipt_to_dict(materialization),
            }
        )
    elif prefix_operation is not None and prefix_operation[0] == "reduction":
        reduced = prefix_operation[1]
        artifacts.append(
            {
                "artifact_kind": "proof_derived_stage_selection_receipt",
                "artifact_sha256": reduced.sha256,
                "value": _formal_prefix_payload_json(reduced),
            }
        )
    elif prefix_operation is not None and prefix_operation[0] == "signature":
        signed = prefix_operation[1]
        if args.stage == "E1":
            encoded = signed_e1_survivor_selection_to_dict(signed)
        elif args.stage == "E2":
            encoded = signed_e2_staged_selection_to_dict(signed)
        else:
            encoded = signed_e4_stage_selection_to_dict(signed)
        artifacts.append(
            {
                "artifact_kind": "verified_offline_signed_stage_selection",
                "artifact_sha256": signed.sha256,
                "value": encoded,
            }
        )
    elif completion_operation and args.operation == "reduce":
        reduced = _formal_operator_reduce_completion(
            args,
            registry_receipt=registry_receipt,
        )
        if reduced is None:
            raise AssertionError((args.stage, args.phase))
        receipt, codec = reduced
        artifacts.append(
            {
                "artifact_kind": "proof_derived_completion_receipt",
                "artifact_sha256": receipt.sha256,
                "value": codec(receipt),
            }
        )
        if args.stage == "E0" and args.phase == "final":
            staged_registry = build_industrial_registry()
            fdr = reduce_formal_e0_breadth_fdr_from_artifact(
                staged_registry,
                args.result_rebuild_artifact,
                now_ns=args.now_ns,
            )
            artifacts.append(
                {
                    "artifact_kind": "proof_derived_e0_breadth_fdr_receipt",
                    "artifact_sha256": fdr.sha256,
                    "value": formal_e0_breadth_fdr_receipt_to_dict(fdr),
                }
            )
    elif completion_operation and args.operation == "sign":
        verified = _formal_operator_verify_completion_signature(
            args,
            registry_receipt=registry_receipt,
        )
        if verified is None:
            raise AssertionError((args.stage, args.phase))
        signed, codec = verified
        artifacts.append(
            {
                "artifact_kind": "verified_offline_signed_completion",
                "artifact_sha256": signed.sha256,
                "value": codec(signed),
            }
        )
        if args.stage == "E0" and args.phase == "final":
            signed_fdr_path = getattr(args, "signed_e0_fdr_result", None)
            if signed_fdr_path is None:
                raise ValueError(
                    "formal E0 final signature verification requires its signed FDR"
                )
            signed_fdr = signed_formal_e0_breadth_fdr_from_dict(
                _load_bound_json(signed_fdr_path)
            )
            policy = registry_receipt.trusted_release_policy(current_ns=args.now_ns)
            signed_fdr.verify(
                registry=build_industrial_registry(),
                final_result_rebuild_artifact_path=args.result_rebuild_artifact,
                policy=policy,
                expected_policy_sha256=policy.sha256,
                now_ns=args.now_ns,
            )
            artifacts.append(
                {
                    "artifact_kind": "verified_offline_signed_e0_breadth_fdr",
                    "artifact_sha256": signed_fdr.sha256,
                    "value": signed_formal_e0_breadth_fdr_to_dict(signed_fdr),
                }
            )
    elif recursively_reduced:
        sources = _formal_operator_signed_sources(
            stage=args.stage,
            phase=args.phase,
            registry_receipt=registry_receipt,
            artifact=artifact,
            dag=dag,
        )
        for signed, codec in sources:
            signed_value = codec(signed)
            artifacts.append(
                {
                    "artifact_kind": (
                        "offline_signed_source_authority"
                        if args.operation == "sign"
                        else "reducer_source_payload"
                    ),
                    "artifact_sha256": (
                        signed.sha256
                        if args.operation == "sign"
                        else signed.payload.sha256
                    ),
                    "value": (
                        signed_value
                        if args.operation == "sign"
                        else signed_value["payload"]
                    ),
                }
            )
    else:
        raise ValueError(
            "formal stage operation is NON_OPERATOR_BLOCKED without a typed "
            "proof-rebuild authority"
        )
    status = (
        "PROOF_MATERIALIZED"
        if args.operation == "materialize"
        and (
            recursively_reduced
            or created_upstream is not None
            or prefix_operation is not None
        )
        else "NON_OPERATOR_BLOCKED"
        if args.operation == "materialize"
        else "PROOF_REDUCED"
        if completion_operation and args.operation == "reduce"
        else "OFFLINE_SIGNATURE_VERIFIED"
        if completion_operation and args.operation == "sign"
        else "OFFLINE_SIGNATURE_VERIFIED"
        if args.operation == "sign"
        and (recursively_reduced or prefix_operation is not None)
        else "PROOF_REDUCED"
        if args.operation == "reduce"
        and (recursively_reduced or prefix_operation is not None)
        else "NON_OPERATOR_BLOCKED"
    )
    result = {
        "schema_version": 1,
        "kind": "lightcone_formal_stage_operator_result",
        "operation": args.operation,
        "stage": args.stage,
        "phase": args.phase,
        "registry_verification_receipt_sha256": registry_receipt.sha256,
        "e0_authority_artifact_sha256": (None if artifact is None else artifact.sha256),
        "verification_mode": "public_recursive_replay_no_private_key",
        "formal_dispatch_authorized": False,
        "status": status,
        "artifacts": artifacts,
    }
    _write_json(args.output, result)
    print(_canonical_sha256(result))
    return 0


def _load_formal_registry_power_sources(
    args: argparse.Namespace,
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]]:
    """Strictly decode the three source-owned downstream power authorities."""

    return (
        tuple(
            signed_e3b_power_prefix_from_dict(_load_bound_json(path))
            for path in getattr(args, "signed_e3b_power_prefix", ())
        ),
        tuple(
            signed_e5_power_and_anchor_from_dict(_load_bound_json(path))
            for path in getattr(args, "signed_e5_power_and_anchor_prefix", ())
        ),
        tuple(
            signed_e6_power_prefix_from_dict(_load_bound_json(path))
            for path in getattr(args, "signed_e6_power_prefix", ())
        ),
    )


def _assemble_formal_registry(args: argparse.Namespace) -> int:
    raise ValueError(
        "assemble-formal-registry-manifest is NON_OPERATOR_BLOCKED; use the "
        "proof-replay root reservation and schema-5 registry layer"
    )
    lock = signed_protocol_lock_from_dict(  # pragma: no cover - blocked legacy body
        _load_bound_json(args.signed_protocol_lock)
    )
    materializations = tuple(
        signed_stage_materialization_from_dict(_load_bound_json(path))
        for path in args.signed_materialization
    )
    coverage = tuple(
        signed_stage_coverage_from_dict(_load_bound_json(path))
        for path in args.signed_coverage
    )
    controls = tuple(
        ControlArtifactAttestation.from_dict(_load_bound_json(path))
        for path in args.control_attestation
    )
    candidate_replay_proof_artifact_paths = tuple(
        args.candidate_state_replay_proof_artifact
    )
    e3b_power, e5_power_and_anchor, e6_power = _load_formal_registry_power_sources(args)
    manifest = assemble_and_reserve_formal_registry_manifest(
        lock,
        signed_materializations=materializations,
        signed_coverage=coverage,
        tts_calibration_authorities=tuple(
            tts_calibration_authority_from_dict(_load_bound_json(path))
            for path in getattr(args, "tts_calibration_authority", ())
        ),
        signed_tts_calibration_seals=tuple(
            signed_tts_calibration_seal_from_dict(_load_bound_json(path))
            for path in getattr(args, "signed_tts_calibration_seal", ())
        ),
        signed_e3b_power_prefixes=e3b_power,
        signed_e5_power_and_anchor_prefixes=e5_power_and_anchor,
        signed_e6_power_prefixes=e6_power,
        control_attestations=controls,
        candidate_replay_proof_artifact_paths=(candidate_replay_proof_artifact_paths),
        expected_inventory_sha256=args.inventory_sha256,
        replay_store=ChallengeReplayStore(args.control_replay_store),
        now_ns=args.now_ns,
    )
    _write_json(args.output, manifest.to_dict())
    print(manifest.sha256)
    return 0


def _reserve_formal_registry_verification(args: argparse.Namespace) -> int:
    """Consume the immutable ProtocolLock control exactly once."""

    signed_lock = _load_formal_scientific_signed_path(
        args.signed_protocol_lock,
        artifact_type="protocol-lock",
        decoder=signed_protocol_lock_from_dict,
        now_ns=args.now_ns,
    )
    receipt = reserve_formal_registry_verification_receipt(
        signed_lock,
        control_attestation=ControlArtifactAttestation.from_dict(
            _load_bound_json(args.control_attestation)
        ),
        expected_inventory_sha256=args.inventory_sha256,
        replay_store=ChallengeReplayStore(args.control_replay_store),
        now_ns=args.now_ns,
    )
    layer = bind_formal_registry_layer_artifact(
        receipt,
        prior_layer_path=None,
        signed_protocol_lock_path=args.signed_protocol_lock,
        signed_materialization_paths=(),
        signed_coverage_paths=(),
        formal_stage_prefix_paths=(),
    )
    publish_formal_registry_layer_artifact(layer, args.output)
    print(receipt.sha256)
    return 0


def _extend_formal_registry_verification(args: argparse.Namespace) -> int:
    """Atomically append newly signed rows to a durable registry prefix."""

    prior_receipt = _load_formal_registry_receipt_path(
        args.prior_receipt,
        now_ns=args.now_ns,
    )
    materializations = tuple(
        load_formal_signed_materialization_path(path, now_ns=args.now_ns)
        for path in args.signed_materialization
    )
    prefix_paths = tuple(getattr(args, "formal_stage_prefix_artifact", ()))
    legacy_e2_evidence = tuple(getattr(args, "e2_staged_evidence_manifest", ()))
    legacy_e4_evidence = tuple(getattr(args, "e4_staged_evidence_manifest", ()))
    if legacy_e2_evidence or legacy_e4_evidence:
        raise ValueError(
            "formal registry evidence is derived from stage-prefix proof shards"
        )
    rebuilt_prefixes = tuple(
        load_and_rebuild_formal_stage_prefix(path, now_ns=args.now_ns)
        for path in prefix_paths
    )
    e2_evidence = tuple(
        row.evidence_manifest
        for row in rebuilt_prefixes
        if row.artifact.phase.startswith("e2_round")
    )
    e4_evidence = tuple(
        row.evidence_manifest
        for row in rebuilt_prefixes
        if row.artifact.phase in {"e4_screen", "e4_local"}
    )
    if any(row is None for row in (*e2_evidence, *e4_evidence)):
        raise ValueError("formal registry stage prefix lacks reducer evidence")
    coverages = tuple(
        load_formal_signed_coverage_path(
            path,
            formal_stage_prefix_paths=prefix_paths,
            now_ns=args.now_ns,
        )
        for path in args.signed_coverage
    )
    e0_materializations = tuple(
        row.payload for row in materializations if row.payload.stage == "E0"
    )
    e0_artifact_paths = tuple(getattr(args, "e0_authority_bundle", ()))
    if e0_artifact_paths and (
        len(e0_artifact_paths) != 1 or len(e0_materializations) != 1
    ):
        raise ValueError(
            "E0 CLI append requires one bundle and one exact E0 materialization"
        )
    e0_bundles = tuple(
        load_e0_formal_registry_authority_bundle(
            path,
            registry_verification_receipt=prior_receipt,
            materialization=e0_materializations[0],
            now_ns=args.now_ns,
        )
        for path in e0_artifact_paths
    )
    e3b_power, e5_power_and_anchor, e6_power = _load_formal_registry_power_sources(args)
    e3a_proof_paths = tuple(getattr(args, "e3a_staged_selection_proof", ()))
    e3a_reductions = tuple(
        revalidate_formal_e3a_staged_selection_proof_artifact(
            path,
            now_ns=args.now_ns,
        )
        for path in e3a_proof_paths
    )
    signed_e3a_paths = tuple(getattr(args, "signed_e3a_staged_selection", ()))
    signed_e3a = tuple(
        _load_formal_scientific_signed_path(
            path,
            artifact_type="e3a-staged-selection",
            decoder=signed_e3a_staged_selection_from_dict,
            now_ns=args.now_ns,
        )
        for path in signed_e3a_paths
    )
    receipt = extend_formal_registry_verification_receipt(
        prior_receipt,
        appended_signed_materializations=materializations,
        appended_signed_coverage=coverages,
        appended_e3a_staged_selection_artifacts=tuple(
            artifact for artifact, _receipt in e3a_reductions
        ),
        appended_signed_e3a_staged_selections=signed_e3a,
        appended_e2_staged_evidence_manifests=e2_evidence,
        appended_signed_e2_staged_selections=tuple(
            _load_formal_scientific_signed_path(
                path,
                artifact_type="e2-staged-selection",
                decoder=signed_e2_staged_selection_from_dict,
                now_ns=args.now_ns,
            )
            for path in getattr(args, "signed_e2_staged_selection", ())
        ),
        appended_signed_e1_survivor_selections=tuple(
            _load_formal_scientific_signed_path(
                path,
                artifact_type="e1-survivor-selection",
                decoder=signed_e1_survivor_selection_from_dict,
                now_ns=args.now_ns,
            )
            for path in getattr(args, "signed_e1_survivor_selection", ())
        ),
        appended_e4_staged_evidence_manifests=e4_evidence,
        appended_signed_e4_stage_selections=tuple(
            _load_formal_scientific_signed_path(
                path,
                artifact_type="e4-stage-selection",
                decoder=signed_e4_stage_selection_from_dict,
                now_ns=args.now_ns,
            )
            for path in getattr(args, "signed_e4_stage_selection", ())
        ),
        appended_tts_calibration_authorities=tuple(
            tts_calibration_authority_from_dict(_load_bound_json(path))
            for path in getattr(args, "tts_calibration_authority", ())
        ),
        appended_signed_tts_calibration_seals=tuple(
            signed_tts_calibration_seal_from_dict(_load_bound_json(path))
            for path in getattr(args, "signed_tts_calibration_seal", ())
        ),
        appended_signed_e3b_power_prefixes=e3b_power,
        appended_signed_e5_power_and_anchor_prefixes=e5_power_and_anchor,
        appended_signed_e6_power_prefixes=e6_power,
        appended_e0_authority_bundles=e0_bundles,
        formal_stage_prefix_artifact_paths=prefix_paths,
        control_attestations=tuple(
            ControlArtifactAttestation.from_dict(_load_bound_json(path))
            for path in args.control_attestation
        ),
        candidate_replay_proof_artifact_paths=tuple(
            args.candidate_state_replay_proof_artifact
        ),
        replay_store=ChallengeReplayStore(args.control_replay_store),
        now_ns=args.now_ns,
    )
    replay_proof_paths = tuple(args.candidate_state_replay_proof_artifact)
    replay_shard_count = (len(replay_proof_paths) + 255) // 256
    layer_output = Path(args.output)
    replay_shard_paths = tuple(
        layer_output.with_name(f"{layer_output.stem}.candidate-replay-{index:04d}.json")
        for index in range(replay_shard_count)
    )
    publish_formal_registry_replay_proof_shards(
        receipt,
        prior_receipt=prior_receipt,
        candidate_replay_proof_paths=replay_proof_paths,
        shard_output_paths=replay_shard_paths,
    )
    layer = bind_formal_registry_layer_artifact(
        receipt,
        prior_layer_path=args.prior_receipt,
        signed_materialization_paths=tuple(args.signed_materialization),
        signed_coverage_paths=tuple(args.signed_coverage),
        formal_stage_prefix_paths=prefix_paths,
        candidate_replay_proof_shard_paths=replay_shard_paths,
        tts_calibration_reduction_proof_paths=tuple(
            getattr(args, "tts_calibration_reduction_proof", ())
        ),
        e3a_staged_selection_proof_paths=e3a_proof_paths,
        signed_e3a_staged_selection_paths=signed_e3a_paths,
    )
    publish_formal_registry_layer_artifact(layer, args.output)
    print(receipt.sha256)
    return 0


def _verify_formal_registry_verification(args: argparse.Namespace) -> int:
    """Deep-reopen a durable receipt without consuming another challenge."""

    receipt = _load_formal_registry_receipt_path(
        args.receipt,
        now_ns=args.now_ns,
    )
    manifest = receipt.manifest
    _write_json(
        args.output,
        {
            "schema_version": 1,
            "kind": "verified_formal_registry_verification_receipt",
            "receipt_sha256": receipt.sha256,
            "manifest_sha256": manifest.sha256,
            "status": manifest.status,
            "verification_mode": "diagnostic_only_non_consuming",
            "formal_dispatch_authorized": False,
        },
    )
    print(receipt.sha256)
    return 0


def _authorize_formal_preflight_dispatch_cli(args: argparse.Namespace) -> int:
    """Atomically authorize and persist the exact ten-cell preflight token."""

    from lightcone_spec.experiments.budget_authority import (
        replay_budget_activation_authority,
    )
    from lightcone_spec.experiments.formal_dispatch import (
        authorize_formal_preflight_dispatch,
        publish_formal_preflight_dispatch_receipt,
    )
    from lightcone_spec.experiments.gpu_pool import GpuDispatchExecutionContext

    registry_receipt = _load_formal_registry_receipt_path(
        args.registry_verification_receipt,
        now_ns=args.now_ns,
    )
    signed_materialization = signed_stage_materialization_from_dict(
        _load_bound_json(args.signed_materialization)
    )
    inventory = _load_gpu_inventory(args.inventory)
    activation = registry_stage_activation_from_dict(
        _load_bound_json(args.stage_activation)
    )
    budget_plan = budget_plan_from_dict(_load_bound_json(args.budget_plan))
    budget_authority = budget_materialization_authority_binding_from_dict(
        _load_bound_json(args.budget_authority)
    )
    replay = replay_budget_activation_authority(budget_authority.activation)
    if (
        replay.activation_artifact != activation
        or replay.family_activations
        or replay.family_power_reductions
        or replay.stage_family_authorities
        or replay.auxiliary_authority is not None
    ):
        raise ValueError(
            "formal preflight budget authority is not the exact stage activation"
        )
    registry = build_industrial_registry()
    bootstrap = materialize_interference_calibration_bootstrap_authority(
        registry,
        inventory,
        activation,
    )
    context = GpuDispatchExecutionContext(
        registry=registry,
        inventory=inventory,
        interference_envelope=bootstrap.bootstrap_envelope,
        budgets=budget_plan.diagnostic_budgets,
        receipts=replay.dependency_receipts,
        activation_artifact=activation,
        budget_plan=budget_plan,
        budget_materialization_authority=budget_authority,
        interference_calibration_bootstrap_authority=bootstrap,
    )
    dispatch_plan = GpuDispatchPlan.from_dict(
        _load_bound_json(args.dispatch_plan),
        planning_context=context,
    )
    capacity_schedule = StageCapacitySchedule.from_dict(
        _load_bound_json(args.capacity_schedule)
    )
    capacity_gate = StageCapacityGate.from_dict(_load_bound_json(args.capacity_gate))
    capacity_control = ControlArtifactAttestation.from_dict(
        _load_bound_json(args.capacity_control)
    )
    dispatch_control = ControlArtifactAttestation.from_dict(
        _load_bound_json(args.dispatch_control)
    )
    replay_store = ChallengeReplayStore(args.control_replay_store)
    token = authorize_formal_preflight_dispatch(
        registry_receipt,
        signed_materialization=signed_materialization,
        capacity_control_attestation=capacity_control,
        dispatch_control_attestation=dispatch_control,
        dispatch_context=context,
        dispatch_plan=dispatch_plan,
        capacity_schedule=capacity_schedule,
        capacity_gate=capacity_gate,
        replay_store=replay_store,
        now_ns=args.now_ns,
    )
    binding = publish_formal_preflight_dispatch_receipt(
        token,
        registry_verification_receipt=registry_receipt,
        signed_materialization=signed_materialization,
        capacity_schedule=capacity_schedule,
        capacity_gate=capacity_gate,
        capacity_control_attestation=capacity_control,
        dispatch_control_attestation=dispatch_control,
        replay_store=replay_store,
        verified_ns=args.now_ns,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "formal_preflight_dispatch_authorization_result",
                "status": "AUTHORIZED",
                "dispatch_sha256": token.sha256,
                "receipt_raw_sha256": binding.raw_sha256,
                "receipt_semantic_sha256": binding.semantic_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _publish_formal_preflight_remote_error(
    output_path: str | Path,
    *,
    dispatch_receipt_path: str | Path,
    phase: str,
    error: BaseException,
) -> None:
    """Publish a path-safe error terminal without serializing exception text."""

    from lightcone_spec.runtime.proof_artifact import (
        CanonicalJsonProofBinding,
        publish_canonical_json_no_replace,
    )

    try:
        receipt_sha256: str | None = CanonicalJsonProofBinding.bind(
            dispatch_receipt_path
        ).semantic_sha256
    except (OSError, TypeError, ValueError):
        receipt_sha256 = None
    reason_code = getattr(error, "reason_code", None)
    if type(reason_code) is not str or not reason_code.isidentifier():
        reason_code = "formal_preflight_remote_phase_failed"
    publish_canonical_json_no_replace(
        output_path,
        {
            "schema_version": 1,
            "kind": "formal_preflight_remote_error_terminal",
            "status": "ERROR",
            "phase": phase,
            "reason_code": reason_code,
            "exception_type": type(error).__name__,
            "dispatch_receipt_semantic_sha256": receipt_sha256,
            "formal_coverage_complete": False,
            "requires_local_control": True,
        },
    )


def _materialize_formal_preflight_launch_cap_schedule_cli(
    args: argparse.Namespace,
) -> int:
    from lightcone_spec.experiments.formal_preflight_launch import (
        materialize_formal_preflight_launch_cap_schedule,
    )

    schedule = materialize_formal_preflight_launch_cap_schedule(
        dispatch_receipt_path=args.dispatch_receipt,
        output_path=args.output,
        current_ns=args.now_ns,
    )
    print(
        json.dumps(
            {
                "status": "READY",
                "schedule_sha256": schedule.sha256,
                "cell_count": len(schedule.cell_caps),
                "wave_count": len({row.wave_index for row in schedule.cell_caps}),
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


def _execute_formal_preflight_raw_cli(args: argparse.Namespace) -> int:
    """Run remote unsigned sources and stop at the stable-pull boundary."""

    from lightcone_spec.experiments.formal_dispatch import (
        load_formal_preflight_dispatch_receipt,
    )
    from lightcone_spec.experiments.formal_preflight_execution import (
        FormalPreflightInterferenceExecutionManifest,
        execute_formal_preflight_compile_raw,
        execute_formal_preflight_exactness_raw,
        execute_formal_preflight_interference_raw,
        publish_formal_preflight_remote_raw_evidence_receipt,
    )
    from lightcone_spec.orchestration.live_sglang import PinnedNvidiaSmiTool
    from lightcone_spec.runtime.compile_runner import CompileAssignmentPlan
    from lightcone_spec.runtime.preflight_runner import ExactnessPreflightAssignment
    from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

    phase = "load_dispatch"
    try:
        dispatch_receipt = CanonicalJsonProofBinding.bind(args.dispatch_receipt)
        token = load_formal_preflight_dispatch_receipt(
            args.dispatch_receipt,
            current_ns=args.now_ns,
        )
        manifest = FormalPreflightInterferenceExecutionManifest.from_dict(
            _load_bound_json(args.interference_execution_manifest)
        )
        if (
            manifest.dispatch_receipt_semantic_sha256
            != dispatch_receipt.semantic_sha256
        ):
            raise ValueError(
                "interference execution manifest belongs to another dispatch receipt"
            )
        replay_store = ChallengeReplayStore(args.control_replay_store)
        phase = "compile"
        compile_pointer, compile_launch_consumption = (
            execute_formal_preflight_compile_raw(
                token,
                launch_cap_schedule_path=args.launch_cap_schedule,
                assignment_plan_path=args.compile_assignment_plan,
                prepared_content_verification_receipt_path=(
                    args.prepared_content_verification_receipt
                ),
                control_attestation=ControlArtifactAttestation.from_dict(
                    _load_bound_json(args.compile_control)
                ),
                replay_store=replay_store,
                now_ns=args.now_ns,
            )
        )
        compile_result_path = CompileAssignmentPlan.load(
            args.compile_assignment_plan
        ).result_pointer_path
        if compile_pointer != type(compile_pointer).load(compile_result_path):
            raise RuntimeError("compile runner returned another result pointer")

        phase = "exactness"
        exactness_pointer, exactness_launch_consumption = (
            execute_formal_preflight_exactness_raw(
                token,
                launch_cap_schedule_path=args.launch_cap_schedule,
                assignment_path=args.exactness_assignment,
                dispatch_attestation=ControlArtifactAttestation.from_dict(
                    _load_bound_json(args.exactness_control)
                ),
                replay_store=replay_store,
                now_ns=args.now_ns,
            )
        )
        exactness_result_path = (
            Path(
                ExactnessPreflightAssignment.load(
                    args.exactness_assignment
                ).evidence_directory
            )
            / "result.json"
        )
        if exactness_pointer != type(exactness_pointer).load(exactness_result_path):
            raise RuntimeError("exactness runner returned another result pointer")

        phase = "interference"
        interference = asyncio.run(
            execute_formal_preflight_interference_raw(
                token,
                launch_cap_schedule_path=args.launch_cap_schedule,
                execution_inputs={row.registry_cell_id: row for row in manifest.inputs},
                nvidia_smi_tool=PinnedNvidiaSmiTool.bind(args.nvidia_smi),
                evidence_root=args.evidence_root,
                now_ns=args.now_ns,
            )
        )
        if interference.status != "WAITING_FOR_LOCAL_CONTROL":
            raise RuntimeError("interference remote phase did not reach stable pull")
        phase = "publish_waiting_terminal"
        binding = publish_formal_preflight_remote_raw_evidence_receipt(
            token,
            launch_cap_schedule_path=args.launch_cap_schedule,
            launch_consumption_paths=tuple(
                row.absolute_path
                for row in (
                    compile_launch_consumption,
                    exactness_launch_consumption,
                    *interference.launch_consumptions,
                )
            ),
            compile_result_path=compile_result_path,
            exactness_result_path=exactness_result_path,
            interference_raw_batch_path=interference.raw_batch.absolute_path,
            output_path=args.output,
        )
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        _publish_formal_preflight_remote_error(
            args.output,
            dispatch_receipt_path=args.dispatch_receipt,
            phase=phase,
            error=error,
        )
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "formal_preflight_remote_execution_result",
                    "status": "ERROR",
                    "phase": phase,
                    "formal_coverage_complete": False,
                },
                sort_keys=True,
            )
        )
        return 42
    print(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "formal_preflight_remote_execution_result",
                "status": "WAITING_FOR_LOCAL_CONTROL",
                "raw_receipt_semantic_sha256": binding.semantic_sha256,
                "formal_coverage_complete": False,
            },
            sort_keys=True,
        )
    )
    return 42


def _qualify_formal_preflight_exactness_cli(args: argparse.Namespace) -> int:
    """Locally bind the pulled exactness result to rank-aggregate control."""

    from lightcone_spec.experiments.formal_dispatch import (
        load_formal_preflight_dispatch_receipt,
    )
    from lightcone_spec.experiments.formal_preflight_execution import (
        qualify_formal_preflight_exactness_locally,
    )

    token = load_formal_preflight_dispatch_receipt(
        args.dispatch_receipt,
        current_ns=args.now_ns,
    )
    pointer = qualify_formal_preflight_exactness_locally(
        token,
        remote_raw_receipt_path=args.remote_raw_receipt,
        rank_aggregate_control_attestation=ControlArtifactAttestation.from_dict(
            _load_bound_json(args.rank_aggregate_control)
        ),
        replay_store=ChallengeReplayStore(args.control_replay_store),
        now_ns=args.now_ns,
        proof_artifact_path=args.proof_output,
        qualified_result_pointer_path=args.qualified_output,
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "formal_preflight_local_exactness_result",
                "status": "WAITING_FOR_INTERFERENCE_AND_SUITE_PROOFS",
                "qualified_result_sha256": pointer.sha256,
                "formal_coverage_complete": False,
            },
            sort_keys=True,
        )
    )
    return 42


def _parse_preflight_cell_proof_paths(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        cell_id, separator, path = value.partition("=")
        if (
            separator != "="
            or not _is_lower_sha256(cell_id)
            or not path
            or cell_id in parsed
        ):
            raise ValueError("preflight proof paths must be unique CELL_ID=PATH values")
        parsed[cell_id] = path
    if len(parsed) != 8:
        raise ValueError("preflight proof paths must cover exactly eight cells")
    return parsed


def _qualify_formal_preflight_interference_cli(args: argparse.Namespace) -> int:
    """Locally control and deep-reopen the pulled exact-eight serving batch."""

    from lightcone_spec.experiments.formal_dispatch import (
        load_formal_preflight_dispatch_receipt,
    )
    from lightcone_spec.experiments.formal_preflight_execution import (
        qualify_formal_preflight_interference_locally,
    )

    token = load_formal_preflight_dispatch_receipt(
        args.dispatch_receipt,
        current_ns=args.now_ns,
    )
    proof = qualify_formal_preflight_interference_locally(
        token,
        remote_raw_receipt_path=args.remote_raw_receipt,
        native_result_proof_paths=_parse_preflight_cell_proof_paths(
            args.native_result_proof
        ),
        native_itl_proof_paths=_parse_preflight_cell_proof_paths(args.native_itl_proof),
        aggregate_control_attestation=ControlArtifactAttestation.from_dict(
            _load_bound_json(args.aggregate_control)
        ),
        replay_store=ChallengeReplayStore(args.control_replay_store),
        now_ns=args.now_ns,
        proof_artifact_path=args.output,
    )
    passed = proof.status == "PASSED"
    print(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "formal_preflight_local_interference_result",
                "status": (
                    "WAITING_FOR_EXACTNESS_AND_SUITE_PROOFS" if passed else "FAILED"
                ),
                "interference_proof_sha256": proof.artifact_sha256,
                "formal_coverage_complete": False,
            },
            sort_keys=True,
        )
    )
    return 42


def _parse_preflight_qualification_proofs(
    values: list[str],
) -> dict[str, tuple[str, str]]:
    from lightcone_spec.experiments.preflight_authority import (
        PREFLIGHT_REQUIRED_QUALIFICATION_SUITES,
    )

    parsed: dict[str, tuple[str, str]] = {}
    for value in values:
        suite, separator, paths = value.partition("=")
        result, comma, artifact = paths.partition(",")
        if (
            separator != "="
            or comma != ","
            or not suite
            or not result
            or not artifact
            or suite in parsed
        ):
            raise ValueError(
                "qualification proof must be unique SUITE=RESULT_POINTER,PROOF_ARTIFACT"
            )
        parsed[suite] = (result, artifact)
    if tuple(sorted(parsed)) != PREFLIGHT_REQUIRED_QUALIFICATION_SUITES:
        raise ValueError("qualification proofs do not cover the exact eight suites")
    return parsed


def _finalize_formal_preflight_evidence_cli(args: argparse.Namespace) -> int:
    """Deep-reduce qualified 1+1+8 sources into complete coverage."""

    from lightcone_spec.experiments.formal_dispatch import (
        FormalPreflightDispatchReceipt,
        load_formal_preflight_dispatch_receipt,
    )
    from lightcone_spec.experiments.formal_preflight_execution import (
        finalize_formal_preflight_evidence,
    )
    from lightcone_spec.runtime.proof_artifact import publish_canonical_json_no_replace

    token = load_formal_preflight_dispatch_receipt(
        args.dispatch_receipt,
        current_ns=args.now_ns,
    )
    durable_dispatch = FormalPreflightDispatchReceipt.from_dict(
        _load_bound_json(args.dispatch_receipt)
    )
    if durable_dispatch.revalidate(current_ns=args.now_ns).sha256 != token.sha256:
        raise ValueError(
            "formal preflight dispatch receipt changed during finalization"
        )
    replay_proofs = tuple(args.candidate_replay_proof)
    if len(replay_proofs) != 2 or len(set(replay_proofs)) != 2:
        raise ValueError("preflight finalization requires two candidate replay proofs")
    evidence = finalize_formal_preflight_evidence(
        token,
        remote_raw_receipt_path=args.remote_raw_receipt,
        exactness_result_path=args.exactness_result,
        interference_proof_artifact_path=args.interference_proof,
        qualification_proof_paths=_parse_preflight_qualification_proofs(
            args.qualification_proof
        ),
        materialization=durable_dispatch.signed_materialization.payload,
        candidate_state_coverage=tts_l0_candidate_state_coverage_from_dict(
            _load_bound_json(args.candidate_state_coverage)
        ),
        candidate_replay_proof_paths=(replay_proofs[0], replay_proofs[1]),
        now_ns=args.now_ns,
    )
    publish_canonical_json_no_replace(
        args.source_output,
        evidence.source_authority.to_dict(),
    )
    publish_canonical_json_no_replace(
        args.activation_output,
        registry_stage_activation_to_dict(evidence.activation),
    )
    publish_canonical_json_no_replace(
        args.coverage_output,
        evidence.coverage.to_dict(),
    )
    publish_canonical_json_no_replace(
        args.stage_coverage_output,
        stage_coverage_receipt_to_dict(evidence.stage_coverage),
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "formal_preflight_finalization_result",
                "status": "COMPLETE",
                "source_authority_sha256": evidence.source_authority.sha256,
                "activation_sha256": evidence.activation.sha256,
                "coverage_sha256": evidence.coverage.sha256,
                "stage_coverage_sha256": evidence.stage_coverage.sha256,
                "next_required_action": (
                    "externally_sign_stage_coverage_then_extend_formal_registry"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _build_industrial_registry(args: argparse.Namespace) -> int:
    logical_slots = tuple(args.logical_gpu_slot)
    if (
        not logical_slots
        or len(set(logical_slots)) != len(logical_slots)
        or any(
            not isinstance(slot, str)
            or not slot.strip()
            or "\n" in slot
            or "\r" in slot
            for slot in logical_slots
        )
    ):
        raise ValueError("logical GPU rank slots must be one or more unique names")
    registry_builder = (
        build_legacy_industrial_registry
        if args.legacy_diagnostic
        else build_industrial_registry
    )
    registry = registry_builder(
        gpu_uuids=logical_slots,
        base_port=args.base_port,
        cache_root=args.cache_root,
        evidence_root=args.evidence_root,
        seed=args.seed,
    )
    artifact = _industrial_registry_artifact(
        registry,
        base_port=args.base_port,
        cache_root=args.cache_root,
        evidence_root=args.evidence_root,
        seed=args.seed,
    )
    _write_json(args.output, artifact)
    print(registry.sha256)
    return 0


def _validate_e2_final_seal_authority(
    *,
    registry: ExperimentRegistry,
    reduction,
    inventory: GpuInventory,
    runtime_sha256: str,
    split_sha256: str,
    direct_dependency_receipt_sha256: str | None,
    completed_cell_ids: tuple[str, ...],
    completed_cells_path: str,
    locked_output_paths: dict[str, str],
) -> None:
    """Require the exact raw halving_3 candidate artifact before E2 sealing."""

    evidence = reduction.stage_evidence
    completed_value = _load_bound_json(completed_cells_path)
    if not isinstance(completed_value, dict) or not isinstance(
        completed_value.get("rows"), list
    ):
        raise TypeError("E2 completed-cell authority is malformed")
    measured_rows = tuple(
        row
        for row in completed_value["rows"]
        if isinstance(row, dict) and row.get("status") == "MEASURED"
    )
    terminal_receipts = tuple(
        sorted(str(row.get("terminal_receipt_sha256")) for row in measured_rows)
    )
    budget_observations = tuple(
        sorted(
            {
                str(row["budget_observation_sha256"])
                for row in measured_rows
                if row.get("budget_observation_sha256") is not None
            }
        )
    )
    if (
        reduction.registry_sha256 != registry.sha256
        or reduction.stage_index != 3
        or reduction.runtime_sha256 != runtime_sha256
        or reduction.split_sha256 != split_sha256
        or evidence.inventory_sha256 != inventory.sha256
        or evidence.inventory_source_receipt_sha256 != inventory.source_receipt_sha256
        or evidence.fixed_instance_gpu_count != len(inventory.devices)
        or evidence.inventory_host_id
        != (inventory.host_ids[0] if len(inventory.host_ids) == 1 else None)
        or reduction.activation.plan.dependency_receipt_sha256
        != direct_dependency_receipt_sha256
        or tuple(sorted(completed_cell_ids))
        != reduction.survivor_receipt.completed_stage_cell_ids
        or terminal_receipts != evidence.terminal_receipt_sha256s
        or budget_observations != evidence.budget_observation_sha256s
    ):
        raise ValueError("E2 final-stage authority differs from the sealing inputs")
    if set(locked_output_paths) != {"dflash_recipe"}:
        raise ValueError("E2 seal requires exactly one dflash_recipe=PATH output")
    expected = materialize_e2_final_recipe(registry, reduction)
    actual = e2_final_recipe_artifact_from_dict(
        _load_bound_json(locked_output_paths["dflash_recipe"])
    )
    if actual != expected:
        raise ValueError("dflash_recipe is not the raw halving_3 final candidate")


def _validate_preflight_interference_seal_authority(
    *,
    authority_path: str,
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    runtime_sha256: str,
    split_sha256: str,
    completed_cell_ids: tuple[str, ...],
    locked_output_paths: dict[str, str],
) -> None:
    """Reopen raw calibration terminals before sealing the runtime envelope."""

    if set(locked_output_paths) != {"runtime_envelope"}:
        raise ValueError(
            "preflight seal requires exactly one runtime_envelope=PATH output"
        )
    execution_authority = InterferenceCalibrationExecutionAuthority.from_dict(
        _load_bound_json(authority_path)
    )
    raw_authority = execution_authority.reconstruct()
    audit = raw_authority.audit_inputs()
    terminal_plans = tuple(
        terminal.plan for terminal in raw_authority.terminal_authorities
    )
    terminal_cell_ids = tuple(
        sorted(plan.runtime_plan.cell_id for plan in terminal_plans)
    )
    expected_interference_cell_ids = tuple(
        sorted(
            cell.cell_id
            for cell in registry.cells_for("preflight")
            if is_serving_interference_calibration_cell(cell)
        )
    )
    if (
        audit.inventory != inventory
        or terminal_cell_ids != expected_interference_cell_ids
        or not set(terminal_cell_ids) <= set(completed_cell_ids)
        or len(terminal_cell_ids) != len(set(terminal_cell_ids))
        or any(
            plan.dispatch_context.registry != registry
            or plan.dispatch_context.inventory != inventory
            or type(plan.dispatch_context.activation_artifact)
            is not RegistryStageActivationArtifact
            or plan.dispatch_context.activation_artifact.experiment != "preflight"
            or plan.dispatch_context.activation_artifact.runtime_sha256
            != runtime_sha256
            or plan.dispatch_context.activation_artifact.split_sha256 != split_sha256
            for plan in terminal_plans
        )
    ):
        raise ValueError(
            "preflight calibration authority differs from stage lineage or coverage"
        )
    expected = raw_authority.require_envelope()
    actual = _load_interference_envelope(locked_output_paths["runtime_envelope"])
    if actual != expected:
        raise ValueError(
            "preflight runtime envelope differs from raw calibration authority"
        )


def _validate_preflight_control_seal_authorities(
    *,
    coverage_receipt_path: str,
    coverage_attestation_path: str,
    capacity_gate_path: str,
    capacity_attestation_path: str,
    replay_store_path: str,
    registry: ExperimentRegistry,
    activation: RegistryStageActivationArtifact,
    inventory: GpuInventory,
    runtime_sha256: str,
    split_sha256: str,
    raw_completed_cells_sha256: str,
    completed_cell_ids: tuple[str, ...],
) -> PreflightSealControlBinding:
    """Consume exact signed coverage/capacity authority before receipt sealing."""

    if type(activation) is not RegistryStageActivationArtifact:
        raise TypeError("preflight seal requires exact registry activation")
    verify_registry_stage_activation(registry, activation)
    if (
        activation.experiment != "preflight"
        or activation.status != "AVAILABLE"
        or activation.runtime_sha256 != runtime_sha256
        or activation.split_sha256 != split_sha256
    ):
        raise ValueError("preflight activation differs from sealing lineage")
    activated_cell_ids = tuple(
        sorted(
            row.cell_id
            for row in activation.dispositions
            if row.status is RegistryStageDispositionStatus.ACTIVATED
        )
    )
    if activated_cell_ids != tuple(sorted(completed_cell_ids)):
        raise ValueError("preflight completed cells differ from full activation")
    coverage = PreflightCoverageReceipt.from_dict(
        _load_bound_json(coverage_receipt_path)
    )
    verify_preflight_coverage(registry, activation, coverage)
    require_complete_preflight_coverage(coverage)
    if (
        tuple(row.cell_id for row in coverage.terminals) != activated_cell_ids
        or coverage.runtime_sha256 != runtime_sha256
        or coverage.split_sha256 != split_sha256
    ):
        raise ValueError("preflight terminal aggregate differs from sealing inputs")
    capacity = StageCapacityGate.from_dict(_load_bound_json(capacity_gate_path))
    if (
        capacity.registry_sha256 != registry.sha256
        or capacity.experiment != "preflight"
        or capacity.activated_cell_ids != activated_cell_ids
        or capacity.status != "AVAILABLE"
    ):
        raise ValueError("preflight capacity gate is not available for this stage")
    if (
        len(inventory.devices) != 2
        or any(not device.ready for device in inventory.devices)
        or len({device.hardware_envelope_sha256 for device in inventory.devices}) != 1
    ):
        raise ValueError(
            "preflight control authority requires two ready homogeneous GPUs"
        )
    hardware_envelope_sha256 = inventory.devices[0].hardware_envelope_sha256
    coverage_attestation = ControlArtifactAttestation.from_dict(
        _load_bound_json(coverage_attestation_path)
    )
    capacity_attestation = ControlArtifactAttestation.from_dict(
        _load_bound_json(capacity_attestation_path)
    )
    expected_coverage_lineage = preflight_coverage_control_lineage_sha256(
        activation_sha256=activation.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        inventory_sha256=inventory.sha256,
        raw_completed_cells_sha256=raw_completed_cells_sha256,
    )
    expected_capacity_lineage = stage_capacity_control_lineage_sha256(
        activation_sha256=activation.sha256,
        inventory_sha256=inventory.sha256,
        gate=capacity,
    )
    for envelope, artifact_type, artifact_sha256, protocol_sha256, lineage in (
        (
            coverage_attestation,
            "rank_aggregate",
            coverage.sha256,
            PREFLIGHT_COVERAGE_PROTOCOL_SHA256,
            expected_coverage_lineage,
        ),
        (
            capacity_attestation,
            "capacity",
            capacity.sha256,
            STAGE_CAPACITY_GATE_PROTOCOL_SHA256,
            expected_capacity_lineage,
        ),
    ):
        subject = envelope.subject
        if (
            subject.artifact_type != artifact_type
            or subject.artifact_sha256 != artifact_sha256
            or subject.protocol_sha256 != protocol_sha256
            or subject.registry_sha256 != registry.sha256
            or subject.lineage_sha256 != lineage
            or envelope.hardware_envelope_sha256 != hardware_envelope_sha256
        ):
            raise ValueError("preflight control attestation subject is not exact")
    if (
        coverage_attestation.deployment_policy_authorization
        != capacity_attestation.deployment_policy_authorization
    ):
        raise ValueError("preflight control artifacts require one deployment policy")
    reserved_ns = time.time_ns()
    results = verify_and_reserve_release_control_artifact_attestations(
        (coverage_attestation, capacity_attestation),
        expected_inventory_sha256=inventory.sha256,
        now_ns=reserved_ns,
        replay_store=ChallengeReplayStore(
            str(Path(os.path.abspath(replay_store_path)))
        ),
    )
    coverage_result, capacity_result = results
    if (
        coverage_result.artifact_type != "rank_aggregate"
        or capacity_result.artifact_type != "capacity"
        or coverage_result.deployment_policy_authorization_sha256
        != capacity_result.deployment_policy_authorization_sha256
        or coverage_result.trust_bundle_sha256 != capacity_result.trust_bundle_sha256
        or coverage_result.trusted_attester_policy_sha256
        != capacity_result.trusted_attester_policy_sha256
    ):
        raise ValueError("preflight control verifier returned inconsistent authority")
    return PreflightSealControlBinding(
        schema_version=1,
        kind="formal_preflight_seal_control_binding",
        status="SEALED",
        registry_sha256=registry.sha256,
        activation_sha256=activation.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=hardware_envelope_sha256,
        raw_completed_cells_sha256=raw_completed_cells_sha256,
        coverage_receipt_sha256=coverage.sha256,
        coverage_attestation_sha256=coverage_attestation.sha256,
        capacity_gate_sha256=capacity.sha256,
        capacity_attestation_sha256=capacity_attestation.sha256,
        deployment_policy_authorization_sha256=(
            coverage_result.deployment_policy_authorization_sha256
        ),
        trust_bundle_sha256=coverage_result.trust_bundle_sha256,
        trusted_attester_policy_sha256=(coverage_result.trusted_attester_policy_sha256),
        replay_reservation_sha256=control_challenge_reservation_sha256(
            results, reserved_ns=reserved_ns
        ),
    )


def _materialize_stage_capacity_gate_cli(args: argparse.Namespace) -> int:
    """Derive a capacity gate solely from path-bound raw sources and schedule."""

    registry = _load_industrial_registry(args.registry)
    schedule = StageCapacitySchedule.from_dict(_load_bound_json(args.stage_schedule))
    gate = materialize_stage_capacity_gate_from_raw_sources(
        registry,
        experiment=schedule.experiment,
        activated_cell_ids=schedule.activated_cell_ids,
        source_manifest_path=os.path.abspath(args.capacity_source_manifest),
        schedule=schedule,
        now_ns=args.now_ns,
    )
    _write_json(args.output, gate.to_dict())
    reopened = StageCapacityGate.from_dict(_load_bound_json(args.output))
    if reopened != gate:
        raise RuntimeError("written stage capacity gate changed identity")
    revalidate_stage_capacity_gate_sources(
        registry,
        reopened,
        schedule=schedule,
        now_ns=args.now_ns,
    )
    print(gate.sha256)
    return 0 if gate.status == "AVAILABLE" else 42


def _seal_industrial_stage(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    registry.definition(args.experiment)
    if registry.materialization_mode != "legacy_diagnostic":
        raise ValueError(
            "seal-industrial-stage is legacy diagnostic only; signed_staged "
            "registries require the dynamic formal authorization bridge"
        )
    if _TRUSTED_HARDWARE_ATTESTER_ID is None and args.experiment != "preflight":
        artifact = {
            "schema_version": 1,
            "kind": "industrial_stage_seal_decision",
            "status": "BLOCKED",
            "gpu_evidence": "UNMEASURED",
            "reason_code": "trusted_hardware_attester_unavailable",
            "registry_sha256": registry.sha256,
            "experiment": args.experiment,
            "trusted_attester_id": None,
            "execution_mode": "diagnostic_non_authorizing",
            "formal_dispatch_authorized": False,
        }
        _write_json(args.output, artifact)
        print(_canonical_sha256(artifact))
        return 42
    dependencies = _load_industrial_receipts(args.dependency_receipt)
    inventory = _load_gpu_inventory(args.inventory)
    activation_artifact = _load_stage_activation_plan(args.activation_plan)
    family_activations = _load_family_activations(args.family_activation)
    family_power_reductions = _load_family_power_reductions(args.family_power_plan)
    dependency_by_experiment = registry.validate_receipts(dependencies)
    definition = registry.definition(args.experiment)
    direct_dependency_sha256 = (
        None
        if not definition.dependencies
        else dependency_by_experiment[definition.dependencies[-1]].sha256
    )
    runtime_sha256 = _artifact_sha256(args.runtime_artifact)
    split_contract = _load_bound_json(args.split_artifact)
    split_sha256 = _canonical_sha256(split_contract)
    completed_cell_ids, completed_sha256 = _completed_industrial_cells(
        args.completed_cells,
        registry,
        experiment=args.experiment,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        split_contract=split_contract,
        require_industrial_contract=True,
        direct_dependency_receipt_sha256=direct_dependency_sha256,
        activation_artifact=activation_artifact,
        family_activations=family_activations,
        family_power_reductions=family_power_reductions,
        require_stage_sealable=True,
        inventory=inventory,
    )
    if completed_sha256 is None:
        raise ValueError("stage sealing requires content-bound completed-cell evidence")
    receipt_completed_sha256 = completed_sha256
    locked_output_paths = _parse_locked_output_paths(args.locked_output)
    if args.experiment == "preflight":
        required_preflight_paths = {
            "interference calibration authority": (
                args.interference_calibration_authority
            ),
            "preflight coverage receipt": args.preflight_coverage_receipt,
            "preflight coverage attestation": (args.preflight_coverage_attestation),
            "stage capacity gate": args.stage_capacity_gate,
            "stage capacity attestation": args.stage_capacity_attestation,
            "control replay store": args.control_replay_store,
            "preflight control binding output": (args.preflight_control_binding_output),
        }
        missing = tuple(
            label for label, path in required_preflight_paths.items() if path is None
        )
        if missing:
            raise ValueError(
                "preflight seal lacks mandatory control authority: " + ",".join(missing)
            )
        if type(activation_artifact) is not RegistryStageActivationArtifact:
            raise TypeError("preflight seal requires --activation-plan")
        assert args.interference_calibration_authority is not None
        assert args.preflight_coverage_receipt is not None
        assert args.preflight_coverage_attestation is not None
        assert args.stage_capacity_gate is not None
        assert args.stage_capacity_attestation is not None
        assert args.control_replay_store is not None
        assert args.preflight_control_binding_output is not None
        _validate_preflight_interference_seal_authority(
            authority_path=args.interference_calibration_authority,
            registry=registry,
            inventory=inventory,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            completed_cell_ids=completed_cell_ids,
            locked_output_paths=locked_output_paths,
        )
        control_binding = _validate_preflight_control_seal_authorities(
            coverage_receipt_path=args.preflight_coverage_receipt,
            coverage_attestation_path=args.preflight_coverage_attestation,
            capacity_gate_path=args.stage_capacity_gate,
            capacity_attestation_path=args.stage_capacity_attestation,
            replay_store_path=args.control_replay_store,
            registry=registry,
            activation=activation_artifact,
            inventory=inventory,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            raw_completed_cells_sha256=completed_sha256,
            completed_cell_ids=completed_cell_ids,
        )
        _write_json(
            args.preflight_control_binding_output,
            control_binding.to_dict(),
        )
        receipt_completed_sha256 = _artifact_sha256(
            args.preflight_control_binding_output
        )
    elif any(
        value is not None
        for value in (
            args.interference_calibration_authority,
            args.preflight_coverage_receipt,
            args.preflight_coverage_attestation,
            args.stage_capacity_gate,
            args.stage_capacity_attestation,
            args.control_replay_store,
            args.preflight_control_binding_output,
        )
    ):
        raise ValueError("preflight control authority cannot seal another experiment")
    if args.experiment == "E2":
        if args.e2_final_stage_manifest is None:
            raise ValueError("E2 seal requires --e2-final-stage-manifest")
        _validate_e2_final_seal_authority(
            registry=registry,
            reduction=_load_e2_stage_manifest(args.e2_final_stage_manifest),
            inventory=inventory,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            direct_dependency_receipt_sha256=direct_dependency_sha256,
            completed_cell_ids=completed_cell_ids,
            completed_cells_path=args.completed_cells,
            locked_output_paths=locked_output_paths,
        )
    elif args.e2_final_stage_manifest is not None:
        raise ValueError("E2 final-stage authority cannot seal another experiment")
    receipt = registry.make_receipt(
        args.experiment,
        {name: _artifact_sha256(path) for name, path in locked_output_paths.items()},
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        completed_cells_sha256=receipt_completed_sha256,
        dependencies=dependencies,
    )
    _write_json(args.output, receipt.to_dict())
    print(receipt.sha256)
    return 0


def _completed_industrial_cells(
    path: str | None,
    registry: ExperimentRegistry,
    *,
    experiment: str | None = None,
    runtime_sha256: str | None = None,
    split_sha256: str | None = None,
    split_contract: object | None = None,
    require_industrial_contract: bool = False,
    direct_dependency_receipt_sha256: str | None = None,
    activation_artifact=None,
    family_activations=(),
    family_power_reductions=(),
    require_stage_sealable: bool = False,
    inventory: GpuInventory | None = None,
) -> tuple[tuple[str, ...], str | None]:
    if path is None:
        return (), None
    value = _load_bound_json(path)
    if not isinstance(value, dict):
        raise TypeError("completed-cell artifact must be an object")
    schema_version = value.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in {1, 2, 3, 4}
        or value.get("kind") != "industrial_completed_cells"
        or value.get("registry_sha256") != registry.sha256
    ):
        raise ValueError("completed-cell artifact identity mismatch")
    strict = schema_version in {2, 3, 4}
    formal = schema_version == 4
    if require_industrial_contract and not formal:
        raise ValueError(
            "stage sealing requires a schema-version-4 inventory-bound contract"
        )
    if formal and inventory is None:
        raise ValueError("formal completion requires a bound GPU inventory")

    known = {cell.cell_id: cell for cell in registry.cells}
    contract_by_cell: dict[str, dict] = {}
    stage: str | None = None
    activated: tuple[str, ...] = ()
    dispositions: dict[str, dict[str, str]] = {}
    activation_binding: dict[str, object] | None = None
    if strict:
        stage = value.get("experiment")
        embedded_split = value.get("split_contract")
        if (
            not isinstance(stage, str)
            or not stage
            or value.get("runtime_sha256") is None
            or not _is_lower_sha256(value.get("runtime_sha256"))
            or not _is_lower_sha256(value.get("split_sha256"))
            or not isinstance(embedded_split, dict)
            or _canonical_sha256(embedded_split) != value.get("split_sha256")
        ):
            raise ValueError("industrial completion contract identity is incomplete")
        if experiment is not None and stage != experiment:
            raise ValueError("completed cells belong to another industrial stage")
        if runtime_sha256 is not None and value["runtime_sha256"] != runtime_sha256:
            raise ValueError("completed cells bind another runtime artifact")
        if split_sha256 is not None and value["split_sha256"] != split_sha256:
            raise ValueError("completed cells bind another locked split")
        if split_contract is not None and embedded_split != split_contract:
            raise ValueError("embedded locked split differs from the sealing artifact")
        if formal:
            expected_completion_fields = {
                "schema_version",
                "kind",
                "registry_sha256",
                "experiment",
                "runtime_sha256",
                "split_sha256",
                "split_contract",
                "activation_binding",
                "inventory_sha256",
                "inventory_source_receipt_sha256",
                "rows",
            }
            if set(value) != expected_completion_fields:
                raise ValueError("formal completion fields differ from schema")
            if inventory is None:  # pragma: no cover - guarded above
                raise RuntimeError("formal completion lost its GPU inventory")
            if (
                value.get("inventory_sha256") != inventory.sha256
                or value.get("inventory_source_receipt_sha256")
                != inventory.source_receipt_sha256
            ):
                raise ValueError("formal completion binds another GPU inventory")
            activated, dispositions, activation_binding = (
                _industrial_completion_activation_contract(
                    registry,
                    experiment=stage,
                    runtime_sha256=value["runtime_sha256"],
                    split_sha256=value["split_sha256"],
                    direct_dependency_receipt_sha256=(direct_dependency_receipt_sha256),
                    activation_artifact=activation_artifact,
                    family_activations=family_activations,
                    family_power_reductions=family_power_reductions,
                    require_stage_sealable=require_stage_sealable,
                )
            )
            if value.get("activation_binding") != activation_binding:
                raise ValueError("completion activation binding is missing or forged")
        if (
            not isinstance(embedded_split.get("schema_version"), int)
            or isinstance(embedded_split.get("schema_version"), bool)
            or embedded_split.get("schema_version") != 1
            or embedded_split.get("kind") != "industrial_locked_split"
            or embedded_split.get("registry_sha256") != registry.sha256
            or embedded_split.get("experiment") != stage
        ):
            raise ValueError("industrial locked-split identity mismatch")
        if formal and set(embedded_split) != {
            "schema_version",
            "kind",
            "registry_sha256",
            "experiment",
            "cells",
        }:
            raise ValueError("formal locked-split fields differ from schema")
        split_cells = embedded_split.get("cells")
        if not isinstance(split_cells, list) or not all(
            isinstance(row, dict) for row in split_cells
        ):
            raise TypeError("industrial locked split lacks cell contracts")
        runnable = {
            cell.cell_id: cell for cell in registry.cells_for(stage) if cell.runnable
        }
        materialized = (
            {cell_id: runnable[cell_id] for cell_id in activated}
            if formal
            else runnable
        )
        for contract in split_cells:
            cell_id = contract.get("cell_id")
            if cell_id not in materialized or cell_id in contract_by_cell:
                raise ValueError("locked split contains an unknown or duplicate cell")
            cell = materialized[str(cell_id)]
            if formal:
                expected_contract_fields = {
                    "cell_id",
                    "request_ids",
                    "expected_request_rows",
                    "expected_round_rows",
                    "expected_update_rows",
                    "expected_performance_rows",
                    "request_ids_sha256",
                    "corpus_sha256",
                    "arrival_trace_sha256",
                    "sampling_profile_sha256",
                    "model_lock_sha256",
                    "patched_sglang_tree",
                    "workload_contract",
                    "rank_config_sha256s",
                    "physical_assignment",
                    "physical_binding_sha256",
                    "topology_receipt_sha256",
                    "experiment_budget_sha256",
                    "experiment_budget",
                    "execution_plan_sha256",
                    "execution_split_sha256",
                }
                if set(contract) != expected_contract_fields:
                    raise ValueError("locked split cell fields differ from schema")
                assignment = _industrial_physical_assignment_from_dict(
                    contract.get("physical_assignment")
                )
                if inventory is None:  # pragma: no cover - guarded above
                    raise RuntimeError("formal completion lost its GPU inventory")
                _validate_assignment_inventory_authority(assignment, inventory)
                try:
                    experiment_budget = experiment_budget_from_dict(
                        contract.get("experiment_budget")
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "locked split contains a forged ExperimentBudget"
                    ) from error
                if (
                    contract.get("physical_binding_sha256") != assignment.sha256
                    or contract.get("experiment_budget_sha256")
                    != assignment.experiment_budget_sha256
                    or contract.get("experiment_budget_sha256")
                    != experiment_budget.sha256
                    or not _is_lower_sha256(contract.get("topology_receipt_sha256"))
                    or not _is_lower_sha256(contract.get("execution_plan_sha256"))
                    or not _is_lower_sha256(contract.get("execution_split_sha256"))
                ):
                    raise ValueError(
                        "locked split physical assignment/budget binding is invalid"
                    )
                if (
                    experiment_budget.cell_id != cell.cell_id
                    or experiment_budget.experiment != cell.identity.experiment
                    or experiment_budget.method != cell.identity.method
                    or experiment_budget.workload_class
                    is not cell.resources.workload_class
                    or experiment_budget.gpu_count != cell.resources.gpu_count
                    or experiment_budget.topology != cell.identity.topology
                    or experiment_budget.measured_gpu_ms is not None
                    or experiment_budget.fixed_instance_billed_gpu_ms
                    != experiment_budget.wall_time.scale(len(inventory.devices))
                ):
                    raise ValueError(
                        "locked ExperimentBudget differs from its registry cell"
                    )
                expected_topology = {
                    "tp1_dp1": (1, 1),
                    "tp2_dp1": (2, 1),
                    "two_replica_tp1_dp2": (1, 2),
                    "two_gpu_host": (1, 2),
                    "two_independent_tp1": (1, 2),
                }.get(cell.identity.topology)
                if expected_topology is None or expected_topology != (
                    assignment.tensor_parallel_size,
                    assignment.data_parallel_size,
                ):
                    raise ValueError(
                        "physical assignment disagrees with registry topology"
                    )
            request_ids = contract.get("request_ids")
            expected_request_rows = contract.get("expected_request_rows")
            if (
                not isinstance(request_ids, list)
                or not request_ids
                or not all(isinstance(item, str) and item for item in request_ids)
                or len(request_ids) != len(set(request_ids))
                or not isinstance(expected_request_rows, int)
                or isinstance(expected_request_rows, bool)
                or expected_request_rows != len(request_ids)
                or contract.get("request_ids_sha256") != _canonical_sha256(request_ids)
            ):
                raise ValueError("locked split request coverage is invalid")
            for name in (
                "corpus_sha256",
                "arrival_trace_sha256",
                "request_ids_sha256",
                "sampling_profile_sha256",
                "model_lock_sha256",
            ):
                if not _is_lower_sha256(contract.get(name)):
                    raise ValueError(f"locked split {name} is invalid")
            rank_config_sha256s = contract.get("rank_config_sha256s")
            calibration = is_serving_interference_calibration_cell(cell)
            if stage == "preflight" and not calibration:
                if rank_config_sha256s is not None:
                    raise ValueError(
                        "preflight split cannot claim serving RunConfig identities"
                    )
            elif (
                not isinstance(rank_config_sha256s, list)
                or len(rank_config_sha256s)
                != (
                    len(assignment.gpu_uuids)
                    if formal
                    else len(cell.resources.gpu_uuids)
                )
                or not all(_is_lower_sha256(item) for item in rank_config_sha256s)
            ):
                raise ValueError(
                    "serving split must bind one rank-config digest per rank"
                )
            if contract.get("patched_sglang_tree") != PINNED_SGLANG_TREE:
                raise ValueError("locked split uses another patched SGLang tree")
            expected_workload = (
                f"industrial_preflight_{cell.identity.method}"
                if stage == "preflight"
                else (
                    f"industrial_{cell.identity.method}"
                    if cell.identity.method in {"target_only", "static"}
                    else "industrial_adapted"
                )
            )
            if contract.get("workload_contract") != expected_workload:
                raise ValueError(
                    "locked split workload contract disagrees with its cell"
                )
            counts = {
                name: contract.get(name)
                for name in (
                    "expected_round_rows",
                    "expected_update_rows",
                    "expected_performance_rows",
                )
            }
            if (
                any(
                    not isinstance(count, int) or isinstance(count, bool) or count < 0
                    for count in counts.values()
                )
                or counts["expected_performance_rows"] < 1
            ):
                raise ValueError("locked split evidence row counts are invalid")
            adapted = cell.identity.method not in {"target_only", "static"}
            requires_rounds = adapted
            requires_updates = adapted
            if (counts["expected_round_rows"] > 0) is not requires_rounds or (
                counts["expected_update_rows"] > 0
            ) is not requires_updates:
                raise ValueError(
                    "locked split table coverage disagrees with the method"
                )
            contract_by_cell[str(cell_id)] = contract
        if set(contract_by_cell) != set(materialized):
            raise ValueError("locked split does not cover every activated stage cell")

    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError("completed-cell rows are missing")
    if strict:
        if stage is None:
            raise RuntimeError("strict completion lost its stage identity")
        stage_ids = {cell.cell_id for cell in registry.cells_for(stage)}
        if any(row.get("cell_id") not in stage_ids for row in rows):
            raise ValueError("completed-cell rows cross an industrial stage boundary")
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("cell_id")), []).append(row)
        if set(grouped) != stage_ids:
            raise ValueError(
                "completion outcomes do not cover every declared stage cell"
            )
        for cell_id, cell_rows in grouped.items():
            cell = known[cell_id]
            if formal and cell_id not in set(activated):
                if cell_rows != [dispositions[cell_id]]:
                    raise ValueError(
                        "non-activated cells require one exact immutable disposition"
                    )
                continue
            if cell.runnable:
                expected_world_size = (
                    len(
                        _industrial_physical_assignment_from_dict(
                            contract_by_cell[cell_id]["physical_assignment"]
                        ).gpu_uuids
                    )
                    if formal
                    else len(cell.resources.gpu_uuids)
                )
                if len(cell_rows) != expected_world_size or any(
                    row.get("status") != "MEASURED" for row in cell_rows
                ):
                    raise ValueError(
                        "activated cells require one measured outcome per physical rank"
                    )
                continue
            expected_disposition = (
                dispositions[cell_id]
                if formal
                else {
                    "cell_id": cell_id,
                    "status": cell.status.value,
                    "reason_code": cell.reason_code,
                    "reason": cell.reason,
                }
            )
            if cell_rows != [expected_disposition]:
                raise ValueError(
                    "BLOCKED/N/A cells require one exact immutable outcome row"
                )

    completed: list[str] = []
    rank_coverage: dict[str, set[int]] = {}
    consensus: dict[str, dict[str, object]] = {}
    budget_consensus: dict[str, tuple[object, object, object]] = {}
    nonce_owner: dict[str, str] = {}
    run_owner: dict[str, str] = {}
    for row in rows:
        cell_id = row.get("cell_id")
        if (
            strict
            and cell_id in known
            and (
                not known[str(cell_id)].runnable
                or (formal and cell_id not in set(activated))
            )
        ):
            continue
        evidence_root = row.get("evidence_root")
        run_id = row.get("run_id")
        rank = row.get("rank")
        assignment = None
        row_cell = known.get(str(cell_id))
        physical_world_size = (
            0 if row_cell is None else len(row_cell.resources.gpu_uuids)
        )
        if formal and cell_id in contract_by_cell:
            formal_row_fields = {
                "cell_id",
                "evidence_root",
                "run_id",
                "rank",
                "evidence_sha256",
                "terminal_receipt_sha256",
                "physical_gpu_uuid",
                "physical_binding_sha256",
                "experiment_budget_sha256",
                "budget_observation_status",
                "budget_observation_reason_code",
                "budget_observation_path",
                "budget_observation_sha256",
                "preflight_attestation_path",
                "preflight_attestation_sha256",
                "status",
            }
            if set(row) != formal_row_fields:
                raise ValueError("formal completed-rank fields differ from schema")
            assignment = _industrial_physical_assignment_from_dict(
                contract_by_cell[str(cell_id)]["physical_assignment"]
            )
            physical_world_size = len(assignment.gpu_uuids)
        if (
            not _is_lower_sha256(cell_id)
            or cell_id not in known
            or not isinstance(evidence_root, str)
            or not evidence_root
            or not isinstance(run_id, str)
            or not run_id
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 0
            or rank >= physical_world_size
            or not _is_lower_sha256(row.get("evidence_sha256"))
            or not _is_lower_sha256(row.get("terminal_receipt_sha256"))
            or row.get("status") != "MEASURED"
        ):
            raise ValueError(
                "completed cells require known identities and durable measured evidence"
            )
        cell = known[str(cell_id)]
        if formal:
            if assignment is None:
                raise RuntimeError("formal completion lost its physical assignment")
            contract = contract_by_cell[str(cell_id)]
            if (
                row.get("physical_binding_sha256") != assignment.sha256
                or row.get("physical_gpu_uuid") != assignment.gpu_uuids[rank]
                or row.get("experiment_budget_sha256")
                != assignment.experiment_budget_sha256
                or row.get("experiment_budget_sha256")
                != contract["experiment_budget_sha256"]
            ):
                raise ValueError(
                    "completed rank differs from its physical assignment/budget"
                )
        owner = run_owner.setdefault(run_id, str(cell_id))
        if strict and owner != cell_id:
            raise ValueError("industrial run identity is reused across registry cells")
        root = Path(evidence_root)
        if strict and root.resolve() != Path(cell.resources.evidence_root).resolve():
            raise ValueError(
                "completed-cell evidence root differs from its resource claim"
            )
        evidence = load_completed_evidence(root, run_id=run_id, rank=rank)
        if evidence is None:
            raise ValueError("completed cell has no valid terminal evidence receipt")
        receipt_path = root / f"{run_id}.rank{rank}.complete.json"
        if _file_sha256(receipt_path) != row["terminal_receipt_sha256"]:
            raise ValueError("completed-cell terminal receipt digest mismatch")
        try:
            terminal_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("completed-cell terminal receipt is invalid") from error
        if not isinstance(terminal_receipt, dict):
            raise TypeError("completed-cell terminal receipt is malformed")
        if evidence_files_sha256(evidence.values()) != row["evidence_sha256"]:
            raise ValueError("completed-cell evidence digest mismatch")
        run_columns = ["manifest_sha256", "config_sha256", "status"]
        if strict:
            run_columns.extend(
                (
                    "method",
                    "model_pair",
                    "repetition_block",
                    "started_ns",
                    "industrial_cell_id",
                    "rank_config_sha256",
                    "runtime_sha256",
                    "split_sha256",
                    "corpus_sha256",
                    "arrival_trace_sha256",
                    "request_ids_sha256",
                    "sampling_profile_sha256",
                    "model_lock_sha256",
                    "patched_sglang_tree",
                    "run_nonce_sha256",
                    "topology_sha256",
                    "tensor_parallel_size",
                    "data_parallel_size",
                    "world_size",
                    "rank",
                    "expected_request_rows",
                    "expected_round_rows",
                    "expected_update_rows",
                    "expected_performance_rows",
                    "workload_contract",
                    "experiment_budget_sha256",
                    "preflight_attestation_sha256",
                    *_DISABLED_SESSION_RUN_FIELDS,
                )
            )
            run_schema_names = pq.ParquetFile(evidence["run"]).schema_arrow.names
            for field in run_schema_names:
                if field.startswith("session_") and field not in run_columns:
                    run_columns.append(field)
        run = pq.read_table(evidence["run"], columns=run_columns).to_pylist()
        if len(run) != 1:
            raise ValueError("completed evidence must contain exactly one run identity")
        run_row = run[0]
        if any(
            run_row.get(name) != expected
            for name, expected in (
                ("manifest_sha256", registry.sha256),
                ("config_sha256", cell_id),
                ("status", "complete"),
            )
        ):
            raise ValueError("completed evidence is not bound to its registry cell")
        contract: dict | None = None
        topology_sha256: str | None = None
        if strict:
            _validate_disabled_session_run_fields(run_row)
            contract = contract_by_cell[str(cell_id)]
            topology = cell.identity.topology
            if formal:
                if assignment is None:
                    raise RuntimeError("formal completion lost its physical assignment")
                tensor_parallel_size = assignment.tensor_parallel_size
                data_parallel_size = assignment.data_parallel_size
                world_size = len(assignment.gpu_uuids)
                topology_sha256 = _canonical_sha256(
                    {
                        "schema_version": 1,
                        "cell_id": cell_id,
                        "topology": topology,
                        "topology_receipt_sha256": contract["topology_receipt_sha256"],
                        "physical_assignment_sha256": (assignment.assignment_sha256),
                        "physical_binding_sha256": assignment.sha256,
                        "physical_host_id": assignment.host_id,
                        "physical_gpu_uuids": list(assignment.gpu_uuids),
                        "physical_rank_groups": [
                            list(group) for group in assignment.rank_groups
                        ],
                        "physical_ports": list(assignment.ports),
                        "topology_group_ids": [
                            list(group) for group in assignment.topology_group_ids
                        ],
                        "tensor_parallel_size": tensor_parallel_size,
                        "data_parallel_size": data_parallel_size,
                        "world_size": world_size,
                    }
                )
            else:
                tensor_parallel_size = 2 if topology == "tp2_dp1" else 1
                data_parallel_size = 2 if topology == "two_replica_tp1_dp2" else 1
                world_size = len(cell.resources.gpu_uuids)
                topology_sha256 = _canonical_sha256(
                    {
                        "schema_version": 1,
                        "cell_id": cell_id,
                        "topology": topology,
                        "gpu_uuids": list(cell.resources.gpu_uuids),
                        "tensor_parallel_size": tensor_parallel_size,
                        "data_parallel_size": data_parallel_size,
                        "world_size": world_size,
                    }
                )
            expected_run = {
                "method": cell.identity.method,
                "model_pair": cell.identity.model,
                "repetition_block": cell.identity.block,
                "industrial_cell_id": cell_id,
                "rank_config_sha256": (
                    None
                    if stage == "preflight" and not calibration
                    else contract["rank_config_sha256s"][rank]
                ),
                "runtime_sha256": (
                    contract["execution_plan_sha256"]
                    if formal
                    else value["runtime_sha256"]
                ),
                "split_sha256": (
                    contract["execution_split_sha256"]
                    if formal
                    else value["split_sha256"]
                ),
                "corpus_sha256": contract["corpus_sha256"],
                "arrival_trace_sha256": contract["arrival_trace_sha256"],
                "request_ids_sha256": contract["request_ids_sha256"],
                "sampling_profile_sha256": contract["sampling_profile_sha256"],
                "model_lock_sha256": contract["model_lock_sha256"],
                "patched_sglang_tree": PINNED_SGLANG_TREE,
                "topology_sha256": topology_sha256,
                "tensor_parallel_size": tensor_parallel_size,
                "data_parallel_size": data_parallel_size,
                "world_size": world_size,
                "rank": rank,
                "expected_round_rows": contract["expected_round_rows"],
                "expected_update_rows": contract["expected_update_rows"],
                "expected_performance_rows": contract["expected_performance_rows"],
                "workload_contract": contract["workload_contract"],
                "experiment_budget_sha256": (
                    contract["experiment_budget_sha256"] if formal else None
                ),
            }
            closed_loop = str(cell.identity.arrival).startswith("closed_loop")
            if not closed_loop:
                expected_run["expected_request_rows"] = contract[
                    "expected_request_rows"
                ]
            if any(
                run_row.get(name) != expected for name, expected in expected_run.items()
            ):
                raise ValueError(
                    "industrial run identity differs from its locked contract"
                )
            if not _is_lower_sha256(run_row.get("run_nonce_sha256")):
                raise ValueError("industrial run lacks a content-bound nonce")
            nonce = str(run_row["run_nonce_sha256"])
            owner = nonce_owner.setdefault(nonce, str(cell_id))
            if owner != cell_id:
                raise ValueError("industrial run nonce is reused across registry cells")
            if cell.identity.experiment == "preflight" and not calibration:
                attestation_path = row.get("preflight_attestation_path")
                if not isinstance(attestation_path, str) or not attestation_path:
                    raise ValueError("preflight completion lacks an attestation path")
                attestation_source = Path(attestation_path)
                attestation_sidecar = Path(f"{attestation_source}.sha256")
                if (
                    not attestation_source.is_file()
                    or attestation_source.is_symlink()
                    or not attestation_sidecar.is_file()
                    or attestation_sidecar.is_symlink()
                    or attestation_source.resolve().parent != root.resolve()
                ):
                    raise ValueError(
                        "preflight attestation must live in its evidence root"
                    )
                attestation = _load_bound_json(attestation_source)
                attestation_sha256 = _canonical_sha256(attestation)
                if (
                    not isinstance(attestation, dict)
                    or row.get("preflight_attestation_sha256") != attestation_sha256
                    or run_row.get("preflight_attestation_sha256") != attestation_sha256
                ):
                    raise ValueError("preflight attestation digest mismatch")
                required_checks = {
                    "environment_and_patch_preflight": {
                        "identity",
                        "environment",
                        "patch_apply",
                        "compile",
                        "patch_tests",
                        "compatibility",
                    },
                    "exactness_memory_telemetry_preflight": {
                        "exactness",
                        "memory",
                        "telemetry",
                        "target_only_allocation",
                        "static_allocation",
                    },
                    "simultaneous_single_gpu_interference": {
                        "isolated",
                        "simultaneous",
                        "hardware",
                        "paired_blocks",
                    },
                }[cell.identity.task]
                checks = attestation.get("checks")
                source_files = attestation.get("source_files")
                if (
                    not isinstance(attestation.get("schema_version"), int)
                    or isinstance(attestation.get("schema_version"), bool)
                    or attestation.get("schema_version") != 1
                    or attestation.get("kind") != "industrial_preflight_attestation"
                    or attestation.get("status") != "PASS"
                    or attestation.get("registry_sha256") != registry.sha256
                    or attestation.get("cell_id") != cell_id
                    or attestation.get("runtime_sha256") != value["runtime_sha256"]
                    or attestation.get("split_sha256") != value["split_sha256"]
                    or attestation.get("run_nonce_sha256")
                    != run_row["run_nonce_sha256"]
                    or attestation.get("topology_sha256") != topology_sha256
                    or not isinstance(attestation.get("rank"), int)
                    or isinstance(attestation.get("rank"), bool)
                    or attestation.get("rank") != rank
                    or attestation.get("gpu_uuid")
                    != (
                        assignment.gpu_uuids[rank]
                        if formal and assignment is not None
                        else cell.resources.gpu_uuids[rank]
                    )
                    or not isinstance(checks, dict)
                    or set(checks) != required_checks
                    or any(result != "PASS" for result in checks.values())
                    or not isinstance(source_files, list)
                    or not source_files
                    or not all(isinstance(item, str) and item for item in source_files)
                    or len(source_files) != len(set(source_files))
                    or not _is_lower_sha256(attestation.get("source_evidence_sha256"))
                    or evidence_files_sha256(source_files)
                    != attestation.get("source_evidence_sha256")
                ):
                    raise ValueError("preflight attestation contract is incomplete")
                resolved_root = root.resolve()
                source_paths = tuple(Path(item) for item in source_files)
                if any(
                    not source.is_file()
                    or source.is_symlink()
                    or (
                        source.resolve() != resolved_root
                        and resolved_root not in source.resolve().parents
                    )
                    for source in source_paths
                ):
                    raise ValueError(
                        "preflight source evidence must be regular files in its root"
                    )
            elif (
                run_row.get("preflight_attestation_sha256") is not None
                or row.get("preflight_attestation_path") is not None
                or row.get("preflight_attestation_sha256") is not None
            ):
                raise ValueError(
                    "non-preflight completion carries a preflight attestation"
                )

            if formal:
                if assignment is None:
                    raise RuntimeError("formal completion lost its physical assignment")
                non_serving = (
                    stage == "preflight" and not calibration
                ) or cell.resources.workload_class.value in {"compile", "download"}
                budget_status = row.get("budget_observation_status")
                budget_reason = row.get("budget_observation_reason_code")
                budget_path = row.get("budget_observation_path")
                budget_sha256 = row.get("budget_observation_sha256")
                if non_serving:
                    if (
                        budget_status != "NOT_APPLICABLE"
                        or budget_reason != "preflight_or_non_serving_execution"
                        or budget_path is not None
                        or budget_sha256 is not None
                    ):
                        raise ValueError(
                            "non-serving budget observation disposition is not exact"
                        )
                else:
                    expected_budget_status = (
                        "OBSERVED" if rank == 0 else "BOUND_TO_RANK0"
                    )
                    expected_budget_reason = (
                        None if rank == 0 else "gang_observation_published_by_rank0"
                    )
                    if (
                        budget_status != expected_budget_status
                        or budget_reason != expected_budget_reason
                        or not isinstance(budget_path, str)
                        or not _is_lower_sha256(budget_sha256)
                    ):
                        raise ValueError(
                            "serving completion lacks its exact budget observation"
                        )
                    budget_identity = (
                        budget_path,
                        budget_sha256,
                        row["experiment_budget_sha256"],
                    )
                    previous_budget = budget_consensus.setdefault(
                        str(cell_id), budget_identity
                    )
                    if previous_budget != budget_identity:
                        raise ValueError(
                            "cell ranks disagree on one budget observation"
                        )
                    if rank == 0:
                        _validate_budget_observation_receipt(
                            budget_path,
                            expected_sha256=budget_sha256,
                            experiment_budget_sha256=row["experiment_budget_sha256"],
                            prepared_receipt_sha256=terminal_receipt.get(
                                "prepared_receipt_sha256"
                            ),
                            cell=cell,
                            evidence_root=root,
                            fixed_instance_gpu_count=(len(inventory.devices)),
                        )

        performance = pq.read_table(
            evidence["performance"],
            columns=[
                "region",
                "method",
                "optimizer_bytes",
                "adaptation_memory_ledger",
                "trainable_parameters",
                "training_cuda_ms",
                "optimizer_cuda_ms",
                "merge_cuda_ms",
                "publish_cuda_ms",
                "barrier_cuda_ms",
                "exposed_update_ms",
                "main_side_overlap_ratio",
                "updates_launched",
                "updates_published",
                "exactness_violations",
                "version_mismatches",
                "fallbacks",
                "nonfinite_updates",
                "oom_events",
                "retractions",
                "communicator_failures",
                "admission_rejections",
                "timeouts",
                "cancellations",
                "offered_requests",
                "admitted_requests",
                "completed_requests",
                "unfinished_requests",
                "evidence_backpressure_events",
                "evidence_dropped_rows",
            ],
        ).to_pylist()
        if (
            not performance
            or (strict and len(performance) != contract["expected_performance_rows"])
            or any(
                row["method"] != cell.identity.method
                or row["evidence_dropped_rows"] != 0
                for row in performance
            )
        ):
            raise ValueError(
                "industrial completion requires nonempty, lossless performance evidence"
            )
        if cell.identity.task != "failure_injection" and any(
            any(
                row[field] != 0
                for field in (
                    "exactness_violations",
                    "version_mismatches",
                    "fallbacks",
                    "nonfinite_updates",
                    "oom_events",
                    "retractions",
                    "communicator_failures",
                )
            )
            for row in performance
        ):
            raise ValueError("industrial completion contains a safety violation")
        if strict and any(
            not isinstance(row["evidence_backpressure_events"], int)
            or isinstance(row["evidence_backpressure_events"], bool)
            or row["evidence_backpressure_events"] < 0
            for row in performance
        ):
            raise ValueError("industrial completion lacks backpressure accounting")
        if cell.identity.method in {"target_only", "static"}:
            for performance_row in performance:
                _validate_allocation_free_performance(
                    performance_row,
                    method=cell.identity.method,
                )
        elif cell.identity.task != "failure_injection" and any(
            not isinstance(row["updates_launched"], int)
            or row["updates_launched"] < 1
            or not isinstance(row["updates_published"], int)
            or row["updates_published"] < 1
            for row in performance
        ):
            raise ValueError("adapted completion has no launched/published update")
        requests = pq.read_table(
            evidence["request"],
            columns=[
                "request_id",
                "method",
                "repetition_block",
                "finished",
                "outcome_status",
                "output_hash_format",
                "output_tokens",
                "output_sha256",
                "output_token_ids",
                "output_token_ids_sha256",
                "arrival_ns",
                "admitted_ns",
                "completed_ns",
            ],
        ).to_pylist()
        allowed_outcomes = {
            "completed",
            "rejected",
            "timed_out",
            "cancelled",
            "unfinished",
        }
        if not requests or any(
            row["method"] != cell.identity.method
            or row["repetition_block"] != cell.identity.block
            or row["outcome_status"] not in allowed_outcomes
            or row["finished"] is not (row["outcome_status"] == "completed")
            or (row["completed_ns"] is None and row["outcome_status"] != "unfinished")
            or (
                row["completed_ns"] is not None
                and row["outcome_status"] == "unfinished"
            )
            for row in requests
        ):
            raise ValueError(
                "industrial completion has incomplete terminal-outcome evidence"
            )
        for request_row in requests:
            _validate_request_output_identity(request_row)
        request_outputs = {
            request["request_id"]: request["output_sha256"] for request in requests
        }
        if len(request_outputs) != len(requests):
            raise ValueError(
                "industrial request evidence contains duplicate identities"
            )
        if strict:
            if contract is None:
                raise RuntimeError("strict completion lost its split contract")
            closed_loop = str(cell.identity.arrival).startswith("closed_loop")
            if closed_loop:
                concurrency = cell.identity.concurrency
                if concurrency is None and cell.identity.arrival == "closed_loop_c1":
                    concurrency = 1
                if (
                    not isinstance(concurrency, int)
                    or isinstance(concurrency, bool)
                    or concurrency < 1
                    or not requests
                    or run_row["expected_request_rows"] != len(requests)
                ):
                    raise ValueError(
                        "closed-loop request coverage lacks a realized population"
                    )
                actual_ids = set(request_outputs)
                if actual_ids - set(contract["request_ids"]):
                    raise ValueError("closed-loop evidence crosses its request pool")
                request_by_id = {str(row["request_id"]): row for row in requests}
                for lane in range(concurrency):
                    seen_gap = False
                    previous_completed_ns: int | None = None
                    lane_request_ids = contract["request_ids"][lane::concurrency]
                    if not lane_request_ids or lane_request_ids[0] not in actual_ids:
                        raise ValueError(
                            "closed-loop evidence omits a registered client lane"
                        )
                    for request_id in lane_request_ids:
                        present = request_id in actual_ids
                        if seen_gap and present:
                            raise ValueError(
                                "closed-loop evidence is not a per-client prefix"
                            )
                        if present and stage != "preflight":
                            request = request_by_id[request_id]
                            arrival_ns = request.get("arrival_ns")
                            completed_ns = request.get("completed_ns")
                            if (
                                not isinstance(arrival_ns, int)
                                or isinstance(arrival_ns, bool)
                                or not isinstance(completed_ns, int)
                                or isinstance(completed_ns, bool)
                                or (
                                    previous_completed_ns is None
                                    and arrival_ns != run_row["started_ns"]
                                )
                                or (
                                    previous_completed_ns is not None
                                    and arrival_ns != previous_completed_ns
                                )
                            ):
                                raise ValueError(
                                    "closed-loop evidence is not zero-think"
                                )
                            previous_completed_ns = completed_ns
                        seen_gap = seen_gap or not present
            elif len(requests) != contract["expected_request_rows"] or set(
                request_outputs
            ) != set(contract["request_ids"]):
                raise ValueError("request evidence does not match the locked split")
            if contract["expected_round_rows"]:
                rounds = pq.read_table(
                    evidence["round"], columns=["request_id"]
                ).to_pylist()
                if len(rounds) != contract["expected_round_rows"] or {
                    round_row["request_id"] for round_row in rounds
                } != set(contract["request_ids"]):
                    raise ValueError("round evidence does not match the locked split")
            if contract["expected_update_rows"]:
                updates = pq.read_table(
                    evidence["update"], columns=["request_ids"]
                ).to_pylist()
                if len(updates) != contract["expected_update_rows"]:
                    raise ValueError(
                        "update evidence count differs from the locked split"
                    )
                for update in updates:
                    try:
                        update_request_ids = json.loads(update["request_ids"])
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            "update evidence has invalid request identities"
                        ) from exc
                    if (
                        not isinstance(update_request_ids, list)
                        or not update_request_ids
                        or not set(update_request_ids) <= set(contract["request_ids"])
                    ):
                        raise ValueError("update evidence crosses the locked split")
            outcome_counts = {
                outcome: sum(
                    request["outcome_status"] == outcome for request in requests
                )
                for outcome in allowed_outcomes
            }
            if outcome_counts["unfinished"]:
                raise ValueError(
                    "unfinished requests cannot enter a completed claim artifact"
                )
            aggregate_rows = [
                performance_row
                for performance_row in performance
                if performance_row["offered_requests"] is not None
            ]
            if len(aggregate_rows) != 1:
                raise ValueError(
                    "industrial completion requires one exact load-accounting row"
                )
            aggregate = aggregate_rows[0]
            expected_accounting = {
                "offered_requests": len(requests),
                "admitted_requests": sum(
                    request["admitted_ns"] is not None for request in requests
                ),
                "completed_requests": outcome_counts["completed"],
                "admission_rejections": outcome_counts["rejected"],
                "timeouts": outcome_counts["timed_out"],
                "cancellations": outcome_counts["cancelled"],
                "unfinished_requests": outcome_counts["unfinished"],
            }
            if any(
                not isinstance(aggregate.get(name), int)
                or isinstance(aggregate.get(name), bool)
                or aggregate.get(name) != expected
                for name, expected in expected_accounting.items()
            ):
                raise ValueError(
                    "performance load accounting differs from request outcomes"
                )
        covered = rank_coverage.setdefault(str(cell_id), set())
        if rank in covered:
            raise ValueError("completed-cell artifact duplicates a rank receipt")
        covered.add(rank)
        if strict:
            current_consensus = {
                "run_id": run_id,
                "run_nonce_sha256": run_row["run_nonce_sha256"],
                "topology_sha256": topology_sha256,
                "physical_binding_sha256": (
                    None if not formal else row["physical_binding_sha256"]
                ),
                "experiment_budget_sha256": (
                    None if not formal else row["experiment_budget_sha256"]
                ),
                "runtime_sha256": run_row["runtime_sha256"],
                "split_sha256": run_row["split_sha256"],
                "request_ids_sha256": run_row["request_ids_sha256"],
                "request_outputs": request_outputs,
            }
            previous_consensus = consensus.setdefault(str(cell_id), current_consensus)
            if previous_consensus != current_consensus:
                raise ValueError("cell ranks do not agree on one run/output identity")
        completed.append(str(cell_id))
    for cell_id, ranks in rank_coverage.items():
        expected_ranks = set(
            range(
                len(
                    _industrial_physical_assignment_from_dict(
                        contract_by_cell[cell_id]["physical_assignment"]
                    ).gpu_uuids
                )
                if formal
                else len(known[cell_id].resources.gpu_uuids)
            )
        )
        if ranks != expected_ranks:
            raise ValueError("completed cell lacks exact per-rank evidence coverage")
    unique = tuple(dict.fromkeys(completed))
    return unique, _canonical_sha256(value)


def _single_industrial_receipt(
    path: str | Path, *, experiment: str
) -> ExperimentReceipt:
    (receipt,) = _load_industrial_receipts([str(path)])
    if receipt.experiment != experiment:
        raise ValueError(f"expected a sealed {experiment} receipt")
    return receipt


def _write_planning_artifact(
    output: str | Path,
    *,
    artifact,
    encode,
) -> int:
    value = encode(artifact)
    _write_json(output, value)
    print(artifact.sha256)
    return 0


def _reduce_e1_activation(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    receipt = _single_industrial_receipt(args.e3a_receipt, experiment="E3a")
    selection = sealed_e3a_selection_from_dict(_load_bound_json(args.selection))
    artifact = reduce_e1_activation(
        registry,
        e3a_receipt=receipt,
        selection=selection,
    )
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=reducer_activation_artifact_to_dict,
    )


def _reduce_e2_activation(args: argparse.Namespace) -> int:
    registry, receipt, pareto, stage_index, prior = _load_e2_activation_manifest(
        args.manifest
    )
    artifact = reduce_e2_activation(
        registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=stage_index,
        prior_reduction=prior,
    )
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=reducer_activation_artifact_to_dict,
    )


def _reduce_e2_halving(args: argparse.Namespace) -> int:
    artifact = _load_e2_stage_manifest(args.manifest)
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=e2_stage_reduction_artifact_to_dict,
    )


def _materialize_confirmation_pilots(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    family = confirmation_family_identity_from_dict(_load_bound_json(args.family))
    artifact = materialize_confirmation_pilots(registry, family)
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=family_activation_artifact_to_dict,
    )


def _reduce_confirmation_family_power(args: argparse.Namespace) -> int:
    registry, pilot, inventory, envelope, blocks = _load_family_power_manifest(
        args.manifest
    )
    plan = reduce_confirmation_family_power(
        registry=registry,
        pilot_activation=pilot,
        blocks=blocks,
        hardware_envelope=envelope,
        inventory=inventory,
        confirmation_data_visible=False,
    )
    return _write_planning_artifact(
        args.output,
        artifact=plan,
        encode=confirmation_family_power_reduction_artifact_to_dict,
    )


def _materialize_confirmation_final_prefix(args: argparse.Namespace) -> int:
    registry, pilot, inventory, envelope, blocks = _load_family_power_manifest(
        args.power_manifest
    )
    reduction = reduce_confirmation_family_power(
        registry=registry,
        pilot_activation=pilot,
        blocks=blocks,
        hardware_envelope=envelope,
        inventory=inventory,
        confirmation_data_visible=False,
    )
    artifact = materialize_confirmation_prefix(
        registry,
        family=pilot.family,
        reduction=reduction,
        pilot_activation=pilot,
    )
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=family_activation_artifact_to_dict,
    )


def _validate_evidence_alias(args: argparse.Namespace) -> int:
    manifest = raw_evidence_alias_manifest_from_dict(_load_bound_json(args.manifest))
    artifact = reduce_evidence_alias(
        registry=_load_industrial_registry(args.registry),
        manifest=manifest,
        hardware_envelope=_analysis_hardware_envelope(
            _load_bound_json(args.hardware_envelope)
        ),
        inventory=_load_gpu_inventory(args.inventory),
    )
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=evidence_alias_reduction_artifact_to_dict,
    )


def _build_evidence_dependence_map(args: argparse.Namespace) -> int:
    direct_map = evidence_dependence_map_from_dict(_load_bound_json(args.direct_map))
    direct_cell_ids: list[str] = []
    for unit in direct_map.units:
        if unit.member_cell_ids != (unit.source_cell_id,) or unit.unit_sha256 != (
            _canonical_sha256({"direct_observation_cell_id": unit.source_cell_id})
        ):
            raise ValueError(
                "direct dependence-map input must contain singleton observations"
            )
        direct_cell_ids.append(unit.source_cell_id)
    aliases = tuple(
        evidence_alias_reduction_artifact_from_dict(_load_bound_json(path))
        for path in args.alias_reduction
    )
    if len({alias.sha256 for alias in aliases}) != len(aliases):
        raise ValueError("duplicate evidence alias artifact")
    artifact = build_evidence_dependence_map(
        direct_observation_cell_ids=tuple(direct_cell_ids),
        aliases=aliases,
    )
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=evidence_dependence_map_to_dict,
    )


def _collect_gpu_inventory(args: argparse.Namespace) -> int:
    inventory, receipt = collect_gpu_inventory(
        challenge_nonce_sha256=args.challenge_nonce_sha256
    )
    if inventory.source_receipt_sha256 != receipt.get("receipt_sha256"):
        raise RuntimeError("GPU inventory source receipt binding changed")
    _write_json(args.receipt_output, receipt)
    _write_json(args.output, inventory.to_dict())
    reloaded = _load_gpu_inventory(args.output)
    if reloaded != inventory:
        raise RuntimeError("written GPU inventory changed identity")
    print(inventory.sha256)
    return 0


def _assemble_gpu_fleet_inventory(args: argparse.Namespace) -> int:
    if len(args.inventory) != len(args.interference_envelope):
        raise ValueError(
            "fleet assembly requires one --interference-envelope per --inventory"
        )
    bindings: list[HostInventoryBinding] = []
    for inventory_path, envelope_path in zip(
        args.inventory,
        args.interference_envelope,
        strict=True,
    ):
        inventory = _load_gpu_inventory(inventory_path)
        if len(inventory.host_ids) != 1:
            raise ValueError("fleet host inventory must contain exactly one host")
        bindings.append(
            HostInventoryBinding(
                schema_version=1,
                host_id=inventory.host_ids[0],
                inventory=inventory,
                interference_envelope=_load_interference_envelope(envelope_path),
            )
        )
    fleet = assemble_gpu_fleet_inventory(bindings)
    _write_json(args.output, fleet.to_dict())
    if _load_gpu_fleet_inventory(args.output) != fleet:
        raise RuntimeError("written GPU fleet inventory changed identity")
    print(fleet.sha256)
    return 0


def _build_interference_envelope(args: argparse.Namespace) -> int:
    inventory = _load_gpu_inventory(args.inventory)
    envelope, receipt = build_serial_interference_envelope(inventory)
    if envelope.source_receipt_sha256 != receipt.get("receipt_sha256"):
        raise RuntimeError("interference-envelope source receipt binding changed")
    _write_json(args.receipt_output, receipt)
    _write_json(args.output, envelope.to_dict())
    reloaded = _load_interference_envelope(args.output)
    if reloaded != envelope:
        raise RuntimeError("written interference envelope changed identity")
    print(envelope.sha256)
    return 0


def _materialize_interference_calibration_bootstrap(
    args: argparse.Namespace,
) -> int:
    """Write the sole release-derived permission for calibration generation."""

    registry = _load_industrial_registry(args.registry)
    activation = _load_registry_stage_activation_manifest(args.activation_manifest)
    inventory = _load_gpu_inventory(args.inventory)
    authority = materialize_interference_calibration_bootstrap_authority(
        registry,
        inventory,
        activation,
    )
    _write_json(args.receipt_output, authority.source_receipt)
    _write_json(args.output, authority.bootstrap_envelope.to_dict())
    if (
        _load_bound_json(args.receipt_output) != authority.source_receipt
        or _load_interference_envelope(args.output) != authority.bootstrap_envelope
    ):
        raise RuntimeError("written interference bootstrap changed identity")
    print(authority.sha256)
    return 0


def _reduce_interference_calibration(args: argparse.Namespace) -> int:
    """Replay raw terminals and materialize an exact-cardinality envelope."""

    try:
        # The release trust root is checked before opening caller paths.  This
        # keeps no-card/test keys incapable of probing or minting formal data.
        require_release_interference_attester()
    except InterferenceCalibrationBlockedError as error:
        decision = {
            "schema_version": 1,
            "kind": "interference_calibration_reduction_decision",
            "status": "BLOCKED",
            "reason_code": error.reason_code,
            "trusted_attester_id": None,
        }
        _write_json(args.output, decision)
        print(_canonical_sha256(decision))
        return 42

    execution_authority = InterferenceCalibrationExecutionAuthority.from_dict(
        _load_bound_json(args.authority)
    )
    reduction = execution_authority.reconstruct().revalidate()
    envelope = reduction.require_envelope()
    _write_json(args.output, reduction.to_dict())
    _write_json(args.envelope_output, envelope.to_dict())
    if (
        _load_bound_json(args.output) != reduction.to_dict()
        or _load_interference_envelope(args.envelope_output) != envelope
    ):
        raise RuntimeError("written calibrated interference authority changed identity")
    print(reduction.sha256)
    return 0


def _materialize_industrial_budget_plan(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    gpu_inventory = _load_gpu_inventory(args.inventory)
    if not args.activation_plan and not args.family_activation:
        decision = {
            "schema_version": 1,
            "kind": "industrial_budget_materialization_decision",
            "status": "BLOCKED",
            "reason_code": "reducer_owned_activation_manifest_missing",
            "registry_sha256": registry.sha256,
            "gpu_inventory_sha256": gpu_inventory.sha256,
            "detail": ("no bound reducer-owned activation manifest was supplied"),
        }
        _write_json(args.output, decision)
        print(_canonical_sha256(decision))
        return 42
    activations, family_activations, family_power_reductions = (
        _load_budget_activation_bundle(
            activation_paths=args.activation_plan,
            family_activation_paths=args.family_activation,
            family_power_plan_paths=args.family_power_plan,
        )
    )
    (
        policy,
        load_bindings,
        capacity_envelope,
        capacity_authority,
    ) = _load_budget_materialization_inputs(
        policy_path=args.budget_policy,
        load_binding_paths=args.budget_load_binding,
        capacity_envelope_path=args.capacity_envelope,
        capacity_manifest_path=args.capacity_manifest,
        capacity_verification_receipt_path=args.capacity_verification_receipt,
    )
    plan = materialize_industrial_budgets(
        registry,
        activations=activations,
        family_activations=family_activations,
        family_power_reductions=family_power_reductions,
        load_bindings=load_bindings,
        policy=policy,
        inventory=budget_inventory_identity_from_gpu_inventory(gpu_inventory),
        capacity_envelope=capacity_envelope,
        capacity_authority=capacity_authority,
    )
    _write_json(args.output, budget_plan_to_dict(plan))
    if _load_budget_plan(args.output) != plan:
        raise RuntimeError("written BudgetPlan changed identity")
    print(plan.sha256)
    return 0 if plan.status == "READY" else 42


def _bind_industrial_budget_authority(args: argparse.Namespace) -> int:
    """Publish the sole path-bound authority for formal budget consumers."""

    capacity_authority = bind_capacity_authority(
        args.capacity_manifest,
        args.capacity_verification_receipt,
    )
    authority = bind_budget_materialization_authority(
        activation_manifest_path=args.activation_manifest,
        policy_path=args.budget_policy,
        load_binding_paths=tuple(args.budget_load_binding),
        capacity_envelope_path=args.capacity_envelope,
        capacity_authority=capacity_authority,
        declared_plan_path=args.budget_plan,
    )
    value = budget_materialization_authority_binding_to_dict(authority)
    _write_json(args.output, value)
    if _load_bound_json(args.output) != value:
        raise RuntimeError("written budget materialization authority changed identity")
    print(authority.sha256)
    return 0


def _materialize_stage_activation(args: argparse.Namespace) -> int:
    artifact = _load_registry_stage_activation_manifest(args.manifest)
    _write_json(args.output, registry_stage_activation_to_dict(artifact))
    reloaded = registry_stage_activation_from_dict(_load_bound_json(args.output))
    if reloaded != artifact:
        raise RuntimeError("written registry-stage activation changed identity")
    print(artifact.sha256)
    # This command materializes a diagnostic decision.  Successfully writing a
    # BLOCKED artifact is a successful CLI operation; execution boundaries
    # separately require AVAILABLE and therefore cannot mistake exit status for
    # release authority.
    return 0


def _materialize_preflight_pointer_coverage(args: argparse.Namespace) -> int:
    """Deep-reopen the one compile, one exactness, and eight serving terminals."""

    registry = _load_industrial_registry(args.registry)
    source = PreflightExecutionSourceAuthority.bind(
        registry=registry,
        runtime_sha256=_artifact_sha256(args.runtime_artifact),
        split_sha256=_artifact_sha256(args.split_artifact),
        compile_result_path=args.compile_result,
        exactness_result_path=args.exactness_result,
        interference_execution_authority_path=args.interference_authority,
    )
    activation, coverage = materialize_pointer_preflight_coverage(registry, source)
    _write_json(args.source_output, source.to_dict())
    _write_json(
        args.activation_output,
        {
            "schema_version": 1,
            "kind": "formal_preflight_pointer_activation_manifest",
            "registry_artifact": str(Path(args.registry).resolve()),
            "source_authority": str(Path(args.source_output).resolve()),
            "activation": registry_stage_activation_to_dict(activation),
        },
    )
    _write_json(args.coverage_output, coverage.to_dict())
    reopened_source = PreflightExecutionSourceAuthority.from_dict(
        _load_bound_json(args.source_output)
    )
    reopened_activation = _load_preflight_pointer_activation_manifest(
        args.activation_output
    )
    reopened_coverage = PreflightCoverageReceipt.from_dict(
        _load_bound_json(args.coverage_output)
    )
    if reopened_source != source or reopened_activation != activation:
        raise RuntimeError("written preflight pointer authority changed identity")
    verify_preflight_coverage(registry, reopened_activation, reopened_coverage)
    require_complete_preflight_coverage(reopened_coverage)
    print(reopened_coverage.sha256)
    return 0


def _estimate_industrial_budget(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    gpu_inventory = _load_gpu_inventory(args.inventory)
    interference_envelope = _load_interference_envelope(args.interference_envelope)
    plan, _, _, _ = _rematerialize_budget_plan(
        registry=registry,
        gpu_inventory=gpu_inventory,
        declared_plan_path=args.budget_plan,
        activation_paths=args.activation_plan,
        family_activation_paths=args.family_activation,
        family_power_plan_paths=args.family_power_plan,
        policy_path=args.budget_policy,
        load_binding_paths=args.budget_load_binding,
        capacity_envelope_path=args.capacity_envelope,
        capacity_manifest_path=args.capacity_manifest,
        capacity_verification_receipt_path=args.capacity_verification_receipt,
        require_ready=False,
    )
    assumption_values = {
        row.reason_code
        for row in plan.dispositions
        if row.status is BudgetDispositionStatus.UNRESOLVED
    }
    if plan.capacity_envelope is not None and plan.capacity_authority is None:
        # Capacity arithmetic without its path-bound raw-source receipt is
        # useful diagnostic input, but it is never launch authority.  Keep the
        # missing trust lift visible even when incomplete budget coverage is
        # the per-cell disposition selected by the materializer.
        assumption_values.add("capacity_raw_authority_missing")
    plan_assumptions = tuple(sorted(assumption_values))
    diagnostic_budgets = plan.diagnostic_budgets
    diagnostic_cell_ids = tuple(row.cell_id for row in diagnostic_budgets)
    report = estimate_industrial_budget(
        registry,
        # The exact reducer requires one budget for every supplied cell.  An
        # UNRESOLVED plan can intentionally retain only the subset whose
        # source-owned load semantics were materializable.  Report that exact
        # diagnostic subset and carry the missing-cell reasons above; all
        # scheduler/executor boundaries still call ``plan.require_ready()``.
        activated_cell_ids=diagnostic_cell_ids,
        activation_sha256=plan.activation_sha256,
        budgets=diagnostic_budgets,
        inventory=plan.inventory,
        gpu_inventory=gpu_inventory,
        interference_envelope=interference_envelope,
        unresolved_assumptions=plan_assumptions,
    )
    artifact = industrial_budget_report_to_dict(report)
    _write_json(args.output, artifact)
    print(report.sha256)
    return 0 if not report.unresolved_assumptions else 42


def _plan_industrial_dispatch(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    inventory = _load_gpu_inventory(args.inventory)
    envelope = _load_interference_envelope(args.interference_envelope)
    budget_plan, activations, family_activations, family_power_reductions = (
        _rematerialize_budget_plan(
            registry=registry,
            gpu_inventory=inventory,
            declared_plan_path=args.budget_plan,
            activation_paths=args.activation_plan,
            family_activation_paths=args.family_activation,
            family_power_plan_paths=args.family_power_plan,
            policy_path=args.budget_policy,
            load_binding_paths=args.budget_load_binding,
            capacity_envelope_path=args.capacity_envelope,
            capacity_manifest_path=args.capacity_manifest,
            capacity_verification_receipt_path=(args.capacity_verification_receipt),
            require_ready=False,
        )
    )
    if budget_plan.capacity_authority is None:
        raise ValueError(
            "capacity_raw_authority_missing: dispatch requires the path-bound "
            "capacity manifest and verification receipt"
        )
    budget_plan.require_ready()
    if len(activations) > 1:
        raise ValueError("dispatch accepts at most one reducer activation artifact")
    receipts = _load_industrial_receipts(args.receipt)
    activation_artifact = None if not activations else activations[0]
    completed_activation_artifact = activation_artifact
    completed_family_activations = tuple(
        artifact
        for artifact in family_activations
        if artifact.activation_round == "excluded_pilots"
    )
    ready_experiment = registry.ready_experiment(receipts)
    completed_e2_requested = args.completed_e2_stage_manifest is not None
    if ready_experiment == "E2" and args.completed_cells is not None:
        if not completed_e2_requested:
            raise ValueError(
                "E2 completed-cell evidence requires its raw completed-stage manifest"
            )
        completed_reduction = _load_e2_stage_manifest(args.completed_e2_stage_manifest)
        if (
            completed_reduction.registry_sha256 != registry.sha256
            or activation_artifact is None
            or activation_artifact.plan.activation_round
            != f"halving_{completed_reduction.stage_index + 1}"
            or activation_artifact.plan.source_selection_sha256
            != completed_reduction.sha256
        ):
            raise ValueError(
                "E2 completed-stage authority is not the immediate predecessor of "
                "the next activation"
            )
        completed_activation_artifact = completed_reduction.activation
    elif completed_e2_requested:
        raise ValueError(
            "completed E2 stage authority requires E2 completed-cell evidence"
        )
    direct_dependency_sha256 = None
    if ready_experiment is not None:
        definition = registry.definition(ready_experiment)
        if definition.dependencies:
            dependency_name = definition.dependencies[-1]
            matching = tuple(
                receipt for receipt in receipts if receipt.experiment == dependency_name
            )
            if len(matching) != 1:
                raise ValueError(
                    "dispatch requires one exact direct dependency receipt"
                )
            direct_dependency_sha256 = matching[0].sha256
    completed, _ = _completed_industrial_cells(
        args.completed_cells,
        registry,
        experiment=ready_experiment,
        require_industrial_contract=args.completed_cells is not None,
        direct_dependency_receipt_sha256=direct_dependency_sha256,
        activation_artifact=completed_activation_artifact,
        family_activations=completed_family_activations,
        family_power_reductions=(),
        inventory=inventory,
    )
    planning_context = GpuDispatchPlanningContext(
        registry=registry,
        inventory=inventory,
        interference_envelope=envelope,
        budgets=tuple(
            budget
            for budget in budget_plan.require_ready()
            if budget.cell_id not in completed
        ),
        receipts=receipts,
        completed_cell_ids=tuple(sorted(completed)),
        activation_artifact=activation_artifact,
        family_activations=family_activations,
        family_power_reductions=family_power_reductions,
    )
    plan = planning_context.issue_plan()
    artifact = plan.to_dict()
    _write_json(args.output, artifact)
    if (
        GpuDispatchPlan.from_dict(
            artifact,
            planning_context=planning_context,
        )
        != plan
    ):
        raise RuntimeError("written GPU dispatch plan changed identity")
    print(plan.sha256)
    return 0


def _execute_dispatch_wave(args: argparse.Namespace) -> int:
    """Run one receipt-bounded wave, or report an immutable prelaunch block."""

    local_arguments = {
        "--materialization-manifest": args.materialization_manifest,
        "--wave-index": args.wave_index,
        "--resume-receipt": args.resume_receipt,
        "--receipt-output": args.receipt_output,
    }
    if args.host_request_stdin:
        if any(value is not None for value in local_arguments.values()):
            raise ValueError(
                "--host-request-stdin is mutually exclusive with host-local "
                "dispatch arguments"
            )
        stdin = getattr(sys.stdin, "buffer", None)
        stdout = getattr(sys.stdout, "buffer", None)
        if stdin is None or stdout is None:
            raise RuntimeError("remote host-wave mode requires binary stdin/stdout")
        request = stdin.read(MAX_REQUEST_BYTES + 1)
        if len(request) > MAX_REQUEST_BYTES:
            return 42
        try:
            exit_code, response = asyncio.run(execute_host_local_wave_request(request))
        except (TypeError, ValueError):
            return 42
        stdout.write(response)
        stdout.flush()
        return exit_code

    missing = tuple(
        option
        for option in (
            "--materialization-manifest",
            "--wave-index",
            "--receipt-output",
        )
        if local_arguments[option] is None
    )
    if missing:
        raise ValueError("execute-dispatch-wave requires " + ", ".join(missing))

    try:
        receipt = asyncio.run(
            execute_dispatch_wave_bundles(
                args.materialization_manifest,
                wave_index=args.wave_index,
                receipt_output=args.receipt_output,
                resume_receipt_path=args.resume_receipt,
            )
        )
    except ExecutionBundleBlockedError as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "industrial_dispatch_wave_execution_decision",
                    "status": "BLOCKED",
                    "reason_code": error.reason_code,
                    "wave_index": args.wave_index,
                    "materialization_manifest": args.materialization_manifest,
                },
                sort_keys=True,
            )
        )
        return 42
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "industrial_dispatch_wave_execution_decision",
                    "status": "BLOCKED",
                    "reason_code": "industrial_execution_bundle_or_resume_invalid",
                    "wave_index": args.wave_index,
                    "materialization_manifest": args.materialization_manifest,
                    "failure_sha256": _canonical_sha256(
                        {
                            "exception_type": type(error).__qualname__,
                            "message": str(error),
                        }
                    ),
                },
                sort_keys=True,
            )
        )
        return 42
    print(receipt.sha256)
    return 42 if receipt.phase.value == "FAILED" else 0


def _analyze_industrial(args: argparse.Namespace) -> int:
    (
        registry,
        pilot,
        final,
        plan,
        inventory,
        envelope,
        blocks,
        alias_manifests,
        dependence,
        gpu_attestation,
        doctor_report,
        repetitions,
        seed,
        runtime_metrics_authority,
    ) = _load_industrial_analysis_manifest(args.manifest)
    reduction = reduce_industrial_schema_v3(
        registry=registry,
        pilot_activation=pilot,
        final_activation=final,
        confirmation_reduction=plan,
        blocks=blocks,
        hardware_envelope=envelope,
        inventory=inventory,
        evidence_dependence_map=dependence,
        evidence_alias_manifests=alias_manifests,
        gpu_attestation=gpu_attestation,
        doctor_report=doctor_report,
        runtime_metrics_authority=runtime_metrics_authority,
        bootstrap_repetitions=repetitions,
        bootstrap_seed=seed,
    )
    artifact = reduction.artifact.to_dict()
    if _canonical_sha256(artifact) != reduction.artifact.sha256:
        raise RuntimeError("industrial reducer artifact digest is not canonical")
    _write_json(args.output, artifact)
    if _artifact_sha256(args.output) != reduction.artifact.sha256:
        raise RuntimeError("written industrial reducer artifact identity changed")
    print(reduction.artifact.sha256)
    # The current release has no hardware-rooted attester. Content-bound caller
    # artifacts remain diagnostic and can never produce a positive exit claim.
    return 42


def _analyze_e3b_long_context(args: argparse.Namespace) -> int:
    registry, families, inventory, envelope, repetitions, seed = (
        _load_e3b_long_context_analysis_manifest(args.manifest)
    )
    artifact = reduce_e3b_long_context_from_raw(
        registry=registry,
        families=families,
        hardware_envelope=envelope,
        inventory=inventory,
        bootstrap_repetitions=repetitions,
        bootstrap_seed=seed,
    )
    payload = artifact.to_dict()
    if _canonical_sha256(payload) != artifact.sha256:
        raise RuntimeError("E3b long-context reducer digest is not canonical")
    _write_json(args.output, payload)
    if _artifact_sha256(args.output) != artifact.sha256:
        raise RuntimeError("written E3b long-context reducer identity changed")
    print(artifact.sha256)
    return 42


def _load_content_verification_receipt(
    path: str,
    *,
    now_ns: int,
) -> tuple[ContentVerificationReceipt, tuple[object, ...]]:
    receipt = ContentVerificationReceipt.from_dict(
        CanonicalJsonProofBinding.bind(path).reopen()
    )
    return receipt, receipt.revalidate(current_ns=now_ns)


def _verified_workload_sources_from_receipt(
    path: str,
    *,
    now_ns: int,
) -> VerifiedReleaseWorkloadSources:
    _receipt, verified_rows = _load_content_verification_receipt(
        path,
        now_ns=now_ns,
    )
    matches = tuple(
        row for row in verified_rows if type(row) is VerifiedReleaseWorkloadSources
    )
    if len(matches) != 1:
        raise ValueError("content receipt lacks one workload authorization")
    return matches[0]


def _content_artifact_bindings(
    specifications: list[str],
) -> tuple[ContentJsonArtifactBinding, ...]:
    rows: list[ContentJsonArtifactBinding] = []
    for specification in specifications:
        if type(specification) is not str or specification.count("=") != 1:
            raise ValueError("content artifact must be ARTIFACT_ID=ABSOLUTE_PATH")
        artifact_id, path_text = specification.split("=", 1)
        if not artifact_id or not path_text:
            raise ValueError("content artifact must be ARTIFACT_ID=ABSOLUTE_PATH")
        rows.append(ContentJsonArtifactBinding.from_path(artifact_id, path_text))
    result = tuple(sorted(rows, key=lambda row: row.artifact_id))
    if tuple(row.artifact_id for row in result) != tuple(
        sorted({row.artifact_id for row in result})
    ):
        raise ValueError("content artifact IDs must be unique")
    return result


def _verify_content_authorizations_cli(args: argparse.Namespace) -> int:
    authorizations = (
        ContentJsonArtifactBinding.from_path(
            "dataset:burstgpt_six_source", args.burstgpt_authorization
        ),
        ContentJsonArtifactBinding.from_path(
            "dataset:e0_task_native", args.e0_dataset_authorization
        ),
        ContentJsonArtifactBinding.from_path(
            "prepared:formal_dag", args.prepared_model_authorization
        ),
        ContentJsonArtifactBinding.from_path(
            "workload:e3a", args.workload_authorization
        ),
    )
    receipt = verify_and_reserve_content_authorizations(
        verified_ns=args.now_ns,
        authorization_artifacts=authorizations,
        content_artifacts=_content_artifact_bindings(args.content_artifact),
        replay_store=ChallengeReplayStore(args.replay_store),
    )
    publish_canonical_json_no_replace(args.output, receipt.to_dict())
    reopened = ContentVerificationReceipt.from_dict(
        CanonicalJsonProofBinding.bind(args.output).reopen()
    )
    if reopened != receipt or reopened.sha256 != receipt.sha256:
        raise RuntimeError("content verification receipt changed after publication")
    reopened.revalidate(current_ns=args.now_ns)
    print(receipt.sha256)
    return 0


def _scope_content_verification_receipt_cli(args: argparse.Namespace) -> int:
    master_binding = ContentJsonArtifactBinding.from_path(
        "content:master_verification_receipt",
        args.master_receipt,
    )
    master = ContentVerificationReceipt.from_dict(master_binding.load())
    scoped = derive_stage_content_verification_receipt(
        master,
        master_artifact=master_binding,
        stage=args.stage,
        current_ns=args.now_ns,
    )
    publish_canonical_json_no_replace(args.output, scoped.to_dict())
    reopened = ContentVerificationReceipt.from_dict(
        CanonicalJsonProofBinding.bind(args.output).reopen()
    )
    if reopened != scoped or reopened.sha256 != scoped.sha256:
        raise RuntimeError("scoped content receipt changed after publication")
    reopened.revalidate_formal_scope(current_ns=args.now_ns)
    print(scoped.sha256)
    return 0


def _publish_burstgpt_shape_authority_cli(args: argparse.Namespace) -> int:
    from lightcone_spec.runtime.preflight_runner import (
        BurstGptShapeAuthority,
        derive_burstgpt_shape_authority_from_content_receipt,
    )

    receipt, _verified = _load_content_verification_receipt(
        args.content_verification_receipt,
        now_ns=args.now_ns,
    )
    authority = derive_burstgpt_shape_authority_from_content_receipt(
        receipt,
        current_ns=args.now_ns,
    )
    publish_canonical_json_no_replace(args.output, authority.to_dict())
    reopened = BurstGptShapeAuthority.from_dict(
        CanonicalJsonProofBinding.bind(args.output).reopen()
    )
    if reopened != authority or reopened.sha256 != authority.sha256:
        raise RuntimeError("BurstGPT shape authority changed after publication")
    print(authority.sha256)
    return 0


def _bind_formal_workload_cli(args: argparse.Namespace) -> int:
    authorization = _verified_workload_sources_from_receipt(
        args.content_verification_receipt,
        now_ns=args.now_ns,
    )
    authority = bind_authorized_formal_workload_authority(
        args.workload,
        args.source,
        authorization=authorization,
    )
    artifact = formal_workload_authority_cli_artifact(authority)
    publish_canonical_json_no_replace(args.output, artifact)
    _publish_immutable_bytes(
        Path(f"{Path(args.output).resolve()}.sha256"),
        f"{_canonical_sha256(artifact)}\n".encode("ascii"),
        label="formal workload authority sidecar",
    )
    reloaded = formal_workload_authority_from_cli_artifact(
        _load_bound_json(args.output)
    )
    if reloaded != authority or reloaded.sha256 != authority.sha256:
        raise RuntimeError("written formal workload authority changed identity")
    print(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "formal_workload_authority_decision",
                "status": "BOUND_AUTHORIZED_CONTENT",
                "reason_code": None,
                "workload_id": authority.workload_id,
                "authority_sha256": authority.sha256,
                "formal_execution_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _materialize_dispatch_execution_bundles_cli(args: argparse.Namespace) -> int:
    """Atomically publish source-owned schema-v5 bundles and their manifest."""

    from lightcone_spec.orchestration.execution_bundle_materializer import (
        DispatchBundleMaterializationBlocked,
        materialize_dispatch_execution_bundles,
    )

    try:
        manifest = materialize_dispatch_execution_bundles(
            args.request,
            output_directory=args.output_directory,
        )
    except (DispatchBundleMaterializationBlocked, ExecutionBundleBlockedError) as error:
        reason_code = getattr(error, "reason_code", None)
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "industrial_dispatch_bundle_materialization_decision",
                    "status": "BLOCKED",
                    "reason_code": reason_code,
                },
                sort_keys=True,
            )
        )
        return 42
    print(str(manifest))
    return 0


def _revalidate_formal_workload_cli(args: argparse.Namespace) -> int:
    authority = formal_workload_authority_from_cli_artifact(
        _load_bound_json(args.authority)
    )
    authorization = _verified_workload_sources_from_receipt(
        args.content_verification_receipt,
        now_ns=args.now_ns,
    )
    rebound = revalidate_authorized_formal_workload_authority(
        authority,
        authorization=authorization,
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "formal_workload_authority_decision",
                "status": "BOUND_AUTHORIZED_CONTENT",
                "reason_code": None,
                "workload_id": rebound.workload_id,
                "authority_sha256": rebound.sha256,
                "formal_execution_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    single_operator_result = handle_formal_single_operator_command(args)
    if single_operator_result is not None:
        return single_operator_result
    if args.command == "doctor":
        project_root = args.project_root or args.path or "."
        sglang_root = args.sglang_root or args.path
        capacity_options = {}
        if (
            args.stage_capacity_gate is not None
            or args.stage_capacity_schedule is not None
            or args.stage_capacity_attestation is not None
            or args.stage_capacity_activation_sha256 is not None
            or args.stage_capacity_now_ns is not None
        ):
            capacity_options = {
                "stage_capacity_gate_path": args.stage_capacity_gate,
                "stage_capacity_schedule_path": args.stage_capacity_schedule,
                "stage_capacity_attestation_path": (args.stage_capacity_attestation),
                "stage_capacity_activation_sha256": (
                    args.stage_capacity_activation_sha256
                ),
                "stage_capacity_now_ns": args.stage_capacity_now_ns,
            }
        formatted = format_doctor(project_root, sglang_root, **capacity_options)
        print(formatted)
        report = json.loads(formatted)
        return 0 if report.get("status") == "PASS" else 42
    if args.command == "validate-config":
        config = load_run_config(args.config)
        print(config.model_dump_json(indent=2))
        return 0
    if args.command == "build-preliminary-speed-study":
        manifest = PreliminarySpeedStudyManifest.default()
        manifest.write(args.output)
        print(manifest.sha256)
        return 0
    if args.command == "bind-formal-workload-authority":
        return _bind_formal_workload_cli(args)
    if args.command == "revalidate-formal-workload-authority":
        return _revalidate_formal_workload_cli(args)
    if args.command == "verify-content-authorizations":
        return _verify_content_authorizations_cli(args)
    if args.command == "scope-content-verification-receipt":
        return _scope_content_verification_receipt_cli(args)
    if args.command == "publish-burstgpt-shape-authority":
        return _publish_burstgpt_shape_authority_cli(args)
    if args.command == "build-industrial-registry":
        return _build_industrial_registry(args)
    if args.command == "publish-tts-calibration-source-authority":
        return _publish_tts_calibration_source_authority(args)
    if args.command == "publish-chronobelief-source-authority":
        return _publish_chronobelief_source_authority(args)
    if args.command == "publish-e1-recipe-anchor-authority":
        return _publish_e1_recipe_anchor_authority(args)
    if args.command == "publish-formal-protocol-lock-git-snapshot":
        return _publish_formal_protocol_lock_git_snapshot(args)
    if args.command == "publish-formal-protocol-lock-source-proof":
        return _publish_formal_protocol_lock_source_proof(args)
    if args.command == "create-protocol-lock":
        return _create_protocol_lock(args)
    if args.command == "verify-signed-protocol-lock":
        return _verify_signed_protocol_lock(args)
    if args.command == "create-gpu-hour-envelope":
        return _create_gpu_hour_envelope(args)
    if args.command == "reduce-stage-gpu-hour-envelope":
        return _reduce_stage_gpu_hour_envelope(args)
    if args.command == "materialize-preflight-gpu-hour-envelope":
        return _materialize_preflight_gpu_hour_envelope_cli(args)
    if args.command == "materialize-prospective-stage-gpu-hours":
        return _materialize_prospective_stage_gpu_hours(args)
    if args.command == "materialize-staged-prospective-gpu-hours":
        return _materialize_staged_prospective_gpu_hours(args)
    if args.command == "publish-formal-stage-gpu-hour-envelope-proof":
        return _publish_formal_stage_gpu_hour_envelope_proof(args)
    if args.command == "publish-formal-initial-stage-materialization-proof":
        return _publish_formal_initial_stage_materialization_proof(args)
    if args.command == "publish-formal-downstream-materialization-proof":
        return _publish_formal_downstream_materialization_proof(args)
    if args.command == "publish-formal-downstream-pilot-precoverage":
        return _publish_formal_downstream_pilot_precoverage(args)
    if args.command == "publish-formal-portable-stage-coverage-proof":
        return _publish_formal_portable_stage_coverage(args)
    if args.command == "publish-formal-downstream-reduction-proof":
        return _publish_formal_downstream_reduction_proof(args)
    if args.command == "publish-formal-downstream-completed-prefix":
        return _publish_formal_downstream_completed_prefix(args)
    if args.command == "publish-formal-e3a-staged-selection-proof":
        return _publish_formal_e3a_staged_selection_proof(args)
    if args.command == "reserve-formal-stage-gpu-hours":
        return _reserve_formal_stage_gpu_hours(args)
    if args.command == "aggregate-formal-study-gpu-hours":
        return _aggregate_formal_study_gpu_hours(args)
    if args.command == "create-stage-materialization-receipt":
        return _create_stage_materialization(args)
    if args.command == "verify-signed-stage-materialization":
        return _verify_signed_stage_materialization(args)
    if args.command == "create-stage-coverage-receipt":
        return _create_stage_coverage(args)
    if args.command == "verify-signed-stage-coverage":
        return _verify_signed_stage_coverage(args)
    if args.command == "publish-formal-runtime-authority-manifest":
        return _publish_formal_runtime_authority_manifest(args)
    if args.command == "publish-formal-rebuild-artifact":
        return _publish_formal_rebuild_artifact(args)
    if args.command == "publish-formal-tts-calibration-reduction-proof":
        return _publish_formal_tts_calibration_reduction_proof(args)
    if args.command == "publish-formal-stage-execution-shard":
        return _publish_formal_stage_execution_shard(args)
    if args.command == "publish-formal-stage-prefix":
        return _publish_formal_stage_prefix(args)
    if args.command == "publish-scientific-source-validation":
        return _publish_scientific_source_validation(args)
    if args.command == "formal-stage-operation":
        return _formal_stage_operation(args)
    if args.command == "assemble-formal-registry-manifest":
        return _assemble_formal_registry(args)
    if args.command == "reserve-formal-registry-verification":
        return _reserve_formal_registry_verification(args)
    if args.command == "extend-formal-registry-verification":
        return _extend_formal_registry_verification(args)
    if args.command == "verify-formal-registry-verification":
        return _verify_formal_registry_verification(args)
    if args.command == "authorize-formal-preflight-dispatch":
        return _authorize_formal_preflight_dispatch_cli(args)
    if args.command == "materialize-formal-preflight-launch-cap-schedule":
        return _materialize_formal_preflight_launch_cap_schedule_cli(args)
    if args.command == "execute-formal-preflight-raw":
        return _execute_formal_preflight_raw_cli(args)
    if args.command == "qualify-formal-preflight-exactness":
        return _qualify_formal_preflight_exactness_cli(args)
    if args.command == "qualify-formal-preflight-interference":
        return _qualify_formal_preflight_interference_cli(args)
    if args.command == "finalize-formal-preflight-evidence":
        return _finalize_formal_preflight_evidence_cli(args)
    if args.command == "collect-gpu-inventory":
        return _collect_gpu_inventory(args)
    if args.command == "assemble-gpu-fleet-inventory":
        return _assemble_gpu_fleet_inventory(args)
    if args.command == "build-interference-envelope":
        return _build_interference_envelope(args)
    if args.command == "materialize-interference-calibration-bootstrap":
        return _materialize_interference_calibration_bootstrap(args)
    if args.command == "reduce-interference-calibration":
        return _reduce_interference_calibration(args)
    if args.command == "materialize-stage-capacity-gate":
        return _materialize_stage_capacity_gate_cli(args)
    if args.command == "seal-industrial-stage":
        return _seal_industrial_stage(args)
    if args.command == "plan-industrial-dispatch":
        return _plan_industrial_dispatch(args)
    if args.command == "materialize-dispatch-execution-bundles":
        return _materialize_dispatch_execution_bundles_cli(args)
    if args.command == "execute-dispatch-wave":
        return _execute_dispatch_wave(args)
    if args.command == "materialize-industrial-budgets":
        return _materialize_industrial_budget_plan(args)
    if args.command == "bind-industrial-budget-authority":
        return _bind_industrial_budget_authority(args)
    if args.command == "materialize-stage-activation":
        return _materialize_stage_activation(args)
    if args.command == "materialize-preflight-pointer-coverage":
        return _materialize_preflight_pointer_coverage(args)
    if args.command == "estimate-industrial-budget":
        return _estimate_industrial_budget(args)
    if args.command == "reduce-e1-activation":
        return _reduce_e1_activation(args)
    if args.command == "reduce-e2-activation":
        return _reduce_e2_activation(args)
    if args.command == "reduce-e2-successive-halving":
        return _reduce_e2_halving(args)
    if args.command == "materialize-confirmation-pilots":
        return _materialize_confirmation_pilots(args)
    if args.command == "reduce-confirmation-family-power":
        return _reduce_confirmation_family_power(args)
    if args.command == "materialize-confirmation-prefix":
        return _materialize_confirmation_final_prefix(args)
    if args.command == "validate-evidence-alias":
        return _validate_evidence_alias(args)
    if args.command == "build-evidence-dependence-map":
        return _build_evidence_dependence_map(args)
    if args.command == "analyze-industrial":
        return _analyze_industrial(args)
    if args.command == "analyze-e3b-long-context":
        return _analyze_e3b_long_context(args)
    if args.command == "build-onlinespec-study":
        manifest = OnlineSpecManifest.default()
        manifest.write(args.output)
        print(manifest.sha256)
        return 0
    if args.command == "verify-onlinespec-source":
        receipt = verify_onlinespec_source_checkout(args.checkout, args.audit)
        _write_json(args.output, receipt)
        print(_canonical_sha256(receipt))
        return 0
    if args.command == "list-onlinespec-candidates":
        return _list_onlinespec_candidates(args)
    if args.command == "lock-models":
        lock = resolve_model_lock(tuple(args.models), token=os.environ.get("HF_TOKEN"))
        lock.write(args.output)
        print(lock.sha256)
        return 0
    if args.command == "prepare-models":
        lock = ModelLock.load(args.lockfile)
        roots = prepare_models(
            lock,
            args.model_cache,
            token=os.environ.get("HF_TOKEN"),
            local_files_only=args.offline,
        )
        _write_json(
            args.output,
            {
                "schema_version": 2,
                "lock_sha256": lock.sha256,
                "roots": roots,
            },
        )
        return 0
    if args.command == "select-preliminary-speed-config":
        return _select(args)
    if args.command == "select-preliminary-anchor-config":
        return _select_anchor(args)
    if args.command == "select-onlinespec-config":
        return _select_onlinespec(args)
    if args.command == "select-onlinespec-anchor-config":
        return _select_onlinespec_anchor(args)
    if args.command == "render-preliminary-runtime":
        return _render_runtime(args)
    if args.command == "render-onlinespec-runtime":
        return _render_onlinespec_runtime(args)
    if args.command == "render-onlinespec-tuning-runtime":
        return _render_onlinespec_tuning_runtime(args)
    if args.command == "render-preliminary-static-load-runtime":
        return _render_static_load_runtime(args)
    if args.command == "render-preliminary-target-only-runtime":
        return _render_target_only_runtime(args)
    if args.command == "render-preliminary-tuning-runtime":
        return _render_tuning_runtime(args)
    if args.command == "render-preliminary-replication-runtime":
        return _render_replication_runtime(args)
    if args.command == "list-preliminary-tuning-candidates":
        return _list_tuning_candidates(args)
    if args.command == "run-preliminary-controlled-slice":
        return _run_controlled_slice(args)
    if args.command == "run-onlinespec-tuning-slice":
        return _run_onlinespec_tuning_slice(args)
    if args.command == "run-preliminary-natural-slice":
        return _run_natural_slice(args)
    if args.command == "build-preliminary-profiler-plan":
        return _build_profiler_plan(args)
    if args.command == "collect-preliminary-static-load-screen":
        return _collect_static_load(args)
    if args.command == "advance-preliminary-tuning-stage":
        return _advance_tuning(args)
    if args.command == "advance-onlinespec-tuning-stage":
        return _advance_onlinespec_tuning(args)
    if args.command == "run-preliminary-confirmation":
        return _run_confirmation(args)
    if args.command == "run-onlinespec-confirmation":
        return _run_onlinespec_confirmation(args)
    if args.command == "run-preliminary-target-reference":
        return _run_target_reference(args)
    if args.command == "collect-preliminary-speed-study":
        return _collect_speed_study(args)
    if args.command == "collect-onlinespec-study":
        return _collect_onlinespec_study(args)
    if args.command == "build-preliminary-confirmation-queue":
        return _build_confirmation_queue(args)
    if args.command == "build-onlinespec-queue":
        return _build_onlinespec_queue(args)
    if args.command == "attest-preliminary-speed-study":
        return _attest(args)
    if args.command == "attest-onlinespec-study":
        return _attest_onlinespec(args)
    if args.command == "analyze-preliminary-speed-study":
        return _analyze(args)
    if args.command == "analyze-onlinespec-study":
        return _analyze_onlinespec(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
