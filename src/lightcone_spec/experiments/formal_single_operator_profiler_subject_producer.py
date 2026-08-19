"""Source-owned E4 profiler subject requirement production.

The profiler cells are diagnostics of one already measured E4-local headline
stratum.  They therefore cannot accept a caller-selected load, traffic mix,
configuration, launch, or request trace.  This module starts from the exact
current ``e4_profiler`` execution source, reopens its direct ``e4_local``
predecessor, and deep-revalidates the unique saturation/mixed winner run.  The
published requirement only contains immutable bindings to that run's
RunConfig, compile launch, code-owned schedule source, and tokenized request
schedule.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from lightcone_spec.config import load_run_config, run_config_sha256
from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
    FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256,
    FormalSingleOperatorProfilerSubjectRequirement,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorValidatedActual,
    RebuiltFormalSingleOperatorStageCompletion,
    _e4_configuration_from_payload,
    load_formal_single_operator_execution_source,
    rebuild_formal_single_operator_stage_completion,
)
from lightcone_spec.experiments.stage_materialization import (
    E4_SCREEN_FACTOR_LEVELS,
    MaterializedCell,
)
from lightcone_spec.orchestration.formal_physical_dispatch import (
    FormalServingRequestScheduleReceipt,
    FormalServingRunPlan,
)
from lightcone_spec.runtime.compile_cache import (
    CompileCacheLaunchPlan,
    validate_compile_key_for_run_config,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.formal_single_operator import (
    FormalSingleOperatorRunManifest,
    revalidate_formal_single_operator_run_manifest,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _absolute_existing_directory(label: str, value: str | Path) -> Path:
    requested = Path(value)
    resolved = requested.resolve(strict=False)
    if (
        not requested.is_absolute()
        or requested != resolved
        or not resolved.is_dir()
        or resolved.is_symlink()
    ):
        raise ValueError(f"{label} must be an existing normalized directory")
    return resolved


def _configuration(cell: MaterializedCell) -> tuple[tuple[str, str | int], ...]:
    dimensions = dict(cell.dimensions)
    rows = tuple(
        (name, dimensions.get(name)) for name, _levels in E4_SCREEN_FACTOR_LEVELS
    )
    if any(type(value) not in {str, int} for _name, value in rows):
        raise ValueError("E4-local profiler subject cell lacks exact factors")
    return rows  # type: ignore[return-value]


def _manifest_artifact(
    manifest: FormalSingleOperatorRunManifest,
    name: str,
) -> Path:
    matches = tuple(
        row
        for row in manifest.artifacts
        if row.name == name and row.status == "PRESENT"
    )
    if len(matches) != 1:
        raise ValueError(f"E4-local profiler subject lacks {name} artifact")
    path = Path(manifest.run_directory) / matches[0].relative_path
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError("E4-local profiler subject artifact path differs")
    return path


def _selected_actual(
    *,
    predecessor: RebuiltFormalSingleOperatorStageCompletion,
    cell: MaterializedCell,
) -> FormalSingleOperatorValidatedActual:
    if type(predecessor) is not RebuiltFormalSingleOperatorStageCompletion:
        raise TypeError("E4-local profiler subject predecessor is not exact")
    rows = predecessor.artifact.actual_results
    matches = tuple(row for row in rows if row.cell_id == cell.cell_id)
    if (
        len(matches) != 1
        or type(matches[0]) is not FormalSingleOperatorValidatedActual
        or matches[0].status != "COMPLETE"
        or matches[0].validator_kind
        != "formal_single_operator_run_manifest_revalidator"
    ):
        raise ValueError(
            "E4-local profiler subject lacks one deep-validated COMPLETE actual"
        )
    return matches[0]


def _revalidate_selected_run(
    *,
    repository_root: Path,
    predecessor: RebuiltFormalSingleOperatorStageCompletion,
    cell: MaterializedCell,
    actual: FormalSingleOperatorValidatedActual,
) -> tuple[
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
]:
    """Return config, launch, schedule-source, and schedule bindings."""

    if type(predecessor) is not RebuiltFormalSingleOperatorStageCompletion:
        raise TypeError("E4-local profiler subject predecessor is not exact")
    materialization = predecessor.materialization
    node_materialization = predecessor.node_materialization
    manifest = revalidate_formal_single_operator_run_manifest(
        repository_root=repository_root,
        manifest_path=actual.source.absolute_path,
    )
    if (
        manifest.sha256 != actual.result_identity_sha256
        or manifest.completion_status != "COMPLETE"
        or manifest.stage != "E4"
        or manifest.cell_id != cell.cell_id
        or manifest.materialization_sha256 != materialization.sha256
        or manifest.materialization_protocol_lock_sha256
        != materialization.protocol_lock_sha256
        or manifest.role != cell.method_role
        or manifest.backend != cell.backend
        or manifest.target_model_id != cell.model
    ):
        raise ValueError("E4-local profiler subject run manifest lineage differs")

    plan_path = _manifest_artifact(manifest, "run_plan")
    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    if (
        plan.sha256 != plan_binding.semantic_sha256
        or plan.sha256 != manifest.run_plan_sha256
        or plan.stage != "E4"
        or plan.materialized_cell_id != cell.cell_id
        or plan.execution_binding_sha256 != manifest.execution_binding_sha256
        or plan.subject_sha256 != manifest.execution_subject_sha256
    ):
        raise ValueError("E4-local profiler subject run plan differs")

    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    config = load_run_config(launch.run_config_path)
    cache_plan = CompileCacheLaunchPlan.load(launch.compile_cache_plan_path)
    validate_compile_key_for_run_config(cache_plan, config=config)
    config_binding = CanonicalJsonProofBinding.bind(
        launch.run_config_path,
        semantic_sha256=run_config_sha256(config),
    )
    if (
        launch.sha256 != plan.launch_manifest.semantic_sha256
        or launch.sha256 != manifest.launch_manifest_sha256
        or launch.server_argv != manifest.launch_argv
        or launch.server_argv_sha256 != manifest.launch_argv_sha256
        or launch.localhost_port != manifest.localhost_port
        or launch.inventory_sha256 != manifest.inventory_sha256
        or launch.gpu_uuids != plan.gpu_uuids
        or run_config_sha256(config) != manifest.run_config_sha256
        or config.model_dump(mode="json") != manifest.run_config
        or config.runtime.telemetry_detail != "headline"
        or config.adaptation is None
    ):
        raise ValueError("E4-local profiler subject launch/config differs")

    factor_values = (
        ("update_stride", config.adaptation.stride),
        ("microbatch", config.runtime.adaptation_microbatch_size),
        ("coalescing", config.runtime.adaptation_publication_coalescing),
        ("stream_priority", config.runtime.adaptation_stream_priority),
    )
    if factor_values != _configuration(cell):
        raise ValueError("E4-local profiler subject config differs from winner cell")

    schedule_artifact = _manifest_artifact(manifest, "request_schedule")
    schedule_binding = plan.request_schedule_receipt
    if Path(schedule_binding.absolute_path) != schedule_artifact:
        raise ValueError("E4-local profiler subject schedule artifact differs")
    schedule = FormalServingRequestScheduleReceipt.from_dict(schedule_binding.reopen())
    if schedule.sha256 != schedule_binding.semantic_sha256:
        raise ValueError("E4-local profiler subject schedule digest differs")
    schedule.reopen()
    schedule_source = CanonicalJsonProofBinding.bind(schedule.schedule_source.path)
    selected_materialization = CanonicalJsonProofBinding.bind(
        node_materialization.materialization_source.absolute_path
    )
    if (
        schedule.to_dict() != manifest.request_schedule
        or schedule.sha256 != manifest.request_schedule_sha256
        or schedule.materialized_cell_id != cell.cell_id
        or schedule.materialization != selected_materialization
        or schedule.compile_launch_manifest != plan.launch_manifest
        or schedule.execution_binding_sha256 != plan.execution_binding_sha256
        or schedule.subject_sha256 != plan.subject_sha256
        or schedule.topology_mode != plan.topology_mode
        or schedule.schedule_source.path != schedule_source.absolute_path
        or schedule.schedule_source.raw_sha256 != schedule_source.raw_sha256
        or schedule.schedule_source.semantic_sha256 != schedule_source.semantic_sha256
        or schedule.schedule_source.size != schedule_source.size
    ):
        raise ValueError("E4-local profiler subject request trace differs")
    return config_binding, plan.launch_manifest, schedule_source, schedule_binding


def derive_formal_single_operator_profiler_subject_requirement(
    *,
    execution_source_path: str | Path,
    repository_root: str | Path,
) -> FormalSingleOperatorProfilerSubjectRequirement:
    """Deep-rebuild the one profiler subject requirement from the current DAG."""

    repository = _absolute_existing_directory(
        "profiler subject repository root", repository_root
    )
    source = load_formal_single_operator_execution_source(execution_source_path)
    if source.node != "e4_profiler" or source.predecessor_completion_source is None:
        raise ValueError("profiler subject requires the current E4-profiler source")
    current = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="profiler subject current materialization"
        )
    )
    predecessor = rebuild_formal_single_operator_stage_completion(
        source.predecessor_completion_source.absolute_path
    )
    if predecessor.artifact.node != "e4_local":
        raise ValueError("profiler subject direct predecessor is not E4-local")

    winner = _e4_configuration_from_payload(
        predecessor.decision.payload.get("winner_configuration")
    )
    selected_configuration_sha256 = _sha256(winner)
    selected = tuple(
        row
        for row in predecessor.materialization.cells
        if row.task == "winner_neighborhood_local_factorial_headline"
        and dict(row.dimensions).get("load") == "saturation"
        and dict(row.dimensions).get("traffic") == "mixed_prefill_decode"
        and _configuration(row) == winner
    )
    profiler_configuration_ids = {
        dict(row.dimensions).get("selected_configuration_sha256")
        for row in current.cells
        if row.task == "mechanism_profile_only"
    }
    if (
        current.stage != "E4"
        or len(current.cells) != 3
        or len(selected) != 1
        or profiler_configuration_ids != {selected_configuration_sha256}
        or any(row.task != "mechanism_profile_only" for row in current.cells)
    ):
        raise ValueError("profiler subject lacks one exact selected headline stratum")
    cell = selected[0]
    actual = _selected_actual(predecessor=predecessor, cell=cell)
    config, launch, workload, schedule = _revalidate_selected_run(
        repository_root=repository,
        predecessor=predecessor,
        cell=cell,
        actual=actual,
    )
    return FormalSingleOperatorProfilerSubjectRequirement(
        schema_version=1,
        kind="formal_single_operator_profiler_subject_requirement",
        protocol_sha256=(FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256),
        source_headline_cell_id=cell.cell_id,
        selected_configuration_sha256=selected_configuration_sha256,
        selected_full_run_config=config,
        selected_compile_launch_manifest=launch,
        code_owned_profiler_subject_workload=workload,
        code_owned_request_schedule=schedule,
    )


def revalidate_formal_single_operator_profiler_subject_requirement(
    *,
    execution_source_path: str | Path,
    repository_root: str | Path,
    requirement_path: str | Path,
) -> FormalSingleOperatorProfilerSubjectRequirement:
    """Reopen a published requirement and rebuild it from its direct source."""

    binding = CanonicalJsonProofBinding.bind(requirement_path)
    observed = FormalSingleOperatorProfilerSubjectRequirement.from_dict(
        binding.reopen()
    )
    expected = derive_formal_single_operator_profiler_subject_requirement(
        execution_source_path=execution_source_path,
        repository_root=repository_root,
    )
    if observed != expected:
        raise ValueError("profiler subject requirement differs from current E4-local")
    return observed


def publish_formal_single_operator_profiler_subject_requirement(
    *,
    execution_source_path: str | Path,
    repository_root: str | Path,
    output_path: str | Path,
) -> FormalSingleOperatorProfilerSubjectRequirement:
    """Atomically publish one canonical no-replace profiler requirement."""

    destination = Path(output_path)
    resolved = destination.resolve(strict=False)
    if (
        not destination.is_absolute()
        or destination != resolved
        or not destination.parent.is_dir()
        or destination.parent.is_symlink()
    ):
        raise ValueError("profiler subject output path must be absolute/normalized")
    if os.path.lexists(destination):
        raise FileExistsError("profiler subject requirement already exists")
    requirement = derive_formal_single_operator_profiler_subject_requirement(
        execution_source_path=execution_source_path,
        repository_root=repository_root,
    )
    publish_canonical_json_no_replace(destination, requirement.to_dict())
    rebound = revalidate_formal_single_operator_profiler_subject_requirement(
        execution_source_path=execution_source_path,
        repository_root=repository_root,
        requirement_path=destination,
    )
    if rebound != requirement:
        raise RuntimeError("profiler subject requirement changed during publication")
    return rebound


__all__ = [
    "derive_formal_single_operator_profiler_subject_requirement",
    "publish_formal_single_operator_profiler_subject_requirement",
    "revalidate_formal_single_operator_profiler_subject_requirement",
]
