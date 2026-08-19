"""Deterministic early-stage launch inputs for ``formal_single_operator_v1``.

This is intentionally a small trusted-operator adapter.  It maps an exact
current E3a, TTS-Cal, E1, or E2 materialized cell to the existing
``RunConfig`` and ``CompileLaunchManifest`` types.  Recipe values, GPU choice,
port, and argv are derived from the current stage chain and the already-run
preflight launch inputs; callers cannot provide those values.

The final descriptor is an ordinary path-bound input bundle for the physical
operator.  It is not a signature, coverage receipt, or scientific authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.config import (
    AdaptationConfig,
    ModelPair,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
    load_run_config,
    run_config_sha256,
)
from lightcone_spec.experiments.formal_preflight_execution import (
    FormalPreflightInterferenceExecutionManifest,
)
from lightcone_spec.experiments.formal_preflight_inputs import (
    FormalPreflightExecutionInputs,
)
from lightcone_spec.experiments.formal_protocol import (
    ProtocolLock,
    content_sha256,
)
from lightcone_spec.experiments.formal_registry import (
    protocol_lock_from_dict,
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorExecutionSource,
    FormalSingleOperatorJsonBinding,
    FormalSingleOperatorNodeMaterialization,
    FormalSingleOperatorStageCompletion,
    FormalSingleOperatorStageDecision,
)
from lightcone_spec.experiments.gpu_pool import GpuAvailability, GpuInventory
from lightcone_spec.experiments.preflight_authority import (
    PreflightExecutionSourceAuthority,
)
from lightcone_spec.experiments.protocol import DFLASH_LOSS_POSITION_DECAY
from lightcone_spec.experiments.stage_materialization import (
    E1Geometry,
    E2CandidateRecipe,
    MaterializedCell,
    StageMaterializationReceipt,
    default_e2_recipe_grid_authority,
)
from lightcone_spec.orchestration.runtime import _render_server
from lightcone_spec.runtime.compile_cache import (
    CompileCacheLaunchPlan,
    validate_compile_key_for_run_config,
)
from lightcone_spec.runtime.compile_runner import (
    COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    CompileLaunchManifest,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import NativeRuntimeGpuProofArtifact

_EARLY_NODES = frozenset({"e3a", "tts_cal", "e1", "e2_r0", "e2_r1", "e2_r2", "e2_r3"})
_EARLY_STAGES = frozenset({"E3a", "TTS-Cal", "E1", "E2"})
_ADAPTATION_RESERVE_MB = 4096


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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


def _flag_value(argv: tuple[str, ...], flag: str) -> str:
    positions = tuple(index for index, value in enumerate(argv) if value == flag)
    if (
        len(positions) != 1
        or positions[0] + 1 >= len(argv)
        or argv[positions[0] + 1].startswith("--")
    ):
        raise ValueError(f"source preflight launch lacks exact {flag}")
    return argv[positions[0] + 1]


@dataclass(frozen=True)
class _ShallowCompletion:
    artifact: FormalSingleOperatorStageCompletion
    node_materialization: FormalSingleOperatorNodeMaterialization
    materialization: StageMaterializationReceipt
    decision: FormalSingleOperatorStageDecision
    predecessor: _ShallowCompletion | None


def _completion_from_binding(
    source: FormalSingleOperatorJsonBinding,
    *,
    visited: frozenset[str],
) -> _ShallowCompletion:
    if source.absolute_path in visited:
        raise ValueError("single-operator early completion chain contains a cycle")
    raw = source.reopen(label="single-operator early predecessor completion")
    artifact = FormalSingleOperatorStageCompletion.from_dict(raw)
    if content_sha256(artifact.to_dict()) != source.semantic_sha256:
        raise ValueError("single-operator early predecessor binding differs")
    node_materialization = FormalSingleOperatorNodeMaterialization.from_dict(
        artifact.node_materialization_source.reopen(
            label="single-operator early node materialization"
        )
    )
    materialization = stage_materialization_receipt_from_dict(
        node_materialization.materialization_source.reopen(
            label="single-operator early materialization"
        )
    )
    decision = FormalSingleOperatorStageDecision.from_dict(
        artifact.decision_source.reopen(label="single-operator early decision")
    )
    if (
        artifact.node_materialization_sha256 != node_materialization.sha256
        or artifact.materialization_sha256 != materialization.sha256
        or artifact.decision_sha256 != decision.sha256
        or artifact.node != node_materialization.node
        or artifact.node != decision.node
        or artifact.materialization_sha256 != decision.materialization_sha256
    ):
        raise ValueError("single-operator early completion lineage differs")
    predecessor = (
        None
        if artifact.predecessor_source is None
        else _completion_from_binding(
            artifact.predecessor_source,
            visited=visited | {source.absolute_path},
        )
    )
    if (predecessor is None) != (artifact.predecessor_completion_sha256 is None) or (
        predecessor is not None
        and predecessor.artifact.sha256 != artifact.predecessor_completion_sha256
    ):
        raise ValueError("single-operator early predecessor identity differs")
    return _ShallowCompletion(
        artifact=artifact,
        node_materialization=node_materialization,
        materialization=materialization,
        decision=decision,
        predecessor=predecessor,
    )


def _completion_for_node(
    value: _ShallowCompletion,
    node: str,
) -> _ShallowCompletion | None:
    current: _ShallowCompletion | None = value
    while current is not None:
        if current.artifact.node == node:
            return current
        current = current.predecessor
    return None


@dataclass(frozen=True)
class _CurrentEarlySource:
    binding: CanonicalJsonProofBinding
    source: FormalSingleOperatorExecutionSource
    protocol_lock: ProtocolLock
    materialization: StageMaterializationReceipt
    cell: MaterializedCell
    predecessor: _ShallowCompletion


def _current_source(
    execution_source_path: str | Path,
    *,
    materialized_cell_id: str,
) -> _CurrentEarlySource:
    binding = CanonicalJsonProofBinding.bind(execution_source_path)
    source = FormalSingleOperatorExecutionSource.from_dict(binding.reopen())
    if content_sha256(source.to_dict()) != binding.semantic_sha256:
        raise ValueError("single-operator early execution source digest differs")
    if source.node not in _EARLY_NODES or source.stage not in _EARLY_STAGES:
        raise ValueError("single-operator early mapper received another stage")
    protocol_lock = protocol_lock_from_dict(
        source.protocol_lock_source.reopen(label="single-operator early ProtocolLock")
    )
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="single-operator early current materialization"
        )
    )
    cells = tuple(
        row for row in materialization.cells if row.cell_id == materialized_cell_id
    )
    if len(cells) != 1:
        raise ValueError("single-operator early cell is outside materialization")
    if source.predecessor_completion_source is None:
        raise ValueError("single-operator early stage lacks preflight lineage")
    predecessor = _completion_from_binding(
        source.predecessor_completion_source,
        visited=frozenset({binding.absolute_path}),
    )
    if (
        protocol_lock.sha256 != source.protocol_lock_sha256
        or materialization.sha256 != source.materialization_sha256
        or materialization.stage != source.stage
        or source.predecessor_completion_sha256 != predecessor.artifact.sha256
        or source.predecessor_decision_sha256 != predecessor.decision.sha256
    ):
        raise ValueError("single-operator early current source lineage differs")
    return _CurrentEarlySource(
        binding=binding,
        source=source,
        protocol_lock=protocol_lock,
        materialization=materialization,
        cell=cells[0],
        predecessor=predecessor,
    )


@dataclass(frozen=True)
class _PreflightLaunchSources:
    inputs_binding: CanonicalJsonProofBinding
    inputs: FormalPreflightExecutionInputs
    template_launch: CompileLaunchManifest
    inventory: GpuInventory


def _preflight_launch_sources(
    current: _CurrentEarlySource,
    *,
    preflight_inputs_path: str | Path,
) -> _PreflightLaunchSources:
    inputs_binding = CanonicalJsonProofBinding.bind(preflight_inputs_path)
    inputs = FormalPreflightExecutionInputs.from_dict(inputs_binding.reopen())
    if inputs.sha256 != inputs_binding.semantic_sha256:
        raise ValueError("single-operator preflight input digest differs")
    # These local inputs are the exact files used by the completed preflight.
    # The early mapper accepts no independent model, workload, or inventory path.
    inputs.content_receipt.reopen()
    inputs.workload_authority.load()
    inventory = GpuInventory.from_dict(inputs.inventory.reopen())
    if inventory.sha256 != inputs.inventory.semantic_sha256:
        raise ValueError("single-operator early inventory identity differs")
    manifest = FormalPreflightInterferenceExecutionManifest.from_dict(
        inputs.interference_manifest.reopen()
    )
    if manifest.sha256 != inputs.interference_manifest.semantic_sha256:
        raise ValueError("single-operator preflight launch manifest digest differs")
    launches: list[tuple[str, CompileLaunchManifest]] = []
    for row in manifest.inputs:
        launch = CompileLaunchManifest.load(row.launch_manifest_path)
        if (
            len(launch.gpu_uuids) != 1
            or launch.inventory_sha256 != inventory.sha256
            or launch.target_model_id != current.cell.model
            or launch.target_content_authority_sha256
            != current.source.prepared_model_content_authorization_sha256
            or launch.tokenizer_content_authority_sha256
            != current.source.prepared_model_content_authorization_sha256
        ):
            continue
        device = inventory.device(launch.gpu_uuids[0])
        if device.availability is not GpuAvailability.READY:
            continue
        launches.append((row.registry_cell_id, launch))
    by_gpu: dict[str, tuple[str, CompileLaunchManifest]] = {}
    for row in sorted(launches, key=lambda item: (item[1].gpu_uuids[0], item[0])):
        by_gpu.setdefault(row[1].gpu_uuids[0], row)
    if not by_gpu:
        raise ValueError("single-operator preflight has no reusable TP1 launch")
    ordered = tuple(by_gpu[gpu] for gpu in sorted(by_gpu))
    selected = int(current.cell.cell_id[:16], 16) % len(ordered)
    template = ordered[selected][1]
    return _PreflightLaunchSources(
        inputs_binding=inputs_binding,
        inputs=inputs,
        template_launch=template,
        inventory=inventory,
    )


def _tts_selection(current: _CurrentEarlySource) -> tuple[float, int, str]:
    completion = _completion_for_node(current.predecessor, "tts_cal")
    if completion is None:
        raise ValueError("single-operator early chain lacks TTS selection")
    payload = completion.decision.payload
    learning_rate = payload.get("learning_rate")
    stride = payload.get("stride")
    candidate = payload.get("candidate_id")
    if (
        type(learning_rate) is not float
        or type(stride) is not int
        or type(candidate) is not str
        or candidate
        != content_sha256(
            {
                "authority_sha256": (
                    current.protocol_lock.tts_calibration_authority_sha256
                ),
                "learning_rate": learning_rate,
                "stride": stride,
            }
        )
    ):
        raise ValueError("single-operator TTS selection differs from current chain")
    return learning_rate, stride, candidate


def _tts_adaptation(
    *,
    stage: str,
    cell: MaterializedCell,
    width: int,
    learning_rate: float,
    stride: int,
) -> AdaptationConfig:
    return AdaptationConfig(
        weight_update_mode="full",
        parameter_scope="all",
        adaptation_group_id=f"formal-single-{stage.lower()}-{cell.cell_id[:24]}",
        optimizer=OptimizerConfig(
            name="adam",
            learning_rate=learning_rate,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            grad_clip=None,
            schedule="constant",
        ),
        rank=None,
        lora_alpha=None,
        stride=stride,
        canvas_tokens=width,
        loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
    )


def _e1_candidate_adaptation(
    cell: MaterializedCell,
    *,
    width: int,
    recipe_anchor_authority_sha256: str,
) -> AdaptationConfig:
    dimensions = dict(cell.dimensions)
    anchor = dimensions.get("optimizer_anchor")
    if anchor == "adamw":
        optimizer = OptimizerConfig(
            name="adamw",
            learning_rate=1e-4,
            weight_decay=0.01,
        )
    elif anchor == "sgdm":
        optimizer = OptimizerConfig(
            name="sgdm",
            learning_rate=1e-3,
            weight_decay=0.0,
            momentum=0.9,
        )
    else:
        raise ValueError("single-operator E1 optimizer anchor differs")
    rank_value = dimensions.get("rank")
    rank = None if rank_value == "none" else int(rank_value)  # type: ignore[arg-type]
    geometry = E1Geometry(
        scope=str(dimensions.get("scope")),
        parameterization=str(dimensions.get("parameterization")),  # type: ignore[arg-type]
        rank=rank,
        alpha_over_rank=(
            None
            if dimensions.get("alpha_over_rank") == "none"
            else float(dimensions["alpha_over_rank"])
        ),
    )
    expected_recipe = content_sha256(
        {
            "kind": "e1_lightcone_candidate",
            "geometry": geometry,
            "optimizer_anchor": anchor,
            "matched_width": width,
            "recipe_anchor_authority_sha256": recipe_anchor_authority_sha256,
        }
    )
    if cell.recipe_sha256 != expected_recipe:
        raise ValueError("single-operator E1 candidate recipe differs")
    return AdaptationConfig(
        weight_update_mode=geometry.parameterization,
        parameter_scope=geometry.scope,
        adaptation_group_id=f"e1:{cell.cell_id}",
        optimizer=optimizer,
        rank=geometry.rank,
        lora_alpha=geometry.rank,
        stride=10,
        canvas_tokens=width,
        loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
    )


def _chronobelief_proof_sha256(
    inputs: FormalPreflightExecutionInputs,
) -> str:
    authority = PreflightExecutionSourceAuthority.from_dict(
        inputs.execution_authority.reopen()
    )
    if authority.sha256 != inputs.execution_authority.semantic_sha256:
        raise ValueError("single-operator preflight execution authority differs")
    matches = tuple(
        row
        for row in authority.qualification_proofs
        if row.suite_id == "chronobelief_gpu_parity"
    )
    if len(matches) != 1:
        raise ValueError("single-operator preflight lacks ChronoBelief result")
    artifact = NativeRuntimeGpuProofArtifact.from_dict(
        matches[0].proof_artifact.reopen()
    )
    if content_sha256(artifact.to_dict()) != matches[0].proof_artifact.semantic_sha256:
        raise ValueError("single-operator ChronoBelief preflight result changed")
    return artifact.verified_proof_sha256


def _e2_candidate_adaptation(
    current: _CurrentEarlySource,
    *,
    width: int,
    preflight_inputs: FormalPreflightExecutionInputs,
) -> AdaptationConfig:
    dimensions = dict(current.cell.dimensions)
    rank_value = dimensions.get("rank")
    alpha_value = dimensions.get("alpha_over_rank")
    geometry = E1Geometry(
        scope=str(dimensions.get("scope")),
        parameterization=str(dimensions.get("parameterization")),  # type: ignore[arg-type]
        rank=None if rank_value == "none" else int(rank_value),  # type: ignore[arg-type]
        alpha_over_rank=(
            None if alpha_value == "none" else float(alpha_value)  # type: ignore[arg-type]
        ),
    )
    grid = default_e2_recipe_grid_authority()
    if grid.sha256 != current.protocol_lock.e2_recipe_grid_authority_sha256:
        raise ValueError("single-operator E2 grid differs from ProtocolLock")
    candidate = E2CandidateRecipe(
        geometry=geometry,
        optimizer=str(dimensions.get("optimizer")),
        schedule=str(dimensions.get("schedule")),
        learning_rate=float(dimensions.get("learning_rate")),
        optimizer_recipe_authority_sha256=grid.optimizer_recipe_authority.sha256,
    )
    if (
        current.cell.recipe_sha256 != candidate.sha256
        or dimensions.get("geometry_sha256") != geometry.sha256
    ):
        raise ValueError("single-operator E2 candidate recipe differs")
    chronobelief = (
        _chronobelief_proof_sha256(preflight_inputs)
        if candidate.optimizer == "chronobelief"
        else None
    )
    return grid.adaptation_config_for(
        candidate,
        canvas_tokens=width,
        adaptation_group_id=f"e2:{current.cell.cell_id}",
        chronobelief_gpu_proof_sha256=chronobelief,
    )


def _run_config(
    current: _CurrentEarlySource,
    *,
    template: CompileLaunchManifest,
    gpu_uuid: str,
    preflight_inputs: FormalPreflightExecutionInputs,
) -> RunConfig:
    template_config = load_run_config(template.run_config_path)
    dimensions = dict(current.cell.dimensions)
    if current.source.stage == "E3a":
        width = dimensions.get("width")
        concurrency = dimensions.get("concurrency")
    elif current.source.stage == "TTS-Cal":
        width = dimensions.get("width")
        e3a = _completion_for_node(current.predecessor, "e3a")
        concurrency = None if e3a is None else e3a.decision.payload.get("common_load")
    else:
        width = dimensions.get("matched_width")
        concurrency = dimensions.get("common_load")
    if (
        type(width) is not int
        or width < 2
        or type(concurrency) is not int
        or concurrency < 1
    ):
        raise ValueError("single-operator early width/load dimensions differ")
    role = current.cell.method_role
    method = {
        "Target-only": "target_only",
        "Static": "static",
        "TTS-calibration-candidate": "tts",
        "TTS": "tts",
        "L0-naive": "l0",
        "LightCone-candidate": "l0",
    }.get(role)
    if method is None:
        raise ValueError("single-operator early method role differs")
    model = template_config.model.model_copy(
        update={
            "target": current.cell.model,
            "algorithm": "DFLASH",
            "draft_depth": width - 1,
        }
    )
    runtime = template_config.runtime.model_copy(
        update={
            "speculation_enabled": method != "target_only",
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "node_count": 1,
            "device_identity": gpu_uuid,
            "max_running_requests": concurrency,
            "speculative_num_draft_tokens": width,
            "telemetry_detail": "headline",
            "adaptation_microbatch_size": 1,
            "adaptation_publication_coalescing": 1,
            "adaptation_stream_priority": "default",
        }
    )
    adaptation: AdaptationConfig | None = None
    if role == "TTS-calibration-candidate":
        learning_rate = dimensions.get("learning_rate")
        stride = dimensions.get("stride")
        if type(learning_rate) is not float or type(stride) is not int:
            raise ValueError("single-operator TTS-Cal candidate dimensions differ")
        expected = content_sha256(
            {
                "authority_sha256": (
                    current.protocol_lock.tts_calibration_authority_sha256
                ),
                "learning_rate": learning_rate,
                "stride": stride,
            }
        )
        if current.cell.recipe_sha256 != expected:
            raise ValueError("single-operator TTS-Cal recipe differs")
        adaptation = _tts_adaptation(
            stage="tts-cal",
            cell=current.cell,
            width=width,
            learning_rate=learning_rate,
            stride=stride,
        )
    elif role in {"TTS", "L0-naive"}:
        learning_rate, stride, candidate = _tts_selection(current)
        if current.cell.recipe_sha256 != candidate:
            raise ValueError("single-operator frozen TTS recipe differs")
        adaptation = _tts_adaptation(
            stage=current.source.stage,
            cell=current.cell,
            width=width,
            learning_rate=learning_rate,
            stride=stride,
        )
    elif role == "LightCone-candidate" and current.source.stage == "E1":
        adaptation = _e1_candidate_adaptation(
            current.cell,
            width=width,
            recipe_anchor_authority_sha256=(
                current.protocol_lock.e1_recipe_anchor_authority_sha256
            ),
        )
    elif role == "LightCone-candidate" and current.source.stage == "E2":
        adaptation = _e2_candidate_adaptation(
            current,
            width=width,
            preflight_inputs=preflight_inputs,
        )
    config = RunConfig(
        method=method,  # type: ignore[arg-type]
        model=ModelPair.model_validate(model.model_dump(mode="json")),
        runtime=RuntimeConfig.model_validate(runtime.model_dump(mode="json")),
        adaptation=adaptation,
    )
    return config


def _port(current: _CurrentEarlySource) -> int:
    return (
        20_000
        + int(
            _sha256(
                {
                    "kind": "formal_single_operator_early_port",
                    "execution_source_sha256": current.source.sha256,
                    "cell_id": current.cell.cell_id,
                }
            )[:8],
            16,
        )
        % 40_000
    )


def _launch_value(
    *,
    current: _CurrentEarlySource,
    sources: _PreflightLaunchSources,
    config: RunConfig,
    config_path: Path,
    cache_plan: CompileCacheLaunchPlan,
    cache_path: Path,
    server_argv: tuple[str, ...],
) -> CompileLaunchManifest:
    template = sources.template_launch
    config_binding = CanonicalJsonProofBinding.bind(config_path)
    cache_binding = CanonicalJsonProofBinding.bind(cache_path)
    target_only = config.method == "target_only"
    launch = replace(
        template,
        protocol_sha256=COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
        run_config_path=str(config_path),
        run_config_raw_sha256=config_binding.raw_sha256,
        run_config_semantic_sha256=run_config_sha256(config),
        compile_cache_plan_path=str(cache_path),
        compile_cache_plan_raw_sha256=cache_binding.raw_sha256,
        compile_cache_plan_sha256=cache_plan.sha256,
        drafter_content_member_id=(
            None if target_only else template.drafter_content_member_id
        ),
        drafter_model_id=None if target_only else template.drafter_model_id,
        drafter_snapshot_path=(None if target_only else template.drafter_snapshot_path),
        drafter_revision=None if target_only else template.drafter_revision,
        drafter_content_authority_sha256=(
            None if target_only else template.drafter_content_authority_sha256
        ),
        server_argv=server_argv,
        server_argv_sha256=_sha256({"argv": list(server_argv)}),
        localhost_port=_port(current),
        physical_assignment_sha256=_sha256(
            {
                "kind": "formal_single_operator_early_assignment",
                "execution_source_sha256": current.source.sha256,
                "cell_id": current.cell.cell_id,
                "inventory_sha256": sources.inventory.sha256,
                "gpu_uuids": template.gpu_uuids,
            }
        ),
        experiment_budget_sha256=_sha256(
            {
                "kind": "formal_single_operator_early_budget_subject",
                "materialization_sha256": current.materialization.sha256,
                "cell_id": current.cell.cell_id,
            }
        ),
        budget_materialization_authority_sha256=current.source.sha256,
        inventory_sha256=sources.inventory.sha256,
        gpu_uuids=template.gpu_uuids,
    )
    launch.validate(reopen_inputs=True)
    return launch


@dataclass(frozen=True)
class FormalSingleOperatorEarlyLaunchContext:
    execution_source: FormalSingleOperatorExecutionSource
    materialization: StageMaterializationReceipt
    cell: MaterializedCell
    run_config: RunConfig
    launch: CompileLaunchManifest
    inventory: GpuInventory

    @cached_property
    def sha256(self) -> str:
        return _sha256(
            {
                "execution_source_sha256": self.execution_source.sha256,
                "materialization_sha256": self.materialization.sha256,
                "cell_id": self.cell.cell_id,
                "run_config_sha256": run_config_sha256(self.run_config),
                "launch_sha256": self.launch.sha256,
                "inventory_sha256": self.inventory.sha256,
            }
        )


def _derive_launch_context(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    preflight_inputs_path: str | Path,
    output_root: Path | None,
    launch_path: Path | None,
) -> FormalSingleOperatorEarlyLaunchContext:
    current = _current_source(
        execution_source_path,
        materialized_cell_id=materialized_cell_id,
    )
    sources = _preflight_launch_sources(
        current,
        preflight_inputs_path=preflight_inputs_path,
    )
    template = sources.template_launch
    gpu_uuid = template.gpu_uuids[0]
    config = _run_config(
        current,
        template=template,
        gpu_uuid=gpu_uuid,
        preflight_inputs=sources.inputs,
    )
    if output_root is not None:
        paths = (
            output_root / "formal-single-operator-early-compile-cache-plan.json",
            output_root / "formal-single-operator-early-compile-launch.json",
            output_root / config.method / "run-config.json",
        )
        if any(path.exists() or Path(f"{path}.sha256").exists() for path in paths):
            raise RuntimeError(
                "single-operator early mapper refuses to replace outputs"
            )
        template_plan = CompileCacheLaunchPlan.load(template.compile_cache_plan_path)
        key = replace(
            template_plan.key,
            target_revision=config.model.target_revision,
            drafter_revision=(
                None
                if config.method == "target_only"
                else config.model.drafter_revision
            ),
            tensor_parallel_size=1,
            context_limit=config.runtime.context_length,
            max_running_requests=config.runtime.max_running_requests,
            graph_buckets=(1,),
        )
        cache_plan = CompileCacheLaunchPlan.issue(
            key=key,
            cache_root=output_root / "compile-cache",
            cache_mode="build",
        )
        cache_path = (
            output_root / "formal-single-operator-early-compile-cache-plan.json"
        )
        cache_plan.write(cache_path)
        roots = {template.target_model_id: template.target_snapshot_path}
        if (
            template.drafter_model_id is not None
            and template.drafter_snapshot_path is not None
        ):
            roots[template.drafter_model_id] = template.drafter_snapshot_path
        rendered = _render_server(
            output=output_root,
            method=config.method,
            config=config,
            verified_checkout=Path(template.patched_sglang_checkout),
            roots=roots,
            target_id=template.target_model_id,
            drafter_id=str(template.drafter_model_id),
            adaptation_reserve_mb=(
                0 if config.adaptation is None else _ADAPTATION_RESERVE_MB
            ),
            mem_fraction_static=float(
                _flag_value(template.server_argv, "--mem-fraction-static")
            ),
            host="127.0.0.1",
            port=_port(current),
            compile_cache_plan_path=cache_path,
        )
        config_path = Path(rendered.run_config).resolve()
        launch = _launch_value(
            current=current,
            sources=sources,
            config=config,
            config_path=config_path,
            cache_plan=cache_plan,
            cache_path=cache_path,
            server_argv=rendered.argv,
        )
        launch.write(output_root / "formal-single-operator-early-compile-launch.json")
    else:
        assert launch_path is not None
        launch = CompileLaunchManifest.load(launch_path)
        config_path = Path(launch.run_config_path)
        observed_config = load_run_config(config_path)
        if observed_config != config:
            raise ValueError("single-operator early RunConfig output changed")
        cache_path = Path(launch.compile_cache_plan_path)
        cache_plan = CompileCacheLaunchPlan.load(cache_path)
        validate_compile_key_for_run_config(cache_plan, config=config)
        roots = {template.target_model_id: template.target_snapshot_path}
        if (
            template.drafter_model_id is not None
            and template.drafter_snapshot_path is not None
        ):
            roots[template.drafter_model_id] = template.drafter_snapshot_path
        expected_argv = _render_server(
            output=launch_path.parent,
            method=config.method,
            config=config,
            verified_checkout=Path(template.patched_sglang_checkout),
            roots=roots,
            target_id=template.target_model_id,
            drafter_id=str(template.drafter_model_id),
            adaptation_reserve_mb=(
                0 if config.adaptation is None else _ADAPTATION_RESERVE_MB
            ),
            mem_fraction_static=float(
                _flag_value(template.server_argv, "--mem-fraction-static")
            ),
            host="127.0.0.1",
            port=_port(current),
            compile_cache_plan_path=cache_path,
        ).argv
        expected = _launch_value(
            current=current,
            sources=sources,
            config=config,
            config_path=config_path,
            cache_plan=cache_plan,
            cache_path=cache_path,
            server_argv=expected_argv,
        )
        if launch != expected or launch.server_argv != expected_argv:
            raise ValueError("single-operator early launch differs from mapper")
    return FormalSingleOperatorEarlyLaunchContext(
        execution_source=current.source,
        materialization=current.materialization,
        cell=current.cell,
        run_config=config,
        launch=launch,
        inventory=sources.inventory,
    )


def materialize_formal_single_operator_early_compile_launch(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    preflight_inputs_path: str | Path,
    private_output_root: str | Path,
) -> FormalSingleOperatorEarlyLaunchContext:
    """Publish one source-owned E3a/TTS-Cal/E1/E2 launch."""

    root = _absolute_existing_directory(
        "single-operator early private output root",
        private_output_root,
    )
    return _derive_launch_context(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
        preflight_inputs_path=preflight_inputs_path,
        output_root=root,
        launch_path=None,
    )


def revalidate_formal_single_operator_early_compile_launch(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    preflight_inputs_path: str | Path,
    compile_launch_manifest_path: str | Path,
) -> FormalSingleOperatorEarlyLaunchContext:
    """Rebuild one early launch and byte-compare its source-owned values."""

    requested = Path(compile_launch_manifest_path)
    path = requested.resolve(strict=False)
    if (
        not requested.is_absolute()
        or requested != path
        or path.name != "formal-single-operator-early-compile-launch.json"
    ):
        raise ValueError("single-operator early launch path differs")
    return _derive_launch_context(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
        preflight_inputs_path=preflight_inputs_path,
        output_root=None,
        launch_path=path,
    )


@dataclass(frozen=True)
class FormalSingleOperatorEarlyRunPlanInputs:
    """Ordinary current-run inputs consumed by the direct trusted operator."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_early_run_plan_inputs"]
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    materialized_cell_id: str
    stage: Literal["E3a", "TTS-Cal", "E1", "E2"]
    materialization: CanonicalJsonProofBinding
    materialization_sha256: str
    preflight_inputs: CanonicalJsonProofBinding
    compile_launch_manifest: CanonicalJsonProofBinding
    private_output_root: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_early_run_plan_inputs"
            or self.stage not in _EARLY_STAGES
        ):
            raise ValueError("single-operator early plan inputs schema differs")
        for binding in (
            self.execution_source,
            self.materialization,
            self.preflight_inputs,
            self.compile_launch_manifest,
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("single-operator early plan input is not path-bound")
            if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                raise ValueError("single-operator early plan input changed")
        source = FormalSingleOperatorExecutionSource.from_dict(
            self.execution_source.reopen()
        )
        materialization = stage_materialization_receipt_from_dict(
            self.materialization.reopen()
        )
        preflight_inputs = FormalPreflightExecutionInputs.from_dict(
            self.preflight_inputs.reopen()
        )
        if (
            content_sha256(source.to_dict()) != self.execution_source.semantic_sha256
            or source.sha256 != self.execution_source_sha256
            or materialization.sha256 != self.materialization_sha256
            or source.materialization_sha256 != materialization.sha256
            or preflight_inputs.sha256 != self.preflight_inputs.semantic_sha256
        ):
            raise ValueError("single-operator early plan input identities differ")
        _absolute_existing_directory(
            "single-operator early plan private root",
            self.private_output_root,
        )

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "execution_source": self.execution_source.to_dict(),
            "execution_source_sha256": self.execution_source_sha256,
            "materialized_cell_id": self.materialized_cell_id,
            "stage": self.stage,
            "materialization": self.materialization.to_dict(),
            "materialization_sha256": self.materialization_sha256,
            "preflight_inputs": self.preflight_inputs.to_dict(),
            "compile_launch_manifest": self.compile_launch_manifest.to_dict(),
            "private_output_root": self.private_output_root,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("single-operator early plan input fields differ")
        row = dict(value)
        for name in (
            "execution_source",
            "materialization",
            "preflight_inputs",
            "compile_launch_manifest",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        return cls(**row)  # type: ignore[arg-type]


def materialize_formal_single_operator_early_run_plan_inputs(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    preflight_inputs_path: str | Path,
    private_output_root: str | Path,
) -> FormalSingleOperatorEarlyRunPlanInputs:
    """Publish mapper-owned launch plus ordinary direct-plan inputs."""

    root = _absolute_existing_directory(
        "single-operator early private output root",
        private_output_root,
    )
    context = materialize_formal_single_operator_early_compile_launch(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
        preflight_inputs_path=preflight_inputs_path,
        private_output_root=root,
    )
    value = FormalSingleOperatorEarlyRunPlanInputs(
        schema_version=1,
        kind="formal_single_operator_early_run_plan_inputs",
        execution_source=CanonicalJsonProofBinding.bind(execution_source_path),
        execution_source_sha256=context.execution_source.sha256,
        materialized_cell_id=context.cell.cell_id,
        stage=context.materialization.stage,  # type: ignore[arg-type]
        materialization=CanonicalJsonProofBinding.bind(
            context.execution_source.materialization_source.absolute_path
        ),
        materialization_sha256=context.materialization.sha256,
        preflight_inputs=CanonicalJsonProofBinding.bind(preflight_inputs_path),
        compile_launch_manifest=CanonicalJsonProofBinding.bind(
            root / "formal-single-operator-early-compile-launch.json"
        ),
        private_output_root=str(root),
    )
    output = root / "formal-single-operator-early-run-plan-inputs.json"
    publish_canonical_json_no_replace(output, value.to_dict())
    rebound = revalidate_formal_single_operator_early_run_plan_inputs(output)
    if rebound != value:
        raise RuntimeError("single-operator early plan inputs changed on publication")
    return value


def revalidate_formal_single_operator_early_run_plan_inputs(
    path: str | Path,
) -> FormalSingleOperatorEarlyRunPlanInputs:
    """Reopen the ordinary direct-plan descriptor and exact launch mapping."""

    binding = CanonicalJsonProofBinding.bind(path)
    value = FormalSingleOperatorEarlyRunPlanInputs.from_dict(binding.reopen())
    if value.sha256 != binding.semantic_sha256:
        raise ValueError("single-operator early plan input digest differs")
    context = revalidate_formal_single_operator_early_compile_launch(
        execution_source_path=value.execution_source.absolute_path,
        materialized_cell_id=value.materialized_cell_id,
        preflight_inputs_path=value.preflight_inputs.absolute_path,
        compile_launch_manifest_path=(value.compile_launch_manifest.absolute_path),
    )
    if (
        context.execution_source.sha256 != value.execution_source_sha256
        or context.materialization.sha256 != value.materialization_sha256
        or context.cell.cell_id != value.materialized_cell_id
        or context.materialization.stage != value.stage
    ):
        raise ValueError("single-operator early plan input replay differs")
    return value


__all__ = [
    "FormalSingleOperatorEarlyLaunchContext",
    "FormalSingleOperatorEarlyRunPlanInputs",
    "materialize_formal_single_operator_early_compile_launch",
    "materialize_formal_single_operator_early_run_plan_inputs",
    "revalidate_formal_single_operator_early_compile_launch",
    "revalidate_formal_single_operator_early_run_plan_inputs",
]
