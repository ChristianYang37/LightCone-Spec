"""Source-owned E4 headline execution mapping for ``formal_single_operator_v1``.

Only the strength-2 screen and the winner-neighbourhood local factorial are
handled here.  The mapper starts from the exact current execution source and
the immediate completed predecessor, reopens the E2 winner's actual run, and
derives the current :class:`RunConfig`, compile launch, and physical run plan.
There is intentionally no caller input for a recipe, port, argv, topology, or
GPU assignment.  Profiler rows remain outside this mapper.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

from lightcone_spec.config import RunConfig, load_run_config, run_config_sha256
from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorExecutionSource,
    FormalSingleOperatorValidatedActual,
    RebuiltFormalSingleOperatorStageCompletion,
    _e2_recipe_from_payload,
    _e4_configuration_from_payload,
    load_formal_single_operator_execution_source,
    rebuild_formal_single_operator_stage_completion,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.registry import E3A_CONCURRENCY_GRID
from lightcone_spec.experiments.stage_materialization import (
    E4_SCREEN_FACTOR_LEVELS,
    E2CandidateRecipe,
    MaterializedCell,
    StageMaterializationReceipt,
    default_e2_recipe_grid_authority,
)
from lightcone_spec.orchestration.formal_physical_dispatch import (
    FormalServingRunPlan,
    materialize_formal_serving_run_plan,
)
from lightcone_spec.orchestration.runtime import (
    _adaptation_mechanism_argv,
    _execution_argv,
    _execution_role,
    _render_server,
)
from lightcone_spec.runtime.compile_cache import (
    CompileCacheLaunchPlan,
    validate_compile_key_for_run_config,
)
from lightcone_spec.runtime.compile_runner import (
    COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    CompileLaunchManifest,
)
from lightcone_spec.runtime.formal_single_operator import (
    FormalSingleOperatorRunManifest,
    revalidate_formal_single_operator_run_manifest,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding
from lightcone_spec.sglang_bridge.config import sglang_adaptation_payload


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


FORMAL_SINGLE_OPERATOR_E4_EXECUTION_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e4_headline_execution_mapper",
        "current_source": "exact_screen_or_local_materialization",
        "recipe": "immediate_chain_e2_r3_winner_actual",
        "screen_loads": {
            "low": 1,
            "moderate": "e3a_common_load",
            "saturation": max(E3A_CONCURRENCY_GRID),
        },
        "local_template": "same_load_traffic_screen_winner_actual",
        "configuration": "e2_winner_plus_exact_current_e4_factor_row",
        "assignment": "validated_predecessor_actual_gpu_and_toolchain",
        "port": "sha256(current_execution_source,current_cell)_20000_59999",
        "argv": "source_renderer_only",
        "profiler": "blocked",
    }
)


class FormalSingleOperatorE4ExecutionBlocked(RuntimeError):
    """The current node is not an executable E4 headline row."""


@dataclass(frozen=True)
class FormalSingleOperatorE4LaunchContext:
    execution_source: FormalSingleOperatorExecutionSource
    materialization: StageMaterializationReceipt
    cell: MaterializedCell
    predecessor: RebuiltFormalSingleOperatorStageCompletion
    e2_completion: RebuiltFormalSingleOperatorStageCompletion
    lightcone_recipe: E2CandidateRecipe
    recipe_actual: FormalSingleOperatorValidatedActual
    assignment_actual: FormalSingleOperatorValidatedActual
    recipe_manifest: FormalSingleOperatorRunManifest
    assignment_manifest: FormalSingleOperatorRunManifest
    run_config: RunConfig
    launch: CompileLaunchManifest
    load_concurrency: int
    traffic: str

    @cached_property
    def sha256(self) -> str:
        return _sha256(
            {
                "protocol_sha256": (
                    FORMAL_SINGLE_OPERATOR_E4_EXECUTION_PROTOCOL_SHA256
                ),
                "execution_source_sha256": self.execution_source.sha256,
                "materialization_sha256": self.materialization.sha256,
                "cell_id": self.cell.cell_id,
                "predecessor_completion_sha256": self.predecessor.artifact.sha256,
                "e2_completion_sha256": self.e2_completion.artifact.sha256,
                "lightcone_recipe_sha256": self.lightcone_recipe.sha256,
                "recipe_actual_sha256": self.recipe_actual.sha256,
                "assignment_actual_sha256": self.assignment_actual.sha256,
                "recipe_manifest_sha256": self.recipe_manifest.sha256,
                "assignment_manifest_sha256": self.assignment_manifest.sha256,
                "run_config_sha256": run_config_sha256(self.run_config),
                "compile_launch_manifest_sha256": self.launch.sha256,
                "load_concurrency": self.load_concurrency,
                "traffic": self.traffic,
            }
        )


@dataclass(frozen=True)
class FormalSingleOperatorE4RunPlan:
    execution_binding: object
    launch_context: FormalSingleOperatorE4LaunchContext
    run_plan: FormalServingRunPlan


def _current_source(
    execution_source_path: str | Path,
    *,
    materialized_cell_id: str,
) -> tuple[
    FormalSingleOperatorExecutionSource,
    StageMaterializationReceipt,
    MaterializedCell,
    RebuiltFormalSingleOperatorStageCompletion,
]:
    source = load_formal_single_operator_execution_source(execution_source_path)
    if source.node not in {"e4_screen", "e4_local"}:
        raise FormalSingleOperatorE4ExecutionBlocked(
            "only E4 screen/local headline rows use this mapper"
        )
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="single-operator E4 current materialization"
        )
    )
    matches = tuple(
        row for row in materialization.cells if row.cell_id == materialized_cell_id
    )
    if len(matches) != 1:
        raise ValueError("single-operator E4 cell is outside current materialization")
    cell = matches[0]
    if (
        materialization.stage != "E4"
        or cell.task == "mechanism_profile_only"
        or cell.task
        not in {
            "mechanism_strength2_screen_headline",
            "winner_neighborhood_local_factorial_headline",
        }
        or source.predecessor_completion_source is None
    ):
        raise FormalSingleOperatorE4ExecutionBlocked(
            "E4 profiler or non-headline execution requires another mapper"
        )
    predecessor = rebuild_formal_single_operator_stage_completion(
        source.predecessor_completion_source.absolute_path
    )
    if predecessor.artifact.sha256 != source.predecessor_completion_sha256:
        raise ValueError("single-operator E4 predecessor changed")
    return source, materialization, cell, predecessor


def _e2_completion(
    source: FormalSingleOperatorExecutionSource,
    predecessor: RebuiltFormalSingleOperatorStageCompletion,
) -> RebuiltFormalSingleOperatorStageCompletion:
    if source.node == "e4_screen":
        if predecessor.artifact.node != "e2_r3":
            raise ValueError("E4 screen does not immediately follow E2 round3")
        return predecessor
    if (
        predecessor.artifact.node != "e4_screen"
        or predecessor.predecessor is None
        or predecessor.predecessor.artifact.node != "e2_r3"
    ):
        raise ValueError("E4 local lacks its immediate screen/E2 chain")
    return predecessor.predecessor


def _recipe_from_e2(
    e2: RebuiltFormalSingleOperatorStageCompletion,
) -> E2CandidateRecipe:
    payload = e2.decision.payload
    if payload.get("round_index") != 3:
        raise ValueError("single-operator E4 recipe source is not E2 round3")
    recipe = _e2_recipe_from_payload(payload.get("final_recipe"))
    if type(recipe) is not E2CandidateRecipe:
        raise TypeError("single-operator E4 recipe is not an E2 candidate")
    return recipe


def _actual_for_cell(
    completion: RebuiltFormalSingleOperatorStageCompletion,
    *,
    cell_id: str,
) -> FormalSingleOperatorValidatedActual:
    matches = tuple(
        row for row in completion.artifact.actual_results if row.cell_id == cell_id
    )
    if len(matches) != 1 or matches[0].status != "COMPLETE":
        raise ValueError("single-operator predecessor lacks one complete actual")
    return matches[0]


def _recipe_actual(
    e2: RebuiltFormalSingleOperatorStageCompletion,
    recipe: E2CandidateRecipe,
) -> FormalSingleOperatorValidatedActual:
    matches = tuple(
        row
        for row in e2.materialization.cells
        if row.method_role == "LightCone-candidate"
        and row.recipe_sha256 == recipe.sha256
    )
    if len(matches) != 1:
        raise ValueError("E2 round3 does not contain one winning recipe cell")
    return _actual_for_cell(e2, cell_id=matches[0].cell_id)


def _configuration(cell: MaterializedCell) -> tuple[tuple[str, str | int], ...]:
    dimensions = dict(cell.dimensions)
    configuration = tuple(
        (name, dimensions[name]) for name, _levels in E4_SCREEN_FACTOR_LEVELS
    )
    if any(type(value) not in {str, int} for _name, value in configuration):
        raise ValueError("E4 cell lacks its exact operational factors")
    return configuration  # type: ignore[return-value]


def _assignment_actual(
    *,
    source: FormalSingleOperatorExecutionSource,
    predecessor: RebuiltFormalSingleOperatorStageCompletion,
    recipe_actual: FormalSingleOperatorValidatedActual,
    cell: MaterializedCell,
) -> FormalSingleOperatorValidatedActual:
    if source.node == "e4_screen":
        return recipe_actual
    winner = _e4_configuration_from_payload(
        predecessor.decision.payload.get("winner_configuration")
    )
    dimensions = dict(cell.dimensions)
    matches = tuple(
        row
        for row in predecessor.materialization.cells
        if dict(row.dimensions).get("load") == dimensions.get("load")
        and dict(row.dimensions).get("traffic") == dimensions.get("traffic")
        and _configuration(row) == winner
    )
    if len(matches) != 1:
        raise ValueError("E4 local lacks one same-stratum screen winner actual")
    return _actual_for_cell(predecessor, cell_id=matches[0].cell_id)


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
        raise ValueError(f"single-operator predecessor lacks {name} artifact")
    return Path(manifest.run_directory) / matches[0].relative_path


def _validated_manifest(
    actual: FormalSingleOperatorValidatedActual,
    *,
    repository_root: str | Path,
    completion: RebuiltFormalSingleOperatorStageCompletion,
) -> tuple[FormalSingleOperatorRunManifest, CompileLaunchManifest, RunConfig]:
    manifest = revalidate_formal_single_operator_run_manifest(
        repository_root=repository_root,
        manifest_path=actual.source.absolute_path,
    )
    cell = next(
        row for row in completion.materialization.cells if row.cell_id == actual.cell_id
    )
    if (
        manifest.sha256 != actual.result_identity_sha256
        or manifest.completion_status != "COMPLETE"
        or manifest.stage != completion.materialization.stage
        or manifest.cell_id != cell.cell_id
        or manifest.materialization_sha256 != completion.materialization.sha256
        or manifest.materialization_protocol_lock_sha256
        != completion.materialization.protocol_lock_sha256
        or manifest.role != cell.method_role
        or manifest.backend != cell.backend
        or manifest.target_model_id != cell.model
    ):
        raise ValueError("single-operator predecessor manifest lineage differs")
    plan_path = _manifest_artifact(manifest, "run_plan")
    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    if (
        plan.sha256 != plan_binding.semantic_sha256
        or plan.sha256 != manifest.run_plan_sha256
        or plan.stage != manifest.stage
        or plan.materialized_cell_id != manifest.cell_id
        or plan.execution_binding_sha256 != manifest.execution_binding_sha256
        or plan.subject_sha256 != manifest.execution_subject_sha256
    ):
        raise ValueError("single-operator predecessor run plan differs")
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    config = load_run_config(launch.run_config_path)
    compile_cache_plan = CompileCacheLaunchPlan.load(launch.compile_cache_plan_path)
    validate_compile_key_for_run_config(compile_cache_plan, config=config)
    if (
        launch.sha256 != plan.launch_manifest.semantic_sha256
        or launch.sha256 != manifest.launch_manifest_sha256
        or launch.server_argv != manifest.launch_argv
        or launch.server_argv_sha256 != manifest.launch_argv_sha256
        or launch.localhost_port != manifest.localhost_port
        or run_config_sha256(config) != manifest.run_config_sha256
        or config.model_dump(mode="json") != manifest.run_config
    ):
        raise ValueError("single-operator predecessor launch/config differs")
    return manifest, launch, config


def _screen_loads(e2: RebuiltFormalSingleOperatorStageCompletion) -> dict[str, int]:
    common_load = e2.decision.payload.get("common_load")
    if type(common_load) is not int or common_load not in E3A_CONCURRENCY_GRID:
        raise ValueError("E2 round3 lacks the selected E3a common load")
    return {
        "low": 1,
        "moderate": common_load,
        "saturation": max(E3A_CONCURRENCY_GRID),
    }


def _expected_run_config(
    *,
    source: FormalSingleOperatorExecutionSource,
    cell: MaterializedCell,
    e2: RebuiltFormalSingleOperatorStageCompletion,
    recipe: E2CandidateRecipe,
    recipe_config: RunConfig,
    gpu_uuid: str,
) -> tuple[RunConfig, int, str]:
    dimensions = dict(cell.dimensions)
    load = dimensions.get("load")
    traffic = dimensions.get("traffic")
    if load not in {"low", "moderate", "saturation"} or traffic not in {
        "pure_decode",
        "mixed_prefill_decode",
    }:
        raise ValueError("E4 headline cell lacks its fixed load/traffic stratum")
    if (
        recipe_config.method != "l0"
        or recipe_config.model.algorithm != "DFLASH"
        or recipe_config.adaptation is None
        or recipe_config.online_spec is not None
        or recipe_config.runtime.topology_mode != "tp1_dp1"
    ):
        raise ValueError("E2 winning actual is not the LightCone TP1 recipe")
    matched_width = e2.decision.payload.get("matched_width")
    common_load = e2.decision.payload.get("common_load")
    if (
        type(matched_width) is not int
        or matched_width != recipe_config.runtime.speculative_num_draft_tokens
        or type(common_load) is not int
        or common_load != recipe_config.runtime.max_running_requests
    ):
        raise ValueError("E2 winning actual differs from its selected width/load")
    grid = default_e2_recipe_grid_authority()
    adaptation = recipe_config.adaptation
    expected_adaptation = grid.adaptation_config_for(
        recipe,
        canvas_tokens=adaptation.canvas_tokens,
        adaptation_group_id=adaptation.adaptation_group_id,
        chronobelief_gpu_proof_sha256=adaptation.chronobelief_gpu_proof_sha256,
    )
    if adaptation != expected_adaptation:
        raise ValueError("E2 winning actual config differs from its exact recipe")
    concurrency = _screen_loads(e2)[str(load)]
    update_stride = dimensions.get("update_stride")
    microbatch = dimensions.get("microbatch")
    coalescing = dimensions.get("coalescing")
    stream_priority = dimensions.get("stream_priority")
    if (
        type(update_stride) is not int
        or update_stride not in {1, 5, 10, 15, 20, 30, 40, 50}
        or type(microbatch) is not int
        or microbatch not in {1, 2, 4, 8}
        or type(coalescing) is not int
        or coalescing not in {1, 2, 4, 8}
        or stream_priority not in {"default", "high"}
    ):
        raise ValueError("E4 current factor row is outside runtime schema")
    current_adaptation = adaptation.model_copy(
        update={
            "adaptation_group_id": f"formal-single-e4-{cell.cell_id[:24]}",
            "stride": update_stride,
        }
    )
    current_runtime = recipe_config.runtime.model_copy(
        update={
            "device_identity": gpu_uuid,
            "max_running_requests": concurrency,
            "telemetry_detail": "headline",
            "adaptation_microbatch_size": microbatch,
            "adaptation_publication_coalescing": coalescing,
            "adaptation_stream_priority": stream_priority,
        }
    )
    result = recipe_config.model_copy(
        update={"runtime": current_runtime, "adaptation": current_adaptation}
    )
    result = RunConfig.model_validate(result.model_dump(mode="json"))
    if result.model.target != cell.model or cell.recipe_sha256 != recipe.sha256:
        raise ValueError("E4 current cell differs from E2 winner recipe/model")
    return result, concurrency, str(traffic)


def _flag_value(argv: tuple[str, ...], flag: str) -> str:
    positions = tuple(index for index, value in enumerate(argv) if value == flag)
    if (
        len(positions) != 1
        or positions[0] + 1 >= len(argv)
        or argv[positions[0] + 1].startswith("--")
    ):
        raise ValueError(f"single-operator predecessor argv lacks exact {flag}")
    return argv[positions[0] + 1]


def _port(source: FormalSingleOperatorExecutionSource, cell: MaterializedCell) -> int:
    identity = _sha256(
        {
            "protocol_sha256": FORMAL_SINGLE_OPERATOR_E4_EXECUTION_PROTOCOL_SHA256,
            "execution_source_sha256": source.sha256,
            "cell_id": cell.cell_id,
        }
    )
    return 20_000 + int(identity[:8], 16) % 40_000


def _derive_context(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    repository_root: str | Path,
    inventory_path: str | Path,
    launch_path: Path | None,
    output_root: Path | None,
) -> FormalSingleOperatorE4LaunchContext:
    source, materialization, cell, predecessor = _current_source(
        execution_source_path,
        materialized_cell_id=materialized_cell_id,
    )
    e2 = _e2_completion(source, predecessor)
    recipe = _recipe_from_e2(e2)
    recipe_actual = _recipe_actual(e2, recipe)
    assignment_actual = _assignment_actual(
        source=source,
        predecessor=predecessor,
        recipe_actual=recipe_actual,
        cell=cell,
    )
    recipe_manifest, _recipe_launch, recipe_config = _validated_manifest(
        recipe_actual,
        repository_root=repository_root,
        completion=e2,
    )
    assignment_completion = e2 if source.node == "e4_screen" else predecessor
    assignment_manifest, assignment_launch, _assignment_config = _validated_manifest(
        assignment_actual,
        repository_root=repository_root,
        completion=assignment_completion,
    )
    inventory_binding = CanonicalJsonProofBinding.bind(inventory_path)
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    if inventory.sha256 != inventory_binding.semantic_sha256:
        raise ValueError("single-operator E4 inventory identity differs")
    if (
        assignment_launch.inventory_sha256 != inventory.sha256
        or assignment_manifest.inventory_sha256 != inventory.sha256
        or len(assignment_launch.gpu_uuids) != 1
        or assignment_launch.target_content_authority_sha256
        != source.prepared_model_content_authorization_sha256
        or assignment_launch.drafter_content_authority_sha256
        != source.prepared_model_content_authorization_sha256
        or assignment_launch.tokenizer_content_authority_sha256
        != source.prepared_model_content_authorization_sha256
    ):
        raise ValueError(
            "E4 predecessor assignment/content differs from current source"
        )
    gpu_uuid = assignment_launch.gpu_uuids[0]
    inventory.device(gpu_uuid)
    config, concurrency, traffic = _expected_run_config(
        source=source,
        cell=cell,
        e2=e2,
        recipe=recipe,
        recipe_config=recipe_config,
        gpu_uuid=gpu_uuid,
    )
    if launch_path is None:
        if output_root is None:
            raise AssertionError("E4 launch derivation lost its output root")
        launch = _materialize_launch(
            output_root=output_root,
            source=source,
            materialization=materialization,
            cell=cell,
            config=config,
            assignment_launch=assignment_launch,
        )
    else:
        launch = CompileLaunchManifest.load(launch_path)
        expected_root = launch_path.parent
        expected = _expected_existing_launch(
            output_root=expected_root,
            source=source,
            materialization=materialization,
            cell=cell,
            config=config,
            assignment_launch=assignment_launch,
        )
        if launch != expected or launch.sha256 != expected.sha256:
            raise ValueError("single-operator E4 launch differs from source mapper")
    return FormalSingleOperatorE4LaunchContext(
        execution_source=source,
        materialization=materialization,
        cell=cell,
        predecessor=predecessor,
        e2_completion=e2,
        lightcone_recipe=recipe,
        recipe_actual=recipe_actual,
        assignment_actual=assignment_actual,
        recipe_manifest=recipe_manifest,
        assignment_manifest=assignment_manifest,
        run_config=config,
        launch=launch,
        load_concurrency=concurrency,
        traffic=traffic,
    )


def _expected_server_argv(
    *,
    output_root: Path,
    source: FormalSingleOperatorExecutionSource,
    cell: MaterializedCell,
    config: RunConfig,
    assignment_launch: CompileLaunchManifest,
    cache_plan: CompileCacheLaunchPlan,
) -> tuple[str, ...]:
    config_path = output_root / "l0" / "run-config.json"
    adaptation_path = output_root / "l0" / "adaptation-config.json"
    telemetry_path = output_root / "l0" / "adaptation-telemetry.json"
    cache_path = output_root / "formal-single-operator-e4-compile-cache-plan.json"
    values = [
        sys.executable,
        "-m",
        "lightcone_spec.sglang_bridge.launch",
        "--checkout",
        assignment_launch.patched_sglang_checkout,
        "--compile-cache-plan",
        str(cache_path),
        "--compile-cache-plan-sha256",
        cache_plan.sha256,
        "--compile-cache-key-sha256",
        cache_plan.key.sha256,
        "--run-config",
        str(config_path),
        "--run-config-sha256",
        run_config_sha256(config),
        "--",
        "--model-path",
        assignment_launch.target_snapshot_path,
        "--max-running-requests",
        str(config.runtime.max_running_requests),
        "--mem-fraction-static",
        _flag_value(assignment_launch.server_argv, "--mem-fraction-static"),
        "--tp-size",
        str(config.runtime.tensor_parallel_size),
        "--dtype",
        cache_plan.key.dtype,
        "--host",
        "127.0.0.1",
        "--port",
        str(_port(source, cell)),
    ]
    values.extend(_execution_argv(config.runtime, role=_execution_role("l0")))
    values.extend(
        (
            "--speculative-algorithm",
            config.model.algorithm,
            "--speculative-draft-model-path",
            str(assignment_launch.drafter_snapshot_path),
            "--speculative-num-draft-tokens",
            str(config.runtime.speculative_num_draft_tokens),
            "--speculative-draft-window-size",
            str(config.runtime.speculative_num_draft_tokens),
            "--speculative-accept-threshold-single",
            "1.0",
            "--speculative-accept-threshold-acc",
            "1.0",
            "--speculative-use-rejection-sampling",
            "--speculative-speed-study-metrics",
        )
    )
    values.extend(_adaptation_mechanism_argv(config.runtime))
    values.extend(
        (
            "--speculative-adaptation-config",
            str(adaptation_path),
            "--speculative-adaptation-reserve-mb",
            _flag_value(
                assignment_launch.server_argv,
                "--speculative-adaptation-reserve-mb",
            ),
            "--speculative-adaptation-telemetry-path",
            str(telemetry_path),
        )
    )
    return tuple(values)


def _launch_value(
    *,
    source: FormalSingleOperatorExecutionSource,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
    config: RunConfig,
    assignment_launch: CompileLaunchManifest,
    cache_plan: CompileCacheLaunchPlan,
    cache_path: Path,
    server_argv: tuple[str, ...],
    config_path: Path,
) -> CompileLaunchManifest:
    config_binding = CanonicalJsonProofBinding.bind(config_path)
    cache_binding = CanonicalJsonProofBinding.bind(cache_path)
    launch = replace(
        assignment_launch,
        protocol_sha256=COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
        run_config_path=str(config_path),
        run_config_raw_sha256=config_binding.raw_sha256,
        run_config_semantic_sha256=run_config_sha256(config),
        compile_cache_plan_path=str(cache_path),
        compile_cache_plan_raw_sha256=cache_binding.raw_sha256,
        compile_cache_plan_sha256=cache_plan.sha256,
        server_argv=server_argv,
        server_argv_sha256=_sha256({"argv": list(server_argv)}),
        localhost_port=_port(source, cell),
        physical_assignment_sha256=_sha256(
            {
                "kind": "formal_single_operator_e4_assignment",
                "protocol_sha256": FORMAL_SINGLE_OPERATOR_E4_EXECUTION_PROTOCOL_SHA256,
                "execution_source_sha256": source.sha256,
                "cell_id": cell.cell_id,
                "inventory_sha256": assignment_launch.inventory_sha256,
                "gpu_uuids": assignment_launch.gpu_uuids,
            }
        ),
        experiment_budget_sha256=_sha256(
            {
                "kind": "formal_single_operator_e4_runtime_budget_subject",
                "materialization_sha256": materialization.sha256,
                "cell_id": cell.cell_id,
            }
        ),
        budget_materialization_authority_sha256=source.sha256,
    )
    launch.validate(reopen_inputs=True)
    return launch


def _materialize_launch(
    *,
    output_root: Path,
    source: FormalSingleOperatorExecutionSource,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
    config: RunConfig,
    assignment_launch: CompileLaunchManifest,
) -> CompileLaunchManifest:
    for path in (
        output_root / "formal-single-operator-e4-compile-cache-plan.json",
        output_root / "formal-single-operator-e4-compile-launch.json",
        output_root / "l0" / "run-config.json",
        output_root / "l0" / "adaptation-config.json",
    ):
        if path.exists() or Path(f"{path}.sha256").exists():
            raise RuntimeError("single-operator E4 mapper refuses to replace outputs")
    template_plan = CompileCacheLaunchPlan.load(
        assignment_launch.compile_cache_plan_path
    )
    key = replace(
        template_plan.key,
        max_running_requests=config.runtime.max_running_requests,
    )
    cache_plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=template_plan.cache_root,
        cache_mode="build",
    )
    cache_path = output_root / "formal-single-operator-e4-compile-cache-plan.json"
    cache_plan.write(cache_path)
    server = _render_server(
        output=output_root,
        method="l0",
        config=config,
        verified_checkout=Path(assignment_launch.patched_sglang_checkout),
        roots={
            assignment_launch.target_model_id: assignment_launch.target_snapshot_path,
            str(assignment_launch.drafter_model_id): str(
                assignment_launch.drafter_snapshot_path
            ),
        },
        target_id=assignment_launch.target_model_id,
        drafter_id=str(assignment_launch.drafter_model_id),
        adaptation_reserve_mb=int(
            _flag_value(
                assignment_launch.server_argv,
                "--speculative-adaptation-reserve-mb",
            )
        ),
        mem_fraction_static=float(
            _flag_value(assignment_launch.server_argv, "--mem-fraction-static")
        ),
        host="127.0.0.1",
        port=_port(source, cell),
        compile_cache_plan_path=cache_path,
    )
    launch = _launch_value(
        source=source,
        materialization=materialization,
        cell=cell,
        config=config,
        assignment_launch=assignment_launch,
        cache_plan=cache_plan,
        cache_path=cache_path,
        server_argv=server.argv,
        config_path=Path(server.run_config).resolve(),
    )
    expected_argv = _expected_server_argv(
        output_root=output_root,
        source=source,
        cell=cell,
        config=config,
        assignment_launch=assignment_launch,
        cache_plan=cache_plan,
    )
    if server.argv != expected_argv:
        raise RuntimeError("single-operator E4 renderer differs from mapper protocol")
    launch.write(output_root / "formal-single-operator-e4-compile-launch.json")
    return launch


def _expected_existing_launch(
    *,
    output_root: Path,
    source: FormalSingleOperatorExecutionSource,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
    config: RunConfig,
    assignment_launch: CompileLaunchManifest,
) -> CompileLaunchManifest:
    cache_path = output_root / "formal-single-operator-e4-compile-cache-plan.json"
    config_path = output_root / "l0" / "run-config.json"
    cache_plan = CompileCacheLaunchPlan.load(cache_path)
    validate_compile_key_for_run_config(cache_plan, config=config)
    config_binding = CanonicalJsonProofBinding.bind(config_path)
    if (
        config_binding.semantic_sha256 != run_config_sha256(config)
        or load_run_config(config_path) != config
    ):
        raise ValueError("single-operator E4 RunConfig output changed")
    launch_path = output_root / "formal-single-operator-e4-compile-launch.json"
    observed = CompileLaunchManifest.load(launch_path)
    server_argv = _expected_server_argv(
        output_root=output_root,
        source=source,
        cell=cell,
        config=config,
        assignment_launch=assignment_launch,
        cache_plan=cache_plan,
    )
    expected = _launch_value(
        source=source,
        materialization=materialization,
        cell=cell,
        config=config,
        assignment_launch=assignment_launch,
        cache_plan=cache_plan,
        cache_path=cache_path,
        server_argv=server_argv,
        config_path=config_path,
    )
    if (
        observed.server_argv != server_argv
        or _flag_value(server_argv, "--host") != "127.0.0.1"
        or int(_flag_value(server_argv, "--port")) != _port(source, cell)
        or int(_flag_value(server_argv, "--max-running-requests"))
        != config.runtime.max_running_requests
        or int(_flag_value(server_argv, "--lightcone-adaptation-microbatch-size"))
        != config.runtime.adaptation_microbatch_size
        or int(
            _flag_value(
                server_argv,
                "--lightcone-adaptation-publication-coalescing",
            )
        )
        != config.runtime.adaptation_publication_coalescing
        or _flag_value(server_argv, "--lightcone-adaptation-stream-priority")
        != config.runtime.adaptation_stream_priority
        or _flag_value(server_argv, "--run-config") != str(config_path)
        or _flag_value(server_argv, "--compile-cache-plan") != str(cache_path)
    ):
        raise ValueError("single-operator E4 rendered argv differs")
    adaptation_binding = CanonicalJsonProofBinding.bind(
        output_root / "l0" / "adaptation-config.json"
    )
    if adaptation_binding.reopen() != sglang_adaptation_payload(config):
        raise ValueError("single-operator E4 adaptation config output changed")
    return expected


def materialize_formal_single_operator_e4_compile_launch(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    repository_root: str | Path,
    inventory_path: str | Path,
    private_output_root: str | Path,
) -> FormalSingleOperatorE4LaunchContext:
    """Publish the exact current E4 RunConfig/cache plan/compile launch."""

    requested_root = Path(private_output_root)
    root = requested_root.resolve(strict=False)
    if not requested_root.is_absolute() or requested_root != root:
        raise ValueError(
            "single-operator E4 private output root must be absolute and normalized"
        )
    if not root.is_dir() or root.is_symlink():
        raise ValueError(
            "single-operator E4 private output root must be an existing directory"
        )
    return _derive_context(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
        repository_root=repository_root,
        inventory_path=inventory_path,
        launch_path=None,
        output_root=root,
    )


def revalidate_formal_single_operator_e4_compile_launch(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    repository_root: str | Path,
    inventory_path: str | Path,
    compile_launch_manifest_path: str | Path,
) -> FormalSingleOperatorE4LaunchContext:
    """Rebuild and compare one mapper-owned E4 launch without caller scalars."""

    requested_path = Path(compile_launch_manifest_path)
    launch_path = requested_path.resolve(strict=False)
    if not requested_path.is_absolute() or requested_path != launch_path:
        raise ValueError(
            "single-operator E4 launch path must be absolute and normalized"
        )
    if launch_path.name != "formal-single-operator-e4-compile-launch.json":
        raise ValueError("single-operator E4 launch path name differs")
    return _derive_context(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
        repository_root=repository_root,
        inventory_path=inventory_path,
        launch_path=launch_path,
        output_root=None,
    )


def materialize_formal_single_operator_e4_run_plan(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    repository_root: str | Path,
    formal_runtime_authority_manifest_path: str | Path,
    inventory_path: str | Path,
    content_verification_receipt_path: str | Path,
    workload_authority_path: str | Path,
    runtime_gpu_proof_artifact_paths: tuple[str, ...],
    private_output_root: str | Path,
    now_ns: int,
) -> FormalSingleOperatorE4RunPlan:
    """Materialize the mapper-owned launch, sealed binding, and physical plan."""

    from lightcone_spec.experiments.formal_stage_execution import (
        verify_formal_single_operator_execution_binding,
    )

    context = materialize_formal_single_operator_e4_compile_launch(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
        repository_root=repository_root,
        inventory_path=inventory_path,
        private_output_root=private_output_root,
    )
    launch_path = (
        Path(private_output_root).resolve(strict=False)
        / "formal-single-operator-e4-compile-launch.json"
    )
    binding = verify_formal_single_operator_execution_binding(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
        formal_runtime_authority_manifest_path=(formal_runtime_authority_manifest_path),
        compile_launch_manifest_path=launch_path,
        inventory_path=inventory_path,
        content_verification_receipt_path=content_verification_receipt_path,
        runtime_gpu_proof_artifact_paths=runtime_gpu_proof_artifact_paths,
        repository_root=repository_root,
        now_ns=now_ns,
    )
    plan = materialize_formal_serving_run_plan(
        execution_binding=binding,
        content_verification_receipt_path=content_verification_receipt_path,
        workload_authority_path=workload_authority_path,
        materialization_path=context.execution_source.materialization_source.absolute_path,
        compile_launch_manifest_path=launch_path,
        private_output_root=private_output_root,
        now_ns=now_ns,
    )
    return FormalSingleOperatorE4RunPlan(
        execution_binding=binding,
        launch_context=context,
        run_plan=plan,
    )


__all__ = [
    "FORMAL_SINGLE_OPERATOR_E4_EXECUTION_PROTOCOL_SHA256",
    "FormalSingleOperatorE4ExecutionBlocked",
    "FormalSingleOperatorE4LaunchContext",
    "FormalSingleOperatorE4RunPlan",
    "materialize_formal_single_operator_e4_compile_launch",
    "materialize_formal_single_operator_e4_run_plan",
    "revalidate_formal_single_operator_e4_compile_launch",
]
