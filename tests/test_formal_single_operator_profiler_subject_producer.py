from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import (
    formal_single_operator_profiler_subject_producer as producer,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorJsonBinding,
    FormalSingleOperatorValidatedActual,
    RebuiltFormalSingleOperatorStageCompletion,
    publish_formal_single_operator_json_artifact,
)
from lightcone_spec.experiments.stage_materialization import (
    E4_SCREEN_FACTOR_LEVELS,
    MaterializedCell,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json_binding(tmp_path: Path, name: str) -> CanonicalJsonProofBinding:
    path = (tmp_path / f"{name}.json").resolve()
    publish_canonical_json_no_replace(path, {"kind": "fixture", "name": name})
    return CanonicalJsonProofBinding.bind(path)


def _local_cell() -> MaterializedCell:
    dimensions = {name: levels[0] for name, levels in E4_SCREEN_FACTOR_LEVELS}
    dimensions.update(
        {"load": "saturation", "local_row": 0, "traffic": "mixed_prefill_decode"}
    )
    return MaterializedCell(
        stage="E4",
        method_role="LightCone",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="winner_neighborhood_local_factorial_headline",
        publication_policy="first_ready",
        recipe_sha256=_sha("recipe"),
        dimensions=tuple(sorted(dimensions.items())),
    )


def _actual(
    tmp_path: Path, cell: MaterializedCell
) -> FormalSingleOperatorValidatedActual:
    path = (tmp_path / "formal-single-operator-manifest.json").resolve()
    publish_formal_single_operator_json_artifact(path, {"kind": "fixture-manifest"})
    return FormalSingleOperatorValidatedActual(
        node="e4_local",
        stage="E4",
        materialization_sha256=_sha("local-materialization"),
        cell_id=cell.cell_id,
        status="COMPLETE",
        started_ns=10,
        finished_ns=20,
        result_identity_sha256=_sha("selected-manifest"),
        validator_kind="formal_single_operator_run_manifest_revalidator",
        validator_protocol_sha256=_sha("run-manifest-validator"),
        source=FormalSingleOperatorJsonBinding.bind(path, label="fixture manifest"),
        reducer_payload={"kind": "fixture-serving-observation"},
    )


def _producer_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MaterializedCell, dict[str, object]]:
    cell = _local_cell()
    winner = tuple(
        (name, dict(cell.dimensions)[name]) for name, _levels in E4_SCREEN_FACTOR_LEVELS
    )
    configuration_sha256 = producer._sha256(winner)
    profiler_cells = tuple(
        MaterializedCell(
            stage="E4",
            method_role="LightCone",
            model=cell.model,
            backend="DFLASH",
            task="mechanism_profile_only",
            publication_policy="diagnostic_only",
            recipe_sha256=cell.recipe_sha256,
            dimensions=(
                ("profiler", tool),
                ("selected_configuration_sha256", configuration_sha256),
            ),
        )
        for tool in ("nvtx", "nsight_systems", "nsight_compute")
    )
    predecessor_path = _json_binding(tmp_path, "local-completion")
    current_materialization_path = _json_binding(tmp_path, "profiler-materialization")
    actual = _actual(tmp_path, cell)
    predecessor = RebuiltFormalSingleOperatorStageCompletion(
        artifact=SimpleNamespace(node="e4_local", actual_results=(actual,)),
        predecessor=None,
        node_materialization=SimpleNamespace(
            materialization_source=current_materialization_path
        ),
        materialization=SimpleNamespace(
            cells=(cell,),
            sha256=_sha("local-materialization"),
            protocol_lock_sha256=_sha("lock"),
        ),
        decision=SimpleNamespace(
            payload={"winner_configuration": [[name, value] for name, value in winner]}
        ),
    )
    source = SimpleNamespace(
        node="e4_profiler",
        predecessor_completion_source=SimpleNamespace(
            absolute_path=predecessor_path.absolute_path
        ),
        materialization_source=SimpleNamespace(reopen=lambda **_kwargs: {}),
    )
    current = SimpleNamespace(stage="E4", cells=profiler_cells)
    bindings = {
        "config": _json_binding(tmp_path, "selected-config"),
        "launch": _json_binding(tmp_path, "selected-launch"),
        "workload": _json_binding(tmp_path, "selected-schedule-source"),
        "schedule": _json_binding(tmp_path, "selected-request-schedule"),
    }
    observed: dict[str, object] = {"selected_cells": []}
    monkeypatch.setattr(
        producer,
        "load_formal_single_operator_execution_source",
        lambda _path: source,
    )
    monkeypatch.setattr(
        producer,
        "stage_materialization_receipt_from_dict",
        lambda _value: current,
    )
    monkeypatch.setattr(
        producer,
        "rebuild_formal_single_operator_stage_completion",
        lambda _path: predecessor,
    )

    def selected_run(**kwargs):
        observed["selected_cells"].append(kwargs["cell"])
        assert kwargs["actual"] == actual
        return (
            bindings["config"],
            bindings["launch"],
            bindings["workload"],
            bindings["schedule"],
        )

    monkeypatch.setattr(producer, "_revalidate_selected_run", selected_run)
    observed.update(
        {
            "predecessor": predecessor,
            "current": current,
            "bindings": bindings,
            "configuration_sha256": configuration_sha256,
        }
    )
    return cell, observed


def test_publisher_selects_unique_saturation_mixed_winner_and_refuses_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell, observed = _producer_case(tmp_path, monkeypatch)
    output = (tmp_path / "profiler-subject-requirement.json").resolve()
    requirement = producer.publish_formal_single_operator_profiler_subject_requirement(
        execution_source_path=(tmp_path / "profiler-source.json").resolve(),
        repository_root=tmp_path.resolve(),
        output_path=output,
    )
    bindings = observed["bindings"]
    assert requirement.source_headline_cell_id == cell.cell_id
    assert requirement.selected_configuration_sha256 == observed["configuration_sha256"]
    assert requirement.selected_full_run_config == bindings["config"]
    assert requirement.selected_compile_launch_manifest == bindings["launch"]
    assert requirement.code_owned_profiler_subject_workload == bindings["workload"]
    assert requirement.code_owned_request_schedule == bindings["schedule"]
    assert observed["selected_cells"] == [cell, cell]
    with pytest.raises(FileExistsError, match="already exists"):
        producer.publish_formal_single_operator_profiler_subject_requirement(
            execution_source_path=(tmp_path / "profiler-source.json").resolve(),
            repository_root=tmp_path.resolve(),
            output_path=output,
        )


def test_derivation_rejects_ambiguous_selected_headline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell, observed = _producer_case(tmp_path, monkeypatch)
    predecessor = observed["predecessor"]
    predecessor.materialization.cells = (cell, cell)
    with pytest.raises(ValueError, match="one exact selected headline"):
        producer.derive_formal_single_operator_profiler_subject_requirement(
            execution_source_path=(tmp_path / "profiler-source.json").resolve(),
            repository_root=tmp_path.resolve(),
        )


def test_selected_run_is_deep_replayed_and_binds_exact_schedule_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _local_cell()
    actual = _actual(tmp_path, cell)
    materialization_binding = _json_binding(tmp_path, "local-materialization")
    config_binding = _json_binding(tmp_path, "headline-run-config")
    launch_binding = _json_binding(tmp_path, "headline-compile-launch")
    plan_binding = _json_binding(tmp_path, "headline-run-plan")
    schedule_binding = _json_binding(tmp_path, "headline-request-schedule")
    schedule_source_binding = _json_binding(tmp_path, "headline-schedule-source")
    cache_binding = _json_binding(tmp_path, "headline-compile-cache")
    factor_values = dict(cell.dimensions)
    config_value = config_binding.reopen()
    config = SimpleNamespace(
        adaptation=SimpleNamespace(stride=factor_values["update_stride"]),
        runtime=SimpleNamespace(
            telemetry_detail="headline",
            adaptation_microbatch_size=factor_values["microbatch"],
            adaptation_publication_coalescing=factor_values["coalescing"],
            adaptation_stream_priority=factor_values["stream_priority"],
        ),
        model_dump=lambda **_kwargs: config_value,
    )
    launch = SimpleNamespace(
        sha256=launch_binding.semantic_sha256,
        run_config_path=config_binding.absolute_path,
        compile_cache_plan_path=cache_binding.absolute_path,
        server_argv=("python", "-m", "fixture"),
        server_argv_sha256=_sha("argv"),
        localhost_port=31001,
        inventory_sha256=_sha("inventory"),
        gpu_uuids=("GPU-0",),
    )
    plan = SimpleNamespace(
        sha256=plan_binding.semantic_sha256,
        stage="E4",
        materialized_cell_id=cell.cell_id,
        execution_binding_sha256=_sha("execution-binding"),
        subject_sha256=_sha("execution-subject"),
        launch_manifest=launch_binding,
        request_schedule_receipt=schedule_binding,
        gpu_uuids=("GPU-0",),
        topology_mode="tp1_dp1",
    )
    schedule_value = schedule_binding.reopen()
    reopened = {"count": 0}

    def reopen_schedule() -> None:
        reopened["count"] += 1

    schedule = SimpleNamespace(
        sha256=schedule_binding.semantic_sha256,
        to_dict=lambda: schedule_value,
        reopen=reopen_schedule,
        materialized_cell_id=cell.cell_id,
        materialization=materialization_binding,
        compile_launch_manifest=launch_binding,
        execution_binding_sha256=plan.execution_binding_sha256,
        subject_sha256=plan.subject_sha256,
        topology_mode=plan.topology_mode,
        schedule_source=SimpleNamespace(
            path=schedule_source_binding.absolute_path,
            raw_sha256=schedule_source_binding.raw_sha256,
            semantic_sha256=schedule_source_binding.semantic_sha256,
            size=schedule_source_binding.size,
        ),
    )
    manifest = SimpleNamespace(
        sha256=actual.result_identity_sha256,
        completion_status="COMPLETE",
        stage="E4",
        cell_id=cell.cell_id,
        materialization_sha256=_sha("local-materialization"),
        materialization_protocol_lock_sha256=_sha("lock"),
        role=cell.method_role,
        backend=cell.backend,
        target_model_id=cell.model,
        run_directory=str(tmp_path.resolve()),
        artifacts=(
            SimpleNamespace(
                name="request_schedule",
                status="PRESENT",
                relative_path=Path(schedule_binding.absolute_path).name,
            ),
            SimpleNamespace(
                name="run_plan",
                status="PRESENT",
                relative_path=Path(plan_binding.absolute_path).name,
            ),
        ),
        run_plan_sha256=plan.sha256,
        execution_binding_sha256=plan.execution_binding_sha256,
        execution_subject_sha256=plan.subject_sha256,
        launch_manifest_sha256=launch.sha256,
        launch_argv=launch.server_argv,
        launch_argv_sha256=launch.server_argv_sha256,
        localhost_port=launch.localhost_port,
        inventory_sha256=launch.inventory_sha256,
        run_config_sha256=config_binding.semantic_sha256,
        run_config=config_value,
        request_schedule=schedule_value,
        request_schedule_sha256=schedule.sha256,
    )
    predecessor = RebuiltFormalSingleOperatorStageCompletion(
        artifact=SimpleNamespace(node="e4_local", actual_results=(actual,)),
        predecessor=None,
        node_materialization=SimpleNamespace(
            materialization_source=SimpleNamespace(
                absolute_path=materialization_binding.absolute_path
            )
        ),
        materialization=SimpleNamespace(
            sha256=manifest.materialization_sha256,
            protocol_lock_sha256=manifest.materialization_protocol_lock_sha256,
        ),
        decision=SimpleNamespace(payload={}),
    )
    validated_cache: list[object] = []
    monkeypatch.setattr(
        producer,
        "revalidate_formal_single_operator_run_manifest",
        lambda **_kwargs: manifest,
    )
    monkeypatch.setattr(
        producer.FormalServingRunPlan,
        "from_dict",
        lambda _value: plan,
    )
    monkeypatch.setattr(
        producer.CompileLaunchManifest,
        "load",
        lambda _path: launch,
    )
    monkeypatch.setattr(producer, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        producer,
        "run_config_sha256",
        lambda _config: config_binding.semantic_sha256,
    )
    monkeypatch.setattr(
        producer.CompileCacheLaunchPlan,
        "load",
        lambda _path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        producer,
        "validate_compile_key_for_run_config",
        lambda plan, **_kwargs: validated_cache.append(plan),
    )
    monkeypatch.setattr(
        producer.FormalServingRequestScheduleReceipt,
        "from_dict",
        lambda _value: schedule,
    )
    result = producer._revalidate_selected_run(
        repository_root=tmp_path.resolve(),
        predecessor=predecessor,
        cell=cell,
        actual=actual,
    )
    assert result == (
        config_binding,
        launch_binding,
        schedule_source_binding,
        schedule_binding,
    )
    assert reopened == {"count": 1}
    assert len(validated_cache) == 1
