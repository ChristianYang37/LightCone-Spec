from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import (
    formal_single_operator_prerequisite_launch_producer as producer,
)
from lightcone_spec.experiments.stage_materialization import MaterializedCell


def _sha(character: str) -> str:
    return character * 64


def _cell(
    *,
    stage: str,
    model: str,
    backend: str,
    task: str,
    role: str = "Static",
    dimensions: dict[str, str | int] | None = None,
) -> MaterializedCell:
    return MaterializedCell(
        stage=stage,
        method_role=role,
        model=model,
        backend=backend,
        task=task,
        publication_policy="none",
        recipe_sha256=None,
        dimensions=tuple(sorted((dimensions or {}).items())),
    )


def _demands(
    *, node: str, stage: str, phase: str, cells: tuple[MaterializedCell, ...]
) -> tuple[producer.PrerequisiteLaunchDemand, ...]:
    return producer.materialized_prerequisite_launch_demands(
        source=SimpleNamespace(node=node, stage=stage, phase=phase),
        materialization=SimpleNamespace(cells=cells),
    )


def test_public_producer_is_path_only_and_accepts_no_scientific_knobs() -> None:
    parameters = inspect.signature(
        producer.publish_formal_single_operator_prerequisite_launch_index
    ).parameters

    assert tuple(parameters) == (
        "execution_source_path",
        "base_environment_launch_manifest_path",
        "repository_root",
        "private_output_root",
    )
    assert not {
        "model",
        "backend",
        "topology",
        "run_config",
        "learning_rate",
        "stride",
        "width",
        "load",
    }.intersection(parameters)


def test_bootstrap_discovers_qualification_launches_from_completion_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = SimpleNamespace(predecessor=None)
    latest = SimpleNamespace(predecessor=first)
    expected = {
        suite: SimpleNamespace(
            binding=SimpleNamespace(absolute_path=f"/proof/{suite}.json")
        )
        for suite in (
            "dspark_dp2",
            "dspark_tp1",
            "dspark_tp2",
            "tp1_dp2",
            "tp2_dp1",
        )
    }
    monkeypatch.setattr(
        producer,
        "rebuild_formal_single_operator_stage_completion",
        lambda path: latest,
    )
    monkeypatch.setattr(
        producer,
        "_preflight_qualification_authorities",
        lambda chain: expected if chain == (latest, first) else {},
    )

    observed = producer.trusted_preflight_qualification_launch_paths_from_completion(
        "/run/current-completion.json"
    )

    assert observed == {suite: f"/proof/{suite}.json" for suite in sorted(expected)}


def test_e5_demands_cover_tp1_tp2_dp2_for_both_backends() -> None:
    cells = tuple(
        _cell(
            stage="E5",
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="production_slo_power_prefix",
            dimensions={"backend_authority": backend, "topology": topology},
        )
        for backend in ("DFLASH", "DSPARK")
        for topology in ("tp1_dp1", "tp2_dp1", "tp1_dp2")
    )

    observed = _demands(node="e5_final", stage="E5", phase="final", cells=cells)

    assert {(row.backend, row.topology_mode) for row in observed} == {
        (backend, topology)
        for backend in ("DFLASH", "DSPARK")
        for topology in ("tp1_dp1", "tp2_dp1", "tp1_dp2")
    }


def test_e4_profiler_has_one_exact_selected_launch_demand() -> None:
    observed = _demands(
        node="e4_profiler",
        stage="E4",
        phase="profiler",
        cells=tuple(
            _cell(
                stage="E4",
                model="Qwen/Qwen3-8B",
                backend="DFLASH",
                task="mechanism_profile_only",
                role="LightCone",
                dimensions={
                    "profiler": profiler,
                    "selected_configuration_sha256": "a" * 64,
                },
            )
            for profiler in ("nvtx", "nsight_systems", "nsight_compute")
        ),
    )

    assert observed == (
        producer.PrerequisiteLaunchDemand(
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            topology_mode="tp1_dp1",
        ),
    )


def test_e6_preflights_and_e0_compatibility_rows_do_not_create_launch_demand() -> None:
    e6 = _demands(
        node="e6_pilot",
        stage="E6",
        phase="excluded_pilot",
        cells=(
            _cell(
                stage="E6",
                model="Qwen/Qwen3.6-35B-A3B",
                backend="NEXTN",
                task="immutable_metadata_interface_and_fit_preflight",
                role="Target-only",
                dimensions={"topology": "tp2_dp1"},
            ),
            _cell(
                stage="E6",
                model="Qwen/Qwen3.6-35B-A3B",
                backend="NEXTN",
                task="LiveCodeBench",
                dimensions={"topology": "tp2_dp1"},
            ),
            _cell(
                stage="E6",
                model="Qwen/Qwen3.5-122B-A10B-FP8",
                backend="NEXTN",
                task="MATH-500",
                dimensions={"topology": "tp2_dp1"},
            ),
        ),
    )
    assert {(row.model, row.backend, row.topology_mode) for row in e6} == {
        ("Qwen/Qwen3.6-35B-A3B", "NEXTN", "tp2_dp1"),
        ("Qwen/Qwen3.5-122B-A10B-FP8", "NEXTN", "tp2_dp1"),
    }

    e0 = _demands(
        node="e0_tuning",
        stage="E0",
        phase="tuning",
        cells=(
            _cell(
                stage="E0",
                model="Qwen/Qwen3-4B",
                backend="DFLASH",
                task="compatibility_decision",
                role="Compatibility",
            ),
            _cell(
                stage="E0",
                model="Qwen/Qwen3-4B",
                backend="EAGLE3",
                task="GSM8K",
                role="OnlineSPEC-OGD-candidate",
            ),
        ),
    )
    assert e0 == (
        producer.PrerequisiteLaunchDemand(
            model="Qwen/Qwen3-4B",
            backend="EAGLE3",
            topology_mode="tp1_dp1",
        ),
    )

    with pytest.raises(ValueError, match="no launch demand"):
        _demands(
            node="e0_tuning",
            stage="E0",
            phase="tuning",
            cells=(
                _cell(
                    stage="E0",
                    model="Qwen/Qwen3-4B",
                    backend="DFLASH",
                    task="compatibility_decision",
                    role="Compatibility",
                ),
            ),
        )


def _execution_source_stub(
    *,
    node: str,
    phase: str,
    materialization: object,
    materialization_source: object,
    predecessor: object | None = None,
) -> object:
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FormalSingleOperatorExecutionSource,
    )

    source = object.__new__(FormalSingleOperatorExecutionSource)
    values = {
        "node": node,
        "stage": "E0",
        "phase": phase,
        "materialization_source": materialization_source,
        "materialization_sha256": materialization.sha256,
        "materialization_source_decision_sha256": (
            materialization.source_decision_sha256
        ),
        "materialization_upstream_receipt_sha256s": (
            materialization.upstream_receipt_sha256s
        ),
        "predecessor_completion_source": (
            None
            if predecessor is None
            else SimpleNamespace(absolute_path=f"/{node}-predecessor.json")
        ),
        "predecessor_completion_sha256": (
            None if predecessor is None else predecessor.artifact.sha256
        ),
        "predecessor_decision_sha256": (
            None if predecessor is None else predecessor.decision.sha256
        ),
    }
    for name, value in values.items():
        object.__setattr__(source, name, value)
    return source


def _bind_materialization(tmp_path: Path, name: str, materialization: object) -> object:
    from lightcone_spec.experiments.formal_registry import (
        stage_materialization_receipt_to_dict,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        publish_formal_single_operator_json_artifact,
    )

    return publish_formal_single_operator_json_artifact(
        tmp_path / f"{name}.json",
        stage_materialization_receipt_to_dict(materialization),
    )


def _all_na_compatibility() -> object:
    from lightcone_spec.experiments.stage_materialization import (
        E0_BACKENDS,
        E0_MODELS,
        E0_TASKS,
        E0CompatibilityDecision,
        E0CompatibilityReceipt,
    )

    decisions = tuple(
        sorted(
            (
                E0CompatibilityDecision(
                    model=model,
                    backend=backend,
                    task=task,
                    disposition="N/A",
                    reason_code="unsupported_exact_interface",
                    interface_sha256=_sha("1"),
                    task_native_workload_sha256=_sha("2"),
                )
                for model in E0_MODELS
                for backend in E0_BACKENDS
                for task in E0_TASKS
            ),
            key=lambda row: row.decision_id,
        )
    )
    return E0CompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("3"),
        upstream_e6_receipt_sha256=_sha("4"),
        decisions=decisions,
    )


def test_exact_e0_all_na_chain_is_the_only_legal_empty_launch_demand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments.formal_registry import (
        e0_compatibility_receipt_to_dict,
    )
    from lightcone_spec.experiments.stage_materialization import (
        GpuHourEstimate,
        StageMaterializationReceipt,
    )

    compatibility = _all_na_compatibility()
    bundle_sha256 = _sha("5")
    cells = tuple(
        sorted(
            (
                MaterializedCell(
                    stage="E0",
                    method_role="Compatibility",
                    model=decision.model,
                    backend=decision.backend,
                    task="compatibility_decision",
                    publication_policy="decision_only",
                    recipe_sha256=None,
                    dimensions=tuple(
                        sorted(
                            {
                                "compatibility_decision_id": decision.decision_id,
                                "deployment_task": decision.task,
                                "disposition": "N/A",
                                "reason_code": decision.reason_code,
                                "interface_sha256": decision.interface_sha256,
                                "task_native_workload_sha256": (
                                    decision.task_native_workload_sha256
                                ),
                                "compatibility_receipt_sha256": compatibility.sha256,
                                "compatibility_evidence_manifest_sha256": _sha("6"),
                                "e0_compatibility_bundle_sha256": bundle_sha256,
                            }.items()
                        )
                    ),
                )
                for decision in compatibility.decisions
            ),
            key=lambda cell: cell.cell_id,
        )
    )
    tuning = StageMaterializationReceipt(
        schema_version=1,
        stage="E0",
        protocol_lock_sha256=_sha("3"),
        upstream_receipt_sha256s=(_sha("7"),),
        source_decision_sha256=bundle_sha256,
        materialization_rule="108_compatibility_decisions_plus_239_rows_per_valid",
        expected_cell_count=108,
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    tuning_source = _execution_source_stub(
        node="e0_tuning",
        phase="tuning",
        materialization=tuning,
        materialization_source=_bind_materialization(tmp_path, "tuning", tuning),
    )
    assert producer.execution_source_prerequisite_launch_demands(tuning_source) == ()

    compatibility_payload = e0_compatibility_receipt_to_dict(compatibility)
    tuning_completion = SimpleNamespace(
        artifact=SimpleNamespace(node="e0_tuning", sha256=_sha("8")),
        materialization=SimpleNamespace(sha256=tuning.sha256),
        decision=SimpleNamespace(
            sha256=_sha("9"),
            decision_kind="e0_tuning_actual_reduced",
            next_materialization_source_decision_sha256=_sha("a"),
            next_materialization_upstream_receipt_sha256s=(tuning.sha256,),
            payload={
                "status": "ALL_NA",
                "valid_count": 0,
                "compatibility": compatibility_payload,
            },
        ),
    )
    pilot = StageMaterializationReceipt(
        schema_version=1,
        stage="E0",
        protocol_lock_sha256=_sha("3"),
        upstream_receipt_sha256s=(tuning.sha256,),
        source_decision_sha256=_sha("a"),
        materialization_rule=("valid_x_8_roles_x_2_loads_x_4_excluded_pilots"),
        expected_cell_count=0,
        cells=(),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    pilot_source = _execution_source_stub(
        node="e0_pilot",
        phase="excluded_pilot",
        materialization=pilot,
        materialization_source=_bind_materialization(tmp_path, "pilot", pilot),
        predecessor=tuning_completion,
    )
    pilot_completion = SimpleNamespace(
        artifact=SimpleNamespace(node="e0_pilot", sha256=_sha("b")),
        materialization=SimpleNamespace(sha256=pilot.sha256),
        decision=SimpleNamespace(
            sha256=_sha("c"),
            decision_kind="e0_pilot_all_na",
            next_materialization_source_decision_sha256=_sha("d"),
            next_materialization_upstream_receipt_sha256s=(pilot.sha256,),
            payload={
                "status": "ALL_NA",
                "selected_final_blocks": 0,
                "compatibility": compatibility_payload,
            },
        ),
    )
    final = StageMaterializationReceipt(
        schema_version=1,
        stage="E0",
        protocol_lock_sha256=_sha("3"),
        upstream_receipt_sha256s=(pilot.sha256,),
        source_decision_sha256=_sha("d"),
        materialization_rule="valid_x_8_roles_x_2_loads_x_powered_final_blocks",
        expected_cell_count=0,
        cells=(),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    final_source = _execution_source_stub(
        node="e0_final",
        phase="final",
        materialization=final,
        materialization_source=_bind_materialization(tmp_path, "final", final),
        predecessor=pilot_completion,
    )
    completions = {
        "/e0_pilot-predecessor.json": tuning_completion,
        "/e0_final-predecessor.json": pilot_completion,
    }
    monkeypatch.setattr(
        producer,
        "rebuild_formal_single_operator_stage_completion",
        completions.__getitem__,
    )
    assert producer.execution_source_prerequisite_launch_demands(pilot_source) == ()
    assert producer.execution_source_prerequisite_launch_demands(final_source) == ()

    pseudo_empty = SimpleNamespace(
        node="e5_final",
        stage="E5",
        phase="final",
    )
    with pytest.raises(ValueError, match="no launch demand"):
        producer.materialized_prerequisite_launch_demands(
            source=pseudo_empty,
            materialization=SimpleNamespace(cells=()),
        )

    # A caller cannot turn one VALID decision into an auxiliary-only plan by
    # omitting its 239 required serving rows.
    invalid_dimensions = dict(cells[0].dimensions)
    invalid_dimensions["disposition"] = "VALID"
    forged = StageMaterializationReceipt(
        schema_version=1,
        stage="E0",
        protocol_lock_sha256=tuning.protocol_lock_sha256,
        upstream_receipt_sha256s=tuning.upstream_receipt_sha256s,
        source_decision_sha256=tuning.source_decision_sha256,
        materialization_rule=tuning.materialization_rule,
        expected_cell_count=108,
        cells=tuple(
            sorted(
                (
                    MaterializedCell(
                        stage=cells[0].stage,
                        method_role=cells[0].method_role,
                        model=cells[0].model,
                        backend=cells[0].backend,
                        task=cells[0].task,
                        publication_policy=cells[0].publication_policy,
                        recipe_sha256=None,
                        dimensions=tuple(sorted(invalid_dimensions.items())),
                    ),
                    *cells[1:],
                ),
                key=lambda cell: cell.cell_id,
            )
        ),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    forged_source = _execution_source_stub(
        node="e0_tuning",
        phase="tuning",
        materialization=forged,
        materialization_source=_bind_materialization(tmp_path, "forged", forged),
    )
    with pytest.raises(ValueError, match="no launch demand"):
        producer.execution_source_prerequisite_launch_demands(forged_source)
