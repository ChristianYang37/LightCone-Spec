from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import (
    formal_single_operator_prerequisite_launch_producer as producer,
)
from lightcone_spec.experiments.stage_materialization import MaterializedCell


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
