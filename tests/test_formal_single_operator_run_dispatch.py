from __future__ import annotations

import pytest

from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
    route_formal_single_operator_auxiliary,
    route_formal_single_operator_materialized_cell,
)
from lightcone_spec.experiments.formal_stage_execution import _method_for_cell
from lightcone_spec.experiments.stage_materialization import (
    E5_FAILURES,
    MaterializedCell,
)


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _cell(
    *,
    stage: str,
    task: str,
    role: str = "LightCone",
    backend: str = "DFLASH",
    dimensions: tuple[tuple[str, str | int | float], ...] = (),
) -> MaterializedCell:
    return MaterializedCell(
        stage=stage,
        method_role=role,
        model="Qwen/Qwen3-8B",
        backend=backend,
        task=task,
        publication_policy=(
            "none" if role in {"Target-only", "Static"} else "first_ready"
        ),
        recipe_sha256=(
            None if role in {"Target-only", "Static"} else _sha(f"recipe:{role}")
        ),
        dimensions=dimensions,
    )


@pytest.mark.parametrize(
    ("node", "phase", "cell", "expected"),
    (
        (
            "e4_profiler",
            "profiler",
            _cell(stage="E4", task="mechanism_profile_only"),
            "profiler",
        ),
        (
            "e3b_pilot",
            "excluded_pilot",
            _cell(stage="E3b", task="long_context_confirmation"),
            "serving",
        ),
        (
            "e1a",
            "verification",
            _cell(stage="E1a", task="LiveCodeBench_tuning_disjoint_from_E5"),
            "serving",
        ),
        (
            "e5_final",
            "final",
            _cell(stage="E5", task="production_slo_power_prefix"),
            "serving",
        ),
        (
            "e5_final",
            "final",
            _cell(stage="E5", task="deterministic_failure_injection"),
            "e5_failure",
        ),
        (
            "e6_pilot",
            "excluded_pilot",
            _cell(
                stage="E6",
                task="immutable_metadata_interface_and_fit_preflight",
                role="Target-only",
                backend="NEXTN",
            ),
            "e6_interface_preflight",
        ),
        (
            "e6_final",
            "final",
            _cell(stage="E6", task="MATH-500", backend="NEXTN"),
            "serving",
        ),
        (
            "e0_tuning",
            "tuning",
            _cell(
                stage="E0",
                task="compatibility_decision",
                role="Compatibility",
            ),
            "e0_compatibility_decision",
        ),
        (
            "e0_tuning",
            "tuning",
            _cell(
                stage="E0",
                task="independent_onlinespec_tuning",
                role="OnlineSPEC-OGD-candidate",
            ),
            "serving",
        ),
    ),
)
def test_downstream_cell_routes_are_closed_and_task_specific(
    node: str,
    phase: str,
    cell: MaterializedCell,
    expected: str,
) -> None:
    route = route_formal_single_operator_materialized_cell(
        node=node,  # type: ignore[arg-type]
        phase=phase,
        cell=cell,
    )
    assert route.physical_kind == expected
    assert route.materialized_cell_id == cell.cell_id


def test_e0_onlinespec_display_roles_reuse_the_registered_runtime_methods() -> None:
    expected = {
        "OnlineSPEC-OGD": "onlinespec_ogd",
        "OnlineSPEC-OPT": "onlinespec_opt",
        "OnlineSPEC-ENS": "onlinespec_ens",
        "OnlineSPEC-Optimistic-OGD": "onlinespec_opt",
        "OnlineSPEC-Hedge": "onlinespec_ens",
        "OnlineSPEC-OGD-candidate": "onlinespec_ogd",
        "OnlineSPEC-OPT-candidate": "onlinespec_opt",
        "OnlineSPEC-ENS-candidate": "onlinespec_ens",
        "OnlineSPEC-Optimistic-OGD-candidate": "onlinespec_opt",
        "OnlineSPEC-Hedge-candidate": "onlinespec_ens",
    }
    for role, method in expected.items():
        cell = _cell(stage="E0", task="task-native", role=role)
        route = route_formal_single_operator_materialized_cell(
            node="e0_final",
            phase="final",
            cell=cell,
        )
        assert route.physical_kind == "serving"
        assert _method_for_cell(cell) == method


def test_route_rejects_cross_stage_and_wrong_specialized_tasks() -> None:
    with pytest.raises(ValueError, match="node/stage/phase"):
        route_formal_single_operator_materialized_cell(
            node="e6_final",
            phase="final",
            cell=_cell(stage="E5", task="production_slo_power_prefix"),
        )
    with pytest.raises(ValueError, match="E5 cell"):
        route_formal_single_operator_materialized_cell(
            node="e5_final",
            phase="final",
            cell=_cell(stage="E5", task="foreign_failure_like_task"),
        )
    with pytest.raises(ValueError, match="E6 cell"):
        route_formal_single_operator_materialized_cell(
            node="e6_final",
            phase="final",
            cell=_cell(stage="E6", task="Qwen3.5-35B-A3B", backend="NEXTN"),
        )


@pytest.mark.parametrize("scenario", E5_FAILURES)
def test_every_registered_failure_kind_routes_to_the_one_shot_runner(
    scenario: str,
) -> None:
    cell = _cell(
        stage="E5",
        task="deterministic_failure_injection",
        dimensions=(("failure", scenario),),
    )
    route = route_formal_single_operator_materialized_cell(
        node="e5_final",
        phase="final",
        cell=cell,
    )
    assert route.physical_kind == "e5_failure"
    assert route.expected_terminal_kind == (
        "formal_single_operator_e5_physical_outcome"
    )


def test_compatibility_source_is_auxiliary_and_decisions_are_non_gpu_cells() -> None:
    assert route_formal_single_operator_auxiliary("e0_compatibility") == (
        "e0_compatibility"
    )
    with pytest.raises(ValueError, match="unsupported"):
        route_formal_single_operator_auxiliary("future_compatibility")
