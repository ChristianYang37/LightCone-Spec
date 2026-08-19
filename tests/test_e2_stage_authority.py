from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

import lightcone_spec.experiments.e2_stage_authority as e2_authority
from lightcone_spec import experiments
from lightcone_spec.config.schema import AdaptationConfig, OptimizerConfig
from lightcone_spec.experiments.e2_stage_authority import (
    E2CellExecutionEvidence,
    E2StagedCandidateEvaluation,
)
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.stage_materialization import (
    E2_OPTIMIZERS,
    E2_SCHEDULES,
    E2OptimizerNumericRecipe,
    default_e2_recipe_grid_authority,
    e1_geometries,
    e2_candidate_recipes,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _evaluations(recipes):
    rows = []
    for index, recipe in enumerate(recipes):
        rows.append(
            E2StagedCandidateEvaluation(
                recipe=recipe,
                cell_id=_sha(f"cell:{recipe.sha256}"),
                confidence_lower_request_rate_ratio=2.0 - index / 10_000_000,
                peak_hbm_bytes=1_000 + index,
                p99_itl_us=100 + index,
                exposed_update_us=10 + index,
                launched_updates=1,
                published_updates=1,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.recipe.sha256))


def test_e2_numeric_authority_builds_all_seven_configs_without_schema_defaults() -> (
    None
):
    grid = default_e2_recipe_grid_authority()
    authority = grid.optimizer_recipe_authority
    assert tuple(row.optimizer for row in authority.optimizer_recipes) == E2_OPTIMIZERS
    assert tuple(row.schedule for row in authority.schedule_recipes) == E2_SCHEDULES
    assert authority.schedule_recipe("cosine_to_zero").total_published_updates == 64
    assert {row.stride for row in authority.optimizer_recipes} == {10}
    assert {row.grad_clip for row in authority.optimizer_recipes} == {1.0}
    assert authority.optimizer_recipe("adam").weight_decay == 0.0
    assert authority.optimizer_recipe("adamw").decay_semantics == "decoupled"
    assert authority.optimizer_recipe("sgdm").decay_semantics == "coupled_l2"
    assert authority.optimizer_recipe("chronobelief").decay_semantics == "decoupled"
    for optimizer in E2_OPTIMIZERS:
        for schedule in E2_SCHEDULES:
            config = authority.optimizer_config(
                optimizer=optimizer,
                learning_rate=1e-6,
                schedule=schedule,
            )
            assert type(config) is OptimizerConfig
            assert set(config.model_fields_set) == set(OptimizerConfig.model_fields)
            assert config.schedule_total_published_updates == (
                64 if schedule == "cosine_to_zero" else None
            )


def test_e2_candidate_digest_binds_complete_optimizer_numeric_authority() -> None:
    grid = default_e2_recipe_grid_authority()
    candidate = e2_candidate_recipes((e1_geometries()[0],), grid=grid)[0]
    assert (
        candidate.optimizer_recipe_authority_sha256
        == grid.optimizer_recipe_authority.sha256
    )
    numeric = grid.optimizer_recipe_authority.optimizer_recipes[0]
    changed_numeric = replace(numeric, beta1=0.8)
    assert type(changed_numeric) is E2OptimizerNumericRecipe
    changed_authority = replace(
        grid.optimizer_recipe_authority,
        optimizer_recipes=(
            changed_numeric,
            *grid.optimizer_recipe_authority.optimizer_recipes[1:],
        ),
    )
    changed_grid = replace(grid, optimizer_recipe_authority=changed_authority)
    changed_candidate = next(
        row
        for row in e2_candidate_recipes((candidate.geometry,), grid=changed_grid)
        if row.optimizer == candidate.optimizer
        and row.schedule == candidate.schedule
        and row.learning_rate == candidate.learning_rate
    )
    assert changed_candidate.optimizer == candidate.optimizer
    assert changed_candidate.schedule == candidate.schedule
    assert changed_candidate.learning_rate == candidate.learning_rate
    assert changed_candidate.sha256 != candidate.sha256
    with pytest.raises(ValueError, match="numeric/grid authority"):
        changed_grid.optimizer_config_for(candidate)

    adam_candidate = next(
        row
        for row in e2_candidate_recipes((candidate.geometry,), grid=grid)
        if row.optimizer == "adam" and row.schedule == "constant"
    )
    adaptation = grid.adaptation_config_for(
        adam_candidate,
        canvas_tokens=16,
        adaptation_group_id=f"e2:{_sha('cell')}",
    )
    assert type(adaptation) is AdaptationConfig
    assert set(adaptation.model_fields_set) == set(AdaptationConfig.model_fields)
    assert set(adaptation.optimizer.model_fields_set) == set(
        OptimizerConfig.model_fields
    )


@pytest.mark.parametrize(
    ("geometry_count", "expected_round_counts"),
    ((1, (105, 27, 21, 21)), (32, (3_360, 840, 210, 53))),
)
def test_staged_e2_halving_keeps_exact_quarter_and_21_family_floor(
    geometry_count: int,
    expected_round_counts: tuple[int, int, int, int],
) -> None:
    source = e2_candidate_recipes(
        e1_geometries()[:geometry_count],
        grid=default_e2_recipe_grid_authority(),
    )
    observed = []
    for round_index in range(4):
        observed.append(len(source))
        survivors, final_recipe = e2_authority._select_survivor_recipes(
            source_recipes=source,
            evaluations=_evaluations(source),
            round_index=round_index,
        )
        if round_index == 3:
            assert len(survivors) == 1
            assert final_recipe == survivors[0]
            break
        assert final_recipe is None
        assert {(row.optimizer, row.schedule) for row in survivors} == {
            (optimizer, schedule)
            for optimizer in E2_OPTIMIZERS
            for schedule in E2_SCHEDULES
        }
        source = survivors
    assert tuple(observed) == expected_round_counts


def test_staged_e2_safe_elimination_uses_source_denominator_and_rejects_family_loss() -> (
    None
):
    source = e2_candidate_recipes(
        e1_geometries()[:1],
        grid=default_e2_recipe_grid_authority(),
    )
    evaluations = _evaluations(source)
    survivors, _ = e2_authority._select_survivor_recipes(
        source_recipes=source,
        evaluations=evaluations[:-1],
        round_index=0,
    )
    assert len(survivors) == 27

    missing_family = (E2_OPTIMIZERS[0], E2_SCHEDULES[0])
    with pytest.raises(ValueError, match="every optimizer/schedule family"):
        e2_authority._select_survivor_recipes(
            source_recipes=source,
            evaluations=tuple(
                row
                for row in evaluations
                if (row.recipe.optimizer, row.recipe.schedule) != missing_family
            ),
            round_index=0,
        )


def test_e2_evidence_is_path_bound_and_rejects_reused_artifact(tmp_path: Path) -> None:
    result_path = (tmp_path / "result.json").resolve()
    timing_path = (tmp_path / "timing.json").resolve()
    publish_canonical_json_no_replace(result_path, {"kind": "result"})
    publish_canonical_json_no_replace(timing_path, {"kind": "timing"})
    result = CanonicalJsonProofBinding.bind(result_path)
    timing = CanonicalJsonProofBinding.bind(timing_path)
    identity = StageItlExecutionIdentity(
        schema_version=1,
        kind="stage_itl_execution_identity",
        materialized_cell_id=_sha("cell"),
        inventory_sha256=_sha("inventory"),
        registry_sha256=_sha("registry"),
        execution_plan_sha256=_sha("plan"),
        rank_config_sha256=_sha("rank"),
        run_id="e2-run",
        run_nonce_sha256=_sha("nonce"),
        attempt_id="attempt-0",
        method="l0",
    )
    row = E2CellExecutionEvidence(
        schema_version=1,
        materialized_cell_id=identity.materialized_cell_id,
        execution_binding_sha256=_sha("binding"),
        execution_identity=identity,
        native_result_proof_path=result.absolute_path,
        native_result_proof_raw_sha256=result.raw_sha256,
        native_result_proof_semantic_sha256=result.semantic_sha256,
        stage_itl_proof_path=timing.absolute_path,
        stage_itl_proof_raw_sha256=timing.raw_sha256,
        stage_itl_proof_semantic_sha256=timing.semantic_sha256,
    )
    assert row.execution_identity == identity
    with pytest.raises(ValueError, match="must be distinct"):
        replace(
            row,
            stage_itl_proof_path=row.native_result_proof_path,
            stage_itl_proof_raw_sha256=row.native_result_proof_raw_sha256,
            stage_itl_proof_semantic_sha256=(row.native_result_proof_semantic_sha256),
        )


def test_package_exposes_only_staged_e2_selection_authority() -> None:
    assert experiments.E2StagedRoundSelectionReceipt is (
        e2_authority.E2StagedRoundSelectionReceipt
    )
    assert "E2RoundSelectionReceipt" not in experiments.__all__
    assert "E2StageReductionAuthority" not in experiments.__all__
    assert "bind_e2_stage_reduction_authority" not in experiments.__all__
