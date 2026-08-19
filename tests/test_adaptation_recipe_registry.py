from __future__ import annotations

from dataclasses import replace

import pytest

from lightcone_spec.config.schema import AdaptationConfig, OptimizerConfig
from lightcone_spec.experiments.planning import E2CandidateIdentity
from lightcone_spec.experiments.registry import (
    E2_DRAFT_WIDTH_SELECTOR,
    E2_HALVING_STAGES,
    FROZEN_TTS_RECIPE_SENTINEL,
    AdaptationRecipeBlocker,
    CellStatus,
    ExperimentRegistry,
    content_sha256,
)
from lightcone_spec.experiments.registry import (
    build_legacy_industrial_registry as build_industrial_registry,
)


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_industrial_registry(
        gpu_uuids=("GPU-recipe-a", "GPU-recipe-b"),
        cache_root="runtime-cache/recipe-test",
        evidence_root="artifacts/recipe-test",
    )


def _adaptive_cells(registry: ExperimentRegistry, experiment: str):
    return tuple(
        cell
        for cell in registry.cells_for(experiment)
        if cell.identity.method == "l0"
        and cell.identity.optimizer != FROZEN_TTS_RECIPE_SENTINEL
    )


def test_e1_recipe_lookup_is_lc_candidate_owned_and_all_fields_explicit(
    registry: ExperimentRegistry,
) -> None:
    cells = _adaptive_cells(registry, "E1")
    assert cells
    assert {registry.adaptation_recipe_for_cell(cell).status for cell in cells} == {
        "AVAILABLE"
    }

    source = next(
        cell
        for cell in cells
        if cell.identity.scope == "last3"
        and cell.identity.parameterization == "lora"
        and cell.identity.rank == 8
        and cell.identity.optimizer == "adamw"
        and cell.identity.width == 16
    )
    peer = next(
        cell
        for cell in cells
        if cell.identity.scope == source.identity.scope
        and cell.identity.parameterization == source.identity.parameterization
        and cell.identity.rank == source.identity.rank
        and cell.identity.optimizer == source.identity.optimizer
        and cell.identity.width == source.identity.width
        and cell.identity.concurrency != source.identity.concurrency
    )
    declaration = registry.adaptation_recipe_for_cell(source.cell_id)
    assert declaration is registry.adaptation_recipe_for_cell(peer)
    assert declaration.optimizer.learning_rate == 1e-4
    assert declaration.optimizer.weight_decay == 0.01
    assert declaration.stride == 10
    assert declaration.canvas_tokens == 16

    config = declaration.to_adaptation_config()
    assert set(config.model_fields_set) == set(AdaptationConfig.model_fields)
    assert set(config.optimizer.model_fields_set) == set(OptimizerConfig.model_fields)
    assert config.rank == config.lora_alpha == 8
    assert config.optimizer.epsilon == 1e-8


def test_e1_full_and_momentum_anchor_values_come_from_registered_grid(
    registry: ExperimentRegistry,
) -> None:
    cells = _adaptive_cells(registry, "E1")
    full_sgdm = next(
        registry.adaptation_recipe_for_cell(cell)
        for cell in cells
        if cell.identity.parameterization == "full"
        and cell.identity.scope == "last1"
        and cell.identity.optimizer == "sgdm"
    )
    lora_sgdm = next(
        registry.adaptation_recipe_for_cell(cell)
        for cell in cells
        if cell.identity.parameterization == "lora"
        and cell.identity.scope == "last1"
        and cell.identity.rank == 1
        and cell.identity.optimizer == "sgdm"
    )
    assert full_sgdm.optimizer.learning_rate == 1e-4
    assert lora_sgdm.optimizer.learning_rate == 1e-3
    assert full_sgdm.optimizer.momentum == lora_sgdm.optimizer.momentum == 0.9


def test_e1_source_authority_binds_explicit_canonical_epsilon(
    registry: ExperimentRegistry,
) -> None:
    declaration = next(
        row
        for row in registry.adaptation_recipe_declarations
        if row.lookup_key.experiment == "E1"
        and row.lookup_key.scope == "last1"
        and row.lookup_key.parameterization == "full"
        and row.lookup_key.optimizer == "adamw"
    )
    from lightcone_spec.experiments.protocol import (
        DFLASH_LOSS_POSITION_DECAY,
        tuning_candidates,
    )

    candidate = next(
        row
        for row in tuning_candidates()
        if (
            row.parameter_scope,
            row.weight_update_mode,
            row.rank,
            row.optimizer,
        )
        == ("last1", "full", None, "adamw")
    )
    assert declaration.source_authority_sha256 == content_sha256(
        {
            "candidate_id": candidate.candidate_id,
            "fixed_semantics": {
                "kv_history_policy": "frozen",
                "adaptation_scope": "cohort",
                "lora_matrix_policy": "registered_matrices_v1",
                "native_head_policy": "frozen",
                "max_in_flight": 1,
                "loss_position_decay": DFLASH_LOSS_POSITION_DECAY,
                "extra_logical_delay": 0,
                "teacher_row_policy": "update_round",
                "verification_mode": "native_scheduler",
                "fixed_verification_budget": None,
                "confidence_loss_weight": None,
                "optimizer_epsilon": 1e-8,
            },
        }
    )


def test_e2_selected_width_is_dependency_owned_and_source_values_stay_blocked(
    registry: ExperimentRegistry,
) -> None:
    cells = _adaptive_cells(registry, "E2")
    assert cells
    assert {cell.identity.width for cell in cells} == {None}
    assert {cell.status for cell in cells} == {CellStatus.BLOCKED}
    declarations = tuple(
        row
        for row in registry.adaptation_recipe_declarations
        if row.lookup_key.experiment == "E2"
    )
    assert len(declarations) * len(E2_HALVING_STAGES) == len(cells)
    assert {row.status for row in declarations} == {"BLOCKED"}
    assert {row.lookup_key.draft_width for row in declarations} == {None}
    assert {row.lookup_key.draft_width_selector for row in declarations} == {
        E2_DRAFT_WIDTH_SELECTOR
    }
    assert all(row.canvas_tokens is None for row in declarations)
    assert all("canvas_tokens" not in row.unresolved_fields for row in declarations)
    assert all("extra_logical_delay" in row.unresolved_fields for row in declarations)
    assert all("stride" in row.unresolved_fields for row in declarations)
    assert all(
        "learning_rate" in row.optimizer.unresolved_fields for row in declarations
    )
    assert all(row.optimizer.learning_rate is None for row in declarations)
    assert all("teacher_row_policy" in row.unresolved_fields for row in declarations)
    assert all("verification_mode" in row.unresolved_fields for row in declarations)
    assert all(
        "fixed_verification_budget" in row.unresolved_fields for row in declarations
    )
    assert all(
        {
            "e2_weight_decay_unregistered",
            "e2_beta1_unregistered",
            "e2_beta2_unregistered",
            "e2_epsilon_unregistered",
            "e2_grad_clip_unregistered",
        }
        <= set(row.blocker_codes)
        for row in declarations
    )
    assert all(
        "e2_draft_width_selector_unresolved" not in row.blocker_codes
        for row in declarations
    )
    assert all(row.teacher_row_policy is None for row in declarations)
    assert all(row.verification_mode is None for row in declarations)
    assert all(row.fixed_verification_budget is None for row in declarations)
    with pytest.raises(ValueError, match="adaptation recipe is BLOCKED"):
        declarations[0].to_adaptation_config()


def test_e2_missing_optimizer_semantics_have_named_blockers(
    registry: ExperimentRegistry,
) -> None:
    declarations = tuple(
        row
        for row in registry.adaptation_recipe_declarations
        if row.lookup_key.experiment == "E2"
    )

    def one(optimizer: str, schedule: str = "constant"):
        return next(
            row
            for row in declarations
            if row.lookup_key.optimizer == optimizer
            and row.lookup_key.schedule == schedule
        )

    adamw = one("adamw")
    assert {
        "e2_weight_decay_unregistered",
        "e2_beta1_unregistered",
        "e2_beta2_unregistered",
        "e2_epsilon_unregistered",
        "e2_grad_clip_unregistered",
        "e2_learning_rate_unregistered",
        "e2_update_stride_unregistered",
        "e2_extra_logical_delay_unregistered",
    } <= set(adamw.blocker_codes)
    assert {
        ("optimizer.weight_decay", "e2_weight_decay_unregistered"),
        ("optimizer.beta1", "e2_beta1_unregistered"),
        ("optimizer.beta2", "e2_beta2_unregistered"),
        ("optimizer.epsilon", "e2_epsilon_unregistered"),
        ("optimizer.grad_clip", "e2_grad_clip_unregistered"),
        ("optimizer.learning_rate", "e2_learning_rate_unregistered"),
        ("stride", "e2_update_stride_unregistered"),
        ("extra_logical_delay", "e2_extra_logical_delay_unregistered"),
        ("teacher_row_policy", "e2_teacher_row_policy_unregistered"),
        ("verification_mode", "e2_verification_mode_unregistered"),
        (
            "fixed_verification_budget",
            "e2_fixed_verification_budget_unregistered",
        ),
    } <= {(row.field, row.reason_code) for row in adamw.blocker_matrix}
    assert "schedule_total_published_updates" not in (adamw.optimizer.unresolved_fields)
    assert adamw.optimizer.learning_rate is None

    sgdm = one("sgdm")
    assert "e2_momentum_unregistered" in sgdm.blocker_codes
    assert "momentum" in sgdm.optimizer.unresolved_fields

    muon = one("muon")
    assert {
        "e2_muon_ns_steps_unregistered",
        "e2_muon_auxiliary_learning_rate_unregistered",
        "e2_muon_auxiliary_weight_decay_unregistered",
    } <= set(muon.blocker_codes)
    assert {
        "muon_ns_steps",
        "muon_auxiliary_learning_rate",
        "muon_auxiliary_weight_decay",
    } <= set(muon.optimizer.unresolved_fields)

    cosine = one("adamw", "cosine_to_zero")
    assert "e2_cosine_horizon_unregistered" in cosine.blocker_codes
    assert "schedule_total_published_updates" in cosine.optimizer.unresolved_fields

    chronobelief = one("chronobelief")
    assert "chronobelief_equation_unregistered" not in chronobelief.blocker_codes
    assert "e2_learning_rate_unregistered" in chronobelief.blocker_codes
    assert chronobelief.optimizer.learning_rate is None


def test_e2_candidate_identity_binds_selector_instead_of_middle_grid_value(
    registry: ExperimentRegistry,
) -> None:
    cell = next(
        cell
        for cell in _adaptive_cells(registry, "E2")
        if cell.identity.optimizer == "adamw"
    )
    candidate = E2CandidateIdentity.from_cell(cell, registry=registry)
    assert candidate.width is None
    assert candidate.draft_width_selector == E2_DRAFT_WIDTH_SELECTOR
    with pytest.raises(ValueError, match="exactly one width authority"):
        replace(candidate, width=8)


def test_registry_digest_binds_recipe_declarations_and_rejects_fixed_e2_width(
    registry: ExperimentRegistry,
) -> None:
    payload = registry.to_dict()
    assert payload["adaptation_recipe_declarations_sha256"] == content_sha256(
        registry.adaptation_recipe_declarations
    )
    assert len(payload["adaptation_recipe_declarations"]) == len(
        registry.adaptation_recipe_declarations
    )
    source = next(
        cell
        for cell in _adaptive_cells(registry, "E2")
        if cell.identity.optimizer == "adamw"
    )
    edited = replace(source, identity=replace(source.identity, width=8))
    cells = tuple(
        edited if cell.cell_id == source.cell_id else cell for cell in registry.cells
    )
    with pytest.raises(ValueError, match="width templates"):
        replace(registry, cells=cells)


def test_lookup_rejects_non_registry_owned_cell(registry: ExperimentRegistry) -> None:
    source = next(iter(_adaptive_cells(registry, "E1")))
    edited = replace(source, reason="caller-edited declaration source")
    with pytest.raises(ValueError, match="not registry-owned"):
        registry.adaptation_recipe_for_cell(edited)


def test_recipe_protocol_values_reject_bool_and_text_subclasses(
    registry: ExperimentRegistry,
) -> None:
    recipe = next(
        row
        for row in registry.adaptation_recipe_declarations
        if row.lookup_key.experiment == "E2" and row.lookup_key.optimizer == "adamw"
    )
    with pytest.raises(ValueError, match="schema 1"):
        replace(recipe, schema_version=True)
    with pytest.raises(TypeError, match="exact float"):
        replace(recipe.optimizer, beta1=True)

    class CallerText(str):
        pass

    with pytest.raises(ValueError, match="exact non-empty"):
        AdaptationRecipeBlocker(
            field=CallerText("optimizer.beta1"),
            reason_code="e2_beta1_unregistered",
        )
    with pytest.raises(ValueError, match="reason code"):
        AdaptationRecipeBlocker(
            field="optimizer.beta1",
            reason_code=CallerText("e2_beta1_unregistered"),
        )


def test_e2_blockers_exactly_cover_every_unresolved_field(
    registry: ExperimentRegistry,
) -> None:
    recipe = next(
        row
        for row in registry.adaptation_recipe_declarations
        if row.lookup_key.experiment == "E2" and row.lookup_key.optimizer == "adamw"
    )
    with pytest.raises(ValueError, match="unresolved field set"):
        replace(recipe, blocker_codes=("e2_learning_rate_unregistered",))
    with pytest.raises(ValueError, match="lacks a field mapping"):
        replace(recipe, blocker_codes=("caller_invented_blocker",))

    forged_optimizer = replace(
        recipe.optimizer,
        learning_rate=recipe.lookup_key.learning_rate,
        unresolved_fields=tuple(
            field
            for field in recipe.optimizer.unresolved_fields
            if field != "learning_rate"
        ),
    )
    with pytest.raises(ValueError, match="optimizer unresolved fields"):
        replace(
            recipe,
            optimizer=forged_optimizer,
            blocker_codes=tuple(
                code
                for code in recipe.blocker_codes
                if code != "e2_learning_rate_unregistered"
            ),
        )

    with pytest.raises(ValueError, match="adaptation unresolved fields"):
        replace(
            recipe,
            teacher_row_policy="update_round",
            unresolved_fields=tuple(
                field
                for field in recipe.unresolved_fields
                if field != "teacher_row_policy"
            ),
            blocker_codes=tuple(
                code
                for code in recipe.blocker_codes
                if code != "e2_teacher_row_policy_unregistered"
            ),
        )
