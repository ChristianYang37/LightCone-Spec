from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.experiments.registry import (
    CONFIRMATION_METHOD_ROLES,
    E0_METHOD_ROLES,
    FROZEN_TTS_RECIPE_SENTINEL,
    LEGACY_DIAGNOSTIC_TTS_RECONSTRUCTION_AUTHORITY,
    REGISTERED_CONFIRMATION_BLOCKS,
    SEALED_E2_RECIPE_SENTINEL,
    CellStatus,
    ExperimentRegistry,
    ScientificMethodRole,
    build_legacy_industrial_registry,
    content_sha256,
    scientific_role_for_cell,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_legacy_industrial_registry(
        gpu_uuids=("GPU-scientific-a", "GPU-scientific-b"),
        cache_root="runtime-cache/scientific-identity-test",
        evidence_root="artifacts/scientific-identity-test",
    )


def test_tts_recipe_authority_is_manifest_bound_and_formally_blocked() -> None:
    path = ROOT / "manifests" / "provenance" / "tts_recipe_authority_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    sidecar = path.with_suffix(path.suffix + ".sha256").read_text().strip()

    authority = LEGACY_DIAGNOSTIC_TTS_RECONSTRUCTION_AUTHORITY
    assert payload == authority.to_dict()
    assert content_sha256(payload) == sidecar == authority.sha256
    assert authority.provenance_status == ("TTS-paper-reconstruction")
    assert authority.status == "BLOCKED"
    assert not authority.formal_eligible
    assert authority.historical_diagnostic_classification == (
        "matched_recipe_publication_policy_diagnostic_not_tts_reproduction"
    )


def test_tts_recipe_authority_is_declared_as_wheel_data() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["setuptools"]["data-files"][
        "share/lightcone-spec/manifests/provenance"
    ] == [
        "manifests/provenance/tts_recipe_authority_v1.json",
        "manifests/provenance/tts_recipe_authority_v1.json.sha256",
    ]


def test_registry_binds_authority_and_revised_stage_cardinalities(
    registry: ExperimentRegistry,
) -> None:
    payload = registry.to_dict()
    assert payload["legacy_frozen_tts_recipe_authority"] == (
        LEGACY_DIAGNOSTIC_TTS_RECONSTRUCTION_AUTHORITY.to_dict()
    )
    assert payload["legacy_frozen_tts_recipe_authority_sha256"] == (
        LEGACY_DIAGNOSTIC_TTS_RECONSTRUCTION_AUTHORITY.sha256
    )
    assert {
        stage: len(registry.cells_for(stage))
        for stage in (
            "E0",
            "E1",
            "E2",
            "E3b",
            "E5",
            "E6",
        )
    } == {
        "E0": 2144,
        "E1": 1428,
        "E2": 13456,
        "E3b": 11520,
        "E5": 11064,
        "E6": 2498,
    }


def test_e1_e2_search_only_l0_policy_candidates(
    registry: ExperimentRegistry,
) -> None:
    for experiment in ("E1", "E2"):
        cells = registry.cells_for(experiment)
        search = tuple(
            cell
            for cell in cells
            if cell.identity.method == "l0"
            and cell.identity.optimizer != FROZEN_TTS_RECIPE_SENTINEL
        )
        anchors = tuple(
            cell
            for cell in cells
            if cell.identity.optimizer == FROZEN_TTS_RECIPE_SENTINEL
        )
        assert search
        assert {cell.identity.method for cell in search} == {"l0"}
        assert {scientific_role_for_cell(registry, cell) for cell in search} == {
            ScientificMethodRole.LC_CANDIDATE.value
        }
        assert {cell.identity.method for cell in anchors} == {"tts", "l0"}
        assert {cell.status for cell in anchors} == {CellStatus.BLOCKED}
        assert {cell.reason_code for cell in anchors} == {
            "tts_official_recipe_unavailable"
        }
    for experiment, expected in (("E1", 42), ("E2", 8)):
        anchors = tuple(
            cell
            for cell in registry.cells_for(experiment)
            if cell.identity.optimizer == FROZEN_TTS_RECIPE_SENTINEL
        )
        assert len(anchors) == expected
        assert {cell.identity.rank for cell in anchors} == {None}
        assert {cell.identity.scope for cell in anchors} == {FROZEN_TTS_RECIPE_SENTINEL}
        assert {cell.identity.parameterization for cell in anchors} == {
            FROZEN_TTS_RECIPE_SENTINEL
        }
    assert {
        row.lookup_key.experiment for row in registry.adaptation_recipe_declarations
    } == {"E1", "E2", "frozen_tts"}
    assert {
        row.lookup_key.authority_kind for row in registry.adaptation_recipe_declarations
    } == {"frozen_tts", "lc_candidate"}


def test_tts_rejects_search_or_e2_selected_recipe_authority(
    registry: ExperimentRegistry,
) -> None:
    source = next(
        cell for cell in registry.cells_for("E2") if cell.identity.optimizer == "adamw"
    )
    forged = replace(source, identity=replace(source.identity, method="tts"))
    cells = tuple(
        forged if cell.cell_id == source.cell_id else cell for cell in registry.cells
    )
    with pytest.raises(ValueError, match="frozen TTS recipe identity is not exact"):
        replace(registry, cells=cells)

    tts = next(
        cell for cell in registry.cells_for("E1") if cell.identity.method == "tts"
    )
    mixed = replace(
        tts,
        identity=replace(
            tts.identity,
            scope="selected_e2",
            parameterization="lora",
            rank=8,
            alpha_over_rank=1.0,
            learning_rate=1e-4,
        ),
    )
    with pytest.raises(ValueError, match="frozen TTS recipe identity is not exact"):
        replace(
            registry,
            cells=tuple(mixed if cell == tts else cell for cell in registry.cells),
        )


@pytest.mark.parametrize("method", ("target_only", "static"))
def test_zero_adaptation_roles_reject_forged_scope(
    registry: ExperimentRegistry, method: str
) -> None:
    source = next(
        cell for cell in registry.cells_for("E1") if cell.identity.method == method
    )
    forged = replace(source, identity=replace(source.identity, scope="selected_e2"))
    with pytest.raises(ValueError, match="cannot carry adaptation recipe state"):
        replace(
            registry,
            cells=tuple(forged if cell == source else cell for cell in registry.cells),
        )


def test_frozen_tts_and_l0_naive_share_only_recipe_authority(
    registry: ExperimentRegistry,
) -> None:
    tts = next(
        cell for cell in registry.cells_for("E1") if cell.identity.method == "tts"
    )
    naive = next(
        cell
        for cell in registry.cells_for("E1")
        if cell.identity.method == "l0"
        and cell.identity.optimizer == FROZEN_TTS_RECIPE_SENTINEL
        and cell.identity.scope == tts.identity.scope
        and cell.identity.rank == tts.identity.rank
        and cell.identity.width == tts.identity.width
        and cell.identity.concurrency == tts.identity.concurrency
    )
    assert (
        tts.identity.optimizer
        == naive.identity.optimizer
        == (FROZEN_TTS_RECIPE_SENTINEL)
    )
    assert scientific_role_for_cell(registry, tts) == ScientificMethodRole.TTS.value
    assert scientific_role_for_cell(registry, naive) == (
        ScientificMethodRole.L0_NAIVE.value
    )
    assert tts.cell_id != naive.cell_id
    tts_recipe = registry.adaptation_recipe_for_cell(tts)
    assert tts_recipe is registry.adaptation_recipe_for_cell(naive)
    assert tts_recipe.lookup_key.authority_kind == "frozen_tts"
    assert tts_recipe.source_authority_sha256 == (
        LEGACY_DIAGNOSTIC_TTS_RECONSTRUCTION_AUTHORITY.sha256
    )
    assert tts_recipe.optimizer.name == "adam"
    assert tts_recipe.optimizer.schedule is None
    assert tts_recipe.status == "BLOCKED"
    with pytest.raises(ValueError, match="adaptation recipe is BLOCKED"):
        tts_recipe.to_adaptation_config()


def test_lightcone_role_requires_registry_owned_sealed_template(
    registry: ExperimentRegistry,
) -> None:
    lightcone = next(
        cell
        for cell in registry.cells_for("E3b")
        if cell.identity.optimizer == SEALED_E2_RECIPE_SENTINEL
    )
    assert scientific_role_for_cell(registry, lightcone) == (
        ScientificMethodRole.LIGHTCONE_TEMPLATE.value
    )
    assert lightcone.status is CellStatus.BLOCKED
    assert lightcone.reason_code == "sealed_e2_recipe_receipt_required"
    forged = replace(
        lightcone,
        status=CellStatus.UNMEASURED,
        reason_code="unmeasured",
        reason="forged unsealed LightCone",
    )
    with pytest.raises(ValueError, match="require an E2 seal"):
        replace(
            registry,
            cells=tuple(
                forged if cell.cell_id == lightcone.cell_id else cell
                for cell in registry.cells
            ),
        )
    with pytest.raises(ValueError, match="not registry-owned"):
        scientific_role_for_cell(
            registry,
            replace(lightcone, reason="caller-forged LightCone label"),
        )


def test_runtime_method_literal_cannot_forge_a_lightcone_role(
    registry: ExperimentRegistry,
) -> None:
    source = next(
        cell for cell in registry.cells if cell.identity.method == "target_only"
    )
    forged = replace(source, identity=replace(source.identity, method="lightcone"))
    with pytest.raises(ValueError, match="no registered scientific role"):
        replace(
            registry,
            cells=tuple(forged if cell == source else cell for cell in registry.cells),
        )
    with pytest.raises(ValueError, match="not registry-owned"):
        scientific_role_for_cell(registry, forged)


def test_confirmation_and_e6_roles_have_registered_repetition_and_load_axes(
    registry: ExperimentRegistry,
) -> None:
    assert tuple(
        axis.values
        for axis in registry.definition("E3b").axes
        if axis.name == "method_role"
    ) == (CONFIRMATION_METHOD_ROLES,)
    assert tuple(
        axis.values
        for axis in registry.definition("E0").axes
        if axis.name == "method_role"
    ) == (E0_METHOD_ROLES,)

    e6 = registry.cells_for("E6")
    assert all(
        cell.identity.optimizer != "transferred_e2"
        and cell.identity.schedule != "transferred_e2"
        for cell in e6
    )
    headline = tuple(
        cell for cell in e6 if "compatibility_transfer" in cell.identity.variant
    )
    assert {cell.identity.block for cell in headline} == set(
        REGISTERED_CONFIRMATION_BLOCKS
    )
    assert {cell.identity.arrival for cell in headline} == {
        "closed_loop_c1",
        "common_slo_load",
    }
    assert {scientific_role_for_cell(registry, cell) for cell in headline} == {
        ScientificMethodRole.TARGET_ONLY.value,
        ScientificMethodRole.STATIC.value,
        ScientificMethodRole.TTS.value,
        ScientificMethodRole.LIGHTCONE_TEMPLATE.value,
    }
    anchors = tuple(
        cell
        for cell in e6
        if "largest_feasible_model_anchor_template" in cell.identity.variant
    )
    assert len(anchors) == 2 * 2 * 2 * len(REGISTERED_CONFIRMATION_BLOCKS)
    assert {cell.identity.task for cell in anchors} == {"LiveCodeBench"}
    assert {cell.identity.context for cell in anchors} == {16384, 32768}
    assert {scientific_role_for_cell(registry, cell) for cell in anchors} == {
        ScientificMethodRole.L0_NAIVE.value
    }


def test_target_only_and_static_allocate_no_adaptation_identity(
    registry: ExperimentRegistry,
) -> None:
    baselines = tuple(
        cell
        for cell in registry.cells
        if cell.identity.method in {"target_only", "static"}
    )
    assert baselines
    assert all(
        cell.identity.optimizer is None
        and cell.identity.learning_rate is None
        and cell.identity.schedule is None
        and cell.identity.rank is None
        and cell.identity.alpha_over_rank is None
        and cell.identity.parameterization == "none"
        for cell in baselines
    )
