from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_trainable_plan_authority import _inputs

from lightcone_spec.adaptation.plan_authority import (
    TrainablePlanAuthorityBinding,
    trainable_plan_authority_binding_to_dict,
)
from lightcone_spec.experiments.formal_method_authority import (
    TTS_DRAFTER_NATIVE_LOSS_SOURCE,
    TTS_TUNING_WINDOW_SOURCE_KIND,
    build_source_chronobelief_authority_artifact,
    build_source_tts_calibration_authority_artifact,
    load_chronobelief_authority_artifact,
    load_tts_calibration_authority_artifact,
    publish_chronobelief_authority_artifact,
    publish_tts_calibration_authority_artifact,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_stage_execution import (
    E1RecipeAnchorAuthorityArtifact,
    build_source_e1_recipe_anchor_authority_artifact,
    load_e1_recipe_anchor_authority_artifact,
    publish_e1_recipe_anchor_authority_artifact,
)
from lightcone_spec.runtime.content_authorization import (
    TtsCalibrationTuningWindow,
    TtsCalibrationTuningWindowEntry,
)
from lightcone_spec.runtime.proof_artifact import publish_canonical_json_no_replace


def _plan_source(
    tmp_path: Path, *, scope: str
) -> tuple[Path, TrainablePlanAuthorityBinding]:
    values = _inputs(tmp_path / "plan-inputs", mode="full", scope=scope)
    binding = values["binding"]
    assert isinstance(binding, TrainablePlanAuthorityBinding)
    path = (tmp_path / "trainable-plan-authority.json").resolve()
    publish_canonical_json_no_replace(
        path,
        trainable_plan_authority_binding_to_dict(binding),
    )
    return path, binding


def _tts_window() -> TtsCalibrationTuningWindow:
    rows = tuple(
        TtsCalibrationTuningWindowEntry(
            workload_id="livecodebench_v6_hard",
            source_sample_id=f"sample-{index}",
            source_descriptor_sha256=content_sha256("workload-descriptor"),
            prompt_sha256=content_sha256(f"prompt-{index}"),
        )
        for index in range(6)
    )
    return TtsCalibrationTuningWindow(
        schema_version=2,
        kind=TTS_TUNING_WINDOW_SOURCE_KIND,
        tuning_entries=tuple(sorted(rows[:2], key=lambda row: row.entry_id)),
        excluded_pilot_entries=tuple(sorted(rows[2:], key=lambda row: row.entry_id)),
    )


def test_e1_anchor_authority_deep_reopens_plan_and_exact_two_recipes(
    tmp_path: Path,
) -> None:
    plan_path, binding = _plan_source(tmp_path, scope="last1")
    artifact = build_source_e1_recipe_anchor_authority_artifact(plan_path)
    assert isinstance(artifact, E1RecipeAnchorAuthorityArtifact)
    assert artifact.authority.trainable_plan_sha256 == binding.trainable_plan_sha256
    assert tuple(row.anchor_name for row in artifact.authority.anchors) == (
        "adamw",
        "sgdm",
    )
    assert artifact.authority.anchor("adamw").optimizer.learning_rate == 1e-4
    assert artifact.authority.anchor("sgdm").optimizer.momentum == 0.9

    output = (tmp_path / "e1-anchor-authority.json").resolve()
    publish_e1_recipe_anchor_authority_artifact(artifact, output)
    assert load_e1_recipe_anchor_authority_artifact(output) == artifact
    with pytest.raises(RuntimeError, match="target already exists"):
        publish_e1_recipe_anchor_authority_artifact(artifact, output)

    plan_path.chmod(0o600)
    plan_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        load_e1_recipe_anchor_authority_artifact(output)


def test_tts_source_authority_binds_full_plan_disjoint_window_and_loss(
    tmp_path: Path,
) -> None:
    plan_path, binding = _plan_source(tmp_path, scope="all")
    assert binding.mode == "full" and binding.scope == "all"
    paper_pdf = (tmp_path / "tts-v2.pdf").resolve()
    paper_source = (tmp_path / "tts-v2.tex").resolve()
    paper_pdf.write_bytes(b"%PDF-1.7\nsource-owned TTS v2 fixture\n")
    paper_source.write_text("TTS source v2\n", encoding="utf-8")
    tuning_path = (tmp_path / "tts-window.json").resolve()
    tuning = _tts_window()
    publish_canonical_json_no_replace(tuning_path, tuning.to_dict())
    loss_path = (tmp_path / "tts-loss.json").resolve()
    publish_canonical_json_no_replace(loss_path, TTS_DRAFTER_NATIVE_LOSS_SOURCE)

    artifact = build_source_tts_calibration_authority_artifact(
        paper_pdf_path=paper_pdf,
        paper_source_path=paper_source,
        tuning_window_path=tuning_path,
        trainable_plan_authority_path=plan_path,
        drafter_native_loss_path=loss_path,
    )
    assert artifact.authority.trainable_plan_sha256 == binding.trainable_plan_sha256
    assert artifact.authority.paper_pdf_sha256 == artifact.paper_pdf_source.raw_sha256
    output = (tmp_path / "tts-authority.json").resolve()
    publish_tts_calibration_authority_artifact(artifact, output)
    assert load_tts_calibration_authority_artifact(output) == artifact

    overlapping = {
        **tuning.to_dict(),
        "excluded_pilot_entries": [
            row.to_dict()
            for row in sorted(
                (
                    tuning.tuning_entries[0],
                    *tuning.excluded_pilot_entries[1:],
                ),
                key=lambda row: row.entry_id,
            )
        ],
    }
    overlap_path = (tmp_path / "overlap.json").resolve()
    publish_canonical_json_no_replace(overlap_path, overlapping)
    with pytest.raises(ValueError, match="overlaps"):
        build_source_tts_calibration_authority_artifact(
            paper_pdf_path=paper_pdf,
            paper_source_path=paper_source,
            tuning_window_path=overlap_path,
            trainable_plan_authority_path=plan_path,
            drafter_native_loss_path=loss_path,
        )


def test_chronobelief_source_authority_reopens_pdf_tex_and_fixed_equations(
    tmp_path: Path,
) -> None:
    paper_pdf = (tmp_path / "paper.pdf").resolve()
    tex_source = (tmp_path / "paper.tex").resolve()
    paper_pdf.write_bytes(b"%PDF-1.7\nChronoBelief fixture\n")
    tex_source.write_text("equations 5.5--5.8\n", encoding="utf-8")
    artifact = build_source_chronobelief_authority_artifact(
        paper_pdf_path=paper_pdf,
        tex_source_path=tex_source,
    )
    assert len(artifact.authority.equations) == 4
    assert artifact.authority.bias_correction == "standard_update_count"
    assert artifact.authority.weight_decay_semantics == "decoupled"
    assert artifact.authority.age_semantics == "safe_boundary_age"

    output = (tmp_path / "chronobelief-authority.json").resolve()
    publish_chronobelief_authority_artifact(artifact, output)
    assert load_chronobelief_authority_artifact(output) == artifact
    with pytest.raises(ValueError, match="source-owned replay"):
        replace(
            artifact,
            authority=replace(
                artifact.authority,
                paper_pdf_sha256="9" * 64,
            ),
        )
