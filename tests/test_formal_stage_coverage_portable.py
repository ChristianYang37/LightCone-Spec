from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_stage_coverage import (
    FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
    FormalStageCoverageProofArtifact,
)
from lightcone_spec.experiments.formal_stage_coverage_portable import (
    FORMAL_PORTABLE_STAGE_COVERAGE_PROTOCOL_SHA256,
    FormalPortableStageCoverageProofArtifact,
    bind_formal_portable_stage_coverage_proof_artifact,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return content_sha256({"portable-coverage-test": label})


def _binding(tmp_path: Path, label: str) -> CanonicalJsonProofBinding:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = (tmp_path / f"{label}.json").resolve()
    publish_canonical_json_no_replace(path, {"kind": label})
    return CanonicalJsonProofBinding.bind(path)


def _artifact(
    tmp_path: Path,
    *,
    stage: str,
    phase: str,
) -> FormalPortableStageCoverageProofArtifact:
    downstream_phase = (stage, phase) == ("E3b", "excluded_pilot")
    prefix_phase = not downstream_phase and (stage, phase) not in {
        ("E3a", "capacity"),
        ("TTS-Cal", "calibration"),
        ("E1", "selection"),
    }
    return FormalPortableStageCoverageProofArtifact(
        schema_version=3,
        kind="formal_portable_stage_coverage_proof_artifact",
        protocol_sha256=FORMAL_PORTABLE_STAGE_COVERAGE_PROTOCOL_SHA256,
        stage=stage,  # type: ignore[arg-type]
        phase=phase,
        coverage_receipt_sha256=_sha("coverage"),
        materialization_receipt_sha256=_sha("materialization"),
        registry_verification_receipt_sha256=_sha("registry"),
        coverage_proof_source=_binding(tmp_path, "coverage-proof"),
        registry_layer_source=_binding(tmp_path, "registry-layer"),
        prior_prefix_source=(
            _binding(tmp_path, "prior-prefix") if prefix_phase else None
        ),
        e1_recipe_anchor_authority_source=(
            _binding(tmp_path, "e1-anchor") if stage == "E1" else None
        ),
        downstream_pilot_precoverage_source=(
            _binding(tmp_path, "downstream-pilot-precoverage")
            if downstream_phase
            else None
        ),
    )


def _low_level_coverage_path(
    tmp_path: Path,
    *,
    stage: str,
    phase: str,
) -> Path:
    common = {
        name: _binding(tmp_path, name)
        for name in (
            "protocol-lock",
            "runtime-authority",
            "materialization",
            "inventory",
            "derived-coverage",
        )
    }
    is_tts = stage == "TTS-Cal"
    artifact = FormalStageCoverageProofArtifact(
        schema_version=1,
        kind="formal_stage_coverage_proof_artifact",
        protocol_sha256=FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
        stage=stage,
        phase=phase,
        protocol_lock_sha256=_sha("protocol-lock"),
        formal_runtime_authority_manifest_sha256=_sha("runtime-authority"),
        materialization_receipt_sha256=_sha("materialization"),
        inventory_sha256=_sha("inventory"),
        coverage_receipt_sha256=_sha("coverage"),
        protocol_lock_source=common["protocol-lock"],
        runtime_authority_source=common["runtime-authority"],
        materialization_source=common["materialization"],
        inventory_source=common["inventory"],
        tts_authority_source=(_binding(tmp_path, "tts-authority") if is_tts else None),
        raw_tts_evidence_source=(
            _binding(tmp_path, "raw-tts-evidence") if is_tts else None
        ),
        stage_source_rebuild_input_source=(
            None if is_tts else _binding(tmp_path, "stage-source")
        ),
        evidence_shard_sources=(
            () if is_tts else (_binding(tmp_path, "evidence-shard"),)
        ),
        execution_rebuild_shard_sources=(
            () if is_tts else (_binding(tmp_path, "execution-shard"),)
        ),
        candidate_replay_proof_sources=(),
        derived_coverage_shard_sources=(common["derived-coverage"],),
    )
    path = (tmp_path / "low-level-coverage.json").resolve()
    publish_canonical_json_no_replace(path, artifact.to_dict())
    return path


@pytest.mark.parametrize(
    ("stage", "phase"),
    (
        ("E3a", "capacity"),
        ("TTS-Cal", "calibration"),
        ("E1", "selection"),
        ("E2", "round0"),
        ("E2", "round3"),
        ("E4", "screen"),
        ("E4", "profiler"),
        ("E3b", "excluded_pilot"),
    ),
)
def test_portable_coverage_codec_has_closed_phase_dependency_union(
    tmp_path: Path,
    stage: str,
    phase: str,
) -> None:
    artifact = _artifact(tmp_path, stage=stage, phase=phase)
    assert (
        FormalPortableStageCoverageProofArtifact.from_dict(artifact.to_dict())
        == artifact
    )
    assert len(json.dumps(artifact.to_dict()).encode()) < 2 * 1024 * 1024


def test_portable_coverage_requires_registry_and_exact_predecessor_kind(
    tmp_path: Path,
) -> None:
    tts = _artifact(tmp_path / "tts", stage="TTS-Cal", phase="calibration")
    with pytest.raises(TypeError, match="registry is not path-bound"):
        replace(tts, registry_layer_source=None)  # type: ignore[arg-type]

    e2 = _artifact(tmp_path / "e2", stage="E2", phase="round0")
    with pytest.raises(ValueError, match="predecessor union"):
        replace(e2, prior_prefix_source=None)
    with pytest.raises(ValueError, match="E1 anchor union"):
        replace(
            e2,
            e1_recipe_anchor_authority_source=_binding(
                tmp_path / "foreign", "future-anchor"
            ),
        )

    e3b = _artifact(tmp_path / "e3b", stage="E3b", phase="excluded_pilot")
    with pytest.raises(ValueError, match="downstream coverage source union"):
        replace(e3b, downstream_pilot_precoverage_source=None)


def test_portable_coverage_rejects_nested_digest_and_path_alias(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path, stage="E1", phase="selection")
    tampered = artifact.to_dict()
    tampered["coverage_receipt_sha256"] = _sha("foreign-coverage")
    with pytest.raises(ValueError, match="digest differs"):
        FormalPortableStageCoverageProofArtifact.from_dict(tampered)
    with pytest.raises(ValueError, match="reuses a top-level source path"):
        replace(
            artifact,
            e1_recipe_anchor_authority_source=artifact.coverage_proof_source,
        )


@pytest.mark.parametrize(
    ("stage", "phase", "message"),
    (
        (
            "TTS-Cal",
            "calibration",
            "exact-288 source-owned execution artifact",
        ),
    ),
)
def test_portable_coverage_fail_closes_unregistered_source_owned_boundaries(
    tmp_path: Path,
    stage: str,
    phase: str,
    message: str,
) -> None:
    coverage_path = _low_level_coverage_path(
        tmp_path,
        stage=stage,
        phase=phase,
    )
    with pytest.raises(ValueError, match=message):
        bind_formal_portable_stage_coverage_proof_artifact(
            coverage_path,
            registry_layer_path=tmp_path / "must-not-be-opened.json",
            now_ns=1,
        )
