from __future__ import annotations

import pytest

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_stage_execution import (
    FormalStageSourceRebuildInput,
)


def _sha(label: str) -> str:
    return content_sha256({"formal-stage-source-rebuild-test": label})


@pytest.mark.parametrize(
    ("stage", "phase"),
    (
        ("E4", "screen"),
        ("E4", "local"),
        ("E4", "profiler"),
        ("E3b", "excluded_pilot"),
        ("E3b", "final"),
        ("E1a", "verification"),
        ("E5", "excluded_pilot"),
        ("E5", "final_and_one_shot_failure"),
        ("E6", "excluded_pilot_and_model_preflight"),
        ("E6", "final"),
        ("E0", "onlinespec_tuning"),
        ("E0", "excluded_pilot"),
        ("E0", "final"),
    ),
)
def test_formal_stage_source_descriptor_closed_phase_round_trip(
    stage: str,
    phase: str,
) -> None:
    descriptor = FormalStageSourceRebuildInput(
        schema_version=1,
        kind="formal_stage_source_rebuild_input",
        stage=stage,  # type: ignore[arg-type]
        phase=phase,  # type: ignore[arg-type]
        materialization_receipt_sha256=_sha(f"{stage}-{phase}-materialization"),
        source_decision_sha256=_sha(f"{stage}-{phase}-source"),
        registry_verification_receipt_sha256=_sha("registry"),
        source_input_commitment_sha256=_sha(f"{stage}-{phase}-inputs"),
        expected_stage_source_sha256=_sha(f"{stage}-{phase}-sealed-source"),
    )
    assert FormalStageSourceRebuildInput.from_dict(descriptor.to_dict()) == descriptor


def test_formal_stage_source_descriptor_rejects_foreign_phase_or_digest_fallback() -> (
    None
):
    with pytest.raises(ValueError, match="phase is unsupported"):
        FormalStageSourceRebuildInput(
            schema_version=1,
            kind="formal_stage_source_rebuild_input",
            stage="E5",
            phase="pilot",  # type: ignore[arg-type]
            materialization_receipt_sha256=_sha("materialization"),
            source_decision_sha256=_sha("source"),
            registry_verification_receipt_sha256=_sha("registry"),
            source_input_commitment_sha256=_sha("inputs"),
            expected_stage_source_sha256=_sha("sealed-source"),
        )

    descriptor = FormalStageSourceRebuildInput(
        schema_version=1,
        kind="formal_stage_source_rebuild_input",
        stage="E4",
        phase="screen",
        materialization_receipt_sha256=_sha("materialization"),
        source_decision_sha256=_sha("source"),
        registry_verification_receipt_sha256=_sha("registry"),
        source_input_commitment_sha256=_sha("inputs"),
        expected_stage_source_sha256=_sha("sealed-source"),
    )
    encoded = descriptor.to_dict()
    encoded["digest_only_fallback"] = descriptor.sha256
    with pytest.raises(ValueError, match="fields differ"):
        FormalStageSourceRebuildInput.from_dict(encoded)
