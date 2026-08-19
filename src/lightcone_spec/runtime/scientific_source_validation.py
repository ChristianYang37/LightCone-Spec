"""Path-bound proof replay required before signing derived science results.

An Ed25519 signature proves who signed a payload; it does not prove that the
payload is the output of the preregistered reducer.  This module closes that
gap for the sequential E1/E2/E4 reducers.  The validation artifact binds one
immutable formal-stage prefix, and both candidate creation and finalization
reopen the prefix and rerun its reducer before accepting the payload.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.relocatable_evidence import (
    activate_relocatable_evidence_bundle,
    validate_relocatable_evidence_bundle,
)

SCIENTIFIC_SOURCE_VALIDATION_ARTIFACT_KIND = (
    "lightcone_scientific_source_validation_artifact"
)
FORMAL_PREFIX_SCIENTIFIC_ARTIFACT_TYPES = frozenset(
    {
        "e1-survivor-selection",
        "e2-staged-selection",
        "e4-profiler-completion",
        "e4-stage-selection",
    }
)
COVERAGE_SCIENTIFIC_ARTIFACT_TYPES = frozenset({"stage-coverage"})
MATERIALIZATION_SCIENTIFIC_ARTIFACT_TYPES = frozenset({"stage-materialization"})
TTS_REDUCTION_SCIENTIFIC_ARTIFACT_TYPES = frozenset({"tts-calibration-seal"})
E3A_SELECTION_SCIENTIFIC_ARTIFACT_TYPES = frozenset({"e3a-staged-selection"})
GPU_HOUR_SCIENTIFIC_ARTIFACT_TYPES = frozenset({"stage-gpu-hour-envelope"})
PROTOCOL_LOCK_SCIENTIFIC_ARTIFACT_TYPES = frozenset({"protocol-lock"})
# The typed proof roots already replay E3b reducers, but the currently released
# reducers still use per-request rate where the protocol requires SLO-goodput.
# Keep every downstream result outside the signing allowlist until the shared
# integer SLO-goodput reducer is the code-owned implementation.
DOWNSTREAM_REDUCTION_SCIENTIFIC_ARTIFACT_TYPES: frozenset[str] = frozenset()
E0_FINAL_SCIENTIFIC_ARTIFACT_TYPES = frozenset(
    {"e0-final-completion", "e0-formal-breadth-fdr"}
)
PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES = (
    FORMAL_PREFIX_SCIENTIFIC_ARTIFACT_TYPES
    | COVERAGE_SCIENTIFIC_ARTIFACT_TYPES
    | MATERIALIZATION_SCIENTIFIC_ARTIFACT_TYPES
    | TTS_REDUCTION_SCIENTIFIC_ARTIFACT_TYPES
    | E3A_SELECTION_SCIENTIFIC_ARTIFACT_TYPES
    | GPU_HOUR_SCIENTIFIC_ARTIFACT_TYPES
    | PROTOCOL_LOCK_SCIENTIFIC_ARTIFACT_TYPES
    | DOWNSTREAM_REDUCTION_SCIENTIFIC_ARTIFACT_TYPES
    | E0_FINAL_SCIENTIFIC_ARTIFACT_TYPES
)


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


@dataclass(frozen=True)
class ScientificSourceValidationArtifact:
    """Small durable binding to one source-owned proof reducer invocation."""

    schema_version: Literal[2]
    kind: Literal["lightcone_scientific_source_validation_artifact"]
    artifact_type: str
    proof_kind: Literal[
        "formal_stage_prefix",
        "formal_stage_coverage",
        "formal_materialization",
        "tts_calibration_reduction",
        "formal_e3a_staged_selection",
        "formal_stage_gpu_hour_envelope",
        "formal_protocol_lock",
        "formal_downstream_reduction",
        "e0_final_result",
    ]
    proof_bundle_source: CanonicalJsonProofBinding
    proof_entry_remote_absolute_path: str
    expected_payload_sha256: str
    created_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.kind != SCIENTIFIC_SOURCE_VALIDATION_ARTIFACT_KIND
            or self.artifact_type not in PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES
            or self.proof_kind != _proof_kind_for_artifact_type(self.artifact_type)
        ):
            raise ValueError("scientific source-validation identity is unsupported")
        if type(self.proof_bundle_source) is not CanonicalJsonProofBinding:
            raise TypeError("scientific source-validation bundle is not path-bound")
        proof_entry = Path(self.proof_entry_remote_absolute_path)
        if (
            type(self.proof_entry_remote_absolute_path) is not str
            or not proof_entry.is_absolute()
            or proof_entry != Path(os.path.abspath(proof_entry))
        ):
            raise ValueError("scientific source-validation entry is not absolute")
        _sha256("scientific source-validation payload", self.expected_payload_sha256)
        if type(self.created_ns) is not int or self.created_ns < 1:
            raise ValueError("scientific source-validation time must be positive")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "artifact_type": self.artifact_type,
            "proof_kind": self.proof_kind,
            "proof_bundle_source": self.proof_bundle_source.to_dict(),
            "proof_entry_remote_absolute_path": (self.proof_entry_remote_absolute_path),
            "expected_payload_sha256": self.expected_payload_sha256,
            "created_ns": self.created_ns,
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "artifact_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("scientific source-validation fields differ")
        row = dict(value)
        declared = _sha256(
            "scientific source-validation artifact", row.pop("artifact_sha256")
        )
        row["proof_bundle_source"] = CanonicalJsonProofBinding.from_dict(
            row["proof_bundle_source"]
        )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("scientific source-validation digest differs")
        return artifact


def _formal_prefix_expected_payload(
    artifact_type: str,
    proof_path: str | Path,
    *,
    now_ns: int,
) -> object:
    from lightcone_spec.experiments.formal_stage_prefix import (
        load_and_rebuild_formal_stage_prefix,
        reduce_formal_stage_prefix,
    )

    prefix = load_and_rebuild_formal_stage_prefix(proof_path, now_ns=now_ns)
    expected_phase = {
        "e1-survivor-selection": "e1_selection",
        "e2-staged-selection": "e2_",
        "e4-profiler-completion": "e4_profiler",
        "e4-stage-selection": "e4_",
    }[artifact_type]
    phase = prefix.artifact.phase
    if (
        (expected_phase == "e1_selection" and phase != expected_phase)
        or (expected_phase != "e1_selection" and not phase.startswith(expected_phase))
        or (artifact_type == "e4-stage-selection" and phase == "e4_profiler")
    ):
        raise ValueError("scientific validation proof phase differs from payload type")
    return reduce_formal_stage_prefix(prefix, now_ns=now_ns)


def _proof_kind_for_artifact_type(artifact_type: str) -> str:
    if artifact_type in FORMAL_PREFIX_SCIENTIFIC_ARTIFACT_TYPES:
        return "formal_stage_prefix"
    if artifact_type in COVERAGE_SCIENTIFIC_ARTIFACT_TYPES:
        return "formal_stage_coverage"
    if artifact_type in MATERIALIZATION_SCIENTIFIC_ARTIFACT_TYPES:
        return "formal_materialization"
    if artifact_type in TTS_REDUCTION_SCIENTIFIC_ARTIFACT_TYPES:
        return "tts_calibration_reduction"
    if artifact_type in E3A_SELECTION_SCIENTIFIC_ARTIFACT_TYPES:
        return "formal_e3a_staged_selection"
    if artifact_type in GPU_HOUR_SCIENTIFIC_ARTIFACT_TYPES:
        return "formal_stage_gpu_hour_envelope"
    if artifact_type in PROTOCOL_LOCK_SCIENTIFIC_ARTIFACT_TYPES:
        return "formal_protocol_lock"
    if artifact_type in DOWNSTREAM_REDUCTION_SCIENTIFIC_ARTIFACT_TYPES:
        return "formal_downstream_reduction"
    return "e0_final_result"


def _materialization_expected_payload(
    proof_path: str | Path,
    *,
    now_ns: int,
) -> object:
    """Rebuild a materialization from its typed predecessor, not its bytes.

    Materialization shards are bounded transport only: accepting an arbitrary
    structurally valid sharded receipt here would let the signer invent cells.
    Initial preflight/E3a/TTS-Cal/E1 rows use their exact schema-5 registry
    predecessor, E2/E4 use the completed sequential prefix, and post-E4 rows
    use the current-only downstream predecessor proof.  Unsupported downstream
    phases remain fail-closed until their typed predecessor adapters are
    registered in that proof dispatcher.
    """

    from lightcone_spec.experiments.formal_downstream_prefix import (
        FORMAL_DOWNSTREAM_MATERIALIZATION_PROOF_KIND,
        rebuild_formal_downstream_materialization_proof,
    )
    from lightcone_spec.experiments.formal_initial_stage_proof import (
        FORMAL_INITIAL_STAGE_MATERIALIZATION_PROOF_KIND,
        revalidate_formal_initial_stage_materialization_proof_artifact,
    )
    from lightcone_spec.experiments.formal_stage_prefix import (
        FORMAL_STAGE_PREFIX_ARTIFACT_KIND,
        load_and_rebuild_formal_stage_prefix,
        materialize_next_formal_stage_from_prefix,
    )

    binding = CanonicalJsonProofBinding.bind(proof_path)
    value = binding.reopen()
    if type(value) is not dict or type(value.get("kind")) is not str:
        raise ValueError("stage materialization proof kind is invalid")
    if value["kind"] == FORMAL_INITIAL_STAGE_MATERIALIZATION_PROOF_KIND:
        return revalidate_formal_initial_stage_materialization_proof_artifact(
            binding.absolute_path,
            now_ns=now_ns,
        )
    if value["kind"] == FORMAL_DOWNSTREAM_MATERIALIZATION_PROOF_KIND:
        return rebuild_formal_downstream_materialization_proof(
            binding.absolute_path,
            now_ns=now_ns,
        ).materialization
    if value["kind"] != FORMAL_STAGE_PREFIX_ARTIFACT_KIND:
        raise ValueError(
            "stage materialization requires a typed predecessor reducer proof"
        )
    prefix = load_and_rebuild_formal_stage_prefix(
        binding.absolute_path,
        now_ns=now_ns,
    )
    return materialize_next_formal_stage_from_prefix(
        prefix,
        registry_verification_receipt=prefix.registry_verification_receipt,
        now_ns=now_ns,
    )


def _coverage_expected_payload(
    proof_path: str | Path,
    *,
    now_ns: int,
) -> object:
    from lightcone_spec.experiments.formal_stage_coverage_portable import (
        revalidate_portable_formal_stage_coverage_proof_artifact,
    )

    # The low-level coverage graph is only a reducer input.  Scientific signing
    # additionally requires its exact proof-carrying pre-coverage registry and
    # predecessor prefix, from which all verifier-private inputs are rebuilt.
    return revalidate_portable_formal_stage_coverage_proof_artifact(
        proof_path,
        now_ns=now_ns,
    ).coverage


def _tts_reduction_expected_payload(
    proof_path: str | Path,
    *,
    now_ns: int,
) -> object:
    from lightcone_spec.experiments.tts_calibration_authority import (
        revalidate_formal_tts_calibration_reduction_proof_artifact,
    )

    _reduction, seal = revalidate_formal_tts_calibration_reduction_proof_artifact(
        proof_path,
        now_ns=now_ns,
    )
    return seal


def _e3a_selection_expected_payload(
    proof_path: str | Path,
    *,
    now_ns: int,
) -> object:
    from lightcone_spec.experiments.e3a_staged_selection_proof import (
        revalidate_formal_e3a_staged_selection_proof_artifact,
    )

    _selection, receipt = revalidate_formal_e3a_staged_selection_proof_artifact(
        proof_path,
        now_ns=now_ns,
    )
    return receipt


def _gpu_hour_expected_payload(
    proof_path: str | Path,
    *,
    now_ns: int,
) -> object:
    from lightcone_spec.experiments.formal_gpu_hour_proof import (
        revalidate_formal_stage_gpu_hour_envelope_proof_artifact,
    )

    return revalidate_formal_stage_gpu_hour_envelope_proof_artifact(
        proof_path,
        now_ns=now_ns,
    )


def _protocol_lock_expected_payload(
    proof_path: str | Path,
    *,
    now_ns: int,
) -> object:
    from lightcone_spec.experiments.formal_protocol_lock_proof import (
        revalidate_formal_protocol_lock_source_proof_artifact,
    )

    return revalidate_formal_protocol_lock_source_proof_artifact(
        proof_path,
        now_ns=now_ns,
    )


def _downstream_reduction_expected_payload(
    proof_path: str | Path,
    *,
    now_ns: int,
) -> object:
    from lightcone_spec.experiments.formal_downstream_prefix import (
        rebuild_formal_downstream_reduction_proof,
    )

    return rebuild_formal_downstream_reduction_proof(
        proof_path,
        now_ns=now_ns,
    ).reduction


def _e0_final_expected_payload(
    artifact_type: str,
    proof_path: str | Path,
    *,
    now_ns: int,
) -> object:
    from lightcone_spec.experiments.breadth_fdr_authority import (
        reduce_formal_e0_breadth_fdr_from_artifact,
    )
    from lightcone_spec.experiments.e0_authority_artifact import (
        reduce_e0_final_completion_from_artifact,
    )
    from lightcone_spec.experiments.registry import build_industrial_registry

    if artifact_type == "e0-final-completion":
        return reduce_e0_final_completion_from_artifact(proof_path, now_ns=now_ns)
    if artifact_type == "e0-formal-breadth-fdr":
        return reduce_formal_e0_breadth_fdr_from_artifact(
            build_industrial_registry(),
            proof_path,
            now_ns=now_ns,
        )
    raise ValueError("E0 final scientific source-validation type is unsupported")


def _expected_payload(
    artifact_type: str,
    proof_path: str | Path,
    *,
    now_ns: int,
) -> object:
    if artifact_type in FORMAL_PREFIX_SCIENTIFIC_ARTIFACT_TYPES:
        return _formal_prefix_expected_payload(
            artifact_type,
            proof_path,
            now_ns=now_ns,
        )
    if artifact_type in COVERAGE_SCIENTIFIC_ARTIFACT_TYPES:
        return _coverage_expected_payload(proof_path, now_ns=now_ns)
    if artifact_type in MATERIALIZATION_SCIENTIFIC_ARTIFACT_TYPES:
        return _materialization_expected_payload(proof_path, now_ns=now_ns)
    if artifact_type in TTS_REDUCTION_SCIENTIFIC_ARTIFACT_TYPES:
        return _tts_reduction_expected_payload(proof_path, now_ns=now_ns)
    if artifact_type in E3A_SELECTION_SCIENTIFIC_ARTIFACT_TYPES:
        return _e3a_selection_expected_payload(proof_path, now_ns=now_ns)
    if artifact_type in GPU_HOUR_SCIENTIFIC_ARTIFACT_TYPES:
        return _gpu_hour_expected_payload(proof_path, now_ns=now_ns)
    if artifact_type in PROTOCOL_LOCK_SCIENTIFIC_ARTIFACT_TYPES:
        return _protocol_lock_expected_payload(proof_path, now_ns=now_ns)
    if artifact_type in DOWNSTREAM_REDUCTION_SCIENTIFIC_ARTIFACT_TYPES:
        return _downstream_reduction_expected_payload(proof_path, now_ns=now_ns)
    return _e0_final_expected_payload(
        artifact_type,
        proof_path,
        now_ns=now_ns,
    )


def build_scientific_source_validation_artifact(
    *,
    artifact_type: str,
    proof_bundle_path: str | Path,
    proof_entry_remote_absolute_path: str | Path | None = None,
    now_ns: int,
) -> ScientificSourceValidationArtifact:
    """Deep-reduce a portable proof bundle and bind its exact payload."""

    if artifact_type not in PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES:
        raise ValueError("scientific source-validation type is unsupported")
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("scientific source-validation time must be positive")
    bundle = CanonicalJsonProofBinding.bind(proof_bundle_path)
    validated = validate_relocatable_evidence_bundle(bundle.absolute_path)
    entries = validated.artifact.entry_remote_absolute_paths
    if proof_entry_remote_absolute_path is None:
        if len(entries) != 1:
            raise ValueError(
                "scientific source-validation bundle requires one explicit entry"
            )
        proof_entry = entries[0]
    else:
        proof_entry = str(Path(proof_entry_remote_absolute_path))
        if proof_entry not in entries:
            raise ValueError(
                "scientific source-validation entry is outside its proof bundle"
            )
    with activate_relocatable_evidence_bundle(bundle.absolute_path):
        expected = _expected_payload(
            artifact_type,
            proof_entry,
            now_ns=now_ns,
        )
    return ScientificSourceValidationArtifact(
        schema_version=2,
        kind=SCIENTIFIC_SOURCE_VALIDATION_ARTIFACT_KIND,
        artifact_type=artifact_type,
        proof_kind=_proof_kind_for_artifact_type(artifact_type),
        proof_bundle_source=bundle,
        proof_entry_remote_absolute_path=proof_entry,
        expected_payload_sha256=content_sha256(expected),
        created_ns=now_ns,
    )


def publish_scientific_source_validation_artifact(
    *,
    artifact_type: str,
    proof_bundle_path: str | Path,
    proof_entry_remote_absolute_path: str | Path | None = None,
    now_ns: int,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    artifact = build_scientific_source_validation_artifact(
        artifact_type=artifact_type,
        proof_bundle_path=proof_bundle_path,
        proof_entry_remote_absolute_path=proof_entry_remote_absolute_path,
        now_ns=now_ns,
    )
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output_path)


def revalidate_scientific_payload_source(
    source_path: str | Path,
    *,
    artifact_type: str,
    payload: object,
    now_ns: int,
) -> ScientificSourceValidationArtifact:
    """Reopen and rerun the bound reducer, then exact-compare the payload."""

    artifact, expected = rebuild_scientific_payload_from_source(
        source_path,
        artifact_type=artifact_type,
        now_ns=now_ns,
    )
    expected_sha256 = content_sha256(expected)
    if (
        expected_sha256 != artifact.expected_payload_sha256
        or content_sha256(payload) != expected_sha256
        or payload != expected
    ):
        raise ValueError("scientific payload differs from its proof reducer")
    return artifact


def rebuild_scientific_payload_from_source(
    source_path: str | Path,
    *,
    artifact_type: str,
    now_ns: int,
) -> tuple[ScientificSourceValidationArtifact, object]:
    """Deep-replay a portable source artifact and return its typed payload."""

    source_binding = CanonicalJsonProofBinding.bind(source_path)
    artifact = ScientificSourceValidationArtifact.from_dict(source_binding.reopen())
    if artifact.artifact_type != artifact_type:
        raise ValueError("scientific source-validation artifact type differs")
    if (
        CanonicalJsonProofBinding.bind(artifact.proof_bundle_source.absolute_path)
        != artifact.proof_bundle_source
    ):
        raise ValueError("scientific source-validation bundle identity changed")
    validated = validate_relocatable_evidence_bundle(
        artifact.proof_bundle_source.absolute_path
    )
    if (
        artifact.proof_entry_remote_absolute_path
        not in validated.artifact.entry_remote_absolute_paths
    ):
        raise ValueError("scientific source-validation entry left its proof bundle")
    with activate_relocatable_evidence_bundle(
        artifact.proof_bundle_source.absolute_path
    ):
        expected = _expected_payload(
            artifact_type,
            artifact.proof_entry_remote_absolute_path,
            now_ns=now_ns,
        )
    if content_sha256(expected) != artifact.expected_payload_sha256:
        raise ValueError("scientific source-validation reducer identity changed")
    if CanonicalJsonProofBinding.bind(source_path) != source_binding:
        raise RuntimeError("scientific source-validation artifact changed")
    return artifact, expected


__all__ = (
    "COVERAGE_SCIENTIFIC_ARTIFACT_TYPES",
    "DOWNSTREAM_REDUCTION_SCIENTIFIC_ARTIFACT_TYPES",
    "E0_FINAL_SCIENTIFIC_ARTIFACT_TYPES",
    "E3A_SELECTION_SCIENTIFIC_ARTIFACT_TYPES",
    "FORMAL_PREFIX_SCIENTIFIC_ARTIFACT_TYPES",
    "GPU_HOUR_SCIENTIFIC_ARTIFACT_TYPES",
    "MATERIALIZATION_SCIENTIFIC_ARTIFACT_TYPES",
    "PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES",
    "PROTOCOL_LOCK_SCIENTIFIC_ARTIFACT_TYPES",
    "SCIENTIFIC_SOURCE_VALIDATION_ARTIFACT_KIND",
    "TTS_REDUCTION_SCIENTIFIC_ARTIFACT_TYPES",
    "ScientificSourceValidationArtifact",
    "build_scientific_source_validation_artifact",
    "publish_scientific_source_validation_artifact",
    "rebuild_scientific_payload_from_source",
    "revalidate_scientific_payload_source",
)
