"""Strict JSON codecs and verified manifest for the signed formal registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any, Literal

from lightcone_spec.experiments.downstream_stage_authority import (
    E3bPowerPrefixReceipt,
    E5PowerAndAnchorReceipt,
    FormalFamilyPowerCommitment,
    SignedE3bPowerPrefixReceipt,
    SignedE5PowerAndAnchorReceipt,
)
from lightcone_spec.experiments.e0_stage_authority import (
    E0FormalRegistryAuthorityBundle,
    E0OnlineSpecSelectedRecipe,
    E0OnlineSpecSourceAuthority,
    E0OnlineSpecTuningSeal,
    E0PowerPrefixReceipt,
    SignedE0OnlineSpecTuningSeal,
    SignedE0PowerPrefixReceipt,
)
from lightcone_spec.experiments.e2_stage_authority import (
    E2CellExecutionEvidence,
    E2StagedCandidateEvaluation,
    E2StagedRoundEvidenceManifest,
    E2StagedRoundSelectionReceipt,
    SignedE2StagedRoundSelectionReceipt,
)
from lightcone_spec.experiments.e3a_stage_authority import (
    E3aCapacityObservation,
    E3aLockedOutput,
    E3aStagedSelectionArtifact,
    E3aStagedSelectionReceipt,
    SignedE3aStagedSelectionReceipt,
)
from lightcone_spec.experiments.e4_stage_authority import (
    E4CellExecutionEvidence,
    E4ConfigurationEvaluation,
    E4StagedEvidenceManifest,
    E4StageSelectionReceipt,
    SignedE4StageSelectionReceipt,
)
from lightcone_spec.experiments.e6_stage_authority import (
    E6PowerPrefixReceipt,
    SignedE6PowerPrefixReceipt,
)
from lightcone_spec.experiments.formal_protocol import (
    E0_METHOD_ROLES,
    FORMAL_STAGE_DAG,
    CandidateStateReplay,
    CandidateStateTerminalPair,
    FormalRuntimeAuthorityManifest,
    FormalRuntimeAuthorityMember,
    ProtocolLock,
    SignedProtocolLock,
    SignedTtsCalibrationSeal,
    TtsCalibrationAuthority,
    TtsCalibrationSeal,
    TtsL0CandidateStateCoverage,
    content_sha256,
    reject_banned_model_identity,
    verify_signed_payload,
)
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_decisions import (
    E1SurvivorSelectionReceipt,
    SignedE1SurvivorSelectionReceipt,
)
from lightcone_spec.experiments.stage_materialization import (
    E0_LOADS,
    E0CompatibilityDecision,
    E0CompatibilityReceipt,
    E1Geometry,
    E2CandidateRecipe,
    E5AnchorSelectionReceipt,
    E5SelectedP99Anchor,
    GpuHourEstimate,
    MaterializedCell,
    PilotDurationObservation,
    PilotDurationReceipt,
    SignedE0CompatibilityReceipt,
    SignedE5AnchorSelectionReceipt,
    SignedPilotDurationReceipt,
    SignedStageCoverageReceipt,
    SignedStageGpuHourEnvelope,
    SignedStageMaterializationReceipt,
    StageCellDisposition,
    StageCoverageReceipt,
    StageGpuHourEnvelope,
    StageMaterializationReceipt,
    default_e2_recipe_grid_authority,
    e2_candidate_recipes,
)
from lightcone_spec.experiments.statistics import ContrastPower, PowerSizingPlan
from lightcone_spec.orchestration.native_terminal import (
    validate_candidate_state_replay_proof_artifact,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    VerifiedControlArtifact,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _require_sha256_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _strict(label: str, value: object, fields: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a JSON object with exact string keys")
    if set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _array(label: str, value: object) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON array")
    return value


def _tuple_text(label: str, value: object) -> tuple[str, ...]:
    rows = _array(label, value)
    if any(type(row) is not str for row in rows):
        raise TypeError(f"{label} must contain exact strings")
    return tuple(rows)


def _json_tree(value: object) -> object:
    """Convert an already validated dataclass tree to JSON container types."""

    if type(value) is dict:
        return {key: _json_tree(item) for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_json_tree(item) for item in value]
    return value


def challenge_from_dict(value: object) -> AttestationChallenge:
    row = _strict(
        "attestation challenge",
        value,
        frozenset(
            {
                "schema_version",
                "kind",
                "challenge_id",
                "nonce_base64",
                "subject_sha256",
                "issued_ns",
                "expires_ns",
            }
        ),
    )
    challenge = AttestationChallenge(**row)
    challenge.validate()
    return challenge


def signed_attestation_from_dict(value: object) -> SignedAttestation:
    row = _strict(
        "signed attestation",
        value,
        frozenset(
            {
                "schema_version",
                "kind",
                "algorithm",
                "attester_id",
                "key_id",
                "environment",
                "public_key_base64",
                "challenge_sha256",
                "payload_sha256",
                "signature_base64",
            }
        ),
    )
    attestation = SignedAttestation(**row)
    attestation.validate()
    return attestation


def protocol_lock_to_dict(value: ProtocolLock) -> dict[str, object]:
    if type(value) is not ProtocolLock:
        raise TypeError("ProtocolLock codec requires an exact payload")
    row = asdict(value)
    for name in (
        "method_roles",
        "stage_dag",
        "primary_holm_family",
        "secondary_mechanism_contrasts",
        "e6_models",
    ):
        row[name] = list(row[name])
    if value.schema_version == 4:
        # Preserve the byte-level schema-4 payload consumed by existing signed
        # releases.  The two tagged-source fields are schema-5-only even though
        # their dataclass defaults permit old in-process constructors.
        row.pop("content_source_mode")
        row.pop("trusted_single_operator_content_bundle_sha256")
    return row


def protocol_lock_from_dict(value: object) -> ProtocolLock:
    if type(value) is not dict:
        raise TypeError("ProtocolLock must be an object")
    schema_version = value.get("schema_version")
    expected = set(ProtocolLock.__dataclass_fields__)
    if schema_version == 4:
        expected -= {
            "content_source_mode",
            "trusted_single_operator_content_bundle_sha256",
        }
    row = _strict("ProtocolLock", value, frozenset(expected))
    if schema_version == 4:
        row["content_source_mode"] = "offline_root_signed"
        row["trusted_single_operator_content_bundle_sha256"] = None
    for name in (
        "method_roles",
        "stage_dag",
        "primary_holm_family",
        "secondary_mechanism_contrasts",
        "e6_models",
    ):
        row[name] = _tuple_text(f"ProtocolLock {name}", row[name])
    return ProtocolLock(**row)


def signed_protocol_lock_to_dict(value: SignedProtocolLock) -> dict[str, object]:
    if type(value) is not SignedProtocolLock:
        raise TypeError("signed ProtocolLock codec requires an exact wrapper")
    return {
        "payload": protocol_lock_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
    }


def signed_protocol_lock_from_dict(value: object) -> SignedProtocolLock:
    row = _strict(
        "signed ProtocolLock",
        value,
        frozenset({"payload", "payload_sha256", "challenge", "attestation"}),
    )
    return SignedProtocolLock(
        payload=protocol_lock_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )


def formal_runtime_authority_manifest_to_dict(
    value: FormalRuntimeAuthorityManifest,
) -> dict[str, object]:
    if type(value) is not FormalRuntimeAuthorityManifest:
        raise TypeError("formal runtime authority codec requires an exact manifest")
    return {
        "schema_version": value.schema_version,
        "authority_id": value.authority_id,
        "members": [asdict(row) for row in value.members],
        "manifest_sha256": value.sha256,
    }


def formal_runtime_authority_manifest_from_dict(
    value: object,
) -> FormalRuntimeAuthorityManifest:
    row = _strict(
        "formal runtime authority manifest",
        value,
        frozenset(
            {
                "schema_version",
                "authority_id",
                "members",
                "manifest_sha256",
            }
        ),
    )
    declared_sha256 = _require_sha256_text(
        "formal runtime authority manifest digest", row.pop("manifest_sha256")
    )
    members = tuple(
        FormalRuntimeAuthorityMember(
            **_strict(
                "formal runtime authority member",
                item,
                frozenset(FormalRuntimeAuthorityMember.__dataclass_fields__),
            )
        )
        for item in _array("formal runtime authority members", row.pop("members"))
    )
    manifest = FormalRuntimeAuthorityManifest(members=members, **row)
    if manifest.sha256 != declared_sha256:
        raise ValueError("formal runtime authority digest differs from content")
    return manifest


def publish_formal_runtime_authority_manifest(
    path: str,
    manifest: FormalRuntimeAuthorityManifest,
) -> CanonicalJsonProofBinding:
    """Publish and re-open one immutable canonical runtime-authority artifact."""

    payload = formal_runtime_authority_manifest_to_dict(manifest)
    publish_canonical_json_no_replace(path, payload)
    binding = CanonicalJsonProofBinding.bind(path)
    reopened = formal_runtime_authority_manifest_from_dict(binding.reopen())
    if reopened != manifest:
        raise RuntimeError("formal runtime authority changed during publication")
    return binding


def revalidate_formal_runtime_authority_manifest(
    path: str,
    *,
    expected_manifest_sha256: str,
) -> FormalRuntimeAuthorityManifest:
    """Deep-open a path-bound manifest and require its ProtocolLock identity."""

    _require_sha256_text(
        "expected formal runtime authority manifest", expected_manifest_sha256
    )
    before = CanonicalJsonProofBinding.bind(path)
    manifest = formal_runtime_authority_manifest_from_dict(before.reopen())
    after = CanonicalJsonProofBinding.bind(path)
    if before != after:
        raise RuntimeError("formal runtime authority changed while reopened")
    if manifest.sha256 != expected_manifest_sha256:
        raise ValueError("formal runtime authority differs from ProtocolLock")
    return manifest


def tts_calibration_authority_to_dict(
    value: TtsCalibrationAuthority,
) -> dict[str, object]:
    if type(value) is not TtsCalibrationAuthority:
        raise TypeError("TTS calibration authority codec requires an exact value")
    normalized = _json_tree(asdict(value))
    assert type(normalized) is dict
    return normalized


def tts_calibration_authority_from_dict(
    value: object,
) -> TtsCalibrationAuthority:
    row = _strict(
        "TTS calibration authority",
        value,
        frozenset(TtsCalibrationAuthority.__dataclass_fields__),
    )
    for name in ("learning_rates", "strides", "excluded_pilot_blocks"):
        row[name] = tuple(_array(f"TTS authority {name}", row[name]))
    return TtsCalibrationAuthority(**row)


def signed_tts_calibration_seal_to_dict(
    value: SignedTtsCalibrationSeal,
) -> dict[str, object]:
    if type(value) is not SignedTtsCalibrationSeal:
        raise TypeError("signed TTS seal codec requires an exact wrapper")
    payload = _json_tree(asdict(value.payload))
    assert type(payload) is dict
    return {
        "payload": payload,
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_seal_sha256": value.sha256,
    }


def signed_tts_calibration_seal_from_dict(
    value: object,
) -> SignedTtsCalibrationSeal:
    """Decode an untrusted signed seal; policy verification remains mandatory."""

    row = _strict(
        "signed TTS calibration seal",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_seal_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("signed_seal_sha256")
    payload_row = _strict(
        "TTS calibration seal payload",
        row["payload"],
        frozenset(TtsCalibrationSeal.__dataclass_fields__),
    )
    payload_row["selected_pilot_run_binding_sha256s"] = tuple(
        _array(
            "TTS selected pilot run bindings",
            payload_row["selected_pilot_run_binding_sha256s"],
        )
    )
    # TtsCalibrationSeal intentionally cannot be caller-constructed.  A codec
    # may reconstruct the untrusted signed payload, but it remains unusable
    # until SignedTtsCalibrationSeal.verify() validates the exact authority and
    # release attestation policy.
    payload = object.__new__(TtsCalibrationSeal)
    for name, item in payload_row.items():
        object.__setattr__(payload, name, item)
    payload.__post_init__()
    signed = SignedTtsCalibrationSeal(
        payload=payload,
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if expected_sha256 != signed.sha256:
        raise ValueError("signed TTS calibration seal digest differs from content")
    return signed


def e3a_staged_selection_artifact_to_dict(
    value: E3aStagedSelectionArtifact,
) -> dict[str, object]:
    if type(value) is not E3aStagedSelectionArtifact:
        raise TypeError("E3a selection artifact codec requires an exact value")
    normalized = _json_tree(asdict(value))
    assert type(normalized) is dict
    return {**normalized, "artifact_sha256": value.sha256}


def e3a_staged_selection_artifact_from_dict(
    value: object,
) -> E3aStagedSelectionArtifact:
    row = _strict(
        "E3a staged selection artifact",
        value,
        frozenset(
            (*E3aStagedSelectionArtifact.__dataclass_fields__, "artifact_sha256")
        ),
    )
    declared = _require_sha256_text(
        "E3a staged selection artifact digest", row.pop("artifact_sha256")
    )
    row["observations"] = tuple(
        E3aCapacityObservation(
            **_strict(
                "E3a capacity observation",
                item,
                frozenset(E3aCapacityObservation.__dataclass_fields__),
            )
        )
        for item in _array("E3a capacity observations", row["observations"])
    )
    row["locked_outputs"] = tuple(
        E3aLockedOutput(
            **_strict(
                "E3a locked output",
                item,
                frozenset(E3aLockedOutput.__dataclass_fields__),
            )
        )
        for item in _array("E3a locked outputs", row["locked_outputs"])
    )
    artifact = E3aStagedSelectionArtifact(**row)
    if artifact.sha256 != declared:
        raise ValueError("E3a staged selection artifact digest differs from content")
    return artifact


def signed_e3a_staged_selection_to_dict(
    value: SignedE3aStagedSelectionReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE3aStagedSelectionReceipt:
        raise TypeError("signed E3a selection codec requires an exact wrapper")
    payload = _json_tree(asdict(value.payload))
    assert type(payload) is dict
    return {
        "payload": payload,
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e3a_staged_selection_from_dict(
    value: object,
) -> SignedE3aStagedSelectionReceipt:
    row = _strict(
        "signed E3a staged selection",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    declared = _require_sha256_text(
        "signed E3a staged selection digest", row.pop("signed_receipt_sha256")
    )
    payload_row = _strict(
        "E3a staged selection payload",
        row["payload"],
        frozenset(E3aStagedSelectionReceipt.__dataclass_fields__),
    )
    payload_row["locked_outputs"] = tuple(
        E3aLockedOutput(
            **_strict(
                "E3a staged receipt locked output",
                item,
                frozenset(E3aLockedOutput.__dataclass_fields__),
            )
        )
        for item in _array(
            "E3a staged receipt locked outputs", payload_row["locked_outputs"]
        )
    )
    signed = SignedE3aStagedSelectionReceipt(
        payload=E3aStagedSelectionReceipt(**payload_row),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != declared:
        raise ValueError("signed E3a staged selection digest differs from content")
    return signed


def _e2_candidate_recipe_from_dict(value: object) -> E2CandidateRecipe:
    row = _strict(
        "E2 candidate recipe",
        value,
        frozenset(E2CandidateRecipe.__dataclass_fields__),
    )
    row["geometry"] = E1Geometry(
        **_strict(
            "E2 candidate geometry",
            row["geometry"],
            frozenset(E1Geometry.__dataclass_fields__),
        )
    )
    return E2CandidateRecipe(**row)


def e2_staged_evidence_manifest_to_dict(
    value: E2StagedRoundEvidenceManifest,
) -> dict[str, object]:
    if type(value) is not E2StagedRoundEvidenceManifest:
        raise TypeError("E2 evidence manifest codec requires an exact value")
    normalized = _json_tree(asdict(value))
    assert type(normalized) is dict
    return {**normalized, "manifest_sha256": value.sha256}


def e2_staged_evidence_manifest_from_dict(
    value: object,
) -> E2StagedRoundEvidenceManifest:
    row = _strict(
        "E2 staged evidence manifest",
        value,
        frozenset(
            (*E2StagedRoundEvidenceManifest.__dataclass_fields__, "manifest_sha256")
        ),
    )
    declared = _require_sha256_text(
        "E2 staged evidence manifest digest", row.pop("manifest_sha256")
    )
    cells = []
    for item in _array("E2 staged evidence cells", row["cells"]):
        cell = _strict(
            "E2 staged evidence cell",
            item,
            frozenset(E2CellExecutionEvidence.__dataclass_fields__),
        )
        cell["execution_identity"] = StageItlExecutionIdentity(
            **_strict(
                "E2 staged execution identity",
                cell["execution_identity"],
                frozenset(StageItlExecutionIdentity.__dataclass_fields__),
            )
        )
        cells.append(E2CellExecutionEvidence(**cell))
    row["cells"] = tuple(cells)
    manifest = E2StagedRoundEvidenceManifest(**row)
    if manifest.sha256 != declared:
        raise ValueError("E2 staged evidence manifest digest differs from content")
    return manifest


def signed_e2_staged_selection_to_dict(
    value: SignedE2StagedRoundSelectionReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE2StagedRoundSelectionReceipt:
        raise TypeError("signed E2 selection codec requires an exact wrapper")
    normalized = _json_tree(asdict(value.payload))
    assert type(normalized) is dict
    return {
        "payload": normalized,
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e2_staged_selection_from_dict(
    value: object,
) -> SignedE2StagedRoundSelectionReceipt:
    row = _strict(
        "signed E2 staged selection",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    declared = _require_sha256_text(
        "signed E2 staged selection digest", row.pop("signed_receipt_sha256")
    )
    payload_row = _strict(
        "E2 staged selection payload",
        row["payload"],
        frozenset(E2StagedRoundSelectionReceipt.__dataclass_fields__),
    )
    payload_row["evaluations"] = tuple(
        E2StagedCandidateEvaluation(
            **{
                **_strict(
                    "E2 staged candidate evaluation",
                    item,
                    frozenset(E2StagedCandidateEvaluation.__dataclass_fields__),
                ),
                "recipe": _e2_candidate_recipe_from_dict(
                    _strict(
                        "E2 staged candidate evaluation",
                        item,
                        frozenset(E2StagedCandidateEvaluation.__dataclass_fields__),
                    )["recipe"]
                ),
            }
        )
        for item in _array("E2 staged evaluations", payload_row["evaluations"])
    )
    payload_row["survivor_recipes"] = tuple(
        _e2_candidate_recipe_from_dict(item)
        for item in _array(
            "E2 staged survivor recipes", payload_row["survivor_recipes"]
        )
    )
    if payload_row["final_recipe"] is not None:
        payload_row["final_recipe"] = _e2_candidate_recipe_from_dict(
            payload_row["final_recipe"]
        )
    signed = SignedE2StagedRoundSelectionReceipt(
        payload=E2StagedRoundSelectionReceipt(**payload_row),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != declared:
        raise ValueError("signed E2 staged selection digest differs from content")
    return signed


def e4_staged_evidence_manifest_to_dict(
    value: E4StagedEvidenceManifest,
) -> dict[str, object]:
    if type(value) is not E4StagedEvidenceManifest:
        raise TypeError("E4 evidence manifest codec requires an exact value")
    normalized = _json_tree(asdict(value))
    assert type(normalized) is dict
    return {**normalized, "manifest_sha256": value.sha256}


def e4_staged_evidence_manifest_from_dict(
    value: object,
) -> E4StagedEvidenceManifest:
    row = _strict(
        "E4 staged evidence manifest",
        value,
        frozenset((*E4StagedEvidenceManifest.__dataclass_fields__, "manifest_sha256")),
    )
    declared = _require_sha256_text(
        "E4 staged evidence manifest digest", row.pop("manifest_sha256")
    )
    cells = []
    for item in _array("E4 staged evidence cells", row["cells"]):
        cell = _strict(
            "E4 staged evidence cell",
            item,
            frozenset(E4CellExecutionEvidence.__dataclass_fields__),
        )
        cell["execution_identity"] = StageItlExecutionIdentity(
            **_strict(
                "E4 staged execution identity",
                cell["execution_identity"],
                frozenset(StageItlExecutionIdentity.__dataclass_fields__),
            )
        )
        cells.append(E4CellExecutionEvidence(**cell))
    row["cells"] = tuple(cells)
    manifest = E4StagedEvidenceManifest(**row)
    if manifest.sha256 != declared:
        raise ValueError("E4 staged evidence manifest digest differs from content")
    return manifest


def signed_e4_stage_selection_to_dict(
    value: SignedE4StageSelectionReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE4StageSelectionReceipt:
        raise TypeError("signed E4 selection codec requires an exact wrapper")
    normalized = _json_tree(asdict(value.payload))
    assert type(normalized) is dict
    return {
        "payload": normalized,
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e4_stage_selection_from_dict(
    value: object,
) -> SignedE4StageSelectionReceipt:
    row = _strict(
        "signed E4 staged selection",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    declared = _require_sha256_text(
        "signed E4 staged selection digest", row.pop("signed_receipt_sha256")
    )
    payload_row = _strict(
        "E4 staged selection payload",
        row["payload"],
        frozenset(E4StageSelectionReceipt.__dataclass_fields__),
    )
    payload_row["evaluations"] = tuple(
        E4ConfigurationEvaluation(
            **{
                **_strict(
                    "E4 configuration evaluation",
                    item,
                    frozenset(E4ConfigurationEvaluation.__dataclass_fields__),
                ),
                "configuration": tuple(
                    tuple(pair)
                    for pair in _array(
                        "E4 evaluation configuration",
                        _strict(
                            "E4 configuration evaluation",
                            item,
                            frozenset(E4ConfigurationEvaluation.__dataclass_fields__),
                        )["configuration"],
                    )
                ),
                "cell_ids": _tuple_text(
                    "E4 evaluation cells",
                    _strict(
                        "E4 configuration evaluation",
                        item,
                        frozenset(E4ConfigurationEvaluation.__dataclass_fields__),
                    )["cell_ids"],
                ),
            }
        )
        for item in _array("E4 staged evaluations", payload_row["evaluations"])
    )
    payload_row["winner_configuration"] = tuple(
        tuple(pair)
        for pair in _array(
            "E4 winner configuration", payload_row["winner_configuration"]
        )
    )
    if payload_row["factor_neighborhoods"] is not None:
        payload_row["factor_neighborhoods"] = tuple(
            tuple(item)
            for item in _array(
                "E4 factor neighborhoods", payload_row["factor_neighborhoods"]
            )
        )
    signed = SignedE4StageSelectionReceipt(
        payload=E4StageSelectionReceipt(**payload_row),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != declared:
        raise ValueError("signed E4 staged selection digest differs from content")
    return signed


def signed_e1_survivor_selection_to_dict(
    value: SignedE1SurvivorSelectionReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE1SurvivorSelectionReceipt:
        raise TypeError("signed E1 survivor codec requires an exact wrapper")
    normalized = _json_tree(asdict(value.payload))
    assert type(normalized) is dict
    return {
        "payload": normalized,
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e1_survivor_selection_from_dict(
    value: object,
) -> SignedE1SurvivorSelectionReceipt:
    row = _strict(
        "signed E1 survivor selection",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    declared = _require_sha256_text(
        "signed E1 survivor selection digest", row.pop("signed_receipt_sha256")
    )
    payload_row = _strict(
        "E1 survivor selection payload",
        row["payload"],
        frozenset(E1SurvivorSelectionReceipt.__dataclass_fields__),
    )
    payload_row["surviving_geometries"] = tuple(
        E1Geometry(
            **_strict(
                "E1 survivor geometry",
                item,
                frozenset(E1Geometry.__dataclass_fields__),
            )
        )
        for item in _array(
            "E1 survivor geometries", payload_row["surviving_geometries"]
        )
    )
    signed = SignedE1SurvivorSelectionReceipt(
        payload=E1SurvivorSelectionReceipt(**payload_row),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != declared:
        raise ValueError("signed E1 survivor selection digest differs from content")
    return signed


def gpu_hour_estimate_to_dict(value: GpuHourEstimate) -> dict[str, object]:
    if type(value) is not GpuHourEstimate:
        raise TypeError("GPU-hour codec requires an exact estimate")
    return asdict(value)


def gpu_hour_estimate_from_dict(value: object) -> GpuHourEstimate:
    row = _strict(
        "GPU-hour estimate",
        value,
        frozenset(GpuHourEstimate.__dataclass_fields__),
    )
    return GpuHourEstimate(**row)


def pilot_duration_receipt_to_dict(
    value: PilotDurationReceipt,
) -> dict[str, object]:
    if type(value) is not PilotDurationReceipt:
        raise TypeError("pilot-duration codec requires an exact receipt")
    return {
        "schema_version": value.schema_version,
        "protocol_lock_sha256": value.protocol_lock_sha256,
        "materialization_receipt_sha256": value.materialization_receipt_sha256,
        "schedule_sha256": value.schedule_sha256,
        "inventory_gpu_count": value.inventory_gpu_count,
        "observations": [asdict(row) for row in value.observations],
        "retry_reserve_fraction": value.retry_reserve_fraction,
        "profile_reserve_gpu_hours": value.profile_reserve_gpu_hours,
        "evidence_reserve_gpu_hours": value.evidence_reserve_gpu_hours,
        "receipt_sha256": value.sha256,
    }


def pilot_duration_receipt_from_dict(value: object) -> PilotDurationReceipt:
    row = _strict(
        "pilot-duration receipt",
        value,
        frozenset(
            {
                "schema_version",
                "protocol_lock_sha256",
                "materialization_receipt_sha256",
                "schedule_sha256",
                "inventory_gpu_count",
                "observations",
                "retry_reserve_fraction",
                "profile_reserve_gpu_hours",
                "evidence_reserve_gpu_hours",
                "receipt_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("receipt_sha256")
    observations = []
    for item in _array("pilot duration observations", row["observations"]):
        observation = _strict(
            "pilot duration observation",
            item,
            frozenset(PilotDurationObservation.__dataclass_fields__),
        )
        observations.append(PilotDurationObservation(**observation))
    row["observations"] = tuple(observations)
    receipt = PilotDurationReceipt(**row)
    if expected_sha256 != receipt.sha256:
        raise ValueError("pilot-duration receipt digest differs from content")
    return receipt


def signed_pilot_duration_to_dict(
    value: SignedPilotDurationReceipt,
) -> dict[str, object]:
    if type(value) is not SignedPilotDurationReceipt:
        raise TypeError("signed pilot-duration codec requires an exact wrapper")
    return {
        "payload": pilot_duration_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_pilot_duration_from_dict(value: object) -> SignedPilotDurationReceipt:
    row = _strict(
        "signed pilot-duration receipt",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("signed_receipt_sha256")
    signed = SignedPilotDurationReceipt(
        payload=pilot_duration_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if expected_sha256 != signed.sha256:
        raise ValueError("signed pilot-duration digest differs from content")
    return signed


def stage_gpu_hour_envelope_to_dict(
    value: StageGpuHourEnvelope,
) -> dict[str, object]:
    if type(value) is not StageGpuHourEnvelope:
        raise TypeError("stage GPU-hour codec requires an exact envelope")
    return {
        "schema_version": value.schema_version,
        "protocol_lock_sha256": value.protocol_lock_sha256,
        "materialization_receipt_sha256": value.materialization_receipt_sha256,
        "signed_pilot_receipt_sha256": value.signed_pilot_receipt_sha256,
        "schedule_sha256": value.schedule_sha256,
        "estimate": gpu_hour_estimate_to_dict(value.estimate),
        "envelope_sha256": value.sha256,
    }


def stage_gpu_hour_envelope_from_dict(value: object) -> StageGpuHourEnvelope:
    row = _strict(
        "stage GPU-hour envelope",
        value,
        frozenset(
            {
                "schema_version",
                "protocol_lock_sha256",
                "materialization_receipt_sha256",
                "signed_pilot_receipt_sha256",
                "schedule_sha256",
                "estimate",
                "envelope_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("envelope_sha256")
    row["estimate"] = gpu_hour_estimate_from_dict(row["estimate"])
    envelope = StageGpuHourEnvelope(**row)
    if expected_sha256 != envelope.sha256:
        raise ValueError("stage GPU-hour envelope digest differs from content")
    return envelope


def signed_stage_gpu_hour_to_dict(
    value: SignedStageGpuHourEnvelope,
) -> dict[str, object]:
    if type(value) is not SignedStageGpuHourEnvelope:
        raise TypeError("signed stage GPU-hour codec requires an exact wrapper")
    return {
        "payload": stage_gpu_hour_envelope_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_envelope_sha256": value.sha256,
    }


def signed_stage_gpu_hour_from_dict(value: object) -> SignedStageGpuHourEnvelope:
    row = _strict(
        "signed stage GPU-hour envelope",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_envelope_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("signed_envelope_sha256")
    signed = SignedStageGpuHourEnvelope(
        payload=stage_gpu_hour_envelope_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if expected_sha256 != signed.sha256:
        raise ValueError("signed stage GPU-hour digest differs from content")
    return signed


def materialized_cell_to_dict(value: MaterializedCell) -> dict[str, object]:
    if type(value) is not MaterializedCell:
        raise TypeError("materialized-cell codec requires an exact cell")
    return {
        **asdict(value),
        "dimensions": [list(row) for row in value.dimensions],
        "cell_id": value.cell_id,
    }


def materialized_cell_from_dict(value: object) -> MaterializedCell:
    row = _strict(
        "materialized cell",
        value,
        frozenset(
            {
                "stage",
                "method_role",
                "model",
                "backend",
                "task",
                "publication_policy",
                "recipe_sha256",
                "dimensions",
                "cell_id",
            }
        ),
    )
    raw_dimensions = _array("materialized dimensions", row.pop("dimensions"))
    dimensions = []
    for item in raw_dimensions:
        pair = _array("materialized dimension", item)
        if len(pair) != 2 or type(pair[0]) is not str:
            raise ValueError("materialized dimension must be a name/value pair")
        dimensions.append((pair[0], pair[1]))
    expected_id = row.pop("cell_id")
    cell = MaterializedCell(dimensions=tuple(dimensions), **row)
    if expected_id != cell.cell_id:
        raise ValueError("materialized cell ID differs from content")
    return cell


def stage_materialization_receipt_to_dict(
    value: StageMaterializationReceipt,
) -> dict[str, object]:
    if type(value) is not StageMaterializationReceipt:
        raise TypeError("materialization codec requires an exact receipt")
    return {
        "schema_version": value.schema_version,
        "stage": value.stage,
        "protocol_lock_sha256": value.protocol_lock_sha256,
        "upstream_receipt_sha256s": list(value.upstream_receipt_sha256s),
        "source_decision_sha256": value.source_decision_sha256,
        "materialization_rule": value.materialization_rule,
        "expected_cell_count": value.expected_cell_count,
        "cells": [materialized_cell_to_dict(cell) for cell in value.cells],
        "gpu_hours": gpu_hour_estimate_to_dict(value.gpu_hours),
        "receipt_sha256": value.sha256,
    }


def stage_materialization_receipt_from_dict(
    value: object,
) -> StageMaterializationReceipt:
    row = _strict(
        "stage materialization receipt",
        value,
        frozenset(
            {
                "schema_version",
                "stage",
                "protocol_lock_sha256",
                "upstream_receipt_sha256s",
                "source_decision_sha256",
                "materialization_rule",
                "expected_cell_count",
                "cells",
                "gpu_hours",
                "receipt_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("receipt_sha256")
    row["upstream_receipt_sha256s"] = _tuple_text(
        "materialization upstream receipts", row["upstream_receipt_sha256s"]
    )
    row["cells"] = tuple(
        materialized_cell_from_dict(item)
        for item in _array("materialization cells", row["cells"])
    )
    row["gpu_hours"] = gpu_hour_estimate_from_dict(row["gpu_hours"])
    receipt = StageMaterializationReceipt(**row)
    if expected_sha256 != receipt.sha256:
        raise ValueError("materialization receipt digest differs from content")
    return receipt


def signed_stage_materialization_to_dict(
    value: SignedStageMaterializationReceipt,
) -> dict[str, object]:
    if type(value) is not SignedStageMaterializationReceipt:
        raise TypeError("signed materialization codec requires an exact wrapper")
    return {
        "payload": stage_materialization_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_stage_materialization_from_dict(
    value: object,
) -> SignedStageMaterializationReceipt:
    row = _strict(
        "signed materialization",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("signed_receipt_sha256")
    signed = SignedStageMaterializationReceipt(
        payload=stage_materialization_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if expected_sha256 != signed.sha256:
        raise ValueError("signed materialization digest differs from content")
    return signed


def e0_compatibility_receipt_to_dict(
    value: E0CompatibilityReceipt,
) -> dict[str, object]:
    if type(value) is not E0CompatibilityReceipt:
        raise TypeError("E0 compatibility codec requires an exact receipt")
    return {
        "schema_version": value.schema_version,
        "protocol_lock_sha256": value.protocol_lock_sha256,
        "upstream_e6_receipt_sha256": value.upstream_e6_receipt_sha256,
        "decisions": [asdict(row) for row in value.decisions],
        "receipt_sha256": value.sha256,
    }


def e0_compatibility_receipt_from_dict(value: object) -> E0CompatibilityReceipt:
    row = _strict(
        "E0 compatibility receipt",
        value,
        frozenset(
            {
                "schema_version",
                "protocol_lock_sha256",
                "upstream_e6_receipt_sha256",
                "decisions",
                "receipt_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("receipt_sha256")
    decisions = []
    for item in _array("E0 compatibility decisions", row["decisions"]):
        decision = _strict(
            "E0 compatibility decision",
            item,
            frozenset(E0CompatibilityDecision.__dataclass_fields__),
        )
        decisions.append(E0CompatibilityDecision(**decision))
    row["decisions"] = tuple(decisions)
    receipt = E0CompatibilityReceipt(**row)
    if expected_sha256 != receipt.sha256:
        raise ValueError("E0 compatibility receipt digest differs from content")
    return receipt


def signed_e0_compatibility_to_dict(
    value: SignedE0CompatibilityReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE0CompatibilityReceipt:
        raise TypeError("signed E0 compatibility codec requires an exact wrapper")
    return {
        "payload": e0_compatibility_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e0_compatibility_from_dict(
    value: object,
) -> SignedE0CompatibilityReceipt:
    row = _strict(
        "signed E0 compatibility",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("signed_receipt_sha256")
    signed = SignedE0CompatibilityReceipt(
        payload=e0_compatibility_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if expected_sha256 != signed.sha256:
        raise ValueError("signed E0 compatibility digest differs from content")
    return signed


def e0_onlinespec_source_authority_to_dict(
    value: E0OnlineSpecSourceAuthority,
) -> dict[str, object]:
    if type(value) is not E0OnlineSpecSourceAuthority:
        raise TypeError("E0 OnlineSPEC source codec requires an exact authority")
    return {**asdict(value), "authority_sha256": value.sha256}


def e0_onlinespec_source_authority_from_dict(
    value: object,
) -> E0OnlineSpecSourceAuthority:
    row = _strict(
        "E0 OnlineSPEC source authority",
        value,
        frozenset(
            (*E0OnlineSpecSourceAuthority.__dataclass_fields__, "authority_sha256")
        ),
    )
    expected_sha256 = _require_sha256_text(
        "E0 OnlineSPEC source authority digest", row.pop("authority_sha256")
    )
    authority = E0OnlineSpecSourceAuthority(**row)
    if authority.sha256 != expected_sha256:
        raise ValueError("E0 OnlineSPEC source authority digest differs from content")
    return authority


def publish_e0_onlinespec_source_authority(
    *,
    checkout_path: str,
    audit_path: str,
    output_path: str,
) -> CanonicalJsonProofBinding:
    """Bind and publish the registered OnlineSPEC tree from paths only."""

    authority = E0OnlineSpecSourceAuthority.bind(
        checkout_path=checkout_path,
        audit_path=audit_path,
    )
    publish_canonical_json_no_replace(
        output_path,
        e0_onlinespec_source_authority_to_dict(authority),
    )
    binding = CanonicalJsonProofBinding.bind(output_path)
    rebound = e0_onlinespec_source_authority_from_dict(binding.reopen())
    rebound.revalidate()
    if rebound != authority:
        raise RuntimeError("E0 OnlineSPEC source authority changed during publication")
    return binding


def e0_onlinespec_tuning_seal_to_dict(
    value: E0OnlineSpecTuningSeal,
) -> dict[str, object]:
    if type(value) is not E0OnlineSpecTuningSeal:
        raise TypeError("E0 OnlineSPEC tuning codec requires an exact seal")
    row = asdict(value)
    row["selected_recipes"] = [asdict(item) for item in value.selected_recipes]
    row["seal_sha256"] = value.sha256
    return row


def e0_onlinespec_tuning_seal_from_dict(value: object) -> E0OnlineSpecTuningSeal:
    row = _strict(
        "E0 OnlineSPEC tuning seal",
        value,
        frozenset((*E0OnlineSpecTuningSeal.__dataclass_fields__, "seal_sha256")),
    )
    expected_sha256 = _require_sha256_text(
        "E0 OnlineSPEC tuning seal digest", row.pop("seal_sha256")
    )
    row["selected_recipes"] = tuple(
        E0OnlineSpecSelectedRecipe(
            **_strict(
                "E0 OnlineSPEC selected recipe",
                item,
                frozenset(E0OnlineSpecSelectedRecipe.__dataclass_fields__),
            )
        )
        for item in _array("E0 OnlineSPEC selected recipes", row["selected_recipes"])
    )
    seal = E0OnlineSpecTuningSeal(**row)
    if seal.sha256 != expected_sha256:
        raise ValueError("E0 OnlineSPEC tuning seal digest differs from content")
    return seal


def signed_e0_onlinespec_tuning_seal_to_dict(
    value: SignedE0OnlineSpecTuningSeal,
) -> dict[str, object]:
    if type(value) is not SignedE0OnlineSpecTuningSeal:
        raise TypeError("signed E0 OnlineSPEC tuning codec requires an exact wrapper")
    return {
        "payload": e0_onlinespec_tuning_seal_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e0_onlinespec_tuning_seal_from_dict(
    value: object,
) -> SignedE0OnlineSpecTuningSeal:
    row = _strict(
        "signed E0 OnlineSPEC tuning seal",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    expected_sha256 = _require_sha256_text(
        "signed E0 OnlineSPEC tuning digest",
        row.pop("signed_receipt_sha256"),
    )
    signed = SignedE0OnlineSpecTuningSeal(
        payload=e0_onlinespec_tuning_seal_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != expected_sha256:
        raise ValueError("signed E0 OnlineSPEC tuning digest differs from content")
    return signed


def _power_sizing_to_dict(value: PowerSizingPlan) -> dict[str, object]:
    if type(value) is not PowerSizingPlan:
        raise TypeError("E0 power sizing codec requires an exact plan")
    return {
        "status": value.status,
        "pilot_block_ids": list(value.pilot_block_ids),
        "selected_final_blocks": value.selected_final_blocks,
        "minimum_final_blocks": value.minimum_final_blocks,
        "maximum_final_blocks": value.maximum_final_blocks,
        "target_power": value.target_power,
        "family_alpha": value.family_alpha,
        "adjusted_alpha": value.adjusted_alpha,
        "minimum_relative_effect": value.minimum_relative_effect,
        "minimum_log_effect": value.minimum_log_effect,
        "pilot_log_standard_deviations": [
            list(item) for item in value.pilot_log_standard_deviations
        ],
        "power_grid": [asdict(item) for item in value.power_grid],
    }


def _power_sizing_from_dict(value: object) -> PowerSizingPlan:
    row = _strict(
        "E0 power sizing",
        value,
        frozenset(PowerSizingPlan.__dataclass_fields__),
    )
    row["pilot_block_ids"] = _tuple_text(
        "E0 power sizing pilot block IDs", row["pilot_block_ids"]
    )
    deviations = []
    for item in _array(
        "E0 power sizing pilot deviations", row["pilot_log_standard_deviations"]
    ):
        if type(item) is not list or len(item) != 2 or type(item[0]) is not str:
            raise TypeError("E0 power pilot deviation row is not exact")
        deviations.append((item[0], item[1]))
    row["pilot_log_standard_deviations"] = tuple(deviations)
    row["power_grid"] = tuple(
        ContrastPower(
            **_strict(
                "E0 power grid cell",
                item,
                frozenset(ContrastPower.__dataclass_fields__),
            )
        )
        for item in _array("E0 power grid", row["power_grid"])
    )
    return PowerSizingPlan(**row)


def _family_power_commitment_to_dict(
    value: FormalFamilyPowerCommitment,
) -> dict[str, object]:
    if type(value) is not FormalFamilyPowerCommitment:
        raise TypeError("family power codec requires an exact commitment")
    return {
        "schema_version": value.schema_version,
        "stage": value.stage,
        "model": value.model,
        "task": value.task,
        "family_dimensions": [list(item) for item in value.family_dimensions],
        "family_sha256": value.family_sha256,
        "slo_goodput_protocol_sha256": value.slo_goodput_protocol_sha256,
        "pilot_goodput_observation_sha256s": [
            list(item) for item in value.pilot_goodput_observation_sha256s
        ],
        "power_sizing": _power_sizing_to_dict(value.power_sizing),
    }


def _family_power_commitment_from_dict(value: object) -> FormalFamilyPowerCommitment:
    row = _strict(
        "family power commitment",
        value,
        frozenset(FormalFamilyPowerCommitment.__dataclass_fields__),
    )
    dimensions = []
    for item in _array("family power dimensions", row["family_dimensions"]):
        if type(item) is not list or len(item) != 2 or type(item[0]) is not str:
            raise TypeError("family power dimension row is not exact")
        dimensions.append((item[0], item[1]))
    observations = []
    for item in _array(
        "family power pilot observations",
        row["pilot_goodput_observation_sha256s"],
    ):
        if (
            type(item) is not list
            or len(item) != 3
            or type(item[0]) is not int
            or type(item[1]) is not str
            or type(item[2]) is not str
        ):
            raise TypeError("family power pilot observation row is not exact")
        observations.append((item[0], item[1], item[2]))
    row["family_dimensions"] = tuple(dimensions)
    row["pilot_goodput_observation_sha256s"] = tuple(observations)
    row["power_sizing"] = _power_sizing_from_dict(row["power_sizing"])
    return FormalFamilyPowerCommitment(**row)


def e3b_power_prefix_receipt_to_dict(
    value: E3bPowerPrefixReceipt,
) -> dict[str, object]:
    if type(value) is not E3bPowerPrefixReceipt:
        raise TypeError("E3b power-prefix codec requires an exact receipt")
    row = asdict(value)
    row["family_power_commitments"] = [
        _family_power_commitment_to_dict(item)
        for item in value.family_power_commitments
    ]
    row["selected_final_prefix"] = list(value.selected_final_prefix)
    row["receipt_sha256"] = value.sha256
    return row


def e3b_power_prefix_receipt_from_dict(value: object) -> E3bPowerPrefixReceipt:
    row = _strict(
        "E3b power-prefix receipt",
        value,
        frozenset((*E3bPowerPrefixReceipt.__dataclass_fields__, "receipt_sha256")),
    )
    expected_sha256 = _require_sha256_text(
        "E3b power-prefix receipt digest", row.pop("receipt_sha256")
    )
    row["family_power_commitments"] = tuple(
        _family_power_commitment_from_dict(item)
        for item in _array(
            "E3b family power commitments", row["family_power_commitments"]
        )
    )
    row["selected_final_prefix"] = tuple(
        _array("E3b selected final prefix", row["selected_final_prefix"])
    )
    receipt = E3bPowerPrefixReceipt(**row)
    if receipt.sha256 != expected_sha256:
        raise ValueError("E3b power-prefix receipt digest differs from content")
    return receipt


def signed_e3b_power_prefix_to_dict(
    value: SignedE3bPowerPrefixReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE3bPowerPrefixReceipt:
        raise TypeError("signed E3b power-prefix codec requires an exact wrapper")
    return {
        "payload": e3b_power_prefix_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e3b_power_prefix_from_dict(
    value: object,
) -> SignedE3bPowerPrefixReceipt:
    row = _strict(
        "signed E3b power-prefix receipt",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    expected_sha256 = _require_sha256_text(
        "signed E3b power-prefix digest", row.pop("signed_receipt_sha256")
    )
    signed = SignedE3bPowerPrefixReceipt(
        payload=e3b_power_prefix_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != expected_sha256:
        raise ValueError("signed E3b power-prefix digest differs from content")
    return signed


def e5_power_and_anchor_receipt_to_dict(
    value: E5PowerAndAnchorReceipt,
) -> dict[str, object]:
    if type(value) is not E5PowerAndAnchorReceipt:
        raise TypeError("E5 power/anchor codec requires an exact receipt")
    row = asdict(value)
    row["family_power_commitments"] = [
        _family_power_commitment_to_dict(item)
        for item in value.family_power_commitments
    ]
    row["selected_final_prefix"] = list(value.selected_final_prefix)
    row["p99_anchors"] = [asdict(anchor) for anchor in value.p99_anchors]
    row["receipt_sha256"] = value.sha256
    return row


def e5_power_and_anchor_receipt_from_dict(value: object) -> E5PowerAndAnchorReceipt:
    row = _strict(
        "E5 power/anchor receipt",
        value,
        frozenset((*E5PowerAndAnchorReceipt.__dataclass_fields__, "receipt_sha256")),
    )
    expected_sha256 = _require_sha256_text(
        "E5 power/anchor receipt digest", row.pop("receipt_sha256")
    )
    row["family_power_commitments"] = tuple(
        _family_power_commitment_from_dict(item)
        for item in _array(
            "E5 family power commitments", row["family_power_commitments"]
        )
    )
    row["selected_final_prefix"] = tuple(
        _array("E5 selected final prefix", row["selected_final_prefix"])
    )
    row["p99_anchors"] = tuple(
        E5SelectedP99Anchor(
            **_strict(
                "E5 selected p99 anchor",
                item,
                frozenset(E5SelectedP99Anchor.__dataclass_fields__),
            )
        )
        for item in _array("E5 selected p99 anchors", row["p99_anchors"])
    )
    receipt = E5PowerAndAnchorReceipt(**row)
    if receipt.sha256 != expected_sha256:
        raise ValueError("E5 power/anchor receipt digest differs from content")
    return receipt


def signed_e5_power_and_anchor_to_dict(
    value: SignedE5PowerAndAnchorReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE5PowerAndAnchorReceipt:
        raise TypeError("signed E5 power/anchor codec requires an exact wrapper")
    return {
        "payload": e5_power_and_anchor_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e5_power_and_anchor_from_dict(
    value: object,
) -> SignedE5PowerAndAnchorReceipt:
    row = _strict(
        "signed E5 power/anchor receipt",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    expected_sha256 = _require_sha256_text(
        "signed E5 power/anchor digest", row.pop("signed_receipt_sha256")
    )
    signed = SignedE5PowerAndAnchorReceipt(
        payload=e5_power_and_anchor_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != expected_sha256:
        raise ValueError("signed E5 power/anchor digest differs from content")
    return signed


def e6_power_prefix_receipt_to_dict(
    value: E6PowerPrefixReceipt,
) -> dict[str, object]:
    if type(value) is not E6PowerPrefixReceipt:
        raise TypeError("E6 power-prefix codec requires an exact receipt")
    row = asdict(value)
    row["power_sizing"] = _power_sizing_to_dict(value.power_sizing)
    row["selected_final_prefix"] = list(value.selected_final_prefix)
    row["receipt_sha256"] = value.sha256
    return row


def e6_power_prefix_receipt_from_dict(value: object) -> E6PowerPrefixReceipt:
    row = _strict(
        "E6 power-prefix receipt",
        value,
        frozenset((*E6PowerPrefixReceipt.__dataclass_fields__, "receipt_sha256")),
    )
    expected_sha256 = _require_sha256_text(
        "E6 power-prefix receipt digest", row.pop("receipt_sha256")
    )
    row["power_sizing"] = _power_sizing_from_dict(row["power_sizing"])
    row["selected_final_prefix"] = tuple(
        _array("E6 selected final prefix", row["selected_final_prefix"])
    )
    receipt = E6PowerPrefixReceipt(**row)
    if receipt.sha256 != expected_sha256:
        raise ValueError("E6 power-prefix receipt digest differs from content")
    return receipt


def signed_e6_power_prefix_to_dict(
    value: SignedE6PowerPrefixReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE6PowerPrefixReceipt:
        raise TypeError("signed E6 power-prefix codec requires an exact wrapper")
    return {
        "payload": e6_power_prefix_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e6_power_prefix_from_dict(
    value: object,
) -> SignedE6PowerPrefixReceipt:
    row = _strict(
        "signed E6 power-prefix receipt",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    expected_sha256 = _require_sha256_text(
        "signed E6 power-prefix digest", row.pop("signed_receipt_sha256")
    )
    signed = SignedE6PowerPrefixReceipt(
        payload=e6_power_prefix_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != expected_sha256:
        raise ValueError("signed E6 power-prefix digest differs from content")
    return signed


def e0_power_prefix_receipt_to_dict(
    value: E0PowerPrefixReceipt,
) -> dict[str, object]:
    if type(value) is not E0PowerPrefixReceipt:
        raise TypeError("E0 power-prefix codec requires an exact receipt")
    row = asdict(value)
    row["signed_tuning_seal_sha256s"] = list(value.signed_tuning_seal_sha256s)
    row["power_sizing"] = _power_sizing_to_dict(value.power_sizing)
    row["selected_final_prefix"] = list(value.selected_final_prefix)
    row["receipt_sha256"] = value.sha256
    return row


def e0_power_prefix_receipt_from_dict(value: object) -> E0PowerPrefixReceipt:
    row = _strict(
        "E0 power-prefix receipt",
        value,
        frozenset((*E0PowerPrefixReceipt.__dataclass_fields__, "receipt_sha256")),
    )
    expected_sha256 = _require_sha256_text(
        "E0 power-prefix receipt digest", row.pop("receipt_sha256")
    )
    row["signed_tuning_seal_sha256s"] = _tuple_text(
        "E0 power-prefix tuning seals", row["signed_tuning_seal_sha256s"]
    )
    row["selected_final_prefix"] = tuple(
        _array("E0 selected final prefix", row["selected_final_prefix"])
    )
    row["power_sizing"] = _power_sizing_from_dict(row["power_sizing"])
    receipt = E0PowerPrefixReceipt(**row)
    if receipt.sha256 != expected_sha256:
        raise ValueError("E0 power-prefix receipt digest differs from content")
    return receipt


def signed_e0_power_prefix_to_dict(
    value: SignedE0PowerPrefixReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE0PowerPrefixReceipt:
        raise TypeError("signed E0 power-prefix codec requires an exact wrapper")
    return {
        "payload": e0_power_prefix_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e0_power_prefix_from_dict(value: object) -> SignedE0PowerPrefixReceipt:
    row = _strict(
        "signed E0 power-prefix receipt",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    expected_sha256 = _require_sha256_text(
        "signed E0 power-prefix digest", row.pop("signed_receipt_sha256")
    )
    signed = SignedE0PowerPrefixReceipt(
        payload=e0_power_prefix_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != expected_sha256:
        raise ValueError("signed E0 power-prefix digest differs from content")
    return signed


def e5_anchor_selection_receipt_to_dict(
    value: E5AnchorSelectionReceipt,
) -> dict[str, object]:
    if type(value) is not E5AnchorSelectionReceipt:
        raise TypeError("E5 anchor-selection codec requires an exact receipt")
    return {
        "schema_version": value.schema_version,
        "protocol_lock_sha256": value.protocol_lock_sha256,
        "upstream_e1a_receipt_sha256": value.upstream_e1a_receipt_sha256,
        "power_prefix_decision_sha256": value.power_prefix_decision_sha256,
        "anchors": [
            {**asdict(anchor), "anchor_id": anchor.anchor_id}
            for anchor in value.anchors
        ],
        "receipt_sha256": value.sha256,
    }


def e5_anchor_selection_receipt_from_dict(
    value: object,
) -> E5AnchorSelectionReceipt:
    row = _strict(
        "E5 anchor-selection receipt",
        value,
        frozenset(
            {
                "schema_version",
                "protocol_lock_sha256",
                "upstream_e1a_receipt_sha256",
                "power_prefix_decision_sha256",
                "anchors",
                "receipt_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("receipt_sha256")
    anchors = []
    for item in _array("E5 selected p99 anchors", row["anchors"]):
        anchor_row = _strict(
            "E5 selected p99 anchor",
            item,
            frozenset((*E5SelectedP99Anchor.__dataclass_fields__, "anchor_id")),
        )
        expected_anchor_id = anchor_row.pop("anchor_id")
        anchor = E5SelectedP99Anchor(**anchor_row)
        if expected_anchor_id != anchor.anchor_id:
            raise ValueError("E5 selected p99 anchor ID differs from content")
        anchors.append(anchor)
    row["anchors"] = tuple(anchors)
    receipt = E5AnchorSelectionReceipt(**row)
    if expected_sha256 != receipt.sha256:
        raise ValueError("E5 anchor-selection receipt digest differs from content")
    return receipt


def signed_e5_anchor_selection_to_dict(
    value: SignedE5AnchorSelectionReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE5AnchorSelectionReceipt:
        raise TypeError("signed E5 anchor-selection codec requires an exact wrapper")
    return {
        "payload": e5_anchor_selection_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e5_anchor_selection_from_dict(
    value: object,
) -> SignedE5AnchorSelectionReceipt:
    row = _strict(
        "signed E5 anchor-selection",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("signed_receipt_sha256")
    signed = SignedE5AnchorSelectionReceipt(
        payload=e5_anchor_selection_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if expected_sha256 != signed.sha256:
        raise ValueError("signed E5 anchor-selection digest differs from content")
    return signed


def stage_coverage_receipt_to_dict(value: StageCoverageReceipt) -> dict[str, object]:
    if type(value) is not StageCoverageReceipt:
        raise TypeError("coverage codec requires an exact receipt")
    return {
        "schema_version": value.schema_version,
        "stage": value.stage,
        "protocol_lock_sha256": value.protocol_lock_sha256,
        "materialization_receipt_sha256": value.materialization_receipt_sha256,
        "dispositions": [asdict(row) for row in value.dispositions],
        "tts_l0_candidate_state_coverages": [
            tts_l0_candidate_state_coverage_to_dict(row)
            for row in value.tts_l0_candidate_state_coverages
        ],
        "receipt_sha256": value.sha256,
    }


def tts_l0_candidate_state_coverage_to_dict(
    value: TtsL0CandidateStateCoverage,
) -> dict[str, object]:
    if type(value) is not TtsL0CandidateStateCoverage:
        raise TypeError("candidate-state coverage codec requires an exact receipt")
    return {
        "schema_version": value.schema_version,
        "stage": value.stage,
        "scope": value.scope,
        "protocol_lock_sha256": value.protocol_lock_sha256,
        "materialization_receipt_sha256": (value.materialization_receipt_sha256),
        "pair_id": value.pair_id,
        "tts_cell_id": value.tts_cell_id,
        "l0_naive_cell_id": value.l0_naive_cell_id,
        "tts_native_replay_pointer_sha256": (value.tts_native_replay_pointer_sha256),
        "l0_naive_native_replay_pointer_sha256": (
            value.l0_naive_native_replay_pointer_sha256
        ),
        "qualification_cell_id": value.qualification_cell_id,
        "source_round_plan_sha256": value.source_round_plan_sha256,
        "trainable_plan_sha256": value.trainable_plan_sha256,
        "expected_source_rounds": list(value.expected_source_rounds),
        "tts_observations": [asdict(row) for row in value.tts_observations],
        "l0_naive_observations": [asdict(row) for row in value.l0_naive_observations],
        "terminal_pairs": [asdict(row) for row in value.terminal_pairs],
        "receipt_sha256": value.sha256,
    }


def tts_l0_candidate_state_coverage_from_dict(
    value: object,
) -> TtsL0CandidateStateCoverage:
    row = _strict(
        "TTS/L0 candidate-state coverage",
        value,
        frozenset(
            {
                "schema_version",
                "stage",
                "scope",
                "protocol_lock_sha256",
                "materialization_receipt_sha256",
                "pair_id",
                "tts_cell_id",
                "l0_naive_cell_id",
                "tts_native_replay_pointer_sha256",
                "l0_naive_native_replay_pointer_sha256",
                "qualification_cell_id",
                "source_round_plan_sha256",
                "trainable_plan_sha256",
                "expected_source_rounds",
                "tts_observations",
                "l0_naive_observations",
                "terminal_pairs",
                "receipt_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("receipt_sha256")
    rounds = _array("candidate-state expected rounds", row["expected_source_rounds"])
    if any(type(item) is not int for item in rounds):
        raise TypeError("candidate-state expected rounds must be integers")
    row["expected_source_rounds"] = tuple(rounds)
    for field_name in ("tts_observations", "l0_naive_observations"):
        observations = []
        for item in _array(f"candidate-state {field_name}", row[field_name]):
            observations.append(
                CandidateStateReplay(
                    **_strict(
                        f"candidate-state {field_name} row",
                        item,
                        frozenset(CandidateStateReplay.__dataclass_fields__),
                    )
                )
            )
        row[field_name] = tuple(observations)
    terminal_pairs = []
    for item in _array("candidate-state terminal pairs", row["terminal_pairs"]):
        terminal_pairs.append(
            CandidateStateTerminalPair(
                **_strict(
                    "candidate-state terminal pair",
                    item,
                    frozenset(CandidateStateTerminalPair.__dataclass_fields__),
                )
            )
        )
    row["terminal_pairs"] = tuple(terminal_pairs)
    receipt = TtsL0CandidateStateCoverage(**row)
    if expected_sha256 != receipt.sha256:
        raise ValueError("candidate-state coverage digest differs from content")
    return receipt


def stage_coverage_receipt_from_dict(value: object) -> StageCoverageReceipt:
    row = _strict(
        "stage coverage receipt",
        value,
        frozenset(
            {
                "schema_version",
                "stage",
                "protocol_lock_sha256",
                "materialization_receipt_sha256",
                "dispositions",
                "tts_l0_candidate_state_coverages",
                "receipt_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("receipt_sha256")
    dispositions = []
    for item in _array("coverage dispositions", row["dispositions"]):
        disposition = _strict(
            "coverage disposition",
            item,
            frozenset(StageCellDisposition.__dataclass_fields__),
        )
        dispositions.append(StageCellDisposition(**disposition))
    row["dispositions"] = tuple(dispositions)
    row["tts_l0_candidate_state_coverages"] = tuple(
        tts_l0_candidate_state_coverage_from_dict(item)
        for item in _array(
            "TTS/L0 candidate-state pair coverages",
            row["tts_l0_candidate_state_coverages"],
        )
    )
    receipt = StageCoverageReceipt(**row)
    if expected_sha256 != receipt.sha256:
        raise ValueError("coverage receipt digest differs from content")
    return receipt


def signed_stage_coverage_to_dict(
    value: SignedStageCoverageReceipt,
) -> dict[str, object]:
    if type(value) is not SignedStageCoverageReceipt:
        raise TypeError("signed coverage codec requires an exact wrapper")
    return {
        "payload": stage_coverage_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_stage_coverage_from_dict(value: object) -> SignedStageCoverageReceipt:
    row = _strict(
        "signed coverage",
        value,
        frozenset(
            {
                "payload",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_receipt_sha256",
            }
        ),
    )
    expected_sha256 = row.pop("signed_receipt_sha256")
    signed = SignedStageCoverageReceipt(
        payload=stage_coverage_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=challenge_from_dict(row["challenge"]),
        attestation=signed_attestation_from_dict(row["attestation"]),
    )
    if expected_sha256 != signed.sha256:
        raise ValueError("signed coverage digest differs from content")
    return signed


@dataclass(frozen=True)
class FormalMaterializationBinding:
    stage: str
    materialization_receipt_sha256: str
    signed_receipt_sha256: str
    source_decision_sha256: str
    expected_cell_count: int
    materialization_rule: str
    pilot_materialization_receipt_sha256: str | None = None
    pilot_coverage_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        for label, digest in (
            ("materialization", self.materialization_receipt_sha256),
            ("signed receipt", self.signed_receipt_sha256),
            ("source decision", self.source_decision_sha256),
        ):
            _require_sha256_text(f"formal materialization {label}", digest)
        if (self.pilot_materialization_receipt_sha256 is None) != (
            self.pilot_coverage_receipt_sha256 is None
        ):
            raise ValueError("formal materialization pilot lineage is incomplete")
        if self.pilot_materialization_receipt_sha256 is not None:
            _require_sha256_text(
                "formal materialization pilot receipt",
                self.pilot_materialization_receipt_sha256,
            )
            _require_sha256_text(
                "formal materialization pilot coverage",
                self.pilot_coverage_receipt_sha256,
            )
        if self.stage in {"E3b", "E5", "E6", "E0"} and (
            self.pilot_materialization_receipt_sha256 is None
        ):
            raise ValueError("powered formal stage lacks its tuning-only pilot lineage")


@dataclass(frozen=True)
class FormalCoverageBinding:
    stage: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    signed_receipt_sha256: str
    disposition_count: int
    complete_cell_count: int
    terminal_complete: bool


@dataclass(frozen=True)
class FormalSourceAuthorityBinding:
    """A separately signed upstream decision consumed by one stage append."""

    stage: str
    authority_kind: str
    signed_authority_sha256: str
    payload_sha256: str
    authority_sha256: str
    challenge_sha256: str

    def __post_init__(self) -> None:
        supported = (
            (self.stage, self.authority_kind)
            in {
                ("E3a", "e3a_staged_selection"),
                ("E1", "e1_staged_pareto_survivors"),
                ("TTS-Cal", "tts_calibration_seal"),
                ("E3b", "e3b_power_prefix"),
                ("E1a", "e3b_confirmation"),
                ("E5", "e5_power_and_anchor_prefix"),
                ("E6", "e6_power_prefix"),
                ("E0", "e0_compatibility"),
                ("E0", "e0_onlinespec_tuning_seal"),
                ("E0", "e0_power_prefix"),
            }
            or (
                self.stage == "E2"
                and self.authority_kind
                in {f"e2_round_{index}_staged_selection" for index in range(4)}
            )
            or (
                self.stage == "E4"
                and self.authority_kind in {"e4_screen_selection", "e4_local_selection"}
            )
        )
        if not supported:
            raise ValueError("formal source-authority binding is unsupported")
        for label, digest in (
            ("signed authority", self.signed_authority_sha256),
            ("payload", self.payload_sha256),
            ("authority", self.authority_sha256),
            ("challenge", self.challenge_sha256),
        ):
            _require_sha256_text(f"formal source-authority {label}", digest)


@dataclass(frozen=True)
class FormalCandidateReplayProofBinding:
    """Path/raw identity for one already reserved candidate replay proof."""

    pointer_commitment_sha256: str
    proof_artifact: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            type(self.pointer_commitment_sha256) is not str
            or len(self.pointer_commitment_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.pointer_commitment_sha256
            )
        ):
            raise ValueError("formal candidate replay commitment must be a SHA-256")
        if type(self.proof_artifact) is not CanonicalJsonProofBinding:
            raise TypeError("formal candidate replay proof binding is not exact")


@dataclass(frozen=True)
class FormalRegistryManifest:
    schema_version: int
    kind: Literal["lightcone_formal_signed_registry_manifest"]
    registry_sha256: str
    protocol_lock_sha256: str
    signed_protocol_lock_sha256: str
    prior_registry_verification_receipt_sha256: str | None
    inventory_sha256: str
    deployment_policy_authorization_sha256: str
    trusted_attester_policy_sha256: str
    control_lineage_sha256: str
    control_envelope_sha256s: tuple[str, ...]
    challenge_reservation_sha256: str
    stage_order: tuple[str, ...]
    materializations: tuple[FormalMaterializationBinding, ...]
    coverage: tuple[FormalCoverageBinding, ...]
    source_authorities: tuple[FormalSourceAuthorityBinding, ...]
    candidate_replay_proofs: tuple[FormalCandidateReplayProofBinding, ...]
    status: Literal["LOCKED", "MATERIALIZED_PENDING_COVERAGE", "COVERED"]
    formal_dispatch_authorized: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 2
            or self.kind != "lightcone_formal_signed_registry_manifest"
        ):
            raise ValueError("formal registry manifest schema is unsupported")
        if self.stage_order != FORMAL_STAGE_DAG:
            raise ValueError("formal registry manifest stage order differs")
        for label, digest in (
            ("registry", self.registry_sha256),
            ("protocol lock", self.protocol_lock_sha256),
            ("signed protocol lock", self.signed_protocol_lock_sha256),
            ("inventory", self.inventory_sha256),
            (
                "deployment policy authorization",
                self.deployment_policy_authorization_sha256,
            ),
            ("trusted attester policy", self.trusted_attester_policy_sha256),
            ("control lineage", self.control_lineage_sha256),
            ("challenge reservation", self.challenge_reservation_sha256),
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"formal registry {label} must be a SHA-256")
        if self.prior_registry_verification_receipt_sha256 is not None and (
            type(self.prior_registry_verification_receipt_sha256) is not str
            or len(self.prior_registry_verification_receipt_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.prior_registry_verification_receipt_sha256
            )
        ):
            raise ValueError(
                "formal registry prior verification receipt must be a SHA-256"
            )
        if not self.control_envelope_sha256s or self.control_envelope_sha256s != tuple(
            sorted(set(self.control_envelope_sha256s))
        ):
            raise ValueError("formal registry control envelopes are not canonical")
        if self.formal_dispatch_authorized is not False:
            raise ValueError("registry assembly cannot authorize dispatch")
        if any(
            type(row) is not FormalMaterializationBinding
            for row in self.materializations
        ) or len(
            {row.materialization_receipt_sha256 for row in self.materializations}
        ) != len(self.materializations):
            raise ValueError("formal materialization bindings are not exact and unique")
        if any(type(row) is not FormalCoverageBinding for row in self.coverage):
            raise TypeError("formal coverage bindings are not exact")
        if any(
            type(row) is not FormalSourceAuthorityBinding
            for row in self.source_authorities
        ) or tuple(
            row.signed_authority_sha256 for row in self.source_authorities
        ) != tuple(
            sorted({row.signed_authority_sha256 for row in self.source_authorities})
        ):
            raise ValueError("formal source-authority bindings are not canonical")
        if any(
            type(row) is not FormalCandidateReplayProofBinding
            for row in self.candidate_replay_proofs
        ):
            raise TypeError("formal candidate replay proof bindings are not exact")
        proof_commitments = tuple(
            row.pointer_commitment_sha256 for row in self.candidate_replay_proofs
        )
        proof_artifacts = tuple(
            row.proof_artifact.semantic_sha256 for row in self.candidate_replay_proofs
        )
        if proof_commitments != tuple(sorted(set(proof_commitments))) or len(
            proof_artifacts
        ) != len(set(proof_artifacts)):
            raise ValueError("formal candidate replay proof bindings are not canonical")
        all_terminal = bool(self.coverage) and all(
            row.terminal_complete for row in self.coverage
        )
        expected_status = (
            "LOCKED"
            if not self.materializations
            else "COVERED"
            if len(self.coverage) == len(self.materializations) and all_terminal
            else "MATERIALIZED_PENDING_COVERAGE"
        )
        if self.status != expected_status:
            raise ValueError("formal registry manifest status is inconsistent")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "manifest_sha256": self.sha256}


def formal_registry_manifest_from_dict(value: object) -> FormalRegistryManifest:
    """Decode one manifest without trusting its declared digest or nested paths."""

    row = _strict(
        "formal registry manifest",
        value,
        frozenset((*FormalRegistryManifest.__dataclass_fields__, "manifest_sha256")),
    )
    declared_sha256 = _require_sha256_text(
        "formal registry manifest digest", row.pop("manifest_sha256")
    )
    row["stage_order"] = _tuple_text("formal registry stage order", row["stage_order"])
    row["control_envelope_sha256s"] = _tuple_text(
        "formal registry control envelopes", row["control_envelope_sha256s"]
    )
    row["materializations"] = tuple(
        FormalMaterializationBinding(
            **_strict(
                "formal materialization binding",
                item,
                frozenset(FormalMaterializationBinding.__dataclass_fields__),
            )
        )
        for item in _array("formal materialization bindings", row["materializations"])
    )
    row["coverage"] = tuple(
        FormalCoverageBinding(
            **_strict(
                "formal coverage binding",
                item,
                frozenset(FormalCoverageBinding.__dataclass_fields__),
            )
        )
        for item in _array("formal coverage bindings", row["coverage"])
    )
    row["source_authorities"] = tuple(
        FormalSourceAuthorityBinding(
            **_strict(
                "formal source-authority binding",
                item,
                frozenset(FormalSourceAuthorityBinding.__dataclass_fields__),
            )
        )
        for item in _array(
            "formal source-authority bindings", row["source_authorities"]
        )
    )
    replay_proofs = []
    for item in _array(
        "formal candidate replay proof bindings", row["candidate_replay_proofs"]
    ):
        proof = _strict(
            "formal candidate replay proof binding",
            item,
            frozenset(FormalCandidateReplayProofBinding.__dataclass_fields__),
        )
        replay_proofs.append(
            FormalCandidateReplayProofBinding(
                pointer_commitment_sha256=proof["pointer_commitment_sha256"],
                proof_artifact=CanonicalJsonProofBinding.from_dict(
                    proof["proof_artifact"]
                ),
            )
        )
    row["candidate_replay_proofs"] = tuple(replay_proofs)
    manifest = FormalRegistryManifest(**row)
    if manifest.sha256 != declared_sha256:
        raise ValueError("formal registry manifest digest differs from content")
    return manifest


@dataclass(frozen=True)
class FormalControlReservation:
    """Root-authorized, inventory-bound, atomically reserved control batch."""

    inventory_sha256: str
    deployment_policy_authorization_sha256: str
    trusted_attester_policy_sha256: str
    control_lineage_sha256: str
    control_envelope_sha256s: tuple[str, ...]
    challenge_reservation_sha256: str
    verified_artifacts: tuple[VerifiedControlArtifact, ...]

    def __post_init__(self) -> None:
        if (
            type(self.verified_artifacts) is not tuple
            or not self.verified_artifacts
            or any(
                type(row) is not VerifiedControlArtifact
                for row in self.verified_artifacts
            )
        ):
            raise TypeError("formal control reservation requires verified artifacts")
        expected_envelopes = tuple(
            sorted(row.envelope_sha256 for row in self.verified_artifacts)
        )
        if self.control_envelope_sha256s != expected_envelopes:
            raise ValueError("formal control envelope set differs from verification")
        if any(
            row.deployment_policy_authorization_sha256
            != self.deployment_policy_authorization_sha256
            or row.trusted_attester_policy_sha256 != self.trusted_attester_policy_sha256
            for row in self.verified_artifacts
        ):
            raise ValueError("formal controls use different deployment policies")


@dataclass(frozen=True)
class FormalRegistryVerificationReceipt:
    """Durable, replay-bound verification of one append-only registry prefix.

    The first receipt consumes the ProtocolLock challenge exactly once.  Every
    later receipt nests that immutable prefix and reserves only newly appended
    materialization/coverage challenges.  Revalidation uses each layer's
    recorded reservation time, so a short-lived signature is never silently
    re-dated or required to remain unexpired for the lifetime of the study.
    """

    schema_version: int
    kind: Literal["lightcone_formal_registry_verification_receipt"]
    verified_ns: int
    inventory_sha256: str
    signed_protocol_lock: SignedProtocolLock
    prior_receipt: FormalRegistryVerificationReceipt | None
    appended_signed_materializations: tuple[SignedStageMaterializationReceipt, ...]
    appended_signed_coverage: tuple[SignedStageCoverageReceipt, ...]
    appended_e3a_staged_selection_artifacts: tuple[E3aStagedSelectionArtifact, ...]
    appended_signed_e3a_staged_selections: tuple[SignedE3aStagedSelectionReceipt, ...]
    appended_tts_calibration_authorities: tuple[TtsCalibrationAuthority, ...]
    appended_signed_tts_calibration_seals: tuple[SignedTtsCalibrationSeal, ...]
    control_attestations: tuple[ControlArtifactAttestation, ...]
    reservation: ChallengeReplayReservationBinding
    manifest: FormalRegistryManifest
    appended_e2_staged_evidence_manifests: tuple[
        E2StagedRoundEvidenceManifest, ...
    ] = ()
    appended_signed_e2_staged_selections: tuple[
        SignedE2StagedRoundSelectionReceipt, ...
    ] = ()
    appended_signed_e1_survivor_selections: tuple[
        SignedE1SurvivorSelectionReceipt, ...
    ] = ()
    appended_e4_staged_evidence_manifests: tuple[E4StagedEvidenceManifest, ...] = ()
    appended_signed_e4_stage_selections: tuple[SignedE4StageSelectionReceipt, ...] = ()
    appended_signed_e3b_power_prefixes: tuple[SignedE3bPowerPrefixReceipt, ...] = ()
    appended_signed_e5_power_and_anchor_prefixes: tuple[
        SignedE5PowerAndAnchorReceipt, ...
    ] = ()
    appended_signed_e6_power_prefixes: tuple[SignedE6PowerPrefixReceipt, ...] = ()
    appended_e0_onlinespec_source_authorities: tuple[
        E0OnlineSpecSourceAuthority, ...
    ] = ()
    appended_signed_e0_compatibilities: tuple[SignedE0CompatibilityReceipt, ...] = ()
    appended_signed_e0_onlinespec_tuning_seals: tuple[
        SignedE0OnlineSpecTuningSeal, ...
    ] = ()
    appended_signed_e0_power_prefixes: tuple[SignedE0PowerPrefixReceipt, ...] = ()
    appended_formal_stage_prefix_artifacts: tuple[CanonicalJsonProofBinding, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.kind != "lightcone_formal_registry_verification_receipt"
        ):
            raise ValueError("formal registry verification receipt is unsupported")
        if type(self.verified_ns) is not int or self.verified_ns < 1:
            raise ValueError("formal registry verification time is invalid")
        _require_sha256_text(
            "formal registry verification inventory", self.inventory_sha256
        )
        if type(self.signed_protocol_lock) is not SignedProtocolLock:
            raise TypeError(
                "formal registry verification requires a signed ProtocolLock"
            )
        for label, rows, expected in (
            (
                "materialization",
                self.appended_signed_materializations,
                SignedStageMaterializationReceipt,
            ),
            ("coverage", self.appended_signed_coverage, SignedStageCoverageReceipt),
            (
                "E3a staged selection artifact",
                self.appended_e3a_staged_selection_artifacts,
                E3aStagedSelectionArtifact,
            ),
            (
                "signed E3a staged selection",
                self.appended_signed_e3a_staged_selections,
                SignedE3aStagedSelectionReceipt,
            ),
            (
                "E2 staged evidence manifest",
                self.appended_e2_staged_evidence_manifests,
                E2StagedRoundEvidenceManifest,
            ),
            (
                "signed E2 staged selection",
                self.appended_signed_e2_staged_selections,
                SignedE2StagedRoundSelectionReceipt,
            ),
            (
                "signed E1 survivor selection",
                self.appended_signed_e1_survivor_selections,
                SignedE1SurvivorSelectionReceipt,
            ),
            (
                "E4 staged evidence manifest",
                self.appended_e4_staged_evidence_manifests,
                E4StagedEvidenceManifest,
            ),
            (
                "signed E4 staged selection",
                self.appended_signed_e4_stage_selections,
                SignedE4StageSelectionReceipt,
            ),
            (
                "signed E3b power prefix",
                self.appended_signed_e3b_power_prefixes,
                SignedE3bPowerPrefixReceipt,
            ),
            (
                "signed E5 power/anchor prefix",
                self.appended_signed_e5_power_and_anchor_prefixes,
                SignedE5PowerAndAnchorReceipt,
            ),
            (
                "signed E6 power prefix",
                self.appended_signed_e6_power_prefixes,
                SignedE6PowerPrefixReceipt,
            ),
            (
                "E0 OnlineSPEC source authority",
                self.appended_e0_onlinespec_source_authorities,
                E0OnlineSpecSourceAuthority,
            ),
            (
                "signed E0 compatibility",
                self.appended_signed_e0_compatibilities,
                SignedE0CompatibilityReceipt,
            ),
            (
                "signed E0 OnlineSPEC tuning seal",
                self.appended_signed_e0_onlinespec_tuning_seals,
                SignedE0OnlineSpecTuningSeal,
            ),
            (
                "signed E0 power prefix",
                self.appended_signed_e0_power_prefixes,
                SignedE0PowerPrefixReceipt,
            ),
            (
                "formal stage prefix artifact",
                self.appended_formal_stage_prefix_artifacts,
                CanonicalJsonProofBinding,
            ),
            ("control", self.control_attestations, ControlArtifactAttestation),
            (
                "TTS calibration authority",
                self.appended_tts_calibration_authorities,
                TtsCalibrationAuthority,
            ),
            (
                "signed TTS calibration seal",
                self.appended_signed_tts_calibration_seals,
                SignedTtsCalibrationSeal,
            ),
        ):
            if type(rows) is not tuple or any(
                type(row) is not expected for row in rows
            ):
                raise TypeError(
                    f"formal registry verification {label} rows are not exact"
                )
        if type(self.reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("formal registry verification requires a replay binding")
        if type(self.manifest) is not FormalRegistryManifest:
            raise TypeError("formal registry verification requires an exact manifest")
        if self.reservation.reserved_ns != self.verified_ns:
            raise ValueError(
                "formal registry verification time differs from reservation"
            )
        if self.manifest.inventory_sha256 != self.inventory_sha256:
            raise ValueError("formal registry verification inventory differs")
        current_rows = (
            *self.appended_signed_materializations,
            *self.appended_signed_coverage,
        )
        if len(self.appended_tts_calibration_authorities) != len(
            self.appended_signed_tts_calibration_seals
        ):
            raise ValueError("formal registry TTS authority/seal coverage is not exact")
        if len(self.appended_e3a_staged_selection_artifacts) != len(
            self.appended_signed_e3a_staged_selections
        ):
            raise ValueError(
                "formal registry E3a artifact/receipt coverage is not exact"
            )
        if len(self.appended_e2_staged_evidence_manifests) != len(
            self.appended_signed_e2_staged_selections
        ):
            raise ValueError(
                "formal registry E2 evidence/receipt coverage is not exact"
            )
        if len(self.appended_e4_staged_evidence_manifests) != len(
            self.appended_signed_e4_stage_selections
        ):
            raise ValueError(
                "formal registry E4 evidence/receipt coverage is not exact"
            )
        if self.prior_receipt is None:
            if (
                current_rows
                or self.appended_e3a_staged_selection_artifacts
                or self.appended_signed_e3a_staged_selections
                or self.appended_e2_staged_evidence_manifests
                or self.appended_signed_e2_staged_selections
                or self.appended_signed_e1_survivor_selections
                or self.appended_e4_staged_evidence_manifests
                or self.appended_signed_e4_stage_selections
                or self.appended_signed_e3b_power_prefixes
                or self.appended_signed_e5_power_and_anchor_prefixes
                or self.appended_signed_e6_power_prefixes
                or self.appended_e0_onlinespec_source_authorities
                or self.appended_signed_e0_compatibilities
                or self.appended_signed_e0_onlinespec_tuning_seals
                or self.appended_signed_e0_power_prefixes
                or self.appended_formal_stage_prefix_artifacts
                or self.appended_tts_calibration_authorities
                or self.appended_signed_tts_calibration_seals
            ):
                raise ValueError("initial registry verification may only lock protocol")
            if self.manifest.prior_registry_verification_receipt_sha256 is not None:
                raise ValueError(
                    "initial registry verification cannot claim a prior receipt"
                )
        else:
            if type(self.prior_receipt) is not FormalRegistryVerificationReceipt:
                raise TypeError(
                    "formal registry prior verification receipt is not exact"
                )
            if not current_rows:
                raise ValueError("registry extension must append a signed stage row")
            if (
                self.manifest.prior_registry_verification_receipt_sha256
                != self.prior_receipt.sha256
            ):
                raise ValueError(
                    "registry extension does not bind its exact prior receipt"
                )
        expected_controls = 1 if self.prior_receipt is None else len(current_rows)
        if len(self.control_attestations) != expected_controls:
            raise ValueError(
                "formal registry verification control coverage is not exact"
            )

    @property
    def cumulative_signed_materializations(
        self,
    ) -> tuple[SignedStageMaterializationReceipt, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_materializations
        )
        return (*prior, *self.appended_signed_materializations)

    @property
    def cumulative_signed_coverage(self) -> tuple[SignedStageCoverageReceipt, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_coverage
        )
        return (*prior, *self.appended_signed_coverage)

    @property
    def cumulative_tts_calibration_authorities(
        self,
    ) -> tuple[TtsCalibrationAuthority, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_tts_calibration_authorities
        )
        return (*prior, *self.appended_tts_calibration_authorities)

    @property
    def cumulative_e3a_staged_selection_artifacts(
        self,
    ) -> tuple[E3aStagedSelectionArtifact, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_e3a_staged_selection_artifacts
        )
        return (*prior, *self.appended_e3a_staged_selection_artifacts)

    @property
    def cumulative_signed_e3a_staged_selections(
        self,
    ) -> tuple[SignedE3aStagedSelectionReceipt, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_e3a_staged_selections
        )
        return (*prior, *self.appended_signed_e3a_staged_selections)

    @property
    def cumulative_e2_staged_evidence_manifests(
        self,
    ) -> tuple[E2StagedRoundEvidenceManifest, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_e2_staged_evidence_manifests
        )
        return (*prior, *self.appended_e2_staged_evidence_manifests)

    @property
    def cumulative_signed_e2_staged_selections(
        self,
    ) -> tuple[SignedE2StagedRoundSelectionReceipt, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_e2_staged_selections
        )
        return (*prior, *self.appended_signed_e2_staged_selections)

    @property
    def cumulative_signed_e1_survivor_selections(
        self,
    ) -> tuple[SignedE1SurvivorSelectionReceipt, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_e1_survivor_selections
        )
        return (*prior, *self.appended_signed_e1_survivor_selections)

    @property
    def cumulative_e4_staged_evidence_manifests(
        self,
    ) -> tuple[E4StagedEvidenceManifest, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_e4_staged_evidence_manifests
        )
        return (*prior, *self.appended_e4_staged_evidence_manifests)

    @property
    def cumulative_signed_e4_stage_selections(
        self,
    ) -> tuple[SignedE4StageSelectionReceipt, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_e4_stage_selections
        )
        return (*prior, *self.appended_signed_e4_stage_selections)

    @property
    def cumulative_signed_e3b_power_prefixes(
        self,
    ) -> tuple[SignedE3bPowerPrefixReceipt, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_e3b_power_prefixes
        )
        return (*prior, *self.appended_signed_e3b_power_prefixes)

    @property
    def cumulative_signed_e5_power_and_anchor_prefixes(
        self,
    ) -> tuple[SignedE5PowerAndAnchorReceipt, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_e5_power_and_anchor_prefixes
        )
        return (*prior, *self.appended_signed_e5_power_and_anchor_prefixes)

    @property
    def cumulative_signed_e6_power_prefixes(
        self,
    ) -> tuple[SignedE6PowerPrefixReceipt, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_e6_power_prefixes
        )
        return (*prior, *self.appended_signed_e6_power_prefixes)

    @property
    def cumulative_e0_onlinespec_source_authorities(
        self,
    ) -> tuple[E0OnlineSpecSourceAuthority, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_e0_onlinespec_source_authorities
        )
        return (*prior, *self.appended_e0_onlinespec_source_authorities)

    @property
    def cumulative_signed_e0_compatibilities(
        self,
    ) -> tuple[SignedE0CompatibilityReceipt, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_e0_compatibilities
        )
        return (*prior, *self.appended_signed_e0_compatibilities)

    @property
    def cumulative_signed_e0_onlinespec_tuning_seals(
        self,
    ) -> tuple[SignedE0OnlineSpecTuningSeal, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_e0_onlinespec_tuning_seals
        )
        return (*prior, *self.appended_signed_e0_onlinespec_tuning_seals)

    @property
    def cumulative_signed_e0_power_prefixes(
        self,
    ) -> tuple[SignedE0PowerPrefixReceipt, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_e0_power_prefixes
        )
        return (*prior, *self.appended_signed_e0_power_prefixes)

    @property
    def cumulative_formal_stage_prefix_artifacts(
        self,
    ) -> tuple[CanonicalJsonProofBinding, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_formal_stage_prefix_artifacts
        )
        return (*prior, *self.appended_formal_stage_prefix_artifacts)

    @property
    def cumulative_signed_tts_calibration_seals(
        self,
    ) -> tuple[SignedTtsCalibrationSeal, ...]:
        prior = (
            ()
            if self.prior_receipt is None
            else self.prior_receipt.cumulative_signed_tts_calibration_seals
        )
        return (*prior, *self.appended_signed_tts_calibration_seals)

    @property
    def verification_ns_by_signed_sha256(self) -> dict[str, int]:
        result = (
            {self.signed_protocol_lock.sha256: self.verified_ns}
            if self.prior_receipt is None
            else dict(self.prior_receipt.verification_ns_by_signed_sha256)
        )
        for row in (
            *self.appended_signed_materializations,
            *self.appended_signed_coverage,
            *self.appended_signed_e3a_staged_selections,
            *self.appended_signed_e2_staged_selections,
            *self.appended_signed_e1_survivor_selections,
            *self.appended_signed_e4_stage_selections,
            *self.appended_signed_e3b_power_prefixes,
            *self.appended_signed_e5_power_and_anchor_prefixes,
            *self.appended_signed_e6_power_prefixes,
            *self.appended_signed_tts_calibration_seals,
            *self.appended_signed_e0_compatibilities,
            *self.appended_signed_e0_onlinespec_tuning_seals,
            *self.appended_signed_e0_power_prefixes,
        ):
            if row.sha256 in result:
                raise ValueError("formal registry verification repeats a signed row")
            result[row.sha256] = self.verified_ns
        return result

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def revalidate(self, *, current_ns: int) -> FormalRegistryManifest:
        """Deep-reopen the prefix and its append-only replay reservation chain."""

        self.__post_init__()
        if type(current_ns) is not int or current_ns < self.verified_ns:
            raise ValueError("formal registry revalidation precedes verification")
        if self.prior_receipt is not None:
            prior_manifest = self.prior_receipt.revalidate(current_ns=current_ns)
            if (
                self.signed_protocol_lock != self.prior_receipt.signed_protocol_lock
                or self.inventory_sha256 != self.prior_receipt.inventory_sha256
                or prior_manifest.trusted_attester_policy_sha256
                != self.manifest.trusted_attester_policy_sha256
            ):
                raise ValueError("formal registry extension changes its immutable root")
            _verify_formal_stage_prefix_append(
                prior_receipt=self.prior_receipt,
                prefix_bindings=self.appended_formal_stage_prefix_artifacts,
                appended_signed_materializations=(
                    self.appended_signed_materializations
                ),
                appended_signed_coverage=self.appended_signed_coverage,
                appended_e2_staged_evidence_manifests=(
                    self.appended_e2_staged_evidence_manifests
                ),
                appended_signed_e2_staged_selections=(
                    self.appended_signed_e2_staged_selections
                ),
                appended_signed_e1_survivor_selections=(
                    self.appended_signed_e1_survivor_selections
                ),
                appended_e4_staged_evidence_manifests=(
                    self.appended_e4_staged_evidence_manifests
                ),
                appended_signed_e4_stage_selections=(
                    self.appended_signed_e4_stage_selections
                ),
                now_ns=self.verified_ns,
            )
        reserved_challenges = self.reservation.revalidate()
        current_rows = (
            (self.signed_protocol_lock,)
            if self.prior_receipt is None
            else (
                *self.appended_signed_materializations,
                *self.appended_signed_coverage,
            )
        )
        prepared = _prepare_formal_registry_manifest(
            self.signed_protocol_lock,
            signed_materializations=self.cumulative_signed_materializations,
            signed_coverage=self.cumulative_signed_coverage,
            e3a_staged_selection_artifacts=(
                self.cumulative_e3a_staged_selection_artifacts
            ),
            signed_e3a_staged_selections=(self.cumulative_signed_e3a_staged_selections),
            e2_staged_evidence_manifests=(self.cumulative_e2_staged_evidence_manifests),
            signed_e2_staged_selections=(self.cumulative_signed_e2_staged_selections),
            signed_e1_survivor_selections=(
                self.cumulative_signed_e1_survivor_selections
            ),
            e4_staged_evidence_manifests=(self.cumulative_e4_staged_evidence_manifests),
            signed_e4_stage_selections=(self.cumulative_signed_e4_stage_selections),
            signed_e3b_power_prefixes=(self.cumulative_signed_e3b_power_prefixes),
            signed_e5_power_and_anchor_prefixes=(
                self.cumulative_signed_e5_power_and_anchor_prefixes
            ),
            signed_e6_power_prefixes=(self.cumulative_signed_e6_power_prefixes),
            e0_onlinespec_source_authorities=(
                self.cumulative_e0_onlinespec_source_authorities
            ),
            signed_e0_compatibilities=(self.cumulative_signed_e0_compatibilities),
            signed_e0_onlinespec_tuning_seals=(
                self.cumulative_signed_e0_onlinespec_tuning_seals
            ),
            signed_e0_power_prefixes=(self.cumulative_signed_e0_power_prefixes),
            tts_calibration_authorities=(self.cumulative_tts_calibration_authorities),
            signed_tts_calibration_seals=(self.cumulative_signed_tts_calibration_seals),
            control_attestations=self.control_attestations,
            candidate_replay_proof_artifact_paths=tuple(
                row.proof_artifact.absolute_path
                for row in self.manifest.candidate_replay_proofs
            ),
            controlled_signed_row_sha256s=frozenset(row.sha256 for row in current_rows),
            controlled_signed_source_authority_sha256s=frozenset(
                row.sha256
                for row in (
                    *self.appended_signed_e3a_staged_selections,
                    *self.appended_signed_e2_staged_selections,
                    *self.appended_signed_e1_survivor_selections,
                    *self.appended_signed_e4_stage_selections,
                    *self.appended_signed_e3b_power_prefixes,
                    *self.appended_signed_e5_power_and_anchor_prefixes,
                    *self.appended_signed_e6_power_prefixes,
                    *self.appended_signed_tts_calibration_seals,
                    *self.appended_signed_e0_compatibilities,
                    *self.appended_signed_e0_onlinespec_tuning_seals,
                    *self.appended_signed_e0_power_prefixes,
                )
            ),
            prior_registry_verification_receipt_sha256=(
                None if self.prior_receipt is None else self.prior_receipt.sha256
            ),
            verification_ns_by_signed_sha256=(self.verification_ns_by_signed_sha256),
            expected_inventory_sha256=self.inventory_sha256,
            now_ns=self.verified_ns,
        )
        verified = tuple(
            verify_release_control_artifact_attestation(
                control,
                expected_inventory_sha256=self.inventory_sha256,
                now_ns=self.verified_ns,
                consumed_challenge_sha256s=(),
            )
            for control in prepared.ordered_controls
        )
        expected_reservation_sha256 = control_challenge_reservation_sha256(
            verified,
            additional_challenge_sha256s=(prepared.additional_challenge_sha256s),
            reserved_ns=self.verified_ns,
        )
        if (
            expected_reservation_sha256 != self.reservation.reservation_sha256
            or self.manifest != prepared.manifest
            or set(prepared.additional_challenge_sha256s) - set(reserved_challenges)
        ):
            raise ValueError("formal registry durable verification receipt changed")
        return self.manifest

    def trusted_release_policy(self, *, current_ns: int) -> TrustedAttesterPolicy:
        """Return only the root-authorized policy sealed by this durable receipt."""

        manifest = self.revalidate(current_ns=current_ns)
        if not self.control_attestations:
            raise ValueError("formal registry receipt has no root-authorized control")
        control = self.control_attestations[0]
        authorization = control.deployment_policy_authorization
        policy = authorization.bundle.trusted_attester_policy
        if (
            authorization.root_manifest_sha256
            != self.signed_protocol_lock.payload.offline_release_trust_root_sha256
            or policy.sha256 != manifest.trusted_attester_policy_sha256
            or control.trusted_attester_policy_sha256 != policy.sha256
        ):
            raise ValueError("formal registry signing policy differs from release root")
        return policy

    def to_dict(self) -> dict[str, object]:
        """Serialize public verification material; no private key is present."""

        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "verified_ns": self.verified_ns,
            "inventory_sha256": self.inventory_sha256,
            "signed_protocol_lock": signed_protocol_lock_to_dict(
                self.signed_protocol_lock
            ),
            "prior_receipt": (
                None if self.prior_receipt is None else self.prior_receipt.to_dict()
            ),
            "appended_signed_materializations": [
                signed_stage_materialization_to_dict(row)
                for row in self.appended_signed_materializations
            ],
            "appended_signed_coverage": [
                signed_stage_coverage_to_dict(row)
                for row in self.appended_signed_coverage
            ],
            "appended_e3a_staged_selection_artifacts": [
                e3a_staged_selection_artifact_to_dict(row)
                for row in self.appended_e3a_staged_selection_artifacts
            ],
            "appended_signed_e3a_staged_selections": [
                signed_e3a_staged_selection_to_dict(row)
                for row in self.appended_signed_e3a_staged_selections
            ],
            "appended_e2_staged_evidence_manifests": [
                e2_staged_evidence_manifest_to_dict(row)
                for row in self.appended_e2_staged_evidence_manifests
            ],
            "appended_signed_e2_staged_selections": [
                signed_e2_staged_selection_to_dict(row)
                for row in self.appended_signed_e2_staged_selections
            ],
            "appended_signed_e1_survivor_selections": [
                signed_e1_survivor_selection_to_dict(row)
                for row in self.appended_signed_e1_survivor_selections
            ],
            "appended_e4_staged_evidence_manifests": [
                e4_staged_evidence_manifest_to_dict(row)
                for row in self.appended_e4_staged_evidence_manifests
            ],
            "appended_signed_e4_stage_selections": [
                signed_e4_stage_selection_to_dict(row)
                for row in self.appended_signed_e4_stage_selections
            ],
            "appended_signed_e3b_power_prefixes": [
                signed_e3b_power_prefix_to_dict(row)
                for row in self.appended_signed_e3b_power_prefixes
            ],
            "appended_signed_e5_power_and_anchor_prefixes": [
                signed_e5_power_and_anchor_to_dict(row)
                for row in self.appended_signed_e5_power_and_anchor_prefixes
            ],
            "appended_signed_e6_power_prefixes": [
                signed_e6_power_prefix_to_dict(row)
                for row in self.appended_signed_e6_power_prefixes
            ],
            "appended_e0_onlinespec_source_authorities": [
                e0_onlinespec_source_authority_to_dict(row)
                for row in self.appended_e0_onlinespec_source_authorities
            ],
            "appended_signed_e0_compatibilities": [
                signed_e0_compatibility_to_dict(row)
                for row in self.appended_signed_e0_compatibilities
            ],
            "appended_signed_e0_onlinespec_tuning_seals": [
                signed_e0_onlinespec_tuning_seal_to_dict(row)
                for row in self.appended_signed_e0_onlinespec_tuning_seals
            ],
            "appended_signed_e0_power_prefixes": [
                signed_e0_power_prefix_to_dict(row)
                for row in self.appended_signed_e0_power_prefixes
            ],
            "appended_formal_stage_prefix_artifacts": [
                row.to_dict() for row in self.appended_formal_stage_prefix_artifacts
            ],
            "appended_tts_calibration_authorities": [
                tts_calibration_authority_to_dict(row)
                for row in self.appended_tts_calibration_authorities
            ],
            "appended_signed_tts_calibration_seals": [
                signed_tts_calibration_seal_to_dict(row)
                for row in self.appended_signed_tts_calibration_seals
            ],
            "control_attestations": [
                row.to_dict() for row in self.control_attestations
            ],
            "reservation": self.reservation.to_dict(),
            "manifest": self.manifest.to_dict(),
            "receipt_sha256": self.sha256,
        }
        normalized = _json_tree(value)
        assert type(normalized) is dict
        return normalized


def formal_registry_verification_receipt_from_dict(
    value: object,
) -> FormalRegistryVerificationReceipt:
    """Decode the append-only receipt and verify every declared nested digest."""

    row = _strict(
        "formal registry verification receipt",
        value,
        frozenset(
            (*FormalRegistryVerificationReceipt.__dataclass_fields__, "receipt_sha256")
        ),
    )
    declared_sha256 = _require_sha256_text(
        "formal registry verification receipt digest",
        row.pop("receipt_sha256"),
    )
    prior_value = row.pop("prior_receipt")
    prior_receipt = (
        None
        if prior_value is None
        else formal_registry_verification_receipt_from_dict(prior_value)
    )
    receipt = FormalRegistryVerificationReceipt(
        schema_version=row["schema_version"],
        kind=row["kind"],
        verified_ns=row["verified_ns"],
        inventory_sha256=row["inventory_sha256"],
        signed_protocol_lock=signed_protocol_lock_from_dict(
            row["signed_protocol_lock"]
        ),
        prior_receipt=prior_receipt,
        appended_signed_materializations=tuple(
            signed_stage_materialization_from_dict(item)
            for item in _array(
                "appended signed materializations",
                row["appended_signed_materializations"],
            )
        ),
        appended_signed_coverage=tuple(
            signed_stage_coverage_from_dict(item)
            for item in _array(
                "appended signed coverage", row["appended_signed_coverage"]
            )
        ),
        appended_e3a_staged_selection_artifacts=tuple(
            e3a_staged_selection_artifact_from_dict(item)
            for item in _array(
                "appended E3a staged selection artifacts",
                row["appended_e3a_staged_selection_artifacts"],
            )
        ),
        appended_signed_e3a_staged_selections=tuple(
            signed_e3a_staged_selection_from_dict(item)
            for item in _array(
                "appended signed E3a staged selections",
                row["appended_signed_e3a_staged_selections"],
            )
        ),
        appended_e2_staged_evidence_manifests=tuple(
            e2_staged_evidence_manifest_from_dict(item)
            for item in _array(
                "appended E2 staged evidence manifests",
                row["appended_e2_staged_evidence_manifests"],
            )
        ),
        appended_signed_e2_staged_selections=tuple(
            signed_e2_staged_selection_from_dict(item)
            for item in _array(
                "appended signed E2 staged selections",
                row["appended_signed_e2_staged_selections"],
            )
        ),
        appended_signed_e1_survivor_selections=tuple(
            signed_e1_survivor_selection_from_dict(item)
            for item in _array(
                "appended signed E1 survivor selections",
                row["appended_signed_e1_survivor_selections"],
            )
        ),
        appended_e4_staged_evidence_manifests=tuple(
            e4_staged_evidence_manifest_from_dict(item)
            for item in _array(
                "appended E4 staged evidence manifests",
                row["appended_e4_staged_evidence_manifests"],
            )
        ),
        appended_signed_e4_stage_selections=tuple(
            signed_e4_stage_selection_from_dict(item)
            for item in _array(
                "appended signed E4 staged selections",
                row["appended_signed_e4_stage_selections"],
            )
        ),
        appended_signed_e3b_power_prefixes=tuple(
            signed_e3b_power_prefix_from_dict(item)
            for item in _array(
                "appended signed E3b power prefixes",
                row["appended_signed_e3b_power_prefixes"],
            )
        ),
        appended_signed_e5_power_and_anchor_prefixes=tuple(
            signed_e5_power_and_anchor_from_dict(item)
            for item in _array(
                "appended signed E5 power/anchor prefixes",
                row["appended_signed_e5_power_and_anchor_prefixes"],
            )
        ),
        appended_signed_e6_power_prefixes=tuple(
            signed_e6_power_prefix_from_dict(item)
            for item in _array(
                "appended signed E6 power prefixes",
                row["appended_signed_e6_power_prefixes"],
            )
        ),
        appended_e0_onlinespec_source_authorities=tuple(
            e0_onlinespec_source_authority_from_dict(item)
            for item in _array(
                "appended E0 OnlineSPEC source authorities",
                row["appended_e0_onlinespec_source_authorities"],
            )
        ),
        appended_signed_e0_compatibilities=tuple(
            signed_e0_compatibility_from_dict(item)
            for item in _array(
                "appended signed E0 compatibilities",
                row["appended_signed_e0_compatibilities"],
            )
        ),
        appended_signed_e0_onlinespec_tuning_seals=tuple(
            signed_e0_onlinespec_tuning_seal_from_dict(item)
            for item in _array(
                "appended signed E0 OnlineSPEC tuning seals",
                row["appended_signed_e0_onlinespec_tuning_seals"],
            )
        ),
        appended_signed_e0_power_prefixes=tuple(
            signed_e0_power_prefix_from_dict(item)
            for item in _array(
                "appended signed E0 power prefixes",
                row["appended_signed_e0_power_prefixes"],
            )
        ),
        appended_formal_stage_prefix_artifacts=tuple(
            CanonicalJsonProofBinding.from_dict(item)
            for item in _array(
                "appended formal stage prefix artifacts",
                row["appended_formal_stage_prefix_artifacts"],
            )
        ),
        appended_tts_calibration_authorities=tuple(
            tts_calibration_authority_from_dict(item)
            for item in _array(
                "appended TTS calibration authorities",
                row["appended_tts_calibration_authorities"],
            )
        ),
        appended_signed_tts_calibration_seals=tuple(
            signed_tts_calibration_seal_from_dict(item)
            for item in _array(
                "appended signed TTS calibration seals",
                row["appended_signed_tts_calibration_seals"],
            )
        ),
        control_attestations=tuple(
            ControlArtifactAttestation.from_dict(item)
            for item in _array(
                "formal registry control attestations",
                row["control_attestations"],
            )
        ),
        reservation=ChallengeReplayReservationBinding.from_dict(row["reservation"]),
        manifest=formal_registry_manifest_from_dict(row["manifest"]),
    )
    if receipt.sha256 != declared_sha256:
        raise ValueError(
            "formal registry verification receipt digest differs from content"
        )
    return receipt


def formal_registry_verification_receipt_to_dict(
    value: FormalRegistryVerificationReceipt,
) -> dict[str, object]:
    if type(value) is not FormalRegistryVerificationReceipt:
        raise TypeError("formal registry receipt codec requires an exact receipt")
    return value.to_dict()


def _e2_round(receipt: StageMaterializationReceipt) -> int:
    rounds = {dict(cell.dimensions).get("round") for cell in receipt.cells}
    if len(rounds) != 1:
        raise ValueError("E2 materialization does not bind one exact round")
    round_index = next(iter(rounds))
    if type(round_index) is not int or round_index not in range(4):
        raise ValueError("E2 materialization round is outside [0, 4)")
    expected_rule = (
        "e2_round_0_105_per_geometry_plus_four_anchors"
        if round_index == 0
        else "e2_quarter_retention_floor_21_plus_four_anchors"
    )
    if receipt.materialization_rule != expected_rule:
        raise ValueError("E2 materialization rule differs from its round")
    return round_index


_E4_RULE_ORDER = (
    "strength2_8_rows_x_3_loads_x_2_traffic",
    "winner_neighborhood_2pow4_x_3_loads_x_2_traffic",
    "three_profiler_only_rows_separate_from_headline",
)

_TUNING_ONLY_PILOT_MATERIALIZATION_RULES = frozenset(
    {
        "e3b_exact_480_rows_x_4_excluded_pilot_blocks",
        "e5_exact_450_headline_rows_x_4_excluded_pilot_blocks",
        ("e6_exact_two_model_preflights_plus_60_rows_x_4_excluded_pilot_blocks"),
        "e0_full_registered_onlinespec_grid_per_valid_combination_tuning_only",
        "e0_exact_16_rows_per_valid_combination_x_4_excluded_pilot_blocks",
    }
)


def _validate_downstream_main_materialization(
    receipt: StageMaterializationReceipt,
) -> None:
    """Reject pilot overlap and cardinality drift in the main DAG receipts."""

    if receipt.stage not in {"E3b", "E5", "E6", "E0"}:
        return
    if receipt.stage == "E3b":
        expected_rule = (
            "five_roles_x_8_contexts_x_3_regimes_x_2_loads_x_2_widths_final_only"
        )
        block_cells = receipt.cells
        per_block = 480
        fixed_rows = 0
        source_dimension = "signed_power_prefix_sha256"
    elif receipt.stage == "E5":
        expected_rule = (
            "450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics"
        )
        block_cells = tuple(
            cell for cell in receipt.cells if cell.task == "production_slo_power_prefix"
        )
        failures = tuple(
            cell
            for cell in receipt.cells
            if cell.task == "deterministic_failure_injection"
        )
        if (
            len(failures) != 264
            or len(block_cells) + len(failures) != len(receipt.cells)
            or any("block" in dict(cell.dimensions) for cell in failures)
        ):
            raise ValueError("E5 main materialization failure rows are not exact")
        per_block = 450
        fixed_rows = 264
        source_dimension = "signed_power_and_anchor_prefix_sha256"
    elif receipt.stage == "E6":
        expected_rule = "60_final_rows_per_block_reusing_global_model_preflights"
        preflights = tuple(
            cell
            for cell in receipt.cells
            if cell.task == "immutable_metadata_interface_and_fit_preflight"
        )
        block_cells = tuple(
            cell
            for cell in receipt.cells
            if cell.task != "immutable_metadata_interface_and_fit_preflight"
        )
        if preflights or len(block_cells) != len(receipt.cells):
            raise ValueError("E6 final must not repeat global model preflights")
        per_block = 60
        fixed_rows = 0
        source_dimension = "signed_power_prefix_sha256"
    else:
        expected_rule = (
            "valid_compatibilities_x_8_roles_x_2_loads_x_final_only_powered_prefix"
        )
        block_cells = receipt.cells
        decision_ids = {
            dict(cell.dimensions).get("compatibility_decision_id")
            for cell in block_cells
        }
        if (
            not decision_ids
            or None in decision_ids
            or any(type(value) is not str for value in decision_ids)
        ):
            raise ValueError("E0 main materialization has no exact VALID decisions")
        per_block = 16 * len(decision_ids)
        fixed_rows = 0
        source_dimension = "signed_power_prefix_sha256"
    if receipt.materialization_rule != expected_rule:
        raise ValueError(f"{receipt.stage} main materialization rule is not exact")
    lineage = tuple(dict(cell.dimensions) for cell in receipt.cells)
    source_values = {row.get(source_dimension) for row in lineage}
    pilot_materializations = {
        row.get("pilot_materialization_receipt_sha256") for row in lineage
    }
    pilot_coverages = {row.get("pilot_coverage_receipt_sha256") for row in lineage}
    if (
        source_values != {receipt.source_decision_sha256}
        or len(pilot_materializations) != 1
        or None in pilot_materializations
        or len(pilot_coverages) != 1
        or None in pilot_coverages
    ):
        raise ValueError(
            f"{receipt.stage} main materialization lacks typed pilot/power lineage"
        )
    _require_sha256_text(
        f"{receipt.stage} pilot materialization",
        next(iter(pilot_materializations)),
    )
    _require_sha256_text(f"{receipt.stage} pilot coverage", next(iter(pilot_coverages)))
    if receipt.stage == "E0":
        compatibility_values = {
            row.get("signed_e0_compatibility_sha256") for row in lineage
        }
        tuning_set_values = {
            row.get("signed_e0_tuning_seal_set_sha256") for row in lineage
        }
        e6_values = {row.get("signed_e6_confirmation_sha256") for row in lineage}
        if any(
            len(values) != 1 or None in values
            for values in (
                compatibility_values,
                tuning_set_values,
                e6_values,
            )
        ):
            raise ValueError(
                "E0 main materialization typed source lineage is ambiguous"
            )
        for label, values in (
            ("compatibility", compatibility_values),
            ("tuning seal set", tuning_set_values),
            ("E6 confirmation", e6_values),
        ):
            _require_sha256_text(f"E0 {label}", next(iter(values)))
        expected_members = {
            (block, decision_id, role, load)
            for block in {dict(cell.dimensions).get("block") for cell in block_cells}
            for decision_id in decision_ids
            for role in E0_METHOD_ROLES
            for load in E0_LOADS
        }
        observed_members = {
            (
                dict(cell.dimensions).get("block"),
                dict(cell.dimensions).get("compatibility_decision_id"),
                cell.method_role,
                dict(cell.dimensions).get("load"),
            )
            for cell in block_cells
        }
        if observed_members != expected_members or len(block_cells) != len(
            expected_members
        ):
            raise ValueError("E0 main matrix is not exact 8-role/two-load coverage")
    block_values = {dict(cell.dimensions).get("block") for cell in block_cells}
    if (
        any(type(block) is not int for block in block_values)
        or not 12 <= len(block_values) <= 20
        or block_values != set(range(4, 4 + len(block_values)))
        or len(block_cells) != per_block * len(block_values)
        or len(receipt.cells) != per_block * len(block_values) + fixed_rows
        or any(
            dict(cell.dimensions).get("block_phase") != "final" for cell in block_cells
        )
    ):
        raise ValueError(
            f"{receipt.stage} main materialization contains pilot or non-prefix rows"
        )


def _ordered_materializations(
    receipts: tuple[StageMaterializationReceipt, ...],
) -> tuple[StageMaterializationReceipt, ...]:
    grouped = {stage: [] for stage in FORMAL_STAGE_DAG}
    for receipt in receipts:
        if receipt.materialization_rule in _TUNING_ONLY_PILOT_MATERIALIZATION_RULES:
            raise ValueError(
                "excluded-pilot tuning materialization cannot enter the formal "
                "stage registry"
            )
        _validate_downstream_main_materialization(receipt)
        grouped[receipt.stage].append(receipt)
    for stage in FORMAL_STAGE_DAG:
        limit = 4 if stage == "E2" else 3 if stage == "E4" else 1
        if len(grouped[stage]) > limit:
            raise ValueError(f"formal registry repeats too many {stage} receipts")

    ordered: list[StageMaterializationReceipt] = []
    gap_seen = False
    for stage in FORMAL_STAGE_DAG:
        rows = grouped[stage]
        if not rows:
            gap_seen = True
            continue
        if gap_seen:
            raise ValueError("formal materializations must form a DAG prefix")
        if stage == "E2":
            rows = sorted(rows, key=_e2_round)
            rounds = tuple(_e2_round(row) for row in rows)
            if rounds != tuple(range(len(rows))):
                raise ValueError("E2 materializations must be the exact round prefix")
        elif stage == "E4":
            by_rule = {row.materialization_rule: row for row in rows}
            if len(by_rule) != len(rows) or any(
                rule not in _E4_RULE_ORDER for rule in by_rule
            ):
                raise ValueError("E4 materializations have unknown or duplicate phases")
            phases = tuple(rule for rule in _E4_RULE_ORDER if rule in by_rule)
            if phases != _E4_RULE_ORDER[: len(rows)]:
                raise ValueError(
                    "E4 materializations must be screen-local-profile prefix"
                )
            rows = [by_rule[rule] for rule in phases]
        ordered.extend(rows)

    for index, receipt in enumerate(ordered):
        expected_upstream = () if index == 0 else (ordered[index - 1].sha256,)
        if receipt.stage in {"E3a", "TTS-Cal"}:
            if len(receipt.upstream_receipt_sha256s) != 1:
                raise ValueError(
                    f"{receipt.stage} must bind one exact signed upstream authority"
                )
            continue
        if receipt.stage == "E1":
            if (
                len(receipt.upstream_receipt_sha256s) != 3
                or receipt.upstream_receipt_sha256s[2] != receipt.source_decision_sha256
                or receipt.upstream_receipt_sha256s[1]
                in {
                    receipt.upstream_receipt_sha256s[0],
                    receipt.source_decision_sha256,
                }
            ):
                raise ValueError(
                    "E1 must bind TTS-Cal, its exact signed seal, then E3a selection"
                )
            expected_upstream = receipt.upstream_receipt_sha256s
        if receipt.upstream_receipt_sha256s != expected_upstream:
            raise ValueError("formal materialization upstream linkage is not exact")

    present = {row.stage for row in ordered}
    if (
        any(stage in present for stage in FORMAL_STAGE_DAG[5:])
        and len(grouped["E2"]) != 4
    ):
        raise ValueError("downstream materialization requires all four E2 rounds")
    if (
        any(stage in present for stage in FORMAL_STAGE_DAG[6:])
        and len(grouped["E4"]) != 3
    ):
        raise ValueError("downstream materialization requires all three E4 phases")
    return tuple(ordered)


def _validate_candidate_coverage_replay_uniqueness(
    coverages: tuple[StageCoverageReceipt, ...],
) -> None:
    """Reject reuse of live candidate evidence across signed stage receipts.

    E2 has four independently materialized rounds.  Per-receipt validation is
    therefore insufficient: a signer must not be able to attach a round-zero
    terminal, run, pair, or proposal identity to a later round (or to another
    formal stage) and have both receipts appear complete in one manifest.
    """

    pair_coverages = tuple(
        pair
        for coverage in coverages
        for pair in coverage.tts_l0_candidate_state_coverages
    )
    pair_ids = tuple(pair.pair_id for pair in pair_coverages)
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("formal coverage reuses a TTS/L0 matched-pair identity")
    run_ids = tuple(
        next(iter({observation.run_id for observation in observations}))
        for pair in pair_coverages
        for observations in (pair.tts_observations, pair.l0_naive_observations)
    )
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("formal coverage reuses a candidate-state run identity")
    terminal_receipts = tuple(
        receipt_sha256
        for pair in pair_coverages
        for receipt_sha256 in (
            pair.terminal_pairs[0].tts_terminal_receipt_sha256,
            pair.terminal_pairs[0].l0_naive_terminal_receipt_sha256,
        )
    )
    if len(terminal_receipts) != len(set(terminal_receipts)):
        raise ValueError("formal coverage reuses a candidate terminal receipt")
    proposal_evidence = tuple(
        observation.proposal_evidence_sha256
        for pair in pair_coverages
        for observation in pair.tts_observations
    )
    if len(proposal_evidence) != len(set(proposal_evidence)):
        raise ValueError("formal coverage reuses candidate proposal evidence")
    replay_pointers = tuple(
        pointer_sha256
        for pair in pair_coverages
        for pointer_sha256 in (
            pair.tts_native_replay_pointer_sha256,
            pair.l0_naive_native_replay_pointer_sha256,
        )
    )
    if len(replay_pointers) != len(set(replay_pointers)):
        raise ValueError("formal coverage reuses a native replay pointer")


def _build_formal_registry_manifest_with_policy(
    signed_protocol_lock: SignedProtocolLock,
    *,
    signed_materializations: tuple[SignedStageMaterializationReceipt, ...],
    signed_coverage: tuple[SignedStageCoverageReceipt, ...],
    e3a_staged_selection_artifacts: tuple[E3aStagedSelectionArtifact, ...] = (),
    signed_e3a_staged_selections: tuple[SignedE3aStagedSelectionReceipt, ...] = (),
    e2_staged_evidence_manifests: tuple[E2StagedRoundEvidenceManifest, ...] = (),
    signed_e2_staged_selections: tuple[SignedE2StagedRoundSelectionReceipt, ...] = (),
    signed_e1_survivor_selections: tuple[SignedE1SurvivorSelectionReceipt, ...] = (),
    e4_staged_evidence_manifests: tuple[E4StagedEvidenceManifest, ...] = (),
    signed_e4_stage_selections: tuple[SignedE4StageSelectionReceipt, ...] = (),
    signed_e3b_power_prefixes: tuple[SignedE3bPowerPrefixReceipt, ...] = (),
    signed_e5_power_and_anchor_prefixes: tuple[SignedE5PowerAndAnchorReceipt, ...] = (),
    signed_e6_power_prefixes: tuple[SignedE6PowerPrefixReceipt, ...] = (),
    e0_onlinespec_source_authorities: tuple[E0OnlineSpecSourceAuthority, ...] = (),
    signed_e0_compatibilities: tuple[SignedE0CompatibilityReceipt, ...] = (),
    signed_e0_onlinespec_tuning_seals: tuple[SignedE0OnlineSpecTuningSeal, ...] = (),
    signed_e0_power_prefixes: tuple[SignedE0PowerPrefixReceipt, ...] = (),
    tts_calibration_authorities: tuple[TtsCalibrationAuthority, ...] = (),
    signed_tts_calibration_seals: tuple[SignedTtsCalibrationSeal, ...] = (),
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    inventory_sha256: str,
    deployment_policy_authorization_sha256: str,
    control_lineage_sha256: str,
    control_envelope_sha256s: tuple[str, ...],
    challenge_reservation_sha256: str,
    candidate_replay_proofs: tuple[FormalCandidateReplayProofBinding, ...] = (),
    prior_registry_verification_receipt_sha256: str | None = None,
    verification_ns_by_signed_sha256: dict[str, int] | None = None,
    now_ns: int | None = None,
) -> FormalRegistryManifest:
    typed_signed_rows = (
        (signed_protocol_lock, "dispatch"),
        *((row, "dispatch") for row in signed_materializations),
        *((row, "rank_aggregate") for row in signed_coverage),
    )
    signed_source_rows = (
        *signed_e3a_staged_selections,
        *signed_e2_staged_selections,
        *signed_e1_survivor_selections,
        *signed_e4_stage_selections,
        *signed_e3b_power_prefixes,
        *signed_e5_power_and_anchor_prefixes,
        *signed_e6_power_prefixes,
        *signed_tts_calibration_seals,
        *signed_e0_compatibilities,
        *signed_e0_onlinespec_tuning_seals,
        *signed_e0_power_prefixes,
    )
    new_challenges = (
        *(row.challenge.sha256 for row, _ in typed_signed_rows),
        *(row.challenge.sha256 for row in signed_source_rows),
    )
    if len(new_challenges) != len(set(new_challenges)):
        raise ValueError("formal signing challenge is duplicated")
    verification_times = (
        {}
        if verification_ns_by_signed_sha256 is None
        else dict(verification_ns_by_signed_sha256)
    )
    known_signed_sha256s = {
        *(row.sha256 for row, _ in typed_signed_rows),
        *(row.sha256 for row in signed_source_rows),
    }
    if set(verification_times) - known_signed_sha256s:
        raise ValueError("formal verification-time map names a foreign signed row")

    def verification_time(row: object) -> int | None:
        signed_sha256 = row.sha256  # type: ignore[attr-defined]
        return verification_times.get(signed_sha256, now_ns)

    lock = signed_protocol_lock.verify(
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=verification_time(signed_protocol_lock),
    )
    registry = build_industrial_registry()
    if lock.registry_sha256 != registry.sha256:
        raise ValueError("ProtocolLock does not bind the staged registry identity")
    materialized: dict[str, StageMaterializationReceipt] = {}
    for signed in signed_materializations:
        payload = signed.verify(
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed),
        )
        if payload.protocol_lock_sha256 != lock.sha256:
            raise ValueError("materialization belongs to another ProtocolLock")
        if payload.sha256 in materialized:
            raise ValueError("formal registry repeats a materialization receipt")
        materialized[payload.sha256] = payload
    ordered_payloads = _ordered_materializations(tuple(materialized.values()))
    signed_by_payload = {
        signed.payload.sha256: signed for signed in signed_materializations
    }

    def pilot_lineage(
        payload: StageMaterializationReceipt,
    ) -> tuple[str | None, str | None]:
        if payload.stage not in {"E3b", "E5", "E6", "E0"}:
            return None, None
        dimensions = tuple(dict(cell.dimensions) for cell in payload.cells)
        pilot_materializations = {
            row.get("pilot_materialization_receipt_sha256") for row in dimensions
        }
        pilot_coverages = {
            row.get("pilot_coverage_receipt_sha256") for row in dimensions
        }
        if (
            len(pilot_materializations) != 1
            or None in pilot_materializations
            or len(pilot_coverages) != 1
            or None in pilot_coverages
        ):
            raise ValueError("powered formal stage pilot lineage is not exact")
        pilot_materialization = next(iter(pilot_materializations))
        pilot_coverage = next(iter(pilot_coverages))
        assert type(pilot_materialization) is str
        assert type(pilot_coverage) is str
        return pilot_materialization, pilot_coverage

    materialization_bindings = []
    for payload in ordered_payloads:
        pilot_materialization, pilot_coverage = pilot_lineage(payload)
        materialization_bindings.append(
            FormalMaterializationBinding(
                stage=payload.stage,
                materialization_receipt_sha256=payload.sha256,
                signed_receipt_sha256=signed_by_payload[payload.sha256].sha256,
                source_decision_sha256=payload.source_decision_sha256,
                expected_cell_count=payload.expected_cell_count,
                materialization_rule=payload.materialization_rule,
                pilot_materialization_receipt_sha256=pilot_materialization,
                pilot_coverage_receipt_sha256=pilot_coverage,
            )
        )
    coverage_bindings = []
    coverage_payloads = []
    covered: set[str] = set()
    for signed in signed_coverage:
        materialization = materialized.get(
            signed.payload.materialization_receipt_sha256
        )
        if materialization is None:
            raise ValueError("coverage has no verified materialization in the manifest")
        payload = signed.verify(
            materialization=materialization,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed),
        )
        if payload.materialization_receipt_sha256 in covered:
            raise ValueError("formal registry repeats coverage for a materialization")
        covered.add(payload.materialization_receipt_sha256)
        coverage_payloads.append(payload)
        complete_count = sum(row.status == "COMPLETE" for row in payload.dispositions)
        terminal_complete = complete_count == len(payload.dispositions)
        coverage_bindings.append(
            FormalCoverageBinding(
                stage=payload.stage,
                materialization_receipt_sha256=(payload.materialization_receipt_sha256),
                coverage_receipt_sha256=payload.sha256,
                signed_receipt_sha256=signed.sha256,
                disposition_count=len(payload.dispositions),
                complete_cell_count=complete_count,
                terminal_complete=terminal_complete,
            )
        )
    signed_coverage_by_payload = {row.payload.sha256: row for row in signed_coverage}
    preflight_materializations = tuple(
        row for row in ordered_payloads if row.stage == "preflight"
    )
    preflight_coverages = tuple(
        row for row in coverage_payloads if row.stage == "preflight"
    )
    e3a_materializations = tuple(row for row in ordered_payloads if row.stage == "E3a")
    e3a_coverages = tuple(row for row in coverage_payloads if row.stage == "E3a")
    if e3a_materializations:
        if (
            len(e3a_materializations) != 1
            or len(preflight_materializations) != 1
            or len(preflight_coverages) != 1
        ):
            raise ValueError("E3a materialization requires exact preflight coverage")
        signed_preflight_coverage = signed_coverage_by_payload[
            preflight_coverages[0].sha256
        ]
        if (
            e3a_materializations[0].upstream_receipt_sha256s
            != (signed_preflight_coverage.sha256,)
            or e3a_materializations[0].source_decision_sha256
            != lock.formal_workload_e3a_authorization_sha256
        ):
            raise ValueError(
                "E3a materialization differs from signed preflight/workload lineage"
            )
    source_authority_bindings_list: list[FormalSourceAuthorityBinding] = []
    if e3a_coverages:
        if (
            len(e3a_materializations) != 1
            or len(e3a_coverages) != 1
            or len(e3a_staged_selection_artifacts) != 1
            or len(signed_e3a_staged_selections) != 1
        ):
            raise ValueError(
                "E3a coverage requires one exact staged selection artifact/receipt"
            )
        artifact = e3a_staged_selection_artifacts[0]
        signed_selection = signed_e3a_staged_selections[0]
        selection = signed_selection.verify(
            artifact=artifact,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed_selection),
        )
        if (
            artifact.protocol_lock_sha256 != lock.sha256
            or artifact.registry_sha256 != lock.registry_sha256
            or artifact.materialization_receipt_sha256 != e3a_materializations[0].sha256
            or artifact.coverage_receipt_sha256 != e3a_coverages[0].sha256
            or artifact.inventory_sha256 != inventory_sha256
            or selection.e3a_materialization_receipt_sha256
            != e3a_materializations[0].sha256
            or selection.e3a_coverage_receipt_sha256 != e3a_coverages[0].sha256
        ):
            raise ValueError("E3a staged selection differs from registry lineage")
        source_authority_bindings_list.append(
            FormalSourceAuthorityBinding(
                stage="E3a",
                authority_kind="e3a_staged_selection",
                signed_authority_sha256=signed_selection.sha256,
                payload_sha256=selection.sha256,
                authority_sha256=artifact.sha256,
                challenge_sha256=signed_selection.challenge.sha256,
            )
        )
        tts_cal_materializations = tuple(
            row for row in ordered_payloads if row.stage == "TTS-Cal"
        )
        if tts_cal_materializations and (
            len(tts_cal_materializations) != 1
            or tts_cal_materializations[0].upstream_receipt_sha256s
            != (signed_selection.sha256,)
            or tts_cal_materializations[0].source_decision_sha256
            != lock.tts_calibration_authority_sha256
        ):
            raise ValueError(
                "TTS-Cal materialization differs from signed E3a/TTS authority lineage"
            )
    elif e3a_staged_selection_artifacts or signed_e3a_staged_selections:
        raise ValueError("E3a selection authority cannot appear before E3a coverage")

    e1_materializations = tuple(row for row in ordered_payloads if row.stage == "E1")
    tts_cal_materializations = tuple(
        row for row in ordered_payloads if row.stage == "TTS-Cal"
    )
    tts_cal_coverages = tuple(
        row for row in coverage_payloads if row.stage == "TTS-Cal"
    )
    verified_tts_seal = None
    signed_tts_seal = None
    if tts_calibration_authorities or signed_tts_calibration_seals:
        if (
            len(tts_calibration_authorities) != 1
            or len(signed_tts_calibration_seals) != 1
            or len(tts_cal_materializations) != 1
            or len(tts_cal_coverages) != 1
        ):
            raise ValueError(
                "TTS-Cal source lineage requires exact coverage, authority, and seal"
            )
        authority = tts_calibration_authorities[0]
        signed_tts_seal = signed_tts_calibration_seals[0]
        if (
            type(authority) is not TtsCalibrationAuthority
            or authority.sha256 != lock.tts_calibration_authority_sha256
        ):
            raise ValueError("TTS-Cal authority differs from ProtocolLock")
        verified_tts_seal = signed_tts_seal.verify(
            authority=authority,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed_tts_seal),
        )
        tts_cal_materialization = tts_cal_materializations[0]
        tts_cal_coverage = tts_cal_coverages[0]
        if (
            verified_tts_seal.protocol_lock_sha256 != lock.sha256
            or verified_tts_seal.materialization_receipt_sha256
            != tts_cal_materialization.sha256
            or verified_tts_seal.coverage_receipt_sha256 != tts_cal_coverage.sha256
        ):
            raise ValueError("signed TTS seal differs from exact TTS-Cal lineage")
        source_authority_bindings_list.append(
            FormalSourceAuthorityBinding(
                stage="TTS-Cal",
                authority_kind="tts_calibration_seal",
                signed_authority_sha256=signed_tts_seal.sha256,
                payload_sha256=verified_tts_seal.sha256,
                authority_sha256=authority.sha256,
                challenge_sha256=signed_tts_seal.challenge.sha256,
            )
        )
    elif e1_materializations:
        raise ValueError("E1 cannot materialize before the signed TTS-Cal seal")

    if e1_materializations:
        if (
            len(e1_materializations) != 1
            or verified_tts_seal is None
            or signed_tts_seal is None
            or len(tts_cal_materializations) != 1
            or len(tts_cal_coverages) != 1
            or len(signed_e3a_staged_selections) != 1
        ):
            raise ValueError("E1 registry lineage lacks exact staged authorities")
        tts_cal_materialization = tts_cal_materializations[0]
        tts_cal_coverage = tts_cal_coverages[0]
        e1 = e1_materializations[0]
        frozen_anchor_recipes = {
            cell.recipe_sha256
            for cell in e1.cells
            if cell.method_role in {"TTS", "L0-naive"}
        }
        signed_tts_cal_coverage = signed_coverage_by_payload[tts_cal_coverage.sha256]
        signed_e3a_selection = signed_e3a_staged_selections[0]
        if (
            e1.upstream_receipt_sha256s
            != (
                signed_tts_cal_coverage.sha256,
                signed_tts_seal.sha256,
                signed_e3a_selection.sha256,
            )
            or e1.source_decision_sha256 != signed_e3a_selection.sha256
            or frozen_anchor_recipes != {verified_tts_seal.selected_candidate_id}
        ):
            raise ValueError("E1 materialization differs from staged source lineage")

    e2_materializations = tuple(row for row in ordered_payloads if row.stage == "E2")
    e1_coverages = tuple(row for row in coverage_payloads if row.stage == "E1")
    if e2_materializations:
        if (
            len(e1_materializations) != 1
            or len(e1_coverages) != 1
            or len(signed_e1_survivor_selections) != 1
            or len(signed_e3a_staged_selections) != 1
        ):
            raise ValueError(
                "E2 round zero requires exact E1 coverage and signed survivors"
            )
        e1_materialization = e1_materializations[0]
        e1_coverage = e1_coverages[0]
        signed_e1_selection = signed_e1_survivor_selections[0]
        e1_selection = signed_e1_selection.payload
        verify_signed_payload(
            e1_selection,
            payload_sha256=signed_e1_selection.payload_sha256,
            challenge=signed_e1_selection.challenge,
            attestation=signed_e1_selection.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed_e1_selection),
        )
        round_zero = e2_materializations[0]
        round_zero_candidates = {
            cell.recipe_sha256
            for cell in round_zero.cells
            if cell.method_role == "LightCone-candidate"
        }
        round_zero_models = {cell.model for cell in round_zero.cells}
        frozen_anchor_recipes = {
            cell.recipe_sha256
            for cell in round_zero.cells
            if cell.method_role in {"TTS", "L0-naive"}
        }
        grid = default_e2_recipe_grid_authority()
        expected_candidates = {
            row.sha256
            for row in e2_candidate_recipes(
                e1_selection.surviving_geometries,
                grid=grid,
            )
        }
        if (
            e1_selection.protocol_lock_sha256 != lock.sha256
            or e1_selection.registry_sha256 != lock.registry_sha256
            or e1_selection.e1_materialization_receipt_sha256
            != e1_materialization.sha256
            or e1_selection.e1_coverage_receipt_sha256 != e1_coverage.sha256
            or e1_selection.e3a_selection_receipt_sha256
            != signed_e3a_staged_selections[0].sha256
            or e1_selection.inventory_sha256 != inventory_sha256
            or lock.e2_recipe_grid_authority_sha256 != grid.sha256
            or round_zero.upstream_receipt_sha256s != (e1_materialization.sha256,)
            or round_zero.source_decision_sha256 != signed_e1_selection.sha256
            or round_zero_models != {e1_selection.model}
            or frozen_anchor_recipes != {e1_selection.frozen_tts_recipe_sha256}
            or round_zero_candidates != expected_candidates
            or len(round_zero_candidates)
            != 105 * len(e1_selection.surviving_geometries)
        ):
            raise ValueError("E2 round zero differs from signed E1 survivor lineage")
        source_authority_bindings_list.append(
            FormalSourceAuthorityBinding(
                stage="E1",
                authority_kind="e1_staged_pareto_survivors",
                signed_authority_sha256=signed_e1_selection.sha256,
                payload_sha256=e1_selection.sha256,
                authority_sha256=e1_selection.staged_pareto_artifact_sha256,
                challenge_sha256=signed_e1_selection.challenge.sha256,
            )
        )
    elif signed_e1_survivor_selections:
        raise ValueError("E1 survivor authority cannot appear before E2 round zero")
    e2_coverage_by_materialization = {
        row.materialization_receipt_sha256: row
        for row in coverage_payloads
        if row.stage == "E2"
    }
    expected_e2_selection_count = max(0, len(e2_materializations) - 1)
    if (
        len(e2_staged_evidence_manifests) != expected_e2_selection_count
        or len(signed_e2_staged_selections) != expected_e2_selection_count
    ):
        raise ValueError(
            "later E2 rounds require one exact staged selection per prior round"
        )
    e2_manifest_by_round = {
        row.round_index: row for row in e2_staged_evidence_manifests
    }
    e2_selection_by_round = {
        row.payload.round_index: row for row in signed_e2_staged_selections
    }
    if (
        len(e2_manifest_by_round) != len(e2_staged_evidence_manifests)
        or len(e2_selection_by_round) != len(signed_e2_staged_selections)
        or set(e2_manifest_by_round) != set(range(expected_e2_selection_count))
        or set(e2_selection_by_round) != set(range(expected_e2_selection_count))
    ):
        raise ValueError("E2 staged selections are not the exact round prefix")
    for round_index in range(expected_e2_selection_count):
        source = e2_materializations[round_index]
        destination = e2_materializations[round_index + 1]
        coverage = e2_coverage_by_materialization.get(source.sha256)
        manifest = e2_manifest_by_round[round_index]
        signed_selection = e2_selection_by_round[round_index]
        selection = signed_selection.payload
        verify_signed_payload(
            selection,
            payload_sha256=signed_selection.payload_sha256,
            challenge=signed_selection.challenge,
            attestation=signed_selection.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed_selection),
        )
        source_cells = {cell.cell_id for cell in source.cells}
        source_recipe_ids = {
            cell.recipe_sha256
            for cell in source.cells
            if cell.method_role == "LightCone-candidate"
        }
        destination_recipe_ids = {
            cell.recipe_sha256
            for cell in destination.cells
            if cell.method_role == "LightCone-candidate"
        }
        if (
            coverage is None
            or any(row.status != "COMPLETE" for row in coverage.dispositions)
            or manifest.protocol_lock_sha256 != lock.sha256
            or manifest.materialization_receipt_sha256 != source.sha256
            or manifest.coverage_receipt_sha256 != coverage.sha256
            or manifest.source_selection_sha256 != source.source_decision_sha256
            or manifest.inventory_sha256 != inventory_sha256
            or manifest.round_index != round_index
            or {row.materialized_cell_id for row in manifest.cells} != source_cells
            or len(manifest.cells) != len(source_cells)
            or selection.protocol_lock_sha256 != lock.sha256
            or selection.registry_sha256 != lock.registry_sha256
            or selection.materialization_receipt_sha256 != source.sha256
            or selection.coverage_receipt_sha256 != coverage.sha256
            or selection.source_selection_sha256 != source.source_decision_sha256
            or selection.evidence_manifest_sha256 != manifest.sha256
            or selection.inventory_sha256 != inventory_sha256
            or selection.round_index != round_index
            or selection.source_candidate_count != len(source_recipe_ids)
            or not {row.recipe.sha256 for row in selection.evaluations}
            <= source_recipe_ids
            or destination.upstream_receipt_sha256s != (source.sha256,)
            or destination.source_decision_sha256 != signed_selection.sha256
            or destination_recipe_ids
            != {row.sha256 for row in selection.survivor_recipes}
        ):
            raise ValueError("E2 staged selection differs from exact round lineage")
        source_authority_bindings_list.append(
            FormalSourceAuthorityBinding(
                stage="E2",
                authority_kind=f"e2_round_{round_index}_staged_selection",
                signed_authority_sha256=signed_selection.sha256,
                payload_sha256=selection.sha256,
                authority_sha256=manifest.sha256,
                challenge_sha256=signed_selection.challenge.sha256,
            )
        )

    e4_materializations = tuple(row for row in ordered_payloads if row.stage == "E4")
    e4_coverage_by_materialization = {
        row.materialization_receipt_sha256: row
        for row in coverage_payloads
        if row.stage == "E4"
    }
    expected_e4_selection_count = max(0, len(e4_materializations) - 1)
    if (
        len(e4_staged_evidence_manifests) != expected_e4_selection_count
        or len(signed_e4_stage_selections) != expected_e4_selection_count
    ):
        raise ValueError(
            "later E4 phases require one exact staged selection per prior phase"
        )
    e4_manifest_by_phase = {row.phase: row for row in e4_staged_evidence_manifests}
    e4_selection_by_phase = {
        row.payload.phase: row for row in signed_e4_stage_selections
    }
    expected_e4_phases = ("screen", "local")[:expected_e4_selection_count]
    if (
        len(e4_manifest_by_phase) != len(e4_staged_evidence_manifests)
        or len(e4_selection_by_phase) != len(signed_e4_stage_selections)
        or tuple(sorted(e4_manifest_by_phase, key=expected_e4_phases.index))
        != expected_e4_phases
        or tuple(sorted(e4_selection_by_phase, key=expected_e4_phases.index))
        != expected_e4_phases
    ):
        raise ValueError("E4 staged selections are not the exact phase prefix")
    for phase_index, phase in enumerate(expected_e4_phases):
        source = e4_materializations[phase_index]
        destination = e4_materializations[phase_index + 1]
        coverage = e4_coverage_by_materialization.get(source.sha256)
        evidence = e4_manifest_by_phase[phase]
        signed_selection = e4_selection_by_phase[phase]
        selection = signed_selection.payload
        verify_signed_payload(
            selection,
            payload_sha256=signed_selection.payload_sha256,
            challenge=signed_selection.challenge,
            attestation=signed_selection.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed_selection),
        )
        source_cells = {cell.cell_id: cell for cell in source.cells}
        source_models = {cell.model for cell in source.cells}
        source_recipes = {cell.recipe_sha256 for cell in source.cells}
        evidence_cell_ids = {row.materialized_cell_id for row in evidence.cells}
        evaluated_cell_ids = tuple(
            cell_id
            for evaluation in selection.evaluations
            for cell_id in evaluation.cell_ids
        )
        evaluation_configuration_is_exact = all(
            all(cell_id in source_cells for cell_id in evaluation.cell_ids)
            and {
                tuple(
                    (name, dict(source_cells[cell_id].dimensions).get(name))
                    for name, _value in evaluation.configuration
                )
                for cell_id in evaluation.cell_ids
            }
            == {evaluation.configuration}
            for evaluation in selection.evaluations
        )
        if (
            coverage is None
            or any(row.status != "COMPLETE" for row in coverage.dispositions)
            or evidence.phase != phase
            or evidence.protocol_lock_sha256 != lock.sha256
            or evidence.materialization_receipt_sha256 != source.sha256
            or evidence.coverage_receipt_sha256 != coverage.sha256
            or evidence.upstream_signed_authority_sha256
            != source.source_decision_sha256
            or evidence.inventory_sha256 != inventory_sha256
            or evidence_cell_ids != set(source_cells)
            or len(evidence.cells) != len(source_cells)
            or selection.phase != phase
            or selection.protocol_lock_sha256 != lock.sha256
            or selection.registry_sha256 != lock.registry_sha256
            or selection.materialization_receipt_sha256 != source.sha256
            or selection.coverage_receipt_sha256 != coverage.sha256
            or selection.upstream_signed_authority_sha256
            != source.source_decision_sha256
            or selection.evidence_manifest_sha256 != evidence.sha256
            or selection.inventory_sha256 != inventory_sha256
            or source_models != {selection.model}
            or source_recipes != {selection.lightcone_recipe_sha256}
            or len(evaluated_cell_ids) != len(set(evaluated_cell_ids))
            or not set(evaluated_cell_ids) <= set(source_cells)
            or not evaluation_configuration_is_exact
            or destination.upstream_receipt_sha256s != (source.sha256,)
            or destination.source_decision_sha256 != signed_selection.sha256
        ):
            raise ValueError("E4 staged selection differs from exact phase lineage")
        destination_models = {cell.model for cell in destination.cells}
        destination_recipes = {cell.recipe_sha256 for cell in destination.cells}
        if destination_models != {selection.model} or destination_recipes != {
            selection.lightcone_recipe_sha256
        }:
            raise ValueError("E4 destination changes the selected model or recipe")
        if phase == "screen":
            neighborhoods = selection.factor_neighborhoods
            if neighborhoods is None:
                raise ValueError("E4 screen selection lacks its local neighborhood")
            level_by_name = {name: {left, right} for name, left, right in neighborhoods}
            destination_configurations = {
                tuple(
                    (name, dict(cell.dimensions).get(name))
                    for name, _left, _right in neighborhoods
                )
                for cell in destination.cells
            }
            if (
                len(destination.cells) != 96
                or len(destination_configurations) != 16
                or any(
                    dict(cell.dimensions).get(name) not in levels
                    for cell in destination.cells
                    for name, levels in level_by_name.items()
                )
                or any(
                    sum(
                        tuple(
                            (name, dict(cell.dimensions).get(name))
                            for name, _left, _right in neighborhoods
                        )
                        == configuration
                        for cell in destination.cells
                    )
                    != 6
                    for configuration in destination_configurations
                )
            ):
                raise ValueError("E4 local factorial differs from signed neighborhood")
        else:
            selected_configuration_sha256 = content_sha256(
                selection.winner_configuration
            )
            if (
                selection.factor_neighborhoods is not None
                or len(destination.cells) != 3
                or {dict(cell.dimensions).get("profiler") for cell in destination.cells}
                != {"nvtx", "nsight_systems", "nsight_compute"}
                or {
                    dict(cell.dimensions).get("selected_configuration_sha256")
                    for cell in destination.cells
                }
                != {selected_configuration_sha256}
            ):
                raise ValueError("E4 profiler rows differ from local winner")
        source_authority_bindings_list.append(
            FormalSourceAuthorityBinding(
                stage="E4",
                authority_kind=f"e4_{phase}_selection",
                signed_authority_sha256=signed_selection.sha256,
                payload_sha256=selection.sha256,
                authority_sha256=evidence.sha256,
                challenge_sha256=signed_selection.challenge.sha256,
            )
        )
    e3b_materializations = tuple(row for row in ordered_payloads if row.stage == "E3b")
    if e3b_materializations or signed_e3b_power_prefixes:
        if len(e3b_materializations) != 1 or len(signed_e3b_power_prefixes) != 1:
            raise ValueError("E3b main registry row requires one signed power prefix")
        destination = e3b_materializations[0]
        signed_power = signed_e3b_power_prefixes[0]
        if type(signed_power) is not SignedE3bPowerPrefixReceipt:
            raise TypeError("formal E3b power prefix has the wrong signed type")
        power = signed_power.payload
        power.__post_init__()
        verify_signed_payload(
            power,
            payload_sha256=signed_power.payload_sha256,
            challenge=signed_power.challenge,
            attestation=signed_power.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed_power),
        )
        dimensions = tuple(dict(cell.dimensions) for cell in destination.cells)
        blocks = tuple(sorted({row.get("block") for row in dimensions}))
        if (
            power.protocol_lock_sha256 != lock.sha256
            or power.registry_sha256 != lock.registry_sha256
            or power.inventory_sha256 != inventory_sha256
            or destination.source_decision_sha256 != signed_power.sha256
            or blocks != power.selected_final_prefix
            or {row.get("signed_power_prefix_sha256") for row in dimensions}
            != {signed_power.sha256}
            or {row.get("pilot_materialization_receipt_sha256") for row in dimensions}
            != {power.pilot_materialization_receipt_sha256}
            or {row.get("pilot_coverage_receipt_sha256") for row in dimensions}
            != {power.pilot_coverage_receipt_sha256}
        ):
            raise ValueError("E3b signed power prefix differs from main registry row")
        source_authority_bindings_list.append(
            FormalSourceAuthorityBinding(
                stage="E3b",
                authority_kind="e3b_power_prefix",
                signed_authority_sha256=signed_power.sha256,
                payload_sha256=power.sha256,
                authority_sha256=power.evidence_manifest_sha256,
                challenge_sha256=signed_power.challenge.sha256,
            )
        )

    e5_materializations = tuple(row for row in ordered_payloads if row.stage == "E5")
    if e5_materializations or signed_e5_power_and_anchor_prefixes:
        if (
            len(e5_materializations) != 1
            or len(signed_e5_power_and_anchor_prefixes) != 1
        ):
            raise ValueError(
                "E5 main registry row requires one signed power/anchor prefix"
            )
        destination = e5_materializations[0]
        signed_power = signed_e5_power_and_anchor_prefixes[0]
        if type(signed_power) is not SignedE5PowerAndAnchorReceipt:
            raise TypeError("formal E5 power/anchor prefix has the wrong signed type")
        power = signed_power.payload
        power.__post_init__()
        verify_signed_payload(
            power,
            payload_sha256=signed_power.payload_sha256,
            challenge=signed_power.challenge,
            attestation=signed_power.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed_power),
        )
        headline = tuple(
            cell
            for cell in destination.cells
            if cell.task == "production_slo_power_prefix"
        )
        dimensions = tuple(dict(cell.dimensions) for cell in destination.cells)
        headline_dimensions = tuple(dict(cell.dimensions) for cell in headline)
        blocks = tuple(sorted({row.get("block") for row in headline_dimensions}))
        anchor_ids = {row.anchor_id for row in power.p99_anchors}
        if (
            power.protocol_lock_sha256 != lock.sha256
            or power.registry_sha256 != lock.registry_sha256
            or power.inventory_sha256 != inventory_sha256
            or destination.source_decision_sha256 != signed_power.sha256
            or blocks != power.selected_final_prefix
            or {cell.model for cell in destination.cells} != {power.model}
            or {row.get("signed_power_and_anchor_prefix_sha256") for row in dimensions}
            != {signed_power.sha256}
            or {row.get("upstream_e1a_verification_sha256") for row in dimensions}
            != {power.upstream_e1a_verification_sha256}
            or {row.get("pilot_materialization_receipt_sha256") for row in dimensions}
            != {power.pilot_materialization_receipt_sha256}
            or {row.get("pilot_coverage_receipt_sha256") for row in dimensions}
            != {power.pilot_coverage_receipt_sha256}
            or {
                row.get("p99_anchor_id")
                for row in headline_dimensions
                if row.get("p99_anchor_id") is not None
            }
            != anchor_ids
        ):
            raise ValueError(
                "E5 signed power/anchor prefix differs from main registry row"
            )
        source_authority_bindings_list.append(
            FormalSourceAuthorityBinding(
                stage="E5",
                authority_kind="e5_power_and_anchor_prefix",
                signed_authority_sha256=signed_power.sha256,
                payload_sha256=power.sha256,
                authority_sha256=power.evidence_manifest_sha256,
                challenge_sha256=signed_power.challenge.sha256,
            )
        )

    e6_materializations = tuple(row for row in ordered_payloads if row.stage == "E6")
    if e6_materializations or signed_e6_power_prefixes:
        if len(e6_materializations) != 1 or len(signed_e6_power_prefixes) != 1:
            raise ValueError("E6 main registry row requires one signed power prefix")
        destination = e6_materializations[0]
        signed_power = signed_e6_power_prefixes[0]
        if type(signed_power) is not SignedE6PowerPrefixReceipt:
            raise TypeError("formal E6 power prefix has the wrong signed type")
        power = signed_power.payload
        power.__post_init__()
        verify_signed_payload(
            power,
            payload_sha256=signed_power.payload_sha256,
            challenge=signed_power.challenge,
            attestation=signed_power.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed_power),
        )
        dimensions = tuple(dict(cell.dimensions) for cell in destination.cells)
        headline_dimensions = tuple(
            row for row in dimensions if type(row.get("block")) is int
        )
        blocks = tuple(sorted({row.get("block") for row in headline_dimensions}))
        if (
            power.protocol_lock_sha256 != lock.sha256
            or power.registry_sha256 != lock.registry_sha256
            or power.inventory_sha256 != inventory_sha256
            or destination.source_decision_sha256 != signed_power.sha256
            or blocks != power.selected_final_prefix
            or {row.get("signed_power_prefix_sha256") for row in dimensions}
            != {signed_power.sha256}
            or {row.get("pilot_materialization_receipt_sha256") for row in dimensions}
            != {power.pilot_materialization_receipt_sha256}
            or {row.get("pilot_coverage_receipt_sha256") for row in dimensions}
            != {power.pilot_coverage_receipt_sha256}
            or {row.get("upstream_e5_confirmation_sha256") for row in dimensions}
            != {power.upstream_e5_confirmation_sha256}
            or {row.get("signed_e6_model_compatibility_sha256") for row in dimensions}
            != {power.signed_model_compatibility_sha256}
        ):
            raise ValueError("E6 signed power prefix differs from main registry row")
        source_authority_bindings_list.append(
            FormalSourceAuthorityBinding(
                stage="E6",
                authority_kind="e6_power_prefix",
                signed_authority_sha256=signed_power.sha256,
                payload_sha256=power.sha256,
                authority_sha256=power.evidence_manifest_sha256,
                challenge_sha256=signed_power.challenge.sha256,
            )
        )

    e0_materializations = tuple(row for row in ordered_payloads if row.stage == "E0")
    e0_source_rows_present = any(
        (
            e0_onlinespec_source_authorities,
            signed_e0_compatibilities,
            signed_e0_onlinespec_tuning_seals,
            signed_e0_power_prefixes,
        )
    )
    if e0_materializations or e0_source_rows_present:
        if (
            len(e0_materializations) != 1
            or len(e0_onlinespec_source_authorities) != 1
            or len(signed_e0_compatibilities) != 1
            or len(signed_e0_power_prefixes) != 1
            or not signed_e0_onlinespec_tuning_seals
        ):
            raise ValueError(
                "E0 main registry row requires one exact durable typed source set"
            )
        source_authority = e0_onlinespec_source_authorities[0]
        if type(source_authority) is not E0OnlineSpecSourceAuthority:
            raise TypeError("formal E0 source authority has the wrong type")
        source_authority.revalidate()
        signed_compatibility = signed_e0_compatibilities[0]
        if type(signed_compatibility) is not SignedE0CompatibilityReceipt:
            raise TypeError("formal E0 compatibility has the wrong signed type")
        compatibility = signed_compatibility.verify(
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed_compatibility),
        )
        signed_power = signed_e0_power_prefixes[0]
        if type(signed_power) is not SignedE0PowerPrefixReceipt:
            raise TypeError("formal E0 power prefix has the wrong signed type")
        power = signed_power.payload
        power.__post_init__()
        verify_signed_payload(
            power,
            payload_sha256=signed_power.payload_sha256,
            challenge=signed_power.challenge,
            attestation=signed_power.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=verification_time(signed_power),
        )
        if type(signed_e0_onlinespec_tuning_seals) is not tuple or any(
            type(row) is not SignedE0OnlineSpecTuningSeal
            for row in signed_e0_onlinespec_tuning_seals
        ):
            raise TypeError("formal E0 tuning seals have the wrong signed type")
        tuning_payloads = []
        for signed_tuning in signed_e0_onlinespec_tuning_seals:
            tuning = signed_tuning.payload
            tuning.__post_init__()
            verify_signed_payload(
                tuning,
                payload_sha256=signed_tuning.payload_sha256,
                challenge=signed_tuning.challenge,
                attestation=signed_tuning.attestation,
                policy=policy,
                expected_policy_sha256=expected_policy_sha256,
                now_ns=verification_time(signed_tuning),
            )
            tuning_payloads.append(tuning)
        valid_decisions = tuple(
            row for row in compatibility.decisions if row.disposition == "VALID"
        )
        signed_tuning_by_decision = {
            row.payload.decision_id: row for row in signed_e0_onlinespec_tuning_seals
        }
        tuning_by_decision = {row.decision_id: row for row in tuning_payloads}
        final = e0_materializations[0]
        dimensions = tuple(dict(cell.dimensions) for cell in final.cells)
        block_values = {row.get("block") for row in dimensions}
        if any(type(block) is not int for block in block_values):
            raise ValueError("E0 durable final block set is not integral")
        final_blocks = tuple(
            sorted(block for block in block_values if type(block) is int)
        )
        e6_confirmation_sha256s = {
            *(row.upstream_e6_confirmation_sha256 for row in tuning_payloads),
            power.upstream_e6_confirmation_sha256,
            *(row.get("signed_e6_confirmation_sha256") for row in dimensions),
        }
        if (
            compatibility.protocol_lock_sha256 != lock.sha256
            or compatibility.upstream_e6_receipt_sha256
            != final.upstream_receipt_sha256s[0]
            or tuple(sorted(tuning_by_decision))
            != tuple(sorted(row.decision_id for row in valid_decisions))
            or len(tuning_by_decision) != len(tuning_payloads)
            or len(signed_tuning_by_decision) != len(tuning_payloads)
            or power.protocol_lock_sha256 != lock.sha256
            or power.registry_sha256 != lock.registry_sha256
            or power.signed_compatibility_sha256 != signed_compatibility.sha256
            or power.signed_tuning_seal_sha256s
            != tuple(sorted(row.sha256 for row in signed_e0_onlinespec_tuning_seals))
            or final.source_decision_sha256 != signed_power.sha256
            or final_blocks != power.selected_final_prefix
            or e6_confirmation_sha256s != {power.upstream_e6_confirmation_sha256}
            or power.inventory_sha256 != inventory_sha256
        ):
            raise ValueError("E0 durable source set differs from main registry lineage")
        for decision in valid_decisions:
            tuning = tuning_by_decision[decision.decision_id]
            signed_tuning = signed_tuning_by_decision[decision.decision_id]
            if (
                tuning.protocol_lock_sha256 != lock.sha256
                or tuning.registry_sha256 != lock.registry_sha256
                or tuning.signed_compatibility_sha256 != signed_compatibility.sha256
                or tuning.onlinespec_source_authority_sha256 != source_authority.sha256
                or tuning.model != decision.model
                or tuning.backend != decision.backend
                or tuning.task != decision.task
                or tuning.interface_sha256 != decision.interface_sha256
                or tuning.task_native_workload_sha256
                != decision.task_native_workload_sha256
                or {
                    row.get("signed_e0_tuning_seal_sha256")
                    for row in dimensions
                    if row.get("compatibility_decision_id") == decision.decision_id
                }
                != {signed_tuning.sha256}
            ):
                raise ValueError("E0 tuning seal differs from compatibility/main cells")
        tuning_set_sha256 = content_sha256(
            tuple(sorted(row.sha256 for row in signed_e0_onlinespec_tuning_seals))
        )
        if (
            {row.get("signed_e0_compatibility_sha256") for row in dimensions}
            != {signed_compatibility.sha256}
            or {row.get("signed_e0_tuning_seal_set_sha256") for row in dimensions}
            != {tuning_set_sha256}
            or {row.get("pilot_materialization_receipt_sha256") for row in dimensions}
            != {power.pilot_materialization_receipt_sha256}
            or {row.get("pilot_coverage_receipt_sha256") for row in dimensions}
            != {power.pilot_coverage_receipt_sha256}
        ):
            raise ValueError("E0 main cells change durable tuning/pilot lineage")
        source_authority_bindings_list.append(
            FormalSourceAuthorityBinding(
                stage="E0",
                authority_kind="e0_compatibility",
                signed_authority_sha256=signed_compatibility.sha256,
                payload_sha256=compatibility.sha256,
                authority_sha256=compatibility.sha256,
                challenge_sha256=signed_compatibility.challenge.sha256,
            )
        )
        source_authority_bindings_list.extend(
            FormalSourceAuthorityBinding(
                stage="E0",
                authority_kind="e0_onlinespec_tuning_seal",
                signed_authority_sha256=signed.sha256,
                payload_sha256=signed.payload.sha256,
                authority_sha256=source_authority.sha256,
                challenge_sha256=signed.challenge.sha256,
            )
            for signed in signed_e0_onlinespec_tuning_seals
        )
        source_authority_bindings_list.append(
            FormalSourceAuthorityBinding(
                stage="E0",
                authority_kind="e0_power_prefix",
                signed_authority_sha256=signed_power.sha256,
                payload_sha256=power.sha256,
                authority_sha256=power.evidence_manifest_sha256,
                challenge_sha256=signed_power.challenge.sha256,
            )
        )
    source_authority_bindings = tuple(
        sorted(
            source_authority_bindings_list,
            key=lambda row: row.signed_authority_sha256,
        )
    )
    _validate_candidate_coverage_replay_uniqueness(tuple(coverage_payloads))
    materialization_rows = tuple(materialization_bindings)
    materialization_index = {
        row.materialization_receipt_sha256: index
        for index, row in enumerate(materialization_rows)
    }
    coverage_rows = tuple(
        sorted(
            coverage_bindings,
            key=lambda row: materialization_index[row.materialization_receipt_sha256],
        )
    )
    terminal_by_materialization = {
        row.materialization_receipt_sha256: row.terminal_complete
        for row in coverage_rows
    }
    for prior in materialization_rows[:-1]:
        if not terminal_by_materialization.get(
            prior.materialization_receipt_sha256, False
        ):
            raise ValueError(
                "downstream materialization requires COMPLETE upstream coverage"
            )
    status = (
        "LOCKED"
        if not materialization_rows
        else "COVERED"
        if len(coverage_rows) == len(materialization_rows)
        and all(row.terminal_complete for row in coverage_rows)
        else "MATERIALIZED_PENDING_COVERAGE"
    )
    return FormalRegistryManifest(
        schema_version=2,
        kind="lightcone_formal_signed_registry_manifest",
        registry_sha256=registry.sha256,
        protocol_lock_sha256=lock.sha256,
        signed_protocol_lock_sha256=signed_protocol_lock.sha256,
        prior_registry_verification_receipt_sha256=(
            prior_registry_verification_receipt_sha256
        ),
        inventory_sha256=inventory_sha256,
        deployment_policy_authorization_sha256=(deployment_policy_authorization_sha256),
        trusted_attester_policy_sha256=expected_policy_sha256,
        control_lineage_sha256=control_lineage_sha256,
        control_envelope_sha256s=control_envelope_sha256s,
        challenge_reservation_sha256=challenge_reservation_sha256,
        stage_order=FORMAL_STAGE_DAG,
        materializations=materialization_rows,
        coverage=coverage_rows,
        source_authorities=source_authority_bindings,
        candidate_replay_proofs=candidate_replay_proofs,
        status=status,
    )


@dataclass(frozen=True)
class _PreparedFormalRegistryManifest:
    """Purely verified manifest plus the controls still needing reservation."""

    manifest: FormalRegistryManifest
    ordered_controls: tuple[ControlArtifactAttestation, ...]
    additional_challenge_sha256s: tuple[str, ...]
    deployment_policy_authorization_sha256: str
    trusted_attester_policy_sha256: str


def _prepare_formal_registry_manifest(
    signed_protocol_lock: SignedProtocolLock,
    *,
    signed_materializations: tuple[SignedStageMaterializationReceipt, ...],
    signed_coverage: tuple[SignedStageCoverageReceipt, ...],
    e3a_staged_selection_artifacts: tuple[E3aStagedSelectionArtifact, ...] = (),
    signed_e3a_staged_selections: tuple[SignedE3aStagedSelectionReceipt, ...] = (),
    e2_staged_evidence_manifests: tuple[E2StagedRoundEvidenceManifest, ...] = (),
    signed_e2_staged_selections: tuple[SignedE2StagedRoundSelectionReceipt, ...] = (),
    signed_e1_survivor_selections: tuple[SignedE1SurvivorSelectionReceipt, ...] = (),
    e4_staged_evidence_manifests: tuple[E4StagedEvidenceManifest, ...] = (),
    signed_e4_stage_selections: tuple[SignedE4StageSelectionReceipt, ...] = (),
    signed_e3b_power_prefixes: tuple[SignedE3bPowerPrefixReceipt, ...] = (),
    signed_e5_power_and_anchor_prefixes: tuple[SignedE5PowerAndAnchorReceipt, ...] = (),
    signed_e6_power_prefixes: tuple[SignedE6PowerPrefixReceipt, ...] = (),
    e0_onlinespec_source_authorities: tuple[E0OnlineSpecSourceAuthority, ...] = (),
    signed_e0_compatibilities: tuple[SignedE0CompatibilityReceipt, ...] = (),
    signed_e0_onlinespec_tuning_seals: tuple[SignedE0OnlineSpecTuningSeal, ...] = (),
    signed_e0_power_prefixes: tuple[SignedE0PowerPrefixReceipt, ...] = (),
    tts_calibration_authorities: tuple[TtsCalibrationAuthority, ...] = (),
    signed_tts_calibration_seals: tuple[SignedTtsCalibrationSeal, ...] = (),
    control_attestations: tuple[ControlArtifactAttestation, ...],
    candidate_replay_proof_artifact_paths: tuple[str, ...] = (),
    controlled_signed_row_sha256s: frozenset[str] | None = None,
    controlled_signed_source_authority_sha256s: frozenset[str] | None = None,
    prior_registry_verification_receipt_sha256: str | None = None,
    verification_ns_by_signed_sha256: dict[str, int] | None = None,
    expected_inventory_sha256: str,
    now_ns: int,
) -> _PreparedFormalRegistryManifest:
    """Verify all registry structure/signatures without consuming a challenge."""

    typed_signed_rows = (
        (signed_protocol_lock, "dispatch"),
        *((row, "dispatch") for row in signed_materializations),
        *((row, "rank_aggregate") for row in signed_coverage),
    )
    all_signed_artifact_types = {
        row.sha256: artifact_type for row, artifact_type in typed_signed_rows
    }
    if (
        type(e3a_staged_selection_artifacts) is not tuple
        or any(
            type(row) is not E3aStagedSelectionArtifact
            for row in e3a_staged_selection_artifacts
        )
        or len({row.sha256 for row in e3a_staged_selection_artifacts})
        != len(e3a_staged_selection_artifacts)
        or type(signed_e3a_staged_selections) is not tuple
        or any(
            type(row) is not SignedE3aStagedSelectionReceipt
            for row in signed_e3a_staged_selections
        )
        or len({row.sha256 for row in signed_e3a_staged_selections})
        != len(signed_e3a_staged_selections)
    ):
        raise TypeError("formal E3a source authorities are not exact unique tuples")
    if (
        type(e2_staged_evidence_manifests) is not tuple
        or any(
            type(row) is not E2StagedRoundEvidenceManifest
            for row in e2_staged_evidence_manifests
        )
        or len({row.sha256 for row in e2_staged_evidence_manifests})
        != len(e2_staged_evidence_manifests)
        or type(signed_e2_staged_selections) is not tuple
        or any(
            type(row) is not SignedE2StagedRoundSelectionReceipt
            for row in signed_e2_staged_selections
        )
        or len({row.sha256 for row in signed_e2_staged_selections})
        != len(signed_e2_staged_selections)
    ):
        raise TypeError("formal E2 source authorities are not exact unique tuples")
    if (
        type(signed_e1_survivor_selections) is not tuple
        or any(
            type(row) is not SignedE1SurvivorSelectionReceipt
            for row in signed_e1_survivor_selections
        )
        or len({row.sha256 for row in signed_e1_survivor_selections})
        != len(signed_e1_survivor_selections)
    ):
        raise TypeError("formal E1 survivor authorities are not an exact unique tuple")
    if (
        type(e4_staged_evidence_manifests) is not tuple
        or any(
            type(row) is not E4StagedEvidenceManifest
            for row in e4_staged_evidence_manifests
        )
        or len({row.sha256 for row in e4_staged_evidence_manifests})
        != len(e4_staged_evidence_manifests)
        or type(signed_e4_stage_selections) is not tuple
        or any(
            type(row) is not SignedE4StageSelectionReceipt
            for row in signed_e4_stage_selections
        )
        or len({row.sha256 for row in signed_e4_stage_selections})
        != len(signed_e4_stage_selections)
    ):
        raise TypeError("formal E4 source authorities are not exact unique tuples")
    if (
        type(tts_calibration_authorities) is not tuple
        or any(
            type(row) is not TtsCalibrationAuthority
            for row in tts_calibration_authorities
        )
        or len({row.sha256 for row in tts_calibration_authorities})
        != len(tts_calibration_authorities)
        or type(signed_tts_calibration_seals) is not tuple
        or any(
            type(row) is not SignedTtsCalibrationSeal
            for row in signed_tts_calibration_seals
        )
        or len({row.sha256 for row in signed_tts_calibration_seals})
        != len(signed_tts_calibration_seals)
    ):
        raise TypeError("formal TTS source authorities are not exact unique tuples")
    for label, rows, expected_type in (
        (
            "signed E3b power prefix",
            signed_e3b_power_prefixes,
            SignedE3bPowerPrefixReceipt,
        ),
        (
            "signed E5 power/anchor prefix",
            signed_e5_power_and_anchor_prefixes,
            SignedE5PowerAndAnchorReceipt,
        ),
        (
            "signed E6 power prefix",
            signed_e6_power_prefixes,
            SignedE6PowerPrefixReceipt,
        ),
        (
            "E0 OnlineSPEC source",
            e0_onlinespec_source_authorities,
            E0OnlineSpecSourceAuthority,
        ),
        (
            "signed E0 compatibility",
            signed_e0_compatibilities,
            SignedE0CompatibilityReceipt,
        ),
        (
            "signed E0 OnlineSPEC tuning",
            signed_e0_onlinespec_tuning_seals,
            SignedE0OnlineSpecTuningSeal,
        ),
        (
            "signed E0 power prefix",
            signed_e0_power_prefixes,
            SignedE0PowerPrefixReceipt,
        ),
    ):
        if (
            type(rows) is not tuple
            or any(type(row) is not expected_type for row in rows)
            or len({row.sha256 for row in rows}) != len(rows)
        ):
            raise TypeError(f"formal {label} authorities are not exact unique tuples")
    all_source_authority_sha256s = frozenset(
        row.sha256
        for row in (
            *signed_e3a_staged_selections,
            *signed_e2_staged_selections,
            *signed_e1_survivor_selections,
            *signed_e4_stage_selections,
            *signed_e3b_power_prefixes,
            *signed_e5_power_and_anchor_prefixes,
            *signed_e6_power_prefixes,
            *signed_tts_calibration_seals,
            *signed_e0_compatibilities,
            *signed_e0_onlinespec_tuning_seals,
            *signed_e0_power_prefixes,
        )
    )
    controlled_source_sha256s = (
        all_source_authority_sha256s
        if controlled_signed_source_authority_sha256s is None
        else controlled_signed_source_authority_sha256s
    )
    if (
        type(controlled_source_sha256s) is not frozenset
        or not controlled_source_sha256s <= all_source_authority_sha256s
    ):
        raise ValueError("formal current signed source-authority set is invalid")
    controlled_sha256s = (
        frozenset(all_signed_artifact_types)
        if controlled_signed_row_sha256s is None
        else controlled_signed_row_sha256s
    )
    if (
        type(controlled_sha256s) is not frozenset
        or not controlled_sha256s
        or not controlled_sha256s <= set(all_signed_artifact_types)
    ):
        raise ValueError(
            "formal registry current control set must be a non-empty signed-row subset"
        )
    if (
        type(control_attestations) is not tuple
        or len(control_attestations) != len(controlled_sha256s)
        or any(
            type(row) is not ControlArtifactAttestation for row in control_attestations
        )
    ):
        raise TypeError("formal registry requires one exact control per signed row")
    registry_sha256 = build_industrial_registry().sha256
    signed_artifact_types = {
        digest: all_signed_artifact_types[digest] for digest in controlled_sha256s
    }
    signed_artifact_sha256s = tuple(sorted(signed_artifact_types))
    if len(signed_artifact_sha256s) != len(set(signed_artifact_sha256s)):
        raise ValueError("formal registry signed rows must be unique")
    control_lineage = {
        "schema_version": 1,
        "kind": "lightcone_formal_registry_control_lineage",
        "protocol_lock_sha256": signed_protocol_lock.payload.sha256,
        "registry_sha256": registry_sha256,
        "signed_artifacts": tuple(
            (digest, signed_artifact_types[digest])
            for digest in signed_artifact_sha256s
        ),
    }
    if prior_registry_verification_receipt_sha256 is not None:
        control_lineage["prior_registry_verification_receipt_sha256"] = (
            prior_registry_verification_receipt_sha256
        )
    if controlled_source_sha256s:
        source_kinds = {
            **{
                row.sha256: "e3a_staged_selection"
                for row in signed_e3a_staged_selections
            },
            **{
                row.sha256: f"e2_round_{row.payload.round_index}_staged_selection"
                for row in signed_e2_staged_selections
            },
            **{
                row.sha256: "e1_staged_pareto_survivors"
                for row in signed_e1_survivor_selections
            },
            **{
                row.sha256: f"e4_{row.payload.phase}_selection"
                for row in signed_e4_stage_selections
            },
            **{row.sha256: "e3b_power_prefix" for row in signed_e3b_power_prefixes},
            **{
                row.sha256: "e5_power_and_anchor_prefix"
                for row in signed_e5_power_and_anchor_prefixes
            },
            **{row.sha256: "e6_power_prefix" for row in signed_e6_power_prefixes},
            **{
                row.sha256: "tts_calibration_seal"
                for row in signed_tts_calibration_seals
            },
            **{row.sha256: "e0_compatibility" for row in signed_e0_compatibilities},
            **{
                row.sha256: "e0_onlinespec_tuning_seal"
                for row in signed_e0_onlinespec_tuning_seals
            },
            **{row.sha256: "e0_power_prefix" for row in signed_e0_power_prefixes},
        }
        control_lineage["signed_source_authorities"] = tuple(
            (digest, source_kinds[digest])
            for digest in sorted(controlled_source_sha256s)
        )
    control_lineage_sha256 = content_sha256(control_lineage)
    ordered_controls = tuple(
        sorted(control_attestations, key=lambda row: row.subject.artifact_sha256)
    )
    if tuple(row.subject.artifact_sha256 for row in ordered_controls) != (
        signed_artifact_sha256s
    ):
        raise ValueError("formal controls do not cover every and only signed row")
    first = ordered_controls[0]
    authorization = first.deployment_policy_authorization
    bundle = authorization.bundle
    policy = bundle.trusted_attester_policy
    if (
        signed_protocol_lock.payload.offline_release_trust_root_sha256
        != authorization.root_manifest_sha256
    ):
        raise ValueError(
            "ProtocolLock offline release root differs from deployment authorization"
        )
    for control in ordered_controls:
        subject = control.subject
        if (
            subject.artifact_type != signed_artifact_types[subject.artifact_sha256]
            or subject.protocol_sha256 != signed_protocol_lock.payload.sha256
            or subject.registry_sha256 != registry_sha256
            or subject.lineage_sha256 != control_lineage_sha256
            or control.deployment_policy_authorization.sha256 != authorization.sha256
            or control.trust_bundle_sha256 != bundle.sha256
            or control.trusted_attester_policy_sha256 != policy.sha256
        ):
            raise ValueError("formal control subject or deployment lineage differs")
    if type(candidate_replay_proof_artifact_paths) is not tuple or any(
        type(path) is not str for path in candidate_replay_proof_artifact_paths
    ):
        raise TypeError("candidate replay proof paths must be an exact string tuple")
    replay_rows = []
    for path in candidate_replay_proof_artifact_paths:
        before = CanonicalJsonProofBinding.bind(path)
        pointer = validate_candidate_state_replay_proof_artifact(
            path,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_registry_sha256=registry_sha256,
            expected_root_manifest_sha256=(
                signed_protocol_lock.payload.offline_release_trust_root_sha256
            ),
            now_ns=now_ns,
        )
        after = CanonicalJsonProofBinding.bind(path)
        if before != after:
            raise RuntimeError("candidate replay proof changed while validated")
        replay_rows.append((pointer, after))
    if any(
        pointer.authority_kind != "external_release_control"
        or pointer.trusted_attester_policy_sha256 != policy.sha256
        for pointer, _binding in replay_rows
    ):
        raise ValueError(
            "candidate replay proof uses another external-control release policy"
        )
    replay_pointers = tuple(pointer for pointer, _binding in replay_rows)
    replay_pointers_by_sha = {
        pointer.semantic_commitment_sha256: pointer for pointer in replay_pointers
    }
    if len(replay_pointers_by_sha) != len(replay_pointers):
        raise ValueError("candidate replay proof artifacts are duplicated")
    candidate_coverages = tuple(
        candidate
        for signed in signed_coverage
        for candidate in signed.payload.tts_l0_candidate_state_coverages
    )
    expected_pointer_sha256s = {
        pointer_sha256
        for candidate in candidate_coverages
        for pointer_sha256 in (
            candidate.tts_native_replay_pointer_sha256,
            candidate.l0_naive_native_replay_pointer_sha256,
        )
    }
    if set(replay_pointers_by_sha) != expected_pointer_sha256s:
        raise ValueError(
            "candidate replay proof artifacts do not cover every and only signed "
            "TTS/L0 replay commitment"
        )
    for candidate in candidate_coverages:
        candidate.validate_native_replay_pointers(replay_pointers)
    additional_challenges = (
        *(
            row.challenge.sha256
            for row, _ in typed_signed_rows
            if row.sha256 in controlled_sha256s
        ),
        *(
            row.challenge.sha256
            for row in (
                *signed_e3a_staged_selections,
                *signed_e2_staged_selections,
                *signed_e1_survivor_selections,
                *signed_e4_stage_selections,
                *signed_e3b_power_prefixes,
                *signed_e5_power_and_anchor_prefixes,
                *signed_e6_power_prefixes,
                *signed_tts_calibration_seals,
                *signed_e0_compatibilities,
                *signed_e0_onlinespec_tuning_seals,
                *signed_e0_power_prefixes,
            )
            if row.sha256 in controlled_source_sha256s
        ),
    )
    reservation_challenges = tuple(
        sorted(
            {
                authorization.challenge.sha256,
                *(row.challenge.sha256 for row in ordered_controls),
                *additional_challenges,
            }
        )
    )
    if len(reservation_challenges) != 1 + len(ordered_controls) + len(
        additional_challenges
    ):
        raise ValueError("formal control and signed-row challenges must be distinct")
    predicted_reservation_sha256 = content_sha256(
        {
            "schema_version": 2,
            "kind": "lightcone_control_challenge_reservation",
            "reserved_ns": now_ns,
            "challenge_sha256s": reservation_challenges,
        }
    )
    manifest = _build_formal_registry_manifest_with_policy(
        signed_protocol_lock,
        signed_materializations=signed_materializations,
        signed_coverage=signed_coverage,
        e3a_staged_selection_artifacts=e3a_staged_selection_artifacts,
        signed_e3a_staged_selections=signed_e3a_staged_selections,
        e2_staged_evidence_manifests=e2_staged_evidence_manifests,
        signed_e2_staged_selections=signed_e2_staged_selections,
        signed_e1_survivor_selections=signed_e1_survivor_selections,
        e4_staged_evidence_manifests=e4_staged_evidence_manifests,
        signed_e4_stage_selections=signed_e4_stage_selections,
        signed_e3b_power_prefixes=signed_e3b_power_prefixes,
        signed_e5_power_and_anchor_prefixes=(signed_e5_power_and_anchor_prefixes),
        signed_e6_power_prefixes=signed_e6_power_prefixes,
        e0_onlinespec_source_authorities=e0_onlinespec_source_authorities,
        signed_e0_compatibilities=signed_e0_compatibilities,
        signed_e0_onlinespec_tuning_seals=signed_e0_onlinespec_tuning_seals,
        signed_e0_power_prefixes=signed_e0_power_prefixes,
        tts_calibration_authorities=tts_calibration_authorities,
        signed_tts_calibration_seals=signed_tts_calibration_seals,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        inventory_sha256=expected_inventory_sha256,
        deployment_policy_authorization_sha256=authorization.sha256,
        control_lineage_sha256=control_lineage_sha256,
        control_envelope_sha256s=tuple(sorted(row.sha256 for row in ordered_controls)),
        challenge_reservation_sha256=predicted_reservation_sha256,
        candidate_replay_proofs=tuple(
            sorted(
                (
                    FormalCandidateReplayProofBinding(
                        pointer_commitment_sha256=(pointer.semantic_commitment_sha256),
                        proof_artifact=binding,
                    )
                    for pointer, binding in replay_rows
                ),
                key=lambda row: row.pointer_commitment_sha256,
            )
        ),
        prior_registry_verification_receipt_sha256=(
            prior_registry_verification_receipt_sha256
        ),
        verification_ns_by_signed_sha256=verification_ns_by_signed_sha256,
        now_ns=now_ns,
    )
    return _PreparedFormalRegistryManifest(
        manifest=manifest,
        ordered_controls=ordered_controls,
        additional_challenge_sha256s=additional_challenges,
        deployment_policy_authorization_sha256=authorization.sha256,
        trusted_attester_policy_sha256=policy.sha256,
    )


def assemble_and_reserve_formal_registry_manifest(
    signed_protocol_lock: SignedProtocolLock,
    *,
    signed_materializations: tuple[SignedStageMaterializationReceipt, ...],
    signed_coverage: tuple[SignedStageCoverageReceipt, ...],
    e3a_staged_selection_artifacts: tuple[E3aStagedSelectionArtifact, ...] = (),
    signed_e3a_staged_selections: tuple[SignedE3aStagedSelectionReceipt, ...] = (),
    e2_staged_evidence_manifests: tuple[E2StagedRoundEvidenceManifest, ...] = (),
    signed_e2_staged_selections: tuple[SignedE2StagedRoundSelectionReceipt, ...] = (),
    signed_e1_survivor_selections: tuple[SignedE1SurvivorSelectionReceipt, ...] = (),
    e4_staged_evidence_manifests: tuple[E4StagedEvidenceManifest, ...] = (),
    signed_e4_stage_selections: tuple[SignedE4StageSelectionReceipt, ...] = (),
    signed_e3b_power_prefixes: tuple[SignedE3bPowerPrefixReceipt, ...] = (),
    signed_e5_power_and_anchor_prefixes: tuple[SignedE5PowerAndAnchorReceipt, ...] = (),
    signed_e6_power_prefixes: tuple[SignedE6PowerPrefixReceipt, ...] = (),
    tts_calibration_authorities: tuple[TtsCalibrationAuthority, ...] = (),
    signed_tts_calibration_seals: tuple[SignedTtsCalibrationSeal, ...] = (),
    control_attestations: tuple[ControlArtifactAttestation, ...],
    candidate_replay_proof_artifact_paths: tuple[str, ...] = (),
    expected_inventory_sha256: str,
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> FormalRegistryManifest:
    """Verify the dynamic release policy and atomically reserve every challenge."""

    prepared = _prepare_formal_registry_manifest(
        signed_protocol_lock,
        signed_materializations=signed_materializations,
        signed_coverage=signed_coverage,
        e3a_staged_selection_artifacts=e3a_staged_selection_artifacts,
        signed_e3a_staged_selections=signed_e3a_staged_selections,
        e2_staged_evidence_manifests=e2_staged_evidence_manifests,
        signed_e2_staged_selections=signed_e2_staged_selections,
        signed_e1_survivor_selections=signed_e1_survivor_selections,
        e4_staged_evidence_manifests=e4_staged_evidence_manifests,
        signed_e4_stage_selections=signed_e4_stage_selections,
        signed_e3b_power_prefixes=signed_e3b_power_prefixes,
        signed_e5_power_and_anchor_prefixes=(signed_e5_power_and_anchor_prefixes),
        signed_e6_power_prefixes=signed_e6_power_prefixes,
        tts_calibration_authorities=tts_calibration_authorities,
        signed_tts_calibration_seals=signed_tts_calibration_seals,
        control_attestations=control_attestations,
        candidate_replay_proof_artifact_paths=(candidate_replay_proof_artifact_paths),
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=now_ns,
    )
    verified = verify_and_reserve_release_control_artifact_attestations(
        prepared.ordered_controls,
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=prepared.additional_challenge_sha256s,
    )
    reservation = FormalControlReservation(
        inventory_sha256=expected_inventory_sha256,
        deployment_policy_authorization_sha256=(
            prepared.deployment_policy_authorization_sha256
        ),
        trusted_attester_policy_sha256=(prepared.trusted_attester_policy_sha256),
        control_lineage_sha256=prepared.manifest.control_lineage_sha256,
        control_envelope_sha256s=tuple(sorted(row.envelope_sha256 for row in verified)),
        challenge_reservation_sha256=control_challenge_reservation_sha256(
            verified,
            additional_challenge_sha256s=prepared.additional_challenge_sha256s,
            reserved_ns=now_ns,
        ),
        verified_artifacts=verified,
    )
    if (
        reservation.challenge_reservation_sha256
        != prepared.manifest.challenge_reservation_sha256
        or reservation.control_envelope_sha256s
        != prepared.manifest.control_envelope_sha256s
    ):
        raise RuntimeError("formal control reservation differs from prepared manifest")
    return prepared.manifest


def reserve_formal_registry_verification_receipt(
    signed_protocol_lock: SignedProtocolLock,
    *,
    control_attestation: ControlArtifactAttestation,
    expected_inventory_sha256: str,
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> FormalRegistryVerificationReceipt:
    """Consume the ProtocolLock once and publish the durable registry root."""

    prepared = _prepare_formal_registry_manifest(
        signed_protocol_lock,
        signed_materializations=(),
        signed_coverage=(),
        control_attestations=(control_attestation,),
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=now_ns,
    )
    verified = verify_and_reserve_release_control_artifact_attestations(
        prepared.ordered_controls,
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=prepared.additional_challenge_sha256s,
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified,
        additional_challenge_sha256s=prepared.additional_challenge_sha256s,
        reserved_ns=now_ns,
    )
    receipt = FormalRegistryVerificationReceipt(
        schema_version=2,
        kind="lightcone_formal_registry_verification_receipt",
        verified_ns=now_ns,
        inventory_sha256=expected_inventory_sha256,
        signed_protocol_lock=signed_protocol_lock,
        prior_receipt=None,
        appended_signed_materializations=(),
        appended_signed_coverage=(),
        appended_e3a_staged_selection_artifacts=(),
        appended_signed_e3a_staged_selections=(),
        appended_tts_calibration_authorities=(),
        appended_signed_tts_calibration_seals=(),
        control_attestations=(control_attestation,),
        reservation=replay_store.bind_reservation(reservation_sha256),
        manifest=prepared.manifest,
    )
    receipt.revalidate(current_ns=now_ns)
    return receipt


def _verify_formal_stage_prefix_append(
    *,
    prior_receipt: FormalRegistryVerificationReceipt,
    prefix_bindings: tuple[CanonicalJsonProofBinding, ...],
    appended_signed_materializations: tuple[SignedStageMaterializationReceipt, ...],
    appended_signed_coverage: tuple[SignedStageCoverageReceipt, ...],
    appended_e2_staged_evidence_manifests: tuple[E2StagedRoundEvidenceManifest, ...],
    appended_signed_e2_staged_selections: tuple[
        SignedE2StagedRoundSelectionReceipt, ...
    ],
    appended_signed_e1_survivor_selections: tuple[
        SignedE1SurvivorSelectionReceipt, ...
    ],
    appended_e4_staged_evidence_manifests: tuple[E4StagedEvidenceManifest, ...],
    appended_signed_e4_stage_selections: tuple[SignedE4StageSelectionReceipt, ...],
    now_ns: int,
) -> None:
    """Deep-rebuild the mandatory current-only E1/E2/E4 proof prefix."""

    from lightcone_spec.experiments.formal_stage_prefix import (
        load_and_rebuild_formal_stage_prefix,
        verify_signed_formal_stage_prefix_result,
    )

    if type(prefix_bindings) is not tuple or any(
        type(row) is not CanonicalJsonProofBinding for row in prefix_bindings
    ):
        raise TypeError("formal stage prefix bindings are not exact")
    early_coverages = tuple(
        row
        for row in appended_signed_coverage
        if row.payload.stage in {"E1", "E2", "E4"}
    )
    early_selections = (
        *appended_signed_e1_survivor_selections,
        *appended_signed_e2_staged_selections,
        *appended_signed_e4_stage_selections,
    )
    if not early_coverages and not early_selections:
        if prefix_bindings:
            raise ValueError("formal registry append carries an unused stage prefix")
        if any(row.payload.stage == "E3b" for row in appended_signed_materializations):
            profiler_prefixes = tuple(
                load_and_rebuild_formal_stage_prefix(
                    row.absolute_path,
                    now_ns=now_ns,
                )
                for row in prior_receipt.cumulative_formal_stage_prefix_artifacts
            )
            completed = tuple(
                row
                for row in profiler_prefixes
                if row.artifact.phase == "e4_profiler"
                and any(
                    signed.payload == row.materialization
                    for signed in prior_receipt.cumulative_signed_materializations
                )
                and any(
                    signed.payload == row.coverage
                    for signed in prior_receipt.cumulative_signed_coverage
                )
                and all(
                    disposition.status == "COMPLETE"
                    for disposition in row.coverage.dispositions
                )
            )
            if len(completed) != 1:
                raise ValueError(
                    "formal E3b append lacks one proof-derived profiler completion"
                )
        return
    if len(prefix_bindings) != 1 or len(early_coverages) != 1:
        raise ValueError(
            "formal E1/E2/E4 append requires one exact path-bound current prefix"
        )
    binding = prefix_bindings[0]
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("formal stage prefix artifact path changed before rebuild")
    prefix = load_and_rebuild_formal_stage_prefix(
        binding.absolute_path,
        now_ns=now_ns,
    )
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise RuntimeError("formal stage prefix artifact changed while rebuilt")
    if (
        prefix.registry_verification_receipt.sha256 != prior_receipt.sha256
        or prefix.materialization.sha256
        not in {
            row.payload.sha256
            for row in prior_receipt.cumulative_signed_materializations
        }
        or early_coverages[0].payload != prefix.coverage
    ):
        raise ValueError("formal stage prefix differs from the current registry layer")

    phase = prefix.artifact.phase
    expected_e1: tuple[SignedE1SurvivorSelectionReceipt, ...] = ()
    expected_e2: tuple[SignedE2StagedRoundSelectionReceipt, ...] = ()
    expected_e4: tuple[SignedE4StageSelectionReceipt, ...] = ()
    expected_e2_evidence: tuple[E2StagedRoundEvidenceManifest, ...] = ()
    expected_e4_evidence: tuple[E4StagedEvidenceManifest, ...] = ()
    selected_signed: object | None = None
    if phase == "e1_selection":
        expected_e1 = appended_signed_e1_survivor_selections
        if len(expected_e1) != 1:
            raise ValueError("formal E1 coverage requires one proof-derived selection")
        selected_signed = expected_e1[0]
    elif phase.startswith("e2_round"):
        round_index = int(phase.removeprefix("e2_round"))
        expected_e2 = tuple(
            row
            for row in appended_signed_e2_staged_selections
            if row.payload.round_index == round_index
        )
        if len(expected_e2) != 1 or not isinstance(
            prefix.evidence_manifest, E2StagedRoundEvidenceManifest
        ):
            raise ValueError("formal E2 coverage requires its exact staged reducer")
        selected_signed = expected_e2[0]
        expected_e2_evidence = (prefix.evidence_manifest,)
    elif phase in {"e4_screen", "e4_local"}:
        expected_phase = phase.removeprefix("e4_")
        expected_e4 = tuple(
            row
            for row in appended_signed_e4_stage_selections
            if row.payload.phase == expected_phase
        )
        if len(expected_e4) != 1 or not isinstance(
            prefix.evidence_manifest, E4StagedEvidenceManifest
        ):
            raise ValueError("formal E4 headline coverage requires its exact reducer")
        selected_signed = expected_e4[0]
        expected_e4_evidence = (prefix.evidence_manifest,)
    elif phase == "e4_profiler":
        if early_selections:
            raise ValueError("formal E4 profiler completion cannot claim a selection")
        if any(row.status != "COMPLETE" for row in prefix.coverage.dispositions):
            raise ValueError("formal E4 profiler prefix is not terminal-complete")
    else:  # pragma: no cover - closed by FormalStagePrefixArtifact
        raise AssertionError(phase)

    if (
        appended_signed_e1_survivor_selections != expected_e1
        or appended_signed_e2_staged_selections != expected_e2
        or appended_signed_e4_stage_selections != expected_e4
        or appended_e2_staged_evidence_manifests != expected_e2_evidence
        or appended_e4_staged_evidence_manifests != expected_e4_evidence
    ):
        raise ValueError("formal stage prefix sources are incomplete or foreign")
    if selected_signed is None:
        return
    verified = verify_signed_formal_stage_prefix_result(
        prefix,
        selected_signed,  # type: ignore[arg-type]
        now_ns=now_ns,
    )
    if verified != selected_signed.payload:  # type: ignore[attr-defined]
        raise AssertionError("formal stage prefix reducer verification changed payload")

    successors = tuple(
        row.payload
        for row in appended_signed_materializations
        if row.payload.source_decision_sha256 == selected_signed.sha256  # type: ignore[attr-defined]
        and row.payload.upstream_receipt_sha256s == (prefix.materialization.sha256,)
    )
    if len(successors) != 1:
        raise ValueError(
            "formal stage reducer append lacks its exact next materialization"
        )
    successor = successors[0]
    if phase == "e1_selection":
        valid_successor = successor.stage == "E2" and _e2_round(successor) == 0
    elif phase in {"e2_round0", "e2_round1", "e2_round2"}:
        valid_successor = (
            successor.stage == "E2"
            and _e2_round(successor) == int(phase.removeprefix("e2_round")) + 1
        )
    elif phase == "e2_round3":
        valid_successor = (
            successor.stage == "E4"
            and successor.materialization_rule
            == "strength2_8_rows_x_3_loads_x_2_traffic"
        )
    elif phase == "e4_screen":
        valid_successor = (
            successor.stage == "E4"
            and successor.materialization_rule
            == "winner_neighborhood_2pow4_x_3_loads_x_2_traffic"
        )
    else:
        valid_successor = (
            successor.stage == "E4"
            and successor.materialization_rule
            == "three_profiler_only_rows_separate_from_headline"
        )
    if not valid_successor:
        raise ValueError(
            "formal stage prefix selection targets the wrong DAG successor"
        )


def extend_formal_registry_verification_receipt(
    prior_receipt: FormalRegistryVerificationReceipt,
    *,
    appended_signed_materializations: tuple[
        SignedStageMaterializationReceipt, ...
    ] = (),
    appended_signed_coverage: tuple[SignedStageCoverageReceipt, ...] = (),
    appended_e3a_staged_selection_artifacts: tuple[
        E3aStagedSelectionArtifact, ...
    ] = (),
    appended_signed_e3a_staged_selections: tuple[
        SignedE3aStagedSelectionReceipt, ...
    ] = (),
    appended_e2_staged_evidence_manifests: tuple[
        E2StagedRoundEvidenceManifest, ...
    ] = (),
    appended_signed_e2_staged_selections: tuple[
        SignedE2StagedRoundSelectionReceipt, ...
    ] = (),
    appended_signed_e1_survivor_selections: tuple[
        SignedE1SurvivorSelectionReceipt, ...
    ] = (),
    appended_e4_staged_evidence_manifests: tuple[E4StagedEvidenceManifest, ...] = (),
    appended_signed_e4_stage_selections: tuple[SignedE4StageSelectionReceipt, ...] = (),
    appended_signed_e3b_power_prefixes: tuple[SignedE3bPowerPrefixReceipt, ...] = (),
    appended_signed_e5_power_and_anchor_prefixes: tuple[
        SignedE5PowerAndAnchorReceipt, ...
    ] = (),
    appended_signed_e6_power_prefixes: tuple[SignedE6PowerPrefixReceipt, ...] = (),
    appended_e0_authority_bundles: tuple[E0FormalRegistryAuthorityBundle, ...] = (),
    appended_tts_calibration_authorities: tuple[TtsCalibrationAuthority, ...] = (),
    appended_signed_tts_calibration_seals: tuple[SignedTtsCalibrationSeal, ...] = (),
    formal_stage_prefix_artifact_paths: tuple[str, ...] = (),
    control_attestations: tuple[ControlArtifactAttestation, ...],
    candidate_replay_proof_artifact_paths: tuple[str, ...] = (),
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> FormalRegistryVerificationReceipt:
    """Append one signed DAG layer without replaying any prior challenge."""

    if type(prior_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("formal registry extension requires an exact prior receipt")
    prior_receipt.revalidate(current_ns=now_ns)
    appended_main_stages = {
        row.payload.stage for row in appended_signed_materializations
    }
    for stage, rows, expected_type in (
        (
            "E3b",
            appended_signed_e3b_power_prefixes,
            SignedE3bPowerPrefixReceipt,
        ),
        (
            "E5",
            appended_signed_e5_power_and_anchor_prefixes,
            SignedE5PowerAndAnchorReceipt,
        ),
        (
            "E6",
            appended_signed_e6_power_prefixes,
            SignedE6PowerPrefixReceipt,
        ),
    ):
        if (
            type(rows) is not tuple
            or any(type(row) is not expected_type for row in rows)
            or len(rows) not in {0, 1}
        ):
            raise TypeError(f"formal registry {stage} power sources are not exact")
        if (stage in appended_main_stages) != (len(rows) == 1):
            raise ValueError(
                f"{stage} main append requires exactly one signed power source"
            )
    if type(appended_e0_authority_bundles) is not tuple or any(
        type(row) is not E0FormalRegistryAuthorityBundle
        for row in appended_e0_authority_bundles
    ):
        raise TypeError("formal registry E0 authority bundles are not exact")
    appended_e0_materializations = tuple(
        row.payload
        for row in appended_signed_materializations
        if row.payload.stage == "E0"
    )
    if appended_e0_materializations or appended_e0_authority_bundles:
        if (
            len(appended_e0_materializations) != 1
            or len(appended_e0_authority_bundles) != 1
        ):
            raise ValueError("E0 main append requires one exact deep authority bundle")
        e0_bundle = appended_e0_authority_bundles[0]
        e0_bundle.verify_against(
            registry_verification_receipt=prior_receipt,
            materialization=appended_e0_materializations[0],
            now_ns=now_ns,
        )
        appended_e0_sources = (e0_bundle.source_authority,)
        appended_e0_compatibilities = (e0_bundle.signed_compatibility,)
        appended_e0_tuning_seals = e0_bundle.signed_tuning_seals
        appended_e0_power_prefixes = (e0_bundle.signed_power_prefix,)
    else:
        appended_e0_sources = ()
        appended_e0_compatibilities = ()
        appended_e0_tuning_seals = ()
        appended_e0_power_prefixes = ()
    appended_rows = (
        *appended_signed_materializations,
        *appended_signed_coverage,
    )
    if not appended_rows:
        raise ValueError("formal registry extension must append a signed row")
    if type(formal_stage_prefix_artifact_paths) is not tuple or any(
        type(path) is not str for path in formal_stage_prefix_artifact_paths
    ):
        raise TypeError("formal stage prefix artifact paths must be an exact tuple")
    prefix_bindings = tuple(
        CanonicalJsonProofBinding.bind(path)
        for path in formal_stage_prefix_artifact_paths
    )
    if len({row.absolute_path for row in prefix_bindings}) != len(prefix_bindings):
        raise ValueError("formal registry extension repeats a stage prefix artifact")
    _verify_formal_stage_prefix_append(
        prior_receipt=prior_receipt,
        prefix_bindings=prefix_bindings,
        appended_signed_materializations=appended_signed_materializations,
        appended_signed_coverage=appended_signed_coverage,
        appended_e2_staged_evidence_manifests=(appended_e2_staged_evidence_manifests),
        appended_signed_e2_staged_selections=(appended_signed_e2_staged_selections),
        appended_signed_e1_survivor_selections=(appended_signed_e1_survivor_selections),
        appended_e4_staged_evidence_manifests=(appended_e4_staged_evidence_manifests),
        appended_signed_e4_stage_selections=(appended_signed_e4_stage_selections),
        now_ns=now_ns,
    )
    if (
        tuple(
            CanonicalJsonProofBinding.bind(row.absolute_path) for row in prefix_bindings
        )
        != prefix_bindings
    ):
        raise RuntimeError("formal stage prefix artifact changed before reservation")
    prior_proof_paths = tuple(
        row.proof_artifact.absolute_path
        for row in prior_receipt.manifest.candidate_replay_proofs
    )
    all_proof_paths = (*prior_proof_paths, *candidate_replay_proof_artifact_paths)
    if len(all_proof_paths) != len(set(all_proof_paths)):
        raise ValueError("formal registry extension repeats a candidate replay proof")
    verification_times = dict(prior_receipt.verification_ns_by_signed_sha256)
    for row in appended_rows:
        if row.sha256 in verification_times:
            raise ValueError("formal registry extension repeats a signed row")
        verification_times[row.sha256] = now_ns
    for row in appended_signed_tts_calibration_seals:
        if row.sha256 in verification_times:
            raise ValueError("formal registry extension repeats a signed TTS seal")
        verification_times[row.sha256] = now_ns
    for row in appended_signed_e3a_staged_selections:
        if row.sha256 in verification_times:
            raise ValueError("formal registry extension repeats a signed E3a selection")
        verification_times[row.sha256] = now_ns
    for row in appended_signed_e2_staged_selections:
        if row.sha256 in verification_times:
            raise ValueError("formal registry extension repeats a signed E2 selection")
        verification_times[row.sha256] = now_ns
    for row in appended_signed_e1_survivor_selections:
        if row.sha256 in verification_times:
            raise ValueError("formal registry extension repeats a signed E1 selection")
        verification_times[row.sha256] = now_ns
    for row in appended_signed_e4_stage_selections:
        if row.sha256 in verification_times:
            raise ValueError("formal registry extension repeats a signed E4 selection")
        verification_times[row.sha256] = now_ns
    for row in (
        *appended_signed_e3b_power_prefixes,
        *appended_signed_e5_power_and_anchor_prefixes,
        *appended_signed_e6_power_prefixes,
    ):
        if row.sha256 in verification_times:
            raise ValueError("formal registry extension repeats a signed power source")
        verification_times[row.sha256] = now_ns
    for row in (
        *appended_e0_compatibilities,
        *appended_e0_tuning_seals,
        *appended_e0_power_prefixes,
    ):
        if row.sha256 in verification_times:
            raise ValueError("formal registry extension repeats a signed E0 authority")
        verification_times[row.sha256] = now_ns
    prepared = _prepare_formal_registry_manifest(
        prior_receipt.signed_protocol_lock,
        signed_materializations=(
            *prior_receipt.cumulative_signed_materializations,
            *appended_signed_materializations,
        ),
        signed_coverage=(
            *prior_receipt.cumulative_signed_coverage,
            *appended_signed_coverage,
        ),
        e3a_staged_selection_artifacts=(
            *prior_receipt.cumulative_e3a_staged_selection_artifacts,
            *appended_e3a_staged_selection_artifacts,
        ),
        signed_e3a_staged_selections=(
            *prior_receipt.cumulative_signed_e3a_staged_selections,
            *appended_signed_e3a_staged_selections,
        ),
        e2_staged_evidence_manifests=(
            *prior_receipt.cumulative_e2_staged_evidence_manifests,
            *appended_e2_staged_evidence_manifests,
        ),
        signed_e2_staged_selections=(
            *prior_receipt.cumulative_signed_e2_staged_selections,
            *appended_signed_e2_staged_selections,
        ),
        signed_e1_survivor_selections=(
            *prior_receipt.cumulative_signed_e1_survivor_selections,
            *appended_signed_e1_survivor_selections,
        ),
        e4_staged_evidence_manifests=(
            *prior_receipt.cumulative_e4_staged_evidence_manifests,
            *appended_e4_staged_evidence_manifests,
        ),
        signed_e4_stage_selections=(
            *prior_receipt.cumulative_signed_e4_stage_selections,
            *appended_signed_e4_stage_selections,
        ),
        signed_e3b_power_prefixes=(
            *prior_receipt.cumulative_signed_e3b_power_prefixes,
            *appended_signed_e3b_power_prefixes,
        ),
        signed_e5_power_and_anchor_prefixes=(
            *prior_receipt.cumulative_signed_e5_power_and_anchor_prefixes,
            *appended_signed_e5_power_and_anchor_prefixes,
        ),
        signed_e6_power_prefixes=(
            *prior_receipt.cumulative_signed_e6_power_prefixes,
            *appended_signed_e6_power_prefixes,
        ),
        e0_onlinespec_source_authorities=(
            *prior_receipt.cumulative_e0_onlinespec_source_authorities,
            *appended_e0_sources,
        ),
        signed_e0_compatibilities=(
            *prior_receipt.cumulative_signed_e0_compatibilities,
            *appended_e0_compatibilities,
        ),
        signed_e0_onlinespec_tuning_seals=(
            *prior_receipt.cumulative_signed_e0_onlinespec_tuning_seals,
            *appended_e0_tuning_seals,
        ),
        signed_e0_power_prefixes=(
            *prior_receipt.cumulative_signed_e0_power_prefixes,
            *appended_e0_power_prefixes,
        ),
        tts_calibration_authorities=(
            *prior_receipt.cumulative_tts_calibration_authorities,
            *appended_tts_calibration_authorities,
        ),
        signed_tts_calibration_seals=(
            *prior_receipt.cumulative_signed_tts_calibration_seals,
            *appended_signed_tts_calibration_seals,
        ),
        control_attestations=control_attestations,
        candidate_replay_proof_artifact_paths=all_proof_paths,
        controlled_signed_row_sha256s=frozenset(row.sha256 for row in appended_rows),
        controlled_signed_source_authority_sha256s=frozenset(
            row.sha256
            for row in (
                *appended_signed_e3a_staged_selections,
                *appended_signed_e2_staged_selections,
                *appended_signed_e1_survivor_selections,
                *appended_signed_e4_stage_selections,
                *appended_signed_e3b_power_prefixes,
                *appended_signed_e5_power_and_anchor_prefixes,
                *appended_signed_e6_power_prefixes,
                *appended_signed_tts_calibration_seals,
                *appended_e0_compatibilities,
                *appended_e0_tuning_seals,
                *appended_e0_power_prefixes,
            )
        ),
        prior_registry_verification_receipt_sha256=prior_receipt.sha256,
        verification_ns_by_signed_sha256=verification_times,
        expected_inventory_sha256=prior_receipt.inventory_sha256,
        now_ns=now_ns,
    )
    if (
        prepared.trusted_attester_policy_sha256
        != prior_receipt.manifest.trusted_attester_policy_sha256
    ):
        raise ValueError("formal registry extension changes attester policy")
    verified = verify_and_reserve_release_control_artifact_attestations(
        prepared.ordered_controls,
        expected_inventory_sha256=prior_receipt.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=prepared.additional_challenge_sha256s,
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified,
        additional_challenge_sha256s=prepared.additional_challenge_sha256s,
        reserved_ns=now_ns,
    )
    receipt = FormalRegistryVerificationReceipt(
        schema_version=2,
        kind="lightcone_formal_registry_verification_receipt",
        verified_ns=now_ns,
        inventory_sha256=prior_receipt.inventory_sha256,
        signed_protocol_lock=prior_receipt.signed_protocol_lock,
        prior_receipt=prior_receipt,
        appended_signed_materializations=appended_signed_materializations,
        appended_signed_coverage=appended_signed_coverage,
        appended_e3a_staged_selection_artifacts=(
            appended_e3a_staged_selection_artifacts
        ),
        appended_signed_e3a_staged_selections=(appended_signed_e3a_staged_selections),
        appended_e2_staged_evidence_manifests=(appended_e2_staged_evidence_manifests),
        appended_signed_e2_staged_selections=(appended_signed_e2_staged_selections),
        appended_signed_e1_survivor_selections=(appended_signed_e1_survivor_selections),
        appended_e4_staged_evidence_manifests=(appended_e4_staged_evidence_manifests),
        appended_signed_e4_stage_selections=(appended_signed_e4_stage_selections),
        appended_signed_e3b_power_prefixes=(appended_signed_e3b_power_prefixes),
        appended_signed_e5_power_and_anchor_prefixes=(
            appended_signed_e5_power_and_anchor_prefixes
        ),
        appended_signed_e6_power_prefixes=(appended_signed_e6_power_prefixes),
        appended_e0_onlinespec_source_authorities=appended_e0_sources,
        appended_signed_e0_compatibilities=appended_e0_compatibilities,
        appended_signed_e0_onlinespec_tuning_seals=appended_e0_tuning_seals,
        appended_signed_e0_power_prefixes=appended_e0_power_prefixes,
        appended_formal_stage_prefix_artifacts=prefix_bindings,
        appended_tts_calibration_authorities=(appended_tts_calibration_authorities),
        appended_signed_tts_calibration_seals=(appended_signed_tts_calibration_seals),
        control_attestations=control_attestations,
        reservation=replay_store.bind_reservation(reservation_sha256),
        manifest=prepared.manifest,
    )
    receipt.revalidate(current_ns=now_ns)
    return receipt


__all__ = [
    "FormalCandidateReplayProofBinding",
    "FormalControlReservation",
    "FormalCoverageBinding",
    "FormalMaterializationBinding",
    "FormalRegistryManifest",
    "FormalRegistryVerificationReceipt",
    "FormalSourceAuthorityBinding",
    "assemble_and_reserve_formal_registry_manifest",
    "challenge_from_dict",
    "e0_compatibility_receipt_from_dict",
    "e0_compatibility_receipt_to_dict",
    "e0_onlinespec_source_authority_from_dict",
    "e0_onlinespec_source_authority_to_dict",
    "e0_onlinespec_tuning_seal_from_dict",
    "e0_onlinespec_tuning_seal_to_dict",
    "e0_power_prefix_receipt_from_dict",
    "e0_power_prefix_receipt_to_dict",
    "e2_staged_evidence_manifest_from_dict",
    "e2_staged_evidence_manifest_to_dict",
    "e3a_staged_selection_artifact_from_dict",
    "e3a_staged_selection_artifact_to_dict",
    "e3b_power_prefix_receipt_from_dict",
    "e3b_power_prefix_receipt_to_dict",
    "e4_staged_evidence_manifest_from_dict",
    "e4_staged_evidence_manifest_to_dict",
    "e5_anchor_selection_receipt_from_dict",
    "e5_anchor_selection_receipt_to_dict",
    "e5_power_and_anchor_receipt_from_dict",
    "e5_power_and_anchor_receipt_to_dict",
    "e6_power_prefix_receipt_from_dict",
    "e6_power_prefix_receipt_to_dict",
    "extend_formal_registry_verification_receipt",
    "formal_registry_manifest_from_dict",
    "formal_registry_verification_receipt_from_dict",
    "formal_registry_verification_receipt_to_dict",
    "formal_runtime_authority_manifest_from_dict",
    "formal_runtime_authority_manifest_to_dict",
    "gpu_hour_estimate_from_dict",
    "gpu_hour_estimate_to_dict",
    "materialized_cell_from_dict",
    "materialized_cell_to_dict",
    "pilot_duration_receipt_from_dict",
    "pilot_duration_receipt_to_dict",
    "protocol_lock_from_dict",
    "protocol_lock_to_dict",
    "publish_e0_onlinespec_source_authority",
    "publish_formal_runtime_authority_manifest",
    "reserve_formal_registry_verification_receipt",
    "revalidate_formal_runtime_authority_manifest",
    "signed_attestation_from_dict",
    "signed_e0_compatibility_from_dict",
    "signed_e0_compatibility_to_dict",
    "signed_e0_onlinespec_tuning_seal_from_dict",
    "signed_e0_onlinespec_tuning_seal_to_dict",
    "signed_e0_power_prefix_from_dict",
    "signed_e0_power_prefix_to_dict",
    "signed_e1_survivor_selection_from_dict",
    "signed_e1_survivor_selection_to_dict",
    "signed_e2_staged_selection_from_dict",
    "signed_e2_staged_selection_to_dict",
    "signed_e3a_staged_selection_from_dict",
    "signed_e3a_staged_selection_to_dict",
    "signed_e3b_power_prefix_from_dict",
    "signed_e3b_power_prefix_to_dict",
    "signed_e4_stage_selection_from_dict",
    "signed_e4_stage_selection_to_dict",
    "signed_e5_anchor_selection_from_dict",
    "signed_e5_anchor_selection_to_dict",
    "signed_e5_power_and_anchor_from_dict",
    "signed_e5_power_and_anchor_to_dict",
    "signed_e6_power_prefix_from_dict",
    "signed_e6_power_prefix_to_dict",
    "signed_pilot_duration_from_dict",
    "signed_pilot_duration_to_dict",
    "signed_protocol_lock_from_dict",
    "signed_protocol_lock_to_dict",
    "signed_stage_coverage_from_dict",
    "signed_stage_coverage_to_dict",
    "signed_stage_gpu_hour_from_dict",
    "signed_stage_gpu_hour_to_dict",
    "signed_stage_materialization_from_dict",
    "signed_stage_materialization_to_dict",
    "signed_tts_calibration_seal_from_dict",
    "signed_tts_calibration_seal_to_dict",
    "stage_coverage_receipt_from_dict",
    "stage_coverage_receipt_to_dict",
    "stage_gpu_hour_envelope_from_dict",
    "stage_gpu_hour_envelope_to_dict",
    "stage_materialization_receipt_from_dict",
    "stage_materialization_receipt_to_dict",
    "tts_calibration_authority_from_dict",
    "tts_calibration_authority_to_dict",
    "tts_l0_candidate_state_coverage_from_dict",
    "tts_l0_candidate_state_coverage_to_dict",
]
