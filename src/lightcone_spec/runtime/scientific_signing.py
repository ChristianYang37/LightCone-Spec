"""Closed, typed ceremony adapter for signed scientific authorities.

The experiment modules own every payload schema, signed wrapper, codec, and
verification rule.  This module only makes those existing APIs reachable from
an offline signing host.  It deliberately has no generic JSON signing path.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.experiments.formal_protocol import (
    content_sha256,
    verify_signed_payload,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    attestation_message,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    verify_source_signed_deployment_policy,
)
from lightcone_spec.runtime.scientific_source_validation import (
    PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES,
    rebuild_scientific_payload_from_source,
    revalidate_scientific_payload_source,
)

_CANDIDATE_KIND = "lightcone_offline_scientific_signature_candidate"
_PROOF_WRAPPER_KIND = "lightcone_scientific_signed_proof_wrapper"
_LEDGER_KIND = "lightcone_offline_scientific_challenge_reservation"
_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "artifact_type",
        "deployment_policy_authorization_sha256",
        "trusted_attester_policy_sha256",
        "source_validation_artifact",
        "payload_sha256",
        "challenge",
        "attestation",
        "signed_artifact_sha256",
        "signed_artifact",
        "finalized",
        "candidate_sha256",
    }
)


@dataclass(frozen=True)
class ScientificSigningSpec:
    artifact_type: str
    wrapper_type: type[Any]
    signed_decoder: Callable[[object], Any]
    signed_encoder: Callable[[Any], dict[str, object]]
    declared_digest_field: str | None
    payload_decoder: Callable[[object], Any] | None = None
    payload_encoder: Callable[[Any], dict[str, object]] | None = None


def _scientific_specs() -> Mapping[str, ScientificSigningSpec]:
    """Return the closed allowlist backed only by first-party typed codecs."""

    from lightcone_spec.experiments import e0_authority_artifact as e0_artifact
    from lightcone_spec.experiments import formal_registry as registry
    from lightcone_spec.experiments.breadth_fdr_authority import (
        SignedE0FormalBreadthFdrReceipt,
        formal_e0_breadth_fdr_receipt_from_dict,
        formal_e0_breadth_fdr_receipt_to_dict,
        signed_formal_e0_breadth_fdr_from_dict,
        signed_formal_e0_breadth_fdr_to_dict,
    )
    from lightcone_spec.experiments.downstream_stage_authority import (
        SignedE1aVerificationReceipt,
        SignedE3bConfirmationReceipt,
        SignedE3bPowerPrefixReceipt,
        SignedE5ConfirmationReceipt,
        SignedE5PowerAndAnchorReceipt,
    )
    from lightcone_spec.experiments.e0_stage_authority import (
        SignedE0OnlineSpecTuningSeal,
        SignedE0PowerPrefixReceipt,
    )
    from lightcone_spec.experiments.e2_stage_authority import (
        SignedE2StagedRoundSelectionReceipt,
    )
    from lightcone_spec.experiments.e3a_stage_authority import (
        SignedE3aStagedSelectionReceipt,
    )
    from lightcone_spec.experiments.e4_stage_authority import (
        SignedE4ProfilerCompletionReceipt,
        SignedE4StageSelectionReceipt,
        e4_profiler_completion_receipt_from_dict,
        e4_profiler_completion_receipt_to_dict,
        signed_e4_profiler_completion_from_dict,
        signed_e4_profiler_completion_to_dict,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        SignedE6ConfirmationReceipt,
        SignedE6ModelCompatibilityReceipt,
        SignedE6PowerPrefixReceipt,
    )
    from lightcone_spec.experiments.formal_protocol import (
        SignedProtocolLock,
        SignedTtsCalibrationSeal,
    )
    from lightcone_spec.experiments.stage_decisions import (
        SignedE1SurvivorSelectionReceipt,
    )
    from lightcone_spec.experiments.stage_materialization import (
        SignedE0CompatibilityReceipt,
        SignedE5AnchorSelectionReceipt,
        SignedPilotDurationReceipt,
        SignedStageCoverageReceipt,
        SignedStageGpuHourEnvelope,
        SignedStageMaterializationReceipt,
    )

    receipt = "signed_receipt_sha256"
    rows = (
        ScientificSigningSpec(
            "protocol-lock",
            SignedProtocolLock,
            registry.signed_protocol_lock_from_dict,
            registry.signed_protocol_lock_to_dict,
            None,
            registry.protocol_lock_from_dict,
            registry.protocol_lock_to_dict,
        ),
        ScientificSigningSpec(
            "tts-calibration-seal",
            SignedTtsCalibrationSeal,
            registry.signed_tts_calibration_seal_from_dict,
            registry.signed_tts_calibration_seal_to_dict,
            "signed_seal_sha256",
        ),
        ScientificSigningSpec(
            "stage-materialization",
            SignedStageMaterializationReceipt,
            registry.signed_stage_materialization_from_dict,
            registry.signed_stage_materialization_to_dict,
            receipt,
            registry.stage_materialization_receipt_from_dict,
            registry.stage_materialization_receipt_to_dict,
        ),
        ScientificSigningSpec(
            "stage-coverage",
            SignedStageCoverageReceipt,
            registry.signed_stage_coverage_from_dict,
            registry.signed_stage_coverage_to_dict,
            receipt,
            registry.stage_coverage_receipt_from_dict,
            registry.stage_coverage_receipt_to_dict,
        ),
        ScientificSigningSpec(
            "pilot-duration",
            SignedPilotDurationReceipt,
            registry.signed_pilot_duration_from_dict,
            registry.signed_pilot_duration_to_dict,
            receipt,
            registry.pilot_duration_receipt_from_dict,
            registry.pilot_duration_receipt_to_dict,
        ),
        ScientificSigningSpec(
            "stage-gpu-hour-envelope",
            SignedStageGpuHourEnvelope,
            registry.signed_stage_gpu_hour_from_dict,
            registry.signed_stage_gpu_hour_to_dict,
            "signed_envelope_sha256",
            registry.stage_gpu_hour_envelope_from_dict,
            registry.stage_gpu_hour_envelope_to_dict,
        ),
        ScientificSigningSpec(
            "e3a-staged-selection",
            SignedE3aStagedSelectionReceipt,
            registry.signed_e3a_staged_selection_from_dict,
            registry.signed_e3a_staged_selection_to_dict,
            receipt,
        ),
        ScientificSigningSpec(
            "e1-survivor-selection",
            SignedE1SurvivorSelectionReceipt,
            registry.signed_e1_survivor_selection_from_dict,
            registry.signed_e1_survivor_selection_to_dict,
            receipt,
        ),
        ScientificSigningSpec(
            "e2-staged-selection",
            SignedE2StagedRoundSelectionReceipt,
            registry.signed_e2_staged_selection_from_dict,
            registry.signed_e2_staged_selection_to_dict,
            receipt,
        ),
        ScientificSigningSpec(
            "e4-stage-selection",
            SignedE4StageSelectionReceipt,
            registry.signed_e4_stage_selection_from_dict,
            registry.signed_e4_stage_selection_to_dict,
            receipt,
        ),
        ScientificSigningSpec(
            "e4-profiler-completion",
            SignedE4ProfilerCompletionReceipt,
            signed_e4_profiler_completion_from_dict,
            signed_e4_profiler_completion_to_dict,
            receipt,
            e4_profiler_completion_receipt_from_dict,
            e4_profiler_completion_receipt_to_dict,
        ),
        ScientificSigningSpec(
            "e3b-power-prefix",
            SignedE3bPowerPrefixReceipt,
            registry.signed_e3b_power_prefix_from_dict,
            registry.signed_e3b_power_prefix_to_dict,
            receipt,
            registry.e3b_power_prefix_receipt_from_dict,
            registry.e3b_power_prefix_receipt_to_dict,
        ),
        ScientificSigningSpec(
            "e3b-confirmation",
            SignedE3bConfirmationReceipt,
            e0_artifact.signed_e3b_confirmation_from_dict,
            e0_artifact.signed_e3b_confirmation_to_dict,
            receipt,
        ),
        ScientificSigningSpec(
            "e1a-verification",
            SignedE1aVerificationReceipt,
            e0_artifact.signed_e1a_verification_from_dict,
            e0_artifact.signed_e1a_verification_to_dict,
            receipt,
        ),
        ScientificSigningSpec(
            "e5-power-and-anchor",
            SignedE5PowerAndAnchorReceipt,
            registry.signed_e5_power_and_anchor_from_dict,
            registry.signed_e5_power_and_anchor_to_dict,
            receipt,
            registry.e5_power_and_anchor_receipt_from_dict,
            registry.e5_power_and_anchor_receipt_to_dict,
        ),
        ScientificSigningSpec(
            "e5-anchor-selection",
            SignedE5AnchorSelectionReceipt,
            registry.signed_e5_anchor_selection_from_dict,
            registry.signed_e5_anchor_selection_to_dict,
            receipt,
            registry.e5_anchor_selection_receipt_from_dict,
            registry.e5_anchor_selection_receipt_to_dict,
        ),
        ScientificSigningSpec(
            "e5-confirmation",
            SignedE5ConfirmationReceipt,
            e0_artifact.signed_e5_confirmation_from_dict,
            e0_artifact.signed_e5_confirmation_to_dict,
            receipt,
        ),
        ScientificSigningSpec(
            "e6-model-compatibility",
            SignedE6ModelCompatibilityReceipt,
            e0_artifact.signed_e6_model_compatibility_from_dict,
            e0_artifact.signed_e6_model_compatibility_to_dict,
            receipt,
        ),
        ScientificSigningSpec(
            "e6-power-prefix",
            SignedE6PowerPrefixReceipt,
            registry.signed_e6_power_prefix_from_dict,
            registry.signed_e6_power_prefix_to_dict,
            receipt,
            registry.e6_power_prefix_receipt_from_dict,
            registry.e6_power_prefix_receipt_to_dict,
        ),
        ScientificSigningSpec(
            "e6-confirmation",
            SignedE6ConfirmationReceipt,
            e0_artifact.signed_e6_confirmation_from_dict,
            e0_artifact.signed_e6_confirmation_to_dict,
            receipt,
        ),
        ScientificSigningSpec(
            "e0-compatibility",
            SignedE0CompatibilityReceipt,
            registry.signed_e0_compatibility_from_dict,
            registry.signed_e0_compatibility_to_dict,
            receipt,
            registry.e0_compatibility_receipt_from_dict,
            registry.e0_compatibility_receipt_to_dict,
        ),
        ScientificSigningSpec(
            "e0-formal-breadth-fdr",
            SignedE0FormalBreadthFdrReceipt,
            signed_formal_e0_breadth_fdr_from_dict,
            signed_formal_e0_breadth_fdr_to_dict,
            receipt,
            formal_e0_breadth_fdr_receipt_from_dict,
            formal_e0_breadth_fdr_receipt_to_dict,
        ),
        ScientificSigningSpec(
            "e0-onlinespec-tuning-seal",
            SignedE0OnlineSpecTuningSeal,
            registry.signed_e0_onlinespec_tuning_seal_from_dict,
            registry.signed_e0_onlinespec_tuning_seal_to_dict,
            receipt,
            registry.e0_onlinespec_tuning_seal_from_dict,
            registry.e0_onlinespec_tuning_seal_to_dict,
        ),
        ScientificSigningSpec(
            "e0-power-prefix",
            SignedE0PowerPrefixReceipt,
            registry.signed_e0_power_prefix_from_dict,
            registry.signed_e0_power_prefix_to_dict,
            receipt,
            registry.e0_power_prefix_receipt_from_dict,
            registry.e0_power_prefix_receipt_to_dict,
        ),
        ScientificSigningSpec(
            "e0-final-completion",
            e0_artifact.SignedE0FinalCompletionReceipt,
            e0_artifact.signed_e0_final_completion_from_dict,
            e0_artifact.signed_e0_final_completion_to_dict,
            receipt,
            e0_artifact.e0_final_completion_receipt_from_dict,
            e0_artifact.e0_final_completion_receipt_to_dict,
        ),
    )
    # A typed decoder proves only that caller JSON has the right shape.  Formal
    # derived science is signable only when the source-validation dispatcher
    # can reopen its path-bound proof graph and rerun the owning reducer.  Keep
    # unsupported downstream/GPU-hour schemas out of the offline signer until
    # their reducer replay authorities are registered; otherwise a trusted key
    # could bless a structurally valid caller-authored result.
    signable = PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES
    return {row.artifact_type: row for row in rows if row.artifact_type in signable}


SCIENTIFIC_ARTIFACT_TYPES = tuple(_scientific_specs())


def _strict(value: object, *, fields: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a JSON object with string keys")
    if set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _json_tree(value: object) -> object:
    """Round-trip through strict JSON containers before calling JSON codecs."""

    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def _json_payload(value: object, *, artifact_type: str) -> dict[str, object]:
    value = _json_tree(value)
    row = _strict(
        value,
        fields=frozenset(value) if type(value) is dict else frozenset(),
        label=f"{artifact_type} payload",
    )
    if artifact_type != "protocol-lock" or set(row) != {
        "schema_version",
        "kind",
        "payload",
        "payload_sha256",
        "formal_dispatch_authorized",
    }:
        return row
    if (
        row["schema_version"] != 1
        or row["kind"] != "unsigned_protocol_lock_payload"
        or row["formal_dispatch_authorized"] is not False
    ):
        raise ValueError("unsigned ProtocolLock envelope is not exact")
    payload = _strict(
        row["payload"],
        fields=frozenset(row["payload"])
        if type(row["payload"]) is dict
        else frozenset(),
        label="unsigned ProtocolLock payload",
    )
    if row["payload_sha256"] != content_sha256(payload):
        raise ValueError("unsigned ProtocolLock digest differs from content")
    return payload


def _placeholder_challenge(payload_sha256: str) -> AttestationChallenge:
    return AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id="offline-typed-decode",
        nonce_base64=base64.b64encode(bytes(32)).decode("ascii"),
        subject_sha256=payload_sha256,
        issued_ns=1,
        expires_ns=2,
    )


def _placeholder_attestation(
    challenge: AttestationChallenge, payload_sha256: str
) -> SignedAttestation:
    return SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id="offline-typed-decoder",
        key_id="offline-typed-decoder-key",
        environment="release",
        public_key_base64=base64.b64encode(bytes(32)).decode("ascii"),
        challenge_sha256=challenge.sha256,
        payload_sha256=payload_sha256,
        signature_base64=base64.b64encode(bytes(64)).decode("ascii"),
    )


def _decode_payload(spec: ScientificSigningSpec, value: object) -> Any:
    raw_payload = _json_payload(value, artifact_type=spec.artifact_type)
    if spec.payload_decoder is not None:
        payload = spec.payload_decoder(raw_payload)
        if spec.payload_encoder is None:
            raise RuntimeError("typed payload codec registration is incomplete")
        if spec.payload_decoder(_json_tree(spec.payload_encoder(payload))) != payload:
            raise ValueError("typed payload codec round-trip is unstable")
        return payload

    payload_sha256 = content_sha256(raw_payload)
    challenge = _placeholder_challenge(payload_sha256)
    attestation = _placeholder_attestation(challenge, payload_sha256)
    wrapper_row: dict[str, object] = {
        "payload": raw_payload,
        "payload_sha256": payload_sha256,
        "challenge": asdict(challenge),
        "attestation": asdict(attestation),
    }
    if spec.declared_digest_field is not None:
        wrapper_row[spec.declared_digest_field] = content_sha256(wrapper_row)
    wrapper = spec.signed_decoder(wrapper_row)
    if type(wrapper) is not spec.wrapper_type:
        raise TypeError("typed signed decoder returned the wrong wrapper")
    if content_sha256(wrapper.payload) != payload_sha256:
        raise ValueError("typed payload differs from its canonical JSON")
    return wrapper.payload


def scientific_payload_sha256(
    *,
    artifact_type: str,
    payload_json: object,
    source_validation_artifact_path: str | os.PathLike[str] | None = None,
    now_ns: int | None = None,
) -> str:
    """Strictly decode an allowlisted payload and hash its typed content."""

    specs = _scientific_specs()
    if type(artifact_type) is not str or artifact_type not in specs:
        raise ValueError("scientific artifact type is not allowlisted")
    payload = _decode_payload(specs[artifact_type], payload_json)
    if artifact_type in PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES:
        if type(now_ns) is not int or now_ns < 1:
            raise ValueError(
                "proof-derived scientific payload validation time is required"
            )
        _validated_source_binding(
            artifact_type=artifact_type,
            payload=payload,
            source_validation_artifact_path=source_validation_artifact_path,
            now_ns=now_ns,
        )
    elif source_validation_artifact_path is not None:
        raise ValueError(
            "scientific source-validation artifact is not registered for this type"
        )
    return content_sha256(payload)


def scientific_payload_json_from_source(
    *,
    artifact_type: str,
    source_validation_artifact_path: str | os.PathLike[str],
    now_ns: int,
) -> dict[str, object]:
    """Rebuild and encode a proof-derived payload without a large input file."""

    if artifact_type not in PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES:
        raise ValueError("scientific source payload type is not proof-derived")
    _artifact, payload = rebuild_scientific_payload_from_source(
        source_validation_artifact_path,
        artifact_type=artifact_type,
        now_ns=now_ns,
    )
    spec = _scientific_specs()[artifact_type]
    encoded = (
        spec.payload_encoder(payload)
        if spec.payload_encoder is not None
        else asdict(payload)
    )
    normalized = _json_tree(encoded)
    if type(normalized) is not dict:
        raise TypeError("scientific source reducer payload is not an object")
    if _decode_payload(spec, normalized) != payload:
        raise ValueError("scientific source reducer payload codec differs")
    return normalized


def _validated_source_binding(
    *,
    artifact_type: str,
    payload: object,
    source_validation_artifact_path: str | os.PathLike[str] | None,
    now_ns: int,
) -> CanonicalJsonProofBinding | None:
    mandatory = artifact_type in PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES
    if source_validation_artifact_path is None:
        if mandatory:
            raise ValueError(
                "proof-derived scientific payload requires source validation"
            )
        return None
    if not mandatory:
        raise ValueError(
            "scientific source-validation artifact is not registered for this type"
        )
    path = Path(source_validation_artifact_path)
    revalidate_scientific_payload_source(
        path,
        artifact_type=artifact_type,
        payload=payload,
        now_ns=now_ns,
    )
    return CanonicalJsonProofBinding.bind(path)


def _verified_deployment(
    authorization: DeploymentPolicyAuthorization, *, now_ns: int
) -> Any:
    return verify_source_signed_deployment_policy(
        authorization,
        expected_inventory_sha256=authorization.inventory_sha256,
        now_ns=now_ns,
        consumed_challenge_sha256s=(),
    )


def _check_challenge_policy(
    challenge: AttestationChallenge,
    authorization: DeploymentPolicyAuthorization,
    *,
    now_ns: int,
) -> None:
    challenge.validate(now_ns=now_ns)
    policy = authorization.bundle.nonce_policy
    lifetime = challenge.expires_ns - challenge.issued_ns
    if not policy.minimum_lifetime_ns <= lifetime <= policy.maximum_lifetime_ns:
        raise ValueError("scientific challenge lifetime violates deployment policy")
    if challenge.issued_ns > now_ns + policy.maximum_clock_skew_ns:
        raise ValueError("scientific challenge was issued too far in the future")
    if challenge.sha256 == authorization.challenge.sha256:
        raise ValueError("deployment and scientific challenges must be distinct")


def _signed_artifact_sha256(wrapper: Any) -> str:
    digest = getattr(wrapper, "sha256", None)
    if type(digest) is not str or len(digest) != 64:
        raise TypeError("typed signed wrapper has no exact SHA-256")
    return digest


def sign_scientific_candidate(
    *,
    artifact_type: str,
    payload_json: object,
    deployment_policy_authorization: DeploymentPolicyAuthorization,
    challenge: AttestationChallenge,
    private_key: Ed25519PrivateKey,
    attester_id: str,
    key_id: str,
    now_ns: int,
    source_validation_artifact_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Create a typed, non-final candidate for one allowlisted authority."""

    specs = _scientific_specs()
    if type(artifact_type) is not str or artifact_type not in specs:
        raise ValueError("scientific artifact type is not allowlisted")
    if type(deployment_policy_authorization) is not DeploymentPolicyAuthorization:
        raise TypeError("scientific signing requires an exact deployment policy")
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("scientific signing requires an Ed25519 key")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("scientific signing time is invalid")
    spec = specs[artifact_type]
    payload = _decode_payload(spec, payload_json)
    source_validation = _validated_source_binding(
        artifact_type=artifact_type,
        payload=payload,
        source_validation_artifact_path=source_validation_artifact_path,
        now_ns=now_ns,
    )
    payload_sha256 = content_sha256(payload)
    if challenge.subject_sha256 != payload_sha256:
        raise ValueError("scientific challenge is not payload-bound")
    verified = _verified_deployment(deployment_policy_authorization, now_ns=now_ns)
    _check_challenge_policy(challenge, deployment_policy_authorization, now_ns=now_ns)
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    identity = (attester_id, key_id, public_key_sha256)
    policy = verified.bundle.trusted_attester_policy
    if identity not in policy.trusted_attesters:
        raise ValueError("scientific signer identity is not authorized")
    attestation = SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id=attester_id,
        key_id=key_id,
        environment="release",
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
        challenge_sha256=challenge.sha256,
        payload_sha256=payload_sha256,
        signature_base64=base64.b64encode(
            private_key.sign(
                attestation_message(challenge, payload_sha256=payload_sha256)
            )
        ).decode("ascii"),
    )
    wrapper = spec.wrapper_type(
        payload=payload,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=attestation,
    )
    encoded = _json_tree(spec.signed_encoder(wrapper))
    if type(encoded) is not dict:
        raise TypeError("typed signed encoder did not return a JSON object")
    decoded = spec.signed_decoder(encoded)
    if type(decoded) is not spec.wrapper_type or decoded != wrapper:
        raise ValueError("typed signed wrapper codec round-trip is unstable")
    verify_signed_payload(
        decoded.payload,
        payload_sha256=decoded.payload_sha256,
        challenge=decoded.challenge,
        attestation=decoded.attestation,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "kind": _CANDIDATE_KIND,
        "artifact_type": artifact_type,
        "deployment_policy_authorization_sha256": (
            deployment_policy_authorization.sha256
        ),
        "trusted_attester_policy_sha256": policy.sha256,
        "source_validation_artifact": (
            None if source_validation is None else source_validation.to_dict()
        ),
        "payload_sha256": payload_sha256,
        "challenge": asdict(challenge),
        "attestation": asdict(attestation),
        "signed_artifact_sha256": _signed_artifact_sha256(decoded),
        "signed_artifact": None if source_validation is not None else encoded,
        "finalized": False,
    }
    return {**body, "candidate_sha256": content_sha256(body)}


def verify_scientific_candidate(
    *,
    artifact_type: str,
    candidate_json: object,
    deployment_policy_authorization: DeploymentPolicyAuthorization,
    now_ns: int,
) -> Any:
    """Strictly decode and verify one candidate without consuming it."""

    specs = _scientific_specs()
    if type(artifact_type) is not str or artifact_type not in specs:
        raise ValueError("scientific artifact type is not allowlisted")
    row = _strict(
        candidate_json,
        fields=_CANDIDATE_FIELDS,
        label="scientific signature candidate",
    )
    declared = row.pop("candidate_sha256")
    if declared != content_sha256(row):
        raise ValueError("scientific candidate digest differs from content")
    if (
        row["schema_version"] != 1
        or row["kind"] != _CANDIDATE_KIND
        or row["artifact_type"] != artifact_type
        or row["finalized"] is not False
    ):
        raise ValueError("scientific candidate identity is not exact")
    verified = _verified_deployment(deployment_policy_authorization, now_ns=now_ns)
    policy = verified.bundle.trusted_attester_policy
    if (
        row["deployment_policy_authorization_sha256"]
        != deployment_policy_authorization.sha256
        or row["trusted_attester_policy_sha256"] != policy.sha256
    ):
        raise ValueError("scientific candidate root or policy binding differs")
    spec = specs[artifact_type]
    source_value = row["source_validation_artifact"]
    mandatory_source = artifact_type in PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES
    if (source_value is None) != (not mandatory_source):
        raise ValueError("scientific candidate source-validation coverage differs")
    challenge_value = row["challenge"]
    attestation_value = row["attestation"]
    if (
        type(challenge_value) is not dict
        or set(challenge_value) != set(AttestationChallenge.__dataclass_fields__)
        or type(attestation_value) is not dict
        or set(attestation_value) != set(SignedAttestation.__dataclass_fields__)
    ):
        raise ValueError("scientific candidate signature header fields differ")
    challenge = AttestationChallenge(**challenge_value)
    attestation = SignedAttestation(**attestation_value)
    if source_value is not None:
        if row["signed_artifact"] is not None:
            raise ValueError("proof-derived scientific candidate embeds its payload")
        source = CanonicalJsonProofBinding.from_dict(source_value)
        if CanonicalJsonProofBinding.bind(source.absolute_path) != source:
            raise ValueError("scientific candidate source-validation identity changed")
        _artifact, payload = rebuild_scientific_payload_from_source(
            source.absolute_path,
            artifact_type=artifact_type,
            now_ns=now_ns,
        )
        wrapper = spec.wrapper_type(
            payload=payload,
            payload_sha256=row["payload_sha256"],
            challenge=challenge,
            attestation=attestation,
        )
    else:
        wrapper = spec.signed_decoder(row["signed_artifact"])
        if (
            wrapper.payload_sha256 != row["payload_sha256"]
            or wrapper.challenge != challenge
            or wrapper.attestation != attestation
        ):
            raise ValueError("scientific candidate signature header differs")
    if type(wrapper) is not spec.wrapper_type:
        raise TypeError("scientific candidate has the wrong signed wrapper type")
    if row["signed_artifact_sha256"] != _signed_artifact_sha256(wrapper):
        raise ValueError("scientific candidate wrapper digest differs")
    _check_challenge_policy(
        wrapper.challenge, deployment_policy_authorization, now_ns=now_ns
    )
    verify_signed_payload(
        wrapper.payload,
        payload_sha256=wrapper.payload_sha256,
        challenge=wrapper.challenge,
        attestation=wrapper.attestation,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if spec.signed_decoder(_json_tree(spec.signed_encoder(wrapper))) != wrapper:
        raise ValueError("scientific signed wrapper codec round-trip is unstable")
    return wrapper


def _private_ledger_directory(path: str | os.PathLike[str]) -> Path:
    directory = Path(path)
    if not directory.is_absolute() or Path(os.path.abspath(directory)) != directory:
        raise ValueError("scientific challenge ledger must be absolute and normalized")
    status = directory.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise ValueError("scientific challenge ledger must be current-user 0700")
    return directory


def finalize_scientific_candidate(
    *,
    artifact_type: str,
    candidate_json: object,
    deployment_policy_authorization: DeploymentPolicyAuthorization,
    challenge_ledger: str | os.PathLike[str],
    now_ns: int,
) -> dict[str, object]:
    """Verify, single-use reserve, and return the exact formal signed wrapper."""

    wrapper = verify_scientific_candidate(
        artifact_type=artifact_type,
        candidate_json=candidate_json,
        deployment_policy_authorization=deployment_policy_authorization,
        now_ns=now_ns,
    )
    candidate = _strict(
        candidate_json,
        fields=_CANDIDATE_FIELDS,
        label="scientific signature candidate",
    )
    directory = _private_ledger_directory(challenge_ledger)
    challenge_id_sha256 = hashlib.sha256(
        wrapper.challenge.challenge_id.encode("utf-8")
    ).hexdigest()
    reservation = {
        "schema_version": 1,
        "kind": _LEDGER_KIND,
        "artifact_type": artifact_type,
        "challenge_id": wrapper.challenge.challenge_id,
        "challenge_sha256": wrapper.challenge.sha256,
        "subject_sha256": wrapper.challenge.subject_sha256,
        "candidate_sha256": candidate["candidate_sha256"],
        "signed_artifact_sha256": _signed_artifact_sha256(wrapper),
    }
    publish_canonical_json_no_replace(
        directory / f"challenge-{challenge_id_sha256}.json",
        reservation,
    )
    if artifact_type in PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES:
        body: dict[str, object] = {
            "schema_version": 1,
            "kind": _PROOF_WRAPPER_KIND,
            "artifact_type": artifact_type,
            "source_validation_artifact": candidate["source_validation_artifact"],
            "payload_sha256": wrapper.payload_sha256,
            "challenge": asdict(wrapper.challenge),
            "attestation": asdict(wrapper.attestation),
            "signed_artifact_sha256": _signed_artifact_sha256(wrapper),
        }
        return {**body, "wrapper_sha256": content_sha256(body)}
    encoded = _json_tree(_scientific_specs()[artifact_type].signed_encoder(wrapper))
    if type(encoded) is not dict:
        raise TypeError("typed signed encoder did not return a JSON object")
    return encoded


def rebuild_scientific_signed_proof_wrapper(
    wrapper_path: str | os.PathLike[str],
    *,
    now_ns: int,
) -> Any:
    """Deep-replay a compact finalized proof wrapper into its typed original."""

    binding = CanonicalJsonProofBinding.bind(wrapper_path)
    row = _strict(
        binding.reopen(),
        fields=frozenset(
            {
                "schema_version",
                "kind",
                "artifact_type",
                "source_validation_artifact",
                "payload_sha256",
                "challenge",
                "attestation",
                "signed_artifact_sha256",
                "wrapper_sha256",
            }
        ),
        label="scientific signed proof wrapper",
    )
    declared = row.pop("wrapper_sha256")
    if declared != content_sha256(row):
        raise ValueError("scientific signed proof wrapper digest differs")
    artifact_type = row["artifact_type"]
    if (
        row["schema_version"] != 1
        or row["kind"] != _PROOF_WRAPPER_KIND
        or artifact_type not in PROOF_REPLAY_SCIENTIFIC_ARTIFACT_TYPES
    ):
        raise ValueError("scientific signed proof wrapper identity differs")
    source = CanonicalJsonProofBinding.from_dict(row["source_validation_artifact"])
    if CanonicalJsonProofBinding.bind(source.absolute_path) != source:
        raise ValueError("scientific signed proof source identity changed")
    artifact, payload = rebuild_scientific_payload_from_source(
        source.absolute_path,
        artifact_type=artifact_type,
        now_ns=now_ns,
    )
    if artifact.expected_payload_sha256 != row["payload_sha256"]:
        raise ValueError("scientific signed proof wrapper payload differs")
    challenge_value = row["challenge"]
    attestation_value = row["attestation"]
    if type(challenge_value) is not dict or type(attestation_value) is not dict:
        raise TypeError("scientific signed proof wrapper signature is invalid")
    challenge = AttestationChallenge(**challenge_value)
    attestation = SignedAttestation(**attestation_value)
    spec = _scientific_specs()[artifact_type]
    wrapper = spec.wrapper_type(
        payload=payload,
        payload_sha256=row["payload_sha256"],
        challenge=challenge,
        attestation=attestation,
    )
    if _signed_artifact_sha256(wrapper) != row["signed_artifact_sha256"]:
        raise ValueError("scientific signed proof wrapper signature differs")
    if CanonicalJsonProofBinding.bind(wrapper_path) != binding:
        raise RuntimeError("scientific signed proof wrapper changed")
    return wrapper


__all__ = [
    "SCIENTIFIC_ARTIFACT_TYPES",
    "ScientificSigningSpec",
    "finalize_scientific_candidate",
    "rebuild_scientific_signed_proof_wrapper",
    "scientific_payload_json_from_source",
    "scientific_payload_sha256",
    "sign_scientific_candidate",
    "verify_scientific_candidate",
]
