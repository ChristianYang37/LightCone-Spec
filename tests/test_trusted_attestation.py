from __future__ import annotations

import base64
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.runtime.attestation import (
    NO_TRUSTED_ATTESTERS,
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
    attestation_message,
    verify_attestation_signature,
)


def _signed(
    *, environment: str = "test"
) -> tuple[
    AttestationChallenge,
    SignedAttestation,
    str,
]:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id="challenge-1",
        nonce_base64=base64.b64encode(b"n" * 32).decode(),
        subject_sha256="a" * 64,
        issued_ns=100,
        expires_ns=1000,
    )
    payload_sha256 = "b" * 64
    signature = private.sign(
        attestation_message(challenge, payload_sha256=payload_sha256)
    )
    value = SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id="test-fixture-attester",
        key_id="ephemeral-test-key",
        environment=environment,
        public_key_base64=base64.b64encode(public).decode(),
        challenge_sha256=challenge.sha256,
        payload_sha256=payload_sha256,
        signature_base64=base64.b64encode(signature).decode(),
    )
    return challenge, value, payload_sha256


def test_nonce_bound_signature_verifies_cryptographically() -> None:
    challenge, attestation, payload = _signed()
    verify_attestation_signature(
        challenge,
        attestation,
        payload_sha256=payload,
        now_ns=500,
    )
    with pytest.raises(ValueError, match="challenge/payload"):
        verify_attestation_signature(
            challenge,
            attestation,
            payload_sha256="c" * 64,
            now_ns=500,
        )
    with pytest.raises(ValueError, match="expired"):
        verify_attestation_signature(
            challenge,
            attestation,
            payload_sha256=payload,
            now_ns=1001,
        )


def test_test_key_is_always_rejected_by_release_policy() -> None:
    challenge, attestation, payload = _signed()
    with pytest.raises(ValueError, match="test attestation"):
        NO_TRUSTED_ATTESTERS.verify_release(
            challenge,
            attestation,
            payload_sha256=payload,
            now_ns=500,
        )


def test_valid_signature_still_requires_an_explicit_release_allowlist() -> None:
    challenge, attestation, payload = _signed(environment="release")
    with pytest.raises(ValueError, match="not in the release policy"):
        NO_TRUSTED_ATTESTERS.verify_release(
            challenge,
            attestation,
            payload_sha256=payload,
            now_ns=500,
        )
    with pytest.raises(ValueError, match="policy entries"):
        TrustedAttesterPolicy(
            policy_id="bad",
            trusted_attesters=(
                (
                    attestation.attester_id,
                    attestation.key_id,
                    attestation.public_key_sha256,
                ),
            ),
        ).validate()


def test_signature_tamper_is_rejected() -> None:
    challenge, attestation, payload = _signed()
    signature = bytearray(base64.b64decode(attestation.signature_base64))
    signature[0] ^= 1
    tampered = replace(
        attestation,
        signature_base64=base64.b64encode(signature).decode(),
    )
    with pytest.raises(ValueError, match="signature is invalid"):
        verify_attestation_signature(
            challenge,
            tampered,
            payload_sha256=payload,
            now_ns=500,
        )
