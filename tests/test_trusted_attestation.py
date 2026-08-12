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
    require_release_trusted_attester_policy,
    verify_attestation_signature,
)


def _signed(
    *,
    environment: str = "test",
    attester_id: str = "test-fixture-attester",
    key_id: str = "ephemeral-test-key",
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
        attester_id=attester_id,
        key_id=key_id,
        environment=environment,
        public_key_base64=base64.b64encode(public).decode(),
        challenge_sha256=challenge.sha256,
        payload_sha256=payload_sha256,
        signature_base64=base64.b64encode(signature).decode(),
    )
    return challenge, value, payload_sha256


def _release_policy(attestation: SignedAttestation) -> TrustedAttesterPolicy:
    policy = TrustedAttesterPolicy(
        policy_id="release-attesters-v1",
        trusted_attesters=(
            (
                attestation.attester_id,
                attestation.key_id,
                attestation.public_key_sha256,
            ),
        ),
        public_keys=(
            (
                attestation.public_key_sha256,
                attestation.public_key_base64,
            ),
        ),
    )
    policy.validate()
    return policy


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
    challenge, attestation, payload = _signed(
        environment="release",
        attester_id="prod-hsm-1",
        key_id="release-key-1",
    )
    with pytest.raises(ValueError, match="not in the release policy"):
        NO_TRUSTED_ATTESTERS.verify_release(
            challenge,
            attestation,
            payload_sha256=payload,
            now_ns=500,
        )
    with pytest.raises(ValueError, match="release-owned public keys differ"):
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


def test_release_policy_owns_and_verifies_the_allowlisted_ed25519_key() -> None:
    challenge, attestation, payload = _signed(
        environment="release",
        attester_id="prod-hsm-1",
        key_id="release-key-1",
    )
    policy = _release_policy(attestation)

    assert policy.release_ready is True
    assert policy.allows_terminal_attester(attestation.attester_id) is True
    policy.verify_release(
        challenge,
        attestation,
        payload_sha256=payload,
        now_ns=500,
    )

    _, replacement, replacement_payload = _signed(
        environment="release",
        attester_id=attestation.attester_id,
        key_id=attestation.key_id,
    )
    with pytest.raises(ValueError, match="not in the release policy"):
        policy.verify_release(
            challenge,
            replace(
                replacement,
                challenge_sha256=challenge.sha256,
                payload_sha256=replacement_payload,
            ),
            payload_sha256=replacement_payload,
            now_ns=500,
        )


def test_caller_owned_valid_policy_is_not_the_release_trust_root() -> None:
    _, attestation, _ = _signed(
        environment="release",
        attester_id="prod-hsm-1",
        key_id="release-key-1",
    )
    caller_policy = _release_policy(attestation)

    with pytest.raises(ValueError, match="caller-supplied"):
        require_release_trusted_attester_policy(caller_policy)

    assert require_release_trusted_attester_policy(NO_TRUSTED_ATTESTERS) is (
        NO_TRUSTED_ATTESTERS
    )


@pytest.mark.parametrize(
    ("attester_id", "key_id"),
    [
        ("test-hsm-1", "release-key-1"),
        ("fixture-hsm-1", "release-key-1"),
        ("cpu-signer-1", "release-key-1"),
        ("prod-hsm-1", "test-key-1"),
        ("prod-hsm-1", "cpu-key-1"),
    ],
)
def test_test_fixture_and_cpu_identities_cannot_enter_release_policy(
    attester_id: str,
    key_id: str,
) -> None:
    _, attestation, _ = _signed(
        environment="release",
        attester_id=attester_id,
        key_id=key_id,
    )
    with pytest.raises(ValueError, match="policy entries are invalid"):
        _release_policy(attestation)


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
