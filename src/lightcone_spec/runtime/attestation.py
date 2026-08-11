"""Nonce-bound Ed25519 attestation verification.

This module contains verification and challenge generation only.  Private keys
are provisioned out of band and must never enter the repository or evidence
artifacts.  Cryptographically valid test signatures remain categorically
ineligible for release claims unless their public-key digest is present in an
explicit release policy.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import secrets
import time
from dataclasses import asdict, dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_NONCE_BYTES = 32


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _decode_base64(name: str, value: str, *, expected_length: int) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be canonical base64") from error
    if len(decoded) != expected_length or base64.b64encode(decoded).decode() != value:
        raise ValueError(f"{name} has the wrong length or noncanonical encoding")
    return decoded


@dataclass(frozen=True)
class AttestationChallenge:
    schema_version: int
    kind: str
    challenge_id: str
    nonce_base64: str
    subject_sha256: str
    issued_ns: int
    expires_ns: int

    @classmethod
    def issue(
        cls,
        *,
        challenge_id: str,
        subject_sha256: str,
        lifetime_s: float = 300.0,
        now_ns: int | None = None,
    ) -> AttestationChallenge:
        if now_ns is None:
            now_ns = time.time_ns()
        if not math.isfinite(lifetime_s) or lifetime_s <= 0:
            raise ValueError("challenge lifetime must be finite and positive")
        value = cls(
            schema_version=1,
            kind="lightcone_attestation_challenge",
            challenge_id=challenge_id,
            nonce_base64=base64.b64encode(secrets.token_bytes(_NONCE_BYTES)).decode(),
            subject_sha256=subject_sha256,
            issued_ns=now_ns,
            expires_ns=now_ns + round(lifetime_s * 1_000_000_000),
        )
        value.validate(now_ns=now_ns)
        return value

    def validate(self, *, now_ns: int | None = None) -> None:
        if self.schema_version != 1 or self.kind != "lightcone_attestation_challenge":
            raise ValueError("attestation challenge schema is unsupported")
        if not _SAFE_ID.fullmatch(self.challenge_id):
            raise ValueError("attestation challenge ID is unsafe")
        _decode_base64(
            "challenge nonce", self.nonce_base64, expected_length=_NONCE_BYTES
        )
        if not _SHA256.fullmatch(self.subject_sha256):
            raise ValueError("attestation subject must be a lowercase SHA-256")
        if (
            isinstance(self.issued_ns, bool)
            or isinstance(self.expires_ns, bool)
            or self.issued_ns < 0
            or self.expires_ns <= self.issued_ns
        ):
            raise ValueError("attestation challenge lifetime is invalid")
        if now_ns is not None and now_ns > self.expires_ns:
            raise ValueError("attestation challenge expired")

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256(asdict(self))


@dataclass(frozen=True)
class SignedAttestation:
    schema_version: int
    kind: str
    algorithm: str
    attester_id: str
    key_id: str
    environment: str
    public_key_base64: str
    challenge_sha256: str
    payload_sha256: str
    signature_base64: str

    def validate(self) -> None:
        if self.schema_version != 1 or self.kind != "lightcone_signed_attestation":
            raise ValueError("signed-attestation schema is unsupported")
        if self.algorithm != "Ed25519":
            raise ValueError("only Ed25519 attestations are supported")
        for name in ("attester_id", "key_id"):
            if not _SAFE_ID.fullmatch(getattr(self, name)):
                raise ValueError(f"attestation {name} is unsafe")
        if self.environment not in {"test", "release"}:
            raise ValueError("attestation environment must be test or release")
        _decode_base64("public key", self.public_key_base64, expected_length=32)
        _decode_base64("signature", self.signature_base64, expected_length=64)
        for name in ("challenge_sha256", "payload_sha256"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")

    @property
    def public_key_sha256(self) -> str:
        self.validate()
        return hashlib.sha256(
            _decode_base64(
                "public key",
                self.public_key_base64,
                expected_length=32,
            )
        ).hexdigest()

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256(asdict(self))


def attestation_message(
    challenge: AttestationChallenge,
    *,
    payload_sha256: str,
) -> bytes:
    challenge.validate()
    if not _SHA256.fullmatch(payload_sha256):
        raise ValueError("attestation payload must be a lowercase SHA-256")
    return _canonical_bytes(
        {
            "schema_version": 1,
            "domain": "lightcone-spec-hardware-attestation",
            "challenge_sha256": challenge.sha256,
            "nonce_base64": challenge.nonce_base64,
            "subject_sha256": challenge.subject_sha256,
            "payload_sha256": payload_sha256,
        }
    )


def verify_attestation_signature(
    challenge: AttestationChallenge,
    attestation: SignedAttestation,
    *,
    payload_sha256: str,
    now_ns: int | None = None,
) -> None:
    challenge.validate(now_ns=time.time_ns() if now_ns is None else now_ns)
    attestation.validate()
    if (
        attestation.challenge_sha256 != challenge.sha256
        or attestation.payload_sha256 != payload_sha256
    ):
        raise ValueError("signed attestation differs from challenge/payload")
    public_key = Ed25519PublicKey.from_public_bytes(
        _decode_base64(
            "public key",
            attestation.public_key_base64,
            expected_length=32,
        )
    )
    try:
        public_key.verify(
            _decode_base64(
                "signature",
                attestation.signature_base64,
                expected_length=64,
            ),
            attestation_message(challenge, payload_sha256=payload_sha256),
        )
    except InvalidSignature as error:
        raise ValueError("signed attestation signature is invalid") from error


@dataclass(frozen=True)
class TrustedAttesterPolicy:
    """Release allowlist; an empty policy deliberately permits no GPU claim."""

    policy_id: str
    trusted_attesters: tuple[tuple[str, str, str], ...]

    def validate(self) -> None:
        if not _SAFE_ID.fullmatch(self.policy_id):
            raise ValueError("trusted-attester policy ID is unsafe")
        identities = tuple(row[0] for row in self.trusted_attesters)
        keys = tuple((row[0], row[1]) for row in self.trusted_attesters)
        if (
            keys != tuple(sorted(set(keys)))
            or any(
                not _SAFE_ID.fullmatch(attester_id)
                or not _SAFE_ID.fullmatch(key_id)
                or not _SHA256.fullmatch(public_key_sha256)
                for attester_id, key_id, public_key_sha256 in self.trusted_attesters
            )
            or any(identity.lower().startswith("test") for identity in identities)
        ):
            raise ValueError("trusted-attester policy entries are invalid")

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256(
            {
                "schema_version": 1,
                "policy_id": self.policy_id,
                "trusted_attesters": [list(row) for row in self.trusted_attesters],
            }
        )

    def verify_release(
        self,
        challenge: AttestationChallenge,
        attestation: SignedAttestation,
        *,
        payload_sha256: str,
        now_ns: int | None = None,
    ) -> None:
        self.validate()
        verify_attestation_signature(
            challenge,
            attestation,
            payload_sha256=payload_sha256,
            now_ns=now_ns,
        )
        if attestation.environment != "release":
            raise ValueError(
                "test attestation identities cannot support release claims"
            )
        expected = (
            attestation.attester_id,
            attestation.key_id,
            attestation.public_key_sha256,
        )
        if expected not in self.trusted_attesters:
            raise ValueError("attester public key is not in the release policy")


NO_TRUSTED_ATTESTERS = TrustedAttesterPolicy(
    policy_id="lightcone-release-no-trusted-attester-v1",
    trusted_attesters=(),
)
