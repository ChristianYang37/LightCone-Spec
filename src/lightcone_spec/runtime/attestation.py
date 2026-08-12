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
_LOWER_HEX = re.compile(r"[0-9a-f]+\Z")
_NONCE_BYTES = 32
_FORBIDDEN_RELEASE_PREFIXES = ("test", "fixture", "cpu")


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


def _forbidden_release_identity(value: str) -> bool:
    return value.lower().startswith(_FORBIDDEN_RELEASE_PREFIXES)


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
    # Terminal wire evidence intentionally never carries a public key.  The
    # release policy therefore owns the key bytes as well as the allowlisted
    # digest; accepting key material from the terminal envelope would recreate
    # a caller-selected trust root.  Rows are ``(public_key_sha256, base64)``.
    public_keys: tuple[tuple[str, str], ...] = ()

    def validate(self) -> None:
        if not _SAFE_ID.fullmatch(self.policy_id):
            raise ValueError("trusted-attester policy ID is unsafe")
        identities = tuple(row[0] for row in self.trusted_attesters)
        key_ids = tuple(row[1] for row in self.trusted_attesters)
        keys = tuple((row[0], row[1]) for row in self.trusted_attesters)
        public_key_digests = tuple(row[0] for row in self.public_keys)
        if (
            keys != tuple(sorted(set(keys)))
            or identities != tuple(sorted(set(identities)))
            or key_ids != tuple(sorted(set(key_ids)))
            or any(
                not _SAFE_ID.fullmatch(attester_id)
                or not _SAFE_ID.fullmatch(key_id)
                or not _SHA256.fullmatch(public_key_sha256)
                or _forbidden_release_identity(attester_id)
                or _forbidden_release_identity(key_id)
                for attester_id, key_id, public_key_sha256 in self.trusted_attesters
            )
        ):
            raise ValueError("trusted-attester policy entries are invalid")
        if public_key_digests != tuple(sorted(set(public_key_digests))):
            raise ValueError("trusted-attester public keys are duplicated or unsorted")
        decoded_keys: dict[str, bytes] = {}
        for public_key_sha256, public_key_base64 in self.public_keys:
            if not _SHA256.fullmatch(public_key_sha256):
                raise ValueError("trusted-attester public-key digest is invalid")
            public_key = _decode_base64(
                "trusted-attester public key",
                public_key_base64,
                expected_length=32,
            )
            if hashlib.sha256(public_key).hexdigest() != public_key_sha256:
                raise ValueError("trusted-attester public key differs from its digest")
            decoded_keys[public_key_sha256] = public_key
        required_digests = {row[2] for row in self.trusted_attesters}
        if set(decoded_keys) != required_digests:
            raise ValueError(
                "trusted-attester allowlist and release-owned public keys differ"
            )

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the public, content-addressed release verification policy."""

        self.validate()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "trusted_attesters": [list(row) for row in self.trusted_attesters],
            "public_keys": [list(row) for row in self.public_keys],
        }

    @property
    def release_ready(self) -> bool:
        self.validate()
        return bool(self.trusted_attesters)

    def allows_terminal_attester(self, attester_id: str) -> bool:
        """Return whether terminal verification may select this policy identity."""

        self.validate()
        return (
            isinstance(attester_id, str)
            and not _forbidden_release_identity(attester_id)
            and sum(row[0] == attester_id for row in self.trusted_attesters) == 1
        )

    def _release_key(self, attester_id: str) -> tuple[str, str, bytes]:
        self.validate()
        if not _SAFE_ID.fullmatch(attester_id) or _forbidden_release_identity(
            attester_id
        ):
            raise ValueError("test, fixture, and CPU attesters cannot be trusted")
        matches = tuple(row for row in self.trusted_attesters if row[0] == attester_id)
        if len(matches) != 1:
            raise ValueError("attester identity is not uniquely allowlisted")
        _, key_id, public_key_sha256 = matches[0]
        public_key_base64 = dict(self.public_keys)[public_key_sha256]
        return (
            key_id,
            public_key_sha256,
            _decode_base64(
                "trusted-attester public key",
                public_key_base64,
                expected_length=32,
            ),
        )

    def verify_terminal_signature(
        self,
        *,
        attester_id: str,
        trust_domain: str,
        message: bytes,
        signature_hex: str,
    ) -> tuple[str, str]:
        """Verify the pinned terminal wire message against this release policy.

        The terminal hook exposes only an attester ID and signature.  Key ID and
        public key are selected exclusively from this immutable policy.
        """

        if trust_domain != "hardware":
            raise ValueError("only hardware terminal attestations can be trusted")
        if not isinstance(message, bytes) or not message:
            raise ValueError("terminal attestation message must be non-empty bytes")
        if (
            not isinstance(signature_hex, str)
            or len(signature_hex) != 128
            or _LOWER_HEX.fullmatch(signature_hex) is None
        ):
            raise ValueError("terminal Ed25519 signature is not canonical")
        key_id, public_key_sha256, public_key = self._release_key(attester_id)
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                bytes.fromhex(signature_hex),
                message,
            )
        except InvalidSignature as error:
            raise ValueError("terminal Ed25519 signature is invalid") from error
        return key_id, public_key_sha256

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
        if _forbidden_release_identity(
            attestation.attester_id
        ) or _forbidden_release_identity(attestation.key_id):
            raise ValueError("test, fixture, and CPU identities cannot support claims")
        expected = (
            attestation.attester_id,
            attestation.key_id,
            attestation.public_key_sha256,
        )
        if expected not in self.trusted_attesters:
            raise ValueError("attester public key is not in the release policy")
        _, expected_digest, expected_public_key = self._release_key(
            attestation.attester_id
        )
        observed_public_key = _decode_base64(
            "public key", attestation.public_key_base64, expected_length=32
        )
        if (
            expected_digest != attestation.public_key_sha256
            or expected_public_key != observed_public_key
        ):
            raise ValueError("signed attestation does not use the release-owned key")


NO_TRUSTED_ATTESTERS = TrustedAttesterPolicy(
    policy_id="lightcone-release-no-trusted-attester-v1",
    trusted_attesters=(),
)

# This is the trust root compiled into the current source release.  Public-key
# verification remains reusable as a library primitive, but formal execution
# and reduction must never promote an arbitrary caller-supplied policy into a
# release authority.  A future hardware rollout must change this constant in a
# reviewed source release (and update the runtime manifest) before any signer
# can unlock claims.
RELEASE_TRUSTED_ATTESTER_POLICY = NO_TRUSTED_ATTESTERS


def require_release_trusted_attester_policy(
    policy: TrustedAttesterPolicy,
) -> TrustedAttesterPolicy:
    """Return the policy only when it is the source-owned release trust root."""

    if type(policy) is not TrustedAttesterPolicy:
        raise TypeError("release trust requires an exact TrustedAttesterPolicy")
    policy.validate()
    if policy != RELEASE_TRUSTED_ATTESTER_POLICY:
        raise ValueError(
            "caller-supplied trusted-attester policies cannot authorize this release"
        )
    return policy
