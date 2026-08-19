"""Source-pinned offline Ed25519 root and dynamic hardware-policy authority.

The repository contains only one public root key and its fingerprints.  A
hardware allowlist is *not* compiled into the source tree: after inventory, an
operator creates a short-lived deployment policy bundle and the offline root
signs a challenge bound to that bundle and exact inventory.  Formal consumers
verify both signatures and reserve both challenges in an external replay
store.  No signing or private-key API exists here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from collections.abc import Collection
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    attestation_message,
)
from lightcone_spec.runtime.attester_bundle import TrustedAttesterPolicyBundle

SOURCE_RELEASE_ROOT_MANIFEST_SEMANTIC_SHA256 = (
    "20f578377b3c18a63a1c04bc05fe01dc2521409fb45bae9ea92af496289d1f42"
)
SOURCE_RELEASE_ROOT_MANIFEST_FILE_SHA256 = (
    "dcb03cc1832d906ff55592dd1bcf263f3a4274ee784b0481dbbc48a9ff9bae4f"
)
SOURCE_RELEASE_ROOT_PUBLIC_KEY_SHA256 = (
    "ed590d62969510a08fb23e8caa40aa2cbd09244810311ddc6e4d5856112cfae7"
)
SOURCE_RELEASE_ROOT_SPKI_SHA256 = (
    "0652b5cfac2ee9d81a08aab68ddd81fd8e809dff3429e3b745e0f704611e63c0"
)
SOURCE_RELEASE_ROOT_PUBLIC_KEY_BASE64 = "rbjX3Uy+RO4etS/6rcVfPoUlGxmY5bwAoUBwhyCkWNU="

DEPLOYMENT_POLICY_MINIMUM_LIFETIME_NS = 1_000_000_000
DEPLOYMENT_POLICY_MAXIMUM_LIFETIME_NS = 600_000_000_000
DEPLOYMENT_POLICY_MAXIMUM_CLOCK_SKEW_NS = 30_000_000_000


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _decode_base64(label: str, value: object, *, length: int) -> bytes:
    if type(value) is not str:
        raise TypeError(f"{label} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ValueError(f"{label} is not canonical base64") from error
    if len(decoded) != length or base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{label} has an invalid length or encoding")
    return decoded


def _strict_object(
    label: str,
    value: object,
    fields: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a JSON object with string keys")
    if set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return value


def _stable_regular_bytes(path: Path, *, maximum_size: int) -> bytes:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError("release-root path must be absolute and normalized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > maximum_size
        ):
            raise ValueError("release-root source must be one bounded regular file")
        body = bytearray()
        while len(body) <= before.st_size:
            chunk = os.read(descriptor, min(64 * 1024, before.st_size + 1 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if (
            len(body) != before.st_size
            or identity(before) != identity(after)
            or identity(after) != identity(current)
        ):
            raise RuntimeError("release-root source changed while it was read")
        return bytes(body)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class SourceReleaseEd25519Root:
    schema_version: int
    kind: Literal["lightcone_source_release_ed25519_root"]
    root_id: str
    key_id: str
    algorithm: Literal["Ed25519"]
    public_key_base64: str
    public_key_sha256: str
    spki_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "lightcone_source_release_ed25519_root"
            or self.algorithm != "Ed25519"
        ):
            raise ValueError("source release root schema is unsupported")
        if self.root_id != "lightcone-release-root-2026q3" or self.key_id != (
            "lightcone-release-root-key-2026q3"
        ):
            raise ValueError("source release root identity is unsupported")
        public_key = _decode_base64(
            "source release root public key",
            self.public_key_base64,
            length=32,
        )
        _require_sha256(
            "source release root public-key fingerprint", self.public_key_sha256
        )
        _require_sha256("source release root SPKI fingerprint", self.spki_sha256)
        spki = Ed25519PublicKey.from_public_bytes(public_key).public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        if (
            hashlib.sha256(public_key).hexdigest() != self.public_key_sha256
            or hashlib.sha256(spki).hexdigest() != self.spki_sha256
        ):
            raise ValueError("source release root fingerprints differ from public key")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "source release root",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "root_id",
                    "key_id",
                    "algorithm",
                    "public_key_base64",
                    "public_key_sha256",
                    "spki_sha256",
                }
            ),
        )
        return cls(**row)


@dataclass(frozen=True)
class SourceReleaseRootDescriptor:
    manifest_path: str
    semantic_sha256: str
    file_sha256: str

    def __post_init__(self) -> None:
        source = Path(self.manifest_path)
        if not source.is_absolute() or Path(os.path.abspath(source)) != source:
            raise ValueError("source release root manifest path is not normalized")
        _require_sha256("source release root semantic identity", self.semantic_sha256)
        _require_sha256("source release root file identity", self.file_sha256)


_RUNTIME_PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_RELEASE_ED25519_ROOT = SourceReleaseRootDescriptor(
    manifest_path=str(_RUNTIME_PACKAGE_ROOT / "trust" / "release_ed25519_root_v1.json"),
    semantic_sha256=SOURCE_RELEASE_ROOT_MANIFEST_SEMANTIC_SHA256,
    file_sha256=SOURCE_RELEASE_ROOT_MANIFEST_FILE_SHA256,
)


@dataclass(frozen=True)
class SourceReleaseRootBinding:
    root: SourceReleaseEd25519Root
    path: str
    sidecar_path: str
    semantic_sha256: str
    file_sha256: str
    sidecar_file_sha256: str

    def __post_init__(self) -> None:
        if type(self.root) is not SourceReleaseEd25519Root:
            raise TypeError("source release root binding requires an exact root")
        for label, digest in (
            ("semantic", self.semantic_sha256),
            ("file", self.file_sha256),
            ("sidecar file", self.sidecar_file_sha256),
        ):
            _require_sha256(f"source release root {label}", digest)
        if (
            self.root.sha256 != self.semantic_sha256
            or self.sidecar_path != f"{self.path}.sha256"
        ):
            raise ValueError("source release root binding is inconsistent")

    @cached_property
    def sha256(self) -> str:
        # Filesystem locations are diagnostic provenance, not cryptographic
        # identity.  The same wheel installed at two prefixes must expose one
        # portable trust anchor.
        return _content_sha256(
            {
                "root": self.root.to_dict(),
                "semantic_sha256": self.semantic_sha256,
                "file_sha256": self.file_sha256,
                "sidecar_file_sha256": self.sidecar_file_sha256,
            }
        )


def load_source_release_ed25519_root() -> SourceReleaseRootBinding:
    """Reopen and validate the source-pinned public root and raw sidecar."""

    descriptor = SOURCE_RELEASE_ED25519_ROOT
    if type(descriptor) is not SourceReleaseRootDescriptor:
        raise RuntimeError("source release Ed25519 root is malformed")
    path = Path(descriptor.manifest_path)
    sidecar_path = Path(f"{path}.sha256")
    body = _stable_regular_bytes(path, maximum_size=16 * 1024)
    sidecar = _stable_regular_bytes(sidecar_path, maximum_size=65)
    if body != _stable_regular_bytes(
        path, maximum_size=16 * 1024
    ) or sidecar != _stable_regular_bytes(sidecar_path, maximum_size=65):
        raise RuntimeError("source release root or sidecar changed while loaded")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source release root is not UTF-8 JSON") from error
    root = SourceReleaseEd25519Root.from_dict(value)
    if body != _canonical_bytes(root.to_dict()) + b"\n":
        raise ValueError("source release root bytes are not canonical")
    file_sha256 = hashlib.sha256(body).hexdigest()
    if (
        root.sha256 != descriptor.semantic_sha256
        or file_sha256 != descriptor.file_sha256
        or sidecar != f"{file_sha256}\n".encode("ascii")
        or root.public_key_base64 != SOURCE_RELEASE_ROOT_PUBLIC_KEY_BASE64
        or root.public_key_sha256 != SOURCE_RELEASE_ROOT_PUBLIC_KEY_SHA256
        or root.spki_sha256 != SOURCE_RELEASE_ROOT_SPKI_SHA256
    ):
        raise ValueError("source release root differs from compiled fingerprints")
    return SourceReleaseRootBinding(
        root=root,
        path=str(path),
        sidecar_path=str(sidecar_path),
        semantic_sha256=root.sha256,
        file_sha256=file_sha256,
        sidecar_file_sha256=hashlib.sha256(sidecar).hexdigest(),
    )


@dataclass(frozen=True)
class DeploymentPolicyAuthorization:
    """Short-lived root signature over an inventory-bound hardware policy."""

    schema_version: int
    kind: Literal["lightcone_deployment_policy_authorization"]
    root_manifest_sha256: str
    inventory_sha256: str
    bundle: TrustedAttesterPolicyBundle
    challenge: AttestationChallenge
    signature_base64: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "lightcone_deployment_policy_authorization"
        ):
            raise ValueError("deployment policy authorization schema is unsupported")
        _require_sha256("deployment root manifest", self.root_manifest_sha256)
        _require_sha256("deployment inventory", self.inventory_sha256)
        if type(self.bundle) is not TrustedAttesterPolicyBundle:
            raise TypeError("deployment policy requires an exact public bundle")
        self.bundle.validate()
        if type(self.challenge) is not AttestationChallenge:
            raise TypeError("deployment policy requires an exact challenge")
        self.challenge.validate()
        if self.challenge.subject_sha256 != self.subject_sha256:
            raise ValueError("deployment policy challenge subject is not exact")
        _decode_base64("deployment policy signature", self.signature_base64, length=64)

    @cached_property
    def subject_sha256(self) -> str:
        return deployment_policy_subject_sha256(
            root_manifest_sha256=self.root_manifest_sha256,
            inventory_sha256=self.inventory_sha256,
            bundle_sha256=self.bundle.sha256,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "root_manifest_sha256": self.root_manifest_sha256,
            "inventory_sha256": self.inventory_sha256,
            "bundle": self.bundle.to_dict(),
            "challenge": asdict(self.challenge),
            "signature_base64": self.signature_base64,
        }

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"authorization_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "deployment policy authorization",
            value,
            frozenset(
                {
                    "authorization_sha256",
                    "schema_version",
                    "kind",
                    "root_manifest_sha256",
                    "inventory_sha256",
                    "bundle",
                    "challenge",
                    "signature_base64",
                }
            ),
        )
        declared = row.pop("authorization_sha256")
        bundle = TrustedAttesterPolicyBundle.from_dict(row.pop("bundle"))
        challenge_row = _strict_object(
            "deployment policy challenge",
            row.pop("challenge"),
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
        authorization = cls(
            bundle=bundle,
            challenge=AttestationChallenge(**challenge_row),
            **row,
        )
        if declared != authorization.sha256:
            raise ValueError("deployment policy authorization SHA-256 mismatch")
        return authorization


@dataclass(frozen=True)
class VerifiedDeploymentPolicy:
    bundle: TrustedAttesterPolicyBundle
    root_binding: SourceReleaseRootBinding
    authorization_sha256: str
    challenge_sha256: str
    inventory_sha256: str


def deployment_policy_subject_sha256(
    *,
    root_manifest_sha256: str,
    inventory_sha256: str,
    bundle_sha256: str,
) -> str:
    """Return the exact public subject an offline signer must authorize.

    This helper does not sign, load a key, or accept hardware descriptions.
    It only makes the verifier-owned domain separation available to the
    out-of-band signing ceremony without duplicating canonicalization logic.
    """

    for label, digest in (
        ("deployment root manifest", root_manifest_sha256),
        ("deployment inventory", inventory_sha256),
        ("deployment bundle", bundle_sha256),
    ):
        _require_sha256(label, digest)
    return _content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_deployment_policy_subject",
            "root_manifest_sha256": root_manifest_sha256,
            "inventory_sha256": inventory_sha256,
            "bundle_sha256": bundle_sha256,
        }
    )


def verify_source_signed_deployment_policy(
    authorization: DeploymentPolicyAuthorization,
    *,
    expected_inventory_sha256: str,
    now_ns: int,
    consumed_challenge_sha256s: Collection[str],
) -> VerifiedDeploymentPolicy:
    """Verify root signature, freshness, replay, inventory, and policy validity."""

    if type(authorization) is not DeploymentPolicyAuthorization:
        raise TypeError("deployment verification requires an exact authorization")
    _require_sha256("expected deployment inventory", expected_inventory_sha256)
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("deployment verification time must be non-negative")
    consumed = tuple(consumed_challenge_sha256s)
    if isinstance(consumed_challenge_sha256s, (str, bytes)) or any(
        _require_sha256("consumed deployment challenge", item) != item
        for item in consumed
    ):
        raise TypeError("deployment replay snapshot must contain challenge digests")
    if len(consumed) != len(set(consumed)):
        raise ValueError("deployment replay snapshot contains duplicate digests")
    root_binding = load_source_release_ed25519_root()
    if (
        authorization.root_manifest_sha256 != root_binding.semantic_sha256
        or authorization.inventory_sha256 != expected_inventory_sha256
    ):
        raise ValueError("deployment policy differs from source root or inventory")
    authorization.__post_init__()
    challenge = authorization.challenge
    challenge.validate(now_ns=now_ns)
    lifetime = challenge.expires_ns - challenge.issued_ns
    if not (
        DEPLOYMENT_POLICY_MINIMUM_LIFETIME_NS
        <= lifetime
        <= DEPLOYMENT_POLICY_MAXIMUM_LIFETIME_NS
    ):
        raise ValueError("deployment policy challenge lifetime violates release bounds")
    if challenge.issued_ns > now_ns + DEPLOYMENT_POLICY_MAXIMUM_CLOCK_SKEW_NS:
        raise ValueError("deployment policy challenge was issued too far in the future")
    if challenge.sha256 in consumed:
        raise ValueError("deployment policy challenge was already consumed")
    public_key = _decode_base64(
        "source release root public key",
        root_binding.root.public_key_base64,
        length=32,
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            _decode_base64(
                "deployment policy signature",
                authorization.signature_base64,
                length=64,
            ),
            attestation_message(
                challenge,
                payload_sha256=authorization.bundle.sha256,
            ),
        )
    except InvalidSignature as error:
        raise ValueError("deployment policy root signature is invalid") from error
    authorization.bundle.validate(now_ns=now_ns)
    return VerifiedDeploymentPolicy(
        bundle=authorization.bundle,
        root_binding=root_binding,
        authorization_sha256=authorization.sha256,
        challenge_sha256=challenge.sha256,
        inventory_sha256=authorization.inventory_sha256,
    )


__all__ = [
    "DEPLOYMENT_POLICY_MAXIMUM_CLOCK_SKEW_NS",
    "DEPLOYMENT_POLICY_MAXIMUM_LIFETIME_NS",
    "DEPLOYMENT_POLICY_MINIMUM_LIFETIME_NS",
    "SOURCE_RELEASE_ED25519_ROOT",
    "SOURCE_RELEASE_ROOT_MANIFEST_FILE_SHA256",
    "SOURCE_RELEASE_ROOT_MANIFEST_SEMANTIC_SHA256",
    "SOURCE_RELEASE_ROOT_PUBLIC_KEY_BASE64",
    "SOURCE_RELEASE_ROOT_PUBLIC_KEY_SHA256",
    "SOURCE_RELEASE_ROOT_SPKI_SHA256",
    "DeploymentPolicyAuthorization",
    "SourceReleaseEd25519Root",
    "SourceReleaseRootBinding",
    "SourceReleaseRootDescriptor",
    "VerifiedDeploymentPolicy",
    "deployment_policy_subject_sha256",
    "load_source_release_ed25519_root",
    "verify_source_signed_deployment_policy",
]
