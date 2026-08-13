"""Externally provisioned, content-bound trusted-attester policy bundles.

The bundle is public verification material.  It may contain Ed25519 public
keys, but never signing keys.  Trust is supplied by a separately provisioned
anchor descriptor that fixes both the absolute bundle path and its semantic
SHA-256.  The loader deliberately accepts only that descriptor: accepting a
caller-selected bundle path alongside a caller-selected digest would make the
untrusted bundle its own trust root.

The source release currently has no configured anchor and therefore fails
closed.  ``operator_configuration`` descriptors are deployment capabilities;
they must be created from operator-owned configuration, never from request or
experiment input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn, Self

from .attestation import AttestationChallenge, TrustedAttesterPolicy

AnchorAuthority = Literal["source_release", "operator_configuration"]

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_FORBIDDEN_IDENTITY_PREFIXES = ("test", "fixture", "cpu")
_MAX_BUNDLE_BYTES = 1024 * 1024
_BUNDLE_FIELDS = {
    "schema_version",
    "kind",
    "bundle_id",
    "valid_from_ns",
    "expires_ns",
    "nonce_policy",
    "hardware_envelope_sha256_allowlist",
    "trusted_attester_policy",
}
_NONCE_POLICY_FIELDS = {
    "schema_version",
    "kind",
    "nonce_bytes",
    "minimum_lifetime_ns",
    "maximum_lifetime_ns",
    "maximum_clock_skew_ns",
    "replay_policy",
    "subject_binding_required",
}
_TRUSTED_POLICY_FIELDS = {
    "schema_version",
    "policy_id",
    "trusted_attesters",
    "public_keys",
}
_PRIVATE_FIELD_NAMES = {
    "private",
    "private_key",
    "private_key_base64",
    "private_key_hex",
    "private_key_pem",
    "secret",
    "secret_key",
    "signing_key",
    "signing_seed",
    "seed_phrase",
}
_TEST_IDENTITY_FIELD_NAMES = {
    "cpu_identity",
    "fixture_identity",
    "test_identity",
    "test_key",
    "test_private_key",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(label: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_now_ns(now_ns: int | None) -> int:
    value = time.time_ns() if now_ns is None else now_ns
    if type(value) is not int or value < 0:
        raise ValueError("trusted-attester validation time must be non-negative")
    return value


def _forbidden_identity(value: str) -> bool:
    return value.lower().startswith(_FORBIDDEN_IDENTITY_PREFIXES)


def _require_release_id(label: str, value: object) -> str:
    if (
        type(value) is not str
        or _SAFE_ID.fullmatch(value) is None
        or _forbidden_identity(value)
    ):
        raise ValueError(f"{label} must be a non-test release identity")
    return value


def _absolute_lexical_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_anchor_path(value: object) -> Path:
    if type(value) is not str:
        raise TypeError("trusted-attester bundle path must be a string")
    path = Path(value)
    if not path.is_absolute() or _absolute_lexical_path(path) != path:
        raise ValueError("trusted-attester bundle path must be absolute and normalized")
    return path


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_single_link_bytes(
    path: Path,
    *,
    label: str,
    maximum_size: int,
) -> bytes:
    """Read a stable regular leaf without following symlinks or hard links."""

    if not path.is_absolute() or _absolute_lexical_path(path) != path:
        raise ValueError(f"{label} path must be absolute and normalized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is not a readable regular file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        if before.st_size < 1 or before.st_size > maximum_size:
            raise ValueError(f"{label} size is outside the supported bound")
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b"".join(chunks)
        after = os.fstat(descriptor)
        try:
            current = path.stat(follow_symlinks=False)
        except OSError as error:
            raise RuntimeError(f"{label} changed while it was read") from error
        if (
            len(body) != before.st_size
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)
            or not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(
                f"trusted-attester bundle contains duplicate field {key!r}"
            )
        value[key] = item
    return value


def _reject_constant(raw: str) -> NoReturn:
    raise ValueError(f"trusted-attester bundle contains non-finite value {raw!r}")


def _reject_forbidden_material(value: object) -> None:
    """Reject signing/test material before ordinary schema validation."""

    if type(value) is list:
        for item in value:
            _reject_forbidden_material(item)
        return
    if type(value) is not dict:
        return
    for key, item in value.items():
        normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if normalized in _PRIVATE_FIELD_NAMES:
            raise ValueError(
                "trusted-attester bundle must not contain private material"
            )
        if normalized in _TEST_IDENTITY_FIELD_NAMES:
            raise ValueError("trusted-attester bundle must not contain test identities")
        _reject_forbidden_material(item)


def _strict_json_object(body: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("trusted-attester bundle is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise TypeError("trusted-attester bundle must contain one JSON object")

    def require_finite(item: object) -> None:
        if type(item) is float and not math.isfinite(item):
            raise ValueError("trusted-attester bundle contains a non-finite number")
        if type(item) is list:
            for child in item:
                require_finite(child)
        elif type(item) is dict:
            for child in item.values():
                require_finite(child)

    require_finite(value)
    _reject_forbidden_material(value)
    return value


def _strict_object(
    label: str,
    value: object,
    fields: set[str],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema v1")
    return value


def _read_bundle_pair(path: Path) -> tuple[bytes, bytes]:
    sidecar_path = Path(f"{path}.sha256")
    body = _stable_single_link_bytes(
        path,
        label="trusted-attester bundle",
        maximum_size=_MAX_BUNDLE_BYTES,
    )
    sidecar = _stable_single_link_bytes(
        sidecar_path,
        label="trusted-attester bundle sidecar",
        maximum_size=65,
    )
    # Reopen both leaves so a cross-file replacement cannot produce one
    # internally inconsistent observation window.
    if body != _stable_single_link_bytes(
        path,
        label="trusted-attester bundle",
        maximum_size=_MAX_BUNDLE_BYTES,
    ) or sidecar != _stable_single_link_bytes(
        sidecar_path,
        label="trusted-attester bundle sidecar",
        maximum_size=65,
    ):
        raise RuntimeError("trusted-attester bundle or sidecar changed while loaded")
    return body, sidecar


@dataclass(frozen=True)
class AttestationNoncePolicy:
    """Challenge freshness and replay requirements for release evidence."""

    schema_version: int
    kind: Literal["lightcone_attestation_nonce_policy"]
    nonce_bytes: int
    minimum_lifetime_ns: int
    maximum_lifetime_ns: int
    maximum_clock_skew_ns: int
    replay_policy: Literal["external_single_use_store"]
    subject_binding_required: bool

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "lightcone_attestation_nonce_policy"
        ):
            raise ValueError("attestation nonce policy schema is unsupported")
        if type(self.nonce_bytes) is not int or self.nonce_bytes != 32:
            raise ValueError("release attestation nonces must contain exactly 32 bytes")
        if (
            type(self.minimum_lifetime_ns) is not int
            or type(self.maximum_lifetime_ns) is not int
            or self.minimum_lifetime_ns <= 0
            or self.maximum_lifetime_ns < self.minimum_lifetime_ns
        ):
            raise ValueError("attestation nonce lifetime bounds are invalid")
        if (
            type(self.maximum_clock_skew_ns) is not int
            or self.maximum_clock_skew_ns < 0
            or self.maximum_clock_skew_ns > self.maximum_lifetime_ns
        ):
            raise ValueError("attestation nonce clock-skew bound is invalid")
        if self.replay_policy != "external_single_use_store":
            raise ValueError("release attestation requires an external replay store")
        if self.subject_binding_required is not True:
            raise ValueError("release attestation requires subject-bound challenges")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "nonce_bytes": self.nonce_bytes,
            "minimum_lifetime_ns": self.minimum_lifetime_ns,
            "maximum_lifetime_ns": self.maximum_lifetime_ns,
            "maximum_clock_skew_ns": self.maximum_clock_skew_ns,
            "replay_policy": self.replay_policy,
            "subject_binding_required": self.subject_binding_required,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object("attestation nonce policy", value, _NONCE_POLICY_FIELDS)
        policy = cls(
            schema_version=row["schema_version"],
            kind=row["kind"],
            nonce_bytes=row["nonce_bytes"],
            minimum_lifetime_ns=row["minimum_lifetime_ns"],
            maximum_lifetime_ns=row["maximum_lifetime_ns"],
            maximum_clock_skew_ns=row["maximum_clock_skew_ns"],
            replay_policy=row["replay_policy"],
            subject_binding_required=row["subject_binding_required"],
        )
        policy.validate()
        return policy

    def validate_challenge(
        self,
        challenge: AttestationChallenge,
        *,
        now_ns: int,
        consumed_challenge_sha256s: Collection[str],
    ) -> str:
        """Validate one challenge against an authoritative replay snapshot.

        This method does not mutate the external replay store.  The consumer
        must atomically reserve the returned digest before accepting evidence.
        """

        self.validate()
        if type(challenge) is not AttestationChallenge:
            raise TypeError("nonce policy requires an exact AttestationChallenge")
        current = _require_now_ns(now_ns)
        if isinstance(consumed_challenge_sha256s, (str, bytes)) or not isinstance(
            consumed_challenge_sha256s, Collection
        ):
            raise TypeError("nonce validation requires a challenge-digest collection")
        consumed = tuple(consumed_challenge_sha256s)
        for digest in consumed:
            _require_sha256("consumed challenge identity", digest)
        if len(consumed) != len(set(consumed)):
            raise ValueError("consumed challenge identities must be unique")
        challenge.validate(now_ns=current)
        lifetime = challenge.expires_ns - challenge.issued_ns
        if not self.minimum_lifetime_ns <= lifetime <= self.maximum_lifetime_ns:
            raise ValueError("attestation challenge lifetime violates policy")
        if challenge.issued_ns > current + self.maximum_clock_skew_ns:
            raise ValueError("attestation challenge was issued too far in the future")
        challenge_sha256 = challenge.sha256
        if challenge_sha256 in consumed:
            raise ValueError("attestation challenge nonce was already consumed")
        return challenge_sha256


@dataclass(frozen=True)
class TrustedAttesterPolicyBundle:
    """Canonical public keys and policy constraints for one release interval."""

    schema_version: int
    kind: Literal["lightcone_trusted_attester_policy_bundle"]
    bundle_id: str
    valid_from_ns: int
    expires_ns: int
    nonce_policy: AttestationNoncePolicy
    hardware_envelope_sha256_allowlist: tuple[str, ...]
    trusted_attester_policy: TrustedAttesterPolicy

    def validate(self, *, now_ns: int | None = None) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "lightcone_trusted_attester_policy_bundle"
        ):
            raise ValueError("trusted-attester bundle schema is unsupported")
        _require_release_id("trusted-attester bundle ID", self.bundle_id)
        if (
            type(self.valid_from_ns) is not int
            or type(self.expires_ns) is not int
            or self.valid_from_ns < 0
            or self.expires_ns <= self.valid_from_ns
        ):
            raise ValueError("trusted-attester bundle validity interval is invalid")
        if type(self.nonce_policy) is not AttestationNoncePolicy:
            raise TypeError("trusted-attester bundle nonce policy is invalid")
        self.nonce_policy.validate()
        envelopes = self.hardware_envelope_sha256_allowlist
        if (
            type(envelopes) is not tuple
            or not envelopes
            or envelopes != tuple(sorted(set(envelopes)))
        ):
            raise ValueError(
                "hardware-envelope allowlist must be non-empty, unique, and sorted"
            )
        for digest in envelopes:
            _require_sha256("hardware-envelope identity", digest)
        if type(self.trusted_attester_policy) is not TrustedAttesterPolicy:
            raise TypeError("trusted-attester bundle policy type is invalid")
        self.trusted_attester_policy.validate()
        _require_release_id(
            "trusted-attester policy ID", self.trusted_attester_policy.policy_id
        )
        if not self.trusted_attester_policy.release_ready:
            raise ValueError("trusted-attester bundle contains no release attester")
        if now_ns is not None:
            current = _require_now_ns(now_ns)
            if current < self.valid_from_ns:
                raise ValueError("trusted-attester bundle is not yet valid")
            if current > self.expires_ns:
                raise ValueError("trusted-attester bundle expired")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "bundle_id": self.bundle_id,
            "valid_from_ns": self.valid_from_ns,
            "expires_ns": self.expires_ns,
            "nonce_policy": self.nonce_policy.to_dict(),
            "hardware_envelope_sha256_allowlist": list(
                self.hardware_envelope_sha256_allowlist
            ),
            "trusted_attester_policy": self.trusted_attester_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object("trusted-attester bundle", value, _BUNDLE_FIELDS)
        raw_envelopes = row["hardware_envelope_sha256_allowlist"]
        if type(raw_envelopes) is not list:
            raise TypeError("hardware-envelope allowlist must be a JSON array")
        policy_row = _strict_object(
            "trusted-attester policy",
            row["trusted_attester_policy"],
            _TRUSTED_POLICY_FIELDS,
        )
        if (
            type(policy_row["schema_version"]) is not int
            or policy_row["schema_version"] != 1
        ):
            raise ValueError("trusted-attester policy schema is unsupported")
        raw_attesters = policy_row["trusted_attesters"]
        raw_public_keys = policy_row["public_keys"]
        if type(raw_attesters) is not list or type(raw_public_keys) is not list:
            raise TypeError("trusted-attester policy entries must be JSON arrays")

        def rows(
            label: str,
            values: list[object],
            width: int,
        ) -> tuple[tuple[str, ...], ...]:
            parsed: list[tuple[str, ...]] = []
            for item in values:
                if (
                    type(item) is not list
                    or len(item) != width
                    or any(type(field) is not str for field in item)
                ):
                    raise ValueError(f"{label} row is invalid")
                parsed.append(tuple(item))
            return tuple(parsed)

        policy_id = _require_release_id(
            "trusted-attester policy ID", policy_row["policy_id"]
        )
        policy = TrustedAttesterPolicy(
            policy_id=policy_id,
            trusted_attesters=rows("trusted-attester allowlist", raw_attesters, 3),
            public_keys=rows("trusted-attester public keys", raw_public_keys, 2),
        )
        bundle = cls(
            schema_version=row["schema_version"],
            kind=row["kind"],
            bundle_id=row["bundle_id"],
            valid_from_ns=row["valid_from_ns"],
            expires_ns=row["expires_ns"],
            nonce_policy=AttestationNoncePolicy.from_dict(row["nonce_policy"]),
            hardware_envelope_sha256_allowlist=tuple(raw_envelopes),
            trusted_attester_policy=policy,
        )
        bundle.validate()
        return bundle

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict()) + b"\n"

    def require_hardware_envelope(self, hardware_envelope_sha256: str) -> None:
        self.validate()
        digest = _require_sha256("hardware-envelope identity", hardware_envelope_sha256)
        if digest not in self.hardware_envelope_sha256_allowlist:
            raise ValueError("hardware envelope is not allowlisted by the trust bundle")

    def validate_challenge(
        self,
        challenge: AttestationChallenge,
        *,
        now_ns: int,
        consumed_challenge_sha256s: Collection[str],
    ) -> str:
        current = _require_now_ns(now_ns)
        self.validate(now_ns=current)
        return self.nonce_policy.validate_challenge(
            challenge,
            now_ns=current,
            consumed_challenge_sha256s=consumed_challenge_sha256s,
        )


@dataclass(frozen=True)
class TrustedAttesterAnchorDescriptor:
    """Authority-bearing descriptor that fixes one external bundle in advance."""

    schema_version: int
    kind: Literal["lightcone_trusted_attester_anchor"]
    anchor_id: str
    authority: AnchorAuthority
    bundle_path: str
    bundle_sha256: str
    valid_from_ns: int
    expires_ns: int

    def validate(self, *, now_ns: int | None = None) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "lightcone_trusted_attester_anchor"
        ):
            raise ValueError("trusted-attester anchor schema is unsupported")
        _require_release_id("trusted-attester anchor ID", self.anchor_id)
        if self.authority not in {"source_release", "operator_configuration"}:
            raise ValueError("trusted-attester anchor authority is unsupported")
        _require_anchor_path(self.bundle_path)
        _require_sha256("trusted-attester anchored bundle", self.bundle_sha256)
        if (
            type(self.valid_from_ns) is not int
            or type(self.expires_ns) is not int
            or self.valid_from_ns < 0
            or self.expires_ns <= self.valid_from_ns
        ):
            raise ValueError("trusted-attester anchor validity interval is invalid")
        if now_ns is not None:
            current = _require_now_ns(now_ns)
            if current < self.valid_from_ns:
                raise ValueError("trusted-attester anchor is not yet valid")
            if current > self.expires_ns:
                raise ValueError("trusted-attester anchor expired")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "anchor_id": self.anchor_id,
            "authority": self.authority,
            "bundle_path": self.bundle_path,
            "bundle_sha256": self.bundle_sha256,
            "valid_from_ns": self.valid_from_ns,
            "expires_ns": self.expires_ns,
        }

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())


# No signer can authorize this source release until a reviewed release changes
# this constant.  Operator deployments use an explicit
# ``operator_configuration`` descriptor supplied outside experiment input.
SOURCE_RELEASE_TRUSTED_ATTESTER_ANCHOR: TrustedAttesterAnchorDescriptor | None = None


@dataclass(frozen=True)
class TrustedAttesterPolicyBundleBinding:
    """Raw-byte binding for a bundle and exact semantic sidecar."""

    schema_version: int
    kind: Literal["lightcone_trusted_attester_policy_bundle_binding"]
    anchor: TrustedAttesterAnchorDescriptor
    anchor_sha256: str
    path: str
    sidecar_path: str
    bundle_sha256: str
    file_sha256: str
    sidecar_file_sha256: str
    size: int
    sidecar_size: int

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "lightcone_trusted_attester_policy_bundle_binding"
        ):
            raise ValueError("trusted-attester bundle binding schema is unsupported")
        if type(self.anchor) is not TrustedAttesterAnchorDescriptor:
            raise TypeError("trusted-attester bundle binding anchor is invalid")
        self.anchor.validate()
        if self.anchor_sha256 != self.anchor.sha256:
            raise ValueError("trusted-attester bundle binding anchor differs")
        source = _require_anchor_path(self.path)
        if source != Path(self.anchor.bundle_path):
            raise ValueError("trusted-attester binding path differs from its anchor")
        if self.sidecar_path != f"{self.path}.sha256":
            raise ValueError("trusted-attester binding sidecar path is not exact")
        if self.bundle_sha256 != self.anchor.bundle_sha256:
            raise ValueError("trusted-attester binding digest differs from its anchor")
        for label, digest in (
            ("bundle", self.bundle_sha256),
            ("bundle file", self.file_sha256),
            ("bundle sidecar file", self.sidecar_file_sha256),
        ):
            _require_sha256(f"trusted-attester {label}", digest)
        if type(self.size) is not int or not 1 <= self.size <= _MAX_BUNDLE_BYTES:
            raise ValueError("trusted-attester bundle binding size is invalid")
        if type(self.sidecar_size) is not int or self.sidecar_size != 65:
            raise ValueError("trusted-attester bundle sidecar must be one digest line")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "anchor": self.anchor.to_dict(),
            "anchor_sha256": self.anchor_sha256,
            "path": self.path,
            "sidecar_path": self.sidecar_path,
            "bundle_sha256": self.bundle_sha256,
            "file_sha256": self.file_sha256,
            "sidecar_file_sha256": self.sidecar_file_sha256,
            "size": self.size,
            "sidecar_size": self.sidecar_size,
        }

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    def reopen(self, *, now_ns: int | None = None) -> LoadedTrustedAttesterPolicyBundle:
        """Reopen every bound byte and reapply validity and anchor checks."""

        self.validate()
        body, sidecar = _read_bundle_pair(Path(self.path))
        if (
            len(body) != self.size
            or len(sidecar) != self.sidecar_size
            or hashlib.sha256(body).hexdigest() != self.file_sha256
            or hashlib.sha256(sidecar).hexdigest() != self.sidecar_file_sha256
        ):
            raise RuntimeError("trusted-attester bundle or sidecar changed after load")
        loaded = load_trusted_attester_policy_bundle(self.anchor, now_ns=now_ns)
        if loaded.binding != self:
            raise RuntimeError("trusted-attester bundle or sidecar changed after load")
        return loaded


@dataclass(frozen=True)
class LoadedTrustedAttesterPolicyBundle:
    """Validated policy plus the path-bound evidence needed to reopen it."""

    bundle: TrustedAttesterPolicyBundle
    binding: TrustedAttesterPolicyBundleBinding

    def __post_init__(self) -> None:
        if type(self.bundle) is not TrustedAttesterPolicyBundle:
            raise TypeError("loaded trusted-attester bundle type is invalid")
        if type(self.binding) is not TrustedAttesterPolicyBundleBinding:
            raise TypeError("loaded trusted-attester binding type is invalid")
        self.bundle.validate()
        self.binding.validate()
        if self.bundle.sha256 != self.binding.bundle_sha256:
            raise ValueError("loaded trusted-attester bundle differs from its binding")

    @property
    def policy(self) -> TrustedAttesterPolicy:
        return self.bundle.trusted_attester_policy

    def reopen(self, *, now_ns: int | None = None) -> Self:
        return self.binding.reopen(now_ns=now_ns)


def _require_authoritative_anchor(
    anchor: TrustedAttesterAnchorDescriptor,
) -> None:
    if type(anchor) is not TrustedAttesterAnchorDescriptor:
        raise TypeError("trusted-attester loading requires an exact anchor descriptor")
    anchor.validate()
    if anchor.authority == "source_release":
        configured = SOURCE_RELEASE_TRUSTED_ATTESTER_ANCHOR
        if configured is None:
            raise RuntimeError("source-release trusted-attester anchor is unavailable")
        if type(configured) is not TrustedAttesterAnchorDescriptor:
            raise RuntimeError("source-release trusted-attester anchor is malformed")
        configured.validate()
        if anchor != configured:
            raise ValueError(
                "caller-selected source-release anchor is not authoritative"
            )


def load_trusted_attester_policy_bundle(
    anchor: TrustedAttesterAnchorDescriptor,
    *,
    now_ns: int | None = None,
) -> LoadedTrustedAttesterPolicyBundle:
    """Load the bundle fixed by ``anchor``; no caller path override exists."""

    _require_authoritative_anchor(anchor)
    current = _require_now_ns(now_ns)
    anchor.validate(now_ns=current)
    source = Path(anchor.bundle_path)
    body, sidecar = _read_bundle_pair(source)
    value = _strict_json_object(body)
    bundle = TrustedAttesterPolicyBundle.from_dict(value)
    canonical = bundle.canonical_bytes
    if body != canonical:
        raise ValueError("trusted-attester bundle bytes are not canonical")
    if bundle.sha256 != anchor.bundle_sha256:
        raise ValueError("trusted-attester bundle differs from its trust anchor")
    if sidecar != f"{bundle.sha256}\n".encode("ascii"):
        raise ValueError("trusted-attester bundle sidecar differs from content")
    bundle.validate(now_ns=current)
    if (
        bundle.valid_from_ns < anchor.valid_from_ns
        or bundle.expires_ns > anchor.expires_ns
    ):
        raise ValueError("trusted-attester bundle validity exceeds its anchor")
    binding = TrustedAttesterPolicyBundleBinding(
        schema_version=1,
        kind="lightcone_trusted_attester_policy_bundle_binding",
        anchor=anchor,
        anchor_sha256=anchor.sha256,
        path=str(source),
        sidecar_path=f"{source}.sha256",
        bundle_sha256=bundle.sha256,
        file_sha256=hashlib.sha256(body).hexdigest(),
        sidecar_file_sha256=hashlib.sha256(sidecar).hexdigest(),
        size=len(body),
        sidecar_size=len(sidecar),
    )
    binding.validate()
    return LoadedTrustedAttesterPolicyBundle(bundle=bundle, binding=binding)


def load_source_release_trusted_attester_policy_bundle(
    *,
    now_ns: int | None = None,
) -> LoadedTrustedAttesterPolicyBundle:
    """Load the compiled source anchor, failing closed while none exists."""

    anchor = SOURCE_RELEASE_TRUSTED_ATTESTER_ANCHOR
    if anchor is None:
        raise RuntimeError("source-release trusted-attester anchor is unavailable")
    if (
        type(anchor) is not TrustedAttesterAnchorDescriptor
        or anchor.authority != "source_release"
    ):
        raise RuntimeError("source-release trusted-attester anchor is malformed")
    return load_trusted_attester_policy_bundle(anchor, now_ns=now_ns)
