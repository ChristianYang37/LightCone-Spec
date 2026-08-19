"""One offline Ed25519 verification path for formal control artifacts.

Dispatch, compile, non-serving terminal, capacity, interference, and all-rank
aggregate evidence use different payload schemas, but they share one trust
decision.  This module binds the artifact *type* into the challenge subject,
the exact artifact digest into the signed payload, and validates freshness,
expiry, hardware scope, and an externally maintained single-use replay set.

Only the source-release public bundle can authorize formal work.  The private
key is deliberately absent from this repository and this API contains no
signing operation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
)
from lightcone_spec.runtime.proof_artifact import relocated_evidence_path
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    verify_source_signed_deployment_policy,
)

ControlArtifactType = Literal[
    "dispatch",
    "compile",
    "non_serving_terminal",
    "capacity",
    "interference",
    "rank_aggregate",
]

CONTROL_ARTIFACT_TYPES: tuple[ControlArtifactType, ...] = (
    "capacity",
    "compile",
    "dispatch",
    "interference",
    "non_serving_terminal",
    "rank_aggregate",
)

_RESERVATION_NAME = re.compile(r"reservation-([0-9a-f]{64})\.json\Z")

# The largest fixed all-cell source batch is E3a (360 controls).  Reserving a
# distinct deployment challenge per control plus 48 nested signed-authority
# challenges requires at most 768 digests.  The canonical record remains below
# the separately enforced 64 KiB evidence bound.  Larger transactions must be
# split by a source-owned stage protocol; they cannot silently create an
# unreadable replay record.
MAXIMUM_FORMAL_ATOMIC_CHALLENGES = (2 * 360) + 48
MAXIMUM_CHALLENGE_RESERVATION_BYTES = 64 * 1024


class ReleaseControlAttestationBlocked(RuntimeError):
    """The source release has no usable public trust root."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"release control attestation is BLOCKED: {reason_code}")


@dataclass(frozen=True)
class ChallengeReplayStore:
    """Operator-owned append-only store for atomic challenge groups.

    A single immutable reservation file contains both the root authorization
    challenge and every artifact challenge accepted in one decision.  The
    directory lock covers replay snapshot, signature verification, and the
    no-replace write, closing the usual verify/commit race.
    """

    root: str

    def __post_init__(self) -> None:
        path = Path(self.root)
        if not path.is_absolute() or Path(os.path.abspath(path)) != path:
            raise ValueError("challenge replay store path must be absolute")

    @staticmethod
    def _read_reservation(directory_fd: int, name: str) -> tuple[int, tuple[str, ...]]:
        match = _RESERVATION_NAME.fullmatch(name)
        if match is None:
            raise ValueError("challenge replay store contains an unknown entry")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size < 1
                or before.st_size > MAXIMUM_CHALLENGE_RESERVATION_BYTES
            ):
                raise ValueError("challenge reservation is not one bounded file")
            body = os.read(descriptor, before.st_size + 1)
            after = os.fstat(descriptor)
            if len(body) != before.st_size or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError("challenge reservation changed while read")
        finally:
            os.close(descriptor)
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("challenge reservation is not UTF-8 JSON") from error
        row = _strict_object(
            "challenge reservation",
            value,
            frozenset({"schema_version", "kind", "reserved_ns", "challenge_sha256s"}),
        )
        reserved_ns = row["reserved_ns"]
        if type(reserved_ns) is not int or reserved_ns < 1:
            raise ValueError("challenge reservation time is invalid")
        raw = row["challenge_sha256s"]
        if type(raw) is not list:
            raise TypeError("challenge reservation digest set must be an array")
        challenges = tuple(raw)
        if (
            not challenges
            or len(challenges) > MAXIMUM_FORMAL_ATOMIC_CHALLENGES
            or challenges != tuple(sorted(set(challenges)))
            or any(
                _require_sha256("reserved challenge", item) != item
                for item in challenges
            )
        ):
            raise ValueError("challenge reservation digests are not canonical")
        canonical = {
            "schema_version": 2,
            "kind": "lightcone_control_challenge_reservation",
            "reserved_ns": reserved_ns,
            "challenge_sha256s": list(challenges),
        }
        if body != (
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        ) or match.group(1) != _canonical_sha256(canonical):
            raise ValueError("challenge reservation bytes or name are not canonical")
        return reserved_ns, challenges

    def _locked_verify_and_reserve(
        self,
        envelopes: tuple[ControlArtifactAttestation, ...],
        *,
        expected_inventory_sha256: str,
        now_ns: int,
        additional_challenge_sha256s: tuple[str, ...],
    ) -> tuple[VerifiedControlArtifact, ...]:
        root = Path(self.root)
        if (2 * len(envelopes)) + len(
            additional_challenge_sha256s
        ) > MAXIMUM_FORMAL_ATOMIC_CHALLENGES:
            raise ValueError("formal atomic challenge batch exceeds source bound")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        directory_fd = os.open(root, flags)
        try:
            directory_stat = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_stat.st_mode):
                raise ValueError("challenge replay store must be a directory")
            if directory_stat.st_uid != os.geteuid() or (
                stat.S_IMODE(directory_stat.st_mode) & 0o022
            ):
                raise ValueError(
                    "challenge replay store must be current-user owned and "
                    "not group/world writable"
                )
            lock_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
            lock_flags |= getattr(os, "O_NONBLOCK", 0)
            if hasattr(os, "O_NOFOLLOW"):
                lock_flags |= os.O_NOFOLLOW
            created_lock = False
            try:
                lock_fd = os.open(
                    ".lock",
                    lock_flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_fd,
                )
                created_lock = True
            except FileExistsError:
                lock_fd = os.open(".lock", lock_flags, dir_fd=directory_fd)
            try:
                if created_lock:
                    os.fchmod(lock_fd, 0o600)
                before_lock = os.fstat(lock_fd)
                if (
                    not stat.S_ISREG(before_lock.st_mode)
                    or before_lock.st_nlink != 1
                    or before_lock.st_uid != os.geteuid()
                    or stat.S_IMODE(before_lock.st_mode) != 0o600
                    or before_lock.st_size != 0
                ):
                    raise ValueError(
                        "challenge replay lock must be one empty current-user "
                        "0600 regular file"
                    )
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                after_lock = os.fstat(lock_fd)
                if (
                    before_lock.st_dev,
                    before_lock.st_ino,
                    before_lock.st_nlink,
                    before_lock.st_uid,
                    stat.S_IMODE(before_lock.st_mode),
                    before_lock.st_size,
                ) != (
                    after_lock.st_dev,
                    after_lock.st_ino,
                    after_lock.st_nlink,
                    after_lock.st_uid,
                    stat.S_IMODE(after_lock.st_mode),
                    after_lock.st_size,
                ):
                    raise RuntimeError("challenge replay lock changed while locked")
                consumed_records = tuple(
                    self._read_reservation(directory_fd, name)
                    for name in sorted(os.listdir(directory_fd))
                    if name != ".lock"
                )
                consumed = tuple(
                    item for _, challenges in consumed_records for item in challenges
                )
                if len(consumed) != len(set(consumed)):
                    raise ValueError("challenge replay store contains duplicate claims")
                if set(additional_challenge_sha256s) & set(consumed):
                    raise ValueError("additional challenge is already consumed")
                results = tuple(
                    verify_release_control_artifact_attestation(
                        envelope,
                        expected_inventory_sha256=expected_inventory_sha256,
                        now_ns=now_ns,
                        consumed_challenge_sha256s=consumed,
                    )
                    for envelope in envelopes
                )
                artifact_challenges = tuple(row.challenge_sha256 for row in results)
                if len(artifact_challenges) != len(set(artifact_challenges)):
                    raise ValueError("control artifacts reuse one artifact challenge")
                deployment_by_challenge: dict[str, str] = {}
                for row in results:
                    prior = deployment_by_challenge.setdefault(
                        row.deployment_policy_challenge_sha256,
                        row.deployment_policy_authorization_sha256,
                    )
                    if prior != row.deployment_policy_authorization_sha256:
                        raise ValueError(
                            "one deployment challenge names different authorizations"
                        )
                if set(artifact_challenges) & set(deployment_by_challenge):
                    raise ValueError("artifact and deployment challenges overlap")
                control_challenges = {
                    *artifact_challenges,
                    *deployment_by_challenge,
                }
                if control_challenges & set(additional_challenge_sha256s):
                    raise ValueError("additional and control challenges overlap")
                reservations = tuple(
                    sorted(
                        (
                            *artifact_challenges,
                            *deployment_by_challenge,
                            *additional_challenge_sha256s,
                        )
                    )
                )
                if len(reservations) > MAXIMUM_FORMAL_ATOMIC_CHALLENGES:
                    raise ValueError(
                        "formal atomic challenge batch exceeds source bound"
                    )
                canonical = {
                    "schema_version": 2,
                    "kind": "lightcone_control_challenge_reservation",
                    "reserved_ns": now_ns,
                    "challenge_sha256s": list(reservations),
                }
                identity = _canonical_sha256(canonical)
                body = (
                    json.dumps(
                        canonical,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                if len(body) > MAXIMUM_CHALLENGE_RESERVATION_BYTES:
                    raise ValueError(
                        "challenge reservation exceeds the bounded record size"
                    )
                output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                output_flags |= getattr(os, "O_CLOEXEC", 0)
                if hasattr(os, "O_NOFOLLOW"):
                    output_flags |= os.O_NOFOLLOW
                output_fd = os.open(
                    f"reservation-{identity}.json",
                    output_flags,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    offset = 0
                    while offset < len(body):
                        offset += os.write(output_fd, body[offset:])
                    os.fsync(output_fd)
                finally:
                    os.close(output_fd)
                os.fsync(directory_fd)
                return results
            finally:
                os.close(lock_fd)
        finally:
            os.close(directory_fd)

    def bind_reservation(
        self, reservation_sha256: str
    ) -> ChallengeReplayReservationBinding:
        """Return a path/raw-bound view of one immutable reservation record."""

        return ChallengeReplayReservationBinding.from_store(
            self, reservation_sha256=reservation_sha256
        )

    def reserve_verified_content_challenges(
        self,
        challenge_sha256s: tuple[str, ...],
        *,
        reserved_ns: int,
    ) -> ChallengeReplayReservationBinding:
        """Atomically reserve a preverified content-authorization batch.

        This primitive does not confer content authority.  The content verifier
        must first validate the typed root signatures and subsequently embeds
        this immutable binding in a ``ContentVerificationReceipt``.  The lock
        still closes concurrent replay: a competing process can consume none of
        the same challenges between the verifier's snapshot and this commit.
        """

        if type(challenge_sha256s) is not tuple:
            raise TypeError("content challenge reservation requires an exact tuple")
        if not challenge_sha256s or challenge_sha256s != tuple(
            sorted(set(challenge_sha256s))
        ):
            raise ValueError("content challenge reservation is not canonical")
        for digest in challenge_sha256s:
            _require_sha256("content authorization challenge", digest)
        if type(reserved_ns) is not int or reserved_ns < 1:
            raise ValueError("content challenge reservation time is invalid")
        self._locked_verify_and_reserve(
            (),
            expected_inventory_sha256="0" * 64,
            now_ns=reserved_ns,
            additional_challenge_sha256s=challenge_sha256s,
        )
        canonical = {
            "schema_version": 2,
            "kind": "lightcone_control_challenge_reservation",
            "reserved_ns": reserved_ns,
            "challenge_sha256s": list(challenge_sha256s),
        }
        return self.bind_reservation(_canonical_sha256(canonical))


@dataclass(frozen=True)
class ChallengeReplayReservationBinding:
    schema_version: int
    kind: Literal["lightcone_challenge_replay_reservation_binding"]
    path: str
    reservation_sha256: str
    raw_sha256: str
    size: int
    reserved_ns: int
    challenge_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_challenge_replay_reservation_binding"
        ):
            raise ValueError("challenge reservation binding schema is unsupported")
        path = Path(self.path)
        if not path.is_absolute() or Path(os.path.abspath(path)) != path:
            raise ValueError("challenge reservation binding path must be absolute")
        _require_sha256(
            "challenge reservation binding identity", self.reservation_sha256
        )
        _require_sha256("challenge reservation binding raw file", self.raw_sha256)
        if path.name != f"reservation-{self.reservation_sha256}.json":
            raise ValueError("challenge reservation binding path differs from identity")
        if (
            type(self.size) is not int
            or self.size < 1
            or self.size > MAXIMUM_CHALLENGE_RESERVATION_BYTES
        ):
            raise ValueError("challenge reservation binding size is invalid")
        if type(self.reserved_ns) is not int or self.reserved_ns < 1:
            raise ValueError("challenge reservation binding time is invalid")
        if (
            type(self.challenge_sha256s) is not tuple
            or not self.challenge_sha256s
            or self.challenge_sha256s != tuple(sorted(set(self.challenge_sha256s)))
        ):
            raise ValueError(
                "challenge reservation binding challenges are not canonical"
            )
        for digest in self.challenge_sha256s:
            _require_sha256("challenge reservation binding challenge", digest)

    @staticmethod
    def _observe(
        root: Path, reservation_sha256: str
    ) -> tuple[Path, bytes, int, tuple[str, ...]]:
        _require_sha256("challenge reservation identity", reservation_sha256)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        directory_fd = os.open(root, flags)
        try:
            status = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) & 0o022
            ):
                raise ValueError("challenge replay store directory is unsafe")
            name = f"reservation-{reservation_sha256}.json"
            reserved_ns, challenges = ChallengeReplayStore._read_reservation(
                directory_fd, name
            )
            canonical = {
                "schema_version": 2,
                "kind": "lightcone_control_challenge_reservation",
                "reserved_ns": reserved_ns,
                "challenge_sha256s": list(challenges),
            }
            body = (
                json.dumps(
                    canonical,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n"
            )
            if _canonical_sha256(canonical) != reservation_sha256:
                raise ValueError("challenge reservation identity differs")
            return root / name, body, reserved_ns, challenges
        finally:
            os.close(directory_fd)

    @classmethod
    def from_store(
        cls,
        store: ChallengeReplayStore,
        *,
        reservation_sha256: str,
    ) -> Self:
        if type(store) is not ChallengeReplayStore:
            raise TypeError("reservation binding requires an exact replay store")
        path, body, reserved_ns, challenges = cls._observe(
            Path(store.root), reservation_sha256
        )
        return cls(
            schema_version=1,
            kind="lightcone_challenge_replay_reservation_binding",
            path=str(path),
            reservation_sha256=reservation_sha256,
            raw_sha256=hashlib.sha256(body).hexdigest(),
            size=len(body),
            reserved_ns=reserved_ns,
            challenge_sha256s=challenges,
        )

    def revalidate(self) -> tuple[str, ...]:
        self.__post_init__()
        identity = Path(self.path)
        rebound = relocated_evidence_path(identity)
        path, body, reserved_ns, challenges = self._observe(
            rebound.parent, self.reservation_sha256
        )
        if (
            path.name != identity.name
            or hashlib.sha256(body).hexdigest() != self.raw_sha256
            or len(body) != self.size
            or reserved_ns != self.reserved_ns
            or challenges != self.challenge_sha256s
        ):
            raise ValueError("challenge reservation binding changed")
        return challenges

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "path": self.path,
            "reservation_sha256": self.reservation_sha256,
            "raw_sha256": self.raw_sha256,
            "size": self.size,
            "reserved_ns": self.reserved_ns,
            "challenge_sha256s": list(self.challenge_sha256s),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "challenge reservation binding",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "path",
                    "reservation_sha256",
                    "raw_sha256",
                    "size",
                    "reserved_ns",
                    "challenge_sha256s",
                }
            ),
        )
        challenges = row.pop("challenge_sha256s")
        if type(challenges) is not list:
            raise TypeError("challenge reservation binding challenges must be an array")
        return cls(**row, challenge_sha256s=tuple(challenges))


def _canonical_sha256(value: object) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


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


@dataclass(frozen=True)
class ControlArtifactSubject:
    """Typed lineage bound into an attestation challenge subject."""

    schema_version: int
    kind: Literal["lightcone_control_artifact_subject"]
    artifact_type: ControlArtifactType
    artifact_sha256: str
    protocol_sha256: str
    registry_sha256: str
    lineage_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "lightcone_control_artifact_subject"
        ):
            raise ValueError("control artifact subject schema is unsupported")
        if self.artifact_type not in CONTROL_ARTIFACT_TYPES:
            raise ValueError("control artifact type is unsupported")
        for label, digest in (
            ("artifact", self.artifact_sha256),
            ("protocol", self.protocol_sha256),
            ("registry", self.registry_sha256),
            ("lineage", self.lineage_sha256),
        ):
            _require_sha256(f"control subject {label}", digest)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "artifact_type": self.artifact_type,
            "artifact_sha256": self.artifact_sha256,
            "protocol_sha256": self.protocol_sha256,
            "registry_sha256": self.registry_sha256,
            "lineage_sha256": self.lineage_sha256,
        }

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "control artifact subject",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "artifact_type",
                    "artifact_sha256",
                    "protocol_sha256",
                    "registry_sha256",
                    "lineage_sha256",
                }
            ),
        )
        return cls(**row)


def _challenge_from_dict(value: object) -> AttestationChallenge:
    row = _strict_object(
        "control attestation challenge",
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


def _attestation_from_dict(value: object) -> SignedAttestation:
    row = _strict_object(
        "control signed attestation",
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


@dataclass(frozen=True)
class ControlArtifactAttestation:
    """Public-key-only signature envelope for one typed control artifact."""

    schema_version: int
    kind: Literal["lightcone_control_artifact_attestation"]
    subject: ControlArtifactSubject
    hardware_envelope_sha256: str
    trust_anchor_sha256: str
    trust_bundle_sha256: str
    trusted_attester_policy_sha256: str
    deployment_policy_authorization: DeploymentPolicyAuthorization
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "lightcone_control_artifact_attestation"
        ):
            raise ValueError("control artifact attestation schema is unsupported")
        if type(self.subject) is not ControlArtifactSubject:
            raise TypeError("control attestation requires an exact typed subject")
        if type(self.challenge) is not AttestationChallenge:
            raise TypeError("control attestation requires an exact challenge")
        if type(self.attestation) is not SignedAttestation:
            raise TypeError("control attestation requires an exact signature")
        if type(self.deployment_policy_authorization) is not (
            DeploymentPolicyAuthorization
        ):
            raise TypeError("control attestation requires an exact deployment policy")
        for label, digest in (
            ("hardware envelope", self.hardware_envelope_sha256),
            ("trust anchor", self.trust_anchor_sha256),
            ("trust bundle", self.trust_bundle_sha256),
            ("trusted policy", self.trusted_attester_policy_sha256),
        ):
            _require_sha256(f"control attestation {label}", digest)
        self.challenge.validate()
        self.attestation.validate()
        if self.challenge.subject_sha256 != self.subject.sha256:
            raise ValueError("control challenge subject differs from typed artifact")
        if (
            self.attestation.challenge_sha256 != self.challenge.sha256
            or self.attestation.payload_sha256 != self.subject.artifact_sha256
        ):
            raise ValueError("control signature differs from challenge or artifact")

    def _payload(self) -> dict[str, object]:
        from dataclasses import asdict

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "subject": self.subject.to_dict(),
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "trust_anchor_sha256": self.trust_anchor_sha256,
            "trust_bundle_sha256": self.trust_bundle_sha256,
            "trusted_attester_policy_sha256": (self.trusted_attester_policy_sha256),
            "deployment_policy_authorization": (
                self.deployment_policy_authorization.to_dict()
            ),
            "challenge": asdict(self.challenge),
            "attestation": asdict(self.attestation),
        }

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"envelope_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "control artifact attestation",
            value,
            frozenset(
                {
                    "envelope_sha256",
                    "schema_version",
                    "kind",
                    "subject",
                    "hardware_envelope_sha256",
                    "trust_anchor_sha256",
                    "trust_bundle_sha256",
                    "trusted_attester_policy_sha256",
                    "deployment_policy_authorization",
                    "challenge",
                    "attestation",
                }
            ),
        )
        declared = row.pop("envelope_sha256")
        subject = ControlArtifactSubject.from_dict(row.pop("subject"))
        deployment_policy_authorization = DeploymentPolicyAuthorization.from_dict(
            row.pop("deployment_policy_authorization")
        )
        challenge = _challenge_from_dict(row.pop("challenge"))
        attestation = _attestation_from_dict(row.pop("attestation"))
        envelope = cls(
            subject=subject,
            deployment_policy_authorization=deployment_policy_authorization,
            challenge=challenge,
            attestation=attestation,
            **row,
        )
        if declared != envelope.sha256:
            raise ValueError("control artifact envelope SHA-256 mismatch")
        return envelope


@dataclass(frozen=True)
class VerifiedControlArtifact:
    artifact_type: ControlArtifactType
    artifact_sha256: str
    envelope_sha256: str
    challenge_sha256: str
    deployment_policy_challenge_sha256: str
    deployment_policy_authorization_sha256: str
    trust_bundle_sha256: str
    trusted_attester_policy_sha256: str

    def __post_init__(self) -> None:
        if self.artifact_type not in CONTROL_ARTIFACT_TYPES:
            raise ValueError("verified control artifact type is unsupported")
        for label, digest in (
            ("artifact", self.artifact_sha256),
            ("envelope", self.envelope_sha256),
            ("challenge", self.challenge_sha256),
            (
                "deployment policy challenge",
                self.deployment_policy_challenge_sha256,
            ),
            (
                "deployment policy authorization",
                self.deployment_policy_authorization_sha256,
            ),
            ("trust bundle", self.trust_bundle_sha256),
            ("trusted policy", self.trusted_attester_policy_sha256),
        ):
            _require_sha256(f"verified control {label}", digest)


def verify_release_control_artifact_attestation(
    envelope: ControlArtifactAttestation,
    *,
    expected_inventory_sha256: str,
    now_ns: int,
    consumed_challenge_sha256s: tuple[str, ...],
) -> VerifiedControlArtifact:
    """Verify and return the challenge digest that must be atomically reserved.

    The caller must commit both ``result.challenge_sha256`` and
    ``result.deployment_policy_challenge_sha256`` to its external single-use
    store in the same transaction that accepts the artifact.  A serialized
    replay set supplied by the artifact is never trusted.
    """

    if type(envelope) is not ControlArtifactAttestation:
        raise TypeError("release control verification requires an exact envelope")
    envelope.__post_init__()
    _require_sha256("release control inventory", expected_inventory_sha256)
    try:
        deployment = verify_source_signed_deployment_policy(
            envelope.deployment_policy_authorization,
            expected_inventory_sha256=expected_inventory_sha256,
            now_ns=now_ns,
            consumed_challenge_sha256s=consumed_challenge_sha256s,
        )
    except RuntimeError as error:
        raise ReleaseControlAttestationBlocked(
            "source_release_trust_root_unavailable"
        ) from error
    if (
        envelope.trust_anchor_sha256 != deployment.root_binding.sha256
        or envelope.trust_bundle_sha256 != deployment.bundle.sha256
        or envelope.trusted_attester_policy_sha256
        != deployment.bundle.trusted_attester_policy.sha256
    ):
        raise ValueError("control attestation uses another release trust root")
    deployment.bundle.require_hardware_envelope(envelope.hardware_envelope_sha256)
    if deployment.bundle.hardware_envelope_sha256_allowlist != (
        envelope.hardware_envelope_sha256,
    ):
        raise ValueError(
            "deployment policy must name exactly the observed hardware envelope"
        )
    challenge_sha256 = deployment.bundle.validate_challenge(
        envelope.challenge,
        now_ns=now_ns,
        consumed_challenge_sha256s=consumed_challenge_sha256s,
    )
    if challenge_sha256 == deployment.challenge_sha256:
        raise ValueError("deployment and artifact challenges must be distinct")
    deployment.bundle.trusted_attester_policy.verify_release(
        envelope.challenge,
        envelope.attestation,
        payload_sha256=envelope.subject.artifact_sha256,
        now_ns=now_ns,
    )
    return VerifiedControlArtifact(
        artifact_type=envelope.subject.artifact_type,
        artifact_sha256=envelope.subject.artifact_sha256,
        envelope_sha256=envelope.sha256,
        challenge_sha256=challenge_sha256,
        deployment_policy_challenge_sha256=deployment.challenge_sha256,
        deployment_policy_authorization_sha256=(deployment.authorization_sha256),
        trust_bundle_sha256=deployment.bundle.sha256,
        trusted_attester_policy_sha256=(
            deployment.bundle.trusted_attester_policy.sha256
        ),
    )


def verify_and_reserve_release_control_artifact_attestations(
    envelopes: tuple[ControlArtifactAttestation, ...],
    *,
    expected_inventory_sha256: str,
    now_ns: int,
    replay_store: ChallengeReplayStore,
    additional_challenge_sha256s: tuple[str, ...] = (),
) -> tuple[VerifiedControlArtifact, ...]:
    """Verify a decision batch and atomically reserve every challenge layer.

    ``additional_challenge_sha256s`` is for signed wrapper authorities whose
    inner challenges must commit in the same replay-store transaction as the
    deployment and control-artifact challenges.  The digests are never trusted
    as proof of a wrapper signature; callers must verify that signature first.
    """

    if (
        type(envelopes) is not tuple
        or not envelopes
        or any(type(row) is not ControlArtifactAttestation for row in envelopes)
    ):
        raise TypeError("control verification batch requires exact envelopes")
    if type(replay_store) is not ChallengeReplayStore:
        raise TypeError("control verification requires an exact replay store")
    if type(additional_challenge_sha256s) is not tuple:
        raise TypeError("additional challenges must be an exact tuple")
    for digest in additional_challenge_sha256s:
        _require_sha256("additional challenge", digest)
    if len(additional_challenge_sha256s) != len(set(additional_challenge_sha256s)):
        raise ValueError("additional challenges must be unique")
    return replay_store._locked_verify_and_reserve(
        envelopes,
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=now_ns,
        additional_challenge_sha256s=additional_challenge_sha256s,
    )


def control_challenge_reservation_sha256(
    results: tuple[VerifiedControlArtifact, ...],
    *,
    reserved_ns: int,
    additional_challenge_sha256s: tuple[str, ...] = (),
) -> str:
    """Return the immutable reservation-record identity for verified results."""

    if (
        type(results) is not tuple
        or not results
        or any(type(row) is not VerifiedControlArtifact for row in results)
    ):
        raise TypeError("challenge reservation identity requires verified results")
    if type(reserved_ns) is not int or reserved_ns < 1:
        raise ValueError("challenge reservation time is invalid")
    if type(additional_challenge_sha256s) is not tuple:
        raise TypeError("additional challenges must be an exact tuple")
    for digest in additional_challenge_sha256s:
        _require_sha256("additional challenge", digest)
    if len(additional_challenge_sha256s) != len(set(additional_challenge_sha256s)):
        raise ValueError("additional challenges must be unique")
    control_challenges = {
        *(row.challenge_sha256 for row in results),
        *(row.deployment_policy_challenge_sha256 for row in results),
    }
    if control_challenges & set(additional_challenge_sha256s):
        raise ValueError("additional and control challenges overlap")
    challenges = tuple(
        sorted(
            {
                *control_challenges,
                *additional_challenge_sha256s,
            }
        )
    )
    if len(challenges) > MAXIMUM_FORMAL_ATOMIC_CHALLENGES:
        raise ValueError("formal atomic challenge batch exceeds source bound")
    canonical = {
        "schema_version": 2,
        "kind": "lightcone_control_challenge_reservation",
        "reserved_ns": reserved_ns,
        "challenge_sha256s": list(challenges),
    }
    if (
        len(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        > MAXIMUM_CHALLENGE_RESERVATION_BYTES
    ):
        raise ValueError("challenge reservation exceeds the bounded record size")
    return _canonical_sha256(canonical)


__all__ = [
    "CONTROL_ARTIFACT_TYPES",
    "MAXIMUM_CHALLENGE_RESERVATION_BYTES",
    "MAXIMUM_FORMAL_ATOMIC_CHALLENGES",
    "ChallengeReplayReservationBinding",
    "ChallengeReplayStore",
    "ControlArtifactAttestation",
    "ControlArtifactSubject",
    "ReleaseControlAttestationBlocked",
    "VerifiedControlArtifact",
    "control_challenge_reservation_sha256",
    "verify_and_reserve_release_control_artifact_attestations",
    "verify_release_control_artifact_attestation",
]
