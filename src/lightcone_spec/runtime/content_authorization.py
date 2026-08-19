"""Offline-root authorizations for dynamic formal data and model content.

The source release pins only the public Ed25519 root.  Exact workload bytes and
prepared snapshot manifests are observed after deployment, then authorized by
short-lived, challenge-bound wrappers.  This module verifies those wrappers;
it deliberately exposes no signer and never accepts a bare digest as release
authority.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import stat
from collections.abc import Collection
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from lightcone_spec.runtime.attestation import AttestationChallenge, attestation_message
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
)
from lightcone_spec.runtime.proof_artifact import relocated_evidence_path
from lightcone_spec.runtime.release_trust_root import (
    DEPLOYMENT_POLICY_MAXIMUM_CLOCK_SKEW_NS,
    DEPLOYMENT_POLICY_MAXIMUM_LIFETIME_NS,
    DEPLOYMENT_POLICY_MINIMUM_LIFETIME_NS,
    load_source_release_ed25519_root,
)

_WORKLOAD_IDS = ("livecodebench_v6_hard", "math500_level5")
_MODEL_ROLES = frozenset({"target", "drafter", "tokenizer"})
_DATASET_DOMAINS = frozenset({"burstgpt_six_source", "e0_task_native"})
_DATASET_FORMATS = frozenset(
    {"canonical_json_array", "canonical_jsonl", "rfc4180_csv_utf8"}
)
_FORMAL_STAGE_IDS = (
    "preflight",
    "E3a",
    "TTS-Cal",
    "E1",
    "E2",
    "E4",
    "E3b",
    "E1a",
    "E5",
    "E6",
    "E0",
)
_BANNED_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
_MAX_DATASET_SOURCE_BYTES = 512 * 1024 * 1024
_MAX_DATASET_DERIVED_BYTES = 512 * 1024 * 1024
_VERIFICATION_SEAL = object()
_MASTER_CONTENT_ARTIFACT_ID = "content:master_verification_receipt"
_POST_MASTER_DERIVED_CONTENT_PREFIXES = (
    "derived_formal_serving_request_schedule:",
    "formal_workload_authority:",
)
_MASTER_AUTHORIZATION_IDS = (
    "dataset:burstgpt_six_source",
    "dataset:e0_task_native",
    "prepared:formal_dag",
    "workload:e3a",
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _require_revision(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one immutable 40-hex revision")
    return value


def _reject_banned_model(value: object) -> None:
    """Apply the protocol-wide ban without trusting the offline signer."""

    if value == _BANNED_MODEL_ID:
        raise ValueError("content authorization contains the globally banned model")
    if isinstance(value, str):
        if _BANNED_MODEL_ID in value:
            raise ValueError("content authorization contains the globally banned model")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_banned_model(key)
            _reject_banned_model(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _reject_banned_model(item)


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or "\n" in value
        or "\r" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be non-empty single-line text")
    return value


def _require_filter_literal(label: str, value: object) -> str | int:
    if type(value) is str:
        return _require_text(label, value)
    if type(value) is int:
        return value
    raise TypeError(f"{label} must be exact JSON string or integer")


def _positive_int(label: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strict_object(label: str, value: object, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    if set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


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


def _challenge_from_dict(value: object) -> AttestationChallenge:
    row = _strict_object(
        "content authorization challenge",
        value,
        {
            "schema_version",
            "kind",
            "challenge_id",
            "nonce_base64",
            "subject_sha256",
            "issued_ns",
            "expires_ns",
        },
    )
    challenge = AttestationChallenge(**row)
    challenge.validate()
    return challenge


def _validate_authorization_challenge(
    challenge: AttestationChallenge,
    *,
    subject_sha256: str,
    now_ns: int,
    consumed_challenge_sha256s: Collection[str],
) -> None:
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("content authorization verification time is invalid")
    if challenge.subject_sha256 != subject_sha256:
        raise ValueError("content authorization challenge subject differs")
    challenge.validate(now_ns=now_ns)
    lifetime = challenge.expires_ns - challenge.issued_ns
    if not (
        DEPLOYMENT_POLICY_MINIMUM_LIFETIME_NS
        <= lifetime
        <= DEPLOYMENT_POLICY_MAXIMUM_LIFETIME_NS
    ):
        raise ValueError("content authorization challenge lifetime is unsupported")
    if challenge.issued_ns > now_ns + DEPLOYMENT_POLICY_MAXIMUM_CLOCK_SKEW_NS:
        raise ValueError("content authorization challenge is too far in the future")
    consumed = tuple(consumed_challenge_sha256s)
    if isinstance(consumed_challenge_sha256s, (str, bytes)) or any(
        _require_sha256("consumed content challenge", digest) != digest
        for digest in consumed
    ):
        raise TypeError("content authorization replay snapshot is invalid")
    if len(consumed) != len(set(consumed)):
        raise ValueError("content authorization replay snapshot has duplicates")
    if challenge.sha256 in consumed:
        raise ValueError("content authorization challenge was already consumed")


def _verify_root_signature(
    *,
    root_manifest_sha256: str,
    challenge: AttestationChallenge,
    payload_sha256: str,
    signature_base64: str,
) -> str:
    root = load_source_release_ed25519_root()
    if root_manifest_sha256 != root.semantic_sha256:
        raise ValueError("content authorization uses another source root")
    public = _decode_base64(
        "content authorization root public key",
        root.root.public_key_base64,
        length=32,
    )
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(
            _decode_base64(
                "content authorization signature", signature_base64, length=64
            ),
            attestation_message(challenge, payload_sha256=payload_sha256),
        )
    except InvalidSignature as error:
        raise ValueError("content authorization root signature is invalid") from error
    return root.sha256


@dataclass(frozen=True)
class ContentJsonArtifactBinding:
    """Immutable path/raw/semantic binding for one authorization or input."""

    artifact_id: str
    path: str
    raw_sha256: str
    semantic_sha256: str
    size: int

    def __post_init__(self) -> None:
        _require_text("content artifact ID", self.artifact_id)
        path = Path(self.path)
        if not path.is_absolute() or path.resolve(strict=False) != path:
            raise ValueError("content artifact path must be absolute and resolved")
        _require_sha256("content artifact raw file", self.raw_sha256)
        _require_sha256("content artifact semantic value", self.semantic_sha256)
        _positive_int("content artifact size", self.size)

    @classmethod
    def from_path(cls, artifact_id: str, path: str | Path) -> Self:
        source = Path(path)
        body = _stable_content_bytes(
            str(source),
            label=f"content artifact {artifact_id}",
            maximum_bytes=64 * 1024 * 1024,
        )
        value = _canonical_json_file(body, label=f"content artifact {artifact_id}")
        return cls(
            artifact_id=artifact_id,
            path=str(source),
            raw_sha256=hashlib.sha256(body).hexdigest(),
            semantic_sha256=_canonical_sha256(value),
            size=len(body),
        )

    def load(self) -> object:
        body = _stable_content_bytes(
            str(relocated_evidence_path(self.path)),
            label=f"bound content artifact {self.artifact_id}",
            maximum_bytes=64 * 1024 * 1024,
        )
        value = _canonical_json_file(
            body, label=f"bound content artifact {self.artifact_id}"
        )
        if (
            len(body) != self.size
            or hashlib.sha256(body).hexdigest() != self.raw_sha256
            or _canonical_sha256(value) != self.semantic_sha256
        ):
            raise RuntimeError(f"bound content artifact {self.artifact_id} changed")
        return value

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict_object(
                "content JSON artifact binding",
                value,
                {field.name for field in dataclass_fields(cls)},
            )
        )


def _reject_post_master_derived_content(
    content: tuple[ContentJsonArtifactBinding, ...],
) -> None:
    if any(
        row.artifact_id.startswith(_POST_MASTER_DERIVED_CONTENT_PREFIXES)
        for row in content
    ):
        raise ValueError(
            "content verification cannot pre-authorize a post-master derived artifact"
        )


@dataclass(frozen=True)
class AuthorizedWorkloadSource:
    workload_id: Literal["livecodebench_v6_hard", "math500_level5"]
    repository: str
    dataset_config: str
    split: str
    repository_revision: str
    raw_file_sha256: str
    raw_file_size: int
    raw_row_count: int
    filter_field: str
    filter_value: str | int
    prompt_compiler: str
    selection_policy: str
    selected_row_count: int
    selected_rows_sha256: str
    protocol_sha256: str

    def __post_init__(self) -> None:
        if self.workload_id not in _WORKLOAD_IDS:
            raise ValueError("authorized workload ID is unsupported")
        for label, value in (
            ("repository", self.repository),
            ("dataset config", self.dataset_config),
            ("split", self.split),
            ("filter field", self.filter_field),
            ("prompt compiler", self.prompt_compiler),
            ("selection policy", self.selection_policy),
        ):
            _require_text(f"authorized workload {label}", value)
        _require_filter_literal(
            "authorized workload filter value",
            self.filter_value,
        )
        _require_revision("authorized workload revision", self.repository_revision)
        _require_sha256("authorized workload raw file", self.raw_file_sha256)
        _require_sha256("authorized workload selected rows", self.selected_rows_sha256)
        _require_sha256("authorized workload protocol", self.protocol_sha256)
        _positive_int("authorized workload raw file size", self.raw_file_size)
        _positive_int("authorized workload raw row count", self.raw_row_count)
        _positive_int("authorized workload selected row count", self.selected_row_count)
        if self.selected_row_count > self.raw_row_count:
            raise ValueError("authorized workload selection exceeds raw rows")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "authorized workload source",
            value,
            {field.name for field in dataclass_fields(cls)},
        )
        return cls(**row)


@dataclass(frozen=True)
class TtsCalibrationTuningWindowEntry:
    """One disjoint TTS request bound to a root-authorized workload row."""

    workload_id: str
    source_sample_id: str
    source_descriptor_sha256: str
    prompt_sha256: str

    def __post_init__(self) -> None:
        if self.workload_id not in _WORKLOAD_IDS:
            raise ValueError("TTS window workload is not registered")
        _require_text("TTS window source sample", self.source_sample_id)
        _require_sha256("TTS window source descriptor", self.source_descriptor_sha256)
        _require_sha256("TTS window prompt", self.prompt_sha256)

    @cached_property
    def entry_id(self) -> str:
        return _canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict_object(
                "TTS calibration tuning-window entry",
                value,
                set(cls.__dataclass_fields__),
            )
        )


@dataclass(frozen=True)
class TtsCalibrationTuningWindow:
    """Disjoint tuning/exclusion rows; prompts remain in signed raw sources."""

    schema_version: Literal[2]
    kind: Literal["lightcone_tts_disjoint_tuning_window_source"]
    tuning_entries: tuple[TtsCalibrationTuningWindowEntry, ...]
    excluded_pilot_entries: tuple[TtsCalibrationTuningWindowEntry, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.kind != "lightcone_tts_disjoint_tuning_window_source"
        ):
            raise ValueError("TTS calibration tuning-window schema differs")
        for label, rows in (
            ("tuning", self.tuning_entries),
            ("excluded pilot", self.excluded_pilot_entries),
        ):
            if (
                type(rows) is not tuple
                or not rows
                or any(type(row) is not TtsCalibrationTuningWindowEntry for row in rows)
                or tuple(row.entry_id for row in rows)
                != tuple(sorted({row.entry_id for row in rows}))
            ):
                raise ValueError(f"TTS calibration {label} entries are not canonical")
        if len(self.excluded_pilot_entries) != 4:
            raise ValueError("TTS calibration requires exactly four excluded pilots")
        if {row.entry_id for row in self.tuning_entries} & {
            row.entry_id for row in self.excluded_pilot_entries
        }:
            raise ValueError("TTS calibration tuning window overlaps excluded pilots")
        sample_keys = tuple(
            (row.workload_id, row.source_sample_id)
            for row in (*self.tuning_entries, *self.excluded_pilot_entries)
        )
        if len(sample_keys) != len(set(sample_keys)):
            raise ValueError("TTS calibration window reuses a source sample")
        if len({row.workload_id for row in self.entries}) != 1:
            raise ValueError("TTS calibration window must use one exact workload")

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @property
    def entries(self) -> tuple[TtsCalibrationTuningWindowEntry, ...]:
        return self.tuning_entries + self.excluded_pilot_entries

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "tuning_entries": [row.to_dict() for row in self.tuning_entries],
            "excluded_pilot_entries": [
                row.to_dict() for row in self.excluded_pilot_entries
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "TTS calibration tuning window",
            value,
            set(cls.__dataclass_fields__),
        )
        tuning = row.pop("tuning_entries")
        excluded = row.pop("excluded_pilot_entries")
        if type(tuning) is not list or type(excluded) is not list:
            raise TypeError("TTS calibration tuning-window rows are not arrays")
        return cls(
            **row,
            tuning_entries=tuple(
                TtsCalibrationTuningWindowEntry.from_dict(item) for item in tuning
            ),
            excluded_pilot_entries=tuple(
                TtsCalibrationTuningWindowEntry.from_dict(item) for item in excluded
            ),
        )


@dataclass(frozen=True)
class ReleaseWorkloadSourceAuthorizationSource:
    """Unsigned, typed source input for the offline workload ceremony."""

    schema_version: int
    kind: Literal["lightcone_release_workload_source_authorization_source"]
    root_manifest_sha256: str
    workload_sources: tuple[AuthorizedWorkloadSource, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "lightcone_release_workload_source_authorization_source"
        ):
            raise ValueError("workload authorization source schema is unsupported")
        _require_sha256("workload authorization source root", self.root_manifest_sha256)
        if (
            type(self.workload_sources) is not tuple
            or tuple(row.workload_id for row in self.workload_sources) != _WORKLOAD_IDS
            or any(
                type(row) is not AuthorizedWorkloadSource
                for row in self.workload_sources
            )
        ):
            raise ValueError(
                "workload authorization source must cover both sources exactly"
            )

    @cached_property
    def subject_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_release_workload_source_subject",
                "root_manifest_sha256": self.root_manifest_sha256,
                "workload_sources": [row.to_dict() for row in self.workload_sources],
            }
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "root_manifest_sha256": self.root_manifest_sha256,
            "workload_sources": [row.to_dict() for row in self.workload_sources],
        }

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"source_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "workload authorization source",
            value,
            {
                "source_sha256",
                "schema_version",
                "kind",
                "root_manifest_sha256",
                "workload_sources",
            },
        )
        declared = row.pop("source_sha256")
        raw_sources = row.pop("workload_sources")
        if type(raw_sources) is not list:
            raise TypeError("workload authorization source rows must be an array")
        source = cls(
            **row,
            workload_sources=tuple(
                AuthorizedWorkloadSource.from_dict(item) for item in raw_sources
            ),
        )
        if declared != source.sha256:
            raise ValueError("workload authorization source SHA-256 mismatch")
        return source


@dataclass(frozen=True)
class ReleaseWorkloadSourceAuthorization:
    schema_version: int
    kind: Literal["lightcone_release_workload_source_authorization"]
    root_manifest_sha256: str
    workload_sources: tuple[AuthorizedWorkloadSource, ...]
    challenge: AttestationChallenge
    signature_base64: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "lightcone_release_workload_source_authorization"
        ):
            raise ValueError("workload authorization schema is unsupported")
        _require_sha256("workload authorization root", self.root_manifest_sha256)
        if (
            type(self.workload_sources) is not tuple
            or tuple(row.workload_id for row in self.workload_sources) != _WORKLOAD_IDS
            or any(
                type(row) is not AuthorizedWorkloadSource
                for row in self.workload_sources
            )
        ):
            raise ValueError("workload authorization must cover both sources exactly")
        if type(self.challenge) is not AttestationChallenge:
            raise TypeError("workload authorization requires an exact challenge")
        self.challenge.validate()
        if self.challenge.subject_sha256 != self.subject_sha256:
            raise ValueError("workload authorization challenge subject differs")
        _decode_base64(
            "workload authorization signature", self.signature_base64, length=64
        )

    @cached_property
    def subject_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_release_workload_source_subject",
                "root_manifest_sha256": self.root_manifest_sha256,
                "workload_sources": [row.to_dict() for row in self.workload_sources],
            }
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "root_manifest_sha256": self.root_manifest_sha256,
            "workload_sources": [row.to_dict() for row in self.workload_sources],
            "challenge": asdict(self.challenge),
            "signature_base64": self.signature_base64,
        }

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"authorization_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "workload source authorization",
            value,
            {
                "authorization_sha256",
                "schema_version",
                "kind",
                "root_manifest_sha256",
                "workload_sources",
                "challenge",
                "signature_base64",
            },
        )
        declared = row.pop("authorization_sha256")
        raw_sources = row.pop("workload_sources")
        if type(raw_sources) is not list:
            raise TypeError("authorized workload sources must be an array")
        challenge = _challenge_from_dict(row.pop("challenge"))
        authorization = cls(
            **row,
            workload_sources=tuple(
                AuthorizedWorkloadSource.from_dict(item) for item in raw_sources
            ),
            challenge=challenge,
        )
        if declared != authorization.sha256:
            raise ValueError("workload authorization SHA-256 mismatch")
        return authorization


class VerifiedReleaseWorkloadSources:
    __slots__ = ("_authorization", "_root_binding_sha256")

    def __init__(
        self,
        authorization: ReleaseWorkloadSourceAuthorization,
        root_binding_sha256: str,
        *,
        _seal: object,
    ) -> None:
        if _seal is not _VERIFICATION_SEAL:
            raise TypeError("verified workload authority is verifier-owned")
        self._authorization = authorization
        self._root_binding_sha256 = root_binding_sha256

    @property
    def authorization(self) -> ReleaseWorkloadSourceAuthorization:
        return self._authorization

    @property
    def authorization_sha256(self) -> str:
        return self._authorization.sha256

    @property
    def challenge_sha256(self) -> str:
        return self._authorization.challenge.sha256

    @property
    def root_binding_sha256(self) -> str:
        return self._root_binding_sha256

    def source(self, workload_id: str) -> AuthorizedWorkloadSource:
        matches = tuple(
            row
            for row in self._authorization.workload_sources
            if row.workload_id == workload_id
        )
        if len(matches) != 1:
            raise ValueError("verified workload source is not exact")
        return matches[0]


def verify_release_workload_source_authorization(
    authorization: ReleaseWorkloadSourceAuthorization,
    *,
    now_ns: int,
    consumed_challenge_sha256s: Collection[str] = (),
) -> VerifiedReleaseWorkloadSources:
    if type(authorization) is not ReleaseWorkloadSourceAuthorization:
        raise TypeError("workload verification requires an exact authorization")
    authorization.__post_init__()
    _validate_authorization_challenge(
        authorization.challenge,
        subject_sha256=authorization.subject_sha256,
        now_ns=now_ns,
        consumed_challenge_sha256s=consumed_challenge_sha256s,
    )
    root_binding = _verify_root_signature(
        root_manifest_sha256=authorization.root_manifest_sha256,
        challenge=authorization.challenge,
        payload_sha256=authorization.subject_sha256,
        signature_base64=authorization.signature_base64,
    )
    return VerifiedReleaseWorkloadSources(
        authorization, root_binding, _seal=_VERIFICATION_SEAL
    )


@dataclass(frozen=True)
class AuthorizedPreparedModel:
    member_id: str
    backend: str
    role: Literal["target", "drafter", "tokenizer"]
    model_id: str
    revision: str
    snapshot_manifest_raw_sha256: str
    snapshot_manifest_semantic_sha256: str

    def __post_init__(self) -> None:
        if self.role not in _MODEL_ROLES:
            raise ValueError("authorized prepared-model role is unsupported")
        _require_text("authorized prepared-model member ID", self.member_id)
        _require_text("authorized prepared-model backend", self.backend)
        _require_text("authorized prepared-model ID", self.model_id)
        _reject_banned_model(self.to_dict())
        _require_revision("authorized prepared-model revision", self.revision)
        _require_sha256(
            "authorized prepared-model raw manifest",
            self.snapshot_manifest_raw_sha256,
        )
        _require_sha256(
            "authorized prepared-model semantic manifest",
            self.snapshot_manifest_semantic_sha256,
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "authorized prepared model",
            value,
            {field.name for field in dataclass_fields(cls)},
        )
        return cls(**row)


@dataclass(frozen=True)
class PreparedModelStageMembership:
    stage_id: str
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("prepared-model stage ID", self.stage_id)
        if self.stage_id not in _FORMAL_STAGE_IDS:
            raise ValueError("prepared-model stage is outside the formal DAG")
        if (
            type(self.member_ids) is not tuple
            or not self.member_ids
            or self.member_ids != tuple(sorted(set(self.member_ids)))
        ):
            raise ValueError(
                "prepared-model stage members must be sorted, unique, and non-empty"
            )
        for member_id in self.member_ids:
            _require_text("prepared-model stage member ID", member_id)

    def to_dict(self) -> dict[str, object]:
        return {"stage_id": self.stage_id, "member_ids": list(self.member_ids)}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "prepared-model stage membership",
            value,
            {"stage_id", "member_ids"},
        )
        member_ids = row["member_ids"]
        if type(member_ids) is not list:
            raise TypeError("prepared-model stage members must be an array")
        return cls(stage_id=row["stage_id"], member_ids=tuple(member_ids))


@dataclass(frozen=True)
class PreparedModelContentReleaseAuthorizationSource:
    """Unsigned, typed source input for the offline prepared-model ceremony."""

    schema_version: int
    kind: Literal["lightcone_prepared_model_content_release_authorization_source"]
    root_manifest_sha256: str
    model_lock_sha256: str
    prepared_model_set_sha256: str
    content_manifest_raw_sha256: str
    content_manifest_semantic_sha256: str
    content_manifest_size: int
    models: tuple[AuthorizedPreparedModel, ...]
    stage_memberships: tuple[PreparedModelStageMembership, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "lightcone_prepared_model_content_release_authorization_source"
        ):
            raise ValueError(
                "prepared-model authorization source schema is unsupported"
            )
        for label, digest in (
            ("root", self.root_manifest_sha256),
            ("model lock", self.model_lock_sha256),
            ("prepared model set", self.prepared_model_set_sha256),
            ("content manifest raw", self.content_manifest_raw_sha256),
            ("content manifest semantic", self.content_manifest_semantic_sha256),
        ):
            _require_sha256(f"prepared-model authorization source {label}", digest)
        _positive_int(
            "prepared-model authorization source manifest size",
            self.content_manifest_size,
        )
        if (
            type(self.models) is not tuple
            or not self.models
            or any(type(row) is not AuthorizedPreparedModel for row in self.models)
            or tuple(row.member_id for row in self.models)
            != tuple(sorted({row.member_id for row in self.models}))
        ):
            raise ValueError(
                "prepared-model authorization source members are not canonical"
            )
        if not {"target", "tokenizer"} <= {row.role for row in self.models}:
            raise ValueError(
                "prepared-model authorization source lacks target or tokenizer"
            )
        if (
            type(self.stage_memberships) is not tuple
            or not self.stage_memberships
            or any(
                type(row) is not PreparedModelStageMembership
                for row in self.stage_memberships
            )
            or tuple(row.stage_id for row in self.stage_memberships)
            != tuple(sorted({row.stage_id for row in self.stage_memberships}))
        ):
            raise ValueError(
                "prepared-model authorization source memberships are not canonical"
            )
        members = {row.member_id for row in self.models}
        referenced = {
            member_id
            for stage in self.stage_memberships
            for member_id in stage.member_ids
        }
        if referenced != members:
            raise ValueError(
                "prepared-model authorization source does not cover members exactly"
            )
        _reject_banned_model(self._payload())

    @cached_property
    def subject_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_prepared_model_content_release_subject",
                "root_manifest_sha256": self.root_manifest_sha256,
                "model_lock_sha256": self.model_lock_sha256,
                "prepared_model_set_sha256": self.prepared_model_set_sha256,
                "content_manifest_raw_sha256": self.content_manifest_raw_sha256,
                "content_manifest_semantic_sha256": self.content_manifest_semantic_sha256,
                "content_manifest_size": self.content_manifest_size,
                "models": [row.to_dict() for row in self.models],
                "stage_memberships": [row.to_dict() for row in self.stage_memberships],
            }
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "root_manifest_sha256": self.root_manifest_sha256,
            "model_lock_sha256": self.model_lock_sha256,
            "prepared_model_set_sha256": self.prepared_model_set_sha256,
            "content_manifest_raw_sha256": self.content_manifest_raw_sha256,
            "content_manifest_semantic_sha256": self.content_manifest_semantic_sha256,
            "content_manifest_size": self.content_manifest_size,
            "models": [row.to_dict() for row in self.models],
            "stage_memberships": [row.to_dict() for row in self.stage_memberships],
        }

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"source_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "prepared-model authorization source",
            value,
            {
                "source_sha256",
                "schema_version",
                "kind",
                "root_manifest_sha256",
                "model_lock_sha256",
                "prepared_model_set_sha256",
                "content_manifest_raw_sha256",
                "content_manifest_semantic_sha256",
                "content_manifest_size",
                "models",
                "stage_memberships",
            },
        )
        declared = row.pop("source_sha256")
        raw_models = row.pop("models")
        raw_memberships = row.pop("stage_memberships")
        if type(raw_models) is not list or type(raw_memberships) is not list:
            raise TypeError("prepared-model authorization source rows must be arrays")
        source = cls(
            **row,
            models=tuple(
                AuthorizedPreparedModel.from_dict(item) for item in raw_models
            ),
            stage_memberships=tuple(
                PreparedModelStageMembership.from_dict(item) for item in raw_memberships
            ),
        )
        if declared != source.sha256:
            raise ValueError("prepared-model authorization source SHA-256 mismatch")
        return source


@dataclass(frozen=True)
class PreparedModelContentReleaseAuthorization:
    schema_version: int
    kind: Literal["lightcone_prepared_model_content_release_authorization"]
    root_manifest_sha256: str
    model_lock_sha256: str
    prepared_model_set_sha256: str
    content_manifest_raw_sha256: str
    content_manifest_semantic_sha256: str
    content_manifest_size: int
    models: tuple[AuthorizedPreparedModel, ...]
    stage_memberships: tuple[PreparedModelStageMembership, ...]
    challenge: AttestationChallenge
    signature_base64: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "lightcone_prepared_model_content_release_authorization"
        ):
            raise ValueError("prepared-model authorization schema is unsupported")
        for label, digest in (
            ("root", self.root_manifest_sha256),
            ("model lock", self.model_lock_sha256),
            ("prepared model set", self.prepared_model_set_sha256),
            ("content manifest raw", self.content_manifest_raw_sha256),
            ("content manifest semantic", self.content_manifest_semantic_sha256),
        ):
            _require_sha256(f"prepared-model authorization {label}", digest)
        _positive_int(
            "prepared-model content manifest size", self.content_manifest_size
        )
        if (
            type(self.models) is not tuple
            or not self.models
            or any(type(row) is not AuthorizedPreparedModel for row in self.models)
            or tuple(row.member_id for row in self.models)
            != tuple(sorted({row.member_id for row in self.models}))
        ):
            raise ValueError("prepared-model authorization members are not canonical")
        roles = {row.role for row in self.models}
        if not {"target", "tokenizer"} <= roles:
            raise ValueError(
                "prepared-model authorization lacks target or tokenizer authority"
            )
        if (
            type(self.stage_memberships) is not tuple
            or not self.stage_memberships
            or any(
                type(row) is not PreparedModelStageMembership
                for row in self.stage_memberships
            )
            or tuple(row.stage_id for row in self.stage_memberships)
            != tuple(sorted({row.stage_id for row in self.stage_memberships}))
        ):
            raise ValueError(
                "prepared-model authorization stage memberships are not canonical"
            )
        members = {row.member_id for row in self.models}
        referenced = {
            member_id
            for stage in self.stage_memberships
            for member_id in stage.member_ids
        }
        if referenced != members:
            raise ValueError(
                "prepared-model stage membership does not cover members exactly"
            )
        if type(self.challenge) is not AttestationChallenge:
            raise TypeError("prepared-model authorization requires an exact challenge")
        self.challenge.validate()
        if self.challenge.subject_sha256 != self.subject_sha256:
            raise ValueError("prepared-model authorization challenge subject differs")
        _decode_base64(
            "prepared-model authorization signature", self.signature_base64, length=64
        )
        _reject_banned_model(
            {
                "models": [row.to_dict() for row in self.models],
                "stage_memberships": [row.to_dict() for row in self.stage_memberships],
            }
        )

    @cached_property
    def subject_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_prepared_model_content_release_subject",
                "root_manifest_sha256": self.root_manifest_sha256,
                "model_lock_sha256": self.model_lock_sha256,
                "prepared_model_set_sha256": self.prepared_model_set_sha256,
                "content_manifest_raw_sha256": self.content_manifest_raw_sha256,
                "content_manifest_semantic_sha256": (
                    self.content_manifest_semantic_sha256
                ),
                "content_manifest_size": self.content_manifest_size,
                "models": [row.to_dict() for row in self.models],
                "stage_memberships": [row.to_dict() for row in self.stage_memberships],
            }
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "root_manifest_sha256": self.root_manifest_sha256,
            "model_lock_sha256": self.model_lock_sha256,
            "prepared_model_set_sha256": self.prepared_model_set_sha256,
            "content_manifest_raw_sha256": self.content_manifest_raw_sha256,
            "content_manifest_semantic_sha256": self.content_manifest_semantic_sha256,
            "content_manifest_size": self.content_manifest_size,
            "models": [row.to_dict() for row in self.models],
            "stage_memberships": [row.to_dict() for row in self.stage_memberships],
            "challenge": asdict(self.challenge),
            "signature_base64": self.signature_base64,
        }

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"authorization_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "prepared-model content authorization",
            value,
            {
                "authorization_sha256",
                "schema_version",
                "kind",
                "root_manifest_sha256",
                "model_lock_sha256",
                "prepared_model_set_sha256",
                "content_manifest_raw_sha256",
                "content_manifest_semantic_sha256",
                "content_manifest_size",
                "models",
                "stage_memberships",
                "challenge",
                "signature_base64",
            },
        )
        declared = row.pop("authorization_sha256")
        raw_models = row.pop("models")
        raw_memberships = row.pop("stage_memberships")
        if type(raw_models) is not list:
            raise TypeError("authorized prepared models must be an array")
        if type(raw_memberships) is not list:
            raise TypeError("prepared-model stage memberships must be an array")
        challenge = _challenge_from_dict(row.pop("challenge"))
        authorization = cls(
            **row,
            models=tuple(
                AuthorizedPreparedModel.from_dict(item) for item in raw_models
            ),
            stage_memberships=tuple(
                PreparedModelStageMembership.from_dict(item) for item in raw_memberships
            ),
            challenge=challenge,
        )
        if declared != authorization.sha256:
            raise ValueError("prepared-model authorization SHA-256 mismatch")
        return authorization


class VerifiedPreparedModelContentRelease:
    __slots__ = ("_authorization", "_root_binding_sha256")

    def __init__(
        self,
        authorization: PreparedModelContentReleaseAuthorization,
        root_binding_sha256: str,
        *,
        _seal: object,
    ) -> None:
        if _seal is not _VERIFICATION_SEAL:
            raise TypeError("verified prepared-model authority is verifier-owned")
        self._authorization = authorization
        self._root_binding_sha256 = root_binding_sha256

    @property
    def authorization(self) -> PreparedModelContentReleaseAuthorization:
        return self._authorization

    @property
    def authorization_sha256(self) -> str:
        return self._authorization.sha256

    @property
    def challenge_sha256(self) -> str:
        return self._authorization.challenge.sha256

    @property
    def root_binding_sha256(self) -> str:
        return self._root_binding_sha256

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(row.stage_id for row in self._authorization.stage_memberships)

    def member(self, member_id: str) -> AuthorizedPreparedModel:
        matches = tuple(
            row for row in self._authorization.models if row.member_id == member_id
        )
        if len(matches) != 1:
            raise ValueError("verified prepared-model member is not exact")
        return matches[0]

    def require_stage(self, stage_id: str) -> tuple[AuthorizedPreparedModel, ...]:
        matches = tuple(
            row
            for row in self._authorization.stage_memberships
            if row.stage_id == stage_id
        )
        if len(matches) != 1:
            raise ValueError("verified prepared-model stage is not exact")
        return tuple(self.member(member_id) for member_id in matches[0].member_ids)

    def require_exact_stage(
        self,
        stage_id: str,
        expected_members: Collection[AuthorizedPreparedModel],
    ) -> tuple[AuthorizedPreparedModel, ...]:
        """Exact-match requirements derived from ProtocolLock/materialized cells."""

        if isinstance(expected_members, (str, bytes)):
            raise TypeError("prepared-model stage requirement must be typed members")
        expected = tuple(sorted(expected_members, key=lambda row: row.member_id))
        if (
            not expected
            or any(type(row) is not AuthorizedPreparedModel for row in expected)
            or tuple(row.member_id for row in expected)
            != tuple(sorted({row.member_id for row in expected}))
        ):
            raise ValueError("prepared-model stage requirement is not canonical")
        observed = self.require_stage(stage_id)
        if observed != expected:
            raise ValueError(
                "prepared-model authorization differs from the exact stage requirement"
            )
        return observed


def verify_prepared_model_content_release_authorization(
    authorization: PreparedModelContentReleaseAuthorization,
    *,
    now_ns: int,
    consumed_challenge_sha256s: Collection[str] = (),
) -> VerifiedPreparedModelContentRelease:
    if type(authorization) is not PreparedModelContentReleaseAuthorization:
        raise TypeError("prepared-model verification requires an exact authorization")
    authorization.__post_init__()
    _validate_authorization_challenge(
        authorization.challenge,
        subject_sha256=authorization.subject_sha256,
        now_ns=now_ns,
        consumed_challenge_sha256s=consumed_challenge_sha256s,
    )
    root_binding = _verify_root_signature(
        root_manifest_sha256=authorization.root_manifest_sha256,
        challenge=authorization.challenge,
        payload_sha256=authorization.subject_sha256,
        signature_base64=authorization.signature_base64,
    )
    return VerifiedPreparedModelContentRelease(
        authorization, root_binding, _seal=_VERIFICATION_SEAL
    )


@dataclass(frozen=True)
class AuthorizedDatasetContentMember:
    member_id: str
    source_uri: str
    revision: str
    data_format: Literal["canonical_json_array", "canonical_jsonl", "rfc4180_csv_utf8"]
    raw_file_sha256: str
    raw_file_size: int
    raw_row_count: int
    selected_rows_raw_sha256: str
    selected_rows_sha256: str
    selected_rows_size: int
    selected_row_count: int
    request_shape_raw_sha256: str
    request_shape_sha256: str
    request_shape_size: int
    protocol_sha256: str

    def __post_init__(self) -> None:
        _require_text("authorized dataset member ID", self.member_id)
        _require_text("authorized dataset source URI", self.source_uri)
        _require_revision("authorized dataset revision", self.revision)
        if self.data_format not in _DATASET_FORMATS:
            raise ValueError("authorized dataset format is unsupported")
        for label, digest in (
            ("raw file", self.raw_file_sha256),
            ("selected rows raw", self.selected_rows_raw_sha256),
            ("selected rows", self.selected_rows_sha256),
            ("request shape raw", self.request_shape_raw_sha256),
            ("request shape", self.request_shape_sha256),
            ("protocol", self.protocol_sha256),
        ):
            _require_sha256(f"authorized dataset {label}", digest)
        _positive_int("authorized dataset raw file size", self.raw_file_size)
        _positive_int("authorized dataset raw row count", self.raw_row_count)
        _positive_int("authorized dataset selected rows size", self.selected_rows_size)
        _positive_int("authorized dataset selected row count", self.selected_row_count)
        _positive_int("authorized dataset request shape size", self.request_shape_size)
        if self.selected_row_count > self.raw_row_count:
            raise ValueError("authorized dataset selection exceeds raw rows")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "authorized dataset content member",
            value,
            {field.name for field in dataclass_fields(cls)},
        )
        return cls(**row)


@dataclass(frozen=True)
class DatasetContentReleaseAuthorizationSource:
    """Unsigned, typed source input for one offline dataset ceremony."""

    schema_version: int
    kind: Literal["lightcone_dataset_content_release_authorization_source"]
    authority_domain: Literal["burstgpt_six_source", "e0_task_native"]
    root_manifest_sha256: str
    members: tuple[AuthorizedDatasetContentMember, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_dataset_content_release_authorization_source"
            or self.authority_domain not in _DATASET_DOMAINS
        ):
            raise ValueError("dataset authorization source schema is unsupported")
        _require_sha256("dataset authorization source root", self.root_manifest_sha256)
        if (
            type(self.members) is not tuple
            or not self.members
            or any(
                type(row) is not AuthorizedDatasetContentMember for row in self.members
            )
            or tuple(row.member_id for row in self.members)
            != tuple(sorted({row.member_id for row in self.members}))
        ):
            raise ValueError("dataset authorization source members are not canonical")
        if self.authority_domain == "burstgpt_six_source" and len(self.members) != 6:
            raise ValueError(
                "BurstGPT authorization source requires exactly six sources"
            )

    @cached_property
    def subject_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_dataset_content_release_subject",
                "authority_domain": self.authority_domain,
                "root_manifest_sha256": self.root_manifest_sha256,
                "members": [row.to_dict() for row in self.members],
            }
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "authority_domain": self.authority_domain,
            "root_manifest_sha256": self.root_manifest_sha256,
            "members": [row.to_dict() for row in self.members],
        }

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"source_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "dataset authorization source",
            value,
            {
                "source_sha256",
                "schema_version",
                "kind",
                "authority_domain",
                "root_manifest_sha256",
                "members",
            },
        )
        declared = row.pop("source_sha256")
        raw_members = row.pop("members")
        if type(raw_members) is not list:
            raise TypeError("dataset authorization source members must be an array")
        source = cls(
            **row,
            members=tuple(
                AuthorizedDatasetContentMember.from_dict(item) for item in raw_members
            ),
        )
        if declared != source.sha256:
            raise ValueError("dataset authorization source SHA-256 mismatch")
        return source


@dataclass(frozen=True)
class DatasetContentReleaseAuthorization:
    schema_version: int
    kind: Literal["lightcone_dataset_content_release_authorization"]
    authority_domain: Literal["burstgpt_six_source", "e0_task_native"]
    root_manifest_sha256: str
    members: tuple[AuthorizedDatasetContentMember, ...]
    challenge: AttestationChallenge
    signature_base64: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_dataset_content_release_authorization"
            or self.authority_domain not in _DATASET_DOMAINS
        ):
            raise ValueError("dataset content authorization schema is unsupported")
        _require_sha256("dataset content authorization root", self.root_manifest_sha256)
        if (
            type(self.members) is not tuple
            or not self.members
            or any(
                type(row) is not AuthorizedDatasetContentMember for row in self.members
            )
            or tuple(row.member_id for row in self.members)
            != tuple(sorted({row.member_id for row in self.members}))
        ):
            raise ValueError("dataset content members are not canonical")
        if self.authority_domain == "burstgpt_six_source" and len(self.members) != 6:
            raise ValueError("BurstGPT authorization requires exactly six sources")
        if type(self.challenge) is not AttestationChallenge:
            raise TypeError("dataset content authorization requires an exact challenge")
        self.challenge.validate()
        if self.challenge.subject_sha256 != self.subject_sha256:
            raise ValueError("dataset content authorization challenge subject differs")
        _decode_base64(
            "dataset content authorization signature", self.signature_base64, length=64
        )

    @cached_property
    def subject_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_dataset_content_release_subject",
                "authority_domain": self.authority_domain,
                "root_manifest_sha256": self.root_manifest_sha256,
                "members": [row.to_dict() for row in self.members],
            }
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "authority_domain": self.authority_domain,
            "root_manifest_sha256": self.root_manifest_sha256,
            "members": [row.to_dict() for row in self.members],
            "challenge": asdict(self.challenge),
            "signature_base64": self.signature_base64,
        }

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"authorization_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "dataset content authorization",
            value,
            {
                "authorization_sha256",
                "schema_version",
                "kind",
                "authority_domain",
                "root_manifest_sha256",
                "members",
                "challenge",
                "signature_base64",
            },
        )
        declared = row.pop("authorization_sha256")
        raw_members = row.pop("members")
        if type(raw_members) is not list:
            raise TypeError("authorized dataset members must be an array")
        challenge = _challenge_from_dict(row.pop("challenge"))
        authorization = cls(
            **row,
            members=tuple(
                AuthorizedDatasetContentMember.from_dict(item) for item in raw_members
            ),
            challenge=challenge,
        )
        if declared != authorization.sha256:
            raise ValueError("dataset content authorization SHA-256 mismatch")
        return authorization


ContentAuthorizationSource = (
    ReleaseWorkloadSourceAuthorizationSource
    | PreparedModelContentReleaseAuthorizationSource
    | DatasetContentReleaseAuthorizationSource
)

CONTENT_AUTHORIZATION_SOURCE_TYPES = (
    "dataset",
    "prepared_model",
    "workload",
)


def content_authorization_source_from_dict(
    artifact_type: str,
    value: object,
) -> ContentAuthorizationSource:
    """Strictly decode one member of the closed offline content allowlist."""

    decoders = {
        "dataset": DatasetContentReleaseAuthorizationSource.from_dict,
        "prepared_model": PreparedModelContentReleaseAuthorizationSource.from_dict,
        "workload": ReleaseWorkloadSourceAuthorizationSource.from_dict,
    }
    if artifact_type not in decoders:
        raise ValueError("content authorization source type is unsupported")
    return decoders[artifact_type](value)


def build_content_authorization_from_source(
    source: ContentAuthorizationSource,
    *,
    challenge: AttestationChallenge,
    signature_base64: str,
) -> (
    ReleaseWorkloadSourceAuthorization
    | PreparedModelContentReleaseAuthorization
    | DatasetContentReleaseAuthorization
):
    """Construct an existing verifier-owned wrapper from one typed source."""

    if type(source) is ReleaseWorkloadSourceAuthorizationSource:
        authorization = ReleaseWorkloadSourceAuthorization(
            schema_version=1,
            kind="lightcone_release_workload_source_authorization",
            root_manifest_sha256=source.root_manifest_sha256,
            workload_sources=source.workload_sources,
            challenge=challenge,
            signature_base64=signature_base64,
        )
    elif type(source) is PreparedModelContentReleaseAuthorizationSource:
        authorization = PreparedModelContentReleaseAuthorization(
            schema_version=1,
            kind="lightcone_prepared_model_content_release_authorization",
            root_manifest_sha256=source.root_manifest_sha256,
            model_lock_sha256=source.model_lock_sha256,
            prepared_model_set_sha256=source.prepared_model_set_sha256,
            content_manifest_raw_sha256=source.content_manifest_raw_sha256,
            content_manifest_semantic_sha256=(source.content_manifest_semantic_sha256),
            content_manifest_size=source.content_manifest_size,
            models=source.models,
            stage_memberships=source.stage_memberships,
            challenge=challenge,
            signature_base64=signature_base64,
        )
    elif type(source) is DatasetContentReleaseAuthorizationSource:
        authorization = DatasetContentReleaseAuthorization(
            schema_version=1,
            kind="lightcone_dataset_content_release_authorization",
            authority_domain=source.authority_domain,
            root_manifest_sha256=source.root_manifest_sha256,
            members=source.members,
            challenge=challenge,
            signature_base64=signature_base64,
        )
    else:
        raise TypeError("content authorization source is not an exact typed source")
    if authorization.subject_sha256 != source.subject_sha256:
        raise RuntimeError("content authorization source changed subject identity")
    return authorization


class VerifiedDatasetContentRelease:
    __slots__ = ("_authorization", "_root_binding_sha256")

    def __init__(
        self,
        authorization: DatasetContentReleaseAuthorization,
        root_binding_sha256: str,
        *,
        _seal: object,
    ) -> None:
        if _seal is not _VERIFICATION_SEAL:
            raise TypeError("verified dataset content authority is verifier-owned")
        self._authorization = authorization
        self._root_binding_sha256 = root_binding_sha256

    @property
    def authorization(self) -> DatasetContentReleaseAuthorization:
        return self._authorization

    @property
    def authorization_sha256(self) -> str:
        return self._authorization.sha256

    @property
    def challenge_sha256(self) -> str:
        return self._authorization.challenge.sha256

    @property
    def root_binding_sha256(self) -> str:
        return self._root_binding_sha256

    @property
    def authority_domain(self) -> str:
        return self._authorization.authority_domain

    def require_members(
        self, member_ids: Collection[str]
    ) -> tuple[AuthorizedDatasetContentMember, ...]:
        required = tuple(sorted(member_ids))
        if isinstance(member_ids, (str, bytes)) or required != tuple(
            sorted(set(required))
        ):
            raise ValueError("required dataset member IDs are not canonical")
        by_id = {row.member_id: row for row in self._authorization.members}
        if any(member_id not in by_id for member_id in required):
            raise ValueError("dataset content authorization lacks a required member")
        return tuple(by_id[member_id] for member_id in required)

    def require_exact_members(
        self,
        expected_members: Collection[AuthorizedDatasetContentMember],
    ) -> tuple[AuthorizedDatasetContentMember, ...]:
        """Exact-match a domain requirement derived from sealed stage cells."""

        if isinstance(expected_members, (str, bytes)):
            raise TypeError("dataset stage requirement must be typed members")
        expected = tuple(sorted(expected_members, key=lambda row: row.member_id))
        if (
            not expected
            or any(type(row) is not AuthorizedDatasetContentMember for row in expected)
            or tuple(row.member_id for row in expected)
            != tuple(sorted({row.member_id for row in expected}))
        ):
            raise ValueError("dataset stage requirement is not canonical")
        observed = self.require_members(tuple(row.member_id for row in expected))
        if observed != expected:
            raise ValueError(
                "dataset authorization differs from the exact stage requirement"
            )
        return observed


def _stable_content_bytes(
    path_value: str,
    *,
    label: str,
    maximum_bytes: int,
) -> bytes:
    if type(path_value) is not str:
        raise TypeError(f"{label} path must be text")
    path = Path(path_value)
    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise ValueError(f"{label} path must be absolute and resolved")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} is not a readable regular file") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RuntimeError(f"{label} must be one regular, non-hardlinked file")
        if opened.st_size < 1 or opened.st_size > maximum_bytes:
            raise ValueError(f"{label} size is unsupported")
        body = b""
        while len(body) <= opened.st_size:
            chunk = os.read(
                descriptor, min(1024 * 1024, opened.st_size + 1 - len(body))
            )
            if not chunk:
                break
            body += chunk
        reopened = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)

        def identity(row: os.stat_result) -> tuple[int, ...]:
            return (
                row.st_dev,
                row.st_ino,
                row.st_mode,
                row.st_nlink,
                row.st_size,
                row.st_mtime_ns,
                row.st_ctime_ns,
            )

        if (
            len(body) != opened.st_size
            or identity(opened) != identity(reopened)
            or identity(reopened) != identity(current)
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


def _canonical_json_file(body: bytes, *, label: str) -> object:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    expected = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if body != expected:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _dataset_raw_row_count(body: bytes, *, data_format: str) -> int:
    if data_format == "canonical_json_array":
        value = _canonical_json_file(body, label="dataset raw JSON array")
        if type(value) is not list or not value:
            raise ValueError("dataset raw JSON array must contain rows")
        return len(value)
    if data_format == "canonical_jsonl":
        if not body.endswith(b"\n"):
            raise ValueError("dataset JSONL must end with one newline")
        lines = body.splitlines()
        if not lines:
            raise ValueError("dataset JSONL must contain rows")
        for line in lines:
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("dataset JSONL contains an invalid row") from error
            canonical = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            if line != canonical or type(value) is not dict:
                raise ValueError("dataset JSONL rows must be canonical objects")
        return len(lines)
    if data_format != "rfc4180_csv_utf8":
        raise ValueError("dataset raw format is unsupported")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("dataset CSV is not UTF-8") from error
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError("dataset CSV is invalid RFC4180") from error
    if len(rows) < 2 or not rows[0] or len(rows[0]) != len(set(rows[0])):
        raise ValueError("dataset CSV requires a unique header and data rows")
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError("dataset CSV rows differ from the header width")
    return len(rows) - 1


def _selected_request_rows(body: bytes) -> tuple[dict[str, Any], ...]:
    value = _canonical_json_file(body, label="selected request rows")
    if (
        type(value) is not list
        or not value
        or any(type(row) is not dict for row in value)
    ):
        raise ValueError("selected request rows must be a non-empty object array")
    request_ids = tuple(row.get("request_id") for row in value)
    if any(type(item) is not str or not item for item in request_ids):
        raise ValueError("selected request rows require non-empty request_id values")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("selected request rows contain duplicate request IDs")
    return tuple(value)


def _request_shape_manifest(
    body: bytes,
    *,
    selected_request_ids: tuple[str, ...],
) -> dict[str, Any]:
    value = _canonical_json_file(body, label="request shape manifest")
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "kind", "requests"}
        or value["schema_version"] != 1
        or value["kind"] != "lightcone_request_shape_manifest"
        or type(value["requests"]) is not list
        or not value["requests"]
        or any(type(row) is not dict for row in value["requests"])
    ):
        raise ValueError("request shape manifest schema is unsupported")
    shape_ids = tuple(row.get("request_id") for row in value["requests"])
    if shape_ids != selected_request_ids or len(shape_ids) != len(set(shape_ids)):
        raise ValueError("request shape rows differ from selected request IDs/order")
    return value


@dataclass(frozen=True)
class DatasetContentMemberPathBinding:
    member_id: str
    raw_path: str
    selected_rows_path: str
    request_shape_path: str

    def __post_init__(self) -> None:
        _require_text("dataset binding member ID", self.member_id)
        for label, value in (
            ("raw source", self.raw_path),
            ("selected rows", self.selected_rows_path),
            ("request shape", self.request_shape_path),
        ):
            path = Path(value)
            if not path.is_absolute() or path.resolve(strict=False) != path:
                raise ValueError(f"dataset {label} path must be absolute and resolved")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict_object(
                "dataset member path binding",
                value,
                {field.name for field in dataclass_fields(cls)},
            )
        )


@dataclass(frozen=True)
class DatasetContentPathBinding:
    schema_version: int
    kind: Literal["lightcone_dataset_content_path_binding"]
    authorization_sha256: str
    authority_domain: Literal["burstgpt_six_source", "e0_task_native"]
    members: tuple[DatasetContentMemberPathBinding, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_dataset_content_path_binding"
        ):
            raise ValueError("dataset content path binding schema is unsupported")
        _require_sha256("dataset path binding authorization", self.authorization_sha256)
        if self.authority_domain not in _DATASET_DOMAINS:
            raise ValueError("dataset path binding domain is unsupported")
        if (
            type(self.members) is not tuple
            or not self.members
            or any(
                type(row) is not DatasetContentMemberPathBinding for row in self.members
            )
            or tuple(row.member_id for row in self.members)
            != tuple(sorted({row.member_id for row in self.members}))
        ):
            raise ValueError("dataset path binding members are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "authorization_sha256": self.authorization_sha256,
            "authority_domain": self.authority_domain,
            "members": [row.to_dict() for row in self.members],
        }

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "dataset content path binding",
            value,
            {
                "schema_version",
                "kind",
                "authorization_sha256",
                "authority_domain",
                "members",
            },
        )
        raw_members = row.pop("members")
        if type(raw_members) is not list:
            raise TypeError("dataset path binding members must be an array")
        return cls(
            **row,
            members=tuple(
                DatasetContentMemberPathBinding.from_dict(item) for item in raw_members
            ),
        )


def revalidate_authorized_dataset_content_release(
    binding: DatasetContentPathBinding,
    *,
    authorization: VerifiedDatasetContentRelease,
) -> tuple[AuthorizedDatasetContentMember, ...]:
    """Deep-reopen exact raw, selected-row, and request-shape bytes."""

    if type(binding) is not DatasetContentPathBinding:
        raise TypeError("dataset replay requires an exact path binding")
    if type(authorization) is not VerifiedDatasetContentRelease:
        raise TypeError("dataset replay requires a verified authorization")
    binding.__post_init__()
    if (
        binding.authorization_sha256 != authorization.authorization_sha256
        or binding.authority_domain != authorization.authority_domain
    ):
        raise ValueError("dataset path binding differs from its authorization")
    authorized = authorization.authorization.members
    by_id = {row.member_id: row for row in authorized}
    if tuple(row.member_id for row in binding.members) != tuple(by_id):
        raise ValueError(
            "dataset path binding does not cover authorized members exactly"
        )
    for path_binding in binding.members:
        member = by_id[path_binding.member_id]
        raw = _stable_content_bytes(
            str(relocated_evidence_path(path_binding.raw_path)),
            label=f"dataset {member.member_id} raw source",
            maximum_bytes=_MAX_DATASET_SOURCE_BYTES,
        )
        selected = _stable_content_bytes(
            str(relocated_evidence_path(path_binding.selected_rows_path)),
            label=f"dataset {member.member_id} selected rows",
            maximum_bytes=_MAX_DATASET_DERIVED_BYTES,
        )
        shape = _stable_content_bytes(
            str(relocated_evidence_path(path_binding.request_shape_path)),
            label=f"dataset {member.member_id} request shape",
            maximum_bytes=_MAX_DATASET_DERIVED_BYTES,
        )
        selected_rows = _selected_request_rows(selected)
        selected_ids = tuple(row["request_id"] for row in selected_rows)
        shape_value = _request_shape_manifest(shape, selected_request_ids=selected_ids)
        if (
            hashlib.sha256(raw).hexdigest() != member.raw_file_sha256
            or len(raw) != member.raw_file_size
            or _dataset_raw_row_count(raw, data_format=member.data_format)
            != member.raw_row_count
            or hashlib.sha256(selected).hexdigest() != member.selected_rows_raw_sha256
            or _canonical_sha256(list(selected_rows)) != member.selected_rows_sha256
            or len(selected) != member.selected_rows_size
            or len(selected_rows) != member.selected_row_count
            or hashlib.sha256(shape).hexdigest() != member.request_shape_raw_sha256
            or _canonical_sha256(shape_value) != member.request_shape_sha256
            or len(shape) != member.request_shape_size
        ):
            raise ValueError(
                "dataset content bytes, counts, or request shape differ from authorization"
            )
    return authorized


def bind_authorized_dataset_content_release(
    *,
    authorization: VerifiedDatasetContentRelease,
    member_paths: Collection[DatasetContentMemberPathBinding],
) -> DatasetContentPathBinding:
    if type(authorization) is not VerifiedDatasetContentRelease:
        raise TypeError("dataset binding requires a verified authorization")
    if isinstance(member_paths, (str, bytes)):
        raise TypeError("dataset binding paths must be typed members")
    binding = DatasetContentPathBinding(
        schema_version=1,
        kind="lightcone_dataset_content_path_binding",
        authorization_sha256=authorization.authorization_sha256,
        authority_domain=authorization.authority_domain,
        members=tuple(sorted(member_paths, key=lambda row: row.member_id)),
    )
    revalidate_authorized_dataset_content_release(binding, authorization=authorization)
    return binding


def verify_dataset_content_release_authorization(
    authorization: DatasetContentReleaseAuthorization,
    *,
    expected_authority_domain: str,
    now_ns: int,
    consumed_challenge_sha256s: Collection[str] = (),
) -> VerifiedDatasetContentRelease:
    if type(authorization) is not DatasetContentReleaseAuthorization:
        raise TypeError("dataset content verification requires an exact authorization")
    if authorization.authority_domain != expected_authority_domain:
        raise ValueError("dataset content authorization uses another domain")
    authorization.__post_init__()
    _validate_authorization_challenge(
        authorization.challenge,
        subject_sha256=authorization.subject_sha256,
        now_ns=now_ns,
        consumed_challenge_sha256s=consumed_challenge_sha256s,
    )
    root_binding = _verify_root_signature(
        root_manifest_sha256=authorization.root_manifest_sha256,
        challenge=authorization.challenge,
        payload_sha256=authorization.subject_sha256,
        signature_base64=authorization.signature_base64,
    )
    return VerifiedDatasetContentRelease(
        authorization, root_binding, _seal=_VERIFICATION_SEAL
    )


def _reverify_content_authorization_artifact(
    artifact: ContentJsonArtifactBinding,
    *,
    verified_ns: int,
) -> tuple[str, str, object]:
    value = artifact.load()
    if artifact.artifact_id == "workload:e3a":
        authorization = ReleaseWorkloadSourceAuthorization.from_dict(value)
        verified = verify_release_workload_source_authorization(
            authorization, now_ns=verified_ns
        )
    elif artifact.artifact_id == "prepared:formal_dag":
        authorization = PreparedModelContentReleaseAuthorization.from_dict(value)
        verified = verify_prepared_model_content_release_authorization(
            authorization, now_ns=verified_ns
        )
    elif artifact.artifact_id in {
        "dataset:burstgpt_six_source",
        "dataset:e0_task_native",
    }:
        domain = artifact.artifact_id.split(":", 1)[1]
        authorization = DatasetContentReleaseAuthorization.from_dict(value)
        verified = verify_dataset_content_release_authorization(
            authorization,
            expected_authority_domain=domain,
            now_ns=verified_ns,
        )
    else:
        raise ValueError("content verification receipt has an unknown authorization")
    if (
        type(value) is not dict
        or value.get("authorization_sha256") != verified.authorization_sha256
    ):
        raise ValueError("content authorization artifact wrapper SHA differs")
    return verified.challenge_sha256, verified.root_binding_sha256, verified


def _validate_master_content_use_universe(
    verified_rows: tuple[object, ...],
    content: tuple[ContentJsonArtifactBinding, ...],
) -> None:
    """Require every immutable master member to have one registered consumer."""

    prepared_rows = tuple(
        row for row in verified_rows if type(row) is VerifiedPreparedModelContentRelease
    )
    dataset_rows = tuple(
        row for row in verified_rows if type(row) is VerifiedDatasetContentRelease
    )
    workload_rows = tuple(
        row for row in verified_rows if type(row) is VerifiedReleaseWorkloadSources
    )
    if (
        len(prepared_rows) != 1
        or len(dataset_rows) != 2
        or len(workload_rows) != 1
        or {row.authority_domain for row in dataset_rows}
        != {"burstgpt_six_source", "e0_task_native"}
    ):
        raise ValueError("master content authority universe is not exact")
    prepared = prepared_rows[0]
    datasets = {row.authority_domain: row for row in dataset_rows}
    models = {row.member_id: row for row in prepared.authorization.models}
    allowed_ids = {
        "dataset:burstgpt_six_source:path_binding",
        "dataset:e0_task_native:path_binding",
        "tts_calibration_tuning_window",
        *(f"snapshot:{member_id}" for member_id in models),
        *(
            f"eagle3_official_selector:{member_id}"
            for member_id, model in models.items()
            if model.backend == "EAGLE3" and model.role == "drafter"
        ),
    }
    observed_ids = {row.artifact_id for row in content}
    required_ids = {
        "dataset:burstgpt_six_source:path_binding",
        "dataset:e0_task_native:path_binding",
        "tts_calibration_tuning_window",
    }
    if not required_ids <= observed_ids or not observed_ids <= allowed_ids:
        raise ValueError("master content contains an unknown or unused artifact")

    by_id = {row.artifact_id: row for row in content}
    workload = workload_rows[0]
    tuning_window = TtsCalibrationTuningWindow.from_dict(
        by_id["tts_calibration_tuning_window"].load()
    )
    if any(
        entry.source_descriptor_sha256 != workload.source(entry.workload_id).sha256
        for entry in tuning_window.entries
    ):
        raise ValueError(
            "TTS calibration window differs from root-authorized workload sources"
        )
    for domain, verified in datasets.items():
        artifact_id = f"dataset:{domain}:path_binding"
        path_binding = DatasetContentPathBinding.from_dict(by_id[artifact_id].load())
        revalidate_authorized_dataset_content_release(
            path_binding,
            authorization=verified,
        )

    snapshot_bindings = tuple(
        row for row in content if row.artifact_id.startswith("snapshot:")
    )
    for binding in snapshot_bindings:
        member_id = binding.artifact_id.removeprefix("snapshot:")
        member = models[member_id]
        if (
            binding.raw_sha256 != member.snapshot_manifest_raw_sha256
            or binding.semantic_sha256 != member.snapshot_manifest_semantic_sha256
        ):
            raise ValueError("master snapshot differs from prepared-model authority")
    for member in models.values():
        if not any(
            binding.raw_sha256 == member.snapshot_manifest_raw_sha256
            and binding.semantic_sha256 == member.snapshot_manifest_semantic_sha256
            for binding in snapshot_bindings
        ):
            raise ValueError("master content does not cover every prepared model")

    selector_ids = tuple(
        row.artifact_id
        for row in content
        if row.artifact_id.startswith("eagle3_official_selector:")
    )
    expected_selectors = tuple(
        sorted(
            f"eagle3_official_selector:{member_id}"
            for member_id, model in models.items()
            if model.backend == "EAGLE3" and model.role == "drafter"
        )
    )
    if selector_ids and tuple(sorted(selector_ids)) != expected_selectors:
        raise ValueError("master EAGLE3 selector use coverage differs")


@dataclass(frozen=True)
class ContentVerificationReceipt:
    """Durable proof that short-lived content controls were accepted once.

    The original root signatures are rechecked at the recorded acceptance time,
    never at an invented later time and never with expiry ignored.  The exact
    challenges must coexist in the same immutable reservation as the dispatch
    control.  Later stages additionally reopen every path-bound content artifact.
    """

    schema_version: int
    kind: Literal["lightcone_content_verification_receipt"]
    verified_ns: int
    root_binding_sha256: str
    authorization_artifacts: tuple[ContentJsonArtifactBinding, ...]
    content_artifacts: tuple[ContentJsonArtifactBinding, ...]
    authorization_challenge_sha256s: tuple[str, ...]
    reservation: ChallengeReplayReservationBinding

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_content_verification_receipt"
        ):
            raise ValueError("content verification receipt schema is unsupported")
        _positive_int("content verification time", self.verified_ns)
        _require_sha256("content verification root binding", self.root_binding_sha256)
        for label, rows in (
            ("authorization", self.authorization_artifacts),
            ("content", self.content_artifacts),
        ):
            if (
                type(rows) is not tuple
                or not rows
                or any(type(row) is not ContentJsonArtifactBinding for row in rows)
                or tuple(row.artifact_id for row in rows)
                != tuple(sorted({row.artifact_id for row in rows}))
            ):
                raise ValueError(
                    f"content verification {label} artifacts are not canonical"
                )
        _reject_post_master_derived_content(self.content_artifacts)
        if (
            type(self.authorization_challenge_sha256s) is not tuple
            or not self.authorization_challenge_sha256s
            or self.authorization_challenge_sha256s
            != tuple(sorted(set(self.authorization_challenge_sha256s)))
        ):
            raise ValueError("content verification challenges are not canonical")
        for digest in self.authorization_challenge_sha256s:
            _require_sha256("content verification challenge", digest)
        if type(self.reservation) is not ChallengeReplayReservationBinding:
            raise TypeError(
                "content verification receipt requires a replay reservation"
            )
        if self.verified_ns != self.reservation.reserved_ns:
            raise ValueError(
                "content verification time differs from atomic reservation time"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "verified_ns": self.verified_ns,
            "root_binding_sha256": self.root_binding_sha256,
            "authorization_artifacts": [
                row.to_dict() for row in self.authorization_artifacts
            ],
            "content_artifacts": [row.to_dict() for row in self.content_artifacts],
            "authorization_challenge_sha256s": list(
                self.authorization_challenge_sha256s
            ),
            "reservation": self.reservation.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "content verification receipt",
            value,
            {
                "schema_version",
                "kind",
                "verified_ns",
                "root_binding_sha256",
                "authorization_artifacts",
                "content_artifacts",
                "authorization_challenge_sha256s",
                "reservation",
            },
        )
        raw_authorizations = row.pop("authorization_artifacts")
        raw_content = row.pop("content_artifacts")
        raw_challenges = row.pop("authorization_challenge_sha256s")
        if type(raw_authorizations) is not list or type(raw_content) is not list:
            raise TypeError("content verification artifacts must be arrays")
        if type(raw_challenges) is not list:
            raise TypeError("content verification challenges must be an array")
        reservation = ChallengeReplayReservationBinding.from_dict(
            row.pop("reservation")
        )
        return cls(
            **row,
            authorization_artifacts=tuple(
                ContentJsonArtifactBinding.from_dict(item)
                for item in raw_authorizations
            ),
            content_artifacts=tuple(
                ContentJsonArtifactBinding.from_dict(item) for item in raw_content
            ),
            authorization_challenge_sha256s=tuple(raw_challenges),
            reservation=reservation,
        )

    def revalidate(self, *, current_ns: int) -> tuple[object, ...]:
        self.__post_init__()
        if type(current_ns) is not int or current_ns < self.verified_ns:
            raise ValueError("content verification replay time precedes acceptance")
        reserved = self.reservation.revalidate()
        if self.verified_ns != self.reservation.reserved_ns:
            raise ValueError(
                "content verification time differs from atomic reservation time"
            )
        verified_rows = tuple(
            _reverify_content_authorization_artifact(
                artifact, verified_ns=self.verified_ns
            )
            for artifact in self.authorization_artifacts
        )
        challenges = tuple(sorted(row[0] for row in verified_rows))
        roots = {row[1] for row in verified_rows}
        if (
            challenges != self.authorization_challenge_sha256s
            or not set(challenges) <= set(reserved)
            or roots != {self.root_binding_sha256}
        ):
            raise ValueError(
                "content verification receipt differs from root or replay reservation"
            )
        if (
            tuple(row.artifact_id for row in self.authorization_artifacts)
            == _MASTER_AUTHORIZATION_IDS
        ):
            _validate_master_content_use_universe(
                tuple(row[2] for row in verified_rows),
                self.content_artifacts,
            )
        for artifact in self.content_artifacts:
            artifact.load()
        return tuple(row[2] for row in verified_rows)

    def revalidate_formal_scope(self, *, current_ns: int) -> tuple[object, ...]:
        """Require either the atomic master or a child derived from that master."""

        verified_rows = self.revalidate(current_ns=current_ns)
        authorization_ids = tuple(
            row.artifact_id for row in self.authorization_artifacts
        )
        if authorization_ids == _MASTER_AUTHORIZATION_IDS:
            if self.reservation.challenge_sha256s != (
                self.authorization_challenge_sha256s
            ):
                raise ValueError("content verification master reservation is not exact")
            return verified_rows
        allowed_scopes = {
            ("prepared:formal_dag",),
            ("dataset:e0_task_native", "prepared:formal_dag"),
            ("prepared:formal_dag", "workload:e3a"),
        }
        if authorization_ids not in allowed_scopes:
            raise ValueError("content verification child scope is unsupported")
        master_artifacts = tuple(
            row
            for row in self.content_artifacts
            if row.artifact_id == _MASTER_CONTENT_ARTIFACT_ID
        )
        if len(master_artifacts) != 1:
            raise ValueError("content verification child lacks one master receipt")
        master = ContentVerificationReceipt.from_dict(master_artifacts[0].load())
        if (
            tuple(row.artifact_id for row in master.authorization_artifacts)
            != _MASTER_AUTHORIZATION_IDS
        ):
            raise ValueError("content verification child master is not complete")
        master.revalidate(current_ns=current_ns)
        master_authorizations = {
            row.artifact_id: row for row in master.authorization_artifacts
        }
        master_content = {row.artifact_id: row for row in master.content_artifacts}
        child_content = tuple(
            row
            for row in self.content_artifacts
            if row.artifact_id != _MASTER_CONTENT_ARTIFACT_ID
        )
        if (
            self.verified_ns != master.verified_ns
            or self.root_binding_sha256 != master.root_binding_sha256
            or self.reservation != master.reservation
            or any(
                master_authorizations.get(row.artifact_id) != row
                for row in self.authorization_artifacts
            )
            or any(master_content.get(row.artifact_id) != row for row in child_content)
        ):
            raise ValueError("content verification child differs from its master")
        return verified_rows


def build_content_verification_receipt(
    *,
    verified_ns: int,
    authorization_artifacts: Collection[ContentJsonArtifactBinding],
    content_artifacts: Collection[ContentJsonArtifactBinding],
    reservation: ChallengeReplayReservationBinding,
) -> ContentVerificationReceipt:
    """Build only after the final control transaction reserved all challenges."""

    if isinstance(authorization_artifacts, (str, bytes)) or isinstance(
        content_artifacts, (str, bytes)
    ):
        raise TypeError("content verification artifacts must be typed collections")
    authorizations = tuple(
        sorted(authorization_artifacts, key=lambda row: row.artifact_id)
    )
    content = tuple(sorted(content_artifacts, key=lambda row: row.artifact_id))
    _reject_post_master_derived_content(content)
    verified_rows = tuple(
        _reverify_content_authorization_artifact(row, verified_ns=verified_ns)
        for row in authorizations
    )
    if tuple(row.artifact_id for row in authorizations) == _MASTER_AUTHORIZATION_IDS:
        _validate_master_content_use_universe(
            tuple(row[2] for row in verified_rows),
            content,
        )
    roots = {row[1] for row in verified_rows}
    if len(roots) != 1:
        raise ValueError("content authorizations do not share one source root")
    receipt = ContentVerificationReceipt(
        schema_version=1,
        kind="lightcone_content_verification_receipt",
        verified_ns=verified_ns,
        root_binding_sha256=next(iter(roots)),
        authorization_artifacts=authorizations,
        content_artifacts=content,
        authorization_challenge_sha256s=tuple(sorted(row[0] for row in verified_rows)),
        reservation=reservation,
    )
    receipt.revalidate(current_ns=verified_ns)
    return receipt


def verify_and_reserve_content_authorizations(
    *,
    verified_ns: int,
    authorization_artifacts: Collection[ContentJsonArtifactBinding],
    content_artifacts: Collection[ContentJsonArtifactBinding],
    replay_store: ChallengeReplayStore,
) -> ContentVerificationReceipt:
    """Verify the exact formal content set and commit its challenges once."""

    if type(replay_store) is not ChallengeReplayStore:
        raise TypeError("content verification requires an exact replay store")
    if isinstance(authorization_artifacts, (str, bytes)) or isinstance(
        content_artifacts, (str, bytes)
    ):
        raise TypeError("content verification artifacts must be typed collections")
    authorizations = tuple(
        sorted(authorization_artifacts, key=lambda row: row.artifact_id)
    )
    if tuple(row.artifact_id for row in authorizations) != _MASTER_AUTHORIZATION_IDS:
        raise ValueError(
            "formal content verification requires the exact four authorities"
        )
    content = tuple(sorted(content_artifacts, key=lambda row: row.artifact_id))
    if not content:
        raise ValueError("formal content verification requires path-bound content")
    _reject_post_master_derived_content(content)
    if any(row.artifact_id == _MASTER_CONTENT_ARTIFACT_ID for row in content):
        raise ValueError("master content artifact ID is reserved for derived receipts")
    verified_rows = tuple(
        _reverify_content_authorization_artifact(row, verified_ns=verified_ns)
        for row in authorizations
    )
    _validate_master_content_use_universe(
        tuple(row[2] for row in verified_rows),
        content,
    )
    challenge_sha256s = tuple(sorted(row[0] for row in verified_rows))
    if len(challenge_sha256s) != len(set(challenge_sha256s)):
        raise ValueError("formal content authorizations reuse one challenge")
    if len({row[1] for row in verified_rows}) != 1:
        raise ValueError("formal content authorizations do not share one source root")
    reservation = replay_store.reserve_verified_content_challenges(
        challenge_sha256s,
        reserved_ns=verified_ns,
    )
    return build_content_verification_receipt(
        verified_ns=verified_ns,
        authorization_artifacts=authorizations,
        content_artifacts=content,
        reservation=reservation,
    )


def derive_stage_content_verification_receipt(
    master: ContentVerificationReceipt,
    *,
    master_artifact: ContentJsonArtifactBinding,
    stage: str,
    current_ns: int,
) -> ContentVerificationReceipt:
    """Create a stage-minimal view without reserving any challenge again."""

    if type(master) is not ContentVerificationReceipt:
        raise TypeError("content scope derivation requires an exact master receipt")
    if type(master_artifact) is not ContentJsonArtifactBinding:
        raise TypeError("content scope derivation requires a bound master artifact")
    if master_artifact.artifact_id != _MASTER_CONTENT_ARTIFACT_ID:
        raise ValueError("content scope master artifact has another ID")
    if ContentVerificationReceipt.from_dict(master_artifact.load()) != master:
        raise ValueError("content scope master binding differs from receipt")
    master.revalidate_formal_scope(current_ns=current_ns)
    if stage == "TTS-Cal":
        required = ("prepared:formal_dag", "workload:e3a")
    elif stage == "E0":
        required = ("dataset:e0_task_native", "prepared:formal_dag")
    elif stage in {"E3a", "E1", "E2", "E4", "E3b", "E1a", "E5", "E6"}:
        required = ("prepared:formal_dag", "workload:e3a")
    else:
        raise ValueError("content scope stage is unsupported")
    by_id = {row.artifact_id: row for row in master.authorization_artifacts}
    scoped_authorizations = tuple(by_id[artifact_id] for artifact_id in required)
    challenges = tuple(
        sorted(
            _reverify_content_authorization_artifact(
                row,
                verified_ns=master.verified_ns,
            )[0]
            for row in scoped_authorizations
        )
    )
    child = ContentVerificationReceipt(
        schema_version=1,
        kind="lightcone_content_verification_receipt",
        verified_ns=master.verified_ns,
        root_binding_sha256=master.root_binding_sha256,
        authorization_artifacts=scoped_authorizations,
        content_artifacts=tuple(
            sorted(
                (*master.content_artifacts, master_artifact),
                key=lambda row: row.artifact_id,
            )
        ),
        authorization_challenge_sha256s=challenges,
        reservation=master.reservation,
    )
    child.revalidate_formal_scope(current_ns=current_ns)
    return child


__all__ = [
    "CONTENT_AUTHORIZATION_SOURCE_TYPES",
    "AuthorizedDatasetContentMember",
    "AuthorizedPreparedModel",
    "AuthorizedWorkloadSource",
    "ContentAuthorizationSource",
    "ContentJsonArtifactBinding",
    "ContentVerificationReceipt",
    "DatasetContentMemberPathBinding",
    "DatasetContentPathBinding",
    "DatasetContentReleaseAuthorization",
    "DatasetContentReleaseAuthorizationSource",
    "PreparedModelContentReleaseAuthorization",
    "PreparedModelContentReleaseAuthorizationSource",
    "PreparedModelStageMembership",
    "ReleaseWorkloadSourceAuthorization",
    "ReleaseWorkloadSourceAuthorizationSource",
    "TtsCalibrationTuningWindow",
    "TtsCalibrationTuningWindowEntry",
    "VerifiedDatasetContentRelease",
    "VerifiedPreparedModelContentRelease",
    "VerifiedReleaseWorkloadSources",
    "bind_authorized_dataset_content_release",
    "build_content_authorization_from_source",
    "build_content_verification_receipt",
    "content_authorization_source_from_dict",
    "derive_stage_content_verification_receipt",
    "revalidate_authorized_dataset_content_release",
    "verify_and_reserve_content_authorizations",
    "verify_dataset_content_release_authorization",
    "verify_prepared_model_content_release_authorization",
    "verify_release_workload_source_authorization",
]
