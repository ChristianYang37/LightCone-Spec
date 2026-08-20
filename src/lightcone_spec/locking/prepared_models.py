"""Strict bindings for revision-addressed local model snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

from lightcone_spec.runtime.content_authorization import (
    VerifiedPreparedModelContentRelease,
)
from lightcone_spec.runtime.proof_artifact import relocated_evidence_path

from .models import ModelLock


def _canonical_sha256(value: object) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


PREPARED_MODEL_BINDING_PROTOCOL_SHA256 = _canonical_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_prepared_model_binding_protocol",
        "requirements": [
            "exact_schema_v2_model_lock",
            "model_id_sorted_unique_complete_coverage",
            "absolute_resolved_regular_snapshot_directory",
            "snapshot_parent_component_is_snapshots",
            "snapshot_leaf_equals_locked_revision",
        ],
    }
)

_CONTENT_PROFILE_PAYLOAD = {
    "Qwen/Qwen3-8B": {
        "profile": "qwen3_target_sharded_safetensors_v1",
        "critical_files": [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
        "weight_kind": "sharded_safetensors",
    },
    "z-lab/Qwen3-8B-DFlash-b16": {
        "profile": "qwen3_dflash_single_safetensors_v1",
        "critical_files": [
            "config.json",
            "dflash.py",
            "modeling_dflash.py",
            "utils.py",
        ],
        "weight_kind": "single_safetensors",
        "tokenizer_source": "Qwen/Qwen3-8B",
    },
}

_GENERIC_CONTENT_PROFILE_PAYLOAD = {
    "profile": "generic_complete_lightweight_safetensors_v1",
    "discovery": (
        "all_regular_non_safetensors_files_recursive_plus_exact_safetensors_"
        "index_or_single_model"
    ),
    "filesystem": (
        "regular_files_or_single_hop_repo_local_hf_blob_links_bounded_stable_reopen"
    ),
    "weight_kind": "derived_exactly_from_model_safetensors_layout",
    "weight_identity": "complete_payload_sha256_plus_header_tensor_metadata",
}

PREPARED_MODEL_CONTENT_PROTOCOL_SHA256 = _canonical_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_prepared_model_content_protocol",
        "release_profiles": _CONTENT_PROFILE_PAYLOAD,
        "generic_profile": _GENERIC_CONTENT_PROFILE_PAYLOAD,
        "filesystem": (
            "absolute_resolved_revision_root_no_symlink_no_hardlink_"
            "nofollow_stable_stat"
        ),
        "manifest": "path_bound_strict_json_sidecar_and_external_release_sha256",
        "scope": (
            "all_registered_or_generic_lightweight_files_plus_complete_"
            "safetensors_payload_sha256_header_and_tensor_metadata"
        ),
    }
)

PREPARED_MODEL_CONTENT_TRUSTED_REPLAY_PROTOCOL_SHA256 = _canonical_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_prepared_model_content_protocol",
        "legacy_protocol_sha256": PREPARED_MODEL_CONTENT_PROTOCOL_SHA256,
        "release_profiles": _CONTENT_PROFILE_PAYLOAD,
        "generic_profile": _GENERIC_CONTENT_PROFILE_PAYLOAD,
        "trusted_content": (
            "runtime_bound_bundle_schema2_exact_logical_member_and_physical_"
            "content_replay_closure"
        ),
        "filesystem": (
            "complete_metadata_replay_before_and_after_bounded_prepared_reads"
        ),
        "lightweight_files": "bounded_bytes_sha256_equals_exact_member_file_row",
        "weights": (
            "exact_member_file_sha256_projection_plus_bounded_safetensors_"
            "header_and_complete_tensor_inventory_without_payload_rehash"
        ),
        "failure": "metadata_or_bundle_or_member_or_closure_drift_blocks",
    }
)

# Formal release pins are source-owned and reviewable.  A raw bundle or local
# snapshot may mirror one of these digests but can never add an entry.  The
# current release deliberately ships no audited prepared-snapshot manifest.
PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_SHA256S: Mapping[str, str] = MappingProxyType(
    {}
)

_SHA256_LENGTH = 64
_MAX_CRITICAL_FILE_BYTES = 64 * 1024 * 1024
_MAX_CONTENT_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_MAX_GENERIC_SNAPSHOT_FILES = 16_384
_BANNED_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
_CONTENT_FILE_FIELDS = frozenset({"relative_path", "size", "raw_sha256"})
_TENSOR_FIELDS = frozenset({"name", "shape", "dtype", "data_start", "data_end"})
_HEADER_FIELDS = frozenset(
    {
        "relative_path",
        "file_size",
        "device",
        "inode",
        "mtime_ns",
        "ctime_ns",
        "header_size",
        "header_sha256",
        "raw_sha256",
        "tensors",
    }
)
_SNAPSHOT_CONTENT_FIELDS = frozenset(
    {
        "model_id",
        "revision",
        "root",
        "profile",
        "critical_files",
        "weight_kind",
        "weight_headers",
        "tensor_metadata_sha256",
    }
)
_TRUSTED_CONTENT_FILE_FIELDS = frozenset(
    {
        "relative_path",
        "size",
        "sha256",
        "storage_kind",
        "symlink_target",
        "resolved_relative_path",
    }
)
_TRUSTED_CONTENT_MEMBER_FIELDS = frozenset(
    {
        "model_id",
        "revision",
        "role",
        "root",
        "member_sha256",
        "tree_sha256",
        "content_sha256",
        "storage_mode",
        "content_cache_root",
        "files",
    }
)
_TRUSTED_CONTENT_REPLAY_FIELDS = frozenset(
    {
        "content_bundle_path",
        "content_bundle_size",
        "content_bundle_raw_sha256",
        "content_bundle_semantic_sha256",
        "content_bundle_runtime_binding_status",
        "content_replay_path",
        "content_replay_size",
        "content_replay_raw_sha256",
        "content_replay_semantic_sha256",
        "content_replay_protocol_sha256",
    }
)
_SNAPSHOT_CONTENT_FIELDS_V2 = _SNAPSHOT_CONTENT_FIELDS | frozenset(
    {"trusted_content_member"}
)
_CONTENT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol_sha256",
        "model_lock_sha256",
        "prepared_model_set_sha256",
        "snapshots",
    }
)
_CONTENT_MANIFEST_FIELDS_V2 = _CONTENT_MANIFEST_FIELDS | frozenset(
    {"trusted_content_replay"}
)
_RAW_MANIFEST_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "path",
        "sidecar_path",
        "semantic_sha256",
        "file_sha256",
        "sidecar_file_sha256",
        "size",
        "sidecar_size",
    }
)
_CONTENT_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol_sha256",
        "release_manifest_sha256",
        "model_lock_sha256",
        "prepared_model_set",
        "manifest",
    }
)
_SAFETENSORS_DTYPES = {
    "BOOL": ("torch.bool", 1),
    "U8": ("torch.uint8", 1),
    "I8": ("torch.int8", 1),
    "I16": ("torch.int16", 2),
    "I32": ("torch.int32", 4),
    "I64": ("torch.int64", 8),
    "F16": ("torch.float16", 2),
    "BF16": ("torch.bfloat16", 2),
    "F32": ("torch.float32", 4),
    "F64": ("torch.float64", 8),
    "C64": ("torch.complex64", 8),
    "C128": ("torch.complex128", 16),
}


@dataclass(frozen=True)
class _ContentProfile:
    name: str
    critical_files: tuple[str, ...]
    weight_kind: Literal["sharded_safetensors", "single_safetensors"]
    tokenizer_source: str | None = None


_CONTENT_PROFILES = {
    model_id: _ContentProfile(
        name=str(row["profile"]),
        critical_files=tuple(str(item) for item in row["critical_files"]),
        weight_kind=row["weight_kind"],
        tokenizer_source=(
            None
            if row.get("tokenizer_source") is None
            else str(row["tokenizer_source"])
        ),
    )
    for model_id, row in _CONTENT_PROFILE_PAYLOAD.items()
}


class PreparedModelContentAuthorityBlocked(RuntimeError):
    """A named fail-closed outcome for an unavailable release-owned profile."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def prepared_model_content_release_identity_sha256(
    *,
    model_lock_sha256: str,
    prepared: PreparedModelSet,
) -> str:
    """Return the path-independent key for one exact locked revision set."""

    _require_sha256("prepared content release model lock", model_lock_sha256)
    if type(prepared) is not PreparedModelSet:
        raise TypeError("prepared content release identity requires PreparedModelSet")
    prepared.validate()
    if prepared.model_lock_sha256 != model_lock_sha256:
        raise ValueError("prepared content release identity differs from model lock")
    return _canonical_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_prepared_model_content_release_identity",
            "protocol_sha256": PREPARED_MODEL_CONTENT_PROTOCOL_SHA256,
            "model_lock_sha256": model_lock_sha256,
            "models": [
                {
                    "model_id": snapshot.model_id,
                    "revision": snapshot.revision,
                }
                for snapshot in prepared.snapshots
            ],
        }
    )


def require_prepared_model_content_release_manifest_sha256(
    *,
    model_lock_sha256: str,
    prepared: PreparedModelSet,
    claimed_manifest_sha256: str,
) -> str:
    """Resolve one audited source-owned pin; caller mirrors grant no trust."""

    claimed = _require_sha256(
        "claimed prepared content release manifest", claimed_manifest_sha256
    )
    identity = prepared_model_content_release_identity_sha256(
        model_lock_sha256=model_lock_sha256,
        prepared=prepared,
    )
    expected = PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_SHA256S.get(identity)
    if expected is None or claimed != expected:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_release_manifest_pin_unavailable",
            "no matching audited source-owned manifest pin exists for the locked revisions",
        )
    return expected


def has_prepared_model_content_release_manifest_sha256(
    *,
    model_lock_sha256: str,
    prepared: PreparedModelSet,
    claimed_manifest_sha256: str,
) -> bool:
    """Return whether a claimed digest equals one source-owned release pin."""

    claimed = _require_sha256(
        "claimed prepared content release manifest", claimed_manifest_sha256
    )
    identity = prepared_model_content_release_identity_sha256(
        model_lock_sha256=model_lock_sha256,
        prepared=prepared,
    )
    return PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_SHA256S.get(identity) == claimed


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(name: str, value: object) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be lower-case SHA-256")
    return value


def _require_text(name: str, value: object) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or "\n" in value
        or "\r" in value
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be non-empty single-line text")
    return value


def _strict_object(name: str, value: object, fields: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    if set(value) != fields:
        missing = sorted(fields - set(value))
        unknown = sorted(set(value) - fields)
        raise ValueError(f"{name} fields differ: missing={missing}, unknown={unknown}")
    return value


def _strict_list(name: str, value: object) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number is forbidden")
        return
    if type(value) is str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("unpaired JSON surrogate is forbidden")
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            _validate_json_value(key)
            _validate_json_value(item)
        return
    raise TypeError(f"unsupported strict JSON value {type(value).__name__}")


def _strict_json(body: bytes, *, label: str) -> object:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    _validate_json_value(value)
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _safe_relative_path(value: object, *, label: str) -> str:
    text = _require_text(label, value)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != text
    ):
        raise ValueError(f"{label} must be one canonical snapshot-relative path")
    return text


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


@dataclass(frozen=True)
class _OpenedSnapshotSource:
    entry_stat: os.stat_result
    target_path: str
    target_stat: os.stat_result
    link_text: str | None
    blob_name: str | None
    blobs_root_path: str | None
    blobs_root_stat: os.stat_result | None


def _open_snapshot_file(
    root: Path,
    relative_path: str,
    *,
    label: str,
) -> tuple[int, os.stat_result, _OpenedSnapshotSource]:
    relative = _safe_relative_path(relative_path, label=f"{label} relative path")
    root_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        root_flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        root_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        root_flags |= os.O_NOFOLLOW
    try:
        root_descriptor = os.open(root, root_flags)
    except OSError as error:
        raise ValueError(f"{label} snapshot root is unsafe") from error
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        entry = os.stat(relative, dir_fd=root_descriptor, follow_symlinks=False)
        link_text: str | None = None
        blob_name: str | None = None
        if stat.S_ISREG(entry.st_mode) and entry.st_nlink == 1:
            target_path = root / relative
            descriptor = os.open(relative, flags, dir_fd=root_descriptor)
            blobs_root_path: str | None = None
            blobs_root_stat: os.stat_result | None = None
        elif stat.S_ISLNK(entry.st_mode):
            link_text = os.readlink(relative, dir_fd=root_descriptor)
            link = Path(link_text)
            if link.is_absolute() or "\x00" in link_text:
                raise ValueError(f"{label} file {relative!r} has an unsafe cache link")
            repository_root = root.parent.parent
            blobs_root = repository_root / "blobs"
            target_path = Path(os.path.abspath((root / relative).parent / link))
            if (
                not repository_root.name.startswith("models--")
                or target_path.parent != blobs_root
                or len(target_path.name) not in {40, 64}
                or any(
                    character not in "0123456789abcdef"
                    for character in target_path.name
                )
            ):
                raise ValueError(
                    f"{label} file {relative!r} escapes the canonical HF blobs root"
                )
            expected_link = Path(
                os.path.relpath(target_path, start=(root / relative).parent)
            ).as_posix()
            if link_text != expected_link:
                raise ValueError(
                    f"{label} file {relative!r} has a non-canonical HF blob link"
                )
            blobs_root_stat = os.lstat(blobs_root)
            if (
                not stat.S_ISDIR(blobs_root_stat.st_mode)
                or stat.S_ISLNK(blobs_root_stat.st_mode)
                or blobs_root_stat.st_uid != os.geteuid()
                or stat.S_IMODE(blobs_root_stat.st_mode) & 0o022
            ):
                raise ValueError(
                    f"{label} file {relative!r} uses an unsafe HF blobs directory"
                )
            blobs_root_path = str(blobs_root)
            target_lstat = os.lstat(target_path)
            if not stat.S_ISREG(target_lstat.st_mode) or target_lstat.st_nlink != 1:
                raise ValueError(
                    f"{label} file {relative!r} has a chained or hardlinked blob"
                )
            descriptor = os.open(target_path, flags)
            blob_name = target_path.name
        else:
            raise ValueError(
                f"{label} file {relative!r} is a hardlink or unsupported entry"
            )
        opened = os.fstat(descriptor)
    except ValueError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_required_files_unavailable",
            f"{label} file {relative!r} is absent or unreadable",
        ) from error
    finally:
        os.close(root_descriptor)
    if descriptor is None:
        raise RuntimeError(f"{label} file descriptor was not opened")
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or _stat_identity(opened) != _stat_identity(os.lstat(target_path))
    ):
        os.close(descriptor)
        raise ValueError(f"{label} file {relative!r} target changed while opening")
    return (
        descriptor,
        opened,
        _OpenedSnapshotSource(
            entry_stat=entry,
            target_path=str(target_path),
            target_stat=opened,
            link_text=link_text,
            blob_name=blob_name,
            blobs_root_path=blobs_root_path,
            blobs_root_stat=blobs_root_stat,
        ),
    )


def _finish_stable_read(
    descriptor: int,
    opened: os.stat_result,
    *,
    root: Path,
    relative_path: str,
    source: _OpenedSnapshotSource,
    expected_bytes: int | None,
    label: str,
) -> None:
    reopened = os.fstat(descriptor)
    root_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        root_flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        root_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        root_flags |= os.O_NOFOLLOW
    root_descriptor = os.open(root, root_flags)
    try:
        entry = os.stat(relative_path, dir_fd=root_descriptor, follow_symlinks=False)
        current_link = (
            None
            if source.link_text is None
            else os.readlink(relative_path, dir_fd=root_descriptor)
        )
    finally:
        os.close(root_descriptor)
    current_target = os.lstat(source.target_path)
    current_blobs_root = (
        None if source.blobs_root_path is None else os.lstat(source.blobs_root_path)
    )
    if (
        _stat_identity(opened) != _stat_identity(reopened)
        or _stat_identity(reopened) != _stat_identity(current_target)
        or _stat_identity(source.entry_stat) != _stat_identity(entry)
        or source.link_text != current_link
        or (
            source.blobs_root_stat is not None
            and (
                current_blobs_root is None
                or _stat_identity(source.blobs_root_stat)
                != _stat_identity(current_blobs_root)
            )
        )
        or (expected_bytes is not None and reopened.st_size != expected_bytes)
    ):
        raise RuntimeError(f"{label} changed while it was read")


def _read_snapshot_file(
    root: Path,
    relative_path: str,
    *,
    label: str,
    maximum_bytes: int = _MAX_CRITICAL_FILE_BYTES,
) -> tuple[bytes, os.stat_result]:
    descriptor, opened, source = _open_snapshot_file(root, relative_path, label=label)
    try:
        if opened.st_size < 1 or opened.st_size > maximum_bytes:
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_required_files_unavailable",
                f"{label} file {relative_path!r} has an unsupported size",
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read(opened.st_size + 1)
        _finish_stable_read(
            descriptor,
            opened,
            root=root,
            relative_path=relative_path,
            source=source,
            expected_bytes=len(body),
            label=f"{label} file {relative_path!r}",
        )
        if len(body) != opened.st_size:
            raise RuntimeError(f"{label} file {relative_path!r} was truncated")
        if (
            source.blob_name is not None
            and len(source.blob_name) == 64
            and hashlib.sha256(body).hexdigest() != source.blob_name
        ):
            raise ValueError(
                f"{label} file {relative_path!r} differs from its HF blob SHA-256"
            )
        return body, opened
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class PreparedModelContentFile:
    relative_path: str
    size: int
    raw_sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.relative_path, label="content file relative path")
        _positive_int("content file size", self.size)
        _require_sha256("content file raw digest", self.raw_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size": self.size,
            "raw_sha256": self.raw_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelContentFile:
        row = _strict_object("prepared content file", value, _CONTENT_FILE_FIELDS)
        return cls(
            relative_path=row["relative_path"],
            size=row["size"],
            raw_sha256=row["raw_sha256"],
        )


@dataclass(frozen=True)
class PreparedModelTrustedContentFile:
    """One exact file row projected from the trusted content replay closure."""

    relative_path: str
    size: int
    sha256: str
    storage_kind: Literal["regular", "symlinked_blob"]
    symlink_target: str | None
    resolved_relative_path: str | None

    def __post_init__(self) -> None:
        _safe_relative_path(
            self.relative_path, label="trusted prepared content relative path"
        )
        _nonnegative_int("trusted prepared content file size", self.size)
        _require_sha256("trusted prepared content file", self.sha256)
        if self.storage_kind == "regular":
            if (
                self.symlink_target is not None
                or self.resolved_relative_path is not None
            ):
                raise ValueError(
                    "regular trusted prepared content carries link metadata"
                )
        elif self.storage_kind == "symlinked_blob":
            _require_text(
                "trusted prepared content symlink target", self.symlink_target
            )
            _safe_relative_path(
                self.resolved_relative_path,
                label="trusted prepared content resolved cache path",
            )
        else:
            raise ValueError("trusted prepared content storage kind is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size": self.size,
            "sha256": self.sha256,
            "storage_kind": self.storage_kind,
            "symlink_target": self.symlink_target,
            "resolved_relative_path": self.resolved_relative_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelTrustedContentFile:
        row = _strict_object(
            "trusted prepared content file", value, _TRUSTED_CONTENT_FILE_FIELDS
        )
        return cls(
            relative_path=row["relative_path"],
            size=row["size"],
            sha256=row["sha256"],
            storage_kind=row["storage_kind"],
            symlink_target=row["symlink_target"],
            resolved_relative_path=row["resolved_relative_path"],
        )


@dataclass(frozen=True)
class PreparedModelTrustedContentMember:
    """Exact logical member identity used by one prepared snapshot."""

    model_id: str
    revision: str
    role: Literal["target", "drafter"]
    root: str
    member_sha256: str
    tree_sha256: str
    content_sha256: str
    storage_mode: Literal["regular_tree", "huggingface_cache_symlinks"]
    content_cache_root: str | None
    files: tuple[PreparedModelTrustedContentFile, ...]

    def __post_init__(self) -> None:
        _require_text("trusted prepared member model ID", self.model_id)
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError(
                "trusted prepared member revision must be an immutable SHA"
            )
        if self.role not in {"target", "drafter"}:
            raise ValueError("trusted prepared member role is unsupported")
        _absolute_resolved_path(self.root, label="trusted prepared member root")
        for label, value in (
            ("member", self.member_sha256),
            ("tree", self.tree_sha256),
            ("content", self.content_sha256),
        ):
            _require_sha256(f"trusted prepared member {label}", value)
        if self.storage_mode == "regular_tree":
            if self.content_cache_root is not None:
                raise ValueError("regular trusted prepared member names a cache root")
        elif self.storage_mode == "huggingface_cache_symlinks":
            if self.content_cache_root is None:
                raise ValueError("trusted prepared cache member lacks a cache root")
            cache = _absolute_resolved_path(
                self.content_cache_root, label="trusted prepared member cache root"
            )
            try:
                Path(self.root).relative_to(cache)
            except ValueError as error:
                raise ValueError(
                    "trusted prepared member root leaves its cache root"
                ) from error
        else:
            raise ValueError("trusted prepared member storage mode is unsupported")
        if (
            type(self.files) is not tuple
            or not self.files
            or any(
                type(item) is not PreparedModelTrustedContentFile for item in self.files
            )
        ):
            raise TypeError("trusted prepared member requires exact files")
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("trusted prepared member files are not canonical")
        expected_tree = _canonical_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_model_snapshot_tree",
                "files": [item.to_dict() for item in self.files],
            }
        )
        expected_content = _canonical_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_model_snapshot_content",
                "files": [
                    {"size": item.size, "sha256": item.sha256} for item in self.files
                ],
            }
        )
        if self.tree_sha256 != expected_tree or self.content_sha256 != expected_content:
            raise ValueError("trusted prepared member file digests differ")

    def file(self, relative_path: str) -> PreparedModelTrustedContentFile:
        matches = tuple(
            item for item in self.files if item.relative_path == relative_path
        )
        if len(matches) != 1:
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_required_files_unavailable",
                f"trusted member {self.model_id} lacks exact file {relative_path!r}",
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "role": self.role,
            "root": self.root,
            "member_sha256": self.member_sha256,
            "tree_sha256": self.tree_sha256,
            "content_sha256": self.content_sha256,
            "storage_mode": self.storage_mode,
            "content_cache_root": self.content_cache_root,
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelTrustedContentMember:
        row = _strict_object(
            "trusted prepared content member", value, _TRUSTED_CONTENT_MEMBER_FIELDS
        )
        return cls(
            model_id=row["model_id"],
            revision=row["revision"],
            role=row["role"],
            root=row["root"],
            member_sha256=row["member_sha256"],
            tree_sha256=row["tree_sha256"],
            content_sha256=row["content_sha256"],
            storage_mode=row["storage_mode"],
            content_cache_root=row["content_cache_root"],
            files=tuple(
                PreparedModelTrustedContentFile.from_dict(item)
                for item in _strict_list("trusted prepared member files", row["files"])
            ),
        )


@dataclass(frozen=True)
class PreparedModelTrustedContentReplay:
    """Path-bound trusted bundle and replay-authority identities."""

    content_bundle_path: str
    content_bundle_size: int
    content_bundle_raw_sha256: str
    content_bundle_semantic_sha256: str
    content_bundle_runtime_binding_status: Literal["BOUND"]
    content_replay_path: str
    content_replay_size: int
    content_replay_raw_sha256: str
    content_replay_semantic_sha256: str
    content_replay_protocol_sha256: str

    def __post_init__(self) -> None:
        _absolute_resolved_path(
            self.content_bundle_path, label="trusted content bundle path"
        )
        _positive_int("trusted content bundle size", self.content_bundle_size)
        _absolute_resolved_path(
            self.content_replay_path, label="trusted content replay path"
        )
        _positive_int("trusted content replay size", self.content_replay_size)
        for label, value in (
            ("bundle raw", self.content_bundle_raw_sha256),
            ("bundle semantic", self.content_bundle_semantic_sha256),
            ("replay raw", self.content_replay_raw_sha256),
            ("replay semantic", self.content_replay_semantic_sha256),
            ("replay protocol", self.content_replay_protocol_sha256),
        ):
            _require_sha256(f"trusted content {label}", value)
        if self.content_bundle_runtime_binding_status != "BOUND":
            raise ValueError("trusted prepared content requires a BOUND bundle")

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "content_bundle_path",
                "content_bundle_size",
                "content_bundle_raw_sha256",
                "content_bundle_semantic_sha256",
                "content_bundle_runtime_binding_status",
                "content_replay_path",
                "content_replay_size",
                "content_replay_raw_sha256",
                "content_replay_semantic_sha256",
                "content_replay_protocol_sha256",
            )
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelTrustedContentReplay:
        row = _strict_object(
            "trusted prepared content replay", value, _TRUSTED_CONTENT_REPLAY_FIELDS
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class SnapshotTensorMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    data_start: int
    data_end: int

    def __post_init__(self) -> None:
        _require_text("snapshot tensor name", self.name)
        if type(self.shape) is not tuple or any(
            type(dimension) is not int or dimension < 1 for dimension in self.shape
        ):
            raise ValueError(
                "snapshot tensor shape must contain only positive integers"
            )
        if self.dtype not in {value[0] for value in _SAFETENSORS_DTYPES.values()}:
            raise ValueError("snapshot tensor dtype is unsupported")
        _nonnegative_int("snapshot tensor data start", self.data_start)
        _positive_int("snapshot tensor data end", self.data_end)
        if self.data_end <= self.data_start:
            raise ValueError("snapshot tensor data range must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "data_start": self.data_start,
            "data_end": self.data_end,
        }

    @classmethod
    def from_dict(cls, value: object) -> SnapshotTensorMetadata:
        row = _strict_object("snapshot tensor metadata", value, _TENSOR_FIELDS)
        shape = _strict_list("snapshot tensor shape", row["shape"])
        return cls(
            name=row["name"],
            shape=tuple(shape),
            dtype=row["dtype"],
            data_start=row["data_start"],
            data_end=row["data_end"],
        )


@dataclass(frozen=True)
class SafetensorsHeaderBinding:
    relative_path: str
    file_size: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int
    header_size: int
    header_sha256: str
    raw_sha256: str
    tensors: tuple[SnapshotTensorMetadata, ...]

    def __post_init__(self) -> None:
        path = _safe_relative_path(
            self.relative_path, label="safetensors relative path"
        )
        if not path.endswith(".safetensors"):
            raise ValueError("safetensors binding must name a .safetensors file")
        _positive_int("safetensors file size", self.file_size)
        _nonnegative_int("safetensors device", self.device)
        _positive_int("safetensors inode", self.inode)
        _positive_int("safetensors mtime", self.mtime_ns)
        _positive_int("safetensors ctime", self.ctime_ns)
        _positive_int("safetensors header size", self.header_size)
        _require_sha256("safetensors header digest", self.header_sha256)
        _require_sha256("safetensors payload digest", self.raw_sha256)
        if (
            type(self.tensors) is not tuple
            or not self.tensors
            or any(type(item) is not SnapshotTensorMetadata for item in self.tensors)
        ):
            raise TypeError("safetensors header requires exact tensor metadata")
        names = tuple(item.name for item in self.tensors)
        if names != tuple(sorted(set(names))):
            raise ValueError("safetensors tensor names must be sorted and unique")
        ranges = tuple(
            sorted((item.data_start, item.data_end) for item in self.tensors)
        )
        if ranges[0][0] != 0 or any(
            right[0] != left[1] for left, right in pairwise(ranges)
        ):
            raise ValueError("safetensors data ranges must be exactly contiguous")
        if 8 + self.header_size + ranges[-1][1] != self.file_size:
            raise ValueError(
                "safetensors header and tensor ranges do not cover the file"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "file_size": self.file_size,
            "device": self.device,
            "inode": self.inode,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "header_size": self.header_size,
            "header_sha256": self.header_sha256,
            "raw_sha256": self.raw_sha256,
            "tensors": [item.to_dict() for item in self.tensors],
        }

    @classmethod
    def from_dict(cls, value: object) -> SafetensorsHeaderBinding:
        row = _strict_object("safetensors header binding", value, _HEADER_FIELDS)
        return cls(
            relative_path=row["relative_path"],
            file_size=row["file_size"],
            device=row["device"],
            inode=row["inode"],
            mtime_ns=row["mtime_ns"],
            ctime_ns=row["ctime_ns"],
            header_size=row["header_size"],
            header_sha256=row["header_sha256"],
            raw_sha256=row["raw_sha256"],
            tensors=tuple(
                SnapshotTensorMetadata.from_dict(item)
                for item in _strict_list("safetensors tensors", row["tensors"])
            ),
        )


@dataclass(frozen=True)
class PreparedModelSnapshotContent:
    model_id: str
    revision: str
    root: str
    profile: str
    critical_files: tuple[PreparedModelContentFile, ...]
    weight_kind: Literal["sharded_safetensors", "single_safetensors"]
    weight_headers: tuple[SafetensorsHeaderBinding, ...]
    tensor_metadata_sha256: str
    trusted_content_member: PreparedModelTrustedContentMember | None = None

    def __post_init__(self) -> None:
        _require_text("snapshot content model ID", self.model_id)
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError("snapshot content revision must be an immutable Git SHA")
        root = Path(self.root)
        if not root.is_absolute() or root.resolve(strict=False) != root:
            raise ValueError("snapshot content root must be absolute and resolved")
        _require_text("snapshot content profile", self.profile)
        if self.weight_kind not in {"sharded_safetensors", "single_safetensors"}:
            raise ValueError("snapshot content weight kind is unsupported")
        if (
            type(self.critical_files) is not tuple
            or not self.critical_files
            or any(
                type(item) is not PreparedModelContentFile
                for item in self.critical_files
            )
        ):
            raise TypeError("snapshot content requires exact critical files")
        paths = tuple(item.relative_path for item in self.critical_files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError(
                "snapshot content critical files must be sorted and unique"
            )
        if (
            type(self.weight_headers) is not tuple
            or not self.weight_headers
            or any(
                type(item) is not SafetensorsHeaderBinding
                for item in self.weight_headers
            )
        ):
            raise TypeError("snapshot content requires exact safetensors headers")
        header_paths = tuple(item.relative_path for item in self.weight_headers)
        if header_paths != tuple(sorted(set(header_paths))):
            raise ValueError("snapshot content headers must be path-sorted and unique")
        if self.weight_kind == "single_safetensors" and header_paths != (
            "model.safetensors",
        ):
            raise ValueError("single-safetensors profile requires model.safetensors")
        tensor_names = tuple(item.name for item in self.tensors)
        if tensor_names != tuple(sorted(set(tensor_names))):
            raise ValueError("snapshot tensor inventory must be globally unique")
        _require_sha256("snapshot tensor metadata digest", self.tensor_metadata_sha256)
        if self.tensor_metadata_sha256 != _canonical_sha256(
            [item.to_dict() for item in self.tensors]
        ):
            raise ValueError("snapshot tensor metadata digest differs from headers")
        if self.trusted_content_member is not None:
            if (
                type(self.trusted_content_member)
                is not PreparedModelTrustedContentMember
            ):
                raise TypeError("snapshot trusted content member type differs")
            member = self.trusted_content_member
            if (
                member.model_id != self.model_id
                or member.revision != self.revision
                or member.root != self.root
            ):
                raise ValueError(
                    "snapshot content differs from its trusted logical member"
                )

    @property
    def tensors(self) -> tuple[SnapshotTensorMetadata, ...]:
        return tuple(
            sorted(
                (tensor for header in self.weight_headers for tensor in header.tensors),
                key=lambda item: item.name,
            )
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "model_id": self.model_id,
            "revision": self.revision,
            "root": self.root,
            "profile": self.profile,
            "critical_files": [item.to_dict() for item in self.critical_files],
            "weight_kind": self.weight_kind,
            "weight_headers": [item.to_dict() for item in self.weight_headers],
            "tensor_metadata_sha256": self.tensor_metadata_sha256,
        }
        if self.trusted_content_member is not None:
            result["trusted_content_member"] = self.trusted_content_member.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelSnapshotContent:
        if type(value) is not dict:
            raise TypeError("prepared snapshot content must be an object")
        fields = (
            _SNAPSHOT_CONTENT_FIELDS_V2
            if "trusted_content_member" in value
            else _SNAPSHOT_CONTENT_FIELDS
        )
        row = _strict_object("prepared snapshot content", value, fields)
        trusted = row.get("trusted_content_member")
        return cls(
            model_id=row["model_id"],
            revision=row["revision"],
            root=row["root"],
            profile=row["profile"],
            critical_files=tuple(
                PreparedModelContentFile.from_dict(item)
                for item in _strict_list(
                    "snapshot critical files", row["critical_files"]
                )
            ),
            weight_kind=row["weight_kind"],
            weight_headers=tuple(
                SafetensorsHeaderBinding.from_dict(item)
                for item in _strict_list(
                    "snapshot safetensors headers", row["weight_headers"]
                )
            ),
            tensor_metadata_sha256=row["tensor_metadata_sha256"],
            trusted_content_member=(
                None
                if trusted is None
                else PreparedModelTrustedContentMember.from_dict(trusted)
            ),
        )


def _absolute_resolved_path(value: object, *, label: str) -> Path:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string path")
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise ValueError(f"{label} must be absolute, resolved, and symlink-free")
    return path


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    _absolute_resolved_path(str(path), label=label)
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
            raise RuntimeError(f"{label} must not be a symlink or hardlink")
        if opened.st_size < 1 or opened.st_size > _MAX_CONTENT_MANIFEST_BYTES:
            raise RuntimeError(f"{label} has an unsupported size")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read()
        reopened = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or _stat_identity(opened) != _stat_identity(reopened)
            or _stat_identity(reopened) != _stat_identity(current)
            or len(body) != opened.st_size
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class PreparedModelContentManifestBinding:
    schema_version: int
    kind: Literal["lightcone_prepared_model_content_manifest_binding"]
    path: str
    sidecar_path: str
    semantic_sha256: str
    file_sha256: str
    sidecar_file_sha256: str
    size: int
    sidecar_size: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("prepared content manifest binding schema is unsupported")
        if self.kind != "lightcone_prepared_model_content_manifest_binding":
            raise ValueError("prepared content manifest binding kind is unsupported")
        path = _absolute_resolved_path(self.path, label="content manifest path")
        sidecar = _absolute_resolved_path(
            self.sidecar_path, label="content manifest sidecar"
        )
        if sidecar != Path(f"{path}.sha256"):
            raise ValueError("content manifest sidecar path is not exact")
        for name in ("semantic_sha256", "file_sha256", "sidecar_file_sha256"):
            _require_sha256(f"content manifest {name}", getattr(self, name))
        _positive_int("content manifest size", self.size)
        if type(self.sidecar_size) is not int or self.sidecar_size != 65:
            raise ValueError("content manifest sidecar must be one SHA-256 line")

    @classmethod
    def from_path(cls, path: str | Path) -> PreparedModelContentManifestBinding:
        source = _absolute_resolved_path(str(path), label="content manifest path")
        sidecar = _absolute_resolved_path(
            str(Path(f"{source}.sha256")), label="content manifest sidecar"
        )
        body = _regular_file_bytes(source, label="prepared content manifest")
        sidecar_body = _regular_file_bytes(
            sidecar, label="prepared content manifest sidecar"
        )
        value = _strict_json(body, label="prepared content manifest")
        semantic = _canonical_sha256(value)
        if sidecar_body != f"{semantic}\n".encode("ascii"):
            raise ValueError("prepared content manifest sidecar is invalid")
        return cls(
            schema_version=1,
            kind="lightcone_prepared_model_content_manifest_binding",
            path=str(source),
            sidecar_path=str(sidecar),
            semantic_sha256=semantic,
            file_sha256=hashlib.sha256(body).hexdigest(),
            sidecar_file_sha256=hashlib.sha256(sidecar_body).hexdigest(),
            size=len(body),
            sidecar_size=len(sidecar_body),
        )

    def load(self) -> object:
        body = _regular_file_bytes(
            relocated_evidence_path(self.path), label="bound content manifest"
        )
        sidecar = _regular_file_bytes(
            relocated_evidence_path(self.sidecar_path),
            label="bound content manifest sidecar",
        )
        value = _strict_json(body, label="bound content manifest")
        semantic = _canonical_sha256(value)
        if (
            len(body) != self.size
            or len(sidecar) != self.sidecar_size
            or hashlib.sha256(body).hexdigest() != self.file_sha256
            or hashlib.sha256(sidecar).hexdigest() != self.sidecar_file_sha256
            or semantic != self.semantic_sha256
            or sidecar != f"{semantic}\n".encode("ascii")
        ):
            raise RuntimeError("bound prepared content manifest or sidecar changed")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "path": self.path,
            "sidecar_path": self.sidecar_path,
            "semantic_sha256": self.semantic_sha256,
            "file_sha256": self.file_sha256,
            "sidecar_file_sha256": self.sidecar_file_sha256,
            "size": self.size,
            "sidecar_size": self.sidecar_size,
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelContentManifestBinding:
        row = _strict_object(
            "prepared content manifest binding",
            value,
            _RAW_MANIFEST_BINDING_FIELDS,
        )
        return cls(
            schema_version=row["schema_version"],
            kind=row["kind"],
            path=row["path"],
            sidecar_path=row["sidecar_path"],
            semantic_sha256=row["semantic_sha256"],
            file_sha256=row["file_sha256"],
            sidecar_file_sha256=row["sidecar_file_sha256"],
            size=row["size"],
            sidecar_size=row["sidecar_size"],
        )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class PreparedModelSnapshot:
    """One model ID bound to its immutable local revision directory."""

    model_id: str
    revision: str
    root: str

    def validate(self) -> None:
        if not self.model_id:
            raise ValueError("prepared model ID must be non-empty")
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError("prepared model revision must be an immutable Git SHA")
        identity = Path(self.root)
        if not identity.is_absolute() or identity.resolve() != identity:
            raise ValueError("prepared model root must be absolute and resolved")
        if identity.name != self.revision or identity.parent.name != "snapshots":
            raise ValueError(
                "prepared model root must be the locked revision snapshot directory"
            )
        root = relocated_evidence_path(identity)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("prepared model root must be a regular directory")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "root": self.root,
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelSnapshot:
        if type(value) is not dict or set(value) != {"model_id", "revision", "root"}:
            raise ValueError("prepared model snapshot fields are invalid")
        if any(type(value[name]) is not str for name in value):
            raise TypeError("prepared model snapshot values must be strings")
        result = cls(
            model_id=value["model_id"],
            revision=value["revision"],
            root=value["root"],
        )
        result.validate()
        return result


@dataclass(frozen=True)
class PreparedModelSet:
    """Complete local materialization of one exact :class:`ModelLock`."""

    schema_version: int
    kind: str
    model_lock_sha256: str
    snapshots: tuple[PreparedModelSnapshot, ...]
    protocol_sha256: str = PREPARED_MODEL_BINDING_PROTOCOL_SHA256

    def validate(self) -> None:
        if self.schema_version != 1 or self.kind != "lightcone_prepared_model_set":
            raise ValueError("prepared model set schema is unsupported")
        for name, value in (
            ("model lock", self.model_lock_sha256),
            ("protocol", self.protocol_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"prepared model {name} digest is invalid")
        if self.protocol_sha256 != PREPARED_MODEL_BINDING_PROTOCOL_SHA256:
            raise ValueError("prepared model binding protocol is unsupported")
        if not self.snapshots or any(
            type(snapshot) is not PreparedModelSnapshot for snapshot in self.snapshots
        ):
            raise TypeError("prepared model set requires exact snapshot bindings")
        for snapshot in self.snapshots:
            snapshot.validate()
        identities = tuple(snapshot.model_id for snapshot in self.snapshots)
        if identities != tuple(sorted(set(identities))):
            raise ValueError(
                "prepared model snapshots must be model-ID sorted and unique"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "model_lock_sha256": self.model_lock_sha256,
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "protocol_sha256": self.protocol_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelSet:
        fields = {
            "schema_version",
            "kind",
            "model_lock_sha256",
            "snapshots",
            "protocol_sha256",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("prepared model set fields are invalid")
        if type(value["snapshots"]) is not list:
            raise TypeError("prepared model snapshots must be a list")
        result = cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            model_lock_sha256=value["model_lock_sha256"],
            snapshots=tuple(
                PreparedModelSnapshot.from_dict(item) for item in value["snapshots"]
            ),
            protocol_sha256=value["protocol_sha256"],
        )
        result.validate()
        return result


def bind_prepared_models(
    model_lock: ModelLock,
    roots: Mapping[str, str | Path],
) -> PreparedModelSet:
    """Bind exact Hugging Face snapshot roots to a validated model lock."""

    if type(model_lock) is not ModelLock:
        raise TypeError("prepared model binding requires an exact ModelLock")
    model_lock.validate()
    locked = {model.model_id: model.revision for model in model_lock.models}
    if set(roots) != set(locked):
        raise ValueError("prepared model roots do not cover the model lock exactly")
    canonical_roots: dict[str, str] = {}
    for model_id in locked:
        root = Path(roots[model_id])
        if not root.is_absolute() or root.is_symlink() or root.resolve() != root:
            raise ValueError(
                "prepared model root must be the locked revision snapshot directory"
            )
        canonical_roots[model_id] = str(root)
    result = PreparedModelSet(
        schema_version=1,
        kind="lightcone_prepared_model_set",
        model_lock_sha256=model_lock.sha256,
        snapshots=tuple(
            PreparedModelSnapshot(
                model_id=model_id,
                revision=locked[model_id],
                root=canonical_roots[model_id],
            )
            for model_id in sorted(locked)
        ),
    )
    result.validate()
    return result


def revalidate_prepared_models(
    model_lock: ModelLock,
    prepared: PreparedModelSet,
) -> dict[str, str]:
    """Reopen and exact-compare a prepared model binding before launch."""

    if type(model_lock) is not ModelLock or type(prepared) is not PreparedModelSet:
        raise TypeError("prepared model revalidation requires exact authority types")
    model_lock.validate()
    prepared.validate()
    if prepared.model_lock_sha256 != model_lock.sha256:
        raise ValueError("prepared models bind a different model lock")
    locked = {model.model_id: model.revision for model in model_lock.models}
    observed = {snapshot.model_id: snapshot.revision for snapshot in prepared.snapshots}
    if observed != locked:
        raise ValueError("prepared model revisions differ from the model lock")
    return {snapshot.model_id: snapshot.root for snapshot in prepared.snapshots}


def _trusted_content_replay_from_bindings(
    bundle_binding: object,
    replay_binding: object,
) -> PreparedModelTrustedContentReplay:
    return PreparedModelTrustedContentReplay(
        content_bundle_path=bundle_binding.absolute_path,
        content_bundle_size=bundle_binding.size,
        content_bundle_raw_sha256=bundle_binding.raw_sha256,
        content_bundle_semantic_sha256=bundle_binding.semantic_sha256,
        content_bundle_runtime_binding_status=(bundle_binding.runtime_binding_status),
        content_replay_path=replay_binding.absolute_path,
        content_replay_size=replay_binding.size,
        content_replay_raw_sha256=replay_binding.raw_sha256,
        content_replay_semantic_sha256=replay_binding.semantic_sha256,
        content_replay_protocol_sha256=replay_binding.protocol_sha256,
    )


def _trusted_member_projection(member: object) -> PreparedModelTrustedContentMember:
    return PreparedModelTrustedContentMember(
        model_id=member.model_id,
        revision=member.revision,
        role=member.role,
        root=member.local_snapshot_path,
        member_sha256=member.sha256,
        tree_sha256=member.tree_sha256,
        content_sha256=member.content_sha256,
        storage_mode=member.storage_mode,
        content_cache_root=member.content_cache_root,
        files=tuple(
            PreparedModelTrustedContentFile(
                relative_path=item.relative_path,
                size=item.size,
                sha256=item.sha256,
                storage_kind=item.storage_kind,
                symlink_target=item.symlink_target,
                resolved_relative_path=item.resolved_relative_path,
            )
            for item in member.files
        ),
    )


def _trusted_binding_matches_source(
    binding: object,
    source: PreparedModelTrustedContentReplay,
) -> bool:
    return (
        binding.absolute_path == source.content_bundle_path
        and binding.size == source.content_bundle_size
        and binding.raw_sha256 == source.content_bundle_raw_sha256
        and binding.semantic_sha256 == source.content_bundle_semantic_sha256
        and binding.runtime_binding_status
        == source.content_bundle_runtime_binding_status
    )


def _trusted_replay_binding_matches_source(
    binding: object,
    source: PreparedModelTrustedContentReplay,
) -> bool:
    return (
        binding.absolute_path == source.content_replay_path
        and binding.size == source.content_replay_size
        and binding.raw_sha256 == source.content_replay_raw_sha256
        and binding.semantic_sha256 == source.content_replay_semantic_sha256
        and binding.protocol_sha256 == source.content_replay_protocol_sha256
    )


def _reopen_trusted_content_replay(
    source: PreparedModelTrustedContentReplay,
) -> tuple[object, object]:
    """Reopen schema-2 bundle and metadata-only replay without model payload I/O."""

    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundleBinding,
        TrustedSingleOperatorContentReplayAuthorityBinding,
    )

    bundle_binding = TrustedSingleOperatorContentBundleBinding.bind(
        source.content_bundle_path
    )
    if not _trusted_binding_matches_source(bundle_binding, source):
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_bundle_changed",
            "trusted content bundle binding differs from prepared authority",
        )
    bundle = bundle_binding.reopen()
    if bundle.schema_version != 2 or bundle.runtime_binding_status != "BOUND":
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_bundle_changed",
            "trusted prepared content requires one runtime-BOUND schema-2 bundle",
        )
    replay_binding = bundle.content_replay_authority
    if (
        type(replay_binding) is not TrustedSingleOperatorContentReplayAuthorityBinding
        or not _trusted_replay_binding_matches_source(replay_binding, source)
    ):
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_replay_changed",
            "trusted bundle replay binding differs from prepared authority",
        )
    replay = replay_binding.reopen()
    if (
        replay.semantic_sha256 != source.content_replay_semantic_sha256
        or replay.protocol_sha256 != source.content_replay_protocol_sha256
        or replay.absolute_path != source.content_replay_path
    ):
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_replay_changed",
            "trusted content replay identity differs from prepared authority",
        )
    return bundle, replay


def _project_trusted_prepared_members(
    bundle: object,
    replay: object,
    prepared: PreparedModelSet,
    expected_roles: Mapping[str, Literal["target", "drafter"]],
) -> dict[str, PreparedModelTrustedContentMember]:
    if (
        type(expected_roles) is not dict
        or set(expected_roles) != {snapshot.model_id for snapshot in prepared.snapshots}
        or tuple(sorted(expected_roles.values())) != ("drafter", "target")
    ):
        raise ValueError(
            "trusted prepared roles must cover one target and one drafter exactly"
        )
    projections: dict[str, PreparedModelTrustedContentMember] = {}
    for snapshot in prepared.snapshots:
        expected_role = expected_roles[snapshot.model_id]
        matches = tuple(
            item
            for item in bundle.model_members
            if item.model_id == snapshot.model_id
            and item.revision == snapshot.revision
            and item.local_snapshot_path == snapshot.root
            and item.role == expected_role
        )
        if len(matches) != 1:
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_trusted_member_changed",
                f"{snapshot.model_id} lacks its exact {expected_role} bundle member",
            )
        member = matches[0]
        replay_member = replay.member(model_id=member.model_id, role=member.role)
        closure = replay.closure_for_member(model_id=member.model_id, role=member.role)
        if (
            replay_member != member
            or closure.local_snapshot_path != member.local_snapshot_path
            or closure.storage_mode != member.storage_mode
            or closure.content_cache_root != member.content_cache_root
            or closure.files != member.files
            or closure.tree_sha256 != member.tree_sha256
            or closure.content_sha256 != member.content_sha256
        ):
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_trusted_member_changed",
                f"{snapshot.model_id} differs from its exact replay closure",
            )
        projections[snapshot.model_id] = _trusted_member_projection(member)
    return projections


@dataclass(frozen=True)
class _TrustedPreparedContentContext:
    source: PreparedModelTrustedContentReplay
    members: Mapping[str, PreparedModelTrustedContentMember]
    expected_roles: Mapping[str, Literal["target", "drafter"]]

    def __post_init__(self) -> None:
        if type(self.source) is not PreparedModelTrustedContentReplay:
            raise TypeError("trusted prepared context source type differs")
        if type(self.members) is not dict or any(
            type(item) is not PreparedModelTrustedContentMember
            for item in self.members.values()
        ):
            raise TypeError("trusted prepared context member type differs")
        if (
            type(self.expected_roles) is not dict
            or set(self.expected_roles) != set(self.members)
            or tuple(sorted(self.expected_roles.values())) != ("drafter", "target")
            or any(
                self.members[model_id].role != role
                for model_id, role in self.expected_roles.items()
            )
        ):
            raise ValueError("trusted prepared context role projection differs")

    def finish(self, prepared: PreparedModelSet) -> None:
        bundle, replay = _reopen_trusted_content_replay(self.source)
        if (
            _project_trusted_prepared_members(
                bundle,
                replay,
                prepared,
                self.expected_roles,
            )
            != self.members
        ):
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_trusted_member_changed",
                "trusted prepared members changed during bounded replay",
            )


def _trusted_prepared_context_from_source(
    source: PreparedModelTrustedContentReplay,
    prepared: PreparedModelSet,
    expected_roles: Mapping[str, Literal["target", "drafter"]],
) -> _TrustedPreparedContentContext:
    bundle, replay = _reopen_trusted_content_replay(source)
    return _TrustedPreparedContentContext(
        source=source,
        members=_project_trusted_prepared_members(
            bundle,
            replay,
            prepared,
            expected_roles,
        ),
        expected_roles=expected_roles,
    )


def _trusted_prepared_context_from_bundle(
    bundle_binding: object,
    prepared: PreparedModelSet,
    expected_roles: Mapping[str, Literal["target", "drafter"]],
) -> _TrustedPreparedContentContext:
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundleBinding,
        TrustedSingleOperatorContentReplayAuthorityBinding,
    )

    if type(bundle_binding) is not TrustedSingleOperatorContentBundleBinding:
        raise TypeError(
            "trusted prepared content requires exact content bundle binding"
        )
    rebound = TrustedSingleOperatorContentBundleBinding.bind(
        bundle_binding.absolute_path
    )
    if rebound != bundle_binding:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_bundle_changed",
            "trusted content bundle changed before prepared materialization",
        )
    bundle = rebound.reopen()
    replay_binding = bundle.content_replay_authority
    if (
        bundle.schema_version != 2
        or bundle.runtime_binding_status != "BOUND"
        or type(replay_binding)
        is not TrustedSingleOperatorContentReplayAuthorityBinding
    ):
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_bundle_changed",
            "trusted prepared materialization requires schema-2 replay bundle",
        )
    source = _trusted_content_replay_from_bindings(rebound, replay_binding)
    return _trusted_prepared_context_from_source(
        source,
        prepared,
        expected_roles,
    )


def _tensor_numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _require_opened_trusted_content_file(
    *,
    opened: os.stat_result,
    source: _OpenedSnapshotSource,
    member: PreparedModelTrustedContentMember,
    expected: PreparedModelTrustedContentFile,
    label: str,
) -> None:
    if opened.st_size != expected.size:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_member_changed",
            f"{label} size differs from the exact trusted file row",
        )
    if expected.storage_kind == "regular":
        if source.link_text is not None:
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_trusted_member_changed",
                f"{label} storage kind changed from regular to symlink",
            )
        return
    if source.link_text != expected.symlink_target or member.content_cache_root is None:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_member_changed",
            f"{label} symlink identity differs from the exact trusted file row",
        )
    try:
        resolved = Path(source.target_path).relative_to(member.content_cache_root)
    except ValueError as error:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_member_changed",
            f"{label} resolved blob leaves the trusted cache root",
        ) from error
    if resolved.as_posix() != expected.resolved_relative_path:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_member_changed",
            f"{label} resolved blob path differs from the exact trusted file row",
        )
    if (
        source.blob_name is not None
        and len(source.blob_name) == 64
        and source.blob_name != expected.sha256
    ):
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_member_changed",
            f"{label} blob name differs from the exact trusted payload digest",
        )


def _read_safetensors_header(
    root: Path,
    relative_path: str,
    *,
    label: str,
    trusted_member: PreparedModelTrustedContentMember | None = None,
    trusted_file: PreparedModelTrustedContentFile | None = None,
) -> SafetensorsHeaderBinding:
    if (trusted_member is None) != (trusted_file is None):
        raise TypeError("trusted safetensors member/file must be supplied together")
    descriptor, opened, source = _open_snapshot_file(root, relative_path, label=label)
    try:
        if trusted_member is not None and trusted_file is not None:
            _require_opened_trusted_content_file(
                opened=opened,
                source=source,
                member=trusted_member,
                expected=trusted_file,
                label=label,
            )
        prefix = os.read(descriptor, 8)
        if len(prefix) != 8:
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_safetensors_layout_unsupported",
                f"{label} has no complete safetensors header length",
            )
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > _MAX_SAFETENSORS_HEADER_BYTES:
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_safetensors_layout_unsupported",
                f"{label} safetensors header size is unsupported",
            )
        header = b""
        while len(header) < header_size:
            chunk = os.read(descriptor, header_size - len(header))
            if not chunk:
                break
            header += chunk
        if len(header) != header_size or 8 + header_size > opened.st_size:
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_safetensors_layout_unsupported",
                f"{label} safetensors header is truncated",
            )
        raw_sha256: str
        if trusted_file is None:
            os.lseek(descriptor, 0, os.SEEK_SET)
            payload_hasher = hashlib.sha256()
            payload_bytes = 0
            while True:
                chunk = os.read(descriptor, 8 * 1024 * 1024)
                if not chunk:
                    break
                payload_hasher.update(chunk)
                payload_bytes += len(chunk)
            if payload_bytes != opened.st_size:
                raise RuntimeError(f"{label} payload was truncated")
            raw_sha256 = payload_hasher.hexdigest()
        else:
            # The trusted replay authority already performed the sole complete
            # payload SHA-256 scan.  Prepared replay reads only the bounded
            # safetensors header and projects the exact member file digest.
            raw_sha256 = trusted_file.sha256
        _finish_stable_read(
            descriptor,
            opened,
            root=root,
            relative_path=relative_path,
            source=source,
            expected_bytes=opened.st_size,
            label=label,
        )
        if (
            source.blob_name is not None
            and len(source.blob_name) == 64
            and raw_sha256 != source.blob_name
        ):
            raise ValueError(f"{label} differs from its HF blob SHA-256")
    finally:
        os.close(descriptor)
    raw_header = _strict_json(header, label=f"{label} safetensors header")
    if type(raw_header) is not dict or not raw_header:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_safetensors_layout_unsupported",
            f"{label} safetensors header must be a non-empty object",
        )
    metadata = raw_header.pop("__metadata__", None)
    if metadata is not None and (
        type(metadata) is not dict
        or any(
            type(key) is not str or type(value) is not str
            for key, value in metadata.items()
        )
    ):
        raise ValueError(f"{label} safetensors metadata is invalid")
    tensors: list[SnapshotTensorMetadata] = []
    for name in sorted(raw_header):
        _require_text("safetensors tensor name", name)
        tensor = _strict_object(
            f"safetensors tensor {name!r}",
            raw_header[name],
            frozenset({"dtype", "shape", "data_offsets"}),
        )
        dtype_code = _require_text("safetensors dtype", tensor["dtype"])
        try:
            dtype, item_size = _SAFETENSORS_DTYPES[dtype_code]
        except KeyError as error:
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_safetensors_layout_unsupported",
                f"{label} uses unsupported dtype {dtype_code!r}",
            ) from error
        raw_shape = _strict_list("safetensors shape", tensor["shape"])
        if any(type(dimension) is not int or dimension < 0 for dimension in raw_shape):
            raise ValueError(f"{label} safetensors shape is invalid")
        shape = tuple(raw_shape)
        offsets = _strict_list("safetensors data offsets", tensor["data_offsets"])
        if len(offsets) != 2:
            raise ValueError(f"{label} safetensors data offsets are invalid")
        start = _nonnegative_int("safetensors data start", offsets[0])
        end = _nonnegative_int("safetensors data end", offsets[1])
        expected_bytes = _tensor_numel(shape) * item_size
        if end - start != expected_bytes or expected_bytes < 1:
            raise ValueError(f"{label} safetensors tensor byte range is invalid")
        tensors.append(
            SnapshotTensorMetadata(
                name=name,
                shape=shape,
                dtype=dtype,
                data_start=start,
                data_end=end,
            )
        )
    return SafetensorsHeaderBinding(
        relative_path=relative_path,
        file_size=opened.st_size,
        device=opened.st_dev,
        inode=opened.st_ino,
        mtime_ns=opened.st_mtime_ns,
        ctime_ns=opened.st_ctime_ns,
        header_size=header_size,
        header_sha256=hashlib.sha256(header).hexdigest(),
        raw_sha256=raw_sha256,
        tensors=tuple(tensors),
    )


def _critical_files(
    snapshot: PreparedModelSnapshot,
    profile: _ContentProfile,
    trusted_member: PreparedModelTrustedContentMember | None = None,
) -> tuple[PreparedModelContentFile, ...]:
    root = relocated_evidence_path(snapshot.root)
    rows: list[PreparedModelContentFile] = []
    for relative_path in profile.critical_files:
        body, opened = _read_snapshot_file(
            root,
            relative_path,
            label=f"prepared snapshot {snapshot.model_id}",
        )
        if trusted_member is not None:
            expected = trusted_member.file(relative_path)
            if (
                opened.st_size != expected.size
                or hashlib.sha256(body).hexdigest() != expected.sha256
            ):
                raise PreparedModelContentAuthorityBlocked(
                    "prepared_model_content_trusted_member_changed",
                    f"{snapshot.model_id} file {relative_path!r} differs from "
                    "the exact trusted member row",
                )
        rows.append(
            PreparedModelContentFile(
                relative_path=relative_path,
                size=opened.st_size,
                raw_sha256=hashlib.sha256(body).hexdigest(),
            )
        )
    return tuple(sorted(rows, key=lambda item: item.relative_path))


def _generic_content_profile(
    snapshot: PreparedModelSnapshot,
    trusted_member: PreparedModelTrustedContentMember | None = None,
) -> _ContentProfile:
    """Discover one complete lightweight manifest without model-name trust.

    The fallback is source-owned and deliberately stricter than a signer-picked
    file list: every non-weight regular file below the immutable snapshot root
    is hashed, while weight payloads are represented by exact safetensors
    headers and stable file identities.  Unknown layouts remain BLOCKED.
    """

    root = relocated_evidence_path(snapshot.root)
    discovered: list[str] = []
    safetensors: list[str] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories.sort()
        files.sort()
        for directory in directories:
            candidate = current_path / directory
            status = os.lstat(candidate)
            if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
                raise PreparedModelContentAuthorityBlocked(
                    "prepared_model_content_required_files_unavailable",
                    f"{snapshot.model_id} contains a non-regular directory entry",
                )
        for filename in files:
            candidate = current_path / filename
            relative = candidate.relative_to(root).as_posix()
            _safe_relative_path(relative, label="generic snapshot relative path")
            status = os.lstat(candidate)
            if not (
                (stat.S_ISREG(status.st_mode) and status.st_nlink == 1)
                or stat.S_ISLNK(status.st_mode)
            ):
                raise PreparedModelContentAuthorityBlocked(
                    "prepared_model_content_required_files_unavailable",
                    f"{snapshot.model_id} contains an unsupported file entry",
                )
            if relative.endswith(".incomplete"):
                raise PreparedModelContentAuthorityBlocked(
                    "prepared_model_content_required_files_unavailable",
                    f"{snapshot.model_id} contains an incomplete download",
                )
            if relative.endswith(".safetensors"):
                safetensors.append(relative)
            else:
                discovered.append(relative)
            if len(discovered) + len(safetensors) > _MAX_GENERIC_SNAPSHOT_FILES:
                raise PreparedModelContentAuthorityBlocked(
                    "prepared_model_content_required_files_unavailable",
                    f"{snapshot.model_id} contains too many files",
                )
    if "config.json" not in discovered:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_required_files_unavailable",
            f"{snapshot.model_id} has no config.json",
        )
    if trusted_member is not None and set(discovered) | set(safetensors) != {
        item.relative_path for item in trusted_member.files
    }:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_member_changed",
            f"{snapshot.model_id} file set differs from its exact trusted member",
        )
    has_index = "model.safetensors.index.json" in discovered
    has_single = "model.safetensors" in safetensors
    if has_index == has_single:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_safetensors_layout_unsupported",
            f"{snapshot.model_id} must have exactly one indexed or single layout",
        )
    if has_index:
        weight_kind: Literal["sharded_safetensors", "single_safetensors"] = (
            "sharded_safetensors"
        )
    else:
        if safetensors != ["model.safetensors"]:
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_safetensors_layout_unsupported",
                f"{snapshot.model_id} single layout contains foreign weight files",
            )
        weight_kind = "single_safetensors"
    return _ContentProfile(
        name=str(_GENERIC_CONTENT_PROFILE_PAYLOAD["profile"]),
        critical_files=tuple(discovered),
        weight_kind=weight_kind,
    )


def _sharded_headers(
    snapshot: PreparedModelSnapshot,
    critical: tuple[PreparedModelContentFile, ...],
    trusted_member: PreparedModelTrustedContentMember | None = None,
) -> tuple[SafetensorsHeaderBinding, ...]:
    root = relocated_evidence_path(snapshot.root)
    index_file = next(
        (
            item
            for item in critical
            if item.relative_path == "model.safetensors.index.json"
        ),
        None,
    )
    if index_file is None:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_required_files_unavailable",
            f"{snapshot.model_id} has no registered safetensors index",
        )
    body, _ = _read_snapshot_file(
        root,
        index_file.relative_path,
        label=f"prepared snapshot {snapshot.model_id} index",
    )
    if trusted_member is not None:
        expected_index = trusted_member.file(index_file.relative_path)
        if (
            len(body) != expected_index.size
            or hashlib.sha256(body).hexdigest() != expected_index.sha256
        ):
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_trusted_member_changed",
                f"{snapshot.model_id} index differs from its exact trusted file row",
            )
    index = _strict_object(
        "safetensors index",
        _strict_json(body, label="safetensors index"),
        frozenset({"metadata", "weight_map"}),
    )
    metadata = _strict_object(
        "safetensors index metadata",
        index["metadata"],
        frozenset({"total_size"}),
    )
    total_size = _positive_int("safetensors index total size", metadata["total_size"])
    weight_map = index["weight_map"]
    if type(weight_map) is not dict or not weight_map:
        raise ValueError("safetensors index weight map must be a non-empty object")
    shard_names: set[str] = set()
    for name, shard in weight_map.items():
        _require_text("safetensors index tensor name", name)
        shard_names.add(
            _safe_relative_path(shard, label="safetensors index shard path")
        )
    if any(not name.endswith(".safetensors") for name in shard_names):
        raise ValueError("safetensors index must name only .safetensors shards")
    if trusted_member is not None and shard_names != {
        item.relative_path
        for item in trusted_member.files
        if item.relative_path.endswith(".safetensors")
    }:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_safetensors_layout_unsupported",
            f"{snapshot.model_id} index does not cover exact trusted weight files",
        )
    headers = tuple(
        _read_safetensors_header(
            root,
            relative_path,
            label=f"prepared snapshot {snapshot.model_id} shard {relative_path}",
            trusted_member=trusted_member,
            trusted_file=(
                None if trusted_member is None else trusted_member.file(relative_path)
            ),
        )
        for relative_path in sorted(shard_names)
    )
    observed: dict[str, str] = {}
    tensor_bytes = 0
    for header in headers:
        for tensor in header.tensors:
            if tensor.name in observed:
                raise ValueError("safetensors shards contain duplicate tensor names")
            observed[tensor.name] = header.relative_path
            tensor_bytes += tensor.data_end - tensor.data_start
    if observed != weight_map:
        raise ValueError("safetensors index differs from exact shard headers")
    if tensor_bytes != total_size:
        raise ValueError("safetensors index total size differs from shard headers")
    return headers


def _scan_snapshot_content(
    snapshot: PreparedModelSnapshot,
    prepared: PreparedModelSet,
    trusted_member: PreparedModelTrustedContentMember | None = None,
) -> PreparedModelSnapshotContent:
    if snapshot.model_id == _BANNED_MODEL_ID:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_banned_model",
            f"{snapshot.model_id} is globally prohibited",
        )
    profile = _CONTENT_PROFILES.get(snapshot.model_id)
    if profile is None:
        profile = _generic_content_profile(snapshot, trusted_member)
    if profile.tokenizer_source is not None and profile.tokenizer_source not in {
        item.model_id for item in prepared.snapshots
    }:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_required_files_unavailable",
            f"{snapshot.model_id} requires tokenizer source {profile.tokenizer_source}",
        )
    if trusted_member is not None and (
        trusted_member.model_id != snapshot.model_id
        or trusted_member.revision != snapshot.revision
        or trusted_member.root != snapshot.root
    ):
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_trusted_member_changed",
            f"{snapshot.model_id} prepared identity differs from trusted content",
        )
    critical = _critical_files(snapshot, profile, trusted_member)
    if profile.weight_kind == "sharded_safetensors":
        headers = _sharded_headers(snapshot, critical, trusted_member)
    else:
        if trusted_member is not None and {
            item.relative_path
            for item in trusted_member.files
            if item.relative_path.endswith(".safetensors")
        } != {"model.safetensors"}:
            raise PreparedModelContentAuthorityBlocked(
                "prepared_model_content_safetensors_layout_unsupported",
                f"{snapshot.model_id} single layout differs from trusted file set",
            )
        headers = (
            _read_safetensors_header(
                relocated_evidence_path(snapshot.root),
                "model.safetensors",
                label=f"prepared snapshot {snapshot.model_id} model.safetensors",
                trusted_member=trusted_member,
                trusted_file=(
                    None
                    if trusted_member is None
                    else trusted_member.file("model.safetensors")
                ),
            ),
        )
    tensors = tuple(
        sorted(
            (tensor for header in headers for tensor in header.tensors),
            key=lambda item: item.name,
        )
    )
    return PreparedModelSnapshotContent(
        model_id=snapshot.model_id,
        revision=snapshot.revision,
        root=snapshot.root,
        profile=profile.name,
        critical_files=critical,
        weight_kind=profile.weight_kind,
        weight_headers=headers,
        tensor_metadata_sha256=_canonical_sha256(
            [tensor.to_dict() for tensor in tensors]
        ),
        trusted_content_member=trusted_member,
    )


def _content_manifest(
    model_lock: ModelLock,
    prepared: PreparedModelSet,
    *,
    trusted_context: _TrustedPreparedContentContext | None = None,
) -> tuple[dict[str, object], tuple[PreparedModelSnapshotContent, ...]]:
    revalidate_prepared_models(model_lock, prepared)
    snapshots = tuple(
        _scan_snapshot_content(
            snapshot,
            prepared,
            (
                None
                if trusted_context is None
                else trusted_context.members[snapshot.model_id]
            ),
        )
        for snapshot in prepared.snapshots
    )
    if trusted_context is not None:
        trusted_context.finish(prepared)
    manifest: dict[str, object] = {
        "schema_version": 1 if trusted_context is None else 2,
        "kind": "lightcone_prepared_model_content_manifest",
        "protocol_sha256": (
            PREPARED_MODEL_CONTENT_PROTOCOL_SHA256
            if trusted_context is None
            else PREPARED_MODEL_CONTENT_TRUSTED_REPLAY_PROTOCOL_SHA256
        ),
        "model_lock_sha256": model_lock.sha256,
        "prepared_model_set_sha256": prepared.sha256,
        "snapshots": [snapshot.to_dict() for snapshot in snapshots],
    }
    if trusted_context is not None:
        manifest["trusted_content_replay"] = trusted_context.source.to_dict()
    return manifest, snapshots


def _parse_content_manifest(
    value: object,
) -> tuple[
    dict[str, Any],
    tuple[PreparedModelSnapshotContent, ...],
    PreparedModelTrustedContentReplay | None,
]:
    if type(value) is not dict:
        raise TypeError("prepared model content manifest must be an object")
    version = value.get("schema_version")
    fields = _CONTENT_MANIFEST_FIELDS_V2 if version == 2 else _CONTENT_MANIFEST_FIELDS
    row = _strict_object("prepared model content manifest", value, fields)
    expected_protocol = (
        PREPARED_MODEL_CONTENT_TRUSTED_REPLAY_PROTOCOL_SHA256
        if version == 2
        else PREPARED_MODEL_CONTENT_PROTOCOL_SHA256
    )
    if (
        type(row["schema_version"]) is not int
        or version not in {1, 2}
        or row["kind"] != "lightcone_prepared_model_content_manifest"
        or row["protocol_sha256"] != expected_protocol
    ):
        raise ValueError("prepared model content manifest identity is invalid")
    _require_sha256("content manifest model lock", row["model_lock_sha256"])
    _require_sha256(
        "content manifest prepared model set", row["prepared_model_set_sha256"]
    )
    snapshots = tuple(
        PreparedModelSnapshotContent.from_dict(item)
        for item in _strict_list("content manifest snapshots", row["snapshots"])
    )
    identities = tuple(snapshot.model_id for snapshot in snapshots)
    if not snapshots or identities != tuple(sorted(set(identities))):
        raise ValueError(
            "content manifest snapshots must be model-ID sorted and unique"
        )
    trusted = (
        None
        if version == 1
        else PreparedModelTrustedContentReplay.from_dict(row["trusted_content_replay"])
    )
    if (
        version == 1
        and any(snapshot.trusted_content_member is not None for snapshot in snapshots)
    ) or (
        version == 2
        and any(snapshot.trusted_content_member is None for snapshot in snapshots)
    ):
        raise ValueError("prepared content trusted member/schema coverage differs")
    return row, snapshots, trusted


def materialize_prepared_model_content_manifest(
    model_lock: ModelLock,
    prepared: PreparedModelSet,
) -> dict[str, object]:
    """Scan the release-owned lightweight files and safetensors headers."""

    if type(model_lock) is not ModelLock or type(prepared) is not PreparedModelSet:
        raise TypeError(
            "prepared content materialization requires exact authority types"
        )
    manifest, _ = _content_manifest(model_lock, prepared)
    return manifest


def materialize_trusted_prepared_model_content_manifest(
    model_lock: ModelLock,
    prepared: PreparedModelSet,
    *,
    trusted_content_bundle_binding: object,
    target_model_id: str,
    drafter_model_id: str,
) -> dict[str, object]:
    """Project exact trusted members and read only bounded prepared metadata."""

    if type(model_lock) is not ModelLock or type(prepared) is not PreparedModelSet:
        raise TypeError(
            "trusted prepared content materialization requires exact authority types"
        )
    _require_text("trusted prepared target model", target_model_id)
    _require_text("trusted prepared drafter model", drafter_model_id)
    if target_model_id == drafter_model_id:
        raise ValueError("trusted prepared target and drafter must be distinct")
    expected_roles: dict[str, Literal["target", "drafter"]] = {
        target_model_id: "target",
        drafter_model_id: "drafter",
    }
    context = _trusted_prepared_context_from_bundle(
        trusted_content_bundle_binding,
        prepared,
        expected_roles,
    )
    manifest, _ = _content_manifest(
        model_lock,
        prepared,
        trusted_context=context,
    )
    return manifest


def _trusted_roles_from_snapshots(
    snapshots: tuple[PreparedModelSnapshotContent, ...],
) -> dict[str, Literal["target", "drafter"]]:
    if any(snapshot.trusted_content_member is None for snapshot in snapshots):
        raise ValueError("trusted prepared snapshots lack member roles")
    roles: dict[str, Literal["target", "drafter"]] = {}
    for snapshot in snapshots:
        member = snapshot.trusted_content_member
        assert member is not None
        roles[snapshot.model_id] = member.role
    if tuple(sorted(roles.values())) != ("drafter", "target"):
        raise ValueError(
            "trusted prepared snapshots must bind one target and one drafter"
        )
    return roles


@dataclass(frozen=True)
class PreparedModelContentAuthorityBinding:
    schema_version: int
    kind: Literal["lightcone_prepared_model_content_authority"]
    protocol_sha256: str
    release_manifest_sha256: str
    model_lock_sha256: str
    prepared_model_set: PreparedModelSet
    manifest: PreparedModelContentManifestBinding

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version not in {1, 2}
            or self.kind != "lightcone_prepared_model_content_authority"
        ):
            raise ValueError("prepared model content authority identity is invalid")
        expected_protocol = (
            PREPARED_MODEL_CONTENT_TRUSTED_REPLAY_PROTOCOL_SHA256
            if self.schema_version == 2
            else PREPARED_MODEL_CONTENT_PROTOCOL_SHA256
        )
        if self.protocol_sha256 != expected_protocol:
            raise ValueError("prepared model content authority protocol is unsupported")
        _require_sha256(
            "prepared content release manifest", self.release_manifest_sha256
        )
        _require_sha256("prepared content model lock", self.model_lock_sha256)
        if type(self.prepared_model_set) is not PreparedModelSet:
            raise TypeError(
                "prepared content authority requires exact PreparedModelSet"
            )
        self.prepared_model_set.validate()
        if type(self.manifest) is not PreparedModelContentManifestBinding:
            raise TypeError(
                "prepared content authority requires exact manifest binding"
            )
        if (
            self.release_manifest_sha256 != self.manifest.semantic_sha256
            or self.model_lock_sha256 != self.prepared_model_set.model_lock_sha256
        ):
            raise ValueError("prepared content authority differs from a bound identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "release_manifest_sha256": self.release_manifest_sha256,
            "model_lock_sha256": self.model_lock_sha256,
            "prepared_model_set": self.prepared_model_set.to_dict(),
            "manifest": self.manifest.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelContentAuthorityBinding:
        row = _strict_object(
            "prepared model content authority", value, _CONTENT_AUTHORITY_FIELDS
        )
        return cls(
            schema_version=row["schema_version"],
            kind=row["kind"],
            protocol_sha256=row["protocol_sha256"],
            release_manifest_sha256=row["release_manifest_sha256"],
            model_lock_sha256=row["model_lock_sha256"],
            prepared_model_set=PreparedModelSet.from_dict(row["prepared_model_set"]),
            manifest=PreparedModelContentManifestBinding.from_dict(row["manifest"]),
        )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def revalidate(
        self,
        model_lock: ModelLock,
        *,
        expected_release_manifest_sha256: str,
    ) -> PreparedModelContentAuthorityResult:
        return revalidate_prepared_model_content_authority(
            model_lock,
            self,
            expected_release_manifest_sha256=expected_release_manifest_sha256,
        )


@dataclass(frozen=True)
class PreparedModelContentAuthorityResult:
    binding: PreparedModelContentAuthorityBinding
    snapshots: tuple[PreparedModelSnapshotContent, ...]

    def __post_init__(self) -> None:
        if type(self.binding) is not PreparedModelContentAuthorityBinding:
            raise TypeError("prepared content result requires exact authority binding")
        if (
            type(self.snapshots) is not tuple
            or not self.snapshots
            or any(
                type(item) is not PreparedModelSnapshotContent
                for item in self.snapshots
            )
        ):
            raise TypeError("prepared content result requires exact snapshot content")

    def snapshot(self, model_id: str) -> PreparedModelSnapshotContent:
        matches = tuple(item for item in self.snapshots if item.model_id == model_id)
        if len(matches) != 1:
            raise ValueError("prepared content authority lacks an exact model snapshot")
        return matches[0]


def bind_prepared_model_content_authority(
    model_lock: ModelLock,
    prepared: PreparedModelSet,
    manifest_path: str | Path,
    *,
    expected_release_manifest_sha256: str,
) -> PreparedModelContentAuthorityBinding:
    """Bind a first-party manifest whose digest is pinned outside the snapshot."""

    if type(model_lock) is not ModelLock or type(prepared) is not PreparedModelSet:
        raise TypeError("prepared content binding requires exact authority types")
    expected = _require_sha256(
        "expected release content manifest", expected_release_manifest_sha256
    )
    manifest = PreparedModelContentManifestBinding.from_path(manifest_path)
    if manifest.semantic_sha256 != expected:
        raise ValueError("prepared content manifest differs from the release digest")
    serialized, snapshots, trusted = _parse_content_manifest(manifest.load())
    trusted_context = (
        None
        if trusted is None
        else _trusted_prepared_context_from_source(
            trusted,
            prepared,
            _trusted_roles_from_snapshots(snapshots),
        )
    )
    observed, _ = _content_manifest(
        model_lock,
        prepared,
        trusted_context=trusted_context,
    )
    if serialized != observed:
        raise ValueError("prepared content manifest differs from live snapshot content")
    return PreparedModelContentAuthorityBinding(
        schema_version=serialized["schema_version"],
        kind="lightcone_prepared_model_content_authority",
        protocol_sha256=serialized["protocol_sha256"],
        release_manifest_sha256=expected,
        model_lock_sha256=model_lock.sha256,
        prepared_model_set=prepared,
        manifest=manifest,
    )


def _portable_snapshot_content_identity(
    snapshot: PreparedModelSnapshotContent,
) -> dict[str, object]:
    """Drop only host-local TOCTOU fields after a complete B-side rescan."""

    value = snapshot.to_dict()
    value.pop("root")
    headers = value["weight_headers"]
    if type(headers) is not list:
        raise TypeError("prepared snapshot headers are not an array")
    for header in headers:
        if type(header) is not dict:
            raise TypeError("prepared snapshot header is not an object")
        for name in ("device", "inode", "mtime_ns", "ctime_ns"):
            header.pop(name)
    return value


def revalidate_prepared_model_content_authority(
    model_lock: ModelLock,
    authority: PreparedModelContentAuthorityBinding,
    *,
    expected_release_manifest_sha256: str,
) -> PreparedModelContentAuthorityResult:
    """Reopen the manifest and every registered lightweight snapshot source."""

    if (
        type(model_lock) is not ModelLock
        or type(authority) is not PreparedModelContentAuthorityBinding
    ):
        raise TypeError("prepared content replay requires exact authority types")
    expected = _require_sha256(
        "expected release content manifest", expected_release_manifest_sha256
    )
    authority.__post_init__()
    if (
        authority.release_manifest_sha256 != expected
        or authority.model_lock_sha256 != model_lock.sha256
    ):
        raise ValueError("prepared content authority differs from release/model lock")
    serialized, snapshots, trusted = _parse_content_manifest(authority.manifest.load())
    if (
        authority.schema_version != serialized["schema_version"]
        or authority.protocol_sha256 != serialized["protocol_sha256"]
    ):
        raise ValueError("prepared content authority differs from its manifest")
    trusted_context = (
        None
        if trusted is None
        else _trusted_prepared_context_from_source(
            trusted,
            authority.prepared_model_set,
            _trusted_roles_from_snapshots(snapshots),
        )
    )
    observed, rescanned = _content_manifest(
        model_lock,
        authority.prepared_model_set,
        trusted_context=trusted_context,
    )
    if serialized == observed and snapshots == rescanned:
        return PreparedModelContentAuthorityResult(
            binding=authority,
            snapshots=rescanned,
        )
    if trusted is not None:
        raise ValueError("prepared content authority differs from trusted replay")
    relocated = tuple(
        relocated_evidence_path(snapshot.root) != Path(snapshot.root)
        for snapshot in authority.prepared_model_set.snapshots
    )
    if (
        not relocated
        or not all(relocated)
        or tuple(
            _portable_snapshot_content_identity(snapshot) for snapshot in snapshots
        )
        != tuple(
            _portable_snapshot_content_identity(snapshot) for snapshot in rescanned
        )
    ):
        raise ValueError("prepared content authority differs from live snapshot replay")
    # The serialized A-side rows remain the scientific identity.  The B-side
    # scan proves the same complete bytes/tensors while its device, inode and
    # timestamp fields serve only as local TOCTOU guards.
    return PreparedModelContentAuthorityResult(binding=authority, snapshots=snapshots)


def _require_authorized_prepared_model_content_release(
    model_lock: ModelLock,
    prepared: PreparedModelSet,
    manifest: PreparedModelContentManifestBinding,
    authorization: VerifiedPreparedModelContentRelease,
) -> None:
    """Match one verifier-owned root authorization to live, reopened content.

    Per-role snapshot digests use the canonical nested snapshot object from the
    complete content manifest.  Its canonical byte representation has no
    whitespace or trailing newline, so its raw and semantic SHA-256 are
    intentionally identical.  This prevents a signer from authorizing only a
    model name/revision while leaving the measured tensor/header inventory
    caller-selected.
    """

    if type(authorization) is not VerifiedPreparedModelContentRelease:
        raise TypeError("prepared content binding requires a verified authorization")
    release = authorization.authorization
    if (
        release.model_lock_sha256 != model_lock.sha256
        or release.prepared_model_set_sha256 != prepared.sha256
        or release.content_manifest_raw_sha256 != manifest.file_sha256
        or release.content_manifest_semantic_sha256 != manifest.semantic_sha256
        or release.content_manifest_size != manifest.size
    ):
        raise ValueError(
            "prepared content authorization differs from lock, set, or manifest"
        )
    serialized, snapshots, _ = _parse_content_manifest(manifest.load())
    if (
        serialized["model_lock_sha256"] != model_lock.sha256
        or serialized["prepared_model_set_sha256"] != prepared.sha256
    ):
        raise ValueError("authorized prepared content manifest names another set")
    by_identity = {
        (snapshot.model_id, snapshot.revision): snapshot for snapshot in snapshots
    }
    non_tokenizer = tuple(
        (row.model_id, row.revision)
        for row in release.models
        if row.role != "tokenizer"
    )
    if set(non_tokenizer) != set(by_identity):
        raise ValueError(
            "prepared content authorization does not cover snapshots exactly"
        )
    for row in release.models:
        snapshot = by_identity.get((row.model_id, row.revision))
        if snapshot is None:
            raise ValueError(
                "prepared content authorization role names an unknown snapshot"
            )
        snapshot_sha256 = _canonical_sha256(snapshot.to_dict())
        if (
            row.snapshot_manifest_raw_sha256 != snapshot_sha256
            or row.snapshot_manifest_semantic_sha256 != snapshot_sha256
        ):
            raise ValueError("prepared content authorization snapshot digest differs")


def bind_authorized_prepared_model_content_authority(
    model_lock: ModelLock,
    prepared: PreparedModelSet,
    manifest_path: str | Path,
    *,
    authorization: VerifiedPreparedModelContentRelease,
) -> PreparedModelContentAuthorityBinding:
    """Bind live model bytes only under a verified offline-root wrapper."""

    if type(authorization) is not VerifiedPreparedModelContentRelease:
        raise TypeError("prepared content binding requires a verified authorization")
    release = authorization.authorization
    binding = bind_prepared_model_content_authority(
        model_lock,
        prepared,
        manifest_path,
        expected_release_manifest_sha256=(release.content_manifest_semantic_sha256),
    )
    _require_authorized_prepared_model_content_release(
        model_lock,
        prepared,
        binding.manifest,
        authorization,
    )
    return binding


def revalidate_authorized_prepared_model_content_authority(
    model_lock: ModelLock,
    authority: PreparedModelContentAuthorityBinding,
    *,
    authorization: VerifiedPreparedModelContentRelease,
) -> PreparedModelContentAuthorityResult:
    """Reopen the manifest/snapshots and reject authorization or TOCTOU drift."""

    if type(authorization) is not VerifiedPreparedModelContentRelease:
        raise TypeError("prepared content replay requires a verified authorization")
    release = authorization.authorization
    result = revalidate_prepared_model_content_authority(
        model_lock,
        authority,
        expected_release_manifest_sha256=(release.content_manifest_semantic_sha256),
    )
    _require_authorized_prepared_model_content_release(
        model_lock,
        authority.prepared_model_set,
        authority.manifest,
        authorization,
    )
    return result


def prepared_model_content_authority_to_dict(
    authority: PreparedModelContentAuthorityBinding,
) -> dict[str, object]:
    if type(authority) is not PreparedModelContentAuthorityBinding:
        raise TypeError("prepared content codec requires exact authority binding")
    return authority.to_dict()


def prepared_model_content_authority_from_dict(
    value: object,
) -> PreparedModelContentAuthorityBinding:
    return PreparedModelContentAuthorityBinding.from_dict(value)


__all__ = [
    "PREPARED_MODEL_BINDING_PROTOCOL_SHA256",
    "PREPARED_MODEL_CONTENT_PROTOCOL_SHA256",
    "PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_SHA256S",
    "PREPARED_MODEL_CONTENT_TRUSTED_REPLAY_PROTOCOL_SHA256",
    "PreparedModelContentAuthorityBinding",
    "PreparedModelContentAuthorityBlocked",
    "PreparedModelContentAuthorityResult",
    "PreparedModelContentFile",
    "PreparedModelContentManifestBinding",
    "PreparedModelSet",
    "PreparedModelSnapshot",
    "PreparedModelSnapshotContent",
    "PreparedModelTrustedContentFile",
    "PreparedModelTrustedContentMember",
    "PreparedModelTrustedContentReplay",
    "SafetensorsHeaderBinding",
    "SnapshotTensorMetadata",
    "bind_authorized_prepared_model_content_authority",
    "bind_prepared_model_content_authority",
    "bind_prepared_models",
    "has_prepared_model_content_release_manifest_sha256",
    "materialize_prepared_model_content_manifest",
    "materialize_trusted_prepared_model_content_manifest",
    "prepared_model_content_authority_from_dict",
    "prepared_model_content_authority_to_dict",
    "prepared_model_content_release_identity_sha256",
    "require_prepared_model_content_release_manifest_sha256",
    "revalidate_authorized_prepared_model_content_authority",
    "revalidate_prepared_model_content_authority",
    "revalidate_prepared_models",
]
