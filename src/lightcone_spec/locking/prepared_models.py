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

PREPARED_MODEL_CONTENT_PROTOCOL_SHA256 = _canonical_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_prepared_model_content_protocol",
        "release_profiles": _CONTENT_PROFILE_PAYLOAD,
        "filesystem": (
            "absolute_resolved_revision_root_no_symlink_no_hardlink_"
            "nofollow_stable_stat"
        ),
        "manifest": "path_bound_strict_json_sidecar_and_external_release_sha256",
        "scope": (
            "critical_files_and_safetensors_index_headers_plus_local_shard_"
            "stat_identity_not_weight_payload_hash"
        ),
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
        or len(path.parts) != 1
        or path.parts[0] in {".", ".."}
        or str(path) != text
    ):
        raise ValueError(f"{label} must be one canonical snapshot-relative file")
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


def _open_snapshot_file(
    root: Path,
    relative_path: str,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
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
        current = os.stat(relative, dir_fd=root_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            raise ValueError(f"{label} file {relative!r} is a symlink or hardlink")
        descriptor = os.open(relative, flags, dir_fd=root_descriptor)
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
        or not stat.S_ISREG(current.st_mode)
        or opened.st_nlink != 1
        or current.st_nlink != 1
        or _stat_identity(opened) != _stat_identity(current)
    ):
        os.close(descriptor)
        raise ValueError(f"{label} file {relative!r} is a symlink or hardlink")
    return descriptor, opened


def _finish_stable_read(
    descriptor: int,
    opened: os.stat_result,
    *,
    root: Path,
    relative_path: str,
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
        current = os.stat(relative_path, dir_fd=root_descriptor, follow_symlinks=False)
    finally:
        os.close(root_descriptor)
    if (
        _stat_identity(opened) != _stat_identity(reopened)
        or _stat_identity(reopened) != _stat_identity(current)
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
    descriptor, opened = _open_snapshot_file(root, relative_path, label=label)
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
            expected_bytes=len(body),
            label=f"{label} file {relative_path!r}",
        )
        if len(body) != opened.st_size:
            raise RuntimeError(f"{label} file {relative_path!r} was truncated")
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

    @property
    def tensors(self) -> tuple[SnapshotTensorMetadata, ...]:
        return tuple(
            sorted(
                (tensor for header in self.weight_headers for tensor in header.tensors),
                key=lambda item: item.name,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "root": self.root,
            "profile": self.profile,
            "critical_files": [item.to_dict() for item in self.critical_files],
            "weight_kind": self.weight_kind,
            "weight_headers": [item.to_dict() for item in self.weight_headers],
            "tensor_metadata_sha256": self.tensor_metadata_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelSnapshotContent:
        row = _strict_object(
            "prepared snapshot content", value, _SNAPSHOT_CONTENT_FIELDS
        )
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
        body = _regular_file_bytes(Path(self.path), label="bound content manifest")
        sidecar = _regular_file_bytes(
            Path(self.sidecar_path), label="bound content manifest sidecar"
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
        root = Path(self.root)
        if not root.is_absolute() or root.resolve() != root:
            raise ValueError("prepared model root must be absolute and resolved")
        if root.is_symlink() or not root.is_dir():
            raise ValueError("prepared model root must be a regular directory")
        if root.name != self.revision or root.parent.name != "snapshots":
            raise ValueError(
                "prepared model root must be the locked revision snapshot directory"
            )

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


def _tensor_numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _read_safetensors_header(
    root: Path,
    relative_path: str,
    *,
    label: str,
) -> SafetensorsHeaderBinding:
    descriptor, opened = _open_snapshot_file(root, relative_path, label=label)
    try:
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
        _finish_stable_read(
            descriptor,
            opened,
            root=root,
            relative_path=relative_path,
            expected_bytes=opened.st_size,
            label=label,
        )
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
        tensors=tuple(tensors),
    )


def _critical_files(
    snapshot: PreparedModelSnapshot,
    profile: _ContentProfile,
) -> tuple[PreparedModelContentFile, ...]:
    root = Path(snapshot.root)
    rows: list[PreparedModelContentFile] = []
    for relative_path in profile.critical_files:
        body, opened = _read_snapshot_file(
            root,
            relative_path,
            label=f"prepared snapshot {snapshot.model_id}",
        )
        rows.append(
            PreparedModelContentFile(
                relative_path=relative_path,
                size=opened.st_size,
                raw_sha256=hashlib.sha256(body).hexdigest(),
            )
        )
    return tuple(sorted(rows, key=lambda item: item.relative_path))


def _sharded_headers(
    snapshot: PreparedModelSnapshot,
    critical: tuple[PreparedModelContentFile, ...],
) -> tuple[SafetensorsHeaderBinding, ...]:
    root = Path(snapshot.root)
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
    headers = tuple(
        _read_safetensors_header(
            root,
            relative_path,
            label=f"prepared snapshot {snapshot.model_id} shard {relative_path}",
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
) -> PreparedModelSnapshotContent:
    try:
        profile = _CONTENT_PROFILES[snapshot.model_id]
    except KeyError as error:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_profile_unregistered",
            f"no release-owned content profile exists for {snapshot.model_id!r}",
        ) from error
    if profile.tokenizer_source is not None and profile.tokenizer_source not in {
        item.model_id for item in prepared.snapshots
    }:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_required_files_unavailable",
            f"{snapshot.model_id} requires tokenizer source {profile.tokenizer_source}",
        )
    critical = _critical_files(snapshot, profile)
    if profile.weight_kind == "sharded_safetensors":
        headers = _sharded_headers(snapshot, critical)
    else:
        headers = (
            _read_safetensors_header(
                Path(snapshot.root),
                "model.safetensors",
                label=f"prepared snapshot {snapshot.model_id} model.safetensors",
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
    )


def _content_manifest(
    model_lock: ModelLock,
    prepared: PreparedModelSet,
) -> tuple[dict[str, object], tuple[PreparedModelSnapshotContent, ...]]:
    revalidate_prepared_models(model_lock, prepared)
    snapshots = tuple(
        _scan_snapshot_content(snapshot, prepared) for snapshot in prepared.snapshots
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "lightcone_prepared_model_content_manifest",
        "protocol_sha256": PREPARED_MODEL_CONTENT_PROTOCOL_SHA256,
        "model_lock_sha256": model_lock.sha256,
        "prepared_model_set_sha256": prepared.sha256,
        "snapshots": [snapshot.to_dict() for snapshot in snapshots],
    }
    return manifest, snapshots


def _parse_content_manifest(
    value: object,
) -> tuple[dict[str, Any], tuple[PreparedModelSnapshotContent, ...]]:
    row = _strict_object(
        "prepared model content manifest", value, _CONTENT_MANIFEST_FIELDS
    )
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != 1
        or row["kind"] != "lightcone_prepared_model_content_manifest"
        or row["protocol_sha256"] != PREPARED_MODEL_CONTENT_PROTOCOL_SHA256
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
    return row, snapshots


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
            or self.schema_version != 1
            or self.kind != "lightcone_prepared_model_content_authority"
        ):
            raise ValueError("prepared model content authority identity is invalid")
        if self.protocol_sha256 != PREPARED_MODEL_CONTENT_PROTOCOL_SHA256:
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
    serialized, _ = _parse_content_manifest(manifest.load())
    observed, _ = _content_manifest(model_lock, prepared)
    if serialized != observed:
        raise ValueError("prepared content manifest differs from live snapshot content")
    return PreparedModelContentAuthorityBinding(
        schema_version=1,
        kind="lightcone_prepared_model_content_authority",
        protocol_sha256=PREPARED_MODEL_CONTENT_PROTOCOL_SHA256,
        release_manifest_sha256=expected,
        model_lock_sha256=model_lock.sha256,
        prepared_model_set=prepared,
        manifest=manifest,
    )


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
    serialized, snapshots = _parse_content_manifest(authority.manifest.load())
    observed, rescanned = _content_manifest(model_lock, authority.prepared_model_set)
    if serialized != observed or snapshots != rescanned:
        raise ValueError("prepared content authority differs from live snapshot replay")
    return PreparedModelContentAuthorityResult(binding=authority, snapshots=rescanned)


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
    "PreparedModelContentAuthorityBinding",
    "PreparedModelContentAuthorityBlocked",
    "PreparedModelContentAuthorityResult",
    "PreparedModelContentFile",
    "PreparedModelContentManifestBinding",
    "PreparedModelSet",
    "PreparedModelSnapshot",
    "PreparedModelSnapshotContent",
    "SafetensorsHeaderBinding",
    "SnapshotTensorMetadata",
    "bind_prepared_model_content_authority",
    "bind_prepared_models",
    "has_prepared_model_content_release_manifest_sha256",
    "materialize_prepared_model_content_manifest",
    "prepared_model_content_authority_from_dict",
    "prepared_model_content_authority_to_dict",
    "prepared_model_content_release_identity_sha256",
    "require_prepared_model_content_release_manifest_sha256",
    "revalidate_prepared_model_content_authority",
    "revalidate_prepared_models",
]
