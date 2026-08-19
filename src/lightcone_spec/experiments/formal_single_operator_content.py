"""Path-bound content provenance for the trusted single-operator workflow.

This module is intentionally separate from the offline-root-signed release
authorization path.  It does not manufacture a signed wrapper and it never
upgrades trusted-operator evidence to formal ``MEASURED`` evidence.  Instead,
it derives every content digest from an absolute local path, publishes one
canonical no-replace bundle, and can deep-reopen all paths before execution.

The public producer APIs accept identities and paths, not caller-supplied
content digests.  Codec constructors necessarily contain digests, but a bundle
cannot be published or revalidated unless those values are reproduced from the
bound files.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from lightcone_spec.experiments.formal_single_operator_loads import (
    BURSTGPT_V2_ACTIVE_ASSET,
    BURSTGPT_V2_ASSETS,
    BurstGptV2ReleaseVerification,
    verify_burstgpt_v2_release,
)
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.experiments.workload_authority import (
    LiveCodeBenchV6HardVerificationMetadata,
    Math500Level5VerificationMetadata,
    bind_formal_workload_authority,
    build_livecodebench_v6_hard_verification_metadata,
    build_math500_level5_verification_metadata,
)

TrustedModelRole = Literal["target", "drafter", "tokenizer"]
TrustedSnapshotStorageMode = Literal[
    "regular_tree",
    "huggingface_cache_symlinks",
]
TrustedRuntimeBindingStatus = Literal["PENDING_REMOTE_BINDING", "BOUND"]
TrustedE0DescriptorStatus = Literal[
    "NOT_PROVIDED",
    "PATH_BOUND_NO_COMPLETENESS_CLAIM",
]
TrustedAuxiliaryBackend = Literal["DFLASH", "DSPARK", "EAGLE3", "NEXTN"]

_SHA256 = frozenset("0123456789abcdef")
_GIT_OID_LENGTH = 40
_SOURCE_PATCH_MANIFEST = "patches/sglang/manifest.json"
_FORMAL_STAGE_ORDER = (
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
_FORMAL_STAGES = frozenset(_FORMAL_STAGE_ORDER)
_MAX_JSON_BYTES = 256 * 1024 * 1024


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_git_oid(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != _GIT_OID_LENGTH
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a full lower-case Git object ID")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or "\n" in value
        or "\r" in value
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError(f"{label} must be non-empty single-line NFC text")
    return value


def _require_positive_int(label: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _strict_object(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    if set(value) != expected:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _strict_list(label: str, value: object) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an array")
    return value


def _from_fields(label: str, cls: type, value: object) -> dict[str, object]:
    return _strict_object(label, value, {field.name for field in fields(cls)})


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _strict_json(raw: bytes, *, label: str) -> object:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _resolved_directory(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} path is missing") from error
    if resolved != path or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} path must be a resolved non-symlink directory")
    return path


def _resolved_file(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} path is missing") from error
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} path must be a resolved regular non-symlink file")
    return path


def _stable_file_digest(path: Path, *, label: str) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 0:
            raise ValueError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or size != before.st_size
            or identity(before) != identity(after)
            or identity(after) != identity(current)
        ):
            raise RuntimeError(f"{label} changed while being hashed")
        return size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _stable_file_bytes(
    path_value: str | Path,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[Path, bytes]:
    path = _resolved_file(path_value, label=label)
    size, expected = _stable_file_digest(path, label=label)
    if size > maximum_bytes:
        raise ValueError(f"{label} exceeds its bounded size")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        raw = b""
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"{label} changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew while being read")
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"{label} changed between hashing and reading")
    return path, raw


def _safe_relative_path(value: object, *, label: str) -> str:
    text = _require_text(label, value)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return text


@dataclass(frozen=True)
class TrustedContentFile:
    relative_path: str
    size: int
    sha256: str
    storage_kind: Literal["regular", "symlinked_blob"] = "regular"
    symlink_target: str | None = None
    resolved_relative_path: str | None = None

    def __post_init__(self) -> None:
        _safe_relative_path(self.relative_path, label="trusted content relative path")
        _require_nonnegative_int("trusted content file size", self.size)
        _require_sha256("trusted content file", self.sha256)
        if self.storage_kind == "regular":
            if (
                self.symlink_target is not None
                or self.resolved_relative_path is not None
            ):
                raise ValueError("regular trusted content carries symlink metadata")
        elif self.storage_kind == "symlinked_blob":
            _require_text("trusted content symlink target", self.symlink_target)
            _safe_relative_path(
                self.resolved_relative_path,
                label="trusted content resolved cache path",
            )
        else:
            raise ValueError("trusted content storage kind is unsupported")

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
    def from_dict(cls, value: object) -> Self:
        return cls(**_from_fields("trusted content file", cls, value))  # type: ignore[arg-type]


def _scan_directory(
    root_value: str | Path,
    *,
    label: str,
    storage_mode: TrustedSnapshotStorageMode = "regular_tree",
    content_cache_root: str | Path | None = None,
) -> tuple[TrustedContentFile, ...]:
    root = _resolved_directory(root_value, label=label)
    if storage_mode not in {"regular_tree", "huggingface_cache_symlinks"}:
        raise ValueError(f"{label} storage mode is unsupported")
    cache_root = (
        None
        if content_cache_root is None
        else _resolved_directory(content_cache_root, label=f"{label} cache root")
    )
    if (storage_mode == "regular_tree") != (cache_root is None):
        raise ValueError(f"{label} cache root differs from storage mode")
    if cache_root is not None:
        try:
            root.relative_to(cache_root)
        except ValueError as error:
            raise ValueError(f"{label} snapshot leaves its bound cache root") from error
    seen_inodes: set[tuple[int, int]] = set()
    rows: list[TrustedContentFile] = []

    def visit(directory: Path, prefix: PurePosixPath | None) -> None:
        before = directory.stat(follow_symlinks=False)
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        for entry in entries:
            if unicodedata.normalize("NFC", entry.name) != entry.name:
                raise ValueError(f"{label} contains a non-NFC path")
            metadata = entry.stat(follow_symlinks=False)
            relative = (
                PurePosixPath(entry.name) if prefix is None else prefix / entry.name
            )
            if stat.S_ISLNK(metadata.st_mode):
                if storage_mode != "huggingface_cache_symlinks" or cache_root is None:
                    raise ValueError(f"{label} contains a symlink")
                link_path = Path(entry.path)
                target_text = os.readlink(link_path)
                _require_text(f"{label} symlink target", target_text)
                try:
                    resolved = link_path.resolve(strict=True)
                    resolved_relative = resolved.relative_to(cache_root)
                except (OSError, ValueError) as error:
                    raise ValueError(
                        f"{label} symlink target leaves its bound cache root"
                    ) from error
                resolved_status = resolved.stat(follow_symlinks=False)
                if not stat.S_ISREG(resolved_status.st_mode) or resolved.is_symlink():
                    raise ValueError(f"{label} symlink does not resolve to one blob")
                size, digest = _stable_file_digest(resolved, label=label)
                after_link = link_path.stat(follow_symlinks=False)
                if os.readlink(link_path) != target_text or (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                ) != (
                    after_link.st_dev,
                    after_link.st_ino,
                    after_link.st_mode,
                    after_link.st_mtime_ns,
                    after_link.st_ctime_ns,
                ):
                    raise RuntimeError(f"{label} symlink changed during scan")
                rows.append(
                    TrustedContentFile(
                        relative_path=relative.as_posix(),
                        size=size,
                        sha256=digest,
                        storage_kind="symlinked_blob",
                        symlink_target=target_text,
                        resolved_relative_path=resolved_relative.as_posix(),
                    )
                )
                continue
            if stat.S_ISDIR(metadata.st_mode):
                visit(Path(entry.path), relative)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"{label} contains a non-regular entry")
            inode = (metadata.st_dev, metadata.st_ino)
            if inode in seen_inodes:
                raise ValueError(f"{label} contains duplicate hard-linked files")
            seen_inodes.add(inode)
            size, digest = _stable_file_digest(Path(entry.path), label=label)
            rows.append(
                TrustedContentFile(
                    relative_path=relative.as_posix(),
                    size=size,
                    sha256=digest,
                )
            )
        after = directory.stat(follow_symlinks=False)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"{label} directory changed during scan")

    visit(root, None)
    result = tuple(sorted(rows, key=lambda row: row.relative_path))
    if not result:
        raise ValueError(f"{label} contains no regular files")
    if len({row.relative_path for row in result}) != len(result):
        raise ValueError(f"{label} contains duplicate relative paths")
    return result


@dataclass(frozen=True, order=True)
class TrustedModelRuntimeBinding:
    """Frozen target/backend meaning for one prepared drafter snapshot."""

    stage: Literal["preflight", "E6", "E0"]
    target_model_id: str
    backend: TrustedAuxiliaryBackend
    draft_depth: int

    def __post_init__(self) -> None:
        _require_text("trusted runtime target model", self.target_model_id)
        _require_positive_int("trusted runtime draft depth", self.draft_depth)
        from lightcone_spec.experiments.formal_protocol import E6_MODELS
        from lightcone_spec.experiments.registry import E0_BACKENDS, E0_MODELS

        if self.stage == "preflight":
            if (
                self.backend not in {"DFLASH", "DSPARK"}
                or self.target_model_id != "Qwen/Qwen3-8B"
                or self.draft_depth != 15
            ):
                raise ValueError(
                    "trusted preflight runtime binding differs from protocol"
                )
        elif self.stage == "E6":
            if self.backend != "NEXTN" or self.target_model_id not in E6_MODELS:
                raise ValueError("trusted E6 runtime binding differs from protocol")
        elif (
            self.stage != "E0"
            or self.backend not in E0_BACKENDS
            or self.target_model_id not in E0_MODELS
        ):
            raise ValueError("trusted E0 runtime binding differs from protocol")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "target_model_id": self.target_model_id,
            "backend": self.backend,
            "draft_depth": self.draft_depth,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(**_from_fields("trusted model runtime binding", cls, value))


@dataclass(frozen=True)
class TrustedModelSnapshotSpec:
    model_id: str
    revision: str
    role: TrustedModelRole
    stages: tuple[str, ...]
    local_snapshot_path: str
    storage_mode: TrustedSnapshotStorageMode = "regular_tree"
    content_cache_root: str | None = None
    runtime_bindings: tuple[TrustedModelRuntimeBinding, ...] = ()

    def __post_init__(self) -> None:
        _require_text("trusted model ID", self.model_id)
        _require_text("trusted model revision", self.revision)
        if self.role not in {"target", "drafter", "tokenizer"}:
            raise ValueError("trusted model role is unsupported")
        if (
            type(self.runtime_bindings) is not tuple
            or any(
                type(row) is not TrustedModelRuntimeBinding
                for row in self.runtime_bindings
            )
            or self.runtime_bindings != tuple(sorted(set(self.runtime_bindings)))
            or any(row.stage not in self.stages for row in self.runtime_bindings)
            or any(
                (row.stage == "preflight" and self.role != "drafter")
                or (
                    row.stage == "E6"
                    and (
                        self.role != "target"
                        or row.backend != "NEXTN"
                        or row.target_model_id != self.model_id
                    )
                )
                or (row.stage == "E0" and self.role != "drafter")
                for row in self.runtime_bindings
            )
        ):
            raise ValueError("trusted model runtime bindings are not canonical")
        if (
            type(self.stages) is not tuple
            or not self.stages
            or set(self.stages) - _FORMAL_STAGES
            or self.stages
            != tuple(
                stage for stage in _FORMAL_STAGE_ORDER if stage in set(self.stages)
            )
        ):
            raise ValueError("trusted model stages are not canonical formal stages")
        _resolved_directory(self.local_snapshot_path, label="trusted model snapshot")
        if self.storage_mode == "regular_tree":
            if self.content_cache_root is not None:
                raise ValueError("regular trusted model must not name a cache root")
        elif self.storage_mode == "huggingface_cache_symlinks":
            if self.content_cache_root is None:
                raise ValueError("Hugging Face trusted model lacks its cache root")
            cache_root = _resolved_directory(
                self.content_cache_root,
                label="trusted model cache root",
            )
            try:
                Path(self.local_snapshot_path).relative_to(cache_root)
            except ValueError as error:
                raise ValueError(
                    "trusted model snapshot leaves its cache root"
                ) from error
        else:
            raise ValueError("trusted model storage mode is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "role": self.role,
            "stages": list(self.stages),
            "local_snapshot_path": self.local_snapshot_path,
            "storage_mode": self.storage_mode,
            "content_cache_root": self.content_cache_root,
            "runtime_bindings": [row.to_dict() for row in self.runtime_bindings],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "trusted model snapshot path spec",
            value,
            set(cls.__dataclass_fields__),
        )
        stages = _strict_list("trusted model snapshot spec stages", row.pop("stages"))
        runtime = _strict_list(
            "trusted model snapshot spec runtime bindings",
            row.pop("runtime_bindings"),
        )
        return cls(
            **row,
            stages=tuple(stages),
            runtime_bindings=tuple(
                TrustedModelRuntimeBinding.from_dict(item) for item in runtime
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrustedModelSnapshotMember:
    model_id: str
    revision: str
    role: TrustedModelRole
    stages: tuple[str, ...]
    local_snapshot_path: str
    files: tuple[TrustedContentFile, ...]
    tree_sha256: str
    content_sha256: str
    storage_mode: TrustedSnapshotStorageMode = "regular_tree"
    content_cache_root: str | None = None
    runtime_bindings: tuple[TrustedModelRuntimeBinding, ...] = ()

    def __post_init__(self) -> None:
        TrustedModelSnapshotSpec(
            model_id=self.model_id,
            revision=self.revision,
            role=self.role,
            stages=self.stages,
            local_snapshot_path=self.local_snapshot_path,
            storage_mode=self.storage_mode,
            content_cache_root=self.content_cache_root,
            runtime_bindings=self.runtime_bindings,
        )
        if (
            type(self.files) is not tuple
            or not self.files
            or any(type(row) is not TrustedContentFile for row in self.files)
            or tuple(row.relative_path for row in self.files)
            != tuple(sorted({row.relative_path for row in self.files}))
        ):
            raise ValueError("trusted model content files are not canonical")
        _require_sha256("trusted model tree", self.tree_sha256)
        _require_sha256("trusted model content", self.content_sha256)
        expected_tree = content_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_model_snapshot_tree",
                "files": [row.to_dict() for row in self.files],
            }
        )
        expected_content = content_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_model_snapshot_content",
                "files": [
                    {"size": row.size, "sha256": row.sha256} for row in self.files
                ],
            }
        )
        if self.tree_sha256 != expected_tree or self.content_sha256 != expected_content:
            raise ValueError("trusted model snapshot digest differs from its files")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "role": self.role,
            "stages": list(self.stages),
            "local_snapshot_path": self.local_snapshot_path,
            "files": [row.to_dict() for row in self.files],
            "tree_sha256": self.tree_sha256,
            "content_sha256": self.content_sha256,
            "storage_mode": self.storage_mode,
            "content_cache_root": self.content_cache_root,
            "runtime_bindings": [row.to_dict() for row in self.runtime_bindings],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _from_fields("trusted model snapshot member", cls, value)
        raw_stages = _strict_list("trusted model stages", row.pop("stages"))
        raw_files = _strict_list("trusted model files", row.pop("files"))
        raw_runtime = _strict_list(
            "trusted model runtime bindings", row.pop("runtime_bindings")
        )
        return cls(
            **row,
            stages=tuple(raw_stages),
            files=tuple(TrustedContentFile.from_dict(item) for item in raw_files),
            runtime_bindings=tuple(
                TrustedModelRuntimeBinding.from_dict(item) for item in raw_runtime
            ),
        )  # type: ignore[arg-type]


def bind_trusted_model_snapshot_member(
    spec: TrustedModelSnapshotSpec,
) -> TrustedModelSnapshotMember:
    if type(spec) is not TrustedModelSnapshotSpec:
        raise TypeError("trusted model binder requires an exact path-only spec")
    files = _scan_directory(
        spec.local_snapshot_path,
        label="trusted model snapshot",
        storage_mode=spec.storage_mode,
        content_cache_root=spec.content_cache_root,
    )
    tree_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_model_snapshot_tree",
            "files": [row.to_dict() for row in files],
        }
    )
    content_digest = content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_model_snapshot_content",
            "files": [{"size": row.size, "sha256": row.sha256} for row in files],
        }
    )
    return TrustedModelSnapshotMember(
        model_id=spec.model_id,
        revision=spec.revision,
        role=spec.role,
        stages=spec.stages,
        local_snapshot_path=spec.local_snapshot_path,
        files=files,
        tree_sha256=tree_sha256,
        content_sha256=content_digest,
        storage_mode=spec.storage_mode,
        content_cache_root=spec.content_cache_root,
        runtime_bindings=spec.runtime_bindings,
    )


@dataclass(frozen=True)
class TrustedSglangPatch:
    relative_path: str
    absolute_path: str
    size: int
    sha256: str
    changed_files: tuple[str, ...]

    def __post_init__(self) -> None:
        _safe_relative_path(self.relative_path, label="trusted SGLang patch path")
        path = _resolved_file(self.absolute_path, label="trusted SGLang patch")
        if not self.relative_path.startswith("patches/sglang/"):
            raise ValueError("trusted SGLang patch leaves its source directory")
        if path.as_posix().endswith(self.relative_path) is False:
            raise ValueError("trusted SGLang patch absolute/relative paths differ")
        _require_positive_int("trusted SGLang patch size", self.size)
        _require_sha256("trusted SGLang patch", self.sha256)
        if (
            type(self.changed_files) is not tuple
            or not self.changed_files
            or self.changed_files != tuple(sorted(set(self.changed_files)))
        ):
            raise ValueError("trusted SGLang changed-file list is not canonical")
        for value in self.changed_files:
            _safe_relative_path(value, label="trusted SGLang changed file")

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "absolute_path": self.absolute_path,
            "size": self.size,
            "sha256": self.sha256,
            "changed_files": list(self.changed_files),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _from_fields("trusted SGLang patch", cls, value)
        changed = _strict_list("trusted SGLang changed files", row.pop("changed_files"))
        return cls(**row, changed_files=tuple(changed))  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrustedSourceSnapshot:
    repository_root: str
    git_head: str
    git_tree: str
    files: tuple[TrustedContentFile, ...]
    source_snapshot_sha256: str
    patch_manifest_path: str
    patch_manifest_raw_sha256: str
    patch_manifest_semantic_sha256: str
    sglang_upstream_repository: str
    sglang_upstream_commit: str
    patched_sglang_tree: str
    patches: tuple[TrustedSglangPatch, ...]

    def __post_init__(self) -> None:
        root = _resolved_directory(
            self.repository_root, label="trusted source repository"
        )
        if not (root / ".git").exists():
            raise ValueError("trusted source repository is not a Git checkout")
        _require_git_oid("trusted source Git HEAD", self.git_head)
        _require_git_oid("trusted source Git tree", self.git_tree)
        _require_sha256("trusted source snapshot", self.source_snapshot_sha256)
        manifest = _resolved_file(
            self.patch_manifest_path,
            label="trusted SGLang patch manifest",
        )
        if manifest != root / _SOURCE_PATCH_MANIFEST:
            raise ValueError("trusted SGLang patch manifest path differs")
        _require_sha256(
            "trusted SGLang patch manifest raw", self.patch_manifest_raw_sha256
        )
        _require_sha256(
            "trusted SGLang patch manifest semantic",
            self.patch_manifest_semantic_sha256,
        )
        _require_text(
            "trusted SGLang upstream repository", self.sglang_upstream_repository
        )
        _require_git_oid("trusted SGLang upstream", self.sglang_upstream_commit)
        _require_git_oid("trusted patched SGLang tree", self.patched_sglang_tree)
        if (
            type(self.files) is not tuple
            or not self.files
            or tuple(row.relative_path for row in self.files)
            != tuple(sorted({row.relative_path for row in self.files}))
        ):
            raise ValueError("trusted source files are not canonical")
        expected_source = content_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_single_operator_source_snapshot",
                "git_head": self.git_head,
                "git_tree": self.git_tree,
                "files": [row.to_dict() for row in self.files],
            }
        )
        if self.source_snapshot_sha256 != expected_source:
            raise ValueError("trusted source snapshot digest differs")
        if (
            type(self.patches) is not tuple
            or len(self.patches) != 7
            or any(type(row) is not TrustedSglangPatch for row in self.patches)
            or tuple(row.relative_path for row in self.patches)
            != tuple(sorted({row.relative_path for row in self.patches}))
        ):
            raise ValueError("trusted SGLang patch series must contain seven patches")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_root": self.repository_root,
            "git_head": self.git_head,
            "git_tree": self.git_tree,
            "files": [row.to_dict() for row in self.files],
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "patch_manifest_path": self.patch_manifest_path,
            "patch_manifest_raw_sha256": self.patch_manifest_raw_sha256,
            "patch_manifest_semantic_sha256": self.patch_manifest_semantic_sha256,
            "sglang_upstream_repository": self.sglang_upstream_repository,
            "sglang_upstream_commit": self.sglang_upstream_commit,
            "patched_sglang_tree": self.patched_sglang_tree,
            "patches": [row.to_dict() for row in self.patches],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _from_fields("trusted source snapshot", cls, value)
        raw_files = _strict_list("trusted source files", row.pop("files"))
        raw_patches = _strict_list("trusted SGLang patches", row.pop("patches"))
        return cls(
            **row,
            files=tuple(TrustedContentFile.from_dict(item) for item in raw_files),
            patches=tuple(TrustedSglangPatch.from_dict(item) for item in raw_patches),
        )  # type: ignore[arg-type]


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _tracked_source_files(root: Path) -> tuple[TrustedContentFile, ...]:
    if _git_bytes(root, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise RuntimeError("trusted source snapshot requires a clean Git checkout")
    raw_paths = _git_bytes(root, "ls-files", "-z")
    try:
        paths = tuple(item.decode("utf-8") for item in raw_paths.split(b"\x00") if item)
    except UnicodeDecodeError as error:
        raise ValueError("trusted source path is not UTF-8") from error
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("trusted source tracked-file list is empty or duplicated")
    rows: list[TrustedContentFile] = []
    seen_inodes: set[tuple[int, int]] = set()
    for relative in sorted(paths):
        safe = _safe_relative_path(relative, label="trusted tracked source path")
        path = root / PurePosixPath(safe)
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise ValueError("trusted source contains a non-regular tracked entry")
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in seen_inodes:
            raise ValueError("trusted source contains duplicate hard-linked files")
        seen_inodes.add(inode)
        size, digest = _stable_file_digest(path, label="trusted source file")
        rows.append(TrustedContentFile(safe, size, digest))
    return tuple(rows)


def bind_trusted_source_snapshot(repository_root: str | Path) -> TrustedSourceSnapshot:
    root = _resolved_directory(repository_root, label="trusted source repository")
    if not (root / ".git").exists():
        raise ValueError("trusted source repository is not a Git checkout")
    head = _require_git_oid("trusted source Git HEAD", _git(root, "rev-parse", "HEAD"))
    tree = _require_git_oid(
        "trusted source Git tree",
        _git(root, "rev-parse", "HEAD^{tree}"),
    )
    files = _tracked_source_files(root)
    manifest_path, manifest_raw = _stable_file_bytes(
        root / _SOURCE_PATCH_MANIFEST,
        label="trusted SGLang patch manifest",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    manifest_value = _strict_json(manifest_raw, label="trusted SGLang patch manifest")
    manifest = _strict_object(
        "trusted SGLang patch manifest",
        manifest_value,
        {"schema_version", "upstream", "expected_tree", "patches"},
    )
    upstream = _strict_object(
        "trusted SGLang upstream",
        manifest["upstream"],
        {"repository", "commit"},
    )
    patch_values = _strict_list("trusted SGLang patch series", manifest["patches"])
    if manifest["schema_version"] != 2 or len(patch_values) != 7:
        raise ValueError(
            "trusted SGLang patch manifest must be schema 2 with seven patches"
        )
    patches: list[TrustedSglangPatch] = []
    for raw_patch in patch_values:
        patch = _strict_object(
            "trusted SGLang patch row",
            raw_patch,
            {"file", "sha256", "files"},
        )
        file_name = _safe_relative_path(
            patch["file"],
            label="trusted SGLang manifest patch file",
        )
        if PurePosixPath(file_name).parent != PurePosixPath("."):
            raise ValueError("trusted SGLang patch manifest file must be a basename")
        absolute = root / "patches" / "sglang" / file_name
        size, digest = _stable_file_digest(absolute, label="trusted SGLang patch")
        if digest != _require_sha256("trusted SGLang manifest patch", patch["sha256"]):
            raise ValueError("trusted SGLang patch bytes differ from the manifest")
        changed = tuple(
            sorted(
                _safe_relative_path(item, label="trusted SGLang changed file")
                for item in _strict_list(
                    "trusted SGLang patch changed files",
                    patch["files"],
                )
            )
        )
        patches.append(
            TrustedSglangPatch(
                relative_path=f"patches/sglang/{file_name}",
                absolute_path=str(absolute),
                size=size,
                sha256=digest,
                changed_files=changed,
            )
        )
    result = TrustedSourceSnapshot(
        repository_root=str(root),
        git_head=head,
        git_tree=tree,
        files=files,
        source_snapshot_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_single_operator_source_snapshot",
                "git_head": head,
                "git_tree": tree,
                "files": [row.to_dict() for row in files],
            }
        ),
        patch_manifest_path=str(manifest_path),
        patch_manifest_raw_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        patch_manifest_semantic_sha256=content_sha256(manifest),
        sglang_upstream_repository=_require_text(
            "trusted SGLang upstream repository",
            upstream["repository"],
        ),
        sglang_upstream_commit=_require_git_oid(
            "trusted SGLang upstream commit",
            upstream["commit"],
        ),
        patched_sglang_tree=_require_git_oid(
            "trusted patched SGLang tree",
            manifest["expected_tree"],
        ),
        patches=tuple(sorted(patches, key=lambda row: row.relative_path)),
    )
    if (
        head != _git(root, "rev-parse", "HEAD")
        or tree != _git(root, "rev-parse", "HEAD^{tree}")
        or _git_bytes(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
    ):
        raise RuntimeError("trusted source changed while being bound")
    return result


@dataclass(frozen=True)
class TrustedLockedWorkload:
    workload_id: Literal["livecodebench_v6_hard", "math500_level5"]
    raw_source_path: str
    raw_file_size: int
    raw_file_sha256: str
    repository_revision: str
    raw_row_count: int
    selected_row_count: int
    selected_source_row_ids: tuple[str, ...]
    selected_raw_rows_sha256: str
    formal_samples_sha256: str
    protocol_sha256: str
    source_lock_sha256: str
    authority_sha256: str
    verification_metadata_sha256: str
    verification_metadata: (
        LiveCodeBenchV6HardVerificationMetadata | Math500Level5VerificationMetadata
    )

    def __post_init__(self) -> None:
        _resolved_file(self.raw_source_path, label="trusted workload raw source")
        _require_positive_int("trusted workload raw size", self.raw_file_size)
        _require_sha256("trusted workload raw file", self.raw_file_sha256)
        _require_git_oid("trusted workload revision", self.repository_revision)
        _require_positive_int("trusted workload raw rows", self.raw_row_count)
        _require_positive_int("trusted workload selected rows", self.selected_row_count)
        if (
            type(self.selected_source_row_ids) is not tuple
            or len(self.selected_source_row_ids) != self.selected_row_count
            or len(set(self.selected_source_row_ids)) != self.selected_row_count
        ):
            raise ValueError("trusted workload selected source IDs differ")
        for source_id in self.selected_source_row_ids:
            _require_text("trusted workload selected source ID", source_id)
        for label, digest in (
            ("selected raw rows", self.selected_raw_rows_sha256),
            ("formal samples", self.formal_samples_sha256),
            ("protocol", self.protocol_sha256),
            ("source lock", self.source_lock_sha256),
            ("authority", self.authority_sha256),
            ("verification metadata", self.verification_metadata_sha256),
        ):
            _require_sha256(f"trusted workload {label}", digest)
        expected_metadata_type = (
            LiveCodeBenchV6HardVerificationMetadata
            if self.workload_id == "livecodebench_v6_hard"
            else Math500Level5VerificationMetadata
        )
        if (
            type(self.verification_metadata) is not expected_metadata_type
            or self.verification_metadata.sha256 != self.verification_metadata_sha256
            or self.verification_metadata.workload_id != self.workload_id
            or self.verification_metadata.raw_file_sha256 != self.raw_file_sha256
            or self.verification_metadata.repository_revision
            != self.repository_revision
            or self.verification_metadata.raw_row_count != self.raw_row_count
            or self.verification_metadata.selected_row_count != self.selected_row_count
            or self.verification_metadata.selected_raw_rows_sha256
            != self.selected_raw_rows_sha256
            or self.verification_metadata.formal_samples_sha256
            != self.formal_samples_sha256
            or self.verification_metadata.protocol_sha256 != self.protocol_sha256
            or self.verification_metadata.source_lock_sha256 != self.source_lock_sha256
        ):
            raise ValueError("trusted workload verification metadata digest differs")
        metadata_source_ids = (
            self.verification_metadata.selected_question_ids
            if type(self.verification_metadata)
            is LiveCodeBenchV6HardVerificationMetadata
            else self.verification_metadata.selected_source_row_ids
        )
        if metadata_source_ids != self.selected_source_row_ids:
            raise ValueError("trusted workload selected IDs differ from metadata")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "workload_id": self.workload_id,
            "raw_source_path": self.raw_source_path,
            "raw_file_size": self.raw_file_size,
            "raw_file_sha256": self.raw_file_sha256,
            "repository_revision": self.repository_revision,
            "raw_row_count": self.raw_row_count,
            "selected_row_count": self.selected_row_count,
            "selected_source_row_ids": list(self.selected_source_row_ids),
            "selected_raw_rows_sha256": self.selected_raw_rows_sha256,
            "formal_samples_sha256": self.formal_samples_sha256,
            "protocol_sha256": self.protocol_sha256,
            "source_lock_sha256": self.source_lock_sha256,
            "authority_sha256": self.authority_sha256,
            "verification_metadata_sha256": self.verification_metadata_sha256,
            "verification_metadata": self.verification_metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _from_fields("trusted locked workload", cls, value)
        raw_ids = _strict_list(
            "trusted workload selected source IDs",
            row.pop("selected_source_row_ids"),
        )
        metadata = _strict_object(
            "trusted workload verification metadata",
            row.pop("verification_metadata"),
            set(
                LiveCodeBenchV6HardVerificationMetadata.__dataclass_fields__
                if row["workload_id"] == "livecodebench_v6_hard"
                else Math500Level5VerificationMetadata.__dataclass_fields__
            )
            | {"verification_metadata_sha256"},
        )
        if row["workload_id"] == "livecodebench_v6_hard":
            decoded = LiveCodeBenchV6HardVerificationMetadata.from_dict(metadata)
        elif row["workload_id"] == "math500_level5":
            decoded = Math500Level5VerificationMetadata.from_dict(metadata)
        else:
            raise ValueError("trusted workload ID is unsupported")
        return cls(
            **row,
            selected_source_row_ids=tuple(raw_ids),
            verification_metadata=decoded,
        )  # type: ignore[arg-type]


def bind_trusted_locked_workload(
    workload_id: Literal["livecodebench_v6_hard", "math500_level5"],
    raw_source_path: str | Path,
) -> TrustedLockedWorkload:
    authority = bind_formal_workload_authority(workload_id, raw_source_path)
    metadata: (
        LiveCodeBenchV6HardVerificationMetadata | Math500Level5VerificationMetadata
    )
    if workload_id == "livecodebench_v6_hard":
        metadata = build_livecodebench_v6_hard_verification_metadata(authority)
        selected_ids = metadata.selected_question_ids
    else:
        metadata = build_math500_level5_verification_metadata(authority)
        selected_ids = metadata.selected_source_row_ids
    size, raw_digest = _stable_file_digest(
        Path(authority.raw_source_path),
        label="trusted workload raw source",
    )
    if raw_digest != authority.raw_file_sha256:
        raise RuntimeError("trusted workload changed after authority binding")
    return TrustedLockedWorkload(
        workload_id=workload_id,
        raw_source_path=authority.raw_source_path,
        raw_file_size=size,
        raw_file_sha256=authority.raw_file_sha256,
        repository_revision=authority.repository_revision,
        raw_row_count=authority.raw_row_count,
        selected_row_count=authority.selected_row_count,
        selected_source_row_ids=selected_ids,
        selected_raw_rows_sha256=metadata.selected_raw_rows_sha256,
        formal_samples_sha256=metadata.formal_samples_sha256,
        protocol_sha256=authority.protocol_sha256,
        source_lock_sha256=authority.source_lock_sha256,
        authority_sha256=authority.sha256,
        verification_metadata_sha256=metadata.sha256,
        verification_metadata=metadata,
    )


@dataclass(frozen=True)
class TrustedBurstGptAssetPath:
    name: str
    absolute_path: str
    size: int
    sha256: str
    row_count: int

    def __post_init__(self) -> None:
        _require_text("trusted BurstGPT asset name", self.name)
        path = _resolved_file(self.absolute_path, label="trusted BurstGPT asset")
        if path.name != self.name:
            raise ValueError("trusted BurstGPT asset path/name differs")
        _require_positive_int("trusted BurstGPT asset size", self.size)
        _require_sha256("trusted BurstGPT asset", self.sha256)
        _require_positive_int("trusted BurstGPT row count", self.row_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "absolute_path": self.absolute_path,
            "size": self.size,
            "sha256": self.sha256,
            "row_count": self.row_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(**_from_fields("trusted BurstGPT asset", cls, value))  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrustedBurstGptRelease:
    active_asset: str
    assets: tuple[TrustedBurstGptAssetPath, ...]
    release_verification: BurstGptV2ReleaseVerification
    release_verification_sha256: str

    def __post_init__(self) -> None:
        if self.active_asset != BURSTGPT_V2_ACTIVE_ASSET:
            raise ValueError("trusted BurstGPT active asset differs")
        expected_names = tuple(row.name for row in BURSTGPT_V2_ASSETS)
        if (
            type(self.assets) is not tuple
            or tuple(row.name for row in self.assets) != expected_names
            or any(type(row) is not TrustedBurstGptAssetPath for row in self.assets)
        ):
            raise ValueError("trusted BurstGPT path coverage differs")
        if type(self.release_verification) is not BurstGptV2ReleaseVerification:
            raise TypeError("trusted BurstGPT verification type differs")
        _require_sha256(
            "trusted BurstGPT release verification",
            self.release_verification_sha256,
        )
        if self.release_verification.sha256 != self.release_verification_sha256:
            raise ValueError("trusted BurstGPT release verification digest differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "active_asset": self.active_asset,
            "assets": [row.to_dict() for row in self.assets],
            "release_verification": self.release_verification.to_dict(),
            "release_verification_sha256": self.release_verification_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _from_fields("trusted BurstGPT release", cls, value)
        assets = _strict_list("trusted BurstGPT assets", row.pop("assets"))
        verification = row.pop("release_verification")
        return cls(
            **row,
            assets=tuple(TrustedBurstGptAssetPath.from_dict(item) for item in assets),
            release_verification=BurstGptV2ReleaseVerification.from_dict(verification),
        )  # type: ignore[arg-type]


def bind_trusted_burstgpt_release(
    asset_paths: Mapping[str, str | Path],
) -> TrustedBurstGptRelease:
    if type(asset_paths) is not dict:
        asset_paths = dict(asset_paths)
    resolved = {
        name: _resolved_file(path, label=f"trusted BurstGPT {name}")
        for name, path in asset_paths.items()
    }
    verification = verify_burstgpt_v2_release(resolved)
    by_name = {row.name: row for row in verification.assets}
    return TrustedBurstGptRelease(
        active_asset=verification.active_asset,
        assets=tuple(
            TrustedBurstGptAssetPath(
                name=expected.name,
                absolute_path=str(resolved[expected.name]),
                size=by_name[expected.name].size,
                sha256=by_name[expected.name].sha256,
                row_count=by_name[expected.name].row_count,
            )
            for expected in BURSTGPT_V2_ASSETS
        ),
        release_verification=verification,
        release_verification_sha256=verification.sha256,
    )


@dataclass(frozen=True)
class TrustedJsonArtifact:
    artifact_id: str
    absolute_path: str
    size: int
    raw_sha256: str
    semantic_sha256: str

    def __post_init__(self) -> None:
        _require_text("trusted JSON artifact ID", self.artifact_id)
        _resolved_file(self.absolute_path, label="trusted JSON artifact")
        _require_positive_int("trusted JSON artifact size", self.size)
        _require_sha256("trusted JSON artifact raw", self.raw_sha256)
        _require_sha256("trusted JSON artifact semantic", self.semantic_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "absolute_path": self.absolute_path,
            "size": self.size,
            "raw_sha256": self.raw_sha256,
            "semantic_sha256": self.semantic_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(**_from_fields("trusted JSON artifact", cls, value))  # type: ignore[arg-type]


def bind_trusted_json_artifact(
    artifact_id: str,
    path: str | Path,
) -> TrustedJsonArtifact:
    source, raw = _stable_file_bytes(
        path,
        label=f"trusted JSON artifact {artifact_id}",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    value = _strict_json(raw, label=f"trusted JSON artifact {artifact_id}")
    return TrustedJsonArtifact(
        artifact_id=_require_text("trusted JSON artifact ID", artifact_id),
        absolute_path=str(source),
        size=len(raw),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=content_sha256(value),
    )


def revalidate_trusted_json_artifact(
    binding: TrustedJsonArtifact,
) -> TrustedJsonArtifact:
    if type(binding) is not TrustedJsonArtifact:
        raise TypeError("trusted JSON revalidator requires an exact binding")
    rebound = bind_trusted_json_artifact(binding.artifact_id, binding.absolute_path)
    if rebound != binding:
        raise RuntimeError(f"trusted JSON artifact {binding.artifact_id} changed")
    return rebound


@dataclass(frozen=True)
class TrustedE0TaskNativeDescriptorSpec:
    descriptor_id: str
    task: str
    repository: str
    revision: str
    descriptor_path: str

    def __post_init__(self) -> None:
        _require_text("trusted E0 descriptor ID", self.descriptor_id)
        _require_text("trusted E0 task", self.task)
        _require_text("trusted E0 repository", self.repository)
        _require_text("trusted E0 revision", self.revision)
        _resolved_file(self.descriptor_path, label="trusted E0 descriptor")

    def to_dict(self) -> dict[str, object]:
        return {
            "descriptor_id": self.descriptor_id,
            "task": self.task,
            "repository": self.repository,
            "revision": self.revision,
            "descriptor_path": self.descriptor_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_from_fields("trusted E0 task-native descriptor path spec", cls, value)
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrustedE0TaskNativeDescriptor:
    descriptor_id: str
    task: str
    repository: str
    revision: str
    source: TrustedJsonArtifact

    def __post_init__(self) -> None:
        _require_text("trusted E0 descriptor ID", self.descriptor_id)
        _require_text("trusted E0 task", self.task)
        _require_text("trusted E0 repository", self.repository)
        _require_text("trusted E0 revision", self.revision)
        if (
            type(self.source) is not TrustedJsonArtifact
            or self.source.artifact_id != f"e0_task_native:{self.descriptor_id}"
        ):
            raise ValueError("trusted E0 descriptor source identity differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "descriptor_id": self.descriptor_id,
            "task": self.task,
            "repository": self.repository,
            "revision": self.revision,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _from_fields("trusted E0 task-native descriptor", cls, value)
        source = row.pop("source")
        return cls(**row, source=TrustedJsonArtifact.from_dict(source))  # type: ignore[arg-type]


def bind_trusted_e0_task_native_descriptor(
    spec: TrustedE0TaskNativeDescriptorSpec,
) -> TrustedE0TaskNativeDescriptor:
    if type(spec) is not TrustedE0TaskNativeDescriptorSpec:
        raise TypeError("trusted E0 descriptor binder requires an exact path-only spec")
    return TrustedE0TaskNativeDescriptor(
        descriptor_id=spec.descriptor_id,
        task=spec.task,
        repository=spec.repository,
        revision=spec.revision,
        source=bind_trusted_json_artifact(
            f"e0_task_native:{spec.descriptor_id}",
            spec.descriptor_path,
        ),
    )


@dataclass(frozen=True)
class TrustedRuntimeObservations:
    inventory: TrustedJsonArtifact
    doctor: TrustedJsonArtifact

    def __post_init__(self) -> None:
        if (
            type(self.inventory) is not TrustedJsonArtifact
            or self.inventory.artifact_id != "remote_gpu_inventory"
            or type(self.doctor) is not TrustedJsonArtifact
            or self.doctor.artifact_id != "remote_runtime_doctor"
            or self.inventory.absolute_path == self.doctor.absolute_path
        ):
            raise ValueError("trusted runtime observation bindings differ")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "inventory": self.inventory.to_dict(),
            "doctor": self.doctor.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _from_fields("trusted runtime observations", cls, value)
        return cls(
            inventory=TrustedJsonArtifact.from_dict(row["inventory"]),
            doctor=TrustedJsonArtifact.from_dict(row["doctor"]),
        )


@dataclass(frozen=True, order=True)
class TrustedNamedInputPath:
    """One code-named path in the reproducible trusted-content input spec."""

    name: str
    absolute_path: str

    def __post_init__(self) -> None:
        _require_text("trusted named input", self.name)
        _resolved_file(self.absolute_path, label=f"trusted named input {self.name}")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "absolute_path": self.absolute_path}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(**_from_fields("trusted named input path", cls, value))  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrustedSingleOperatorContentPathSpec:
    """Canonical path-only recipe for publishing one runtime-bound bundle.

    The spec intentionally contains no caller-provided content digest.  All
    source, model, workload, trace, inventory, and doctor identities are
    scanned again by the source-owned publishers at execution time.
    """

    schema_version: Literal[1]
    kind: Literal["trusted_single_operator_content_path_spec"]
    repository_root: str
    model_specs: tuple[TrustedModelSnapshotSpec, ...]
    livecodebench_raw_path: str
    math500_raw_path: str
    burstgpt_asset_paths: tuple[TrustedNamedInputPath, ...]
    e0_task_native_specs: tuple[TrustedE0TaskNativeDescriptorSpec, ...]
    inventory_path: str
    doctor_path: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "trusted_single_operator_content_path_spec"
        ):
            raise ValueError("trusted content path spec identity differs")
        _resolved_directory(self.repository_root, label="trusted source repository")
        if (
            type(self.model_specs) is not tuple
            or not self.model_specs
            or any(
                type(row) is not TrustedModelSnapshotSpec for row in self.model_specs
            )
            or self.model_specs
            != tuple(
                sorted(
                    self.model_specs,
                    key=lambda row: (
                        row.role,
                        row.model_id,
                        row.revision,
                        row.local_snapshot_path,
                    ),
                )
            )
            or len({(row.role, row.model_id, row.revision) for row in self.model_specs})
            != len(self.model_specs)
        ):
            raise ValueError("trusted content model path specs are not canonical")
        _resolved_file(
            self.livecodebench_raw_path,
            label="trusted LiveCodeBench raw source",
        )
        _resolved_file(self.math500_raw_path, label="trusted MATH-500 raw source")
        expected_assets = tuple(row.name for row in BURSTGPT_V2_ASSETS)
        if (
            type(self.burstgpt_asset_paths) is not tuple
            or any(
                type(row) is not TrustedNamedInputPath
                for row in self.burstgpt_asset_paths
            )
            or tuple(row.name for row in self.burstgpt_asset_paths) != expected_assets
        ):
            raise ValueError("trusted content BurstGPT path coverage differs")
        if (
            type(self.e0_task_native_specs) is not tuple
            or any(
                type(row) is not TrustedE0TaskNativeDescriptorSpec
                for row in self.e0_task_native_specs
            )
            or self.e0_task_native_specs
            != tuple(
                sorted(self.e0_task_native_specs, key=lambda row: row.descriptor_id)
            )
            or len({row.descriptor_id for row in self.e0_task_native_specs})
            != len(self.e0_task_native_specs)
        ):
            raise ValueError("trusted content E0 descriptor specs are not canonical")
        inventory = _resolved_file(
            self.inventory_path,
            label="trusted remote inventory",
        )
        doctor = _resolved_file(self.doctor_path, label="trusted remote doctor")
        if inventory == doctor:
            raise ValueError("trusted inventory and doctor paths must differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "repository_root": self.repository_root,
            "model_specs": [row.to_dict() for row in self.model_specs],
            "livecodebench_raw_path": self.livecodebench_raw_path,
            "math500_raw_path": self.math500_raw_path,
            "burstgpt_asset_paths": [
                row.to_dict() for row in self.burstgpt_asset_paths
            ],
            "e0_task_native_specs": [
                row.to_dict() for row in self.e0_task_native_specs
            ],
            "inventory_path": self.inventory_path,
            "doctor_path": self.doctor_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "trusted single-operator content path spec",
            value,
            set(cls.__dataclass_fields__),
        )
        models = _strict_list(
            "trusted content model path specs", row.pop("model_specs")
        )
        burst = _strict_list(
            "trusted content BurstGPT asset paths",
            row.pop("burstgpt_asset_paths"),
        )
        e0 = _strict_list(
            "trusted content E0 descriptor path specs",
            row.pop("e0_task_native_specs"),
        )
        return cls(
            **row,
            model_specs=tuple(
                TrustedModelSnapshotSpec.from_dict(item) for item in models
            ),
            burstgpt_asset_paths=tuple(
                TrustedNamedInputPath.from_dict(item) for item in burst
            ),
            e0_task_native_specs=tuple(
                TrustedE0TaskNativeDescriptorSpec.from_dict(item) for item in e0
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrustedSingleOperatorContentBundle:
    schema_version: Literal[1]
    kind: Literal["trusted_single_operator_content_bundle"]
    trust_mode: Literal["trusted_single_operator_no_signature"]
    signature: None
    formal_measured_authorization: Literal[False]
    claim_scope: Literal["trusted_single_operator_empirical_content_provenance"]
    source_snapshot: TrustedSourceSnapshot
    model_members: tuple[TrustedModelSnapshotMember, ...]
    locked_workloads: tuple[TrustedLockedWorkload, ...]
    burstgpt_release: TrustedBurstGptRelease
    e0_task_native_descriptor_status: TrustedE0DescriptorStatus
    e0_task_native_descriptors: tuple[TrustedE0TaskNativeDescriptor, ...]
    runtime_binding_status: TrustedRuntimeBindingStatus
    runtime_observations: TrustedRuntimeObservations | None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "trusted_single_operator_content_bundle"
            or self.trust_mode != "trusted_single_operator_no_signature"
            or self.signature is not None
            or self.formal_measured_authorization is not False
            or self.claim_scope
            != "trusted_single_operator_empirical_content_provenance"
        ):
            raise ValueError("trusted single-operator content schema differs")
        if type(self.source_snapshot) is not TrustedSourceSnapshot:
            raise TypeError("trusted content source snapshot type differs")
        if (
            type(self.model_members) is not tuple
            or any(
                type(row) is not TrustedModelSnapshotMember
                for row in self.model_members
            )
            or self.model_members
            != tuple(
                sorted(
                    self.model_members,
                    key=lambda row: (
                        row.role,
                        row.model_id,
                        row.revision,
                        row.local_snapshot_path,
                    ),
                )
            )
            or len(
                {(row.role, row.model_id, row.revision) for row in self.model_members}
            )
            != len(self.model_members)
            or {row.role for row in self.model_members}
            != {"target", "drafter", "tokenizer"}
        ):
            raise ValueError("trusted model member coverage is not canonical")
        if tuple(row.workload_id for row in self.locked_workloads) != (
            "livecodebench_v6_hard",
            "math500_level5",
        ):
            raise ValueError("trusted workload coverage differs")
        if type(self.burstgpt_release) is not TrustedBurstGptRelease:
            raise TypeError("trusted BurstGPT release type differs")
        expected_e0_status: TrustedE0DescriptorStatus = (
            "NOT_PROVIDED"
            if not self.e0_task_native_descriptors
            else "PATH_BOUND_NO_COMPLETENESS_CLAIM"
        )
        if (
            self.e0_task_native_descriptor_status != expected_e0_status
            or self.e0_task_native_descriptors
            != tuple(
                sorted(
                    self.e0_task_native_descriptors,
                    key=lambda row: row.descriptor_id,
                )
            )
            or len({row.descriptor_id for row in self.e0_task_native_descriptors})
            != len(self.e0_task_native_descriptors)
        ):
            raise ValueError("trusted E0 descriptor state differs")
        expected_runtime_status: TrustedRuntimeBindingStatus = (
            "PENDING_REMOTE_BINDING" if self.runtime_observations is None else "BOUND"
        )
        if self.runtime_binding_status != expected_runtime_status:
            raise ValueError("trusted runtime binding status differs")
        if (
            self.runtime_observations is not None
            and type(self.runtime_observations) is not TrustedRuntimeObservations
        ):
            raise TypeError("trusted runtime observation type differs")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "trust_mode": self.trust_mode,
            "signature": self.signature,
            "formal_measured_authorization": self.formal_measured_authorization,
            "claim_scope": self.claim_scope,
            "source_snapshot": self.source_snapshot.to_dict(),
            "model_members": [row.to_dict() for row in self.model_members],
            "locked_workloads": [row.to_dict() for row in self.locked_workloads],
            "burstgpt_release": self.burstgpt_release.to_dict(),
            "e0_task_native_descriptor_status": (self.e0_task_native_descriptor_status),
            "e0_task_native_descriptors": [
                row.to_dict() for row in self.e0_task_native_descriptors
            ],
            "runtime_binding_status": self.runtime_binding_status,
            "runtime_observations": (
                None
                if self.runtime_observations is None
                else self.runtime_observations.to_dict()
            ),
        }

    @cached_property
    def semantic_sha256(self) -> str:
        return content_sha256(self._payload())

    @property
    def protocol_lock_content_sha256(self) -> str:
        """Digest a trusted ProtocolLock/execution source may bind explicitly."""

        return self.semantic_sha256

    def to_dict(self) -> dict[str, object]:
        return {"semantic_sha256": self.semantic_sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "trusted single-operator content bundle",
            value,
            set(cls.__dataclass_fields__) | {"semantic_sha256"},
        )
        declared = _require_sha256(
            "trusted single-operator content bundle",
            row.pop("semantic_sha256"),
        )
        source = row.pop("source_snapshot")
        models = _strict_list("trusted model members", row.pop("model_members"))
        workloads = _strict_list(
            "trusted locked workloads", row.pop("locked_workloads")
        )
        burstgpt = row.pop("burstgpt_release")
        e0 = _strict_list(
            "trusted E0 task-native descriptors",
            row.pop("e0_task_native_descriptors"),
        )
        runtime = row.pop("runtime_observations")
        bundle = cls(
            **row,
            source_snapshot=TrustedSourceSnapshot.from_dict(source),
            model_members=tuple(
                TrustedModelSnapshotMember.from_dict(item) for item in models
            ),
            locked_workloads=tuple(
                TrustedLockedWorkload.from_dict(item) for item in workloads
            ),
            burstgpt_release=TrustedBurstGptRelease.from_dict(burstgpt),
            e0_task_native_descriptors=tuple(
                TrustedE0TaskNativeDescriptor.from_dict(item) for item in e0
            ),
            runtime_observations=(
                None
                if runtime is None
                else TrustedRuntimeObservations.from_dict(runtime)
            ),
        )  # type: ignore[arg-type]
        if bundle.semantic_sha256 != declared:
            raise ValueError("trusted single-operator content bundle digest differs")
        return bundle


def build_trusted_single_operator_content_bundle(
    *,
    repository_root: str | Path,
    model_specs: Sequence[TrustedModelSnapshotSpec],
    livecodebench_raw_path: str | Path,
    math500_raw_path: str | Path,
    burstgpt_asset_paths: Mapping[str, str | Path],
    e0_task_native_specs: Sequence[TrustedE0TaskNativeDescriptorSpec] = (),
) -> TrustedSingleOperatorContentBundle:
    if isinstance(model_specs, (str, bytes)) or not isinstance(model_specs, Sequence):
        raise TypeError("trusted model specs must be a sequence")
    if isinstance(e0_task_native_specs, (str, bytes)) or not isinstance(
        e0_task_native_specs,
        Sequence,
    ):
        raise TypeError("trusted E0 descriptor specs must be a sequence")
    models = tuple(
        sorted(
            (bind_trusted_model_snapshot_member(spec) for spec in model_specs),
            key=lambda row: (
                row.role,
                row.model_id,
                row.revision,
                row.local_snapshot_path,
            ),
        )
    )
    e0 = tuple(
        sorted(
            (
                bind_trusted_e0_task_native_descriptor(spec)
                for spec in e0_task_native_specs
            ),
            key=lambda row: row.descriptor_id,
        )
    )
    return TrustedSingleOperatorContentBundle(
        schema_version=1,
        kind="trusted_single_operator_content_bundle",
        trust_mode="trusted_single_operator_no_signature",
        signature=None,
        formal_measured_authorization=False,
        claim_scope="trusted_single_operator_empirical_content_provenance",
        source_snapshot=bind_trusted_source_snapshot(repository_root),
        model_members=models,
        locked_workloads=(
            bind_trusted_locked_workload(
                "livecodebench_v6_hard",
                livecodebench_raw_path,
            ),
            bind_trusted_locked_workload("math500_level5", math500_raw_path),
        ),
        burstgpt_release=bind_trusted_burstgpt_release(burstgpt_asset_paths),
        e0_task_native_descriptor_status=(
            "NOT_PROVIDED" if not e0 else "PATH_BOUND_NO_COMPLETENESS_CLAIM"
        ),
        e0_task_native_descriptors=e0,
        runtime_binding_status="PENDING_REMOTE_BINDING",
        runtime_observations=None,
    )


def bind_trusted_single_operator_runtime_observations(
    bundle: TrustedSingleOperatorContentBundle,
    *,
    inventory_path: str | Path,
    doctor_path: str | Path,
) -> TrustedSingleOperatorContentBundle:
    if type(bundle) is not TrustedSingleOperatorContentBundle:
        raise TypeError("trusted runtime binder requires an exact content bundle")
    if bundle.runtime_observations is not None:
        raise ValueError("trusted runtime observations are already bound")
    observations = TrustedRuntimeObservations(
        inventory=bind_trusted_json_artifact("remote_gpu_inventory", inventory_path),
        doctor=bind_trusted_json_artifact("remote_runtime_doctor", doctor_path),
    )
    return replace(
        bundle,
        runtime_binding_status="BOUND",
        runtime_observations=observations,
    )


def _rebind_model(member: TrustedModelSnapshotMember) -> TrustedModelSnapshotMember:
    return bind_trusted_model_snapshot_member(
        TrustedModelSnapshotSpec(
            model_id=member.model_id,
            revision=member.revision,
            role=member.role,
            stages=member.stages,
            local_snapshot_path=member.local_snapshot_path,
            storage_mode=member.storage_mode,
            content_cache_root=member.content_cache_root,
            runtime_bindings=member.runtime_bindings,
        )
    )


def revalidate_trusted_model_snapshot_member(
    member: TrustedModelSnapshotMember,
) -> TrustedModelSnapshotMember:
    """Reopen one model member and reject any path, link, or blob mutation."""

    if type(member) is not TrustedModelSnapshotMember:
        raise TypeError("trusted model revalidator requires an exact member")
    rebound = _rebind_model(member)
    if rebound != member:
        raise RuntimeError("trusted model snapshot member changed")
    return rebound


def revalidate_trusted_single_operator_content_bundle(
    bundle: TrustedSingleOperatorContentBundle,
) -> TrustedSingleOperatorContentBundle:
    if type(bundle) is not TrustedSingleOperatorContentBundle:
        raise TypeError("trusted content revalidator requires an exact bundle")
    source = bind_trusted_source_snapshot(bundle.source_snapshot.repository_root)
    models = tuple(
        revalidate_trusted_model_snapshot_member(member)
        for member in bundle.model_members
    )
    workloads = tuple(
        bind_trusted_locked_workload(row.workload_id, row.raw_source_path)
        for row in bundle.locked_workloads
    )
    burstgpt = bind_trusted_burstgpt_release(
        {row.name: row.absolute_path for row in bundle.burstgpt_release.assets}
    )
    e0 = tuple(
        bind_trusted_e0_task_native_descriptor(
            TrustedE0TaskNativeDescriptorSpec(
                descriptor_id=row.descriptor_id,
                task=row.task,
                repository=row.repository,
                revision=row.revision,
                descriptor_path=row.source.absolute_path,
            )
        )
        for row in bundle.e0_task_native_descriptors
    )
    runtime = bundle.runtime_observations
    rebound_runtime = (
        None
        if runtime is None
        else TrustedRuntimeObservations(
            inventory=revalidate_trusted_json_artifact(runtime.inventory),
            doctor=revalidate_trusted_json_artifact(runtime.doctor),
        )
    )
    rebound = replace(
        bundle,
        source_snapshot=source,
        model_members=models,
        locked_workloads=workloads,
        burstgpt_release=burstgpt,
        e0_task_native_descriptors=e0,
        runtime_observations=rebound_runtime,
    )
    if rebound != bundle or rebound.semantic_sha256 != bundle.semantic_sha256:
        raise RuntimeError(
            "trusted single-operator content changed during revalidation"
        )
    return rebound


def _atomic_publish_no_replace(path: Path, body: bytes) -> None:
    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise ValueError("trusted content output path must be absolute and resolved")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("trusted content output parent must be a regular directory")
    if path.exists() or path.is_symlink():
        raise FileExistsError("trusted content output already exists")
    temporary = parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def publish_trusted_single_operator_content_bundle(
    bundle: TrustedSingleOperatorContentBundle,
    output_path: str | Path,
) -> Path:
    validated = revalidate_trusted_single_operator_content_bundle(bundle)
    output = Path(output_path)
    try:
        output.relative_to(Path(validated.source_snapshot.repository_root))
    except ValueError:
        pass
    else:
        raise ValueError("trusted content bundle must be published outside source Git")
    body = _canonical_bytes(validated.to_dict()) + b"\n"
    _atomic_publish_no_replace(output, body)
    loaded = load_trusted_single_operator_content_bundle(output)
    if loaded != validated:
        raise RuntimeError("published trusted content bundle differs")
    return output


def load_trusted_single_operator_content_bundle(
    path: str | Path,
) -> TrustedSingleOperatorContentBundle:
    source, raw = _stable_file_bytes(
        path,
        label="trusted single-operator content bundle",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    value = _strict_json(raw, label="trusted single-operator content bundle")
    bundle = TrustedSingleOperatorContentBundle.from_dict(value)
    if raw != _canonical_bytes(bundle.to_dict()) + b"\n":
        raise ValueError("trusted single-operator content bundle is not canonical JSON")
    if source != Path(path):
        raise ValueError("trusted single-operator content bundle path differs")
    return bundle


@dataclass(frozen=True)
class TrustedSingleOperatorContentBundleBinding:
    absolute_path: str
    size: int
    raw_sha256: str
    semantic_sha256: str
    runtime_binding_status: TrustedRuntimeBindingStatus

    def __post_init__(self) -> None:
        _resolved_file(self.absolute_path, label="trusted content bundle binding")
        _require_positive_int("trusted content bundle size", self.size)
        _require_sha256("trusted content bundle raw", self.raw_sha256)
        _require_sha256("trusted content bundle semantic", self.semantic_sha256)
        if self.runtime_binding_status not in {"PENDING_REMOTE_BINDING", "BOUND"}:
            raise ValueError("trusted content bundle runtime status differs")

    @classmethod
    def bind(cls, path: str | Path) -> Self:
        bundle = load_trusted_single_operator_content_bundle(path)
        revalidate_trusted_single_operator_content_bundle(bundle)
        source = Path(path)
        size, raw_sha256 = _stable_file_digest(
            source,
            label="trusted content bundle binding",
        )
        return cls(
            absolute_path=str(source),
            size=size,
            raw_sha256=raw_sha256,
            semantic_sha256=bundle.semantic_sha256,
            runtime_binding_status=bundle.runtime_binding_status,
        )

    def reopen(self) -> TrustedSingleOperatorContentBundle:
        size, raw_sha256 = _stable_file_digest(
            Path(self.absolute_path),
            label="trusted content bundle binding",
        )
        bundle = load_trusted_single_operator_content_bundle(self.absolute_path)
        if (
            size != self.size
            or raw_sha256 != self.raw_sha256
            or bundle.semantic_sha256 != self.semantic_sha256
            or bundle.runtime_binding_status != self.runtime_binding_status
        ):
            raise RuntimeError("trusted content bundle binding changed")
        return revalidate_trusted_single_operator_content_bundle(bundle)

    def to_dict(self) -> dict[str, object]:
        return {
            "absolute_path": self.absolute_path,
            "size": self.size,
            "raw_sha256": self.raw_sha256,
            "semantic_sha256": self.semantic_sha256,
            "runtime_binding_status": self.runtime_binding_status,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(**_from_fields("trusted content bundle binding", cls, value))  # type: ignore[arg-type]


def load_trusted_single_operator_content_path_spec(
    path: str | Path,
) -> TrustedSingleOperatorContentPathSpec:
    """Deep-open one canonical path recipe without accepting derived digests."""

    source, raw = _stable_file_bytes(
        path,
        label="trusted single-operator content path spec",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    value = _strict_json(raw, label="trusted single-operator content path spec")
    spec = TrustedSingleOperatorContentPathSpec.from_dict(value)
    if raw != _canonical_bytes(spec.to_dict()) + b"\n":
        raise ValueError("trusted content path spec is not canonical JSON")
    if source != Path(path):
        raise ValueError("trusted content path spec path differs")
    return spec


def publish_runtime_bound_trusted_single_operator_content_from_spec(
    *,
    spec_path: str | Path,
    output_path: str | Path,
) -> TrustedSingleOperatorContentBundleBinding:
    """Build, runtime-bind, and no-replace publish from one path-only spec."""

    spec = load_trusted_single_operator_content_path_spec(spec_path)
    pending = build_trusted_single_operator_content_bundle(
        repository_root=spec.repository_root,
        model_specs=spec.model_specs,
        livecodebench_raw_path=spec.livecodebench_raw_path,
        math500_raw_path=spec.math500_raw_path,
        burstgpt_asset_paths={
            row.name: row.absolute_path for row in spec.burstgpt_asset_paths
        },
        e0_task_native_specs=spec.e0_task_native_specs,
    )
    bound = bind_trusted_single_operator_runtime_observations(
        pending,
        inventory_path=spec.inventory_path,
        doctor_path=spec.doctor_path,
    )
    publish_trusted_single_operator_content_bundle(bound, output_path)
    binding = TrustedSingleOperatorContentBundleBinding.bind(output_path)
    if (
        binding.runtime_binding_status != "BOUND"
        or binding.semantic_sha256 != bound.semantic_sha256
        or load_trusted_single_operator_content_path_spec(spec_path) != spec
    ):
        raise RuntimeError("trusted content path publication changed")
    return binding


def publish_trusted_preflight_workload_authority_from_content(
    *,
    trusted_content_bundle_path: str | Path,
    output_path: str | Path,
) -> object:
    """Publish the locked LiveCodeBench authority from the trusted bundle.

    The caller supplies no workload ID, revision, filter, row count, or digest.
    Those values are replayed by the existing workload authority binder and
    compared with the content bundle's independently frozen member.
    """

    from lightcone_spec.experiments.workload_authority import (
        bind_formal_workload_authority,
        formal_workload_authority_cli_artifact,
        formal_workload_authority_from_cli_artifact,
        revalidate_formal_workload_authority,
    )
    from lightcone_spec.runtime.proof_artifact import (
        CanonicalJsonProofBinding,
        publish_canonical_json_no_replace,
    )

    content = TrustedSingleOperatorContentBundleBinding.bind(
        trusted_content_bundle_path
    )
    if content.runtime_binding_status != "BOUND":
        raise ValueError("trusted preflight workload requires BOUND content")
    bundle = content.reopen()
    matches = tuple(
        row
        for row in bundle.locked_workloads
        if row.workload_id == "livecodebench_v6_hard"
    )
    if len(matches) != 1:
        raise ValueError("trusted content lacks one LiveCodeBench hard workload")
    locked = matches[0]
    authority = bind_formal_workload_authority(
        "livecodebench_v6_hard",
        locked.raw_source_path,
    )
    if (
        authority.sha256 != locked.authority_sha256
        or authority.raw_file_sha256 != locked.raw_file_sha256
        or authority.repository_revision != locked.repository_revision
        or authority.raw_row_count != locked.raw_row_count
        or authority.selected_row_count != locked.selected_row_count
        or authority.selected_rows_sha256 != locked.formal_samples_sha256
        or authority.source_lock_sha256 != locked.source_lock_sha256
        or authority.protocol_sha256 != locked.protocol_sha256
    ):
        raise ValueError("trusted preflight workload differs from content member")
    output = Path(output_path)
    try:
        output.relative_to(Path(bundle.source_snapshot.repository_root))
    except ValueError:
        pass
    else:
        raise ValueError("trusted workload authority must be published outside Git")
    # The downstream root-verifying reducer accepts only the public diagnostic
    # wrapper, not a bare authority object.  Its proof identity binds that
    # wrapper and is intentionally distinct from the nested authority identity.
    publish_canonical_json_no_replace(
        output,
        formal_workload_authority_cli_artifact(authority),
    )
    binding = CanonicalJsonProofBinding.bind(output)
    rebound = revalidate_formal_workload_authority(
        formal_workload_authority_from_cli_artifact(binding.reopen())
    )
    if rebound != authority:
        raise RuntimeError("trusted preflight workload publication changed")
    return binding


__all__ = [
    "TrustedBurstGptAssetPath",
    "TrustedBurstGptRelease",
    "TrustedContentFile",
    "TrustedE0TaskNativeDescriptor",
    "TrustedE0TaskNativeDescriptorSpec",
    "TrustedJsonArtifact",
    "TrustedLockedWorkload",
    "TrustedModelRuntimeBinding",
    "TrustedModelSnapshotMember",
    "TrustedModelSnapshotSpec",
    "TrustedNamedInputPath",
    "TrustedRuntimeObservations",
    "TrustedSglangPatch",
    "TrustedSingleOperatorContentBundle",
    "TrustedSingleOperatorContentBundleBinding",
    "TrustedSingleOperatorContentPathSpec",
    "TrustedSnapshotStorageMode",
    "TrustedSourceSnapshot",
    "bind_trusted_burstgpt_release",
    "bind_trusted_e0_task_native_descriptor",
    "bind_trusted_json_artifact",
    "bind_trusted_locked_workload",
    "bind_trusted_model_snapshot_member",
    "bind_trusted_single_operator_runtime_observations",
    "bind_trusted_source_snapshot",
    "build_trusted_single_operator_content_bundle",
    "load_trusted_single_operator_content_bundle",
    "load_trusted_single_operator_content_path_spec",
    "publish_runtime_bound_trusted_single_operator_content_from_spec",
    "publish_trusted_preflight_workload_authority_from_content",
    "publish_trusted_single_operator_content_bundle",
    "revalidate_trusted_json_artifact",
    "revalidate_trusted_model_snapshot_member",
    "revalidate_trusted_single_operator_content_bundle",
]
