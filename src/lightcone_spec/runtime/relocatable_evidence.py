"""Relocatable, immutable bundles for pulled first-party JSON evidence.

GPU hosts publish path-bound canonical JSON.  A verifier normally runs on a
different filesystem, so a remote absolute path cannot be treated as a local
authority.  This module copies the transitive closure of canonical proof
bindings into a private local root and records an exact relative-path manifest.
Validation then installs a narrowly scoped reopen mapping: serialized bindings
retain their original remote identity and digests, while bytes are read from
the verified local member.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
    relocated_evidence_path,
    use_canonical_json_relocation,
)

RELOCATABLE_EVIDENCE_BUNDLE_FILENAME = ".relocatable-evidence-bundle.json"
RELOCATABLE_EVIDENCE_BUNDLE_PROTOCOL_SHA256 = hashlib.sha256(
    b"lightcone-relocatable-closed-binding-evidence-bundle-v3"
).hexdigest()
MAXIMUM_RELOCATABLE_EVIDENCE_MEMBER_BYTES = 2 * 1024 * 1024
_CANONICAL_BINDING_FIELDS = {
    "absolute_path",
    "raw_sha256",
    "semantic_sha256",
    "size",
}
_RAW_BINDING_FIELDS = {"absolute_path", "raw_sha256", "size"}
_CONTENT_BINDING_FIELDS = {
    "artifact_id",
    "path",
    "raw_sha256",
    "semantic_sha256",
    "size",
}
_CAPACITY_BINDING_FIELDS = {
    "schema_version",
    "path",
    "sidecar_path",
    "semantic_sha256",
    "file_sha256",
    "sidecar_file_sha256",
    "size",
    "sidecar_size",
}
_SIDECAR_ROLE_BINDING_FIELDS = _CAPACITY_BINDING_FIELDS | {"role"}
_SIDECAR_KIND_BINDING_FIELDS = _CAPACITY_BINDING_FIELDS | {"kind"}
_BOUND_JSON_FIELDS = {
    "path",
    "canonical_sha256",
    "semantic_sha256",
    "file_sha256",
    "sidecar_file_sha256",
    "size",
}
_BUDGET_RAW_JSON_DESCRIPTOR_FIELDS = _SIDECAR_ROLE_BINDING_FIELDS | {"canonical_sha256"}
_REPLAY_BINDING_FIELDS = {
    "schema_version",
    "kind",
    "path",
    "reservation_sha256",
    "raw_sha256",
    "size",
    "reserved_ns",
    "challenge_sha256s",
}
_DATASET_MEMBER_PATH_FIELDS = {
    "member_id",
    "raw_path",
    "selected_rows_path",
    "request_shape_path",
}
_PREPARED_MODEL_SNAPSHOT_FIELDS = {"model_id", "revision", "root"}
_KNOWN_NONDEPENDENCY_PATH_FIELDS = {
    "private_output_root",
    "terminal_output_path",
    "native_itl_pointer_output_path",
    "live_run_receipt_output_path",
    "lifecycle_timing_output_path",
    "server_log_output_path",
    "before_gpu_snapshot_output_path",
    "ready_gpu_snapshot_output_path",
    "after_gpu_snapshot_output_path",
    "formal_gang_terminal_output_path",
    "fatal_output_path",
    "proof_entry_remote_absolute_path",
}
_FORBIDDEN_SECRET_FIELDS = {
    "access_token",
    "api_token",
    "password",
    "private_key",
    "private_key_b64",
    "private_key_path",
    "secret",
    "signing_key",
}
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"\AeyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\Z"),
    re.compile(r"\Ahf_[A-Za-z0-9]{20,}\Z"),
    re.compile(r"\Ask-[A-Za-z0-9_-]{20,}\Z"),
    re.compile(r"\AAKIA[A-Z0-9]{16}\Z"),
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


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _absolute(label: str, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ValueError(f"{label} must be absolute and normalized")
    return path


def _private_directory(label: str, value: str | Path) -> Path:
    path = _absolute(label, value)
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be an existing symlink-free directory")
    status = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise ValueError(f"{label} must be private and current-user-owned")
    return path


def _stable_raw_member(path: Path, *, label: str) -> bytes:
    """Read one bounded immutable member without following its final link."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o022
            or not 1 <= before.st_size <= MAXIMUM_RELOCATABLE_EVIDENCE_MEMBER_BYTES
        ):
            raise ValueError(f"{label} is not one bounded immutable regular file")
        body = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (
            len(body) != before.st_size
            or identity
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or identity
            != (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            )
        ):
            raise RuntimeError(f"{label} changed while read")
        return body
    finally:
        os.close(descriptor)


def _stable_directory_file(path: Path, *, label: str) -> tuple[str, int]:
    """Hash one arbitrarily sized rehydrated asset without trusting its name."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is not a readable regular file") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_size < 0
        ):
            raise ValueError(f"{label} is unsafe")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or identity
            != (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_nlink,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            or size != before.st_size
        ):
            raise RuntimeError(f"{label} changed while it was hashed")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _scan_directory_rebind(
    remote: str | Path,
    local: str | Path,
) -> RelocatableDirectoryRebind:
    remote_root = _absolute("relocatable directory remote root", remote)
    local_root = _absolute("relocatable directory local root", local)
    try:
        resolved = local_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("relocatable directory local root is unavailable") from error
    if resolved != local_root or local_root.is_symlink() or not local_root.is_dir():
        raise ValueError("relocatable directory local root is not symlink-free")
    rows: list[RelocatableDirectoryMember] = []
    for current, directories, files in os.walk(
        local_root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        current_status = os.lstat(current_path)
        if (
            not stat.S_ISDIR(current_status.st_mode)
            or stat.S_ISLNK(current_status.st_mode)
            or current_status.st_uid != os.geteuid()
            or stat.S_IMODE(current_status.st_mode) & 0o022
        ):
            raise ValueError("relocatable directory contains an unsafe directory")
        directories.sort()
        files.sort()
        for directory in directories:
            candidate = current_path / directory
            status = os.lstat(candidate)
            if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
                raise ValueError("relocatable directory contains a linked directory")
        for filename in files:
            candidate = current_path / filename
            relative = candidate.relative_to(local_root).as_posix()
            _validate_relative(relative)
            raw_sha256, size = _stable_directory_file(
                candidate,
                label="relocatable directory member",
            )
            rows.append(
                RelocatableDirectoryMember(
                    relative_path=relative,
                    raw_sha256=raw_sha256,
                    size=size,
                )
            )
    members = tuple(sorted(rows, key=lambda row: row.relative_path))
    if not members:
        raise ValueError("relocatable directory root is empty")
    tree_sha256 = _canonical_sha256(
        {
            "remote_absolute_path": str(remote_root),
            "local_absolute_path": str(local_root),
            "members": tuple(row.to_dict() for row in members),
        }
    )
    return RelocatableDirectoryRebind(
        remote_absolute_path=str(remote_root),
        local_absolute_path=str(local_root),
        members=members,
        tree_sha256=tree_sha256,
    )


def _publish_raw_no_replace(path: Path, body: bytes) -> None:
    if not 1 <= len(body) <= MAXIMUM_RELOCATABLE_EVIDENCE_MEMBER_BYTES:
        raise ValueError("relocatable evidence raw member exceeds portability cap")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive OS boundary
                raise OSError("relocatable evidence raw publication made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_value(body: bytes) -> object | None:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _scan_secret_text(body: bytes) -> None:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return
    if any(pattern.search(text) is not None for pattern in _SECRET_VALUE_PATTERNS):
        raise ValueError("relocatable evidence contains secret-shaped material")


def _relative_member(remote_root: Path, remote_path: Path) -> str:
    try:
        relative = remote_path.relative_to(remote_root)
    except ValueError as error:
        raise ValueError("relocatable evidence member escapes remote root") from error
    pure = PurePosixPath(relative.as_posix())
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() == RELOCATABLE_EVIDENCE_BUNDLE_FILENAME
    ):
        raise ValueError("relocatable evidence relative member is unsafe")
    return pure.as_posix()


def _safe_destination(root: Path, relative_path: str) -> Path:
    pure = _validate_relative(relative_path)
    destination = root.joinpath(*pure.parts)
    if destination != Path(os.path.abspath(destination)):
        raise ValueError("relocatable evidence destination is not normalized")
    current = root
    for component in pure.parts[:-1]:
        current = current / component
        if current.exists():
            status = current.lstat()
            if not stat.S_ISDIR(status.st_mode) or current.is_symlink():
                raise ValueError("relocatable evidence ancestor is unsafe")
        else:
            current.mkdir(mode=0o700)
    return destination


def _existing_safe_member(root: Path, relative_path: str) -> Path:
    pure = _validate_relative(relative_path)
    current = root
    for component in pure.parts[:-1]:
        current = current / component
        try:
            status = current.lstat()
        except FileNotFoundError as error:
            raise ValueError("relocatable evidence ancestor is missing") from error
        if not stat.S_ISDIR(status.st_mode) or current.is_symlink():
            raise ValueError("relocatable evidence ancestor is unsafe")
    return root.joinpath(*pure.parts)


def _validate_relative(relative_path: str) -> PurePosixPath:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("relocatable evidence member path traverses bundle root")
    return pure


@dataclass(frozen=True)
class _DiscoveredFile:
    remote_absolute_path: str
    raw_sha256: str | None
    semantic_sha256: str | None
    size: int | None
    binding_kind: str


def _declared_file(
    *,
    path: object,
    raw_sha256: object | None,
    semantic_sha256: object | None,
    size: object | None,
    binding_kind: str,
) -> _DiscoveredFile:
    if type(path) is not str:
        raise TypeError("relocatable evidence binding path must be text")
    if raw_sha256 is not None:
        _sha256("relocatable evidence declared raw digest", raw_sha256)
    if semantic_sha256 is not None:
        _sha256("relocatable evidence declared semantic digest", semantic_sha256)
    if size is not None and (type(size) is not int or size < 1):
        raise ValueError("relocatable evidence declared size is invalid")
    return _DiscoveredFile(
        remote_absolute_path=str(_absolute("relocatable evidence binding", path)),
        raw_sha256=raw_sha256,
        semantic_sha256=semantic_sha256,
        size=size,
        binding_kind=binding_kind,
    )


def _binding_files(value: object) -> Iterator[_DiscoveredFile]:
    if type(value) is dict:
        fields = set(value)
        if value.get("kind") == "lightcone_relocatable_evidence_bundle":
            RelocatableEvidenceBundle.from_dict(value)
            return
        if fields == _CANONICAL_BINDING_FIELDS:
            binding = CanonicalJsonProofBinding.from_dict(value)
            yield _declared_file(
                path=binding.absolute_path,
                raw_sha256=binding.raw_sha256,
                semantic_sha256=binding.semantic_sha256,
                size=binding.size,
                binding_kind="canonical_json",
            )
            return
        if fields == _RAW_BINDING_FIELDS:
            yield _declared_file(
                path=value["absolute_path"],
                raw_sha256=value["raw_sha256"],
                semantic_sha256=None,
                size=value["size"],
                binding_kind="raw_file",
            )
            return
        if fields == _CONTENT_BINDING_FIELDS:
            yield _declared_file(
                path=value["path"],
                raw_sha256=value["raw_sha256"],
                semantic_sha256=value["semantic_sha256"],
                size=value["size"],
                binding_kind="content_json",
            )
            return
        if fields == _CAPACITY_BINDING_FIELDS:
            yield _declared_file(
                path=value["path"],
                raw_sha256=value["file_sha256"],
                semantic_sha256=value["semantic_sha256"],
                size=value["size"],
                binding_kind="capacity_json",
            )
            yield _declared_file(
                path=value["sidecar_path"],
                raw_sha256=value["sidecar_file_sha256"],
                semantic_sha256=None,
                size=value["sidecar_size"],
                binding_kind="capacity_sidecar",
            )
            return
        if frozenset(fields) in {
            frozenset(_SIDECAR_ROLE_BINDING_FIELDS),
            frozenset(_SIDECAR_KIND_BINDING_FIELDS),
        }:
            yield _declared_file(
                path=value["path"],
                raw_sha256=value["file_sha256"],
                semantic_sha256=value["semantic_sha256"],
                size=value["size"],
                binding_kind="sidecar_bound_json",
            )
            yield _declared_file(
                path=value["sidecar_path"],
                raw_sha256=value["sidecar_file_sha256"],
                semantic_sha256=None,
                size=value["sidecar_size"],
                binding_kind="sidecar_bound_json_sidecar",
            )
            return
        if fields == _BOUND_JSON_FIELDS:
            source = str(_absolute("relocatable bound JSON", value["path"]))
            yield _declared_file(
                path=source,
                raw_sha256=value["file_sha256"],
                semantic_sha256=value["semantic_sha256"],
                size=value["size"],
                binding_kind="bound_json",
            )
            yield _declared_file(
                path=f"{source}.sha256",
                raw_sha256=value["sidecar_file_sha256"],
                semantic_sha256=None,
                size=65,
                binding_kind="bound_json_sidecar",
            )
            return
        if fields == _BUDGET_RAW_JSON_DESCRIPTOR_FIELDS:
            source = _absolute("relocatable budget raw JSON", value["path"])
            sidecar = _absolute(
                "relocatable budget raw JSON sidecar", value["sidecar_path"]
            )
            if sidecar != Path(f"{source}.sha256"):
                raise ValueError("relocatable budget raw JSON sidecar differs")
            for label in (
                "canonical_sha256",
                "semantic_sha256",
                "file_sha256",
                "sidecar_file_sha256",
            ):
                _sha256(f"relocatable budget raw JSON {label}", value[label])
            if (
                type(value["size"]) is not int
                or value["size"] < 1
                or type(value["sidecar_size"]) is not int
                or value["sidecar_size"] != 65
            ):
                raise ValueError("relocatable budget raw JSON size differs")
            # This exact descriptor is embedded provenance in the source-owned
            # trainable-plan authority.  The plan reducer does not reopen it;
            # transport authority comes from the enclosing canonical plan.
            return
        if fields == _REPLAY_BINDING_FIELDS and value.get("kind") == (
            "lightcone_challenge_replay_reservation_binding"
        ):
            yield _declared_file(
                path=value["path"],
                raw_sha256=value["raw_sha256"],
                semantic_sha256=value["reservation_sha256"],
                size=value["size"],
                binding_kind="challenge_replay_reservation",
            )
            return
        if fields == _DATASET_MEMBER_PATH_FIELDS:
            for name in ("raw_path", "selected_rows_path", "request_shape_path"):
                yield _declared_file(
                    path=value[name],
                    raw_sha256=None,
                    semantic_sha256=None,
                    size=None,
                    binding_kind=f"dataset_{name}",
                )
            return
        if fields == _PREPARED_MODEL_SNAPSHOT_FIELDS:
            _absolute("relocatable prepared-model snapshot root", value["root"])
            return
        for key, child in value.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if (
                normalized_key in _FORBIDDEN_SECRET_FIELDS
                or normalized_key.startswith(("private_key_", "signing_key_"))
                or normalized_key.endswith(
                    ("_password", "_secret", "_access_token", "_api_token")
                )
            ):
                raise ValueError(
                    "relocatable evidence contains a forbidden secret field"
                )
            if (
                normalized_key not in _KNOWN_NONDEPENDENCY_PATH_FIELDS
                and (
                    normalized_key in {"path", "absolute_path", "journal_path"}
                    or normalized_key.endswith(("_path", "_root"))
                )
                and type(child) is str
                and Path(child).is_absolute()
            ):
                raise ValueError(
                    "relocatable evidence contains an unknown path-bearing field"
                )
            yield from _binding_files(child)
    elif type(value) is list:
        for child in value:
            yield from _binding_files(child)
    elif type(value) is str and any(
        pattern.search(value) is not None for pattern in _SECRET_VALUE_PATTERNS
    ):
        raise ValueError("relocatable evidence contains secret-shaped material")


def _metadata_directory_identities(value: object) -> Iterator[str]:
    if type(value) is dict:
        if set(value) == _PREPARED_MODEL_SNAPSHOT_FIELDS:
            yield str(
                _absolute("relocatable prepared-model snapshot root", value["root"])
            )
            return
        for child in value.values():
            yield from _metadata_directory_identities(child)
    elif type(value) is list:
        for child in value:
            yield from _metadata_directory_identities(child)


@dataclass(frozen=True)
class RelocatableEvidenceMember:
    remote_absolute_path: str
    relative_path: str
    raw_sha256: str
    semantic_sha256: str | None
    size: int
    binding_kind: str = "canonical_json"

    def __post_init__(self) -> None:
        _absolute("relocatable evidence remote member", self.remote_absolute_path)
        _sha256("relocatable evidence raw digest", self.raw_sha256)
        if self.semantic_sha256 is not None:
            _sha256("relocatable evidence semantic digest", self.semantic_sha256)
        _validate_relative(self.relative_path)
        if (
            type(self.binding_kind) is not str
            or not self.binding_kind
            or type(self.size) is not int
            or not 1 <= self.size <= MAXIMUM_RELOCATABLE_EVIDENCE_MEMBER_BYTES
        ):
            raise ValueError("relocatable evidence member size is invalid")

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("relocatable evidence member fields differ")
        return cls(**value)


@dataclass(frozen=True)
class RelocatableDirectoryMember:
    """One immutable regular file in an offline-rehydrated snapshot root."""

    relative_path: str
    raw_sha256: str
    size: int

    def __post_init__(self) -> None:
        _validate_relative(self.relative_path)
        _sha256("relocatable directory member digest", self.raw_sha256)
        if type(self.size) is not int or self.size < 0:
            raise ValueError("relocatable directory member size is invalid")

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("relocatable directory member fields differ")
        return cls(**value)


@dataclass(frozen=True)
class RelocatableDirectoryRebind:
    """Exact B-side filesystem tree replacing one A-side snapshot identity."""

    remote_absolute_path: str
    local_absolute_path: str
    members: tuple[RelocatableDirectoryMember, ...]
    tree_sha256: str

    def __post_init__(self) -> None:
        remote = _absolute(
            "relocatable directory remote root", self.remote_absolute_path
        )
        local = _absolute("relocatable directory local root", self.local_absolute_path)
        if remote == local:
            raise ValueError("relocatable directory roots must differ")
        if (
            type(self.members) is not tuple
            or not self.members
            or self.members
            != tuple(sorted(self.members, key=lambda row: row.relative_path))
            or len({row.relative_path for row in self.members}) != len(self.members)
        ):
            raise ValueError("relocatable directory members are not canonical")
        expected = _canonical_sha256(
            {
                "remote_absolute_path": str(remote),
                "local_absolute_path": str(local),
                "members": tuple(row.to_dict() for row in self.members),
            }
        )
        if self.tree_sha256 != expected:
            raise ValueError("relocatable directory tree identity differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "remote_absolute_path": self.remote_absolute_path,
            "local_absolute_path": self.local_absolute_path,
            "members": [row.to_dict() for row in self.members],
            "tree_sha256": self.tree_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("relocatable directory rebind fields differ")
        row: dict[str, Any] = dict(value)
        members = row["members"]
        if type(members) is not list:
            raise TypeError("relocatable directory members must be an array")
        row["members"] = tuple(
            RelocatableDirectoryMember.from_dict(member) for member in members
        )
        return cls(**row)


@dataclass(frozen=True)
class RelocatableEvidenceBundle:
    schema_version: Literal[3]
    kind: Literal["lightcone_relocatable_evidence_bundle"]
    protocol_sha256: str
    remote_root: str
    remote_root_sha256: str
    entry_remote_absolute_paths: tuple[str, ...]
    members: tuple[RelocatableEvidenceMember, ...]
    directory_rebindings: tuple[RelocatableDirectoryRebind, ...]
    bundle_root_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 3
            or self.kind != "lightcone_relocatable_evidence_bundle"
            or self.protocol_sha256 != RELOCATABLE_EVIDENCE_BUNDLE_PROTOCOL_SHA256
        ):
            raise ValueError("relocatable evidence bundle schema differs")
        root = _absolute("relocatable evidence remote root", self.remote_root)
        if self.remote_root_sha256 != _canonical_sha256(str(root)):
            raise ValueError("relocatable evidence remote-root identity differs")
        if (
            not self.members
            or self.members
            != tuple(sorted(self.members, key=lambda row: row.relative_path))
            or len({row.relative_path for row in self.members}) != len(self.members)
            or len({row.remote_absolute_path for row in self.members})
            != len(self.members)
        ):
            raise ValueError("relocatable evidence member set is not canonical")
        remote_members = {row.remote_absolute_path for row in self.members}
        for member in self.members:
            expected_remote = root.joinpath(*PurePosixPath(member.relative_path).parts)
            if str(expected_remote) != member.remote_absolute_path:
                raise ValueError("relocatable evidence member root binding differs")
        if (
            not self.entry_remote_absolute_paths
            or self.entry_remote_absolute_paths
            != tuple(sorted(set(self.entry_remote_absolute_paths)))
            or not set(self.entry_remote_absolute_paths) <= remote_members
        ):
            raise ValueError("relocatable evidence entry set differs")
        if (
            type(self.directory_rebindings) is not tuple
            or self.directory_rebindings
            != tuple(
                sorted(
                    self.directory_rebindings,
                    key=lambda row: row.remote_absolute_path,
                )
            )
            or len({row.remote_absolute_path for row in self.directory_rebindings})
            != len(self.directory_rebindings)
            or len({row.local_absolute_path for row in self.directory_rebindings})
            != len(self.directory_rebindings)
        ):
            raise ValueError("relocatable directory rebind set is not canonical")
        expected_root = _canonical_sha256(
            {
                "remote_root_sha256": self.remote_root_sha256,
                "entry_remote_absolute_paths": self.entry_remote_absolute_paths,
                "members": tuple(row.to_dict() for row in self.members),
                "directory_rebindings": tuple(
                    row.to_dict() for row in self.directory_rebindings
                ),
            }
        )
        if self.bundle_root_sha256 != expected_root:
            raise ValueError("relocatable evidence bundle-root binding differs")

    @cached_property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "remote_root": self.remote_root,
            "remote_root_sha256": self.remote_root_sha256,
            "entry_remote_absolute_paths": list(self.entry_remote_absolute_paths),
            "members": [row.to_dict() for row in self.members],
            "directory_rebindings": [
                row.to_dict() for row in self.directory_rebindings
            ],
            "bundle_root_sha256": self.bundle_root_sha256,
        }
        if include_digest:
            value["bundle_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "bundle_sha256",
        }:
            raise ValueError("relocatable evidence bundle fields differ")
        row: dict[str, Any] = dict(value)
        declared = _sha256("relocatable evidence bundle", row.pop("bundle_sha256"))
        entries = row["entry_remote_absolute_paths"]
        members = row["members"]
        rebindings = row["directory_rebindings"]
        if (
            type(entries) is not list
            or type(members) is not list
            or type(rebindings) is not list
        ):
            raise TypeError("relocatable evidence bundle arrays are invalid")
        row["entry_remote_absolute_paths"] = tuple(entries)
        row["members"] = tuple(
            RelocatableEvidenceMember.from_dict(member) for member in members
        )
        row["directory_rebindings"] = tuple(
            RelocatableDirectoryRebind.from_dict(rebinding) for rebinding in rebindings
        )
        artifact = cls(**row)
        if artifact.sha256 != declared:
            raise ValueError("relocatable evidence bundle digest differs")
        return artifact


@dataclass(frozen=True)
class ValidatedRelocatableEvidenceBundle:
    artifact: RelocatableEvidenceBundle
    manifest: CanonicalJsonProofBinding
    local_root: str
    relocation: dict[str, str]


def materialize_relocatable_evidence_bundle(
    *,
    remote_root: str | Path,
    entry_paths: tuple[str | Path, ...],
    local_root: str | Path,
    directory_rebindings: Mapping[str | Path, str | Path] | None = None,
) -> CanonicalJsonProofBinding:
    """Copy the exact transitive canonical-JSON closure from root A to root B."""

    source_root = _private_directory("relocatable evidence remote root", remote_root)
    destination_root = _private_directory("relocatable evidence local root", local_root)
    if source_root == destination_root:
        raise ValueError("relocatable evidence requires distinct remote/local roots")
    if any(destination_root.iterdir()):
        raise ValueError("relocatable evidence local root must start empty")
    queue = [str(_absolute("relocatable evidence entry", path)) for path in entry_paths]
    if not queue:
        raise ValueError("relocatable evidence requires at least one entry")
    entries = tuple(sorted(set(queue)))
    pending = [
        _DiscoveredFile(
            remote_absolute_path=entry,
            raw_sha256=None,
            semantic_sha256=None,
            size=None,
            binding_kind="entry",
        )
        for entry in entries
    ]
    discovered: dict[str, tuple[_DiscoveredFile, bytes]] = {}
    metadata_directories: set[str] = set()
    inherited_directory_rebindings: dict[str, str] = {}
    while pending:
        declaration = pending.pop()
        remote = declaration.remote_absolute_path
        if remote in discovered:
            previous, _body = discovered[remote]
            if any(
                expected is not None and expected != observed
                for expected, observed in (
                    (declaration.raw_sha256, previous.raw_sha256),
                    (declaration.semantic_sha256, previous.semantic_sha256),
                    (declaration.size, previous.size),
                )
            ):
                raise ValueError(
                    "relocatable evidence aliases one path with another identity"
                )
            continue
        remote_path = _absolute("relocatable evidence member", remote)
        _relative_member(source_root, remote_path)
        body = _stable_raw_member(remote_path, label="relocatable evidence member")
        raw_sha256 = hashlib.sha256(body).hexdigest()
        if declaration.raw_sha256 is not None and (
            raw_sha256 != declaration.raw_sha256
        ):
            raise ValueError("relocatable evidence declared raw digest differs")
        if declaration.size is not None and len(body) != declaration.size:
            raise ValueError("relocatable evidence declared size differs")
        value = _json_value(body)
        semantic_sha256 = declaration.semantic_sha256
        binding_kind = declaration.binding_kind
        if value is not None:
            _scan_secret_text(body)
            tuple(_binding_files(value))  # secret/path fail-closed scan first
            metadata_directories.update(_metadata_directory_identities(value))
            if type(value) is dict and value.get("kind") == (
                "lightcone_relocatable_evidence_bundle"
            ):
                nested_bundle = RelocatableEvidenceBundle.from_dict(value)
                nested_root = remote_path.parent
                for member in nested_bundle.members:
                    nested_path = nested_root.joinpath(
                        *PurePosixPath(member.relative_path).parts
                    )
                    pending.append(
                        _DiscoveredFile(
                            remote_absolute_path=str(nested_path),
                            raw_sha256=member.raw_sha256,
                            semantic_sha256=member.semantic_sha256,
                            size=member.size,
                            binding_kind=member.binding_kind,
                        )
                    )
                for rebound in nested_bundle.directory_rebindings:
                    previous = inherited_directory_rebindings.setdefault(
                        rebound.remote_absolute_path,
                        rebound.local_absolute_path,
                    )
                    if previous != rebound.local_absolute_path:
                        raise ValueError("nested relocatable directory rebind differs")
            if binding_kind == "entry":
                try:
                    canonical = CanonicalJsonProofBinding.bind(remote_path)
                except ValueError:
                    binding_kind = "raw_entry"
                else:
                    semantic_sha256 = canonical.semantic_sha256
                    binding_kind = "canonical_json"
            for nested in _binding_files(value):
                nested_path = _absolute(
                    "relocatable evidence nested member",
                    nested.remote_absolute_path,
                )
                _relative_member(source_root, nested_path)
                pending.append(nested)
        else:
            _scan_secret_text(body)
            if binding_kind == "entry":
                binding_kind = "raw_entry"
        normalized = _DiscoveredFile(
            remote_absolute_path=remote,
            raw_sha256=raw_sha256,
            semantic_sha256=semantic_sha256,
            size=len(body),
            binding_kind=binding_kind,
        )
        discovered[remote] = (normalized, body)
    members = []
    for remote, (declaration, body) in discovered.items():
        relative = _relative_member(source_root, Path(remote))
        destination = _safe_destination(destination_root, relative)
        _publish_raw_no_replace(destination, body)
        copied = _stable_raw_member(
            destination, label="relocatable evidence copied member"
        )
        if copied != body:
            raise RuntimeError("relocatable evidence copy changed member identity")
        members.append(
            RelocatableEvidenceMember(
                remote_absolute_path=remote,
                relative_path=relative,
                binding_kind=declaration.binding_kind,
                raw_sha256=declaration.raw_sha256 or hashlib.sha256(body).hexdigest(),
                semantic_sha256=declaration.semantic_sha256,
                size=declaration.size or len(body),
            )
        )
    ordered = tuple(sorted(members, key=lambda row: row.relative_path))
    declared_rebindings = {} if directory_rebindings is None else directory_rebindings
    if not isinstance(declared_rebindings, Mapping):
        raise TypeError("relocatable directory rebindings must be a mapping")
    normalized_rebindings = {
        str(_absolute("relocatable directory remote root", remote)): str(
            _absolute("relocatable directory local root", local)
        )
        for remote, local in declared_rebindings.items()
    }
    for remote, local in inherited_directory_rebindings.items():
        previous = normalized_rebindings.setdefault(remote, local)
        if previous != local:
            raise ValueError("relocatable directory rebind override differs")
    if set(normalized_rebindings) != metadata_directories:
        raise ValueError(
            "relocatable directory rebindings differ from prepared snapshots"
        )
    if len(set(normalized_rebindings.values())) != len(normalized_rebindings):
        raise ValueError("relocatable directory rebindings alias a local root")
    rebound_directories = tuple(
        _scan_directory_rebind(remote, local)
        for remote, local in sorted(normalized_rebindings.items())
    )
    root_sha = _canonical_sha256(str(source_root))
    bundle_root = _canonical_sha256(
        {
            "remote_root_sha256": root_sha,
            "entry_remote_absolute_paths": entries,
            "members": tuple(row.to_dict() for row in ordered),
            "directory_rebindings": tuple(row.to_dict() for row in rebound_directories),
        }
    )
    artifact = RelocatableEvidenceBundle(
        schema_version=3,
        kind="lightcone_relocatable_evidence_bundle",
        protocol_sha256=RELOCATABLE_EVIDENCE_BUNDLE_PROTOCOL_SHA256,
        remote_root=str(source_root),
        remote_root_sha256=root_sha,
        entry_remote_absolute_paths=entries,
        members=ordered,
        directory_rebindings=rebound_directories,
        bundle_root_sha256=bundle_root,
    )
    manifest_path = destination_root / RELOCATABLE_EVIDENCE_BUNDLE_FILENAME
    publish_canonical_json_no_replace(manifest_path, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(manifest_path)
    validate_relocatable_evidence_bundle(binding.absolute_path)
    return binding


def validate_relocatable_evidence_bundle(
    manifest_path: str | Path,
) -> ValidatedRelocatableEvidenceBundle:
    manifest_identity = _absolute("relocatable evidence manifest", manifest_path)
    manifest_path_value = relocated_evidence_path(manifest_identity)
    if manifest_identity.name != RELOCATABLE_EVIDENCE_BUNDLE_FILENAME:
        raise ValueError("relocatable evidence manifest filename differs")
    local_root = _private_directory(
        "relocatable evidence local root", manifest_path_value.parent
    )
    manifest = CanonicalJsonProofBinding.bind(manifest_identity)
    artifact = RelocatableEvidenceBundle.from_dict(manifest.reopen())
    expected_relative = {row.relative_path for row in artifact.members}
    actual_relative: set[str] = set()
    inodes: set[tuple[int, int]] = set()
    for path in local_root.rglob("*"):
        status = path.lstat()
        if path.is_symlink():
            raise ValueError("relocatable evidence bundle contains a symlink")
        if stat.S_ISDIR(status.st_mode):
            if status.st_uid != os.geteuid() or stat.S_IMODE(status.st_mode) & 0o077:
                raise ValueError("relocatable evidence directory is unsafe")
            continue
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise ValueError("relocatable evidence member is unsafe")
        relative = path.relative_to(local_root).as_posix()
        if relative == RELOCATABLE_EVIDENCE_BUNDLE_FILENAME:
            continue
        actual_relative.add(relative)
        inode = (status.st_dev, status.st_ino)
        if inode in inodes:
            raise ValueError("relocatable evidence bundle aliases a member")
        inodes.add(inode)
    if actual_relative != expected_relative:
        raise ValueError("relocatable evidence bundle has missing or extra members")
    relocation: dict[str, str] = {}
    metadata_directories: set[str] = set()
    remote_root = Path(artifact.remote_root)
    for member in artifact.members:
        local = _existing_safe_member(local_root, member.relative_path)
        body = _stable_raw_member(local, label="relocatable evidence local member")
        raw_sha256 = hashlib.sha256(body).hexdigest()
        value = _json_value(body)
        if value is not None:
            metadata_directories.update(_metadata_directory_identities(value))
        if member.binding_kind == "canonical_json":
            binding = CanonicalJsonProofBinding.bind(local)
            semantic_sha256 = binding.semantic_sha256
        else:
            semantic_sha256 = member.semantic_sha256
        if (
            raw_sha256 != member.raw_sha256
            or semantic_sha256 != member.semantic_sha256
            or len(body) != member.size
            or str(remote_root.joinpath(*PurePosixPath(member.relative_path).parts))
            != member.remote_absolute_path
        ):
            raise ValueError("relocatable evidence local member identity differs")
        relocation[member.remote_absolute_path] = str(local)
    rebound_remote = {row.remote_absolute_path for row in artifact.directory_rebindings}
    if rebound_remote != metadata_directories:
        raise ValueError(
            "relocatable directory rebindings differ from prepared snapshots"
        )
    for rebound in artifact.directory_rebindings:
        rescanned = _scan_directory_rebind(
            rebound.remote_absolute_path,
            rebound.local_absolute_path,
        )
        if rescanned != rebound:
            raise ValueError("relocatable rehydrated directory changed")
        relocation[rebound.remote_absolute_path] = rebound.local_absolute_path
    return ValidatedRelocatableEvidenceBundle(
        artifact=artifact,
        manifest=manifest,
        local_root=str(local_root),
        relocation=relocation,
    )


@contextmanager
def activate_relocatable_evidence_bundle(
    manifest_path: str | Path,
) -> Iterator[ValidatedRelocatableEvidenceBundle]:
    """Validate a bundle and safely rebind its remote paths for one operation."""

    validated = validate_relocatable_evidence_bundle(manifest_path)
    with use_canonical_json_relocation(validated.relocation):
        for entry in validated.artifact.entry_remote_absolute_paths:
            member = next(
                row
                for row in validated.artifact.members
                if row.remote_absolute_path == entry
            )
            if member.binding_kind == "canonical_json":
                CanonicalJsonProofBinding.bind(entry).reopen()
            else:
                body = _stable_raw_member(
                    Path(validated.relocation[entry]),
                    label="relocatable evidence raw entry",
                )
                if hashlib.sha256(body).hexdigest() != member.raw_sha256:
                    raise ValueError("relocatable evidence raw entry changed")
        yield validated


__all__ = (
    "MAXIMUM_RELOCATABLE_EVIDENCE_MEMBER_BYTES",
    "RELOCATABLE_EVIDENCE_BUNDLE_FILENAME",
    "RELOCATABLE_EVIDENCE_BUNDLE_PROTOCOL_SHA256",
    "RelocatableEvidenceBundle",
    "RelocatableEvidenceMember",
    "ValidatedRelocatableEvidenceBundle",
    "activate_relocatable_evidence_bundle",
    "materialize_relocatable_evidence_bundle",
    "validate_relocatable_evidence_bundle",
)
