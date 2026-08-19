"""Stable local bindings for unsigned GPU qualification receipts.

GPU workers never receive the offline release signing key.  They publish one
canonical JSON receipt, which is pulled to the verifier and bound by a local
``ControlArtifactAttestation``.  This module provides the small, shared file
primitive used by native and distributed qualification artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Self

_EVIDENCE_RELOCATION: ContextVar[dict[str, str] | None] = ContextVar(
    "lightcone_evidence_relocation",
    default=None,
)


def _require_sha256(label: str, value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _absolute_path(label: str, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError(f"{label} path must be absolute and normalized")
    return path


def relocated_evidence_path(identity: str | Path) -> Path:
    """Resolve an immutable remote identity inside an activated pull bundle.

    The serialized identity is never rewritten.  This process-local mapping
    only selects the verified local bytes used by reopen operations, and is
    shared by canonical JSON and raw evidence bindings.
    """

    identity_path = _absolute_path("evidence identity", str(identity))
    mapping = _EVIDENCE_RELOCATION.get()
    if mapping is None:
        return identity_path
    rebound = mapping.get(str(identity_path))
    return (
        identity_path
        if rebound is None
        else _absolute_path("relocated evidence", rebound)
    )


@contextmanager
def use_evidence_relocation(
    mapping: Mapping[str, str],
) -> Iterator[None]:
    """Temporarily reopen immutable remote identities from a verified local root.

    The mapping is intentionally process-local and never serialized into an
    authority object.  ``CanonicalJsonProofBinding`` continues to expose and
    compare the original remote absolute path while all bytes are read from the
    verifier-owned local member selected by a validated relocatable bundle.
    """

    if type(mapping) is not dict or not mapping:
        raise ValueError("canonical JSON relocation mapping must be nonempty")
    normalized: dict[str, str] = {}
    targets: set[str] = set()
    for remote, local in mapping.items():
        if type(remote) is not str or type(local) is not str:
            raise TypeError("canonical JSON relocation paths must be strings")
        remote_path = _absolute_path("remote GPU proof", remote)
        local_path = _absolute_path("local GPU proof", local)
        if str(remote_path) in normalized or str(local_path) in targets:
            raise ValueError("canonical JSON relocation aliases an identity")
        normalized[str(remote_path)] = str(local_path)
        targets.add(str(local_path))
    token = _EVIDENCE_RELOCATION.set(normalized)
    try:
        yield
    finally:
        _EVIDENCE_RELOCATION.reset(token)


# Kept for existing callers; relocation now covers the closed raw-file binding
# union as well as CanonicalJsonProofBinding.
use_canonical_json_relocation = use_evidence_relocation


def _semantic_sha256(value: object) -> str:
    """Canonical object identity (the raw identity additionally binds newline)."""

    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _open_safe_parent(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    """Open every ancestor without following links and pin the evidence parent."""

    if not path.name or path == Path(path.anchor):
        raise ValueError(f"{label} must name one file below a directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parent.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(
                    f"{label} ancestors must be existing symlink-free directories"
                ) from error
            os.close(descriptor)
            descriptor = child
        status = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(status.st_mode)
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) & 0o022
        ):
            raise ValueError(
                f"{label} parent must be a current-user-owned non-writable directory"
            )
        return descriptor, status
    except Exception:
        os.close(descriptor)
        raise


def _stable_canonical_json(path: Path, *, label: str) -> tuple[dict, str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_descriptor, parent_before = _open_safe_parent(path, label=label)
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        os.close(parent_descriptor)
        raise ValueError(f"{label} must be a symlink-free regular file") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_size < 2
            or before.st_size > 2 * 1024 * 1024
        ):
            raise ValueError(
                f"{label} must be one bounded current-user-owned non-writable file"
            )
        body = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        parent_after = os.fstat(parent_descriptor)
        if (
            len(body) != before.st_size
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            != (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            )
            or (
                parent_before.st_dev,
                parent_before.st_ino,
                parent_before.st_mode,
                parent_before.st_uid,
                parent_before.st_mtime_ns,
            )
            != (
                parent_after.st_dev,
                parent_after.st_ino,
                parent_after.st_mode,
                parent_after.st_uid,
                parent_after.st_mtime_ns,
            )
        ):
            raise RuntimeError(f"{label} changed while read")
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    if type(value) is not dict or body != _canonical_bytes(value):
        raise ValueError(f"{label} is not one canonical JSON object")
    return value, hashlib.sha256(body).hexdigest(), len(body)


def publish_canonical_json_no_replace(
    path: str | Path,
    value: object,
) -> tuple[str, int]:
    """Atomically publish one unsigned canonical receipt without replacement."""

    destination = _absolute_path("GPU proof", str(path))
    parent_descriptor, _parent_status = _open_safe_parent(
        destination, label="GPU proof"
    )
    try:
        body = _canonical_bytes(value)
        if len(body) > 2 * 1024 * 1024:
            raise ValueError("GPU proof receipt exceeds the bounded schema")
        temporary = destination.with_name(f".{destination.name}.tmp.{uuid.uuid4().hex}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(temporary.name, flags, 0o600, dir_fd=parent_descriptor)
        try:
            offset = 0
            while offset < len(body):
                offset += os.write(descriptor, body[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary.name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise RuntimeError("GPU proof target already exists") from error
        finally:
            try:
                os.unlink(temporary.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.fsync(parent_descriptor)
        return hashlib.sha256(body).hexdigest(), len(body)
    finally:
        os.close(parent_descriptor)


@dataclass(frozen=True)
class CanonicalJsonProofBinding:
    """Path/raw/semantic identity for one unsigned remote proof receipt."""

    absolute_path: str
    raw_sha256: str
    semantic_sha256: str
    size: int

    def __post_init__(self) -> None:
        _absolute_path("GPU proof", self.absolute_path)
        _require_sha256("GPU proof raw digest", self.raw_sha256)
        _require_sha256("GPU proof semantic digest", self.semantic_sha256)
        if type(self.size) is not int or self.size < 2 or self.size > 2 * 1024 * 1024:
            raise ValueError("GPU proof file size is invalid")

    @classmethod
    def bind(
        cls,
        path: str | Path,
        *,
        semantic_sha256: str | None = None,
    ) -> Self:
        source = _absolute_path("GPU proof", str(path))
        value, raw_sha256, size = _stable_canonical_json(
            relocated_evidence_path(source), label="GPU proof"
        )
        computed_semantic_sha256 = _semantic_sha256(value)
        if semantic_sha256 is not None and semantic_sha256 != computed_semantic_sha256:
            raise ValueError("GPU proof expected semantic digest differs from content")
        return cls(
            absolute_path=str(source),
            raw_sha256=raw_sha256,
            semantic_sha256=computed_semantic_sha256,
            size=size,
        )

    def reopen(self) -> dict:
        value, raw_sha256, size = _stable_canonical_json(
            relocated_evidence_path(self.absolute_path),
            label="GPU proof",
        )
        if (
            raw_sha256 != self.raw_sha256
            or _semantic_sha256(value) != self.semantic_sha256
            or size != self.size
        ):
            raise ValueError("GPU proof raw or semantic file identity changed")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "absolute_path": self.absolute_path,
            "raw_sha256": self.raw_sha256,
            "semantic_sha256": self.semantic_sha256,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "absolute_path",
            "raw_sha256",
            "semantic_sha256",
            "size",
        }:
            raise ValueError("GPU proof binding fields differ")
        return cls(**value)


__all__ = (
    "CanonicalJsonProofBinding",
    "publish_canonical_json_no_replace",
    "relocated_evidence_path",
    "use_canonical_json_relocation",
    "use_evidence_relocation",
)
