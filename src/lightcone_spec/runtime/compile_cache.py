"""Content-addressed, immutable compile-cache contracts.

The cache deliberately separates reusable read-only objects from a private
writable overlay owned by one process/attempt.  A completed object is selected
only through a content-bound receipt; directory discovery is never evidence
that a cache entry is safe to reuse.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or "\n" in value:
        raise ValueError(f"{name} must be non-empty single-line text")


@dataclass(frozen=True)
class CompileCacheKey:
    """Every identity allowed to affect compiled code or graph buckets."""

    patched_sglang_tree: str
    source_sha256: str
    python_version: str
    torch_version: str
    triton_version: str
    cuda_version: str
    driver_version: str
    sm_architecture: str
    gpu_model: str
    dtype: str
    target_revision: str
    drafter_revision: str | None
    tensor_parallel_size: int
    context_limit: int
    max_running_requests: int
    graph_buckets: tuple[int, ...]
    allocator: str
    build_flags: tuple[str, ...]
    serialized_cuda_graphs: bool = False

    def validate(self) -> None:
        if not _GIT_OBJECT.fullmatch(self.patched_sglang_tree):
            raise ValueError("patched_sglang_tree must be a lowercase Git object")
        if not _SHA256.fullmatch(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256")
        for name in (
            "python_version",
            "torch_version",
            "triton_version",
            "cuda_version",
            "driver_version",
            "sm_architecture",
            "gpu_model",
            "dtype",
            "target_revision",
            "allocator",
        ):
            _require_text(name, getattr(self, name))
        if self.drafter_revision is not None:
            _require_text("drafter_revision", self.drafter_revision)
        for name in (
            "tensor_parallel_size",
            "context_limit",
            "max_running_requests",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not self.graph_buckets
            or tuple(sorted(set(self.graph_buckets))) != self.graph_buckets
            or any(
                isinstance(bucket, bool) or not isinstance(bucket, int) or bucket < 1
                for bucket in self.graph_buckets
            )
        ):
            raise ValueError("graph_buckets must be unique increasing integers")
        if tuple(sorted(set(self.build_flags))) != self.build_flags or any(
            not flag or "\n" in flag for flag in self.build_flags
        ):
            raise ValueError("build_flags must be unique sorted single-line values")
        if self.serialized_cuda_graphs:
            raise ValueError("CUDA graphs are live-process state and cannot be cached")

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256({"schema_version": 1, **asdict(self)})


@dataclass(frozen=True)
class CompileCacheFile:
    relative_path: str
    size: int
    sha256: str

    def validate(self) -> None:
        path = Path(self.relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("cache file path must be a safe relative path")
        if isinstance(self.size, bool) or self.size < 0:
            raise ValueError("cache file size must be non-negative")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("cache file digest must be a lowercase SHA-256")


@dataclass(frozen=True)
class CompileCacheReceipt:
    schema_version: int
    kind: str
    key_sha256: str
    content_sha256: str
    attempt_id: str
    process_id: int
    jit_duration_ns: int
    files: tuple[CompileCacheFile, ...]

    def validate(self) -> None:
        if self.schema_version != 1 or self.kind != "compile_cache_receipt":
            raise ValueError("compile-cache receipt schema is unsupported")
        if not _SHA256.fullmatch(self.key_sha256):
            raise ValueError("compile-cache key digest is invalid")
        if not _SHA256.fullmatch(self.content_sha256):
            raise ValueError("compile-cache content digest is invalid")
        if not _SAFE_COMPONENT.fullmatch(self.attempt_id):
            raise ValueError("compile-cache attempt ID is unsafe")
        if isinstance(self.process_id, bool) or self.process_id < 1:
            raise ValueError("compile-cache process ID must be positive")
        if isinstance(self.jit_duration_ns, bool) or self.jit_duration_ns < 0:
            raise ValueError("JIT duration must be non-negative")
        paths = tuple(item.relative_path for item in self.files)
        if not self.files or paths != tuple(sorted(set(paths))):
            raise ValueError("compile-cache files must be non-empty and sorted")
        for item in self.files:
            item.validate()
        expected = _content_sha256(
            {
                "key_sha256": self.key_sha256,
                "files": [asdict(item) for item in self.files],
            }
        )
        if self.content_sha256 != expected:
            raise ValueError("compile-cache content digest does not match files")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _content_sha256(asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        resolved = Path(path)
        if resolved.is_symlink() or not resolved.is_file():
            raise ValueError("compile-cache receipt must be a regular file")
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version",
            "kind",
            "key_sha256",
            "content_sha256",
            "attempt_id",
            "process_id",
            "jit_duration_ns",
            "files",
        }:
            raise ValueError("compile-cache receipt has unknown or missing fields")
        files = raw.pop("files")
        if not isinstance(files, list):
            raise TypeError("compile-cache receipt files must be a list")
        value = cls(
            **raw,
            files=tuple(CompileCacheFile(**item) for item in files),
        )
        value.validate()
        sidecar = resolved.with_name(f"{resolved.name}.sha256")
        if sidecar.is_symlink() or not sidecar.is_file():
            raise ValueError("compile-cache receipt sidecar is missing")
        if sidecar.read_text(encoding="utf-8").strip() != value.receipt_sha256:
            raise ValueError("compile-cache receipt sidecar differs from content")
        return value


@dataclass(frozen=True)
class CompileCacheOverlay:
    key: CompileCacheKey
    attempt_id: str
    process_id: int
    path: Path
    started_ns: int

    def validate(self) -> None:
        self.key.validate()
        if not _SAFE_COMPONENT.fullmatch(self.attempt_id):
            raise ValueError("compile-cache attempt ID is unsafe")
        if isinstance(self.process_id, bool) or self.process_id < 1:
            raise ValueError("compile-cache process ID must be positive")
        if self.path.is_symlink() or not self.path.is_dir():
            raise ValueError("compile-cache overlay must be a private directory")
        if isinstance(self.started_ns, bool) or self.started_ns < 0:
            raise ValueError("compile-cache start time must be non-negative")


class CompileCacheCorruptionError(RuntimeError):
    """A content-bound cache object no longer matches its terminal receipt."""


class ImmutableCompileCache:
    """Store verified cache objects and issue process-private writable overlays."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.objects = self.root / "objects"
        self.overlays = self.root / "overlays"
        self.receipts = self.root / "receipts"
        for path in (self.root, self.objects, self.overlays, self.receipts):
            path.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise ValueError("compile-cache roots cannot be symlinks")

    def create_overlay(
        self,
        key: CompileCacheKey,
        *,
        process_id: int,
        attempt_id: str,
    ) -> CompileCacheOverlay:
        key.validate()
        if isinstance(process_id, bool) or process_id < 1:
            raise ValueError("compile-cache process ID must be positive")
        if not _SAFE_COMPONENT.fullmatch(attempt_id):
            raise ValueError("compile-cache attempt ID is unsafe")
        path = self.overlays / f"{key.sha256}.pid{process_id}.{attempt_id}"
        path.mkdir(mode=0o700)
        value = CompileCacheOverlay(
            key=key,
            attempt_id=attempt_id,
            process_id=process_id,
            path=path,
            started_ns=time.monotonic_ns(),
        )
        value.validate()
        return value

    @staticmethod
    def _inventory(path: Path) -> tuple[CompileCacheFile, ...]:
        files: list[CompileCacheFile] = []
        for item in sorted(path.rglob("*")):
            if item.is_symlink():
                raise ValueError("compile-cache objects cannot contain symlinks")
            if not item.is_file():
                continue
            relative = item.relative_to(path).as_posix()
            files.append(
                CompileCacheFile(
                    relative_path=relative,
                    size=item.stat().st_size,
                    sha256=_file_sha256(item),
                )
            )
        return tuple(files)

    def seal_overlay(self, overlay: CompileCacheOverlay) -> tuple[Path, Path]:
        overlay.validate()
        if not overlay.path.is_relative_to(self.overlays):
            raise ValueError("compile-cache overlay belongs to another store")
        files = self._inventory(overlay.path)
        content_sha256 = _content_sha256(
            {
                "key_sha256": overlay.key.sha256,
                "files": [asdict(item) for item in files],
            }
        )
        receipt = CompileCacheReceipt(
            schema_version=1,
            kind="compile_cache_receipt",
            key_sha256=overlay.key.sha256,
            content_sha256=content_sha256,
            attempt_id=overlay.attempt_id,
            process_id=overlay.process_id,
            jit_duration_ns=max(0, time.monotonic_ns() - overlay.started_ns),
            files=files,
        )
        receipt.validate()
        object_path = self.objects / content_sha256
        if not object_path.exists():
            temporary = self.objects / f".{content_sha256}.tmp.{uuid.uuid4().hex}"
            temporary.mkdir(mode=0o700)
            try:
                for item in files:
                    source = overlay.path / item.relative_path
                    target = temporary / item.relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    os.chmod(target, 0o444)
                    _fsync_file(target)
                for directory in sorted(
                    (item for item in temporary.rglob("*") if item.is_dir()),
                    reverse=True,
                ):
                    os.chmod(directory, 0o555)
                    _fsync_directory(directory)
                os.chmod(temporary, 0o555)
                try:
                    os.rename(temporary, object_path)
                except FileExistsError:
                    shutil.rmtree(temporary)
                _fsync_directory(self.objects)
            except BaseException:
                if temporary.exists():
                    shutil.rmtree(temporary)
                raise
        receipt_path = self.receipts / f"{receipt.receipt_sha256}.json"
        if not receipt_path.exists():
            _atomic_json(receipt_path, asdict(receipt))
            _atomic_text(
                receipt_path.with_name(f"{receipt_path.name}.sha256"),
                receipt.receipt_sha256,
            )
        self.verify(overlay.key, receipt_path)
        return object_path, receipt_path

    def verify(self, key: CompileCacheKey, receipt_path: str | Path) -> Path:
        key.validate()
        receipt = CompileCacheReceipt.load(receipt_path)
        if receipt.key_sha256 != key.sha256:
            raise CompileCacheCorruptionError("cache receipt belongs to another key")
        object_path = self.objects / receipt.content_sha256
        if object_path.is_symlink() or not object_path.is_dir():
            raise CompileCacheCorruptionError("cache object is missing")
        try:
            inventory = self._inventory(object_path)
        except ValueError as error:
            raise CompileCacheCorruptionError(str(error)) from error
        if inventory != receipt.files:
            raise CompileCacheCorruptionError("cache object content changed")
        return object_path
