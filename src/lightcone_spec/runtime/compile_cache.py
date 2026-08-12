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
import stat
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from lightcone_spec import PINNED_SGLANG_TREE

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

# These values are release data, not launch-time assertions supplied by a
# caller.  The manifest digest is over canonical JSON (the same encoding used
# for every contract in this module); the patch digest is over the exact mail
# patch bytes registered by that manifest.
PINNED_SGLANG_PATCH_MANIFEST_SHA256 = (
    "d0902d27704a98edf0f87f4bbdbe88854fa423c7b56463c22a2efe644afc05a1"
)
PINNED_SGLANG_PATCH_SHA256 = (
    "369f72a3edda128881c79d8af34f0ecaacfc0fd3ee78adc99ad96a7e091154a7"
)
SGLANG_FIRST_PARTY_COMPILE_BUILDER = "lightcone_spec.sglang_bridge.launch_server.v1"

_CACHE_ENVIRONMENT = {
    "CCACHE_DIR": "ccache",
    "CUDA_CACHE_PATH": "cuda",
    "DG_JIT_CACHE_DIR": "deep_gemm",
    "FLASHINFER_JIT_DIR": "flashinfer/cached_ops",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "flash_attention_cute",
    "NUMBA_CACHE_DIR": "numba",
    "SGLANG_CACHE_DIR": "sglang",
    "SGLANG_DG_CACHE_DIR": "deep_gemm",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "TRITON_CACHE_DIR": "triton",
    "XDG_CACHE_HOME": "xdg",
}
COMPILE_CACHE_ENVIRONMENT_VARIABLES = tuple(sorted(_CACHE_ENVIRONMENT))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


# Release-owned semantic source identity for compiled SGLang artifacts.  It is
# intentionally derived only from repository constants: a caller cannot mint a
# launch plan for a different source tree while retaining the release builder.
PINNED_SGLANG_COMPILE_SOURCE_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_sglang_compile_source",
        "patched_sglang_tree": PINNED_SGLANG_TREE,
        "patch_manifest_sha256": PINNED_SGLANG_PATCH_MANIFEST_SHA256,
        "patch_sha256": PINNED_SGLANG_PATCH_SHA256,
    }
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("compile-cache objects must contain regular files")
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


def _publish_json(path: Path, value: object) -> None:
    """Publish immutable JSON without replacing a concurrent winner."""

    encoded = _canonical_bytes(value) + b"\n"
    temporary = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise ValueError("immutable compile-cache artifact already differs")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_text(path: Path, value: str) -> None:
    encoded = f"{value}\n".encode()
    temporary = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise ValueError("immutable compile-cache sidecar already differs")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or "\n" in value:
        raise ValueError(f"{name} must be non-empty single-line text")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _strict_absolute_path(name: str, value: str) -> Path:
    _require_text(name, value)
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{name} must be an absolute normalized path")
    if path == Path(path.anchor):
        raise ValueError(f"{name} cannot be a filesystem root")
    return path


@dataclass(frozen=True)
class CompileCacheKey:
    """Every identity allowed to affect compiled code or graph buckets."""

    patched_sglang_tree: str
    patch_manifest_sha256: str
    patch_sha256: str
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
        _require_sha256("patch_manifest_sha256", self.patch_manifest_sha256)
        _require_sha256("patch_sha256", self.patch_sha256)
        _require_sha256("source_sha256", self.source_sha256)
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
        if not isinstance(self.serialized_cuda_graphs, bool):
            raise TypeError("serialized_cuda_graphs must be boolean")
        if self.serialized_cuda_graphs:
            raise ValueError("CUDA graphs are live-process state and cannot be cached")

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256({"schema_version": 1, **asdict(self)})

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        if not isinstance(raw, dict) or set(raw) != {
            "patched_sglang_tree",
            "patch_manifest_sha256",
            "patch_sha256",
            "source_sha256",
            "python_version",
            "torch_version",
            "triton_version",
            "cuda_version",
            "driver_version",
            "sm_architecture",
            "gpu_model",
            "dtype",
            "target_revision",
            "drafter_revision",
            "tensor_parallel_size",
            "context_limit",
            "max_running_requests",
            "graph_buckets",
            "allocator",
            "build_flags",
            "serialized_cuda_graphs",
        }:
            raise ValueError("compile-cache key has unknown or missing fields")
        graph_buckets = raw["graph_buckets"]
        build_flags = raw["build_flags"]
        if not isinstance(graph_buckets, list) or not isinstance(build_flags, list):
            raise TypeError("compile-cache key tuple fields must be JSON lists")
        value = cls(
            **{
                name: item
                for name, item in raw.items()
                if name not in {"graph_buckets", "build_flags"}
            },
            graph_buckets=tuple(graph_buckets),
            build_flags=tuple(build_flags),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class CompileCacheFile:
    relative_path: str
    size: int
    sha256: str

    def validate(self) -> None:
        _require_text("cache file path", self.relative_path)
        path = Path(self.relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("cache file path must be a safe relative path")
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
        ):
            raise ValueError("cache file size must be non-negative")
        _require_sha256("cache file digest", self.sha256)


@dataclass(frozen=True)
class CompileCacheReceipt:
    schema_version: int
    kind: str
    key_sha256: str
    content_sha256: str
    builder_id: str
    launch_plan_sha256: str | None
    attempt_id: str
    process_id: int
    jit_duration_ns: int
    files: tuple[CompileCacheFile, ...]

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 2
            or self.kind != "compile_cache_receipt"
        ):
            raise ValueError("compile-cache receipt schema is unsupported")
        _require_sha256("compile-cache key digest", self.key_sha256)
        _require_sha256("compile-cache content digest", self.content_sha256)
        if self.builder_id == SGLANG_FIRST_PARTY_COMPILE_BUILDER:
            if self.launch_plan_sha256 is None:
                raise ValueError("first-party cache receipt lacks its launch plan")
            _require_sha256("launch_plan_sha256", self.launch_plan_sha256)
        elif self.builder_id == "unattributed_manual_builder.v1":
            if self.launch_plan_sha256 is not None:
                raise ValueError("manual cache receipt cannot claim a launch plan")
        else:
            raise ValueError("compile-cache receipt builder is unsupported")
        if not _SAFE_COMPONENT.fullmatch(self.attempt_id):
            raise ValueError("compile-cache attempt ID is unsafe")
        if (
            isinstance(self.process_id, bool)
            or not isinstance(self.process_id, int)
            or self.process_id < 1
        ):
            raise ValueError("compile-cache process ID must be positive")
        if (
            isinstance(self.jit_duration_ns, bool)
            or not isinstance(self.jit_duration_ns, int)
            or self.jit_duration_ns < 0
        ):
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
            "builder_id",
            "launch_plan_sha256",
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
class CompileCacheLaunchPlan:
    """Release-issued instruction for one fail-closed SGLang cache launch."""

    schema_version: int
    kind: str
    key: CompileCacheKey
    cache_root: str
    cache_mode: str
    builder_id: str
    base_receipt_path: str | None
    base_receipt_sha256: str | None

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "compile_cache_launch_plan"
        ):
            raise ValueError("compile-cache launch plan schema is unsupported")
        self.key.validate()
        if self.key.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("compile-cache plan has a foreign patched SGLang tree")
        if self.key.patch_manifest_sha256 != PINNED_SGLANG_PATCH_MANIFEST_SHA256:
            raise ValueError("compile-cache plan has a foreign patch manifest")
        if self.key.patch_sha256 != PINNED_SGLANG_PATCH_SHA256:
            raise ValueError("compile-cache plan has a foreign semantic patch")
        if self.key.source_sha256 != PINNED_SGLANG_COMPILE_SOURCE_SHA256:
            raise ValueError("compile-cache plan has a foreign source identity")
        root = _strict_absolute_path("cache_root", self.cache_root)
        if self.cache_mode not in {"build", "reuse"}:
            raise ValueError("compile-cache mode must be build or reuse")
        if self.builder_id != SGLANG_FIRST_PARTY_COMPILE_BUILDER:
            raise ValueError("compile-cache builder is not release-owned")
        if self.cache_mode == "build":
            if (
                self.base_receipt_path is not None
                or self.base_receipt_sha256 is not None
            ):
                raise ValueError("a cache build cannot claim an existing base receipt")
            return
        if self.base_receipt_path is None or self.base_receipt_sha256 is None:
            raise ValueError("cache reuse requires an exact base receipt")
        receipt_path = _strict_absolute_path(
            "base_receipt_path", self.base_receipt_path
        )
        if not receipt_path.is_relative_to(root / "receipts"):
            raise ValueError("base receipt must belong to the selected cache root")
        _require_sha256("base_receipt_sha256", self.base_receipt_sha256)

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256(asdict(self))

    @classmethod
    def issue(
        cls,
        *,
        key: CompileCacheKey,
        cache_root: str | Path,
        cache_mode: str,
        base_receipt_path: str | Path | None = None,
    ) -> Self:
        """Issue a plan from locally verified state, never cache discovery."""

        key.validate()
        if key.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("compile-cache plan has a foreign patched SGLang tree")
        if key.patch_manifest_sha256 != PINNED_SGLANG_PATCH_MANIFEST_SHA256:
            raise ValueError("compile-cache plan has a foreign patch manifest")
        if key.patch_sha256 != PINNED_SGLANG_PATCH_SHA256:
            raise ValueError("compile-cache plan has a foreign semantic patch")
        if key.source_sha256 != PINNED_SGLANG_COMPILE_SOURCE_SHA256:
            raise ValueError("compile-cache plan has a foreign source identity")
        requested_root = Path(cache_root)
        if requested_root.is_symlink():
            raise ValueError("compile-cache root cannot be a symlink")
        root = requested_root.resolve()
        cache = ImmutableCompileCache(root)
        selected_path: str | None = None
        selected_sha256: str | None = None
        if cache_mode == "reuse":
            if base_receipt_path is None:
                raise ValueError("cache reuse requires an explicit receipt")
            requested_receipt = Path(base_receipt_path)
            if requested_receipt.is_symlink():
                raise ValueError("base receipt cannot be a symlink")
            receipt_path = requested_receipt.resolve()
            if not receipt_path.is_relative_to(root / "receipts"):
                raise ValueError("base receipt must belong to the selected cache root")
            receipt = CompileCacheReceipt.load(receipt_path)
            if (
                receipt.builder_id != SGLANG_FIRST_PARTY_COMPILE_BUILDER
                or receipt.launch_plan_sha256 is None
            ):
                raise CompileCacheForeignIdentityError(
                    "cache receipt was not produced by the release builder"
                )
            if receipt.key_sha256 != key.sha256:
                raise CompileCacheForeignIdentityError(
                    "cache receipt belongs to another key"
                )
            cache.verify(key, receipt_path)
            selected_path = str(receipt_path)
            selected_sha256 = receipt.receipt_sha256
        elif cache_mode != "build":
            raise ValueError("compile-cache mode must be build or reuse")
        elif base_receipt_path is not None:
            raise ValueError("a cache build cannot accept a base receipt")
        value = cls(
            schema_version=1,
            kind="compile_cache_launch_plan",
            key=key,
            cache_root=str(root),
            cache_mode=cache_mode,
            builder_id=SGLANG_FIRST_PARTY_COMPILE_BUILDER,
            base_receipt_path=selected_path,
            base_receipt_sha256=selected_sha256,
        )
        value.validate()
        return value

    def write(self, path: str | Path) -> Path:
        self.validate()
        resolved = Path(path)
        if resolved.is_symlink() or not resolved.parent.is_dir():
            raise ValueError("compile-cache plan parent must be a regular directory")
        _publish_json(resolved, asdict(self))
        _publish_text(resolved.with_name(f"{resolved.name}.sha256"), self.sha256)
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> Self:
        resolved = Path(path)
        if resolved.is_symlink() or not resolved.is_file():
            raise ValueError("compile-cache plan must be a regular file")
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version",
            "kind",
            "key",
            "cache_root",
            "cache_mode",
            "builder_id",
            "base_receipt_path",
            "base_receipt_sha256",
        }:
            raise ValueError("compile-cache plan has unknown or missing fields")
        key = CompileCacheKey.from_dict(raw.pop("key"))
        value = cls(**raw, key=key)
        value.validate()
        sidecar = resolved.with_name(f"{resolved.name}.sha256")
        if sidecar.is_symlink() or not sidecar.is_file():
            raise ValueError("compile-cache plan sidecar is missing")
        if sidecar.read_text(encoding="utf-8").strip() != value.sha256:
            raise ValueError("compile-cache plan sidecar differs from content")
        return value


@dataclass(frozen=True)
class CompileCacheAttemptReceipt:
    """Immutable lifecycle evidence for a launch cache attempt."""

    schema_version: int
    kind: str
    plan_sha256: str
    key_sha256: str
    attempt_id: str
    process_id: int
    state: str
    started_ns: int
    finished_ns: int | None
    overlay_name: str
    base_receipt_sha256: str | None
    result_receipt_sha256: str | None
    failure_code: str | None
    failure_detail_sha256: str | None
    environment: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "compile_cache_attempt_receipt"
        ):
            raise ValueError("compile-cache attempt receipt schema is unsupported")
        _require_sha256("plan_sha256", self.plan_sha256)
        _require_sha256("key_sha256", self.key_sha256)
        if not _SAFE_COMPONENT.fullmatch(self.attempt_id):
            raise ValueError("compile-cache attempt ID is unsafe")
        if (
            isinstance(self.process_id, bool)
            or not isinstance(self.process_id, int)
            or self.process_id < 1
        ):
            raise ValueError("compile-cache process ID must be positive")
        if self.state not in {"started", "ready", "complete", "failed"}:
            raise ValueError("compile-cache attempt state is invalid")
        if (
            isinstance(self.started_ns, bool)
            or not isinstance(self.started_ns, int)
            or self.started_ns < 0
        ):
            raise ValueError("compile-cache attempt start is invalid")
        if self.finished_ns is not None and (
            isinstance(self.finished_ns, bool)
            or not isinstance(self.finished_ns, int)
            or self.finished_ns < self.started_ns
        ):
            raise ValueError("compile-cache attempt finish is invalid")
        if not _SAFE_COMPONENT.fullmatch(self.overlay_name):
            raise ValueError("compile-cache overlay name is unsafe")
        for name, value in (
            ("base_receipt_sha256", self.base_receipt_sha256),
            ("result_receipt_sha256", self.result_receipt_sha256),
            ("failure_detail_sha256", self.failure_detail_sha256),
        ):
            if value is not None:
                _require_sha256(name, value)
        expected_environment = tuple(sorted(_CACHE_ENVIRONMENT.items()))
        if self.environment != expected_environment:
            raise ValueError("compile-cache environment binding is incomplete")
        if self.state == "started":
            if any(
                value is not None
                for value in (
                    self.finished_ns,
                    self.result_receipt_sha256,
                    self.failure_code,
                    self.failure_detail_sha256,
                )
            ):
                raise ValueError("started cache attempt contains terminal fields")
        elif self.state == "ready":
            if (
                self.finished_ns is None
                or self.result_receipt_sha256 is not None
                or self.failure_code is not None
                or self.failure_detail_sha256 is not None
            ):
                raise ValueError("ready cache attempt fields are inconsistent")
        elif self.state == "complete":
            if (
                self.finished_ns is None
                or self.result_receipt_sha256 is None
                or self.failure_code is not None
                or self.failure_detail_sha256 is not None
            ):
                raise ValueError("complete cache attempt fields are inconsistent")
        elif (
            self.finished_ns is None
            or self.result_receipt_sha256 is not None
            or self.failure_code is None
            or not _SAFE_COMPONENT.fullmatch(self.failure_code)
            or self.failure_detail_sha256 is None
        ):
            raise ValueError("failed cache attempt fields are inconsistent")

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256(asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        resolved = Path(path)
        if resolved.is_symlink() or not resolved.is_file():
            raise ValueError("compile-cache attempt receipt must be a regular file")
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "kind",
            "plan_sha256",
            "key_sha256",
            "attempt_id",
            "process_id",
            "state",
            "started_ns",
            "finished_ns",
            "overlay_name",
            "base_receipt_sha256",
            "result_receipt_sha256",
            "failure_code",
            "failure_detail_sha256",
            "environment",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(
                "compile-cache attempt receipt has unknown or missing fields"
            )
        environment = raw.pop("environment")
        if not isinstance(environment, list) or any(
            not isinstance(row, list) or len(row) != 2 for row in environment
        ):
            raise TypeError("compile-cache attempt environment must be JSON pairs")
        value = cls(**raw, environment=tuple(tuple(row) for row in environment))
        value.validate()
        sidecar = resolved.with_name(f"{resolved.name}.sha256")
        if sidecar.is_symlink() or not sidecar.is_file():
            raise ValueError("compile-cache attempt receipt sidecar is missing")
        if sidecar.read_text(encoding="utf-8").strip() != value.sha256:
            raise ValueError("compile-cache attempt receipt sidecar differs")
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
        if (
            isinstance(self.process_id, bool)
            or not isinstance(self.process_id, int)
            or self.process_id < 1
        ):
            raise ValueError("compile-cache process ID must be positive")
        if self.path.is_symlink() or not self.path.is_dir():
            raise ValueError("compile-cache overlay must be a private directory")
        if (
            isinstance(self.started_ns, bool)
            or not isinstance(self.started_ns, int)
            or self.started_ns < 0
        ):
            raise ValueError("compile-cache start time must be non-negative")


class CompileCacheCorruptionError(RuntimeError):
    """A content-bound cache object no longer matches its terminal receipt."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "cache_object_corrupt",
    ) -> None:
        super().__init__(message)
        if not _SAFE_COMPONENT.fullmatch(reason_code):
            raise ValueError("compile-cache failure reason is unsafe")
        self.reason_code = reason_code


class CompileCacheForeignIdentityError(CompileCacheCorruptionError):
    """A selected receipt or source identity belongs to another cache key."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="foreign_cache_identity")


class CompileCacheIncompleteError(CompileCacheCorruptionError):
    """A first-party builder terminated without any cache object files."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="incomplete_cache_build")


class ImmutableCompileCache:
    """Store verified cache objects and issue process-private writable overlays."""

    def __init__(self, root: str | Path) -> None:
        supplied = Path(root)
        if supplied.is_symlink():
            raise ValueError("compile-cache root cannot be a symlink")
        self.root = supplied.resolve()
        if self.root == Path(self.root.anchor):
            raise ValueError("compile-cache root cannot be a filesystem root")
        self.objects = self.root / "objects"
        self.overlays = self.root / "overlays"
        self.receipts = self.root / "receipts"
        self.attempts = self.root / "attempts"
        for path in (
            self.root,
            self.objects,
            self.overlays,
            self.receipts,
            self.attempts,
        ):
            path.mkdir(parents=True, exist_ok=True)
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("compile-cache roots must be regular directories")

    @classmethod
    def _open_existing_read_only(cls, root: str | Path) -> Self:
        """Open only the immutable portion of an existing store without mutation."""

        supplied = Path(root)
        if supplied.is_symlink():
            raise CompileCacheCorruptionError(
                "compile-cache root is a symlink",
                reason_code="invalid_cache_store",
            )
        resolved = supplied.resolve(strict=False)
        if resolved == Path(resolved.anchor):
            raise ValueError("compile-cache root cannot be a filesystem root")
        value = cls.__new__(cls)
        value.root = resolved
        value.objects = resolved / "objects"
        value.overlays = resolved / "overlays"
        value.receipts = resolved / "receipts"
        value.attempts = resolved / "attempts"
        for path in (value.root, value.objects, value.receipts):
            try:
                metadata = path.lstat()
            except OSError as error:
                raise CompileCacheCorruptionError(
                    "compile-cache immutable store is missing",
                    reason_code="invalid_cache_store",
                ) from error
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise CompileCacheCorruptionError(
                    "compile-cache immutable store is not a regular directory",
                    reason_code="invalid_cache_store",
                )
        return value

    def create_overlay(
        self,
        key: CompileCacheKey,
        *,
        process_id: int,
        attempt_id: str,
    ) -> CompileCacheOverlay:
        key.validate()
        if (
            isinstance(process_id, bool)
            or not isinstance(process_id, int)
            or process_id < 1
        ):
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
            metadata = item.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("compile-cache objects cannot contain symlinks")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("compile-cache objects must contain regular files")
            relative = item.relative_to(path).as_posix()
            files.append(
                CompileCacheFile(
                    relative_path=relative,
                    size=metadata.st_size,
                    sha256=_file_sha256(item),
                )
            )
        return tuple(files)

    @staticmethod
    def _verify_immutable_object(path: Path) -> None:
        for item in (path, *sorted(path.rglob("*"))):
            metadata = item.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise CompileCacheCorruptionError("cache object contains a symlink")
            if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                raise CompileCacheCorruptionError(
                    "cache object contains a special file"
                )
            if metadata.st_mode & 0o222:
                raise CompileCacheCorruptionError("cache object is writable")

    @staticmethod
    def _discard_staging_directory(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("compile-cache staging path is invalid")
        for directory in sorted(
            (item for item in path.rglob("*") if item.is_dir()), reverse=True
        ):
            os.chmod(directory, 0o700)
        os.chmod(path, 0o700)
        shutil.rmtree(path)

    def _publish_object_directory(self, temporary: Path, object_path: Path) -> None:
        """Publish one directory under an atomic per-content claim."""

        claim = self.objects / f".{object_path.name}.publish"
        try:
            claim.mkdir(mode=0o700)
        except FileExistsError:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not claim.exists() and object_path.exists():
                    break
                time.sleep(0.01)
            self._discard_staging_directory(temporary)
            if claim.exists() or not object_path.exists():
                raise CompileCacheCorruptionError(
                    "compile-cache publication is incomplete",
                    reason_code="incomplete_cache_publication",
                )
            return
        try:
            if object_path.exists():
                self._discard_staging_directory(temporary)
            else:
                os.rename(temporary, object_path)
                _fsync_directory(self.objects)
        finally:
            claim.rmdir()
            _fsync_directory(self.objects)

    def _seal_overlay(
        self,
        overlay: CompileCacheOverlay,
        *,
        builder_id: str,
        launch_plan_sha256: str | None,
    ) -> tuple[Path, Path]:
        overlay.validate()
        if not overlay.path.is_relative_to(self.overlays):
            raise ValueError("compile-cache overlay belongs to another store")
        files = self._inventory(overlay.path)
        if not files:
            raise CompileCacheIncompleteError(
                "first-party compile-cache builder produced no files"
            )
        content_sha256 = _content_sha256(
            {
                "key_sha256": overlay.key.sha256,
                "files": [asdict(item) for item in files],
            }
        )
        receipt = CompileCacheReceipt(
            schema_version=2,
            kind="compile_cache_receipt",
            key_sha256=overlay.key.sha256,
            content_sha256=content_sha256,
            builder_id=builder_id,
            launch_plan_sha256=launch_plan_sha256,
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
                _fsync_directory(temporary)
                self._publish_object_directory(temporary, object_path)
            except BaseException:
                if temporary.exists():
                    self._discard_staging_directory(temporary)
                raise
        receipt_path = self.receipts / f"{receipt.receipt_sha256}.json"
        _publish_json(receipt_path, asdict(receipt))
        _publish_text(
            receipt_path.with_name(f"{receipt_path.name}.sha256"),
            receipt.receipt_sha256,
        )
        self.verify(overlay.key, receipt_path)
        return object_path, receipt_path

    def seal_overlay(self, overlay: CompileCacheOverlay) -> tuple[Path, Path]:
        """Seal a non-formal object that cannot be selected by a launch plan."""

        return self._seal_overlay(
            overlay,
            builder_id="unattributed_manual_builder.v1",
            launch_plan_sha256=None,
        )

    def seal_launch_overlay(
        self,
        overlay: CompileCacheOverlay,
        *,
        plan: CompileCacheLaunchPlan,
    ) -> tuple[Path, Path]:
        """Seal locally measured output of the registered SGLang builder."""

        plan.validate()
        if overlay.key.sha256 != plan.key.sha256:
            raise CompileCacheForeignIdentityError(
                "compile-cache overlay differs from its launch plan"
            )
        return self._seal_overlay(
            overlay,
            builder_id=plan.builder_id,
            launch_plan_sha256=plan.sha256,
        )

    def verify(self, key: CompileCacheKey, receipt_path: str | Path) -> Path:
        key.validate()
        receipt = CompileCacheReceipt.load(receipt_path)
        if receipt.key_sha256 != key.sha256:
            raise CompileCacheForeignIdentityError(
                "cache receipt belongs to another key"
            )
        object_path = self.objects / receipt.content_sha256
        if object_path.is_symlink() or not object_path.is_dir():
            raise CompileCacheCorruptionError("cache object is missing")
        try:
            inventory = self._inventory(object_path)
        except (OSError, ValueError) as error:
            raise CompileCacheCorruptionError(str(error)) from error
        if inventory != receipt.files:
            raise CompileCacheCorruptionError("cache object content changed")
        self._verify_immutable_object(object_path)
        return object_path

    def seed_overlay(
        self,
        overlay: CompileCacheOverlay,
        *,
        receipt_path: str | Path,
        expected_receipt_sha256: str,
    ) -> CompileCacheReceipt:
        """Copy one verified immutable base into a process-private overlay."""

        overlay.validate()
        if not overlay.path.is_relative_to(self.overlays):
            raise ValueError("compile-cache overlay belongs to another store")
        try:
            receipt = CompileCacheReceipt.load(receipt_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise CompileCacheCorruptionError(
                "cache receipt is missing or invalid",
                reason_code="invalid_cache_receipt",
            ) from error
        if receipt.receipt_sha256 != expected_receipt_sha256:
            raise CompileCacheForeignIdentityError(
                "selected cache receipt differs from the launch plan"
            )
        if receipt.key_sha256 != overlay.key.sha256:
            raise CompileCacheForeignIdentityError(
                "selected cache receipt belongs to another key"
            )
        if (
            receipt.builder_id != SGLANG_FIRST_PARTY_COMPILE_BUILDER
            or receipt.launch_plan_sha256 is None
        ):
            raise CompileCacheForeignIdentityError(
                "selected cache receipt was not produced by the release builder"
            )
        object_path = self.verify(overlay.key, receipt_path)
        for item in receipt.files:
            source = object_path / item.relative_path
            target = overlay.path / item.relative_path
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source, target)
            os.chmod(target, 0o600)
        if self._inventory(overlay.path) != receipt.files:
            raise CompileCacheCorruptionError(
                "private cache overlay differs from the verified base"
            )
        # Detect source mutation that raced the copy.  The copied inventory is
        # also checked, so no byte can be imported solely on caller authority.
        self.verify(overlay.key, receipt_path)
        return receipt

    def write_attempt(self, receipt: CompileCacheAttemptReceipt) -> Path:
        receipt.validate()
        path = self.attempts / f"{receipt.sha256}.json"
        _publish_json(path, asdict(receipt))
        _publish_text(path.with_name(f"{path.name}.sha256"), receipt.sha256)
        return path


def _error_digest(error: BaseException) -> str:
    return hashlib.sha256(
        f"{type(error).__module__}.{type(error).__qualname__}:{error}".encode()
    ).hexdigest()


def _failure_code(error: BaseException, fallback: str) -> str:
    if isinstance(error, CompileCacheCorruptionError):
        return error.reason_code
    return fallback


class CompileCacheLaunchSession:
    """A verified base plus the only writable cache tree for one process."""

    def __init__(
        self,
        *,
        plan: CompileCacheLaunchPlan,
        cache: ImmutableCompileCache,
        overlay: CompileCacheOverlay,
        base_receipt_sha256: str | None,
    ) -> None:
        self.plan = plan
        self.cache = cache
        self.overlay = overlay
        self.base_receipt_sha256 = base_receipt_sha256
        self._terminal = False

    def _attempt(
        self,
        *,
        state: str,
        finished_ns: int | None,
        result_receipt_sha256: str | None = None,
        failure_code: str | None = None,
        failure_detail_sha256: str | None = None,
    ) -> CompileCacheAttemptReceipt:
        return CompileCacheAttemptReceipt(
            schema_version=1,
            kind="compile_cache_attempt_receipt",
            plan_sha256=self.plan.sha256,
            key_sha256=self.plan.key.sha256,
            attempt_id=self.overlay.attempt_id,
            process_id=self.overlay.process_id,
            state=state,
            started_ns=self.overlay.started_ns,
            finished_ns=finished_ns,
            overlay_name=self.overlay.path.name,
            base_receipt_sha256=self.base_receipt_sha256,
            result_receipt_sha256=result_receipt_sha256,
            failure_code=failure_code,
            failure_detail_sha256=failure_detail_sha256,
            environment=tuple(sorted(_CACHE_ENVIRONMENT.items())),
        )

    def environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return an environment whose compile writers all target this overlay."""

        if self._terminal:
            raise RuntimeError("compile-cache launch session is already terminal")
        environment = dict(os.environ if base is None else base)
        for name, relative in _CACHE_ENVIRONMENT.items():
            target = self.overlay.path / relative
            target.mkdir(mode=0o700, parents=True, exist_ok=True)
            if target.is_symlink() or not target.is_dir():
                raise ValueError("compile-cache environment path is not a directory")
            if not target.is_relative_to(self.overlay.path):
                raise ValueError("compile-cache environment escaped its overlay")
            environment[name] = str(target)
        return environment

    def complete(self) -> tuple[Path, Path, Path]:
        """Measure and seal local files; no caller can declare cache success."""

        if self._terminal:
            raise RuntimeError("compile-cache launch session is already terminal")
        try:
            object_path, receipt_path = self.cache.seal_launch_overlay(
                self.overlay,
                plan=self.plan,
            )
            receipt = CompileCacheReceipt.load(receipt_path)
            attempt_path = self.cache.write_attempt(
                self._attempt(
                    state="complete",
                    finished_ns=time.monotonic_ns(),
                    result_receipt_sha256=receipt.receipt_sha256,
                )
            )
        except BaseException as error:
            self._terminal = True
            self.cache.write_attempt(
                self._attempt(
                    state="failed",
                    finished_ns=time.monotonic_ns(),
                    failure_code=_failure_code(error, "cache_seal_failed"),
                    failure_detail_sha256=_error_digest(error),
                )
            )
            raise
        self._terminal = True
        return object_path, receipt_path, attempt_path

    def fail(self, error: BaseException, *, reason_code: str) -> Path:
        if self._terminal:
            raise RuntimeError("compile-cache launch session is already terminal")
        if not _SAFE_COMPONENT.fullmatch(reason_code):
            raise ValueError("compile-cache failure reason is unsafe")
        self._terminal = True
        return self.cache.write_attempt(
            self._attempt(
                state="failed",
                finished_ns=time.monotonic_ns(),
                failure_code=reason_code,
                failure_detail_sha256=_error_digest(error),
            )
        )


def preflight_compile_cache_launch(
    plan: CompileCacheLaunchPlan,
) -> CompileCacheReceipt | None:
    """Revalidate a reuse base without creating a store, overlay, or receipt.

    Build plans have no base authority to inspect and therefore return ``None``.
    Reuse plans return the exact release-builder receipt only after reopening its
    sidecar and immutable object from disk.  Callers must repeat this check at
    every process/execution boundary; a serialized plan is not cache authority.
    """

    if type(plan) is not CompileCacheLaunchPlan:
        raise TypeError("compile-cache preflight requires an exact launch plan")
    plan.validate()
    if plan.cache_mode == "build":
        return None
    if plan.base_receipt_path is None or plan.base_receipt_sha256 is None:
        raise AssertionError("validated reuse plan lost its receipt")
    cache = ImmutableCompileCache._open_existing_read_only(plan.cache_root)
    receipt_path = Path(plan.base_receipt_path)
    if receipt_path.parent != cache.receipts:
        raise CompileCacheForeignIdentityError(
            "selected cache receipt is not a canonical receipt in this store"
        )
    try:
        receipt = CompileCacheReceipt.load(receipt_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CompileCacheCorruptionError(
            "cache receipt is missing or invalid",
            reason_code="invalid_cache_receipt",
        ) from error
    if (
        receipt.receipt_sha256 != plan.base_receipt_sha256
        or receipt_path.name != f"{receipt.receipt_sha256}.json"
    ):
        raise CompileCacheForeignIdentityError(
            "selected cache receipt differs from the launch plan"
        )
    if receipt.key_sha256 != plan.key.sha256:
        raise CompileCacheForeignIdentityError(
            "selected cache receipt belongs to another key"
        )
    if (
        receipt.builder_id != SGLANG_FIRST_PARTY_COMPILE_BUILDER
        or receipt.launch_plan_sha256 is None
    ):
        raise CompileCacheForeignIdentityError(
            "selected cache receipt was not produced by the release builder"
        )
    try:
        cache.verify(plan.key, receipt_path)
    except CompileCacheCorruptionError:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CompileCacheCorruptionError(
            "cache receipt or immutable object changed during preflight"
        ) from error
    # Close the receipt/object race window as far as a path-based filesystem
    # contract permits.  The launch process repeats verification before import.
    try:
        reopened = CompileCacheReceipt.load(receipt_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CompileCacheCorruptionError(
            "cache receipt changed during preflight",
            reason_code="invalid_cache_receipt",
        ) from error
    if reopened != receipt:
        raise CompileCacheCorruptionError("cache receipt changed during preflight")
    return receipt


def start_compile_cache_launch(
    plan: CompileCacheLaunchPlan,
    *,
    process_id: int | None = None,
    attempt_id: str | None = None,
) -> CompileCacheLaunchSession:
    """Verify a plan/base and create a private overlay before SGLang import."""

    plan.validate()
    cache = ImmutableCompileCache(plan.cache_root)
    selected_process_id = os.getpid() if process_id is None else process_id
    selected_attempt_id = (
        f"launch-{uuid.uuid4().hex}" if attempt_id is None else attempt_id
    )
    overlay = cache.create_overlay(
        plan.key,
        process_id=selected_process_id,
        attempt_id=selected_attempt_id,
    )
    session = CompileCacheLaunchSession(
        plan=plan,
        cache=cache,
        overlay=overlay,
        base_receipt_sha256=plan.base_receipt_sha256,
    )
    cache.write_attempt(session._attempt(state="started", finished_ns=None))
    try:
        if plan.cache_mode == "reuse":
            if plan.base_receipt_path is None or plan.base_receipt_sha256 is None:
                raise AssertionError("validated reuse plan lost its receipt")
            cache.seed_overlay(
                overlay,
                receipt_path=plan.base_receipt_path,
                expected_receipt_sha256=plan.base_receipt_sha256,
            )
        cache.write_attempt(
            session._attempt(state="ready", finished_ns=time.monotonic_ns())
        )
    except BaseException as error:
        session.fail(
            error,
            reason_code=_failure_code(error, "cache_prepare_failed"),
        )
        raise
    return session
