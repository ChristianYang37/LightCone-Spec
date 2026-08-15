"""Content-addressed, immutable compile-cache contracts.

The cache deliberately separates reusable read-only objects from a private
writable overlay owned by one process/attempt.  A completed object is selected
only through a content-bound receipt; directory discovery is never evidence
that a cache entry is safe to reuse.

This module also freezes the *future* compile-only assignment envelope.  The
current release intentionally cannot execute that envelope: it has no exact
prewarm/finalization implementation or atomic result-pointer replay.  A valid
typed envelope is therefore diagnostic input only and is deterministically
blocked before a cache store, overlay, model process, or GPU state is created.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NoReturn, Self

from lightcone_spec import PINNED_SGLANG_TREE

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

# These values are release data, not launch-time assertions supplied by a
# caller.  The manifest digest is over canonical JSON (the same encoding used
# for every contract in this module); the patch digest is over the exact mail
# patch bytes registered by that manifest.
PINNED_SGLANG_PATCH_MANIFEST_SHA256 = (
    "7bfb2ea4f1497dd782a70a7afcad0857495b2871d80ad8ac858b2cb81e32ef7b"
)
PINNED_SGLANG_PATCH_SHA256 = (
    "8b0d05ba862fb0a9ec02092a35990ed487d56e294eb7b10d210c67ca1e84b163"
)
SGLANG_FIRST_PARTY_COMPILE_BUILDER = "lightcone_spec.sglang_bridge.launch_server.v1"
RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE = (
    "release_compile_assignment_contract_unavailable"
)

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
        allow_nan=False,
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

# These protocol identities describe the minimum missing terminal boundary.
# They do not claim that a runner or result pointer exists in this release.
COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "compile_only_graceful_shutdown_protocol",
        "required_acknowledgement": (
            "assignment_sha256",
            "compile_plan_sha256",
            "prewarm_manifest_sha256",
            "attempt_id",
            "process_id",
            "shutdown_requested_ns",
            "process_exited_ns",
            "exit_code",
            "active_requests",
            "queued_requests",
            "final_cache_receipt_sha256",
        ),
        "success_requires_zero_active_and_queued_requests": True,
        "provider_summary_is_not_authority": True,
    }
)
COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "compile_only_atomic_result_pointer_protocol",
        "required_path_bindings": (
            "assignment_manifest",
            "prewarm_manifest",
            "attempt_receipt",
            "graceful_shutdown_receipt",
            "final_cache_receipt",
            "immutable_cache_object_manifest",
        ),
        "each_binding_requires": ("absolute_path", "raw_sha256", "size"),
        "publication": "atomic_no_replace_with_exact_sidecar",
        "resume": "reopen_and_revalidate_every_bound_path",
        "serialized_summary_is_not_authority": True,
    }
)
COMPILE_ONLY_ASSIGNMENT_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "compile_only_assignment_protocol",
        "required_authority": (
            "registry_runtime_split",
            "physical_assignment",
            "experiment_budget",
            "budget_materialization_authority",
            "inventory_and_source_receipt",
            "exact_gpu_uuids_and_host",
            "compile_plan_and_key",
            "model_revisions",
            "tp_context_concurrency_graph_buckets",
            "deterministic_prewarm_payloads",
            "graceful_shutdown_acknowledgement",
            "atomic_result_pointer",
        ),
        "graceful_shutdown_protocol_sha256": (
            COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256
        ),
        "result_pointer_protocol_sha256": (COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256),
        "release_execution_available": False,
        "blocked_before_mutation_reason": (
            RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE
        ),
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


def _stable_regular_file_bytes(path: Path, *, label: str) -> bytes:
    """Read one absolute regular file without following or racing a symlink."""

    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise ValueError(f"{label} path must be absolute and normalized")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a readable regular file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read()
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_size,
            row.st_mtime_ns,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or identity(before) != identity(after)
            or identity(after) != identity(current)
            or len(body) != after.st_size
        ):
            raise ValueError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


def _strict_json_object(body: bytes, *, label: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"{label} contains non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error

    def require_finite(item: object) -> None:
        if type(item) is float and not math.isfinite(item):
            raise ValueError(f"{label} contains a non-finite JSON number")
        if type(item) is list:
            for child in item:
                require_finite(child)
        elif type(item) is dict:
            for child in item.values():
                require_finite(child)

    require_finite(value)
    if type(value) is not dict:
        raise TypeError(f"{label} must contain one JSON object")
    if body != _canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must use canonical JSON encoding")
    return value


def _load_canonical_json_with_sidecar(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, object], str]:
    body = _stable_regular_file_bytes(path, label=label)
    value = _strict_json_object(body, label=label)
    semantic_sha256 = _content_sha256(value)
    sidecar = _stable_regular_file_bytes(
        Path(f"{path}.sha256"), label=f"{label} SHA-256 sidecar"
    )
    if sidecar != f"{semantic_sha256}\n".encode("ascii"):
        raise ValueError(f"{label} SHA-256 sidecar differs from content")
    return value, semantic_sha256


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
        if self.drafter_revision is not None and not _GIT_OBJECT.fullmatch(
            self.drafter_revision
        ):
            raise ValueError("drafter_revision must be an immutable Git SHA")
        if not _GIT_OBJECT.fullmatch(self.target_revision):
            raise ValueError("target_revision must be an immutable Git SHA")
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


def validate_compile_runtime_toolchain(
    key: CompileCacheKey,
    *,
    python_version: object,
    torch_version: object,
    triton_version: object,
    torch_cuda_version: object,
    nvcc_cuda_version: object,
    driver_version: object,
    gpu_model: object,
    sm_architecture: object,
) -> None:
    """Match one observed launch environment to its compile-cache key."""

    key.validate()
    observed = (
        python_version,
        torch_version,
        triton_version,
        torch_cuda_version,
        nvcc_cuda_version,
        driver_version,
        gpu_model,
        sm_architecture,
    )
    expected = (
        key.python_version,
        key.torch_version,
        key.triton_version,
        key.cuda_version,
        key.cuda_version,
        key.driver_version,
        key.gpu_model,
        key.sm_architecture,
    )
    if observed != expected:
        raise ValueError("compile-cache key differs from exact runtime toolchain")


def validate_compile_key_for_run_config(
    plan: CompileCacheLaunchPlan,
    *,
    config: object,
) -> None:
    """Bind one cache identity to an exact strict RunConfig without orchestration."""

    from lightcone_spec.config.schema import RunConfig

    if type(plan) is not CompileCacheLaunchPlan or type(config) is not RunConfig:
        raise TypeError("compile-cache binding requires exact plan and RunConfig")
    plan.validate()
    expected_drafter = (
        None if config.method == "target_only" else config.model.drafter_revision
    )
    key = plan.key
    if (
        key.source_sha256 != PINNED_SGLANG_COMPILE_SOURCE_SHA256
        or key.target_revision != config.model.target_revision
        or key.drafter_revision != expected_drafter
        or key.tensor_parallel_size != config.runtime.tensor_parallel_size
        or key.context_limit != config.runtime.context_length
        or key.max_running_requests != config.runtime.max_running_requests
    ):
        raise ValueError("compile-cache key differs from the exact RunConfig")


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
    def from_dict(cls, raw: object) -> Self:
        if type(raw) is not dict or set(raw) != {
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
        payload = dict(raw)
        key = CompileCacheKey.from_dict(payload.pop("key"))
        value = cls(**payload, key=key)
        value.validate()
        return value

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
        selected_path: str | None = None
        selected_sha256: str | None = None
        if cache_mode == "reuse":
            if base_receipt_path is None:
                raise ValueError("cache reuse requires an explicit receipt")
            cache = ImmutableCompileCache._open_existing_read_only(root)
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
        requested = Path(path)
        if requested.is_symlink():
            raise ValueError("compile-cache plan cannot be a symlink")
        resolved = requested.resolve(strict=False)
        raw, semantic_sha256 = _load_canonical_json_with_sidecar(
            resolved,
            label="compile-cache plan",
        )
        value = cls.from_dict(raw)
        if semantic_sha256 != value.sha256:
            raise ValueError("compile-cache plan semantic identity differs")
        return value


@dataclass(frozen=True)
class CompileOnlyPrewarmPayload:
    """One exact, ordered input used by a future graph/JIT prewarm lifecycle."""

    request_id: str
    graph_bucket: int
    input_token_ids: tuple[int, ...]
    requested_output_tokens: int
    sampling_seed: int

    def validate(self) -> None:
        if not _SAFE_COMPONENT.fullmatch(self.request_id):
            raise ValueError("compile prewarm request ID is unsafe")
        for name in ("graph_bucket", "requested_output_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"compile prewarm {name} must be positive")
        if type(self.sampling_seed) is not int or self.sampling_seed < 0:
            raise ValueError("compile prewarm sampling seed must be non-negative")
        if not self.input_token_ids or any(
            type(token_id) is not int or token_id < 0
            for token_id in self.input_token_ids
        ):
            raise ValueError("compile prewarm input token IDs are invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "request_id": self.request_id,
            "graph_bucket": self.graph_bucket,
            "input_token_ids": list(self.input_token_ids),
            "requested_output_tokens": self.requested_output_tokens,
            "sampling_seed": self.sampling_seed,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        if type(raw) is not dict or set(raw) != {
            "request_id",
            "graph_bucket",
            "input_token_ids",
            "requested_output_tokens",
            "sampling_seed",
        }:
            raise ValueError("compile prewarm payload fields differ from schema")
        token_ids = raw.get("input_token_ids")
        if type(token_ids) is not list:
            raise TypeError("compile prewarm input token IDs must be a JSON array")
        value = cls(
            request_id=raw.get("request_id"),
            graph_bucket=raw.get("graph_bucket"),
            input_token_ids=tuple(token_ids),
            requested_output_tokens=raw.get("requested_output_tokens"),
            sampling_seed=raw.get("sampling_seed"),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class CompileOnlyPrewarmManifest:
    """Exact payload order for a future release-owned compile-only attempt."""

    schema_version: int
    kind: str
    model_lock_sha256: str
    sampling_profile_sha256: str
    payloads: tuple[CompileOnlyPrewarmPayload, ...]

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "compile_only_prewarm_manifest"
        ):
            raise ValueError("compile prewarm manifest schema is unsupported")
        _require_sha256("compile prewarm model lock", self.model_lock_sha256)
        _require_sha256(
            "compile prewarm sampling profile", self.sampling_profile_sha256
        )
        if not self.payloads or any(
            type(payload) is not CompileOnlyPrewarmPayload for payload in self.payloads
        ):
            raise TypeError("compile prewarm manifest requires exact payloads")
        for payload in self.payloads:
            payload.validate()
        request_ids = tuple(payload.request_id for payload in self.payloads)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("compile prewarm request IDs must be unique")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "model_lock_sha256": self.model_lock_sha256,
            "sampling_profile_sha256": self.sampling_profile_sha256,
            "payloads": [payload.to_dict() for payload in self.payloads],
        }

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        if type(raw) is not dict or set(raw) != {
            "schema_version",
            "kind",
            "model_lock_sha256",
            "sampling_profile_sha256",
            "payloads",
        }:
            raise ValueError("compile prewarm manifest fields differ from schema")
        payloads = raw.get("payloads")
        if type(payloads) is not list:
            raise TypeError("compile prewarm payloads must be a JSON array")
        value = cls(
            schema_version=raw.get("schema_version"),
            kind=raw.get("kind"),
            model_lock_sha256=raw.get("model_lock_sha256"),
            sampling_profile_sha256=raw.get("sampling_profile_sha256"),
            payloads=tuple(
                CompileOnlyPrewarmPayload.from_dict(payload) for payload in payloads
            ),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class CompileOnlyAssignmentContract:
    """Complete future compile assignment shape, never current launch authority.

    The object deliberately carries all fields required by the registered
    future boundary.  :func:`require_release_compile_only_assignment` still
    rejects it unconditionally because this release cannot create or replay the
    required graceful-shutdown receipt and atomic result pointer.
    """

    schema_version: int
    kind: str
    assignment_protocol_sha256: str
    cell_id: str
    registry_sha256: str
    runtime_sha256: str
    split_sha256: str
    physical_assignment_sha256: str
    experiment_budget_sha256: str
    budget_materialization_authority_sha256: str
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    gpu_uuids: tuple[str, ...]
    host_id: str
    fixed_instance_gpu_count: int
    compile_cache_plan: CompileCacheLaunchPlan
    prewarm_manifest: CompileOnlyPrewarmManifest
    graceful_shutdown_protocol_sha256: str
    result_pointer_protocol_sha256: str
    result_pointer_path: str

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "compile_only_assignment_contract"
        ):
            raise ValueError("compile-only assignment schema is unsupported")
        if self.assignment_protocol_sha256 != (COMPILE_ONLY_ASSIGNMENT_PROTOCOL_SHA256):
            raise ValueError("compile-only assignment uses another release protocol")
        for name in (
            "cell_id",
            "registry_sha256",
            "runtime_sha256",
            "split_sha256",
            "physical_assignment_sha256",
            "experiment_budget_sha256",
            "budget_materialization_authority_sha256",
            "inventory_sha256",
            "inventory_source_receipt_sha256",
        ):
            _require_sha256(f"compile-only {name}", getattr(self, name))
        if (
            not self.gpu_uuids
            or len(self.gpu_uuids) != len(set(self.gpu_uuids))
            or any(
                type(gpu_uuid) is not str
                or not gpu_uuid.strip()
                or "\n" in gpu_uuid
                or "\r" in gpu_uuid
                for gpu_uuid in self.gpu_uuids
            )
        ):
            raise ValueError("compile-only GPU UUIDs must be non-empty and unique")
        _require_text("compile-only host ID", self.host_id)
        if type(
            self.fixed_instance_gpu_count
        ) is not int or self.fixed_instance_gpu_count < len(self.gpu_uuids):
            raise ValueError(
                "compile-only fixed-instance GPU count cannot be smaller than its gang"
            )
        if type(self.compile_cache_plan) is not CompileCacheLaunchPlan:
            raise TypeError("compile-only assignment requires an exact cache plan")
        self.compile_cache_plan.validate()
        if self.compile_cache_plan.cache_mode != "build":
            raise ValueError("compile-only assignments can build only a new cache base")
        if type(self.prewarm_manifest) is not CompileOnlyPrewarmManifest:
            raise TypeError(
                "compile-only assignment requires an exact prewarm manifest"
            )
        self.prewarm_manifest.validate()
        key = self.compile_cache_plan.key
        manifest_buckets = tuple(
            sorted({payload.graph_bucket for payload in self.prewarm_manifest.payloads})
        )
        if manifest_buckets != key.graph_buckets:
            raise ValueError(
                "compile prewarm payloads must exactly cover the cache graph buckets"
            )
        if any(
            len(payload.input_token_ids) + payload.requested_output_tokens
            > key.context_limit
            for payload in self.prewarm_manifest.payloads
        ):
            raise ValueError("compile prewarm payload exceeds the cache context limit")
        if self.graceful_shutdown_protocol_sha256 != (
            COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256
        ):
            raise ValueError("compile-only assignment uses another shutdown protocol")
        if self.result_pointer_protocol_sha256 != (
            COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256
        ):
            raise ValueError(
                "compile-only assignment uses another result-pointer protocol"
            )
        _strict_absolute_path(
            "compile-only result pointer path", self.result_pointer_path
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "assignment_protocol_sha256": self.assignment_protocol_sha256,
            "cell_id": self.cell_id,
            "registry_sha256": self.registry_sha256,
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "physical_assignment_sha256": self.physical_assignment_sha256,
            "experiment_budget_sha256": self.experiment_budget_sha256,
            "budget_materialization_authority_sha256": (
                self.budget_materialization_authority_sha256
            ),
            "inventory_sha256": self.inventory_sha256,
            "inventory_source_receipt_sha256": (self.inventory_source_receipt_sha256),
            "gpu_uuids": list(self.gpu_uuids),
            "host_id": self.host_id,
            "fixed_instance_gpu_count": self.fixed_instance_gpu_count,
            "compile_cache_plan": asdict(self.compile_cache_plan),
            "prewarm_manifest": self.prewarm_manifest.to_dict(),
            "graceful_shutdown_protocol_sha256": (
                self.graceful_shutdown_protocol_sha256
            ),
            "result_pointer_protocol_sha256": (self.result_pointer_protocol_sha256),
            "result_pointer_path": self.result_pointer_path,
        }

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        fields = {
            "schema_version",
            "kind",
            "assignment_protocol_sha256",
            "cell_id",
            "registry_sha256",
            "runtime_sha256",
            "split_sha256",
            "physical_assignment_sha256",
            "experiment_budget_sha256",
            "budget_materialization_authority_sha256",
            "inventory_sha256",
            "inventory_source_receipt_sha256",
            "gpu_uuids",
            "host_id",
            "fixed_instance_gpu_count",
            "compile_cache_plan",
            "prewarm_manifest",
            "graceful_shutdown_protocol_sha256",
            "result_pointer_protocol_sha256",
            "result_pointer_path",
        }
        if type(raw) is not dict or set(raw) != fields:
            raise ValueError("compile-only assignment fields differ from schema")
        gpu_uuids = raw.get("gpu_uuids")
        if type(gpu_uuids) is not list:
            raise TypeError("compile-only GPU UUIDs must be a JSON array")
        value = cls(
            schema_version=raw.get("schema_version"),
            kind=raw.get("kind"),
            assignment_protocol_sha256=raw.get("assignment_protocol_sha256"),
            cell_id=raw.get("cell_id"),
            registry_sha256=raw.get("registry_sha256"),
            runtime_sha256=raw.get("runtime_sha256"),
            split_sha256=raw.get("split_sha256"),
            physical_assignment_sha256=raw.get("physical_assignment_sha256"),
            experiment_budget_sha256=raw.get("experiment_budget_sha256"),
            budget_materialization_authority_sha256=raw.get(
                "budget_materialization_authority_sha256"
            ),
            inventory_sha256=raw.get("inventory_sha256"),
            inventory_source_receipt_sha256=raw.get("inventory_source_receipt_sha256"),
            gpu_uuids=tuple(gpu_uuids),
            host_id=raw.get("host_id"),
            fixed_instance_gpu_count=raw.get("fixed_instance_gpu_count"),
            compile_cache_plan=CompileCacheLaunchPlan.from_dict(
                raw.get("compile_cache_plan")
            ),
            prewarm_manifest=CompileOnlyPrewarmManifest.from_dict(
                raw.get("prewarm_manifest")
            ),
            graceful_shutdown_protocol_sha256=raw.get(
                "graceful_shutdown_protocol_sha256"
            ),
            result_pointer_protocol_sha256=raw.get("result_pointer_protocol_sha256"),
            result_pointer_path=raw.get("result_pointer_path"),
        )
        value.validate()
        return value

    def write(self, path: str | Path) -> Path:
        self.validate()
        destination = _strict_absolute_path("compile-only assignment", str(path))
        if destination.parent.is_symlink() or not destination.parent.is_dir():
            raise ValueError("compile-only assignment parent must be a directory")
        _publish_json(destination, self.to_dict())
        _publish_text(Path(f"{destination}.sha256"), self.sha256)
        if self.load(destination) != self:
            raise RuntimeError("compile-only assignment changed during publication")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _strict_absolute_path("compile-only assignment", str(path))
        raw, semantic_sha256 = _load_canonical_json_with_sidecar(
            source, label="compile-only assignment"
        )
        value = cls.from_dict(raw)
        if semantic_sha256 != value.sha256:
            raise ValueError("compile-only assignment semantic digest differs")
        return value


class CompileOnlyAssignmentUnavailableError(RuntimeError):
    """A compile-only request reached a release with no terminal authority."""

    reason_code = RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE


def require_release_compile_only_assignment(
    contract: CompileOnlyAssignmentContract | None = None,
) -> NoReturn:
    """Deterministically block compile-only work before any runtime mutation."""

    if contract is not None:
        if type(contract) is not CompileOnlyAssignmentContract:
            raise TypeError("compile-only gate requires an exact assignment contract")
        contract.validate()
    raise CompileOnlyAssignmentUnavailableError(
        "compile-only execution is BLOCKED: "
        f"{RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE}"
    )


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
                # The object directory is renamed only after every file and
                # directory has been flushed and made read-only.  Its atomic
                # appearance is therefore the publication boundary; a later
                # contender may legitimately acquire the short-lived claim
                # before this waiter discards its private staging directory.
                if object_path.exists():
                    _fsync_directory(self.objects)
                    self._discard_staging_directory(temporary)
                    return
                if not claim.exists():
                    break
                time.sleep(0.01)
            if object_path.exists():
                _fsync_directory(self.objects)
                self._discard_staging_directory(temporary)
                return
            self._discard_staging_directory(temporary)
            raise CompileCacheCorruptionError(
                "compile-cache publication is incomplete",
                reason_code="incomplete_cache_publication",
            )
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
        release_builder_receipt: bool,
    ) -> None:
        self.plan = plan
        self.cache = cache
        self.overlay = overlay
        self.base_receipt_sha256 = base_receipt_sha256
        self._release_builder_receipt = release_builder_receipt
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
            if self._release_builder_receipt:
                object_path, receipt_path = self.cache.seal_launch_overlay(
                    self.overlay,
                    plan=self.plan,
                )
            else:
                object_path, receipt_path = self.cache.seal_overlay(self.overlay)
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
    _release_builder_receipt: bool = True,
) -> CompileCacheLaunchSession:
    """Verify a plan/base and create a private overlay before SGLang import."""

    plan.validate()
    if not isinstance(_release_builder_receipt, bool):
        raise TypeError("compile-cache receipt attribution selector must be boolean")
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
        release_builder_receipt=_release_builder_receipt,
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
