"""Registered server execution policy for formal GPU evidence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Self

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE

ExecutionRole = Literal["target_reference", "speculative"]

FIXED_ADDRESS_GRAPH_EXECUTION_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "schema_version": 1,
            "kind": "fixed_address_publication_graph_execution_policy",
            "pinned_sglang_commit": PINNED_SGLANG_COMMIT,
            "patched_sglang_tree": PINNED_SGLANG_TREE,
            "graph_batch_sizes": [1],
            "fixed_addresses_required": True,
            "eager_fallback_allowed": False,
            "blocking_d2h_allowed": False,
            "host_synchronization_allowed": False,
            "formal_authority": "dynamic_native_hot_path_gpu_proof",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def _absolute_lexical_path(path: str | Path) -> Path:
    """Normalize path syntax without resolving a possibly-symlinked leaf."""

    return Path(os.path.abspath(os.fspath(path)))


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate execution-policy field: {key}")
        value[key] = item
    return value


def _reject_constant(raw: str) -> None:
    raise ValueError(f"non-finite execution-policy value: {raw}")


def _read_regular(path: Path, *, label: str) -> bytes:
    """Read one stable, single-link regular file without following its leaf."""

    if not path.is_absolute() or _absolute_lexical_path(path) != path:
        raise ValueError(f"{label} path must be absolute and resolved")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is not a readable regular file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} is not a single-link regular file")
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b"".join(chunks)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if (
            len(body) != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
            or current.st_dev != before.st_dev
            or current.st_ino != before.st_ino
            or current.st_size != before.st_size
            or current.st_mtime_ns != before.st_mtime_ns
            or current.st_ctime_ns != before.st_ctime_ns
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


def _publish_immutable(path: Path, body: bytes, *, label: str) -> None:
    """Create once, or prove that the existing immutable bytes are identical."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError:
        if _read_regular(path, label=label) != body:
            raise ValueError(f"{label} is immutable; choose a new output path")
        return
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ControlledExecutionPolicy:
    """GPU-tested exactness controls shared by both endpoint roles.

    The field set, defaults, canonical serialization, and SHA-256 identity are
    intentionally identical to the policy exercised by ``main@505f4d9``.  File
    publication and loading are stricter here so a formal execution bundle can
    bind stable raw bytes instead of trusting caller-authored values.
    """

    schema_version: int = 2
    context_length: int = 40960
    random_seed: int = 1
    disable_radix_cache: bool = True
    disable_cuda_graph: bool = True
    target_reference_disable_overlap_schedule: bool = True
    speculative_disable_overlap_schedule: bool = False
    enable_deterministic_inference: bool = False
    incremental_streaming_output: bool = False

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("execution policy must use schema version 2")
        if type(self.context_length) is not int or self.context_length != 40960:
            raise ValueError("execution context must equal the locked model limit")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("execution random seed must be a non-negative integer")
        boolean_fields = (
            self.disable_radix_cache,
            self.disable_cuda_graph,
            self.target_reference_disable_overlap_schedule,
            self.speculative_disable_overlap_schedule,
            self.enable_deterministic_inference,
            self.incremental_streaming_output,
        )
        if any(type(value) is not bool for value in boolean_fields):
            raise ValueError("execution switches must be booleans")
        if not self.disable_radix_cache or not self.disable_cuda_graph:
            raise ValueError(
                "formal DFlash exactness requires radix cache and CUDA graph disabled"
            )
        if not self.target_reference_disable_overlap_schedule:
            raise ValueError("target reference must disable overlap scheduling")
        if self.speculative_disable_overlap_schedule:
            raise ValueError("formal speculative runs must retain overlap scheduling")
        if self.enable_deterministic_inference:
            raise ValueError("unregistered deterministic execution variant")
        if self.incremental_streaming_output:
            raise ValueError("formal evidence requires complete output token IDs")

    def to_dict(self) -> dict[str, int | bool]:
        self.validate()
        return asdict(self)

    @property
    def sha256(self) -> str:
        body = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(body).hexdigest()

    def overlap_disabled(self, *, role: ExecutionRole) -> bool:
        """Return the registered overlap switch for one endpoint role."""

        self.validate()
        if role == "target_reference":
            return self.target_reference_disable_overlap_schedule
        if role == "speculative":
            return self.speculative_disable_overlap_schedule
        raise ValueError(f"unknown execution-policy role: {role}")

    def server_info_fields(self, *, role: ExecutionRole) -> dict[str, int | bool]:
        """Return the exact public ``/server_info`` values to attest."""

        self.validate()
        return {
            "context_length": self.context_length,
            "random_seed": self.random_seed,
            "disable_radix_cache": self.disable_radix_cache,
            "disable_cuda_graph": self.disable_cuda_graph,
            "disable_overlap_schedule": self.overlap_disabled(role=role),
            "enable_deterministic_inference": self.enable_deterministic_inference,
            "incremental_streaming_output": self.incremental_streaming_output,
        }

    def validate_server_info(self, server_info: object, *, role: ExecutionRole) -> None:
        """Fail closed unless one live server reports the exact registered policy."""

        if type(server_info) is not dict:
            raise TypeError("server_info must be an object")
        for name, expected in self.server_info_fields(role=role).items():
            value = server_info.get(name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"server execution policy mismatch: {name}")

    def write(self, path: str | Path) -> None:
        """Publish canonical policy bytes without overwriting any existing file."""

        output = _absolute_lexical_path(path)
        body = (
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        _publish_immutable(output, body, label="execution policy")
        _publish_immutable(
            Path(f"{output}.sha256"),
            f"{self.sha256}\n".encode("ascii"),
            label="execution policy sidecar",
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(asdict(cls())):
            raise ValueError("execution policy fields do not match schema-v2")
        policy = cls(**value)
        policy.validate()
        return policy

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load exact canonical bytes and their exact semantic sidecar."""

        source = _absolute_lexical_path(path)
        body = _read_regular(source, label="execution policy")
        try:
            value = json.loads(
                body.decode("utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("execution policy is not strict UTF-8 JSON") from error
        policy = cls.from_dict(value)
        canonical = (
            json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if body != canonical:
            raise ValueError("execution policy bytes are not canonical")
        sidecar = _read_regular(
            Path(f"{source}.sha256"), label="execution policy sidecar"
        )
        if sidecar != f"{policy.sha256}\n".encode("ascii"):
            raise ValueError("execution policy sidecar is missing or invalid")
        return policy


@dataclass(frozen=True)
class FixedAddressGraphExecutionPolicy:
    """Explicit graph-enabled policy; dynamic GPU proof remains orthogonal."""

    schema_version: int = 1
    kind: str = "fixed_address_publication_graph_execution_policy"
    protocol_sha256: str = FIXED_ADDRESS_GRAPH_EXECUTION_PROTOCOL_SHA256
    native_runtime_release_capability_sha256: str = ""
    context_length: int = 40960
    random_seed: int = 1
    disable_radix_cache: bool = True
    disable_cuda_graph: bool = False
    target_reference_disable_overlap_schedule: bool = True
    speculative_disable_overlap_schedule: bool = False
    enable_deterministic_inference: bool = False
    incremental_streaming_output: bool = False
    graph_batch_sizes: tuple[int, ...] = (1,)
    fixed_addresses_required: bool = True
    eager_fallback_allowed: bool = False
    blocking_d2h_allowed: bool = False
    host_synchronization_allowed: bool = False

    def validate(self) -> None:
        from lightcone_spec.runtime.readiness import (
            NATIVE_RUNTIME_RELEASE_CAPABILITY,
        )

        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "fixed_address_publication_graph_execution_policy"
            or self.protocol_sha256 != FIXED_ADDRESS_GRAPH_EXECUTION_PROTOCOL_SHA256
        ):
            raise ValueError("fixed-address graph policy schema is unsupported")
        if len(self.native_runtime_release_capability_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.native_runtime_release_capability_sha256
        ):
            raise ValueError("fixed-address graph policy lacks source capability")
        if (
            self.native_runtime_release_capability_sha256
            != NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256
        ):
            raise ValueError(
                "fixed-address graph policy source capability is unregistered"
            )
        if type(self.context_length) is not int or self.context_length != 40960:
            raise ValueError(
                "graph execution context must equal the locked model limit"
            )
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("graph execution random seed must be non-negative")
        boolean_fields = (
            self.disable_radix_cache,
            self.disable_cuda_graph,
            self.target_reference_disable_overlap_schedule,
            self.speculative_disable_overlap_schedule,
            self.enable_deterministic_inference,
            self.incremental_streaming_output,
            self.fixed_addresses_required,
            self.eager_fallback_allowed,
            self.blocking_d2h_allowed,
            self.host_synchronization_allowed,
        )
        if any(type(value) is not bool for value in boolean_fields):
            raise ValueError("graph execution switches must be booleans")
        if (
            not self.disable_radix_cache
            or self.disable_cuda_graph
            or not self.target_reference_disable_overlap_schedule
            or self.speculative_disable_overlap_schedule
            or self.enable_deterministic_inference
            or self.incremental_streaming_output
            or not self.fixed_addresses_required
            or self.eager_fallback_allowed
            or self.blocking_d2h_allowed
            or self.host_synchronization_allowed
            or self.graph_batch_sizes != (1,)
        ):
            raise ValueError(
                "fixed-address graph policy weakens the registered contract"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        value = asdict(self)
        value["graph_batch_sizes"] = list(self.graph_batch_sizes)
        return value

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def overlap_disabled(self, *, role: ExecutionRole) -> bool:
        self.validate()
        if role == "target_reference":
            return self.target_reference_disable_overlap_schedule
        if role == "speculative":
            return self.speculative_disable_overlap_schedule
        raise ValueError(f"unknown execution-policy role: {role}")

    def server_info_fields(self, *, role: ExecutionRole) -> dict[str, object]:
        self.validate()
        return {
            "context_length": self.context_length,
            "random_seed": self.random_seed,
            "disable_radix_cache": self.disable_radix_cache,
            "disable_cuda_graph": self.disable_cuda_graph,
            "disable_overlap_schedule": self.overlap_disabled(role=role),
            "enable_deterministic_inference": self.enable_deterministic_inference,
            "incremental_streaming_output": self.incremental_streaming_output,
            "lightcone_fixed_address_publication_graph": True,
            "lightcone_graph_batch_sizes": list(self.graph_batch_sizes),
            "lightcone_graph_eager_fallback": self.eager_fallback_allowed,
        }

    def validate_server_info(self, server_info: object, *, role: ExecutionRole) -> None:
        if type(server_info) is not dict:
            raise TypeError("server_info must be an object")
        for name, expected in self.server_info_fields(role=role).items():
            value = server_info.get(name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"server graph execution policy mismatch: {name}")

    def write(self, path: str | Path) -> None:
        """Publish canonical graph-policy bytes without overwriting."""

        output = _absolute_lexical_path(path)
        body = (
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        _publish_immutable(output, body, label="fixed-address graph policy")
        _publish_immutable(
            Path(f"{output}.sha256"),
            f"{self.sha256}\n".encode("ascii"),
            label="fixed-address graph policy sidecar",
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = set(asdict(cls()))
        if type(value) is not dict or set(value) != expected:
            raise ValueError("fixed-address graph policy fields differ")
        row = dict(value)
        graph_batch_sizes = row.pop("graph_batch_sizes")
        if type(graph_batch_sizes) is not list:
            raise TypeError("fixed-address graph batch sizes must be an array")
        policy = cls(**row, graph_batch_sizes=tuple(graph_batch_sizes))
        policy.validate()
        return policy

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_lexical_path(path)
        body = _read_regular(source, label="fixed-address graph policy")
        try:
            value = json.loads(
                body.decode("utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "fixed-address graph policy is not strict UTF-8 JSON"
            ) from error
        policy = cls.from_dict(value)
        canonical = (
            json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if body != canonical:
            raise ValueError("fixed-address graph policy bytes are not canonical")
        sidecar = _read_regular(
            Path(f"{source}.sha256"),
            label="fixed-address graph policy sidecar",
        )
        if sidecar != f"{policy.sha256}\n".encode("ascii"):
            raise ValueError("fixed-address graph policy sidecar is invalid")
        return policy
