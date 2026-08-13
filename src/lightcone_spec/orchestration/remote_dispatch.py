"""Fail-closed SSH transport for one host-local dispatch wave.

The coordinator sends only a canonical control request.  Execution bundles are
never copied over this channel: the remote worker must reopen an absolute,
host-local materialization manifest and verify its declared digest.  Routing
details are deliberately kept in :class:`SshHostRoute`, which has no artifact
serialization and is never embedded in a receipt.

This module does not authorize cross-host collectives.  A TP/DP assignment
whose GPU bindings name more than one host is rejected before a subprocess can
be created with the stable ``cross_host_collectives_unvalidated`` reason.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, Self

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,255}\Z")
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")
_SSH_DESTINATION = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_.-]{0,63}@)?"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,252}[A-Za-z0-9])?\Z"
)

REMOTE_HOST_WAVE_COMMAND = (
    "lightcone-spec",
    "execute-dispatch-wave",
    "--host-request-stdin",
)
DEFAULT_STDOUT_LIMIT_BYTES = 64 * 1024
DEFAULT_STDERR_LIMIT_BYTES = 16 * 1024
DEFAULT_REMOTE_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_FLEET_MAX_CONCURRENCY = 8
MAX_REQUEST_BYTES = 256 * 1024
MAX_RECONCILE_EVIDENCE_BYTES = 96 * 1024 * 1024
MAX_RECONCILE_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_RECONCILE_FILE_BYTES = 32 * 1024 * 1024
MAX_RECONCILE_RAW_BYTES = 64 * 1024 * 1024
MAX_RECONCILE_FILE_COUNT = 64
MAX_RECONCILE_ASSIGNMENTS = 64
MAX_MANIFEST_BYTES = 16 * 1024 * 1024

_SSH_IDENTITY_POLICY = {
    "schema_version": 1,
    "kind": "lightcone_ssh_server_identity_policy",
    "config_file": "/dev/null",
    "batch_mode": True,
    "password_authentication": False,
    "interactive_authentication": False,
    "strict_host_key_checking": True,
    "global_known_hosts_disabled": True,
    "agent_forwarding": False,
    "all_forwarding_disabled": True,
    "local_command_disabled": True,
    "connection_attempts": 1,
    "worker_command": list(REMOTE_HOST_WAVE_COMMAND),
}


def canonical_json_bytes(value: object) -> bytes:
    """Encode exact canonical JSON used by the remote stdin protocol."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_envelope_sha256(body: bytes) -> str:
    value = _strict_json_object(body)
    if canonical_json_bytes(value) + b"\n" != body:
        raise ValueError("dispatch receipt envelope is not canonical JSON")
    return canonical_sha256(value)


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lower-case SHA-256")
    return value


def _require_id(name: str, value: object) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical opaque identifier")
    return value


def _require_reason(name: str, value: object) -> str:
    if type(value) is not str or _REASON_CODE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable lower-case reason code")
    return value


def _strict_object(
    name: str,
    value: object,
    fields: frozenset[str],
) -> Mapping[str, Any]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise TypeError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _strict_list(name: str, value: object) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    return value


def _strict_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _optional_sha256(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_sha256(name, value)


def _strict_json_object(body: bytes) -> Mapping[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"remote response contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"remote response contains non-finite value {value}")

    try:
        value = json.loads(
            body,
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("remote response is not UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError("remote response must be a JSON object")
    return value


def _decode_canonical_json_line(
    stdin: bytes,
    *,
    label: str,
    limit_bytes: int,
) -> Mapping[str, Any]:
    """Decode one bounded canonical JSON line without accepting prefixes."""

    if type(stdin) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if not stdin or len(stdin) > limit_bytes:
        raise ValueError(f"{label} is empty or exceeds its byte limit")
    if not stdin.endswith(b"\n") or stdin.endswith(b"\n\n"):
        raise ValueError(f"{label} must contain one canonical JSON line")
    body = stdin[:-1]
    value = _strict_json_object(body)
    if canonical_json_bytes(value) != body:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _require_host_local_path(name: str, value: object) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{name} must be non-empty text")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be single-line text")
    parsed = PurePosixPath(value)
    if not parsed.is_absolute() or str(parsed) != value or ".." in parsed.parts:
        raise ValueError(f"{name} must be an absolute canonical POSIX path")
    return value


def _optional_host_local_path(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_host_local_path(name, value)


def _require_namespace(name: str, value: object) -> str:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be non-empty single-line text")
    return value


class CrossHostCollectivesUnvalidated(ValueError):
    """Raised when one atomic assignment names GPUs on multiple hosts."""

    reason_code = "cross_host_collectives_unvalidated"

    def __init__(self) -> None:
        super().__init__(self.reason_code)


@dataclass(frozen=True, repr=False)
class SshHostRoute:
    """Coordinator-local SSH authority which is intentionally not serializable."""

    host_id: str
    destination: str
    known_hosts_path: str
    agent_socket_path: str
    port: int = 22
    connect_timeout_seconds: int = 30

    def __post_init__(self) -> None:
        _require_id("route host_id", self.host_id)
        if (
            type(self.destination) is not str
            or _SSH_DESTINATION.fullmatch(self.destination) is None
            or self.destination.startswith("-")
        ):
            raise ValueError("SSH destination is not a safe user/host token")
        if type(self.port) is not int or not 1 <= self.port <= 65_535:
            raise ValueError("SSH port must be in [1, 65535]")
        if (
            type(self.connect_timeout_seconds) is not int
            or not 1 <= self.connect_timeout_seconds <= 600
        ):
            raise ValueError("SSH connect timeout must be in [1, 600] seconds")
        _validate_known_hosts(self.known_hosts_path)
        _validate_agent_socket(self.agent_socket_path)

    def __repr__(self) -> str:
        return f"SshHostRoute(host_id={self.host_id!r})"

    @property
    def authority_sha256(self) -> str:
        """Non-secret endpoint/host-key identity retained across reconciliation."""

        known_hosts = _read_route_authority_leaf(self.known_hosts_path)
        return _ssh_route_authority_sha256(self, known_hosts)


def _validate_known_hosts(value: object) -> Path:
    if type(value) is not str:
        raise TypeError("known_hosts_path must be text")
    path = Path(value)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or path.stat().st_size < 1
        or path.stat().st_mode & 0o022
    ):
        raise ValueError(
            "known_hosts_path must be an absolute, non-symlink, non-writable file"
        )
    return path


def _read_route_authority_leaf(value: object) -> bytes:
    """Double-open the small public host-key file without retaining its path."""

    path = _validate_known_hosts(value)
    body, identity = _read_stable_leaf(
        path,
        label="known-hosts authority",
        limit_bytes=1024 * 1024,
    )
    reopened, reopened_identity = _read_stable_leaf(
        path,
        label="known-hosts authority",
        limit_bytes=1024 * 1024,
    )
    if identity != reopened_identity or body != reopened:
        raise ValueError("known-hosts authority changed between bound reads")
    return body


def _ssh_route_authority_sha256(route: SshHostRoute, known_hosts: bytes) -> str:
    destination = route.destination.encode("utf-8")
    return canonical_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_ssh_host_route_authority",
            "host_id": route.host_id,
            "destination_sha256": hashlib.sha256(destination).hexdigest(),
            "destination_size_bytes": len(destination),
            "port": route.port,
            "known_hosts_sha256": hashlib.sha256(known_hosts).hexdigest(),
            "known_hosts_size_bytes": len(known_hosts),
            "ssh_identity_policy_sha256": canonical_sha256(_SSH_IDENTITY_POLICY),
        }
    )


def _validate_agent_socket(value: object) -> Path:
    if type(value) is not str:
        raise TypeError("agent_socket_path must be text")
    path = Path(value)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ValueError(
            "agent_socket_path must name an existing Unix socket"
        ) from error
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve() != path
        or not stat.S_ISSOCK(mode)
    ):
        raise ValueError(
            "agent_socket_path must be an absolute, non-symlink Unix socket"
        )
    return path


def _build_ssh_argv(
    route: SshHostRoute,
    *,
    known_hosts_path: Path,
) -> tuple[str, ...]:
    return (
        "ssh",
        "-F",
        "/dev/null",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "ChallengeResponseAuthentication=no",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts_path}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "NumberOfPasswordPrompts=0",
        "-o",
        "LogLevel=ERROR",
        "-o",
        f"ConnectTimeout={route.connect_timeout_seconds}",
        "-p",
        str(route.port),
        "--",
        route.destination,
        *REMOTE_HOST_WAVE_COMMAND,
    )


def build_ssh_argv(route: SshHostRoute) -> tuple[str, ...]:
    """Build the complete fixed-policy OpenSSH argv for one route."""

    if type(route) is not SshHostRoute:
        raise TypeError("SSH argv requires an exact SshHostRoute")
    # This public renderer remains useful for inspection. Actual dispatch uses
    # the immutable snapshot below so the bytes hashed into route authority are
    # exactly the bytes OpenSSH consumes.
    known_hosts = _validate_known_hosts(route.known_hosts_path)
    _validate_agent_socket(route.agent_socket_path)
    return _build_ssh_argv(route, known_hosts_path=known_hosts)


@contextmanager
def _prepared_ssh_route(
    route: SshHostRoute,
) -> Iterator[tuple[tuple[str, ...], str]]:
    """Snapshot the exact public host-key authority for one SSH subprocess."""

    if type(route) is not SshHostRoute:
        raise TypeError("prepared SSH route requires an exact SshHostRoute")
    known_hosts = _read_route_authority_leaf(route.known_hosts_path)
    _validate_agent_socket(route.agent_socket_path)
    descriptor, snapshot_name = tempfile.mkstemp(prefix="lightcone-known-hosts-")
    snapshot = Path(snapshot_name).resolve()
    try:
        os.fchmod(descriptor, 0o400)
        remaining = memoryview(known_hosts)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:  # pragma: no cover - defensive OS boundary
                raise OSError("known-hosts snapshot write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        reopened = _read_route_authority_leaf(str(snapshot))
        if reopened != known_hosts:
            raise ValueError("known-hosts snapshot differs from route authority")
        yield (
            _build_ssh_argv(route, known_hosts_path=snapshot),
            _ssh_route_authority_sha256(route, known_hosts),
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        snapshot.unlink(missing_ok=True)


@dataclass(frozen=True)
class HostAssignmentBinding:
    """Host placement and collision domains for one atomic assignment."""

    assignment_sha256: str
    gpu_host_bindings: tuple[tuple[str, str], ...]
    ports: tuple[int, ...]
    cache_namespace: str
    evidence_namespace: str

    def __post_init__(self) -> None:
        _require_sha256("assignment_sha256", self.assignment_sha256)
        if not self.gpu_host_bindings:
            raise ValueError("host assignment must bind at least one GPU")
        if self.gpu_host_bindings != tuple(sorted(self.gpu_host_bindings)):
            raise ValueError("GPU host bindings must be canonically sorted")
        gpu_uuids = tuple(uuid for uuid, _ in self.gpu_host_bindings)
        if len(gpu_uuids) != len(set(gpu_uuids)):
            raise ValueError("host assignment contains duplicate GPU UUIDs")
        for uuid, host_id in self.gpu_host_bindings:
            _require_id("GPU UUID", uuid)
            _require_id("GPU host_id", host_id)
        if len({host_id for _, host_id in self.gpu_host_bindings}) != 1:
            raise CrossHostCollectivesUnvalidated()
        if not self.ports or len(self.ports) != len(set(self.ports)):
            raise ValueError("host assignment ports must be non-empty and unique")
        if any(
            type(port) is not int or not 1024 <= port <= 65_535 for port in self.ports
        ):
            raise ValueError("host assignment ports must be in [1024, 65535]")
        _require_namespace("cache_namespace", self.cache_namespace)
        _require_namespace("evidence_namespace", self.evidence_namespace)

    @property
    def host_id(self) -> str:
        return self.gpu_host_bindings[0][1]

    @property
    def gpu_uuids(self) -> tuple[str, ...]:
        return tuple(uuid for uuid, _ in self.gpu_host_bindings)

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_sha256": self.assignment_sha256,
            "gpu_host_bindings": [
                {"gpu_uuid": uuid, "host_id": host_id}
                for uuid, host_id in self.gpu_host_bindings
            ],
            "ports": list(self.ports),
            "cache_namespace": self.cache_namespace,
            "evidence_namespace": self.evidence_namespace,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "host assignment binding",
            value,
            frozenset(
                {
                    "assignment_sha256",
                    "gpu_host_bindings",
                    "ports",
                    "cache_namespace",
                    "evidence_namespace",
                }
            ),
        )
        gpu_bindings: list[tuple[str, str]] = []
        for item in _strict_list("GPU host bindings", row["gpu_host_bindings"]):
            binding = _strict_object(
                "GPU host binding", item, frozenset({"gpu_uuid", "host_id"})
            )
            gpu_bindings.append(
                (
                    _require_id("GPU UUID", binding["gpu_uuid"]),
                    _require_id("GPU host_id", binding["host_id"]),
                )
            )
        return cls(
            assignment_sha256=_require_sha256(
                "assignment_sha256", row["assignment_sha256"]
            ),
            gpu_host_bindings=tuple(gpu_bindings),
            ports=tuple(
                _strict_int("host assignment port", item)
                for item in _strict_list("host assignment ports", row["ports"])
            ),
            cache_namespace=_require_namespace(
                "cache_namespace", row["cache_namespace"]
            ),
            evidence_namespace=_require_namespace(
                "evidence_namespace", row["evidence_namespace"]
            ),
        )


@dataclass(frozen=True)
class RemoteHostExecutionBinding:
    """Transport binding for a host-local subset of one fleet wave."""

    schema_version: int
    host_id: str
    fleet_inventory_sha256: str
    host_inventory_sha256: str
    fleet_dispatch_plan_sha256: str
    host_dispatch_plan_sha256: str
    fleet_wave_index: int
    fleet_wave_sha256: str
    host_wave_index: int
    host_wave_sha256: str
    execution_bundle_manifest_path: str
    execution_bundle_manifest_sha256: str
    receipt_output_path: str
    resume_receipt_path: str | None
    resume_receipt_sha256: str | None
    resume_receipt_envelope_sha256: str | None
    assignments: tuple[HostAssignmentBinding, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only host-execution binding schema version 1 is supported"
            )
        _require_id("binding host_id", self.host_id)
        for name in (
            "fleet_inventory_sha256",
            "host_inventory_sha256",
            "fleet_dispatch_plan_sha256",
            "host_dispatch_plan_sha256",
            "fleet_wave_sha256",
            "host_wave_sha256",
            "execution_bundle_manifest_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in ("fleet_wave_index", "host_wave_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _require_host_local_path(
            "host-local manifest path", self.execution_bundle_manifest_path
        )
        _require_host_local_path(
            "host-local receipt output path", self.receipt_output_path
        )
        if self.resume_receipt_path is not None:
            _require_host_local_path(
                "host-local resume receipt path", self.resume_receipt_path
            )
        resume_values = (
            self.resume_receipt_path,
            self.resume_receipt_sha256,
            self.resume_receipt_envelope_sha256,
        )
        if any(value is None for value in resume_values) and any(
            value is not None for value in resume_values
        ):
            raise ValueError(
                "host-local resume receipt path and identities must be complete"
            )
        if self.resume_receipt_sha256 is not None:
            _require_sha256("resume_receipt_sha256", self.resume_receipt_sha256)
            _require_sha256(
                "resume_receipt_envelope_sha256",
                self.resume_receipt_envelope_sha256,
            )
        if self.receipt_output_path in {
            self.execution_bundle_manifest_path,
            self.resume_receipt_path,
        }:
            raise ValueError("host receipt output path must be a distinct artifact")
        if not self.assignments or any(
            type(item) is not HostAssignmentBinding for item in self.assignments
        ):
            raise ValueError("host execution binding requires exact assignments")
        assignment_ids = tuple(item.assignment_sha256 for item in self.assignments)
        if assignment_ids != tuple(sorted(set(assignment_ids))):
            raise ValueError("host assignments must be sorted and unique")
        if any(item.host_id != self.host_id for item in self.assignments):
            raise ValueError("host assignment differs from execution binding host")
        gpu_uuids = tuple(
            uuid for assignment in self.assignments for uuid in assignment.gpu_uuids
        )
        ports = tuple(
            port for assignment in self.assignments for port in assignment.ports
        )
        cache_namespaces = tuple(item.cache_namespace for item in self.assignments)
        evidence_namespaces = tuple(
            item.evidence_namespace for item in self.assignments
        )
        if len(gpu_uuids) != len(set(gpu_uuids)):
            raise ValueError("host execution binding overlaps a GPU")
        if len(ports) != len(set(ports)):
            raise ValueError("host execution binding overlaps a host-local port")
        if len(cache_namespaces) != len(set(cache_namespaces)):
            raise ValueError("host execution binding overlaps a cache namespace")
        if len(evidence_namespaces) != len(set(evidence_namespaces)):
            raise ValueError("host execution binding overlaps an evidence namespace")

    @cached_property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def assignment_sha256(self) -> tuple[str, ...]:
        return tuple(item.assignment_sha256 for item in self.assignments)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "lightcone_host_execution_binding",
            "host_id": self.host_id,
            "fleet_inventory_sha256": self.fleet_inventory_sha256,
            "host_inventory_sha256": self.host_inventory_sha256,
            "fleet_dispatch_plan_sha256": self.fleet_dispatch_plan_sha256,
            "host_dispatch_plan_sha256": self.host_dispatch_plan_sha256,
            "fleet_wave_index": self.fleet_wave_index,
            "fleet_wave_sha256": self.fleet_wave_sha256,
            "host_wave_index": self.host_wave_index,
            "host_wave_sha256": self.host_wave_sha256,
            "execution_bundle_manifest_path": self.execution_bundle_manifest_path,
            "execution_bundle_manifest_sha256": (self.execution_bundle_manifest_sha256),
            "receipt_output_path": self.receipt_output_path,
            "resume_receipt_path": self.resume_receipt_path,
            "resume_receipt_sha256": self.resume_receipt_sha256,
            "resume_receipt_envelope_sha256": (self.resume_receipt_envelope_sha256),
            "assignments": [item.to_dict() for item in self.assignments],
            "assignment_sha256": list(self.assignment_sha256),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "host execution binding",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "host_id",
                    "fleet_inventory_sha256",
                    "host_inventory_sha256",
                    "fleet_dispatch_plan_sha256",
                    "host_dispatch_plan_sha256",
                    "fleet_wave_index",
                    "fleet_wave_sha256",
                    "host_wave_index",
                    "host_wave_sha256",
                    "execution_bundle_manifest_path",
                    "execution_bundle_manifest_sha256",
                    "receipt_output_path",
                    "resume_receipt_path",
                    "resume_receipt_sha256",
                    "resume_receipt_envelope_sha256",
                    "assignments",
                    "assignment_sha256",
                }
            ),
        )
        if row["kind"] != "lightcone_host_execution_binding":
            raise ValueError("host execution binding kind is unsupported")
        assignments = tuple(
            HostAssignmentBinding.from_dict(item)
            for item in _strict_list("host assignments", row["assignments"])
        )
        declared = tuple(
            _require_sha256("assignment_sha256", item)
            for item in _strict_list(
                "host assignment SHA-256 list", row["assignment_sha256"]
            )
        )
        if declared != tuple(item.assignment_sha256 for item in assignments):
            raise ValueError("host assignment SHA-256 list mismatch")
        return cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            host_id=_require_id("host_id", row["host_id"]),
            fleet_inventory_sha256=_require_sha256(
                "fleet_inventory_sha256", row["fleet_inventory_sha256"]
            ),
            host_inventory_sha256=_require_sha256(
                "host_inventory_sha256", row["host_inventory_sha256"]
            ),
            fleet_dispatch_plan_sha256=_require_sha256(
                "fleet_dispatch_plan_sha256", row["fleet_dispatch_plan_sha256"]
            ),
            host_dispatch_plan_sha256=_require_sha256(
                "host_dispatch_plan_sha256", row["host_dispatch_plan_sha256"]
            ),
            fleet_wave_index=_strict_int("fleet_wave_index", row["fleet_wave_index"]),
            fleet_wave_sha256=_require_sha256(
                "fleet_wave_sha256", row["fleet_wave_sha256"]
            ),
            host_wave_index=_strict_int("host_wave_index", row["host_wave_index"]),
            host_wave_sha256=_require_sha256(
                "host_wave_sha256", row["host_wave_sha256"]
            ),
            execution_bundle_manifest_path=_require_host_local_path(
                "host-local manifest path", row["execution_bundle_manifest_path"]
            ),
            execution_bundle_manifest_sha256=_require_sha256(
                "execution_bundle_manifest_sha256",
                row["execution_bundle_manifest_sha256"],
            ),
            receipt_output_path=_require_host_local_path(
                "host-local receipt output path", row["receipt_output_path"]
            ),
            resume_receipt_path=_optional_host_local_path(
                "host-local resume receipt path", row["resume_receipt_path"]
            ),
            resume_receipt_sha256=_optional_sha256(
                "resume_receipt_sha256", row["resume_receipt_sha256"]
            ),
            resume_receipt_envelope_sha256=_optional_sha256(
                "resume_receipt_envelope_sha256",
                row["resume_receipt_envelope_sha256"],
            ),
            assignments=assignments,
        )


@dataclass(frozen=True)
class RemoteHostWaveRequest:
    """Canonical stdin request for one host; contains no routing material."""

    schema_version: int
    challenge_nonce_sha256: str
    ssh_route_authority_sha256: str
    binding: RemoteHostExecutionBinding
    prior_fleet_wave_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only remote host-wave request schema version 1 is supported"
            )
        _require_sha256("challenge_nonce_sha256", self.challenge_nonce_sha256)
        _require_sha256("ssh_route_authority_sha256", self.ssh_route_authority_sha256)
        if type(self.binding) is not RemoteHostExecutionBinding:
            raise TypeError(
                "remote request requires an exact RemoteHostExecutionBinding"
            )
        if self.prior_fleet_wave_receipt_sha256 is not None:
            _require_sha256(
                "prior_fleet_wave_receipt_sha256",
                self.prior_fleet_wave_receipt_sha256,
            )
        if (
            self.binding.resume_receipt_path is not None
            and self.prior_fleet_wave_receipt_sha256 is None
        ):
            raise ValueError(
                "host-local resume requires a prior fleet-wave receipt identity"
            )

    @cached_property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "lightcone_remote_host_wave_request",
            "challenge_nonce_sha256": self.challenge_nonce_sha256,
            "ssh_route_authority_sha256": self.ssh_route_authority_sha256,
            "binding": self.binding.to_dict(),
            "binding_sha256": self.binding.sha256,
            "prior_fleet_wave_receipt_sha256": (self.prior_fleet_wave_receipt_sha256),
        }

    def canonical_stdin(self) -> bytes:
        body = canonical_json_bytes(self.to_dict()) + b"\n"
        if len(body) > MAX_REQUEST_BYTES:
            raise ValueError("remote host-wave request exceeds the bounded stdin size")
        return body

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "remote host-wave request",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "challenge_nonce_sha256",
                    "ssh_route_authority_sha256",
                    "binding",
                    "binding_sha256",
                    "prior_fleet_wave_receipt_sha256",
                }
            ),
        )
        if row["kind"] != "lightcone_remote_host_wave_request":
            raise ValueError("remote host-wave request kind is unsupported")
        binding = RemoteHostExecutionBinding.from_dict(row["binding"])
        declared = _require_sha256("binding_sha256", row["binding_sha256"])
        if declared != binding.sha256:
            raise ValueError("remote request binding SHA-256 mismatch")
        return cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            challenge_nonce_sha256=_require_sha256(
                "challenge_nonce_sha256", row["challenge_nonce_sha256"]
            ),
            ssh_route_authority_sha256=_require_sha256(
                "ssh_route_authority_sha256", row["ssh_route_authority_sha256"]
            ),
            binding=binding,
            prior_fleet_wave_receipt_sha256=_optional_sha256(
                "prior_fleet_wave_receipt_sha256",
                row["prior_fleet_wave_receipt_sha256"],
            ),
        )


def decode_remote_host_wave_request(stdin: bytes) -> RemoteHostWaveRequest:
    """Decode one bounded, newline-terminated canonical stdin request."""

    value = _decode_canonical_json_line(
        stdin,
        label="remote host-wave stdin",
        limit_bytes=MAX_REQUEST_BYTES,
    )
    request = RemoteHostWaveRequest.from_dict(value)
    if request.canonical_stdin() != stdin:
        raise ValueError("remote host-wave stdin changed after strict decoding")
    return request


@dataclass(frozen=True)
class _RemoteEvidenceRequest:
    """Private fixed-host request for an ambiguous attempt's raw evidence."""

    schema_version: int
    source_request: RemoteHostWaveRequest
    unknown_result_sha256: str
    limit_bytes: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only remote raw-evidence request schema 1 is supported")
        if type(self.source_request) is not RemoteHostWaveRequest:
            raise TypeError("raw-evidence request requires an exact source request")
        _require_sha256("unknown_result_sha256", self.unknown_result_sha256)
        if (
            type(self.limit_bytes) is not int
            or not 1 <= self.limit_bytes <= MAX_RECONCILE_EVIDENCE_BYTES
        ):
            raise ValueError("raw-evidence request limit is outside protocol bounds")
        if len(self.source_request.binding.assignments) > MAX_RECONCILE_ASSIGNMENTS:
            raise ValueError("raw-evidence request has too many assignments")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "lightcone_remote_raw_evidence_request",
            "source_request": self.source_request.to_dict(),
            "source_request_sha256": self.source_request.sha256,
            "unknown_result_sha256": self.unknown_result_sha256,
            "limit_bytes": self.limit_bytes,
        }

    def canonical_stdin(self) -> bytes:
        body = canonical_json_bytes(self.to_dict()) + b"\n"
        if len(body) > MAX_REQUEST_BYTES:
            raise ValueError("remote raw-evidence request exceeds bounded stdin")
        return body

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "remote raw-evidence request",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "source_request",
                    "source_request_sha256",
                    "unknown_result_sha256",
                    "limit_bytes",
                }
            ),
        )
        if row["kind"] != "lightcone_remote_raw_evidence_request":
            raise ValueError("remote raw-evidence request kind is unsupported")
        source = RemoteHostWaveRequest.from_dict(row["source_request"])
        if row["source_request_sha256"] != source.sha256:
            raise ValueError("remote raw-evidence source request digest differs")
        return cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            source_request=source,
            unknown_result_sha256=_require_sha256(
                "unknown_result_sha256", row["unknown_result_sha256"]
            ),
            limit_bytes=_strict_int("limit_bytes", row["limit_bytes"]),
        )


@dataclass(frozen=True)
class _RemoteRawBlob:
    """One bounded raw file transported only inside the SSH response."""

    size_bytes: int
    sha256: str
    body_base64: str

    def __post_init__(self) -> None:
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("remote raw blob must be non-empty")
        _require_sha256("remote raw blob SHA-256", self.sha256)
        if type(self.body_base64) is not str or not self.body_base64:
            raise ValueError("remote raw blob body must be base64 text")
        try:
            body = base64.b64decode(self.body_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("remote raw blob is not strict base64") from error
        if (
            base64.b64encode(body).decode("ascii") != self.body_base64
            or len(body) != self.size_bytes
            or hashlib.sha256(body).hexdigest() != self.sha256
        ):
            raise ValueError("remote raw blob content identity differs")

    @classmethod
    def from_body(cls, body: bytes) -> Self:
        if type(body) is not bytes or not body:
            raise ValueError("remote raw blob body must be non-empty bytes")
        return cls(
            size_bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            body_base64=base64.b64encode(body).decode("ascii"),
        )

    @property
    def body(self) -> bytes:
        return base64.b64decode(self.body_base64, validate=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "body_base64": self.body_base64,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "remote raw blob",
            value,
            frozenset({"size_bytes", "sha256", "body_base64"}),
        )
        return cls(
            size_bytes=_strict_int("raw blob size", row["size_bytes"]),
            sha256=_require_sha256("raw blob SHA-256", row["sha256"]),
            body_base64=row["body_base64"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class _RemoteRawEvidenceFile:
    assignment_sha256: str
    evidence_index: int
    blob: _RemoteRawBlob

    def __post_init__(self) -> None:
        _require_sha256("raw evidence assignment", self.assignment_sha256)
        if type(self.evidence_index) is not int or self.evidence_index < 0:
            raise ValueError("raw evidence index must be non-negative")
        if type(self.blob) is not _RemoteRawBlob:
            raise TypeError("raw evidence file requires an exact blob")
        if self.blob.size_bytes > MAX_RECONCILE_FILE_BYTES:
            raise ValueError("remote evidence file exceeds its byte limit")

    @property
    def key(self) -> tuple[str, int]:
        return self.assignment_sha256, self.evidence_index

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_sha256": self.assignment_sha256,
            "evidence_index": self.evidence_index,
            "blob": self.blob.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "remote raw evidence file",
            value,
            frozenset({"assignment_sha256", "evidence_index", "blob"}),
        )
        return cls(
            assignment_sha256=_require_sha256(
                "raw evidence assignment", row["assignment_sha256"]
            ),
            evidence_index=_strict_int("raw evidence index", row["evidence_index"]),
            blob=_RemoteRawBlob.from_dict(row["blob"]),
        )


@dataclass(frozen=True)
class _RemoteRawEvidenceBundle:
    """Ephemeral raw evidence; never nested in a durable fleet receipt."""

    schema_version: int
    host_id: str
    source_request_sha256: str
    source_binding_sha256: str
    unknown_result_sha256: str
    schedule_envelope: _RemoteRawBlob
    evidence_files: tuple[_RemoteRawEvidenceFile, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only remote raw-evidence bundle schema 1 is supported")
        _require_id("raw-evidence host_id", self.host_id)
        for name in (
            "source_request_sha256",
            "source_binding_sha256",
            "unknown_result_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if type(self.schedule_envelope) is not _RemoteRawBlob:
            raise TypeError("raw-evidence bundle requires an exact receipt envelope")
        if self.schedule_envelope.size_bytes > MAX_RECONCILE_RECEIPT_BYTES:
            raise ValueError("remote receipt envelope exceeds its byte limit")
        if len(self.evidence_files) > MAX_RECONCILE_FILE_COUNT or any(
            type(item) is not _RemoteRawEvidenceFile for item in self.evidence_files
        ):
            raise ValueError("remote evidence file count exceeds protocol bounds")
        keys = tuple(item.key for item in self.evidence_files)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("remote evidence files must be sorted and unique")
        raw_size = self.schedule_envelope.size_bytes + sum(
            item.blob.size_bytes for item in self.evidence_files
        )
        if raw_size > MAX_RECONCILE_RAW_BYTES:
            raise ValueError("remote raw evidence exceeds its total byte limit")

    @cached_property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "lightcone_remote_raw_evidence_bundle",
            "host_id": self.host_id,
            "source_request_sha256": self.source_request_sha256,
            "source_binding_sha256": self.source_binding_sha256,
            "unknown_result_sha256": self.unknown_result_sha256,
            "schedule_envelope": self.schedule_envelope.to_dict(),
            "evidence_files": [item.to_dict() for item in self.evidence_files],
        }

    def canonical_payload(self, *, limit_bytes: int) -> bytes:
        body = canonical_json_bytes(self.to_dict()) + b"\n"
        if len(body) > limit_bytes or len(body) > MAX_RECONCILE_EVIDENCE_BYTES:
            raise ValueError("remote raw-evidence bundle exceeds payload bounds")
        return body

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "remote raw-evidence bundle",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "host_id",
                    "source_request_sha256",
                    "source_binding_sha256",
                    "unknown_result_sha256",
                    "schedule_envelope",
                    "evidence_files",
                }
            ),
        )
        if row["kind"] != "lightcone_remote_raw_evidence_bundle":
            raise ValueError("remote raw-evidence bundle kind is unsupported")
        return cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            host_id=_require_id("host_id", row["host_id"]),
            source_request_sha256=_require_sha256(
                "source_request_sha256", row["source_request_sha256"]
            ),
            source_binding_sha256=_require_sha256(
                "source_binding_sha256", row["source_binding_sha256"]
            ),
            unknown_result_sha256=_require_sha256(
                "unknown_result_sha256", row["unknown_result_sha256"]
            ),
            schedule_envelope=_RemoteRawBlob.from_dict(row["schedule_envelope"]),
            evidence_files=tuple(
                _RemoteRawEvidenceFile.from_dict(item)
                for item in _strict_list(
                    "remote raw evidence files", row["evidence_files"]
                )
            ),
        )


def _decode_remote_raw_evidence_bundle(
    payload: bytes,
    *,
    limit_bytes: int,
) -> _RemoteRawEvidenceBundle:
    value = _decode_canonical_json_line(
        payload,
        label="remote raw-evidence bundle",
        limit_bytes=min(limit_bytes, MAX_RECONCILE_EVIDENCE_BYTES),
    )
    bundle = _RemoteRawEvidenceBundle.from_dict(value)
    if bundle.canonical_payload(limit_bytes=limit_bytes) != payload:
        raise ValueError("remote raw-evidence bundle changed after strict decoding")
    return bundle


class RemoteWorkerStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class RemoteHostWaveResponse:
    """Canonical response emitted by the host-local CLI worker."""

    schema_version: int
    host_id: str
    request_sha256: str
    binding_sha256: str
    status: RemoteWorkerStatus
    reason_code: str | None
    dispatch_schedule_receipt_sha256: str | None
    dispatch_schedule_receipt_envelope_sha256: str | None
    completed_assignment_sha256: tuple[str, ...]
    failed_assignment_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only remote host-wave response schema version 1 is supported"
            )
        _require_id("response host_id", self.host_id)
        _require_sha256("request_sha256", self.request_sha256)
        _require_sha256("binding_sha256", self.binding_sha256)
        if not isinstance(self.status, RemoteWorkerStatus):
            raise TypeError("remote response status must be a RemoteWorkerStatus")
        if self.reason_code is not None:
            _require_reason("remote response reason_code", self.reason_code)
        if self.dispatch_schedule_receipt_sha256 is not None:
            _require_sha256(
                "dispatch_schedule_receipt_sha256",
                self.dispatch_schedule_receipt_sha256,
            )
        if self.dispatch_schedule_receipt_envelope_sha256 is not None:
            _require_sha256(
                "dispatch_schedule_receipt_envelope_sha256",
                self.dispatch_schedule_receipt_envelope_sha256,
            )
        if (self.dispatch_schedule_receipt_sha256 is None) != (
            self.dispatch_schedule_receipt_envelope_sha256 is None
        ):
            raise ValueError("receipt and envelope identities must be paired")
        for name in (
            "completed_assignment_sha256",
            "failed_assignment_sha256",
        ):
            identities = getattr(self, name)
            if identities != tuple(sorted(set(identities))):
                raise ValueError(f"{name} must be sorted and unique")
            for identity in identities:
                _require_sha256(name, identity)
        if set(self.completed_assignment_sha256) & set(self.failed_assignment_sha256):
            raise ValueError("remote response assignment outcomes overlap")
        if self.status is RemoteWorkerStatus.SUCCEEDED:
            if (
                self.reason_code is not None
                or self.dispatch_schedule_receipt_sha256 is None
                or self.dispatch_schedule_receipt_envelope_sha256 is None
                or self.failed_assignment_sha256
            ):
                raise ValueError("successful remote response is incomplete")
        elif self.status is RemoteWorkerStatus.BLOCKED:
            if (
                self.reason_code is None
                or self.dispatch_schedule_receipt_sha256 is not None
                or self.dispatch_schedule_receipt_envelope_sha256 is not None
                or self.completed_assignment_sha256
                or self.failed_assignment_sha256
            ):
                raise ValueError("blocked remote response cannot claim execution")
        elif (
            self.reason_code is None
            or self.dispatch_schedule_receipt_sha256 is None
            or self.dispatch_schedule_receipt_envelope_sha256 is None
        ):
            raise ValueError("failed remote response requires receipt authority")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "lightcone_remote_host_wave_response",
            "host_id": self.host_id,
            "request_sha256": self.request_sha256,
            "binding_sha256": self.binding_sha256,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "dispatch_schedule_receipt_sha256": (self.dispatch_schedule_receipt_sha256),
            "dispatch_schedule_receipt_envelope_sha256": (
                self.dispatch_schedule_receipt_envelope_sha256
            ),
            "completed_assignment_sha256": list(self.completed_assignment_sha256),
            "failed_assignment_sha256": list(self.failed_assignment_sha256),
        }

    def canonical_stdout(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "remote worker response",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "host_id",
                    "request_sha256",
                    "binding_sha256",
                    "status",
                    "reason_code",
                    "dispatch_schedule_receipt_sha256",
                    "dispatch_schedule_receipt_envelope_sha256",
                    "completed_assignment_sha256",
                    "failed_assignment_sha256",
                }
            ),
        )
        if row["kind"] != "lightcone_remote_host_wave_response":
            raise ValueError("remote host-wave response kind is unsupported")
        try:
            status = RemoteWorkerStatus(row["status"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "remote host-wave response status is unsupported"
            ) from error
        reason = row["reason_code"]
        if reason is not None:
            reason = _require_reason("remote response reason_code", reason)
        return cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            host_id=_require_id("host_id", row["host_id"]),
            request_sha256=_require_sha256("request_sha256", row["request_sha256"]),
            binding_sha256=_require_sha256("binding_sha256", row["binding_sha256"]),
            status=status,
            reason_code=reason,
            dispatch_schedule_receipt_sha256=_optional_sha256(
                "dispatch_schedule_receipt_sha256",
                row["dispatch_schedule_receipt_sha256"],
            ),
            dispatch_schedule_receipt_envelope_sha256=_optional_sha256(
                "dispatch_schedule_receipt_envelope_sha256",
                row["dispatch_schedule_receipt_envelope_sha256"],
            ),
            completed_assignment_sha256=tuple(
                _require_sha256("completed assignment", item)
                for item in _strict_list(
                    "completed assignments", row["completed_assignment_sha256"]
                )
            ),
            failed_assignment_sha256=tuple(
                _require_sha256("failed assignment", item)
                for item in _strict_list(
                    "failed assignments", row["failed_assignment_sha256"]
                )
            ),
        )


def _blocked_worker_response(
    request: RemoteHostWaveRequest,
    reason_code: str,
) -> RemoteHostWaveResponse:
    return RemoteHostWaveResponse(
        schema_version=1,
        host_id=request.binding.host_id,
        request_sha256=request.sha256,
        binding_sha256=request.binding.sha256,
        status=RemoteWorkerStatus.BLOCKED,
        reason_code=_require_reason("worker BLOCKED reason", reason_code),
        dispatch_schedule_receipt_sha256=None,
        dispatch_schedule_receipt_envelope_sha256=None,
        completed_assignment_sha256=(),
        failed_assignment_sha256=(),
    )


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_leaf(
    path: Path,
    *,
    label: str,
    limit_bytes: int,
) -> tuple[bytes, tuple[int, ...]]:
    """Read one bounded single-link leaf with path/fd identity agreement."""

    initial = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(initial.st_mode)
        or initial.st_nlink != 1
        or initial.st_size < 1
        or initial.st_size > limit_bytes
    ):
        raise ValueError(f"{label} is not one bounded single-link file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            _stat_identity(initial) != _stat_identity(opened)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size < 1
            or opened.st_size > limit_bytes
        ):
            raise ValueError(f"{label} changed before it was opened")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{label} shrank while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} grew while it was read")
        body = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = _stat_identity(opened)
    if (
        len(body) != opened.st_size
        or identity != _stat_identity(after)
        or identity != _stat_identity(current)
        or not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
    ):
        raise ValueError(f"{label} changed while it was read")
    return body, identity


def _read_stable_manifest_leaf(path: Path) -> tuple[bytes, tuple[int, ...]]:
    return _read_stable_leaf(
        path,
        label="host-local manifest",
        limit_bytes=MAX_MANIFEST_BYTES,
    )


def _load_verified_host_local_publication(
    binding: RemoteHostExecutionBinding,
) -> tuple[object, tuple[object, ...]]:
    """Double-reopen and semantically bind one host-local publication."""

    path = Path(binding.execution_bundle_manifest_path)
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("host-local manifest path is unresolved or a symlink")
    body, identity = _read_stable_manifest_leaf(path)
    reopened_body, reopened_identity = _read_stable_manifest_leaf(path)
    if identity != reopened_identity or body != reopened_body:
        raise ValueError("host-local manifest changed between bound reads")
    canonical_body = body[:-1] if body.endswith(b"\n") else body
    if not canonical_body or body not in {canonical_body, canonical_body + b"\n"}:
        raise ValueError("host-local manifest framing is invalid")
    value = _strict_json_object(canonical_body)
    if canonical_json_bytes(value) != canonical_body:
        raise ValueError("host-local manifest is not canonical JSON")
    if canonical_sha256(value) != binding.execution_bundle_manifest_sha256:
        raise ValueError("host-local manifest SHA-256 differs from its binding")

    # Reopen the complete source-owned publication before any executor call.
    # The materializer replays the manifest, dispatch plan, raw construction
    # inputs, bundle sidecars, and exact assignment coverage.  This additional
    # projection binds the remote control request to that verified local plan
    # so a stale or foreign request cannot launch a valid-but-different wave.
    from lightcone_spec.orchestration.execution_bundle_materializer import (
        load_materialized_dispatch_execution_bundle_publication,
    )

    publication = load_materialized_dispatch_execution_bundle_publication(path)
    if publication.manifest.to_dict() != value:
        raise ValueError("host-local publication manifest changed during replay")
    plans = tuple(bundle.reconstruct_execution_plan() for bundle in publication.bundles)
    if not plans:
        raise ValueError("host-local publication contains no execution plans")
    dispatch_plan = plans[0].dispatch_plan
    inventory = plans[0].dispatch_context.inventory
    if any(
        plan.dispatch_plan != dispatch_plan
        or plan.dispatch_context.inventory != inventory
        for plan in plans[1:]
    ):
        raise ValueError("host-local publication mixes dispatch authorities")
    if (
        dispatch_plan.sha256 != binding.host_dispatch_plan_sha256
        or inventory.sha256 != binding.host_inventory_sha256
        or binding.host_wave_index >= len(dispatch_plan.waves)
    ):
        raise ValueError("host-local publication differs from host plan authority")
    wave = dispatch_plan.waves[binding.host_wave_index]
    if (
        wave.wave_index != binding.host_wave_index
        or wave.sha256 != binding.host_wave_sha256
        or tuple(assignment.assignment_id for assignment in wave.assignments)
        != binding.assignment_sha256
    ):
        raise ValueError("host-local publication differs from host wave authority")
    bound_assignments = {
        assignment.assignment_sha256: assignment for assignment in binding.assignments
    }
    if any(
        assignment.gpu_uuids != bound_assignments[assignment.assignment_id].gpu_uuids
        or assignment.ports != bound_assignments[assignment.assignment_id].ports
        for assignment in wave.assignments
    ):
        raise ValueError("host-local publication differs from physical placement")
    plan_by_assignment = {
        plan.runtime_plan.physical_assignment.assignment_sha256: plan
        for plan in plans
        if plan.runtime_plan.physical_assignment is not None
    }
    if set(binding.assignment_sha256) - set(plan_by_assignment) or any(
        plan_by_assignment[assignment_sha256].runtime_plan.physical_assignment.host_id
        != binding.host_id
        for assignment_sha256 in binding.assignment_sha256
    ):
        raise ValueError("host-local publication differs from host identity")

    return publication, plans


def _read_stable_evidence_leaf(
    path_value: str,
    *,
    limit_bytes: int = MAX_RECONCILE_FILE_BYTES,
) -> bytes:
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ValueError("remote evidence path is unresolved or a symlink")
    body, identity = _read_stable_leaf(
        path,
        label="remote evidence file",
        limit_bytes=limit_bytes,
    )
    reopened, reopened_identity = _read_stable_leaf(
        path,
        label="remote evidence file",
        limit_bytes=limit_bytes,
    )
    if identity != reopened_identity or body != reopened:
        raise ValueError("remote evidence file changed between bound reads")
    return body


def _worker_response_from_receipt(
    request: RemoteHostWaveRequest,
    receipt: object,
    *,
    receipt_envelope_sha256: str,
) -> RemoteHostWaveResponse:
    """Validate the first-party schedule receipt against transport authority."""

    from lightcone_spec.experiments.gpu_pool import (
        AssignmentExecutionStatus,
        DispatchScheduleReceipt,
    )

    if type(receipt) is not DispatchScheduleReceipt:
        raise TypeError("host executor returned a non-exact schedule receipt")
    _require_sha256("receipt_envelope_sha256", receipt_envelope_sha256)
    binding = request.binding
    if (
        receipt.plan_sha256 != binding.host_dispatch_plan_sha256
        or receipt.inventory_sha256 != binding.host_inventory_sha256
        or len(receipt.wave_receipts) != binding.host_wave_index + 1
    ):
        raise ValueError("host schedule receipt differs from request authority")
    if any(
        assignment.plan_sha256 != binding.host_dispatch_plan_sha256
        or assignment.wave_sha256 != wave_receipt.wave_sha256
        or (
            assignment.terminal_binding is not None
            and (
                assignment.terminal_binding.dispatch_plan_sha256
                != binding.host_dispatch_plan_sha256
                or assignment.terminal_binding.inventory_sha256
                != binding.host_inventory_sha256
            )
        )
        for wave_receipt in receipt.wave_receipts
        for assignment in wave_receipt.assignment_receipts
    ):
        raise ValueError("host schedule child receipt differs from wave authority")
    wave = receipt.wave_receipts[binding.host_wave_index]
    if (
        wave.wave_index != binding.host_wave_index
        or wave.wave_sha256 != binding.host_wave_sha256
    ):
        raise ValueError("host schedule receipt differs from requested wave")
    actual = tuple(sorted(row.assignment_sha256 for row in wave.assignment_receipts))
    if actual != binding.assignment_sha256:
        raise ValueError("host schedule receipt assignment coverage differs")
    assignment_bindings = {
        assignment.assignment_sha256: assignment for assignment in binding.assignments
    }
    if any(
        row.terminal_binding is not None
        and row.terminal_binding.physical_gpu_uuids
        != assignment_bindings[row.assignment_sha256].gpu_uuids
        for row in wave.assignment_receipts
    ):
        raise ValueError("host schedule terminal GPU coverage differs")
    completed = tuple(
        sorted(
            row.assignment_sha256
            for row in wave.assignment_receipts
            if row.status is AssignmentExecutionStatus.SUCCEEDED
        )
    )
    failed = tuple(
        sorted(
            row.assignment_sha256
            for row in wave.assignment_receipts
            if row.status is AssignmentExecutionStatus.FAILED
        )
    )
    succeeded = not failed and completed == binding.assignment_sha256
    return RemoteHostWaveResponse(
        schema_version=1,
        host_id=binding.host_id,
        request_sha256=request.sha256,
        binding_sha256=binding.sha256,
        status=(
            RemoteWorkerStatus.SUCCEEDED if succeeded else RemoteWorkerStatus.FAILED
        ),
        reason_code=None if succeeded else "remote_host_wave_failed",
        dispatch_schedule_receipt_sha256=receipt.sha256,
        dispatch_schedule_receipt_envelope_sha256=receipt_envelope_sha256,
        completed_assignment_sha256=completed,
        failed_assignment_sha256=failed,
    )


def _collect_remote_raw_evidence_bundle(
    evidence_request: _RemoteEvidenceRequest,
) -> _RemoteRawEvidenceBundle:
    """Reopen production receipt authority and collect only bounded raw bytes."""

    from lightcone_spec.experiments.completion_authority import (
        AssignmentTerminalAuthority,
    )
    from lightcone_spec.experiments.gpu_pool import (
        AssignmentExecutionStatus,
        DispatchExecutionPhase,
        validate_dispatch_resume,
    )
    from lightcone_spec.orchestration.execution_bundle import (
        DispatchAttemptJournal,
    )

    request = evidence_request.source_request
    binding = request.binding
    publication, verified_plans = _load_verified_host_local_publication(binding)
    # The preceding stable leaf digest and the materializer's own canonical
    # source binding jointly establish the manifest identity.  Do not compare
    # the semantic manifest hash here: BoundJsonSource uses canonical JSON.
    if not publication.bundles:
        raise ValueError("remote publication contains no execution bundles")

    plan_by_assignment: dict[str, object] = {}
    dispatch_plan = None
    base_context = None
    for bundle, plan in zip(publication.bundles, verified_plans, strict=True):
        physical = plan.runtime_plan.physical_assignment
        if (
            physical is None
            or physical.assignment_sha256 != bundle.assignment_sha256
            or bundle.assignment_sha256 in plan_by_assignment
        ):
            raise ValueError("remote publication assignment plan is malformed")
        if dispatch_plan is None:
            dispatch_plan = plan.dispatch_plan
            base_context = plan.dispatch_context
        elif (
            plan.dispatch_plan != dispatch_plan or plan.dispatch_context != base_context
        ):
            raise ValueError("remote publication mixes dispatch authorities")
        plan_by_assignment[bundle.assignment_sha256] = plan
    if dispatch_plan is None or base_context is None:  # pragma: no cover - guarded
        raise RuntimeError("remote publication reconstruction returned no plan")
    if (
        dispatch_plan.sha256 != binding.host_dispatch_plan_sha256
        or base_context.inventory.sha256 != binding.host_inventory_sha256
        or binding.host_wave_index >= len(dispatch_plan.waves)
    ):
        raise ValueError("remote publication differs from host wave authority")
    wave = dispatch_plan.waves[binding.host_wave_index]
    if (
        wave.wave_index != binding.host_wave_index
        or wave.sha256 != binding.host_wave_sha256
        or tuple(assignment.assignment_id for assignment in wave.assignments)
        != binding.assignment_sha256
    ):
        raise ValueError("remote publication differs from host wave coverage")
    binding_by_assignment = {row.assignment_sha256: row for row in binding.assignments}
    if any(
        assignment.gpu_uuids
        != binding_by_assignment[assignment.assignment_id].gpu_uuids
        for assignment in wave.assignments
    ):
        raise ValueError("remote publication GPU placement differs from binding")

    receipt_body = _read_stable_evidence_leaf(
        binding.receipt_output_path,
        limit_bytes=MAX_RECONCILE_RECEIPT_BYTES,
    )
    receipt, journal_binding = _decode_raw_schedule_envelope(receipt_body)
    if journal_binding is None:
        raise ValueError("remote evidence requires a schema-v2 receipt envelope")
    resume_receipt = None
    resume_journal = None
    if binding.resume_receipt_path is None:
        expected_journal_path = Path(binding.receipt_output_path).with_name(
            f"{Path(binding.receipt_output_path).name}.attempt-journal"
        )
    else:
        resume_body = _read_stable_evidence_leaf(
            binding.resume_receipt_path,
            limit_bytes=MAX_RECONCILE_RECEIPT_BYTES,
        )
        if (
            _canonical_envelope_sha256(resume_body)
            != binding.resume_receipt_envelope_sha256
        ):
            raise ValueError("remote resume envelope differs from its binding")
        resume_receipt, resume_journal = _decode_raw_schedule_envelope(resume_body)
        if (
            resume_receipt.sha256 != binding.resume_receipt_sha256
            or resume_journal is None
        ):
            raise ValueError("remote resume receipt differs from its binding")
        expected_journal_path = Path(resume_journal.journal_path)
    if Path(journal_binding.journal_path) != expected_journal_path:
        raise ValueError("remote receipt journal differs from its output binding")
    journal = DispatchAttemptJournal.open_existing(
        journal_binding.journal_path,
        plan=dispatch_plan,
        execution_context=base_context,
        expected_prefix=journal_binding,
        execution_bundle_manifest_sha256=publication.manifest.sha256,
    )
    snapshot = journal.replay()
    snapshot.require_complete_cost_authority()
    if (
        snapshot.binding != journal_binding
        or snapshot.receipt != receipt
        or snapshot.replay_authority is None
    ):
        raise ValueError("remote receipt differs from exact raw journal replay")
    if resume_journal is not None:
        prefix = journal.replay(event_count=resume_journal.event_count)
        prefix.require_complete_cost_authority()
        if prefix.binding != resume_journal or prefix.receipt != resume_receipt:
            raise ValueError("remote retry journal differs from its resume authority")

    terminal_authorities = []
    for terminal in snapshot.terminal_bindings:
        plan = plan_by_assignment.get(terminal.assignment_sha256)
        if plan is None:
            raise ValueError("remote terminal lacks its execution bundle")
        terminal_authorities.append(
            AssignmentTerminalAuthority.from_binding(terminal, plan=plan)
        )
    execution_context = replace(
        base_context,
        resume_terminal_authorities=tuple(terminal_authorities),
    )
    validate_dispatch_resume(
        dispatch_plan,
        receipt,
        execution_context=execution_context,
        attempt_journal_replay=snapshot.replay_authority,
    )
    _worker_response_from_receipt(
        request,
        receipt,
        receipt_envelope_sha256=_canonical_envelope_sha256(receipt_body),
    )

    reopened_receipt, reopened_journal = _decode_raw_schedule_envelope(receipt_body)
    if reopened_receipt != receipt or reopened_journal != journal_binding:
        raise ValueError("remote receipt changed after raw authority replay")
    evidence_files: list[_RemoteRawEvidenceFile] = []
    evidence_total_bytes = 0
    target = receipt.wave_receipts[binding.host_wave_index]
    if receipt.phase is DispatchExecutionPhase.RUNNING:
        raise ValueError("remote target wave receipt is not terminal")
    for row in target.assignment_receipts:
        if row.status is not AssignmentExecutionStatus.SUCCEEDED:
            continue
        terminal = row.terminal_binding
        if terminal is None:  # pragma: no cover - receipt invariant
            raise ValueError("remote successful receipt lacks terminal evidence")
        if len(terminal.evidence_file_paths) != len(terminal.evidence_file_sha256s):
            raise ValueError("remote terminal evidence coverage differs")
        if (
            len(evidence_files) + len(terminal.evidence_file_paths)
            > MAX_RECONCILE_FILE_COUNT
        ):
            raise ValueError("remote evidence file count exceeds protocol bounds")
        for index, (path, expected_sha256) in enumerate(
            zip(
                terminal.evidence_file_paths,
                terminal.evidence_file_sha256s,
                strict=True,
            )
        ):
            body = _read_stable_evidence_leaf(path)
            evidence_total_bytes += len(body)
            if evidence_total_bytes + len(receipt_body) > MAX_RECONCILE_RAW_BYTES:
                raise ValueError("remote raw evidence exceeds its total byte limit")
            if hashlib.sha256(body).hexdigest() != expected_sha256:
                raise ValueError("remote evidence file differs from terminal binding")
            evidence_files.append(
                _RemoteRawEvidenceFile(
                    assignment_sha256=row.assignment_sha256,
                    evidence_index=index,
                    blob=_RemoteRawBlob.from_body(body),
                )
            )
    bundle = _RemoteRawEvidenceBundle(
        schema_version=1,
        host_id=binding.host_id,
        source_request_sha256=request.sha256,
        source_binding_sha256=binding.sha256,
        unknown_result_sha256=evidence_request.unknown_result_sha256,
        schedule_envelope=_RemoteRawBlob.from_body(receipt_body),
        evidence_files=tuple(sorted(evidence_files, key=lambda item: item.key)),
    )
    bundle.canonical_payload(limit_bytes=evidence_request.limit_bytes)
    return bundle


@dataclass(frozen=True)
class _RemoteEvidenceProjection:
    """Private in-memory result of verifying one raw reconciliation bundle."""

    dispatch_schedule_receipt_sha256: str
    dispatch_schedule_receipt_envelope_sha256: str
    completed_assignment_sha256: tuple[str, ...]
    failed_assignment_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(
            "dispatch_schedule_receipt_sha256",
            self.dispatch_schedule_receipt_sha256,
        )
        _require_sha256(
            "dispatch_schedule_receipt_envelope_sha256",
            self.dispatch_schedule_receipt_envelope_sha256,
        )
        for name in (
            "completed_assignment_sha256",
            "failed_assignment_sha256",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
            for value in values:
                _require_sha256(name, value)
        if set(self.completed_assignment_sha256) & set(self.failed_assignment_sha256):
            raise ValueError("remote evidence projection overlaps terminal states")

    @property
    def succeeded(self) -> bool:
        return not self.failed_assignment_sha256


async def execute_host_local_wave_request(
    stdin: bytes,
    *,
    execute_wave: Any = None,
) -> tuple[int, bytes]:
    """Decode coordinator stdin and invoke the existing host-local wave executor.

    The CLI adapter may write the returned bytes directly to stdout.  No raw
    exception text, host path, SSH route, or execution bundle enters it.
    """

    raw_request = _decode_canonical_json_line(
        stdin,
        label="remote worker stdin",
        limit_bytes=MAX_REQUEST_BYTES,
    )
    if raw_request.get("kind") == "lightcone_remote_raw_evidence_request":
        evidence_request = _RemoteEvidenceRequest.from_dict(raw_request)
        request = evidence_request.source_request
        try:
            bundle = await asyncio.to_thread(
                _collect_remote_raw_evidence_bundle,
                evidence_request,
            )
            stdout = bundle.canonical_payload(limit_bytes=evidence_request.limit_bytes)
        except Exception:  # noqa: BLE001 - sensitive-safe fixed-host boundary
            response = _blocked_worker_response(
                request,
                "remote_reconciliation_evidence_unavailable",
            )
            return 42, response.canonical_stdout()
        return 0, stdout
    request = RemoteHostWaveRequest.from_dict(raw_request)
    if request.canonical_stdin() != stdin:
        raise ValueError("remote host-wave stdin changed after strict decoding")
    try:
        publication, verified_plans = _load_verified_host_local_publication(
            request.binding
        )
    except Exception as error:  # noqa: BLE001 - fixed safe worker response
        from lightcone_spec.orchestration.execution_bundle import (
            ExecutionBundleBlockedError,
        )

        reason = (
            error.reason_code
            if isinstance(error, ExecutionBundleBlockedError)
            and _REASON_CODE.fullmatch(error.reason_code) is not None
            else "remote_host_manifest_invalid"
        )
        response = _blocked_worker_response(request, reason)
        return 42, response.canonical_stdout()
    if execute_wave is None:
        from lightcone_spec.orchestration.execution_bundle import (
            execute_dispatch_wave_bundles,
        )

        execute_wave = execute_dispatch_wave_bundles
    try:
        receipt = await execute_wave(
            request.binding.execution_bundle_manifest_path,
            wave_index=request.binding.host_wave_index,
            receipt_output=request.binding.receipt_output_path,
            resume_receipt_path=request.binding.resume_receipt_path,
            expected_resume_receipt_sha256=(request.binding.resume_receipt_sha256),
            expected_resume_receipt_envelope_sha256=(
                request.binding.resume_receipt_envelope_sha256
            ),
            _verified_publication=publication,
            _verified_plans=verified_plans,
        )
    except Exception as error:  # noqa: BLE001 - sensitive-safe process boundary
        from lightcone_spec.orchestration.execution_bundle import (
            ExecutionBundleBlockedError,
        )

        if isinstance(error, ExecutionBundleBlockedError):
            reason = error.reason_code
            if _REASON_CODE.fullmatch(reason) is None:
                reason = "remote_host_wave_blocked"
            response = _blocked_worker_response(request, reason)
        else:
            # The executor may already have durably written INTENT/FINISH
            # journal events.  Without an exact schedule envelope neither
            # failure nor non-execution is known, so emit no authoritative
            # response; the SSH caller records UNKNOWN and must reconcile the
            # same output/journal on the same route.
            return 43, b""
        return 42, response.canonical_stdout()
    try:
        receipt_body = _read_stable_evidence_leaf(
            request.binding.receipt_output_path,
            limit_bytes=MAX_RECONCILE_RECEIPT_BYTES,
        )
        reopened_receipt, _ = _decode_raw_schedule_envelope(receipt_body)
        if reopened_receipt != receipt:
            raise ValueError("worker receipt differs from its published envelope")
        response = _worker_response_from_receipt(
            request,
            receipt,
            receipt_envelope_sha256=_canonical_envelope_sha256(receipt_body),
        )
    except (TypeError, ValueError):
        # A returned-but-foreign receipt is also ambiguous after executor
        # mutation and cannot authorize a direct retry.
        return 43, b""
    return (
        0 if response.status is RemoteWorkerStatus.SUCCEEDED else 42,
        response.canonical_stdout(),
    )


class RemoteTransportOutcome(str, Enum):
    REMOTE_SUCCEEDED = "REMOTE_SUCCEEDED"
    REMOTE_BLOCKED = "REMOTE_BLOCKED"
    REMOTE_FAILED = "REMOTE_FAILED"
    RECONCILED_SUCCEEDED = "RECONCILED_SUCCEEDED"
    RECONCILED_FAILED = "RECONCILED_FAILED"
    REMOTE_OUTCOME_UNKNOWN = "REMOTE_OUTCOME_UNKNOWN"
    SSH_FAILED = "SSH_FAILED"


_OUTCOME_REASON = {
    RemoteTransportOutcome.REMOTE_SUCCEEDED: "remote_wave_succeeded",
    RemoteTransportOutcome.REMOTE_BLOCKED: "remote_wave_blocked",
    RemoteTransportOutcome.REMOTE_FAILED: "remote_wave_failed",
    RemoteTransportOutcome.RECONCILED_SUCCEEDED: "remote_wave_reconciled_succeeded",
    RemoteTransportOutcome.RECONCILED_FAILED: "remote_wave_reconciled_failed",
    RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN: "remote_wave_outcome_unknown",
    RemoteTransportOutcome.SSH_FAILED: "ssh_transport_failed",
}


@dataclass(frozen=True)
class RemoteHostWaveResult:
    """Sensitive-data-free result of one host transport attempt."""

    schema_version: int
    host_id: str
    fleet_inventory_sha256: str
    host_inventory_sha256: str
    fleet_dispatch_plan_sha256: str
    host_dispatch_plan_sha256: str
    fleet_wave_index: int
    fleet_wave_sha256: str
    host_wave_index: int
    host_wave_sha256: str
    execution_bundle_manifest_sha256: str
    assignment_sha256: tuple[str, ...]
    request_sha256: str
    binding_sha256: str
    ssh_route_authority_sha256: str
    transport_outcome: RemoteTransportOutcome
    reason_code: str
    exit_code: int | None
    stdout_size_bytes: int
    stdout_sha256: str
    stderr_size_bytes: int
    stderr_sha256: str
    remote_reason_sha256: str | None
    dispatch_schedule_receipt_sha256: str | None
    dispatch_schedule_receipt_envelope_sha256: str | None
    completed_assignment_sha256: tuple[str, ...]
    failed_assignment_sha256: tuple[str, ...]
    reconciliation_evidence_sha256: str | None = None
    reconciles_unknown_result_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only remote host-wave result schema version 1 is supported"
            )
        _require_id("result host_id", self.host_id)
        for name in (
            "fleet_inventory_sha256",
            "host_inventory_sha256",
            "fleet_dispatch_plan_sha256",
            "host_dispatch_plan_sha256",
            "fleet_wave_sha256",
            "host_wave_sha256",
            "execution_bundle_manifest_sha256",
            "request_sha256",
            "binding_sha256",
            "ssh_route_authority_sha256",
            "stdout_sha256",
            "stderr_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in ("fleet_wave_index", "host_wave_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.transport_outcome, RemoteTransportOutcome):
            raise TypeError("transport_outcome must be a RemoteTransportOutcome")
        if self.reason_code != _OUTCOME_REASON[self.transport_outcome]:
            raise ValueError("result reason code differs from transport outcome")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise TypeError("exit_code must be an integer or null")
        for name in ("stdout_size_bytes", "stderr_size_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.remote_reason_sha256 is not None:
            _require_sha256("remote_reason_sha256", self.remote_reason_sha256)
        if self.dispatch_schedule_receipt_sha256 is not None:
            _require_sha256(
                "dispatch_schedule_receipt_sha256",
                self.dispatch_schedule_receipt_sha256,
            )
        if self.dispatch_schedule_receipt_envelope_sha256 is not None:
            _require_sha256(
                "dispatch_schedule_receipt_envelope_sha256",
                self.dispatch_schedule_receipt_envelope_sha256,
            )
        for name in (
            "reconciliation_evidence_sha256",
            "reconciles_unknown_result_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(name, value)
        for name in (
            "assignment_sha256",
            "completed_assignment_sha256",
            "failed_assignment_sha256",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
            for digest in values:
                _require_sha256(name, digest)
        if set(self.completed_assignment_sha256) & set(self.failed_assignment_sha256):
            raise ValueError("completed and failed assignment identities overlap")
        if not self.assignment_sha256:
            raise ValueError("remote result requires expected assignment identities")
        if (
            set(self.completed_assignment_sha256) | set(self.failed_assignment_sha256)
        ) - set(self.assignment_sha256):
            raise ValueError("remote result claims an assignment outside its binding")
        reconciled = self.transport_outcome in {
            RemoteTransportOutcome.RECONCILED_SUCCEEDED,
            RemoteTransportOutcome.RECONCILED_FAILED,
        }
        has_reconciliation_evidence = self.reconciliation_evidence_sha256 is not None
        has_unknown_authority = self.reconciles_unknown_result_sha256 is not None
        if (
            reconciled and not (has_reconciliation_evidence and has_unknown_authority)
        ) or (
            not reconciled and (has_reconciliation_evidence or has_unknown_authority)
        ):
            raise ValueError("reconciled result lacks exact prior/evidence authority")
        if self.transport_outcome is RemoteTransportOutcome.REMOTE_SUCCEEDED:
            if (
                self.exit_code != 0
                or self.remote_reason_sha256 is not None
                or self.dispatch_schedule_receipt_sha256 is None
                or self.dispatch_schedule_receipt_envelope_sha256 is None
                or self.completed_assignment_sha256 != self.assignment_sha256
                or self.failed_assignment_sha256
            ):
                raise ValueError("successful remote result is incomplete")
        elif self.transport_outcome is RemoteTransportOutcome.RECONCILED_SUCCEEDED:
            if (
                self.exit_code is not None
                or self.remote_reason_sha256 is not None
                or self.dispatch_schedule_receipt_sha256 is None
                or self.dispatch_schedule_receipt_envelope_sha256 is None
                or self.completed_assignment_sha256 != self.assignment_sha256
                or self.failed_assignment_sha256
            ):
                raise ValueError("reconciled successful result is incomplete")
        elif self.transport_outcome in {
            RemoteTransportOutcome.REMOTE_BLOCKED,
            RemoteTransportOutcome.REMOTE_FAILED,
        }:
            if self.exit_code != 42 or self.remote_reason_sha256 is None:
                raise ValueError("negative remote result lacks valid remote authority")
            if self.transport_outcome is RemoteTransportOutcome.REMOTE_BLOCKED and (
                self.dispatch_schedule_receipt_sha256 is not None
                or self.dispatch_schedule_receipt_envelope_sha256 is not None
                or self.completed_assignment_sha256
                or self.failed_assignment_sha256
            ):
                raise ValueError("blocked remote result cannot claim execution")
            if self.transport_outcome is RemoteTransportOutcome.REMOTE_FAILED and (
                self.dispatch_schedule_receipt_sha256 is None
                or self.dispatch_schedule_receipt_envelope_sha256 is None
                or set(self.completed_assignment_sha256)
                | set(self.failed_assignment_sha256)
                != set(self.assignment_sha256)
            ):
                raise ValueError(
                    "failed remote result has incomplete assignment coverage"
                )
        elif self.transport_outcome is RemoteTransportOutcome.SSH_FAILED:
            empty_sha256 = hashlib.sha256(b"").hexdigest()
            if (
                self.exit_code is not None
                or self.stdout_size_bytes != 0
                or self.stdout_sha256 != empty_sha256
                or self.stderr_size_bytes != 0
                or self.stderr_sha256 != empty_sha256
                or self.remote_reason_sha256 is not None
                or self.dispatch_schedule_receipt_sha256 is not None
                or self.dispatch_schedule_receipt_envelope_sha256 is not None
                or self.completed_assignment_sha256
                or self.failed_assignment_sha256
            ):
                raise ValueError("SSH failure must precede remote request dispatch")
        elif self.transport_outcome is RemoteTransportOutcome.RECONCILED_FAILED:
            if (
                self.exit_code is not None
                or self.remote_reason_sha256 is None
                or self.dispatch_schedule_receipt_sha256 is None
                or self.dispatch_schedule_receipt_envelope_sha256 is None
                or set(self.completed_assignment_sha256)
                | set(self.failed_assignment_sha256)
                != set(self.assignment_sha256)
            ):
                raise ValueError("reconciled failed result is incomplete")
        elif (
            self.remote_reason_sha256 is not None
            or self.dispatch_schedule_receipt_sha256 is not None
            or self.dispatch_schedule_receipt_envelope_sha256 is not None
            or self.completed_assignment_sha256
            or self.failed_assignment_sha256
        ):
            raise ValueError("transport failure cannot claim remote completion")

    @cached_property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def succeeded(self) -> bool:
        return self.transport_outcome in {
            RemoteTransportOutcome.REMOTE_SUCCEEDED,
            RemoteTransportOutcome.RECONCILED_SUCCEEDED,
        }

    @property
    def outcome_unknown(self) -> bool:
        return self.transport_outcome is RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN

    @property
    def retryable(self) -> bool:
        return self.transport_outcome in {
            RemoteTransportOutcome.REMOTE_BLOCKED,
            RemoteTransportOutcome.REMOTE_FAILED,
            RemoteTransportOutcome.RECONCILED_FAILED,
            RemoteTransportOutcome.SSH_FAILED,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "lightcone_remote_host_wave_result",
            "host_id": self.host_id,
            "fleet_inventory_sha256": self.fleet_inventory_sha256,
            "host_inventory_sha256": self.host_inventory_sha256,
            "fleet_dispatch_plan_sha256": self.fleet_dispatch_plan_sha256,
            "host_dispatch_plan_sha256": self.host_dispatch_plan_sha256,
            "fleet_wave_index": self.fleet_wave_index,
            "fleet_wave_sha256": self.fleet_wave_sha256,
            "host_wave_index": self.host_wave_index,
            "host_wave_sha256": self.host_wave_sha256,
            "execution_bundle_manifest_sha256": (self.execution_bundle_manifest_sha256),
            "assignment_sha256": list(self.assignment_sha256),
            "request_sha256": self.request_sha256,
            "binding_sha256": self.binding_sha256,
            "ssh_route_authority_sha256": self.ssh_route_authority_sha256,
            "transport_outcome": self.transport_outcome.value,
            "reason_code": self.reason_code,
            "exit_code": self.exit_code,
            "stdout_size_bytes": self.stdout_size_bytes,
            "stdout_sha256": self.stdout_sha256,
            "stderr_size_bytes": self.stderr_size_bytes,
            "stderr_sha256": self.stderr_sha256,
            "remote_reason_sha256": self.remote_reason_sha256,
            "dispatch_schedule_receipt_sha256": (self.dispatch_schedule_receipt_sha256),
            "dispatch_schedule_receipt_envelope_sha256": (
                self.dispatch_schedule_receipt_envelope_sha256
            ),
            "completed_assignment_sha256": list(self.completed_assignment_sha256),
            "failed_assignment_sha256": list(self.failed_assignment_sha256),
            "reconciliation_evidence_sha256": (self.reconciliation_evidence_sha256),
            "reconciles_unknown_result_sha256": (self.reconciles_unknown_result_sha256),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = frozenset(
            {
                "schema_version",
                "kind",
                "host_id",
                "fleet_inventory_sha256",
                "host_inventory_sha256",
                "fleet_dispatch_plan_sha256",
                "host_dispatch_plan_sha256",
                "fleet_wave_index",
                "fleet_wave_sha256",
                "host_wave_index",
                "host_wave_sha256",
                "execution_bundle_manifest_sha256",
                "assignment_sha256",
                "request_sha256",
                "binding_sha256",
                "ssh_route_authority_sha256",
                "transport_outcome",
                "reason_code",
                "exit_code",
                "stdout_size_bytes",
                "stdout_sha256",
                "stderr_size_bytes",
                "stderr_sha256",
                "remote_reason_sha256",
                "dispatch_schedule_receipt_sha256",
                "dispatch_schedule_receipt_envelope_sha256",
                "completed_assignment_sha256",
                "failed_assignment_sha256",
                "reconciliation_evidence_sha256",
                "reconciles_unknown_result_sha256",
            }
        )
        row = _strict_object("remote host-wave result", value, fields)
        if row["kind"] != "lightcone_remote_host_wave_result":
            raise ValueError("remote host-wave result kind is unsupported")
        try:
            outcome = RemoteTransportOutcome(row["transport_outcome"])
        except (TypeError, ValueError) as error:
            raise ValueError("remote transport outcome is unsupported") from error
        exit_code = row["exit_code"]
        if exit_code is not None:
            exit_code = _strict_int("exit_code", exit_code)
        return cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            host_id=_require_id("host_id", row["host_id"]),
            fleet_inventory_sha256=_require_sha256(
                "fleet_inventory_sha256", row["fleet_inventory_sha256"]
            ),
            host_inventory_sha256=_require_sha256(
                "host_inventory_sha256", row["host_inventory_sha256"]
            ),
            fleet_dispatch_plan_sha256=_require_sha256(
                "fleet_dispatch_plan_sha256", row["fleet_dispatch_plan_sha256"]
            ),
            host_dispatch_plan_sha256=_require_sha256(
                "host_dispatch_plan_sha256", row["host_dispatch_plan_sha256"]
            ),
            fleet_wave_index=_strict_int("fleet_wave_index", row["fleet_wave_index"]),
            fleet_wave_sha256=_require_sha256(
                "fleet_wave_sha256", row["fleet_wave_sha256"]
            ),
            host_wave_index=_strict_int("host_wave_index", row["host_wave_index"]),
            host_wave_sha256=_require_sha256(
                "host_wave_sha256", row["host_wave_sha256"]
            ),
            execution_bundle_manifest_sha256=_require_sha256(
                "execution_bundle_manifest_sha256",
                row["execution_bundle_manifest_sha256"],
            ),
            assignment_sha256=tuple(
                _require_sha256("assignment", item)
                for item in _strict_list(
                    "expected assignments", row["assignment_sha256"]
                )
            ),
            request_sha256=_require_sha256("request_sha256", row["request_sha256"]),
            binding_sha256=_require_sha256("binding_sha256", row["binding_sha256"]),
            ssh_route_authority_sha256=_require_sha256(
                "ssh_route_authority_sha256",
                row["ssh_route_authority_sha256"],
            ),
            transport_outcome=outcome,
            reason_code=_require_reason("reason_code", row["reason_code"]),
            exit_code=exit_code,
            stdout_size_bytes=_strict_int(
                "stdout_size_bytes", row["stdout_size_bytes"]
            ),
            stdout_sha256=_require_sha256("stdout_sha256", row["stdout_sha256"]),
            stderr_size_bytes=_strict_int(
                "stderr_size_bytes", row["stderr_size_bytes"]
            ),
            stderr_sha256=_require_sha256("stderr_sha256", row["stderr_sha256"]),
            remote_reason_sha256=_optional_sha256(
                "remote_reason_sha256", row["remote_reason_sha256"]
            ),
            dispatch_schedule_receipt_sha256=_optional_sha256(
                "dispatch_schedule_receipt_sha256",
                row["dispatch_schedule_receipt_sha256"],
            ),
            dispatch_schedule_receipt_envelope_sha256=_optional_sha256(
                "dispatch_schedule_receipt_envelope_sha256",
                row["dispatch_schedule_receipt_envelope_sha256"],
            ),
            completed_assignment_sha256=tuple(
                _require_sha256("completed assignment", item)
                for item in _strict_list(
                    "completed assignments", row["completed_assignment_sha256"]
                )
            ),
            failed_assignment_sha256=tuple(
                _require_sha256("failed assignment", item)
                for item in _strict_list(
                    "failed assignments", row["failed_assignment_sha256"]
                )
            ),
            reconciliation_evidence_sha256=_optional_sha256(
                "reconciliation_evidence_sha256",
                row["reconciliation_evidence_sha256"],
            ),
            reconciles_unknown_result_sha256=_optional_sha256(
                "reconciles_unknown_result_sha256",
                row["reconciles_unknown_result_sha256"],
            ),
        )


def _decode_unbound_schedule_receipt(value: object) -> object:
    """Recompute receipt structure without trusting a remote plan summary."""

    from lightcone_spec.experiments.gpu_pool import (
        DispatchExecutionPhase,
        DispatchScheduleReceipt,
        DispatchWaveExecutionReceipt,
    )

    fields = frozenset(
        {
            "plan_sha256",
            "phase",
            "wave_receipts",
            "wave_receipt_sha256",
            "inventory_sha256",
            "fixed_instance_gpu_count",
            "active_intervals_monotonic_ns",
            "active_interval_union_ns",
            "fixed_instance_actual_billed_gpu_ns",
            "per_assignment_attributed_gpu_ns",
            "per_assignment_attributed_fixed_instance_gpu_ns",
            "prior_schedule_receipt_sha256",
            "accounting_semantics",
        }
    )
    row = _strict_object("raw dispatch schedule receipt", value, fields)
    waves = tuple(
        DispatchWaveExecutionReceipt.from_dict(item)
        for item in _strict_list("raw dispatch wave receipts", row["wave_receipts"])
    )
    declared_waves = tuple(
        _require_sha256("raw wave receipt SHA-256", item)
        for item in _strict_list(
            "raw wave receipt SHA-256 list", row["wave_receipt_sha256"]
        )
    )
    if declared_waves != tuple(wave.sha256 for wave in waves):
        raise ValueError("raw schedule wave receipt identities differ")
    intervals: list[tuple[int, int]] = []
    for item in _strict_list(
        "raw schedule active intervals", row["active_intervals_monotonic_ns"]
    ):
        endpoints = _strict_list("raw schedule active interval", item)
        if len(endpoints) != 2:
            raise ValueError("raw schedule active interval must have two endpoints")
        intervals.append(
            (
                _strict_int("raw interval start", endpoints[0]),
                _strict_int("raw interval finish", endpoints[1]),
            )
        )
    prior = row["prior_schedule_receipt_sha256"]
    if prior is not None:
        prior = _require_sha256("prior_schedule_receipt_sha256", prior)
    try:
        phase = DispatchExecutionPhase(row["phase"])
    except (TypeError, ValueError) as error:
        raise ValueError("raw schedule phase is unsupported") from error
    receipt = DispatchScheduleReceipt(
        plan_sha256=_require_sha256("raw schedule plan", row["plan_sha256"]),
        phase=phase,
        wave_receipts=waves,
        inventory_sha256=_require_sha256(
            "raw schedule inventory", row["inventory_sha256"]
        ),
        fixed_instance_gpu_count=_strict_int(
            "raw schedule GPU count", row["fixed_instance_gpu_count"]
        ),
        active_intervals_monotonic_ns=tuple(intervals),
        fixed_instance_actual_billed_gpu_ns=_strict_int(
            "raw schedule billed GPU ns",
            row["fixed_instance_actual_billed_gpu_ns"],
        ),
        per_assignment_attributed_gpu_ns=_strict_int(
            "raw schedule attributed GPU ns",
            row["per_assignment_attributed_gpu_ns"],
        ),
        per_assignment_attributed_fixed_instance_gpu_ns=_strict_int(
            "raw schedule fixed-instance attributed GPU ns",
            row["per_assignment_attributed_fixed_instance_gpu_ns"],
        ),
        prior_schedule_receipt_sha256=prior,
        accounting_semantics=row["accounting_semantics"],  # type: ignore[arg-type]
    )
    if receipt.to_dict() != row:
        raise ValueError("raw schedule receipt changed during strict replay")
    return receipt


def _decode_raw_schedule_envelope(body: bytes) -> tuple[object, object]:
    """Recompute the canonical schema-v2 receipt and journal binding."""

    from lightcone_spec.experiments.gpu_pool import ArtifactSidecar
    from lightcone_spec.orchestration.execution_bundle import (
        DispatchAttemptJournalBinding,
    )

    envelope = _decode_canonical_json_line(
        body,
        label="raw dispatch schedule envelope",
        limit_bytes=MAX_RECONCILE_RECEIPT_BYTES,
    )
    row = _strict_object(
        "raw dispatch schedule envelope",
        envelope,
        frozenset(
            {
                "schema_version",
                "kind",
                "receipt",
                "sidecar",
                "attempt_journal",
            }
        ),
    )
    if (
        row["schema_version"] != 2
        or row["kind"] != "industrial_dispatch_schedule_receipt_envelope"
    ):
        raise ValueError("raw dispatch receipt envelope is not schema-v2")
    receipt = _decode_unbound_schedule_receipt(row["receipt"])
    sidecar = ArtifactSidecar.from_dict(row["sidecar"])
    if (
        sidecar.artifact_kind != "dispatch_schedule_receipt.v1"
        or sidecar.artifact_sha256 != receipt.sha256
    ):
        raise ValueError("raw dispatch receipt sidecar identity differs")
    journal = DispatchAttemptJournalBinding.from_dict(row["attempt_journal"])
    return receipt, journal


def _projection_from_raw_bundle(
    raw: _RemoteRawEvidenceBundle,
    *,
    request: RemoteHostWaveRequest,
    unknown_result_sha256: str,
) -> _RemoteEvidenceProjection:
    """Recompute every transferable identity before deriving a path-free view."""

    from lightcone_spec.experiments.gpu_pool import (
        AssignmentExecutionStatus,
        DispatchExecutionPhase,
    )

    binding = request.binding
    if (
        raw.host_id != binding.host_id
        or raw.source_request_sha256 != request.sha256
        or raw.source_binding_sha256 != binding.sha256
        or raw.unknown_result_sha256 != unknown_result_sha256
    ):
        raise ValueError("raw reconciliation bundle differs from UNKNOWN authority")
    receipt, _ = _decode_raw_schedule_envelope(raw.schedule_envelope.body)
    response = _worker_response_from_receipt(
        request,
        receipt,
        receipt_envelope_sha256=raw.schedule_envelope.sha256,
    )
    if receipt.phase is DispatchExecutionPhase.RUNNING:
        raise ValueError("raw target wave receipt is not terminal")
    wave = receipt.wave_receipts[binding.host_wave_index]
    expected_files: dict[tuple[str, int], str] = {}
    for row in wave.assignment_receipts:
        if row.status is not AssignmentExecutionStatus.SUCCEEDED:
            continue
        terminal = row.terminal_binding
        if terminal is None:  # pragma: no cover - receipt invariant
            raise ValueError("raw successful receipt lacks terminal evidence")
        if len(terminal.evidence_file_paths) != len(terminal.evidence_file_sha256s):
            raise ValueError("raw terminal evidence coverage differs")
        for index, digest in enumerate(terminal.evidence_file_sha256s):
            expected_files[(row.assignment_sha256, index)] = digest
    actual_files = {item.key: item for item in raw.evidence_files}
    if set(actual_files) != set(expected_files):
        raise ValueError("raw reconciliation evidence file coverage differs")
    for key, expected_sha256 in expected_files.items():
        blob = actual_files[key].blob
        if (
            blob.sha256 != expected_sha256
            or hashlib.sha256(blob.body).hexdigest() != expected_sha256
        ):
            raise ValueError("raw reconciliation evidence content differs")
    return _RemoteEvidenceProjection(
        dispatch_schedule_receipt_sha256=receipt.sha256,
        dispatch_schedule_receipt_envelope_sha256=(
            _canonical_envelope_sha256(raw.schedule_envelope.body)
        ),
        completed_assignment_sha256=response.completed_assignment_sha256,
        failed_assignment_sha256=response.failed_assignment_sha256,
    )


@dataclass(frozen=True)
class _SshRemoteEvidenceFetcher:
    """Production fixed-host raw fetch."""

    timeout_seconds: float = DEFAULT_REMOTE_TIMEOUT_SECONDS
    stderr_limit_bytes: int = DEFAULT_STDERR_LIMIT_BYTES

    async def fetch(
        self,
        *,
        route: SshHostRoute,
        request: RemoteHostWaveRequest,
        unknown_result_sha256: str,
        limit_bytes: int,
    ) -> bytes | None:
        evidence_request = _RemoteEvidenceRequest(
            schema_version=1,
            source_request=request,
            unknown_result_sha256=unknown_result_sha256,
            limit_bytes=limit_bytes,
        )
        try:
            stdin = evidence_request.canonical_stdin()
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
        try:
            with _prepared_ssh_route(route) as (argv, route_authority):
                if route_authority != request.ssh_route_authority_sha256:
                    return None
                process = await AsyncioSshTransport().run(
                    argv=argv,
                    stdin=stdin,
                    environment={
                        "LC_ALL": "C",
                        "SSH_AUTH_SOCK": route.agent_socket_path,
                    },
                    timeout_seconds=_positive_finite(
                        "reconciliation timeout", self.timeout_seconds
                    ),
                    stdout_limit_bytes=limit_bytes,
                    stderr_limit_bytes=_positive_limit(
                        "reconciliation stderr limit", self.stderr_limit_bytes
                    ),
                )
        except Exception:  # noqa: BLE001 - unavailable evidence preserves UNKNOWN
            return None
        if (
            type(process) is not SshProcessResult
            or process.exit_code != 0
            or len(process.stdout) > limit_bytes
            or len(process.stderr) > self.stderr_limit_bytes
        ):
            return None
        return process.stdout


def _result_matches_request(
    result: RemoteHostWaveResult,
    request: RemoteHostWaveRequest,
) -> bool:
    binding = request.binding
    return (
        result.host_id == binding.host_id
        and result.fleet_inventory_sha256 == binding.fleet_inventory_sha256
        and result.host_inventory_sha256 == binding.host_inventory_sha256
        and result.fleet_dispatch_plan_sha256 == binding.fleet_dispatch_plan_sha256
        and result.host_dispatch_plan_sha256 == binding.host_dispatch_plan_sha256
        and result.fleet_wave_index == binding.fleet_wave_index
        and result.fleet_wave_sha256 == binding.fleet_wave_sha256
        and result.host_wave_index == binding.host_wave_index
        and result.host_wave_sha256 == binding.host_wave_sha256
        and result.execution_bundle_manifest_sha256
        == binding.execution_bundle_manifest_sha256
        and result.assignment_sha256 == binding.assignment_sha256
        and result.request_sha256 == request.sha256
        and result.binding_sha256 == binding.sha256
        and result.ssh_route_authority_sha256 == request.ssh_route_authority_sha256
    )


async def reconcile_remote_host_wave(
    route: SshHostRoute,
    request: RemoteHostWaveRequest,
    unknown_result: RemoteHostWaveResult,
    *,
    evidence_limit_bytes: int = MAX_RECONCILE_EVIDENCE_BYTES,
) -> RemoteHostWaveResult:
    """Resolve one timeout only from fixed-host raw remote evidence.

    Missing or temporarily unreachable evidence preserves the exact UNKNOWN.
    The path-bearing receipt exists only in the bounded SSH response in memory;
    the returned artifact contains only identities recomputed from its bytes.
    """

    if type(route) is not SshHostRoute:
        raise TypeError("remote reconciliation requires an exact SshHostRoute")
    if type(request) is not RemoteHostWaveRequest:
        raise TypeError("remote reconciliation requires an exact request")
    if type(unknown_result) is not RemoteHostWaveResult:
        raise TypeError("remote reconciliation requires an exact UNKNOWN result")
    if route.host_id != request.binding.host_id:
        raise ValueError("reconciliation route host differs from its request")
    if route.authority_sha256 != request.ssh_route_authority_sha256:
        raise ValueError("reconciliation route authority differs from its request")
    if not unknown_result.outcome_unknown:
        raise ValueError("only an UNKNOWN remote outcome can be reconciled")
    if not _result_matches_request(unknown_result, request):
        raise ValueError("UNKNOWN result differs from its original request")
    limit = _positive_limit("evidence_limit_bytes", evidence_limit_bytes)
    if limit > MAX_RECONCILE_EVIDENCE_BYTES:
        raise ValueError("evidence_limit_bytes exceeds the protocol maximum")
    fetcher = _SshRemoteEvidenceFetcher()
    try:
        payload = await fetcher.fetch(
            route=route,
            request=request,
            unknown_result_sha256=unknown_result.sha256,
            limit_bytes=limit,
        )
    except (OSError, TimeoutError):
        return unknown_result
    if payload is None:
        return unknown_result
    if type(payload) is not bytes:
        raise TypeError("remote evidence fetcher must return bytes or null")
    if len(payload) > limit:
        raise ValueError("remote evidence fetch exceeds the requested byte limit")
    raw = _decode_remote_raw_evidence_bundle(payload, limit_bytes=limit)
    evidence = _projection_from_raw_bundle(
        raw,
        request=request,
        unknown_result_sha256=unknown_result.sha256,
    )
    outcome = (
        RemoteTransportOutcome.RECONCILED_SUCCEEDED
        if evidence.succeeded
        else RemoteTransportOutcome.RECONCILED_FAILED
    )
    return RemoteHostWaveResult(
        schema_version=1,
        host_id=unknown_result.host_id,
        fleet_inventory_sha256=unknown_result.fleet_inventory_sha256,
        host_inventory_sha256=unknown_result.host_inventory_sha256,
        fleet_dispatch_plan_sha256=unknown_result.fleet_dispatch_plan_sha256,
        host_dispatch_plan_sha256=unknown_result.host_dispatch_plan_sha256,
        fleet_wave_index=unknown_result.fleet_wave_index,
        fleet_wave_sha256=unknown_result.fleet_wave_sha256,
        host_wave_index=unknown_result.host_wave_index,
        host_wave_sha256=unknown_result.host_wave_sha256,
        execution_bundle_manifest_sha256=(
            unknown_result.execution_bundle_manifest_sha256
        ),
        assignment_sha256=unknown_result.assignment_sha256,
        request_sha256=unknown_result.request_sha256,
        binding_sha256=unknown_result.binding_sha256,
        ssh_route_authority_sha256=unknown_result.ssh_route_authority_sha256,
        transport_outcome=outcome,
        reason_code=_OUTCOME_REASON[outcome],
        exit_code=None,
        stdout_size_bytes=unknown_result.stdout_size_bytes,
        stdout_sha256=unknown_result.stdout_sha256,
        stderr_size_bytes=unknown_result.stderr_size_bytes,
        stderr_sha256=unknown_result.stderr_sha256,
        remote_reason_sha256=(
            None
            if evidence.succeeded
            else canonical_sha256({"reason_code": "remote_host_wave_failed"})
        ),
        dispatch_schedule_receipt_sha256=(evidence.dispatch_schedule_receipt_sha256),
        dispatch_schedule_receipt_envelope_sha256=(
            evidence.dispatch_schedule_receipt_envelope_sha256
        ),
        completed_assignment_sha256=evidence.completed_assignment_sha256,
        failed_assignment_sha256=evidence.failed_assignment_sha256,
        reconciliation_evidence_sha256=raw.sha256,
        reconciles_unknown_result_sha256=unknown_result.sha256,
    )


class FleetWaveOutcome(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RemoteFleetWaveReceipt:
    """Aggregate host outcomes without discarding a successful node receipt."""

    schema_version: int
    fleet_inventory_sha256: str
    fleet_dispatch_plan_sha256: str
    fleet_wave_index: int
    fleet_wave_sha256: str
    host_results: tuple[RemoteHostWaveResult, ...]
    prior_fleet_wave_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only fleet-wave receipt schema version 1 is supported")
        _require_sha256("fleet_inventory_sha256", self.fleet_inventory_sha256)
        _require_sha256("fleet_dispatch_plan_sha256", self.fleet_dispatch_plan_sha256)
        _require_sha256("fleet_wave_sha256", self.fleet_wave_sha256)
        if type(self.fleet_wave_index) is not int or self.fleet_wave_index < 0:
            raise ValueError("fleet_wave_index must be a non-negative integer")
        if not self.host_results or any(
            type(item) is not RemoteHostWaveResult for item in self.host_results
        ):
            raise ValueError("fleet-wave receipt requires exact host results")
        host_ids = tuple(result.host_id for result in self.host_results)
        if host_ids != tuple(sorted(set(host_ids))):
            raise ValueError("fleet host results must be sorted and unique")
        if any(
            result.fleet_inventory_sha256 != self.fleet_inventory_sha256
            or result.fleet_dispatch_plan_sha256 != self.fleet_dispatch_plan_sha256
            or result.fleet_wave_index != self.fleet_wave_index
            or result.fleet_wave_sha256 != self.fleet_wave_sha256
            for result in self.host_results
        ):
            raise ValueError("fleet-wave receipt mixes fleet execution authorities")
        assignment_ids = tuple(
            assignment
            for result in self.host_results
            for assignment in result.assignment_sha256
        )
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("fleet-wave receipt migrates or duplicates an assignment")
        if self.prior_fleet_wave_receipt_sha256 is not None:
            _require_sha256(
                "prior_fleet_wave_receipt_sha256",
                self.prior_fleet_wave_receipt_sha256,
            )

    @property
    def outcome(self) -> FleetWaveOutcome:
        if any(result.outcome_unknown for result in self.host_results):
            return FleetWaveOutcome.UNKNOWN
        successes = sum(result.succeeded for result in self.host_results)
        if successes == len(self.host_results):
            return FleetWaveOutcome.COMPLETE
        if successes:
            return FleetWaveOutcome.PARTIAL
        return FleetWaveOutcome.FAILED

    @cached_property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "lightcone_fleet_wave_receipt",
            "fleet_inventory_sha256": self.fleet_inventory_sha256,
            "fleet_dispatch_plan_sha256": self.fleet_dispatch_plan_sha256,
            "fleet_wave_index": self.fleet_wave_index,
            "fleet_wave_sha256": self.fleet_wave_sha256,
            "outcome": self.outcome.value,
            "host_results": [result.to_dict() for result in self.host_results],
            "host_result_sha256": [result.sha256 for result in self.host_results],
            "prior_fleet_wave_receipt_sha256": (self.prior_fleet_wave_receipt_sha256),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "fleet-wave receipt",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "fleet_inventory_sha256",
                    "fleet_dispatch_plan_sha256",
                    "fleet_wave_index",
                    "fleet_wave_sha256",
                    "outcome",
                    "host_results",
                    "host_result_sha256",
                    "prior_fleet_wave_receipt_sha256",
                }
            ),
        )
        if row["kind"] != "lightcone_fleet_wave_receipt":
            raise ValueError("fleet-wave receipt kind is unsupported")
        results = tuple(
            RemoteHostWaveResult.from_dict(item)
            for item in _strict_list("fleet host results", row["host_results"])
        )
        declared = tuple(
            _require_sha256("host result SHA-256", item)
            for item in _strict_list(
                "host result SHA-256 list", row["host_result_sha256"]
            )
        )
        if declared != tuple(result.sha256 for result in results):
            raise ValueError("fleet host-result SHA-256 list mismatch")
        receipt = cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            fleet_inventory_sha256=_require_sha256(
                "fleet_inventory_sha256", row["fleet_inventory_sha256"]
            ),
            fleet_dispatch_plan_sha256=_require_sha256(
                "fleet_dispatch_plan_sha256", row["fleet_dispatch_plan_sha256"]
            ),
            fleet_wave_index=_strict_int("fleet_wave_index", row["fleet_wave_index"]),
            fleet_wave_sha256=_require_sha256(
                "fleet_wave_sha256", row["fleet_wave_sha256"]
            ),
            host_results=results,
            prior_fleet_wave_receipt_sha256=_optional_sha256(
                "prior_fleet_wave_receipt_sha256",
                row["prior_fleet_wave_receipt_sha256"],
            ),
        )
        if row["outcome"] != receipt.outcome.value:
            raise ValueError("fleet-wave outcome differs from host results")
        return receipt


def reconcile_fleet_wave_receipt(
    prior_receipt: RemoteFleetWaveReceipt,
    reconciled_results: Sequence[RemoteHostWaveResult],
) -> RemoteFleetWaveReceipt:
    """Replace only UNKNOWN hosts with exact evidence-bound reconciliations."""

    if type(prior_receipt) is not RemoteFleetWaveReceipt:
        raise TypeError("fleet reconciliation requires an exact prior receipt")
    if RemoteFleetWaveReceipt.from_dict(prior_receipt.to_dict()) != prior_receipt:
        raise ValueError("prior fleet receipt changed during strict replay")
    if not reconciled_results:
        raise ValueError("fleet reconciliation requires at least one host result")
    if any(type(item) is not RemoteHostWaveResult for item in reconciled_results):
        raise TypeError("fleet reconciliation requires exact host results")
    replacement_by_host = {item.host_id: item for item in reconciled_results}
    if len(replacement_by_host) != len(reconciled_results):
        raise ValueError("fleet reconciliation contains duplicate host results")
    prior_by_host = {item.host_id: item for item in prior_receipt.host_results}
    if not set(replacement_by_host) <= set(prior_by_host):
        raise ValueError("fleet reconciliation introduces a foreign host")
    for host_id, replacement in replacement_by_host.items():
        previous = prior_by_host[host_id]
        if not previous.outcome_unknown:
            raise ValueError("fleet reconciliation can replace only UNKNOWN hosts")
        if replacement.transport_outcome not in {
            RemoteTransportOutcome.RECONCILED_SUCCEEDED,
            RemoteTransportOutcome.RECONCILED_FAILED,
        }:
            raise ValueError("fleet reconciliation result lacks remote evidence")
        if (
            replacement.reconciles_unknown_result_sha256 != previous.sha256
            or replacement.host_id != previous.host_id
            or replacement.fleet_inventory_sha256 != previous.fleet_inventory_sha256
            or replacement.host_inventory_sha256 != previous.host_inventory_sha256
            or replacement.fleet_dispatch_plan_sha256
            != previous.fleet_dispatch_plan_sha256
            or replacement.host_dispatch_plan_sha256
            != previous.host_dispatch_plan_sha256
            or replacement.fleet_wave_index != previous.fleet_wave_index
            or replacement.fleet_wave_sha256 != previous.fleet_wave_sha256
            or replacement.host_wave_index != previous.host_wave_index
            or replacement.host_wave_sha256 != previous.host_wave_sha256
            or replacement.execution_bundle_manifest_sha256
            != previous.execution_bundle_manifest_sha256
            or replacement.assignment_sha256 != previous.assignment_sha256
            or replacement.request_sha256 != previous.request_sha256
            or replacement.binding_sha256 != previous.binding_sha256
            or replacement.ssh_route_authority_sha256
            != previous.ssh_route_authority_sha256
        ):
            raise ValueError("fleet reconciliation changes host execution identity")
    merged = dict(prior_by_host)
    merged.update(replacement_by_host)
    return RemoteFleetWaveReceipt(
        schema_version=1,
        fleet_inventory_sha256=prior_receipt.fleet_inventory_sha256,
        fleet_dispatch_plan_sha256=prior_receipt.fleet_dispatch_plan_sha256,
        fleet_wave_index=prior_receipt.fleet_wave_index,
        fleet_wave_sha256=prior_receipt.fleet_wave_sha256,
        host_results=tuple(merged[key] for key in sorted(merged)),
        prior_fleet_wave_receipt_sha256=prior_receipt.sha256,
    )


@dataclass(frozen=True)
class SshProcessResult:
    exit_code: int
    stdout: bytes
    stderr: bytes

    def __post_init__(self) -> None:
        if type(self.exit_code) is not int:
            raise TypeError("SSH exit code must be an integer")
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise TypeError("SSH output streams must be bytes")


class SshTransportTimedOut(TimeoutError):
    pass


class SshOutputLimitExceeded(RuntimeError):
    pass


class AsyncSshTransport(Protocol):
    async def run(
        self,
        *,
        argv: tuple[str, ...],
        stdin: bytes,
        environment: Mapping[str, str],
        timeout_seconds: float,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
    ) -> SshProcessResult: ...


async def _read_bounded(
    stream: asyncio.StreamReader,
    *,
    limit_bytes: int,
) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await stream.read(8192):
        size += len(chunk)
        if size > limit_bytes:
            raise SshOutputLimitExceeded("SSH output exceeded its byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


class AsyncioSshTransport:
    """OpenSSH subprocess transport with concurrent bounded stream draining."""

    async def run(
        self,
        *,
        argv: tuple[str, ...],
        stdin: bytes,
        environment: Mapping[str, str],
        timeout_seconds: float,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
    ) -> SshProcessResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(environment),
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            await process.wait()
            raise RuntimeError("SSH subprocess pipes are unavailable")
        stdout_task = asyncio.create_task(
            _read_bounded(process.stdout, limit_bytes=stdout_limit_bytes)
        )
        stderr_task = asyncio.create_task(
            _read_bounded(process.stderr, limit_bytes=stderr_limit_bytes)
        )

        async def finish() -> SshProcessResult:
            process.stdin.write(stdin)
            try:
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            process.stdin.close()
            try:
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
            stdout, stderr, exit_code = await asyncio.gather(
                stdout_task,
                stderr_task,
                process.wait(),
            )
            return SshProcessResult(exit_code, stdout, stderr)

        try:
            async with asyncio.timeout(timeout_seconds):
                return await finish()
        except TimeoutError as error:
            process.kill()
            await process.wait()
            await _cancel_stream_tasks(stdout_task, stderr_task)
            raise SshTransportTimedOut("SSH subprocess timed out") from error
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            await _cancel_stream_tasks(stdout_task, stderr_task)
            raise


async def _cancel_stream_tasks(*tasks: asyncio.Task[bytes]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _stream_identity(value: bytes) -> tuple[int, str]:
    return len(value), hashlib.sha256(value).hexdigest()


def _empty_result(
    request: RemoteHostWaveRequest,
    outcome: RemoteTransportOutcome,
    *,
    exit_code: int | None,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> RemoteHostWaveResult:
    binding = request.binding
    stdout_size, stdout_sha256 = _stream_identity(stdout)
    stderr_size, stderr_sha256 = _stream_identity(stderr)
    return RemoteHostWaveResult(
        schema_version=1,
        host_id=binding.host_id,
        fleet_inventory_sha256=binding.fleet_inventory_sha256,
        host_inventory_sha256=binding.host_inventory_sha256,
        fleet_dispatch_plan_sha256=binding.fleet_dispatch_plan_sha256,
        host_dispatch_plan_sha256=binding.host_dispatch_plan_sha256,
        fleet_wave_index=binding.fleet_wave_index,
        fleet_wave_sha256=binding.fleet_wave_sha256,
        host_wave_index=binding.host_wave_index,
        host_wave_sha256=binding.host_wave_sha256,
        execution_bundle_manifest_sha256=(binding.execution_bundle_manifest_sha256),
        assignment_sha256=binding.assignment_sha256,
        request_sha256=request.sha256,
        binding_sha256=binding.sha256,
        ssh_route_authority_sha256=request.ssh_route_authority_sha256,
        transport_outcome=outcome,
        reason_code=_OUTCOME_REASON[outcome],
        exit_code=exit_code,
        stdout_size_bytes=stdout_size,
        stdout_sha256=stdout_sha256,
        stderr_size_bytes=stderr_size,
        stderr_sha256=stderr_sha256,
        remote_reason_sha256=None,
        dispatch_schedule_receipt_sha256=None,
        dispatch_schedule_receipt_envelope_sha256=None,
        completed_assignment_sha256=(),
        failed_assignment_sha256=(),
    )


def _decode_remote_response(
    request: RemoteHostWaveRequest,
    process: SshProcessResult,
) -> RemoteHostWaveResult:
    if process.exit_code == 255:
        return _empty_result(
            request,
            RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN,
            exit_code=process.exit_code,
            stdout=process.stdout,
            stderr=process.stderr,
        )
    if not process.stdout.endswith(b"\n") or process.stdout.endswith(b"\n\n"):
        return _empty_result(
            request,
            RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN,
            exit_code=process.exit_code,
            stdout=process.stdout,
            stderr=process.stderr,
        )
    body = process.stdout[:-1]
    try:
        row = _strict_json_object(body)
        if canonical_json_bytes(row) != body:
            raise ValueError("remote response is not canonical JSON")
        response = RemoteHostWaveResponse.from_dict(row)
        if (
            response.host_id != request.binding.host_id
            or response.request_sha256 != request.sha256
            or response.binding_sha256 != request.binding.sha256
        ):
            raise ValueError("remote response authority mismatch")
        if response.status is RemoteWorkerStatus.SUCCEEDED and process.exit_code == 0:
            outcome = RemoteTransportOutcome.REMOTE_SUCCEEDED
        elif response.status is RemoteWorkerStatus.BLOCKED and process.exit_code == 42:
            outcome = RemoteTransportOutcome.REMOTE_BLOCKED
        elif response.status is RemoteWorkerStatus.FAILED and process.exit_code == 42:
            outcome = RemoteTransportOutcome.REMOTE_FAILED
        else:
            raise ValueError("remote response status and exit code disagree")
        if outcome is RemoteTransportOutcome.REMOTE_SUCCEEDED:
            remote_reason_sha256 = None
        else:
            remote_reason_sha256 = canonical_sha256(
                {"reason_code": response.reason_code}
            )
        receipt_sha256 = response.dispatch_schedule_receipt_sha256
        receipt_envelope_sha256 = response.dispatch_schedule_receipt_envelope_sha256
        completed = response.completed_assignment_sha256
        failed = response.failed_assignment_sha256
        expected = set(request.binding.assignment_sha256)
        if (set(completed) | set(failed)) - expected or (
            outcome is RemoteTransportOutcome.REMOTE_FAILED
            and set(completed) | set(failed) != expected
        ):
            raise ValueError("remote assignment outcome coverage is invalid")
        if outcome is RemoteTransportOutcome.REMOTE_SUCCEEDED and (
            completed != request.binding.assignment_sha256
            or failed
            or receipt_sha256 is None
        ):
            raise ValueError("successful remote response is incomplete")
    except (TypeError, ValueError):
        return _empty_result(
            request,
            RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN,
            exit_code=process.exit_code,
            stdout=process.stdout,
            stderr=process.stderr,
        )
    stdout_size, stdout_sha256 = _stream_identity(process.stdout)
    stderr_size, stderr_sha256 = _stream_identity(process.stderr)
    return RemoteHostWaveResult(
        schema_version=1,
        host_id=request.binding.host_id,
        fleet_inventory_sha256=request.binding.fleet_inventory_sha256,
        host_inventory_sha256=request.binding.host_inventory_sha256,
        fleet_dispatch_plan_sha256=request.binding.fleet_dispatch_plan_sha256,
        host_dispatch_plan_sha256=request.binding.host_dispatch_plan_sha256,
        fleet_wave_index=request.binding.fleet_wave_index,
        fleet_wave_sha256=request.binding.fleet_wave_sha256,
        host_wave_index=request.binding.host_wave_index,
        host_wave_sha256=request.binding.host_wave_sha256,
        execution_bundle_manifest_sha256=(
            request.binding.execution_bundle_manifest_sha256
        ),
        assignment_sha256=request.binding.assignment_sha256,
        request_sha256=request.sha256,
        binding_sha256=request.binding.sha256,
        ssh_route_authority_sha256=request.ssh_route_authority_sha256,
        transport_outcome=outcome,
        reason_code=_OUTCOME_REASON[outcome],
        exit_code=process.exit_code,
        stdout_size_bytes=stdout_size,
        stdout_sha256=stdout_sha256,
        stderr_size_bytes=stderr_size,
        stderr_sha256=stderr_sha256,
        remote_reason_sha256=remote_reason_sha256,
        dispatch_schedule_receipt_sha256=receipt_sha256,
        dispatch_schedule_receipt_envelope_sha256=receipt_envelope_sha256,
        completed_assignment_sha256=completed,
        failed_assignment_sha256=failed,
    )


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _positive_limit(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


async def execute_remote_host_wave(
    route: SshHostRoute,
    request: RemoteHostWaveRequest,
    *,
    transport: AsyncSshTransport | None = None,
    timeout_seconds: float = DEFAULT_REMOTE_TIMEOUT_SECONDS,
    stdout_limit_bytes: int = DEFAULT_STDOUT_LIMIT_BYTES,
    stderr_limit_bytes: int = DEFAULT_STDERR_LIMIT_BYTES,
) -> RemoteHostWaveResult:
    """Execute one remote host request without exposing route or output text."""

    if type(route) is not SshHostRoute:
        raise TypeError("remote execution requires an exact SshHostRoute")
    if type(request) is not RemoteHostWaveRequest:
        raise TypeError("remote execution requires an exact RemoteHostWaveRequest")
    if route.host_id != request.binding.host_id:
        raise ValueError("SSH route host differs from remote execution binding")
    timeout = _positive_finite("remote timeout", timeout_seconds)
    stdout_limit = _positive_limit("stdout_limit_bytes", stdout_limit_bytes)
    stderr_limit = _positive_limit("stderr_limit_bytes", stderr_limit_bytes)
    try:
        stdin = request.canonical_stdin()
    except (OSError, RuntimeError, TypeError, ValueError):
        return _empty_result(
            request,
            RemoteTransportOutcome.SSH_FAILED,
            exit_code=None,
        )
    environment = {
        "LC_ALL": "C",
        "SSH_AUTH_SOCK": route.agent_socket_path,
    }
    runner = AsyncioSshTransport() if transport is None else transport
    dispatched = False
    try:
        with _prepared_ssh_route(route) as (argv, route_authority):
            if route_authority != request.ssh_route_authority_sha256:
                return _empty_result(
                    request,
                    RemoteTransportOutcome.SSH_FAILED,
                    exit_code=None,
                )
            dispatched = True
            process = await runner.run(
                argv=argv,
                stdin=stdin,
                environment=environment,
                timeout_seconds=timeout,
                stdout_limit_bytes=stdout_limit,
                stderr_limit_bytes=stderr_limit,
            )
    except (SshTransportTimedOut, TimeoutError):
        outcome = (
            RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN
            if dispatched
            else RemoteTransportOutcome.SSH_FAILED
        )
        return _empty_result(request, outcome, exit_code=None)
    except Exception:  # noqa: BLE001 - no exception proves the remote outcome
        outcome = (
            RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN
            if dispatched
            else RemoteTransportOutcome.SSH_FAILED
        )
        return _empty_result(request, outcome, exit_code=None)
    if type(process) is not SshProcessResult:
        return _empty_result(
            request,
            RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN,
            exit_code=None,
        )
    if len(process.stdout) > stdout_limit or len(process.stderr) > stderr_limit:
        return _empty_result(
            request,
            RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN,
            exit_code=process.exit_code,
            stdout=process.stdout,
            stderr=process.stderr,
        )
    return _decode_remote_response(request, process)


async def execute_fleet_wave(
    routes: Sequence[SshHostRoute],
    requests: Sequence[RemoteHostWaveRequest],
    *,
    transport: AsyncSshTransport | None = None,
    timeout_seconds: float = DEFAULT_REMOTE_TIMEOUT_SECONDS,
    stdout_limit_bytes: int = DEFAULT_STDOUT_LIMIT_BYTES,
    stderr_limit_bytes: int = DEFAULT_STDERR_LIMIT_BYTES,
    max_concurrency: int = DEFAULT_FLEET_MAX_CONCURRENCY,
    prior_fleet_wave_receipt: RemoteFleetWaveReceipt | None = None,
) -> RemoteFleetWaveReceipt:
    """Run host-local children and retain successful nodes across retries.

    A retry consumes the complete prior receipt object, not a caller-copied
    digest.  Only hosts with a negative prior result may be contacted, their
    host-local plan/wave/assignment identity cannot change, and untouched host
    results remain object-for-object identical in the new receipt.
    """

    if not routes or not requests:
        raise ValueError("fleet execution requires at least one host")
    concurrency = _positive_limit("max_concurrency", max_concurrency)
    route_by_host = {route.host_id: route for route in routes}
    request_by_host = {request.binding.host_id: request for request in requests}
    if len(route_by_host) != len(routes) or len(request_by_host) != len(requests):
        raise ValueError("fleet routes and requests must have unique host IDs")
    request_hosts = set(request_by_host)
    if set(route_by_host) != request_hosts:
        raise ValueError("fleet routes do not exactly cover remote requests")
    ordered_requests = tuple(request_by_host[key] for key in sorted(request_by_host))
    authority = ordered_requests[0].binding
    if any(
        request.binding.fleet_inventory_sha256 != authority.fleet_inventory_sha256
        or request.binding.fleet_dispatch_plan_sha256
        != authority.fleet_dispatch_plan_sha256
        or request.binding.fleet_wave_index != authority.fleet_wave_index
        or request.binding.fleet_wave_sha256 != authority.fleet_wave_sha256
        for request in ordered_requests
    ):
        raise ValueError("fleet remote requests mix fleet execution authorities")
    assignment_ids = tuple(
        assignment
        for request in ordered_requests
        for assignment in request.binding.assignment_sha256
    )
    if len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("fleet remote requests migrate or duplicate an assignment")

    retained_results: dict[str, RemoteHostWaveResult] = {}
    prior_sha256: str | None = None
    if prior_fleet_wave_receipt is None:
        if any(
            request.prior_fleet_wave_receipt_sha256 is not None
            or request.binding.resume_receipt_path is not None
            for request in ordered_requests
        ):
            raise ValueError("initial fleet execution cannot claim retry authority")
    else:
        if type(prior_fleet_wave_receipt) is not RemoteFleetWaveReceipt:
            raise TypeError("fleet retry requires an exact prior receipt object")
        replayed = RemoteFleetWaveReceipt.from_dict(prior_fleet_wave_receipt.to_dict())
        if replayed != prior_fleet_wave_receipt:
            raise ValueError("prior fleet-wave receipt changed during replay")
        prior_sha256 = prior_fleet_wave_receipt.sha256
        if (
            prior_fleet_wave_receipt.fleet_inventory_sha256
            != authority.fleet_inventory_sha256
            or prior_fleet_wave_receipt.fleet_dispatch_plan_sha256
            != authority.fleet_dispatch_plan_sha256
            or prior_fleet_wave_receipt.fleet_wave_index != authority.fleet_wave_index
            or prior_fleet_wave_receipt.fleet_wave_sha256 != authority.fleet_wave_sha256
        ):
            raise ValueError("fleet retry belongs to another fleet wave")
        prior_by_host = {
            result.host_id: result for result in prior_fleet_wave_receipt.host_results
        }
        if not request_hosts <= set(prior_by_host):
            raise ValueError("fleet retry introduces a host outside the prior receipt")
        for request in ordered_requests:
            binding = request.binding
            previous = prior_by_host[binding.host_id]
            if previous.succeeded:
                raise ValueError("fleet retry cannot re-execute a successful host")
            if previous.outcome_unknown:
                raise ValueError(
                    "UNKNOWN remote outcome requires exact evidence reconciliation"
                )
            if not previous.retryable:
                raise ValueError("fleet host result is not directly retryable")
            if request.prior_fleet_wave_receipt_sha256 != prior_sha256:
                raise ValueError("fleet retry authority differs from its host request")
            if (
                binding.host_inventory_sha256 != previous.host_inventory_sha256
                or binding.host_dispatch_plan_sha256
                != previous.host_dispatch_plan_sha256
                or binding.host_wave_index != previous.host_wave_index
                or binding.host_wave_sha256 != previous.host_wave_sha256
                or binding.execution_bundle_manifest_sha256
                != previous.execution_bundle_manifest_sha256
                or binding.assignment_sha256 != previous.assignment_sha256
                or request.ssh_route_authority_sha256
                != previous.ssh_route_authority_sha256
            ):
                raise ValueError("fleet retry changes a host-local execution identity")
            has_remote_receipt = previous.dispatch_schedule_receipt_sha256 is not None
            if has_remote_receipt != (binding.resume_receipt_path is not None):
                raise ValueError(
                    "fleet retry resume path differs from prior remote receipt authority"
                )
            if binding.resume_receipt_sha256 != (
                previous.dispatch_schedule_receipt_sha256
            ):
                raise ValueError(
                    "fleet retry resume content differs from prior remote receipt"
                )
            if binding.resume_receipt_envelope_sha256 != (
                previous.dispatch_schedule_receipt_envelope_sha256
            ):
                raise ValueError(
                    "fleet retry resume envelope differs from prior remote receipt"
                )
            if request.sha256 == previous.request_sha256:
                raise ValueError("fleet retry must create a new receipt-bound attempt")
        retained_results = prior_by_host
    semaphore = asyncio.Semaphore(concurrency)

    async def run_bounded(
        request: RemoteHostWaveRequest,
    ) -> RemoteHostWaveResult:
        async with semaphore:
            return await execute_remote_host_wave(
                route_by_host[request.binding.host_id],
                request,
                transport=transport,
                timeout_seconds=timeout_seconds,
                stdout_limit_bytes=stdout_limit_bytes,
                stderr_limit_bytes=stderr_limit_bytes,
            )

    results = await asyncio.gather(
        *(run_bounded(request) for request in ordered_requests)
    )
    result_by_host = dict(retained_results)
    result_by_host.update((result.host_id, result) for result in results)
    return RemoteFleetWaveReceipt(
        schema_version=1,
        fleet_inventory_sha256=authority.fleet_inventory_sha256,
        fleet_dispatch_plan_sha256=authority.fleet_dispatch_plan_sha256,
        fleet_wave_index=authority.fleet_wave_index,
        fleet_wave_sha256=authority.fleet_wave_sha256,
        host_results=tuple(result_by_host[key] for key in sorted(result_by_host)),
        prior_fleet_wave_receipt_sha256=prior_sha256,
    )


__all__ = [
    "AsyncSshTransport",
    "AsyncioSshTransport",
    "CrossHostCollectivesUnvalidated",
    "FleetWaveOutcome",
    "HostAssignmentBinding",
    "RemoteFleetWaveReceipt",
    "RemoteHostExecutionBinding",
    "RemoteHostWaveRequest",
    "RemoteHostWaveResponse",
    "RemoteHostWaveResult",
    "RemoteTransportOutcome",
    "RemoteWorkerStatus",
    "SshHostRoute",
    "SshOutputLimitExceeded",
    "SshProcessResult",
    "SshTransportTimedOut",
    "build_ssh_argv",
    "canonical_json_bytes",
    "canonical_sha256",
    "decode_remote_host_wave_request",
    "execute_fleet_wave",
    "execute_host_local_wave_request",
    "execute_remote_host_wave",
    "reconcile_fleet_wave_receipt",
    "reconcile_remote_host_wave",
]
