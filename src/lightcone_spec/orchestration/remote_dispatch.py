"""Fail-closed SSH transport for one host-local dispatch wave.

The controller sends only a canonical control request.  Execution bundles are
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
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
MAX_REQUEST_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


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
    """Controller-local SSH authority which is intentionally not serializable."""

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


def build_ssh_argv(route: SshHostRoute) -> tuple[str, ...]:
    """Build the complete fixed-policy OpenSSH argv for one route."""

    if type(route) is not SshHostRoute:
        raise TypeError("SSH argv requires an exact SshHostRoute")
    # Recheck local trust paths immediately before use to narrow replacement
    # races.  Neither path nor the destination is copied into any artifact.
    known_hosts = _validate_known_hosts(route.known_hosts_path)
    _validate_agent_socket(route.agent_socket_path)
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
        f"UserKnownHostsFile={known_hosts}",
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
    dispatch_plan_sha256: str
    wave_index: int
    wave_sha256: str
    execution_bundle_manifest_path: str
    execution_bundle_manifest_sha256: str
    receipt_output_path: str
    resume_receipt_path: str | None
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
            "dispatch_plan_sha256",
            "wave_sha256",
            "execution_bundle_manifest_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if type(self.wave_index) is not int or self.wave_index < 0:
            raise ValueError("wave_index must be a non-negative integer")
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
            "dispatch_plan_sha256": self.dispatch_plan_sha256,
            "wave_index": self.wave_index,
            "wave_sha256": self.wave_sha256,
            "execution_bundle_manifest_path": self.execution_bundle_manifest_path,
            "execution_bundle_manifest_sha256": (self.execution_bundle_manifest_sha256),
            "receipt_output_path": self.receipt_output_path,
            "resume_receipt_path": self.resume_receipt_path,
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
                    "dispatch_plan_sha256",
                    "wave_index",
                    "wave_sha256",
                    "execution_bundle_manifest_path",
                    "execution_bundle_manifest_sha256",
                    "receipt_output_path",
                    "resume_receipt_path",
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
            dispatch_plan_sha256=_require_sha256(
                "dispatch_plan_sha256", row["dispatch_plan_sha256"]
            ),
            wave_index=_strict_int("wave_index", row["wave_index"]),
            wave_sha256=_require_sha256("wave_sha256", row["wave_sha256"]),
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
            assignments=assignments,
        )


@dataclass(frozen=True)
class RemoteHostWaveRequest:
    """Canonical stdin request for one host; contains no routing material."""

    schema_version: int
    challenge_nonce_sha256: str
    binding: RemoteHostExecutionBinding
    prior_fleet_wave_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only remote host-wave request schema version 1 is supported"
            )
        _require_sha256("challenge_nonce_sha256", self.challenge_nonce_sha256)
        if type(self.binding) is not RemoteHostExecutionBinding:
            raise TypeError(
                "remote request requires an exact RemoteHostExecutionBinding"
            )
        if self.prior_fleet_wave_receipt_sha256 is not None:
            _require_sha256(
                "prior_fleet_wave_receipt_sha256",
                self.prior_fleet_wave_receipt_sha256,
            )
        if (self.prior_fleet_wave_receipt_sha256 is None) != (
            self.binding.resume_receipt_path is None
        ):
            raise ValueError(
                "host-local resume requires the prior fleet-wave receipt identity"
            )

    @cached_property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "lightcone_remote_host_wave_request",
            "challenge_nonce_sha256": self.challenge_nonce_sha256,
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
            binding=binding,
            prior_fleet_wave_receipt_sha256=_optional_sha256(
                "prior_fleet_wave_receipt_sha256",
                row["prior_fleet_wave_receipt_sha256"],
            ),
        )


def decode_remote_host_wave_request(stdin: bytes) -> RemoteHostWaveRequest:
    """Decode one bounded, newline-terminated canonical stdin request."""

    if type(stdin) is not bytes:
        raise TypeError("remote host-wave stdin must be bytes")
    if not stdin or len(stdin) > MAX_REQUEST_BYTES:
        raise ValueError("remote host-wave stdin is empty or exceeds its byte limit")
    if not stdin.endswith(b"\n") or stdin.endswith(b"\n\n"):
        raise ValueError("remote host-wave stdin must contain one canonical JSON line")
    body = stdin[:-1]
    value = _strict_json_object(body)
    if canonical_json_bytes(value) != body:
        raise ValueError("remote host-wave stdin is not canonical JSON")
    request = RemoteHostWaveRequest.from_dict(value)
    if request.canonical_stdin() != stdin:
        raise ValueError("remote host-wave stdin changed after strict decoding")
    return request


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
                or self.failed_assignment_sha256
            ):
                raise ValueError("successful remote response is incomplete")
        elif self.status is RemoteWorkerStatus.BLOCKED:
            if (
                self.reason_code is None
                or self.dispatch_schedule_receipt_sha256 is not None
                or self.completed_assignment_sha256
                or self.failed_assignment_sha256
            ):
                raise ValueError("blocked remote response cannot claim execution")
        elif self.reason_code is None:
            raise ValueError("failed remote response requires a stable reason code")

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


def _read_stable_manifest_leaf(path: Path) -> tuple[bytes, tuple[int, ...]]:
    """Read one non-empty single-link leaf with path/fd identity agreement."""

    initial = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(initial.st_mode)
        or initial.st_nlink != 1
        or initial.st_size < 1
        or initial.st_size > MAX_MANIFEST_BYTES
    ):
        raise ValueError("host-local manifest is not one bounded single-link file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            _stat_identity(initial) != _stat_identity(opened)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size < 1
            or opened.st_size > MAX_MANIFEST_BYTES
        ):
            raise ValueError("host-local manifest changed before it was opened")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise ValueError("host-local manifest shrank while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("host-local manifest grew while it was read")
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
        raise ValueError("host-local manifest changed while it was read")
    return body, identity


def _verify_host_local_manifest(binding: RemoteHostExecutionBinding) -> None:
    """Double-reopen and digest one stable host-local manifest leaf."""

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


def _worker_response_from_receipt(
    request: RemoteHostWaveRequest,
    receipt: object,
) -> RemoteHostWaveResponse:
    """Validate the first-party schedule receipt against transport authority."""

    from lightcone_spec.experiments.gpu_pool import (
        AssignmentExecutionStatus,
        DispatchScheduleReceipt,
    )

    if type(receipt) is not DispatchScheduleReceipt:
        raise TypeError("host executor returned a non-exact schedule receipt")
    binding = request.binding
    if (
        receipt.plan_sha256 != binding.dispatch_plan_sha256
        or receipt.inventory_sha256 != binding.host_inventory_sha256
        or len(receipt.wave_receipts) <= binding.wave_index
    ):
        raise ValueError("host schedule receipt differs from request authority")
    wave = receipt.wave_receipts[binding.wave_index]
    if wave.wave_index != binding.wave_index or wave.wave_sha256 != binding.wave_sha256:
        raise ValueError("host schedule receipt differs from requested wave")
    actual = tuple(sorted(row.assignment_sha256 for row in wave.assignment_receipts))
    if actual != binding.assignment_sha256:
        raise ValueError("host schedule receipt assignment coverage differs")
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
        completed_assignment_sha256=completed,
        failed_assignment_sha256=failed,
    )


async def execute_host_local_wave_request(
    stdin: bytes,
    *,
    execute_wave: Any = None,
) -> tuple[int, bytes]:
    """Decode controller stdin and invoke the existing host-local wave executor.

    The CLI adapter may write the returned bytes directly to stdout.  No raw
    exception text, host path, SSH route, or execution bundle enters it.
    """

    request = decode_remote_host_wave_request(stdin)
    try:
        _verify_host_local_manifest(request.binding)
    except (OSError, TypeError, ValueError):
        response = _blocked_worker_response(request, "remote_host_manifest_invalid")
        return 42, response.canonical_stdout()
    if execute_wave is None:
        from lightcone_spec.orchestration.execution_bundle import (
            execute_dispatch_wave_bundles,
        )

        execute_wave = execute_dispatch_wave_bundles
    try:
        receipt = await execute_wave(
            request.binding.execution_bundle_manifest_path,
            wave_index=request.binding.wave_index,
            receipt_output=request.binding.receipt_output_path,
            resume_receipt_path=request.binding.resume_receipt_path,
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
            response = RemoteHostWaveResponse(
                schema_version=1,
                host_id=request.binding.host_id,
                request_sha256=request.sha256,
                binding_sha256=request.binding.sha256,
                status=RemoteWorkerStatus.FAILED,
                reason_code="remote_host_wave_execution_failed",
                dispatch_schedule_receipt_sha256=None,
                completed_assignment_sha256=(),
                failed_assignment_sha256=request.binding.assignment_sha256,
            )
        return 42, response.canonical_stdout()
    try:
        response = _worker_response_from_receipt(request, receipt)
    except (TypeError, ValueError):
        response = RemoteHostWaveResponse(
            schema_version=1,
            host_id=request.binding.host_id,
            request_sha256=request.sha256,
            binding_sha256=request.binding.sha256,
            status=RemoteWorkerStatus.FAILED,
            reason_code="remote_host_receipt_invalid",
            dispatch_schedule_receipt_sha256=None,
            completed_assignment_sha256=(),
            failed_assignment_sha256=request.binding.assignment_sha256,
        )
    return (
        0 if response.status is RemoteWorkerStatus.SUCCEEDED else 42,
        response.canonical_stdout(),
    )


class RemoteTransportOutcome(str, Enum):
    REMOTE_SUCCEEDED = "REMOTE_SUCCEEDED"
    REMOTE_BLOCKED = "REMOTE_BLOCKED"
    REMOTE_FAILED = "REMOTE_FAILED"
    SSH_FAILED = "SSH_FAILED"
    TIMED_OUT = "TIMED_OUT"
    OUTPUT_LIMIT_EXCEEDED = "OUTPUT_LIMIT_EXCEEDED"
    INVALID_RESPONSE = "INVALID_RESPONSE"


_OUTCOME_REASON = {
    RemoteTransportOutcome.REMOTE_SUCCEEDED: "remote_wave_succeeded",
    RemoteTransportOutcome.REMOTE_BLOCKED: "remote_wave_blocked",
    RemoteTransportOutcome.REMOTE_FAILED: "remote_wave_failed",
    RemoteTransportOutcome.SSH_FAILED: "ssh_transport_failed",
    RemoteTransportOutcome.TIMED_OUT: "ssh_transport_timed_out",
    RemoteTransportOutcome.OUTPUT_LIMIT_EXCEEDED: "ssh_output_limit_exceeded",
    RemoteTransportOutcome.INVALID_RESPONSE: "remote_response_invalid",
}


@dataclass(frozen=True)
class RemoteHostWaveResult:
    """Sensitive-data-free result of one host transport attempt."""

    schema_version: int
    host_id: str
    fleet_inventory_sha256: str
    dispatch_plan_sha256: str
    wave_index: int
    wave_sha256: str
    request_sha256: str
    binding_sha256: str
    transport_outcome: RemoteTransportOutcome
    reason_code: str
    exit_code: int | None
    stdout_size_bytes: int
    stdout_sha256: str
    stderr_size_bytes: int
    stderr_sha256: str
    remote_reason_sha256: str | None
    dispatch_schedule_receipt_sha256: str | None
    completed_assignment_sha256: tuple[str, ...]
    failed_assignment_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only remote host-wave result schema version 1 is supported"
            )
        _require_id("result host_id", self.host_id)
        for name in (
            "fleet_inventory_sha256",
            "dispatch_plan_sha256",
            "wave_sha256",
            "request_sha256",
            "binding_sha256",
            "stdout_sha256",
            "stderr_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if type(self.wave_index) is not int or self.wave_index < 0:
            raise ValueError("wave_index must be a non-negative integer")
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
        for name in (
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
        if self.transport_outcome is RemoteTransportOutcome.REMOTE_SUCCEEDED:
            if (
                self.exit_code != 0
                or self.remote_reason_sha256 is not None
                or self.dispatch_schedule_receipt_sha256 is None
                or self.failed_assignment_sha256
            ):
                raise ValueError("successful remote result is incomplete")
        elif self.transport_outcome in {
            RemoteTransportOutcome.REMOTE_BLOCKED,
            RemoteTransportOutcome.REMOTE_FAILED,
        }:
            if self.exit_code != 42 or self.remote_reason_sha256 is None:
                raise ValueError("negative remote result lacks valid remote authority")
        elif (
            self.remote_reason_sha256 is not None
            or self.dispatch_schedule_receipt_sha256 is not None
            or self.completed_assignment_sha256
            or self.failed_assignment_sha256
        ):
            raise ValueError("transport failure cannot claim remote completion")

    @cached_property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def succeeded(self) -> bool:
        return self.transport_outcome is RemoteTransportOutcome.REMOTE_SUCCEEDED

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "lightcone_remote_host_wave_result",
            "host_id": self.host_id,
            "fleet_inventory_sha256": self.fleet_inventory_sha256,
            "dispatch_plan_sha256": self.dispatch_plan_sha256,
            "wave_index": self.wave_index,
            "wave_sha256": self.wave_sha256,
            "request_sha256": self.request_sha256,
            "binding_sha256": self.binding_sha256,
            "transport_outcome": self.transport_outcome.value,
            "reason_code": self.reason_code,
            "exit_code": self.exit_code,
            "stdout_size_bytes": self.stdout_size_bytes,
            "stdout_sha256": self.stdout_sha256,
            "stderr_size_bytes": self.stderr_size_bytes,
            "stderr_sha256": self.stderr_sha256,
            "remote_reason_sha256": self.remote_reason_sha256,
            "dispatch_schedule_receipt_sha256": (self.dispatch_schedule_receipt_sha256),
            "completed_assignment_sha256": list(self.completed_assignment_sha256),
            "failed_assignment_sha256": list(self.failed_assignment_sha256),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = frozenset(
            {
                "schema_version",
                "kind",
                "host_id",
                "fleet_inventory_sha256",
                "dispatch_plan_sha256",
                "wave_index",
                "wave_sha256",
                "request_sha256",
                "binding_sha256",
                "transport_outcome",
                "reason_code",
                "exit_code",
                "stdout_size_bytes",
                "stdout_sha256",
                "stderr_size_bytes",
                "stderr_sha256",
                "remote_reason_sha256",
                "dispatch_schedule_receipt_sha256",
                "completed_assignment_sha256",
                "failed_assignment_sha256",
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
            dispatch_plan_sha256=_require_sha256(
                "dispatch_plan_sha256", row["dispatch_plan_sha256"]
            ),
            wave_index=_strict_int("wave_index", row["wave_index"]),
            wave_sha256=_require_sha256("wave_sha256", row["wave_sha256"]),
            request_sha256=_require_sha256("request_sha256", row["request_sha256"]),
            binding_sha256=_require_sha256("binding_sha256", row["binding_sha256"]),
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


class FleetWaveOutcome(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


@dataclass(frozen=True)
class RemoteFleetWaveReceipt:
    """Aggregate host outcomes without discarding a successful node receipt."""

    schema_version: int
    fleet_inventory_sha256: str
    dispatch_plan_sha256: str
    wave_index: int
    wave_sha256: str
    host_results: tuple[RemoteHostWaveResult, ...]
    prior_fleet_wave_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only fleet-wave receipt schema version 1 is supported")
        _require_sha256("fleet_inventory_sha256", self.fleet_inventory_sha256)
        _require_sha256("dispatch_plan_sha256", self.dispatch_plan_sha256)
        _require_sha256("wave_sha256", self.wave_sha256)
        if type(self.wave_index) is not int or self.wave_index < 0:
            raise ValueError("wave_index must be a non-negative integer")
        if not self.host_results or any(
            type(item) is not RemoteHostWaveResult for item in self.host_results
        ):
            raise ValueError("fleet-wave receipt requires exact host results")
        host_ids = tuple(result.host_id for result in self.host_results)
        if host_ids != tuple(sorted(set(host_ids))):
            raise ValueError("fleet host results must be sorted and unique")
        if any(
            result.fleet_inventory_sha256 != self.fleet_inventory_sha256
            or result.dispatch_plan_sha256 != self.dispatch_plan_sha256
            or result.wave_index != self.wave_index
            or result.wave_sha256 != self.wave_sha256
            for result in self.host_results
        ):
            raise ValueError("fleet-wave receipt mixes execution authorities")
        if self.prior_fleet_wave_receipt_sha256 is not None:
            _require_sha256(
                "prior_fleet_wave_receipt_sha256",
                self.prior_fleet_wave_receipt_sha256,
            )

    @property
    def outcome(self) -> FleetWaveOutcome:
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
            "dispatch_plan_sha256": self.dispatch_plan_sha256,
            "wave_index": self.wave_index,
            "wave_sha256": self.wave_sha256,
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
                    "dispatch_plan_sha256",
                    "wave_index",
                    "wave_sha256",
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
            dispatch_plan_sha256=_require_sha256(
                "dispatch_plan_sha256", row["dispatch_plan_sha256"]
            ),
            wave_index=_strict_int("wave_index", row["wave_index"]),
            wave_sha256=_require_sha256("wave_sha256", row["wave_sha256"]),
            host_results=results,
            prior_fleet_wave_receipt_sha256=_optional_sha256(
                "prior_fleet_wave_receipt_sha256",
                row["prior_fleet_wave_receipt_sha256"],
            ),
        )
        if row["outcome"] != receipt.outcome.value:
            raise ValueError("fleet-wave outcome differs from host results")
        return receipt


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
        dispatch_plan_sha256=binding.dispatch_plan_sha256,
        wave_index=binding.wave_index,
        wave_sha256=binding.wave_sha256,
        request_sha256=request.sha256,
        binding_sha256=binding.sha256,
        transport_outcome=outcome,
        reason_code=_OUTCOME_REASON[outcome],
        exit_code=exit_code,
        stdout_size_bytes=stdout_size,
        stdout_sha256=stdout_sha256,
        stderr_size_bytes=stderr_size,
        stderr_sha256=stderr_sha256,
        remote_reason_sha256=None,
        dispatch_schedule_receipt_sha256=None,
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
            RemoteTransportOutcome.SSH_FAILED,
            exit_code=process.exit_code,
            stdout=process.stdout,
            stderr=process.stderr,
        )
    if not process.stdout.endswith(b"\n") or process.stdout.endswith(b"\n\n"):
        return _empty_result(
            request,
            RemoteTransportOutcome.INVALID_RESPONSE,
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
            RemoteTransportOutcome.INVALID_RESPONSE,
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
        dispatch_plan_sha256=request.binding.dispatch_plan_sha256,
        wave_index=request.binding.wave_index,
        wave_sha256=request.binding.wave_sha256,
        request_sha256=request.sha256,
        binding_sha256=request.binding.sha256,
        transport_outcome=outcome,
        reason_code=_OUTCOME_REASON[outcome],
        exit_code=process.exit_code,
        stdout_size_bytes=stdout_size,
        stdout_sha256=stdout_sha256,
        stderr_size_bytes=stderr_size,
        stderr_sha256=stderr_sha256,
        remote_reason_sha256=remote_reason_sha256,
        dispatch_schedule_receipt_sha256=receipt_sha256,
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
        argv = build_ssh_argv(route)
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
    try:
        process = await runner.run(
            argv=argv,
            stdin=stdin,
            environment=environment,
            timeout_seconds=timeout,
            stdout_limit_bytes=stdout_limit,
            stderr_limit_bytes=stderr_limit,
        )
    except (SshTransportTimedOut, TimeoutError):
        return _empty_result(request, RemoteTransportOutcome.TIMED_OUT, exit_code=None)
    except SshOutputLimitExceeded:
        return _empty_result(
            request,
            RemoteTransportOutcome.OUTPUT_LIMIT_EXCEEDED,
            exit_code=None,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return _empty_result(
            request,
            RemoteTransportOutcome.SSH_FAILED,
            exit_code=None,
        )
    if type(process) is not SshProcessResult:
        return _empty_result(
            request,
            RemoteTransportOutcome.SSH_FAILED,
            exit_code=None,
        )
    if len(process.stdout) > stdout_limit or len(process.stderr) > stderr_limit:
        return _empty_result(
            request,
            RemoteTransportOutcome.OUTPUT_LIMIT_EXCEEDED,
            exit_code=process.exit_code,
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
    prior_fleet_wave_receipt_sha256: str | None = None,
) -> RemoteFleetWaveReceipt:
    """Run independent host subsets concurrently and retain every outcome."""

    if not routes or not requests:
        raise ValueError("fleet execution requires at least one host")
    route_by_host = {route.host_id: route for route in routes}
    request_by_host = {request.binding.host_id: request for request in requests}
    if len(route_by_host) != len(routes) or len(request_by_host) != len(requests):
        raise ValueError("fleet routes and requests must have unique host IDs")
    if set(route_by_host) != set(request_by_host):
        raise ValueError("fleet routes do not exactly cover remote requests")
    ordered_requests = tuple(request_by_host[key] for key in sorted(request_by_host))
    authority = ordered_requests[0].binding
    if any(
        request.binding.fleet_inventory_sha256 != authority.fleet_inventory_sha256
        or request.binding.dispatch_plan_sha256 != authority.dispatch_plan_sha256
        or request.binding.wave_index != authority.wave_index
        or request.binding.wave_sha256 != authority.wave_sha256
        for request in ordered_requests
    ):
        raise ValueError("fleet remote requests mix execution authorities")
    if prior_fleet_wave_receipt_sha256 is not None:
        _require_sha256(
            "prior_fleet_wave_receipt_sha256", prior_fleet_wave_receipt_sha256
        )
    if any(
        request.prior_fleet_wave_receipt_sha256 != prior_fleet_wave_receipt_sha256
        for request in ordered_requests
    ):
        raise ValueError("fleet retry authority differs from its host requests")
    results = await asyncio.gather(
        *(
            execute_remote_host_wave(
                route_by_host[request.binding.host_id],
                request,
                transport=transport,
                timeout_seconds=timeout_seconds,
                stdout_limit_bytes=stdout_limit_bytes,
                stderr_limit_bytes=stderr_limit_bytes,
            )
            for request in ordered_requests
        )
    )
    return RemoteFleetWaveReceipt(
        schema_version=1,
        fleet_inventory_sha256=authority.fleet_inventory_sha256,
        dispatch_plan_sha256=authority.dispatch_plan_sha256,
        wave_index=authority.wave_index,
        wave_sha256=authority.wave_sha256,
        host_results=tuple(results),
        prior_fleet_wave_receipt_sha256=prior_fleet_wave_receipt_sha256,
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
]
