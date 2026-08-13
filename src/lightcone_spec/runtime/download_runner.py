"""CPU-test DOWNLOAD subprocess lifecycle without download authority.

The registered DOWNLOAD plan is already a path-bound raw authority in
``experiments.nonserving_authority``.  This module can exercise a caller-owned
child process against that exact plan.  The runner neither derives nor adds a
cache path, credential, provider configuration, or payload bytes; the caller
still owns the arbitrary argv.  The child is not an activity authority: its
zero network/write counters are explicitly untrusted self-reports.  The
resulting receipt proves only CPU protocol coverage.  It does not prove absence
of network or payload activity, and never proves that a model was downloaded.

The formal entry point remains fail closed.  Both the source-owned command map
and the exact-plan allowlist are empty in this release and are checked before
the caller-named plan path is opened or any process/output is created.
"""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import signal
import stat
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NoReturn, Self

from lightcone_spec.experiments.nonserving_authority import (
    RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON,
    BoundDownloadJson,
    DownloadExecutionBlocked,
    DownloadPlan,
    DownloadPlanAuthority,
    bind_download_plan_authority,
    require_release_download_execution,
    revalidate_download_plan_authority,
)
from lightcone_spec.experiments.registry import content_sha256

_MAX_MESSAGE_BYTES = 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_MAX_JSON_BYTES = 16 * 1024 * 1024
CPU_DIAGNOSTIC_SELF_REPORTED = "CPU_DIAGNOSTIC_SELF_REPORTED"
UNTRUSTED_CHILD_ACTIVITY_COUNTERS = "UNTRUSTED_CHILD_SELF_REPORT"

DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "cpu_test_download_subprocess_self_reported_lifecycle",
        "transport": "canonical_json_lines_over_private_stdin_stdout",
        "ordered_messages": (
            "ready",
            "start_message_omits_cache_credentials_and_activity_instructions",
            "started",
            "each_locked_model_revision_exactly_once",
            "each_expected_output_exactly_once",
            "drain_with_untrusted_child_self_reported_activity_counters",
            "parent_observed_zero_exit",
        ),
        "formal_authority": "NONE_CPU_TEST_ONLY",
        "formal_promotion": False,
        "diagnostic_only": True,
        "network_and_payload_activity_observation": "NOT_IMPLEMENTED",
        "activity_counter_authority": UNTRUSTED_CHILD_ACTIVITY_COUNTERS,
    }
)
DOWNLOAD_DIAGNOSTIC_RESULT_POINTER_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "download_cpu_test_self_reported_result_pointer",
        "bindings": (
            "raw_plan_and_sidecar",
            "raw_subprocess_terminal_and_sidecar",
            "exact_executable_and_argv",
            "model_revision_and_output_manifests",
        ),
        "publication": "exclusive_body_then_sidecar_commit_marker_last",
        "resume": "reopen_every_binding_without_spawning",
        "formal_completion": False,
        "activity_counter_authority": UNTRUSTED_CHILD_ACTIVITY_COUNTERS,
    }
)
DOWNLOAD_DIAGNOSTIC_ATTEMPT_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "download_diagnostic_failed_attempt",
        "retention": "failure_is_retained_but_never_referenced_by_result_pointer",
        "formal_completion": False,
    }
)

RELEASE_DOWNLOAD_PLAN_ALLOWLIST_EMPTY = "release_download_plan_allowlist_empty"
RELEASE_DOWNLOAD_PLAN_UNTRUSTED = "release_download_plan_not_allowlisted"
RELEASE_TRUSTED_DOWNLOAD_PLAN_SHA256S: tuple[str, ...] = ()


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{label} must be canonical single-line text")
    return value


def _require_argument(label: str, value: object) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{label} must be non-empty text without NUL")
    return value


def _absolute_path(label: str, value: object) -> Path:
    text = _require_text(label, value)
    path = Path(text)
    if (
        not path.is_absolute()
        or path == Path(path.anchor)
        or path != path.resolve(strict=False)
    ):
        raise ValueError(f"{label} must be absolute, normalized, and non-root")
    return path


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_object(body: bytes, *, label: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains non-finite constant {value}")

    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    return value


def _exact_json(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's ``False == 0`` coercion."""

    return _canonical_bytes(actual) == _canonical_bytes(expected)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_bytes(
    path: Path,
    *,
    label: str,
    require_single_link: bool = False,
    maximum_bytes: int | None = None,
) -> bytes:
    normalized = _absolute_path(label, str(path))
    if normalized.is_symlink():
        raise ValueError(f"{label} must be a non-symlink regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(normalized, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or (require_single_link and before.st_nlink != 1)
        ):
            raise ValueError(f"{label} must be a regular file")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds the raw size limit")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        try:
            current = normalized.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{label} disappeared during coordinated read") from error
        if (
            not stat.S_ISREG(current.st_mode)
            or (require_single_link and before.st_nlink != 1)
            or (require_single_link and after.st_nlink != 1)
            or (require_single_link and current.st_nlink != 1)
            or _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(current)
            or size != after.st_size
        ):
            raise ValueError(f"{label} changed during coordinated read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _raw_sha256(path: Path, *, label: str) -> tuple[str, int]:
    body = _stable_file_bytes(path, label=label)
    return hashlib.sha256(body).hexdigest(), len(body)


def _regular_directory(path: Path, *, label: str) -> None:
    normalized = _absolute_path(label, str(path))
    if normalized.is_symlink():
        raise ValueError(f"{label} must be a non-symlink directory")
    try:
        status = normalized.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"{label} must be a directory")


def _require_single_link_json_pair(path: Path, *, label: str) -> None:
    """Reject aliases for evidence bodies and their commit-marker sidecars."""

    _stable_file_bytes(
        path,
        label=label,
        require_single_link=True,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    _stable_file_bytes(
        Path(f"{path}.sha256"),
        label=f"{label} sidecar",
        require_single_link=True,
        maximum_bytes=_MAX_JSON_BYTES,
    )


def _publish_file_exclusive(path: Path, body: bytes, *, label: str) -> None:
    destination = _absolute_path(label, str(path))
    parent = destination.parent
    _regular_directory(parent, label=f"{label} parent")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_descriptor = os.open(parent, directory_flags)
    except OSError as error:
        raise ValueError(f"{label} parent cannot be opened safely") from error
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            destination.name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        view = memoryview(body)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count < 1:
                raise OSError("short write")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        current = destination.stat(follow_symlinks=False)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or opened.st_nlink != 1
            or _file_identity(current) != _file_identity(opened)
            or current.st_size != len(body)
        ):
            raise ValueError(f"{label} changed during exclusive publication")
        os.fsync(parent_descriptor)
    except FileExistsError as error:
        raise ValueError(f"{label} already exists") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _publish_semantic_json(
    path: Path,
    value: object,
    *,
    label: str,
) -> BoundDownloadJson:
    """Publish an immutable body, then its sidecar commit marker last."""

    body = _canonical_bytes(value)
    if len(body) > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds the raw JSON size limit")
    _publish_file_exclusive(path, body, label=label)
    semantic_sha256 = content_sha256(value)
    _publish_file_exclusive(
        Path(f"{path}.sha256"),
        f"{semantic_sha256}\n".encode("ascii"),
        label=f"{label} sidecar",
    )
    _require_single_link_json_pair(path, label=label)
    binding, raw = BoundDownloadJson.bind(
        path,
        expected_path=str(path),
        label=label,
    )
    if raw != value or binding.semantic_sha256 != semantic_sha256:
        raise ValueError(f"{label} differs immediately after publication")
    _require_single_link_json_pair(path, label=label)
    return binding


def _bound_json_from_dict(raw: object, *, label: str) -> BoundDownloadJson:
    expected = {
        "schema_version",
        "path",
        "size",
        "raw_sha256",
        "semantic_sha256",
        "sidecar_path",
        "sidecar_size",
        "sidecar_raw_sha256",
    }
    if type(raw) is not dict or set(raw) != expected:
        raise ValueError(f"{label} binding fields differ from schema")
    return BoundDownloadJson(**raw)


@dataclass(frozen=True)
class ReleaseDownloadSubprocess:
    """One source-owned exact command eligible for future formal execution."""

    argv: tuple[str, ...]
    executable_raw_sha256: str
    protocol_sha256: str

    def validate(self, *, reopen_executable: bool) -> None:
        if type(self.argv) is not tuple or not self.argv:
            raise TypeError("release download argv must be a non-empty tuple")
        for argument in self.argv:
            _require_argument("release download argument", argument)
        executable = _absolute_path("release download executable", self.argv[0])
        _require_sha256("release download executable", self.executable_raw_sha256)
        if self.protocol_sha256 != DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256:
            raise ValueError("release download subprocess uses another protocol")
        if reopen_executable:
            digest, size = _raw_sha256(
                executable,
                label="release download executable",
            )
            if size < 1 or digest != self.executable_raw_sha256:
                raise ValueError("release download executable differs from source pin")

    @property
    def sha256(self) -> str:
        self.validate(reopen_executable=False)
        return content_sha256(asdict(self))


# A future reviewed release must populate both source-owned policies.  Caller
# data cannot extend them.
RELEASE_DOWNLOAD_SUBPROCESSES: tuple[ReleaseDownloadSubprocess, ...] = ()


@dataclass(frozen=True)
class DownloadSubprocessEvent:
    sequence: int
    direction: str
    canonical_json: str
    raw_sha256: str

    def validate(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("download subprocess event sequence is invalid")
        if self.direction not in {"parent_to_child", "child_to_parent"}:
            raise ValueError("download subprocess event direction is invalid")
        if type(self.canonical_json) is not str or "\n" in self.canonical_json:
            raise ValueError("download subprocess event must be one JSON line")
        encoded = f"{self.canonical_json}\n".encode()
        row = _strict_object(encoded, label="download subprocess event")
        if encoded != _canonical_bytes(row):
            raise ValueError("download subprocess event is not canonical JSON")
        if hashlib.sha256(encoded).hexdigest() != self.raw_sha256:
            raise ValueError("download subprocess event raw digest differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        if type(raw) is not dict or set(raw) != {
            "sequence",
            "direction",
            "canonical_json",
            "raw_sha256",
        }:
            raise ValueError("download subprocess event fields differ from schema")
        value = cls(**raw)
        value.validate()
        return value


def _require_exact_fields(
    row: dict[str, object],
    fields: set[str],
    *,
    label: str,
) -> None:
    if set(row) != fields:
        raise ValueError(f"{label} fields differ from protocol")


@dataclass(frozen=True)
class DownloadSubprocessLifecycleReceipt:
    schema_version: int
    kind: str
    protocol_sha256: str
    download_plan_sha256: str
    plan_authority_sha256: str
    model_revision_manifest_sha256: str
    output_manifest_sha256: str
    executable_path: str
    executable_raw_sha256: str
    executable_size: int
    argv_sha256: str
    source_authority_sha256: str | None
    process_id: int
    process_started_ns: int
    process_exited_ns: int
    exit_code: int
    events: tuple[DownloadSubprocessEvent, ...]
    diagnostic_status: str
    formal_execution_authorized: bool

    def validate(self, *, reopen_executable: bool) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "download_subprocess_lifecycle_raw_receipt"
        ):
            raise ValueError("download subprocess receipt schema is unsupported")
        if self.protocol_sha256 != DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256:
            raise ValueError("download subprocess receipt uses another protocol")
        for label, value in (
            ("download plan", self.download_plan_sha256),
            ("plan authority", self.plan_authority_sha256),
            ("model revision manifest", self.model_revision_manifest_sha256),
            ("output manifest", self.output_manifest_sha256),
            ("executable", self.executable_raw_sha256),
            ("argv", self.argv_sha256),
        ):
            _require_sha256(label, value)
        executable = _absolute_path(
            "download subprocess executable",
            self.executable_path,
        )
        if type(self.executable_size) is not int or self.executable_size < 1:
            raise ValueError("download subprocess executable size is invalid")
        for label, value in (
            ("process ID", self.process_id),
            ("process start", self.process_started_ns),
            ("process exit", self.process_exited_ns),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"download subprocess {label} is invalid")
        if self.process_exited_ns < self.process_started_ns:
            raise ValueError("download subprocess time order is invalid")
        if type(self.exit_code) is not int or self.exit_code != 0:
            raise ValueError("download subprocess receipt requires zero exit")
        if type(self.events) is not tuple or not self.events:
            raise TypeError("download subprocess receipt requires exact events")
        for event in self.events:
            if type(event) is not DownloadSubprocessEvent:
                raise TypeError("download subprocess receipt event type is invalid")
            event.validate()
        if tuple(event.sequence for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("download subprocess event coverage is incomplete")
        if self.diagnostic_status != CPU_DIAGNOSTIC_SELF_REPORTED:
            raise ValueError("download subprocess diagnostic status is unsupported")
        if self.formal_execution_authorized is True:
            raise DownloadExecutionBlocked(
                RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON
            )
        elif self.formal_execution_authorized is not False:
            raise TypeError("download subprocess formal flag must be boolean")
        if self.source_authority_sha256 is not None:
            raise ValueError(
                "diagnostic download receipt cannot claim source authority"
            )
        if reopen_executable:
            digest, size = _raw_sha256(
                executable,
                label="download subprocess executable",
            )
            if digest != self.executable_raw_sha256 or size != self.executable_size:
                raise ValueError("download executable changed after execution")

    def validate_against(
        self,
        *,
        plan: DownloadPlan,
        plan_authority: DownloadPlanAuthority,
    ) -> None:
        self.validate(reopen_executable=True)
        if type(plan) is not DownloadPlan:
            raise TypeError("download receipt requires an exact DownloadPlan")
        if type(plan_authority) is not DownloadPlanAuthority:
            raise TypeError("download receipt requires an exact plan authority")
        plan.__post_init__()
        plan_authority.__post_init__()
        if (
            self.download_plan_sha256 != plan.sha256
            or self.plan_authority_sha256 != plan_authority.sha256
            or self.model_revision_manifest_sha256
            != plan.inputs.model_revision_manifest_sha256
            or self.output_manifest_sha256 != plan.output_manifest_sha256
        ):
            raise ValueError("download subprocess receipt differs from exact plan")
        rows = tuple(json.loads(event.canonical_json) for event in self.events)
        expected_count = 5 + 2 * (
            len(plan.inputs.model_revisions) + len(plan.expected_outputs)
        )
        if len(rows) != expected_count:
            raise ValueError("download subprocess transcript coverage is incomplete")
        base = {
            "protocol_sha256": DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
            "download_plan_sha256": plan.sha256,
            "plan_authority_sha256": plan_authority.sha256,
        }
        ready = rows[0]
        _require_exact_fields(
            ready,
            {"kind", "protocol_sha256", "process_id"},
            label="download ready",
        )
        if (
            not _exact_json(
                ready,
                {
                    "kind": "download_subprocess_ready",
                    "protocol_sha256": DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
                    "process_id": self.process_id,
                },
            )
            or self.events[0].direction != "child_to_parent"
        ):
            raise ValueError("download subprocess ready event differs")
        start = rows[1]
        if (
            not _exact_json(
                start,
                {
                    "kind": "download_subprocess_start",
                    **base,
                    "diagnostic_only": True,
                    "network_activity_requested": False,
                    "payload_materialization_requested": False,
                    "activity_observation_available": False,
                    "model_revision_manifest_sha256": (
                        plan.inputs.model_revision_manifest_sha256
                    ),
                    "output_manifest_sha256": plan.output_manifest_sha256,
                },
            )
            or self.events[1].direction != "parent_to_child"
        ):
            raise ValueError("download subprocess start event differs")
        started = rows[2]
        if (
            not _exact_json(
                started,
                {
                    "kind": "download_subprocess_started",
                    **base,
                    "process_id": self.process_id,
                    "model_revision_manifest_sha256": (
                        plan.inputs.model_revision_manifest_sha256
                    ),
                    "output_manifest_sha256": plan.output_manifest_sha256,
                },
            )
            or self.events[2].direction != "child_to_parent"
        ):
            raise ValueError("download subprocess started event differs")
        cursor = 3
        for index, revision in enumerate(plan.inputs.model_revisions):
            request = rows[cursor]
            response = rows[cursor + 1]
            if (
                not _exact_json(
                    request,
                    {
                        "kind": "download_subprocess_model_revision",
                        **base,
                        "index": index,
                        "model_revision": revision.to_dict(),
                        "model_revision_sha256": revision.sha256,
                    },
                )
                or not _exact_json(
                    response,
                    {
                        "kind": "download_subprocess_model_revision_accepted",
                        **base,
                        "index": index,
                        "role": revision.role,
                        "revision": revision.revision,
                        "model_revision_sha256": revision.sha256,
                    },
                )
                or self.events[cursor].direction != "parent_to_child"
                or self.events[cursor + 1].direction != "child_to_parent"
            ):
                raise ValueError("download model revision exchange differs from plan")
            cursor += 2
        for index, expectation in enumerate(plan.expected_outputs):
            request = rows[cursor]
            response = rows[cursor + 1]
            if (
                not _exact_json(
                    request,
                    {
                        "kind": "download_subprocess_output_expectation",
                        **base,
                        "index": index,
                        "expectation": expectation.to_dict(),
                    },
                )
                or not _exact_json(
                    response,
                    {
                        "kind": "download_subprocess_output_expectation_accepted",
                        **base,
                        "index": index,
                        **expectation.to_dict(),
                    },
                )
                or self.events[cursor].direction != "parent_to_child"
                or self.events[cursor + 1].direction != "child_to_parent"
            ):
                raise ValueError("download output exchange differs from plan")
            cursor += 2
        if (
            not _exact_json(rows[cursor], {"kind": "download_subprocess_drain", **base})
            or self.events[cursor].direction != "parent_to_child"
            or not _exact_json(
                rows[cursor + 1],
                {
                    "kind": "download_subprocess_drained",
                    **base,
                    "active_transfers": 0,
                    "queued_transfers": 0,
                    "network_requests": 0,
                    "bytes_written": 0,
                    "activity_counter_authority": (UNTRUSTED_CHILD_ACTIVITY_COUNTERS),
                },
            )
            or self.events[cursor + 1].direction != "child_to_parent"
        ):
            raise ValueError("download subprocess drain event differs")

    def to_dict(self) -> dict[str, object]:
        self.validate(reopen_executable=False)
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "download_plan_sha256": self.download_plan_sha256,
            "plan_authority_sha256": self.plan_authority_sha256,
            "model_revision_manifest_sha256": self.model_revision_manifest_sha256,
            "output_manifest_sha256": self.output_manifest_sha256,
            "executable_path": self.executable_path,
            "executable_raw_sha256": self.executable_raw_sha256,
            "executable_size": self.executable_size,
            "argv_sha256": self.argv_sha256,
            "source_authority_sha256": self.source_authority_sha256,
            "process_id": self.process_id,
            "process_started_ns": self.process_started_ns,
            "process_exited_ns": self.process_exited_ns,
            "exit_code": self.exit_code,
            "events": [event.to_dict() for event in self.events],
            "diagnostic_status": self.diagnostic_status,
            "formal_execution_authorized": self.formal_execution_authorized,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        expected = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "download_plan_sha256",
            "plan_authority_sha256",
            "model_revision_manifest_sha256",
            "output_manifest_sha256",
            "executable_path",
            "executable_raw_sha256",
            "executable_size",
            "argv_sha256",
            "source_authority_sha256",
            "process_id",
            "process_started_ns",
            "process_exited_ns",
            "exit_code",
            "events",
            "diagnostic_status",
            "formal_execution_authorized",
        }
        if type(raw) is not dict or set(raw) != expected:
            raise ValueError("download subprocess receipt fields differ from schema")
        payload = dict(raw)
        events = payload.pop("events")
        if type(events) is not list:
            raise TypeError("download subprocess events must be a JSON array")
        value = cls(
            **payload,
            events=tuple(DownloadSubprocessEvent.from_dict(row) for row in events),
        )
        value.validate(reopen_executable=False)
        return value


@dataclass(frozen=True)
class DownloadDiagnosticResultPointer:
    schema_version: int
    kind: str
    protocol_sha256: str
    result_pointer_path: str
    download_plan_sha256: str
    plan_authority_sha256: str
    plan_source: BoundDownloadJson
    subprocess_terminal: BoundDownloadJson
    subprocess_lifecycle_sha256: str
    executable_path: str
    executable_raw_sha256: str
    executable_size: int
    argv_sha256: str
    source_authority_sha256: str | None
    model_revision_manifest_sha256: str
    output_manifest_sha256: str
    diagnostic_status: str
    formal_execution_authorized: bool

    def validate(self, *, reopen_executable: bool) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "download_diagnostic_result_pointer"
        ):
            raise ValueError("download diagnostic pointer schema is unsupported")
        if self.protocol_sha256 != DOWNLOAD_DIAGNOSTIC_RESULT_POINTER_PROTOCOL_SHA256:
            raise ValueError("download diagnostic pointer uses another protocol")
        pointer_path = _absolute_path(
            "download diagnostic result pointer",
            self.result_pointer_path,
        )
        if (
            type(self.plan_source) is not BoundDownloadJson
            or type(self.subprocess_terminal) is not BoundDownloadJson
        ):
            raise TypeError("download diagnostic pointer requires exact raw bindings")
        for label, value in (
            ("download plan", self.download_plan_sha256),
            ("plan authority", self.plan_authority_sha256),
            ("subprocess lifecycle", self.subprocess_lifecycle_sha256),
            ("executable", self.executable_raw_sha256),
            ("argv", self.argv_sha256),
            ("model revision manifest", self.model_revision_manifest_sha256),
            ("output manifest", self.output_manifest_sha256),
        ):
            _require_sha256(label, value)
        executable = _absolute_path(
            "download diagnostic executable",
            self.executable_path,
        )
        if type(self.executable_size) is not int or self.executable_size < 1:
            raise ValueError("download diagnostic executable size is invalid")
        if (
            self.plan_source.semantic_sha256 != self.download_plan_sha256
            or self.subprocess_terminal.semantic_sha256
            != self.subprocess_lifecycle_sha256
            or Path(self.plan_source.path) == pointer_path
            or Path(self.subprocess_terminal.path) == pointer_path
            or self.diagnostic_status != CPU_DIAGNOSTIC_SELF_REPORTED
        ):
            raise ValueError("download diagnostic pointer bindings differ")
        if self.formal_execution_authorized is True:
            raise DownloadExecutionBlocked(
                RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON
            )
        elif self.formal_execution_authorized is not False:
            raise TypeError("download diagnostic pointer formal flag must be boolean")
        if self.source_authority_sha256 is not None:
            raise ValueError(
                "diagnostic download pointer cannot claim source authority"
            )
        if reopen_executable:
            digest, size = _raw_sha256(
                executable,
                label="download diagnostic executable",
            )
            if digest != self.executable_raw_sha256 or size != self.executable_size:
                raise ValueError("download diagnostic executable changed")

    def to_dict(self) -> dict[str, object]:
        self.validate(reopen_executable=False)
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "result_pointer_path": self.result_pointer_path,
            "download_plan_sha256": self.download_plan_sha256,
            "plan_authority_sha256": self.plan_authority_sha256,
            "plan_source": self.plan_source.to_dict(),
            "subprocess_terminal": self.subprocess_terminal.to_dict(),
            "subprocess_lifecycle_sha256": self.subprocess_lifecycle_sha256,
            "executable_path": self.executable_path,
            "executable_raw_sha256": self.executable_raw_sha256,
            "executable_size": self.executable_size,
            "argv_sha256": self.argv_sha256,
            "source_authority_sha256": self.source_authority_sha256,
            "model_revision_manifest_sha256": self.model_revision_manifest_sha256,
            "output_manifest_sha256": self.output_manifest_sha256,
            "diagnostic_status": self.diagnostic_status,
            "formal_execution_authorized": self.formal_execution_authorized,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        expected = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "result_pointer_path",
            "download_plan_sha256",
            "plan_authority_sha256",
            "plan_source",
            "subprocess_terminal",
            "subprocess_lifecycle_sha256",
            "executable_path",
            "executable_raw_sha256",
            "executable_size",
            "argv_sha256",
            "source_authority_sha256",
            "model_revision_manifest_sha256",
            "output_manifest_sha256",
            "diagnostic_status",
            "formal_execution_authorized",
        }
        if type(raw) is not dict or set(raw) != expected:
            raise ValueError("download diagnostic pointer fields differ from schema")
        payload = dict(raw)
        payload["plan_source"] = _bound_json_from_dict(
            payload["plan_source"],
            label="download diagnostic plan",
        )
        payload["subprocess_terminal"] = _bound_json_from_dict(
            payload["subprocess_terminal"],
            label="download diagnostic terminal",
        )
        value = cls(**payload)
        value.validate(reopen_executable=False)
        return value

    def reopen(self, *, expected_plan: DownloadPlan) -> None:
        self.validate(reopen_executable=True)
        if type(expected_plan) is not DownloadPlan:
            raise TypeError("download pointer replay requires an exact plan")
        expected_plan.__post_init__()
        plan_path = Path(self.plan_source.path)
        _require_single_link_json_pair(plan_path, label="download plan")
        authority = bind_download_plan_authority(
            self.plan_source.path,
            expected_plan=expected_plan,
        )
        if (
            authority.source != self.plan_source
            or authority.sha256 != self.plan_authority_sha256
            or expected_plan.sha256 != self.download_plan_sha256
            or expected_plan.result_pointer_path != self.result_pointer_path
            or expected_plan.terminal_receipt_path != self.subprocess_terminal.path
            or expected_plan.inputs.model_revision_manifest_sha256
            != self.model_revision_manifest_sha256
            or expected_plan.output_manifest_sha256 != self.output_manifest_sha256
        ):
            raise ValueError("download diagnostic pointer plan binding changed")
        _require_single_link_json_pair(plan_path, label="download plan")
        terminal_path = Path(self.subprocess_terminal.path)
        _require_single_link_json_pair(
            terminal_path,
            label="download diagnostic subprocess terminal",
        )
        terminal_raw = self.subprocess_terminal.reopen(
            label="download diagnostic subprocess terminal"
        )
        receipt = DownloadSubprocessLifecycleReceipt.from_dict(terminal_raw)
        receipt.validate_against(plan=expected_plan, plan_authority=authority)
        if (
            receipt.sha256 != self.subprocess_lifecycle_sha256
            or receipt.executable_path != self.executable_path
            or receipt.executable_raw_sha256 != self.executable_raw_sha256
            or receipt.executable_size != self.executable_size
            or receipt.argv_sha256 != self.argv_sha256
            or receipt.source_authority_sha256 != self.source_authority_sha256
            or receipt.formal_execution_authorized
            is not self.formal_execution_authorized
        ):
            raise ValueError("download diagnostic pointer terminal binding changed")
        _require_single_link_json_pair(
            terminal_path,
            label="download diagnostic subprocess terminal",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_plan: DownloadPlan,
    ) -> Self:
        source = _absolute_path("download diagnostic result pointer", str(path))
        _require_single_link_json_pair(
            source,
            label="download diagnostic result pointer",
        )
        binding, raw = BoundDownloadJson.bind(
            source,
            expected_path=str(source),
            label="download diagnostic result pointer",
        )
        value = cls.from_dict(raw)
        if (
            value.result_pointer_path != str(source)
            or binding.semantic_sha256 != value.sha256
        ):
            raise ValueError("download diagnostic result pointer sidecar differs")
        value.reopen(expected_plan=expected_plan)
        _require_single_link_json_pair(
            source,
            label="download diagnostic result pointer",
        )
        return value


class _DownloadSubprocessDriver:
    def __init__(
        self,
        *,
        argv: tuple[str, ...],
        plan: DownloadPlan,
        plan_authority: DownloadPlanAuthority,
        timeout_seconds: float,
    ) -> None:
        if type(argv) is not tuple or not argv:
            raise TypeError("download subprocess argv must be a non-empty tuple")
        for argument in argv:
            _require_argument("download subprocess argument", argument)
        executable = _absolute_path("download subprocess executable", argv[0])
        digest, size = _raw_sha256(executable, label="download subprocess executable")
        if size < 1:
            raise ValueError("download subprocess executable is empty")
        if (
            type(timeout_seconds) not in {int, float}
            or isinstance(timeout_seconds, bool)
            or not 0 < float(timeout_seconds) <= 600
        ):
            raise ValueError("download subprocess timeout must be in (0, 600]")
        self.argv = argv
        self.plan = plan
        self.plan_authority = plan_authority
        self.timeout_seconds = float(timeout_seconds)
        self.executable_path = executable
        self.executable_raw_sha256 = digest
        self.executable_size = size
        self.argv_sha256 = content_sha256({"argv": list(argv)})
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_buffer = b""
        self._events: list[DownloadSubprocessEvent] = []
        self._process_started_ns: int | None = None
        self._process_exited_ns: int | None = None
        self._exit_code: int | None = None

    @property
    def process_id(self) -> int:
        if self._process is None or self._process.pid is None:
            raise RuntimeError("download subprocess has not been spawned")
        return self._process.pid

    @property
    def process_id_or_none(self) -> int | None:
        if self._process is None:
            return None
        return self._process.pid

    @property
    def events(self) -> tuple[DownloadSubprocessEvent, ...]:
        return tuple(self._events)

    @staticmethod
    def _encoded_message(value: Mapping[str, object]) -> bytes:
        return _canonical_bytes(dict(value))

    def _record(self, direction: str, encoded: bytes) -> None:
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise ValueError("download subprocess protocol message is too large")
        row = _strict_object(encoded, label="download subprocess protocol message")
        if encoded != _canonical_bytes(row):
            raise ValueError("download subprocess protocol message is not canonical")
        if row.get("protocol_sha256") != DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256:
            raise ValueError("download subprocess message uses another protocol")
        event = DownloadSubprocessEvent(
            sequence=len(self._events),
            direction=direction,
            canonical_json=encoded[:-1].decode("utf-8"),
            raw_sha256=hashlib.sha256(encoded).hexdigest(),
        )
        event.validate()
        self._events.append(event)

    def _send(self, value: Mapping[str, object]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("download subprocess stdin is unavailable")
        encoded = self._encoded_message(value)
        self._record("parent_to_child", encoded)
        try:
            self._process.stdin.write(encoded)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(
                "download subprocess closed its command channel"
            ) from error

    def _read_line(self) -> bytes:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("download subprocess stdout is unavailable")
        deadline = time.monotonic() + self.timeout_seconds
        descriptor = self._process.stdout.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            while b"\n" not in self._stdout_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("download subprocess response timed out")
                chunk = os.read(descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    raise RuntimeError(
                        "download subprocess exited before its protocol response"
                    )
                self._stdout_buffer += chunk
                if len(self._stdout_buffer) > _MAX_MESSAGE_BYTES:
                    raise ValueError("download subprocess response is too large")
        encoded, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
        encoded += b"\n"
        self._record("child_to_parent", encoded)
        return encoded

    def _read_remainder_to_eof(self) -> bytes:
        """Read post-protocol bytes without trusting pipe EOF to arrive."""

        if self._process is None or self._process.stdout is None:
            raise RuntimeError("download subprocess stdout is unavailable")
        remainder = bytearray(self._stdout_buffer)
        self._stdout_buffer = b""
        deadline = time.monotonic() + min(self.timeout_seconds, 2.0)
        descriptor = self._process.stdout.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError(
                        "download subprocess stdout did not reach bounded EOF"
                    )
                chunk = os.read(descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                remainder.extend(chunk)
                if len(remainder) > _MAX_MESSAGE_BYTES:
                    raise ValueError(
                        "download subprocess post-drain output is too large"
                    )
        return bytes(remainder)

    def _process_group_exists(self) -> bool:
        if self._process is None:
            return False
        try:
            os.killpg(self._process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _receive(self, *, kind: str, fields: set[str]) -> dict[str, object]:
        row = _strict_object(
            self._read_line(),
            label="download subprocess response",
        )
        expected = {"kind", "protocol_sha256", *fields}
        if set(row) != expected:
            raise ValueError("download subprocess response fields differ from protocol")
        if row["kind"] != kind:
            raise ValueError("download subprocess response kind is out of order")
        return row

    @property
    def _base(self) -> dict[str, object]:
        return {
            "protocol_sha256": DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
            "download_plan_sha256": self.plan.sha256,
            "plan_authority_sha256": self.plan_authority.sha256,
        }

    def spawn(self) -> None:
        if self._process is not None:
            raise RuntimeError("download subprocess was already spawned")
        # Caller environment is intentionally not inherited.  In particular,
        # provider tokens, proxy configuration, Python injection, and cache
        # roots never reach the diagnostic child.
        environment = {
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        self._process_started_ns = time.monotonic_ns()
        try:
            self._process = subprocess.Popen(
                self.argv,
                executable=str(self.executable_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
            )
            ready = self._receive(
                kind="download_subprocess_ready",
                fields={"process_id"},
            )
            if type(ready["process_id"]) is not int or ready["process_id"] != (
                self.process_id
            ):
                raise ValueError("download ready event names another process")
        except BaseException:
            self.abort()
            raise

    def run(self) -> DownloadSubprocessLifecycleReceipt:
        if self._process is None:
            raise RuntimeError("download subprocess was not spawned")
        self._send(
            {
                "kind": "download_subprocess_start",
                **self._base,
                "diagnostic_only": True,
                "network_activity_requested": False,
                "payload_materialization_requested": False,
                "activity_observation_available": False,
                "model_revision_manifest_sha256": (
                    self.plan.inputs.model_revision_manifest_sha256
                ),
                "output_manifest_sha256": self.plan.output_manifest_sha256,
            }
        )
        started = self._receive(
            kind="download_subprocess_started",
            fields={
                "download_plan_sha256",
                "plan_authority_sha256",
                "process_id",
                "model_revision_manifest_sha256",
                "output_manifest_sha256",
            },
        )
        if not _exact_json(
            started,
            {
                "kind": "download_subprocess_started",
                **self._base,
                "process_id": self.process_id,
                "model_revision_manifest_sha256": (
                    self.plan.inputs.model_revision_manifest_sha256
                ),
                "output_manifest_sha256": self.plan.output_manifest_sha256,
            },
        ):
            raise ValueError("download started event differs from plan")
        for index, revision in enumerate(self.plan.inputs.model_revisions):
            self._send(
                {
                    "kind": "download_subprocess_model_revision",
                    **self._base,
                    "index": index,
                    "model_revision": revision.to_dict(),
                    "model_revision_sha256": revision.sha256,
                }
            )
            accepted = self._receive(
                kind="download_subprocess_model_revision_accepted",
                fields={
                    "download_plan_sha256",
                    "plan_authority_sha256",
                    "index",
                    "role",
                    "revision",
                    "model_revision_sha256",
                },
            )
            if not _exact_json(
                accepted,
                {
                    "kind": "download_subprocess_model_revision_accepted",
                    **self._base,
                    "index": index,
                    "role": revision.role,
                    "revision": revision.revision,
                    "model_revision_sha256": revision.sha256,
                },
            ):
                raise ValueError("download model revision acknowledgement differs")
        for index, expectation in enumerate(self.plan.expected_outputs):
            self._send(
                {
                    "kind": "download_subprocess_output_expectation",
                    **self._base,
                    "index": index,
                    "expectation": expectation.to_dict(),
                }
            )
            accepted = self._receive(
                kind="download_subprocess_output_expectation_accepted",
                fields={
                    "download_plan_sha256",
                    "plan_authority_sha256",
                    "index",
                    "relative_path",
                    "size",
                    "sha256",
                },
            )
            if not _exact_json(
                accepted,
                {
                    "kind": "download_subprocess_output_expectation_accepted",
                    **self._base,
                    "index": index,
                    **expectation.to_dict(),
                },
            ):
                raise ValueError("download output acknowledgement differs")
        self._send({"kind": "download_subprocess_drain", **self._base})
        drained = self._receive(
            kind="download_subprocess_drained",
            fields={
                "download_plan_sha256",
                "plan_authority_sha256",
                "active_transfers",
                "queued_transfers",
                "network_requests",
                "bytes_written",
                "activity_counter_authority",
            },
        )
        if not _exact_json(
            drained,
            {
                "kind": "download_subprocess_drained",
                **self._base,
                "active_transfers": 0,
                "queued_transfers": 0,
                "network_requests": 0,
                "bytes_written": 0,
                "activity_counter_authority": UNTRUSTED_CHILD_ACTIVITY_COUNTERS,
            },
        ):
            raise ValueError("download subprocess activity self-report differs")
        try:
            exit_code = self._process.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                "download subprocess did not exit after drain"
            ) from error
        self._process_exited_ns = time.monotonic_ns()
        self._exit_code = exit_code
        # A descendant may inherit stdout and keep the pipe open after the
        # direct child exits.  Prove the complete process group is gone (and
        # kill it if necessary) before attempting the bounded EOF read.
        if self._process_group_exists():
            self.abort()
            raise ValueError("download subprocess left a live child process group")
        if self._read_remainder_to_eof():
            raise ValueError("download subprocess emitted output after drain")
        receipt = DownloadSubprocessLifecycleReceipt(
            schema_version=1,
            kind="download_subprocess_lifecycle_raw_receipt",
            protocol_sha256=DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
            download_plan_sha256=self.plan.sha256,
            plan_authority_sha256=self.plan_authority.sha256,
            model_revision_manifest_sha256=(
                self.plan.inputs.model_revision_manifest_sha256
            ),
            output_manifest_sha256=self.plan.output_manifest_sha256,
            executable_path=str(self.executable_path),
            executable_raw_sha256=self.executable_raw_sha256,
            executable_size=self.executable_size,
            argv_sha256=self.argv_sha256,
            source_authority_sha256=None,
            process_id=self.process_id,
            process_started_ns=self._process_started_ns or 0,
            process_exited_ns=self._process_exited_ns,
            exit_code=self._exit_code,
            events=tuple(self._events),
            diagnostic_status=CPU_DIAGNOSTIC_SELF_REPORTED,
            formal_execution_authorized=False,
        )
        receipt.validate_against(plan=self.plan, plan_authority=self.plan_authority)
        return receipt

    def abort(self) -> None:
        process = self._process
        if process is None:
            return

        def wait_for_group(deadline: float) -> bool:
            while self._process_group_exists():
                process.poll()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(0.01, remaining))
            process.poll()
            return True

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.poll()
            return
        except OSError:
            pass
        if wait_for_group(time.monotonic() + min(self.timeout_seconds, 2.0)):
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            process.poll()
            return
        if not wait_for_group(time.monotonic() + 2.0):
            raise RuntimeError("download subprocess process group survived SIGKILL")


def _publish_failed_attempt(
    *,
    plan: DownloadPlan,
    authority: DownloadPlanAuthority,
    driver: _DownloadSubprocessDriver,
    error: BaseException,
) -> None:
    attempt_time_ns = time.monotonic_ns()
    process_id = driver.process_id_or_none
    value = {
        "schema_version": 1,
        "kind": "download_diagnostic_failed_attempt",
        "protocol_sha256": DOWNLOAD_DIAGNOSTIC_ATTEMPT_PROTOCOL_SHA256,
        "download_plan_sha256": plan.sha256,
        "plan_authority_sha256": authority.sha256,
        "executable_path": str(driver.executable_path),
        "executable_raw_sha256": driver.executable_raw_sha256,
        "executable_size": driver.executable_size,
        "argv_sha256": driver.argv_sha256,
        "process_id": process_id,
        "attempt_finished_ns": attempt_time_ns,
        "state": "FAILED",
        "reason_code": "download_diagnostic_subprocess_failed",
        "failure_type": type(error).__name__,
        "events": [event.to_dict() for event in driver.events],
        "formal_execution_authorized": False,
    }
    suffix = f"{attempt_time_ns}-{0 if process_id is None else process_id}"
    path = Path(plan.inputs.evidence_root) / (
        f"{plan.inputs.cell_id}.download.attempt-{suffix}.json"
    )
    _publish_semantic_json(path, value, label="download failed attempt")


def _preflight_result(
    *,
    plan: DownloadPlan,
    executable_path: Path,
    executable_raw_sha256: str,
    executable_size: int,
    argv_sha256: str,
) -> DownloadDiagnosticResultPointer | None:
    pointer_path = Path(plan.result_pointer_path)
    pointer_sidecar = Path(f"{pointer_path}.sha256")
    terminal_path = Path(plan.terminal_receipt_path)
    terminal_sidecar = Path(f"{terminal_path}.sha256")
    pointer_exists = os.path.lexists(pointer_path)
    marker_exists = os.path.lexists(pointer_sidecar)
    if pointer_exists or marker_exists:
        if not pointer_exists or not marker_exists:
            raise ValueError("download result pointer commit marker is incomplete")
        pointer = DownloadDiagnosticResultPointer.load(
            pointer_path,
            expected_plan=plan,
        )
        if (
            pointer.executable_path != str(executable_path)
            or pointer.executable_raw_sha256 != executable_raw_sha256
            or pointer.executable_size != executable_size
            or pointer.argv_sha256 != argv_sha256
            or pointer.source_authority_sha256 is not None
            or pointer.formal_execution_authorized is not False
        ):
            raise ValueError("download result pointer uses another subprocess")
        return pointer
    if os.path.lexists(terminal_path) or os.path.lexists(terminal_sidecar):
        raise ValueError("download terminal is an uncommitted prior attempt")
    _regular_directory(pointer_path.parent, label="download result pointer parent")
    return None


def _execute_download_subprocess_path(
    plan_path: str | Path,
    *,
    expected_plan: DownloadPlan,
    argv: tuple[str, ...],
    timeout_seconds: float,
) -> DownloadDiagnosticResultPointer:
    if type(expected_plan) is not DownloadPlan:
        raise TypeError("download subprocess requires an exact expected plan")
    expected_plan.__post_init__()
    normalized_plan_path = _absolute_path("download plan", str(plan_path))
    _require_single_link_json_pair(normalized_plan_path, label="download plan")
    authority = bind_download_plan_authority(
        normalized_plan_path,
        expected_plan=expected_plan,
    )
    plan = revalidate_download_plan_authority(
        authority,
        expected_plan=expected_plan,
    )
    _require_single_link_json_pair(normalized_plan_path, label="download plan")
    driver = _DownloadSubprocessDriver(
        argv=argv,
        plan=plan,
        plan_authority=authority,
        timeout_seconds=timeout_seconds,
    )
    resumed = _preflight_result(
        plan=plan,
        executable_path=driver.executable_path,
        executable_raw_sha256=driver.executable_raw_sha256,
        executable_size=driver.executable_size,
        argv_sha256=driver.argv_sha256,
    )
    if resumed is not None:
        return resumed
    try:
        driver.spawn()
        receipt = driver.run()
    except BaseException as error:
        try:
            driver.abort()
        except (OSError, RuntimeError, ValueError) as abort_error:
            error.add_note(
                f"process-group cleanup also failed: {type(abort_error).__name__}"
            )
        try:
            _publish_failed_attempt(
                plan=plan,
                authority=authority,
                driver=driver,
                error=error,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as publication_error:
            error.add_note(
                "failed-attempt publication also failed: "
                f"{type(publication_error).__name__}"
            )
        raise
    finally:
        driver.abort()
    terminal_binding = _publish_semantic_json(
        Path(plan.terminal_receipt_path),
        receipt.to_dict(),
        label="download diagnostic subprocess terminal",
    )
    pointer = DownloadDiagnosticResultPointer(
        schema_version=1,
        kind="download_diagnostic_result_pointer",
        protocol_sha256=DOWNLOAD_DIAGNOSTIC_RESULT_POINTER_PROTOCOL_SHA256,
        result_pointer_path=plan.result_pointer_path,
        download_plan_sha256=plan.sha256,
        plan_authority_sha256=authority.sha256,
        plan_source=authority.source,
        subprocess_terminal=terminal_binding,
        subprocess_lifecycle_sha256=receipt.sha256,
        executable_path=receipt.executable_path,
        executable_raw_sha256=receipt.executable_raw_sha256,
        executable_size=receipt.executable_size,
        argv_sha256=receipt.argv_sha256,
        source_authority_sha256=receipt.source_authority_sha256,
        model_revision_manifest_sha256=receipt.model_revision_manifest_sha256,
        output_manifest_sha256=receipt.output_manifest_sha256,
        diagnostic_status=CPU_DIAGNOSTIC_SELF_REPORTED,
        formal_execution_authorized=receipt.formal_execution_authorized,
    )
    pointer.reopen(expected_plan=plan)
    pointer_binding = _publish_semantic_json(
        Path(plan.result_pointer_path),
        pointer.to_dict(),
        label="download diagnostic result pointer",
    )
    if pointer_binding.semantic_sha256 != pointer.sha256:
        raise ValueError("download diagnostic result pointer publication differs")
    pointer.reopen(expected_plan=plan)
    return pointer


def execute_download_subprocess_for_cpu_test(
    plan_path: str | Path,
    *,
    expected_plan: DownloadPlan,
    argv: tuple[str, ...],
    timeout_seconds: float = 30.0,
) -> DownloadDiagnosticResultPointer:
    """Run a caller-owned CPU-test child and retain only self-reported evidence.

    This helper provides neither network/filesystem containment nor activity
    observation.  It is deliberately incapable of minting formal completion.
    """

    return _execute_download_subprocess_path(
        plan_path,
        expected_plan=expected_plan,
        argv=argv,
        timeout_seconds=timeout_seconds,
    )


def execute_release_download_plan(
    plan_path: str | Path,
    *,
    expected_plan: DownloadPlan,
) -> NoReturn:
    """Named BLOCK before plan-path access, process spawn, network, or output."""

    # These source-owned empty policies intentionally precede even validation
    # of the caller-supplied path and expected plan.
    if len(RELEASE_DOWNLOAD_SUBPROCESSES) != 1:
        raise DownloadExecutionBlocked(RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON)
    if not RELEASE_TRUSTED_DOWNLOAD_PLAN_SHA256S:
        raise DownloadExecutionBlocked(RELEASE_DOWNLOAD_PLAN_ALLOWLIST_EMPTY)
    if type(expected_plan) is not DownloadPlan:
        raise TypeError("release download runner requires an exact expected plan")
    source = RELEASE_DOWNLOAD_SUBPROCESSES[0]
    source.validate(reopen_executable=True)
    normalized = _absolute_path("release download plan", str(plan_path))
    authority = bind_download_plan_authority(
        normalized,
        expected_plan=expected_plan,
    )
    plan = revalidate_download_plan_authority(
        authority,
        expected_plan=expected_plan,
    )
    if plan.sha256 not in RELEASE_TRUSTED_DOWNLOAD_PLAN_SHA256S:
        raise DownloadExecutionBlocked(RELEASE_DOWNLOAD_PLAN_UNTRUSTED)
    # The existing release gate remains independently unconditional.  Updating
    # this runner's command maps cannot silently authorize a downloader.
    require_release_download_execution(plan)


__all__ = [
    "CPU_DIAGNOSTIC_SELF_REPORTED",
    "DOWNLOAD_DIAGNOSTIC_ATTEMPT_PROTOCOL_SHA256",
    "DOWNLOAD_DIAGNOSTIC_RESULT_POINTER_PROTOCOL_SHA256",
    "DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256",
    "RELEASE_DOWNLOAD_PLAN_ALLOWLIST_EMPTY",
    "RELEASE_DOWNLOAD_PLAN_UNTRUSTED",
    "RELEASE_DOWNLOAD_SUBPROCESSES",
    "RELEASE_TRUSTED_DOWNLOAD_PLAN_SHA256S",
    "UNTRUSTED_CHILD_ACTIVITY_COUNTERS",
    "DownloadDiagnosticResultPointer",
    "DownloadSubprocessEvent",
    "DownloadSubprocessLifecycleReceipt",
    "ReleaseDownloadSubprocess",
    "execute_download_subprocess_for_cpu_test",
    "execute_release_download_plan",
]
