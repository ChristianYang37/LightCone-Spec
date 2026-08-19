"""Durable state and scheduling policy for long-running formal experiments.

SQLite is authoritative, process launch is ordered after a durable ``RUNNING``
transition, and human-readable progress files are projections of that database.
OS-facing process, GPU, terminal-evidence, and archive operations are injected
through callbacks so the policy stays deterministic and CPU-testable.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self

CellAttemptStatus = Literal[
    "PENDING",
    "RUNNING",
    "COMPLETE",
    "FAILED",
    "BLOCKED",
    "STALE_IDENTITY",
]
TerminalAttemptStatus = Literal["COMPLETE", "FAILED", "BLOCKED", "STALE_IDENTITY"]
WatchdogSeverity = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
InterferenceMode = Literal["UNRESOLVED", "ISOLATED", "DUAL_SINGLE"]
CommandTimingClass = Literal[
    "HEADLINE",
    "SAFE_AUXILIARY",
    "EXCLUSIVE",
    "PROFILER",
    "FAILURE",
    "ARCHIVE",
]
TerminalFailureClass = Literal[
    "INFRASTRUCTURE",
    "SCIENTIFIC",
    "UNSAFE",
    "OOM_CANDIDATE",
    "EXACTNESS",
    "FAILURE_DIAGNOSTIC",
]

CELL_ATTEMPT_STATUSES = frozenset(
    {"PENDING", "RUNNING", "COMPLETE", "FAILED", "BLOCKED", "STALE_IDENTITY"}
)
TERMINAL_ATTEMPT_STATUSES = frozenset(
    {"COMPLETE", "FAILED", "BLOCKED", "STALE_IDENTITY"}
)
WATCHDOG_SEVERITIES = frozenset({"INFO", "WARNING", "ERROR", "CRITICAL"})
REMOTE_SPOOL_SAFETY_RESERVE_BYTES = 15 * 1024**3

_SCHEMA_VERSION = 7
_PREVIOUS_SCHEMA_VERSIONS = frozenset({2, 3, 4, 5, 6})
_DEFAULT_COMMAND_MAX_RUNTIME_SECONDS = 6 * 60 * 60
_DEFAULT_COMMAND_MAX_LOG_STALL_SECONDS = 30 * 60
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_IDENTITY_SHA_FIELDS = (
    "source_sha256",
    "patch_sha256",
    "registry_sha256",
)


class ExperimentOperatorError(RuntimeError):
    """Base error for operator state violations."""


class OperatorAlreadyRunningError(ExperimentOperatorError):
    """Raised when another process owns the singleton operator lock."""


class AttemptTransitionError(ExperimentOperatorError):
    """Raised when an attempt lifecycle transition is not allowed."""


@dataclass(frozen=True)
class InterferenceEnvelope:
    """Frozen authorization for one or two independent single-GPU workers."""

    mode: InterferenceMode
    gpu_uuids: tuple[str, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in {"UNRESOLVED", "ISOLATED", "DUAL_SINGLE"}:
            raise ValueError("interference mode is not registered")
        uuids = _validated_gpu_uuids(self.gpu_uuids)
        if uuids != tuple(sorted(uuids)):
            raise ValueError("interference GPU UUIDs must be canonical")
        if self.mode == "DUAL_SINGLE" and len(uuids) < 2:
            raise ValueError("dual-single mode requires at least two GPUs")
        _require_sha256(self.evidence_sha256, "interference evidence SHA-256")


@dataclass(frozen=True)
class QueuedCommandSpec:
    """Exact, already-materialized command consumed by the scheduler."""

    cell_id: str
    attempt: int
    argv: tuple[str, ...]
    launch_compatibility_key: str
    required_gpu_count: int
    timing_class: CommandTimingClass
    predicted_high_water_bytes: int
    monitored_path: str
    log_path: str
    expected_terminal_path: str
    expected_junit_path: str
    expected_raw_log_path: str
    atomic_pointer_path: str
    child_exit_receipt_path: str
    environment: tuple[tuple[str, str], ...] = ()
    paired_gpu_key: str | None = None
    preferred_gpu_index: int | None = None
    priority: int = 0
    max_runtime_seconds: int = _DEFAULT_COMMAND_MAX_RUNTIME_SECONDS
    max_log_stall_seconds: int = _DEFAULT_COMMAND_MAX_LOG_STALL_SECONDS

    def __post_init__(self) -> None:
        _require_text(self.cell_id, "queued cell ID")
        _require_positive_int(self.attempt, "queued attempt")
        if (
            type(self.argv) is not tuple
            or not self.argv
            or any(type(value) is not str or not value for value in self.argv)
        ):
            raise ValueError("queued argv must be a non-empty tuple of strings")
        _require_text(self.launch_compatibility_key, "launch compatibility key")
        _require_positive_int(self.required_gpu_count, "required GPU count")
        if self.timing_class not in {
            "HEADLINE",
            "SAFE_AUXILIARY",
            "EXCLUSIVE",
            "PROFILER",
            "FAILURE",
            "ARCHIVE",
        }:
            raise ValueError("queued timing class is not registered")
        if self.timing_class == "HEADLINE" and self.required_gpu_count != 1:
            raise ValueError("headline cells must be complete single-GPU cells")
        if (
            isinstance(self.predicted_high_water_bytes, bool)
            or not isinstance(self.predicted_high_water_bytes, int)
            or self.predicted_high_water_bytes < 0
        ):
            raise ValueError("predicted high-water bytes must be non-negative")
        for label, value in (
            ("monitored path", self.monitored_path),
            ("log path", self.log_path),
            ("expected terminal path", self.expected_terminal_path),
            ("expected JUnit path", self.expected_junit_path),
            ("expected raw-log path", self.expected_raw_log_path),
            ("atomic pointer path", self.atomic_pointer_path),
            ("child exit-receipt path", self.child_exit_receipt_path),
        ):
            _require_text(value, label)
            if not Path(value).is_absolute():
                raise ValueError(f"queued {label} must be absolute")
        paths = (
            self.log_path,
            self.expected_terminal_path,
            self.expected_junit_path,
            self.expected_raw_log_path,
            self.atomic_pointer_path,
            self.child_exit_receipt_path,
        )
        if len(set(paths)) != len(paths):
            raise ValueError("queued command evidence paths must be distinct")
        if type(self.environment) is not tuple:
            raise TypeError("queued environment must be a canonical tuple")
        prior_name: str | None = None
        for row in self.environment:
            if (
                type(row) is not tuple
                or len(row) != 2
                or any(type(value) is not str for value in row)
            ):
                raise TypeError("queued environment rows must be string pairs")
            name, value = row
            _require_text(name, "queued environment name")
            if "=" in name or "\x00" in name or "\x00" in value:
                raise ValueError("queued environment contains an invalid entry")
            if prior_name is not None and name <= prior_name:
                raise ValueError("queued environment must be uniquely sorted by name")
            prior_name = name
        if any(name == "CUDA_VISIBLE_DEVICES" for name, _ in self.environment):
            raise ValueError("CUDA_VISIBLE_DEVICES is assigned only by the scheduler")
        if self.paired_gpu_key is not None:
            _require_text(self.paired_gpu_key, "paired GPU key")
        if self.preferred_gpu_index is not None and (
            isinstance(self.preferred_gpu_index, bool)
            or not isinstance(self.preferred_gpu_index, int)
            or self.preferred_gpu_index < 0
        ):
            raise ValueError("preferred GPU index must be non-negative or null")
        if self.preferred_gpu_index is not None and self.required_gpu_count != 1:
            raise ValueError("preferred GPU index applies only to single-GPU commands")
        if (self.paired_gpu_key is None) != (self.preferred_gpu_index is None):
            raise ValueError(
                "paired GPU key and preferred GPU index must be present together"
            )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("queued priority must be an integer")
        for label, value in (
            ("maximum runtime", self.max_runtime_seconds),
            ("maximum log stall", self.max_log_stall_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"queued {label} must be a positive integer")
        if self.max_log_stall_seconds > self.max_runtime_seconds:
            raise ValueError("queued log-stall limit exceeds maximum runtime")

    @property
    def command_sha256(self) -> str:
        identity = {
            "argv": self.argv,
            "environment": self.environment,
            "launch_compatibility_key": self.launch_compatibility_key,
            "required_gpu_count": self.required_gpu_count,
            "timing_class": self.timing_class,
            "paired_gpu_key": self.paired_gpu_key,
            "preferred_gpu_index": self.preferred_gpu_index,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_log_stall_seconds": self.max_log_stall_seconds,
        }
        return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TerminalEvidence:
    """Result returned only after an injected atomic terminal validator succeeds."""

    status: Literal["COMPLETE", "FAILED"]
    exit_code: int | None
    atomic_publication_sha256: str
    terminal_sha256: str | None = None
    junit_sha256: str | None = None
    raw_log_sha256: str | None = None
    evidence_files: Mapping[str, str] | None = None
    failure_class: TerminalFailureClass | None = None
    failure_code: str | None = None
    exclusion_reason: str | None = None
    included_in_analysis: bool = True
    started_ns: int | None = None
    finished_ns: int | None = None

    def __post_init__(self) -> None:
        if self.status not in {"COMPLETE", "FAILED"}:
            raise ValueError("terminal evidence status differs")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise ValueError("terminal evidence exit code must be integer or null")
        _require_sha256(
            self.atomic_publication_sha256,
            "atomic terminal publication SHA-256",
        )
        for label, value in (
            ("terminal SHA-256", self.terminal_sha256),
            ("JUnit SHA-256", self.junit_sha256),
            ("raw-log SHA-256", self.raw_log_sha256),
        ):
            if value is not None:
                _require_sha256(value, label)
        for path, digest in dict(self.evidence_files or {}).items():
            _require_text(path, "terminal evidence path")
            _require_sha256(digest, f"terminal evidence SHA-256 for {path}")
        if (self.started_ns is None) != (self.finished_ns is None):
            raise ValueError("terminal timing must be both present or absent")
        if self.started_ns is not None and (
            type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or self.started_ns <= 0
            or self.finished_ns <= self.started_ns
        ):
            raise ValueError("terminal timing must be positive and increasing")
        if self.status == "COMPLETE":
            if (
                self.exit_code != 0
                or None
                in (self.terminal_sha256, self.junit_sha256, self.raw_log_sha256)
                or self.failure_class is not None
                or self.failure_code is not None
                or self.included_in_analysis == (self.exclusion_reason is not None)
            ):
                raise ValueError("COMPLETE terminal evidence is incomplete")
        elif (
            self.failure_class
            not in {
                "INFRASTRUCTURE",
                "SCIENTIFIC",
                "UNSAFE",
                "OOM_CANDIDATE",
                "EXACTNESS",
                "FAILURE_DIAGNOSTIC",
            }
            or not self.failure_code
            or not self.exclusion_reason
            or self.included_in_analysis
        ):
            raise ValueError("FAILED terminal evidence lacks classified exclusion")


RetryBuilder = Callable[
    [QueuedCommandSpec, int],
    tuple["CellAttemptSpec", QueuedCommandSpec],
]


@dataclass(frozen=True)
class SchedulerCallbacks:
    launch: Callable[[QueuedCommandSpec, tuple[str, ...]], SpawnedProcess]
    process_probe: Callable[[int, int], ProcessObservation]
    log_size_bytes: Callable[[QueuedCommandSpec], int]
    gpu_snapshot: Callable[[tuple[str, ...]], Mapping[str, Any]]
    terminal_validator: Callable[
        [QueuedCommandSpec, Mapping[str, Any], ProcessObservation],
        TerminalEvidence | None,
    ]
    free_disk_bytes: Callable[[str], int]
    retry_builder: RetryBuilder | None = None
    recover_started_process: (
        Callable[[QueuedCommandSpec], RecoveredProcessStart | None] | None
    ) = None
    worker_heartbeat: Callable[[QueuedCommandSpec], WorkerHeartbeat | None] | None = (
        None
    )
    worker_heartbeat_required: Callable[[QueuedCommandSpec], bool] | None = None
    send_term: Callable[[QueuedCommandSpec, int, int], None] | None = None
    send_kill: Callable[[QueuedCommandSpec, int, int], None] | None = None
    process_group_alive: Callable[[int], bool] | None = None
    independent_process_groups: (
        Callable[[QueuedCommandSpec], tuple[int, ...]] | None
    ) = None
    partial_evidence: Callable[[QueuedCommandSpec], Mapping[str, str]] | None = None


@dataclass(frozen=True)
class SchedulerCycleResult:
    reconciled: tuple[tuple[str, int, str], ...]
    dispatched: tuple[tuple[str, int, tuple[str, ...]], ...]
    dispatch_state: Literal["RUN", "STOP"]
    stop_reason: str | None


@dataclass(frozen=True)
class StagePlanEntry:
    node: str
    ordinal: int
    stage: str
    phase: str
    expected_formula: str
    known_expected_cells: int | None
    estimated_remaining_gpu_hours: float | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("node", self.node),
            ("stage", self.stage),
            ("phase", self.phase),
            ("expected_formula", self.expected_formula),
        ):
            _require_text(value, label)
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("stage-plan ordinal must be a non-negative integer")
        if self.known_expected_cells is not None and (
            isinstance(self.known_expected_cells, bool)
            or not isinstance(self.known_expected_cells, int)
            or self.known_expected_cells < 0
        ):
            raise ValueError("known expected cells must be non-negative or null")
        _require_nonnegative_finite_or_none(
            self.estimated_remaining_gpu_hours,
            "estimated remaining GPU-hours",
        )


@dataclass(frozen=True)
class ControllerArtifactBinding:
    """Path and raw SHA-256 for one durable DAG-controller artifact."""

    absolute_path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_text(self.absolute_path, "controller artifact path")
        path = Path(self.absolute_path)
        if not path.is_absolute() or path != path.resolve(strict=False):
            raise ValueError("controller artifact path must be absolute and normalized")
        _require_sha256(self.sha256, "controller artifact SHA-256")

    @classmethod
    def bind(cls, path: str | Path) -> ControllerArtifactBinding:
        source = Path(path)
        if not source.is_absolute() or source != source.resolve(strict=False):
            raise ValueError("controller artifact path must be absolute and normalized")
        if source.is_symlink() or not source.is_file():
            raise ValueError("controller artifact must be one regular file")
        before = source.stat(follow_symlinks=False)
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = source.stat(follow_symlinks=False)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError("controller artifact changed while hashed")
        return cls(str(source), digest.hexdigest())


@dataclass(frozen=True)
class AuxiliaryJobSpec:
    """One real pre-materialization GPU observation awaiting cell adoption."""

    job_id: str
    attempt: int
    adoption_key: str
    scientific_axes: Mapping[str, Any]
    identity: Mapping[str, Any]
    command_sha256: str
    output_directory: str

    def __post_init__(self) -> None:
        _require_text(self.job_id, "auxiliary job ID")
        _require_positive_int(self.attempt, "auxiliary job attempt")
        _require_text(self.adoption_key, "auxiliary adoption key")
        _canonical_mapping(
            self.scientific_axes,
            "auxiliary scientific axes",
            allow_empty=False,
        )
        identity = _canonical_mapping(
            self.identity,
            "auxiliary identity",
            allow_empty=False,
        )
        for field in _REQUIRED_IDENTITY_SHA_FIELDS:
            _require_sha256(identity.get(field), f"auxiliary identity {field}")
        _require_sha256(self.command_sha256, "auxiliary command SHA-256")
        _require_text(self.output_directory, "auxiliary output directory")
        output = Path(self.output_directory)
        if not output.is_absolute() or output != output.resolve(strict=False):
            raise ValueError(
                "auxiliary output directory must be absolute and normalized"
            )


@dataclass(frozen=True)
class AuxiliaryPhysicalGroupSpec:
    """One durable physical campaign executed before scientific cell IDs exist."""

    group_id: str
    attempt: int
    node: str
    source_kind: str
    jobs: tuple[AuxiliaryJobSpec, ...]
    assigned_gpu_uuids: tuple[str, ...]
    launch_command_sha256: str
    output_directory: str
    process_hard_timeout_ns: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.group_id, "auxiliary group ID")
        _require_positive_int(self.attempt, "auxiliary group attempt")
        _require_text(self.node, "auxiliary controller node")
        _require_text(self.source_kind, "auxiliary source kind")
        if (
            type(self.jobs) is not tuple
            or not self.jobs
            or any(type(job) is not AuxiliaryJobSpec for job in self.jobs)
        ):
            raise TypeError("auxiliary group requires exact non-empty job specs")
        identities = tuple((job.job_id, job.attempt) for job in self.jobs)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("auxiliary jobs must be uniquely sorted by identity")
        if any(job.attempt != self.attempt for job in self.jobs):
            raise ValueError("auxiliary job attempts differ from their group attempt")
        adoption_keys = tuple(job.adoption_key for job in self.jobs)
        if len(set(adoption_keys)) != len(adoption_keys):
            raise ValueError("auxiliary adoption keys must be unique within a group")
        output_directories = tuple(job.output_directory for job in self.jobs)
        if len(set(output_directories)) != len(output_directories):
            raise ValueError("auxiliary job output directories must be unique")
        uuids = _validated_gpu_uuids(self.assigned_gpu_uuids)
        if not uuids:
            raise ValueError("auxiliary GPU campaign requires at least one GPU")
        if uuids != self.assigned_gpu_uuids:
            raise ValueError("auxiliary GPU UUIDs must retain canonical tuple form")
        _require_sha256(
            self.launch_command_sha256,
            "auxiliary launch-command SHA-256",
        )
        _require_text(self.output_directory, "auxiliary group output directory")
        output = Path(self.output_directory)
        if not output.is_absolute() or output != output.resolve(strict=False):
            raise ValueError(
                "auxiliary group output directory must be absolute and normalized"
            )
        if self.process_hard_timeout_ns is not None and (
            type(self.process_hard_timeout_ns) is not int
            or self.process_hard_timeout_ns < 1
        ):
            raise ValueError("auxiliary process hard timeout is invalid")


@dataclass(frozen=True)
class AuxiliaryGroupTerminal:
    """Deep-bound group publication and exact per-job terminal evidence."""

    publication: ControllerArtifactBinding
    terminals: Mapping[str, TerminalEvidence]
    compute_gpu_seconds: float
    reserved_gpu_seconds: float
    billed_gpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        if type(self.publication) is not ControllerArtifactBinding:
            raise TypeError("auxiliary terminal publication requires an exact binding")
        if not isinstance(self.terminals, Mapping) or not self.terminals:
            raise TypeError("auxiliary terminal requires a non-empty job mapping")
        if any(
            type(job_id) is not str or not job_id or type(value) is not TerminalEvidence
            for job_id, value in self.terminals.items()
        ):
            raise TypeError("auxiliary terminal mapping differs")
        publications = {
            value.atomic_publication_sha256 for value in self.terminals.values()
        }
        if len(publications) != 1:
            raise ValueError(
                "auxiliary job terminals do not share one atomic publication"
            )
        for label, value in (
            ("auxiliary compute GPU-seconds", self.compute_gpu_seconds),
            ("auxiliary reserved GPU-seconds", self.reserved_gpu_seconds),
            ("auxiliary billed GPU-seconds", self.billed_gpu_seconds),
        ):
            _require_nonnegative_finite(value, label)
        if self.reserved_gpu_seconds < self.compute_gpu_seconds:
            raise ValueError("auxiliary reserved GPU time is below compute GPU time")


@dataclass(frozen=True)
class AuxiliaryCellAdoption:
    """Map one completed auxiliary observation to its later exact cell ID."""

    job_id: str
    job_attempt: int
    adoption_key: str
    attempt: CellAttemptSpec

    def __post_init__(self) -> None:
        _require_text(self.job_id, "auxiliary adoption job ID")
        _require_positive_int(self.job_attempt, "auxiliary adoption job attempt")
        _require_text(self.adoption_key, "auxiliary adoption key")
        if type(self.attempt) is not CellAttemptSpec:
            raise TypeError("auxiliary adoption requires an exact cell-attempt spec")


@dataclass(frozen=True)
class CellAttemptSpec:
    cell_id: str
    attempt: int
    stage: str
    phase: str
    block: str | None
    seed: int | None
    scientific_axes: Mapping[str, Any]
    identity: Mapping[str, Any]
    command_sha256: str
    output_directory: str
    scientific_command_sha256: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("cell_id", self.cell_id),
            ("stage", self.stage),
            ("phase", self.phase),
        ):
            _require_text(value, label)
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("attempt must be a positive integer")
        if self.block is not None:
            _require_text(self.block, "block")
        if self.seed is not None and isinstance(self.seed, bool):
            raise ValueError("seed must be an integer or null")
        if not isinstance(self.seed, (int, type(None))):
            raise TypeError("seed must be an integer or null")
        _canonical_mapping(self.scientific_axes, "scientific axes", allow_empty=False)
        identity = _canonical_mapping(self.identity, "identity", allow_empty=False)
        for field in _REQUIRED_IDENTITY_SHA_FIELDS:
            _require_sha256(identity.get(field), f"identity {field}")
        _require_sha256(self.command_sha256, "command SHA-256")
        if self.scientific_command_sha256 is not None:
            _require_sha256(
                self.scientific_command_sha256,
                "path-independent scientific command SHA-256",
            )
        _require_text(self.output_directory, "attempt output directory")
        output = Path(self.output_directory)
        if not output.is_absolute():
            raise ValueError("attempt output directory must be absolute")


@dataclass(frozen=True)
class PhysicalAttemptGroupMemberSpec:
    """One logical attempt executed by a registered shared physical parent."""

    attempt: CellAttemptSpec
    command: QueuedCommandSpec
    logical_kind: Literal["compile", "exactness", "interference", "serving"]

    def __post_init__(self) -> None:
        if (
            type(self.attempt) is not CellAttemptSpec
            or type(self.command) is not QueuedCommandSpec
        ):
            raise TypeError("physical group member requires exact operator specs")
        if (self.attempt.cell_id, self.attempt.attempt) != (
            self.command.cell_id,
            self.command.attempt,
        ):
            raise ValueError("physical group member attempt and command differ")
        if self.attempt.command_sha256 != self.command.command_sha256:
            raise ValueError("physical group member command digest differs")
        if self.logical_kind not in {
            "compile",
            "exactness",
            "interference",
            "serving",
        }:
            raise ValueError("physical group logical kind is not registered")


def _physical_attempt_group_kind(
    members: Sequence[PhysicalAttemptGroupMemberSpec] | Sequence[Mapping[str, Any]],
) -> Literal["preflight_exact_ten", "tp1_serving_session"]:
    kinds = tuple(
        row.logical_kind
        if type(row) is PhysicalAttemptGroupMemberSpec
        else str(row["logical_kind"])
        for row in members
    )
    if (
        len(kinds) == 10
        and kinds.count("compile") == 1
        and kinds.count("exactness") == 1
        and kinds.count("interference") == 8
    ):
        return "preflight_exact_ten"
    if 2 <= len(kinds) <= 32 and set(kinds) == {"serving"}:
        return "tp1_serving_session"
    raise ValueError("physical attempt group coverage is not registered")


@dataclass(frozen=True)
class SpawnedProcess:
    """Process metadata returned by an injected launcher."""

    pid: int
    pgid: int
    process_start_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.pid, "PID")
        _require_positive_int(self.pgid, "PGID")
        if self.process_start_receipt_sha256 is not None:
            _require_sha256(
                self.process_start_receipt_sha256,
                "spawned process start receipt SHA-256",
            )


@dataclass(frozen=True)
class RecoveredProcessStart:
    """Deep-validated wrapper start receipt used to close commit/spawn crashes."""

    pid: int
    pgid: int
    started_ns: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        _require_positive_int(self.pid, "recovered PID")
        _require_positive_int(self.pgid, "recovered PGID")
        _require_positive_int(self.started_ns, "recovered process start time")
        if self.pid != self.pgid:
            raise ValueError("recovered process must be a setsid session leader")
        _require_sha256(self.receipt_sha256, "process start receipt SHA-256")


@dataclass(frozen=True)
class WorkerHeartbeat:
    """One child-authored progress heartbeat, distinct from scheduler sampling."""

    command_sha256: str
    worker_pid: int
    sequence: int
    observed_at_ns: int
    phase: str

    def __post_init__(self) -> None:
        _require_sha256(self.command_sha256, "worker heartbeat command SHA-256")
        _require_positive_int(self.worker_pid, "worker heartbeat PID")
        _require_positive_int(self.sequence, "worker heartbeat sequence")
        _require_positive_int(self.observed_at_ns, "worker heartbeat time")
        _require_text(self.phase, "worker heartbeat phase")


@dataclass(frozen=True)
class ProcessObservation:
    pid: int
    alive: bool
    observed_pgid: int | None
    reason: str
    exit_code: int | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.pid, "observed PID")
        if not isinstance(self.alive, bool):
            raise TypeError("process observation alive flag must be boolean")
        if self.observed_pgid is not None:
            _require_positive_int(self.observed_pgid, "observed PGID")
        _require_text(self.reason, "process observation reason")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise ValueError("exit code must be an integer or null")


@dataclass(frozen=True)
class WatchdogPolicy:
    process_attach_grace_seconds: float = 30.0
    heartbeat_timeout_seconds: float = 120.0
    log_stall_timeout_seconds: float = 300.0
    termination_grace_seconds: float = 60.0
    event_repeat_seconds: float = 300.0
    minimum_free_disk_bytes: int = 15 * 1024**3

    def __post_init__(self) -> None:
        for label, value in (
            ("process attach grace", self.process_attach_grace_seconds),
            ("heartbeat timeout", self.heartbeat_timeout_seconds),
            ("log stall timeout", self.log_stall_timeout_seconds),
            ("termination grace", self.termination_grace_seconds),
            ("event repeat", self.event_repeat_seconds),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{label} must be finite and positive")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{label} must be finite and positive")
        if (
            isinstance(self.minimum_free_disk_bytes, bool)
            or not isinstance(self.minimum_free_disk_bytes, int)
            or self.minimum_free_disk_bytes < 0
        ):
            raise ValueError("minimum free disk bytes must be non-negative")


@dataclass(frozen=True)
class WatchdogFinding:
    event_id: int
    event_type: str
    severity: WatchdogSeverity
    cell_id: str | None
    attempt: int | None
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class MetricRecord:
    stage: str
    phase: str
    cell_id: str
    attempt: int
    metric_name: str
    metric_kind: Literal["headline", "descriptive"]
    point_estimate: float
    ci_low: float | None
    ci_high: float | None
    independent_block_count: int | None
    request_count: int | None
    paired: bool | None
    reducer_method: str
    attributes: Mapping[str, Any]

    def __post_init__(self) -> None:
        for label, value in (
            ("stage", self.stage),
            ("phase", self.phase),
            ("cell_id", self.cell_id),
            ("metric_name", self.metric_name),
            ("reducer method", self.reducer_method),
        ):
            _require_text(value, label)
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("metric attempt must be positive")
        if self.metric_kind not in {"headline", "descriptive"}:
            raise ValueError("metric kind must be headline or descriptive")
        _require_finite(self.point_estimate, "point estimate")
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError(
                "confidence interval bounds must be both present or absent"
            )
        if self.ci_low is not None and self.ci_high is not None:
            _require_finite(self.ci_low, "confidence interval low")
            _require_finite(self.ci_high, "confidence interval high")
            if self.ci_low > self.ci_high:
                raise ValueError("confidence interval low exceeds high")
        for label, value in (
            ("independent block count", self.independent_block_count),
            ("request count", self.request_count),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{label} must be non-negative or null")
        if self.metric_kind == "headline":
            if self.ci_low is None or self.ci_high is None:
                raise ValueError("headline metrics require a 95% confidence interval")
            if self.independent_block_count is None or self.request_count is None:
                raise ValueError("headline metrics require block and request counts")
            if self.paired is None:
                raise ValueError("headline metrics require paired/unpaired identity")
        _canonical_mapping(self.attributes, "metric attributes", allow_empty=True)


@dataclass(frozen=True)
class ExportManifest:
    run_id: str
    exported_at_ns: int
    files: Mapping[str, str]


@dataclass(frozen=True)
class ProviderRuntimeSample:
    """Credential-free provider observation used for instance billing.

    ``response_sha256`` binds the redacted raw API response stored in the run
    evidence tree.  The credential itself is never accepted by this state
    layer.  A shutdown sample carries the provider's final stop time; running
    samples intentionally leave it null.
    """

    instance_uuid: str
    state: Literal["running", "shutdown"]
    observed_at_ns: int
    provider_started_at_ns: int
    provider_stopped_at_ns: int | None
    gpu_count: int
    response_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.instance_uuid, "provider instance UUID")
        if self.state not in {"running", "shutdown"}:
            raise ValueError("provider runtime state is not registered")
        for label, value in (
            ("provider observation time", self.observed_at_ns),
            ("provider start time", self.provider_started_at_ns),
        ):
            _require_positive_int(value, label)
        if self.observed_at_ns < self.provider_started_at_ns:
            raise ValueError("provider observation precedes instance start")
        if isinstance(self.gpu_count, bool) or not isinstance(self.gpu_count, int):
            raise TypeError("provider GPU count must be an integer")
        if self.gpu_count < 1:
            raise ValueError("provider GPU count must be positive")
        _require_sha256(self.response_sha256, "provider response SHA-256")
        if self.state == "running":
            if self.provider_stopped_at_ns is not None:
                raise ValueError("running provider sample cannot carry a stop time")
        else:
            if self.provider_stopped_at_ns is None:
                raise ValueError("shutdown provider sample requires a stop time")
            _require_positive_int(self.provider_stopped_at_ns, "provider stop time")
            if not (
                self.provider_started_at_ns
                < self.provider_stopped_at_ns
                <= self.observed_at_ns
            ):
                raise ValueError("provider shutdown timing is inconsistent")

    @property
    def sample_id(self) -> str:
        return hashlib.sha256(_canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LegacyStaleAttempt:
    """A pre-takeover attempt imported only as excluded historical evidence."""

    spec: CellAttemptSpec
    original_status: str
    exclusion_reason: str
    started_at_ns: int | None = None
    finished_at_ns: int | None = None
    exit_code: int | None = None
    terminal_sha256: str | None = None
    junit_sha256: str | None = None
    raw_log_sha256: str | None = None
    evidence_files: Mapping[str, str] | None = None
    compute_gpu_seconds: float = 0.0
    reserved_gpu_seconds: float = 0.0
    billed_gpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        _require_text(self.original_status, "legacy original status")
        _require_text(self.exclusion_reason, "legacy exclusion reason")
        if self.started_at_ns is not None:
            _require_positive_int(self.started_at_ns, "legacy start time")
        if self.finished_at_ns is not None:
            _require_positive_int(self.finished_at_ns, "legacy finish time")
        if (
            self.started_at_ns is not None
            and self.finished_at_ns is not None
            and self.finished_at_ns < self.started_at_ns
        ):
            raise ValueError("legacy finish time precedes start time")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise TypeError("legacy exit code must be an integer or null")
        for label, value in (
            ("legacy terminal SHA-256", self.terminal_sha256),
            ("legacy JUnit SHA-256", self.junit_sha256),
            ("legacy raw-log SHA-256", self.raw_log_sha256),
        ):
            if value is not None:
                _require_sha256(value, label)
        for path, digest in dict(self.evidence_files or {}).items():
            _require_text(path, "legacy evidence path")
            _require_sha256(digest, f"legacy evidence SHA-256 for {path}")
        for label, value in (
            ("legacy compute GPU-seconds", self.compute_gpu_seconds),
            ("legacy reserved GPU-seconds", self.reserved_gpu_seconds),
            ("legacy billed GPU-seconds", self.billed_gpu_seconds),
        ):
            _require_nonnegative_finite(value, label)


@dataclass(frozen=True)
class DiskDispatchDecision:
    action: Literal["ALLOW", "STOP"]
    free_bytes: int
    predicted_next_wave_high_water_bytes: int
    safety_reserve_bytes: int
    required_free_bytes: int
    reason: str


@dataclass(frozen=True)
class ArchiveRequest:
    archive_id: str
    safe_boundary: str
    remote_payload_root: str
    local_partial_root: str
    local_final_root: str
    remote_manifest_sha256: str
    predicted_payload_bytes: int = 0
    cell_id: str | None = None
    attempt: int | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("archive ID", self.archive_id),
            ("archive safe boundary", self.safe_boundary),
            ("remote payload root", self.remote_payload_root),
            ("local partial root", self.local_partial_root),
            ("local final root", self.local_final_root),
        ):
            _require_text(value, label)
        for label, value in (
            ("remote payload root", self.remote_payload_root),
            ("local partial root", self.local_partial_root),
            ("local final root", self.local_final_root),
        ):
            if not Path(value).is_absolute():
                raise ValueError(f"{label} must be absolute")
        if self.local_partial_root == self.local_final_root:
            raise ValueError("archive partial and final roots must differ")
        _require_sha256(self.remote_manifest_sha256, "remote manifest SHA-256")
        if (
            isinstance(self.predicted_payload_bytes, bool)
            or not isinstance(self.predicted_payload_bytes, int)
            or self.predicted_payload_bytes < 0
        ):
            raise ValueError("archive predicted payload bytes must be non-negative")
        if (self.cell_id is None) != (self.attempt is None):
            raise ValueError(
                "archive cell ID and attempt must be both present or absent"
            )
        if self.cell_id is not None:
            _require_text(self.cell_id, "archive cell ID")
            _require_positive_int(self.attempt, "archive attempt")


ArchiveStep = Literal["TRANSFER", "LOCAL_SHA_VERIFY", "REHYDRATE_VERIFY"]


@dataclass(frozen=True)
class ArchiveStepReceipt:
    step: ArchiveStep
    manifest_sha256: str
    evidence_sha256: str
    checked_file_count: int
    checked_bytes: int
    content_tree_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.step not in {"TRANSFER", "LOCAL_SHA_VERIFY", "REHYDRATE_VERIFY"}:
            raise ValueError("unsupported archive step")
        _require_sha256(self.manifest_sha256, "archive manifest SHA-256")
        _require_sha256(self.evidence_sha256, "archive step evidence SHA-256")
        for label, value in (
            ("checked file count", self.checked_file_count),
            ("checked bytes", self.checked_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"archive {label} must be a non-negative integer")
        if self.checked_file_count < 1:
            raise ValueError("archive steps must account at least one file")
        if self.step == "REHYDRATE_VERIFY":
            _require_sha256(self.content_tree_sha256, "rehydrated content-tree SHA-256")
        elif self.content_tree_sha256 is not None:
            raise ValueError("only rehydrate receipts carry a content-tree SHA-256")


@dataclass(frozen=True)
class ArchiveCallbacks:
    transfer: Callable[[ArchiveRequest, ArchiveStepReceipt | None], ArchiveStepReceipt]
    verify_local_sha: Callable[
        [ArchiveRequest, ArchiveStepReceipt | None], ArchiveStepReceipt
    ]
    verify_rehydrate: Callable[
        [ArchiveRequest, ArchiveStepReceipt | None], ArchiveStepReceipt
    ]


@dataclass(frozen=True)
class RemoteEvictionAuthorization:
    archive_id: str
    remote_payload_root: str
    manifest_sha256: str
    local_final_root: str
    local_sha_evidence_sha256: str
    rehydrate_evidence_sha256: str
    rehydrated_content_tree_sha256: str
    authorized_at_ns: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS operator_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_plan (
    node TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL UNIQUE CHECK (ordinal >= 0),
    stage TEXT NOT NULL,
    phase TEXT NOT NULL,
    expected_formula TEXT NOT NULL,
    known_expected_cells INTEGER CHECK (
        known_expected_cells IS NULL OR known_expected_cells >= 0
    ),
    estimated_remaining_gpu_hours REAL CHECK (
        estimated_remaining_gpu_hours IS NULL
        OR estimated_remaining_gpu_hours >= 0
    ),
    created_at_ns INTEGER NOT NULL CHECK (created_at_ns > 0),
    last_update_ns INTEGER NOT NULL CHECK (last_update_ns > 0),
    UNIQUE (stage, phase)
);

CREATE TABLE IF NOT EXISTS controller_nodes (
    node TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL UNIQUE CHECK (ordinal >= 0),
    state TEXT NOT NULL CHECK (
        state IN (
            'UNMATERIALIZED', 'MATERIALIZED', 'PLANNED', 'REDUCED', 'BLOCKED'
        )
    ),
    materialization_path TEXT,
    materialization_sha256 TEXT,
    node_materialization_path TEXT,
    node_materialization_sha256 TEXT,
    execution_source_path TEXT,
    execution_source_sha256 TEXT,
    prepared_launch_path TEXT,
    prepared_launch_sha256 TEXT,
    decision_path TEXT,
    decision_sha256 TEXT,
    completion_path TEXT,
    completion_sha256 TEXT,
    expected_cell_count INTEGER CHECK (
        expected_cell_count IS NULL OR expected_cell_count >= 0
    ),
    expected_cell_ids_sha256 TEXT,
    auxiliary_sources_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(auxiliary_sources_json)),
    blocker_reason TEXT,
    updated_at_ns INTEGER NOT NULL CHECK (updated_at_ns > 0),
    CHECK (
        (materialization_path IS NULL) = (materialization_sha256 IS NULL)
        AND (node_materialization_path IS NULL) =
            (node_materialization_sha256 IS NULL)
        AND (execution_source_path IS NULL) = (execution_source_sha256 IS NULL)
        AND (prepared_launch_path IS NULL) = (prepared_launch_sha256 IS NULL)
        AND (decision_path IS NULL) = (decision_sha256 IS NULL)
        AND (completion_path IS NULL) = (completion_sha256 IS NULL)
        AND (expected_cell_count IS NULL) = (expected_cell_ids_sha256 IS NULL)
    ),
    FOREIGN KEY (node) REFERENCES stage_plan(node)
);

CREATE TABLE IF NOT EXISTS controller_auxiliary_groups (
    group_id TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    node TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    launch_command_sha256 TEXT NOT NULL,
    output_directory TEXT NOT NULL,
    assigned_gpu_uuids_json TEXT NOT NULL CHECK (
        json_valid(assigned_gpu_uuids_json)
    ),
    status TEXT NOT NULL CHECK (
        status IN ('PENDING', 'RUNNING', 'COMPLETE', 'FAILED')
    ),
    pid INTEGER CHECK (pid IS NULL OR pid > 0),
    pgid INTEGER CHECK (pgid IS NULL OR pgid > 0),
    process_start_receipt_sha256 TEXT,
    heartbeat_at_ns INTEGER CHECK (heartbeat_at_ns IS NULL OR heartbeat_at_ns > 0),
    heartbeat_sequence INTEGER NOT NULL DEFAULT 0 CHECK (heartbeat_sequence >= 0),
    last_log_size_bytes INTEGER CHECK (
        last_log_size_bytes IS NULL OR last_log_size_bytes >= 0
    ),
    last_log_growth_ns INTEGER CHECK (
        last_log_growth_ns IS NULL OR last_log_growth_ns > 0
    ),
    gpu_observation_json TEXT CHECK (
        gpu_observation_json IS NULL OR json_valid(gpu_observation_json)
    ),
    termination_reason TEXT,
    termination_requested_at_ns INTEGER CHECK (
        termination_requested_at_ns IS NULL OR termination_requested_at_ns > 0
    ),
    term_sent_at_ns INTEGER CHECK (term_sent_at_ns IS NULL OR term_sent_at_ns > 0),
    kill_sent_at_ns INTEGER CHECK (kill_sent_at_ns IS NULL OR kill_sent_at_ns > 0),
    started_at_ns INTEGER CHECK (started_at_ns IS NULL OR started_at_ns > 0),
    finished_at_ns INTEGER CHECK (finished_at_ns IS NULL OR finished_at_ns > 0),
    publication_path TEXT,
    publication_sha256 TEXT,
    atomic_publication_sha256 TEXT,
    failure_code TEXT,
    failure_class TEXT,
    exclusion_reason TEXT,
    compute_gpu_seconds REAL NOT NULL DEFAULT 0 CHECK (compute_gpu_seconds >= 0),
    reserved_gpu_seconds REAL NOT NULL DEFAULT 0 CHECK (
        reserved_gpu_seconds >= 0
    ),
    billed_gpu_seconds REAL NOT NULL DEFAULT 0 CHECK (billed_gpu_seconds >= 0),
    adopted_at_ns INTEGER CHECK (adopted_at_ns IS NULL OR adopted_at_ns > 0),
    created_at_ns INTEGER NOT NULL CHECK (created_at_ns > 0),
    updated_at_ns INTEGER NOT NULL CHECK (updated_at_ns > 0),
    PRIMARY KEY (group_id, attempt),
    UNIQUE (node, attempt),
    UNIQUE (output_directory),
    CHECK (
        (pid IS NULL) = (pgid IS NULL)
        AND (publication_path IS NULL) = (publication_sha256 IS NULL)
        AND (publication_path IS NULL) = (atomic_publication_sha256 IS NULL)
        AND (termination_reason IS NULL) = (termination_requested_at_ns IS NULL)
    ),
    FOREIGN KEY (node) REFERENCES controller_nodes(node)
);

CREATE INDEX IF NOT EXISTS controller_auxiliary_group_order
ON controller_auxiliary_groups(node, attempt DESC);

CREATE TABLE IF NOT EXISTS controller_auxiliary_jobs (
    group_id TEXT NOT NULL,
    group_attempt INTEGER NOT NULL CHECK (group_attempt >= 1),
    job_id TEXT NOT NULL,
    job_attempt INTEGER NOT NULL CHECK (job_attempt >= 1),
    adoption_key TEXT NOT NULL,
    member_ordinal INTEGER NOT NULL CHECK (member_ordinal >= 0),
    scientific_axes_json TEXT NOT NULL CHECK (json_valid(scientific_axes_json)),
    identity_json TEXT NOT NULL CHECK (json_valid(identity_json)),
    command_sha256 TEXT NOT NULL,
    output_directory TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('PENDING', 'RUNNING', 'COMPLETE', 'FAILED')
    ),
    exit_code INTEGER,
    terminal_sha256 TEXT,
    junit_sha256 TEXT,
    raw_log_sha256 TEXT,
    evidence_files_json TEXT NOT NULL DEFAULT '{}' CHECK (
        json_valid(evidence_files_json)
    ),
    failure_class TEXT,
    failure_code TEXT,
    included_in_analysis INTEGER NOT NULL DEFAULT 0 CHECK (
        included_in_analysis IN (0, 1)
    ),
    exclusion_reason TEXT,
    evidence_started_at_ns INTEGER CHECK (
        evidence_started_at_ns IS NULL OR evidence_started_at_ns > 0
    ),
    evidence_finished_at_ns INTEGER CHECK (
        evidence_finished_at_ns IS NULL OR evidence_finished_at_ns > 0
    ),
    adopted_cell_id TEXT,
    adopted_cell_attempt INTEGER CHECK (
        adopted_cell_attempt IS NULL OR adopted_cell_attempt >= 1
    ),
    PRIMARY KEY (job_id, job_attempt),
    UNIQUE (group_id, group_attempt, member_ordinal),
    UNIQUE (group_id, group_attempt, adoption_key),
    UNIQUE (output_directory),
    CHECK (
        (adopted_cell_id IS NULL) = (adopted_cell_attempt IS NULL)
    ),
    FOREIGN KEY (group_id, group_attempt)
        REFERENCES controller_auxiliary_groups(group_id, attempt),
    FOREIGN KEY (adopted_cell_id, adopted_cell_attempt)
        REFERENCES cell_attempts(cell_id, attempt)
);

CREATE TABLE IF NOT EXISTS cell_attempts (
    cell_id TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    stage TEXT NOT NULL,
    phase TEXT NOT NULL,
    block_id TEXT,
    seed INTEGER,
    scientific_axes_json TEXT NOT NULL CHECK (json_valid(scientific_axes_json)),
    identity_json TEXT NOT NULL CHECK (json_valid(identity_json)),
    is_legacy_import INTEGER NOT NULL DEFAULT 0 CHECK (is_legacy_import IN (0, 1)),
    legacy_original_status TEXT,
    status TEXT NOT NULL CHECK (
        status IN (
            'PENDING', 'RUNNING', 'COMPLETE', 'FAILED', 'BLOCKED',
            'STALE_IDENTITY'
        )
    ),
    command_sha256 TEXT NOT NULL,
    scientific_command_sha256 TEXT,
    output_directory TEXT NOT NULL,
    assigned_gpu_uuids_json TEXT NOT NULL
        DEFAULT '[]' CHECK (json_valid(assigned_gpu_uuids_json)),
    pid INTEGER CHECK (pid IS NULL OR pid > 0),
    pgid INTEGER CHECK (pgid IS NULL OR pgid > 0),
    process_start_receipt_sha256 TEXT,
    started_at_ns INTEGER CHECK (started_at_ns IS NULL OR started_at_ns > 0),
    finished_at_ns INTEGER CHECK (finished_at_ns IS NULL OR finished_at_ns > 0),
    heartbeat_at_ns INTEGER CHECK (heartbeat_at_ns IS NULL OR heartbeat_at_ns > 0),
    heartbeat_sequence INTEGER NOT NULL DEFAULT 0 CHECK (heartbeat_sequence >= 0),
    last_log_size_bytes INTEGER CHECK (
        last_log_size_bytes IS NULL OR last_log_size_bytes >= 0
    ),
    last_log_growth_ns INTEGER CHECK (
        last_log_growth_ns IS NULL OR last_log_growth_ns > 0
    ),
    gpu_observation_json TEXT CHECK (
        gpu_observation_json IS NULL OR json_valid(gpu_observation_json)
    ),
    termination_reason TEXT,
    termination_requested_at_ns INTEGER CHECK (
        termination_requested_at_ns IS NULL OR termination_requested_at_ns > 0
    ),
    term_sent_at_ns INTEGER CHECK (term_sent_at_ns IS NULL OR term_sent_at_ns > 0),
    kill_sent_at_ns INTEGER CHECK (kill_sent_at_ns IS NULL OR kill_sent_at_ns > 0),
    exit_code INTEGER,
    terminal_sha256 TEXT,
    junit_sha256 TEXT,
    raw_log_sha256 TEXT,
    evidence_files_json TEXT NOT NULL
        DEFAULT '{}' CHECK (json_valid(evidence_files_json)),
    failure_code TEXT,
    retry_decision TEXT,
    included_in_analysis INTEGER NOT NULL DEFAULT 0
        CHECK (included_in_analysis IN (0, 1)),
    exclusion_reason TEXT,
    compute_gpu_seconds REAL NOT NULL DEFAULT 0 CHECK (compute_gpu_seconds >= 0),
    reserved_gpu_seconds REAL NOT NULL DEFAULT 0 CHECK (reserved_gpu_seconds >= 0),
    billed_gpu_seconds REAL NOT NULL DEFAULT 0 CHECK (billed_gpu_seconds >= 0),
    created_at_ns INTEGER NOT NULL CHECK (created_at_ns > 0),
    updated_at_ns INTEGER NOT NULL CHECK (updated_at_ns > 0),
    PRIMARY KEY (cell_id, attempt),
    CHECK ((termination_reason IS NULL) = (termination_requested_at_ns IS NULL)),
    FOREIGN KEY (stage, phase) REFERENCES stage_plan(stage, phase)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_running_attempt_per_cell
ON cell_attempts(cell_id) WHERE status = 'RUNNING';

CREATE UNIQUE INDEX IF NOT EXISTS unique_attempt_output_directory
ON cell_attempts(output_directory);

CREATE TABLE IF NOT EXISTS command_queue (
    cell_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    argv_json TEXT NOT NULL CHECK (json_valid(argv_json)),
    environment_json TEXT NOT NULL CHECK (json_valid(environment_json)),
    launch_compatibility_key TEXT NOT NULL,
    required_gpu_count INTEGER NOT NULL CHECK (required_gpu_count > 0),
    timing_class TEXT NOT NULL CHECK (
        timing_class IN (
            'HEADLINE', 'SAFE_AUXILIARY', 'EXCLUSIVE', 'PROFILER',
            'FAILURE', 'ARCHIVE'
        )
    ),
    predicted_high_water_bytes INTEGER NOT NULL CHECK (
        predicted_high_water_bytes >= 0
    ),
    monitored_path TEXT NOT NULL,
    log_path TEXT NOT NULL,
    expected_terminal_path TEXT NOT NULL,
    expected_junit_path TEXT NOT NULL,
    expected_raw_log_path TEXT NOT NULL,
    atomic_pointer_path TEXT NOT NULL,
    child_exit_receipt_path TEXT NOT NULL,
    paired_gpu_key TEXT,
    preferred_gpu_index INTEGER CHECK (
        preferred_gpu_index IS NULL OR preferred_gpu_index >= 0
    ),
    priority INTEGER NOT NULL,
    max_runtime_seconds INTEGER NOT NULL CHECK (max_runtime_seconds > 0),
    max_log_stall_seconds INTEGER NOT NULL CHECK (
        max_log_stall_seconds > 0
        AND max_log_stall_seconds <= max_runtime_seconds
    ),
    enqueued_at_ns INTEGER NOT NULL CHECK (enqueued_at_ns > 0),
    PRIMARY KEY (cell_id, attempt),
    FOREIGN KEY (cell_id, attempt) REFERENCES cell_attempts(cell_id, attempt)
);

CREATE INDEX IF NOT EXISTS command_queue_dispatch_order
ON command_queue(priority DESC, launch_compatibility_key, cell_id, attempt);

CREATE TABLE IF NOT EXISTS physical_attempt_groups (
    group_id TEXT PRIMARY KEY,
    leader_cell_id TEXT NOT NULL,
    leader_attempt INTEGER NOT NULL CHECK (leader_attempt >= 1),
    status TEXT NOT NULL CHECK (
        status IN ('PENDING', 'RUNNING', 'COMPLETE', 'FAILED')
    ),
    shared_evidence_sha256 TEXT,
    created_at_ns INTEGER NOT NULL CHECK (created_at_ns > 0),
    updated_at_ns INTEGER NOT NULL CHECK (updated_at_ns > 0),
    FOREIGN KEY (leader_cell_id, leader_attempt)
        REFERENCES cell_attempts(cell_id, attempt)
);

CREATE TABLE IF NOT EXISTS physical_attempt_group_members (
    group_id TEXT NOT NULL,
    cell_id TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    logical_kind TEXT NOT NULL CHECK (
        logical_kind IN ('compile', 'exactness', 'interference', 'serving')
    ),
    member_ordinal INTEGER NOT NULL CHECK (member_ordinal >= 0),
    PRIMARY KEY (group_id, cell_id, attempt),
    UNIQUE (cell_id, attempt),
    UNIQUE (group_id, member_ordinal),
    FOREIGN KEY (group_id) REFERENCES physical_attempt_groups(group_id),
    FOREIGN KEY (cell_id, attempt) REFERENCES cell_attempts(cell_id, attempt)
);

CREATE INDEX IF NOT EXISTS physical_attempt_group_leader
ON physical_attempt_groups(leader_cell_id, leader_attempt);

CREATE TABLE IF NOT EXISTS watchdog_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at_ns INTEGER NOT NULL CHECK (occurred_at_ns > 0),
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (
        severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')
    ),
    cell_id TEXT,
    attempt INTEGER,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    FOREIGN KEY (cell_id, attempt) REFERENCES cell_attempts(cell_id, attempt)
);

CREATE INDEX IF NOT EXISTS watchdog_event_lookup
ON watchdog_events(cell_id, attempt, event_type, occurred_at_ns DESC);

CREATE TABLE IF NOT EXISTS selection_decisions (
    decision_id TEXT PRIMARY KEY,
    occurred_at_ns INTEGER NOT NULL CHECK (occurred_at_ns > 0),
    stage TEXT NOT NULL,
    phase TEXT NOT NULL,
    decision_kind TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    decision_json TEXT NOT NULL CHECK (json_valid(decision_json)),
    FOREIGN KEY (stage, phase) REFERENCES stage_plan(stage, phase)
);

CREATE TABLE IF NOT EXISTS metrics_long (
    stage TEXT NOT NULL,
    phase TEXT NOT NULL,
    cell_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_kind TEXT NOT NULL CHECK (metric_kind IN ('headline', 'descriptive')),
    point_estimate REAL NOT NULL,
    ci_low REAL,
    ci_high REAL,
    independent_block_count INTEGER,
    request_count INTEGER,
    paired INTEGER CHECK (paired IS NULL OR paired IN (0, 1)),
    reducer_method TEXT NOT NULL,
    attributes_json TEXT NOT NULL CHECK (json_valid(attributes_json)),
    recorded_at_ns INTEGER NOT NULL CHECK (recorded_at_ns > 0),
    PRIMARY KEY (cell_id, attempt, metric_name, attributes_json),
    FOREIGN KEY (cell_id, attempt) REFERENCES cell_attempts(cell_id, attempt)
);

CREATE TABLE IF NOT EXISTS archive_checkpoints (
    archive_id TEXT PRIMARY KEY,
    safe_boundary TEXT NOT NULL,
    cell_id TEXT,
    attempt INTEGER,
    remote_payload_root TEXT NOT NULL,
    local_partial_root TEXT NOT NULL,
    local_final_root TEXT NOT NULL,
    remote_manifest_sha256 TEXT NOT NULL,
    predicted_payload_bytes INTEGER NOT NULL DEFAULT 0 CHECK (
        predicted_payload_bytes >= 0
    ),
    state TEXT NOT NULL CHECK (
        state IN (
            'REGISTERED', 'TRANSFERRED', 'LOCAL_SHA_VERIFIED',
            'REHYDRATE_VERIFIED', 'EVICTION_AUTHORIZED'
        )
    ),
    transfer_receipt_json TEXT CHECK (
        transfer_receipt_json IS NULL OR json_valid(transfer_receipt_json)
    ),
    local_sha_receipt_json TEXT CHECK (
        local_sha_receipt_json IS NULL OR json_valid(local_sha_receipt_json)
    ),
    rehydrate_receipt_json TEXT CHECK (
        rehydrate_receipt_json IS NULL OR json_valid(rehydrate_receipt_json)
    ),
    eviction_authorized_at_ns INTEGER CHECK (
        eviction_authorized_at_ns IS NULL OR eviction_authorized_at_ns > 0
    ),
    created_at_ns INTEGER NOT NULL CHECK (created_at_ns > 0),
    updated_at_ns INTEGER NOT NULL CHECK (updated_at_ns > 0),
    FOREIGN KEY (cell_id, attempt) REFERENCES cell_attempts(cell_id, attempt)
);

CREATE TABLE IF NOT EXISTS provider_runtime_samples (
    sample_id TEXT PRIMARY KEY,
    instance_uuid TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('running', 'shutdown')),
    observed_at_ns INTEGER NOT NULL CHECK (observed_at_ns > 0),
    provider_started_at_ns INTEGER NOT NULL CHECK (provider_started_at_ns > 0),
    provider_stopped_at_ns INTEGER CHECK (
        provider_stopped_at_ns IS NULL OR provider_stopped_at_ns > 0
    ),
    gpu_count INTEGER NOT NULL CHECK (gpu_count > 0),
    response_sha256 TEXT NOT NULL,
    CHECK (
        (state = 'running' AND provider_stopped_at_ns IS NULL)
        OR (state = 'shutdown' AND provider_stopped_at_ns IS NOT NULL)
    ),
    UNIQUE (instance_uuid, observed_at_ns, response_sha256)
);

CREATE INDEX IF NOT EXISTS provider_runtime_lifecycle
ON provider_runtime_samples(instance_uuid, provider_started_at_ns, observed_at_ns);
"""


def default_formal_stage_plan() -> tuple[StagePlanEntry, ...]:
    """Return the registered 21-node stage order with protocol formulas."""

    from lightcone_spec.experiments.formal_single_operator_stages import (
        FORMAL_SINGLE_OPERATOR_NODE_SPECS,
    )

    formula: dict[str, tuple[str, int | None]] = {
        "preflight": ("10", 10),
        "e3a": ("360", 360),
        "tts_cal": ("288", 288),
        "e1": ("68", 68),
        "e2_r0": ("4+n0; n0=105g", None),
        "e2_r1": ("4+n1; n1=max(ceil(n0/4),21)", None),
        "e2_r2": ("4+n2; n2=max(ceil(n1/4),21)", None),
        "e2_r3": ("4+n3; n3=max(ceil(n2/4),21)", None),
        "e4_screen": ("48", 48),
        "e4_local": ("96", 96),
        "e4_profiler": ("3", 3),
        "e3b_pilot": ("480*4", 1_920),
        "e3b_final": ("480*N_E3b", None),
        "e1a": ("58*2", 116),
        "e5_pilot": ("450*4", 1_800),
        "e5_final": ("450*N_E5+264", None),
        "e6_pilot": ("2+60*4", 242),
        "e6_final": ("60*N_E6", None),
        "e0_tuning": ("108+239*V", None),
        "e0_pilot": ("16*V*4", None),
        "e0_final": ("16*V*N_E0", None),
    }
    return tuple(
        StagePlanEntry(
            node=spec.node,
            ordinal=spec.ordinal,
            stage=spec.stage,
            phase=spec.phase,
            expected_formula=formula[spec.node][0],
            known_expected_cells=formula[spec.node][1],
        )
        for spec in FORMAL_SINGLE_OPERATOR_NODE_SPECS
    )


class SingletonOperatorLock:
    """Non-blocking process singleton implemented with ``flock``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._descriptor: int | None = None

    def acquire(self, *, blocking: bool = False) -> None:
        if self._descriptor is not None:
            raise ExperimentOperatorError(
                "operator lock is already held by this object"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as error:
            holder = os.read(descriptor, 4096).decode("utf-8", errors="replace").strip()
            os.close(descriptor)
            suffix = f": {holder}" if holder else ""
            raise OperatorAlreadyRunningError(
                f"another formal experiment operator holds {self.path}{suffix}"
            ) from error
        payload = _canonical_json(
            {"pid": os.getpid(), "acquired_at_ns": time.time_ns()}
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, payload)
        os.fsync(descriptor)
        self._descriptor = descriptor

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *_error: object) -> None:
        self.release()


class ExperimentOperatorStore:
    """SQLite-backed authority for stage and cell-attempt lifecycle state."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.path = Path(path)
        existed = self.path.exists()
        if existed and self.path.is_symlink():
            raise ValueError("operator database cannot be a symlink")
        if not existed and run_id is None:
            raise ValueError("run_id is required for a new operator database")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock_ns = clock_ns
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            user_version = int(
                self._connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if user_version not in {
                0,
                *_PREVIOUS_SCHEMA_VERSIONS,
                _SCHEMA_VERSION,
            }:
                raise ExperimentOperatorError(
                    f"unsupported SQLite user_version {user_version}"
                )
            if existed and user_version == 0:
                metadata_table = self._connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = 'operator_meta'
                    """
                ).fetchone()
                if metadata_table is None:
                    raise ExperimentOperatorError(
                        "refusing to adopt an unrecognized SQLite database"
                    )
            journal_mode = self._connection.execute(
                "PRAGMA journal_mode=WAL"
            ).fetchone()[0]
            if str(journal_mode).lower() != "wal":
                raise ExperimentOperatorError(
                    "operator database requires SQLite WAL mode"
                )
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.executescript(_SCHEMA)
            self._migrate_schema_if_needed(user_version)
            with self._transaction():
                existing_version = self._metadata_value("schema_version")
                if existing_version is None or existing_version in {
                    str(value) for value in _PREVIOUS_SCHEMA_VERSIONS
                }:
                    self._set_metadata("schema_version", str(_SCHEMA_VERSION))
                elif existing_version != str(_SCHEMA_VERSION):
                    raise ExperimentOperatorError(
                        f"unsupported operator schema version {existing_version}"
                    )
                existing_run_id = self._metadata_value("run_id")
                if existing_run_id is None:
                    if run_id is None:
                        raise ValueError(
                            "run_id is required for a new operator database"
                        )
                    _require_text(run_id, "run ID")
                    self._set_metadata("run_id", run_id)
                    self._set_metadata("created_at_ns", str(self._now()))
                elif run_id is not None and existing_run_id != run_id:
                    raise ExperimentOperatorError(
                        "operator database belongs to run "
                        f"{existing_run_id!r}, not {run_id!r}"
                    )
            self._connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        except BaseException:
            self._connection.close()
            raise

    @property
    def run_id(self) -> str:
        value = self._metadata_value("run_id")
        if value is None:
            raise ExperimentOperatorError("operator database has no run identity")
        return value

    @property
    def journal_mode(self) -> str:
        return str(self._connection.execute("PRAGMA journal_mode").fetchone()[0])

    @property
    def synchronous_mode(self) -> int:
        return int(self._connection.execute("PRAGMA synchronous").fetchone()[0])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def _migrate_schema_if_needed(self, user_version: int) -> None:
        """Add backward-compatible operator columns from earlier local WALs.

        The v03 operator had not yet been deployed when schema 3 was added, but
        accepting a schema-2 WAL here keeps CPU fixtures and interrupted local
        dry runs recoverable.  Existing commands receive the exact code-owned
        limits that schema 2 implicitly used; no scientific field is changed.
        """

        columns = {
            str(row["name"])
            for row in self._connection.execute(
                "PRAGMA table_info(command_queue)"
            ).fetchall()
        }
        additions = (
            (
                "max_runtime_seconds",
                _DEFAULT_COMMAND_MAX_RUNTIME_SECONDS,
            ),
            (
                "max_log_stall_seconds",
                _DEFAULT_COMMAND_MAX_LOG_STALL_SECONDS,
            ),
        )
        missing = tuple(
            (name, default) for name, default in additions if name not in columns
        )
        if missing and user_version not in {0, *_PREVIOUS_SCHEMA_VERSIONS}:
            raise ExperimentOperatorError(
                "operator command watchdog columns are missing from current schema"
            )
        for name, default in missing:
            self._connection.execute(
                f"ALTER TABLE command_queue ADD COLUMN {name} "
                f"INTEGER NOT NULL DEFAULT {default} CHECK ({name} > 0)"
            )
        archive_columns = {
            str(row["name"])
            for row in self._connection.execute(
                "PRAGMA table_info(archive_checkpoints)"
            ).fetchall()
        }
        if "predicted_payload_bytes" not in archive_columns:
            self._connection.execute(
                "ALTER TABLE archive_checkpoints ADD COLUMN "
                "predicted_payload_bytes INTEGER NOT NULL DEFAULT 0 "
                "CHECK (predicted_payload_bytes >= 0)"
            )

        migration_columns: dict[str, tuple[tuple[str, str], ...]] = {
            "cell_attempts": (
                ("scientific_command_sha256", "TEXT"),
                ("process_start_receipt_sha256", "TEXT"),
                ("termination_reason", "TEXT"),
                (
                    "termination_requested_at_ns",
                    (
                        "INTEGER CHECK (termination_requested_at_ns IS NULL "
                        "OR termination_requested_at_ns > 0)"
                    ),
                ),
                (
                    "term_sent_at_ns",
                    "INTEGER CHECK (term_sent_at_ns IS NULL OR term_sent_at_ns > 0)",
                ),
                (
                    "kill_sent_at_ns",
                    "INTEGER CHECK (kill_sent_at_ns IS NULL OR kill_sent_at_ns > 0)",
                ),
            ),
            "controller_auxiliary_groups": (
                ("process_start_receipt_sha256", "TEXT"),
                (
                    "heartbeat_at_ns",
                    "INTEGER CHECK (heartbeat_at_ns IS NULL OR heartbeat_at_ns > 0)",
                ),
                (
                    "heartbeat_sequence",
                    "INTEGER NOT NULL DEFAULT 0 CHECK (heartbeat_sequence >= 0)",
                ),
                (
                    "last_log_size_bytes",
                    (
                        "INTEGER CHECK (last_log_size_bytes IS NULL "
                        "OR last_log_size_bytes >= 0)"
                    ),
                ),
                (
                    "last_log_growth_ns",
                    (
                        "INTEGER CHECK (last_log_growth_ns IS NULL "
                        "OR last_log_growth_ns > 0)"
                    ),
                ),
                (
                    "gpu_observation_json",
                    (
                        "TEXT CHECK (gpu_observation_json IS NULL "
                        "OR json_valid(gpu_observation_json))"
                    ),
                ),
                ("termination_reason", "TEXT"),
                (
                    "termination_requested_at_ns",
                    (
                        "INTEGER CHECK (termination_requested_at_ns IS NULL "
                        "OR termination_requested_at_ns > 0)"
                    ),
                ),
                (
                    "term_sent_at_ns",
                    "INTEGER CHECK (term_sent_at_ns IS NULL OR term_sent_at_ns > 0)",
                ),
                (
                    "kill_sent_at_ns",
                    "INTEGER CHECK (kill_sent_at_ns IS NULL OR kill_sent_at_ns > 0)",
                ),
                ("failure_class", "TEXT"),
            ),
            "controller_auxiliary_jobs": (("failure_class", "TEXT"),),
        }
        for table_name, table_additions in migration_columns.items():
            existing_columns = {
                str(row["name"])
                for row in self._connection.execute(
                    f"PRAGMA table_info({table_name})"
                ).fetchall()
            }
            for column_name, declaration in table_additions:
                if column_name in existing_columns:
                    continue
                self._connection.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {declaration}"
                )
        self._migrate_physical_attempt_group_members_for_serving()

    def _migrate_physical_attempt_group_members_for_serving(self) -> None:
        """Rebuild the pre-serving schema-7 member table without losing rows.

        The first schema-7 draft admitted only the exact-ten logical kinds.  A
        schema version alone therefore cannot distinguish an interrupted WAL
        created by that draft from the current schema.  Inspect the canonical
        table DDL and, when necessary, perform one transactional copy-table
        migration.  Groups, commands, attempts, and watchdog events are not
        rewritten.
        """

        row = self._connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table' AND name = 'physical_attempt_group_members'
            """
        ).fetchone()
        if row is None or not isinstance(row["sql"], str):
            raise ExperimentOperatorError(
                "physical-attempt-group member schema is unavailable"
            )
        table_sql = str(row["sql"])
        logical_kind_check = re.search(
            r"logical_kind\s+IN\s*\([^)]*'serving'",
            table_sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if logical_kind_check is not None:
            return

        migration_table = "physical_attempt_group_members__serving_migration"
        if (
            self._connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ?", (migration_table,)
            ).fetchone()
            is not None
        ):
            raise ExperimentOperatorError(
                "physical-attempt-group member migration table already exists"
            )
        with self._transaction():
            invalid = self._connection.execute(
                """
                SELECT logical_kind FROM physical_attempt_group_members
                WHERE logical_kind NOT IN (
                    'compile', 'exactness', 'interference', 'serving'
                )
                LIMIT 1
                """
            ).fetchone()
            if invalid is not None:
                raise ExperimentOperatorError(
                    "physical-attempt-group member has an unsupported logical kind"
                )
            before_count = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM physical_attempt_group_members"
                ).fetchone()[0]
            )
            self._connection.execute(
                f"""
                CREATE TABLE {migration_table} (
                    group_id TEXT NOT NULL,
                    cell_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL CHECK (attempt >= 1),
                    logical_kind TEXT NOT NULL CHECK (
                        logical_kind IN (
                            'compile', 'exactness', 'interference', 'serving'
                        )
                    ),
                    member_ordinal INTEGER NOT NULL CHECK (member_ordinal >= 0),
                    PRIMARY KEY (group_id, cell_id, attempt),
                    UNIQUE (cell_id, attempt),
                    UNIQUE (group_id, member_ordinal),
                    FOREIGN KEY (group_id)
                        REFERENCES physical_attempt_groups(group_id),
                    FOREIGN KEY (cell_id, attempt)
                        REFERENCES cell_attempts(cell_id, attempt)
                )
                """
            )
            self._connection.execute(
                f"""
                INSERT INTO {migration_table} (
                    group_id, cell_id, attempt, logical_kind, member_ordinal
                )
                SELECT group_id, cell_id, attempt, logical_kind, member_ordinal
                FROM physical_attempt_group_members
                ORDER BY group_id, member_ordinal
                """
            )
            after_count = int(
                self._connection.execute(
                    f"SELECT COUNT(*) FROM {migration_table}"
                ).fetchone()[0]
            )
            if after_count != before_count:
                raise ExperimentOperatorError(
                    "physical-attempt-group member migration lost rows"
                )
            self._connection.execute("DROP TABLE physical_attempt_group_members")
            self._connection.execute(
                f"ALTER TABLE {migration_table} "
                "RENAME TO physical_attempt_group_members"
            )
            foreign_key_errors = self._connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
            if foreign_key_errors:
                raise ExperimentOperatorError(
                    "physical-attempt-group member migration violated foreign keys"
                )

    def initialize_stage_plan(self, entries: Sequence[StagePlanEntry]) -> None:
        if not entries:
            raise ValueError("stage plan cannot be empty")
        if len({entry.node for entry in entries}) != len(entries):
            raise ValueError("stage plan contains duplicate nodes")
        if len({(entry.stage, entry.phase) for entry in entries}) != len(entries):
            raise ValueError("stage plan contains duplicate stage/phase pairs")
        if tuple(sorted(entry.ordinal for entry in entries)) != tuple(
            range(len(entries))
        ):
            raise ValueError("stage plan ordinals must be contiguous from zero")
        now = self._now()
        with self._transaction():
            count = int(
                self._connection.execute("SELECT COUNT(*) FROM stage_plan").fetchone()[
                    0
                ]
            )
            if count:
                existing = self._stage_plan_rows()
                requested = tuple(
                    (
                        entry.node,
                        entry.ordinal,
                        entry.stage,
                        entry.phase,
                        entry.expected_formula,
                        entry.known_expected_cells,
                    )
                    for entry in sorted(entries, key=lambda item: item.ordinal)
                )
                current = tuple(
                    (
                        row["node"],
                        row["ordinal"],
                        row["stage"],
                        row["phase"],
                        row["expected_formula"],
                        row["known_expected_cells"],
                    )
                    for row in existing
                )
                if current != requested:
                    raise ExperimentOperatorError(
                        "refusing to replace an initialized stage plan"
                    )
                self._connection.executemany(
                    """
                    INSERT OR IGNORE INTO controller_nodes (
                        node, ordinal, state, updated_at_ns
                    ) VALUES (?, ?, 'UNMATERIALIZED', ?)
                    """,
                    ((entry.node, entry.ordinal, now) for entry in entries),
                )
                return
            self._connection.executemany(
                """
                INSERT INTO stage_plan (
                    node, ordinal, stage, phase, expected_formula,
                    known_expected_cells, estimated_remaining_gpu_hours,
                    created_at_ns, last_update_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        entry.node,
                        entry.ordinal,
                        entry.stage,
                        entry.phase,
                        entry.expected_formula,
                        entry.known_expected_cells,
                        entry.estimated_remaining_gpu_hours,
                        now,
                        now,
                    )
                    for entry in entries
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO controller_nodes (
                    node, ordinal, state, updated_at_ns
                ) VALUES (?, ?, 'UNMATERIALIZED', ?)
                """,
                ((entry.node, entry.ordinal, now) for entry in entries),
            )

    def update_stage_expectation(
        self,
        *,
        node: str,
        expected_formula: str,
        known_expected_cells: int,
        estimated_remaining_gpu_hours: float | None,
    ) -> None:
        _require_text(node, "node")
        _require_text(expected_formula, "expected formula")
        if (
            isinstance(known_expected_cells, bool)
            or not isinstance(known_expected_cells, int)
            or known_expected_cells < 0
        ):
            raise ValueError("known expected cells must be non-negative")
        _require_nonnegative_finite_or_none(
            estimated_remaining_gpu_hours,
            "estimated remaining GPU-hours",
        )
        with self._transaction():
            plan = self._connection.execute(
                "SELECT stage, phase FROM stage_plan WHERE node = ?", (node,)
            ).fetchone()
            if plan is None:
                raise KeyError(f"unknown stage-plan node {node!r}")
            materialized = int(
                self._connection.execute(
                    """
                    SELECT COUNT(DISTINCT cell_id) FROM cell_attempts
                    WHERE stage = ? AND phase = ? AND is_legacy_import = 0
                    """,
                    (plan["stage"], plan["phase"]),
                ).fetchone()[0]
            )
            if known_expected_cells < materialized:
                raise ExperimentOperatorError(
                    "known expected cells cannot be below materialized coverage"
                )
            cursor = self._connection.execute(
                """
                UPDATE stage_plan
                SET expected_formula = ?, known_expected_cells = ?,
                    estimated_remaining_gpu_hours = ?, last_update_ns = ?
                WHERE node = ?
                """,
                (
                    expected_formula,
                    known_expected_cells,
                    estimated_remaining_gpu_hours,
                    self._now(),
                    node,
                ),
            )
            if cursor.rowcount != 1:
                raise AssertionError("stage-plan row disappeared during transaction")

    def record_controller_materialization(
        self,
        *,
        node: str,
        materialization: ControllerArtifactBinding,
        node_materialization: ControllerArtifactBinding,
        expected_cell_ids: Sequence[str],
        auxiliary_sources: Mapping[str, ControllerArtifactBinding] | None = None,
        recorded_at_ns: int | None = None,
    ) -> None:
        """Bind the next real materialization before any cell is registered."""

        _require_text(node, "controller node")
        if (
            type(materialization) is not ControllerArtifactBinding
            or type(node_materialization) is not ControllerArtifactBinding
        ):
            raise TypeError("controller materialization requires exact bindings")
        materialization = self._reopen_controller_binding(materialization)
        node_materialization = self._reopen_controller_binding(node_materialization)
        if materialization.absolute_path == node_materialization.absolute_path:
            raise ValueError("controller materialization artifacts must be distinct")
        cell_ids = tuple(expected_cell_ids)
        if any(
            type(value) is not str or not value for value in cell_ids
        ) or cell_ids != tuple(sorted(set(cell_ids))):
            raise ValueError("controller expected cell IDs must be uniquely sorted")
        expected_sha256 = hashlib.sha256(
            _canonical_json(cell_ids).encode("utf-8")
        ).hexdigest()
        auxiliary: dict[str, dict[str, str]] = {}
        for kind, binding in sorted(dict(auxiliary_sources or {}).items()):
            _require_text(kind, "controller auxiliary source kind")
            if type(binding) is not ControllerArtifactBinding:
                raise TypeError("controller auxiliary source requires an exact binding")
            auxiliary[kind] = asdict(self._reopen_controller_binding(binding))
        recorded = self._validated_time(recorded_at_ns)
        with self._transaction():
            row = self._require_controller_node(node)
            if row["state"] not in {"UNMATERIALIZED", "BLOCKED"} or (
                row["materialization_path"] is not None
            ):
                raise AttemptTransitionError(
                    "controller node is already materially initialized"
                )
            incomplete_prior = self._connection.execute(
                """
                SELECT node FROM controller_nodes
                WHERE ordinal < ? AND state != 'REDUCED'
                ORDER BY ordinal LIMIT 1
                """,
                (row["ordinal"],),
            ).fetchone()
            if incomplete_prior is not None:
                raise AttemptTransitionError(
                    "controller cannot materialize ahead of its predecessor"
                )
            plan = self._connection.execute(
                "SELECT known_expected_cells FROM stage_plan WHERE node = ?",
                (node,),
            ).fetchone()
            known = plan["known_expected_cells"]
            if known is not None and int(known) != len(cell_ids):
                raise ExperimentOperatorError(
                    "controller materialization cell count differs from stage plan"
                )
            if known is None:
                self._connection.execute(
                    """
                    UPDATE stage_plan
                    SET known_expected_cells = ?, last_update_ns = ?
                    WHERE node = ?
                    """,
                    (len(cell_ids), recorded, node),
                )
            self._connection.execute(
                """
                UPDATE controller_nodes
                SET state = 'MATERIALIZED', materialization_path = ?,
                    materialization_sha256 = ?, node_materialization_path = ?,
                    node_materialization_sha256 = ?, expected_cell_count = ?,
                    expected_cell_ids_sha256 = ?, auxiliary_sources_json = ?,
                    blocker_reason = NULL, updated_at_ns = ?
                WHERE node = ?
                """,
                (
                    materialization.absolute_path,
                    materialization.sha256,
                    node_materialization.absolute_path,
                    node_materialization.sha256,
                    len(cell_ids),
                    expected_sha256,
                    _canonical_json(auxiliary),
                    recorded,
                    node,
                ),
            )
            self._insert_event(
                event_type="DAG_NODE_MATERIALIZED",
                severity="INFO",
                cell_id=None,
                attempt=None,
                payload={
                    "node": node,
                    "expected_cell_count": len(cell_ids),
                    "expected_cell_ids_sha256": expected_sha256,
                    "materialization_sha256": materialization.sha256,
                    "node_materialization_sha256": node_materialization.sha256,
                    "auxiliary_sources": auxiliary,
                },
                occurred_at_ns=recorded,
            )

    def record_controller_execution_plan(
        self,
        *,
        node: str,
        execution_source: ControllerArtifactBinding,
        prepared_launch: ControllerArtifactBinding | None = None,
        recorded_at_ns: int | None = None,
    ) -> None:
        """Bind the source-owned execution input set for one materialized node."""

        _require_text(node, "controller node")
        if type(execution_source) is not ControllerArtifactBinding:
            raise TypeError("controller execution source requires an exact binding")
        execution_source = self._reopen_controller_binding(execution_source)
        if prepared_launch is not None:
            if type(prepared_launch) is not ControllerArtifactBinding:
                raise TypeError("controller prepared launch requires an exact binding")
            prepared_launch = self._reopen_controller_binding(prepared_launch)
        recorded = self._validated_time(recorded_at_ns)
        with self._transaction():
            row = self._require_controller_node(node)
            if row["state"] != "MATERIALIZED":
                raise AttemptTransitionError(
                    "controller execution planning requires MATERIALIZED state"
                )
            self._connection.execute(
                """
                UPDATE controller_nodes
                SET state = 'PLANNED', execution_source_path = ?,
                    execution_source_sha256 = ?, prepared_launch_path = ?,
                    prepared_launch_sha256 = ?, blocker_reason = NULL,
                    updated_at_ns = ? WHERE node = ?
                """,
                (
                    execution_source.absolute_path,
                    execution_source.sha256,
                    None if prepared_launch is None else prepared_launch.absolute_path,
                    None if prepared_launch is None else prepared_launch.sha256,
                    recorded,
                    node,
                ),
            )
            self._insert_event(
                event_type="DAG_NODE_EXECUTION_PLANNED",
                severity="INFO",
                cell_id=None,
                attempt=None,
                payload={
                    "node": node,
                    "execution_source_sha256": execution_source.sha256,
                    "prepared_launch_sha256": (
                        None if prepared_launch is None else prepared_launch.sha256
                    ),
                },
                occurred_at_ns=recorded,
            )

    def record_controller_reduction(
        self,
        *,
        node: str,
        decision: ControllerArtifactBinding,
        completion: ControllerArtifactBinding,
        recorded_at_ns: int | None = None,
    ) -> None:
        """Seal a node only after exact latest-attempt coverage is COMPLETE."""

        _require_text(node, "controller node")
        if (
            type(decision) is not ControllerArtifactBinding
            or type(completion) is not ControllerArtifactBinding
        ):
            raise TypeError("controller reduction requires exact bindings")
        decision = self._reopen_controller_binding(decision)
        completion = self._reopen_controller_binding(completion)
        if decision.absolute_path == completion.absolute_path:
            raise ValueError("controller decision and completion must be distinct")
        recorded = self._validated_time(recorded_at_ns)
        with self._transaction():
            row = self._require_controller_node(node)
            if row["state"] != "PLANNED":
                raise AttemptTransitionError(
                    "controller reduction requires a PLANNED node"
                )
            counts = self._connection.execute(
                """
                WITH latest AS (
                    SELECT a.* FROM cell_attempts AS a
                    JOIN (
                        SELECT cell_id, MAX(attempt) AS attempt
                        FROM cell_attempts WHERE is_legacy_import = 0
                        GROUP BY cell_id
                    ) AS chosen
                    ON a.cell_id = chosen.cell_id AND a.attempt = chosen.attempt
                )
                SELECT COUNT(*) AS total,
                    SUM(CASE WHEN status = 'COMPLETE' THEN 1 ELSE 0 END) AS complete
                FROM latest
                WHERE stage = (SELECT stage FROM stage_plan WHERE node = ?)
                  AND phase = (SELECT phase FROM stage_plan WHERE node = ?)
                """,
                (node, node),
            ).fetchone()
            total = int(counts["total"])
            complete = int(counts["complete"] or 0)
            expected = int(row["expected_cell_count"])
            if total != expected or complete != expected:
                raise AttemptTransitionError(
                    "controller reduction lacks exact COMPLETE attempt coverage"
                )
            self._connection.execute(
                """
                UPDATE controller_nodes
                SET state = 'REDUCED', decision_path = ?, decision_sha256 = ?,
                    completion_path = ?, completion_sha256 = ?,
                    blocker_reason = NULL, updated_at_ns = ? WHERE node = ?
                """,
                (
                    decision.absolute_path,
                    decision.sha256,
                    completion.absolute_path,
                    completion.sha256,
                    recorded,
                    node,
                ),
            )
            self._insert_event(
                event_type="DAG_NODE_REDUCED",
                severity="INFO",
                cell_id=None,
                attempt=None,
                payload={
                    "node": node,
                    "decision_sha256": decision.sha256,
                    "completion_sha256": completion.sha256,
                },
                occurred_at_ns=recorded,
            )

    def mark_controller_blocked(
        self,
        *,
        node: str,
        reason: str,
        recorded_at_ns: int | None = None,
    ) -> None:
        _require_text(node, "controller node")
        _require_text(reason, "controller blocker reason")
        recorded = self._validated_time(recorded_at_ns)
        with self._transaction():
            row = self._require_controller_node(node)
            if row["state"] == "REDUCED":
                raise AttemptTransitionError("reduced controller node cannot block")
            self._connection.execute(
                """
                UPDATE controller_nodes SET state = 'BLOCKED', blocker_reason = ?,
                    updated_at_ns = ? WHERE node = ?
                """,
                (reason, recorded, node),
            )
            self._connection.execute(
                "UPDATE stage_plan SET last_update_ns = ? WHERE node = ?",
                (recorded, node),
            )
            self._insert_event(
                event_type="DAG_NODE_BLOCKED",
                severity="CRITICAL",
                cell_id=None,
                attempt=None,
                payload={"node": node, "reason": reason},
                occurred_at_ns=recorded,
            )

    def resume_controller_node(
        self,
        *,
        node: str,
        reason: str,
        recorded_at_ns: int | None = None,
    ) -> None:
        """Explicitly clear a blocker without changing any bound artifact."""

        _require_text(node, "controller node")
        _require_text(reason, "controller resume reason")
        recorded = self._validated_time(recorded_at_ns)
        with self._transaction():
            row = self._require_controller_node(node)
            if row["state"] != "BLOCKED":
                raise AttemptTransitionError("controller node is not BLOCKED")
            if row["execution_source_path"] is not None:
                restored = "PLANNED"
            elif row["materialization_path"] is not None:
                restored = "MATERIALIZED"
            else:
                restored = "UNMATERIALIZED"
            self._connection.execute(
                """
                UPDATE controller_nodes SET state = ?, blocker_reason = NULL,
                    updated_at_ns = ? WHERE node = ?
                """,
                (restored, recorded, node),
            )
            self._insert_event(
                event_type="DAG_NODE_RESUMED",
                severity="WARNING",
                cell_id=None,
                attempt=None,
                payload={"node": node, "reason": reason, "restored_state": restored},
                occurred_at_ns=recorded,
            )

    def controller_node(self, node: str) -> dict[str, Any]:
        return _decoded_controller_node(self._require_controller_node(node))

    def controller_nodes(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _decoded_controller_node(row)
            for row in self._connection.execute(
                "SELECT * FROM controller_nodes ORDER BY ordinal"
            ).fetchall()
        )

    def register_controller_auxiliary_group(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        *,
        registered_at_ns: int | None = None,
    ) -> bool:
        """Persist a real pre-materialization campaign without creating cells.

        ``True`` means this call created a new durable attempt.  Rebuilding the
        byte-identical current attempt is idempotent; a new attempt is accepted
        only after a retained FAILED attempt.
        """

        if type(spec) is not AuxiliaryPhysicalGroupSpec:
            raise TypeError("controller auxiliary registration requires an exact spec")
        registered = self._validated_time(registered_at_ns)
        with self._transaction():
            controller = self._require_controller_node(spec.node)
            if controller["state"] != "UNMATERIALIZED":
                raise AttemptTransitionError(
                    "auxiliary registration requires an UNMATERIALIZED node"
                )
            prior = self._connection.execute(
                """
                SELECT node FROM controller_nodes
                WHERE ordinal < ? AND state != 'REDUCED'
                ORDER BY ordinal LIMIT 1
                """,
                (controller["ordinal"],),
            ).fetchone()
            if prior is not None:
                raise AttemptTransitionError(
                    "controller auxiliary cannot run ahead of its predecessor"
                )
            existing = self._connection.execute(
                """
                SELECT * FROM controller_auxiliary_groups
                WHERE group_id = ? AND attempt = ?
                """,
                (spec.group_id, spec.attempt),
            ).fetchone()
            if existing is not None:
                self._require_controller_auxiliary_spec_matches_locked(existing, spec)
                return False
            latest = self._connection.execute(
                """
                SELECT * FROM controller_auxiliary_groups
                WHERE node = ? ORDER BY attempt DESC LIMIT 1
                """,
                (spec.node,),
            ).fetchone()
            if latest is None:
                if spec.attempt != 1:
                    raise AttemptTransitionError(
                        "first controller auxiliary attempt must be 1"
                    )
            else:
                if spec.group_id != latest["group_id"]:
                    raise AttemptTransitionError(
                        "controller auxiliary retry changed its group identity"
                    )
                if spec.attempt != int(latest["attempt"]) + 1:
                    raise AttemptTransitionError(
                        "controller auxiliary retry attempt is not contiguous"
                    )
                if latest["status"] != "FAILED":
                    raise AttemptTransitionError(
                        "controller auxiliary retry requires a retained failure"
                    )
                if latest["failure_class"] != "INFRASTRUCTURE":
                    raise AttemptTransitionError(
                        "controller auxiliary retry is infrastructure-only"
                    )
                if int(latest["attempt"]) >= 3:
                    raise AttemptTransitionError(
                        "controller auxiliary infrastructure retry limit is exhausted"
                    )
            try:
                self._connection.execute(
                    """
                    INSERT INTO controller_auxiliary_groups (
                        group_id, attempt, node, source_kind,
                        launch_command_sha256, output_directory,
                        assigned_gpu_uuids_json, status,
                        created_at_ns, updated_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
                    """,
                    (
                        spec.group_id,
                        spec.attempt,
                        spec.node,
                        spec.source_kind,
                        spec.launch_command_sha256,
                        spec.output_directory,
                        _canonical_json(spec.assigned_gpu_uuids),
                        registered,
                        registered,
                    ),
                )
                self._connection.executemany(
                    """
                    INSERT INTO controller_auxiliary_jobs (
                        group_id, group_attempt, job_id, job_attempt,
                        adoption_key, member_ordinal, scientific_axes_json,
                        identity_json, command_sha256, output_directory, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                    """,
                    (
                        (
                            spec.group_id,
                            spec.attempt,
                            job.job_id,
                            job.attempt,
                            job.adoption_key,
                            ordinal,
                            _canonical_json(dict(job.scientific_axes)),
                            _canonical_json(dict(job.identity)),
                            job.command_sha256,
                            job.output_directory,
                        )
                        for ordinal, job in enumerate(spec.jobs)
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ExperimentOperatorError(
                    "controller auxiliary identity or output directory is registered"
                ) from error
            stage = self._connection.execute(
                "SELECT stage, phase FROM stage_plan WHERE node = ?",
                (spec.node,),
            ).fetchone()
            self._touch_stage(stage["stage"], stage["phase"], registered)
            self._insert_event(
                event_type="DAG_AUXILIARY_REGISTERED",
                severity="INFO",
                cell_id=None,
                attempt=None,
                payload={
                    "node": spec.node,
                    "group_id": spec.group_id,
                    "attempt": spec.attempt,
                    "source_kind": spec.source_kind,
                    "job_count": len(spec.jobs),
                    "assigned_gpu_uuids": list(spec.assigned_gpu_uuids),
                },
                occurred_at_ns=registered,
            )
        return True

    def latest_controller_auxiliary_group(self, node: str) -> dict[str, Any] | None:
        self._require_controller_node(node)
        row = self._connection.execute(
            """
            SELECT * FROM controller_auxiliary_groups
            WHERE node = ? ORDER BY attempt DESC LIMIT 1
            """,
            (node,),
        ).fetchone()
        return None if row is None else self._decoded_controller_auxiliary_group(row)

    def controller_auxiliary_groups(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self._decoded_controller_auxiliary_group(row)
            for row in self._connection.execute(
                """
                SELECT * FROM controller_auxiliary_groups
                ORDER BY (SELECT ordinal FROM controller_nodes
                          WHERE node = controller_auxiliary_groups.node), attempt
                """
            ).fetchall()
        )

    def controller_auxiliary_adopted_cell_ids(self, node: str) -> tuple[str, ...]:
        self._require_controller_node(node)
        rows = self._connection.execute(
            """
            SELECT j.adopted_cell_id FROM controller_auxiliary_jobs AS j
            JOIN controller_auxiliary_groups AS g
              ON g.group_id = j.group_id AND g.attempt = j.group_attempt
            WHERE g.node = ? AND g.adopted_at_ns IS NOT NULL
            ORDER BY j.adopted_cell_id
            """,
            (node,),
        ).fetchall()
        cell_ids = tuple(str(row["adopted_cell_id"]) for row in rows)
        if cell_ids != tuple(sorted(set(cell_ids))):
            raise ExperimentOperatorError(
                "controller auxiliary adopted-cell coverage differs"
            )
        return cell_ids

    def start_controller_auxiliary_group_with_launcher(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        *,
        launcher: Callable[[], SpawnedProcess],
        started_at_ns: int | None = None,
    ) -> SpawnedProcess:
        """Commit all auxiliary jobs RUNNING before invoking the launcher."""

        if type(spec) is not AuxiliaryPhysicalGroupSpec:
            raise TypeError("auxiliary launch requires an exact group spec")
        started = self._validated_time(started_at_ns)
        with self._transaction():
            row = self._require_controller_auxiliary_group(
                spec.group_id,
                spec.attempt,
            )
            self._require_controller_auxiliary_spec_matches_locked(row, spec)
            if row["status"] != "PENDING":
                raise AttemptTransitionError(
                    "controller auxiliary launch requires PENDING status"
                )
            self._connection.execute(
                """
                UPDATE controller_auxiliary_groups
                SET status = 'RUNNING', started_at_ns = ?, updated_at_ns = ?
                WHERE group_id = ? AND attempt = ?
                """,
                (started, started, spec.group_id, spec.attempt),
            )
            self._connection.execute(
                """
                UPDATE controller_auxiliary_jobs SET status = 'RUNNING'
                WHERE group_id = ? AND group_attempt = ?
                """,
                (spec.group_id, spec.attempt),
            )
            stage = self._connection.execute(
                "SELECT stage, phase FROM stage_plan WHERE node = ?",
                (spec.node,),
            ).fetchone()
            self._touch_stage(stage["stage"], stage["phase"], started)
            self._insert_event(
                event_type="DAG_AUXILIARY_RUNNING_BEFORE_SPAWN",
                severity="INFO",
                cell_id=None,
                attempt=None,
                payload={
                    "node": spec.node,
                    "group_id": spec.group_id,
                    "attempt": spec.attempt,
                    "assigned_gpu_uuids": list(spec.assigned_gpu_uuids),
                },
                occurred_at_ns=started,
            )
        process = launcher()
        if type(process) is not SpawnedProcess:
            raise TypeError("auxiliary launcher returned another process type")
        if process.pid != process.pgid:
            raise ValueError("auxiliary child is not a setsid session leader")
        attached = self._now()
        with self._transaction():
            row = self._require_controller_auxiliary_group(
                spec.group_id,
                spec.attempt,
            )
            if row["status"] != "RUNNING" or row["pid"] is not None:
                raise AttemptTransitionError(
                    "controller auxiliary process attachment differs"
                )
            self._connection.execute(
                """
                UPDATE controller_auxiliary_groups
                SET pid = ?, pgid = ?, process_start_receipt_sha256 = ?,
                    updated_at_ns = ?
                WHERE group_id = ? AND attempt = ?
                """,
                (
                    process.pid,
                    process.pgid,
                    process.process_start_receipt_sha256,
                    attached,
                    spec.group_id,
                    spec.attempt,
                ),
            )
        return process

    def fail_controller_auxiliary_spawn(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        *,
        exception_type: str,
        finished_at_ns: int | None = None,
    ) -> None:
        """Retain one failed launch attempt; never silently resubmit it."""

        if type(spec) is not AuxiliaryPhysicalGroupSpec:
            raise TypeError("auxiliary spawn failure requires an exact group spec")
        _require_text(exception_type, "auxiliary spawn exception type")
        finished = self._validated_time(finished_at_ns)
        with self._transaction():
            row = self._require_controller_auxiliary_group(
                spec.group_id,
                spec.attempt,
            )
            self._require_controller_auxiliary_spec_matches_locked(row, spec)
            if row["status"] != "RUNNING":
                raise AttemptTransitionError(
                    "auxiliary spawn failure requires RUNNING status"
                )
            reason = f"spawn_failed:{exception_type}"
            self._connection.execute(
                """
                UPDATE controller_auxiliary_groups
                SET status = 'FAILED', finished_at_ns = ?,
                    failure_code = 'INFRASTRUCTURE:SPAWN_FAILED',
                    failure_class = 'INFRASTRUCTURE',
                    exclusion_reason = ?, updated_at_ns = ?
                WHERE group_id = ? AND attempt = ?
                """,
                (finished, reason, finished, spec.group_id, spec.attempt),
            )
            self._connection.execute(
                """
                UPDATE controller_auxiliary_jobs
                SET status = 'FAILED', failure_class = 'INFRASTRUCTURE',
                    failure_code = 'SPAWN_FAILED', included_in_analysis = 0,
                    exclusion_reason = ?
                WHERE group_id = ? AND group_attempt = ?
                """,
                (reason, spec.group_id, spec.attempt),
            )
            stage = self._connection.execute(
                "SELECT stage, phase FROM stage_plan WHERE node = ?",
                (spec.node,),
            ).fetchone()
            self._touch_stage(stage["stage"], stage["phase"], finished)
            self._insert_event(
                event_type="DAG_AUXILIARY_SPAWN_FAILED",
                severity="ERROR",
                cell_id=None,
                attempt=None,
                payload={
                    "node": spec.node,
                    "group_id": spec.group_id,
                    "attempt": spec.attempt,
                    "exception_type": exception_type,
                },
                occurred_at_ns=finished,
            )

    def attach_controller_auxiliary_group_process(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        *,
        pid: int,
        pgid: int,
        process_start_receipt_sha256: str,
        attached_at_ns: int | None = None,
    ) -> None:
        """Recover an auxiliary wrapper after commit/spawn/attach interruption."""

        if type(spec) is not AuxiliaryPhysicalGroupSpec:
            raise TypeError("auxiliary process attachment requires an exact spec")
        _require_positive_int(pid, "auxiliary PID")
        _require_positive_int(pgid, "auxiliary PGID")
        if pid != pgid:
            raise ValueError("auxiliary wrapper must be a setsid session leader")
        _require_sha256(
            process_start_receipt_sha256,
            "auxiliary process start receipt SHA-256",
        )
        attached = self._validated_time(attached_at_ns)
        with self._transaction():
            row = self._require_controller_auxiliary_group(
                spec.group_id,
                spec.attempt,
            )
            self._require_controller_auxiliary_spec_matches_locked(row, spec)
            if row["status"] != "RUNNING":
                raise AttemptTransitionError(
                    "auxiliary process attachment requires RUNNING status"
                )
            if row["pid"] is not None:
                if (int(row["pid"]), int(row["pgid"])) != (pid, pgid):
                    raise AttemptTransitionError(
                        "auxiliary process attachment differs from durable state"
                    )
                registered = row["process_start_receipt_sha256"]
                if (
                    registered is not None
                    and registered != process_start_receipt_sha256
                ):
                    raise AttemptTransitionError(
                        "auxiliary process start receipt differs from durable state"
                    )
                if registered is None:
                    self._connection.execute(
                        """
                        UPDATE controller_auxiliary_groups
                        SET process_start_receipt_sha256 = ?, updated_at_ns = ?
                        WHERE group_id = ? AND attempt = ?
                        """,
                        (
                            process_start_receipt_sha256,
                            attached,
                            spec.group_id,
                            spec.attempt,
                        ),
                    )
                return
            self._connection.execute(
                """
                UPDATE controller_auxiliary_groups
                SET pid = ?, pgid = ?, process_start_receipt_sha256 = ?,
                    updated_at_ns = ?
                WHERE group_id = ? AND attempt = ?
                """,
                (
                    pid,
                    pgid,
                    process_start_receipt_sha256,
                    attached,
                    spec.group_id,
                    spec.attempt,
                ),
            )
            self._insert_event(
                event_type="DAG_AUXILIARY_PROCESS_RECOVERED",
                severity="INFO",
                cell_id=None,
                attempt=None,
                payload={
                    "node": spec.node,
                    "group_id": spec.group_id,
                    "group_attempt": spec.attempt,
                    "pid": pid,
                    "pgid": pgid,
                    "process_start_receipt_sha256": (process_start_receipt_sha256),
                },
                occurred_at_ns=attached,
            )

    def record_controller_auxiliary_observation(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        *,
        log_size_bytes: int,
        heartbeat: WorkerHeartbeat | None,
        gpu_observation: Mapping[str, Any],
        observed_at_ns: int | None = None,
    ) -> None:
        """Persist child-owned auxiliary progress plus diagnostic observations."""

        if type(spec) is not AuxiliaryPhysicalGroupSpec:
            raise TypeError("auxiliary observation requires an exact spec")
        if type(log_size_bytes) is not int or log_size_bytes < 0:
            raise ValueError("auxiliary observed log size is invalid")
        if heartbeat is not None and type(heartbeat) is not WorkerHeartbeat:
            raise TypeError("auxiliary heartbeat requires the exact runtime type")
        gpu = _canonical_mapping(
            gpu_observation,
            "auxiliary GPU observation",
            allow_empty=True,
        )
        observed = self._validated_time(observed_at_ns)
        with self._transaction():
            row = self._require_controller_auxiliary_group(
                spec.group_id,
                spec.attempt,
            )
            self._require_controller_auxiliary_spec_matches_locked(row, spec)
            if row["status"] != "RUNNING":
                raise AttemptTransitionError(
                    "auxiliary observation requires RUNNING status"
                )
            prior_size = row["last_log_size_bytes"]
            growth_ns = row["last_log_growth_ns"]
            if prior_size is None or log_size_bytes > int(prior_size):
                growth_ns = observed
            elif log_size_bytes < int(prior_size):
                raise ValueError("auxiliary log size moved backwards")
            heartbeat_at = row["heartbeat_at_ns"]
            heartbeat_sequence = int(row["heartbeat_sequence"])
            if heartbeat is not None:
                if heartbeat.command_sha256 != spec.launch_command_sha256:
                    raise ValueError("auxiliary heartbeat command identity differs")
                if heartbeat.sequence < heartbeat_sequence:
                    raise ValueError("auxiliary heartbeat sequence moved backwards")
                if heartbeat.observed_at_ns > observed:
                    raise ValueError("auxiliary heartbeat is from the future")
                if heartbeat.sequence > heartbeat_sequence:
                    heartbeat_sequence = heartbeat.sequence
                    heartbeat_at = heartbeat.observed_at_ns
            self._connection.execute(
                """
                UPDATE controller_auxiliary_groups
                SET heartbeat_at_ns = ?, heartbeat_sequence = ?,
                    last_log_size_bytes = ?, last_log_growth_ns = ?,
                    gpu_observation_json = ?, updated_at_ns = ?
                WHERE group_id = ? AND attempt = ?
                """,
                (
                    heartbeat_at,
                    heartbeat_sequence,
                    log_size_bytes,
                    growth_ns,
                    _canonical_json(gpu),
                    observed,
                    spec.group_id,
                    spec.attempt,
                ),
            )

    def request_controller_auxiliary_termination(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        *,
        reason: str,
        requested_at_ns: int | None = None,
    ) -> None:
        if type(spec) is not AuxiliaryPhysicalGroupSpec:
            raise TypeError("auxiliary termination requires an exact spec")
        _require_text(reason, "auxiliary termination reason")
        requested = self._validated_time(requested_at_ns)
        with self._transaction():
            row = self._require_controller_auxiliary_group(
                spec.group_id,
                spec.attempt,
            )
            self._require_controller_auxiliary_spec_matches_locked(row, spec)
            if row["status"] != "RUNNING":
                raise AttemptTransitionError(
                    "auxiliary termination requires RUNNING status"
                )
            if row["termination_reason"] is not None:
                if row["termination_reason"] != reason:
                    raise AttemptTransitionError(
                        "auxiliary termination reason is immutable"
                    )
                return
            self._connection.execute(
                """
                UPDATE controller_auxiliary_groups
                SET termination_reason = ?, termination_requested_at_ns = ?,
                    updated_at_ns = ?
                WHERE group_id = ? AND attempt = ?
                """,
                (reason, requested, requested, spec.group_id, spec.attempt),
            )
            self._insert_event(
                event_type="DAG_AUXILIARY_TERMINATION_REQUESTED",
                severity="ERROR",
                cell_id=None,
                attempt=None,
                payload={
                    "node": spec.node,
                    "group_id": spec.group_id,
                    "group_attempt": spec.attempt,
                    "reason": reason,
                },
                occurred_at_ns=requested,
            )

    def record_controller_auxiliary_termination_signal(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        *,
        signal_name: Literal["TERM", "KILL"],
        sent_at_ns: int | None = None,
    ) -> None:
        if signal_name not in {"TERM", "KILL"}:
            raise ValueError("auxiliary termination signal is not registered")
        sent = self._validated_time(sent_at_ns)
        column = "term_sent_at_ns" if signal_name == "TERM" else "kill_sent_at_ns"
        with self._transaction():
            row = self._require_controller_auxiliary_group(
                spec.group_id,
                spec.attempt,
            )
            self._require_controller_auxiliary_spec_matches_locked(row, spec)
            if row["status"] != "RUNNING" or row["termination_reason"] is None:
                raise AttemptTransitionError(
                    "auxiliary signal requires requested RUNNING termination"
                )
            if signal_name == "KILL" and row["term_sent_at_ns"] is None:
                raise AttemptTransitionError("auxiliary KILL requires prior TERM")
            if row[column] is not None:
                return
            self._connection.execute(
                f"""
                UPDATE controller_auxiliary_groups SET {column} = ?, updated_at_ns = ?
                WHERE group_id = ? AND attempt = ?
                """,
                (sent, sent, spec.group_id, spec.attempt),
            )

    def finish_controller_auxiliary_group(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        terminal: AuxiliaryGroupTerminal,
        *,
        finished_at_ns: int | None = None,
    ) -> None:
        """Deep-validate and atomically terminalize one auxiliary campaign."""

        if type(spec) is not AuxiliaryPhysicalGroupSpec:
            raise TypeError("auxiliary terminal requires an exact group spec")
        if type(terminal) is not AuxiliaryGroupTerminal:
            raise TypeError("auxiliary terminal requires an exact terminal bundle")
        publication = self._reopen_controller_binding(terminal.publication)
        terminal_by_job = dict(terminal.terminals)
        expected_job_ids = tuple(job.job_id for job in spec.jobs)
        if tuple(sorted(terminal_by_job)) != expected_job_ids:
            raise ValueError("auxiliary terminal job coverage differs")
        timings = tuple(
            (terminal_by_job[job_id].started_ns, terminal_by_job[job_id].finished_ns)
            for job_id in expected_job_ids
        )
        if any(start is None or end is None for start, end in timings):
            raise ValueError("auxiliary terminals require durable lifecycle timing")
        evidence_started = min(int(start) for start, _end in timings)
        evidence_finished = max(int(end) for _start, end in timings)
        finished = self._validated_time(finished_at_ns)
        if evidence_finished > finished:
            raise ValueError("auxiliary evidence finishes after ledger time")
        group_status = (
            "COMPLETE"
            if all(value.status == "COMPLETE" for value in terminal_by_job.values())
            else "FAILED"
        )
        compute_gpu_seconds = float(terminal.compute_gpu_seconds)
        reserved_gpu_seconds = float(terminal.reserved_gpu_seconds)
        billed_gpu_seconds = float(terminal.billed_gpu_seconds)
        atomic_publication_sha256 = next(
            iter(
                {value.atomic_publication_sha256 for value in terminal_by_job.values()}
            )
        )
        with self._transaction():
            row = self._require_controller_auxiliary_group(
                spec.group_id,
                spec.attempt,
            )
            self._require_controller_auxiliary_spec_matches_locked(row, spec)
            if row["status"] != "RUNNING":
                raise AttemptTransitionError(
                    "auxiliary terminal requires RUNNING status"
                )
            if evidence_started < int(row["started_at_ns"]):
                raise ValueError("auxiliary evidence starts before RUNNING commit")
            for job in spec.jobs:
                value = terminal_by_job[job.job_id]
                self._connection.execute(
                    """
                    UPDATE controller_auxiliary_jobs
                    SET status = ?, exit_code = ?, terminal_sha256 = ?,
                        junit_sha256 = ?, raw_log_sha256 = ?,
                        evidence_files_json = ?, failure_class = ?,
                        failure_code = ?, included_in_analysis = ?,
                        exclusion_reason = ?, evidence_started_at_ns = ?,
                        evidence_finished_at_ns = ?
                    WHERE job_id = ? AND job_attempt = ?
                    """,
                    (
                        value.status,
                        value.exit_code,
                        value.terminal_sha256,
                        value.junit_sha256,
                        value.raw_log_sha256,
                        _canonical_json(dict(value.evidence_files or {})),
                        value.failure_class,
                        value.failure_code,
                        int(value.included_in_analysis),
                        value.exclusion_reason,
                        value.started_ns,
                        value.finished_ns,
                        job.job_id,
                        job.attempt,
                    ),
                )
            failure_code = None
            failure_class = None
            exclusion_reason = None
            if group_status == "FAILED":
                failed = tuple(
                    sorted(
                        job_id
                        for job_id, value in terminal_by_job.items()
                        if value.status != "COMPLETE"
                    )
                )
                failed_classes = tuple(
                    sorted(
                        {
                            str(value.failure_class)
                            for value in terminal_by_job.values()
                            if value.status == "FAILED"
                        }
                    )
                )
                failure_class = (
                    failed_classes[0] if len(failed_classes) == 1 else "MIXED"
                )
                failure_code = "AUXILIARY_MEMBER_FAILED"
                exclusion_reason = "terminal_failed_jobs:" + ",".join(failed)
            self._connection.execute(
                """
                UPDATE controller_auxiliary_groups
                SET status = ?, finished_at_ns = ?, publication_path = ?,
                    publication_sha256 = ?, atomic_publication_sha256 = ?,
                    failure_code = ?, failure_class = ?, exclusion_reason = ?,
                    compute_gpu_seconds = ?,
                    reserved_gpu_seconds = ?, billed_gpu_seconds = ?,
                    updated_at_ns = ?
                WHERE group_id = ? AND attempt = ?
                """,
                (
                    group_status,
                    finished,
                    publication.absolute_path,
                    publication.sha256,
                    atomic_publication_sha256,
                    failure_code,
                    failure_class,
                    exclusion_reason,
                    compute_gpu_seconds,
                    reserved_gpu_seconds,
                    billed_gpu_seconds,
                    finished,
                    spec.group_id,
                    spec.attempt,
                ),
            )
            stage = self._connection.execute(
                "SELECT stage, phase FROM stage_plan WHERE node = ?",
                (spec.node,),
            ).fetchone()
            self._touch_stage(stage["stage"], stage["phase"], finished)
            self._insert_event(
                event_type="DAG_AUXILIARY_TERMINAL_ACCEPTED",
                severity="INFO" if group_status == "COMPLETE" else "ERROR",
                cell_id=None,
                attempt=None,
                payload={
                    "node": spec.node,
                    "group_id": spec.group_id,
                    "attempt": spec.attempt,
                    "status": group_status,
                    "failure_class": failure_class,
                    "publication_sha256": publication.sha256,
                    "atomic_publication_sha256": atomic_publication_sha256,
                    "compute_gpu_seconds": compute_gpu_seconds,
                    "reserved_gpu_seconds": reserved_gpu_seconds,
                    "billed_gpu_seconds": billed_gpu_seconds,
                    "job_count": len(spec.jobs),
                },
                occurred_at_ns=finished,
            )

    def adopt_controller_auxiliary_jobs(
        self,
        *,
        node: str,
        group_id: str,
        group_attempt: int,
        adoptions: tuple[AuxiliaryCellAdoption, ...],
        adopted_at_ns: int | None = None,
    ) -> None:
        """Atomically adopt exact later cell IDs without charging GPU time twice."""

        _require_text(node, "auxiliary adoption node")
        _require_text(group_id, "auxiliary adoption group ID")
        _require_positive_int(group_attempt, "auxiliary adoption group attempt")
        if (
            type(adoptions) is not tuple
            or not adoptions
            or any(type(value) is not AuxiliaryCellAdoption for value in adoptions)
        ):
            raise TypeError("auxiliary adoption requires exact non-empty rows")
        identities = tuple((value.job_id, value.job_attempt) for value in adoptions)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("auxiliary adoptions must be uniquely sorted by job")
        cell_ids = tuple(value.attempt.cell_id for value in adoptions)
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("auxiliary adoptions contain duplicate cell IDs")
        adopted = self._validated_time(adopted_at_ns)
        with self._transaction():
            controller = self._require_controller_node(node)
            if controller["state"] != "MATERIALIZED":
                raise AttemptTransitionError(
                    "auxiliary adoption requires a MATERIALIZED node"
                )
            group = self._require_controller_auxiliary_group(group_id, group_attempt)
            if group["node"] != node or group["status"] != "COMPLETE":
                raise AttemptTransitionError(
                    "only the node's COMPLETE auxiliary group may be adopted"
                )
            auxiliary_sources = json.loads(controller["auxiliary_sources_json"])
            expected_source = auxiliary_sources.get(group["source_kind"])
            if expected_source != {
                "absolute_path": group["publication_path"],
                "sha256": group["publication_sha256"],
            }:
                raise ExperimentOperatorError(
                    "materialization switched its completed auxiliary source"
                )
            jobs = self._connection.execute(
                """
                SELECT * FROM controller_auxiliary_jobs
                WHERE group_id = ? AND group_attempt = ?
                ORDER BY job_id, job_attempt
                """,
                (group_id, group_attempt),
            ).fetchall()
            if tuple((row["job_id"], int(row["job_attempt"])) for row in jobs) != (
                identities
            ):
                raise ValueError("auxiliary adoption does not cover every exact job")
            if group["adopted_at_ns"] is not None:
                persisted = tuple(
                    (
                        row["job_id"],
                        int(row["job_attempt"]),
                        row["adoption_key"],
                        row["adopted_cell_id"],
                        int(row["adopted_cell_attempt"]),
                    )
                    for row in jobs
                )
                requested = tuple(
                    (
                        value.job_id,
                        value.job_attempt,
                        value.adoption_key,
                        value.attempt.cell_id,
                        value.attempt.attempt,
                    )
                    for value in adoptions
                )
                if persisted != requested:
                    raise ExperimentOperatorError(
                        "completed auxiliary adoption mapping differs"
                    )
                for value in adoptions:
                    actual = self.latest_attempt(value.attempt.cell_id)
                    if actual is None:
                        raise ExperimentOperatorError(
                            "adopted auxiliary cell disappeared from the ledger"
                        )
                    _require_cell_attempt_spec_matches(actual, value.attempt)
                return
            stage = self._connection.execute(
                "SELECT stage, phase FROM stage_plan WHERE node = ?",
                (node,),
            ).fetchone()
            accounting_owner = str(jobs[0]["job_id"])
            by_job = {(value.job_id, value.job_attempt): value for value in adoptions}
            for job in jobs:
                key = (str(job["job_id"]), int(job["job_attempt"]))
                value = by_job[key]
                spec = value.attempt
                if (
                    value.adoption_key != job["adoption_key"]
                    or spec.stage != stage["stage"]
                    or spec.phase != stage["phase"]
                    or _canonical_json(dict(spec.scientific_axes))
                    != job["scientific_axes_json"]
                    or _canonical_json(dict(spec.identity)) != job["identity_json"]
                    or spec.command_sha256 != job["command_sha256"]
                    or spec.output_directory != job["output_directory"]
                    or job["status"] != "COMPLETE"
                ):
                    raise ExperimentOperatorError(
                        "auxiliary adoption differs from completed job identity"
                    )
                self._materialize_attempt_locked(spec, now=adopted)
                owns_accounting = str(job["job_id"]) == accounting_owner
                self._connection.execute(
                    """
                    UPDATE cell_attempts
                    SET status = 'COMPLETE', assigned_gpu_uuids_json = ?,
                        pid = ?, pgid = ?, started_at_ns = ?, finished_at_ns = ?,
                        heartbeat_at_ns = ?, exit_code = ?, terminal_sha256 = ?,
                        junit_sha256 = ?, raw_log_sha256 = ?,
                        evidence_files_json = ?, included_in_analysis = ?,
                        exclusion_reason = ?, compute_gpu_seconds = ?,
                        reserved_gpu_seconds = ?, billed_gpu_seconds = ?,
                        updated_at_ns = ?
                    WHERE cell_id = ? AND attempt = ?
                    """,
                    (
                        group["assigned_gpu_uuids_json"],
                        group["pid"],
                        group["pgid"],
                        job["evidence_started_at_ns"],
                        job["evidence_finished_at_ns"],
                        job["evidence_finished_at_ns"],
                        job["exit_code"],
                        job["terminal_sha256"],
                        job["junit_sha256"],
                        job["raw_log_sha256"],
                        job["evidence_files_json"],
                        job["included_in_analysis"],
                        job["exclusion_reason"],
                        float(group["compute_gpu_seconds"]) if owns_accounting else 0.0,
                        float(group["reserved_gpu_seconds"])
                        if owns_accounting
                        else 0.0,
                        float(group["billed_gpu_seconds"]) if owns_accounting else 0.0,
                        adopted,
                        spec.cell_id,
                        spec.attempt,
                    ),
                )
                self._connection.execute(
                    """
                    UPDATE controller_auxiliary_jobs
                    SET adopted_cell_id = ?, adopted_cell_attempt = ?
                    WHERE job_id = ? AND job_attempt = ?
                    """,
                    (spec.cell_id, spec.attempt, job["job_id"], job["job_attempt"]),
                )
            self._connection.execute(
                """
                UPDATE controller_auxiliary_groups
                SET adopted_at_ns = ?, updated_at_ns = ?
                WHERE group_id = ? AND attempt = ?
                """,
                (adopted, adopted, group_id, group_attempt),
            )
            self._touch_stage(stage["stage"], stage["phase"], adopted)
            self._insert_event(
                event_type="DAG_AUXILIARY_ADOPTED",
                severity="INFO",
                cell_id=None,
                attempt=None,
                payload={
                    "node": node,
                    "group_id": group_id,
                    "attempt": group_attempt,
                    "cell_count": len(adoptions),
                    "accounting_owner_job_id": accounting_owner,
                    "compute_gpu_seconds": float(group["compute_gpu_seconds"]),
                },
                occurred_at_ns=adopted,
            )

    def materialize_attempt(self, spec: CellAttemptSpec) -> None:
        now = self._now()
        with self._transaction():
            self._materialize_attempt_locked(spec, now=now)

    def configure_interference_envelope(
        self,
        envelope: InterferenceEnvelope,
        *,
        configured_at_ns: int | None = None,
    ) -> None:
        """Persist the only GPU parallelism authority consumed by the daemon."""

        if type(envelope) is not InterferenceEnvelope:
            raise TypeError("scheduler requires an exact interference envelope")
        configured = self._validated_time(configured_at_ns)
        encoded = _canonical_json(asdict(envelope))
        with self._transaction():
            running = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM cell_attempts WHERE status = 'RUNNING'"
                ).fetchone()[0]
            )
            current = self._metadata_value("interference_envelope")
            if current == encoded:
                return
            if running:
                raise AttemptTransitionError(
                    "interference envelope cannot change while attempts run"
                )
            self._set_metadata("interference_envelope", encoded)
            self._insert_event(
                event_type="INTERFERENCE_ENVELOPE_CONFIGURED",
                severity="INFO" if envelope.mode != "UNRESOLVED" else "WARNING",
                cell_id=None,
                attempt=None,
                payload=asdict(envelope),
                occurred_at_ns=configured,
            )

    def interference_envelope(self) -> InterferenceEnvelope:
        value = self._metadata_value("interference_envelope")
        if value is None:
            raise ExperimentOperatorError("interference envelope is not configured")
        decoded = json.loads(value)
        if type(decoded) is not dict:
            raise ExperimentOperatorError("interference envelope metadata differs")
        decoded["gpu_uuids"] = tuple(decoded["gpu_uuids"])
        return InterferenceEnvelope(**decoded)

    def set_dispatch_stop(
        self,
        reason: str,
        *,
        stopped_at_ns: int | None = None,
    ) -> None:
        """Persist a fail-closed STOP gate; reconciliation may continue."""

        _require_text(reason, "dispatch STOP reason")
        stopped = self._validated_time(stopped_at_ns)
        with self._transaction():
            current_state = self._metadata_value("dispatch_state") or "RUN"
            current_reason = self._metadata_value("dispatch_stop_reason") or None
            if current_state == "STOP":
                if current_reason != reason:
                    self._insert_event(
                        event_type="SCHEDULER_STOP_SECONDARY_REASON",
                        severity="CRITICAL",
                        cell_id=None,
                        attempt=None,
                        payload={
                            "primary_reason": current_reason,
                            "secondary_reason": reason,
                        },
                        occurred_at_ns=stopped,
                    )
                return
            self._set_metadata("dispatch_state", "STOP")
            self._set_metadata("dispatch_stop_reason", reason)
            self._set_metadata("dispatch_running_recovery_evidence", "")
            self._insert_event(
                event_type="SCHEDULER_STOPPED",
                severity="CRITICAL",
                cell_id=None,
                attempt=None,
                payload={"reason": reason},
                occurred_at_ns=stopped,
            )

    def _require_running_recovery_evidence_locked(
        self,
        evidence: Mapping[str, Any],
        *,
        stop_reason: str,
    ) -> None:
        expected_fields = {
            "schema_version",
            "kind",
            "mode",
            "stop_reason",
            "verified_at_ns",
            "processes",
            "heartbeat_observations",
            "manual_evidence",
        }
        mode = evidence.get("mode")
        processes = evidence.get("processes")
        heartbeats = evidence.get("heartbeat_observations")
        manual = evidence.get("manual_evidence")
        if (
            set(evidence) != expected_fields
            or evidence.get("schema_version") != 1
            or evidence.get("kind") != "formal_experiment_dispatch_running_recovery"
            or mode not in {"FRESH_CHILD_HEARTBEAT", "MANUAL_OPERATOR_EVIDENCE"}
            or evidence.get("stop_reason") != stop_reason
            or type(evidence.get("verified_at_ns")) is not int
            or int(evidence["verified_at_ns"]) < 1
            or type(processes) is not list
            or not processes
            or type(heartbeats) is not list
            or (mode == "FRESH_CHILD_HEARTBEAT" and manual is not None)
            or (mode == "MANUAL_OPERATOR_EVIDENCE" and type(manual) is not dict)
        ):
            raise AttemptTransitionError("dispatch RUNNING recovery envelope differs")
        running_rows = self._connection.execute(
            """
            SELECT * FROM cell_attempts
            WHERE status = 'RUNNING' ORDER BY cell_id, attempt
            """
        ).fetchall()
        running = {
            (str(row["cell_id"]), int(row["attempt"])): row for row in running_rows
        }
        covered: set[tuple[str, int]] = set()
        prior_process: tuple[str, int] | None = None
        for process in processes:
            if type(process) is not dict or set(process) != {
                "cell_id",
                "attempt",
                "command_sha256",
                "pid",
                "pgid",
                "process_start_receipt_sha256",
                "covered_attempts",
            }:
                raise AttemptTransitionError(
                    "dispatch RUNNING recovery process fields differ"
                )
            identity = (process.get("cell_id"), process.get("attempt"))
            if (
                type(identity[0]) is not str
                or not identity[0]
                or type(identity[1]) is not int
                or identity[1] < 1
                or (prior_process is not None and identity <= prior_process)
            ):
                raise AttemptTransitionError(
                    "dispatch RUNNING recovery process order differs"
                )
            prior_process = identity  # type: ignore[assignment]
            leader = running.get(identity)  # type: ignore[arg-type]
            command = self._connection.execute(
                """
                SELECT * FROM command_queue
                WHERE cell_id = ? AND attempt = ?
                """,
                identity,
            ).fetchone()
            if (
                leader is None
                or command is None
                or process.get("command_sha256")
                != _decoded_command(command).command_sha256
                or process.get("pid") != leader["pid"]
                or process.get("pgid") != leader["pgid"]
                or process.get("process_start_receipt_sha256")
                != leader["process_start_receipt_sha256"]
            ):
                raise AttemptTransitionError(
                    "dispatch RUNNING recovery leader identity differs"
                )
            _require_sha256(
                process["process_start_receipt_sha256"],
                "dispatch RUNNING recovery start receipt",
            )
            members = process.get("covered_attempts")
            if type(members) is not list or not members:
                raise AttemptTransitionError(
                    "dispatch RUNNING recovery coverage is empty"
                )
            prior_member: tuple[str, int] | None = None
            for member in members:
                if type(member) is not dict or set(member) != {"cell_id", "attempt"}:
                    raise AttemptTransitionError(
                        "dispatch RUNNING recovery member fields differ"
                    )
                member_identity = (member.get("cell_id"), member.get("attempt"))
                member_row = running.get(member_identity)  # type: ignore[arg-type]
                if (
                    type(member_identity[0]) is not str
                    or not member_identity[0]
                    or type(member_identity[1]) is not int
                    or member_identity[1] < 1
                    or (prior_member is not None and member_identity <= prior_member)
                    or member_row is None
                    or member_row["pid"] != leader["pid"]
                    or member_row["pgid"] != leader["pgid"]
                    or member_row["process_start_receipt_sha256"]
                    != leader["process_start_receipt_sha256"]
                ):
                    raise AttemptTransitionError(
                        "dispatch RUNNING recovery member identity differs"
                    )
                prior_member = member_identity  # type: ignore[assignment]
                covered.add(member_identity)  # type: ignore[arg-type]
        if covered != set(running):
            raise AttemptTransitionError(
                "dispatch RUNNING recovery coverage differs from ledger"
            )
        if mode == "FRESH_CHILD_HEARTBEAT" and len(heartbeats) != len(processes):
            raise AttemptTransitionError(
                "dispatch RUNNING recovery lacks fresh heartbeat coverage"
            )
        if mode == "MANUAL_OPERATOR_EVIDENCE" and set(manual) != {
            "absolute_path",
            "sha256",
        }:
            raise AttemptTransitionError(
                "dispatch manual recovery binding fields differ"
            )
        if mode == "MANUAL_OPERATOR_EVIDENCE":
            _require_sha256(manual["sha256"], "dispatch manual recovery binding")

    def _require_auxiliary_running_recovery_evidence_locked(
        self,
        evidence: Mapping[str, Any],
        *,
        stop_reason: str,
    ) -> None:
        group = evidence.get("group")
        heartbeat = None if type(group) is not dict else group.get("heartbeat")
        if (
            set(evidence)
            != {
                "schema_version",
                "kind",
                "mode",
                "stop_reason",
                "verified_at_ns",
                "group",
                "manual_evidence",
            }
            or evidence.get("schema_version") != 1
            or evidence.get("kind")
            != "formal_experiment_auxiliary_dispatch_running_recovery"
            or evidence.get("mode") != "FRESH_CHILD_HEARTBEAT"
            or evidence.get("stop_reason") != stop_reason
            or type(evidence.get("verified_at_ns")) is not int
            or int(evidence["verified_at_ns"]) < 1
            or evidence.get("manual_evidence") is not None
            or type(group) is not dict
            or set(group)
            != {
                "group_id",
                "attempt",
                "node",
                "command_sha256",
                "pid",
                "pgid",
                "process_start_receipt_sha256",
                "heartbeat",
            }
            or type(heartbeat) is not dict
            or set(heartbeat)
            != {
                "command_sha256",
                "worker_pid",
                "sequence",
                "observed_at_ns",
                "phase",
            }
        ):
            raise AttemptTransitionError(
                "dispatch auxiliary RUNNING recovery envelope differs"
            )
        rows = self._connection.execute(
            """
            SELECT * FROM controller_auxiliary_groups
            WHERE status = 'RUNNING'
            """
        ).fetchall()
        if len(rows) != 1:
            raise AttemptTransitionError(
                "dispatch auxiliary recovery requires exact one RUNNING group"
            )
        row = rows[0]
        if (
            group["group_id"] != row["group_id"]
            or group["attempt"] != row["attempt"]
            or group["node"] != row["node"]
            or group["command_sha256"] != row["launch_command_sha256"]
            or group["pid"] != row["pid"]
            or group["pgid"] != row["pgid"]
            or group["process_start_receipt_sha256"]
            != row["process_start_receipt_sha256"]
            or heartbeat["command_sha256"] != row["launch_command_sha256"]
            or heartbeat["sequence"] < row["heartbeat_sequence"]
        ):
            raise AttemptTransitionError(
                "dispatch auxiliary RUNNING recovery identity differs"
            )
        _require_sha256(
            group["process_start_receipt_sha256"],
            "dispatch auxiliary recovery start receipt",
        )

    def clear_dispatch_stop(
        self,
        *,
        reason: str,
        running_recovery_evidence: Mapping[str, Any] | None = None,
        cleared_at_ns: int | None = None,
    ) -> None:
        """Explicit operator action that re-enables spawning after a STOP."""

        _require_text(reason, "dispatch resume reason")
        recovery_evidence = (
            None
            if running_recovery_evidence is None
            else _canonical_mapping(
                running_recovery_evidence,
                "dispatch running recovery evidence",
                allow_empty=False,
            )
        )
        cleared = self._validated_time(cleared_at_ns)
        with self._transaction():
            current_state = self._metadata_value("dispatch_state") or "RUN"
            current_stop_reason = self._metadata_value("dispatch_stop_reason") or None
            if current_state != "STOP" or current_stop_reason is None:
                raise AttemptTransitionError(
                    "dispatch resume requires a durable STOP state"
                )
            running_attempts = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM cell_attempts WHERE status = 'RUNNING'"
                ).fetchone()[0]
            )
            running_auxiliary_groups = int(
                self._connection.execute(
                    """
                    SELECT COUNT(*) FROM controller_auxiliary_groups
                    WHERE status = 'RUNNING'
                    """
                ).fetchone()[0]
            )
            if running_auxiliary_groups and running_attempts:
                raise AttemptTransitionError(
                    "dispatch STOP cannot clear with mixed RUNNING work"
                )
            if running_auxiliary_groups and recovery_evidence is None:
                raise AttemptTransitionError(
                    "dispatch STOP cannot clear with unverified auxiliary work"
                )
            if running_attempts and recovery_evidence is None:
                raise AttemptTransitionError(
                    "dispatch STOP cannot clear with unverified RUNNING attempts"
                )
            if running_attempts:
                assert recovery_evidence is not None
                self._require_running_recovery_evidence_locked(
                    recovery_evidence,
                    stop_reason=current_stop_reason,
                )
            elif running_auxiliary_groups:
                assert recovery_evidence is not None
                self._require_auxiliary_running_recovery_evidence_locked(
                    recovery_evidence,
                    stop_reason=current_stop_reason,
                )
            self._set_metadata("dispatch_state", "RUN")
            self._set_metadata("dispatch_stop_reason", "")
            self._set_metadata(
                "dispatch_running_recovery_evidence",
                "" if recovery_evidence is None else _canonical_json(recovery_evidence),
            )
            self._insert_event(
                event_type="SCHEDULER_RESUMED",
                severity="WARNING",
                cell_id=None,
                attempt=None,
                payload={
                    "reason": reason,
                    "running_recovery_evidence": recovery_evidence,
                },
                occurred_at_ns=cleared,
            )

    def dispatch_control(self) -> tuple[Literal["RUN", "STOP"], str | None]:
        state = self._metadata_value("dispatch_state") or "RUN"
        if state not in {"RUN", "STOP"}:
            raise ExperimentOperatorError("dispatch control metadata differs")
        reason = self._metadata_value("dispatch_stop_reason") or None
        return state, reason  # type: ignore[return-value]

    def dispatch_running_recovery_evidence(self) -> dict[str, Any] | None:
        """Return the exact manual/fresh recovery envelope for the current RUN."""

        raw = self._metadata_value("dispatch_running_recovery_evidence")
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ExperimentOperatorError(
                "dispatch running recovery evidence is invalid JSON"
            ) from error
        return _canonical_mapping(
            value,
            "dispatch running recovery evidence",
            allow_empty=False,
        )

    def request_attempt_termination(
        self,
        cell_id: str,
        attempt: int,
        *,
        reason: str,
        requested_at_ns: int | None = None,
    ) -> None:
        """Persist termination intent for one process lease before signalling it."""

        _require_text(reason, "attempt termination reason")
        requested = self._validated_time(requested_at_ns)
        with self._transaction():
            row = self._require_attempt(cell_id, attempt)
            if row["status"] != "RUNNING":
                raise AttemptTransitionError("termination requires a RUNNING attempt")
            group = self._connection.execute(
                """
                SELECT g.group_id FROM physical_attempt_groups AS g
                JOIN physical_attempt_group_members AS m ON m.group_id = g.group_id
                WHERE m.cell_id = ? AND m.attempt = ?
                """,
                (cell_id, attempt),
            ).fetchone()
            rows = (
                (row,)
                if group is None
                else tuple(
                    self._connection.execute(
                        """
                        SELECT a.* FROM cell_attempts AS a
                        JOIN physical_attempt_group_members AS m
                          ON m.cell_id = a.cell_id AND m.attempt = a.attempt
                        WHERE m.group_id = ? ORDER BY m.member_ordinal
                        """,
                        (group["group_id"],),
                    ).fetchall()
                )
            )
            for member in rows:
                if member["status"] != "RUNNING":
                    raise AttemptTransitionError(
                        "termination process lease contains a non-RUNNING member"
                    )
                if member["termination_reason"] is not None:
                    if member["termination_reason"] != reason:
                        raise AttemptTransitionError(
                            "attempt termination reason is immutable"
                        )
                    continue
                self._connection.execute(
                    """
                    UPDATE cell_attempts
                    SET termination_reason = ?, termination_requested_at_ns = ?,
                        updated_at_ns = ?
                    WHERE cell_id = ? AND attempt = ?
                    """,
                    (
                        reason,
                        requested,
                        requested,
                        member["cell_id"],
                        member["attempt"],
                    ),
                )

    def record_attempt_termination_signal(
        self,
        cell_id: str,
        attempt: int,
        *,
        signal_name: Literal["TERM", "KILL"],
        sent_at_ns: int | None = None,
    ) -> None:
        """Record one successfully issued escalation signal on the process lease."""

        if signal_name not in {"TERM", "KILL"}:
            raise ValueError("termination signal must be TERM or KILL")
        sent = self._validated_time(sent_at_ns)
        column = "term_sent_at_ns" if signal_name == "TERM" else "kill_sent_at_ns"
        with self._transaction():
            row = self._require_attempt(cell_id, attempt)
            if row["status"] != "RUNNING" or row["termination_reason"] is None:
                raise AttemptTransitionError(
                    "termination signal requires persisted RUNNING intent"
                )
            group = self._connection.execute(
                """
                SELECT g.group_id FROM physical_attempt_groups AS g
                JOIN physical_attempt_group_members AS m ON m.group_id = g.group_id
                WHERE m.cell_id = ? AND m.attempt = ?
                """,
                (cell_id, attempt),
            ).fetchone()
            rows = (
                (row,)
                if group is None
                else tuple(
                    self._connection.execute(
                        """
                        SELECT a.* FROM cell_attempts AS a
                        JOIN physical_attempt_group_members AS m
                          ON m.cell_id = a.cell_id AND m.attempt = a.attempt
                        WHERE m.group_id = ? ORDER BY m.member_ordinal
                        """,
                        (group["group_id"],),
                    ).fetchall()
                )
            )
            for member in rows:
                if member["termination_reason"] != row["termination_reason"]:
                    raise AttemptTransitionError(
                        "termination intent differs across one process lease"
                    )
                if signal_name == "KILL" and member["term_sent_at_ns"] is None:
                    raise AttemptTransitionError("KILL cannot precede TERM")
                prior = member[column]
                if prior is not None:
                    continue
                self._connection.execute(
                    f"""
                    UPDATE cell_attempts SET {column} = ?, updated_at_ns = ?
                    WHERE cell_id = ? AND attempt = ?
                    """,
                    (sent, sent, member["cell_id"], member["attempt"]),
                )

    def running_termination_count(self) -> int:
        cell_count = int(
            self._connection.execute(
                """
                SELECT COUNT(*) FROM cell_attempts
                WHERE status = 'RUNNING' AND termination_reason IS NOT NULL
                """
            ).fetchone()[0]
        )
        auxiliary_count = int(
            self._connection.execute(
                """
                SELECT COUNT(*) FROM controller_auxiliary_groups
                WHERE status = 'RUNNING' AND termination_reason IS NOT NULL
                """
            ).fetchone()[0]
        )
        return cell_count + auxiliary_count

    def materialize_physical_attempt_group(
        self,
        *,
        group_id: str,
        members: tuple[PhysicalAttemptGroupMemberSpec, ...],
        leader_cell_id: str,
        group_kind: Literal[
            "preflight_exact_ten", "tp1_serving_session"
        ] = "preflight_exact_ten",
        materialized_at_ns: int | None = None,
    ) -> None:
        """Atomically register one approved shared physical execution."""

        _require_text(group_id, "physical attempt group ID")
        _require_text(leader_cell_id, "physical attempt group leader")
        if type(members) is not tuple or any(
            type(member) is not PhysicalAttemptGroupMemberSpec for member in members
        ):
            raise TypeError("physical attempt group members must be an exact tuple")
        identities = tuple(
            (member.attempt.cell_id, member.attempt.attempt) for member in members
        )
        if group_kind == "preflight_exact_ten" and (
            len(members) != 10 or identities != tuple(sorted(set(identities)))
        ):
            raise ValueError(
                "preflight physical attempt group requires ten uniquely sorted members"
            )
        if len(set(identities)) != len(identities):
            raise ValueError("physical attempt group members must be unique")
        inferred_kind = _physical_attempt_group_kind(members)
        if inferred_kind != group_kind:
            raise ValueError("physical attempt group kind differs from its coverage")
        if leader_cell_id != members[0].attempt.cell_id:
            raise ValueError(
                "physical attempt group leader must be canonical first member"
            )
        stage_phases = {
            (member.attempt.stage, member.attempt.phase) for member in members
        }
        if len(stage_phases) != 1:
            raise ValueError("physical attempt group cannot cross stage/phase")
        if group_kind == "preflight_exact_ten":
            if any(
                member.attempt.stage != "preflight"
                or member.attempt.phase != "final"
                or member.command.required_gpu_count != 2
                or member.command.timing_class != "EXCLUSIVE"
                for member in members
            ):
                raise ValueError(
                    "preflight physical attempt group requires final dual-GPU "
                    "exclusive rows"
                )
        elif any(
            member.command.required_gpu_count != 1
            or member.command.timing_class != "HEADLINE"
            for member in members
        ):
            raise ValueError(
                "TP1 serving session group requires single-GPU headline rows"
            )
        if (
            len({member.attempt.attempt for member in members}) != 1
            or len({member.command.command_sha256 for member in members}) != 1
            or len({member.command.launch_compatibility_key for member in members}) != 1
        ):
            raise ValueError(
                "physical attempt group members must share attempt, command, and "
                "launch key"
            )
        materialized = self._validated_time(materialized_at_ns)
        with self._transaction():
            existing = self._connection.execute(
                "SELECT * FROM physical_attempt_groups WHERE group_id = ?",
                (group_id,),
            ).fetchone()
            if existing is not None:
                self._require_physical_group_matches_locked(
                    existing,
                    members=members,
                    leader_cell_id=leader_cell_id,
                )
                return
            for member in members:
                self._materialize_attempt_locked(member.attempt, now=materialized)
            leader_attempt = members[0].attempt.attempt
            self._connection.execute(
                """
                INSERT INTO physical_attempt_groups (
                    group_id, leader_cell_id, leader_attempt, status,
                    created_at_ns, updated_at_ns
                ) VALUES (?, ?, ?, 'PENDING', ?, ?)
                """,
                (
                    group_id,
                    leader_cell_id,
                    leader_attempt,
                    materialized,
                    materialized,
                ),
            )
            for ordinal, member in enumerate(members):
                self._connection.execute(
                    """
                    INSERT INTO physical_attempt_group_members (
                        group_id, cell_id, attempt, logical_kind, member_ordinal
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        group_id,
                        member.attempt.cell_id,
                        member.attempt.attempt,
                        member.logical_kind,
                        ordinal,
                    ),
                )
                self._enqueue_command_locked(
                    member.command,
                    enqueued=materialized,
                )
            self._insert_event(
                event_type="PHYSICAL_ATTEMPT_GROUP_MATERIALIZED",
                severity="INFO",
                cell_id=leader_cell_id,
                attempt=leader_attempt,
                payload={
                    "group_id": group_id,
                    "group_kind": group_kind,
                    "logical_attempt_count": len(members),
                    "logical_coverage": {
                        kind: sum(member.logical_kind == kind for member in members)
                        for kind in sorted({member.logical_kind for member in members})
                    },
                },
                occurred_at_ns=materialized,
            )

    def enqueue_command(
        self,
        command: QueuedCommandSpec,
        *,
        enqueued_at_ns: int | None = None,
    ) -> None:
        """Queue only an exact command whose attempt was already materialized."""

        if type(command) is not QueuedCommandSpec:
            raise TypeError("queue requires an exact command spec")
        enqueued = self._validated_time(enqueued_at_ns)
        with self._transaction():
            self._enqueue_command_locked(command, enqueued=enqueued)

    def queued_commands(
        self, *, status: str = "PENDING"
    ) -> tuple[QueuedCommandSpec, ...]:
        if status not in CELL_ATTEMPT_STATUSES:
            raise ValueError("queued command status differs")
        rows = self._connection.execute(
            """
            SELECT q.* FROM command_queue AS q
            JOIN cell_attempts AS a
              ON a.cell_id = q.cell_id AND a.attempt = q.attempt
            WHERE a.status = ?
            ORDER BY q.priority DESC, q.launch_compatibility_key,
                     q.cell_id, q.attempt
            """,
            (status,),
        ).fetchall()
        return tuple(_decoded_command(row) for row in rows)

    def physical_commands(
        self, *, status: str = "PENDING"
    ) -> tuple[QueuedCommandSpec, ...]:
        """Return standalone commands plus one canonical leader per group."""

        if status not in CELL_ATTEMPT_STATUSES:
            raise ValueError("physical command status differs")
        rows = self._connection.execute(
            """
            SELECT q.* FROM command_queue AS q
            JOIN cell_attempts AS a
              ON a.cell_id = q.cell_id AND a.attempt = q.attempt
            LEFT JOIN physical_attempt_group_members AS m
              ON m.cell_id = q.cell_id AND m.attempt = q.attempt
            LEFT JOIN physical_attempt_groups AS g ON g.group_id = m.group_id
            WHERE a.status = ? AND (
                m.group_id IS NULL
                OR (q.cell_id = g.leader_cell_id AND q.attempt = g.leader_attempt)
            )
            ORDER BY q.priority DESC, q.launch_compatibility_key,
                     q.cell_id, q.attempt
            """,
            (status,),
        ).fetchall()
        return tuple(_decoded_command(row) for row in rows)

    def physical_attempt_group_for_attempt(
        self,
        cell_id: str,
        attempt: int,
    ) -> dict[str, Any] | None:
        self._require_attempt(cell_id, attempt)
        group = self._connection.execute(
            """
            SELECT g.* FROM physical_attempt_groups AS g
            JOIN physical_attempt_group_members AS m ON m.group_id = g.group_id
            WHERE m.cell_id = ? AND m.attempt = ?
            """,
            (cell_id, attempt),
        ).fetchone()
        if group is None:
            return None
        members = self._connection.execute(
            """
            SELECT * FROM physical_attempt_group_members
            WHERE group_id = ? ORDER BY member_ordinal
            """,
            (group["group_id"],),
        ).fetchall()
        _physical_attempt_group_kind(members)
        return {
            "group_id": str(group["group_id"]),
            "leader_cell_id": str(group["leader_cell_id"]),
            "leader_attempt": int(group["leader_attempt"]),
            "status": str(group["status"]),
            "shared_evidence_sha256": group["shared_evidence_sha256"],
            "members": tuple(
                {
                    "cell_id": str(row["cell_id"]),
                    "attempt": int(row["attempt"]),
                    "logical_kind": str(row["logical_kind"]),
                    "member_ordinal": int(row["member_ordinal"]),
                }
                for row in members
            ),
        }

    def physical_attempt_group_commands(
        self,
        group_id: str,
    ) -> tuple[QueuedCommandSpec, ...]:
        _require_text(group_id, "physical attempt group ID")
        rows = self._connection.execute(
            """
            SELECT q.*, m.logical_kind FROM command_queue AS q
            JOIN physical_attempt_group_members AS m
              ON m.cell_id = q.cell_id AND m.attempt = q.attempt
            WHERE m.group_id = ? ORDER BY m.member_ordinal
            """,
            (group_id,),
        ).fetchall()
        _physical_attempt_group_kind(rows)
        return tuple(_decoded_command(row) for row in rows)

    def physical_attempt_groups(self) -> tuple[dict[str, Any], ...]:
        groups = self._connection.execute(
            "SELECT * FROM physical_attempt_groups ORDER BY group_id"
        ).fetchall()
        output = []
        for group in groups:
            value = self.physical_attempt_group_for_attempt(
                str(group["leader_cell_id"]),
                int(group["leader_attempt"]),
            )
            if value is None:
                raise ExperimentOperatorError(
                    "physical attempt group leader lost its membership"
                )
            output.append(value)
        return tuple(output)

    def record_launch_compatibility_key(self, value: str) -> None:
        _require_text(value, "launch compatibility key")
        with self._transaction():
            self._set_metadata("last_launch_compatibility_key", value)

    def last_launch_compatibility_key(self) -> str | None:
        return self._metadata_value("last_launch_compatibility_key")

    def infrastructure_failure_count(self, cell_id: str) -> int:
        _require_text(cell_id, "infrastructure retry cell ID")
        return int(
            self._connection.execute(
                """
                SELECT COUNT(*) FROM cell_attempts
                WHERE cell_id = ? AND status = 'FAILED'
                  AND failure_code LIKE 'INFRASTRUCTURE:%'
                """,
                (cell_id,),
            ).fetchone()[0]
        )

    def import_legacy_stale_attempts(
        self, attempts: Sequence[LegacyStaleAttempt]
    ) -> int:
        """Import old attempts only as excluded ``STALE_IDENTITY`` evidence."""

        if not attempts:
            raise ValueError("legacy stale import cannot be empty")
        keys = tuple((item.spec.cell_id, item.spec.attempt) for item in attempts)
        if len(set(keys)) != len(keys):
            raise ValueError("legacy stale import contains duplicate attempt keys")
        now = self._now()
        with self._transaction():
            for item in attempts:
                spec = item.spec
                if not self._stage_phase_exists(spec.stage, spec.phase):
                    raise KeyError(f"unknown stage/phase {spec.stage}/{spec.phase}")
                finished = item.finished_at_ns or item.started_at_ns or now
                try:
                    self._connection.execute(
                        """
                        INSERT INTO cell_attempts (
                            cell_id, attempt, stage, phase, block_id, seed,
                            scientific_axes_json, identity_json, is_legacy_import,
                            legacy_original_status, status, command_sha256,
                            scientific_command_sha256,
                            output_directory, started_at_ns, finished_at_ns,
                            exit_code, terminal_sha256, junit_sha256, raw_log_sha256,
                            evidence_files_json, retry_decision,
                            included_in_analysis, exclusion_reason,
                            compute_gpu_seconds, reserved_gpu_seconds,
                            billed_gpu_seconds, created_at_ns, updated_at_ns
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 'STALE_IDENTITY', ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, 'RERUN_UNDER_FROZEN_IDENTITY',
                            0, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            spec.cell_id,
                            spec.attempt,
                            spec.stage,
                            spec.phase,
                            spec.block,
                            spec.seed,
                            _canonical_json(dict(spec.scientific_axes)),
                            _canonical_json(dict(spec.identity)),
                            item.original_status,
                            spec.command_sha256,
                            spec.scientific_command_sha256,
                            spec.output_directory,
                            item.started_at_ns,
                            finished,
                            item.exit_code,
                            item.terminal_sha256,
                            item.junit_sha256,
                            item.raw_log_sha256,
                            _canonical_json(dict(item.evidence_files or {})),
                            item.exclusion_reason,
                            float(item.compute_gpu_seconds),
                            float(item.reserved_gpu_seconds),
                            float(item.billed_gpu_seconds),
                            now,
                            now,
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise ExperimentOperatorError(
                        "legacy attempt key or output directory is already registered"
                    ) from error
                self._insert_event(
                    event_type="LEGACY_ATTEMPT_IMPORTED_STALE",
                    severity="WARNING",
                    cell_id=spec.cell_id,
                    attempt=spec.attempt,
                    payload={
                        "original_status": item.original_status,
                        "exclusion_reason": item.exclusion_reason,
                        "included_in_analysis": False,
                    },
                    occurred_at_ns=now,
                )
                self._touch_stage(spec.stage, spec.phase, now)
        return len(attempts)

    def mark_running_before_spawn(
        self,
        cell_id: str,
        attempt: int,
        *,
        assigned_gpu_uuids: Sequence[str],
        started_at_ns: int | None = None,
    ) -> None:
        """Commit ``RUNNING`` before any caller-owned process is spawned."""

        gpu_uuids = _validated_gpu_uuids(assigned_gpu_uuids)
        started = self._validated_time(started_at_ns)
        with self._transaction():
            row = self._require_attempt(cell_id, attempt)
            if row["status"] != "PENDING":
                raise AttemptTransitionError(
                    f"attempt must be PENDING before spawn, got {row['status']}"
                )
            self._connection.execute(
                """
                UPDATE cell_attempts
                SET status = 'RUNNING', assigned_gpu_uuids_json = ?,
                    started_at_ns = ?, heartbeat_at_ns = NULL,
                    heartbeat_sequence = 0, updated_at_ns = ?
                WHERE cell_id = ? AND attempt = ?
                """,
                (
                    _canonical_json(gpu_uuids),
                    started,
                    started,
                    cell_id,
                    attempt,
                ),
            )
            self._touch_stage(row["stage"], row["phase"], started)

    def start_attempt_with_launcher(
        self,
        cell_id: str,
        attempt: int,
        *,
        assigned_gpu_uuids: Sequence[str],
        launcher: Callable[[], SpawnedProcess],
        started_at_ns: int | None = None,
    ) -> SpawnedProcess:
        """Durably claim an attempt, then invoke an injected process launcher."""

        self.mark_running_before_spawn(
            cell_id,
            attempt,
            assigned_gpu_uuids=assigned_gpu_uuids,
            started_at_ns=started_at_ns,
        )
        try:
            process = launcher()
            if not isinstance(process, SpawnedProcess):
                raise TypeError("launcher must return SpawnedProcess")
        except BaseException as error:
            self.record_watchdog_event(
                event_type="SPAWN_OUTCOME_UNKNOWN",
                severity="CRITICAL",
                cell_id=cell_id,
                attempt=attempt,
                payload={"exception_type": type(error).__name__},
            )
            raise
        try:
            self.attach_process(
                cell_id,
                attempt,
                pid=process.pid,
                pgid=process.pgid,
                process_start_receipt_sha256=(process.process_start_receipt_sha256),
            )
        except BaseException as error:
            self.record_watchdog_event(
                event_type="PROCESS_METADATA_ATTACH_FAILED",
                severity="CRITICAL",
                cell_id=cell_id,
                attempt=attempt,
                payload={
                    "pid": process.pid,
                    "pgid": process.pgid,
                    "exception_type": type(error).__name__,
                },
            )
            raise
        return process

    def start_physical_attempt_group_with_launcher(
        self,
        group_id: str,
        *,
        assigned_gpu_uuids: Sequence[str],
        launcher: Callable[[], SpawnedProcess],
        started_at_ns: int | None = None,
    ) -> SpawnedProcess:
        """Commit every logical row before spawning its one shared parent."""

        _require_text(group_id, "physical attempt group ID")
        gpu_uuids = _validated_gpu_uuids(assigned_gpu_uuids)
        started = self._validated_time(started_at_ns)
        with self._transaction():
            group = self._connection.execute(
                "SELECT * FROM physical_attempt_groups WHERE group_id = ?",
                (group_id,),
            ).fetchone()
            if group is None:
                raise KeyError(f"unknown physical attempt group {group_id!r}")
            if group["status"] != "PENDING":
                raise AttemptTransitionError(
                    "physical attempt group must be PENDING before spawn"
                )
            members = self._connection.execute(
                """
                SELECT m.logical_kind, a.* FROM cell_attempts AS a
                JOIN physical_attempt_group_members AS m
                  ON m.cell_id = a.cell_id AND m.attempt = a.attempt
                WHERE m.group_id = ? ORDER BY m.member_ordinal
                """,
                (group_id,),
            ).fetchall()
            group_kind = _physical_attempt_group_kind(members)
            if any(row["status"] != "PENDING" for row in members):
                raise AttemptTransitionError(
                    "physical attempt group logical rows are not all PENDING"
                )
            expected_gpu_count = 2 if group_kind == "preflight_exact_ten" else 1
            if len(gpu_uuids) != expected_gpu_count:
                raise ValueError(
                    "physical attempt group GPU assignment differs from group kind"
                )
            for row in members:
                self._connection.execute(
                    """
                    UPDATE cell_attempts
                    SET status = 'RUNNING', assigned_gpu_uuids_json = ?,
                        started_at_ns = ?, heartbeat_at_ns = NULL,
                        heartbeat_sequence = 0, updated_at_ns = ?
                    WHERE cell_id = ? AND attempt = ?
                    """,
                    (
                        _canonical_json(gpu_uuids),
                        started,
                        started,
                        row["cell_id"],
                        row["attempt"],
                    ),
                )
            self._connection.execute(
                """
                UPDATE physical_attempt_groups
                SET status = 'RUNNING', updated_at_ns = ? WHERE group_id = ?
                """,
                (started, group_id),
            )
            self._touch_stage(
                str(members[0]["stage"]), str(members[0]["phase"]), started
            )
        try:
            process = launcher()
            if type(process) is not SpawnedProcess:
                raise TypeError("launcher must return SpawnedProcess")
        except BaseException as error:
            self.record_watchdog_event(
                event_type="PHYSICAL_GROUP_SPAWN_OUTCOME_UNKNOWN",
                severity="CRITICAL",
                cell_id=str(group["leader_cell_id"]),
                attempt=int(group["leader_attempt"]),
                payload={
                    "group_id": group_id,
                    "exception_type": type(error).__name__,
                },
            )
            raise
        try:
            self.attach_physical_attempt_group_process(
                group_id,
                pid=process.pid,
                pgid=process.pgid,
                process_start_receipt_sha256=(process.process_start_receipt_sha256),
            )
        except BaseException as error:
            self.record_watchdog_event(
                event_type="PHYSICAL_GROUP_PROCESS_METADATA_ATTACH_FAILED",
                severity="CRITICAL",
                cell_id=str(group["leader_cell_id"]),
                attempt=int(group["leader_attempt"]),
                payload={
                    "group_id": group_id,
                    "pid": process.pid,
                    "pgid": process.pgid,
                    "exception_type": type(error).__name__,
                },
            )
            raise
        return process

    def attach_physical_attempt_group_process(
        self,
        group_id: str,
        *,
        pid: int,
        pgid: int,
        process_start_receipt_sha256: str | None = None,
    ) -> None:
        _require_text(group_id, "physical attempt group ID")
        _require_positive_int(pid, "PID")
        _require_positive_int(pgid, "PGID")
        if process_start_receipt_sha256 is not None:
            _require_sha256(
                process_start_receipt_sha256,
                "physical group process start receipt SHA-256",
            )
        now = self._now()
        with self._transaction():
            group = self._connection.execute(
                "SELECT * FROM physical_attempt_groups WHERE group_id = ?",
                (group_id,),
            ).fetchone()
            if group is None or group["status"] != "RUNNING":
                raise AttemptTransitionError(
                    "physical group process metadata requires RUNNING status"
                )
            members = self._connection.execute(
                """
                SELECT m.logical_kind, a.* FROM cell_attempts AS a
                JOIN physical_attempt_group_members AS m
                  ON m.cell_id = a.cell_id AND m.attempt = a.attempt
                WHERE m.group_id = ? ORDER BY m.member_ordinal
                """,
                (group_id,),
            ).fetchall()
            _physical_attempt_group_kind(members)
            if any(row["status"] != "RUNNING" for row in members):
                raise AttemptTransitionError(
                    "physical group process metadata lacks all RUNNING rows"
                )
            for row in members:
                if row["pid"] is not None or row["pgid"] is not None:
                    if (row["pid"], row["pgid"]) == (pid, pgid):
                        registered = row["process_start_receipt_sha256"]
                        if (
                            process_start_receipt_sha256 is not None
                            and registered is not None
                            and registered != process_start_receipt_sha256
                        ):
                            raise AttemptTransitionError(
                                "physical group start receipt digest is immutable"
                            )
                        if (
                            registered is None
                            and process_start_receipt_sha256 is not None
                        ):
                            self._connection.execute(
                                """
                                UPDATE cell_attempts
                                SET process_start_receipt_sha256 = ?, updated_at_ns = ?
                                WHERE cell_id = ? AND attempt = ?
                                """,
                                (
                                    process_start_receipt_sha256,
                                    now,
                                    row["cell_id"],
                                    row["attempt"],
                                ),
                            )
                        continue
                    raise AttemptTransitionError(
                        "physical group process metadata is immutable"
                    )
                self._connection.execute(
                    """
                    UPDATE cell_attempts
                    SET pid = ?, pgid = ?, process_start_receipt_sha256 = ?,
                        updated_at_ns = ?
                    WHERE cell_id = ? AND attempt = ?
                    """,
                    (
                        pid,
                        pgid,
                        process_start_receipt_sha256,
                        now,
                        row["cell_id"],
                        row["attempt"],
                    ),
                )

    def attach_process(
        self,
        cell_id: str,
        attempt: int,
        *,
        pid: int,
        pgid: int,
        process_start_receipt_sha256: str | None = None,
    ) -> None:
        _require_positive_int(pid, "PID")
        _require_positive_int(pgid, "PGID")
        if process_start_receipt_sha256 is not None:
            _require_sha256(
                process_start_receipt_sha256,
                "process start receipt SHA-256",
            )
        now = self._now()
        with self._transaction():
            row = self._require_attempt(cell_id, attempt)
            if row["status"] != "RUNNING":
                raise AttemptTransitionError("process metadata requires RUNNING status")
            if row["pid"] is not None or row["pgid"] is not None:
                if (row["pid"], row["pgid"]) == (pid, pgid):
                    registered = row["process_start_receipt_sha256"]
                    if (
                        process_start_receipt_sha256 is not None
                        and registered is not None
                        and registered != process_start_receipt_sha256
                    ):
                        raise AttemptTransitionError(
                            "process start receipt digest is immutable"
                        )
                    if registered is None and process_start_receipt_sha256 is not None:
                        self._connection.execute(
                            """
                            UPDATE cell_attempts
                            SET process_start_receipt_sha256 = ?, updated_at_ns = ?
                            WHERE cell_id = ? AND attempt = ?
                            """,
                            (process_start_receipt_sha256, now, cell_id, attempt),
                        )
                    return
                raise AttemptTransitionError("attempt process metadata is immutable")
            self._connection.execute(
                """
                UPDATE cell_attempts
                SET pid = ?, pgid = ?, process_start_receipt_sha256 = ?,
                    updated_at_ns = ?
                WHERE cell_id = ? AND attempt = ?
                """,
                (
                    pid,
                    pgid,
                    process_start_receipt_sha256,
                    now,
                    cell_id,
                    attempt,
                ),
            )

    def record_heartbeat(
        self,
        cell_id: str,
        attempt: int,
        *,
        pid: int,
        pgid: int,
        log_size_bytes: int,
        gpu_observation: Mapping[str, Any],
        observed_at_ns: int | None = None,
        heartbeat_sequence: int | None = None,
    ) -> None:
        _require_positive_int(pid, "heartbeat PID")
        _require_positive_int(pgid, "heartbeat PGID")
        if isinstance(log_size_bytes, bool) or log_size_bytes < 0:
            raise ValueError("heartbeat log size must be non-negative")
        if heartbeat_sequence is not None:
            _require_positive_int(heartbeat_sequence, "heartbeat sequence")
        observation_json = _canonical_json(
            _canonical_mapping(gpu_observation, "GPU observation", allow_empty=False)
        )
        observed = self._validated_time(observed_at_ns)
        with self._transaction():
            row = self._require_attempt(cell_id, attempt)
            if row["status"] != "RUNNING":
                raise AttemptTransitionError("heartbeat requires RUNNING status")
            if (row["pid"], row["pgid"]) != (pid, pgid):
                raise AttemptTransitionError(
                    "heartbeat PID/PGID differs from assignment"
                )
            if row["heartbeat_at_ns"] is not None and observed <= int(
                row["heartbeat_at_ns"]
            ):
                raise AttemptTransitionError(
                    "heartbeat timestamps must increase monotonically"
                )
            next_sequence = (
                int(row["heartbeat_sequence"]) + 1
                if heartbeat_sequence is None
                else heartbeat_sequence
            )
            if next_sequence <= int(row["heartbeat_sequence"]):
                raise AttemptTransitionError(
                    "heartbeat sequence must increase monotonically"
                )
            previous_size = row["last_log_size_bytes"]
            if previous_size is not None and log_size_bytes < previous_size:
                raise AttemptTransitionError("heartbeat log size cannot decrease")
            growth_ns = row["last_log_growth_ns"]
            if previous_size is None or log_size_bytes > previous_size:
                growth_ns = observed
            self._connection.execute(
                """
                UPDATE cell_attempts
                SET heartbeat_at_ns = ?, heartbeat_sequence = ?,
                    last_log_size_bytes = ?, last_log_growth_ns = ?,
                    gpu_observation_json = ?, updated_at_ns = ?
                WHERE cell_id = ? AND attempt = ?
                """,
                (
                    observed,
                    next_sequence,
                    log_size_bytes,
                    growth_ns,
                    observation_json,
                    observed,
                    cell_id,
                    attempt,
                ),
            )

    def record_physical_attempt_group_heartbeat(
        self,
        group_id: str,
        *,
        pid: int,
        pgid: int,
        log_size_bytes: int,
        gpu_observation: Mapping[str, Any],
        observed_at_ns: int | None = None,
        heartbeat_sequence: int | None = None,
    ) -> None:
        """Record one parent observation on every logical member atomically."""

        _require_text(group_id, "physical attempt group ID")
        _require_positive_int(pid, "heartbeat PID")
        _require_positive_int(pgid, "heartbeat PGID")
        if isinstance(log_size_bytes, bool) or log_size_bytes < 0:
            raise ValueError("heartbeat log size must be non-negative")
        if heartbeat_sequence is not None:
            _require_positive_int(heartbeat_sequence, "heartbeat sequence")
        observation_json = _canonical_json(
            _canonical_mapping(gpu_observation, "GPU observation", allow_empty=False)
        )
        observed = self._validated_time(observed_at_ns)
        with self._transaction():
            rows = self._connection.execute(
                """
                SELECT m.logical_kind, a.* FROM cell_attempts AS a
                JOIN physical_attempt_group_members AS m
                  ON m.cell_id = a.cell_id AND m.attempt = a.attempt
                WHERE m.group_id = ? ORDER BY m.member_ordinal
                """,
                (group_id,),
            ).fetchall()
            _physical_attempt_group_kind(rows)
            for row in rows:
                if row["status"] != "RUNNING" or (row["pid"], row["pgid"]) != (
                    pid,
                    pgid,
                ):
                    raise AttemptTransitionError(
                        "physical group heartbeat differs from RUNNING parent"
                    )
                if row["heartbeat_at_ns"] is not None and observed <= int(
                    row["heartbeat_at_ns"]
                ):
                    raise AttemptTransitionError(
                        "heartbeat timestamps must increase monotonically"
                    )
                next_sequence = (
                    int(row["heartbeat_sequence"]) + 1
                    if heartbeat_sequence is None
                    else heartbeat_sequence
                )
                if next_sequence <= int(row["heartbeat_sequence"]):
                    raise AttemptTransitionError(
                        "heartbeat sequence must increase monotonically"
                    )
                previous_size = row["last_log_size_bytes"]
                if previous_size is not None and log_size_bytes < previous_size:
                    raise AttemptTransitionError("heartbeat log size cannot decrease")
                growth_ns = row["last_log_growth_ns"]
                if previous_size is None or log_size_bytes > previous_size:
                    growth_ns = observed
                self._connection.execute(
                    """
                    UPDATE cell_attempts
                    SET heartbeat_at_ns = ?,
                        heartbeat_sequence = ?,
                        last_log_size_bytes = ?, last_log_growth_ns = ?,
                        gpu_observation_json = ?, updated_at_ns = ?
                    WHERE cell_id = ? AND attempt = ?
                    """,
                    (
                        observed,
                        next_sequence,
                        log_size_bytes,
                        growth_ns,
                        observation_json,
                        observed,
                        row["cell_id"],
                        row["attempt"],
                    ),
                )

    def record_runtime_observation(
        self,
        cell_id: str,
        attempt: int,
        *,
        pid: int,
        pgid: int,
        log_size_bytes: int,
        gpu_observation: Mapping[str, Any],
        observed_at_ns: int | None = None,
    ) -> None:
        """Persist scheduler diagnostics without manufacturing a heartbeat."""

        _require_positive_int(pid, "runtime observation PID")
        _require_positive_int(pgid, "runtime observation PGID")
        if isinstance(log_size_bytes, bool) or log_size_bytes < 0:
            raise ValueError("runtime observation log size must be non-negative")
        observation_json = _canonical_json(
            _canonical_mapping(gpu_observation, "GPU observation", allow_empty=False)
        )
        observed = self._validated_time(observed_at_ns)
        with self._transaction():
            row = self._require_attempt(cell_id, attempt)
            if row["status"] != "RUNNING" or (row["pid"], row["pgid"]) != (
                pid,
                pgid,
            ):
                raise AttemptTransitionError(
                    "runtime observation differs from RUNNING assignment"
                )
            previous_size = row["last_log_size_bytes"]
            if previous_size is not None and log_size_bytes < previous_size:
                raise AttemptTransitionError("runtime observation log size decreased")
            growth_ns = row["last_log_growth_ns"]
            if previous_size is None or log_size_bytes > previous_size:
                growth_ns = observed
            self._connection.execute(
                """
                UPDATE cell_attempts
                SET last_log_size_bytes = ?, last_log_growth_ns = ?,
                    gpu_observation_json = ?, updated_at_ns = ?
                WHERE cell_id = ? AND attempt = ?
                """,
                (
                    log_size_bytes,
                    growth_ns,
                    observation_json,
                    observed,
                    cell_id,
                    attempt,
                ),
            )

    def record_physical_attempt_group_runtime_observation(
        self,
        group_id: str,
        *,
        pid: int,
        pgid: int,
        log_size_bytes: int,
        gpu_observation: Mapping[str, Any],
        observed_at_ns: int | None = None,
    ) -> None:
        """Fan one scheduler diagnostic sample across a shared parent."""

        _require_text(group_id, "physical attempt group ID")
        _require_positive_int(pid, "runtime observation PID")
        _require_positive_int(pgid, "runtime observation PGID")
        if isinstance(log_size_bytes, bool) or log_size_bytes < 0:
            raise ValueError("runtime observation log size must be non-negative")
        observation_json = _canonical_json(
            _canonical_mapping(gpu_observation, "GPU observation", allow_empty=False)
        )
        observed = self._validated_time(observed_at_ns)
        with self._transaction():
            rows = self._connection.execute(
                """
                SELECT m.logical_kind, a.* FROM cell_attempts AS a
                JOIN physical_attempt_group_members AS m
                  ON m.cell_id = a.cell_id AND m.attempt = a.attempt
                WHERE m.group_id = ? ORDER BY m.member_ordinal
                """,
                (group_id,),
            ).fetchall()
            _physical_attempt_group_kind(rows)
            for row in rows:
                if row["status"] != "RUNNING" or (row["pid"], row["pgid"]) != (
                    pid,
                    pgid,
                ):
                    raise AttemptTransitionError(
                        "physical group runtime observation differs from parent"
                    )
                previous_size = row["last_log_size_bytes"]
                if previous_size is not None and log_size_bytes < previous_size:
                    raise AttemptTransitionError(
                        "physical group runtime observation log size decreased"
                    )
                growth_ns = row["last_log_growth_ns"]
                if previous_size is None or log_size_bytes > previous_size:
                    growth_ns = observed
                self._connection.execute(
                    """
                    UPDATE cell_attempts
                    SET last_log_size_bytes = ?, last_log_growth_ns = ?,
                        gpu_observation_json = ?, updated_at_ns = ?
                    WHERE cell_id = ? AND attempt = ?
                    """,
                    (
                        log_size_bytes,
                        growth_ns,
                        observation_json,
                        observed,
                        row["cell_id"],
                        row["attempt"],
                    ),
                )

    def finish_attempt(
        self,
        cell_id: str,
        attempt: int,
        *,
        status: TerminalAttemptStatus,
        exit_code: int | None,
        terminal_sha256: str | None = None,
        junit_sha256: str | None = None,
        raw_log_sha256: str | None = None,
        evidence_files: Mapping[str, str] | None = None,
        failure_code: str | None = None,
        retry_decision: str | None = None,
        included_in_analysis: bool,
        exclusion_reason: str | None,
        compute_gpu_seconds: float = 0.0,
        reserved_gpu_seconds: float = 0.0,
        billed_gpu_seconds: float = 0.0,
        finished_at_ns: int | None = None,
    ) -> None:
        if status not in TERMINAL_ATTEMPT_STATUSES:
            raise ValueError("finish status must be terminal")
        if exit_code is not None and isinstance(exit_code, bool):
            raise ValueError("exit code must be an integer or null")
        if not isinstance(exit_code, (int, type(None))):
            raise TypeError("exit code must be an integer or null")
        for label, value in (
            ("terminal SHA-256", terminal_sha256),
            ("JUnit SHA-256", junit_sha256),
            ("raw log SHA-256", raw_log_sha256),
        ):
            if value is not None:
                _require_sha256(value, label)
        evidence = dict(evidence_files or {})
        for path, digest in evidence.items():
            _require_text(path, "evidence file path")
            _require_sha256(digest, f"evidence SHA-256 for {path}")
        for label, value in (
            ("compute GPU-seconds", compute_gpu_seconds),
            ("reserved GPU-seconds", reserved_gpu_seconds),
            ("billed GPU-seconds", billed_gpu_seconds),
        ):
            _require_nonnegative_finite(value, label)
        if status == "COMPLETE":
            if exit_code != 0:
                raise ValueError("COMPLETE attempts require exit code zero")
            if None in (terminal_sha256, junit_sha256, raw_log_sha256):
                raise ValueError(
                    "COMPLETE attempts require terminal, JUnit, and raw-log SHA-256"
                )
            if failure_code is not None or retry_decision is not None:
                raise ValueError(
                    "COMPLETE attempts cannot carry failure or retry codes"
                )
            if included_in_analysis == (exclusion_reason is not None):
                raise ValueError(
                    "COMPLETE analysis inclusion and exclusion reason disagree"
                )
        else:
            if included_in_analysis:
                raise ValueError("non-complete attempts cannot enter analysis")
            _require_text(exclusion_reason, "terminal exclusion reason")
            if status in {"FAILED", "BLOCKED"}:
                _require_text(failure_code, "failure code")
                _require_text(retry_decision, "retry decision")
            elif status == "STALE_IDENTITY" and retry_decision is not None:
                _require_text(retry_decision, "retry decision")
        finished = self._validated_time(finished_at_ns)
        with self._transaction():
            row = self._require_attempt(cell_id, attempt)
            if row["status"] not in {"PENDING", "RUNNING"}:
                raise AttemptTransitionError(
                    f"attempt is already terminal with status {row['status']}"
                )
            if status == "COMPLETE" and row["status"] != "RUNNING":
                raise AttemptTransitionError("COMPLETE requires a RUNNING attempt")
            if row["started_at_ns"] is not None and finished < row["started_at_ns"]:
                raise ValueError("finished time precedes started time")
            if row["heartbeat_at_ns"] is not None and finished < row["heartbeat_at_ns"]:
                raise ValueError("finished time precedes latest heartbeat")
            self._connection.execute(
                """
                UPDATE cell_attempts
                SET status = ?, finished_at_ns = ?, exit_code = ?,
                    terminal_sha256 = ?, junit_sha256 = ?, raw_log_sha256 = ?,
                    evidence_files_json = ?, failure_code = ?, retry_decision = ?,
                    included_in_analysis = ?, exclusion_reason = ?,
                    compute_gpu_seconds = ?, reserved_gpu_seconds = ?,
                    billed_gpu_seconds = ?, updated_at_ns = ?
                WHERE cell_id = ? AND attempt = ?
                """,
                (
                    status,
                    finished,
                    exit_code,
                    terminal_sha256,
                    junit_sha256,
                    raw_log_sha256,
                    _canonical_json(evidence),
                    failure_code,
                    retry_decision,
                    int(included_in_analysis),
                    exclusion_reason,
                    float(compute_gpu_seconds),
                    float(reserved_gpu_seconds),
                    float(billed_gpu_seconds),
                    finished,
                    cell_id,
                    attempt,
                ),
            )
            self._touch_stage(row["stage"], row["phase"], finished)

    def finish_physical_attempt_group(
        self,
        group_id: str,
        *,
        terminals: Mapping[str, TerminalEvidence],
        finished_at_ns: int | None = None,
    ) -> None:
        """Atomically fan one validated parent result into all logical terminals."""

        _require_text(group_id, "physical attempt group ID")
        if not isinstance(terminals, Mapping) or any(
            type(value) is not TerminalEvidence for value in terminals.values()
        ):
            raise TypeError("physical group terminals require exact terminal evidence")
        terminal_by_cell = dict(terminals)
        finished = self._validated_time(finished_at_ns)
        with self._transaction():
            group = self._connection.execute(
                "SELECT * FROM physical_attempt_groups WHERE group_id = ?",
                (group_id,),
            ).fetchone()
            if group is None:
                raise KeyError(f"unknown physical attempt group {group_id!r}")
            if group["status"] != "RUNNING":
                raise AttemptTransitionError(
                    "physical attempt group terminal requires RUNNING status"
                )
            rows = self._connection.execute(
                """
                SELECT a.*, m.member_ordinal, m.logical_kind
                FROM cell_attempts AS a
                JOIN physical_attempt_group_members AS m
                  ON m.cell_id = a.cell_id AND m.attempt = a.attempt
                WHERE m.group_id = ? ORDER BY m.member_ordinal
                """,
                (group_id,),
            ).fetchall()
            group_kind = _physical_attempt_group_kind(rows)
            expected_cells = tuple(str(row["cell_id"]) for row in rows)
            if set(terminal_by_cell) != set(expected_cells):
                raise ValueError(
                    "physical attempt group terminal coverage is incomplete"
                )
            if any(row["status"] != "RUNNING" for row in rows):
                raise AttemptTransitionError(
                    "physical attempt group logical row is not RUNNING"
                )
            shared_evidence = {
                terminal_by_cell[cell_id].atomic_publication_sha256
                for cell_id in expected_cells
            }
            if len(shared_evidence) != 1:
                raise ValueError(
                    "physical attempt group terminals lack one shared publication"
                )
            timings = tuple(
                (
                    terminal_by_cell[cell_id].started_ns,
                    terminal_by_cell[cell_id].finished_ns,
                )
                for cell_id in expected_cells
            )
            if any(start is None or end is None for start, end in timings):
                raise ValueError(
                    "physical attempt group terminals require durable lifecycle timing"
                )
            evidence_started = min(int(start) for start, _end in timings)
            evidence_finished = max(int(end) for _start, end in timings)
            if evidence_finished > finished:
                raise ValueError("physical group evidence finishes after ledger time")
            leader_cell_id = str(group["leader_cell_id"])
            gpu_count = len(json.loads(rows[0]["assigned_gpu_uuids_json"]))
            physical_gpu_seconds = (
                (evidence_finished - evidence_started) / 1e9 * gpu_count
            )
            group_status = (
                "COMPLETE"
                if all(
                    terminal_by_cell[cell_id].status == "COMPLETE"
                    for cell_id in expected_cells
                )
                else "FAILED"
            )
            for row in rows:
                cell_id = str(row["cell_id"])
                terminal = terminal_by_cell[cell_id]
                if row["heartbeat_at_ns"] is not None and finished < int(
                    row["heartbeat_at_ns"]
                ):
                    raise ValueError(
                        "physical group ledger finish precedes latest heartbeat"
                    )
                failure_code = None
                retry_decision = None
                if terminal.status == "FAILED":
                    assert terminal.failure_class is not None
                    failure_code = f"{terminal.failure_class}:{terminal.failure_code}"
                    retry_decision = (
                        "NO_BLIND_GROUP_RETRY"
                        if group_kind == "preflight_exact_ten"
                        else "RETRY_INFRASTRUCTURE_AUTOMATIC"
                        if terminal.failure_class == "INFRASTRUCTURE"
                        else "NO_SCIENTIFIC_RETRY"
                    )
                self._connection.execute(
                    """
                    UPDATE cell_attempts
                    SET status = ?, finished_at_ns = ?, exit_code = ?,
                        terminal_sha256 = ?, junit_sha256 = ?, raw_log_sha256 = ?,
                        evidence_files_json = ?, failure_code = ?, retry_decision = ?,
                        included_in_analysis = ?, exclusion_reason = ?,
                        compute_gpu_seconds = ?, reserved_gpu_seconds = ?,
                        billed_gpu_seconds = 0, updated_at_ns = ?
                    WHERE cell_id = ? AND attempt = ?
                    """,
                    (
                        terminal.status,
                        finished,
                        terminal.exit_code,
                        terminal.terminal_sha256,
                        terminal.junit_sha256,
                        terminal.raw_log_sha256,
                        _canonical_json(dict(terminal.evidence_files or {})),
                        failure_code,
                        retry_decision,
                        int(terminal.included_in_analysis),
                        terminal.exclusion_reason,
                        physical_gpu_seconds if cell_id == leader_cell_id else 0.0,
                        physical_gpu_seconds if cell_id == leader_cell_id else 0.0,
                        finished,
                        cell_id,
                        row["attempt"],
                    ),
                )
            shared = next(iter(shared_evidence))
            self._connection.execute(
                """
                UPDATE physical_attempt_groups
                SET status = ?, shared_evidence_sha256 = ?, updated_at_ns = ?
                WHERE group_id = ?
                """,
                (group_status, shared, finished, group_id),
            )
            self._touch_stage(str(rows[0]["stage"]), str(rows[0]["phase"]), finished)
            self._insert_event(
                event_type="PHYSICAL_ATTEMPT_GROUP_TERMINAL_ACCEPTED",
                severity="INFO" if group_status == "COMPLETE" else "ERROR",
                cell_id=leader_cell_id,
                attempt=int(group["leader_attempt"]),
                payload={
                    "group_id": group_id,
                    "group_kind": group_kind,
                    "logical_attempt_count": len(rows),
                    "physical_accounting_owner": leader_cell_id,
                    "physical_gpu_seconds": physical_gpu_seconds,
                    "shared_evidence_sha256": shared,
                    "status": group_status,
                },
                occurred_at_ns=finished,
            )

    def fail_physical_attempt_group_spawn(
        self,
        group_id: str,
        *,
        exception_type: str,
        finished_at_ns: int | None = None,
    ) -> None:
        """Terminalize every logical row after one parent spawn failure."""

        _require_text(group_id, "physical attempt group ID")
        _require_text(exception_type, "physical group spawn exception type")
        finished = self._validated_time(finished_at_ns)
        with self._transaction():
            group = self._connection.execute(
                "SELECT * FROM physical_attempt_groups WHERE group_id = ?",
                (group_id,),
            ).fetchone()
            if group is None or group["status"] != "RUNNING":
                raise AttemptTransitionError(
                    "physical group spawn failure requires RUNNING status"
                )
            rows = self._connection.execute(
                """
                SELECT m.logical_kind, a.* FROM cell_attempts AS a
                JOIN physical_attempt_group_members AS m
                  ON m.cell_id = a.cell_id AND m.attempt = a.attempt
                WHERE m.group_id = ? ORDER BY m.member_ordinal
                """,
                (group_id,),
            ).fetchall()
            group_kind = _physical_attempt_group_kind(rows)
            if any(row["status"] != "RUNNING" for row in rows):
                raise AttemptTransitionError(
                    "physical group spawn failure coverage is not all RUNNING"
                )
            retry_decision = (
                "NO_BLIND_GROUP_RETRY"
                if group_kind == "preflight_exact_ten"
                else "RETRY_INFRASTRUCTURE_AUTOMATIC"
            )
            for row in rows:
                self._connection.execute(
                    """
                    UPDATE cell_attempts
                    SET status = 'FAILED', finished_at_ns = ?, exit_code = NULL,
                        failure_code = 'INFRASTRUCTURE:SPAWN_FAILED',
                        retry_decision = ?,
                        included_in_analysis = 0, exclusion_reason = ?,
                        compute_gpu_seconds = 0, reserved_gpu_seconds = 0,
                        billed_gpu_seconds = 0, updated_at_ns = ?
                    WHERE cell_id = ? AND attempt = ?
                    """,
                    (
                        finished,
                        retry_decision,
                        f"spawn_failed:{exception_type}",
                        finished,
                        row["cell_id"],
                        row["attempt"],
                    ),
                )
            self._connection.execute(
                """
                UPDATE physical_attempt_groups
                SET status = 'FAILED', updated_at_ns = ? WHERE group_id = ?
                """,
                (finished, group_id),
            )
            self._touch_stage(str(rows[0]["stage"]), str(rows[0]["phase"]), finished)

    def fail_physical_attempt_group_infrastructure(
        self,
        group_id: str,
        *,
        failure_code: str,
        exclusion_reason: str,
        evidence_files: Mapping[str, str] | None = None,
        finished_at_ns: int | None = None,
    ) -> None:
        """Terminalize a dead shared process lease without fake terminals."""

        _require_text(group_id, "physical attempt group ID")
        _require_text(failure_code, "physical group infrastructure failure code")
        if not failure_code.startswith("INFRASTRUCTURE:"):
            raise ValueError("physical group failure must be infrastructure-classified")
        _require_text(exclusion_reason, "physical group exclusion reason")
        evidence = dict(evidence_files or {})
        for path, digest in evidence.items():
            _require_text(path, "physical group partial evidence path")
            _require_sha256(
                digest, f"physical group partial evidence SHA-256 for {path}"
            )
        finished = self._validated_time(finished_at_ns)
        with self._transaction():
            group = self._connection.execute(
                "SELECT * FROM physical_attempt_groups WHERE group_id = ?",
                (group_id,),
            ).fetchone()
            if group is None or group["status"] != "RUNNING":
                raise AttemptTransitionError(
                    "physical group infrastructure failure requires RUNNING status"
                )
            rows = self._connection.execute(
                """
                SELECT m.logical_kind, a.* FROM cell_attempts AS a
                JOIN physical_attempt_group_members AS m
                  ON m.cell_id = a.cell_id AND m.attempt = a.attempt
                WHERE m.group_id = ? ORDER BY m.member_ordinal
                """,
                (group_id,),
            ).fetchall()
            group_kind = _physical_attempt_group_kind(rows)
            if any(row["status"] != "RUNNING" for row in rows):
                raise AttemptTransitionError(
                    "physical group infrastructure coverage is not all RUNNING"
                )
            leader_cell_id = str(group["leader_cell_id"])
            leader_started = next(
                int(row["started_at_ns"])
                for row in rows
                if row["cell_id"] == leader_cell_id
            )
            gpu_count = len(json.loads(rows[0]["assigned_gpu_uuids_json"]))
            physical_gpu_seconds = (
                max(0.0, (finished - leader_started) / 1e9) * gpu_count
            )
            evidence_json = _canonical_json(evidence)
            retry_decision = (
                "NO_BLIND_GROUP_RETRY"
                if group_kind == "preflight_exact_ten"
                else "RETRY_INFRASTRUCTURE_AUTOMATIC"
            )
            for row in rows:
                owner = row["cell_id"] == leader_cell_id
                self._connection.execute(
                    """
                    UPDATE cell_attempts
                    SET status = 'FAILED', finished_at_ns = ?, exit_code = NULL,
                        evidence_files_json = ?, failure_code = ?,
                        retry_decision = ?,
                        included_in_analysis = 0, exclusion_reason = ?,
                        compute_gpu_seconds = ?, reserved_gpu_seconds = ?,
                        billed_gpu_seconds = 0, updated_at_ns = ?
                    WHERE cell_id = ? AND attempt = ?
                    """,
                    (
                        finished,
                        evidence_json,
                        failure_code,
                        retry_decision,
                        exclusion_reason,
                        physical_gpu_seconds if owner else 0.0,
                        physical_gpu_seconds if owner else 0.0,
                        finished,
                        row["cell_id"],
                        row["attempt"],
                    ),
                )
            self._connection.execute(
                """
                UPDATE physical_attempt_groups
                SET status = 'FAILED', updated_at_ns = ? WHERE group_id = ?
                """,
                (finished, group_id),
            )
            self._touch_stage(str(rows[0]["stage"]), str(rows[0]["phase"]), finished)

    def mark_stale_identity(
        self,
        cell_id: str,
        attempt: int,
        *,
        reason: str,
        retry_decision: str = "RERUN_UNDER_FROZEN_IDENTITY",
        marked_at_ns: int | None = None,
    ) -> None:
        _require_text(reason, "stale identity reason")
        _require_text(retry_decision, "stale identity retry decision")
        marked = self._validated_time(marked_at_ns)
        with self._transaction():
            row = self._require_attempt(cell_id, attempt)
            if row["status"] == "STALE_IDENTITY":
                if row["exclusion_reason"] == reason:
                    return
                raise AttemptTransitionError("stale identity reason is immutable")
            if row["status"] == "RUNNING":
                raise AttemptTransitionError(
                    "stop and terminalize a running process before marking it stale"
                )
            if marked < int(row["updated_at_ns"]):
                raise ValueError("stale identity time precedes latest attempt update")
            prior_status = str(row["status"])
            self._connection.execute(
                """
                UPDATE cell_attempts
                SET status = 'STALE_IDENTITY', included_in_analysis = 0,
                    exclusion_reason = ?, retry_decision = ?,
                    finished_at_ns = COALESCE(finished_at_ns, ?), updated_at_ns = ?
                WHERE cell_id = ? AND attempt = ?
                """,
                (reason, retry_decision, marked, marked, cell_id, attempt),
            )
            self._insert_event(
                event_type="IDENTITY_MARKED_STALE",
                severity="WARNING",
                cell_id=cell_id,
                attempt=attempt,
                payload={"prior_status": prior_status, "reason": reason},
                occurred_at_ns=marked,
            )
            self._touch_stage(row["stage"], row["phase"], marked)

    def record_watchdog_event(
        self,
        *,
        event_type: str,
        severity: WatchdogSeverity,
        payload: Mapping[str, Any],
        cell_id: str | None = None,
        attempt: int | None = None,
        occurred_at_ns: int | None = None,
    ) -> int:
        _require_text(event_type, "watchdog event type")
        if severity not in WATCHDOG_SEVERITIES:
            raise ValueError("unsupported watchdog severity")
        body = _canonical_mapping(payload, "watchdog payload", allow_empty=True)
        if (cell_id is None) != (attempt is None):
            raise ValueError(
                "watchdog cell ID and attempt must be both present or absent"
            )
        occurred = self._validated_time(occurred_at_ns)
        with self._transaction():
            if cell_id is not None and attempt is not None:
                self._require_attempt(cell_id, attempt)
            return self._insert_event(
                event_type=event_type,
                severity=severity,
                cell_id=cell_id,
                attempt=attempt,
                payload=body,
                occurred_at_ns=occurred,
            )

    def watchdog_once(
        self,
        *,
        policy: WatchdogPolicy,
        process_probe: Callable[[int, int], ProcessObservation] | None = None,
        monitored_path: str | Path | None = None,
        now_ns: int | None = None,
    ) -> tuple[WatchdogFinding, ...]:
        """Inspect current attempts and record anomalies without sending signals."""

        probe = process_probe or inspect_local_process
        now = self._validated_time(now_ns)
        findings: list[WatchdogFinding] = []
        rows = self._connection.execute(
            "SELECT * FROM cell_attempts WHERE status = 'RUNNING' ORDER BY cell_id"
        ).fetchall()
        for row in rows:
            if now < int(row["started_at_ns"]):
                raise ExperimentOperatorError(
                    "watchdog time precedes a RUNNING attempt start"
                )
            age_seconds = (now - int(row["started_at_ns"])) / 1e9
            if row["pid"] is None or row["pgid"] is None:
                if age_seconds > policy.process_attach_grace_seconds:
                    finding = self._record_watchdog_finding_once(
                        event_type="PROCESS_NOT_ATTACHED",
                        severity="CRITICAL",
                        row=row,
                        payload={"running_age_seconds": age_seconds},
                        now_ns=now,
                        repeat_seconds=policy.event_repeat_seconds,
                    )
                    if finding is not None:
                        findings.append(finding)
            else:
                observation = probe(int(row["pid"]), int(row["pgid"]))
                if observation.pid != row["pid"]:
                    raise ExperimentOperatorError(
                        "process probe returned a different PID"
                    )
                if not observation.alive or observation.observed_pgid != row["pgid"]:
                    finding = self._record_watchdog_finding_once(
                        event_type="PROCESS_NOT_ALIVE",
                        severity="CRITICAL",
                        row=row,
                        payload={
                            "expected_pgid": row["pgid"],
                            "observed_pgid": observation.observed_pgid,
                            "reason": observation.reason,
                            "exit_code": observation.exit_code,
                        },
                        now_ns=now,
                        repeat_seconds=policy.event_repeat_seconds,
                    )
                    if finding is not None:
                        findings.append(finding)
            heartbeat_at_ns = row["heartbeat_at_ns"]
            heartbeat_age = (
                None if heartbeat_at_ns is None else (now - int(heartbeat_at_ns)) / 1e9
            )
            if (
                heartbeat_age is not None
                and heartbeat_age > policy.heartbeat_timeout_seconds
            ):
                finding = self._record_watchdog_finding_once(
                    event_type="HEARTBEAT_STALE",
                    severity="WARNING",
                    row=row,
                    payload={
                        "heartbeat_age_seconds": heartbeat_age,
                        "automatic_signal": False,
                    },
                    now_ns=now,
                    repeat_seconds=policy.event_repeat_seconds,
                )
                if finding is not None:
                    findings.append(finding)
            growth_ns = row["last_log_growth_ns"]
            if growth_ns is not None:
                log_stall = (now - int(growth_ns)) / 1e9
                if log_stall > policy.log_stall_timeout_seconds:
                    finding = self._record_watchdog_finding_once(
                        event_type="LOG_STALLED",
                        severity="WARNING",
                        row=row,
                        payload={
                            "log_stall_seconds": log_stall,
                            "last_log_size_bytes": row["last_log_size_bytes"],
                            "automatic_signal": False,
                        },
                        now_ns=now,
                        repeat_seconds=policy.event_repeat_seconds,
                    )
                    if finding is not None:
                        findings.append(finding)
        if monitored_path is not None:
            free_bytes = shutil.disk_usage(monitored_path).free
            if free_bytes < policy.minimum_free_disk_bytes:
                finding = self._record_general_finding_once(
                    event_type="DISK_SPACE_LOW",
                    severity="CRITICAL",
                    payload={
                        "free_bytes": free_bytes,
                        "minimum_free_bytes": policy.minimum_free_disk_bytes,
                        "monitored_path": str(Path(monitored_path).resolve()),
                    },
                    now_ns=now,
                    repeat_seconds=policy.event_repeat_seconds,
                )
                if finding is not None:
                    findings.append(finding)
        return tuple(findings)

    def check_dispatch_disk_capacity(
        self,
        *,
        monitored_path: str | Path,
        predicted_next_wave_high_water_bytes: int,
        safety_reserve_bytes: int = REMOTE_SPOOL_SAFETY_RESERVE_BYTES,
        observed_at_ns: int | None = None,
    ) -> DiskDispatchDecision:
        """Return a fail-closed dispatch gate using live filesystem capacity."""

        free_bytes = shutil.disk_usage(monitored_path).free
        decision = evaluate_dispatch_disk_gate(
            free_bytes=free_bytes,
            predicted_next_wave_high_water_bytes=(predicted_next_wave_high_water_bytes),
            safety_reserve_bytes=safety_reserve_bytes,
        )
        if decision.action == "STOP":
            self.record_watchdog_event(
                event_type="DISPATCH_STOP_DISK_HIGH_WATER",
                severity="CRITICAL",
                payload={
                    **asdict(decision),
                    "monitored_path": str(Path(monitored_path).resolve()),
                },
                occurred_at_ns=observed_at_ns,
            )
        return decision

    def register_archive_safe_boundary(
        self,
        request: ArchiveRequest,
        *,
        registered_at_ns: int | None = None,
    ) -> None:
        """Register one resumable archive unit at a terminal safe boundary."""

        registered = self._validated_time(registered_at_ns)
        request_identity = (
            request.safe_boundary,
            request.cell_id,
            request.attempt,
            request.remote_payload_root,
            request.local_partial_root,
            request.local_final_root,
            request.remote_manifest_sha256,
            request.predicted_payload_bytes,
        )
        with self._transaction():
            existing = self._connection.execute(
                "SELECT * FROM archive_checkpoints WHERE archive_id = ?",
                (request.archive_id,),
            ).fetchone()
            if existing is not None:
                existing_identity = (
                    existing["safe_boundary"],
                    existing["cell_id"],
                    existing["attempt"],
                    existing["remote_payload_root"],
                    existing["local_partial_root"],
                    existing["local_final_root"],
                    existing["remote_manifest_sha256"],
                    existing["predicted_payload_bytes"],
                )
                if existing_identity != request_identity:
                    raise ExperimentOperatorError(
                        "archive ID is already bound to different content"
                    )
                return
            if request.cell_id is not None and request.attempt is not None:
                attempt = self._require_attempt(request.cell_id, request.attempt)
                if attempt["status"] not in TERMINAL_ATTEMPT_STATUSES:
                    raise AttemptTransitionError(
                        "archive safe boundary requires a terminal cell attempt"
                    )
            self._connection.execute(
                """
                INSERT INTO archive_checkpoints (
                    archive_id, safe_boundary, cell_id, attempt,
                    remote_payload_root, local_partial_root, local_final_root,
                    remote_manifest_sha256, predicted_payload_bytes, state,
                    created_at_ns, updated_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'REGISTERED', ?, ?)
                """,
                (
                    request.archive_id,
                    request.safe_boundary,
                    request.cell_id,
                    request.attempt,
                    request.remote_payload_root,
                    request.local_partial_root,
                    request.local_final_root,
                    request.remote_manifest_sha256,
                    request.predicted_payload_bytes,
                    registered,
                    registered,
                ),
            )
            self._insert_event(
                event_type="ARCHIVE_SAFE_BOUNDARY_REGISTERED",
                severity="INFO",
                cell_id=request.cell_id,
                attempt=request.attempt,
                payload={
                    "archive_id": request.archive_id,
                    "safe_boundary": request.safe_boundary,
                    "remote_manifest_sha256": request.remote_manifest_sha256,
                    "predicted_payload_bytes": request.predicted_payload_bytes,
                    "state": "REGISTERED",
                },
                occurred_at_ns=registered,
            )

    def record_archive_step(
        self,
        archive_id: str,
        receipt: ArchiveStepReceipt,
        *,
        recorded_at_ns: int | None = None,
    ) -> None:
        """Advance an archive only through the registered verification order."""

        _require_text(archive_id, "archive ID")
        recorded = self._validated_time(recorded_at_ns)
        transitions = {
            "TRANSFER": (
                "REGISTERED",
                "TRANSFERRED",
                "transfer_receipt_json",
                "ARCHIVE_TRANSFER_COMPLETE",
            ),
            "LOCAL_SHA_VERIFY": (
                "TRANSFERRED",
                "LOCAL_SHA_VERIFIED",
                "local_sha_receipt_json",
                "ARCHIVE_LOCAL_SHA_VERIFIED",
            ),
            "REHYDRATE_VERIFY": (
                "LOCAL_SHA_VERIFIED",
                "REHYDRATE_VERIFIED",
                "rehydrate_receipt_json",
                "ARCHIVE_REHYDRATE_VERIFIED",
            ),
        }
        expected_state, next_state, column, event_type = transitions[receipt.step]
        body = _canonical_json(asdict(receipt))
        with self._transaction():
            row = self._require_archive(archive_id)
            if receipt.manifest_sha256 != row["remote_manifest_sha256"]:
                raise ExperimentOperatorError(
                    "archive step manifest differs from remote manifest"
                )
            if recorded < int(row["updated_at_ns"]):
                raise ValueError("archive step time precedes prior archive state")
            if row[column] is not None:
                if row[column] == body:
                    return
                raise ExperimentOperatorError("archive step receipt is immutable")
            if row["state"] != expected_state:
                raise AttemptTransitionError(
                    f"archive {receipt.step} requires {expected_state}, "
                    f"got {row['state']}"
                )
            prior_column = {
                "LOCAL_SHA_VERIFY": "transfer_receipt_json",
                "REHYDRATE_VERIFY": "local_sha_receipt_json",
            }.get(receipt.step)
            if prior_column is not None:
                prior = ArchiveStepReceipt(**json.loads(row[prior_column]))
                if (
                    receipt.checked_file_count != prior.checked_file_count
                    or receipt.checked_bytes != prior.checked_bytes
                ):
                    raise ExperimentOperatorError(
                        "archive step coverage differs from the preceding step"
                    )
            self._connection.execute(
                f"""
                UPDATE archive_checkpoints
                SET {column} = ?, state = ?, updated_at_ns = ?
                WHERE archive_id = ?
                """,
                (body, next_state, recorded, archive_id),
            )
            self._insert_event(
                event_type=event_type,
                severity="INFO",
                cell_id=row["cell_id"],
                attempt=row["attempt"],
                payload={
                    "archive_id": archive_id,
                    "state": next_state,
                    "manifest_sha256": receipt.manifest_sha256,
                    "evidence_sha256": receipt.evidence_sha256,
                    "checked_file_count": receipt.checked_file_count,
                    "checked_bytes": receipt.checked_bytes,
                },
                occurred_at_ns=recorded,
            )

    def authorize_remote_eviction(
        self,
        archive_id: str,
        *,
        authorized_at_ns: int | None = None,
    ) -> RemoteEvictionAuthorization:
        """Authorize, but never perform, remote eviction after both verifications."""

        _require_text(archive_id, "archive ID")
        authorized = self._validated_time(authorized_at_ns)
        with self._transaction():
            row = self._require_archive(archive_id)
            if row["state"] == "EVICTION_AUTHORIZED":
                authorized = int(row["eviction_authorized_at_ns"])
            else:
                if row["state"] != "REHYDRATE_VERIFIED":
                    raise AttemptTransitionError(
                        "remote eviction requires local SHA and rehydrate verification"
                    )
                if authorized < int(row["updated_at_ns"]):
                    raise ValueError(
                        "eviction authorization time precedes verification"
                    )
                self._connection.execute(
                    """
                    UPDATE archive_checkpoints
                    SET state = 'EVICTION_AUTHORIZED',
                        eviction_authorized_at_ns = ?, updated_at_ns = ?
                    WHERE archive_id = ?
                    """,
                    (authorized, authorized, archive_id),
                )
                self._insert_event(
                    event_type="ARCHIVE_REMOTE_EVICTION_AUTHORIZED",
                    severity="WARNING",
                    cell_id=row["cell_id"],
                    attempt=row["attempt"],
                    payload={
                        "archive_id": archive_id,
                        "remote_payload_root": row["remote_payload_root"],
                        "remote_manifest_sha256": row["remote_manifest_sha256"],
                        "state": "EVICTION_AUTHORIZED",
                        "deletion_performed": False,
                    },
                    occurred_at_ns=authorized,
                )
            local_receipt = ArchiveStepReceipt(
                **json.loads(row["local_sha_receipt_json"])
            )
            rehydrate_receipt = ArchiveStepReceipt(
                **json.loads(row["rehydrate_receipt_json"])
            )
            if rehydrate_receipt.content_tree_sha256 is None:
                raise AssertionError("rehydrate receipt lacks content-tree identity")
            return RemoteEvictionAuthorization(
                archive_id=archive_id,
                remote_payload_root=row["remote_payload_root"],
                manifest_sha256=row["remote_manifest_sha256"],
                local_final_root=row["local_final_root"],
                local_sha_evidence_sha256=local_receipt.evidence_sha256,
                rehydrate_evidence_sha256=rehydrate_receipt.evidence_sha256,
                rehydrated_content_tree_sha256=(rehydrate_receipt.content_tree_sha256),
                authorized_at_ns=authorized,
            )

    def run_archive_callbacks(
        self,
        request: ArchiveRequest,
        callbacks: ArchiveCallbacks,
    ) -> RemoteEvictionAuthorization:
        """Run injected archive steps; no callback for remote deletion exists."""

        self.register_archive_safe_boundary(request)
        callback_by_state = {
            "REGISTERED": (callbacks.transfer, "TRANSFER"),
            "TRANSFERRED": (callbacks.verify_local_sha, "LOCAL_SHA_VERIFY"),
            "LOCAL_SHA_VERIFIED": (
                callbacks.verify_rehydrate,
                "REHYDRATE_VERIFY",
            ),
        }
        while True:
            row = self._require_archive(request.archive_id)
            if row["state"] in {"REHYDRATE_VERIFIED", "EVICTION_AUTHORIZED"}:
                break
            callback, expected_step = callback_by_state[row["state"]]
            previous = _archive_previous_receipt(row)
            try:
                receipt = callback(request, previous)
                if not isinstance(receipt, ArchiveStepReceipt):
                    raise TypeError("archive callback must return ArchiveStepReceipt")
                if receipt.step != expected_step:
                    raise ValueError(
                        f"archive callback returned {receipt.step}, expected {expected_step}"
                    )
                self.record_archive_step(request.archive_id, receipt)
            except BaseException as error:
                self.record_watchdog_event(
                    event_type="ARCHIVE_STEP_FAILED",
                    severity="ERROR",
                    cell_id=request.cell_id,
                    attempt=request.attempt,
                    payload={
                        "archive_id": request.archive_id,
                        "expected_step": expected_step,
                        "exception_type": type(error).__name__,
                        "remote_eviction_authorized": False,
                    },
                )
                raise
        return self.authorize_remote_eviction(request.archive_id)

    def archive_checkpoint(self, archive_id: str) -> dict[str, Any]:
        row = self._require_archive(archive_id)
        return _decoded_archive(row)

    def record_provider_runtime_sample(self, sample: ProviderRuntimeSample) -> None:
        """Persist one redacted API observation without storing credentials."""

        if type(sample) is not ProviderRuntimeSample:
            raise TypeError("provider sample must use the exact runtime type")
        with self._transaction():
            existing = self._connection.execute(
                "SELECT * FROM provider_runtime_samples WHERE sample_id = ?",
                (sample.sample_id,),
            ).fetchone()
            if existing is not None:
                return
            lifecycle = self._connection.execute(
                """
                SELECT * FROM provider_runtime_samples
                WHERE instance_uuid = ? AND provider_started_at_ns = ?
                ORDER BY observed_at_ns, sample_id
                """,
                (sample.instance_uuid, sample.provider_started_at_ns),
            ).fetchall()
            if any(int(row["gpu_count"]) != sample.gpu_count for row in lifecycle):
                raise ExperimentOperatorError(
                    "provider lifecycle changed its GPU count"
                )
            known_stops = {
                int(row["provider_stopped_at_ns"])
                for row in lifecycle
                if row["provider_stopped_at_ns"] is not None
            }
            if sample.provider_stopped_at_ns is not None:
                known_stops.add(sample.provider_stopped_at_ns)
            if len(known_stops) > 1:
                raise ExperimentOperatorError(
                    "provider lifecycle has conflicting stop times"
                )
            stop_ns = next(iter(known_stops), None)
            if stop_ns is not None:
                running_observations = [
                    int(row["observed_at_ns"])
                    for row in lifecycle
                    if row["state"] == "running"
                ]
                if sample.state == "running":
                    running_observations.append(sample.observed_at_ns)
                if running_observations and max(running_observations) > stop_ns:
                    raise ExperimentOperatorError(
                        "provider running observation follows its stop time"
                    )
            self._connection.execute(
                """
                INSERT INTO provider_runtime_samples (
                    sample_id, instance_uuid, state, observed_at_ns,
                    provider_started_at_ns, provider_stopped_at_ns,
                    gpu_count, response_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample.sample_id,
                    sample.instance_uuid,
                    sample.state,
                    sample.observed_at_ns,
                    sample.provider_started_at_ns,
                    sample.provider_stopped_at_ns,
                    sample.gpu_count,
                    sample.response_sha256,
                ),
            )
            self._insert_event(
                event_type="PROVIDER_RUNTIME_SAMPLE_RECORDED",
                severity="INFO",
                cell_id=None,
                attempt=None,
                payload={
                    "sample_id": sample.sample_id,
                    "instance_uuid": sample.instance_uuid,
                    "state": sample.state,
                    "response_sha256": sample.response_sha256,
                    "credential_stored": False,
                },
                occurred_at_ns=sample.observed_at_ns,
            )

    def provider_billing_intervals(self) -> tuple[dict[str, Any], ...]:
        """Reduce provider observations into non-overlapping boot intervals."""

        rows = self._connection.execute(
            """
            SELECT * FROM provider_runtime_samples
            ORDER BY instance_uuid, provider_started_at_ns,
                     observed_at_ns, sample_id
            """
        ).fetchall()
        grouped: dict[tuple[str, int], list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(
                (str(row["instance_uuid"]), int(row["provider_started_at_ns"])),
                [],
            ).append(row)
        intervals: list[dict[str, Any]] = []
        previous_end_by_instance: dict[str, int] = {}
        for (instance_uuid, started_ns), samples in grouped.items():
            gpu_counts = {int(row["gpu_count"]) for row in samples}
            stopped_values = {
                int(row["provider_stopped_at_ns"])
                for row in samples
                if row["provider_stopped_at_ns"] is not None
            }
            if len(gpu_counts) != 1 or len(stopped_values) > 1:
                raise ExperimentOperatorError(
                    "provider billing lifecycle is internally inconsistent"
                )
            complete = bool(stopped_values)
            stopped_ns = (
                next(iter(stopped_values))
                if complete
                else max(int(row["observed_at_ns"]) for row in samples)
            )
            previous_end = previous_end_by_instance.get(instance_uuid)
            if previous_end is not None and started_ns < previous_end:
                raise ExperimentOperatorError(
                    "provider billing intervals overlap for one instance"
                )
            previous_end_by_instance[instance_uuid] = stopped_ns
            duration_seconds = (stopped_ns - started_ns) / 1e9
            gpu_count = next(iter(gpu_counts))
            intervals.append(
                {
                    "instance_uuid": instance_uuid,
                    "provider_started_at_ns": started_ns,
                    "provider_stopped_or_observed_at_ns": stopped_ns,
                    "complete": complete,
                    "gpu_count": gpu_count,
                    "duration_seconds": duration_seconds,
                    "whole_instance_billed_gpu_seconds": (duration_seconds * gpu_count),
                    "sample_count": len(samples),
                    "response_sha256s": tuple(
                        sorted({str(row["response_sha256"]) for row in samples})
                    ),
                }
            )
        return tuple(intervals)

    def whole_instance_billed_gpu_seconds(
        self, *, require_complete: bool = False
    ) -> float:
        intervals = self.provider_billing_intervals()
        if require_complete and any(not row["complete"] for row in intervals):
            raise ExperimentOperatorError(
                "whole-instance billing has an open provider interval"
            )
        return sum(float(row["whole_instance_billed_gpu_seconds"]) for row in intervals)

    def record_selection_decision(
        self,
        *,
        decision_id: str,
        stage: str,
        phase: str,
        decision_kind: str,
        source_sha256: str,
        decision: Mapping[str, Any],
        occurred_at_ns: int | None = None,
    ) -> None:
        for label, value in (
            ("decision ID", decision_id),
            ("stage", stage),
            ("phase", phase),
            ("decision kind", decision_kind),
        ):
            _require_text(value, label)
        _require_sha256(source_sha256, "selection source SHA-256")
        body = _canonical_json(
            _canonical_mapping(decision, "selection decision", allow_empty=False)
        )
        occurred = self._validated_time(occurred_at_ns)
        with self._transaction():
            if not self._stage_phase_exists(stage, phase):
                raise KeyError(f"unknown stage/phase {stage}/{phase}")
            try:
                self._connection.execute(
                    """
                    INSERT INTO selection_decisions (
                        decision_id, occurred_at_ns, stage, phase, decision_kind,
                        source_sha256, decision_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id,
                        occurred,
                        stage,
                        phase,
                        decision_kind,
                        source_sha256,
                        body,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ExperimentOperatorError(
                    f"selection decision {decision_id!r} already exists"
                ) from error

    def record_metric(
        self, metric: MetricRecord, *, recorded_at_ns: int | None = None
    ) -> None:
        recorded = self._validated_time(recorded_at_ns)
        with self._transaction():
            row = self._require_attempt(metric.cell_id, metric.attempt)
            if row["status"] != "COMPLETE":
                raise AttemptTransitionError("metrics require a COMPLETE attempt")
            if (row["stage"], row["phase"]) != (metric.stage, metric.phase):
                raise ValueError("metric stage/phase differs from its cell attempt")
            try:
                self._connection.execute(
                    """
                    INSERT INTO metrics_long (
                        stage, phase, cell_id, attempt, metric_name, metric_kind,
                        point_estimate, ci_low, ci_high, independent_block_count,
                        request_count, paired, reducer_method, attributes_json,
                        recorded_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metric.stage,
                        metric.phase,
                        metric.cell_id,
                        metric.attempt,
                        metric.metric_name,
                        metric.metric_kind,
                        metric.point_estimate,
                        metric.ci_low,
                        metric.ci_high,
                        metric.independent_block_count,
                        metric.request_count,
                        None if metric.paired is None else int(metric.paired),
                        metric.reducer_method,
                        _canonical_json(dict(metric.attributes)),
                        recorded,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ExperimentOperatorError("duplicate metric identity") from error

    def attempt(self, cell_id: str, attempt: int) -> dict[str, Any]:
        row = self._require_attempt(cell_id, attempt)
        return _decoded_attempt(row)

    def latest_attempt(self, cell_id: str) -> dict[str, Any] | None:
        _require_text(cell_id, "cell ID")
        row = self._connection.execute(
            """
            SELECT * FROM cell_attempts
            WHERE cell_id = ? AND is_legacy_import = 0
            ORDER BY attempt DESC LIMIT 1
            """,
            (cell_id,),
        ).fetchone()
        return None if row is None else _decoded_attempt(row)

    def command_for_attempt(
        self, cell_id: str, attempt: int
    ) -> QueuedCommandSpec | None:
        self._require_attempt(cell_id, attempt)
        row = self._connection.execute(
            "SELECT * FROM command_queue WHERE cell_id = ? AND attempt = ?",
            (cell_id, attempt),
        ).fetchone()
        return None if row is None else _decoded_command(row)

    def latest_stage_attempts(self, node: str) -> tuple[dict[str, Any], ...]:
        controller = self._require_controller_node(node)
        rows = self._connection.execute(
            """
            WITH latest AS (
                SELECT cell_id, MAX(attempt) AS attempt
                FROM cell_attempts WHERE is_legacy_import = 0
                GROUP BY cell_id
            )
            SELECT a.* FROM cell_attempts AS a
            JOIN latest AS chosen
              ON a.cell_id = chosen.cell_id AND a.attempt = chosen.attempt
            WHERE a.stage = (SELECT stage FROM stage_plan WHERE node = ?)
              AND a.phase = (SELECT phase FROM stage_plan WHERE node = ?)
            ORDER BY a.cell_id
            """,
            (controller["node"], controller["node"]),
        ).fetchall()
        return tuple(_decoded_attempt(row) for row in rows)

    def snapshot(self) -> dict[str, Any]:
        with self._read_transaction():
            run_id = self.run_id
            stages = self._stage_summary_rows()
            attempts = [
                _decoded_attempt(row)
                for row in self._connection.execute(
                    """
                    SELECT a.*, m.group_id AS physical_group_id,
                        m.logical_kind AS physical_group_logical_kind,
                        g.shared_evidence_sha256 AS physical_group_evidence_sha256,
                        CASE
                            WHEN m.group_id IS NULL THEN NULL
                            WHEN g.leader_cell_id = a.cell_id
                             AND g.leader_attempt = a.attempt THEN 1
                            ELSE 0
                        END AS physical_accounting_owner
                    FROM cell_attempts AS a
                    LEFT JOIN physical_attempt_group_members AS m
                      ON m.cell_id = a.cell_id AND m.attempt = a.attempt
                    LEFT JOIN physical_attempt_groups AS g ON g.group_id = m.group_id
                    ORDER BY a.stage, a.phase, a.cell_id, a.attempt
                    """
                ).fetchall()
            ]
            archives = [
                _decoded_archive(row)
                for row in self._connection.execute(
                    "SELECT * FROM archive_checkpoints ORDER BY created_at_ns, archive_id"
                ).fetchall()
            ]
            commands = [
                asdict(_decoded_command(row))
                for row in self._connection.execute(
                    """
                    SELECT * FROM command_queue
                    ORDER BY priority DESC, launch_compatibility_key,
                             cell_id, attempt
                    """
                ).fetchall()
            ]
            provider_intervals = self.provider_billing_intervals()
            controller_nodes = self.controller_nodes()
            controller_auxiliary_groups = self.controller_auxiliary_groups()
            physical_attempt_groups = self.physical_attempt_groups()
            dispatch_state, dispatch_stop_reason = self.dispatch_control()
        return {
            "schema_version": _SCHEMA_VERSION,
            "run_id": run_id,
            "journal_mode": self.journal_mode,
            "synchronous": self.synchronous_mode,
            "stage_plan": stages,
            "controller_nodes": controller_nodes,
            "controller_auxiliary_groups": controller_auxiliary_groups,
            "attempts": attempts,
            "commands": commands,
            "physical_attempt_groups": physical_attempt_groups,
            "archives": archives,
            "provider_billing_intervals": provider_intervals,
            "whole_instance_billed_gpu_seconds": sum(
                float(row["whole_instance_billed_gpu_seconds"])
                for row in provider_intervals
            ),
            "dispatch_state": dispatch_state,
            "dispatch_stop_reason": dispatch_stop_reason,
        }

    def export_progress(
        self,
        output_root: str | Path,
        *,
        exported_at_ns: int | None = None,
    ) -> ExportManifest:
        """Atomically replace all human-readable projections from SQLite."""

        root = Path(output_root)
        if root.exists() and root.is_symlink():
            raise ValueError("progress export root cannot be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        exported = self._validated_time(exported_at_ns)
        with self._read_transaction():
            run_id = self.run_id
            stage_rows = self._stage_summary_rows()
            attempt_rows = [
                _decoded_attempt(row)
                for row in self._connection.execute(
                    """
                    SELECT * FROM cell_attempts
                    ORDER BY stage, phase, cell_id, attempt
                    """
                ).fetchall()
            ]
            decision_rows = self._selection_rows()
            event_rows = self._event_rows()
            metric_rows = self._metric_rows()
            provider_rows = self.provider_billing_intervals()
            controller_rows = self.controller_nodes()

        paths = {
            "stage_plan.csv": root / "stage_plan.csv",
            "cell_ledger.csv": root / "cell_ledger.csv",
            "stage_summary.csv": root / "stage_summary.csv",
            "selection_decisions.jsonl": root / "selection_decisions.jsonl",
            "watchdog_events.jsonl": root / "watchdog_events.jsonl",
            "dashboard.md": root / "dashboard.md",
            "metrics_long.parquet": root / "metrics_long.parquet",
            "instance_billing.csv": root / "instance_billing.csv",
            "controller_state.csv": root / "controller_state.csv",
        }
        _atomic_write_text(paths["stage_plan.csv"], _stage_plan_csv(stage_rows))
        _atomic_write_text(paths["cell_ledger.csv"], _cell_ledger_csv(attempt_rows))
        _atomic_write_text(paths["stage_summary.csv"], _stage_summary_csv(stage_rows))
        _atomic_write_text(
            paths["selection_decisions.jsonl"], _json_lines(decision_rows)
        )
        _atomic_write_text(paths["watchdog_events.jsonl"], _json_lines(event_rows))
        _atomic_write_text(
            paths["dashboard.md"],
            _dashboard_markdown(
                run_id,
                exported,
                stage_rows,
                controller_rows,
                event_rows,
                provider_rows,
            ),
        )
        _atomic_write_metrics_parquet(paths["metrics_long.parquet"], metric_rows)
        _atomic_write_text(
            paths["instance_billing.csv"], _provider_billing_csv(provider_rows)
        )
        _atomic_write_text(
            paths["controller_state.csv"], _controller_state_csv(controller_rows)
        )
        digests = {name: _file_sha256(path) for name, path in paths.items()}
        manifest = ExportManifest(
            run_id=run_id,
            exported_at_ns=exported,
            files=digests,
        )
        _atomic_write_text(
            root / "export_manifest.json",
            _canonical_json(asdict(manifest)) + "\n",
        )
        return manifest

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    @contextmanager
    def _read_transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def _now(self) -> int:
        value = self._clock_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "operator clock must return a positive integer nanosecond time"
            )
        return value

    def _validated_time(self, value: int | None) -> int:
        result = self._now() if value is None else value
        if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
            raise ValueError("timestamp must be a positive integer nanosecond time")
        return result

    def _reopen_controller_binding(
        self, binding: ControllerArtifactBinding
    ) -> ControllerArtifactBinding:
        reopened = ControllerArtifactBinding.bind(binding.absolute_path)
        if reopened != binding:
            raise ExperimentOperatorError(
                "controller artifact differs from its durable binding"
            )
        return reopened

    def _require_controller_node(self, node: str) -> sqlite3.Row:
        _require_text(node, "controller node")
        row = self._connection.execute(
            "SELECT * FROM controller_nodes WHERE node = ?", (node,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown controller node {node!r}")
        return row

    def _require_controller_auxiliary_group(
        self,
        group_id: str,
        attempt: int,
    ) -> sqlite3.Row:
        _require_text(group_id, "controller auxiliary group ID")
        _require_positive_int(attempt, "controller auxiliary group attempt")
        row = self._connection.execute(
            """
            SELECT * FROM controller_auxiliary_groups
            WHERE group_id = ? AND attempt = ?
            """,
            (group_id, attempt),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown controller auxiliary group {group_id!r}/{attempt}")
        return row

    def _require_controller_auxiliary_spec_matches_locked(
        self,
        row: sqlite3.Row,
        spec: AuxiliaryPhysicalGroupSpec,
    ) -> None:
        stored_jobs = self._connection.execute(
            """
            SELECT * FROM controller_auxiliary_jobs
            WHERE group_id = ? AND group_attempt = ?
            ORDER BY member_ordinal
            """,
            (row["group_id"], row["attempt"]),
        ).fetchall()
        stored = (
            row["group_id"],
            int(row["attempt"]),
            row["node"],
            row["source_kind"],
            row["launch_command_sha256"],
            row["output_directory"],
            tuple(json.loads(row["assigned_gpu_uuids_json"])),
            tuple(
                (
                    job["job_id"],
                    int(job["job_attempt"]),
                    job["adoption_key"],
                    job["scientific_axes_json"],
                    job["identity_json"],
                    job["command_sha256"],
                    job["output_directory"],
                )
                for job in stored_jobs
            ),
        )
        requested = (
            spec.group_id,
            spec.attempt,
            spec.node,
            spec.source_kind,
            spec.launch_command_sha256,
            spec.output_directory,
            spec.assigned_gpu_uuids,
            tuple(
                (
                    job.job_id,
                    job.attempt,
                    job.adoption_key,
                    _canonical_json(dict(job.scientific_axes)),
                    _canonical_json(dict(job.identity)),
                    job.command_sha256,
                    job.output_directory,
                )
                for job in spec.jobs
            ),
        )
        if stored != requested:
            raise ExperimentOperatorError(
                "durable controller auxiliary group differs from rebuilt plan"
            )

    def _decoded_controller_auxiliary_group(
        self,
        row: sqlite3.Row,
    ) -> dict[str, Any]:
        jobs = self._connection.execute(
            """
            SELECT * FROM controller_auxiliary_jobs
            WHERE group_id = ? AND group_attempt = ?
            ORDER BY member_ordinal
            """,
            (row["group_id"], row["attempt"]),
        ).fetchall()
        output = dict(row)
        output["attempt"] = int(output["attempt"])
        output["assigned_gpu_uuids"] = tuple(
            json.loads(output.pop("assigned_gpu_uuids_json"))
        )
        output["jobs"] = tuple(_decoded_controller_auxiliary_job(job) for job in jobs)
        for source, target in (
            ("created_at_ns", "created_at"),
            ("updated_at_ns", "updated_at"),
            ("started_at_ns", "started_at"),
            ("finished_at_ns", "finished_at"),
            ("adopted_at_ns", "adopted_at"),
        ):
            value = output[source]
            output[target] = None if value is None else _iso_utc(int(value))
        return output

    def _metadata_value(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM operator_meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row["value"])

    def _set_metadata(self, key: str, value: str) -> None:
        self._connection.execute(
            """
            INSERT INTO operator_meta(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _stage_plan_rows(self) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM stage_plan ORDER BY ordinal"
        ).fetchall()

    def _stage_phase_exists(self, stage: str, phase: str) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM stage_plan WHERE stage = ? AND phase = ?",
                (stage, phase),
            ).fetchone()
            is not None
        )

    def _materialize_attempt_locked(
        self,
        spec: CellAttemptSpec,
        *,
        now: int,
    ) -> None:
        axes_json = _canonical_json(dict(spec.scientific_axes))
        identity_json = _canonical_json(dict(spec.identity))
        if not self._stage_phase_exists(spec.stage, spec.phase):
            raise KeyError(f"unknown stage/phase {spec.stage}/{spec.phase}")
        latest = self._connection.execute(
            """
            SELECT * FROM cell_attempts
            WHERE cell_id = ? ORDER BY attempt DESC LIMIT 1
            """,
            (spec.cell_id,),
        ).fetchone()
        if latest is None:
            if spec.attempt != 1:
                raise AttemptTransitionError("first materialized attempt must be 1")
            plan = self._connection.execute(
                """
                SELECT known_expected_cells FROM stage_plan
                WHERE stage = ? AND phase = ?
                """,
                (spec.stage, spec.phase),
            ).fetchone()
            materialized = int(
                self._connection.execute(
                    """
                    SELECT COUNT(DISTINCT cell_id) FROM cell_attempts
                    WHERE stage = ? AND phase = ? AND is_legacy_import = 0
                    """,
                    (spec.stage, spec.phase),
                ).fetchone()[0]
            )
            if plan["known_expected_cells"] is not None and materialized >= int(
                plan["known_expected_cells"]
            ):
                raise ExperimentOperatorError(
                    "materialization would exceed known stage coverage"
                )
        else:
            expected = int(latest["attempt"]) + 1
            if spec.attempt != expected:
                raise AttemptTransitionError(
                    f"next attempt for {spec.cell_id!r} must be {expected}"
                )
            if latest["status"] not in {"FAILED", "BLOCKED", "STALE_IDENTITY"}:
                raise AttemptTransitionError(
                    "a retry requires a failed, blocked, or stale prior attempt"
                )
            stable_identity = (
                latest["stage"],
                latest["phase"],
                latest["block_id"],
                latest["seed"],
                latest["scientific_axes_json"],
            )
            requested_identity = (
                spec.stage,
                spec.phase,
                spec.block,
                spec.seed,
                axes_json,
            )
            if stable_identity != requested_identity:
                raise AttemptTransitionError(
                    "a retry cannot change stage, block, seed, or scientific axes"
                )
            if latest["status"] in {"FAILED", "BLOCKED"} and (
                latest["identity_json"] != identity_json
                or latest["scientific_command_sha256"] != spec.scientific_command_sha256
            ):
                raise AttemptTransitionError(
                    "an infrastructure retry cannot change scientific identity"
                )
            command_changed = latest["command_sha256"] != spec.command_sha256
            if (
                command_changed
                and latest["status"] != "STALE_IDENTITY"
                and (
                    latest["status"] != "FAILED"
                    or latest["failure_code"] is None
                    or not str(latest["failure_code"]).startswith("INFRASTRUCTURE:")
                    or latest["retry_decision"] != "RETRY_INFRASTRUCTURE_AUTOMATIC"
                    or spec.scientific_command_sha256 is None
                )
            ):
                raise AttemptTransitionError(
                    "a changed retry command requires sealed infrastructure failure "
                    "and stable path-independent scientific identity"
                )
        try:
            self._connection.execute(
                """
                INSERT INTO cell_attempts (
                    cell_id, attempt, stage, phase, block_id, seed,
                    scientific_axes_json, identity_json, status, command_sha256,
                    scientific_command_sha256,
                    output_directory, created_at_ns, updated_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?)
                """,
                (
                    spec.cell_id,
                    spec.attempt,
                    spec.stage,
                    spec.phase,
                    spec.block,
                    spec.seed,
                    axes_json,
                    identity_json,
                    spec.command_sha256,
                    spec.scientific_command_sha256,
                    spec.output_directory,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ExperimentOperatorError(
                "attempt identity or output directory is already registered"
            ) from error
        self._touch_stage(spec.stage, spec.phase, now)

    def _enqueue_command_locked(
        self,
        command: QueuedCommandSpec,
        *,
        enqueued: int,
    ) -> None:
        row = self._require_attempt(command.cell_id, command.attempt)
        if row["status"] != "PENDING":
            raise AttemptTransitionError("only PENDING attempts may be queued")
        if row["command_sha256"] != command.command_sha256:
            raise AttemptTransitionError(
                "queued argv differs from materialized command SHA-256"
            )
        try:
            self._connection.execute(
                """
                INSERT INTO command_queue (
                    cell_id, attempt, argv_json, environment_json,
                    launch_compatibility_key, required_gpu_count,
                    timing_class, predicted_high_water_bytes,
                    monitored_path, log_path, expected_terminal_path,
                    expected_junit_path, expected_raw_log_path,
                    atomic_pointer_path, child_exit_receipt_path,
                    paired_gpu_key, preferred_gpu_index, priority,
                    max_runtime_seconds, max_log_stall_seconds,
                    enqueued_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    command.cell_id,
                    command.attempt,
                    _canonical_json(command.argv),
                    _canonical_json(command.environment),
                    command.launch_compatibility_key,
                    command.required_gpu_count,
                    command.timing_class,
                    command.predicted_high_water_bytes,
                    command.monitored_path,
                    command.log_path,
                    command.expected_terminal_path,
                    command.expected_junit_path,
                    command.expected_raw_log_path,
                    command.atomic_pointer_path,
                    command.child_exit_receipt_path,
                    command.paired_gpu_key,
                    command.preferred_gpu_index,
                    command.priority,
                    command.max_runtime_seconds,
                    command.max_log_stall_seconds,
                    enqueued,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ExperimentOperatorError(
                "command attempt is already queued"
            ) from error

    def _touch_stage(self, stage: str, phase: str, timestamp_ns: int) -> None:
        self._connection.execute(
            """
            UPDATE stage_plan SET last_update_ns = ? WHERE stage = ? AND phase = ?
            """,
            (timestamp_ns, stage, phase),
        )

    def _require_attempt(self, cell_id: str, attempt: int) -> sqlite3.Row:
        _require_text(cell_id, "cell ID")
        if isinstance(attempt, bool) or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        row = self._connection.execute(
            "SELECT * FROM cell_attempts WHERE cell_id = ? AND attempt = ?",
            (cell_id, attempt),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown attempt {cell_id!r}/{attempt}")
        return row

    def _require_physical_group_matches_locked(
        self,
        row: sqlite3.Row,
        *,
        members: tuple[PhysicalAttemptGroupMemberSpec, ...],
        leader_cell_id: str,
    ) -> None:
        if (
            row["leader_cell_id"] != leader_cell_id
            or int(row["leader_attempt"]) != members[0].attempt.attempt
        ):
            raise ExperimentOperatorError(
                "durable physical attempt group leader differs"
            )
        stored = self._connection.execute(
            """
            SELECT * FROM physical_attempt_group_members
            WHERE group_id = ? ORDER BY member_ordinal
            """,
            (row["group_id"],),
        ).fetchall()
        expected_members = tuple(
            (
                member.attempt.cell_id,
                member.attempt.attempt,
                member.logical_kind,
                ordinal,
            )
            for ordinal, member in enumerate(members)
        )
        actual_members = tuple(
            (
                str(member["cell_id"]),
                int(member["attempt"]),
                str(member["logical_kind"]),
                int(member["member_ordinal"]),
            )
            for member in stored
        )
        if actual_members != expected_members:
            raise ExperimentOperatorError(
                "durable physical attempt group membership differs"
            )
        for member in members:
            attempt = _decoded_attempt(
                self._require_attempt(
                    member.attempt.cell_id,
                    member.attempt.attempt,
                )
            )
            expected_attempt = {
                "cell_id": member.attempt.cell_id,
                "attempt": member.attempt.attempt,
                "stage": member.attempt.stage,
                "phase": member.attempt.phase,
                "block_id": member.attempt.block,
                "seed": member.attempt.seed,
                "scientific_axes": dict(member.attempt.scientific_axes),
                "identity": dict(member.attempt.identity),
                "command_sha256": member.attempt.command_sha256,
                "scientific_command_sha256": (member.attempt.scientific_command_sha256),
                "output_directory": member.attempt.output_directory,
            }
            if any(attempt[key] != value for key, value in expected_attempt.items()):
                raise ExperimentOperatorError(
                    "durable physical attempt group attempt differs"
                )
            queued = self._connection.execute(
                "SELECT * FROM command_queue WHERE cell_id = ? AND attempt = ?",
                (member.attempt.cell_id, member.attempt.attempt),
            ).fetchone()
            if queued is None or _decoded_command(queued) != member.command:
                raise ExperimentOperatorError(
                    "durable physical attempt group command differs"
                )

    def _require_archive(self, archive_id: str) -> sqlite3.Row:
        _require_text(archive_id, "archive ID")
        row = self._connection.execute(
            "SELECT * FROM archive_checkpoints WHERE archive_id = ?", (archive_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown archive checkpoint {archive_id!r}")
        return row

    def _insert_event(
        self,
        *,
        event_type: str,
        severity: WatchdogSeverity,
        cell_id: str | None,
        attempt: int | None,
        payload: Mapping[str, Any],
        occurred_at_ns: int,
    ) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO watchdog_events (
                occurred_at_ns, event_type, severity, cell_id, attempt, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                occurred_at_ns,
                event_type,
                severity,
                cell_id,
                attempt,
                _canonical_json(dict(payload)),
            ),
        )
        return int(cursor.lastrowid)

    def _record_watchdog_finding_once(
        self,
        *,
        event_type: str,
        severity: WatchdogSeverity,
        row: sqlite3.Row,
        payload: Mapping[str, Any],
        now_ns: int,
        repeat_seconds: float,
    ) -> WatchdogFinding | None:
        return self._record_finding_once(
            event_type=event_type,
            severity=severity,
            cell_id=str(row["cell_id"]),
            attempt=int(row["attempt"]),
            payload=payload,
            now_ns=now_ns,
            repeat_seconds=repeat_seconds,
        )

    def _record_general_finding_once(
        self,
        *,
        event_type: str,
        severity: WatchdogSeverity,
        payload: Mapping[str, Any],
        now_ns: int,
        repeat_seconds: float,
    ) -> WatchdogFinding | None:
        return self._record_finding_once(
            event_type=event_type,
            severity=severity,
            cell_id=None,
            attempt=None,
            payload=payload,
            now_ns=now_ns,
            repeat_seconds=repeat_seconds,
        )

    def _record_finding_once(
        self,
        *,
        event_type: str,
        severity: WatchdogSeverity,
        cell_id: str | None,
        attempt: int | None,
        payload: Mapping[str, Any],
        now_ns: int,
        repeat_seconds: float,
    ) -> WatchdogFinding | None:
        cutoff = now_ns - int(repeat_seconds * 1e9)
        with self._transaction():
            last = self._connection.execute(
                """
                SELECT occurred_at_ns FROM watchdog_events
                WHERE event_type = ? AND cell_id IS ? AND attempt IS ?
                ORDER BY occurred_at_ns DESC LIMIT 1
                """,
                (event_type, cell_id, attempt),
            ).fetchone()
            if last is not None and int(last["occurred_at_ns"]) >= cutoff:
                return None
            event_id = self._insert_event(
                event_type=event_type,
                severity=severity,
                cell_id=cell_id,
                attempt=attempt,
                payload=payload,
                occurred_at_ns=now_ns,
            )
        return WatchdogFinding(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            cell_id=cell_id,
            attempt=attempt,
            payload=dict(payload),
        )

    def _stage_summary_rows(self) -> list[dict[str, Any]]:
        latest = """
            WITH latest AS (
                SELECT a.* FROM cell_attempts AS a
                JOIN (
                    SELECT cell_id, MAX(attempt) AS attempt
                    FROM cell_attempts
                    WHERE is_legacy_import = 0
                    GROUP BY cell_id
                ) AS chosen
                ON a.cell_id = chosen.cell_id AND a.attempt = chosen.attempt
            ), counts AS (
                SELECT stage, phase,
                    COUNT(*) AS materialized,
                    SUM(status = 'COMPLETE') AS completed,
                    SUM(status = 'RUNNING') AS running,
                    SUM(status = 'FAILED') AS failed,
                    SUM(status = 'BLOCKED') AS blocked
                FROM latest GROUP BY stage, phase
            ), stale_counts AS (
                SELECT stage, phase, COUNT(*) AS stale
                FROM cell_attempts
                WHERE status = 'STALE_IDENTITY'
                GROUP BY stage, phase
            ), accounting AS (
                SELECT stage, phase,
                    SUM(compute_gpu_seconds) AS compute_gpu_seconds,
                    SUM(reserved_gpu_seconds) AS reserved_gpu_seconds,
                    SUM(billed_gpu_seconds) AS billed_gpu_seconds
                FROM cell_attempts GROUP BY stage, phase
            ), pending_auxiliary_accounting AS (
                SELECT g.node,
                    SUM(g.compute_gpu_seconds) AS compute_gpu_seconds,
                    SUM(g.reserved_gpu_seconds) AS reserved_gpu_seconds,
                    SUM(g.billed_gpu_seconds) AS billed_gpu_seconds
                FROM controller_auxiliary_groups AS g
                WHERE g.adopted_at_ns IS NULL
                GROUP BY g.node
            )
            SELECT p.*, COALESCE(c.materialized, 0) AS materialized,
                COALESCE(c.completed, 0) AS completed,
                COALESCE(c.running, 0) AS running,
                COALESCE(c.failed, 0) AS failed,
                COALESCE(c.blocked, 0) AS blocked,
                COALESCE(s.stale, 0) AS stale,
                COALESCE(a.compute_gpu_seconds, 0)
                    + COALESCE(x.compute_gpu_seconds, 0) AS compute_gpu_seconds,
                COALESCE(a.reserved_gpu_seconds, 0)
                    + COALESCE(x.reserved_gpu_seconds, 0) AS reserved_gpu_seconds,
                COALESCE(a.billed_gpu_seconds, 0)
                    + COALESCE(x.billed_gpu_seconds, 0) AS billed_gpu_seconds
            FROM stage_plan AS p
            LEFT JOIN counts AS c ON p.stage = c.stage AND p.phase = c.phase
            LEFT JOIN stale_counts AS s
                ON p.stage = s.stage AND p.phase = s.phase
            LEFT JOIN accounting AS a ON p.stage = a.stage AND p.phase = a.phase
            LEFT JOIN pending_auxiliary_accounting AS x ON p.node = x.node
            ORDER BY p.ordinal
        """
        output: list[dict[str, Any]] = []
        for row in self._connection.execute(latest).fetchall():
            known = row["known_expected_cells"]
            completed = int(row["completed"])
            materialized = int(row["materialized"])
            if known is None:
                progress = None
            elif int(known) == 0:
                progress = 100.0 if materialized == 0 else 0.0
            else:
                progress = 100.0 * completed / int(known)
            output.append(
                {
                    "node": row["node"],
                    "ordinal": int(row["ordinal"]),
                    "stage": row["stage"],
                    "phase": row["phase"],
                    "expected_formula": row["expected_formula"],
                    "known_expected_cells": known,
                    "materialized_cells": materialized,
                    "completed": completed,
                    "running": int(row["running"]),
                    "failed": int(row["failed"]),
                    "blocked": int(row["blocked"]),
                    "stale": int(row["stale"]),
                    "progress_percent": progress,
                    "actual_gpu_hours": float(row["compute_gpu_seconds"]) / 3600.0,
                    "reserved_gpu_hours": float(row["reserved_gpu_seconds"]) / 3600.0,
                    "billed_gpu_hours": float(row["billed_gpu_seconds"]) / 3600.0,
                    "estimated_remaining_gpu_hours": row[
                        "estimated_remaining_gpu_hours"
                    ],
                    "last_update_ns": int(row["last_update_ns"]),
                    "last_update": _iso_utc(int(row["last_update_ns"])),
                }
            )
        return output

    def _selection_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "decision_id": row["decision_id"],
                "occurred_at_ns": int(row["occurred_at_ns"]),
                "stage": row["stage"],
                "phase": row["phase"],
                "decision_kind": row["decision_kind"],
                "source_sha256": row["source_sha256"],
                "decision": json.loads(row["decision_json"]),
            }
            for row in self._connection.execute(
                "SELECT * FROM selection_decisions ORDER BY occurred_at_ns, decision_id"
            ).fetchall()
        ]

    def _event_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "event_id": int(row["event_id"]),
                "occurred_at_ns": int(row["occurred_at_ns"]),
                "event_type": row["event_type"],
                "severity": row["severity"],
                "cell_id": row["cell_id"],
                "attempt": row["attempt"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in self._connection.execute(
                "SELECT * FROM watchdog_events ORDER BY event_id"
            ).fetchall()
        ]

    def _metric_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "stage": row["stage"],
                "phase": row["phase"],
                "cell_id": row["cell_id"],
                "attempt": int(row["attempt"]),
                "metric_name": row["metric_name"],
                "metric_kind": row["metric_kind"],
                "point_estimate": float(row["point_estimate"]),
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "independent_block_count": row["independent_block_count"],
                "request_count": row["request_count"],
                "paired": None if row["paired"] is None else bool(row["paired"]),
                "reducer_method": row["reducer_method"],
                "attributes_json": row["attributes_json"],
                "recorded_at_ns": int(row["recorded_at_ns"]),
            }
            for row in self._connection.execute(
                """
                SELECT * FROM metrics_long
                ORDER BY stage, phase, cell_id, attempt, metric_name, attributes_json
                """
            ).fetchall()
        ]


class FormalExperimentSchedulerDaemon:
    """Small callback-driven scheduler over the durable command queue."""

    def __init__(
        self,
        store: ExperimentOperatorStore,
        *,
        lock_path: str | Path,
        callbacks: SchedulerCallbacks,
        watchdog_policy: WatchdogPolicy | None = None,
        interval_seconds: float = 30.0,
        clock_ns: Callable[[], int] = time.time_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if type(store) is not ExperimentOperatorStore:
            raise TypeError("scheduler requires an exact operator store")
        if type(callbacks) is not SchedulerCallbacks:
            raise TypeError("scheduler requires exact injected callbacks")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or float(interval_seconds) != 30.0
        ):
            raise ValueError("scheduler heartbeat interval is fixed at 30 seconds")
        self.store = store
        self.lock_path = Path(lock_path)
        self.callbacks = callbacks
        self.watchdog_policy = watchdog_policy or WatchdogPolicy()
        self.interval_seconds = float(interval_seconds)
        self.clock_ns = clock_ns
        self.sleeper = sleeper

    def run_once(self) -> SchedulerCycleResult:
        """Reconcile terminal state, then fill only authorized free GPU slots."""

        reconciled = list(self._reconcile_running())
        state, reason = self.store.dispatch_control()
        dispatched: list[tuple[str, int, tuple[str, ...]]] = []
        if state == "STOP":
            return SchedulerCycleResult(tuple(reconciled), (), state, reason)
        envelope = self.store.interference_envelope()
        while True:
            selected = self._select_dispatch(envelope)
            if selected is None:
                break
            command, gpu_uuids = selected
            free_bytes = self.callbacks.free_disk_bytes(command.monitored_path)
            decision = evaluate_dispatch_disk_gate(
                free_bytes=free_bytes,
                predicted_next_wave_high_water_bytes=(
                    command.predicted_high_water_bytes
                ),
            )
            if decision.action == "STOP":
                reason = (
                    "disk_high_water_gate:"
                    f"{decision.free_bytes}<{decision.required_free_bytes}"
                )
                self.store.record_watchdog_event(
                    event_type="DISPATCH_STOP_DISK_HIGH_WATER",
                    severity="CRITICAL",
                    cell_id=command.cell_id,
                    attempt=command.attempt,
                    payload={
                        **asdict(decision),
                        "monitored_path": command.monitored_path,
                    },
                    occurred_at_ns=self._now(),
                )
                self.store.set_dispatch_stop(reason, stopped_at_ns=self._now())
                state = "STOP"
                break
            started = self._now()

            def launch(
                selected_command: QueuedCommandSpec = command,
                selected_gpu_uuids: tuple[str, ...] = gpu_uuids,
            ) -> SpawnedProcess:
                process = self.callbacks.launch(
                    selected_command,
                    selected_gpu_uuids,
                )
                if type(process) is not SpawnedProcess:
                    raise TypeError("scheduler launcher returned another type")
                if process.pid != process.pgid:
                    raise ValueError("scheduler child is not a setsid session leader")
                return process

            try:
                group = self.store.physical_attempt_group_for_attempt(
                    command.cell_id,
                    command.attempt,
                )
                if group is None:
                    self.store.start_attempt_with_launcher(
                        command.cell_id,
                        command.attempt,
                        assigned_gpu_uuids=gpu_uuids,
                        launcher=launch,
                        started_at_ns=started,
                    )
                else:
                    self.store.start_physical_attempt_group_with_launcher(
                        str(group["group_id"]),
                        assigned_gpu_uuids=gpu_uuids,
                        launcher=launch,
                        started_at_ns=started,
                    )
            except Exception as error:  # noqa: BLE001 - injected launcher boundary
                group = self.store.physical_attempt_group_for_attempt(
                    command.cell_id,
                    command.attempt,
                )
                if group is None:
                    self._terminalize_spawn_failure(command, error)
                else:
                    group_commands = self.store.physical_attempt_group_commands(
                        str(group["group_id"])
                    )
                    self.store.fail_physical_attempt_group_spawn(
                        str(group["group_id"]),
                        exception_type=type(error).__name__,
                        finished_at_ns=self._now(),
                    )
                    self._materialize_serving_group_retries(
                        group,
                        group_commands,
                    )
                reconciled.append((command.cell_id, command.attempt, "SPAWN_FAILED"))
                state, reason = self.store.dispatch_control()
                if state == "STOP":
                    break
                continue
            self.store.record_launch_compatibility_key(command.launch_compatibility_key)
            self.store.record_watchdog_event(
                event_type="COMMAND_DISPATCHED",
                severity="INFO",
                cell_id=command.cell_id,
                attempt=command.attempt,
                payload={
                    "gpu_uuids": list(gpu_uuids),
                    "launch_compatibility_key": command.launch_compatibility_key,
                    "timing_class": command.timing_class,
                },
                occurred_at_ns=self._now(),
            )
            dispatched.append((command.cell_id, command.attempt, gpu_uuids))
        state, reason = self.store.dispatch_control()
        return SchedulerCycleResult(
            tuple(reconciled),
            tuple(dispatched),
            state,
            reason,
        )

    def run_forever(
        self,
        *,
        stop_requested: Callable[[], bool] = lambda: False,
        max_cycles: int | None = None,
    ) -> tuple[SchedulerCycleResult, ...]:
        """Hold a singleton flock and execute one 30-second cycle at a time."""

        if max_cycles is not None and (
            isinstance(max_cycles, bool)
            or not isinstance(max_cycles, int)
            or max_cycles < 1
        ):
            raise ValueError("scheduler max cycles must be positive or null")
        results = []
        with SingletonOperatorLock(self.lock_path):
            while not stop_requested():
                result = self.run_once()
                results.append(result)
                if max_cycles is not None and len(results) >= max_cycles:
                    break
                if (
                    result.dispatch_state == "STOP"
                    and self.store.running_termination_count() == 0
                ):
                    break
                self.sleeper(self.interval_seconds)
        return tuple(results)

    def _manual_heartbeat_waiver(
        self,
        command: QueuedCommandSpec,
        attempt: Mapping[str, Any],
    ) -> bool:
        evidence = self.store.dispatch_running_recovery_evidence()
        if evidence is None or set(evidence) != {
            "schema_version",
            "kind",
            "mode",
            "stop_reason",
            "verified_at_ns",
            "processes",
            "heartbeat_observations",
            "manual_evidence",
        }:
            return False
        if (
            evidence.get("schema_version") != 1
            or evidence.get("kind") != "formal_experiment_dispatch_running_recovery"
            or evidence.get("mode") != "MANUAL_OPERATOR_EVIDENCE"
            or evidence.get("stop_reason") != "child_heartbeat_stale"
            or type(evidence.get("verified_at_ns")) is not int
            or int(evidence["verified_at_ns"]) < 1
            or type(evidence.get("manual_evidence")) is not dict
            or type(evidence.get("processes")) is not list
            or type(evidence.get("heartbeat_observations")) is not list
        ):
            return False
        manual_value = evidence["manual_evidence"]
        if set(manual_value) != {"absolute_path", "sha256"}:
            return False
        try:
            manual_binding = ControllerArtifactBinding(**manual_value)
            if ControllerArtifactBinding.bind(manual_binding.absolute_path) != (
                manual_binding
            ):
                return False
        except (TypeError, ValueError, OSError, RuntimeError):
            return False
        covered_identity = {"cell_id": command.cell_id, "attempt": command.attempt}
        matches = [
            row
            for row in evidence["processes"]
            if type(row) is dict and covered_identity in row.get("covered_attempts", [])
        ]
        if len(matches) != 1:
            return False
        row = matches[0]
        if (
            set(row)
            != {
                "cell_id",
                "attempt",
                "command_sha256",
                "pid",
                "pgid",
                "process_start_receipt_sha256",
                "covered_attempts",
            }
            or row.get("cell_id") != command.cell_id
            or row.get("attempt") != command.attempt
            or row.get("command_sha256") != command.command_sha256
            or row.get("pid") != attempt["pid"]
            or row.get("pgid") != attempt["pgid"]
            or row.get("process_start_receipt_sha256")
            != attempt["process_start_receipt_sha256"]
        ):
            return False
        recover = self.callbacks.recover_started_process
        if recover is None:
            return False
        recovered = recover(command)
        return recovered is not None and (
            recovered.pid,
            recovered.pgid,
            recovered.receipt_sha256,
        ) == (
            attempt["pid"],
            attempt["pgid"],
            attempt["process_start_receipt_sha256"],
        )

    def _reconcile_running(self) -> tuple[tuple[str, int, str], ...]:
        reconciled = []
        now = self._now()
        for command in self.store.physical_commands(status="RUNNING"):
            attempt = self.store.attempt(command.cell_id, command.attempt)
            group = self.store.physical_attempt_group_for_attempt(
                command.cell_id,
                command.attempt,
            )
            pid = attempt["pid"]
            pgid = attempt["pgid"]
            if type(pid) is not int or type(pgid) is not int:
                recover = self.callbacks.recover_started_process
                if recover is None:
                    continue
                try:
                    recovered = recover(command)
                    if (
                        recovered is not None
                        and type(recovered) is not RecoveredProcessStart
                    ):
                        raise TypeError("process-start recovery returned another type")
                except Exception as error:  # noqa: BLE001 - receipt trust boundary
                    self.store.record_watchdog_event(
                        event_type="PROCESS_START_RECEIPT_VALIDATION_FAILED",
                        severity="CRITICAL",
                        cell_id=command.cell_id,
                        attempt=command.attempt,
                        payload={"exception_type": type(error).__name__},
                        occurred_at_ns=now,
                    )
                    self.store.set_dispatch_stop(
                        "process_start_receipt_validation_failed",
                        stopped_at_ns=now,
                    )
                    reconciled.append((command.cell_id, command.attempt, "STOPPED"))
                    continue
                if recovered is None:
                    started_at_ns = attempt["started_at_ns"]
                    age_seconds = (
                        0.0
                        if type(started_at_ns) is not int
                        else (now - started_at_ns) / 1e9
                    )
                    if age_seconds <= self.watchdog_policy.process_attach_grace_seconds:
                        reconciled.append(
                            (command.cell_id, command.attempt, "WAITING_START_RECEIPT")
                        )
                        continue
                    if group is None:
                        self._terminalize_missing_start_receipt(command, now_ns=now)
                    else:
                        self.store.fail_physical_attempt_group_spawn(
                            str(group["group_id"]),
                            exception_type="START_RECEIPT_NOT_PUBLISHED",
                            finished_at_ns=now,
                        )
                    self.store.record_watchdog_event(
                        event_type="PROCESS_START_RECEIPT_NOT_PUBLISHED",
                        severity="ERROR",
                        cell_id=command.cell_id,
                        attempt=command.attempt,
                        payload={
                            "age_seconds": age_seconds,
                            "process_attach_grace_seconds": (
                                self.watchdog_policy.process_attach_grace_seconds
                            ),
                            "job_launch_proven_absent": True,
                        },
                        occurred_at_ns=now,
                    )
                    reconciled.append(
                        (command.cell_id, command.attempt, "START_NOT_PUBLISHED")
                    )
                    continue
                if recovered.started_ns < int(attempt["started_at_ns"]):
                    self.store.record_watchdog_event(
                        event_type="PROCESS_START_RECEIPT_TIMING_INVALID",
                        severity="CRITICAL",
                        cell_id=command.cell_id,
                        attempt=command.attempt,
                        payload={
                            "receipt_started_ns": recovered.started_ns,
                            "ledger_started_ns": attempt["started_at_ns"],
                        },
                        occurred_at_ns=now,
                    )
                    self.store.set_dispatch_stop(
                        "process_start_receipt_timing_invalid",
                        stopped_at_ns=now,
                    )
                    reconciled.append((command.cell_id, command.attempt, "STOPPED"))
                    continue
                if group is None:
                    self.store.attach_process(
                        command.cell_id,
                        command.attempt,
                        pid=recovered.pid,
                        pgid=recovered.pgid,
                        process_start_receipt_sha256=recovered.receipt_sha256,
                    )
                else:
                    self.store.attach_physical_attempt_group_process(
                        str(group["group_id"]),
                        pid=recovered.pid,
                        pgid=recovered.pgid,
                        process_start_receipt_sha256=recovered.receipt_sha256,
                    )
                self.store.record_watchdog_event(
                    event_type="PROCESS_METADATA_RECOVERED_FROM_START_RECEIPT",
                    severity="INFO",
                    cell_id=command.cell_id,
                    attempt=command.attempt,
                    payload={
                        "pid": recovered.pid,
                        "pgid": recovered.pgid,
                        "process_start_receipt_sha256": recovered.receipt_sha256,
                    },
                    occurred_at_ns=now,
                )
                attempt = self.store.attempt(command.cell_id, command.attempt)
                pid = recovered.pid
                pgid = recovered.pgid
            observation = self.callbacks.process_probe(pid, pgid)
            if type(observation) is not ProcessObservation or observation.pid != pid:
                self.store.set_dispatch_stop(
                    "invalid_process_probe_result",
                    stopped_at_ns=now,
                )
                reconciled.append((command.cell_id, command.attempt, "STOPPED"))
                continue
            if observation.alive and observation.observed_pgid == pgid:
                try:
                    log_size = self.callbacks.log_size_bytes(command)
                    gpu = self.callbacks.gpu_snapshot(
                        tuple(attempt["assigned_gpu_uuids"])
                    )
                    heartbeat = (
                        None
                        if self.callbacks.worker_heartbeat is None
                        else self.callbacks.worker_heartbeat(command)
                    )
                    if heartbeat is not None and type(heartbeat) is not WorkerHeartbeat:
                        raise TypeError(
                            "worker heartbeat callback returned another type"
                        )
                    heartbeat_recorded = False
                    if heartbeat is not None:
                        if (
                            heartbeat.command_sha256 != command.command_sha256
                            or heartbeat.observed_at_ns > now
                            or heartbeat.observed_at_ns < int(attempt["started_at_ns"])
                        ):
                            raise ValueError(
                                "worker heartbeat identity or timing differs"
                            )
                        worker_observation = self.callbacks.process_probe(
                            heartbeat.worker_pid,
                            pgid,
                        )
                        if (
                            worker_observation.alive
                            and worker_observation.observed_pgid != pgid
                        ):
                            raise ValueError(
                                "heartbeat worker is outside process group"
                            )
                        if heartbeat.sequence > int(attempt["heartbeat_sequence"]):
                            if group is None:
                                self.store.record_heartbeat(
                                    command.cell_id,
                                    command.attempt,
                                    pid=pid,
                                    pgid=pgid,
                                    log_size_bytes=log_size,
                                    gpu_observation=gpu,
                                    observed_at_ns=heartbeat.observed_at_ns,
                                    heartbeat_sequence=heartbeat.sequence,
                                )
                            else:
                                self.store.record_physical_attempt_group_heartbeat(
                                    str(group["group_id"]),
                                    pid=pid,
                                    pgid=pgid,
                                    log_size_bytes=log_size,
                                    gpu_observation=gpu,
                                    observed_at_ns=heartbeat.observed_at_ns,
                                    heartbeat_sequence=heartbeat.sequence,
                                )
                            heartbeat_recorded = True
                        elif (
                            heartbeat.sequence != int(attempt["heartbeat_sequence"])
                            or heartbeat.observed_at_ns != attempt["heartbeat_at_ns"]
                        ):
                            raise ValueError("worker heartbeat regressed or changed")
                    if not heartbeat_recorded:
                        if group is None:
                            self.store.record_runtime_observation(
                                command.cell_id,
                                command.attempt,
                                pid=pid,
                                pgid=pgid,
                                log_size_bytes=log_size,
                                gpu_observation=gpu,
                                observed_at_ns=now,
                            )
                        else:
                            self.store.record_physical_attempt_group_runtime_observation(
                                str(group["group_id"]),
                                pid=pid,
                                pgid=pgid,
                                log_size_bytes=log_size,
                                gpu_observation=gpu,
                                observed_at_ns=now,
                            )
                except Exception as error:  # noqa: BLE001 - monitoring trust boundary
                    self.store.record_watchdog_event(
                        event_type="LIVE_PROCESS_MONITORING_FAILED",
                        severity="CRITICAL",
                        cell_id=command.cell_id,
                        attempt=command.attempt,
                        payload={"exception_type": type(error).__name__},
                        occurred_at_ns=now,
                    )
                    self.store.set_dispatch_stop(
                        "live_process_monitoring_failed",
                        stopped_at_ns=now,
                    )
                    reconciled.append((command.cell_id, command.attempt, "STOPPED"))
                    continue
                current = self.store.attempt(command.cell_id, command.attempt)
                runtime_seconds = (now - int(current["started_at_ns"])) / 1e9
                growth_ns = current["last_log_growth_ns"]
                log_stall_seconds = (
                    None if growth_ns is None else (now - int(growth_ns)) / 1e9
                )
                if (
                    log_stall_seconds is not None
                    and log_stall_seconds > command.max_log_stall_seconds
                ):
                    self.store._record_watchdog_finding_once(
                        event_type="COMMAND_LOG_STALL_WARNING",
                        severity="WARNING",
                        row=self.store._require_attempt(
                            command.cell_id,
                            command.attempt,
                        ),
                        payload={
                            "log_stall_seconds": log_stall_seconds,
                            "max_log_stall_seconds": (command.max_log_stall_seconds),
                            "last_log_size_bytes": current["last_log_size_bytes"],
                        },
                        now_ns=now,
                        repeat_seconds=self.watchdog_policy.event_repeat_seconds,
                    )
                heartbeat_required = (
                    self.callbacks.worker_heartbeat_required is not None
                    and self.callbacks.worker_heartbeat_required(command)
                )
                manual_heartbeat_waiver = False
                heartbeat_reference = current["heartbeat_at_ns"]
                if heartbeat_required:
                    heartbeat_age_seconds = (
                        now
                        - int(
                            current["started_at_ns"]
                            if heartbeat_reference is None
                            else heartbeat_reference
                        )
                    ) / 1e9
                    if (
                        heartbeat_age_seconds
                        > self.watchdog_policy.heartbeat_timeout_seconds
                    ):
                        manual_heartbeat_waiver = self._manual_heartbeat_waiver(
                            command,
                            current,
                        )
                        self.store._record_watchdog_finding_once(
                            event_type=(
                                "CHILD_HEARTBEAT_STALE_MANUALLY_ACKNOWLEDGED"
                                if manual_heartbeat_waiver
                                else "CHILD_HEARTBEAT_STALE"
                            ),
                            severity=(
                                "WARNING" if manual_heartbeat_waiver else "CRITICAL"
                            ),
                            row=self.store._require_attempt(
                                command.cell_id,
                                command.attempt,
                            ),
                            payload={
                                "heartbeat_age_seconds": heartbeat_age_seconds,
                                "heartbeat_sequence": current["heartbeat_sequence"],
                                "automatic_signal": False,
                                "hard_termination": False,
                                "manual_waiver": manual_heartbeat_waiver,
                            },
                            now_ns=now,
                            repeat_seconds=(self.watchdog_policy.event_repeat_seconds),
                        )
                        if not manual_heartbeat_waiver:
                            self.store.set_dispatch_stop(
                                "child_heartbeat_stale",
                                stopped_at_ns=now,
                            )
                            reconciled.append(
                                (command.cell_id, command.attempt, "STOP_GATE")
                            )
                            continue
                if current["termination_reason"] is not None:
                    reconciled.append(
                        (
                            command.cell_id,
                            command.attempt,
                            self._advance_termination(
                                command,
                                current,
                                now_ns=now,
                            ),
                        )
                    )
                    continue
                if runtime_seconds > command.max_runtime_seconds:
                    reconciled.append(
                        (
                            command.cell_id,
                            command.attempt,
                            self._begin_runtime_termination(
                                command,
                                current,
                                runtime_seconds=runtime_seconds,
                                now_ns=now,
                            ),
                        )
                    )
                    continue
                reconciled.append(
                    (
                        command.cell_id,
                        command.attempt,
                        (
                            "MANUAL_HEARTBEAT_WAIVER"
                            if manual_heartbeat_waiver
                            else "HEARTBEAT"
                            if heartbeat_recorded
                            else "OBSERVED"
                        ),
                    )
                )
                continue
            try:
                independent_pgids = (
                    ()
                    if self.callbacks.independent_process_groups is None
                    else self.callbacks.independent_process_groups(command)
                )
                if (
                    type(independent_pgids) is not tuple
                    or len(independent_pgids) != len(set(independent_pgids))
                    or any(
                        type(value) is not int or value < 1
                        for value in independent_pgids
                    )
                ):
                    raise ValueError("independent process-group targets differ")
                if independent_pgids and self.callbacks.process_group_alive is None:
                    raise RuntimeError(
                        "independent process-group targets lack a liveness probe"
                    )
                live_independent_pgids = tuple(
                    value
                    for value in independent_pgids
                    if self.callbacks.process_group_alive is not None
                    and self.callbacks.process_group_alive(value)
                )
                wrapper_group_alive = (
                    self.callbacks.process_group_alive is not None
                    and self.callbacks.process_group_alive(pgid)
                )
            except Exception as error:  # noqa: BLE001 - durable target boundary
                self.store.record_watchdog_event(
                    event_type="INDEPENDENT_PROCESS_GROUP_VALIDATION_FAILED",
                    severity="CRITICAL",
                    cell_id=command.cell_id,
                    attempt=command.attempt,
                    payload={"exception_type": type(error).__name__},
                    occurred_at_ns=now,
                )
                self.store.set_dispatch_stop(
                    "independent_process_group_validation_failed",
                    stopped_at_ns=now,
                )
                reconciled.append((command.cell_id, command.attempt, "STOPPED"))
                continue
            if live_independent_pgids:
                current = self.store.attempt(command.cell_id, command.attempt)
                is_resident_group = (
                    group is not None
                    and _physical_attempt_group_kind(group["members"])
                    == "tp1_serving_session"
                )
                if (
                    not is_resident_group
                    or self.callbacks.send_term is None
                    or self.callbacks.send_kill is None
                ):
                    self.store.record_watchdog_event(
                        event_type="WRAPPER_EXITED_WITH_LIVE_INDEPENDENT_PROCESS_GROUP",
                        severity="CRITICAL",
                        cell_id=command.cell_id,
                        attempt=command.attempt,
                        payload={
                            "pgids": list(live_independent_pgids),
                            "automatic_signal": False,
                        },
                        occurred_at_ns=now,
                    )
                    self.store.set_dispatch_stop(
                        "wrapper_exited_with_live_independent_process_group",
                        stopped_at_ns=now,
                    )
                    reconciled.append((command.cell_id, command.attempt, "STOP_GATE"))
                elif current["termination_reason"] is None:
                    try:
                        self.store.request_attempt_termination(
                            command.cell_id,
                            command.attempt,
                            reason="RESIDENT_SERVER_ORPHAN_AFTER_WRAPPER_EXIT",
                            requested_at_ns=now,
                        )
                        # Preserve the generic TERM-before-KILL lease ordering.
                        # The wrapper is already gone, so production TERM is a
                        # no-op after revalidating the durable target; KILL then
                        # reaches both the independent server PGID and any
                        # surviving wrapper descendants.
                        self.callbacks.send_term(command, pid, pgid)
                        self.store.record_attempt_termination_signal(
                            command.cell_id,
                            command.attempt,
                            signal_name="TERM",
                            sent_at_ns=now,
                        )
                        self.callbacks.send_kill(command, pid, pgid)
                        self.store.record_attempt_termination_signal(
                            command.cell_id,
                            command.attempt,
                            signal_name="KILL",
                            sent_at_ns=now,
                        )
                    except Exception as error:  # noqa: BLE001 - OS signal boundary
                        self.store.record_watchdog_event(
                            event_type="RESIDENT_ORPHAN_KILL_FAILED",
                            severity="CRITICAL",
                            cell_id=command.cell_id,
                            attempt=command.attempt,
                            payload={"exception_type": type(error).__name__},
                            occurred_at_ns=now,
                        )
                        self.store.set_dispatch_stop(
                            "resident_orphan_kill_failed",
                            stopped_at_ns=now,
                        )
                        reconciled.append(
                            (command.cell_id, command.attempt, "KILL_FAILED")
                        )
                    else:
                        self.store.record_watchdog_event(
                            event_type="RESIDENT_ORPHAN_KILL_SENT",
                            severity="CRITICAL",
                            cell_id=command.cell_id,
                            attempt=command.attempt,
                            payload={
                                "server_pgids": list(live_independent_pgids),
                                "wrapper_pgid": pgid,
                                "source_bound_targets": True,
                            },
                            occurred_at_ns=now,
                        )
                        reconciled.append(
                            (command.cell_id, command.attempt, "RESIDENT_KILL_SENT")
                        )
                else:
                    kill_sent_at = current["kill_sent_at_ns"]
                    if (
                        kill_sent_at is not None
                        and (now - int(kill_sent_at)) / 1e9
                        > self.watchdog_policy.termination_grace_seconds
                    ):
                        self.store._record_watchdog_finding_once(
                            event_type="RESIDENT_PROCESS_GROUP_ALIVE_AFTER_KILL",
                            severity="CRITICAL",
                            row=self.store._require_attempt(
                                command.cell_id, command.attempt
                            ),
                            payload={"server_pgids": list(live_independent_pgids)},
                            now_ns=now,
                            repeat_seconds=self.watchdog_policy.event_repeat_seconds,
                        )
                        self.store.set_dispatch_stop(
                            "resident_process_group_alive_after_kill",
                            stopped_at_ns=now,
                        )
                    reconciled.append(
                        (command.cell_id, command.attempt, "WAITING_RESIDENT_EXIT")
                    )
                continue
            if wrapper_group_alive:
                current = self.store.attempt(command.cell_id, command.attempt)
                if current["termination_reason"] is not None:
                    reconciled.append(
                        (
                            command.cell_id,
                            command.attempt,
                            self._advance_termination(command, current, now_ns=now),
                        )
                    )
                else:
                    self.store.record_watchdog_event(
                        event_type="WRAPPER_EXITED_WITH_LIVE_PROCESS_GROUP",
                        severity="CRITICAL",
                        cell_id=command.cell_id,
                        attempt=command.attempt,
                        payload={"pgid": pgid, "automatic_signal": False},
                        occurred_at_ns=now,
                    )
                    self.store.set_dispatch_stop(
                        "wrapper_exited_with_live_process_group",
                        stopped_at_ns=now,
                    )
                    reconciled.append((command.cell_id, command.attempt, "STOP_GATE"))
                continue
            current = self.store.attempt(command.cell_id, command.attempt)
            termination_requested = current["termination_reason"] is not None
            try:
                commands = (
                    (command,)
                    if group is None
                    else self.store.physical_attempt_group_commands(
                        str(group["group_id"])
                    )
                )
                terminals: dict[str, TerminalEvidence] = {}
                waiting = False
                for logical_command in commands:
                    logical_attempt = self.store.attempt(
                        logical_command.cell_id,
                        logical_command.attempt,
                    )
                    terminal = self.callbacks.terminal_validator(
                        logical_command,
                        logical_attempt,
                        observation,
                    )
                    if terminal is not None and type(terminal) is not TerminalEvidence:
                        raise TypeError("terminal validator returned another type")
                    if terminal is None:
                        waiting = True
                    else:
                        terminals[logical_command.cell_id] = terminal
            except Exception as error:  # noqa: BLE001 - validator fail-closed boundary
                if termination_requested:
                    self._terminalize_forced_termination(
                        command,
                        group=group,
                        attempt=current,
                        now_ns=now,
                        validator_error=error,
                    )
                    reconciled.append((command.cell_id, command.attempt, "FAILED"))
                    continue
                self.store.record_watchdog_event(
                    event_type="TERMINAL_VALIDATION_FAILED",
                    severity="CRITICAL",
                    cell_id=command.cell_id,
                    attempt=command.attempt,
                    payload={"exception_type": type(error).__name__},
                    occurred_at_ns=now,
                )
                self.store.set_dispatch_stop(
                    "terminal_evidence_validation_failed",
                    stopped_at_ns=now,
                )
                reconciled.append((command.cell_id, command.attempt, "STOPPED"))
                continue
            if waiting:
                if termination_requested:
                    self._terminalize_forced_termination(
                        command,
                        group=group,
                        attempt=current,
                        now_ns=now,
                        validator_error=None,
                    )
                    reconciled.append((command.cell_id, command.attempt, "FAILED"))
                    continue
                if (
                    group is not None
                    and _physical_attempt_group_kind(group["members"])
                    == "tp1_serving_session"
                ):
                    # The shared wrapper is already dead and its PGID is empty.
                    # A missing member terminal can therefore never become an
                    # atomic close/fanout publication.  Preserve the immutable
                    # prefix, fail every logical attempt together, then rebuild
                    # each member as an independent fresh attempt.  Never
                    # reconstruct the failed resident group on retry.
                    evidence: Mapping[str, str] = {}
                    if self.callbacks.partial_evidence is not None:
                        try:
                            evidence = self.callbacks.partial_evidence(command)
                        except Exception as error:  # noqa: BLE001
                            self.store.record_watchdog_event(
                                event_type="PARTIAL_EVIDENCE_COLLECTION_FAILED",
                                severity="ERROR",
                                cell_id=command.cell_id,
                                attempt=command.attempt,
                                payload={"exception_type": type(error).__name__},
                                occurred_at_ns=now,
                            )
                    self.store.fail_physical_attempt_group_infrastructure(
                        str(group["group_id"]),
                        failure_code=("INFRASTRUCTURE:RESIDENT_GROUP_UNSEALED_EXIT"),
                        exclusion_reason=(
                            "resident_wrapper_exited_without_atomic_close_fanout"
                        ),
                        evidence_files=evidence,
                        finished_at_ns=now,
                    )
                    self._materialize_serving_group_retries(group, commands)
                    for logical_command in commands:
                        reconciled.append(
                            (
                                logical_command.cell_id,
                                logical_command.attempt,
                                "FAILED",
                            )
                        )
                    continue
                self.store.record_watchdog_event(
                    event_type="TERMINAL_EVIDENCE_NOT_YET_ATOMIC",
                    severity="WARNING",
                    cell_id=command.cell_id,
                    attempt=command.attempt,
                    payload={"process_reason": observation.reason},
                    occurred_at_ns=now,
                )
                reconciled.append((command.cell_id, command.attempt, "WAITING"))
                continue
            if group is None:
                terminal = terminals[command.cell_id]
                self._finish_from_terminal(command, terminal, now_ns=now)
                reconciled.append((command.cell_id, command.attempt, terminal.status))
            else:
                try:
                    self.store.finish_physical_attempt_group(
                        str(group["group_id"]),
                        terminals=terminals,
                        finished_at_ns=now,
                    )
                except Exception as error:  # noqa: BLE001 - group atomic boundary
                    self.store.record_watchdog_event(
                        event_type="PHYSICAL_GROUP_TERMINAL_FANOUT_FAILED",
                        severity="CRITICAL",
                        cell_id=command.cell_id,
                        attempt=command.attempt,
                        payload={
                            "group_id": str(group["group_id"]),
                            "exception_type": type(error).__name__,
                        },
                        occurred_at_ns=now,
                    )
                    self.store.set_dispatch_stop(
                        "physical_group_terminal_fanout_failed",
                        stopped_at_ns=now,
                    )
                    reconciled.append((command.cell_id, command.attempt, "STOPPED"))
                    continue
                self._materialize_serving_group_retries(
                    group,
                    tuple(
                        logical_command
                        for logical_command in commands
                        if terminals[logical_command.cell_id].status == "FAILED"
                        and terminals[logical_command.cell_id].failure_class
                        == "INFRASTRUCTURE"
                    ),
                )
                for logical_command in commands:
                    reconciled.append(
                        (
                            logical_command.cell_id,
                            logical_command.attempt,
                            terminals[logical_command.cell_id].status,
                        )
                    )
        self.store.watchdog_once(
            policy=self.watchdog_policy,
            process_probe=self.callbacks.process_probe,
            now_ns=now,
        )
        return tuple(reconciled)

    def _begin_runtime_termination(
        self,
        command: QueuedCommandSpec,
        attempt: Mapping[str, Any],
        *,
        runtime_seconds: float,
        now_ns: int,
    ) -> str:
        self.store.request_attempt_termination(
            command.cell_id,
            command.attempt,
            reason="SOURCE_BOUND_RUNTIME_LIMIT",
            requested_at_ns=now_ns,
        )
        self.store.record_watchdog_event(
            event_type="COMMAND_RUNTIME_LIMIT_EXCEEDED",
            severity="CRITICAL",
            cell_id=command.cell_id,
            attempt=command.attempt,
            payload={
                "runtime_seconds": runtime_seconds,
                "max_runtime_seconds": command.max_runtime_seconds,
                "termination_intent_persisted": True,
            },
            occurred_at_ns=now_ns,
        )
        state, _reason = self.store.dispatch_control()
        if state != "STOP":
            self.store.set_dispatch_stop(
                "command_runtime_limit_exceeded",
                stopped_at_ns=now_ns,
            )
        current = self.store.attempt(command.cell_id, command.attempt)
        return self._advance_termination(command, current, now_ns=now_ns)

    def _advance_termination(
        self,
        command: QueuedCommandSpec,
        attempt: Mapping[str, Any],
        *,
        now_ns: int,
    ) -> str:
        pid = attempt["pid"]
        pgid = attempt["pgid"]
        if type(pid) is not int or type(pgid) is not int:
            self.store.set_dispatch_stop(
                "terminating_attempt_lacks_process_metadata",
                stopped_at_ns=now_ns,
            )
            return "STOP_NO_PROCESS_METADATA"
        term_sent = attempt["term_sent_at_ns"]
        kill_sent = attempt["kill_sent_at_ns"]
        if term_sent is None:
            callback = self.callbacks.send_term
            if callback is None:
                self.store.record_watchdog_event(
                    event_type="TERMINATION_CALLBACK_UNAVAILABLE",
                    severity="CRITICAL",
                    cell_id=command.cell_id,
                    attempt=command.attempt,
                    payload={"requested_signal": "TERM"},
                    occurred_at_ns=now_ns,
                )
                self.store.set_dispatch_stop(
                    "termination_callback_unavailable",
                    stopped_at_ns=now_ns,
                )
                return "STOP_NO_TERM_CALLBACK"
            try:
                callback(command, pid, pgid)
                self.store.record_attempt_termination_signal(
                    command.cell_id,
                    command.attempt,
                    signal_name="TERM",
                    sent_at_ns=now_ns,
                )
            except Exception as error:  # noqa: BLE001 - OS signal boundary
                self.store.record_watchdog_event(
                    event_type="PROCESS_TERM_FAILED",
                    severity="CRITICAL",
                    cell_id=command.cell_id,
                    attempt=command.attempt,
                    payload={"exception_type": type(error).__name__},
                    occurred_at_ns=now_ns,
                )
                self.store.set_dispatch_stop(
                    "process_term_failed",
                    stopped_at_ns=now_ns,
                )
                return "TERM_FAILED"
            self.store.record_watchdog_event(
                event_type="PROCESS_TERM_SENT",
                severity="ERROR",
                cell_id=command.cell_id,
                attempt=command.attempt,
                payload={"pid": pid, "pgid": pgid, "target": "wrapper_pid"},
                occurred_at_ns=now_ns,
            )
            return "TERM_SENT"
        term_age_seconds = (now_ns - int(term_sent)) / 1e9
        if kill_sent is None and (
            term_age_seconds > self.watchdog_policy.termination_grace_seconds
        ):
            callback = self.callbacks.send_kill
            if callback is None:
                self.store.record_watchdog_event(
                    event_type="TERMINATION_CALLBACK_UNAVAILABLE",
                    severity="CRITICAL",
                    cell_id=command.cell_id,
                    attempt=command.attempt,
                    payload={"requested_signal": "KILL"},
                    occurred_at_ns=now_ns,
                )
                self.store.set_dispatch_stop(
                    "termination_callback_unavailable",
                    stopped_at_ns=now_ns,
                )
                return "STOP_NO_KILL_CALLBACK"
            try:
                callback(command, pid, pgid)
                self.store.record_attempt_termination_signal(
                    command.cell_id,
                    command.attempt,
                    signal_name="KILL",
                    sent_at_ns=now_ns,
                )
            except Exception as error:  # noqa: BLE001 - OS signal boundary
                self.store.record_watchdog_event(
                    event_type="PROCESS_KILL_FAILED",
                    severity="CRITICAL",
                    cell_id=command.cell_id,
                    attempt=command.attempt,
                    payload={"exception_type": type(error).__name__},
                    occurred_at_ns=now_ns,
                )
                self.store.set_dispatch_stop(
                    "process_kill_failed",
                    stopped_at_ns=now_ns,
                )
                return "KILL_FAILED"
            self.store.record_watchdog_event(
                event_type="PROCESS_GROUP_KILL_SENT",
                severity="CRITICAL",
                cell_id=command.cell_id,
                attempt=command.attempt,
                payload={"pid": pid, "pgid": pgid, "target": "process_group"},
                occurred_at_ns=now_ns,
            )
            return "KILL_SENT"
        if kill_sent is not None:
            kill_age_seconds = (now_ns - int(kill_sent)) / 1e9
            if kill_age_seconds > self.watchdog_policy.termination_grace_seconds:
                self.store._record_watchdog_finding_once(
                    event_type="PROCESS_GROUP_ALIVE_AFTER_KILL",
                    severity="CRITICAL",
                    row=self.store._require_attempt(command.cell_id, command.attempt),
                    payload={
                        "pgid": pgid,
                        "kill_age_seconds": kill_age_seconds,
                    },
                    now_ns=now_ns,
                    repeat_seconds=self.watchdog_policy.event_repeat_seconds,
                )
            return "WAITING_GROUP_EXIT_AFTER_KILL"
        return "WAITING_TERM_GRACE"

    def _terminalize_forced_termination(
        self,
        command: QueuedCommandSpec,
        *,
        group: Mapping[str, Any] | None,
        attempt: Mapping[str, Any],
        now_ns: int,
        validator_error: BaseException | None,
    ) -> None:
        evidence: Mapping[str, str] = {}
        if self.callbacks.partial_evidence is not None:
            try:
                evidence = self.callbacks.partial_evidence(command)
            except Exception as error:  # noqa: BLE001 - partial evidence boundary
                self.store.record_watchdog_event(
                    event_type="PARTIAL_EVIDENCE_COLLECTION_FAILED",
                    severity="ERROR",
                    cell_id=command.cell_id,
                    attempt=command.attempt,
                    payload={"exception_type": type(error).__name__},
                    occurred_at_ns=now_ns,
                )
        reason = str(attempt["termination_reason"])
        exclusion = f"forced_termination:{reason.lower()}"
        if validator_error is not None:
            exclusion += f":{type(validator_error).__name__}"
        if group is not None:
            group_commands = self.store.physical_attempt_group_commands(
                str(group["group_id"])
            )
            resident_orphan = (
                _physical_attempt_group_kind(group["members"]) == "tp1_serving_session"
                and reason == "RESIDENT_SERVER_ORPHAN_AFTER_WRAPPER_EXIT"
            )
            self.store.fail_physical_attempt_group_infrastructure(
                str(group["group_id"]),
                failure_code=(
                    "INFRASTRUCTURE:RESIDENT_GROUP_UNSEALED_EXIT"
                    if resident_orphan
                    else "INFRASTRUCTURE:RUNTIME_LIMIT_EXCEEDED"
                ),
                exclusion_reason=(
                    "resident_wrapper_exited_without_atomic_close_fanout"
                    if resident_orphan
                    else exclusion
                ),
                evidence_files=evidence,
                finished_at_ns=now_ns,
            )
            self._materialize_serving_group_retries(group, group_commands)
            return
        automatic_retry = (
            self.store.infrastructure_failure_count(command.cell_id) < 2
            and self.callbacks.retry_builder is not None
        )
        duration_seconds = max(
            0.0,
            (now_ns - int(attempt["started_at_ns"])) / 1e9,
        )
        gpu_count = len(attempt["assigned_gpu_uuids"])
        self.store.finish_attempt(
            command.cell_id,
            command.attempt,
            status="FAILED",
            exit_code=None,
            evidence_files=evidence,
            failure_code="INFRASTRUCTURE:RUNTIME_LIMIT_EXCEEDED",
            retry_decision=(
                "RETRY_INFRASTRUCTURE_AUTOMATIC"
                if automatic_retry
                else "NO_RETRY_BUILDER_OR_LIMIT"
            ),
            included_in_analysis=False,
            exclusion_reason=exclusion,
            compute_gpu_seconds=duration_seconds * gpu_count,
            reserved_gpu_seconds=duration_seconds * gpu_count,
            finished_at_ns=now_ns,
        )
        if automatic_retry:
            self._materialize_retry(command)
        self.store.record_watchdog_event(
            event_type="FORCED_TERMINATION_TERMINALIZED_INFRASTRUCTURE",
            severity="ERROR",
            cell_id=command.cell_id,
            attempt=command.attempt,
            payload={
                "termination_reason": reason,
                "partial_evidence_count": len(evidence),
            },
            occurred_at_ns=now_ns,
        )

    def _finish_from_terminal(
        self,
        command: QueuedCommandSpec,
        terminal: TerminalEvidence,
        *,
        now_ns: int,
    ) -> None:
        attempt = self.store.attempt(command.cell_id, command.attempt)
        gpu_count = len(attempt["assigned_gpu_uuids"])
        compute_gpu_seconds = 0.0
        reserved_gpu_seconds = 0.0
        finished_at_ns = now_ns
        if terminal.started_ns is not None and terminal.finished_ns is not None:
            duration_seconds = (terminal.finished_ns - terminal.started_ns) / 1e9
            compute_gpu_seconds = duration_seconds * gpu_count
            reserved_gpu_seconds = duration_seconds * gpu_count
            finished_at_ns = terminal.finished_ns
        if terminal.status == "COMPLETE":
            self.store.finish_attempt(
                command.cell_id,
                command.attempt,
                status="COMPLETE",
                exit_code=terminal.exit_code,
                terminal_sha256=terminal.terminal_sha256,
                junit_sha256=terminal.junit_sha256,
                raw_log_sha256=terminal.raw_log_sha256,
                evidence_files=terminal.evidence_files,
                included_in_analysis=terminal.included_in_analysis,
                exclusion_reason=terminal.exclusion_reason,
                compute_gpu_seconds=compute_gpu_seconds,
                reserved_gpu_seconds=reserved_gpu_seconds,
                finished_at_ns=finished_at_ns,
            )
            self.store.record_watchdog_event(
                event_type="ATOMIC_TERMINAL_ACCEPTED",
                severity="INFO",
                cell_id=command.cell_id,
                attempt=command.attempt,
                payload={
                    "atomic_publication_sha256": (terminal.atomic_publication_sha256)
                },
                occurred_at_ns=now_ns,
            )
            return
        assert terminal.failure_class is not None
        automatic_retry = (
            terminal.failure_class == "INFRASTRUCTURE"
            and self.store.infrastructure_failure_count(command.cell_id) < 2
            and self.callbacks.retry_builder is not None
        )
        failure_code = f"{terminal.failure_class}:{terminal.failure_code}"
        self.store.finish_attempt(
            command.cell_id,
            command.attempt,
            status="FAILED",
            exit_code=terminal.exit_code,
            terminal_sha256=terminal.terminal_sha256,
            junit_sha256=terminal.junit_sha256,
            raw_log_sha256=terminal.raw_log_sha256,
            evidence_files=terminal.evidence_files,
            failure_code=failure_code,
            retry_decision=(
                "RETRY_INFRASTRUCTURE_AUTOMATIC"
                if automatic_retry
                else "NO_BLIND_RETRY"
            ),
            included_in_analysis=False,
            exclusion_reason=terminal.exclusion_reason,
            compute_gpu_seconds=compute_gpu_seconds,
            reserved_gpu_seconds=reserved_gpu_seconds,
            finished_at_ns=finished_at_ns,
        )
        if automatic_retry:
            self._materialize_retry(command)

    def _terminalize_spawn_failure(
        self,
        command: QueuedCommandSpec,
        error: BaseException,
    ) -> None:
        now = self._now()
        automatic_retry = (
            self.store.infrastructure_failure_count(command.cell_id) < 2
            and self.callbacks.retry_builder is not None
        )
        self.store.finish_attempt(
            command.cell_id,
            command.attempt,
            status="FAILED",
            exit_code=None,
            failure_code="INFRASTRUCTURE:SPAWN_FAILED",
            retry_decision=(
                "RETRY_INFRASTRUCTURE_AUTOMATIC"
                if automatic_retry
                else "NO_RETRY_BUILDER_OR_LIMIT"
            ),
            included_in_analysis=False,
            exclusion_reason=f"spawn_failed:{type(error).__name__}",
            finished_at_ns=now,
        )
        if automatic_retry:
            self._materialize_retry(command)

    def _terminalize_missing_start_receipt(
        self,
        command: QueuedCommandSpec,
        *,
        now_ns: int,
    ) -> None:
        automatic_retry = (
            self.store.infrastructure_failure_count(command.cell_id) < 2
            and self.callbacks.retry_builder is not None
        )
        self.store.finish_attempt(
            command.cell_id,
            command.attempt,
            status="FAILED",
            exit_code=None,
            failure_code="INFRASTRUCTURE:START_RECEIPT_NOT_PUBLISHED",
            retry_decision=(
                "RETRY_INFRASTRUCTURE_AUTOMATIC"
                if automatic_retry
                else "NO_RETRY_BUILDER_OR_LIMIT"
            ),
            included_in_analysis=False,
            exclusion_reason="wrapper_start_receipt_absent_after_attach_grace",
            finished_at_ns=now_ns,
        )
        if automatic_retry:
            self._materialize_retry(command)

    def _materialize_retry(self, command: QueuedCommandSpec) -> None:
        builder = self.callbacks.retry_builder
        if builder is None:
            return
        next_attempt = command.attempt + 1
        try:
            spec, retry_command = builder(command, next_attempt)
            if (
                type(spec) is not CellAttemptSpec
                or type(retry_command) is not QueuedCommandSpec
                or (spec.cell_id, spec.attempt) != (command.cell_id, next_attempt)
                or (retry_command.cell_id, retry_command.attempt)
                != (command.cell_id, next_attempt)
            ):
                raise ValueError("retry builder returned a different attempt")
            prior = self.store.attempt(command.cell_id, command.attempt)
            if spec.output_directory == prior["output_directory"]:
                raise ValueError("retry builder reused the prior output directory")
            prior_evidence_paths = {
                command.log_path,
                command.expected_terminal_path,
                command.expected_junit_path,
                command.expected_raw_log_path,
                command.atomic_pointer_path,
                command.child_exit_receipt_path,
            }
            retry_evidence_paths = {
                retry_command.log_path,
                retry_command.expected_terminal_path,
                retry_command.expected_junit_path,
                retry_command.expected_raw_log_path,
                retry_command.atomic_pointer_path,
                retry_command.child_exit_receipt_path,
            }
            if prior_evidence_paths & retry_evidence_paths:
                raise ValueError("retry builder reused prior attempt evidence paths")
            self.store.materialize_attempt(spec)
            self.store.enqueue_command(retry_command)
            self.store.record_watchdog_event(
                event_type="INFRASTRUCTURE_RETRY_QUEUED",
                severity="WARNING",
                cell_id=command.cell_id,
                attempt=next_attempt,
                payload={"prior_attempt": command.attempt},
                occurred_at_ns=self._now(),
            )
        except Exception as error:  # noqa: BLE001 - injected retry-builder boundary
            self.store.record_watchdog_event(
                event_type="INFRASTRUCTURE_RETRY_BUILD_FAILED",
                severity="CRITICAL",
                cell_id=command.cell_id,
                attempt=command.attempt,
                payload={"exception_type": type(error).__name__},
                occurred_at_ns=self._now(),
            )
            self.store.set_dispatch_stop(
                "infrastructure_retry_builder_failed",
                stopped_at_ns=self._now(),
            )

    def _materialize_serving_group_retries(
        self,
        group: Mapping[str, Any],
        commands: Sequence[QueuedCommandSpec],
    ) -> None:
        """Retry failed resident members only as independent fresh attempts."""

        members = group.get("members")
        if (
            type(members) is not tuple
            or not (2 <= len(members) <= 32)
            or {row.get("logical_kind") for row in members if type(row) is dict}
            != {"serving"}
            or self.callbacks.retry_builder is None
        ):
            return
        for command in commands:
            previous = self.store.attempt(command.cell_id, command.attempt)
            if (
                previous["status"] == "FAILED"
                and previous["retry_decision"] == "RETRY_INFRASTRUCTURE_AUTOMATIC"
                and self.store.infrastructure_failure_count(command.cell_id) < 2
            ):
                self._materialize_retry(command)

    def _select_dispatch(
        self,
        envelope: InterferenceEnvelope,
    ) -> tuple[QueuedCommandSpec, tuple[str, ...]] | None:
        snapshot = self.store.snapshot()
        running = [row for row in snapshot["attempts"] if row["status"] == "RUNNING"]
        running_commands = self.store.physical_commands(status="RUNNING")
        occupied = {
            gpu_uuid for row in running for gpu_uuid in row["assigned_gpu_uuids"]
        }
        inventory = envelope.gpu_uuids
        if not occupied <= set(inventory):
            self.store.set_dispatch_stop(
                "running_attempt_uses_gpu_outside_interference_inventory",
                stopped_at_ns=self._now(),
            )
            return None
        preferred = self.store.last_launch_compatibility_key()
        pending = sorted(
            self.store.physical_commands(status="PENDING"),
            key=lambda row: (
                -row.priority,
                row.launch_compatibility_key != preferred,
                row.launch_compatibility_key,
                row.cell_id,
                row.attempt,
            ),
        )
        available = tuple(gpu for gpu in inventory if gpu not in occupied)
        exclusive_classes = {"EXCLUSIVE", "PROFILER", "FAILURE", "ARCHIVE"}
        if any(
            command.timing_class in exclusive_classes for command in running_commands
        ):
            return None
        for command in pending:
            if command.required_gpu_count > len(inventory):
                continue
            if (
                command.required_gpu_count > 1
                or command.timing_class in exclusive_classes
            ):
                if running or command.required_gpu_count > len(available):
                    continue
                return command, available[: command.required_gpu_count]
            if envelope.mode == "ISOLATED" and occupied:
                continue
            if command.timing_class == "HEADLINE" and envelope.mode == "UNRESOLVED":
                continue
            if command.preferred_gpu_index is not None:
                if command.preferred_gpu_index >= len(inventory):
                    self.store.set_dispatch_stop(
                        "preferred_gpu_index_outside_interference_inventory",
                        stopped_at_ns=self._now(),
                    )
                    return None
                preferred_gpu = inventory[command.preferred_gpu_index]
                if preferred_gpu in available:
                    return command, (preferred_gpu,)
                continue
            if available:
                return command, (available[0],)
        return None

    def _now(self) -> int:
        value = self.clock_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("scheduler clock must return positive nanoseconds")
        return value


def inspect_local_process(pid: int, expected_pgid: int) -> ProcessObservation:
    """Read-only local PID/PGID check that detects PID reuse."""

    _require_positive_int(pid, "PID")
    _require_positive_int(expected_pgid, "expected PGID")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return ProcessObservation(pid, False, None, "process_not_found")
    except PermissionError:
        reason = "process_exists_permission_denied"
    else:
        reason = "process_exists"
    try:
        observed_pgid = os.getpgid(pid)
    except ProcessLookupError:
        return ProcessObservation(pid, False, None, "process_exited_during_probe")
    except PermissionError:
        return ProcessObservation(pid, True, None, reason)
    alive = observed_pgid == expected_pgid
    return ProcessObservation(
        pid=pid,
        alive=alive,
        observed_pgid=observed_pgid,
        reason=reason if alive else "pgid_mismatch_or_pid_reuse",
    )


def evaluate_dispatch_disk_gate(
    *,
    free_bytes: int,
    predicted_next_wave_high_water_bytes: int,
    safety_reserve_bytes: int = REMOTE_SPOOL_SAFETY_RESERVE_BYTES,
) -> DiskDispatchDecision:
    """Require next-wave high-water plus the fixed spool safety reserve."""

    for label, value in (
        ("free bytes", free_bytes),
        ("predicted next-wave high-water bytes", predicted_next_wave_high_water_bytes),
        ("safety reserve bytes", safety_reserve_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    required = predicted_next_wave_high_water_bytes + safety_reserve_bytes
    allowed = free_bytes >= required
    return DiskDispatchDecision(
        action="ALLOW" if allowed else "STOP",
        free_bytes=free_bytes,
        predicted_next_wave_high_water_bytes=predicted_next_wave_high_water_bytes,
        safety_reserve_bytes=safety_reserve_bytes,
        required_free_bytes=required,
        reason=(
            "capacity_covers_next_wave_high_water_plus_reserve"
            if allowed
            else "insufficient_capacity_for_next_wave_high_water_plus_reserve"
        ),
    )


def _archive_previous_receipt(row: sqlite3.Row) -> ArchiveStepReceipt | None:
    for column in (
        "rehydrate_receipt_json",
        "local_sha_receipt_json",
        "transfer_receipt_json",
    ):
        if row[column] is not None:
            return ArchiveStepReceipt(**json.loads(row[column]))
    return None


def _decoded_archive(row: sqlite3.Row) -> dict[str, Any]:
    output = dict(row)
    for column in (
        "transfer_receipt_json",
        "local_sha_receipt_json",
        "rehydrate_receipt_json",
    ):
        value = output.pop(column)
        output[column.removesuffix("_json")] = (
            None if value is None else json.loads(value)
        )
    for source, target in (
        ("created_at_ns", "created_at"),
        ("updated_at_ns", "updated_at"),
        ("eviction_authorized_at_ns", "eviction_authorized_at"),
    ):
        value = output[source]
        output[target] = None if value is None else _iso_utc(int(value))
    return output


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{label} cannot contain NUL")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_nonnegative_finite(value: object, label: str) -> float:
    result = _require_finite(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _require_nonnegative_finite_or_none(value: object, label: str) -> None:
    if value is not None:
        _require_nonnegative_finite(value, label)


def _canonical_mapping(
    value: Mapping[str, Any], label: str, *, allow_empty: bool
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result = dict(value)
    if not allow_empty and not result:
        raise ValueError(f"{label} cannot be empty")
    if not all(isinstance(key, str) and key for key in result):
        raise ValueError(f"{label} keys must be non-empty strings")
    try:
        _canonical_json(result)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be canonical JSON data") from error
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validated_gpu_uuids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("assigned GPU UUIDs must be a sequence")
    result = tuple(_require_text(value, "GPU UUID") for value in values)
    if len(set(result)) != len(result):
        raise ValueError("assigned GPU UUIDs contain duplicates")
    return result


def _decoded_command(row: sqlite3.Row) -> QueuedCommandSpec:
    argv = json.loads(row["argv_json"])
    if type(argv) is not list:
        raise ExperimentOperatorError("queued argv storage differs")
    environment = json.loads(row["environment_json"])
    if type(environment) is not list or any(
        type(value) is not list for value in environment
    ):
        raise ExperimentOperatorError("queued environment storage differs")
    return QueuedCommandSpec(
        cell_id=row["cell_id"],
        attempt=int(row["attempt"]),
        argv=tuple(argv),
        environment=tuple(tuple(value) for value in environment),
        launch_compatibility_key=row["launch_compatibility_key"],
        required_gpu_count=int(row["required_gpu_count"]),
        timing_class=row["timing_class"],
        predicted_high_water_bytes=int(row["predicted_high_water_bytes"]),
        monitored_path=row["monitored_path"],
        log_path=row["log_path"],
        expected_terminal_path=row["expected_terminal_path"],
        expected_junit_path=row["expected_junit_path"],
        expected_raw_log_path=row["expected_raw_log_path"],
        atomic_pointer_path=row["atomic_pointer_path"],
        child_exit_receipt_path=row["child_exit_receipt_path"],
        paired_gpu_key=row["paired_gpu_key"],
        preferred_gpu_index=row["preferred_gpu_index"],
        priority=int(row["priority"]),
        max_runtime_seconds=int(row["max_runtime_seconds"]),
        max_log_stall_seconds=int(row["max_log_stall_seconds"]),
    )


def _decoded_controller_node(row: sqlite3.Row) -> dict[str, Any]:
    output = dict(row)
    auxiliary = json.loads(output.pop("auxiliary_sources_json"))
    if type(auxiliary) is not dict:
        raise ExperimentOperatorError("controller auxiliary sources storage differs")
    output["auxiliary_sources"] = auxiliary
    output["updated_at"] = _iso_utc(int(output["updated_at_ns"]))
    return output


def _decoded_controller_auxiliary_job(row: sqlite3.Row) -> dict[str, Any]:
    output = dict(row)
    output["job_attempt"] = int(output["job_attempt"])
    output["group_attempt"] = int(output["group_attempt"])
    output["scientific_axes"] = json.loads(output.pop("scientific_axes_json"))
    output["identity"] = json.loads(output.pop("identity_json"))
    output["evidence_files"] = json.loads(output.pop("evidence_files_json"))
    output["included_in_analysis"] = bool(output["included_in_analysis"])
    return output


def _require_cell_attempt_spec_matches(
    actual: Mapping[str, Any],
    expected: CellAttemptSpec,
) -> None:
    projection = {
        "cell_id": actual["cell_id"],
        "attempt": actual["attempt"],
        "stage": actual["stage"],
        "phase": actual["phase"],
        "block": actual["block_id"],
        "seed": actual["seed"],
        "scientific_axes": actual["scientific_axes"],
        "identity": actual["identity"],
        "command_sha256": actual["command_sha256"],
        "scientific_command_sha256": actual["scientific_command_sha256"],
        "output_directory": actual["output_directory"],
    }
    expected_projection = {
        "cell_id": expected.cell_id,
        "attempt": expected.attempt,
        "stage": expected.stage,
        "phase": expected.phase,
        "block": expected.block,
        "seed": expected.seed,
        "scientific_axes": dict(expected.scientific_axes),
        "identity": dict(expected.identity),
        "command_sha256": expected.command_sha256,
        "scientific_command_sha256": expected.scientific_command_sha256,
        "output_directory": expected.output_directory,
    }
    if projection != expected_projection or actual["status"] != "COMPLETE":
        raise ExperimentOperatorError(
            "durable adopted attempt differs from auxiliary adoption"
        )


def _decoded_attempt(row: sqlite3.Row) -> dict[str, Any]:
    output = dict(row)
    for column in (
        "scientific_axes_json",
        "identity_json",
        "assigned_gpu_uuids_json",
        "gpu_observation_json",
        "evidence_files_json",
    ):
        value = output.pop(column)
        output[column.removesuffix("_json")] = (
            None if value is None else json.loads(value)
        )
    output["included_in_analysis"] = bool(output["included_in_analysis"])
    output["is_legacy_import"] = bool(output["is_legacy_import"])
    for source, target in (
        ("created_at_ns", "created_at"),
        ("updated_at_ns", "updated_at"),
        ("started_at_ns", "started_at"),
        ("finished_at_ns", "finished_at"),
        ("heartbeat_at_ns", "heartbeat_at"),
    ):
        value = output[source]
        output[target] = None if value is None else _iso_utc(int(value))
    return output


def _iso_utc(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds))
    return f"{base}.{nanoseconds:09d}Z"


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                column: (
                    _canonical_json(value)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for column, value in row.items()
            }
        )
    return buffer.getvalue()


def _stage_plan_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "node",
        "ordinal",
        "stage",
        "phase",
        "expected_formula",
        "known_expected_cells",
        "materialized_cells",
        "completed",
        "running",
        "failed",
        "blocked",
        "stale",
        "progress_percent",
        "actual_gpu_hours",
        "reserved_gpu_hours",
        "billed_gpu_hours",
        "estimated_remaining_gpu_hours",
        "last_update",
    )
    return _csv_text(rows, columns)


def _stage_summary_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "stage",
        "phase",
        "completed",
        "materialized_cells",
        "progress_percent",
        "actual_gpu_hours",
        "reserved_gpu_hours",
        "billed_gpu_hours",
    )
    return _csv_text(rows, columns)


def _cell_ledger_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "stage",
        "phase",
        "cell_id",
        "attempt",
        "block_id",
        "seed",
        "scientific_axes",
        "identity",
        "is_legacy_import",
        "legacy_original_status",
        "status",
        "assigned_gpu_uuids",
        "pid",
        "pgid",
        "started_at",
        "finished_at",
        "command_sha256",
        "scientific_command_sha256",
        "output_directory",
        "terminal_sha256",
        "junit_sha256",
        "raw_log_sha256",
        "evidence_files",
        "failure_code",
        "retry_decision",
        "included_in_analysis",
        "exclusion_reason",
        "heartbeat_at",
        "heartbeat_sequence",
        "physical_group_id",
        "physical_group_logical_kind",
        "physical_group_evidence_sha256",
        "physical_accounting_owner",
        "compute_gpu_seconds",
        "reserved_gpu_seconds",
        "billed_gpu_seconds",
    )
    return _csv_text(rows, columns)


def _json_lines(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json(dict(row)) + "\n" for row in rows)


def _provider_billing_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "instance_uuid",
        "provider_started_at_ns",
        "provider_stopped_or_observed_at_ns",
        "complete",
        "gpu_count",
        "duration_seconds",
        "whole_instance_billed_gpu_seconds",
        "sample_count",
        "response_sha256s",
    )
    return _csv_text(rows, columns)


def _controller_state_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "node",
        "ordinal",
        "state",
        "materialization_path",
        "materialization_sha256",
        "node_materialization_path",
        "node_materialization_sha256",
        "execution_source_path",
        "execution_source_sha256",
        "prepared_launch_path",
        "prepared_launch_sha256",
        "decision_path",
        "decision_sha256",
        "completion_path",
        "completion_sha256",
        "expected_cell_count",
        "expected_cell_ids_sha256",
        "auxiliary_sources",
        "blocker_reason",
        "updated_at",
    )
    return _csv_text(rows, columns)


def _dashboard_markdown(
    run_id: str,
    exported_at_ns: int,
    stage_rows: Sequence[Mapping[str, Any]],
    controller_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    provider_rows: Sequence[Mapping[str, Any]],
) -> str:
    billed_gpu_seconds = sum(
        float(row["whole_instance_billed_gpu_seconds"]) for row in provider_rows
    )
    billing_status = (
        "NO_SAMPLES"
        if not provider_rows
        else (
            "COMPLETE"
            if all(bool(row["complete"]) for row in provider_rows)
            else "OPEN_INTERVAL"
        )
    )
    lines = [
        "# Formal experiment progress",
        "",
        f"Run: `{run_id}`  ",
        f"Exported: `{_iso_utc(exported_at_ns)}`  ",
        (
            "Whole-instance billed GPU-hours: "
            f"`{billed_gpu_seconds / 3600.0:.6f}` (`{billing_status}`)"
        ),
        "",
        "| Node | Stage / phase | Complete | Running | Failed | Blocked | Stale | Progress |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage_rows:
        progress = row["progress_percent"]
        progress_text = "N/A" if progress is None else f"{progress:.2f}%"
        lines.append(
            "| {node} | {stage} / {phase} | {completed} | {running} | {failed} | "
            "{blocked} | {stale} | {progress} |".format(**row, progress=progress_text)
        )
    lines.extend(
        [
            "",
            "## DAG controller",
            "",
            "| Node | State | Expected cells | Blocker |",
            "|---|---|---:|---|",
        ]
    )
    for row in controller_rows:
        blocker = (
            str(row["blocker_reason"] or "").replace("\n", " ").replace("|", "\\|")
        )
        expected = (
            "N/A"
            if row["expected_cell_count"] is None
            else str(row["expected_cell_count"])
        )
        lines.append(f"| {row['node']} | {row['state']} | {expected} | {blocker} |")
    critical = [row for row in event_rows if row["severity"] in {"ERROR", "CRITICAL"}][
        -10:
    ]
    lines.extend(["", "## Recent watchdog errors", ""])
    if not critical:
        lines.append("None.")
    else:
        for row in critical:
            identity = (
                "global"
                if row["cell_id"] is None
                else f"{row['cell_id']}/{row['attempt']}"
            )
            lines.append(f"- `{row['event_type']}` on `{identity}`")
    lines.append("")
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    encoded = text.encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_metrics_parquet(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("stage", pa.string()),
            ("phase", pa.string()),
            ("cell_id", pa.string()),
            ("attempt", pa.int64()),
            ("metric_name", pa.string()),
            ("metric_kind", pa.string()),
            ("point_estimate", pa.float64()),
            ("ci_low", pa.float64()),
            ("ci_high", pa.float64()),
            ("independent_block_count", pa.int64()),
            ("request_count", pa.int64()),
            ("paired", pa.bool_()),
            ("reducer_method", pa.string()),
            ("attributes_json", pa.string()),
            ("recorded_at_ns", pa.int64()),
        ]
    )
    table = pa.Table.from_pylist(list(rows), schema=schema)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CELL_ATTEMPT_STATUSES",
    "REMOTE_SPOOL_SAFETY_RESERVE_BYTES",
    "TERMINAL_ATTEMPT_STATUSES",
    "ArchiveCallbacks",
    "ArchiveRequest",
    "ArchiveStepReceipt",
    "AttemptTransitionError",
    "AuxiliaryCellAdoption",
    "AuxiliaryGroupTerminal",
    "AuxiliaryJobSpec",
    "AuxiliaryPhysicalGroupSpec",
    "CellAttemptSpec",
    "ControllerArtifactBinding",
    "DiskDispatchDecision",
    "ExperimentOperatorError",
    "ExperimentOperatorStore",
    "ExportManifest",
    "LegacyStaleAttempt",
    "MetricRecord",
    "OperatorAlreadyRunningError",
    "PhysicalAttemptGroupMemberSpec",
    "ProcessObservation",
    "ProviderRuntimeSample",
    "RemoteEvictionAuthorization",
    "SingletonOperatorLock",
    "SpawnedProcess",
    "StagePlanEntry",
    "WatchdogFinding",
    "WatchdogPolicy",
    "default_formal_stage_plan",
    "evaluate_dispatch_disk_gate",
    "inspect_local_process",
]
