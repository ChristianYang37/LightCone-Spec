"""Production control-plane bridge for one resident TP1 serving group.

This module connects the path-bound resident group executor to the production
operator without changing either scientific cell identity or the operator
ledger state machine.  It is deliberately a trusted-single-operator empirical
bridge: no signature, verifier token, or ``MEASURED`` claim is created here.

The publication boundary is intentionally strict.  A resident shared close is
deeply reopened first, then every member's actual result is dispatched through
the registered cell validator, then the ordinary per-cell control files are
published.  One shared publication is written last.  Until that final file
exists, no terminal revalidator in this module returns a terminal result.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.orchestration.experiment_operator import (
    ProcessObservation,
    QueuedCommandSpec,
    TerminalEvidence,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    child_heartbeat_path,
    child_start_receipt_path,
    file_sha256,
    publish_atomic_terminal_result,
    revalidate_child_start_receipt,
)
from lightcone_spec.orchestration.formal_cell_worker import (
    FormalCellWorkerSpec,
    load_formal_cell_worker_spec,
)
from lightcone_spec.orchestration.formal_serving_session_group_physical import (
    FormalServingResidentActiveProcessTarget,
    FormalServingResidentPhysicalRuntime,
    FormalServingResidentSharedCloseReceipt,
    revalidate_formal_serving_resident_shared_close_receipt,
    revalidate_formal_serving_resident_shared_launch_receipt,
)
from lightcone_spec.orchestration.formal_serving_session_group_sglang import (
    PinnedSglangResidentProcessFactory,
)
from lightcone_spec.orchestration.formal_serving_session_group_worker import (
    FormalServingSessionGroupCellArtifact,
    FormalServingSessionGroupExecutionResult,
    RevalidatedFormalServingSessionGroupExecution,
    execute_formal_serving_session_group,
    revalidate_formal_serving_session_group_execution,
)
from lightcone_spec.orchestration.live_sglang import PinnedNvidiaSmiTool
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV = (
    "LIGHTCONE_FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_PATH"
)
FORMAL_SERVING_SESSION_GROUP_PRODUCTION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_serving_session_group_production",
        "input": (
            "path_bound_group_execution_original_cell_worker_specs_and_final_"
            "queued_command_identities"
        ),
        "member_bounds": "inclusive_2_to_32",
        "disk_high_water": (
            "sum_source_member_bounds_plus_1073741824_shared_base_plus_"
            "67108864_per_member"
        ),
        "server_watch": (
            "durable_wrapper_start_receipt_and_linux_server_start_identity"
        ),
        "heartbeat": "child_owned_phase_trace_sealed_is_not_a_ledger_status",
        "actual": "registered_per_task_actual_validator_deep_reopened",
        "publication": (
            "shared_close_then_all_cell_controls_then_one_shared_publish_last"
        ),
        "terminal": "path_only_shared_publication_fanout_one_atomic_sha",
        "claim": "trusted_single_operator_empirical_no_signature",
        "formal_measured": False,
    }
)

_SPEC_KIND = "formal_serving_session_group_production_spec"
_PUBLICATION_KIND = "formal_serving_session_group_production_publication"
_CONTROL_KIND = "formal_serving_session_group_control_evidence"
_WATCH_TARGET_KIND = "formal_serving_session_group_server_watch_target"
_EVIDENCE_LEVEL = "trusted_single_operator_empirical_no_signature"
_MAX_MEMBER_COUNT = 32
_SHARED_EVIDENCE_BASE_BYTES = 1 * 1024**3
_SHARED_EVIDENCE_PER_MEMBER_BYTES = 64 * 1024**2


class FormalServingSessionGroupProductionError(RuntimeError):
    """Raised when the production bridge cannot prove an exact boundary."""


def formal_serving_session_group_shared_evidence_bound_bytes(
    member_count: int,
) -> int:
    """Return the protocol-bound resident-only spool allowance.

    Per-cell raw/result bounds remain owned by their original queued commands.
    This allowance covers the one shared process log/telemetry/receipt chain and
    therefore must be added to, never substituted for, the member-bound sum.
    """

    if type(member_count) is not int or not 2 <= member_count <= _MAX_MEMBER_COUNT:
        raise ValueError("resident shared evidence bound requires 2-32 members")
    return (
        _SHARED_EVIDENCE_BASE_BYTES + member_count * _SHARED_EVIDENCE_PER_MEMBER_BYTES
    )


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _text(label: str, value: object) -> str:
    if type(value) is not str or not value or any(c in value for c in "\n\r\x00"):
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


def _absolute_path(label: str, value: object) -> Path:
    if type(value) is not str:
        raise TypeError(f"{label} must be a path string")
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or path == Path(path.anchor)
    ):
        raise ValueError(f"{label} must be an absolute normalized non-root path")
    return path


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


def _read_object(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise FormalServingSessionGroupProductionError(
            f"cannot read production evidence: {source}"
        ) from error
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalServingSessionGroupProductionError(
            f"production evidence is not JSON: {source}"
        ) from error
    if type(value) is not dict or _canonical_bytes(value) != raw:
        raise FormalServingSessionGroupProductionError(
            f"production evidence is not one canonical object: {source}"
        )
    return value


def _atomic_write_new_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"production output already exists: {path}")
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"production output already exists: {path}") from None
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_new_json(path: Path, value: object) -> None:
    _atomic_write_new_bytes(path, _canonical_bytes(value))


def _atomic_replace_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".heartbeat", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as stream:
            stream.write(_canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _reopen_binding(
    value: CanonicalJsonProofBinding, *, label: str
) -> CanonicalJsonProofBinding:
    if type(value) is not CanonicalJsonProofBinding:
        raise TypeError(f"{label} must be a canonical JSON binding")
    rebound = CanonicalJsonProofBinding.bind(value.absolute_path)
    if rebound != value:
        raise ValueError(f"{label} changed after publication")
    return rebound


@dataclass(frozen=True)
class FormalServingSessionGroupProductionMemberSpec:
    """One original logical command and cell-worker contract in group order."""

    cell_id: str
    attempt: int
    command_sha256: str
    cell_worker_spec_path: str
    cell_worker_spec_sha256: str
    expected_terminal_path: str
    expected_junit_path: str
    expected_raw_log_path: str
    atomic_pointer_path: str

    def __post_init__(self) -> None:
        _text("production member cell", self.cell_id)
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("production member attempt must be positive")
        _sha256("production member command", self.command_sha256)
        _sha256("production member cell-worker spec", self.cell_worker_spec_sha256)
        paths = tuple(
            str(_absolute_path(f"production member path {index}", value))
            for index, value in enumerate(
                (
                    self.cell_worker_spec_path,
                    self.expected_terminal_path,
                    self.expected_junit_path,
                    self.expected_raw_log_path,
                    self.atomic_pointer_path,
                )
            )
        )
        if len(paths) != len(set(paths)):
            raise ValueError("production member control paths alias")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("production member fields differ")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalServingSessionGroupProductionSpec:
    """Immutable path-only worker instruction for one physical group."""

    schema_version: Literal[1]
    kind: Literal["formal_serving_session_group_production_spec"]
    protocol_sha256: str
    production_spec_path: str
    group_execution_spec_path: str
    repository_root: str
    resident_evidence_root: str
    shared_close_path: str
    shared_publication_path: str
    server_watch_target_path: str
    wrapper_start_receipt_path: str
    heartbeat_path: str
    shared_evidence_bound_bytes: int
    nvidia_smi_tool: PinnedNvidiaSmiTool
    members: tuple[FormalServingSessionGroupProductionMemberSpec, ...]
    evidence_level: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _SPEC_KIND
            or self.protocol_sha256
            != FORMAL_SERVING_SESSION_GROUP_PRODUCTION_PROTOCOL_SHA256
            or self.evidence_level != _EVIDENCE_LEVEL
            or self.formal_measured is not False
        ):
            raise ValueError("production group spec identity differs")
        paths = tuple(
            str(_absolute_path(f"production spec path {index}", value))
            for index, value in enumerate(
                (
                    self.production_spec_path,
                    self.group_execution_spec_path,
                    self.repository_root,
                    self.resident_evidence_root,
                    self.shared_close_path,
                    self.shared_publication_path,
                    self.server_watch_target_path,
                    self.wrapper_start_receipt_path,
                    self.heartbeat_path,
                )
            )
        )
        if len(paths) != len(set(paths)):
            raise ValueError("production group top-level paths alias")
        if type(self.nvidia_smi_tool) is not PinnedNvidiaSmiTool:
            raise TypeError("production group requires a pinned nvidia-smi tool")
        if (
            type(self.members) is not tuple
            or not 2 <= len(self.members) <= _MAX_MEMBER_COUNT
            or any(
                type(row) is not FormalServingSessionGroupProductionMemberSpec
                for row in self.members
            )
            or len({(row.cell_id, row.attempt) for row in self.members})
            != len(self.members)
        ):
            raise ValueError("production group requires 2-32 unique members")
        if self.shared_evidence_bound_bytes != (
            formal_serving_session_group_shared_evidence_bound_bytes(len(self.members))
        ):
            raise ValueError("production group shared evidence bound differs")
        control_paths = tuple(
            path
            for member in self.members
            for path in (
                member.expected_terminal_path,
                member.expected_junit_path,
                member.expected_raw_log_path,
                member.atomic_pointer_path,
            )
        )
        if len(control_paths) != len(set(control_paths)) or set(control_paths) & set(
            paths
        ):
            raise ValueError("production group member outputs alias")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["nvidia_smi_tool"] = self.nvidia_smi_tool.to_dict()
        value["members"] = [row.to_dict() for row in self.members]
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("production group spec fields differ")
        row = dict(value)
        raw_tool = row.pop("nvidia_smi_tool")
        raw_members = row.pop("members")
        if type(raw_members) is not list:
            raise TypeError("production group members must be an array")
        return cls(
            **row,
            nvidia_smi_tool=PinnedNvidiaSmiTool.from_dict(raw_tool),
            members=tuple(
                FormalServingSessionGroupProductionMemberSpec.from_dict(item)
                for item in raw_members
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class RevalidatedFormalServingSessionGroupProductionSpec:
    binding: CanonicalJsonProofBinding
    spec: FormalServingSessionGroupProductionSpec
    execution: RevalidatedFormalServingSessionGroupExecution
    cell_worker_specs: tuple[FormalCellWorkerSpec, ...]


def formal_serving_session_group_production_environment(
    production_spec_path: str | Path,
) -> tuple[tuple[str, str], ...]:
    """Return the non-cyclic worker env: path only, never the spec digest."""

    path = _absolute_path("production worker spec", str(production_spec_path))
    return ((FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV, str(path)),)


def build_formal_serving_session_group_production_spec(
    *,
    production_spec_path: str | Path,
    group_execution_spec_path: str | Path,
    cell_worker_spec_paths: Sequence[str | Path],
    commands: Sequence[QueuedCommandSpec],
    nvidia_smi_tool: PinnedNvidiaSmiTool,
    resident_evidence_root: str | Path,
    shared_publication_path: str | Path,
    server_watch_target_path: str | Path,
) -> FormalServingSessionGroupProductionSpec:
    """Bind the final path-env commands to the original worker specifications."""

    production_path = _absolute_path(
        "production spec output", str(production_spec_path)
    )
    execution = revalidate_formal_serving_session_group_execution(
        group_execution_spec_path
    )
    if not 2 <= len(execution.plan.members) <= _MAX_MEMBER_COUNT:
        raise ValueError("production serving group is outside the 2-32 bound")
    if len(cell_worker_spec_paths) != len(commands) or len(commands) != len(
        execution.plan.members
    ):
        raise ValueError("production group input coverage differs")
    expected_environment = dict(
        formal_serving_session_group_production_environment(production_path)
    )
    rows: list[FormalServingSessionGroupProductionMemberSpec] = []
    worker_specs: list[FormalCellWorkerSpec] = []
    for plan_member, worker_path_raw, command in zip(
        execution.plan.members,
        cell_worker_spec_paths,
        commands,
        strict=True,
    ):
        if type(command) is not QueuedCommandSpec:
            raise TypeError("production group commands must be exact queued commands")
        worker_path = _absolute_path("cell-worker spec", str(worker_path_raw))
        worker, worker_sha256 = load_formal_cell_worker_spec(worker_path)
        environment = dict(command.environment)
        if (
            environment.get(FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV)
            != expected_environment[FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV]
            or worker.cell_id != plan_member.materialized_cell_id
            or worker.attempt != plan_member.attempt
            or command.cell_id != plan_member.materialized_cell_id
            or command.attempt != plan_member.attempt
            or worker.cell_id != command.cell_id
            or worker.attempt != command.attempt
        ):
            raise ValueError("production member plan/worker/command identity differs")
        worker_specs.append(worker)
        rows.append(
            FormalServingSessionGroupProductionMemberSpec(
                cell_id=command.cell_id,
                attempt=command.attempt,
                command_sha256=command.command_sha256,
                cell_worker_spec_path=str(worker_path),
                cell_worker_spec_sha256=worker_sha256,
                expected_terminal_path=command.expected_terminal_path,
                expected_junit_path=command.expected_junit_path,
                expected_raw_log_path=command.expected_raw_log_path,
                atomic_pointer_path=command.atomic_pointer_path,
            )
        )
    repositories = {worker.repository_root for worker in worker_specs}
    nodes = {worker.node_materialization_path for worker in worker_specs}
    if len(repositories) != 1 or len(nodes) != 1:
        raise ValueError(
            "production group workers leave one repository/materialization"
        )
    resident_root = _absolute_path(
        "resident production evidence root", str(resident_evidence_root)
    )
    close_path = resident_root / execution.plan.group_id / "shared-close.json"
    first_command = commands[0]
    return FormalServingSessionGroupProductionSpec(
        schema_version=1,
        kind=_SPEC_KIND,
        protocol_sha256=FORMAL_SERVING_SESSION_GROUP_PRODUCTION_PROTOCOL_SHA256,
        production_spec_path=str(production_path),
        group_execution_spec_path=str(
            _absolute_path("group execution spec", str(group_execution_spec_path))
        ),
        repository_root=next(iter(repositories)),
        resident_evidence_root=str(resident_root),
        shared_close_path=str(close_path.resolve(strict=False)),
        shared_publication_path=str(
            _absolute_path(
                "shared production publication", str(shared_publication_path)
            )
        ),
        server_watch_target_path=str(
            _absolute_path("server watch target", str(server_watch_target_path))
        ),
        wrapper_start_receipt_path=str(child_start_receipt_path(first_command)),
        heartbeat_path=str(child_heartbeat_path(first_command)),
        shared_evidence_bound_bytes=(
            formal_serving_session_group_shared_evidence_bound_bytes(len(rows))
        ),
        nvidia_smi_tool=nvidia_smi_tool,
        members=tuple(rows),
        evidence_level=_EVIDENCE_LEVEL,
        formal_measured=False,
    )


def publish_formal_serving_session_group_production_spec(
    *,
    spec: FormalServingSessionGroupProductionSpec,
    output_path: str | Path | None = None,
) -> CanonicalJsonProofBinding:
    if type(spec) is not FormalServingSessionGroupProductionSpec:
        raise TypeError("production spec publisher requires an exact spec")
    path = Path(spec.production_spec_path if output_path is None else output_path)
    if path != Path(spec.production_spec_path):
        raise ValueError("production spec publisher path differs from bound identity")
    publish_canonical_json_no_replace(path, spec.to_dict())
    return revalidate_formal_serving_session_group_production_spec(path).binding


def revalidate_formal_serving_session_group_production_spec(
    path: str | Path,
) -> RevalidatedFormalServingSessionGroupProductionSpec:
    binding = CanonicalJsonProofBinding.bind(path)
    spec = FormalServingSessionGroupProductionSpec.from_dict(binding.reopen())
    if (
        binding.absolute_path != spec.production_spec_path
        or binding.semantic_sha256 != spec.sha256
    ):
        raise ValueError("production spec path or identity differs")
    execution = revalidate_formal_serving_session_group_execution(
        spec.group_execution_spec_path
    )
    expected_close = (
        Path(spec.resident_evidence_root)
        / execution.plan.group_id
        / "shared-close.json"
    ).resolve(strict=False)
    if (
        execution.plan.execution_mode != "shared_session_tp1"
        or not 2 <= len(execution.plan.members) <= _MAX_MEMBER_COUNT
        or str(expected_close) != spec.shared_close_path
        or tuple(
            (row.materialized_cell_id, row.attempt) for row in execution.plan.members
        )
        != tuple((row.cell_id, row.attempt) for row in spec.members)
    ):
        raise ValueError("production spec leaves its shared group plan")
    workers: list[FormalCellWorkerSpec] = []
    for member, worker_member in zip(spec.members, execution.plan.members, strict=True):
        worker, digest = load_formal_cell_worker_spec(member.cell_worker_spec_path)
        if (
            digest != member.cell_worker_spec_sha256
            or worker.cell_id != member.cell_id
            or worker.attempt != member.attempt
            or worker.repository_root != spec.repository_root
            or worker.cell_id != worker_member.materialized_cell_id
            or worker.attempt != worker_member.attempt
        ):
            raise ValueError("production cell-worker binding differs")
        workers.append(worker)
    spec.nvidia_smi_tool.revalidate()
    return RevalidatedFormalServingSessionGroupProductionSpec(
        binding=binding,
        spec=spec,
        execution=execution,
        cell_worker_specs=tuple(workers),
    )


def formal_serving_session_group_production_spec_path_from_command(
    command: QueuedCommandSpec,
) -> Path:
    """Deep-open a member's production spec from its path-only environment."""

    if type(command) is not QueuedCommandSpec:
        raise TypeError("production path lookup requires an exact queued command")
    environment = dict(command.environment)
    raw = environment.get(FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV)
    if raw is None:
        raise ValueError("queued command lacks the production group spec path")
    path = _absolute_path("queued production group spec", raw)
    validated = revalidate_formal_serving_session_group_production_spec(path)
    matches = tuple(
        member
        for member in validated.spec.members
        if (member.cell_id, member.attempt, member.command_sha256)
        == (command.cell_id, command.attempt, command.command_sha256)
    )
    if len(matches) != 1:
        raise ValueError("queued command is not an exact production group member")
    return path


def ensure_formal_serving_session_group_production_outputs_unoccupied(
    spec_or_path: FormalServingSessionGroupProductionSpec | str | Path,
) -> None:
    """Fail before GPU allocation when any publish-once output is occupied."""

    spec = (
        spec_or_path
        if type(spec_or_path) is FormalServingSessionGroupProductionSpec
        else revalidate_formal_serving_session_group_production_spec(spec_or_path).spec
    )
    assert type(spec) is FormalServingSessionGroupProductionSpec
    paths = [
        Path(spec.shared_publication_path),
        Path(spec.server_watch_target_path),
        Path(spec.heartbeat_path),
        Path(spec.shared_close_path),
        *(Path(row.expected_terminal_path) for row in spec.members),
        *(Path(row.expected_junit_path) for row in spec.members),
        *(Path(row.expected_raw_log_path) for row in spec.members),
        *(Path(row.atomic_pointer_path) for row in spec.members),
    ]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"production group output is occupied: {path}")
    execution = revalidate_formal_serving_session_group_execution(
        spec.group_execution_spec_path
    )
    execution_output = Path(execution.spec.output_directory)
    if execution_output.exists() and (
        execution_output.is_symlink()
        or not execution_output.is_dir()
        or any(execution_output.iterdir())
    ):
        raise FileExistsError("production group execution output is occupied")
    group_evidence = Path(spec.resident_evidence_root) / execution.plan.group_id
    if group_evidence.exists() and (
        group_evidence.is_symlink()
        or not group_evidence.is_dir()
        or any(group_evidence.iterdir())
    ):
        raise FileExistsError("production resident group evidence is occupied")
    for member in spec.members:
        worker, _digest = load_formal_cell_worker_spec(member.cell_worker_spec_path)
        actual = Path(worker.actual_result_path)
        if actual.exists() or actual.is_symlink():
            raise FileExistsError(f"production member actual is occupied: {actual}")


class FormalServingSessionGroupChildHeartbeatPublisher:
    """Mutable child heartbeat; phases never mutate scientific ledger status."""

    def __init__(
        self,
        *,
        path: str | Path,
        cell_id: str,
        attempt: int,
        command_sha256: str,
        clock_ns: Callable[[], int] = time.time_ns,
        interval_seconds: float = 30.0,
    ) -> None:
        self.path = _absolute_path("group heartbeat", str(path))
        self.cell_id = _text("group heartbeat cell", cell_id)
        if type(attempt) is not int or attempt < 1:
            raise ValueError("group heartbeat attempt must be positive")
        self.attempt = attempt
        self.command_sha256 = _sha256("group heartbeat command", command_sha256)
        if not isinstance(interval_seconds, (int, float)) or interval_seconds <= 0:
            raise ValueError("group heartbeat interval must be positive")
        self.interval_seconds = float(interval_seconds)
        self.clock_ns = clock_ns
        self.sequence = 0
        self._phase = "STARTING"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self, phase: str = "RUNNING") -> None:
        if self._thread is not None:
            raise RuntimeError("group heartbeat was already started")
        self.set_phase(phase)
        self._thread = threading.Thread(
            target=self._run,
            name=f"formal-serving-group-heartbeat-{self.cell_id}",
            daemon=True,
        )
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        canonical = _text("group heartbeat phase", phase)
        with self._lock:
            self._phase = canonical
            self._publish_locked()

    def stop(self, final_phase: str = "FINALIZING") -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self.interval_seconds * 2))
            if thread.is_alive():
                raise RuntimeError("group heartbeat publisher did not stop")
        if self._error is not None:
            raise FormalServingSessionGroupProductionError(
                "group heartbeat publisher failed"
            ) from self._error
        self.set_phase(final_phase)

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval_seconds):
                with self._lock:
                    self._publish_locked()
        except BaseException as error:  # noqa: BLE001 - surfaced by stop
            self._error = error
            self._stop.set()

    def _publish_locked(self) -> None:
        self.sequence += 1
        observed_at_ns = int(self.clock_ns())
        if observed_at_ns < 1:
            raise ValueError("group heartbeat clock is invalid")
        _atomic_replace_json(
            self.path,
            {
                "schema_version": 1,
                "kind": "formal_experiment_child_heartbeat",
                "cell_id": self.cell_id,
                "attempt": self.attempt,
                "command_sha256": self.command_sha256,
                "worker_pid": os.getpid(),
                "sequence": self.sequence,
                "observed_at_ns": observed_at_ns,
                "phase": self._phase,
            },
        )


def _linux_process_start_identity(pid: int) -> tuple[str, int]:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        boot_id = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        )
    except (OSError, UnicodeDecodeError) as error:
        raise FormalServingSessionGroupProductionError(
            "cannot bind resident server Linux process start identity"
        ) from error
    closing = stat_text.rfind(")")
    fields = stat_text[closing + 2 :].split() if closing >= 0 else []
    if len(fields) <= 19 or not fields[19].isdigit() or not boot_id:
        raise FormalServingSessionGroupProductionError(
            "resident server Linux process start identity is malformed"
        )
    return boot_id, int(fields[19])


@dataclass(frozen=True)
class FormalServingSessionGroupServerWatchTarget:
    schema_version: Literal[1]
    kind: Literal["formal_serving_session_group_server_watch_target"]
    protocol_sha256: str
    production_spec: CanonicalJsonProofBinding
    group_plan: CanonicalJsonProofBinding
    shared_launch: CanonicalJsonProofBinding
    wrapper_start_receipt: CanonicalJsonProofBinding
    wrapper_process_id: int
    wrapper_process_group_id: int
    wrapper_started_ns: int
    server_process_id: int
    server_process_group_id: int
    server_process_started_ns: int
    server_boot_id: str
    server_start_time_ticks: int
    published_ns: int
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _WATCH_TARGET_KIND
            or self.protocol_sha256
            != FORMAL_SERVING_SESSION_GROUP_PRODUCTION_PROTOCOL_SHA256
            or self.formal_measured is not False
        ):
            raise ValueError("server watch target identity differs")
        for label, binding in (
            ("production spec", self.production_spec),
            ("group plan", self.group_plan),
            ("shared launch", self.shared_launch),
            ("wrapper start", self.wrapper_start_receipt),
        ):
            _reopen_binding(binding, label=f"watch target {label}")
        for value in (
            self.wrapper_process_id,
            self.wrapper_process_group_id,
            self.wrapper_started_ns,
            self.server_process_id,
            self.server_process_group_id,
            self.server_process_started_ns,
            self.server_start_time_ticks,
            self.published_ns,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("server watch target lifecycle differs")
        _text("server watch target boot ID", self.server_boot_id)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "production_spec",
            "group_plan",
            "shared_launch",
            "wrapper_start_receipt",
        ):
            value[name] = getattr(self, name).to_dict()
        if include_sha256:
            value["target_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "target_sha256",
        }:
            raise ValueError("server watch target fields differ")
        row = dict(value)
        declared = _sha256("server watch target", row.pop("target_sha256"))
        for name in (
            "production_spec",
            "group_plan",
            "shared_launch",
            "wrapper_start_receipt",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        result = cls(**row)  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("server watch target digest differs")
        return result


def publish_formal_serving_session_group_server_watch_target(
    *,
    production_spec_path: str | Path,
    shared_launch_path: str | Path,
    output_path: str | Path | None = None,
    published_ns: int | None = None,
) -> CanonicalJsonProofBinding:
    validated = revalidate_formal_serving_session_group_production_spec(
        production_spec_path
    )
    launch_binding, launch = revalidate_formal_serving_resident_shared_launch_receipt(
        shared_launch_path
    )
    member = validated.spec.members[0]
    wrapper = revalidate_child_start_receipt(
        validated.spec.wrapper_start_receipt_path,
        command_sha256=member.command_sha256,
    )
    wrapper_binding = CanonicalJsonProofBinding.bind(
        validated.spec.wrapper_start_receipt_path
    )
    boot_id, start_ticks = _linux_process_start_identity(launch.server_process_id)
    target = FormalServingSessionGroupServerWatchTarget(
        schema_version=1,
        kind=_WATCH_TARGET_KIND,
        protocol_sha256=FORMAL_SERVING_SESSION_GROUP_PRODUCTION_PROTOCOL_SHA256,
        production_spec=validated.binding,
        group_plan=validated.execution.plan_binding,
        shared_launch=launch_binding,
        wrapper_start_receipt=wrapper_binding,
        wrapper_process_id=wrapper.pid,
        wrapper_process_group_id=wrapper.pgid,
        wrapper_started_ns=wrapper.started_ns,
        server_process_id=launch.server_process_id,
        server_process_group_id=launch.server_process_group_id,
        server_process_started_ns=launch.server_process_started_ns,
        server_boot_id=boot_id,
        server_start_time_ticks=start_ticks,
        published_ns=time.time_ns() if published_ns is None else published_ns,
        formal_measured=False,
    )
    path = Path(
        validated.spec.server_watch_target_path if output_path is None else output_path
    )
    if path != Path(validated.spec.server_watch_target_path):
        raise ValueError("server watch target output path differs")
    publish_canonical_json_no_replace(path, target.to_dict())
    return revalidate_formal_serving_session_group_server_watch_target(path)[0]


def revalidate_formal_serving_session_group_server_watch_target(
    path: str | Path,
) -> tuple[
    CanonicalJsonProofBinding,
    FormalServingSessionGroupServerWatchTarget,
]:
    binding = CanonicalJsonProofBinding.bind(path)
    target = FormalServingSessionGroupServerWatchTarget.from_dict(binding.reopen())
    validated = revalidate_formal_serving_session_group_production_spec(
        target.production_spec.absolute_path
    )
    launch_binding, launch = revalidate_formal_serving_resident_shared_launch_receipt(
        target.shared_launch.absolute_path
    )
    wrapper = revalidate_child_start_receipt(
        target.wrapper_start_receipt.absolute_path,
        command_sha256=validated.spec.members[0].command_sha256,
    )
    if (
        binding.absolute_path != validated.spec.server_watch_target_path
        or target.production_spec != validated.binding
        or target.group_plan != validated.execution.plan_binding
        or target.shared_launch != launch_binding
        or (
            target.wrapper_process_id,
            target.wrapper_process_group_id,
            target.wrapper_started_ns,
        )
        != (wrapper.pid, wrapper.pgid, wrapper.started_ns)
        or (
            target.server_process_id,
            target.server_process_group_id,
            target.server_process_started_ns,
        )
        != (
            launch.server_process_id,
            launch.server_process_group_id,
            launch.server_process_started_ns,
        )
    ):
        raise ValueError("server watch target leaves wrapper/server lineage")
    process_is_live = Path(f"/proc/{target.server_process_id}/stat").exists()
    if process_is_live and _linux_process_start_identity(target.server_process_id) != (
        target.server_boot_id,
        target.server_start_time_ticks,
    ):
        raise ValueError("server watch target PID was reused")
    if process_is_live:
        try:
            observed_pgid = os.getpgid(target.server_process_id)
        except ProcessLookupError:
            observed_pgid = None
        if (
            observed_pgid is not None
            and observed_pgid != target.server_process_group_id
        ):
            raise ValueError("live server left its registered process group")
    return binding, target


def formal_serving_session_group_active_target_publisher(
    production_spec_path: str | Path,
) -> Callable[[FormalServingResidentActiveProcessTarget], None]:
    """Build the synchronous launch callback used before the first trace."""

    validated = revalidate_formal_serving_session_group_production_spec(
        production_spec_path
    )

    def publish(target: FormalServingResidentActiveProcessTarget) -> None:
        if type(target) is not FormalServingResidentActiveProcessTarget:
            raise TypeError("resident active target callback type differs")
        binding = publish_formal_serving_session_group_server_watch_target(
            production_spec_path=validated.binding.absolute_path,
            shared_launch_path=target.shared_launch.absolute_path,
        )
        _watch_binding, watch = (
            revalidate_formal_serving_session_group_server_watch_target(
                binding.absolute_path
            )
        )
        if (
            watch.server_process_id,
            watch.server_process_group_id,
            watch.server_process_started_ns,
            watch.shared_launch,
        ) != (
            target.process_id,
            target.process_group_id,
            target.process_started_ns,
            target.shared_launch,
        ):
            raise ValueError("durable watch target differs from active server")

    return publish


@dataclass(frozen=True)
class FormalServingSessionGroupControlEvidence:
    cell_id: str
    attempt: int
    command_sha256: str
    status: Literal["COMPLETE", "FAILED"]
    cell_artifact: CanonicalJsonProofBinding
    actual_result: CanonicalJsonProofBinding | None
    result_identity_sha256: str | None
    validator_kind: str | None
    validator_protocol_sha256: str | None
    terminal: CanonicalJsonProofBinding
    junit: EvidenceFileBinding
    raw_log: CanonicalJsonProofBinding
    atomic_pointer: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        _text("group control cell", self.cell_id)
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("group control attempt differs")
        _sha256("group control command", self.command_sha256)
        if self.status not in {"COMPLETE", "FAILED"}:
            raise ValueError("group control status differs")
        for label, binding in (
            ("cell artifact", self.cell_artifact),
            ("terminal", self.terminal),
            ("raw log", self.raw_log),
            ("atomic pointer", self.atomic_pointer),
        ):
            _reopen_binding(binding, label=f"group control {label}")
        if type(self.junit) is not EvidenceFileBinding:
            raise TypeError("group control JUnit binding differs")
        self.junit.reopen(label="group control JUnit")
        values = (
            self.result_identity_sha256,
            self.validator_kind,
            self.validator_protocol_sha256,
        )
        if self.status == "COMPLETE":
            if type(self.actual_result) is not CanonicalJsonProofBinding or any(
                value is None for value in values
            ):
                raise ValueError("complete group control lacks actual validation")
            _reopen_binding(self.actual_result, label="group control actual")
            _sha256("group control result identity", self.result_identity_sha256)
            _text("group control validator", self.validator_kind)
            _sha256("group control validator protocol", self.validator_protocol_sha256)
        elif self.actual_result is not None or any(
            value is not None for value in values
        ):
            raise ValueError("failed group control claims actual validation")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for name in ("cell_artifact", "terminal", "raw_log", "atomic_pointer"):
            value[name] = getattr(self, name).to_dict()
        value["actual_result"] = (
            None if self.actual_result is None else self.actual_result.to_dict()
        )
        value["junit"] = self.junit.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("group control fields differ")
        row = dict(value)
        for name in ("cell_artifact", "terminal", "raw_log", "atomic_pointer"):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        if row["actual_result"] is not None:
            row["actual_result"] = CanonicalJsonProofBinding.from_dict(
                row["actual_result"]
            )
        row["junit"] = EvidenceFileBinding.from_dict(
            row["junit"], label="group control JUnit"
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalServingSessionGroupProductionPublication:
    schema_version: Literal[1]
    kind: Literal["formal_serving_session_group_production_publication"]
    protocol_sha256: str
    production_spec: CanonicalJsonProofBinding
    group_execution_result: CanonicalJsonProofBinding
    shared_close: CanonicalJsonProofBinding
    server_watch_target: CanonicalJsonProofBinding
    controls: tuple[FormalServingSessionGroupControlEvidence, ...]
    worker_started_ns: int
    worker_finished_ns: int
    commit_marker: Literal["SHARED_CLOSE_AND_ALL_MEMBER_CONTROLS_PUBLISHED"]
    evidence_level: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _PUBLICATION_KIND
            or self.protocol_sha256
            != FORMAL_SERVING_SESSION_GROUP_PRODUCTION_PROTOCOL_SHA256
            or self.commit_marker != "SHARED_CLOSE_AND_ALL_MEMBER_CONTROLS_PUBLISHED"
            or self.evidence_level != _EVIDENCE_LEVEL
            or self.formal_measured is not False
        ):
            raise ValueError("production shared publication identity differs")
        for label, binding in (
            ("production spec", self.production_spec),
            ("execution result", self.group_execution_result),
            ("shared close", self.shared_close),
            ("server watch target", self.server_watch_target),
        ):
            _reopen_binding(binding, label=f"production publication {label}")
        if (
            type(self.controls) is not tuple
            or not 2 <= len(self.controls) <= _MAX_MEMBER_COUNT
            or len({(row.cell_id, row.attempt) for row in self.controls})
            != len(self.controls)
            or type(self.worker_started_ns) is not int
            or type(self.worker_finished_ns) is not int
            or self.worker_started_ns < 1
            or self.worker_finished_ns <= self.worker_started_ns
        ):
            raise ValueError("production shared publication coverage/lifecycle differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "production_spec",
            "group_execution_result",
            "shared_close",
            "server_watch_target",
        ):
            value[name] = getattr(self, name).to_dict()
        value["controls"] = [row.to_dict() for row in self.controls]
        if include_sha256:
            value["publication_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "publication_sha256",
        }:
            raise ValueError("production shared publication fields differ")
        row = dict(value)
        declared = _sha256(
            "production shared publication", row.pop("publication_sha256")
        )
        for name in (
            "production_spec",
            "group_execution_result",
            "shared_close",
            "server_watch_target",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        raw_controls = row.pop("controls")
        if type(raw_controls) is not list:
            raise TypeError("production controls must be an array")
        result = cls(
            **row,
            controls=tuple(
                FormalServingSessionGroupControlEvidence.from_dict(item)
                for item in raw_controls
            ),
        )  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("production shared publication digest differs")
        return result


@dataclass(frozen=True)
class RevalidatedFormalServingSessionGroupProductionPublication:
    binding: CanonicalJsonProofBinding
    publication: FormalServingSessionGroupProductionPublication
    spec: RevalidatedFormalServingSessionGroupProductionSpec
    result: FormalServingSessionGroupExecutionResult
    artifacts: tuple[FormalServingSessionGroupCellArtifact, ...]
    terminals: Mapping[str, TerminalEvidence]


def _execution_result_path(
    validated: RevalidatedFormalServingSessionGroupProductionSpec,
) -> Path:
    return Path(validated.execution.spec.output_directory) / "result.json"


def _revalidate_group_result_and_close(
    validated: RevalidatedFormalServingSessionGroupProductionSpec,
    *,
    close_revalidator: Callable[
        [str | Path],
        tuple[CanonicalJsonProofBinding, FormalServingResidentSharedCloseReceipt],
    ] = revalidate_formal_serving_resident_shared_close_receipt,
) -> tuple[
    CanonicalJsonProofBinding,
    FormalServingSessionGroupExecutionResult,
    tuple[FormalServingSessionGroupCellArtifact, ...],
    CanonicalJsonProofBinding,
    FormalServingResidentSharedCloseReceipt,
]:
    result_binding = CanonicalJsonProofBinding.bind(_execution_result_path(validated))
    result = FormalServingSessionGroupExecutionResult.from_dict(result_binding.reopen())
    if (
        result.execution_spec != validated.execution.spec_binding
        or result.group_plan != validated.execution.plan_binding
        or result.reset_authority != validated.execution.authority_binding
        or result.group_id != validated.execution.plan.group_id
        or len(result.cell_artifacts) != len(validated.spec.members)
    ):
        raise ValueError("production group execution result lineage differs")
    artifacts: list[FormalServingSessionGroupCellArtifact] = []
    for index, (binding, plan_member) in enumerate(
        zip(result.cell_artifacts, validated.execution.plan.members, strict=True)
    ):
        _reopen_binding(binding, label="production cell artifact")
        artifact = FormalServingSessionGroupCellArtifact.from_dict(binding.reopen())
        if (
            artifact.group_id != result.group_id
            or artifact.materialized_cell_id != plan_member.materialized_cell_id
            or artifact.attempt != plan_member.attempt
            or artifact.member_index != index
        ):
            raise ValueError("production cell artifact order/identity differs")
        artifacts.append(artifact)
    if (
        sum(row.execution_mode == "shared_session_tp1" for row in artifacts)
        != result.shared_completed
        or sum(row.execution_mode == "fresh_process_fallback" for row in artifacts)
        != result.fresh_fallback_completed
        or sum(row.status == "FAILED" for row in artifacts) != result.failed
    ):
        raise ValueError("production group execution result counts differ")
    close_binding, close = close_revalidator(validated.spec.shared_close_path)
    if (
        close_binding.absolute_path != validated.spec.shared_close_path
        or close.group_plan != validated.execution.plan_binding
        or close.process_group_empty is not True
        or len(close.member_trace_receipts) != result.shared_completed
    ):
        raise ValueError("production shared close leaves execution result")
    return result_binding, result, tuple(artifacts), close_binding, close


def _write_junit(
    path: Path,
    *,
    cell_id: str,
    elapsed_seconds: float,
    failure: str | None,
) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "formal_serving_session_group_production",
            "tests": "1",
            "failures": "0" if failure is None else "1",
            "errors": "0",
            "skipped": "0",
            "time": f"{max(0.0, elapsed_seconds):.9f}",
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {
            "classname": "lightcone_spec.formal_serving_session_group",
            "name": cell_id,
            "time": f"{max(0.0, elapsed_seconds):.9f}",
        },
    )
    if failure is not None:
        node = ET.SubElement(case, "failure", {"message": failure})
        node.text = failure
    _atomic_write_new_bytes(path, ET.tostring(suite, encoding="utf-8") + b"\n")


def _validate_junit(path: Path, *, require_clean: bool) -> None:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError("production group JUnit is invalid") from error
    if (
        root.tag != "testsuite"
        or root.attrib.get("tests") != "1"
        or root.attrib.get("errors") != "0"
        or root.attrib.get("skipped") != "0"
        or root.attrib.get("failures") != ("0" if require_clean else "1")
        or len(root.findall("testcase")) != 1
    ):
        raise ValueError("production group JUnit counters differ")


def _member_command(
    member: FormalServingSessionGroupProductionMemberSpec,
) -> QueuedCommandSpec:
    """Minimal exact command view used only by the standard pointer publisher."""

    # ``publish_atomic_terminal_result`` needs only these identity/control fields,
    # but insists on the exact class.  The remaining values are fixed, inert
    # control-plane placeholders and never launch a process.
    return QueuedCommandSpec(
        cell_id=member.cell_id,
        attempt=member.attempt,
        argv=("/bin/false",),
        launch_compatibility_key="formal-serving-session-group-control-only",
        required_gpu_count=1,
        timing_class="SAFE_AUXILIARY",
        predicted_high_water_bytes=0,
        monitored_path=member.expected_raw_log_path,
        log_path=str(
            Path(member.expected_raw_log_path).with_name(
                f".{Path(member.expected_raw_log_path).name}.control-wrapper.log"
            )
        ),
        expected_terminal_path=member.expected_terminal_path,
        expected_junit_path=member.expected_junit_path,
        expected_raw_log_path=member.expected_raw_log_path,
        atomic_pointer_path=member.atomic_pointer_path,
        child_exit_receipt_path=str(
            Path(member.atomic_pointer_path).with_name(
                f".{Path(member.atomic_pointer_path).name}.control-exit.json"
            )
        ),
    )


def _publish_member_control(
    *,
    production: RevalidatedFormalServingSessionGroupProductionSpec,
    member_index: int,
    artifact_binding: CanonicalJsonProofBinding,
    artifact: FormalServingSessionGroupCellArtifact,
    actual_validator: Callable[..., Any],
    clock_ns: Callable[[], int],
    fallback_started_ns: int,
) -> FormalServingSessionGroupControlEvidence:
    member = production.spec.members[member_index]
    worker = production.cell_worker_specs[member_index]
    status = artifact.status
    validation: Any | None = None
    actual_binding: CanonicalJsonProofBinding | None = None
    if status == "COMPLETE":
        if artifact.result_pointer is None:
            raise ValueError("complete group member lacks an actual result")
        actual_binding = _reopen_binding(
            artifact.result_pointer, label="production member actual"
        )
        if actual_binding.absolute_path != worker.actual_result_path:
            raise ValueError("group member actual path differs from worker contract")
        validation = actual_validator(
            node_materialization_path=worker.node_materialization_path,
            cell_id=member.cell_id,
            actual_result_path=actual_binding.absolute_path,
            repository_root=worker.repository_root,
        )
        if getattr(validation, "status", None) != "COMPLETE":
            raise ValueError("production actual validator returned non-COMPLETE")
    started_ns = artifact.started_ns
    finished_ns = artifact.finished_ns
    if (
        type(started_ns) is not int
        or type(finished_ns) is not int
        or started_ns < 1
        or finished_ns <= started_ns
    ):
        started_ns = fallback_started_ns
        finished_ns = max(int(clock_ns()), started_ns + 1)
    failure = (
        None
        if status == "COMPLETE"
        else (artifact.failure_code or "FORMAL_SERVING_SESSION_GROUP_MEMBER_FAILED")
    )
    _atomic_write_new_json(
        Path(member.expected_raw_log_path),
        {
            "schema_version": 1,
            "kind": _CONTROL_KIND,
            "protocol_sha256": FORMAL_SERVING_SESSION_GROUP_PRODUCTION_PROTOCOL_SHA256,
            "cell_id": member.cell_id,
            "attempt": member.attempt,
            "command_sha256": member.command_sha256,
            "status": status,
            "cell_artifact": artifact_binding.to_dict(),
            "actual_result": (
                None if actual_binding is None else actual_binding.to_dict()
            ),
            "result_identity_sha256": (
                None if validation is None else validation.result_identity_sha256
            ),
            "validator_kind": (
                None if validation is None else validation.validator_kind
            ),
            "validator_protocol_sha256": (
                None if validation is None else validation.validator_protocol_sha256
            ),
            "formal_measured": False,
        },
    )
    _write_junit(
        Path(member.expected_junit_path),
        cell_id=member.cell_id,
        elapsed_seconds=(finished_ns - started_ns) / 1e9,
        failure=failure,
    )
    # Use the registered command identity, not the inert command projection's
    # computed digest.  OperatorTerminalContext avoids any digest substitution.
    from lightcone_spec.orchestration.experiment_operator_production import (
        OperatorTerminalContext,
    )

    context = OperatorTerminalContext(
        cell_id=member.cell_id,
        attempt=member.attempt,
        command_sha256=member.command_sha256,
        expected_terminal_path=member.expected_terminal_path,
        expected_junit_path=member.expected_junit_path,
        expected_raw_log_path=member.expected_raw_log_path,
        atomic_pointer_path=member.atomic_pointer_path,
    )
    complete_exclusion = worker.complete_exclusion_reason
    publish_atomic_terminal_result(
        context,
        status=status,
        exit_code=0 if status == "COMPLETE" else (artifact.exit_code or 70),
        started_ns=started_ns,
        finished_ns=finished_ns,
        failure_class=(None if status == "COMPLETE" else artifact.failure_class),
        failure_code=None if status == "COMPLETE" else failure,
        exclusion_reason=(
            complete_exclusion
            if status == "COMPLETE"
            else "formal_serving_session_group_member_failed"
        ),
        included_in_analysis=(
            worker.included_in_analysis_on_complete if status == "COMPLETE" else False
        ),
    )
    return FormalServingSessionGroupControlEvidence(
        cell_id=member.cell_id,
        attempt=member.attempt,
        command_sha256=member.command_sha256,
        status=status,
        cell_artifact=artifact_binding,
        actual_result=actual_binding,
        result_identity_sha256=(
            None if validation is None else validation.result_identity_sha256
        ),
        validator_kind=None if validation is None else validation.validator_kind,
        validator_protocol_sha256=(
            None if validation is None else validation.validator_protocol_sha256
        ),
        terminal=CanonicalJsonProofBinding.bind(member.expected_terminal_path),
        junit=EvidenceFileBinding.bind(
            member.expected_junit_path, label="production member JUnit"
        ),
        raw_log=CanonicalJsonProofBinding.bind(member.expected_raw_log_path),
        atomic_pointer=CanonicalJsonProofBinding.bind(member.atomic_pointer_path),
    )


async def _default_server_watch_target_publisher(
    production: RevalidatedFormalServingSessionGroupProductionSpec,
) -> CanonicalJsonProofBinding:
    """Wait for the synchronous runtime callback's durable publication."""

    target_path = Path(production.spec.server_watch_target_path)
    while not target_path.is_file():
        await asyncio.sleep(0.05)
    return CanonicalJsonProofBinding.bind(target_path)


async def execute_formal_serving_session_group_production(
    production_spec_path: str | Path,
    *,
    runtime: FormalServingResidentPhysicalRuntime | None = None,
    execution_runner: Callable[
        ..., Awaitable[FormalServingSessionGroupExecutionResult]
    ] = (execute_formal_serving_session_group),
    actual_validator: Callable[..., Any] | None = None,
    close_revalidator: Callable[..., Any] = (
        revalidate_formal_serving_resident_shared_close_receipt
    ),
    watch_target_publisher: Callable[
        [RevalidatedFormalServingSessionGroupProductionSpec],
        Awaitable[CanonicalJsonProofBinding],
    ] = _default_server_watch_target_publisher,
    watch_target_revalidator: Callable[..., Any] = (
        revalidate_formal_serving_session_group_server_watch_target
    ),
    clock_ns: Callable[[], int] = time.time_ns,
    heartbeat_interval_seconds: float = 30.0,
) -> FormalServingSessionGroupProductionPublication:
    """Execute, close, validate, fan out controls, then publish one commit."""

    production = revalidate_formal_serving_session_group_production_spec(
        production_spec_path
    )
    ensure_formal_serving_session_group_production_outputs_unoccupied(production.spec)
    if actual_validator is None:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            validate_formal_single_operator_cell_actual,
        )

        actual_validator = validate_formal_single_operator_cell_actual
    owned_runtime = runtime
    if owned_runtime is None:
        factory = PinnedSglangResidentProcessFactory(
            nvidia_smi_tool=production.spec.nvidia_smi_tool,
            repository_root=production.spec.repository_root,
        )
        owned_runtime = FormalServingResidentPhysicalRuntime(
            factory=factory,
            evidence_root=production.spec.resident_evidence_root,
            repository_root=production.spec.repository_root,
            active_target_publisher=(
                formal_serving_session_group_active_target_publisher(
                    production.binding.absolute_path
                )
            ),
        )
    leader = production.spec.members[0]
    heartbeat = FormalServingSessionGroupChildHeartbeatPublisher(
        path=production.spec.heartbeat_path,
        cell_id=leader.cell_id,
        attempt=leader.attempt,
        command_sha256=leader.command_sha256,
        clock_ns=clock_ns,
        interval_seconds=heartbeat_interval_seconds,
    )
    worker_started_ns = int(clock_ns())
    if worker_started_ns < 1:
        raise ValueError("production worker clock is invalid")
    heartbeat.start("RUNNING")
    watch_task: asyncio.Task[CanonicalJsonProofBinding] | None = None
    completed = False
    try:
        watch_task = asyncio.create_task(watch_target_publisher(production))
        await execution_runner(
            execution_spec_path=production.execution.spec_binding.absolute_path,
            runtime=owned_runtime,
        )
        heartbeat.set_phase("TRACE_SEALED")
        (
            result_binding,
            result,
            artifacts,
            close_binding,
            _close,
        ) = _revalidate_group_result_and_close(
            production, close_revalidator=close_revalidator
        )
        heartbeat.set_phase("FINALIZING")
        if watch_task.done():
            watch_binding = await watch_task
        else:
            try:
                watch_binding = await asyncio.wait_for(watch_task, timeout=30.0)
            except TimeoutError as error:
                raise FormalServingSessionGroupProductionError(
                    "resident server watch target was not durably published"
                ) from error
        rebound_watch, _target = watch_target_revalidator(
            production.spec.server_watch_target_path
        )
        if rebound_watch != watch_binding:
            raise ValueError("resident server watch target changed")
        controls = tuple(
            _publish_member_control(
                production=production,
                member_index=index,
                artifact_binding=result.cell_artifacts[index],
                artifact=artifact,
                actual_validator=actual_validator,
                clock_ns=clock_ns,
                fallback_started_ns=worker_started_ns,
            )
            for index, artifact in enumerate(artifacts)
        )
        worker_finished_ns = max(int(clock_ns()), worker_started_ns + 1)
        publication = FormalServingSessionGroupProductionPublication(
            schema_version=1,
            kind=_PUBLICATION_KIND,
            protocol_sha256=FORMAL_SERVING_SESSION_GROUP_PRODUCTION_PROTOCOL_SHA256,
            production_spec=production.binding,
            group_execution_result=result_binding,
            shared_close=close_binding,
            server_watch_target=watch_binding,
            controls=controls,
            worker_started_ns=worker_started_ns,
            worker_finished_ns=worker_finished_ns,
            commit_marker="SHARED_CLOSE_AND_ALL_MEMBER_CONTROLS_PUBLISHED",
            evidence_level=_EVIDENCE_LEVEL,
            formal_measured=False,
        )
        # This is intentionally the final immutable file of a successful group.
        publish_canonical_json_no_replace(
            production.spec.shared_publication_path, publication.to_dict()
        )
        reopened = revalidate_formal_serving_session_group_production_publication(
            production.binding.absolute_path,
            actual_validator=actual_validator,
            close_revalidator=close_revalidator,
            watch_target_revalidator=watch_target_revalidator,
        )
        if reopened.publication != publication:
            raise RuntimeError("production shared publication changed")
        completed = True
        return publication
    finally:
        if watch_task is not None and not watch_task.done():
            watch_task.cancel()
            await asyncio.gather(watch_task, return_exceptions=True)
        close_error: BaseException | None = None
        try:
            await owned_runtime.force_close_active()
        except BaseException as error:  # noqa: BLE001 - do not skip heartbeat
            close_error = error
        heartbeat.stop("COMPLETE" if completed else "FAILED")
        if close_error is not None and completed:
            raise FormalServingSessionGroupProductionError(
                "production runtime final force-close failed"
            ) from close_error


def _revalidate_pointer_and_terminal(
    *,
    member: FormalServingSessionGroupProductionMemberSpec,
    worker: FormalCellWorkerSpec,
    control: FormalServingSessionGroupControlEvidence,
    actual_validator: Callable[..., Any],
    shared_sha256: str,
    common_evidence: Mapping[str, str],
) -> TerminalEvidence:
    pointer = _read_object(member.atomic_pointer_path)
    terminal = _read_object(member.expected_terminal_path)
    if (
        pointer.get("kind") != "formal_experiment_atomic_result_pointer"
        or pointer.get("cell_id") != member.cell_id
        or pointer.get("attempt") != member.attempt
        or pointer.get("command_sha256") != member.command_sha256
        or pointer.get("terminal_path") != member.expected_terminal_path
        or pointer.get("junit_path") != member.expected_junit_path
        or pointer.get("raw_log_path") != member.expected_raw_log_path
    ):
        raise ValueError("production member pointer identity differs")
    pointer_without = dict(pointer)
    pointer_digest = pointer_without.pop("pointer_sha256", None)
    if pointer_digest != hashlib.sha256(_canonical_bytes(pointer_without)).hexdigest():
        raise ValueError("production member pointer digest differs")
    expected_files = {
        "terminal_sha256": file_sha256(member.expected_terminal_path),
        "junit_sha256": file_sha256(member.expected_junit_path),
        "raw_log_sha256": file_sha256(member.expected_raw_log_path),
    }
    if any(pointer.get(key) != digest for key, digest in expected_files.items()):
        raise ValueError("production member pointer file digest differs")
    if (
        terminal.get("schema_version") != 1
        or terminal.get("kind") != "formal_experiment_terminal"
        or terminal.get("cell_id") != member.cell_id
        or terminal.get("attempt") != member.attempt
        or terminal.get("command_sha256") != member.command_sha256
        or terminal.get("status") != control.status
    ):
        raise ValueError("production member terminal identity differs")
    _validate_junit(
        Path(member.expected_junit_path), require_clean=control.status == "COMPLETE"
    )
    raw = _read_object(member.expected_raw_log_path)
    if (
        raw.get("kind") != _CONTROL_KIND
        or raw.get("protocol_sha256")
        != FORMAL_SERVING_SESSION_GROUP_PRODUCTION_PROTOCOL_SHA256
        or raw.get("cell_id") != member.cell_id
        or raw.get("attempt") != member.attempt
        or raw.get("command_sha256") != member.command_sha256
        or raw.get("status") != control.status
        or raw.get("cell_artifact") != control.cell_artifact.to_dict()
    ):
        raise ValueError("production member raw control identity differs")
    evidence = {
        **common_evidence,
        member.cell_worker_spec_path: file_sha256(member.cell_worker_spec_path),
        control.cell_artifact.absolute_path: file_sha256(
            control.cell_artifact.absolute_path
        ),
        member.expected_terminal_path: expected_files["terminal_sha256"],
        member.expected_junit_path: expected_files["junit_sha256"],
        member.expected_raw_log_path: expected_files["raw_log_sha256"],
        member.atomic_pointer_path: file_sha256(member.atomic_pointer_path),
    }
    if control.status == "COMPLETE":
        if control.actual_result is None:
            raise ValueError("complete production control lacks actual")
        validation = actual_validator(
            node_materialization_path=worker.node_materialization_path,
            cell_id=member.cell_id,
            actual_result_path=control.actual_result.absolute_path,
            repository_root=worker.repository_root,
        )
        if (
            validation.status != "COMPLETE"
            or validation.result_identity_sha256 != control.result_identity_sha256
            or validation.validator_kind != control.validator_kind
            or validation.validator_protocol_sha256 != control.validator_protocol_sha256
            or raw.get("actual_result") != control.actual_result.to_dict()
            or raw.get("result_identity_sha256") != control.result_identity_sha256
            or raw.get("validator_kind") != control.validator_kind
            or raw.get("validator_protocol_sha256") != control.validator_protocol_sha256
        ):
            raise ValueError("production member actual validation differs")
        evidence[control.actual_result.absolute_path] = file_sha256(
            control.actual_result.absolute_path
        )
    status = control.status
    started_ns = terminal.get("started_ns")
    finished_ns = terminal.get("finished_ns")
    if type(started_ns) is not int or type(finished_ns) is not int:
        raise ValueError("production member terminal lifecycle differs")
    return TerminalEvidence(
        status=status,
        exit_code=terminal.get("exit_code"),  # type: ignore[arg-type]
        atomic_publication_sha256=shared_sha256,
        terminal_sha256=expected_files["terminal_sha256"],
        junit_sha256=expected_files["junit_sha256"],
        raw_log_sha256=expected_files["raw_log_sha256"],
        evidence_files=evidence,
        failure_class=terminal.get("failure_class"),  # type: ignore[arg-type]
        failure_code=terminal.get("failure_code"),  # type: ignore[arg-type]
        exclusion_reason=terminal.get("exclusion_reason"),  # type: ignore[arg-type]
        included_in_analysis=bool(terminal.get("included_in_analysis")),
        started_ns=started_ns,
        finished_ns=finished_ns,
    )


def revalidate_formal_serving_session_group_production_publication(
    production_spec_path: str | Path,
    *,
    actual_validator: Callable[..., Any] | None = None,
    close_revalidator: Callable[..., Any] = (
        revalidate_formal_serving_resident_shared_close_receipt
    ),
    watch_target_revalidator: Callable[..., Any] = (
        revalidate_formal_serving_session_group_server_watch_target
    ),
) -> RevalidatedFormalServingSessionGroupProductionPublication:
    """Path-only deep reopen; accepts no expected result or verifier token."""

    production = revalidate_formal_serving_session_group_production_spec(
        production_spec_path
    )
    if actual_validator is None:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            validate_formal_single_operator_cell_actual,
        )

        actual_validator = validate_formal_single_operator_cell_actual
    publication_binding = CanonicalJsonProofBinding.bind(
        production.spec.shared_publication_path
    )
    publication = FormalServingSessionGroupProductionPublication.from_dict(
        publication_binding.reopen()
    )
    (
        result_binding,
        result,
        artifacts,
        close_binding,
        _close,
    ) = _revalidate_group_result_and_close(
        production, close_revalidator=close_revalidator
    )
    watch_binding, _target = watch_target_revalidator(
        production.spec.server_watch_target_path
    )
    if (
        publication.production_spec != production.binding
        or publication.group_execution_result != result_binding
        or publication.shared_close != close_binding
        or publication.server_watch_target != watch_binding
        or tuple((row.cell_id, row.attempt) for row in publication.controls)
        != tuple((row.cell_id, row.attempt) for row in production.spec.members)
    ):
        raise ValueError("production shared publication lineage differs")
    shared_raw_sha256 = file_sha256(publication_binding.absolute_path)
    common_evidence = {
        production.binding.absolute_path: file_sha256(production.binding.absolute_path),
        result_binding.absolute_path: file_sha256(result_binding.absolute_path),
        close_binding.absolute_path: file_sha256(close_binding.absolute_path),
        watch_binding.absolute_path: file_sha256(watch_binding.absolute_path),
        publication_binding.absolute_path: shared_raw_sha256,
    }
    terminals: dict[str, TerminalEvidence] = {}
    for index, (member, worker, control, artifact) in enumerate(
        zip(
            production.spec.members,
            production.cell_worker_specs,
            publication.controls,
            artifacts,
            strict=True,
        )
    ):
        if (
            control.command_sha256 != member.command_sha256
            or control.cell_artifact != result.cell_artifacts[index]
            or control.status != artifact.status
            or control.terminal.absolute_path != member.expected_terminal_path
            or control.junit.absolute_path != member.expected_junit_path
            or control.raw_log.absolute_path != member.expected_raw_log_path
            or control.atomic_pointer.absolute_path != member.atomic_pointer_path
        ):
            raise ValueError("production shared control binding differs")
        terminals[member.cell_id] = _revalidate_pointer_and_terminal(
            member=member,
            worker=worker,
            control=control,
            actual_validator=actual_validator,
            shared_sha256=shared_raw_sha256,
            common_evidence=common_evidence,
        )
    if len({row.atomic_publication_sha256 for row in terminals.values()}) != 1:
        raise RuntimeError("production group terminal fanout lost shared identity")
    return RevalidatedFormalServingSessionGroupProductionPublication(
        binding=publication_binding,
        publication=publication,
        spec=production,
        result=result,
        artifacts=artifacts,
        terminals=terminals,
    )


def revalidate_formal_serving_session_group_production_terminals(
    production_spec_path: str | Path,
    commands: Sequence[QueuedCommandSpec],
    *,
    actual_validator: Callable[..., Any] | None = None,
    close_revalidator: Callable[..., Any] = (
        revalidate_formal_serving_resident_shared_close_receipt
    ),
    watch_target_revalidator: Callable[..., Any] = (
        revalidate_formal_serving_session_group_server_watch_target
    ),
) -> Mapping[str, TerminalEvidence]:
    reopened = revalidate_formal_serving_session_group_production_publication(
        production_spec_path,
        actual_validator=actual_validator,
        close_revalidator=close_revalidator,
        watch_target_revalidator=watch_target_revalidator,
    )
    command_rows = tuple(
        (row.cell_id, row.attempt, row.command_sha256) for row in commands
    )
    expected = tuple(
        (row.cell_id, row.attempt, row.command_sha256)
        for row in reopened.spec.spec.members
    )
    if command_rows != expected:
        raise ValueError("production terminal commands differ from group order")
    return reopened.terminals


def revalidate_formal_serving_session_group_production_terminal(
    command: QueuedCommandSpec,
    attempt: Mapping[str, object],
    observation: ProcessObservation,
    *,
    actual_validator: Callable[..., Any] | None = None,
) -> TerminalEvidence | None:
    """ProductionSchedulerRuntime callback; TRACE_SEALED still returns None."""

    spec_path = formal_serving_session_group_production_spec_path_from_command(command)
    spec = revalidate_formal_serving_session_group_production_spec(spec_path)
    if not Path(spec.spec.shared_publication_path).is_file():
        return None
    if type(observation) is not ProcessObservation or observation.alive:
        return None
    if attempt.get("cell_id") not in {None, command.cell_id} or attempt.get(
        "attempt"
    ) not in {None, command.attempt}:
        raise ValueError("production terminal ledger attempt identity differs")
    reopened = revalidate_formal_serving_session_group_production_publication(
        spec_path, actual_validator=actual_validator
    )
    return reopened.terminals[command.cell_id]


async def _run_with_signals(spec_path: str | Path) -> int:
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(
        execute_formal_serving_session_group_production(spec_path)
    )
    installed: list[signal.Signals] = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signum, task.cancel)
        except (NotImplementedError, RuntimeError):
            continue
        installed.append(signum)
    try:
        await task
        return 0
    except asyncio.CancelledError:
        return 143
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return asyncio.run(_run_with_signals(arguments.spec))


__all__ = (
    "FORMAL_SERVING_SESSION_GROUP_PRODUCTION_PROTOCOL_SHA256",
    "FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV",
    "FormalServingSessionGroupChildHeartbeatPublisher",
    "FormalServingSessionGroupControlEvidence",
    "FormalServingSessionGroupProductionError",
    "FormalServingSessionGroupProductionMemberSpec",
    "FormalServingSessionGroupProductionPublication",
    "FormalServingSessionGroupProductionSpec",
    "FormalServingSessionGroupServerWatchTarget",
    "RevalidatedFormalServingSessionGroupProductionPublication",
    "RevalidatedFormalServingSessionGroupProductionSpec",
    "build_formal_serving_session_group_production_spec",
    "ensure_formal_serving_session_group_production_outputs_unoccupied",
    "execute_formal_serving_session_group_production",
    "formal_serving_session_group_active_target_publisher",
    "formal_serving_session_group_production_environment",
    "formal_serving_session_group_production_spec_path_from_command",
    "formal_serving_session_group_shared_evidence_bound_bytes",
    "main",
    "publish_formal_serving_session_group_production_spec",
    "publish_formal_serving_session_group_server_watch_target",
    "revalidate_formal_serving_session_group_production_publication",
    "revalidate_formal_serving_session_group_production_spec",
    "revalidate_formal_serving_session_group_production_terminal",
    "revalidate_formal_serving_session_group_production_terminals",
    "revalidate_formal_serving_session_group_server_watch_target",
)


if __name__ == "__main__":
    raise SystemExit(main())
