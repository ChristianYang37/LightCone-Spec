"""Production OS callbacks for the formal experiment scheduler.

The state machine lives in :mod:`experiment_operator`.  This module is the
small trusted boundary that launches an already-materialized argv without a
shell, records a durable child exit receipt, samples local GPU/disk state, and
deep-validates atomic terminal and archive evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from lightcone_spec.orchestration.experiment_operator import (
    ArchiveCallbacks,
    ArchiveRequest,
    ArchiveStepReceipt,
    ProcessObservation,
    QueuedCommandSpec,
    RecoveredProcessStart,
    RetryBuilder,
    SchedulerCallbacks,
    SpawnedProcess,
    TerminalEvidence,
    WorkerHeartbeat,
    inspect_local_process,
)

_SHA256 = frozenset("0123456789abcdef")
_TERMINAL_FAILURE_CLASSES = frozenset(
    {
        "INFRASTRUCTURE",
        "SCIENTIFIC",
        "UNSAFE",
        "OOM_CANDIDATE",
        "EXACTNESS",
        "FAILURE_DIAGNOSTIC",
    }
)
_START_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "command_sha256",
        "wrapper_pid",
        "wrapper_pgid",
        "process_start_identity",
        "started_ns",
        "receipt_sha256",
    }
)
_HEARTBEAT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "cell_id",
        "attempt",
        "command_sha256",
        "worker_pid",
        "sequence",
        "observed_at_ns",
        "phase",
    }
)
_EXIT_RECEIPT_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "kind",
        "command_sha256",
        "wrapper_pid",
        "wrapper_pgid",
        "child_pid",
        "started_ns",
        "finished_ns",
        "exit_code",
        "launch_error_type",
        "receipt_sha256",
    }
)
_EXIT_RECEIPT_FIELDS_V2 = _EXIT_RECEIPT_FIELDS_V1 | frozenset(
    {"process_start_receipt_sha256"}
)
_TERMINAL_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "kind",
        "cell_id",
        "attempt",
        "command_sha256",
        "status",
        "exit_code",
        "failure_class",
        "failure_code",
        "exclusion_reason",
        "included_in_analysis",
        "started_ns",
        "finished_ns",
    }
)
_TERMINAL_VALIDATION_FIELDS = frozenset(
    {
        "worker_spec_path",
        "worker_spec_sha256",
        "node_materialization_path",
        "actual_result_path",
        "actual_result_raw_sha256",
        "result_identity_sha256",
        "validator_kind",
        "validator_protocol_sha256",
        "evidence_manifest_path",
        "evidence_manifest_sha256",
    }
)
_TERMINAL_FIELDS_V2 = _TERMINAL_FIELDS_V1 | _TERMINAL_VALIDATION_FIELDS
_POINTER_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "cell_id",
        "attempt",
        "command_sha256",
        "terminal_path",
        "terminal_sha256",
        "junit_path",
        "junit_sha256",
        "raw_log_path",
        "raw_log_sha256",
        "published_ns",
        "pointer_sha256",
    }
)
_MANIFEST_FIELDS = frozenset({"schema_version", "kind", "files"})
_MANIFEST_ROW_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_MANIFEST_NAME = "sha256_manifest.json"
MINIMUM_LOCAL_ARCHIVE_FREE_BYTES = 100 * 1024**3


class ProductionCallbackError(RuntimeError):
    """Raised when an OS callback cannot establish an exact result."""


@dataclass(frozen=True)
class NvidiaSmiGpu:
    index: int
    uuid: str
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    power_draw_watts: float


@dataclass(frozen=True)
class OperatorTerminalContext:
    """Scheduler-injected identity needed to atomically publish one result."""

    cell_id: str
    attempt: int
    command_sha256: str
    expected_terminal_path: str
    expected_junit_path: str
    expected_raw_log_path: str
    atomic_pointer_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise ValueError("operator terminal cell ID is empty")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int):
            raise TypeError("operator terminal attempt must be an integer")
        if self.attempt < 1:
            raise ValueError("operator terminal attempt must be positive")
        _require_sha256(self.command_sha256, "operator terminal command SHA-256")
        paths = (
            self.expected_terminal_path,
            self.expected_junit_path,
            self.expected_raw_log_path,
            self.atomic_pointer_path,
        )
        if len(set(paths)) != len(paths):
            raise ValueError("operator terminal output paths must be distinct")
        for value in paths:
            path = Path(value)
            if not path.is_absolute() or path != path.resolve(strict=False):
                raise ValueError("operator terminal output path must be absolute")

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> OperatorTerminalContext:
        values = os.environ if environment is None else environment
        try:
            attempt = int(values["LIGHTCONE_OPERATOR_ATTEMPT"])
            return cls(
                cell_id=values["LIGHTCONE_OPERATOR_CELL_ID"],
                attempt=attempt,
                command_sha256=values["LIGHTCONE_OPERATOR_COMMAND_SHA256"],
                expected_terminal_path=values["LIGHTCONE_OPERATOR_TERMINAL_PATH"],
                expected_junit_path=values["LIGHTCONE_OPERATOR_JUNIT_PATH"],
                expected_raw_log_path=values["LIGHTCONE_OPERATOR_RAW_LOG_PATH"],
                atomic_pointer_path=values["LIGHTCONE_OPERATOR_POINTER_PATH"],
            )
        except (KeyError, ValueError) as error:
            raise ProductionCallbackError(
                "scheduler-owned terminal environment is incomplete"
            ) from error


def canonical_json_bytes(value: object) -> bytes:
    """Return the one accepted JSON byte representation for new receipts."""

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


def file_sha256(path: str | Path) -> str:
    source = _regular_file(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_nvidia_smi(
    *,
    executable: str = "nvidia-smi",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[NvidiaSmiGpu, ...]:
    """Query exact UUID-index telemetry without changing GPU state."""

    completed = runner(
        [
            executable,
            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    rows: list[NvidiaSmiGpu] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 6:
            raise ProductionCallbackError("nvidia-smi returned a malformed row")
        try:
            row = NvidiaSmiGpu(
                index=int(columns[0]),
                uuid=columns[1],
                utilization_percent=float(columns[2]),
                memory_used_mib=float(columns[3]),
                memory_total_mib=float(columns[4]),
                power_draw_watts=float(columns[5]),
            )
        except ValueError as error:
            raise ProductionCallbackError(
                "nvidia-smi returned a non-numeric telemetry value"
            ) from error
        if row.index < 0 or not row.uuid.startswith("GPU-"):
            raise ProductionCallbackError("nvidia-smi returned invalid GPU identity")
        rows.append(row)
    rows.sort(key=lambda row: row.index)
    if (
        not rows
        or len({row.index for row in rows}) != len(rows)
        or len({row.uuid for row in rows}) != len(rows)
    ):
        raise ProductionCallbackError("nvidia-smi GPU inventory is empty or duplicate")
    return tuple(rows)


def statvfs_free_bytes(path: str | Path) -> int:
    """Return unprivileged bytes available on the path's filesystem."""

    target = Path(path)
    probe = target if target.exists() else target.parent
    values = os.statvfs(probe)
    return int(values.f_frsize) * int(values.f_bavail)


def regular_file_size(path: str | Path) -> int:
    return _regular_file(path).stat().st_size


class ProductionSchedulerRuntime:
    """Concrete process/GPU/evidence callbacks used by ``scheduler-run``."""

    def __init__(
        self,
        *,
        nvidia_smi_executable: str = "nvidia-smi",
        python_executable: str = sys.executable,
        nvidia_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        retry_builder: RetryBuilder | None = None,
    ) -> None:
        if retry_builder is not None and not callable(retry_builder):
            raise TypeError("production retry builder must be callable or null")
        self.nvidia_smi_executable = nvidia_smi_executable
        self.python_executable = python_executable
        self.nvidia_runner = nvidia_runner
        self.popen = popen
        self.retry_builder = retry_builder
        self._children: dict[int, subprocess.Popen[bytes]] = {}

    def callbacks(self) -> SchedulerCallbacks:
        return SchedulerCallbacks(
            launch=self.launch,
            process_probe=self.process_probe,
            log_size_bytes=self.log_size_bytes,
            gpu_snapshot=self.gpu_snapshot,
            terminal_validator=self.terminal_validator,
            free_disk_bytes=statvfs_free_bytes,
            retry_builder=self.retry_builder,
            recover_started_process=self.recover_started_process,
            worker_heartbeat=self.worker_heartbeat,
            worker_heartbeat_required=self.worker_heartbeat_required,
            send_term=self.send_term,
            send_kill=self.send_kill,
            process_group_alive=self.process_group_alive,
            independent_process_groups=self.independent_process_groups,
            partial_evidence=self.partial_evidence,
        )

    def launch(
        self,
        command: QueuedCommandSpec,
        gpu_uuids: tuple[str, ...],
    ) -> SpawnedProcess:
        from lightcone_spec.orchestration.formal_preflight_exact_ten_group_worker import (
            ensure_formal_preflight_exact_ten_group_outputs_unoccupied,
            formal_preflight_exact_ten_group_spec_path,
            revalidate_formal_preflight_exact_ten_group_worker_spec,
        )
        from lightcone_spec.orchestration.formal_serving_session_group_production import (
            FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV,
            ensure_formal_serving_session_group_production_outputs_unoccupied,
            formal_serving_session_group_production_spec_path_from_command,
            revalidate_formal_serving_session_group_production_spec,
        )

        inventory = query_nvidia_smi(
            executable=self.nvidia_smi_executable,
            runner=self.nvidia_runner,
        )
        index_by_uuid = {row.uuid: row.index for row in inventory}
        if len(gpu_uuids) != command.required_gpu_count or any(
            gpu_uuid not in index_by_uuid for gpu_uuid in gpu_uuids
        ):
            raise ProductionCallbackError(
                "assigned GPU UUIDs differ from the current nvidia-smi inventory"
            )
        group_spec_path = formal_preflight_exact_ten_group_spec_path(command)
        resident_spec_path: Path | None = None
        child_argv = list(command.argv)
        if group_spec_path is not None:
            try:
                group_spec = revalidate_formal_preflight_exact_ten_group_worker_spec(
                    group_spec_path,
                    expected_command=command,
                )
                if gpu_uuids != group_spec.gpu_uuids:
                    raise ValueError(
                        "assigned GPUs differ from exact-ten source authority"
                    )
                ensure_formal_preflight_exact_ten_group_outputs_unoccupied(group_spec)
            except (TypeError, ValueError, RuntimeError, OSError) as error:
                raise ProductionCallbackError(
                    "exact-ten group launch spec failed deep validation"
                ) from error
            child_argv = [
                self.python_executable,
                "-m",
                (
                    "lightcone_spec.orchestration."
                    "formal_preflight_exact_ten_group_worker"
                ),
                "--group-spec",
                group_spec_path,
            ]
        elif FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV in dict(
            command.environment
        ):
            try:
                resident_spec_path = (
                    formal_serving_session_group_production_spec_path_from_command(
                        command
                    )
                )
                resident = revalidate_formal_serving_session_group_production_spec(
                    resident_spec_path
                )
                if gpu_uuids != resident.execution.plan.assigned_gpu_uuids:
                    raise ValueError(
                        "assigned GPU differs from resident group source authority"
                    )
                ensure_formal_serving_session_group_production_outputs_unoccupied(
                    resident.spec
                )
            except (TypeError, ValueError, RuntimeError, OSError) as error:
                raise ProductionCallbackError(
                    "resident serving group launch spec failed deep validation"
                ) from error
            child_argv = [
                self.python_executable,
                "-m",
                (
                    "lightcone_spec.orchestration."
                    "formal_serving_session_group_production"
                ),
                "--spec",
                str(resident_spec_path),
            ]
        receipt = Path(command.child_exit_receipt_path)
        start_receipt = child_start_receipt_path(command)
        heartbeat = child_heartbeat_path(command)
        if receipt.exists() or receipt.is_symlink():
            raise ProductionCallbackError("child exit receipt path is already occupied")
        if start_receipt.exists() or start_receipt.is_symlink():
            raise ProductionCallbackError(
                "child start receipt path is already occupied"
            )
        if heartbeat.exists() or heartbeat.is_symlink():
            raise ProductionCallbackError("child heartbeat path is already occupied")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        log_descriptor = _open_command_log(command.log_path)
        environment = os.environ.copy()
        environment.update(dict(command.environment))
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_uuids)
        environment["LIGHTCONE_ASSIGNED_GPU_UUIDS"] = ",".join(gpu_uuids)
        environment["LIGHTCONE_GPU_UUID_TO_INDEX_JSON"] = json.dumps(
            {gpu_uuid: index_by_uuid[gpu_uuid] for gpu_uuid in gpu_uuids},
            sort_keys=True,
            separators=(",", ":"),
        )
        environment.update(
            {
                "LIGHTCONE_OPERATOR_CELL_ID": command.cell_id,
                "LIGHTCONE_OPERATOR_ATTEMPT": str(command.attempt),
                "LIGHTCONE_OPERATOR_COMMAND_SHA256": command.command_sha256,
                "LIGHTCONE_OPERATOR_TERMINAL_PATH": (command.expected_terminal_path),
                "LIGHTCONE_OPERATOR_JUNIT_PATH": command.expected_junit_path,
                "LIGHTCONE_OPERATOR_RAW_LOG_PATH": command.expected_raw_log_path,
                "LIGHTCONE_OPERATOR_POINTER_PATH": command.atomic_pointer_path,
                "LIGHTCONE_OPERATOR_COMMAND_LOG_PATH": command.log_path,
                "LIGHTCONE_OPERATOR_CHILD_EXIT_RECEIPT_PATH": (
                    command.child_exit_receipt_path
                ),
                "LIGHTCONE_OPERATOR_CHILD_START_RECEIPT_PATH": str(start_receipt),
                "LIGHTCONE_OPERATOR_HEARTBEAT_PATH": str(child_heartbeat_path(command)),
            }
        )
        wrapper_argv = [
            self.python_executable,
            "-m",
            __name__,
            "child-wrapper",
            "--start-receipt",
            str(start_receipt),
            "--exit-receipt",
            command.child_exit_receipt_path,
            "--command-sha256",
            command.command_sha256,
            "--",
            *child_argv,
        ]
        try:
            child = self.popen(
                wrapper_argv,
                stdin=subprocess.DEVNULL,
                stdout=log_descriptor,
                stderr=subprocess.STDOUT,
                env=environment,
                shell=False,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            os.close(log_descriptor)
        pgid = os.getpgid(child.pid)
        if pgid != child.pid:
            child.terminate()
            raise ProductionCallbackError(
                "child wrapper did not become a session leader"
            )
        start_receipt_sha256: str | None = None
        start_deadline = time.monotonic() + 2.0
        while time.monotonic() < start_deadline:
            if start_receipt.exists() or start_receipt.is_symlink():
                start = _validated_start_receipt(start_receipt, command=command)
                if start["wrapper_pid"] != child.pid or start["wrapper_pgid"] != pgid:
                    child.terminate()
                    raise ProductionCallbackError(
                        "child start receipt process identity differs"
                    )
                start_receipt_sha256 = file_sha256(start_receipt)
                break
            if child.poll() is not None:
                break
            time.sleep(0.01)
        self._children[child.pid] = child
        return SpawnedProcess(child.pid, pgid, start_receipt_sha256)

    def recover_started_process(
        self,
        command: QueuedCommandSpec,
    ) -> RecoveredProcessStart | None:
        """Recover one wrapper only from its immutable, source-bound receipt."""

        path = child_start_receipt_path(command)
        if not path.exists() and not path.is_symlink():
            return None
        receipt = _validated_start_receipt(path, command=command)
        pid = int(receipt["wrapper_pid"])
        pgid = int(receipt["wrapper_pgid"])
        if pid != pgid:
            raise ProductionCallbackError(
                "recoverable child wrapper is not a setsid session leader"
            )
        if receipt["process_start_identity"].get("kind") != "linux_proc_start_v1":
            raise ProductionCallbackError(
                "process start recovery requires Linux /proc identity"
            )
        current_identity = _linux_process_start_identity(pid)
        if current_identity is not None and (
            current_identity != receipt["process_start_identity"]
        ):
            raise ProductionCallbackError(
                "child start receipt PID was reused by another process"
            )
        return RecoveredProcessStart(
            pid=pid,
            pgid=pgid,
            started_ns=int(receipt["started_ns"]),
            receipt_sha256=file_sha256(path),
        )

    def process_probe(self, pid: int, expected_pgid: int) -> ProcessObservation:
        known = self._children.get(pid)
        if known is not None:
            exit_code = known.poll()
            if exit_code is not None:
                self._children.pop(pid, None)
                return ProcessObservation(
                    pid,
                    False,
                    None,
                    "known_child_exited",
                    exit_code=exit_code,
                )
        observation = inspect_local_process(pid, expected_pgid)
        if not observation.alive and known is not None:
            try:
                exit_code = known.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                exit_code = None
            if exit_code is not None:
                self._children.pop(pid, None)
                return ProcessObservation(
                    pid,
                    False,
                    None,
                    "known_child_exited_during_probe",
                    exit_code=exit_code,
                )
        if observation.alive and _linux_process_is_zombie(pid):
            return ProcessObservation(pid, False, None, "process_is_zombie")
        return observation

    @staticmethod
    def log_size_bytes(command: QueuedCommandSpec) -> int:
        total = regular_file_size(command.log_path)
        raw_path = Path(command.expected_raw_log_path)
        if raw_path.exists() or raw_path.is_symlink():
            total += regular_file_size(raw_path)
        encoded_progress = dict(command.environment).get(
            "LIGHTCONE_OPERATOR_PROGRESS_LOG_PATHS_JSON"
        )
        if encoded_progress is not None:
            try:
                progress_paths = json.loads(encoded_progress)
            except json.JSONDecodeError as error:
                raise ProductionCallbackError(
                    "operator progress-log paths are not JSON"
                ) from error
            if (
                type(progress_paths) is not list
                or not progress_paths
                or len(progress_paths) != len(set(progress_paths))
                or any(
                    type(value) is not str
                    or not value
                    or not Path(value).is_absolute()
                    or Path(value) != Path(value).resolve(strict=False)
                    for value in progress_paths
                )
            ):
                raise ProductionCallbackError(
                    "operator progress-log paths differ from the bound contract"
                )
            for value in progress_paths:
                path = Path(value)
                if path.exists() or path.is_symlink():
                    total += regular_file_size(path)
        return total

    @staticmethod
    def worker_heartbeat_required(command: QueuedCommandSpec) -> bool:
        environment = dict(command.environment)
        return (
            "LIGHTCONE_CELL_WORKER_SPEC_SHA256" in environment
            or "LIGHTCONE_PREFLIGHT_EXACT_TEN_GROUP_SPEC_PATH" in environment
            or (
                "LIGHTCONE_FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_PATH"
                in environment
            )
        )

    @staticmethod
    def worker_heartbeat(command: QueuedCommandSpec) -> WorkerHeartbeat | None:
        path = child_heartbeat_path(command)
        if not path.exists() and not path.is_symlink():
            return None
        value = _read_canonical_json(path)
        _require_exact_fields(value, _HEARTBEAT_FIELDS, "child heartbeat")
        if (
            value["schema_version"] != 1
            or value["kind"] != "formal_experiment_child_heartbeat"
            or value["cell_id"] != command.cell_id
            or value["attempt"] != command.attempt
            or value["command_sha256"] != command.command_sha256
        ):
            raise ProductionCallbackError("child heartbeat identity differs")
        try:
            return WorkerHeartbeat(
                command_sha256=value["command_sha256"],
                worker_pid=value["worker_pid"],
                sequence=value["sequence"],
                observed_at_ns=value["observed_at_ns"],
                phase=value["phase"],
            )
        except (TypeError, ValueError) as error:
            raise ProductionCallbackError("child heartbeat values differ") from error

    @staticmethod
    def process_group_alive(pgid: int) -> bool:
        if isinstance(pgid, bool) or not isinstance(pgid, int) or pgid < 1:
            raise ValueError("process-group probe PGID must be positive")
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def send_term(self, command: QueuedCommandSpec, pid: int, pgid: int) -> None:
        # Reopen the independent resident-server target before signalling its
        # wrapper.  TERM remains graceful: the production worker catches it,
        # calls force_close_active(), and publishes the close evidence.
        self._resident_server_watch_target(command)
        if not self._revalidate_live_wrapper(command, pid, pgid):
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def send_kill(self, command: QueuedCommandSpec, pid: int, pgid: int) -> None:
        # The resident server is a deliberate setsid descendant and therefore
        # is outside the wrapper PGID.  Kill only the immutable target recorded
        # before the first trace; never rescan descendants during escalation.
        target = self._resident_server_watch_target(command)
        if target is not None:
            identity = linux_process_start_identity(target.server_process_id)
            if identity is not None:
                if identity != {
                    "kind": "linux_proc_start_v1",
                    "boot_id": target.server_boot_id,
                    "start_time_ticks": target.server_start_time_ticks,
                }:
                    raise ProductionCallbackError("resident KILL target PID was reused")
                try:
                    observed_pgid = os.getpgid(target.server_process_id)
                except ProcessLookupError:
                    observed_pgid = None
                if observed_pgid is not None and observed_pgid != (
                    target.server_process_group_id
                ):
                    raise ProductionCallbackError(
                        "resident KILL target escaped its registered PGID"
                    )
                if observed_pgid is not None:
                    try:
                        os.killpg(target.server_process_group_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        wrapper_live = self._revalidate_live_wrapper(command, pid, pgid)
        if not wrapper_live:
            return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return

    def independent_process_groups(self, command: QueuedCommandSpec) -> tuple[int, ...]:
        """Return only source-bound setsid targets, never a process-tree scan."""

        target = self._resident_server_watch_target(command)
        return () if target is None else (target.server_process_group_id,)

    @staticmethod
    def _resident_server_watch_target(command: QueuedCommandSpec) -> Any | None:
        from lightcone_spec.orchestration.formal_serving_session_group_production import (
            FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV,
            formal_serving_session_group_production_spec_path_from_command,
            revalidate_formal_serving_session_group_production_spec,
            revalidate_formal_serving_session_group_server_watch_target,
        )

        if FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV not in dict(
            command.environment
        ):
            return None
        production_path = (
            formal_serving_session_group_production_spec_path_from_command(command)
        )
        production = revalidate_formal_serving_session_group_production_spec(
            production_path
        )
        target_path = Path(production.spec.server_watch_target_path)
        if not target_path.exists() and not target_path.is_symlink():
            return None
        _binding, target = revalidate_formal_serving_session_group_server_watch_target(
            target_path
        )
        return target

    @staticmethod
    def partial_evidence(command: QueuedCommandSpec) -> Mapping[str, str]:
        candidates: tuple[Path, ...] = (
            child_start_receipt_path(command),
            Path(command.child_exit_receipt_path),
            Path(command.log_path),
            Path(command.expected_raw_log_path),
            Path(command.expected_junit_path),
            Path(command.expected_terminal_path),
            Path(command.atomic_pointer_path),
        )
        try:
            from lightcone_spec.orchestration.formal_serving_session_group_production import (
                FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV,
                formal_serving_session_group_production_spec_path_from_command,
                revalidate_formal_serving_session_group_production_spec,
            )

            if FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV in dict(
                command.environment
            ):
                production_path = (
                    formal_serving_session_group_production_spec_path_from_command(
                        command
                    )
                )
                production = revalidate_formal_serving_session_group_production_spec(
                    production_path
                )
                candidates = (
                    *candidates,
                    Path(production_path),
                    Path(production.spec.server_watch_target_path),
                    Path(production.spec.shared_close_path),
                    Path(production.spec.shared_publication_path),
                )
        except (TypeError, ValueError, RuntimeError, OSError):
            # The scheduler records whatever immutable prefix exists; terminal
            # validation remains the fail-closed place for malformed lineage.
            pass
        evidence: dict[str, str] = {}
        for path in candidates:
            if path.exists() or path.is_symlink():
                evidence[str(path)] = file_sha256(path)
        return evidence

    @staticmethod
    def _revalidate_live_wrapper(
        command: QueuedCommandSpec,
        pid: int,
        pgid: int,
    ) -> bool:
        receipt = _validated_start_receipt(
            child_start_receipt_path(command),
            command=command,
        )
        if (
            receipt["wrapper_pid"] != pid
            or receipt["wrapper_pgid"] != pgid
            or pid != pgid
        ):
            raise ProductionCallbackError("signal target differs from start receipt")
        identity = _linux_process_start_identity(pid)
        if identity is None:
            return False
        if identity != receipt["process_start_identity"]:
            raise ProductionCallbackError("signal target PID was reused")
        try:
            observed_pgid = os.getpgid(pid)
        except ProcessLookupError:
            return False
        if observed_pgid != pgid:
            raise ProductionCallbackError("signal target escaped its registered group")
        return True

    def gpu_snapshot(self, gpu_uuids: tuple[str, ...]) -> Mapping[str, Any]:
        inventory = query_nvidia_smi(
            executable=self.nvidia_smi_executable,
            runner=self.nvidia_runner,
        )
        by_uuid = {row.uuid: row for row in inventory}
        if any(gpu_uuid not in by_uuid for gpu_uuid in gpu_uuids):
            raise ProductionCallbackError("assigned GPU disappeared from nvidia-smi")
        return {
            gpu_uuid: {
                "index": by_uuid[gpu_uuid].index,
                "utilization_percent": by_uuid[gpu_uuid].utilization_percent,
                "memory_used_mib": by_uuid[gpu_uuid].memory_used_mib,
                "memory_total_mib": by_uuid[gpu_uuid].memory_total_mib,
                "power_draw_watts": by_uuid[gpu_uuid].power_draw_watts,
            }
            for gpu_uuid in gpu_uuids
        }

    def terminal_validator(
        self,
        command: QueuedCommandSpec,
        attempt: Mapping[str, Any],
        observation: ProcessObservation,
    ) -> TerminalEvidence | None:
        from lightcone_spec.orchestration.formal_preflight_exact_ten_group_worker import (
            FormalPreflightExactTenGroupError,
            formal_preflight_exact_ten_group_spec_path,
            revalidate_formal_preflight_exact_ten_group_terminal,
            revalidate_formal_preflight_exact_ten_group_worker_spec,
        )
        from lightcone_spec.orchestration.formal_serving_session_group_physical import (
            revalidate_formal_serving_resident_shared_close_receipt,
        )
        from lightcone_spec.orchestration.formal_serving_session_group_production import (
            FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV,
            FormalServingSessionGroupProductionError,
            formal_serving_session_group_production_spec_path_from_command,
            revalidate_formal_serving_session_group_production_spec,
            revalidate_formal_serving_session_group_production_terminal,
            revalidate_formal_serving_session_group_server_watch_target,
        )

        if FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV in dict(
            command.environment
        ):
            try:
                terminal = revalidate_formal_serving_session_group_production_terminal(
                    command,
                    attempt,
                    observation,
                )
                if terminal is None:
                    return None
                production_path = (
                    formal_serving_session_group_production_spec_path_from_command(
                        command
                    )
                )
                production = revalidate_formal_serving_session_group_production_spec(
                    production_path
                )
                close_binding, close = (
                    revalidate_formal_serving_resident_shared_close_receipt(
                        production.spec.shared_close_path
                    )
                )
                watch_binding, watch = (
                    revalidate_formal_serving_session_group_server_watch_target(
                        production.spec.server_watch_target_path
                    )
                )
                start_path = Path(production.spec.wrapper_start_receipt_path)
                start = _validated_start_receipt(start_path, command=command)
                start_sha256 = file_sha256(start_path)
                if (
                    (start["wrapper_pid"], start["wrapper_pgid"])
                    != (attempt.get("pid"), attempt.get("pgid"))
                    or (
                        watch.wrapper_process_id,
                        watch.wrapper_process_group_id,
                        watch.wrapper_started_ns,
                    )
                    != (
                        start["wrapper_pid"],
                        start["wrapper_pgid"],
                        start["started_ns"],
                    )
                    or watch.wrapper_start_receipt.raw_sha256 != start_sha256
                ):
                    raise ProductionCallbackError(
                        "resident group wrapper/watch-target lineage differs"
                    )
                registered_start = attempt.get("process_start_receipt_sha256")
                if registered_start is not None and registered_start != start_sha256:
                    raise ProductionCallbackError(
                        "resident group ledger/start-receipt digest differs"
                    )
                exit_path = Path(command.child_exit_receipt_path)
                exit_receipt = _validated_exit_receipt(
                    exit_path,
                    command=command,
                    expected_pid=attempt.get("pid"),
                    expected_pgid=attempt.get("pgid"),
                )
                if observation.exit_code is not None and observation.exit_code != (
                    _normalized_wrapper_exit(exit_receipt["exit_code"])
                ):
                    raise ProductionCallbackError(
                        "resident group process observation and exit receipt differ"
                    )
                return replace(
                    terminal,
                    # The operator charges the physical group once through its
                    # leader.  Use the one resident-process lifetime for that
                    # accounting row rather than one member's scored trace.
                    started_ns=close.server_process_started_ns,
                    finished_ns=close.process_exited_ns,
                    evidence_files={
                        **dict(terminal.evidence_files or {}),
                        str(start_path): start_sha256,
                        str(exit_path): file_sha256(exit_path),
                        watch_binding.absolute_path: watch_binding.raw_sha256,
                        close_binding.absolute_path: close_binding.raw_sha256,
                    },
                )
            except (
                FormalServingSessionGroupProductionError,
                TypeError,
                ValueError,
                RuntimeError,
                OSError,
            ) as error:
                raise ProductionCallbackError(
                    "resident serving group terminal failed deep validation"
                ) from error

        group_spec_path = formal_preflight_exact_ten_group_spec_path(command)
        if group_spec_path is not None:
            try:
                terminal = revalidate_formal_preflight_exact_ten_group_terminal(
                    command,
                    attempt,
                    observation,
                )
                if terminal is None:
                    return None
                group_spec = revalidate_formal_preflight_exact_ten_group_worker_spec(
                    group_spec_path,
                    expected_command=command,
                )
                leader_exit = Path(group_spec.members[0].child_exit_receipt_path)
                start_path = leader_exit.with_name(f"{leader_exit.name}.start.json")
                start = _validated_start_receipt(start_path, command=command)
                if (
                    start["wrapper_pid"] != attempt.get("pid")
                    or start["wrapper_pgid"] != attempt.get("pgid")
                    or start["started_ns"] != terminal.started_ns
                ):
                    raise ProductionCallbackError(
                        "exact-ten start receipt lineage differs"
                    )
                start_file_sha = file_sha256(start_path)
                registered_start_sha = attempt.get("process_start_receipt_sha256")
                if (
                    registered_start_sha is not None
                    and registered_start_sha != start_file_sha
                ):
                    raise ProductionCallbackError(
                        "exact-ten ledger and start receipt digest disagree"
                    )
                return replace(
                    terminal,
                    evidence_files={
                        **dict(terminal.evidence_files or {}),
                        str(start_path): start_file_sha,
                    },
                )
            except (
                FormalPreflightExactTenGroupError,
                TypeError,
                ValueError,
                OSError,
            ) as error:
                raise ProductionCallbackError(
                    "exact-ten group terminal failed deep validation"
                ) from error
        receipt_path = Path(command.child_exit_receipt_path)
        pointer_path = Path(command.atomic_pointer_path)
        if not receipt_path.exists():
            return None
        if not pointer_path.exists():
            raise ProductionCallbackError(
                "durable child exit exists without an atomic terminal pointer"
            )
        receipt = _validated_exit_receipt(
            receipt_path,
            command=command,
            expected_pid=attempt.get("pid"),
            expected_pgid=attempt.get("pgid"),
        )
        start_path = child_start_receipt_path(command)
        start = _validated_start_receipt(start_path, command=command)
        start_file_sha = file_sha256(start_path)
        registered_start_sha = attempt.get("process_start_receipt_sha256")
        if registered_start_sha is not None and registered_start_sha != start_file_sha:
            raise ProductionCallbackError(
                "ledger and child start receipt SHA-256 disagree"
            )
        if (
            start["wrapper_pid"] != receipt["wrapper_pid"]
            or start["wrapper_pgid"] != receipt["wrapper_pgid"]
            or start["started_ns"] != receipt["started_ns"]
        ):
            raise ProductionCallbackError("child start/exit receipt lineage differs")
        if observation.exit_code is not None and observation.exit_code != (
            _normalized_wrapper_exit(receipt["exit_code"])
        ):
            raise ProductionCallbackError(
                "process observation and durable exit receipt disagree"
            )
        pointer = _read_canonical_json(pointer_path)
        _require_exact_fields(pointer, _POINTER_FIELDS, "atomic pointer")
        if (
            pointer["schema_version"] != 1
            or pointer["kind"] != "formal_experiment_atomic_result_pointer"
            or pointer["cell_id"] != command.cell_id
            or pointer["attempt"] != command.attempt
            or pointer["command_sha256"] != command.command_sha256
            or pointer["terminal_path"] != command.expected_terminal_path
            or pointer["junit_path"] != command.expected_junit_path
            or pointer["raw_log_path"] != command.expected_raw_log_path
            or type(pointer["published_ns"]) is not int
            or pointer["published_ns"] <= 0
        ):
            raise ProductionCallbackError("atomic pointer identity differs")
        pointer_without_digest = dict(pointer)
        pointer_digest = pointer_without_digest.pop("pointer_sha256")
        _require_sha256(pointer_digest, "pointer self digest")
        if _content_sha256(pointer_without_digest) != pointer_digest:
            raise ProductionCallbackError("atomic pointer self digest differs")

        terminal_path = Path(command.expected_terminal_path)
        junit_path = Path(command.expected_junit_path)
        raw_log_path = Path(command.expected_raw_log_path)
        terminal_sha = file_sha256(terminal_path)
        junit_sha = file_sha256(junit_path)
        raw_log_sha = file_sha256(raw_log_path)
        for label, actual, registered in (
            ("terminal", terminal_sha, pointer["terminal_sha256"]),
            ("JUnit", junit_sha, pointer["junit_sha256"]),
            ("raw log", raw_log_sha, pointer["raw_log_sha256"]),
        ):
            _require_sha256(registered, f"{label} pointer digest")
            if actual != registered:
                raise ProductionCallbackError(f"{label} SHA-256 differs")

        terminal = _read_canonical_json(terminal_path)
        schema_version = terminal.get("schema_version")
        terminal_fields = (
            _TERMINAL_FIELDS_V1 if schema_version == 1 else _TERMINAL_FIELDS_V2
        )
        _require_exact_fields(terminal, terminal_fields, "terminal")
        if (
            schema_version not in {1, 2}
            or terminal["kind"] != "formal_experiment_terminal"
            or terminal["cell_id"] != command.cell_id
            or terminal["attempt"] != command.attempt
            or terminal["command_sha256"] != command.command_sha256
            or terminal["exit_code"] != receipt["exit_code"]
            or type(terminal["started_ns"]) is not int
            or type(terminal["finished_ns"]) is not int
            or terminal["started_ns"] <= 0
            or terminal["finished_ns"] <= terminal["started_ns"]
            or terminal["started_ns"] < receipt["started_ns"]
            or terminal["finished_ns"] > receipt["finished_ns"]
        ):
            raise ProductionCallbackError("terminal identity or timing differs")
        _validate_junit(junit_path, require_clean=terminal["status"] == "COMPLETE")
        status = terminal["status"]
        if status == "COMPLETE":
            if (
                receipt["exit_code"] != 0
                or terminal["failure_class"] is not None
                or terminal["failure_code"] is not None
                or terminal["included_in_analysis"]
                == (terminal["exclusion_reason"] is not None)
            ):
                raise ProductionCallbackError("COMPLETE terminal semantics differ")
        elif status == "FAILED":
            if (
                terminal["failure_class"] not in _TERMINAL_FAILURE_CLASSES
                or not isinstance(terminal["failure_code"], str)
                or not terminal["failure_code"]
                or not isinstance(terminal["exclusion_reason"], str)
                or not terminal["exclusion_reason"]
                or terminal["included_in_analysis"] is not False
            ):
                raise ProductionCallbackError("FAILED terminal semantics differ")
        else:
            raise ProductionCallbackError("terminal status is not registered")
        validation_evidence: dict[str, str] = {}
        if schema_version == 2:
            from lightcone_spec.orchestration.formal_cell_worker import (
                revalidate_formal_cell_worker_terminal,
            )

            validation_evidence = revalidate_formal_cell_worker_terminal(
                terminal,
                command=command,
            )
        return TerminalEvidence(
            status=status,
            exit_code=receipt["exit_code"],
            atomic_publication_sha256=file_sha256(pointer_path),
            terminal_sha256=terminal_sha,
            junit_sha256=junit_sha,
            raw_log_sha256=raw_log_sha,
            evidence_files={
                command.atomic_pointer_path: file_sha256(pointer_path),
                str(start_path): start_file_sha,
                command.child_exit_receipt_path: file_sha256(receipt_path),
                **validation_evidence,
            },
            failure_class=terminal["failure_class"],
            failure_code=terminal["failure_code"],
            exclusion_reason=terminal["exclusion_reason"],
            included_in_analysis=terminal["included_in_analysis"],
            started_ns=receipt["started_ns"],
            finished_ns=receipt["finished_ns"],
        )


class ProductionArchiveRuntime:
    """Rsync, verify, atomically publish, and rehydrate an archive payload."""

    def __init__(
        self,
        *,
        rsync_executable: str = "rsync",
        runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
        full_rehydrate: bool = True,
        minimum_local_free_bytes: int = MINIMUM_LOCAL_ARCHIVE_FREE_BYTES,
        rsync_source: Callable[[ArchiveRequest], str] | None = None,
        rsync_remote_shell: str | None = None,
    ) -> None:
        if not isinstance(full_rehydrate, bool):
            raise TypeError("full_rehydrate must be boolean")
        if (
            isinstance(minimum_local_free_bytes, bool)
            or not isinstance(minimum_local_free_bytes, int)
            or minimum_local_free_bytes < 0
        ):
            raise ValueError("minimum local archive free bytes must be non-negative")
        self.rsync_executable = rsync_executable
        self.runner = runner
        self.full_rehydrate = full_rehydrate
        self.minimum_local_free_bytes = minimum_local_free_bytes
        self.rsync_source = rsync_source
        if rsync_remote_shell is not None and (
            not isinstance(rsync_remote_shell, str)
            or not rsync_remote_shell
            or "\x00" in rsync_remote_shell
            or "\n" in rsync_remote_shell
            or "\r" in rsync_remote_shell
        ):
            raise ValueError("rsync remote shell must be canonical single-line text")
        self.rsync_remote_shell = rsync_remote_shell

    def callbacks(self) -> ArchiveCallbacks:
        return ArchiveCallbacks(self.transfer, self.verify_local_sha, self.rehydrate)

    def transfer(
        self,
        request: ArchiveRequest,
        previous: ArchiveStepReceipt | None,
    ) -> ArchiveStepReceipt:
        if previous is not None:
            raise ProductionCallbackError("archive transfer must be the first step")
        partial = Path(request.local_partial_root)
        final = Path(request.local_final_root)
        if final.exists() or final.is_symlink():
            raise ProductionCallbackError("archive final path already exists")
        partial.parent.mkdir(parents=True, exist_ok=True)
        if partial.is_symlink() or (partial.exists() and not partial.is_dir()):
            raise ProductionCallbackError("archive partial path is unsafe")
        free_bytes = statvfs_free_bytes(partial.parent)
        required_free_bytes = (
            request.predicted_payload_bytes + self.minimum_local_free_bytes
        )
        if free_bytes < required_free_bytes:
            raise ProductionCallbackError(
                "local archive capacity is below payload plus retained reserve"
            )
        partial.mkdir(mode=0o700, exist_ok=True)
        source = (
            request.remote_payload_root
            if self.rsync_source is None
            else self.rsync_source(request)
        )
        if (
            not isinstance(source, str)
            or not source
            or "\x00" in source
            or "\n" in source
            or "\r" in source
        ):
            raise ProductionCallbackError("archive rsync source is invalid")
        argv = [self.rsync_executable, "-a", "--checksum"]
        if self.rsync_remote_shell is not None:
            argv.extend(("-e", self.rsync_remote_shell))
        argv.extend(("--", source.rstrip("/") + "/", str(partial) + "/"))
        self.runner(
            argv,
            check=True,
            shell=False,
        )
        manifest, rows = _load_and_verify_manifest(
            partial,
            request.remote_manifest_sha256,
        )
        checked_bytes = sum(row["size_bytes"] for row in rows)
        if (
            request.predicted_payload_bytes > 0
            and checked_bytes != request.predicted_payload_bytes
        ):
            raise ProductionCallbackError(
                "archive payload bytes differ from the registered prediction"
            )
        return ArchiveStepReceipt(
            "TRANSFER",
            request.remote_manifest_sha256,
            _archive_evidence_sha("TRANSFER", manifest, rows),
            len(rows),
            checked_bytes,
        )

    def verify_local_sha(
        self,
        request: ArchiveRequest,
        previous: ArchiveStepReceipt | None,
    ) -> ArchiveStepReceipt:
        if previous is None or previous.step != "TRANSFER":
            raise ProductionCallbackError(
                "local SHA verification lacks transfer receipt"
            )
        partial = Path(request.local_partial_root)
        final = Path(request.local_final_root)
        root = final if final.is_dir() and not partial.exists() else partial
        manifest, rows = _load_and_verify_manifest(
            root,
            request.remote_manifest_sha256,
        )
        if root == partial:
            if final.exists() or final.is_symlink():
                raise ProductionCallbackError("archive final path became occupied")
            os.rename(partial, final)
            _fsync_directory(final.parent)
        checked_bytes = sum(row["size_bytes"] for row in rows)
        return ArchiveStepReceipt(
            "LOCAL_SHA_VERIFY",
            request.remote_manifest_sha256,
            _archive_evidence_sha("LOCAL_SHA_VERIFY", manifest, rows),
            len(rows),
            checked_bytes,
        )

    def rehydrate(
        self,
        request: ArchiveRequest,
        previous: ArchiveStepReceipt | None,
    ) -> ArchiveStepReceipt:
        if previous is None or previous.step != "LOCAL_SHA_VERIFY":
            raise ProductionCallbackError("rehydrate lacks local SHA receipt")
        final = Path(request.local_final_root)
        manifest, rows = _load_and_verify_manifest(
            final,
            request.remote_manifest_sha256,
        )
        selected = rows if self.full_rehydrate else _deterministic_sample(rows)
        verified: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(
            prefix=f".{request.archive_id}.rehydrate.",
            dir=final.parent,
        ) as temporary:
            temporary_root = Path(temporary)
            for row in selected:
                source = _manifest_file(final, row["path"])
                target = temporary_root / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target, follow_symlinks=False)
                if file_sha256(target) != row["sha256"]:
                    raise ProductionCallbackError("rehydrated file SHA-256 differs")
                verified.append(dict(row))
        content_tree_sha = _content_sha256(
            {"manifest_sha256": request.remote_manifest_sha256, "files": verified}
        )
        return ArchiveStepReceipt(
            "REHYDRATE_VERIFY",
            request.remote_manifest_sha256,
            _archive_evidence_sha("REHYDRATE_VERIFY", manifest, verified),
            len(verified),
            sum(row["size_bytes"] for row in verified),
            content_tree_sha256=content_tree_sha,
        )


def publish_atomic_terminal_result(
    command: QueuedCommandSpec | OperatorTerminalContext,
    *,
    status: str,
    exit_code: int,
    started_ns: int,
    finished_ns: int,
    failure_class: str | None = None,
    failure_code: str | None = None,
    exclusion_reason: str | None = None,
    included_in_analysis: bool = True,
    validation_metadata: Mapping[str, object] | None = None,
) -> None:
    """Publish terminal/pointer after the job wrote distinct JUnit and raw logs.

    ``command.log`` is only the wrapper's combined stdout/stderr stream.  The
    physical job is responsible for publishing ``expected_raw_log_path`` as a
    separate protocol artifact before calling this function.
    """

    _regular_file(command.expected_junit_path)
    _regular_file(command.expected_raw_log_path)
    if type(command) not in {QueuedCommandSpec, OperatorTerminalContext}:
        raise TypeError("atomic terminal requires an exact operator command identity")
    metadata = dict(validation_metadata or {})
    if metadata and set(metadata) != _TERMINAL_VALIDATION_FIELDS:
        raise ValueError("atomic terminal validation metadata fields differ")
    terminal = {
        "schema_version": 2 if metadata else 1,
        "kind": "formal_experiment_terminal",
        "cell_id": command.cell_id,
        "attempt": command.attempt,
        "command_sha256": command.command_sha256,
        "status": status,
        "exit_code": exit_code,
        "failure_class": failure_class,
        "failure_code": failure_code,
        "exclusion_reason": exclusion_reason,
        "included_in_analysis": included_in_analysis,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        **metadata,
    }
    _atomic_write_new_json(Path(command.expected_terminal_path), terminal)
    pointer_without_digest = {
        "schema_version": 1,
        "kind": "formal_experiment_atomic_result_pointer",
        "cell_id": command.cell_id,
        "attempt": command.attempt,
        "command_sha256": command.command_sha256,
        "terminal_path": command.expected_terminal_path,
        "terminal_sha256": file_sha256(command.expected_terminal_path),
        "junit_path": command.expected_junit_path,
        "junit_sha256": file_sha256(command.expected_junit_path),
        "raw_log_path": command.expected_raw_log_path,
        "raw_log_sha256": file_sha256(command.expected_raw_log_path),
        "published_ns": time.time_ns(),
    }
    pointer = {
        **pointer_without_digest,
        "pointer_sha256": _content_sha256(pointer_without_digest),
    }
    _atomic_write_new_json(Path(command.atomic_pointer_path), pointer)


def child_start_receipt_path(command: QueuedCommandSpec) -> Path:
    """Return the deterministic sibling start receipt for one queued command."""

    if type(command) is not QueuedCommandSpec:
        raise TypeError("start receipt path requires an exact queued command")
    exit_path = Path(command.child_exit_receipt_path)
    return exit_path.with_name(f"{exit_path.name}.start.json")


def child_heartbeat_path(command: QueuedCommandSpec) -> Path:
    """Return the mutable child-owned heartbeat path for one attempt."""

    if type(command) is not QueuedCommandSpec:
        raise TypeError("heartbeat path requires an exact queued command")
    exit_path = Path(command.child_exit_receipt_path)
    return exit_path.with_name(f"{exit_path.name}.heartbeat.json")


def _linux_process_start_identity(pid: int) -> dict[str, object] | None:
    """Return Linux PID-reuse-resistant identity, or ``None`` after process exit."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise ValueError("process identity PID must be positive")
    stat_path = Path(f"/proc/{pid}/stat")
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    try:
        stat_text = stat_path.read_text(encoding="ascii")
        boot_id = boot_path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as error:
        raise ProductionCallbackError(
            "cannot read Linux process start identity"
        ) from error
    closing = stat_text.rfind(")")
    fields = stat_text[closing + 2 :].split() if closing >= 0 else []
    if len(fields) <= 19 or not fields[19].isdigit() or not boot_id:
        raise ProductionCallbackError("Linux process start identity is malformed")
    return {
        "kind": "linux_proc_start_v1",
        "boot_id": boot_id,
        "start_time_ticks": int(fields[19]),
    }


def linux_process_start_identity(pid: int) -> dict[str, object] | None:
    """Public read-only Linux PID identity for durable descendant targets."""

    identity = _linux_process_start_identity(pid)
    return None if identity is None else dict(identity)


def _wrapper_process_start_identity(pid: int) -> dict[str, object]:
    identity = _linux_process_start_identity(pid)
    if identity is not None:
        return identity
    return {
        "kind": "non_linux_process_start_unrecoverable_v1",
        "pid": pid,
        "platform": sys.platform,
    }


def run_child_wrapper(
    argv: Sequence[str],
    *,
    exit_receipt_path: str | Path,
    command_sha256: str,
    start_receipt_path: str | Path | None = None,
    require_session_leader: bool = False,
) -> int:
    """Run one argv in this wrapper's persistent process group and seal exit."""

    if not argv or any(type(value) is not str or not value for value in argv):
        raise ValueError("child wrapper argv must be non-empty strings")
    _require_sha256(command_sha256, "child wrapper command SHA-256")
    receipt_path = Path(exit_receipt_path)
    start_path = (
        Path(start_receipt_path)
        if start_receipt_path is not None
        else receipt_path.with_name(f"{receipt_path.name}.start.json")
    )
    if start_path == receipt_path:
        raise ProductionCallbackError("child start and exit receipts must differ")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ProductionCallbackError("child exit receipt already exists")
    if start_path.exists() or start_path.is_symlink():
        raise ProductionCallbackError("child start receipt already exists")
    started_ns = time.time_ns()
    wrapper_pid = os.getpid()
    wrapper_pgid = os.getpgrp()
    if require_session_leader and wrapper_pid != wrapper_pgid:
        raise ProductionCallbackError("child wrapper is not a setsid session leader")
    start_without_digest = {
        "schema_version": 1,
        "kind": "formal_experiment_child_start_receipt",
        "command_sha256": command_sha256,
        "wrapper_pid": wrapper_pid,
        "wrapper_pgid": wrapper_pgid,
        "process_start_identity": _wrapper_process_start_identity(wrapper_pid),
        "started_ns": started_ns,
    }
    start_receipt = {
        **start_without_digest,
        "receipt_sha256": _content_sha256(start_without_digest),
    }
    _atomic_write_new_json(start_path, start_receipt)
    child_pid: int | None = None
    launch_error_type: str | None = None
    prior_handlers: dict[int, Any] = {}
    try:
        child = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            shell=False,
            start_new_session=False,
            close_fds=True,
        )
        child_pid = child.pid

        def forward(signum: int, _frame: object) -> None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                prior_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, forward)
            exit_code = child.wait()
        finally:
            for signum, prior in prior_handlers.items():
                signal.signal(signum, prior)
    except OSError as error:  # wrapper must leave a durable launch-failure outcome
        launch_error_type = type(error).__name__
        exit_code = 127
    finished_ns = max(time.time_ns(), started_ns + 1)
    receipt_without_digest = {
        "schema_version": 1,
        "kind": "formal_experiment_child_exit_receipt",
        "command_sha256": command_sha256,
        "wrapper_pid": wrapper_pid,
        "wrapper_pgid": wrapper_pgid,
        "child_pid": child_pid,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "exit_code": exit_code,
        "launch_error_type": launch_error_type,
    }
    receipt = {
        **receipt_without_digest,
        "receipt_sha256": _content_sha256(receipt_without_digest),
    }
    _atomic_write_new_json(receipt_path, receipt)
    return _normalized_wrapper_exit(exit_code)


def _validated_start_receipt(
    path: Path,
    *,
    command: QueuedCommandSpec,
) -> dict[str, Any]:
    return _validated_start_receipt_for_sha256(
        path,
        command_sha256=command.command_sha256,
    )


def _validated_start_receipt_for_sha256(
    path: Path,
    *,
    command_sha256: str,
) -> dict[str, Any]:
    _require_sha256(command_sha256, "child start expected command SHA-256")
    value = _read_canonical_json(path)
    _require_exact_fields(value, _START_RECEIPT_FIELDS, "child start receipt")
    digest = value["receipt_sha256"]
    _require_sha256(digest, "child start receipt self digest")
    without_digest = dict(value)
    without_digest.pop("receipt_sha256")
    if _content_sha256(without_digest) != digest:
        raise ProductionCallbackError("child start receipt self digest differs")
    identity = value["process_start_identity"]
    if type(identity) is not dict:
        raise ProductionCallbackError("child start process identity is not an object")
    identity_kind = identity.get("kind")
    if identity_kind == "linux_proc_start_v1":
        if (
            set(identity) != {"kind", "boot_id", "start_time_ticks"}
            or type(identity["boot_id"]) is not str
            or not identity["boot_id"]
            or type(identity["start_time_ticks"]) is not int
            or identity["start_time_ticks"] < 1
        ):
            raise ProductionCallbackError("Linux process start identity differs")
    elif identity_kind == "non_linux_process_start_unrecoverable_v1":
        if (
            set(identity) != {"kind", "pid", "platform"}
            or identity["pid"] != value["wrapper_pid"]
            or type(identity["platform"]) is not str
            or not identity["platform"]
        ):
            raise ProductionCallbackError("portable process start identity differs")
    else:
        raise ProductionCallbackError("process start identity kind is not registered")
    if (
        value["schema_version"] != 1
        or value["kind"] != "formal_experiment_child_start_receipt"
        or value["command_sha256"] != command_sha256
        or type(value["wrapper_pid"]) is not int
        or value["wrapper_pid"] < 1
        or type(value["wrapper_pgid"]) is not int
        or value["wrapper_pgid"] < 1
        or type(value["started_ns"]) is not int
        or value["started_ns"] < 1
    ):
        raise ProductionCallbackError("child start receipt identity differs")
    return value


def revalidate_child_start_receipt(
    path: str | Path,
    *,
    command_sha256: str,
) -> RecoveredProcessStart:
    """Deep-validate one generic wrapper receipt and resist live PID reuse."""

    source = Path(path)
    value = _validated_start_receipt_for_sha256(
        source,
        command_sha256=command_sha256,
    )
    pid = int(value["wrapper_pid"])
    pgid = int(value["wrapper_pgid"])
    if pid != pgid:
        raise ProductionCallbackError("child start wrapper is not a session leader")
    identity = value["process_start_identity"]
    if identity.get("kind") == "linux_proc_start_v1":
        current = _linux_process_start_identity(pid)
        if current is not None and current != identity:
            raise ProductionCallbackError(
                "child start receipt PID was reused by another process"
            )
    return RecoveredProcessStart(
        pid=pid,
        pgid=pgid,
        started_ns=int(value["started_ns"]),
        receipt_sha256=file_sha256(source),
    )


def _validated_exit_receipt(
    path: Path,
    *,
    command: QueuedCommandSpec,
    expected_pid: object,
    expected_pgid: object,
) -> dict[str, Any]:
    value = _read_canonical_json(path)
    expected_fields = (
        _EXIT_RECEIPT_FIELDS_V2
        if value.get("schema_version") == 2
        else _EXIT_RECEIPT_FIELDS_V1
    )
    _require_exact_fields(value, expected_fields, "child exit receipt")
    digest = value["receipt_sha256"]
    _require_sha256(digest, "child exit receipt self digest")
    without_digest = dict(value)
    without_digest.pop("receipt_sha256")
    if _content_sha256(without_digest) != digest:
        raise ProductionCallbackError("child exit receipt self digest differs")
    if (
        value["schema_version"] not in {1, 2}
        or value["kind"] != "formal_experiment_child_exit_receipt"
        or value["command_sha256"] != command.command_sha256
        or value["wrapper_pid"] != expected_pid
        or value["wrapper_pgid"] != expected_pgid
        or type(value["started_ns"]) is not int
        or type(value["finished_ns"]) is not int
        or value["started_ns"] <= 0
        or value["finished_ns"] <= value["started_ns"]
        or type(value["exit_code"]) is not int
        or value["launch_error_type"] is not None
    ):
        raise ProductionCallbackError("child exit receipt identity differs")
    if value["schema_version"] == 2:
        start_path = child_start_receipt_path(command)
        start = _validated_start_receipt(start_path, command=command)
        if (
            file_sha256(start_path) != value["process_start_receipt_sha256"]
            or start["wrapper_pid"] != value["wrapper_pid"]
            or start["wrapper_pgid"] != value["wrapper_pgid"]
            or start["started_ns"] != value["started_ns"]
        ):
            raise ProductionCallbackError("child start/exit receipt lineage differs")
    return value


def _open_command_log(path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(target, flags, 0o600)
    except FileExistsError:
        source = _regular_file(target)
        append_flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        append_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source, append_flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ProductionCallbackError("command log is not regular")
        return descriptor


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    source = _regular_file(path)
    data = source.read_bytes()
    if not data or len(data) > 16 * 1024 * 1024:
        raise ProductionCallbackError("evidence JSON is empty or too large")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProductionCallbackError("evidence JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProductionCallbackError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionCallbackError("evidence JSON is invalid") from error
    if type(value) is not dict or data != canonical_json_bytes(value):
        raise ProductionCallbackError("evidence JSON is not canonical")
    return value


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ProductionCallbackError(f"refusing to overwrite evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        payload = canonical_json_bytes(dict(value))
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _regular_file(path: str | Path) -> Path:
    source = Path(path)
    try:
        metadata = source.lstat()
    except FileNotFoundError as error:
        raise ProductionCallbackError(f"required file is missing: {source}") from error
    if not stat.S_ISREG(metadata.st_mode) or source.is_symlink():
        raise ProductionCallbackError(f"required path is not a regular file: {source}")
    return source


def _validate_junit(path: Path, *, require_clean: bool) -> None:
    try:
        root = ET.parse(_regular_file(path)).getroot()
    except ET.ParseError as error:
        raise ProductionCallbackError("JUnit XML is malformed") from error
    tag = root.tag.rsplit("}", 1)[-1]
    suites = [root] if tag == "testsuite" else list(root) if tag == "testsuites" else []
    if not suites:
        raise ProductionCallbackError("JUnit root is not testsuite/testsuites")
    totals = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for name in totals:
            try:
                totals[name] += int(suite.attrib.get(name, "0"))
            except ValueError as error:
                raise ProductionCallbackError("JUnit count is non-integral") from error
    if totals["tests"] < 1:
        raise ProductionCallbackError("JUnit reports no tests")
    if require_clean and any(
        totals[name] for name in ("failures", "errors", "skipped")
    ):
        raise ProductionCallbackError("COMPLETE JUnit is not clean")


def _load_and_verify_manifest(
    root: Path,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_sha256(expected_manifest_sha256, "archive manifest SHA-256")
    if root.is_symlink() or not root.is_dir():
        raise ProductionCallbackError("archive root is not a safe directory")
    manifest_path = root / _MANIFEST_NAME
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ProductionCallbackError("archive manifest SHA-256 differs")
    manifest = _read_canonical_json(manifest_path)
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "archive manifest")
    if (
        manifest["schema_version"] != 1
        or manifest["kind"] != "formal_archive_sha256_manifest"
        or type(manifest["files"]) is not list
        or not manifest["files"]
    ):
        raise ProductionCallbackError("archive manifest identity differs")
    rows: list[dict[str, Any]] = []
    prior_path: str | None = None
    for raw in manifest["files"]:
        if type(raw) is not dict:
            raise ProductionCallbackError("archive manifest row is not an object")
        _require_exact_fields(raw, _MANIFEST_ROW_FIELDS, "archive manifest row")
        relative = raw["path"]
        if type(relative) is not str or not relative:
            raise ProductionCallbackError("archive manifest path is empty")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
            raise ProductionCallbackError("archive manifest path escapes its root")
        if prior_path is not None and relative <= prior_path:
            raise ProductionCallbackError(
                "archive manifest paths are not unique/sorted"
            )
        prior_path = relative
        _require_sha256(raw["sha256"], "archive payload SHA-256")
        if type(raw["size_bytes"]) is not int or raw["size_bytes"] < 0:
            raise ProductionCallbackError("archive payload size is invalid")
        source = _manifest_file(root, relative)
        if (
            source.stat().st_size != raw["size_bytes"]
            or file_sha256(source) != raw["sha256"]
        ):
            raise ProductionCallbackError("archive payload identity differs")
        rows.append(dict(raw))
    registered = {row["path"] for row in rows}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != _MANIFEST_NAME
    }
    if actual != registered:
        raise ProductionCallbackError("archive contains unregistered or missing files")
    return manifest, rows


def _manifest_file(root: Path, relative: str) -> Path:
    source = root.joinpath(*PurePosixPath(relative).parts)
    resolved_root = root.resolve(strict=True)
    resolved_source = source.resolve(strict=True)
    if (
        resolved_source.parent != resolved_root
        and resolved_root not in resolved_source.parents
    ):
        raise ProductionCallbackError("archive payload resolves outside its root")
    return _regular_file(source)


def _deterministic_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) <= 3:
        return rows
    indexes = sorted({0, len(rows) // 2, len(rows) - 1})
    return [rows[index] for index in indexes]


def _archive_evidence_sha(
    step: str,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    return _content_sha256({"step": step, "manifest": manifest, "checked": list(rows)})


def _content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ProductionCallbackError(f"{label} is not lowercase SHA-256")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    if frozenset(value) != expected:
        raise ProductionCallbackError(f"{label} field set differs")


def _normalized_wrapper_exit(exit_code: int) -> int:
    if exit_code < 0:
        return min(255, 128 + abs(exit_code))
    return min(255, exit_code)


def _linux_process_is_zombie(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="ascii").split()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return False
    return len(fields) > 2 and fields[2] == "Z"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    wrapper = commands.add_parser("child-wrapper")
    wrapper.add_argument("--start-receipt", required=True)
    wrapper.add_argument("--exit-receipt", required=True)
    wrapper.add_argument("--command-sha256", required=True)
    wrapper.add_argument("argv", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _wrapper_parser().parse_args(argv)
    if arguments.command != "child-wrapper":
        raise AssertionError("unhandled production callback command")
    child_argv = list(arguments.argv)
    if child_argv and child_argv[0] == "--":
        child_argv.pop(0)
    return run_child_wrapper(
        child_argv,
        start_receipt_path=arguments.start_receipt,
        exit_receipt_path=arguments.exit_receipt,
        command_sha256=arguments.command_sha256,
        require_session_leader=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MINIMUM_LOCAL_ARCHIVE_FREE_BYTES",
    "NvidiaSmiGpu",
    "OperatorTerminalContext",
    "ProductionArchiveRuntime",
    "ProductionCallbackError",
    "ProductionSchedulerRuntime",
    "canonical_json_bytes",
    "child_heartbeat_path",
    "child_start_receipt_path",
    "file_sha256",
    "linux_process_start_identity",
    "publish_atomic_terminal_result",
    "query_nvidia_smi",
    "regular_file_size",
    "revalidate_child_start_receipt",
    "run_child_wrapper",
    "statvfs_free_bytes",
]
