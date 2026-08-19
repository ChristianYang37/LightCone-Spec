"""Publish the remote safe-boundary observation required for AutoDL power-off."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from lightcone_spec.orchestration.experiment_operator import (
    ExperimentOperatorStore,
    SingletonOperatorLock,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    canonical_json_bytes,
)
from lightcone_spec.runtime.proof_artifact import publish_canonical_json_no_replace

FORMAL_SHUTDOWN_PROBE_PROTOCOL_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "kind": "formal_shutdown_probe_protocol",
            "schema_version": 1,
            "checks": [
                "sqlite_dispatch_stop_before_and_after",
                "sqlite_zero_running_attempts_before_and_after",
                "proc_zero_writable_run_root_processes_before_and_after",
                "nvidia_smi_zero_compute_processes_before_and_after",
                "proc_net_zero_registered_listen_ports_before_and_after",
                "run_root_zero_file_size_or_membership_change",
            ],
        }
    )
).hexdigest()


class FormalShutdownProbeError(RuntimeError):
    """The shutdown probe could not be observed faithfully."""


def collect_formal_shutdown_probe(
    *,
    database_path: str | Path,
    instance_uuid: str,
    run_root: str | Path,
    measurement_ports: tuple[int, ...],
    observation_window_seconds: int = 5,
    proc_root: str | Path = "/proc",
    command_runner: Callable[..., Any] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    clock_ns: Callable[[], int] = time.time_ns,
    self_pid: int | None = None,
    readonly_database: bool = False,
) -> dict[str, object]:
    """Observe a stable idle interval; nonzero rows remain publishable evidence."""

    database = _absolute_path(database_path, "operator database")
    root = _absolute_path(run_root, "formal run root")
    proc = _absolute_path(proc_root, "proc root")
    if root.is_symlink() or not root.is_dir():
        raise FormalShutdownProbeError("formal run root is not a safe directory")
    if proc.is_symlink() or not proc.is_dir():
        raise FormalShutdownProbeError("proc root is not a safe directory")
    if (
        not instance_uuid.startswith("pro-")
        or isinstance(observation_window_seconds, bool)
        or not isinstance(observation_window_seconds, int)
        or not 5 <= observation_window_seconds <= 60
        or measurement_ports != tuple(sorted(set(measurement_ports)))
        or any(
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
            for port in measurement_ports
        )
    ):
        raise FormalShutdownProbeError("shutdown probe inputs differ")
    excluded_pid = os.getpid() if self_pid is None else self_pid
    store_reader = _store_state_readonly if readonly_database else _store_state
    before_store = store_reader(database)
    before_files = _file_sizes(root)
    before_writers = _writable_run_root_processes(
        proc,
        root,
        excluded_pid=excluded_pid,
    )
    before_gpu = _gpu_compute_processes(command_runner)
    before_ports = _listening_measurement_ports(proc, measurement_ports)
    sleeper(float(observation_window_seconds))
    after_store = store_reader(database)
    after_files = _file_sizes(root)
    after_writers = _writable_run_root_processes(
        proc,
        root,
        excluded_pid=excluded_pid,
    )
    after_gpu = _gpu_compute_processes(command_runner)
    after_ports = _listening_measurement_ports(proc, measurement_ports)
    log_growth_bytes = _file_change_bytes(before_files, after_files)
    command_identity = {
        "protocol_sha256": FORMAL_SHUTDOWN_PROBE_PROTOCOL_SHA256,
        "database_path": str(database),
        "instance_uuid": instance_uuid,
        "run_root": str(root),
        "measurement_ports": list(measurement_ports),
        "observation_window_seconds": observation_window_seconds,
    }
    return {
        "schema_version": 1,
        "kind": "autodl_power_off_safety_probe",
        "instance_uuid": instance_uuid,
        "run_id": before_store[0],
        "observed_at_ns": int(clock_ns()),
        "observation_window_seconds": observation_window_seconds,
        "scheduler_control_state": (
            "STOP" if before_store[1] == after_store[1] == "STOP" else "UNSAFE"
        ),
        "running_attempt_count": max(before_store[2], after_store[2]),
        "evidence_writer_process_count": len(before_writers | after_writers),
        "gpu_compute_process_count": len(before_gpu | after_gpu),
        "open_measurement_port_count": len(before_ports | after_ports),
        "log_growth_bytes": log_growth_bytes,
        "probe_command_sha256": hashlib.sha256(
            canonical_json_bytes(command_identity)
        ).hexdigest(),
    }


def shutdown_probe_is_safe(value: dict[str, object]) -> bool:
    return value.get("scheduler_control_state") == "STOP" and all(
        value.get(field) == 0
        for field in (
            "running_attempt_count",
            "evidence_writer_process_count",
            "gpu_compute_process_count",
            "open_measurement_port_count",
            "log_growth_bytes",
        )
    )


def _store_state(database: Path) -> tuple[str, str, int]:
    with ExperimentOperatorStore(database) as store:
        state, _reason = store.dispatch_control()
        running = sum(
            row["status"] == "RUNNING" for row in store.snapshot()["attempts"]
        )
        return store.run_id, state, running


def _store_state_readonly(database: Path) -> tuple[str, str, int]:
    """Read the shutdown guard without opening the write-capable store.

    Cross-host finalization calls this after publishing the scientific-closure
    receipt.  At that point even an idempotent schema/PRAGMA write would violate
    the promise that the remote SQLite authority is permanently closed.
    """

    if database.is_symlink() or not database.is_file():
        raise FormalShutdownProbeError("operator database is not a regular file")
    connection = sqlite3.connect(
        f"file:{database.as_posix()}?mode=ro",
        uri=True,
        isolation_level=None,
    )
    try:
        rows = dict(
            connection.execute(
                "SELECT key, value FROM operator_meta "
                "WHERE key IN ('run_id', 'dispatch_state')"
            ).fetchall()
        )
        run_id = rows.get("run_id")
        state = rows.get("dispatch_state", "RUN")
        running = int(
            connection.execute(
                "SELECT COUNT(*) FROM cell_attempts WHERE status = 'RUNNING'"
            ).fetchone()[0]
        )
        running += int(
            connection.execute(
                "SELECT COUNT(*) FROM controller_auxiliary_groups "
                "WHERE status = 'RUNNING'"
            ).fetchone()[0]
        )
    except sqlite3.Error as error:
        raise FormalShutdownProbeError(
            "read-only operator shutdown guard is unreadable"
        ) from error
    finally:
        connection.close()
    if type(run_id) is not str or not run_id or state not in {"RUN", "STOP"}:
        raise FormalShutdownProbeError("read-only operator shutdown guard differs")
    return run_id, state, running


def _file_sizes(root: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    for directory, names, filenames in os.walk(root, followlinks=False):
        names[:] = sorted(
            name for name in names if not Path(directory, name).is_symlink()
        )
        for name in sorted(filenames):
            path = Path(directory, name)
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            rows[relative] = path.stat(follow_symlinks=False).st_size
    return rows


def _file_change_bytes(before: dict[str, int], after: dict[str, int]) -> int:
    return sum(
        abs(after.get(path, 0) - before.get(path, 0))
        for path in set(before) | set(after)
    )


def _writable_run_root_processes(
    proc_root: Path,
    run_root: Path,
    *,
    excluded_pid: int,
) -> set[int]:
    output: set[int] = set()
    prefix = str(run_root).rstrip("/") + "/"
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) == excluded_pid:
            continue
        descriptor_root = process / "fd"
        try:
            descriptors = tuple(descriptor_root.iterdir())
        except (FileNotFoundError, PermissionError, NotADirectoryError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor).removesuffix(" (deleted)")
                fdinfo = (process / "fdinfo" / descriptor.name).read_text(
                    encoding="utf-8"
                )
            except (FileNotFoundError, PermissionError, OSError, UnicodeError):
                continue
            flags = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in fdinfo.splitlines()
                    if line.startswith("flags:")
                ),
                None,
            )
            if flags is None:
                continue
            try:
                writable = int(flags, 8) & os.O_ACCMODE in {os.O_WRONLY, os.O_RDWR}
            except ValueError:
                continue
            if writable and (target == str(run_root) or target.startswith(prefix)):
                output.add(int(process.name))
                break
    return output


def _gpu_compute_processes(runner: Callable[..., Any]) -> set[tuple[str, int]]:
    completed = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise FormalShutdownProbeError("nvidia-smi compute query failed")
    output: set[tuple[str, int]] = set()
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = tuple(part.strip() for part in line.split(","))
        if len(fields) != 2 or not fields[0].startswith("GPU-"):
            raise FormalShutdownProbeError("nvidia-smi compute row differs")
        try:
            pid = int(fields[1])
        except ValueError as error:
            raise FormalShutdownProbeError("nvidia-smi compute PID differs") from error
        if pid < 1:
            raise FormalShutdownProbeError("nvidia-smi compute PID differs")
        output.add((fields[0], pid))
    return output


def _listening_measurement_ports(
    proc_root: Path,
    ports: tuple[int, ...],
) -> set[int]:
    registered = set(ports)
    listening: set[int] = set()
    for name in ("tcp", "tcp6"):
        path = proc_root / "net" / name
        try:
            lines = path.read_text(encoding="ascii").splitlines()[1:]
        except (FileNotFoundError, PermissionError, UnicodeError) as error:
            raise FormalShutdownProbeError(
                "proc network table is unreadable"
            ) from error
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":
                continue
            try:
                port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError) as error:
                raise FormalShutdownProbeError("proc listen row differs") from error
            if port in registered:
                listening.add(port)
    return listening


def _absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise FormalShutdownProbeError(f"{label} must be absolute and normalized")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--instance-uuid", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--measurement-port", type=int, action="append", default=[])
    parser.add_argument("--observation-window-seconds", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = _absolute_path(arguments.output, "shutdown probe output")
    with SingletonOperatorLock(arguments.lock):
        probe = collect_formal_shutdown_probe(
            database_path=arguments.db,
            instance_uuid=arguments.instance_uuid,
            run_root=arguments.run_root,
            measurement_ports=tuple(sorted(arguments.measurement_port)),
            observation_window_seconds=arguments.observation_window_seconds,
        )
        publish_canonical_json_no_replace(output, probe)
    print(json.dumps(probe, sort_keys=True))
    return 0 if shutdown_probe_is_safe(probe) else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAL_SHUTDOWN_PROBE_PROTOCOL_SHA256",
    "FormalShutdownProbeError",
    "collect_formal_shutdown_probe",
    "shutdown_probe_is_safe",
]
