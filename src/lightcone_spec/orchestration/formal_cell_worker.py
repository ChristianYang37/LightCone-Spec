"""Source-owned bridge from one physical cell command to the durable operator.

The resident scheduler deliberately launches only immutable argv.  This module
is that argv for formal cells: it runs one already-prepared physical command,
deep-validates the resulting current-stage actual, seals every file below the
cell evidence root, emits a control JUnit, and only then publishes the atomic
operator terminal.  A successful child process alone is never sufficient for
``COMPLETE``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import threading
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from lightcone_spec.orchestration.experiment_operator import QueuedCommandSpec
from lightcone_spec.orchestration.experiment_operator_production import (
    OperatorTerminalContext,
    file_sha256,
    publish_atomic_terminal_result,
)

_MAX_SPEC_BYTES = 16 * 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_FAILURE_CLASSES = frozenset(
    {
        "INFRASTRUCTURE",
        "SCIENTIFIC",
        "UNSAFE",
        "OOM_CANDIDATE",
        "EXACTNESS",
        "FAILURE_DIAGNOSTIC",
    }
)
_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "cell_id",
        "attempt",
        "repository_root",
        "node_materialization_path",
        "actual_result_path",
        "evidence_root",
        "evidence_manifest_path",
        "job_argv",
        "failure_class_on_nonzero",
        "included_in_analysis_on_complete",
        "complete_exclusion_reason",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "cell_id",
        "attempt",
        "worker_spec_sha256",
        "evidence_root",
        "actual_result_path",
        "excluded_control_paths",
        "files",
    }
)
_MANIFEST_ROW_FIELDS = frozenset({"relative_path", "sha256", "size_bytes"})


class FormalCellWorkerError(RuntimeError):
    """Raised when a cell cannot be admitted as durable formal evidence."""


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


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise FormalCellWorkerError(f"{label} is not lowercase SHA-256")
    return value


def _absolute_path(value: object, label: str) -> Path:
    if type(value) is not str or not value:
        raise FormalCellWorkerError(f"{label} is empty")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise FormalCellWorkerError(f"{label} must be absolute and normalized")
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _stable_file(path: Path, *, maximum_bytes: int | None = None) -> bytes:
    if path.is_symlink():
        raise FormalCellWorkerError(f"evidence file cannot be a symlink: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise FormalCellWorkerError(f"evidence is not one regular file: {path}")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise FormalCellWorkerError(f"evidence file exceeds its schema: {path}")
        body = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            body.extend(chunk)
            if maximum_bytes is not None and len(body) > maximum_bytes:
                raise FormalCellWorkerError(f"evidence file exceeds its schema: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.stat(follow_symlinks=False)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    identity_current = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    if identity_before != identity_after or identity_after != identity_current:
        raise FormalCellWorkerError(f"evidence file changed while read: {path}")
    return bytes(body)


def _read_canonical_object(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    body = _stable_file(path, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalCellWorkerError(f"evidence is not UTF-8 JSON: {path}") from error
    if type(value) is not dict or body != _canonical_bytes(value):
        raise FormalCellWorkerError(f"evidence is not one canonical object: {path}")
    return value


def _atomic_write_new(path: Path, body: bytes) -> None:
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise FormalCellWorkerError("cell worker output path is not normalized")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FormalCellWorkerError(f"cell worker output already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
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
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as error:
        raise FormalCellWorkerError("cell worker output became occupied") from error
    finally:
        temporary.unlink(missing_ok=True)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _atomic_write_new_json(path: Path, value: object) -> None:
    _atomic_write_new(path, _canonical_bytes(value))


def _atomic_replace_json(path: Path, value: object) -> None:
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise FormalCellWorkerError("heartbeat path is not absolute and normalized")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FormalCellWorkerError("heartbeat path cannot be a symlink")
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
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
        body = _canonical_bytes(value)
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        if path.is_symlink():
            raise FormalCellWorkerError("heartbeat path became a symlink")
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


class ChildHeartbeatPublisher:
    """Publish worker-owned liveness without claiming scientific progress."""

    def __init__(
        self,
        *,
        path: Path,
        context: OperatorTerminalContext,
        clock_ns: Any,
        interval_seconds: float = 30.0,
    ) -> None:
        self.path = path
        self.context = context
        self.clock_ns = clock_ns
        self.interval_seconds = interval_seconds
        self.sequence = 0
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._publish("RUNNING")
        self._thread = threading.Thread(
            target=self._run,
            name=f"formal-cell-heartbeat-{self.context.cell_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self.interval_seconds))
            if thread.is_alive():
                raise FormalCellWorkerError("heartbeat publisher did not stop")
        if self._error is not None:
            raise FormalCellWorkerError("heartbeat publisher failed") from self._error
        self._publish("FINALIZING")

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval_seconds):
                self._publish("RUNNING")
        except BaseException as error:  # noqa: BLE001 - surfaced by stop()
            self._error = error
            self._stop.set()

    def _publish(self, phase: str) -> None:
        self.sequence += 1
        observed_at_ns = int(self.clock_ns())
        if observed_at_ns < 1:
            raise FormalCellWorkerError("heartbeat clock is invalid")
        _atomic_replace_json(
            self.path,
            {
                "schema_version": 1,
                "kind": "formal_experiment_child_heartbeat",
                "cell_id": self.context.cell_id,
                "attempt": self.context.attempt,
                "command_sha256": self.context.command_sha256,
                "worker_pid": os.getpid(),
                "sequence": self.sequence,
                "observed_at_ns": observed_at_ns,
                "phase": phase,
            },
        )


@dataclass(frozen=True)
class FormalCellWorkerSpec:
    schema_version: int
    kind: str
    cell_id: str
    attempt: int
    repository_root: str
    node_materialization_path: str
    actual_result_path: str
    evidence_root: str
    evidence_manifest_path: str
    job_argv: tuple[str, ...]
    failure_class_on_nonzero: str
    included_in_analysis_on_complete: bool
    complete_exclusion_reason: str | None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_cell_worker"
        ):
            raise FormalCellWorkerError("cell worker spec schema differs")
        if type(self.cell_id) is not str or not self.cell_id:
            raise FormalCellWorkerError("cell worker cell ID is empty")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int):
            raise FormalCellWorkerError("cell worker attempt must be an integer")
        if self.attempt < 1:
            raise FormalCellWorkerError("cell worker attempt must be positive")
        repository = _absolute_path(self.repository_root, "repository root")
        if not repository.is_dir() or repository.is_symlink():
            raise FormalCellWorkerError("cell worker repository is not a directory")
        node = _absolute_path(
            self.node_materialization_path,
            "node materialization path",
        )
        if not node.is_file() or node.is_symlink():
            raise FormalCellWorkerError("node materialization is not a regular file")
        evidence_root = _absolute_path(self.evidence_root, "evidence root")
        actual = _absolute_path(self.actual_result_path, "actual result path")
        manifest = _absolute_path(
            self.evidence_manifest_path,
            "evidence manifest path",
        )
        if not _is_within(actual, evidence_root) or not _is_within(
            manifest, evidence_root
        ):
            raise FormalCellWorkerError(
                "actual result and evidence manifest must stay below the cell root"
            )
        if actual == manifest:
            raise FormalCellWorkerError("actual result and manifest paths collide")
        if (
            type(self.job_argv) is not tuple
            or not self.job_argv
            or any(type(value) is not str or not value for value in self.job_argv)
        ):
            raise FormalCellWorkerError("cell worker job argv is not exact")
        executable = Path(self.job_argv[0])
        if not executable.is_absolute() or not executable.is_file():
            raise FormalCellWorkerError("cell worker executable must be absolute")
        if self.failure_class_on_nonzero not in _FAILURE_CLASSES:
            raise FormalCellWorkerError("cell worker nonzero failure class differs")
        if type(self.included_in_analysis_on_complete) is not bool:
            raise FormalCellWorkerError("cell worker inclusion flag is not boolean")
        if self.complete_exclusion_reason is not None and (
            type(self.complete_exclusion_reason) is not str
            or not self.complete_exclusion_reason
        ):
            raise FormalCellWorkerError("cell worker exclusion reason is empty")
        if self.included_in_analysis_on_complete == (
            self.complete_exclusion_reason is not None
        ):
            raise FormalCellWorkerError(
                "cell worker inclusion and exclusion reason are inconsistent"
            )

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "cell_id": self.cell_id,
            "attempt": self.attempt,
            "repository_root": self.repository_root,
            "node_materialization_path": self.node_materialization_path,
            "actual_result_path": self.actual_result_path,
            "evidence_root": self.evidence_root,
            "evidence_manifest_path": self.evidence_manifest_path,
            "job_argv": list(self.job_argv),
            "failure_class_on_nonzero": self.failure_class_on_nonzero,
            "included_in_analysis_on_complete": (self.included_in_analysis_on_complete),
            "complete_exclusion_reason": self.complete_exclusion_reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> FormalCellWorkerSpec:
        if type(value) is not dict or set(value) != _SPEC_FIELDS:
            raise FormalCellWorkerError("cell worker spec fields differ")
        row = dict(value)
        raw_argv = row.pop("job_argv")
        if type(raw_argv) is not list:
            raise FormalCellWorkerError("cell worker job argv must be a list")
        return cls(**row, job_argv=tuple(raw_argv))


def publish_formal_cell_worker_spec(
    spec: FormalCellWorkerSpec,
    output_path: str | Path,
) -> str:
    if type(spec) is not FormalCellWorkerSpec:
        raise TypeError("cell worker publisher requires an exact spec")
    path = _absolute_path(str(output_path), "cell worker spec output")
    _atomic_write_new_json(path, spec.to_dict())
    loaded, digest = load_formal_cell_worker_spec(path)
    if loaded != spec or digest != spec.sha256:
        raise RuntimeError("published cell worker spec changed")
    return digest


def load_formal_cell_worker_spec(
    path: str | Path,
) -> tuple[FormalCellWorkerSpec, str]:
    source = _absolute_path(str(path), "cell worker spec")
    raw = _read_canonical_object(source, maximum_bytes=_MAX_SPEC_BYTES)
    spec = FormalCellWorkerSpec.from_dict(raw)
    digest = hashlib.sha256(
        _stable_file(source, maximum_bytes=_MAX_SPEC_BYTES)
    ).hexdigest()
    if digest != spec.sha256:
        raise FormalCellWorkerError("cell worker spec raw and semantic SHA differ")
    return spec, digest


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
            "name": "formal_single_operator_cell_worker",
            "tests": "1",
            "failures": "0" if failure is None else "1",
            "errors": "0",
            "skipped": "0",
            "time": f"{elapsed_seconds:.9f}",
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {
            "classname": "formal_single_operator.cell_worker",
            "name": cell_id,
            "time": f"{elapsed_seconds:.9f}",
        },
    )
    if failure is not None:
        node = ET.SubElement(case, "failure", {"type": "FormalCellWorkerError"})
        node.text = failure
    body = ET.tostring(suite, encoding="utf-8", xml_declaration=True) + b"\n"
    _atomic_write_new(path, body)


def _control_exclusions(
    context: OperatorTerminalContext,
    spec: FormalCellWorkerSpec,
    environment: Mapping[str, str],
) -> tuple[Path, ...]:
    values = {
        Path(context.expected_terminal_path),
        Path(context.atomic_pointer_path),
        Path(spec.evidence_manifest_path),
    }
    for name in (
        "LIGHTCONE_OPERATOR_COMMAND_LOG_PATH",
        "LIGHTCONE_OPERATOR_CHILD_START_RECEIPT_PATH",
        "LIGHTCONE_OPERATOR_CHILD_EXIT_RECEIPT_PATH",
        "LIGHTCONE_OPERATOR_HEARTBEAT_PATH",
    ):
        raw = environment.get(name)
        if raw:
            values.add(_absolute_path(raw, name))
    return tuple(sorted(values, key=str))


def _build_evidence_manifest(
    spec: FormalCellWorkerSpec,
    *,
    spec_sha256: str,
    excluded_paths: tuple[Path, ...],
) -> dict[str, object]:
    root = Path(spec.evidence_root)
    if not root.is_dir() or root.is_symlink():
        raise FormalCellWorkerError("cell evidence root is not one directory")
    excluded = set(excluded_paths)
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise FormalCellWorkerError(f"cell evidence contains a symlink: {path}")
        if path.is_dir():
            continue
        if path in excluded:
            continue
        if not path.is_file():
            raise FormalCellWorkerError(f"cell evidence is not regular: {path}")
        relative = path.relative_to(root).as_posix()
        body = _stable_file(path)
        rows.append(
            {
                "relative_path": relative,
                "sha256": hashlib.sha256(body).hexdigest(),
                "size_bytes": len(body),
            }
        )
    required = {
        str(Path(spec.actual_result_path).relative_to(root).as_posix()),
    }
    observed = {str(row["relative_path"]) for row in rows}
    if Path(spec.actual_result_path).is_file() and not required <= observed:
        raise FormalCellWorkerError(
            "actual result is absent from the evidence manifest"
        )
    return {
        "schema_version": 1,
        "kind": "formal_single_operator_cell_evidence_manifest",
        "cell_id": spec.cell_id,
        "attempt": spec.attempt,
        "worker_spec_sha256": spec_sha256,
        "evidence_root": spec.evidence_root,
        "actual_result_path": spec.actual_result_path,
        "excluded_control_paths": [str(path) for path in excluded_paths],
        "files": rows,
    }


def _validate_manifest(
    value: object,
    *,
    spec: FormalCellWorkerSpec,
    spec_sha256: str,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != _MANIFEST_FIELDS:
        raise FormalCellWorkerError("cell evidence manifest fields differ")
    if (
        value["schema_version"] != 1
        or value["kind"] != "formal_single_operator_cell_evidence_manifest"
        or value["cell_id"] != spec.cell_id
        or value["attempt"] != spec.attempt
        or value["worker_spec_sha256"] != spec_sha256
        or value["evidence_root"] != spec.evidence_root
        or value["actual_result_path"] != spec.actual_result_path
        or type(value["excluded_control_paths"]) is not list
        or type(value["files"]) is not list
    ):
        raise FormalCellWorkerError("cell evidence manifest identity differs")
    root = Path(spec.evidence_root)
    evidence: dict[str, str] = {}
    prior: str | None = None
    for raw in value["files"]:
        if type(raw) is not dict or set(raw) != _MANIFEST_ROW_FIELDS:
            raise FormalCellWorkerError("cell evidence manifest row differs")
        relative = raw["relative_path"]
        if type(relative) is not str or not relative:
            raise FormalCellWorkerError("cell evidence relative path is empty")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or relative != pure.as_posix():
            raise FormalCellWorkerError("cell evidence relative path escapes its root")
        if prior is not None and relative <= prior:
            raise FormalCellWorkerError("cell evidence manifest is not uniquely sorted")
        prior = relative
        size = raw["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise FormalCellWorkerError("cell evidence size is invalid")
        digest = _require_sha256(raw["sha256"], "cell evidence SHA-256")
        path = root.joinpath(*pure.parts)
        body = _stable_file(path)
        if len(body) != size or hashlib.sha256(body).hexdigest() != digest:
            raise FormalCellWorkerError("cell evidence file identity differs")
        evidence[str(path)] = digest
    if (
        spec.actual_result_path not in evidence
        and Path(spec.actual_result_path).is_file()
    ):
        raise FormalCellWorkerError("actual result lacks a manifest row")
    return evidence


def _normalized_exit(value: int) -> int:
    return min(255, 128 + abs(value)) if value < 0 else min(255, value)


def _run_physical_command(
    spec: FormalCellWorkerSpec,
    *,
    environment: Mapping[str, str],
    raw_log: Any,
) -> tuple[subprocess.CompletedProcess[bytes], int | None]:
    """Run the job while forwarding operator TERM/INT to the direct child."""

    child = subprocess.Popen(
        list(spec.job_argv),
        cwd=spec.repository_root,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=raw_log,
        stderr=subprocess.STDOUT,
        shell=False,
        close_fds=True,
        start_new_session=False,
    )
    requested_signal: int | None = None
    prior_handlers: dict[int, Any] = {}

    def forward(signum: int, _frame: object) -> None:
        nonlocal requested_signal
        requested_signal = signum
        try:
            child.send_signal(signum)
        except ProcessLookupError:
            pass

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            prior_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward)
        return_code = child.wait()
    finally:
        for signum, prior in prior_handlers.items():
            signal.signal(signum, prior)
    return subprocess.CompletedProcess(
        list(spec.job_argv), return_code
    ), requested_signal


def _worker_argv_matches(command: QueuedCommandSpec, spec_path: str) -> bool:
    argv = command.argv
    return (
        len(argv) == 5
        and Path(argv[0]).is_absolute()
        and argv[1:4]
        == ("-m", "lightcone_spec.orchestration.formal_cell_worker", "--spec")
        and argv[4] == spec_path
    )


def run_formal_cell_worker(
    spec_path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    runner: Any = subprocess.run,
    clock_ns: Any = time.time_ns,
) -> int:
    """Execute, validate, and atomically terminalize one formal cell."""

    env = dict(os.environ if environment is None else environment)
    context = OperatorTerminalContext.from_environment(env)
    spec, spec_sha256 = load_formal_cell_worker_spec(spec_path)
    if (
        spec.cell_id != context.cell_id
        or spec.attempt != context.attempt
        or env.get("LIGHTCONE_CELL_WORKER_SPEC_SHA256") != spec_sha256
    ):
        raise FormalCellWorkerError("scheduler command and cell worker spec differ")
    root = Path(spec.evidence_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for raw in (
        context.expected_terminal_path,
        context.expected_junit_path,
        context.expected_raw_log_path,
        context.atomic_pointer_path,
    ):
        path = Path(raw)
        if not _is_within(path, root):
            raise FormalCellWorkerError(
                "operator output escapes the cell evidence root"
            )
        if path.exists() or path.is_symlink():
            raise FormalCellWorkerError("operator output path is already occupied")
    started_ns = int(clock_ns())
    if started_ns < 1:
        raise FormalCellWorkerError("cell worker clock is invalid")
    exit_code = 70
    status: Literal["COMPLETE", "FAILED"] = "FAILED"
    failure_class: str | None = "INFRASTRUCTURE"
    failure_code: str | None = "CELL_WORKER_EXCEPTION"
    exclusion_reason: str | None = "cell_worker_exception"
    included_in_analysis = False
    validation = None
    failure_text: str | None = None
    raw_path = Path(context.expected_raw_log_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_descriptor = os.open(
        raw_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    heartbeat_publisher: ChildHeartbeatPublisher | None = None
    heartbeat_raw = env.get("LIGHTCONE_OPERATOR_HEARTBEAT_PATH")
    if heartbeat_raw:
        heartbeat_publisher = ChildHeartbeatPublisher(
            path=_absolute_path(heartbeat_raw, "operator heartbeat path"),
            context=context,
            clock_ns=clock_ns,
        )
        heartbeat_publisher.start()
    try:
        with os.fdopen(raw_descriptor, "wb", buffering=0) as raw_log:
            try:
                operator_signal: int | None = None
                if runner is subprocess.run:
                    completed, operator_signal = _run_physical_command(
                        spec,
                        environment=env,
                        raw_log=raw_log,
                    )
                else:
                    completed = runner(
                        list(spec.job_argv),
                        cwd=spec.repository_root,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=raw_log,
                        stderr=subprocess.STDOUT,
                        shell=False,
                        close_fds=True,
                    )
                exit_code = _normalized_exit(int(completed.returncode))
                if operator_signal is not None:
                    failure_class = "INFRASTRUCTURE"
                    failure_code = f"OPERATOR_SIGNAL_{operator_signal}"
                    exclusion_reason = "operator_requested_termination"
                    failure_text = failure_code
                elif exit_code != 0:
                    failure_class = spec.failure_class_on_nonzero
                    failure_code = f"PHYSICAL_COMMAND_EXIT_{exit_code}"
                    exclusion_reason = "physical_command_nonzero"
                    failure_text = failure_code
                else:
                    from lightcone_spec.experiments.formal_single_operator_stages import (
                        validate_formal_single_operator_cell_actual,
                    )

                    validation = validate_formal_single_operator_cell_actual(
                        node_materialization_path=spec.node_materialization_path,
                        cell_id=spec.cell_id,
                        actual_result_path=spec.actual_result_path,
                        repository_root=spec.repository_root,
                    )
                    if validation.status != "COMPLETE":
                        raise FormalCellWorkerError(
                            "formal actual validator returned a non-complete status"
                        )
                    status = "COMPLETE"
                    failure_class = None
                    failure_code = None
                    exclusion_reason = spec.complete_exclusion_reason
                    included_in_analysis = spec.included_in_analysis_on_complete
            except BaseException as error:  # noqa: BLE001 - seal every worker failure
                if failure_text is None:
                    failure_text = f"{type(error).__name__}: {error}"
                    raw_log.write(
                        traceback.format_exc().encode("utf-8", errors="replace")
                    )
                if isinstance(error, (OSError, subprocess.SubprocessError)):
                    failure_class = "INFRASTRUCTURE"
                else:
                    failure_class = "SCIENTIFIC"
                failure_code = f"CELL_WORKER_{type(error).__name__.upper()}"
                exclusion_reason = "cell_worker_validation_failed"
                status = "FAILED"
                included_in_analysis = False
                exit_code = 70
    finally:
        if heartbeat_publisher is not None:
            heartbeat_publisher.stop()
    finished_job_ns = max(int(clock_ns()), started_ns + 1)
    _write_junit(
        Path(context.expected_junit_path),
        cell_id=spec.cell_id,
        elapsed_seconds=(finished_job_ns - started_ns) / 1e9,
        failure=failure_text,
    )
    excluded = _control_exclusions(context, spec, env)
    manifest = _build_evidence_manifest(
        spec,
        spec_sha256=spec_sha256,
        excluded_paths=excluded,
    )
    manifest_path = Path(spec.evidence_manifest_path)
    _atomic_write_new_json(manifest_path, manifest)
    evidence = _validate_manifest(manifest, spec=spec, spec_sha256=spec_sha256)
    manifest_sha256 = file_sha256(manifest_path)
    actual_path = Path(spec.actual_result_path)
    actual_sha256 = file_sha256(actual_path) if actual_path.is_file() else None
    metadata = {
        "worker_spec_path": str(_absolute_path(str(spec_path), "worker spec")),
        "worker_spec_sha256": spec_sha256,
        "node_materialization_path": spec.node_materialization_path,
        "actual_result_path": spec.actual_result_path,
        "actual_result_raw_sha256": actual_sha256,
        "result_identity_sha256": (
            None if validation is None else validation.result_identity_sha256
        ),
        "validator_kind": None if validation is None else validation.validator_kind,
        "validator_protocol_sha256": (
            None if validation is None else validation.validator_protocol_sha256
        ),
        "evidence_manifest_path": spec.evidence_manifest_path,
        "evidence_manifest_sha256": manifest_sha256,
    }
    if status == "COMPLETE" and spec.actual_result_path not in evidence:
        raise FormalCellWorkerError("complete actual is absent from sealed evidence")
    finished_ns = max(int(clock_ns()), finished_job_ns + 1)
    publish_atomic_terminal_result(
        context,
        status=status,
        exit_code=exit_code,
        started_ns=started_ns,
        finished_ns=finished_ns,
        failure_class=failure_class,
        failure_code=failure_code,
        exclusion_reason=exclusion_reason,
        included_in_analysis=included_in_analysis,
        validation_metadata=metadata,
    )
    return exit_code


def revalidate_formal_cell_worker_terminal(
    terminal: Mapping[str, object],
    *,
    command: QueuedCommandSpec,
) -> dict[str, str]:
    """Deep-replay schema-2 worker evidence after the process has exited."""

    if type(command) is not QueuedCommandSpec:
        raise TypeError("cell worker terminal requires an exact queued command")
    spec_path = _absolute_path(terminal.get("worker_spec_path"), "worker spec path")
    spec, spec_sha256 = load_formal_cell_worker_spec(spec_path)
    declared_spec_sha = _require_sha256(
        terminal.get("worker_spec_sha256"),
        "worker spec terminal SHA-256",
    )
    environment = dict(command.environment)
    if (
        declared_spec_sha != spec_sha256
        or environment.get("LIGHTCONE_CELL_WORKER_SPEC_SHA256") != spec_sha256
        or not _worker_argv_matches(command, str(spec_path))
        or spec.cell_id != command.cell_id
        or spec.attempt != command.attempt
        or terminal.get("node_materialization_path") != spec.node_materialization_path
        or terminal.get("actual_result_path") != spec.actual_result_path
        or terminal.get("evidence_manifest_path") != spec.evidence_manifest_path
    ):
        raise FormalCellWorkerError("cell worker terminal command/spec lineage differs")
    manifest_path = Path(spec.evidence_manifest_path)
    manifest_sha = file_sha256(manifest_path)
    if manifest_sha != _require_sha256(
        terminal.get("evidence_manifest_sha256"),
        "cell evidence manifest SHA-256",
    ):
        raise FormalCellWorkerError("cell evidence manifest digest differs")
    manifest = _read_canonical_object(
        manifest_path,
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    evidence = _validate_manifest(
        manifest,
        spec=spec,
        spec_sha256=spec_sha256,
    )
    evidence[str(manifest_path)] = manifest_sha
    actual_sha = terminal.get("actual_result_raw_sha256")
    if actual_sha is not None:
        _require_sha256(actual_sha, "cell actual raw SHA-256")
        if file_sha256(spec.actual_result_path) != actual_sha:
            raise FormalCellWorkerError("cell actual raw SHA-256 differs")
    if terminal.get("status") == "COMPLETE":
        from lightcone_spec.experiments.formal_single_operator_stages import (
            validate_formal_single_operator_cell_actual,
        )

        if actual_sha is None:
            raise FormalCellWorkerError("complete cell lacks an actual result digest")
        validation = validate_formal_single_operator_cell_actual(
            node_materialization_path=spec.node_materialization_path,
            cell_id=spec.cell_id,
            actual_result_path=spec.actual_result_path,
            repository_root=spec.repository_root,
        )
        if (
            validation.status != "COMPLETE"
            or terminal.get("result_identity_sha256")
            != validation.result_identity_sha256
            or terminal.get("validator_kind") != validation.validator_kind
            or terminal.get("validator_protocol_sha256")
            != validation.validator_protocol_sha256
        ):
            raise FormalCellWorkerError("cell actual validation identity differs")
    elif any(
        terminal.get(name) is not None
        for name in (
            "result_identity_sha256",
            "validator_kind",
            "validator_protocol_sha256",
        )
    ):
        raise FormalCellWorkerError("failed cell claims a successful validation")
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return run_formal_cell_worker(arguments.spec)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ChildHeartbeatPublisher",
    "FormalCellWorkerError",
    "FormalCellWorkerSpec",
    "load_formal_cell_worker_spec",
    "publish_formal_cell_worker_spec",
    "revalidate_formal_cell_worker_terminal",
    "run_formal_cell_worker",
]
