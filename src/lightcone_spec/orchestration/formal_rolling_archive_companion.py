"""Non-LLM rolling companion for the 21-node formal v03 DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from lightcone_spec.orchestration.experiment_operator import (
    REMOTE_SPOOL_SAFETY_RESERVE_BYTES,
    ArchiveRequest,
    RemoteEvictionAuthorization,
    SingletonOperatorLock,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    MINIMUM_LOCAL_ARCHIVE_FREE_BYTES,
    canonical_json_bytes,
)
from lightcone_spec.orchestration.formal_remote_archive import (
    FormalRemoteArchiveError,
    RemoteArchiveResult,
    SshRemoteOperatorClient,
    SshRsyncArchiveEndpoint,
    load_remote_archive_result,
    run_remote_archive,
)
from lightcone_spec.orchestration.formal_rolling_archive import (
    RemoteEvictionPlan,
    RemoteEvictionReceipt,
    load_formal_archive_sha256_manifest,
    load_remote_eviction_plan,
    load_remote_eviction_receipt,
    remote_eviction_authorization_sha256,
)

_INDEX_KIND = "formal_rolling_archive_companion_index"
_RESTORE_INDEX_KIND = "formal_rolling_archive_companion_restore_index"
_EVENT_KIND = "formal_rolling_archive_companion_event"
_REMOTE_ARGUMENT = re.compile(r"^[A-Za-z0-9_./:@+-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FormalRollingArchiveCompanionError(RuntimeError):
    """The local rolling companion failed closed."""


@dataclass(frozen=True)
class RollingArchiveCompanionConfig:
    endpoint: SshRsyncArchiveEndpoint
    remote_run_root: str
    local_results_root: str
    state_root: str
    lock_path: str
    poll_interval_seconds: Literal[30] = 30
    minimum_local_free_bytes: int = MINIMUM_LOCAL_ARCHIVE_FREE_BYTES
    minimum_remote_restore_free_bytes: int = REMOTE_SPOOL_SAFETY_RESERVE_BYTES

    def __post_init__(self) -> None:
        if type(self.endpoint) is not SshRsyncArchiveEndpoint:
            raise TypeError("rolling companion requires an exact SSH endpoint")
        remote = _absolute(self.remote_run_root, "remote v03 run root")
        results = _absolute(self.local_results_root, "local results root")
        state = _absolute(self.state_root, "rolling companion state root")
        lock = _absolute(self.lock_path, "rolling companion lock")
        if (
            not _has_v03_marker(remote)
            or results.name != "results"
            or state == results
            or state.is_relative_to(results)
            or results.is_relative_to(state)
            or lock.is_relative_to(remote)
            or self.poll_interval_seconds != 30
        ):
            raise ValueError("rolling companion path or polling identity differs")
        if (
            isinstance(self.minimum_local_free_bytes, bool)
            or not isinstance(self.minimum_local_free_bytes, int)
            or self.minimum_local_free_bytes < 0
        ):
            raise ValueError("minimum local free bytes must be non-negative")
        if (
            isinstance(self.minimum_remote_restore_free_bytes, bool)
            or not isinstance(self.minimum_remote_restore_free_bytes, int)
            or self.minimum_remote_restore_free_bytes < 0
        ):
            raise ValueError("minimum remote restore free bytes must be non-negative")

    @property
    def run_id(self) -> str:
        return Path(self.remote_run_root).name


@dataclass(frozen=True)
class CompanionNodeIndex:
    schema_version: Literal[1]
    kind: Literal["formal_rolling_archive_companion_index"]
    run_id: str
    node: str
    ordinal: int
    request_sha256: str
    local_archive_result_path: str
    local_archive_result_sha256: str
    local_plan_path: str
    plan_sha256: str
    local_eviction_receipt_path: str
    eviction_receipt_sha256: str
    remote_plan_path: str
    remote_eviction_receipt_path: str
    archived_at_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _INDEX_KIND
            or not self.run_id
            or not self.node
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
            or not isinstance(self.archived_at_ns, int)
            or self.archived_at_ns <= 0
        ):
            raise ValueError("rolling companion index identity differs")
        for label, value in (
            ("request", self.request_sha256),
            ("archive result", self.local_archive_result_sha256),
            ("plan", self.plan_sha256),
            ("eviction receipt", self.eviction_receipt_sha256),
        ):
            _sha(value, f"{label} SHA-256")
        for label, value in (
            ("local result", self.local_archive_result_path),
            ("local plan", self.local_plan_path),
            ("local eviction receipt", self.local_eviction_receipt_path),
            ("remote plan", self.remote_plan_path),
            ("remote eviction receipt", self.remote_eviction_receipt_path),
        ):
            _absolute(value, f"{label} path")

    @property
    def sha256(self) -> str:
        return _semantic_sha(asdict(self))


@dataclass(frozen=True)
class CompanionRestoreIndex:
    schema_version: Literal[1]
    kind: Literal["formal_rolling_archive_companion_restore_index"]
    run_id: str
    node: str
    ordinal: int
    archive_index_sha256: str
    restore_receipt_sha256: str
    restored_at_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _RESTORE_INDEX_KIND
            or not self.run_id
            or not self.node
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
            or not isinstance(self.restored_at_ns, int)
            or self.restored_at_ns <= 0
        ):
            raise ValueError("rolling companion restore index identity differs")
        _sha(self.archive_index_sha256, "archive index SHA-256")
        _sha(self.restore_receipt_sha256, "restore receipt SHA-256")

    @property
    def sha256(self) -> str:
        return _semantic_sha(asdict(self))


@dataclass(frozen=True)
class CompanionEvent:
    schema_version: Literal[1]
    kind: Literal["formal_rolling_archive_companion_event"]
    action: Literal["IDLE", "ARCHIVED", "FAILED", "RESTORED"]
    run_id: str
    node: str | None
    detail: str
    changed: bool
    occurred_at_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _EVENT_KIND
            or self.action not in {"IDLE", "ARCHIVED", "FAILED", "RESTORED"}
            or not self.run_id
            or not self.detail
            or not isinstance(self.changed, bool)
            or not isinstance(self.occurred_at_ns, int)
            or self.occurred_at_ns <= 0
            or (self.action == "IDLE" and self.node is not None)
            or (self.action != "IDLE" and not self.node)
        ):
            raise ValueError("rolling companion event identity differs")


class RollingArchiveTransport(Protocol):
    def probe(self, *, node: str, ordinal: int) -> Mapping[str, Any]: ...

    def prepare(self, *, node: str, ordinal: int) -> ArchiveRequest: ...

    def archive(
        self,
        *,
        node: str,
        ordinal: int,
        request: ArchiveRequest,
        result_path: Path,
        lock_path: Path,
    ) -> RemoteArchiveResult: ...

    def evict(
        self,
        *,
        node: str,
        ordinal: int,
        request: ArchiveRequest,
        result: RemoteArchiveResult,
    ) -> Mapping[str, Any]: ...

    def restore(self, index: CompanionNodeIndex) -> Mapping[str, Any]: ...

    def stop(self, reason: str) -> None: ...


class SshRollingArchiveTransport:
    """Production SSH/rsync transport; no credential value enters artifacts."""

    def __init__(
        self,
        config: RollingArchiveCompanionConfig,
        *,
        ssh_runner: Any = subprocess.run,
        rsync_runner: Any = subprocess.run,
    ) -> None:
        self.config = config
        self.ssh_runner = ssh_runner
        self.rsync_runner = rsync_runner
        self.operator = SshRemoteOperatorClient(
            config.endpoint,
            runner=ssh_runner,
        )

    def probe(self, *, node: str, ordinal: int) -> Mapping[str, Any]:
        return self._call(
            "probe-node",
            (
                "--run-root",
                self.config.remote_run_root,
                "--node",
                node,
                "--ordinal",
                str(ordinal),
            ),
        )

    def prepare(self, *, node: str, ordinal: int) -> ArchiveRequest:
        paths = self._remote_paths(node, ordinal)
        value = self._call(
            "prepare-node",
            (
                "--run-root",
                self.config.remote_run_root,
                "--retained-manifest",
                paths["retained"],
                "--local-results-root",
                self.config.local_results_root,
                "--wave",
                f"wave-{ordinal:02d}",
                "--request-output",
                paths["request"],
                "--lock",
                paths["prepare_lock"],
            ),
        )
        return ArchiveRequest(**value)

    def archive(
        self,
        *,
        node: str,
        ordinal: int,
        request: ArchiveRequest,
        result_path: Path,
        lock_path: Path,
    ) -> RemoteArchiveResult:
        del node, ordinal
        return run_remote_archive(
            endpoint=self.config.endpoint,
            request=request,
            result_output_path=result_path,
            local_lock_path=lock_path,
            minimum_local_free_bytes=self.config.minimum_local_free_bytes,
            ssh_runner=self.ssh_runner,
            rsync_runner=self.rsync_runner,
        )

    def evict(
        self,
        *,
        node: str,
        ordinal: int,
        request: ArchiveRequest,
        result: RemoteArchiveResult,
    ) -> Mapping[str, Any]:
        paths = self._remote_paths(node, ordinal)
        authorization_value = self.operator.call(
            "archive-authorize",
            ("--archive-id", request.archive_id),
        )
        authorization_row = dict(authorization_value)
        authorization_row.pop("remote_deletion_performed", None)
        authorization = RemoteEvictionAuthorization(**authorization_row)
        authorization_sha = remote_eviction_authorization_sha256(authorization)
        self.stop("rolling_archive_eviction_boundary")
        self._call(
            "stage-chain",
            (
                "--request",
                paths["request"],
                "--input",
                "-",
                "--result-output",
                paths["result"],
                "--authorization-output",
                paths["authorization"],
                "--lock",
                paths["stage_lock"],
            ),
            stdin_object={
                "result": {**asdict(result), "result_sha256": result.sha256},
                "authorization": {
                    **asdict(authorization),
                    "authorization_sha256": authorization_sha,
                    "remote_deletion_performed": False,
                },
            },
        )
        return self._call(
            "evict-staged",
            (
                "--request",
                paths["request"],
                "--archive-result",
                paths["result"],
                "--authorization",
                paths["authorization"],
                "--retained-manifest",
                paths["retained"],
                "--operator-db",
                self.config.endpoint.remote_operator_database,
                "--plan-output",
                paths["plan"],
                "--receipt-output",
                paths["receipt"],
                "--plan-lock",
                paths["plan_lock"],
                "--executor-lock",
                paths["executor_lock"],
            ),
        )

    def restore(self, index: CompanionNodeIndex) -> Mapping[str, Any]:
        if type(index) is not CompanionNodeIndex:
            raise TypeError("SSH restore requires an exact companion index")
        _validate_node_index_scope(self.config, index)
        plan = load_remote_eviction_plan(index.local_plan_path)
        eviction = load_remote_eviction_receipt(index.local_eviction_receipt_path)
        result = load_remote_archive_result(index.local_archive_result_path)
        planned = {
            (
                row.absolute_path,
                row.archive_relative_path,
                row.size_bytes,
                row.sha256,
            )
            for row in plan.files
        }
        deleted = {
            (
                row.absolute_path,
                row.archive_relative_path,
                row.size_bytes,
                row.sha256,
            )
            for row in eviction.deleted_files
        }
        expected_final = (
            Path(self.config.local_results_root)
            / self.config.run_id
            / index.node
            / f"wave-{index.ordinal:02d}.final"
        )
        if (
            plan.sha256 != index.plan_sha256
            or eviction.sha256 != index.eviction_receipt_sha256
            or result.sha256 != index.local_archive_result_sha256
            or plan.run_id != index.run_id
            or plan.run_root != self.config.remote_run_root
            or plan.node != index.node
            or eviction.status != "COMPLETE"
            or eviction.plan_sha256 != plan.sha256
            or eviction.archive_id != plan.archive_id
            or eviction.archive_authorization_sha256
            != plan.archive_authorization_sha256
            or deleted != planned
            or result.archive_id != plan.archive_id
            or result.remote_payload_root != plan.archive_candidate_root
            or result.manifest_sha256 != plan.archive_manifest_sha256
            or result.local_final_root != str(expected_final)
            or result.ssh_target != self.config.endpoint.ssh_target
            or result.ssh_port != self.config.endpoint.ssh_port
            or result.remote_deletion_performed is not False
        ):
            raise FormalRollingArchiveCompanionError(
                "local restore index authorities differ"
            )
        final_root = Path(result.local_final_root)
        manifest = load_formal_archive_sha256_manifest(
            final_root / "sha256_manifest.json",
            verify_root=True,
        )
        if manifest.sha256 != plan.archive_manifest_sha256:
            raise FormalRollingArchiveCompanionError(
                "local final archive manifest differs from plan"
            )
        if (
            result.checked_file_count != len(manifest.files)
            or result.checked_bytes != manifest.payload_bytes
        ):
            raise FormalRollingArchiveCompanionError(
                "local final archive coverage differs from result"
            )
        rows = {row.path: row for row in manifest.files}
        control = Path(index.remote_plan_path).parent
        if Path(index.remote_eviction_receipt_path).parent != control:
            raise FormalRollingArchiveCompanionError(
                "remote restore control paths diverge"
            )
        progress_root = control / "stream-restore-progress"
        for file_index, binding in enumerate(plan.files):
            row = rows.get(binding.archive_relative_path)
            if (
                row is None
                or row.size_bytes != binding.size_bytes
                or row.sha256 != binding.sha256
            ):
                raise FormalRollingArchiveCompanionError(
                    "planned restore member differs from local final manifest"
                )
            pure = PurePosixPath(binding.archive_relative_path)
            source = final_root.joinpath(*pure.parts)
            if (
                not source.is_relative_to(final_root)
                or source.is_symlink()
                or not source.is_file()
            ):
                raise FormalRollingArchiveCompanionError(
                    "local restore member path is unsafe"
                )
            self._call_stream(
                "restore-member",
                (
                    "--plan",
                    index.remote_plan_path,
                    "--eviction-receipt",
                    index.remote_eviction_receipt_path,
                    "--archive-result",
                    str(control / "remote-archive-result.json"),
                    "--relative-path",
                    binding.archive_relative_path,
                    "--progress-output",
                    str(progress_root / f"{file_index:08d}.json"),
                    "--operator-db",
                    self.config.endpoint.remote_operator_database,
                    "--lock",
                    str(control / "stream-restore.lock"),
                    "--minimum-free-bytes",
                    str(self.config.minimum_remote_restore_free_bytes),
                ),
                source=source,
            )
        response = self._call(
            "finalize-stream-restore",
            (
                "--plan",
                index.remote_plan_path,
                "--eviction-receipt",
                index.remote_eviction_receipt_path,
                "--archive-result",
                str(control / "remote-archive-result.json"),
                "--progress-root",
                str(progress_root),
                "--receipt-output",
                str(control / "remote-restore-receipt.json"),
                "--operator-db",
                self.config.endpoint.remote_operator_database,
                "--lock",
                str(control / "stream-restore-finalize.lock"),
            ),
        )
        receipt_sha = response.get("receipt_sha256")
        if type(receipt_sha) is not str or _SHA256.fullmatch(receipt_sha) is None:
            raise FormalRollingArchiveCompanionError(
                "remote streamed restore receipt digest is invalid"
            )
        return response

    def stop(self, reason: str) -> None:
        self.operator.call("scheduler-stop", ("--reason", reason))

    def _call(
        self,
        operation: str,
        arguments: tuple[str, ...],
        *,
        stdin_object: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if _REMOTE_ARGUMENT.fullmatch(operation) is None or any(
            _REMOTE_ARGUMENT.fullmatch(value) is None for value in arguments
        ):
            raise ValueError("rolling companion remote argument is not canonical")
        remote_argv = (
            self.config.endpoint.remote_python,
            "-m",
            "lightcone_spec.orchestration.formal_rolling_archive",
            operation,
            *arguments,
        )
        completed = self.ssh_runner(
            [
                *self.config.endpoint.ssh_transport_argv,
                self.config.endpoint.ssh_target,
                shlex.join(remote_argv),
            ],
            input=(
                None
                if stdin_object is None
                else canonical_json_bytes(dict(stdin_object))
            ),
            check=False,
            capture_output=True,
            shell=False,
        )
        if completed.returncode != 0:
            raise FormalRollingArchiveCompanionError(
                f"remote rolling command failed with exit {completed.returncode}"
            )
        try:
            value = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FormalRollingArchiveCompanionError(
                "remote rolling command returned invalid JSON"
            ) from error
        if type(value) is not dict:
            raise FormalRollingArchiveCompanionError(
                "remote rolling command returned a non-object"
            )
        return value

    def _call_stream(
        self,
        operation: str,
        arguments: tuple[str, ...],
        *,
        source: Path,
    ) -> dict[str, Any]:
        if _REMOTE_ARGUMENT.fullmatch(operation) is None or any(
            _REMOTE_ARGUMENT.fullmatch(value) is None for value in arguments
        ):
            raise ValueError("rolling companion stream argument is not canonical")
        remote_argv = (
            self.config.endpoint.remote_python,
            "-m",
            "lightcone_spec.orchestration.formal_rolling_archive",
            operation,
            *arguments,
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(source, flags)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                completed = self.ssh_runner(
                    [
                        *self.config.endpoint.ssh_transport_argv,
                        self.config.endpoint.ssh_target,
                        shlex.join(remote_argv),
                    ],
                    stdin=handle,
                    check=False,
                    capture_output=True,
                    shell=False,
                )
        finally:
            os.close(descriptor)
        if completed.returncode != 0:
            raise FormalRollingArchiveCompanionError(
                f"remote streamed restore failed with exit {completed.returncode}"
            )
        try:
            value = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FormalRollingArchiveCompanionError(
                "remote streamed restore returned invalid JSON"
            ) from error
        if type(value) is not dict:
            raise FormalRollingArchiveCompanionError(
                "remote streamed restore returned a non-object"
            )
        return value

    def _remote_paths(self, node: str, ordinal: int) -> dict[str, str]:
        node_root = (
            Path(self.config.remote_run_root)
            / "formal-dag-nodes"
            / f"{ordinal:02d}-{node}"
        )
        control = (
            Path(self.config.remote_run_root)
            / "rolling-archive-control"
            / f"{ordinal:02d}-{node}"
        )
        return {
            "retained": str(
                node_root / "reduction" / "retained-future-dependency-manifest.json"
            ),
            "request": str(control / "archive-request.json"),
            "result": str(control / "remote-archive-result.json"),
            "authorization": str(control / "remote-eviction-authorization.json"),
            "plan": str(control / "remote-eviction-plan.json"),
            "receipt": str(control / "remote-eviction-receipt.json"),
            "prepare_lock": str(control / "prepare.lock"),
            "stage_lock": str(control / "stage.lock"),
            "plan_lock": str(control / "plan.lock"),
            "executor_lock": str(control / "executor.lock"),
        }


class RollingArchiveCompanion:
    def __init__(
        self,
        config: RollingArchiveCompanionConfig,
        *,
        transport: RollingArchiveTransport | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if type(config) is not RollingArchiveCompanionConfig:
            raise TypeError("rolling companion requires an exact config")
        self.config = config
        self.transport = transport or SshRollingArchiveTransport(config)
        self.clock_ns = clock_ns
        self.sleeper = sleeper

    def run_once(self) -> CompanionEvent:
        state_root = Path(self.config.state_root)
        state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with SingletonOperatorLock(self.config.lock_path):
            for ordinal, node in enumerate(_node_order()):
                try:
                    index_path = self._index_path(node, ordinal)
                    if os.path.lexists(index_path):
                        load_companion_node_index(index_path)
                        continue
                    probe = dict(self.transport.probe(node=node, ordinal=ordinal))
                    if (
                        probe.get("node") != node
                        or probe.get("ordinal") != ordinal
                        or probe.get("run_id") != self.config.run_id
                        or probe.get("status") not in {"ABSENT", "AVAILABLE"}
                    ):
                        return self._failure(node, "remote_boundary_probe_differs")
                    if probe["status"] == "ABSENT":
                        continue
                    return self._archive_node(node=node, ordinal=ordinal)
                except (
                    OSError,
                    TypeError,
                    ValueError,
                    FormalRemoteArchiveError,
                    FormalRollingArchiveCompanionError,
                ) as error:
                    return self._failure(
                        node,
                        f"{type(error).__name__}:{error}",
                    )
            return CompanionEvent(
                schema_version=1,
                kind=_EVENT_KIND,
                action="IDLE",
                run_id=self.config.run_id,
                node=None,
                detail="no_new_sealed_boundary",
                changed=False,
                occurred_at_ns=self._now(),
            )

    def run(
        self,
        *,
        max_cycles: int | None = None,
        event_sink: Callable[[CompanionEvent], None] | None = None,
    ) -> tuple[CompanionEvent, ...]:
        if max_cycles is not None and (
            isinstance(max_cycles, bool)
            or not isinstance(max_cycles, int)
            or max_cycles < 1
        ):
            raise ValueError("max cycles must be a positive integer or null")
        emitted = []
        prior_signature: tuple[str, str | None, str] | None = None
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            event = self.run_once()
            signature = (event.action, event.node, event.detail)
            if event.changed and signature != prior_signature:
                emitted.append(event)
                if event_sink is not None:
                    event_sink(event)
            prior_signature = signature
            cycles += 1
            if max_cycles is None or cycles < max_cycles:
                self.sleeper(float(self.config.poll_interval_seconds))
        return tuple(emitted)

    def restore_all(
        self,
        *,
        order: Literal["forward", "reverse"] = "reverse",
    ) -> tuple[CompanionEvent, ...]:
        if order not in {"forward", "reverse"}:
            raise ValueError("restore order must be forward or reverse")
        rows = list(enumerate(_node_order()))
        if order == "reverse":
            rows.reverse()
        events = []
        with SingletonOperatorLock(self.config.lock_path):
            for ordinal, node in rows:
                try:
                    index_path = self._index_path(node, ordinal)
                    if not os.path.lexists(index_path):
                        continue
                    index = load_companion_node_index(index_path)
                    _validate_node_index_scope(
                        self.config,
                        index,
                        expected_node=node,
                        expected_ordinal=ordinal,
                    )
                    restore_path = self._restore_index_path(node, ordinal)
                    existing_restore = None
                    if os.path.lexists(restore_path):
                        existing_restore = load_companion_restore_index(restore_path)
                        if (
                            existing_restore.run_id != index.run_id
                            or existing_restore.node != node
                            or existing_restore.ordinal != ordinal
                            or existing_restore.archive_index_sha256 != index.sha256
                        ):
                            raise FormalRollingArchiveCompanionError(
                                "companion restore index scope differs"
                            )
                    response = dict(self.transport.restore(index))
                    receipt_sha = _sha(
                        response.get("receipt_sha256"),
                        "restore response receipt SHA-256",
                    )
                    if existing_restore is not None:
                        if existing_restore.restore_receipt_sha256 != receipt_sha:
                            raise FormalRollingArchiveCompanionError(
                                "remote restore receipt changed during replay"
                            )
                        continue
                    restore = CompanionRestoreIndex(
                        schema_version=1,
                        kind=_RESTORE_INDEX_KIND,
                        run_id=index.run_id,
                        node=index.node,
                        ordinal=index.ordinal,
                        archive_index_sha256=index.sha256,
                        restore_receipt_sha256=receipt_sha,
                        restored_at_ns=self._now(),
                    )
                    _publish_no_replace(restore_path, asdict(restore))
                    events.append(
                        CompanionEvent(
                            schema_version=1,
                            kind=_EVENT_KIND,
                            action="RESTORED",
                            run_id=index.run_id,
                            node=index.node,
                            detail=restore.sha256,
                            changed=True,
                            occurred_at_ns=self._now(),
                        )
                    )
                except (
                    OSError,
                    TypeError,
                    ValueError,
                    FormalRemoteArchiveError,
                    FormalRollingArchiveCompanionError,
                ) as error:
                    events.append(
                        self._failure(
                            node,
                            f"restore:{type(error).__name__}:{error}",
                        )
                    )
                    break
        return tuple(events)

    def _archive_node(self, *, node: str, ordinal: int) -> CompanionEvent:
        request = self.transport.prepare(node=node, ordinal=ordinal)
        if request.remote_payload_root.startswith(
            str(Path(self.config.remote_run_root) / "v02")
        ):
            raise FormalRollingArchiveCompanionError("v02 request is forbidden")
        node_state = Path(self.config.state_root) / f"{ordinal:02d}-{node}"
        node_state.mkdir(mode=0o700, parents=True, exist_ok=True)
        result_path = node_state / "remote-archive-result.json"
        result = self.transport.archive(
            node=node,
            ordinal=ordinal,
            request=request,
            result_path=result_path,
            lock_path=node_state / "archive.lock",
        )
        eviction = dict(
            self.transport.evict(
                node=node,
                ordinal=ordinal,
                request=request,
                result=result,
            )
        )
        plan, plan_envelope = _plan_from_response(eviction.get("plan"))
        receipt, receipt_envelope = _receipt_from_response(eviction.get("receipt"))
        if (
            receipt.status != "COMPLETE"
            or receipt.plan_sha256 != plan.sha256
            or plan.archive_id != request.archive_id
            or result.archive_id != request.archive_id
        ):
            raise FormalRollingArchiveCompanionError(
                "remote eviction response is not complete and path-bound"
            )
        local_plan_path = node_state / "remote-eviction-plan.json"
        local_receipt_path = node_state / "remote-eviction-receipt.json"
        _publish_or_replay(local_plan_path, plan_envelope)
        _publish_or_replay(local_receipt_path, receipt_envelope)
        index = CompanionNodeIndex(
            schema_version=1,
            kind=_INDEX_KIND,
            run_id=self.config.run_id,
            node=node,
            ordinal=ordinal,
            request_sha256=_semantic_sha(asdict(request)),
            local_archive_result_path=str(result_path),
            local_archive_result_sha256=result.sha256,
            local_plan_path=str(local_plan_path),
            plan_sha256=plan.sha256,
            local_eviction_receipt_path=str(local_receipt_path),
            eviction_receipt_sha256=receipt.sha256,
            remote_plan_path=str(
                _absolute(eviction.get("plan_path"), "remote plan path")
            ),
            remote_eviction_receipt_path=str(
                _absolute(eviction.get("receipt_path"), "remote receipt path")
            ),
            archived_at_ns=self._now(),
        )
        _publish_no_replace(self._index_path(node, ordinal), asdict(index))
        return CompanionEvent(
            schema_version=1,
            kind=_EVENT_KIND,
            action="ARCHIVED",
            run_id=self.config.run_id,
            node=node,
            detail=index.sha256,
            changed=True,
            occurred_at_ns=self._now(),
        )

    def _failure(self, node: str, detail: str) -> CompanionEvent:
        try:
            self.transport.stop("rolling_archive_companion_failure")
        except BaseException as stop_error:  # noqa: BLE001
            detail = f"{detail};scheduler_stop:{type(stop_error).__name__}"
        return CompanionEvent(
            schema_version=1,
            kind=_EVENT_KIND,
            action="FAILED",
            run_id=self.config.run_id,
            node=node,
            detail=detail[:2048],
            changed=True,
            occurred_at_ns=self._now(),
        )

    def _index_path(self, node: str, ordinal: int) -> Path:
        return (
            Path(self.config.state_root)
            / f"{ordinal:02d}-{node}"
            / "rolling-archive-index.json"
        )

    def _restore_index_path(self, node: str, ordinal: int) -> Path:
        return (
            Path(self.config.state_root)
            / f"{ordinal:02d}-{node}"
            / "rolling-restore-index.json"
        )

    def _now(self) -> int:
        value = self.clock_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("companion clock must return a positive integer")
        return value


def load_companion_node_index(path: str | Path) -> CompanionNodeIndex:
    value = _read_canonical(path, "rolling companion index")
    if set(value) != set(CompanionNodeIndex.__dataclass_fields__):
        raise ValueError("rolling companion index fields differ")
    return CompanionNodeIndex(**value)


def load_companion_restore_index(path: str | Path) -> CompanionRestoreIndex:
    value = _read_canonical(path, "rolling companion restore index")
    if set(value) != set(CompanionRestoreIndex.__dataclass_fields__):
        raise ValueError("rolling companion restore index fields differ")
    return CompanionRestoreIndex(**value)


def _validate_node_index_scope(
    config: RollingArchiveCompanionConfig,
    index: CompanionNodeIndex,
    *,
    expected_node: str | None = None,
    expected_ordinal: int | None = None,
) -> None:
    order = _node_order()
    if (
        index.run_id != config.run_id
        or index.ordinal >= len(order)
        or order[index.ordinal] != index.node
        or (expected_node is not None and index.node != expected_node)
        or (expected_ordinal is not None and index.ordinal != expected_ordinal)
    ):
        raise FormalRollingArchiveCompanionError(
            "rolling companion index node scope differs"
        )
    node_state = Path(config.state_root) / f"{index.ordinal:02d}-{index.node}"
    control = (
        Path(config.remote_run_root)
        / "rolling-archive-control"
        / f"{index.ordinal:02d}-{index.node}"
    )
    expected_paths = (
        (index.local_archive_result_path, node_state / "remote-archive-result.json"),
        (index.local_plan_path, node_state / "remote-eviction-plan.json"),
        (
            index.local_eviction_receipt_path,
            node_state / "remote-eviction-receipt.json",
        ),
        (index.remote_plan_path, control / "remote-eviction-plan.json"),
        (
            index.remote_eviction_receipt_path,
            control / "remote-eviction-receipt.json",
        ),
    )
    if any(Path(observed) != expected for observed, expected in expected_paths):
        raise FormalRollingArchiveCompanionError(
            "rolling companion index artifact paths differ"
        )


def _node_order() -> tuple[str, ...]:
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FORMAL_SINGLE_OPERATOR_NODE_ORDER,
    )

    order = tuple(FORMAL_SINGLE_OPERATOR_NODE_ORDER)
    if len(order) != 21 or len(set(order)) != 21:
        raise FormalRollingArchiveCompanionError("formal DAG node order differs")
    return order


def _plan_from_response(
    value: object,
) -> tuple[RemoteEvictionPlan, dict[str, object]]:
    if type(value) is not dict:
        raise FormalRollingArchiveCompanionError("remote plan response is absent")
    row = dict(value)
    expected = row.pop("plan_sha256", None)
    plan = RemoteEvictionPlan.from_dict(row)
    if expected != plan.sha256:
        raise FormalRollingArchiveCompanionError("remote plan digest differs")
    return plan, {**plan.to_dict(), "plan_sha256": plan.sha256}


def _receipt_from_response(
    value: object,
) -> tuple[RemoteEvictionReceipt, dict[str, object]]:
    if type(value) is not dict:
        raise FormalRollingArchiveCompanionError("remote receipt response is absent")
    row = dict(value)
    expected = row.pop("receipt_sha256", None)
    receipt = RemoteEvictionReceipt.from_dict(row)
    if expected != receipt.sha256:
        raise FormalRollingArchiveCompanionError("remote receipt digest differs")
    return receipt, {**receipt.to_dict(), "receipt_sha256": receipt.sha256}


def _absolute(value: object, label: str) -> Path:
    if type(value) is not str:
        raise TypeError(f"{label} must be text")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    return path


def _has_v03_marker(path: Path) -> bool:
    return any(
        re.search(r"(?:^|[-_.])v03(?:$|[-_.])", part, re.IGNORECASE)
        for part in path.parts
    )


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _semantic_sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = _absolute(str(path), label)
    if source.is_symlink() or not source.is_file():
        raise FormalRollingArchiveCompanionError(f"{label} is not a regular file")
    body = source.read_bytes()
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalRollingArchiveCompanionError(f"{label} is not JSON") from error
    if type(value) is not dict or body != canonical_json_bytes(value):
        raise FormalRollingArchiveCompanionError(f"{label} is not canonical")
    return value


def _publish_no_replace(path: Path, value: object) -> None:
    path = _absolute(str(path), "companion publication")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        body = canonical_json_bytes(value)
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _publish_or_replay(path: Path, value: object) -> None:
    if os.path.lexists(path):
        if _read_canonical(path, "companion replay artifact") != value:
            raise FormalRollingArchiveCompanionError(
                "companion replay artifact is immutable"
            )
        return
    _publish_no_replace(path, value)


def _load_endpoint(path: str | Path) -> SshRsyncArchiveEndpoint:
    value = _read_canonical(path, "rolling companion endpoint")
    forbidden = {"password", "token", "secret", "private_key"}
    if forbidden & {key.casefold() for key in value}:
        raise ValueError("endpoint must not contain inline credentials")
    return SshRsyncArchiveEndpoint(**value)


def _load_config(arguments: argparse.Namespace) -> RollingArchiveCompanionConfig:
    return RollingArchiveCompanionConfig(
        endpoint=_load_endpoint(arguments.endpoint),
        remote_run_root=arguments.remote_run_root,
        local_results_root=arguments.local_results_root,
        state_root=arguments.state_root,
        lock_path=arguments.lock,
        minimum_local_free_bytes=arguments.minimum_local_free_bytes,
        minimum_remote_restore_free_bytes=(arguments.minimum_remote_restore_free_bytes),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--remote-run-root", required=True)
    parser.add_argument("--local-results-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument(
        "--minimum-local-free-bytes",
        type=int,
        default=MINIMUM_LOCAL_ARCHIVE_FREE_BYTES,
    )
    parser.add_argument(
        "--minimum-remote-restore-free-bytes",
        type=int,
        default=REMOTE_SPOOL_SAFETY_RESERVE_BYTES,
    )
    operations = parser.add_subparsers(dest="operation", required=True)
    operations.add_parser("run-once", allow_abbrev=False)
    run = operations.add_parser("run", allow_abbrev=False)
    run.add_argument("--max-cycles", type=int)
    restore = operations.add_parser("restore-all", allow_abbrev=False)
    restore.add_argument("--order", choices=("forward", "reverse"), default="reverse")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        companion = RollingArchiveCompanion(_load_config(arguments))
        if arguments.operation == "run-once":
            events = (companion.run_once(),)
        elif arguments.operation == "run":
            events = companion.run(max_cycles=arguments.max_cycles)
        elif arguments.operation == "restore-all":
            events = companion.restore_all(order=arguments.order)
        else:
            raise AssertionError(
                f"unhandled rolling companion operation: {arguments.operation}"
            )
    except (
        OSError,
        TypeError,
        ValueError,
        FormalRemoteArchiveError,
        FormalRollingArchiveCompanionError,
    ) as error:
        print(f"formal rolling archive companion: {error}", file=os.sys.stderr)
        return 2
    for event in events:
        if event.changed or arguments.operation == "run-once":
            print(canonical_json_bytes(asdict(event)).decode("utf-8"), end="")
    return 2 if any(event.action == "FAILED" for event in events) else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CompanionEvent",
    "CompanionNodeIndex",
    "CompanionRestoreIndex",
    "FormalRollingArchiveCompanionError",
    "RollingArchiveCompanion",
    "RollingArchiveCompanionConfig",
    "RollingArchiveTransport",
    "SshRollingArchiveTransport",
    "load_companion_node_index",
    "load_companion_restore_index",
    "main",
]
