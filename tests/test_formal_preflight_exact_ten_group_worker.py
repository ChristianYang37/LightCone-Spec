from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

import lightcone_spec.orchestration.formal_preflight_exact_ten_group_worker as group_worker
from lightcone_spec.orchestration.experiment_operator import (
    CellAttemptSpec,
    ExperimentOperatorStore,
    FormalExperimentSchedulerDaemon,
    InterferenceEnvelope,
    PhysicalAttemptGroupMemberSpec,
    ProcessObservation,
    QueuedCommandSpec,
    SchedulerCallbacks,
    SpawnedProcess,
    StagePlanEntry,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    ProductionCallbackError,
    ProductionSchedulerRuntime,
    canonical_json_bytes,
    child_heartbeat_path,
    child_start_receipt_path,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass(frozen=True)
class _CompletionRow:
    materialized_cell_id: str
    registry_cell_id: str
    runner_kind: str
    status: str
    started_ns: int
    finished_ns: int
    result_sha256: str

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class _Completion:
    rows: tuple[_CompletionRow, ...]
    status: str
    started_ns: int
    finished_ns: int


@dataclass(frozen=True)
class _Execution:
    status: str
    completion: CanonicalJsonProofBinding


def _fake_nvidia_smi(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout=(
            "0, GPU-000, 12, 1024, 49140, 80.5\n1, GPU-111, 34, 2048, 49140, 91.5\n"
        ),
        stderr="",
    )


def _group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_ns: int,
) -> tuple[
    str,
    tuple[PhysicalAttemptGroupMemberSpec, ...],
    CanonicalJsonProofBinding,
    _Execution,
    _Completion,
]:
    source_root = tmp_path / "source"
    group_root = tmp_path / "group"
    source_root.mkdir(mode=0o700)
    group_root.mkdir(mode=0o700)
    inputs_path = (source_root / "formal-preflight-execution-inputs.json").resolve()
    publish_canonical_json_no_replace(
        inputs_path,
        {"schema_version": 1, "kind": "test-exact-ten-inputs"},
    )
    spec_path = (
        group_root / "formal-preflight-exact-ten-group-worker-spec.json"
    ).resolve()
    environment = group_worker.formal_preflight_exact_ten_group_environment(spec_path)
    parent_argv = (
        "lightcone-spec",
        "formal-single-operator",
        "execute-preflight",
        "--execution-inputs",
        str(inputs_path),
        "--current-ns",
        str(base_ns),
    )
    kinds = ("compile", "exactness", *("interference" for _ in range(8)))
    members = []
    for index, kind in enumerate(kinds):
        cell_id = f"preflight-cell-{index:02d}"
        prefix = (group_root / f"member-{index:02d}").resolve()
        command = QueuedCommandSpec(
            cell_id=cell_id,
            attempt=1,
            argv=parent_argv,
            launch_compatibility_key="preflight:exact-ten:tp2",
            required_gpu_count=2,
            timing_class="EXCLUSIVE",
            predicted_high_water_bytes=1024,
            monitored_path=str(tmp_path.resolve()),
            log_path=str(prefix.with_suffix(".command.log")),
            expected_terminal_path=str(prefix.with_suffix(".terminal.json")),
            expected_junit_path=str(prefix.with_suffix(".junit.xml")),
            expected_raw_log_path=str(prefix.with_suffix(".raw.jsonl")),
            atomic_pointer_path=str(prefix.with_suffix(".pointer.json")),
            child_exit_receipt_path=str(prefix.with_suffix(".exit.json")),
            environment=environment,
        )
        attempt = CellAttemptSpec(
            cell_id=cell_id,
            attempt=1,
            stage="preflight",
            phase="final",
            block="preflight",
            seed=17,
            scientific_axes={"logical_kind": kind, "topology": "tp2_dp1"},
            identity={
                "source_sha256": _sha("source"),
                "patch_sha256": _sha("patch"),
                "registry_sha256": _sha("registry"),
            },
            command_sha256=command.command_sha256,
            output_directory=str((group_root / f"output-{index:02d}").resolve()),
        )
        members.append(PhysicalAttemptGroupMemberSpec(attempt, command, kind))
    member_tuple = tuple(members)
    source_kinds = {
        row.attempt.cell_id: {
            "compile": "first_party_compile",
            "exactness": "first_party_exactness",
            "interference": "first_party_interference",
        }[row.logical_kind]
        for row in member_tuple
    }
    monkeypatch.setattr(
        group_worker,
        "_revalidate_source_projection",
        lambda _path, *, current_ns: (
            source_kinds,
            ("GPU-000", "GPU-111"),
        ),
    )
    monkeypatch.setenv("LIGHTCONE_ASSIGNED_GPU_UUIDS", "GPU-000,GPU-111")
    spec_binding = group_worker.publish_formal_preflight_exact_ten_group_worker_spec(
        group_id="preflight-exact-ten-parent",
        members=member_tuple,
        leader_cell_id=member_tuple[0].attempt.cell_id,
        output_path=spec_path,
    )
    completion_path = (source_root / "exact-ten-completion.json").resolve()
    publish_canonical_json_no_replace(
        completion_path,
        {"schema_version": 1, "kind": "test-exact-ten-completion"},
    )
    completion_binding = CanonicalJsonProofBinding.bind(completion_path)
    rows = tuple(
        _CompletionRow(
            materialized_cell_id=member.attempt.cell_id,
            registry_cell_id=_sha(f"registry-{index}"),
            runner_kind=source_kinds[member.attempt.cell_id],
            status="COMPLETE",
            started_ns=base_ns + 100 + index * 10,
            finished_ns=base_ns + 105 + index * 10,
            result_sha256=_sha(f"result-{index}"),
        )
        for index, member in enumerate(member_tuple)
    )
    completion = _Completion(
        rows=rows,
        status="COMPLETE",
        started_ns=min(row.started_ns for row in rows),
        finished_ns=max(row.finished_ns for row in rows),
    )
    execution = _Execution("COMPLETE", completion_binding)
    monkeypatch.setattr(
        group_worker,
        "_revalidate_execution",
        lambda _path, *, current_ns: execution,
    )
    monkeypatch.setattr(
        group_worker,
        "_revalidate_completion",
        lambda _path, *, current_ns: completion,
    )
    return (
        "preflight-exact-ten-parent",
        member_tuple,
        spec_binding,
        execution,
        completion,
    )


def _publish_group(
    members: tuple[PhysicalAttemptGroupMemberSpec, ...],
    spec_binding: CanonicalJsonProofBinding,
) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []
    execution_path = (
        Path(spec_binding.absolute_path).parent.parent
        / "source"
        / "formal-single-operator-preflight-execution.json"
    )

    def runner(argv, **kwargs):
        calls.append(tuple(argv))
        assert kwargs["shell"] is False
        publish_canonical_json_no_replace(
            execution_path,
            {"schema_version": 1, "kind": "test-exact-ten-execution"},
        )
        return subprocess.CompletedProcess(argv, 0)

    assert (
        group_worker.run_formal_preflight_exact_ten_group_worker(
            spec_binding.absolute_path,
            parent_runner=runner,
        )
        == 0
    )
    assert calls == [members[0].command.argv]
    return calls


def _publish_shared_exit(
    command: QueuedCommandSpec,
    *,
    pid: int,
    pgid: int,
    started_ns: int,
    finished_ns: int,
) -> None:
    start = {
        "schema_version": 1,
        "kind": "formal_experiment_child_start_receipt",
        "command_sha256": command.command_sha256,
        "wrapper_pid": pid,
        "wrapper_pgid": pgid,
        "process_start_identity": {
            "kind": "non_linux_process_start_unrecoverable_v1",
            "pid": pid,
            "platform": "test-fixture",
        },
        "started_ns": started_ns,
    }
    publish_canonical_json_no_replace(
        child_start_receipt_path(command),
        {
            **start,
            "receipt_sha256": hashlib.sha256(canonical_json_bytes(start)).hexdigest(),
        },
    )
    value = {
        "schema_version": 1,
        "kind": "formal_experiment_child_exit_receipt",
        "command_sha256": command.command_sha256,
        "wrapper_pid": pid,
        "wrapper_pgid": pgid,
        "child_pid": pid + 1,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "exit_code": 0,
        "launch_error_type": None,
    }
    publish_canonical_json_no_replace(
        command.child_exit_receipt_path,
        {
            **value,
            "receipt_sha256": hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
        },
    )


def test_group_worker_runs_one_parent_and_publishes_exact_ten_distinct_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_ns = time.time_ns()
    _group_id, members, spec_binding, _execution, completion = _group(
        tmp_path,
        monkeypatch,
        base_ns=base_ns,
    )
    calls = _publish_group(members, spec_binding)
    assert len(calls) == 1
    spec = group_worker.revalidate_formal_preflight_exact_ten_group_worker_spec(
        spec_binding.absolute_path
    )
    assert Path(spec.shared_publication_path).is_file()
    assert len({row.command.command_sha256 for row in members}) == 1
    assert (
        len({Path(row.command.expected_terminal_path).read_bytes() for row in members})
        == 10
    )
    for member, completion_row in zip(members, completion.rows, strict=True):
        assert Path(member.command.expected_junit_path).is_file()
        assert Path(member.command.expected_raw_log_path).is_file()
        terminal = json.loads(Path(member.command.expected_terminal_path).read_text())
        assert terminal["cell_id"] == member.attempt.cell_id
        assert terminal["started_ns"] == completion_row.started_ns
        assert terminal["status"] == "COMPLETE"


def test_group_worker_publishes_typed_child_owned_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_ns = time.time_ns()
    _group_id, members, spec_binding, _execution, _completion = _group(
        tmp_path,
        monkeypatch,
        base_ns=base_ns,
    )
    leader = members[0].command
    heartbeat_path = child_heartbeat_path(leader)
    environment = {
        "LIGHTCONE_OPERATOR_CELL_ID": leader.cell_id,
        "LIGHTCONE_OPERATOR_ATTEMPT": str(leader.attempt),
        "LIGHTCONE_OPERATOR_COMMAND_SHA256": leader.command_sha256,
        "LIGHTCONE_OPERATOR_TERMINAL_PATH": leader.expected_terminal_path,
        "LIGHTCONE_OPERATOR_JUNIT_PATH": leader.expected_junit_path,
        "LIGHTCONE_OPERATOR_RAW_LOG_PATH": leader.expected_raw_log_path,
        "LIGHTCONE_OPERATOR_POINTER_PATH": leader.atomic_pointer_path,
        "LIGHTCONE_OPERATOR_HEARTBEAT_PATH": str(heartbeat_path),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    execution_path = (
        Path(spec_binding.absolute_path).parent.parent
        / "source"
        / "formal-single-operator-preflight-execution.json"
    )
    runtime = ProductionSchedulerRuntime(nvidia_runner=_fake_nvidia_smi)

    def runner(argv, **_kwargs):
        heartbeat = runtime.worker_heartbeat(leader)
        assert heartbeat is not None
        assert heartbeat.worker_pid > 0
        assert heartbeat.phase == "RUNNING"
        publish_canonical_json_no_replace(
            execution_path,
            {"schema_version": 1, "kind": "test-exact-ten-execution"},
        )
        return subprocess.CompletedProcess(argv, 0)

    assert runtime.worker_heartbeat_required(leader) is True
    assert (
        group_worker.run_formal_preflight_exact_ten_group_worker(
            spec_binding.absolute_path,
            parent_runner=runner,
            heartbeat_interval_seconds=0.01,
        )
        == 0
    )
    heartbeat = runtime.worker_heartbeat(leader)
    assert heartbeat is not None
    assert heartbeat.phase == "FINALIZING"
    assert heartbeat.sequence >= 2


def test_fresh_group_heartbeat_keeps_long_exact_ten_parent_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_ns = time.time_ns()
    group_id, members, _spec_binding, _execution, _completion = _group(
        tmp_path,
        monkeypatch,
        base_ns=base_ns,
    )
    leader = members[0].command
    heartbeat_ns = base_ns + 199_000_000_000
    Path(leader.log_path).write_text("running\n", encoding="utf-8")
    child_heartbeat_path(leader).write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "kind": "formal_experiment_child_heartbeat",
                "cell_id": leader.cell_id,
                "attempt": leader.attempt,
                "command_sha256": leader.command_sha256,
                "worker_pid": 76_001,
                "sequence": 7,
                "observed_at_ns": heartbeat_ns,
                "phase": "RUNNING",
            }
        )
    )
    runtime = ProductionSchedulerRuntime(nvidia_runner=_fake_nvidia_smi)
    production = runtime.callbacks()
    callbacks = SchedulerCallbacks(
        launch=production.launch,
        process_probe=lambda pid, _pgid: ProcessObservation(
            pid,
            True,
            76_000,
            "alive",
        ),
        log_size_bytes=production.log_size_bytes,
        gpu_snapshot=production.gpu_snapshot,
        terminal_validator=production.terminal_validator,
        free_disk_bytes=production.free_disk_bytes,
        worker_heartbeat=production.worker_heartbeat,
        worker_heartbeat_required=production.worker_heartbeat_required,
    )
    with ExperimentOperatorStore(
        tmp_path / "long-group.sqlite3",
        run_id="long-group",
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("preflight", 0, "preflight", "final", "10", 10),)
        )
        store.configure_interference_envelope(
            InterferenceEnvelope(
                "UNRESOLVED",
                ("GPU-000", "GPU-111"),
                _sha("interference"),
            )
        )
        store.materialize_physical_attempt_group(
            group_id=group_id,
            members=members,
            leader_cell_id=members[0].attempt.cell_id,
        )
        store.start_physical_attempt_group_with_launcher(
            group_id,
            assigned_gpu_uuids=("GPU-000", "GPU-111"),
            launcher=lambda: SpawnedProcess(76_000, 76_000),
            started_at_ns=base_ns,
        )
        cycle = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "long-group.scheduler.lock",
            callbacks=callbacks,
            clock_ns=lambda: base_ns + 200_000_000_000,
        ).run_once()
        assert cycle.reconciled == ((leader.cell_id, 1, "HEARTBEAT"),), [
            (event["event_type"], event["payload"]) for event in store._event_rows()
        ]
        assert store.dispatch_control() == ("RUN", None)
        assert not any(
            event["event_type"] == "CHILD_HEARTBEAT_STALE"
            for event in store._event_rows()
        )


def test_production_launch_substitutes_one_group_worker_for_parent_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_ns = time.time_ns()
    _group_id, members, _spec_binding, _execution, _completion = _group(
        tmp_path,
        monkeypatch,
        base_ns=base_ns,
    )
    captured: dict[str, object] = {}

    class _Child:
        pid = 73000

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def terminate() -> None:
            raise AssertionError("valid setsid child must not be terminated")

    def popen(argv, **kwargs):
        captured["argv"] = tuple(argv)
        captured["environment"] = kwargs["env"]
        return _Child()

    monkeypatch.setattr(
        "lightcone_spec.orchestration.experiment_operator_production.os.getpgid",
        lambda pid: pid,
    )
    runtime = ProductionSchedulerRuntime(
        nvidia_runner=_fake_nvidia_smi,
        popen=popen,
    )
    process = runtime.launch(members[0].command, ("GPU-000", "GPU-111"))
    assert process == SpawnedProcess(73000, 73000)
    argv = captured["argv"]
    assert isinstance(argv, tuple)
    separator = argv.index("--")
    physical_child = argv[separator + 1 :]
    assert physical_child[:3] == (
        runtime.python_executable,
        "-m",
        "lightcone_spec.orchestration.formal_preflight_exact_ten_group_worker",
    )
    assert members[0].command.argv[0] not in physical_child
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-000,GPU-111"


def test_restart_deep_validates_shared_group_and_accounts_physical_time_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_ns = time.time_ns()
    group_id, members, spec_binding, _execution, _completion = _group(
        tmp_path,
        monkeypatch,
        base_ns=base_ns,
    )
    _publish_group(members, spec_binding)
    publication = json.loads(
        Path(
            group_worker.revalidate_formal_preflight_exact_ten_group_worker_spec(
                spec_binding.absolute_path
            ).shared_publication_path
        ).read_text()
    )
    pid = 74000
    _publish_shared_exit(
        members[0].command,
        pid=pid,
        pgid=pid,
        started_ns=base_ns,
        finished_ns=publication["published_ns"] + 1_000,
    )
    database = tmp_path / "operator.sqlite3"
    with ExperimentOperatorStore(database, run_id="exact-ten-restart") as store:
        store.initialize_stage_plan(
            (StagePlanEntry("preflight", 0, "preflight", "final", "10", 10),)
        )
        store.configure_interference_envelope(
            InterferenceEnvelope(
                "UNRESOLVED",
                ("GPU-000", "GPU-111"),
                _sha("interference"),
            )
        )
        store.materialize_physical_attempt_group(
            group_id=group_id,
            members=members,
            leader_cell_id=members[0].attempt.cell_id,
        )
        store.start_physical_attempt_group_with_launcher(
            group_id,
            assigned_gpu_uuids=("GPU-000", "GPU-111"),
            launcher=lambda: SpawnedProcess(pid, pid),
            started_at_ns=base_ns - 1,
        )
    restarted_runtime = ProductionSchedulerRuntime(nvidia_runner=_fake_nvidia_smi)
    callbacks = restarted_runtime.callbacks()
    callbacks = SchedulerCallbacks(
        launch=callbacks.launch,
        process_probe=lambda child_pid, _pgid: ProcessObservation(
            child_pid,
            False,
            None,
            "restarted_parent_exited",
            exit_code=0,
        ),
        log_size_bytes=callbacks.log_size_bytes,
        gpu_snapshot=callbacks.gpu_snapshot,
        terminal_validator=callbacks.terminal_validator,
        free_disk_bytes=callbacks.free_disk_bytes,
        retry_builder=None,
    )
    with ExperimentOperatorStore(database) as store:
        assert (
            callbacks.terminal_validator(
                members[0].command,
                store.attempt(members[0].attempt.cell_id, 1),
                ProcessObservation(pid, False, None, "restarted_parent_exited", 0),
            )
            is not None
        )
        cycle = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "scheduler.lock",
            callbacks=callbacks,
            clock_ns=lambda: publication["published_ns"] + 2_000,
        ).run_once()
        assert len(cycle.reconciled) == 10
        attempts = tuple(
            store.attempt(member.attempt.cell_id, member.attempt.attempt)
            for member in members
        )
        assert {row["status"] for row in attempts} == {"COMPLETE"}
        assert [row["compute_gpu_seconds"] > 0 for row in attempts] == [
            True,
            *([False] * 9),
        ]
        assert sum(row["reserved_gpu_seconds"] for row in attempts) == pytest.approx(
            attempts[0]["compute_gpu_seconds"]
        )
        group = store.physical_attempt_groups()[0]
        assert (
            group["shared_evidence_sha256"]
            == CanonicalJsonProofBinding.bind(
                group_worker.revalidate_formal_preflight_exact_ten_group_worker_spec(
                    spec_binding.absolute_path
                ).shared_publication_path
            ).raw_sha256
        )


def test_group_terminal_rejects_any_member_or_shared_completion_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_ns = time.time_ns()
    _group_id, members, spec_binding, _execution, _completion = _group(
        tmp_path,
        monkeypatch,
        base_ns=base_ns,
    )
    _publish_group(members, spec_binding)
    spec = group_worker.revalidate_formal_preflight_exact_ten_group_worker_spec(
        spec_binding.absolute_path
    )
    publication = json.loads(Path(spec.shared_publication_path).read_text())
    pid = 75000
    _publish_shared_exit(
        members[0].command,
        pid=pid,
        pgid=pid,
        started_ns=base_ns,
        finished_ns=publication["published_ns"] + 1_000,
    )
    Path(members[-1].command.expected_raw_log_path).write_text(
        '{"tampered":true}\n',
        encoding="utf-8",
    )
    runtime = ProductionSchedulerRuntime(nvidia_runner=_fake_nvidia_smi)
    with pytest.raises(ProductionCallbackError, match="failed deep validation"):
        runtime.terminal_validator(
            members[0].command,
            {"pid": pid, "pgid": pid},
            ProcessObservation(pid, False, None, "exited", exit_code=0),
        )

    completion_path = Path(publication["completion"]["absolute_path"])
    completion_path.write_text('{"coordinated":"replacement"}\n', encoding="utf-8")
    with pytest.raises(ProductionCallbackError, match="failed deep validation"):
        runtime.terminal_validator(
            members[1].command,
            {"pid": pid, "pgid": pid},
            ProcessObservation(pid, False, None, "exited", exit_code=0),
        )
