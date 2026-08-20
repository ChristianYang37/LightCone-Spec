from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from lightcone_spec.orchestration import (
    experiment_operator as experiment_operator_module,
)
from lightcone_spec.orchestration import (
    experiment_operator_production as experiment_operator_production_module,
)
from lightcone_spec.orchestration.experiment_operator import (
    ArchiveCallbacks,
    ArchiveRequest,
    ArchiveStepReceipt,
    AttemptTransitionError,
    CellAttemptSpec,
    ControllerArtifactBinding,
    ExperimentOperatorStore,
    FormalExperimentSchedulerDaemon,
    InterferenceEnvelope,
    LegacyStaleAttempt,
    MetricRecord,
    OperatorAlreadyRunningError,
    PhysicalAttemptGroupMemberSpec,
    ProcessObservation,
    ProviderRuntimeSample,
    QueuedCommandSpec,
    RecoveredProcessStart,
    SchedulerCallbacks,
    SingletonOperatorLock,
    SpawnedProcess,
    StagePlanEntry,
    TerminalEvidence,
    WatchdogPolicy,
    default_formal_stage_plan,
    evaluate_dispatch_disk_gate,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    ProductionArchiveRuntime,
    ProductionCallbackError,
    ProductionSchedulerRuntime,
    canonical_json_bytes,
    file_sha256,
    publish_atomic_terminal_result,
    query_nvidia_smi,
    run_child_wrapper,
    statvfs_free_bytes,
)


def _sha(character: str) -> str:
    return character * 64


def _plan() -> tuple[StagePlanEntry, ...]:
    return (
        StagePlanEntry("preflight", 0, "preflight", "final", "10", 10),
        StagePlanEntry("e3a", 1, "E3a", "selection", "360", 360),
    )


def _spec(tmp_path: Path, *, cell_id: str = "preflight:compile", attempt: int = 1):
    return CellAttemptSpec(
        cell_id=cell_id,
        attempt=attempt,
        stage="preflight",
        phase="final",
        block="pilot-00",
        seed=17,
        scientific_axes={"task": "compile", "topology": "tp2_dp1"},
        identity={
            "source_sha256": _sha("a"),
            "patch_sha256": _sha("b"),
            "registry_sha256": _sha("c"),
            "model_revision": "immutable-revision",
        },
        command_sha256=_sha("d"),
        output_directory=str((tmp_path / f"output-{attempt}").resolve()),
    )


def _queued_pair(
    tmp_path: Path,
    *,
    cell_id: str,
    attempt: int = 1,
    timing_class: str = "HEADLINE",
    required_gpu_count: int = 1,
    compatibility_key: str = "model:dflash:tp1",
    priority: int = 0,
) -> tuple[CellAttemptSpec, QueuedCommandSpec]:
    argv = ("python", "formal-runner.py", "--cell-id", cell_id)
    prefix = tmp_path / f"{cell_id.replace(':', '-')}-{attempt}"
    command = QueuedCommandSpec(
        cell_id=cell_id,
        attempt=attempt,
        argv=argv,
        launch_compatibility_key=compatibility_key,
        required_gpu_count=required_gpu_count,
        timing_class=timing_class,
        predicted_high_water_bytes=1_024,
        monitored_path=str(tmp_path.resolve()),
        log_path=str(prefix.with_suffix(".command.log").resolve()),
        expected_terminal_path=str(prefix.with_suffix(".terminal.json").resolve()),
        expected_junit_path=str(prefix.with_suffix(".junit.xml").resolve()),
        expected_raw_log_path=str(prefix.with_suffix(".raw.jsonl").resolve()),
        atomic_pointer_path=str(prefix.with_suffix(".pointer.json").resolve()),
        child_exit_receipt_path=str(prefix.with_suffix(".exit.json").resolve()),
        priority=priority,
    )
    base = _spec(tmp_path, cell_id=cell_id, attempt=attempt)
    spec = CellAttemptSpec(
        **{
            **vars(base),
            "command_sha256": command.command_sha256,
            "output_directory": str(
                (tmp_path / f"output-{cell_id.replace(':', '-')}-{attempt}").resolve()
            ),
        }
    )
    return spec, command


def test_production_log_growth_includes_source_bound_progress_logs(
    tmp_path: Path,
) -> None:
    _attempt, command = _queued_pair(tmp_path, cell_id="preflight:progress")
    progress = tuple((tmp_path / f"server-{index}.log").resolve() for index in range(3))
    Path(command.log_path).write_bytes(b"command")
    progress[0].write_bytes(b"stdout")
    progress[1].write_bytes(b"stderr-more")
    command = replace(
        command,
        environment=(
            (
                "LIGHTCONE_OPERATOR_PROGRESS_LOG_PATHS_JSON",
                json.dumps([str(path) for path in progress]),
            ),
        ),
    )

    assert ProductionSchedulerRuntime.log_size_bytes(command) == (
        len(b"command") + len(b"stdout") + len(b"stderr-more")
    )


def _exact_ten_group(
    tmp_path: Path,
) -> tuple[str, tuple[PhysicalAttemptGroupMemberSpec, ...]]:
    kinds = ("compile", "exactness", *("interference" for _ in range(8)))
    members = []
    for index, kind in enumerate(kinds):
        cell_id = f"preflight:{index:02d}-{kind}"
        prefix = tmp_path / cell_id.replace(":", "-")
        command = QueuedCommandSpec(
            cell_id=cell_id,
            attempt=1,
            argv=("python3", "-m", "fixture.exact_ten_parent"),
            launch_compatibility_key="preflight:exact-ten:tp2",
            required_gpu_count=2,
            timing_class="EXCLUSIVE",
            predicted_high_water_bytes=1_024,
            monitored_path=str(tmp_path.resolve()),
            log_path=str(prefix.with_suffix(".command.log").resolve()),
            expected_terminal_path=str(prefix.with_suffix(".terminal.json").resolve()),
            expected_junit_path=str(prefix.with_suffix(".junit.xml").resolve()),
            expected_raw_log_path=str(prefix.with_suffix(".raw.jsonl").resolve()),
            atomic_pointer_path=str(prefix.with_suffix(".pointer.json").resolve()),
            child_exit_receipt_path=str(prefix.with_suffix(".exit.json").resolve()),
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
                "source_sha256": _sha("a"),
                "patch_sha256": _sha("b"),
                "registry_sha256": _sha("c"),
            },
            command_sha256=command.command_sha256,
            output_directory=str((tmp_path / f"output-{index:02d}").resolve()),
        )
        members.append(PhysicalAttemptGroupMemberSpec(attempt, command, kind))
    return "preflight-exact-ten-parent", tuple(members)


class _TickClock:
    def __init__(self) -> None:
        self.value = time.time_ns()

    def __call__(self) -> int:
        self.value += 1_000_000_000
        return self.value


def _scheduler_callbacks(
    *,
    launch,
    process_probe=lambda pid, pgid: ProcessObservation(
        pid,
        True,
        pgid,
        "alive",
    ),
    terminal_validator=lambda _command, _attempt, _process: None,
    free_disk_bytes=lambda _path: 100 * 1024**3,
    retry_builder=None,
) -> SchedulerCallbacks:
    return SchedulerCallbacks(
        launch=launch,
        process_probe=process_probe,
        log_size_bytes=lambda _command: 10,
        gpu_snapshot=lambda gpu_uuids: {
            gpu_uuid: {"utilization_percent": 99} for gpu_uuid in gpu_uuids
        },
        terminal_validator=terminal_validator,
        free_disk_bytes=free_disk_bytes,
        retry_builder=retry_builder,
    )


def _store(tmp_path: Path) -> ExperimentOperatorStore:
    store = ExperimentOperatorStore(tmp_path / "operator.sqlite3", run_id="run-v03")
    store.initialize_stage_plan(_plan())
    return store


def _schema_six_fixture(path: Path) -> None:
    schema = experiment_operator_module._SCHEMA
    aux_start = schema.index("CREATE TABLE IF NOT EXISTS controller_auxiliary_groups")
    aux_end = schema.index(
        "CREATE INDEX IF NOT EXISTS controller_auxiliary_group_order"
    )
    aux = schema[aux_start:aux_end]
    for removed in (
        "    process_start_receipt_sha256 TEXT,\n",
        """    heartbeat_at_ns INTEGER CHECK (heartbeat_at_ns IS NULL OR heartbeat_at_ns > 0),
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
""",
        """    termination_reason TEXT,
    termination_requested_at_ns INTEGER CHECK (
        termination_requested_at_ns IS NULL OR termination_requested_at_ns > 0
    ),
    term_sent_at_ns INTEGER CHECK (term_sent_at_ns IS NULL OR term_sent_at_ns > 0),
    kill_sent_at_ns INTEGER CHECK (kill_sent_at_ns IS NULL OR kill_sent_at_ns > 0),
""",
        "    failure_class TEXT,\n",
        "        AND (termination_reason IS NULL) = (termination_requested_at_ns IS NULL)\n",
    ):
        assert removed in aux
        aux = aux.replace(removed, "", 1)
    schema = schema[:aux_start] + aux + schema[aux_end:]
    jobs_start = schema.index("CREATE TABLE IF NOT EXISTS controller_auxiliary_jobs")
    jobs_end = schema.index("CREATE TABLE IF NOT EXISTS cell_attempts", jobs_start)
    jobs = schema[jobs_start:jobs_end]
    jobs_failure = "    failure_class TEXT,\n    failure_code TEXT,\n"
    assert jobs_failure in jobs
    jobs = jobs.replace(jobs_failure, "    failure_code TEXT,\n", 1)
    schema = schema[:jobs_start] + jobs + schema[jobs_end:]
    cell_start = schema.index("CREATE TABLE IF NOT EXISTS cell_attempts")
    cell_end = schema.index(
        "CREATE UNIQUE INDEX IF NOT EXISTS one_running_attempt_per_cell"
    )
    cells = schema[cell_start:cell_end]
    for removed in (
        "    scientific_command_sha256 TEXT,\n",
        "    process_start_receipt_sha256 TEXT,\n",
        """    termination_reason TEXT,
    termination_requested_at_ns INTEGER CHECK (
        termination_requested_at_ns IS NULL OR termination_requested_at_ns > 0
    ),
    term_sent_at_ns INTEGER CHECK (term_sent_at_ns IS NULL OR term_sent_at_ns > 0),
    kill_sent_at_ns INTEGER CHECK (kill_sent_at_ns IS NULL OR kill_sent_at_ns > 0),
""",
        "    CHECK ((termination_reason IS NULL) = (termination_requested_at_ns IS NULL)),\n",
    ):
        assert removed in cells
        cells = cells.replace(removed, "", 1)
    schema = schema[:cell_start] + cells + schema[cell_end:]
    connection = sqlite3.connect(path)
    try:
        connection.executescript(schema)
        connection.executemany(
            "INSERT INTO operator_meta(key, value) VALUES (?, ?)",
            (
                ("schema_version", "6"),
                ("run_id", "schema-six-run"),
                ("created_at_ns", "1"),
            ),
        )
        connection.execute("PRAGMA user_version=6")
        connection.commit()
    finally:
        connection.close()


def _downgrade_group_members_to_exact_ten_only_schema_seven(path: Path) -> None:
    """Recreate the short-lived schema-7 CHECK used before serving groups."""

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE physical_attempt_group_members_old_v7 (
                group_id TEXT NOT NULL,
                cell_id TEXT NOT NULL,
                attempt INTEGER NOT NULL CHECK (attempt >= 1),
                logical_kind TEXT NOT NULL CHECK (
                    logical_kind IN ('compile', 'exactness', 'interference')
                ),
                member_ordinal INTEGER NOT NULL CHECK (member_ordinal >= 0),
                PRIMARY KEY (group_id, cell_id, attempt),
                UNIQUE (cell_id, attempt),
                UNIQUE (group_id, member_ordinal),
                FOREIGN KEY (group_id) REFERENCES physical_attempt_groups(group_id),
                FOREIGN KEY (cell_id, attempt)
                    REFERENCES cell_attempts(cell_id, attempt)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO physical_attempt_group_members_old_v7
            SELECT group_id, cell_id, attempt, logical_kind, member_ordinal
            FROM physical_attempt_group_members
            """
        )
        connection.execute("DROP TABLE physical_attempt_group_members")
        connection.execute(
            "ALTER TABLE physical_attempt_group_members_old_v7 "
            "RENAME TO physical_attempt_group_members"
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _serving_group(
    tmp_path: Path,
) -> tuple[str, tuple[PhysicalAttemptGroupMemberSpec, ...]]:
    members = []
    scientific_digests = (_sha("8"), _sha("9"))
    for index in range(2):
        cell_id = f"E3a:resident-{index:02d}"
        prefix = tmp_path / cell_id.replace(":", "-")
        command = QueuedCommandSpec(
            cell_id=cell_id,
            attempt=1,
            argv=("python3", "-m", "fixture.resident_group", "--group", "two"),
            launch_compatibility_key="qwen3-8b:dflash:tp1:resident",
            required_gpu_count=1,
            timing_class="HEADLINE",
            predicted_high_water_bytes=2_048,
            monitored_path=str(tmp_path.resolve()),
            log_path=str(prefix.with_suffix(".command.log").resolve()),
            expected_terminal_path=str(prefix.with_suffix(".terminal.json").resolve()),
            expected_junit_path=str(prefix.with_suffix(".junit.xml").resolve()),
            expected_raw_log_path=str(prefix.with_suffix(".raw.jsonl").resolve()),
            atomic_pointer_path=str(prefix.with_suffix(".pointer.json").resolve()),
            child_exit_receipt_path=str(prefix.with_suffix(".exit.json").resolve()),
        )
        attempt = CellAttemptSpec(
            cell_id=cell_id,
            attempt=1,
            stage="E3a",
            phase="selection",
            block="capacity",
            seed=17 + index,
            scientific_axes={"logical_kind": "serving", "trace": index},
            identity={
                "source_sha256": _sha("a"),
                "patch_sha256": _sha("b"),
                "registry_sha256": _sha("c"),
            },
            command_sha256=command.command_sha256,
            scientific_command_sha256=scientific_digests[index],
            output_directory=str((tmp_path / f"resident-output-{index}").resolve()),
        )
        members.append(PhysicalAttemptGroupMemberSpec(attempt, command, "serving"))
    return "resident-two-member", tuple(members)


def _complete(store: ExperimentOperatorStore, spec: CellAttemptSpec) -> None:
    store.materialize_attempt(spec)
    store.mark_running_before_spawn(
        spec.cell_id,
        spec.attempt,
        assigned_gpu_uuids=("GPU-000",),
        started_at_ns=10,
    )
    store.attach_process(spec.cell_id, spec.attempt, pid=101, pgid=101)
    store.record_heartbeat(
        spec.cell_id,
        spec.attempt,
        pid=101,
        pgid=101,
        log_size_bytes=4096,
        gpu_observation={"GPU-000": {"utilization_percent": 98, "memory_bytes": 1}},
        observed_at_ns=20,
    )
    store.finish_attempt(
        spec.cell_id,
        spec.attempt,
        status="COMPLETE",
        exit_code=0,
        terminal_sha256=_sha("e"),
        junit_sha256=_sha("f"),
        raw_log_sha256=_sha("1"),
        evidence_files={"terminal.json": _sha("e"), "junit.xml": _sha("f")},
        included_in_analysis=True,
        exclusion_reason=None,
        compute_gpu_seconds=20,
        reserved_gpu_seconds=22,
        billed_gpu_seconds=44,
        finished_at_ns=30,
    )


def test_schema_six_wal_migrates_all_schema_seven_runtime_columns(
    tmp_path: Path,
) -> None:
    database = tmp_path / "schema-six.sqlite3"
    _schema_six_fixture(database)
    expected = {
        "cell_attempts": {
            "scientific_command_sha256",
            "process_start_receipt_sha256",
            "termination_reason",
            "termination_requested_at_ns",
            "term_sent_at_ns",
            "kill_sent_at_ns",
        },
        "controller_auxiliary_groups": {
            "process_start_receipt_sha256",
            "heartbeat_at_ns",
            "heartbeat_sequence",
            "last_log_size_bytes",
            "last_log_growth_ns",
            "gpu_observation_json",
            "termination_reason",
            "termination_requested_at_ns",
            "term_sent_at_ns",
            "kill_sent_at_ns",
            "failure_class",
        },
        "controller_auxiliary_jobs": {"failure_class"},
    }
    with ExperimentOperatorStore(database, run_id="schema-six-run") as store:
        assert store._metadata_value("schema_version") == "7"
        for table_name, expected_columns in expected.items():
            actual_columns = {
                str(row["name"])
                for row in store._connection.execute(
                    f"PRAGMA table_info({table_name})"
                ).fetchall()
            }
            assert expected_columns <= actual_columns
        assert int(store._connection.execute("PRAGMA user_version").fetchone()[0]) == 7
    with ExperimentOperatorStore(database, run_id="schema-six-run") as reopened:
        assert reopened._metadata_value("schema_version") == "7"


def test_exact_ten_only_schema_seven_rebuilds_without_losing_restart_state(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operator.sqlite3"
    exact_group_id, exact_members = _exact_ten_group(tmp_path)
    with _store(tmp_path) as store:
        store.materialize_physical_attempt_group(
            group_id=exact_group_id,
            members=exact_members,
            leader_cell_id=exact_members[0].attempt.cell_id,
        )
        expected_groups = store.physical_attempt_groups()
        expected_commands = store.queued_commands()
        expected_events = tuple(
            tuple(row)
            for row in store._connection.execute(
                "SELECT * FROM watchdog_events ORDER BY event_id"
            ).fetchall()
        )

    _downgrade_group_members_to_exact_ten_only_schema_seven(database)

    with ExperimentOperatorStore(database, run_id="run-v03") as reopened:
        schema = str(
            reopened._connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'physical_attempt_group_members'
                """
            ).fetchone()[0]
        )
        assert "'serving'" in schema
        assert reopened.physical_attempt_groups() == expected_groups
        assert reopened.queued_commands() == expected_commands
        assert (
            tuple(
                tuple(row)
                for row in reopened._connection.execute(
                    "SELECT * FROM watchdog_events ORDER BY event_id"
                ).fetchall()
            )
            == expected_events
        )
        assert reopened._connection.execute("PRAGMA foreign_key_check").fetchall() == []

        serving_group_id, serving_members = _serving_group(tmp_path)
        assert len({member.command.command_sha256 for member in serving_members}) == 1
        assert (
            len(
                {member.attempt.scientific_command_sha256 for member in serving_members}
            )
            == 2
        )
        reopened.materialize_physical_attempt_group(
            group_id=serving_group_id,
            members=serving_members,
            leader_cell_id=serving_members[0].attempt.cell_id,
            group_kind="tp1_serving_session",
        )
        resident = reopened.physical_attempt_group_for_attempt(
            serving_members[1].attempt.cell_id, 1
        )
        assert resident is not None
        assert tuple(row["logical_kind"] for row in resident["members"]) == (
            "serving",
            "serving",
        )

    with ExperimentOperatorStore(database, run_id="run-v03") as restarted:
        resident = restarted.physical_attempt_group_for_attempt(
            serving_members[0].attempt.cell_id, 1
        )
        assert resident is not None
        assert resident["group_id"] == serving_group_id


def test_store_requires_wal_full_and_immutable_stage_plan(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        assert store.run_id == "run-v03"
        assert store.journal_mode.lower() == "wal"
        assert store.synchronous_mode == 2
        store.initialize_stage_plan(_plan())
        with pytest.raises(RuntimeError, match="replace an initialized stage plan"):
            store.initialize_stage_plan(
                (StagePlanEntry("preflight", 0, "preflight", "final", "wrong", 11),)
            )

    with pytest.raises(RuntimeError, match="belongs to run"):
        ExperimentOperatorStore(tmp_path / "operator.sqlite3", run_id="other-run")

    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError, match="run_id is required"):
        ExperimentOperatorStore(missing)
    assert not missing.exists()


def test_controller_journal_enforces_dag_coverage_and_reopens_artifacts(
    tmp_path: Path,
) -> None:
    database = tmp_path / "controller.sqlite3"
    plan = (
        StagePlanEntry("preflight", 0, "preflight", "final", "1", 1),
        StagePlanEntry("e3a", 1, "E3a", "selection", "0", 0),
    )

    def binding(name: str, payload: str) -> ControllerArtifactBinding:
        path = (tmp_path / name).resolve()
        path.write_text(payload, encoding="utf-8")
        return ControllerArtifactBinding.bind(path)

    preflight_materialization = binding("preflight-materialization.json", "m\n")
    preflight_node = binding("preflight-node.json", "n\n")
    preflight_source = binding("preflight-source.json", "s\n")
    preflight_decision = binding("preflight-decision.json", "d\n")
    preflight_completion = binding("preflight-completion.json", "c\n")
    e3a_materialization = binding("e3a-materialization.json", "em\n")
    e3a_node = binding("e3a-node.json", "en\n")
    e3a_source = binding("e3a-source.json", "es\n")
    e3a_decision = binding("e3a-decision.json", "ed\n")
    e3a_completion = binding("e3a-completion.json", "ec\n")

    with ExperimentOperatorStore(database, run_id="controller-run") as store:
        store.initialize_stage_plan(plan)
        with pytest.raises(AttemptTransitionError, match="predecessor"):
            store.record_controller_materialization(
                node="e3a",
                materialization=e3a_materialization,
                node_materialization=e3a_node,
                expected_cell_ids=(),
            )
        store.record_controller_materialization(
            node="preflight",
            materialization=preflight_materialization,
            node_materialization=preflight_node,
            expected_cell_ids=("preflight:compile",),
        )
        store.record_controller_execution_plan(
            node="preflight", execution_source=preflight_source
        )
        with pytest.raises(AttemptTransitionError, match="coverage"):
            store.record_controller_reduction(
                node="preflight",
                decision=preflight_decision,
                completion=preflight_completion,
            )
        _complete(store, _spec(tmp_path))
        store.record_controller_reduction(
            node="preflight",
            decision=preflight_decision,
            completion=preflight_completion,
        )
        store.record_controller_materialization(
            node="e3a",
            materialization=e3a_materialization,
            node_materialization=e3a_node,
            expected_cell_ids=(),
        )
        store.mark_controller_blocked(node="e3a", reason="fixture producer absent")
        store.resume_controller_node(node="e3a", reason="fixture producer restored")
        store.record_controller_execution_plan(node="e3a", execution_source=e3a_source)
        store.record_controller_reduction(
            node="e3a", decision=e3a_decision, completion=e3a_completion
        )
        assert tuple(row["state"] for row in store.controller_nodes()) == (
            "REDUCED",
            "REDUCED",
        )
        assert store.snapshot()["controller_nodes"][0]["expected_cell_count"] == 1

    with ExperimentOperatorStore(database) as reopened:
        assert reopened.controller_node("preflight")["decision_sha256"] == (
            preflight_decision.sha256
        )
        assert reopened.controller_node("e3a")["expected_cell_count"] == 0
        preflight_source_path = Path(preflight_source.absolute_path)
        preflight_source_path.write_text("tampered\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="differs from its durable binding"):
            reopened._reopen_controller_binding(preflight_source)


def test_known_coverage_and_retry_scientific_axes_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "bounded.sqlite3"
    with ExperimentOperatorStore(database, run_id="bounded") as store:
        store.initialize_stage_plan(
            (StagePlanEntry("preflight", 0, "preflight", "final", "1", 1),)
        )
        first = _spec(tmp_path)
        store.materialize_attempt(first)
        with pytest.raises(RuntimeError, match="exceed known stage coverage"):
            store.materialize_attempt(
                _spec(tmp_path, cell_id="preflight:unexpected-extra")
            )
        store.finish_attempt(
            first.cell_id,
            1,
            status="FAILED",
            exit_code=None,
            failure_code="PRELAUNCH_VALIDATION",
            retry_decision="RETRY_AFTER_FIX",
            included_in_analysis=False,
            exclusion_reason="fixture",
            finished_at_ns=10,
        )
        changed = CellAttemptSpec(
            **{
                **vars(_spec(tmp_path, attempt=2)),
                "scientific_axes": {"task": "changed", "topology": "tp2_dp1"},
            }
        )
        with pytest.raises(AttemptTransitionError, match="cannot change"):
            store.materialize_attempt(changed)


def test_sealed_infrastructure_retry_rebinds_paths_under_stable_scientific_digest(
    tmp_path: Path,
) -> None:
    scientific_digest = _sha("7")
    with _store(tmp_path) as store:
        first_spec, first_command = _queued_pair(
            tmp_path,
            cell_id="preflight:path-reseal",
        )
        first_spec = replace(
            first_spec,
            scientific_command_sha256=scientific_digest,
        )
        store.materialize_attempt(first_spec)
        store.enqueue_command(first_command)
        store.finish_attempt(
            first_spec.cell_id,
            first_spec.attempt,
            status="FAILED",
            exit_code=None,
            failure_code="INFRASTRUCTURE:HTTP_TRANSIENT",
            retry_decision="RETRY_INFRASTRUCTURE_AUTOMATIC",
            included_in_analysis=False,
            exclusion_reason="transient",
            finished_at_ns=10,
        )
        second_spec, second_command = _queued_pair(
            tmp_path,
            cell_id=first_spec.cell_id,
            attempt=2,
        )
        second_command = replace(
            second_command,
            argv=(*second_command.argv, "--attempt-owned-plan", "attempt-2.json"),
        )
        second_spec = replace(
            second_spec,
            command_sha256=second_command.command_sha256,
            scientific_command_sha256=scientific_digest,
        )
        assert second_command.command_sha256 != first_command.command_sha256
        store.materialize_attempt(second_spec)
        store.enqueue_command(second_command)
        assert (
            store.attempt(first_spec.cell_id, 2)["scientific_command_sha256"]
            == scientific_digest
        )


def test_changed_retry_command_requires_explicit_stable_scientific_digest(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        first_spec, first_command = _queued_pair(
            tmp_path,
            cell_id="preflight:no-scientific-digest",
        )
        store.materialize_attempt(first_spec)
        store.finish_attempt(
            first_spec.cell_id,
            first_spec.attempt,
            status="FAILED",
            exit_code=None,
            failure_code="INFRASTRUCTURE:HTTP_TRANSIENT",
            retry_decision="RETRY_INFRASTRUCTURE_AUTOMATIC",
            included_in_analysis=False,
            exclusion_reason="transient",
            finished_at_ns=10,
        )
        second_spec, second_command = _queued_pair(
            tmp_path,
            cell_id=first_spec.cell_id,
            attempt=2,
        )
        changed_command = replace(
            second_command,
            argv=(*second_command.argv, "--attempt-owned-plan", "attempt-2.json"),
        )
        second_spec = replace(
            second_spec,
            command_sha256=changed_command.command_sha256,
        )
        assert changed_command.command_sha256 != first_command.command_sha256
        with pytest.raises(
            AttemptTransitionError,
            match="stable path-independent scientific identity",
        ):
            store.materialize_attempt(second_spec)


def test_running_commit_precedes_injected_launcher_and_complete_is_evidence_gated(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operator.sqlite3"
    with _store(tmp_path) as store:
        spec = _spec(tmp_path)
        store.materialize_attempt(spec)

        def launcher() -> SpawnedProcess:
            with ExperimentOperatorStore(database) as observer:
                attempt = observer.attempt(spec.cell_id, 1)
                assert attempt["status"] == "RUNNING"
                assert attempt["pid"] is None
            return SpawnedProcess(pid=4242, pgid=4242)

        process = store.start_attempt_with_launcher(
            spec.cell_id,
            1,
            assigned_gpu_uuids=("GPU-AAA",),
            launcher=launcher,
            started_at_ns=10,
        )
        assert process.pid == 4242
        assert store.attempt(spec.cell_id, 1)["pid"] == 4242

        with pytest.raises(ValueError, match="terminal, JUnit, and raw-log"):
            store.finish_attempt(
                spec.cell_id,
                1,
                status="COMPLETE",
                exit_code=0,
                included_in_analysis=True,
                exclusion_reason=None,
                finished_at_ns=20,
            )
        assert store.attempt(spec.cell_id, 1)["status"] == "RUNNING"

        store.finish_attempt(
            spec.cell_id,
            1,
            status="COMPLETE",
            exit_code=0,
            terminal_sha256=_sha("e"),
            junit_sha256=_sha("f"),
            raw_log_sha256=_sha("1"),
            evidence_files={"raw.json": _sha("2")},
            included_in_analysis=True,
            exclusion_reason=None,
            finished_at_ns=20,
        )
        assert store.attempt(spec.cell_id, 1)["status"] == "COMPLETE"
        with pytest.raises(AttemptTransitionError, match="retry requires"):
            store.materialize_attempt(_spec(tmp_path, attempt=2))


def test_launcher_failure_stays_running_until_reconciled_before_retry(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        first = _spec(tmp_path)
        store.materialize_attempt(first)

        def fail() -> SpawnedProcess:
            raise OSError("fixture launch failure")

        with pytest.raises(OSError, match="fixture launch failure"):
            store.start_attempt_with_launcher(
                first.cell_id,
                1,
                assigned_gpu_uuids=("GPU-AAA",),
                launcher=fail,
                started_at_ns=10,
            )
        failed = store.attempt(first.cell_id, 1)
        assert failed["status"] == "RUNNING"
        assert failed["included_in_analysis"] is False

        store.finish_attempt(
            first.cell_id,
            1,
            status="FAILED",
            exit_code=None,
            failure_code="SPAWN_RECONCILED_NO_CHILD",
            retry_decision="RETRY_INFRASTRUCTURE",
            included_in_analysis=False,
            exclusion_reason="launcher_failed_before_child_creation",
            finished_at_ns=20,
        )

        second = _spec(tmp_path, attempt=2)
        store.materialize_attempt(second)
        assert store.attempt(second.cell_id, 2)["status"] == "PENDING"


def test_scheduler_recovers_running_attempt_without_pid_from_start_receipt(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-000",), _sha("a"))
        )
        spec, command = _queued_pair(tmp_path, cell_id="preflight:recover-start")
        store.materialize_attempt(spec)
        store.enqueue_command(command)
        store.mark_running_before_spawn(
            spec.cell_id,
            spec.attempt,
            assigned_gpu_uuids=("GPU-000",),
            started_at_ns=10,
        )
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "recover-start.lock",
            callbacks=replace(
                _scheduler_callbacks(
                    launch=lambda _command, _gpus: SpawnedProcess(88_001, 88_001)
                ),
                recover_started_process=lambda _command: RecoveredProcessStart(
                    pid=88_001,
                    pgid=88_001,
                    started_ns=11,
                    receipt_sha256=_sha("9"),
                ),
            ),
            clock_ns=lambda: 20,
        )
        cycle = daemon.run_once()
        recovered = store.attempt(spec.cell_id, spec.attempt)
        assert recovered["pid"] == 88_001
        assert recovered["pgid"] == 88_001
        assert recovered["process_start_receipt_sha256"] == _sha("9")
        assert cycle.reconciled == ((spec.cell_id, spec.attempt, "OBSERVED"),)
        assert any(
            row["event_type"] == "PROCESS_METADATA_RECOVERED_FROM_START_RECEIPT"
            for row in store._event_rows()
        )


def test_scheduler_terminalizes_missing_start_receipt_after_attach_grace(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-000",), _sha("a"))
        )
        spec, command = _queued_pair(tmp_path, cell_id="preflight:no-start")
        store.materialize_attempt(spec)
        store.enqueue_command(command)
        store.mark_running_before_spawn(
            spec.cell_id,
            spec.attempt,
            assigned_gpu_uuids=("GPU-000",),
            started_at_ns=1_000_000_000,
        )
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "missing-start.lock",
            callbacks=replace(
                _scheduler_callbacks(
                    launch=lambda _command, _gpus: SpawnedProcess(88_002, 88_002)
                ),
                recover_started_process=lambda _command: None,
            ),
            watchdog_policy=WatchdogPolicy(process_attach_grace_seconds=1),
            clock_ns=lambda: 3_000_000_000,
        )
        cycle = daemon.run_once()
        failed = store.attempt(spec.cell_id, spec.attempt)
        assert failed["status"] == "FAILED"
        assert failed["failure_code"] == ("INFRASTRUCTURE:START_RECEIPT_NOT_PUBLISHED")
        assert cycle.reconciled == (
            (spec.cell_id, spec.attempt, "START_NOT_PUBLISHED"),
        )


def test_stale_identity_preserves_attempt_and_excludes_it(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        spec = _spec(tmp_path)
        _complete(store, spec)
        store.mark_stale_identity(
            spec.cell_id,
            1,
            reason="source_tree_changed",
            marked_at_ns=40,
        )
        stale = store.attempt(spec.cell_id, 1)
        assert stale["status"] == "STALE_IDENTITY"
        assert stale["included_in_analysis"] is False
        assert stale["terminal_sha256"] == _sha("e")

        store.materialize_attempt(_spec(tmp_path, attempt=2))
        assert store.attempt(spec.cell_id, 2)["status"] == "PENDING"


def test_legacy_v02_import_is_forced_stale_and_does_not_consume_current_coverage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy.sqlite3"
    with ExperimentOperatorStore(database, run_id="v03") as store:
        store.initialize_stage_plan(
            (StagePlanEntry("preflight", 0, "preflight", "final", "1", 1),)
        )
        legacy_spec = _spec(tmp_path)
        assert (
            store.import_legacy_stale_attempts(
                (
                    LegacyStaleAttempt(
                        spec=legacy_spec,
                        original_status="RUNNING",
                        exclusion_reason="v02_source_patch_driver_identity_mismatch",
                        started_at_ns=10,
                        finished_at_ns=20,
                        raw_log_sha256=_sha("7"),
                        evidence_files={"v02/stdout.log": _sha("7")},
                    ),
                )
            )
            == 1
        )
        stale = store.attempt(legacy_spec.cell_id, 1)
        assert stale["status"] == "STALE_IDENTITY"
        assert stale["is_legacy_import"] is True
        assert stale["legacy_original_status"] == "RUNNING"
        assert stale["included_in_analysis"] is False

        replacement_identity = dict(legacy_spec.identity)
        replacement_identity["source_sha256"] = _sha("8")
        replacement = CellAttemptSpec(
            **{
                **vars(_spec(tmp_path, attempt=2)),
                "identity": replacement_identity,
                "command_sha256": _sha("9"),
            }
        )
        store.materialize_attempt(replacement)
        assert store.attempt(replacement.cell_id, 2)["is_legacy_import"] is False
        snapshot = store.snapshot()
        assert snapshot["stage_plan"][0]["materialized_cells"] == 1
        assert snapshot["stage_plan"][0]["stale"] == 1


def test_disk_dispatch_gate_requires_high_water_plus_15_gib_reserve(
    tmp_path: Path,
) -> None:
    reserve = 15 * 1024**3
    allowed = evaluate_dispatch_disk_gate(
        free_bytes=reserve + 123,
        predicted_next_wave_high_water_bytes=123,
    )
    assert allowed.action == "ALLOW"
    stopped = evaluate_dispatch_disk_gate(
        free_bytes=reserve + 122,
        predicted_next_wave_high_water_bytes=123,
    )
    assert stopped.action == "STOP"
    assert stopped.required_free_bytes == reserve + 123

    with _store(tmp_path) as store:
        live = store.check_dispatch_disk_capacity(
            monitored_path=tmp_path,
            predicted_next_wave_high_water_bytes=10**18,
        )
        assert live.action == "STOP"
        output = tmp_path / "disk-progress"
        store.export_progress(output)
    events = [
        json.loads(line)
        for line in (output / "watchdog_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-1]["event_type"] == "DISPATCH_STOP_DISK_HIGH_WATER"


def test_archive_callbacks_gate_remote_eviction_on_sha_and_rehydrate(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        spec = _spec(tmp_path)
        _complete(store, spec)
        request = ArchiveRequest(
            archive_id="archive-0001",
            safe_boundary="preflight:compile:attempt-1:terminal",
            remote_payload_root="/srv/lightcone/v03/wave-0001",
            local_partial_root=str((tmp_path / "archive.partial").resolve()),
            local_final_root=str((tmp_path / "archive").resolve()),
            remote_manifest_sha256=_sha("2"),
            cell_id=spec.cell_id,
            attempt=1,
        )
        store.register_archive_safe_boundary(request)
        with pytest.raises(AttemptTransitionError, match="requires local SHA"):
            store.authorize_remote_eviction(request.archive_id)

        calls: list[str] = []

        def transfer(
            callback_request: ArchiveRequest,
            previous: ArchiveStepReceipt | None,
        ) -> ArchiveStepReceipt:
            assert callback_request == request
            assert previous is None
            calls.append("transfer")
            return ArchiveStepReceipt("TRANSFER", _sha("2"), _sha("3"), 3, 100)

        def local_sha(
            callback_request: ArchiveRequest,
            previous: ArchiveStepReceipt | None,
        ) -> ArchiveStepReceipt:
            assert callback_request == request
            assert previous is not None and previous.step == "TRANSFER"
            calls.append("local_sha")
            return ArchiveStepReceipt("LOCAL_SHA_VERIFY", _sha("2"), _sha("4"), 3, 100)

        def rehydrate(
            callback_request: ArchiveRequest,
            previous: ArchiveStepReceipt | None,
        ) -> ArchiveStepReceipt:
            assert callback_request == request
            assert previous is not None and previous.step == "LOCAL_SHA_VERIFY"
            calls.append("rehydrate")
            return ArchiveStepReceipt(
                "REHYDRATE_VERIFY",
                _sha("2"),
                _sha("5"),
                3,
                100,
                content_tree_sha256=_sha("6"),
            )

        authorization = store.run_archive_callbacks(
            request,
            ArchiveCallbacks(transfer, local_sha, rehydrate),
        )
        assert calls == ["transfer", "local_sha", "rehydrate"]
        assert authorization.rehydrated_content_tree_sha256 == _sha("6")
        assert authorization.remote_payload_root == request.remote_payload_root
        assert store.archive_checkpoint(request.archive_id)["state"] == (
            "EVICTION_AUTHORIZED"
        )
        assert not hasattr(ArchiveCallbacks, "delete_remote")


def test_archive_mismatch_never_authorizes_remote_eviction(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        spec = _spec(tmp_path)
        _complete(store, spec)
        request = ArchiveRequest(
            archive_id="archive-mismatch",
            safe_boundary="preflight:terminal",
            remote_payload_root="/srv/lightcone/v03/wave",
            local_partial_root=str((tmp_path / "bad.partial").resolve()),
            local_final_root=str((tmp_path / "bad").resolve()),
            remote_manifest_sha256=_sha("2"),
            cell_id=spec.cell_id,
            attempt=1,
        )

        def wrong_manifest(
            _request: ArchiveRequest,
            _previous: ArchiveStepReceipt | None,
        ) -> ArchiveStepReceipt:
            return ArchiveStepReceipt("TRANSFER", _sha("a"), _sha("3"), 1, 1)

        unused = lambda _request, _previous: ArchiveStepReceipt(
            "LOCAL_SHA_VERIFY", _sha("2"), _sha("4"), 1, 1
        )
        with pytest.raises(RuntimeError, match="manifest differs"):
            store.run_archive_callbacks(
                request,
                ArchiveCallbacks(wrong_manifest, unused, unused),
            )
        checkpoint = store.archive_checkpoint(request.archive_id)
        assert checkpoint["state"] == "REGISTERED"
        assert checkpoint["eviction_authorized_at_ns"] is None


def test_watchdog_records_process_heartbeat_and_log_anomalies_once(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        spec = _spec(tmp_path)
        store.materialize_attempt(spec)
        store.mark_running_before_spawn(
            spec.cell_id,
            1,
            assigned_gpu_uuids=("GPU-AAA",),
            started_at_ns=1_000_000_000,
        )
        store.attach_process(spec.cell_id, 1, pid=99, pgid=99)
        store.record_heartbeat(
            spec.cell_id,
            1,
            pid=99,
            pgid=99,
            log_size_bytes=10,
            gpu_observation={"GPU-AAA": {"utilization_percent": 0}},
            observed_at_ns=2_000_000_000,
        )
        policy = WatchdogPolicy(
            process_attach_grace_seconds=1,
            heartbeat_timeout_seconds=5,
            log_stall_timeout_seconds=5,
            event_repeat_seconds=60,
            minimum_free_disk_bytes=0,
        )

        def dead(pid: int, expected_pgid: int) -> ProcessObservation:
            assert (pid, expected_pgid) == (99, 99)
            return ProcessObservation(
                pid, False, None, "process_not_found", exit_code=9
            )

        findings = store.watchdog_once(
            policy=policy,
            process_probe=dead,
            now_ns=10_000_000_000,
        )
        assert {finding.event_type for finding in findings} == {
            "PROCESS_NOT_ALIVE",
            "HEARTBEAT_STALE",
            "LOG_STALLED",
        }
        assert (
            store.watchdog_once(
                policy=policy,
                process_probe=dead,
                now_ns=11_000_000_000,
            )
            == ()
        )
        assert store.attempt(spec.cell_id, 1)["status"] == "RUNNING"


def test_singleton_lock_is_nonblocking_and_reports_holder(tmp_path: Path) -> None:
    path = tmp_path / "operator.lock"
    first = SingletonOperatorLock(path)
    second = SingletonOperatorLock(path)
    first.acquire()
    try:
        with pytest.raises(OperatorAlreadyRunningError, match="another formal"):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


def test_atomic_exports_are_complete_projections_of_sqlite(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        spec = _spec(tmp_path)
        _complete(store, spec)
        store.record_selection_decision(
            decision_id="preflight-envelope",
            stage="preflight",
            phase="final",
            decision_kind="interference_gate",
            source_sha256=_sha("3"),
            decision={"headline_parallelism": "isolated"},
            occurred_at_ns=40,
        )
        store.record_watchdog_event(
            event_type="CHECKPOINT_PUBLISHED",
            severity="INFO",
            cell_id=spec.cell_id,
            attempt=1,
            payload={"checkpoint": 1},
            occurred_at_ns=41,
        )
        store.record_metric(
            MetricRecord(
                stage="preflight",
                phase="final",
                cell_id=spec.cell_id,
                attempt=1,
                metric_name="paired_goodput_effect",
                metric_kind="headline",
                point_estimate=0.002,
                ci_low=-0.004,
                ci_high=0.008,
                independent_block_count=2,
                request_count=16,
                paired=True,
                reducer_method="paired_bca_bootstrap_v1",
                attributes={"confidence_level": 0.95},
            ),
            recorded_at_ns=42,
        )
        output = tmp_path / "progress"
        manifest = store.export_progress(output, exported_at_ns=50)

        expected = {
            "stage_plan.csv",
            "cell_ledger.csv",
            "controller_state.csv",
            "stage_summary.csv",
            "selection_decisions.jsonl",
            "watchdog_events.jsonl",
            "dashboard.md",
            "metrics_long.parquet",
            "instance_billing.csv",
        }
    assert set(manifest.files) == expected
    assert all(len(digest) == 64 for digest in manifest.files.values())
    assert (output / "export_manifest.json").is_file()
    assert not list(output.glob("*.partial"))

    with (output / "cell_ledger.csv").open(newline="", encoding="utf-8") as handle:
        ledger = list(csv.DictReader(handle))
    assert ledger[0]["status"] == "COMPLETE"
    assert ledger[0]["included_in_analysis"] == "True"
    with (output / "stage_plan.csv").open(newline="", encoding="utf-8") as handle:
        stages = list(csv.DictReader(handle))
    assert stages[0]["completed"] == "1"
    assert float(stages[0]["actual_gpu_hours"]) == pytest.approx(20 / 3600)
    assert pq.read_table(output / "metrics_long.parquet").num_rows == 1
    assert "preflight:compile" in (output / "dashboard.md").read_text(
        encoding="utf-8"
    ) or "preflight" in (output / "dashboard.md").read_text(encoding="utf-8")


def test_provider_samples_reduce_exact_whole_instance_billed_hours(
    tmp_path: Path,
) -> None:
    running = ProviderRuntimeSample(
        instance_uuid="pro-fixture",
        state="running",
        observed_at_ns=5_000_000_000,
        provider_started_at_ns=1_000_000_000,
        provider_stopped_at_ns=None,
        gpu_count=2,
        response_sha256=_sha("a"),
    )
    shutdown = ProviderRuntimeSample(
        instance_uuid="pro-fixture",
        state="shutdown",
        observed_at_ns=12_000_000_000,
        provider_started_at_ns=1_000_000_000,
        provider_stopped_at_ns=11_000_000_000,
        gpu_count=2,
        response_sha256=_sha("b"),
    )
    with _store(tmp_path) as store:
        store.record_provider_runtime_sample(running)
        assert store.whole_instance_billed_gpu_seconds() == pytest.approx(8.0)
        with pytest.raises(Exception, match="open provider interval"):
            store.whole_instance_billed_gpu_seconds(require_complete=True)
        store.record_provider_runtime_sample(shutdown)
        store.record_provider_runtime_sample(shutdown)
        intervals = store.provider_billing_intervals()
        assert len(intervals) == 1
        assert intervals[0]["complete"] is True
        assert intervals[0]["sample_count"] == 2
        assert intervals[0]["duration_seconds"] == pytest.approx(10.0)
        assert store.whole_instance_billed_gpu_seconds(
            require_complete=True
        ) == pytest.approx(20.0)
        output = tmp_path / "billing-progress"
        store.export_progress(output, exported_at_ns=13_000_000_000)
        billing = list(
            csv.DictReader(
                (output / "instance_billing.csv").open(newline="", encoding="utf-8")
            )
        )
        assert billing[0]["whole_instance_billed_gpu_seconds"] == "20.0"
        assert "COMPLETE" in (output / "dashboard.md").read_text(encoding="utf-8")


def test_registered_default_stage_plan_has_all_21_nodes() -> None:
    plan = default_formal_stage_plan()
    assert len(plan) == 21
    assert tuple(entry.ordinal for entry in plan) == tuple(range(21))
    assert plan[0].node == "preflight"
    assert plan[-1].node == "e0_final"
    assert plan[15].expected_formula == "450*N_E5+264"


def test_scheduler_queue_accepts_only_pre_materialized_exact_argv(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        spec, command = _queued_pair(tmp_path, cell_id="preflight:queue")
        with pytest.raises(KeyError, match="unknown attempt"):
            store.enqueue_command(command)
        mismatched = CellAttemptSpec(**{**vars(spec), "command_sha256": _sha("9")})
        store.materialize_attempt(mismatched)
        with pytest.raises(AttemptTransitionError, match="queued argv differs"):
            store.enqueue_command(command)

    other = tmp_path / "exact"
    other.mkdir()
    with _store(other) as store:
        spec, command = _queued_pair(other, cell_id="preflight:queue")
        store.materialize_attempt(spec)
        store.enqueue_command(command)
        assert store.queued_commands() == (command,)


@pytest.mark.parametrize(
    ("mode", "timing_class", "required_gpu_count", "cell_count", "expected"),
    (
        ("ISOLATED", "HEADLINE", 1, 2, 1),
        ("DUAL_SINGLE", "HEADLINE", 1, 2, 2),
        ("DUAL_SINGLE", "EXCLUSIVE", 2, 1, 1),
        ("UNRESOLVED", "HEADLINE", 1, 1, 0),
    ),
)
def test_scheduler_authorizes_isolated_dual_single_and_gpu_gang(
    tmp_path: Path,
    mode: str,
    timing_class: str,
    required_gpu_count: int,
    cell_count: int,
    expected: int,
) -> None:
    launches = []
    clock = _TickClock()
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope(
                mode=mode,
                gpu_uuids=("GPU-0", "GPU-1"),
                evidence_sha256=_sha("a"),
            )
        )
        for index in range(cell_count):
            spec, command = _queued_pair(
                tmp_path,
                cell_id=f"preflight:schedule-{index}",
                timing_class=timing_class,
                required_gpu_count=required_gpu_count,
            )
            store.materialize_attempt(spec)
            store.enqueue_command(command)

        def launch(command, gpu_uuids):
            attempt = store.attempt(command.cell_id, command.attempt)
            assert attempt["status"] == "RUNNING"
            assert attempt["pid"] is None
            launches.append((command.cell_id, gpu_uuids))
            pid = 10_000 + len(launches)
            return SpawnedProcess(pid, pid)

        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "scheduler.lock",
            callbacks=_scheduler_callbacks(launch=launch),
            clock_ns=clock,
        )
        result = daemon.run_once()
        assert len(result.dispatched) == expected
        assert len(launches) == expected
        if timing_class == "EXCLUSIVE" and expected:
            assert launches[0][1] == ("GPU-0", "GPU-1")


def test_scheduler_fans_one_preflight_parent_into_exact_ten_logical_attempts(
    tmp_path: Path,
) -> None:
    clock = _TickClock()
    group_id, members = _exact_ten_group(tmp_path)
    launches: list[tuple[str, tuple[str, ...]]] = []
    validations: list[str] = []
    evidence_started = clock.value
    evidence_finished = evidence_started + 1_000_000_000
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("UNRESOLVED", ("GPU-0", "GPU-1"), _sha("a"))
        )
        store.materialize_physical_attempt_group(
            group_id=group_id,
            members=members,
            leader_cell_id=members[0].attempt.cell_id,
        )
        assert len(store.queued_commands(status="PENDING")) == 10
        assert store.physical_commands(status="PENDING") == (members[0].command,)

        def launch(command, gpu_uuids):
            launches.append((command.cell_id, gpu_uuids))
            rows = tuple(store.attempt(member.attempt.cell_id, 1) for member in members)
            assert {row["status"] for row in rows} == {"RUNNING"}
            assert {row["pid"] for row in rows} == {None}
            assert {tuple(row["assigned_gpu_uuids"]) for row in rows} == {
                ("GPU-0", "GPU-1")
            }
            return SpawnedProcess(31_000, 31_000)

        terminal = TerminalEvidence(
            status="COMPLETE",
            exit_code=0,
            atomic_publication_sha256=_sha("2"),
            terminal_sha256=_sha("3"),
            junit_sha256=_sha("4"),
            raw_log_sha256=_sha("5"),
            evidence_files={"exact-ten-execution.json": _sha("2")},
            started_ns=evidence_started,
            finished_ns=evidence_finished,
        )

        def validate(command, _attempt, _process):
            validations.append(command.cell_id)
            return terminal

        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=launch,
                process_probe=lambda pid, _pgid: ProcessObservation(
                    pid, False, None, "exited", exit_code=0
                ),
                terminal_validator=validate,
            ),
            clock_ns=clock,
        )
        dispatched = daemon.run_once()
        assert len(dispatched.dispatched) == 1
        assert launches == [(members[0].attempt.cell_id, ("GPU-0", "GPU-1"))]
        running = tuple(store.attempt(member.attempt.cell_id, 1) for member in members)
        assert {row["pid"] for row in running} == {31_000}
        assert {row["pgid"] for row in running} == {31_000}

        reconciled = daemon.run_once()
        assert len(reconciled.reconciled) == 10
        assert validations == [member.attempt.cell_id for member in members]
        complete = tuple(store.attempt(member.attempt.cell_id, 1) for member in members)
        assert {row["status"] for row in complete} == {"COMPLETE"}
        assert sum(row["compute_gpu_seconds"] for row in complete) == pytest.approx(2.0)
        assert sum(row["reserved_gpu_seconds"] for row in complete) == pytest.approx(
            2.0
        )
        assert [row["compute_gpu_seconds"] > 0 for row in complete] == [
            True,
            *([False] * 9),
        ]
        group = store.physical_attempt_groups()[0]
        assert group["status"] == "COMPLETE"
        assert group["shared_evidence_sha256"] == _sha("2")


def test_scheduler_fans_resident_group_only_after_shared_close_and_charges_once(
    tmp_path: Path,
) -> None:
    clock = _TickClock()
    group_id, members = _serving_group(tmp_path)
    physical_started = clock.value
    physical_finished = physical_started + 2_000_000_000
    validations: list[str] = []
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        store.materialize_physical_attempt_group(
            group_id=group_id,
            members=members,
            leader_cell_id=members[0].attempt.cell_id,
            group_kind="tp1_serving_session",
        )

        def validate(command, _attempt, _process):
            validations.append(command.cell_id)
            return TerminalEvidence(
                status="COMPLETE",
                exit_code=0,
                atomic_publication_sha256=_sha("7"),
                terminal_sha256=_sha(str(len(validations))),
                junit_sha256=_sha("4"),
                raw_log_sha256=_sha("5"),
                evidence_files={"shared-close.json": _sha("7")},
                started_ns=physical_started,
                finished_ns=physical_finished,
            )

        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "resident.scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=lambda _command, _gpu: SpawnedProcess(33_000, 33_000),
                process_probe=lambda pid, _pgid: ProcessObservation(
                    pid, False, None, "exited", exit_code=0
                ),
                terminal_validator=validate,
            ),
            clock_ns=clock,
        )
        assert len(daemon.run_once().dispatched) == 1
        reconciled = daemon.run_once()
        assert len(reconciled.reconciled) == 2
        assert validations == [member.attempt.cell_id for member in members]
        complete = tuple(store.attempt(member.attempt.cell_id, 1) for member in members)
        assert {row["status"] for row in complete} == {"COMPLETE"}
        assert sum(row["compute_gpu_seconds"] for row in complete) == pytest.approx(2.0)
        assert [row["compute_gpu_seconds"] > 0 for row in complete] == [True, False]
        assert store.physical_attempt_groups()[0]["shared_evidence_sha256"] == _sha("7")


def test_resident_group_retries_only_typed_infrastructure_member_fresh(
    tmp_path: Path,
) -> None:
    clock = _TickClock()
    group_id, members = _serving_group(tmp_path)
    by_cell = {member.attempt.cell_id: member for member in members}
    physical_started = clock.value
    physical_finished = physical_started + 2_000_000_000

    def retry_builder(current: QueuedCommandSpec, attempt: int):
        source = by_cell[current.cell_id]
        root = (tmp_path / "fresh" / current.cell_id.replace(":", "-")).resolve()
        command = QueuedCommandSpec(
            cell_id=current.cell_id,
            attempt=attempt,
            argv=(sys.executable, "-c", "pass"),
            launch_compatibility_key=current.launch_compatibility_key,
            required_gpu_count=1,
            timing_class="HEADLINE",
            predicted_high_water_bytes=current.predicted_high_water_bytes,
            monitored_path=str(root),
            log_path=str(root / "command.log"),
            expected_terminal_path=str(root / "terminal.json"),
            expected_junit_path=str(root / "junit.xml"),
            expected_raw_log_path=str(root / "raw.json"),
            atomic_pointer_path=str(root / "pointer.json"),
            child_exit_receipt_path=str(root / "exit.json"),
        )
        specification = replace(
            source.attempt,
            attempt=attempt,
            command_sha256=command.command_sha256,
            output_directory=str(root / "output"),
        )
        return specification, command

    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        store.materialize_physical_attempt_group(
            group_id=group_id,
            members=members,
            leader_cell_id=members[0].attempt.cell_id,
            group_kind="tp1_serving_session",
        )

        def validate(command, _attempt, _process):
            infrastructure = command.cell_id == members[0].attempt.cell_id
            return TerminalEvidence(
                status="FAILED",
                exit_code=70,
                atomic_publication_sha256=_sha("7"),
                terminal_sha256=(_sha("8") if infrastructure else _sha("9")),
                junit_sha256=_sha("4"),
                raw_log_sha256=_sha("5"),
                evidence_files={"shared-close.json": _sha("7")},
                failure_class=("INFRASTRUCTURE" if infrastructure else "EXACTNESS"),
                failure_code=(
                    "RESIDENT_TRANSPORT_CLOSED"
                    if infrastructure
                    else "TOKEN_TRAJECTORY_MISMATCH"
                ),
                exclusion_reason="resident member failed",
                included_in_analysis=False,
                started_ns=physical_started,
                finished_ns=physical_finished,
            )

        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "resident-typed-failure.scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=lambda _command, _gpu: SpawnedProcess(33_100, 33_100),
                process_probe=lambda pid, _pgid: ProcessObservation(
                    pid, False, None, "exited", exit_code=70
                ),
                terminal_validator=validate,
                retry_builder=retry_builder,
            ),
            clock_ns=clock,
        )
        assert len(daemon.run_once().dispatched) == 1
        daemon.run_once()

        infrastructure = store.attempt(members[0].attempt.cell_id, 1)
        exactness = store.attempt(members[1].attempt.cell_id, 1)
        assert infrastructure["retry_decision"] == ("RETRY_INFRASTRUCTURE_AUTOMATIC")
        assert exactness["retry_decision"] == "NO_SCIENTIFIC_RETRY"
        assert store.attempt(members[0].attempt.cell_id, 2)["status"] in {
            "PENDING",
            "RUNNING",
        }
        assert (
            store.physical_attempt_group_for_attempt(members[0].attempt.cell_id, 2)
            is None
        )
        with pytest.raises(KeyError):
            store.attempt(members[1].attempt.cell_id, 2)


def test_unsealed_resident_wrapper_exit_retries_each_member_fresh_not_grouped(
    tmp_path: Path,
) -> None:
    clock = _TickClock()
    group_id, members = _serving_group(tmp_path)
    by_cell = {member.attempt.cell_id: member for member in members}
    launches: list[tuple[str, int]] = []
    resident_alive = {41_000: True}
    killed: list[int] = []

    def kill_resident(_command: QueuedCommandSpec, _pid: int, _pgid: int) -> None:
        killed.append(41_000)
        resident_alive[41_000] = False

    def retry_builder(current: QueuedCommandSpec, attempt: int):
        source = by_cell[current.cell_id]
        root = (tmp_path / "fresh" / current.cell_id.replace(":", "-")).resolve()
        command = QueuedCommandSpec(
            cell_id=current.cell_id,
            attempt=attempt,
            argv=(sys.executable, "-c", "pass"),
            launch_compatibility_key=current.launch_compatibility_key,
            required_gpu_count=1,
            timing_class="HEADLINE",
            predicted_high_water_bytes=current.predicted_high_water_bytes,
            monitored_path=str(root),
            log_path=str(root / "command.log"),
            expected_terminal_path=str(root / "terminal.json"),
            expected_junit_path=str(root / "junit.xml"),
            expected_raw_log_path=str(root / "raw.json"),
            atomic_pointer_path=str(root / "pointer.json"),
            child_exit_receipt_path=str(root / "exit.json"),
        )
        specification = replace(
            source.attempt,
            attempt=attempt,
            command_sha256=command.command_sha256,
            output_directory=str(root / "output"),
        )
        return specification, command

    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        store.materialize_physical_attempt_group(
            group_id=group_id,
            members=members,
            leader_cell_id=members[0].attempt.cell_id,
            group_kind="tp1_serving_session",
        )
        callbacks = replace(
            _scheduler_callbacks(
                launch=lambda command, _gpu: (
                    launches.append((command.cell_id, command.attempt))
                    or SpawnedProcess(34_000 + len(launches), 34_000 + len(launches))
                ),
                process_probe=lambda pid, _pgid: ProcessObservation(
                    pid, False, None, "unsealed wrapper exited", exit_code=17
                ),
                terminal_validator=lambda _command, _attempt, _process: None,
                retry_builder=retry_builder,
            ),
            independent_process_groups=lambda _command: (41_000,),
            process_group_alive=lambda pgid: resident_alive.get(pgid, False),
            send_term=lambda _command, _pid, _pgid: None,
            send_kill=kill_resident,
        )
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "resident-crash.scheduler.lock",
            callbacks=callbacks,
            clock_ns=clock,
        )
        assert len(daemon.run_once().dispatched) == 1
        orphan_cycle = daemon.run_once()
        assert orphan_cycle.reconciled[-1][2] == "RESIDENT_KILL_SENT"
        assert killed == [41_000]
        assert all(
            store.attempt(member.attempt.cell_id, 1)["status"] == "RUNNING"
            for member in members
        )
        daemon.run_once()

        for member in members:
            failed = store.attempt(member.attempt.cell_id, 1)
            assert failed["status"] == "FAILED"
            assert failed["failure_code"] == (
                "INFRASTRUCTURE:RESIDENT_GROUP_UNSEALED_EXIT"
            )
            assert failed["retry_decision"] == "RETRY_INFRASTRUCTURE_AUTOMATIC"
            retry = store.attempt(member.attempt.cell_id, 2)
            assert retry["status"] in {"PENDING", "RUNNING"}
            assert (
                store.physical_attempt_group_for_attempt(member.attempt.cell_id, 2)
                is None
            )
        assert store.physical_attempt_groups()[0]["status"] == "FAILED"


def test_preflight_group_rejects_incomplete_coverage_and_mixed_publication(
    tmp_path: Path,
) -> None:
    group_id, members = _exact_ten_group(tmp_path)
    with _store(tmp_path) as store:
        with pytest.raises(ValueError, match="ten uniquely sorted"):
            store.materialize_physical_attempt_group(
                group_id=group_id,
                members=members[:-1],
                leader_cell_id=members[0].attempt.cell_id,
            )
        store.materialize_physical_attempt_group(
            group_id=group_id,
            members=members,
            leader_cell_id=members[0].attempt.cell_id,
        )
        store.start_physical_attempt_group_with_launcher(
            group_id,
            assigned_gpu_uuids=("GPU-0", "GPU-1"),
            launcher=lambda: SpawnedProcess(32_000, 32_000),
            started_at_ns=10,
        )
        terminals = {
            member.attempt.cell_id: TerminalEvidence(
                status="COMPLETE",
                exit_code=0,
                atomic_publication_sha256=(_sha("9") if index == 9 else _sha("8")),
                terminal_sha256=_sha("3"),
                junit_sha256=_sha("4"),
                raw_log_sha256=_sha("5"),
                started_ns=11,
                finished_ns=20,
            )
            for index, member in enumerate(members)
        }
        with pytest.raises(ValueError, match="one shared publication"):
            store.finish_physical_attempt_group(
                group_id,
                terminals=terminals,
                finished_at_ns=30,
            )
        assert {
            store.attempt(member.attempt.cell_id, 1)["status"] for member in members
        } == {"RUNNING"}


def test_resident_group_records_no_automatic_retry_when_builder_is_disabled(
    tmp_path: Path,
) -> None:
    group_id, members = _serving_group(tmp_path)
    with _store(tmp_path) as store:
        store.materialize_physical_attempt_group(
            group_id=group_id,
            members=members,
            leader_cell_id=members[0].attempt.cell_id,
            group_kind="tp1_serving_session",
        )
        store.start_physical_attempt_group_with_launcher(
            group_id,
            assigned_gpu_uuids=("GPU-0",),
            launcher=lambda: SpawnedProcess(32_100, 32_100),
            started_at_ns=10,
        )
        terminals = {
            member.attempt.cell_id: TerminalEvidence(
                status="FAILED",
                exit_code=70,
                atomic_publication_sha256=_sha("8"),
                terminal_sha256=_sha("3"),
                junit_sha256=_sha("4"),
                raw_log_sha256=_sha("5"),
                failure_class="INFRASTRUCTURE",
                failure_code="FIXTURE_FAILURE",
                included_in_analysis=False,
                exclusion_reason="fixture infrastructure failure",
                started_ns=11,
                finished_ns=20,
            )
            for member in members
        }
        store.finish_physical_attempt_group(
            group_id,
            terminals=terminals,
            automatic_infrastructure_retry=False,
            finished_at_ns=30,
        )

        assert {
            store.attempt(member.attempt.cell_id, 1)["retry_decision"]
            for member in members
        } == {"NO_RETRY_BUILDER_OR_LIMIT"}


def test_scheduler_disk_stop_is_persistent_and_never_spawns(
    tmp_path: Path,
) -> None:
    launches = []
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        spec, command = _queued_pair(tmp_path, cell_id="preflight:disk-stop")
        store.materialize_attempt(spec)
        store.enqueue_command(command)
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=lambda command, gpu_uuids: launches.append((command, gpu_uuids)),
                free_disk_bytes=lambda _path: 15 * 1024**3,
            ),
            clock_ns=_TickClock(),
        )
        first = daemon.run_once()
        second = daemon.run_once()
        assert first.dispatch_state == "STOP"
        assert second.dispatch_state == "STOP"
        assert launches == []
        assert store.attempt(spec.cell_id, 1)["status"] == "PENDING"


def test_scheduler_counts_running_wave_before_second_dispatch(
    tmp_path: Path,
) -> None:
    launches: list[str] = []
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("DUAL_SINGLE", ("GPU-0", "GPU-1"), _sha("a"))
        )
        pairs = [
            _queued_pair(tmp_path, cell_id=f"preflight:capacity-wave-{index}")
            for index in range(2)
        ]
        for specification, original in pairs:
            command = replace(original, predicted_high_water_bytes=16 * 1024**3)
            specification = replace(
                specification,
                command_sha256=command.command_sha256,
            )
            store.materialize_attempt(specification)
            store.enqueue_command(command)

        def launch(command: QueuedCommandSpec, _gpu_uuids: tuple[str, ...]):
            launches.append(command.cell_id)
            pid = 30_000 + len(launches)
            return SpawnedProcess(pid, pid)

        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "capacity-wave.scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=launch,
                free_disk_bytes=lambda _path: 41_400_000_000,
            ),
            clock_ns=_TickClock(),
        )
        cycle = daemon.run_once()
        assert len(cycle.dispatched) == 1
        assert launches == ["preflight:capacity-wave-0"]
        assert cycle.dispatch_state == "STOP"
        assert cycle.stop_reason is not None
        assert cycle.stop_reason.startswith("disk_high_water_gate:")
        assert store.attempt("preflight:capacity-wave-0", 1)["status"] == "RUNNING"
        assert store.attempt("preflight:capacity-wave-1", 1)["status"] == "PENDING"


def test_scheduler_accepts_atomic_terminal_then_dispatches_next_compatible_cell(
    tmp_path: Path,
) -> None:
    clock = _TickClock()
    launches = []
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        pairs = [
            _queued_pair(tmp_path, cell_id=f"preflight:terminal-{index}")
            for index in range(2)
        ]
        for spec, command in pairs:
            store.materialize_attempt(spec)
            store.enqueue_command(command)

        def launch(command, _gpu_uuids):
            launches.append(command.cell_id)
            pid = 20_000 + len(launches)
            return SpawnedProcess(pid, pid)

        def dead(pid, _pgid):
            return ProcessObservation(pid, False, None, "exited", exit_code=0)

        terminal = TerminalEvidence(
            status="COMPLETE",
            exit_code=0,
            atomic_publication_sha256=_sha("2"),
            terminal_sha256=_sha("3"),
            junit_sha256=_sha("4"),
            raw_log_sha256=_sha("5"),
            evidence_files={"raw.json": _sha("6")},
        )
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=launch,
                process_probe=dead,
                terminal_validator=lambda _command, _attempt, _process: terminal,
            ),
            clock_ns=clock,
        )
        first = daemon.run_once()
        assert len(first.dispatched) == 1
        second = daemon.run_once()
        assert second.reconciled[0][2] == "COMPLETE"
        assert len(second.dispatched) == 1
        assert store.attempt(pairs[0][0].cell_id, 1)["status"] == "COMPLETE"
        assert store.attempt(pairs[1][0].cell_id, 1)["status"] == "RUNNING"


def test_scheduler_records_process_log_and_gpu_without_forging_child_heartbeat(
    tmp_path: Path,
) -> None:
    clock = _TickClock()
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        spec, command = _queued_pair(tmp_path, cell_id="preflight:heartbeat")
        store.materialize_attempt(spec)
        store.enqueue_command(command)
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "scheduler.lock",
            callbacks=SchedulerCallbacks(
                launch=lambda _command, _gpus: SpawnedProcess(25_000, 25_000),
                process_probe=lambda pid, pgid: ProcessObservation(
                    pid,
                    True,
                    pgid,
                    "alive",
                ),
                log_size_bytes=lambda _command: 4_096,
                gpu_snapshot=lambda gpu_uuids: {
                    gpu_uuids[0]: {
                        "utilization_percent": 97,
                        "memory_bytes": 1_024,
                    }
                },
                terminal_validator=lambda _command, _attempt, _process: None,
                free_disk_bytes=lambda _path: 100 * 1024**3,
            ),
            clock_ns=clock,
        )
        daemon.run_once()
        cycle = daemon.run_once()
        attempt = store.attempt(spec.cell_id, 1)
        assert cycle.reconciled == ((spec.cell_id, 1, "OBSERVED"),)
        assert attempt["heartbeat_sequence"] == 0
        assert attempt["heartbeat_at_ns"] is None
        assert attempt["last_log_size_bytes"] == 4_096
        assert attempt["gpu_observation"]["GPU-0"]["utilization_percent"] == 97


def test_scheduler_runtime_cap_persists_intent_then_terms_wrapper(
    tmp_path: Path,
) -> None:
    clock = _TickClock()
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        spec, original = _queued_pair(tmp_path, cell_id="preflight:runtime-cap")
        command = replace(
            original,
            max_runtime_seconds=1,
            max_log_stall_seconds=1,
        )
        spec = replace(spec, command_sha256=command.command_sha256)
        store.materialize_attempt(spec)
        store.enqueue_command(command)
        sent: list[tuple[int, int]] = []
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "scheduler.lock",
            callbacks=replace(
                _scheduler_callbacks(
                    launch=lambda _command, _gpus: SpawnedProcess(26_000, 26_000)
                ),
                send_term=lambda _command, pid, pgid: sent.append((pid, pgid)),
            ),
            clock_ns=clock,
        )
        daemon.run_once()
        result = daemon.run_once()
        for _ in range(4):
            if result.dispatch_state == "STOP":
                break
            result = daemon.run_once()
        assert result.dispatch_state == "STOP"
        assert result.stop_reason == "command_runtime_limit_exceeded"
        running = store.attempt(spec.cell_id, 1)
        assert running["status"] == "RUNNING"
        assert running["termination_reason"] == "SOURCE_BOUND_RUNTIME_LIMIT"
        assert running["term_sent_at_ns"] is not None
        assert sent == [(26_000, 26_000)]
        events = store.snapshot()["attempts"]
        assert len(events) == 1
        watchdog = store._event_rows()
        matches = [
            row
            for row in watchdog
            if row["event_type"] == "COMMAND_RUNTIME_LIMIT_EXCEEDED"
        ]
        assert len(matches) == 1
        assert matches[0]["payload"]["termination_intent_persisted"] is True


def test_scheduler_log_stall_warns_without_stopping_or_signalling(
    tmp_path: Path,
) -> None:
    clock = _TickClock()
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        spec, original = _queued_pair(tmp_path, cell_id="preflight:log-stall")
        command = replace(
            original,
            max_runtime_seconds=100,
            max_log_stall_seconds=1,
        )
        spec = replace(spec, command_sha256=command.command_sha256)
        store.materialize_attempt(spec)
        store.enqueue_command(command)
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=lambda _command, _gpus: SpawnedProcess(26_001, 26_001)
            ),
            clock_ns=clock,
        )
        daemon.run_once()
        for _ in range(4):
            result = daemon.run_once()
        assert result.dispatch_state == "RUN"
        assert store.attempt(spec.cell_id, 1)["termination_reason"] is None
        warnings = [
            row
            for row in store._event_rows()
            if row["event_type"] == "COMMAND_LOG_STALL_WARNING"
        ]
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "WARNING"


def test_scheduler_stale_child_heartbeat_stops_dispatch_without_signal(
    tmp_path: Path,
) -> None:
    clock = _TickClock()
    signals: list[str] = []
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        pairs = tuple(
            _queued_pair(tmp_path, cell_id=f"preflight:heartbeat-stall-{index}")
            for index in range(2)
        )
        for spec, original in pairs:
            command = replace(
                original,
                max_runtime_seconds=100,
                max_log_stall_seconds=100,
            )
            store.materialize_attempt(
                replace(spec, command_sha256=command.command_sha256)
            )
            store.enqueue_command(command)
        callbacks = replace(
            _scheduler_callbacks(
                launch=lambda _command, _gpus: SpawnedProcess(26_002, 26_002)
            ),
            worker_heartbeat=lambda _command: None,
            worker_heartbeat_required=lambda _command: True,
            send_term=lambda *_args: signals.append("TERM"),
            send_kill=lambda *_args: signals.append("KILL"),
        )
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "scheduler.lock",
            callbacks=callbacks,
            watchdog_policy=WatchdogPolicy(
                heartbeat_timeout_seconds=1,
                event_repeat_seconds=60,
            ),
            clock_ns=clock,
            sleeper=lambda _seconds: None,
        )
        results = daemon.run_forever(max_cycles=5)
        result = results[-1]

        assert result.dispatch_state == "STOP"
        assert result.reconciled == ((pairs[0][0].cell_id, 1, "STOP_GATE"),)
        assert result.dispatched == ()
        assert store.attempt(pairs[0][0].cell_id, 1)["status"] == "RUNNING"
        assert store.attempt(pairs[0][0].cell_id, 1)["termination_reason"] is None
        assert store.attempt(pairs[1][0].cell_id, 1)["status"] == "PENDING"
        assert signals == []
        warnings = [
            row
            for row in store._event_rows()
            if row["event_type"] == "CHILD_HEARTBEAT_STALE"
        ]
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "CRITICAL"
        assert warnings[0]["payload"]["hard_termination"] is False
        with SingletonOperatorLock(tmp_path / "scheduler.lock"):
            pass


def test_scheduler_escalates_term_to_group_kill_then_seals_infrastructure(
    tmp_path: Path,
) -> None:
    alive = {"value": True}
    signals: list[str] = []
    times = iter((3_000_000_000, 5_000_000_000, 6_000_000_000))
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        spec, original = _queued_pair(tmp_path, cell_id="preflight:term-kill")
        command = replace(
            original,
            max_runtime_seconds=1,
            max_log_stall_seconds=1,
        )
        spec = replace(spec, command_sha256=command.command_sha256)
        store.materialize_attempt(spec)
        store.enqueue_command(command)
        store.mark_running_before_spawn(
            spec.cell_id,
            spec.attempt,
            assigned_gpu_uuids=("GPU-0",),
            started_at_ns=1_000_000_000,
        )
        store.attach_process(spec.cell_id, spec.attempt, pid=27_000, pgid=27_000)

        def process_probe(pid: int, pgid: int) -> ProcessObservation:
            return ProcessObservation(
                pid,
                alive["value"],
                pgid if alive["value"] else None,
                "alive" if alive["value"] else "exited",
            )

        def send_kill(_command: QueuedCommandSpec, _pid: int, _pgid: int) -> None:
            signals.append("KILL")
            alive["value"] = False

        callbacks = replace(
            _scheduler_callbacks(
                launch=lambda _command, _gpus: SpawnedProcess(27_000, 27_000),
                process_probe=process_probe,
            ),
            send_term=lambda _command, _pid, _pgid: signals.append("TERM"),
            send_kill=send_kill,
            process_group_alive=lambda _pgid: alive["value"],
            partial_evidence=lambda _command: {"partial.raw": _sha("8")},
        )
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "term-kill.lock",
            callbacks=callbacks,
            watchdog_policy=WatchdogPolicy(termination_grace_seconds=1),
            clock_ns=lambda: next(times),
        )
        assert daemon.run_once().reconciled[-1][2] == "TERM_SENT"
        assert daemon.run_once().reconciled[-1][2] == "KILL_SENT"
        assert daemon.run_once().reconciled[-1][2] == "FAILED"
        failed = store.attempt(spec.cell_id, spec.attempt)
        assert signals == ["TERM", "KILL"]
        assert failed["status"] == "FAILED"
        assert failed["failure_code"] == "INFRASTRUCTURE:RUNTIME_LIMIT_EXCEEDED"
        assert failed["evidence_files"] == {"partial.raw": _sha("8")}
        assert failed["retry_decision"] == "NO_RETRY_BUILDER_OR_LIMIT"


def test_command_watchdog_limits_are_part_of_command_identity(tmp_path: Path) -> None:
    _specification, command = _queued_pair(tmp_path, cell_id="preflight:command-limits")
    changed = replace(command, max_runtime_seconds=command.max_runtime_seconds + 1)
    assert changed.command_sha256 != command.command_sha256


def test_scheduler_retries_only_infrastructure_at_most_twice(
    tmp_path: Path,
) -> None:
    clock = _TickClock()
    launches = []
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        first_spec, first_command = _queued_pair(
            tmp_path,
            cell_id="preflight:retry",
        )
        store.materialize_attempt(first_spec)
        store.enqueue_command(first_command)

        def launch(command, _gpu_uuids):
            launches.append(command.attempt)
            pid = 30_000 + command.attempt
            return SpawnedProcess(pid, pid)

        def retry_builder(command, attempt):
            return _queued_pair(
                tmp_path,
                cell_id=command.cell_id,
                attempt=attempt,
                timing_class=command.timing_class,
                required_gpu_count=command.required_gpu_count,
                compatibility_key=command.launch_compatibility_key,
                priority=command.priority,
            )

        failed = TerminalEvidence(
            status="FAILED",
            exit_code=17,
            atomic_publication_sha256=_sha("2"),
            raw_log_sha256=_sha("5"),
            failure_class="INFRASTRUCTURE",
            failure_code="HTTP_TRANSIENT",
            exclusion_reason="transient runner failure",
            included_in_analysis=False,
        )
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=launch,
                process_probe=lambda pid, _pgid: ProcessObservation(
                    pid,
                    False,
                    None,
                    "exited",
                    exit_code=17,
                ),
                terminal_validator=lambda _command, _attempt, _process: failed,
                retry_builder=retry_builder,
            ),
            clock_ns=clock,
        )
        daemon.run_once()
        daemon.run_once()
        daemon.run_once()
        daemon.run_once()
        assert launches == [1, 2, 3]
        assert (
            len(
                [
                    row
                    for row in store.snapshot()["attempts"]
                    if row["cell_id"] == first_spec.cell_id
                ]
            )
            == 3
        )
        assert store.attempt(first_spec.cell_id, 3)["retry_decision"] == (
            "NO_BLIND_RETRY"
        )

    scientific_root = tmp_path / "scientific"
    scientific_root.mkdir()
    scientific_launches = []
    with _store(scientific_root) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        spec, command = _queued_pair(
            scientific_root,
            cell_id="preflight:scientific",
        )
        store.materialize_attempt(spec)
        store.enqueue_command(command)
        scientific = TerminalEvidence(
            status="FAILED",
            exit_code=1,
            atomic_publication_sha256=_sha("2"),
            failure_class="EXACTNESS",
            failure_code="TOKEN_TRAJECTORY_MISMATCH",
            exclusion_reason="scientific exactness failure",
            included_in_analysis=False,
        )
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=scientific_root / "scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=lambda command, _gpu: (
                    scientific_launches.append(command.attempt)
                    or SpawnedProcess(40_000, 40_000)
                ),
                process_probe=lambda pid, _pgid: ProcessObservation(
                    pid,
                    False,
                    None,
                    "exited",
                    exit_code=1,
                ),
                terminal_validator=lambda _command, _attempt, _process: scientific,
                retry_builder=lambda command, attempt: _queued_pair(
                    scientific_root,
                    cell_id=command.cell_id,
                    attempt=attempt,
                ),
            ),
            clock_ns=_TickClock(),
        )
        daemon.run_once()
        daemon.run_once()
        assert scientific_launches == [1]
        assert len(store.snapshot()["attempts"]) == 1


def test_production_runtime_exposes_only_injected_retry_builder(
    tmp_path: Path,
) -> None:
    def retry_builder(command, attempt):
        return _queued_pair(tmp_path, cell_id=command.cell_id, attempt=attempt)

    runtime = ProductionSchedulerRuntime(retry_builder=retry_builder)
    assert runtime.callbacks().retry_builder is retry_builder
    assert ProductionSchedulerRuntime().callbacks().retry_builder is None
    with pytest.raises(TypeError, match="retry builder must be callable"):
        ProductionSchedulerRuntime(retry_builder=object())  # type: ignore[arg-type]


def test_retry_builder_must_use_attempt_specific_evidence_paths(
    tmp_path: Path,
) -> None:
    clock = _TickClock()
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-0",), _sha("a"))
        )
        spec, command = _queued_pair(tmp_path, cell_id="preflight:retry-overlap")
        store.materialize_attempt(spec)
        store.enqueue_command(command)

        def overlapping_retry_builder(current, attempt):
            next_spec, next_command = _queued_pair(
                tmp_path,
                cell_id=current.cell_id,
                attempt=attempt,
            )
            return next_spec, replace(
                next_command,
                expected_raw_log_path=current.expected_raw_log_path,
            )

        failed = TerminalEvidence(
            status="FAILED",
            exit_code=17,
            atomic_publication_sha256=_sha("2"),
            raw_log_sha256=_sha("5"),
            failure_class="INFRASTRUCTURE",
            failure_code="HTTP_TRANSIENT",
            exclusion_reason="transient runner failure",
            included_in_analysis=False,
        )
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "overlap.scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=lambda _command, _gpu: SpawnedProcess(45_000, 45_000),
                process_probe=lambda pid, _pgid: ProcessObservation(
                    pid,
                    False,
                    None,
                    "exited",
                    exit_code=17,
                ),
                terminal_validator=lambda _command, _attempt, _process: failed,
                retry_builder=overlapping_retry_builder,
            ),
            clock_ns=clock,
        )
        daemon.run_once()
        daemon.run_once()
        assert len(store.snapshot()["attempts"]) == 1
        assert store.dispatch_control() == (
            "STOP",
            "infrastructure_retry_builder_failed",
        )
        assert any(
            event["event_type"] == "INFRASTRUCTURE_RETRY_BUILD_FAILED"
            for event in store._event_rows()
        )


def test_operator_cli_initializes_and_reports_status(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "formal_experiment_operator.py"
    database = tmp_path / "cli.sqlite3"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    initialized = subprocess.run(
        [
            sys.executable,
            str(script),
            "--db",
            str(database),
            "init",
            "--run-id",
            "cli-run",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert initialized.returncode == 0, initialized.stderr
    status = subprocess.run(
        [sys.executable, str(script), "--db", str(database), "status"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert status.returncode == 0, status.stderr
    payload = json.loads(status.stdout)
    assert payload["run_id"] == "cli-run"
    assert len(payload["stage_plan"]) == 21

    sample = tmp_path / "provider-sample.json"
    sample.write_text(
        json.dumps(
            {
                "instance_uuid": "pro-cli-fixture",
                "state": "shutdown",
                "observed_at_ns": 4_000_000_000,
                "provider_started_at_ns": 1_000_000_000,
                "provider_stopped_at_ns": 3_000_000_000,
                "gpu_count": 2,
                "response_sha256": _sha("a"),
            }
        ),
        encoding="utf-8",
    )
    recorded = subprocess.run(
        [
            sys.executable,
            str(script),
            "--db",
            str(database),
            "record-provider-sample",
            "--sample",
            str(sample),
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert recorded.returncode == 0, recorded.stderr
    provider = json.loads(recorded.stdout)
    assert provider["whole_instance_billed_gpu_seconds"] == pytest.approx(4.0)
    assert provider["credential_stored"] is False


def _fake_nvidia_smi(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout=(
            "0, GPU-000, 12, 1024, 49140, 80.5\n1, GPU-111, 34, 2048, 49140, 91.5\n"
        ),
        stderr="",
    )


def _production_command(
    tmp_path: Path,
    *,
    cell_id: str,
    argv: tuple[str, ...],
    environment: tuple[tuple[str, str], ...] = (),
) -> QueuedCommandSpec:
    _specification, command = _queued_pair(tmp_path, cell_id=cell_id)
    return replace(command, argv=argv, environment=environment)


def _wait_for_exit(
    runtime: ProductionSchedulerRuntime,
    process: SpawnedProcess,
) -> ProcessObservation:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        observation = runtime.process_probe(process.pid, process.pgid)
        if not observation.alive:
            return observation
        time.sleep(0.02)
    raise AssertionError("short production child did not exit")


def test_production_launcher_uses_setsid_uuid_environment_and_exit_receipt(
    tmp_path: Path,
) -> None:
    command = _production_command(
        tmp_path,
        cell_id="preflight:real-child",
        argv=(
            sys.executable,
            "-c",
            (
                "import os,sys;"
                "print(os.environ['CUDA_VISIBLE_DEVICES']);"
                "print(os.environ['LIGHTCONE_GPU_UUID_TO_INDEX_JSON']);"
                "print(os.environ['LIGHTCONE_OPERATOR_CELL_ID']);"
                "print(os.environ['LIGHTCONE_OPERATOR_COMMAND_SHA256']);"
                "sys.exit(7)"
            ),
        ),
        environment=(("LIGHTCONE_TEST_VALUE", "exact"),),
    )
    Path(command.log_path).write_text("prior-launch-boundary\n")
    runtime = ProductionSchedulerRuntime(nvidia_runner=_fake_nvidia_smi)
    process = runtime.launch(command, ("GPU-111",))
    assert process.pid == process.pgid
    observation = _wait_for_exit(runtime, process)
    assert observation.exit_code == 7
    receipt = json.loads(Path(command.child_exit_receipt_path).read_text())
    assert receipt["wrapper_pid"] == process.pid
    assert receipt["wrapper_pgid"] == process.pgid
    assert receipt["exit_code"] == 7
    self_digest = receipt.pop("receipt_sha256")
    assert hashlib.sha256(canonical_json_bytes(receipt)).hexdigest() == self_digest
    log = Path(command.log_path).read_text()
    assert log.startswith("prior-launch-boundary\n")
    assert "GPU-111" in log
    assert '"GPU-111":1' in log
    assert command.cell_id in log
    assert command.command_sha256 in log


def test_production_kill_uses_durable_resident_server_target_before_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _production_command(
        tmp_path,
        cell_id="E3a:resident-kill-target",
        argv=(sys.executable, "-c", "pass"),
    )
    target = SimpleNamespace(
        server_process_id=41_000,
        server_process_group_id=41_000,
        server_boot_id="boot-resident-test",
        server_start_time_ticks=987_654,
    )
    runtime = ProductionSchedulerRuntime(nvidia_runner=_fake_nvidia_smi)
    monkeypatch.setattr(
        runtime,
        "_resident_server_watch_target",
        lambda _command: target,
    )
    monkeypatch.setattr(
        runtime,
        "_revalidate_live_wrapper",
        lambda _command, _pid, _pgid: True,
    )
    monkeypatch.setattr(
        experiment_operator_production_module,
        "linux_process_start_identity",
        lambda pid: (
            {
                "kind": "linux_proc_start_v1",
                "boot_id": target.server_boot_id,
                "start_time_ticks": target.server_start_time_ticks,
            }
            if pid == target.server_process_id
            else None
        ),
    )
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        os,
        "killpg",
        lambda pgid, requested_signal: killed.append((pgid, requested_signal)),
    )

    assert runtime.independent_process_groups(command) == (
        target.server_process_group_id,
    )
    runtime.send_kill(command, 42_000, 42_000)

    assert killed == [
        (target.server_process_group_id, signal.SIGKILL),
        (42_000, signal.SIGKILL),
    ]


def test_run_child_wrapper_records_crash_exit_without_guessing(tmp_path: Path) -> None:
    receipt_path = tmp_path / "crash.exit.json"
    exit_code = run_child_wrapper(
        (sys.executable, "-c", "import sys; sys.exit(23)"),
        exit_receipt_path=receipt_path,
        command_sha256=_sha("a"),
    )
    receipt = json.loads(receipt_path.read_text())
    assert exit_code == 23
    assert receipt["exit_code"] == 23
    assert receipt["finished_ns"] > receipt["started_ns"]
    assert receipt["launch_error_type"] is None


def test_scheduler_restart_reconciles_durable_exit_and_terminal(
    tmp_path: Path,
) -> None:
    command = _production_command(
        tmp_path,
        cell_id="preflight:restart",
        argv=(sys.executable, "-c", "print('finished')"),
    )
    base = _spec(tmp_path, cell_id=command.cell_id)
    specification = replace(
        base,
        command_sha256=command.command_sha256,
        output_directory=str((tmp_path / "restart-output").resolve()),
    )
    database = tmp_path / "restart.sqlite3"
    first_runtime = ProductionSchedulerRuntime(nvidia_runner=_fake_nvidia_smi)
    with ExperimentOperatorStore(database, run_id="restart-run") as store:
        store.initialize_stage_plan(_plan())
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-000", "GPU-111"), _sha("b"))
        )
        store.materialize_attempt(specification)
        store.enqueue_command(command)
        first = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "restart.scheduler.lock",
            callbacks=first_runtime.callbacks(),
        ).run_once()
        assert first.dispatched == ((command.cell_id, 1, ("GPU-000",)),)
        process_row = store.attempt(command.cell_id, 1)
        pid = process_row["pid"]
        pgid = process_row["pgid"]
    assert type(pid) is int and type(pgid) is int
    observation = _wait_for_exit(first_runtime, SpawnedProcess(pid, pgid))
    assert observation.exit_code == 0
    receipt = json.loads(Path(command.child_exit_receipt_path).read_text())
    Path(command.expected_raw_log_path).write_text('{"request":"terminal"}\n')
    Path(command.expected_junit_path).write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0"/>\n'
    )
    publish_atomic_terminal_result(
        command,
        status="COMPLETE",
        exit_code=0,
        started_ns=receipt["started_ns"],
        finished_ns=receipt["finished_ns"],
    )
    restarted_runtime = ProductionSchedulerRuntime(nvidia_runner=_fake_nvidia_smi)
    with ExperimentOperatorStore(database) as store:
        cycle = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "restart.scheduler.lock",
            callbacks=restarted_runtime.callbacks(),
        ).run_once()
        assert cycle.reconciled == ((command.cell_id, 1, "COMPLETE"),)
        completed = store.attempt(command.cell_id, 1)
        assert completed["exit_code"] == 0
        assert completed["compute_gpu_seconds"] > 0
        assert completed["reserved_gpu_seconds"] > 0


def test_terminal_validator_rejects_post_publication_tamper(tmp_path: Path) -> None:
    command = _production_command(
        tmp_path,
        cell_id="preflight:tamper",
        argv=(sys.executable, "-c", "pass"),
    )
    assert (
        run_child_wrapper(
            (sys.executable, "-c", "pass"),
            exit_receipt_path=command.child_exit_receipt_path,
            command_sha256=command.command_sha256,
        )
        == 0
    )
    receipt = json.loads(Path(command.child_exit_receipt_path).read_text())
    Path(command.expected_raw_log_path).write_text("raw\n")
    Path(command.expected_junit_path).write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0"/>\n'
    )
    publish_atomic_terminal_result(
        command,
        status="COMPLETE",
        exit_code=0,
        started_ns=receipt["started_ns"],
        finished_ns=receipt["finished_ns"],
    )
    Path(command.expected_raw_log_path).write_text("tampered\n")
    runtime = ProductionSchedulerRuntime(nvidia_runner=_fake_nvidia_smi)
    observation = ProcessObservation(os.getpid(), False, None, "exited", exit_code=0)
    with pytest.raises(ProductionCallbackError, match="raw log SHA-256 differs"):
        runtime.terminal_validator(
            command,
            {"pid": os.getpid(), "pgid": os.getpgrp()},
            observation,
        )


def test_nvidia_smi_statvfs_and_log_growth_callbacks(tmp_path: Path) -> None:
    inventory = query_nvidia_smi(runner=_fake_nvidia_smi)
    assert [(row.index, row.uuid) for row in inventory] == [
        (0, "GPU-000"),
        (1, "GPU-111"),
    ]
    assert inventory[1].power_draw_watts == 91.5
    assert statvfs_free_bytes(tmp_path) > 0
    log = tmp_path / "growth.log"
    log.write_text("one")
    first = log.stat().st_size
    with log.open("a") as handle:
        handle.write("two")
    assert log.stat().st_size > first


def test_paired_gpu_preferences_balance_and_isolated_auxiliary_serializes(
    tmp_path: Path,
) -> None:
    launches: list[tuple[str, tuple[str, ...]]] = []
    with _store(tmp_path) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("DUAL_SINGLE", ("GPU-000", "GPU-111"), _sha("a"))
        )
        for index in range(2):
            spec, base = _queued_pair(
                tmp_path,
                cell_id=f"preflight:paired-{index}",
            )
            command = replace(
                base,
                paired_gpu_key="pilot-00:trace-17",
                preferred_gpu_index=index,
            )
            store.materialize_attempt(
                replace(spec, command_sha256=command.command_sha256)
            )
            store.enqueue_command(command)
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "paired.lock",
            callbacks=_scheduler_callbacks(
                launch=lambda command, gpus: (
                    launches.append((command.cell_id, gpus))
                    or SpawnedProcess(50_000 + len(launches), 50_000 + len(launches))
                )
            ),
            clock_ns=_TickClock(),
        )
        assert len(daemon.run_once().dispatched) == 2
    assert [gpu for _cell, gpu in launches] == [("GPU-000",), ("GPU-111",)]

    isolated_root = tmp_path / "isolated"
    isolated_root.mkdir()
    with _store(isolated_root) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("ISOLATED", ("GPU-000", "GPU-111"), _sha("a"))
        )
        for index in range(2):
            spec, command = _queued_pair(
                isolated_root,
                cell_id=f"preflight:aux-{index}",
                timing_class="SAFE_AUXILIARY",
            )
            store.materialize_attempt(spec)
            store.enqueue_command(command)
        result = FormalExperimentSchedulerDaemon(
            store,
            lock_path=isolated_root / "scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=lambda _command, _gpus: SpawnedProcess(60_000, 60_000)
            ),
            clock_ns=_TickClock(),
        ).run_once()
        assert len(result.dispatched) == 1


@pytest.mark.parametrize(
    "timing_class", ("EXCLUSIVE", "PROFILER", "FAILURE", "ARCHIVE")
)
def test_registered_exclusive_timing_classes_never_co_schedule(
    tmp_path: Path,
    timing_class: str,
) -> None:
    root = tmp_path / timing_class.lower()
    root.mkdir()
    with _store(root) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("DUAL_SINGLE", ("GPU-000", "GPU-111"), _sha("a"))
        )
        exclusive_spec, exclusive = _queued_pair(
            root,
            cell_id=f"preflight:{timing_class.lower()}",
            timing_class=timing_class,
            priority=10,
        )
        auxiliary_spec, auxiliary = _queued_pair(
            root,
            cell_id="preflight:auxiliary",
            timing_class="SAFE_AUXILIARY",
        )
        for specification, command in (
            (exclusive_spec, exclusive),
            (auxiliary_spec, auxiliary),
        ):
            store.materialize_attempt(specification)
            store.enqueue_command(command)
        result = FormalExperimentSchedulerDaemon(
            store,
            lock_path=root / "scheduler.lock",
            callbacks=_scheduler_callbacks(
                launch=lambda _command, _gpus: SpawnedProcess(70_000, 70_000)
            ),
            clock_ns=_TickClock(),
        ).run_once()
        assert result.dispatched == ((exclusive.cell_id, 1, ("GPU-000",)),)


def test_terminal_duration_accounts_each_reserved_gpu(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        specification, command = _queued_pair(
            tmp_path,
            cell_id="preflight:duration",
            timing_class="EXCLUSIVE",
            required_gpu_count=2,
        )
        store.materialize_attempt(specification)
        store.enqueue_command(command)
        store.mark_running_before_spawn(
            command.cell_id,
            1,
            assigned_gpu_uuids=("GPU-000", "GPU-111"),
            started_at_ns=500_000_000,
        )
        store.attach_process(command.cell_id, 1, pid=80_000, pgid=80_000)
        terminal = TerminalEvidence(
            status="COMPLETE",
            exit_code=0,
            atomic_publication_sha256=_sha("1"),
            terminal_sha256=_sha("2"),
            junit_sha256=_sha("3"),
            raw_log_sha256=_sha("4"),
            started_ns=1_000_000_000,
            finished_ns=4_000_000_000,
        )
        daemon = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "duration.lock",
            callbacks=_scheduler_callbacks(
                launch=lambda _command, _gpus: SpawnedProcess(1, 1)
            ),
        )
        daemon._finish_from_terminal(command, terminal, now_ns=5_000_000_000)
        completed = store.attempt(command.cell_id, 1)
        assert completed["compute_gpu_seconds"] == 6.0
        assert completed["reserved_gpu_seconds"] == 6.0


def test_production_archive_rsync_verify_publish_and_full_rehydrate(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    payloads = {"logs/a.log": b"alpha", "raw/b.json": b"beta"}
    rows = []
    for relative, payload in sorted(payloads.items()):
        path = remote / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    manifest_path = remote / "sha256_manifest.json"
    manifest_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "kind": "formal_archive_sha256_manifest",
                "files": rows,
            }
        )
    )
    request = ArchiveRequest(
        archive_id="wave-001",
        safe_boundary="preflight:sealed",
        remote_payload_root=str(remote.resolve()),
        local_partial_root=str((tmp_path / "local.partial").resolve()),
        local_final_root=str((tmp_path / "local").resolve()),
        remote_manifest_sha256=file_sha256(manifest_path),
        predicted_payload_bytes=sum(len(payload) for payload in payloads.values()),
    )

    def fake_rsync(argv, *, check, shell):
        assert check is True and shell is False and argv[0] == "rsync"
        source = Path(argv[-2].removesuffix("/"))
        destination = Path(argv[-1].removesuffix("/"))
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return subprocess.CompletedProcess(argv, 0)

    runtime = ProductionArchiveRuntime(runner=fake_rsync, minimum_local_free_bytes=0)
    transfer = runtime.transfer(request, None)
    local = runtime.verify_local_sha(request, transfer)
    rehydrated = runtime.rehydrate(request, local)
    assert transfer.checked_file_count == 2
    assert not Path(request.local_partial_root).exists()
    assert Path(request.local_final_root).is_dir()
    assert rehydrated.checked_file_count == 2
    assert rehydrated.content_tree_sha256 is not None
    assert not hasattr(runtime, "delete_remote")


def test_archive_cli_resumes_verifies_and_never_deletes_remote(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "formal_experiment_operator.py"
    database = tmp_path / "archive-cli.sqlite3"
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    initialized = subprocess.run(
        [
            sys.executable,
            str(script),
            "--db",
            str(database),
            "init",
            "--run-id",
            "archive-cli",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert initialized.returncode == 0, initialized.stderr

    remote = tmp_path / "remote-cli"
    remote.mkdir()
    payload = remote / "raw" / "terminal.json"
    payload.parent.mkdir()
    payload.write_bytes(b"durable evidence\n")
    manifest_path = remote / "sha256_manifest.json"
    manifest_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "kind": "formal_archive_sha256_manifest",
                "files": [
                    {
                        "path": "raw/terminal.json",
                        "sha256": file_sha256(payload),
                        "size_bytes": payload.stat().st_size,
                    }
                ],
            }
        )
    )
    request = ArchiveRequest(
        archive_id="archive-cli-wave",
        safe_boundary="preflight:sealed",
        remote_payload_root=str(remote.resolve()),
        local_partial_root=str((tmp_path / "archive.partial").resolve()),
        local_final_root=str((tmp_path / "archive.final").resolve()),
        remote_manifest_sha256=file_sha256(manifest_path),
        predicted_payload_bytes=payload.stat().st_size,
    )
    request_path = tmp_path / "archive-request.json"
    request_path.write_text(json.dumps(vars(request)), encoding="utf-8")
    fake_rsync = tmp_path / "fake-rsync"
    fake_rsync.write_text(
        "#!" + sys.executable + "\n"
        "import pathlib, shutil, sys\n"
        "source = pathlib.Path(sys.argv[-2].rstrip('/'))\n"
        "target = pathlib.Path(sys.argv[-1].rstrip('/'))\n"
        "shutil.copytree(source, target, dirs_exist_ok=True)\n",
        encoding="utf-8",
    )
    fake_rsync.chmod(0o700)
    archived = subprocess.run(
        [
            sys.executable,
            str(script),
            "--db",
            str(database),
            "archive-run",
            "--request",
            str(request_path),
            "--rsync-executable",
            str(fake_rsync),
            "--minimum-local-free-bytes",
            "0",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert archived.returncode == 0, archived.stderr
    result = json.loads(archived.stdout)
    assert result["remote_deletion_performed"] is False
    assert Path(request.local_final_root, "raw", "terminal.json").is_file()
    assert remote.is_dir() and payload.is_file()
    with ExperimentOperatorStore(database) as store:
        assert store.archive_checkpoint(request.archive_id)["state"] == (
            "EVICTION_AUTHORIZED"
        )


def test_scheduler_run_cli_holds_resident_path_for_one_empty_cycle(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "formal_experiment_operator.py"
    database = tmp_path / "daemon-cli.sqlite3"
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    initialized = subprocess.run(
        [
            sys.executable,
            str(script),
            "--db",
            str(database),
            "init",
            "--run-id",
            "daemon-cli",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert initialized.returncode == 0, initialized.stderr
    with ExperimentOperatorStore(database) as store:
        store.configure_interference_envelope(
            InterferenceEnvelope("UNRESOLVED", ("GPU-000", "GPU-111"), _sha("a"))
        )
    scheduled = subprocess.run(
        [
            sys.executable,
            str(script),
            "--db",
            str(database),
            "scheduler-run",
            "--max-cycles",
            "1",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert scheduled.returncode == 0, scheduled.stderr
    payload = json.loads(scheduled.stdout)
    assert len(payload["cycles"]) == 1
    assert payload["cycles"][0]["dispatch_state"] == "RUN"
