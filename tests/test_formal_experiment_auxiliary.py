from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.orchestration.experiment_operator import (
    AttemptTransitionError,
    AuxiliaryCellAdoption,
    AuxiliaryGroupTerminal,
    AuxiliaryJobSpec,
    AuxiliaryPhysicalGroupSpec,
    CellAttemptSpec,
    ControllerArtifactBinding,
    ExperimentOperatorError,
    ExperimentOperatorStore,
    FormalExperimentSchedulerDaemon,
    ProcessObservation,
    QueuedCommandSpec,
    RecoveredProcessStart,
    SchedulerCallbacks,
    SpawnedProcess,
    StagePlanEntry,
    TerminalEvidence,
)
from lightcone_spec.orchestration.formal_experiment_controller import (
    DagCellLaunch,
    DagControllerCallbacks,
    DagExecutionPlan,
    DagMaterialization,
    DagReduction,
    FormalExperimentDagBlocked,
    FormalExperimentDagController,
)
from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
    DirectoryAuxiliaryPhysicalRuntime,
    FormalSingleOperatorDagDriver,
    _publish_no_replace,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000_000_000

    def __call__(self) -> int:
        self.value += 10_000_000_000
        return self.value


def _sha(character: str) -> str:
    return character * 64


def _binding(tmp_path: Path, name: str) -> ControllerArtifactBinding:
    path = (tmp_path / name).resolve()
    path.write_text(name + "\n", encoding="utf-8")
    return ControllerArtifactBinding.bind(path)


def _identity() -> dict[str, str]:
    return {
        "source_sha256": _sha("a"),
        "patch_sha256": _sha("b"),
        "registry_sha256": _sha("c"),
    }


def _auxiliary_spec(
    tmp_path: Path,
    *,
    node: str,
    source_kind: str,
    count: int,
    attempt: int = 1,
) -> AuxiliaryPhysicalGroupSpec:
    jobs = tuple(
        AuxiliaryJobSpec(
            job_id=f"{node}:aux:{index:03d}",
            attempt=attempt,
            adoption_key=f"decision:{index:03d}",
            scientific_axes={"decision_ordinal": index, "kind": source_kind},
            identity=_identity(),
            command_sha256=_sha("abcdef"[index % 6]),
            output_directory=str(
                (tmp_path / f"{node}-aux-{attempt}-{index:03d}").resolve()
            ),
        )
        for index in range(count)
    )
    return AuxiliaryPhysicalGroupSpec(
        group_id=f"{node}:auxiliary-campaign",
        attempt=attempt,
        node=node,
        source_kind=source_kind,
        jobs=jobs,
        assigned_gpu_uuids=("GPU-0", "GPU-1"),
        launch_command_sha256=_sha("f"),
        output_directory=str((tmp_path / f"{node}-aux-group-{attempt}").resolve()),
    )


def _cell_spec(
    *,
    job: AuxiliaryJobSpec,
    stage: str,
    phase: str,
) -> CellAttemptSpec:
    return CellAttemptSpec(
        cell_id=f"cell:{job.adoption_key}",
        attempt=1,
        stage=stage,
        phase=phase,
        block="auxiliary",
        seed=None,
        scientific_axes=job.scientific_axes,
        identity=job.identity,
        command_sha256=job.command_sha256,
        output_directory=job.output_directory,
    )


def _serving_launch(tmp_path: Path, *, stage: str, phase: str) -> DagCellLaunch:
    cell_id = "cell:serving"
    prefix = (tmp_path / "serving").resolve()
    command = QueuedCommandSpec(
        cell_id=cell_id,
        attempt=1,
        argv=("python3", "-m", "fixture.worker"),
        launch_compatibility_key="serving:tp2",
        required_gpu_count=2,
        timing_class="EXCLUSIVE",
        predicted_high_water_bytes=1,
        monitored_path=str(tmp_path.resolve()),
        log_path=str(prefix.with_suffix(".log")),
        expected_terminal_path=str(prefix.with_suffix(".terminal.json")),
        expected_junit_path=str(prefix.with_suffix(".junit.xml")),
        expected_raw_log_path=str(prefix.with_suffix(".raw.jsonl")),
        atomic_pointer_path=str(prefix.with_suffix(".pointer.json")),
        child_exit_receipt_path=str(prefix.with_suffix(".exit.json")),
    )
    return DagCellLaunch(
        CellAttemptSpec(
            cell_id=cell_id,
            attempt=1,
            stage=stage,
            phase=phase,
            block="pilot-00",
            seed=1,
            scientific_axes={"kind": "serving"},
            identity=_identity(),
            command_sha256=command.command_sha256,
            output_directory=str((tmp_path / "serving-output").resolve()),
        ),
        command,
    )


def _callbacks(
    tmp_path: Path,
    *,
    store: ExperimentOperatorStore,
    node: str,
    stage: str,
    phase: str,
    source_kind: str,
    count: int,
    launch_count: list[int],
    attempt: int = 1,
    serving: DagCellLaunch | None = None,
    terminal_failure: bool = False,
    publish_terminal: bool = True,
) -> DagControllerCallbacks:
    spec = _auxiliary_spec(
        tmp_path,
        node=node,
        source_kind=source_kind,
        count=count,
        attempt=attempt,
    )
    publication = _binding(tmp_path, f"{node}-publication-{attempt}.json")
    materialization = _binding(tmp_path, f"{node}-materialization.json")
    node_materialization = _binding(tmp_path, f"{node}-node-materialization.json")
    execution_source = _binding(tmp_path, f"{node}-execution-source.json")
    decision = _binding(tmp_path, f"{node}-decision.json")
    completion = _binding(tmp_path, f"{node}-completion.json")

    def launch(value: AuxiliaryPhysicalGroupSpec) -> SpawnedProcess:
        latest = store.latest_controller_auxiliary_group(node)
        assert latest is not None and latest["status"] == "RUNNING"
        assert store.latest_stage_attempts(node) == ()
        assert value == spec
        launch_count.append(value.attempt)
        return SpawnedProcess(901 + value.attempt, 901 + value.attempt)

    def terminal(
        value: AuxiliaryPhysicalGroupSpec,
        durable: dict[str, object],
    ) -> AuxiliaryGroupTerminal | None:
        assert value == spec
        if not publish_terminal:
            return None
        started = int(durable["started_at_ns"])
        terminals = {}
        for index, job in enumerate(value.jobs):
            failed = terminal_failure and index == 0
            terminals[job.job_id] = TerminalEvidence(
                status="FAILED" if failed else "COMPLETE",
                exit_code=17 if failed else 0,
                atomic_publication_sha256=publication.sha256,
                terminal_sha256=_sha("1"),
                junit_sha256=_sha("2"),
                raw_log_sha256=_sha("3"),
                evidence_files={f"raw-{index}.json": _sha("4")},
                failure_class="SCIENTIFIC" if failed else None,
                failure_code="COMPATIBILITY_FAILED" if failed else None,
                exclusion_reason="compatibility_failed" if failed else None,
                included_in_analysis=not failed,
                started_ns=started + 1,
                finished_ns=started + 1_000_000_001,
            )
        return AuxiliaryGroupTerminal(
            publication,
            terminals,
            compute_gpu_seconds=2.0,
            reserved_gpu_seconds=2.0,
        )

    adoptions = tuple(
        AuxiliaryCellAdoption(
            job.job_id,
            job.attempt,
            job.adoption_key,
            _cell_spec(job=job, stage=stage, phase=phase),
        )
        for job in spec.jobs
    )
    expected = tuple(
        sorted(
            tuple(value.attempt.cell_id for value in adoptions)
            + (() if serving is None else (serving.attempt.cell_id,))
        )
    )
    return DagControllerCallbacks(
        materialize=lambda _node, _predecessor: (_ for _ in ()).throw(
            AssertionError("auxiliary-aware materializer was bypassed")
        ),
        plan=lambda _node, _binding: DagExecutionPlan(
            execution_source,
            None,
            () if serving is None else (serving,),
        ),
        actual_results=lambda _node, _attempts: {},
        reduce=lambda _node, _binding, _actuals: DagReduction(decision, completion),
        auxiliary_plan=lambda _node, _predecessor: spec,
        auxiliary_launch=launch,
        auxiliary_terminal=terminal,
        materialize_with_auxiliary=lambda _node, _predecessor, sources: (
            DagMaterialization(
                materialization,
                node_materialization,
                expected,
                tuple(sorted(sources.items())),
            )
        ),
        auxiliary_adoptions=lambda _node, _binding, _spec: adoptions,
    )


def test_e6_auxiliary_runs_before_cells_and_is_adopted_without_double_charge(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    database = tmp_path / "e6.sqlite3"
    launch_count: list[int] = []
    with ExperimentOperatorStore(
        database,
        run_id="e6-auxiliary",
        clock_ns=clock,
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e6_pilot", 0, "E6", "pilot", "3", 3),)
        )
        serving = _serving_launch(tmp_path, stage="E6", phase="pilot")
        controller = FormalExperimentDagController(
            store=store,
            callbacks=_callbacks(
                tmp_path,
                store=store,
                node="e6_pilot",
                stage="E6",
                phase="pilot",
                source_kind="e6_interface_fit",
                count=2,
                launch_count=launch_count,
                serving=serving,
            ),
        )
        assert controller.run_once().action == "WAITING"
        assert store.latest_stage_attempts("e6_pilot") == ()
        assert controller.run_once().action == "WAITING"
        assert launch_count == [1]
        assert controller.run_once().action == "WAITING"
        before = store.snapshot()["stage_plan"][0]
        assert before["actual_gpu_hours"] == pytest.approx(2.0 / 3600.0)
        assert controller.run_once().action == "MATERIALIZED"
        attempts = store.latest_stage_attempts("e6_pilot")
        assert len(attempts) == 2
        assert {row["status"] for row in attempts} == {"COMPLETE"}
        assert sum(row["compute_gpu_seconds"] for row in attempts) == pytest.approx(2.0)
        assert controller.run_once().action == "PLANNED"
        assert len(store.latest_stage_attempts("e6_pilot")) == 3
        assert store.command_for_attempt("cell:decision:000", 1) is None
        after = store.snapshot()["stage_plan"][0]
        assert after["actual_gpu_hours"] == pytest.approx(before["actual_gpu_hours"])
        assert store.controller_auxiliary_adopted_cell_ids("e6_pilot") == (
            "cell:decision:000",
            "cell:decision:001",
        )


def test_e0_exact_108_auxiliary_decisions_materialize_no_future_sentinels(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    launch_count: list[int] = []
    with ExperimentOperatorStore(
        tmp_path / "e0.sqlite3",
        run_id="e0-auxiliary",
        clock_ns=clock,
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e0_tuning", 0, "E0", "tuning", "108", 108),)
        )
        callbacks = _callbacks(
            tmp_path,
            store=store,
            node="e0_tuning",
            stage="E0",
            phase="tuning",
            source_kind="e0_compatibility",
            count=108,
            launch_count=launch_count,
        )
        actual = _binding(tmp_path, "e0-all-na-compatibility-actual.json")
        callbacks = replace(
            callbacks,
            actual_results=lambda _node, attempts: {
                str(row["cell_id"]): actual.absolute_path for row in attempts
            },
        )
        controller = FormalExperimentDagController(
            store=store,
            callbacks=callbacks,
        )
        assert controller.run_once().action == "WAITING"
        assert store.latest_stage_attempts("e0_tuning") == ()
        assert len(store.latest_controller_auxiliary_group("e0_tuning")["jobs"]) == 108
        assert controller.run_once().action == "WAITING"
        assert controller.run_once().action == "WAITING"
        assert controller.run_once().action == "MATERIALIZED"
        assert len(store.latest_stage_attempts("e0_tuning")) == 108
        assert controller.run_once().action == "PLANNED"
        assert store.queued_commands() == ()
        assert launch_count == [1]
        assert controller.run_once().action == "REDUCED"
        assert store.controller_node("e0_tuning")["state"] == "REDUCED"
        assert len(store.latest_stage_attempts("e0_tuning")) == 108


def test_running_auxiliary_restart_never_respawns_or_creates_cells(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    database = tmp_path / "restart.sqlite3"
    launch_count: list[int] = []
    with ExperimentOperatorStore(
        database,
        run_id="restart-auxiliary",
        clock_ns=clock,
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e6_pilot", 0, "E6", "pilot", "2", 2),)
        )
        controller = FormalExperimentDagController(
            store=store,
            callbacks=_callbacks(
                tmp_path,
                store=store,
                node="e6_pilot",
                stage="E6",
                phase="pilot",
                source_kind="e6_interface_fit",
                count=2,
                launch_count=launch_count,
                publish_terminal=False,
            ),
        )
        assert controller.run_once().action == "WAITING"
        assert controller.run_once().action == "WAITING"
        assert launch_count == [1]
    with ExperimentOperatorStore(database, clock_ns=clock) as reopened:
        controller = FormalExperimentDagController(
            store=reopened,
            callbacks=_callbacks(
                tmp_path,
                store=reopened,
                node="e6_pilot",
                stage="E6",
                phase="pilot",
                source_kind="e6_interface_fit",
                count=2,
                launch_count=launch_count,
                publish_terminal=False,
            ),
        )
        assert controller.run_once().action == "WAITING"
        assert controller.run_once().action == "WAITING"
        assert launch_count == [1]
        assert reopened.latest_stage_attempts("e6_pilot") == ()


def test_restart_recovers_materialized_node_before_auxiliary_adoption(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    database = tmp_path / "adoption-crash.sqlite3"
    launch_count: list[int] = []
    with ExperimentOperatorStore(
        database,
        run_id="adoption-crash",
        clock_ns=clock,
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e6_pilot", 0, "E6", "pilot", "2", 2),)
        )
        good_callbacks = _callbacks(
            tmp_path,
            store=store,
            node="e6_pilot",
            stage="E6",
            phase="pilot",
            source_kind="e6_interface_fit",
            count=2,
            launch_count=launch_count,
        )
        interrupted = replace(
            good_callbacks,
            auxiliary_adoptions=lambda _node, _binding, _spec: (_ for _ in ()).throw(
                RuntimeError("simulated crash before adoption")
            ),
        )
        controller = FormalExperimentDagController(
            store=store,
            callbacks=interrupted,
        )
        assert controller.run_once().action == "WAITING"
        assert controller.run_once().action == "WAITING"
        assert controller.run_once().action == "WAITING"
        with pytest.raises(RuntimeError, match="simulated crash"):
            controller.run_once()
        assert store.controller_node("e6_pilot")["state"] == "MATERIALIZED"
        assert store.latest_stage_attempts("e6_pilot") == ()
    with ExperimentOperatorStore(database, clock_ns=clock) as reopened:
        controller = FormalExperimentDagController(
            store=reopened,
            callbacks=_callbacks(
                tmp_path,
                store=reopened,
                node="e6_pilot",
                stage="E6",
                phase="pilot",
                source_kind="e6_interface_fit",
                count=2,
                launch_count=launch_count,
            ),
        )
        assert controller.run_once().action == "PLANNED"
        assert len(reopened.latest_stage_attempts("e6_pilot")) == 2
        assert launch_count == [1]


def test_scientific_auxiliary_failure_is_retained_and_cannot_retry(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    launch_count: list[int] = []
    database = tmp_path / "retry.sqlite3"
    with ExperimentOperatorStore(
        database,
        run_id="retry-auxiliary",
        clock_ns=clock,
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e6_pilot", 0, "E6", "pilot", "2", 2),)
        )
        failed_callbacks = _callbacks(
            tmp_path,
            store=store,
            node="e6_pilot",
            stage="E6",
            phase="pilot",
            source_kind="e6_interface_fit",
            count=2,
            launch_count=launch_count,
            terminal_failure=True,
        )
        failed = FormalExperimentDagController(store=store, callbacks=failed_callbacks)
        assert failed.run_once().action == "WAITING"
        assert failed.run_once().action == "WAITING"
        assert failed.run_once().action == "BLOCKED"
        assert len(store.controller_auxiliary_groups()) == 1
        retained = store.latest_controller_auxiliary_group("e6_pilot")
        assert retained is not None
        assert retained["failure_class"] == "SCIENTIFIC"
        assert retained["jobs"][0]["failure_class"] == "SCIENTIFIC"
        assert store.latest_stage_attempts("e6_pilot") == ()
        assert failed.run_once().action == "BLOCKED"
        assert launch_count == [1]
        store.resume_controller_node(
            node="e6_pilot",
            reason="operator-authorized auxiliary infrastructure retry",
        )
        retry = FormalExperimentDagController(
            store=store,
            callbacks=_callbacks(
                tmp_path,
                store=store,
                node="e6_pilot",
                stage="E6",
                phase="pilot",
                source_kind="e6_interface_fit",
                count=2,
                launch_count=launch_count,
                attempt=2,
            ),
        )
        with pytest.raises(ExperimentOperatorError, match="infrastructure-only"):
            retry.run_once()
        assert len(store.controller_auxiliary_groups()) == 1
        assert launch_count == [1]


def test_auxiliary_allows_only_two_infrastructure_retries(tmp_path: Path) -> None:
    clock = _Clock()
    with ExperimentOperatorStore(
        tmp_path / "infra-retry.sqlite3",
        run_id="infra-retry-auxiliary",
        clock_ns=clock,
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e6_pilot", 0, "E6", "pilot", "2", 2),)
        )
        for attempt in (1, 2, 3):
            spec = _auxiliary_spec(
                tmp_path,
                node="e6_pilot",
                source_kind="e6_interface_fit",
                count=2,
                attempt=attempt,
            )
            assert store.register_controller_auxiliary_group(spec)
            store.start_controller_auxiliary_group_with_launcher(
                spec,
                launcher=lambda attempt=attempt: SpawnedProcess(
                    1_000 + attempt,
                    1_000 + attempt,
                ),
            )
            store.fail_controller_auxiliary_spawn(
                spec,
                exception_type="OSError",
            )
        fourth = _auxiliary_spec(
            tmp_path,
            node="e6_pilot",
            source_kind="e6_interface_fit",
            count=2,
            attempt=4,
        )
        with pytest.raises(ExperimentOperatorError, match="limit is exhausted"):
            store.register_controller_auxiliary_group(fourth)
        assert tuple(
            (row["attempt"], row["failure_class"])
            for row in store.controller_auxiliary_groups()
        ) == ((1, "INFRASTRUCTURE"), (2, "INFRASTRUCTURE"), (3, "INFRASTRUCTURE"))


def test_dispatch_stop_cannot_resume_while_auxiliary_group_is_running(
    tmp_path: Path,
) -> None:
    with ExperimentOperatorStore(
        tmp_path / "auxiliary-stop.sqlite3",
        run_id="auxiliary-stop",
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e6_pilot", 0, "E6", "pilot", "2", 2),)
        )
        spec = _auxiliary_spec(
            tmp_path,
            node="e6_pilot",
            source_kind="e6_interface_fit",
            count=2,
        )
        assert store.register_controller_auxiliary_group(spec)
        store.start_controller_auxiliary_group_with_launcher(
            spec,
            launcher=lambda: SpawnedProcess(901, 901),
        )
        store.set_dispatch_stop("fixture auxiliary stop")

        with pytest.raises(AttemptTransitionError, match="unverified auxiliary work"):
            store.clear_dispatch_stop(reason="unsafe early resume")


def test_auxiliary_termination_signal_failure_seals_dispatch_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ExperimentOperatorStore(
        tmp_path / "auxiliary-signal.sqlite3",
        run_id="auxiliary-signal",
    ) as store:
        runtime = object.__new__(DirectoryAuxiliaryPhysicalRuntime)
        runtime.store = store
        spec = _auxiliary_spec(
            tmp_path,
            node="e6_pilot",
            source_kind="e6_interface_fit",
            count=2,
        )

        def fail_signal(*_args: object, **_kwargs: object) -> None:
            raise OSError("fixture signal failure")

        monkeypatch.setattr(runtime, "_signal_process_tree", fail_signal)
        with pytest.raises(FormalExperimentDagBlocked, match="dispatch STOP"):
            runtime._send_auxiliary_termination_signal(
                spec,
                {"pid": 901, "pgid": 901},
                signal_number=15,
                sent_at_ns=1_000_000_000,
            )
        assert store.dispatch_control() == ("STOP", "auxiliary_process_term_failed")


def test_auxiliary_termination_targets_require_exact_group_identity(
    tmp_path: Path,
) -> None:
    spec = _auxiliary_spec(
        tmp_path,
        node="e6_pilot",
        source_kind="e6_interface_fit",
        count=2,
    )
    output = Path(spec.output_directory)
    output.mkdir(parents=True)
    _publish_no_replace(
        output / "auxiliary-termination-targets.json",
        {
            "schema_version": 1,
            "kind": "formal_single_operator_auxiliary_termination_targets",
            "group_id": "another-group",
            "attempt": spec.attempt,
            "process_start_receipt_sha256": _sha("a"),
            "pgids": [901],
        },
    )

    with pytest.raises(ValueError, match="termination targets differ"):
        DirectoryAuxiliaryPhysicalRuntime._termination_targets_alive(spec)


def test_auxiliary_stale_child_heartbeat_stops_without_signalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.orchestration import experiment_operator_production

    with ExperimentOperatorStore(
        tmp_path / "auxiliary-heartbeat-stop.sqlite3",
        run_id="auxiliary-heartbeat-stop",
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e6_pilot", 0, "E6", "pilot", "2", 2),)
        )
        spec = replace(
            _auxiliary_spec(
                tmp_path,
                node="e6_pilot",
                source_kind="e6_interface_fit",
                count=2,
            ),
            process_hard_timeout_ns=1_000_000_000_000_000,
        )
        Path(spec.output_directory).mkdir(parents=True)
        assert store.register_controller_auxiliary_group(spec)
        started_ns = time.time_ns() - 130_000_000_000
        store.start_controller_auxiliary_group_with_launcher(
            spec,
            launcher=lambda: SpawnedProcess(901, 901, _sha("9")),
            started_at_ns=started_ns,
        )
        monkeypatch.setattr(
            experiment_operator_production,
            "revalidate_child_start_receipt",
            lambda *_args, **_kwargs: RecoveredProcessStart(
                901,
                901,
                started_ns + 1,
                _sha("9"),
            ),
        )
        runtime = object.__new__(DirectoryAuxiliaryPhysicalRuntime)
        runtime.store = store
        monkeypatch.setattr(
            runtime,
            "_load_descriptor",
            lambda _path: SimpleNamespace(
                group_id=spec.group_id,
                process_hard_timeout_ns=spec.process_hard_timeout_ns,
            ),
        )
        monkeypatch.setattr(runtime, "_worker_terminal", lambda _descriptor: None)
        monkeypatch.setattr(runtime, "_process_alive", lambda *_args: True)
        monkeypatch.setattr(runtime, "_heartbeat", lambda *_args: None)
        monkeypatch.setattr(
            runtime,
            "_gpu_observation",
            lambda _spec: {"status": "AVAILABLE", "rows": []},
        )
        monkeypatch.setattr(
            runtime,
            "_signal_process_tree",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
        )

        durable = store.latest_controller_auxiliary_group("e6_pilot")
        assert durable is not None
        assert runtime.terminal(spec, durable) is None
        assert store.dispatch_control() == (
            "STOP",
            "auxiliary_child_heartbeat_stale",
        )
        retained = store.latest_controller_auxiliary_group("e6_pilot")
        assert retained is not None
        assert retained["status"] == "RUNNING"
        assert retained["termination_reason"] is None
        events = [
            row
            for row in store._event_rows()
            if row["event_type"] == "DAG_AUXILIARY_CHILD_HEARTBEAT_STALE"
        ]
        assert len(events) == 1
        assert events[0]["payload"]["automatic_signal"] is False


def test_driver_resumes_running_auxiliary_only_after_fresh_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.orchestration import experiment_operator_production

    with ExperimentOperatorStore(
        tmp_path / "auxiliary-heartbeat-resume.sqlite3",
        run_id="auxiliary-heartbeat-resume",
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e6_pilot", 0, "E6", "pilot", "2", 2),)
        )
        spec = replace(
            _auxiliary_spec(
                tmp_path,
                node="e6_pilot",
                source_kind="e6_interface_fit",
                count=2,
            ),
            process_hard_timeout_ns=1_000_000_000_000_000,
        )
        output = Path(spec.output_directory)
        output.mkdir(parents=True)
        assert store.register_controller_auxiliary_group(spec)
        started_ns = time.time_ns() - 130_000_000_000
        store.start_controller_auxiliary_group_with_launcher(
            spec,
            launcher=lambda: SpawnedProcess(901, 901, _sha("9")),
            started_at_ns=started_ns,
        )
        store.set_dispatch_stop("auxiliary_child_heartbeat_stale")
        heartbeat_path = output / "auxiliary-worker-heartbeat.json"
        _publish_no_replace(
            heartbeat_path,
            {
                "schema_version": 1,
                "kind": "formal_experiment_child_heartbeat",
                "cell_id": spec.group_id,
                "attempt": spec.attempt,
                "command_sha256": spec.launch_command_sha256,
                "worker_pid": 902,
                "sequence": 1,
                "observed_at_ns": time.time_ns(),
                "phase": "RUNNING",
            },
        )
        monkeypatch.setattr(
            experiment_operator_production,
            "revalidate_child_start_receipt",
            lambda *_args, **_kwargs: RecoveredProcessStart(
                901,
                901,
                started_ns + 1,
                _sha("9"),
            ),
        )
        monkeypatch.setattr(
            DirectoryAuxiliaryPhysicalRuntime,
            "_load_descriptor",
            staticmethod(
                lambda _path: SimpleNamespace(
                    group_id=spec.group_id,
                    attempt=spec.attempt,
                    heartbeat_output_path=str(heartbeat_path),
                )
            ),
        )
        scheduler = FormalExperimentSchedulerDaemon(
            store,
            lock_path=tmp_path / "auxiliary-heartbeat-resume.lock",
            callbacks=SchedulerCallbacks(
                launch=lambda *_args: (_ for _ in ()).throw(AssertionError()),
                process_probe=lambda pid, pgid: ProcessObservation(
                    pid,
                    True,
                    pgid,
                    "alive",
                ),
                log_size_bytes=lambda _command: 0,
                gpu_snapshot=lambda _gpus: {},
                terminal_validator=lambda *_args: None,
                free_disk_bytes=lambda _path: 100 * 1024**3,
            ),
        )
        driver = FormalSingleOperatorDagDriver(
            store=store,
            callbacks=_callbacks(
                tmp_path,
                store=store,
                node="e6_pilot",
                stage="E6",
                phase="pilot",
                source_kind="e6_interface_fit",
                count=2,
                launch_count=[],
            ),
            scheduler=scheduler,
            lock_path=tmp_path / "auxiliary-heartbeat-resume.lock",
        )

        driver.resume_dispatch(reason="fresh auxiliary heartbeat verified")
        assert store.dispatch_control() == ("RUN", None)
        evidence = store.dispatch_running_recovery_evidence()
        assert evidence is not None
        assert evidence["kind"] == (
            "formal_experiment_auxiliary_dispatch_running_recovery"
        )


def test_auxiliary_duplicate_adoption_mapping_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    launch_count: list[int] = []
    with ExperimentOperatorStore(
        tmp_path / "mismatch.sqlite3",
        run_id="mismatch-auxiliary",
        clock_ns=clock,
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e6_pilot", 0, "E6", "pilot", "2", 2),)
        )
        callbacks = _callbacks(
            tmp_path,
            store=store,
            node="e6_pilot",
            stage="E6",
            phase="pilot",
            source_kind="e6_interface_fit",
            count=2,
            launch_count=launch_count,
        )
        controller = FormalExperimentDagController(store=store, callbacks=callbacks)
        assert controller.run_once().action == "WAITING"
        assert controller.run_once().action == "WAITING"
        assert controller.run_once().action == "WAITING"
        assert controller.run_once().action == "MATERIALIZED"
        group = store.latest_controller_auxiliary_group("e6_pilot")
        assert group is not None
        good = callbacks.auxiliary_adoptions
        assert good is not None
        spec = callbacks.auxiliary_plan
        assert spec is not None
        group_spec = spec("e6_pilot", None)
        assert group_spec is not None
        rows = good(
            "e6_pilot",
            ControllerArtifactBinding(
                store.controller_node("e6_pilot")["node_materialization_path"],
                store.controller_node("e6_pilot")["node_materialization_sha256"],
            ),
            group_spec,
        )
        mismatched = (replace(rows[0], adoption_key="different"), *rows[1:])
        with pytest.raises(ExperimentOperatorError, match="mapping differs"):
            store.adopt_controller_auxiliary_jobs(
                node="e6_pilot",
                group_id=group_spec.group_id,
                group_attempt=1,
                adoptions=mismatched,
            )
