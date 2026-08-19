from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.orchestration.experiment_operator import (
    CellAttemptSpec,
    ControllerArtifactBinding,
    ExperimentOperatorError,
    ExperimentOperatorStore,
    FormalExperimentSchedulerDaemon,
    InterferenceEnvelope,
    MetricRecord,
    ProcessObservation,
    QueuedCommandSpec,
    RecoveredProcessStart,
    SchedulerCallbacks,
    SchedulerCycleResult,
    SingletonOperatorLock,
    StagePlanEntry,
    WorkerHeartbeat,
)
from lightcone_spec.orchestration.formal_cell_worker import FormalCellWorkerSpec
from lightcone_spec.orchestration.formal_experiment_controller import (
    DagCellLaunch,
    DagControllerCallbacks,
    DagControllerStep,
    DagExecutionPlan,
    DagMaterialization,
    DagReduction,
    FormalExperimentDagBlocked,
)
from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
    AuxiliaryInputCatalogBinding,
    DriverFileBinding,
    FormalDagDriverCycle,
    FormalSingleOperatorDagDriver,
    IsolatedInterferenceGateResolver,
    ProductionFormalDagCallbackBuilder,
    RetainedFutureDependencyManifest,
    _explicit_headline_metric_payload_rows,
    _file_sha256,
    _parser,
    _preserve_partial_directory,
    _publish_no_replace,
    _resolve_completed_e0_onlinespec_authority,
    formal_single_operator_dag_code_capabilities,
    load_retained_future_dependency_manifest,
)


def test_stop_cycle_releases_driver_flock_for_explicit_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ExperimentOperatorStore(
        tmp_path / "stop.sqlite3", run_id="driver-stop"
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("preflight", 0, "preflight", "fixture", "1", 1),)
        )
        driver = FormalSingleOperatorDagDriver(
            store=store,
            callbacks=_callbacks(tmp_path),
            scheduler=None,
            lock_path=tmp_path / "stop.lock",
        )
        cycle = FormalDagDriverCycle(
            DagControllerStep("preflight", "WAITING", "watchdog stop"),
            SchedulerCycleResult((), (), "STOP", "source timeout"),
        )
        assert cycle.run_state == "STOPPED"
        monkeypatch.setattr(driver, "_cycle_unlocked", lambda: cycle)

        driver.run_forever()

        with SingletonOperatorLock(tmp_path / "stop.lock"):
            pass


def _sha(character: str) -> str:
    return character * 64


def _binding(root: Path, name: str) -> ControllerArtifactBinding:
    path = (root / name).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(name + "\n", encoding="utf-8")
    return ControllerArtifactBinding.bind(path)


def _patch_completed_e0_terminals(
    monkeypatch: pytest.MonkeyPatch,
    *,
    valid_count: int,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_e0_compatibility
    from lightcone_spec.orchestration import formal_e0_compatibility_physical

    paths = tuple(f"/unused/e0-terminal-{index:03d}.json" for index in range(108))
    terminals = {
        path: type(
            "Terminal",
            (),
            {
                "key": ("model", "backend", f"task-{index:03d}"),
                "disposition": "VALID" if index < valid_count else "N/A",
                "sha256": _sha(format(index, "x")[-1]),
            },
        )()
        for index, path in enumerate(paths)
    }
    monkeypatch.setattr(
        formal_e0_compatibility_physical,
        "completed_e0_compatibility_terminal_paths",
        lambda _path: paths,
    )
    monkeypatch.setattr(
        formal_single_operator_e0_compatibility,
        "load_e0_compatibility_probe_terminal",
        terminals.__getitem__,
    )


def test_e0_bound_authority_is_recorded_but_unused_for_all_na(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_completed_e0_terminals(monkeypatch, valid_count=0)
    authority_path = (tmp_path / "onlinespec-authority.json").resolve()
    authority_path.write_text("{}\n", encoding="utf-8")
    descriptor = type(
        "Descriptor",
        (),
        {
            "node": "e0_tuning",
            "campaign": type(
                "Campaign",
                (),
                {"absolute_path": "/unused/campaign.json", "raw_sha256": _sha("a")},
            )(),
            "onlinespec_source_authority": DriverFileBinding.bind(authority_path),
            "publication_output_path": str((tmp_path / "bundle.json").resolve()),
            "sha256": _sha("b"),
        },
    )()

    assert _resolve_completed_e0_onlinespec_authority(descriptor=descriptor) is None
    record = json.loads(
        (tmp_path / "onlinespec-source-authority-disposition.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["valid_decision_count"] == 0
    assert record["disposition"] == "BOUND_UNUSED_ALL_NA"
    assert record["claimed_by_compatibility_bundle"] is False


def test_e0_valid_probe_requires_and_deep_opens_bound_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import formal_registry

    _patch_completed_e0_terminals(monkeypatch, valid_count=1)
    base = {
        "node": "e0_tuning",
        "campaign": type(
            "Campaign",
            (),
            {"absolute_path": "/unused/campaign.json", "raw_sha256": _sha("a")},
        )(),
        "publication_output_path": str((tmp_path / "bundle.json").resolve()),
        "sha256": _sha("b"),
    }
    with pytest.raises(ValueError, match="require bound OnlineSPEC"):
        _resolve_completed_e0_onlinespec_authority(
            descriptor=type(
                "Descriptor", (), {**base, "onlinespec_source_authority": None}
            )()
        )

    authority_path = (tmp_path / "onlinespec-authority.json").resolve()
    authority_path.write_text("{}\n", encoding="utf-8")
    revalidated: list[bool] = []
    authority = type(
        "Authority",
        (),
        {"revalidate": lambda self: revalidated.append(True)},
    )()
    monkeypatch.setattr(
        formal_registry,
        "e0_onlinespec_source_authority_from_dict",
        lambda _value: authority,
    )
    descriptor = type(
        "Descriptor",
        (),
        {
            **base,
            "onlinespec_source_authority": DriverFileBinding.bind(authority_path),
        },
    )()
    assert (
        _resolve_completed_e0_onlinespec_authority(descriptor=descriptor) is authority
    )
    assert revalidated == [True]


def _launch(root: Path, *, node: str, stage: str) -> DagCellLaunch:
    cell_id = f"{node}:cell"
    cell_root = (root / "cells" / node).resolve()
    command = QueuedCommandSpec(
        cell_id=cell_id,
        attempt=1,
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
        launch_compatibility_key=f"{node}:fixture",
        required_gpu_count=2,
        timing_class="EXCLUSIVE",
        predicted_high_water_bytes=1,
        monitored_path=str(root.resolve()),
        log_path=str(cell_root / "command.log"),
        expected_terminal_path=str(cell_root / "terminal.json"),
        expected_junit_path=str(cell_root / "junit.xml"),
        expected_raw_log_path=str(cell_root / "raw.json"),
        atomic_pointer_path=str(cell_root / "pointer.json"),
        child_exit_receipt_path=str(cell_root / "exit.json"),
    )
    return DagCellLaunch(
        attempt=CellAttemptSpec(
            cell_id=cell_id,
            attempt=1,
            stage=stage,
            phase="fixture",
            block="block-00",
            seed=1,
            scientific_axes={"task": "fixture", "node": node},
            identity={
                "source_sha256": _sha("a"),
                "patch_sha256": _sha("b"),
                "registry_sha256": _sha("c"),
            },
            command_sha256=command.command_sha256,
            output_directory=str(cell_root),
        ),
        command=command,
    )


def _callbacks(root: Path) -> DagControllerCallbacks:
    stage_by_node = {"preflight": "preflight", "e3a": "E3a"}

    def materialize(
        node: str,
        _predecessor: ControllerArtifactBinding | None,
    ) -> DagMaterialization:
        launch = _launch(root, node=node, stage=stage_by_node[node])
        return DagMaterialization(
            materialization=_binding(root, f"{node}/materialization.json"),
            node_materialization=_binding(root, f"{node}/node-materialization.json"),
            expected_cell_ids=(launch.attempt.cell_id,),
        )

    def plan(
        node: str,
        _node_materialization: ControllerArtifactBinding,
    ) -> DagExecutionPlan:
        launch = _launch(root, node=node, stage=stage_by_node[node])
        return DagExecutionPlan(
            execution_source=_binding(root, f"{node}/execution-source.json"),
            prepared_launch=None,
            launches=(launch,),
        )

    def actual_results(
        node: str,
        _attempts: tuple[dict[str, object], ...],
    ) -> dict[str, str]:
        return {f"{node}:cell": _binding(root, f"{node}/actual.json").absolute_path}

    def reduce(
        node: str,
        _node_materialization: ControllerArtifactBinding,
        _actuals: dict[str, str],
    ) -> DagReduction:
        return DagReduction(
            decision=_binding(root, f"{node}/decision.json"),
            completion=_binding(root, f"{node}/completion.json"),
        )

    return DagControllerCallbacks(materialize, plan, actual_results, reduce)


def _finish_fixture_attempt(
    store: ExperimentOperatorStore,
    root: Path,
    *,
    node: str,
) -> None:
    launch = _launch(
        root,
        node=node,
        stage="preflight" if node == "preflight" else "E3a",
    )
    actual = _binding(root, f"{node}/actual.json")
    store.mark_running_before_spawn(
        launch.attempt.cell_id,
        1,
        assigned_gpu_uuids=("GPU-0", "GPU-1"),
        started_at_ns=10,
    )
    store.attach_process(launch.attempt.cell_id, 1, pid=101, pgid=101)
    store.finish_attempt(
        launch.attempt.cell_id,
        1,
        status="COMPLETE",
        exit_code=0,
        terminal_sha256=_sha("d"),
        junit_sha256=_sha("e"),
        raw_log_sha256=_sha("f"),
        evidence_files={actual.absolute_path: actual.sha256},
        included_in_analysis=True,
        exclusion_reason=None,
        finished_at_ns=20,
    )


def test_driver_restart_advances_preflight_then_only_the_next_node(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operator.sqlite3"
    plan = (
        StagePlanEntry("preflight", 0, "preflight", "fixture", "1", 1),
        StagePlanEntry("e3a", 1, "E3a", "fixture", "1", 1),
    )
    with ExperimentOperatorStore(database, run_id="driver-restart") as store:
        store.initialize_stage_plan(plan)
        driver = FormalSingleOperatorDagDriver(
            store=store,
            callbacks=_callbacks(tmp_path),
            scheduler=None,
            lock_path=tmp_path / "operator.lock",
        )
        assert driver.run_once().controller.action == "MATERIALIZED"
        assert store.controller_node("e3a")["state"] == "UNMATERIALIZED"
        assert store.latest_stage_attempts("preflight") == ()
        assert store.latest_stage_attempts("e3a") == ()

    with ExperimentOperatorStore(database) as reopened:
        driver = FormalSingleOperatorDagDriver(
            store=reopened,
            callbacks=_callbacks(tmp_path),
            scheduler=None,
            lock_path=tmp_path / "operator.lock",
        )
        assert driver.run_once().controller.action == "PLANNED"
        assert len(reopened.latest_stage_attempts("preflight")) == 1
        assert reopened.latest_stage_attempts("e3a") == ()
        assert driver.run_once().controller.action == "WAITING"
        _finish_fixture_attempt(reopened, tmp_path, node="preflight")
        assert driver.run_once().controller.action == "REDUCED"
        assert reopened.controller_node("e3a")["state"] == "UNMATERIALIZED"
        assert driver.run_once().controller.action == "MATERIALIZED"
        assert reopened.latest_stage_attempts("e3a") == ()


def test_negative_decision_and_metric_precede_durable_downstream_block(
    tmp_path: Path,
) -> None:
    database = tmp_path / "scientific-block.sqlite3"
    materialize_calls: list[str] = []

    def callbacks_for(store: ExperimentOperatorStore) -> DagControllerCallbacks:
        base = _callbacks(tmp_path)

        def materialize(
            node: str,
            predecessor: ControllerArtifactBinding | None,
        ) -> DagMaterialization:
            materialize_calls.append(node)
            if node == "e3a":
                assert len(store._selection_rows()) == 1
                assert len(store._metric_rows()) == 1
                raise FormalExperimentDagBlocked(
                    "e3a: scientific stage blocked: NO_SAFE_SLO_WINNER"
                )
            return base.materialize(node, predecessor)

        def reduce(
            node: str,
            node_materialization: ControllerArtifactBinding,
            actuals: dict[str, str],
        ) -> DagReduction:
            reduction = base.reduce(node, node_materialization, actuals)
            store.record_selection_decision(
                decision_id="negative-preflight-selection",
                stage="preflight",
                phase="fixture",
                decision_kind="fixture_no_safe_winner",
                source_sha256=reduction.decision.sha256,
                decision={"status": "NO_SAFE_SLO_WINNER"},
            )
            store.record_metric(
                MetricRecord(
                    stage="preflight",
                    phase="fixture",
                    cell_id="preflight:cell",
                    attempt=1,
                    metric_name="scientific_candidate_eligible",
                    metric_kind="descriptive",
                    point_estimate=0.0,
                    ci_low=None,
                    ci_high=None,
                    independent_block_count=None,
                    request_count=None,
                    paired=None,
                    reducer_method="deterministic_negative_selection",
                    attributes={"status": "NO_SAFE_SLO_WINNER"},
                )
            )
            return reduction

        return DagControllerCallbacks(
            materialize=materialize,
            plan=base.plan,
            actual_results=base.actual_results,
            reduce=reduce,
        )

    plan = (
        StagePlanEntry("preflight", 0, "preflight", "fixture", "1", 1),
        StagePlanEntry("e3a", 1, "E3a", "fixture", "1", 1),
    )
    with ExperimentOperatorStore(database, run_id="scientific-block") as store:
        store.initialize_stage_plan(plan)
        driver = FormalSingleOperatorDagDriver(
            store=store,
            callbacks=callbacks_for(store),
            scheduler=None,
            lock_path=tmp_path / "scientific-block.lock",
        )
        assert driver.run_once().controller.action == "MATERIALIZED"
        assert driver.run_once().controller.action == "PLANNED"
        assert driver.run_once().controller.action == "WAITING"
        _finish_fixture_attempt(store, tmp_path, node="preflight")
        assert driver.run_once().controller.action == "REDUCED"
        assert store.controller_node("preflight")["state"] == "REDUCED"
        blocked = driver.run_once().controller
        assert blocked.action == "BLOCKED"
        assert FormalDagDriverCycle(blocked, None).changed is True
        assert store.controller_node("e3a")["state"] == "BLOCKED"
        assert store.controller_node("e3a")["materialization_path"] is None
        assert materialize_calls == ["preflight", "e3a"]

    with ExperimentOperatorStore(database) as reopened:
        driver = FormalSingleOperatorDagDriver(
            store=reopened,
            callbacks=callbacks_for(reopened),
            scheduler=None,
            lock_path=tmp_path / "scientific-block.lock",
        )
        assert driver.run_once().controller.action == "BLOCKED"
        assert materialize_calls == ["preflight", "e3a"]


def test_production_callbacks_translate_typed_scientific_stage_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_stages as stages

    builder = object.__new__(ProductionFormalDagCallbackBuilder)
    builder.nodes_root = tmp_path / "formal-dag-nodes"
    builder.nodes_root.mkdir()
    builder.clock_ns = lambda: 1
    builder.config = SimpleNamespace(repository_root=str(tmp_path))
    predecessor = _binding(tmp_path, "negative-predecessor.json")

    def blocked_materialization(**_kwargs: object) -> None:
        raise stages.FormalSingleOperatorStageBlocked("NO_SAFE_WINNER")

    monkeypatch.setattr(
        stages,
        "materialize_formal_single_operator_node",
        blocked_materialization,
    )
    with pytest.raises(
        FormalExperimentDagBlocked,
        match="e2_r1: scientific stage blocked: NO_SAFE_WINNER",
    ):
        builder.materialize("e2_r1", predecessor)

    node_materialization = _binding(tmp_path, "e2-r0-node-materialization.json")

    def blocked_reduction(**_kwargs: object) -> None:
        raise stages.FormalSingleOperatorStageBlocked("POWER_UNRESOLVED")

    monkeypatch.setattr(
        stages,
        "reduce_formal_single_operator_node",
        blocked_reduction,
    )
    with pytest.raises(
        FormalExperimentDagBlocked,
        match="e2_r0: scientific reduction blocked: POWER_UNRESOLVED",
    ):
        builder.reduce("e2_r0", node_materialization, {})


def test_driver_explicitly_recovers_blocked_node_and_dispatch_stop(
    tmp_path: Path,
) -> None:
    blocked = [True]
    materialization = _binding(tmp_path, "resume/materialization.json")
    node_materialization = _binding(tmp_path, "resume/node-materialization.json")

    def materialize(
        _node: str,
        _predecessor: ControllerArtifactBinding | None,
    ) -> DagMaterialization:
        if blocked[0]:
            raise FormalExperimentDagBlocked("fixture prerequisite missing")
        return DagMaterialization(
            materialization,
            node_materialization,
            ("preflight:cell",),
        )

    callbacks = DagControllerCallbacks(
        materialize=materialize,
        plan=lambda *_args: (_ for _ in ()).throw(AssertionError()),
        actual_results=lambda *_args: {},
        reduce=lambda *_args: (_ for _ in ()).throw(AssertionError()),
    )
    with ExperimentOperatorStore(
        tmp_path / "resume.sqlite3", run_id="driver-resume"
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("preflight", 0, "preflight", "fixture", "1", 1),)
        )
        driver = FormalSingleOperatorDagDriver(
            store=store,
            callbacks=callbacks,
            scheduler=None,
            lock_path=tmp_path / "resume.lock",
        )
        assert driver.run_once().controller.action == "BLOCKED"
        blocked[0] = False
        driver.resume_node(node="preflight", reason="fixture source restored")
        assert store.controller_node("preflight")["state"] == "UNMATERIALIZED"
        assert driver.run_once().controller.action == "MATERIALIZED"

        store.set_dispatch_stop("fixture watchdog stop")
        driver.resume_dispatch(reason="fixture watchdog condition cleared")
        assert store.dispatch_control() == ("RUN", None)
        with pytest.raises(ExperimentOperatorError, match="not STOPPED"):
            driver.resume_dispatch(reason="duplicate resume")


def _heartbeat_resume_driver(
    tmp_path: Path,
    *,
    heartbeat: WorkerHeartbeat | None,
) -> tuple[
    ExperimentOperatorStore,
    FormalSingleOperatorDagDriver,
    DagCellLaunch,
    int,
    str,
]:
    store = ExperimentOperatorStore(
        tmp_path / "heartbeat-resume.sqlite3",
        run_id="heartbeat-resume",
    )
    store.initialize_stage_plan(
        (StagePlanEntry("preflight", 0, "preflight", "fixture", "1", 1),)
    )
    store.configure_interference_envelope(
        InterferenceEnvelope("ISOLATED", ("GPU-0", "GPU-1"), _sha("8"))
    )
    launch = _launch(tmp_path, node="preflight", stage="preflight")
    store.materialize_attempt(launch.attempt)
    store.enqueue_command(launch.command)
    started_ns = time.time_ns() - 200_000_000_000
    receipt_sha256 = _sha("9")
    store.mark_running_before_spawn(
        launch.attempt.cell_id,
        1,
        assigned_gpu_uuids=("GPU-0", "GPU-1"),
        started_at_ns=started_ns,
    )
    store.attach_process(
        launch.attempt.cell_id,
        1,
        pid=101,
        pgid=101,
        process_start_receipt_sha256=receipt_sha256,
    )
    store.set_dispatch_stop("child_heartbeat_stale")
    callbacks = SchedulerCallbacks(
        launch=lambda *_args: (_ for _ in ()).throw(AssertionError()),
        process_probe=lambda pid, pgid: ProcessObservation(
            pid,
            True,
            pgid,
            "alive",
        ),
        log_size_bytes=lambda _command: 0,
        gpu_snapshot=lambda gpus: {gpu: {"utilization_percent": 0} for gpu in gpus},
        terminal_validator=lambda *_args: None,
        free_disk_bytes=lambda _path: 100 * 1024**3,
        recover_started_process=lambda _command: RecoveredProcessStart(
            101,
            101,
            started_ns + 1,
            receipt_sha256,
        ),
        worker_heartbeat=lambda _command: heartbeat,
        worker_heartbeat_required=lambda _command: True,
    )
    scheduler = FormalExperimentSchedulerDaemon(
        store,
        lock_path=tmp_path / "heartbeat-resume.lock",
        callbacks=callbacks,
    )
    driver = FormalSingleOperatorDagDriver(
        store=store,
        callbacks=_callbacks(tmp_path),
        scheduler=scheduler,
        lock_path=tmp_path / "heartbeat-resume.lock",
    )
    return store, driver, launch, started_ns, receipt_sha256


def test_driver_resumes_running_attempt_only_after_fresh_child_heartbeat(
    tmp_path: Path,
) -> None:
    heartbeat = WorkerHeartbeat(
        command_sha256=_launch(
            tmp_path,
            node="preflight",
            stage="preflight",
        ).command.command_sha256,
        worker_pid=102,
        sequence=1,
        observed_at_ns=time.time_ns(),
        phase="RUNNING",
    )
    store, driver, _launch_spec, _started, _receipt = _heartbeat_resume_driver(
        tmp_path,
        heartbeat=heartbeat,
    )
    try:
        driver.resume_dispatch(reason="fresh child heartbeat verified")

        assert store.dispatch_control() == ("RUN", None)
        recovery = store.dispatch_running_recovery_evidence()
        assert recovery is not None
        assert recovery["mode"] == "FRESH_CHILD_HEARTBEAT"
        assert recovery["manual_evidence"] is None
    finally:
        store.close()


def test_driver_requires_bound_manual_evidence_to_waive_stale_heartbeat(
    tmp_path: Path,
) -> None:
    store, driver, launch, _started, receipt_sha256 = _heartbeat_resume_driver(
        tmp_path,
        heartbeat=None,
    )
    try:
        with pytest.raises(ExperimentOperatorError, match="fresh child heartbeats"):
            driver.resume_dispatch(reason="unverified stale heartbeat")
        assert store.dispatch_control() == ("STOP", "child_heartbeat_stale")

        manual_path = (tmp_path / "manual-resume-evidence.json").resolve()
        _publish_no_replace(
            manual_path,
            {
                "schema_version": 1,
                "kind": "formal_experiment_manual_dispatch_resume_evidence",
                "run_id": store.run_id,
                "dispatch_stop_reason": "child_heartbeat_stale",
                "observed_at_ns": time.time_ns(),
                "operator_observation": "wrapper and child inspected read-only",
                "running_processes": [
                    {
                        "cell_id": launch.command.cell_id,
                        "attempt": launch.command.attempt,
                        "command_sha256": launch.command.command_sha256,
                        "pid": 101,
                        "pgid": 101,
                        "process_start_receipt_sha256": receipt_sha256,
                        "covered_attempts": [
                            {
                                "cell_id": launch.command.cell_id,
                                "attempt": launch.command.attempt,
                            }
                        ],
                    }
                ],
            },
        )
        driver.resume_dispatch(
            reason="manual process evidence verified",
            manual_evidence_path=manual_path,
        )
        assert store.dispatch_control() == ("RUN", None)
        recovery = store.dispatch_running_recovery_evidence()
        assert recovery is not None
        assert recovery["mode"] == "MANUAL_OPERATOR_EVIDENCE"

        result = driver.scheduler.run_once()
        assert result.dispatch_state == "RUN"
        assert result.reconciled == (
            (launch.command.cell_id, 1, "MANUAL_HEARTBEAT_WAIVER"),
        )
        assert store.attempt(launch.command.cell_id, 1)["termination_reason"] is None
    finally:
        store.close()


def test_partial_cpu_plan_is_preserved_before_recovery(tmp_path: Path) -> None:
    partial = (tmp_path / "execution" / "work").resolve()
    partial.mkdir(parents=True)
    evidence = partial / "prepared-launch-draft.json"
    evidence.write_text("partial-but-preserved\n", encoding="utf-8")
    destination = _preserve_partial_directory(
        partial,
        label="fixture partial plan",
    )
    assert not partial.exists()
    assert (destination / evidence.name).read_text(encoding="utf-8") == (
        "partial-but-preserved\n"
    )
    partial.mkdir()
    second = _preserve_partial_directory(partial, label="fixture partial plan")
    assert second.name == "attempt-0002"


def test_worker_spec_and_command_identity_are_no_replace(tmp_path: Path) -> None:
    repository = (tmp_path / "repository").resolve()
    repository.mkdir()
    node = repository / "node-materialization.json"
    node.write_text("node\n", encoding="utf-8")
    evidence = (tmp_path / "evidence").resolve()
    evidence.mkdir()
    spec = FormalCellWorkerSpec(
        schema_version=1,
        kind="formal_single_operator_cell_worker",
        cell_id="fixture-cell",
        attempt=1,
        repository_root=str(repository),
        node_materialization_path=str(node),
        actual_result_path=str(evidence / "actual.json"),
        evidence_root=str(evidence),
        evidence_manifest_path=str(evidence / "manifest.json"),
        job_argv=(sys.executable, "-c", "raise SystemExit(0)"),
        failure_class_on_nonzero="SCIENTIFIC",
        included_in_analysis_on_complete=True,
        complete_exclusion_reason=None,
    )
    path = evidence / "worker-spec.json"
    digest = ProductionFormalDagCallbackBuilder._publish_or_reopen_worker_spec(
        spec, path
    )
    assert digest == _file_sha256(path)
    assert (
        ProductionFormalDagCallbackBuilder._publish_or_reopen_worker_spec(spec, path)
        == digest
    )
    command = QueuedCommandSpec(
        cell_id=spec.cell_id,
        attempt=1,
        argv=(sys.executable, "-m", "lightcone_spec.orchestration.formal_cell_worker"),
        launch_compatibility_key="fixture-compatible",
        required_gpu_count=1,
        timing_class="HEADLINE",
        predicted_high_water_bytes=1,
        monitored_path=str(tmp_path.resolve()),
        log_path=str(evidence / "command.log"),
        expected_terminal_path=str(evidence / "terminal.json"),
        expected_junit_path=str(evidence / "junit.xml"),
        expected_raw_log_path=str(evidence / "raw.json"),
        atomic_pointer_path=str(evidence / "pointer.json"),
        child_exit_receipt_path=str(evidence / "exit.json"),
        environment=(("LIGHTCONE_CELL_WORKER_SPEC_SHA256", digest),),
    )
    changed_digest = _sha("9")
    changed_command = replace(
        command,
        environment=(("LIGHTCONE_CELL_WORKER_SPEC_SHA256", changed_digest),),
    )
    assert command.command_sha256 != changed_command.command_sha256
    with pytest.raises(FormalExperimentDagBlocked, match="another command"):
        ProductionFormalDagCallbackBuilder._publish_or_reopen_worker_spec(
            replace(spec, job_argv=(sys.executable, "-c", "raise SystemExit(1)")),
            path,
        )


@pytest.mark.parametrize("member_count", (2, 32))
def test_prepared_tp1_group_requires_authority_then_materializes_one_shared_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_count: int,
) -> None:
    from lightcone_spec.experiments import formal_preflight_inputs
    from lightcone_spec.orchestration.formal_cell_worker import (
        publish_formal_cell_worker_spec,
    )
    from lightcone_spec.orchestration.formal_serving_session_group_production import (
        formal_serving_session_group_shared_evidence_bound_bytes,
        revalidate_formal_serving_session_group_production_spec,
    )
    from lightcone_spec.runtime import preflight_runner

    fixture_path = Path(__file__).with_name("test_formal_serving_session_group.py")
    fixture_spec = importlib.util.spec_from_file_location(
        "_dag_resident_group_fixture", fixture_path
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)

    authority_binding, authority = fixture._published_authority(
        tmp_path / "authority",
        monkeypatch,
        method_family="static",
    )
    specs = []
    launches = []
    node_path = (tmp_path / "node-materialization.json").resolve()
    _publish_no_replace(node_path, {"kind": "test-node-materialization"})
    node_binding = ControllerArtifactBinding.bind(node_path)
    repository_root = Path(__file__).parents[1].resolve()
    for index in range(member_count):
        config = fixture._config(label=f"dag-resident-{index}")
        compile_launch = fixture._producer_generated_launch(
            tmp_path,
            label=f"dag-resident-{index}",
            config=config,
            port=29_000 + index,
        )
        group_spec = fixture._group_spec(
            tmp_path,
            index=800 + index,
            config=config,
            launch=compile_launch,
            method_family="static",
            source_snapshot_sha256=authority.source_snapshot_sha256,
            protocol_lock_sha256=authority.protocol_lock_sha256,
            inventory_sha256=authority.inventory_sha256,
        )
        specs.append(group_spec)
        control_root = (tmp_path / "controls" / str(index)).resolve()
        evidence_root = (tmp_path / "evidence" / str(index)).resolve()
        evidence_root.mkdir(parents=True)
        worker = FormalCellWorkerSpec(
            schema_version=1,
            kind="formal_single_operator_cell_worker",
            cell_id=group_spec.materialized_cell_id,
            attempt=1,
            repository_root=str(repository_root),
            node_materialization_path=node_binding.absolute_path,
            actual_result_path=str(evidence_root / "actual.json"),
            evidence_root=str(evidence_root),
            evidence_manifest_path=str(evidence_root / "manifest.json"),
            job_argv=(sys.executable, "-c", "pass"),
            failure_class_on_nonzero="SCIENTIFIC",
            included_in_analysis_on_complete=True,
            complete_exclusion_reason=None,
        )
        worker_path = (evidence_root / "worker-spec.json").resolve()
        publish_formal_cell_worker_spec(worker, worker_path)
        command = QueuedCommandSpec(
            cell_id=group_spec.materialized_cell_id,
            attempt=1,
            argv=(
                sys.executable,
                "-m",
                "lightcone_spec.orchestration.formal_cell_worker",
                "--spec",
                str(worker_path),
            ),
            launch_compatibility_key="qwen3-8b:dflash:tp1:shared",
            required_gpu_count=1,
            timing_class="HEADLINE",
            predicted_high_water_bytes=4_096,
            monitored_path=str(control_root),
            log_path=str(control_root / "command.log"),
            expected_terminal_path=str(control_root / "terminal.json"),
            expected_junit_path=str(control_root / "junit.xml"),
            expected_raw_log_path=str(control_root / "raw.json"),
            atomic_pointer_path=str(control_root / "pointer.json"),
            child_exit_receipt_path=str(control_root / "exit.json"),
        )
        attempt = CellAttemptSpec(
            cell_id=command.cell_id,
            attempt=1,
            stage="E3b",
            phase="final",
            block=f"block-{index:02d}",
            seed=17 + index,
            scientific_axes={"method": "Static", "trace": index},
            identity={
                "source_sha256": authority.source_snapshot_sha256,
                "patch_sha256": _sha("d"),
                "registry_sha256": _sha("e"),
            },
            command_sha256=command.command_sha256,
            scientific_command_sha256=fixture._sha(
                f"dag-resident-scientific-command-{index}"
            ),
            output_directory=group_spec.output_directory,
        )
        launches.append(DagCellLaunch(attempt=attempt, command=command))

    builder = object.__new__(ProductionFormalDagCallbackBuilder)
    builder.config = SimpleNamespace(session_reset_authority_directory=None)
    builder.python_executable = str(Path(sys.executable).resolve(strict=True))
    monkeypatch.setattr(builder, "_inventory_gpu_uuids", lambda: ("GPU-A", "GPU-B"))
    preflight_path = (tmp_path / "preflight-inputs.json").resolve()
    _publish_no_replace(preflight_path, {"kind": "test-preflight-inputs"})
    exactness_path = (tmp_path / "exactness-assignment.json").resolve()
    _publish_no_replace(exactness_path, {"kind": "test-exactness-assignment"})
    monkeypatch.setattr(builder, "_preflight_inputs_path", lambda: preflight_path)
    monkeypatch.setattr(
        formal_preflight_inputs.FormalPreflightExecutionInputs,
        "from_dict",
        classmethod(
            lambda _cls, _value: SimpleNamespace(
                exactness_assignment=SimpleNamespace(absolute_path=str(exactness_path))
            )
        ),
    )
    monkeypatch.setattr(
        preflight_runner.ExactnessPreflightAssignment,
        "load",
        classmethod(
            lambda _cls, _path: SimpleNamespace(
                nvidia_smi_executable=str(Path(sys.executable).resolve(strict=True))
            )
        ),
    )

    standalone, groups = builder._materialize_serving_session_groups(
        node="e3b_final",
        work_root=(tmp_path / "without-authority").resolve(),
        node_materialization=node_binding,
        launches=tuple(launches),
        specs=tuple(specs),
    )
    assert standalone == tuple(launches)
    assert groups == ()

    monkeypatch.setattr(
        builder,
        "_session_reset_authority_bindings",
        lambda: (authority_binding,),
    )
    standalone, groups = builder._materialize_serving_session_groups(
        node="e3b_final",
        work_root=(tmp_path / "with-authority").resolve(),
        node_materialization=node_binding,
        launches=tuple(launches),
        specs=tuple(specs),
    )
    assert standalone == ()
    assert len(groups) == 1
    group = groups[0]
    assert group.group_kind == "tp1_serving_session"
    assert len(group.members) == member_count
    assert len({member.command.command_sha256 for member in group.members}) == 1
    assert (
        len({member.attempt.scientific_command_sha256 for member in group.members})
        == member_count
    )
    expected_high_water = (
        member_count * 4_096
        + formal_serving_session_group_shared_evidence_bound_bytes(member_count)
    )
    assert {member.command.predicted_high_water_bytes for member in group.members} == {
        expected_high_water
    }
    production_path = Path(
        dict(group.members[0].command.environment)[
            "LIGHTCONE_FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_PATH"
        ]
    )
    production = revalidate_formal_serving_session_group_production_spec(
        production_path
    )
    assert tuple(member.cell_id for member in production.spec.members) == tuple(
        member.attempt.cell_id for member in group.members
    )
    assert production.spec.shared_evidence_bound_bytes == (
        formal_serving_session_group_shared_evidence_bound_bytes(member_count)
    )


def test_cell_launch_consumes_deep_runtime_contract_for_outer_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec import config as config_module
    from lightcone_spec.experiments import formal_single_operator_prepared_launch
    from lightcone_spec.orchestration import formal_physical_dispatch
    from lightcone_spec.runtime import compile_runner

    repository = (tmp_path / "repository").resolve()
    repository.mkdir()
    run_root = (tmp_path / "run").resolve()
    run_root.mkdir()
    cell_root = run_root / "cell"
    cell_root.mkdir()
    node_path = (tmp_path / "node-materialization.json").resolve()
    node_path.write_text("node\n", encoding="utf-8")
    run_plan_path = cell_root / "formal-serving-run-plan.json"
    run_plan_path.write_text("plan\n", encoding="utf-8")
    launch_path = (tmp_path / "launch.json").resolve()
    launch_path.write_text("launch\n", encoding="utf-8")
    progress_logs = tuple(
        str((cell_root / f"server-{index}.log").resolve()) for index in range(3)
    )
    plan_sha = _sha("a")
    scientific_sha = _sha("b")
    contract = SimpleNamespace(
        plan_sha256=plan_sha,
        scientific_command_sha256=scientific_sha,
        outer_max_runtime_seconds=12_345,
        progress_log_paths=progress_logs,
    )
    monkeypatch.setattr(
        formal_physical_dispatch,
        "formal_serving_process_runtime_contract",
        lambda path: contract if Path(path) == run_plan_path else None,
    )
    fake_launch = SimpleNamespace(run_config_path=str(tmp_path / "config.json"))
    monkeypatch.setattr(
        compile_runner.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, _path: fake_launch),
    )
    monkeypatch.setattr(config_module, "load_run_config", lambda _path: object())
    monkeypatch.setattr(
        formal_single_operator_prepared_launch,
        "formal_single_operator_launch_compatibility_key",
        lambda **_kwargs: "source-bound-launch-key",
    )
    builder = object.__new__(ProductionFormalDagCallbackBuilder)
    builder.config = SimpleNamespace(
        repository_root=str(repository),
        run_root=str(run_root),
    )
    builder.python_executable = str(Path(sys.executable).resolve(strict=True))
    monkeypatch.setattr(builder, "_inventory_gpu_uuids", lambda: ("GPU-0", "GPU-1"))
    monkeypatch.setattr(
        builder,
        "_publish_or_reopen_worker_spec",
        lambda _spec, _path: _sha("c"),
    )

    def attempt_spec(**values: object) -> CellAttemptSpec:
        command = values["command"]
        assert type(command) is QueuedCommandSpec
        return CellAttemptSpec(
            cell_id=command.cell_id,
            attempt=command.attempt,
            stage="E3a",
            phase="selection",
            block="pilot-00",
            seed=1,
            scientific_axes={"task": "serving"},
            identity={
                "source_sha256": _sha("d"),
                "patch_sha256": _sha("e"),
                "registry_sha256": _sha("f"),
            },
            command_sha256=command.command_sha256,
            scientific_command_sha256=values["scientific_command_sha256"],
            output_directory=str(values["output_directory"]),
        )

    monkeypatch.setattr(builder, "_attempt_spec", attempt_spec)
    plan = SimpleNamespace(
        sha256=plan_sha,
        gpu_uuids=("GPU-0",),
        launch_manifest=SimpleNamespace(absolute_path=str(launch_path)),
    )
    cell = SimpleNamespace(
        cell_id="e3a:source-bound-timeout",
        stage="E3a",
        backend="DFlash",
        method_role="Static",
        model="Qwen3-8B",
        publication_policy="first_ready",
        recipe_sha256=None,
        task="serving",
        dimensions=(("block", "pilot-00"),),
    )

    launch = builder._cell_launch(
        node="e3a",
        cell=cell,
        attempt_number=1,
        node_materialization=ControllerArtifactBinding.bind(node_path),
        execution_source_path=str(tmp_path / "unused-execution-source.json"),
        run_plan=plan,
        run_plan_path=run_plan_path,
        run_root=cell_root,
        physical_kind="serving",
    )

    assert launch.command.max_runtime_seconds == 12_345
    assert launch.attempt.scientific_command_sha256 == scientific_sha
    assert json.loads(
        dict(launch.command.environment)["LIGHTCONE_OPERATOR_PROGRESS_LOG_PATHS_JSON"]
    ) == list(progress_logs)


def test_retry_builder_uses_fresh_paths_and_stable_scientific_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_early_execution,
        formal_single_operator_stages,
    )
    from lightcone_spec.orchestration import formal_physical_dispatch

    repository = (tmp_path / "repository").resolve()
    repository.mkdir()
    run_root = (tmp_path / "run").resolve()
    run_root.mkdir()
    node_path = (tmp_path / "node-materialization.json").resolve()
    node_path.write_text("node\n", encoding="utf-8")
    execution_path = (tmp_path / "execution-source.json").resolve()
    execution_path.write_text("source\n", encoding="utf-8")
    scientific_sha = _sha("8")
    old_root = (run_root / "old-attempt").resolve()
    old_operator = old_root / "operator"
    old_operator.mkdir(parents=True)
    previous_command = QueuedCommandSpec(
        cell_id="e3a:retry-cell",
        attempt=1,
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
        launch_compatibility_key="source-bound-launch-key",
        required_gpu_count=1,
        timing_class="HEADLINE",
        predicted_high_water_bytes=1,
        monitored_path=str(run_root),
        log_path=str(old_operator / "command.log"),
        expected_terminal_path=str(old_operator / "terminal.json"),
        expected_junit_path=str(old_operator / "junit.xml"),
        expected_raw_log_path=str(old_operator / "raw.json"),
        atomic_pointer_path=str(old_operator / "pointer.json"),
        child_exit_receipt_path=str(old_operator / "exit.json"),
    )
    prior = {
        "status": "FAILED",
        "failure_code": "INFRASTRUCTURE:SPAWN_FAILED",
        "retry_decision": "RETRY_INFRASTRUCTURE_AUTOMATIC",
        "scientific_command_sha256": scientific_sha,
        "stage": "E3a",
        "phase": "selection",
    }
    controller = {
        "node": "e3a",
        "stage": "E3a",
        "phase": "selection",
        "execution_source_path": str(execution_path),
        "node_materialization_path": str(node_path),
    }
    builder = object.__new__(ProductionFormalDagCallbackBuilder)
    builder.store = SimpleNamespace(
        attempt=lambda *_args: prior,
        controller_nodes=lambda: (controller,),
    )
    builder.config = SimpleNamespace(repository_root=str(repository))
    builder.nodes_root = run_root / "formal-dag-nodes"
    builder.nodes_root.mkdir()
    builder.python_executable = str(Path(sys.executable).resolve(strict=True))
    cell = SimpleNamespace(cell_id=previous_command.cell_id)
    monkeypatch.setattr(
        formal_single_operator_stages,
        "rebuild_formal_single_operator_node_materialization",
        lambda _path: SimpleNamespace(materialization=SimpleNamespace(cells=(cell,))),
    )
    monkeypatch.setattr(
        formal_single_operator_early_execution,
        "materialize_formal_single_operator_early_run_plan_inputs",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        formal_physical_dispatch,
        "materialize_formal_single_operator_serving_run_plan",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        builder,
        "_preflight_inputs_path",
        lambda: (tmp_path / "preflight-inputs.json").resolve(),
    )

    def cell_launch(**values: object) -> DagCellLaunch:
        attempt_number = int(values["attempt_number"])
        retry_root = Path(values["run_root"])
        operator = retry_root / "operator"
        operator.mkdir()
        command = QueuedCommandSpec(
            cell_id=previous_command.cell_id,
            attempt=attempt_number,
            argv=(sys.executable, "-c", "raise SystemExit(0)"),
            launch_compatibility_key="source-bound-launch-key",
            required_gpu_count=1,
            timing_class="HEADLINE",
            predicted_high_water_bytes=1,
            monitored_path=str(run_root),
            log_path=str(operator / "command.log"),
            expected_terminal_path=str(operator / "terminal.json"),
            expected_junit_path=str(operator / "junit.xml"),
            expected_raw_log_path=str(operator / "raw.json"),
            atomic_pointer_path=str(operator / "pointer.json"),
            child_exit_receipt_path=str(operator / "exit.json"),
        )
        attempt = CellAttemptSpec(
            cell_id=command.cell_id,
            attempt=command.attempt,
            stage="E3a",
            phase="selection",
            block="pilot-00",
            seed=1,
            scientific_axes={"task": "serving"},
            identity={
                "source_sha256": _sha("a"),
                "patch_sha256": _sha("b"),
                "registry_sha256": _sha("c"),
            },
            command_sha256=command.command_sha256,
            scientific_command_sha256=scientific_sha,
            output_directory=str(retry_root),
        )
        return DagCellLaunch(attempt, command)

    monkeypatch.setattr(builder, "_cell_launch", cell_launch)

    retry_attempt, retry_command = builder.retry_attempt(previous_command, 2)

    assert retry_attempt.attempt == retry_command.attempt == 2
    assert retry_attempt.scientific_command_sha256 == scientific_sha
    assert retry_attempt.output_directory != str(old_root)
    assert not {
        previous_command.log_path,
        previous_command.expected_terminal_path,
        previous_command.expected_raw_log_path,
    } & {
        retry_command.log_path,
        retry_command.expected_terminal_path,
        retry_command.expected_raw_log_path,
    }


def test_interference_fallback_never_authorizes_concurrent_headline(
    tmp_path: Path,
) -> None:
    completion = _binding(tmp_path, "interference/completion.json")
    envelope = IsolatedInterferenceGateResolver().resolve(
        completion=completion,
        actual_result_paths={},
        gpu_uuids=("GPU-0", "GPU-1"),
    )
    assert envelope == InterferenceEnvelope(
        "ISOLATED", ("GPU-0", "GPU-1"), completion.sha256
    )
    assert envelope.mode != "DUAL_SINGLE"


def test_code_capability_is_complete_and_artifact_independent() -> None:
    rows = formal_single_operator_dag_code_capabilities()
    assert len(rows) == 21
    assert tuple(row.node for row in rows) == (
        "preflight",
        "e3a",
        "tts_cal",
        "e1",
        "e2_r0",
        "e2_r1",
        "e2_r2",
        "e2_r3",
        "e4_screen",
        "e4_local",
        "e4_profiler",
        "e3b_pilot",
        "e3b_final",
        "e1a",
        "e5_pilot",
        "e5_final",
        "e6_pilot",
        "e6_final",
        "e0_tuning",
        "e0_pilot",
        "e0_final",
    )
    assert all(row.ready and row.blocker is None for row in rows)


def _contrast(name: str) -> dict[str, object]:
    return {
        "name": name,
        "block_ids": ["block-00", "block-01"],
        "mean_log_ratio": 0.1,
        "mean_relative_gain": 0.105,
        "ci_lower_relative_gain": 0.01,
        "ci_upper_relative_gain": 0.2,
        "raw_p_value": 0.01,
        "confidence": 0.95,
        "independent_unit": "paired_block",
    }


def _family(node: str) -> dict[str, object]:
    family: dict[str, object] = {
        "family_sha256": _sha("1"),
        "result_sha256": _sha("2"),
        "block_count": 12,
        "request_count": 1200,
        "paired": True,
        "reducer": "paired_block_bca",
    }
    if node == "e0_final":
        family.update(
            {
                "compatibility_decision_id": _sha("3"),
                "load": "common_slo_load",
                "contrasts": {"lightcone_vs_static": _contrast("lightcone_vs_static")},
                "holm_primary": [],
            }
        )
    else:
        family.update(
            {
                "dimensions": [["context", 4096]],
                "primary_contrasts": [_contrast("lightcone_vs_static")],
                "mechanism_contrasts": [_contrast("l0_naive_vs_tts")],
                "holm_decisions": [],
                "target_only_gate": {
                    "contrast": _contrast("lightcone_vs_target_only"),
                    "passed": True,
                },
            }
        )
    if node == "e3b_final":
        family["hierarchical_intervals"] = [
            {
                "name": "lightcone_vs_static",
                "mean_log_ratio": 0.1,
                "mean_relative_gain": 0.105,
                "ci_lower_relative_gain": 0.01,
                "ci_upper_relative_gain": 0.2,
                "confidence": 0.95,
                "repetitions": 10_000,
                "independent_units": ["block", "request"],
            }
        ]
    return family


@pytest.mark.parametrize(
    ("node", "expected_count"),
    (
        ("e3b_final", 4),
        ("e5_final", 4),
        ("e6_final", 3),
        ("e0_final", 1),
    ),
)
def test_explicit_final_metric_mapping_has_real_ci_units(
    node: str,
    expected_count: int,
) -> None:
    payload: dict[str, object] = {"family_results": [_family(node)]}
    if node == "e5_final":
        payload["p99_anchor_claims"] = [
            {
                "anchor_id": "anchor-0",
                "status": "CLAIMABLE",
                "point_estimate": 7.0,
                "ci_low": 6.0,
                "ci_high": 8.0,
                "confidence": 0.95,
                "independent_block_count": 12,
                "request_count": 120_000,
                "paired": False,
                "reducer_method": "time_block_bootstrap_linear_native_p99",
                "metric_name": "native_p99_itl_ms",
                "completed_request_count": 120_000,
                "offered_request_count": 120_000,
            }
        ]
    if node == "e0_final":
        payload["breadth_fdr_families"] = []
    rows = _explicit_headline_metric_payload_rows(node, payload)
    assert rows is not None and len(rows) == expected_count
    assert all(
        row["ci_low"] <= row["point_estimate"] <= row["ci_high"]
        and row["block_count"] == 12
        and row["request_count"] >= 1200
        and row["reducer_method"]
        and row["attributes"]["confidence"] == 0.95
        for row in rows
    )
    if node == "e5_final":
        assert any(row["paired"] is False for row in rows)
    else:
        assert all(row["paired"] is True for row in rows)


def test_descriptive_and_unresolved_rows_do_not_receive_fake_ci() -> None:
    assert (
        _explicit_headline_metric_payload_rows(
            "e4_profiler", {"profiler_rows": [{"raw_profile_size_bytes": 4}]}
        )
        is None
    )
    payload = {
        "family_results": [_family("e5_final")],
        "p99_anchor_claims": [
            {
                "anchor_id": "a",
                "status": "UNRESOLVED",
                "reason_codes": ["per_block_minimum_completions_not_met"],
                "independent_block_count": 2,
                "request_count": 20_000,
                "block_evidence": [{"block": 4}, {"block": 5}],
            }
        ],
        "failure_results": [{"status": "FAIL"}],
    }
    rows = _explicit_headline_metric_payload_rows("e5_final", payload)
    assert rows is not None
    assert all(row["metric_name"] != "native_p99_itl_ms" for row in rows)


def test_unresolved_and_excluded_headline_rows_do_not_require_fake_ci() -> None:
    family = _family("e3b_final")
    family["primary_contrasts"] = [
        {
            "name": "lightcone_vs_static",
            "block_ids": [f"block-{index:02d}" for index in range(12)],
            "status": "UNRESOLVED_ZERO_GOODPUT",
            "reason_codes": ["UNRESOLVED_ZERO_GOODPUT"],
            "independent_unit": "paired_block",
        }
    ]
    family["mechanism_contrasts"] = [
        {
            "name": "l0_naive_vs_tts",
            "status": "EXCLUDED_UNSAFE_OR_INACTIVE",
            "reason_codes": ["L0-naive:nonfinite_updates"],
            "excluded_roles": ["L0-naive"],
            "evidence_cell_ids": [_sha("unsafe-l0")],
            "independent_unit": "paired_block",
        }
    ]
    family["hierarchical_intervals"] = [
        {
            "name": "lightcone_vs_static",
            "status": "UNRESOLVED_ZERO_VARIANCE",
            "reason_codes": ["UNRESOLVED_ZERO_VARIANCE"],
            "independent_units": ["block", "request"],
        }
    ]

    rows = _explicit_headline_metric_payload_rows(
        "e3b_final", {"family_results": [family]}
    )

    assert rows is not None and len(rows) == 1
    assert rows[0]["metric_name"] == ("goodput_relative_gain/lightcone_vs_target_only")

    family["primary_contrasts"][0]["status"] = "UNKNOWN_UNRESOLVED_STATUS"  # type: ignore[index]
    with pytest.raises(ValueError, match="contrast status differs"):
        _explicit_headline_metric_payload_rows(
            "e3b_final", {"family_results": [family]}
        )


def test_auxiliary_catalog_and_archive_boundary_are_canonical(tmp_path: Path) -> None:
    inputs = []
    for ordinal in range(2):
        path = (tmp_path / f"launch-{ordinal}.json").resolve()
        path.write_text(f"launch-{ordinal}\n", encoding="utf-8")
        inputs.append(DriverFileBinding.bind(path))
    predecessor = _binding(tmp_path, "completion.json")
    auxiliary = AuxiliaryInputCatalogBinding(
        schema_version=1,
        kind="formal_single_operator_auxiliary_input_catalog_binding",
        node="e6_pilot",
        predecessor_completion=predecessor,
        input_files=tuple(inputs),
        onlinespec_source_authority=None,
    )
    assert AuxiliaryInputCatalogBinding.from_dict(auxiliary.to_dict()) == auxiliary

    run_root = (tmp_path / "formal-run").resolve()
    candidate = run_root / "node" / "work"
    retained_root = candidate / "retained-auxiliary"
    candidate.mkdir(parents=True)
    retained_root.mkdir()
    decision = _binding(run_root, "node/decision.json")
    completion = _binding(run_root, "node/completion.json")
    manifest = RetainedFutureDependencyManifest(
        schema_version=1,
        kind="formal_single_operator_retained_future_dependency_manifest",
        run_id=run_root.name,
        run_root=str(run_root),
        node="preflight",
        completion=completion,
        decision=decision,
        retained_files=(DriverFileBinding.bind(completion.absolute_path),),
        retained_transitive_roots=(str(retained_root),),
        archive_candidate_roots=(str(candidate),),
        archive_safe_after_reduction=True,
        remote_eviction_authorized_for_nonretained_files=True,
        remote_eviction_scope=(
            "archive_candidate_roots_excluding_retained_files_and_transitive_roots"
        ),
        eviction_preconditions=(
            "local_sha_manifest_verified",
            "local_rehydrate_test_passed",
        ),
        transitive_evidence_must_rehydrate_at_original_paths=True,
    )
    path = run_root / "node" / "retained.json"
    _publish_no_replace(path, manifest.to_dict())
    assert load_retained_future_dependency_manifest(path) == manifest
    assert load_retained_future_dependency_manifest(path).sha256 == manifest.sha256


def test_driver_cli_collects_auxiliary_worker_and_binding_operations() -> None:
    worker = _parser().parse_args(
        ["auxiliary-worker", "--descriptor", "/tmp/descriptor.json"]
    )
    assert worker.operation == "auxiliary-worker"
    binding = _parser().parse_args(
        [
            "bind-auxiliary-inputs",
            "--node",
            "e6_pilot",
            "--predecessor-completion",
            "/tmp/completion.json",
            "--input",
            "/tmp/launch-a.json",
            "--input",
            "/tmp/launch-b.json",
            "--output",
            "/tmp/binding.json",
        ]
    )
    assert binding.input == ["/tmp/launch-a.json", "/tmp/launch-b.json"]
