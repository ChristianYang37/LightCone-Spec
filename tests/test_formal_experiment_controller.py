from __future__ import annotations

from pathlib import Path

from lightcone_spec.orchestration.experiment_operator import (
    CellAttemptSpec,
    ControllerArtifactBinding,
    ExperimentOperatorStore,
    PhysicalAttemptGroupMemberSpec,
    QueuedCommandSpec,
    StagePlanEntry,
)
from lightcone_spec.orchestration.formal_experiment_controller import (
    DagCellLaunch,
    DagControllerCallbacks,
    DagExecutionPlan,
    DagMaterialization,
    DagPhysicalAttemptGroup,
    DagReduction,
    FormalExperimentDagBlocked,
    FormalExperimentDagController,
)


def _sha(character: str) -> str:
    return character * 64


def _binding(tmp_path: Path, name: str) -> ControllerArtifactBinding:
    path = (tmp_path / name).resolve()
    path.write_text(name + "\n", encoding="utf-8")
    return ControllerArtifactBinding.bind(path)


def _launch(tmp_path: Path) -> DagCellLaunch:
    cell_id = "preflight:compile"
    prefix = (tmp_path / "cell").resolve()
    command = QueuedCommandSpec(
        cell_id=cell_id,
        attempt=1,
        argv=("python3", "-m", "fixture.worker"),
        launch_compatibility_key="preflight:tp2",
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
    )
    attempt = CellAttemptSpec(
        cell_id=cell_id,
        attempt=1,
        stage="preflight",
        phase="final",
        block="preflight",
        seed=1,
        scientific_axes={"task": "compile", "topology": "tp2_dp1"},
        identity={
            "source_sha256": _sha("a"),
            "patch_sha256": _sha("b"),
            "registry_sha256": _sha("c"),
        },
        command_sha256=command.command_sha256,
        output_directory=str((tmp_path / "output").resolve()),
    )
    return DagCellLaunch(attempt=attempt, command=command)


def _physical_group(tmp_path: Path) -> DagPhysicalAttemptGroup:
    kinds = ("compile", "exactness", *("interference" for _ in range(8)))
    members = []
    for index, kind in enumerate(kinds):
        cell_id = f"preflight:{index:02d}-{kind}"
        prefix = (tmp_path / f"cell-{index:02d}").resolve()
        command = QueuedCommandSpec(
            cell_id=cell_id,
            attempt=1,
            argv=("python3", "-m", "fixture.exact_ten_parent"),
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
        )
        attempt = CellAttemptSpec(
            cell_id=cell_id,
            attempt=1,
            stage="preflight",
            phase="final",
            block="preflight",
            seed=1,
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
    return DagPhysicalAttemptGroup(
        group_id="preflight-exact-ten-parent",
        members=tuple(members),
        leader_cell_id=members[0].attempt.cell_id,
    )


def test_controller_advances_exact_node_and_waits_for_durable_completion(
    tmp_path: Path,
) -> None:
    materialization = _binding(tmp_path, "materialization.json")
    node_materialization = _binding(tmp_path, "node-materialization.json")
    execution_source = _binding(tmp_path, "execution-source.json")
    decision = _binding(tmp_path, "decision.json")
    completion = _binding(tmp_path, "completion.json")
    actual = _binding(tmp_path, "actual.json")
    launch = _launch(tmp_path)
    calls: list[str] = []

    callbacks = DagControllerCallbacks(
        materialize=lambda node, predecessor: (
            calls.append(f"materialize:{node}:{predecessor is None}")
            or DagMaterialization(
                materialization=materialization,
                node_materialization=node_materialization,
                expected_cell_ids=(launch.attempt.cell_id,),
            )
        ),
        plan=lambda node, binding: (
            calls.append(f"plan:{node}:{binding.sha256}")
            or DagExecutionPlan(execution_source, None, (launch,))
        ),
        actual_results=lambda node, attempts: (
            calls.append(f"actuals:{node}:{len(attempts)}")
            or {launch.attempt.cell_id: actual.absolute_path}
        ),
        reduce=lambda node, binding, actuals: (
            calls.append(f"reduce:{node}:{len(actuals)}:{binding.sha256}")
            or DagReduction(decision, completion)
        ),
    )

    database = tmp_path / "operator.sqlite3"
    with ExperimentOperatorStore(database, run_id="controller") as store:
        store.initialize_stage_plan(
            (StagePlanEntry("preflight", 0, "preflight", "final", "1", 1),)
        )
        controller = FormalExperimentDagController(store=store, callbacks=callbacks)
        assert controller.run_once().action == "MATERIALIZED"

        # A crash between attempt registration and node-plan publication is
        # recoverable: the next pass accepts only the byte-identical attempt.
        store.materialize_attempt(launch.attempt)
        assert controller.run_once().action == "PLANNED"
        assert controller.run_once().action == "WAITING"
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
            evidence_files={"actual.json": actual.sha256},
            included_in_analysis=True,
            exclusion_reason=None,
            finished_at_ns=20,
        )
        assert controller.run_once().action == "REDUCED"
        assert controller.run_once().action == "COMPLETE"
        assert store.controller_node("preflight")["state"] == "REDUCED"

    with ExperimentOperatorStore(database) as reopened:
        assert reopened.controller_node("preflight")["completion_sha256"] == (
            completion.sha256
        )
        assert reopened.command_for_attempt(launch.attempt.cell_id, 1) == launch.command
    assert calls[0] == "materialize:preflight:True"
    assert sum(value.startswith("plan:") for value in calls) == 1
    assert sum(value.startswith("reduce:") for value in calls) == 1


def test_controller_registers_exact_ten_as_one_durable_physical_parent(
    tmp_path: Path,
) -> None:
    materialization = _binding(tmp_path, "group-materialization.json")
    node_materialization = _binding(tmp_path, "group-node-materialization.json")
    execution_source = _binding(tmp_path, "group-execution-source.json")
    group = _physical_group(tmp_path)
    cell_ids = tuple(member.attempt.cell_id for member in group.members)
    callbacks = DagControllerCallbacks(
        materialize=lambda _node, _predecessor: DagMaterialization(
            materialization=materialization,
            node_materialization=node_materialization,
            expected_cell_ids=cell_ids,
        ),
        plan=lambda _node, _binding: DagExecutionPlan(
            execution_source,
            None,
            (),
            (group,),
        ),
        actual_results=lambda _node, _attempts: {},
        reduce=lambda _node, _binding, _actuals: (_ for _ in ()).throw(
            AssertionError()
        ),
    )
    database = tmp_path / "group-operator.sqlite3"
    with ExperimentOperatorStore(database, run_id="group-controller") as store:
        store.initialize_stage_plan(
            (StagePlanEntry("preflight", 0, "preflight", "final", "10", 10),)
        )
        controller = FormalExperimentDagController(store=store, callbacks=callbacks)
        assert controller.run_once().action == "MATERIALIZED"
        assert controller.run_once().action == "PLANNED"
        attempts = store.latest_stage_attempts("preflight")
        assert tuple(row["cell_id"] for row in attempts) == cell_ids
        assert {row["status"] for row in attempts} == {"PENDING"}
        assert len(store.queued_commands()) == 10
        assert store.physical_commands() == (group.members[0].command,)
        assert store.physical_attempt_groups() == (
            {
                "group_id": group.group_id,
                "leader_cell_id": group.leader_cell_id,
                "leader_attempt": 1,
                "status": "PENDING",
                "shared_evidence_sha256": None,
                "members": tuple(
                    {
                        "cell_id": member.attempt.cell_id,
                        "attempt": 1,
                        "logical_kind": member.logical_kind,
                        "member_ordinal": ordinal,
                    }
                    for ordinal, member in enumerate(group.members)
                ),
            },
        )
        assert controller.run_once().action == "WAITING"
    with ExperimentOperatorStore(database) as reopened:
        assert len(reopened.latest_stage_attempts("preflight")) == 10
        assert reopened.physical_commands() == (group.members[0].command,)
        assert reopened.physical_attempt_groups()[0]["group_id"] == group.group_id


def test_controller_persists_source_owned_blocker_without_future_cells(
    tmp_path: Path,
) -> None:
    with ExperimentOperatorStore(
        tmp_path / "blocked.sqlite3", run_id="blocked"
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("preflight", 0, "preflight", "final", "10", 10),)
        )
        callbacks = DagControllerCallbacks(
            materialize=lambda _node, _predecessor: (_ for _ in ()).throw(
                FormalExperimentDagBlocked("trusted runtime authority is absent")
            ),
            plan=lambda _node, _binding: (_ for _ in ()).throw(AssertionError()),
            actual_results=lambda _node, _attempts: {},
            reduce=lambda _node, _binding, _actuals: (_ for _ in ()).throw(
                AssertionError()
            ),
        )
        controller = FormalExperimentDagController(store=store, callbacks=callbacks)
        result = controller.run_until_wait()
        assert result.action == "BLOCKED"
        assert store.controller_node("preflight")["state"] == "BLOCKED"
        assert store.latest_stage_attempts("preflight") == ()
