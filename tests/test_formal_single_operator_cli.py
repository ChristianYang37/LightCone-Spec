from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_formal_single_operator_stages import _protocol_lock

from lightcone_spec.cli.main import _parser, main
from lightcone_spec.experiments.formal_registry import protocol_lock_to_dict
from lightcone_spec.experiments.formal_single_operator_gpu_hours import (
    load_formal_single_operator_gpu_hours,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FORMAL_SINGLE_OPERATOR_NODE_ORDER,
    load_formal_single_operator_execution_source,
    publish_formal_single_operator_json_artifact,
)


@pytest.mark.parametrize(
    ("node", "kind"),
    (("e6_pilot", "e6_interface_fit"), ("e0_tuning", "e0_compatibility")),
)
def test_materialize_node_cli_passes_exact_scientific_auxiliary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    node: str,
    kind: str,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_stages as stages

    observed: dict[str, object] = {}

    def materialize(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(artifact=SimpleNamespace(sha256="a" * 64))

    monkeypatch.setattr(stages, "materialize_formal_single_operator_node", materialize)
    auxiliary = (tmp_path / f"{kind}.json").resolve()
    assert (
        main(
            [
                "formal-single-operator",
                "materialize-node",
                "--node",
                node,
                "--predecessor-completion",
                str((tmp_path / "predecessor.json").resolve()),
                "--auxiliary-source",
                f"{kind}={auxiliary}",
                "--materialization-output",
                str((tmp_path / "materialization.json").resolve()),
                "--node-materialization-output",
                str((tmp_path / "node.json").resolve()),
                "--created-ns",
                "10",
            ]
        )
        == 0
    )
    assert observed["auxiliary_source_paths"] == {kind: str(auxiliary)}
    assert capsys.readouterr().out.strip() == "a" * 64


@pytest.mark.parametrize(
    "rows",
    (
        ("foreign_kind=/tmp/source.json",),
        ("e6_interface_fit=relative.json",),
        (
            "e6_interface_fit=/tmp/one.json",
            "e6_interface_fit=/tmp/two.json",
        ),
    ),
)
def test_materialize_node_cli_rejects_invalid_auxiliary_rows(
    tmp_path: Path,
    rows: tuple[str, ...],
) -> None:
    argv = [
        "formal-single-operator",
        "materialize-node",
        "--node",
        "e6_pilot",
        "--predecessor-completion",
        str((tmp_path / "predecessor.json").resolve()),
        "--materialization-output",
        str((tmp_path / "materialization.json").resolve()),
        "--node-materialization-output",
        str((tmp_path / "node.json").resolve()),
        "--created-ns",
        "10",
    ]
    for row in rows:
        argv.extend(("--auxiliary-source", row))
    with pytest.raises(ValueError, match="auxiliary"):
        main(argv)


def test_publish_trusted_content_cli_accepts_only_spec_and_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_content as content

    spec = (tmp_path / "content-spec.json").resolve()
    output = (tmp_path / "content.json").resolve()
    observed: dict[str, object] = {}

    def publish(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            absolute_path=str(output),
            raw_sha256="a" * 64,
            runtime_binding_status="BOUND",
            semantic_sha256="b" * 64,
        )

    monkeypatch.setattr(
        content,
        "publish_runtime_bound_trusted_single_operator_content_from_spec",
        publish,
    )
    assert (
        main(
            [
                "formal-single-operator",
                "publish-trusted-content",
                "--spec",
                str(spec),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert observed == {"spec_path": str(spec), "output_path": str(output)}
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime_binding_status"] == "BOUND"
    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "formal-single-operator",
                "publish-trusted-content",
                "--spec",
                str(spec),
                "--output",
                str(output),
                "--content-sha256",
                "c" * 64,
            ]
        )


def test_publish_preflight_workload_cli_derives_identity_from_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_content as content

    content_path = (tmp_path / "content.json").resolve()
    output = (tmp_path / "workload.json").resolve()
    observed: dict[str, object] = {}

    def publish(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            absolute_path=str(output),
            raw_sha256="a" * 64,
            semantic_sha256="b" * 64,
        )

    monkeypatch.setattr(
        content,
        "publish_trusted_preflight_workload_authority_from_content",
        publish,
    )
    assert (
        main(
            [
                "formal-single-operator",
                "publish-preflight-workload",
                "--content-source",
                str(content_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert observed == {
        "trusted_content_bundle_path": str(content_path),
        "output_path": str(output),
    }
    assert json.loads(capsys.readouterr().out)["semantic_sha256"] == "b" * 64


def test_bootstrap_cli_is_path_only_and_closes_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from lightcone_spec.orchestration import formal_single_operator_bootstrap as boot

    config = (tmp_path / "bootstrap.json").resolve()
    calls: list[str] = []

    class Supervisor:
        def __init__(self, path: str) -> None:
            assert path == str(config)

        def run_once(self):
            calls.append("once")
            return SimpleNamespace(
                controller_action="WAITING",
                to_dict=lambda: {"controller_action": "WAITING"},
            )

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(boot, "FormalSingleOperatorBootstrapSupervisor", Supervisor)
    assert (
        main(
            [
                "formal-single-operator",
                "bootstrap-once",
                "--config",
                str(config),
            ]
        )
        == 0
    )
    assert calls == ["once", "close"]
    assert json.loads(capsys.readouterr().out) == {"controller_action": "WAITING"}


def test_single_operator_cli_materializes_current_node_and_pre_pilot_hours(
    tmp_path: Path,
    capsys,
) -> None:
    protocol_lock = _protocol_lock()
    lock_path = tmp_path / "protocol-lock.json"
    materialization_path = tmp_path / "preflight-materialization.json"
    node_path = tmp_path / "preflight-node-materialization.json"
    source_path = tmp_path / "preflight-execution-source.json"
    gpu_hours_path = tmp_path / "preflight-gpu-hours.json"
    publish_formal_single_operator_json_artifact(
        lock_path,
        protocol_lock_to_dict(protocol_lock),
    )

    assert (
        main(
            [
                "formal-single-operator",
                "materialize-node",
                "--node",
                "preflight",
                "--protocol-lock",
                str(lock_path),
                "--materialization-output",
                str(materialization_path),
                "--node-materialization-output",
                str(node_path),
                "--created-ns",
                "10",
            ]
        )
        == 0
    )
    assert len(capsys.readouterr().out.strip()) == 64

    assert (
        main(
            [
                "formal-single-operator",
                "publish-execution-source",
                "--node-materialization",
                str(node_path),
                "--output",
                str(source_path),
            ]
        )
        == 0
    )
    assert load_formal_single_operator_execution_source(source_path).stage == (
        "preflight"
    )
    capsys.readouterr()

    assert (
        main(
            [
                "formal-single-operator",
                "gpu-hours-pre",
                "--materialization",
                str(materialization_path),
                "--output",
                str(gpu_hours_path),
            ]
        )
        == 0
    )
    output = load_formal_single_operator_gpu_hours(gpu_hours_path)
    assert output.duration_status == "duration_unmeasured"
    assert output.fixed_cell_count == 10
    capsys.readouterr()


def test_gpu_hours_post_cli_routes_actual_results_through_single_charge_reducer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from lightcone_spec.cli import formal_single_operator as cli
    from lightcone_spec.experiments import formal_single_operator_gpu_hours as hours
    from lightcone_spec.runtime.proof_artifact import (
        CanonicalJsonProofBinding,
        publish_canonical_json_no_replace,
    )

    def binding(label: str) -> CanonicalJsonProofBinding:
        path = (tmp_path / f"{label}.json").resolve()
        publish_canonical_json_no_replace(path, {"kind": label})
        return CanonicalJsonProofBinding.bind(path)

    physical = hours._UnifiedPhysicalExecution(
        physical_execution_id="a" * 64,
        execution_kind="resident_session",
        source=binding("shared-session-close"),
        gpu_uuids=("GPU-0",),
        phase_edges_ns=(
            ("server_process_started_ns", 10),
            ("process_exited_ns", 30),
            ("process_group_empty_checked_ns", 35),
            ("evidence_flush_completed_ns", 40),
        ),
    )
    cells = tuple(
        hours._UnifiedCellObservation(
            cell_id=character * 64,
            actual_result=binding(f"actual-{index}"),
            actual_result_sha256=("d" if index == 0 else "e") * 64,
            member_lifecycle=binding(f"lifecycle-{index}"),
            physical_execution_id=physical.physical_execution_id,
            topology="tp1_dp1",
            gang_gpu_count=1,
            provider_reserved_gpu_count=2,
            scored_request_count=8,
            projection_process_ns=5,
            projection_core_wall_ns=5,
            projection_evidence_tail_ns=0,
            projection_source="resident_member_trace",
        )
        for index, character in enumerate(("b", "c"))
    )
    cost = hours._actual_cost_from_unified_observations(
        cells,
        (physical, physical),
        inventory_gpu_count=2,
    )
    assert cost.cell_count == 2
    assert cost.compute_gpu_ns == 20

    pilot = SimpleNamespace(stage="E3a")
    inventory = SimpleNamespace(devices=("GPU-0", "GPU-1"))
    actual_paths = (
        str((tmp_path / "resident-a.json").resolve()),
        str((tmp_path / "resident-b.json").resolve()),
    )
    observed: dict[str, object] = {}

    def derive(**kwargs):
        observed.update(kwargs)
        observed["single_charge_cost"] = cost
        return SimpleNamespace(sha256="f" * 64)

    monkeypatch.setattr(cli, "_load_materialization", lambda _path: pilot)
    monkeypatch.setattr(cli, "_load_inventory", lambda _path: inventory)
    monkeypatch.setattr(
        hours,
        "derive_formal_single_operator_post_pilot_gpu_hours_from_serving_actuals",
        derive,
    )
    monkeypatch.setattr(
        hours,
        "publish_formal_single_operator_gpu_hours",
        lambda output, output_path: observed.update(
            {"published": (output, output_path)}
        ),
    )

    assert (
        main(
            [
                "formal-single-operator",
                "gpu-hours-post",
                "--repository-root",
                str(tmp_path),
                "--pilot-materialization",
                str(tmp_path / "pilot.json"),
                "--inventory",
                str(tmp_path / "inventory.json"),
                "--actual-result",
                actual_paths[0],
                "--actual-result",
                actual_paths[1],
                "--source-output",
                str(tmp_path / "source.json"),
                "--output",
                str(tmp_path / "hours.json"),
            ]
        )
        == 0
    )
    assert observed["pilot_actual_result_paths"] == actual_paths
    assert observed["single_charge_cost"] == cost
    assert capsys.readouterr().out.strip() == "f" * 64


def test_single_operator_cli_status_reports_exact_current_readiness(capsys) -> None:
    assert main(["formal-single-operator", "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    rows = status["nodes"]
    assert len(rows) == 21
    assert rows[0]["node"] == "preflight"
    assert rows[-1]["node"] == "e0_final"
    assert status["mode"] == "formal_single_operator_v1"
    assert len(status["protocol_sha256"]) == 64
    assert tuple(row["node"] for row in rows if row["status"] == "READY") == (
        FORMAL_SINGLE_OPERATOR_NODE_ORDER
    )
    assert status["readiness_scope"] == "code_capability_only"
    assert FORMAL_SINGLE_OPERATOR_NODE_ORDER[7] == "e2_r3"
    assert FORMAL_SINGLE_OPERATOR_NODE_ORDER[8:10] == ("e4_screen", "e4_local")
    assert all(row["status"] == "READY" for row in rows)
    assert all(row["blocker"] is None for row in rows)
    assert all(row["physical_mapper_available"] is True for row in rows)
    assert all(row["producer_available"] is True for row in rows)
    assert all(row["executor_available"] is True for row in rows)
    assert all(row["finalizer_available"] is True for row in rows)
    assert all(row["code_materializer_available"] is True for row in rows)
    assert all(row["code_capability_ready"] is True for row in rows)
    assert rows[10]["node"] == "e4_profiler"
    assert rows[10]["physical_mapper_available"] is True
    assert rows[10]["producer_available"] is True
    assert rows[10]["status"] == "READY"
    assert all(row["structural_status"] == "READY" for row in rows)


def test_prepare_run_routes_profiler_through_generated_capture_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_profiler as profiler
    from lightcone_spec.experiments import (
        formal_single_operator_run_dispatch as dispatch,
    )
    from lightcone_spec.runtime import formal_single_operator as runtime

    cell_id = "1" * 64
    source = SimpleNamespace(node="e4_profiler", stage="E4")
    route = SimpleNamespace(physical_kind="profiler")
    monkeypatch.setattr(
        dispatch,
        "route_formal_single_operator_cell",
        lambda **kwargs: (source, SimpleNamespace(cell_id=cell_id), route),
    )
    run_root = tmp_path / "run"

    def create_run_directory(**kwargs):
        run_root.mkdir(mode=0o700)
        return run_root

    monkeypatch.setattr(
        runtime,
        "create_formal_single_operator_run_directory",
        create_run_directory,
    )
    observed: dict[str, object] = {}

    def materialize(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            sha256="2" * 64,
            subject_run_plan=SimpleNamespace(
                absolute_path=str(run_root / "subject.json")
            ),
        )

    monkeypatch.setattr(
        profiler,
        "materialize_formal_single_operator_profiler_plan",
        materialize,
    )
    assert (
        main(
            [
                "formal-single-operator",
                "prepare-run",
                "--repository-root",
                str(tmp_path),
                "--execution-source",
                str(tmp_path / "execution-source.json"),
                "--cell",
                cell_id,
                "--prepared-launch-bundle",
                str(tmp_path / "bundle.json"),
                "--profiler-tool",
                str(tmp_path / "nsys"),
                "--output-root",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["physical_kind"] == "profiler"
    assert output["run_plan"].endswith("formal-single-operator-profiler-plan.json")
    assert observed["prepared_launch_bundle_path"] == str(tmp_path / "bundle.json")
    assert observed["tool_path"] == str(tmp_path / "nsys")


def test_publish_profiler_subject_cli_exposes_only_source_owned_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_profiler_subject_producer as producer,
    )

    observed: dict[str, object] = {}

    def publish(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            sha256="2" * 64,
            selected_configuration_sha256="3" * 64,
            source_headline_cell_id="4" * 64,
        )

    monkeypatch.setattr(
        producer,
        "publish_formal_single_operator_profiler_subject_requirement",
        publish,
    )
    source = (tmp_path / "e4-profiler-execution-source.json").resolve()
    output = (tmp_path / "profiler-subject-requirement.json").resolve()
    assert (
        main(
            [
                "formal-single-operator",
                "publish-profiler-subject",
                "--execution-source",
                str(source),
                "--repository-root",
                str(tmp_path.resolve()),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert observed == {
        "execution_source_path": str(source),
        "repository_root": str(tmp_path.resolve()),
        "output_path": str(output),
    }
    assert result == {
        "path": str(output),
        "requirement_sha256": "2" * 64,
        "selected_configuration_sha256": "3" * 64,
        "source_headline_cell_id": "4" * 64,
    }
    arguments = _parser().parse_args(
        [
            "formal-single-operator",
            "publish-profiler-subject",
            "--execution-source",
            str(source),
            "--repository-root",
            str(tmp_path.resolve()),
            "--output",
            str(output),
        ]
    )
    for forbidden in (
        "load",
        "traffic",
        "run_config",
        "launch",
        "request_schedule",
        "selected_configuration_sha256",
    ):
        assert not hasattr(arguments, forbidden)


def test_prepare_run_routes_e5_failure_through_public_one_shot_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from lightcone_spec.experiments import (
        formal_failure_execution as failure_execution,
    )
    from lightcone_spec.experiments import (
        formal_single_operator_run_dispatch as run_dispatch,
    )
    from lightcone_spec.orchestration import (
        formal_physical_dispatch,
        formal_single_operator_admission,
    )
    from lightcone_spec.runtime import formal_single_operator as runtime

    cell_id = "1" * 64
    source = SimpleNamespace(node="e5_final", stage="E5")
    route = SimpleNamespace(physical_kind="e5_failure")
    monkeypatch.setattr(
        run_dispatch,
        "route_formal_single_operator_cell",
        lambda **kwargs: (source, SimpleNamespace(cell_id=cell_id), route),
    )
    run_root = (tmp_path / "run").resolve()

    def create_run_directory(**kwargs):
        run_root.mkdir(mode=0o700)
        return run_root

    monkeypatch.setattr(
        runtime,
        "create_formal_single_operator_run_directory",
        create_run_directory,
    )
    observed: dict[str, object] = {}

    def materialize_descriptor(**kwargs):
        observed["descriptor"] = kwargs
        return SimpleNamespace(
            inventory=SimpleNamespace(absolute_path=str(tmp_path / "inventory.json")),
            expected_failure_execution_binding_sha256="2" * 64,
        )

    def materialize_plan(**kwargs):
        observed["plan"] = kwargs
        return SimpleNamespace(sha256="3" * 64)

    def publish_admission(**kwargs):
        observed["admission"] = kwargs
        return SimpleNamespace(absolute_path=str(run_root / "admission.json"))

    monkeypatch.setattr(
        failure_execution,
        "materialize_formal_single_operator_e5_failure_execution_descriptor",
        materialize_descriptor,
    )
    monkeypatch.setattr(
        formal_physical_dispatch,
        "materialize_formal_single_operator_e5_failure_run_plan",
        materialize_plan,
    )
    monkeypatch.setattr(
        formal_single_operator_admission,
        "publish_formal_single_operator_admission",
        publish_admission,
    )
    assert (
        main(
            [
                "formal-single-operator",
                "prepare-run",
                "--repository-root",
                str(tmp_path),
                "--execution-source",
                str(tmp_path / "execution-source.json"),
                "--cell",
                cell_id,
                "--prepared-launch-bundle",
                str(tmp_path / "bundle.json"),
                "--output-root",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["physical_kind"] == "e5_failure"
    assert output["retry_allowance"] == 0
    assert output["exclusive_timing"] is True
    assert observed["descriptor"]["prepared_launch_bundle_path"] == str(  # type: ignore[index]
        tmp_path / "bundle.json"
    )
    assert observed["plan"]["failure_execution_descriptor_path"] == (  # type: ignore[index]
        run_root / "formal-single-operator-e5-failure-execution.json"
    )


def test_single_operator_preflight_builder_exposes_only_source_paths_and_time() -> None:
    arguments = _parser().parse_args(
        [
            "formal-single-operator",
            "build-preflight-inputs",
            "--execution-source",
            "/sources/preflight.json",
            "--repository-root",
            "/source/repository",
            "--runtime-authority-manifest",
            "/sources/runtime.json",
            "--inventory",
            "/sources/inventory.json",
            "--content-verification-receipt",
            "/sources/content.json",
            "--workload-authority",
            "/sources/workload.json",
            "--doctor-report",
            "/sources/doctor.json",
            "--private-output-root",
            "/results/exact-ten",
            "--current-ns",
            "10",
        ]
    )

    assert arguments.single_operator_operation == "build-preflight-inputs"
    for forbidden in (
        "argv",
        "port",
        "token_ids",
        "recipe_sha256",
        "duration",
        "gpu_hours",
    ):
        assert not hasattr(arguments, forbidden)


def test_single_operator_preflight_completion_accepts_only_actual_source_paths() -> (
    None
):
    arguments = _parser().parse_args(
        [
            "formal-single-operator",
            "publish-preflight-completion",
            "--execution-inputs",
            "/results/exact-ten/inputs.json",
            "--compile-result",
            "/results/exact-ten/compile.json",
            "--exactness-result",
            "/results/exact-ten/exactness.json",
            "--interference-terminal",
            f"{'1' * 64}=/results/exact-ten/terminal.json",
            "--interference-lifecycle",
            f"{'1' * 64}=/results/exact-ten/lifecycle.json",
            "--interference-junit",
            f"{'1' * 64}=/results/exact-ten/junit.xml",
            "--output",
            "/results/exact-ten/completion.json",
            "--current-ns",
            "20",
        ]
    )

    assert arguments.single_operator_operation == "publish-preflight-completion"
    for forbidden in ("status", "result_sha256", "duration", "gpu_hours"):
        assert not hasattr(arguments, forbidden)


def test_single_operator_preflight_executor_has_no_runtime_knobs() -> None:
    arguments = _parser().parse_args(
        [
            "formal-single-operator",
            "execute-preflight",
            "--execution-inputs",
            "/results/exact-ten/inputs.json",
            "--current-ns",
            "20",
        ]
    )

    assert arguments.single_operator_operation == "execute-preflight"
    for forbidden in (
        "argv",
        "port",
        "token_ids",
        "timeout",
        "status",
        "result_sha256",
    ):
        assert not hasattr(arguments, forbidden)


def test_single_operator_preflight_reducer_consumes_only_exact_execution() -> None:
    arguments = _parser().parse_args(
        [
            "formal-single-operator",
            "reduce-preflight",
            "--node-materialization",
            "/results/preflight-node.json",
            "--preflight-execution",
            "/results/exact-ten/formal-single-operator-preflight-execution.json",
            "--repository-root",
            "/source/repository",
            "--decision-output",
            "/results/preflight-decision.json",
            "--completion-output",
            "/results/preflight-completion.json",
            "--completed-ns",
            "30",
        ]
    )

    assert arguments.single_operator_operation == "reduce-preflight"
    assert not hasattr(arguments, "status")
    assert not hasattr(arguments, "actual")


def test_single_operator_schema2_finalizer_rebuilds_sources_from_plan() -> None:
    arguments = _parser().parse_args(
        [
            "formal-single-operator",
            "finalize-run",
            "--repository-root",
            "/source/repository",
            "--run-plan",
            "/results/e3a/formal-serving-run-plan.json",
        ]
    )

    assert arguments.single_operator_operation == "finalize-run"
    assert arguments.execution_source is None
    assert arguments.inventory is None


def test_single_operator_run_preparer_accepts_only_current_inputs() -> None:
    arguments = _parser().parse_args(
        [
            "formal-single-operator",
            "prepare-run",
            "--repository-root",
            "/source/repository",
            "--execution-source",
            "/sources/e3a.json",
            "--cell",
            "cell-1",
            "--preflight-inputs",
            "/results/preflight/formal-preflight-execution-inputs.json",
            "--output-root",
            "/results/runs",
        ]
    )

    assert arguments.single_operator_operation == "prepare-run"
    for forbidden in (
        "argv",
        "port",
        "token_ids",
        "recipe_sha256",
        "content_verification_receipt",
        "workload_authority",
        "inventory",
        "runtime_gpu_proof",
    ):
        assert not hasattr(arguments, forbidden)


def test_single_operator_run_executor_accepts_only_plan_and_clean_checkout() -> None:
    arguments = _parser().parse_args(
        [
            "formal-single-operator",
            "execute-run",
            "--repository-root",
            "/source/repository",
            "--run-plan",
            "/results/e3a/formal-serving-run-plan.json",
        ]
    )

    assert arguments.single_operator_operation == "execute-run"
    for forbidden in (
        "argv",
        "port",
        "token_ids",
        "recipe_sha256",
        "execution_source",
        "inventory",
        "nvidia_smi",
        "runtime_gpu_proof",
    ):
        assert not hasattr(arguments, forbidden)
