from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import formal_single_operator_downstream as downstream
from lightcone_spec.experiments import formal_single_operator_stages as stages
from lightcone_spec.experiments.formal_single_operator_stages import (
    FORMAL_SINGLE_OPERATOR_NODE_SPECS,
)
from lightcone_spec.experiments.stage_materialization import MaterializedCell, _cell
from lightcone_spec.orchestration.experiment_operator import (
    CellAttemptSpec,
    ControllerArtifactBinding,
    ExperimentOperatorStore,
    StagePlanEntry,
)
from lightcone_spec.orchestration.formal_experiment_controller import (
    FormalExperimentDagBlocked,
)
from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
    FreshPreflightInterferenceGateResolver,
    ProductionFormalDagCallbackBuilder,
    _explicit_headline_metric_payload_rows,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _serving_observation(*, observable_itl: bool = True) -> dict[str, object]:
    completed_tokens = [100, 200] if observable_itl else [100]
    completed_output = [11, 12] if observable_itl else [11]
    return {
        "source_request_pool_sha256": _sha(499),
        "requests": [
            {
                "request_id": "request-000",
                "input_token_ids": [1, 2],
                "output_token_ids": completed_output,
                "request_started_ns": 1,
                "request_terminal_ns": 500,
                "token_observed_ns": completed_tokens,
                "terminal_status": "completed",
                "terminal_reason": "complete",
                "submitted_to_server": True,
                "native_terminal_status": "completed",
                "offered_at_us": 0,
                "admitted_at_us": 0,
                "effective_deadline_us": 1_000,
                "terminal_at_us": 1,
            },
            {
                "request_id": "request-001",
                "input_token_ids": [3],
                "output_token_ids": [],
                "request_started_ns": 10,
                "request_terminal_ns": 20,
                "token_observed_ns": [],
                "terminal_status": "rejected",
                "terminal_reason": "queue_full",
                "submitted_to_server": False,
                "native_terminal_status": None,
                "offered_at_us": 1,
                "admitted_at_us": None,
                "effective_deadline_us": 1_001,
                "terminal_at_us": 2,
            },
        ],
        "performance_counters": {
            "peak_hbm_bytes": 96,
            "target_calls": 7,
            "committed_tokens": 2,
            "accepted_drafts": None,
            "verified_drafts": None,
            "updates_launched": None,
            "updates_published": None,
        },
    }


def test_serving_projection_preserves_denominator_and_never_fabricates_ci() -> None:
    rows = ProductionFormalDagCallbackBuilder._serving_descriptive_rows(
        _serving_observation()
    )
    by_name = {str(row["metric_name"]): row for row in rows}
    assert {
        "slo_goodput_tokens_per_second",
        "native_p99_itl_ms",
        "native_runtime_counter/peak_hbm_bytes",
        "native_runtime_counter/target_calls",
        "native_runtime_counter/committed_tokens",
    } <= set(by_name)
    for row in rows:
        assert "ci_low" not in row and "ci_high" not in row
        assert row["request_count"] == 2
        assert row["independent_block_count"] == 1
        assert row["paired"] is False
        attributes = row["attributes"]
        assert isinstance(attributes, dict)
        assert attributes["offered_request_count"] == 2
        assert attributes["completed_request_count"] == 1
        assert attributes["rejected_request_count"] == 1
        assert attributes["timed_out_request_count"] == 0
        assert attributes["cancelled_request_count"] == 0
        assert attributes["unfinished_request_count"] == 0
        assert attributes["aborted_request_count"] == 0
    assert by_name["native_p99_itl_ms"]["point_estimate"] == pytest.approx(0.0001)
    assert (
        by_name["slo_goodput_tokens_per_second"]["reducer_method"]
        == "formal_slo_goodput_v2"
    )


def test_unobservable_native_p99_is_omitted_instead_of_zero_or_fake_ci() -> None:
    rows = ProductionFormalDagCallbackBuilder._serving_descriptive_rows(
        _serving_observation(observable_itl=False)
    )
    names = {row["metric_name"] for row in rows}
    assert "slo_goodput_tokens_per_second" in names
    assert "native_p99_itl_ms" not in names


def test_validated_actual_to_slo_and_metrics_preserves_offered_five_state_partition() -> (
    None
):
    statuses = (
        "completed",
        "rejected",
        "timed_out",
        "cancelled",
        "unfinished",
    )
    requests = []
    for index, status in enumerate(statuses):
        completed = status == "completed"
        submitted = status in {"completed", "timed_out", "cancelled"}
        requests.append(
            {
                "request_id": f"request-{index:03d}",
                "input_token_ids": [index + 1],
                "output_token_ids": [11, 12] if completed else [],
                "request_started_ns": 1 + index * 10,
                "request_terminal_ns": 500 + index * 10,
                "token_observed_ns": [100, 200] if completed else [],
                "terminal_status": status,
                "terminal_reason": status,
                "submitted_to_server": submitted,
                "native_terminal_status": (
                    "completed" if completed else "aborted" if submitted else None
                ),
                "offered_at_us": index,
                "admitted_at_us": index if submitted else None,
                "effective_deadline_us": 1_000 + index,
                "terminal_at_us": None if status == "unfinished" else 10 + index,
            }
        )
    observation = {
        "schema_version": 2,
        "kind": "formal_single_operator_serving_observation",
        "materialized_cell_id": "",
        "inventory_sha256": _sha(900),
        "run_id": "run-five-state",
        "run_nonce_sha256": _sha(901),
        "attempt_id": "attempt-five-state",
        "method": "l0",
        "terminal_sha256": _sha(902),
        "terminal_artifact_sha256": _sha(903),
        "native_itl_artifact_sha256": _sha(904),
        "request_schedule_sha256": _sha(905),
        "source_request_pool_sha256": _sha(911),
        "serving_execution_policy_sha256": _sha(906),
        "client_lifecycle_artifact_sha256": _sha(907),
        "scored_phase_origin_ns": 1,
        # One registered closed-loop maximum-pool row was never offered and is
        # intentionally absent from every request-level denominator below.
        "scheduled_request_count": 6,
        "offered_request_count": 5,
        "outcome_counts": {status: 1 for status in statuses},
        "requests": requests,
        "performance_counters": {
            "peak_hbm_bytes": 96,
            "target_calls": 7,
            "committed_tokens": 2,
        },
    }
    cell = _cell(
        stage="E5",
        method_role="LightCone",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="production_slo_power_prefix",
        publication_policy="first_ready",
        recipe_sha256=_sha(908),
        dimensions={"block": 0, "load": "common"},
    )
    observation["materialized_cell_id"] = cell.cell_id
    actual = SimpleNamespace(
        reducer_payload={
            "artifacts": [],
            "exit_code": 0,
            "manifest_sha256": _sha(909),
            "run_directory": "/tmp/five-state",
            "serving_observation": observation,
        }
    )

    reopened = stages._serving_observation(actual, cell)
    evidence = downstream._request_evidence(reopened)
    assert [row.completed for row in evidence] == [True, False, False, False, False]
    assert [row.error for row in evidence] == [False, False, False, False, True]
    slo = downstream._slo(reopened)
    assert slo.eligible_requests == 5
    assert slo.completed_requests == 1
    assert slo.error_requests == 1
    assert slo.status == "FAIL"

    metrics = ProductionFormalDagCallbackBuilder._serving_descriptive_rows(reopened)
    for metric in metrics:
        assert metric["request_count"] == 5
        attributes = metric["attributes"]
        assert attributes["offered_request_count"] == 5
        assert attributes["completed_request_count"] == 1
        assert attributes["rejected_request_count"] == 1
        assert attributes["timed_out_request_count"] == 1
        assert attributes["cancelled_request_count"] == 1
        assert attributes["unfinished_request_count"] == 1
        assert attributes["aborted_request_count"] == 2


@dataclass(frozen=True)
class _Diagnostic:
    status: str = "PASS"
    p99: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": "formal-preflight-static-two-way",
            "simultaneous_jobs": 2,
            "status": self.status,
            "reason_codes": [] if self.status == "PASS" else ["itl_unavailable"],
            "raw_observation_sha256s": [_sha(index) for index in range(20, 28)],
            "goodput_ratios": [
                [repetition, slot, 1.001]
                for repetition in range(2)
                for slot in range(2)
            ],
            "p99_itl_ratios": (
                [
                    [repetition, slot, 1.002]
                    for repetition in range(2)
                    for slot in range(2)
                ]
                if self.p99
                else []
            ),
            "goodput_mean_relative_difference": 0.001,
            "goodput_ci_lower_relative_difference": -0.002,
            "goodput_ci_upper_relative_difference": 0.003,
            "p99_itl_mean_relative_difference": 0.002 if self.p99 else None,
            "p99_itl_ci_lower_relative_difference": -0.003 if self.p99 else None,
            "p99_itl_ci_upper_relative_difference": 0.004 if self.p99 else None,
        }


def _proof_rows() -> tuple[SimpleNamespace, ...]:
    rows = []
    index = 0
    for mode in ("isolated", "concurrent"):
        for repetition in range(2):
            for slot in range(2):
                rows.append(
                    SimpleNamespace(
                        materialized_cell_id=_sha(index + 100),
                        mode=mode,
                        repetition=repetition,
                        slot=slot,
                        sha256=_sha(index + 200),
                        observation=SimpleNamespace(
                            request_ids=("request-a", "request-b"),
                            completed_requests=2,
                        ),
                    )
                )
                index += 1
    return tuple(rows)


def _completion(tmp_path, name: str) -> ControllerArtifactBinding:
    path = (tmp_path / name / "completion.json").resolve()
    path.parent.mkdir(parents=True)
    path.write_text('{"complete":true}\n', encoding="utf-8")
    return ControllerArtifactBinding.bind(path)


def _publish_diagnostic(tmp_path, *, p99: bool = True):
    completion = _completion(tmp_path, "reduction-p99" if p99 else "reduction-no-p99")
    exact_ten = (
        tmp_path / ("exact-p99.json" if p99 else "exact-no-p99.json")
    ).resolve()
    exact_ten.write_text('{"exact_ten":true}\n', encoding="utf-8")
    rows = _proof_rows()
    binding = FreshPreflightInterferenceGateResolver._publish_diagnostic(
        completion=completion,
        exact_ten_completion_path=str(exact_ten),
        gpu_uuids=("GPU-0", "GPU-1"),
        proof_rows=rows,
        diagnostic=_Diagnostic(p99=p99),
    )
    return completion, rows, binding


def test_fresh_interference_diagnostic_is_no_replace_and_envelope_bindable(
    tmp_path,
) -> None:
    completion, rows, binding = _publish_diagnostic(tmp_path)
    rebound = FreshPreflightInterferenceGateResolver._publish_diagnostic(
        completion=completion,
        exact_ten_completion_path=str((tmp_path / "exact-p99.json").resolve()),
        gpu_uuids=("GPU-0", "GPU-1"),
        proof_rows=rows,
        diagnostic=_Diagnostic(),
    )
    assert rebound == binding
    assert (
        FreshPreflightInterferenceGateResolver.diagnostic_binding(completion) == binding
    )
    with pytest.raises(FormalExperimentDagBlocked, match="diagnostic changed"):
        FreshPreflightInterferenceGateResolver._publish_diagnostic(
            completion=completion,
            exact_ten_completion_path=str((tmp_path / "exact-p99.json").resolve()),
            gpu_uuids=("GPU-0", "GPU-1"),
            proof_rows=rows,
            diagnostic=_Diagnostic(status="FAIL"),
        )


class _CompleteStore:
    @staticmethod
    def latest_attempt(_cell_id: str) -> dict[str, object]:
        return {"status": "COMPLETE", "attempt": 1}


def _builder() -> ProductionFormalDagCallbackBuilder:
    builder = object.__new__(ProductionFormalDagCallbackBuilder)
    builder.store = _CompleteStore()
    return builder


def test_interference_metrics_keep_real_bca_and_omit_unresolved_p99(tmp_path) -> None:
    for p99, expected in ((True, 2), (False, 1)):
        _completion_binding, proof_rows, binding = _publish_diagnostic(
            tmp_path,
            p99=p99,
        )
        anchor = min(row.materialized_cell_id for row in proof_rows)
        rebuilt = SimpleNamespace(
            decision=SimpleNamespace(stage="preflight", phase="interference"),
            materialization=SimpleNamespace(cells=(SimpleNamespace(cell_id=anchor),)),
        )
        metrics = _builder()._interference_metrics(
            rebuilt=rebuilt,
            diagnostic_binding=binding,
        )
        assert len(metrics) == expected
        assert metrics[0].metric_kind == "headline"
        assert metrics[0].independent_block_count == 2
        assert metrics[0].request_count == 16
        assert metrics[0].paired is True
        assert metrics[0].ci_low is not None and metrics[0].ci_high is not None
        assert metrics[0].attributes["confidence_level"] == 0.95
        if not p99:
            assert all("p99_itl" not in row.metric_name for row in metrics)


@pytest.mark.parametrize(
    ("stage", "task", "payload", "metric_name"),
    (
        (
            "E4",
            "mechanism_profile_only",
            {
                "raw_profile_size_bytes": 64,
                "profiler_variant": "nsys",
                "raw_profile_sha256": _sha(300),
            },
            "profiler_raw_profile_size_bytes",
        ),
        (
            "E5",
            "deterministic_failure_injection",
            {
                "diagnostic_status": "PASS",
                "failure": "queue_saturation",
                "topology": "tp1_dp1",
                "cohort_count": 1,
                "process_exit_code": 0,
            },
            "failure_diagnostic_pass_indicator",
        ),
        (
            "E6",
            "immutable_metadata_interface_and_fit_preflight",
            {
                "interface_sha256": _sha(301),
                "verified_authority_sha256": _sha(302),
            },
            "e6_interface_fit_pass_indicator",
        ),
        (
            "E0",
            "compatibility_decision",
            {
                "disposition": "N/A",
                "reason_code": "unsupported_interface",
                "decision_id": _sha(303),
                "interface_sha256": _sha(304),
            },
            "e0_compatibility_valid_indicator",
        ),
    ),
)
def test_nonserving_actuals_have_meaningful_descriptive_metrics_without_ci(
    stage: str,
    task: str,
    payload: dict[str, object],
    metric_name: str,
) -> None:
    cell = MaterializedCell(
        stage=stage,
        method_role="Compatibility" if stage == "E0" else "LightCone",
        model="Qwen/test-model",
        backend="DFLASH",
        task=task,
        publication_policy="first_ready",
        recipe_sha256=None,
        dimensions=(),
    )
    actual = SimpleNamespace(
        cell_id=cell.cell_id,
        started_ns=10,
        finished_ns=20,
        result_identity_sha256=_sha(400),
        validator_kind=f"{task}_validator",
        validator_protocol_sha256=_sha(401),
        reducer_payload=payload,
    )
    rebuilt = SimpleNamespace(
        decision=SimpleNamespace(stage=stage, phase=f"{task}_phase"),
        materialization=SimpleNamespace(cells=(cell,)),
        artifact=SimpleNamespace(actual_results=(actual,)),
    )
    metrics = _builder()._actual_metrics(rebuilt)
    names = {row.metric_name for row in metrics}
    assert names == {"validated_actual_wall_seconds", metric_name}
    assert all(
        row.metric_kind == "descriptive" and row.ci_low is None and row.ci_high is None
        for row in metrics
    )


def test_validated_serving_actual_projects_slo_p99_hbm_and_wall_time() -> None:
    cell = MaterializedCell(
        stage="E3a",
        method_role="Static",
        model="Qwen/test-model",
        backend="DFLASH",
        task="controlled_capacity",
        publication_policy="none",
        recipe_sha256=None,
        dimensions=(("block", 0),),
    )
    observation = {
        "schema_version": 2,
        "kind": "formal_single_operator_serving_observation",
        "materialized_cell_id": cell.cell_id,
        "inventory_sha256": _sha(500),
        "run_id": "run-test",
        "run_nonce_sha256": _sha(501),
        "attempt_id": "attempt-1",
        "method": "static",
        "terminal_sha256": _sha(502),
        "terminal_artifact_sha256": _sha(503),
        "native_itl_artifact_sha256": _sha(504),
        "request_schedule_sha256": _sha(505),
        "source_request_pool_sha256": _sha(511),
        "serving_execution_policy_sha256": _sha(509),
        "client_lifecycle_artifact_sha256": _sha(510),
        "scored_phase_origin_ns": 1,
        "scheduled_request_count": 2,
        "offered_request_count": 2,
        "outcome_counts": {
            "completed": 1,
            "rejected": 1,
            "timed_out": 0,
            "cancelled": 0,
            "unfinished": 0,
        },
        **_serving_observation(),
    }
    actual = SimpleNamespace(
        cell_id=cell.cell_id,
        started_ns=10,
        finished_ns=20,
        result_identity_sha256=_sha(506),
        validator_kind="formal_single_operator_run_manifest_revalidator",
        validator_protocol_sha256=_sha(507),
        reducer_payload={
            "artifacts": [],
            "exit_code": 0,
            "manifest_sha256": _sha(508),
            "run_directory": "/formal/run",
            "serving_observation": observation,
        },
    )
    rebuilt = SimpleNamespace(
        decision=SimpleNamespace(stage="E3a", phase="capacity"),
        materialization=SimpleNamespace(cells=(cell,)),
        artifact=SimpleNamespace(actual_results=(actual,)),
    )
    metrics = _builder()._actual_metrics(rebuilt)
    names = {row.metric_name for row in metrics}
    assert {
        "validated_actual_wall_seconds",
        "slo_goodput_tokens_per_second",
        "native_p99_itl_ms",
        "native_runtime_counter/peak_hbm_bytes",
    } <= names
    assert all(row.metric_kind == "descriptive" for row in metrics)
    assert all(row.ci_low is None and row.ci_high is None for row in metrics)


def test_metric_projection_recording_is_restart_idempotent(tmp_path) -> None:
    cell = MaterializedCell(
        stage="E4",
        method_role="LightCone",
        model="Qwen/test-model",
        backend="DFLASH",
        task="mechanism_profile_only",
        publication_policy="first_ready",
        recipe_sha256=None,
        dimensions=(("profiler", "nsys"),),
    )
    with ExperimentOperatorStore(
        tmp_path / "operator.sqlite3",
        run_id="metric-idempotence",
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e4_profiler", 0, "E4", "profiler", "1", 1),)
        )
        store.materialize_attempt(
            CellAttemptSpec(
                cell_id=cell.cell_id,
                attempt=1,
                stage="E4",
                phase="profiler",
                block=None,
                seed=None,
                scientific_axes={"task": cell.task},
                identity={
                    "source_sha256": _sha(600),
                    "patch_sha256": _sha(601),
                    "registry_sha256": _sha(602),
                },
                command_sha256=_sha(603),
                output_directory=str((tmp_path / "attempt").resolve()),
            )
        )
        store.mark_running_before_spawn(
            cell.cell_id,
            1,
            assigned_gpu_uuids=("GPU-0",),
            started_at_ns=10,
        )
        store.attach_process(cell.cell_id, 1, pid=100, pgid=100)
        store.finish_attempt(
            cell.cell_id,
            1,
            status="COMPLETE",
            exit_code=0,
            terminal_sha256=_sha(604),
            junit_sha256=_sha(605),
            raw_log_sha256=_sha(606),
            evidence_files={str((tmp_path / "raw-profile").resolve()): _sha(607)},
            included_in_analysis=False,
            exclusion_reason="profiler_only",
            finished_at_ns=20,
        )
        builder = object.__new__(ProductionFormalDagCallbackBuilder)
        builder.store = store
        actual = SimpleNamespace(
            cell_id=cell.cell_id,
            started_ns=10,
            finished_ns=20,
            result_identity_sha256=_sha(608),
            validator_kind="profiler_terminal",
            validator_protocol_sha256=_sha(609),
            reducer_payload={
                "raw_profile_size_bytes": 64,
                "profiler_variant": "nsys",
                "raw_profile_sha256": _sha(610),
            },
        )
        rebuilt = SimpleNamespace(
            decision=SimpleNamespace(stage="E4", phase="profiler"),
            materialization=SimpleNamespace(cells=(cell,)),
            artifact=SimpleNamespace(actual_results=(actual,)),
        )
        metrics = builder._actual_metrics(rebuilt)
        for metric in metrics:
            builder._record_metric_once(metric, recorded_at_ns=30)
        for metric in metrics:
            builder._record_metric_once(metric, recorded_at_ns=30)
        assert len(store._metric_rows()) == 2


@pytest.mark.parametrize("node", FORMAL_SINGLE_OPERATOR_NODE_SPECS)
def test_each_nonempty_dag_node_has_a_nonfabricated_actual_metric(node) -> None:
    cell = MaterializedCell(
        stage=node.stage,
        method_role="Static",
        model="Qwen/test-model",
        backend="DFLASH",
        task="validated_actual_fixture",
        publication_policy="none",
        recipe_sha256=None,
        dimensions=(("node", node.node),),
    )
    actual = SimpleNamespace(
        cell_id=cell.cell_id,
        started_ns=100,
        finished_ns=200,
        result_identity_sha256=_sha(700 + node.ordinal),
        validator_kind="fixture_validator",
        validator_protocol_sha256=_sha(800 + node.ordinal),
        reducer_payload={},
    )
    rebuilt = SimpleNamespace(
        decision=SimpleNamespace(stage=node.stage, phase=node.phase),
        materialization=SimpleNamespace(cells=(cell,)),
        artifact=SimpleNamespace(actual_results=(actual,)),
    )
    metrics = _builder()._actual_metrics(rebuilt)
    assert len(metrics) == 1
    assert metrics[0].metric_name == "validated_actual_wall_seconds"
    assert metrics[0].stage == node.stage and metrics[0].phase == node.phase
    assert metrics[0].metric_kind == "descriptive"
    assert metrics[0].ci_low is None and metrics[0].ci_high is None


def test_reducer_headline_projection_requires_and_records_95_percent_identity() -> None:
    cell = SimpleNamespace(
        cell_id=_sha(900),
        method_role="LightCone",
        dimensions=(),
    )
    payload = {
        "metric_name": "registered_fixture_effect",
        "point_estimate": 0.1,
        "ci_low": -0.1,
        "ci_high": 0.2,
        "independent_block_count": 12,
        "request_count": 1_200,
        "paired": True,
        "confidence": 0.95,
        "reducer_method": "paired_block_bca",
    }
    rebuilt = SimpleNamespace(
        artifact=SimpleNamespace(node="e1"),
        decision=SimpleNamespace(
            stage="E1",
            phase="geometry",
            sha256=_sha(901),
            payload=payload,
        ),
        materialization=SimpleNamespace(cells=(cell,)),
    )
    metrics = _builder()._reducer_metrics(rebuilt)
    assert len(metrics) == 1
    assert metrics[0].metric_kind == "headline"
    assert metrics[0].attributes["confidence_level"] == 0.95

    rebuilt.decision.payload["confidence"] = 0.9
    with pytest.raises(ValueError, match="95%"):
        _builder()._reducer_metrics(rebuilt)


def test_e5_excluded_p99_anchor_never_projects_a_headline_interval() -> None:
    excluded = {
        "anchor_id": "DFLASH:tp1_dp1:closed_loop_c1",
        "status": "EXCLUDED_UNSAFE_OR_INACTIVE",
        "reason_codes": ["LightCone:nonfinite_updates"],
        "excluded_roles": ["LightCone"],
        "evidence_cell_ids": [_sha(950)],
        "block_evidence": [{"block": 4}],
        "independent_block_count": 1,
        "request_count": 10_000,
    }
    payload = {"family_results": [], "p99_anchor_claims": [excluded]}

    assert _explicit_headline_metric_payload_rows("e5_final", payload) == ()

    excluded["point_estimate"] = 1.0
    excluded["ci_low"] = 0.9
    excluded["ci_high"] = 1.1
    with pytest.raises(ValueError, match="unresolved p99 anchor fields differ"):
        _explicit_headline_metric_payload_rows("e5_final", payload)
