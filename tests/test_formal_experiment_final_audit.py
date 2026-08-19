from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import lightcone_spec.orchestration.formal_experiment_final_audit as audit_module
from lightcone_spec.orchestration.autodl_provider_runtime import (
    AUTODL_LIST_PATH,
    AUTODL_POWER_OFF_PATH,
    AUTODL_STATUS_PATH,
    AutoDlApiResponse,
    transition_autodl_instance_power,
)
from lightcone_spec.orchestration.experiment_operator import (
    ArchiveRequest,
    ArchiveStepReceipt,
    CellAttemptSpec,
    ControllerArtifactBinding,
    ExperimentOperatorStore,
    MetricRecord,
    ProviderRuntimeSample,
    default_formal_stage_plan,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    canonical_json_bytes,
)
from lightcone_spec.orchestration.formal_experiment_final_audit import (
    FINAL_ARCHIVE_SAFE_BOUNDARY,
    FormalExperimentFinalAuditError,
    audit_finalization_readiness,
    load_final_completion,
    publish_final_completion,
    publish_pre_shutdown_audit,
)

INSTANCE = "pro-final-audit-fixture"
RUN_ID = "formal-v03-final-audit"
START = "2026-08-19T04:16:21.123456789+08:00"
STOP = "2026-08-19T04:21:21.123456789+08:00"


def _rfc3339_ns(value: str) -> int:
    whole, zone = value.rsplit("+", 1)
    clock, fraction = whole.split(".")
    parsed = datetime.fromisoformat(f"{clock}+{zone}")
    return int(parsed.timestamp()) * 1_000_000_000 + int(fraction)


START_NS = _rfc3339_ns(START)
STOP_NS = _rfc3339_ns(STOP)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha(payload)


def _write_canonical(path: Path, value: object) -> str:
    return _write(path, canonical_json_bytes(value))


class _Clock:
    def __init__(self, value: int = START_NS + 10_000_000_000) -> None:
        self.value = value

    def __call__(self) -> int:
        self.value += 1_000_000
        return self.value


@dataclass
class _ReadyRun:
    store: ExperimentOperatorStore
    clock: _Clock
    progress_root: Path
    archive_id: str
    shutdown_probe: Path
    pre_output: Path
    archive_sources: dict[str, Path]
    chain: object


def _artifact(
    root: Path, name: str, sources: dict[str, Path]
) -> ControllerArtifactBinding:
    path = (root / "controller" / name).resolve()
    _write(path, (name + "\n").encode())
    binding = ControllerArtifactBinding.bind(path)
    sources[binding.sha256] = path
    return binding


def _fake_completion_chain(
    rows: tuple[dict[str, object], ...], cell_ids: tuple[str, ...]
) -> object:
    prior = None
    for row, cell_id in zip(rows, cell_ids, strict=True):
        materialization_semantic = _sha(f"materialization:{cell_id}".encode())
        node_semantic = _sha(f"node:{cell_id}".encode())
        decision_semantic = _sha(f"decision:{cell_id}".encode())
        node_source = SimpleNamespace(
            absolute_path=row["node_materialization_path"],
            raw_sha256=row["node_materialization_sha256"],
        )
        materialization_source = SimpleNamespace(
            absolute_path=row["materialization_path"],
            raw_sha256=row["materialization_sha256"],
        )
        decision_source = SimpleNamespace(
            absolute_path=row["decision_path"],
            raw_sha256=row["decision_sha256"],
        )
        predecessor_source = (
            None
            if prior is None
            else SimpleNamespace(absolute_path=prior._fixture_completion_path)
        )
        artifact = SimpleNamespace(
            node=row["node"],
            ordinal=row["ordinal"],
            node_materialization_source=node_source,
            node_materialization_sha256=node_semantic,
            materialization_sha256=materialization_semantic,
            decision_source=decision_source,
            decision_sha256=decision_semantic,
            predecessor_source=predecessor_source,
            actual_results=(SimpleNamespace(cell_id=cell_id),),
        )
        prior = SimpleNamespace(
            artifact=artifact,
            predecessor=prior,
            node_materialization=SimpleNamespace(
                materialization_source=materialization_source,
                sha256=node_semantic,
            ),
            materialization=SimpleNamespace(
                cells=(SimpleNamespace(cell_id=cell_id),),
                sha256=materialization_semantic,
            ),
            decision=SimpleNamespace(sha256=decision_semantic),
            _fixture_completion_path=row["completion_path"],
        )
    assert prior is not None
    return prior


def _build_archive(
    run: _ReadyRun,
    *,
    rehydrate: bool,
) -> None:
    root = (run.progress_root.parent / "final-archive").resolve()
    root.mkdir()
    rows = []
    for index, (digest, source) in enumerate(sorted(run.archive_sources.items())):
        destination = root / "payload" / f"{index:04d}-{source.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assert _sha(destination.read_bytes()) == digest
        rows.append(
            {
                "path": destination.relative_to(root).as_posix(),
                "sha256": digest,
                "size_bytes": destination.stat().st_size,
            }
        )
    rows.sort(key=lambda row: row["path"])
    manifest = {
        "schema_version": 1,
        "kind": "formal_archive_sha256_manifest",
        "files": rows,
    }
    manifest_path = root / "sha256_manifest.json"
    manifest_sha = _write_canonical(manifest_path, manifest)
    content_tree = _sha(
        canonical_json_bytes({"manifest_sha256": manifest_sha, "files": rows})
    )
    request = ArchiveRequest(
        archive_id=run.archive_id,
        safe_boundary=FINAL_ARCHIVE_SAFE_BOUNDARY,
        remote_payload_root="/srv/lightcone-tts-runtime/v03/final-wave",
        local_partial_root=str((root.parent / "final-archive.partial").resolve()),
        local_final_root=str(root),
        remote_manifest_sha256=manifest_sha,
        predicted_payload_bytes=sum(row["size_bytes"] for row in rows),
    )
    run.store.register_archive_safe_boundary(request)
    run.store.record_archive_step(
        run.archive_id,
        ArchiveStepReceipt(
            "TRANSFER",
            manifest_sha,
            "a" * 64,
            len(rows),
            request.predicted_payload_bytes,
        ),
    )
    run.store.record_archive_step(
        run.archive_id,
        ArchiveStepReceipt(
            "LOCAL_SHA_VERIFY",
            manifest_sha,
            "b" * 64,
            len(rows),
            request.predicted_payload_bytes,
        ),
    )
    if rehydrate:
        run.store.record_archive_step(
            run.archive_id,
            ArchiveStepReceipt(
                "REHYDRATE_VERIFY",
                manifest_sha,
                "c" * 64,
                len(rows),
                request.predicted_payload_bytes,
                content_tree_sha256=content_tree,
            ),
        )
        run.store.authorize_remote_eviction(run.archive_id)


def _build_ready_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    metric_mode: str = "valid",
    missing_cell: bool = False,
    archive_rehydrated: bool = True,
    double_accounting: bool = False,
    unsafe_probe: bool = False,
) -> _ReadyRun:
    clock = _Clock()
    store = ExperimentOperatorStore(
        tmp_path / "operator.sqlite3",
        run_id=RUN_ID,
        clock_ns=clock,
    )
    plan = tuple(
        replace(row, expected_formula="fixture:1", known_expected_cells=1)
        for row in default_formal_stage_plan()
    )
    store.initialize_stage_plan(plan)
    archive_sources: dict[str, Path] = {}
    evidence_payloads = {
        "terminal": b"terminal evidence\n",
        "junit": b"<testsuite tests='1'/>\n",
        "raw": b'{"request":"terminal"}\n',
        "detail": b"native timestamps and lifecycle\n",
    }
    evidence_paths = {}
    for name, payload in evidence_payloads.items():
        path = (tmp_path / "raw-evidence" / name).resolve()
        digest = _write(path, payload)
        archive_sources[digest] = path
        evidence_paths[name] = (path, digest)

    cell_ids = []
    for index, stage in enumerate(plan):
        cell_id = f"{stage.node}:fixture-cell"
        cell_ids.append(cell_id)
        materialization = _artifact(
            tmp_path, f"{index:02d}-materialization.json", archive_sources
        )
        node_materialization = _artifact(
            tmp_path, f"{index:02d}-node-materialization.json", archive_sources
        )
        execution = _artifact(tmp_path, f"{index:02d}-execution.json", archive_sources)
        decision = _artifact(tmp_path, f"{index:02d}-decision.json", archive_sources)
        completion = _artifact(
            tmp_path, f"{index:02d}-completion.json", archive_sources
        )
        store.record_controller_materialization(
            node=stage.node,
            materialization=materialization,
            node_materialization=node_materialization,
            expected_cell_ids=(cell_id,),
        )
        spec = CellAttemptSpec(
            cell_id=cell_id,
            attempt=1,
            stage=stage.stage,
            phase=stage.phase,
            block=f"block-{index:02d}",
            seed=index + 1,
            scientific_axes={"task": "serving", "node": stage.node},
            identity={
                "source_sha256": "1" * 64,
                "patch_sha256": "2" * 64,
                "registry_sha256": "3" * 64,
            },
            command_sha256="4" * 64,
            output_directory=str((tmp_path / "attempts" / stage.node).resolve()),
        )
        store.materialize_attempt(spec)
        store.record_controller_execution_plan(
            node=stage.node,
            execution_source=execution,
        )
        started = clock()
        store.mark_running_before_spawn(
            cell_id,
            1,
            assigned_gpu_uuids=("GPU-fixture-0",),
            started_at_ns=started,
        )
        store.attach_process(cell_id, 1, pid=10_000 + index, pgid=10_000 + index)
        store.finish_attempt(
            cell_id,
            1,
            status="COMPLETE",
            exit_code=0,
            terminal_sha256=evidence_paths["terminal"][1],
            junit_sha256=evidence_paths["junit"][1],
            raw_log_sha256=evidence_paths["raw"][1],
            evidence_files={
                str(evidence_paths["detail"][0]): evidence_paths["detail"][1]
            },
            included_in_analysis=True,
            exclusion_reason=None,
            compute_gpu_seconds=1.0,
            reserved_gpu_seconds=1.25,
            finished_at_ns=max(clock(), started + 1),
        )
        store.record_controller_reduction(
            node=stage.node,
            decision=decision,
            completion=completion,
        )
        if metric_mode != "none":
            store.record_metric(
                MetricRecord(
                    stage=stage.stage,
                    phase=stage.phase,
                    cell_id=cell_id,
                    attempt=1,
                    metric_name="committed_token_goodput",
                    metric_kind="headline",
                    point_estimate=1.0 + index / 100,
                    ci_low=0.9,
                    ci_high=1.1,
                    independent_block_count=12,
                    request_count=1_000,
                    paired=True,
                    reducer_method="paired_bca_bootstrap_v1",
                    attributes={"confidence_level": 0.95},
                )
            )
        store.record_selection_decision(
            decision_id=f"{stage.node}-selection-authority",
            stage=stage.stage,
            phase=stage.phase,
            decision_kind="sealed_node_output",
            source_sha256="5" * 64,
            decision={"node": stage.node, "sealed": True},
        )
    store.record_provider_runtime_sample(
        ProviderRuntimeSample(
            instance_uuid=INSTANCE,
            state="running",
            observed_at_ns=START_NS + 100_000_000_000,
            provider_started_at_ns=START_NS,
            provider_stopped_at_ns=None,
            gpu_count=2,
            response_sha256="6" * 64,
        )
    )
    store.set_dispatch_stop("DAG_REDUCED_AWAITING_FINAL_AUDIT")

    rows = store.controller_nodes()
    chain = _fake_completion_chain(rows, tuple(cell_ids))
    monkeypatch.setattr(
        audit_module, "_rebuild_final_stage_completion", lambda _path: chain
    )

    if metric_mode == "missing_ci":
        store._connection.execute("UPDATE metrics_long SET ci_low = NULL")
    if metric_mode == "descriptive_ci":
        store._connection.execute(
            "UPDATE metrics_long SET metric_kind = 'descriptive' WHERE cell_id = ?",
            (cell_ids[0],),
        )
    if missing_cell:
        victim = cell_ids[-1]
        store._connection.execute("PRAGMA foreign_keys=OFF")
        store._connection.execute(
            "DELETE FROM metrics_long WHERE cell_id = ?", (victim,)
        )
        store._connection.execute(
            "DELETE FROM cell_attempts WHERE cell_id = ?", (victim,)
        )
        store._connection.execute("PRAGMA foreign_keys=ON")
    if double_accounting:
        members = cell_ids[:10]
        store._connection.execute(
            "INSERT INTO physical_attempt_groups "
            "(group_id, leader_cell_id, leader_attempt, status, shared_evidence_sha256, "
            "created_at_ns, updated_at_ns) VALUES (?, ?, 1, 'COMPLETE', ?, ?, ?)",
            ("double-charge-group", members[0], "7" * 64, clock(), clock()),
        )
        for ordinal, cell_id in enumerate(members):
            store._connection.execute(
                "INSERT INTO physical_attempt_group_members "
                "(group_id, cell_id, attempt, logical_kind, member_ordinal) "
                "VALUES (?, ?, 1, ?, ?)",
                (
                    "double-charge-group",
                    cell_id,
                    (
                        "compile"
                        if ordinal == 0
                        else "exactness"
                        if ordinal == 1
                        else "interference"
                    ),
                    ordinal,
                ),
            )

    run = _ReadyRun(
        store=store,
        clock=clock,
        progress_root=(tmp_path / "progress").resolve(),
        archive_id="final-complete-archive",
        shutdown_probe=(tmp_path / "shutdown-probe.json").resolve(),
        pre_output=(tmp_path / "pre-shutdown-audit.json").resolve(),
        archive_sources=archive_sources,
        chain=chain,
    )
    _build_archive(run, rehydrate=archive_rehydrated)
    store.export_progress(run.progress_root, exported_at_ns=START_NS + 180_000_000_000)
    probe = {
        "schema_version": 1,
        "kind": "autodl_power_off_safety_probe",
        "instance_uuid": INSTANCE,
        "run_id": RUN_ID,
        "observed_at_ns": START_NS + 190_000_000_000,
        "observation_window_seconds": 5,
        "scheduler_control_state": "STOP",
        "running_attempt_count": 0,
        "evidence_writer_process_count": 0,
        "gpu_compute_process_count": 0,
        "open_measurement_port_count": 0,
        "log_growth_bytes": 1 if unsafe_probe else 0,
        "probe_command_sha256": "8" * 64,
    }
    _write_canonical(run.shutdown_probe, probe)
    return run


def _publish_pre(run: _ReadyRun):
    return publish_pre_shutdown_audit(
        store=run.store,
        instance_uuid=INSTANCE,
        progress_export_root=run.progress_root,
        final_archive_id=run.archive_id,
        shutdown_probe_path=run.shutdown_probe,
        output_path=run.pre_output,
        audited_at_ns=START_NS + 200_000_000_000,
    )


def _status(state: str) -> AutoDlApiResponse:
    return AutoDlApiResponse(
        200,
        {
            "code": "Success",
            "data": state,
            "msg": "",
            "request_id": f"status-{state}",
        },
    )


def _listing(state: str) -> AutoDlApiResponse:
    return AutoDlApiResponse(
        200,
        {
            "code": "Success",
            "data": {
                "list": [
                    {
                        "uuid": INSTANCE,
                        "status": state,
                        "req_gpu_amount": 2,
                        "started_at": {"Time": START, "Valid": True},
                        "stopped_at": {
                            "Time": STOP if state == "shutdown" else "",
                            "Valid": state == "shutdown",
                        },
                    }
                ],
                "page_index": 1,
                "page_size": 100,
                "max_page": 1,
            },
            "msg": "",
            "request_id": f"list-{state}",
        },
    )


class _ShutdownClient:
    def request(self, method, path, body):
        if path == AUTODL_POWER_OFF_PATH:
            return AutoDlApiResponse(
                200,
                {
                    "code": "Success",
                    "data": None,
                    "msg": "",
                    "request_id": "power-off-final-audit",
                },
            )
        if path == AUTODL_STATUS_PATH:
            return _status("shutdown")
        assert path == AUTODL_LIST_PATH
        return _listing("shutdown")


def _power_off(run: _ReadyRun) -> Path:
    path = (run.pre_output.parent / "power-off-evidence.json").resolve()
    transition_autodl_instance_power(
        store=run.store,
        operation="power_off",
        instance_uuid=INSTANCE,
        output_path=path,
        safety_probe_path=run.shutdown_probe,
        environment={"AUTODL_DEVELOPER_TOKEN": "fixture-private-token"},
        client_factory=lambda _token: _ShutdownClient(),
        clock_ns=lambda: START_NS + 400_000_000_000,
        confirmation_interval_seconds=0,
    )
    return path


def test_two_phase_audit_is_the_only_final_completion_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _build_ready_run(tmp_path, monkeypatch)
    try:
        pre = _publish_pre(run)
        assert pre.controller_state == "DAG_REDUCED_AWAITING_FINAL_AUDIT"
        assert pre.formal_measured is False
        power = _power_off(run)
        output = (tmp_path / "final-completion.json").resolve()
        final = publish_final_completion(
            store=run.store,
            pre_shutdown_audit_path=run.pre_output,
            power_transition_evidence_path=power,
            output_path=output,
            finalized_at_ns=START_NS + 500_000_000_000,
        )
        assert final.status == "COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL"
        assert final.formal_measured is False
        assert final.billed_gpu_hours > 0
        assert load_final_completion(output) == final
        assert (
            publish_final_completion(
                store=run.store,
                pre_shutdown_audit_path=run.pre_output,
                power_transition_evidence_path=power,
                output_path=output,
                finalized_at_ns=START_NS + 500_000_000_000,
            )
            == final
        )
        with pytest.raises(FormalExperimentFinalAuditError, match="differs"):
            publish_final_completion(
                store=run.store,
                pre_shutdown_audit_path=run.pre_output,
                power_transition_evidence_path=power,
                output_path=output,
                finalized_at_ns=START_NS + 500_000_000_001,
            )
    finally:
        run.store.close()


def test_pre_archive_readiness_reuses_the_deep_scientific_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _build_ready_run(tmp_path, monkeypatch)
    try:
        readiness = audit_finalization_readiness(run.store)
        assert readiness.node_count == 21
        assert readiness.expected_cell_count == 21
        assert readiness.latest_complete_attempt_count == 21
        assert readiness.selection_decision_count == 21
        assert readiness.metric_count == 21
        assert readiness.headline_metric_count == 21
        assert readiness.required_archive_sha256s
    finally:
        run.store.close()


@pytest.mark.parametrize("metric_mode", ["none", "missing_ci", "descriptive_ci"])
def test_reduced_dag_without_complete_metrics_cannot_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_mode: str,
) -> None:
    run = _build_ready_run(tmp_path, monkeypatch, metric_mode=metric_mode)
    try:
        with pytest.raises(FormalExperimentFinalAuditError, match="metric|headline"):
            _publish_pre(run)
    finally:
        run.store.close()


def test_bidirectional_audit_rejects_missing_materialized_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _build_ready_run(tmp_path, monkeypatch, missing_cell=True)
    try:
        with pytest.raises(FormalExperimentFinalAuditError, match="missing"):
            _publish_pre(run)
    finally:
        run.store.close()


def test_final_archive_must_be_fully_rehydrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _build_ready_run(tmp_path, monkeypatch, archive_rehydrated=False)
    try:
        with pytest.raises(FormalExperimentFinalAuditError, match="rehydrate"):
            _publish_pre(run)
    finally:
        run.store.close()


def test_physical_group_double_accounting_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _build_ready_run(tmp_path, monkeypatch, double_accounting=True)
    try:
        with pytest.raises(FormalExperimentFinalAuditError, match="double-accounted"):
            _publish_pre(run)
    finally:
        run.store.close()


def test_unsafe_shutdown_probe_cannot_authorize_pre_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _build_ready_run(tmp_path, monkeypatch, unsafe_probe=True)
    try:
        with pytest.raises(FormalExperimentFinalAuditError, match="shutdown probe"):
            _publish_pre(run)
    finally:
        run.store.close()


def test_finalization_rejects_open_provider_billing_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _build_ready_run(tmp_path, monkeypatch)
    try:
        _publish_pre(run)
        power = _power_off(run)
        run.store._connection.execute(
            "DELETE FROM provider_runtime_samples WHERE state = 'shutdown'"
        )
        with pytest.raises(FormalExperimentFinalAuditError, match="billing interval"):
            publish_final_completion(
                store=run.store,
                pre_shutdown_audit_path=run.pre_output,
                power_transition_evidence_path=power,
                output_path=(tmp_path / "must-not-exist.json").resolve(),
                finalized_at_ns=START_NS + 500_000_000_000,
            )
    finally:
        run.store.close()


def test_finalization_rejects_tampered_or_foreign_power_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _build_ready_run(tmp_path, monkeypatch)
    try:
        _publish_pre(run)
        power = _power_off(run)
        value = json.loads(power.read_bytes())
        value["receipt"]["instance_uuid"] = "pro-foreign-instance"
        power.write_bytes(canonical_json_bytes(value))
        with pytest.raises(FormalExperimentFinalAuditError, match="foreign|receipt"):
            publish_final_completion(
                store=run.store,
                pre_shutdown_audit_path=run.pre_output,
                power_transition_evidence_path=power,
                output_path=(tmp_path / "foreign-final.json").resolve(),
                finalized_at_ns=START_NS + 500_000_000_000,
            )
    finally:
        run.store.close()
