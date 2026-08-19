from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import MethodType

import pytest

from lightcone_spec.orchestration.autodl_provider_runtime import (
    AUTODL_LIST_PATH,
    AUTODL_POWER_OFF_PATH,
    AUTODL_STATUS_PATH,
    AutoDlApiResponse,
)
from lightcone_spec.orchestration.experiment_operator import (
    ArchiveRequest,
    ArchiveStepReceipt,
    CellAttemptSpec,
    ExperimentOperatorStore,
    ProviderRuntimeSample,
    default_formal_stage_plan,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    canonical_json_bytes,
)
from lightcone_spec.orchestration.formal_experiment_final_audit import (
    FINAL_ARCHIVE_SAFE_BOUNDARY,
    FORMAL_EXPERIMENT_FINAL_AUDIT_PROTOCOL_SHA256,
    TRUSTED_SINGLE_OPERATOR_EMPIRICAL,
    FinalAuditArtifactBinding,
    FormalExperimentFinalCompletionReceipt,
    FormalExperimentFinalizationReadiness,
    FormalExperimentPreShutdownAuditReceipt,
)
from lightcone_spec.orchestration.formal_experiment_production_finalizer import (
    FormalExperimentProductionFinalizer,
    FormalExperimentProductionFinalizerError,
    ProductionFinalizerRuntime,
    load_production_finalizer_completion,
    publish_path_bound_production_finalizer_config,
)
from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
    publish_path_bound_formal_dag_driver_config,
)
from lightcone_spec.runtime.proof_artifact import publish_canonical_json_no_replace

INSTANCE = "pro-production-finalizer-fixture"
RUN_ID = "formal-v03-production-finalizer"
START = "2026-08-19T04:16:21.123456789+08:00"
STOP = "2026-08-19T04:18:01.123456789+08:00"


def _rfc3339_ns(value: str) -> int:
    whole, zone = value.rsplit("+", 1)
    clock, fraction = whole.split(".")
    parsed = datetime.fromisoformat(f"{clock}+{zone}")
    return int(parsed.timestamp()) * 1_000_000_000 + int(fraction)


START_NS = _rfc3339_ns(START)
STOP_NS = _rfc3339_ns(STOP)


class _Clock:
    def __init__(self) -> None:
        self.value = STOP_NS + 10_000_000_000

    def __call__(self) -> int:
        self.value += 1_000_000
        return self.value


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
    return path.resolve()


def _dummy(path: Path, name: str) -> Path:
    target = path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((name + "\n").encode())
    return target.resolve()


def _ready_store(database: Path, clock: _Clock) -> ExperimentOperatorStore:
    store = ExperimentOperatorStore(database, run_id=RUN_ID, clock_ns=clock)
    store.initialize_stage_plan(default_formal_stage_plan())
    store._connection.execute("UPDATE controller_nodes SET state = 'REDUCED'")
    store.record_provider_runtime_sample(
        ProviderRuntimeSample(
            instance_uuid=INSTANCE,
            state="running",
            observed_at_ns=START_NS + 1_000_000_000,
            provider_started_at_ns=START_NS,
            provider_stopped_at_ns=None,
            gpu_count=2,
            response_sha256="6" * 64,
        )
    )
    return store


def _config(tmp_path: Path):
    repository = (tmp_path / "repository").resolve()
    run_root = (tmp_path / RUN_ID).resolve()
    repository.mkdir()
    run_root.mkdir()
    catalog = (tmp_path / "prerequisites").resolve()
    catalog.mkdir()
    sources = tmp_path / "sources"
    driver_path = (run_root / "driver-config.json").resolve()
    publish_path_bound_formal_dag_driver_config(
        repository_root=repository,
        run_root=run_root,
        protocol_lock_path=_dummy(sources, "protocol-lock.json"),
        content_source_path=_dummy(sources, "content-source.json"),
        runtime_authority_manifest_path=_dummy(sources, "runtime.json"),
        inventory_path=_dummy(sources, "inventory.json"),
        doctor_report_path=_dummy(sources, "doctor.json"),
        preflight_workload_authority_path=_dummy(sources, "workload.json"),
        profiler_tool_paths=(),
        prerequisite_index_catalog_directory=catalog,
        output_path=driver_path,
    )
    instance = _write(
        sources / "instance.json",
        {
            "schema_version": 1,
            "kind": "formal_autodl_instance_identity",
            "instance_uuid": INSTANCE,
        },
    )
    ports = _write(
        sources / "ports.json",
        {
            "schema_version": 1,
            "kind": "formal_measurement_port_registry",
            "ports": [31900, 31910],
        },
    )
    payload_root = (run_root / "sealed-final-payload").resolve()
    payload_root.mkdir()
    local_final = (tmp_path / "local-results" / "final.final").resolve()
    request = ArchiveRequest(
        archive_id="whole-run-final-archive",
        safe_boundary=FINAL_ARCHIVE_SAFE_BOUNDARY,
        remote_payload_root=str(payload_root),
        local_partial_root=str(local_final.with_suffix(".partial")),
        local_final_root=str(local_final),
        remote_manifest_sha256="a" * 64,
        predicted_payload_bytes=11,
    )
    request_path = _write(sources / "archive-request.json", asdict(request))
    return publish_path_bound_production_finalizer_config(
        dag_driver_config_path=driver_path,
        instance_identity_path=instance,
        measurement_port_registry_path=ports,
        final_archive_request_path=request_path,
        rehydration_catalog_path=None,
        finalization_root=run_root / "finalization",
        output_path=run_root / "finalization-config.json",
    )


def _readiness(store: ExperimentOperatorStore):
    return FormalExperimentFinalizationReadiness(
        run_id=store.run_id,
        node_count=21,
        expected_cell_count=1,
        latest_complete_attempt_count=1,
        retained_retry_attempt_count=0,
        selection_decision_count=1,
        metric_count=1,
        headline_metric_count=1,
        expected_cell_ids_sha256="1" * 64,
        coverage_sha256="2" * 64,
        accounting_sha256="3" * 64,
        compute_gpu_seconds=0.0,
        reserved_gpu_seconds=0.0,
        allocated_billed_gpu_seconds=0.0,
        required_archive_sha256s=frozenset({"4" * 64}),
    )


def _safe_probe(**kwargs):
    return {
        "schema_version": 1,
        "kind": "autodl_power_off_safety_probe",
        "instance_uuid": kwargs["instance_uuid"],
        "run_id": RUN_ID,
        "observed_at_ns": kwargs["clock_ns"](),
        "observation_window_seconds": 5,
        "scheduler_control_state": "STOP",
        "running_attempt_count": 0,
        "evidence_writer_process_count": 0,
        "gpu_compute_process_count": 0,
        "open_measurement_port_count": 0,
        "log_growth_bytes": 0,
        "probe_command_sha256": "5" * 64,
    }


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


class _FakeClient:
    def __init__(self, *, disagree: bool = False) -> None:
        self.disagree = disagree
        self.power_calls = 0

    def request(self, method, path, body):
        if path == AUTODL_POWER_OFF_PATH:
            self.power_calls += 1
            return AutoDlApiResponse(
                200,
                {
                    "code": "Success",
                    "data": None,
                    "msg": "",
                    "request_id": "power-off-once",
                },
            )
        if path == AUTODL_STATUS_PATH:
            return _status("shutdown")
        assert path == AUTODL_LIST_PATH
        return _listing("running" if self.disagree else "shutdown")


def _archive_complete(self, store, request):
    store.register_archive_safe_boundary(request)
    store.record_archive_step(
        request.archive_id,
        ArchiveStepReceipt("TRANSFER", request.remote_manifest_sha256, "7" * 64, 1, 11),
    )
    store.record_archive_step(
        request.archive_id,
        ArchiveStepReceipt(
            "LOCAL_SHA_VERIFY", request.remote_manifest_sha256, "8" * 64, 1, 11
        ),
    )
    store.record_archive_step(
        request.archive_id,
        ArchiveStepReceipt(
            "REHYDRATE_VERIFY",
            request.remote_manifest_sha256,
            "9" * 64,
            1,
            11,
            content_tree_sha256="b" * 64,
        ),
    )
    store.authorize_remote_eviction(request.archive_id)


def _pre_publisher(clock: _Clock):
    def publish(**kwargs):
        progress_manifest = FinalAuditArtifactBinding.bind(
            Path(kwargs["progress_export_root"]) / "export_manifest.json",
            label="progress manifest",
        )
        manifest = json.loads(Path(progress_manifest.absolute_path).read_bytes())
        files = tuple(sorted(manifest["files"].items()))
        probe = FinalAuditArtifactBinding.bind(
            kwargs["shutdown_probe_path"], label="shutdown probe"
        )
        receipt = FormalExperimentPreShutdownAuditReceipt(
            schema_version=1,
            kind="formal_experiment_pre_shutdown_audit",
            protocol_sha256=FORMAL_EXPERIMENT_FINAL_AUDIT_PROTOCOL_SHA256,
            run_id=RUN_ID,
            instance_uuid=INSTANCE,
            trust=TRUSTED_SINGLE_OPERATOR_EMPIRICAL,
            formal_measured=False,
            controller_state="DAG_REDUCED_AWAITING_FINAL_AUDIT",
            audited_at_ns=clock(),
            node_count=21,
            expected_cell_count=1,
            expected_cell_ids_sha256="1" * 64,
            latest_complete_attempt_count=1,
            retained_retry_attempt_count=0,
            selection_decision_count=1,
            metric_count=1,
            headline_metric_count=1,
            coverage_sha256="2" * 64,
            selection_sha256=dict(files)["selection_decisions.jsonl"],
            metrics_sha256=dict(files)["metrics_long.parquet"],
            accounting_sha256="3" * 64,
            compute_gpu_seconds=0.0,
            reserved_gpu_seconds=0.0,
            allocated_billed_gpu_seconds=0.0,
            observed_whole_instance_billed_gpu_seconds=2.0,
            observed_wall_time_seconds=1.0,
            progress_export_manifest=progress_manifest,
            progress_export_files=files,
            final_archive_id="whole-run-final-archive",
            final_archive_manifest_sha256="a" * 64,
            final_archive_content_tree_sha256="b" * 64,
            final_archive_local_root=str(
                (
                    Path(kwargs["progress_export_root"]).parents[2] / "local-final"
                ).resolve()
            ),
            shutdown_probe=probe,
        )
        publish_canonical_json_no_replace(kwargs["output_path"], receipt.to_dict())
        return receipt

    return publish


def _final_publisher(clock: _Clock):
    def publish(**kwargs):
        power = json.loads(Path(kwargs["power_transition_evidence_path"]).read_bytes())
        transition = power["receipt"]
        pre = FormalExperimentPreShutdownAuditReceipt.from_dict(
            json.loads(Path(kwargs["pre_shutdown_audit_path"]).read_bytes())
        )
        receipt = FormalExperimentFinalCompletionReceipt(
            schema_version=1,
            kind="formal_experiment_final_completion",
            protocol_sha256=FORMAL_EXPERIMENT_FINAL_AUDIT_PROTOCOL_SHA256,
            status="COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL",
            trust=TRUSTED_SINGLE_OPERATOR_EMPIRICAL,
            formal_measured=False,
            run_id=RUN_ID,
            instance_uuid=INSTANCE,
            finalized_at_ns=clock(),
            pre_shutdown_audit=FinalAuditArtifactBinding.bind(
                kwargs["pre_shutdown_audit_path"], label="pre audit"
            ),
            power_transition_evidence=FinalAuditArtifactBinding.bind(
                kwargs["power_transition_evidence_path"], label="power evidence"
            ),
            power_transition_receipt_sha256=power["receipt_sha256"],
            provider_request_id=transition["provider_request_id"],
            provider_sample_id=transition["provider_sample_id"],
            provider_response_sha256=transition["provider_response_sha256"],
            shutdown_probe_sha256=pre.shutdown_probe.sha256,
            coverage_sha256=pre.coverage_sha256,
            selection_sha256=pre.selection_sha256,
            metrics_sha256=pre.metrics_sha256,
            archive_manifest_sha256=pre.final_archive_manifest_sha256,
            archive_content_tree_sha256=pre.final_archive_content_tree_sha256,
            progress_export_manifest_sha256=pre.progress_export_manifest.sha256,
            wall_time_seconds=100.0,
            powered_wall_time_seconds=100.0,
            compute_gpu_hours=0.0,
            reserved_gpu_hours=0.0,
            billed_gpu_hours=200.0 / 3600.0,
        )
        publish_canonical_json_no_replace(kwargs["output_path"], receipt.to_dict())
        return receipt

    return publish


def _runtime(clock: _Clock, client: _FakeClient) -> ProductionFinalizerRuntime:
    runtime = ProductionFinalizerRuntime(
        readiness_auditor=_readiness,
        probe_collector=_safe_probe,
        pre_shutdown_publisher=_pre_publisher(clock),
        final_publisher=_final_publisher(clock),
        provider_client_factory=lambda _token: client,
        environment={"AUTODL_DEVELOPER_TOKEN": "in-memory-fixture-token"},
        clock_ns=clock,
        sleeper=lambda _seconds: None,
        maximum_confirmation_attempts=1,
        confirmation_interval_seconds=0,
    )
    runtime.archive = MethodType(_archive_complete, runtime)
    return runtime


def test_success_is_idempotent_and_power_off_is_called_once(tmp_path: Path) -> None:
    config = _config(tmp_path)
    clock = _Clock()
    client = _FakeClient()
    with _ready_store(config.database_path, clock) as store:
        assert store.dispatch_control()[0] == "RUN"
    finalizer = FormalExperimentProductionFinalizer(
        config, runtime=_runtime(clock, client)
    )
    first = finalizer.run()
    second = finalizer.run()
    assert first == second
    assert client.power_calls == 1
    assert (
        load_production_finalizer_completion(config.supervisor_completion_path) == first
    )
    assert first.whole_instance_billed_gpu_hours == pytest.approx(200.0 / 3600.0)
    accounting = json.loads(config.accounting_path.read_bytes())
    assert accounting["archive_checkpoint_windows"][0]["additive"] is True
    assert accounting["idle_and_control_residual_gpu_seconds"] >= 0
    for path in Path(config.finalization_root).iterdir():
        if path.is_file():
            assert b"in-memory-fixture-token" not in path.read_bytes()


def test_running_attempt_blocks_archive_and_shutdown(tmp_path: Path) -> None:
    config = _config(tmp_path)
    clock = _Clock()
    client = _FakeClient()
    with _ready_store(config.database_path, clock) as store:
        spec = CellAttemptSpec(
            cell_id="running-cell",
            attempt=1,
            stage="preflight",
            phase="final",
            block="block-running",
            seed=1,
            scientific_axes={"task": "serving"},
            identity={
                "source_sha256": "1" * 64,
                "patch_sha256": "2" * 64,
                "registry_sha256": "3" * 64,
            },
            command_sha256="4" * 64,
            output_directory=str((config.run_root / "running-cell").resolve()),
        )
        store.materialize_attempt(spec)
        store.mark_running_before_spawn(
            spec.cell_id,
            spec.attempt,
            assigned_gpu_uuids=("GPU-fixture",),
        )
    with pytest.raises(
        FormalExperimentProductionFinalizerError,
        match="RUNNING",
    ):
        FormalExperimentProductionFinalizer(
            config, runtime=_runtime(clock, client)
        ).run()
    assert client.power_calls == 0
    with ExperimentOperatorStore(config.database_path) as store:
        assert store.dispatch_control()[0] == "STOP"


def test_writer_probe_blocks_power_off(tmp_path: Path) -> None:
    config = _config(tmp_path)
    clock = _Clock()
    client = _FakeClient()
    with _ready_store(config.database_path, clock):
        pass
    runtime = _runtime(clock, client)

    def unsafe_probe(**kwargs):
        value = _safe_probe(**kwargs)
        value["evidence_writer_process_count"] = 1
        return value

    runtime.probe_collector = unsafe_probe
    with pytest.raises(
        FormalExperimentProductionFinalizerError,
        match="writer",
    ):
        FormalExperimentProductionFinalizer(config, runtime=runtime).run()
    assert client.power_calls == 0
    assert not config.pre_shutdown_audit_path.exists()


def test_incomplete_archive_blocks_probe_and_power_off(tmp_path: Path) -> None:
    config = _config(tmp_path)
    clock = _Clock()
    client = _FakeClient()
    with _ready_store(config.database_path, clock):
        pass
    runtime = _runtime(clock, client)

    def incomplete(self, store, request):
        store.register_archive_safe_boundary(request)

    runtime.archive = MethodType(incomplete, runtime)
    with pytest.raises(
        FormalExperimentProductionFinalizerError,
        match="archive",
    ):
        FormalExperimentProductionFinalizer(config, runtime=runtime).run()
    assert client.power_calls == 0
    assert not config.shutdown_probe_path.exists()


def test_status_list_disagreement_resumes_without_second_mutation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    clock = _Clock()
    client = _FakeClient(disagree=True)
    with _ready_store(config.database_path, clock):
        pass
    finalizer = FormalExperimentProductionFinalizer(
        config, runtime=_runtime(clock, client)
    )
    with pytest.raises(Exception, match="converge"):
        finalizer.run()
    assert client.power_calls == 1
    assert config.power_request_journal_path.is_file()
    assert not config.power_transition_path.exists()

    # The immutable pre-shutdown probe is now older than the first-mutation
    # freshness window.  A journaled restart must still perform only read-only
    # status/list confirmation and must never issue power_off again.
    clock.value += 301 * 1_000_000_000
    client.disagree = False
    receipt = finalizer.run()
    assert receipt.status == "COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL"
    assert client.power_calls == 1


def test_open_billing_failure_never_reissues_power_off(tmp_path: Path) -> None:
    config = _config(tmp_path)
    clock = _Clock()
    client = _FakeClient()
    with _ready_store(config.database_path, clock):
        pass
    runtime = _runtime(clock, client)
    power_calls = 0

    def incomplete_power(*, store, config, instance_uuid):
        nonlocal power_calls
        power_calls += 1
        _write(
            config.power_request_journal_path,
            {"kind": "fixture-power-request", "request_id": "only-once"},
        )
        _write(
            config.power_transition_path,
            {"kind": "fixture-power-transition", "state": "shutdown"},
        )

    def billing_open(**_kwargs):
        raise FormalExperimentProductionFinalizerError(
            "provider billing interval is open"
        )

    runtime.power_off = incomplete_power
    runtime.final_publisher = billing_open
    finalizer = FormalExperimentProductionFinalizer(config, runtime=runtime)
    with pytest.raises(FormalExperimentProductionFinalizerError, match="billing"):
        finalizer.run()
    with pytest.raises(FormalExperimentProductionFinalizerError, match="billing"):
        finalizer.run()
    assert power_calls == 1
