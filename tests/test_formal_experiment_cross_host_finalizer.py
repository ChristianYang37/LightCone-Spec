from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from lightcone_spec.orchestration.autodl_provider_runtime import (
    AUTODL_LIST_PATH,
    AUTODL_POWER_OFF_PATH,
    AUTODL_STATUS_PATH,
    AutoDlApiResponse,
    AutoDlProviderRuntimeError,
)
from lightcone_spec.orchestration.experiment_operator import (
    CellAttemptSpec,
    ExperimentOperatorStore,
    ProviderRuntimeSample,
    default_formal_stage_plan,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    ProductionArchiveRuntime,
    canonical_json_bytes,
)
from lightcone_spec.orchestration.formal_experiment_cross_host_finalizer import (
    CrossHostFinalizationTransport,
    FormalCrossHostFinalizationError,
    FormalCrossHostProductionFinalizer,
    PathBoundCrossHostFinalizerConfig,
    PathBoundRemoteClosureConfig,
    _parser,
    load_cross_host_final_completion,
    publish_cross_host_finalizer_config,
    publish_cross_host_ssh_endpoint,
    publish_remote_closure_config,
    publish_remote_post_archive_probe,
    seal_remote_scientific_closure,
)
from lightcone_spec.orchestration.formal_experiment_final_audit import (
    FormalExperimentFinalizationReadiness,
)
from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
    publish_path_bound_formal_dag_driver_config,
)

RUN_ID = "formal-v03-cross-host-finalizer"
INSTANCE = "pro-cross-host-finalizer"
START = "2026-08-19T04:16:21.123456789+08:00"
STOP = "2026-08-19T04:21:21.123456789+08:00"


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


def _dummy(root: Path, name: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name + "\n", encoding="utf-8")
    return path.resolve()


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


def _listing(state: str, *, start: str = START) -> AutoDlApiResponse:
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
                        "started_at": {"Time": start, "Valid": True},
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


class _FakeProvider:
    def __init__(self, *, disagree: bool = False, changed_start: bool = False) -> None:
        self.disagree = disagree
        self.changed_start = changed_start
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
                    "request_id": "power-off-exactly-once",
                },
            )
        if path == AUTODL_STATUS_PATH:
            return _status("shutdown")
        assert path == AUTODL_LIST_PATH
        if self.disagree:
            return _listing("running")
        changed = "2026-08-19T04:17:21.123456789+08:00"
        return _listing("shutdown", start=changed if self.changed_start else START)


class _IndeterminateProvider(_FakeProvider):
    def request(self, method, path, body):
        if path == AUTODL_POWER_OFF_PATH:
            self.power_calls += 1
            raise OSError("connection lost after provider accepted request")
        return super().request(method, path, body)


@dataclass
class _Harness:
    remote: PathBoundRemoteClosureConfig
    local: PathBoundCrossHostFinalizerConfig
    clock: _Clock
    readiness: FormalExperimentFinalizationReadiness


def _harness(tmp_path: Path) -> _Harness:
    clock = _Clock()
    repository = (tmp_path / "repository").resolve()
    run_root = (tmp_path / "remote" / RUN_ID).resolve()
    prerequisites = (tmp_path / "prerequisites").resolve()
    sources = (tmp_path / "sources").resolve()
    for path in (repository, run_root, prerequisites, sources):
        path.mkdir(parents=True)
    driver = run_root / "driver.json"
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
        prerequisite_index_catalog_directory=prerequisites,
        output_path=driver,
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
    evidence = _dummy(run_root / "raw", "terminal.log")
    evidence_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
    database = run_root / "operator.sqlite3"
    with ExperimentOperatorStore(database, run_id=RUN_ID, clock_ns=clock) as store:
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
    readiness = FormalExperimentFinalizationReadiness(
        run_id=RUN_ID,
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
        required_archive_sha256s=frozenset({evidence_sha}),
    )
    remote_config_path = run_root / "remote-finalizer.json"
    remote = publish_remote_closure_config(
        dag_driver_config_path=driver,
        instance_identity_path=instance,
        measurement_port_registry_path=ports,
        closure_root=run_root / "scientific-closure",
        payload_root=run_root / "whole-run-payload",
        output_path=remote_config_path,
    )
    endpoint = publish_cross_host_ssh_endpoint(
        output_path=tmp_path / "endpoint.json",
        ssh_target="root@example.test",
        ssh_port=40371,
        remote_python="/opt/lightcone/venv/bin/python",
        remote_finalizer_script="/opt/lightcone/finalizer.py",
        remote_config_path=str(remote_config_path),
        remote_closure_path=str(remote.receipt_path),
        remote_probe_root=str(Path(remote.closure_root) / "post-archive-probes"),
    )
    local = publish_cross_host_finalizer_config(
        endpoint_path=tmp_path / "endpoint.json",
        local_finalization_root=(tmp_path / "local" / RUN_ID).resolve(),
        output_path=tmp_path / "local-finalizer.json",
    )
    assert endpoint.remote_closure_path == str(remote.receipt_path)
    return _Harness(remote=remote, local=local, clock=clock, readiness=readiness)


class _MirrorTransport(CrossHostFinalizationTransport):
    def __init__(self, harness: _Harness, *, writer: bool = False) -> None:
        self.harness = harness
        self.writer = writer
        self.seal_calls = 0
        self.post_probe_calls = 0

    def _probe(self, **kwargs):
        value = _safe_probe(**kwargs)
        if self.writer:
            value["evidence_writer_process_count"] = 1
        return value

    def seal_remote(self) -> None:
        self.seal_calls += 1
        seal_remote_scientific_closure(
            self.harness.remote,
            readiness_auditor=lambda _store: self.harness.readiness,
            probe_collector=self._probe,
            clock_ns=self.harness.clock,
            sleeper=lambda _seconds: None,
        )

    def fetch_file(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, local_path)

    def publish_post_archive_probe(self, remote_output_path: str) -> None:
        self.post_probe_calls += 1
        publish_remote_post_archive_probe(
            self.harness.remote,
            output_path=remote_output_path,
            probe_collector=self._probe,
            clock_ns=self.harness.clock,
            sleeper=lambda _seconds: None,
        )

    def archive_runtime(self) -> ProductionArchiveRuntime:
        remote_payload = Path(self.harness.remote.payload_root)

        def runner(argv, *, check, shell):
            assert check is True and shell is False
            destination = Path(argv[-1].removesuffix("/"))
            shutil.copytree(remote_payload, destination, dirs_exist_ok=True)
            return subprocess.CompletedProcess(argv, 0)

        return ProductionArchiveRuntime(
            runner=runner,
            full_rehydrate=True,
            minimum_local_free_bytes=0,
        )


def _supervisor(
    harness: _Harness,
    transport: _MirrorTransport,
    provider: _FakeProvider,
) -> FormalCrossHostProductionFinalizer:
    return FormalCrossHostProductionFinalizer(
        harness.local,
        transport=transport,
        environment={"AUTODL_DEVELOPER_TOKEN": "memory-only-token"},
        provider_client_factory=lambda _token: provider,
        clock_ns=harness.clock,
        sleeper=lambda _seconds: None,
        maximum_confirmation_attempts=1,
        confirmation_interval_seconds=0,
    )


def test_split_finalizer_archives_rehydrates_and_never_rewrites_remote_sqlite(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    transport = _MirrorTransport(harness)
    transport.seal_remote()
    before = harness.remote.database_path.read_bytes()
    provider = _FakeProvider()
    supervisor = _supervisor(harness, transport, provider)
    first = supervisor.run()
    second = supervisor.run()
    assert (
        first
        == second
        == load_cross_host_final_completion(harness.local.final_completion_path)
    )
    assert provider.power_calls == 1
    assert first.remote_sqlite_writes_after_closure is False
    assert first.whole_instance_billed_gpu_seconds > 0
    assert harness.remote.database_path.read_bytes() == before
    assert harness.local.archive_final_root.is_dir()
    assert first.pre_poweroff_composite.reopen(label="test composite").is_file()
    for path in Path(harness.local.local_finalization_root).rglob("*"):
        if path.is_file():
            assert b"memory-only-token" not in path.read_bytes()


def test_crash_restart_never_reissues_power_off_after_status_list_disagreement(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    transport = _MirrorTransport(harness)
    provider = _FakeProvider(disagree=True)
    supervisor = _supervisor(harness, transport, provider)
    with pytest.raises(Exception, match="converge"):
        supervisor.run()
    assert provider.power_calls == 1
    assert harness.local.power_request_journal_path.is_file()
    assert not harness.local.power_transition_path.exists()
    harness.clock.value += 301 * 1_000_000_000
    provider.disagree = False
    assert supervisor.run().status == "COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL"
    assert provider.power_calls == 1


def test_writer_probe_and_incomplete_archive_block_power_off(tmp_path: Path) -> None:
    writer_harness = _harness(tmp_path / "writer")
    writer_transport = _MirrorTransport(writer_harness, writer=True)
    writer_provider = _FakeProvider()
    with pytest.raises(FormalCrossHostFinalizationError, match="probe"):
        _supervisor(writer_harness, writer_transport, writer_provider).run()
    assert writer_provider.power_calls == 0

    archive_harness = _harness(tmp_path / "archive")
    archive_transport = _MirrorTransport(archive_harness)
    archive_transport.seal_remote()
    member = next(
        path
        for path in Path(archive_harness.remote.payload_root, "objects").rglob("*")
        if path.is_file()
    )
    member.unlink()
    archive_provider = _FakeProvider()
    with pytest.raises(Exception, match="manifest|archive"):
        _supervisor(archive_harness, archive_transport, archive_provider).run()
    assert archive_provider.power_calls == 0

    running_harness = _harness(tmp_path / "running")
    with ExperimentOperatorStore(running_harness.remote.database_path) as store:
        spec = CellAttemptSpec(
            cell_id="still-running",
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
            output_directory=str(
                (running_harness.remote.run_root / "still-running").resolve()
            ),
        )
        store.materialize_attempt(spec)
        store.mark_running_before_spawn(
            spec.cell_id,
            spec.attempt,
            assigned_gpu_uuids=("GPU-fixture",),
        )
    running_provider = _FakeProvider()
    with pytest.raises(FormalCrossHostFinalizationError, match="RUNNING"):
        _supervisor(
            running_harness,
            _MirrorTransport(running_harness),
            running_provider,
        ).run()
    assert running_provider.power_calls == 0


def test_billing_identity_failure_is_terminal_but_does_not_repower(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    transport = _MirrorTransport(harness)
    provider = _FakeProvider(changed_start=True)
    supervisor = _supervisor(harness, transport, provider)
    with pytest.raises(FormalCrossHostFinalizationError, match="billing interval"):
        supervisor.run()
    with pytest.raises(FormalCrossHostFinalizationError, match="billing interval"):
        supervisor.run()
    assert provider.power_calls == 1
    assert harness.local.power_transition_path.is_file()


def test_crash_between_mutation_and_response_is_never_reissued(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    transport = _MirrorTransport(harness)
    provider = _IndeterminateProvider()
    supervisor = _supervisor(harness, transport, provider)
    with pytest.raises(OSError, match="connection lost"):
        supervisor.run()
    assert harness.local.power_intent_journal_path.is_file()
    assert not harness.local.power_request_journal_path.exists()
    with pytest.raises(AutoDlProviderRuntimeError, match="indeterminate"):
        supervisor.run()
    assert provider.power_calls == 1


def test_production_cli_exposes_only_path_parameters() -> None:
    parser = _parser()
    assert parser.parse_args(["run", "--config", "/tmp/config.json"]).config == (
        "/tmp/config.json"
    )
    probe = parser.parse_args(
        [
            "remote-post-archive-probe",
            "--config",
            "/tmp/remote.json",
            "--output",
            "/tmp/probe.json",
        ]
    )
    assert set(vars(probe)) == {"command", "config", "output"}
