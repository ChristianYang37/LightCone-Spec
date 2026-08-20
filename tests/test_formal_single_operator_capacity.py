from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import lightcone_spec.experiments.formal_single_operator_capacity as capacity_module
from lightcone_spec.doctor import (
    doctor_report,
    revalidate_trusted_single_operator_doctor_report,
)
from lightcone_spec.experiments.formal_single_operator_capacity import (
    TRUSTED_SINGLE_OPERATOR_CAPACITY_PROTOCOL_SHA256,
    TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES,
    TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES,
    TrustedSingleOperatorAutomaticRetryDisabled,
    TrustedSingleOperatorCapacityBlocked,
    TrustedSingleOperatorCapacityDecision,
    TrustedSingleOperatorStageCapacityAuthority,
    load_trusted_single_operator_stage_capacity_authority,
    publish_trusted_single_operator_stage_capacity_authority,
    require_trusted_single_operator_operator_wave_capacity,
    require_trusted_single_operator_ordinary_capacity,
    require_trusted_single_operator_resident_group_capacity,
    require_trusted_single_operator_restart_capacity,
    require_trusted_single_operator_retry_capacity,
    trusted_single_operator_resident_group_high_water_bytes,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.orchestration.experiment_operator import (
    AuxiliaryJobSpec,
    AuxiliaryPhysicalGroupSpec,
    CellAttemptSpec,
    ExperimentOperatorStore,
    QueuedCommandSpec,
    SpawnedProcess,
    StagePlanEntry,
)
from lightcone_spec.orchestration.formal_serving_session_group_production import (
    formal_serving_session_group_shared_evidence_bound_bytes,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_FREE_41_POINT_4_GB = 41_400_000_000


def _publish_fixture_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    free_bytes: int = _FREE_41_POINT_4_GB,
    repository_root: Path | None = None,
    doctor_path: Path | None = None,
) -> tuple[Path, TrustedSingleOperatorStageCapacityAuthority, Path]:
    repository = (
        (tmp_path / "repository").resolve()
        if repository_root is None
        else repository_root.resolve()
    )
    if repository_root is None:
        repository.mkdir()
    run_root = (tmp_path / "run").resolve()
    run_root.mkdir()
    proofs = (tmp_path / "proofs").resolve()
    proofs.mkdir()
    path_spec_path = (tmp_path / "v03-content-path-spec.json").resolve()
    publish_canonical_json_no_replace(
        path_spec_path,
        {
            "schema_version": 1,
            "kind": "capacity_test_v03_content_path_spec",
        },
    )
    path_spec_binding = CanonicalJsonProofBinding.bind(path_spec_path)
    pending = SimpleNamespace(
        source_snapshot=SimpleNamespace(repository_root=str(repository)),
        model_members=(),
        locked_workloads=(),
        burstgpt_release=SimpleNamespace(assets=()),
        e0_task_native_descriptors=(),
        runtime_binding_status="PENDING_REMOTE_BINDING",
        runtime_observations=None,
        semantic_sha256="a" * 64,
    )
    inventory_path = (tmp_path / "inventory.json").resolve()
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-00000000-0000-0000-0000-00000000000{index}",
            host_id="gpu-host",
            model="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            memory_bytes=97_887 * 1024**2,
            compute_capability=(12, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root="root-0",
            numa_node=0,
            interconnects=("PCIe",),
            peer_access_class="peer-enabled",
            clock_policy="locked",
            power_limit_watts=600.0,
            thermal_limit_celsius=85.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=("pair",),
        )
        for index in range(2)
    )
    inventory = GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id="pair",
                host_id="gpu-host",
                gpu_uuids=tuple(row.uuid for row in devices),
                fabric="PCIe",
                bandwidth_class="test",
            ),
        ),
        source_receipt_sha256="f" * 64,
    )
    publish_canonical_json_no_replace(inventory_path, inventory.to_dict())
    path_spec = SimpleNamespace(
        repository_root=str(repository),
        doctor_path=str(
            (tmp_path / "future-doctor.json").resolve()
            if doctor_path is None
            else doctor_path.resolve()
        ),
        inventory_path=str(inventory_path),
    )

    def deep_content(_path: str | Path):
        return path_spec_binding, path_spec, pending

    monkeypatch.setattr(capacity_module, "_deep_content_from_path_spec", deep_content)
    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_single_operator_content."
        "load_trusted_single_operator_content_path_spec",
        lambda _path: path_spec,
    )
    monkeypatch.setattr(capacity_module, "_free_bytes", lambda _path: free_bytes)
    authority_path = (proofs / "stage-capacity.json").resolve()
    authority = publish_trusted_single_operator_stage_capacity_authority(
        content_path_spec_path=path_spec_path,
        run_root_path=run_root,
        output_path=authority_path,
    )
    return authority_path, authority, run_root


def _initial_decision(
    authority: TrustedSingleOperatorStageCapacityAuthority,
) -> TrustedSingleOperatorCapacityDecision:
    return TrustedSingleOperatorCapacityDecision(
        stage="preflight",
        wave_kind="ordinary",
        physical_member_count=1,
        observed_free_bytes=authority.captured_free_bytes,
        current_wave_high_water_bytes=authority.current_wave_high_water_bytes,
        running_wave_high_water_bytes=0,
        retry_reserve_bytes=authority.retry_reserve_bytes,
        safety_margin_bytes=authority.safety_margin_bytes,
        required_free_bytes=authority.required_free_bytes,
        status=authority.status,
        reason_code=(
            "trusted_single_operator_wave_capacity_available"
            if authority.status == "AVAILABLE"
            else "trusted_single_operator_wave_capacity_insufficient"
        ),
        authority_sha256=authority.sha256,
    )


def _bypass_deep_revalidation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    authority_path: Path,
    authority: TrustedSingleOperatorStageCapacityAuthority,
) -> None:
    binding = CanonicalJsonProofBinding.bind(authority_path)
    initial = _initial_decision(authority)

    def revalidate(_path: str | Path, **_kwargs):
        return binding, authority, initial

    monkeypatch.setattr(
        capacity_module,
        "revalidate_trusted_single_operator_stage_capacity_authority",
        revalidate,
    )


def _command(run_root: Path, index: int) -> QueuedCommandSpec:
    evidence = (run_root / f"capacity-command-{index}").resolve()
    return QueuedCommandSpec(
        cell_id=f"preflight:capacity-{index:02d}",
        attempt=1,
        argv=("python3", "-c", "pass"),
        launch_compatibility_key="capacity:test:tp1",
        required_gpu_count=1,
        timing_class="HEADLINE",
        predicted_high_water_bytes=TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES,
        monitored_path=str(run_root),
        log_path=str(evidence.with_suffix(".command.log")),
        expected_terminal_path=str(evidence.with_suffix(".terminal.json")),
        expected_junit_path=str(evidence.with_suffix(".junit.xml")),
        expected_raw_log_path=str(evidence.with_suffix(".raw.jsonl")),
        atomic_pointer_path=str(evidence.with_suffix(".pointer.json")),
        child_exit_receipt_path=str(evidence.with_suffix(".exit.json")),
    )


def test_path_only_capacity_producer_and_41_point_4_gb_ordinary_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = inspect.signature(
        publish_trusted_single_operator_stage_capacity_authority
    ).parameters
    assert tuple(parameters) == (
        "content_path_spec_path",
        "run_root_path",
        "output_path",
    )
    assert all(
        row.kind is inspect.Parameter.KEYWORD_ONLY for row in parameters.values()
    )

    authority_path, authority, _run_root = _publish_fixture_authority(
        tmp_path,
        monkeypatch,
    )
    assert authority.status == "AVAILABLE"
    assert authority.required_free_bytes == 31 * 1024**3
    assert authority.captured_free_bytes == _FREE_41_POINT_4_GB
    assert authority.retry_reserve_bytes == 0
    assert authority.formal_measured_authorization is False
    assert authority.protocol_sha256 == TRUSTED_SINGLE_OPERATOR_CAPACITY_PROTOCOL_SHA256

    _proof, reopened, initial = (
        capacity_module.revalidate_trusted_single_operator_stage_capacity_authority(
            authority_path
        )
    )
    assert reopened == authority
    assert initial.status == "AVAILABLE"
    ordinary = require_trusted_single_operator_ordinary_capacity(
        authority_path,
        stage="preflight",
    )
    assert ordinary.status == "AVAILABLE"
    assert ordinary.observed_free_bytes == _FREE_41_POINT_4_GB
    assert ordinary.required_free_bytes == 31 * 1024**3


def test_capacity_fails_closed_below_required_and_for_unknown_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path, authority, _run_root = _publish_fixture_authority(
        tmp_path,
        monkeypatch,
    )
    _bypass_deep_revalidation(
        monkeypatch,
        authority_path=authority_path,
        authority=authority,
    )
    monkeypatch.setattr(
        capacity_module,
        "_free_bytes",
        lambda _path: authority.required_free_bytes - 1,
    )
    with pytest.raises(TrustedSingleOperatorCapacityBlocked) as blocked:
        require_trusted_single_operator_ordinary_capacity(
            authority_path,
            stage="preflight",
        )
    assert blocked.value.decision.status == "BLOCKED"
    assert (
        blocked.value.decision.observed_free_bytes == authority.required_free_bytes - 1
    )

    with pytest.raises(ValueError, match="registered DAG node"):
        require_trusted_single_operator_ordinary_capacity(
            authority_path,
            stage="caller-invented-stage",
        )


def test_resident_group_sums_every_2_and_32_member_before_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path, authority, run_root = _publish_fixture_authority(
        tmp_path,
        monkeypatch,
    )
    _bypass_deep_revalidation(
        monkeypatch,
        authority_path=authority_path,
        authority=authority,
    )
    commands_32 = tuple(_command(run_root, index) for index in range(32))
    for count in (2, 32):
        commands = commands_32[:count]
        assert trusted_single_operator_resident_group_high_water_bytes(commands) == (
            count * TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES
            + formal_serving_session_group_shared_evidence_bound_bytes(count)
        )

    commands_2 = commands_32[:2]
    group_high_water = trusted_single_operator_resident_group_high_water_bytes(
        commands_2
    )
    required = group_high_water + TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES
    monkeypatch.setattr(capacity_module, "_free_bytes", lambda _path: required)
    admitted = require_trusted_single_operator_resident_group_capacity(
        authority_path,
        stage="e3a",
        commands=commands_2,
    )
    assert admitted.status == "AVAILABLE"
    assert admitted.current_wave_high_water_bytes == group_high_water

    monkeypatch.setattr(capacity_module, "_free_bytes", lambda _path: required - 1)
    with pytest.raises(TrustedSingleOperatorCapacityBlocked) as blocked:
        require_trusted_single_operator_resident_group_capacity(
            authority_path,
            stage="e3a",
            commands=commands_2,
        )
    assert blocked.value.decision.required_free_bytes == required


def test_restart_re_reads_free_space_without_double_counting_running_wave(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path, authority, run_root = _publish_fixture_authority(
        tmp_path,
        monkeypatch,
    )
    _bypass_deep_revalidation(
        monkeypatch,
        authority_path=authority_path,
        authority=authority,
    )
    running = (_command(run_root, 0),)
    observed = iter((_FREE_41_POINT_4_GB, authority.safety_margin_bytes - 1))
    monkeypatch.setattr(capacity_module, "_free_bytes", lambda _path: next(observed))

    first = require_trusted_single_operator_restart_capacity(
        authority_path,
        stage="preflight",
        running_commands=running,
    )
    assert first.status == "AVAILABLE"
    assert first.current_wave_high_water_bytes == 0
    assert first.running_wave_high_water_bytes == 0
    assert first.required_free_bytes == authority.safety_margin_bytes
    blocked = require_trusted_single_operator_restart_capacity(
        authority_path,
        stage="preflight",
        running_commands=running,
    )
    assert blocked.status == "BLOCKED"
    assert blocked.required_free_bytes == authority.safety_margin_bytes


def test_running_auxiliary_is_adopted_on_restart_and_charged_to_new_wave(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path, authority, run_root = _publish_fixture_authority(
        tmp_path,
        monkeypatch,
    )
    _bypass_deep_revalidation(
        monkeypatch,
        authority_path=authority_path,
        authority=authority,
    )
    job = AuxiliaryJobSpec(
        job_id="e6-capacity-job",
        attempt=1,
        adoption_key="e6:capacity",
        scientific_axes={"task": "capacity"},
        identity={
            "source_sha256": "1" * 64,
            "patch_sha256": "2" * 64,
            "registry_sha256": "3" * 64,
        },
        command_sha256="4" * 64,
        output_directory=str((run_root / "e6-job").resolve()),
    )
    group = AuxiliaryPhysicalGroupSpec(
        group_id="e6-capacity-group",
        attempt=1,
        node="e6_pilot",
        source_kind="e6_interface_fit",
        jobs=(job,),
        assigned_gpu_uuids=("GPU-0", "GPU-1"),
        launch_command_sha256="5" * 64,
        output_directory=str((run_root / "e6-group").resolve()),
    )
    with ExperimentOperatorStore(
        (tmp_path / "auxiliary-capacity.sqlite3").resolve(),
        run_id="auxiliary-capacity",
    ) as store:
        store.initialize_stage_plan(
            (StagePlanEntry("e6_pilot", 0, "E6", "pilot", "1", 1),)
        )
        assert store.register_controller_auxiliary_group(group)
        store.start_controller_auxiliary_group_with_launcher(
            group,
            launcher=lambda: SpawnedProcess(901, 901),
        )

        monkeypatch.setattr(
            capacity_module,
            "_free_bytes",
            lambda _path: authority.safety_margin_bytes,
        )
        restart = require_trusted_single_operator_restart_capacity(
            authority_path,
            stage="e6_pilot",
            running_commands=(),
            store=store,
        )
        assert restart.status == "AVAILABLE"
        assert restart.physical_member_count == 1
        assert restart.current_wave_high_water_bytes == 0
        assert restart.running_wave_high_water_bytes == 0

        next_required = (
            2 * TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES
            + TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES
        )
        monkeypatch.setattr(
            capacity_module,
            "_free_bytes",
            lambda _path: next_required,
        )
        next_wave = require_trusted_single_operator_operator_wave_capacity(
            authority_path,
            stage="e0_tuning",
            store=store,
        )
        assert next_wave.required_free_bytes == next_required
        assert (
            next_wave.running_wave_high_water_bytes
            == TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES
        )


def test_zero_reserve_retry_is_disabled_before_probe_or_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path, authority, run_root = _publish_fixture_authority(
        tmp_path,
        monkeypatch,
    )
    _bypass_deep_revalidation(
        monkeypatch,
        authority_path=authority_path,
        authority=authority,
    )
    command = _command(run_root, 0)
    specification = CellAttemptSpec(
        cell_id=command.cell_id,
        attempt=1,
        stage="preflight",
        phase="final",
        block="preflight",
        seed=17,
        scientific_axes={"task": "capacity-retry"},
        identity={
            "source_sha256": "1" * 64,
            "patch_sha256": "2" * 64,
            "registry_sha256": "3" * 64,
        },
        command_sha256=command.command_sha256,
        scientific_command_sha256="4" * 64,
        output_directory=str((run_root / "failed-output").resolve()),
    )
    free_probes = 0

    def forbidden_free_probe(_path: Path) -> int:
        nonlocal free_probes
        free_probes += 1
        raise AssertionError("free space was probed before archive proof completed")

    monkeypatch.setattr(capacity_module, "_free_bytes", forbidden_free_probe)
    store = ExperimentOperatorStore(
        (tmp_path / "retry-operator.sqlite3").resolve(),
        run_id="capacity-retry-test",
    )
    try:
        store.initialize_stage_plan(
            (StagePlanEntry("preflight", 0, "preflight", "final", "1", 1),)
        )
        store.materialize_attempt(specification)
        store.finish_attempt(
            command.cell_id,
            command.attempt,
            status="FAILED",
            exit_code=70,
            failure_code="INFRASTRUCTURE:fixture",
            retry_decision="RETRY_INFRASTRUCTURE_AUTOMATIC",
            included_in_analysis=False,
            exclusion_reason="fixture infrastructure failure",
        )
        with pytest.raises(
            TrustedSingleOperatorAutomaticRetryDisabled,
            match="disables automatic retries",
        ):
            require_trusted_single_operator_retry_capacity(
                authority_path,
                store=store,
                previous_command=command,
                stage="preflight",
            )
        assert free_probes == 0
        with pytest.raises(KeyError):
            store.attempt(command.cell_id, 2)
    finally:
        store.close()


def test_authority_tamper_and_changed_filesystem_root_identity_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path, authority, run_root = _publish_fixture_authority(
        tmp_path,
        monkeypatch,
    )
    assert load_trusted_single_operator_stage_capacity_authority(authority_path)[1] == (
        authority
    )

    tampered_path = (authority_path.parent / "tampered-capacity.json").resolve()
    tampered = authority.to_dict()
    tampered["captured_free_bytes"] = authority.captured_free_bytes + 1
    publish_canonical_json_no_replace(tampered_path, tampered)
    with pytest.raises(ValueError, match="digest differs"):
        load_trusted_single_operator_stage_capacity_authority(tampered_path)

    retained_root = (tmp_path / "retained-old-run-root").resolve()
    run_root.rename(retained_root)
    run_root.mkdir()
    with pytest.raises(ValueError, match="directory identity changed"):
        load_trusted_single_operator_stage_capacity_authority(authority_path)


def test_trivial_pass_doctor_is_not_a_trusted_capacity_authority(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "caller-authored-doctor.json").resolve()
    publish_canonical_json_no_replace(
        path,
        {
            "schema_version": 2,
            "status": "PASS",
            "readiness": {
                "status": "PASS",
                "pass_count": 1,
                "fail_count": 0,
                "unknown_count": 0,
            },
            "checks": {"fixture": {"status": "PASS"}},
        },
    )
    with pytest.raises(ValueError, match="lacks trusted single-operator capacity"):
        revalidate_trusted_single_operator_doctor_report(path)


def test_source_produced_doctor_deep_replays_capacity_and_current_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_doctor_industrial import ROOT, _passing_facts

    doctor_path = (tmp_path / "trusted-doctor.json").resolve()
    authority_path, _authority, _run_root = _publish_fixture_authority(
        tmp_path,
        monkeypatch,
        repository_root=ROOT,
        doctor_path=doctor_path,
    )
    sglang = (tmp_path / "patched-sglang").resolve()
    facts = _passing_facts(ROOT, sglang)
    facts["disk"].update(
        {
            "used_bytes": 258_600_000_000,
            "free_bytes": _FREE_41_POINT_4_GB,
        }
    )
    monkeypatch.setattr("lightcone_spec.doctor._collect_facts", lambda *_args: facts)
    report = doctor_report(
        ROOT,
        sglang,
        trusted_single_operator_capacity_path=authority_path,
    )
    assert report["status"] == "PASS"
    publish_canonical_json_no_replace(doctor_path, report)

    binding = revalidate_trusted_single_operator_doctor_report(doctor_path)
    assert binding.absolute_path == str(doctor_path)
    assert (
        binding.reopen()["stage_capacity"]["authority_sha256"]
        == (report["stage_capacity"]["authority_sha256"])
    )

    monkeypatch.setattr(
        capacity_module,
        "_free_bytes",
        lambda _path: TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES - 1,
    )
    with pytest.raises(TrustedSingleOperatorCapacityBlocked):
        revalidate_trusted_single_operator_doctor_report(doctor_path)
    restart_binding = revalidate_trusted_single_operator_doctor_report(
        doctor_path,
        require_capacity_available=False,
    )
    assert restart_binding.absolute_path == str(doctor_path)

    monkeypatch.setattr(
        capacity_module,
        "_free_bytes",
        lambda _path: _FREE_41_POINT_4_GB,
    )
    facts["gpu"]["inventory"]["devices"][0]["uuid"] = "GPU-replaced"
    with pytest.raises(ValueError, match="source replay identity differs"):
        revalidate_trusted_single_operator_doctor_report(doctor_path)
