from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuDispatchExecutionContext,
    GpuInventory,
    GpuTopologyGroup,
    InterferenceEnvelope,
    InterferenceRule,
    registry_pool_work_item,
)
from lightcone_spec.experiments.interference_authority import (
    CALIBRATED_INTERFERENCE_RAW_AUTHORITY_REQUIRED_REASON,
    INTERFERENCE_CALIBRATION_BOOTSTRAP_PROTOCOL_SHA256,
    INTERFERENCE_CALIBRATION_REDUCER_PROTOCOL_SHA256,
    INTERFERENCE_ITL_EVIDENCE_INCOMPLETE_REASON,
    TRUSTED_INTERFERENCE_ATTESTER_UNAVAILABLE_REASON,
    InterferenceCalibrationBlockedError,
    InterferenceCalibrationGroup,
    InterferenceCalibrationManifest,
    InterferenceCalibrationProtocol,
    InterferenceCalibrationRun,
    InterferenceCalibrationSourceAuthority,
    InterferenceHardwareEnvelope,
    InterferenceRawObservation,
    RawInterferenceJsonBinding,
    diagnose_interference_calibration,
    materialize_interference_calibration_bootstrap_authority,
    require_calibrated_interference_execution_authority,
    require_release_interference_attester,
)
from lightcone_spec.experiments.registry import (
    WorkloadClass,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.stage_activation import (
    materialize_registry_stage_activation,
)
from lightcone_spec.orchestration.execution_bundle import (
    BoundJsonSource,
    _replay_interference_bootstrap_authority,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_bound(path: Path, value: object) -> Path:
    body = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    path.write_text(body, encoding="utf-8")
    Path(f"{path}.sha256").write_text(f"{content_sha256(value)}\n", encoding="ascii")
    return path.resolve()


def _inventory_receipt(host_id: str, count: int) -> dict[str, object]:
    content: dict[str, object] = {
        "schema_version": 1,
        "kind": "gpu_inventory_probe_receipt",
        "challenge_nonce_sha256": _sha("challenge"),
        "host_id": host_id,
        "hostname": "calibration-host",
        "machine_id_sha256": _sha("machine"),
        "commands": {
            "gpu": {"argv": ["nvidia-smi"], "stdout": "bound"},
            "processes": {"argv": ["nvidia-smi"], "stdout": ""},
            "topology": {"argv": ["nvidia-smi"], "stdout": "bound"},
        },
        "parsed_topology": {"gpu_rows": [f"GPU{i}" for i in range(count)]},
        "pci_locality": [
            {
                "index": index,
                "uuid": f"GPU-{index:02d}",
                "pci_bus_id": f"0000:{index + 1:02x}:00.0",
                "pci_root": "0000:00:00.0",
                "numa_node": index % 2,
            }
            for index in range(count)
        ],
    }
    return {**content, "receipt_sha256": content_sha256(content)}


def _inventory(count: int) -> tuple[GpuInventory, dict[str, object]]:
    host_id = _sha("host")
    receipt = _inventory_receipt(host_id, count)
    group_id = "same-host-all"
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-{index:02d}",
            host_id=host_id,
            model="NVIDIA-Test",
            memory_bytes=80 * 1024**3,
            compute_capability=(9, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root="0000:00:00.0",
            numa_node=index % 2,
            interconnects=("NV18",),
            peer_access_class="nvswitch",
            clock_policy="locked",
            power_limit_watts=700.0,
            thermal_limit_celsius=85.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=(group_id,),
        )
        for index in range(count)
    )
    inventory = GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id=group_id,
                host_id=host_id,
                gpu_uuids=tuple(device.uuid for device in devices),
                fabric="NVSwitch",
                bandwidth_class="full",
            ),
        ),
        source_receipt_sha256=str(receipt["receipt_sha256"]),
    )
    return inventory, receipt


def _topology_sha(inventory: GpuInventory, gpu_uuid: str) -> str:
    device = inventory.device(gpu_uuid)
    return content_sha256(
        {
            "schema_version": 1,
            "host_ids": [device.host_id],
            "gpu_uuids": [gpu_uuid],
            "rank_groups": [[gpu_uuid]],
            "devices": [
                {
                    "uuid": gpu_uuid,
                    "hardware_envelope_sha256": device.hardware_envelope_sha256,
                    "pci_root": device.pci_root,
                    "numa_node": device.numa_node,
                    "peer_access_class": device.peer_access_class,
                }
            ],
            "covering_topology_groups": [
                group.to_dict()
                for group in inventory.topology_groups
                if gpu_uuid in group.gpu_uuids
            ],
        }
    )


def _run(
    inventory: GpuInventory,
    *,
    mode: str,
    repetition: int,
    slot: int,
    label: str,
    load: str = "registered-load",
) -> InterferenceCalibrationRun:
    uuid = inventory.devices[slot].uuid
    contention = content_sha256(
        {
            "cpu_cores": 8,
            "numa_nodes": (0, 1),
            "disk_io_class": "none",
            "network_class": "loopback",
        }
    )
    return InterferenceCalibrationRun(
        observation_id=f"{mode}-r{repetition}-s{slot}-{label}",
        mode=mode,
        repetition=repetition,
        slot=slot,
        terminal_authority_sha256=_sha(f"terminal-{mode}-{repetition}-{slot}-{label}"),
        assignment_sha256=_sha(f"assignment-{mode}-{repetition}-{slot}-{label}"),
        cell_id=_sha(f"cell-{mode}-{repetition}-{slot}-{label}"),
        execution_plan_sha256=_sha(f"plan-{mode}-{repetition}-{slot}-{label}"),
        budget_sha256=_sha(f"budget-{mode}-{repetition}-{slot}-{label}"),
        load_plan_sha256=_sha(f"load-{label}"),
        run_nonce_sha256=_sha(f"nonce-{mode}-{repetition}-{slot}-{label}"),
        gpu_uuids=(uuid,),
        rank_groups=((uuid,),),
        topology_sha256=_topology_sha(inventory, uuid),
        hardware_envelope_sha256=inventory.device(uuid).hardware_envelope_sha256,
        workload_class=WorkloadClass.HEADLINE,
        co_run_signature="headline-static-c1",
        gang_shape="tp1_dp1",
        load_thermal_power_envelope=load,
        cpu_cores=8,
        numa_nodes=(0, 1),
        ram_bytes=32 * 1024**3,
        disk_io_class="none",
        network_class="loopback",
        contention_class=contention,
        data_partition="interference_calibration_only",
    )


def _group(
    inventory: GpuInventory,
    cardinality: int,
    *,
    label: str = "g",
    repetitions: int = 2,
) -> InterferenceCalibrationGroup:
    return InterferenceCalibrationGroup(
        group_id=f"group-{cardinality}-{label}",
        simultaneous_jobs=cardinality,
        isolated=tuple(
            _run(
                inventory,
                mode="isolated",
                repetition=repetition,
                slot=slot,
                label=label,
            )
            for repetition in range(repetitions)
            for slot in range(cardinality)
        ),
        concurrent=tuple(
            _run(
                inventory,
                mode="concurrent",
                repetition=repetition,
                slot=slot,
                label=label,
            )
            for repetition in range(repetitions)
            for slot in range(cardinality)
        ),
    )


def _protocol(
    inventory: GpuInventory, hardware: InterferenceHardwareEnvelope
) -> InterferenceCalibrationProtocol:
    return InterferenceCalibrationProtocol(
        schema_version=2,
        kind="interference_calibration_protocol",
        reducer_protocol_sha256=INTERFERENCE_CALIBRATION_REDUCER_PROTOCOL_SHA256,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=hardware.sha256,
        data_partition="interference_calibration_only",
        confirmation_data_visible=False,
        acceptance_status="REGISTERED",
        minimum_isolated_repetitions=2,
        minimum_concurrent_repetitions=2,
        maximum_absolute_relative_difference=0.01,
        confidence=0.95,
        interval_method="paired_bca_mean_log_ratio_v1",
        bootstrap_repetitions=10_000,
        bootstrap_seed=0,
    )


def _manifest(
    inventory: GpuInventory,
    hardware: InterferenceHardwareEnvelope,
    protocol: InterferenceCalibrationProtocol,
    groups: tuple[InterferenceCalibrationGroup, ...],
) -> InterferenceCalibrationManifest:
    return InterferenceCalibrationManifest(
        schema_version=1,
        kind="interference_calibration_manifest",
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        hardware_envelope_sha256=hardware.sha256,
        protocol_sha256=protocol.sha256,
        data_partition="interference_calibration_only",
        confirmation_data_visible=False,
        groups=groups,
    )


def _source_authority(
    tmp_path: Path,
    *,
    count: int,
    cardinality: int,
) -> tuple[InterferenceCalibrationSourceAuthority, InterferenceCalibrationGroup]:
    inventory, receipt = _inventory(count)
    hardware = InterferenceHardwareEnvelope.from_inventory(inventory)
    protocol = _protocol(inventory, hardware)
    group = _group(inventory, cardinality)
    manifest = _manifest(inventory, hardware, protocol, (group,))
    paths = {
        "inventory": _write_bound(tmp_path / "inventory.json", inventory.to_dict()),
        "inventory_source_receipt": _write_bound(
            tmp_path / "inventory-receipt.json", receipt
        ),
        "hardware_envelope": _write_bound(
            tmp_path / "hardware.json", hardware.to_dict()
        ),
        "protocol": _write_bound(tmp_path / "protocol.json", protocol.to_dict()),
        "manifest": _write_bound(tmp_path / "manifest.json", manifest.to_dict()),
    }
    return InterferenceCalibrationSourceAuthority.from_paths(**paths), group


@pytest.mark.parametrize("count", (1, 2, 4, 8, 16))
def test_hardware_envelope_covers_arbitrary_same_host_inventory(count: int) -> None:
    inventory, _ = _inventory(count)
    envelope = InterferenceHardwareEnvelope.from_inventory(inventory)

    assert len(envelope.devices) == count
    assert envelope.inventory_sha256 == inventory.sha256
    assert InterferenceHardwareEnvelope.from_dict(envelope.to_dict()) == envelope


def test_bootstrap_authority_only_pairs_registered_concurrent_preflight_cells() -> None:
    registry = build_industrial_registry()
    inventory, _ = _inventory(2)
    activation = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
    )

    authority = materialize_interference_calibration_bootstrap_authority(
        registry,
        inventory,
        activation,
    )
    cells = {
        cell.cell_id: cell
        for cell in registry.cells_for("preflight")
        if cell.identity.task == "simultaneous_single_gpu_interference"
    }
    concurrent = tuple(
        registry_pool_work_item(cell, estimated_duration_seconds=1.0)
        for cell in cells.values()
        if cell.identity.block == 0
        and str(cell.identity.variant).startswith("concurrent_slot_")
    )
    isolated = tuple(
        registry_pool_work_item(cell, estimated_duration_seconds=1.0)
        for cell in cells.values()
        if cell.identity.block == 0
        and str(cell.identity.variant).startswith("isolated_slot_")
    )
    from lightcone_spec.experiments.gpu_pool import GpuAssignment

    concurrent_assignments = tuple(
        GpuAssignment(
            work_item=item,
            gpu_uuids=(inventory.devices[index].uuid,),
            rank_groups=((inventory.devices[index].uuid,),),
            ports=(24_000 + index,),
        )
        for index, item in enumerate(concurrent)
    )
    isolated_assignments = tuple(
        GpuAssignment(
            work_item=item,
            gpu_uuids=(inventory.devices[index].uuid,),
            rank_groups=((inventory.devices[index].uuid,),),
            ports=(24_100 + index,),
        )
        for index, item in enumerate(isolated)
    )

    assert (
        authority.protocol_sha256 == INTERFERENCE_CALIBRATION_BOOTSTRAP_PROTOCOL_SHA256
    )
    assert set(authority.calibration_cell_ids) == set(cells)
    assert authority.bootstrap_envelope.permits(
        concurrent_assignments,
        inventory=inventory,
    )
    assert not authority.bootstrap_envelope.permits(
        isolated_assignments,
        inventory=inventory,
    )


def test_bootstrap_authority_does_not_unlock_headline_or_larger_cardinality() -> None:
    registry = build_industrial_registry()
    inventory, _ = _inventory(4)
    activation = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
    )
    authority = materialize_interference_calibration_bootstrap_authority(
        registry,
        inventory,
        activation,
    )

    assert {rule.simultaneous_jobs for rule in authority.bootstrap_envelope.rules} == {
        2
    }
    assert {rule.workload_class for rule in authority.bootstrap_envelope.rules} == {
        WorkloadClass.CORRECTNESS
    }
    with pytest.raises(ValueError, match="another release protocol"):
        replace(authority, protocol_sha256=_sha("caller-policy"))


def test_bootstrap_receipt_is_canonical_and_binds_the_exact_envelope() -> None:
    registry = build_industrial_registry()
    inventory, _ = _inventory(2)
    activation = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
    )
    authority = materialize_interference_calibration_bootstrap_authority(
        registry,
        inventory,
        activation,
    )

    assert authority.source_receipt == {
        "schema_version": 1,
        "kind": "interference_calibration_bootstrap_receipt",
        "registry_sha256": registry.sha256,
        "inventory_sha256": inventory.sha256,
        "activation_sha256": activation.sha256,
        "protocol_sha256": INTERFERENCE_CALIBRATION_BOOTSTRAP_PROTOCOL_SHA256,
        "receipt_sha256": authority.bootstrap_envelope.source_receipt_sha256,
    }
    forged_envelope = replace(
        authority.bootstrap_envelope,
        source_receipt_sha256=_sha("forged-bootstrap-receipt"),
    )
    with pytest.raises(RuntimeError, match="receipt identity drifted"):
        _ = replace(authority, bootstrap_envelope=forged_envelope).source_receipt


def test_bootstrap_receipt_is_reduced_from_raw_inputs_not_self_authorized(
    tmp_path: Path,
) -> None:
    registry = build_industrial_registry()
    inventory, _ = _inventory(2)
    activation = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
    )
    authority = materialize_interference_calibration_bootstrap_authority(
        registry,
        inventory,
        activation,
    )
    receipt_path = _write_bound(tmp_path / "bootstrap.json", authority.source_receipt)
    source = BoundJsonSource.bind(
        receipt_path,
        semantic_sha256=str(authority.source_receipt["receipt_sha256"]),
    )

    assert (
        _replay_interference_bootstrap_authority(
            registry=registry,
            inventory=inventory,
            activation=activation,
            envelope=authority.bootstrap_envelope,
            receipt_source=source,
        )
        == authority
    )

    forged_content = {
        key: value
        for key, value in authority.source_receipt.items()
        if key != "receipt_sha256"
    }
    forged_content["registry_sha256"] = _sha("foreign-registry")
    forged_receipt = {
        **forged_content,
        "receipt_sha256": content_sha256(forged_content),
    }
    forged_path = _write_bound(tmp_path / "forged-bootstrap.json", forged_receipt)
    forged_source = BoundJsonSource.bind(
        forged_path,
        semantic_sha256=str(forged_receipt["receipt_sha256"]),
    )
    with pytest.raises(ValueError, match="differs from its raw receipt"):
        _replay_interference_bootstrap_authority(
            registry=registry,
            inventory=inventory,
            activation=activation,
            envelope=authority.bootstrap_envelope,
            receipt_source=forged_source,
        )


def test_formal_context_replays_bootstrap_before_budget_authority() -> None:
    registry = build_industrial_registry()
    inventory, _ = _inventory(2)
    activation = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
    )
    authority = materialize_interference_calibration_bootstrap_authority(
        registry,
        inventory,
        activation,
    )

    with pytest.raises(TypeError, match="exact BudgetPlan"):
        GpuDispatchExecutionContext(
            registry=registry,
            inventory=inventory,
            interference_envelope=authority.bootstrap_envelope,
            budgets=(),
            activation_artifact=activation,
            budget_plan=object(),  # type: ignore[arg-type]
            budget_materialization_authority=object(),  # type: ignore[arg-type]
            interference_calibration_bootstrap_authority=authority,
        )

    forged = replace(authority, inventory_sha256=_sha("foreign-inventory"))
    with pytest.raises(ValueError, match="registry/inventory/activation"):
        GpuDispatchExecutionContext(
            registry=registry,
            inventory=inventory,
            interference_envelope=authority.bootstrap_envelope,
            budgets=(),
            activation_artifact=activation,
            budget_plan=object(),  # type: ignore[arg-type]
            budget_materialization_authority=object(),  # type: ignore[arg-type]
            interference_calibration_bootstrap_authority=forged,
        )


@pytest.mark.parametrize("count", (2, 4, 8, 16))
def test_source_authority_audits_exact_cardinality(tmp_path: Path, count: int) -> None:
    authority, group = _source_authority(tmp_path, count=count, cardinality=count)

    audit = authority.audit()

    assert audit.manifest.groups == (group,)
    assert audit.manifest.groups[0].simultaneous_jobs == count


def test_one_gpu_inventory_cannot_claim_two_way_calibration(tmp_path: Path) -> None:
    inventory, receipt = _inventory(1)
    hardware = InterferenceHardwareEnvelope.from_inventory(inventory)
    protocol = _protocol(inventory, hardware)
    two_gpu_inventory, _ = _inventory(2)
    manifest = _manifest(
        inventory,
        hardware,
        protocol,
        (_group(two_gpu_inventory, 2, label="foreign"),),
    )
    paths = {
        "inventory": _write_bound(tmp_path / "inventory.json", inventory.to_dict()),
        "inventory_source_receipt": _write_bound(
            tmp_path / "inventory-receipt.json", receipt
        ),
        "hardware_envelope": _write_bound(
            tmp_path / "hardware.json", hardware.to_dict()
        ),
        "protocol": _write_bound(tmp_path / "protocol.json", protocol.to_dict()),
        "manifest": _write_bound(tmp_path / "manifest.json", manifest.to_dict()),
    }

    with pytest.raises(ValueError, match="cardinality exceeds inventory"):
        InterferenceCalibrationSourceAuthority.from_paths(**paths).audit()


def _observation(
    run: InterferenceCalibrationRun,
    *,
    start: int,
    finish: int,
    token_label: str,
    safety: int = 0,
    goodput_ratio: float = 1.0,
    itl_ratio: float | None = 1.0,
) -> InterferenceRawObservation:
    return InterferenceRawObservation(
        observation_id=run.observation_id,
        terminal_authority_sha256=run.terminal_authority_sha256,
        mode=run.mode,
        repetition=run.repetition,
        slot=run.slot,
        started_ns=start,
        finished_ns=finish,
        request_ids=("request-0", "request-1"),
        token_trajectory_sha256=_sha(token_label),
        completed_requests=2,
        output_tokens=32,
        goodput_tps=(100.0 if run.mode == "isolated" else 100.0 * goodput_ratio),
        p99_itl_ms=(
            None
            if itl_ratio is None
            else 10.0
            if run.mode == "isolated"
            else 10.0 * itl_ratio
        ),
        safety_counters=(
            ("exactness_violations", safety),
            ("version_mismatches", 0),
            ("fallbacks", 0),
            ("nonfinite_updates", 0),
            ("oom_events", 0),
            ("retractions", 0),
            ("communicator_failures", 0),
        ),
        hardware_valid=True,
    )


def _observations(
    group: InterferenceCalibrationGroup,
    *,
    goodput_ratio: float = 1.0,
    itl_ratio: float | None = 1.0,
) -> tuple[InterferenceRawObservation, ...]:
    rows = []
    for index, run in enumerate(group.isolated):
        rows.append(
            _observation(
                run,
                start=index * 100,
                finish=(index + 1) * 100,
                token_label=f"r{run.repetition}-s{run.slot}",
                goodput_ratio=goodput_ratio,
                itl_ratio=itl_ratio,
            )
        )
    for run in group.concurrent:
        start = 1_000 + run.repetition * 200 + run.slot * 10
        rows.append(
            _observation(
                run,
                start=start,
                finish=1_100 + run.repetition * 200,
                token_label=f"r{run.repetition}-s{run.slot}",
                goodput_ratio=goodput_ratio,
                itl_ratio=itl_ratio,
            )
        )
    return tuple(rows)


def test_registered_raw_diagnostic_passes_only_exact_paired_equivalence() -> None:
    inventory, _ = _inventory(2)
    group = _group(inventory, 2)
    hardware = InterferenceHardwareEnvelope.from_inventory(inventory)
    observations = _observations(group)

    result = diagnose_interference_calibration(
        group, observations, protocol=_protocol(inventory, hardware)
    )

    assert result.status == "PASS"
    assert result.reason_codes == ()
    assert tuple(row[2] for row in result.goodput_ratios) == (1.0,) * 4
    assert result.goodput_mean_relative_difference == pytest.approx(0.0)
    assert result.simultaneous_jobs == 2


def test_missing_raw_itl_timing_is_unresolved_not_request_latency() -> None:
    inventory, _ = _inventory(2)
    group = _group(inventory, 2)
    hardware = InterferenceHardwareEnvelope.from_inventory(inventory)

    result = diagnose_interference_calibration(
        group,
        _observations(group, itl_ratio=None),
        protocol=_protocol(inventory, hardware),
    )

    assert result.status == "UNRESOLVED"
    assert result.reason_codes == (INTERFERENCE_ITL_EVIDENCE_INCOMPLETE_REASON,)


@pytest.mark.parametrize(
    ("goodput_ratio", "reason"),
    (
        (0.98, "paired_goodput_difference_exceeds_1pct"),
        (0.995, "paired_goodput_95pct_interval_excludes_zero"),
    ),
)
def test_registered_goodput_threshold_and_interval_fail_closed(
    goodput_ratio: float, reason: str
) -> None:
    inventory, _ = _inventory(2)
    group = _group(inventory, 2)
    hardware = InterferenceHardwareEnvelope.from_inventory(inventory)

    result = diagnose_interference_calibration(
        group,
        _observations(group, goodput_ratio=goodput_ratio),
        protocol=_protocol(inventory, hardware),
    )

    assert result.status == "FAIL"
    assert reason in result.reason_codes


def test_raw_diagnostic_hard_failure_stays_fail() -> None:
    inventory, _ = _inventory(2)
    group = _group(inventory, 2)
    hardware = InterferenceHardwareEnvelope.from_inventory(inventory)
    observations = list(_observations(group))
    observations[0] = replace(
        observations[0],
        finished_ns=150,
        safety_counters=(
            ("exactness_violations", 1),
            *observations[0].safety_counters[1:],
        ),
    )
    concurrent_offset = len(group.isolated)
    observations[concurrent_offset] = replace(
        observations[concurrent_offset],
        token_trajectory_sha256=_sha("changed"),
        finished_ns=1_005,
    )

    result = diagnose_interference_calibration(
        group, observations, protocol=_protocol(inventory, hardware)
    )

    assert result.status == "FAIL"
    assert "isolated_interval_overlap" in result.reason_codes
    assert "concurrent_interval_nonoverlap" in result.reason_codes
    assert "nonzero_safety_counter" in result.reason_codes
    assert "terminal_token_trajectory_change" in result.reason_codes


def test_two_way_manifest_does_not_cover_eight_way() -> None:
    inventory, _ = _inventory(8)
    two_way = _group(inventory, 2)
    hardware = InterferenceHardwareEnvelope.from_inventory(inventory)
    protocol = _protocol(inventory, hardware)
    manifest = _manifest(inventory, hardware, protocol, (two_way,))

    assert tuple(group.simultaneous_jobs for group in manifest.groups) == (2,)
    with pytest.raises(ValueError, match="raw observations differ"):
        diagnose_interference_calibration(
            two_way, (), protocol=_protocol(inventory, hardware)
        )


def test_group_rejects_wrong_load_and_cardinality() -> None:
    inventory, _ = _inventory(4)
    group = _group(inventory, 2)
    wrong_load = replace(group.concurrent[1], load_thermal_power_envelope="other")
    concurrent = list(group.concurrent)
    concurrent[1] = wrong_load
    with pytest.raises(ValueError, match="mixes claim/load/topology"):
        InterferenceCalibrationGroup(
            group_id="wrong-load",
            simultaneous_jobs=2,
            isolated=group.isolated,
            concurrent=tuple(concurrent),
        )
    with pytest.raises(ValueError, match="cardinality lacks exact slot coverage"):
        InterferenceCalibrationGroup(
            group_id="wrong-cardinality",
            simultaneous_jobs=4,
            isolated=group.isolated,
            concurrent=group.concurrent,
        )


def test_source_audit_rejects_wrong_topology(tmp_path: Path) -> None:
    inventory, receipt = _inventory(2)
    hardware = InterferenceHardwareEnvelope.from_inventory(inventory)
    protocol = _protocol(inventory, hardware)
    group = _group(inventory, 2)
    wrong = replace(group.concurrent[0], topology_sha256=_sha("wrong-topology"))
    concurrent = list(group.concurrent)
    concurrent[0] = wrong
    group = replace(group, concurrent=tuple(concurrent))
    manifest = _manifest(inventory, hardware, protocol, (group,))
    paths = {
        "inventory": _write_bound(tmp_path / "inventory.json", inventory.to_dict()),
        "inventory_source_receipt": _write_bound(
            tmp_path / "inventory-receipt.json", receipt
        ),
        "hardware_envelope": _write_bound(
            tmp_path / "hardware.json", hardware.to_dict()
        ),
        "protocol": _write_bound(tmp_path / "protocol.json", protocol.to_dict()),
        "manifest": _write_bound(tmp_path / "manifest.json", manifest.to_dict()),
    }

    with pytest.raises(ValueError, match="topology identity is wrong"):
        InterferenceCalibrationSourceAuthority.from_paths(**paths).audit()


def test_bound_source_rejects_tamper_and_joint_rehash(tmp_path: Path) -> None:
    path = _write_bound(tmp_path / "source.json", {"value": 1})
    binding = RawInterferenceJsonBinding.from_path("gpu_inventory", path)

    _write_bound(path, {"value": 2})

    with pytest.raises(RuntimeError, match="bytes or sidecar changed"):
        binding.load()


def test_raw_source_and_source_authority_wire_round_trip_reopens_files(
    tmp_path: Path,
) -> None:
    source, _ = _source_authority(tmp_path, count=2, cardinality=2)

    assert (
        RawInterferenceJsonBinding.from_dict(source.inventory.to_dict())
        == source.inventory
    )
    assert InterferenceCalibrationSourceAuthority.from_dict(source.to_dict()) == source

    manifest_path = Path(source.manifest.path)
    _write_bound(manifest_path, {"jointly": "rewritten"})
    with pytest.raises(RuntimeError, match="bytes or sidecar changed"):
        InterferenceCalibrationSourceAuthority.from_dict(source.to_dict())


def test_bound_source_rejects_symlink(tmp_path: Path) -> None:
    target = _write_bound(tmp_path / "target.json", {"value": 1})
    link = tmp_path / "link.json"
    link.symlink_to(target)
    Path(f"{link}.sha256").symlink_to(Path(f"{target}.sha256"))

    with pytest.raises(ValueError, match="resolved and non-symlink"):
        RawInterferenceJsonBinding.from_path("gpu_inventory", link)


def test_confirmation_data_and_threshold_injection_are_rejected() -> None:
    inventory, _ = _inventory(2)
    hardware = InterferenceHardwareEnvelope.from_inventory(inventory)
    protocol = _protocol(inventory, hardware)

    with pytest.raises(ValueError, match="confirmation data"):
        replace(protocol, confirmation_data_visible=True)
    with pytest.raises(ValueError, match="threshold is 1%"):
        replace(protocol, maximum_absolute_relative_difference=0.02)


def test_release_trust_is_fixed_and_currently_blocked() -> None:
    with pytest.raises(InterferenceCalibrationBlockedError) as caught:
        require_release_interference_attester()

    assert caught.value.reason_code == TRUSTED_INTERFERENCE_ATTESTER_UNAVAILABLE_REASON


def test_rule_bearing_envelope_rejects_bare_evidence_digest() -> None:
    inventory, _ = _inventory(2)
    rule = InterferenceRule(
        hardware_envelope_sha256=inventory.devices[0].hardware_envelope_sha256,
        workload_class=WorkloadClass.HEADLINE,
        co_run_signature="headline-static-c1",
        simultaneous_jobs=2,
        gang_shape="tp1_dp1",
        load_thermal_power_envelope="registered-load",
        contention_class=_sha("contention"),
        evidence_sha256=_sha("bare-evidence"),
        status="PASS",
    )
    envelope = InterferenceEnvelope(
        schema_version=1,
        rules=(rule,),
        source_receipt_sha256=_sha("bare-receipt"),
    )

    with pytest.raises(InterferenceCalibrationBlockedError) as caught:
        require_calibrated_interference_execution_authority(envelope, authority=None)

    assert caught.value.reason_code == (
        CALIBRATED_INTERFERENCE_RAW_AUTHORITY_REQUIRED_REASON
    )


def test_formal_dispatch_context_blocks_calibrated_rules_before_budget_replay() -> None:
    inventory, _ = _inventory(2)
    rule = InterferenceRule(
        hardware_envelope_sha256=inventory.devices[0].hardware_envelope_sha256,
        workload_class=WorkloadClass.HEADLINE,
        co_run_signature="headline-static-c1",
        simultaneous_jobs=2,
        gang_shape="tp1_dp1",
        load_thermal_power_envelope="registered-load",
        contention_class=_sha("contention"),
        evidence_sha256=_sha("bare-evidence"),
        status="PASS",
    )
    envelope = InterferenceEnvelope(
        schema_version=1,
        rules=(rule,),
        source_receipt_sha256=_sha("bare-receipt"),
    )

    with pytest.raises(InterferenceCalibrationBlockedError) as caught:
        GpuDispatchExecutionContext(
            registry=build_industrial_registry(),
            inventory=inventory,
            interference_envelope=envelope,
            budgets=(),
            budget_plan=object(),  # type: ignore[arg-type]
            budget_materialization_authority=object(),  # type: ignore[arg-type]
        )

    assert caught.value.reason_code == (
        CALIBRATED_INTERFERENCE_RAW_AUTHORITY_REQUIRED_REASON
    )


def test_serial_deny_all_needs_no_calibration_authority() -> None:
    envelope = InterferenceEnvelope.serial(source_receipt_sha256=_sha("serial"))

    assert (
        require_calibrated_interference_execution_authority(envelope, authority=None)
        is None
    )
