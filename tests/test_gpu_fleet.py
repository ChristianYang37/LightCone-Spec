from __future__ import annotations

import copy
from dataclasses import replace
from unittest.mock import patch

import pytest

from lightcone_spec.experiments.completion_authority import (
    AssignmentTerminalBinding,
)
from lightcone_spec.experiments.gpu_fleet import (
    CROSS_HOST_COLLECTIVES_UNVALIDATED,
    FleetCapabilityRejectionError,
    FleetWaveReceipt,
    GpuFleetDispatchPlan,
    GpuFleetInventory,
    GpuFleetPlanningContext,
    GpuFleetScheduler,
    HostExecutionBinding,
    HostInventoryBinding,
    HostWaveReceipt,
    HostWaveStatus,
    assemble_gpu_fleet_inventory,
)
from lightcone_spec.experiments.gpu_pool import (
    AssignmentExecutionReceipt,
    AssignmentExecutionStatus,
    DispatchWaveExecutionReceipt,
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuPoolScheduler,
    GpuTopologyGroup,
    InterferenceEnvelope,
    registry_pool_work_item,
)
from lightcone_spec.experiments.registry import (
    ExperimentRegistry,
    content_sha256,
)
from lightcone_spec.experiments.registry import (
    build_legacy_industrial_registry as build_industrial_registry,
)


def _sha(label: str) -> str:
    return content_sha256({"test": label})


def _host_inventory(host_id: str, size: int) -> GpuInventory:
    uuids = tuple(f"GPU-{host_id}-{index:02d}" for index in range(size))
    group_id = f"all-{host_id}"
    devices = tuple(
        GpuDevice(
            uuid=uuid,
            host_id=host_id,
            model="H100-SXM",
            memory_bytes=80 * 1024**3,
            compute_capability=(9, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root="root-0",
            numa_node=0,
            interconnects=("NVLink4", "PCIe5"),
            peer_access_class="NVSwitch",
            clock_policy="locked-1980MHz",
            power_limit_watts=700.0,
            thermal_limit_celsius=83.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=(group_id,),
        )
        for index, uuid in enumerate(uuids)
    )
    topology = GpuTopologyGroup(
        group_id=group_id,
        host_id=host_id,
        gpu_uuids=uuids,
        fabric="NVSwitch",
        bandwidth_class="high",
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(topology,),
        source_receipt_sha256=_sha(f"inventory-{host_id}-{size}"),
    )


def _fleet(*host_sizes: tuple[str, int]) -> GpuFleetInventory:
    return assemble_gpu_fleet_inventory(
        tuple(
            HostInventoryBinding(
                schema_version=1,
                host_id=host_id,
                inventory=(inventory := _host_inventory(host_id, size)),
                interference_envelope=InterferenceEnvelope.serial(
                    source_receipt_sha256=_sha(
                        f"interference-{host_id}-{inventory.sha256}"
                    )
                ),
            )
            for host_id, size in host_sizes
        )
    )


def _execution_bindings(
    fleet: GpuFleetInventory,
    *,
    shared_namespaces: bool = False,
) -> tuple[HostExecutionBinding, ...]:
    return tuple(
        HostExecutionBinding(
            schema_version=1,
            host_id=host.host_id,
            inventory_sha256=host.inventory.sha256,
            interference_envelope_sha256=host.interference_envelope.sha256,
            port_start=31_000,
            port_end=31_999,
            cache_namespace=(
                "fleet-cache/shared"
                if shared_namespaces
                else f"fleet-cache/{host.host_id}"
            ),
            evidence_namespace=(
                "fleet-evidence/shared"
                if shared_namespaces
                else f"fleet-evidence/{host.host_id}"
            ),
            contention_domain=f"local-{host.host_id}",
        )
        for host in fleet.hosts
    )


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_industrial_registry()


def _static_items(
    registry: ExperimentRegistry,
    count: int,
):
    cells = tuple(
        cell
        for cell in registry.cells_for("E3a")
        if cell.identity.method == "static" and GpuPoolScheduler._dispatchable(cell)
    )
    assert len(cells) >= count
    return tuple(
        registry_pool_work_item(cell, estimated_duration_seconds=10.0)
        for cell in cells[:count]
    )


def _scheduler(
    registry: ExperimentRegistry,
    fleet: GpuFleetInventory,
    *,
    shared_namespaces: bool = False,
) -> GpuFleetScheduler:
    return GpuFleetScheduler(
        registry=registry,
        fleet_inventory=fleet,
        execution_bindings=_execution_bindings(
            fleet,
            shared_namespaces=shared_namespaces,
        ),
        seed=20260811,
    )


def test_fleet_inventory_embeds_host_authority_and_round_trips() -> None:
    fleet = _fleet(("host-b", 2), ("host-a", 1))

    assert fleet.host_ids == ("host-a", "host-b")
    assert fleet.gpu_count == 3
    restored = GpuFleetInventory.from_dict(
        fleet.to_dict(),
        sidecar=fleet.sidecar(),
    )
    assert restored == fleet
    assert restored.host("host-b").inventory.sha256 == (
        fleet.host("host-b").inventory.sha256
    )
    assert restored.host("host-b").interference_envelope.sha256 == (
        fleet.host("host-b").interference_envelope.sha256
    )

    forged = copy.deepcopy(fleet.to_dict())
    forged["hosts"][0]["inventory_sha256"] = _sha("forged")
    with pytest.raises(ValueError, match="host inventory digest mismatch"):
        GpuFleetInventory.from_dict(forged)


def test_fleet_rejects_duplicate_gpu_uuid_across_hosts() -> None:
    host_a = _fleet(("host-a", 1)).hosts[0]
    host_b_inventory = _host_inventory("host-b", 1)
    duplicate = replace(
        host_b_inventory.devices[0],
        uuid=host_a.inventory.devices[0].uuid,
    )
    topology = replace(
        host_b_inventory.topology_groups[0],
        gpu_uuids=(duplicate.uuid,),
    )
    host_b = HostInventoryBinding(
        schema_version=1,
        host_id="host-b",
        inventory=replace(
            host_b_inventory,
            devices=(duplicate,),
            topology_groups=(topology,),
        ),
        interference_envelope=InterferenceEnvelope.serial(
            source_receipt_sha256=_sha("host-b-interference")
        ),
    )

    with pytest.raises(ValueError, match="duplicate physical GPU UUID"):
        assemble_gpu_fleet_inventory((host_a, host_b))


@pytest.mark.parametrize("host_count", [2, 4])
def test_independent_items_balance_and_affinity_stays_on_one_gpu(
    registry: ExperimentRegistry,
    host_count: int,
) -> None:
    fleet = _fleet(*(tuple((f"host-{index}", 2) for index in range(host_count))))
    items = list(_static_items(registry, host_count * 2))
    affinity = _sha(f"paired-affinity-{host_count}")
    items[0] = replace(items[0], affinity_key=affinity)
    items[1] = replace(items[1], affinity_key=affinity)
    plan = _scheduler(registry, fleet).schedule_work_items(
        items,
        receipts_sha256=_sha(f"receipts-{host_count}"),
    )

    counts = {
        host_id: sum(row.host_id == host_id for row in plan.assignments)
        for host_id in fleet.host_ids
    }
    assert max(counts.values()) - min(counts.values()) <= 2
    paired = tuple(
        row
        for row in plan.assignments
        if row.assignment.work_item.affinity_key == affinity
    )
    assert len(paired) == 2
    assert len({row.host_id for row in paired}) == 1
    assert len({row.assignment.gpu_uuids for row in paired}) == 1
    assert all(
        row.host_inventory_sha256 == fleet.host(row.host_id).inventory.sha256
        for row in plan.assignments
    )


def test_serial_confirmation_group_stays_on_one_host_and_separate_waves(
    registry: ExperimentRegistry,
) -> None:
    fleet = _fleet(("host-a", 2), ("host-b", 2))
    items = list(_static_items(registry, 2))
    serial = _sha("confirmation-serial-group")
    items[0] = replace(items[0], serial_group_key=serial)
    items[1] = replace(items[1], serial_group_key=serial)

    plan = _scheduler(registry, fleet).schedule_work_items(
        items,
        receipts_sha256=_sha("serial-group-receipts"),
    )
    rows = tuple(
        row
        for row in plan.assignments
        if row.assignment.work_item.serial_group_key == serial
    )
    assert len(rows) == 2
    assert len({row.host_id for row in rows}) == 1
    assert len({row.local_wave_index for row in rows}) == 2


def test_host_local_ports_and_paths_may_repeat_across_hosts(
    registry: ExperimentRegistry,
) -> None:
    fleet = _fleet(("host-a", 1), ("host-b", 1))
    plan = _scheduler(
        registry,
        fleet,
        shared_namespaces=True,
    ).schedule_work_items(
        _static_items(registry, 2),
        receipts_sha256=_sha("host-local-resources"),
    )
    first_wave = plan.waves[0]
    assert first_wave.host_ids == ("host-a", "host-b")
    assert {row.assignment.ports for row in first_wave.assignments} == {(31_000,)}
    # Identical strings are safe because both namespaces and ports are host-local.
    assert len({row.cache_root for row in first_wave.assignments}) == 2
    assert all(
        row.cache_root.startswith("fleet-cache/shared/")
        for row in first_wave.assignments
    )


def test_plan_strict_codec_reissues_from_raw_context(
    registry: ExperimentRegistry,
) -> None:
    fleet = _fleet(("host-a", 1), ("host-b", 1))
    items = _static_items(registry, 2)
    context = GpuFleetPlanningContext(
        registry=registry,
        fleet_inventory=fleet,
        execution_bindings=_execution_bindings(fleet),
        work_items=items,
        receipts_sha256=_sha("planning-context"),
    )
    plan = context.issue_plan()

    assert (
        GpuFleetDispatchPlan.from_dict(
            plan.to_dict(),
            planning_context=context,
            sidecar=plan.sidecar(),
        )
        == plan
    )
    forged = copy.deepcopy(plan.to_dict())
    forged["seed"] = True
    with pytest.raises(ValueError, match="issued canonical value"):
        GpuFleetDispatchPlan.from_dict(forged, planning_context=context)


def test_empty_plan_is_allowed_only_for_empty_work(
    registry: ExperimentRegistry,
) -> None:
    fleet = _fleet(("host-a", 1), ("host-b", 1))
    scheduler = _scheduler(registry, fleet)
    empty = scheduler.schedule_work_items(
        (),
        receipts_sha256=_sha("empty-fleet"),
    )
    assert empty.host_plans == ()
    assert empty.waves == ()

    nonempty = scheduler.schedule_work_items(
        _static_items(registry, 1),
        receipts_sha256=_sha("nonempty-fleet"),
    )
    with pytest.raises(ValueError, match="empty or non-empty together"):
        replace(nonempty, waves=())
    with pytest.raises(ValueError, match="empty or non-empty together"):
        replace(nonempty, host_plans=())

    host_plan = nonempty.host_plans[0]
    empty_local = replace(host_plan.dispatch_plan, waves=())
    with pytest.raises(ValueError, match="cannot contain an empty plan"):
        replace(host_plan, dispatch_plan=empty_local)


def test_wire_names_child_and_wrapper_plan_digests_unambiguously(
    registry: ExperimentRegistry,
) -> None:
    fleet = _fleet(("host-a", 1), ("host-b", 1))
    plan = _scheduler(registry, fleet).schedule_work_items(
        _static_items(registry, 2),
        receipts_sha256=_sha("plan-digest-wire-names"),
    )
    value = plan.to_dict()
    host_by_id = {host.host_id: host for host in plan.host_plans}

    assert value["host_plan_sha256"] == [host.sha256 for host in plan.host_plans]
    for wave in value["waves"]:
        for assignment in wave["assignments"]:
            host = host_by_id[assignment["host_id"]]
            assert assignment["host_dispatch_plan_sha256"] == (
                host.dispatch_plan.sha256
            )
            assert "host_plan_sha256" not in assignment


def test_gang_is_single_host_or_cross_host_fails_closed(
    registry: ExperimentRegistry,
) -> None:
    gang_cell = next(
        cell
        for cell in registry.cells_for("preflight")
        if cell.identity.method == "target_only" and cell.resources.gpu_count == 2
    )
    gang = registry_pool_work_item(
        gang_cell,
        estimated_duration_seconds=10.0,
    )

    capable = _fleet(("host-a", 2), ("host-b", 1))
    with patch.object(GpuPoolScheduler, "_dispatchable", return_value=True):
        placed = _scheduler(registry, capable).schedule_work_items(
            (gang,),
            receipts_sha256=_sha("gang-capable"),
        )
    assert len(placed.assignments) == 1
    assert len(placed.assignments[0].assignment.gpu_uuids) == 2
    assert placed.assignments[0].host_id == "host-a"

    split_only = _fleet(("host-a", 1), ("host-b", 1))
    with (
        patch.object(GpuPoolScheduler, "_dispatchable", return_value=True),
        pytest.raises(FleetCapabilityRejectionError) as error,
    ):
        _scheduler(registry, split_only).schedule_work_items(
            (gang,),
            receipts_sha256=_sha("gang-split-only"),
        )
    assert error.value.reason_code == CROSS_HOST_COLLECTIVES_UNVALIDATED
    assert CROSS_HOST_COLLECTIVES_UNVALIDATED in str(error.value)


def _success_wave_receipt(
    plan: GpuFleetDispatchPlan,
    *,
    fleet_wave_index: int,
    host_id: str,
) -> HostWaveReceipt:
    fleet_wave = plan.waves[fleet_wave_index]
    fleet_assignments = tuple(
        row for row in fleet_wave.assignments if row.host_id == host_id
    )
    assert len(fleet_assignments) == 1
    wrapped = fleet_assignments[0]
    host_plan = next(row for row in plan.host_plans if row.host_id == host_id)
    local_plan = host_plan.dispatch_plan
    local_wave = local_plan.waves[wrapped.local_wave_index]
    assignment = wrapped.assignment
    budget_sha256 = _sha(f"budget-{assignment.sha256}")
    binding = AssignmentTerminalBinding(
        authority_sha256=_sha(f"authority-{assignment.sha256}"),
        cell_id=assignment.work_item.item_id,
        assignment_sha256=assignment.sha256,
        budget_sha256=budget_sha256,
        inventory_sha256=host_plan.host_inventory_sha256,
        physical_gpu_uuids=assignment.gpu_uuids,
        execution_plan_sha256=_sha(f"execution-{assignment.sha256}"),
        dispatch_plan_sha256=local_plan.sha256,
        run_id=f"run-{assignment.sha256[:8]}",
        run_nonce_sha256=_sha(f"nonce-{assignment.sha256}"),
        terminal_receipt_path=f"/private/tmp/{assignment.sha256}-terminal.json",
        terminal_receipt_sha256=_sha(f"terminal-{assignment.sha256}"),
        budget_observation_path=f"/private/tmp/{assignment.sha256}-budget.json",
        budget_observation_sha256=_sha(f"observation-{assignment.sha256}"),
        budget_observation_sidecar_path=(
            f"/private/tmp/{assignment.sha256}-budget.json.sha256"
        ),
        budget_observation_sidecar_sha256=_sha(
            f"observation-sidecar-{assignment.sha256}"
        ),
        native_terminal_artifact_path=(f"/private/tmp/{assignment.sha256}-native.json"),
        native_terminal_raw_sha256=_sha(f"native-raw-{assignment.sha256}"),
        native_terminal_sha256=_sha(f"native-{assignment.sha256}"),
        trusted_attester_policy_sha256=_sha(f"policy-{assignment.sha256}"),
        evidence_file_paths=(f"/private/tmp/{assignment.sha256}-evidence.json",),
        evidence_file_sha256s=(_sha(f"evidence-{assignment.sha256}"),),
    )
    fixed_gpu_count = len(
        next(row for row in plan.host_plans if row.host_id == host_id)
        .dispatch_plan.waves[0]
        .assignments[0]
        .gpu_uuids
    )
    assignment_receipt = AssignmentExecutionReceipt(
        plan_sha256=local_plan.sha256,
        wave_sha256=local_wave.sha256,
        assignment_sha256=assignment.sha256,
        budget_sha256=budget_sha256,
        attempt=1,
        status=AssignmentExecutionStatus.SUCCEEDED,
        terminal_receipt_sha256=binding.sha256,
        terminal_binding=binding,
        failure_sha256=None,
        prior_attempt_receipt_sha256=None,
        gpu_count=len(assignment.gpu_uuids),
        fixed_instance_gpu_count=fixed_gpu_count,
        attempt_intervals_monotonic_ns=((1, 2),),
        attributed_gpu_ns=len(assignment.gpu_uuids),
        attributed_fixed_instance_gpu_ns=fixed_gpu_count,
    )
    dispatch = DispatchWaveExecutionReceipt(
        plan_sha256=local_plan.sha256,
        wave_index=local_wave.wave_index,
        wave_sha256=local_wave.sha256,
        assignment_receipts=(assignment_receipt,),
        inventory_sha256=host_plan.host_inventory_sha256,
        fixed_instance_gpu_count=fixed_gpu_count,
        active_intervals_monotonic_ns=((1, 2),),
        fixed_instance_actual_billed_gpu_ns=fixed_gpu_count,
        per_assignment_attributed_gpu_ns=len(assignment.gpu_uuids),
        per_assignment_attributed_fixed_instance_gpu_ns=fixed_gpu_count,
    )
    return HostWaveReceipt(
        host_id=host_id,
        fleet_plan_sha256=plan.sha256,
        fleet_wave_sha256=fleet_wave.sha256,
        host_dispatch_plan_sha256=local_plan.sha256,
        local_wave_index=local_wave.wave_index,
        local_wave_sha256=local_wave.sha256,
        attempt=1,
        status=HostWaveStatus.SUCCEEDED,
        dispatch_receipt=dispatch,
        failure_sha256=None,
    )


def _failed_host_receipt(
    plan: GpuFleetDispatchPlan,
    *,
    fleet_wave_index: int,
    host_id: str,
    attempt: int = 1,
    prior: HostWaveReceipt | None = None,
) -> HostWaveReceipt:
    fleet_wave = plan.waves[fleet_wave_index]
    wrapped = next(row for row in fleet_wave.assignments if row.host_id == host_id)
    host_plan = next(row for row in plan.host_plans if row.host_id == host_id)
    return HostWaveReceipt(
        host_id=host_id,
        fleet_plan_sha256=plan.sha256,
        fleet_wave_sha256=fleet_wave.sha256,
        host_dispatch_plan_sha256=host_plan.dispatch_plan.sha256,
        local_wave_index=wrapped.local_wave_index,
        local_wave_sha256=wrapped.local_wave_sha256,
        attempt=attempt,
        status=HostWaveStatus.FAILED,
        dispatch_receipt=None,
        failure_sha256=_sha(f"failure-{host_id}-{attempt}"),
        prior_host_receipt_sha256=None if prior is None else prior.sha256,
    )


def test_receipt_aggregation_preserves_success_and_retries_failed_host_only(
    registry: ExperimentRegistry,
) -> None:
    fleet = _fleet(("host-a", 1), ("host-b", 1))
    plan = _scheduler(registry, fleet).schedule_work_items(
        _static_items(registry, 2),
        receipts_sha256=_sha("receipt-aggregation"),
    )
    assert plan.waves[0].host_ids == ("host-a", "host-b")
    success = _success_wave_receipt(plan, fleet_wave_index=0, host_id="host-a")
    failed = _failed_host_receipt(
        plan,
        fleet_wave_index=0,
        host_id="host-b",
    )
    first = FleetWaveReceipt.aggregate(
        plan,
        fleet_wave_index=0,
        host_receipts=(success, failed),
    )
    assert first.failed_host_ids == ("host-b",)

    retried = _failed_host_receipt(
        plan,
        fleet_wave_index=0,
        host_id="host-b",
        attempt=2,
        prior=failed,
    )
    second = FleetWaveReceipt.aggregate(
        plan,
        fleet_wave_index=0,
        host_receipts=(success, retried),
        prior_receipt=first,
    )
    assert second.prior_fleet_receipt_sha256 == first.sha256
    assert second.host_receipts[0] == success
    assert (
        FleetWaveReceipt.from_dict(
            second.to_dict(),
            plan=plan,
            prior_receipt=first,
            sidecar=second.sidecar(),
        )
        == second
    )

    migrated = replace(retried, host_id="host-a")
    with pytest.raises(ValueError, match="add, remove, or migrate"):
        FleetWaveReceipt.aggregate(
            plan,
            fleet_wave_index=0,
            host_receipts=(success, migrated),
            prior_receipt=first,
        )
