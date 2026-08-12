from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from lightcone_spec.adaptation.parameters import DFlashParameterPlan
from lightcone_spec.config.schema import (
    AdaptationConfig,
    ModelPair,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAssignment,
    GpuAvailability,
    GpuDevice,
    GpuDispatchPlan,
    GpuDispatchPlanningContext,
    GpuDispatchWave,
    GpuInventory,
    GpuPoolScheduler,
    GpuTopologyGroup,
    InterferenceEnvelope,
    registry_pool_work_item,
)
from lightcone_spec.experiments.planning import (
    BudgetJobKind,
    ExpectedMaximumCount,
    ExperimentBudget,
    P99AnchorStatus,
    ScenarioMilliseconds,
)
from lightcone_spec.experiments.protocol import DFLASH_LOSS_POSITION_DECAY
from lightcone_spec.experiments.registry import (
    INDUSTRIAL_EXPERIMENT_ORDER,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    build_industrial_registry,
    content_sha256,
    serving_cell_rejection_reason,
)
from lightcone_spec.experiments.stage_activation import (
    materialize_registry_stage_activation,
)
from lightcone_spec.orchestration.industrial import (
    bind_industrial_gpu_assignment,
    render_assigned_industrial_cell_runtime_plan,
    render_industrial_cell_runtime_plan,
)
from lightcone_spec.runtime.distributed import (
    RankTopologyReceipt,
    TopologyIdentity,
    TopologyReceiptSet,
)


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_industrial_registry(
        gpu_uuids=("GPU-render-a", "GPU-render-b"),
        cache_root="runtime-cache/renderer",
        evidence_root="artifacts/renderer",
    )


def _receipt(
    registry: ExperimentRegistry,
    experiment: str,
    dependencies: tuple[ExperimentReceipt, ...],
) -> ExperimentReceipt:
    definition = registry.definition(experiment)
    return registry.make_receipt(
        experiment,
        {
            name: content_sha256({"experiment": experiment, "output": name})
            for name in definition.locked_outputs
        },
        runtime_sha256=content_sha256({"runtime": experiment}),
        split_sha256=content_sha256({"split": experiment}),
        completed_cells_sha256=content_sha256({"completed": experiment}),
        dependencies=dependencies,
    )


def _receipts_before(
    registry: ExperimentRegistry,
    experiment: str,
) -> tuple[ExperimentReceipt, ...]:
    receipts: list[ExperimentReceipt] = []
    for name in INDUSTRIAL_EXPERIMENT_ORDER:
        if name == experiment:
            return tuple(receipts)
        receipts.append(_receipt(registry, name, tuple(receipts)))
    raise AssertionError(f"unknown experiment {experiment}")


def _runtime_envelope(receipts: tuple[ExperimentReceipt, ...]) -> str | None:
    for receipt in receipts:
        if receipt.experiment != "preflight":
            continue
        return next(
            output.content_sha256
            for output in receipt.outputs
            if output.name == "runtime_envelope"
        )
    return None


def _topology(
    cell: ExperimentCell,
    *,
    tp_size: int = 1,
    dp_size: int = 1,
) -> TopologyReceiptSet:
    world_size = tp_size * dp_size
    assert world_size == len(cell.identity.gpu_uuids)
    receipts = []
    for rank, device in enumerate(cell.identity.gpu_uuids):
        receipts.append(
            RankTopologyReceipt(
                topology=TopologyIdentity(
                    tensor_parallel_size=tp_size,
                    data_parallel_size=dp_size,
                    node_count=1,
                    node_id="renderer-host",
                    node_rank=0,
                    global_rank=rank,
                    local_rank=rank,
                    tensor_parallel_rank=rank % tp_size,
                    data_parallel_rank=rank // tp_size,
                    device_id=device,
                    rendezvous_id="renderer-rendezvous",
                    router_id=("single-replica" if dp_size == 1 else "sticky-router"),
                    clock_id="renderer-clock",
                ),
                process_id=f"renderer-process-{rank}",
                observed_world_size=world_size,
            )
        )
    return TopologyReceiptSet(tuple(receipts))


def _physical_topology(
    gpu_uuids: tuple[str, ...],
    *,
    tp_size: int,
    dp_size: int,
) -> TopologyReceiptSet:
    world_size = tp_size * dp_size
    assert world_size == len(gpu_uuids)
    return TopologyReceiptSet(
        tuple(
            RankTopologyReceipt(
                topology=TopologyIdentity(
                    tensor_parallel_size=tp_size,
                    data_parallel_size=dp_size,
                    node_count=1,
                    node_id="renderer-host",
                    node_rank=0,
                    global_rank=rank,
                    local_rank=rank,
                    tensor_parallel_rank=rank % tp_size,
                    data_parallel_rank=rank // tp_size,
                    device_id=device,
                    rendezvous_id="physical-renderer-rendezvous",
                    router_id=(
                        "single-replica" if dp_size == 1 else "physical-sticky-router"
                    ),
                    clock_id="physical-renderer-clock",
                ),
                process_id=f"physical-renderer-process-{rank}",
                observed_world_size=world_size,
            )
            for rank, device in enumerate(gpu_uuids)
        )
    )


def _physical_inventory() -> GpuInventory:
    gpu_uuids = ("GPU-physical-a", "GPU-physical-b")
    group_id = "renderer-nvlink"
    devices = tuple(
        GpuDevice(
            uuid=uuid,
            host_id="renderer-host",
            model="renderer-gpu",
            memory_bytes=80 * 1024**3,
            compute_capability=(9, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root="renderer-root",
            numa_node=0,
            interconnects=("nvlink",),
            peer_access_class="renderer-peer",
            clock_policy="locked",
            power_limit_watts=700.0,
            thermal_limit_celsius=83.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=(group_id,),
        )
        for index, uuid in enumerate(gpu_uuids)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id=group_id,
                host_id="renderer-host",
                gpu_uuids=gpu_uuids,
                fabric="nvlink",
                bandwidth_class="renderer-full-bandwidth",
            ),
        ),
        source_receipt_sha256=content_sha256({"inventory": "renderer"}),
    )


def _assignment(
    cell: ExperimentCell,
    *,
    gpu_uuids: tuple[str, ...],
    rank_groups: tuple[tuple[str, ...], ...],
    ports: tuple[int, ...],
) -> GpuAssignment:
    return GpuAssignment(
        work_item=registry_pool_work_item(
            cell,
            estimated_duration_seconds=123.0,
        ),
        gpu_uuids=gpu_uuids,
        rank_groups=rank_groups,
        ports=ports,
    )


def _budget(cell: ExperimentCell) -> ExperimentBudget:
    zero = ScenarioMilliseconds(0, 0, 0)
    one = ScenarioMilliseconds(1, 1, 1)
    gpu_time = ScenarioMilliseconds(
        cell.resources.gpu_count,
        cell.resources.gpu_count,
        cell.resources.gpu_count,
    )
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=cell.resources.workload_class,
        job_kind=BudgetJobKind.STANDARD,
        startup_model_load=zero,
        compile_jit_graph_prewarm=zero,
        excluded_warmup=zero,
        excluded_warmup_requests=ExpectedMaximumCount(0, 0),
        scored_arrival=one,
        request_deadline=one,
        drain=zero,
        reset_finalization=zero,
        evidence_flush_shutdown=zero,
        output_tokens=ExpectedMaximumCount(1, 1),
        minimum_completed_requests=1,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=zero,
        failure_injection=zero,
        retry=zero,
        retry_allowance=0,
        profiler=zero,
        download_compile_reservation=zero,
        gpu_count=cell.resources.gpu_count,
        topology=cell.identity.topology,
        reserved_gpu_ms=gpu_time,
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=one.scale(2),
    )


def _planning_context(
    registry: ExperimentRegistry,
    inventory: GpuInventory,
) -> GpuDispatchPlanningContext:
    cells = tuple(
        cell
        for cell in registry.cells_for("E3a")
        if GpuPoolScheduler._dispatchable(cell)
    )
    receipts = _receipts_before(registry, "E3a")
    activation = materialize_registry_stage_activation(
        registry,
        experiment="E3a",
        dependency_receipts=receipts,
        runtime_sha256=content_sha256("renderer-runtime"),
        split_sha256=content_sha256("renderer-split"),
    )
    return GpuDispatchPlanningContext(
        registry=registry,
        inventory=inventory,
        interference_envelope=InterferenceEnvelope.serial(
            source_receipt_sha256=content_sha256("renderer-serial")
        ),
        budgets=tuple(
            sorted((_budget(cell) for cell in cells), key=lambda row: row.cell_id)
        ),
        receipts=receipts,
        activation_artifact=activation,
        port_start=31_000,
        port_end=31_999,
        seed=20260811,
    )


def _dispatch_plan(
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    assignment: GpuAssignment,
    budget: ExperimentBudget,
) -> GpuDispatchPlan:
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("renderer-serial")
    )
    return GpuDispatchPlan(
        schema_version=1,
        registry_sha256=registry.sha256,
        inventory_sha256=inventory.sha256,
        receipts_sha256=content_sha256("renderer-receipts"),
        interference_envelope_sha256=envelope.sha256,
        budget_sha256_by_cell=((assignment.work_item.item_id, budget.sha256),),
        seed=20260811,
        waves=(
            GpuDispatchWave(
                wave_index=0,
                assignments=(assignment,),
                interference_envelope_sha256=envelope.sha256,
            ),
        ),
        completed_cell_ids=(),
    )


def _materialise(config: RunConfig) -> RunConfig:
    """Round-trip an on-disk-style config so every field is explicit."""

    return RunConfig.model_validate(config.model_dump(mode="json"))


def _configs(
    cell: ExperimentCell,
    topology: TopologyReceiptSet,
    receipts: tuple[ExperimentReceipt, ...],
    *,
    adaptation: AdaptationConfig | None = None,
) -> tuple[RunConfig, ...]:
    identity = cell.identity
    algorithm = "DFLASH" if identity.backend == "NONE" else identity.backend
    width = identity.width if identity.width is not None else 16
    configs = []
    for rank in range(topology.world_size):
        rank_identity = topology.receipt_for_rank(rank).topology
        distributed = topology.world_size > 1
        configs.append(
            _materialise(
                RunConfig(
                    method=identity.method,
                    model=ModelPair(
                        target=identity.model,
                        drafter="test/drafter",
                        target_revision="1" * 40,
                        drafter_revision="2" * 40,
                        algorithm=algorithm,
                        max_context_length=identity.context,
                        draft_depth=width - 1,
                    ),
                    runtime=RuntimeConfig(
                        sampling_profile_sha256="3" * 64,
                        speculation_enabled=identity.method != "target_only",
                        tensor_parallel_size=topology.tensor_parallel_size,
                        data_parallel_size=topology.data_parallel_size,
                        tp_rank=rank_identity.tensor_parallel_rank,
                        dp_rank=rank_identity.data_parallel_rank,
                        node_count=1,
                        node_rank=0,
                        device_identity=rank_identity.device_id,
                        rendezvous_identity=rank_identity.rendezvous_id,
                        router_identity=rank_identity.router_id,
                        clock_identity=rank_identity.clock_id,
                        process_group_backend="nccl",
                        distributed_runtime_capability=(
                            "patched_two_gpu_v1" if distributed else "single_rank"
                        ),
                        distributed_capability_receipt_sha256=(
                            _runtime_envelope(receipts) if distributed else None
                        ),
                        speculative_num_draft_tokens=width,
                        speculative_eagle_topk=(
                            1 if algorithm in {"EAGLE", "EAGLE3"} else None
                        ),
                        use_rejection_sampling=True,
                        max_running_requests=identity.concurrency,
                        telemetry_detail="headline",
                        prefill_decode_disaggregation=False,
                        two_batch_overlap=False,
                    ),
                    adaptation=adaptation,
                    online_spec=None,
                    tenant_id="renderer-test",
                )
            )
        )
    return tuple(configs)


def _replace_cell(
    registry: ExperimentRegistry,
    source: ExperimentCell,
    replacement: ExperimentCell,
) -> ExperimentRegistry:
    return replace(
        registry,
        cells=tuple(
            replacement if cell.cell_id == source.cell_id else cell
            for cell in registry.cells
        ),
    )


def test_static_cell_renders_to_stable_content_bound_contract(
    registry: ExperimentRegistry,
) -> None:
    cell = next(
        cell
        for cell in registry.cells_for("E3a")
        if cell.identity.method == "static" and cell.identity.concurrency == 1
    )
    receipts = _receipts_before(registry, "E3a")
    topology = _topology(cell)
    configs = _configs(cell, topology, receipts)

    plan = render_industrial_cell_runtime_plan(
        registry=registry,
        cell_id=cell.cell_id,
        rank_configs=configs,
        topology_receipts=topology,
        dependency_receipts=receipts,
    )
    repeated = render_industrial_cell_runtime_plan(
        registry=registry,
        cell_id=cell.cell_id,
        rank_configs=configs,
        topology_receipts=topology,
        dependency_receipts=receipts,
    )

    assert plan.sha256 == repeated.sha256
    assert plan.registry_sha256 == registry.sha256
    assert plan.cell_id == cell.cell_id
    assert plan.parameter_plan_sha256 is None
    assert plan.to_dict()["resources"]["gpu_uuids"] == list(cell.identity.gpu_uuids)
    assert plan.to_dict()["workload"]["context"] == cell.identity.context
    assert plan.physical_dispatch_ready is False
    with pytest.raises(ValueError, match="no physical GPU assignment"):
        _ = plan.physical_gpu_uuids
    assert plan.to_dict()["physical_gpu_uuids"] is None
    assert plan.to_dict()["physical_rank_groups"] is None
    assert plan.to_dict()["physical_ports"] is None
    assert plan.to_dict()["physical_fixed_instance_gpu_count"] is None
    assert plan.to_dict()["resource_binding"] == {
        "kind": "registry_logical_only",
        "physical_dispatch_ready": False,
        "physical_assignment_sha256": None,
        "physical_binding_sha256": None,
        "physical_assignment": None,
    }


def test_assignment_renderer_binds_tp1_physical_resources_without_changing_cell(
    registry: ExperimentRegistry,
) -> None:
    cell = next(
        cell
        for cell in registry.cells_for("E3a")
        if GpuPoolScheduler._dispatchable(cell)
    )
    receipts = _receipts_before(registry, "E3a")
    inventory = _physical_inventory()
    dispatch_context = _planning_context(registry, inventory)
    dispatch_plan = dispatch_context.issue_plan()
    assignment = next(
        assignment
        for wave in dispatch_plan.waves
        for assignment in wave.assignments
        if assignment.work_item.item_id == cell.cell_id
    )
    physical_gpu = assignment.gpu_uuids
    budget = dispatch_context.budgets_by_cell_id[cell.cell_id]
    topology = _physical_topology(physical_gpu, tp_size=1, dp_size=1)
    configs = _configs(cell, topology, receipts)

    with pytest.raises(TypeError, match="GpuDispatchExecutionContext"):
        render_assigned_industrial_cell_runtime_plan(
            registry=registry,
            cell_id=cell.cell_id,
            assignment=assignment,
            dispatch_plan=dispatch_plan,
            dispatch_context=dispatch_context,
            budget=budget,
            inventory=inventory,
            dispatch_inventory_sha256=inventory.sha256,
            rank_configs=configs,
            topology_receipts=topology,
            dependency_receipts=receipts,
        )


@pytest.mark.parametrize(
    ("topology_name", "tp_size", "dp_size", "rank_groups"),
    (
        ("tp2_dp1", 2, 1, (("GPU-physical-a", "GPU-physical-b"),)),
        (
            "two_replica_tp1_dp2",
            1,
            2,
            (("GPU-physical-a",), ("GPU-physical-b",)),
        ),
    ),
)
def test_assignment_binding_covers_registered_two_rank_gang_shapes(
    registry: ExperimentRegistry,
    topology_name: str,
    tp_size: int,
    dp_size: int,
    rank_groups: tuple[tuple[str, ...], ...],
) -> None:
    source = next(
        cell
        for cell in registry.cells_for("E3a")
        if cell.identity.method == "static" and cell.identity.concurrency == 1
    )
    cell = replace(
        source,
        identity=replace(
            source.identity,
            topology=topology_name,
            gpu_uuids=registry.gpu_uuids,
        ),
        resources=replace(
            source.resources,
            gpu_uuids=registry.gpu_uuids,
            ports=(32_000, 32_001, 32_002),
        ),
    )
    derived = _replace_cell(registry, source, cell)
    inventory = _physical_inventory()
    physical_gpus = ("GPU-physical-a", "GPU-physical-b")
    assignment = _assignment(
        cell,
        gpu_uuids=physical_gpus,
        rank_groups=rank_groups,
        ports=(33_000, 33_001, 33_002),
    )
    budget = _budget(cell)
    dispatch_plan = _dispatch_plan(derived, inventory, assignment, budget)
    dispatch_context = _planning_context(derived, inventory)
    topology = _physical_topology(
        physical_gpus,
        tp_size=tp_size,
        dp_size=dp_size,
    )

    with pytest.raises(TypeError, match="GpuDispatchExecutionContext"):
        bind_industrial_gpu_assignment(
            registry=derived,
            cell_id=cell.cell_id,
            assignment=assignment,
            dispatch_plan=dispatch_plan,
            dispatch_context=dispatch_context,
            budget=budget,
            inventory=inventory,
            dispatch_inventory_sha256=inventory.sha256,
            topology_receipts=topology,
        )


def test_planning_assignment_cannot_cross_physical_binding_boundary(
    registry: ExperimentRegistry,
) -> None:
    cell = next(
        cell
        for cell in registry.cells_for("E3a")
        if GpuPoolScheduler._dispatchable(cell)
    )
    inventory = _physical_inventory()
    dispatch_context = _planning_context(registry, inventory)
    dispatch_plan = dispatch_context.issue_plan()
    assignment = next(
        assignment
        for wave in dispatch_plan.waves
        for assignment in wave.assignments
        if assignment.work_item.item_id == cell.cell_id
    )
    physical_gpus = assignment.gpu_uuids
    ports = assignment.ports
    topology = _physical_topology(physical_gpus, tp_size=1, dp_size=1)
    budget = dispatch_context.budgets_by_cell_id[cell.cell_id]

    with pytest.raises(TypeError, match="GpuDispatchExecutionContext"):
        bind_industrial_gpu_assignment(
            registry=registry,
            cell_id=cell.cell_id,
            assignment=assignment,
            dispatch_plan=dispatch_plan,
            dispatch_context=dispatch_context,
            budget=budget,
            inventory=inventory,
            dispatch_inventory_sha256="f" * 64,
            topology_receipts=topology,
        )

    forged_cell = replace(cell, reason=f"{cell.reason} forged")
    forged_assignment = replace(
        assignment,
        work_item=replace(assignment.work_item, cell=forged_cell),
    )
    with pytest.raises(TypeError, match="GpuDispatchExecutionContext"):
        bind_industrial_gpu_assignment(
            registry=registry,
            cell_id=cell.cell_id,
            assignment=forged_assignment,
            dispatch_plan=dispatch_plan,
            dispatch_context=dispatch_context,
            budget=budget,
            inventory=inventory,
            dispatch_inventory_sha256=inventory.sha256,
            topology_receipts=topology,
        )

    reordered_gpus = tuple(reversed(physical_gpus))
    reordered_assignment = _assignment(
        cell,
        gpu_uuids=reordered_gpus,
        rank_groups=(reordered_gpus,),
        ports=ports,
    )
    reordered_dispatch_plan = _dispatch_plan(
        registry, inventory, reordered_assignment, budget
    )
    with pytest.raises(TypeError, match="GpuDispatchExecutionContext"):
        bind_industrial_gpu_assignment(
            registry=registry,
            cell_id=cell.cell_id,
            assignment=reordered_assignment,
            dispatch_plan=reordered_dispatch_plan,
            dispatch_context=dispatch_context,
            budget=budget,
            inventory=inventory,
            dispatch_inventory_sha256=inventory.sha256,
            topology_receipts=topology,
        )

    reordered_ports = _assignment(
        cell,
        gpu_uuids=physical_gpus,
        rank_groups=(physical_gpus,),
        ports=tuple(reversed(ports)),
    )
    reordered_ports_dispatch_plan = _dispatch_plan(
        registry, inventory, reordered_ports, budget
    )
    with pytest.raises(TypeError, match="GpuDispatchExecutionContext"):
        render_assigned_industrial_cell_runtime_plan(
            registry=registry,
            cell_id=cell.cell_id,
            assignment=reordered_ports,
            dispatch_plan=reordered_ports_dispatch_plan,
            dispatch_context=dispatch_context,
            budget=budget,
            inventory=inventory,
            dispatch_inventory_sha256=inventory.sha256,
            rank_configs=(),
            topology_receipts=topology,
        )

    partial_assignment = _assignment(
        cell,
        gpu_uuids=physical_gpus,
        rank_groups=(physical_gpus,),
        ports=ports,
    )
    object.__setattr__(partial_assignment, "gpu_uuids", physical_gpus[:1])
    object.__setattr__(partial_assignment, "rank_groups", (physical_gpus[:1],))
    partial_dispatch_plan = _dispatch_plan(
        registry, inventory, partial_assignment, budget
    )
    with pytest.raises(TypeError, match="GpuDispatchExecutionContext"):
        bind_industrial_gpu_assignment(
            registry=registry,
            cell_id=cell.cell_id,
            assignment=partial_assignment,
            dispatch_plan=partial_dispatch_plan,
            dispatch_context=dispatch_context,
            budget=budget,
            inventory=inventory,
            dispatch_inventory_sha256=inventory.sha256,
            topology_receipts=topology,
        )


def test_physical_binding_rejects_canonical_e5_cell_before_ready_stage(
    registry: ExperimentRegistry,
) -> None:
    inventory = _physical_inventory()
    dispatch_context = _planning_context(registry, inventory)
    cell = next(
        cell
        for cell in registry.cells_for("E5")
        if GpuPoolScheduler._dispatchable(cell)
    )
    assignment = _assignment(
        cell,
        gpu_uuids=(inventory.devices[0].uuid,),
        rank_groups=((inventory.devices[0].uuid,),),
        ports=(31_000,),
    )
    budget = _budget(cell)
    dispatch_plan = _dispatch_plan(registry, inventory, assignment, budget)
    topology = _physical_topology(
        assignment.gpu_uuids,
        tp_size=1,
        dp_size=1,
    )

    with pytest.raises(TypeError, match="GpuDispatchExecutionContext"):
        bind_industrial_gpu_assignment(
            registry=registry,
            cell_id=cell.cell_id,
            assignment=assignment,
            dispatch_plan=dispatch_plan,
            dispatch_context=dispatch_context,
            budget=budget,
            inventory=inventory,
            dispatch_inventory_sha256=inventory.sha256,
            topology_receipts=topology,
        )


def test_renderer_rejects_missing_chain_defaults_and_identity_drift(
    registry: ExperimentRegistry,
) -> None:
    cell = next(
        cell
        for cell in registry.cells_for("E3a")
        if cell.identity.method == "static" and cell.identity.concurrency == 1
    )
    receipts = _receipts_before(registry, "E3a")
    topology = _topology(cell)
    configs = _configs(cell, topology, receipts)

    with pytest.raises(ValueError, match="complete preceding receipt chain"):
        render_industrial_cell_runtime_plan(
            registry=registry,
            cell_id=cell.cell_id,
            rank_configs=configs,
            topology_receipts=topology,
        )

    implicit = RunConfig(
        method="static",
        model=configs[0].model,
        runtime=configs[0].runtime,
    )
    with pytest.raises(ValueError, match="explicitly materialise every field"):
        render_industrial_cell_runtime_plan(
            registry=registry,
            cell_id=cell.cell_id,
            rank_configs=(implicit,),
            topology_receipts=topology,
            dependency_receipts=receipts,
        )

    changed = configs[0].model_dump(mode="json")
    changed["runtime"]["max_running_requests"] += 1
    with pytest.raises(ValueError, match="admission cap"):
        render_industrial_cell_runtime_plan(
            registry=registry,
            cell_id=cell.cell_id,
            rank_configs=(RunConfig.model_validate(changed),),
            topology_receipts=topology,
            dependency_receipts=receipts,
        )


def test_nonserving_blocked_and_unresolved_cells_fail_before_allocation(
    registry: ExperimentRegistry,
) -> None:
    preflight = registry.cells_for("preflight")[0]
    with pytest.raises(ValueError, match="dedicated non-serving executor"):
        render_industrial_cell_runtime_plan(
            registry=registry,
            cell_id=preflight.cell_id,
            rank_configs=(),
            topology_receipts=_topology(preflight, tp_size=2),
        )

    blocked = next(cell for cell in registry.cells_for("E2") if not cell.runnable)
    with pytest.raises(ValueError, match="only UNMEASURED"):
        render_industrial_cell_runtime_plan(
            registry=registry,
            cell_id=blocked.cell_id,
            rank_configs=(),
            topology_receipts=_topology(blocked),
        )

    unresolved = next(
        cell for cell in registry.cells_for("E1") if cell.identity.method == "l0"
    )
    assert serving_cell_rejection_reason(unresolved) == (
        "cell contains unresolved semantic placeholder 'locked_reference_load'"
    )
    with pytest.raises(ValueError, match="unresolved semantic placeholder"):
        render_industrial_cell_runtime_plan(
            registry=registry,
            cell_id=unresolved.cell_id,
            rank_configs=(),
            topology_receipts=_topology(unresolved),
        )


def test_e2_caller_defaults_cannot_unlock_blocked_recipe_declaration(
    registry: ExperimentRegistry,
) -> None:
    source = next(
        cell
        for cell in registry.cells_for("E2")
        if cell.identity.method == "l0"
        and cell.identity.parameterization == "full"
        and cell.identity.scope == "last1"
        and cell.identity.optimizer == "adamw"
        and cell.identity.schedule == "constant"
    )
    identity = replace(source.identity, arrival="closed_loop", concurrency=1)
    cell = replace(source, identity=identity)
    derived = _replace_cell(registry, source, cell)
    receipts = _receipts_before(derived, "E2")
    topology = _topology(cell)
    declaration = derived.adaptation_recipe_for_cell(cell)
    assert declaration.status == "BLOCKED"
    assert "e2_weight_decay_unregistered" in declaration.blocker_codes
    assert declaration.lookup_key.draft_width_selector is not None
    adaptation = AdaptationConfig(
        weight_update_mode="full",
        parameter_scope="last1",
        adaptation_group_id=identity.cohort,
        optimizer=OptimizerConfig(
            name="adamw",
            learning_rate=identity.learning_rate,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            grad_clip=1.0,
            momentum=None,
            muon_ns_steps=None,
            muon_auxiliary_learning_rate=None,
            muon_auxiliary_weight_decay=None,
            schedule="constant",
            schedule_total_published_updates=None,
        ),
        rank=None,
        lora_alpha=None,
        lora_matrix_policy="registered_matrices_v1",
        native_head_policy="frozen",
        stride=10,
        max_in_flight=1,
        canvas_tokens=16,
        loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
        extra_logical_delay=0,
        teacher_row_policy="update_round",
        verification_mode="native_scheduler",
        fixed_verification_budget=None,
        confidence_loss_weight=None,
    )
    configs = _configs(cell, topology, receipts, adaptation=adaptation)
    parameter_plan = DFlashParameterPlan.build(
        {"layers.0.q_proj.weight": torch.zeros(4, 4)},
        mode="full",
        scope="last1",
    )

    with pytest.raises(ValueError, match="only UNMEASURED"):
        render_industrial_cell_runtime_plan(
            registry=derived,
            cell_id=cell.cell_id,
            rank_configs=configs,
            topology_receipts=topology,
            dependency_receipts=receipts,
            parameter_plan=parameter_plan,
        )


def test_release_blocks_two_rank_cells_and_capability_digest_cannot_bypass(
    registry: ExperimentRegistry,
) -> None:
    blocked = next(
        cell for cell in registry.cells_for("E5") if cell.identity.topology == "tp2_dp1"
    )
    assert not blocked.runnable
    assert blocked.reason_code == "release_topology_executor_unsupported"
    with pytest.raises(ValueError, match="only UNMEASURED"):
        render_industrial_cell_runtime_plan(
            registry=registry,
            cell_id=blocked.cell_id,
            rank_configs=(),
            topology_receipts=_topology(blocked, tp_size=2),
            dependency_receipts=_receipts_before(registry, "E5"),
        )

    # A hand-edited registry plus Pydantic's unsafe model_construct escape hatch
    # still cannot turn an arbitrary capability digest into an executable plan.
    source = next(
        cell
        for cell in registry.cells_for("E3a")
        if cell.identity.method == "static" and cell.identity.concurrency == 1
    )
    identity = replace(
        source.identity,
        gpu_uuids=registry.gpu_uuids,
        topology="tp2_dp1",
    )
    resources = replace(
        source.resources,
        gpu_uuids=registry.gpu_uuids,
        ports=(
            source.resources.ports[0],
            source.resources.ports[0] + 1,
            source.resources.ports[0] + 2,
        ),
    )
    cell = replace(source, identity=identity, resources=resources)
    derived = _replace_cell(registry, source, cell)
    receipts = _receipts_before(derived, "E3a")
    topology = _topology(cell, tp_size=2)
    single_topology = _topology(source)
    single = _configs(source, single_topology, receipts)[0]
    unsafe_configs = []
    capability = _runtime_envelope(receipts)
    assert capability is not None
    for rank in range(2):
        rank_topology = topology.receipt_for_rank(rank).topology
        runtime_value = single.runtime.model_dump(mode="json")
        runtime_value.update(
            {
                "tensor_parallel_size": 2,
                "tp_rank": rank,
                "device_identity": rank_topology.device_id,
                "distributed_runtime_capability": "patched_two_gpu_v1",
                "distributed_capability_receipt_sha256": capability,
            }
        )
        unsafe_runtime = RuntimeConfig.model_construct(
            _fields_set=set(runtime_value),
            **runtime_value,
        )
        root_value = {
            "schema_version": single.schema_version,
            "method": single.method,
            "model": single.model,
            "runtime": unsafe_runtime,
            "adaptation": single.adaptation,
            "online_spec": single.online_spec,
            "tenant_id": single.tenant_id,
        }
        unsafe_configs.append(
            RunConfig.model_construct(
                _fields_set=set(root_value),
                **root_value,
            )
        )

    with pytest.raises(ValueError, match="current strict RunConfig schema"):
        render_industrial_cell_runtime_plan(
            registry=derived,
            cell_id=cell.cell_id,
            rank_configs=tuple(unsafe_configs),
            topology_receipts=topology,
            dependency_receipts=receipts,
        )
