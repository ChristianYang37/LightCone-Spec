from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
    InterferenceEnvelope,
    InterferenceRule,
    registry_pool_work_item,
)
from lightcone_spec.experiments.load import (
    FrozenSamplingParameters,
    ProductionLoadPlan,
    ProductionWindow,
    RequestTemplate,
    closed_loop_corpus,
)
from lightcone_spec.experiments.planning import (
    BUDGET_MATERIALIZATION_PROTOCOL_SHA256,
    ZERO_MILLISECONDS,
    BudgetDispositionStatus,
    BudgetInventoryIdentity,
    BudgetJobKind,
    BudgetJobPolicy,
    BudgetLoadBinding,
    BudgetPolicy,
    CapacityEnvelope,
    CellCapacityRequirement,
    ExpectedMaximumCount,
    ExperimentBudget,
    P99AnchorStatus,
    ScenarioMilliseconds,
    SealedE3aSelection,
    budget_inventory_identity_from_gpu_inventory,
    estimate_industrial_budget,
    materialize_industrial_budgets,
    reduce_e1_activation,
)
from lightcone_spec.experiments.planning_artifacts import (
    PlanningArtifactSidecar,
    budget_load_binding_from_dict,
    budget_load_binding_to_dict,
    budget_plan_from_dict,
    budget_plan_to_dict,
    budget_policy_from_dict,
    budget_policy_to_dict,
    capacity_envelope_from_dict,
    capacity_envelope_to_dict,
)
from lightcone_spec.experiments.registry import (
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    LockedOutput,
    build_industrial_registry,
    content_sha256,
    serving_cell_rejection_reason,
)


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_industrial_registry(
        gpu_uuids=("logical-gpu-a", "logical-gpu-b"),
        base_port=24_000,
        cache_root="runtime-cache/budget-control",
        evidence_root="artifacts/budget-control",
    )


def _sha(label: str) -> str:
    return content_sha256({"budget-control-test": label})


def _scenario(
    optimistic: int, registered: int | None = None, quota: int | None = None
) -> ScenarioMilliseconds:
    return ScenarioMilliseconds(
        optimistic,
        optimistic if registered is None else registered,
        optimistic if quota is None else quota,
    )


def _physical_inventory(size: int) -> GpuInventory:
    group_id = "same-host-fabric"
    uuids = tuple(f"GPU-budget-{index:03d}" for index in range(size))
    devices = tuple(
        GpuDevice(
            uuid=uuid,
            host_id="budget-host",
            model="H100-SXM",
            memory_bytes=80 * 1024**3,
            compute_capability=(9, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root=f"root-{index // 4}",
            numa_node=index // 4,
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
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id=group_id,
                host_id="budget-host",
                gpu_uuids=uuids,
                fabric="NVSwitch",
                bandwidth_class="high",
            ),
        ),
        source_receipt_sha256=_sha(f"inventory-{size}"),
    )


def _interference_envelope(
    inventory: GpuInventory,
    cell: ExperimentCell,
    cardinalities: tuple[int, ...],
) -> InterferenceEnvelope:
    item = registry_pool_work_item(cell, estimated_duration_seconds=1.0)
    rules = tuple(
        sorted(
            (
                InterferenceRule.for_claim(
                    device=inventory.devices[0],
                    claim=item.claim,
                    simultaneous_jobs=cardinality,
                    evidence_sha256=_sha(
                        f"interference-{inventory.sha256}-{cardinality}"
                    ),
                )
                for cardinality in cardinalities
            ),
            key=lambda rule: rule.key,
        )
    )
    return InterferenceEnvelope(
        schema_version=1,
        rules=rules,
        source_receipt_sha256=content_sha256(tuple(rule.sha256 for rule in rules)),
    )


def _target_cells(
    registry: ExperimentRegistry, count: int
) -> tuple[ExperimentCell, ...]:
    cells = tuple(
        cell
        for cell in registry.cells_for("E3a")
        if cell.identity.method == "target_only"
        and serving_cell_rejection_reason(cell) is None
    )
    assert len(cells) >= count
    return cells[:count]


def _estimate_budget(cell: ExperimentCell, inventory_size: int) -> ExperimentBudget:
    wall = _scenario(1_100, 2_200, 3_300)
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=cell.resources.workload_class,
        job_kind=BudgetJobKind.STANDARD,
        startup_model_load=_scenario(100, 200, 300),
        compile_jit_graph_prewarm=ZERO_MILLISECONDS,
        excluded_warmup=ZERO_MILLISECONDS,
        excluded_warmup_requests=ExpectedMaximumCount(0, 0),
        scored_arrival=_scenario(1_000, 2_000, 3_000),
        request_deadline=_scenario(5_000, 5_000, 5_000),
        drain=ZERO_MILLISECONDS,
        reset_finalization=ZERO_MILLISECONDS,
        evidence_flush_shutdown=ZERO_MILLISECONDS,
        output_tokens=ExpectedMaximumCount(32, 32),
        minimum_completed_requests=1,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=ZERO_MILLISECONDS,
        failure_injection=ZERO_MILLISECONDS,
        retry=ZERO_MILLISECONDS,
        retry_allowance=0,
        profiler=ZERO_MILLISECONDS,
        download_compile_reservation=ZERO_MILLISECONDS,
        gpu_count=cell.resources.gpu_count,
        topology=cell.identity.topology,
        reserved_gpu_ms=wall.scale(cell.resources.gpu_count),
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=wall.scale(inventory_size),
    )


@pytest.mark.parametrize("size", (1, 2, 4, 8, 16))
def test_exact_budget_wall_time_uses_arbitrary_size_scheduler(
    registry: ExperimentRegistry, size: int
) -> None:
    inventory = _physical_inventory(size)
    cells = _target_cells(registry, size)
    budgets = tuple(_estimate_budget(cell, size) for cell in cells)
    envelope = _interference_envelope(
        inventory,
        cells[0],
        tuple(range(2, size + 1)),
    )

    report = estimate_industrial_budget(
        registry,
        activated_cell_ids=tuple(cell.cell_id for cell in cells),
        activation_sha256=_sha(f"activation-{size}"),
        budgets=budgets,
        inventory=budget_inventory_identity_from_gpu_inventory(inventory),
        gpu_inventory=inventory,
        interference_envelope=envelope,
    )

    expected_wall = _scenario(1_100, 2_200, 3_300)
    assert report.estimated_wall_ms == expected_wall
    assert report.schedule_fixed_instance_billed_gpu_ms == expected_wall.scale(size)
    assert report.scheduler_gpu_inventory_sha256 == inventory.sha256
    assert report.interference_envelope_sha256 == envelope.sha256
    assert report.unresolved_assumptions == ()


def test_exact_budget_wall_time_honours_serial_and_two_way_interference(
    registry: ExperimentRegistry,
) -> None:
    inventory = _physical_inventory(8)
    cells = _target_cells(registry, 8)
    budgets = tuple(_estimate_budget(cell, 8) for cell in cells)
    identity = budget_inventory_identity_from_gpu_inventory(inventory)
    serial = estimate_industrial_budget(
        registry,
        activated_cell_ids=tuple(cell.cell_id for cell in cells),
        activation_sha256=_sha("serial-activation"),
        budgets=budgets,
        inventory=identity,
        gpu_inventory=inventory,
        interference_envelope=InterferenceEnvelope.serial(
            source_receipt_sha256=_sha("serial-envelope")
        ),
    )
    two_way = estimate_industrial_budget(
        registry,
        activated_cell_ids=tuple(cell.cell_id for cell in cells),
        activation_sha256=_sha("two-way-activation"),
        budgets=budgets,
        inventory=identity,
        gpu_inventory=inventory,
        interference_envelope=_interference_envelope(inventory, cells[0], (2,)),
    )

    wall = _scenario(1_100, 2_200, 3_300)
    assert serial.estimated_wall_ms == wall.scale(8)
    assert two_way.estimated_wall_ms == wall.scale(4)


@pytest.mark.parametrize("size", (2, 8))
def test_budget_report_distinguishes_per_cell_and_schedule_billing(
    registry: ExperimentRegistry, size: int
) -> None:
    inventory = _physical_inventory(size)
    cells = _target_cells(registry, 2)
    budgets = tuple(_estimate_budget(cell, size) for cell in cells)
    report = estimate_industrial_budget(
        registry,
        activated_cell_ids=tuple(cell.cell_id for cell in cells),
        activation_sha256=_sha(f"billing-{size}"),
        budgets=budgets,
        inventory=budget_inventory_identity_from_gpu_inventory(inventory),
        gpu_inventory=inventory,
        interference_envelope=_interference_envelope(inventory, cells[0], (2,)),
    )

    wall = _scenario(1_100, 2_200, 3_300)
    assert report.fixed_instance_billed_gpu_ms == wall.scale(2 * size)
    assert report.schedule_fixed_instance_billed_gpu_ms == wall.scale(size)


def test_budget_estimation_rejects_inventory_tamper_and_missing_authority(
    registry: ExperimentRegistry,
) -> None:
    inventory = _physical_inventory(2)
    cell = _target_cells(registry, 1)[0]
    budget = _estimate_budget(cell, 2)
    identity = budget_inventory_identity_from_gpu_inventory(inventory)
    missing = estimate_industrial_budget(
        registry,
        activated_cell_ids=(cell.cell_id,),
        activation_sha256=_sha("missing-authority"),
        budgets=(budget,),
        inventory=identity,
    )
    assert missing.estimated_wall_ms is None
    assert missing.unresolved_assumptions == (
        (
            "exact_inventory_schedule_unresolved:"
            "full_inventory_and_interference_required"
        ),
    )
    assert missing.scheduler_gpu_inventory_sha256 is None
    assert missing.interference_envelope_sha256 is None

    with pytest.raises(ValueError, match="must be supplied together"):
        estimate_industrial_budget(
            registry,
            activated_cell_ids=(cell.cell_id,),
            activation_sha256=_sha("partial-authority"),
            budgets=(budget,),
            inventory=identity,
            gpu_inventory=inventory,
        )

    with pytest.raises(ValueError, match="differs from the budget inventory"):
        estimate_industrial_budget(
            registry,
            activated_cell_ids=(cell.cell_id,),
            activation_sha256=_sha("tampered-inventory"),
            budgets=(budget,),
            inventory=replace(identity, host_sha256=_sha("forged-host")),
            gpu_inventory=inventory,
            interference_envelope=InterferenceEnvelope.serial(
                source_receipt_sha256=_sha("tamper-envelope")
            ),
        )


def _direct_receipt(
    registry: ExperimentRegistry,
    experiment: str,
    *,
    outputs: dict[str, str],
    runtime_sha256: str,
    split_sha256: str,
) -> ExperimentReceipt:
    definition = registry.definition(experiment)
    return ExperimentReceipt(
        experiment=experiment,
        registry_sha256=registry.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        completed_cells_sha256=_sha(f"{experiment}-completed"),
        dependency_receipts=tuple(
            LockedOutput(name=name, content_sha256=_sha(f"dependency-{name}"))
            for name in definition.dependencies
        ),
        outputs=tuple(
            LockedOutput(name=name, content_sha256=outputs[name])
            for name in sorted(outputs)
        ),
    )


def _e1_activation(registry: ExperimentRegistry):
    selection = SealedE3aSelection(
        schema_version=1,
        registry_sha256=registry.sha256,
        runtime_sha256=_sha("materializer-runtime"),
        split_sha256=_sha("materializer-split"),
        width=8,
        concurrency=4,
        reducer_evidence_sha256=_sha("materializer-evidence"),
    )
    outputs = {
        name: _sha(f"E3a-{name}") for name in registry.definition("E3a").locked_outputs
    }
    outputs["matched_width"] = selection.matched_width_output_sha256
    outputs["e1_reference_load"] = selection.reference_load_output_sha256
    receipt = _direct_receipt(
        registry,
        "E3a",
        outputs=outputs,
        runtime_sha256=selection.runtime_sha256,
        split_sha256=selection.split_sha256,
    )
    return reduce_e1_activation(
        registry,
        e3a_receipt=receipt,
        selection=selection,
    )


def _budget_policy() -> BudgetPolicy:
    rows = []
    for job_kind in sorted(BudgetJobKind, key=lambda value: value.value):
        rows.append(
            BudgetJobPolicy(
                job_kind=job_kind,
                startup_model_load=_scenario(100, 200, 300),
                compile_jit_graph_prewarm=(
                    _scenario(400, 500, 600)
                    if job_kind is BudgetJobKind.COMPILE
                    else ZERO_MILLISECONDS
                ),
                reset_finalization=_scenario(10, 20, 30),
                evidence_flush_shutdown=_scenario(10, 20, 30),
                retry=ZERO_MILLISECONDS,
                retry_allowance=0,
                download_compile_reservation=(
                    _scenario(500, 600, 700)
                    if job_kind is BudgetJobKind.DOWNLOAD
                    else ZERO_MILLISECONDS
                ),
                reserved_gpu_overhead=_scenario(5, 10, 15),
            )
        )
    return BudgetPolicy(
        schema_version=1,
        policy_name="registered-industrial-budget-policy",
        reducer_protocol_sha256=BUDGET_MATERIALIZATION_PROTOCOL_SHA256,
        job_policies=tuple(rows),
    )


def _load_binding(cell_id: str) -> BudgetLoadBinding:
    sampling = FrozenSamplingParameters.from_mapping({"temperature": 0.0, "top_p": 1.0})
    templates = tuple(
        RequestTemplate(
            input_token_ids=(index + 1,),
            requested_output_tokens=32,
            sampling=sampling,
        )
        for index in range(4)
    )
    scored = closed_loop_corpus(
        templates,
        namespace="budget-materializer-score",
        split="tuning",
        concurrency=4,
        cohort_count=1,
        cohort_popularity="uniform",
        cohort_seed=20260811,
    )
    plans = tuple(
        ProductionLoadPlan(
            warmup=None,
            scored=scored,
            window=ProductionWindow(
                warmup_duration_us=0,
                arrival_duration_us=arrival_ms * 1_000,
                request_deadline_us=deadline_ms * 1_000,
                drain_duration_us=drain_ms * 1_000,
            ),
        )
        for arrival_ms, deadline_ms, drain_ms in (
            (1_000, 5_000, 100),
            (2_000, 6_000, 200),
            (3_000, 7_000, 300),
        )
    )
    return BudgetLoadBinding(
        cell_id=cell_id,
        job_kind=BudgetJobKind.STANDARD,
        optimistic_load=plans[0],
        registered_load=plans[1],
        quota_envelope_load=plans[2],
        minimum_completed_requests=4,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
    )


def _budget_inventory() -> BudgetInventoryIdentity:
    return BudgetInventoryIdentity(
        schema_version=1,
        host_sha256=_sha("materializer-host"),
        gpu_uuids=("GPU-materializer-a", "GPU-materializer-b"),
        topology_sha256=_sha("materializer-topology"),
    )


def _capacity_envelope(
    activation,
    inventory: BudgetInventoryIdentity,
    *,
    provider_quota_gpu_ms: int = 10**15,
    host_free_bytes: int = 10**15,
    host_quota_bytes: int = 10**15,
) -> CapacityEnvelope:
    return CapacityEnvelope(
        schema_version=1,
        budget_inventory_sha256=inventory.sha256,
        provider_quota_gpu_ms=provider_quota_gpu_ms,
        host_free_bytes=host_free_bytes,
        host_quota_bytes=host_quota_bytes,
        cell_requirements=tuple(
            CellCapacityRequirement(
                cell_id=cell_id,
                maximum_evidence_bytes=1_000,
                model_staging_bytes=2_000,
                compile_overlay_bytes=3_000,
            )
            for cell_id in activation.plan.activated_cell_ids
        ),
        source_receipt_sha256=_sha(
            f"capacity-{provider_quota_gpu_ms}-{host_free_bytes}-{host_quota_bytes}"
        ),
    )


def test_materializer_retains_diagnostic_budgets_without_raw_capacity_authority(
    registry: ExperimentRegistry,
) -> None:
    activation = _e1_activation(registry)
    activated_count = len(activation.plan.activated_cell_ids)
    assert activated_count > 1
    incomplete_count = activated_count - 1
    bindings = tuple(
        _load_binding(cell_id) for cell_id in activation.plan.activated_cell_ids
    )
    policy = _budget_policy()
    inventory = _budget_inventory()
    capacity = _capacity_envelope(activation, inventory)
    plan = materialize_industrial_budgets(
        registry,
        activations=(activation,),
        load_bindings=bindings,
        policy=policy,
        inventory=inventory,
        capacity_envelope=capacity,
    )

    assert plan.status == "UNRESOLVED"
    assert len(plan.budgets) == activated_count
    assert plan.diagnostic_budgets == plan.budgets
    assert len(plan.dispositions) == activated_count
    assert {row.reason_code for row in plan.dispositions} == {
        "capacity_raw_authority_missing"
    }
    assert all(
        row.status is BudgetDispositionStatus.UNRESOLVED for row in plan.dispositions
    )
    with pytest.raises(ValueError, match="capacity_raw_authority_missing"):
        plan.require_ready()
    first = plan.budgets[0]
    assert first.scored_arrival == _scenario(1_000, 2_000, 3_000)
    assert first.request_deadline == _scenario(5_000, 6_000, 7_000)
    assert first.output_tokens == ExpectedMaximumCount(128, 128)
    assert first.minimum_completed_requests == 4
    assert first.fixed_instance_billed_gpu_ms == first.wall_time.scale(2)
    assert first.reserved_gpu_ms == first.compute_gpu_ms + _scenario(5, 10, 15)

    incomplete = materialize_industrial_budgets(
        registry,
        activations=(activation,),
        load_bindings=bindings[:-1],
        policy=policy,
        inventory=inventory,
        capacity_envelope=capacity,
    )
    assert incomplete.status == "UNRESOLVED"
    assert len(incomplete.diagnostic_budgets) == incomplete_count
    assert (
        sum(
            row.reason_code == "missing_load_semantics"
            for row in incomplete.dispositions
        )
        == 1
    )
    assert (
        sum(
            row.reason_code == "capacity_budget_coverage_incomplete"
            for row in incomplete.dispositions
        )
        == incomplete_count
    )
    with pytest.raises(ValueError, match="contains unresolved cells"):
        materialize_industrial_budgets(
            registry,
            activations=(activation,),
            load_bindings=bindings[:-1],
            policy=policy,
            inventory=inventory,
            capacity_envelope=capacity,
            require_complete=True,
        )

    missing_capacity = materialize_industrial_budgets(
        registry,
        activations=(activation,),
        load_bindings=bindings,
        policy=policy,
        inventory=inventory,
    )
    assert len(missing_capacity.diagnostic_budgets) == activated_count
    assert {row.reason_code for row in missing_capacity.dispositions} == {
        "capacity_envelope_missing"
    }
    partial_capacity = materialize_industrial_budgets(
        registry,
        activations=(activation,),
        load_bindings=bindings,
        policy=policy,
        inventory=inventory,
        capacity_envelope=replace(
            capacity,
            cell_requirements=capacity.cell_requirements[:-1],
        ),
    )
    assert len(partial_capacity.diagnostic_budgets) == activated_count
    assert (
        sum(
            row.reason_code == "capacity_requirement_missing"
            for row in partial_capacity.dispositions
        )
        == 1
    )
    assert (
        sum(
            row.reason_code == "capacity_requirement_coverage_incomplete"
            for row in partial_capacity.dispositions
        )
        == incomplete_count
    )


def test_capacity_envelope_rejects_provider_and_maximum_attempt_disk_overflow(
    registry: ExperimentRegistry,
) -> None:
    activation = _e1_activation(registry)
    activated_count = len(activation.plan.activated_cell_ids)
    assert activated_count > 0
    bindings = tuple(
        _load_binding(cell_id) for cell_id in activation.plan.activated_cell_ids
    )
    inventory = _budget_inventory()
    policy = _budget_policy()
    provider_blocked = materialize_industrial_budgets(
        registry,
        activations=(activation,),
        load_bindings=bindings,
        policy=policy,
        inventory=inventory,
        capacity_envelope=_capacity_envelope(
            activation,
            inventory,
            provider_quota_gpu_ms=0,
        ),
    )
    assert len(provider_blocked.diagnostic_budgets) == activated_count
    assert {row.reason_code for row in provider_blocked.dispositions} == {
        "capacity_provider_quota_exceeded"
    }

    retried_standard = replace(
        policy.for_job(BudgetJobKind.STANDARD),
        retry=_scenario(10, 20, 30),
        retry_allowance=1,
    )
    retried_policy = replace(
        policy,
        job_policies=tuple(
            retried_standard if row.job_kind is BudgetJobKind.STANDARD else row
            for row in policy.job_policies
        ),
    )
    retried_diagnostic = materialize_industrial_budgets(
        registry,
        activations=(activation,),
        load_bindings=bindings,
        policy=retried_policy,
        inventory=inventory,
        capacity_envelope=_capacity_envelope(activation, inventory),
    )
    one_attempt_gpu_ms = sum(
        budget.fixed_instance_billed_gpu_ms.quota_envelope
        for budget in retried_diagnostic.diagnostic_budgets
    )
    retry_quota_blocked = materialize_industrial_budgets(
        registry,
        activations=(activation,),
        load_bindings=bindings,
        policy=retried_policy,
        inventory=inventory,
        capacity_envelope=_capacity_envelope(
            activation,
            inventory,
            provider_quota_gpu_ms=one_attempt_gpu_ms,
        ),
    )
    assert {row.reason_code for row in retry_quota_blocked.dispositions} == {
        "capacity_provider_quota_exceeded"
    }
    one_attempt_bytes = len(bindings) * (1_000 + 2_000 + 3_000)
    disk_blocked = materialize_industrial_budgets(
        registry,
        activations=(activation,),
        load_bindings=bindings,
        policy=retried_policy,
        inventory=inventory,
        capacity_envelope=_capacity_envelope(
            activation,
            inventory,
            host_free_bytes=10**15,
            host_quota_bytes=one_attempt_bytes,
        ),
    )
    assert len(disk_blocked.diagnostic_budgets) == activated_count
    assert {row.reason_code for row in disk_blocked.dispositions} == {
        "capacity_host_disk_exceeded"
    }


def test_budget_policy_and_plan_strict_serialization_and_tamper_rejection(
    registry: ExperimentRegistry,
) -> None:
    activation = _e1_activation(registry)
    policy = _budget_policy()
    standard_policy = policy.for_job(BudgetJobKind.STANDARD)
    with pytest.raises(ValueError, match="retry duration and allowance"):
        replace(standard_policy, retry=_scenario(0, 0, 1))
    with pytest.raises(ValueError, match="only for the compile policy"):
        replace(
            standard_policy,
            compile_jit_graph_prewarm=_scenario(1, 1, 1),
        )
    inventory = _budget_inventory()
    capacity = _capacity_envelope(activation, inventory)
    plan = materialize_industrial_budgets(
        registry,
        activations=(activation,),
        load_bindings=tuple(
            _load_binding(cell_id) for cell_id in activation.plan.activated_cell_ids
        ),
        policy=policy,
        inventory=inventory,
        capacity_envelope=capacity,
    )

    policy_wire = budget_policy_to_dict(policy)
    assert (
        budget_policy_from_dict(
            policy_wire,
            sidecar=PlanningArtifactSidecar(1, "budget_policy", policy.sha256),
        )
        == policy
    )
    capacity_wire = capacity_envelope_to_dict(capacity)
    assert (
        capacity_envelope_from_dict(
            capacity_wire,
            sidecar=PlanningArtifactSidecar(1, "capacity_envelope", capacity.sha256),
        )
        == capacity
    )
    binding = _load_binding(activation.plan.activated_cell_ids[0])
    binding_wire = budget_load_binding_to_dict(binding)
    assert (
        budget_load_binding_from_dict(
            binding_wire,
            sidecar=PlanningArtifactSidecar(1, "budget_load_binding", binding.sha256),
        )
        == binding
    )
    tampered_capacity = deepcopy(capacity_wire)
    tampered_capacity["cell_requirements"][0]["maximum_evidence_bytes"] += 1
    with pytest.raises(ValueError, match="redundant artifact SHA-256 mismatch"):
        capacity_envelope_from_dict(tampered_capacity)
    plan_wire = budget_plan_to_dict(plan)
    assert (
        budget_plan_from_dict(
            plan_wire,
            sidecar=PlanningArtifactSidecar(1, "budget_plan", plan.sha256),
        )
        == plan
    )

    extra = deepcopy(plan_wire)
    extra["unregistered_field"] = 1
    with pytest.raises(ValueError, match="fields differ"):
        budget_plan_from_dict(extra)

    tampered = deepcopy(plan_wire)
    tampered["budgets"][0]["scored_arrival"]["registered"] += 1
    with pytest.raises(ValueError):
        budget_plan_from_dict(tampered)

    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        budget_plan_from_dict(
            plan_wire,
            sidecar=PlanningArtifactSidecar(1, "budget_plan", _sha("forged-plan")),
        )
