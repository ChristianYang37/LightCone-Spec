"""Fail-closed binding from an industrial cell to rank-local run configs.

The industrial registry intentionally contains scientific and scheduling data,
while :class:`~lightcone_spec.config.schema.RunConfig` is the native serving
contract.  This module joins those two identities without choosing values for
the caller.  In particular, it only accepts fully materialised configs and
rejects registry placeholders that still require an upstream locked output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from lightcone_spec.adaptation.parameters import TrainablePlan
from lightcone_spec.config.schema import RunConfig
from lightcone_spec.experiments.gpu_pool import (
    GpuAssignment,
    GpuDispatchExecutionContext,
    GpuDispatchPlan,
    GpuInventory,
    registry_pool_work_item,
    validate_dispatch_plan_for_execution,
)
from lightcone_spec.experiments.planning import ExperimentBudget
from lightcone_spec.experiments.registry import (
    INDUSTRIAL_EXPERIMENT_ORDER,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    WorkloadClass,
    content_sha256,
    serving_cell_rejection_reason,
)
from lightcone_spec.runtime.distributed import TopologyReceiptSet

_TOPOLOGIES = {
    "tp1_dp1": (1, 1),
    "tp2_dp1": (2, 1),
    "two_replica_tp1_dp2": (1, 2),
}


@dataclass(frozen=True)
class IndustrialPhysicalAssignment:
    """Content-bound physical resources for one scheduled registry cell."""

    inventory_sha256: str
    inventory_source_receipt_sha256: str
    dispatch_plan_sha256: str
    experiment_budget_sha256: str
    budget_plan_sha256: str
    capacity_authority_sha256: str
    budget_materialization_authority_sha256: str
    assignment_sha256: str
    work_item_sha256: str
    gpu_uuids: tuple[str, ...]
    rank_groups: tuple[tuple[str, ...], ...]
    ports: tuple[int, ...]
    tensor_parallel_size: int
    data_parallel_size: int
    fixed_instance_gpu_count: int
    host_id: str
    topology_group_ids: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        for name in (
            "inventory_sha256",
            "inventory_source_receipt_sha256",
            "dispatch_plan_sha256",
            "experiment_budget_sha256",
            "budget_plan_sha256",
            "capacity_authority_sha256",
            "budget_materialization_authority_sha256",
            "assignment_sha256",
            "work_item_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be a lower-case SHA-256")
        if (
            not isinstance(self.host_id, str)
            or not self.host_id
            or self.host_id.strip() != self.host_id
        ):
            raise ValueError("physical assignment host_id must be canonical")
        if (
            isinstance(self.tensor_parallel_size, bool)
            or not isinstance(self.tensor_parallel_size, int)
            or self.tensor_parallel_size < 1
            or isinstance(self.data_parallel_size, bool)
            or not isinstance(self.data_parallel_size, int)
            or self.data_parallel_size < 1
        ):
            raise ValueError(
                "physical assignment TP/DP sizes must be positive integers"
            )
        world_size = self.tensor_parallel_size * self.data_parallel_size
        if (
            isinstance(self.fixed_instance_gpu_count, bool)
            or not isinstance(self.fixed_instance_gpu_count, int)
            or self.fixed_instance_gpu_count < world_size
        ):
            raise ValueError(
                "fixed-instance GPU count must cover the complete assigned gang"
            )
        if (
            len(self.gpu_uuids) != world_size
            or len(set(self.gpu_uuids)) != world_size
            or any(not isinstance(uuid, str) or not uuid for uuid in self.gpu_uuids)
        ):
            raise ValueError("physical assignment must bind one unique GPU per rank")
        expected_groups = tuple(
            self.gpu_uuids[
                replica * self.tensor_parallel_size : (replica + 1)
                * self.tensor_parallel_size
            ]
            for replica in range(self.data_parallel_size)
        )
        if self.rank_groups != expected_groups:
            raise ValueError("physical assignment rank groups differ from TP/DP order")
        if len(self.topology_group_ids) != self.data_parallel_size:
            raise ValueError(
                "physical assignment topology-group coverage is incomplete"
            )
        if any(
            len(group_ids) != len(set(group_ids))
            or tuple(sorted(group_ids)) != group_ids
            or any(
                not isinstance(group_id, str) or not group_id for group_id in group_ids
            )
            for group_ids in self.topology_group_ids
        ):
            raise ValueError(
                "physical assignment topology-group identities are invalid"
            )
        if self.tensor_parallel_size > 1 and any(
            not group_ids for group_ids in self.topology_group_ids
        ):
            raise ValueError("physical TP groups require topology-group identities")
        if (
            not self.ports
            or len(set(self.ports)) != len(self.ports)
            or tuple(sorted(self.ports)) != self.ports
            or any(
                isinstance(port, bool)
                or not isinstance(port, int)
                or port < 1024
                or port > 65_535
                for port in self.ports
            )
        ):
            raise ValueError("physical assignment ports must be canonical and unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "kind": "industrial_physical_assignment",
            "inventory_sha256": self.inventory_sha256,
            "inventory_source_receipt_sha256": self.inventory_source_receipt_sha256,
            "dispatch_plan_sha256": self.dispatch_plan_sha256,
            "experiment_budget_sha256": self.experiment_budget_sha256,
            "budget_plan_sha256": self.budget_plan_sha256,
            "capacity_authority_sha256": self.capacity_authority_sha256,
            "budget_materialization_authority_sha256": (
                self.budget_materialization_authority_sha256
            ),
            "assignment_sha256": self.assignment_sha256,
            "work_item_sha256": self.work_item_sha256,
            "gpu_uuids": list(self.gpu_uuids),
            "rank_groups": [list(group) for group in self.rank_groups],
            "ports": list(self.ports),
            "gang_shape": {
                "tensor_parallel_size": self.tensor_parallel_size,
                "data_parallel_size": self.data_parallel_size,
            },
            "fixed_instance_gpu_count": self.fixed_instance_gpu_count,
            "fixed_instance_billing_semantics": "whole_inventory_wall_clock_v1",
            "host_id": self.host_id,
            "topology_group_ids": [
                list(group_ids) for group_ids in self.topology_group_ids
            ],
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class IndustrialRuntimePlan:
    """Content-addressed serving contract for exactly one registry cell."""

    registry_sha256: str
    cell_id: str
    cell_declaration_sha256: str
    dependency_receipt_sha256s: tuple[str, ...]
    topology_receipt_sha256: str
    parameter_plan_sha256: str | None
    rank_configs: tuple[RunConfig, ...]
    cell: ExperimentCell
    physical_assignment: IndustrialPhysicalAssignment | None = None

    @property
    def physical_dispatch_ready(self) -> bool:
        return self.physical_assignment is not None

    def _require_physical_assignment(self) -> IndustrialPhysicalAssignment:
        if self.physical_assignment is None:
            raise ValueError(
                "logical runtime plan has no physical GPU assignment; "
                "use render_assigned_industrial_cell_runtime_plan"
            )
        return self.physical_assignment

    @property
    def physical_gpu_uuids(self) -> tuple[str, ...]:
        """Return assigned UUIDs without falling back to logical registry slots."""

        return self._require_physical_assignment().gpu_uuids

    @property
    def physical_rank_groups(self) -> tuple[tuple[str, ...], ...]:
        """Return the exact assigned TP/DP rank partition."""

        return self._require_physical_assignment().rank_groups

    @property
    def physical_ports(self) -> tuple[int, ...]:
        """Return assigned ports without falling back to logical registry slots."""

        return self._require_physical_assignment().ports

    @property
    def physical_fixed_instance_gpu_count(self) -> int:
        """Return the exact whole-instance GPU count used for billed time."""

        return self._require_physical_assignment().fixed_instance_gpu_count

    def to_dict(self) -> dict[str, Any]:
        identity = self.cell.identity
        resources = self.cell.resources
        logical_resources = {
            "gpu_uuids": list(resources.gpu_uuids),
            "ports": list(resources.ports),
            "cache_root": resources.cache_root,
            "evidence_root": resources.evidence_root,
            "workload_class": resources.workload_class.value,
            "exclusive": resources.exclusive,
        }
        return {
            "schema_version": 2,
            "registry_sha256": self.registry_sha256,
            "cell_id": self.cell_id,
            "cell_declaration_sha256": self.cell_declaration_sha256,
            "dependency_receipt_sha256s": list(self.dependency_receipt_sha256s),
            "topology_receipt_sha256": self.topology_receipt_sha256,
            "parameter_plan_sha256": self.parameter_plan_sha256,
            "rank_config_sha256s": [
                content_sha256(config.model_dump(mode="json"))
                for config in self.rank_configs
            ],
            "rank_configs": [
                config.model_dump(mode="json") for config in self.rank_configs
            ],
            "workload": {
                "experiment": identity.experiment,
                "task": identity.task,
                "context": identity.context,
                "regime": identity.regime,
                "width": identity.width,
                "arrival": identity.arrival,
                "slo": identity.slo,
                "cohort": identity.cohort,
                "seed": identity.seed,
                "block": identity.block,
                "variant": identity.variant,
                "concurrency": identity.concurrency,
                "load_factor": identity.load_factor,
                "cohort_count": identity.cohort_count,
            },
            # ``resources`` remains as a compatibility alias.  New dispatch
            # consumers must use the explicit logical/physical fields below.
            "resources": dict(logical_resources),
            "logical_resources": logical_resources,
            "physical_gpu_uuids": (
                None
                if self.physical_assignment is None
                else list(self.physical_gpu_uuids)
            ),
            "physical_rank_groups": (
                None
                if self.physical_assignment is None
                else [list(group) for group in self.physical_rank_groups]
            ),
            "physical_ports": (
                None if self.physical_assignment is None else list(self.physical_ports)
            ),
            "physical_fixed_instance_gpu_count": (
                None
                if self.physical_assignment is None
                else self.physical_fixed_instance_gpu_count
            ),
            "resource_binding": {
                "kind": (
                    "gpu_assignment"
                    if self.physical_assignment is not None
                    else "registry_logical_only"
                ),
                "physical_dispatch_ready": self.physical_assignment is not None,
                "physical_assignment_sha256": (
                    None
                    if self.physical_assignment is None
                    else self.physical_assignment.assignment_sha256
                ),
                "physical_binding_sha256": (
                    None
                    if self.physical_assignment is None
                    else self.physical_assignment.sha256
                ),
                "physical_assignment": (
                    None
                    if self.physical_assignment is None
                    else self.physical_assignment.to_dict()
                ),
            },
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def _cell_by_id(registry: ExperimentRegistry, cell_id: str) -> ExperimentCell:
    matches = tuple(cell for cell in registry.cells if cell.cell_id == cell_id)
    if len(matches) != 1:
        raise ValueError("cell_id must identify exactly one registry cell")
    return matches[0]


def _require_fully_materialised(value: BaseModel, path: str) -> None:
    fields = set(type(value).model_fields)
    missing = fields - value.model_fields_set
    if missing:
        raise ValueError(
            f"{path} must explicitly materialise every field; missing {sorted(missing)}"
        )


def _validate_explicit_configs(configs: tuple[RunConfig, ...]) -> None:
    for rank, config in enumerate(configs):
        prefix = f"rank_configs[{rank}]"
        _require_fully_materialised(config, prefix)
        _require_fully_materialised(config.model, f"{prefix}.model")
        _require_fully_materialised(config.runtime, f"{prefix}.runtime")
        if config.adaptation is not None:
            _require_fully_materialised(config.adaptation, f"{prefix}.adaptation")
            _require_fully_materialised(
                config.adaptation.optimizer,
                f"{prefix}.adaptation.optimizer",
            )
        if config.online_spec is not None:
            _require_fully_materialised(
                config.online_spec,
                f"{prefix}.online_spec",
            )
        try:
            RunConfig.model_validate(config.model_dump(mode="json"))
        except ValueError as error:
            raise ValueError(
                f"{prefix} does not satisfy the current strict RunConfig schema"
            ) from error


def _dependency_chain(
    registry: ExperimentRegistry,
    experiment: str,
    receipts: tuple[ExperimentReceipt, ...],
) -> tuple[ExperimentReceipt, ...]:
    stage = INDUSTRIAL_EXPERIMENT_ORDER.index(experiment)
    expected = INDUSTRIAL_EXPERIMENT_ORDER[:stage]
    if len(receipts) != len(expected):
        raise ValueError(
            "runtime rendering requires the complete preceding receipt chain"
        )
    by_experiment = registry.validate_receipts(receipts)
    if tuple(by_experiment) != expected:
        raise ValueError(
            "runtime rendering requires the exact ordered preceding receipt chain"
        )
    return tuple(by_experiment[name] for name in expected)


def _locked_output(receipt: ExperimentReceipt, name: str) -> str:
    matches = [row.content_sha256 for row in receipt.outputs if row.name == name]
    if len(matches) != 1:
        raise ValueError(f"receipt lacks exactly one locked output {name!r}")
    return matches[0]


def _reject_unresolved_cell(cell: ExperimentCell) -> None:
    reason = serving_cell_rejection_reason(cell)
    if reason is not None:
        raise ValueError(reason)


def bind_industrial_gpu_assignment(
    *,
    registry: ExperimentRegistry,
    cell_id: str,
    assignment: GpuAssignment,
    dispatch_plan: GpuDispatchPlan,
    dispatch_context: GpuDispatchExecutionContext,
    budget: ExperimentBudget,
    inventory: GpuInventory,
    dispatch_inventory_sha256: str,
    topology_receipts: TopologyReceiptSet,
) -> IndustrialPhysicalAssignment:
    """Validate and bind one scheduler assignment without allocating resources.

    The registry GPU names and ports are logical sharding declarations.  The
    assignment is the authority for rank-local physical UUIDs and ports, but
    only after its embedded cell, canonical resource claim, inventory identity,
    gang, and observed rank topology have all been checked here.
    """

    if not isinstance(assignment, GpuAssignment):
        raise TypeError("assignment must be a GpuAssignment")
    if not isinstance(dispatch_plan, GpuDispatchPlan):
        raise TypeError("dispatch_plan must be a GpuDispatchPlan")
    if not isinstance(dispatch_context, GpuDispatchExecutionContext):
        raise TypeError("dispatch_context must be a GpuDispatchExecutionContext")
    if not isinstance(budget, ExperimentBudget):
        raise TypeError("budget must be an ExperimentBudget")
    if not isinstance(inventory, GpuInventory):
        raise TypeError("inventory must be a GpuInventory")
    if dispatch_inventory_sha256 != inventory.sha256:
        raise ValueError("dispatch inventory SHA-256 differs from the inventory")
    if dispatch_context.registry != registry or dispatch_context.inventory != inventory:
        raise ValueError("dispatch context belongs to another registry or inventory")
    validate_dispatch_plan_for_execution(
        dispatch_plan, execution_context=dispatch_context
    )
    if len(inventory.host_ids) != 1:
        raise ValueError("physical dispatch requires one same-host inventory")

    cell = _cell_by_id(registry, cell_id)
    _reject_unresolved_cell(cell)
    if (
        dispatch_plan.registry_sha256 != registry.sha256
        or dispatch_plan.inventory_sha256 != inventory.sha256
    ):
        raise ValueError("dispatch plan belongs to another registry or inventory")
    if not dispatch_plan.scientific_budget_bound:
        raise ValueError("dispatch plan lacks exact ExperimentBudget bindings")
    assignment_matches = tuple(
        candidate
        for wave in dispatch_plan.waves
        for candidate in wave.assignments
        if candidate.assignment_id == assignment.assignment_id
    )
    if assignment_matches != (assignment,):
        raise ValueError(
            "assignment is absent, duplicated, or changed in dispatch plan"
        )
    if dispatch_plan.budget_sha256_for(cell_id) != budget.sha256:
        raise ValueError("ExperimentBudget differs from the dispatch-plan binding")
    ready_budgets = dispatch_context.require_ready_budget_authority()
    if {row.cell_id: row for row in ready_budgets}.get(cell_id) != budget:
        raise ValueError("ExperimentBudget differs from the READY BudgetPlan")
    context_budget = dispatch_context.budgets_by_cell_id.get(cell_id)
    if context_budget != budget:
        raise ValueError("ExperimentBudget differs from the dispatch context")
    if budget.fixed_instance_billed_gpu_ms != budget.wall_time.scale(
        len(inventory.devices)
    ):
        raise ValueError(
            "ExperimentBudget does not bind whole-inventory fixed-instance billing"
        )
    if (
        budget.cell_id != cell.cell_id
        or budget.experiment != cell.identity.experiment
        or budget.method != cell.identity.method
        or budget.workload_class is not cell.resources.workload_class
        or budget.gpu_count != cell.resources.gpu_count
        or budget.topology != cell.identity.topology
        or budget.measured_gpu_ms is not None
    ):
        raise ValueError("ExperimentBudget differs from its registry cell")
    work_item = assignment.work_item
    if work_item.item_id != cell_id:
        raise ValueError("assignment work item identifies another registry cell")
    if work_item.cell != cell or work_item.cell.sha256 != cell.sha256:
        raise ValueError("assignment work-item cell differs from the registry cell")
    canonical_work_item = registry_pool_work_item(
        cell,
        estimated_duration_seconds=work_item.claim.estimated_duration_seconds,
    )
    if work_item != canonical_work_item:
        raise ValueError("assignment work item changes the canonical resource claim")
    if work_item.sha256 != canonical_work_item.sha256:
        raise ValueError("assignment work-item SHA-256 is stale or forged")

    claim = work_item.claim
    tp_size, dp_size = _TOPOLOGIES[cell.identity.topology]
    if (
        claim.gang_shape.tensor_parallel_size != tp_size
        or claim.gang_shape.data_parallel_size != dp_size
    ):
        raise ValueError("assignment gang shape differs from the registry topology")
    world_size = tp_size * dp_size
    if cell.resources.gpu_count != world_size:
        raise ValueError("cell GPU reservation does not match its topology")
    if type(assignment.gpu_uuids) is not tuple or any(
        not isinstance(uuid, str) or not uuid for uuid in assignment.gpu_uuids
    ):
        raise TypeError("assignment GPU UUIDs must be a tuple of identifiers")
    if type(assignment.rank_groups) is not tuple or any(
        type(group) is not tuple for group in assignment.rank_groups
    ):
        raise TypeError("assignment rank groups must be tuples")
    if type(assignment.ports) is not tuple:
        raise TypeError("assignment ports must be a tuple")
    if (
        len(assignment.gpu_uuids) != world_size
        or len(set(assignment.gpu_uuids)) != world_size
    ):
        raise ValueError("assignment must cover the complete unique GPU gang")
    expected_rank_groups = tuple(
        assignment.gpu_uuids[replica * tp_size : (replica + 1) * tp_size]
        for replica in range(dp_size)
    )
    if assignment.rank_groups != expected_rank_groups:
        raise ValueError("assignment rank groups do not match the ordered TP/DP gang")
    if (
        len(assignment.ports) != claim.port_count
        or len(set(assignment.ports)) != claim.port_count
        or any(
            isinstance(port, bool)
            or not isinstance(port, int)
            or port < 1024
            or port > 65_535
            for port in assignment.ports
        )
    ):
        raise ValueError("assignment ports do not cover the complete valid port set")
    if claim.exact_ports:
        if assignment.ports != claim.exact_ports:
            raise ValueError("assignment differs from the exact requested ports")
    elif assignment.ports != tuple(sorted(assignment.ports)):
        raise ValueError("fungible assignment ports must use canonical ascending order")
    if claim.exact_gpu_uuids and assignment.gpu_uuids != claim.exact_gpu_uuids:
        raise ValueError("assignment differs from the exact requested GPU UUIDs")

    inventory_devices = {device.uuid: device for device in inventory.devices}
    if set(assignment.gpu_uuids) - set(inventory_devices):
        raise ValueError("assignment references a GPU outside the dispatch inventory")
    devices = tuple(inventory_devices[uuid] for uuid in assignment.gpu_uuids)
    if any(not claim.homogeneous.accepts(device) for device in devices):
        raise ValueError("assigned GPU is not ready or violates its capability claim")
    host_ids = {device.host_id for device in devices}
    if len(host_ids) != 1:
        raise ValueError("assignment gang crosses physical hosts")
    host_id = next(iter(host_ids))
    if host_id != inventory.host_ids[0]:
        raise ValueError("assignment belongs to a foreign inventory host")
    if len({device.hardware_envelope_sha256 for device in devices}) != 1:
        raise ValueError("assignment gang crosses hardware envelopes")

    topology_group_ids: list[tuple[str, ...]] = []
    for rank_group in assignment.rank_groups:
        if tp_size == 1:
            topology_group_ids.append(())
            continue
        rank_set = set(rank_group)
        eligible = tuple(
            group.group_id
            for group in inventory.topology_groups
            if group.host_id == host_id
            and rank_set <= set(group.gpu_uuids)
            and (
                not claim.allowed_topology_groups
                or group.group_id in claim.allowed_topology_groups
            )
            and (not claim.allowed_fabrics or group.fabric in claim.allowed_fabrics)
        )
        if not eligible:
            raise ValueError("assigned TP rank group lacks an allowed topology group")
        topology_group_ids.append(eligible)

    if (
        topology_receipts.tensor_parallel_size != tp_size
        or topology_receipts.data_parallel_size != dp_size
        or topology_receipts.world_size != world_size
    ):
        raise ValueError("topology receipts disagree with the assigned gang shape")
    ordered_receipts = tuple(
        topology_receipts.receipt_for_rank(rank) for rank in range(world_size)
    )
    receipt_devices = tuple(row.topology.device_id for row in ordered_receipts)
    if receipt_devices != assignment.gpu_uuids:
        raise ValueError("topology receipts do not bind the assigned GPU rank order")
    receipt_rank_groups = tuple(
        receipt_devices[replica * tp_size : (replica + 1) * tp_size]
        for replica in range(dp_size)
    )
    if receipt_rank_groups != assignment.rank_groups:
        raise ValueError("topology receipts do not bind the assigned rank groups")
    for rank, receipt in enumerate(ordered_receipts):
        topology = receipt.topology
        if (
            topology.node_count != 1
            or topology.node_rank != 0
            or topology.node_id != host_id
            or topology.local_rank != rank
        ):
            raise ValueError(
                "topology receipts do not bind the assigned same-host local ranks"
            )

    canonical_assignment_sha256 = content_sha256(
        {
            "work_item": canonical_work_item.to_dict(),
            "work_item_sha256": canonical_work_item.sha256,
            "gpu_uuids": list(assignment.gpu_uuids),
            "rank_groups": [list(group) for group in assignment.rank_groups],
            "ports": list(assignment.ports),
        }
    )
    if assignment.sha256 != canonical_assignment_sha256:
        raise ValueError("assignment SHA-256 is stale or forged")
    if budget.fixed_instance_billed_gpu_ms != budget.wall_time.scale(
        len(inventory.devices)
    ):
        raise ValueError("physical assignment budget undercounts the fixed inventory")
    capacity_authority = dispatch_context.budget_plan.capacity_authority
    if capacity_authority is None:  # pragma: no cover - execution-context invariant
        raise RuntimeError("execution capacity authority disappeared")
    return IndustrialPhysicalAssignment(
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        dispatch_plan_sha256=dispatch_plan.sha256,
        experiment_budget_sha256=budget.sha256,
        budget_plan_sha256=dispatch_context.budget_plan.sha256,
        capacity_authority_sha256=capacity_authority.sha256,
        budget_materialization_authority_sha256=(
            dispatch_context.budget_materialization_authority.sha256
        ),
        assignment_sha256=canonical_assignment_sha256,
        work_item_sha256=canonical_work_item.sha256,
        gpu_uuids=assignment.gpu_uuids,
        rank_groups=assignment.rank_groups,
        ports=assignment.ports,
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        fixed_instance_gpu_count=len(inventory.devices),
        host_id=host_id,
        topology_group_ids=tuple(topology_group_ids),
    )


def _validate_topology(
    cell: ExperimentCell,
    configs: tuple[RunConfig, ...],
    topology: TopologyReceiptSet,
    dependency_chain: tuple[ExperimentReceipt, ...],
    *,
    expected_gpu_uuids: tuple[str, ...] | None = None,
    expected_host_id: str | None = None,
) -> None:
    tp_size, dp_size = _TOPOLOGIES[cell.identity.topology]
    world_size = tp_size * dp_size
    if len(configs) != world_size or topology.world_size != world_size:
        raise ValueError("rank configs and topology receipts must cover every rank")
    if (
        topology.tensor_parallel_size != tp_size
        or topology.data_parallel_size != dp_size
    ):
        raise ValueError("topology receipt disagrees with the registry topology")
    if cell.resources.gpu_count != world_size:
        raise ValueError("cell GPU reservation does not match its topology")
    expected_port_count = 1 if world_size == 1 else 3
    if len(cell.resources.ports) != expected_port_count:
        raise ValueError("cell does not reserve the exact serving/router port set")
    ordered_receipts = tuple(
        topology.receipt_for_rank(rank) for rank in range(world_size)
    )
    expected_devices = tuple(receipt.topology.device_id for receipt in ordered_receipts)
    bound_devices = (
        cell.identity.gpu_uuids if expected_gpu_uuids is None else expected_gpu_uuids
    )
    if expected_devices != bound_devices:
        raise ValueError("topology receipts do not bind the expected GPU UUID order")
    if expected_host_id is not None and any(
        receipt.topology.node_id != expected_host_id for receipt in ordered_receipts
    ):
        raise ValueError("topology receipts do not bind the assigned inventory host")

    capability_sha256 = None
    if world_size > 1:
        if not dependency_chain or dependency_chain[0].experiment != "preflight":
            raise ValueError("distributed cells require the preflight receipt")
        capability_sha256 = _locked_output(
            dependency_chain[0],
            "runtime_envelope",
        )

    normalized: list[dict[str, Any]] = []
    for global_rank, (config, rank_receipt) in enumerate(
        zip(configs, ordered_receipts, strict=True)
    ):
        runtime = config.runtime
        rank_topology = rank_receipt.topology
        if (
            runtime.tensor_parallel_size != tp_size
            or runtime.data_parallel_size != dp_size
            or runtime.tp_rank != rank_topology.tensor_parallel_rank
            or runtime.dp_rank != rank_topology.data_parallel_rank
            or runtime.node_count != rank_topology.node_count
            or runtime.node_rank != rank_topology.node_rank
            or runtime.device_identity != rank_topology.device_id
            or runtime.rendezvous_identity != rank_topology.rendezvous_id
            or runtime.router_identity != rank_topology.router_id
            or runtime.clock_identity != rank_topology.clock_id
        ):
            raise ValueError(f"rank {global_rank} RunConfig topology is not exact")
        if runtime.process_group_backend != "nccl":
            raise ValueError(
                "industrial GPU serving requires process_group_backend=nccl"
            )
        if world_size > 1:
            if (
                runtime.distributed_runtime_capability != "patched_two_gpu_v1"
                or runtime.distributed_capability_receipt_sha256 != capability_sha256
            ):
                raise ValueError(
                    "distributed RunConfig must bind the preflight runtime envelope"
                )
        elif (
            runtime.distributed_runtime_capability != "single_rank"
            or runtime.distributed_capability_receipt_sha256 is not None
        ):
            raise ValueError(
                "single-rank RunConfig cannot claim distributed capability"
            )

        body = config.model_dump(mode="json")
        body["runtime"]["tp_rank"] = "<rank>"
        body["runtime"]["dp_rank"] = "<rank>"
        body["runtime"]["device_identity"] = "<device>"
        normalized.append(body)
    if len({content_sha256(value) for value in normalized}) != 1:
        raise ValueError("rank configs differ outside rank-local topology identity")


def _validate_cell_config(cell: ExperimentCell, config: RunConfig) -> None:
    identity = cell.identity
    if config.method != identity.method:
        raise ValueError("RunConfig method differs from the cell method")
    if config.model.target != identity.model:
        raise ValueError("RunConfig target model differs from the cell model")
    if identity.backend != "NONE" and config.model.algorithm != identity.backend:
        raise ValueError("RunConfig algorithm differs from the cell backend")
    if config.model.max_context_length != identity.context:
        raise ValueError("RunConfig context differs from the cell context")
    if config.runtime.max_running_requests != identity.concurrency:
        raise ValueError("RunConfig admission cap differs from cell concurrency")
    if config.runtime.telemetry_detail != (
        "profile"
        if cell.resources.workload_class is WorkloadClass.PROFILE
        else "headline"
    ):
        raise ValueError("RunConfig telemetry detail differs from workload class")

    if identity.method in {"target_only", "static"}:
        adaptive_identity = (
            identity.scope,
            identity.rank,
            identity.alpha_over_rank,
            identity.optimizer,
            identity.learning_rate,
            identity.schedule,
            identity.parameterization,
        )
        if adaptive_identity != ("none", None, None, None, None, None, "none"):
            raise ValueError("non-adaptive cell carries adaptive identity fields")

    if identity.method == "target_only":
        if identity.width is not None:
            raise ValueError("target_only cells cannot declare a draft width")
        if config.runtime.speculation_enabled:
            raise ValueError("target_only RunConfig unexpectedly enables speculation")
        return

    if config.runtime.speculative_num_draft_tokens != identity.width:
        raise ValueError("RunConfig draft width differs from the cell width")
    if config.model.draft_depth + 1 != identity.width:
        raise ValueError("RunConfig draft depth differs from the cell width")
    if not config.runtime.speculation_enabled:
        raise ValueError("speculative cell has speculation disabled")
    if identity.method == "static":
        return

    adaptation = config.adaptation
    if adaptation is None:
        raise ValueError("adapted cell lacks adaptation configuration")
    if identity.parameterization not in {"full", "lora"}:
        raise ValueError("adapted cell parameterization is unresolved")
    if adaptation.weight_update_mode != identity.parameterization:
        raise ValueError("adaptation mode differs from the cell parameterization")
    if adaptation.parameter_scope != identity.scope:
        raise ValueError("adaptation scope differs from the cell scope")
    if adaptation.adaptation_group_id != identity.cohort:
        raise ValueError("adaptation group must equal the registered cohort identity")
    if adaptation.rank != identity.rank:
        raise ValueError("adaptation rank differs from the cell rank")
    actual_ratio = (
        None
        if adaptation.rank is None or adaptation.lora_alpha is None
        else adaptation.lora_alpha / adaptation.rank
    )
    if actual_ratio != identity.alpha_over_rank:
        raise ValueError("LoRA alpha/r differs from the cell identity")
    optimizer = adaptation.optimizer
    if optimizer.name != identity.optimizer:
        raise ValueError("optimizer differs from the cell identity")
    if optimizer.learning_rate != identity.learning_rate:
        raise ValueError("optimizer learning rate differs from the cell identity")
    if optimizer.schedule != identity.schedule:
        raise ValueError("optimizer schedule differs from the cell identity")
    if optimizer.schedule == "cosine_to_zero":
        raise ValueError(
            "cosine horizon is absent from CellIdentity and cannot be rendered exactly"
        )


def _validate_parameter_plan(
    cell: ExperimentCell,
    config: RunConfig,
    parameter_plan: TrainablePlan | None,
) -> str | None:
    if cell.identity.method in {"target_only", "static"}:
        if parameter_plan is not None:
            raise ValueError("non-adaptive methods cannot carry a trainable plan")
        return None
    if parameter_plan is None:
        raise ValueError("adapted methods require an exact trainable parameter plan")
    adaptation = config.adaptation
    if adaptation is None:
        raise ValueError("adapted methods require an adaptation configuration")
    expected = (
        cell.identity.backend,
        adaptation.weight_update_mode,
        adaptation.parameter_scope,
        adaptation.rank,
        adaptation.lora_alpha,
    )
    actual = (
        parameter_plan.backend,
        parameter_plan.mode,
        parameter_plan.scope,
        parameter_plan.rank,
        parameter_plan.lora_alpha,
    )
    if actual != expected:
        raise ValueError("trainable parameter plan differs from the cell RunConfig")
    return parameter_plan.sha256


def _render_industrial_runtime_plan(
    *,
    registry: ExperimentRegistry,
    cell_id: str,
    rank_configs: tuple[RunConfig, ...],
    topology_receipts: TopologyReceiptSet,
    dependency_receipts: tuple[ExperimentReceipt, ...],
    parameter_plan: TrainablePlan | None,
    physical_assignment: IndustrialPhysicalAssignment | None,
) -> IndustrialRuntimePlan:
    cell = _cell_by_id(registry, cell_id)
    _reject_unresolved_cell(cell)
    if not rank_configs:
        raise ValueError("at least one rank RunConfig is required")
    _validate_explicit_configs(rank_configs)
    dependencies = _dependency_chain(
        registry,
        cell.identity.experiment,
        dependency_receipts,
    )
    _validate_topology(
        cell,
        rank_configs,
        topology_receipts,
        dependencies,
        expected_gpu_uuids=(
            None if physical_assignment is None else physical_assignment.gpu_uuids
        ),
        expected_host_id=(
            None if physical_assignment is None else physical_assignment.host_id
        ),
    )
    for config in rank_configs:
        _validate_cell_config(cell, config)
    parameter_plan_sha256 = _validate_parameter_plan(
        cell,
        rank_configs[0],
        parameter_plan,
    )
    plan = IndustrialRuntimePlan(
        registry_sha256=registry.sha256,
        cell_id=cell.cell_id,
        cell_declaration_sha256=cell.sha256,
        dependency_receipt_sha256s=tuple(receipt.sha256 for receipt in dependencies),
        topology_receipt_sha256=topology_receipts.receipt_sha256,
        parameter_plan_sha256=parameter_plan_sha256,
        rank_configs=rank_configs,
        cell=cell,
        physical_assignment=physical_assignment,
    )
    # Force canonical serialization now so malformed future extensions fail at
    # the boundary rather than after a server has allocated device memory.
    _ = plan.sha256
    return plan


def render_industrial_cell_runtime_plan(
    *,
    registry: ExperimentRegistry,
    cell_id: str,
    rank_configs: tuple[RunConfig, ...],
    topology_receipts: TopologyReceiptSet,
    dependency_receipts: tuple[ExperimentReceipt, ...] = (),
    parameter_plan: TrainablePlan | None = None,
) -> IndustrialRuntimePlan:
    """Render a logical, backward-compatible registry resource contract.

    This pure helper performs no physical dispatch authorization.  Its output
    is marked ``registry_logical_only`` and ``physical_dispatch_ready=false``.
    Pool-scheduled execution must use
    :func:`render_assigned_industrial_cell_runtime_plan` instead.
    """

    return _render_industrial_runtime_plan(
        registry=registry,
        cell_id=cell_id,
        rank_configs=rank_configs,
        topology_receipts=topology_receipts,
        dependency_receipts=dependency_receipts,
        parameter_plan=parameter_plan,
        physical_assignment=None,
    )


def render_assigned_industrial_cell_runtime_plan(
    *,
    registry: ExperimentRegistry,
    cell_id: str,
    assignment: GpuAssignment,
    dispatch_plan: GpuDispatchPlan,
    dispatch_context: GpuDispatchExecutionContext,
    budget: ExperimentBudget,
    inventory: GpuInventory,
    dispatch_inventory_sha256: str,
    rank_configs: tuple[RunConfig, ...],
    topology_receipts: TopologyReceiptSet,
    dependency_receipts: tuple[ExperimentReceipt, ...] = (),
    parameter_plan: TrainablePlan | None = None,
) -> IndustrialRuntimePlan:
    """Render a registry cell against one exact physical GPU assignment.

    Validation is allocation-free and happens before the rank configs are
    accepted, so a partial/foreign/forged assignment cannot be hidden behind a
    later serving-schema failure.
    """

    physical_assignment = bind_industrial_gpu_assignment(
        registry=registry,
        cell_id=cell_id,
        assignment=assignment,
        dispatch_plan=dispatch_plan,
        dispatch_context=dispatch_context,
        budget=budget,
        inventory=inventory,
        dispatch_inventory_sha256=dispatch_inventory_sha256,
        topology_receipts=topology_receipts,
    )
    return _render_industrial_runtime_plan(
        registry=registry,
        cell_id=cell_id,
        rank_configs=rank_configs,
        topology_receipts=topology_receipts,
        dependency_receipts=dependency_receipts,
        parameter_plan=parameter_plan,
        physical_assignment=physical_assignment,
    )
