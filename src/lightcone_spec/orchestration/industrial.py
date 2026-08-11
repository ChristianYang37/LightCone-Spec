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

    def to_dict(self) -> dict[str, Any]:
        identity = self.cell.identity
        resources = self.cell.resources
        return {
            "schema_version": 1,
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
            "resources": {
                "gpu_uuids": list(resources.gpu_uuids),
                "ports": list(resources.ports),
                "cache_root": resources.cache_root,
                "evidence_root": resources.evidence_root,
                "workload_class": resources.workload_class.value,
                "exclusive": resources.exclusive,
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


def _validate_topology(
    cell: ExperimentCell,
    configs: tuple[RunConfig, ...],
    topology: TopologyReceiptSet,
    dependency_chain: tuple[ExperimentReceipt, ...],
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
    if expected_devices != cell.identity.gpu_uuids:
        raise ValueError("topology receipts do not bind the cell GPU UUID order")

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
    assert adaptation is not None
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


def render_industrial_cell_runtime_plan(
    *,
    registry: ExperimentRegistry,
    cell_id: str,
    rank_configs: tuple[RunConfig, ...],
    topology_receipts: TopologyReceiptSet,
    dependency_receipts: tuple[ExperimentReceipt, ...] = (),
    parameter_plan: TrainablePlan | None = None,
) -> IndustrialRuntimePlan:
    """Bind one runnable registry cell to fully explicit rank-local configs.

    This function is pure: it performs no model lookup, filesystem write,
    checkout mutation, server launch, or device allocation.
    """

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
    _validate_topology(cell, rank_configs, topology_receipts, dependencies)
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
    )
    # Force canonical serialization now so malformed future extensions fail at
    # the boundary rather than after a server has allocated device memory.
    _ = plan.sha256
    return plan
