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
from lightcone_spec.orchestration.industrial import (
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


def test_adapted_cell_binds_optimizer_schedule_and_parameter_plan(
    registry: ExperimentRegistry,
) -> None:
    source = next(
        cell
        for cell in registry.cells_for("E2")
        if cell.runnable
        and cell.identity.method == "l0"
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
        canvas_tokens=identity.width,
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

    plan = render_industrial_cell_runtime_plan(
        registry=derived,
        cell_id=cell.cell_id,
        rank_configs=configs,
        topology_receipts=topology,
        dependency_receipts=receipts,
        parameter_plan=parameter_plan,
    )
    assert plan.parameter_plan_sha256 == parameter_plan.sha256
    assert (
        plan.to_dict()["rank_configs"][0]["adaptation"]["optimizer"]["schedule"]
        == "constant"
    )

    wrong_plan = DFlashParameterPlan.build(
        {"layers.0.q_proj.weight": torch.zeros(4, 4)},
        mode="full",
        scope="all",
    )
    with pytest.raises(ValueError, match="trainable parameter plan differs"):
        render_industrial_cell_runtime_plan(
            registry=derived,
            cell_id=cell.cell_id,
            rank_configs=configs,
            topology_receipts=topology,
            dependency_receipts=receipts,
            parameter_plan=wrong_plan,
        )

    changed = configs[0].model_dump(mode="json")
    changed["adaptation"]["optimizer"]["learning_rate"] *= 3
    with pytest.raises(ValueError, match="learning rate differs"):
        render_industrial_cell_runtime_plan(
            registry=derived,
            cell_id=cell.cell_id,
            rank_configs=(RunConfig.model_validate(changed),),
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
