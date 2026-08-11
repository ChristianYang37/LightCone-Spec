from __future__ import annotations

from dataclasses import fields, replace

import pytest

from lightcone_spec.experiments.registry import (
    CONTEXT_GRID,
    CONTEXT_REGIMES,
    DRAFT_WIDTHS,
    E0_BACKENDS,
    E0_METHODS,
    E0_MODELS,
    E0_TASKS,
    E2_HALVING_STAGES,
    E3A_CONCURRENCY_GRID,
    E5_CLOSED_LOOP_CONCURRENCY,
    E5_COHORT_COUNTS,
    E5_COHORT_DISTRIBUTIONS,
    E5_OPEN_LOOP_LOAD_FACTORS,
    E5_TOPOLOGIES,
    E6_CANDIDATE_MODELS,
    INDUSTRIAL_EXPERIMENT_ORDER,
    INDUSTRIAL_PORT_SPAN,
    LORA_RANKS,
    PILOT_BLOCKS,
    REGISTERED_CONFIRMATION_BLOCKS,
    CellIdentity,
    CellStatus,
    ConfirmationBlockPlan,
    ExperimentReceipt,
    ExperimentRegistry,
    StageActivationPlan,
    TwoGpuResourceScheduler,
    WorkloadClass,
    build_industrial_registry,
    content_sha256,
    e1a_adaptive_configurations,
    serving_cell_rejection_reason,
    unresolved_semantic_placeholder,
)


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_industrial_registry(
        gpu_uuids=("GPU-aaaaaaaa", "GPU-bbbbbbbb"),
        base_port=24000,
        cache_root="runtime-cache/test",
        evidence_root="artifacts/test",
    )


def _axes(registry: ExperimentRegistry, experiment: str) -> dict[str, tuple]:
    return {axis.name: axis.values for axis in registry.definition(experiment).axes}


def _receipt(
    registry: ExperimentRegistry,
    experiment: str,
    dependencies: tuple[ExperimentReceipt, ...] = (),
) -> ExperimentReceipt:
    definition = registry.definition(experiment)
    return registry.make_receipt(
        experiment,
        {
            output: content_sha256({"experiment": experiment, "locked_output": output})
            for output in definition.locked_outputs
        },
        runtime_sha256=content_sha256({"runtime": "test"}),
        split_sha256=content_sha256({"experiment": experiment, "split": "test"}),
        completed_cells_sha256=content_sha256(
            {"experiment": experiment, "completed_cells": "test"}
        ),
        dependencies=dependencies,
    )


def _receipts_through(
    registry: ExperimentRegistry, experiment: str
) -> tuple[ExperimentReceipt, ...]:
    receipts: list[ExperimentReceipt] = []
    for name in INDUSTRIAL_EXPERIMENT_ORDER:
        receipts.append(_receipt(registry, name, tuple(receipts)))
        if name == experiment:
            return tuple(receipts)
    raise AssertionError(f"unknown experiment {experiment}")


def test_registry_digest_and_order_are_stable(registry: ExperimentRegistry) -> None:
    rebuilt = build_industrial_registry(
        gpu_uuids=("GPU-aaaaaaaa", "GPU-bbbbbbbb"),
        base_port=24000,
        cache_root="runtime-cache/test",
        evidence_root="artifacts/test",
    )
    reordered = replace(registry, cells=tuple(reversed(registry.cells)))

    assert registry.sha256 == rebuilt.sha256 == reordered.sha256
    assert tuple(row.name for row in registry.definitions) == (
        INDUSTRIAL_EXPERIMENT_ORDER
    )
    assert tuple(row.dependencies for row in registry.definitions) == (
        (),
        ("preflight",),
        ("E3a",),
        ("E1",),
        ("E2",),
        ("E4",),
        ("E3b",),
        ("E1a",),
        ("E5",),
        ("E6",),
    )
    assert len({cell.cell_id for cell in registry.cells}) == len(registry.cells)
    assert (
        registry.sha256
        != replace(registry, name="different-industrial-registry").sha256
    )


def test_protocol_axes_and_e1a_cardinality_are_complete(
    registry: ExperimentRegistry,
) -> None:
    e3a = _axes(registry, "E3a")
    assert e3a["context"] == CONTEXT_GRID
    assert e3a["concurrency"] == E3A_CONCURRENCY_GRID
    assert e3a["width"] == DRAFT_WIDTHS
    assert e3a["regime"] == CONTEXT_REGIMES

    e1 = _axes(registry, "E1")
    assert e1["rank"] == LORA_RANKS
    assert e1["alpha_over_rank"] == (1.0,)

    unresolved_optimizer = [
        cell
        for cell in registry.cells_for("E2")
        if cell.identity.optimizer == "chronobelief"
    ]
    assert len(unresolved_optimizer) == 96 * len(E2_HALVING_STAGES) * 2
    assert {cell.status for cell in unresolved_optimizer} == {CellStatus.BLOCKED}
    assert {cell.reason_code for cell in unresolved_optimizer} == {
        "optimizer_equation_unresolved"
    }
    implemented_optimizer = [
        cell for cell in registry.cells_for("E2") if cell.identity.optimizer == "adamw"
    ]
    assert len(implemented_optimizer) == 480 * len(E2_HALVING_STAGES) * 2
    assert len({cell.identity.learning_rate for cell in implemented_optimizer}) == 9
    assert all(
        cell.identity.learning_rate is not None for cell in implemented_optimizer
    )
    assert {cell.identity.context for cell in implemented_optimizer} == {
        context for _, context in E2_HALVING_STAGES
    }
    assert all(
        "halving_stage=" in cell.identity.variant for cell in implemented_optimizer
    )

    for experiment in ("E1", "E2"):
        adaptive = [
            cell
            for cell in registry.cells_for(experiment)
            if cell.identity.method in {"tts", "l0"}
        ]
        grouped: dict[tuple[object, ...], list[object]] = {}
        for cell in adaptive:
            identity = cell.identity
            key = (
                identity.scope,
                identity.rank,
                identity.optimizer,
                identity.learning_rate,
                identity.schedule,
                identity.context,
                identity.width,
                identity.parameterization,
                identity.variant,
                identity.concurrency,
            )
            grouped.setdefault(key, []).append(cell)
        assert grouped
        assert all(
            {cell.identity.method for cell in pair} == {"tts", "l0"}
            and len({cell.resources.gpu_uuids for cell in pair}) == 1
            for pair in grouped.values()
        )

    configurations = e1a_adaptive_configurations()
    assert len(configurations) == 56
    assert len({row.sha256 for row in configurations}) == 56
    assert sum(row.parameterization == "full" for row in configurations) == 7
    assert sum(row.parameterization == "lora" for row in configurations) == 49
    assert sum(row.native_head_policy == "frozen" for row in configurations) == 32
    assert sum(row.native_head_policy == "full" for row in configurations) == 24

    e1a_cells = registry.cells_for("E1a")
    adaptive = [row for row in e1a_cells if row.identity.method == "l0"]
    baselines = [row for row in e1a_cells if row.identity.method != "l0"]
    assert len(adaptive) == 56
    assert {row.status for row in adaptive} == {CellStatus.BLOCKED}
    assert {row.reason_code for row in adaptive} == {
        "patched_runtime_backend_unsupported"
    }
    assert {row.identity.method for row in baselines} == {"target_only", "static"}
    assert len(baselines) == 2

    e5 = _axes(registry, "E5")
    assert e5["closed_loop_concurrency"] == E5_CLOSED_LOOP_CONCURRENCY
    assert e5["open_loop_lambda_star"] == E5_OPEN_LOOP_LOAD_FACTORS
    assert e5["cohort_count"] == E5_COHORT_COUNTS
    assert e5["cohort_distribution"] == E5_COHORT_DISTRIBUTIONS
    assert e5["topology"] == E5_TOPOLOGIES
    e5_cells = registry.cells_for("E5")
    assert {cell.identity.block for cell in e5_cells} == set(
        REGISTERED_CONFIRMATION_BLOCKS
    )
    assert {
        cell.identity.block
        for cell in e5_cells
        if cell.identity.variant.startswith("excluded_pilot:")
    } == set(PILOT_BLOCKS)

    e4_cells = registry.cells_for("E4")
    ablations = [cell for cell in e4_cells if cell.identity.task == "systems_ablation"]
    profiles = [cell for cell in e4_cells if cell.identity.task == "isolated_profile"]
    assert len(ablations) == 6 * 3 * 2 * 2 * 2
    assert len(profiles) == 3
    assert len({cell.identity.sha256 for cell in ablations}) == len(ablations)
    assert all(
        "operational_grid=locked_e2" in cell.identity.arrival for cell in ablations
    )

    e3b_cells = registry.cells_for("E3b")
    assert {cell.identity.block for cell in e3b_cells} == set(
        REGISTERED_CONFIRMATION_BLOCKS
    )
    assert {
        cell.identity.block
        for cell in e3b_cells
        if cell.identity.variant.startswith("excluded_pilot:")
    } == set(PILOT_BLOCKS)
    per_block = len(e3b_cells) // len(REGISTERED_CONFIRMATION_BLOCKS)
    assert per_block == 4 * len(CONTEXT_GRID) * len(CONTEXT_REGIMES) * 2 * 2

    assert _axes(registry, "E6")["model"] == E6_CANDIDATE_MODELS
    assert len(registry.cells_for("E0")) == (
        len(E0_MODELS) * len(E0_BACKENDS) * len(E0_TASKS) * len(E0_METHODS)
    )

    unsupported_adaptive = [
        cell
        for cell in registry.cells
        if cell.identity.method in {"tts", "l0"}
        and (
            cell.identity.backend in {"DSPARK", "EAGLE3", "NEXTN", "DFLASH+DSPARK"}
            or cell.identity.topology
            in {"tp2_dp1", "two_replica_tp1_dp2", "tp2_and_two_replica"}
            or cell.identity.schedule
            in {"inverse_sqrt_published_update", "cosine_to_zero"}
        )
    ]
    assert unsupported_adaptive
    assert all(cell.status is CellStatus.BLOCKED for cell in unsupported_adaptive)
    unsupported_topology = [
        cell
        for cell in registry.cells
        if cell.identity.topology in {"tp2_dp1", "two_replica_tp1_dp2"}
        and cell.resources.workload_class is not WorkloadClass.DOWNLOAD
    ]
    assert unsupported_topology
    assert all(cell.status is CellStatus.BLOCKED for cell in unsupported_topology)


def test_every_cell_binds_full_identity_and_isolated_resources(
    registry: ExperimentRegistry,
) -> None:
    required_identity_fields = {
        "model",
        "backend",
        "task",
        "method",
        "scope",
        "rank",
        "alpha_over_rank",
        "optimizer",
        "learning_rate",
        "schedule",
        "context",
        "regime",
        "width",
        "arrival",
        "slo",
        "cohort",
        "topology",
        "seed",
        "block",
        "gpu_uuids",
    }
    assert required_identity_fields <= {field.name for field in fields(CellIdentity)}

    all_ports: list[int] = []
    cache_roots: list[str] = []
    evidence_roots: list[str] = []
    for cell in registry.cells:
        assert cell.identity.gpu_uuids == cell.resources.gpu_uuids
        assert len(cell.resources.gpu_uuids) in {1, 2}
        assert cell.resources.ports
        assert cell.resources.cache_root
        assert cell.resources.evidence_root
        all_ports.extend(cell.resources.ports)
        cache_roots.append(cell.resources.cache_root)
        evidence_roots.append(cell.resources.evidence_root)
    assert len(set(all_ports)) == INDUSTRIAL_PORT_SPAN
    assert len(cache_roots) == len(set(cache_roots))
    assert len(evidence_roots) == len(set(evidence_roots))
    assert {cell.resources.gpu_count for cell in registry.cells} == {1, 2}
    assert set(all_ports) == set(range(24000, 24000 + INDUSTRIAL_PORT_SPAN))
    with pytest.raises(ValueError, match="port span"):
        build_industrial_registry(base_port=65_535 - INDUSTRIAL_PORT_SPAN + 2)


def test_status_reason_is_preserved_without_changing_cell_identity(
    registry: ExperimentRegistry,
) -> None:
    source = registry.cells_for("E1")[0]
    not_applicable = source.with_status(
        CellStatus.NOT_APPLICABLE,
        reason_code="missing_checkpoint_pair",
        reason="The exact model/backend checkpoint pair does not exist.",
    )
    blocked = source.with_status(
        CellStatus.BLOCKED,
        reason_code="memory_fit_failed",
        reason="The least-feasible rank cannot reserve the registered KV margin.",
    )

    assert source.status is CellStatus.UNMEASURED
    assert source.reason_code == "awaiting_registered_measurement"
    assert not_applicable.status.value == "N/A"
    assert blocked.status.value == "BLOCKED"
    assert source.cell_id == not_applicable.cell_id == blocked.cell_id
    assert len({source.sha256, not_applicable.sha256, blocked.sha256}) == 3
    with pytest.raises(ValueError, match="reason_code"):
        source.with_status(
            CellStatus.BLOCKED,
            reason_code="",
            reason="A reason code is required.",
        )

    e6_blocked = [row for row in registry.cells_for("E6") if not row.runnable]
    assert len(e6_blocked) == 72
    assert {row.reason_code for row in e6_blocked} == {
        "native_nextn_preflight_required"
    }


def test_locked_outputs_and_dependency_receipts_fail_closed(
    registry: ExperimentRegistry,
) -> None:
    assert registry.ready_experiment(()) == "preflight"
    preflight = _receipt(registry, "preflight")
    assert registry.ready_experiment((preflight,)) == "E3a"

    with pytest.raises(ValueError, match="every and only"):
        registry.make_receipt(
            "E3a",
            {},
            runtime_sha256="a" * 64,
            split_sha256="b" * 64,
            completed_cells_sha256="c" * 64,
            dependencies=(preflight,),
        )

    e3a = _receipt(registry, "E3a", (preflight,))
    with pytest.raises(ValueError, match="missing dependencies"):
        registry.validate_receipts((e3a,))

    wrong_registry = replace(preflight, registry_sha256="0" * 64)
    with pytest.raises(ValueError, match="another registry"):
        registry.validate_receipts((wrong_registry,))

    through_e2 = _receipts_through(registry, "E2")
    assert registry.ready_experiment(through_e2) == "E4"
    assert tuple(registry.validate_receipts(through_e2)) == (
        "preflight",
        "E3a",
        "E1",
        "E2",
    )


def test_scheduler_dispatches_only_currently_executable_target_only_cells(
    registry: ExperimentRegistry,
) -> None:
    scheduler = TwoGpuResourceScheduler(registry)
    receipts = _receipts_through(registry, "preflight")

    serialized = scheduler.schedule(
        receipts=receipts,
        interference_gate_passed=False,
    )
    parallel = scheduler.schedule(
        receipts=receipts,
        interference_gate_passed=True,
    )

    assert serialized
    assert all(len(wave.cells) == 1 for wave in serialized)
    assert all(len(wave.cells) == 1 for wave in parallel)
    assert {cell.identity.method for wave in parallel for cell in wave.cells} == {
        "target_only"
    }

    repeated = scheduler.schedule(
        receipts=receipts,
        interference_gate_passed=True,
    )
    assert tuple(wave.sha256 for wave in parallel) == tuple(
        wave.sha256 for wave in repeated
    )


def test_scheduler_never_overlaps_exclusive_or_two_gpu_cells(
    registry: ExperimentRegistry,
) -> None:
    scheduler = TwoGpuResourceScheduler(registry)

    preflight_waves = scheduler.schedule(interference_gate_passed=True)
    assert preflight_waves
    assert all(len(wave.cells) == 1 for wave in preflight_waves)
    assert all(wave.cells[0].resources.gpu_count == 2 for wave in preflight_waves)
    assert all(
        wave.cells[0].identity.method == "target_only" for wave in preflight_waves
    )

    e4_waves = scheduler.schedule(
        receipts=_receipts_through(registry, "E2"),
        interference_gate_passed=True,
    )
    assert e4_waves == ()

    e3b_waves = scheduler.schedule(
        receipts=_receipts_through(registry, "E4"),
        interference_gate_passed=True,
    )
    assert e3b_waves == ()

    e5_waves = scheduler.schedule(
        receipts=_receipts_through(registry, "E1a"),
        interference_gate_passed=True,
    )
    assert e5_waves
    assert all(
        serving_cell_rejection_reason(cell) is None
        for wave in e5_waves
        for cell in wave.cells
    )
    assert all(
        len(wave.cells) == 1
        for wave in e5_waves
        if wave.cells[0].resources.gpu_count == 2
    )

    with pytest.raises(ValueError, match="outside the registry"):
        scheduler.schedule(completed_cell_ids=("f" * 64,))


def test_headline_method_order_is_randomized_and_gpu_rotated_by_block(
    registry: ExperimentRegistry,
) -> None:
    e3b = registry.cells_for("E3b")
    for method in ("target_only", "static", "tts", "l0"):
        assignments = [
            cell.resources.gpu_uuids[0]
            for cell in e3b
            if cell.identity.method == method
        ]
        assert assignments.count(registry.gpu_uuids[0]) == assignments.count(
            registry.gpu_uuids[1]
        )

    scheduler = TwoGpuResourceScheduler(registry)
    selected = [
        cell
        for cell in e3b
        if cell.identity.context == 1024
        and cell.identity.regime == CONTEXT_REGIMES[0]
        and cell.identity.arrival == "closed_loop_c1"
        and cell.identity.variant.endswith(":concurrency_one:matched")
    ]
    selected.sort(key=scheduler._dispatch_order_key)
    orders = {
        block: tuple(
            cell.identity.method for cell in selected if cell.identity.block == block
        )
        for block in PILOT_BLOCKS
    }
    assert all(
        set(order) == {"target_only", "static", "tts", "l0"}
        for order in orders.values()
    )
    assert len(set(orders.values())) > 1
    assert (
        scheduler.schedule(
            receipts=_receipts_through(registry, "E4"),
            interference_gate_passed=False,
        )
        == ()
    )


def test_confirmation_final_blocks_require_a_power_locked_pilot_plan(
    registry: ExperimentRegistry,
) -> None:
    scheduler = TwoGpuResourceScheduler(registry)
    receipts = _receipts_through(registry, "E4")
    pilot_cells = tuple(
        cell
        for cell in registry.cells_for("E3b")
        if cell.runnable and cell.identity.block in PILOT_BLOCKS
    )
    initial = scheduler.schedule(receipts=receipts, interference_gate_passed=True)
    assert initial == ()

    pilot_ids = tuple(sorted(cell.cell_id for cell in pilot_cells))
    plan = ConfirmationBlockPlan(
        registry_sha256=registry.sha256,
        experiment="E3b",
        runtime_sha256="a" * 64,
        split_sha256="b" * 64,
        pilot_evidence_sha256="c" * 64,
        completed_pilot_cells_sha256=content_sha256(pilot_ids),
        status="POWERED",
        selected_final_blocks=12,
        reason_code="registered_power_target_met",
    )
    with pytest.raises(ValueError, match="no dispatchable pilot"):
        scheduler.schedule(receipts=receipts, confirmation_block_plan=plan)


def test_e2_requires_dependency_bound_successive_halving_activation(
    registry: ExperimentRegistry,
) -> None:
    receipts = _receipts_through(registry, "E1")
    scheduler = TwoGpuResourceScheduler(registry)
    with pytest.raises(ValueError, match="sealed stage activation"):
        scheduler.schedule(receipts=receipts)

    cells = registry.cells_for("E2")
    blocked = tuple(sorted(cell.cell_id for cell in cells if not cell.runnable))
    stage_zero = [
        cell
        for cell in cells
        if cell.runnable and "halving_stage=0:" in cell.identity.variant
    ]
    activated_rows = [
        cell for cell in stage_zero if cell.identity.method in {"target_only", "static"}
    ]
    selected_l0 = next(cell for cell in stage_zero if cell.identity.method == "l0")
    activated_rows.append(selected_l0)
    activated_rows.append(
        next(
            cell
            for cell in stage_zero
            if cell.identity.method == "tts"
            and cell.identity.scope == selected_l0.identity.scope
            and cell.identity.rank == selected_l0.identity.rank
            and cell.identity.optimizer == selected_l0.identity.optimizer
            and cell.identity.learning_rate == selected_l0.identity.learning_rate
            and cell.identity.schedule == selected_l0.identity.schedule
            and cell.identity.parameterization == selected_l0.identity.parameterization
        )
    )
    activated = tuple(sorted(cell.cell_id for cell in activated_rows))
    not_applicable = tuple(
        sorted(cell.cell_id for cell in stage_zero if cell.cell_id not in activated)
    )
    deferred = tuple(
        sorted(
            cell.cell_id
            for cell in cells
            if cell.runnable and "halving_stage=0:" not in cell.identity.variant
        )
    )
    plan = StageActivationPlan(
        registry_sha256=registry.sha256,
        experiment="E2",
        dependency_receipt_sha256=receipts[-1].sha256,
        runtime_sha256="a" * 64,
        split_sha256="b" * 64,
        source_selection_sha256="c" * 64,
        activation_round="halving_0",
        status="AVAILABLE",
        activated_cell_ids=activated,
        not_applicable_cell_ids=not_applicable,
        blocked_cell_ids=blocked,
        deferred_cell_ids=deferred,
        reason_code="e1_pareto_activation",
    )
    with pytest.raises(ValueError, match="unresolved serving cell"):
        scheduler.schedule(
            receipts=receipts,
            interference_gate_passed=True,
            activation_plan=plan,
        )


def test_scheduler_preserves_blocked_reasons_and_excludes_unresolved_cells(
    registry: ExperimentRegistry,
) -> None:
    unresolved = next(
        cell
        for cell in registry.cells_for("E1")
        if unresolved_semantic_placeholder(cell) is not None
    )
    assert unresolved.status is CellStatus.UNMEASURED
    assert serving_cell_rejection_reason(unresolved) == (
        "cell contains unresolved semantic placeholder 'locked_reference_load'"
    )

    blocked = next(
        cell for cell in registry.cells_for("E6") if cell.status is CellStatus.BLOCKED
    )
    declaration = (blocked.status, blocked.reason_code, blocked.reason)
    assert TwoGpuResourceScheduler(registry).schedule(
        receipts=_receipts_through(registry, "E5"),
        interference_gate_passed=True,
    )
    assert (blocked.status, blocked.reason_code, blocked.reason) == declaration

    e0 = TwoGpuResourceScheduler(registry).schedule(
        receipts=_receipts_through(registry, "E6"),
        interference_gate_passed=True,
    )
    assert e0 == ()


def test_canonical_digest_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        content_sha256({"bad": float("nan")})
