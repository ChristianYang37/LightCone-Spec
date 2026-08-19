from __future__ import annotations

from copy import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_budget_materialization_authority import _build_authority_fixture
from test_execution_semantics import (
    _activation_authority,
    _e3a_selection_and_receipt,
    _load_binding,
    _raw,
)
from test_industrial_executor import _execution_fixture
from test_industrial_renderer import (
    _configs,
    _materialise,
    _receipts_before,
    _topology,
)

from lightcone_spec.config.schema import RunConfig
from lightcone_spec.experiments.execution_semantics import (
    EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON,
    EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON,
    EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON,
    CellExecutionSemanticsBlockedError,
    resolve_cell_execution_semantics,
)
from lightcone_spec.experiments.gpu_pool import GpuDispatchExecutionContext
from lightcone_spec.experiments.planning import (
    BudgetLoadRawBinding,
    BudgetMaterializationAuthorityBinding,
    reduce_e1_activation,
)
from lightcone_spec.experiments.registry import (
    build_legacy_industrial_registry as build_industrial_registry,
)
from lightcone_spec.experiments.registry import (
    scientific_role_for_cell,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.orchestration.execution_bundle import (
    ExecutionBundleBlockedError,
    _resolve_bundle_execution_semantics,
)
from lightcone_spec.orchestration.executor import (
    IndustrialExecutionPlan,
    _require_adapted_execution_semantics_sha256,
    _require_execution_trainable_plan_authority,
    build_industrial_execution_plan,
    render_industrial_execution_plan,
)
from lightcone_spec.orchestration.industrial import (
    IndustrialRuntimePlan,
    _render_industrial_runtime_plan,
    _require_e1_execution_semantics,
    _require_registered_e1_execution_recipe,
    render_industrial_cell_runtime_plan,
)


def _resolved_e1(
    scientific_role: str = "static",
    *,
    registry=None,
    optimizer_name: str | None = None,
):
    if registry is None:
        registry = build_industrial_registry(
            gpu_uuids=("GPU-render-a", "GPU-render-b"),
            cache_root="runtime-cache/renderer",
            evidence_root="artifacts/renderer",
        )
    selection, e3a_receipt = _e3a_selection_and_receipt(registry)
    activation_artifact = reduce_e1_activation(
        registry,
        e3a_receipt=e3a_receipt,
        selection=selection,
    )
    activation = _activation_authority(
        registry,
        selection,
        activation_artifact,
    )
    cell = next(
        candidate
        for candidate in registry.cells_for("E1")
        if scientific_role_for_cell(registry, candidate) == scientific_role
        and "width=8:concurrency=4" in candidate.identity.variant
        and (
            optimizer_name is None
            or registry.adaptation_recipe_for_cell(candidate).optimizer.name
            == optimizer_name
        )
    )
    load_binding = _load_binding(cell)
    semantics = resolve_cell_execution_semantics(
        activation=activation,
        load_binding=load_binding,
        cell=cell,
    )
    receipts = _receipts_before(registry, "E1")
    topology = _topology(cell)
    adaptation = (
        None
        if semantics.adaptation_recipe is None
        else semantics.adaptation_recipe.to_adaptation_config()
    )
    configs = tuple(
        _materialise(
            config.model_copy(
                update={
                    "runtime": config.runtime.model_copy(
                        update={"sampling_profile_sha256": SamplingProfile().sha256}
                    )
                }
            )
        )
        for config in _configs(cell, topology, receipts, adaptation=adaptation)
    )
    return (
        registry,
        activation,
        cell,
        load_binding,
        semantics,
        receipts,
        topology,
        configs,
    )


def _replaced_and_rehashed_recipe_semantics(semantics, mutation: str):
    recipe = semantics.adaptation_recipe
    assert recipe is not None
    if mutation == "epsilon":
        recipe = replace(
            recipe,
            optimizer=replace(recipe.optimizer, epsilon=2e-8),
        )
    elif mutation == "extra_delay":
        recipe = replace(recipe, extra_logical_delay=1)
    elif mutation == "teacher":
        recipe = replace(recipe, teacher_row_policy="quota_shadow")
    elif mutation == "verification":
        recipe = replace(
            recipe,
            verification_mode="fixed_budget",
            fixed_verification_budget=1,
        )
    elif mutation == "joint":
        recipe = replace(
            recipe,
            extra_logical_delay=1,
            teacher_row_policy="quota_shadow",
            verification_mode="fixed_budget",
            fixed_verification_budget=1,
        )
    else:  # pragma: no cover - test parametrization invariant
        raise AssertionError(f"unknown recipe mutation {mutation!r}")
    forged = replace(
        semantics,
        adaptation_recipe=recipe,
        adaptation_recipe_sha256=recipe.sha256,
    )
    assert forged.adaptation_recipe_sha256 == forged.adaptation_recipe.sha256
    assert forged.sha256 != semantics.sha256
    return forged


@pytest.mark.parametrize(
    "mutation",
    ("epsilon", "extra_delay", "teacher", "verification", "joint"),
)
def test_registered_recipe_membership_rejects_replace_and_rehash(
    mutation: str,
) -> None:
    registry, _, cell, _, semantics, *_ = _resolved_e1(
        "lc_candidate", optimizer_name="adamw"
    )
    assert (
        _require_registered_e1_execution_recipe(
            registry=registry,
            cell=cell,
            execution_semantics=semantics,
        )
        is semantics
    )
    forged = _replaced_and_rehashed_recipe_semantics(semantics, mutation)

    with pytest.raises(CellExecutionSemanticsBlockedError) as blocked:
        _require_registered_e1_execution_recipe(
            registry=registry,
            cell=cell,
            execution_semantics=forged,
        )
    assert blocked.value.reason_code == EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON


def test_e1_runtime_plan_binds_replay_semantics_and_exact_config() -> None:
    (
        registry,
        activation,
        cell,
        load_binding,
        semantics,
        receipts,
        topology,
        configs,
    ) = _resolved_e1()

    resolved = _resolve_bundle_execution_semantics(
        registry=registry,
        activation_replay=activation,
        load_binding=load_binding,
        cell=cell,
        run_config=configs[0],
        diagnostic=False,
    )

    plan = _render_industrial_runtime_plan(
        registry=registry,
        cell_id=cell.cell_id,
        rank_configs=configs,
        topology_receipts=topology,
        dependency_receipts=receipts,
        parameter_plan=None,
        execution_semantics=resolved,
        physical_assignment=None,
    )
    wire = plan.to_dict()

    assert wire["schema_version"] == 3
    assert wire["execution_semantics_sha256"] == semantics.sha256
    assert wire["execution_semantics"]["sha256"] == semantics.sha256
    assert wire["execution_semantics"]["activation_semantic_sha256"] == (
        semantics.activation_semantic_sha256
    )
    assert wire["execution_semantics"]["load_binding_sha256"] == (
        semantics.load_binding_sha256
    )
    assert wire["execution_semantics"]["adaptation_recipe_sha256"] is None

    profile_config = configs[0].model_copy(
        update={
            "runtime": configs[0].runtime.model_copy(
                update={"telemetry_detail": "profile"}
            )
        }
    )
    with pytest.raises(CellExecutionSemanticsBlockedError) as profile_blocked:
        replace(plan, rank_configs=(_materialise(profile_config),))
    assert (
        profile_blocked.value.reason_code
        == EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON
    )

    changed = configs[0].model_copy(
        update={
            "runtime": configs[0].runtime.model_copy(update={"max_running_requests": 8})
        }
    )
    with pytest.raises(CellExecutionSemanticsBlockedError) as blocked:
        _render_industrial_runtime_plan(
            registry=registry,
            cell_id=cell.cell_id,
            rank_configs=(_materialise(changed),),
            topology_receipts=topology,
            dependency_receipts=receipts,
            parameter_plan=None,
            execution_semantics=semantics,
            physical_assignment=None,
        )
    assert blocked.value.reason_code == EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON

    # The public logical renderer intentionally has no overlay input.  A caller
    # cannot inject derived semantics and use this diagnostic surface as launch
    # authority; unresolved E1 declarations remain rejected there.
    with pytest.raises(ValueError, match="unresolved semantic placeholder"):
        render_industrial_cell_runtime_plan(
            registry=registry,
            cell_id=cell.cell_id,
            rank_configs=configs,
            topology_receipts=topology,
            dependency_receipts=receipts,
        )


@pytest.mark.parametrize(
    "mutation",
    ("sampling", "random_seed", "runtime_context", "recipe"),
)
def test_bundle_semantics_rejects_config_domain_swaps(mutation: str) -> None:
    registry, activation, cell, load_binding, semantics, *_, configs = _resolved_e1(
        "lc_candidate"
    )
    config = configs[0]
    if mutation in {"sampling", "random_seed", "runtime_context"}:
        runtime_update = {
            "sampling": {"sampling_profile_sha256": "0" * 64},
            # The public schema fixes this field to Literal[1].  model_copy
            # deliberately models a hostile in-process dataclass replacement.
            "random_seed": {"random_seed": 2},
            "runtime_context": {"context_length": config.runtime.context_length - 1},
        }[mutation]
        changed = config.model_copy(
            update={"runtime": config.runtime.model_copy(update=runtime_update)}
        )
    else:
        value = config.model_dump(mode="json")
        value["adaptation"]["optimizer"]["learning_rate"] *= 2
        changed = RunConfig.model_validate(value)

    with pytest.raises(ExecutionBundleBlockedError) as blocked:
        _resolve_bundle_execution_semantics(
            registry=registry,
            activation_replay=activation,
            load_binding=load_binding,
            cell=cell,
            run_config=changed,
            diagnostic=False,
        )
    assert blocked.value.reason_code == EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON
    assert semantics.expected_sampling_profile_sha256 == SamplingProfile().sha256
    assert registry.sha256 == semantics.registry_sha256


def test_bundle_recipe_membership_replay_blocks_before_config_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, activation, cell, load_binding, semantics, *_, configs = _resolved_e1(
        "lc_candidate"
    )
    forged = _replaced_and_rehashed_recipe_semantics(semantics, "joint")
    monkeypatch.setattr(
        "lightcone_spec.experiments.execution_semantics.resolve_cell_execution_semantics",
        lambda **_: forged,
    )

    with pytest.raises(ExecutionBundleBlockedError) as blocked:
        _resolve_bundle_execution_semantics(
            registry=registry,
            activation_replay=activation,
            load_binding=load_binding,
            cell=cell,
            run_config=configs[0],
            diagnostic=False,
        )
    assert blocked.value.reason_code == EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON


def test_runtime_renderer_replays_registered_recipe_before_plan_construction() -> None:
    (
        registry,
        _,
        cell,
        _,
        semantics,
        receipts,
        topology,
        configs,
    ) = _resolved_e1("lc_candidate")
    forged = _replaced_and_rehashed_recipe_semantics(semantics, "joint")

    with pytest.raises(CellExecutionSemanticsBlockedError) as blocked:
        _render_industrial_runtime_plan(
            registry=registry,
            cell_id=cell.cell_id,
            rank_configs=configs,
            topology_receipts=topology,
            dependency_receipts=receipts,
            parameter_plan=None,
            execution_semantics=forged,
            physical_assignment=None,
        )
    assert blocked.value.reason_code == EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON


def _exact_e1_authority_context(root: Path, *, method: str = "static"):
    root.mkdir(parents=True)
    fixture = _build_authority_fixture(root)
    registry, activation, cell, load_binding, semantics, *rest = _resolved_e1(
        method,
        registry=fixture.registry,
    )
    raw_load = BudgetLoadRawBinding(
        cell_id=cell.cell_id,
        source=_raw(
            "budget_load_binding",
            label="integration-exact-load",
            semantic_sha256=load_binding.sha256,
        ),
    )
    authority = replace(
        fixture.authority,
        activation=activation.binding,
        load_bindings=(raw_load,),
        registry_sha256=registry.sha256,
        activation_sha256=activation.activation_sha256,
        budget_load_binding_sha256s=(load_binding.sha256,),
    )
    assert type(authority) is BudgetMaterializationAuthorityBinding
    context = copy(fixture.execution)
    object.__setattr__(context, "budget_materialization_authority", authority)
    assert type(context) is GpuDispatchExecutionContext
    return (
        context,
        authority,
        registry,
        activation,
        cell,
        load_binding,
        semantics,
        *rest,
    )


def test_assigned_semantics_must_match_exact_dispatch_raw_authority(
    tmp_path: Path,
) -> None:
    (
        context,
        authority,
        registry,
        _,
        cell,
        _,
        semantics,
        *_,
    ) = _exact_e1_authority_context((tmp_path / "exact-authority").resolve())

    assert (
        _require_e1_execution_semantics(
            registry=registry,
            cell=cell,
            dispatch_context=context,
            execution_semantics=semantics,
        )
        is semantics
    )

    changed_context = copy(context)
    object.__setattr__(
        changed_context,
        "budget_materialization_authority",
        replace(
            authority,
            load_bindings=(),
            budget_load_binding_sha256s=(),
        ),
    )
    with pytest.raises(CellExecutionSemanticsBlockedError) as blocked:
        _require_e1_execution_semantics(
            registry=registry,
            cell=cell,
            dispatch_context=changed_context,
            execution_semantics=semantics,
        )
    assert blocked.value.reason_code == EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON


@pytest.mark.parametrize("attack", ("baseline_recipe_cargo", "adapted_rehash"))
def test_execution_renderer_replays_recipe_before_file_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    method = "static" if attack == "baseline_recipe_cargo" else "lc_candidate"
    (
        context,
        _,
        registry,
        _,
        cell,
        load_binding,
        semantics,
        _,
        _,
        configs,
    ) = _exact_e1_authority_context(
        (tmp_path / f"{attack}-authority").resolve(),
        method=method,
    )
    runtime = IndustrialRuntimePlan(
        registry_sha256=registry.sha256,
        cell_id=cell.cell_id,
        cell_declaration_sha256=cell.sha256,
        dependency_receipt_sha256s=(),
        topology_receipt_sha256="0" * 64,
        parameter_plan_sha256=None,
        rank_configs=configs,
        cell=cell,
        execution_semantics=semantics,
        physical_assignment=None,
    )
    if attack == "baseline_recipe_cargo":
        adaptive_semantics = _resolved_e1("lc_candidate", registry=registry)[4]
        forged_semantics = copy(semantics)
        object.__setattr__(
            forged_semantics,
            "adaptation_recipe",
            adaptive_semantics.adaptation_recipe,
        )
        object.__setattr__(
            forged_semantics,
            "adaptation_recipe_sha256",
            adaptive_semantics.adaptation_recipe_sha256,
        )
    else:
        forged_semantics = _replaced_and_rehashed_recipe_semantics(
            semantics,
            "joint",
        )
    forged_runtime = copy(runtime)
    object.__setattr__(forged_runtime, "execution_semantics", forged_semantics)

    renderer_called = False

    def _unexpected_renderer(**_):
        nonlocal renderer_called
        renderer_called = True
        raise AssertionError("renderer ran before recipe membership replay")

    monkeypatch.setattr(
        "lightcone_spec.orchestration.executor._render_server",
        _unexpected_renderer,
    )
    output_root = tmp_path / f"{attack}-render-output"
    unused = object()

    with pytest.raises(CellExecutionSemanticsBlockedError) as blocked:
        render_industrial_execution_plan(
            output_root=output_root,
            runtime_plan=forged_runtime,
            dispatch_plan=unused,
            dispatch_context=context,
            budget_plan=unused,
            budget=unused,
            load_plan=load_binding.registered_load,
            dependency_receipts=(),
            dependency_artifacts=(),
            split_artifact=unused,
            sampling_artifact=unused,
            model_lock_artifact=unused,
            sglang_checkout=tmp_path / "unused-sglang",
            compile_cache_plan_path=tmp_path / "unused-compile-plan.json",
            inventory_source_artifact=unused,
            runtime_envelope_artifact=unused,
            model_roots={},
            adaptation_reserve_mb=0 if method == "static" else 1,
            mem_fraction_static=0.8,
        )
    assert blocked.value.reason_code == EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON
    assert renderer_called is False
    assert not output_root.exists()


def test_execution_plan_validate_rejects_caller_replaced_e1_overlay(
    tmp_path: Path,
) -> None:
    base = _execution_fixture(tmp_path / "executor", request_count=1).plan
    (
        context,
        _,
        registry,
        _,
        cell,
        load_binding,
        semantics,
        _,
        _,
        configs,
    ) = _exact_e1_authority_context((tmp_path / "authority").resolve())
    foreign_semantics = _resolved_e1("lc_candidate", registry=registry)[4]
    assert foreign_semantics.cell_declaration != cell
    forged_runtime = copy(base.runtime_plan)
    for name, value in (
        ("registry_sha256", registry.sha256),
        ("cell_id", cell.cell_id),
        ("cell_declaration_sha256", cell.sha256),
        ("rank_configs", configs),
        ("cell", cell),
        # Model a caller mutating/replacing the public dataclass after its
        # constructor checked the legitimate overlay.
        ("execution_semantics", foreign_semantics),
    ):
        object.__setattr__(forged_runtime, name, value)
    forged = replace(
        base,
        runtime_plan=forged_runtime,
        load_plan=load_binding.registered_load,
        dispatch_context=context,
    )

    with pytest.raises(CellExecutionSemanticsBlockedError) as blocked:
        forged.validate()
    assert blocked.value.reason_code == "cell_execution_semantics_foreign_cell"
    assert semantics.cell_declaration == cell

    object.__setattr__(forged_runtime, "execution_semantics", None)
    with pytest.raises(CellExecutionSemanticsBlockedError) as missing:
        _require_adapted_execution_semantics_sha256(forged_runtime)
    assert (
        missing.value.reason_code
        == "cell_execution_semantics_raw_activation_unavailable"
    )


def test_execution_plan_constructor_and_final_gate_replay_registered_recipe() -> None:
    (
        registry,
        _,
        cell,
        load_binding,
        semantics,
        _,
        _,
        configs,
    ) = _resolved_e1("lc_candidate")
    context = SimpleNamespace(registry=registry)
    forged_semantics = _replaced_and_rehashed_recipe_semantics(semantics, "joint")
    runtime = IndustrialRuntimePlan(
        registry_sha256=registry.sha256,
        cell_id=cell.cell_id,
        cell_declaration_sha256=cell.sha256,
        dependency_receipt_sha256s=(),
        topology_receipt_sha256="0" * 64,
        parameter_plan_sha256=None,
        rank_configs=configs,
        cell=cell,
        execution_semantics=semantics,
        physical_assignment=None,
    )
    forged_runtime = copy(runtime)
    object.__setattr__(forged_runtime, "execution_semantics", forged_semantics)
    forged_plan = object.__new__(IndustrialExecutionPlan)
    object.__setattr__(forged_plan, "runtime_plan", forged_runtime)
    object.__setattr__(forged_plan, "dispatch_context", context)
    unused = object()

    with pytest.raises(CellExecutionSemanticsBlockedError) as constructor_blocked:
        build_industrial_execution_plan(
            runtime_plan=forged_runtime,
            dispatch_plan=unused,
            dispatch_context=context,
            budget_plan=unused,
            budget=unused,
            load_plan=load_binding.registered_load,
            server_launch=unused,
            dependency_receipts=(),
            dependency_artifacts=(),
            split_artifact=unused,
            sampling_artifact=unused,
            model_lock_artifact=unused,
            compile_cache_plan=unused,
            inventory_source_artifact=unused,
            runtime_envelope_artifact=unused,
        )
    assert (
        constructor_blocked.value.reason_code
        == EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON
    )

    with pytest.raises(CellExecutionSemanticsBlockedError) as executor_blocked:
        _require_execution_trainable_plan_authority(forged_plan)
    assert (
        executor_blocked.value.reason_code
        == EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON
    )


@pytest.mark.parametrize("experiment", ("E2", "E3b", "E5"))
def test_unimplemented_stage_semantics_remain_a_named_formal_block(
    experiment: str,
) -> None:
    registry = build_industrial_registry()
    cell = registry.cells_for(experiment)[0]

    with pytest.raises(ExecutionBundleBlockedError) as blocked:
        _resolve_bundle_execution_semantics(
            registry=registry,
            activation_replay=None,
            load_binding=None,
            cell=cell,
            run_config=None,
            diagnostic=False,
        )
    assert (
        blocked.value.reason_code == "cell_execution_semantics_experiment_unsupported"
    )

    assert (
        _resolve_bundle_execution_semantics(
            registry=registry,
            activation_replay=None,
            load_binding=None,
            cell=cell,
            run_config=None,
            diagnostic=True,
        )
        is None
    )
