from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.config.schema import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.experiments.budget_authority import (
    BudgetActivationAuthorityResult,
)
from lightcone_spec.experiments.execution_semantics import (
    EXECUTION_SEMANTICS_CELL_NOT_ACTIVATED_REASON,
    EXECUTION_SEMANTICS_E3A_SELECTION_MISMATCH_REASON,
    EXECUTION_SEMANTICS_E3A_SELECTION_UNAVAILABLE_REASON,
    EXECUTION_SEMANTICS_FOREIGN_CELL_REASON,
    EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON,
    EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON,
    EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON,
    EXECUTION_SEMANTICS_UNSUPPORTED_EXPERIMENT_REASON,
    CellExecutionSemanticsBlockedError,
    resolve_cell_execution_semantics,
)
from lightcone_spec.experiments.load import (
    FrozenSamplingParameters,
    ProductionLoadPlan,
    ProductionWindow,
    RequestTemplate,
    closed_loop_corpus,
)
from lightcone_spec.experiments.planning import (
    BudgetJobKind,
    BudgetLoadBinding,
    BudgetRawJsonBinding,
    DependencyGpuInventoryAuthorityBinding,
    E1ActivationAuthorityBinding,
    P99AnchorStatus,
    RegistryStageActivationAuthorityBinding,
    RegistryStageDependencyCompletionAuthorityBinding,
    SealedE3aSelection,
    reduce_e1_activation,
)
from lightcone_spec.experiments.registry import (
    ExperimentCell,
    ExperimentReceipt,
    LockedOutput,
    scientific_role_for_cell,
)
from lightcone_spec.experiments.registry import (
    build_legacy_industrial_registry as build_industrial_registry,
)
from lightcone_spec.experiments.sampling import SamplingProfile


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _raw(
    role: str,
    *,
    label: str,
    semantic_sha256: str | None = None,
) -> BudgetRawJsonBinding:
    source = Path(f"/tmp/lightcone-execution-semantics/{label}.json").resolve()
    canonical = semantic_sha256 or _sha(f"{label}-canonical")
    return BudgetRawJsonBinding(
        schema_version=1,
        role=role,
        path=str(source),
        sidecar_path=f"{source}.sha256",
        canonical_sha256=canonical,
        semantic_sha256=semantic_sha256 or canonical,
        file_sha256=_sha(f"{label}-file"),
        sidecar_file_sha256=_sha(f"{label}-sidecar"),
        size=1,
        sidecar_size=65,
    )


def _e3a_selection_and_receipt(registry):
    selection = SealedE3aSelection(
        schema_version=1,
        registry_sha256=registry.sha256,
        runtime_sha256=_sha("execution-semantics-runtime"),
        split_sha256=_sha("execution-semantics-split"),
        width=8,
        concurrency=4,
        reducer_evidence_sha256=_sha("execution-semantics-e3a-evidence"),
    )
    definition = registry.definition("E3a")
    outputs = {
        name: _sha(f"execution-semantics-E3a-{name}")
        for name in definition.locked_outputs
    }
    outputs["matched_width"] = selection.matched_width_output_sha256
    outputs["e1_reference_load"] = selection.reference_load_output_sha256
    receipt = ExperimentReceipt(
        experiment="E3a",
        registry_sha256=registry.sha256,
        runtime_sha256=selection.runtime_sha256,
        split_sha256=selection.split_sha256,
        completed_cells_sha256=_sha("execution-semantics-E3a-completed"),
        dependency_receipts=tuple(
            LockedOutput(name, _sha(f"execution-semantics-dependency-{name}"))
            for name in definition.dependencies
        ),
        outputs=tuple(LockedOutput(name, outputs[name]) for name in sorted(outputs)),
    )
    return selection, receipt


def _activation_authority(registry, selection, artifact):
    generated_registry = _raw(
        "generated_registry",
        label="generated-registry",
        semantic_sha256=registry.sha256,
    )
    inventory_source = _raw(
        "dependency_gpu_inventory",
        label="gpu-inventory",
        semantic_sha256=_sha("gpu-inventory-semantic"),
    )
    inventory_receipt = _raw(
        "dependency_gpu_inventory_source_receipt",
        label="gpu-inventory-receipt",
        semantic_sha256=_sha("gpu-inventory-receipt-semantic"),
    )
    inventory = DependencyGpuInventoryAuthorityBinding(
        schema_version=1,
        inventory=inventory_source,
        source_receipt=inventory_receipt,
        inventory_sha256=inventory_source.semantic_sha256,
        source_receipt_sha256=inventory_receipt.semantic_sha256,
    )
    generic = RegistryStageActivationAuthorityBinding(
        schema_version=1,
        kind="registry_stage_activation_manifest",
        manifest=_raw("registry_stage_activation_manifest", label="generic-manifest"),
        generated_registry=generated_registry,
        runtime=_raw("activation_runtime", label="generic-runtime"),
        split=_raw("activation_split", label="generic-split"),
        dependency_receipts=(),
        dependency_completion_authorities=(),
        activation_sha256=_sha("generic-activation"),
    )
    dependency_receipt = _raw(
        "activation_dependency_receipt", label="e3a-dependency-receipt"
    )
    completion = RegistryStageDependencyCompletionAuthorityBinding(
        schema_version=1,
        receipt=dependency_receipt,
        completed_cells=_raw("dependency_completed_cells", label="e3a-completed-cells"),
        activation=generic,
        inventory_authority=inventory,
        locked_outputs=(),
        receipt_sha256=dependency_receipt.semantic_sha256,
        completed_authority_sha256=_sha("e3a-completion-authority"),
    )
    binding = E1ActivationAuthorityBinding(
        schema_version=1,
        kind="e1_activation_manifest",
        manifest=_raw("e1_activation_authority_manifest", label="e1-manifest"),
        generated_registry=generated_registry,
        runtime=_raw(
            "activation_runtime",
            label="e1-runtime",
            semantic_sha256=selection.runtime_sha256,
        ),
        split=_raw(
            "activation_split",
            label="e1-split",
            semantic_sha256=selection.split_sha256,
        ),
        dependency_receipt=dependency_receipt,
        dependency_completion_authority=completion,
        selection_manifest=_raw(
            "e3a_selection_raw_manifest", label="e3a-selection-manifest"
        ),
        inventory_authority=inventory,
        hardware_envelope=_raw("activation_hardware_envelope", label="e1-hardware"),
        activation_sha256=artifact.sha256,
        selection_sha256=selection.sha256,
    )
    return BudgetActivationAuthorityResult(
        binding=binding,
        registry=registry,
        activation_artifact=artifact,
        family_activations=(),
        family_power_reductions=(),
        dependency_records=(),
        prior_family_authorities=(),
        e3a_selection=selection,
    )


@pytest.fixture(scope="module")
def e1_sources():
    registry = build_industrial_registry()
    selection, receipt = _e3a_selection_and_receipt(registry)
    artifact = reduce_e1_activation(
        registry,
        e3a_receipt=receipt,
        selection=selection,
    )
    authority = _activation_authority(registry, selection, artifact)
    cells = {
        role: next(
            cell
            for cell in registry.cells_for("E1")
            if scientific_role_for_cell(registry, cell) == role
            and "width=8:concurrency=4" in cell.identity.variant
            and (role != "lc_candidate" or cell.identity.optimizer == "adamw")
        )
        for role in (
            "target_only",
            "static",
            "tts",
            "l0_naive",
            "lc_candidate",
        )
    }
    return registry, selection, authority, cells


def _load_binding(
    cell: ExperimentCell,
    *,
    concurrency: int | None = None,
    cohort_seed: int | None = None,
    sampling_seed: int | None = None,
    sampling_overrides: dict[str, object] | None = None,
) -> BudgetLoadBinding:
    concurrency = cell.identity.concurrency if concurrency is None else concurrency
    cohort_seed = cell.identity.seed if cohort_seed is None else cohort_seed
    sampling_seed = cell.identity.seed if sampling_seed is None else sampling_seed
    sampling_parameters = {
        "temperature": 0.0,
        "top_p": 1.0,
        "sampling_seed": sampling_seed,
        "max_new_tokens": 2,
        "ignore_eos": True,
    }
    sampling_parameters.update(sampling_overrides or {})
    sampling = FrozenSamplingParameters.from_mapping(sampling_parameters)
    corpus = closed_loop_corpus(
        tuple(
            RequestTemplate(
                input_token_ids=(index + 1,),
                requested_output_tokens=2,
                sampling=sampling,
            )
            for index in range(max(4, concurrency))
        ),
        namespace=f"execution-semantics-{cell.cell_id}",
        split="tuning",
        concurrency=concurrency,
        cohort_count=cell.identity.cohort_count,
        cohort_popularity="uniform",
        cohort_seed=cohort_seed,
    )
    load = ProductionLoadPlan(
        warmup=None,
        scored=corpus,
        window=ProductionWindow(
            warmup_duration_us=0,
            arrival_duration_us=10_000,
            request_deadline_us=100_000,
            drain_duration_us=100_000,
        ),
    )
    return BudgetLoadBinding(
        cell_id=cell.cell_id,
        job_kind=BudgetJobKind.SHORT,
        optimistic_load=load,
        registered_load=load,
        quota_envelope_load=load,
        minimum_completed_requests=4,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
    )


def _run_config(cell: ExperimentCell, adaptation=None) -> RunConfig:
    target_only = cell.identity.method == "target_only"
    width = 8
    return RunConfig(
        method=cell.identity.method,
        model=ModelPair(
            key="execution-semantics-test",
            target=cell.identity.model,
            drafter="z-lab/Qwen3-8B-DFlash-b16",
            target_revision="1" * 40,
            drafter_revision="2" * 40,
            algorithm="DFLASH",
            max_context_length=int(cell.identity.context),
            draft_depth=width - 1,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256=SamplingProfile().sha256,
            speculation_enabled=not target_only,
            speculative_num_draft_tokens=width,
            max_running_requests=int(cell.identity.concurrency),
        ),
        adaptation=adaptation,
    )


@pytest.mark.parametrize("role", ("target_only", "static", "lc_candidate"))
def test_e1_executable_roles_resolve_source_owned_semantics(e1_sources, role: str):
    registry, selection, authority, cells = e1_sources
    cell = cells[role]
    semantics = resolve_cell_execution_semantics(
        activation=authority,
        load_binding=_load_binding(cell),
        cell=cell,
    )

    assert semantics.registry_sha256 == registry.sha256
    assert semantics.cell_declaration == cell
    assert semantics.activation_semantic_sha256 == authority.activation_sha256
    assert semantics.e3a_selection == selection
    assert semantics.expected_concurrency == selection.concurrency
    assert semantics.expected_workload_seed == cell.identity.seed
    assert semantics.expected_runtime_random_seed == 1
    assert semantics.expected_workload_seed != semantics.expected_runtime_random_seed
    assert semantics.expected_model_max_context_length == cell.identity.context
    assert semantics.expected_runtime_context_length == 40960
    assert semantics.expected_sampling_profile_sha256 == SamplingProfile().sha256
    assert semantics.registered_request_count == 4
    assert len(semantics.sha256) == 64
    if role == "target_only":
        assert semantics.expected_draft_width is None
        assert semantics.expected_draft_depth is None
    else:
        assert semantics.expected_draft_width == selection.width
        assert semantics.expected_draft_depth == selection.width - 1
    if role in {"target_only", "static"}:
        assert semantics.adaptation_recipe is None
        assert semantics.adaptation_recipe_sha256 is None
        adaptation = None
    else:
        assert semantics.adaptation_recipe is not None
        assert semantics.adaptation_recipe.status == "AVAILABLE"
        assert semantics.expected_learning_rate > 0
        adaptation = semantics.adaptation_recipe.to_adaptation_config()
    semantics.validate_run_config(_run_config(cell, adaptation))
    assert semantics == resolve_cell_execution_semantics(
        activation=authority,
        load_binding=_load_binding(cell),
        cell=cell,
    )


@pytest.mark.parametrize("role", ("tts", "l0_naive"))
def test_frozen_tts_recipe_anchors_are_named_blocks(e1_sources, role: str):
    _, _, authority, cells = e1_sources
    cell = cells[role]

    with pytest.raises(CellExecutionSemanticsBlockedError) as caught:
        resolve_cell_execution_semantics(
            activation=authority,
            load_binding=_load_binding(cell),
            cell=cell,
        )
    assert caught.value.reason_code == EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON


def test_foreign_and_deferred_cells_are_named_blocks(e1_sources):
    registry, _, authority, cells = e1_sources
    active = cells["static"]
    foreign = replace(
        active,
        resources=replace(active.resources, cache_root="caller-authored-cache"),
    )
    with pytest.raises(CellExecutionSemanticsBlockedError) as caught:
        resolve_cell_execution_semantics(
            activation=authority,
            load_binding=_load_binding(active),
            cell=foreign,
        )
    assert caught.value.reason_code == EXECUTION_SEMANTICS_FOREIGN_CELL_REASON

    deferred = next(
        cell
        for cell in registry.cells_for("E1")
        if cell.identity.method == "static"
        and "width=4:concurrency=1" in cell.identity.variant
    )
    with pytest.raises(CellExecutionSemanticsBlockedError) as caught:
        resolve_cell_execution_semantics(
            activation=authority,
            load_binding=_load_binding(deferred),
            cell=deferred,
        )
    assert caught.value.reason_code == EXECUTION_SEMANTICS_CELL_NOT_ACTIVATED_REASON


def test_missing_or_tampered_selection_is_a_named_block(e1_sources):
    _, selection, authority, cells = e1_sources
    cell = cells["static"]
    with pytest.raises(CellExecutionSemanticsBlockedError) as caught:
        resolve_cell_execution_semantics(
            activation=replace(authority, e3a_selection=None),
            load_binding=_load_binding(cell),
            cell=cell,
        )
    assert (
        caught.value.reason_code == EXECUTION_SEMANTICS_E3A_SELECTION_UNAVAILABLE_REASON
    )

    tampered_selection = replace(selection, width=16)
    with pytest.raises(CellExecutionSemanticsBlockedError) as caught:
        resolve_cell_execution_semantics(
            activation=replace(authority, e3a_selection=tampered_selection),
            load_binding=_load_binding(cell),
            cell=cell,
        )
    assert caught.value.reason_code == EXECUTION_SEMANTICS_E3A_SELECTION_MISMATCH_REASON


@pytest.mark.parametrize(
    "load_kwargs",
    (
        {"concurrency": 2},
        {"cohort_seed": 9},
        {"sampling_seed": 9},
        {"sampling_overrides": {"ignore_eos": 1}},
    ),
)
def test_registered_load_tamper_is_a_named_block(e1_sources, load_kwargs):
    _, _, authority, cells = e1_sources
    cell = cells["static"]
    with pytest.raises(CellExecutionSemanticsBlockedError) as caught:
        resolve_cell_execution_semantics(
            activation=authority,
            load_binding=_load_binding(cell, **load_kwargs),
            cell=cell,
        )
    assert caught.value.reason_code == EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON


@pytest.mark.parametrize("experiment", ("E2", "E3b", "E5"))
def test_non_e1_activation_is_stably_unsupported(e1_sources, experiment: str):
    _, _, authority, cells = e1_sources
    artifact = authority.activation_artifact
    altered = replace(
        authority,
        activation_artifact=replace(
            artifact,
            plan=replace(artifact.plan, experiment=experiment),
        ),
    )
    cell = cells["static"]
    with pytest.raises(CellExecutionSemanticsBlockedError) as caught:
        resolve_cell_execution_semantics(
            activation=altered,
            load_binding=_load_binding(cell),
            cell=cell,
        )
    assert caught.value.reason_code == EXECUTION_SEMANTICS_UNSUPPORTED_EXPERIMENT_REASON


def test_run_config_must_match_width_concurrency_and_full_recipe(e1_sources):
    _, _, authority, cells = e1_sources
    cell = cells["lc_candidate"]
    semantics = resolve_cell_execution_semantics(
        activation=authority,
        load_binding=_load_binding(cell),
        cell=cell,
    )
    recipe = semantics.adaptation_recipe
    assert recipe is not None
    config = _run_config(cell, recipe.to_adaptation_config())

    value = config.model_dump(mode="json")
    value["runtime"]["max_running_requests"] = 3
    wrong_concurrency = RunConfig.model_validate(value)
    with pytest.raises(CellExecutionSemanticsBlockedError) as caught:
        semantics.validate_run_config(wrong_concurrency)
    assert caught.value.reason_code == EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON

    value = config.model_dump(mode="json")
    value["adaptation"]["optimizer"]["learning_rate"] *= 2
    wrong_recipe = RunConfig.model_validate(value)
    with pytest.raises(CellExecutionSemanticsBlockedError) as caught:
        semantics.validate_run_config(wrong_recipe)
    assert caught.value.reason_code == EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON

    value = config.model_dump(mode="json")
    value["runtime"]["sampling_profile_sha256"] = _sha("caller-sampling")
    wrong_sampling = RunConfig.model_validate(value)
    with pytest.raises(CellExecutionSemanticsBlockedError) as caught:
        semantics.validate_run_config(wrong_sampling)
    assert caught.value.reason_code == EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON


def test_workload_and_runtime_seed_domains_cannot_be_swapped(e1_sources):
    _, _, authority, cells = e1_sources
    cell = cells["static"]
    semantics = resolve_cell_execution_semantics(
        activation=authority,
        load_binding=_load_binding(cell),
        cell=cell,
    )

    with pytest.raises(ValueError, match="scientific fields"):
        replace(semantics, expected_workload_seed=1)
    with pytest.raises(ValueError, match="runtime random seed"):
        replace(semantics, expected_runtime_random_seed=cell.identity.seed)


def test_bool_and_nonfinite_semantic_mutations_are_rejected(e1_sources):
    _, _, authority, cells = e1_sources
    cell = cells["lc_candidate"]
    semantics = resolve_cell_execution_semantics(
        activation=authority,
        load_binding=_load_binding(cell),
        cell=cell,
    )

    with pytest.raises(ValueError, match="positive integer"):
        replace(semantics, registered_request_count=True)
    with pytest.raises(ValueError, match="learning rate"):
        replace(semantics, expected_learning_rate=float("nan"))
