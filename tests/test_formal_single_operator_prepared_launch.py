from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.config import (
    AdaptationConfig,
    ModelPair,
    OnlineSpecConfig,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
)
from lightcone_spec.experiments import (
    formal_single_operator_prepared_launch as prepared_launch,
)
from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
    FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256,
    FormalSingleOperatorPreparedLaunchBundle,
    FormalSingleOperatorPreparedLaunchEntry,
    FormalSingleOperatorProfilerSubjectRequirement,
    _trusted_adaptation_group,
    _TrustedChainRecipeContext,
    _validate_trusted_chain_run_config,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorJsonBinding,
    publish_formal_single_operator_json_artifact,
)
from lightcone_spec.experiments.onlinespec import onlinespec_candidates
from lightcone_spec.experiments.protocol import DFLASH_LOSS_POSITION_DECAY
from lightcone_spec.experiments.stage_materialization import (
    E1A_FIXED_VERIFICATION_BUDGET,
    E1Geometry,
    E2CandidateRecipe,
    MaterializedCell,
    default_e2_recipe_grid_authority,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _binding(tmp_path: Path, name: str) -> CanonicalJsonProofBinding:
    path = tmp_path / f"{name}.json"
    publish_canonical_json_no_replace(
        path,
        {
            "schema_version": 1,
            "kind": "test_source_owned_artifact",
            "name": name,
        },
    )
    return CanonicalJsonProofBinding.bind(path)


def _profiler_requirement(
    tmp_path: Path,
) -> FormalSingleOperatorProfilerSubjectRequirement:
    selected_config = _binding(tmp_path, "selected-full-config")
    selected_launch = _binding(tmp_path, "selected-compile-launch")
    workload = _binding(tmp_path, "code-owned-profiler-workload")
    schedule = _binding(tmp_path, "code-owned-profiler-schedule")
    return FormalSingleOperatorProfilerSubjectRequirement(
        schema_version=1,
        kind="formal_single_operator_profiler_subject_requirement",
        protocol_sha256=(FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256),
        source_headline_cell_id=_sha("source-headline-cell"),
        selected_configuration_sha256=_sha("selected-configuration"),
        selected_full_run_config=selected_config,
        selected_compile_launch_manifest=selected_launch,
        code_owned_profiler_subject_workload=workload,
        code_owned_request_schedule=schedule,
    )


def _entry(
    tmp_path: Path,
    *,
    label: str,
    physical_kind: str,
    profiler: FormalSingleOperatorProfilerSubjectRequirement | None = None,
) -> FormalSingleOperatorPreparedLaunchEntry:
    return FormalSingleOperatorPreparedLaunchEntry(
        schema_version=1,
        kind="formal_single_operator_prepared_launch_entry",
        protocol_sha256=(FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256),
        materialized_cell_id=_sha(f"cell:{label}"),
        physical_kind=physical_kind,  # type: ignore[arg-type]
        run_config=_binding(tmp_path, f"run-config-{label}"),
        compile_launch_manifest=_binding(tmp_path, f"compile-launch-{label}"),
        request_schedule_receipt=(
            None
            if physical_kind == "profiler"
            else _binding(tmp_path, f"request-schedule-{label}")
        ),
        launch_compatibility_key_sha256=_sha(f"compatibility:{label}"),
        target_content_member_id=f"target:{label}",
        drafter_content_member_id=f"drafter:{label}",
        tokenizer_content_member_id=f"tokenizer:{label}",
        inventory_sha256=_sha("inventory"),
        topology_mode="tp1_dp1",
        gpu_uuids=("GPU-0",),
        server_argv_sha256=_sha(f"argv:{label}"),
        profiler_subject=profiler,
    )


def test_profiler_subject_schema_requires_full_config_and_code_owned_workload(
    tmp_path: Path,
) -> None:
    requirement = _profiler_requirement(tmp_path)
    assert (
        FormalSingleOperatorProfilerSubjectRequirement.from_dict(requirement.to_dict())
        == requirement
    )
    fields = set(FormalSingleOperatorProfilerSubjectRequirement.__dataclass_fields__)
    assert {
        "selected_full_run_config",
        "selected_compile_launch_manifest",
        "code_owned_profiler_subject_workload",
        "code_owned_request_schedule",
    } <= fields
    assert "load" not in fields
    assert "traffic" not in fields
    assert "subject_argv" not in fields

    missing_workload = requirement.to_dict()
    missing_workload.pop("code_owned_profiler_subject_workload")
    with pytest.raises(ValueError, match="fields differ"):
        FormalSingleOperatorProfilerSubjectRequirement.from_dict(missing_workload)


def test_prepared_launch_profiler_coverage_is_exact(tmp_path: Path) -> None:
    requirement = _profiler_requirement(tmp_path)
    profiler = _entry(
        tmp_path,
        label="profile",
        physical_kind="profiler",
        profiler=requirement,
    )
    assert (
        FormalSingleOperatorPreparedLaunchEntry.from_dict(profiler.to_dict())
        == profiler
    )

    with pytest.raises(ValueError, match="only profiler launches"):
        _entry(tmp_path, label="missing-subject", physical_kind="profiler")
    with pytest.raises(ValueError, match="only profiler launches"):
        _entry(
            tmp_path,
            label="serving-with-subject",
            physical_kind="serving",
            profiler=requirement,
        )


def test_prepared_launch_bundle_is_sorted_unique_and_path_bound(
    tmp_path: Path,
) -> None:
    first = _entry(tmp_path, label="a", physical_kind="serving")
    second = _entry(tmp_path, label="b", physical_kind="e5_failure")
    entries = tuple(sorted((first, second), key=lambda row: row.materialized_cell_id))
    bundle = FormalSingleOperatorPreparedLaunchBundle(
        schema_version=1,
        kind="formal_single_operator_prepared_launch_bundle",
        protocol_sha256=(FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256),
        node="e5_final",
        stage="E5",
        phase="final",
        execution_source=_binding(tmp_path, "execution-source"),
        execution_source_sha256=_sha("execution-source"),
        protocol_lock_sha256=_sha("lock"),
        materialization_sha256=_sha("materialization"),
        materialization_source_decision_sha256=_sha("decision"),
        inventory=_binding(tmp_path, "inventory"),
        content_verification_receipt=_binding(tmp_path, "content-receipt"),
        entries=entries,
    )
    assert (
        FormalSingleOperatorPreparedLaunchBundle.from_dict(bundle.to_dict()) == bundle
    )
    assert bundle.entries == entries

    with pytest.raises(ValueError, match="not canonical"):
        replace(bundle, entries=(first, first))


def _trusted_recipe_context(
    *,
    tts_recipe: str,
    selected_online: tuple[tuple[str, str, str], ...] = (),
) -> _TrustedChainRecipeContext:
    grid = default_e2_recipe_grid_authority()
    recipe = E2CandidateRecipe(
        geometry=E1Geometry("last1", "full", None, None),
        optimizer="adam",
        schedule="constant",
        learning_rate=1e-7,
        optimizer_recipe_authority_sha256=(grid.optimizer_recipe_authority.sha256),
    )
    return _TrustedChainRecipeContext(
        protocol_lock=SimpleNamespace(),
        matched_width=8,
        common_load=4,
        frozen_tts_recipe_sha256=tts_recipe,
        tts_learning_rate=1e-5,
        tts_stride=10,
        lightcone_recipe=recipe,
        dspark_selected_configuration=None,
        dspark_selected_recipe_sha256=None,
        e0_selected_recipes=selected_online,
    )


def _model(*, algorithm: str = "DFLASH", width: int = 8) -> ModelPair:
    return ModelPair(
        target_revision="1" * 40,
        drafter_revision="2" * 40,
        algorithm=algorithm,  # type: ignore[arg-type]
        draft_depth=width - 1,
    )


def _runtime(*, width: int = 8, load: int = 1) -> RuntimeConfig:
    return RuntimeConfig(
        sampling_profile_sha256=_sha("sampling"),
        device_identity="GPU-0",
        speculative_num_draft_tokens=width,
        max_running_requests=load,
    )


def _e3b_cell(*, role: str, recipe: str | None) -> MaterializedCell:
    return MaterializedCell(
        stage="E3b",
        method_role=role,
        model="Qwen/Qwen3-8B",
        backend="NONE" if role == "Target-only" else "DFLASH",
        task="heldout_long_context_confirmation",
        publication_policy=(
            "fixed_barrier"
            if role == "TTS"
            else "first_ready"
            if role in {"L0-naive", "LightCone"}
            else "none"
        ),
        recipe_sha256=recipe,
        dimensions=(
            ("block", 0),
            ("block_phase", "excluded_pilot"),
            ("context", 4096),
            ("load", "concurrency_one"),
            ("regime", "short_input_long_generation"),
            ("tts_l0_pair_id", _sha("tts-l0-pair")),
            ("width_panel", "matched"),
        ),
    )


def test_trusted_chain_tts_l0_are_numeric_identical_except_method_policy() -> None:
    tts_recipe = _sha("frozen-tts")
    context = _trusted_recipe_context(tts_recipe=tts_recipe)
    tts = _e3b_cell(role="TTS", recipe=tts_recipe)
    l0 = _e3b_cell(role="L0-naive", recipe=tts_recipe)
    adaptation = AdaptationConfig(
        weight_update_mode="full",
        parameter_scope="all",
        adaptation_group_id=_trusted_adaptation_group(tts, paired_tts_l0=True),
        optimizer=OptimizerConfig(
            name="adam",
            learning_rate=1e-5,
            weight_decay=0.0,
            grad_clip=None,
            schedule="constant",
        ),
        stride=10,
        canvas_tokens=8,
        loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
    )
    for cell, method in ((tts, "tts"), (l0, "l0")):
        config = RunConfig(
            method=method,  # type: ignore[arg-type]
            model=_model(),
            runtime=_runtime(),
            adaptation=adaptation,
        )
        _validate_trusted_chain_run_config(
            context=context,
            source=SimpleNamespace(node="e3b_pilot"),
            cell=cell,
            config=config,
        )
    mutated = AdaptationConfig(
        **{
            **adaptation.model_dump(),
            "optimizer": OptimizerConfig(
                name="adam",
                learning_rate=3e-5,
                grad_clip=None,
            ),
        }
    )
    with pytest.raises(ValueError, match="numeric RunConfig"):
        _validate_trusted_chain_run_config(
            context=context,
            source=SimpleNamespace(node="e3b_pilot"),
            cell=tts,
            config=RunConfig(
                method="tts",
                model=_model(),
                runtime=_runtime(),
                adaptation=mutated,
            ),
        )


def test_trusted_chain_rejects_lightcone_optimizer_and_recipe_mutations() -> None:
    context = _trusted_recipe_context(tts_recipe=_sha("frozen-tts"))
    cell = _e3b_cell(role="LightCone", recipe=context.lightcone_recipe.sha256)
    expected = default_e2_recipe_grid_authority().adaptation_config_for(
        context.lightcone_recipe,
        canvas_tokens=8,
        adaptation_group_id=_trusted_adaptation_group(cell, paired_tts_l0=False),
        chronobelief_gpu_proof_sha256=None,
    )
    config = RunConfig(
        method="l0",
        model=_model(),
        runtime=_runtime(),
        adaptation=expected,
    )
    _validate_trusted_chain_run_config(
        context=context,
        source=SimpleNamespace(node="e3b_pilot"),
        cell=cell,
        config=config,
    )
    with pytest.raises(ValueError, match="recipe differs"):
        _validate_trusted_chain_run_config(
            context=context,
            source=SimpleNamespace(node="e3b_pilot"),
            cell=replace(cell, recipe_sha256=_sha("foreign-recipe")),
            config=config,
        )
    foreign_optimizer = OptimizerConfig(
        **{
            **expected.optimizer.model_dump(),
            "learning_rate": 3e-7,
        }
    )
    mutated = AdaptationConfig(
        **{**expected.model_dump(), "optimizer": foreign_optimizer}
    )
    with pytest.raises(ValueError, match="numeric RunConfig"):
        _validate_trusted_chain_run_config(
            context=context,
            source=SimpleNamespace(node="e3b_pilot"),
            cell=cell,
            config=RunConfig(
                method="l0",
                model=_model(),
                runtime=_runtime(),
                adaptation=mutated,
            ),
        )


def test_trusted_chain_requires_exact_e0_onlinespec_winner() -> None:
    candidate = next(
        row for row in onlinespec_candidates() if row.method == "onlinespec_ogd"
    )
    decision_id = _sha("compatibility-decision")
    context = _trusted_recipe_context(
        tts_recipe=_sha("frozen-tts"),
        selected_online=((decision_id, "OnlineSPEC-OGD", candidate.candidate_id),),
    )
    cell = MaterializedCell(
        stage="E0",
        method_role="OnlineSPEC-OGD",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="LiveCodeBench",
        publication_policy="independent_online",
        recipe_sha256=candidate.candidate_id,
        dimensions=(
            ("block", 0),
            ("compatibility_decision_id", decision_id),
            ("load", "concurrency_one"),
        ),
    )
    adaptation = AdaptationConfig(
        weight_update_mode=candidate.weight_update_mode,  # type: ignore[arg-type]
        parameter_scope=candidate.parameter_scope,
        adaptation_group_id=f"e0:{cell.cell_id}",
        optimizer=OptimizerConfig(
            name="sgd",
            learning_rate=candidate.learning_rate,
            grad_clip=candidate.grad_clip,
        ),
        rank=candidate.rank,
        lora_alpha=(candidate.rank if candidate.weight_update_mode == "lora" else None),
        stride=candidate.stride,
        canvas_tokens=16,
        loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
    )
    config = RunConfig(
        method="onlinespec_ogd",
        model=_model(width=16),
        runtime=_runtime(width=16),
        adaptation=adaptation,
        online_spec=OnlineSpecConfig(
            projection_radius=candidate.projection_radius,
            additional_learning_rates=candidate.additional_learning_rates,
            hedge_learning_rate=candidate.hedge_learning_rate,
        ),
    )
    _validate_trusted_chain_run_config(
        context=context,
        source=SimpleNamespace(node="e0_pilot"),
        cell=cell,
        config=config,
    )
    with pytest.raises(ValueError, match="not its winner"):
        _validate_trusted_chain_run_config(
            context=replace(context, e0_selected_recipes=()),
            source=SimpleNamespace(node="e0_pilot"),
            cell=cell,
            config=config,
        )


def test_trusted_chain_freezes_e1a_fixed_verification_budget_to_eight() -> None:
    context = _trusted_recipe_context(tts_recipe=_sha("frozen-tts"))
    cell = MaterializedCell(
        stage="E1a",
        method_role="LightCone-candidate",
        model="Qwen/Qwen3-8B",
        backend="DSPARK",
        task="LiveCodeBench_tuning_disjoint_from_E5",
        publication_policy="first_ready",
        recipe_sha256=context.lightcone_recipe.sha256,
        dimensions=(
            ("fixed_verification_budget", E1A_FIXED_VERIFICATION_BUDGET),
            ("frozen_tts_recipe_sha256", context.frozen_tts_recipe_sha256),
            ("parameterization", "full"),
            ("rank", "none"),
            ("scope", "last1"),
            ("verification_mode", "fixed_verification_budget"),
        ),
    )
    expected = default_e2_recipe_grid_authority().adaptation_config_for(
        context.lightcone_recipe,
        canvas_tokens=8,
        adaptation_group_id=_trusted_adaptation_group(
            cell,
            paired_tts_l0=False,
        ),
        chronobelief_gpu_proof_sha256=None,
    )
    expected = AdaptationConfig(
        **{
            **expected.model_dump(),
            "verification_mode": "fixed_budget",
            "fixed_verification_budget": E1A_FIXED_VERIFICATION_BUDGET,
        }
    )
    config = RunConfig(
        method="l0",
        model=_model(algorithm="DSPARK"),
        runtime=_runtime(width=8, load=4),
        adaptation=expected,
    )
    _validate_trusted_chain_run_config(
        context=context,
        source=SimpleNamespace(node="e1a"),
        cell=cell,
        config=config,
    )
    mutated_dimensions = tuple(
        (name, 7 if name == "fixed_verification_budget" else value)
        for name, value in cell.dimensions
    )
    with pytest.raises(ValueError, match="fixed verification budget"):
        _validate_trusted_chain_run_config(
            context=context,
            source=SimpleNamespace(node="e1a"),
            cell=replace(cell, dimensions=mutated_dimensions),
            config=config,
        )


def test_prepared_schedule_rebinds_current_materialization_across_binding_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.orchestration import formal_physical_dispatch as dispatch

    materialization_path = (tmp_path / "materialization.json").resolve()
    publish_formal_single_operator_json_artifact(
        materialization_path,
        {"kind": "fixture-current-materialization"},
    )
    source_materialization = FormalSingleOperatorJsonBinding.bind(
        materialization_path,
        label="fixture current materialization",
    )
    canonical_materialization = CanonicalJsonProofBinding.bind(materialization_path)
    schedule_binding = _binding(tmp_path, "prepared-schedule")
    launch_binding = _binding(tmp_path, "prepared-launch")
    content_binding = _binding(tmp_path, "prepared-content")
    schedule_source_binding = _binding(tmp_path, "prepared-schedule-source")
    execution_binding_sha256 = _sha("prepared-execution")
    subject_sha256 = _sha("prepared-subject")
    schedule_source = SimpleNamespace(
        schema_version=3,
        sha256=schedule_source_binding.semantic_sha256,
        subject_sha256=subject_sha256,
        materialization_receipt_sha256=_sha("materialization-receipt"),
        materialized_cell_id=_sha("cell"),
        topology_mode="tp1_dp1",
        max_running_requests=4,
        e5_arrival_plan=None,
        requests=(),
    )
    source_artifact = SimpleNamespace(
        load=dict,
        semantic_sha256=schedule_source_binding.semantic_sha256,
    )
    tokenization_input = SimpleNamespace(
        reopen=lambda: {"schedule_source_sha256": schedule_source.sha256}
    )
    tokenization_output = SimpleNamespace(
        reopen=lambda: {
            "schedule_source_sha256": schedule_source.sha256,
            "requests": [],
        }
    )
    schedule = SimpleNamespace(
        schema_version=3,
        sha256=schedule_binding.semantic_sha256,
        execution_binding_sha256=execution_binding_sha256,
        subject_sha256=subject_sha256,
        materialized_cell_id=_sha("cell"),
        workload_authority_sha256=_sha("workload"),
        content_verification_receipt_sha256=content_binding.semantic_sha256,
        content_verification_receipt=content_binding,
        content_source_binding=None,
        topology_mode="tp1_dp1",
        materialization=canonical_materialization,
        compile_launch_manifest=launch_binding,
        tokenizer_model_id="tokenizer",
        schedule_source=source_artifact,
        workload_source=SimpleNamespace(load=dict),
        sampling_profile=SimpleNamespace(reopen=dict),
        tokenization_input=tokenization_input,
        tokenization_output=tokenization_output,
        e5_arrival_plan=None,
        requests=(),
    )
    bundle = SimpleNamespace(
        schema_version=1,
        stage="E4",
        content_verification_receipt=content_binding,
        materialization_sha256=_sha("materialization-receipt"),
    )
    entry = SimpleNamespace(
        request_schedule_receipt=schedule_binding,
        materialized_cell_id=_sha("cell"),
        topology_mode="tp1_dp1",
        compile_launch_manifest=launch_binding,
    )
    source = SimpleNamespace(
        formal_workload_e0_authorization_sha256=_sha("e0-workload"),
        formal_workload_e3a_authorization_sha256=_sha("workload"),
        materialization_source=source_materialization,
    )
    monkeypatch.setattr(
        prepared_launch,
        "formal_single_operator_prepared_execution_identities",
        lambda **_kwargs: (execution_binding_sha256, subject_sha256),
    )
    monkeypatch.setattr(
        dispatch,
        "FormalServingRequestScheduleReceipt",
        SimpleNamespace(from_dict=lambda _value: schedule),
    )
    monkeypatch.setattr(
        dispatch,
        "FormalServingRequestScheduleSource",
        SimpleNamespace(from_dict=lambda _value: schedule_source),
    )
    monkeypatch.setattr(
        dispatch,
        "formal_serving_request_schedule_rows",
        lambda _value: iter(()),
    )
    monkeypatch.setattr(
        dispatch,
        "formal_serving_request_schedule_source_rows",
        lambda _value: iter(()),
    )
    monkeypatch.setattr(
        prepared_launch.CompileLaunchManifest,
        "load",
        lambda _path: SimpleNamespace(tokenizer_model_id="tokenizer"),
    )
    prepared_launch._validate_prepared_request_schedule(
        bundle=bundle,
        entry=entry,
        source=source,
        cell=SimpleNamespace(stage="E4", dimensions=()),
        config=SimpleNamespace(runtime=SimpleNamespace(max_running_requests=4)),
    )
