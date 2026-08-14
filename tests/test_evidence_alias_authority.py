from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_industrial_executor import _execution_fixture

from lightcone_spec.experiments.budget_authority import (
    BudgetMaterializationBlockedError,
)
from lightcone_spec.experiments.industrial_analysis import (
    AliasExecutionArtifacts,
    BoundArtifact,
    IndustrialCellEvidence,
    RawEvidenceAliasManifest,
    _audit_alias_execution_candidate,
    _independent_method_blocks,
    _load_alias_execution_candidate,
    _paired_dependence_components,
    raw_evidence_alias_manifest_from_dict,
    raw_evidence_alias_manifest_to_dict,
)
from lightcone_spec.experiments.itl_authority import (
    ItlTimestampAuthorityBlocked,
    release_e2_itl_timestamp_plan,
)
from lightcone_spec.experiments.planning import (
    EVIDENCE_ALIAS_REDUCER_PROTOCOL_SHA256,
    EvidenceAliasReductionArtifact,
    ExecutionDerivedAliasSemantics,
    RawEvidenceRunBinding,
    build_evidence_dependence_map,
)
from lightcone_spec.experiments.planning_artifacts import (
    PlanningArtifactSidecar,
    budget_materialization_authority_binding_to_dict,
    evidence_alias_reduction_artifact_from_dict,
    evidence_alias_reduction_artifact_to_dict,
    experiment_budget_to_dict,
    production_load_plan_from_dict,
    production_load_plan_to_dict,
)
from lightcone_spec.experiments.registry import (
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.statistics import HardwareEnvelope
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.orchestration.executor import (
    ArtifactBinding,
    industrial_execution_split_contract,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _bound(path: Path) -> BoundArtifact:
    return BoundArtifact(path, hashlib.sha256(path.read_bytes()).hexdigest())


def _write_json(path: Path, value: object) -> BoundArtifact:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return _bound(path)


def _semantics() -> ExecutionDerivedAliasSemantics:
    return ExecutionDerivedAliasSemantics(
        schema_version=1,
        target_model="Qwen/Qwen3-8B",
        target_revision="1" * 40,
        runtime_authority_sha256=_sha("runtime-authority"),
        patched_tree_identity="2" * 40,
        run_config_sha256=_sha("run-config"),
        sampling_profile_sha256=_sha("sampling"),
        seed=20260811,
        load_plan_sha256=_sha("load"),
        warmup_corpus_sha256=None,
        request_corpus_sha256=_sha("corpus"),
        arrival_trace_sha256=_sha("arrivals"),
        request_ids_sha256=_sha("request-ids"),
        maximum_context_tokens=4096,
        maximum_output_tokens=256,
        split_semantics_sha256=_sha("split"),
        model_lock_sha256=_sha("model-lock"),
        experiment_budget_semantics_sha256=_sha("budget"),
        hardware_envelope_sha256=_sha("hardware"),
        inventory_sha256=_sha("inventory"),
        inventory_source_receipt_sha256=_sha("inventory-receipt"),
        fixed_instance_gpu_count=2,
        topology="tp1_dp1",
        rank_layout_sha256=_sha("rank-layout"),
        method="target_only",
        method_implementation_sha256=_sha("implementation"),
        server_config_sha256=_sha("server"),
        evidence_schema="schema_v3_native_terminal_v1",
        output_token_contract_sha256=_sha("outputs"),
        timing_contract_sha256=_sha("timing"),
    )


def _run_binding(cell_id: str) -> RawEvidenceRunBinding:
    return RawEvidenceRunBinding(
        schema_version=3,
        cell_id=cell_id,
        experiment="E3b",
        method="target_only",
        scientific_role="target_only",
        scientific_unit="evidence_alias:block=4",
        config_sha256=_sha("config"),
        rank_config_sha256s=(_sha("rank-config"),),
        run_id="industrial-alias-source",
        rank_count=1,
        model_pair="Qwen/Qwen3-8B",
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("run-split"),
        corpus_sha256=_sha("run-corpus"),
        arrival_trace_sha256=_sha("run-arrivals"),
        request_ids_sha256=_sha("run-request-ids"),
        sampling_profile_sha256=_sha("run-sampling"),
        model_lock_sha256=_sha("run-model-lock"),
        patched_sglang_tree="2" * 40,
        run_nonce_sha256=_sha("nonce"),
        topology_sha256=_sha("topology"),
        experiment_budget_sha256=_sha("run-budget"),
        physical_gpu_uuids=("GPU-alias",),
        terminal_receipt_sha256s=(_sha("terminal-receipt"),),
        hardware_receipt_sha256=_sha("hardware-receipt"),
        budget_observation_sha256=_sha("budget-observation"),
        execution_plan_sha256=_sha("execution-plan"),
        execution_split_sha256=_sha("execution-split"),
    )


def _alias_artifact(
    *,
    source_cell_id: str | None = None,
    target_cell_id: str | None = None,
) -> EvidenceAliasReductionArtifact:
    source = source_cell_id or _sha("source-cell")
    target = target_cell_id or _sha("target-cell")
    semantics = _semantics()
    return EvidenceAliasReductionArtifact(
        schema_version=1,
        registry_sha256=_sha("registry"),
        source_cell_id=source,
        target_cell_id=target,
        source_cell_declaration_sha256=_sha(f"source-declaration:{source}"),
        target_cell_declaration_sha256=_sha(f"target-declaration:{target}"),
        source_execution_plan_file_sha256=_sha("source-plan-file"),
        source_execution_plan_sha256=_sha("source-plan"),
        target_execution_plan_file_sha256=_sha("target-plan-file"),
        target_execution_plan_sha256=_sha("target-plan"),
        raw_manifest_sha256=_sha("raw-manifest"),
        source_semantics=semantics,
        target_semantics=semantics,
        source_run_binding=_run_binding(source),
        source_native_terminal_sha256s=(_sha("native-terminal"),),
        removed_presentation_axis="analysis_panel",
        source_presentation_value="E3b:matched",
        target_presentation_value="E3b:deployment-optimal",
        reason_code="target_only_cross_analysis_reference",
        target_result_status="ABSENT_REUSED_SOURCE",
        reducer_protocol_sha256=EVIDENCE_ALIAS_REDUCER_PROTOCOL_SHA256,
    )


def _manifest(tmp_path: Path) -> RawEvidenceAliasManifest:
    references = {
        name: BoundArtifact(tmp_path / f"{name}.json", _sha(name))
        for name in (
            "execution-plan",
            "load-plan",
            "run-config",
            "split",
            "sampling",
            "model-lock",
            "budget",
            "budget-materialization-authority",
            "terminal",
            "native-terminal",
            "hardware",
            "budget-observation",
            "completion-contract",
            "inventory-source",
        )
    }
    artifacts = AliasExecutionArtifacts(
        execution_plan=references["execution-plan"],
        load_plan=references["load-plan"],
        run_config=references["run-config"],
        split_artifact=references["split"],
        sampling_artifact=references["sampling"],
        model_lock_artifact=references["model-lock"],
        experiment_budget=references["budget"],
        budget_materialization_authority=references["budget-materialization-authority"],
    )
    return RawEvidenceAliasManifest(
        schema_version=2,
        source=artifacts,
        target=artifacts,
        source_evidence=IndustrialCellEvidence(
            cell_id=_sha("source-cell"),
            terminal_receipts=(references["terminal"],),
            hardware_receipt=references["hardware"],
            budget_observation=references["budget-observation"],
            completion_contract=references["completion-contract"],
        ),
        source_native_terminal_artifacts=(references["native-terminal"],),
        inventory_source_receipt=references["inventory-source"],
        removed_presentation_axis="analysis_panel",
        reason_code="target_only_cross_analysis_reference",
    )


def test_reduction_artifact_is_legal_authority_and_strictly_serialized() -> None:
    artifact = _alias_artifact()
    dependence = build_evidence_dependence_map(
        direct_observation_cell_ids=(artifact.source_cell_id,),
        aliases=(artifact,),
    )
    assert dependence.independent_unit_count == 1
    assert dependence.unit_for(artifact.source_cell_id) == dependence.unit_for(
        artifact.target_cell_id
    )

    wire = evidence_alias_reduction_artifact_to_dict(artifact)
    sidecar = PlanningArtifactSidecar(
        schema_version=1,
        artifact_kind="evidence_alias_reduction_artifact",
        artifact_sha256=artifact.sha256,
    )
    assert (
        evidence_alias_reduction_artifact_from_dict(
            wire,
            sidecar=sidecar,
        )
        == artifact
    )

    edited = deepcopy(wire)
    edited["source_semantics"]["arrival_trace_sha256"] = _sha("edited")
    with pytest.raises(ValueError):
        evidence_alias_reduction_artifact_from_dict(edited)
    with pytest.raises(ValueError, match="sidecar"):
        evidence_alias_reduction_artifact_from_dict(
            wire,
            sidecar=replace(sidecar, artifact_sha256=_sha("wrong-sidecar")),
        )


def test_alias_duplicate_chain_and_independent_target_are_rejected() -> None:
    source = _sha("source-cell")
    target = _sha("target-cell")
    artifact = _alias_artifact(source_cell_id=source, target_cell_id=target)
    with pytest.raises(ValueError, match="only once"):
        build_evidence_dependence_map(
            direct_observation_cell_ids=(source,),
            aliases=(artifact, artifact),
        )
    with pytest.raises(ValueError, match="independent observation"):
        build_evidence_dependence_map(
            direct_observation_cell_ids=(source, target),
            aliases=(artifact,),
        )
    chained = _alias_artifact(
        source_cell_id=target,
        target_cell_id=_sha("chain-target"),
    )
    with pytest.raises(ValueError, match="chains"):
        build_evidence_dependence_map(
            direct_observation_cell_ids=(source,),
            aliases=(artifact, chained),
        )


def test_raw_manifest_roundtrip_rejects_missing_edited_and_nontext_fields(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    wire = raw_evidence_alias_manifest_to_dict(manifest)
    assert raw_evidence_alias_manifest_from_dict(wire) == manifest

    missing = deepcopy(wire)
    missing.pop("target")
    with pytest.raises(ValueError, match="ambiguous"):
        raw_evidence_alias_manifest_from_dict(missing)
    missing_budget_authority = deepcopy(wire)
    missing_budget_authority["source"].pop("budget_materialization_authority")
    with pytest.raises(ValueError, match="BLOCKED.*materialization authority"):
        raw_evidence_alias_manifest_from_dict(missing_budget_authority)
    edited = deepcopy(wire)
    edited["source"]["run_config"]["sha256"] = _sha("edited-run-config")
    with pytest.raises(ValueError, match="redundant"):
        raw_evidence_alias_manifest_from_dict(edited)
    nontext = deepcopy(wire)
    nontext["reason_code"] = 7
    with pytest.raises(ValueError, match="strict"):
        raw_evidence_alias_manifest_from_dict(nontext)

    legacy = deepcopy(wire)
    legacy["schema_version"] = 1
    with pytest.raises(ValueError, match="BLOCKED.*materialization authority"):
        raw_evidence_alias_manifest_from_dict(legacy)


def test_bootstrap_and_covariance_use_the_same_alias_dependence_unit() -> None:
    source = _sha("block-0-target")
    target = _sha("block-1-target")
    static_zero = _sha("block-0-static")
    static_one = _sha("block-1-static")
    artifact = _alias_artifact(source_cell_id=source, target_cell_id=target)
    dependence = build_evidence_dependence_map(
        direct_observation_cell_ids=(source, static_zero, static_one),
        aliases=(artifact,),
    )

    def block(index: int, target_id: str, static_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            block=index,
            cells={
                "target_only": SimpleNamespace(cell=SimpleNamespace(cell_id=target_id)),
                "static": SimpleNamespace(cell=SimpleNamespace(cell_id=static_id)),
            },
        )

    blocks = (block(0, source, static_zero), block(1, target, static_one))
    selected = _independent_method_blocks(
        blocks,
        method="target_only",
        dependence_map=dependence,
    )
    components = _paired_dependence_components(
        blocks,
        numerator="target_only",
        denominator="static",
        dependence_map=dependence,
    )
    assert len(selected) == 1
    assert selected[0][1].block == 0
    assert len(components) == 1
    assert tuple(row.block for row in components[0][1]) == (0, 1)


def _install_nonformal_executor_fixture_bridge(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> None:
    """Give the legacy executor fixture one path-bound audit authority."""

    import test_industrial_executor as fixture_module
    from test_execution_bundle import (
        _budget_load_binding,
        _budget_policy,
        _ensure_bound,
        _raw_capacity_authority,
        _write_bound,
    )

    from lightcone_spec.experiments.budget_authority import (
        bind_budget_materialization_authority,
    )
    from lightcone_spec.experiments.load import (
        FrozenSamplingParameters,
        ProductionLoadPlan,
        ProductionWindow,
        RequestTemplate,
        closed_loop_corpus,
    )
    from lightcone_spec.experiments.planning import (
        BudgetLoadBinding,
        budget_inventory_identity_from_gpu_inventory,
        materialize_industrial_budgets,
    )
    from lightcone_spec.experiments.planning_artifacts import (
        budget_load_binding_to_dict,
        budget_plan_to_dict,
        budget_policy_to_dict,
    )

    context_type = fixture_module.GpuDispatchExecutionContext
    build_execution_plan = fixture_module.build_industrial_execution_plan

    def context_factory(**kwargs):
        if "budget_plan" in kwargs:
            return context_type(**kwargs)
        registry = kwargs["registry"]
        inventory = kwargs["inventory"]
        activation = kwargs["activation_artifact"]
        supplied_budgets = kwargs["budgets"]
        inventory_identity = budget_inventory_identity_from_gpu_inventory(inventory)
        cells_by_id = {cell.cell_id: cell for cell in registry.cells}
        selected_budget = next(
            budget
            for budget in supplied_budgets
            if cells_by_id[budget.cell_id].identity.context == 1024
            and cells_by_id[budget.cell_id].identity.concurrency == 1
            and cells_by_id[budget.cell_id].identity.regime == "long_input_short_output"
        )
        selected_cell = cells_by_id[selected_budget.cell_id]
        request_count = selected_budget.output_tokens.maximum // 2
        sampling = FrozenSamplingParameters.from_mapping(
            {"temperature": 0.0, "top_p": 1.0}
        )
        selected_load = ProductionLoadPlan(
            warmup=None,
            scored=closed_loop_corpus(
                tuple(
                    RequestTemplate(
                        input_token_ids=tuple(range(768)),
                        requested_output_tokens=2,
                        sampling=sampling,
                    )
                    for _ in range(request_count)
                ),
                namespace="alias-audit-selected-load",
                split="tuning",
                concurrency=1,
                cohort_count=1,
                cohort_popularity="uniform",
                cohort_seed=7,
            ),
            window=ProductionWindow(
                warmup_duration_us=(selected_budget.excluded_warmup.registered * 1_000),
                arrival_duration_us=(selected_budget.scored_arrival.registered * 1_000),
                request_deadline_us=(
                    selected_budget.request_deadline.registered * 1_000
                ),
                drain_duration_us=selected_budget.drain.registered * 1_000,
            ),
        )
        load_bindings = tuple(
            sorted(
                [
                    replace(
                        _budget_load_binding(
                            cells_by_id[cell_id],
                            execution_cell_id=selected_cell.cell_id,
                        ),
                        optimistic_load=selected_load,
                        registered_load=selected_load,
                        quota_envelope_load=selected_load,
                        minimum_completed_requests=(
                            selected_budget.minimum_completed_requests
                        ),
                    )
                    if cells_by_id[cell_id].identity.context == 1024
                    and cells_by_id[cell_id].identity.concurrency == 1
                    and cells_by_id[cell_id].identity.regime
                    == "long_input_short_output"
                    else _budget_load_binding(
                        cells_by_id[cell_id],
                        execution_cell_id=selected_cell.cell_id,
                    )
                    for cell_id in activation.activated_cell_ids
                ],
                key=lambda value: value.cell_id,
            )
        )
        assert all(type(value) is BudgetLoadBinding for value in load_bindings)
        registry_path = _write_bound(
            root / "alias-authority-registry.json",
            {
                "schema_version": 3,
                "generator": (
                    "lightcone_spec.experiments.registry.build_industrial_registry:v3"
                ),
                "parameters": {
                    "logical_gpu_slots": list(registry.gpu_uuids),
                    "base_port": 28_000,
                    "cache_root": str(root / "cache"),
                    "evidence_root": str(root / "evidence"),
                    "seed": 20260811,
                },
                "registry_sha256": registry.sha256,
                "registry": registry.to_dict(),
            },
        ).resolve()
        inventory_path = _write_bound(
            root / "alias-authority-inventory.json", inventory.to_dict()
        ).resolve()
        _ensure_bound(root / "gpu-inventory-source-receipt.json")
        (
            capacity,
            capacity_path,
            _,
            capacity_verification_path,
            capacity_authority,
        ) = _raw_capacity_authority(
            root / "alias-raw-capacity",
            registry=registry,
            inventory=inventory,
            inventory_path=inventory_path,
            inventory_source_receipt_path=(
                root / "gpu-inventory-source-receipt.json"
            ).resolve(),
            cell_ids=activation.activated_cell_ids,
        )
        verification = json.loads(
            capacity_verification_path.read_text(encoding="utf-8")
        )
        authority_now_ns = verification["challenge"]["issued_ns"] + 1
        monkeypatch.setattr(
            "lightcone_spec.experiments.capacity_authority.time",
            SimpleNamespace(time_ns=lambda: authority_now_ns),
        )
        policy = _budget_policy()
        budget_plan = materialize_industrial_budgets(
            registry,
            activations=(activation,),
            load_bindings=load_bindings,
            policy=policy,
            inventory=inventory_identity,
            capacity_envelope=capacity,
            capacity_authority=capacity_authority,
            require_complete=False,
        )
        assert (
            next(
                value
                for value in budget_plan.diagnostic_budgets
                if value.cell_id == selected_cell.cell_id
            )
            == selected_budget
        )
        policy_path = _write_bound(
            root / "alias-budget-policy.json", budget_policy_to_dict(policy)
        ).resolve()
        load_paths = tuple(
            _write_bound(
                root / f"alias-budget-load-{index:03d}.json",
                budget_load_binding_to_dict(binding),
            ).resolve()
            for index, binding in enumerate(load_bindings)
        )
        budget_plan_path = _write_bound(
            root / "alias-declared-budget-plan.json",
            budget_plan_to_dict(budget_plan),
        ).resolve()
        runtime_path = _write_bound(
            root / "alias-activation-runtime.json", {"runtime": "E3a"}
        ).resolve()
        split_path = (root / "root-split.json").resolve()
        Path(f"{split_path}.sha256").write_text(
            content_sha256(json.loads(split_path.read_text(encoding="utf-8"))) + "\n",
            encoding="utf-8",
        )
        receipt = activation.dependency_receipts[0]
        receipt_path = _write_bound(
            root / "alias-activation-receipt.json", receipt.to_dict()
        ).resolve()
        activation_path = _write_bound(
            root / "alias-activation-manifest.json",
            {
                "schema_version": 1,
                "kind": "industrial_registry_stage_activation_manifest",
                "registry_artifact": str(registry_path),
                "experiment": "E3a",
                "runtime_artifact": str(runtime_path),
                "split_artifact": str(split_path),
                "dependency_receipts": [str(receipt_path)],
            },
        ).resolve()
        authority = bind_budget_materialization_authority(
            activation_manifest_path=activation_path,
            policy_path=policy_path,
            load_binding_paths=load_paths,
            capacity_envelope_path=capacity_path.resolve(),
            capacity_authority=capacity_authority,
            declared_plan_path=budget_plan_path,
        )
        return context_type(
            **{**kwargs, "budgets": budget_plan.diagnostic_budgets},
            budget_plan=budget_plan,
            budget_materialization_authority=authority,
        )

    def build_plan_bridge(**kwargs):
        kwargs["budget_plan"] = kwargs["dispatch_context"].budget_plan
        return build_execution_plan(**kwargs)

    monkeypatch.setattr(
        context_type,
        "require_ready_budget_authority",
        lambda self: self.budgets,
    )
    monkeypatch.setattr(
        fixture_module,
        "GpuDispatchExecutionContext",
        context_factory,
    )
    monkeypatch.setattr(
        fixture_module,
        "build_industrial_execution_plan",
        build_plan_bridge,
    )


def test_execution_candidate_is_rebuilt_from_current_raw_plan_and_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    _install_nonformal_executor_fixture_bridge(monkeypatch, root)
    base = _execution_fixture(root).plan
    config = base.runtime_plan.rank_configs[0]
    model_lock = ModelLock(
        schema_version=2,
        models=(LockedModel(config.model.target, config.model.target_revision),),
    )
    model_lock_path = root / "alias-model-lock.json"
    model_lock.write(model_lock_path)
    model_lock_binding = ArtifactBinding.from_path(
        name="model-lock",
        path=model_lock_path,
        semantic_sha256=model_lock.sha256,
    )
    split_value = industrial_execution_split_contract(
        registry_sha256=base.runtime_plan.registry_sha256,
        cell=base.runtime_plan.cell,
        load_plan=base.load_plan,
        sampling_profile_sha256=base.sampling_artifact.content_sha256,
        model_lock_sha256=model_lock.sha256,
    )
    split_path = root / "alias-split.json"
    split_path.write_text(
        json.dumps(split_value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    split_binding = ArtifactBinding.from_path(
        name="split",
        path=split_path,
        semantic_sha256=content_sha256(split_value),
    )
    plan = replace(
        base,
        split_artifact=split_binding,
        model_lock_artifact=model_lock_binding,
    )
    plan.validate()
    budget_materialization_authority = (
        plan.dispatch_context.budget_materialization_authority
    )
    artifacts = AliasExecutionArtifacts(
        execution_plan=_write_json(root / "alias-plan.json", plan.to_dict()),
        load_plan=_write_json(
            root / "alias-load.json",
            production_load_plan_to_dict(plan.load_plan),
        ),
        run_config=_bound(Path(plan.server_launch.run_config)),
        split_artifact=_bound(Path(plan.split_artifact.path)),
        sampling_artifact=_bound(Path(plan.sampling_artifact.path)),
        model_lock_artifact=_bound(Path(plan.model_lock_artifact.path)),
        experiment_budget=_write_json(
            root / "alias-budget.json",
            experiment_budget_to_dict(plan.budget),
        ),
        budget_materialization_authority=_write_json(
            root / "alias-budget-materialization-authority.json",
            budget_materialization_authority_binding_to_dict(
                budget_materialization_authority
            ),
        ),
    )
    registry = build_industrial_registry(
        gpu_uuids=("GPU-executor-a", "GPU-executor-b"),
        cache_root=str(root / "cache"),
        evidence_root=str(root / "evidence"),
        base_port=28000,
    )
    envelope = HardwareEnvelope(
        gpu_clock_mhz_min=1500.0,
        gpu_clock_mhz_max=2100.0,
        memory_clock_mhz_min=1000.0,
        memory_clock_mhz_max=1500.0,
        temperature_c_max=80.0,
        power_watts_min=100.0,
        power_watts_max=600.0,
        power_state="P0",
    )
    candidate = _audit_alias_execution_candidate(
        artifacts,
        registry=registry,
        hardware_envelope=envelope,
        inventory=plan.dispatch_context.inventory,
    )
    assert candidate.cell == plan.runtime_plan.cell
    assert candidate.semantics.target_model == config.model.target
    assert candidate.semantics.model_lock_sha256 == model_lock.sha256
    assert candidate.budget_plan == plan.budget_plan
    plan_wire = plan.to_dict()
    assert plan_wire["runtime_plan"]["schema_version"] == 3
    assert plan_wire["runtime_plan"]["execution_semantics_sha256"] is None
    assert plan_wire["runtime_plan"]["execution_semantics"] is None
    assert plan_wire["schema_version"] == 5
    assert plan_wire["itl_timestamp_authority"] is None

    old_schema = deepcopy(plan_wire)
    old_schema["schema_version"] = 4
    with pytest.raises(ValueError, match="execution plan schema 5"):
        _audit_alias_execution_candidate(
            replace(
                artifacts,
                execution_plan=_write_json(
                    root / "alias-plan-old-schema.json",
                    old_schema,
                ),
            ),
            registry=registry,
            hardware_envelope=envelope,
            inventory=plan.dispatch_context.inventory,
        )

    non_e2_itl = deepcopy(plan_wire)
    non_e2_itl["itl_timestamp_authority"] = {
        "plan_sha256": _sha("caller-itl-plan"),
        "producer_sha256": _sha("caller-itl-producer"),
        "protocol_sha256": _sha("caller-itl-protocol"),
    }
    with pytest.raises(ValueError, match="non-E2 alias execution must not carry ITL"):
        _audit_alias_execution_candidate(
            replace(
                artifacts,
                execution_plan=_write_json(
                    root / "alias-plan-non-e2-itl.json",
                    non_e2_itl,
                ),
            ),
            registry=registry,
            hardware_envelope=envelope,
            inventory=plan.dispatch_context.inventory,
        )

    e2_cell = next(cell for cell in registry.cells if cell.identity.experiment == "E2")
    foreign_e2_itl = deepcopy(plan_wire)
    foreign_e2_itl["runtime_plan"]["cell_id"] = e2_cell.cell_id
    foreign_e2_itl["runtime_plan"]["cell_declaration_sha256"] = e2_cell.sha256
    foreign_e2_itl["runtime_plan_sha256"] = content_sha256(
        foreign_e2_itl["runtime_plan"]
    )
    foreign_e2_itl["itl_timestamp_authority"] = {
        "plan_sha256": _sha("foreign-e2-itl-plan"),
        "producer_sha256": None,
        "protocol_sha256": _sha("foreign-e2-itl-protocol"),
    }
    with pytest.raises(ValueError, match="differs from the release plan"):
        _audit_alias_execution_candidate(
            replace(
                artifacts,
                execution_plan=_write_json(
                    root / "alias-plan-foreign-e2-itl.json",
                    foreign_e2_itl,
                ),
            ),
            registry=registry,
            hardware_envelope=envelope,
            inventory=plan.dispatch_context.inventory,
        )

    expected_e2_itl = deepcopy(foreign_e2_itl)
    expected_e2_itl_plan = release_e2_itl_timestamp_plan(registry, e2_cell)
    expected_e2_itl["itl_timestamp_authority"] = {
        "plan_sha256": expected_e2_itl_plan.sha256,
        "producer_sha256": (
            None
            if expected_e2_itl_plan.producer is None
            else expected_e2_itl_plan.producer.sha256
        ),
        "protocol_sha256": expected_e2_itl_plan.protocol_sha256,
    }
    missing_budget_authority = BoundArtifact(
        (root / "missing-budget-materialization-authority.json").resolve(),
        _sha("missing-budget-materialization-authority"),
    )
    with pytest.raises(ItlTimestampAuthorityBlocked):
        _audit_alias_execution_candidate(
            replace(
                artifacts,
                execution_plan=_write_json(
                    root / "alias-plan-release-e2-itl.json",
                    expected_e2_itl,
                ),
                budget_materialization_authority=missing_budget_authority,
            ),
            registry=registry,
            hardware_envelope=envelope,
            inventory=plan.dispatch_context.inventory,
        )

    semantic_alias_tamper = deepcopy(plan_wire)
    semantic_alias_tamper["runtime_plan"]["execution_semantics_sha256"] = _sha(
        "caller-authored-alias-semantics"
    )
    semantic_alias_tamper["runtime_plan"]["execution_semantics"] = {
        "schema_version": 1,
        "sha256": semantic_alias_tamper["runtime_plan"]["execution_semantics_sha256"],
    }
    semantic_alias_tamper["runtime_plan_sha256"] = content_sha256(
        semantic_alias_tamper["runtime_plan"]
    )
    with pytest.raises(
        ValueError,
        match="current_release_semantic_alias_authority_unavailable",
    ):
        _audit_alias_execution_candidate(
            replace(
                artifacts,
                execution_plan=_write_json(
                    root / "alias-plan-semantics-tamper.json",
                    semantic_alias_tamper,
                ),
            ),
            registry=registry,
            hardware_envelope=envelope,
            inventory=plan.dispatch_context.inventory,
        )

    with pytest.raises(
        BudgetMaterializationBlockedError,
        match="dependency_completion_manifest_authority_missing",
    ):
        _load_alias_execution_candidate(
            artifacts,
            registry=registry,
            hardware_envelope=envelope,
            inventory=plan.dispatch_context.inventory,
        )

    for name, field, value in (
        ("trainable-authority", "trainable_plan_authority", {}),
        (
            "prepared-content-pin",
            "prepared_model_content_release_manifest_sha256",
            _sha("caller-prepared-content-pin"),
        ),
    ):
        trainable_tamper = deepcopy(plan_wire)
        trainable_tamper[field] = value
        with pytest.raises(
            ValueError,
            match="must not carry trainable-plan authority",
        ):
            _audit_alias_execution_candidate(
                replace(
                    artifacts,
                    execution_plan=_write_json(
                        root / f"alias-plan-{name}-tamper.json",
                        trainable_tamper,
                    ),
                ),
                registry=registry,
                hardware_envelope=envelope,
                inventory=plan.dispatch_context.inventory,
            )
    top_level_tamper = deepcopy(plan_wire)
    top_level_tamper["budget_materialization_authority_sha256"] = _sha(
        "edited-top-level-budget-authority"
    )
    context_tamper = deepcopy(plan_wire)
    context_tamper["dispatch_authority"]["budget_materialization_authority_sha256"] = (
        _sha("edited-context-budget-authority")
    )
    context_tamper["dispatch_context_sha256"] = content_sha256(
        context_tamper["dispatch_authority"]
    )
    physical_tamper = deepcopy(plan_wire)
    physical_binding = physical_tamper["runtime_plan"]["resource_binding"]
    physical_binding["physical_assignment"][
        "budget_materialization_authority_sha256"
    ] = _sha("edited-physical-budget-authority")
    physical_binding["physical_binding_sha256"] = content_sha256(
        physical_binding["physical_assignment"]
    )
    physical_tamper["runtime_plan_sha256"] = content_sha256(
        physical_tamper["runtime_plan"]
    )
    for name, value in (
        ("top-level", top_level_tamper),
        ("context", context_tamper),
        ("physical", physical_tamper),
    ):
        with pytest.raises(
            ValueError,
            match="raw budget materialization authority",
        ):
            _audit_alias_execution_candidate(
                replace(
                    artifacts,
                    execution_plan=_write_json(
                        root / f"alias-plan-{name}-tamper.json",
                        value,
                    ),
                ),
                registry=registry,
                hardware_envelope=envelope,
                inventory=plan.dispatch_context.inventory,
            )

    load_wire = production_load_plan_to_dict(plan.load_plan)
    load_sidecar = PlanningArtifactSidecar(
        schema_version=1,
        artifact_kind="production_load_plan",
        artifact_sha256=plan.load_plan.paired_replay_sha256,
    )
    assert (
        production_load_plan_from_dict(
            load_wire,
            sidecar=load_sidecar,
        )
        == plan.load_plan
    )

    tampered_config = json.loads(Path(plan.server_launch.run_config).read_text())
    tampered_config["tenant_id"] = "edited"
    edited_run_config = _write_json(root / "edited-run-config.json", tampered_config)
    with pytest.raises(ValueError, match="RunConfig differs"):
        _audit_alias_execution_candidate(
            replace(artifacts, run_config=edited_run_config),
            registry=registry,
            hardware_envelope=envelope,
            inventory=plan.dispatch_context.inventory,
        )

    edited_authority = budget_materialization_authority_binding_to_dict(
        budget_materialization_authority
    )
    edited_authority["registry_sha256"] = _sha("edited-budget-registry")
    with pytest.raises(ValueError, match="strict artifact"):
        _audit_alias_execution_candidate(
            replace(
                artifacts,
                budget_materialization_authority=_write_json(
                    root / "edited-budget-materialization-authority.json",
                    edited_authority,
                ),
            ),
            registry=registry,
            hardware_envelope=envelope,
            inventory=plan.dispatch_context.inventory,
        )

    declared_plan_path = Path(budget_materialization_authority.declared_plan.path)
    edited_declared_plan = json.loads(declared_plan_path.read_text(encoding="utf-8"))
    edited_declared_plan["artifact_sha256"] = _sha("edited-declared-budget-plan")
    _write_json(declared_plan_path, edited_declared_plan)
    Path(f"{declared_plan_path}.sha256").write_text(
        content_sha256(edited_declared_plan) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="source or sidecar changed"):
        _audit_alias_execution_candidate(
            artifacts,
            registry=registry,
            hardware_envelope=envelope,
            inventory=plan.dispatch_context.inventory,
        )
