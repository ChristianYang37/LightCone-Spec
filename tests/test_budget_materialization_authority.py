from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from test_execution_bundle import _bundle_fixture

from lightcone_spec.experiments.budget_authority import (
    DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON,
    BudgetMaterializationBlockedError,
    bind_budget_materialization_authority,
    bind_budget_raw_json,
    load_budget_raw_json,
    load_declared_budget_plan,
    replay_registry_stage_activation_authority,
    revalidate_budget_materialization_authority_binding,
)
from lightcone_spec.experiments.capacity_authority import bind_capacity_authority
from lightcone_spec.experiments.gpu_pool import (
    GpuDispatchExecutionContext,
    GpuDispatchPlanningContext,
    GpuInventory,
    InterferenceEnvelope,
)
from lightcone_spec.experiments.load import ImmutableRequest, RequestTemplate
from lightcone_spec.experiments.planning import (
    BudgetLoadRawBinding,
    BudgetMaterializationAuthorityBinding,
    BudgetPolicy,
    ScenarioMilliseconds,
)
from lightcone_spec.experiments.planning_artifacts import (
    budget_load_binding_from_dict,
    budget_load_binding_to_dict,
    budget_materialization_authority_binding_from_dict,
    budget_materialization_authority_binding_to_dict,
    budget_plan_to_dict,
    budget_policy_from_dict,
    budget_policy_to_dict,
)
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.experiments.stage_activation import (
    materialize_registry_stage_activation,
)


def _write_bound(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(f"{path}.sha256").write_text(
        content_sha256(value) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class _AuthorityFixture:
    root: Path
    authority: BudgetMaterializationAuthorityBinding
    registry: object
    inventory: GpuInventory
    activation: object
    plan: object
    planning: GpuDispatchPlanningContext
    execution: GpuDispatchExecutionContext


def _build_authority_fixture(root: Path) -> _AuthorityFixture:
    root = root.resolve()
    _, bundle = _bundle_fixture(root)
    capacity_authority = bind_capacity_authority(
        bundle.capacity_source_manifest.path,
        bundle.capacity_verification_receipt.path,
    )
    authority = bind_budget_materialization_authority(
        activation_manifest_path=bundle.activation.path,
        policy_path=bundle.budget_policy.path,
        load_binding_paths=tuple(source.path for source in bundle.budget_load_bindings),
        capacity_envelope_path=bundle.capacity_envelope.path,
        capacity_authority=capacity_authority,
        declared_plan_path=bundle.budget_plan.path,
    )
    registry, activation = replay_registry_stage_activation_authority(
        authority.activation
    )
    plan = load_declared_budget_plan(authority)
    inventory = GpuInventory.from_dict(bundle.inventory.load())
    interference = InterferenceEnvelope.from_dict(bundle.interference_envelope.load())
    context = {
        "registry": registry,
        "inventory": inventory,
        "interference_envelope": interference,
        "budgets": plan.diagnostic_budgets,
        "receipts": activation.dependency_receipts,
        "activation_artifact": activation,
        "port_start": 24_000,
        "port_end": 65_535,
        "seed": 0,
    }
    planning = GpuDispatchPlanningContext(**context)
    execution = GpuDispatchExecutionContext(
        **context,
        budget_plan=plan,
        budget_materialization_authority=authority,
    )
    return _AuthorityFixture(
        root=root,
        authority=authority,
        registry=registry,
        inventory=inventory,
        activation=activation,
        plan=plan,
        planning=planning,
        execution=execution,
    )


@pytest.fixture(scope="module")
def authority_fixture(tmp_path_factory: pytest.TempPathFactory) -> _AuthorityFixture:
    return _build_authority_fixture(
        tmp_path_factory.mktemp("budget-materialization-authority")
    )


def _revalidate(
    fixture: _AuthorityFixture,
    authority: BudgetMaterializationAuthorityBinding,
):
    return revalidate_budget_materialization_authority_binding(
        authority,
        expected_registry=fixture.registry,
        expected_inventory=fixture.plan.inventory,
        expected_activation=fixture.activation,
        expected_plan=fixture.plan,
    )


def _mutate_and_restore(
    source: Path,
    value: object,
):
    original_body = source.read_bytes()
    sidecar = Path(f"{source}.sha256")
    original_sidecar = sidecar.read_bytes()
    _write_bound(source, value)

    def restore() -> None:
        source.write_bytes(original_body)
        sidecar.write_bytes(original_sidecar)

    return restore


def test_context_reopens_exact_authority_and_formal_dependency_is_blocked(
    authority_fixture: _AuthorityFixture,
) -> None:
    fixture = authority_fixture
    wire = budget_materialization_authority_binding_to_dict(fixture.authority)
    assert budget_materialization_authority_binding_from_dict(wire) == fixture.authority
    result = _revalidate(fixture, fixture.authority)
    assert result.budget_plan == fixture.plan
    assert len(result.load_bindings) == len(fixture.plan.activated_cell_ids) == 360
    summary = fixture.execution.authority_dict()
    assert summary["schema_version"] == 4
    assert summary["interference_calibration_authority_sha256"] is None
    assert summary["interference_calibration_bootstrap_authority_sha256"] is None
    assert summary["budget_materialization_authority_sha256"] == (
        fixture.authority.sha256
    )
    fixture.planning.issue_plan()
    with pytest.raises(BudgetMaterializationBlockedError) as blocked:
        fixture.execution.issue_plan()
    assert blocked.value.reason_code == (
        DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON
    )


def test_same_count_different_corpus_and_policy_joint_rehash_reject(
    authority_fixture: _AuthorityFixture,
) -> None:
    fixture = authority_fixture
    authority = fixture.authority

    raw_load = authority.load_bindings[0]
    load_path = Path(raw_load.source.path)
    load = budget_load_binding_from_dict(json.loads(load_path.read_text()))
    scored = load.registered_load.scored
    request = scored.requests[0]
    tokens = request.input_token_ids
    changed_tokens = (tokens[0] + 1_000_000, *tokens[1:])
    changed_request = ImmutableRequest.create(
        namespace=request.namespace,
        split=request.split,
        ordinal=request.ordinal,
        template=RequestTemplate(
            input_token_ids=changed_tokens,
            requested_output_tokens=request.requested_output_tokens,
            sampling=request.sampling,
            cancellation_offset_us=request.cancellation_offset_us,
        ),
        arrival_us=request.arrival_us,
        cohort_id=request.cohort_id,
    )
    changed_corpus = replace(
        scored,
        requests=(changed_request, *scored.requests[1:]),
    )
    changed_plan = replace(load.registered_load, scored=changed_corpus)
    changed_load = replace(
        load,
        optimistic_load=changed_plan,
        registered_load=changed_plan,
        quota_envelope_load=changed_plan,
    )
    assert len(changed_corpus.requests) == len(scored.requests)
    assert (
        budget_load_binding_from_dict(budget_load_binding_to_dict(changed_load))
        == changed_load
    )
    restore = _mutate_and_restore(
        load_path,
        budget_load_binding_to_dict(changed_load),
    )
    try:
        changed_source = bind_budget_raw_json(
            load_path,
            role="budget_load_binding",
        )
        changed_raw_load = BudgetLoadRawBinding(
            cell_id=raw_load.cell_id,
            source=changed_source,
        )
        changed_raw_loads = (changed_raw_load, *authority.load_bindings[1:])
        forged = replace(
            authority,
            load_bindings=changed_raw_loads,
            budget_load_binding_sha256s=tuple(
                value.source.semantic_sha256 for value in changed_raw_loads
            ),
        )
        with pytest.raises(ValueError, match="first-party raw rematerialization"):
            _revalidate(fixture, forged)
    finally:
        restore()

    policy_path = Path(authority.policy.path)
    policy = budget_policy_from_dict(json.loads(policy_path.read_text()))
    row = policy.job_policies[0]
    duration = row.startup_model_load
    changed_duration = ScenarioMilliseconds(
        duration.optimistic + 1,
        duration.registered + 1,
        duration.quota_envelope + 1,
    )
    changed_policy = replace(
        policy,
        job_policies=(
            replace(row, startup_model_load=changed_duration),
            *policy.job_policies[1:],
        ),
    )
    assert type(changed_policy) is BudgetPolicy
    restore = _mutate_and_restore(
        policy_path,
        budget_policy_to_dict(changed_policy),
    )
    try:
        changed_source = bind_budget_raw_json(policy_path, role="budget_policy")
        forged = replace(
            authority,
            policy=changed_source,
            budget_policy_sha256=changed_source.semantic_sha256,
        )
        with pytest.raises(ValueError, match="first-party raw rematerialization"):
            _revalidate(fixture, forged)
    finally:
        restore()


@pytest.mark.parametrize("source_name", ("runtime", "split"))
def test_activation_runtime_or_split_coordinated_edit_rejects(
    authority_fixture: _AuthorityFixture,
    source_name: str,
) -> None:
    fixture = authority_fixture
    authority = fixture.authority
    old_activation = authority.activation
    old_source = getattr(old_activation, source_name)
    source_path = Path(old_source.path)
    value = json.loads(source_path.read_text())
    assert type(value) is dict
    changed_value = {**value, "budget_authority_tamper": source_name}
    restore = _mutate_and_restore(source_path, changed_value)
    try:
        changed_source = bind_budget_raw_json(
            source_path,
            role=f"activation_{source_name}",
        )
        runtime = changed_source if source_name == "runtime" else old_activation.runtime
        split = changed_source if source_name == "split" else old_activation.split
        changed_activation = materialize_registry_stage_activation(
            fixture.registry,
            experiment=fixture.activation.experiment,
            dependency_receipts=fixture.activation.dependency_receipts,
            runtime_sha256=runtime.canonical_sha256,
            split_sha256=split.canonical_sha256,
        )
        changed_activation_binding = replace(
            old_activation,
            **{
                source_name: changed_source,
                "activation_sha256": changed_activation.sha256,
            },
        )
        forged = replace(
            authority,
            activation=changed_activation_binding,
            activation_sha256=changed_activation.sha256,
        )
        with pytest.raises(ValueError, match="differs from execution"):
            _revalidate(fixture, forged)
    finally:
        restore()


def test_activation_manifest_coordinated_edit_rejects(
    authority_fixture: _AuthorityFixture,
) -> None:
    fixture = authority_fixture
    authority = fixture.authority
    manifest_path = Path(authority.activation.manifest.path)
    manifest = json.loads(manifest_path.read_text())
    assert type(manifest) is dict
    forged_runtime_path = (fixture.root / "forged-activation-runtime.json").resolve()
    _write_bound(forged_runtime_path, {"runtime": "coordinated manifest edit"})
    changed_manifest = {
        **manifest,
        "runtime_artifact": str(forged_runtime_path),
    }
    restore = _mutate_and_restore(manifest_path, changed_manifest)
    try:
        changed_manifest_source = bind_budget_raw_json(
            manifest_path,
            role="registry_stage_activation_manifest",
        )
        changed_runtime_source = bind_budget_raw_json(
            forged_runtime_path,
            role="activation_runtime",
        )
        changed_activation = materialize_registry_stage_activation(
            fixture.registry,
            experiment=fixture.activation.experiment,
            dependency_receipts=fixture.activation.dependency_receipts,
            runtime_sha256=changed_runtime_source.canonical_sha256,
            split_sha256=authority.activation.split.canonical_sha256,
        )
        changed_activation_binding = replace(
            authority.activation,
            manifest=changed_manifest_source,
            runtime=changed_runtime_source,
            activation_sha256=changed_activation.sha256,
        )
        forged = replace(
            authority,
            activation=changed_activation_binding,
            activation_sha256=changed_activation.sha256,
        )
        with pytest.raises(ValueError, match="differs from execution"):
            _revalidate(fixture, forged)
    finally:
        restore()


def test_forged_dependency_and_declared_plan_joint_rehash_reject(
    authority_fixture: _AuthorityFixture,
) -> None:
    fixture = authority_fixture
    authority = fixture.authority
    receipt = fixture.activation.dependency_receipts[0]
    receipt_path = Path(authority.activation.dependency_receipts[0].path)
    forged_receipt = replace(
        receipt,
        outputs=(
            replace(
                receipt.outputs[0],
                content_sha256=content_sha256("forged dependency output"),
            ),
            *receipt.outputs[1:],
        ),
    )
    restore = _mutate_and_restore(receipt_path, forged_receipt.to_dict())
    try:
        forged_receipt_source = bind_budget_raw_json(
            receipt_path,
            role="activation_dependency_receipt",
        )
        forged_activation = materialize_registry_stage_activation(
            fixture.registry,
            experiment=fixture.activation.experiment,
            dependency_receipts=(forged_receipt,),
            runtime_sha256=authority.activation.runtime.canonical_sha256,
            split_sha256=authority.activation.split.canonical_sha256,
        )
        forged_activation_binding = replace(
            authority.activation,
            dependency_receipts=(forged_receipt_source,),
            activation_sha256=forged_activation.sha256,
        )
        forged = replace(
            authority,
            activation=forged_activation_binding,
            activation_sha256=forged_activation.sha256,
        )
        with pytest.raises(ValueError, match="differs from execution"):
            _revalidate(fixture, forged)
    finally:
        restore()

    plan_path = Path(authority.declared_plan.path)
    policy = fixture.plan.policy
    duration = policy.job_policies[0].startup_model_load
    changed_policy = replace(
        policy,
        job_policies=(
            replace(
                policy.job_policies[0],
                startup_model_load=ScenarioMilliseconds(
                    duration.optimistic + 1,
                    duration.registered + 1,
                    duration.quota_envelope + 1,
                ),
            ),
            *policy.job_policies[1:],
        ),
    )
    forged_plan = replace(fixture.plan, policy=changed_policy)
    restore = _mutate_and_restore(plan_path, budget_plan_to_dict(forged_plan))
    try:
        forged_plan_source = bind_budget_raw_json(
            plan_path,
            role="declared_budget_plan",
        )
        forged = replace(
            authority,
            declared_plan=forged_plan_source,
            declared_plan_sha256=forged_plan_source.semantic_sha256,
        )
        with pytest.raises(ValueError, match="first-party raw rematerialization"):
            revalidate_budget_materialization_authority_binding(
                forged,
                expected_registry=fixture.registry,
                expected_inventory=fixture.plan.inventory,
                expected_activation=fixture.activation,
            )
    finally:
        restore()


def test_raw_source_sidecar_symlink_missing_duplicate_and_nonfinite_reject(
    authority_fixture: _AuthorityFixture,
) -> None:
    root = authority_fixture.root / "raw-boundary"
    root.mkdir()
    bound_load = authority_fixture.authority.load_bindings[0].source
    load_path = Path(bound_load.path)
    load_sidecar_path = Path(bound_load.sidecar_path)

    missing_source = (root / "missing-load.json").resolve()
    missing_source.write_bytes(load_path.read_bytes())
    Path(f"{missing_source}.sha256").write_bytes(load_sidecar_path.read_bytes())
    binding = bind_budget_raw_json(missing_source, role="budget_load_binding")
    missing_source.unlink()
    with pytest.raises(RuntimeError, match="source is missing"):
        load_budget_raw_json(binding)

    symlink = root / "load-source-symlink.json"
    symlink.symlink_to(load_path)
    with pytest.raises(ValueError, match="non-symlink"):
        bind_budget_raw_json(symlink, role="budget_load_binding")

    sidecar_source = (root / "load-sidecar-symlink.json").resolve()
    sidecar_source.write_bytes(load_path.read_bytes())
    Path(f"{sidecar_source}.sha256").symlink_to(load_sidecar_path)
    with pytest.raises(ValueError, match="non-symlink"):
        bind_budget_raw_json(sidecar_source, role="budget_load_binding")

    duplicate = (root / "duplicate.json").resolve()
    duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
    Path(f"{duplicate}.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        bind_budget_raw_json(duplicate, role="activation_runtime")

    nonfinite = (root / "nonfinite.json").resolve()
    nonfinite.write_text('{"x":1e999}\n', encoding="utf-8")
    Path(f"{nonfinite}.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        bind_budget_raw_json(nonfinite, role="activation_runtime")
