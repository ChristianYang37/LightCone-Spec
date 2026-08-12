from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments.budget_authority import (
    E2_STAGE_COMPLETION_AUTHORITY_MISSING_REASON,
    BudgetMaterializationBlockedError,
    bind_budget_activation_authority,
)
from lightcone_spec.experiments.completion_authority import CompletedCellAuthority
from lightcone_spec.experiments.industrial_analysis import (
    BoundArtifact,
    IndustrialBlockEvidence,
    IndustrialCellEvidence,
    RawConfirmationFamilyPowerEvidenceManifest,
    RawE1ParetoEvidenceManifest,
    RawE2StageEvidenceManifest,
    RawE3aSelectionEvidenceManifest,
    _bound_json,
    raw_confirmation_family_power_manifest_from_dict,
    raw_confirmation_family_power_manifest_to_dict,
    raw_e1_pareto_manifest_from_dict,
    raw_e1_pareto_manifest_to_dict,
    raw_e2_stage_manifest_from_dict,
    raw_e2_stage_manifest_to_dict,
    raw_e3a_selection_manifest_from_dict,
    raw_e3a_selection_manifest_to_dict,
    validate_raw_evidence_manifest_sidecars,
)
from lightcone_spec.experiments.planning import (
    BUDGET_MATERIALIZATION_AUTHORITY_PROTOCOL_SHA256,
    CAPACITY_AUTHORITY_PROTOCOL_SHA256,
    CONFIRMATION_AUXILIARY_ACTIVATION_PROTOCOL_SHA256,
    BudgetMaterializationAuthorityBinding,
    BudgetRawJsonBinding,
    CapacityAuthorityBinding,
    CapacityRawJsonBinding,
    CellDisposition,
    ConfirmationAuxiliaryActivationAuthorityBinding,
    ConfirmationAuxiliaryCompletionAuthorityBinding,
    ConfirmationFamilyCompletionAuthorityBinding,
    ConfirmationFinalActivationAuthorityBinding,
    ConfirmationPilotActivationAuthorityBinding,
    ConfirmationStageAggregateAuthorityBinding,
    ConfirmationStageFamilyAuthorityBinding,
    DependencyGpuInventoryAuthorityBinding,
    DispositionStatus,
    E1ActivationAuthorityBinding,
    E2ActivationAuthorityBinding,
    E2StageCompletionAuthorityBinding,
    FamilyPilotCompletionAuthorityBinding,
    ReducerActivationArtifact,
    RegistryStageActivationAuthorityBinding,
    RegistryStageDependencyCompletionAuthorityBinding,
    materialize_confirmation_auxiliary_activation,
)
from lightcone_spec.experiments.planning_artifacts import (
    budget_materialization_authority_binding_from_dict,
    budget_materialization_authority_binding_to_dict,
    confirmation_auxiliary_activation_authority_binding_from_dict,
    confirmation_auxiliary_activation_authority_binding_to_dict,
    confirmation_auxiliary_completion_authority_binding_from_dict,
    confirmation_auxiliary_completion_authority_binding_to_dict,
    confirmation_family_completion_authority_binding_from_dict,
    confirmation_family_completion_authority_binding_to_dict,
    confirmation_final_activation_authority_binding_from_dict,
    confirmation_final_activation_authority_binding_to_dict,
    confirmation_pilot_activation_authority_binding_from_dict,
    confirmation_pilot_activation_authority_binding_to_dict,
    confirmation_stage_aggregate_authority_binding_from_dict,
    confirmation_stage_aggregate_authority_binding_to_dict,
    confirmation_stage_family_authority_binding_from_dict,
    confirmation_stage_family_authority_binding_to_dict,
    e1_activation_authority_binding_from_dict,
    e1_activation_authority_binding_to_dict,
    e2_activation_authority_binding_from_dict,
    e2_activation_authority_binding_to_dict,
    e2_stage_completion_authority_binding_from_dict,
    e2_stage_completion_authority_binding_to_dict,
    family_pilot_completion_authority_binding_from_dict,
    family_pilot_completion_authority_binding_to_dict,
)
from lightcone_spec.experiments.registry import (
    INDUSTRIAL_EXPERIMENT_ORDER,
    PILOT_BLOCKS,
    StageActivationPlan,
    content_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_bound(
    path: Path,
    value: object,
    *,
    sidecar_sha256: str | None = None,
) -> BoundArtifact:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(f"{sidecar_sha256 or digest}\n", encoding="ascii")
    return BoundArtifact(path=path.resolve(), sha256=digest)


def test_raw_selection_manifests_round_trip_and_require_exact_sidecars(
    tmp_path: Path,
) -> None:
    cell_id = _sha("cell")
    budget_semantic_sha256 = _sha("budget-observation")
    cell = IndustrialCellEvidence(
        cell_id=cell_id,
        terminal_receipts=(_write_bound(tmp_path / "terminal.json", {"terminal": 1}),),
        hardware_receipt=_write_bound(tmp_path / "hardware.json", {"hardware": 1}),
        budget_observation=_write_bound(
            tmp_path / "budget.json",
            {"budget_observation_sha256": budget_semantic_sha256},
            sidecar_sha256=budget_semantic_sha256,
        ),
        completion_contract=_write_bound(
            tmp_path / "completed.json",
            {"schema_version": 4, "kind": "industrial_completed_cells"},
            sidecar_sha256=content_sha256(
                {"schema_version": 4, "kind": "industrial_completed_cells"}
            ),
        ),
    )
    e3a = RawE3aSelectionEvidenceManifest(schema_version=2, cells=(cell,))
    e1 = RawE1ParetoEvidenceManifest(schema_version=2, cells=(cell,))
    e2 = RawE2StageEvidenceManifest(schema_version=2, stage_index=0, cells=(cell,))

    assert (
        raw_e3a_selection_manifest_from_dict(raw_e3a_selection_manifest_to_dict(e3a))
        == e3a
    )
    assert raw_e1_pareto_manifest_from_dict(raw_e1_pareto_manifest_to_dict(e1)) == e1
    assert raw_e2_stage_manifest_from_dict(raw_e2_stage_manifest_to_dict(e2)) == e2
    validate_raw_evidence_manifest_sidecars(e3a)

    Path(f"{cell.hardware_receipt.path}.sha256").write_text(
        f"{_sha('swapped')}\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="sidecar mismatch"):
        validate_raw_evidence_manifest_sidecars(e3a)

    Path(f"{cell.hardware_receipt.path}.sha256").write_text(
        f"{cell.hardware_receipt.sha256}\n", encoding="ascii"
    )
    Path(f"{cell.budget_observation.path}.sha256").write_text(
        f"{cell.budget_observation.sha256}\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="sidecar mismatch"):
        validate_raw_evidence_manifest_sidecars(e3a)


@pytest.mark.parametrize(
    "body, message",
    (
        (b'{"value":1,"value":2}\n', "duplicate JSON key"),
        (b'{"value":NaN}\n', "forbidden JSON constant"),
    ),
)
def test_shared_raw_json_loader_rejects_ambiguous_json(
    tmp_path: Path, body: bytes, message: str
) -> None:
    path = (tmp_path / "ambiguous.json").resolve()
    path.write_bytes(body)
    with pytest.raises(ValueError, match=message):
        _bound_json(
            path,
            hashlib.sha256(body).hexdigest(),
            label="ambiguous raw evidence",
        )


def test_raw_confirmation_manifest_binds_four_pilot_sidecar_sets(
    tmp_path: Path,
) -> None:
    blocks: list[IndustrialBlockEvidence] = []
    for block in PILOT_BLOCKS:
        budget_semantic_sha256 = _sha(f"budget-observation-{block}")
        cell = IndustrialCellEvidence(
            cell_id=_sha(f"pilot-cell-{block}"),
            terminal_receipts=(
                _write_bound(
                    tmp_path / f"pilot-{block}-terminal.json",
                    {"terminal": block},
                ),
            ),
            hardware_receipt=_write_bound(
                tmp_path / f"pilot-{block}-hardware.json",
                {"hardware": block},
            ),
            budget_observation=_write_bound(
                tmp_path / f"pilot-{block}-budget.json",
                {"budget_observation_sha256": budget_semantic_sha256},
                sidecar_sha256=budget_semantic_sha256,
            ),
            completion_contract=_write_bound(
                tmp_path / f"pilot-{block}-completed.json",
                {"schema_version": 4, "block": block},
                sidecar_sha256=content_sha256({"schema_version": 4, "block": block}),
            ),
        )
        blocks.append(
            IndustrialBlockEvidence(
                block=block,
                cells=(cell,),
                qualification_lock=_write_bound(
                    tmp_path / f"pilot-{block}-qualification.json",
                    {"qualification": block},
                ),
            )
        )
    manifest = RawConfirmationFamilyPowerEvidenceManifest(
        schema_version=2,
        blocks=tuple(blocks),
    )
    assert (
        raw_confirmation_family_power_manifest_from_dict(
            raw_confirmation_family_power_manifest_to_dict(manifest)
        )
        == manifest
    )
    validate_raw_evidence_manifest_sidecars(manifest)

    Path(f"{blocks[-1].qualification_lock.path}.sha256").write_text(
        f"{_sha('wrong-qualification')}\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="sidecar mismatch"):
        validate_raw_evidence_manifest_sidecars(manifest)


def _specialized_bindings(tmp_path: Path):
    index = 0

    def raw(role: str, *, semantic: str | None = None) -> BudgetRawJsonBinding:
        nonlocal index
        index += 1
        canonical = _sha(f"canonical-{index}")
        path = (tmp_path / f"raw-{index}.json").resolve()
        return BudgetRawJsonBinding(
            schema_version=1,
            role=role,
            path=str(path),
            sidecar_path=f"{path}.sha256",
            canonical_sha256=canonical,
            semantic_sha256=canonical if semantic is None else semantic,
            file_sha256=_sha(f"file-{index}"),
            sidecar_file_sha256=_sha(f"sidecar-{index}"),
            size=1,
            sidecar_size=65,
        )

    registry_sha256 = _sha("registry")
    inventory_sha256 = _sha("inventory")
    inventory_source_sha256 = _sha("inventory-source")
    generated_registry = raw("generated_registry", semantic=registry_sha256)
    inventory = DependencyGpuInventoryAuthorityBinding(
        schema_version=1,
        inventory=raw("dependency_gpu_inventory", semantic=inventory_sha256),
        source_receipt=raw(
            "dependency_gpu_inventory_source_receipt",
            semantic=inventory_source_sha256,
        ),
        inventory_sha256=inventory_sha256,
        source_receipt_sha256=inventory_source_sha256,
    )
    generic = RegistryStageActivationAuthorityBinding(
        schema_version=1,
        kind="registry_stage_activation_manifest",
        manifest=raw("registry_stage_activation_manifest"),
        generated_registry=generated_registry,
        runtime=raw("activation_runtime"),
        split=raw("activation_split"),
        dependency_receipts=(),
        dependency_completion_authorities=(),
        activation_sha256=_sha("generic-activation"),
    )

    e3a_receipt = raw("activation_dependency_receipt")
    e3a_completion = RegistryStageDependencyCompletionAuthorityBinding(
        schema_version=1,
        receipt=e3a_receipt,
        completed_cells=raw("dependency_completed_cells"),
        activation=generic,
        inventory_authority=inventory,
        locked_outputs=(),
        receipt_sha256=e3a_receipt.semantic_sha256,
        completed_authority_sha256=_sha("e3a-completed-authority"),
    )
    e1 = E1ActivationAuthorityBinding(
        schema_version=1,
        kind="e1_activation_manifest",
        manifest=raw("e1_activation_authority_manifest"),
        generated_registry=generated_registry,
        runtime=raw("activation_runtime"),
        split=raw("activation_split"),
        dependency_receipt=e3a_receipt,
        dependency_completion_authority=e3a_completion,
        selection_manifest=raw("e3a_selection_raw_manifest"),
        inventory_authority=inventory,
        hardware_envelope=raw("activation_hardware_envelope"),
        activation_sha256=_sha("e1-activation"),
        selection_sha256=_sha("selection"),
    )

    e1_receipt = raw("activation_dependency_receipt")
    e1_completion = RegistryStageDependencyCompletionAuthorityBinding(
        schema_version=1,
        receipt=e1_receipt,
        completed_cells=raw("dependency_completed_cells"),
        activation=e1,
        inventory_authority=inventory,
        locked_outputs=(),
        receipt_sha256=e1_receipt.semantic_sha256,
        completed_authority_sha256=_sha("e1-completed-authority"),
    )
    e2_stage_zero = E2ActivationAuthorityBinding(
        schema_version=1,
        kind="e2_activation_manifest",
        manifest=raw("e2_activation_authority_manifest"),
        generated_registry=generated_registry,
        runtime=raw("activation_runtime"),
        split=raw("activation_split"),
        dependency_receipt=e1_receipt,
        dependency_completion_authority=e1_completion,
        pareto_manifest=raw("e1_pareto_raw_manifest"),
        prior_stage_manifests=(),
        prior_stage_completion_authorities=(),
        inventory_authority=inventory,
        hardware_envelope=raw("activation_hardware_envelope"),
        stage_index=0,
        activation_sha256=_sha("e2-activation"),
        pareto_sha256=_sha("pareto"),
        prior_stage_reduction_sha256=None,
    )
    e2_stage_completion = E2StageCompletionAuthorityBinding(
        schema_version=1,
        completed_cells=raw("e2_stage_completed_cells"),
        stage_activation=e2_stage_zero,
        inventory_authority=inventory,
        completed_authority_sha256=_sha("e2-stage-completed-authority"),
    )
    e2 = E2ActivationAuthorityBinding(
        schema_version=1,
        kind="e2_activation_manifest",
        manifest=raw("e2_activation_authority_manifest"),
        generated_registry=generated_registry,
        runtime=e2_stage_zero.runtime,
        split=e2_stage_zero.split,
        dependency_receipt=e1_receipt,
        dependency_completion_authority=e1_completion,
        pareto_manifest=e2_stage_zero.pareto_manifest,
        prior_stage_manifests=(raw("e2_stage_raw_manifest"),),
        prior_stage_completion_authorities=(e2_stage_completion,),
        inventory_authority=inventory,
        hardware_envelope=e2_stage_zero.hardware_envelope,
        stage_index=1,
        activation_sha256=_sha("e2-stage-one-activation"),
        pareto_sha256=e2_stage_zero.pareto_sha256,
        prior_stage_reduction_sha256=_sha("e2-stage-zero-reduction"),
    )

    pilot = ConfirmationPilotActivationAuthorityBinding(
        schema_version=1,
        kind="confirmation_pilot_activation_manifest",
        manifest=raw("confirmation_pilot_activation_authority_manifest"),
        generated_registry=generated_registry,
        runtime=raw("activation_runtime"),
        split=raw("activation_split"),
        trace=raw("activation_trace"),
        sampling=raw("activation_sampling"),
        dependency_receipts=(e1_receipt,),
        dependency_completion_authorities=(e1_completion,),
        inventory_authority=inventory,
        hardware_envelope=raw("activation_hardware_envelope"),
        family_sha256=_sha("family"),
        activation_sha256=_sha("pilot-activation"),
    )
    pilot_completion = FamilyPilotCompletionAuthorityBinding(
        schema_version=1,
        completed_cells=raw("family_pilot_completed_cells"),
        pilot_activation=pilot,
        inventory_authority=inventory,
        completed_authority_sha256=_sha("pilot-completed-authority"),
    )
    auxiliary = ConfirmationAuxiliaryActivationAuthorityBinding(
        schema_version=1,
        kind="confirmation_auxiliary_activation_manifest",
        manifest=raw("confirmation_auxiliary_activation_authority_manifest"),
        generated_registry=generated_registry,
        runtime=pilot.runtime,
        split=pilot.split,
        trace=pilot.trace,
        sampling=pilot.sampling,
        dependency_receipts=pilot.dependency_receipts,
        dependency_completion_authorities=pilot.dependency_completion_authorities,
        inventory_authority=inventory,
        hardware_envelope=pilot.hardware_envelope,
        experiment="E5",
        activation_sha256=_sha("auxiliary-activation"),
    )
    auxiliary_completion = ConfirmationAuxiliaryCompletionAuthorityBinding(
        schema_version=1,
        completed_cells=raw("confirmation_auxiliary_completed_cells"),
        activation=auxiliary,
        inventory_authority=inventory,
        completed_authority_sha256=_sha("auxiliary-completed-authority"),
    )
    final = ConfirmationFinalActivationAuthorityBinding(
        schema_version=1,
        kind="confirmation_final_activation_manifest",
        manifest=raw("confirmation_final_activation_authority_manifest"),
        generated_registry=generated_registry,
        pilot_activation_authority=pilot,
        pilot_completion_authority=pilot_completion,
        power_manifest=raw("confirmation_family_power_raw_manifest"),
        family_sha256=pilot.family_sha256,
        power_reduction_sha256=_sha("family-power"),
        activation_sha256=_sha("final-activation"),
    )
    family_completion = ConfirmationFamilyCompletionAuthorityBinding(
        schema_version=1,
        completed_cells=raw("confirmation_family_completed_cells"),
        final_activation=final,
        inventory_authority=inventory,
        completed_authority_sha256=_sha("family-completed-authority"),
    )
    stage_family = ConfirmationStageFamilyAuthorityBinding(
        schema_version=1,
        family_sha256=final.family_sha256,
        final_activation_authority=final,
        completion_authority=family_completion,
    )
    stage_receipt = raw("activation_dependency_receipt")
    aggregate = ConfirmationStageAggregateAuthorityBinding(
        schema_version=1,
        kind="confirmation_stage_aggregate_manifest",
        manifest=raw("confirmation_stage_aggregate_authority_manifest"),
        generated_registry=generated_registry,
        stage_receipt=stage_receipt,
        stage_completed_cells=raw("dependency_completed_cells"),
        runtime=pilot.runtime,
        split=pilot.split,
        inventory_authority=inventory,
        experiment="E3b",
        families=(stage_family,),
        auxiliary_completion_authority=None,
        stage_receipt_sha256=stage_receipt.semantic_sha256,
        family_sha256s=(stage_family.family_sha256,),
        activated_cell_ids=(_sha("family-final-cell"),),
        dispositions_sha256=_sha("stage-dispositions"),
        activation_sha256=_sha("stage-aggregate-activation"),
    )

    capacity_envelope = raw("capacity_envelope", semantic=_sha("capacity"))

    def capacity_raw(label: str, semantic: str) -> CapacityRawJsonBinding:
        path = (tmp_path / f"{label}.json").resolve()
        return CapacityRawJsonBinding(
            schema_version=1,
            path=str(path),
            sidecar_path=f"{path}.sha256",
            semantic_sha256=semantic,
            file_sha256=_sha(f"{label}-file"),
            sidecar_file_sha256=_sha(f"{label}-sidecar"),
            size=1,
            sidecar_size=65,
        )

    capacity = CapacityAuthorityBinding(
        schema_version=1,
        source_manifest=capacity_raw("capacity-source", _sha("capacity-source")),
        verification_receipt=capacity_raw("capacity-receipt", _sha("capacity-receipt")),
        registry_sha256=registry_sha256,
        budget_inventory_sha256=_sha("budget-inventory"),
        capacity_envelope_sha256=capacity_envelope.semantic_sha256,
        gpu_inventory_sha256=inventory_sha256,
        inventory_source_receipt_sha256=inventory_source_sha256,
        trusted_verifier_policy_sha256=_sha("verifier-policy"),
        authority_protocol_sha256=CAPACITY_AUTHORITY_PROTOCOL_SHA256,
    )
    materialization = BudgetMaterializationAuthorityBinding(
        schema_version=1,
        activation=final,
        policy=raw("budget_policy", semantic=_sha("policy")),
        load_bindings=(),
        capacity_envelope=capacity_envelope,
        capacity_authority=capacity,
        declared_plan=raw("declared_budget_plan", semantic=_sha("declared-plan")),
        registry_sha256=registry_sha256,
        budget_inventory_sha256=capacity.budget_inventory_sha256,
        activation_sha256=final.activation_sha256,
        budget_policy_sha256=_sha("policy"),
        budget_load_binding_sha256s=(),
        capacity_envelope_sha256=capacity_envelope.semantic_sha256,
        capacity_authority_sha256=capacity.sha256,
        declared_plan_sha256=_sha("declared-plan"),
        authority_protocol_sha256=(BUDGET_MATERIALIZATION_AUTHORITY_PROTOCOL_SHA256),
    )
    return (
        e1,
        e2,
        e2_stage_completion,
        pilot,
        pilot_completion,
        auxiliary,
        auxiliary_completion,
        final,
        family_completion,
        stage_family,
        aggregate,
        materialization,
    )


def test_specialized_tagged_authority_codecs_are_exact(tmp_path: Path) -> None:
    (
        e1,
        e2,
        e2_stage_completion,
        pilot,
        pilot_completion,
        auxiliary,
        auxiliary_completion,
        final,
        family_completion,
        stage_family,
        aggregate,
        materialization,
    ) = _specialized_bindings(tmp_path)
    aggregate_with_auxiliary = replace(
        aggregate,
        experiment="E5",
        auxiliary_completion_authority=auxiliary_completion,
    )
    cases = (
        (
            e1,
            e1_activation_authority_binding_to_dict,
            e1_activation_authority_binding_from_dict,
        ),
        (
            e2,
            e2_activation_authority_binding_to_dict,
            e2_activation_authority_binding_from_dict,
        ),
        (
            e2_stage_completion,
            e2_stage_completion_authority_binding_to_dict,
            e2_stage_completion_authority_binding_from_dict,
        ),
        (
            pilot,
            confirmation_pilot_activation_authority_binding_to_dict,
            confirmation_pilot_activation_authority_binding_from_dict,
        ),
        (
            pilot_completion,
            family_pilot_completion_authority_binding_to_dict,
            family_pilot_completion_authority_binding_from_dict,
        ),
        (
            auxiliary,
            confirmation_auxiliary_activation_authority_binding_to_dict,
            confirmation_auxiliary_activation_authority_binding_from_dict,
        ),
        (
            auxiliary_completion,
            confirmation_auxiliary_completion_authority_binding_to_dict,
            confirmation_auxiliary_completion_authority_binding_from_dict,
        ),
        (
            final,
            confirmation_final_activation_authority_binding_to_dict,
            confirmation_final_activation_authority_binding_from_dict,
        ),
        (
            family_completion,
            confirmation_family_completion_authority_binding_to_dict,
            confirmation_family_completion_authority_binding_from_dict,
        ),
        (
            stage_family,
            confirmation_stage_family_authority_binding_to_dict,
            confirmation_stage_family_authority_binding_from_dict,
        ),
        (
            aggregate,
            confirmation_stage_aggregate_authority_binding_to_dict,
            confirmation_stage_aggregate_authority_binding_from_dict,
        ),
        (
            aggregate_with_auxiliary,
            confirmation_stage_aggregate_authority_binding_to_dict,
            confirmation_stage_aggregate_authority_binding_from_dict,
        ),
        (
            materialization,
            budget_materialization_authority_binding_to_dict,
            budget_materialization_authority_binding_from_dict,
        ),
    )
    for artifact, encode, decode in cases:
        assert decode(encode(artifact)) == artifact

    with pytest.raises(ValueError, match="SHA-sorted and unique"):
        replace(
            aggregate,
            families=(stage_family, stage_family),
            family_sha256s=(stage_family.family_sha256,) * 2,
        )


def test_stage_aggregate_completion_requires_full_stage_disposition_coverage(
    tmp_path: Path,
) -> None:
    *_, aggregate, _materialization = _specialized_bindings(tmp_path)
    stage_ids = (_sha("stage-a"), _sha("stage-b"))
    registry = SimpleNamespace(
        cells_for=lambda _stage: tuple(
            SimpleNamespace(cell_id=cell_id, runnable=True) for cell_id in stage_ids
        )
    )
    authority = SimpleNamespace(
        raw_activation_authority=aggregate,
        registry=registry,
        direct_dependency_receipt=None,
        activation_artifact=None,
        family_activations=(SimpleNamespace(sha256=_sha("aggregate-family")),),
        family_power_reductions=(),
    )
    authority._replay_raw_activation_authority = lambda: None
    authority._family_activation_rows = lambda **_kwargs: (
        (
            (
                stage_ids[0],
                DispositionStatus.ACTIVATED,
                "family_power_selected_final_prefix",
            ),
        ),
        "final_prefix",
    )
    with pytest.raises(ValueError, match="incomplete disposition coverage"):
        CompletedCellAuthority._activation_contract(
            authority,
            stage="E5",
            runtime_sha256=_sha("runtime"),
            split_sha256=_sha("split"),
        )


def test_stage_aggregate_completion_merges_exact_auxiliary_dispositions(
    tmp_path: Path,
) -> None:
    (
        *_prefix,
        auxiliary_binding,
        auxiliary_completion,
        _final,
        _family_completion,
        _stage_family,
        aggregate,
        _materialization,
    ) = _specialized_bindings(tmp_path)
    family_cell_id = _sha("aggregate-family-cell")
    auxiliary_cell_id = _sha("aggregate-auxiliary-cell")
    dependency_receipt_sha256 = _sha("aggregate-direct-dependency")
    auxiliary_artifact = ReducerActivationArtifact(
        schema_version=1,
        plan=StageActivationPlan(
            registry_sha256=_sha("aggregate-registry"),
            experiment="E5",
            dependency_receipt_sha256=dependency_receipt_sha256,
            runtime_sha256=aggregate.runtime.canonical_sha256,
            split_sha256=aggregate.split.canonical_sha256,
            source_selection_sha256=_sha("aggregate-auxiliary-source"),
            activation_round="confirmation_auxiliary_registry_v1",
            status="AVAILABLE",
            activated_cell_ids=(auxiliary_cell_id,),
            not_applicable_cell_ids=(),
            blocked_cell_ids=(),
            deferred_cell_ids=(),
            reason_code="confirmation_auxiliary_registry_activation",
        ),
        reducer_protocol_sha256=CONFIRMATION_AUXILIARY_ACTIVATION_PROTOCOL_SHA256,
        dispositions=(
            CellDisposition(
                cell_id=auxiliary_cell_id,
                status=DispositionStatus.ACTIVATED,
                reason_code="confirmation_auxiliary_registry_cell",
            ),
        ),
    )
    auxiliary_binding = replace(
        auxiliary_binding,
        activation_sha256=auxiliary_artifact.sha256,
    )
    auxiliary_completion = replace(
        auxiliary_completion,
        activation=auxiliary_binding,
    )
    encoded = tuple(
        sorted(
            (
                {
                    "cell_id": family_cell_id,
                    "status": DispositionStatus.ACTIVATED.value,
                    "reason_code": "family_power_selected_final_prefix",
                },
                {
                    "cell_id": auxiliary_cell_id,
                    "status": DispositionStatus.ACTIVATED.value,
                    "reason_code": "confirmation_auxiliary_registry_cell",
                },
            ),
            key=lambda value: value["cell_id"],
        )
    )
    aggregate = replace(
        aggregate,
        experiment="E5",
        auxiliary_completion_authority=auxiliary_completion,
        activated_cell_ids=tuple(sorted((family_cell_id, auxiliary_cell_id))),
        dispositions_sha256=content_sha256(encoded),
    )
    registry = SimpleNamespace(
        cells_for=lambda _stage: tuple(
            SimpleNamespace(cell_id=cell_id, runnable=True)
            for cell_id in (family_cell_id, auxiliary_cell_id)
        )
    )
    authority = SimpleNamespace(
        raw_activation_authority=aggregate,
        registry=registry,
        direct_dependency_receipt=SimpleNamespace(sha256=dependency_receipt_sha256),
        activation_artifact=auxiliary_artifact,
        family_activations=(SimpleNamespace(sha256=_sha("aggregate-family")),),
        family_power_reductions=(),
    )
    authority._replay_raw_activation_authority = lambda: None
    authority._family_activation_rows = lambda **_kwargs: (
        (
            (
                family_cell_id,
                DispositionStatus.ACTIVATED,
                "family_power_selected_final_prefix",
            ),
        ),
        "final_prefix",
    )
    activated, dispositions, activation_contract = (
        CompletedCellAuthority._activation_contract(
            authority,
            stage="E5",
            runtime_sha256=aggregate.runtime.canonical_sha256,
            split_sha256=aggregate.split.canonical_sha256,
        )
    )
    assert activated == tuple(sorted((family_cell_id, auxiliary_cell_id)))
    assert set(dispositions) == {family_cell_id, auxiliary_cell_id}
    assert activation_contract["dispositions_sha256"] == content_sha256(encoded)


def test_auxiliary_activation_uses_only_the_release_owned_registry_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import planning

    auxiliary_cells = tuple(
        SimpleNamespace(
            cell_id=_sha(f"auxiliary-{index}"),
            runnable=True,
            resources=SimpleNamespace(workload_class=object()),
            status=SimpleNamespace(),
            reason_code="awaiting_registered_measurement",
        )
        for index in range(2)
    )
    monkeypatch.setattr(
        planning,
        "derive_confirmation_stage_partition",
        lambda *_args, **_kwargs: ((object(),), auxiliary_cells),
    )
    registry = SimpleNamespace(
        sha256=_sha("auxiliary-registry"),
        validate_receipts=lambda _receipts: None,
    )
    receipts = tuple(
        SimpleNamespace(experiment=experiment, sha256=_sha(f"receipt-{experiment}"))
        for experiment in INDUSTRIAL_EXPERIMENT_ORDER[:7]
    )
    artifact = materialize_confirmation_auxiliary_activation(
        registry,
        experiment="E5",
        dependency_receipts=receipts,
        runtime_sha256=_sha("auxiliary-runtime"),
        split_sha256=_sha("auxiliary-split"),
        trace_sha256=_sha("auxiliary-trace"),
        sampling_sha256=_sha("auxiliary-sampling"),
        hardware_envelope_sha256=_sha("auxiliary-hardware"),
    )
    assert artifact.reducer_protocol_sha256 == (
        CONFIRMATION_AUXILIARY_ACTIVATION_PROTOCOL_SHA256
    )
    expected = tuple(sorted(cell.cell_id for cell in auxiliary_cells))
    assert artifact.plan.activated_cell_ids == expected
    assert tuple(row.cell_id for row in artifact.dispositions) == expected


def test_family_completion_matches_each_power_reduction_to_its_own_pilots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import completion_authority

    registry_sha256 = _sha("family-registry")
    runtime_sha256 = _sha("family-runtime")
    split_sha256 = _sha("family-split")
    registry = SimpleNamespace(sha256=registry_sha256)
    finals: dict[str, object] = {}
    activations: list[object] = []
    reductions: list[object] = []
    authorities: list[object] = []
    for label in ("a", "b"):
        family_sha256 = _sha(f"family-{label}")
        pilot_id = _sha(f"pilot-{label}")
        final_id = _sha(f"final-{label}")
        terminal_sha256 = _sha(f"terminal-{label}")
        family = SimpleNamespace(
            sha256=family_sha256,
            registry_sha256=registry_sha256,
            experiment="E4",
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
        )
        pilot = SimpleNamespace(
            sha256=_sha(f"pilot-activation-{label}"),
            family=family,
            activation_round="excluded_pilots",
            activated_cell_ids=(pilot_id,),
            dispositions=(
                SimpleNamespace(
                    cell_id=pilot_id,
                    status=DispositionStatus.ACTIVATED,
                    reason_code="family_excluded_pilot",
                ),
            ),
        )
        final = SimpleNamespace(
            sha256=_sha(f"final-activation-{label}"),
            family=family,
            activation_round="final_prefix",
            activated_cell_ids=(final_id,),
            dispositions=(
                SimpleNamespace(
                    cell_id=final_id,
                    status=DispositionStatus.ACTIVATED,
                    reason_code="family_power_selected_final_prefix",
                ),
            ),
        )
        finals[family_sha256] = final
        activations.extend((pilot, final))
        reductions.append(
            SimpleNamespace(
                family=family,
                terminal_receipt_sha256s=(terminal_sha256,),
            )
        )
        result = SimpleNamespace(
            completed_cell_ids=(pilot_id,),
            terminal_bindings=(
                SimpleNamespace(terminal_receipt_sha256=terminal_sha256),
            ),
        )
        authorities.append(SimpleNamespace(revalidate=lambda result=result: result))

    monkeypatch.setattr(
        completion_authority,
        "verify_confirmation_pilot_activation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        completion_authority,
        "materialize_confirmation_prefix",
        lambda _registry, *, family, **_kwargs: finals[family.sha256],
    )
    authority = SimpleNamespace(
        registry=registry,
        family_activations=tuple(activations),
        family_power_reductions=tuple(reductions),
        prior_family_authorities=tuple(authorities),
    )
    rows, activation_round = CompletedCellAuthority._family_activation_rows(
        authority,
        stage="E4",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    assert activation_round == "final_prefix"
    assert {row[0] for row in rows} == {_sha("final-a"), _sha("final-b")}

    reductions[1].terminal_receipt_sha256s = (_sha("terminal-a"),)
    with pytest.raises(ValueError, match="swapped prior terminal"):
        CompletedCellAuthority._family_activation_rows(
            authority,
            stage="E4",
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
        )


def test_raw_activation_replay_rejects_a_swapped_direct_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import budget_authority

    raw_receipt = SimpleNamespace(sha256=_sha("raw-receipt"))
    raw_dependency = SimpleNamespace(sha256=_sha("raw-dependency"))
    registry = object()
    replay = SimpleNamespace(
        registry=registry,
        activation_artifact=None,
        family_activations=(),
        family_power_reductions=(),
        prior_family_authorities=(),
        dependency_records=(
            SimpleNamespace(receipt=raw_receipt, authority=raw_dependency),
        ),
    )
    monkeypatch.setattr(
        budget_authority,
        "replay_budget_activation_authority",
        lambda _binding: replay,
    )
    authority = SimpleNamespace(
        raw_activation_authority=object(),
        registry=registry,
        activation_artifact=None,
        family_activations=(),
        family_power_reductions=(),
        prior_family_authorities=(),
        direct_dependency_receipt=SimpleNamespace(sha256=_sha("wrong-receipt")),
        dependency_authority=raw_dependency,
    )
    with pytest.raises(ValueError, match="changed direct dependency lineage"):
        CompletedCellAuthority._replay_raw_activation_authority(authority)


def test_later_e2_manifest_without_schema_v4_round_completion_is_named_block(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "e2-stage-one-authority.json").resolve()
    value = {
        "schema_version": 1,
        "kind": "industrial_e2_activation_authority_manifest",
        "stage_index": 1,
    }
    body = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(body)
    canonical = hashlib.sha256(body.rstrip(b"\n")).hexdigest()
    Path(f"{path}.sha256").write_text(f"{canonical}\n", encoding="ascii")

    with pytest.raises(BudgetMaterializationBlockedError) as error:
        bind_budget_activation_authority(path)
    assert error.value.reason_code == E2_STAGE_COMPLETION_AUTHORITY_MISSING_REASON


def test_per_family_authority_cannot_mint_a_stage_completion_without_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import budget_authority

    receipt_source = object()
    receipt = SimpleNamespace(
        experiment="E5",
        runtime_sha256=_sha("family-stage-runtime"),
        split_sha256=_sha("family-stage-split"),
    )
    registry = object()
    replay = SimpleNamespace(
        binding=object(),
        registry=registry,
        dependency_records=(),
        dependency_receipts=(),
        experiment=receipt.experiment,
        runtime_sha256=receipt.runtime_sha256,
        split_sha256=receipt.split_sha256,
        family_activations=(object(),),
    )
    monkeypatch.setattr(
        budget_authority,
        "bind_budget_raw_json",
        lambda *_args, **_kwargs: receipt_source,
    )
    monkeypatch.setattr(
        budget_authority,
        "load_budget_raw_json",
        lambda _source: object(),
    )
    monkeypatch.setattr(
        budget_authority,
        "_receipt_from_value",
        lambda _value: receipt,
    )
    monkeypatch.setattr(
        budget_authority,
        "_bind_stage_activation_authority",
        lambda *_args, **_kwargs: replay,
    )

    with pytest.raises(BudgetMaterializationBlockedError) as error:
        budget_authority._bind_dependency_completion(
            {
                "receipt_artifact": "/formal/receipt.json",
                "activation_manifest": "/formal/family.json",
            },
            expected_receipt_source=receipt_source,
            expected_registry=registry,
            earlier_records=None,
            manifest_stack=(),
        )
    assert (
        error.value.reason_code
        == budget_authority.DEPENDENCY_COMPLETION_FAMILY_STAGE_AGGREGATION_MISSING_REASON
    )
