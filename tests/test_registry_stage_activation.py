from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from lightcone_spec.cli.main import _industrial_completion_activation_contract
from lightcone_spec.experiments.registry import (
    INDUSTRIAL_EXPERIMENT_ORDER,
    SEALED_E2_RECIPE_SENTINEL,
    CellStatus,
    ExperimentReceipt,
    build_industrial_registry,
    content_sha256,
    scientific_role_for_cell,
)
from lightcone_spec.experiments.stage_activation import (
    REGISTRY_STAGE_ACTIVATION_PROTOCOL_SHA256,
    REGISTRY_STAGE_RELEASE_CAPABILITY_SHA256,
    RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE,
    RELEASE_DOWNLOAD_ASSIGNMENT_CONTRACT_UNAVAILABLE,
    RegistryStageDispositionStatus,
    materialize_registry_stage_activation,
    registry_stage_activation_from_dict,
    registry_stage_activation_to_dict,
    release_dispatch_rejection_reason,
    release_execution_capability_rejection_reason,
    verify_registry_stage_activation,
)


def _sha(label: str) -> str:
    return content_sha256({"registry-stage-test": label})


def _receipt_prefix(registry, experiment: str) -> tuple[ExperimentReceipt, ...]:
    receipts: list[ExperimentReceipt] = []
    target_index = INDUSTRIAL_EXPERIMENT_ORDER.index(experiment)
    for name in INDUSTRIAL_EXPERIMENT_ORDER[:target_index]:
        definition = registry.definition(name)
        receipt = registry.make_receipt(
            name,
            {
                output: _sha(f"{name}-output-{output}")
                for output in definition.locked_outputs
            },
            runtime_sha256=_sha(f"{name}-runtime"),
            split_sha256=_sha(f"{name}-split"),
            completed_cells_sha256=_sha(f"{name}-completed"),
            dependencies=tuple(receipts),
        )
        receipts.append(receipt)
    return tuple(receipts)


def test_registry_stage_reducer_uses_canonical_genesis_and_release_policy() -> None:
    registry = build_industrial_registry()
    artifact = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=_sha("preflight-runtime"),
        split_sha256=_sha("preflight-split"),
    )

    assert artifact.status == "AVAILABLE"
    assert artifact.reducer_protocol_sha256 == (
        REGISTRY_STAGE_ACTIVATION_PROTOCOL_SHA256
    )
    assert artifact.release_capability_sha256 == (
        REGISTRY_STAGE_RELEASE_CAPABILITY_SHA256
    )
    assert artifact.genesis_authority is not None
    assert artifact.genesis_authority.registry_sha256 == registry.sha256
    assert len(artifact.dispositions) == len(registry.cells_for("preflight")) == 10
    activated = {
        row.cell_id
        for row in artifact.dispositions
        if row.status is RegistryStageDispositionStatus.ACTIVATED
    }
    expected = {
        cell.cell_id
        for cell in registry.cells_for("preflight")
        if release_dispatch_rejection_reason(cell) is None
    }
    assert activated == expected
    assert activated == {
        cell.cell_id
        for cell in registry.cells_for("preflight")
        if cell.identity.task == "simultaneous_single_gpu_interference"
    }
    assert {
        row.reason_code
        for row in artifact.dispositions
        if row.status is RegistryStageDispositionStatus.BLOCKED
    } == {
        RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE,
        "release_preflight_method_unsupported",
    }
    calibration = tuple(
        cell
        for cell in registry.cells_for("preflight")
        if cell.identity.task == "simultaneous_single_gpu_interference"
    )
    assert len(calibration) == 8
    assert {
        (cell.identity.block, cell.identity.variant, cell.resources.gpu_uuids)
        for cell in calibration
    } == {
        (repetition, f"{mode}_slot_{slot}", (registry.gpu_uuids[slot],))
        for repetition in range(2)
        for mode in ("isolated", "concurrent")
        for slot in range(2)
    }

    wire = registry_stage_activation_to_dict(artifact)
    assert registry_stage_activation_from_dict(wire) == artifact
    verify_registry_stage_activation(registry, artifact)

    forged_genesis = deepcopy(wire)
    forged_genesis["genesis_authority"]["registry_sha256"] = "0" * 64
    forged_genesis["artifact_sha256"] = _sha("caller-rehashed")
    with pytest.raises(ValueError, match="genesis|registry-stage"):
        registry_stage_activation_from_dict(forged_genesis)

    boolean_schema = deepcopy(wire)
    boolean_schema["schema_version"] = True
    with pytest.raises(ValueError, match="schema"):
        registry_stage_activation_from_dict(boolean_schema)


def test_generic_reducer_requires_complete_prefix_and_exact_serving_semantics() -> None:
    registry = build_industrial_registry()
    with pytest.raises(ValueError, match="complete dependency receipt prefix"):
        materialize_registry_stage_activation(
            registry,
            experiment="E3a",
            dependency_receipts=(),
            runtime_sha256=_sha("e3a-runtime"),
            split_sha256=_sha("e3a-split"),
        )

    e3a = materialize_registry_stage_activation(
        registry,
        experiment="E3a",
        dependency_receipts=_receipt_prefix(registry, "E3a"),
        runtime_sha256=_sha("e3a-runtime"),
        split_sha256=_sha("e3a-split"),
    )
    assert e3a.status == "AVAILABLE"
    cells = {cell.cell_id: cell for cell in registry.cells_for("E3a")}
    assert {cells[cell_id].identity.method for cell_id in e3a.activated_cell_ids} == {
        "target_only",
        "static",
    }
    assert all(
        row.status is RegistryStageDispositionStatus.ACTIVATED
        for row in e3a.dispositions
        if cells[row.cell_id].identity.method == "static"
    )

    e4 = materialize_registry_stage_activation(
        registry,
        experiment="E4",
        dependency_receipts=_receipt_prefix(registry, "E4"),
        runtime_sha256=_sha("e4-runtime"),
        split_sha256=_sha("e4-split"),
    )
    assert e4.status == "BLOCKED"
    assert e4.activated_cell_ids == ()
    assert {
        row.reason_code
        for row in e4.dispositions
        if row.status is RegistryStageDispositionStatus.BLOCKED
    } <= {
        "release_serving_contract_unresolved",
        "release_topology_executor_unsupported",
        "sealed_e2_recipe_receipt_required",
        "tts_official_recipe_unavailable",
    }
    frozen_tts = next(
        cell
        for cell in registry.cells_for("E4")
        if cell.identity.method == "tts"
        and cell.identity.backend == "DFLASH"
        and cell.identity.topology == "tp1_dp1"
    )
    assert release_execution_capability_rejection_reason(frozen_tts) == (
        "tts_official_recipe_unavailable"
    )
    assert (
        next(row for row in e4.dispositions if row.cell_id == frozen_tts.cell_id).status
        is RegistryStageDispositionStatus.BLOCKED
    )
    lc_candidate = next(
        cell
        for cell in registry.cells_for("E1")
        if scientific_role_for_cell(registry, cell) == "lc_candidate" and cell.runnable
    )
    assert release_execution_capability_rejection_reason(lc_candidate) is None
    assert release_dispatch_rejection_reason(lc_candidate) == (
        "release_serving_contract_unresolved"
    )
    with pytest.raises(ValueError, match="cannot seal without an AVAILABLE"):
        _industrial_completion_activation_contract(
            registry,
            experiment="E4",
            runtime_sha256=e4.runtime_sha256,
            split_sha256=e4.split_sha256,
            direct_dependency_receipt_sha256=(e4.dependency_receipts[-1].sha256),
            activation_artifact=e4,
            family_activations=(),
            family_power_reductions=(),
            require_stage_sealable=True,
        )

    e6 = materialize_registry_stage_activation(
        registry,
        experiment="E6",
        dependency_receipts=_receipt_prefix(registry, "E6"),
        runtime_sha256=_sha("e6-runtime"),
        split_sha256=_sha("e6-split"),
    )
    assert e6.status == "BLOCKED"
    assert e6.activated_cell_ids == ()
    assert {
        row.reason_code
        for row in e6.dispositions
        if row.status is RegistryStageDispositionStatus.BLOCKED
    } == {
        RELEASE_DOWNLOAD_ASSIGNMENT_CONTRACT_UNAVAILABLE,
        "native_nextn_preflight_required",
        "sealed_e2_recipe_receipt_required",
        "tts_official_recipe_unavailable",
    }

    for experiment in ("E1", "E2", "E3b", "E5"):
        with pytest.raises(ValueError, match="bespoke activation reducer"):
            materialize_registry_stage_activation(
                registry,
                experiment=experiment,
                dependency_receipts=(),
                runtime_sha256=_sha(f"{experiment}-runtime"),
                split_sha256=_sha(f"{experiment}-split"),
            )


def test_scheduler_and_completion_replay_reject_edited_generic_activation() -> None:
    registry = build_industrial_registry()
    runtime_sha256 = _sha("completion-runtime")
    split_sha256 = _sha("completion-split")
    artifact = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    blocked_index = next(
        index
        for index, row in enumerate(artifact.dispositions)
        if row.status is RegistryStageDispositionStatus.BLOCKED
    )
    edited_rows = list(artifact.dispositions)
    edited_rows[blocked_index] = replace(
        edited_rows[blocked_index], reason_code="caller_edited_block_reason"
    )
    edited = replace(artifact, dispositions=tuple(edited_rows))
    with pytest.raises(ValueError, match="exact reducer-generated"):
        verify_registry_stage_activation(registry, edited)

    activated, dispositions, binding = _industrial_completion_activation_contract(
        registry,
        experiment="preflight",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        direct_dependency_receipt_sha256=None,
        activation_artifact=artifact,
        family_activations=(),
        family_power_reductions=(),
    )
    assert activated == artifact.activated_cell_ids
    assert set(dispositions) == {
        cell.cell_id for cell in registry.cells_for("preflight")
    }
    assert binding["stage_activation_sha256"] == artifact.sha256
    with pytest.raises(ValueError, match="exact reducer-generated"):
        _industrial_completion_activation_contract(
            registry,
            experiment="preflight",
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            direct_dependency_receipt_sha256=None,
            activation_artifact=edited,
            family_activations=(),
            family_power_reductions=(),
        )


def test_adaptive_recipe_identity_cannot_be_unblocked_by_status_rewrite() -> None:
    registry = build_industrial_registry()
    frozen_tts = next(
        cell
        for cell in registry.cells_for("E4")
        if scientific_role_for_cell(registry, cell) == "tts"
        and cell.identity.topology == "tp1_dp1"
    )
    sealed_template = next(
        cell
        for cell in registry.cells_for("E4")
        if cell.identity.optimizer == SEALED_E2_RECIPE_SENTINEL
        and cell.identity.topology == "tp1_dp1"
    )

    for cell, expected in (
        (frozen_tts, "tts_official_recipe_unavailable"),
        (sealed_template, "sealed_e2_recipe_receipt_required"),
    ):
        caller_unblocked = replace(
            cell,
            status=CellStatus.UNMEASURED,
            reason_code="awaiting_registered_measurement",
            reason="Caller removed the registry-owned blocker.",
        )
        assert release_execution_capability_rejection_reason(caller_unblocked) == (
            expected
        )
        assert release_dispatch_rejection_reason(caller_unblocked) == expected
