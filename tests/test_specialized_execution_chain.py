from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_industrial_analysis import _gpu_inventory

from lightcone_spec.experiments.budget_authority import bind_budget_raw_json
from lightcone_spec.experiments.industrial_analysis import (
    BoundArtifact,
    IndustrialCellEvidence,
    RawE3aSelectionEvidenceManifest,
    _AliasRunIdentity,
    _CellExecutionIdentity,
    _replay_cell_execution_identity,
    _validate_run_row,
    raw_e3a_selection_manifest_to_dict,
)
from lightcone_spec.experiments.registry import (
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.orchestration.execution_bundle import (
    BoundJsonSource,
    _compare_budget_raw_binding,
)


def _write_json(path: Path, value: object) -> BoundArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    path.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    Path(f"{path}.sha256").write_text(content_sha256(value) + "\n", encoding="ascii")
    return BoundArtifact(path=path.resolve(), sha256=digest)


def _cell_contract(cell_id: str, *, execution_plan: str, execution_split: str):
    digest = content_sha256
    return {
        "cell_id": cell_id,
        "request_ids": ["request-0"],
        "expected_request_rows": 1,
        "expected_round_rows": 0,
        "expected_update_rows": 0,
        "expected_performance_rows": 1,
        "request_ids_sha256": digest(["request-0"]),
        "corpus_sha256": digest("corpus"),
        "arrival_trace_sha256": digest("arrival"),
        "sampling_profile_sha256": digest("sampling"),
        "model_lock_sha256": digest("models"),
        "patched_sglang_tree": "1" * 40,
        "workload_contract": "industrial_target_only",
        "rank_config_sha256s": [digest("rank-config")],
        "physical_assignment": {},
        "physical_binding_sha256": digest("physical"),
        "topology_receipt_sha256": digest("topology"),
        "experiment_budget_sha256": digest("budget"),
        "experiment_budget": {},
        "execution_plan_sha256": execution_plan,
        "execution_split_sha256": execution_split,
    }


def _measured_row(cell, terminal_sha256: str):
    digest = content_sha256
    return {
        "cell_id": cell.cell_id,
        "evidence_root": cell.resources.evidence_root,
        "run_id": "formal-run",
        "rank": 0,
        "evidence_sha256": digest("evidence"),
        "terminal_receipt_sha256": terminal_sha256,
        "physical_gpu_uuid": "GPU-bundle-0",
        "physical_binding_sha256": digest("physical"),
        "experiment_budget_sha256": digest("budget"),
        "budget_observation_status": "OBSERVED",
        "budget_observation_reason_code": None,
        "budget_observation_path": "/tmp/observation.json",
        "budget_observation_sha256": digest("observation"),
        "preflight_attestation_path": None,
        "preflight_attestation_sha256": None,
        "status": "MEASURED",
    }


def test_schema_v4_replay_separates_lineage_and_cell_execution_identity(
    tmp_path: Path,
) -> None:
    registry = build_industrial_registry(
        gpu_uuids=("logical-rank-slot-0", "logical-rank-slot-1"),
        cache_root=str(tmp_path / "cache"),
        evidence_root=str(tmp_path / "evidence"),
    )
    cell = next(
        row
        for row in registry.cells_for("E3a")
        if row.runnable and row.identity.method == "target_only"
    )
    lineage_runtime = content_sha256("activation-runtime")
    execution_plan = content_sha256("cell-execution-plan")
    execution_split = content_sha256("cell-execution-split")
    terminal = _write_json(
        tmp_path / "terminal.json",
        {"schema_version": 3, "run_id": "formal-run", "rank": 0},
    )
    contract = _cell_contract(
        cell.cell_id,
        execution_plan=execution_plan,
        execution_split=execution_split,
    )
    split = {
        "schema_version": 1,
        "kind": "industrial_locked_split",
        "registry_sha256": registry.sha256,
        "experiment": "E3a",
        "cells": [contract],
    }
    lineage_split = content_sha256(split)
    inventory = _gpu_inventory()
    completion = _write_json(
        tmp_path / "completed.json",
        {
            "schema_version": 4,
            "kind": "industrial_completed_cells",
            "registry_sha256": registry.sha256,
            "experiment": "E3a",
            "runtime_sha256": lineage_runtime,
            "split_sha256": lineage_split,
            "split_contract": split,
            "activation_binding": {
                "schema_version": 1,
                "kind": "industrial_stage_activation_binding",
                "stage_activation_sha256": content_sha256("activation"),
                "family_activation_sha256s": [],
                "family_power_reduction_sha256s": [],
                "direct_dependency_receipt_sha256": content_sha256("preflight"),
                "activation_round": "e3a_reference_sweep",
                "dispositions_sha256": content_sha256("dispositions"),
            },
            "inventory_sha256": inventory.sha256,
            "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
            "rows": [_measured_row(cell, terminal.sha256)],
        },
    )
    evidence = IndustrialCellEvidence(
        cell_id=cell.cell_id,
        terminal_receipts=(terminal,),
        hardware_receipt=terminal,
        budget_observation=terminal,
        completion_contract=completion,
    )

    identity = _replay_cell_execution_identity(
        evidence,
        registry=registry,
        family=_AliasRunIdentity(
            experiment="E3a",
            runtime_sha256=lineage_runtime,
            split_sha256=lineage_split,
        ),
        cell=cell,
        inventory=inventory,
    )

    assert identity.runtime_sha256 == execution_plan
    assert identity.split_sha256 == execution_split
    assert identity.runtime_sha256 != lineage_runtime
    assert identity.split_sha256 != lineage_split

    run_row = {
        "manifest_sha256": registry.sha256,
        "config_sha256": cell.cell_id,
        "industrial_cell_id": cell.cell_id,
        "runtime_sha256": execution_plan,
        "split_sha256": execution_split,
        "method": "target_only",
        "model_pair": cell.identity.model,
        "repetition_block": cell.identity.block,
        "patched_sglang_tree": "1" * 40,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "world_size": 1,
        "rank": 0,
        "status": "complete",
        "workload_contract": "industrial_target_only",
        "expected_request_rows": 1,
        "expected_round_rows": 0,
        "expected_update_rows": 0,
        "expected_performance_rows": 1,
        "session_plan_sha256": None,
        "session_open_receipt_sha256": None,
        "reset_receipt_sha256": None,
        "session_epoch": None,
        **{
            name: content_sha256(name)
            for name in (
                "rank_config_sha256",
                "corpus_sha256",
                "arrival_trace_sha256",
                "request_ids_sha256",
                "sampling_profile_sha256",
                "model_lock_sha256",
                "run_nonce_sha256",
                "topology_sha256",
                "experiment_budget_sha256",
            )
        },
    }
    _validate_run_row(
        run_row,
        registry=registry,
        family=_CellExecutionIdentity(
            experiment="E3a",
            runtime_sha256=execution_plan,
            split_sha256=execution_split,
        ),
        cell=cell,
        rank=0,
    )
    with pytest.raises(ValueError, match="registry/runtime identity"):
        _validate_run_row(
            run_row,
            registry=registry,
            family=_CellExecutionIdentity(
                experiment="E3a",
                runtime_sha256=lineage_runtime,
                split_sha256=lineage_split,
            ),
            cell=cell,
            rank=0,
        )


def test_formal_raw_manifest_rejects_missing_completion_contract(
    tmp_path: Path,
) -> None:
    artifact = _write_json(tmp_path / "artifact.json", {"fixture": True})
    manifest = RawE3aSelectionEvidenceManifest(
        schema_version=2,
        cells=(
            IndustrialCellEvidence(
                cell_id=content_sha256("cell"),
                terminal_receipts=(artifact,),
                hardware_receipt=artifact,
                budget_observation=artifact,
            ),
        ),
    )
    with pytest.raises(ValueError, match="schema-v4 completion contract"):
        raw_e3a_selection_manifest_to_dict(manifest)

    with pytest.raises(ValueError, match="schema 2"):
        RawE3aSelectionEvidenceManifest(schema_version=1, cells=manifest.cells)


def test_specialized_manifest_uses_activation_output_semantic_domain(
    tmp_path: Path,
) -> None:
    value = {
        "schema_version": 1,
        "kind": "industrial_e1_activation_authority_manifest",
    }
    path = tmp_path / "e1-activation.json"
    _write_json(path, value)
    binding = bind_budget_raw_json(path, role="e1_activation_authority_manifest")
    source = BoundJsonSource.bind(
        path,
        semantic_sha256=content_sha256("reducer-owned-activation-output"),
    )

    _compare_budget_raw_binding("activation manifest", source, binding)

    non_manifest = bind_budget_raw_json(path, role="activation_runtime")
    with pytest.raises(ValueError, match="semantic identity differs"):
        _compare_budget_raw_binding("activation runtime", source, non_manifest)
