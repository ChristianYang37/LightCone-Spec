from __future__ import annotations

import hashlib

import pytest

from lightcone_spec.experiments.formal_dispatch import (
    VerifiedFormalPreflightDispatch,
    _preflight_execution_bindings,
)
from lightcone_spec.experiments.formal_protocol import ProtocolLock
from lightcone_spec.experiments.gpu_pool import (
    GpuAssignment,
    GpuDispatchPlan,
    GpuDispatchWave,
    registry_pool_work_item,
)
from lightcone_spec.experiments.registry import (
    build_industrial_registry,
    build_legacy_industrial_registry,
)
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    materialize_preflight,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _protocol_lock() -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="lightcone-formal-protocol-v2",
        code_git_head="1" * 40,
        code_git_tree="2" * 40,
        patch_manifest_sha256=_sha("patch"),
        registry_sha256=build_industrial_registry().sha256,
        english_protocol_sha256=_sha("english"),
        chinese_protocol_sha256=_sha("chinese"),
        tts_calibration_authority_sha256=_sha("tts"),
        chronobelief_authority_sha256=_sha("chronobelief"),
        e1_recipe_anchor_authority_sha256=_sha("e1"),
        e2_recipe_grid_authority_sha256=_sha("e2"),
        formal_runtime_authority_manifest_sha256=_sha("formal-runtime"),
        offline_release_trust_root_sha256=_sha("release-root"),
        prepared_model_content_authorization_sha256=_sha("prepared-model"),
        formal_workload_e3a_authorization_sha256=_sha("e3a-workload"),
        formal_workload_e0_authorization_sha256=_sha("e0-workload"),
        burstgpt_shape_authorization_sha256=_sha("burstgpt-shape"),
        native_runtime_qualification_protocol_sha256=_sha("native-protocol"),
        native_runtime_qualification_runner_sha256=_sha("native-runner"),
        native_runtime_qualification_test_set_sha256=_sha("native-tests"),
        compile_qualification_protocol_sha256=_sha("compile-protocol"),
        compile_qualification_runner_sha256=_sha("compile-runner"),
        compile_qualification_test_set_sha256=_sha("compile-tests"),
        exactness_qualification_protocol_sha256=_sha("exactness-protocol"),
        exactness_qualification_runner_sha256=_sha("exactness-runner"),
        exactness_qualification_test_set_sha256=_sha("exactness-tests"),
    )


def _dispatch_plan(*, omit_cell_id: str | None = None) -> GpuDispatchPlan:
    registry = build_industrial_registry()
    assignments = []
    budget_rows = []
    next_port = 24_000
    for cell in sorted(registry.cells_for("preflight"), key=lambda row: row.cell_id):
        if cell.cell_id == omit_cell_id:
            continue
        item = registry_pool_work_item(cell, estimated_duration_seconds=1.0)
        gpu_uuids = (
            ("GPU-PHYSICAL-0", "GPU-PHYSICAL-1")
            if item.claim.gang_shape.gpu_count == 2
            else (
                "GPU-PHYSICAL-1"
                if str(cell.identity.variant).endswith("slot_1")
                else "GPU-PHYSICAL-0",
            )
        )
        tp = item.claim.gang_shape.tensor_parallel_size
        rank_groups = tuple(
            gpu_uuids[index : index + tp] for index in range(0, len(gpu_uuids), tp)
        )
        ports = tuple(range(next_port, next_port + item.claim.port_count))
        next_port += item.claim.port_count
        assignments.append(GpuAssignment(item, gpu_uuids, rank_groups, ports))
        budget_rows.append((cell.cell_id, _sha(f"budget-{cell.cell_id}")))
    waves = tuple(
        GpuDispatchWave(
            wave_index=index,
            assignments=(assignment,),
            interference_envelope_sha256=_sha("interference"),
        )
        for index, assignment in enumerate(assignments)
    )
    return GpuDispatchPlan(
        schema_version=1,
        registry_sha256=registry.sha256,
        inventory_sha256=_sha("inventory"),
        receipts_sha256=_sha("receipts"),
        interference_envelope_sha256=_sha("interference"),
        budget_sha256_by_cell=tuple(sorted(budget_rows)),
        seed=20260811,
        waves=waves,
        completed_cell_ids=(),
    )


def test_preflight_bridge_maps_exact_ten_cells_to_first_party_runners() -> None:
    registry = build_industrial_registry()
    lock = _protocol_lock()
    materialization = materialize_preflight(
        protocol_lock_sha256=lock.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    bindings = _preflight_execution_bindings(
        materialization,
        protocol_lock=lock,
        registry=registry,
        dispatch_plan=_dispatch_plan(),
    )
    assert len(bindings) == 10
    assert {row.registry_cell_id for row in bindings} == {
        cell.cell_id for cell in registry.cells_for("preflight")
    }
    assert {
        runner: sum(row.runner_kind == runner for row in bindings)
        for runner in (
            "first_party_compile",
            "first_party_exactness",
            "first_party_interference",
        )
    } == {
        "first_party_compile": 1,
        "first_party_exactness": 1,
        "first_party_interference": 8,
    }
    assert all(row.cell == row.assignment.work_item.cell for row in bindings)
    assert all(row.assignment.sha256 == row.assignment_sha256 for row in bindings)
    assert all(row.assignment.ports for row in bindings)
    assert all(
        row.source_authority_bindings == lock.preflight_source_authority_bindings
        for row in bindings
    )


def test_preflight_bridge_rejects_partial_and_legacy_dispatch() -> None:
    registry = build_industrial_registry()
    lock = _protocol_lock()
    materialization = materialize_preflight(
        protocol_lock_sha256=lock.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    omitted = registry.cells_for("preflight")[0].cell_id
    with pytest.raises(ValueError, match="all ten fresh cells"):
        _preflight_execution_bindings(
            materialization,
            protocol_lock=lock,
            registry=registry,
            dispatch_plan=_dispatch_plan(omit_cell_id=omitted),
        )
    with pytest.raises(ValueError, match="legacy diagnostic registry"):
        _preflight_execution_bindings(
            materialization,
            protocol_lock=lock,
            registry=build_legacy_industrial_registry(),
            dispatch_plan=_dispatch_plan(),
        )


def test_verified_preflight_dispatch_cannot_be_constructed_by_a_caller() -> None:
    with pytest.raises(TypeError, match="verifier-constructed only"):
        VerifiedFormalPreflightDispatch(
            manifest=None,  # type: ignore[arg-type]
            protocol_lock=None,  # type: ignore[arg-type]
            subject=None,  # type: ignore[arg-type]
            capacity_control=None,  # type: ignore[arg-type]
            dispatch_control=None,  # type: ignore[arg-type]
            dispatch_context=None,  # type: ignore[arg-type]
            dispatch_plan=None,  # type: ignore[arg-type]
            challenge_reservation_sha256=_sha("reservation"),
            _construction_seal=object(),
        )
