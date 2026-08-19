from __future__ import annotations

import hashlib
import inspect
import json
from asyncio import run
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments.formal_preflight_execution import (
    _execute_formal_preflight_interference_raw_core,
)
from lightcone_spec.experiments.formal_preflight_inputs import (
    FORMAL_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256,
    FORMAL_SINGLE_OPERATOR_PREFLIGHT_EXECUTION_PROTOCOL_SHA256,
    FormalSingleOperatorPreflightCompletion,
    FormalSingleOperatorPreflightCompletionRow,
    FormalSingleOperatorPreflightExecution,
    FormalSingleOperatorPreflightInterferenceEvidence,
    FormalSingleOperatorPreflightInterferenceExecution,
    _trusted_budget_plan_sha256,
    _trusted_preflight_bindings,
    _validate_completion_junit,
    execute_formal_single_operator_preflight_compile,
    execute_formal_single_operator_preflight_exact_ten,
    execute_formal_single_operator_preflight_exactness,
    execute_formal_single_operator_preflight_interference,
)
from lightcone_spec.experiments.formal_protocol import ProtocolLock
from lightcone_spec.experiments.formal_registry import (
    protocol_lock_to_dict,
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    load_formal_single_operator_execution_source,
    materialize_formal_single_operator_node,
    publish_formal_single_operator_execution_source,
    publish_formal_single_operator_json_artifact,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical_json(path: Path, value: object) -> CanonicalJsonProofBinding:
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    return CanonicalJsonProofBinding.bind(path.resolve())


def _lock() -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="formal-preflight-input-test",
        code_git_head="1" * 40,
        code_git_tree="2" * 40,
        patch_manifest_sha256=_sha("patch"),
        registry_sha256=build_industrial_registry().sha256,
        english_protocol_sha256=_sha("english"),
        chinese_protocol_sha256=_sha("chinese"),
        tts_calibration_authority_sha256=_sha("tts"),
        chronobelief_authority_sha256=_sha("chrono"),
        e1_recipe_anchor_authority_sha256=_sha("e1"),
        e2_recipe_grid_authority_sha256=_sha("e2"),
        formal_runtime_authority_manifest_sha256=_sha("runtime"),
        offline_release_trust_root_sha256=_sha("root"),
        prepared_model_content_authorization_sha256=_sha("prepared"),
        formal_workload_e3a_authorization_sha256=_sha("e3a"),
        formal_workload_e0_authorization_sha256=_sha("e0"),
        burstgpt_shape_authorization_sha256=_sha("burst"),
        native_runtime_qualification_protocol_sha256=_sha("native-protocol"),
        native_runtime_qualification_runner_sha256=_sha("native-runner"),
        native_runtime_qualification_test_set_sha256=_sha("native-tests"),
        compile_qualification_protocol_sha256=_sha("compile-protocol"),
        compile_qualification_runner_sha256=_sha("compile-runner"),
        compile_qualification_test_set_sha256=_sha("compile-tests"),
        exactness_qualification_protocol_sha256=_sha("exact-protocol"),
        exactness_qualification_runner_sha256=_sha("exact-runner"),
        exactness_qualification_test_set_sha256=_sha("exact-tests"),
    )


def _inventory() -> GpuInventory:
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-{slot}",
            host_id="host",
            model="Test GPU",
            memory_bytes=96_000_000_000,
            compute_capability=(12, 0),
            pci_bus_id=f"0000:{slot + 1:02x}:00.0",
            pci_root="root-0",
            numa_node=0,
            interconnects=("PCIe",),
            peer_access_class="peer-enabled",
            clock_policy="locked",
            power_limit_watts=600.0,
            thermal_limit_celsius=85.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=("pair",),
        )
        for slot in range(2)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id="pair",
                host_id="host",
                gpu_uuids=("GPU-0", "GPU-1"),
                fabric="PCIe",
                bandwidth_class="test",
            ),
        ),
        source_receipt_sha256=_sha("inventory"),
    )


def _source(tmp_path: Path):
    lock = _lock()
    lock_path = (tmp_path / "protocol-lock.json").resolve()
    publish_formal_single_operator_json_artifact(
        lock_path,
        protocol_lock_to_dict(lock),
    )
    materialize_formal_single_operator_node(
        node="preflight",
        predecessor_completion_path=None,
        protocol_lock_path=lock_path,
        materialization_output_path=(tmp_path / "materialization.json").resolve(),
        node_materialization_output_path=(
            tmp_path / "node-materialization.json"
        ).resolve(),
        created_ns=1,
    )
    source_path = (tmp_path / "execution-source.json").resolve()
    publish_formal_single_operator_execution_source(
        node_materialization_path=(tmp_path / "node-materialization.json").resolve(),
        output_path=source_path,
    )
    return lock, load_formal_single_operator_execution_source(source_path)


def test_trusted_current_source_projects_exact_ten_assignments(tmp_path: Path) -> None:
    lock, source = _source(tmp_path)
    inventory = _inventory()
    rows = _trusted_preflight_bindings(
        source=source,
        protocol_lock=lock,
        inventory=inventory,
    )
    assert len(rows) == 10
    assert [row.runner_kind for row in rows].count("first_party_compile") == 1
    assert [row.runner_kind for row in rows].count("first_party_exactness") == 1
    assert [row.runner_kind for row in rows].count("first_party_interference") == 8
    assert {row.materialized_cell_id for row in rows} == {
        cell.cell_id
        for cell in stage_materialization_receipt_from_dict(
            source.materialization_source.reopen(label="test preflight materialization")
        ).cells
    }
    assert all(
        row.gpu_uuids == ("GPU-0", "GPU-1")
        for row in rows
        if row.runner_kind != "first_party_interference"
    )
    assert {
        row.gpu_uuids for row in rows if row.runner_kind == "first_party_interference"
    } == {("GPU-0",), ("GPU-1",)}
    assert _trusted_budget_plan_sha256(source, inventory, rows) == (
        _trusted_budget_plan_sha256(source, inventory, rows)
    )


def test_trusted_projection_rejects_nonready_or_foreign_inventory(
    tmp_path: Path,
) -> None:
    lock, source = _source(tmp_path)
    inventory = _inventory()
    unavailable = replace(
        inventory,
        devices=(
            replace(inventory.devices[0], availability=GpuAvailability.UNAVAILABLE),
            inventory.devices[1],
        ),
    )
    with pytest.raises(ValueError, match="exact/current/ready"):
        _trusted_preflight_bindings(
            source=source,
            protocol_lock=lock,
            inventory=unavailable,
        )
    one_gpu = replace(
        inventory,
        devices=(replace(inventory.devices[0], allowed_topology_groups=()),),
        topology_groups=(),
    )
    with pytest.raises(ValueError, match="exact/current/ready"):
        _trusted_preflight_bindings(
            source=source,
            protocol_lock=lock,
            inventory=one_gpu,
        )


def test_exact_ten_completion_codec_and_junit_are_fail_closed(tmp_path: Path) -> None:
    lock, source = _source(tmp_path)
    bindings = _trusted_preflight_bindings(
        source=source,
        protocol_lock=lock,
        inventory=_inventory(),
    )
    common = _canonical_json(tmp_path / "common.json", {"kind": "source"})
    rows = tuple(
        sorted(
            (
                FormalSingleOperatorPreflightCompletionRow(
                    materialized_cell_id=row.materialized_cell_id,
                    registry_cell_id=row.registry_cell_id,
                    runner_kind=row.runner_kind,
                    status="COMPLETE",
                    started_ns=index + 1,
                    finished_ns=index + 2,
                    result_sha256=_sha(f"result-{index}"),
                )
                for index, row in enumerate(bindings)
            ),
            key=lambda row: row.registry_cell_id,
        )
    )
    interference_evidence = []
    for index, row in enumerate(
        item for item in bindings if item.runner_kind == "first_party_interference"
    ):
        terminal = _canonical_json(
            tmp_path / f"terminal-{index}.json", {"kind": "terminal", "index": index}
        )
        lifecycle = _canonical_json(
            tmp_path / f"lifecycle-{index}.json",
            {"kind": "lifecycle", "index": index},
        )
        request_ids = (f"request-{index}-0", f"request-{index}-1")
        junit_path = (tmp_path / f"junit-{index}.xml").resolve()
        junit_path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<testsuite name="lightcone-formal-serving" tests="2" '
            'failures="0" errors="0" skipped="0">'
            f'<testcase classname="lightcone.tp1_dp1" name="{request_ids[0]}"/>'
            f'<testcase classname="lightcone.tp1_dp1" name="{request_ids[1]}"/>'
            "</testsuite>\n",
            encoding="utf-8",
        )
        assert _validate_completion_junit(
            junit_path,
            expected_request_ids=request_ids,
        ) == EvidenceFileBinding.bind(junit_path, label="test JUnit")
        interference_evidence.append(
            FormalSingleOperatorPreflightInterferenceEvidence(
                materialized_cell_id=row.materialized_cell_id,
                registry_cell_id=row.registry_cell_id,
                terminal_result_proof=terminal,
                lifecycle_timing=lifecycle,
                junit_xml=EvidenceFileBinding.bind(junit_path, label="test JUnit"),
            )
        )
    artifact = FormalSingleOperatorPreflightCompletion(
        schema_version=1,
        kind="formal_single_operator_exact_ten_preflight_completion",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256,
        execution_inputs=common,
        compile_result=common,
        exactness_result=common,
        interference_evidence=tuple(
            sorted(interference_evidence, key=lambda row: row.registry_cell_id)
        ),
        rows=rows,
        status="COMPLETE",
        started_ns=min(row.started_ns for row in rows),
        finished_ns=max(row.finished_ns for row in rows),
    )
    assert (
        FormalSingleOperatorPreflightCompletion.from_dict(artifact.to_dict())
        == artifact
    )
    changed = artifact.to_dict()
    changed["rows"][0]["status"] = "FAILED"  # type: ignore[index]
    with pytest.raises(ValueError, match="aggregate outcome differs"):
        FormalSingleOperatorPreflightCompletion.from_dict(changed)
    junit_path.write_text("<testsuite tests='0'/>", encoding="utf-8")
    with pytest.raises(ValueError, match="coverage differs"):
        _validate_completion_junit(
            junit_path,
            expected_request_ids=("request-7-0", "request-7-1"),
        )


def test_trusted_execution_codecs_and_public_surface_are_closed(
    tmp_path: Path,
) -> None:
    common = _canonical_json(tmp_path / "source.json", {"kind": "source"})
    interference = FormalSingleOperatorPreflightInterferenceExecution(
        schema_version=1,
        kind="formal_single_operator_preflight_interference_execution",
        execution_inputs=common,
        raw_batch=common,
        evidence=(),
        status="ERROR",
    )
    assert (
        FormalSingleOperatorPreflightInterferenceExecution.from_dict(
            interference.to_dict()
        )
        == interference
    )
    with pytest.raises(ValueError, match="not exact eight"):
        FormalSingleOperatorPreflightInterferenceExecution(
            **{**interference.__dict__, "status": "WAITING_FOR_COMPLETION"}
        )

    exact_ten = FormalSingleOperatorPreflightExecution(
        schema_version=1,
        kind="formal_single_operator_exact_ten_preflight_execution",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PREFLIGHT_EXECUTION_PROTOCOL_SHA256,
        execution_inputs=common,
        compile_result=common,
        exactness_result=common,
        interference_execution=common,
        completion=None,
        status="ERROR",
    )
    assert (
        FormalSingleOperatorPreflightExecution.from_dict(exact_ten.to_dict())
        == exact_ten
    )
    with pytest.raises(ValueError, match="completion presence differs"):
        FormalSingleOperatorPreflightExecution(
            **{**exact_ten.__dict__, "status": "COMPLETE"}
        )

    for function in (
        execute_formal_single_operator_preflight_compile,
        execute_formal_single_operator_preflight_exact_ten,
        execute_formal_single_operator_preflight_exactness,
        execute_formal_single_operator_preflight_interference,
    ):
        assert tuple(inspect.signature(function).parameters) == (
            "execution_inputs_path",
            "current_ns",
        )
    with pytest.raises(TypeError, match="admission is not exact"):
        run(
            _execute_formal_preflight_interference_raw_core(
                SimpleNamespace(),
                launch_cap_schedule_path=None,
                execution_inputs={},
                nvidia_smi_tool=object(),
                evidence_root=(tmp_path / "unused").resolve(),
                now_ns=1,
            )
        )
