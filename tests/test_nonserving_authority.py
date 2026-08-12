from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import lightcone_spec.experiments.nonserving_authority as download_authority
from lightcone_spec import experiments
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.nonserving_authority import (
    DOWNLOAD_RESULT_POINTER_PROTOCOL_SHA256,
    DOWNLOAD_TERMINAL_PROTOCOL_SHA256,
    RELEASE_DOWNLOAD_POINTER_PUBLISHERS,
    RELEASE_DOWNLOAD_TERMINAL_ISSUERS,
    RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON,
    DownloadExecutionBlocked,
    DownloadModelRevision,
    DownloadOutputArtifact,
    DownloadOutputExpectation,
    DownloadPlan,
    DownloadResultPointer,
    DownloadTerminalReceipt,
    FutureDownloadRawAuthority,
    bind_download_plan_authority,
    bind_future_download_raw_authority,
    issue_download_plan,
    require_release_download_execution,
    revalidate_download_plan_authority,
    revalidate_future_download_raw_authority,
)
from lightcone_spec.experiments.planning import (
    ZERO_COUNT,
    ZERO_MILLISECONDS,
    BudgetJobKind,
    ExperimentBudget,
    P99AnchorStatus,
    ScenarioMilliseconds,
)
from lightcone_spec.experiments.registry import (
    ExperimentCell,
    ExperimentRegistry,
    WorkloadClass,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.orchestration.industrial import IndustrialPhysicalAssignment


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_semantic_json(path: Path, value: object, semantic_sha256: str) -> bytes:
    body = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    Path(f"{path}.sha256").write_text(f"{semantic_sha256}\n", encoding="ascii")
    return body


def _inventory() -> GpuInventory:
    uuids = ("GPU-000", "GPU-001")
    devices = tuple(
        GpuDevice(
            uuid=uuid,
            host_id="host-a",
            model="H100-SXM",
            memory_bytes=80 * 1024**3,
            compute_capability=(9, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root="root-0",
            numa_node=0,
            interconnects=("NVLink4", "PCIe5"),
            peer_access_class="NVSwitch",
            clock_policy="locked-1980MHz",
            power_limit_watts=700.0,
            thermal_limit_celsius=83.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=("pair-00",),
        )
        for index, uuid in enumerate(uuids)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id="pair-00",
                host_id="host-a",
                gpu_uuids=uuids,
                fabric="NVLink",
                bandwidth_class="high",
            ),
        ),
        source_receipt_sha256=content_sha256("download-inventory-receipt"),
    )


def _download_budget(cell: ExperimentCell) -> ExperimentBudget:
    duration = ScenarioMilliseconds(100, 100, 100)
    gpu_duration = duration.scale(2)
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=WorkloadClass.DOWNLOAD,
        job_kind=BudgetJobKind.DOWNLOAD,
        startup_model_load=ZERO_MILLISECONDS,
        compile_jit_graph_prewarm=ZERO_MILLISECONDS,
        excluded_warmup=ZERO_MILLISECONDS,
        excluded_warmup_requests=ZERO_COUNT,
        scored_arrival=ZERO_MILLISECONDS,
        request_deadline=ZERO_MILLISECONDS,
        drain=ZERO_MILLISECONDS,
        reset_finalization=ZERO_MILLISECONDS,
        evidence_flush_shutdown=ZERO_MILLISECONDS,
        output_tokens=ZERO_COUNT,
        minimum_completed_requests=0,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=ZERO_MILLISECONDS,
        failure_injection=ZERO_MILLISECONDS,
        retry=ZERO_MILLISECONDS,
        retry_allowance=0,
        profiler=ZERO_MILLISECONDS,
        download_compile_reservation=duration,
        gpu_count=2,
        topology=cell.identity.topology,
        reserved_gpu_ms=gpu_duration,
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=gpu_duration,
    )


def _assignment(
    inventory: GpuInventory,
    budget: ExperimentBudget,
) -> IndustrialPhysicalAssignment:
    return IndustrialPhysicalAssignment(
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        dispatch_plan_sha256=content_sha256("download-dispatch"),
        experiment_budget_sha256=budget.sha256,
        budget_plan_sha256=content_sha256("download-budget-plan"),
        capacity_authority_sha256=content_sha256("download-capacity"),
        budget_materialization_authority_sha256=content_sha256(
            "download-budget-authority"
        ),
        assignment_sha256=content_sha256("download-scheduler-assignment"),
        work_item_sha256=content_sha256("download-work-item"),
        gpu_uuids=("GPU-000", "GPU-001"),
        rank_groups=(("GPU-000", "GPU-001"),),
        ports=(31_000,),
        tensor_parallel_size=2,
        data_parallel_size=1,
        fixed_instance_gpu_count=2,
        host_id="host-a",
        topology_group_ids=(("pair-00",),),
    )


@dataclass(frozen=True)
class _DownloadCase:
    registry: ExperimentRegistry
    cell: ExperimentCell
    inventory: GpuInventory
    budget: ExperimentBudget
    assignment: IndustrialPhysicalAssignment
    plan: DownloadPlan
    payloads: tuple[tuple[str, bytes], ...]


def _case(tmp_path: Path) -> _DownloadCase:
    cache_root = (tmp_path / "cache").resolve()
    evidence_root = (tmp_path / "evidence").resolve()
    cache_root.mkdir()
    evidence_root.mkdir()
    registry = build_industrial_registry(
        cache_root=str(cache_root),
        evidence_root=str(evidence_root),
    )
    cell = next(
        value
        for value in registry.cells_for("E6")
        if value.resources.workload_class is WorkloadClass.DOWNLOAD
    )
    inventory = _inventory()
    budget = _download_budget(cell)
    assignment = _assignment(inventory, budget)
    payloads = (
        ("snapshots/config.json", b'{"architectures":["Qwen"]}\n'),
        ("snapshots/model.safetensors", b"immutable-model-payload"),
    )
    expected_outputs = tuple(
        DownloadOutputExpectation(
            relative_path=relative_path,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for relative_path, payload in payloads
    )
    revisions = (
        DownloadModelRevision(
            role="target",
            repository=cell.identity.model,
            revision="1" * 40,
            source_manifest_sha256=content_sha256("locked-download-model"),
        ),
    )
    plan = issue_download_plan(
        registry=registry,
        cell=cell,
        model_revisions=revisions,
        inventory=inventory,
        assignment=assignment,
        budget=budget,
        expected_outputs=expected_outputs,
    )
    return _DownloadCase(
        registry=registry,
        cell=cell,
        inventory=inventory,
        budget=budget,
        assignment=assignment,
        plan=plan,
        payloads=payloads,
    )


def _publish_plan(case: _DownloadCase):
    path = Path(case.plan.plan_path)
    _write_semantic_json(path, case.plan.to_dict(), case.plan.sha256)
    return bind_download_plan_authority(path, expected_plan=case.plan)


def _terminal(case: _DownloadCase) -> DownloadTerminalReceipt:
    outputs: list[DownloadOutputArtifact] = []
    for relative_path, payload in case.payloads:
        path = Path(case.plan.inputs.cache_root) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        outputs.append(
            DownloadOutputArtifact(
                relative_path=relative_path,
                absolute_path=str(path),
                size=len(payload),
                raw_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    inputs = case.plan.inputs
    output_tuple = tuple(outputs)
    return DownloadTerminalReceipt(
        schema_version=1,
        kind="download_terminal_receipt",
        terminal_protocol_sha256=DOWNLOAD_TERMINAL_PROTOCOL_SHA256,
        plan_sha256=case.plan.sha256,
        plan_inputs_sha256=inputs.sha256,
        registry_sha256=inputs.registry_sha256,
        cell_id=inputs.cell_id,
        cell_declaration_sha256=inputs.cell_declaration_sha256,
        model_revision_sha256s=inputs.model_revision_sha256s,
        model_revision_manifest_sha256=inputs.model_revision_manifest_sha256,
        inventory_sha256=inputs.inventory_sha256,
        inventory_source_receipt_sha256=inputs.inventory_source_receipt_sha256,
        physical_assignment_sha256=inputs.physical_assignment_sha256,
        assignment_sha256=inputs.assignment_sha256,
        budget_materialization_authority_sha256=(
            inputs.budget_materialization_authority_sha256
        ),
        experiment_budget_sha256=inputs.experiment_budget_sha256,
        started_monotonic_ns=10,
        finished_monotonic_ns=20,
        exit_code=0,
        terminal_status="COMPLETE",
        headline_eligible=False,
        outputs=output_tuple,
        output_manifest_sha256=content_sha256(
            [value.to_dict() for value in output_tuple]
        ),
        issuer_id="future-downloader",
        issuer_version_sha256=content_sha256("future-downloader-version"),
        signature_hex="a" * 128,
    )


def _publish_future(case: _DownloadCase):
    plan_authority = _publish_plan(case)
    terminal = _terminal(case)
    terminal_body = _write_semantic_json(
        Path(case.plan.terminal_receipt_path),
        terminal.to_dict(),
        terminal.sha256,
    )
    pointer = DownloadResultPointer(
        schema_version=1,
        kind="download_result_pointer",
        protocol_sha256=DOWNLOAD_RESULT_POINTER_PROTOCOL_SHA256,
        plan_sha256=case.plan.sha256,
        plan_authority_sha256=plan_authority.sha256,
        terminal_path=case.plan.terminal_receipt_path,
        terminal_size=len(terminal_body),
        terminal_raw_sha256=hashlib.sha256(terminal_body).hexdigest(),
        terminal_semantic_sha256=terminal.sha256,
        outputs=terminal.outputs,
        output_manifest_sha256=terminal.output_manifest_sha256,
        publisher_id="future-download-publisher",
        publisher_version_sha256=content_sha256("future-publisher-version"),
        signature_hex="b" * 128,
    )
    _write_semantic_json(
        Path(case.plan.result_pointer_path),
        pointer.to_dict(),
        pointer.sha256,
    )
    authority = bind_future_download_raw_authority(
        plan_authority,
        expected_plan=case.plan,
    )
    return plan_authority, terminal, pointer, authority


def test_download_plan_binds_every_registered_execution_boundary_and_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    inputs = case.plan.inputs

    assert inputs.registry_sha256 == case.registry.sha256
    assert inputs.cell_id == case.cell.cell_id
    assert inputs.cell_declaration_sha256 == case.cell.sha256
    assert inputs.inventory_sha256 == case.inventory.sha256
    assert (
        inputs.inventory_source_receipt_sha256 == case.inventory.source_receipt_sha256
    )
    assert inputs.physical_assignment_sha256 == case.assignment.sha256
    assert inputs.assignment_sha256 == case.assignment.assignment_sha256
    assert (
        inputs.budget_materialization_authority_sha256
        == case.assignment.budget_materialization_authority_sha256
    )
    assert inputs.experiment_budget_sha256 == case.budget.sha256
    assert case.plan.expected_outputs
    assert RELEASE_DOWNLOAD_TERMINAL_ISSUERS == ()
    assert RELEASE_DOWNLOAD_POINTER_PUBLISHERS == ()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("formal DOWNLOAD gate performed a side effect")

    monkeypatch.setattr(download_authority.os, "open", forbidden)
    monkeypatch.setattr(download_authority, "_stable_json_bytes", forbidden)
    monkeypatch.setattr(download_authority, "_stable_file_digest", forbidden)
    with pytest.raises(DownloadExecutionBlocked) as captured:
        require_release_download_execution(case.plan)
    assert captured.value.reason_code == RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON


def test_download_plan_raw_authority_rejects_symlink_tamper_and_caller_ids(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    authority = _publish_plan(case)
    assert (
        revalidate_download_plan_authority(authority, expected_plan=case.plan)
        == case.plan
    )

    plan_path = Path(case.plan.plan_path)
    backing = plan_path.with_name("download-plan-backing.json")
    plan_path.rename(backing)
    plan_path.symlink_to(backing)
    with pytest.raises(ValueError, match="resolved, and non-symlink"):
        bind_download_plan_authority(plan_path, expected_plan=case.plan)
    plan_path.unlink()
    backing.rename(plan_path)

    forged = case.plan.to_dict()
    forged_inputs = forged["inputs"]
    assert isinstance(forged_inputs, dict)
    forged_inputs["assignment_sha256"] = "f" * 64
    forged_plan = DownloadPlan.from_dict(forged)
    _write_semantic_json(plan_path, forged, forged_plan.sha256)
    with pytest.raises(ValueError, match="release-derived inputs"):
        bind_download_plan_authority(plan_path, expected_plan=case.plan)
    with pytest.raises(ValueError, match="fresh raw replay"):
        revalidate_download_plan_authority(authority, expected_plan=case.plan)


def test_future_download_raw_receipts_replay_but_never_mint_completion(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    _, _, _, authority = _publish_future(case)

    assert isinstance(authority, FutureDownloadRawAuthority)
    assert authority.formal_status == "BLOCKED"
    assert authority.reason_code == RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON
    assert (
        revalidate_future_download_raw_authority(
            authority,
            expected_plan=case.plan,
        )
        == authority
    )
    with pytest.raises(DownloadExecutionBlocked):
        require_release_download_execution(case.plan)


def test_future_download_replay_rejects_joint_terminal_pointer_rehash(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    plan_authority, terminal, _, authority = _publish_future(case)

    changed_terminal = DownloadTerminalReceipt.from_dict(
        {
            **terminal.to_dict(),
            "issuer_id": "jointly-rehashed-downloader",
        }
    )
    changed_terminal_body = _write_semantic_json(
        Path(case.plan.terminal_receipt_path),
        changed_terminal.to_dict(),
        changed_terminal.sha256,
    )
    changed_pointer = DownloadResultPointer(
        schema_version=1,
        kind="download_result_pointer",
        protocol_sha256=DOWNLOAD_RESULT_POINTER_PROTOCOL_SHA256,
        plan_sha256=case.plan.sha256,
        plan_authority_sha256=plan_authority.sha256,
        terminal_path=case.plan.terminal_receipt_path,
        terminal_size=len(changed_terminal_body),
        terminal_raw_sha256=hashlib.sha256(changed_terminal_body).hexdigest(),
        terminal_semantic_sha256=changed_terminal.sha256,
        outputs=changed_terminal.outputs,
        output_manifest_sha256=changed_terminal.output_manifest_sha256,
        publisher_id="jointly-rehashed-publisher",
        publisher_version_sha256=content_sha256("changed-publisher-version"),
        signature_hex="c" * 128,
    )
    _write_semantic_json(
        Path(case.plan.result_pointer_path),
        changed_pointer.to_dict(),
        changed_pointer.sha256,
    )

    with pytest.raises(ValueError, match="fresh raw replay"):
        revalidate_future_download_raw_authority(
            authority,
            expected_plan=case.plan,
        )
    rebound = bind_future_download_raw_authority(
        plan_authority,
        expected_plan=case.plan,
    )
    assert rebound.formal_status == "BLOCKED"
    assert rebound.reason_code == RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON


@pytest.mark.parametrize(
    "path_field",
    ("terminal_receipt_path", "result_pointer_path"),
)
def test_download_raw_receipt_symlinks_are_not_authority(
    tmp_path: Path,
    path_field: str,
) -> None:
    case = _case(tmp_path)
    _, _, _, authority = _publish_future(case)
    path = Path(getattr(case.plan, path_field))
    backing = path.with_name(f"{path.name}.backing")
    path.rename(backing)
    path.symlink_to(backing)

    with pytest.raises(ValueError, match="resolved, and non-symlink"):
        revalidate_future_download_raw_authority(
            authority,
            expected_plan=case.plan,
        )


def test_download_output_symlink_is_not_raw_authority(tmp_path: Path) -> None:
    case = _case(tmp_path)
    _, terminal, _, authority = _publish_future(case)
    output = Path(terminal.outputs[0].absolute_path)
    external = (tmp_path / "same-bytes-outside-cache").resolve()
    external.write_bytes(output.read_bytes())
    output.unlink()
    output.symlink_to(external)

    with pytest.raises(ValueError, match="non-symlink"):
        revalidate_future_download_raw_authority(
            authority,
            expected_plan=case.plan,
        )


def test_download_authority_surface_is_exported() -> None:
    assert experiments.RELEASE_DOWNLOAD_TERMINAL_ISSUERS == ()
    assert experiments.RELEASE_DOWNLOAD_POINTER_PUBLISHERS == ()
    assert experiments.DownloadPlan is DownloadPlan
    assert experiments.DownloadTerminalReceipt is DownloadTerminalReceipt
    assert experiments.DownloadResultPointer is DownloadResultPointer
    assert experiments.bind_download_plan_authority is bind_download_plan_authority
    assert (
        experiments.bind_future_download_raw_authority
        is bind_future_download_raw_authority
    )
    assert (
        experiments.require_release_download_execution
        is require_release_download_execution
    )
