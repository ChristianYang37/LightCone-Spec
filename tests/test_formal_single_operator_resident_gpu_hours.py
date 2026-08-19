from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.config import ModelPair, RunConfig, RuntimeConfig, run_config_sha256
from lightcone_spec.experiments import formal_single_operator_gpu_hours as gpu_hours
from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime import formal_single_operator as single
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding(root: Path, label: str) -> CanonicalJsonProofBinding:
    path = (root / f"{label}.json").resolve()
    publish_canonical_json_no_replace(path, {"kind": label, "value": _sha(label)})
    return CanonicalJsonProofBinding.bind(path)


def _inventory() -> GpuInventory:
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-{index}",
            host_id="resident-hour-host",
            model="RTX PRO 6000 Blackwell Server Edition",
            memory_bytes=96 * 1024**3,
            compute_capability=(12, 0),
            pci_bus_id=f"0000:0{index + 1}:00.0",
            pci_root="root-0",
            numa_node=0,
            interconnects=("PCIe",),
            peer_access_class="P2P",
            clock_policy="locked",
            power_limit_watts=600.0,
            thermal_limit_celsius=83.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=("pair",),
        )
        for index in range(2)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id="pair",
                host_id="resident-hour-host",
                gpu_uuids=tuple(row.uuid for row in devices),
                fabric="PCIe",
                bandwidth_class="local",
            ),
        ),
        source_receipt_sha256=_sha("inventory-source"),
    )


def _materialization() -> StageMaterializationReceipt:
    cell = MaterializedCell(
        stage="E3a",
        method_role="Target-only",
        model="Qwen/Qwen3-8B",
        backend="DFlash",
        task="LiveCodeBench",
        publication_policy="none",
        recipe_sha256=None,
        dimensions=(("case", 0),),
    )
    return StageMaterializationReceipt(
        schema_version=1,
        stage="E3a",
        protocol_lock_sha256=_sha("protocol-lock"),
        upstream_receipt_sha256s=(_sha("preflight"),),
        source_decision_sha256=_sha("e3a-source"),
        materialization_rule="gpu_hour_test_exact_cells",
        expected_cell_count=1,
        cells=(cell,),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _physical(
    root: Path,
    *,
    label: str,
    kind: str,
    gpu_uuids: tuple[str, ...],
    started: int,
    exited: int,
    empty: int,
    flushed: int,
) -> gpu_hours._UnifiedPhysicalExecution:
    return gpu_hours._UnifiedPhysicalExecution(
        physical_execution_id=_sha(f"physical:{label}"),
        execution_kind=kind,
        source=_binding(root, f"physical-{label}"),
        gpu_uuids=gpu_uuids,
        phase_edges_ns=(
            ("server_process_started_ns", started),
            ("process_exited_ns", exited),
            ("process_group_empty_checked_ns", empty),
            ("evidence_flush_completed_ns", flushed),
        ),
    )


def _cell(
    root: Path,
    *,
    label: str,
    physical: gpu_hours._UnifiedPhysicalExecution,
    projection_source: str,
    trace_ns: int = 7,
) -> gpu_hours._UnifiedCellObservation:
    return gpu_hours._UnifiedCellObservation(
        cell_id=_sha(f"cell:{label}"),
        actual_result=_binding(root, f"actual-{label}"),
        actual_result_sha256=_sha(f"actual-semantic:{label}"),
        member_lifecycle=_binding(root, f"member-{label}"),
        physical_execution_id=physical.physical_execution_id,
        topology="tp1_dp1",
        gang_gpu_count=len(physical.gpu_uuids),
        provider_reserved_gpu_count=2,
        scored_request_count=8,
        projection_process_ns=trace_ns,
        projection_core_wall_ns=trace_ns,
        projection_evidence_tail_ns=0,
        projection_source=projection_source,
    )


def _typed_resident_manifest(
    root: Path,
) -> tuple[
    single.FormalSingleOperatorResidentRunManifest,
    Path,
    StageMaterializationReceipt,
    GpuInventory,
]:
    run_root = (root / "typed-resident-run").resolve()
    run_root.mkdir()
    inventory = _inventory()
    materialization = _materialization()
    inventory_path = (root / "typed-inventory.json").resolve()
    materialization_path = (root / "typed-materialization.json").resolve()
    publish_canonical_json_no_replace(inventory_path, inventory.to_dict())
    publish_canonical_json_no_replace(
        materialization_path,
        stage_materialization_receipt_to_dict(materialization),
    )
    config = RunConfig(
        method="target_only",
        model=ModelPair(
            target_revision="1" * 40,
            drafter_revision="2" * 40,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256="3" * 64,
            speculation_enabled=False,
        ),
    )
    argv = ("python", "-m", "sglang.launch_server", "--port", "31001")
    request_schedule = {
        "kind": "formal_serving_request_schedule_receipt",
        "requests": [{"request_id": "scored-0", "phase": "scored"}],
    }
    request_schedule_path = (root / "typed-request-schedule.json").resolve()
    publish_canonical_json_no_replace(request_schedule_path, request_schedule)
    binding_names = (
        "run-plan",
        "launch-manifest",
        "group-plan",
        "reset-authority",
        "shared-launch",
        "reset-boundary",
        "resident-trace",
        "shared-close",
    )
    bindings = {name: _binding(root, f"typed-{name}") for name in binding_names}
    artifact_names = (
        "client_lifecycle",
        "execution_source",
        "inventory",
        "junit",
        "lifecycle",
        "live_run_receipt",
        "materialization",
        "native_itl",
        "protocol_lock",
        "raw_terminal",
        "request_schedule",
        "run_plan",
        "run_plan_inputs",
    )
    artifacts = []
    for name in artifact_names:
        artifact_path = run_root / f"artifact-{name}.json"
        publish_canonical_json_no_replace(
            artifact_path,
            {"kind": f"typed-{name}"},
        )
        artifacts.append(
            single.FormalSingleOperatorArtifact.observe(
                name=name,
                run_root=run_root,
                path=artifact_path,
            )
        )
    manifest = single.FormalSingleOperatorResidentRunManifest(
        schema_version=1,
        kind="formal_single_operator_resident_run_manifest",
        protocol_sha256=single.FORMAL_SINGLE_OPERATOR_RESIDENT_PROTOCOL_SHA256,
        trust_assumptions=single.FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        evidence_level="trusted_single_operator_empirical_no_signature",
        formal_measured=False,
        git_head="4" * 40,
        git_tree="5" * 40,
        sglang_upstream_commit="6" * 40,
        patch_manifest_sha256="7" * 64,
        patched_sglang_tree="8" * 40,
        registry_sha256=build_industrial_registry().sha256,
        run_plan=bindings["run-plan"],
        launch_manifest=bindings["launch-manifest"],
        materialization=CanonicalJsonProofBinding.bind(materialization_path),
        inventory=CanonicalJsonProofBinding.bind(inventory_path),
        request_schedule_binding=CanonicalJsonProofBinding.bind(request_schedule_path),
        group_plan=bindings["group-plan"],
        reset_authority=bindings["reset-authority"],
        shared_launch=bindings["shared-launch"],
        reset_boundary=bindings["reset-boundary"],
        resident_trace=bindings["resident-trace"],
        shared_close=bindings["shared-close"],
        group_id=_sha("group"),
        group_session_binding_sha256=_sha("session"),
        member_index=0,
        session_epoch=1,
        physical_dispatch_protocol_sha256=_sha("dispatch"),
        execution_binding_sha256=_sha("execution-binding"),
        execution_subject_sha256=_sha("execution-subject"),
        materialization_protocol_lock_sha256=(materialization.protocol_lock_sha256),
        materialization_sha256=materialization.sha256,
        inventory_sha256=inventory.sha256,
        run_config_sha256=run_config_sha256(config),
        run_config=config.model_dump(mode="json"),
        registered_launch_argv_sha256=gpu_hours.content_sha256({"argv": list(argv)}),
        registered_launch_argv=argv,
        registered_localhost_port=31001,
        request_schedule_sha256=gpu_hours.content_sha256(request_schedule),
        request_schedule=request_schedule,
        target_model_id="Qwen/Qwen3-8B",
        target_revision="1" * 40,
        target_snapshot_sha256=_sha("target-snapshot"),
        drafter_model_id=None,
        drafter_revision=None,
        tokenizer_model_id="Qwen/Qwen3-8B",
        tokenizer_revision="1" * 40,
        workload_artifact_id="livecodebench-v6-hard",
        workload_member_sha256s=(_sha("workload-member"),),
        workload_raw_sha256=_sha("workload-raw"),
        workload_semantic_sha256=_sha("workload-semantic"),
        stage="E3a",
        cell_id=materialization.cells[0].cell_id,
        role=materialization.cells[0].method_role,
        backend=materialization.cells[0].backend,
        topology="tp1_dp1",
        block=None,
        attempt="attempt-0001",
        run_directory=str(run_root),
        gpu_environment=(
            single.FormalSingleOperatorGpu(
                uuid="GPU-0",
                model=inventory.devices[0].model,
                driver_version="580.95.05",
                cuda_version="13.0",
            ),
        ),
        trace_started_ns=100,
        scored_started_ns=105,
        trace_finished_ns=130,
        completion_status="COMPLETE",
        artifacts=tuple(sorted(artifacts, key=lambda row: row.name)),
    )
    manifest_path = run_root / "formal-single-operator-manifest.json"
    publish_canonical_json_no_replace(manifest_path, manifest.to_dict())
    return manifest, manifest_path, materialization, inventory


def test_resident_members_charge_one_shared_physical_execution(tmp_path: Path) -> None:
    physical = _physical(
        tmp_path,
        label="resident",
        kind="resident_session",
        gpu_uuids=("GPU-0",),
        started=10,
        exited=30,
        empty=35,
        flushed=40,
    )
    cells = tuple(
        _cell(
            tmp_path,
            label=f"resident-{index}",
            physical=physical,
            projection_source="resident_member_trace",
        )
        for index in range(5)
    )

    cost = gpu_hours._actual_cost_from_unified_observations(
        cells,
        (physical,) * len(cells),
        inventory_gpu_count=2,
    )

    assert cost.cell_count == 5
    assert cost.compute_gpu_ns == 20
    assert cost.provider_base_reserved_gpu_ns == 50
    assert cost.wall_ns == 25
    assert cost.evidence_reserve_gpu_ns == 10


def test_two_gpu_parallel_physical_executions_use_global_provider_union(
    tmp_path: Path,
) -> None:
    gpu0 = _physical(
        tmp_path,
        label="gpu0",
        kind="resident_session",
        gpu_uuids=("GPU-0",),
        started=10,
        exited=30,
        empty=35,
        flushed=35,
    )
    gpu1 = _physical(
        tmp_path,
        label="gpu1",
        kind="resident_session",
        gpu_uuids=("GPU-1",),
        started=10,
        exited=30,
        empty=35,
        flushed=35,
    )
    cells = (
        _cell(
            tmp_path,
            label="gpu0",
            physical=gpu0,
            projection_source="resident_member_trace",
        ),
        _cell(
            tmp_path,
            label="gpu1",
            physical=gpu1,
            projection_source="resident_member_trace",
        ),
    )

    cost = gpu_hours._actual_cost_from_unified_observations(
        cells,
        (gpu0, gpu1),
        inventory_gpu_count=2,
    )

    assert cost.compute_gpu_ns == 40
    assert cost.wall_ns == 25
    assert cost.provider_base_reserved_gpu_ns == 50


def test_same_gpu_distinct_physical_execution_overlap_is_rejected(
    tmp_path: Path,
) -> None:
    first = _physical(
        tmp_path,
        label="first",
        kind="resident_session",
        gpu_uuids=("GPU-0",),
        started=10,
        exited=30,
        empty=32,
        flushed=32,
    )
    second = _physical(
        tmp_path,
        label="second",
        kind="fresh_process",
        gpu_uuids=("GPU-0",),
        started=20,
        exited=40,
        empty=42,
        flushed=42,
    )
    cells = (
        _cell(
            tmp_path,
            label="first",
            physical=first,
            projection_source="resident_member_trace",
        ),
        _cell(
            tmp_path,
            label="second",
            physical=second,
            projection_source="fresh_process",
        ),
    )

    with pytest.raises(
        gpu_hours.FormalSingleOperatorGpuHourBlocked,
        match="overlapping_physical_execution_on_gpu:GPU-0",
    ):
        gpu_hours._actual_cost_from_unified_observations(
            cells,
            (first, second),
            inventory_gpu_count=2,
        )


def test_same_gpu_launch_after_exit_but_before_group_empty_is_rejected(
    tmp_path: Path,
) -> None:
    first = _physical(
        tmp_path,
        label="ownership-first",
        kind="resident_session",
        gpu_uuids=("GPU-0",),
        started=10,
        exited=20,
        empty=30,
        flushed=35,
    )
    second = _physical(
        tmp_path,
        label="ownership-second",
        kind="fresh_process",
        gpu_uuids=("GPU-0",),
        started=25,
        exited=40,
        empty=42,
        flushed=42,
    )
    cells = (
        _cell(
            tmp_path,
            label="ownership-first",
            physical=first,
            projection_source="resident_member_trace",
        ),
        _cell(
            tmp_path,
            label="ownership-second",
            physical=second,
            projection_source="fresh_process",
        ),
    )

    with pytest.raises(
        gpu_hours.FormalSingleOperatorGpuHourBlocked,
        match="overlapping_physical_execution_on_gpu:GPU-0",
    ):
        gpu_hours._actual_cost_from_unified_observations(
            cells,
            (first, second),
            inventory_gpu_count=2,
        )


def test_fresh_and_resident_physical_costs_mix_without_aliasing(tmp_path: Path) -> None:
    resident = _physical(
        tmp_path,
        label="resident-mixed",
        kind="resident_session",
        gpu_uuids=("GPU-0",),
        started=10,
        exited=20,
        empty=22,
        flushed=22,
    )
    fresh = _physical(
        tmp_path,
        label="fresh-mixed",
        kind="fresh_process",
        gpu_uuids=("GPU-0",),
        started=22,
        exited=37,
        empty=40,
        flushed=40,
    )
    resident_cells = tuple(
        _cell(
            tmp_path,
            label=f"resident-mixed-{index}",
            physical=resident,
            projection_source="resident_member_trace",
            trace_ns=4 + index,
        )
        for index in range(3)
    )
    fresh_cell = _cell(
        tmp_path,
        label="fresh-mixed",
        physical=fresh,
        projection_source="fresh_process",
        trace_ns=15,
    )

    cost = gpu_hours._actual_cost_from_unified_observations(
        (*resident_cells, fresh_cell),
        (resident, resident, resident, fresh),
        inventory_gpu_count=2,
    )
    projections = gpu_hours._projection_observations_from_unified(
        (*resident_cells, fresh_cell)
    )

    assert cost.cell_count == 4
    assert cost.compute_gpu_ns == 25
    assert cost.wall_ns == 30
    assert cost.provider_base_reserved_gpu_ns == 60
    assert tuple(
        dict(row.phase_edges_ns)["process_exited_ns"]
        - dict(row.phase_edges_ns)["execution_started_ns"]
        for row in projections
    ) == (4, 5, 6, 15)


def test_evidence_tail_excludes_time_already_covered_by_core_union(
    tmp_path: Path,
) -> None:
    first = _physical(
        tmp_path,
        label="tail-first",
        kind="resident_session",
        gpu_uuids=("GPU-0",),
        started=10,
        exited=28,
        empty=30,
        flushed=70,
    )
    second = _physical(
        tmp_path,
        label="tail-second",
        kind="resident_session",
        gpu_uuids=("GPU-1",),
        started=40,
        exited=58,
        empty=60,
        flushed=80,
    )
    cells = (
        _cell(
            tmp_path,
            label="tail-first",
            physical=first,
            projection_source="resident_member_trace",
        ),
        _cell(
            tmp_path,
            label="tail-second",
            physical=second,
            projection_source="resident_member_trace",
        ),
    )

    cost = gpu_hours._actual_cost_from_unified_observations(
        cells,
        (first, second),
        inventory_gpu_count=2,
    )

    assert cost.wall_ns == 40
    assert cost.provider_base_reserved_gpu_ns == 80
    # Evidence [30, 70] and [60, 80] is unioned once, then the overlapping
    # second core interval [40, 60] is removed before inventory multiplication.
    assert cost.evidence_reserve_gpu_ns == 60


def test_typed_resident_manifest_projects_close_once_and_member_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_path, materialization, inventory = _typed_resident_manifest(
        tmp_path
    )
    from lightcone_spec.orchestration import formal_physical_dispatch as dispatch
    from lightcone_spec.orchestration import (
        formal_serving_session_group_physical as resident_physical,
    )

    trace_lifecycle = _binding(tmp_path, "typed-trace-lifecycle")
    trace = SimpleNamespace(
        materialized_cell_id=manifest.cell_id,
        trace_started_ns=manifest.trace_started_ns,
        trace_finished_ns=manifest.trace_finished_ns,
        trace_lifecycle=trace_lifecycle,
    )
    close = SimpleNamespace(
        member_trace_receipts=(manifest.resident_trace,),
        gpu_uuid="GPU-0",
        group_session_binding_sha256=manifest.group_session_binding_sha256,
        process_group_empty=True,
        server_process_started_ns=80,
        process_exited_ns=150,
        process_group_empty_checked_ns=155,
        evidence_flush_completed_ns=165,
    )
    monkeypatch.setattr(
        single,
        "revalidate_formal_single_operator_resident_run_manifest",
        lambda **_kwargs: manifest,
    )
    monkeypatch.setattr(
        resident_physical,
        "revalidate_formal_serving_resident_trace_receipt",
        lambda _path: (manifest.resident_trace, trace),
    )
    monkeypatch.setattr(
        resident_physical,
        "revalidate_formal_serving_resident_shared_close_receipt",
        lambda _path: (manifest.shared_close, close),
    )
    schedule = SimpleNamespace(sha256=manifest.request_schedule_sha256)
    monkeypatch.setattr(
        dispatch.FormalServingRequestScheduleReceipt,
        "from_dict",
        classmethod(lambda _cls, _value: schedule),
    )
    monkeypatch.setattr(
        dispatch,
        "formal_serving_request_schedule_rows",
        lambda _schedule: (SimpleNamespace(phase="scored"),),
    )

    cells, physical_rows = gpu_hours._unified_observations_from_actual_results(
        repository_root=tmp_path,
        actual_result_paths=(manifest_path,),
        materialization=materialization,
        inventory=inventory,
    )
    cell = cells[0]
    physical = physical_rows[0]
    source_path = (tmp_path / "typed-resident-gpu-hour-source.json").resolve()
    publish_canonical_json_no_replace(
        source_path,
        gpu_hours._unified_source_value(
            materialization=materialization,
            inventory=inventory,
            cells=cells,
            physical=physical_rows,
        ),
    )
    source = gpu_hours.revalidate_formal_single_operator_serving_gpu_hour_source(
        repository_root=tmp_path,
        source_manifest_path=source_path,
        materialization=materialization,
        inventory=inventory,
    )

    assert type(manifest) is single.FormalSingleOperatorResidentRunManifest
    assert len(cells) == len(physical_rows) == 1
    assert source == CanonicalJsonProofBinding.bind(source_path)
    assert cell.actual_result == CanonicalJsonProofBinding.bind(manifest_path)
    assert cell.physical_execution_id == manifest.shared_close.semantic_sha256
    assert cell.projection_source == "resident_member_trace"
    assert cell.projection_process_ns == 30
    assert cell.member_lifecycle == trace_lifecycle
    assert physical.execution_kind == "resident_session"
    assert physical.source == manifest.shared_close
    assert dict(physical.phase_edges_ns) == {
        "server_process_started_ns": 80,
        "process_exited_ns": 150,
        "process_group_empty_checked_ns": 155,
        "evidence_flush_completed_ns": 165,
    }
