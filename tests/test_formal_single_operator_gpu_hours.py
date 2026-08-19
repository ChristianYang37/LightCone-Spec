from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_formal_dispatch import _protocol_lock
from test_formal_physical_dispatch import (
    _run_distributed_operator_fixture,
    _run_tp1_operator_fixture,
)
from test_formal_single_operator import _publish_manifest, _source_repository
from test_gpu_hour_authority import _case, _inventory, _runtime_manifest

from lightcone_spec.config import run_config_sha256
from lightcone_spec.experiments import gpu_hour_authority
from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_single_operator_gpu_hours import (
    FormalSingleOperatorGpuHourBlocked,
    FormalSingleOperatorPostPilotGpuHours,
    FormalSingleOperatorPrePilotGpuHours,
    derive_formal_single_operator_post_pilot_gpu_hours,
    derive_formal_single_operator_post_pilot_gpu_hours_from_run_manifests,
    derive_formal_single_operator_pre_pilot_gpu_hours,
    load_formal_single_operator_gpu_hours,
    publish_formal_single_operator_gpu_hours,
    revalidate_formal_single_operator_run_gpu_hour_source,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    StageMaterializationReceipt,
)
from lightcone_spec.orchestration.formal_physical_dispatch import (
    FormalServingRequestScheduleReceipt,
)
from lightcone_spec.orchestration.formal_terminal_result import (
    validate_formal_distributed_physical_outcome,
)
from lightcone_spec.orchestration.live_sglang import (
    UnsignedPinnedSglangLifecycleTimingReceipt,
    UnsignedPinnedSglangServingRunReceipt,
)
from lightcone_spec.runtime import formal_single_operator as single
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_NS_PER_HOUR = 3_600_000_000_000


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _cell(*, stage: str, block: int | None, phase: str | None) -> MaterializedCell:
    dimensions: dict[str, object] = {"topology": "tp1_dp1"}
    if block is not None:
        dimensions.update({"block": block, "block_phase": phase})
    return MaterializedCell(
        stage=stage,
        method_role="Static",
        model="Qwen/Qwen3-8B",
        backend="DFlash",
        task="production_slo_power_prefix"
        if stage in {"E3b", "E5"}
        else "LiveCodeBench",
        publication_policy="none",
        recipe_sha256=None,
        dimensions=tuple(sorted(dimensions.items())),
    )


def _materialization(
    *,
    stage: str,
    lock_sha256: str,
    cells: tuple[MaterializedCell, ...],
    rule: str,
) -> StageMaterializationReceipt:
    ordered = tuple(sorted(cells, key=lambda row: row.cell_id))
    return StageMaterializationReceipt(
        schema_version=1,
        stage=stage,
        protocol_lock_sha256=lock_sha256,
        upstream_receipt_sha256s=(
            () if stage == "preflight" else (_sha(f"{stage}:upstream"),)
        ),
        source_decision_sha256=_sha(f"{stage}:source"),
        materialization_rule=rule,
        expected_cell_count=len(ordered),
        cells=ordered,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _early_materialization(lock_sha256: str) -> StageMaterializationReceipt:
    cells = tuple(
        MaterializedCell(
            stage="E3a",
            method_role="Static",
            model="Qwen/Qwen3-8B",
            backend="DFlash",
            task="LiveCodeBench",
            publication_policy="none",
            recipe_sha256=None,
            dimensions=(("registry_cell_id", _sha(f"registry:{index}")),),
        )
        for index in range(3)
    )
    return _materialization(
        stage="E3a",
        lock_sha256=lock_sha256,
        cells=cells,
        rule="exact_360_row_capacity_width_and_drift_grid",
    )


def _downstream_materializations(
    *,
    stage: str,
    lock_sha256: str,
    include_e5_failures: bool = False,
) -> tuple[StageMaterializationReceipt, StageMaterializationReceipt]:
    pilot = _materialization(
        stage=stage,
        lock_sha256=lock_sha256,
        cells=tuple(
            _cell(stage=stage, block=block, phase="excluded_pilot")
            for block in range(4)
        ),
        rule=f"test_{stage.lower()}_four_excluded_pilots",
    )
    final_cells = [
        _cell(stage=stage, block=block, phase="final") for block in range(4, 16)
    ]
    if include_e5_failures:
        final_cells.extend(
            MaterializedCell(
                stage="E5",
                method_role="LightCone",
                model="Qwen/Qwen3-8B",
                backend="DFlash",
                task="deterministic_failure_injection",
                publication_policy="diagnostic_only",
                recipe_sha256=_sha("recipe"),
                dimensions=(("failure_case", index),),
            )
            for index in range(264)
        )
    final = _materialization(
        stage=stage,
        lock_sha256=lock_sha256,
        cells=tuple(final_cells),
        rule=f"test_{stage.lower()}_twelve_final_blocks",
    )
    return pilot, final


def _real_tp1_single_operator_manifest(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    StageMaterializationReceipt,
    GpuInventory,
    UnsignedPinnedSglangLifecycleTimingReceipt,
]:
    physical_root = tmp_path / "physical"
    physical_root.mkdir()
    plan, _plan_path, verified, inventory_path, _admission, _result = (
        _run_tp1_operator_fixture(monkeypatch, physical_root)
    )
    repository, git_head, git_tree, patch_sha256 = _source_repository(
        tmp_path / "repository"
    )
    run_root = Path(plan.private_output_root)
    manifest_path = run_root / "formal-single-operator-manifest.json"
    inventory = GpuInventory.from_dict(
        CanonicalJsonProofBinding.bind(inventory_path).reopen()
    )
    schedule = FormalServingRequestScheduleReceipt.from_dict(
        plan.request_schedule_receipt.reopen()
    )
    materialization = stage_materialization_receipt_from_dict(
        schedule.materialization.reopen()
    )
    launch = CompileLaunchManifest.from_dict(plan.launch_manifest.reopen())
    config = verified.run_config
    live = UnsignedPinnedSglangServingRunReceipt.from_dict(
        CanonicalJsonProofBinding.bind(plan.live_run_receipt_output_path).reopen()
    )
    lifecycle = UnsignedPinnedSglangLifecycleTimingReceipt.from_dict(
        CanonicalJsonProofBinding.bind(plan.lifecycle_timing_output_path).reopen()
    )
    cell = materialization.cells[0]
    devices = {row.uuid: row for row in inventory.devices}
    artifact_paths = {
        "after_gpu_snapshot": Path(plan.after_gpu_snapshot_output_path),
        "before_gpu_snapshot": Path(plan.before_gpu_snapshot_output_path),
        "junit": Path(plan.junit_output_path),
        "lifecycle": Path(plan.lifecycle_timing_output_path),
        "live_run_receipt": Path(plan.live_run_receipt_output_path),
        "native_itl": Path(plan.native_itl_pointer_output_path),
        "raw_terminal": Path(plan.terminal_output_path),
        "ready_gpu_snapshot": Path(plan.ready_gpu_snapshot_output_path),
        "request_schedule": Path(plan.request_schedule_receipt.absolute_path),
        "run_plan": Path(_plan_path),
        "server_log": Path(plan.server_log_output_path),
        "stderr": Path(plan.server_stderr_output_path),
        "stdout": Path(plan.server_stdout_output_path),
    }
    artifacts = tuple(
        sorted(
            (
                single.FormalSingleOperatorArtifact.observe(
                    name=name,
                    run_root=run_root,
                    path=path,
                )
                for name, path in artifact_paths.items()
            ),
            key=lambda row: row.name,
        )
    )
    manifest = single.FormalSingleOperatorRunManifest(
        schema=single.FORMAL_SINGLE_OPERATOR_MODE,
        protocol_sha256=single.FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256,
        trust_assumptions=single.FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        git_head=git_head,
        git_tree=git_tree,
        sglang_upstream_commit="6" * 40,
        patch_manifest_sha256=patch_sha256,
        patched_sglang_tree="8" * 40,
        registry_sha256=build_industrial_registry().sha256,
        physical_dispatch_protocol_sha256=plan.protocol_sha256,
        run_plan_sha256=plan.sha256,
        launch_manifest_sha256=launch.sha256,
        execution_binding_sha256=plan.execution_binding_sha256,
        execution_subject_sha256=plan.subject_sha256,
        materialization_protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_sha256=materialization.sha256,
        inventory_sha256=inventory.sha256,
        run_config_sha256=run_config_sha256(config),
        run_config=config.model_dump(mode="json"),
        launch_argv_sha256=launch.server_argv_sha256,
        launch_argv=launch.server_argv,
        localhost_port=launch.localhost_port,
        request_schedule_sha256=schedule.sha256,
        request_schedule=schedule.to_dict(),
        target_model_id=launch.target_model_id,
        target_revision=launch.target_revision,
        target_content_sha256=launch.target_content_authority_sha256,
        drafter_model_id=launch.drafter_model_id,
        drafter_revision=launch.drafter_revision,
        drafter_content_sha256=launch.drafter_content_authority_sha256,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_content_sha256=launch.tokenizer_content_authority_sha256,
        workload_artifact_id=schedule.workload_source.artifact_id,
        workload_authority_sha256=schedule.workload_authority_sha256,
        workload_member_sha256s=tuple(
            sorted({row.source_member_sha256 for row in schedule.requests})
        ),
        workload_raw_sha256=schedule.workload_source.raw_sha256,
        workload_semantic_sha256=schedule.workload_source.semantic_sha256,
        stage=cell.stage,
        cell_id=cell.cell_id,
        role=cell.method_role,
        backend=cell.backend,
        topology=plan.topology_mode,
        block=None,
        attempt=plan.native_terminal_binding.attempt_id,
        run_directory=str(run_root),
        gpu_environment=tuple(
            single.FormalSingleOperatorGpu(
                uuid=uuid,
                model=devices[uuid].model,
                driver_version="test-driver",
                cuda_version="test-cuda",
            )
            for uuid in sorted(plan.gpu_uuids)
        ),
        started_ns=live.execution_started_ns,
        finished_ns=live.process_group_empty_checked_ns,
        exit_code=live.process_exit_code,
        completion_status="COMPLETE",
        failure_reason=None,
        artifacts=artifacts,
    )
    _publish_manifest(manifest_path, manifest)
    return repository, manifest_path, materialization, inventory, lifecycle


def _real_distributed_single_operator_manifest(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    topology: str,
) -> tuple[
    Path,
    Path,
    StageMaterializationReceipt,
    GpuInventory,
    dict[str, int],
]:
    physical_root = tmp_path / "physical"
    physical_root.mkdir()
    plan, result, _admission_binding, inventory_path, verified = (
        _run_distributed_operator_fixture(
            monkeypatch,
            physical_root,
            topology=topology,
        )
    )
    repository, git_head, git_tree, patch_sha256 = _source_repository(
        tmp_path / "repository"
    )
    run_root = Path(plan.private_output_root)
    plan_path = run_root / "formal-serving-run-plan.json"
    manifest_path = run_root / "formal-single-operator-manifest.json"
    inventory = GpuInventory.from_dict(
        CanonicalJsonProofBinding.bind(inventory_path).reopen()
    )
    schedule = FormalServingRequestScheduleReceipt.from_dict(
        plan.request_schedule_receipt.reopen()
    )
    materialization = stage_materialization_receipt_from_dict(
        schedule.materialization.reopen()
    )
    launch = CompileLaunchManifest.from_dict(plan.launch_manifest.reopen())
    config = verified.run_config
    registry_sha256 = build_industrial_registry().sha256
    outcome = validate_formal_distributed_physical_outcome(
        plan_path=str(plan_path),
        run_receipt_path=result.receipt.absolute_path,
        expected_inventory_sha256=inventory.sha256,
        expected_registry_sha256=registry_sha256,
    )
    raw_lifecycle = CanonicalJsonProofBinding.bind(
        plan.lifecycle_timing_output_path
    ).reopen()
    assert type(raw_lifecycle) is dict
    raw_edges = raw_lifecycle["phase_edges_ns"]
    assert type(raw_edges) is dict
    edges = {str(name): int(value) for name, value in raw_edges.items()}
    cell = materialization.cells[0]
    devices = {row.uuid: row for row in inventory.devices}
    gang_terminal_path = plan.formal_gang_terminal_output_path
    assert gang_terminal_path is not None
    artifact_paths = {
        "after_gpu_snapshot": Path(plan.after_gpu_snapshot_output_path),
        "before_gpu_snapshot": Path(plan.before_gpu_snapshot_output_path),
        "formal_gang_terminal": Path(gang_terminal_path),
        "junit": Path(plan.junit_output_path),
        "lifecycle": Path(plan.lifecycle_timing_output_path),
        "live_run_receipt": Path(plan.live_run_receipt_output_path),
        "native_itl": Path(plan.native_itl_pointer_output_path),
        "raw_terminal": Path(plan.terminal_output_path),
        "ready_gpu_snapshot": Path(plan.ready_gpu_snapshot_output_path),
        "request_schedule": Path(plan.request_schedule_receipt.absolute_path),
        "run_plan": plan_path,
        "server_log": Path(plan.server_log_output_path),
        "stderr": Path(plan.server_stderr_output_path),
        "stdout": Path(plan.server_stdout_output_path),
    }
    artifacts = tuple(
        sorted(
            (
                single.FormalSingleOperatorArtifact.observe(
                    name=name,
                    run_root=run_root,
                    path=path,
                )
                for name, path in artifact_paths.items()
            ),
            key=lambda row: row.name,
        )
    )
    manifest = single.FormalSingleOperatorRunManifest(
        schema=single.FORMAL_SINGLE_OPERATOR_MODE,
        protocol_sha256=single.FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256,
        trust_assumptions=single.FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        git_head=git_head,
        git_tree=git_tree,
        sglang_upstream_commit="6" * 40,
        patch_manifest_sha256=patch_sha256,
        patched_sglang_tree="8" * 40,
        registry_sha256=registry_sha256,
        physical_dispatch_protocol_sha256=plan.protocol_sha256,
        run_plan_sha256=plan.sha256,
        launch_manifest_sha256=launch.sha256,
        execution_binding_sha256=plan.execution_binding_sha256,
        execution_subject_sha256=plan.subject_sha256,
        materialization_protocol_lock_sha256=(materialization.protocol_lock_sha256),
        materialization_sha256=materialization.sha256,
        inventory_sha256=inventory.sha256,
        run_config_sha256=run_config_sha256(config),
        run_config=config.model_dump(mode="json"),
        launch_argv_sha256=launch.server_argv_sha256,
        launch_argv=launch.server_argv,
        localhost_port=launch.localhost_port,
        request_schedule_sha256=schedule.sha256,
        request_schedule=schedule.to_dict(),
        target_model_id=launch.target_model_id,
        target_revision=launch.target_revision,
        target_content_sha256=launch.target_content_authority_sha256,
        drafter_model_id=launch.drafter_model_id,
        drafter_revision=launch.drafter_revision,
        drafter_content_sha256=launch.drafter_content_authority_sha256,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_content_sha256=launch.tokenizer_content_authority_sha256,
        workload_artifact_id=schedule.workload_source.artifact_id,
        workload_authority_sha256=schedule.workload_authority_sha256,
        workload_member_sha256s=tuple(
            sorted({row.source_member_sha256 for row in schedule.requests})
        ),
        workload_raw_sha256=schedule.workload_source.raw_sha256,
        workload_semantic_sha256=schedule.workload_source.semantic_sha256,
        stage=cell.stage,
        cell_id=cell.cell_id,
        role=cell.method_role,
        backend=cell.backend,
        topology=plan.topology_mode,
        block=None,
        attempt=plan.native_terminal_binding.attempt_id,
        run_directory=str(run_root),
        gpu_environment=tuple(
            single.FormalSingleOperatorGpu(
                uuid=uuid,
                model=devices[uuid].model,
                driver_version="test-driver",
                cuda_version="test-cuda",
            )
            for uuid in sorted(plan.gpu_uuids)
        ),
        started_ns=outcome.execution_started_ns,
        finished_ns=outcome.process_group_empty_checked_ns,
        exit_code=outcome.process_exit_code,
        completion_status="COMPLETE",
        failure_reason=None,
        artifacts=artifacts,
    )
    _publish_manifest(manifest_path, manifest)
    return repository, manifest_path, materialization, inventory, edges


def test_pre_pilot_reports_counts_and_never_accepts_duration_scalars(
    tmp_path: Path,
) -> None:
    materialization = _early_materialization(_sha("lock"))
    output = derive_formal_single_operator_pre_pilot_gpu_hours(materialization)

    assert tuple(
        inspect.signature(derive_formal_single_operator_pre_pilot_gpu_hours).parameters
    ) == ("materialization",)
    assert output.duration_status == "duration_unmeasured"
    assert output.fixed_cell_count == 3
    assert output.projection_stratum_count == 1
    assert output.minimum_pilot_cell_ids == (materialization.cells[0].cell_id,)
    encoded = output.to_dict()
    assert "compute_gpu_hours" not in encoded
    assert "reserved_gpu_hours" not in encoded
    assert FormalSingleOperatorPrePilotGpuHours.from_dict(encoded) == output

    path = (tmp_path / "pre-pilot.json").resolve()
    publish_formal_single_operator_gpu_hours(output, path)
    assert load_formal_single_operator_gpu_hours(path) == output
    with pytest.raises(RuntimeError, match="target already exists"):
        publish_formal_single_operator_gpu_hours(output, path)


def test_pre_pilot_all_na_e0_reports_zero_work_without_fabricating_hours() -> None:
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E0",
        protocol_lock_sha256=_sha("lock"),
        upstream_receipt_sha256s=(_sha("e6"),),
        source_decision_sha256=_sha("all-na"),
        materialization_rule="all_108_compatibility_decisions_are_N_A",
        expected_cell_count=0,
        cells=(),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    output = derive_formal_single_operator_pre_pilot_gpu_hours(materialization)
    assert output.fixed_cell_count == 0
    assert output.projection_stratum_count == 0
    assert output.minimum_pilot_cell_ids == ()
    assert output.duration_status == "duration_unmeasured"


def test_pre_pilot_requires_a_pilot_not_a_final_materialization() -> None:
    pilot, final = _downstream_materializations(
        stage="E3b",
        lock_sha256=_sha("lock"),
    )
    pilot = replace(
        pilot,
        materialization_rule="e3b_exact_480_rows_x_4_excluded_pilot_blocks",
    )
    plan = derive_formal_single_operator_pre_pilot_gpu_hours(pilot)
    assert plan.fixed_cell_count == 4
    assert plan.minimum_pilot_cell_ids == tuple(cell.cell_id for cell in pilot.cells)
    with pytest.raises(
        FormalSingleOperatorGpuHourBlocked,
        match="pre_pilot_plan_requires_current_pilot_materialization",
    ):
        derive_formal_single_operator_pre_pilot_gpu_hours(final)


def test_post_pilot_uses_actual_lifecycle_and_same_stratum_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime.sha256,
    )
    inventory = _inventory()
    materialization = _early_materialization(lock.sha256)
    (
        _lock,
        _runtime,
        _inventory_value,
        _materialization_value,
        proof_inputs,
        _verified,
        _full_source,
        _full_envelope,
    ) = _case(
        tmp_path / "early",
        gangs=(("GPU-0",), ("GPU-0",), ("GPU-0",)),
        starts=(1_000_000_000, 3 * _NS_PER_HOUR, 5 * _NS_PER_HOUR),
        monkeypatch=monkeypatch,
        lock_override=lock,
        runtime_manifest_override=runtime,
        inventory_override=inventory,
        materialization_override=materialization,
    )
    subset_path = (tmp_path / "one-pilot-source.json").resolve()
    gpu_hour_authority.materialize_lifecycle_gpu_hour_subset_source(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime,
        materialization=materialization,
        inventory=inventory,
        expected_cell_ids=(materialization.cells[0].cell_id,),
        proof_inputs=(proof_inputs[0],),
        source_manifest_output_path=str(subset_path),
        now_ns=2_000_000_000,
    )

    output = derive_formal_single_operator_post_pilot_gpu_hours(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime,
        pilot_materialization=materialization,
        inventory=inventory,
        pilot_lifecycle_source_manifest_path=subset_path,
        now_ns=2_000_000_001,
    )
    assert output.pilot_materialization_rule == materialization.materialization_rule
    assert output.final_materialization_rule is None
    assert output.actual_pilot.cell_count == 1
    assert output.projected_remaining.cell_count == 2
    assert output.total.cell_count == 3
    assert output.actual_pilot.compute_gpu_hours == pytest.approx(1.0)
    assert output.projected_remaining.compute_gpu_hours == pytest.approx(2.0)
    assert output.total.compute_gpu_hours == pytest.approx(3.0)
    assert output.actual_one_shot is None
    assert FormalSingleOperatorPostPilotGpuHours.from_dict(output.to_dict()) == output


def test_downstream_projection_uses_four_actual_pilots_without_signed_power(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime.sha256,
    )
    inventory = _inventory()
    pilot, final = _downstream_materializations(
        stage="E3b",
        lock_sha256=lock.sha256,
    )
    *_, source_path, _envelope = _case(
        tmp_path / "e3b",
        gangs=(("GPU-0",),) * 4,
        starts=tuple(1_000_000_000 + index * 2 * _NS_PER_HOUR for index in range(4)),
        monkeypatch=monkeypatch,
        lock_override=lock,
        runtime_manifest_override=runtime,
        inventory_override=inventory,
        materialization_override=pilot,
    )
    output = derive_formal_single_operator_post_pilot_gpu_hours(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime,
        pilot_materialization=pilot,
        final_materialization=final,
        inventory=inventory,
        pilot_lifecycle_source_manifest_path=source_path,
        now_ns=2_000_000_001,
    )
    assert output.pilot_materialization_rule == pilot.materialization_rule
    assert output.final_materialization_rule == final.materialization_rule

    assert output.actual_pilot.cell_count == 4
    assert output.projected_remaining.cell_count == 12
    assert output.total.cell_count == 16
    assert output.total.compute_gpu_hours == pytest.approx(16.0)
    assert output.one_shot_lifecycle_source is None


def test_e5_exact_264_actual_one_shots_are_added_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime.sha256,
    )
    inventory = _inventory()
    pilot, final = _downstream_materializations(
        stage="E5",
        lock_sha256=lock.sha256,
        include_e5_failures=True,
    )
    *_, source_path, _envelope = _case(
        tmp_path / "e5",
        gangs=(("GPU-0",),) * 4,
        starts=tuple(1_000_000_000 + index * 2 * _NS_PER_HOUR for index in range(4)),
        monkeypatch=monkeypatch,
        lock_override=lock,
        runtime_manifest_override=runtime,
        inventory_override=inventory,
        materialization_override=pilot,
    )
    with pytest.raises(
        FormalSingleOperatorGpuHourBlocked,
        match="e5_264_actual_lifecycle_source_missing",
    ):
        derive_formal_single_operator_post_pilot_gpu_hours(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime,
            pilot_materialization=pilot,
            final_materialization=final,
            inventory=inventory,
            pilot_lifecycle_source_manifest_path=source_path,
            now_ns=2_000_000_001,
        )

    one_shot_path = (tmp_path / "e5-one-shot.json").resolve()
    publish_canonical_json_no_replace(
        one_shot_path,
        {"kind": "test-dedicated-e5-integrated-failure-source"},
    )
    pilot_source = gpu_hour_authority.LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(source_path).reopen()
    )
    one_shot_cost = gpu_hour_authority.ProspectiveGpuHourCost(
        category="actual_one_shot",
        cell_count=264,
        compute_gpu_ns=264 * _NS_PER_HOUR,
        provider_base_reserved_gpu_ns=528 * _NS_PER_HOUR,
        wall_ns=264 * _NS_PER_HOUR,
        retry_reserve_gpu_ns=0,
        profile_reserve_gpu_ns=0,
        evidence_reserve_gpu_ns=0,
    )
    monkeypatch.setattr(
        gpu_hour_authority,
        "revalidate_persisted_e5_failure_gpu_hour_source_manifest",
        lambda *_args, **_kwargs: SimpleNamespace(
            cost=one_shot_cost,
            hardware_envelope_sha256=pilot_source.hardware_envelope_sha256,
        ),
    )
    output = derive_formal_single_operator_post_pilot_gpu_hours(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime,
        pilot_materialization=pilot,
        final_materialization=final,
        inventory=inventory,
        pilot_lifecycle_source_manifest_path=source_path,
        e5_one_shot_source_manifest_path=one_shot_path,
        now_ns=2_000_000_002,
    )

    assert output.actual_pilot.cell_count == 4
    assert output.projected_remaining.cell_count == 12
    assert output.actual_one_shot is not None
    assert output.actual_one_shot.cell_count == 264
    assert output.total.cell_count == 280
    assert output.total.compute_gpu_hours == pytest.approx(280.0)
    assert output.total.compute_gpu_ns == (
        output.actual_pilot.compute_gpu_ns
        + output.projected_remaining.compute_gpu_ns
        + output.actual_one_shot.compute_gpu_ns
    )


def test_e5_cannot_reuse_pilot_source_as_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime.sha256,
    )
    inventory = _inventory()
    pilot, final = _downstream_materializations(
        stage="E5",
        lock_sha256=lock.sha256,
        include_e5_failures=True,
    )
    *_, source_path, _envelope = _case(
        tmp_path / "e5-reuse",
        gangs=(("GPU-0",),) * 4,
        starts=tuple(1_000_000_000 + index * 2 * _NS_PER_HOUR for index in range(4)),
        monkeypatch=monkeypatch,
        lock_override=lock,
        runtime_manifest_override=runtime,
        inventory_override=inventory,
        materialization_override=pilot,
    )
    with pytest.raises(ValueError, match="cannot double as one-shot"):
        derive_formal_single_operator_post_pilot_gpu_hours(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime,
            pilot_materialization=pilot,
            final_materialization=final,
            inventory=inventory,
            pilot_lifecycle_source_manifest_path=source_path,
            e5_one_shot_source_manifest_path=source_path,
            now_ns=2_000_000_001,
        )


def test_e6_model_preflight_cost_remains_duration_unmeasured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime.sha256,
    )
    inventory = _inventory()
    pilot, final = _downstream_materializations(
        stage="E6",
        lock_sha256=lock.sha256,
    )
    *_, source_path, _envelope = _case(
        tmp_path / "e6",
        gangs=(("GPU-0",),) * 4,
        starts=tuple(1_000_000_000 + index * 2 * _NS_PER_HOUR for index in range(4)),
        monkeypatch=monkeypatch,
        lock_override=lock,
        runtime_manifest_override=runtime,
        inventory_override=inventory,
        materialization_override=pilot,
    )

    with pytest.raises(
        FormalSingleOperatorGpuHourBlocked,
        match="e6_model_preflight_lifecycle_cost_unavailable",
    ):
        derive_formal_single_operator_post_pilot_gpu_hours(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime,
            pilot_materialization=pilot,
            final_materialization=final,
            inventory=inventory,
            pilot_lifecycle_source_manifest_path=source_path,
            now_ns=2_000_000_001,
        )


def test_run_manifest_bridge_bills_root_revalidated_lifecycle_without_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, manifest_path, materialization, inventory, lifecycle = (
        _real_tp1_single_operator_manifest(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
        )
    )
    source_path = (tmp_path / "single-operator-gpu-hour-source.json").resolve()
    output = derive_formal_single_operator_post_pilot_gpu_hours_from_run_manifests(
        repository_root=repository,
        pilot_materialization=materialization,
        inventory=inventory,
        pilot_run_manifest_paths=(manifest_path,),
        source_manifest_output_path=source_path,
    )

    parameters = set(
        inspect.signature(
            derive_formal_single_operator_post_pilot_gpu_hours_from_run_manifests
        ).parameters
    )
    assert {
        "duration_ns",
        "gpu_hours",
        "compute_gpu_hours",
        "complete_status",
        "terminal_sha256",
        "process_exited_ns",
    }.isdisjoint(parameters)
    edges = lifecycle.phase_edges_ns
    process_ns = edges["process_exited_ns"] - edges["execution_started_ns"]
    core_wall_ns = (
        edges["process_group_empty_checked_ns"] - edges["execution_started_ns"]
    )
    evidence_tail_ns = (
        edges["evidence_flush_finished_ns"] - edges["process_group_empty_checked_ns"]
    )
    assert output.actual_pilot.cell_count == 1
    assert output.projected_remaining.cell_count == 0
    assert output.actual_pilot.compute_gpu_ns == process_ns
    assert output.actual_pilot.provider_base_reserved_gpu_ns == core_wall_ns * 2
    assert output.actual_pilot.evidence_reserve_gpu_ns == evidence_tail_ns * 2
    assert output.total == replace(output.actual_pilot, category="total")
    source_value = output.pilot_lifecycle_source.reopen()
    observations = source_value["observations"]
    assert type(observations) is list and len(observations) == 1
    assert {
        "admission",
        "admission_consumption",
        "terminal",
        "native_itl",
        "before_gpu_snapshot",
        "ready_gpu_snapshot",
        "after_gpu_snapshot",
    }.isdisjoint(observations[0])
    root_manifest = single.FormalSingleOperatorRunManifest.from_dict(
        CanonicalJsonProofBinding.bind(manifest_path).reopen()
    )
    assert {"admission", "admission_consumption"}.isdisjoint(
        row.name for row in root_manifest.artifacts
    )
    assert (
        revalidate_formal_single_operator_run_gpu_hour_source(
            repository_root=repository,
            source_manifest_path=source_path,
            materialization=materialization,
            inventory=inventory,
        )
        == output.pilot_lifecycle_source
    )

    foreign_cell = replace(
        materialization.cells[0],
        dimensions=tuple(
            sorted((*materialization.cells[0].dimensions, ("foreign", "yes")))
        ),
    )
    foreign = replace(materialization, cells=(foreign_cell,))
    with pytest.raises(ValueError, match="foreign stage lineage"):
        derive_formal_single_operator_post_pilot_gpu_hours_from_run_manifests(
            repository_root=repository,
            pilot_materialization=foreign,
            inventory=inventory,
            pilot_run_manifest_paths=(manifest_path,),
            source_manifest_output_path=(tmp_path / "foreign-source.json").resolve(),
        )

    lifecycle_path = Path(
        next(
            row.relative_path
            for row in single.FormalSingleOperatorRunManifest.from_dict(
                CanonicalJsonProofBinding.bind(manifest_path).reopen()
            ).artifacts
            if row.name == "lifecycle"
        )
    )
    lifecycle_path = manifest_path.parent / lifecycle_path
    lifecycle_path.write_bytes(lifecycle_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="output changed after the run"):
        revalidate_formal_single_operator_run_gpu_hour_source(
            repository_root=repository,
            source_manifest_path=source_path,
            materialization=materialization,
            inventory=inventory,
        )


@pytest.mark.parametrize("topology", ("tp2_dp1", "tp1_dp2"))
def test_run_manifest_bridge_revalidates_distributed_tp2_and_dp2_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
) -> None:
    repository, manifest_path, materialization, inventory, edges = (
        _real_distributed_single_operator_manifest(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            topology=topology,
        )
    )
    source_path = (tmp_path / f"{topology}-gpu-hour-source.json").resolve()
    output = derive_formal_single_operator_post_pilot_gpu_hours_from_run_manifests(
        repository_root=repository,
        pilot_materialization=materialization,
        inventory=inventory,
        pilot_run_manifest_paths=(manifest_path,),
        source_manifest_output_path=source_path,
    )

    process_ns = edges["process_exited_ns"] - edges["execution_started_ns"]
    core_wall_ns = (
        edges["process_group_empty_checked_ns"] - edges["execution_started_ns"]
    )
    evidence_tail_ns = (
        edges["evidence_flush_finished_ns"] - edges["process_group_empty_checked_ns"]
    )
    assert output.actual_pilot.cell_count == 1
    assert output.projected_remaining.cell_count == 0
    assert output.actual_pilot.compute_gpu_ns == process_ns * 2
    assert output.actual_pilot.provider_base_reserved_gpu_ns == core_wall_ns * 2
    assert output.actual_pilot.evidence_reserve_gpu_ns == evidence_tail_ns * 2
    source_value = output.pilot_lifecycle_source.reopen()
    observations = source_value["observations"]
    assert type(observations) is list and len(observations) == 1
    assert observations[0]["topology"] == topology
    assert observations[0]["gang_gpu_count"] == 2
    assert observations[0]["provider_reserved_gpu_count"] == 2
    assert (
        revalidate_formal_single_operator_run_gpu_hour_source(
            repository_root=repository,
            source_manifest_path=source_path,
            materialization=materialization,
            inventory=inventory,
        )
        == output.pilot_lifecycle_source
    )


@pytest.mark.parametrize(
    ("stage", "rule", "reason"),
    (
        (
            "E4",
            "three_profiler_only_rows_separate_from_headline",
            "e4_profiler_requires_dedicated_profiler_lifecycle_manifest",
        ),
        (
            "E5",
            "e5_exact_450_rows_x_4_excluded_pilot_blocks",
            "e5_dedicated_failure_run_manifest_union_required",
        ),
        (
            "E6",
            "e6_exact_two_model_preflights_plus_60_rows_x_4_excluded_pilot_blocks",
            "e6_model_preflight_lifecycle_cost_unavailable",
        ),
    ),
)
def test_run_manifest_bridge_rejects_ordinary_serving_for_dedicated_phases(
    tmp_path: Path,
    stage: str,
    rule: str,
    reason: str,
) -> None:
    materialization = _materialization(
        stage=stage,
        lock_sha256=_sha("lock"),
        cells=(_cell(stage=stage, block=0, phase="excluded_pilot"),),
        rule=rule,
    )
    with pytest.raises(FormalSingleOperatorGpuHourBlocked, match=reason):
        derive_formal_single_operator_post_pilot_gpu_hours_from_run_manifests(
            repository_root=tmp_path,
            pilot_materialization=materialization,
            inventory=_inventory(),
            pilot_run_manifest_paths=(),
            source_manifest_output_path=(tmp_path / f"{stage}.json").resolve(),
        )
