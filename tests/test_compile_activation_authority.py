from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.experiments.compile_activation_authority import (
    COMPILE_DIAGNOSTIC_ACTIVATION_PROTOCOL_SHA256,
    CompileDiagnosticActivationAuthority,
    materialize_compile_diagnostic_activation,
    verify_compile_diagnostic_activation,
)
from lightcone_spec.experiments.registry import (
    WorkloadClass,
    build_industrial_registry,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.stage_activation import (
    RegistryStageDispositionStatus,
    materialize_registry_stage_activation,
)
from lightcone_spec.runtime.compile_cache import (
    COMPILE_ONLY_ASSIGNMENT_PROTOCOL_SHA256,
    COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256,
    COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    PINNED_SGLANG_PATCH_MANIFEST_SHA256,
    PINNED_SGLANG_PATCH_SHA256,
    SGLANG_FIRST_PARTY_COMPILE_BUILDER,
    CompileCacheKey,
    CompileCacheLaunchPlan,
    CompileOnlyAssignmentContract,
    CompileOnlyPrewarmManifest,
    CompileOnlyPrewarmPayload,
)
from lightcone_spec.runtime.compile_runner import (
    COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    RELEASE_COMPILE_RUNNER_UNAVAILABLE,
    CompileAssignmentPlan,
    CompileLaunchManifest,
    CompileRunnerBlocked,
    require_release_compile_assignment_plan,
    write_compile_prewarm_manifest,
)


def _sha(label: str) -> str:
    return hashlib.sha256(f"compile-activation:{label}".encode()).hexdigest()


def _write_plan(
    tmp_path: Path,
    *,
    registry_sha256: str,
    cell_id: str,
    gpu_uuids: tuple[str, ...],
    runtime_sha256: str,
    split_sha256: str,
) -> Path:
    key = CompileCacheKey(
        patched_sglang_tree=PINNED_SGLANG_TREE,
        patch_manifest_sha256=PINNED_SGLANG_PATCH_MANIFEST_SHA256,
        patch_sha256=PINNED_SGLANG_PATCH_SHA256,
        source_sha256=PINNED_SGLANG_COMPILE_SOURCE_SHA256,
        python_version="3.12.11",
        torch_version="2.11.0+cu130",
        triton_version="3.6.0",
        cuda_version="13.0",
        driver_version="580.65.06",
        sm_architecture="sm_120",
        gpu_model="RTX PRO 6000 Blackwell Server Edition",
        dtype="bfloat16",
        target_revision="a" * 40,
        drafter_revision=None,
        tensor_parallel_size=len(gpu_uuids),
        context_limit=4096,
        max_running_requests=2,
        graph_buckets=(1, 2),
        allocator="cuda_malloc_async",
        build_flags=("CUDA_ARCH=120",),
    )
    cache_plan = CompileCacheLaunchPlan(
        schema_version=1,
        kind="compile_cache_launch_plan",
        key=key,
        cache_root=str((tmp_path / "cache").resolve()),
        cache_mode="build",
        builder_id=SGLANG_FIRST_PARTY_COMPILE_BUILDER,
        base_receipt_path=None,
        base_receipt_sha256=None,
    )
    prewarm = CompileOnlyPrewarmManifest(
        schema_version=1,
        kind="compile_only_prewarm_manifest",
        model_lock_sha256=_sha("model-lock"),
        sampling_profile_sha256=SamplingProfile().sha256,
        payloads=(
            CompileOnlyPrewarmPayload("bucket-1", 1, (1, 2), 1, 11),
            CompileOnlyPrewarmPayload("bucket-2", 2, (3, 4), 1, 22),
        ),
    )
    assignment = CompileOnlyAssignmentContract(
        schema_version=1,
        kind="compile_only_assignment_contract",
        assignment_protocol_sha256=COMPILE_ONLY_ASSIGNMENT_PROTOCOL_SHA256,
        cell_id=cell_id,
        registry_sha256=registry_sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        physical_assignment_sha256=_sha("physical"),
        experiment_budget_sha256=_sha("budget"),
        budget_materialization_authority_sha256=_sha("budget-authority"),
        inventory_sha256=_sha("inventory"),
        inventory_source_receipt_sha256=_sha("inventory-source"),
        gpu_uuids=gpu_uuids,
        host_id="host-0",
        fixed_instance_gpu_count=8,
        compile_cache_plan=cache_plan,
        prewarm_manifest=prewarm,
        graceful_shutdown_protocol_sha256=(
            COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256
        ),
        result_pointer_protocol_sha256=COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
        result_pointer_path=str((tmp_path / "result.json").resolve()),
    )
    assignment_path = assignment.write((tmp_path / "assignment.json").resolve())
    cache_plan_path = cache_plan.write((tmp_path / "cache-plan.json").resolve())
    prewarm_path = write_compile_prewarm_manifest(
        prewarm,
        (tmp_path / "prewarm.json").resolve(),
    )
    sampling = SamplingProfile()
    sampling_path = (tmp_path / "sampling.json").resolve()
    sampling.write(sampling_path)
    checkout = (tmp_path / "patched-sglang").resolve()
    target = (tmp_path / "models" / "target" / "snapshots" / ("a" * 40)).resolve()
    tokenizer = (tmp_path / "tokenizer" / "snapshots" / ("c" * 40)).resolve()
    cuda_home = (tmp_path / "cuda").resolve()
    library = (tmp_path / "lib").resolve()
    for directory in (checkout, target, tokenizer, cuda_home, library):
        directory.mkdir(parents=True, exist_ok=True)
    run_config_path = (tmp_path / "run-config.json").resolve()
    run_config = {"schema_version": 1, "mode": "compile-activation-test"}
    canonical = json.dumps(
        run_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    run_config_path.write_bytes(canonical + b"\n")
    run_config_semantic = hashlib.sha256(canonical).hexdigest()
    Path(f"{run_config_path}.sha256").write_text(
        f"{run_config_semantic}\n", encoding="ascii"
    )
    content_path = (tmp_path / "prepared-content.json").resolve()
    content_value = {
        "schema_version": 1,
        "kind": "test-prepared-content",
        "protocol_sha256": _sha("prepared-content-protocol"),
        "model_lock_sha256": prewarm.model_lock_sha256,
        "prepared_model_set_sha256": _sha("prepared-model-set"),
        "snapshots": [],
    }
    content_canonical = json.dumps(
        content_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    content_path.write_bytes(content_canonical + b"\n")
    content_semantic = hashlib.sha256(content_canonical).hexdigest()
    Path(f"{content_path}.sha256").write_text(f"{content_semantic}\n", encoding="ascii")
    server_argv = (
        str(Path(os.sys.executable).resolve()),
        "-m",
        "lightcone_spec.sglang_bridge.launch",
        "--host",
        "127.0.0.1",
        "--port",
        "32124",
        "--model-path",
        str(target),
    )
    launch = CompileLaunchManifest(
        schema_version=1,
        kind="first_party_compile_launch_manifest",
        protocol_sha256=COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
        patched_sglang_checkout=str(checkout),
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        run_config_path=str(run_config_path),
        run_config_raw_sha256=hashlib.sha256(run_config_path.read_bytes()).hexdigest(),
        run_config_semantic_sha256=run_config_semantic,
        compile_cache_plan_path=str(cache_plan_path),
        compile_cache_plan_raw_sha256=hashlib.sha256(
            cache_plan_path.read_bytes()
        ).hexdigest(),
        compile_cache_plan_sha256=cache_plan.sha256,
        prewarm_manifest_path=str(prewarm_path),
        prewarm_manifest_raw_sha256=hashlib.sha256(
            prewarm_path.read_bytes()
        ).hexdigest(),
        prewarm_manifest_sha256=prewarm.sha256,
        sampling_profile_path=str(sampling_path),
        sampling_profile_raw_sha256=hashlib.sha256(
            sampling_path.read_bytes()
        ).hexdigest(),
        prepared_model_content_manifest_path=str(content_path),
        prepared_model_content_manifest_raw_sha256=hashlib.sha256(
            content_path.read_bytes()
        ).hexdigest(),
        prepared_model_content_manifest_sha256=content_semantic,
        prepared_model_content_manifest_size=content_path.stat().st_size,
        target_content_member_id="target:test:primary",
        target_model_id="target/test",
        target_snapshot_path=str(target),
        target_revision="a" * 40,
        target_content_authority_sha256=_sha("target-content"),
        drafter_content_member_id=None,
        drafter_model_id=None,
        drafter_snapshot_path=None,
        drafter_revision=None,
        drafter_content_authority_sha256=None,
        tokenizer_content_member_id="tokenizer:test:primary",
        tokenizer_model_id="tokenizer/test",
        tokenizer_snapshot_path=str(tokenizer),
        tokenizer_revision="c" * 40,
        tokenizer_content_authority_sha256=_sha("tokenizer-content"),
        server_argv=server_argv,
        server_argv_sha256=hashlib.sha256(
            json.dumps(
                {"argv": list(server_argv)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        localhost_port=32124,
        model_lock_sha256=prewarm.model_lock_sha256,
        sampling_profile_sha256=sampling.sha256,
        physical_assignment_sha256=assignment.physical_assignment_sha256,
        experiment_budget_sha256=assignment.experiment_budget_sha256,
        budget_materialization_authority_sha256=(
            assignment.budget_materialization_authority_sha256
        ),
        inventory_sha256=assignment.inventory_sha256,
        gpu_uuids=assignment.gpu_uuids,
        path_entries=(str(Path(os.sys.executable).resolve().parent),),
        library_path_entries=(str(library),),
        cuda_home=str(cuda_home),
    )
    launch_path = launch.write((tmp_path / "launch.json").resolve())
    plan = CompileAssignmentPlan.issue(
        assignment_manifest_path=assignment_path,
        compile_cache_plan_path=cache_plan_path,
        prewarm_manifest_path=prewarm_path,
        launch_manifest_path=launch_path,
        result_pointer_path=assignment.result_pointer_path,
        attempt_id="diagnostic-compile",
    )
    return plan.write((tmp_path / "compile-plan.json").resolve())


def _compile_cell(registry):
    cells = tuple(
        cell
        for cell in registry.cells_for("preflight")
        if cell.resources.workload_class is WorkloadClass.COMPILE
    )
    assert len(cells) == 1
    return cells[0]


def test_raw_compile_plan_enters_diagnostic_activation_but_not_formal(
    tmp_path: Path,
) -> None:
    registry = build_industrial_registry()
    cell = _compile_cell(registry)
    runtime_sha256 = _sha("runtime")
    split_sha256 = _sha("split")
    plan_path = _write_plan(
        tmp_path,
        registry_sha256=registry.sha256,
        cell_id=cell.cell_id,
        gpu_uuids=cell.resources.gpu_uuids,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    activation_before = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )

    authority = materialize_compile_diagnostic_activation(
        registry,
        experiment="preflight",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        compile_assignment_plan_path=plan_path,
    )

    assert authority.protocol_sha256 == COMPILE_DIAGNOSTIC_ACTIVATION_PROTOCOL_SHA256
    assert authority.cell_id == cell.cell_id
    assert authority.diagnostic_status == "READY_DIAGNOSTIC"
    assert authority.formal_status == "BLOCKED"
    assert authority.formal_reason_code == RELEASE_COMPILE_RUNNER_UNAVAILABLE
    assert CompileDiagnosticActivationAuthority.from_dict(authority.to_dict()) == (
        authority
    )
    verify_compile_diagnostic_activation(registry, authority)

    activation_after = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    assert activation_after == activation_before
    # Generic activation is planning availability only.  It is independent of
    # this diagnostic raw plan and cannot turn the authority's formal BLOCKED
    # outcome into execution authority; the trusted operator must still build
    # and deep-revalidate the exact source-owned assignment.
    assert authority.cell_id in activation_after.activated_cell_ids
    compile_row = next(
        row for row in activation_after.dispositions if row.cell_id == cell.cell_id
    )
    assert compile_row.status is RegistryStageDispositionStatus.ACTIVATED
    assert compile_row.reason_code == "release_dispatchability_verified"
    with pytest.raises(CompileRunnerBlocked) as blocked:
        require_release_compile_assignment_plan(CompileAssignmentPlan.load(plan_path))
    assert blocked.value.reason_code == RELEASE_COMPILE_RUNNER_UNAVAILABLE
    assert not Path(CompileAssignmentPlan.load(plan_path).result_pointer_path).exists()


def test_diagnostic_reducer_rejects_wrong_registry_runtime_and_cell(
    tmp_path: Path,
) -> None:
    registry = build_industrial_registry()
    compile_cell = _compile_cell(registry)
    runtime_sha256 = _sha("runtime")
    split_sha256 = _sha("split")
    plan_path = _write_plan(
        tmp_path,
        registry_sha256=registry.sha256,
        cell_id=compile_cell.cell_id,
        gpu_uuids=compile_cell.resources.gpu_uuids,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    with pytest.raises(ValueError, match="registry/runtime/split"):
        materialize_compile_diagnostic_activation(
            registry,
            experiment="preflight",
            runtime_sha256=_sha("other-runtime"),
            split_sha256=split_sha256,
            compile_assignment_plan_path=plan_path,
        )

    other = next(
        cell
        for cell in registry.cells_for("preflight")
        if cell.resources.workload_class is WorkloadClass.CORRECTNESS
    )
    wrong_dir = tmp_path / "wrong-cell"
    wrong_dir.mkdir()
    wrong_path = _write_plan(
        wrong_dir,
        registry_sha256=registry.sha256,
        cell_id=other.cell_id,
        gpu_uuids=other.resources.gpu_uuids,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    with pytest.raises(ValueError, match="registered COMPILE"):
        materialize_compile_diagnostic_activation(
            registry,
            experiment="preflight",
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            compile_assignment_plan_path=wrong_path,
        )

    missing_dir = tmp_path / "missing-cell"
    missing_dir.mkdir()
    missing_path = _write_plan(
        missing_dir,
        registry_sha256=registry.sha256,
        cell_id=_sha("absent-cell"),
        gpu_uuids=compile_cell.resources.gpu_uuids,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    with pytest.raises(ValueError, match="absent from the registry"):
        materialize_compile_diagnostic_activation(
            registry,
            experiment="preflight",
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            compile_assignment_plan_path=missing_path,
        )


def test_diagnostic_authority_reopens_raw_plan_and_rejects_forgery(
    tmp_path: Path,
) -> None:
    registry = build_industrial_registry()
    cell = _compile_cell(registry)
    plan_path = _write_plan(
        tmp_path,
        registry_sha256=registry.sha256,
        cell_id=cell.cell_id,
        gpu_uuids=cell.resources.gpu_uuids,
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
    )
    authority = materialize_compile_diagnostic_activation(
        registry,
        experiment="preflight",
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        compile_assignment_plan_path=plan_path,
    )
    with pytest.raises(ValueError, match="disposition"):
        replace(authority, formal_status="READY")

    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    raw["attempt_id"] = "tampered"
    plan_path.write_text(
        json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidecar"):
        verify_compile_diagnostic_activation(registry, authority)
