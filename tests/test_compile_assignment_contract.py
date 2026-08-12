from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.registry import WorkloadClass, build_industrial_registry
from lightcone_spec.experiments.stage_activation import (
    RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE as STAGE_COMPILE_BLOCK_REASON,
)
from lightcone_spec.experiments.stage_activation import (
    release_dispatch_rejection_reason,
)
from lightcone_spec.runtime.compile_cache import (
    COMPILE_ONLY_ASSIGNMENT_PROTOCOL_SHA256,
    COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256,
    COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    PINNED_SGLANG_PATCH_MANIFEST_SHA256,
    PINNED_SGLANG_PATCH_SHA256,
    RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE,
    SGLANG_FIRST_PARTY_COMPILE_BUILDER,
    CompileCacheKey,
    CompileCacheLaunchPlan,
    CompileOnlyAssignmentContract,
    CompileOnlyAssignmentUnavailableError,
    CompileOnlyPrewarmManifest,
    CompileOnlyPrewarmPayload,
    require_release_compile_only_assignment,
)
from lightcone_spec.sglang_bridge.launch import main as launch_main


def _sha(label: str) -> str:
    return hashlib.sha256(f"compile-assignment:{label}".encode()).hexdigest()


def _key() -> CompileCacheKey:
    return CompileCacheKey(
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
        target_revision="target-revision",
        drafter_revision="drafter-revision",
        tensor_parallel_size=2,
        context_limit=4096,
        max_running_requests=2,
        graph_buckets=(1, 2),
        allocator="cuda_malloc_async",
        build_flags=("CUDA_ARCH=120",),
    )


def _contract(tmp_path: Path) -> CompileOnlyAssignmentContract:
    cache_root = (tmp_path / "compile-cache").resolve()
    plan = CompileCacheLaunchPlan(
        schema_version=1,
        kind="compile_cache_launch_plan",
        key=_key(),
        cache_root=str(cache_root),
        cache_mode="build",
        builder_id=SGLANG_FIRST_PARTY_COMPILE_BUILDER,
        base_receipt_path=None,
        base_receipt_sha256=None,
    )
    prewarm = CompileOnlyPrewarmManifest(
        schema_version=1,
        kind="compile_only_prewarm_manifest",
        model_lock_sha256=_sha("model-lock"),
        sampling_profile_sha256=_sha("sampling"),
        payloads=(
            CompileOnlyPrewarmPayload(
                request_id="prewarm-b1",
                graph_bucket=1,
                input_token_ids=(1, 2, 3),
                requested_output_tokens=2,
                sampling_seed=101,
            ),
            CompileOnlyPrewarmPayload(
                request_id="prewarm-b2",
                graph_bucket=2,
                input_token_ids=(4, 5, 6),
                requested_output_tokens=2,
                sampling_seed=202,
            ),
        ),
    )
    return CompileOnlyAssignmentContract(
        schema_version=1,
        kind="compile_only_assignment_contract",
        assignment_protocol_sha256=COMPILE_ONLY_ASSIGNMENT_PROTOCOL_SHA256,
        cell_id=_sha("cell"),
        registry_sha256=_sha("registry"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        physical_assignment_sha256=_sha("physical-assignment"),
        experiment_budget_sha256=_sha("experiment-budget"),
        budget_materialization_authority_sha256=_sha("budget-authority"),
        inventory_sha256=_sha("inventory"),
        inventory_source_receipt_sha256=_sha("inventory-source"),
        gpu_uuids=("GPU-compile-0", "GPU-compile-1"),
        host_id="compile-host",
        fixed_instance_gpu_count=8,
        compile_cache_plan=plan,
        prewarm_manifest=prewarm,
        graceful_shutdown_protocol_sha256=(
            COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256
        ),
        result_pointer_protocol_sha256=(COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256),
        result_pointer_path=str((tmp_path / "compile-result.json").resolve()),
    )


def test_future_compile_assignment_binds_every_registered_boundary(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    contract.validate()
    path = contract.write((tmp_path / "compile-assignment.json").resolve())
    assert CompileOnlyAssignmentContract.load(path) == contract
    assert not Path(contract.compile_cache_plan.cache_root).exists()
    assert not Path(contract.result_pointer_path).exists()

    for mutation, message in (
        (
            replace(contract, physical_assignment_sha256="foreign"),
            "physical_assignment",
        ),
        (
            replace(contract, inventory_source_receipt_sha256="foreign"),
            "inventory_source",
        ),
        (
            replace(contract, gpu_uuids=("GPU-compile-0", "GPU-compile-0")),
            "GPU UUID",
        ),
        (
            replace(
                contract,
                prewarm_manifest=replace(
                    contract.prewarm_manifest,
                    payloads=contract.prewarm_manifest.payloads[:1],
                ),
            ),
            "graph buckets",
        ),
        (
            replace(contract, result_pointer_protocol_sha256="0" * 64),
            "result-pointer protocol",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            mutation.validate()

    duplicate_path = (tmp_path / "duplicate-assignment.json").resolve()
    body = path.read_text(encoding="utf-8")
    duplicate_body = body.replace(
        '"schema_version":1', '"schema_version":1,"schema_version":1', 1
    )
    duplicate_path.write_text(duplicate_body, encoding="utf-8")
    duplicate_semantic = hashlib.sha256(
        json.dumps(
            json.loads(duplicate_body), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    Path(f"{duplicate_path}.sha256").write_text(
        f"{duplicate_semantic}\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        CompileOnlyAssignmentContract.load(duplicate_path)


def test_compile_build_plan_issue_and_release_gate_are_pre_mutation(
    tmp_path: Path,
) -> None:
    cache_root = (tmp_path / "not-created-cache").resolve()
    plan = CompileCacheLaunchPlan.issue(
        key=_key(),
        cache_root=cache_root,
        cache_mode="build",
    )
    assert plan.cache_root == str(cache_root)
    assert not cache_root.exists()

    contract = replace(_contract(tmp_path), compile_cache_plan=plan)
    with pytest.raises(
        CompileOnlyAssignmentUnavailableError,
        match=RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE,
    ) as blocked:
        require_release_compile_only_assignment(contract)
    assert blocked.value.reason_code == (
        RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE
    )
    assert not cache_root.exists()
    assert not Path(contract.result_pointer_path).exists()


@pytest.mark.parametrize("mode", ["assignment", "raw-server-flag"])
def test_sglang_launcher_blocks_compile_only_before_checkout_or_cache_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    contract = _contract(tmp_path)
    assignment_path = contract.write((tmp_path / "compile-assignment.json").resolve())
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.verify_patched_checkout",
        lambda _path: pytest.fail("compile-only must block before checkout"),
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.CompileCacheLaunchPlan.load",
        lambda _path: pytest.fail("compile-only must block before cache-plan loading"),
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.start_compile_cache_launch",
        lambda _plan: pytest.fail("compile-only must block before cache mutation"),
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._validate_blackwell_jit_toolchain",
        lambda: pytest.fail("compile-only must block before GPU/toolchain probing"),
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.runpy.run_module",
        lambda *_args, **_kwargs: pytest.fail(
            "compile-only must block before model/runtime import"
        ),
    )
    arguments = [
        "--checkout",
        str((tmp_path / "checkout").resolve()),
        "--compile-cache-plan",
        str((tmp_path / "missing-plan.json").resolve()),
    ]
    if mode == "assignment":
        arguments.extend(("--compile-only-assignment", str(assignment_path), "--"))
    else:
        arguments.extend(("--", "--compile-only"))

    with pytest.raises(
        CompileOnlyAssignmentUnavailableError,
        match=RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE,
    ):
        launch_main(arguments)
    assert not Path(contract.compile_cache_plan.cache_root).exists()
    assert not Path(contract.result_pointer_path).exists()


def test_every_current_compile_cell_has_the_same_release_owned_block_reason() -> None:
    registry = build_industrial_registry()
    compile_cells = tuple(
        cell
        for cell in registry.cells
        if cell.resources.workload_class is WorkloadClass.COMPILE
    )
    assert compile_cells
    assert RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE == (
        STAGE_COMPILE_BLOCK_REASON
    )
    assert {release_dispatch_rejection_reason(cell) for cell in compile_cells} == {
        RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE
    }
    assert not any(
        Path(cell.resources.evidence_root).exists() for cell in compile_cells
    )
