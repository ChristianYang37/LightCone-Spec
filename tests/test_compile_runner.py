from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec import PINNED_SGLANG_TREE
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
    RELEASE_COMPILE_RUNNER_UNAVAILABLE,
    RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S,
    CompileAssignmentPlan,
    CompilePrewarmObservation,
    CompileResultPointer,
    CompileRunnerBlocked,
    CompileShutdownObservation,
    execute_compile_assignment_for_cpu_test,
    require_release_compile_assignment_plan,
    write_compile_prewarm_manifest,
)
from lightcone_spec.sglang_bridge.launch import main as launch_main


def _sha(label: str) -> str:
    return hashlib.sha256(f"compile-runner:{label}".encode()).hexdigest()


def _key(*, target_revision: str = "target-revision") -> CompileCacheKey:
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
        target_revision=target_revision,
        drafter_revision="drafter-revision",
        tensor_parallel_size=2,
        context_limit=4096,
        max_running_requests=2,
        graph_buckets=(1, 2),
        allocator="cuda_malloc_async",
        build_flags=("CUDA_ARCH=120",),
    )


def _inputs(tmp_path: Path) -> tuple[CompileAssignmentPlan, Path]:
    cache_root = (tmp_path / "cache").resolve()
    cache_plan = CompileCacheLaunchPlan(
        schema_version=1,
        kind="compile_cache_launch_plan",
        key=_key(),
        cache_root=str(cache_root),
        cache_mode="build",
        builder_id=SGLANG_FIRST_PARTY_COMPILE_BUILDER,
        base_receipt_path=None,
        base_receipt_sha256=None,
    )
    manifest = CompileOnlyPrewarmManifest(
        schema_version=1,
        kind="compile_only_prewarm_manifest",
        model_lock_sha256=_sha("model-lock"),
        sampling_profile_sha256=_sha("sampling"),
        payloads=(
            CompileOnlyPrewarmPayload("bucket-1", 1, (1, 2), 1, 11),
            CompileOnlyPrewarmPayload("bucket-2", 2, (3, 4), 1, 22),
        ),
    )
    result_path = (tmp_path / "compile-result.json").resolve()
    assignment = CompileOnlyAssignmentContract(
        schema_version=1,
        kind="compile_only_assignment_contract",
        assignment_protocol_sha256=COMPILE_ONLY_ASSIGNMENT_PROTOCOL_SHA256,
        cell_id=_sha("cell"),
        registry_sha256=_sha("registry"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        physical_assignment_sha256=_sha("physical"),
        experiment_budget_sha256=_sha("budget"),
        budget_materialization_authority_sha256=_sha("budget-authority"),
        inventory_sha256=_sha("inventory"),
        inventory_source_receipt_sha256=_sha("inventory-source"),
        gpu_uuids=("GPU-0", "GPU-1"),
        host_id="host-0",
        fixed_instance_gpu_count=8,
        compile_cache_plan=cache_plan,
        prewarm_manifest=manifest,
        graceful_shutdown_protocol_sha256=(
            COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256
        ),
        result_pointer_protocol_sha256=COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
        result_pointer_path=str(result_path),
    )
    assignment_path = assignment.write((tmp_path / "assignment.json").resolve())
    cache_plan_path = cache_plan.write((tmp_path / "cache-plan.json").resolve())
    manifest_path = write_compile_prewarm_manifest(
        manifest,
        (tmp_path / "prewarm.json").resolve(),
    )
    return (
        CompileAssignmentPlan.issue(
            assignment_manifest_path=assignment_path,
            compile_cache_plan_path=cache_plan_path,
            prewarm_manifest_path=manifest_path,
            result_pointer_path=result_path,
            attempt_id="attempt-001",
        ),
        result_path,
    )


class _FakeDriver:
    process_id = 4242

    def __init__(self, *, active_requests: int = 0, omit_second: bool = False) -> None:
        self.active_requests = active_requests
        self.omit_second = omit_second
        self.started = False

    def start(self, environment: object) -> None:
        assert environment
        self.started = True

    def prewarm(self, payload: CompileOnlyPrewarmPayload) -> CompilePrewarmObservation:
        request_id = payload.request_id
        if self.omit_second and payload.graph_bucket == 2:
            request_id = "wrong-request"
        return CompilePrewarmObservation(
            request_id=request_id,
            graph_bucket=payload.graph_bucket,
            completed=True,
            provider_receipt_sha256=_sha(f"provider:{payload.request_id}"),
        )

    def graceful_shutdown(self) -> CompileShutdownObservation:
        return CompileShutdownObservation(
            process_id=self.process_id,
            shutdown_requested_ns=100,
            process_exited_ns=200,
            exit_code=0,
            active_requests=self.active_requests,
            queued_requests=0,
            provider_ack_sha256=_sha("shutdown-ack"),
        )


def _materialize(overlay: Path) -> None:
    target = overlay / "triton" / "kernel.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"compiled-kernel")


def test_cpu_fake_lifecycle_requires_exact_prewarm_shutdown_and_terminal_pointer(
    tmp_path: Path,
) -> None:
    plan, result_path = _inputs(tmp_path)
    plan_path = plan.write((tmp_path / "compile-assignment-plan.json").resolve())
    assert CompileAssignmentPlan.load(plan_path) == plan
    pointer = execute_compile_assignment_for_cpu_test(
        plan,
        _FakeDriver(),
        materialize_cache_files=_materialize,
    )

    assert result_path.is_file()
    assert pointer.formal_execution_authorized is False
    assert Path(f"{result_path}.sha256").read_text().strip() == pointer.sha256
    assert pointer.assignment_plan_sha256 == plan.sha256
    assert set(pointer.bindings()) == {
        "assignment_manifest",
        "compile_cache_plan",
        "prewarm_manifest",
        "attempt_receipt",
        "graceful_shutdown_receipt",
        "final_cache_receipt",
        "immutable_cache_object_manifest",
    }
    pointer.reopen()
    assert CompileResultPointer.load(result_path) == pointer


@pytest.mark.parametrize(
    "driver, message",
    [
        (_FakeDriver(omit_second=True), "exactly cover"),
        (_FakeDriver(active_requests=1), "not drained"),
    ],
)
def test_incomplete_prewarm_or_shutdown_never_publishes_result(
    tmp_path: Path,
    driver: _FakeDriver,
    message: str,
) -> None:
    plan, result_path = _inputs(tmp_path)
    with pytest.raises(ValueError, match=message):
        execute_compile_assignment_for_cpu_test(
            plan,
            driver,
            materialize_cache_files=_materialize,
        )
    assert not result_path.exists()


def test_assignment_plan_rejects_caller_key_and_revalidation_tamper(
    tmp_path: Path,
) -> None:
    plan, _ = _inputs(tmp_path)
    foreign = replace(
        CompileCacheLaunchPlan.load(plan.compile_cache_plan_path),
        key=_key(target_revision="caller-selected-revision"),
    )
    foreign_path = foreign.write((tmp_path / "foreign-plan.json").resolve())
    with pytest.raises(ValueError, match="differs from assignment"):
        CompileAssignmentPlan.issue(
            assignment_manifest_path=plan.assignment_manifest_path,
            compile_cache_plan_path=foreign_path,
            prewarm_manifest_path=plan.prewarm_manifest_path,
            result_pointer_path=plan.result_pointer_path,
            attempt_id="attempt-foreign",
        )

    Path(plan.prewarm_manifest_path).write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="sidecar differs|fields differ"):
        plan.revalidate()


def test_resume_reopens_every_pointer_binding_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    plan, result_path = _inputs(tmp_path)
    pointer = execute_compile_assignment_for_cpu_test(
        plan,
        _FakeDriver(),
        materialize_cache_files=_materialize,
    )
    victim = Path(pointer.graceful_shutdown_receipt.absolute_path)
    victim.write_bytes(victim.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after terminal publication"):
        CompileResultPointer.load(result_path)


def test_release_entrypoint_is_named_block_before_any_path_read(tmp_path: Path) -> None:
    assert RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S == ()
    missing = CompileAssignmentPlan(
        schema_version=1,
        kind="first_party_compile_assignment_plan",
        protocol_sha256=(
            "6a2b33824bb1b45b1cf207220fd7762f254182331520467b38718a81b7b90487"
        ),
        assignment_manifest_path=str((tmp_path / "missing-assignment").resolve()),
        assignment_sha256="0" * 64,
        compile_cache_plan_path=str((tmp_path / "missing-plan").resolve()),
        compile_cache_plan_sha256="1" * 64,
        prewarm_manifest_path=str((tmp_path / "missing-prewarm").resolve()),
        prewarm_manifest_sha256="2" * 64,
        compile_key_sha256="3" * 64,
        model_lock_sha256="4" * 64,
        target_revision="target",
        drafter_revision=None,
        physical_assignment_sha256="5" * 64,
        experiment_budget_sha256="6" * 64,
        inventory_sha256="7" * 64,
        gpu_uuids=("GPU-0",),
        host_id="host-0",
        tensor_parallel_size=1,
        context_limit=1,
        max_running_requests=1,
        graph_buckets=(1,),
        graceful_shutdown_protocol_sha256=(
            COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256
        ),
        result_pointer_protocol_sha256=COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
        attempt_id="blocked",
        result_pointer_path=str((tmp_path / "must-not-exist").resolve()),
    )
    # Use the module constant so this test cannot silently bless a stale digest.
    from lightcone_spec.runtime.compile_runner import (
        COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256,
    )

    missing = replace(
        missing,
        protocol_sha256=COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256,
    )
    with pytest.raises(CompileRunnerBlocked) as blocked:
        require_release_compile_assignment_plan(missing)
    assert blocked.value.reason_code == RELEASE_COMPILE_RUNNER_UNAVAILABLE
    assert not Path(missing.result_pointer_path).exists()


def test_launcher_accepts_exact_compile_flags_but_blocks_before_path_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.verify_patched_checkout",
        lambda _path: pytest.fail("compile BLOCK must precede checkout verification"),
    )
    result_path = (tmp_path / "must-not-exist.json").resolve()
    with pytest.raises(CompileRunnerBlocked) as blocked:
        launch_main(
            [
                "--checkout",
                str((tmp_path / "missing-checkout").resolve()),
                "--compile-cache-plan",
                str((tmp_path / "missing-cache-plan").resolve()),
                "--compile-only-assignment",
                str((tmp_path / "missing-assignment").resolve()),
                "--compile-only-manifest",
                str((tmp_path / "missing-manifest").resolve()),
                "--compile-result-pointer",
                str(result_path),
                "--",
            ]
        )
    assert blocked.value.reason_code == RELEASE_COMPILE_RUNNER_UNAVAILABLE
    assert not result_path.exists()
