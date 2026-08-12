from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.runtime.compile_cache import (
    COMPILE_CACHE_ENVIRONMENT_VARIABLES,
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    PINNED_SGLANG_PATCH_MANIFEST_SHA256,
    PINNED_SGLANG_PATCH_SHA256,
    SGLANG_FIRST_PARTY_COMPILE_BUILDER,
    CompileCacheAttemptReceipt,
    CompileCacheCorruptionError,
    CompileCacheForeignIdentityError,
    CompileCacheIncompleteError,
    CompileCacheKey,
    CompileCacheLaunchPlan,
    CompileCacheReceipt,
    ImmutableCompileCache,
    preflight_compile_cache_launch,
    start_compile_cache_launch,
)
from lightcone_spec.sglang_bridge.launch import main as launch_main

ROOT = Path(__file__).resolve().parents[1]


def _key(**updates: object) -> CompileCacheKey:
    values: dict[str, object] = {
        "patched_sglang_tree": PINNED_SGLANG_TREE,
        "patch_manifest_sha256": PINNED_SGLANG_PATCH_MANIFEST_SHA256,
        "patch_sha256": PINNED_SGLANG_PATCH_SHA256,
        "source_sha256": PINNED_SGLANG_COMPILE_SOURCE_SHA256,
        "python_version": "3.12.11",
        "torch_version": "2.11.0+cu130",
        "triton_version": "3.6.0",
        "cuda_version": "13.0",
        "driver_version": "580.65.06",
        "sm_architecture": "sm_120",
        "gpu_model": "RTX PRO 6000 Blackwell Server Edition",
        "dtype": "bfloat16",
        "target_revision": "target-revision",
        "drafter_revision": "drafter-revision",
        "tensor_parallel_size": 1,
        "context_limit": 40960,
        "max_running_requests": 16,
        "graph_buckets": (1, 2, 4, 8, 16),
        "allocator": "cuda_malloc_async",
        "build_flags": ("CUDA_ARCH=120", "USE_FLASHINFER=1"),
    }
    values.update(updates)
    return CompileCacheKey(**values)  # type: ignore[arg-type]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_base(
    root: Path,
    *,
    key: CompileCacheKey | None = None,
    payload: bytes = b"compiled",
) -> tuple[CompileCacheKey, Path, Path]:
    cache_key = _key() if key is None else key
    plan = CompileCacheLaunchPlan.issue(
        key=cache_key,
        cache_root=root,
        cache_mode="build",
    )
    session = start_compile_cache_launch(
        plan,
        process_id=os.getpid(),
        attempt_id=f"base-{hashlib.sha256(payload).hexdigest()[:12]}",
    )
    environment = session.environment({})
    (Path(environment["TRITON_CACHE_DIR"]) / "kernel.bin").write_bytes(payload)
    object_path, receipt_path, _ = session.complete()
    return cache_key, object_path, receipt_path


def _attempts(root: Path) -> list[CompileCacheAttemptReceipt]:
    return [
        CompileCacheAttemptReceipt.load(path)
        for path in sorted((root / "attempts").glob("*.json"))
    ]


def _concurrent_build_worker(
    plan: CompileCacheLaunchPlan,
    barrier: Any,
    queue: Any,
    index: int,
) -> None:
    try:
        session = start_compile_cache_launch(
            plan,
            process_id=os.getpid(),
            attempt_id=f"build-{index}",
        )
        environment = session.environment({})
        kernel = Path(environment["TRITON_CACHE_DIR"]) / "kernel.bin"
        kernel.write_bytes(b"same-compiled-kernel")
        barrier.wait(timeout=20)
        object_path, receipt_path, _ = session.complete()
        queue.put(
            (
                "ok",
                object_path.name,
                CompileCacheReceipt.load(receipt_path).receipt_sha256,
            )
        )
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as error:
        queue.put(("error", type(error).__name__, str(error)))


def _concurrent_reuse_worker(
    plan: CompileCacheLaunchPlan,
    barrier: Any,
    queue: Any,
    index: int,
) -> None:
    try:
        session = start_compile_cache_launch(
            plan,
            process_id=os.getpid(),
            attempt_id=f"reuse-{index}",
        )
        private_file = session.overlay.path / "triton" / "kernel.bin"
        inode = private_file.stat().st_ino
        barrier.wait(timeout=20)
        object_path, _, _ = session.complete()
        queue.put(("ok", object_path.name, inode))
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as error:
        queue.put(("error", type(error).__name__, str(error)))


def _corrupt_reuse_worker(
    plan: CompileCacheLaunchPlan,
    barrier: Any,
    queue: Any,
    index: int,
) -> None:
    barrier.wait(timeout=20)
    try:
        start_compile_cache_launch(
            plan,
            process_id=os.getpid(),
            attempt_id=f"corrupt-{index}",
        )
    except CompileCacheCorruptionError as error:
        queue.put(("blocked", error.reason_code))
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as error:
        queue.put(("error", type(error).__name__))
    else:  # pragma: no cover - parent asserts the payload
        queue.put(("error", "unexpected-success"))


def _run_processes(target: Any, plan: CompileCacheLaunchPlan, count: int) -> list[Any]:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(count)
    queue = context.Queue()
    processes = [
        context.Process(target=target, args=(plan, barrier, queue, index))
        for index in range(count)
    ]
    for process in processes:
        process.start()
    payloads = [queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    return payloads


def test_release_compile_identity_matches_registered_manifest_and_patch(
    tmp_path: Path,
) -> None:
    manifest_path = ROOT / "patches" / "sglang" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patch = manifest_path.parent / manifest["patches"][0]["file"]
    assert _canonical_sha256(manifest) == PINNED_SGLANG_PATCH_MANIFEST_SHA256
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == PINNED_SGLANG_PATCH_SHA256
    assert PINNED_SGLANG_COMPILE_SOURCE_SHA256 == _canonical_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_sglang_compile_source",
            "patched_sglang_tree": PINNED_SGLANG_TREE,
            "patch_manifest_sha256": PINNED_SGLANG_PATCH_MANIFEST_SHA256,
            "patch_sha256": PINNED_SGLANG_PATCH_SHA256,
        }
    )
    with pytest.raises(ValueError, match="foreign source identity"):
        CompileCacheLaunchPlan.issue(
            key=_key(source_sha256="0" * 64),
            cache_root=tmp_path / "foreign-cache",
            cache_mode="build",
        )


def test_launch_plan_is_release_owned_strict_and_content_bound(tmp_path: Path) -> None:
    plan = CompileCacheLaunchPlan.issue(
        key=_key(), cache_root=tmp_path / "cache", cache_mode="build"
    )
    assert plan.builder_id == SGLANG_FIRST_PARTY_COMPILE_BUILDER
    path = plan.write(tmp_path / "plan.json")
    assert CompileCacheLaunchPlan.load(path) == plan

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["key"]["patch_sha256"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="foreign semantic patch"):
        CompileCacheLaunchPlan.load(path)

    with pytest.raises(ValueError, match="release-owned"):
        replace(plan, builder_id="caller.claimed.builder").validate()


def test_build_preflight_is_read_only_and_does_not_create_a_store(
    tmp_path: Path,
) -> None:
    cache_root = (tmp_path / "not-created").resolve()
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

    assert preflight_compile_cache_launch(plan) is None
    assert not cache_root.exists()


@pytest.mark.parametrize("mutation", ["delete", "tamper"])
def test_reuse_preflight_reopens_base_without_creating_launch_artifacts(
    tmp_path: Path,
    mutation: str,
) -> None:
    cache_root = tmp_path / "cache"
    key, object_path, receipt_path = _build_base(cache_root)
    plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=cache_root,
        cache_mode="reuse",
        base_receipt_path=receipt_path,
    )
    receipt = CompileCacheReceipt.load(receipt_path)
    assert preflight_compile_cache_launch(plan) == receipt
    overlays_before = tuple(
        sorted(path.name for path in (cache_root / "overlays").iterdir())
    )
    attempts_before = tuple(
        sorted(path.name for path in (cache_root / "attempts").iterdir())
    )
    base_file = object_path / "triton" / "kernel.bin"
    if mutation == "delete":
        os.chmod(base_file.parent, 0o755)
        base_file.unlink()
    else:
        os.chmod(base_file, 0o644)
        base_file.write_bytes(b"tampered")

    with pytest.raises(CompileCacheCorruptionError):
        preflight_compile_cache_launch(plan)

    assert (
        tuple(sorted(path.name for path in (cache_root / "overlays").iterdir()))
        == overlays_before
    )
    assert (
        tuple(sorted(path.name for path in (cache_root / "attempts").iterdir()))
        == attempts_before
    )


def test_cache_launch_uses_private_overlay_and_never_writes_base(
    tmp_path: Path,
) -> None:
    key, object_path, receipt_path = _build_base(tmp_path / "cache")
    base_file = object_path / "triton" / "kernel.bin"
    before = (
        base_file.read_bytes(),
        base_file.stat().st_mtime_ns,
        base_file.stat().st_ino,
    )
    plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=tmp_path / "cache",
        cache_mode="reuse",
        base_receipt_path=receipt_path,
    )
    session = start_compile_cache_launch(plan, process_id=101, attempt_id="private")
    environment = session.environment({"TRITON_CACHE_DIR": "/caller/shared"})
    private_file = session.overlay.path / "triton" / "kernel.bin"
    assert private_file.read_bytes() == b"compiled"
    assert private_file.stat().st_ino != base_file.stat().st_ino
    assert environment["TRITON_CACHE_DIR"].startswith(str(session.overlay.path))
    for name in COMPILE_CACHE_ENVIRONMENT_VARIABLES:
        assert Path(environment[name]).is_relative_to(session.overlay.path)
    private_file.write_bytes(b"process-private-update")
    session.complete()
    after = (
        base_file.read_bytes(),
        base_file.stat().st_mtime_ns,
        base_file.stat().st_ino,
    )
    assert after == before
    private_attempts = [
        row for row in _attempts(tmp_path / "cache") if row.attempt_id == "private"
    ]
    assert sorted(row.state for row in private_attempts) == [
        "complete",
        "ready",
        "started",
    ]


def test_concurrent_build_contention_publishes_one_atomic_object(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    plan = CompileCacheLaunchPlan.issue(
        key=_key(), cache_root=cache_root, cache_mode="build"
    )
    payloads = _run_processes(_concurrent_build_worker, plan, 4)
    assert {row[0] for row in payloads} == {"ok"}, payloads
    assert len({row[1] for row in payloads}) == 1
    assert len(tuple((cache_root / "objects").iterdir())) == 1
    assert (
        sum(
            row.state == "complete" and row.attempt_id.startswith("build-")
            for row in _attempts(cache_root)
        )
        == 4
    )


def test_concurrent_reuse_copies_base_per_process_without_hardlinks(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    key, object_path, receipt_path = _build_base(cache_root)
    plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=cache_root,
        cache_mode="reuse",
        base_receipt_path=receipt_path,
    )
    payloads = _run_processes(_concurrent_reuse_worker, plan, 4)
    assert {row[0] for row in payloads} == {"ok"}, payloads
    assert len({row[1] for row in payloads}) == 1
    base_inode = (object_path / "triton" / "kernel.bin").stat().st_ino
    assert all(row[2] != base_inode for row in payloads)
    assert (
        sum(
            row.state == "complete" and row.attempt_id.startswith("reuse-")
            for row in _attempts(cache_root)
        )
        == 4
    )


def test_corrupt_base_blocks_all_concurrent_reuse_with_diagnostics(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    key, object_path, receipt_path = _build_base(cache_root)
    plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=cache_root,
        cache_mode="reuse",
        base_receipt_path=receipt_path,
    )
    base_file = object_path / "triton" / "kernel.bin"
    os.chmod(base_file, 0o644)
    base_file.write_bytes(b"corrupt")
    payloads = _run_processes(_corrupt_reuse_worker, plan, 3)
    assert payloads == [("blocked", "cache_object_corrupt")] * 3
    failed = [row for row in _attempts(cache_root) if row.state == "failed"]
    assert len(failed) == 3
    assert {row.failure_code for row in failed} == {"cache_object_corrupt"}
    assert not any(
        row.state == "ready" and row.attempt_id.startswith("corrupt-")
        for row in _attempts(cache_root)
    )


def test_foreign_receipt_and_incomplete_builder_fail_atomically(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    key, _, _ = _build_base(cache_root, payload=b"first")
    foreign_key = _key(driver_version="foreign-driver")
    _, _, foreign_receipt_path = _build_base(
        cache_root, key=foreign_key, payload=b"foreign"
    )
    foreign_receipt = CompileCacheReceipt.load(foreign_receipt_path)
    foreign_plan = CompileCacheLaunchPlan(
        schema_version=1,
        kind="compile_cache_launch_plan",
        key=key,
        cache_root=str(cache_root.resolve()),
        cache_mode="reuse",
        builder_id=SGLANG_FIRST_PARTY_COMPILE_BUILDER,
        base_receipt_path=str(foreign_receipt_path.resolve()),
        base_receipt_sha256=foreign_receipt.receipt_sha256,
    )
    foreign_plan.validate()
    with pytest.raises(CompileCacheForeignIdentityError):
        start_compile_cache_launch(foreign_plan, process_id=202, attempt_id="foreign")

    build_plan = CompileCacheLaunchPlan.issue(
        key=key, cache_root=cache_root, cache_mode="build"
    )
    empty = start_compile_cache_launch(build_plan, process_id=303, attempt_id="empty")
    with pytest.raises(CompileCacheIncompleteError):
        empty.complete()
    failures = {
        row.attempt_id: row.failure_code
        for row in _attempts(cache_root)
        if row.state == "failed"
    }
    assert failures == {
        "empty": "incomplete_cache_build",
        "foreign": "foreign_cache_identity",
    }


def test_caller_sealed_receipt_cannot_be_selected_as_first_party(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    cache = ImmutableCompileCache(cache_root)
    key = _key()
    overlay = cache.create_overlay(key, process_id=404, attempt_id="caller")
    (overlay.path / "claimed.bin").write_bytes(b"caller-claimed-success")
    _, receipt_path = cache.seal_overlay(overlay)
    receipt = CompileCacheReceipt.load(receipt_path)
    assert receipt.builder_id == "unattributed_manual_builder.v1"
    assert receipt.launch_plan_sha256 is None
    with pytest.raises(
        CompileCacheForeignIdentityError,
        match="not produced by the release builder",
    ):
        CompileCacheLaunchPlan.issue(
            key=key,
            cache_root=cache_root,
            cache_mode="reuse",
            base_receipt_path=receipt_path,
        )


def test_runtime_launcher_verifies_cache_before_import_and_seals_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "python").mkdir(parents=True)
    cache_root = tmp_path / "cache"
    plan = CompileCacheLaunchPlan.issue(
        key=_key(), cache_root=cache_root, cache_mode="build"
    )
    plan_path = plan.write(tmp_path / "plan.json")
    events: list[str] = []

    def verify(path: str) -> Path:
        assert Path(path) == checkout
        events.append("checkout-verified")
        return checkout

    def run_module(name: str, *, run_name: str) -> None:
        assert name == "sglang.launch_server"
        assert run_name == "__main__"
        assert events == ["checkout-verified"]
        triton = Path(os.environ["TRITON_CACHE_DIR"])
        assert triton.is_relative_to(cache_root / "overlays")
        (triton / "kernel.bin").write_bytes(b"release-owned-jit")
        events.append("sglang-imported")

    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.verify_patched_checkout", verify
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._validate_blackwell_jit_toolchain",
        lambda: None,
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.runpy.run_module", run_module
    )
    monkeypatch.setenv("TRITON_CACHE_DIR", "/caller/shared-cache")

    assert (
        launch_main(
            [
                "--checkout",
                str(checkout),
                "--compile-cache-plan",
                str(plan_path),
                "--",
                "--model-path",
                "/model",
            ]
        )
        == 0
    )
    assert events == ["checkout-verified", "sglang-imported"]
    assert os.environ["TRITON_CACHE_DIR"] == "/caller/shared-cache"
    assert sorted(row.state for row in _attempts(cache_root)) == [
        "complete",
        "ready",
        "started",
    ]
    assert len(tuple((cache_root / "objects").iterdir())) == 1


def test_runtime_launcher_rejects_corruption_before_sglang_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "python").mkdir(parents=True)
    cache_root = tmp_path / "cache"
    key, object_path, receipt_path = _build_base(cache_root)
    plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=cache_root,
        cache_mode="reuse",
        base_receipt_path=receipt_path,
    )
    plan_path = plan.write(tmp_path / "plan.json")
    base_file = object_path / "triton" / "kernel.bin"
    os.chmod(base_file, 0o644)
    base_file.write_bytes(b"corrupt")
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.verify_patched_checkout",
        lambda _path: checkout,
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.runpy.run_module",
        lambda *_args, **_kwargs: pytest.fail("SGLang import must not run"),
    )
    with pytest.raises(CompileCacheCorruptionError, match="content changed"):
        launch_main(
            [
                "--checkout",
                str(checkout),
                "--compile-cache-plan",
                str(plan_path),
                "--",
                "--model-path",
                "/model",
            ]
        )
    failed = [row for row in _attempts(cache_root) if row.state == "failed"]
    assert len(failed) == 1
    assert failed[0].failure_code == "cache_object_corrupt"
