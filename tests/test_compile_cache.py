from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lightcone_spec.runtime.compile_cache import (
    CompileCacheCorruptionError,
    CompileCacheKey,
    ImmutableCompileCache,
)


def _key(**updates: object) -> CompileCacheKey:
    values: dict[str, object] = {
        "patched_sglang_tree": "a" * 40,
        "source_sha256": "b" * 64,
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


def test_compile_cache_key_binds_every_compile_identity() -> None:
    baseline = _key()
    baseline.validate()
    for name, value in (
        ("driver_version", "581.00"),
        ("gpu_model", "another-gpu"),
        ("dtype", "float16"),
        ("tensor_parallel_size", 2),
        ("context_limit", 32768),
        ("max_running_requests", 8),
        ("graph_buckets", (1, 2, 4)),
        ("allocator", "native"),
        ("build_flags", ("CUDA_ARCH=120",)),
    ):
        assert _key(**{name: value}).sha256 != baseline.sha256
    with pytest.raises(ValueError, match="CUDA graphs"):
        _key(serialized_cuda_graphs=True).validate()


def test_private_overlays_seal_to_one_read_only_content_object(tmp_path: Path) -> None:
    cache = ImmutableCompileCache(tmp_path)
    key = _key()
    first = cache.create_overlay(key, process_id=101, attempt_id="first")
    second = cache.create_overlay(key, process_id=202, attempt_id="second")
    assert first.path != second.path
    (first.path / "kernel.bin").write_bytes(b"compiled")
    (second.path / "kernel.bin").write_bytes(b"compiled")

    first_object, first_receipt = cache.seal_overlay(first)
    second_object, second_receipt = cache.seal_overlay(second)
    assert first_object == second_object
    assert first_receipt != second_receipt
    assert cache.verify(key, first_receipt) == first_object
    assert cache.verify(key, second_receipt) == second_object
    assert os.stat(first_object / "kernel.bin").st_mode & 0o222 == 0


def test_corruption_fails_and_rebuild_uses_a_new_attempt_identity(
    tmp_path: Path,
) -> None:
    cache = ImmutableCompileCache(tmp_path)
    key = _key()
    overlay = cache.create_overlay(key, process_id=1, attempt_id="attempt-one")
    (overlay.path / "kernel.bin").write_bytes(b"compiled")
    object_path, receipt_path = cache.seal_overlay(overlay)
    os.chmod(object_path / "kernel.bin", 0o644)
    (object_path / "kernel.bin").write_bytes(b"corrupt")
    with pytest.raises(CompileCacheCorruptionError, match="content changed"):
        cache.verify(key, receipt_path)

    rebuilt = cache.create_overlay(key, process_id=1, attempt_id="attempt-two")
    (rebuilt.path / "kernel.bin").write_bytes(b"recompiled")
    rebuilt_object, rebuilt_receipt = cache.seal_overlay(rebuilt)
    assert rebuilt_object != object_path
    assert rebuilt_receipt != receipt_path
    assert cache.verify(key, rebuilt_receipt) == rebuilt_object


def test_receipt_and_sidecar_are_strict(tmp_path: Path) -> None:
    cache = ImmutableCompileCache(tmp_path)
    key = _key()
    overlay = cache.create_overlay(key, process_id=9, attempt_id="strict")
    (overlay.path / "kernel.bin").write_bytes(b"compiled")
    _, receipt_path = cache.seal_overlay(overlay)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["unknown"] = True
    receipt_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown or missing"):
        cache.verify(key, receipt_path)


def test_overlay_identity_is_exclusive(tmp_path: Path) -> None:
    cache = ImmutableCompileCache(tmp_path)
    key = _key()
    cache.create_overlay(key, process_id=1, attempt_id="same")
    with pytest.raises(FileExistsError):
        cache.create_overlay(key, process_id=1, attempt_id="same")
