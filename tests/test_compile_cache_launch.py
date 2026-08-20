from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_schema import config_value

import lightcone_spec.runtime.compile_cache as compile_cache_module
from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.config import RunConfig, run_config_sha256
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
from lightcone_spec.runtime.distributed import (
    DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
)
from lightcone_spec.runtime.proof_artifact import publish_canonical_json_no_replace
from lightcone_spec.sglang_bridge.config import (
    sglang_adaptation_payload,
    sglang_adaptation_sha256,
)
from lightcone_spec.sglang_bridge.launch import (
    _bind_runtime_adaptation_config,
    _raw_config_requires_formal_runtime_authority,
    _validate_compile_runtime_environment,
)
from lightcone_spec.sglang_bridge.launch import (
    main as launch_main,
)

ROOT = Path(__file__).resolve().parents[1]


def test_formal_runtime_authority_requirement_is_exactly_adaptive_native_or_multirank() -> (
    None
):
    dflash_tp1 = config_value("tts")
    assert _raw_config_requires_formal_runtime_authority(dflash_tp1) is False

    dflash_tp2 = config_value("tts")
    dflash_tp2["runtime"]["tensor_parallel_size"] = 2
    assert _raw_config_requires_formal_runtime_authority(dflash_tp2) is True

    dspark_tp1 = config_value("l0")
    dspark_tp1["model"]["algorithm"] = "DSPARK"
    assert _raw_config_requires_formal_runtime_authority(dspark_tp1) is True

    static_dspark = config_value("static")
    static_dspark["model"]["algorithm"] = "DSPARK"
    assert _raw_config_requires_formal_runtime_authority(static_dspark) is False


def test_launcher_binds_the_native_adaptation_payload_to_run_config(
    tmp_path: Path,
) -> None:
    config = RunConfig.model_validate(config_value("tts"))
    payload = sglang_adaptation_payload(config)
    assert payload is not None
    path = (tmp_path / "adaptation.json").resolve()
    publish_canonical_json_no_replace(path, payload)

    binding = _bind_runtime_adaptation_config(
        config,
        ["--speculative-adaptation-config", str(path)],
    )
    assert binding.semantic_sha256 == sglang_adaptation_sha256(config)

    foreign = (tmp_path / "foreign-adaptation.json").resolve()
    publish_canonical_json_no_replace(
        foreign,
        {**payload, "stride": int(payload["stride"]) + 1},
    )
    with pytest.raises(ValueError, match="differs from RunConfig"):
        _bind_runtime_adaptation_config(
            config,
            ["--speculative-adaptation-config", str(foreign)],
        )

    static = RunConfig.model_validate(config_value("static"))
    with pytest.raises(ValueError, match="allocation-free"):
        _bind_runtime_adaptation_config(
            static,
            ["--speculative-adaptation-config", str(path)],
        )


def test_launcher_module_is_torch_free_in_a_fresh_interpreter() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import sys; import lightcone_spec.sglang_bridge.launch; "
                "print('torch' in sys.modules)"
            ),
        ),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.strip() == "False"


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
        "target_revision": "a" * 40,
        "drafter_revision": "b" * 40,
        "tensor_parallel_size": 1,
        "context_limit": 40960,
        "max_running_requests": 16,
        "graph_buckets": (1, 2, 4, 8, 16),
        "allocator": "cuda_malloc_async",
        "build_flags": ("CUDA_ARCH=120", "USE_FLASHINFER=1"),
    }
    values.update(updates)
    return CompileCacheKey(**values)  # type: ignore[arg-type]


def _diagnostic_key(**updates: object) -> CompileCacheKey:
    values = {"graph_buckets": (1,), "build_flags": ()}
    values.update(updates)
    return _key(**values)


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
    patch = manifest_path.parent / manifest["patches"][-1]["file"]
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


def _mock_compile_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    def command(argv: list[str]) -> str | None:
        if argv[0] == "nvidia-smi":
            return (
                "GPU-test, RTX PRO 6000 Blackwell Server Edition, 98304, "
                "580.65.06, 12.0, 0000:01:00.0"
            )
        if argv[0] == "nvcc":
            return "Cuda compilation tools, release 13.0, V13.0.0"
        return None

    monkeypatch.setattr("lightcone_spec.sglang_bridge.launch._command", command)
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._torch_runtime",
        lambda: {
            "importable": True,
            "version": "2.11.0+cu130",
            "cuda_build": "13.0",
            "cuda_available": True,
            "device_count": 1,
        },
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._package_version",
        lambda name: "3.6.0" if name == "triton" else None,
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.platform.python_version",
        lambda: "3.12.11",
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._torch_preimported", lambda: False
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)


def _run_config(key: CompileCacheKey) -> RunConfig:
    raw = config_value("target_only" if key.drafter_revision is None else "static")
    raw["model"]["target_revision"] = key.target_revision
    if key.drafter_revision is not None:
        raw["model"]["drafter_revision"] = key.drafter_revision
    raw["runtime"]["tensor_parallel_size"] = key.tensor_parallel_size
    raw["runtime"]["max_running_requests"] = key.max_running_requests
    raw["runtime"]["device_identity"] = "GPU-test"
    return RunConfig.model_validate(raw)


def _compile_server_argv(
    tmp_path: Path,
    key: CompileCacheKey,
    config: RunConfig,
) -> list[str]:
    target = tmp_path / "target" / "snapshots" / key.target_revision
    target.mkdir(parents=True, exist_ok=True)
    argv = [
        "--model-path",
        str(target),
        "--max-running-requests",
        str(key.max_running_requests),
        "--tp-size",
        str(key.tensor_parallel_size),
        "--dtype",
        key.dtype,
        "--context-length",
        str(key.context_limit),
        "--disable-cuda-graph",
        "--speculative-speed-study-metrics",
    ]
    if key.drafter_revision is not None:
        drafter = tmp_path / "drafter" / "snapshots" / key.drafter_revision
        drafter.mkdir(parents=True, exist_ok=True)
        argv.extend(
            (
                "--speculative-algorithm",
                config.model.algorithm,
                "--speculative-draft-model-path",
                str(drafter),
                "--speculative-num-draft-tokens",
                str(config.runtime.speculative_num_draft_tokens),
                "--speculative-draft-window-size",
                str(config.runtime.speculative_num_draft_tokens),
                "--speculative-accept-threshold-single",
                "1.0",
                "--speculative-accept-threshold-acc",
                "1.0",
                "--speculative-use-rejection-sampling",
            )
        )
    return argv


def _write_run_config(
    tmp_path: Path,
    key: CompileCacheKey,
    *,
    name: str = "run-config.json",
) -> tuple[RunConfig, Path, str]:
    config = _run_config(key)
    path = (tmp_path / name).resolve()
    body = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = run_config_sha256(config)
    path.write_text(body + "\n", encoding="utf-8")
    Path(f"{path}.sha256").write_text(digest + "\n", encoding="ascii")
    return config, path, digest


def _wrapper_authority_argv(
    *,
    plan: CompileCacheLaunchPlan,
    plan_path: Path,
    run_config_path: Path,
    run_config_sha256_value: str,
) -> list[str]:
    return [
        "--compile-cache-plan",
        str(plan_path),
        "--compile-cache-plan-sha256",
        plan.sha256,
        "--compile-cache-key-sha256",
        plan.key.sha256,
        "--run-config",
        str(run_config_path),
        "--run-config-sha256",
        run_config_sha256_value,
    ]


def test_runtime_launcher_observes_exact_compile_toolchain_before_cache_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_compile_runtime_environment(monkeypatch)
    plan = CompileCacheLaunchPlan.issue(
        key=_diagnostic_key(),
        cache_root=tmp_path / "must-not-exist",
        cache_mode="build",
    )
    config = _run_config(plan.key)
    assert _validate_compile_runtime_environment(
        plan,
        config.model_dump(mode="json"),
        _compile_server_argv(tmp_path, plan.key, config),
    ) == (None, ("GPU-test",))
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync"
    assert not Path(plan.cache_root).exists()


@pytest.mark.parametrize("variable", ("CUDA_HOME", "CUDA_PATH"))
def test_runtime_launcher_rejects_conflicting_cuda_toolkit_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variable: str,
) -> None:
    _mock_compile_runtime_environment(monkeypatch)
    selected_root = (tmp_path / "selected-cuda").resolve()
    foreign_root = (tmp_path / "foreign-cuda").resolve()
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.shutil.which",
        lambda executable: (
            str(selected_root / "bin" / "nvcc") if executable == "nvcc" else None
        ),
    )
    monkeypatch.setenv(variable, str(foreign_root))
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._torch_runtime",
        lambda: pytest.fail("CUDA toolkit conflict must precede Torch import"),
    )
    plan = CompileCacheLaunchPlan.issue(
        key=_diagnostic_key(),
        cache_root=tmp_path / "must-not-exist",
        cache_mode="build",
    )
    config = _run_config(plan.key)

    with pytest.raises(RuntimeError, match=rf"{variable} conflicts.*finalized PATH"):
        _validate_compile_runtime_environment(
            plan,
            config.model_dump(mode="json"),
            _compile_server_argv(tmp_path, plan.key, config),
        )

    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert not Path(plan.cache_root).exists()


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"python_version": "3.12.10"}, "toolchain"),
        ({"torch_version": "2.11.0"}, "toolchain"),
        ({"triton_version": "3.5.0"}, "toolchain"),
        ({"cuda_version": "12.9"}, "toolchain"),
        ({"driver_version": "580.65.05"}, "toolchain"),
        ({"gpu_model": "foreign-gpu"}, "toolchain"),
        ({"sm_architecture": "sm_90"}, "toolchain"),
        ({"dtype": "float16"}, "diagnostic contract"),
        (
            {"allocator": "native"},
            "diagnostic contract",
        ),
        (
            {"graph_buckets": (1, 2)},
            "diagnostic contract",
        ),
        (
            {"build_flags": ("USE_FLASHINFER=1",)},
            "diagnostic contract",
        ),
    ),
)
def test_runtime_launcher_rejects_caller_minted_compile_toolchain_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    _mock_compile_runtime_environment(monkeypatch)
    plan = CompileCacheLaunchPlan.issue(
        key=_diagnostic_key(**updates),
        cache_root=tmp_path / "must-not-exist",
        cache_mode="build",
    )
    config = _run_config(plan.key)
    with pytest.raises(ValueError, match=message):
        _validate_compile_runtime_environment(
            plan,
            config.model_dump(mode="json"),
            _compile_server_argv(tmp_path, plan.key, config),
        )
    assert not Path(plan.cache_root).exists()


@pytest.mark.parametrize(
    "torch_runtime",
    (
        {
            "importable": False,
            "version": "2.11.0+cu130",
            "cuda_build": "13.0",
            "cuda_available": False,
            "device_count": 0,
        },
        {
            "importable": True,
            "version": "2.11.0+cu130",
            "cuda_build": "13.0",
            "cuda_available": False,
            "device_count": 1,
        },
        {
            "importable": True,
            "version": "2.11.0+cu130",
            "cuda_build": "13.0",
            "cuda_available": True,
            "device_count": 0,
        },
        {
            "importable": True,
            "version": "2.11.0+cu130",
            "cuda_build": "13.0",
            "cuda_available": True,
            "device_count": 2,
        },
    ),
)
def test_runtime_launcher_requires_exact_usable_torch_cuda_ranks_before_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    torch_runtime: dict[str, object],
) -> None:
    _mock_compile_runtime_environment(monkeypatch)
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._torch_runtime",
        lambda: torch_runtime,
    )
    plan = CompileCacheLaunchPlan.issue(
        key=_diagnostic_key(),
        cache_root=tmp_path / "must-not-exist",
        cache_mode="build",
    )
    config = _run_config(plan.key)
    with pytest.raises(ValueError, match="exact usable Torch CUDA ranks"):
        _validate_compile_runtime_environment(
            plan,
            config.model_dump(mode="json"),
            _compile_server_argv(tmp_path, plan.key, config),
        )
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert not Path(plan.cache_root).exists()


def test_runtime_launcher_accepts_exact_two_gpu_tp2_visibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_compile_runtime_environment(monkeypatch)

    def command(argv: list[str]) -> str | None:
        if argv[0] == "nvidia-smi":
            return (
                "GPU-test-0, RTX PRO 6000 Blackwell Server Edition, 98304, "
                "580.65.06, 12.0, 0000:01:00.0\n"
                "GPU-test-1, RTX PRO 6000 Blackwell Server Edition, 98304, "
                "580.65.06, 12.0, 0000:02:00.0"
            )
        if argv[0] == "nvcc":
            return "Cuda compilation tools, release 13.0, V13.0.0"
        return None

    monkeypatch.setattr("lightcone_spec.sglang_bridge.launch._command", command)
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._torch_runtime",
        lambda: {
            "importable": True,
            "version": "2.11.0+cu130",
            "cuda_build": "13.0",
            "cuda_available": True,
            "device_count": 2,
        },
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-0,GPU-test-1")
    plan = CompileCacheLaunchPlan.issue(
        key=_diagnostic_key(tensor_parallel_size=2),
        cache_root=tmp_path / "must-not-exist",
        cache_mode="build",
    )
    raw = config_value("static")
    raw["model"]["target_revision"] = plan.key.target_revision
    raw["model"]["drafter_revision"] = plan.key.drafter_revision
    raw["runtime"].update(
        {
            "tensor_parallel_size": 2,
            "max_running_requests": plan.key.max_running_requests,
            "device_identity": "GPU-test-0,GPU-test-1",
            "distributed_runtime_capability": "patched_two_gpu_v1",
            "distributed_release_capability_sha256": (
                DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES["tp2_dp1"].sha256
            ),
            "distributed_capability_receipt_sha256": "d" * 64,
        }
    )
    config = RunConfig.model_validate(raw)
    assert _validate_compile_runtime_environment(
        plan,
        config.model_dump(mode="json"),
        _compile_server_argv(tmp_path, plan.key, config),
    ) == (None, ("GPU-test-0", "GPU-test-1"))
    assert not Path(plan.cache_root).exists()


def test_runtime_launcher_accepts_exact_two_gpu_dp2_visibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_compile_runtime_environment(monkeypatch)

    def command(argv: list[str]) -> str | None:
        if argv[0] == "nvidia-smi":
            return (
                "GPU-test-0, RTX PRO 6000 Blackwell Server Edition, 98304, "
                "580.65.06, 12.0, 0000:01:00.0\n"
                "GPU-test-1, RTX PRO 6000 Blackwell Server Edition, 98304, "
                "580.65.06, 12.0, 0000:02:00.0"
            )
        if argv[0] == "nvcc":
            return "Cuda compilation tools, release 13.0, V13.0.0"
        return None

    monkeypatch.setattr("lightcone_spec.sglang_bridge.launch._command", command)
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._torch_runtime",
        lambda: {
            "importable": True,
            "version": "2.11.0+cu130",
            "cuda_build": "13.0",
            "cuda_available": True,
            "device_count": 2,
        },
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-0,GPU-test-1")
    plan = CompileCacheLaunchPlan.issue(
        key=_diagnostic_key(),
        cache_root=tmp_path / "must-not-exist",
        cache_mode="build",
    )
    raw = config_value("static")
    raw["model"]["target_revision"] = plan.key.target_revision
    raw["model"]["drafter_revision"] = plan.key.drafter_revision
    raw["runtime"].update(
        {
            "data_parallel_size": 2,
            "max_running_requests": plan.key.max_running_requests,
            "device_identity": "GPU-test-0,GPU-test-1",
            "router_identity": "preflight-qualified-sticky-router-v1",
            "distributed_runtime_capability": "patched_two_gpu_v1",
            "distributed_release_capability_sha256": (
                DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES["tp1_dp2"].sha256
            ),
            "distributed_capability_receipt_sha256": "d" * 64,
            "process_group_backend": "none",
        }
    )
    config = RunConfig.model_validate(raw)
    server_argv = _compile_server_argv(tmp_path, plan.key, config)
    server_argv.extend(("--dp-size", "2"))
    assert _validate_compile_runtime_environment(
        plan,
        config.model_dump(mode="json"),
        server_argv,
    ) == (None, ("GPU-test-0", "GPU-test-1"))
    assert not Path(plan.cache_root).exists()


@pytest.mark.parametrize(
    ("flag", "replacement", "message"),
    (
        ("--tp-size", "2", "dimensions"),
        ("--context-length", "32768", "dimensions"),
        ("--max-running-requests", "8", "dimensions"),
        ("--speculative-num-draft-tokens", "8", "exact RunConfig"),
    ),
)
def test_runtime_launcher_binds_inner_sglang_argv_to_run_config_and_cache_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flag: str,
    replacement: str,
    message: str,
) -> None:
    _mock_compile_runtime_environment(monkeypatch)
    plan = CompileCacheLaunchPlan.issue(
        key=_diagnostic_key(),
        cache_root=tmp_path / "must-not-exist",
        cache_mode="build",
    )
    config = _run_config(plan.key)
    argv = _compile_server_argv(tmp_path, plan.key, config)
    argv[argv.index(flag) + 1] = replacement
    with pytest.raises(ValueError, match=message):
        _validate_compile_runtime_environment(
            plan, config.model_dump(mode="json"), argv
        )
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert not Path(plan.cache_root).exists()


@pytest.mark.parametrize(
    ("flag", "revision", "role"),
    (
        ("--model-path", "c" * 40, "target"),
        ("--speculative-draft-model-path", "d" * 40, "drafter"),
    ),
)
def test_runtime_launcher_binds_model_roots_to_locked_revisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flag: str,
    revision: str,
    role: str,
) -> None:
    _mock_compile_runtime_environment(monkeypatch)
    plan = CompileCacheLaunchPlan.issue(
        key=_diagnostic_key(),
        cache_root=tmp_path / "must-not-exist",
        cache_mode="build",
    )
    config = _run_config(plan.key)
    argv = _compile_server_argv(tmp_path, plan.key, config)
    foreign = tmp_path / f"foreign-{role}" / "snapshots" / revision
    foreign.mkdir(parents=True)
    argv[argv.index(flag) + 1] = str(foreign)
    with pytest.raises(ValueError, match="locked revision snapshot directory"):
        _validate_compile_runtime_environment(
            plan, config.model_dump(mode="json"), argv
        )
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert not Path(plan.cache_root).exists()


@pytest.mark.parametrize(
    "forbidden",
    (
        ("--speculative-algorithm", "DFLASH"),
        ("--speculative-draft-model-path", "/foreign/drafter"),
        ("--speculative-adaptation-config", "/foreign/adaptation.json"),
    ),
)
def test_target_only_child_forbids_speculative_and_adaptation_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    forbidden: tuple[str, str],
) -> None:
    _mock_compile_runtime_environment(monkeypatch)
    plan = CompileCacheLaunchPlan.issue(
        key=_diagnostic_key(drafter_revision=None),
        cache_root=tmp_path / "must-not-exist",
        cache_mode="build",
    )
    config = _run_config(plan.key)
    argv = [*_compile_server_argv(tmp_path, plan.key, config), *forbidden]
    with pytest.raises(ValueError, match="forbids speculative/adaptation"):
        _validate_compile_runtime_environment(
            plan, config.model_dump(mode="json"), argv
        )
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert not Path(plan.cache_root).exists()


def test_runtime_launcher_rejects_preimported_torch_before_allocator_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_compile_runtime_environment(monkeypatch)
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._torch_preimported", lambda: True
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._torch_runtime",
        lambda: pytest.fail("preimport gate must precede Torch observation"),
    )
    plan = CompileCacheLaunchPlan.issue(
        key=_diagnostic_key(),
        cache_root=tmp_path / "must-not-exist",
        cache_mode="build",
    )
    config = _run_config(plan.key)
    with pytest.raises(RuntimeError, match="imported before.*allocator"):
        _validate_compile_runtime_environment(
            plan,
            config.model_dump(mode="json"),
            _compile_server_argv(tmp_path, plan.key, config),
        )
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert not Path(plan.cache_root).exists()


def test_runtime_launcher_rejects_conflicting_allocator_before_torch_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_compile_runtime_environment(monkeypatch)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._torch_runtime",
        lambda: pytest.fail("allocator conflict must precede Torch import"),
    )
    plan = CompileCacheLaunchPlan.issue(
        key=_diagnostic_key(),
        cache_root=tmp_path / "must-not-exist",
        cache_mode="build",
    )
    config = _run_config(plan.key)
    with pytest.raises(ValueError, match="allocator environment"):
        _validate_compile_runtime_environment(
            plan,
            config.model_dump(mode="json"),
            _compile_server_argv(tmp_path, plan.key, config),
        )
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native"
    assert not Path(plan.cache_root).exists()


def test_launch_plan_is_release_owned_strict_and_content_bound(tmp_path: Path) -> None:
    plan = CompileCacheLaunchPlan.issue(
        key=_key(), cache_root=tmp_path / "cache", cache_mode="build"
    )
    assert plan.builder_id == SGLANG_FIRST_PARTY_COMPILE_BUILDER
    path = plan.write(tmp_path / "plan.json")
    assert CompileCacheLaunchPlan.load(path) == plan

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["key"]["patch_sha256"] = "0" * 64
    path.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    Path(f"{path}.sha256").write_text(
        _canonical_sha256(raw) + "\n",
        encoding="ascii",
    )
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


def test_publish_waiter_accepts_atomic_object_while_claim_is_still_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ImmutableCompileCache(tmp_path / "cache")
    object_path = store.objects / ("a" * 64)
    object_path.mkdir(mode=0o555)
    claim = store.objects / f".{object_path.name}.publish"
    claim.mkdir(mode=0o700)
    temporary = store.objects / f".{object_path.name}.tmp.waiter"
    temporary.mkdir(mode=0o700)
    (temporary / "kernel.bin").write_bytes(b"redundant")
    fsyncs: list[Path] = []

    def observe_fsync(path: Path) -> None:
        assert temporary.exists()
        fsyncs.append(path)

    monkeypatch.setattr(compile_cache_module, "_fsync_directory", observe_fsync)

    store._publish_object_directory(temporary, object_path)

    assert fsyncs == [store.objects]
    assert object_path.is_dir()
    assert claim.is_dir()
    assert not temporary.exists()


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
        key=_diagnostic_key(), cache_root=cache_root, cache_mode="build"
    )
    plan_path = plan.write(tmp_path / "plan.json")
    _, run_config_path, run_config_digest = _write_run_config(tmp_path, plan.key)
    events: list[str] = []
    observed_paths: list[str | None] = []

    def verify(path: str) -> Path:
        assert Path(path) == checkout
        events.append("checkout-verified")
        return checkout

    def run_module(name: str, *, run_name: str) -> None:
        assert name == "sglang.launch_server"
        assert run_name == "__main__"
        assert events == ["checkout-verified"]
        observed_paths.append(os.environ.get("PATH"))
        triton = Path(os.environ["TRITON_CACHE_DIR"])
        assert triton.is_relative_to(cache_root / "overlays")
        (triton / "kernel.bin").write_bytes(b"release-owned-jit")
        events.append("sglang-imported")

    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.verify_patched_checkout", verify
    )

    def validate(
        _plan: object, _config: object, _argv: object
    ) -> tuple[None, tuple[str, ...]]:
        observed_paths.append(os.environ.get("PATH"))
        return None, ("GPU-fixture",)

    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._validate_compile_runtime_environment",
        validate,
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.runpy.run_module", run_module
    )
    monkeypatch.setenv("TRITON_CACHE_DIR", "/caller/shared-cache")
    monkeypatch.setenv("PATH", "/caller/bin")

    assert (
        launch_main(
            [
                "--checkout",
                str(checkout),
                *_wrapper_authority_argv(
                    plan=plan,
                    plan_path=plan_path,
                    run_config_path=run_config_path,
                    run_config_sha256_value=run_config_digest,
                ),
                "--",
                "--model-path",
                "/model",
                "--dtype",
                "bfloat16",
            ]
        )
        == 0
    )
    assert events == ["checkout-verified", "sglang-imported"]
    assert os.environ["TRITON_CACHE_DIR"] == "/caller/shared-cache"
    assert os.environ["PATH"] == "/caller/bin"
    assert len(observed_paths) == 2 and observed_paths[0] == observed_paths[1]
    assert observed_paths[0] is not None
    assert observed_paths[0].split(os.pathsep)[0] == str(
        Path(sys.executable).parent.resolve()
    )
    assert sorted(row.state for row in _attempts(cache_root)) == [
        "complete",
        "ready",
        "started",
    ]
    assert len(tuple((cache_root / "objects").iterdir())) == 1
    result_receipt_path = next((cache_root / "receipts").glob("*.json"))
    result_receipt = CompileCacheReceipt.load(result_receipt_path)
    assert result_receipt.builder_id == "unattributed_manual_builder.v1"
    assert result_receipt.launch_plan_sha256 is None
    with pytest.raises(CompileCacheForeignIdentityError, match="release builder"):
        CompileCacheLaunchPlan.issue(
            key=plan.key,
            cache_root=cache_root,
            cache_mode="reuse",
            base_receipt_path=result_receipt_path,
        )


@pytest.mark.parametrize("mutated_identity", ("plan", "key"))
def test_runtime_launcher_rejects_mutated_expected_compile_identity_before_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutated_identity: str,
) -> None:
    plan = CompileCacheLaunchPlan.issue(
        key=_key(),
        cache_root=tmp_path / "cache",
        cache_mode="build",
    )
    plan_path = plan.write(tmp_path / "plan.json")
    expected_plan = "0" * 64 if mutated_identity == "plan" else plan.sha256
    expected_key = "0" * 64 if mutated_identity == "key" else plan.key.sha256
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.verify_patched_checkout",
        lambda _path: pytest.fail("identity mismatch must precede checkout"),
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.start_compile_cache_launch",
        lambda _plan: pytest.fail("identity mismatch must precede cache mutation"),
    )

    with pytest.raises(ValueError, match="launch authority"):
        launch_main(
            [
                "--checkout",
                str((tmp_path / "checkout").resolve()),
                "--compile-cache-plan",
                str(plan_path),
                "--compile-cache-plan-sha256",
                expected_plan,
                "--compile-cache-key-sha256",
                expected_key,
                "--run-config",
                str((tmp_path / "missing-run-config.json").resolve()),
                "--run-config-sha256",
                "0" * 64,
                "--",
                "--model-path",
                "/model",
                "--dtype",
                "bfloat16",
            ]
        )
    assert not Path(plan.cache_root).exists()


def test_runtime_launcher_rejects_valid_plan_pair_swap_before_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = CompileCacheLaunchPlan.issue(
        key=_key(),
        cache_root=tmp_path / "original-cache",
        cache_mode="build",
    )
    plan_path = original.write(tmp_path / "plan.json")
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    replacement = CompileCacheLaunchPlan.issue(
        key=_key(driver_version="replacement-driver"),
        cache_root=tmp_path / "replacement-cache",
        cache_mode="build",
    )
    replacement_path = replacement.write(replacement_root / "plan.json")
    os.replace(replacement_path, plan_path)
    os.replace(Path(f"{replacement_path}.sha256"), Path(f"{plan_path}.sha256"))
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.verify_patched_checkout",
        lambda _path: pytest.fail("swapped identity must precede checkout"),
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.start_compile_cache_launch",
        lambda _plan: pytest.fail("swapped identity must precede cache mutation"),
    )

    with pytest.raises(ValueError, match="launch authority"):
        launch_main(
            [
                "--checkout",
                str((tmp_path / "checkout").resolve()),
                "--compile-cache-plan",
                str(plan_path),
                "--compile-cache-plan-sha256",
                original.sha256,
                "--compile-cache-key-sha256",
                original.key.sha256,
                "--run-config",
                str((tmp_path / "missing-run-config.json").resolve()),
                "--run-config-sha256",
                "0" * 64,
                "--",
                "--model-path",
                "/model",
                "--dtype",
                "bfloat16",
            ]
        )
    assert not Path(original.cache_root).exists()
    assert not Path(replacement.cache_root).exists()


def test_runtime_launcher_rejects_diagnostic_reuse_before_sglang_import(
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
    _, run_config_path, run_config_digest = _write_run_config(tmp_path, plan.key)
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.verify_patched_checkout",
        lambda _path: checkout,
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.runpy.run_module",
        lambda *_args, **_kwargs: pytest.fail("SGLang import must not run"),
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._validate_compile_runtime_environment",
        lambda _plan, _config, _argv: (None, ("GPU-fixture",)),
    )
    with pytest.raises(RuntimeError, match="reuse_requires_model_content_authority"):
        launch_main(
            [
                "--checkout",
                str(checkout),
                *_wrapper_authority_argv(
                    plan=plan,
                    plan_path=plan_path,
                    run_config_path=run_config_path,
                    run_config_sha256_value=run_config_digest,
                ),
                "--",
                "--model-path",
                "/model",
            ]
        )
    assert object_path.is_dir()
    assert not [row for row in _attempts(cache_root) if row.state == "failed"]
