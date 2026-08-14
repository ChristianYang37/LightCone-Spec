"""Launch SGLang only from a verified disposable patched checkout."""

from __future__ import annotations

import argparse
import hmac
import os
import platform
import runpy
import shutil
import sys
from pathlib import Path

from lightcone_spec.doctor import (
    _command,
    _nvcc_release,
    _package_version,
    _parse_gpu_inventory,
    _torch_runtime,
)
from lightcone_spec.locking.prepared_models import PreparedModelSnapshot
from lightcone_spec.runtime.compile_cache import (
    COMPILE_CACHE_ENVIRONMENT_VARIABLES,
    CompileCacheLaunchPlan,
    CompileOnlyAssignmentContract,
    _load_canonical_json_with_sidecar,
    require_release_compile_only_assignment,
    start_compile_cache_launch,
    validate_compile_key_for_run_config,
    validate_compile_runtime_toolchain,
)
from lightcone_spec.runtime.compile_runner import (
    CompileRunnerBlocked,
    require_release_compile_assignment_plan,
)

from .checkout import verify_patched_checkout


def _bind_interpreter_tools() -> None:
    """Make console tools installed beside this Python visible to SGLang JITs."""
    # Resolve the directory, not the executable: venv Python is commonly a
    # symlink to the system interpreter, while tools such as ninja live beside
    # the symlink in the venv's bin directory.
    interpreter_bin = str(Path(sys.executable).parent.resolve())
    entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    os.environ["PATH"] = os.pathsep.join(
        [interpreter_bin, *(entry for entry in entries if entry != interpreter_bin)]
    )


_GPU_INVENTORY_ARGV = [
    "nvidia-smi",
    "--query-gpu=uuid,name,memory.total,driver_version,compute_cap,pci.bus_id",
    "--format=csv,noheader,nounits",
]
_ALLOCATOR_ENVIRONMENT = "PYTORCH_CUDA_ALLOC_CONF"
_ALLOCATOR_VALUE = "backend:cudaMallocAsync"


def _torch_preimported() -> bool:
    return "torch" in sys.modules


def _validate_cuda_toolkit_environment() -> None:
    configured = {
        name: os.environ[name]
        for name in ("CUDA_HOME", "CUDA_PATH")
        if name in os.environ
    }
    if not configured:
        return
    selected_nvcc = shutil.which("nvcc")
    if selected_nvcc is None:
        raise RuntimeError("configured CUDA toolkit has no nvcc on finalized PATH")
    resolved_nvcc = Path(selected_nvcc).resolve(strict=False)
    for name, value in configured.items():
        root = Path(value)
        if (
            not value
            or value.strip() != value
            or "\n" in value
            or "\r" in value
            or not root.is_absolute()
            or (root / "bin" / "nvcc").resolve(strict=False) != resolved_nvcc
        ):
            raise RuntimeError(
                f"{name} conflicts with the nvcc selected on finalized PATH"
            )


def _selected_gpu(inventory: list[dict[str, object]]) -> dict[str, object]:
    selector = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selector is None or not selector.strip():
        if len(inventory) != 1:
            raise ValueError(
                "compile-cache launch requires one unambiguous visible GPU"
            )
        return inventory[0]
    selectors = [value.strip() for value in selector.split(",") if value.strip()]
    if len(selectors) != 1:
        raise ValueError("compile-cache launch requires exactly one visible GPU")
    selected = selectors[0]
    if selected.isdecimal():
        index = int(selected)
        if index >= len(inventory):
            raise ValueError("CUDA_VISIBLE_DEVICES index is outside GPU inventory")
        return inventory[index]
    matches = [device for device in inventory if device.get("uuid") == selected]
    if len(matches) != 1:
        raise ValueError("CUDA_VISIBLE_DEVICES does not select one inventory GPU")
    return matches[0]


def _server_dtype(server_argv: list[str]) -> str:
    if any(argument.startswith("--dtype=") for argument in server_argv):
        raise ValueError("SGLang dtype must use one canonical --dtype argument")
    positions = [
        index for index, argument in enumerate(server_argv) if argument == "--dtype"
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(server_argv):
        raise ValueError("SGLang launch lacks one exact dtype")
    return server_argv[positions[0] + 1]


def _server_flag_value(server_argv: list[str], flag: str) -> str:
    if any(argument.startswith(f"{flag}=") for argument in server_argv):
        raise ValueError(f"SGLang {flag} must use one canonical argument pair")
    positions = [
        index for index, argument in enumerate(server_argv) if argument == flag
    ]
    if (
        len(positions) != 1
        or positions[0] + 1 >= len(server_argv)
        or server_argv[positions[0] + 1].startswith("--")
    ):
        raise ValueError(f"SGLang launch requires one exact {flag}")
    return server_argv[positions[0] + 1]


def _positive_server_int(server_argv: list[str], flag: str) -> int:
    value = _server_flag_value(server_argv, flag)
    if not value.isdecimal() or int(value) < 1:
        raise ValueError(f"SGLang {flag} must be a canonical positive integer")
    return int(value)


def _validate_model_revision_root(
    server_argv: list[str],
    *,
    flag: str,
    revision: str,
    role: str,
) -> None:
    root = _server_flag_value(server_argv, flag)
    PreparedModelSnapshot(model_id=role, revision=revision, root=root).validate()


def _validate_compile_server_argv(
    plan: CompileCacheLaunchPlan,
    config: object,
    server_argv: list[str],
) -> None:
    """Bind compile-affecting SGLang inputs to the release-issued key."""

    validate_compile_key_for_run_config(plan, config=config)
    key = plan.key
    if _server_dtype(server_argv) != key.dtype:
        raise ValueError("SGLang dtype differs from the compile-cache key")
    expected_ints = {
        "--tp-size": key.tensor_parallel_size,
        "--context-length": key.context_limit,
        "--max-running-requests": key.max_running_requests,
    }
    if any(
        _positive_server_int(server_argv, flag) != expected
        for flag, expected in expected_ints.items()
    ):
        raise ValueError("SGLang launch dimensions differ from the compile-cache key")
    if any(
        argument == "--dp-size" or argument.startswith("--dp-size=")
        for argument in server_argv
    ):
        raise ValueError(
            "single-GPU compile-cache launch cannot carry data parallelism"
        )
    if server_argv.count("--disable-cuda-graph") != 1 or any(
        argument.startswith("--disable-cuda-graph=") for argument in server_argv
    ):
        raise ValueError("diagnostic compile-cache launch must disable CUDA graphs")
    _validate_model_revision_root(
        server_argv,
        flag="--model-path",
        revision=config.model.target_revision,
        role="target",
    )

    speculative_arguments = [
        argument
        for argument in server_argv
        if argument.startswith("--speculative-")
        and argument != "--speculative-speed-study-metrics"
    ]
    if server_argv.count("--speculative-speed-study-metrics") != 1 or any(
        argument.startswith("--speculative-speed-study-metrics=")
        for argument in server_argv
    ):
        raise ValueError("SGLang launch requires exact diagnostic metrics accounting")
    if config.method == "target_only":
        if speculative_arguments:
            raise ValueError(
                "target-only compile-cache key forbids speculative/adaptation argv"
            )
        return
    _validate_model_revision_root(
        server_argv,
        flag="--speculative-draft-model-path",
        revision=config.model.drafter_revision,
        role="drafter",
    )
    expected_speculative = {
        "--speculative-algorithm": config.model.algorithm,
        "--speculative-num-draft-tokens": str(
            config.runtime.speculative_num_draft_tokens
        ),
        "--speculative-draft-window-size": str(
            config.runtime.speculative_num_draft_tokens
        ),
        "--speculative-accept-threshold-single": "1.0",
        "--speculative-accept-threshold-acc": "1.0",
    }
    if any(
        _server_flag_value(server_argv, flag) != expected
        for flag, expected in expected_speculative.items()
    ):
        raise ValueError("SGLang speculative argv differs from the exact RunConfig")
    rejection_count = server_argv.count("--speculative-use-rejection-sampling")
    if rejection_count != int(config.runtime.use_rejection_sampling) or any(
        argument.startswith("--speculative-use-rejection-sampling=")
        for argument in server_argv
    ):
        raise ValueError("SGLang rejection sampling differs from the exact RunConfig")
    adaptation_flags = (
        "--speculative-adaptation-config",
        "--speculative-adaptation-reserve-mb",
        "--speculative-adaptation-telemetry-path",
    )
    if config.adaptation is None and any(
        argument == flag or argument.startswith(f"{flag}=")
        for flag in adaptation_flags
        for argument in server_argv
    ):
        raise ValueError("allocation-free RunConfig forbids adaptation argv")


def _load_bound_run_config(
    path: str,
    expected_sha256: str,
) -> dict[str, object]:
    requested = Path(path)
    if requested.is_symlink():
        raise ValueError("run-config path cannot be a symlink")
    resolved = requested.resolve(strict=False)
    raw, semantic_sha256 = _load_canonical_json_with_sidecar(
        resolved,
        label="run config",
    )
    if not hmac.compare_digest(semantic_sha256, expected_sha256):
        raise ValueError("run-config identity differs from launch authority")
    return raw


def _validate_compile_runtime_environment(
    plan: CompileCacheLaunchPlan,
    raw_config: dict[str, object],
    server_argv: list[str],
) -> str | None:
    """Observe the child process environment before cache mutation or import."""

    key = plan.key
    if (
        key.dtype != "bfloat16"
        or key.allocator != "cuda_malloc_async"
        or key.graph_buckets != (1,)
        or key.build_flags != ()
    ):
        raise ValueError("compile-cache key is outside the fixed diagnostic contract")
    _validate_cuda_toolkit_environment()
    previous_allocator = os.environ.get(_ALLOCATOR_ENVIRONMENT)
    if previous_allocator not in {None, _ALLOCATOR_VALUE}:
        raise ValueError("PyTorch allocator environment conflicts with compile key")
    if _torch_preimported():
        raise RuntimeError(
            "Torch was imported before compile-cache allocator initialization"
        )
    os.environ[_ALLOCATOR_ENVIRONMENT] = _ALLOCATOR_VALUE
    try:
        # Importing the strict config package currently initializes Torch via
        # adaptation package exports.  Do it only after the allocator contract
        # is installed; the wrapper module itself must remain Torch-free.
        from lightcone_spec.config import RunConfig

        config = RunConfig.model_validate(raw_config)
        if raw_config != config.model_dump(mode="json"):
            raise ValueError("run-config semantic identity differs after validation")
        _validate_compile_server_argv(plan, config, server_argv)
        raw_inventory = _command(_GPU_INVENTORY_ARGV)
        parsed = _parse_gpu_inventory(raw_inventory)
        inventory = parsed.get("devices")
        if parsed.get("parse_error") is not None or not isinstance(inventory, list):
            raise ValueError("compile-cache launch lacks a valid GPU inventory")
        selected = _selected_gpu(inventory)
        capability = selected.get("compute_capability")
        if not isinstance(capability, str):
            raise TypeError("compile-cache launch lacks GPU compute capability")
        components = capability.split(".")
        if len(components) != 2 or any(
            not component.isdecimal() for component in components
        ):
            raise ValueError("compile-cache launch has invalid GPU compute capability")
        torch_runtime = _torch_runtime()
        if (
            torch_runtime.get("importable") is not True
            or torch_runtime.get("cuda_available") is not True
            or torch_runtime.get("device_count") != 1
        ):
            raise ValueError(
                "compile-cache launch requires one usable Torch CUDA device"
            )
        validate_compile_runtime_toolchain(
            key,
            python_version=platform.python_version(),
            torch_version=torch_runtime.get("version"),
            triton_version=_package_version("triton"),
            torch_cuda_version=torch_runtime.get("cuda_build"),
            nvcc_cuda_version=_nvcc_release(_command(["nvcc", "--version"])),
            driver_version=selected.get("driver_version"),
            gpu_model=selected.get("name"),
            sm_architecture="sm_" + "".join(components),
        )
    except BaseException:
        if previous_allocator is None:
            os.environ.pop(_ALLOCATOR_ENVIRONMENT, None)
        else:
            os.environ[_ALLOCATOR_ENVIRONMENT] = previous_allocator
        raise
    return previous_allocator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lightcone-sglang-launch")
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--compile-cache-plan", required=True)
    parser.add_argument("--compile-cache-plan-sha256", required=True)
    parser.add_argument("--compile-cache-key-sha256", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--run-config-sha256", required=True)
    parser.add_argument("--compile-only-assignment")
    parser.add_argument("--compile-only-manifest")
    parser.add_argument("--compile-result-pointer")
    parser.add_argument("server_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    server_argv = list(args.server_argv)
    if server_argv and server_argv[0] == "--":
        server_argv = server_argv[1:]
    compile_only_flag = any(
        argument == "--compile-only" or argument.startswith("--compile-only=")
        for argument in server_argv
    )
    # Compile-only is a distinct assignment lifecycle, not an SGLang server
    # flag that may borrow the serving cache launcher.  Block it before checkout
    # verification, cache-plan loading, directory creation, model import, or GPU
    # mutation.  A future release must replace this gate only together with its
    # registered prewarm/finalization and atomic result-pointer implementation.
    compile_terminal_values = (
        args.compile_only_assignment,
        args.compile_only_manifest,
        args.compile_result_pointer,
    )
    if all(value is not None for value in compile_terminal_values):
        # The release-owned plan allowlist/actuator is empty today.  Preserve a
        # named pre-side-effect BLOCK before opening caller paths.  Once that
        # release authority exists, this gate is replaced atomically with
        # CompileAssignmentPlan.load(...) and the first-party lifecycle runner.
        require_release_compile_assignment_plan()
    if any(value is not None for value in compile_terminal_values) and not all(
        value is not None for value in compile_terminal_values
    ):
        if (
            args.compile_only_assignment is not None
            and args.compile_only_manifest is None
            and args.compile_result_pointer is None
        ):
            assignment = CompileOnlyAssignmentContract.load(
                Path(args.compile_only_assignment)
            )
            require_release_compile_only_assignment(assignment)
        raise CompileRunnerBlocked("compile_assignment_cli_contract_incomplete")
    if compile_only_flag:
        require_release_compile_only_assignment()
    if not server_argv:
        raise ValueError("SGLang server arguments are required after --")
    if "sglang" in sys.modules:
        raise RuntimeError("sglang was imported before checkout verification")
    previous_path = os.environ.get("PATH")
    _bind_interpreter_tools()
    try:
        plan = CompileCacheLaunchPlan.load(args.compile_cache_plan)
        if not hmac.compare_digest(
            plan.sha256, args.compile_cache_plan_sha256
        ) or not hmac.compare_digest(plan.key.sha256, args.compile_cache_key_sha256):
            raise ValueError(
                "compile-cache plan identity differs from launch authority"
            )
        config = _load_bound_run_config(args.run_config, args.run_config_sha256)
        if plan.cache_mode != "build":
            raise RuntimeError(
                "diagnostic_compile_cache_reuse_requires_model_content_authority"
            )
        checkout = verify_patched_checkout(args.checkout)
        previous_allocator = _validate_compile_runtime_environment(
            plan, config, server_argv
        )
    except BaseException:
        if previous_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous_path
        raise
    try:
        # Preliminary launches bind revision paths but do not carry the formal
        # PreparedModelContentAuthority replay.  Their cache output is useful
        # only inside this diagnostic process and can never authorize reuse or
        # a release-builder claim.
        session = start_compile_cache_launch(
            plan,
            _release_builder_receipt=False,
        )
    except BaseException:
        if previous_allocator is None:
            os.environ.pop(_ALLOCATOR_ENVIRONMENT, None)
        else:
            os.environ[_ALLOCATOR_ENVIRONMENT] = previous_allocator
        if previous_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous_path
        raise

    python_root = str(checkout / "python")
    original_argv = sys.argv
    original_path = list(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    managed_environment = {
        name: os.environ.get(name) for name in COMPILE_CACHE_ENVIRONMENT_VARIABLES
    }
    managed_environment[_ALLOCATOR_ENVIRONMENT] = previous_allocator
    managed_environment["PATH"] = previous_path
    try:
        cache_environment = session.environment(os.environ)
        for name in cache_environment.keys() & managed_environment.keys():
            os.environ[name] = cache_environment[name]
        # A verified disposable checkout is source, never a bytecode cache.
        sys.dont_write_bytecode = True
        sys.path.insert(0, python_root)
        sys.argv = ["sglang.launch_server", *server_argv]
        runpy.run_module("sglang.launch_server", run_name="__main__")
    except SystemExit as error:
        if error.code is not None and error.code != 0:
            session.fail(error, reason_code="sglang_launch_failed")
        else:
            session.complete()
        raise
    except BaseException as error:
        session.fail(error, reason_code="sglang_launch_failed")
        raise
    else:
        session.complete()
        return 0
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_dont_write_bytecode
        for name, value in managed_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
