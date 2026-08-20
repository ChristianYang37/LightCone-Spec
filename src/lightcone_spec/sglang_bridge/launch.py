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
_ADAPTATION_CONFIG_SHA256_ENVIRONMENT = "LIGHTCONE_FORMAL_ADAPTATION_CONFIG_SHA256"


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


def _selected_gpus(
    inventory: list[dict[str, object]],
    *,
    expected_count: int,
) -> tuple[dict[str, object], ...]:
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("compile-cache launch GPU count is invalid")
    selector = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selector is None or not selector.strip():
        if len(inventory) != expected_count:
            raise ValueError(
                "compile-cache launch lacks exact unambiguous visible GPUs"
            )
        return tuple(inventory)
    selectors = [value.strip() for value in selector.split(",") if value.strip()]
    if len(selectors) != expected_count or len(set(selectors)) != expected_count:
        raise ValueError("compile-cache launch requires its exact visible GPU count")
    selected_devices: list[dict[str, object]] = []
    for selected in selectors:
        if selected.isdecimal():
            index = int(selected)
            if index >= len(inventory):
                raise ValueError("CUDA_VISIBLE_DEVICES index is outside GPU inventory")
            device = inventory[index]
        else:
            matches = [device for device in inventory if device.get("uuid") == selected]
            if len(matches) != 1:
                raise ValueError(
                    "CUDA_VISIBLE_DEVICES does not select one inventory GPU"
                )
            device = matches[0]
        if any(device is prior for prior in selected_devices):
            raise ValueError("CUDA_VISIBLE_DEVICES repeats one inventory GPU")
        selected_devices.append(device)
    return tuple(selected_devices)


def _selected_gpu(inventory: list[dict[str, object]]) -> dict[str, object]:
    """Compatibility wrapper for the single-GPU compile contract."""

    return _selected_gpus(inventory, expected_count=1)[0]


def _config_gpu_uuids(
    config: object,
    *,
    expected_count: int,
    allow_local_default: bool = False,
) -> tuple[str, ...] | None:
    identity = getattr(getattr(config, "runtime", None), "device_identity", None)
    if type(identity) is not str:
        raise ValueError("RunConfig lacks exact ordered GPU identity")
    if allow_local_default and identity == "local-device-0":
        return None
    values = tuple(identity.split(","))
    if (
        len(values) != expected_count
        or len(set(values)) != expected_count
        or any(not value or value.strip() != value for value in values)
    ):
        raise ValueError("RunConfig ordered GPU identity differs from topology")
    return values


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


def _bind_runtime_adaptation_config(config: object, server_argv: list[str]):
    """Bind the independent native payload to the exact validated RunConfig."""

    from lightcone_spec.config import RunConfig
    from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding
    from lightcone_spec.sglang_bridge.config import (
        sglang_adaptation_payload,
        sglang_adaptation_sha256,
    )

    if type(config) is dict:
        config = RunConfig.model_validate(config)
    if type(config) is not RunConfig:
        raise TypeError("adaptation payload requires an exact RunConfig")

    flag = "--speculative-adaptation-config"
    present = flag in server_argv or any(
        argument.startswith(f"{flag}=") for argument in server_argv
    )
    expected = sglang_adaptation_payload(config)
    if expected is None:
        if present:
            raise ValueError("allocation-free launch carries an adaptation payload")
        return None
    path = Path(_server_flag_value(server_argv, flag))
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or not path.is_file()
        or path.is_symlink()
    ):
        raise ValueError("adaptation payload path is unavailable")
    binding = CanonicalJsonProofBinding.bind(path)
    expected_sha256 = sglang_adaptation_sha256(config)
    if (
        binding.reopen() != expected
        or expected_sha256 is None
        or binding.semantic_sha256 != expected_sha256
    ):
        raise ValueError("adaptation payload differs from RunConfig")
    return binding


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
    carries_dp = any(
        argument == "--dp-size" or argument.startswith("--dp-size=")
        for argument in server_argv
    )
    if config.runtime.data_parallel_size == 1:
        if carries_dp:
            raise ValueError("TP1/DP1 compile-cache launch cannot carry DP argv")
    elif (
        not carries_dp
        or _positive_server_int(server_argv, "--dp-size")
        != config.runtime.data_parallel_size
    ):
        raise ValueError("distributed compile-cache launch DP argv differs")
    graph_switch = "--lightcone-fixed-address-publication-graph"
    graph_batches = "--lightcone-graph-batch-sizes"
    graph_no_fallback = "--lightcone-disable-graph-eager-fallback"
    graph_arguments = {
        graph_switch,
        graph_batches,
        graph_no_fallback,
    }
    runtime = config.runtime
    if runtime.cuda_graph_mode == "disabled":
        if (
            server_argv.count("--disable-cuda-graph") != 1
            or any(
                argument.startswith("--disable-cuda-graph=") for argument in server_argv
            )
            or any(argument in graph_arguments for argument in server_argv)
        ):
            raise ValueError(
                "disabled compile-cache launch has unregistered CUDA graph argv"
            )
    elif runtime.cuda_graph_mode == "fixed_address_publication_v1":
        if (
            "--disable-cuda-graph" in server_argv
            or any(
                argument.startswith("--disable-cuda-graph=") for argument in server_argv
            )
            or server_argv.count(graph_switch) != 1
            or server_argv.count(graph_batches) != 1
            or _server_flag_value(server_argv, graph_batches) != "1"
            or server_argv.count(graph_no_fallback) != 1
            or _server_flag_value(server_argv, "--cuda-graph-backend-decode") != "full"
            or _positive_server_int(server_argv, "--cuda-graph-max-bs-decode") != 1
            or _server_flag_value(server_argv, "--cuda-graph-bs-decode") != "1"
        ):
            raise ValueError("fixed-address compile-cache launch graph argv differs")
    else:  # pragma: no cover - RuntimeConfig literal invariant
        raise ValueError("compile-cache launch graph mode is unsupported")
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
    mechanism_flags = {
        "--lightcone-adaptation-microbatch-size": str(
            config.runtime.adaptation_microbatch_size
        ),
        "--lightcone-adaptation-publication-coalescing": str(
            config.runtime.adaptation_publication_coalescing
        ),
        "--lightcone-adaptation-stream-priority": (
            config.runtime.adaptation_stream_priority
        ),
    }
    if config.adaptation is None:
        if any(
            argument == flag or argument.startswith(f"{flag}=")
            for flag in mechanism_flags
            for argument in server_argv
        ):
            raise ValueError(
                "allocation-free RunConfig forbids adaptation mechanism argv"
            )
    elif any(
        _server_flag_value(server_argv, flag) != expected
        for flag, expected in mechanism_flags.items()
    ):
        raise ValueError("SGLang adaptation mechanism argv differs from RunConfig")
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
) -> tuple[str | None, tuple[str, ...]]:
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
        _bind_runtime_adaptation_config(config, server_argv)
        raw_inventory = _command(_GPU_INVENTORY_ARGV)
        parsed = _parse_gpu_inventory(raw_inventory)
        inventory = parsed.get("devices")
        if parsed.get("parse_error") is not None or not isinstance(inventory, list):
            raise ValueError("compile-cache launch lacks a valid GPU inventory")
        expected_devices = (
            config.runtime.tensor_parallel_size * config.runtime.data_parallel_size
        )
        selected_devices = _selected_gpus(
            inventory,
            expected_count=expected_devices,
        )
        selected_gpu_uuids = tuple(device.get("uuid") for device in selected_devices)
        expected_gpu_uuids = _config_gpu_uuids(
            config,
            expected_count=expected_devices,
            allow_local_default=True,
        )
        if expected_gpu_uuids is not None and selected_gpu_uuids != expected_gpu_uuids:
            raise ValueError("visible GPU UUID order differs from RunConfig")
        torch_runtime = _torch_runtime()
        if (
            torch_runtime.get("importable") is not True
            or torch_runtime.get("cuda_available") is not True
            or torch_runtime.get("device_count") != expected_devices
        ):
            raise ValueError(
                "compile-cache launch requires its exact usable Torch CUDA ranks"
            )
        for selected in selected_devices:
            capability = selected.get("compute_capability")
            if not isinstance(capability, str):
                raise TypeError("compile-cache launch lacks GPU compute capability")
            components = capability.split(".")
            if len(components) != 2 or any(
                not component.isdecimal() for component in components
            ):
                raise ValueError(
                    "compile-cache launch has invalid GPU compute capability"
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
    return previous_allocator, selected_gpu_uuids


def _validate_qualification_runtime_environment(
    raw_config: dict[str, object],
    server_argv: list[str],
) -> tuple[str | None, tuple[str, ...]]:
    """Validate a root-dispatched live suite without the single-GPU cache key."""

    _validate_cuda_toolkit_environment()
    previous_allocator = os.environ.get(_ALLOCATOR_ENVIRONMENT)
    if previous_allocator not in {None, _ALLOCATOR_VALUE}:
        raise ValueError("PyTorch allocator environment conflicts with qualification")
    if _torch_preimported():
        raise RuntimeError("Torch was imported before qualification allocator setup")
    os.environ[_ALLOCATOR_ENVIRONMENT] = _ALLOCATOR_VALUE
    try:
        from lightcone_spec.config import RunConfig

        config = RunConfig.model_validate(raw_config)
        if raw_config != config.model_dump(mode="json"):
            raise ValueError("qualification RunConfig identity changed")
        _bind_runtime_adaptation_config(config, server_argv)
        expected_pairs = {
            "--host": "127.0.0.1",
            "--port": _server_flag_value(server_argv, "--port"),
            "--model-path": str(
                Path(_server_flag_value(server_argv, "--model-path")).resolve()
            ),
            "--tp-size": str(config.runtime.tensor_parallel_size),
        }
        if config.runtime.data_parallel_size > 1:
            expected_pairs["--dp-size"] = str(config.runtime.data_parallel_size)
        elif any(
            argument == "--dp-size" or argument.startswith("--dp-size=")
            for argument in server_argv
        ):
            raise ValueError("TP1/DP1 qualification cannot carry DP argv")
        if any(
            _server_flag_value(server_argv, flag) != expected
            for flag, expected in expected_pairs.items()
        ):
            raise ValueError("qualification topology/server argv differs")
        if (
            config.model.algorithm != "NONE"
            and _server_flag_value(server_argv, "--speculative-algorithm")
            != config.model.algorithm
        ):
            raise ValueError("qualification algorithm argv differs")
        mechanism_flags = {
            "--lightcone-adaptation-microbatch-size": str(
                config.runtime.adaptation_microbatch_size
            ),
            "--lightcone-adaptation-publication-coalescing": str(
                config.runtime.adaptation_publication_coalescing
            ),
            "--lightcone-adaptation-stream-priority": (
                config.runtime.adaptation_stream_priority
            ),
        }
        if config.adaptation is not None and any(
            _server_flag_value(server_argv, flag) != expected
            for flag, expected in mechanism_flags.items()
        ):
            raise ValueError("qualification adaptation mechanism argv differs")
        if config.runtime.cuda_graph_mode == "fixed_address_publication_v1" and (
            server_argv.count("--lightcone-fixed-address-publication-graph") != 1
            or _server_flag_value(server_argv, "--lightcone-graph-batch-sizes") != "1"
            or server_argv.count("--lightcone-disable-graph-eager-fallback") != 1
            or "--disable-cuda-graph" in server_argv
            or _server_flag_value(server_argv, "--cuda-graph-backend-decode") != "full"
            or _positive_server_int(server_argv, "--cuda-graph-max-bs-decode") != 1
            or _server_flag_value(server_argv, "--cuda-graph-bs-decode") != "1"
        ):
            raise ValueError("qualification fixed-address graph argv differs")
        expected_devices = (
            config.runtime.tensor_parallel_size * config.runtime.data_parallel_size
        )
        raw_inventory = _command(_GPU_INVENTORY_ARGV)
        parsed = _parse_gpu_inventory(raw_inventory)
        inventory = parsed.get("devices")
        if parsed.get("parse_error") is not None or not isinstance(inventory, list):
            raise ValueError("qualification launch lacks a valid GPU inventory")
        selected_devices = _selected_gpus(
            inventory,
            expected_count=expected_devices,
        )
        selected_gpu_uuids = tuple(device.get("uuid") for device in selected_devices)
        if selected_gpu_uuids != _config_gpu_uuids(
            config,
            expected_count=expected_devices,
        ):
            raise ValueError("qualification visible GPU UUID order differs")
        torch_runtime = _torch_runtime()
        if (
            torch_runtime.get("importable") is not True
            or torch_runtime.get("cuda_available") is not True
            or torch_runtime.get("device_count") != expected_devices
        ):
            raise ValueError("qualification lacks its exact visible CUDA ranks")
    except BaseException:
        if previous_allocator is None:
            os.environ.pop(_ALLOCATOR_ENVIRONMENT, None)
        else:
            os.environ[_ALLOCATOR_ENVIRONMENT] = previous_allocator
        raise
    return previous_allocator, selected_gpu_uuids


def _load_qualification_runtime_bridge_environment():
    """Deep-reopen the exact qualification source inherited by a child."""

    from lightcone_spec.experiments.formal_single_operator_preflight_qualification import (
        load_formal_single_operator_preflight_qualification_plan,
    )
    from lightcone_spec.runtime.native_qualification_runner import (
        NativeRuntimeQualificationAssignment,
        NativeRuntimeQualificationDispatchAuthority,
    )
    from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

    assignment_path = os.environ.get("LIGHTCONE_NATIVE_QUALIFICATION_ASSIGNMENT_PATH")
    dispatch_path = os.environ.get("LIGHTCONE_NATIVE_QUALIFICATION_DISPATCH_PATH")
    dispatch_sha256 = os.environ.get("LIGHTCONE_NATIVE_QUALIFICATION_DISPATCH_SHA256")
    trusted_path = os.environ.get(
        "LIGHTCONE_NATIVE_QUALIFICATION_TRUSTED_AUTHORITY_PATH"
    )
    trusted_sha256 = os.environ.get(
        "LIGHTCONE_NATIVE_QUALIFICATION_TRUSTED_AUTHORITY_SHA256"
    )
    if assignment_path is None:
        raise RuntimeError("qualification launch lacks dispatch-bound inputs")
    if (dispatch_path is None) != (dispatch_sha256 is None) or (
        trusted_path is None
    ) != (trusted_sha256 is None):
        raise RuntimeError("qualification launch carries an incomplete authority pair")
    assignment_binding = CanonicalJsonProofBinding.bind(assignment_path)
    if assignment_binding.semantic_sha256 != os.environ.get(
        "LIGHTCONE_NATIVE_QUALIFICATION_ASSIGNMENT_SHA256"
    ):
        raise ValueError("qualification launch environment identity differs")
    assignment = NativeRuntimeQualificationAssignment.load(
        assignment_binding.absolute_path
    )
    if assignment.schema_version == 1 and dispatch_path is not None:
        if trusted_path is not None:
            raise RuntimeError("qualification launch mixes trust authorities")
        dispatch_binding = CanonicalJsonProofBinding.bind(dispatch_path)
        if dispatch_binding.semantic_sha256 != dispatch_sha256:
            raise ValueError("qualification dispatch environment differs")
        dispatch = NativeRuntimeQualificationDispatchAuthority.from_dict(
            dispatch_binding.reopen()
        )
        dispatch.revalidate(assignment=assignment)
        qualification_authority_sha256 = dispatch_binding.semantic_sha256
    elif dispatch_path is None and trusted_path is not None:
        trusted_binding = CanonicalJsonProofBinding.bind(trusted_path)
        if trusted_binding.semantic_sha256 != trusted_sha256:
            raise ValueError("trusted qualification authority environment differs")
        trusted_value = trusted_binding.reopen()
        trusted_kind = (
            trusted_value.get("kind") if type(trusted_value) is dict else None
        )
        if trusted_kind == "formal_single_operator_preflight_qualification_plan":
            if (
                assignment.schema_version != 2
                or trusted_binding != assignment.trusted_single_operator_authority
            ):
                raise ValueError("trusted preflight authority differs from assignment")
            trusted_plan = load_formal_single_operator_preflight_qualification_plan(
                trusted_binding.absolute_path
            )
            if (
                trusted_plan.sha256 != trusted_binding.semantic_sha256
                or trusted_plan.suite_id != assignment.suite_id
                or trusted_plan.launch_manifest != assignment.launch_manifest
                or Path(trusted_plan.assignment_path)
                != Path(assignment_binding.absolute_path)
            ):
                raise ValueError("trusted qualification plan differs from assignment")
            qualification_authority_sha256 = (
                trusted_plan.dispatch_authority.semantic_sha256
            )
        elif trusted_kind == "formal_single_operator_e6_interface_fit_plan":
            if assignment.schema_version != 1 or assignment.suite_id != "nextn_tp2":
                raise ValueError("trusted E6 authority differs from assignment suite")
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                revalidate_formal_single_operator_e6_interface_fit_plan,
            )

            trusted_plan = revalidate_formal_single_operator_e6_interface_fit_plan(
                trusted_binding.absolute_path
            )
            if (
                trusted_plan.sha256 != trusted_binding.semantic_sha256
                or trusted_plan.native_assignment != assignment_binding
                or trusted_plan.launch_manifest != assignment.launch_manifest
                or trusted_plan.gpu_uuids != assignment.gpu_uuids
                or trusted_plan.topology_sha256 != assignment.topology_sha256
                or assignment.inventory_sha256 != trusted_plan.inventory.semantic_sha256
                or Path(trusted_plan.evidence_directory)
                != Path(assignment.evidence_directory)
                or Path(assignment_binding.absolute_path).parent
                != Path(assignment.evidence_directory)
            ):
                raise ValueError(
                    "trusted E6 qualification plan differs from assignment"
                )
            qualification_authority_sha256 = trusted_plan.sha256
        else:
            raise ValueError("trusted qualification authority kind is unsupported")
        if CanonicalJsonProofBinding.bind(trusted_path) != trusted_binding:
            raise ValueError("trusted qualification authority changed during replay")
    else:
        if assignment.schema_version == 2:
            raise RuntimeError(
                "trusted qualification launch lacks exact no-signature authority"
            )
        raise RuntimeError("signed qualification launch lacks exact dispatch")
    if CanonicalJsonProofBinding.bind(assignment_path) != assignment_binding:
        raise ValueError("qualification assignment changed during replay")
    return assignment, qualification_authority_sha256


def _install_qualification_runtime_bridge(
    *,
    python_root: str,
    raw_config: dict[str, object],
    verified_source=None,
) -> None:
    """Install an unsigned qualification-only provider after root dispatch replay."""

    if os.environ.get("LIGHTCONE_NATIVE_QUALIFICATION_MODE") != "1":
        return
    from lightcone_spec.config import RunConfig
    from lightcone_spec.runtime.distributed import (
        DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
    )
    from lightcone_spec.runtime.readiness import (
        NATIVE_RUNTIME_RELEASE_CAPABILITY,
        NATIVE_RUNTIME_SUITE_CAPABILITIES,
    )

    assignment, qualification_authority_sha256 = (
        _load_qualification_runtime_bridge_environment()
        if verified_source is None
        else verified_source
    )
    config = RunConfig.model_validate(raw_config)
    topology_mode = config.runtime.topology_mode
    if topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
        raise ValueError("qualification topology is not registered")

    original_path = list(sys.path)
    sys.path.insert(0, python_root)
    try:
        from sglang.srt.speculative.native_runtime_release import (
            NativeRuntimeQualificationBootstrap,
            qualification_rank_publication_hook_provider,
            register_rank_publication_hook_provider,
            register_runtime_gpu_proof_provider,
        )
    finally:
        sys.path[:] = original_path

    common = {
        "assignment_sha256": assignment.sha256,
        "dispatch_sha256": qualification_authority_sha256,
        "suite_id": assignment.suite_id,
        "source_identity_sha256": assignment.source_identity_sha256,
        "inventory_sha256": assignment.inventory_sha256,
        "hardware_envelope_sha256": assignment.hardware_envelope_sha256,
        "topology_mode": topology_mode,
        "topology_sha256": assignment.topology_sha256,
        "gpu_uuids": assignment.gpu_uuids,
        "eagle3_selector_status": assignment.eagle3_selector_status,
        "eagle3_compatibility_authority_sha256": (
            assignment.eagle3_compatibility_authority_sha256
        ),
        "eagle3_model_selector_sha256": assignment.eagle3_model_selector_sha256,
    }
    supplied: dict[str, object] = {}
    if topology_mode != "tp1_dp1":
        capability = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES[topology_mode]
        if config.runtime.distributed_release_capability_sha256 != capability.sha256:
            raise ValueError("qualification distributed capability differs")
        supplied["distributed"] = NativeRuntimeQualificationBootstrap(
            **common,
            source_capability_sha256=capability.sha256,
            backend_capabilities=(),
        )
    required = tuple(NATIVE_RUNTIME_SUITE_CAPABILITIES.get(assignment.suite_id, ()))
    if config.model.algorithm in {"DSPARK", "NEXTN", "EAGLE3"}:
        supplied["native"] = NativeRuntimeQualificationBootstrap(
            **common,
            source_capability_sha256=NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256,
            backend_capabilities=required,
        )
    register_runtime_gpu_proof_provider(lambda: dict(supplied))
    if topology_mode == "tp2_dp1":
        register_rank_publication_hook_provider(
            qualification_rank_publication_hook_provider
        )


def _bind_formal_runtime_bridge_source_environment():
    """Statically bind the production source before CUDA/runtime observation."""

    from lightcone_spec.runtime.trusted_single_operator_runtime import (
        TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT,
        bind_trusted_single_operator_runtime_authority_environment,
    )

    binding = bind_trusted_single_operator_runtime_authority_environment(os.environ)
    qualification = os.environ.get("LIGHTCONE_NATIVE_QUALIFICATION_MODE") == "1"
    if qualification and (
        binding is not None
        or any(
            name in os.environ
            for name in TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT
        )
    ):
        raise ValueError("qualification and formal runtime authority modes conflict")
    return binding


def _raw_config_requires_formal_runtime_authority(
    raw_config: dict[str, object],
) -> bool:
    """Fail closed on the source-owned fields without importing Torch/config."""

    model = raw_config.get("model")
    runtime = raw_config.get("runtime")
    if type(model) is not dict or type(runtime) is not dict:
        raise ValueError("formal runtime authority RunConfig shape differs")
    algorithm = model.get("algorithm")
    tp = runtime.get("tensor_parallel_size")
    # Schema-v3 keeps TP1/DP1 compact by omitting the default DP field.  This
    # pre-import gate must mirror that exact source encoding without importing
    # the pydantic config (and therefore Torch-adjacent runtime modules).
    dp = runtime.get("data_parallel_size", 1)
    if (
        algorithm not in {"NONE", "DFLASH", "DSPARK", "NEXTN", "EAGLE3"}
        or type(tp) is not int
        or type(dp) is not int
        or tp < 1
        or dp < 1
    ):
        raise ValueError("formal runtime authority RunConfig identity differs")
    adaptive = raw_config.get("adaptation") is not None
    return adaptive and not (algorithm == "DFLASH" and (tp, dp) == (1, 1))


def _expected_wrapper_argv(
    *,
    checkout: Path,
    compile_cache_plan: CompileCacheLaunchPlan,
    compile_cache_plan_path: str,
    run_config_path: str,
    run_config_sha256: str,
    server_argv: list[str],
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "lightcone_spec.sglang_bridge.launch",
        "--checkout",
        str(checkout),
        "--compile-cache-plan",
        str(Path(compile_cache_plan_path).resolve(strict=False)),
        "--compile-cache-plan-sha256",
        compile_cache_plan.sha256,
        "--compile-cache-key-sha256",
        compile_cache_plan.key.sha256,
        "--run-config",
        str(Path(run_config_path).resolve(strict=False)),
        "--run-config-sha256",
        run_config_sha256,
        "--",
        *server_argv,
    )


def _revalidate_qualification_runtime_bridge_source(
    *,
    raw_config: dict[str, object],
    compile_cache_plan: CompileCacheLaunchPlan,
    compile_cache_plan_path: str,
    run_config_path: str,
    run_config_sha256: str,
    checkout: Path,
    server_argv: list[str],
    selected_gpu_uuids: tuple[str, ...],
):
    """Join a qualification assignment to this exact wrapper invocation."""

    from lightcone_spec.config import RunConfig, load_run_config
    from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

    assignment, qualification_authority_sha256 = (
        _load_qualification_runtime_bridge_environment()
    )
    config = RunConfig.model_validate(raw_config)
    launch = CompileLaunchManifest.load(assignment.launch_manifest.absolute_path)
    source_config = load_run_config(launch.run_config_path)
    expected_wrapper_argv = _expected_wrapper_argv(
        checkout=checkout,
        compile_cache_plan=compile_cache_plan,
        compile_cache_plan_path=compile_cache_plan_path,
        run_config_path=run_config_path,
        run_config_sha256=run_config_sha256,
        server_argv=server_argv,
    )
    if (
        source_config != config
        or Path(launch.run_config_path) != Path(run_config_path).resolve(strict=False)
        or launch.run_config_semantic_sha256 != run_config_sha256
        or Path(launch.compile_cache_plan_path)
        != Path(compile_cache_plan_path).resolve(strict=False)
        or launch.compile_cache_plan_sha256 != compile_cache_plan.sha256
        or Path(launch.patched_sglang_checkout) != checkout
        or launch.server_argv != expected_wrapper_argv
        or launch.gpu_uuids != assignment.gpu_uuids
        or launch.gpu_uuids != selected_gpu_uuids
    ):
        raise ValueError("qualification authority differs from child invocation")
    return assignment, qualification_authority_sha256


def _revalidate_formal_runtime_bridge_source(
    *,
    source_binding,
    raw_config: dict[str, object],
    compile_cache_plan: CompileCacheLaunchPlan,
    compile_cache_plan_path: str,
    run_config_path: str,
    run_config_sha256: str,
    checkout: Path,
    server_argv: list[str],
    selected_gpu_uuids: tuple[str, ...],
):
    """Deep-replay one trusted authority against the exact child invocation."""

    from lightcone_spec.config import RunConfig, load_run_config
    from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
    from lightcone_spec.runtime.trusted_single_operator_runtime import (
        verify_trusted_single_operator_runtime_authority_source,
    )

    config = RunConfig.model_validate(raw_config)
    required = _raw_config_requires_formal_runtime_authority(raw_config)
    if source_binding is None:
        if required:
            raise RuntimeError("adaptive runtime lacks trusted empirical authority")
        return None
    if not required:
        raise ValueError("allocation-free runtime carries adaptive GPU authority")
    source, tokens = verify_trusted_single_operator_runtime_authority_source(
        source_binding.absolute_path,
        expected_source_binding=source_binding,
    )
    launch = CompileLaunchManifest.load(source.launch_manifest.absolute_path)
    source_config = load_run_config(launch.run_config_path)
    expected_wrapper_argv = _expected_wrapper_argv(
        checkout=checkout,
        compile_cache_plan=compile_cache_plan,
        compile_cache_plan_path=compile_cache_plan_path,
        run_config_path=run_config_path,
        run_config_sha256=run_config_sha256,
        server_argv=server_argv,
    )
    if (
        source_config != config
        or source.algorithm != config.model.algorithm
        or source.topology_mode != config.runtime.topology_mode
        or Path(launch.run_config_path) != Path(run_config_path).resolve(strict=False)
        or launch.run_config_semantic_sha256 != run_config_sha256
        or Path(launch.compile_cache_plan_path)
        != Path(compile_cache_plan_path).resolve(strict=False)
        or launch.compile_cache_plan_sha256 != compile_cache_plan.sha256
        or Path(launch.patched_sglang_checkout) != checkout
        or launch.server_argv != expected_wrapper_argv
        or launch.gpu_uuids != source.gpu_uuids
        or launch.gpu_uuids != selected_gpu_uuids
        or tuple(token.role for token in tokens)
        != tuple(role.role for role in source.roles)
    ):
        raise ValueError("formal runtime authority differs from child invocation")
    return source, tokens


def _install_formal_runtime_bridge(*, python_root: str, verified_source) -> None:
    """Install only deeply-issued trusted tokens into the patched runtime."""

    if verified_source is None:
        return
    source, tokens = verified_source
    original_path = list(sys.path)
    sys.path.insert(0, python_root)
    try:
        from sglang.srt.speculative.native_runtime_release import (
            formal_rank_publication_hook_provider,
            register_rank_publication_hook_provider,
            register_runtime_gpu_proof_provider,
        )
    finally:
        sys.path[:] = original_path
    supplied = {token.role: token for token in tokens}
    register_runtime_gpu_proof_provider(lambda: dict(supplied))
    if source.topology_mode == "tp2_dp1":
        register_rank_publication_hook_provider(formal_rank_publication_hook_provider)


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
    if _ADAPTATION_CONFIG_SHA256_ENVIRONMENT in os.environ:
        raise ValueError("caller injected an adaptation payload authority")
    formal_source_binding = _bind_formal_runtime_bridge_source_environment()
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
        qualification_mode = (
            os.environ.get("LIGHTCONE_NATIVE_QUALIFICATION_MODE") == "1"
        )
        if (
            not qualification_mode
            and formal_source_binding is None
            and _raw_config_requires_formal_runtime_authority(config)
        ):
            raise RuntimeError("adaptive runtime lacks trusted empirical authority")
        if plan.cache_mode != "build":
            raise RuntimeError(
                "diagnostic_compile_cache_reuse_requires_model_content_authority"
            )
        checkout = verify_patched_checkout(args.checkout)
        previous_allocator, selected_gpu_uuids = (
            _validate_qualification_runtime_environment(config, server_argv)
            if qualification_mode
            else _validate_compile_runtime_environment(plan, config, server_argv)
        )
        verified_qualification_source = (
            _revalidate_qualification_runtime_bridge_source(
                raw_config=config,
                compile_cache_plan=plan,
                compile_cache_plan_path=args.compile_cache_plan,
                run_config_path=args.run_config,
                run_config_sha256=args.run_config_sha256,
                checkout=checkout,
                server_argv=server_argv,
                selected_gpu_uuids=selected_gpu_uuids,
            )
            if qualification_mode
            else None
        )
        verified_formal_source = (
            None
            if qualification_mode
            else _revalidate_formal_runtime_bridge_source(
                source_binding=formal_source_binding,
                raw_config=config,
                compile_cache_plan=plan,
                compile_cache_plan_path=args.compile_cache_plan,
                run_config_path=args.run_config,
                run_config_sha256=args.run_config_sha256,
                checkout=checkout,
                server_argv=server_argv,
                selected_gpu_uuids=selected_gpu_uuids,
            )
        )
        adaptation_config_binding = _bind_runtime_adaptation_config(config, server_argv)
        if adaptation_config_binding is not None:
            os.environ[_ADAPTATION_CONFIG_SHA256_ENVIRONMENT] = (
                adaptation_config_binding.semantic_sha256
            )
    except BaseException:
        os.environ.pop(_ADAPTATION_CONFIG_SHA256_ENVIRONMENT, None)
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
        os.environ.pop(_ADAPTATION_CONFIG_SHA256_ENVIRONMENT, None)
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
    managed_environment[_ADAPTATION_CONFIG_SHA256_ENVIRONMENT] = None
    try:
        cache_environment = session.environment(os.environ)
        for name in cache_environment.keys() & managed_environment.keys():
            os.environ[name] = cache_environment[name]
        # A verified disposable checkout is source, never a bytecode cache.
        if _bind_runtime_adaptation_config(config, server_argv) != (
            adaptation_config_binding
        ):
            raise ValueError("adaptation payload changed before SGLang import")
        sys.dont_write_bytecode = True
        sys.path.insert(0, python_root)
        _install_qualification_runtime_bridge(
            python_root=python_root,
            raw_config=config,
            verified_source=verified_qualification_source,
        )
        _install_formal_runtime_bridge(
            python_root=python_root,
            verified_source=verified_formal_source,
        )
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
