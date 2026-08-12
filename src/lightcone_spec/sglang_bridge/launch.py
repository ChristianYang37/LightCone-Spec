"""Launch SGLang only from a verified disposable patched checkout."""

from __future__ import annotations

import argparse
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from lightcone_spec.runtime.compile_cache import (
    COMPILE_CACHE_ENVIRONMENT_VARIABLES,
    CompileCacheLaunchPlan,
    CompileOnlyAssignmentContract,
    require_release_compile_only_assignment,
    start_compile_cache_launch,
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


def _command_output(executable: str, *arguments: str) -> str | None:
    path = shutil.which(executable)
    if path is None:
        return None
    result = subprocess.run(
        (path, *arguments),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_blackwell_jit_toolchain() -> None:
    """Require the toolkit version needed to JIT kernels for SM 12.x."""
    capabilities = _command_output(
        "nvidia-smi",
        "--query-gpu=compute_cap",
        "--format=csv,noheader,nounits",
    )
    if capabilities is None:
        return
    parsed = []
    for line in capabilities.splitlines():
        match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", line)
        if match is None:
            raise RuntimeError("nvidia-smi returned an invalid compute capability")
        parsed.append((int(match.group(1)), int(match.group(2))))
    if not parsed or max(parsed) < (12, 0):
        return
    nvcc = _command_output("nvcc", "--version")
    match = re.search(r"\brelease\s+(\d+)\.(\d+)\b", nvcc or "")
    if match is None or (int(match.group(1)), int(match.group(2))) < (12, 9):
        raise RuntimeError(
            "SM 12.x requires CUDA toolkit >= 12.9 on PATH; set CUDA_HOME, "
            "prepend $CUDA_HOME/bin to PATH, and add $CUDA_HOME/lib64 to "
            "LD_LIBRARY_PATH before launching SGLang"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lightcone-sglang-launch")
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--compile-cache-plan", required=True)
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
    checkout = verify_patched_checkout(args.checkout)
    plan = CompileCacheLaunchPlan.load(args.compile_cache_plan)
    session = start_compile_cache_launch(plan)

    python_root = str(checkout / "python")
    original_argv = sys.argv
    original_path = list(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    managed_environment = {
        name: os.environ.get(name)
        for name in (*COMPILE_CACHE_ENVIRONMENT_VARIABLES, "PATH")
    }
    try:
        cache_environment = session.environment(os.environ)
        for name in cache_environment.keys() & managed_environment.keys():
            os.environ[name] = cache_environment[name]
        _bind_interpreter_tools()
        _validate_blackwell_jit_toolchain()
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
