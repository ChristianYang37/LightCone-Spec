"""Read-only host preflight for native SGLang/LightCone deployments."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path


SUPPORTED_CUDA = ((13, 0, "cu130"), (12, 9, "cu129"))


def _cuda_version(output: str | None, pattern: str) -> tuple[int, int] | None:
    match = re.search(pattern, output or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def _command(argv: list[str], timeout: int = 15) -> dict:
    exe = shutil.which(argv[0])
    if exe is None:
        return {"available": False, "output": None}
    try:
        proc = subprocess.run(
            [exe, *argv[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": True, "ok": False, "error": str(exc)}
    output = (proc.stdout or proc.stderr).strip()
    return {
        "available": True,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "output": output,
    }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_cuda_toolkit(runtime_root: str | Path) -> dict:
    """Prefer an isolated runtime toolkit over a stale system symlink."""
    root = Path(runtime_root).expanduser().resolve()
    candidates = sorted(root.glob("cuda-*"), reverse=True)
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(Path(cuda_home))
    system_nvcc = shutil.which("nvcc")
    if system_nvcc:
        candidates.append(Path(system_nvcc).resolve().parent.parent)
    seen = set()
    toolkits = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        nvcc = candidate / "bin" / "nvcc"
        if not nvcc.is_file():
            continue
        command = _command([str(nvcc), "--version"])
        version = _cuda_version(command.get("output"), r"release\s+(\d+)\.(\d+)")
        if command.get("ok") and version:
            toolkits.append((version, candidate, nvcc, command))
    if not toolkits:
        return {
            "available": False,
            "root": None,
            "nvcc": None,
            "version": None,
            "supported": False,
        }
    version, candidate, nvcc, command = max(toolkits, key=lambda item: item[0])
    supported = any(version >= required[:2] for required in SUPPORTED_CUDA)
    return {
        "available": True,
        "root": str(candidate),
        "nvcc": str(nvcc),
        "version": f"{version[0]}.{version[1]}",
        "version_tuple": version,
        "supported": supported,
        "command": command,
    }


def configure_runtime_threads() -> int:
    """Normalize invalid image-level OpenMP settings before native imports."""
    omp_value = os.environ.get("OMP_NUM_THREADS", "")
    if not omp_value.isdigit() or int(omp_value) < 1:
        # Some GPU images export OMP_NUM_THREADS=0, which libgomp rejects and
        # can leave preprocessing/detokenization effectively misconfigured.
        os.environ["OMP_NUM_THREADS"] = str(min(os.cpu_count() or 1, 8))
    return int(os.environ["OMP_NUM_THREADS"])


def configure_runtime_cuda_toolkit(runtime_root: str | Path) -> dict:
    """Expose the isolated toolkit to this process and inherited workers."""
    omp_num_threads = configure_runtime_threads()
    toolkit = resolve_cuda_toolkit(runtime_root)
    toolkit["omp_num_threads"] = omp_num_threads
    if not toolkit["supported"]:
        return toolkit
    root = str(toolkit["root"])
    os.environ["CUDA_HOME"] = root
    for variable, paths in (
        ("PATH", [str(Path(root) / "bin")]),
        ("LD_LIBRARY_PATH", [str(Path(root) / "lib"), str(Path(root) / "lib64")]),
    ):
        existing = [value for value in os.environ.get(variable, "").split(":") if value]
        os.environ[variable] = ":".join(paths + [value for value in existing if value not in paths])
    return toolkit


def _network_probe() -> dict:
    result = {"dns": False, "huggingface_https": False}
    try:
        socket.getaddrinfo("huggingface.co", 443)
        result["dns"] = True
    except OSError as exc:
        result["error"] = f"dns: {exc}"
        return result
    try:
        req = urllib.request.Request(
            "https://huggingface.co/", method="HEAD", headers={"User-Agent": "lightcone-doctor/1"}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            result["huggingface_https"] = 200 <= response.status < 500
    except Exception as exc:
        result["error"] = f"https: {exc}"
    return result


def collect_doctor_report(
    runtime_root: str | Path,
    *,
    min_free_gib: int = 80,
    check_network: bool = True,
) -> dict:
    root = Path(runtime_root).expanduser().resolve()
    disk_target = root if root.exists() else root.parent
    while not disk_target.exists() and disk_target != disk_target.parent:
        disk_target = disk_target.parent
    usage = shutil.disk_usage(disk_target)
    nvidia = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    nvidia_header = _command(["nvidia-smi"], timeout=20)
    driver_cuda = _cuda_version(
        nvidia_header.get("output"), r"CUDA Version:\s*(\d+)\.(\d+)"
    )
    toolkit = resolve_cuda_toolkit(root)
    nvcc = toolkit.get("command") or {"available": False, "output": None}
    toolkit_cuda = toolkit.get("version_tuple")
    usable_cuda = min(driver_cuda, toolkit_cuda) if driver_cuda and toolkit_cuda else None
    supported_cuda = [
        tag for major, minor, tag in SUPPORTED_CUDA
        if usable_cuda is not None and usable_cuda >= (major, minor)
    ]
    torch_report = {
        "installed": False,
        "version": _version("torch"),
        "cuda_available": False,
        "cuda_runtime": None,
        "devices": [],
    }
    try:
        import torch

        torch_report.update(
            {
                "installed": True,
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "devices": [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": list(torch.cuda.get_device_capability(i)),
                        "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                    }
                    for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else [],
            }
        )
    except Exception as exc:
        torch_report["import_error"] = str(exc)

    free_ok = usage.free >= min_free_gib * (1 << 30)
    gpu_visible = bool(nvidia.get("ok"))
    report = {
        "schema_version": 1,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "runtime_root": str(root),
        "disk": {
            "probe_path": str(disk_target),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
            "minimum_free_bytes": min_free_gib * (1 << 30),
            "ok": free_ok,
        },
        "commands": {
            "nvidia_smi_query": nvidia,
            "nvidia_smi": nvidia_header,
            "nvcc": nvcc,
            "gcc": _command(["gcc", "--version"]),
            "gxx": _command(["g++", "--version"]),
            "git": _command(["git", "--version"]),
            "rustc": _command(["rustc", "--version"]),
            "cargo": _command(["cargo", "--version"]),
            "uv": _command(["uv", "--version"]),
        },
        "packages": {
            "torch": torch_report,
            "triton": _version("triton"),
            "sglang": _version("sglang"),
            "flashinfer_python": _version("flashinfer-python"),
            "huggingface_hub": _version("huggingface-hub"),
            "nvidia_ml_py": _version("nvidia-ml-py"),
        },
        "network": _network_probe() if check_network else {"skipped": True},
        "cuda_compatibility": {
            "driver_max": (
                f"{driver_cuda[0]}.{driver_cuda[1]}" if driver_cuda else None
            ),
            "toolkit": (
                f"{toolkit_cuda[0]}.{toolkit_cuda[1]}" if toolkit_cuda else None
            ),
            "toolkit_root": toolkit.get("root"),
            "supported_tags": supported_cuda,
            "selected": supported_cuda[0] if supported_cuda else None,
            "sm120_jit_compatible": bool(
                toolkit_cuda is not None
                and toolkit_cuda >= (12, 9)
            ) if any(
                int(device["capability"][0]) >= 12
                for device in torch_report["devices"]
            ) else True,
        },
    }
    network_ok = not check_network or (
        report["network"].get("dns")
        and report["network"].get("huggingface_https")
    )
    commands = report["commands"]
    toolchain_ok = all(
        bool(commands[name].get("ok"))
        for name in ("nvcc", "gcc", "gxx", "git", "rustc", "cargo")
    )
    python_ok = sys.version_info[:2] == (3, 12) or bool(
        commands["uv"].get("ok")
    )
    report["ready_for_install"] = bool(
        gpu_visible
        and free_ok
        and network_ok
        and toolchain_ok
        and python_ok
        and supported_cuda
    )
    report["ready_for_gpu_debug"] = bool(
        report["ready_for_install"]
        and report["cuda_compatibility"]["sm120_jit_compatible"]
        and torch_report["cuda_available"]
        and report["packages"]["sglang"]
        and report["packages"]["flashinfer_python"]
        and report["packages"]["nvidia_ml_py"]
    )
    return report


def doctor_json(*args, **kwargs) -> str:
    return json.dumps(collect_doctor_report(*args, **kwargs), indent=2, sort_keys=True)
