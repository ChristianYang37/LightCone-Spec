"""Read-only, fail-closed industrial runtime readiness report."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lightcone_spec import (
    PINNED_SGLANG_COMMIT,
    PINNED_SGLANG_PATCH_COUNT,
    PINNED_SGLANG_TREE,
)

RUNTIME_MANIFEST = Path("manifests/runtime/industrial_compatibility_v1.json")
_STATUS = frozenset(("PASS", "FAIL", "UNKNOWN"))
_UNSUPPORTED_MODE_GATES = (
    ("multi_node", "release_multi_node_unsupported"),
    ("adaptive_eagle", "release_adaptive_backend_unsupported"),
    (
        "adaptive_dflash_dspark",
        "release_combined_backend_unsupported",
    ),
)
_PENDING_DYNAMIC_GPU_PROOF_MODE_GATES = (
    ("tp2", "distributed_tp2_gpu_qualification_required"),
    ("dp2", "distributed_dp2_gpu_qualification_required"),
    ("adaptive_dspark", "dspark_gpu_qualification_required"),
    ("adaptive_nextn", "nextn_gpu_qualification_required"),
    (
        "adaptive_chronobelief",
        "chronobelief_gpu_qualification_required",
    ),
    ("native_itl", "native_itl_gpu_qualification_required"),
    (
        "fixed_address_publication_graph",
        "fixed_address_graph_gpu_qualification_required",
    ),
    ("session_reuse", "session_reset_gpu_qualification_required"),
)
_EXTERNAL_AUTHORITY_MODE_GATES = (
    (
        "static_speculative",
        "release_trusted_terminal_attester_unavailable",
    ),
    ("tts", "release_trusted_terminal_attester_unavailable"),
    ("l0", "release_trusted_terminal_attester_unavailable"),
    (
        "adaptive_eagle3",
        "release_official_eagle3_compatibility_authority_unavailable",
    ),
    (
        "quota_shadow_backend_acquisition",
        "release_quota_shadow_backend_acquisition_unavailable",
    ),
)


def _command(args: list[str]) -> str | None:
    executable = shutil.which(args[0])
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, *args[1:]],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "-C", str(root), *arguments),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=15,
    )


def _project_tree(root: Path) -> dict[str, Any]:
    try:
        head = _git(root, "rev-parse", "HEAD")
        tree = _git(root, "rev-parse", "HEAD^{tree}")
        top = _git(root, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.TimeoutExpired):
        return {
            "path": str(root),
            "is_git_checkout": False,
            "root_matches_toplevel": False,
            "head": None,
            "tree": None,
            "dirty": None,
        }
    if head.returncode != 0 or tree.returncode != 0 or top.returncode != 0:
        return {
            "path": str(root),
            "is_git_checkout": False,
            "root_matches_toplevel": False,
            "head": None,
            "tree": None,
            "dirty": None,
        }
    try:
        status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.TimeoutExpired):
        status = subprocess.CompletedProcess((), 1, "")
    try:
        top_level = Path(top.stdout.strip()).resolve()
    except (OSError, RuntimeError):
        top_level = Path(top.stdout.strip())
    return {
        "path": str(root),
        "is_git_checkout": True,
        "root_matches_toplevel": top_level == root,
        "head": head.stdout.strip(),
        "tree": tree.stdout.strip(),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _project_runtime_source(root: Path) -> dict[str, Any]:
    """Digest exact present LightCone Python bytes from one source checkout."""

    try:
        listed = _git(root, "ls-files", "-z", "--", "src/lightcone_spec")
        untracked = _git(
            root,
            "ls-files",
            "-z",
            "--others",
            "--exclude-standard",
            "--",
            "src/lightcone_spec",
        )
    except (OSError, subprocess.TimeoutExpired):
        listed = subprocess.CompletedProcess((), 1, "")
        untracked = subprocess.CompletedProcess((), 1, "")
    if listed.returncode != 0 or untracked.returncode != 0:
        return {
            "schema_version": 1,
            "kind": "lightcone_project_runtime_source",
            "valid": False,
            "file_count": 0,
            "total_size_bytes": 0,
            "content_sha256": None,
        }
    names = tuple(
        sorted(
            {
                name
                for output in (listed.stdout, untracked.stdout)
                for name in output.split("\0")
                if name
            }
        )
    )
    if not names or len(names) > 2_048:
        return {
            "schema_version": 1,
            "kind": "lightcone_project_runtime_source",
            "valid": False,
            "file_count": len(names),
            "total_size_bytes": 0,
            "content_sha256": None,
        }
    rows: list[dict[str, object]] = []
    total_size = 0
    for relative_text in names:
        relative = Path(relative_text)
        unresolved_path = root / relative
        fd: int | None = None
        try:
            path = unresolved_path.resolve()
            path.relative_to(root)
            if unresolved_path.is_symlink():
                raise ValueError("project runtime source leaf is a symlink")
            flags = os.O_RDONLY
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(unresolved_path, flags)
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size > 8 * 1024 * 1024
            ):
                raise ValueError("project runtime source leaf is not bounded")
            body = bytearray()
            while len(body) <= 8 * 1024 * 1024:
                chunk = os.read(
                    fd,
                    min(1024 * 1024, 8 * 1024 * 1024 + 1 - len(body)),
                )
                if not chunk:
                    break
                body.extend(chunk)
            after = os.fstat(fd)
            current = unresolved_path.lstat()
        except (OSError, ValueError):
            rows = []
            break
        finally:
            if fd is not None:
                os.close(fd)
        if (
            len(body) > 8 * 1024 * 1024
            or len(body) != before.st_size
            or _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(current)
            or not path.is_file()
        ):
            rows = []
            break
        total_size += len(body)
        if total_size > 128 * 1024 * 1024:
            rows = []
            break
        rows.append(
            {
                "path": relative.as_posix(),
                "size_bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    valid = len(rows) == len(names)
    content_sha256 = (
        hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if valid
        else None
    )
    return {
        "schema_version": 1,
        "kind": "lightcone_project_runtime_source",
        "valid": valid,
        "file_count": len(rows),
        "total_size_bytes": total_size if valid else 0,
        "content_sha256": content_sha256,
    }


def _require_project_runtime_source_identity(
    report: Mapping[str, Any],
    *,
    executing_file: Path,
) -> Path:
    """Reopen the exact clean LightCone source identity bound by a doctor report."""

    roots = report.get("roots")
    project_source = report.get("project_source_tree")
    runtime_source = report.get("project_runtime_source")
    if (
        not isinstance(roots, Mapping)
        or not isinstance(project_source, Mapping)
        or not isinstance(runtime_source, Mapping)
    ):
        raise TypeError("doctor report lacks the LightCone source identity")
    project_root_text = roots.get("project")
    if not isinstance(project_root_text, str):
        raise TypeError("doctor project root must be a string")
    project_root = Path(project_root_text)
    try:
        resolved_root = project_root.resolve(strict=True)
        Path(executing_file).resolve(strict=True).relative_to(
            resolved_root / "src" / "lightcone_spec"
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "doctor project root does not own the executing LightCone source"
        ) from exc
    observed_tree = _project_tree(resolved_root)
    observed_runtime_source = _project_runtime_source(resolved_root)
    expected_tree = {
        "path": str(resolved_root),
        "is_git_checkout": True,
        "root_matches_toplevel": True,
        "head": project_source.get("head"),
        "tree": project_source.get("tree"),
    }
    observed_tree_identity = {name: observed_tree.get(name) for name in expected_tree}
    if (
        not project_root.is_absolute()
        or project_root_text != str(resolved_root)
        or {name: project_source.get(name) for name in expected_tree} != expected_tree
        or observed_tree_identity != expected_tree
        or observed_tree.get("dirty") is not False
        or project_source.get("dirty") is not False
        or not re.fullmatch(r"[0-9a-f]{40}", str(project_source.get("head")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(project_source.get("tree")))
        or runtime_source.get("valid") is not True
        or not re.fullmatch(r"[0-9a-f]{64}", str(runtime_source.get("content_sha256")))
        or dict(runtime_source) != observed_runtime_source
    ):
        raise ValueError("doctor LightCone source identity is stale or incomplete")
    return resolved_root


def _source_tree(root: Path) -> dict[str, Any]:
    """Inspect the patched SGLang checkout without importing it."""

    result = _project_tree(root)
    if result["is_git_checkout"] is not True:
        return {
            **result,
            "pinned_ancestor": False,
            "patch_commits": None,
        }
    try:
        ancestor = _git(
            root,
            "merge-base",
            "--is-ancestor",
            PINNED_SGLANG_COMMIT,
            "HEAD",
        )
        count = _git(
            root,
            "rev-list",
            "--count",
            f"{PINNED_SGLANG_COMMIT}..HEAD",
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            **result,
            "pinned_ancestor": False,
            "patch_commits": None,
        }
    return {
        **result,
        "pinned_ancestor": ancestor.returncode == 0,
        "patch_commits": (int(count.stdout.strip()) if count.returncode == 0 else None),
    }


def _physical_memory_bytes() -> int | None:
    """Return host RAM without importing a platform-specific dependency."""

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    total = pages * page_size
    return total if total > 0 else None


def _disk_report(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(root),
        "total_bytes": None,
        "used_bytes": None,
        "free_bytes": None,
        "total_inodes": None,
        "free_inodes": None,
        "error": None,
    }
    try:
        usage = shutil.disk_usage(root)
    except OSError as error:
        result["error"] = type(error).__name__
        return result
    result.update(
        {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    )
    try:
        filesystem = os.statvfs(root)
    except (AttributeError, OSError):
        return result
    result["total_inodes"] = int(filesystem.f_files)
    result["free_inodes"] = int(filesystem.f_favail)
    return result


def _dns_probe(host: str) -> dict[str, Any]:
    """Report DNS reachability only; never send application traffic."""

    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as error:
        return {
            "host": host,
            "resolved": False,
            "address_count": 0,
            "error": type(error).__name__,
        }
    unique = {
        str(address[4][0])
        for address in addresses
        if isinstance(address[4], tuple) and address[4]
    }
    return {
        "host": host,
        "resolved": bool(unique),
        "address_count": len(unique),
        "error": None,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_spec(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except (AttributeError, ImportError, ValueError):
        spec = None
    return {
        "importable": spec is not None,
        "origin": None if spec is None else spec.origin,
    }


def _torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError, RuntimeError) as error:
        return {
            "importable": False,
            "import_error": type(error).__name__,
            "version": None,
            "cuda_build": None,
            "cuda_available": False,
            "device_count": 0,
        }
    return {
        "importable": True,
        "import_error": None,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }


def _parse_gpu_inventory(value: str | None) -> dict[str, Any]:
    if not value:
        return {"devices": [], "parse_error": None}
    devices: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(value)):
        fields = [field.strip() for field in row]
        if len(fields) != 6:
            return {"devices": devices, "parse_error": "unexpected_column_count"}
        try:
            memory_mib = int(fields[2])
        except ValueError:
            return {"devices": devices, "parse_error": "invalid_memory_total"}
        devices.append(
            {
                "uuid": fields[0],
                "name": fields[1],
                "memory_total_mib": memory_mib,
                "driver_version": fields[3],
                "compute_capability": fields[4],
                "pci_bus_id": fields[5],
            }
        )
    return {"devices": devices, "parse_error": None}


def _parse_topology(value: str | None) -> dict[str, Any]:
    if not value:
        return {
            "gpu_rows": [],
            "pairs": [],
            "parse_error": None,
        }
    # Recent NVIDIA drivers underline the topology header even when stdout is
    # redirected, for example ``\x1b[4mGPU0 ... GPU NUMA ID\x1b[0m``.  Strip
    # only SGR styling; every other control sequence remains an ambiguous
    # external-input failure rather than being silently normalized.
    normalized = re.sub(r"\x1b\[[0-9;]*m", "", value)
    if any(
        (ord(character) < 32 and character not in "\t\n\r")
        or ord(character) == 127
        or 128 <= ord(character) <= 159
        for character in normalized
    ):
        return {
            "gpu_rows": [],
            "pairs": [],
            "parse_error": "unsupported_control_character",
        }
    lines = [line.split() for line in normalized.splitlines() if line.strip()]
    header = next(
        (
            line
            for line in lines
            if line and line[0] == "GPU0" and re.fullmatch(r"GPU\d+", line[0])
        ),
        None,
    )
    rows = {
        line[0]: line for line in lines if line and re.fullmatch(r"GPU\d+", line[0])
    }
    if header is None:
        return {
            "gpu_rows": sorted(rows, key=lambda name: int(name[3:])),
            "pairs": [],
            "parse_error": "missing_gpu_matrix",
        }
    header_gpus = tuple(token for token in header if re.fullmatch(r"GPU\d+", token))
    expected_gpus = tuple(f"GPU{index}" for index in range(len(header_gpus)))
    if not header_gpus or header_gpus != expected_gpus or set(rows) != set(header_gpus):
        return {
            "gpu_rows": sorted(rows, key=lambda name: int(name[3:])),
            "pairs": [],
            "parse_error": "inconsistent_gpu_matrix",
        }
    columns = {gpu: header.index(gpu) + 1 for gpu in header_gpus}
    if any(len(rows[gpu]) <= max(columns.values()) for gpu in header_gpus):
        return {
            "gpu_rows": list(header_gpus),
            "pairs": [],
            "parse_error": "truncated_gpu_matrix",
        }
    pairs = []
    for left_index, left in enumerate(header_gpus):
        if rows[left][columns[left]] != "X":
            return {
                "gpu_rows": list(header_gpus),
                "pairs": [],
                "parse_error": "invalid_gpu_diagonal",
            }
        for right in header_gpus[left_index + 1 :]:
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "link": rows[left][columns[right]],
                    "reciprocal_link": rows[right][columns[left]],
                }
            )
    return {
        "gpu_rows": list(header_gpus),
        "pairs": pairs,
        "parse_error": None,
    }


def _collect_facts(project_root: Path, sglang_root: Path) -> dict[str, Any]:
    packages = {
        name: _package_version(name)
        for name in (
            "torch",
            "triton",
            "sglang",
            "flashinfer-python",
            "nvidia-cutlass-dsl-libs-cu13",
        )
    }
    nvidia_query = _command(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    topology_raw = _command(["nvidia-smi", "topo", "-m"])
    return {
        "python": {
            "version": platform.python_version(),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "detail": sys.version,
            "executable": sys.executable,
        },
        "host": {
            "system": platform.system(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
            "physical_memory_bytes": _physical_memory_bytes(),
            "cpu_detail": _command(["lscpu"])
            or _command(["sysctl", "-n", "machdep.cpu.brand_string"]),
        },
        "project_source_tree": _project_tree(project_root),
        "project_runtime_source": _project_runtime_source(project_root),
        "source_tree": _source_tree(sglang_root),
        "disk": _disk_report(project_root),
        "network": {
            host: _dns_probe(host)
            for host in (
                "huggingface.co",
                "modelscope.cn",
                "pypi.tuna.tsinghua.edu.cn",
            )
        },
        "gpu": {
            "inventory_raw": nvidia_query,
            "inventory": _parse_gpu_inventory(nvidia_query),
            "topology_raw": topology_raw,
            "topology": _parse_topology(topology_raw),
            "torch": _torch_runtime(),
        },
        "commands": {
            "nvidia_smi": nvidia_query,
            "nvcc": _command(["nvcc", "--version"]),
            "compiler": _command(["c++", "--version"]),
            "ninja": _command(["ninja", "--version"]),
            "git": _command(["git", "--version"]),
        },
        "packages": packages,
        "sglang_import": _module_spec("sglang"),
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _manifest_structure_error(value: Mapping[str, Any]) -> str | None:
    try:
        if value["schema_version"] != 1:
            return "unsupported_schema_version"
        if value["name"] != "lightcone-industrial-runtime-compatibility":
            return "unexpected_manifest_name"
        if value["sglang"]["upstream_commit"] != PINNED_SGLANG_COMMIT:
            return "sglang_upstream_constant_mismatch"
        if value["sglang"]["expected_tree"] != PINNED_SGLANG_TREE:
            return "sglang_tree_constant_mismatch"
        if value["sglang"]["expected_patch_count"] != PINNED_SGLANG_PATCH_COUNT:
            return "sglang_patch_count_constant_mismatch"
        for path in (
            ("host", "python_minor"),
            ("host", "operating_system"),
            ("packages", "torch"),
            ("packages", "triton"),
            ("packages", "flashinfer_python"),
            ("cuda", "torch_build"),
            ("cuda", "toolkit_release"),
            ("system", "minimum_driver"),
            ("sglang", "patches"),
            ("gpu", "minimum_count"),
            ("gpu", "reference_count"),
            ("gpu", "same_host_required"),
            ("gpu", "model"),
            ("gpu", "compute_capability"),
            ("gpu", "minimum_memory_mib"),
            ("gpu", "nominal_memory_gib"),
            ("gpu", "accepted_topology_links"),
            ("disk", "minimum_total_bytes"),
            ("disk", "minimum_free_bytes"),
            ("network", "artifact_dns_any_of"),
            ("network", "package_index_dns"),
            ("release_capabilities", "supported"),
            ("release_capabilities", "implemented_pending_dynamic_gpu_proof"),
            ("release_capabilities", "external_authority_required"),
            ("release_capabilities", "unsupported"),
        ):
            if value[path[0]][path[1]] is None:
                return f"missing_{path[0]}_{path[1]}"
        flashinfer = value["packages"]["flashinfer_python"]
        for key in (
            "distribution",
            "version",
            "cuda_extra",
            "cuda_marker_distribution",
        ):
            if not flashinfer[key]:
                return f"missing_packages_flashinfer_python_{key}"
        capabilities = value["release_capabilities"]
        pending = tuple(
            (row["mode"], row["reason_code"])
            for row in capabilities["implemented_pending_dynamic_gpu_proof"]
        )
        external = tuple(
            (row["mode"], row["reason_code"])
            for row in capabilities["external_authority_required"]
        )
        unsupported = tuple(
            (row["mode"], row["reason_code"]) for row in capabilities["unsupported"]
        )
        if (
            pending != _PENDING_DYNAMIC_GPU_PROOF_MODE_GATES
            or external != _EXTERNAL_AUTHORITY_MODE_GATES
            or unsupported != _UNSUPPORTED_MODE_GATES
        ):
            return "release_mode_gates_mismatch"
        minimum_count = value["gpu"]["minimum_count"]
        reference_count = value["gpu"]["reference_count"]
        if (
            not isinstance(minimum_count, int)
            or isinstance(minimum_count, bool)
            or minimum_count < 1
            or not isinstance(reference_count, int)
            or isinstance(reference_count, bool)
            or reference_count < minimum_count
            or value["gpu"]["same_host_required"] is not True
        ):
            return "invalid_gpu_pool_cardinality_contract"
    except (KeyError, TypeError):
        return "missing_or_invalid_required_field"
    return None


def _load_manifest(project_root: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    path = project_root / RUNTIME_MANIFEST
    sidecar = Path(f"{path}.sha256")
    report: dict[str, Any] = {
        "path": str(path),
        "sidecar_path": str(sidecar),
        "sha256": None,
        "sidecar_sha256": None,
        "valid": False,
        "error": None,
    }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = sidecar.read_text(encoding="utf-8").strip()
    except (OSError, json.JSONDecodeError) as error:
        report["error"] = type(error).__name__
        return None, report
    if not isinstance(value, dict):
        report["error"] = "manifest_not_object"
        return None, report
    actual = _canonical_sha256(value)
    structure_error = _manifest_structure_error(value)
    sidecar_valid = bool(re.fullmatch(r"[0-9a-f]{64}", expected)) and (
        actual == expected
    )
    valid = sidecar_valid and structure_error is None
    report.update(
        {
            "sha256": actual,
            "sidecar_sha256": expected,
            "valid": valid,
            "error": (
                structure_error
                if sidecar_valid and structure_error is not None
                else None
                if valid
                else "sidecar_mismatch"
            ),
        }
    )
    return (value if report["valid"] else None), report


def _check(
    status: str,
    *,
    expected: Any,
    observed: Any,
    reason_code: str,
) -> dict[str, Any]:
    if status not in _STATUS:
        raise ValueError(f"invalid doctor status {status!r}")
    return {
        "status": status,
        "expected": expected,
        "observed": observed,
        "reason_code": reason_code,
    }


def _required_equal(observed: Any, expected: Any, reason_code: str) -> dict[str, Any]:
    status = "PASS" if observed == expected else "FAIL"
    return _check(
        status,
        expected=expected,
        observed=observed,
        reason_code=reason_code,
    )


def _version_tuple(value: str) -> tuple[int, ...] | None:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if match is None:
        return None
    return tuple(int(part or 0) for part in match.groups())


def _nvcc_release(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.search(r"\brelease\s+(\d+\.\d+)\b", value)
    return match.group(1) if match else None


def _patch_binding(project_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = project_root / "patches/sglang/manifest.json"
    try:
        patch_manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"path": str(path), "matches": None, "error": type(error).__name__}
    if not isinstance(patch_manifest, dict):
        return {
            "path": str(path),
            "matches": False,
            "error": "patch_manifest_not_object",
        }
    expected = manifest["sglang"]
    observed = {
        "upstream_commit": patch_manifest.get("upstream", {}).get("commit"),
        "expected_tree": patch_manifest.get("expected_tree"),
        "patches": [
            {"file": row.get("file"), "sha256": row.get("sha256")}
            for row in patch_manifest.get("patches", [])
            if isinstance(row, dict)
        ],
    }
    required = {
        "upstream_commit": expected["upstream_commit"],
        "expected_tree": expected["expected_tree"],
        "patches": expected["patches"],
    }
    patch_files = [
        {
            "file": row["file"],
            "sha256": _file_sha256(project_root / "patches/sglang" / row["file"]),
        }
        for row in expected["patches"]
    ]
    return {
        "path": str(path),
        "matches": observed == required and patch_files == expected["patches"],
        "error": None,
        "expected": required,
        "observed": {**observed, "patch_files": patch_files},
    }


def _mode_gates(
    manifest: Mapping[str, Any] | None,
    *,
    distributed_gpu_proofs: tuple[object, ...] = (),
    native_gpu_proofs: tuple[object, ...] = (),
) -> dict[str, Any]:
    if manifest is None:
        pending = [
            {"mode": mode, "reason_code": reason}
            for mode, reason in _PENDING_DYNAMIC_GPU_PROOF_MODE_GATES
        ]
        external = [
            {"mode": mode, "reason_code": reason}
            for mode, reason in _EXTERNAL_AUTHORITY_MODE_GATES
        ]
        unsupported = [
            {"mode": mode, "reason_code": reason}
            for mode, reason in _UNSUPPORTED_MODE_GATES
        ]
        supported: dict[str, Any] = {}
        source = "fail_closed_builtin"
    else:
        capabilities = manifest["release_capabilities"]
        pending = capabilities["implemented_pending_dynamic_gpu_proof"]
        external = capabilities["external_authority_required"]
        unsupported = capabilities["unsupported"]
        supported = capabilities["supported"]
        source = "canonical_manifest"
    available: dict[str, dict[str, object]] = {}
    verified_modes: dict[str, str] = {}
    if distributed_gpu_proofs or native_gpu_proofs:
        from lightcone_spec.runtime.distributed import (
            DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
            VerifiedDistributedRuntimeGpuProof,
        )
        from lightcone_spec.runtime.readiness import (
            NATIVE_RUNTIME_RELEASE_CAPABILITY,
            VerifiedNativeRuntimeGpuProof,
        )

        if (
            type(distributed_gpu_proofs) is not tuple
            or type(native_gpu_proofs) is not tuple
        ):
            raise TypeError("doctor dynamic proof projection requires exact tuples")
        for proof in distributed_gpu_proofs:
            if type(proof) is not VerifiedDistributedRuntimeGpuProof:
                raise TypeError("doctor distributed proof must be verified")
            capability = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES[proof.topology_mode]
            if proof.source_capability_sha256 != capability.sha256:
                raise ValueError("doctor distributed proof changed source capability")
            mode = "tp2" if proof.topology_mode == "tp2_dp1" else "dp2"
            verified_modes[mode] = proof.receipt_sha256
        native_modes = {
            "chronobelief_gpu_parity": ("adaptive_chronobelief",),
            "dspark_tp1": ("adaptive_dspark",),
            "nextn_tp1": ("adaptive_nextn",),
            "nextn_tp2": ("adaptive_nextn",),
            "native_hot_path_tp1": (
                "native_itl",
                "fixed_address_publication_graph",
            ),
            "session_reset_tp1": ("session_reuse",),
        }
        for proof in native_gpu_proofs:
            if type(proof) is not VerifiedNativeRuntimeGpuProof:
                raise TypeError("doctor native proof must be verified")
            if (
                proof.source_capability_sha256
                != NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256
            ):
                raise ValueError("doctor native proof changed source capability")
            for mode in native_modes[proof.suite_id]:
                verified_modes[mode] = proof.receipt_sha256
    pending_projection: dict[str, dict[str, object]] = {}
    for row in pending:
        mode = row["mode"]
        proof_sha256 = verified_modes.get(mode)
        if proof_sha256 is None:
            pending_projection[mode] = {
                "status": "BLOCKED",
                "implementation_status": ("IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF"),
                "reason_code": row["reason_code"],
            }
        else:
            available[mode] = {
                "status": "AVAILABLE",
                "qualification_receipt_sha256": proof_sha256,
            }
    return {
        "source": source,
        "supported": supported,
        "available_after_dynamic_proof": available,
        "pending_dynamic_gpu_proof": pending_projection,
        "external_authority_required": {
            row["mode"]: {
                "status": "BLOCKED",
                "reason_code": row["reason_code"],
            }
            for row in external
        },
        "unsupported": {
            row["mode"]: {
                "status": "UNSUPPORTED",
                "reason_code": row["reason_code"],
            }
            for row in unsupported
        },
    }


def _evaluate(
    manifest: Mapping[str, Any] | None,
    manifest_report: Mapping[str, Any],
    facts: Mapping[str, Any],
    *,
    project_root: Path,
    sglang_root: Path,
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    checks["compatibility_manifest"] = _check(
        "PASS" if manifest_report["valid"] is True else "FAIL",
        expected="canonical JSON matching its .sha256 sidecar",
        observed={
            "sha256": manifest_report["sha256"],
            "sidecar_sha256": manifest_report["sidecar_sha256"],
            "error": manifest_report["error"],
        },
        reason_code="runtime_manifest_identity",
    )
    if manifest is None:
        for name in (
            "project_patch_binding",
            "python",
            "linux_host",
            "torch",
            "triton",
            "flashinfer",
            "flashinfer_cuda_flavor",
            "cuda_build",
            "cuda_toolkit",
            "torch_cuda_visibility",
            "driver",
            "gpu_count",
            "gpu_identity",
            "gpu_memory",
            "gpu_topology",
            "compiler",
            "disk",
            "network",
            "sglang_commit_lineage",
            "sglang_tree",
            "sglang_import",
        ):
            checks[name] = _check(
                "UNKNOWN",
                expected="valid compatibility manifest required",
                observed=None,
                reason_code="runtime_manifest_unavailable",
            )
        checks["project_sglang_roots_distinct"] = _required_equal(
            project_root != sglang_root,
            True,
            "project_sglang_roots_must_be_distinct",
        )
        return checks

    host_requirement = manifest["host"]
    package_requirement = manifest["packages"]
    cuda_requirement = manifest["cuda"]
    gpu_requirement = manifest["gpu"]
    disk_requirement = manifest["disk"]
    network_requirement = manifest["network"]
    sglang_requirement = manifest["sglang"]

    patch_binding = _patch_binding(project_root, manifest)
    checks["project_patch_binding"] = _check(
        (
            "UNKNOWN"
            if patch_binding["matches"] is None
            else "PASS"
            if patch_binding["matches"]
            else "FAIL"
        ),
        expected=patch_binding.get("expected", "runtime manifest patch identity"),
        observed=patch_binding.get("observed"),
        reason_code="project_patch_manifest_binding",
    )
    checks["project_sglang_roots_distinct"] = _required_equal(
        project_root != sglang_root,
        True,
        "project_sglang_roots_must_be_distinct",
    )
    project_tree = facts["project_source_tree"]
    project_ok = (
        project_tree["is_git_checkout"] is True
        and project_tree["root_matches_toplevel"] is True
        and project_tree["dirty"] is False
    )
    checks["project_source_tree"] = _check(
        "PASS" if project_ok else "FAIL",
        expected="exact clean project checkout root",
        observed=project_tree,
        reason_code="project_source_tree_identity",
    )
    project_runtime_source = facts["project_runtime_source"]
    checks["project_runtime_source"] = _check(
        "PASS" if project_runtime_source.get("valid") is True else "FAIL",
        expected="content-bound tracked LightCone Python source",
        observed=project_runtime_source,
        reason_code="project_runtime_source_identity",
    )
    checks["python"] = _required_equal(
        facts["python"]["major_minor"],
        host_requirement["python_minor"],
        "python_minor_exact",
    )
    checks["linux_host"] = _required_equal(
        facts["host"]["system"],
        host_requirement["operating_system"],
        "linux_host_required",
    )
    packages = facts["packages"]
    torch_runtime = facts["gpu"]["torch"]
    torch_runtime_version = torch_runtime.get("version")
    torch_runtime_base = (
        str(torch_runtime_version).partition("+")[0]
        if torch_runtime_version is not None
        else None
    )
    torch_version_ok = (
        packages.get("torch") == package_requirement["torch"]
        and torch_runtime.get("importable") is True
        and torch_runtime_base == package_requirement["torch"]
    )
    checks["torch"] = _check(
        "PASS" if torch_version_ok else "FAIL",
        expected={
            "distribution": package_requirement["torch"],
            "import_runtime": package_requirement["torch"],
        },
        observed={
            "distribution": packages.get("torch"),
            "import_runtime": torch_runtime_version,
            "importable": torch_runtime.get("importable"),
            "import_error": torch_runtime.get("import_error"),
        },
        reason_code="torch_version_exact",
    )
    checks["triton"] = _required_equal(
        packages.get("triton"),
        package_requirement["triton"],
        "triton_version_exact",
    )
    flashinfer = package_requirement["flashinfer_python"]
    checks["flashinfer"] = _required_equal(
        packages.get(flashinfer["distribution"]),
        flashinfer["version"],
        "flashinfer_version_exact",
    )
    cuda_marker = packages.get(flashinfer["cuda_marker_distribution"])
    checks["flashinfer_cuda_flavor"] = _check(
        "PASS" if cuda_marker is not None else "FAIL",
        expected={
            "extra": flashinfer["cuda_extra"],
            "marker_distribution": flashinfer["cuda_marker_distribution"],
        },
        observed={"marker_version": cuda_marker},
        reason_code="flashinfer_cuda_extra_exact",
    )
    cuda_observed = torch_runtime.get("cuda_build")
    checks["cuda_build"] = _check(
        (
            "UNKNOWN"
            if torch_runtime.get("importable") is not True
            else "PASS"
            if cuda_observed == cuda_requirement["torch_build"]
            else "FAIL"
        ),
        expected=cuda_requirement["torch_build"],
        observed=cuda_observed,
        reason_code="torch_cuda_build_exact",
    )
    nvcc_release = _nvcc_release(facts["commands"].get("nvcc"))
    checks["cuda_toolkit"] = _required_equal(
        nvcc_release,
        cuda_requirement["toolkit_release"],
        "cuda_toolkit_release_exact",
    )
    if torch_runtime.get("importable") is not True:
        torch_visibility_status = "UNKNOWN"
    else:
        torch_visibility_status = (
            "PASS"
            if torch_runtime.get("cuda_available") is True
            and torch_runtime.get("device_count")
            == len(facts["gpu"]["inventory"]["devices"])
            and torch_runtime.get("device_count")
            >= int(gpu_requirement["minimum_count"])
            else "FAIL"
        )
    checks["torch_cuda_visibility"] = _check(
        torch_visibility_status,
        expected={
            "cuda_available": True,
            "device_count_matches_nvidia_inventory": True,
            "minimum_count": gpu_requirement["minimum_count"],
        },
        observed={
            "cuda_available": torch_runtime.get("cuda_available"),
            "device_count": torch_runtime.get("device_count"),
        },
        reason_code="torch_cuda_device_visibility",
    )

    inventory = facts["gpu"]["inventory"]
    devices = inventory["devices"]
    minimum_count = int(gpu_requirement["minimum_count"])
    observed_count = len(devices)
    checks["gpu_count"] = _check(
        (
            "PASS"
            if inventory["parse_error"] is None and observed_count >= minimum_count
            else "FAIL"
        ),
        expected={"minimum_count": minimum_count, "same_host": True},
        observed={"count": observed_count, "parse_error": inventory["parse_error"]},
        reason_code="gpu_pool_count_supported",
    )
    if not devices:
        identity_status = memory_status = driver_status = "UNKNOWN"
    else:
        unique_uuid = len({device["uuid"] for device in devices}) == len(devices)
        unique_bus = len({device["pci_bus_id"] for device in devices}) == len(devices)
        identity_status = (
            "PASS"
            if observed_count >= minimum_count
            and unique_uuid
            and unique_bus
            and all(
                device["name"] == gpu_requirement["model"]
                and device["compute_capability"]
                == gpu_requirement["compute_capability"]
                for device in devices
            )
            else "FAIL"
        )
        memory_status = (
            "PASS"
            if observed_count >= minimum_count
            and all(
                device["memory_total_mib"] >= int(gpu_requirement["minimum_memory_mib"])
                for device in devices
            )
            else "FAIL"
        )
        versions = [device["driver_version"] for device in devices]
        parsed_versions = [_version_tuple(version) for version in versions]
        minimum = _version_tuple(manifest["system"]["minimum_driver"])
        driver_status = (
            "PASS"
            if minimum is not None
            and all(
                version is not None and version >= minimum
                for version in parsed_versions
            )
            and len(set(versions)) == 1
            else "FAIL"
        )
    checks["gpu_identity"] = _check(
        identity_status,
        expected={
            "minimum_count": minimum_count,
            "model": gpu_requirement["model"],
            "compute_capability": gpu_requirement["compute_capability"],
            "unique_uuid_and_pci_bus_id": True,
        },
        observed=devices,
        reason_code="gpu_identity_exact",
    )
    checks["gpu_memory"] = _check(
        memory_status,
        expected={
            "nominal_memory_gib": gpu_requirement["nominal_memory_gib"],
            "minimum_memory_mib": gpu_requirement["minimum_memory_mib"],
        },
        observed=[device["memory_total_mib"] for device in devices],
        reason_code="gpu_memory_capacity",
    )
    checks["driver"] = _check(
        driver_status,
        expected={
            "minimum": manifest["system"]["minimum_driver"],
            "same_on_all_devices": True,
        },
        observed=[device["driver_version"] for device in devices],
        reason_code="nvidia_driver_compatible",
    )
    topology = facts["gpu"]["topology"]
    accepted_links = set(gpu_requirement["accepted_topology_links"])
    expected_gpu_rows = [f"GPU{index}" for index in range(observed_count)]
    pairs = topology.get("pairs")
    topology_pairs_valid = isinstance(pairs, list) and len(pairs) == (
        observed_count * (observed_count - 1) // 2
    )
    if topology_pairs_valid:
        topology_pairs_valid = all(
            isinstance(pair, dict)
            and set(pair) == {"left", "right", "link", "reciprocal_link"}
            and pair["link"] == pair["reciprocal_link"]
            and pair["link"] in accepted_links
            for pair in pairs
        )
    if facts["gpu"].get("topology_raw") is None:
        topology_status = "UNKNOWN"
    else:
        topology_status = (
            "PASS"
            if topology["parse_error"] is None
            and topology["gpu_rows"] == expected_gpu_rows
            and topology_pairs_valid
            else "FAIL"
        )
    checks["gpu_topology"] = _check(
        topology_status,
        expected={
            "gpu_rows": expected_gpu_rows,
            "all_pairs_symmetric_link_one_of": sorted(accepted_links),
        },
        observed=topology,
        reason_code="same_host_gpu_pool_topology_declared",
    )
    compiler_observed = {
        "compiler": facts["commands"].get("compiler"),
        "ninja": facts["commands"].get("ninja"),
    }
    compiler_ok = bool(compiler_observed["compiler"] and compiler_observed["ninja"])
    checks["compiler"] = _check(
        "PASS" if compiler_ok else "FAIL",
        expected={"cxx": True, "ninja": True},
        observed=compiler_observed,
        reason_code="compiler_toolchain_available",
    )
    disk = facts["disk"]
    if disk["total_bytes"] is None or disk["free_bytes"] is None:
        disk_status = "UNKNOWN"
    else:
        disk_status = (
            "PASS"
            if disk["total_bytes"] >= disk_requirement["minimum_total_bytes"]
            and disk["free_bytes"] >= disk_requirement["minimum_free_bytes"]
            and (disk["free_inodes"] is None or disk["free_inodes"] > 0)
            else "FAIL"
        )
    checks["disk"] = _check(
        disk_status,
        expected=disk_requirement,
        observed=disk,
        reason_code="runtime_disk_capacity",
    )
    network = facts["network"]
    artifact_resolved = any(
        network.get(host, {}).get("resolved") is True
        for host in network_requirement["artifact_dns_any_of"]
    )
    package_resolved = (
        network.get(network_requirement["package_index_dns"], {}).get("resolved")
        is True
    )
    checks["network"] = _check(
        "PASS" if artifact_resolved and package_resolved else "FAIL",
        expected=network_requirement,
        observed=network,
        reason_code="china_compatible_dns_paths",
    )

    source = facts["source_tree"]
    commit_ok = (
        source["is_git_checkout"] is True
        and source["root_matches_toplevel"] is True
        and source["dirty"] is False
        and source["pinned_ancestor"] is True
        and source["patch_commits"] == sglang_requirement["expected_patch_count"]
    )
    checks["sglang_commit_lineage"] = _check(
        "PASS" if commit_ok else "FAIL",
        expected={
            "upstream_commit": sglang_requirement["upstream_commit"],
            "patch_commits": sglang_requirement["expected_patch_count"],
            "clean": True,
            "exact_checkout_root": True,
        },
        observed=source,
        reason_code="sglang_commit_lineage_exact",
    )
    checks["sglang_tree"] = _required_equal(
        source.get("tree"),
        sglang_requirement["expected_tree"],
        "sglang_patched_tree_exact",
    )
    sglang_package = packages.get("sglang")
    sglang_import = facts["sglang_import"]
    origin = sglang_import.get("origin")
    try:
        origin_path = Path(origin).resolve() if isinstance(origin, str) else None
    except (OSError, RuntimeError):
        origin_path = None
    origin_matches = origin_path is not None and (
        origin_path == sglang_root or sglang_root in origin_path.parents
    )
    checks["sglang_import"] = _check(
        "PASS"
        if sglang_package is not None
        and sglang_import.get("importable") is True
        and origin_matches
        else "FAIL",
        expected={
            "distribution_installed": True,
            "module_under_patched_sglang_root": str(sglang_root),
        },
        observed={
            "distribution_version": sglang_package,
            "importable": sglang_import.get("importable"),
            "origin": origin,
        },
        reason_code="sglang_runtime_importable",
    )
    return checks


def doctor_report(
    project_root: str | Path = ".",
    sglang_root: str | Path | None = None,
    *,
    stage_capacity_gate_path: str | Path | None = None,
    stage_capacity_schedule_path: str | Path | None = None,
    stage_capacity_attestation_path: str | Path | None = None,
    stage_capacity_activation_sha256: str | None = None,
    stage_capacity_now_ns: int | None = None,
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    sglang = Path(sglang_root).resolve() if sglang_root is not None else project
    manifest, manifest_report = _load_manifest(project)
    evaluation_manifest = manifest
    capacity_report: dict[str, Any]
    capacity_error: str | None = None
    capacity_inputs = (
        stage_capacity_gate_path,
        stage_capacity_schedule_path,
        stage_capacity_attestation_path,
        stage_capacity_activation_sha256,
    )
    if all(value is None for value in capacity_inputs):
        capacity_report = {
            "mode": "LEGACY_100GB_FALLBACK",
            "status": "FALLBACK",
            "reason_code": "stage_capacity_authority_absent",
            "required_free_bytes": (
                None
                if manifest is None
                else manifest.get("disk", {}).get("minimum_free_bytes")
            ),
            "gate_sha256": None,
            "schedule_sha256": None,
        }
    elif any(value is None for value in capacity_inputs):
        capacity_report = {
            "mode": "SIGNED_STAGE_ENVELOPE",
            "status": "FAIL",
            "reason_code": "stage_capacity_gate_schedule_control_required_together",
            "required_free_bytes": None,
            "gate_sha256": None,
            "schedule_sha256": None,
        }
        capacity_error = capacity_report["reason_code"]
    else:
        from lightcone_spec.experiments.capacity_authority import (
            CapacityAuthorityUnavailableError as CapacityUnavailable,
        )

        try:
            from lightcone_spec.experiments.registry import (
                build_industrial_registry,
            )
            from lightcone_spec.experiments.stage_capacity import (
                STAGE_CAPACITY_GATE_PROTOCOL_SHA256,
                StageCapacityGate,
                StageCapacitySchedule,
                revalidate_stage_capacity_gate_sources,
                stage_capacity_control_lineage_sha256,
            )
            from lightcone_spec.runtime.control_attestation import (
                ControlArtifactAttestation,
                verify_release_control_artifact_attestation,
            )
            from lightcone_spec.runtime.proof_artifact import (
                CanonicalJsonProofBinding,
            )

            verification_ns = (
                time.time_ns()
                if stage_capacity_now_ns is None
                else stage_capacity_now_ns
            )
            if type(verification_ns) is not int or verification_ns < 0:
                raise ValueError("stage capacity verification time is invalid")
            gate_source = CanonicalJsonProofBinding.bind(stage_capacity_gate_path)
            schedule_source = CanonicalJsonProofBinding.bind(
                stage_capacity_schedule_path
            )
            attestation_source = CanonicalJsonProofBinding.bind(
                stage_capacity_attestation_path
            )
            gate = StageCapacityGate.from_dict(gate_source.reopen())
            schedule = StageCapacitySchedule.from_dict(schedule_source.reopen())
            attestation = ControlArtifactAttestation.from_dict(
                attestation_source.reopen()
            )
            registry = build_industrial_registry()
            if (
                gate_source.semantic_sha256 != gate.sha256
                or schedule_source.semantic_sha256 != schedule.sha256
                or gate.schema_version != 3
                or gate.experiment != "preflight"
                or gate.schedule_sha256 != schedule.sha256
            ):
                raise ValueError("stage capacity identity differs from preflight")
            revalidate_stage_capacity_gate_sources(
                registry,
                gate,
                schedule=schedule,
                now_ns=verification_ns,
            )
            if (
                type(stage_capacity_activation_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", stage_capacity_activation_sha256)
                is None
            ):
                raise ValueError("stage capacity activation SHA-256 is invalid")
            expected_lineage = stage_capacity_control_lineage_sha256(
                activation_sha256=stage_capacity_activation_sha256,
                inventory_sha256=schedule.gpu_inventory_sha256,
                gate=gate,
            )
            subject = attestation.subject
            if (
                attestation_source.semantic_sha256 != attestation.sha256
                or subject.artifact_type != "capacity"
                or subject.artifact_sha256 != gate.sha256
                or subject.protocol_sha256 != STAGE_CAPACITY_GATE_PROTOCOL_SHA256
                or subject.registry_sha256 != registry.sha256
                or subject.lineage_sha256 != expected_lineage
            ):
                raise ValueError("stage capacity control subject differs")
            verified_capacity = verify_release_control_artifact_attestation(
                attestation,
                expected_inventory_sha256=schedule.gpu_inventory_sha256,
                now_ns=verification_ns,
                consumed_challenge_sha256s=(),
            )
            capacity_report = {
                "mode": gate.mode,
                "status": gate.status,
                "reason_code": gate.reason_code,
                "required_free_bytes": gate.required_free_bytes,
                "observed_free_bytes": gate.observed_free_bytes,
                "retained_evidence_bytes": gate.retained_evidence_bytes,
                "maximum_concurrent_transient_bytes": (
                    gate.maximum_concurrent_transient_bytes
                ),
                "safety_margin_bytes": gate.safety_margin_bytes,
                "gate_sha256": gate.sha256,
                "schedule_sha256": schedule.sha256,
                "control_envelope_sha256": attestation.sha256,
                "control_challenge_sha256": verified_capacity.challenge_sha256,
            }
            if manifest is not None:
                evaluation_manifest = {
                    **manifest,
                    "disk": {
                        **manifest["disk"],
                        "minimum_free_bytes": gate.required_free_bytes,
                    },
                }
        except CapacityUnavailable:
            capacity_report = {
                "mode": "LEGACY_100GB_FALLBACK",
                "status": "FALLBACK",
                "reason_code": "stage_capacity_authority_unavailable",
                "required_free_bytes": (
                    None
                    if manifest is None
                    else manifest.get("disk", {}).get("minimum_free_bytes")
                ),
                "gate_sha256": None,
                "schedule_sha256": None,
            }
        except (OSError, TypeError, ValueError, RuntimeError, KeyError) as error:
            capacity_error = f"{type(error).__name__}:{error}"
            capacity_report = {
                "mode": "SIGNED_STAGE_ENVELOPE",
                "status": "FAIL",
                "reason_code": "stage_capacity_authority_invalid",
                "required_free_bytes": None,
                "gate_sha256": None,
                "schedule_sha256": None,
                "error": capacity_error,
            }
    facts = _collect_facts(project, sglang)
    try:
        checks = _evaluate(
            evaluation_manifest,
            manifest_report,
            facts,
            project_root=project,
            sglang_root=sglang,
        )
    except (KeyError, TypeError, ValueError) as error:
        manifest = None
        manifest_report = {
            **manifest_report,
            "valid": False,
            "error": f"invalid_manifest_contract:{type(error).__name__}",
        }
        checks = _evaluate(
            None,
            manifest_report,
            facts,
            project_root=project,
            sglang_root=sglang,
        )
    if capacity_error is not None or capacity_report["status"] == "BLOCKED":
        checks["disk"] = _check(
            "FAIL",
            expected={
                "stage_capacity": "valid AVAILABLE schema-3 path-bound gate",
                "legacy_fallback_bytes": 100_000_000_000,
            },
            observed=capacity_report,
            reason_code="runtime_disk_capacity",
        )
    statuses = [check["status"] for check in checks.values()]
    readiness = (
        "FAIL" if "FAIL" in statuses else "UNKNOWN" if "UNKNOWN" in statuses else "PASS"
    )
    mode_gates = _mode_gates(manifest)
    inventory_raw = facts["gpu"]["inventory_raw"]
    return {
        "schema_version": 2,
        "status": readiness,
        "readiness": {
            "status": readiness,
            "pass_count": statuses.count("PASS"),
            "fail_count": statuses.count("FAIL"),
            "unknown_count": statuses.count("UNKNOWN"),
        },
        "roots": {
            "project": str(project),
            "patched_sglang": str(sglang),
            "distinct": project != sglang,
        },
        "runtime_manifest": manifest_report,
        "checks": checks,
        "mode_gates": mode_gates,
        "python": {
            **facts["python"],
            "supported": checks["python"]["status"] == "PASS",
            "remote_python_3_12": facts["python"]["major_minor"] == "3.12",
        },
        "host": facts["host"],
        "project_source_tree": facts["project_source_tree"],
        "project_runtime_source": facts["project_runtime_source"],
        # Retained for v2 attestation readers: this is always the SGLang tree.
        "source_tree": facts["source_tree"],
        "disk": facts["disk"],
        "network": {
            "huggingface_dns": facts["network"]["huggingface.co"],
            "modelscope_dns": facts["network"]["modelscope.cn"],
            "tsinghua_pypi_dns": facts["network"]["pypi.tuna.tsinghua.edu.cn"],
        },
        "gpu": {
            "inventory": inventory_raw,
            "parsed_inventory": facts["gpu"]["inventory"],
            "topology": facts["gpu"]["topology_raw"],
            "parsed_topology": facts["gpu"]["topology"],
            "torch": facts["gpu"]["torch"],
            "visible_gpu_count": len(facts["gpu"]["inventory"]["devices"]),
            "gpu_pool_visible": checks["gpu_count"]["status"] == "PASS",
            "two_gpu_visible": len(facts["gpu"]["inventory"]["devices"]) == 2,
        },
        "commands": facts["commands"],
        "packages": facts["packages"],
        "compatibility": {
            "status": readiness,
            "python_supported": checks["python"]["status"] == "PASS",
            "single_node_only": True,
            "multi_node_supported": False,
            "sglang_commit": PINNED_SGLANG_COMMIT,
            "sglang_tree": PINNED_SGLANG_TREE,
            "patch_count": PINNED_SGLANG_PATCH_COUNT,
            "manifest_sha256": manifest_report["sha256"],
        },
        "stage_capacity": capacity_report,
    }


def format_doctor(
    project_root: str | Path = ".",
    sglang_root: str | Path | None = None,
    *,
    stage_capacity_gate_path: str | Path | None = None,
    stage_capacity_schedule_path: str | Path | None = None,
    stage_capacity_attestation_path: str | Path | None = None,
    stage_capacity_activation_sha256: str | None = None,
    stage_capacity_now_ns: int | None = None,
) -> str:
    return json.dumps(
        doctor_report(
            project_root,
            sglang_root,
            stage_capacity_gate_path=stage_capacity_gate_path,
            stage_capacity_schedule_path=stage_capacity_schedule_path,
            stage_capacity_attestation_path=stage_capacity_attestation_path,
            stage_capacity_activation_sha256=stage_capacity_activation_sha256,
            stage_capacity_now_ns=stage_capacity_now_ns,
        ),
        indent=2,
        sort_keys=True,
    )
