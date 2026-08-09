"""Read-only local/remote readiness report."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from lightcone_spec import PINNED_SGLANG_COMMIT


def _command(args: list[str]) -> str | None:
    executable = shutil.which(args[0])
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, *args[1:]],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=15,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _source_tree(root: Path) -> dict:
    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", "-C", str(root), *arguments),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=15,
        )

    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    if head.returncode != 0 or tree.returncode != 0:
        return {
            "is_git_checkout": False,
            "head": None,
            "tree": None,
            "dirty": None,
            "pinned_ancestor": False,
            "patch_commits": None,
        }
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    ancestor = git(
        "merge-base", "--is-ancestor", PINNED_SGLANG_COMMIT, "HEAD"
    )
    count = git("rev-list", "--count", f"{PINNED_SGLANG_COMMIT}..HEAD")
    return {
        "is_git_checkout": True,
        "head": head.stdout.strip(),
        "tree": tree.stdout.strip(),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "pinned_ancestor": ancestor.returncode == 0,
        "patch_commits": (
            int(count.stdout.strip()) if count.returncode == 0 else None
        ),
    }


def doctor_report(path: str | Path = ".") -> dict:
    root = Path(path).resolve()
    usage = shutil.disk_usage(root)
    packages = {}
    for name in ("torch", "triton", "sglang", "flashinfer-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "source_tree": _source_tree(root),
        "disk": {
            "path": str(root),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        },
        "commands": {
            "nvidia_smi": _command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "nvcc": _command(["nvcc", "--version"]),
            "compiler": _command(["c++", "--version"]),
        },
        "packages": packages,
    }


def format_doctor(path: str | Path = ".") -> str:
    return json.dumps(doctor_report(path), indent=2, sort_keys=True)
