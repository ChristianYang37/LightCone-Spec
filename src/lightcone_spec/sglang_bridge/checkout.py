"""Verify the disposable SGLang checkout used by a measured server."""

from __future__ import annotations

import subprocess
from pathlib import Path

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(checkout), *arguments),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"cannot verify patched SGLang checkout: {completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def verify_patched_checkout(path: str | Path) -> Path:
    """Return an exact, clean six-patch checkout or fail before launch."""
    checkout = Path(path).resolve()
    if not (checkout / "python" / "sglang" / "__init__.py").is_file():
        raise ValueError("SGLang checkout lacks python/sglang/__init__.py")
    tree = _git(checkout, "rev-parse", "HEAD^{tree}")
    if tree != PINNED_SGLANG_TREE:
        raise ValueError(
            f"patched SGLang tree mismatch: expected {PINNED_SGLANG_TREE}, got {tree}"
        )
    if _git(checkout, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("patched SGLang checkout must be clean before launch")
    completed = subprocess.run(
        (
            "git",
            "-C",
            str(checkout),
            "merge-base",
            "--is-ancestor",
            PINNED_SGLANG_COMMIT,
            "HEAD",
        ),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise ValueError("patched SGLang history does not descend from the pin")
    count = int(
        _git(checkout, "rev-list", "--count", f"{PINNED_SGLANG_COMMIT}..HEAD")
    )
    if count != 6:
        raise ValueError("patched SGLang checkout must contain the six-patch series")
    return checkout
