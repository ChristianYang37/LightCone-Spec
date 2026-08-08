"""Offline lockfile verification (fail closed, spec 2.3).

Given a lockfile and a local artifact root (e.g. an HF cache export or a
models directory), verify that every locked file exists with the exact
size and hash. Missing files, hash mismatch, or revision drift return a
LockError; nothing is silently tolerated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from lightcone_spec.exit_codes import LockError
from lightcone_spec.locking.hashing import sha256_file
from lightcone_spec.locking.lockfile import Lockfile


def verify_lockfile_offline(
    lock: Lockfile,
    local_roots: dict[str, str | Path],
    require_all: bool = True,
) -> list[str]:
    """Verify locked snapshots against local directories.

    local_roots maps repo_id -> local snapshot directory. Returns the
    list of verified repo_ids; raises LockError on any mismatch. When
    require_all is True every locked snapshot must have a local root.
    """
    verified: list[str] = []
    roots = {k: Path(v) for k, v in local_roots.items()}
    for snap in lock.hf_snapshots:
        root = roots.get(snap.repo_id)
        if root is None:
            if require_all:
                raise LockError(
                    f"no local root provided for locked snapshot {snap.repo_id} "
                    "(fail closed)"
                )
            continue
        if not root.is_dir():
            raise LockError(f"local root missing for {snap.repo_id}: {root}")
        root_resolved = root.resolve()
        expected_relpaths = {f.relpath for f in snap.files}
        actual_relpaths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and ".cache" not in path.relative_to(root).parts
        }
        unexpected = sorted(actual_relpaths - expected_relpaths)
        if unexpected:
            raise LockError(
                f"{snap.repo_id}: unexpected files outside the locked snapshot: "
                f"{unexpected[:8]}"
            )
        for f in snap.files:
            path = root / f.relpath
            try:
                path.resolve().relative_to(root_resolved)
            except ValueError as exc:
                raise LockError(
                    f"{snap.repo_id}: locked path escapes snapshot root: {f.relpath}"
                ) from exc
            if not path.is_file():
                raise LockError(f"{snap.repo_id}: locked file missing: {f.relpath}")
            size = path.stat().st_size
            if size != f.size_bytes:
                raise LockError(
                    f"{snap.repo_id}:{f.relpath}: size drift "
                    f"(locked {f.size_bytes}, found {size})"
                )
            if f.sha256 == "unknown":
                raise LockError(
                    f"{snap.repo_id}:{f.relpath}: lock contains no content hash"
                )
            if f.sha256.startswith("gitblob:"):
                expected_blob = f.sha256.split(":", 1)[1]
                content = path.read_bytes()
                actual_blob = hashlib.sha1(
                    f"blob {len(content)}\0".encode("ascii") + content
                ).hexdigest()
                if actual_blob != expected_blob:
                    raise LockError(
                        f"{snap.repo_id}:{f.relpath}: git blob hash drift "
                        f"(locked {expected_blob}, found {actual_blob})"
                    )
            else:
                actual = sha256_file(path)
                if actual != f.sha256:
                    raise LockError(
                        f"{snap.repo_id}:{f.relpath}: sha256 drift "
                        f"(locked {f.sha256}, found {actual})"
                    )
        verified.append(snap.repo_id)
    return verified


def check_pair_lock(lock: Lockfile, target_repo: str, drafter_repo: str) -> None:
    """Both halves of a model pair must be locked before serve starts."""
    lock.find_snapshot(target_repo)
    lock.find_snapshot(drafter_repo)
