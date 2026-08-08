"""Download locked Hugging Face snapshots to explicit verified roots."""

from __future__ import annotations

import json
from pathlib import Path

from lightcone_spec.exit_codes import LockError
from lightcone_spec.locking.hashing import canonical_json, sha256_bytes
from lightcone_spec.locking.lockfile import Lockfile
from lightcone_spec.locking.verify import verify_lockfile_offline


def _snapshot_dir(cache_root: Path, repo_id: str, sha: str) -> Path:
    safe = repo_id.replace("/", "--")
    return cache_root / "models" / safe / sha


def prepare_locked_models(
    lock: Lockfile,
    cache_root: str | Path,
    *,
    repo_ids: list[str] | None = None,
) -> dict[str, str]:
    """Download exact lockfile revisions and verify every locked file.

    Authentication is intentionally delegated to ``HF_TOKEN`` or the user's
    existing Hugging Face credential store; tokens never enter arguments or
    output artifacts.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise LockError("prepare-models requires huggingface_hub") from exc

    selected = set(repo_ids or [s.repo_id for s in lock.hf_snapshots])
    unknown = selected - {s.repo_id for s in lock.hf_snapshots}
    if unknown:
        raise LockError(f"requested repos are not locked: {sorted(unknown)}")
    root = Path(cache_root).expanduser().resolve()
    roots: dict[str, str] = {}
    for snap in lock.hf_snapshots:
        if snap.repo_id not in selected:
            continue
        local_dir = _snapshot_dir(root, snap.repo_id, snap.snapshot_sha)
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            path = snapshot_download(
                repo_id=snap.repo_id,
                revision=snap.snapshot_sha,
                local_dir=str(local_dir),
                token=None,
            )
        except Exception as exc:
            raise LockError(
                f"cannot download locked snapshot {snap.repo_id}@{snap.snapshot_sha}: {exc}"
            ) from exc
        roots[snap.repo_id] = str(Path(path).resolve())
    verify_lockfile_offline(lock, roots, require_all=False)
    return roots


def write_model_roots(roots: dict[str, str], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json(roots)
    target.write_text(body)
    Path(str(target) + ".sha256").write_text(
        sha256_bytes(body.encode("utf-8")) + "\n"
    )


def load_model_roots(path: str | Path) -> dict[str, str]:
    target = Path(path)
    sidecar = Path(str(target) + ".sha256")
    if not target.is_file() or not sidecar.is_file():
        raise LockError(f"model-roots file or hash sidecar missing: {target}")
    body = target.read_text()
    actual = sha256_bytes(body.encode("utf-8"))
    if actual != sidecar.read_text().strip():
        raise LockError(f"model-roots hash drift: {target}")
    try:
        roots = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LockError(f"model-roots JSON invalid: {target}: {exc}") from exc
    if not isinstance(roots, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in roots.items()
    ):
        raise LockError("model-roots must map repository ids to local paths")
    return roots
