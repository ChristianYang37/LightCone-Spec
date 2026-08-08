"""Online resolvers used by `lightcone-spec lock`.

Network access happens only here and only when the user runs `lock`
(or `lock --refresh`). Everything else in the system consumes the frozen
lockfile. Resolvers never download model weights; they resolve immutable
revisions and per-file metadata/hashes exposed by the hubs.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

from lightcone_spec import (
    PINNED_DEEPSPEC_COMMIT,
    PINNED_ONLINESPEC_COMMIT,
    PINNED_SGLANG_COMMIT,
)
from lightcone_spec.exit_codes import LockError
from lightcone_spec.locking.lockfile import (
    LockedEnvironment,
    LockedFile,
    LockedGPU,
    LockedGitRepo,
    LockedHFSnapshot,
)

PINNED_GIT_REPOS = {
    "sglang": ("https://github.com/sgl-project/sglang.git", PINNED_SGLANG_COMMIT),
    "deepspec": ("https://github.com/deepseek-ai/DeepSpec.git", PINNED_DEEPSPEC_COMMIT),
    "onlinespec": (
        "https://github.com/ZinYY/OnlineSPEC.git",
        PINNED_ONLINESPEC_COMMIT,
    ),
}

_RUNTIME_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".py"}


def _runtime_source_sha256(root: Path) -> Optional[str]:
    """Hash executable source without depending on a clean Git worktree."""
    if not root.is_dir():
        return None
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in _RUNTIME_SOURCE_SUFFIXES
        and "__pycache__" not in path.parts
        and not any(part.startswith("._") for part in path.parts)
    )
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _git_head_from_files(repo: Path) -> Optional[str]:
    """Read HEAD when Git refuses a mixed-ownership deployment tree."""
    git_dir = repo / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head if len(head) == 40 else None
        ref = head[5:]
        loose = git_dir / ref
        if loose.is_file():
            value = loose.read_text().strip()
            return value if len(value) == 40 else None
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if not line.startswith(("#", "^")) and line.endswith(f" {ref}"):
                    value = line.split(" ", 1)[0]
                    return value if len(value) == 40 else None
    except OSError:
        pass
    return None


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_git_repo(name: str, url: str, commit_sha: str) -> LockedGitRepo:
    """Verify the commit exists on the remote; never resolves branches."""
    if len(commit_sha) != 40:
        raise LockError(f"git revision for {name} must be a full 40-char SHA")
    try:
        out = subprocess.run(
            ["git", "ls-remote", url],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        raise LockError(f"cannot reach git remote {url}: {exc}") from exc
    # ls-remote lists refs only; a reachable unreferenced SHA is still
    # valid, so fall back to a shallow existence probe via fetch dry-run
    # only when the SHA is not directly listed.
    if commit_sha not in out:
        probe = subprocess.run(
            ["git", "fetch", "--dry-run", "--depth", "1", url, commit_sha],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if probe.returncode != 0:
            raise LockError(
                f"commit {commit_sha} not fetchable from {url}: {probe.stderr.strip()}"
            )
    return LockedGitRepo(name=name, url=url, commit_sha=commit_sha, dirty=False)


def resolve_hf_snapshot(
    repo_id: str,
    role: str,
    revision: str,
    include_chat_template: bool = False,
) -> LockedHFSnapshot:
    """Resolve an immutable HF snapshot: commit SHA plus per-file sizes and
    hashes from the hub metadata (LFS sha256 where available)."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise LockError("huggingface_hub is required for lock resolution") from exc
    api = HfApi()
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
    except Exception as exc:
        raise LockError(f"cannot resolve HF snapshot {repo_id}@{revision}: {exc}") from exc
    files: list[LockedFile] = []
    chat_template_sha: Optional[str] = None
    remote_code_sha: Optional[str] = None
    for sibling in info.siblings or []:
        digest = None
        if sibling.lfs is not None:
            digest = sibling.lfs.sha256
        elif sibling.blob_id is not None:
            digest = f"gitblob:{sibling.blob_id}"
        files.append(
            LockedFile(
                relpath=sibling.rfilename,
                size_bytes=int(sibling.size or 0),
                sha256=digest or "unknown",
            )
        )
        if sibling.rfilename.endswith(".py"):
            remote_code_sha = digest or remote_code_sha
    if include_chat_template:
        try:
            from huggingface_hub import hf_hub_download

            from lightcone_spec.locking.hashing import sha256_file

            cfg = hf_hub_download(
                repo_id, "tokenizer_config.json", revision=info.sha
            )
            chat_template_sha = sha256_file(cfg)
        except Exception as exc:
            raise LockError(
                f"cannot lock chat template for {repo_id}: {exc}"
            ) from exc
    return LockedHFSnapshot(
        repo_id=repo_id,
        snapshot_sha=info.sha,
        role=role,
        files=files,
        chat_template_sha256=chat_template_sha,
        trust_remote_code_sha256=remote_code_sha,
    )


def resolve_environment() -> LockedEnvironment:
    import torch

    cuda_version = getattr(torch.version, "cuda", None)
    triton_version = None
    try:
        import triton  # type: ignore

        triton_version = triton.__version__
    except Exception:
        triton_version = None
    sglang_version = None
    try:
        import sglang  # type: ignore

        sglang_version = sglang.__version__
    except Exception:
        sglang_version = None
    driver_version = None
    try:
        driver_version = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.splitlines()[0].strip()
    except (subprocess.SubprocessError, OSError, IndexError):
        pass

    compiler_versions = {"platform": platform.platform()}
    for key, argv in {
        "gcc": ["gcc", "--version"],
        "nvcc": ["nvcc", "--version"],
    }.items():
        try:
            output = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            ).stdout.strip()
            lines = output.splitlines()
            compiler_versions[key] = lines[0] if key == "gcc" else lines[-1]
        except (subprocess.SubprocessError, OSError, IndexError):
            continue
    lightcone_source = Path(__file__).resolve().parents[1]
    lightcone_digest = _runtime_source_sha256(lightcone_source)
    if lightcone_digest is not None:
        compiler_versions["lightcone_runtime_source_sha256"] = lightcone_digest
    explicit_sglang = os.environ.get("LIGHTCONE_SGLANG_SOURCE_ROOT")
    installed_sglang = find_spec("sglang")
    if explicit_sglang:
        sglang_package = Path(explicit_sglang).expanduser().resolve()
    elif installed_sglang is not None and installed_sglang.submodule_search_locations:
        sglang_package = Path(
            next(iter(installed_sglang.submodule_search_locations))
        ).resolve()
    else:
        sglang_package = (
            Path(__file__).resolve().parents[3] / "sglang" / "python" / "sglang"
        )
    workspace_sglang = sglang_package.parents[1]
    sglang_digest = _runtime_source_sha256(sglang_package)
    if sglang_digest is not None:
        compiler_versions["sglang_runtime_source_sha256"] = sglang_digest
    if (workspace_sglang / ".git").exists():
        try:
            compiler_versions["sglang_fork_commit"] = subprocess.run(
                ["git", "-C", str(workspace_sglang), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            ).stdout.strip()
            compiler_versions["sglang_fork_dirty"] = str(
                bool(
                    subprocess.run(
                        ["git", "-C", str(workspace_sglang), "status", "--porcelain"],
                        capture_output=True,
                        text=True,
                        timeout=20,
                        check=True,
                    ).stdout.strip()
                )
            ).lower()
        except (subprocess.SubprocessError, OSError):
            fallback_head = _git_head_from_files(workspace_sglang)
            if fallback_head is not None:
                compiler_versions["sglang_fork_commit"] = fallback_head
    return LockedEnvironment(
        docker_image_digest=None,
        python_version=sys.version.split()[0],
        cuda_version=cuda_version,
        driver_version=driver_version,
        torch_version=torch.__version__,
        triton_version=triton_version,
        sglang_version=sglang_version,
        compiler_versions=compiler_versions,
    )


def resolve_gpus() -> list[LockedGPU]:
    """Enumerate local GPUs via nvidia-smi; empty on non-CUDA hosts."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            LockedGPU(
                model_name=parts[0],
                uuid=parts[1],
                memory_bytes=int(float(parts[2]) * 1024 * 1024),
                interconnect=None,
                power_limit_watts=float(parts[3]) if parts[3] not in ("", "[N/A]") else None,
            )
        )
    return gpus
