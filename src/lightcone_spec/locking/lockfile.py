"""Lockfile schema and IO (spec 2.3).

A lockfile pins every mutable input: git repos, HF snapshots,
tokenizers, per-file hashes, datasets, container image, software stack
and GPU inventory. After generation, run commands accept only immutable
revisions and fail closed on any drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lightcone_spec.exit_codes import LockError
from lightcone_spec.locking.hashing import canonical_json, sha256_bytes

LOCK_SCHEMA_VERSION = 1


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LockedGitRepo(_Strict):
    name: str
    url: str
    commit_sha: str
    dirty: bool = False


class LockedFile(_Strict):
    relpath: str
    size_bytes: int
    sha256: str


class LockedHFSnapshot(_Strict):
    repo_id: str
    snapshot_sha: str
    role: str  # target | drafter | tokenizer | judge
    files: list[LockedFile] = Field(default_factory=list)
    chat_template_sha256: Optional[str] = None
    trust_remote_code_sha256: Optional[str] = None


class LockedDataset(_Strict):
    adapter_key: str
    source: str
    config: Optional[str] = None
    split: str
    revision: str
    files: list[LockedFile] = Field(default_factory=list)
    sample_ids_sha256: str
    num_samples: int
    script_sha256: Optional[str] = None
    license_note: str = ""


class LockedEnvironment(_Strict):
    docker_image_digest: Optional[str] = None
    python_version: str
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    torch_version: str
    triton_version: Optional[str] = None
    sglang_version: Optional[str] = None
    compiler_versions: dict[str, str] = Field(default_factory=dict)


class LockedGPU(_Strict):
    model_name: str
    uuid: str
    memory_bytes: int
    interconnect: Optional[str] = None
    power_limit_watts: Optional[float] = None


class Lockfile(_Strict):
    schema_version: int = LOCK_SCHEMA_VERSION
    created_utc: str
    git_repos: list[LockedGitRepo] = Field(default_factory=list)
    hf_snapshots: list[LockedHFSnapshot] = Field(default_factory=list)
    datasets: list[LockedDataset] = Field(default_factory=list)
    environment: LockedEnvironment
    gpus: list[LockedGPU] = Field(default_factory=list)

    # ---- content addressing ------------------------------------------

    def body_dict(self) -> dict:
        return self.model_dump(mode="json")

    def content_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.body_dict()).encode("utf-8"))

    def write(self, path: str | Path) -> str:
        """Write lockfile plus a sibling .sha256; returns the hash."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = canonical_json(self.body_dict())
        path.write_text(body)
        digest = sha256_bytes(body.encode("utf-8"))
        Path(str(path) + ".sha256").write_text(digest + "\n")
        return digest

    # ---- lookups -------------------------------------------------------

    def find_snapshot(self, repo_id: str) -> LockedHFSnapshot:
        for snap in self.hf_snapshots:
            if snap.repo_id == repo_id:
                return snap
        raise LockError(f"repo {repo_id!r} not present in lockfile (fail closed)")

    def find_dataset(self, adapter_key: str) -> LockedDataset:
        for ds in self.datasets:
            if ds.adapter_key == adapter_key:
                return ds
        raise LockError(f"dataset {adapter_key!r} not present in lockfile (fail closed)")


def load_lockfile(path: str | Path, verify_hash: bool = True) -> Lockfile:
    path = Path(path)
    if not path.is_file():
        raise LockError(f"lockfile not found: {path}")
    text = path.read_text()
    try:
        lock = Lockfile.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LockError(f"lockfile invalid: {exc}") from exc
    if verify_hash:
        sha_path = Path(str(path) + ".sha256")
        if not sha_path.is_file():
            raise LockError(f"lockfile hash sidecar missing: {sha_path}")
        expected = sha_path.read_text().strip()
        actual = sha256_bytes(canonical_json(lock.body_dict()).encode("utf-8"))
        if actual != expected:
            raise LockError(
                f"lockfile hash drift: expected {expected}, computed {actual}"
            )
    return lock
