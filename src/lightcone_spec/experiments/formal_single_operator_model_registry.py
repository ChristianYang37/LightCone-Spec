"""Source-owned model and workload path closure for formal v03.

The trusted content scanner intentionally hashes local bytes at publication
time.  This module owns the orthogonal scientific identity: the exact model
repositories, revisions, protocol-facing model labels, roles, stages, runtime
bindings, and seven task-native E0 source authorities that must be present
before a runtime-bound v03 bundle may enter bootstrap.

No content digest is registered here.  Operators provide only absolute local
paths to revision-addressed Hugging Face snapshots and already-published E0
source authorities; the content publisher scans those paths independently.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedE0TaskNativeDescriptor,
    TrustedE0TaskNativeDescriptorSpec,
    TrustedJsonArtifact,
    TrustedModelRuntimeBinding,
    TrustedModelSnapshotMember,
    TrustedModelSnapshotSpec,
    TrustedNamedInputPath,
    TrustedSingleOperatorContentBundle,
    TrustedSingleOperatorContentPathSpec,
    bind_trusted_json_artifact,
    load_trusted_single_operator_content_path_spec,
    publish_trusted_single_operator_content_path_spec,
    revalidate_trusted_json_artifact,
)
from lightcone_spec.experiments.formal_single_operator_e0_workloads import (
    E0_TASK_NATIVE_SOURCE_PINS,
    E0TaskNativeSourceAuthority,
    load_e0_task_native_source_authority,
    publish_e0_task_native_source_authority,
    scan_e0_task_native_source,
)
from lightcone_spec.experiments.formal_single_operator_loads import (
    BURSTGPT_V2_ASSETS,
)
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FormalV03ModelRole = Literal["target", "drafter", "tokenizer"]

_SHA256_CHARS = frozenset("0123456789abcdef")
_SNAPSHOT_KEY = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")
FORMAL_V03_E0_SOURCE_AUTHORITY_INDEX_FILE_NAME = "formal-v03-e0-source-authorities.json"
_CORE_STAGES = (
    "preflight",
    "E3a",
    "TTS-Cal",
    "E1",
    "E2",
    "E4",
    "E3b",
    "E1a",
    "E5",
)


def _strict_object(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    if set(value) != expected:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _strict_list(label: str, value: object) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an array")
    return value


def _from_fields(label: str, cls: type, value: object) -> dict[str, object]:
    return _strict_object(label, value, {field.name for field in fields(cls)})


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\n" in value
        or "\r" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be non-empty single-line text")
    return value


def _require_git_revision(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be a full lower-case Git revision")
    return value


def _resolved_directory(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError(f"{label} must be absolute and normalized")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    if resolved != path or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a resolved non-symlink directory")
    return path


def _resolved_file(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError(f"{label} must be absolute and normalized")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a resolved non-symlink file")
    return path


def _normalized_future_file(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path or not path.name:
        raise ValueError(f"{label} must be absolute and normalized")
    _resolved_directory(path.parent, label=f"{label} parent")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if path.exists():
        return _resolved_file(path, label=label)
    return path


@dataclass(frozen=True, order=True)
class FormalV03ModelMemberRegistryEntry:
    """One protocol-facing role materialized from a registered snapshot."""

    model_id: str
    role: FormalV03ModelRole
    stages: tuple[str, ...]
    runtime_bindings: tuple[TrustedModelRuntimeBinding, ...] = ()

    def __post_init__(self) -> None:
        _require_text("formal v03 protocol model ID", self.model_id)
        if self.role not in {"target", "drafter", "tokenizer"}:
            raise ValueError("formal v03 model role is unsupported")
        if not self.stages:
            raise ValueError("formal v03 model member lacks stages")
        if (
            type(self.runtime_bindings) is not tuple
            or any(
                type(row) is not TrustedModelRuntimeBinding
                for row in self.runtime_bindings
            )
            or self.runtime_bindings != tuple(sorted(set(self.runtime_bindings)))
        ):
            raise ValueError("formal v03 runtime bindings are not canonical")


@dataclass(frozen=True, order=True)
class FormalV03ModelSnapshotRegistryEntry:
    """One immutable upstream snapshot and all protocol roles it supplies."""

    snapshot_key: str
    snapshot_model_id: str
    revision: str
    members: tuple[FormalV03ModelMemberRegistryEntry, ...]

    def __post_init__(self) -> None:
        if type(self.snapshot_key) is not str or not _SNAPSHOT_KEY.fullmatch(
            self.snapshot_key
        ):
            raise ValueError("formal v03 snapshot key is not canonical")
        _require_text("formal v03 snapshot model ID", self.snapshot_model_id)
        _require_git_revision("formal v03 snapshot revision", self.revision)
        if (
            type(self.members) is not tuple
            or not self.members
            or any(
                type(row) is not FormalV03ModelMemberRegistryEntry
                for row in self.members
            )
            or self.members
            != tuple(sorted(self.members, key=lambda row: (row.role, row.model_id)))
            or len({(row.role, row.model_id) for row in self.members})
            != len(self.members)
        ):
            raise ValueError("formal v03 snapshot members are not canonical")


def _runtime(
    *,
    stage: Literal["preflight", "E6", "E0"],
    target: str,
    backend: Literal["DFLASH", "DSPARK", "EAGLE3", "NEXTN"],
    depth: int,
) -> tuple[TrustedModelRuntimeBinding, ...]:
    return (
        TrustedModelRuntimeBinding(
            stage=stage,
            target_model_id=target,
            backend=backend,
            draft_depth=depth,
        ),
    )


def _member(
    model_id: str,
    role: FormalV03ModelRole,
    stages: tuple[str, ...],
    runtime_bindings: tuple[TrustedModelRuntimeBinding, ...] = (),
) -> FormalV03ModelMemberRegistryEntry:
    return FormalV03ModelMemberRegistryEntry(
        model_id=model_id,
        role=role,
        stages=stages,
        runtime_bindings=tuple(sorted(runtime_bindings)),
    )


def _snapshot(
    key: str,
    snapshot_model_id: str,
    revision: str,
    *members: FormalV03ModelMemberRegistryEntry,
) -> FormalV03ModelSnapshotRegistryEntry:
    return FormalV03ModelSnapshotRegistryEntry(
        snapshot_key=key,
        snapshot_model_id=snapshot_model_id,
        revision=revision,
        members=tuple(sorted(members, key=lambda row: (row.role, row.model_id))),
    )


def _e0_drafter(
    *,
    key: str,
    snapshot_model_id: str,
    revision: str,
    target_model_id: str,
    backend: Literal["DFLASH", "DSPARK", "EAGLE3"],
    preflight_dspark: bool = False,
) -> FormalV03ModelSnapshotRegistryEntry:
    bindings = list(
        _runtime(stage="E0", target=target_model_id, backend=backend, depth=7)
    )
    stages = ("E0",)
    if preflight_dspark:
        bindings.extend(
            _runtime(
                stage="preflight",
                target="Qwen/Qwen3-8B",
                backend="DSPARK",
                # The block7 checkpoint identity is orthogonal to the core
                # preflight runtime's registered speculative depth.
                depth=15,
            )
        )
        stages = ("preflight", "E1a", "E5", "E0")
    return _snapshot(
        key,
        snapshot_model_id,
        revision,
        _member(snapshot_model_id, "drafter", stages, tuple(bindings)),
    )


# These revisions were recovered from the operator-audited, offline cache.  The
# registry intentionally contains no file/tree digest: the trusted content
# scanner derives those from each exact local snapshot path.
FORMAL_V03_MODEL_SNAPSHOT_REGISTRY: tuple[FormalV03ModelSnapshotRegistryEntry, ...] = (
    tuple(
        sorted(
            (
                _snapshot(
                    "qwen3_8b_target",
                    "Qwen/Qwen3-8B",
                    "b968826d9c46dd6066d109eabc6255188de91218",
                    _member("Qwen/Qwen3-8B", "target", (*_CORE_STAGES, "E0")),
                    _member("Qwen/Qwen3-8B", "tokenizer", (*_CORE_STAGES, "E0")),
                ),
                _snapshot(
                    "qwen3_8b_dflash_core",
                    "z-lab/Qwen3-8B-DFlash-b16",
                    "9b41424b7109f9c5413454f481b09a82b85333f4",
                    _member(
                        "z-lab/Qwen3-8B-DFlash-b16",
                        "drafter",
                        _CORE_STAGES,
                        _runtime(
                            stage="preflight",
                            target="Qwen/Qwen3-8B",
                            backend="DFLASH",
                            depth=15,
                        ),
                    ),
                ),
                _snapshot(
                    "qwen35_122b_a10b_fp8_nextn",
                    "Qwen/Qwen3.5-122B-A10B-FP8",
                    "a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9",
                    _member(
                        "Qwen/Qwen3.5-122B-A10B-FP8",
                        "target",
                        ("E6",),
                        _runtime(
                            stage="E6",
                            target="Qwen/Qwen3.5-122B-A10B-FP8",
                            backend="NEXTN",
                            depth=1,
                        ),
                    ),
                    _member("Qwen/Qwen3.5-122B-A10B-FP8", "tokenizer", ("E6",)),
                ),
                _snapshot(
                    "qwen36_35b_a3b_nextn",
                    "Qwen/Qwen3.6-35B-A3B",
                    "995ad96eacd98c81ed38be0c5b274b04031597b0",
                    _member(
                        "Qwen/Qwen3.6-35B-A3B",
                        "target",
                        ("E6",),
                        _runtime(
                            stage="E6",
                            target="Qwen/Qwen3.6-35B-A3B",
                            backend="NEXTN",
                            depth=1,
                        ),
                    ),
                    _member("Qwen/Qwen3.6-35B-A3B", "tokenizer", ("E6",)),
                ),
                _snapshot(
                    "qwen3_4b_target",
                    "Qwen/Qwen3-4B",
                    "1cfa9a7208912126459214e8b04321603b3df60c",
                    _member("Qwen/Qwen3-4B", "target", ("E0",)),
                    _member("Qwen/Qwen3-4B", "tokenizer", ("E0",)),
                ),
                _snapshot(
                    "qwen3_14b_target",
                    "Qwen/Qwen3-14B",
                    "40c069824f4251a91eefaf281ebe4c544efd3e18",
                    _member("Qwen/Qwen3-14B", "target", ("E0",)),
                    _member("Qwen/Qwen3-14B", "tokenizer", ("E0",)),
                ),
                _snapshot(
                    "gemma4_12b_target",
                    "google/gemma-4-12B-it",
                    "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
                    # E0's protocol label intentionally differs from the official
                    # Hugging Face repository.  The snapshot path is checked
                    # against snapshot_model_id while serving uses model_id.
                    _member("Gemma4-12B", "target", ("E0",)),
                    _member("Gemma4-12B", "tokenizer", ("E0",)),
                ),
                _e0_drafter(
                    key="qwen3_4b_dflash_e0",
                    snapshot_model_id="deepseek-ai/dflash_qwen3_4b_block7",
                    revision="02d530b7962ea1412beaf41a05c0b8e36d5f9b1d",
                    target_model_id="Qwen/Qwen3-4B",
                    backend="DFLASH",
                ),
                _e0_drafter(
                    key="qwen3_8b_dflash_e0",
                    snapshot_model_id="deepseek-ai/dflash_qwen3_8b_block7",
                    revision="9e44dbbb6cb68b0c943abf9c5fc3c17c00897cdf",
                    target_model_id="Qwen/Qwen3-8B",
                    backend="DFLASH",
                ),
                _e0_drafter(
                    key="qwen3_14b_dflash_e0",
                    snapshot_model_id="deepseek-ai/dflash_qwen3_14b_block7",
                    revision="ab0a8b28236654620bb41d64b336d00a14cb467f",
                    target_model_id="Qwen/Qwen3-14B",
                    backend="DFLASH",
                ),
                _e0_drafter(
                    key="gemma4_12b_dflash_e0",
                    snapshot_model_id="deepseek-ai/dflash_gemma4_12b_block7",
                    revision="7490ce60c7630107917fe558e2bbe3dcec6195cb",
                    target_model_id="Gemma4-12B",
                    backend="DFLASH",
                ),
                _e0_drafter(
                    key="qwen3_4b_dspark_e0",
                    snapshot_model_id="deepseek-ai/dspark_qwen3_4b_block7",
                    revision="3457dff1417cb84927f6098a5fcb7cee85c934b7",
                    target_model_id="Qwen/Qwen3-4B",
                    backend="DSPARK",
                ),
                _e0_drafter(
                    key="qwen3_8b_dspark_e0_core",
                    snapshot_model_id="deepseek-ai/dspark_qwen3_8b_block7",
                    revision="03326e5043815da1f81b109078b2889737c26017",
                    target_model_id="Qwen/Qwen3-8B",
                    backend="DSPARK",
                    preflight_dspark=True,
                ),
                _e0_drafter(
                    key="qwen3_14b_dspark_e0",
                    snapshot_model_id="deepseek-ai/dspark_qwen3_14b_block7",
                    revision="83207b416acf99f41c2184648923632fccea6dd0",
                    target_model_id="Qwen/Qwen3-14B",
                    backend="DSPARK",
                ),
                _e0_drafter(
                    key="gemma4_12b_dspark_e0",
                    snapshot_model_id="deepseek-ai/dspark_gemma4_12b_block7",
                    revision="2fa72e765eec2965fc4d86a8663ce6769eba6218",
                    target_model_id="Gemma4-12B",
                    backend="DSPARK",
                ),
                _e0_drafter(
                    key="qwen3_4b_eagle3_e0",
                    snapshot_model_id="deepseek-ai/eagle3_qwen3_4b_ttt7",
                    revision="b0b90fd15d052217c226be5e46d468d8d129e0cd",
                    target_model_id="Qwen/Qwen3-4B",
                    backend="EAGLE3",
                ),
                _e0_drafter(
                    key="qwen3_8b_eagle3_e0",
                    snapshot_model_id="deepseek-ai/eagle3_qwen3_8b_ttt7",
                    revision="f6485ba8d21e11942958617dbe7e71b467f38f38",
                    target_model_id="Qwen/Qwen3-8B",
                    backend="EAGLE3",
                ),
                _e0_drafter(
                    key="qwen3_14b_eagle3_e0",
                    snapshot_model_id="deepseek-ai/eagle3_qwen3_14b_ttt7",
                    revision="d7ea05d0b0009badfff0df2dcaedf82cce0f74f8",
                    target_model_id="Qwen/Qwen3-14B",
                    backend="EAGLE3",
                ),
                _e0_drafter(
                    key="gemma4_12b_eagle3_e0",
                    snapshot_model_id="deepseek-ai/eagle3_gemma4_12b_ttt7",
                    revision="0bc24c312350910419cf371e54082f040d65cc82",
                    target_model_id="Gemma4-12B",
                    backend="EAGLE3",
                ),
            ),
            key=lambda row: row.snapshot_key,
        )
    )
)


def build_formal_v03_model_lock() -> ModelLock:
    """Build the exact offline model lock owned by the formal-v03 registry."""

    snapshots = FORMAL_V03_MODEL_SNAPSHOT_REGISTRY
    if (
        type(snapshots) is not tuple
        or len(snapshots) != 19
        or any(
            type(row) is not FormalV03ModelSnapshotRegistryEntry for row in snapshots
        )
    ):
        raise RuntimeError("formal v03 model-lock registry is not exact nineteen")
    identities = tuple(
        sorted((row.snapshot_model_id, row.revision) for row in snapshots)
    )
    if len({model_id for model_id, _revision in identities}) != 19:
        raise RuntimeError(
            "formal v03 model-lock registry contains duplicate model IDs"
        )
    lock = ModelLock(
        schema_version=2,
        models=tuple(
            LockedModel(model_id=model_id, revision=revision)
            for model_id, revision in identities
        ),
    )
    lock.validate()
    if tuple((row.model_id, row.revision) for row in lock.models) != identities:
        raise RuntimeError("formal v03 model lock is not canonical")
    return lock


def _write_formal_v03_model_lock_staging_file(path: Path, body: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_formal_v03_model_lock_link_if_owned(
    path: Path,
    staging_path: Path,
) -> None:
    try:
        published = path.stat(follow_symlinks=False)
        staged = staging_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (published.st_dev, published.st_ino) == (staged.st_dev, staged.st_ino):
        path.unlink()


def _fsync_formal_v03_model_lock_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_formal_v03_model_lock(*, output_path: str | Path) -> ModelLock:
    """Publish the code-owned lock and sidecar without network or replacement."""

    output = Path(output_path)
    if (
        not output.is_absolute()
        or Path(os.path.abspath(output)) != output
        or output.resolve(strict=False) != output
        or not output.name
    ):
        raise ValueError("formal v03 model-lock output must be absolute and normalized")
    parent = _resolved_directory(output.parent, label="formal v03 model-lock output")
    sidecar = Path(f"{output}.sha256")
    if any(path.exists() or path.is_symlink() for path in (output, sidecar)):
        raise FileExistsError("formal v03 model-lock output or sidecar already exists")

    expected = build_formal_v03_model_lock()
    body = json.dumps(
        expected.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(body).hexdigest() != expected.sha256:
        raise RuntimeError("formal v03 model-lock encoding differs from ModelLock")
    sidecar_body = f"{expected.sha256}\n".encode("ascii")
    nonce = uuid.uuid4().hex
    staged_output = parent / f".{output.name}.tmp.{nonce}"
    staged_sidecar = parent / f".{sidecar.name}.tmp.{nonce}"
    sidecar_published = False
    output_published = False
    try:
        _write_formal_v03_model_lock_staging_file(staged_output, body)
        _write_formal_v03_model_lock_staging_file(staged_sidecar, sidecar_body)
        try:
            os.link(staged_sidecar, sidecar, follow_symlinks=False)
            sidecar_published = True
            os.link(staged_output, output, follow_symlinks=False)
            output_published = True
        except FileExistsError as error:
            raise FileExistsError(
                "formal v03 model-lock output or sidecar already exists"
            ) from error
        _fsync_formal_v03_model_lock_directory(parent)
    finally:
        if sidecar_published and not output_published:
            _unlink_formal_v03_model_lock_link_if_owned(sidecar, staged_sidecar)
            _fsync_formal_v03_model_lock_directory(parent)
        staged_output.unlink(missing_ok=True)
        staged_sidecar.unlink(missing_ok=True)

    rebound = ModelLock.load(output)
    if rebound != expected or rebound.sha256 != expected.sha256:
        raise RuntimeError("formal v03 model lock changed during publication")
    return rebound


def _registry_by_key() -> dict[str, FormalV03ModelSnapshotRegistryEntry]:
    result = {row.snapshot_key: row for row in FORMAL_V03_MODEL_SNAPSHOT_REGISTRY}
    if len(result) != len(FORMAL_V03_MODEL_SNAPSHOT_REGISTRY):
        raise RuntimeError("formal v03 snapshot registry contains duplicate keys")
    return result


def _expected_member_registry() -> dict[
    tuple[str, str, FormalV03ModelRole], FormalV03ModelSnapshotRegistryEntry
]:
    result: dict[
        tuple[str, str, FormalV03ModelRole], FormalV03ModelSnapshotRegistryEntry
    ] = {}
    for snapshot in FORMAL_V03_MODEL_SNAPSHOT_REGISTRY:
        for member in snapshot.members:
            identity = (member.model_id, snapshot.revision, member.role)
            if identity in result:
                raise RuntimeError("formal v03 model member identity is duplicated")
            result[identity] = snapshot
    return result


@dataclass(frozen=True, order=True)
class FormalV03NamedDirectoryPath:
    name: str
    absolute_path: str

    def __post_init__(self) -> None:
        _require_text("formal v03 directory path name", self.name)
        _resolved_directory(
            self.absolute_path, label=f"formal v03 {self.name} directory"
        )

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "absolute_path": self.absolute_path}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(**_from_fields("formal v03 named directory path", cls, value))  # type: ignore[arg-type]


@dataclass(frozen=True, order=True)
class FormalV03NamedFilePath:
    name: str
    absolute_path: str

    def __post_init__(self) -> None:
        _require_text("formal v03 file path name", self.name)
        _resolved_file(self.absolute_path, label=f"formal v03 {self.name} file")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "absolute_path": self.absolute_path}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(**_from_fields("formal v03 named file path", cls, value))  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalV03E0RawSourcePathInputs:
    """Exact-seven, path-only input for publishing E0 source authorities."""

    schema_version: Literal[1]
    kind: Literal["formal_v03_e0_raw_source_path_inputs"]
    source_paths: tuple[FormalV03NamedFilePath, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_v03_e0_raw_source_path_inputs"
        ):
            raise ValueError("formal v03 E0 raw-source input identity differs")
        expected_tasks = tuple(sorted(E0_TASK_NATIVE_SOURCE_PINS))
        if (
            type(self.source_paths) is not tuple
            or any(type(row) is not FormalV03NamedFilePath for row in self.source_paths)
            or tuple(row.name for row in self.source_paths) != expected_tasks
            or any(
                Path(row.absolute_path).name
                != E0_TASK_NATIVE_SOURCE_PINS[row.name].source_file_name
                for row in self.source_paths
            )
            or len({row.absolute_path for row in self.source_paths})
            != len(self.source_paths)
        ):
            raise ValueError("formal v03 E0 raw-source path coverage differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source_paths": [row.to_dict() for row in self.source_paths],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "formal v03 E0 raw-source path inputs",
            value,
            {field.name for field in fields(cls)},
        )
        sources = _strict_list(
            "formal v03 E0 raw-source paths", row.pop("source_paths")
        )
        return cls(
            **row,
            source_paths=tuple(
                FormalV03NamedFilePath.from_dict(item) for item in sources
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalV03E0SourceAuthorityIndex:
    """Path index published only after all seven authorities deep-reopen."""

    schema_version: Literal[1]
    kind: Literal["formal_v03_e0_source_authority_index"]
    authority_paths: tuple[FormalV03NamedFilePath, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_v03_e0_source_authority_index"
        ):
            raise ValueError("formal v03 E0 authority index identity differs")
        expected_tasks = tuple(sorted(E0_TASK_NATIVE_SOURCE_PINS))
        if (
            type(self.authority_paths) is not tuple
            or any(
                type(row) is not FormalV03NamedFilePath for row in self.authority_paths
            )
            or tuple(row.name for row in self.authority_paths) != expected_tasks
            or len({row.absolute_path for row in self.authority_paths})
            != len(self.authority_paths)
        ):
            raise ValueError("formal v03 E0 authority index coverage differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "authority_paths": [row.to_dict() for row in self.authority_paths],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "formal v03 E0 source authority index",
            value,
            {field.name for field in fields(cls)},
        )
        authorities = _strict_list(
            "formal v03 E0 source authority index paths",
            row.pop("authority_paths"),
        )
        return cls(
            **row,
            authority_paths=tuple(
                FormalV03NamedFilePath.from_dict(item) for item in authorities
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalV03ContentPathInputs:
    """Path-only operator input for the source-owned v03 content recipe."""

    schema_version: Literal[1]
    kind: Literal["formal_v03_content_path_inputs"]
    repository_root: str
    model_snapshot_paths: tuple[FormalV03NamedDirectoryPath, ...]
    livecodebench_raw_path: str
    math500_raw_path: str
    burstgpt_asset_paths: tuple[FormalV03NamedFilePath, ...]
    e0_source_authority_paths: tuple[FormalV03NamedFilePath, ...]
    inventory_path: str
    doctor_path: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "formal_v03_content_path_inputs":
            raise ValueError("formal v03 content input identity differs")
        _resolved_directory(self.repository_root, label="formal v03 source repository")
        expected_models = tuple(sorted(_registry_by_key()))
        if (
            type(self.model_snapshot_paths) is not tuple
            or any(
                type(row) is not FormalV03NamedDirectoryPath
                for row in self.model_snapshot_paths
            )
            or tuple(row.name for row in self.model_snapshot_paths) != expected_models
        ):
            raise ValueError("formal v03 model snapshot path coverage differs")
        _resolved_file(
            self.livecodebench_raw_path,
            label="formal v03 LiveCodeBench raw source",
        )
        _resolved_file(self.math500_raw_path, label="formal v03 MATH-500 raw source")
        expected_burst = tuple(row.name for row in BURSTGPT_V2_ASSETS)
        if (
            type(self.burstgpt_asset_paths) is not tuple
            or any(
                type(row) is not FormalV03NamedFilePath
                for row in self.burstgpt_asset_paths
            )
            or tuple(row.name for row in self.burstgpt_asset_paths) != expected_burst
        ):
            raise ValueError("formal v03 BurstGPT path coverage differs")
        expected_e0 = tuple(sorted(E0_TASK_NATIVE_SOURCE_PINS))
        if (
            type(self.e0_source_authority_paths) is not tuple
            or any(
                type(row) is not FormalV03NamedFilePath
                for row in self.e0_source_authority_paths
            )
            or tuple(row.name for row in self.e0_source_authority_paths) != expected_e0
        ):
            raise ValueError("formal v03 E0 source authority coverage differs")
        inventory = _resolved_file(
            self.inventory_path, label="formal v03 GPU inventory"
        )
        doctor = _normalized_future_file(
            self.doctor_path,
            label="formal v03 runtime doctor",
        )
        if inventory == doctor:
            raise ValueError("formal v03 inventory and doctor paths alias")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "repository_root": self.repository_root,
            "model_snapshot_paths": [
                row.to_dict() for row in self.model_snapshot_paths
            ],
            "livecodebench_raw_path": self.livecodebench_raw_path,
            "math500_raw_path": self.math500_raw_path,
            "burstgpt_asset_paths": [
                row.to_dict() for row in self.burstgpt_asset_paths
            ],
            "e0_source_authority_paths": [
                row.to_dict() for row in self.e0_source_authority_paths
            ],
            "inventory_path": self.inventory_path,
            "doctor_path": self.doctor_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "formal v03 content path inputs",
            value,
            {field.name for field in fields(cls)},
        )
        models = _strict_list(
            "formal v03 model snapshot paths", row.pop("model_snapshot_paths")
        )
        burst = _strict_list(
            "formal v03 BurstGPT paths", row.pop("burstgpt_asset_paths")
        )
        e0 = _strict_list(
            "formal v03 E0 source authority paths",
            row.pop("e0_source_authority_paths"),
        )
        return cls(
            **row,
            model_snapshot_paths=tuple(
                FormalV03NamedDirectoryPath.from_dict(item) for item in models
            ),
            burstgpt_asset_paths=tuple(
                FormalV03NamedFilePath.from_dict(item) for item in burst
            ),
            e0_source_authority_paths=tuple(
                FormalV03NamedFilePath.from_dict(item) for item in e0
            ),
        )  # type: ignore[arg-type]


def load_formal_v03_content_path_inputs(
    path: str | Path,
) -> FormalV03ContentPathInputs:
    binding = CanonicalJsonProofBinding.bind(path)
    return FormalV03ContentPathInputs.from_dict(binding.reopen())


def load_formal_v03_e0_raw_source_path_inputs(
    path: str | Path,
) -> FormalV03E0RawSourcePathInputs:
    binding = CanonicalJsonProofBinding.bind(path)
    return FormalV03E0RawSourcePathInputs.from_dict(binding.reopen())


def _named_directory_paths(
    values: Mapping[str, str | Path],
) -> tuple[FormalV03NamedDirectoryPath, ...]:
    if not isinstance(values, Mapping) or any(type(name) is not str for name in values):
        raise TypeError(
            "formal v03 named directory paths must be a string-keyed mapping"
        )
    return tuple(
        FormalV03NamedDirectoryPath(name=name, absolute_path=str(values[name]))
        for name in sorted(values)
    )


def _named_file_paths(
    values: Mapping[str, str | Path],
) -> tuple[FormalV03NamedFilePath, ...]:
    if not isinstance(values, Mapping) or any(type(name) is not str for name in values):
        raise TypeError("formal v03 named file paths must be a string-keyed mapping")
    return tuple(
        FormalV03NamedFilePath(name=name, absolute_path=str(values[name]))
        for name in sorted(values)
    )


def publish_formal_v03_e0_raw_source_path_inputs(
    *,
    source_paths: Mapping[str, str | Path],
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish exact-seven raw-source paths without caller-supplied identity."""

    inputs = FormalV03E0RawSourcePathInputs(
        schema_version=1,
        kind="formal_v03_e0_raw_source_path_inputs",
        source_paths=_named_file_paths(source_paths),
    )
    publish_canonical_json_no_replace(output_path, inputs.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    if load_formal_v03_e0_raw_source_path_inputs(output_path) != inputs:
        raise RuntimeError("formal v03 E0 raw-source path inputs changed")
    return binding


def publish_formal_v03_content_path_inputs(
    *,
    repository_root: str | Path,
    model_snapshot_paths: Mapping[str, str | Path],
    livecodebench_raw_path: str | Path,
    math500_raw_path: str | Path,
    burstgpt_asset_paths: Mapping[str, str | Path],
    e0_source_authority_paths: Mapping[str, str | Path],
    inventory_path: str | Path,
    doctor_output_path: str | Path,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish the registry-complete, path-only v03 content input handoff."""

    doctor = Path(doctor_output_path)
    if doctor == Path(output_path):
        raise ValueError("formal v03 inputs and future doctor outputs must differ")
    if doctor.exists() or doctor.is_symlink():
        raise FileExistsError("formal v03 doctor output must be a future file")
    inputs = FormalV03ContentPathInputs(
        schema_version=1,
        kind="formal_v03_content_path_inputs",
        repository_root=str(repository_root),
        model_snapshot_paths=_named_directory_paths(model_snapshot_paths),
        livecodebench_raw_path=str(livecodebench_raw_path),
        math500_raw_path=str(math500_raw_path),
        burstgpt_asset_paths=_named_file_paths(burstgpt_asset_paths),
        e0_source_authority_paths=_named_file_paths(e0_source_authority_paths),
        inventory_path=str(inventory_path),
        doctor_path=str(doctor),
    )
    publish_canonical_json_no_replace(output_path, inputs.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    if load_formal_v03_content_path_inputs(output_path) != inputs:
        raise RuntimeError("formal v03 content path inputs changed")
    return binding


def _require_e0_authority_identity(
    authority: E0TaskNativeSourceAuthority,
    *,
    task: str,
) -> None:
    expected_status = "UNSUPPORTED" if task == "MT-Bench" else "READY"
    if (
        type(authority) is not E0TaskNativeSourceAuthority
        or authority.task != task
        or authority.support_status != expected_status
    ):
        raise ValueError("formal v03 E0 source authority identity differs")


def _deep_validate_e0_authority_index(
    index: FormalV03E0SourceAuthorityIndex,
) -> None:
    for row in index.authority_paths:
        authority = load_e0_task_native_source_authority(row.absolute_path)
        _require_e0_authority_identity(authority, task=row.name)


def load_formal_v03_e0_source_authority_index(
    path: str | Path,
) -> FormalV03E0SourceAuthorityIndex:
    binding = CanonicalJsonProofBinding.bind(path)
    index = FormalV03E0SourceAuthorityIndex.from_dict(binding.reopen())
    _deep_validate_e0_authority_index(index)
    return index


def require_formal_v03_pass_runtime_doctor(
    path: str | Path,
    *,
    expected_bound_content_bundle: TrustedSingleOperatorContentBundle | None = None,
    require_capacity_available: bool = True,
) -> CanonicalJsonProofBinding:
    """Delegate doctor trust exclusively to the source-replay authority."""

    from lightcone_spec.doctor import (
        revalidate_trusted_single_operator_doctor_report,
    )

    binding = revalidate_trusted_single_operator_doctor_report(
        path,
        expected_bound_content_bundle=expected_bound_content_bundle,
        require_capacity_available=require_capacity_available,
    )
    if type(binding) is not CanonicalJsonProofBinding:
        raise TypeError("formal v03 doctor revalidator returned a foreign binding")
    return binding


def _e0_authority_file_name(task: str) -> str:
    return f"formal-v03-e0-{task.lower().replace('-', '_')}-source-authority.json"


def publish_formal_v03_e0_source_authorities_from_inputs(
    *,
    inputs_path: str | Path,
    output_directory: str | Path,
) -> CanonicalJsonProofBinding:
    """Scan and publish the exact seven E0 authorities without network input."""

    inputs = load_formal_v03_e0_raw_source_path_inputs(inputs_path)
    output = _resolved_directory(
        output_directory, label="formal v03 E0 authority output"
    )
    if any(output.iterdir()):
        raise FileExistsError("formal v03 E0 authority output directory is not empty")

    # Complete every expensive raw-source scan before creating the first output.
    # MT-Bench's registered N/A disposition therefore cannot mask a missing task.
    scanned: list[E0TaskNativeSourceAuthority] = []
    for row in inputs.source_paths:
        authority = scan_e0_task_native_source(
            task=row.name,  # type: ignore[arg-type]
            raw_source_path=row.absolute_path,
        )
        _require_e0_authority_identity(authority, task=row.name)
        authority.revalidate()
        scanned.append(authority)
    if load_formal_v03_e0_raw_source_path_inputs(inputs_path) != inputs:
        raise RuntimeError("formal v03 E0 raw-source inputs changed during scan")
    if any(output.iterdir()):
        raise RuntimeError("formal v03 E0 authority output changed during scan")

    published_rows: list[FormalV03NamedFilePath] = []
    for authority in scanned:
        destination = output / _e0_authority_file_name(authority.task)
        publish_e0_task_native_source_authority(authority, output_path=destination)
        rebound = load_e0_task_native_source_authority(destination)
        _require_e0_authority_identity(rebound, task=authority.task)
        if rebound != authority:
            raise RuntimeError("formal v03 E0 source authority changed on publication")
        published_rows.append(
            FormalV03NamedFilePath(
                name=authority.task,
                absolute_path=str(destination),
            )
        )

    index = FormalV03E0SourceAuthorityIndex(
        schema_version=1,
        kind="formal_v03_e0_source_authority_index",
        authority_paths=tuple(sorted(published_rows, key=lambda row: row.name)),
    )
    index_path = output / FORMAL_V03_E0_SOURCE_AUTHORITY_INDEX_FILE_NAME
    publish_canonical_json_no_replace(index_path, index.to_dict())
    rebound_index = load_formal_v03_e0_source_authority_index(index_path)
    if rebound_index != index:
        raise RuntimeError("formal v03 E0 source authority index changed")
    expected_names = {
        FORMAL_V03_E0_SOURCE_AUTHORITY_INDEX_FILE_NAME,
        *(_e0_authority_file_name(row.name) for row in inputs.source_paths),
    }
    if {row.name for row in output.iterdir()} != expected_names:
        raise RuntimeError("formal v03 E0 authority output coverage changed")
    return CanonicalJsonProofBinding.bind(index_path)


def _snapshot_cache_root(
    snapshot: FormalV03ModelSnapshotRegistryEntry,
    path_value: str | Path,
) -> Path:
    path = _resolved_directory(
        path_value, label=f"formal v03 snapshot {snapshot.snapshot_key}"
    )
    expected_repository_directory = "models--" + snapshot.snapshot_model_id.replace(
        "/", "--"
    )
    if (
        path.name != snapshot.revision
        or path.parent.name != "snapshots"
        or path.parent.parent.name != expected_repository_directory
    ):
        raise ValueError(
            f"formal v03 snapshot {snapshot.snapshot_key} path/revision differs"
        )
    cache_root = path.parent.parent
    _resolved_directory(
        cache_root, label=f"formal v03 snapshot {snapshot.snapshot_key} cache root"
    )
    return cache_root


def _model_specs_from_paths(
    paths: tuple[FormalV03NamedDirectoryPath, ...],
) -> tuple[TrustedModelSnapshotSpec, ...]:
    registry = _registry_by_key()
    specs: list[TrustedModelSnapshotSpec] = []
    for binding in paths:
        snapshot = registry[binding.name]
        cache_root = _snapshot_cache_root(snapshot, binding.absolute_path)
        for member in snapshot.members:
            specs.append(
                TrustedModelSnapshotSpec(
                    model_id=member.model_id,
                    revision=snapshot.revision,
                    role=member.role,
                    stages=member.stages,
                    local_snapshot_path=binding.absolute_path,
                    storage_mode="huggingface_cache_symlinks",
                    content_cache_root=str(cache_root),
                    runtime_bindings=member.runtime_bindings,
                )
            )
    return tuple(
        sorted(
            specs,
            key=lambda row: (
                row.role,
                row.model_id,
                row.revision,
                row.local_snapshot_path,
            ),
        )
    )


def _e0_descriptor_id(task: str) -> str:
    return "formal-v03-e0-" + task.lower().replace("-", "_")


def _e0_descriptor_specs_from_paths(
    paths: tuple[FormalV03NamedFilePath, ...],
) -> tuple[TrustedE0TaskNativeDescriptorSpec, ...]:
    result = []
    for binding in paths:
        authority = load_e0_task_native_source_authority(binding.absolute_path)
        if authority.task != binding.name:
            raise ValueError("formal v03 E0 authority path/task differs")
        result.append(
            TrustedE0TaskNativeDescriptorSpec(
                descriptor_id=_e0_descriptor_id(authority.task),
                task=authority.task,
                repository=authority.repository,
                revision=authority.repository_revision,
                descriptor_path=binding.absolute_path,
            )
        )
    return tuple(sorted(result, key=lambda row: row.descriptor_id))


def build_formal_v03_content_path_spec(
    inputs: FormalV03ContentPathInputs,
) -> TrustedSingleOperatorContentPathSpec:
    """Derive the digest-free trusted-content recipe from exact path inputs."""

    if type(inputs) is not FormalV03ContentPathInputs:
        raise TypeError("formal v03 content producer requires exact path inputs")
    spec = TrustedSingleOperatorContentPathSpec(
        schema_version=1,
        kind="trusted_single_operator_content_path_spec",
        repository_root=inputs.repository_root,
        model_specs=_model_specs_from_paths(inputs.model_snapshot_paths),
        livecodebench_raw_path=inputs.livecodebench_raw_path,
        math500_raw_path=inputs.math500_raw_path,
        burstgpt_asset_paths=tuple(
            TrustedNamedInputPath(name=row.name, absolute_path=row.absolute_path)
            for row in inputs.burstgpt_asset_paths
        ),
        e0_task_native_specs=_e0_descriptor_specs_from_paths(
            inputs.e0_source_authority_paths
        ),
        inventory_path=inputs.inventory_path,
        doctor_path=inputs.doctor_path,
    )
    require_formal_v03_content_path_spec(spec)
    return spec


def _member_registry_row(
    snapshot: FormalV03ModelSnapshotRegistryEntry,
    *,
    model_id: str,
    role: FormalV03ModelRole,
) -> FormalV03ModelMemberRegistryEntry:
    matches = tuple(
        row for row in snapshot.members if row.model_id == model_id and row.role == role
    )
    if len(matches) != 1:
        raise RuntimeError("formal v03 registry member lookup is ambiguous")
    return matches[0]


def _require_formal_v03_model_coverage(
    rows: tuple[TrustedModelSnapshotSpec, ...] | tuple[TrustedModelSnapshotMember, ...],
) -> None:
    expected = _expected_member_registry()
    if len(rows) != len(expected):
        raise ValueError("formal v03 model member coverage differs")
    observed: dict[
        tuple[str, str, FormalV03ModelRole],
        TrustedModelSnapshotSpec | TrustedModelSnapshotMember,
    ] = {}
    snapshot_paths: dict[str, str] = {}
    for row in rows:
        identity = (row.model_id, row.revision, row.role)
        snapshot = expected.get(identity)
        if snapshot is None or identity in observed:
            raise ValueError("formal v03 model member is foreign or duplicated")
        member = _member_registry_row(snapshot, model_id=row.model_id, role=row.role)
        cache_root = _snapshot_cache_root(snapshot, row.local_snapshot_path)
        if (
            row.stages != member.stages
            or row.runtime_bindings != member.runtime_bindings
            or row.storage_mode != "huggingface_cache_symlinks"
            or row.content_cache_root != str(cache_root)
        ):
            raise ValueError("formal v03 model member binding differs from registry")
        previous = snapshot_paths.setdefault(
            snapshot.snapshot_key, row.local_snapshot_path
        )
        if previous != row.local_snapshot_path:
            raise ValueError("formal v03 snapshot roles use different local paths")
        observed[identity] = row
    if set(observed) != set(expected):
        raise ValueError("formal v03 model member coverage is incomplete")
    if len(set(snapshot_paths.values())) != len(snapshot_paths):
        raise ValueError("formal v03 distinct snapshots alias one local path")


def _validated_e0_authority(
    *,
    task: str,
    descriptor_id: str,
    repository: str,
    revision: str,
    path: str,
) -> E0TaskNativeSourceAuthority:
    authority = load_e0_task_native_source_authority(path)
    expected_status = "UNSUPPORTED" if task == "MT-Bench" else "READY"
    if (
        task not in E0_TASK_NATIVE_SOURCE_PINS
        or descriptor_id != _e0_descriptor_id(task)
        or authority.task != task
        or authority.repository != repository
        or authority.repository_revision != revision
        or authority.support_status != expected_status
    ):
        raise ValueError("formal v03 E0 source authority identity differs")
    return authority


def _require_formal_v03_e0_spec_coverage(
    rows: tuple[TrustedE0TaskNativeDescriptorSpec, ...],
) -> None:
    expected = set(E0_TASK_NATIVE_SOURCE_PINS)
    if len(rows) != len(expected):
        raise ValueError("formal v03 E0 source authority coverage differs")
    observed = set()
    paths = set()
    for row in rows:
        _validated_e0_authority(
            task=row.task,
            descriptor_id=row.descriptor_id,
            repository=row.repository,
            revision=row.revision,
            path=row.descriptor_path,
        )
        if row.task in observed or row.descriptor_path in paths:
            raise ValueError("formal v03 E0 source authority is duplicated")
        observed.add(row.task)
        paths.add(row.descriptor_path)
    if observed != expected:
        raise ValueError("formal v03 E0 source authority coverage is incomplete")


def _require_formal_v03_e0_member_coverage(
    rows: tuple[TrustedE0TaskNativeDescriptor, ...],
) -> None:
    expected = set(E0_TASK_NATIVE_SOURCE_PINS)
    if len(rows) != len(expected):
        raise ValueError("formal v03 E0 source member coverage differs")
    observed = set()
    paths = set()
    for row in rows:
        _validated_e0_authority(
            task=row.task,
            descriptor_id=row.descriptor_id,
            repository=row.repository,
            revision=row.revision,
            path=row.source.absolute_path,
        )
        rebound = bind_trusted_json_artifact(
            f"e0_task_native:{row.descriptor_id}", row.source.absolute_path
        )
        if (
            rebound != row.source
            or row.task in observed
            or row.source.absolute_path in paths
        ):
            raise ValueError("formal v03 E0 source member differs or is duplicated")
        observed.add(row.task)
        paths.add(row.source.absolute_path)
    if observed != expected:
        raise ValueError("formal v03 E0 source member coverage is incomplete")


def require_formal_v03_content_path_spec(
    spec: TrustedSingleOperatorContentPathSpec,
) -> None:
    """Reject missing, foreign, aliased, or mutated formal-v03 path members."""

    if type(spec) is not TrustedSingleOperatorContentPathSpec:
        raise TypeError("formal v03 coverage requires an exact content path spec")
    # Model closure is deliberately checked before the MT-Bench N/A authority:
    # an unsupported task can never hide a missing model/backend binding.
    _require_formal_v03_model_coverage(spec.model_specs)
    _require_formal_v03_e0_spec_coverage(spec.e0_task_native_specs)


def _canonical_artifact_binding(
    artifact: TrustedJsonArtifact,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not TrustedJsonArtifact:
        raise TypeError("formal v03 runtime artifact binding is foreign")
    revalidate_trusted_json_artifact(artifact)
    binding = CanonicalJsonProofBinding.bind(artifact.absolute_path)
    if (
        binding.absolute_path != artifact.absolute_path
        or binding.size != artifact.size
        or binding.raw_sha256 != artifact.raw_sha256
        or binding.semantic_sha256 != artifact.semantic_sha256
    ):
        raise ValueError("formal v03 runtime artifact is not canonical and exact")
    return binding


def _require_formal_v03_runtime_observation_identity(
    bundle: TrustedSingleOperatorContentBundle,
    *,
    require_capacity_available: bool = True,
    revalidate_runtime_observations: bool = True,
) -> None:
    if (
        type(require_capacity_available) is not bool
        or type(revalidate_runtime_observations) is not bool
    ):
        raise TypeError("formal v03 runtime replay policies must be boolean")
    runtime = bundle.runtime_observations
    if runtime is None:
        raise ValueError("formal v03 runtime observations are absent")

    from lightcone_spec.experiments.gpu_pool import GpuInventory

    inventory_binding = _canonical_artifact_binding(runtime.inventory)
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    if inventory.sha256 != runtime.inventory.semantic_sha256:
        raise ValueError("formal v03 runtime inventory identity differs")
    if len(inventory.devices) != 2:
        raise ValueError("formal v03 runtime inventory must contain exactly two GPUs")

    exact_doctor_binding = _canonical_artifact_binding(runtime.doctor)
    if revalidate_runtime_observations:
        doctor_binding = require_formal_v03_pass_runtime_doctor(
            runtime.doctor.absolute_path,
            expected_bound_content_bundle=bundle,
            require_capacity_available=require_capacity_available,
        )
        if doctor_binding != exact_doctor_binding:
            raise ValueError("formal v03 runtime doctor binding differs")
    else:
        doctor_binding = exact_doctor_binding
    doctor = doctor_binding.reopen()
    gpu = doctor.get("gpu")
    parsed = None if type(gpu) is not dict else gpu.get("parsed_inventory")
    raw_devices = None if type(parsed) is not dict else parsed.get("devices")
    if type(raw_devices) is not list:
        raise ValueError("formal v03 doctor lacks its parsed GPU device set")
    observed: list[tuple[str, str, str]] = []
    for row in raw_devices:
        if type(row) is not dict:
            raise ValueError("formal v03 doctor GPU device is not an object")
        uuid = row.get("uuid")
        model = row.get("name")
        compute_capability = row.get("compute_capability")
        if any(type(value) is not str for value in (uuid, model, compute_capability)):
            raise ValueError("formal v03 doctor GPU identity fields differ")
        observed.append((uuid, model, compute_capability))  # type: ignore[arg-type]
    expected = tuple(
        sorted(
            (
                device.uuid,
                device.model,
                f"{device.compute_capability[0]}.{device.compute_capability[1]}",
            )
            for device in inventory.devices
        )
    )
    if (
        len(observed) != len({row[0] for row in observed})
        or tuple(sorted(observed)) != expected
    ):
        raise ValueError("formal v03 inventory and doctor GPU device sets differ")


def require_formal_v03_bound_content_bundle(
    bundle: TrustedSingleOperatorContentBundle,
    *,
    require_capacity_available: bool = True,
    revalidate_runtime_observations: bool = True,
) -> None:
    """Deep-check the exact source/model closure immediately before bootstrap."""

    if (
        type(require_capacity_available) is not bool
        or type(revalidate_runtime_observations) is not bool
    ):
        raise TypeError("formal v03 bound-content replay policies must be boolean")
    if type(bundle) is not TrustedSingleOperatorContentBundle:
        raise TypeError("formal v03 coverage requires an exact content bundle")
    if bundle.runtime_binding_status != "BOUND" or bundle.runtime_observations is None:
        raise ValueError("formal v03 content bundle is not runtime-BOUND")
    _require_formal_v03_model_coverage(bundle.model_members)
    _require_formal_v03_e0_member_coverage(bundle.e0_task_native_descriptors)
    _require_formal_v03_runtime_observation_identity(
        bundle,
        require_capacity_available=require_capacity_available,
        revalidate_runtime_observations=revalidate_runtime_observations,
    )
    from lightcone_spec.doctor import (
        revalidate_trusted_single_operator_doctor_report,
    )

    if revalidate_runtime_observations:
        assert bundle.runtime_observations is not None
        revalidate_trusted_single_operator_doctor_report(
            bundle.runtime_observations.doctor.absolute_path,
            expected_bound_content_bundle=bundle,
            require_capacity_available=require_capacity_available,
        )


def publish_formal_v03_content_path_spec_from_inputs(
    *,
    inputs_path: str | Path,
    output_path: str | Path,
) -> TrustedSingleOperatorContentPathSpec:
    """Publish one no-replace digest-free v03 content recipe from paths only."""

    inputs = load_formal_v03_content_path_inputs(inputs_path)
    spec = build_formal_v03_content_path_spec(inputs)
    publish_trusted_single_operator_content_path_spec(spec, output_path)
    rebound = load_trusted_single_operator_content_path_spec(output_path)
    require_formal_v03_content_path_spec(rebound)
    if rebound != spec or load_formal_v03_content_path_inputs(inputs_path) != inputs:
        raise RuntimeError("formal v03 content path inputs changed during publication")
    return rebound


__all__ = [
    "FORMAL_V03_E0_SOURCE_AUTHORITY_INDEX_FILE_NAME",
    "FORMAL_V03_MODEL_SNAPSHOT_REGISTRY",
    "FormalV03ContentPathInputs",
    "FormalV03E0RawSourcePathInputs",
    "FormalV03E0SourceAuthorityIndex",
    "FormalV03ModelMemberRegistryEntry",
    "FormalV03ModelSnapshotRegistryEntry",
    "FormalV03NamedDirectoryPath",
    "FormalV03NamedFilePath",
    "build_formal_v03_content_path_spec",
    "build_formal_v03_model_lock",
    "load_formal_v03_content_path_inputs",
    "load_formal_v03_e0_raw_source_path_inputs",
    "load_formal_v03_e0_source_authority_index",
    "publish_formal_v03_content_path_inputs",
    "publish_formal_v03_content_path_spec_from_inputs",
    "publish_formal_v03_e0_raw_source_path_inputs",
    "publish_formal_v03_e0_source_authorities_from_inputs",
    "publish_formal_v03_model_lock",
    "require_formal_v03_bound_content_bundle",
    "require_formal_v03_content_path_spec",
    "require_formal_v03_pass_runtime_doctor",
]
