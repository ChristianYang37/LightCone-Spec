"""Portable source replay for the formal :class:`ProtocolLock`.

The lock is the root of the scientific DAG.  A signature over a structurally
valid lock is insufficient: the signer must reproduce it from a clean Git
snapshot plus the independently published runtime, method, content, workload,
and BurstGPT authorities.  The Git snapshot is a standard bundle split into
bounded raw shards, so the proof can be pulled to an offline signer without
retaining the online checkout or embedding a recursive repository JSON blob.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from lightcone_spec.experiments.formal_method_authority import (
    load_chronobelief_authority_artifact,
    load_tts_calibration_authority_artifact,
)
from lightcone_spec.experiments.formal_protocol import (
    ProtocolLock,
    code_owned_qualification_source_identities,
    content_sha256,
)
from lightcone_spec.experiments.formal_registry import (
    formal_runtime_authority_manifest_from_dict,
)
from lightcone_spec.experiments.formal_runtime_manifest import (
    build_source_formal_runtime_authority_manifest,
)
from lightcone_spec.experiments.formal_stage_execution import (
    load_e1_recipe_anchor_authority_artifact,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_materialization import (
    default_e2_recipe_grid_authority,
)
from lightcone_spec.runtime.content_authorization import (
    ContentVerificationReceipt,
    VerifiedDatasetContentRelease,
    VerifiedPreparedModelContentRelease,
    VerifiedReleaseWorkloadSources,
)
from lightcone_spec.runtime.preflight_runner import (
    BurstGptShapeAuthority,
    EvidenceFileBinding,
    derive_burstgpt_shape_authority_from_content_receipt,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
    relocated_evidence_path,
)
from lightcone_spec.runtime.release_trust_root import (
    SourceReleaseEd25519Root,
    load_source_release_ed25519_root,
)

FORMAL_PROTOCOL_LOCK_GIT_SNAPSHOT_KIND = "formal_protocol_lock_git_snapshot_index"
FORMAL_PROTOCOL_LOCK_SOURCE_PROOF_KIND = "formal_protocol_lock_source_proof_artifact"
FORMAL_PROTOCOL_LOCK_GIT_CHUNK_BYTES = 1024 * 1024
_RELEASE_ROOT_MANIFEST_RELATIVE_PATH = (
    "src/lightcone_spec/runtime/trust/release_ed25519_root_v1.json"
)
_RELEASE_ROOT_SIDECAR_RELATIVE_PATH = f"{_RELEASE_ROOT_MANIFEST_RELATIVE_PATH}.sha256"


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _git_oid(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case Git object ID")
    return value


def _strict(value: object, fields: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            "formal ProtocolLock Git command failed: " + completed.stderr.strip()
        )
    return completed.stdout.strip()


def _clean_git_identity(project_root: str | Path) -> tuple[Path, str, str]:
    root = Path(os.path.abspath(os.fspath(project_root)))
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("formal ProtocolLock project root is not resolved")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("formal ProtocolLock source proof requires clean Git")
    head = _git(root, "rev-parse", "--verify", "HEAD")
    tree = _git(root, "rev-parse", "--verify", "HEAD^{tree}")
    _git_oid("formal ProtocolLock Git HEAD", head)
    _git_oid("formal ProtocolLock Git tree", tree)
    return root, head, tree


def _safe_output_parent(path: Path, *, label: str) -> None:
    if not path.is_absolute() or path != Path(os.path.abspath(path)) or not path.name:
        raise ValueError(f"{label} must be an absolute normalized file path")
    current = Path(path.anchor)
    for component in path.parent.parts[1:]:
        current /= component
        try:
            status = current.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"{label} parent is missing") from error
        if not stat.S_ISDIR(status.st_mode) or current.is_symlink():
            raise ValueError(f"{label} ancestors must be symlink-free directories")
    status = path.parent.stat()
    if status.st_uid != os.geteuid() or stat.S_IMODE(status.st_mode) & 0o022:
        raise ValueError(f"{label} parent is not current-user owned and private")


def _publish_bytes_no_replace(path: Path, body: bytes, *, label: str) -> None:
    _safe_output_parent(path, label=label)
    if not body:
        raise ValueError(f"{label} cannot be empty")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive OS boundary
                raise OSError("formal Git snapshot publication made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stable_raw(binding: EvidenceFileBinding, *, label: str) -> bytes:
    if type(binding) is not EvidenceFileBinding:
        raise TypeError(f"{label} is not an exact raw-file binding")
    binding.reopen(label=label)
    path = relocated_evidence_path(binding.absolute_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} is not one regular file")
        body = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            body.extend(chunk)
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(body) != binding.size
            or hashlib.sha256(body).hexdigest() != binding.raw_sha256
        ):
            raise RuntimeError(f"{label} changed while reopened")
        return bytes(body)
    finally:
        os.close(descriptor)


def _stable_json(binding: CanonicalJsonProofBinding, *, label: str) -> object:
    before = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if before != binding:
        raise ValueError(f"{label} path identity changed")
    value = binding.reopen()
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != before:
        raise RuntimeError(f"{label} changed while reopened")
    return value


def _relative_source(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a repository-relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} is not a safe repository-relative path")
    return path.as_posix()


@dataclass(frozen=True)
class FormalProtocolLockGitSnapshotIndex:
    schema_version: Literal[1]
    kind: Literal["formal_protocol_lock_git_snapshot_index"]
    git_head: str
    git_tree: str
    bundle_raw_sha256: str
    bundle_size: int
    chunk_bytes: int
    chunks: tuple[EvidenceFileBinding, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != FORMAL_PROTOCOL_LOCK_GIT_SNAPSHOT_KIND
        ):
            raise ValueError("formal ProtocolLock Git snapshot schema differs")
        _git_oid("formal ProtocolLock Git snapshot HEAD", self.git_head)
        _git_oid("formal ProtocolLock Git snapshot tree", self.git_tree)
        _sha256("formal ProtocolLock Git bundle", self.bundle_raw_sha256)
        if type(self.bundle_size) is not int or self.bundle_size < 1:
            raise ValueError("formal ProtocolLock Git bundle size is invalid")
        if self.chunk_bytes != FORMAL_PROTOCOL_LOCK_GIT_CHUNK_BYTES:
            raise ValueError("formal ProtocolLock Git chunk bound differs")
        if (
            type(self.chunks) is not tuple
            or not self.chunks
            or any(type(row) is not EvidenceFileBinding for row in self.chunks)
            or len({row.absolute_path for row in self.chunks}) != len(self.chunks)
            or any(row.size > self.chunk_bytes for row in self.chunks)
            or sum(row.size for row in self.chunks) != self.bundle_size
        ):
            raise ValueError("formal ProtocolLock Git chunks are not exact")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "git_head": self.git_head,
            "git_tree": self.git_tree,
            "bundle_raw_sha256": self.bundle_raw_sha256,
            "bundle_size": self.bundle_size,
            "chunk_bytes": self.chunk_bytes,
            "chunks": [row.to_dict() for row in self.chunks],
        }
        if include_sha256:
            value["index_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            value,
            {*cls.__dataclass_fields__, "index_sha256"},
            label="formal ProtocolLock Git snapshot",
        )
        declared = _sha256("formal ProtocolLock Git snapshot", row.pop("index_sha256"))
        raw_chunks = row["chunks"]
        if type(raw_chunks) is not list:
            raise TypeError("formal ProtocolLock Git chunks must be an array")
        row["chunks"] = tuple(
            EvidenceFileBinding.from_dict(item, label="formal ProtocolLock Git chunk")
            for item in raw_chunks
        )
        index = cls(**row)  # type: ignore[arg-type]
        if index.sha256 != declared:
            raise ValueError("formal ProtocolLock Git snapshot digest differs")
        return index


def _write_snapshot_bundle(
    index: FormalProtocolLockGitSnapshotIndex, path: Path
) -> None:
    digest = hashlib.sha256()
    total = 0
    with path.open("xb") as handle:
        for number, binding in enumerate(index.chunks):
            body = _stable_raw(
                binding,
                label=f"formal ProtocolLock Git chunk {number}",
            )
            handle.write(body)
            digest.update(body)
            total += len(body)
    if total != index.bundle_size or digest.hexdigest() != index.bundle_raw_sha256:
        raise ValueError("formal ProtocolLock Git bundle reconstruction differs")


@contextlib.contextmanager
def checkout_formal_protocol_lock_git_snapshot(
    index_path: str | Path,
) -> Iterator[tuple[Path, FormalProtocolLockGitSnapshotIndex]]:
    """Reconstruct a bounded Git bundle and yield its detached clean checkout."""

    binding = CanonicalJsonProofBinding.bind(index_path)
    index = FormalProtocolLockGitSnapshotIndex.from_dict(binding.reopen())
    with tempfile.TemporaryDirectory(prefix="lightcone-protocol-lock-") as directory:
        temporary = Path(directory)
        bundle = temporary / "snapshot.bundle"
        _write_snapshot_bundle(index, bundle)
        heads = subprocess.run(
            ("git", "bundle", "list-heads", str(bundle)),
            check=False,
            capture_output=True,
            text=True,
        )
        if heads.returncode != 0:
            raise ValueError("formal ProtocolLock Git bundle is invalid")
        rows = tuple(
            tuple(line.split()) for line in heads.stdout.splitlines() if line.strip()
        )
        if rows != ((index.git_head, "HEAD"),):
            raise ValueError("formal ProtocolLock Git bundle HEAD differs")
        checkout = temporary / "checkout"
        cloned = subprocess.run(
            ("git", "clone", "--no-checkout", str(bundle), str(checkout)),
            check=False,
            capture_output=True,
            text=True,
        )
        if cloned.returncode != 0:
            raise ValueError("formal ProtocolLock Git bundle cannot be cloned")
        _git(checkout, "checkout", "--detach", "--force", index.git_head)
        head = _git(checkout, "rev-parse", "HEAD")
        tree = _git(checkout, "rev-parse", "HEAD^{tree}")
        if (
            head != index.git_head
            or tree != index.git_tree
            or _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
        ):
            raise ValueError("formal ProtocolLock Git checkout identity differs")
        yield checkout, index
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise RuntimeError("formal ProtocolLock Git snapshot changed while replayed")


def publish_formal_protocol_lock_git_snapshot(
    *,
    project_root: str | Path,
    chunk_output_directory: str | Path,
    index_output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Snapshot one clean checkout into bounded immutable Git bundle shards."""

    root, head, tree = _clean_git_identity(project_root)
    chunk_directory = Path(os.path.abspath(os.fspath(chunk_output_directory)))
    if (
        chunk_directory.is_symlink()
        or not chunk_directory.is_dir()
        or chunk_directory.resolve() != chunk_directory
    ):
        raise ValueError("formal ProtocolLock Git chunk directory is unsafe")
    index_output = Path(os.path.abspath(os.fspath(index_output_path)))
    _safe_output_parent(index_output, label="formal ProtocolLock Git index")
    try:
        chunk_directory.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("formal ProtocolLock Git chunks must be outside the checkout")
    with tempfile.TemporaryDirectory(prefix="lightcone-git-bundle-") as directory:
        bundle = Path(directory) / "snapshot.bundle"
        completed = subprocess.run(
            ("git", "-C", str(root), "bundle", "create", str(bundle), "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ValueError("formal ProtocolLock Git bundle creation failed")
        body = bundle.read_bytes()
    if not body:
        raise ValueError("formal ProtocolLock Git bundle is empty")
    chunks = []
    for index, offset in enumerate(
        range(0, len(body), FORMAL_PROTOCOL_LOCK_GIT_CHUNK_BYTES)
    ):
        path = chunk_directory / f"protocol-lock-git-{index:05d}.bundlepart"
        _publish_bytes_no_replace(
            path,
            body[offset : offset + FORMAL_PROTOCOL_LOCK_GIT_CHUNK_BYTES],
            label="formal ProtocolLock Git chunk",
        )
        chunks.append(
            EvidenceFileBinding.bind(path, label="formal ProtocolLock Git chunk")
        )
    index = FormalProtocolLockGitSnapshotIndex(
        schema_version=1,
        kind=FORMAL_PROTOCOL_LOCK_GIT_SNAPSHOT_KIND,
        git_head=head,
        git_tree=tree,
        bundle_raw_sha256=hashlib.sha256(body).hexdigest(),
        bundle_size=len(body),
        chunk_bytes=FORMAL_PROTOCOL_LOCK_GIT_CHUNK_BYTES,
        chunks=tuple(chunks),
    )
    publish_canonical_json_no_replace(index_output, index.to_dict())
    with checkout_formal_protocol_lock_git_snapshot(index_output) as (
        checkout,
        rebuilt,
    ):
        if rebuilt != index or _git(checkout, "rev-parse", "HEAD") != head:
            raise RuntimeError("published formal ProtocolLock Git snapshot changed")
    root_after, head_after, tree_after = _clean_git_identity(root)
    if root_after != root or head_after != head or tree_after != tree:
        raise RuntimeError("formal checkout changed while Git snapshot was published")
    return CanonicalJsonProofBinding.bind(index_output)


@dataclass(frozen=True)
class FormalProtocolLockSourceProofArtifact:
    schema_version: Literal[1]
    kind: Literal["formal_protocol_lock_source_proof_artifact"]
    protocol_id: str
    verified_ns: int
    expected_protocol_lock_sha256: str
    patch_manifest_relative_path: str
    english_protocol_relative_path: str
    chinese_protocol_relative_path: str
    git_snapshot_source: CanonicalJsonProofBinding
    runtime_authority_source: CanonicalJsonProofBinding
    tts_calibration_authority_source: CanonicalJsonProofBinding
    chronobelief_authority_source: CanonicalJsonProofBinding
    e1_recipe_anchor_authority_source: CanonicalJsonProofBinding
    content_verification_receipt_source: CanonicalJsonProofBinding
    burstgpt_shape_authority_source: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != FORMAL_PROTOCOL_LOCK_SOURCE_PROOF_KIND
            or type(self.protocol_id) is not str
            or not self.protocol_id
        ):
            raise ValueError("formal ProtocolLock source proof schema differs")
        if type(self.verified_ns) is not int or self.verified_ns < 1:
            raise ValueError("formal ProtocolLock source proof time is invalid")
        _sha256(
            "formal ProtocolLock source proof expected payload",
            self.expected_protocol_lock_sha256,
        )
        for field in (
            "patch_manifest_relative_path",
            "english_protocol_relative_path",
            "chinese_protocol_relative_path",
        ):
            _relative_source(getattr(self, field), label=field)
        bindings = tuple(
            getattr(self, field)
            for field in self.__dataclass_fields__
            if field.endswith("_source")
        )
        if any(type(row) is not CanonicalJsonProofBinding for row in bindings):
            raise TypeError("formal ProtocolLock proof source is not path-bound")
        if len({row.absolute_path for row in bindings}) != len(bindings):
            raise ValueError("formal ProtocolLock source proof aliases inputs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {}
        for field in self.__dataclass_fields__:
            item = getattr(self, field)
            value[field] = item.to_dict() if field.endswith("_source") else item
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            value,
            {*cls.__dataclass_fields__, "artifact_sha256"},
            label="formal ProtocolLock source proof",
        )
        declared = _sha256(
            "formal ProtocolLock source proof", row.pop("artifact_sha256")
        )
        for field in cls.__dataclass_fields__:
            if field.endswith("_source"):
                row[field] = CanonicalJsonProofBinding.from_dict(row[field])
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("formal ProtocolLock source proof digest differs")
        return artifact


def _repo_file_sha256(root: Path, relative: str, *, label: str) -> str:
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"formal ProtocolLock {label} is missing") from error
    if not stat.S_ISREG(status.st_mode) or path.is_symlink():
        raise ValueError(f"formal ProtocolLock {label} is not a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_release_root_snapshot(checkout: Path) -> str:
    manifest_path = checkout.joinpath(
        *PurePosixPath(_RELEASE_ROOT_MANIFEST_RELATIVE_PATH).parts
    )
    sidecar_path = checkout.joinpath(
        *PurePosixPath(_RELEASE_ROOT_SIDECAR_RELATIVE_PATH).parts
    )
    for label, path in (("manifest", manifest_path), ("sidecar", sidecar_path)):
        status = path.lstat()
        if not stat.S_ISREG(status.st_mode) or path.is_symlink():
            raise ValueError(f"formal ProtocolLock release-root {label} is unsafe")
    manifest_body = manifest_path.read_bytes()
    sidecar_body = sidecar_path.read_bytes()
    try:
        root = SourceReleaseEd25519Root.from_dict(
            json.loads(manifest_body.decode("utf-8"))
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            "formal ProtocolLock release-root manifest is invalid"
        ) from error
    expected_body = (
        json.dumps(
            root.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    file_sha256 = hashlib.sha256(manifest_body).hexdigest()
    loaded = load_source_release_ed25519_root()
    if (
        manifest_body != expected_body
        or sidecar_body != f"{file_sha256}\n".encode("ascii")
        or root != loaded.root
        or root.sha256 != loaded.semantic_sha256
        or file_sha256 != loaded.file_sha256
        or hashlib.sha256(sidecar_body).hexdigest() != loaded.sidecar_file_sha256
    ):
        raise ValueError("formal ProtocolLock release root differs from Git snapshot")
    return root.sha256


def _build_protocol_lock(
    artifact: FormalProtocolLockSourceProofArtifact,
    *,
    now_ns: int,
) -> ProtocolLock:
    if type(now_ns) is not int or now_ns < artifact.verified_ns:
        raise ValueError("formal ProtocolLock proof replay predates publication")
    for field in artifact.__dataclass_fields__:
        if field.endswith("_source"):
            _stable_json(getattr(artifact, field), label=f"ProtocolLock {field}")
    with checkout_formal_protocol_lock_git_snapshot(
        artifact.git_snapshot_source.absolute_path
    ) as (checkout, snapshot):
        release_root_sha256 = _verify_release_root_snapshot(checkout)
        runtime = formal_runtime_authority_manifest_from_dict(
            _stable_json(
                artifact.runtime_authority_source,
                label="ProtocolLock runtime authority",
            )
        )
        if runtime != build_source_formal_runtime_authority_manifest(checkout):
            raise ValueError(
                "formal ProtocolLock runtime authority differs from Git snapshot"
            )
        tts = load_tts_calibration_authority_artifact(
            artifact.tts_calibration_authority_source.absolute_path
        )
        chronobelief = load_chronobelief_authority_artifact(
            artifact.chronobelief_authority_source.absolute_path
        )
        e1 = load_e1_recipe_anchor_authority_artifact(
            artifact.e1_recipe_anchor_authority_source.absolute_path
        )
        content_receipt = ContentVerificationReceipt.from_dict(
            _stable_json(
                artifact.content_verification_receipt_source,
                label="ProtocolLock content receipt",
            )
        )
        verified_content = content_receipt.revalidate_formal_scope(current_ns=now_ns)
        prepared = tuple(
            row
            for row in verified_content
            if type(row) is VerifiedPreparedModelContentRelease
        )
        workload = tuple(
            row
            for row in verified_content
            if type(row) is VerifiedReleaseWorkloadSources
        )
        e0 = tuple(
            row
            for row in verified_content
            if type(row) is VerifiedDatasetContentRelease
            and row.authority_domain == "e0_task_native"
        )
        if len(prepared) != 1 or len(workload) != 1 or len(e0) != 1:
            raise ValueError("formal ProtocolLock content authority coverage differs")
        roots = {row.authorization.root_manifest_sha256 for row in verified_content}
        if len(roots) != 1 or roots != {release_root_sha256}:
            raise ValueError("formal ProtocolLock content roots differ")
        burstgpt = BurstGptShapeAuthority.from_dict(
            _stable_json(
                artifact.burstgpt_shape_authority_source,
                label="ProtocolLock BurstGPT authority",
            )
        )
        derived_burstgpt = derive_burstgpt_shape_authority_from_content_receipt(
            content_receipt,
            current_ns=now_ns,
        )
        if burstgpt != derived_burstgpt:
            raise ValueError("formal ProtocolLock BurstGPT authority differs")
        qualifications = code_owned_qualification_source_identities()
        native = qualifications["native_runtime"]
        compile_identity = qualifications["compile"]
        exactness = qualifications["exactness"]
        return ProtocolLock(
            schema_version=4,
            protocol_id=artifact.protocol_id,
            code_git_head=snapshot.git_head,
            code_git_tree=snapshot.git_tree,
            patch_manifest_sha256=_repo_file_sha256(
                checkout,
                artifact.patch_manifest_relative_path,
                label="patch manifest",
            ),
            registry_sha256=build_industrial_registry().sha256,
            english_protocol_sha256=_repo_file_sha256(
                checkout,
                artifact.english_protocol_relative_path,
                label="English protocol",
            ),
            chinese_protocol_sha256=_repo_file_sha256(
                checkout,
                artifact.chinese_protocol_relative_path,
                label="Chinese protocol",
            ),
            tts_calibration_authority_sha256=tts.authority.sha256,
            chronobelief_authority_sha256=chronobelief.authority.sha256,
            e1_recipe_anchor_authority_sha256=e1.authority.sha256,
            e2_recipe_grid_authority_sha256=(default_e2_recipe_grid_authority().sha256),
            formal_runtime_authority_manifest_sha256=runtime.sha256,
            offline_release_trust_root_sha256=next(iter(roots)),
            prepared_model_content_authorization_sha256=(
                prepared[0].authorization_sha256
            ),
            formal_workload_e3a_authorization_sha256=(workload[0].authorization_sha256),
            formal_workload_e0_authorization_sha256=e0[0].authorization_sha256,
            burstgpt_shape_authorization_sha256=burstgpt.sha256,
            native_runtime_qualification_protocol_sha256=native[0],
            native_runtime_qualification_runner_sha256=native[1],
            native_runtime_qualification_test_set_sha256=native[2],
            compile_qualification_protocol_sha256=compile_identity[0],
            compile_qualification_runner_sha256=compile_identity[1],
            compile_qualification_test_set_sha256=compile_identity[2],
            exactness_qualification_protocol_sha256=exactness[0],
            exactness_qualification_runner_sha256=exactness[1],
            exactness_qualification_test_set_sha256=exactness[2],
        )


def bind_formal_protocol_lock_source_proof_artifact(
    *,
    protocol_id: str,
    git_snapshot_path: str | Path,
    patch_manifest_relative_path: str,
    english_protocol_relative_path: str,
    chinese_protocol_relative_path: str,
    runtime_authority_path: str | Path,
    tts_calibration_authority_path: str | Path,
    chronobelief_authority_path: str | Path,
    e1_recipe_anchor_authority_path: str | Path,
    content_verification_receipt_path: str | Path,
    burstgpt_shape_authority_path: str | Path,
    now_ns: int,
) -> FormalProtocolLockSourceProofArtifact:
    bindings = {
        "git_snapshot_source": CanonicalJsonProofBinding.bind(git_snapshot_path),
        "runtime_authority_source": CanonicalJsonProofBinding.bind(
            runtime_authority_path
        ),
        "tts_calibration_authority_source": CanonicalJsonProofBinding.bind(
            tts_calibration_authority_path
        ),
        "chronobelief_authority_source": CanonicalJsonProofBinding.bind(
            chronobelief_authority_path
        ),
        "e1_recipe_anchor_authority_source": CanonicalJsonProofBinding.bind(
            e1_recipe_anchor_authority_path
        ),
        "content_verification_receipt_source": CanonicalJsonProofBinding.bind(
            content_verification_receipt_path
        ),
        "burstgpt_shape_authority_source": CanonicalJsonProofBinding.bind(
            burstgpt_shape_authority_path
        ),
    }
    placeholder = FormalProtocolLockSourceProofArtifact(
        schema_version=1,
        kind=FORMAL_PROTOCOL_LOCK_SOURCE_PROOF_KIND,
        protocol_id=protocol_id,
        verified_ns=now_ns,
        expected_protocol_lock_sha256="0" * 64,
        patch_manifest_relative_path=_relative_source(
            patch_manifest_relative_path,
            label="patch manifest",
        ),
        english_protocol_relative_path=_relative_source(
            english_protocol_relative_path,
            label="English protocol",
        ),
        chinese_protocol_relative_path=_relative_source(
            chinese_protocol_relative_path,
            label="Chinese protocol",
        ),
        **bindings,
    )
    lock = _build_protocol_lock(placeholder, now_ns=now_ns)
    artifact = FormalProtocolLockSourceProofArtifact(
        **{
            **{
                field: getattr(placeholder, field)
                for field in placeholder.__dataclass_fields__
            },
            "expected_protocol_lock_sha256": lock.sha256,
        }
    )
    if _build_protocol_lock(artifact, now_ns=now_ns) != lock:
        raise RuntimeError("formal ProtocolLock source proof changed during binding")
    return artifact


def publish_formal_protocol_lock_source_proof_artifact(
    artifact: FormalProtocolLockSourceProofArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not FormalProtocolLockSourceProofArtifact:
        raise TypeError("formal ProtocolLock proof publisher requires exact input")
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    lock = revalidate_formal_protocol_lock_source_proof_artifact(
        binding.absolute_path,
        now_ns=artifact.verified_ns,
    )
    if lock.sha256 != artifact.expected_protocol_lock_sha256:
        raise RuntimeError("published formal ProtocolLock proof changed")
    return binding


def revalidate_formal_protocol_lock_source_proof_artifact(
    artifact_path: str | Path,
    *,
    now_ns: int,
    relocatable_bundle_manifest_path: str | Path | None = None,
) -> ProtocolLock:
    if relocatable_bundle_manifest_path is not None:
        from lightcone_spec.runtime.relocatable_evidence import (
            activate_relocatable_evidence_bundle,
        )

        with activate_relocatable_evidence_bundle(
            relocatable_bundle_manifest_path
        ) as bundle:
            entry = str(artifact_path)
            if entry not in bundle.artifact.entry_remote_absolute_paths:
                raise ValueError("formal ProtocolLock proof is not a bundle entry")
            return revalidate_formal_protocol_lock_source_proof_artifact(
                entry,
                now_ns=now_ns,
            )
    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalProtocolLockSourceProofArtifact.from_dict(binding.reopen())
    lock = _build_protocol_lock(artifact, now_ns=now_ns)
    if (
        lock.sha256 != artifact.expected_protocol_lock_sha256
        or CanonicalJsonProofBinding.bind(binding.absolute_path) != binding
    ):
        raise ValueError("formal ProtocolLock proof replay differs")
    return lock


__all__ = (
    "FORMAL_PROTOCOL_LOCK_GIT_CHUNK_BYTES",
    "FORMAL_PROTOCOL_LOCK_GIT_SNAPSHOT_KIND",
    "FORMAL_PROTOCOL_LOCK_SOURCE_PROOF_KIND",
    "FormalProtocolLockGitSnapshotIndex",
    "FormalProtocolLockSourceProofArtifact",
    "bind_formal_protocol_lock_source_proof_artifact",
    "checkout_formal_protocol_lock_git_snapshot",
    "publish_formal_protocol_lock_git_snapshot",
    "publish_formal_protocol_lock_source_proof_artifact",
    "revalidate_formal_protocol_lock_source_proof_artifact",
)
