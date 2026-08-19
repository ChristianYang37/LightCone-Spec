"""Exact rolling archive, remote eviction, and no-replace restoration.

This module is the destructive boundary for v03 rolling archives. An archive
manifest proves one sealed candidate tree, formal_remote_archive proves that an
exact local copy and a full rehydrate check exist, and the DAG driver's
retained-future manifest identifies everything that must remain remote. Only
then may an immutable, inode-bound eviction plan be published.

The executor never removes directories and has no recursive-delete primitive.
It calls os.unlink only for a file named in the published plan after a fresh
lstat/device/inode/size/SHA-256 check. Per-file progress records make a restart
deterministic; an unlink that was not durably recorded is treated as ambiguous
partial progress and stops the scheduler.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Literal

from lightcone_spec.orchestration.experiment_operator import (
    REMOTE_SPOOL_SAFETY_RESERVE_BYTES,
    ArchiveRequest,
    ArchiveStepReceipt,
    ExperimentOperatorStore,
    RemoteEvictionAuthorization,
    SingletonOperatorLock,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    canonical_json_bytes,
)
from lightcone_spec.orchestration.formal_remote_archive import (
    RemoteArchiveResult,
    load_remote_archive_result,
)

_ARCHIVE_MANIFEST_NAME = "sha256_manifest.json"
_ARCHIVE_MANIFEST_KIND = "formal_archive_sha256_manifest"
_EVICTION_PLAN_KIND = "formal_remote_eviction_plan"
_EVICTION_RECEIPT_KIND = "formal_remote_eviction_receipt"
_EVICTION_PROGRESS_KIND = "formal_remote_eviction_file_progress"
_RESTORE_RECEIPT_KIND = "formal_archive_restore_receipt"
_STREAM_RESTORE_PROGRESS_KIND = "formal_remote_stream_restore_progress"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VERSION_TOKEN_V03 = re.compile(r"(?:^|[-_.])v03(?:$|[-_.])", re.IGNORECASE)
_VERSION_TOKEN_V02 = re.compile(r"(?:^|[-_.])v02(?:$|[-_.])", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_TREE_PARTS = frozenset(
    {
        ".cache",
        "cache",
        "caches",
        "model",
        "models",
        "model-cache",
        "model_cache",
        "weights",
    }
)
_OPERATOR_DATABASE_NAMES = frozenset(
    {
        "operator.db",
        "operator.db-shm",
        "operator.db-wal",
        "operator.sqlite",
        "operator.sqlite-shm",
        "operator.sqlite-wal",
        "operator.sqlite3",
        "operator.sqlite3-shm",
        "operator.sqlite3-wal",
    }
)


class FormalRollingArchiveError(RuntimeError):
    """A rolling-archive identity or safety gate failed closed."""


class SimulatedEvictionCrash(BaseException):
    """Test-only crash signal deliberately not caught by the executor."""


@dataclass(frozen=True)
class ArchiveManifestFile:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        pure = PurePosixPath(self.path)
        if (
            type(self.path) is not str
            or not self.path
            or pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != self.path
            or pure.name == _ARCHIVE_MANIFEST_NAME
        ):
            raise ValueError("archive manifest path is not safe relative POSIX text")
        _require_sha256(self.sha256, "archive member SHA-256")
        _require_nonnegative_int(self.size_bytes, "archive member size")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FormalArchiveSha256Manifest:
    schema_version: Literal[1]
    kind: Literal["formal_archive_sha256_manifest"]
    files: tuple[ArchiveManifestFile, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != _ARCHIVE_MANIFEST_KIND:
            raise ValueError("archive manifest identity differs")
        if type(self.files) is not tuple or not self.files:
            raise ValueError("archive manifest must contain at least one file")
        paths = tuple(row.path for row in self.files)
        if any(
            type(row) is not ArchiveManifestFile for row in self.files
        ) or paths != tuple(sorted(set(paths))):
            raise ValueError("archive manifest paths must be exact, unique, and sorted")

    @property
    def payload_bytes(self) -> int:
        return sum(row.size_bytes for row in self.files)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "files": [row.to_dict() for row in self.files],
        }

    @classmethod
    def from_dict(cls, value: object) -> FormalArchiveSha256Manifest:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "kind",
            "files",
        }:
            raise ValueError("archive manifest fields differ")
        raw_files = value["files"]
        if type(raw_files) is not list:
            raise TypeError("archive manifest files must be a list")
        rows = []
        for raw in raw_files:
            if type(raw) is not dict or set(raw) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise ValueError("archive manifest row fields differ")
            rows.append(ArchiveManifestFile(**raw))
        return cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            files=tuple(rows),
        )


@dataclass(frozen=True)
class PublishedArchiveManifest:
    path: str
    manifest_sha256: str
    checked_file_count: int
    predicted_payload_bytes: int

    def __post_init__(self) -> None:
        _absolute_normalized(self.path, "archive manifest path")
        _require_sha256(self.manifest_sha256, "archive manifest SHA-256")
        _require_positive_int(self.checked_file_count, "archive file count")
        _require_nonnegative_int(
            self.predicted_payload_bytes, "archive predicted payload bytes"
        )


@dataclass(frozen=True)
class StagedArchiveChain:
    request_path: str
    result_path: str
    authorization_path: str
    request_sha256: str
    result_sha256: str
    authorization_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("staged request", self.request_path),
            ("staged result", self.result_path),
            ("staged authorization", self.authorization_path),
        ):
            _absolute_normalized(value, f"{label} path")
        for label, value in (
            ("request", self.request_sha256),
            ("result", self.result_sha256),
            ("authorization", self.authorization_sha256),
        ):
            _require_sha256(value, f"staged {label} SHA-256")


@dataclass(frozen=True)
class EvictionFileBinding:
    absolute_path: str
    archive_relative_path: str
    device: int
    inode: int
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _absolute_normalized(self.absolute_path, "eviction file path")
        ArchiveManifestFile(
            path=self.archive_relative_path,
            sha256=self.sha256,
            size_bytes=self.size_bytes,
        )
        _require_nonnegative_int(self.device, "eviction file device")
        _require_positive_int(self.inode, "eviction file inode")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> EvictionFileBinding:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("eviction file binding fields differ")
        return cls(**value)


@dataclass(frozen=True)
class RemoteEvictionPlan:
    schema_version: Literal[1]
    kind: Literal["formal_remote_eviction_plan"]
    run_id: str
    run_root: str
    node: str
    archive_id: str
    archive_candidate_root: str
    archive_candidate_root_device: int
    archive_candidate_root_inode: int
    archive_manifest_path: str
    archive_manifest_sha256: str
    archive_request_sha256: str
    remote_archive_result_sha256: str
    archive_authorization_sha256: str
    retained_dependency_manifest_path: str
    retained_dependency_manifest_sha256: str
    operator_checkpoint_sha256: str
    operator_snapshot_sha256: str
    files: tuple[EvictionFileBinding, ...]
    planned_bytes: int
    created_at_ns: int
    remote_eviction_authorized: Literal[True]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _EVICTION_PLAN_KIND
            or self.remote_eviction_authorized is not True
            or not _safe_component(self.run_id)
            or not _safe_component(self.node)
            or not self.archive_id
        ):
            raise ValueError("remote eviction plan identity differs")
        run_root = _absolute_normalized(self.run_root, "eviction plan run root")
        candidate = _absolute_normalized(
            self.archive_candidate_root, "eviction plan candidate root"
        )
        if candidate == run_root or not candidate.is_relative_to(run_root):
            raise ValueError("eviction candidate lies outside its exact run root")
        manifest_path = _absolute_normalized(
            self.archive_manifest_path, "eviction plan manifest path"
        )
        if manifest_path != candidate / _ARCHIVE_MANIFEST_NAME:
            raise ValueError("eviction plan manifest path differs from candidate")
        retained_path = _absolute_normalized(
            self.retained_dependency_manifest_path,
            "retained dependency manifest path",
        )
        if retained_path.is_relative_to(candidate):
            raise ValueError("retained dependency manifest is inside eviction scope")
        _require_nonnegative_int(
            self.archive_candidate_root_device, "candidate root device"
        )
        _require_positive_int(self.archive_candidate_root_inode, "candidate root inode")
        for label, value in (
            ("archive manifest", self.archive_manifest_sha256),
            ("archive request", self.archive_request_sha256),
            ("remote archive result", self.remote_archive_result_sha256),
            ("archive authorization", self.archive_authorization_sha256),
            (
                "retained dependency manifest",
                self.retained_dependency_manifest_sha256,
            ),
            ("operator checkpoint", self.operator_checkpoint_sha256),
            ("operator snapshot", self.operator_snapshot_sha256),
        ):
            _require_sha256(value, f"{label} SHA-256")
        _require_positive_int(self.created_at_ns, "eviction plan creation time")
        _require_nonnegative_int(self.planned_bytes, "planned eviction bytes")
        paths = tuple(row.absolute_path for row in self.files)
        relatives = tuple(row.archive_relative_path for row in self.files)
        if (
            type(self.files) is not tuple
            or not self.files
            or any(type(row) is not EvictionFileBinding for row in self.files)
            or paths != tuple(sorted(set(paths)))
            or len(set(relatives)) != len(relatives)
            or self.planned_bytes != sum(row.size_bytes for row in self.files)
        ):
            raise ValueError("eviction plan file coverage differs")
        for row in self.files:
            member = candidate.joinpath(*PurePosixPath(row.archive_relative_path).parts)
            if member != Path(row.absolute_path):
                raise ValueError("eviction file is not bound to its archive path")

    @property
    def sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "files": [row.to_dict() for row in self.files],
        }

    @classmethod
    def from_dict(cls, value: object) -> RemoteEvictionPlan:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("remote eviction plan fields differ")
        row = dict(value)
        raw_files = row.pop("files")
        if type(raw_files) is not list:
            raise TypeError("remote eviction plan files must be a list")
        return cls(
            **row,
            files=tuple(EvictionFileBinding.from_dict(item) for item in raw_files),
        )


@dataclass(frozen=True)
class RemoteEvictionGate:
    remote_eviction_authorized: bool
    blocker: str | None
    plan: RemoteEvictionPlan | None

    def __post_init__(self) -> None:
        if self.remote_eviction_authorized:
            if self.blocker is not None or type(self.plan) is not RemoteEvictionPlan:
                raise ValueError("authorized eviction gate lacks an exact plan")
        elif self.blocker is None or self.plan is not None:
            raise ValueError("blocked eviction gate identity differs")


@dataclass(frozen=True)
class EvictedFileReceipt:
    absolute_path: str
    archive_relative_path: str
    size_bytes: int
    sha256: str
    deleted_at_ns: int

    def __post_init__(self) -> None:
        _absolute_normalized(self.absolute_path, "deleted file path")
        ArchiveManifestFile(
            self.archive_relative_path,
            self.sha256,
            self.size_bytes,
        )
        _require_positive_int(self.deleted_at_ns, "file deletion time")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> EvictedFileReceipt:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("deleted file receipt fields differ")
        return cls(**value)


EvictionReceiptStatus = Literal["COMPLETE", "FAILED_ZERO", "FAILED_PARTIAL"]


@dataclass(frozen=True)
class RemoteEvictionReceipt:
    schema_version: Literal[1]
    kind: Literal["formal_remote_eviction_receipt"]
    plan_sha256: str
    archive_id: str
    archive_authorization_sha256: str
    status: EvictionReceiptStatus
    deleted_files: tuple[EvictedFileReceipt, ...]
    deleted_bytes: int
    failure_code: str | None
    failure_path: str | None
    missing_unrecorded_files: tuple[str, ...]
    scheduler_stop_requested: bool
    scheduler_stop_succeeded: bool
    finished_at_ns: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != _EVICTION_RECEIPT_KIND:
            raise ValueError("remote eviction receipt identity differs")
        _require_sha256(self.plan_sha256, "eviction plan SHA-256")
        _require_sha256(
            self.archive_authorization_sha256, "archive authorization SHA-256"
        )
        if not self.archive_id or self.status not in {
            "COMPLETE",
            "FAILED_ZERO",
            "FAILED_PARTIAL",
        }:
            raise ValueError("remote eviction receipt status differs")
        _require_nonnegative_int(self.deleted_bytes, "deleted bytes")
        _require_positive_int(self.finished_at_ns, "eviction finish time")
        paths = tuple(row.absolute_path for row in self.deleted_files)
        if (
            type(self.deleted_files) is not tuple
            or any(type(row) is not EvictedFileReceipt for row in self.deleted_files)
            or paths != tuple(sorted(set(paths)))
            or self.deleted_bytes != sum(row.size_bytes for row in self.deleted_files)
            or self.missing_unrecorded_files
            != tuple(sorted(set(self.missing_unrecorded_files)))
        ):
            raise ValueError("remote eviction receipt coverage differs")
        for path in self.missing_unrecorded_files:
            _absolute_normalized(path, "missing unrecorded eviction path")
        if self.status == "COMPLETE":
            if (
                self.failure_code is not None
                or self.failure_path is not None
                or self.missing_unrecorded_files
                or self.scheduler_stop_requested
                or self.scheduler_stop_succeeded
            ):
                raise ValueError("complete eviction receipt carries failure state")
        else:
            if (
                not self.failure_code
                or not self.scheduler_stop_requested
                or (
                    self.status == "FAILED_ZERO"
                    and (self.deleted_files or self.missing_unrecorded_files)
                )
                or (
                    self.status == "FAILED_PARTIAL"
                    and not (self.deleted_files or self.missing_unrecorded_files)
                )
            ):
                raise ValueError("failed eviction receipt state differs")
            if self.failure_path is not None:
                _absolute_normalized(self.failure_path, "eviction failure path")

    @property
    def sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "deleted_files": [row.to_dict() for row in self.deleted_files],
            "missing_unrecorded_files": list(self.missing_unrecorded_files),
        }

    @classmethod
    def from_dict(cls, value: object) -> RemoteEvictionReceipt:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("remote eviction receipt fields differ")
        row = dict(value)
        raw_files = row.pop("deleted_files")
        raw_missing = row.pop("missing_unrecorded_files")
        if type(raw_files) is not list or type(raw_missing) is not list:
            raise TypeError("remote eviction receipt arrays differ")
        return cls(
            **row,
            deleted_files=tuple(
                EvictedFileReceipt.from_dict(item) for item in raw_files
            ),
            missing_unrecorded_files=tuple(raw_missing),
        )


@dataclass(frozen=True)
class RestoredFileReceipt:
    absolute_path: str
    archive_relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _absolute_normalized(self.absolute_path, "restored file path")
        ArchiveManifestFile(
            self.archive_relative_path,
            self.sha256,
            self.size_bytes,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> RestoredFileReceipt:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("restored file receipt fields differ")
        return cls(**value)


@dataclass(frozen=True)
class RemoteStreamRestoreProgress:
    schema_version: Literal[1]
    kind: Literal["formal_remote_stream_restore_progress"]
    plan_sha256: str
    eviction_receipt_sha256: str
    remote_archive_result_sha256: str
    file_index: int
    file: RestoredFileReceipt
    disposition: Literal["RESTORED", "ALREADY_PRESENT"]
    restored_at_ns: int
    existing_file_overwritten: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _STREAM_RESTORE_PROGRESS_KIND
            or self.disposition not in {"RESTORED", "ALREADY_PRESENT"}
            or type(self.file) is not RestoredFileReceipt
            or self.existing_file_overwritten is not False
        ):
            raise ValueError("remote stream restore progress identity differs")
        for label, value in (
            ("plan", self.plan_sha256),
            ("eviction receipt", self.eviction_receipt_sha256),
            ("remote archive result", self.remote_archive_result_sha256),
        ):
            _require_sha256(value, f"stream restore {label} SHA-256")
        _require_nonnegative_int(self.file_index, "stream restore file index")
        _require_positive_int(self.restored_at_ns, "stream restore time")

    @property
    def sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "file": self.file.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> RemoteStreamRestoreProgress:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("remote stream restore progress fields differ")
        row = dict(value)
        row["file"] = RestoredFileReceipt.from_dict(row["file"])
        return cls(**row)


@dataclass(frozen=True)
class ArchiveRestoreReceipt:
    schema_version: Literal[1]
    kind: Literal["formal_archive_restore_receipt"]
    plan_sha256: str
    archive_id: str
    archive_manifest_sha256: str
    remote_archive_result_sha256: str
    restored_files: tuple[RestoredFileReceipt, ...]
    already_present_files: tuple[RestoredFileReceipt, ...]
    restored_bytes: int
    completed_at_ns: int
    existing_files_overwritten: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _RESTORE_RECEIPT_KIND
            or self.existing_files_overwritten is not False
            or not self.archive_id
        ):
            raise ValueError("archive restore receipt identity differs")
        for label, value in (
            ("eviction plan", self.plan_sha256),
            ("archive manifest", self.archive_manifest_sha256),
            ("remote archive result", self.remote_archive_result_sha256),
        ):
            _require_sha256(value, f"{label} SHA-256")
        _require_nonnegative_int(self.restored_bytes, "restored bytes")
        _require_positive_int(self.completed_at_ns, "restore completion time")
        restored = tuple(row.absolute_path for row in self.restored_files)
        present = tuple(row.absolute_path for row in self.already_present_files)
        if (
            any(type(row) is not RestoredFileReceipt for row in self.restored_files)
            or any(
                type(row) is not RestoredFileReceipt
                for row in self.already_present_files
            )
            or restored != tuple(sorted(set(restored)))
            or present != tuple(sorted(set(present)))
            or set(restored) & set(present)
            or self.restored_bytes != sum(row.size_bytes for row in self.restored_files)
        ):
            raise ValueError("archive restore receipt coverage differs")

    @property
    def sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "restored_files": [row.to_dict() for row in self.restored_files],
            "already_present_files": [
                row.to_dict() for row in self.already_present_files
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> ArchiveRestoreReceipt:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("archive restore receipt fields differ")
        row = dict(value)
        raw_restored = row.pop("restored_files")
        raw_present = row.pop("already_present_files")
        if type(raw_restored) is not list or type(raw_present) is not list:
            raise TypeError("archive restore receipt arrays differ")
        return cls(
            **row,
            restored_files=tuple(
                RestoredFileReceipt.from_dict(item) for item in raw_restored
            ),
            already_present_files=tuple(
                RestoredFileReceipt.from_dict(item) for item in raw_present
            ),
        )


def publish_formal_archive_sha256_manifest(
    *,
    run_root: str | Path,
    candidate_root: str | Path,
    retained_dependency_manifest_path: str | Path,
    lock_path: str | Path,
) -> PublishedArchiveManifest:
    """Publish or replay one exact no-replace manifest for a sealed v03 tree."""

    root = _existing_directory(run_root, "v03 run root")
    candidate = _existing_directory(candidate_root, "archive candidate root")
    retained_path = _existing_file(
        retained_dependency_manifest_path, "retained dependency manifest"
    )
    retained = _load_retained_dependency_manifest(retained_path)
    _validate_retained_scope(retained, root, candidate)
    _validate_v03_scope(root, candidate, retained.run_id)
    if retained_path.is_relative_to(candidate):
        raise FormalRollingArchiveError(
            "retained dependency manifest cannot be inside an archive candidate"
        )
    output = candidate / _ARCHIVE_MANIFEST_NAME
    lock = _outside_candidate(lock_path, candidate, "archive manifest lock")
    with SingletonOperatorLock(lock):
        if os.path.lexists(output):
            manifest = load_formal_archive_sha256_manifest(output, verify_root=True)
            return PublishedArchiveManifest(
                path=str(output),
                manifest_sha256=manifest.sha256,
                checked_file_count=len(manifest.files),
                predicted_payload_bytes=manifest.payload_bytes,
            )
        manifest = FormalArchiveSha256Manifest(
            schema_version=1,
            kind=_ARCHIVE_MANIFEST_KIND,
            files=_scan_archive_tree(candidate, allow_root_manifest=False),
        )
        _verify_manifest_tree(candidate, manifest, require_manifest=False)
        _publish_canonical_no_replace(output, manifest.to_dict())
        rebound = load_formal_archive_sha256_manifest(output, verify_root=True)
        if rebound != manifest:
            raise FormalRollingArchiveError(
                "published archive manifest differs from the sealed tree"
            )
        return PublishedArchiveManifest(
            path=str(output),
            manifest_sha256=manifest.sha256,
            checked_file_count=len(manifest.files),
            predicted_payload_bytes=manifest.payload_bytes,
        )


def load_formal_archive_sha256_manifest(
    path: str | Path,
    *,
    verify_root: bool = True,
) -> FormalArchiveSha256Manifest:
    source = _existing_file(path, "formal archive SHA-256 manifest")
    if source.name != _ARCHIVE_MANIFEST_NAME:
        raise FormalRollingArchiveError("archive manifest has a noncanonical name")
    manifest = FormalArchiveSha256Manifest.from_dict(
        _read_canonical_object(source, "formal archive SHA-256 manifest")
    )
    if verify_root:
        _verify_manifest_tree(source.parent, manifest, require_manifest=True)
    return manifest


def build_archive_request(
    *,
    manifest_path: str | Path,
    retained_dependency_manifest_path: str | Path,
    local_results_root: str | Path,
    wave: str,
    archive_id: str | None = None,
    safe_boundary: str | None = None,
    cell_id: str | None = None,
    attempt: int | None = None,
) -> ArchiveRequest:
    """Bind a sealed candidate to results/<run>/<node>/<wave> partial/final."""

    manifest_source = _existing_file(manifest_path, "archive manifest")
    manifest = load_formal_archive_sha256_manifest(manifest_source, verify_root=True)
    retained_path = _existing_file(
        retained_dependency_manifest_path, "retained dependency manifest"
    )
    retained = _load_retained_dependency_manifest(retained_path)
    run_root = _existing_directory(retained.run_root, "retained v03 run root")
    candidate = manifest_source.parent
    _validate_retained_scope(retained, run_root, candidate)
    _validate_v03_scope(run_root, candidate, retained.run_id)
    if not _safe_component(wave):
        raise ValueError("archive wave must be one safe path component")
    results = _absolute_normalized(local_results_root, "local results root")
    if results.name != "results":
        raise ValueError("local archive root must be the exact results directory")
    if results.exists() and (results.is_symlink() or not results.is_dir()):
        raise ValueError("local results root is not a safe directory")
    if _paths_overlap(results, candidate) or _paths_overlap(results, run_root):
        raise ValueError("local results root overlaps the remote run scope")
    local_node_root = results / retained.run_id / retained.node
    partial = local_node_root / f"{wave}.partial"
    final = local_node_root / f"{wave}.final"
    request_archive_id = archive_id or (
        f"{retained.run_id}.{retained.node}.{wave}.{manifest.sha256[:16]}"
    )
    boundary = safe_boundary or (
        f"{retained.node}:reduced:{retained.completion.sha256}"
    )
    return ArchiveRequest(
        archive_id=request_archive_id,
        safe_boundary=boundary,
        remote_payload_root=str(candidate),
        local_partial_root=str(partial),
        local_final_root=str(final),
        remote_manifest_sha256=manifest.sha256,
        predicted_payload_bytes=manifest.payload_bytes,
        cell_id=cell_id,
        attempt=attempt,
    )


def publish_archive_request(
    path: str | Path,
    request: ArchiveRequest,
    *,
    lock_path: str | Path,
) -> ArchiveRequest:
    """Publish one canonical, immutable ArchiveRequest with idempotent replay."""

    if type(request) is not ArchiveRequest:
        raise TypeError("archive request must use the exact operator type")
    candidate = Path(request.remote_payload_root)
    output = _outside_candidate(path, candidate, "archive request output")
    lock = _outside_candidate(lock_path, candidate, "archive request lock")
    with SingletonOperatorLock(lock):
        if os.path.lexists(output):
            existing = load_archive_request(output)
            if existing != request:
                raise FormalRollingArchiveError("archive request is immutable")
            return existing
        _publish_canonical_no_replace(output, asdict(request))
        return load_archive_request(output)


def load_archive_request(path: str | Path) -> ArchiveRequest:
    value = _read_canonical_object(path, "archive request")
    if set(value) != set(ArchiveRequest.__dataclass_fields__):
        raise ValueError("archive request fields differ")
    return ArchiveRequest(**value)


def probe_retained_archive_boundary(
    *,
    run_root: str | Path,
    node: str,
    ordinal: int,
) -> dict[str, object]:
    """Probe one code-owned retained path without scanning the remote tree."""

    root = _existing_directory(run_root, "remote v03 run root")
    if not _safe_component(node):
        raise ValueError("rolling archive node is not a safe component")
    _require_nonnegative_int(ordinal, "rolling archive node ordinal")
    retained_path = (
        root
        / "formal-dag-nodes"
        / f"{ordinal:02d}-{node}"
        / "reduction"
        / "retained-future-dependency-manifest.json"
    )
    if not os.path.lexists(retained_path):
        return {
            "schema_version": 1,
            "kind": "formal_rolling_archive_boundary_probe",
            "run_id": root.name,
            "node": node,
            "ordinal": ordinal,
            "status": "ABSENT",
            "retained_manifest_path": str(retained_path),
        }
    retained = _load_retained_dependency_manifest(
        _existing_file(retained_path, "retained dependency manifest")
    )
    if retained.node != node or retained.run_root != str(root):
        raise FormalRollingArchiveError("retained boundary belongs to another node")
    if len(retained.archive_candidate_roots) != 1:
        raise FormalRollingArchiveError(
            "rolling companion requires one exact candidate root per node"
        )
    candidate = _existing_directory(
        retained.archive_candidate_roots[0], "archive candidate root"
    )
    _validate_retained_scope(retained, root, candidate)
    _validate_v03_scope(root, candidate, retained.run_id)
    return {
        "schema_version": 1,
        "kind": "formal_rolling_archive_boundary_probe",
        "run_id": retained.run_id,
        "node": node,
        "ordinal": ordinal,
        "status": "AVAILABLE",
        "retained_manifest_path": str(retained_path),
        "retained_manifest_sha256": retained.sha256,
        "archive_candidate_root": str(candidate),
    }


def prepare_rolling_archive_node(
    *,
    run_root: str | Path,
    retained_dependency_manifest_path: str | Path,
    local_results_root: str | Path,
    wave: str,
    request_output_path: str | Path,
    lock_path: str | Path,
) -> ArchiveRequest:
    """Idempotently produce the manifest and request for one sealed node."""

    retained_path = _existing_file(
        retained_dependency_manifest_path, "retained dependency manifest"
    )
    retained = _load_retained_dependency_manifest(retained_path)
    if len(retained.archive_candidate_roots) != 1:
        raise FormalRollingArchiveError(
            "rolling companion requires one exact candidate root per node"
        )
    candidate = Path(retained.archive_candidate_roots[0])
    published = publish_formal_archive_sha256_manifest(
        run_root=run_root,
        candidate_root=candidate,
        retained_dependency_manifest_path=retained_path,
        lock_path=lock_path,
    )
    request = build_archive_request(
        manifest_path=published.path,
        retained_dependency_manifest_path=retained_path,
        local_results_root=local_results_root,
        wave=wave,
    )
    return publish_archive_request(
        request_output_path,
        request,
        lock_path=Path(lock_path).with_name(f"{Path(lock_path).name}.request"),
    )


def stage_remote_archive_chain(
    *,
    request_path: str | Path,
    result_value: Mapping[str, Any],
    authorization_value: Mapping[str, Any],
    result_output_path: str | Path,
    authorization_output_path: str | Path,
    lock_path: str | Path,
) -> StagedArchiveChain:
    """Publish canonical local result/authorization metadata on the remote host."""

    request_source = _existing_file(request_path, "staged archive request")
    request = load_archive_request(request_source)
    candidate = Path(request.remote_payload_root)
    result_row = dict(_plain_mapping(result_value, "remote archive result"))
    expected_result_sha = result_row.pop("result_sha256", None)
    result = RemoteArchiveResult(**result_row)
    if expected_result_sha != result.sha256:
        raise FormalRollingArchiveError("staged remote archive result digest differs")
    authorization_row = dict(
        _plain_mapping(authorization_value, "remote eviction authorization")
    )
    authorization_row.pop("remote_deletion_performed", None)
    expected_authorization_sha = authorization_row.pop("authorization_sha256", None)
    authorization = RemoteEvictionAuthorization(**authorization_row)
    authorization_sha = remote_eviction_authorization_sha256(authorization)
    if expected_authorization_sha not in {None, authorization_sha}:
        raise FormalRollingArchiveError(
            "staged remote eviction authorization digest differs"
        )
    if (
        result.archive_id != request.archive_id
        or result.remote_payload_root != request.remote_payload_root
        or result.local_final_root != request.local_final_root
        or result.manifest_sha256 != request.remote_manifest_sha256
        or authorization.archive_id != request.archive_id
        or authorization.remote_payload_root != request.remote_payload_root
        or authorization.local_final_root != request.local_final_root
        or authorization.manifest_sha256 != request.remote_manifest_sha256
        or authorization.authorized_at_ns != result.authorized_at_ns
        or authorization.rehydrated_content_tree_sha256
        != result.rehydrated_content_tree_sha256
    ):
        raise FormalRollingArchiveError("staged archive chain identities differ")
    result_output = _outside_candidate(
        result_output_path, candidate, "staged archive result"
    )
    authorization_output = _outside_candidate(
        authorization_output_path, candidate, "staged archive authorization"
    )
    lock = _outside_candidate(lock_path, candidate, "archive staging lock")
    with SingletonOperatorLock(lock):
        result_payload = {**asdict(result), "result_sha256": result.sha256}
        if os.path.lexists(result_output):
            if _read_canonical_object(result_output, "staged archive result") != (
                result_payload
            ):
                raise FormalRollingArchiveError("staged archive result is immutable")
        else:
            _publish_canonical_no_replace(result_output, result_payload)
        authorization_payload = {
            **asdict(authorization),
            "authorization_sha256": authorization_sha,
            "remote_deletion_performed": False,
        }
        if os.path.lexists(authorization_output):
            if (
                _read_canonical_object(
                    authorization_output, "staged archive authorization"
                )
                != authorization_payload
            ):
                raise FormalRollingArchiveError(
                    "staged archive authorization is immutable"
                )
        else:
            _publish_canonical_no_replace(authorization_output, authorization_payload)
    return StagedArchiveChain(
        request_path=str(request_source),
        result_path=str(result_output),
        authorization_path=str(authorization_output),
        request_sha256=_semantic_sha256(asdict(request)),
        result_sha256=result.sha256,
        authorization_sha256=authorization_sha,
    )


def run_staged_remote_eviction(
    *,
    request_path: str | Path,
    result_path: str | Path,
    authorization_path: str | Path,
    retained_dependency_manifest_path: str | Path,
    operator_database_path: str | Path,
    plan_output_path: str | Path,
    receipt_output_path: str | Path,
    plan_lock_path: str | Path,
    executor_lock_path: str | Path,
) -> tuple[RemoteEvictionPlan, RemoteEvictionReceipt]:
    """Resume remote plan publication and exact unlink under durable STOP."""

    request = load_archive_request(request_path)
    result_value = _read_canonical_object(result_path, "staged archive result")
    expected_result_sha = result_value.pop("result_sha256", None)
    result = RemoteArchiveResult(**result_value)
    if expected_result_sha != result.sha256:
        raise FormalRollingArchiveError("staged archive result digest differs")
    authorization_value = _read_canonical_object(
        authorization_path, "staged archive authorization"
    )
    authorization_value.pop("remote_deletion_performed", None)
    expected_authorization_sha = authorization_value.pop("authorization_sha256", None)
    authorization = RemoteEvictionAuthorization(**authorization_value)
    if expected_authorization_sha != remote_eviction_authorization_sha256(
        authorization
    ):
        raise FormalRollingArchiveError("staged archive authorization digest differs")
    database = _existing_file(operator_database_path, "operator database")
    candidate = Path(request.remote_payload_root)
    if database == candidate or database.is_relative_to(candidate):
        raise FormalRollingArchiveError("operator database is inside eviction scope")
    with ExperimentOperatorStore(database) as store:
        store.set_dispatch_stop("rolling_archive_eviction_boundary")
        if os.path.lexists(plan_output_path):
            plan = load_remote_eviction_plan(plan_output_path)
        else:
            plan = build_remote_eviction_plan(
                request=request,
                remote_archive_result=result,
                authorization=authorization,
                retained_dependency_manifest_path=(retained_dependency_manifest_path),
                operator_checkpoint=store.archive_checkpoint(request.archive_id),
                operator_snapshot=store.snapshot(),
            )
            publish_remote_eviction_plan(
                plan_output_path,
                plan,
                lock_path=plan_lock_path,
            )
        receipt = execute_remote_eviction_plan(
            plan_path=plan_output_path,
            receipt_path=receipt_output_path,
            lock_path=executor_lock_path,
            operator_snapshot=store.snapshot,
            scheduler_stop=store.set_dispatch_stop,
        )
    return plan, receipt


def evaluate_remote_eviction(
    **arguments: Any,
) -> RemoteEvictionGate:
    """Return an explicit false gate instead of leaking a partial plan."""

    try:
        plan = build_remote_eviction_plan(**arguments)
    except (OSError, TypeError, ValueError, FormalRollingArchiveError) as error:
        return RemoteEvictionGate(
            remote_eviction_authorized=False,
            blocker=f"{type(error).__name__}:{error}",
            plan=None,
        )
    return RemoteEvictionGate(True, None, plan)


def build_remote_eviction_plan(
    *,
    request: ArchiveRequest,
    remote_archive_result: RemoteArchiveResult,
    authorization: RemoteEvictionAuthorization,
    retained_dependency_manifest_path: str | Path,
    operator_checkpoint: Mapping[str, Any],
    operator_snapshot: Mapping[str, Any],
    active_writer_probe: Callable[[Path], bool] | None = None,
    process_probe: Callable[[int, int], bool] | None = None,
    created_at_ns: int | None = None,
) -> RemoteEvictionPlan:
    """Deep-replay every authority and return an exact path-bound plan."""

    if type(request) is not ArchiveRequest:
        raise TypeError("eviction planning requires an exact ArchiveRequest")
    if type(remote_archive_result) is not RemoteArchiveResult:
        raise TypeError("eviction planning requires an exact RemoteArchiveResult")
    if type(authorization) is not RemoteEvictionAuthorization:
        raise TypeError("eviction planning requires an exact authorization")
    checkpoint = _plain_mapping(operator_checkpoint, "operator archive checkpoint")
    snapshot = _plain_mapping(operator_snapshot, "operator snapshot")
    retained_path = _existing_file(
        retained_dependency_manifest_path, "retained dependency manifest"
    )
    retained = _load_retained_dependency_manifest(retained_path)
    run_root = _existing_directory(retained.run_root, "retained v03 run root")
    candidate = _existing_directory(request.remote_payload_root, "archive candidate")
    _validate_retained_scope(retained, run_root, candidate)
    _validate_v03_scope(run_root, candidate, retained.run_id)
    manifest_path = candidate / _ARCHIVE_MANIFEST_NAME
    manifest = load_formal_archive_sha256_manifest(manifest_path, verify_root=True)
    _validate_archive_chain(
        request=request,
        result=remote_archive_result,
        authorization=authorization,
        manifest=manifest,
        checkpoint=checkpoint,
        snapshot=snapshot,
        run_id=retained.run_id,
    )
    _validate_operator_quiescence(snapshot, process_probe=process_probe)
    writer_probe = active_writer_probe or _default_active_writer_probe
    retained_files = {
        _absolute_normalized(row.absolute_path, "retained file")
        for row in retained.retained_files
    }
    retained_roots = tuple(
        _absolute_normalized(path, "retained transitive root")
        for path in retained.retained_transitive_roots
    )
    planned = []
    for row in manifest.files:
        member = _safe_archive_member(candidate, row.path, must_exist=True)
        if member in retained_files or any(
            member == root or member.is_relative_to(root) for root in retained_roots
        ):
            continue
        _reject_forbidden_eviction_target(member, run_root=run_root)
        metadata, digest = _stable_file_identity(member)
        if metadata.st_nlink != 1:
            raise FormalRollingArchiveError(
                f"archive eviction rejects hard-linked file: {member}"
            )
        if metadata.st_size != row.size_bytes or digest != row.sha256:
            raise FormalRollingArchiveError(
                f"archive member changed before eviction planning: {member}"
            )
        writer_state = writer_probe(member)
        if type(writer_state) is not bool:
            raise TypeError("active writer probe must return bool")
        if writer_state:
            raise FormalRollingArchiveError(
                f"archive member has an active writer: {member}"
            )
        planned.append(
            EvictionFileBinding(
                absolute_path=str(member),
                archive_relative_path=row.path,
                device=metadata.st_dev,
                inode=metadata.st_ino,
                size_bytes=metadata.st_size,
                sha256=digest,
            )
        )
    if not planned:
        raise FormalRollingArchiveError("archive has no nonretained files to evict")
    candidate_metadata = candidate.lstat()
    if not stat.S_ISDIR(candidate_metadata.st_mode) or candidate.is_symlink():
        raise FormalRollingArchiveError(
            "archive candidate root changed during planning"
        )
    for binding in planned:
        _verify_eviction_file_binding(
            binding,
            candidate=candidate,
            candidate_device=candidate_metadata.st_dev,
            candidate_inode=candidate_metadata.st_ino,
            active_writer_probe=writer_probe,
        )
    authorization_sha = remote_eviction_authorization_sha256(authorization)
    return RemoteEvictionPlan(
        schema_version=1,
        kind=_EVICTION_PLAN_KIND,
        run_id=retained.run_id,
        run_root=str(run_root),
        node=retained.node,
        archive_id=request.archive_id,
        archive_candidate_root=str(candidate),
        archive_candidate_root_device=candidate_metadata.st_dev,
        archive_candidate_root_inode=candidate_metadata.st_ino,
        archive_manifest_path=str(manifest_path),
        archive_manifest_sha256=manifest.sha256,
        archive_request_sha256=_semantic_sha256(asdict(request)),
        remote_archive_result_sha256=remote_archive_result.sha256,
        archive_authorization_sha256=authorization_sha,
        retained_dependency_manifest_path=str(retained_path),
        retained_dependency_manifest_sha256=retained.sha256,
        operator_checkpoint_sha256=_semantic_sha256(checkpoint),
        operator_snapshot_sha256=_semantic_sha256(snapshot),
        files=tuple(sorted(planned, key=lambda row: row.absolute_path)),
        planned_bytes=sum(row.size_bytes for row in planned),
        created_at_ns=_positive_time(created_at_ns),
        remote_eviction_authorized=True,
    )


def publish_remote_eviction_plan(
    path: str | Path,
    plan: RemoteEvictionPlan,
    *,
    lock_path: str | Path,
    active_writer_probe: Callable[[Path], bool] | None = None,
) -> RemoteEvictionPlan:
    if type(plan) is not RemoteEvictionPlan:
        raise TypeError("remote eviction plan must use the exact plan type")
    candidate = Path(plan.archive_candidate_root)
    output = _outside_candidate(path, candidate, "plan")
    lock = _outside_candidate(lock_path, candidate, "eviction plan lock")
    with SingletonOperatorLock(lock):
        if os.path.lexists(output):
            existing = load_remote_eviction_plan(output)
            if existing != plan:
                raise FormalRollingArchiveError("remote eviction plan is immutable")
            return existing
        writer_probe = active_writer_probe or _default_active_writer_probe
        for binding in plan.files:
            _verify_eviction_file_binding(
                binding,
                candidate=candidate,
                candidate_device=plan.archive_candidate_root_device,
                candidate_inode=plan.archive_candidate_root_inode,
                active_writer_probe=writer_probe,
            )
        _publish_digest_envelope(output, plan.to_dict(), "plan_sha256", plan.sha256)
        return load_remote_eviction_plan(output)


def load_remote_eviction_plan(path: str | Path) -> RemoteEvictionPlan:
    value = _read_canonical_object(path, "remote eviction plan")
    expected = value.pop("plan_sha256", None)
    plan = RemoteEvictionPlan.from_dict(value)
    if expected != plan.sha256:
        raise FormalRollingArchiveError("remote eviction plan digest differs")
    return plan


def remote_eviction_authorization_sha256(
    authorization: RemoteEvictionAuthorization,
) -> str:
    if type(authorization) is not RemoteEvictionAuthorization:
        raise TypeError("authorization must use the exact operator type")
    _validate_authorization(authorization)
    return _semantic_sha256(asdict(authorization))


def execute_remote_eviction_plan(
    *,
    plan_path: str | Path,
    receipt_path: str | Path,
    lock_path: str | Path,
    operator_snapshot: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
    scheduler_stop: Callable[[str], None],
    active_writer_probe: Callable[[Path], bool] | None = None,
    process_probe: Callable[[int, int], bool] | None = None,
    after_unlink: Callable[[EvictionFileBinding], None] | None = None,
    clock_ns: Callable[[], int] = time.time_ns,
) -> RemoteEvictionReceipt:
    """Execute only a published plan, with durable file-by-file progress."""

    plan_source = _existing_file(plan_path, "published remote eviction plan")
    plan = load_remote_eviction_plan(plan_source)
    candidate = Path(plan.archive_candidate_root)
    _outside_candidate(plan_source, candidate, "plan")
    output = _outside_candidate(receipt_path, candidate, "eviction receipt")
    progress_root = output.with_name(f".{output.name}.progress")
    _outside_candidate(progress_root, candidate, "eviction progress")
    lock = _outside_candidate(lock_path, candidate, "eviction executor lock")
    if not callable(scheduler_stop):
        raise TypeError("scheduler STOP callback must be callable")
    writer_probe = active_writer_probe or _default_active_writer_probe
    with SingletonOperatorLock(lock):
        if os.path.lexists(output):
            receipt = load_remote_eviction_receipt(output)
            if receipt.plan_sha256 != plan.sha256:
                raise FormalRollingArchiveError(
                    "eviction receipt belongs to another plan"
                )
            return receipt
        snapshot = (
            operator_snapshot() if callable(operator_snapshot) else operator_snapshot
        )
        progress: tuple[EvictedFileReceipt, ...] = ()
        try:
            _validate_runtime_eviction_gate(
                plan,
                _plain_mapping(snapshot, "current operator snapshot"),
                process_probe=process_probe,
            )
            progress = _load_eviction_progress(progress_root, plan)
            _validate_progress_state(plan, progress)
        except (OSError, TypeError, ValueError, FormalRollingArchiveError) as error:
            missing = tuple(
                sorted(
                    binding.absolute_path
                    for binding in plan.files
                    if not os.path.lexists(binding.absolute_path)
                )
            )
            return _publish_eviction_failure(
                output=output,
                plan=plan,
                deleted=progress,
                missing_unrecorded=missing,
                failure_code=f"PREEXECUTION_{type(error).__name__.upper()}",
                failure_path=missing[0] if missing else None,
                scheduler_stop=scheduler_stop,
                clock_ns=clock_ns,
            )
        deleted_by_path = {row.absolute_path: row for row in progress}
        for binding in plan.files:
            prior = deleted_by_path.get(binding.absolute_path)
            if prior is not None:
                continue
            try:
                _verify_eviction_file_binding(
                    binding,
                    candidate=candidate,
                    candidate_device=plan.archive_candidate_root_device,
                    candidate_inode=plan.archive_candidate_root_inode,
                    active_writer_probe=writer_probe,
                )
                os.unlink(binding.absolute_path)
                _fsync_directory(Path(binding.absolute_path).parent)
                if after_unlink is not None:
                    after_unlink(binding)
                deleted = EvictedFileReceipt(
                    absolute_path=binding.absolute_path,
                    archive_relative_path=binding.archive_relative_path,
                    size_bytes=binding.size_bytes,
                    sha256=binding.sha256,
                    deleted_at_ns=_positive_time(clock_ns()),
                )
                _publish_eviction_progress(
                    progress_root=progress_root,
                    plan=plan,
                    index=plan.files.index(binding),
                    receipt=deleted,
                )
                deleted_by_path[binding.absolute_path] = deleted
            except Exception as error:  # noqa: BLE001 - destructive fail boundary
                missing = tuple(
                    sorted(
                        row.absolute_path
                        for row in plan.files
                        if (
                            row.absolute_path not in deleted_by_path
                            and not os.path.lexists(row.absolute_path)
                        )
                    )
                )
                return _publish_eviction_failure(
                    output=output,
                    plan=plan,
                    deleted=tuple(
                        sorted(
                            deleted_by_path.values(),
                            key=lambda row: row.absolute_path,
                        )
                    ),
                    missing_unrecorded=missing,
                    failure_code=f"EXECUTION_{type(error).__name__.upper()}",
                    failure_path=binding.absolute_path,
                    scheduler_stop=scheduler_stop,
                    clock_ns=clock_ns,
                )
        deleted = tuple(
            sorted(deleted_by_path.values(), key=lambda row: row.absolute_path)
        )
        if {row.absolute_path for row in deleted} != {
            row.absolute_path for row in plan.files
        }:
            return _publish_eviction_failure(
                output=output,
                plan=plan,
                deleted=deleted,
                missing_unrecorded=(),
                failure_code="INCOMPLETE_DURABLE_PROGRESS",
                failure_path=None,
                scheduler_stop=scheduler_stop,
                clock_ns=clock_ns,
            )
        receipt = RemoteEvictionReceipt(
            schema_version=1,
            kind=_EVICTION_RECEIPT_KIND,
            plan_sha256=plan.sha256,
            archive_id=plan.archive_id,
            archive_authorization_sha256=plan.archive_authorization_sha256,
            status="COMPLETE",
            deleted_files=deleted,
            deleted_bytes=sum(row.size_bytes for row in deleted),
            failure_code=None,
            failure_path=None,
            missing_unrecorded_files=(),
            scheduler_stop_requested=False,
            scheduler_stop_succeeded=False,
            finished_at_ns=_positive_time(clock_ns()),
        )
        _publish_digest_envelope(
            output, receipt.to_dict(), "receipt_sha256", receipt.sha256
        )
        return load_remote_eviction_receipt(output)


def load_remote_eviction_receipt(path: str | Path) -> RemoteEvictionReceipt:
    value = _read_canonical_object(path, "remote eviction receipt")
    expected = value.pop("receipt_sha256", None)
    receipt = RemoteEvictionReceipt.from_dict(value)
    if expected != receipt.sha256:
        raise FormalRollingArchiveError("remote eviction receipt digest differs")
    return receipt


def restore_evicted_files(
    *,
    plan_path: str | Path,
    remote_archive_result_path: str | Path,
    receipt_path: str | Path,
    lock_path: str | Path,
    clock_ns: Callable[[], int] = time.time_ns,
) -> ArchiveRestoreReceipt:
    """Restore only missing planned files from local final using link no-replace."""

    plan_source = _existing_file(plan_path, "published remote eviction plan")
    plan = load_remote_eviction_plan(plan_source)
    candidate = _existing_directory(
        plan.archive_candidate_root, "archive restore candidate root"
    )
    _verify_candidate_identity(
        candidate,
        device=plan.archive_candidate_root_device,
        inode=plan.archive_candidate_root_inode,
    )
    result_source = _existing_file(remote_archive_result_path, "remote archive result")
    result = load_remote_archive_result(result_source)
    if (
        result.sha256 != plan.remote_archive_result_sha256
        or result.archive_id != plan.archive_id
        or result.manifest_sha256 != plan.archive_manifest_sha256
        or result.remote_payload_root != plan.archive_candidate_root
        or result.remote_deletion_performed is not False
    ):
        raise FormalRollingArchiveError(
            "restore result differs from the published eviction plan"
        )
    final_root = _existing_directory(result.local_final_root, "local final archive")
    manifest_path = final_root / _ARCHIVE_MANIFEST_NAME
    manifest = load_formal_archive_sha256_manifest(manifest_path, verify_root=True)
    if manifest.sha256 != plan.archive_manifest_sha256:
        raise FormalRollingArchiveError("restore archive manifest digest differs")
    rows = {row.path: row for row in manifest.files}
    output = _outside_candidate(receipt_path, candidate, "restore receipt")
    if output.is_relative_to(final_root):
        raise ValueError("restore receipt must be outside the local final archive")
    lock = _outside_candidate(lock_path, candidate, "archive restore lock")
    with SingletonOperatorLock(lock):
        if os.path.lexists(output):
            receipt = load_archive_restore_receipt(output)
            if (
                receipt.plan_sha256 != plan.sha256
                or receipt.remote_archive_result_sha256 != result.sha256
            ):
                raise FormalRollingArchiveError(
                    "archive restore receipt belongs to another plan"
                )
            return receipt
        restored = []
        already_present = []
        for binding in plan.files:
            row = rows.get(binding.archive_relative_path)
            if (
                row is None
                or row.size_bytes != binding.size_bytes
                or row.sha256 != binding.sha256
            ):
                raise FormalRollingArchiveError(
                    "restore plan member differs from the local archive manifest"
                )
            source = _safe_archive_member(
                final_root, binding.archive_relative_path, must_exist=True
            )
            source_metadata, source_sha = _stable_file_identity(source)
            if (
                source_metadata.st_size != binding.size_bytes
                or source_sha != binding.sha256
            ):
                raise FormalRollingArchiveError("local restore source identity differs")
            target = Path(binding.absolute_path)
            _validate_restore_parent(target, candidate=candidate)
            restored_row = RestoredFileReceipt(
                absolute_path=binding.absolute_path,
                archive_relative_path=binding.archive_relative_path,
                size_bytes=binding.size_bytes,
                sha256=binding.sha256,
            )
            if os.path.lexists(target):
                metadata, digest = _stable_file_identity(target)
                if (
                    metadata.st_nlink != 1
                    or metadata.st_size != binding.size_bytes
                    or digest != binding.sha256
                ):
                    raise FormalRollingArchiveError(
                        f"restore refuses to overwrite existing path: {target}"
                    )
                already_present.append(restored_row)
                continue
            _restore_one_no_replace(
                source=source,
                target=target,
                expected=binding,
                candidate=candidate,
            )
            restored.append(restored_row)
        receipt = ArchiveRestoreReceipt(
            schema_version=1,
            kind=_RESTORE_RECEIPT_KIND,
            plan_sha256=plan.sha256,
            archive_id=plan.archive_id,
            archive_manifest_sha256=plan.archive_manifest_sha256,
            remote_archive_result_sha256=result.sha256,
            restored_files=tuple(sorted(restored, key=lambda row: row.absolute_path)),
            already_present_files=tuple(
                sorted(already_present, key=lambda row: row.absolute_path)
            ),
            restored_bytes=sum(row.size_bytes for row in restored),
            completed_at_ns=_positive_time(clock_ns()),
            existing_files_overwritten=False,
        )
        if {
            row.absolute_path
            for row in (*receipt.restored_files, *receipt.already_present_files)
        } != {row.absolute_path for row in plan.files}:
            raise AssertionError("restore receipt does not cover the exact plan")
        _publish_digest_envelope(
            output, receipt.to_dict(), "receipt_sha256", receipt.sha256
        )
        return load_archive_restore_receipt(output)


def load_archive_restore_receipt(path: str | Path) -> ArchiveRestoreReceipt:
    value = _read_canonical_object(path, "archive restore receipt")
    expected = value.pop("receipt_sha256", None)
    receipt = ArchiveRestoreReceipt.from_dict(value)
    if expected != receipt.sha256:
        raise FormalRollingArchiveError("archive restore receipt digest differs")
    return receipt


def restore_remote_member_from_stream(
    *,
    plan_path: str | Path,
    eviction_receipt_path: str | Path,
    remote_archive_result_path: str | Path,
    archive_relative_path: str,
    progress_output_path: str | Path,
    lock_path: str | Path,
    stream: BinaryIO,
    minimum_free_bytes: int = REMOTE_SPOOL_SAFETY_RESERVE_BYTES,
    free_bytes_probe: Callable[[Path], int] | None = None,
    after_target_link: Callable[[EvictionFileBinding], None] | None = None,
    clock_ns: Callable[[], int] = time.time_ns,
) -> RemoteStreamRestoreProgress:
    """Stream one planned member into a same-directory no-replace target."""

    plan, eviction, result = _load_stream_restore_authorities(
        plan_path=plan_path,
        eviction_receipt_path=eviction_receipt_path,
        remote_archive_result_path=remote_archive_result_path,
    )
    matches = [
        (index, binding)
        for index, binding in enumerate(plan.files)
        if binding.archive_relative_path == archive_relative_path
    ]
    if len(matches) != 1:
        raise FormalRollingArchiveError(
            "stream restore path is not one exact evicted plan member"
        )
    index, binding = matches[0]
    candidate = _existing_directory(
        plan.archive_candidate_root, "stream restore candidate root"
    )
    _verify_candidate_identity(
        candidate,
        device=plan.archive_candidate_root_device,
        inode=plan.archive_candidate_root_inode,
    )
    target = Path(binding.absolute_path)
    _validate_restore_parent(target, candidate=candidate)
    output = _outside_candidate(
        progress_output_path, candidate, "stream restore progress"
    )
    lock = _outside_candidate(lock_path, candidate, "stream restore lock")
    _require_nonnegative_int(minimum_free_bytes, "stream restore free-byte reserve")
    if not hasattr(stream, "read"):
        raise TypeError("stream restore input must be a binary reader")
    expected_row = RestoredFileReceipt(
        absolute_path=binding.absolute_path,
        archive_relative_path=binding.archive_relative_path,
        size_bytes=binding.size_bytes,
        sha256=binding.sha256,
    )
    with SingletonOperatorLock(lock):
        existing_progress: RemoteStreamRestoreProgress | None = None
        if os.path.lexists(output):
            existing_progress = load_remote_stream_restore_progress(output)
            if (
                existing_progress.plan_sha256 != plan.sha256
                or existing_progress.eviction_receipt_sha256 != eviction.sha256
                or existing_progress.remote_archive_result_sha256 != result.sha256
                or existing_progress.file_index != index
                or existing_progress.file != expected_row
            ):
                raise FormalRollingArchiveError(
                    "stream restore progress belongs to another member"
                )
        free_probe = free_bytes_probe or _filesystem_free_bytes
        for filesystem_path, remaining_bytes in _stream_restore_high_waters(
            plan,
            candidate=candidate,
        ):
            free_bytes = free_probe(filesystem_path)
            _require_nonnegative_int(free_bytes, "stream restore free bytes")
            required = remaining_bytes + minimum_free_bytes
            if free_bytes < required:
                raise FormalRollingArchiveError(
                    "remote restore capacity is below remaining high-water plus "
                    "safety reserve"
                )
        target_exists = os.path.lexists(target)
        if target_exists:
            _verify_restored_target(target, binding)
            streamed_size, streamed_sha = _consume_restore_stream(
                stream, descriptor=None, expected_size=binding.size_bytes
            )
            disposition: Literal["RESTORED", "ALREADY_PRESENT"] = "ALREADY_PRESENT"
        else:
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.rehydrate")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o600)
            linked = False
            try:
                streamed_size, streamed_sha = _consume_restore_stream(
                    stream,
                    descriptor=descriptor,
                    expected_size=binding.size_bytes,
                )
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = -1
                temporary_metadata, temporary_sha = _stable_file_identity(temporary)
                if (
                    temporary_metadata.st_nlink != 1
                    or temporary_metadata.st_size != binding.size_bytes
                    or temporary_sha != binding.sha256
                ):
                    raise FormalRollingArchiveError(
                        "stream restore temporary identity differs"
                    )
                if os.path.lexists(target):
                    raise FormalRollingArchiveError(
                        "stream restore target became occupied"
                    )
                os.link(temporary, target, follow_symlinks=False)
                linked = True
                _fsync_directory(target.parent)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                _fsync_directory(target.parent)
            if not linked:
                raise FormalRollingArchiveError(
                    "stream restore target was not atomically published"
                )
            _verify_restored_target(target, binding)
            if after_target_link is not None:
                after_target_link(binding)
            disposition = "RESTORED"
        if streamed_size != binding.size_bytes or streamed_sha != binding.sha256:
            raise FormalRollingArchiveError(
                "streamed local archive member identity differs"
            )
        if existing_progress is not None:
            _verify_restored_target(target, binding)
            return existing_progress
        progress = RemoteStreamRestoreProgress(
            schema_version=1,
            kind=_STREAM_RESTORE_PROGRESS_KIND,
            plan_sha256=plan.sha256,
            eviction_receipt_sha256=eviction.sha256,
            remote_archive_result_sha256=result.sha256,
            file_index=index,
            file=expected_row,
            disposition=disposition,
            restored_at_ns=_positive_time(clock_ns()),
            existing_file_overwritten=False,
        )
        _publish_digest_envelope(
            output, progress.to_dict(), "progress_sha256", progress.sha256
        )
        return load_remote_stream_restore_progress(output)


def load_remote_stream_restore_progress(
    path: str | Path,
) -> RemoteStreamRestoreProgress:
    value = _read_canonical_object(path, "remote stream restore progress")
    expected = value.pop("progress_sha256", None)
    progress = RemoteStreamRestoreProgress.from_dict(value)
    if expected != progress.sha256:
        raise FormalRollingArchiveError("remote stream restore progress digest differs")
    return progress


def finalize_remote_stream_restore(
    *,
    plan_path: str | Path,
    eviction_receipt_path: str | Path,
    remote_archive_result_path: str | Path,
    progress_root: str | Path,
    receipt_output_path: str | Path,
    lock_path: str | Path,
    clock_ns: Callable[[], int] = time.time_ns,
) -> ArchiveRestoreReceipt:
    """Publish one restore receipt after exact progress and target replay."""

    plan, eviction, result = _load_stream_restore_authorities(
        plan_path=plan_path,
        eviction_receipt_path=eviction_receipt_path,
        remote_archive_result_path=remote_archive_result_path,
    )
    candidate = _existing_directory(
        plan.archive_candidate_root, "stream restore candidate root"
    )
    progress_directory = _absolute_normalized(
        progress_root, "stream restore progress root"
    )
    if progress_directory == candidate or progress_directory.is_relative_to(candidate):
        raise ValueError("stream restore progress root is inside candidate")
    output = _outside_candidate(
        receipt_output_path, candidate, "stream restore receipt"
    )
    lock = _outside_candidate(lock_path, candidate, "stream restore finalize lock")
    with SingletonOperatorLock(lock):
        if os.path.lexists(output):
            receipt = load_archive_restore_receipt(output)
            _verify_stream_restore_receipt(receipt, plan=plan, result=result)
            return receipt
        restored = []
        already_present = []
        for index, binding in enumerate(plan.files):
            progress_path = progress_directory / f"{index:08d}.json"
            progress = load_remote_stream_restore_progress(progress_path)
            expected = RestoredFileReceipt(
                binding.absolute_path,
                binding.archive_relative_path,
                binding.size_bytes,
                binding.sha256,
            )
            if (
                progress.plan_sha256 != plan.sha256
                or progress.eviction_receipt_sha256 != eviction.sha256
                or progress.remote_archive_result_sha256 != result.sha256
                or progress.file_index != index
                or progress.file != expected
            ):
                raise FormalRollingArchiveError(
                    "stream restore progress coverage differs"
                )
            _verify_restored_target(Path(binding.absolute_path), binding)
            if progress.disposition == "RESTORED":
                restored.append(expected)
            else:
                already_present.append(expected)
        receipt = ArchiveRestoreReceipt(
            schema_version=1,
            kind=_RESTORE_RECEIPT_KIND,
            plan_sha256=plan.sha256,
            archive_id=plan.archive_id,
            archive_manifest_sha256=plan.archive_manifest_sha256,
            remote_archive_result_sha256=result.sha256,
            restored_files=tuple(restored),
            already_present_files=tuple(already_present),
            restored_bytes=sum(row.size_bytes for row in restored),
            completed_at_ns=_positive_time(clock_ns()),
            existing_files_overwritten=False,
        )
        _publish_digest_envelope(
            output, receipt.to_dict(), "receipt_sha256", receipt.sha256
        )
        published = load_archive_restore_receipt(output)
        _verify_stream_restore_receipt(published, plan=plan, result=result)
        return published


def _load_stream_restore_authorities(
    *,
    plan_path: str | Path,
    eviction_receipt_path: str | Path,
    remote_archive_result_path: str | Path,
) -> tuple[RemoteEvictionPlan, RemoteEvictionReceipt, RemoteArchiveResult]:
    plan = load_remote_eviction_plan(plan_path)
    eviction = load_remote_eviction_receipt(eviction_receipt_path)
    result_value = _read_canonical_object(
        remote_archive_result_path, "staged remote archive result"
    )
    expected_result_sha = result_value.pop("result_sha256", None)
    result = RemoteArchiveResult(**result_value)
    planned = {
        (
            row.absolute_path,
            row.archive_relative_path,
            row.size_bytes,
            row.sha256,
        )
        for row in plan.files
    }
    deleted = {
        (
            row.absolute_path,
            row.archive_relative_path,
            row.size_bytes,
            row.sha256,
        )
        for row in eviction.deleted_files
    }
    if (
        expected_result_sha != result.sha256
        or result.sha256 != plan.remote_archive_result_sha256
        or result.archive_id != plan.archive_id
        or result.remote_payload_root != plan.archive_candidate_root
        or result.manifest_sha256 != plan.archive_manifest_sha256
        or eviction.status != "COMPLETE"
        or eviction.plan_sha256 != plan.sha256
        or eviction.archive_id != plan.archive_id
        or eviction.archive_authorization_sha256 != plan.archive_authorization_sha256
        or planned != deleted
    ):
        raise FormalRollingArchiveError(
            "stream restore plan/result/eviction receipt identities differ"
        )
    return plan, eviction, result


def _consume_restore_stream(
    stream: BinaryIO,
    *,
    descriptor: int | None,
    expected_size: int,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while total < expected_size:
        chunk = stream.read(min(1024 * 1024, expected_size - total))
        if not isinstance(chunk, bytes):
            raise TypeError("stream restore reader must return bytes")
        if not chunk:
            break
        digest.update(chunk)
        if descriptor is not None:
            offset = 0
            while offset < len(chunk):
                offset += os.write(descriptor, chunk[offset:])
        total += len(chunk)
    extra = stream.read(1)
    if not isinstance(extra, bytes):
        raise TypeError("stream restore reader must return bytes")
    if total != expected_size or extra:
        raise FormalRollingArchiveError("stream restore byte count differs")
    return total, digest.hexdigest()


def _verify_restored_target(
    target: Path,
    binding: EvictionFileBinding,
) -> None:
    metadata, digest = _stable_file_identity(target)
    if (
        metadata.st_nlink != 1
        or metadata.st_size != binding.size_bytes
        or digest != binding.sha256
    ):
        raise FormalRollingArchiveError(f"restored target identity differs: {target}")


def _stream_restore_high_waters(
    plan: RemoteEvictionPlan,
    *,
    candidate: Path,
) -> tuple[tuple[Path, int], ...]:
    remaining: dict[int, tuple[Path, int]] = {}
    for binding in plan.files:
        target = Path(binding.absolute_path)
        _validate_restore_parent(target, candidate=candidate)
        if os.path.lexists(target):
            _verify_restored_target(target, binding)
        else:
            parent_device = target.parent.stat().st_dev
            representative, size = remaining.get(
                parent_device,
                (target.parent, 0),
            )
            remaining[parent_device] = (representative, size + binding.size_bytes)
    return tuple(remaining[key] for key in sorted(remaining))


def _verify_stream_restore_receipt(
    receipt: ArchiveRestoreReceipt,
    *,
    plan: RemoteEvictionPlan,
    result: RemoteArchiveResult,
) -> None:
    expected = {
        (
            row.absolute_path,
            row.archive_relative_path,
            row.size_bytes,
            row.sha256,
        )
        for row in plan.files
    }
    observed = {
        (
            row.absolute_path,
            row.archive_relative_path,
            row.size_bytes,
            row.sha256,
        )
        for row in (*receipt.restored_files, *receipt.already_present_files)
    }
    if (
        receipt.plan_sha256 != plan.sha256
        or receipt.archive_id != plan.archive_id
        or receipt.archive_manifest_sha256 != plan.archive_manifest_sha256
        or receipt.remote_archive_result_sha256 != result.sha256
        or observed != expected
    ):
        raise FormalRollingArchiveError(
            "stream restore receipt belongs to another exact restore scope"
        )
    for binding in plan.files:
        _verify_restored_target(Path(binding.absolute_path), binding)


def _filesystem_free_bytes(path: Path) -> int:
    values = os.statvfs(path)
    return values.f_bavail * values.f_frsize


def _load_retained_dependency_manifest(path: Path) -> Any:
    # Lazy by design: the DAG driver imports many concrete stage modules and may
    # itself consume this rolling boundary. Importing it at module load would
    # create a fragile orchestration cycle.
    from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
        RetainedFutureDependencyManifest,
        load_retained_future_dependency_manifest,
    )

    retained = load_retained_future_dependency_manifest(path)
    if type(retained) is not RetainedFutureDependencyManifest:
        raise TypeError("retained dependency loader returned another type")
    return retained


def _validate_retained_scope(retained: Any, run_root: Path, candidate: Path) -> None:
    if (
        Path(retained.run_root) != run_root
        or Path(retained.run_root).name != retained.run_id
        or candidate not in {Path(path) for path in retained.archive_candidate_roots}
        or retained.archive_safe_after_reduction is not True
        or retained.remote_eviction_authorized_for_nonretained_files is not True
        or retained.remote_eviction_scope
        != "archive_candidate_roots_excluding_retained_files_and_transitive_roots"
        or retained.eviction_preconditions
        != (
            "local_sha_manifest_verified",
            "local_rehydrate_test_passed",
        )
        or retained.transitive_evidence_must_rehydrate_at_original_paths is not True
    ):
        raise FormalRollingArchiveError(
            "retained dependency manifest does not seal this archive candidate"
        )
    for label, binding in (
        ("retained completion", retained.completion),
        ("retained decision", retained.decision),
    ):
        source = _existing_file(binding.absolute_path, label)
        if _stable_file_identity(source)[1] != binding.sha256:
            raise FormalRollingArchiveError(f"{label} content changed")
    for path in retained.retained_transitive_roots:
        retained_root = _existing_directory(path, "retained transitive root")
        if not retained_root.is_relative_to(run_root):
            raise FormalRollingArchiveError(
                "retained transitive root belongs to another run"
            )


def _validate_v03_scope(run_root: Path, candidate: Path, run_id: str) -> None:
    run_root = _existing_directory(run_root, "v03 run root")
    candidate = _existing_directory(candidate, "archive candidate root")
    if not _safe_component(run_id) or run_root.name != run_id:
        raise FormalRollingArchiveError("run root is not bound to its exact run ID")
    home = Path.home().resolve(strict=False)
    repository = Path(__file__).resolve().parents[3]
    if (
        run_root == Path("/")
        or run_root == home
        or _paths_overlap(run_root, repository)
        or candidate == run_root
        or not candidate.is_relative_to(run_root)
    ):
        raise FormalRollingArchiveError("archive scope is a broad or foreign path")
    parts = tuple(part.casefold() for part in run_root.parts)
    candidate_parts = tuple(part.casefold() for part in candidate.parts)
    if (
        not any(_VERSION_TOKEN_V03.search(part) for part in parts)
        or any(_VERSION_TOKEN_V02.search(part) for part in candidate_parts)
        or any(part in _FORBIDDEN_TREE_PARTS for part in candidate_parts)
    ):
        raise FormalRollingArchiveError(
            "archive scope is not an exact non-cache v03 run path"
        )


def _scan_archive_tree(
    root: Path,
    *,
    allow_root_manifest: bool,
) -> tuple[ArchiveManifestFile, ...]:
    root = _existing_directory(root, "archive candidate root")
    rows: list[ArchiveManifestFile] = []

    def walk(directory: Path) -> None:
        before = directory.lstat()
        if not stat.S_ISDIR(before.st_mode) or directory.is_symlink():
            raise FormalRollingArchiveError(
                f"archive member directory is unsafe: {directory}"
            )
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise FormalRollingArchiveError(
                f"cannot enumerate archive directory: {directory}"
            ) from error
        for entry in entries:
            path = directory / entry.name
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise FormalRollingArchiveError(
                    f"archive tree contains a symbolic link: {path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                walk(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise FormalRollingArchiveError(
                    f"archive tree contains a FIFO/socket/device: {path}"
                )
            if path.name == _ARCHIVE_MANIFEST_NAME:
                if allow_root_manifest and path == root / _ARCHIVE_MANIFEST_NAME:
                    continue
                raise FormalRollingArchiveError(
                    f"archive manifest cannot include itself or a nested copy: {path}"
                )
            relative = path.relative_to(root).as_posix()
            stable, digest = _stable_file_identity(path)
            if (
                stable.st_dev != metadata.st_dev
                or stable.st_ino != metadata.st_ino
                or stable.st_size != metadata.st_size
            ):
                raise FormalRollingArchiveError(
                    f"archive member changed during enumeration: {path}"
                )
            rows.append(ArchiveManifestFile(relative, digest, stable.st_size))
        after = directory.lstat()
        if _stat_identity(before, include_link_count=False) != _stat_identity(
            after, include_link_count=False
        ):
            raise FormalRollingArchiveError(
                f"archive directory changed during enumeration: {directory}"
            )

    walk(root)
    rows.sort(key=lambda row: row.path)
    paths = [row.path for row in rows]
    if not rows or len(paths) != len(set(paths)):
        raise FormalRollingArchiveError(
            "archive tree is empty or has duplicate relative paths"
        )
    return tuple(rows)


def _verify_manifest_tree(
    root: Path,
    manifest: FormalArchiveSha256Manifest,
    *,
    require_manifest: bool,
) -> None:
    if require_manifest:
        manifest_path = _existing_file(
            root / _ARCHIVE_MANIFEST_NAME, "archive manifest"
        )
        if _stable_file_identity(manifest_path)[1] != manifest.sha256:
            raise FormalRollingArchiveError("archive manifest bytes differ")
    actual = _scan_archive_tree(root, allow_root_manifest=require_manifest)
    if actual != manifest.files:
        raise FormalRollingArchiveError(
            "archive tree has unregistered, missing, duplicate, or changed files"
        )


def _safe_archive_member(
    root: Path,
    relative: str,
    *,
    must_exist: bool,
) -> Path:
    pure = PurePosixPath(relative)
    if (
        type(relative) is not str
        or not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or str(pure) != relative
        or pure.name == _ARCHIVE_MANIFEST_NAME
    ):
        raise FormalRollingArchiveError("archive member path escapes its root")
    member = root.joinpath(*pure.parts)
    if not member.is_relative_to(root):
        raise FormalRollingArchiveError("archive member path escapes its root")
    if must_exist:
        member = _existing_file(member, "archive member")
        if member.resolve(strict=True) != member:
            raise FormalRollingArchiveError("archive member traverses a symlink")
    return member


def _validate_archive_chain(
    *,
    request: ArchiveRequest,
    result: RemoteArchiveResult,
    authorization: RemoteEvictionAuthorization,
    manifest: FormalArchiveSha256Manifest,
    checkpoint: dict[str, Any],
    snapshot: dict[str, Any],
    run_id: str,
) -> None:
    _validate_authorization(authorization)
    if (
        request.remote_manifest_sha256 != manifest.sha256
        or request.predicted_payload_bytes != manifest.payload_bytes
        or result.schema_version != 1
        or result.kind != "formal_remote_archive_result"
        or result.archive_id != request.archive_id
        or result.remote_payload_root != request.remote_payload_root
        or result.local_final_root != request.local_final_root
        or result.manifest_sha256 != manifest.sha256
        or result.checked_file_count != len(manifest.files)
        or result.checked_bytes != manifest.payload_bytes
        or result.remote_deletion_performed is not False
        or authorization.archive_id != request.archive_id
        or authorization.remote_payload_root != request.remote_payload_root
        or authorization.manifest_sha256 != manifest.sha256
        or authorization.local_final_root != request.local_final_root
        or authorization.rehydrated_content_tree_sha256
        != result.rehydrated_content_tree_sha256
        or authorization.authorized_at_ns != result.authorized_at_ns
    ):
        raise FormalRollingArchiveError(
            "archive request/result/authorization identities differ"
        )
    expected_checkpoint = {
        "archive_id": request.archive_id,
        "safe_boundary": request.safe_boundary,
        "remote_payload_root": request.remote_payload_root,
        "local_partial_root": request.local_partial_root,
        "local_final_root": request.local_final_root,
        "remote_manifest_sha256": manifest.sha256,
        "predicted_payload_bytes": manifest.payload_bytes,
        "state": "EVICTION_AUTHORIZED",
        "eviction_authorized_at_ns": authorization.authorized_at_ns,
    }
    if any(checkpoint.get(key) != value for key, value in expected_checkpoint.items()):
        raise FormalRollingArchiveError(
            "durable archive checkpoint is not EVICTION_AUTHORIZED"
        )
    receipts = []
    for field, expected_step in (
        ("transfer_receipt", "TRANSFER"),
        ("local_sha_receipt", "LOCAL_SHA_VERIFY"),
        ("rehydrate_receipt", "REHYDRATE_VERIFY"),
    ):
        raw = checkpoint.get(field)
        if type(raw) is not dict:
            raise FormalRollingArchiveError(f"archive checkpoint lacks {field}")
        receipt = ArchiveStepReceipt(**raw)
        if (
            receipt.step != expected_step
            or receipt.manifest_sha256 != manifest.sha256
            or receipt.checked_file_count != len(manifest.files)
            or receipt.checked_bytes != manifest.payload_bytes
        ):
            raise FormalRollingArchiveError(f"archive checkpoint {field} differs")
        receipts.append(receipt)
    local_receipt, rehydrate_receipt = receipts[1], receipts[2]
    if (
        local_receipt.evidence_sha256 != authorization.local_sha_evidence_sha256
        or rehydrate_receipt.evidence_sha256 != authorization.rehydrate_evidence_sha256
        or rehydrate_receipt.content_tree_sha256
        != authorization.rehydrated_content_tree_sha256
    ):
        raise FormalRollingArchiveError(
            "archive verification receipts differ from authorization"
        )
    if snapshot.get("run_id") != run_id:
        raise FormalRollingArchiveError("operator snapshot belongs to another run")
    raw_archives = snapshot.get("archives")
    if type(raw_archives) is not list:
        raise FormalRollingArchiveError("operator snapshot archive ledger is absent")
    matches = [
        row
        for row in raw_archives
        if type(row) is dict and row.get("archive_id") == request.archive_id
    ]
    if len(matches) != 1 or any(
        matches[0].get(key) != value for key, value in expected_checkpoint.items()
    ):
        raise FormalRollingArchiveError(
            "operator snapshot does not replay the authorized checkpoint"
        )


def _validate_authorization(authorization: RemoteEvictionAuthorization) -> None:
    for label, value in (
        ("authorization archive ID", authorization.archive_id),
        ("authorization remote root", authorization.remote_payload_root),
        ("authorization local root", authorization.local_final_root),
    ):
        if type(value) is not str or not value or "\x00" in value:
            raise ValueError(f"{label} is invalid")
    _absolute_normalized(
        authorization.remote_payload_root, "authorization remote payload root"
    )
    _absolute_normalized(authorization.local_final_root, "authorization local root")
    for label, value in (
        ("manifest", authorization.manifest_sha256),
        ("local SHA evidence", authorization.local_sha_evidence_sha256),
        ("rehydrate evidence", authorization.rehydrate_evidence_sha256),
        ("rehydrated content tree", authorization.rehydrated_content_tree_sha256),
    ):
        _require_sha256(value, f"authorization {label} SHA-256")
    _require_positive_int(authorization.authorized_at_ns, "authorization time")


def _validate_operator_quiescence(
    snapshot: Mapping[str, Any],
    *,
    process_probe: Callable[[int, int], bool] | None,
) -> None:
    if snapshot.get("dispatch_state") != "STOP":
        raise FormalRollingArchiveError(
            "scheduler dispatch must be durably STOP before remote eviction"
        )
    attempts = snapshot.get("attempts")
    if type(attempts) is not list:
        raise FormalRollingArchiveError("operator snapshot attempts are absent")
    probe = process_probe or _process_is_alive_in_group
    for raw in attempts:
        if type(raw) is not dict:
            raise FormalRollingArchiveError("operator attempt row is not an object")
        if raw.get("status") == "RUNNING":
            raise FormalRollingArchiveError("a RUNNING attempt blocks remote eviction")
        pid = raw.get("pid")
        pgid = raw.get("pgid")
        if (pid is None) != (pgid is None):
            raise FormalRollingArchiveError("attempt PID/PGID binding is incomplete")
        if pid is not None:
            _require_positive_int(pid, "attempt PID")
            _require_positive_int(pgid, "attempt PGID")
            active = probe(pid, pgid)
            if type(active) is not bool:
                raise TypeError("process probe must return bool")
            if active:
                raise FormalRollingArchiveError(
                    "a live attempt PID/PGID blocks remote eviction"
                )


def _validate_runtime_eviction_gate(
    plan: RemoteEvictionPlan,
    snapshot: Mapping[str, Any],
    *,
    process_probe: Callable[[int, int], bool] | None,
) -> None:
    if snapshot.get("run_id") != plan.run_id:
        raise FormalRollingArchiveError("current operator snapshot is a foreign run")
    _validate_operator_quiescence(snapshot, process_probe=process_probe)
    archives = snapshot.get("archives")
    if type(archives) is not list:
        raise FormalRollingArchiveError("current archive ledger is absent")
    matches = [
        row
        for row in archives
        if type(row) is dict and row.get("archive_id") == plan.archive_id
    ]
    if (
        len(matches) != 1
        or matches[0].get("state") != "EVICTION_AUTHORIZED"
        or matches[0].get("remote_payload_root") != plan.archive_candidate_root
        or matches[0].get("remote_manifest_sha256") != plan.archive_manifest_sha256
    ):
        raise FormalRollingArchiveError(
            "current archive ledger lost durable eviction authorization"
        )
    _verify_candidate_identity(
        Path(plan.archive_candidate_root),
        device=plan.archive_candidate_root_device,
        inode=plan.archive_candidate_root_inode,
    )


def _verify_candidate_identity(candidate: Path, *, device: int, inode: int) -> None:
    candidate = _existing_directory(candidate, "archive candidate root")
    metadata = candidate.lstat()
    if (
        metadata.st_dev != device
        or metadata.st_ino != inode
        or candidate.resolve(strict=True) != candidate
    ):
        raise FormalRollingArchiveError("archive candidate root identity changed")


def _verify_eviction_file_binding(
    binding: EvictionFileBinding,
    *,
    candidate: Path,
    candidate_device: int,
    candidate_inode: int,
    active_writer_probe: Callable[[Path], bool],
) -> None:
    _verify_candidate_identity(
        candidate,
        device=candidate_device,
        inode=candidate_inode,
    )
    path = _safe_archive_member(
        candidate, binding.archive_relative_path, must_exist=True
    )
    if path != Path(binding.absolute_path):
        raise FormalRollingArchiveError("planned file path binding changed")
    metadata, digest = _stable_file_identity(path)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_dev != binding.device
        or metadata.st_ino != binding.inode
        or metadata.st_size != binding.size_bytes
        or digest != binding.sha256
    ):
        raise FormalRollingArchiveError(
            f"planned file identity changed before unlink: {path}"
        )
    writer_state = active_writer_probe(path)
    if type(writer_state) is not bool:
        raise TypeError("active writer probe must return bool")
    if writer_state:
        raise FormalRollingArchiveError(f"planned file has an active writer: {path}")
    final = path.lstat()
    if (
        final.st_dev != binding.device
        or final.st_ino != binding.inode
        or final.st_size != binding.size_bytes
        or final.st_nlink != 1
        or not stat.S_ISREG(final.st_mode)
    ):
        raise FormalRollingArchiveError(
            f"planned file mutated at the unlink boundary: {path}"
        )


def _reject_forbidden_eviction_target(path: Path, *, run_root: Path) -> None:
    if path == run_root or not path.is_relative_to(run_root):
        raise FormalRollingArchiveError("eviction target is outside the exact run")
    relative_parts = tuple(part.casefold() for part in path.relative_to(run_root).parts)
    name = path.name.casefold()
    if (
        any(_VERSION_TOKEN_V02.search(part) for part in path.parts)
        or any(part in _FORBIDDEN_TREE_PARTS for part in relative_parts)
        or name in _OPERATOR_DATABASE_NAMES
        or (
            name.endswith(("-wal", "-shm"))
            and ("operator" in name or ".sqlite" in name or ".db" in name)
        )
        or name.endswith(".lock")
    ):
        raise FormalRollingArchiveError(
            f"eviction target is a v02/cache/model/operator artifact: {path}"
        )


def _default_active_writer_probe(path: Path) -> bool:
    """Best-effort OS writer gate plus a cooperative exclusive-lock gate."""

    metadata = path.lstat()
    proc = Path("/proc")
    if proc.is_dir():
        for process in proc.iterdir():
            if not process.name.isdigit() or int(process.name) == os.getpid():
                continue
            descriptors = process / "fd"
            try:
                entries = tuple(descriptors.iterdir())
            except FileNotFoundError:
                continue
            except (PermissionError, OSError):
                return True
            for descriptor_path in entries:
                try:
                    descriptor_metadata = descriptor_path.stat()
                except FileNotFoundError:
                    continue
                except (PermissionError, OSError):
                    return True
                if (
                    descriptor_metadata.st_dev != metadata.st_dev
                    or descriptor_metadata.st_ino != metadata.st_ino
                ):
                    continue
                try:
                    flags_line = next(
                        line
                        for line in (process / "fdinfo" / descriptor_path.name)
                        .read_text(encoding="ascii")
                        .splitlines()
                        if line.startswith("flags:")
                    )
                    flags = int(flags_line.split()[1], 8)
                except (
                    FileNotFoundError,
                    PermissionError,
                    OSError,
                    StopIteration,
                    ValueError,
                ):
                    return True
                if flags & os.O_ACCMODE in {os.O_WRONLY, os.O_RDWR}:
                    return True
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(descriptor)
    return False


def _process_is_alive_in_group(pid: int, pgid: int) -> bool:
    try:
        os.kill(pid, 0)
        return os.getpgid(pid) == pgid
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True


def _load_eviction_progress(
    progress_root: Path,
    plan: RemoteEvictionPlan,
) -> tuple[EvictedFileReceipt, ...]:
    if not os.path.lexists(progress_root):
        return ()
    if progress_root.is_symlink() or not progress_root.is_dir():
        raise FormalRollingArchiveError("eviction progress path is unsafe")
    known_names = {f"{index:08d}.json" for index, _binding in enumerate(plan.files)}
    actual_names = {path.name for path in progress_root.iterdir()}
    if not actual_names <= known_names:
        raise FormalRollingArchiveError("eviction progress has unknown entries")
    receipts = []
    for index, binding in enumerate(plan.files):
        path = progress_root / f"{index:08d}.json"
        if not os.path.lexists(path):
            continue
        value = _read_canonical_object(path, "eviction file progress")
        expected = value.pop("progress_sha256", None)
        if set(value) != {
            "schema_version",
            "kind",
            "plan_sha256",
            "index",
            "file",
        }:
            raise FormalRollingArchiveError("eviction progress fields differ")
        progress_sha = _semantic_sha256(value)
        raw_file = value["file"]
        receipt = EvictedFileReceipt.from_dict(raw_file)
        if (
            expected != progress_sha
            or value["schema_version"] != 1
            or value["kind"] != _EVICTION_PROGRESS_KIND
            or value["plan_sha256"] != plan.sha256
            or value["index"] != index
            or receipt.absolute_path != binding.absolute_path
            or receipt.archive_relative_path != binding.archive_relative_path
            or receipt.size_bytes != binding.size_bytes
            or receipt.sha256 != binding.sha256
        ):
            raise FormalRollingArchiveError("eviction progress identity differs")
        receipts.append(receipt)
    return tuple(receipts)


def _validate_progress_state(
    plan: RemoteEvictionPlan,
    progress: tuple[EvictedFileReceipt, ...],
) -> None:
    progressed = {row.absolute_path for row in progress}
    for binding in plan.files:
        exists = os.path.lexists(binding.absolute_path)
        if binding.absolute_path in progressed and exists:
            raise FormalRollingArchiveError(
                "a durably deleted path was recreated before resume"
            )
        if binding.absolute_path not in progressed and not exists:
            raise FormalRollingArchiveError(
                "an unrecorded planned path is missing on resume"
            )


def _publish_eviction_progress(
    *,
    progress_root: Path,
    plan: RemoteEvictionPlan,
    index: int,
    receipt: EvictedFileReceipt,
) -> None:
    progress_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if progress_root.is_symlink() or not progress_root.is_dir():
        raise FormalRollingArchiveError("eviction progress root is unsafe")
    path = progress_root / f"{index:08d}.json"
    value = {
        "schema_version": 1,
        "kind": _EVICTION_PROGRESS_KIND,
        "plan_sha256": plan.sha256,
        "index": index,
        "file": receipt.to_dict(),
    }
    digest = _semantic_sha256(value)
    if os.path.lexists(path):
        existing = _read_canonical_object(path, "eviction file progress")
        if existing != {**value, "progress_sha256": digest}:
            raise FormalRollingArchiveError("eviction progress is immutable")
        return
    _publish_digest_envelope(path, value, "progress_sha256", digest)


def _publish_eviction_failure(
    *,
    output: Path,
    plan: RemoteEvictionPlan,
    deleted: tuple[EvictedFileReceipt, ...],
    missing_unrecorded: tuple[str, ...],
    failure_code: str,
    failure_path: str | None,
    scheduler_stop: Callable[[str], None],
    clock_ns: Callable[[], int],
) -> RemoteEvictionReceipt:
    stop_succeeded = False
    try:
        scheduler_stop("remote_eviction_identity_or_progress_mismatch")
        stop_succeeded = True
    except Exception:  # noqa: BLE001 - receipt records STOP failure
        stop_succeeded = False
    status: EvictionReceiptStatus = (
        "FAILED_PARTIAL" if deleted or missing_unrecorded else "FAILED_ZERO"
    )
    receipt = RemoteEvictionReceipt(
        schema_version=1,
        kind=_EVICTION_RECEIPT_KIND,
        plan_sha256=plan.sha256,
        archive_id=plan.archive_id,
        archive_authorization_sha256=plan.archive_authorization_sha256,
        status=status,
        deleted_files=tuple(sorted(deleted, key=lambda row: row.absolute_path)),
        deleted_bytes=sum(row.size_bytes for row in deleted),
        failure_code=failure_code,
        failure_path=failure_path,
        missing_unrecorded_files=tuple(sorted(set(missing_unrecorded))),
        scheduler_stop_requested=True,
        scheduler_stop_succeeded=stop_succeeded,
        finished_at_ns=_positive_time(clock_ns()),
    )
    _publish_digest_envelope(
        output, receipt.to_dict(), "receipt_sha256", receipt.sha256
    )
    return load_remote_eviction_receipt(output)


def _validate_restore_parent(target: Path, *, candidate: Path) -> None:
    if not target.is_absolute() or target != target.resolve(strict=False):
        raise FormalRollingArchiveError("restore target is not absolute and normalized")
    if not target.is_relative_to(candidate) or target == candidate:
        raise FormalRollingArchiveError("restore target escapes the candidate root")
    parent = target.parent
    if (
        parent.is_symlink()
        or not parent.is_dir()
        or parent.resolve(strict=True) != parent
    ):
        raise FormalRollingArchiveError("restore target parent is unsafe or missing")


def _restore_one_no_replace(
    *,
    source: Path,
    target: Path,
    expected: EvictionFileBinding,
    candidate: Path,
) -> None:
    _validate_restore_parent(target, candidate=candidate)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.restore")
    source_metadata = source.lstat()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, stat.S_IMODE(source_metadata.st_mode))
    try:
        source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            source_flags |= os.O_NOFOLLOW
        source_descriptor = os.open(source, source_flags)
        try:
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                offset = 0
                while offset < len(chunk):
                    offset += os.write(descriptor, chunk[offset:])
        finally:
            os.close(source_descriptor)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        metadata, digest = _stable_file_identity(temporary)
        if metadata.st_size != expected.size_bytes or digest != expected.sha256:
            raise FormalRollingArchiveError("temporary restored file identity differs")
        os.link(temporary, target, follow_symlinks=False)
        _fsync_directory(target.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        _fsync_directory(target.parent)
    metadata, digest = _stable_file_identity(target)
    if (
        metadata.st_nlink != 1
        or metadata.st_size != expected.size_bytes
        or digest != expected.sha256
    ):
        raise FormalRollingArchiveError("restored target identity differs")


def _stable_file_identity(path: str | Path) -> tuple[os.stat_result, str]:
    source = Path(path)
    try:
        before = source.lstat()
    except FileNotFoundError as error:
        raise FormalRollingArchiveError(
            f"required file is missing: {source}"
        ) from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise FormalRollingArchiveError(f"path is not one regular file: {source}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(opened):
            raise FormalRollingArchiveError(
                f"file changed while it was opened: {source}"
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = source.lstat()
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(
        before
    ) != _stat_identity(final):
        raise FormalRollingArchiveError(f"file changed while hashing: {source}")
    return final, digest.hexdigest()


def _stat_identity(
    metadata: os.stat_result,
    *,
    include_link_count: bool = True,
) -> tuple[int, ...]:
    values = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    return (*values, metadata.st_nlink) if include_link_count else values


def _absolute_normalized(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or value != value.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    return value


def _existing_file(path: str | Path, label: str) -> Path:
    value = _absolute_normalized(path, label)
    try:
        metadata = value.lstat()
    except FileNotFoundError as error:
        raise FormalRollingArchiveError(f"{label} is missing") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise FormalRollingArchiveError(f"{label} is not one regular file")
    if value.resolve(strict=True) != value:
        raise FormalRollingArchiveError(f"{label} traverses a symbolic link")
    return value


def _existing_directory(path: str | Path, label: str) -> Path:
    value = _absolute_normalized(path, label)
    try:
        metadata = value.lstat()
    except FileNotFoundError as error:
        raise FormalRollingArchiveError(f"{label} is missing") from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise FormalRollingArchiveError(f"{label} is not one directory")
    if value.resolve(strict=True) != value:
        raise FormalRollingArchiveError(f"{label} traverses a symbolic link")
    return value


def _outside_candidate(path: str | Path, candidate: Path, label: str) -> Path:
    value = _absolute_normalized(path, label)
    if value == candidate or value.is_relative_to(candidate):
        raise ValueError(f"{label} must be outside the archive candidate root")
    return value


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _read_canonical_object(path: str | Path, label: str) -> dict[str, Any]:
    source = _existing_file(path, label)
    before, before_sha = _stable_file_identity(source)
    body = source.read_bytes()
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalRollingArchiveError(f"{label} is not UTF-8 JSON") from error
    if type(value) is not dict or body != canonical_json_bytes(value):
        raise FormalRollingArchiveError(f"{label} is not canonical JSON")
    after, after_sha = _stable_file_identity(source)
    if _stat_identity(before) != _stat_identity(after) or before_sha != after_sha:
        raise FormalRollingArchiveError(f"{label} changed while it was read")
    return value


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    source = _existing_file(path, label)
    before, before_sha = _stable_file_identity(source)
    try:
        value = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalRollingArchiveError(f"{label} is not JSON") from error
    if type(value) is not dict:
        raise FormalRollingArchiveError(f"{label} is not a JSON object")
    after, after_sha = _stable_file_identity(source)
    if _stat_identity(before) != _stat_identity(after) or before_sha != after_sha:
        raise FormalRollingArchiveError(f"{label} changed while it was read")
    return value


def _read_json_input(path: str | Path, label: str) -> dict[str, Any]:
    if str(path) != "-":
        return _read_json_object(path, label)
    body = os.sys.stdin.buffer.read(16 * 1024 * 1024 + 1)
    if not body or len(body) > 16 * 1024 * 1024:
        raise FormalRollingArchiveError(f"{label} stdin is empty or too large")
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalRollingArchiveError(f"{label} stdin is not JSON") from error
    if type(value) is not dict:
        raise FormalRollingArchiveError(f"{label} stdin is not an object")
    return value


def _publish_canonical_no_replace(path: str | Path, value: object) -> None:
    destination = _absolute_normalized(path, "publication path")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise FormalRollingArchiveError("publication parent is unsafe")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    body = canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination, follow_symlinks=False)
        _fsync_directory(destination.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _publish_digest_envelope(
    path: str | Path,
    value: Mapping[str, object],
    digest_field: str,
    digest: str,
) -> None:
    _require_sha256(digest, digest_field)
    _publish_canonical_no_replace(path, {**value, digest_field: digest})


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_time(value: int | None) -> int:
    timestamp = time.time_ns() if value is None else value
    return _require_positive_int(timestamp, "artifact timestamp")


def _safe_component(value: object) -> bool:
    return type(value) is str and _SAFE_COMPONENT.fullmatch(value) is not None


def _plain_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result = dict(value)
    try:
        canonical_json_bytes(result)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be canonical JSON data") from error
    return result


def _load_authorization(path: str | Path) -> RemoteEvictionAuthorization:
    value = _read_json_object(path, "remote eviction authorization")
    deletion = value.pop("remote_deletion_performed", False)
    expected_sha = value.pop("authorization_sha256", None)
    if deletion is not False or set(value) != set(
        RemoteEvictionAuthorization.__dataclass_fields__
    ):
        raise ValueError("remote eviction authorization fields differ")
    authorization = RemoteEvictionAuthorization(**value)
    _validate_authorization(authorization)
    if expected_sha not in {
        None,
        remote_eviction_authorization_sha256(authorization),
    }:
        raise ValueError("remote eviction authorization digest differs")
    return authorization


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    operations = parser.add_subparsers(dest="operation", required=True)

    probe = operations.add_parser("probe-node", allow_abbrev=False)
    probe.add_argument("--run-root", required=True)
    probe.add_argument("--node", required=True)
    probe.add_argument("--ordinal", required=True, type=int)

    prepare = operations.add_parser("prepare-node", allow_abbrev=False)
    prepare.add_argument("--run-root", required=True)
    prepare.add_argument("--retained-manifest", required=True)
    prepare.add_argument("--local-results-root", required=True)
    prepare.add_argument("--wave", required=True)
    prepare.add_argument("--request-output", required=True)
    prepare.add_argument("--lock", required=True)

    stage = operations.add_parser("stage-chain", allow_abbrev=False)
    stage.add_argument("--request", required=True)
    stage.add_argument("--input", required=True)
    stage.add_argument("--result-output", required=True)
    stage.add_argument("--authorization-output", required=True)
    stage.add_argument("--lock", required=True)

    staged_evict = operations.add_parser("evict-staged", allow_abbrev=False)
    staged_evict.add_argument("--request", required=True)
    staged_evict.add_argument("--archive-result", required=True)
    staged_evict.add_argument("--authorization", required=True)
    staged_evict.add_argument("--retained-manifest", required=True)
    staged_evict.add_argument("--operator-db", required=True)
    staged_evict.add_argument("--plan-output", required=True)
    staged_evict.add_argument("--receipt-output", required=True)
    staged_evict.add_argument("--plan-lock", required=True)
    staged_evict.add_argument("--executor-lock", required=True)

    restore_member = operations.add_parser("restore-member", allow_abbrev=False)
    restore_member.add_argument("--plan", required=True)
    restore_member.add_argument("--eviction-receipt", required=True)
    restore_member.add_argument("--archive-result", required=True)
    restore_member.add_argument("--relative-path", required=True)
    restore_member.add_argument("--progress-output", required=True)
    restore_member.add_argument("--operator-db", required=True)
    restore_member.add_argument("--lock", required=True)
    restore_member.add_argument(
        "--minimum-free-bytes",
        type=int,
        default=REMOTE_SPOOL_SAFETY_RESERVE_BYTES,
    )

    finalize_stream = operations.add_parser(
        "finalize-stream-restore", allow_abbrev=False
    )
    finalize_stream.add_argument("--plan", required=True)
    finalize_stream.add_argument("--eviction-receipt", required=True)
    finalize_stream.add_argument("--archive-result", required=True)
    finalize_stream.add_argument("--progress-root", required=True)
    finalize_stream.add_argument("--receipt-output", required=True)
    finalize_stream.add_argument("--operator-db", required=True)
    finalize_stream.add_argument("--lock", required=True)

    manifest = operations.add_parser("manifest", allow_abbrev=False)
    manifest.add_argument("--run-root", required=True)
    manifest.add_argument("--candidate-root", required=True)
    manifest.add_argument("--retained-manifest", required=True)
    manifest.add_argument("--lock", required=True)

    request = operations.add_parser("request", allow_abbrev=False)
    request.add_argument("--manifest", required=True)
    request.add_argument("--retained-manifest", required=True)
    request.add_argument("--local-results-root", required=True)
    request.add_argument("--wave", required=True)
    request.add_argument("--archive-id")
    request.add_argument("--safe-boundary")
    request.add_argument("--cell-id")
    request.add_argument("--attempt", type=int)
    request.add_argument("--output", required=True)
    request.add_argument("--lock", required=True)

    plan = operations.add_parser("plan", allow_abbrev=False)
    plan.add_argument("--request", required=True)
    plan.add_argument("--archive-result", required=True)
    plan.add_argument("--authorization", required=True)
    plan.add_argument("--retained-manifest", required=True)
    plan.add_argument("--operator-checkpoint", required=True)
    plan.add_argument("--operator-snapshot", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--lock", required=True)

    execute = operations.add_parser("execute", allow_abbrev=False)
    execute.add_argument("--plan", required=True)
    execute.add_argument("--operator-db", required=True)
    execute.add_argument("--receipt", required=True)
    execute.add_argument("--lock", required=True)

    restore = operations.add_parser("restore", allow_abbrev=False)
    restore.add_argument("--plan", required=True)
    restore.add_argument("--archive-result", required=True)
    restore.add_argument("--receipt", required=True)
    restore.add_argument("--lock", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.operation == "probe-node":
            output = probe_retained_archive_boundary(
                run_root=arguments.run_root,
                node=arguments.node,
                ordinal=arguments.ordinal,
            )
        elif arguments.operation == "prepare-node":
            output = asdict(
                prepare_rolling_archive_node(
                    run_root=arguments.run_root,
                    retained_dependency_manifest_path=(arguments.retained_manifest),
                    local_results_root=arguments.local_results_root,
                    wave=arguments.wave,
                    request_output_path=arguments.request_output,
                    lock_path=arguments.lock,
                )
            )
        elif arguments.operation == "stage-chain":
            staged_input = _read_json_input(arguments.input, "archive chain")
            if set(staged_input) != {"result", "authorization"}:
                raise ValueError("staged archive chain fields differ")
            output = asdict(
                stage_remote_archive_chain(
                    request_path=arguments.request,
                    result_value=staged_input["result"],
                    authorization_value=staged_input["authorization"],
                    result_output_path=arguments.result_output,
                    authorization_output_path=arguments.authorization_output,
                    lock_path=arguments.lock,
                )
            )
        elif arguments.operation == "evict-staged":
            plan, receipt = run_staged_remote_eviction(
                request_path=arguments.request,
                result_path=arguments.archive_result,
                authorization_path=arguments.authorization,
                retained_dependency_manifest_path=arguments.retained_manifest,
                operator_database_path=arguments.operator_db,
                plan_output_path=arguments.plan_output,
                receipt_output_path=arguments.receipt_output,
                plan_lock_path=arguments.plan_lock,
                executor_lock_path=arguments.executor_lock,
            )
            output = {
                "plan": {**plan.to_dict(), "plan_sha256": plan.sha256},
                "receipt": {
                    **receipt.to_dict(),
                    "receipt_sha256": receipt.sha256,
                },
                "plan_path": str(
                    _absolute_normalized(arguments.plan_output, "plan output")
                ),
                "receipt_path": str(
                    _absolute_normalized(arguments.receipt_output, "receipt output")
                ),
            }
        elif arguments.operation == "restore-member":
            database = _existing_file(arguments.operator_db, "operator database")
            with ExperimentOperatorStore(database) as store:
                store.set_dispatch_stop("rolling_archive_stream_restore")
                progress = restore_remote_member_from_stream(
                    plan_path=arguments.plan,
                    eviction_receipt_path=arguments.eviction_receipt,
                    remote_archive_result_path=arguments.archive_result,
                    archive_relative_path=arguments.relative_path,
                    progress_output_path=arguments.progress_output,
                    lock_path=arguments.lock,
                    stream=os.sys.stdin.buffer,
                    minimum_free_bytes=arguments.minimum_free_bytes,
                )
            output = {
                **progress.to_dict(),
                "progress_sha256": progress.sha256,
            }
        elif arguments.operation == "finalize-stream-restore":
            database = _existing_file(arguments.operator_db, "operator database")
            with ExperimentOperatorStore(database) as store:
                store.set_dispatch_stop("rolling_archive_stream_restore_finalize")
                receipt = finalize_remote_stream_restore(
                    plan_path=arguments.plan,
                    eviction_receipt_path=arguments.eviction_receipt,
                    remote_archive_result_path=arguments.archive_result,
                    progress_root=arguments.progress_root,
                    receipt_output_path=arguments.receipt_output,
                    lock_path=arguments.lock,
                )
            output = {
                **receipt.to_dict(),
                "receipt_sha256": receipt.sha256,
            }
        elif arguments.operation == "manifest":
            output: object = asdict(
                publish_formal_archive_sha256_manifest(
                    run_root=arguments.run_root,
                    candidate_root=arguments.candidate_root,
                    retained_dependency_manifest_path=arguments.retained_manifest,
                    lock_path=arguments.lock,
                )
            )
        elif arguments.operation == "request":
            request = build_archive_request(
                manifest_path=arguments.manifest,
                retained_dependency_manifest_path=arguments.retained_manifest,
                local_results_root=arguments.local_results_root,
                wave=arguments.wave,
                archive_id=arguments.archive_id,
                safe_boundary=arguments.safe_boundary,
                cell_id=arguments.cell_id,
                attempt=arguments.attempt,
            )
            output = asdict(
                publish_archive_request(
                    arguments.output,
                    request,
                    lock_path=arguments.lock,
                )
            )
        elif arguments.operation == "plan":
            request = load_archive_request(arguments.request)
            result = load_remote_archive_result(arguments.archive_result)
            authorization = _load_authorization(arguments.authorization)
            plan = build_remote_eviction_plan(
                request=request,
                remote_archive_result=result,
                authorization=authorization,
                retained_dependency_manifest_path=arguments.retained_manifest,
                operator_checkpoint=_read_json_object(
                    arguments.operator_checkpoint, "operator archive checkpoint"
                ),
                operator_snapshot=_read_json_object(
                    arguments.operator_snapshot, "operator snapshot"
                ),
            )
            output = publish_remote_eviction_plan(
                arguments.output,
                plan,
                lock_path=arguments.lock,
            ).to_dict()
        elif arguments.operation == "execute":
            plan = load_remote_eviction_plan(arguments.plan)
            database = _existing_file(arguments.operator_db, "operator database")
            if database == Path(plan.archive_candidate_root) or database.is_relative_to(
                Path(plan.archive_candidate_root)
            ):
                raise ValueError("operator database cannot be inside eviction scope")
            with ExperimentOperatorStore(database, run_id=plan.run_id) as store:
                receipt = execute_remote_eviction_plan(
                    plan_path=arguments.plan,
                    receipt_path=arguments.receipt,
                    lock_path=arguments.lock,
                    operator_snapshot=store.snapshot,
                    scheduler_stop=store.set_dispatch_stop,
                )
            output = receipt.to_dict()
        elif arguments.operation == "restore":
            output = restore_evicted_files(
                plan_path=arguments.plan,
                remote_archive_result_path=arguments.archive_result,
                receipt_path=arguments.receipt,
                lock_path=arguments.lock,
            ).to_dict()
        else:
            raise AssertionError(
                f"unhandled rolling archive operation: {arguments.operation}"
            )
    except (
        OSError,
        TypeError,
        ValueError,
        FormalRollingArchiveError,
    ) as error:
        print(f"formal rolling archive: {error}", file=os.sys.stderr)
        return 2
    print(canonical_json_bytes(output).decode("utf-8"), end="")
    if isinstance(output, dict) and output.get("status", "COMPLETE") != "COMPLETE":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArchiveManifestFile",
    "ArchiveRestoreReceipt",
    "EvictedFileReceipt",
    "EvictionFileBinding",
    "FormalArchiveSha256Manifest",
    "FormalRollingArchiveError",
    "PublishedArchiveManifest",
    "RemoteEvictionGate",
    "RemoteEvictionPlan",
    "RemoteEvictionReceipt",
    "RemoteStreamRestoreProgress",
    "RestoredFileReceipt",
    "SimulatedEvictionCrash",
    "StagedArchiveChain",
    "build_archive_request",
    "build_remote_eviction_plan",
    "evaluate_remote_eviction",
    "execute_remote_eviction_plan",
    "finalize_remote_stream_restore",
    "load_archive_request",
    "load_archive_restore_receipt",
    "load_formal_archive_sha256_manifest",
    "load_remote_eviction_plan",
    "load_remote_eviction_receipt",
    "load_remote_stream_restore_progress",
    "main",
    "prepare_rolling_archive_node",
    "probe_retained_archive_boundary",
    "publish_archive_request",
    "publish_formal_archive_sha256_manifest",
    "publish_remote_eviction_plan",
    "remote_eviction_authorization_sha256",
    "restore_evicted_files",
    "restore_remote_member_from_stream",
    "run_staged_remote_eviction",
    "stage_remote_archive_chain",
]
