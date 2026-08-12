"""Fail-closed DOWNLOAD planning and future raw-evidence authority.

The industrial registry contains exclusive-host DOWNLOAD cells, but this
release has no release-owned downloader terminal.  This module therefore
provides two intentionally separate surfaces:

* a release-derived ``DownloadPlan`` that binds the registry cell, locked
  model revisions, GPU inventory, physical assignment, ExperimentBudget, and
  every expected output hash; and
* a diagnostic raw replay surface for a future downloader terminal and atomic
  result pointer.

The diagnostic surface can prove that bytes and paths are internally
consistent.  It can never mint formal completion in this release.  The formal
execution gate is an unconditional named block and touches no filesystem, GPU,
provider, or terminal state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Literal, NoReturn, Self

from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.planning import BudgetJobKind, ExperimentBudget
from lightcone_spec.experiments.registry import (
    ExperimentCell,
    ExperimentRegistry,
    WorkloadClass,
    content_sha256,
)
from lightcone_spec.orchestration.industrial import IndustrialPhysicalAssignment

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_STREAM_CHUNK_BYTES = 1024 * 1024

RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON = (
    "release_owned_downloader_terminal_unavailable"
)

# Source-owned and intentionally empty.  Callers cannot add an issuer or key to
# either the formal gate or the diagnostic raw reducer.
RELEASE_DOWNLOAD_TERMINAL_ISSUERS: tuple[tuple[str, str], ...] = ()
RELEASE_DOWNLOAD_POINTER_PUBLISHERS: tuple[tuple[str, str], ...] = ()

DOWNLOAD_PLAN_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "download_plan_protocol",
        "bindings": (
            "registry_cell",
            "locked_model_revisions",
            "gpu_inventory_and_source_receipt",
            "physical_and_scheduler_assignment",
            "capacity_and_budget_authorities",
            "experiment_budget",
            "expected_output_paths_sizes_hashes",
            "exclusive_host",
        ),
        "formal_terminal_available": False,
        "blocked_reason": RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON,
    }
)
DOWNLOAD_TERMINAL_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "future_download_terminal_protocol",
        "required": (
            "exact_plan_and_input_identities",
            "successful_monotonic_lifecycle",
            "complete_expected_output_paths_sizes_hashes",
            "release_owned_terminal_signature",
        ),
        "current_release_issuers": RELEASE_DOWNLOAD_TERMINAL_ISSUERS,
    }
)
DOWNLOAD_RESULT_POINTER_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "future_download_atomic_result_pointer_protocol",
        "bindings": (
            "plan_raw_authority",
            "terminal_absolute_path_raw_sha256_size_semantic_sha256",
            "sorted_output_absolute_paths_raw_sha256_sizes",
            "output_manifest_sha256",
            "release_owned_pointer_signature",
        ),
        "publication": "atomic_no_replace_with_exact_sidecar",
        "resume": "reopen_and_revalidate_every_bound_path",
        "current_release_publishers": RELEASE_DOWNLOAD_POINTER_PUBLISHERS,
        "serialized_summary_is_not_authority": True,
    }
)
FUTURE_DOWNLOAD_RAW_AUTHORITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "future_download_raw_authority_protocol",
        "sources": (
            "release_derived_plan_raw_and_sidecar",
            "future_terminal_raw_and_sidecar",
            "future_result_pointer_raw_and_sidecar",
            "all_download_output_bytes",
        ),
        "fresh_replay_required": True,
        "formal_status": "BLOCKED",
        "blocked_reason": RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON,
    }
)


class DownloadExecutionBlocked(RuntimeError):
    """Raised before any DOWNLOAD execution-side mutation."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(f"DOWNLOAD execution is BLOCKED: {reason_code}")
        self.reason_code = reason_code


def _require_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_safe_id(label: str, value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


def _require_absolute_normalized(label: str, value: object) -> str:
    text = _require_text(label, value)
    path = PurePath(text)
    if (
        not path.is_absolute()
        or path == PurePath(path.anchor)
        or str(path) != text
        or any(part == ".." for part in path.parts)
    ):
        raise ValueError(f"{label} must be absolute, normalized, and non-root")
    return text


def _require_relative_path(label: str, value: object) -> str:
    text = _require_text(label, value)
    path = PurePath(text)
    if (
        path.is_absolute()
        or not path.parts
        or str(path) != text
        or any(part in {"", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative path")
    return text


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json(body: bytes, *, label: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains non-finite constant {value}")

    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    return value


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_resolved_regular(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{label} path must be absolute, resolved, and non-symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} path is unavailable") from error
    if resolved != path:
        raise ValueError(f"{label} path must be absolute, resolved, and non-symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0:
        os.close(descriptor)
        raise ValueError(f"{label} must be a regular file")
    return descriptor, before


def _assert_stable_open_file(
    path: Path,
    descriptor: int,
    before: os.stat_result,
    *,
    label: str,
    bytes_read: int,
) -> os.stat_result:
    after = os.fstat(descriptor)
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} disappeared during coordinated read") from error
    if (
        not stat.S_ISREG(current.st_mode)
        or _identity(before) != _identity(after)
        or _identity(after) != _identity(current)
        or bytes_read != after.st_size
    ):
        raise ValueError(f"{label} changed during coordinated read")
    return after


def _stable_json_bytes(path: Path, *, label: str) -> bytes:
    descriptor, before = _open_resolved_regular(path, label=label)
    try:
        if before.st_size > _MAX_JSON_BYTES:
            raise ValueError(f"{label} exceeds the raw JSON size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read()
        _assert_stable_open_file(
            path,
            descriptor,
            before,
            label=label,
            bytes_read=len(body),
        )
        return body
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class _StableFileDigest:
    size: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


def _stable_file_digest(path: Path, *, label: str) -> _StableFileDigest:
    descriptor, before = _open_resolved_regular(path, label=label)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                chunk = handle.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        after = _assert_stable_open_file(
            path,
            descriptor,
            before,
            label=label,
            bytes_read=size,
        )
        return _StableFileDigest(
            size=size,
            sha256=digest.hexdigest(),
            device=after.st_dev,
            inode=after.st_ino,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class DownloadModelRevision:
    role: str
    repository: str
    revision: str
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        _require_safe_id("model role", self.role)
        _require_text("model repository", self.repository)
        if (
            not isinstance(self.revision, str)
            or _REVISION.fullmatch(self.revision) is None
        ):
            raise ValueError("model revision must be a locked 40/64-hex identity")
        _require_sha256("model source manifest", self.source_manifest_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "repository": self.repository,
            "revision": self.revision,
            "source_manifest_sha256": self.source_manifest_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {"role", "repository", "revision", "source_manifest_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("download model revision fields differ from schema")
        return cls(
            role=value["role"],
            repository=value["repository"],
            revision=value["revision"],
            source_manifest_sha256=value["source_manifest_sha256"],
        )

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class DownloadOutputExpectation:
    relative_path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _require_relative_path("download output", self.relative_path)
        if type(self.size) is not int or self.size < 0:
            raise ValueError("download output size must be non-negative")
        _require_sha256("download output digest", self.sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "relative_path",
            "size",
            "sha256",
        }:
            raise ValueError("download output expectation fields differ from schema")
        return cls(
            relative_path=value["relative_path"],
            size=value["size"],
            sha256=value["sha256"],
        )


@dataclass(frozen=True)
class DownloadExecutionInputs:
    registry_sha256: str
    cell_id: str
    cell_declaration_sha256: str
    experiment: str
    model: str
    backend: str
    variant: str
    workload_class: str
    model_revisions: tuple[DownloadModelRevision, ...]
    model_revision_manifest_sha256: str
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    inventory_gpu_uuids: tuple[str, ...]
    host_id: str
    physical_assignment_sha256: str
    dispatch_plan_sha256: str
    budget_plan_sha256: str
    capacity_authority_sha256: str
    budget_materialization_authority_sha256: str
    assignment_sha256: str
    work_item_sha256: str
    experiment_budget_sha256: str
    assigned_gpu_uuids: tuple[str, ...]
    fixed_instance_gpu_count: int
    cache_root: str
    evidence_root: str

    def __post_init__(self) -> None:
        for name in (
            "registry_sha256",
            "cell_id",
            "cell_declaration_sha256",
            "model_revision_manifest_sha256",
            "inventory_sha256",
            "inventory_source_receipt_sha256",
            "physical_assignment_sha256",
            "dispatch_plan_sha256",
            "budget_plan_sha256",
            "capacity_authority_sha256",
            "budget_materialization_authority_sha256",
            "assignment_sha256",
            "work_item_sha256",
            "experiment_budget_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in ("experiment", "model", "backend", "variant", "host_id"):
            _require_text(name, getattr(self, name))
        if self.workload_class != WorkloadClass.DOWNLOAD.value:
            raise ValueError("download inputs require workload_class=download")
        if not self.model_revisions or any(
            type(value) is not DownloadModelRevision for value in self.model_revisions
        ):
            raise TypeError("download inputs require exact locked model revisions")
        roles = tuple(value.role for value in self.model_revisions)
        if roles != tuple(sorted(set(roles))):
            raise ValueError("download model revisions must be role-sorted and unique")
        target = tuple(
            value for value in self.model_revisions if value.role == "target"
        )
        if len(target) != 1 or target[0].repository != self.model:
            raise ValueError("download inputs require one matching target revision")
        expected_revision_manifest = content_sha256(
            [value.to_dict() for value in self.model_revisions]
        )
        if self.model_revision_manifest_sha256 != expected_revision_manifest:
            raise ValueError("download model revision manifest differs")
        if (
            not self.inventory_gpu_uuids
            or self.inventory_gpu_uuids != tuple(sorted(set(self.inventory_gpu_uuids)))
            or not self.assigned_gpu_uuids
            or len(self.assigned_gpu_uuids) != len(set(self.assigned_gpu_uuids))
            or not set(self.assigned_gpu_uuids) <= set(self.inventory_gpu_uuids)
        ):
            raise ValueError("download inventory/assignment GPU coverage is invalid")
        if (
            type(self.fixed_instance_gpu_count) is not int
            or self.fixed_instance_gpu_count != len(self.inventory_gpu_uuids)
            or self.fixed_instance_gpu_count < len(self.assigned_gpu_uuids)
        ):
            raise ValueError("download fixed-instance GPU coverage is invalid")
        _require_absolute_normalized("download cache root", self.cache_root)
        _require_absolute_normalized("download evidence root", self.evidence_root)

    @property
    def model_revision_sha256s(self) -> tuple[str, ...]:
        return tuple(value.sha256 for value in self.model_revisions)

    def to_dict(self) -> dict[str, object]:
        return {
            "registry_sha256": self.registry_sha256,
            "cell_id": self.cell_id,
            "cell_declaration_sha256": self.cell_declaration_sha256,
            "experiment": self.experiment,
            "model": self.model,
            "backend": self.backend,
            "variant": self.variant,
            "workload_class": self.workload_class,
            "model_revisions": [value.to_dict() for value in self.model_revisions],
            "model_revision_manifest_sha256": self.model_revision_manifest_sha256,
            "inventory_sha256": self.inventory_sha256,
            "inventory_source_receipt_sha256": (self.inventory_source_receipt_sha256),
            "inventory_gpu_uuids": list(self.inventory_gpu_uuids),
            "host_id": self.host_id,
            "physical_assignment_sha256": self.physical_assignment_sha256,
            "dispatch_plan_sha256": self.dispatch_plan_sha256,
            "budget_plan_sha256": self.budget_plan_sha256,
            "capacity_authority_sha256": self.capacity_authority_sha256,
            "budget_materialization_authority_sha256": (
                self.budget_materialization_authority_sha256
            ),
            "assignment_sha256": self.assignment_sha256,
            "work_item_sha256": self.work_item_sha256,
            "experiment_budget_sha256": self.experiment_budget_sha256,
            "assigned_gpu_uuids": list(self.assigned_gpu_uuids),
            "fixed_instance_gpu_count": self.fixed_instance_gpu_count,
            "cache_root": self.cache_root,
            "evidence_root": self.evidence_root,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {
            "registry_sha256",
            "cell_id",
            "cell_declaration_sha256",
            "experiment",
            "model",
            "backend",
            "variant",
            "workload_class",
            "model_revisions",
            "model_revision_manifest_sha256",
            "inventory_sha256",
            "inventory_source_receipt_sha256",
            "inventory_gpu_uuids",
            "host_id",
            "physical_assignment_sha256",
            "dispatch_plan_sha256",
            "budget_plan_sha256",
            "capacity_authority_sha256",
            "budget_materialization_authority_sha256",
            "assignment_sha256",
            "work_item_sha256",
            "experiment_budget_sha256",
            "assigned_gpu_uuids",
            "fixed_instance_gpu_count",
            "cache_root",
            "evidence_root",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("download execution input fields differ from schema")
        revisions = value["model_revisions"]
        inventory = value["inventory_gpu_uuids"]
        assigned = value["assigned_gpu_uuids"]
        if (
            type(revisions) is not list
            or type(inventory) is not list
            or type(assigned) is not list
        ):
            raise TypeError("download execution input arrays are malformed")
        payload = dict(value)
        payload["model_revisions"] = tuple(
            DownloadModelRevision.from_dict(row) for row in revisions
        )
        payload["inventory_gpu_uuids"] = tuple(inventory)
        payload["assigned_gpu_uuids"] = tuple(assigned)
        return cls(**payload)

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def _registry_cell(
    registry: ExperimentRegistry,
    cell: ExperimentCell,
) -> ExperimentCell:
    if type(registry) is not ExperimentRegistry or type(cell) is not ExperimentCell:
        raise TypeError("download planning requires exact registry/cell values")
    matches = tuple(value for value in registry.cells if value.cell_id == cell.cell_id)
    if matches != (cell,):
        raise ValueError("download cell is absent, duplicated, or changed in registry")
    return cell


def _derive_download_inputs(
    *,
    registry: ExperimentRegistry,
    cell: ExperimentCell,
    model_revisions: tuple[DownloadModelRevision, ...],
    inventory: GpuInventory,
    assignment: IndustrialPhysicalAssignment,
    budget: ExperimentBudget,
) -> DownloadExecutionInputs:
    cell = _registry_cell(registry, cell)
    if type(inventory) is not GpuInventory:
        raise TypeError("download planning requires an exact GpuInventory")
    if type(assignment) is not IndustrialPhysicalAssignment:
        raise TypeError("download planning requires an exact physical assignment")
    if type(budget) is not ExperimentBudget:
        raise TypeError("download planning requires an exact ExperimentBudget")
    if type(model_revisions) is not tuple or any(
        type(value) is not DownloadModelRevision for value in model_revisions
    ):
        raise TypeError("download planning requires a tuple of exact model revisions")
    inventory.__post_init__()
    assignment.__post_init__()
    budget.__post_init__()
    if (
        cell.resources.workload_class is not WorkloadClass.DOWNLOAD
        or not cell.resources.exclusive
        or budget.workload_class is not WorkloadClass.DOWNLOAD
        or budget.job_kind is not BudgetJobKind.DOWNLOAD
        or budget.cell_id != cell.cell_id
        or budget.experiment != cell.identity.experiment
        or budget.method != cell.identity.method
        or budget.topology != cell.identity.topology
        or budget.gpu_count != cell.resources.gpu_count
        or budget.measured_gpu_ms is not None
    ):
        raise ValueError("download registry cell and ExperimentBudget differ")
    inventory_uuids = tuple(device.uuid for device in inventory.devices)
    if (
        assignment.inventory_sha256 != inventory.sha256
        or assignment.inventory_source_receipt_sha256 != inventory.source_receipt_sha256
        or assignment.experiment_budget_sha256 != budget.sha256
        or assignment.fixed_instance_gpu_count != len(inventory.devices)
        or len(assignment.gpu_uuids) != cell.resources.gpu_count
        or not set(assignment.gpu_uuids) <= set(inventory_uuids)
        or len(inventory.host_ids) != 1
        or assignment.host_id != inventory.host_ids[0]
    ):
        raise ValueError("download inventory, assignment, and budget differ")
    roles = tuple(value.role for value in model_revisions)
    if roles != tuple(sorted(set(roles))):
        raise ValueError("download model revisions must be role-sorted and unique")
    return DownloadExecutionInputs(
        registry_sha256=registry.sha256,
        cell_id=cell.cell_id,
        cell_declaration_sha256=cell.sha256,
        experiment=cell.identity.experiment,
        model=cell.identity.model,
        backend=cell.identity.backend,
        variant=cell.identity.variant,
        workload_class=WorkloadClass.DOWNLOAD.value,
        model_revisions=model_revisions,
        model_revision_manifest_sha256=content_sha256(
            [value.to_dict() for value in model_revisions]
        ),
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        inventory_gpu_uuids=inventory_uuids,
        host_id=assignment.host_id,
        physical_assignment_sha256=assignment.sha256,
        dispatch_plan_sha256=assignment.dispatch_plan_sha256,
        budget_plan_sha256=assignment.budget_plan_sha256,
        capacity_authority_sha256=assignment.capacity_authority_sha256,
        budget_materialization_authority_sha256=(
            assignment.budget_materialization_authority_sha256
        ),
        assignment_sha256=assignment.assignment_sha256,
        work_item_sha256=assignment.work_item_sha256,
        experiment_budget_sha256=budget.sha256,
        assigned_gpu_uuids=assignment.gpu_uuids,
        fixed_instance_gpu_count=assignment.fixed_instance_gpu_count,
        cache_root=_require_absolute_normalized(
            "download cache root", cell.resources.cache_root
        ),
        evidence_root=_require_absolute_normalized(
            "download evidence root", cell.resources.evidence_root
        ),
    )


def _download_lifecycle_paths(inputs: DownloadExecutionInputs) -> tuple[str, str, str]:
    root = PurePath(inputs.evidence_root)
    prefix = f"{inputs.cell_id}.download"
    return (
        str(root / f"{prefix}.plan.json"),
        str(root / f"{prefix}.terminal.json"),
        str(root / f"{prefix}.result-pointer.json"),
    )


@dataclass(frozen=True)
class DownloadPlan:
    schema_version: int
    kind: str
    protocol_sha256: str
    inputs: DownloadExecutionInputs
    expected_outputs: tuple[DownloadOutputExpectation, ...]
    output_manifest_sha256: str
    terminal_protocol_sha256: str
    result_pointer_protocol_sha256: str
    plan_path: str
    terminal_receipt_path: str
    result_pointer_path: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "download_plan"
        ):
            raise ValueError("download plan schema is unsupported")
        if self.protocol_sha256 != DOWNLOAD_PLAN_PROTOCOL_SHA256:
            raise ValueError("download plan uses another release protocol")
        if type(self.inputs) is not DownloadExecutionInputs:
            raise TypeError("download plan requires exact execution inputs")
        self.inputs.__post_init__()
        if not self.expected_outputs or any(
            type(value) is not DownloadOutputExpectation
            for value in self.expected_outputs
        ):
            raise TypeError("download plan requires exact expected output hashes")
        paths = tuple(value.relative_path for value in self.expected_outputs)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("download outputs must be path-sorted and unique")
        expected_manifest = content_sha256(
            [value.to_dict() for value in self.expected_outputs]
        )
        if self.output_manifest_sha256 != expected_manifest:
            raise ValueError("download output manifest digest differs")
        if self.terminal_protocol_sha256 != DOWNLOAD_TERMINAL_PROTOCOL_SHA256:
            raise ValueError("download plan uses another terminal protocol")
        if (
            self.result_pointer_protocol_sha256
            != DOWNLOAD_RESULT_POINTER_PROTOCOL_SHA256
        ):
            raise ValueError("download plan uses another result-pointer protocol")
        paths = (
            self.plan_path,
            self.terminal_receipt_path,
            self.result_pointer_path,
        )
        if paths != _download_lifecycle_paths(self.inputs):
            raise ValueError("download lifecycle paths differ from release derivation")
        if len(set(paths)) != len(paths):
            raise ValueError("download lifecycle paths must be distinct")
        root = PurePath(self.inputs.evidence_root)
        for label, raw in zip(
            ("download plan", "download terminal", "download result pointer"),
            paths,
            strict=True,
        ):
            path = PurePath(_require_absolute_normalized(label, raw))
            if path.parent != root:
                raise ValueError(f"{label} must be a direct evidence-root child")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "inputs": self.inputs.to_dict(),
            "expected_outputs": [value.to_dict() for value in self.expected_outputs],
            "output_manifest_sha256": self.output_manifest_sha256,
            "terminal_protocol_sha256": self.terminal_protocol_sha256,
            "result_pointer_protocol_sha256": self.result_pointer_protocol_sha256,
            "plan_path": self.plan_path,
            "terminal_receipt_path": self.terminal_receipt_path,
            "result_pointer_path": self.result_pointer_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "inputs",
            "expected_outputs",
            "output_manifest_sha256",
            "terminal_protocol_sha256",
            "result_pointer_protocol_sha256",
            "plan_path",
            "terminal_receipt_path",
            "result_pointer_path",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("download plan fields differ from schema")
        outputs = value["expected_outputs"]
        if type(outputs) is not list:
            raise TypeError("download expected outputs must be an array")
        return cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            protocol_sha256=value["protocol_sha256"],
            inputs=DownloadExecutionInputs.from_dict(value["inputs"]),
            expected_outputs=tuple(
                DownloadOutputExpectation.from_dict(row) for row in outputs
            ),
            output_manifest_sha256=value["output_manifest_sha256"],
            terminal_protocol_sha256=value["terminal_protocol_sha256"],
            result_pointer_protocol_sha256=value["result_pointer_protocol_sha256"],
            plan_path=value["plan_path"],
            terminal_receipt_path=value["terminal_receipt_path"],
            result_pointer_path=value["result_pointer_path"],
        )

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def issue_download_plan(
    *,
    registry: ExperimentRegistry,
    cell: ExperimentCell,
    model_revisions: tuple[DownloadModelRevision, ...],
    inventory: GpuInventory,
    assignment: IndustrialPhysicalAssignment,
    budget: ExperimentBudget,
    expected_outputs: tuple[DownloadOutputExpectation, ...],
) -> DownloadPlan:
    """Derive an immutable DOWNLOAD plan without touching its lifecycle paths."""

    if type(expected_outputs) is not tuple or any(
        type(value) is not DownloadOutputExpectation for value in expected_outputs
    ):
        raise TypeError("download expected outputs must be an exact tuple")
    output_paths = tuple(value.relative_path for value in expected_outputs)
    if not expected_outputs or output_paths != tuple(sorted(set(output_paths))):
        raise ValueError("download expected outputs must be path-sorted and unique")
    inputs = _derive_download_inputs(
        registry=registry,
        cell=cell,
        model_revisions=model_revisions,
        inventory=inventory,
        assignment=assignment,
        budget=budget,
    )
    plan_path, terminal_path, pointer_path = _download_lifecycle_paths(inputs)
    return DownloadPlan(
        schema_version=1,
        kind="download_plan",
        protocol_sha256=DOWNLOAD_PLAN_PROTOCOL_SHA256,
        inputs=inputs,
        expected_outputs=expected_outputs,
        output_manifest_sha256=content_sha256(
            [value.to_dict() for value in expected_outputs]
        ),
        terminal_protocol_sha256=DOWNLOAD_TERMINAL_PROTOCOL_SHA256,
        result_pointer_protocol_sha256=DOWNLOAD_RESULT_POINTER_PROTOCOL_SHA256,
        plan_path=plan_path,
        terminal_receipt_path=terminal_path,
        result_pointer_path=pointer_path,
    )


def require_release_download_execution(
    plan: DownloadPlan | None = None,
) -> NoReturn:
    """Block before any path, GPU, provider, process, or cache side effect."""

    if plan is not None:
        if type(plan) is not DownloadPlan:
            raise TypeError("download gate requires an exact DownloadPlan")
        # Pure value/PurePath validation only.  No pathlib.Path call is allowed
        # on the formal gate path.
        plan.__post_init__()
    raise DownloadExecutionBlocked(RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON)


@dataclass(frozen=True)
class BoundDownloadJson:
    schema_version: int
    path: str
    size: int
    raw_sha256: str
    semantic_sha256: str
    sidecar_path: str
    sidecar_size: int
    sidecar_raw_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("bound download JSON schema is unsupported")
        _require_absolute_normalized("bound download JSON path", self.path)
        _require_absolute_normalized(
            "bound download JSON sidecar path", self.sidecar_path
        )
        if type(self.size) is not int or self.size < 1:
            raise ValueError("bound download JSON size must be positive")
        if type(self.sidecar_size) is not int or self.sidecar_size != 65:
            raise ValueError("bound download JSON sidecar must be one digest line")
        for name in ("raw_sha256", "semantic_sha256", "sidecar_raw_sha256"):
            _require_sha256(name, getattr(self, name))
        if self.sidecar_path != f"{self.path}.sha256":
            raise ValueError("bound download JSON sidecar path differs")

    @classmethod
    def bind(
        cls,
        path: str | Path,
        *,
        expected_path: str,
        label: str,
    ) -> tuple[Self, dict[str, object]]:
        expected = _require_absolute_normalized(f"{label} expected path", expected_path)
        requested = Path(path)
        if str(requested) != expected:
            raise ValueError(f"{label} path differs from the release plan")
        body = _stable_json_bytes(requested, label=label)
        sidecar = Path(f"{requested}.sha256")
        sidecar_body = _stable_json_bytes(sidecar, label=f"{label} sidecar")
        value = _strict_json(body, label=label)
        if body != _canonical_bytes(value):
            raise ValueError(f"{label} is not canonical JSON")
        try:
            declared = sidecar_body.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} sidecar is not ASCII") from error
        if not declared.endswith("\n") or declared.count("\n") != 1:
            raise ValueError(f"{label} sidecar is not canonical")
        semantic = _require_sha256(f"{label} semantic digest", declared[:-1])
        return (
            cls(
                schema_version=1,
                path=str(requested),
                size=len(body),
                raw_sha256=hashlib.sha256(body).hexdigest(),
                semantic_sha256=semantic,
                sidecar_path=str(sidecar),
                sidecar_size=len(sidecar_body),
                sidecar_raw_sha256=hashlib.sha256(sidecar_body).hexdigest(),
            ),
            value,
        )

    def reopen(self, *, label: str) -> dict[str, object]:
        body = _stable_json_bytes(Path(self.path), label=label)
        sidecar = _stable_json_bytes(
            Path(self.sidecar_path),
            label=f"{label} sidecar",
        )
        if (
            len(body) != self.size
            or len(sidecar) != self.sidecar_size
            or hashlib.sha256(body).hexdigest() != self.raw_sha256
            or hashlib.sha256(sidecar).hexdigest() != self.sidecar_raw_sha256
            or sidecar != f"{self.semantic_sha256}\n".encode("ascii")
        ):
            raise ValueError(f"{label} differs from its fresh raw replay")
        value = _strict_json(body, label=label)
        if body != _canonical_bytes(value):
            raise ValueError(f"{label} is no longer canonical JSON")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "size": self.size,
            "raw_sha256": self.raw_sha256,
            "semantic_sha256": self.semantic_sha256,
            "sidecar_path": self.sidecar_path,
            "sidecar_size": self.sidecar_size,
            "sidecar_raw_sha256": self.sidecar_raw_sha256,
        }


@dataclass(frozen=True)
class DownloadPlanAuthority:
    schema_version: int
    source: BoundDownloadJson
    plan_sha256: str
    registry_sha256: str
    cell_id: str
    cell_declaration_sha256: str
    model_revision_manifest_sha256: str
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    physical_assignment_sha256: str
    assignment_sha256: str
    budget_materialization_authority_sha256: str
    experiment_budget_sha256: str
    output_manifest_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("download plan authority schema is unsupported")
        if type(self.source) is not BoundDownloadJson:
            raise TypeError("download plan authority requires exact raw JSON")
        for name in (
            "plan_sha256",
            "registry_sha256",
            "cell_id",
            "cell_declaration_sha256",
            "model_revision_manifest_sha256",
            "inventory_sha256",
            "inventory_source_receipt_sha256",
            "physical_assignment_sha256",
            "assignment_sha256",
            "budget_materialization_authority_sha256",
            "experiment_budget_sha256",
            "output_manifest_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.source.semantic_sha256 != self.plan_sha256:
            raise ValueError("download plan sidecar differs from semantic plan")

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": self.schema_version,
                "kind": "download_plan_raw_authority",
                "source": self.source.to_dict(),
                "plan_sha256": self.plan_sha256,
                "registry_sha256": self.registry_sha256,
                "cell_id": self.cell_id,
                "cell_declaration_sha256": self.cell_declaration_sha256,
                "model_revision_manifest_sha256": (self.model_revision_manifest_sha256),
                "inventory_sha256": self.inventory_sha256,
                "inventory_source_receipt_sha256": (
                    self.inventory_source_receipt_sha256
                ),
                "physical_assignment_sha256": self.physical_assignment_sha256,
                "assignment_sha256": self.assignment_sha256,
                "budget_materialization_authority_sha256": (
                    self.budget_materialization_authority_sha256
                ),
                "experiment_budget_sha256": self.experiment_budget_sha256,
                "output_manifest_sha256": self.output_manifest_sha256,
            }
        )


def bind_download_plan_authority(
    path: str | Path,
    *,
    expected_plan: DownloadPlan,
) -> DownloadPlanAuthority:
    if type(expected_plan) is not DownloadPlan:
        raise TypeError("plan authority requires an exact DownloadPlan")
    expected_plan.__post_init__()
    source, raw = BoundDownloadJson.bind(
        path,
        expected_path=expected_plan.plan_path,
        label="download plan",
    )
    plan = DownloadPlan.from_dict(raw)
    if plan != expected_plan or plan.sha256 != expected_plan.sha256:
        raise ValueError("raw download plan differs from release-derived inputs")
    if source.semantic_sha256 != plan.sha256:
        raise ValueError("download plan sidecar differs from semantic content")
    inputs = plan.inputs
    return DownloadPlanAuthority(
        schema_version=1,
        source=source,
        plan_sha256=plan.sha256,
        registry_sha256=inputs.registry_sha256,
        cell_id=inputs.cell_id,
        cell_declaration_sha256=inputs.cell_declaration_sha256,
        model_revision_manifest_sha256=inputs.model_revision_manifest_sha256,
        inventory_sha256=inputs.inventory_sha256,
        inventory_source_receipt_sha256=inputs.inventory_source_receipt_sha256,
        physical_assignment_sha256=inputs.physical_assignment_sha256,
        assignment_sha256=inputs.assignment_sha256,
        budget_materialization_authority_sha256=(
            inputs.budget_materialization_authority_sha256
        ),
        experiment_budget_sha256=inputs.experiment_budget_sha256,
        output_manifest_sha256=plan.output_manifest_sha256,
    )


def revalidate_download_plan_authority(
    authority: DownloadPlanAuthority,
    *,
    expected_plan: DownloadPlan,
) -> DownloadPlan:
    if type(authority) is not DownloadPlanAuthority:
        raise TypeError("download plan replay requires an exact authority")
    if type(expected_plan) is not DownloadPlan:
        raise TypeError("download plan replay requires an exact expected plan")
    authority.__post_init__()
    raw = authority.source.reopen(label="download plan")
    plan = DownloadPlan.from_dict(raw)
    inputs = plan.inputs
    if (
        plan != expected_plan
        or plan.sha256 != authority.plan_sha256
        or authority.source.path != plan.plan_path
        or authority.registry_sha256 != inputs.registry_sha256
        or authority.cell_id != inputs.cell_id
        or authority.cell_declaration_sha256 != inputs.cell_declaration_sha256
        or authority.model_revision_manifest_sha256
        != inputs.model_revision_manifest_sha256
        or authority.inventory_sha256 != inputs.inventory_sha256
        or authority.inventory_source_receipt_sha256
        != inputs.inventory_source_receipt_sha256
        or authority.physical_assignment_sha256 != inputs.physical_assignment_sha256
        or authority.assignment_sha256 != inputs.assignment_sha256
        or authority.budget_materialization_authority_sha256
        != inputs.budget_materialization_authority_sha256
        or authority.experiment_budget_sha256 != inputs.experiment_budget_sha256
        or authority.output_manifest_sha256 != plan.output_manifest_sha256
    ):
        raise ValueError("download plan authority differs from fresh raw replay")
    return plan


@dataclass(frozen=True)
class DownloadOutputArtifact:
    relative_path: str
    absolute_path: str
    size: int
    raw_sha256: str

    def __post_init__(self) -> None:
        _require_relative_path("download output relative path", self.relative_path)
        _require_absolute_normalized(
            "download output absolute path",
            self.absolute_path,
        )
        if type(self.size) is not int or self.size < 0:
            raise ValueError("download output size must be non-negative")
        _require_sha256("download output digest", self.raw_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "absolute_path": self.absolute_path,
            "size": self.size,
            "raw_sha256": self.raw_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "relative_path",
            "absolute_path",
            "size",
            "raw_sha256",
        }:
            raise ValueError("download output artifact fields differ from schema")
        return cls(
            relative_path=value["relative_path"],
            absolute_path=value["absolute_path"],
            size=value["size"],
            raw_sha256=value["raw_sha256"],
        )


@dataclass(frozen=True)
class DownloadTerminalReceipt:
    schema_version: int
    kind: str
    terminal_protocol_sha256: str
    plan_sha256: str
    plan_inputs_sha256: str
    registry_sha256: str
    cell_id: str
    cell_declaration_sha256: str
    model_revision_sha256s: tuple[str, ...]
    model_revision_manifest_sha256: str
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    physical_assignment_sha256: str
    assignment_sha256: str
    budget_materialization_authority_sha256: str
    experiment_budget_sha256: str
    started_monotonic_ns: int
    finished_monotonic_ns: int
    exit_code: int
    terminal_status: str
    headline_eligible: bool
    outputs: tuple[DownloadOutputArtifact, ...]
    output_manifest_sha256: str
    issuer_id: str
    issuer_version_sha256: str
    signature_hex: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "download_terminal_receipt"
        ):
            raise ValueError("download terminal schema is unsupported")
        if self.terminal_protocol_sha256 != DOWNLOAD_TERMINAL_PROTOCOL_SHA256:
            raise ValueError("download terminal uses another protocol")
        for name in (
            "plan_sha256",
            "plan_inputs_sha256",
            "registry_sha256",
            "cell_id",
            "cell_declaration_sha256",
            "model_revision_manifest_sha256",
            "inventory_sha256",
            "inventory_source_receipt_sha256",
            "physical_assignment_sha256",
            "assignment_sha256",
            "budget_materialization_authority_sha256",
            "experiment_budget_sha256",
            "output_manifest_sha256",
            "issuer_version_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            not self.model_revision_sha256s
            or len(self.model_revision_sha256s) != len(set(self.model_revision_sha256s))
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in self.model_revision_sha256s
            )
        ):
            raise ValueError("download terminal model revision coverage is invalid")
        if (
            type(self.started_monotonic_ns) is not int
            or self.started_monotonic_ns < 0
            or type(self.finished_monotonic_ns) is not int
            or self.finished_monotonic_ns < self.started_monotonic_ns
            or type(self.exit_code) is not int
            or self.exit_code != 0
            or self.terminal_status != "COMPLETE"
        ):
            raise ValueError("download terminal lifecycle is not successful")
        if self.headline_eligible is not False:
            raise ValueError("DOWNLOAD terminal cannot be headline evidence")
        if not self.outputs or any(
            type(value) is not DownloadOutputArtifact for value in self.outputs
        ):
            raise TypeError("download terminal requires exact output artifacts")
        output_paths = tuple(value.relative_path for value in self.outputs)
        if output_paths != tuple(sorted(set(output_paths))):
            raise ValueError("download terminal outputs are not canonical")
        expected_manifest = content_sha256([value.to_dict() for value in self.outputs])
        if self.output_manifest_sha256 != expected_manifest:
            raise ValueError("download terminal output manifest differs")
        _require_safe_id("download terminal issuer", self.issuer_id)
        if (
            not isinstance(self.signature_hex, str)
            or len(self.signature_hex) != 128
            or any(value not in "0123456789abcdef" for value in self.signature_hex)
        ):
            raise ValueError("download terminal signature is malformed")

    def validate_against(self, plan: DownloadPlan) -> None:
        if type(plan) is not DownloadPlan:
            raise TypeError("terminal validation requires an exact DownloadPlan")
        plan.__post_init__()
        inputs = plan.inputs
        if (
            self.plan_sha256 != plan.sha256
            or self.plan_inputs_sha256 != inputs.sha256
            or self.registry_sha256 != inputs.registry_sha256
            or self.cell_id != inputs.cell_id
            or self.cell_declaration_sha256 != inputs.cell_declaration_sha256
            or self.model_revision_sha256s != inputs.model_revision_sha256s
            or self.model_revision_manifest_sha256
            != inputs.model_revision_manifest_sha256
            or self.inventory_sha256 != inputs.inventory_sha256
            or self.inventory_source_receipt_sha256
            != inputs.inventory_source_receipt_sha256
            or self.physical_assignment_sha256 != inputs.physical_assignment_sha256
            or self.assignment_sha256 != inputs.assignment_sha256
            or self.budget_materialization_authority_sha256
            != inputs.budget_materialization_authority_sha256
            or self.experiment_budget_sha256 != inputs.experiment_budget_sha256
        ):
            raise ValueError("download terminal differs from its exact plan inputs")
        expected = {
            value.relative_path: (value.size, value.sha256)
            for value in plan.expected_outputs
        }
        actual = {
            value.relative_path: (value.size, value.raw_sha256)
            for value in self.outputs
        }
        if actual != expected:
            raise ValueError("download terminal output hashes differ from plan")
        root = PurePath(inputs.cache_root)
        if any(
            PurePath(value.absolute_path) != root / value.relative_path
            for value in self.outputs
        ):
            raise ValueError("download terminal output paths differ from cache root")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "terminal_protocol_sha256": self.terminal_protocol_sha256,
            "plan_sha256": self.plan_sha256,
            "plan_inputs_sha256": self.plan_inputs_sha256,
            "registry_sha256": self.registry_sha256,
            "cell_id": self.cell_id,
            "cell_declaration_sha256": self.cell_declaration_sha256,
            "model_revision_sha256s": list(self.model_revision_sha256s),
            "model_revision_manifest_sha256": (self.model_revision_manifest_sha256),
            "inventory_sha256": self.inventory_sha256,
            "inventory_source_receipt_sha256": (self.inventory_source_receipt_sha256),
            "physical_assignment_sha256": self.physical_assignment_sha256,
            "assignment_sha256": self.assignment_sha256,
            "budget_materialization_authority_sha256": (
                self.budget_materialization_authority_sha256
            ),
            "experiment_budget_sha256": self.experiment_budget_sha256,
            "started_monotonic_ns": self.started_monotonic_ns,
            "finished_monotonic_ns": self.finished_monotonic_ns,
            "exit_code": self.exit_code,
            "terminal_status": self.terminal_status,
            "headline_eligible": self.headline_eligible,
            "outputs": [value.to_dict() for value in self.outputs],
            "output_manifest_sha256": self.output_manifest_sha256,
            "issuer_id": self.issuer_id,
            "issuer_version_sha256": self.issuer_version_sha256,
            "signature_hex": self.signature_hex,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {
            "schema_version",
            "kind",
            "terminal_protocol_sha256",
            "plan_sha256",
            "plan_inputs_sha256",
            "registry_sha256",
            "cell_id",
            "cell_declaration_sha256",
            "model_revision_sha256s",
            "model_revision_manifest_sha256",
            "inventory_sha256",
            "inventory_source_receipt_sha256",
            "physical_assignment_sha256",
            "assignment_sha256",
            "budget_materialization_authority_sha256",
            "experiment_budget_sha256",
            "started_monotonic_ns",
            "finished_monotonic_ns",
            "exit_code",
            "terminal_status",
            "headline_eligible",
            "outputs",
            "output_manifest_sha256",
            "issuer_id",
            "issuer_version_sha256",
            "signature_hex",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("download terminal fields differ from schema")
        revisions = value["model_revision_sha256s"]
        outputs = value["outputs"]
        if type(revisions) is not list or type(outputs) is not list:
            raise TypeError("download terminal arrays are malformed")
        payload = dict(value)
        payload["model_revision_sha256s"] = tuple(revisions)
        payload["outputs"] = tuple(
            DownloadOutputArtifact.from_dict(row) for row in outputs
        )
        return cls(**payload)

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class DownloadResultPointer:
    schema_version: int
    kind: str
    protocol_sha256: str
    plan_sha256: str
    plan_authority_sha256: str
    terminal_path: str
    terminal_size: int
    terminal_raw_sha256: str
    terminal_semantic_sha256: str
    outputs: tuple[DownloadOutputArtifact, ...]
    output_manifest_sha256: str
    publisher_id: str
    publisher_version_sha256: str
    signature_hex: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "download_result_pointer"
        ):
            raise ValueError("download result pointer schema is unsupported")
        if self.protocol_sha256 != DOWNLOAD_RESULT_POINTER_PROTOCOL_SHA256:
            raise ValueError("download result pointer uses another protocol")
        for name in (
            "plan_sha256",
            "plan_authority_sha256",
            "terminal_raw_sha256",
            "terminal_semantic_sha256",
            "output_manifest_sha256",
            "publisher_version_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_absolute_normalized(
            "download result-pointer terminal",
            self.terminal_path,
        )
        if type(self.terminal_size) is not int or self.terminal_size < 1:
            raise ValueError("download result-pointer terminal size must be positive")
        if not self.outputs or any(
            type(value) is not DownloadOutputArtifact for value in self.outputs
        ):
            raise TypeError("download result pointer requires exact outputs")
        output_paths = tuple(value.relative_path for value in self.outputs)
        if output_paths != tuple(sorted(set(output_paths))):
            raise ValueError("download result-pointer outputs are not canonical")
        if self.output_manifest_sha256 != content_sha256(
            [value.to_dict() for value in self.outputs]
        ):
            raise ValueError("download result-pointer output manifest differs")
        _require_safe_id("download result-pointer publisher", self.publisher_id)
        if (
            not isinstance(self.signature_hex, str)
            or len(self.signature_hex) != 128
            or any(value not in "0123456789abcdef" for value in self.signature_hex)
        ):
            raise ValueError("download result-pointer signature is malformed")

    def validate_against(
        self,
        *,
        plan: DownloadPlan,
        plan_authority: DownloadPlanAuthority,
        terminal: DownloadTerminalReceipt,
        terminal_binding: BoundDownloadJson,
    ) -> None:
        if (
            self.plan_sha256 != plan.sha256
            or self.plan_authority_sha256 != plan_authority.sha256
            or self.terminal_path != plan.terminal_receipt_path
            or self.terminal_path != terminal_binding.path
            or self.terminal_size != terminal_binding.size
            or self.terminal_raw_sha256 != terminal_binding.raw_sha256
            or self.terminal_semantic_sha256 != terminal.sha256
            or self.terminal_semantic_sha256 != terminal_binding.semantic_sha256
            or self.outputs != terminal.outputs
            or self.output_manifest_sha256 != terminal.output_manifest_sha256
        ):
            raise ValueError(
                "download result pointer differs from plan/terminal raw authority"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "plan_sha256": self.plan_sha256,
            "plan_authority_sha256": self.plan_authority_sha256,
            "terminal_path": self.terminal_path,
            "terminal_size": self.terminal_size,
            "terminal_raw_sha256": self.terminal_raw_sha256,
            "terminal_semantic_sha256": self.terminal_semantic_sha256,
            "outputs": [value.to_dict() for value in self.outputs],
            "output_manifest_sha256": self.output_manifest_sha256,
            "publisher_id": self.publisher_id,
            "publisher_version_sha256": self.publisher_version_sha256,
            "signature_hex": self.signature_hex,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "plan_sha256",
            "plan_authority_sha256",
            "terminal_path",
            "terminal_size",
            "terminal_raw_sha256",
            "terminal_semantic_sha256",
            "outputs",
            "output_manifest_sha256",
            "publisher_id",
            "publisher_version_sha256",
            "signature_hex",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("download result-pointer fields differ from schema")
        outputs = value["outputs"]
        if type(outputs) is not list:
            raise TypeError("download result-pointer outputs must be an array")
        payload = dict(value)
        payload["outputs"] = tuple(
            DownloadOutputArtifact.from_dict(row) for row in outputs
        )
        return cls(**payload)

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class BoundDownloadOutput:
    artifact: DownloadOutputArtifact
    size: int
    raw_sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    def __post_init__(self) -> None:
        if type(self.artifact) is not DownloadOutputArtifact:
            raise TypeError("bound download output requires an exact artifact")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("bound download output size must be non-negative")
        _require_sha256("bound download output digest", self.raw_sha256)
        for name in ("device", "inode", "mtime_ns", "ctime_ns"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"bound download output {name} must be non-negative")
        if (
            self.size != self.artifact.size
            or self.raw_sha256 != self.artifact.raw_sha256
        ):
            raise ValueError("bound download output differs from terminal artifact")

    @classmethod
    def bind(
        cls,
        artifact: DownloadOutputArtifact,
        *,
        cache_root: str,
    ) -> Self:
        if type(artifact) is not DownloadOutputArtifact:
            raise TypeError("output binding requires an exact DownloadOutputArtifact")
        root = Path(_require_absolute_normalized("download cache root", cache_root))
        path = Path(artifact.absolute_path)
        if root.is_symlink() or path.is_symlink():
            raise ValueError("download output and cache root must be non-symlink")
        try:
            resolved_root = root.resolve(strict=True)
            resolved_path = path.resolve(strict=True)
        except OSError as error:
            raise ValueError("download output or cache root is unavailable") from error
        if (
            resolved_root != root
            or resolved_path != path
            or path != root / artifact.relative_path
            or not path.is_relative_to(root)
        ):
            raise ValueError("download output escaped its registered cache root")
        digest = _stable_file_digest(path, label="download output")
        if digest.size != artifact.size or digest.sha256 != artifact.raw_sha256:
            raise ValueError("download output differs from terminal hash")
        return cls(
            artifact=artifact,
            size=digest.size,
            raw_sha256=digest.sha256,
            device=digest.device,
            inode=digest.inode,
            mtime_ns=digest.mtime_ns,
            ctime_ns=digest.ctime_ns,
        )

    def reopen(self, *, cache_root: str) -> None:
        rebound = self.bind(self.artifact, cache_root=cache_root)
        if rebound != self:
            raise ValueError("download output differs from its fresh raw replay")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact": self.artifact.to_dict(),
            "size": self.size,
            "raw_sha256": self.raw_sha256,
            "device": self.device,
            "inode": self.inode,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class FutureDownloadRawAuthority:
    schema_version: int
    protocol_sha256: str
    plan_authority: DownloadPlanAuthority
    terminal: BoundDownloadJson
    result_pointer: BoundDownloadJson
    outputs: tuple[BoundDownloadOutput, ...]
    terminal_sha256: str
    result_pointer_sha256: str
    formal_status: Literal["BLOCKED"]
    reason_code: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("future download raw authority schema is unsupported")
        if self.protocol_sha256 != FUTURE_DOWNLOAD_RAW_AUTHORITY_PROTOCOL_SHA256:
            raise ValueError("future download raw authority uses another protocol")
        if type(self.plan_authority) is not DownloadPlanAuthority:
            raise TypeError("future download raw authority requires a plan authority")
        if (
            type(self.terminal) is not BoundDownloadJson
            or type(self.result_pointer) is not BoundDownloadJson
        ):
            raise TypeError("future download raw authority requires raw JSON bindings")
        if not self.outputs or any(
            type(value) is not BoundDownloadOutput for value in self.outputs
        ):
            raise TypeError("future download raw authority requires bound outputs")
        paths = tuple(value.artifact.relative_path for value in self.outputs)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("future download output coverage is not canonical")
        _require_sha256("future terminal digest", self.terminal_sha256)
        _require_sha256("future result-pointer digest", self.result_pointer_sha256)
        if (
            self.terminal.semantic_sha256 != self.terminal_sha256
            or self.result_pointer.semantic_sha256 != self.result_pointer_sha256
        ):
            raise ValueError("future download semantic sidecars differ")
        if (
            self.formal_status != "BLOCKED"
            or self.reason_code != RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON
        ):
            raise ValueError("current release cannot mint DOWNLOAD completion")

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": self.schema_version,
                "kind": "future_download_raw_authority",
                "protocol_sha256": self.protocol_sha256,
                "plan_authority_sha256": self.plan_authority.sha256,
                "terminal": self.terminal.to_dict(),
                "result_pointer": self.result_pointer.to_dict(),
                "outputs": [value.to_dict() for value in self.outputs],
                "terminal_sha256": self.terminal_sha256,
                "result_pointer_sha256": self.result_pointer_sha256,
                "formal_status": self.formal_status,
                "reason_code": self.reason_code,
            }
        )


def bind_future_download_raw_authority(
    plan_authority: DownloadPlanAuthority,
    *,
    expected_plan: DownloadPlan,
) -> FutureDownloadRawAuthority:
    """Bind future-format bytes but retain a named formal BLOCKED status."""

    plan = revalidate_download_plan_authority(
        plan_authority,
        expected_plan=expected_plan,
    )
    terminal_binding, terminal_raw = BoundDownloadJson.bind(
        plan.terminal_receipt_path,
        expected_path=plan.terminal_receipt_path,
        label="download terminal receipt",
    )
    terminal = DownloadTerminalReceipt.from_dict(terminal_raw)
    terminal.validate_against(plan)
    if terminal_binding.semantic_sha256 != terminal.sha256:
        raise ValueError("download terminal sidecar differs from semantic content")
    pointer_binding, pointer_raw = BoundDownloadJson.bind(
        plan.result_pointer_path,
        expected_path=plan.result_pointer_path,
        label="download result pointer",
    )
    pointer = DownloadResultPointer.from_dict(pointer_raw)
    pointer.validate_against(
        plan=plan,
        plan_authority=plan_authority,
        terminal=terminal,
        terminal_binding=terminal_binding,
    )
    if pointer_binding.semantic_sha256 != pointer.sha256:
        raise ValueError("download result-pointer sidecar differs from content")
    outputs = tuple(
        BoundDownloadOutput.bind(value, cache_root=plan.inputs.cache_root)
        for value in terminal.outputs
    )
    return FutureDownloadRawAuthority(
        schema_version=1,
        protocol_sha256=FUTURE_DOWNLOAD_RAW_AUTHORITY_PROTOCOL_SHA256,
        plan_authority=plan_authority,
        terminal=terminal_binding,
        result_pointer=pointer_binding,
        outputs=outputs,
        terminal_sha256=terminal.sha256,
        result_pointer_sha256=pointer.sha256,
        formal_status="BLOCKED",
        reason_code=RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON,
    )


def revalidate_future_download_raw_authority(
    authority: FutureDownloadRawAuthority,
    *,
    expected_plan: DownloadPlan,
) -> FutureDownloadRawAuthority:
    if type(authority) is not FutureDownloadRawAuthority:
        raise TypeError("future download replay requires an exact authority")
    authority.__post_init__()
    plan = revalidate_download_plan_authority(
        authority.plan_authority,
        expected_plan=expected_plan,
    )
    terminal_raw = authority.terminal.reopen(label="download terminal receipt")
    terminal = DownloadTerminalReceipt.from_dict(terminal_raw)
    terminal.validate_against(plan)
    pointer_raw = authority.result_pointer.reopen(label="download result pointer")
    pointer = DownloadResultPointer.from_dict(pointer_raw)
    pointer.validate_against(
        plan=plan,
        plan_authority=authority.plan_authority,
        terminal=terminal,
        terminal_binding=authority.terminal,
    )
    for output in authority.outputs:
        output.reopen(cache_root=plan.inputs.cache_root)
    if (
        terminal.sha256 != authority.terminal_sha256
        or pointer.sha256 != authority.result_pointer_sha256
        or tuple(value.artifact for value in authority.outputs) != terminal.outputs
    ):
        raise ValueError("future download raw authority changed during replay")
    return authority


__all__ = [
    "DOWNLOAD_PLAN_PROTOCOL_SHA256",
    "DOWNLOAD_RESULT_POINTER_PROTOCOL_SHA256",
    "DOWNLOAD_TERMINAL_PROTOCOL_SHA256",
    "FUTURE_DOWNLOAD_RAW_AUTHORITY_PROTOCOL_SHA256",
    "RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON",
    "RELEASE_DOWNLOAD_POINTER_PUBLISHERS",
    "RELEASE_DOWNLOAD_TERMINAL_ISSUERS",
    "BoundDownloadJson",
    "BoundDownloadOutput",
    "DownloadExecutionBlocked",
    "DownloadExecutionInputs",
    "DownloadModelRevision",
    "DownloadOutputArtifact",
    "DownloadOutputExpectation",
    "DownloadPlan",
    "DownloadPlanAuthority",
    "DownloadResultPointer",
    "DownloadTerminalReceipt",
    "FutureDownloadRawAuthority",
    "bind_download_plan_authority",
    "bind_future_download_raw_authority",
    "issue_download_plan",
    "require_release_download_execution",
    "revalidate_download_plan_authority",
    "revalidate_future_download_raw_authority",
]
