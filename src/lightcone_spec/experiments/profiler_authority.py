"""Raw execution and terminal authority for isolated E4 profiler cells.

Profiler traces are diagnostic mechanism evidence.  They are never headline
timing evidence and may never contend with a headline assignment.  This module
binds the three registered PROFILE cells to exact Nsight command templates,
metric sets, an exclusive physical assignment, and an explicit profiler budget.
It also replays a terminal pointer and hashes every raw profile file onsite.

The current release tool allowlist is intentionally empty.  Therefore execution
is blocked before a token capable of launching ``nsys`` or ``ncu`` is returned.
Missing tools, terminal pointers, raw files, or metrics remain named BLOCKED
states with measurement values represented as ``None`` rather than invented
zeros.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

from lightcone_spec.experiments.gpu_pool import GpuAssignment
from lightcone_spec.experiments.planning import BudgetJobKind, ExperimentBudget
from lightcone_spec.experiments.registry import (
    ExperimentCell,
    ExperimentRegistry,
    WorkloadClass,
    content_sha256,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_PROFILE_BYTES = 128 * 1024 * 1024 * 1024

PROFILER_TOOL_UNAVAILABLE_REASON = "release_profiler_tool_allowlist_empty"
PROFILER_TOOL_VERSION_MISSING_REASON = "profiler_tool_version_missing"
PROFILER_TERMINAL_POINTER_MISSING_REASON = "profiler_terminal_pointer_missing"
PROFILER_RAW_PROFILE_MISSING_REASON = "profiler_raw_profile_missing"
PROFILER_REQUIRED_METRIC_MISSING_REASON = "profiler_required_metric_missing"

# Source-owned ``(tool, absolute binary, exact version, version SHA-256)`` rows.
# Caller-provided probes cannot populate this allowlist.  A future release must
# add reviewed rows together with GPU-marked command and output-parser tests.
RELEASE_PROFILER_TOOL_ALLOWLIST: tuple[tuple[str, str, str, str], ...] = ()

PROFILER_AUTHORITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e4_profiler_raw_authority_protocol",
        "cells": ["nvtx", "nsight_systems", "nsight_compute"],
        "isolation": "fresh_process_exclusive_host_exact_gpu_assignment",
        "budget": "explicit_profiler_duration",
        "raw_output": "path_bound_hash_replayed_onsite",
        "terminal": "content_bound_pointer_published_last",
        "headline_eligible": False,
        "missing_measurements": "BLOCKED_and_None_never_zero_filled",
    }
)

_NSYS_METRICS = (
    "cuda_api_duration_ns",
    "cuda_kernel_duration_ns",
    "cuda_memcpy_bytes",
    "gpu_kernel_gap_ns",
    "nvtx_range_duration_ns",
)
_NCU_METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
)


class ProfilerAuthorityBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"profiler authority is BLOCKED: {reason}")
        self.reason = reason


def _require_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_safe_id(label: str, value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    return value


def _strict_mapping(label: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _strict_sequence(label: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def _strict_keys(label: str, row: Mapping[str, object], expected: set[str]) -> None:
    if "headline" in row or "headline_goodput" in row:
        raise ValueError("profiler output cannot declare headline evidence")
    if set(row) != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - set(row))}, "
            f"extra={sorted(set(row) - expected)}"
        )


def _strict_bool(label: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    return value


def _resolved_regular_path(
    value: str | Path,
    *,
    label: str,
    maximum_bytes: int,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} path is missing") from error
    if resolved != path:
        raise ValueError(f"{label} path must be resolved and non-symlink")
    current = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_size <= 0
        or current.st_size > maximum_bytes
    ):
        raise ValueError(f"{label} must be a bounded non-empty regular file")
    return path


def _read_stable(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    path = _resolved_regular_path(path, label=label, maximum_bytes=maximum_bytes)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"{label} changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} grew while being read")
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or before.st_size > maximum_bytes
            or identity(before) != identity(after)
            or identity(after) != identity(current)
        ):
            raise ValueError(f"{label} changed during coordinated read")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _load_json(path: Path, *, label: str) -> tuple[Mapping[str, object], str]:
    raw = _read_stable(path, label=label, maximum_bytes=_MAX_JSON_BYTES)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return _strict_mapping(label, value), hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ProfilerToolContract:
    variant: Literal["nvtx", "nsight_systems", "nsight_compute"]
    tool: Literal["nsys", "ncu"]
    command_template: tuple[str, ...]
    required_metrics: tuple[str, ...]
    raw_profile_role: Literal["nsys_report", "ncu_report"]

    def __post_init__(self) -> None:
        if self.variant not in {"nvtx", "nsight_systems", "nsight_compute"}:
            raise ValueError("profiler variant is unsupported")
        if self.tool not in {"nsys", "ncu"}:
            raise ValueError("profiler tool is unsupported")
        if not self.command_template or "{subject_argv}" not in self.command_template:
            raise ValueError("profiler command template must bind the subject argv")
        if self.required_metrics != tuple(sorted(set(self.required_metrics))):
            raise ValueError("profiler metrics must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "tool": self.tool,
            "command_template": list(self.command_template),
            "required_metrics": list(self.required_metrics),
            "raw_profile_role": self.raw_profile_role,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def _tool_contract(variant: str) -> ProfilerToolContract:
    if variant == "nvtx":
        return ProfilerToolContract(
            variant="nvtx",
            tool="nsys",
            command_template=(
                "nsys",
                "profile",
                "--trace=cuda,nvtx",
                "--sample=none",
                "--cpuctxsw=none",
                "--force-overwrite=false",
                "--output={output_base}",
                "--",
                "{subject_argv}",
            ),
            required_metrics=tuple(sorted(_NSYS_METRICS)),
            raw_profile_role="nsys_report",
        )
    if variant == "nsight_systems":
        return ProfilerToolContract(
            variant="nsight_systems",
            tool="nsys",
            command_template=(
                "nsys",
                "profile",
                "--trace=cuda,nvtx,osrt,cublas,cudnn",
                "--sample=none",
                "--cpuctxsw=none",
                "--force-overwrite=false",
                "--output={output_base}",
                "--",
                "{subject_argv}",
            ),
            required_metrics=tuple(sorted(_NSYS_METRICS)),
            raw_profile_role="nsys_report",
        )
    if variant == "nsight_compute":
        return ProfilerToolContract(
            variant="nsight_compute",
            tool="ncu",
            command_template=(
                "ncu",
                "--target-processes=all",
                "--metrics=" + ",".join(sorted(_NCU_METRICS)),
                "--export={output_base}",
                "--force-overwrite=false",
                "--",
                "{subject_argv}",
            ),
            required_metrics=tuple(sorted(_NCU_METRICS)),
            raw_profile_role="ncu_report",
        )
    raise ValueError("profiler variant is not registered")


@dataclass(frozen=True)
class ProfilerExecutionPlan:
    schema_version: int
    kind: str
    registry_sha256: str
    cell_id: str
    cell_declaration_sha256: str
    assignment_sha256: str
    physical_gpu_uuids: tuple[str, ...]
    budget_sha256: str
    profiler_duration_ms: int
    isolated_process: bool
    exclusive_host: bool
    headline_eligible: bool
    tool_contract: ProfilerToolContract
    tool_path: str | None
    tool_version: str | None
    tool_version_sha256: str | None
    subject_plan_sha256: str
    protocol_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "e4_profiler_execution_plan":
            raise ValueError("profiler execution plan schema is unsupported")
        for label, value in (
            ("profiler registry", self.registry_sha256),
            ("profiler cell", self.cell_id),
            ("profiler declaration", self.cell_declaration_sha256),
            ("profiler assignment", self.assignment_sha256),
            ("profiler budget", self.budget_sha256),
            ("profiler subject plan", self.subject_plan_sha256),
            ("profiler protocol", self.protocol_sha256),
        ):
            _require_sha256(label, value)
        if self.profiler_duration_ms <= 0:
            raise ValueError("profiler duration must be explicitly positive")
        if len(self.physical_gpu_uuids) != 2 or len(set(self.physical_gpu_uuids)) != 2:
            raise ValueError("E4 profiler plan requires two exact physical GPUs")
        if self.isolated_process is not True or self.exclusive_host is not True:
            raise ValueError("profiler plan must use an isolated exclusive host")
        if self.headline_eligible is not False:
            raise ValueError("profiler plan cannot authorize headline evidence")
        if self.protocol_sha256 != PROFILER_AUTHORITY_PROTOCOL_SHA256:
            raise ValueError("profiler plan uses another release protocol")
        optional = (self.tool_path, self.tool_version, self.tool_version_sha256)
        if any(value is None for value in optional) != all(
            value is None for value in optional
        ):
            raise ValueError("profiler tool identity must be wholly present or absent")
        if self.tool_path is not None:
            if not Path(self.tool_path).is_absolute():
                raise ValueError("profiler tool path must be absolute")
            _require_safe_id("profiler tool version", self.tool_version)
            _require_sha256("profiler tool version", self.tool_version_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "registry_sha256": self.registry_sha256,
            "cell_id": self.cell_id,
            "cell_declaration_sha256": self.cell_declaration_sha256,
            "assignment_sha256": self.assignment_sha256,
            "physical_gpu_uuids": list(self.physical_gpu_uuids),
            "budget_sha256": self.budget_sha256,
            "profiler_duration_ms": self.profiler_duration_ms,
            "isolated_process": self.isolated_process,
            "exclusive_host": self.exclusive_host,
            "headline_eligible": self.headline_eligible,
            "tool_contract": self.tool_contract.to_dict(),
            "tool_path": self.tool_path,
            "tool_version": self.tool_version,
            "tool_version_sha256": self.tool_version_sha256,
            "subject_plan_sha256": self.subject_plan_sha256,
            "protocol_sha256": self.protocol_sha256,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def release_profiler_plan(
    registry: ExperimentRegistry,
    cell: ExperimentCell,
    assignment: GpuAssignment,
    budget: ExperimentBudget,
    *,
    subject_plan_sha256: str,
) -> ProfilerExecutionPlan:
    """Derive one exact plan from release registry, placement, and budget inputs."""

    if type(registry) is not ExperimentRegistry or type(cell) is not ExperimentCell:
        raise TypeError("profiler planning requires exact registry and cell objects")
    if type(assignment) is not GpuAssignment or type(budget) is not ExperimentBudget:
        raise TypeError(
            "profiler planning requires exact assignment and budget objects"
        )
    _require_sha256("profiler subject plan", subject_plan_sha256)
    matches = tuple(
        row for row in registry.cells_for("E4") if row.cell_id == cell.cell_id
    )
    if len(matches) != 1 or matches[0] != cell:
        raise ValueError("profiler cell is foreign to the registry")
    identity = cell.identity
    claim = assignment.work_item.claim
    if (
        cell.resources.workload_class is not WorkloadClass.PROFILE
        or identity.task != "isolated_profile"
        or identity.slo != "headline_evidence_forbidden"
        or identity.variant not in {"nvtx", "nsight_systems", "nsight_compute"}
        or not cell.resources.exclusive
        or cell.resources.gpu_count != 2
    ):
        raise ValueError("cell is not an exact isolated E4 PROFILE cell")
    if (
        assignment.work_item.cell != cell
        or not claim.exclusive_gpu
        or not claim.exclusive_host
        or not claim.same_host
        or claim.workload_class is not WorkloadClass.PROFILE
        or len(assignment.gpu_uuids) != 2
    ):
        raise ValueError("profiler assignment is not an exclusive exact cell placement")
    if (
        budget.cell_id != cell.cell_id
        or budget.experiment != "E4"
        or budget.method != cell.identity.method
        or budget.workload_class is not WorkloadClass.PROFILE
        or budget.job_kind is not BudgetJobKind.PROFILER
        or budget.gpu_count != 2
        or budget.topology != cell.identity.topology
        or budget.profiler.registered <= 0
        or budget.measured_gpu_ms is not None
    ):
        raise ValueError("profiler budget is missing or belongs to another assignment")
    contract = _tool_contract(identity.variant)
    tool_rows = tuple(
        row for row in RELEASE_PROFILER_TOOL_ALLOWLIST if row[0] == contract.tool
    )
    if not tool_rows:
        tool_path = tool_version = tool_version_sha256 = None
    elif len(tool_rows) == 1:
        _, tool_path, tool_version, tool_version_sha256 = tool_rows[0]
        _require_sha256("release profiler tool version", tool_version_sha256)
    else:
        raise RuntimeError("release profiler tool allowlist is ambiguous")
    return ProfilerExecutionPlan(
        schema_version=1,
        kind="e4_profiler_execution_plan",
        registry_sha256=registry.sha256,
        cell_id=cell.cell_id,
        cell_declaration_sha256=cell.sha256,
        assignment_sha256=assignment.sha256,
        physical_gpu_uuids=assignment.gpu_uuids,
        budget_sha256=budget.sha256,
        profiler_duration_ms=budget.profiler.registered,
        isolated_process=True,
        exclusive_host=True,
        headline_eligible=False,
        tool_contract=contract,
        tool_path=tool_path,
        tool_version=tool_version,
        tool_version_sha256=tool_version_sha256,
        subject_plan_sha256=subject_plan_sha256,
        protocol_sha256=PROFILER_AUTHORITY_PROTOCOL_SHA256,
    )


def _contract_from_dict(value: object) -> ProfilerToolContract:
    row = _strict_mapping("profiler tool contract", value)
    _strict_keys(
        "profiler tool contract",
        row,
        {"variant", "tool", "command_template", "required_metrics", "raw_profile_role"},
    )
    return ProfilerToolContract(
        variant=row["variant"],  # type: ignore[arg-type]
        tool=row["tool"],  # type: ignore[arg-type]
        command_template=tuple(
            str(item)
            for item in _strict_sequence("profiler command", row["command_template"])
        ),
        required_metrics=tuple(
            str(item)
            for item in _strict_sequence("profiler metrics", row["required_metrics"])
        ),
        raw_profile_role=row["raw_profile_role"],  # type: ignore[arg-type]
    )


def _plan_from_dict(value: object) -> ProfilerExecutionPlan:
    row = _strict_mapping("profiler execution plan", value)
    expected = set(ProfilerExecutionPlan.__dataclass_fields__)
    _strict_keys("profiler execution plan", row, expected)
    gpu_values = _strict_sequence("profiler GPU UUIDs", row["physical_gpu_uuids"])
    duration = row["profiler_duration_ms"]
    if isinstance(duration, bool) or not isinstance(duration, int):
        raise TypeError("profiler duration must be an integer")
    return ProfilerExecutionPlan(
        schema_version=row["schema_version"],  # type: ignore[arg-type]
        kind=row["kind"],  # type: ignore[arg-type]
        registry_sha256=row["registry_sha256"],  # type: ignore[arg-type]
        cell_id=row["cell_id"],  # type: ignore[arg-type]
        cell_declaration_sha256=row["cell_declaration_sha256"],  # type: ignore[arg-type]
        assignment_sha256=row["assignment_sha256"],  # type: ignore[arg-type]
        physical_gpu_uuids=tuple(str(item) for item in gpu_values),
        budget_sha256=row["budget_sha256"],  # type: ignore[arg-type]
        profiler_duration_ms=duration,
        isolated_process=_strict_bool("isolated process", row["isolated_process"]),
        exclusive_host=_strict_bool("exclusive host", row["exclusive_host"]),
        headline_eligible=_strict_bool("headline eligible", row["headline_eligible"]),
        tool_contract=_contract_from_dict(row["tool_contract"]),
        tool_path=row["tool_path"],  # type: ignore[arg-type]
        tool_version=row["tool_version"],  # type: ignore[arg-type]
        tool_version_sha256=row["tool_version_sha256"],  # type: ignore[arg-type]
        subject_plan_sha256=row["subject_plan_sha256"],  # type: ignore[arg-type]
        protocol_sha256=row["protocol_sha256"],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class ProfilerPlanBinding:
    plan_path: str
    raw_sha256: str
    plan_sha256: str
    cell_id: str
    assignment_sha256: str
    budget_sha256: str
    subject_plan_sha256: str

    def __post_init__(self) -> None:
        _resolved_regular_path(
            self.plan_path, label="profiler plan", maximum_bytes=_MAX_JSON_BYTES
        )
        for label, value in (
            ("profiler raw plan", self.raw_sha256),
            ("profiler plan", self.plan_sha256),
            ("profiler cell", self.cell_id),
            ("profiler assignment", self.assignment_sha256),
            ("profiler budget", self.budget_sha256),
            ("profiler subject", self.subject_plan_sha256),
        ):
            _require_sha256(label, value)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "profiler_plan_binding",
                **self.__dict__,
            }
        )


@dataclass(frozen=True)
class ProfilerPlanAuthorityResult:
    binding: ProfilerPlanBinding
    plan: ProfilerExecutionPlan
    status: Literal["READY", "BLOCKED"]
    reason: str | None
    tool_version: str | None
    tool_version_sha256: str | None

    def __post_init__(self) -> None:
        if self.status == "BLOCKED" and (
            not self.reason
            or self.tool_version is not None
            or self.tool_version_sha256 is not None
        ):
            raise ValueError("blocked profiler tool values must remain None")


def bind_profiler_plan_authority(
    plan_path: str | Path,
    *,
    registry: ExperimentRegistry,
    cell: ExperimentCell,
    assignment: GpuAssignment,
    budget: ExperimentBudget,
    subject_plan_sha256: str,
) -> ProfilerPlanBinding:
    path = _resolved_regular_path(
        plan_path, label="profiler plan", maximum_bytes=_MAX_JSON_BYTES
    )
    row, raw_sha256 = _load_json(path, label="profiler plan")
    plan = _plan_from_dict(row)
    expected = release_profiler_plan(
        registry,
        cell,
        assignment,
        budget,
        subject_plan_sha256=subject_plan_sha256,
    )
    if plan != expected:
        raise ValueError("serialized profiler plan differs from release derivation")
    return ProfilerPlanBinding(
        plan_path=str(path),
        raw_sha256=raw_sha256,
        plan_sha256=plan.sha256,
        cell_id=plan.cell_id,
        assignment_sha256=plan.assignment_sha256,
        budget_sha256=plan.budget_sha256,
        subject_plan_sha256=plan.subject_plan_sha256,
    )


def revalidate_profiler_plan_authority(
    binding: ProfilerPlanBinding,
    *,
    registry: ExperimentRegistry,
    cell: ExperimentCell,
    assignment: GpuAssignment,
    budget: ExperimentBudget,
) -> ProfilerPlanAuthorityResult:
    if type(binding) is not ProfilerPlanBinding:
        raise TypeError("profiler replay requires an exact plan binding")
    expected = bind_profiler_plan_authority(
        binding.plan_path,
        registry=registry,
        cell=cell,
        assignment=assignment,
        budget=budget,
        subject_plan_sha256=binding.subject_plan_sha256,
    )
    if expected != binding:
        raise ValueError("profiler plan binding differs from fresh raw replay")
    row, _ = _load_json(Path(binding.plan_path), label="profiler plan")
    plan = _plan_from_dict(row)
    if plan.tool_version is None:
        return ProfilerPlanAuthorityResult(
            binding=binding,
            plan=plan,
            status="BLOCKED",
            reason=PROFILER_TOOL_UNAVAILABLE_REASON,
            tool_version=None,
            tool_version_sha256=None,
        )
    return ProfilerPlanAuthorityResult(
        binding=binding,
        plan=plan,
        status="READY",
        reason=None,
        tool_version=plan.tool_version,
        tool_version_sha256=plan.tool_version_sha256,
    )


@dataclass(frozen=True)
class ProfilerExecutionToken:
    authority_sha256: str
    plan_sha256: str
    tool_path: str
    tool_version_sha256: str
    command_template: tuple[str, ...]


def require_profiler_execution_authority(
    binding: ProfilerPlanBinding,
    *,
    registry: ExperimentRegistry,
    cell: ExperimentCell,
    assignment: GpuAssignment,
    budget: ExperimentBudget,
) -> ProfilerExecutionToken:
    result = revalidate_profiler_plan_authority(
        binding,
        registry=registry,
        cell=cell,
        assignment=assignment,
        budget=budget,
    )
    if result.status != "READY":
        raise ProfilerAuthorityBlocked(
            result.reason or PROFILER_TOOL_UNAVAILABLE_REASON
        )
    plan = result.plan
    if (
        plan.tool_path is None
        or plan.tool_version is None
        or plan.tool_version_sha256 is None
    ):
        raise ProfilerAuthorityBlocked(PROFILER_TOOL_VERSION_MISSING_REASON)
    return ProfilerExecutionToken(
        authority_sha256=binding.sha256,
        plan_sha256=plan.sha256,
        tool_path=plan.tool_path,
        tool_version_sha256=plan.tool_version_sha256,
        command_template=plan.tool_contract.command_template,
    )


@dataclass(frozen=True)
class ProfilerTerminalResult:
    status: Literal["READY", "BLOCKED"]
    reason: str | None
    plan_sha256: str
    terminal_pointer_sha256: str | None
    raw_profile_sha256s: tuple[str, ...] | None
    metrics: tuple[tuple[str, float | None], ...]
    headline_eligible: bool

    def __post_init__(self) -> None:
        _require_sha256("profiler terminal plan", self.plan_sha256)
        names = tuple(name for name, _ in self.metrics)
        if names != tuple(sorted(set(names))):
            raise ValueError("profiler terminal metrics must be sorted and unique")
        if self.headline_eligible is not False:
            raise ValueError("profiler terminal cannot authorize headline evidence")
        if self.status == "BLOCKED":
            if (
                not self.reason
                or any(value is not None for _, value in self.metrics)
                or self.raw_profile_sha256s is not None
            ):
                raise ValueError("blocked profiler measurements must remain None")
        elif self.status == "READY":
            if (
                self.reason is not None
                or self.terminal_pointer_sha256 is None
                or not self.raw_profile_sha256s
                or any(value is None for _, value in self.metrics)
            ):
                raise ValueError("ready profiler terminal result is incomplete")
            _require_sha256("profiler terminal pointer", self.terminal_pointer_sha256)
            for value in self.raw_profile_sha256s:
                _require_sha256("raw profiler output", value)
        else:
            raise ValueError("profiler terminal result is inconsistent")


def _blocked_terminal(
    plan: ProfilerExecutionPlan,
    reason: str,
) -> ProfilerTerminalResult:
    return ProfilerTerminalResult(
        status="BLOCKED",
        reason=reason,
        plan_sha256=plan.sha256,
        terminal_pointer_sha256=None,
        raw_profile_sha256s=None,
        metrics=tuple((name, None) for name in plan.tool_contract.required_metrics),
        headline_eligible=False,
    )


def reduce_profiler_terminal_authority(
    binding: ProfilerPlanBinding,
    terminal_pointer_path: str | Path | None,
    *,
    registry: ExperimentRegistry,
    cell: ExperimentCell,
    assignment: GpuAssignment,
    budget: ExperimentBudget,
) -> ProfilerTerminalResult:
    plan_result = revalidate_profiler_plan_authority(
        binding,
        registry=registry,
        cell=cell,
        assignment=assignment,
        budget=budget,
    )
    plan = plan_result.plan
    if terminal_pointer_path is None:
        return _blocked_terminal(plan, PROFILER_TERMINAL_POINTER_MISSING_REASON)
    pointer_path = _resolved_regular_path(
        terminal_pointer_path,
        label="profiler terminal pointer",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    row, pointer_raw_sha256 = _load_json(
        pointer_path, label="profiler terminal pointer"
    )
    _strict_keys(
        "profiler terminal pointer",
        row,
        {
            "schema_version",
            "kind",
            "plan_sha256",
            "authority_sha256",
            "assignment_sha256",
            "budget_sha256",
            "tool_contract_sha256",
            "tool_version_sha256",
            "process_id",
            "process_start_ns",
            "physical_gpu_uuids",
            "concurrent_headline_processes",
            "terminal_status",
            "isolated_process",
            "headline_eligible",
            "raw_profiles",
            "metrics",
        },
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != "e4_profiler_terminal_pointer"
        or row["plan_sha256"] != plan.sha256
        or row["authority_sha256"] != binding.sha256
        or row["assignment_sha256"] != assignment.sha256
        or row["budget_sha256"] != budget.sha256
        or row["tool_contract_sha256"] != plan.tool_contract.sha256
        or row["tool_version_sha256"] != plan.tool_version_sha256
        or isinstance(row["process_id"], bool)
        or not isinstance(row["process_id"], int)
        or row["process_id"] <= 0
        or isinstance(row["process_start_ns"], bool)
        or not isinstance(row["process_start_ns"], int)
        or row["process_start_ns"] <= 0
        or tuple(
            str(item)
            for item in _strict_sequence(
                "terminal physical GPU UUIDs", row["physical_gpu_uuids"]
            )
        )
        != plan.physical_gpu_uuids
        or row["concurrent_headline_processes"] != 0
        or row["terminal_status"] != "COMPLETE"
        or _strict_bool("terminal isolation", row["isolated_process"]) is not True
        or _strict_bool("terminal headline", row["headline_eligible"]) is not False
    ):
        raise ValueError("profiler terminal pointer differs from its exact plan")
    raw_profiles = _strict_sequence("raw profiler outputs", row["raw_profiles"])
    if len(raw_profiles) != 1:
        return _blocked_terminal(plan, PROFILER_RAW_PROFILE_MISSING_REASON)
    profile_row = _strict_mapping("raw profiler output", raw_profiles[0])
    _strict_keys("raw profiler output", profile_row, {"role", "path", "sha256"})
    if profile_row["role"] != plan.tool_contract.raw_profile_role:
        raise ValueError("raw profiler output role differs from the tool contract")
    profile_path_value = profile_row["path"]
    if not isinstance(profile_path_value, str):
        raise TypeError("raw profiler output path must be text")
    try:
        profile_raw = _read_stable(
            Path(profile_path_value),
            label="raw profiler output",
            maximum_bytes=_MAX_PROFILE_BYTES,
        )
    except ValueError:
        return _blocked_terminal(plan, PROFILER_RAW_PROFILE_MISSING_REASON)
    observed_profile_sha256 = hashlib.sha256(profile_raw).hexdigest()
    if profile_row["sha256"] != observed_profile_sha256:
        raise ValueError("raw profiler output hash differs from onsite bytes")
    metrics_row = _strict_mapping("profiler metrics", row["metrics"])
    if set(metrics_row) != set(plan.tool_contract.required_metrics):
        return _blocked_terminal(plan, PROFILER_REQUIRED_METRIC_MISSING_REASON)
    metrics: list[tuple[str, float]] = []
    for name in plan.tool_contract.required_metrics:
        value = metrics_row[name]
        if value is None:
            return _blocked_terminal(plan, PROFILER_REQUIRED_METRIC_MISSING_REASON)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("profiler metric values must be finite numbers or null")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("profiler metric values must be finite and non-negative")
        metrics.append((name, number))
    if plan_result.status != "READY":
        return _blocked_terminal(
            plan, plan_result.reason or PROFILER_TOOL_UNAVAILABLE_REASON
        )
    return ProfilerTerminalResult(
        status="READY",
        reason=None,
        plan_sha256=plan.sha256,
        terminal_pointer_sha256=pointer_raw_sha256,
        raw_profile_sha256s=(observed_profile_sha256,),
        metrics=tuple(metrics),
        headline_eligible=False,
    )


__all__ = [
    "PROFILER_AUTHORITY_PROTOCOL_SHA256",
    "PROFILER_RAW_PROFILE_MISSING_REASON",
    "PROFILER_REQUIRED_METRIC_MISSING_REASON",
    "PROFILER_TERMINAL_POINTER_MISSING_REASON",
    "PROFILER_TOOL_UNAVAILABLE_REASON",
    "PROFILER_TOOL_VERSION_MISSING_REASON",
    "ProfilerAuthorityBlocked",
    "ProfilerExecutionPlan",
    "ProfilerExecutionToken",
    "ProfilerPlanAuthorityResult",
    "ProfilerPlanBinding",
    "ProfilerTerminalResult",
    "ProfilerToolContract",
    "bind_profiler_plan_authority",
    "reduce_profiler_terminal_authority",
    "release_profiler_plan",
    "require_profiler_execution_authority",
    "revalidate_profiler_plan_authority",
]
