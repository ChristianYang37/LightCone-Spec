"""Durable, release-verifiable authority for completed industrial cells.

The scheduler must never accept a copied list of cell SHA-256 values as proof
that work completed.  This module keeps the raw schema-v4 completion artifact
and every release input needed to replay it.  A derivation reopens the files,
revalidates the final -> observation -> prepared evidence chain, verifies the
native terminal signature, and only then returns cell identities.

The current source release deliberately has no trusted attester configured.
That is a valid BLOCKED state: a content-consistent CPU fixture cannot become
execution authority merely by labelling its rows ``MEASURED``.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lightcone_spec.experiments.evidence import evidence_files_sha256
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.planning import (
    BudgetActivationAuthorityBinding,
    ConfirmationAuxiliaryActivationAuthorityBinding,
    ConfirmationFamilyPowerReductionArtifact,
    ConfirmationStageAggregateAuthorityBinding,
    DispositionStatus,
    E2ActivationAuthorityBinding,
    FamilyActivationArtifact,
    ReducerActivationArtifact,
    _is_budget_activation_authority_binding,
    materialize_confirmation_prefix,
    verify_confirmation_pilot_activation,
)
from lightcone_spec.experiments.planning_artifacts import (
    experiment_budget_from_dict,
)
from lightcone_spec.experiments.registry import (
    ExperimentReceipt,
    ExperimentRegistry,
    WorkloadClass,
    content_sha256,
)
from lightcone_spec.experiments.stage_activation import (
    RegistryStageActivationArtifact,
    RegistryStageDispositionStatus,
    verify_registry_stage_activation,
)
from lightcone_spec.orchestration.native_terminal import (
    PINNED_SGLANG_TREE,
    validate_native_terminal_artifact,
)
from lightcone_spec.runtime.attestation import (
    RELEASE_TRUSTED_ATTESTER_POLICY,
    TrustedAttesterPolicy,
    require_release_trusted_attester_policy,
)
from lightcone_spec.telemetry.writer import (
    DEFAULT_EVIDENCE_WRITER_POLICY,
    evidence_writer_policy_from_receipt,
    load_completed_evidence,
)

_COMPLETION_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "registry_sha256",
        "experiment",
        "runtime_sha256",
        "split_sha256",
        "split_contract",
        "activation_binding",
        "inventory_sha256",
        "inventory_source_receipt_sha256",
        "rows",
    }
)
_SPLIT_FIELDS = frozenset(
    {"schema_version", "kind", "registry_sha256", "experiment", "cells"}
)
_CONTRACT_FIELDS = frozenset(
    {
        "cell_id",
        "request_ids",
        "expected_request_rows",
        "expected_round_rows",
        "expected_update_rows",
        "expected_performance_rows",
        "request_ids_sha256",
        "corpus_sha256",
        "arrival_trace_sha256",
        "sampling_profile_sha256",
        "model_lock_sha256",
        "patched_sglang_tree",
        "workload_contract",
        "rank_config_sha256s",
        "physical_assignment",
        "physical_binding_sha256",
        "topology_receipt_sha256",
        "experiment_budget_sha256",
        "experiment_budget",
        "execution_plan_sha256",
        "execution_split_sha256",
    }
)
_MEASURED_ROW_FIELDS = frozenset(
    {
        "cell_id",
        "evidence_root",
        "run_id",
        "rank",
        "evidence_sha256",
        "terminal_receipt_sha256",
        "physical_gpu_uuid",
        "physical_binding_sha256",
        "experiment_budget_sha256",
        "budget_observation_status",
        "budget_observation_reason_code",
        "budget_observation_path",
        "budget_observation_sha256",
        "preflight_attestation_path",
        "preflight_attestation_sha256",
        "status",
    }
)
_ASSIGNMENT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "inventory_sha256",
        "inventory_source_receipt_sha256",
        "dispatch_plan_sha256",
        "experiment_budget_sha256",
        "budget_plan_sha256",
        "capacity_authority_sha256",
        "budget_materialization_authority_sha256",
        "assignment_sha256",
        "work_item_sha256",
        "gpu_uuids",
        "rank_groups",
        "ports",
        "gang_shape",
        "fixed_instance_gpu_count",
        "fixed_instance_billing_semantics",
        "host_id",
        "topology_group_ids",
    }
)
_DISABLED_SESSION_FIELDS = (
    "session_plan_sha256",
    "session_open_receipt_sha256",
    "reset_receipt_sha256",
    "session_epoch",
)
_SAFETY_COUNTERS = (
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "communicator_failures",
)
_SHA256_LENGTH = 64


class CompletionAuthorityUnavailableError(RuntimeError):
    """The raw inputs are honest but cannot authorize execution this release."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(name: str, value: object) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lower-case SHA-256")
    return value


def _strict_object(
    name: str,
    value: object,
    fields: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be a JSON object")
    if set(value) != fields:
        raise ValueError(f"{name} fields differ from schema")
    return value


def _strict_list(name: str, value: object) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    return value


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError(f"{label} path must be absolute and resolved")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} is not a readable regular file") from error
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read()
        opened_after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or (opened_before.st_dev, opened_before.st_ino)
            != (opened_after.st_dev, opened_after.st_ino)
            or opened_before.st_size != opened_after.st_size
            or opened_before.st_mtime_ns != opened_after.st_mtime_ns
            or current.st_dev != opened_after.st_dev
            or current.st_ino != opened_after.st_ino
            or current.st_size != opened_after.st_size
            or current.st_mtime_ns != opened_after.st_mtime_ns
            or opened_after.st_size != len(body)
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class DurableJsonArtifactBinding:
    """Raw-byte and semantic binding for one JSON file plus its sidecar."""

    path: str
    sidecar_path: str
    semantic_sha256: str
    file_sha256: str
    sidecar_file_sha256: str
    size: int
    sidecar_size: int

    @classmethod
    def from_path(cls, path: str | Path) -> DurableJsonArtifactBinding:
        source = Path(path).resolve()
        sidecar = Path(f"{source}.sha256").resolve()
        body = _regular_file_bytes(source, label="JSON artifact")
        sidecar_body = _regular_file_bytes(sidecar, label="JSON artifact sidecar")
        try:
            value = json.loads(
                body.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("JSON artifact is not strict UTF-8 JSON") from error
        semantic_sha256 = content_sha256(value)
        if sidecar_body != f"{semantic_sha256}\n".encode("ascii"):
            raise ValueError("JSON artifact sidecar is missing or invalid")
        binding = cls(
            path=str(source),
            sidecar_path=str(sidecar),
            semantic_sha256=semantic_sha256,
            file_sha256=hashlib.sha256(body).hexdigest(),
            sidecar_file_sha256=hashlib.sha256(sidecar_body).hexdigest(),
            size=len(body),
            sidecar_size=len(sidecar_body),
        )
        binding.load()
        return binding

    def load(self) -> object:
        """Reopen and revalidate both names; no cached JSON is trusted."""

        for name in ("semantic_sha256", "file_sha256", "sidecar_file_sha256"):
            _require_sha256(name, getattr(self, name))
        if type(self.size) is not int or self.size < 1:
            raise ValueError("JSON artifact size is invalid")
        if type(self.sidecar_size) is not int or self.sidecar_size < 1:
            raise ValueError("JSON artifact sidecar size is invalid")
        source = Path(self.path)
        sidecar = Path(self.sidecar_path)
        if sidecar != Path(f"{source}.sha256"):
            raise ValueError("JSON artifact sidecar path is not exact")
        body = _regular_file_bytes(source, label="bound JSON artifact")
        sidecar_body = _regular_file_bytes(sidecar, label="bound JSON artifact sidecar")
        if (
            len(body) != self.size
            or len(sidecar_body) != self.sidecar_size
            or hashlib.sha256(body).hexdigest() != self.file_sha256
            or hashlib.sha256(sidecar_body).hexdigest() != self.sidecar_file_sha256
            or sidecar_body != f"{self.semantic_sha256}\n".encode("ascii")
        ):
            raise RuntimeError("bound JSON artifact or sidecar changed")
        try:
            value = json.loads(
                body.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("bound JSON artifact became invalid") from error
        if content_sha256(value) != self.semantic_sha256:
            raise RuntimeError("bound JSON artifact semantic identity changed")
        return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        value[key] = item
    return value


@dataclass(frozen=True)
class CompletedRankTerminalBinding:
    """Durable raw terminal identity recovered from one measured rank."""

    cell_id: str
    run_id: str
    rank: int
    physical_assignment_sha256: str
    physical_gpu_uuid: str
    experiment_budget_sha256: str
    terminal_receipt_path: str
    terminal_receipt_sha256: str
    prepared_receipt_sha256: str
    budget_observation_path: str
    budget_observation_sha256: str
    native_terminal_artifact_path: str
    native_terminal_raw_sha256: str
    native_terminal_sha256: str
    trusted_attester_policy_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("terminal cell", self.cell_id)
        for name in (
            "terminal_receipt_sha256",
            "prepared_receipt_sha256",
            "budget_observation_sha256",
            "native_terminal_raw_sha256",
            "native_terminal_sha256",
            "trusted_attester_policy_sha256",
            "physical_assignment_sha256",
            "experiment_budget_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("terminal rank must be a non-negative integer")
        if type(self.physical_gpu_uuid) is not str or not self.physical_gpu_uuid:
            raise ValueError("terminal physical GPU UUID is invalid")
        for name in (
            "terminal_receipt_path",
            "budget_observation_path",
            "native_terminal_artifact_path",
        ):
            path = Path(getattr(self, name))
            if not path.is_absolute() or path.resolve() != path:
                raise ValueError(f"{name} must be an absolute resolved path")


@dataclass(frozen=True)
class AssignmentTerminalBinding:
    """Serializable per-assignment raw-file pointer accepted on resume."""

    authority_sha256: str
    cell_id: str
    assignment_sha256: str
    budget_sha256: str
    inventory_sha256: str
    physical_gpu_uuids: tuple[str, ...]
    execution_plan_sha256: str
    dispatch_plan_sha256: str
    run_id: str
    run_nonce_sha256: str
    terminal_receipt_path: str
    terminal_receipt_sha256: str
    budget_observation_path: str
    budget_observation_sha256: str
    budget_observation_sidecar_path: str
    budget_observation_sidecar_sha256: str
    native_terminal_artifact_path: str
    native_terminal_raw_sha256: str
    native_terminal_sha256: str
    trusted_attester_policy_sha256: str
    evidence_file_paths: tuple[str, ...]
    evidence_file_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "authority_sha256",
            "cell_id",
            "assignment_sha256",
            "budget_sha256",
            "inventory_sha256",
            "execution_plan_sha256",
            "dispatch_plan_sha256",
            "run_nonce_sha256",
            "terminal_receipt_sha256",
            "budget_observation_sha256",
            "budget_observation_sidecar_sha256",
            "native_terminal_raw_sha256",
            "native_terminal_sha256",
            "trusted_attester_policy_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not self.physical_gpu_uuids or len(set(self.physical_gpu_uuids)) != len(
            self.physical_gpu_uuids
        ):
            raise ValueError("assignment terminal binding has invalid GPU coverage")
        if type(self.run_id) is not str or not self.run_id or "\n" in self.run_id:
            raise ValueError("assignment terminal run ID is invalid")
        paths = (
            self.terminal_receipt_path,
            self.budget_observation_path,
            self.budget_observation_sidecar_path,
            self.native_terminal_artifact_path,
            *self.evidence_file_paths,
        )
        if any(
            not Path(path).is_absolute() or Path(path).resolve() != Path(path)
            for path in paths
        ):
            raise ValueError("assignment terminal path must be absolute and resolved")
        if (
            not self.evidence_file_paths
            or tuple(sorted(self.evidence_file_paths)) != self.evidence_file_paths
            or len(set(self.evidence_file_paths)) != len(self.evidence_file_paths)
            or len(self.evidence_file_paths) != len(self.evidence_file_sha256s)
            or any(not _is_sha256(value) for value in self.evidence_file_sha256s)
        ):
            raise ValueError("assignment terminal evidence coverage is malformed")

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "authority_sha256": self.authority_sha256,
            "cell_id": self.cell_id,
            "assignment_sha256": self.assignment_sha256,
            "budget_sha256": self.budget_sha256,
            "inventory_sha256": self.inventory_sha256,
            "physical_gpu_uuids": list(self.physical_gpu_uuids),
            "execution_plan_sha256": self.execution_plan_sha256,
            "dispatch_plan_sha256": self.dispatch_plan_sha256,
            "run_id": self.run_id,
            "run_nonce_sha256": self.run_nonce_sha256,
            "terminal_receipt_path": self.terminal_receipt_path,
            "terminal_receipt_sha256": self.terminal_receipt_sha256,
            "budget_observation_path": self.budget_observation_path,
            "budget_observation_sha256": self.budget_observation_sha256,
            "budget_observation_sidecar_path": (self.budget_observation_sidecar_path),
            "budget_observation_sidecar_sha256": (
                self.budget_observation_sidecar_sha256
            ),
            "native_terminal_artifact_path": self.native_terminal_artifact_path,
            "native_terminal_raw_sha256": self.native_terminal_raw_sha256,
            "native_terminal_sha256": self.native_terminal_sha256,
            "trusted_attester_policy_sha256": (self.trusted_attester_policy_sha256),
            "evidence_file_paths": list(self.evidence_file_paths),
            "evidence_file_sha256s": list(self.evidence_file_sha256s),
        }

    @classmethod
    def from_dict(cls, value: object) -> AssignmentTerminalBinding:
        fields = frozenset(
            {
                "authority_sha256",
                "cell_id",
                "assignment_sha256",
                "budget_sha256",
                "inventory_sha256",
                "physical_gpu_uuids",
                "execution_plan_sha256",
                "dispatch_plan_sha256",
                "run_id",
                "run_nonce_sha256",
                "terminal_receipt_path",
                "terminal_receipt_sha256",
                "budget_observation_path",
                "budget_observation_sha256",
                "budget_observation_sidecar_path",
                "budget_observation_sidecar_sha256",
                "native_terminal_artifact_path",
                "native_terminal_raw_sha256",
                "native_terminal_sha256",
                "trusted_attester_policy_sha256",
                "evidence_file_paths",
                "evidence_file_sha256s",
            }
        )
        row = _strict_object("assignment terminal binding", value, fields)
        return cls(
            authority_sha256=row["authority_sha256"],
            cell_id=row["cell_id"],
            assignment_sha256=row["assignment_sha256"],
            budget_sha256=row["budget_sha256"],
            inventory_sha256=row["inventory_sha256"],
            physical_gpu_uuids=tuple(
                _strict_list("terminal physical GPUs", row["physical_gpu_uuids"])
            ),
            execution_plan_sha256=row["execution_plan_sha256"],
            dispatch_plan_sha256=row["dispatch_plan_sha256"],
            run_id=row["run_id"],
            run_nonce_sha256=row["run_nonce_sha256"],
            terminal_receipt_path=row["terminal_receipt_path"],
            terminal_receipt_sha256=row["terminal_receipt_sha256"],
            budget_observation_path=row["budget_observation_path"],
            budget_observation_sha256=row["budget_observation_sha256"],
            budget_observation_sidecar_path=row["budget_observation_sidecar_path"],
            budget_observation_sidecar_sha256=row["budget_observation_sidecar_sha256"],
            native_terminal_artifact_path=row["native_terminal_artifact_path"],
            native_terminal_raw_sha256=row["native_terminal_raw_sha256"],
            native_terminal_sha256=row["native_terminal_sha256"],
            trusted_attester_policy_sha256=row["trusted_attester_policy_sha256"],
            evidence_file_paths=tuple(
                _strict_list("terminal evidence paths", row["evidence_file_paths"])
            ),
            evidence_file_sha256s=tuple(
                _strict_list("terminal evidence SHA-256s", row["evidence_file_sha256s"])
            ),
        )


@dataclass(frozen=True)
class AssignmentTerminalAuthority:
    """Per-assignment runner result backed by first-party serving evidence."""

    plan: object
    result: object
    run_nonce_sha256: str

    def __post_init__(self) -> None:
        from lightcone_spec.orchestration.executor import (
            IndustrialExecutionPlan,
            IndustrialExecutionResult,
        )

        if type(self.plan) is not IndustrialExecutionPlan:
            raise TypeError("assignment terminal requires an exact execution plan")
        if type(self.result) is not IndustrialExecutionResult:
            raise TypeError("assignment terminal requires an exact execution result")
        _require_sha256("assignment terminal run nonce", self.run_nonce_sha256)

    @classmethod
    def from_binding(
        cls,
        binding: AssignmentTerminalBinding,
        *,
        plan: object,
    ) -> AssignmentTerminalAuthority:
        """Rebuild restart authority from a persisted binding and raw files."""

        from lightcone_spec.orchestration.executor import (
            IndustrialExecutionPlan,
            IndustrialExecutionResult,
        )

        if type(binding) is not AssignmentTerminalBinding:
            raise TypeError("restart terminal authority requires an exact binding")
        if type(plan) is not IndustrialExecutionPlan:
            raise TypeError("restart terminal authority requires an exact plan")
        physical = plan.runtime_plan.physical_assignment
        if physical is None:
            raise ValueError("restart terminal authority lacks a physical assignment")
        if (
            binding.cell_id != plan.runtime_plan.cell_id
            or binding.execution_plan_sha256 != plan.sha256
            or binding.dispatch_plan_sha256 != plan.dispatch_plan.sha256
            or binding.assignment_sha256 != physical.assignment_sha256
            or binding.budget_sha256 != plan.budget.sha256
            or binding.inventory_sha256 != plan.dispatch_context.inventory.sha256
            or binding.physical_gpu_uuids != physical.gpu_uuids
        ):
            raise ValueError("restart terminal binding belongs to another exact plan")
        result = IndustrialExecutionResult(
            run_id=binding.run_id,
            execution_plan_sha256=binding.execution_plan_sha256,
            experiment_budget_sha256=binding.budget_sha256,
            rank_config_sha256=plan.rank_config_sha256,
            topology_sha256=plan.topology_sha256,
            resumed=True,
            terminal_receipt=binding.terminal_receipt_path,
            terminal_receipt_sha256=binding.terminal_receipt_sha256,
            budget_observation=binding.budget_observation_path,
            budget_observation_sidecar=binding.budget_observation_sidecar_path,
            budget_observation_sha256=binding.budget_observation_sha256,
            evidence_files=binding.evidence_file_paths,
            accounting=None,
        )
        authority = cls(
            plan=plan,
            result=result,
            run_nonce_sha256=binding.run_nonce_sha256,
        )
        if authority.sha256 != binding.authority_sha256:
            raise ValueError("restart terminal authority identity differs")
        expected = authority.revalidate(
            registry=plan.dispatch_context.registry,
            inventory=plan.dispatch_context.inventory,
            assignment_sha256=physical.assignment_sha256,
            budget_sha256=plan.budget.sha256,
            physical_gpu_uuids=physical.gpu_uuids,
        )
        if expected != binding:
            raise ValueError("restart terminal binding differs from raw files")
        return authority

    @property
    def registry(self) -> ExperimentRegistry:
        return self.plan.dispatch_context.registry

    @property
    def inventory(self) -> GpuInventory:
        return self.plan.dispatch_context.inventory

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "industrial_assignment_terminal_authority",
                "execution_plan_sha256": self.plan.sha256,
                "run_id": self.result.run_id,
                "run_nonce_sha256": self.run_nonce_sha256,
                "terminal_receipt_path": self.result.terminal_receipt,
                "terminal_receipt_sha256": self.result.terminal_receipt_sha256,
                "budget_observation_path": self.result.budget_observation,
                "budget_observation_sha256": (self.result.budget_observation_sha256),
                "budget_observation_sidecar": (self.result.budget_observation_sidecar),
                "evidence_files": sorted(self.result.evidence_files),
                "trusted_attester_policy_sha256": (
                    self.plan.trusted_attester_policy.sha256
                ),
            }
        )

    def revalidate(
        self,
        *,
        registry: ExperimentRegistry,
        inventory: GpuInventory,
        assignment_sha256: str,
        budget_sha256: str,
        physical_gpu_uuids: tuple[str, ...],
    ) -> AssignmentTerminalBinding:
        from lightcone_spec.orchestration.executor import (
            revalidate_industrial_execution_result,
        )

        if self.registry != registry or self.inventory != inventory:
            raise ValueError("assignment terminal authority is foreign")
        binding = revalidate_industrial_execution_result(
            plan=self.plan,
            result=self.result,
            run_nonce_sha256=self.run_nonce_sha256,
        )
        if (
            binding.cell_id != self.plan.runtime_plan.cell_id
            or binding.assignment_sha256 != assignment_sha256
            or binding.experiment_budget_sha256 != budget_sha256
            or binding.inventory_sha256 != inventory.sha256
            or binding.physical_gpu_uuids != physical_gpu_uuids
        ):
            raise ValueError(
                "assignment terminal differs from scheduler assignment/budget"
            )
        policy = self.plan.trusted_attester_policy
        if (
            not policy.release_ready
            or not binding.trusted_attestation
            or binding.trusted_attester_policy_sha256 != policy.sha256
        ):
            raise CompletionAuthorityUnavailableError(
                "assignment terminal authority is BLOCKED: "
                "trusted_hardware_attester_unavailable"
            )
        return AssignmentTerminalBinding(
            authority_sha256=self.sha256,
            cell_id=binding.cell_id,
            assignment_sha256=assignment_sha256,
            budget_sha256=budget_sha256,
            inventory_sha256=inventory.sha256,
            physical_gpu_uuids=physical_gpu_uuids,
            execution_plan_sha256=binding.execution_plan_sha256,
            dispatch_plan_sha256=binding.dispatch_plan_sha256,
            run_id=binding.run_id,
            run_nonce_sha256=binding.run_nonce_sha256,
            terminal_receipt_path=binding.terminal_receipt_path,
            terminal_receipt_sha256=binding.terminal_receipt_sha256,
            budget_observation_path=binding.budget_observation_path,
            budget_observation_sha256=binding.budget_observation_sha256,
            budget_observation_sidecar_path=(binding.budget_observation_sidecar_path),
            budget_observation_sidecar_sha256=(
                binding.budget_observation_sidecar_sha256
            ),
            native_terminal_artifact_path=binding.native_terminal_artifact_path,
            native_terminal_raw_sha256=binding.native_terminal_raw_sha256,
            native_terminal_sha256=binding.native_terminal_sha256,
            trusted_attester_policy_sha256=(binding.trusted_attester_policy_sha256),
            evidence_file_paths=binding.evidence_file_paths,
            evidence_file_sha256s=binding.evidence_file_sha256s,
        )


@dataclass(frozen=True)
class CompletedCellAuthorityResult:
    """Fresh result of replaying one raw completion authority."""

    completed_cell_ids: tuple[str, ...]
    completed_cells_sha256: str
    experiment: str
    terminal_bindings: tuple[CompletedRankTerminalBinding, ...]

    def __post_init__(self) -> None:
        _require_sha256("completed artifact", self.completed_cells_sha256)
        if self.completed_cell_ids != tuple(sorted(set(self.completed_cell_ids))):
            raise ValueError("completed cell IDs must be sorted and unique")
        if any(not _is_sha256(cell_id) for cell_id in self.completed_cell_ids):
            raise ValueError("completed cell ID is malformed")
        if type(self.experiment) is not str or not self.experiment:
            raise ValueError("completed experiment is invalid")


@dataclass(frozen=True)
class CompletedCellAuthority:
    """All raw authority needed to rederive schema-v4 completed cell IDs.

    Generic stage artifacts are verified directly.  E1/E2 and confirmation
    family activations additionally require a path-bound tagged raw authority;
    every revalidation reruns that authority before trusting reducer outputs.
    """

    completed_cells: DurableJsonArtifactBinding
    registry: ExperimentRegistry
    inventory: GpuInventory
    trusted_attester_policy: TrustedAttesterPolicy = RELEASE_TRUSTED_ATTESTER_POLICY
    direct_dependency_receipt: ExperimentReceipt | None = None
    dependency_authority: CompletedCellAuthority | None = None
    activation_artifact: (
        ReducerActivationArtifact | RegistryStageActivationArtifact | None
    ) = None
    family_activations: tuple[FamilyActivationArtifact, ...] = ()
    family_power_reductions: tuple[ConfirmationFamilyPowerReductionArtifact, ...] = ()
    prior_family_authorities: tuple[CompletedCellAuthority, ...] = ()
    raw_activation_authority: BudgetActivationAuthorityBinding | None = None

    def __post_init__(self) -> None:
        if type(self.completed_cells) is not DurableJsonArtifactBinding:
            raise TypeError("completion authority requires an exact durable binding")
        if type(self.registry) is not ExperimentRegistry:
            raise TypeError("completion authority requires an exact registry")
        if type(self.inventory) is not GpuInventory:
            raise TypeError("completion authority requires an exact GPU inventory")
        if self.trusted_attester_policy is not RELEASE_TRUSTED_ATTESTER_POLICY:
            raise ValueError(
                "completion authority rejects caller-selected trust roots; this "
                "release accepts only its source-owned policy constant"
            )
        require_release_trusted_attester_policy(self.trusted_attester_policy)
        if (
            self.direct_dependency_receipt is not None
            and type(self.direct_dependency_receipt) is not ExperimentReceipt
        ):
            raise TypeError("direct dependency must be an exact experiment receipt")
        if (
            self.dependency_authority is not None
            and type(self.dependency_authority) is not CompletedCellAuthority
        ):
            raise TypeError("dependency authority must be exact")
        if self.activation_artifact is not None and type(
            self.activation_artifact
        ) not in {ReducerActivationArtifact, RegistryStageActivationArtifact}:
            raise TypeError("activation input must be an exact reducer artifact")
        if any(
            type(row) is not FamilyActivationArtifact for row in self.family_activations
        ):
            raise TypeError("family activation input must be exact")
        if any(
            type(row) is not ConfirmationFamilyPowerReductionArtifact
            for row in self.family_power_reductions
        ):
            raise TypeError("family power input must be an exact raw reduction")
        if any(
            type(row) is not CompletedCellAuthority
            for row in self.prior_family_authorities
        ):
            raise TypeError("prior family authority must be exact")
        if len({id(row) for row in self.prior_family_authorities}) != len(
            self.prior_family_authorities
        ):
            raise ValueError("prior family authorities are duplicated")
        if self.raw_activation_authority is not None and not (
            _is_budget_activation_authority_binding(self.raw_activation_authority)
        ):
            raise TypeError("raw activation authority must be an exact tagged binding")

    @classmethod
    def from_path(
        cls,
        completed_cells_path: str | Path,
        *,
        registry: ExperimentRegistry,
        inventory: GpuInventory,
        trusted_attester_policy: TrustedAttesterPolicy = (
            RELEASE_TRUSTED_ATTESTER_POLICY
        ),
        direct_dependency_receipt: ExperimentReceipt | None = None,
        dependency_authority: CompletedCellAuthority | None = None,
        activation_artifact: (
            ReducerActivationArtifact | RegistryStageActivationArtifact | None
        ) = None,
        family_activations: tuple[FamilyActivationArtifact, ...] = (),
        family_power_reductions: tuple[
            ConfirmationFamilyPowerReductionArtifact, ...
        ] = (),
        prior_family_authorities: tuple[CompletedCellAuthority, ...] = (),
        raw_activation_authority: BudgetActivationAuthorityBinding | None = None,
    ) -> CompletedCellAuthority:
        return cls(
            completed_cells=DurableJsonArtifactBinding.from_path(completed_cells_path),
            registry=registry,
            inventory=inventory,
            trusted_attester_policy=trusted_attester_policy,
            direct_dependency_receipt=direct_dependency_receipt,
            dependency_authority=dependency_authority,
            activation_artifact=activation_artifact,
            family_activations=family_activations,
            family_power_reductions=family_power_reductions,
            prior_family_authorities=prior_family_authorities,
            raw_activation_authority=raw_activation_authority,
        )

    @property
    def sha256(self) -> str:
        """Content identity of the authority inputs, never of derived success."""

        return content_sha256(
            {
                "schema_version": 1,
                "kind": "industrial_completed_cell_authority",
                "completed_cells_sha256": self.completed_cells.semantic_sha256,
                "completed_cells_file_sha256": self.completed_cells.file_sha256,
                "completed_cells_sidecar_sha256": (
                    self.completed_cells.sidecar_file_sha256
                ),
                "registry_sha256": self.registry.sha256,
                "inventory_sha256": self.inventory.sha256,
                "inventory_source_receipt_sha256": (
                    self.inventory.source_receipt_sha256
                ),
                "trusted_attester_policy_sha256": (self.trusted_attester_policy.sha256),
                "direct_dependency_receipt_sha256": (
                    None
                    if self.direct_dependency_receipt is None
                    else self.direct_dependency_receipt.sha256
                ),
                "dependency_authority_sha256": (
                    None
                    if self.dependency_authority is None
                    else self.dependency_authority.sha256
                ),
                "activation_artifact_sha256": (
                    None
                    if self.activation_artifact is None
                    else self.activation_artifact.sha256
                ),
                "family_activation_sha256s": [
                    row.sha256 for row in self.family_activations
                ],
                "family_power_reduction_sha256s": [
                    row.sha256 for row in self.family_power_reductions
                ],
                "prior_family_authority_sha256s": [
                    row.sha256 for row in self.prior_family_authorities
                ],
                "raw_activation_authority_sha256": (
                    None
                    if self.raw_activation_authority is None
                    else self.raw_activation_authority.sha256
                ),
            }
        )

    def derive_completed_cell_ids(self) -> tuple[str, ...]:
        """Reopen and revalidate every raw artifact before returning IDs."""

        return self.revalidate().completed_cell_ids

    def revalidate(self) -> CompletedCellAuthorityResult:
        value = _strict_object(
            "schema-v4 completed-cell artifact",
            self.completed_cells.load(),
            _COMPLETION_FIELDS,
        )
        if (
            value["schema_version"] != 4
            or value["kind"] != "industrial_completed_cells"
            or value["registry_sha256"] != self.registry.sha256
            or value["inventory_sha256"] != self.inventory.sha256
            or value["inventory_source_receipt_sha256"]
            != self.inventory.source_receipt_sha256
        ):
            raise ValueError("completed-cell authority identity mismatch")
        stage = value["experiment"]
        if type(stage) is not str or not stage:
            raise ValueError("completed-cell authority has no exact stage")
        runtime_sha256 = _require_sha256("completion runtime", value["runtime_sha256"])
        split_sha256 = _require_sha256("completion split", value["split_sha256"])
        activated, dispositions, activation_binding = self._activation_contract(
            stage=stage,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
        )
        if value["activation_binding"] != activation_binding:
            raise ValueError("completion activation binding is missing or forged")
        contracts = self._validate_split(
            value["split_contract"],
            stage=stage,
            split_sha256=split_sha256,
            activated=activated,
        )
        rows = _strict_list("completed-cell rows", value["rows"])
        if any(type(row) is not dict for row in rows):
            raise TypeError("completed-cell rows must be JSON objects")
        measured = self._validate_row_coverage(
            rows,
            stage=stage,
            activated=activated,
            dispositions=dispositions,
            contracts=contracts,
        )
        terminals = tuple(
            self._validate_measured_row(
                row,
                stage=stage,
                contract=contracts[str(row["cell_id"])],
            )
            for row in measured
        )
        self._validate_rank_consensus(terminals, contracts=contracts)
        self._validate_dependency_lineage(
            stage=stage,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
        )
        cell_ids = tuple(sorted({binding.cell_id for binding in terminals}))
        return CompletedCellAuthorityResult(
            completed_cell_ids=cell_ids,
            completed_cells_sha256=self.completed_cells.semantic_sha256,
            experiment=stage,
            terminal_bindings=terminals,
        )

    def _activation_contract(
        self,
        *,
        stage: str,
        runtime_sha256: str,
        split_sha256: str,
    ) -> tuple[tuple[str, ...], dict[str, dict[str, str]], dict[str, object]]:
        self._replay_raw_activation_authority()
        stage_cells = {cell.cell_id: cell for cell in self.registry.cells_for(stage)}
        if not stage_cells:
            raise ValueError("completion stage is absent from the exact registry")
        direct_sha256 = (
            None
            if self.direct_dependency_receipt is None
            else self.direct_dependency_receipt.sha256
        )
        family_scoped = False
        if type(self.activation_artifact) is RegistryStageActivationArtifact:
            if self.family_activations or self.family_power_reductions:
                raise ValueError("generic activation cannot carry family inputs")
            artifact = self.activation_artifact
            verify_registry_stage_activation(self.registry, artifact)
            if (
                artifact.experiment != stage
                or artifact.runtime_sha256 != runtime_sha256
                or artifact.split_sha256 != split_sha256
                or artifact.direct_dependency_receipt_sha256 != direct_sha256
            ):
                raise ValueError("registry activation identity/lineage mismatch")
            status_map = {
                RegistryStageDispositionStatus.ACTIVATED: DispositionStatus.ACTIVATED,
                RegistryStageDispositionStatus.BLOCKED: DispositionStatus.BLOCKED,
                RegistryStageDispositionStatus.NOT_APPLICABLE: (
                    DispositionStatus.NOT_APPLICABLE
                ),
            }
            rows = tuple(
                (row.cell_id, status_map[row.status], row.reason_code)
                for row in artifact.dispositions
            )
            activation_round = artifact.activation_round
            activation_sha256: str | None = artifact.sha256
            family_sha256s: list[str] = []
            power_sha256s: list[str] = []
        elif self.family_activations:
            family_scoped = (
                type(self.raw_activation_authority)
                is not ConfirmationStageAggregateAuthorityBinding
            )
            if self.raw_activation_authority is None:
                raise CompletionAuthorityUnavailableError(
                    "completed-cell authority is BLOCKED: confirmation family raw "
                    "activation authority is unavailable"
                )
            if (
                self.activation_artifact is not None
                and type(self.raw_activation_authority)
                is not ConfirmationStageAggregateAuthorityBinding
            ):
                raise ValueError("family activation cannot carry a stage artifact")
            rows, activation_round = self._family_activation_rows(
                stage=stage,
                runtime_sha256=runtime_sha256,
                split_sha256=split_sha256,
            )
            if type(self.raw_activation_authority) is (
                ConfirmationStageAggregateAuthorityBinding
            ):
                auxiliary = self.activation_artifact
                expected_auxiliary = (
                    self.raw_activation_authority.auxiliary_completion_authority
                )
                if (auxiliary is None) != (expected_auxiliary is None):
                    raise ValueError(
                        "confirmation aggregate auxiliary activation differs"
                    )
                if auxiliary is not None:
                    if (
                        type(auxiliary) is not ReducerActivationArtifact
                        or expected_auxiliary is None
                        or auxiliary.sha256
                        != expected_auxiliary.activation.activation_sha256
                        or auxiliary.plan.experiment != stage
                        or auxiliary.plan.runtime_sha256 != runtime_sha256
                        or auxiliary.plan.split_sha256 != split_sha256
                        or auxiliary.plan.dependency_receipt_sha256 != direct_sha256
                    ):
                        raise ValueError(
                            "confirmation aggregate auxiliary activation identity differs"
                        )
                    rows = rows + tuple(
                        (row.cell_id, row.status, row.reason_code)
                        for row in auxiliary.dispositions
                    )
            activation_sha256 = (
                self.raw_activation_authority.activation_sha256
                if type(self.raw_activation_authority)
                is ConfirmationStageAggregateAuthorityBinding
                else None
            )
            family_sha256s = sorted(row.sha256 for row in self.family_activations)
            power_sha256s = sorted(row.sha256 for row in self.family_power_reductions)
        elif type(self.activation_artifact) is ReducerActivationArtifact:
            if type(self.raw_activation_authority) is (
                ConfirmationAuxiliaryActivationAuthorityBinding
            ):
                family_scoped = True
            if self.raw_activation_authority is None:
                raise CompletionAuthorityUnavailableError(
                    "completed-cell authority is BLOCKED: E1/E2 raw reducer source "
                    "bundle is unavailable"
                )
            artifact = self.activation_artifact
            plan = artifact.plan
            if (
                plan.registry_sha256 != self.registry.sha256
                or plan.experiment != stage
                or plan.runtime_sha256 != runtime_sha256
                or plan.split_sha256 != split_sha256
                or plan.dependency_receipt_sha256 != direct_sha256
            ):
                raise ValueError("specialized activation identity/lineage mismatch")
            rows = tuple(
                (row.cell_id, row.status, row.reason_code)
                for row in artifact.dispositions
            )
            activation_round = plan.activation_round
            activation_sha256 = artifact.sha256
            family_sha256s = []
            power_sha256s = []
        else:
            raise ValueError("completion authority lacks reducer-owned activation")
        dispositions: dict[str, dict[str, str]] = {}
        for cell_id, status, reason_code in rows:
            if cell_id not in stage_cells or cell_id in dispositions:
                raise ValueError("activation has an unknown or duplicate disposition")
            cell = stage_cells[cell_id]
            if status is DispositionStatus.ACTIVATED and not cell.runnable:
                raise ValueError("activation promotes a registry-blocked cell")
            dispositions[cell_id] = {
                "cell_id": cell_id,
                "status": status.value,
                "reason_code": reason_code,
            }
        expected_dispositions = (
            {cell_id for cell_id, _, _ in rows} if family_scoped else set(stage_cells)
        )
        if not expected_dispositions or set(dispositions) != expected_dispositions:
            raise ValueError("activation has incomplete disposition coverage")
        activated = tuple(
            sorted(
                cell_id
                for cell_id, row in dispositions.items()
                if row["status"] == DispositionStatus.ACTIVATED.value
            )
        )
        encoded = [dispositions[cell_id] for cell_id in sorted(dispositions)]
        if type(self.raw_activation_authority) is (
            ConfirmationStageAggregateAuthorityBinding
        ) and (
            activated != self.raw_activation_authority.activated_cell_ids
            or content_sha256(encoded)
            != self.raw_activation_authority.dispositions_sha256
        ):
            raise ValueError(
                "confirmation aggregate activated/disposition union differs"
            )
        return (
            activated,
            dispositions,
            {
                "schema_version": 1,
                "kind": "industrial_stage_activation_binding",
                "stage_activation_sha256": activation_sha256,
                "family_activation_sha256s": family_sha256s,
                "family_power_reduction_sha256s": power_sha256s,
                "direct_dependency_receipt_sha256": direct_sha256,
                "activation_round": activation_round,
                "dispositions_sha256": content_sha256(encoded),
            },
        )

    def _replay_raw_activation_authority(self) -> None:
        binding = self.raw_activation_authority
        if binding is None:
            return
        # Local import avoids a module cycle: budget authority constructs this
        # completion authority while recursively reopening dependency manifests.
        from lightcone_spec.experiments.budget_authority import (
            BudgetMaterializationBlockedError,
            replay_budget_activation_authority,
            require_ready_budget_activation_dependency_completions,
        )

        try:
            replay = replay_budget_activation_authority(binding)
        except BudgetMaterializationBlockedError as error:
            raise CompletionAuthorityUnavailableError(
                "completed-cell raw activation authority is BLOCKED: "
                f"{error.reason_code}"
            ) from error
        if replay.registry != self.registry:
            raise ValueError("raw activation authority swaps the exact registry")
        if replay.activation_artifact != self.activation_artifact:
            raise ValueError("raw activation authority output differs from activation")
        if replay.family_activations != self.family_activations:
            raise ValueError("raw family activation authority output differs")
        if replay.family_power_reductions != self.family_power_reductions:
            raise ValueError("raw family power authority output differs")
        if tuple(row.sha256 for row in replay.prior_family_authorities) != tuple(
            row.sha256 for row in self.prior_family_authorities
        ):
            raise ValueError("raw family completion lineage differs")
        replay_direct = (
            None if not replay.dependency_records else replay.dependency_records[-1]
        )
        if (None if replay_direct is None else replay_direct.receipt) != (
            self.direct_dependency_receipt
        ) or (None if replay_direct is None else replay_direct.authority.sha256) != (
            None
            if self.dependency_authority is None
            else self.dependency_authority.sha256
        ):
            raise ValueError(
                "raw activation authority changed direct dependency lineage"
            )
        if (
            type(binding) is E2ActivationAuthorityBinding and binding.stage_index > 0
        ) or type(binding) in {
            ConfirmationAuxiliaryActivationAuthorityBinding,
            ConfirmationStageAggregateAuthorityBinding,
        }:
            try:
                require_ready_budget_activation_dependency_completions(
                    binding,
                    expected_registry=self.registry,
                    expected_gpu_inventory=self.inventory,
                )
            except BudgetMaterializationBlockedError as error:
                raise CompletionAuthorityUnavailableError(
                    "completed-cell raw dependency authority is BLOCKED: "
                    f"{error.reason_code}"
                ) from error

    def _family_activation_rows(
        self,
        *,
        stage: str,
        runtime_sha256: str,
        split_sha256: str,
    ) -> tuple[tuple[tuple[str, DispositionStatus, str], ...], str]:
        by_round: dict[tuple[str, str], FamilyActivationArtifact] = {}
        for artifact in self.family_activations:
            family = artifact.family
            if (
                family.registry_sha256 != self.registry.sha256
                or family.experiment != stage
                or family.runtime_sha256 != runtime_sha256
                or family.split_sha256 != split_sha256
            ):
                raise ValueError("family activation identity mismatch")
            key = (family.sha256, artifact.activation_round)
            if key in by_round:
                raise ValueError("duplicate family activation round")
            by_round[key] = artifact
        reductions = {row.family.sha256: row for row in self.family_power_reductions}
        if len(reductions) != len(self.family_power_reductions):
            raise ValueError("duplicate family power reduction")
        family_ids = {family_sha256 for family_sha256, _ in by_round}
        final_ids = {
            family_sha256
            for family_sha256, round_name in by_round
            if round_name == "final_prefix"
        }
        prior_by_scope: dict[frozenset[str], tuple[str, ...]] = {}
        for authority in self.prior_family_authorities:
            result = authority.revalidate()
            scope = frozenset(result.completed_cell_ids)
            if not scope or scope in prior_by_scope:
                raise ValueError("family pilot completion scopes are duplicated")
            prior_by_scope[scope] = tuple(
                sorted(
                    binding.terminal_receipt_sha256
                    for binding in result.terminal_bindings
                )
            )
        consumed_prior_scopes: set[frozenset[str]] = set()
        current: list[tuple[str, DispositionStatus, str]] = []
        for family_sha256 in sorted(family_ids):
            pilot = by_round.get((family_sha256, "excluded_pilots"))
            final = by_round.get((family_sha256, "final_prefix"))
            if pilot is None:
                raise ValueError("family activation lacks its pilot reducer output")
            verify_confirmation_pilot_activation(
                self.registry,
                family=pilot.family,
                artifact=pilot,
            )
            selected = pilot
            if final is not None:
                reduction = reductions.get(family_sha256)
                if reduction is None:
                    raise ValueError(
                        "family final activation lacks raw power reduction"
                    )
                pilot_scope = frozenset(pilot.activated_cell_ids)
                prior_terminal_sha256s = prior_by_scope.get(pilot_scope)
                if prior_terminal_sha256s is None:
                    raise CompletionAuthorityUnavailableError(
                        "family final activation is BLOCKED: durable prior pilot "
                        "completion authority is incomplete"
                    )
                consumed_prior_scopes.add(pilot_scope)
                reduction_terminals = tuple(sorted(reduction.terminal_receipt_sha256s))
                if prior_terminal_sha256s != reduction_terminals:
                    raise ValueError(
                        "family power reduction swapped prior terminal receipts"
                    )
                expected = materialize_confirmation_prefix(
                    self.registry,
                    family=pilot.family,
                    reduction=reduction,
                    pilot_activation=pilot,
                )
                if final != expected:
                    raise ValueError("family final activation is not reducer-generated")
                selected = final
            for row in selected.dispositions:
                current.append((row.cell_id, row.status, row.reason_code))
        if set(reductions) - final_ids:
            raise ValueError("family power reduction has no matching final activation")
        if consumed_prior_scopes != set(prior_by_scope):
            raise ValueError("family completion authority has no matching final family")
        activation_round = (
            "final_prefix"
            if final_ids == family_ids
            else "excluded_pilots"
            if not final_ids
            else "family_incremental"
        )
        return tuple(current), activation_round

    def _validate_split(
        self,
        value: object,
        *,
        stage: str,
        split_sha256: str,
        activated: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        split = _strict_object("industrial locked split", value, _SPLIT_FIELDS)
        if (
            split["schema_version"] != 1
            or split["kind"] != "industrial_locked_split"
            or split["registry_sha256"] != self.registry.sha256
            or split["experiment"] != stage
            or content_sha256(split) != split_sha256
        ):
            raise ValueError("industrial locked-split identity mismatch")
        stage_cells = {cell.cell_id: cell for cell in self.registry.cells_for(stage)}
        contracts: dict[str, dict[str, Any]] = {}
        for raw in _strict_list("locked split cells", split["cells"]):
            contract = _strict_object("locked split cell", raw, _CONTRACT_FIELDS)
            cell_id = contract["cell_id"]
            if cell_id not in activated or cell_id in contracts:
                raise ValueError("locked split has unknown or duplicate activated cell")
            self._validate_contract(
                contract, cell=stage_cells[str(cell_id)], stage=stage
            )
            contracts[str(cell_id)] = contract
        if set(contracts) != set(activated):
            raise ValueError("locked split does not cover every activated cell")
        return contracts

    def _validate_contract(self, contract: dict[str, Any], *, cell, stage: str) -> None:
        from lightcone_spec.orchestration.industrial import (
            IndustrialPhysicalAssignment,
        )

        assignment_value = _strict_object(
            "industrial physical assignment",
            contract["physical_assignment"],
            _ASSIGNMENT_FIELDS,
        )
        if (
            assignment_value["schema_version"] != 3
            or assignment_value["kind"] != "industrial_physical_assignment"
        ):
            raise ValueError("industrial physical assignment identity mismatch")
        if (
            assignment_value["fixed_instance_billing_semantics"]
            != "whole_inventory_wall_clock_v1"
        ):
            raise ValueError("industrial physical assignment billing mismatch")
        gang = _strict_object(
            "industrial gang shape",
            assignment_value["gang_shape"],
            frozenset({"tensor_parallel_size", "data_parallel_size"}),
        )
        try:
            assignment = IndustrialPhysicalAssignment(
                inventory_sha256=assignment_value["inventory_sha256"],
                inventory_source_receipt_sha256=assignment_value[
                    "inventory_source_receipt_sha256"
                ],
                dispatch_plan_sha256=assignment_value["dispatch_plan_sha256"],
                experiment_budget_sha256=assignment_value["experiment_budget_sha256"],
                budget_plan_sha256=assignment_value["budget_plan_sha256"],
                capacity_authority_sha256=assignment_value["capacity_authority_sha256"],
                budget_materialization_authority_sha256=assignment_value[
                    "budget_materialization_authority_sha256"
                ],
                assignment_sha256=assignment_value["assignment_sha256"],
                work_item_sha256=assignment_value["work_item_sha256"],
                gpu_uuids=tuple(
                    _strict_list("assignment GPUs", assignment_value["gpu_uuids"])
                ),
                rank_groups=tuple(
                    tuple(_strict_list("assignment rank group", row))
                    for row in _strict_list(
                        "assignment rank groups", assignment_value["rank_groups"]
                    )
                ),
                ports=tuple(
                    _strict_list("assignment ports", assignment_value["ports"])
                ),
                tensor_parallel_size=gang["tensor_parallel_size"],
                data_parallel_size=gang["data_parallel_size"],
                fixed_instance_gpu_count=assignment_value["fixed_instance_gpu_count"],
                host_id=assignment_value["host_id"],
                topology_group_ids=tuple(
                    tuple(_strict_list("assignment topology group", row))
                    for row in _strict_list(
                        "assignment topology groups",
                        assignment_value["topology_group_ids"],
                    )
                ),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("industrial physical assignment is invalid") from error
        if assignment.to_dict() != assignment_value:
            raise ValueError("industrial physical assignment is not canonical")
        self._validate_assignment_inventory(assignment)
        try:
            budget = experiment_budget_from_dict(contract["experiment_budget"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "locked split contains a forged ExperimentBudget"
            ) from error
        if (
            contract["cell_id"] != cell.cell_id
            or contract["physical_binding_sha256"] != assignment.sha256
            or contract["experiment_budget_sha256"] != budget.sha256
            or contract["experiment_budget_sha256"]
            != assignment.experiment_budget_sha256
            or budget.cell_id != cell.cell_id
            or budget.experiment != cell.identity.experiment
            or budget.method != cell.identity.method
            or budget.workload_class is not cell.resources.workload_class
            or budget.gpu_count != cell.resources.gpu_count
            or budget.topology != cell.identity.topology
            or budget.measured_gpu_ms is not None
            or budget.fixed_instance_billed_gpu_ms
            != budget.wall_time.scale(len(self.inventory.devices))
        ):
            raise ValueError("locked physical assignment/budget binding is invalid")
        expected_topology = {
            "tp1_dp1": (1, 1),
            "tp2_dp1": (2, 1),
            "two_replica_tp1_dp2": (1, 2),
            "two_gpu_host": (1, 2),
            "two_independent_tp1": (1, 2),
        }.get(cell.identity.topology)
        if expected_topology != (
            assignment.tensor_parallel_size,
            assignment.data_parallel_size,
        ):
            raise ValueError("physical assignment disagrees with registry topology")
        request_ids = _strict_list("locked request IDs", contract["request_ids"])
        if (
            not request_ids
            or any(type(item) is not str or not item for item in request_ids)
            or len(request_ids) != len(set(request_ids))
            or contract["expected_request_rows"] != len(request_ids)
            or contract["request_ids_sha256"] != content_sha256(request_ids)
        ):
            raise ValueError("locked split request coverage is invalid")
        for name in (
            "corpus_sha256",
            "arrival_trace_sha256",
            "request_ids_sha256",
            "sampling_profile_sha256",
            "model_lock_sha256",
            "topology_receipt_sha256",
            "execution_plan_sha256",
            "execution_split_sha256",
        ):
            _require_sha256(f"locked {name}", contract[name])
        if contract["patched_sglang_tree"] != PINNED_SGLANG_TREE:
            raise ValueError("locked split uses another patched SGLang tree")
        expected_workload = (
            f"industrial_preflight_{cell.identity.method}"
            if stage == "preflight"
            else (
                f"industrial_{cell.identity.method}"
                if cell.identity.method in {"target_only", "static"}
                else "industrial_adapted"
            )
        )
        if contract["workload_contract"] != expected_workload:
            raise ValueError("locked workload contract differs from registry cell")
        rank_configs = contract["rank_config_sha256s"]
        if stage == "preflight":
            if rank_configs is not None:
                raise ValueError("preflight split cannot claim serving configs")
        elif (
            type(rank_configs) is not list
            or len(rank_configs) != len(assignment.gpu_uuids)
            or any(not _is_sha256(item) for item in rank_configs)
        ):
            raise ValueError("serving split lacks one config digest per rank")
        for name in (
            "expected_round_rows",
            "expected_update_rows",
            "expected_performance_rows",
        ):
            count = contract[name]
            if type(count) is not int or count < (
                1 if name == "expected_performance_rows" else 0
            ):
                raise ValueError("locked evidence row count is invalid")

    def _validate_assignment_inventory(self, assignment) -> None:
        if len(self.inventory.host_ids) != 1:
            raise ValueError("formal completion requires one whole-instance host")
        if (
            assignment.inventory_sha256 != self.inventory.sha256
            or assignment.inventory_source_receipt_sha256
            != self.inventory.source_receipt_sha256
            or assignment.fixed_instance_gpu_count != len(self.inventory.devices)
            or assignment.host_id != self.inventory.host_ids[0]
        ):
            raise ValueError("physical assignment differs from exact inventory")
        devices = {device.uuid: device for device in self.inventory.devices}
        if any(
            uuid not in devices or devices[uuid].host_id != assignment.host_id
            for uuid in assignment.gpu_uuids
        ):
            raise ValueError("physical assignment names a foreign GPU")
        topology_groups = {
            group.group_id: group for group in self.inventory.topology_groups
        }
        for rank_group, group_ids in zip(
            assignment.rank_groups,
            assignment.topology_group_ids,
            strict=True,
        ):
            if assignment.tensor_parallel_size == 1:
                if group_ids:
                    raise ValueError("TP1 assignment cannot claim topology groups")
                continue
            rank_set = set(rank_group)
            if any(
                group_id not in topology_groups
                or topology_groups[group_id].host_id != assignment.host_id
                or not rank_set <= set(topology_groups[group_id].gpu_uuids)
                or any(
                    group_id not in devices[uuid].allowed_topology_groups
                    for uuid in rank_group
                )
                for group_id in group_ids
            ):
                raise ValueError("physical TP assignment lacks topology authority")

    def _validate_row_coverage(
        self,
        rows: list[Any],
        *,
        stage: str,
        activated: tuple[str, ...],
        dispositions: dict[str, dict[str, str]],
        contracts: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        stage_cells = {cell.cell_id: cell for cell in self.registry.cells_for(stage)}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for raw in rows:
            if type(raw) is not dict or raw.get("cell_id") not in stage_cells:
                raise ValueError("completion rows cross the stage boundary")
            grouped.setdefault(str(raw["cell_id"]), []).append(raw)
        if set(grouped) != set(dispositions):
            raise ValueError("completion rows do not cover the activation scope")
        measured: list[dict[str, Any]] = []
        activated_set = set(activated)
        for cell_id, cell_rows in grouped.items():
            if cell_id not in activated_set:
                if cell_rows != [dispositions[cell_id]]:
                    raise ValueError("non-activated cell disposition is forged")
                continue
            assignment = self._assignment_for_contract(contracts[cell_id])
            if len(cell_rows) != len(assignment.gpu_uuids) or any(
                row.get("status") != "MEASURED" for row in cell_rows
            ):
                raise ValueError(
                    "activated cell requires one measured row per physical rank"
                )
            measured.extend(cell_rows)
        return tuple(measured)

    @staticmethod
    def _assignment_for_contract(contract: dict[str, Any]):
        from lightcone_spec.orchestration.industrial import (
            IndustrialPhysicalAssignment,
        )

        value = contract["physical_assignment"]
        gang = value["gang_shape"]
        return IndustrialPhysicalAssignment(
            inventory_sha256=value["inventory_sha256"],
            inventory_source_receipt_sha256=value["inventory_source_receipt_sha256"],
            dispatch_plan_sha256=value["dispatch_plan_sha256"],
            experiment_budget_sha256=value["experiment_budget_sha256"],
            budget_plan_sha256=value["budget_plan_sha256"],
            capacity_authority_sha256=value["capacity_authority_sha256"],
            budget_materialization_authority_sha256=value[
                "budget_materialization_authority_sha256"
            ],
            assignment_sha256=value["assignment_sha256"],
            work_item_sha256=value["work_item_sha256"],
            gpu_uuids=tuple(value["gpu_uuids"]),
            rank_groups=tuple(tuple(row) for row in value["rank_groups"]),
            ports=tuple(value["ports"]),
            tensor_parallel_size=gang["tensor_parallel_size"],
            data_parallel_size=gang["data_parallel_size"],
            fixed_instance_gpu_count=value["fixed_instance_gpu_count"],
            host_id=value["host_id"],
            topology_group_ids=tuple(tuple(row) for row in value["topology_group_ids"]),
        )

    def _validate_measured_row(
        self,
        row: dict[str, Any],
        *,
        stage: str,
        contract: dict[str, Any],
    ) -> CompletedRankTerminalBinding:
        _strict_object("formal completed-rank row", row, _MEASURED_ROW_FIELDS)
        cell_id = _require_sha256("completed row cell", row["cell_id"])
        cell = next(cell for cell in self.registry.cells if cell.cell_id == cell_id)
        assignment = self._assignment_for_contract(contract)
        rank = row["rank"]
        if type(rank) is not int or rank < 0 or rank >= len(assignment.gpu_uuids):
            raise ValueError("completed row rank is invalid")
        if (
            row["status"] != "MEASURED"
            or row["physical_gpu_uuid"] != assignment.gpu_uuids[rank]
            or row["physical_binding_sha256"] != assignment.sha256
            or row["experiment_budget_sha256"] != assignment.experiment_budget_sha256
        ):
            raise ValueError("completed rank differs from physical authority")
        for name in ("evidence_sha256", "terminal_receipt_sha256"):
            _require_sha256(f"completed row {name}", row[name])
        root = Path(row["evidence_root"])
        expected_root = Path(cell.resources.evidence_root).resolve()
        if (
            not root.is_absolute()
            or root.resolve() != root
            or root != expected_root
            or root.is_symlink()
            or not root.is_dir()
        ):
            raise ValueError("completed evidence root differs from resource claim")
        run_id = row["run_id"]
        if type(run_id) is not str or not run_id:
            raise ValueError("completed row run ID is invalid")
        if stage == "preflight" or cell.resources.workload_class in {
            WorkloadClass.COMPILE,
            WorkloadClass.DOWNLOAD,
        }:
            raise CompletionAuthorityUnavailableError(
                "completed-cell authority is BLOCKED: non-serving execution has "
                "no release terminal contract"
            )
        evidence = load_completed_evidence(root, run_id=run_id, rank=rank)
        if evidence is None:
            raise ValueError("completed row has no durable final receipt")
        terminal_path = (root / f"{run_id}.rank{rank}.complete.json").resolve()
        terminal_body = _regular_file_bytes(
            terminal_path, label="completed final receipt"
        )
        if hashlib.sha256(terminal_body).hexdigest() != row["terminal_receipt_sha256"]:
            raise ValueError("completed final receipt digest mismatch")
        try:
            final = json.loads(
                terminal_body.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("completed final receipt is invalid JSON") from error
        if type(final) is not dict:
            raise TypeError("completed final receipt must be an object")
        prepared_name = final.get("prepared_receipt_name")
        checkpoint = final.get("checkpoint")
        checkpoint_name = (
            None if type(checkpoint) is not dict else checkpoint.get("name")
        )
        if (
            type(prepared_name) is not str
            or Path(prepared_name).name != prepared_name
            or type(checkpoint_name) is not str
            or Path(checkpoint_name).name != checkpoint_name
        ):
            raise ValueError("completed receipt lacks writer-policy source files")
        prepared_path = (root / prepared_name).resolve()
        checkpoint_path = (root / checkpoint_name).resolve()
        if prepared_path.parent != root or checkpoint_path.parent != root:
            raise ValueError("completed writer-policy source escaped evidence root")
        if any(
            evidence_writer_policy_from_receipt(path) != DEFAULT_EVIDENCE_WRITER_POLICY
            for path in (terminal_path, prepared_path, checkpoint_path)
        ):
            raise ValueError(
                "completed evidence lacks the release EvidenceWriterPolicy"
            )
        if evidence_files_sha256(evidence.values()) != row["evidence_sha256"]:
            raise ValueError("completed evidence file digest mismatch")
        prepared_sha256 = _require_sha256(
            "prepared receipt", final.get("prepared_receipt_sha256")
        )
        observation = final.get("budget_observation")
        if type(observation) is not dict:
            raise ValueError("completed final receipt lacks budget observation")
        observation_path = Path(row["budget_observation_path"])
        expected_observation_path = (
            root
            / str(observation.get("directory"))
            / str(observation.get("receipt_name"))
        ).resolve()
        if (
            row["budget_observation_status"]
            != ("OBSERVED" if rank == 0 else "BOUND_TO_RANK0")
            or row["budget_observation_reason_code"]
            != (None if rank == 0 else "gang_observation_published_by_rank0")
            or not observation_path.is_absolute()
            or observation_path.resolve() != observation_path
            or observation_path != expected_observation_path
            or row["budget_observation_sha256"]
            != observation.get("budget_observation_sha256")
        ):
            raise ValueError("completed row swapped its budget observation")
        budget_observation_sha256 = _require_sha256(
            "budget observation", row["budget_observation_sha256"]
        )
        run_row = self._load_and_validate_run(
            evidence,
            cell=cell,
            contract=contract,
            assignment=assignment,
            run_id=run_id,
            rank=rank,
        )
        native_path, native_raw_sha256, native_sha256 = self._validate_native_terminal(
            root=root,
            evidence=evidence,
            run_row=run_row,
            cell=cell,
            contract=contract,
            run_id=run_id,
            rank=rank,
        )
        self._validate_performance_and_requests(
            evidence,
            cell=cell,
            contract=contract,
            native_path=native_path,
            native_raw_sha256=native_raw_sha256,
        )
        return CompletedRankTerminalBinding(
            cell_id=cell_id,
            run_id=run_id,
            rank=rank,
            physical_assignment_sha256=assignment.assignment_sha256,
            physical_gpu_uuid=assignment.gpu_uuids[rank],
            experiment_budget_sha256=assignment.experiment_budget_sha256,
            terminal_receipt_path=str(terminal_path),
            terminal_receipt_sha256=row["terminal_receipt_sha256"],
            prepared_receipt_sha256=prepared_sha256,
            budget_observation_path=str(observation_path),
            budget_observation_sha256=budget_observation_sha256,
            native_terminal_artifact_path=str(native_path),
            native_terminal_raw_sha256=native_raw_sha256,
            native_terminal_sha256=native_sha256,
            trusted_attester_policy_sha256=self.trusted_attester_policy.sha256,
        )

    def _load_and_validate_run(
        self,
        evidence: dict[str, Path],
        *,
        cell,
        contract: dict[str, Any],
        assignment,
        run_id: str,
        rank: int,
    ) -> dict[str, Any]:
        columns = [
            "run_id",
            "manifest_sha256",
            "config_sha256",
            "method",
            "model_pair",
            "repetition_block",
            "status",
            "industrial_cell_id",
            "rank_config_sha256",
            "runtime_sha256",
            "split_sha256",
            "corpus_sha256",
            "arrival_trace_sha256",
            "request_ids_sha256",
            "sampling_profile_sha256",
            "model_lock_sha256",
            "patched_sglang_tree",
            "run_nonce_sha256",
            "topology_sha256",
            "tensor_parallel_size",
            "data_parallel_size",
            "world_size",
            "rank",
            "expected_request_rows",
            "expected_round_rows",
            "expected_update_rows",
            "expected_performance_rows",
            "workload_contract",
            "experiment_budget_sha256",
            "preflight_attestation_sha256",
            "native_terminal_artifact_path",
            "native_terminal_artifact_size",
            "native_terminal_raw_sha256",
            "native_terminal_sha256",
            "trusted_attester_policy_sha256",
            *_DISABLED_SESSION_FIELDS,
        ]
        try:
            rows = pq.read_table(evidence["run"], columns=columns).to_pylist()
        except (KeyError, pa.ArrowException) as error:
            raise ValueError("completed run evidence lacks release bindings") from error
        if len(rows) != 1:
            raise ValueError("completed evidence requires one run row")
        run = rows[0]
        topology_sha256 = self._topology_sha256(
            cell=cell,
            assignment=assignment,
            topology_receipt_sha256=contract["topology_receipt_sha256"],
        )
        expected = {
            "run_id": run_id,
            "manifest_sha256": self.registry.sha256,
            "config_sha256": cell.cell_id,
            "method": cell.identity.method,
            "model_pair": cell.identity.model,
            "repetition_block": cell.identity.block,
            "status": "complete",
            "industrial_cell_id": cell.cell_id,
            "rank_config_sha256": contract["rank_config_sha256s"][rank],
            "runtime_sha256": contract["execution_plan_sha256"],
            "split_sha256": contract["execution_split_sha256"],
            "corpus_sha256": contract["corpus_sha256"],
            "arrival_trace_sha256": contract["arrival_trace_sha256"],
            "request_ids_sha256": contract["request_ids_sha256"],
            "sampling_profile_sha256": contract["sampling_profile_sha256"],
            "model_lock_sha256": contract["model_lock_sha256"],
            "patched_sglang_tree": PINNED_SGLANG_TREE,
            "topology_sha256": topology_sha256,
            "tensor_parallel_size": assignment.tensor_parallel_size,
            "data_parallel_size": assignment.data_parallel_size,
            "world_size": len(assignment.gpu_uuids),
            "rank": rank,
            "expected_round_rows": contract["expected_round_rows"],
            "expected_update_rows": contract["expected_update_rows"],
            "expected_performance_rows": contract["expected_performance_rows"],
            "workload_contract": contract["workload_contract"],
            "experiment_budget_sha256": contract["experiment_budget_sha256"],
            "preflight_attestation_sha256": None,
        }
        if not str(cell.identity.arrival).startswith("closed_loop"):
            expected["expected_request_rows"] = contract["expected_request_rows"]
        if any(run.get(name) != value for name, value in expected.items()):
            raise ValueError("completed run differs from locked execution contract")
        if not _is_sha256(run.get("run_nonce_sha256")):
            raise ValueError("completed run lacks a content-bound nonce")
        if any(run.get(name) is not None for name in _DISABLED_SESSION_FIELDS):
            raise ValueError("completed run claims unsupported shared-session evidence")
        return run

    def _validate_native_terminal(
        self,
        *,
        root: Path,
        evidence: dict[str, Path],
        run_row: dict[str, Any],
        cell,
        contract: dict[str, Any],
        run_id: str,
        rank: int,
    ) -> tuple[Path, str, str]:
        name = run_row.get("native_terminal_artifact_path")
        size = run_row.get("native_terminal_artifact_size")
        raw_sha256 = run_row.get("native_terminal_raw_sha256")
        terminal_sha256 = run_row.get("native_terminal_sha256")
        policy_sha256 = run_row.get("trusted_attester_policy_sha256")
        if (
            type(name) is not str
            or Path(name).name != name
            or type(size) is not int
            or size < 1
            or not _is_sha256(raw_sha256)
            or not _is_sha256(terminal_sha256)
            or policy_sha256 != self.trusted_attester_policy.sha256
        ):
            raise ValueError("completed run lacks exact native terminal binding")
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError("native terminal artifact escaped evidence root")
        body = _regular_file_bytes(path, label="native terminal artifact")
        if len(body) != size or hashlib.sha256(body).hexdigest() != raw_sha256:
            raise ValueError("native terminal raw artifact binding changed")
        try:
            artifact = json.loads(
                body.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("native terminal artifact is invalid JSON") from error
        if not self.trusted_attester_policy.release_ready:
            raise CompletionAuthorityUnavailableError(
                "completed-cell authority is BLOCKED: "
                "trusted_hardware_attester_unavailable"
            )
        validated = validate_native_terminal_artifact(
            artifact,
            trusted_attester_policy=self.trusted_attester_policy,
        )
        binding = validated.binding
        if (
            not validated.trusted_attestation
            or validated.terminal_sha256 != terminal_sha256
            or binding.run_id != run_id
            or binding.run_nonce_sha256 != run_row["run_nonce_sha256"]
            or binding.execution_plan_sha256 != contract["execution_plan_sha256"]
            or binding.rank_config_sha256 != contract["rank_config_sha256s"][rank]
            or binding.method != cell.identity.method
        ):
            raise ValueError("native terminal signature has foreign run identity")
        return path, str(raw_sha256), str(terminal_sha256)

    def _validate_performance_and_requests(
        self,
        evidence: dict[str, Path],
        *,
        cell,
        contract: dict[str, Any],
        native_path: Path,
        native_raw_sha256: str,
    ) -> None:
        try:
            performance = pq.read_table(
                evidence["performance"],
                columns=[
                    "method",
                    "updates_launched",
                    "updates_published",
                    "evidence_dropped_rows",
                    *_SAFETY_COUNTERS,
                ],
            ).to_pylist()
            requests = pq.read_table(
                evidence["request"],
                columns=[
                    "request_id",
                    "method",
                    "repetition_block",
                    "finished",
                    "outcome_status",
                    "output_tokens",
                    "output_sha256",
                    "output_token_ids",
                    "output_token_ids_sha256",
                ],
            ).to_pylist()
        except (KeyError, pa.ArrowException) as error:
            raise ValueError("completed evidence lacks terminal tables") from error
        if len(performance) != contract["expected_performance_rows"] or any(
            row["method"] != cell.identity.method
            or row["evidence_dropped_rows"] != 0
            or any(row[name] not in {0, None} for name in _SAFETY_COUNTERS)
            for row in performance
        ):
            raise ValueError("completed performance evidence is unsafe or lossy")
        if cell.identity.method not in {"target_only", "static"} and any(
            row["updates_launched"] is None
            or row["updates_launched"] < 1
            or row["updates_published"] is None
            or row["updates_published"] < 1
            for row in performance
        ):
            raise ValueError("adapted completion has no native update")
        if not requests or any(
            row["method"] != cell.identity.method
            or row["repetition_block"] != cell.identity.block
            or row["finished"] is not (row["outcome_status"] == "completed")
            for row in requests
        ):
            raise ValueError("completed request terminal outcomes are invalid")
        request_by_id = {str(row["request_id"]): row for row in requests}
        if len(request_by_id) != len(requests):
            raise ValueError("completed request identities are duplicated")
        native_body = _regular_file_bytes(
            native_path, label="reopened native terminal artifact"
        )
        if hashlib.sha256(native_body).hexdigest() != native_raw_sha256:
            raise RuntimeError("native terminal artifact changed during validation")
        try:
            artifact = json.loads(
                native_body.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("native terminal artifact is invalid JSON") from error
        validated = validate_native_terminal_artifact(
            artifact,
            trusted_attester_policy=self.trusted_attester_policy,
        )
        if set(validated.binding.scored_request_ids) != set(request_by_id):
            raise ValueError("native terminal request population differs")
        native_by_id = {row.request_id: row for row in validated.requests}
        if set(native_by_id) != set(request_by_id):
            raise ValueError("native terminal request identities differ")
        for request_id, row in request_by_id.items():
            native = native_by_id[request_id]
            if row["outcome_status"] == "completed":
                try:
                    output_ids = tuple(json.loads(row["output_token_ids"]))
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        "completed output token IDs are invalid"
                    ) from error
                if (
                    native.output_token_ids != output_ids
                    or row["output_tokens"] != len(output_ids)
                    or row["output_sha256"] != content_sha256(list(output_ids))
                    or row["output_token_ids_sha256"]
                    != content_sha256(list(output_ids))
                ):
                    raise ValueError("native and telemetry output identities differ")
            elif native.submitted_to_server or native.output_token_ids is not None:
                raise ValueError("non-completed request has forged native output")

    @staticmethod
    def _topology_sha256(*, cell, assignment, topology_receipt_sha256: str) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "cell_id": cell.cell_id,
                "topology": cell.identity.topology,
                "topology_receipt_sha256": topology_receipt_sha256,
                "physical_assignment_sha256": assignment.assignment_sha256,
                "physical_binding_sha256": assignment.sha256,
                "physical_host_id": assignment.host_id,
                "physical_gpu_uuids": list(assignment.gpu_uuids),
                "physical_rank_groups": [
                    list(group) for group in assignment.rank_groups
                ],
                "physical_ports": list(assignment.ports),
                "topology_group_ids": [
                    list(group) for group in assignment.topology_group_ids
                ],
                "tensor_parallel_size": assignment.tensor_parallel_size,
                "data_parallel_size": assignment.data_parallel_size,
                "world_size": len(assignment.gpu_uuids),
            }
        )

    @staticmethod
    def _validate_rank_consensus(
        terminals: tuple[CompletedRankTerminalBinding, ...],
        *,
        contracts: dict[str, dict[str, Any]],
    ) -> None:
        grouped: dict[str, list[CompletedRankTerminalBinding]] = {}
        for terminal in terminals:
            grouped.setdefault(terminal.cell_id, []).append(terminal)
        for cell_id, rows in grouped.items():
            assignment = CompletedCellAuthority._assignment_for_contract(
                contracts[cell_id]
            )
            ranks = tuple(sorted(row.rank for row in rows))
            if ranks != tuple(range(len(assignment.gpu_uuids))):
                raise ValueError("completed cell lacks exact per-rank terminals")
            if len({row.run_id for row in rows}) != 1:
                raise ValueError("completed cell ranks disagree on run identity")

    def _validate_dependency_lineage(
        self,
        *,
        stage: str,
        runtime_sha256: str,
        split_sha256: str,
    ) -> None:
        definition = self.registry.definition(stage)
        requires_dependency = bool(definition.dependencies)
        if not requires_dependency:
            if (
                self.direct_dependency_receipt is not None
                or self.dependency_authority is not None
            ):
                raise ValueError("root completion cannot claim dependency authority")
            return
        if self.direct_dependency_receipt is None:
            raise CompletionAuthorityUnavailableError(
                "completed-cell authority is BLOCKED: direct dependency receipt missing"
            )
        receipt = self.direct_dependency_receipt
        if (
            receipt.registry_sha256 != self.registry.sha256
            or receipt.experiment != definition.dependencies[-1]
        ):
            raise ValueError("direct dependency receipt belongs to another lineage")
        if self.dependency_authority is None:
            raise CompletionAuthorityUnavailableError(
                "completed-cell authority is BLOCKED: durable direct dependency "
                "authority missing"
            )
        dependency = self.dependency_authority.revalidate()
        if (
            dependency.experiment != receipt.experiment
            or dependency.completed_cells_sha256 != receipt.completed_cells_sha256
            or self.dependency_authority.registry != self.registry
            or self.dependency_authority.inventory != self.inventory
        ):
            raise ValueError("direct dependency receipt swapped completed authority")
        activation = self.activation_artifact
        if type(activation) is RegistryStageActivationArtifact and (
            activation.runtime_sha256 != runtime_sha256
            or activation.split_sha256 != split_sha256
            or activation.direct_dependency_receipt_sha256 != receipt.sha256
        ):
            raise ValueError("activation changed direct dependency lineage")


__all__ = [
    "AssignmentTerminalAuthority",
    "AssignmentTerminalBinding",
    "CompletedCellAuthority",
    "CompletedCellAuthorityResult",
    "CompletedRankTerminalBinding",
    "CompletionAuthorityUnavailableError",
    "DurableJsonArtifactBinding",
]
