"""Source-owned input graph for dispatch execution-bundle materialization.

``plan-industrial-dispatch`` produces an immutable :class:`GpuDispatchPlan`,
whereas ``execute-dispatch-wave`` consumes one fully bound execution bundle per
assignment.  This module closes the structural gap between those two surfaces:
it accepts paths to release-owned raw artifacts, derives assignment identities
from the dispatch plan itself, binds every byte and sidecar, and requires exact
assignment coverage.

The request schema deliberately has no assignment digest, execution-plan
digest, execution summary, output root, or caller-provided semantic hash.
Those values are downstream reducer outputs.  Input binding does not call a GPU
library or allocate an evidence root; publication becomes valid only when the
complete manifest and its sidecar are written last.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.adaptation.plan_authority import (
    TrainablePlanAuthorityBinding,
    replay_trainable_plan_authority,
    trainable_plan_authority_binding_from_dict,
)
from lightcone_spec.config import load_run_config, run_config_sha256
from lightcone_spec.experiments.budget_authority import (
    bind_budget_materialization_authority,
    load_declared_budget_plan,
    replay_budget_activation_authority,
)
from lightcone_spec.experiments.capacity_authority import bind_capacity_authority
from lightcone_spec.experiments.failure_authority import (
    bind_failure_injection_authority,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAssignment,
    GpuDispatchWave,
    GpuInventory,
    InterferenceEnvelope,
)
from lightcone_spec.experiments.inventory import build_serial_interference_envelope
from lightcone_spec.experiments.itl_authority import (
    ItlTimestampAuthorityBlocked,
    replay_e2_itl_timestamp_plan,
    require_e2_itl_timestamp_prelaunch,
)
from lightcone_spec.experiments.planning import (
    BudgetActivationAuthorityBinding,
    BudgetRawJsonBinding,
    _budget_activation_raw_sources,
)
from lightcone_spec.experiments.planning_artifacts import (
    budget_load_binding_from_dict,
    production_load_plan_from_dict,
)
from lightcone_spec.experiments.registry import (
    ExperimentReceipt,
    ExperimentRegistry,
    content_sha256,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.locking.models import ModelLock
from lightcone_spec.orchestration.execution_bundle import (
    AssignmentLaunchMaterializationPolicy,
    BoundExecutionArtifact,
    BoundJsonSource,
    IndustrialAssignmentExecutionBundle,
    InterferenceCalibrationExecutionAuthority,
    _load_registry,
    _receipt_from_dict,
    _topology_from_dict,
    finalize_materialized_execution_bundle,
    server_launch_to_dict,
)
from lightcone_spec.runtime.compile_cache import CompileCacheLaunchPlan

_REQUEST_KIND = "industrial_dispatch_bundle_materialization_request"
_BOUND_KIND = "industrial_dispatch_bundle_materialization_inputs"

_SHARED_SINGLE_ROLES = (
    "registry",
    "inventory",
    "interference_envelope",
    "interference_source_receipt",
    "budget_plan",
    "budget_policy",
    "capacity_envelope",
    "capacity_source_manifest",
    "capacity_verification_receipt",
    "activation",
    "activation_runtime",
    "activation_split",
    "dispatch_context",
    "dispatch_plan",
)
_SHARED_MULTI_ROLES = ("budget_load_bindings", "dependency_receipts")
_SHARED_OPTIONAL_SINGLE_ROLES = ("interference_calibration_execution_authority",)
_ASSIGNMENT_REQUIRED_SINGLE_ROLES = (
    "topology_receipts",
    "production_load",
    "run_config",
    "launch_policy",
    "run_nonce_receipt",
    "split_artifact",
    "sampling_artifact",
    "model_lock_artifact",
    "prepared_models",
    "compile_cache_plan",
    "inventory_source_artifact",
    "runtime_envelope_artifact",
    "execution_policy",
)
_ASSIGNMENT_OPTIONAL_SINGLE_ROLES = (
    "trainable_plan_authority_binding",
    "failure_injection_authority_plan",
    "itl_timestamp_authority_plan",
    "prepared_model_content_release_manifest",
)
_MANIFEST_KIND = "industrial_dispatch_execution_bundle_manifest"


class DispatchBundleMaterializationBlocked(RuntimeError):
    """The raw input graph is honest but incomplete for bundle reduction."""

    def __init__(self, reason_code: str) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("materialization BLOCKED reason must be non-empty")
        self.reason_code = reason_code
        super().__init__(f"dispatch bundle materialization is BLOCKED: {reason_code}")


def _strict_object(
    label: str, value: object, expected: frozenset[str]
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a JSON object")
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} fields differ: missing={missing}, extra={extra}")
    return value


def _strict_list(label: str, value: object) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON array")
    return value


def _strict_text(label: str, value: object) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def _strict_int(label: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _absolute_resolved_path(label: str, value: object) -> str:
    raw = _strict_text(label, value)
    path = Path(raw)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError(f"{label} must be an absolute resolved path")
    return str(path)


@dataclass(frozen=True)
class RawArtifactRole:
    """A named path, without caller-selected content identity."""

    role: str
    path: str

    def __post_init__(self) -> None:
        _strict_text("raw artifact role", self.role)
        _absolute_resolved_path("raw artifact", self.path)

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "path": self.path}


@dataclass(frozen=True)
class DependencyArtifactPath:
    """One locked dependency output named by its receipt role."""

    experiment: str
    name: str
    path: str

    def __post_init__(self) -> None:
        _strict_text("dependency experiment", self.experiment)
        _strict_text("dependency artifact name", self.name)
        _absolute_resolved_path("dependency artifact", self.path)

    @property
    def key(self) -> tuple[str, str]:
        return self.experiment, self.name

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment": self.experiment,
            "name": self.name,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "dependency artifact path",
            value,
            frozenset({"experiment", "name", "path"}),
        )
        return cls(
            experiment=_strict_text("dependency experiment", row["experiment"]),
            name=_strict_text("dependency artifact name", row["name"]),
            path=_absolute_resolved_path("dependency artifact", row["path"]),
        )


@dataclass(frozen=True)
class AssignmentRunNonceReceipt:
    """Fresh source-issued nonce bound to one frozen dispatch assignment."""

    schema_version: int
    kind: Literal["industrial_assignment_run_nonce_receipt"]
    dispatch_plan_sha256: str
    assignment_sha256: str
    cell_id: str
    run_nonce_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != ("industrial_assignment_run_nonce_receipt")
        ):
            raise ValueError("assignment run-nonce receipt is unsupported")
        for label, digest in (
            ("dispatch plan", self.dispatch_plan_sha256),
            ("assignment", self.assignment_sha256),
            ("cell", self.cell_id),
            ("run nonce", self.run_nonce_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "dispatch_plan_sha256": self.dispatch_plan_sha256,
            "assignment_sha256": self.assignment_sha256,
            "cell_id": self.cell_id,
            "run_nonce_sha256": self.run_nonce_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "assignment run-nonce receipt",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "dispatch_plan_sha256",
                    "assignment_sha256",
                    "cell_id",
                    "run_nonce_sha256",
                }
            ),
        )
        return cls(
            schema_version=_strict_int("nonce schema", row["schema_version"]),
            kind=_strict_text("nonce kind", row["kind"]),  # type: ignore[arg-type]
            dispatch_plan_sha256=_strict_text(
                "nonce dispatch plan", row["dispatch_plan_sha256"]
            ),
            assignment_sha256=_strict_text(
                "nonce assignment", row["assignment_sha256"]
            ),
            cell_id=_strict_text("nonce cell", row["cell_id"]),
            run_nonce_sha256=_strict_text("run nonce", row["run_nonce_sha256"]),
        )

    @classmethod
    def issue(
        cls,
        *,
        dispatch_plan_sha256: str,
        assignment: GpuAssignment,
    ) -> Self:
        """Mint entropy locally; callers cannot choose the nonce digest."""

        return cls(
            schema_version=1,
            kind="industrial_assignment_run_nonce_receipt",
            dispatch_plan_sha256=dispatch_plan_sha256,
            assignment_sha256=assignment.assignment_id,
            cell_id=assignment.work_item.item_id,
            run_nonce_sha256=content_sha256(
                {
                    "protocol": "industrial_assignment_run_nonce.v1",
                    "entropy": secrets.token_hex(32),
                    "dispatch_plan_sha256": dispatch_plan_sha256,
                    "assignment_sha256": assignment.assignment_id,
                }
            ),
        )


@dataclass(frozen=True)
class AssignmentRuntimeSourcePaths:
    """Release-owned raw paths for a cell; assignment identity is not input."""

    cell_id: str
    sources: tuple[RawArtifactRole, ...]
    dependency_artifacts: tuple[DependencyArtifactPath, ...]

    def __post_init__(self) -> None:
        if len(self.cell_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.cell_id
        ):
            raise ValueError("assignment runtime cell_id must be a lowercase SHA-256")
        roles = tuple(source.role for source in self.sources)
        if roles != tuple(sorted(set(roles))):
            raise ValueError("assignment source roles must be sorted and unique")
        required = set(_ASSIGNMENT_REQUIRED_SINGLE_ROLES)
        optional = set(_ASSIGNMENT_OPTIONAL_SINGLE_ROLES)
        present = set(roles)
        unknown = present - required - optional
        if unknown:
            raise ValueError(
                f"assignment source roles are unsupported: {sorted(unknown)}"
            )
        missing = required - present
        if missing:
            role = min(missing)
            raise DispatchBundleMaterializationBlocked(
                f"bundle_runtime_{role}_source_missing"
            )
        keys = tuple(artifact.key for artifact in self.dependency_artifacts)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("dependency artifacts must be role-sorted and unique")
        runtime_source = next(
            source
            for source in self.sources
            if source.role == "runtime_envelope_artifact"
        )
        runtime_dependencies = tuple(
            artifact
            for artifact in self.dependency_artifacts
            if artifact.name == "runtime_envelope"
        )
        if (
            len(runtime_dependencies) != 1
            or runtime_dependencies[0].path != runtime_source.path
        ):
            raise DispatchBundleMaterializationBlocked(
                "runtime_envelope_locked_dependency_binding_required"
            )
        paths = tuple(
            source.path
            for source in self.sources
            if source.role != "runtime_envelope_artifact"
        ) + tuple(artifact.path for artifact in self.dependency_artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("assignment runtime source paths are duplicated")

    @property
    def by_role(self) -> dict[str, str]:
        return {source.role: source.path for source in self.sources}

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "sources": [source.to_dict() for source in self.sources],
            "dependency_artifacts": [
                artifact.to_dict() for artifact in self.dependency_artifacts
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "assignment runtime sources",
            value,
            frozenset({"cell_id", "sources", "dependency_artifacts"}),
        )
        sources: list[RawArtifactRole] = []
        for item in _strict_list("assignment sources", row["sources"]):
            source_row = _strict_object(
                "assignment source",
                item,
                frozenset({"role", "path"}),
            )
            sources.append(
                RawArtifactRole(
                    role=_strict_text("assignment source role", source_row["role"]),
                    path=_absolute_resolved_path(
                        "assignment source path", source_row["path"]
                    ),
                )
            )
        return cls(
            cell_id=_strict_text("assignment cell_id", row["cell_id"]),
            sources=tuple(sources),
            dependency_artifacts=tuple(
                DependencyArtifactPath.from_dict(item)
                for item in _strict_list(
                    "assignment dependency artifacts", row["dependency_artifacts"]
                )
            ),
        )


@dataclass(frozen=True)
class DispatchBundleMaterializationRequest:
    """Path-only request for first-party plan-to-bundle reduction."""

    schema_version: int
    kind: Literal["industrial_dispatch_bundle_materialization_request"]
    shared_sources: tuple[RawArtifactRole, ...]
    shared_multi_sources: tuple[RawArtifactRole, ...]
    shared_optional_sources: tuple[RawArtifactRole, ...]
    assignments: tuple[AssignmentRuntimeSourcePaths, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != _REQUEST_KIND
        ):
            raise ValueError("dispatch bundle materialization request is unsupported")
        single_roles = tuple(source.role for source in self.shared_sources)
        if single_roles != tuple(sorted(_SHARED_SINGLE_ROLES)):
            raise ValueError(
                "shared single-source roles are incomplete or noncanonical"
            )
        multi_roles = tuple(source.role for source in self.shared_multi_sources)
        if multi_roles != tuple(sorted(multi_roles)) or any(
            role not in _SHARED_MULTI_ROLES for role in multi_roles
        ):
            raise ValueError("shared multi-source roles are noncanonical")
        for role in _SHARED_MULTI_ROLES:
            if role not in multi_roles:
                raise DispatchBundleMaterializationBlocked(
                    f"bundle_shared_{role}_source_missing"
                )
        optional_roles = tuple(source.role for source in self.shared_optional_sources)
        if optional_roles != tuple(sorted(set(optional_roles))) or any(
            role not in _SHARED_OPTIONAL_SINGLE_ROLES for role in optional_roles
        ):
            raise ValueError("shared optional-source roles are noncanonical")
        cells = tuple(assignment.cell_id for assignment in self.assignments)
        if cells != tuple(sorted(set(cells))):
            raise ValueError(
                "assignment runtime sources must be cell-sorted and unique"
            )
        if not cells:
            raise DispatchBundleMaterializationBlocked(
                "bundle_assignment_runtime_sources_missing"
            )
        all_paths = (
            tuple(source.path for source in self.shared_sources)
            + tuple(source.path for source in self.shared_multi_sources)
            + tuple(source.path for source in self.shared_optional_sources)
        )
        if len(all_paths) != len(set(all_paths)):
            raise ValueError("shared source paths are duplicated")

    @property
    def shared_by_role(self) -> dict[str, str]:
        return {source.role: source.path for source in self.shared_sources}

    @property
    def shared_optional_by_role(self) -> dict[str, str]:
        return {source.role: source.path for source in self.shared_optional_sources}

    def to_dict(self) -> dict[str, object]:
        grouped = {
            role: [
                source.path
                for source in self.shared_multi_sources
                if source.role == role
            ]
            for role in _SHARED_MULTI_ROLES
        }
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "shared_sources": {
                source.role: source.path for source in self.shared_sources
            },
            "shared_multi_sources": grouped,
            "shared_optional_sources": {
                role: next(
                    (
                        source.path
                        for source in self.shared_optional_sources
                        if source.role == role
                    ),
                    None,
                )
                for role in _SHARED_OPTIONAL_SINGLE_ROLES
            },
            "assignments": [assignment.to_dict() for assignment in self.assignments],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "dispatch bundle materialization request",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "shared_sources",
                    "shared_multi_sources",
                    "shared_optional_sources",
                    "assignments",
                }
            ),
        )
        if (
            _strict_int("materialization request schema", row["schema_version"]) != 1
            or row["kind"] != _REQUEST_KIND
        ):
            raise ValueError("dispatch bundle materialization request is unsupported")
        shared = _strict_object(
            "shared sources", row["shared_sources"], frozenset(_SHARED_SINGLE_ROLES)
        )
        multi = _strict_object(
            "shared multi sources",
            row["shared_multi_sources"],
            frozenset(_SHARED_MULTI_ROLES),
        )
        optional = _strict_object(
            "shared optional sources",
            row["shared_optional_sources"],
            frozenset(_SHARED_OPTIONAL_SINGLE_ROLES),
        )
        shared_multi: list[RawArtifactRole] = []
        for role in _SHARED_MULTI_ROLES:
            paths = _strict_list(f"shared {role}", multi[role])
            if not paths:
                raise DispatchBundleMaterializationBlocked(
                    f"bundle_shared_{role}_source_missing"
                )
            for path in paths:
                shared_multi.append(
                    RawArtifactRole(
                        role=role,
                        path=_absolute_resolved_path(f"shared {role}", path),
                    )
                )
        request = cls(
            schema_version=1,
            kind=_REQUEST_KIND,
            shared_sources=tuple(
                RawArtifactRole(
                    role=role,
                    path=_absolute_resolved_path(f"shared {role}", shared[role]),
                )
                for role in sorted(_SHARED_SINGLE_ROLES)
            ),
            shared_multi_sources=tuple(
                sorted(shared_multi, key=lambda source: (source.role, source.path))
            ),
            shared_optional_sources=tuple(
                RawArtifactRole(
                    role=role,
                    path=_absolute_resolved_path(
                        f"shared optional {role}", optional[role]
                    ),
                )
                for role in sorted(_SHARED_OPTIONAL_SINGLE_ROLES)
                if optional[role] is not None
            ),
            assignments=tuple(
                AssignmentRuntimeSourcePaths.from_dict(item)
                for item in _strict_list(
                    "assignment runtime sources", row["assignments"]
                )
            ),
        )
        if request.to_dict() != value:
            raise ValueError("dispatch bundle materialization request is noncanonical")
        return request


@dataclass(frozen=True)
class BoundAssignmentRuntimeSources:
    """Derived assignment identity plus content-bound raw runtime sources."""

    assignment_sha256: str
    cell_id: str
    run_nonce_sha256: str
    output_root: str
    sources: tuple[tuple[str, BoundJsonSource], ...]
    dependency_artifacts: tuple[tuple[str, str, BoundJsonSource], ...]

    def __post_init__(self) -> None:
        for label, digest in (
            ("assignment", self.assignment_sha256),
            ("cell", self.cell_id),
            ("run nonce", self.run_nonce_sha256),
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"bound {label} identity must be a lowercase SHA-256")
        root = Path(self.output_root)
        if not root.is_absolute() or root.resolve() != root:
            raise ValueError("derived evidence root must be absolute and resolved")

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_sha256": self.assignment_sha256,
            "cell_id": self.cell_id,
            "run_nonce_sha256": self.run_nonce_sha256,
            "output_root": self.output_root,
            "sources": [
                {"role": role, "source": source.to_dict()}
                for role, source in self.sources
            ],
            "dependency_artifacts": [
                {
                    "experiment": experiment,
                    "name": name,
                    "source": source.to_dict(),
                }
                for experiment, name, source in self.dependency_artifacts
            ],
        }


@dataclass(frozen=True)
class BoundDispatchBundleMaterializationInputs:
    """Complete, immutable structural input graph for an atomic bundle set."""

    schema_version: int
    kind: Literal["industrial_dispatch_bundle_materialization_inputs"]
    request: BoundJsonSource
    dispatch_plan: BoundJsonSource
    shared_sources: tuple[tuple[str, BoundJsonSource], ...]
    shared_multi_sources: tuple[tuple[str, BoundJsonSource], ...]
    shared_optional_sources: tuple[tuple[str, BoundJsonSource], ...]
    assignments: tuple[BoundAssignmentRuntimeSources, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != _BOUND_KIND
        ):
            raise ValueError("bound dispatch bundle inputs are unsupported")
        cells = tuple(row.cell_id for row in self.assignments)
        if cells != tuple(sorted(set(cells))):
            raise ValueError("bound assignments must be cell-sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "request": self.request.to_dict(),
            "dispatch_plan": self.dispatch_plan.to_dict(),
            "shared_sources": [
                {"role": role, "source": source.to_dict()}
                for role, source in self.shared_sources
            ],
            "shared_multi_sources": [
                {"role": role, "source": source.to_dict()}
                for role, source in self.shared_multi_sources
            ],
            "shared_optional_sources": [
                {"role": role, "source": source.to_dict()}
                for role, source in self.shared_optional_sources
            ],
            "assignments": [row.to_dict() for row in self.assignments],
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class MaterializedAssignmentBundleReceipt:
    """Manifest membership for one source-constructed schema-v5 bundle."""

    assignment_sha256: str
    cell_id: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    launch_policy: BoundJsonSource
    run_nonce_receipt: BoundJsonSource
    bundle: BoundJsonSource

    def __post_init__(self) -> None:
        for label, digest in (
            ("manifest assignment", self.assignment_sha256),
            ("manifest cell", self.cell_id),
            ("manifest run nonce", self.run_nonce_sha256),
            ("manifest execution plan", self.execution_plan_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256")
        if any(
            type(source) is not BoundJsonSource
            for source in (self.launch_policy, self.run_nonce_receipt, self.bundle)
        ):
            raise TypeError("materialized bundle receipt requires exact raw bindings")

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_sha256": self.assignment_sha256,
            "cell_id": self.cell_id,
            "run_nonce_sha256": self.run_nonce_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "launch_policy": self.launch_policy.to_dict(),
            "run_nonce_receipt": self.run_nonce_receipt.to_dict(),
            "bundle": self.bundle.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "materialized assignment bundle receipt",
            value,
            frozenset(
                {
                    "assignment_sha256",
                    "cell_id",
                    "run_nonce_sha256",
                    "execution_plan_sha256",
                    "launch_policy",
                    "run_nonce_receipt",
                    "bundle",
                }
            ),
        )
        return cls(
            assignment_sha256=_strict_text(
                "manifest assignment", row["assignment_sha256"]
            ),
            cell_id=_strict_text("manifest cell", row["cell_id"]),
            run_nonce_sha256=_strict_text(
                "manifest run nonce", row["run_nonce_sha256"]
            ),
            execution_plan_sha256=_strict_text(
                "manifest execution plan", row["execution_plan_sha256"]
            ),
            launch_policy=BoundJsonSource.from_dict(row["launch_policy"]),
            run_nonce_receipt=BoundJsonSource.from_dict(row["run_nonce_receipt"]),
            bundle=BoundJsonSource.from_dict(row["bundle"]),
        )


@dataclass(frozen=True)
class DispatchExecutionBundleManifest:
    """Manifest-last commit marker for a complete dispatch bundle set."""

    schema_version: int
    kind: Literal["industrial_dispatch_execution_bundle_manifest"]
    bundle_schema_version: Literal[5]
    materialization_inputs_sha256: str
    request: BoundJsonSource
    dispatch_plan: BoundJsonSource
    assignments: tuple[MaterializedAssignmentBundleReceipt, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != _MANIFEST_KIND
            or type(self.bundle_schema_version) is not int
            or self.bundle_schema_version != 5
        ):
            raise ValueError("dispatch execution-bundle manifest is unsupported")
        if len(self.materialization_inputs_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.materialization_inputs_sha256
        ):
            raise ValueError("materialization input identity must be a SHA-256")
        if any(
            type(source) is not BoundJsonSource
            for source in (self.request, self.dispatch_plan)
        ):
            raise TypeError("bundle manifest requires exact shared raw bindings")
        keys = tuple(
            (receipt.cell_id, receipt.assignment_sha256) for receipt in self.assignments
        )
        if not keys or keys != tuple(sorted(set(keys))):
            raise ValueError("bundle manifest assignments must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "bundle_schema_version": self.bundle_schema_version,
            "materialization_inputs_sha256": self.materialization_inputs_sha256,
            "request": self.request.to_dict(),
            "dispatch_plan": self.dispatch_plan.to_dict(),
            "assignments": [receipt.to_dict() for receipt in self.assignments],
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "dispatch execution-bundle manifest",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "bundle_schema_version",
                    "materialization_inputs_sha256",
                    "request",
                    "dispatch_plan",
                    "assignments",
                }
            ),
        )
        return cls(
            schema_version=_strict_int("manifest schema", row["schema_version"]),
            kind=_strict_text("manifest kind", row["kind"]),  # type: ignore[arg-type]
            bundle_schema_version=_strict_int(
                "manifest bundle schema", row["bundle_schema_version"]
            ),  # type: ignore[arg-type]
            materialization_inputs_sha256=_strict_text(
                "materialization input identity",
                row["materialization_inputs_sha256"],
            ),
            request=BoundJsonSource.from_dict(row["request"]),
            dispatch_plan=BoundJsonSource.from_dict(row["dispatch_plan"]),
            assignments=tuple(
                MaterializedAssignmentBundleReceipt.from_dict(item)
                for item in _strict_list(
                    "materialized assignment receipts", row["assignments"]
                )
            ),
        )


@dataclass(frozen=True)
class MaterializedDispatchExecutionBundlePublication:
    """A committed manifest together with its exact schema-v5 members."""

    manifest: DispatchExecutionBundleManifest
    bundles: tuple[IndustrialAssignmentExecutionBundle, ...]

    def __post_init__(self) -> None:
        if type(self.manifest) is not DispatchExecutionBundleManifest:
            raise TypeError("materialized publication requires an exact manifest")
        if any(
            type(bundle) is not IndustrialAssignmentExecutionBundle
            for bundle in self.bundles
        ):
            raise TypeError("materialized publication contains a non-exact bundle")
        if len(self.bundles) != len(self.manifest.assignments):
            raise ValueError("materialized publication bundle coverage is incomplete")


@dataclass(frozen=True)
class _MaterializationAuthority:
    registry: ExperimentRegistry
    inventory: GpuInventory
    interference_envelope: InterferenceEnvelope
    interference_calibration_authority: InterferenceCalibrationExecutionAuthority | None
    dependency_receipts: tuple[ExperimentReceipt, ...]
    registry_source: BoundJsonSource
    inventory_source: BoundJsonSource
    interference_envelope_source: BoundJsonSource
    interference_receipt_source: BoundJsonSource
    budget_plan_source: BoundJsonSource
    budget_policy_source: BoundJsonSource
    budget_load_sources: tuple[BoundJsonSource, ...]
    capacity_envelope_source: BoundJsonSource
    capacity_manifest_source: BoundJsonSource
    capacity_verification_source: BoundJsonSource
    dependency_receipt_sources: tuple[BoundJsonSource, ...]
    activation_source: BoundJsonSource
    activation_runtime_source: BoundJsonSource
    activation_split_source: BoundJsonSource
    dispatch_context_source: BoundJsonSource
    dispatch_plan_source: BoundJsonSource
    inventory_receipt_path: str


def _dispatch_assignments(value: object) -> tuple[GpuAssignment, ...]:
    """Strictly decode the plan's assignment layer without trusting a summary."""

    row = _strict_object(
        "GPU dispatch plan",
        value,
        frozenset(
            {
                "schema_version",
                "registry_sha256",
                "inventory_sha256",
                "receipts_sha256",
                "interference_envelope_sha256",
                "budget_sha256_by_cell",
                "scientific_budget_bound",
                "seed",
                "waves",
                "wave_sha256",
                "completed_cell_ids",
                "estimated_wall_seconds",
                "estimated_gpu_seconds",
                "estimated_gpu_hours",
            }
        ),
    )
    if (
        _strict_int("GPU dispatch schema", row["schema_version"]) != 1
        or row["scientific_budget_bound"] is not True
    ):
        raise DispatchBundleMaterializationBlocked(
            "dispatch_plan_scientific_budget_authority_missing"
        )
    waves = tuple(
        GpuDispatchWave.from_dict(item)
        for item in _strict_list("GPU dispatch waves", row["waves"])
    )
    if tuple(wave.wave_index for wave in waves) != tuple(range(len(waves))):
        raise ValueError("GPU dispatch wave indexes are not contiguous")
    declared_wave_sha256s = tuple(
        _strict_text("dispatch wave SHA-256", item)
        for item in _strict_list("dispatch wave SHA-256s", row["wave_sha256"])
    )
    if declared_wave_sha256s != tuple(wave.sha256 for wave in waves):
        raise ValueError("GPU dispatch wave identities changed")
    assignments = tuple(assignment for wave in waves for assignment in wave.assignments)
    if not assignments:
        raise DispatchBundleMaterializationBlocked("dispatch_plan_has_no_assignments")
    cells = tuple(assignment.work_item.item_id for assignment in assignments)
    if len(cells) != len(set(cells)):
        raise ValueError("GPU dispatch plan duplicates a cell")
    budget_rows = _strict_list("dispatch budget bindings", row["budget_sha256_by_cell"])
    budget_cells: list[str] = []
    for item in budget_rows:
        budget_row = _strict_object(
            "dispatch budget binding",
            item,
            frozenset({"cell_id", "experiment_budget_sha256"}),
        )
        budget_cells.append(_strict_text("dispatch budget cell", budget_row["cell_id"]))
        _strict_text("dispatch budget SHA-256", budget_row["experiment_budget_sha256"])
    if tuple(sorted(budget_cells)) != tuple(sorted(cells)):
        raise DispatchBundleMaterializationBlocked(
            "dispatch_plan_budget_assignment_coverage_incomplete"
        )
    return assignments


def bind_dispatch_bundle_materialization_inputs(
    request_path: str | Path,
) -> BoundDispatchBundleMaterializationInputs:
    """Bind a path-only request and derive exact assignment coverage onsite."""

    request_source = BoundJsonSource.bind(request_path)
    request = DispatchBundleMaterializationRequest.from_dict(request_source.load())
    shared_paths = request.shared_by_role
    dispatch_source = BoundJsonSource.bind(shared_paths["dispatch_plan"])
    assignments = _dispatch_assignments(dispatch_source.load())
    assignment_by_cell = {
        assignment.work_item.item_id: assignment for assignment in assignments
    }
    requested_by_cell = {row.cell_id: row for row in request.assignments}
    if set(requested_by_cell) != set(assignment_by_cell):
        raise DispatchBundleMaterializationBlocked(
            "bundle_assignment_runtime_source_coverage_incomplete"
        )

    shared_bound = tuple(
        (source.role, BoundJsonSource.bind(source.path))
        for source in request.shared_sources
    )
    shared_multi_bound = tuple(
        (source.role, BoundJsonSource.bind(source.path))
        for source in request.shared_multi_sources
    )
    bound_assignments: list[BoundAssignmentRuntimeSources] = []
    for cell_id in sorted(assignment_by_cell):
        assignment = assignment_by_cell[cell_id]
        runtime = requested_by_cell[cell_id]
        roles = runtime.by_role
        launch_policy_source = BoundJsonSource.bind(roles["launch_policy"])
        AssignmentLaunchMaterializationPolicy.from_dict(launch_policy_source.load())
        nonce_source = BoundJsonSource.bind(roles["run_nonce_receipt"])
        nonce = AssignmentRunNonceReceipt.from_dict(nonce_source.load())
        if (
            nonce.dispatch_plan_sha256 != dispatch_source.semantic_sha256
            or nonce.assignment_sha256 != assignment.assignment_id
            or nonce.cell_id != cell_id
        ):
            raise ValueError("run-nonce receipt differs from the dispatch assignment")
        method = assignment.work_item.cell.identity.method
        failure_cell = assignment.work_item.cell.identity.task == "failure_injection"
        failure_source = "failure_injection_authority_plan" in roles
        if failure_cell and not failure_source:
            raise DispatchBundleMaterializationBlocked(
                "failure_injection_raw_plan_authority_required"
            )
        if not failure_cell and failure_source:
            raise ValueError("non-failure assignment cannot carry failure authority")
        e2_cell = assignment.work_item.cell.identity.experiment == "E2"
        has_itl_plan = "itl_timestamp_authority_plan" in roles
        if e2_cell and not has_itl_plan:
            raise DispatchBundleMaterializationBlocked(
                "e2_itl_timestamp_plan_path_required"
            )
        if not e2_cell and has_itl_plan:
            raise ValueError("non-E2 assignment cannot carry an ITL authority plan")
        has_trainable = "trainable_plan_authority_binding" in roles
        has_prepared_release = "prepared_model_content_release_manifest" in roles
        if method in {"target_only", "static"}:
            if has_trainable or has_prepared_release:
                raise ValueError(
                    "Target-only/Static assignment cannot carry trainable authority"
                )
        elif method in {"tts", "l0"}:
            if not has_trainable:
                raise DispatchBundleMaterializationBlocked(
                    "trainable_plan_raw_authority_unavailable"
                )
            if not has_prepared_release:
                raise DispatchBundleMaterializationBlocked(
                    "prepared_model_content_release_manifest_pin_unavailable"
                )
        else:
            raise DispatchBundleMaterializationBlocked(
                "current_release_core_trainable_plan_method_required"
            )
        derived_root = str(
            Path(assignment.work_item.cell.resources.evidence_root).resolve()
        )
        bound_assignments.append(
            BoundAssignmentRuntimeSources(
                assignment_sha256=assignment.assignment_id,
                cell_id=cell_id,
                run_nonce_sha256=nonce.run_nonce_sha256,
                output_root=derived_root,
                sources=tuple(
                    (
                        source.role,
                        launch_policy_source
                        if source.role == "launch_policy"
                        else nonce_source
                        if source.role == "run_nonce_receipt"
                        else BoundJsonSource.bind(source.path),
                    )
                    for source in runtime.sources
                ),
                dependency_artifacts=tuple(
                    (
                        artifact.experiment,
                        artifact.name,
                        BoundJsonSource.bind(artifact.path),
                    )
                    for artifact in runtime.dependency_artifacts
                ),
            )
        )
    return BoundDispatchBundleMaterializationInputs(
        schema_version=1,
        kind=_BOUND_KIND,
        request=request_source,
        dispatch_plan=dispatch_source,
        shared_sources=shared_bound,
        shared_multi_sources=shared_multi_bound,
        shared_optional_sources=tuple(
            (source.role, BoundJsonSource.bind(source.path))
            for source in request.shared_optional_sources
        ),
        assignments=tuple(bound_assignments),
    )


def _sources_by_role(
    rows: tuple[tuple[str, BoundJsonSource], ...],
) -> dict[str, tuple[BoundJsonSource, ...]]:
    grouped: dict[str, list[BoundJsonSource]] = {}
    for role, source in rows:
        grouped.setdefault(role, []).append(source)
    return {role: tuple(sources) for role, sources in grouped.items()}


def _one_bound_source(
    rows: tuple[tuple[str, BoundJsonSource], ...], role: str
) -> BoundJsonSource:
    matches = tuple(source for name, source in rows if name == role)
    if len(matches) != 1:
        raise DispatchBundleMaterializationBlocked(
            f"bundle_{role}_source_coverage_incomplete"
        )
    return matches[0]


def _require_exact_raw_binding(
    source: BoundJsonSource,
    binding: object,
    *,
    label: str,
    compare_semantic: bool = True,
) -> None:
    canonical_sha256 = getattr(
        binding, "canonical_sha256", getattr(binding, "semantic_sha256", None)
    )
    if (
        source.path != getattr(binding, "path", None)
        or source.canonical_sha256 != canonical_sha256
        or source.file_sha256 != getattr(binding, "file_sha256", None)
        or source.sidecar_file_sha256 != getattr(binding, "sidecar_file_sha256", None)
        or source.size != getattr(binding, "size", None)
        or getattr(binding, "sidecar_path", f"{source.path}.sha256")
        != f"{source.path}.sha256"
        or getattr(binding, "sidecar_size", 65) != 65
        or (
            compare_semantic
            and source.semantic_sha256 != getattr(binding, "semantic_sha256", None)
        )
    ):
        raise ValueError(f"{label} differs from its first-party raw binding")


def _activation_binding_for_semantic(
    activation_binding: BudgetActivationAuthorityBinding,
    *,
    role: str,
    semantic_sha256: str,
) -> BudgetRawJsonBinding:
    matches = tuple(
        source
        for source in _budget_activation_raw_sources(activation_binding)
        if source.role == role and source.semantic_sha256 == semantic_sha256
    )
    if len(matches) != 1:
        raise DispatchBundleMaterializationBlocked(
            f"activation_{role}_path_bound_authority_ambiguous"
        )
    return matches[0]


def _capacity_declared_source_path(
    manifest_value: object,
    *,
    role: str,
) -> str:
    manifest = _strict_object(
        "capacity source manifest",
        manifest_value,
        frozenset(
            {
                "schema_version",
                "kind",
                "authority_protocol_sha256",
                "registry_sha256",
                "budget_inventory_sha256",
                "collection_nonce_sha256",
                "maximum_source_age_ns",
                "sources",
            }
        ),
    )
    sources = _strict_object(
        "capacity source manifest sources",
        manifest["sources"],
        frozenset(
            {
                "capacity_envelope",
                "gpu_inventory",
                "gpu_inventory_source_receipt",
                "provider_quota_receipt",
                "host_capacity_receipt",
                "cell_sizing_receipts",
            }
        ),
    )
    binding = sources[role]
    if type(binding) is not dict or type(binding.get("path")) is not str:
        raise TypeError(f"capacity {role} binding lacks an exact path")
    return _absolute_resolved_path(f"capacity {role} path", binding["path"])


def _reconstruct_materialization_authority(
    bound: BoundDispatchBundleMaterializationInputs,
) -> _MaterializationAuthority:
    shared = _sources_by_role(bound.shared_sources)

    raw_registry = shared["registry"][0]
    registry = _load_registry(raw_registry.load())
    registry_source = BoundJsonSource.bind(
        raw_registry.path, semantic_sha256=registry.sha256
    )

    raw_inventory = shared["inventory"][0]
    inventory = GpuInventory.from_dict(raw_inventory.load())
    inventory_source = BoundJsonSource.bind(
        raw_inventory.path, semantic_sha256=inventory.sha256
    )

    capacity_authority = bind_capacity_authority(
        shared["capacity_source_manifest"][0].path,
        shared["capacity_verification_receipt"][0].path,
    )
    capacity_manifest_source = BoundJsonSource.bind(
        capacity_authority.source_manifest.path,
        semantic_sha256=capacity_authority.source_manifest.semantic_sha256,
    )
    capacity_verification_source = BoundJsonSource.bind(
        capacity_authority.verification_receipt.path,
        semantic_sha256=capacity_authority.verification_receipt.semantic_sha256,
    )
    _require_exact_raw_binding(
        capacity_manifest_source,
        capacity_authority.source_manifest,
        label="capacity source manifest",
    )
    _require_exact_raw_binding(
        capacity_verification_source,
        capacity_authority.verification_receipt,
        label="capacity verification receipt",
    )
    manifest_value = capacity_manifest_source.load()
    declared_inventory_path = _capacity_declared_source_path(
        manifest_value, role="gpu_inventory"
    )
    if inventory_source.path != declared_inventory_path:
        raise ValueError("bundle inventory is not the raw capacity inventory")
    inventory_receipt_path = _capacity_declared_source_path(
        manifest_value, role="gpu_inventory_source_receipt"
    )

    multi = _sources_by_role(bound.shared_multi_sources)
    requested_loads: list[tuple[str, str]] = []
    for source in multi["budget_load_bindings"]:
        load = budget_load_binding_from_dict(source.load())
        requested_loads.append((load.cell_id, source.path))
    requested_loads.sort()
    if len({cell_id for cell_id, _ in requested_loads}) != len(requested_loads):
        raise ValueError("budget load inputs duplicate a cell")

    budget_authority = bind_budget_materialization_authority(
        activation_manifest_path=shared["activation"][0].path,
        policy_path=shared["budget_policy"][0].path,
        load_binding_paths=tuple(path for _, path in requested_loads),
        capacity_envelope_path=shared["capacity_envelope"][0].path,
        capacity_authority=capacity_authority,
        declared_plan_path=shared["budget_plan"][0].path,
    )
    activation_replay = replay_budget_activation_authority(budget_authority.activation)
    budget_plan = load_declared_budget_plan(budget_authority)
    if registry != activation_replay.registry:
        raise ValueError("materializer registry differs from activation replay")
    _require_exact_raw_binding(
        registry_source,
        budget_authority.activation.generated_registry,
        label="generated registry",
    )

    budget_plan_source = BoundJsonSource.bind(
        budget_authority.declared_plan.path,
        semantic_sha256=budget_plan.sha256,
    )
    budget_policy_source = BoundJsonSource.bind(
        budget_authority.policy.path,
        semantic_sha256=budget_authority.policy.semantic_sha256,
    )
    _require_exact_raw_binding(
        budget_plan_source, budget_authority.declared_plan, label="declared BudgetPlan"
    )
    _require_exact_raw_binding(
        budget_policy_source, budget_authority.policy, label="budget policy"
    )
    if {path for _, path in requested_loads} != {
        binding.source.path for binding in budget_authority.load_bindings
    }:
        raise DispatchBundleMaterializationBlocked(
            "budget_load_binding_path_coverage_incomplete"
        )
    budget_load_sources = tuple(
        BoundJsonSource.bind(
            binding.source.path,
            semantic_sha256=binding.source.semantic_sha256,
        )
        for binding in budget_authority.load_bindings
    )
    for source, binding in zip(
        budget_load_sources, budget_authority.load_bindings, strict=True
    ):
        _require_exact_raw_binding(source, binding.source, label="budget load binding")

    capacity_envelope_source = BoundJsonSource.bind(
        budget_authority.capacity_envelope.path,
        semantic_sha256=budget_authority.capacity_envelope.semantic_sha256,
    )
    _require_exact_raw_binding(
        capacity_envelope_source,
        budget_authority.capacity_envelope,
        label="capacity envelope",
    )
    if capacity_envelope_source.path != _capacity_declared_source_path(
        manifest_value, role="capacity_envelope"
    ):
        raise ValueError("bundle capacity envelope is not the raw capacity source")

    activation_source = BoundJsonSource.bind(
        budget_authority.activation.manifest.path,
        semantic_sha256=activation_replay.activation_sha256,
    )
    _require_exact_raw_binding(
        activation_source,
        budget_authority.activation.manifest,
        label="activation manifest",
        compare_semantic=False,
    )
    runtime_binding = _activation_binding_for_semantic(
        budget_authority.activation,
        role="activation_runtime",
        semantic_sha256=activation_replay.runtime_sha256,
    )
    split_binding = _activation_binding_for_semantic(
        budget_authority.activation,
        role="activation_split",
        semantic_sha256=activation_replay.split_sha256,
    )
    activation_runtime_source = BoundJsonSource.bind(
        shared["activation_runtime"][0].path,
        semantic_sha256=activation_replay.runtime_sha256,
    )
    activation_split_source = BoundJsonSource.bind(
        shared["activation_split"][0].path,
        semantic_sha256=activation_replay.split_sha256,
    )
    _require_exact_raw_binding(
        activation_runtime_source, runtime_binding, label="activation runtime"
    )
    _require_exact_raw_binding(
        activation_split_source, split_binding, label="activation split"
    )

    requested_receipts = {
        source.path: source for source in multi["dependency_receipts"]
    }
    receipt_sources: list[BoundJsonSource] = []
    for receipt in activation_replay.dependency_receipts:
        receipt_binding = _activation_binding_for_semantic(
            budget_authority.activation,
            role="activation_dependency_receipt",
            semantic_sha256=receipt.sha256,
        )
        if receipt_binding.path not in requested_receipts:
            raise DispatchBundleMaterializationBlocked(
                "dependency_receipt_path_coverage_incomplete"
            )
        source = BoundJsonSource.bind(
            receipt_binding.path, semantic_sha256=receipt.sha256
        )
        _require_exact_raw_binding(
            source, receipt_binding, label="activation dependency receipt"
        )
        if _receipt_from_dict(source.load()) != receipt:
            raise ValueError("dependency receipt differs from activation replay")
        receipt_sources.append(source)
    if set(requested_receipts) != {source.path for source in receipt_sources}:
        raise DispatchBundleMaterializationBlocked(
            "dependency_receipt_path_coverage_incomplete"
        )

    raw_envelope = shared["interference_envelope"][0]
    envelope = InterferenceEnvelope.from_dict(raw_envelope.load())
    envelope_source = BoundJsonSource.bind(
        raw_envelope.path, semantic_sha256=envelope.sha256
    )
    raw_interference_receipt = shared["interference_source_receipt"][0]
    calibration = None
    optional = _sources_by_role(bound.shared_optional_sources)
    if "interference_calibration_execution_authority" in optional:
        if not envelope.rules:
            raise ValueError("serial interference cannot carry calibration authority")
        calibration = InterferenceCalibrationExecutionAuthority.from_dict(
            optional["interference_calibration_execution_authority"][0].load()
        )
        calibration.reconstruct()
        receipt_semantic = calibration.source.manifest.semantic_sha256
    elif not envelope.rules:
        expected_envelope, expected_receipt = build_serial_interference_envelope(
            inventory
        )
        if envelope != expected_envelope or raw_interference_receipt.load() != (
            expected_receipt
        ):
            raise ValueError("serial interference raw authority differs")
        receipt_semantic = expected_receipt["receipt_sha256"]
    else:
        # The schema-v5 replay will accept only the registered bootstrap receipt
        # for this case; no caller-produced summary is treated as calibration.
        receipt_value = raw_interference_receipt.load()
        if (
            type(receipt_value) is not dict
            or type(receipt_value.get("receipt_sha256")) is not str
        ):
            raise DispatchBundleMaterializationBlocked(
                "calibrated_interference_raw_authority_required"
            )
        receipt_semantic = receipt_value["receipt_sha256"]
    interference_receipt_source = BoundJsonSource.bind(
        raw_interference_receipt.path,
        semantic_sha256=receipt_semantic,
    )

    dispatch_context_source = BoundJsonSource.bind(shared["dispatch_context"][0].path)
    dispatch_plan_source = BoundJsonSource.bind(
        bound.dispatch_plan.path,
        semantic_sha256=content_sha256(bound.dispatch_plan.load()),
    )
    if dispatch_plan_source.semantic_sha256 != bound.dispatch_plan.semantic_sha256:
        raise ValueError("dispatch plan identity changed during materialization")

    return _MaterializationAuthority(
        registry=registry,
        inventory=inventory,
        interference_envelope=envelope,
        interference_calibration_authority=calibration,
        dependency_receipts=activation_replay.dependency_receipts,
        registry_source=registry_source,
        inventory_source=inventory_source,
        interference_envelope_source=envelope_source,
        interference_receipt_source=interference_receipt_source,
        budget_plan_source=budget_plan_source,
        budget_policy_source=budget_policy_source,
        budget_load_sources=budget_load_sources,
        capacity_envelope_source=capacity_envelope_source,
        capacity_manifest_source=capacity_manifest_source,
        capacity_verification_source=capacity_verification_source,
        dependency_receipt_sources=tuple(receipt_sources),
        activation_source=activation_source,
        activation_runtime_source=activation_runtime_source,
        activation_split_source=activation_split_source,
        dispatch_context_source=dispatch_context_source,
        dispatch_plan_source=dispatch_plan_source,
        inventory_receipt_path=inventory_receipt_path,
    )


def _materialize_assignment_provisional(
    *,
    assignment: GpuAssignment,
    bound_assignment: BoundAssignmentRuntimeSources,
    authority: _MaterializationAuthority,
) -> tuple[
    IndustrialAssignmentExecutionBundle,
    AssignmentLaunchMaterializationPolicy,
    BoundJsonSource,
    BoundJsonSource,
]:
    roles = {role: source for role, source in bound_assignment.sources}
    cell_id = assignment.work_item.item_id
    if (
        bound_assignment.assignment_sha256 != assignment.assignment_id
        or bound_assignment.cell_id != cell_id
    ):
        raise ValueError("bound assignment identity changed before reduction")

    itl_timestamp_plan_source = None
    itl_timestamp_plan_sha256 = None
    itl_timestamp_producer_sha256 = None
    raw_itl_plan_source = roles.get("itl_timestamp_authority_plan")
    if assignment.work_item.cell.identity.experiment == "E2":
        if raw_itl_plan_source is None:
            raise DispatchBundleMaterializationBlocked(
                "e2_itl_timestamp_plan_path_required"
            )
        itl_timestamp_plan = replay_e2_itl_timestamp_plan(
            authority.registry,
            assignment.work_item.cell,
            raw_itl_plan_source.load(),
        )
        try:
            producer = require_e2_itl_timestamp_prelaunch(itl_timestamp_plan)
        except ItlTimestampAuthorityBlocked as error:
            raise DispatchBundleMaterializationBlocked(error.reason) from error
        itl_timestamp_plan_source = BoundJsonSource.bind(
            raw_itl_plan_source.path,
            semantic_sha256=itl_timestamp_plan.sha256,
        )
        itl_timestamp_plan_sha256 = itl_timestamp_plan.sha256
        itl_timestamp_producer_sha256 = producer.sha256
    elif raw_itl_plan_source is not None:
        raise ValueError("non-E2 materialization cannot carry ITL authority")

    launch_policy_source = roles["launch_policy"]
    launch_policy = AssignmentLaunchMaterializationPolicy.from_dict(
        launch_policy_source.load()
    )
    nonce_source = roles["run_nonce_receipt"]
    nonce = AssignmentRunNonceReceipt.from_dict(nonce_source.load())
    if (
        nonce.assignment_sha256 != assignment.assignment_id
        or nonce.cell_id != cell_id
        or nonce.run_nonce_sha256 != bound_assignment.run_nonce_sha256
        or nonce.dispatch_plan_sha256 != authority.dispatch_plan_source.semantic_sha256
    ):
        raise ValueError("assignment run nonce changed before bundle reduction")

    topology = _topology_from_dict(roles["topology_receipts"].load())
    topology_source = BoundJsonSource.bind(
        roles["topology_receipts"].path,
        semantic_sha256=topology.receipt_sha256,
    )
    production_load = production_load_plan_from_dict(roles["production_load"].load())
    production_load_source = BoundJsonSource.bind(
        roles["production_load"].path,
        semantic_sha256=production_load.paired_replay_sha256,
    )
    run_config = load_run_config(roles["run_config"].path)
    run_config_source = BoundJsonSource.bind(
        roles["run_config"].path,
        semantic_sha256=run_config_sha256(run_config),
    )
    if run_config.model_dump(mode="json") != run_config_source.load():
        raise ValueError("assignment RunConfig is not canonical")

    split_value = roles["split_artifact"].load()
    split_source = BoundJsonSource.bind(
        roles["split_artifact"].path,
        semantic_sha256=content_sha256(split_value),
    )
    sampling = SamplingProfile.load(roles["sampling_artifact"].path)
    sampling_source = BoundJsonSource.bind(
        roles["sampling_artifact"].path,
        semantic_sha256=sampling.sha256,
    )
    model_lock = ModelLock.load(roles["model_lock_artifact"].path)
    model_lock_source = BoundJsonSource.bind(
        roles["model_lock_artifact"].path,
        semantic_sha256=model_lock.sha256,
    )
    prepared_models_source = BoundJsonSource.bind(roles["prepared_models"].path)
    prepared_models_source.load()
    compile_plan = CompileCacheLaunchPlan.load(roles["compile_cache_plan"].path)
    compile_source = BoundJsonSource.bind(
        roles["compile_cache_plan"].path,
        semantic_sha256=compile_plan.sha256,
    )
    execution_policy_source = BoundJsonSource.bind(roles["execution_policy"].path)
    execution_policy_source.load()

    expected_outputs = tuple(
        sorted(
            (receipt.experiment, output.name, output.content_sha256)
            for receipt in authority.dependency_receipts
            for output in receipt.outputs
        )
    )
    expected_keys = tuple(
        (experiment, name) for experiment, name, _ in expected_outputs
    )
    if expected_keys != tuple(sorted(set(expected_keys))):
        raise ValueError("dependency receipts duplicate a locked output role")
    requested_artifacts = {
        (experiment, name): source
        for experiment, name, source in bound_assignment.dependency_artifacts
    }
    if set(requested_artifacts) != set(expected_keys):
        raise DispatchBundleMaterializationBlocked(
            "dependency_artifact_locked_output_coverage_incomplete"
        )
    dependency_artifacts = tuple(
        BoundExecutionArtifact(
            name=name,
            experiment=experiment,
            source=BoundJsonSource.bind(
                requested_artifacts[(experiment, name)].path,
                semantic_sha256=semantic_sha256,
            ),
        )
        for experiment, name, semantic_sha256 in expected_outputs
    )
    runtime_envelopes = tuple(
        artifact
        for artifact in dependency_artifacts
        if artifact.name == "runtime_envelope"
    )
    if len(runtime_envelopes) != 1:
        raise DispatchBundleMaterializationBlocked(
            "runtime_envelope_locked_dependency_binding_required"
        )
    runtime_envelope_artifact = runtime_envelopes[0]
    if runtime_envelope_artifact.source.path != roles["runtime_envelope_artifact"].path:
        raise ValueError("runtime envelope role differs from its locked dependency")

    inventory_receipt_source = BoundJsonSource.bind(
        roles["inventory_source_artifact"].path,
        semantic_sha256=authority.inventory.source_receipt_sha256,
    )
    if inventory_receipt_source.path != authority.inventory_receipt_path:
        raise ValueError(
            "assignment inventory receipt is not the raw capacity inventory receipt"
        )

    method = assignment.work_item.cell.identity.method
    trainable_authority: TrainablePlanAuthorityBinding | None = None
    prepared_release_sha256: str | None = None
    if method in {"tts", "l0"}:
        outer = roles.get("trainable_plan_authority_binding")
        release = roles.get("prepared_model_content_release_manifest")
        if outer is None:
            raise DispatchBundleMaterializationBlocked(
                "trainable_plan_raw_authority_unavailable"
            )
        if release is None:
            raise DispatchBundleMaterializationBlocked(
                "prepared_model_content_release_manifest_pin_unavailable"
            )
        decoded = trainable_plan_authority_binding_from_dict(outer.load())
        replayed = replay_trainable_plan_authority(decoded)
        if replayed.binding != decoded:
            raise ValueError("trainable-plan binding differs from raw replay")
        trainable_authority = replayed.binding
        _require_exact_raw_binding(
            model_lock_source,
            trainable_authority.model_lock,
            label="trainable-plan model lock",
        )
        _require_exact_raw_binding(
            run_config_source,
            trainable_authority.run_config,
            label="trainable-plan RunConfig",
        )
        _require_exact_raw_binding(
            split_source,
            trainable_authority.split,
            label="trainable-plan split",
        )
        prepared_release_sha256 = (
            trainable_authority.prepared_model_content_authority.release_manifest_sha256
        )
        prepared_release_source = BoundJsonSource.bind(
            release.path,
            semantic_sha256=prepared_release_sha256,
        )
        _require_exact_raw_binding(
            prepared_release_source,
            trainable_authority.prepared_model_content_authority.manifest,
            label="prepared model content release manifest",
        )
    elif method in {"target_only", "static"}:
        if (
            "trainable_plan_authority_binding" in roles
            or "prepared_model_content_release_manifest" in roles
        ):
            raise ValueError(
                "Target-only/Static materialization cannot carry trainable authority"
            )
    else:
        raise DispatchBundleMaterializationBlocked(
            "current_release_core_trainable_plan_method_required"
        )

    failure_authority = None
    failure_source = roles.get("failure_injection_authority_plan")
    failure_cell = assignment.work_item.cell.identity.task == "failure_injection"
    if failure_cell:
        if failure_source is None:
            raise DispatchBundleMaterializationBlocked(
                "failure_injection_raw_plan_authority_required"
            )
        failure_authority = bind_failure_injection_authority(
            failure_source.path,
            registry=authority.registry,
        )
        if failure_authority.cell_id != cell_id:
            raise ValueError("failure authority names another assignment")
    elif failure_source is not None:
        raise ValueError("non-failure assignment cannot carry failure authority")

    # Construction placeholders are existing, immutable nonce bytes.  The
    # materialization-only replay never reads these two fields; finalization
    # replaces both with reducer-owned server-launch and plan-summary sources.
    provisional_plan_sha256 = content_sha256(
        {
            "protocol": "industrial_execution_plan_materialization_pending.v1",
            "assignment_sha256": assignment.assignment_id,
            "run_nonce_sha256": nonce.run_nonce_sha256,
        }
    )
    provisional = IndustrialAssignmentExecutionBundle(
        schema_version=5,
        kind="industrial_assignment_execution_bundle",
        assignment_sha256=assignment.assignment_id,
        cell_id=cell_id,
        execution_plan_sha256=provisional_plan_sha256,
        run_nonce_sha256=nonce.run_nonce_sha256,
        output_root=bound_assignment.output_root,
        registry=authority.registry_source,
        inventory=authority.inventory_source,
        interference_envelope=authority.interference_envelope_source,
        interference_source_receipt=authority.interference_receipt_source,
        interference_calibration_authority=(
            authority.interference_calibration_authority
        ),
        budget_plan=authority.budget_plan_source,
        budget_policy=authority.budget_policy_source,
        budget_load_bindings=authority.budget_load_sources,
        capacity_envelope=authority.capacity_envelope_source,
        capacity_source_manifest=authority.capacity_manifest_source,
        capacity_verification_receipt=authority.capacity_verification_source,
        dependency_receipts=authority.dependency_receipt_sources,
        activation=authority.activation_source,
        activation_runtime=authority.activation_runtime_source,
        activation_split=authority.activation_split_source,
        dispatch_context=authority.dispatch_context_source,
        dispatch_plan=authority.dispatch_plan_source,
        topology_receipts=topology_source,
        production_load=production_load_source,
        itl_timestamp_plan=itl_timestamp_plan_source,
        itl_timestamp_plan_sha256=itl_timestamp_plan_sha256,
        itl_timestamp_producer_sha256=itl_timestamp_producer_sha256,
        run_config=run_config_source,
        server_launch=nonce_source,
        execution_plan_summary=nonce_source,
        dependency_artifacts=dependency_artifacts,
        split_artifact=BoundExecutionArtifact(
            name="split", experiment=None, source=split_source
        ),
        sampling_artifact=BoundExecutionArtifact(
            name="sampling", experiment=None, source=sampling_source
        ),
        model_lock_artifact=BoundExecutionArtifact(
            name="model-lock", experiment=None, source=model_lock_source
        ),
        prepared_models=prepared_models_source,
        trainable_plan_authority=trainable_authority,
        failure_injection_authority=failure_authority,
        prepared_model_content_release_manifest_sha256=prepared_release_sha256,
        compile_cache_plan=compile_source,
        inventory_source_artifact=BoundExecutionArtifact(
            name="gpu_inventory_source_receipt",
            experiment=None,
            source=inventory_receipt_source,
        ),
        runtime_envelope_artifact=runtime_envelope_artifact,
        execution_policy=execution_policy_source,
    )
    return provisional, launch_policy, launch_policy_source, nonce_source


def _preflight_publication_directory(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_absolute():
        raise ValueError("bundle publication directory must be absolute and resolved")
    if output.is_symlink():
        raise DispatchBundleMaterializationBlocked(
            "fresh_bundle_publication_directory_required"
        )
    if output.resolve() != output:
        raise ValueError("bundle publication directory must be absolute and resolved")
    parent = output.parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or parent.resolve() != parent
        or output.exists()
    ):
        raise DispatchBundleMaterializationBlocked(
            "fresh_bundle_publication_directory_required"
        )
    return output


def _open_stable_directory(path: Path, *, label: str) -> int:
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ValueError(f"{label} must be an absolute resolved non-symlink directory")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} cannot be opened safely") from error
    opened = os.fstat(descriptor)
    current = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or opened.st_dev != current.st_dev
        or opened.st_ino != current.st_ino
        or opened.st_uid != os.geteuid()
        or opened.st_mode & 0o077
    ):
        os.close(descriptor)
        raise RuntimeError(f"{label} changed while it was opened")
    return descriptor


def _create_fresh_directory(path: Path, *, label: str) -> None:
    parent_descriptor = _open_stable_directory(path.parent, label=f"{label} parent")
    try:
        if os.path.lexists(path):
            raise DispatchBundleMaterializationBlocked(
                "fresh_bundle_publication_directory_required"
            )
        try:
            os.mkdir(path.name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError as error:
            raise DispatchBundleMaterializationBlocked(
                "fresh_bundle_publication_directory_required"
            ) from error
        child_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            child_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            child_flags |= os.O_NOFOLLOW
        child_descriptor = os.open(
            path.name,
            child_flags,
            dir_fd=parent_descriptor,
        )
        try:
            opened = os.fstat(child_descriptor)
            current = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(current.st_mode)
                or opened.st_dev != current.st_dev
                or opened.st_ino != current.st_ino
            ):
                raise RuntimeError(f"{label} changed during creation")
        finally:
            os.close(child_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_exclusive_bound_json(path: Path, value: object) -> None:
    canonical = _canonical_json_bytes(value)
    body = canonical + b"\n"
    digest = hashlib.sha256(canonical).hexdigest()
    directory_descriptor = _open_stable_directory(
        path.parent,
        label="bundle publication directory",
    )

    def write_one(name: str, payload: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise RuntimeError(
                    "published bundle member is not a fresh regular file"
                )
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)

    try:
        write_one(path.name, body)
        write_one(f"{path.name}.sha256", f"{digest}\n".encode("ascii"))
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def materialize_dispatch_execution_bundles(
    request_path: str | Path,
    *,
    output_directory: str | Path,
) -> Path:
    """Construct and publish a complete schema-v5 bundle set manifest-last."""

    output = _preflight_publication_directory(output_directory)
    bound = bind_dispatch_bundle_materialization_inputs(request_path)
    authority = _reconstruct_materialization_authority(bound)
    assignments = _dispatch_assignments(bound.dispatch_plan.load())
    assignment_by_cell = {
        assignment.work_item.item_id: assignment for assignment in assignments
    }

    provisional_rows: list[
        tuple[
            IndustrialAssignmentExecutionBundle,
            AssignmentLaunchMaterializationPolicy,
            BoundJsonSource,
            BoundJsonSource,
        ]
    ] = []
    for row in bound.assignments:
        evidence_root = Path(row.output_root)
        if (
            output == evidence_root
            or output.is_relative_to(evidence_root)
            or evidence_root.is_relative_to(output)
        ):
            raise ValueError(
                "bundle publication and registered evidence roots must be disjoint"
            )
        provisional = _materialize_assignment_provisional(
            assignment=assignment_by_cell[row.cell_id],
            bound_assignment=row,
            authority=authority,
        )
        provisional_rows.append(provisional)

    render_roots = tuple(
        output / f"assignment-{provisional[0].cell_id}-runtime"
        for provisional in provisional_rows
    )
    # Replay every raw/source gate for every assignment before creating the
    # publication directory or rendering the first runtime artifact.
    for provisional, render_root in zip(
        provisional_rows,
        render_roots,
        strict=True,
    ):
        provisional[0].preflight_execution_plan_materialization(
            provisional[1],
            render_root=render_root,
        )

    _create_fresh_directory(output, label="bundle publication directory")
    for provisional, render_root in zip(
        provisional_rows,
        render_roots,
        strict=True,
    ):
        _create_fresh_directory(render_root, label="assignment render root")
        method = _strict_text(
            "assignment render method",
            assignment_by_cell[provisional[0].cell_id].work_item.cell.identity.method,
        )
        _create_fresh_directory(
            render_root / method,
            label="assignment method render root",
        )

    plans = []
    for provisional, render_root in zip(
        provisional_rows,
        render_roots,
        strict=True,
    ):
        method_root = (
            render_root
            / assignment_by_cell[provisional[0].cell_id].work_item.cell.identity.method
        )
        method_descriptor = _open_stable_directory(
            method_root,
            label="assignment method render root",
        )
        os.close(method_descriptor)
        plans.append(
            provisional[0].reconstruct_execution_plan_for_materialization(
                provisional[1],
                render_root=render_root,
            )
        )
        method_descriptor = _open_stable_directory(
            method_root,
            label="assignment method render root",
        )
        os.close(method_descriptor)

    receipts: list[MaterializedAssignmentBundleReceipt] = []
    for (
        provisional,
        launch_policy,
        launch_policy_source,
        nonce_source,
    ), plan, render_root in zip(provisional_rows, plans, render_roots, strict=True):
        prefix = f"assignment-{provisional.cell_id}"
        launch_path = output / f"{prefix}-server-launch.json"
        summary_path = output / f"{prefix}-execution-plan.json"
        bundle_path = output / f"{prefix}-bundle.json"
        _write_exclusive_bound_json(
            launch_path, server_launch_to_dict(plan.server_launch)
        )
        _write_exclusive_bound_json(summary_path, plan.to_dict())
        launch_source = BoundJsonSource.bind(launch_path)
        summary_source = BoundJsonSource.bind(summary_path, semantic_sha256=plan.sha256)
        final = finalize_materialized_execution_bundle(
            provisional,
            launch_policy=launch_policy,
            render_root=render_root,
            server_launch=launch_source,
            execution_plan_summary=summary_source,
        )
        if final.execution_plan_sha256 != plan.sha256:
            raise RuntimeError("bundle finalizer returned another execution plan")
        _write_exclusive_bound_json(bundle_path, final.to_dict())
        bundle_source = BoundJsonSource.bind(bundle_path, semantic_sha256=final.sha256)
        loaded = IndustrialAssignmentExecutionBundle.load(bundle_path)
        if loaded != final:
            raise RuntimeError("published execution bundle changed during reload")
        receipts.append(
            MaterializedAssignmentBundleReceipt(
                assignment_sha256=final.assignment_sha256,
                cell_id=final.cell_id,
                run_nonce_sha256=final.run_nonce_sha256,
                execution_plan_sha256=final.execution_plan_sha256,
                launch_policy=launch_policy_source,
                run_nonce_receipt=nonce_source,
                bundle=bundle_source,
            )
        )
    manifest = DispatchExecutionBundleManifest(
        schema_version=1,
        kind=_MANIFEST_KIND,
        bundle_schema_version=5,
        materialization_inputs_sha256=bound.sha256,
        request=bound.request,
        dispatch_plan=authority.dispatch_plan_source,
        assignments=tuple(sorted(receipts, key=lambda receipt: receipt.cell_id)),
    )
    manifest_path = output / "dispatch-execution-bundle-manifest.json"
    # The sidecar written by this final call is the publication commit marker.
    # An interrupted directory is retained as evidence but is never consumable;
    # retry requires a distinct fresh directory rather than destructive cleanup.
    _write_exclusive_bound_json(manifest_path, manifest.to_dict())
    directory_descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return manifest_path


def load_materialized_dispatch_execution_bundle_publication(
    manifest_path: str | Path,
) -> MaterializedDispatchExecutionBundlePublication:
    """Reopen manifest membership and all source-owned construction inputs."""

    unresolved_manifest = Path(manifest_path)
    if (
        not unresolved_manifest.is_absolute()
        or unresolved_manifest.is_symlink()
        or unresolved_manifest.resolve() != unresolved_manifest
    ):
        raise ValueError(
            "dispatch execution-bundle manifest path must be absolute and non-symlink"
        )
    manifest_source = BoundJsonSource.bind(unresolved_manifest)
    if Path(manifest_source.path).name != "dispatch-execution-bundle-manifest.json":
        raise ValueError("dispatch execution-bundle manifest name is not canonical")
    manifest = DispatchExecutionBundleManifest.from_dict(manifest_source.load())
    if manifest_source.semantic_sha256 != manifest.sha256:
        raise ValueError("dispatch execution-bundle manifest is not canonical")
    rebound = bind_dispatch_bundle_materialization_inputs(manifest.request.path)
    if (
        rebound.sha256 != manifest.materialization_inputs_sha256
        or rebound.request != manifest.request
    ):
        raise ValueError("materialization request changed after bundle publication")
    dispatch_source = BoundJsonSource.bind(
        rebound.dispatch_plan.path,
        semantic_sha256=content_sha256(rebound.dispatch_plan.load()),
    )
    if dispatch_source != manifest.dispatch_plan:
        raise ValueError("published manifest swapped its dispatch plan")
    rebound_assignments = {row.cell_id: row for row in rebound.assignments}
    rebound_pairs = tuple(
        sorted((row.cell_id, row.assignment_sha256) for row in rebound.assignments)
    )
    manifest_pairs = tuple(
        (row.cell_id, row.assignment_sha256) for row in manifest.assignments
    )
    if rebound_pairs != manifest_pairs:
        raise ValueError("published manifest assignment coverage changed")

    bundles: list[IndustrialAssignmentExecutionBundle] = []
    publication_root = Path(manifest_source.path).parent
    for receipt in manifest.assignments:
        bound_assignment = rebound_assignments[receipt.cell_id]
        source_by_role = {role: source for role, source in bound_assignment.sources}
        if (
            receipt.launch_policy != source_by_role["launch_policy"]
            or receipt.run_nonce_receipt != source_by_role["run_nonce_receipt"]
        ):
            raise ValueError("published manifest swapped construction authority")
        AssignmentLaunchMaterializationPolicy.from_dict(receipt.launch_policy.load())
        nonce = AssignmentRunNonceReceipt.from_dict(receipt.run_nonce_receipt.load())
        expected_prefix = f"assignment-{receipt.cell_id}"
        if (
            Path(receipt.bundle.path).parent != publication_root
            or Path(receipt.bundle.path).name != f"{expected_prefix}-bundle.json"
        ):
            raise ValueError("published bundle is outside its manifest directory")
        bundle = IndustrialAssignmentExecutionBundle.load(receipt.bundle.path)
        _require_published_bundle_itl_source(bundle, source_by_role)
        if (
            Path(bundle.server_launch.path).parent != publication_root
            or Path(bundle.server_launch.path).name
            != f"{expected_prefix}-server-launch.json"
            or Path(bundle.execution_plan_summary.path).parent != publication_root
            or Path(bundle.execution_plan_summary.path).name
            != f"{expected_prefix}-execution-plan.json"
        ):
            raise ValueError("published plan sources are outside their manifest")
        if (
            receipt.bundle.canonical_sha256 != bundle.sha256
            or receipt.bundle.semantic_sha256 != bundle.sha256
            or receipt.assignment_sha256 != bundle.assignment_sha256
            or receipt.assignment_sha256 != bound_assignment.assignment_sha256
            or receipt.cell_id != bundle.cell_id
            or receipt.run_nonce_sha256 != bundle.run_nonce_sha256
            or receipt.run_nonce_sha256 != nonce.run_nonce_sha256
            or receipt.execution_plan_sha256 != bundle.execution_plan_sha256
            or bundle.dispatch_plan != manifest.dispatch_plan
        ):
            raise ValueError("published bundle differs from manifest membership")
        receipt.bundle.load()
        bundles.append(bundle)
    return MaterializedDispatchExecutionBundlePublication(
        manifest=manifest,
        bundles=tuple(bundles),
    )


def _require_published_bundle_itl_source(
    bundle: IndustrialAssignmentExecutionBundle,
    source_by_role: dict[str, BoundJsonSource],
) -> None:
    """Bind the published optional ITL source to its exact request path."""

    if bundle.itl_timestamp_plan != source_by_role.get("itl_timestamp_authority_plan"):
        raise ValueError(
            "published bundle swapped its path-bound ITL construction source"
        )


def load_materialized_dispatch_execution_bundles(
    manifest_path: str | Path,
) -> tuple[IndustrialAssignmentExecutionBundle, ...]:
    """Return exact members only after reopening their commit manifest."""

    return load_materialized_dispatch_execution_bundle_publication(
        manifest_path
    ).bundles


__all__ = [
    "AssignmentLaunchMaterializationPolicy",
    "AssignmentRunNonceReceipt",
    "AssignmentRuntimeSourcePaths",
    "BoundAssignmentRuntimeSources",
    "BoundDispatchBundleMaterializationInputs",
    "DependencyArtifactPath",
    "DispatchBundleMaterializationBlocked",
    "DispatchBundleMaterializationRequest",
    "DispatchExecutionBundleManifest",
    "MaterializedAssignmentBundleReceipt",
    "MaterializedDispatchExecutionBundlePublication",
    "RawArtifactRole",
    "bind_dispatch_bundle_materialization_inputs",
    "load_materialized_dispatch_execution_bundle_publication",
    "load_materialized_dispatch_execution_bundles",
    "materialize_dispatch_execution_bundles",
]
