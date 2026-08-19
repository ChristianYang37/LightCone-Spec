"""Source-owned diagnostic activation for registered COMPILE work.

This authority deliberately stops short of execution.  It reopens the raw,
sidecar-bound :class:`CompileAssignmentPlan`, checks its embedded assignment
against the immutable registry and dependency prefix, and records the one
registered COMPILE cell as a diagnostic candidate.  The current release has no
trusted GPU compile actuator, so the same artifact always carries a formal
``BLOCKED`` disposition and cannot be consumed as scheduler activation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Self

from lightcone_spec.experiments.registry import (
    INDUSTRIAL_EXPERIMENT_ORDER,
    CellStatus,
    ExperimentReceipt,
    ExperimentRegistry,
    WorkloadClass,
    content_sha256,
)
from lightcone_spec.runtime.compile_runner import (
    RELEASE_COMPILE_RUNNER_UNAVAILABLE,
    CompileAssignmentPlan,
)

COMPILE_DIAGNOSTIC_ACTIVATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "industrial_compile_diagnostic_activation_reducer",
        "inputs": (
            "exact_registry",
            "complete_validated_dependency_receipt_prefix",
            "content_bound_runtime_identity",
            "content_bound_split_identity",
            "path_and_sidecar_bound_compile_assignment_plan",
        ),
        "plan_revalidation": "reopen_all_compile_assignment_inputs",
        "diagnostic_candidate_only": True,
        "scheduler_activation_authority": False,
        "formal_status": "BLOCKED",
        "formal_reason_code": RELEASE_COMPILE_RUNNER_UNAVAILABLE,
    }
)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be non-empty single-line text")
    return value


def _strict_object(label: str, value: object, fields: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a JSON object with string keys")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise ValueError(
            f"{label} fields differ: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return value


@dataclass(frozen=True)
class CompileDiagnosticActivationAuthority:
    schema_version: int
    kind: str
    protocol_sha256: str
    registry_sha256: str
    experiment: str
    cell_id: str
    runtime_sha256: str
    split_sha256: str
    dependency_receipt_sha256s: tuple[str, ...]
    compile_assignment_plan_path: str
    compile_assignment_plan_sha256: str
    assignment_contract_sha256: str
    compile_key_sha256: str
    physical_assignment_sha256: str
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    diagnostic_status: str
    diagnostic_reason_code: str
    formal_status: str
    formal_reason_code: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("compile diagnostic activation schema is unsupported")
        if self.kind != "industrial_compile_diagnostic_activation_authority":
            raise ValueError("compile diagnostic activation kind is invalid")
        if self.protocol_sha256 != COMPILE_DIAGNOSTIC_ACTIVATION_PROTOCOL_SHA256:
            raise ValueError("compile diagnostic activation uses another protocol")
        for label, value in (
            ("registry", self.registry_sha256),
            ("cell", self.cell_id),
            ("runtime", self.runtime_sha256),
            ("split", self.split_sha256),
            ("compile assignment plan", self.compile_assignment_plan_sha256),
            ("assignment contract", self.assignment_contract_sha256),
            ("compile key", self.compile_key_sha256),
            ("physical assignment", self.physical_assignment_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(label, value)
        _require_text("experiment", self.experiment)
        for digest in self.dependency_receipt_sha256s:
            _require_sha256("dependency receipt", digest)
        if len(set(self.dependency_receipt_sha256s)) != len(
            self.dependency_receipt_sha256s
        ):
            raise ValueError("compile diagnostic dependency receipts are duplicated")
        plan_path = Path(
            _require_text(
                "compile assignment plan path", self.compile_assignment_plan_path
            )
        )
        if not plan_path.is_absolute() or plan_path != plan_path.resolve(strict=False):
            raise ValueError(
                "compile assignment plan path must be absolute and normalized"
            )
        if not self.gpu_uuids or len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError(
                "compile diagnostic GPU UUIDs must be unique and non-empty"
            )
        for gpu_uuid in self.gpu_uuids:
            _require_text("compile diagnostic GPU UUID", gpu_uuid)
        if (
            self.diagnostic_status != "READY_DIAGNOSTIC"
            or self.diagnostic_reason_code != "raw_compile_assignment_revalidated"
            or self.formal_status != "BLOCKED"
            or self.formal_reason_code != RELEASE_COMPILE_RUNNER_UNAVAILABLE
        ):
            raise ValueError("compile diagnostic/formal disposition is not exact")

    def _payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "registry_sha256": self.registry_sha256,
            "experiment": self.experiment,
            "cell_id": self.cell_id,
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "dependency_receipt_sha256s": list(self.dependency_receipt_sha256s),
            "compile_assignment_plan_path": self.compile_assignment_plan_path,
            "compile_assignment_plan_sha256": self.compile_assignment_plan_sha256,
            "assignment_contract_sha256": self.assignment_contract_sha256,
            "compile_key_sha256": self.compile_key_sha256,
            "physical_assignment_sha256": self.physical_assignment_sha256,
            "inventory_sha256": self.inventory_sha256,
            "gpu_uuids": list(self.gpu_uuids),
            "diagnostic_status": self.diagnostic_status,
            "diagnostic_reason_code": self.diagnostic_reason_code,
            "formal_status": self.formal_status,
            "formal_reason_code": self.formal_reason_code,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self._payload_dict())

    def to_dict(self) -> dict[str, object]:
        return {"artifact_sha256": self.sha256, **self._payload_dict()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = frozenset(
            {
                "artifact_sha256",
                "schema_version",
                "kind",
                "protocol_sha256",
                "registry_sha256",
                "experiment",
                "cell_id",
                "runtime_sha256",
                "split_sha256",
                "dependency_receipt_sha256s",
                "compile_assignment_plan_path",
                "compile_assignment_plan_sha256",
                "assignment_contract_sha256",
                "compile_key_sha256",
                "physical_assignment_sha256",
                "inventory_sha256",
                "gpu_uuids",
                "diagnostic_status",
                "diagnostic_reason_code",
                "formal_status",
                "formal_reason_code",
            }
        )
        row = _strict_object("compile diagnostic activation", value, fields)
        receipt_values = row["dependency_receipt_sha256s"]
        gpu_values = row["gpu_uuids"]
        if type(receipt_values) is not list or type(gpu_values) is not list:
            raise TypeError("compile diagnostic tuple fields must be JSON arrays")
        artifact = cls(
            schema_version=row["schema_version"],
            kind=row["kind"],
            protocol_sha256=row["protocol_sha256"],
            registry_sha256=row["registry_sha256"],
            experiment=row["experiment"],
            cell_id=row["cell_id"],
            runtime_sha256=row["runtime_sha256"],
            split_sha256=row["split_sha256"],
            dependency_receipt_sha256s=tuple(receipt_values),
            compile_assignment_plan_path=row["compile_assignment_plan_path"],
            compile_assignment_plan_sha256=row["compile_assignment_plan_sha256"],
            assignment_contract_sha256=row["assignment_contract_sha256"],
            compile_key_sha256=row["compile_key_sha256"],
            physical_assignment_sha256=row["physical_assignment_sha256"],
            inventory_sha256=row["inventory_sha256"],
            gpu_uuids=tuple(gpu_values),
            diagnostic_status=row["diagnostic_status"],
            diagnostic_reason_code=row["diagnostic_reason_code"],
            formal_status=row["formal_status"],
            formal_reason_code=row["formal_reason_code"],
        )
        if row["artifact_sha256"] != artifact.sha256:
            raise ValueError("compile diagnostic activation SHA-256 mismatch")
        return artifact


def _validated_dependency_prefix(
    registry: ExperimentRegistry,
    *,
    experiment: str,
    dependency_receipts: Sequence[ExperimentReceipt],
) -> tuple[ExperimentReceipt, ...]:
    if experiment not in INDUSTRIAL_EXPERIMENT_ORDER:
        raise ValueError("compile diagnostic experiment is not registered")
    expected_names = INDUSTRIAL_EXPERIMENT_ORDER[
        : INDUSTRIAL_EXPERIMENT_ORDER.index(experiment)
    ]
    receipts = tuple(dependency_receipts)
    if any(type(receipt) is not ExperimentReceipt for receipt in receipts):
        raise TypeError("compile diagnostic dependencies must be exact receipts")
    validated = registry.validate_receipts(receipts)
    if set(validated) != set(expected_names):
        raise ValueError(
            "compile diagnostic activation requires the complete dependency receipt prefix"
        )
    return tuple(validated[name] for name in expected_names)


def materialize_compile_diagnostic_activation(
    registry: ExperimentRegistry,
    *,
    experiment: str,
    dependency_receipts: Sequence[ExperimentReceipt] = (),
    runtime_sha256: str,
    split_sha256: str,
    compile_assignment_plan_path: str | Path,
) -> CompileDiagnosticActivationAuthority:
    """Reopen raw COMPILE authority without granting execution authority."""

    if type(registry) is not ExperimentRegistry:
        raise TypeError("compile diagnostic activation requires an exact registry")
    _require_sha256("compile diagnostic runtime", runtime_sha256)
    _require_sha256("compile diagnostic split", split_sha256)
    receipts = _validated_dependency_prefix(
        registry,
        experiment=experiment,
        dependency_receipts=dependency_receipts,
    )
    plan_path = Path(compile_assignment_plan_path)
    plan = CompileAssignmentPlan.load(plan_path)
    assignment, _cache_plan, _prewarm, _launch = plan.revalidate()
    if (
        assignment.registry_sha256 != registry.sha256
        or assignment.runtime_sha256 != runtime_sha256
        or assignment.split_sha256 != split_sha256
    ):
        raise ValueError(
            "compile assignment differs from registry/runtime/split authority"
        )
    cell = next(
        (row for row in registry.cells if row.cell_id == assignment.cell_id),
        None,
    )
    if cell is None:
        raise ValueError("compile assignment cell is absent from the registry")
    if cell.identity.experiment != experiment:
        raise ValueError("compile assignment belongs to another experiment")
    if (
        cell.status is not CellStatus.UNMEASURED
        or cell.resources.workload_class is not WorkloadClass.COMPILE
    ):
        raise ValueError(
            "compile assignment does not name runnable registered COMPILE work"
        )
    if assignment.gpu_uuids != cell.resources.gpu_uuids:
        raise ValueError(
            "compile assignment GPU identity differs from the registry cell"
        )
    return CompileDiagnosticActivationAuthority(
        schema_version=1,
        kind="industrial_compile_diagnostic_activation_authority",
        protocol_sha256=COMPILE_DIAGNOSTIC_ACTIVATION_PROTOCOL_SHA256,
        registry_sha256=registry.sha256,
        experiment=experiment,
        cell_id=cell.cell_id,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        dependency_receipt_sha256s=tuple(receipt.sha256 for receipt in receipts),
        compile_assignment_plan_path=str(plan_path.resolve()),
        compile_assignment_plan_sha256=plan.sha256,
        assignment_contract_sha256=assignment.sha256,
        compile_key_sha256=plan.compile_key_sha256,
        physical_assignment_sha256=plan.physical_assignment_sha256,
        inventory_sha256=plan.inventory_sha256,
        gpu_uuids=plan.gpu_uuids,
        diagnostic_status="READY_DIAGNOSTIC",
        diagnostic_reason_code="raw_compile_assignment_revalidated",
        formal_status="BLOCKED",
        formal_reason_code=RELEASE_COMPILE_RUNNER_UNAVAILABLE,
    )


def verify_compile_diagnostic_activation(
    registry: ExperimentRegistry,
    authority: CompileDiagnosticActivationAuthority,
    *,
    dependency_receipts: Sequence[ExperimentReceipt] = (),
) -> None:
    """Replay all raw sources and reject edited or stale diagnostic authority."""

    if type(authority) is not CompileDiagnosticActivationAuthority:
        raise TypeError("compile diagnostic verifier requires the exact authority")
    expected = materialize_compile_diagnostic_activation(
        registry,
        experiment=authority.experiment,
        dependency_receipts=dependency_receipts,
        runtime_sha256=authority.runtime_sha256,
        split_sha256=authority.split_sha256,
        compile_assignment_plan_path=authority.compile_assignment_plan_path,
    )
    if authority != expected:
        raise ValueError(
            "compile diagnostic activation is not the exact reducer output"
        )


__all__ = [
    "COMPILE_DIAGNOSTIC_ACTIVATION_PROTOCOL_SHA256",
    "CompileDiagnosticActivationAuthority",
    "materialize_compile_diagnostic_activation",
    "verify_compile_diagnostic_activation",
]
