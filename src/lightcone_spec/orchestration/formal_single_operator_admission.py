"""Narrow launch admission for the trusted single-operator workflow.

This artifact is deliberately not a deployment attestation or a GPU-hour
budget receipt.  It reopens the source-owned physical plan, launch manifest,
materialized cell, and local GPU inventory, then records the exact process
bound used by the callback-free physical runner.  The output and its one-shot
consumption live in the run-specific private directory and are published with
no-replace semantics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.orchestration.formal_physical_dispatch import (
    FormalServingRequestScheduleReceipt,
    FormalServingRunPlan,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_SINGLE_OPERATOR_MODE = "formal_single_operator_v1"
FORMAL_SINGLE_OPERATOR_PROCESS_HARD_TIMEOUT_NS = 3_600_000_000_000
FORMAL_SINGLE_OPERATOR_PROVIDER_RELEASE_GRACE_NS = 300_000_000_000
FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS = (
    "one_trusted_operator",
    "controlled_local_workspace",
    "source_owned_plan_launch_and_request_schedule",
    "run_specific_no_replace_outputs",
    "local_inventory_is_the_provider_allocation",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


FORMAL_SINGLE_OPERATOR_ADMISSION_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_admission_protocol",
        "mode": FORMAL_SINGLE_OPERATOR_MODE,
        "trust_assumptions": FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        "sources": (
            "formal_serving_run_plan",
            "compile_launch_manifest",
            "request_schedule_materialization",
            "local_gpu_inventory",
            "code_owned_process_and_provider_bounds",
        ),
        "consumption": "one_no_replace_file_in_exact_private_run_root",
        "adversarial_attestation_claim": False,
    }
)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"{label} must be canonical text")
    return value


def _strict_object(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _private_root(plan: FormalServingRunPlan) -> Path:
    root = Path(plan.private_output_root)
    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise ValueError("single-operator private output root is invalid")
    return root


@dataclass(frozen=True)
class FormalSingleOperatorAdmission:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_admission"]
    protocol_sha256: str
    mode: Literal["formal_single_operator_v1"]
    trust_assumptions: tuple[str, ...]
    plan: CanonicalJsonProofBinding
    launch_manifest: CanonicalJsonProofBinding
    request_schedule: CanonicalJsonProofBinding
    materialization: CanonicalJsonProofBinding
    inventory: CanonicalJsonProofBinding
    execution_binding_sha256: str
    registry_sha256: str
    registry_cell_id: str | None
    inventory_sha256: str
    stage: str
    materialized_cell_id: str
    topology_mode: str
    attempt_id: str
    gpu_uuids: tuple[str, ...]
    process_hard_timeout_ns: int
    provider_wave_hard_timeout_ns: int
    provider_reserved_gpu_uuids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_admission"
            or self.protocol_sha256 != FORMAL_SINGLE_OPERATOR_ADMISSION_PROTOCOL_SHA256
            or self.mode != FORMAL_SINGLE_OPERATOR_MODE
            or self.trust_assumptions != FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS
        ):
            raise ValueError("single-operator admission schema differs")
        for value in (
            self.plan,
            self.launch_manifest,
            self.request_schedule,
            self.materialization,
            self.inventory,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("single-operator admission lost a path binding")
        for label, value in (
            ("execution binding", self.execution_binding_sha256),
            ("registry", self.registry_sha256),
            ("inventory", self.inventory_sha256),
            ("materialized cell", self.materialized_cell_id),
        ):
            _require_sha256(f"single-operator admission {label}", value)
        if self.registry_cell_id is not None:
            _require_sha256(
                "single-operator admission registry cell", self.registry_cell_id
            )
        for label, value in (
            ("stage", self.stage),
            ("topology", self.topology_mode),
            ("attempt", self.attempt_id),
        ):
            _require_text(f"single-operator admission {label}", value)
        if (
            type(self.gpu_uuids) is not tuple
            or not self.gpu_uuids
            or len(self.gpu_uuids) != len(set(self.gpu_uuids))
            or type(self.provider_reserved_gpu_uuids) is not tuple
            or not self.provider_reserved_gpu_uuids
            or tuple(sorted(self.provider_reserved_gpu_uuids))
            != self.provider_reserved_gpu_uuids
            or not set(self.gpu_uuids).issubset(self.provider_reserved_gpu_uuids)
            or self.process_hard_timeout_ns
            != FORMAL_SINGLE_OPERATOR_PROCESS_HARD_TIMEOUT_NS
            or self.provider_wave_hard_timeout_ns
            != FORMAL_SINGLE_OPERATOR_PROCESS_HARD_TIMEOUT_NS
            + FORMAL_SINGLE_OPERATOR_PROVIDER_RELEASE_GRACE_NS
        ):
            raise ValueError("single-operator admission resource bounds differ")
        for value in (*self.gpu_uuids, *self.provider_reserved_gpu_uuids):
            _require_text("single-operator admission GPU UUID", value)

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "mode": self.mode,
            "trust_assumptions": list(self.trust_assumptions),
            "plan": self.plan.to_dict(),
            "launch_manifest": self.launch_manifest.to_dict(),
            "request_schedule": self.request_schedule.to_dict(),
            "materialization": self.materialization.to_dict(),
            "inventory": self.inventory.to_dict(),
            "execution_binding_sha256": self.execution_binding_sha256,
            "registry_sha256": self.registry_sha256,
            "registry_cell_id": self.registry_cell_id,
            "inventory_sha256": self.inventory_sha256,
            "stage": self.stage,
            "materialized_cell_id": self.materialized_cell_id,
            "topology_mode": self.topology_mode,
            "attempt_id": self.attempt_id,
            "gpu_uuids": list(self.gpu_uuids),
            "process_hard_timeout_ns": self.process_hard_timeout_ns,
            "provider_wave_hard_timeout_ns": self.provider_wave_hard_timeout_ns,
            "provider_reserved_gpu_uuids": list(self.provider_reserved_gpu_uuids),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "single-operator admission",
            value,
            {field.name for field in fields(cls)},
        )
        for name in (
            "plan",
            "launch_manifest",
            "request_schedule",
            "materialization",
            "inventory",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        for name in (
            "trust_assumptions",
            "gpu_uuids",
            "provider_reserved_gpu_uuids",
        ):
            array = row[name]
            if type(array) is not list:
                raise TypeError(f"single-operator admission {name} must be an array")
            row[name] = tuple(array)
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorAdmissionConsumption:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_admission_consumption"]
    protocol_sha256: str
    admission: CanonicalJsonProofBinding
    plan_sha256: str
    execution_binding_sha256: str
    materialized_cell_id: str
    attempt_id: str
    consumed_ns: int
    process_hard_timeout_ns: int
    provider_wave_hard_timeout_ns: int
    compute_gpu_count: int
    provider_reserved_gpu_count: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_admission_consumption"
            or self.protocol_sha256 != FORMAL_SINGLE_OPERATOR_ADMISSION_PROTOCOL_SHA256
            or type(self.admission) is not CanonicalJsonProofBinding
        ):
            raise ValueError("single-operator consumption schema differs")
        for label, value in (
            ("plan", self.plan_sha256),
            ("execution binding", self.execution_binding_sha256),
            ("materialized cell", self.materialized_cell_id),
        ):
            _require_sha256(f"single-operator consumption {label}", value)
        _require_text("single-operator consumption attempt", self.attempt_id)
        if (
            type(self.consumed_ns) is not int
            or self.consumed_ns < 1
            or self.process_hard_timeout_ns
            != FORMAL_SINGLE_OPERATOR_PROCESS_HARD_TIMEOUT_NS
            or self.provider_wave_hard_timeout_ns
            != FORMAL_SINGLE_OPERATOR_PROCESS_HARD_TIMEOUT_NS
            + FORMAL_SINGLE_OPERATOR_PROVIDER_RELEASE_GRACE_NS
            or type(self.compute_gpu_count) is not int
            or self.compute_gpu_count < 1
            or type(self.provider_reserved_gpu_count) is not int
            or self.provider_reserved_gpu_count < self.compute_gpu_count
        ):
            raise ValueError("single-operator consumption resource values differ")

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "admission": self.admission.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "single-operator admission consumption",
            value,
            {field.name for field in fields(cls)},
        )
        row["admission"] = CanonicalJsonProofBinding.from_dict(row["admission"])
        return cls(**row)  # type: ignore[arg-type]


def _rebuild_admission_sources(
    artifact: FormalSingleOperatorAdmission,
) -> tuple[FormalServingRunPlan, CompileLaunchManifest]:
    plan_binding = CanonicalJsonProofBinding.bind(artifact.plan.absolute_path)
    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    if plan_binding != artifact.plan or plan.sha256 != plan_binding.semantic_sha256:
        raise ValueError("single-operator admission plan changed")
    root = _private_root(plan)
    if Path(artifact.plan.absolute_path).parent != root:
        raise ValueError("single-operator admission plan leaves private root")
    launch_binding = CanonicalJsonProofBinding.bind(
        artifact.launch_manifest.absolute_path
    )
    launch = CompileLaunchManifest.from_dict(launch_binding.reopen())
    if (
        launch_binding != artifact.launch_manifest
        or launch.sha256 != launch_binding.semantic_sha256
        or artifact.launch_manifest != plan.launch_manifest
        or launch.inventory_sha256 != plan.inventory_sha256
        or launch.gpu_uuids != plan.gpu_uuids
    ):
        raise ValueError("single-operator admission launch changed")
    schedule_binding = CanonicalJsonProofBinding.bind(
        artifact.request_schedule.absolute_path
    )
    schedule = FormalServingRequestScheduleReceipt.from_dict(schedule_binding.reopen())
    if (
        schedule_binding != artifact.request_schedule
        or schedule.sha256 != schedule_binding.semantic_sha256
        or artifact.request_schedule != plan.request_schedule_receipt
        or schedule.execution_binding_sha256 != plan.execution_binding_sha256
        or schedule.materialized_cell_id != plan.materialized_cell_id
        or schedule.topology_mode != plan.topology_mode
    ):
        raise ValueError("single-operator admission request schedule changed")
    from lightcone_spec.experiments.formal_registry import (
        stage_materialization_receipt_from_dict,
    )

    materialization_binding = CanonicalJsonProofBinding.bind(
        artifact.materialization.absolute_path
    )
    materialization = stage_materialization_receipt_from_dict(
        materialization_binding.reopen()
    )
    cells = tuple(
        row for row in materialization.cells if row.cell_id == plan.materialized_cell_id
    )
    if (
        materialization_binding != artifact.materialization
        or schedule.materialization != artifact.materialization
        or len(cells) != 1
        or cells[0].stage != plan.stage
    ):
        raise ValueError("single-operator admission materialized cell changed")
    registry = build_industrial_registry()
    registry_cell_id = dict(cells[0].dimensions).get("registry_cell_id")
    if registry_cell_id is not None:
        matches = tuple(
            row
            for row in registry.cells_for(plan.stage)
            if row.cell_id == registry_cell_id
        )
        if len(matches) != 1:
            raise ValueError("single-operator admission registry cell is not exact")
    inventory_binding = CanonicalJsonProofBinding.bind(artifact.inventory.absolute_path)
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    if (
        inventory_binding != artifact.inventory
        or inventory.sha256 != inventory_binding.semantic_sha256
        or inventory.sha256 != plan.inventory_sha256
        or set(plan.gpu_uuids) - {row.uuid for row in inventory.devices}
    ):
        raise ValueError("single-operator admission inventory changed")
    expected = FormalSingleOperatorAdmission(
        schema_version=1,
        kind="formal_single_operator_admission",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_ADMISSION_PROTOCOL_SHA256,
        mode=FORMAL_SINGLE_OPERATOR_MODE,
        trust_assumptions=FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        plan=plan_binding,
        launch_manifest=launch_binding,
        request_schedule=schedule_binding,
        materialization=materialization_binding,
        inventory=inventory_binding,
        execution_binding_sha256=plan.execution_binding_sha256,
        registry_sha256=registry.sha256,
        registry_cell_id=registry_cell_id,
        inventory_sha256=inventory.sha256,
        stage=plan.stage,
        materialized_cell_id=plan.materialized_cell_id,
        topology_mode=plan.topology_mode,
        attempt_id=plan.native_terminal_binding.attempt_id,
        gpu_uuids=plan.gpu_uuids,
        process_hard_timeout_ns=FORMAL_SINGLE_OPERATOR_PROCESS_HARD_TIMEOUT_NS,
        provider_wave_hard_timeout_ns=(
            FORMAL_SINGLE_OPERATOR_PROCESS_HARD_TIMEOUT_NS
            + FORMAL_SINGLE_OPERATOR_PROVIDER_RELEASE_GRACE_NS
        ),
        provider_reserved_gpu_uuids=tuple(
            sorted(row.uuid for row in inventory.devices)
        ),
    )
    if artifact != expected:
        raise ValueError("single-operator admission differs from rebuilt sources")
    return plan, launch


def publish_formal_single_operator_admission(
    *,
    plan_path: str | Path,
    inventory_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish the one trusted-operator admission at its fixed run-root path."""

    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    if plan.sha256 != plan_binding.semantic_sha256:
        raise ValueError("single-operator plan semantic identity differs")
    schedule = FormalServingRequestScheduleReceipt.from_dict(
        plan.request_schedule_receipt.reopen()
    )
    inventory_binding = CanonicalJsonProofBinding.bind(inventory_path)
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    from lightcone_spec.experiments.formal_registry import (
        stage_materialization_receipt_from_dict,
    )

    materialization = stage_materialization_receipt_from_dict(
        schedule.materialization.reopen()
    )
    cells = tuple(
        row for row in materialization.cells if row.cell_id == plan.materialized_cell_id
    )
    if len(cells) != 1:
        raise ValueError("single-operator admission lacks one materialized cell")
    registry = build_industrial_registry()
    registry_cell_id = dict(cells[0].dimensions).get("registry_cell_id")
    artifact = FormalSingleOperatorAdmission(
        schema_version=1,
        kind="formal_single_operator_admission",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_ADMISSION_PROTOCOL_SHA256,
        mode=FORMAL_SINGLE_OPERATOR_MODE,
        trust_assumptions=FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        plan=plan_binding,
        launch_manifest=plan.launch_manifest,
        request_schedule=plan.request_schedule_receipt,
        materialization=schedule.materialization,
        inventory=inventory_binding,
        execution_binding_sha256=plan.execution_binding_sha256,
        registry_sha256=registry.sha256,
        registry_cell_id=registry_cell_id,
        inventory_sha256=inventory.sha256,
        stage=plan.stage,
        materialized_cell_id=plan.materialized_cell_id,
        topology_mode=plan.topology_mode,
        attempt_id=plan.native_terminal_binding.attempt_id,
        gpu_uuids=plan.gpu_uuids,
        process_hard_timeout_ns=FORMAL_SINGLE_OPERATOR_PROCESS_HARD_TIMEOUT_NS,
        provider_wave_hard_timeout_ns=(
            FORMAL_SINGLE_OPERATOR_PROCESS_HARD_TIMEOUT_NS
            + FORMAL_SINGLE_OPERATOR_PROVIDER_RELEASE_GRACE_NS
        ),
        provider_reserved_gpu_uuids=tuple(
            sorted(row.uuid for row in inventory.devices)
        ),
    )
    _rebuild_admission_sources(artifact)
    destination = _private_root(plan) / "formal-single-operator-admission.json"
    publish_canonical_json_no_replace(destination, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(
        destination, semantic_sha256=artifact.sha256
    )
    if FormalSingleOperatorAdmission.from_dict(binding.reopen()) != artifact:
        raise RuntimeError("single-operator admission changed during publication")
    return binding


def validate_formal_single_operator_admission(
    admission_path: str | Path,
    *,
    plan_path: str | Path,
) -> FormalSingleOperatorAdmission:
    binding = CanonicalJsonProofBinding.bind(admission_path)
    artifact = FormalSingleOperatorAdmission.from_dict(binding.reopen())
    plan, _launch = _rebuild_admission_sources(artifact)
    expected_path = _private_root(plan) / "formal-single-operator-admission.json"
    if (
        binding.semantic_sha256 != artifact.sha256
        or Path(admission_path) != expected_path
        or artifact.plan != CanonicalJsonProofBinding.bind(plan_path)
    ):
        raise ValueError("single-operator admission path or plan differs")
    return artifact


def consume_formal_single_operator_admission(
    artifact: FormalSingleOperatorAdmission,
    *,
    consumed_ns: int,
) -> CanonicalJsonProofBinding:
    plan, _launch = _rebuild_admission_sources(artifact)
    admission = CanonicalJsonProofBinding.bind(
        _private_root(plan) / "formal-single-operator-admission.json",
        semantic_sha256=artifact.sha256,
    )
    consumption = FormalSingleOperatorAdmissionConsumption(
        schema_version=1,
        kind="formal_single_operator_admission_consumption",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_ADMISSION_PROTOCOL_SHA256,
        admission=admission,
        plan_sha256=plan.sha256,
        execution_binding_sha256=plan.execution_binding_sha256,
        materialized_cell_id=plan.materialized_cell_id,
        attempt_id=plan.native_terminal_binding.attempt_id,
        consumed_ns=consumed_ns,
        process_hard_timeout_ns=artifact.process_hard_timeout_ns,
        provider_wave_hard_timeout_ns=artifact.provider_wave_hard_timeout_ns,
        compute_gpu_count=len(artifact.gpu_uuids),
        provider_reserved_gpu_count=len(artifact.provider_reserved_gpu_uuids),
    )
    destination = _private_root(plan) / "formal-single-operator-admission-consumed.json"
    publish_canonical_json_no_replace(destination, consumption.to_dict())
    binding = CanonicalJsonProofBinding.bind(
        destination, semantic_sha256=consumption.sha256
    )
    if (
        FormalSingleOperatorAdmissionConsumption.from_dict(binding.reopen())
        != consumption
    ):
        raise RuntimeError("single-operator consumption changed during publication")
    return binding


def validate_formal_single_operator_admission_consumption(
    consumption_path: str | Path,
    *,
    admission_path: str | Path,
    plan_path: str | Path,
) -> FormalSingleOperatorAdmissionConsumption:
    artifact = validate_formal_single_operator_admission(
        admission_path,
        plan_path=plan_path,
    )
    plan, _launch = _rebuild_admission_sources(artifact)
    expected_path = (
        _private_root(plan) / "formal-single-operator-admission-consumed.json"
    )
    binding = CanonicalJsonProofBinding.bind(consumption_path)
    consumption = FormalSingleOperatorAdmissionConsumption.from_dict(binding.reopen())
    expected = FormalSingleOperatorAdmissionConsumption(
        schema_version=1,
        kind="formal_single_operator_admission_consumption",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_ADMISSION_PROTOCOL_SHA256,
        admission=CanonicalJsonProofBinding.bind(
            admission_path, semantic_sha256=artifact.sha256
        ),
        plan_sha256=plan.sha256,
        execution_binding_sha256=plan.execution_binding_sha256,
        materialized_cell_id=plan.materialized_cell_id,
        attempt_id=plan.native_terminal_binding.attempt_id,
        consumed_ns=consumption.consumed_ns,
        process_hard_timeout_ns=artifact.process_hard_timeout_ns,
        provider_wave_hard_timeout_ns=artifact.provider_wave_hard_timeout_ns,
        compute_gpu_count=len(artifact.gpu_uuids),
        provider_reserved_gpu_count=len(artifact.provider_reserved_gpu_uuids),
    )
    if (
        Path(consumption_path) != expected_path
        or binding.semantic_sha256 != consumption.sha256
        or consumption != expected
    ):
        raise ValueError("single-operator consumption differs from admission")
    return consumption


__all__ = [
    "FORMAL_SINGLE_OPERATOR_ADMISSION_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_MODE",
    "FORMAL_SINGLE_OPERATOR_PROCESS_HARD_TIMEOUT_NS",
    "FORMAL_SINGLE_OPERATOR_PROVIDER_RELEASE_GRACE_NS",
    "FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS",
    "FormalSingleOperatorAdmission",
    "FormalSingleOperatorAdmissionConsumption",
    "consume_formal_single_operator_admission",
    "publish_formal_single_operator_admission",
    "validate_formal_single_operator_admission",
    "validate_formal_single_operator_admission_consumption",
]
