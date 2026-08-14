"""Strict canonical JSON boundaries for industrial planning artifacts.

The planning dataclasses deliberately contain no permissive JSON constructors.
This module is the single CLI-facing wire boundary: it emits complete content,
adds a redundant content digest, and reconstructs the real dataclasses so that
their ``__post_init__`` invariants run on every load.
"""

from __future__ import annotations

import math
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import cache, cached_property
from typing import (
    Any,
    Literal,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from lightcone_spec.experiments.load import ProductionLoadPlan
from lightcone_spec.experiments.planning import (
    BudgetGroupTotal,
    BudgetInventoryIdentity,
    BudgetLoadBinding,
    BudgetMaterializationAuthorityBinding,
    BudgetPlan,
    BudgetPolicy,
    CapacityAuthorityBinding,
    CapacityEnvelope,
    ConfirmationAuxiliaryActivationAuthorityBinding,
    ConfirmationAuxiliaryCompletionAuthorityBinding,
    ConfirmationFamilyCompletionAuthorityBinding,
    ConfirmationFamilyIdentity,
    ConfirmationFamilyPowerPlan,
    ConfirmationFamilyPowerReductionArtifact,
    ConfirmationFinalActivationAuthorityBinding,
    ConfirmationPilotActivationAuthorityBinding,
    ConfirmationStageAggregateAuthorityBinding,
    ConfirmationStageFamilyAuthorityBinding,
    E1ActivationAuthorityBinding,
    E1ParetoArtifact,
    E2ActivationAuthorityBinding,
    E2FinalRecipeArtifact,
    E2StageCompletionAuthorityBinding,
    E2StageEvidenceArtifact,
    E2StageReductionArtifact,
    E2SurvivorReceipt,
    EvidenceAliasReceipt,
    EvidenceAliasReductionArtifact,
    EvidenceDependenceMap,
    ExactScenarioHours,
    ExpectedMaximumCount,
    ExperimentBudget,
    FamilyActivationArtifact,
    FamilyPilotCompletionAuthorityBinding,
    IndustrialBudgetReport,
    ReducerActivationArtifact,
    RegistryStageActivationAuthorityBinding,
    ScenarioMilliseconds,
    SealedE3aSelection,
)
from lightcone_spec.experiments.registry import StageActivationPlan, content_sha256

_ARTIFACT_KIND_FIELD = "artifact_kind"
_ARTIFACT_SHA256_FIELD = "artifact_sha256"
_SHA256_LENGTH = 64
_BUDGET_COMPONENT_FIELDS = (
    "startup_model_load",
    "compile_jit_graph_prewarm",
    "excluded_warmup",
    "scored_arrival",
    "drain",
    "reset_finalization",
    "evidence_flush_shutdown",
    "soak",
    "failure_injection",
    "retry",
    "profiler",
    "download_compile_reservation",
)
_P99_STATUS_VALUES = ("not_required", "required_unresolved", "locked")


def _strict_object(
    name: str, value: object, expected_fields: frozenset[str]
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    row = value
    actual = set(row)
    missing = expected_fields - actual
    unknown = actual - expected_fields
    if missing or unknown:
        raise ValueError(
            f"{name} fields differ: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return row


def _strict_text(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be JSON text")
    return value


def _strict_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a JSON integer")
    return value


def _strict_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a JSON boolean")
    return value


def _strict_float(name: str, value: object) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a JSON floating-point number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_sha256(name: str, value: object) -> str:
    checked = _strict_text(name, value)
    if len(checked) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in checked
    ):
        raise ValueError(f"{name} must be lower-case SHA-256")
    return checked


def _require_single_line_text(name: str, value: str) -> None:
    if not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be non-empty single-line text")


def _encode_value(value: object, *, path: str) -> Any:
    if isinstance(value, Enum):
        return _encode_value(value.value, path=path)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _encode_value(
                getattr(value, field.name), path=f"{path}.{field.name}"
            )
            for field in fields(value)
        }
    if type(value) is tuple:
        return [
            _encode_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return 0.0 if value == 0 else value
    if value is None or type(value) in {str, int, bool}:
        return value
    raise TypeError(f"{path} has unsupported canonical type {type(value).__name__}")


@cache
def _resolved_hints(model: type[object]) -> dict[str, Any]:
    return get_type_hints(model)


def _decode_literal(expected: object, value: object, *, path: str) -> object:
    choices = get_args(expected)
    for choice in choices:
        if type(value) is type(choice) and value == choice:
            return value
    raise ValueError(f"{path} is outside the registered literal values")


def _decode_value(expected: object, value: object, *, path: str) -> Any:
    if isinstance(expected, TypeAliasType):
        return _decode_value(expected.__value__, value, path=path)
    origin = get_origin(expected)
    arguments = get_args(expected)
    if origin is Literal:
        return _decode_literal(expected, value, path=path)
    if origin in {types.UnionType, Union}:
        non_null = tuple(
            argument for argument in arguments if argument is not type(None)
        )
        if len(non_null) == 1 and len(non_null) != len(arguments):
            if value is None:
                return None
            return _decode_value(non_null[0], value, path=path)
        failures: list[Exception] = []
        for argument in arguments:
            try:
                return _decode_value(argument, value, path=path)
            except (TypeError, ValueError) as exc:
                failures.append(exc)
        raise TypeError(
            f"{path} does not match any registered union member"
        ) from failures[-1]
    if origin is tuple:
        if type(value) is not list:
            raise TypeError(f"{path} must be a JSON array")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _decode_value(arguments[0], item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise ValueError(f"{path} has the wrong fixed-array length")
        return tuple(
            _decode_value(argument, item, path=f"{path}[{index}]")
            for index, (argument, item) in enumerate(zip(arguments, value, strict=True))
        )
    if expected is str:
        return _strict_text(path, value)
    if expected is int:
        return _strict_int(path, value)
    if expected is bool:
        return _strict_bool(path, value)
    if expected is float:
        return _strict_float(path, value)
    if expected is type(None):
        if value is not None:
            raise TypeError(f"{path} must be null")
        return None
    if isinstance(expected, type) and issubclass(expected, Enum):
        wire_value = _strict_text(path, value)
        try:
            return expected(wire_value)
        except ValueError as exc:
            raise ValueError(f"{path} has an unknown enum value") from exc
    if isinstance(expected, type) and is_dataclass(expected):
        return _decode_dataclass(expected, value, path=path)
    raise TypeError(f"{path} uses unsupported annotation {expected!r}")


def _decode_dataclass(model: type[Any], value: object, *, path: str) -> Any:
    model_fields = fields(model)
    names = frozenset(field.name for field in model_fields)
    row = _strict_object(path, value, names)
    hints = _resolved_hints(model)
    decoded = {
        field.name: _decode_value(
            hints[field.name], row[field.name], path=f"{path}.{field.name}"
        )
        for field in model_fields
    }
    return model(**decoded)


@dataclass(frozen=True)
class PlanningArtifactSidecar:
    """Strict optional sidecar binding a wire kind to its content identity."""

    schema_version: int
    artifact_kind: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only planning sidecar schema version 1 is supported")
        if type(self.artifact_kind) is not str:
            raise TypeError("sidecar artifact_kind must be text")
        _require_single_line_text("sidecar artifact_kind", self.artifact_kind)
        _require_sha256("sidecar artifact_sha256", self.artifact_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> PlanningArtifactSidecar:
        row = _strict_object(
            "planning artifact sidecar",
            value,
            frozenset({"schema_version", "artifact_kind", "artifact_sha256"}),
        )
        return cls(
            schema_version=_strict_int("sidecar.schema_version", row["schema_version"]),
            artifact_kind=_strict_text("sidecar.artifact_kind", row["artifact_kind"]),
            artifact_sha256=_require_sha256(
                "sidecar.artifact_sha256", row["artifact_sha256"]
            ),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


SidecarInput = PlanningArtifactSidecar | Mapping[str, Any] | None


def _validate_sidecar(
    *, artifact_kind: str, artifact_sha256: str, sidecar: SidecarInput
) -> None:
    if sidecar is None:
        return
    parsed = (
        sidecar
        if type(sidecar) is PlanningArtifactSidecar
        else PlanningArtifactSidecar.from_dict(sidecar)
    )
    if parsed.artifact_kind != artifact_kind:
        raise ValueError("planning artifact sidecar kind mismatch")
    if parsed.artifact_sha256 != artifact_sha256:
        raise ValueError("planning artifact sidecar SHA-256 mismatch")


def _artifact_to_dict(
    value: Any, *, model: type[Any], artifact_kind: str
) -> dict[str, Any]:
    if type(value) is not model:
        raise TypeError(f"{artifact_kind} must be an exact {model.__name__}")
    content = _encode_value(value, path=artifact_kind)
    rebuilt = _decode_dataclass(model, content, path=artifact_kind)
    _validate_loaded_artifact(rebuilt)
    if rebuilt != value:
        raise ValueError(f"{artifact_kind} is not canonically representable")
    artifact_sha256 = _require_sha256(f"{artifact_kind}.sha256", value.sha256)
    return {
        _ARTIFACT_KIND_FIELD: artifact_kind,
        _ARTIFACT_SHA256_FIELD: artifact_sha256,
        **content,
    }


def _artifact_from_dict(
    value: object,
    *,
    model: type[Any],
    artifact_kind: str,
    sidecar: SidecarInput,
) -> Any:
    expected = frozenset(field.name for field in fields(model)) | frozenset(
        {_ARTIFACT_KIND_FIELD, _ARTIFACT_SHA256_FIELD}
    )
    row = _strict_object(artifact_kind, value, expected)
    wire_kind = _strict_text(
        f"{artifact_kind}.artifact_kind", row[_ARTIFACT_KIND_FIELD]
    )
    if wire_kind != artifact_kind:
        raise ValueError(f"{artifact_kind} wire kind mismatch")
    declared_sha256 = _require_sha256(
        f"{artifact_kind}.artifact_sha256", row[_ARTIFACT_SHA256_FIELD]
    )
    content = {field.name: row[field.name] for field in fields(model)}
    artifact = _decode_dataclass(model, content, path=artifact_kind)
    _validate_loaded_artifact(artifact)
    actual_sha256 = _require_sha256(f"{artifact_kind}.computed_sha256", artifact.sha256)
    if declared_sha256 != actual_sha256:
        raise ValueError(f"{artifact_kind} redundant artifact SHA-256 mismatch")
    _validate_sidecar(
        artifact_kind=artifact_kind,
        artifact_sha256=actual_sha256,
        sidecar=sidecar,
    )
    return artifact


def _scenario_at_least(
    upper: ScenarioMilliseconds, lower: ScenarioMilliseconds
) -> bool:
    return all(
        getattr(upper, name) >= getattr(lower, name)
        for name in ("optimistic", "registered", "quota_envelope")
    )


def _sum_scenarios(values: Sequence[ScenarioMilliseconds]) -> ScenarioMilliseconds:
    return ScenarioMilliseconds(
        sum(value.optimistic for value in values),
        sum(value.registered for value in values),
        sum(value.quota_envelope for value in values),
    )


def _sum_counts(
    values: Sequence[ExpectedMaximumCount],
) -> ExpectedMaximumCount:
    if not values:
        return ExpectedMaximumCount(0, 0)
    result = values[0]
    for value in values[1:]:
        result = result + value
    return result


def _validate_exact_hours(
    name: str, milliseconds: ScenarioMilliseconds, hours: ExactScenarioHours
) -> None:
    if hours != ExactScenarioHours.from_milliseconds(milliseconds):
        raise ValueError(f"{name} exact-hour numerator mismatch")


def _validate_component_rows(
    name: str, rows: tuple[tuple[str, ScenarioMilliseconds], ...]
) -> None:
    if tuple(component for component, _ in rows) != _BUDGET_COMPONENT_FIELDS:
        raise ValueError(f"{name} must cover every budget component in canonical order")


def _validate_status_counts(
    name: str, rows: tuple[tuple[str, int], ...], cells: int
) -> None:
    if tuple(status for status, _ in rows) != _P99_STATUS_VALUES:
        raise ValueError(f"{name} must cover every p99 status in canonical order")
    if any(count < 0 for _, count in rows) or sum(count for _, count in rows) != cells:
        raise ValueError(f"{name} counts do not match the cell total")


def _validate_budget_group(group: BudgetGroupTotal) -> None:
    for name in ("experiment", "method", "workload_class", "topology"):
        _require_single_line_text(f"budget group {name}", getattr(group, name))
    for name in (
        "cells",
        "gpu_cell_units",
        "minimum_completed_requests",
        "retry_allowance",
    ):
        if getattr(group, name) < 0:
            raise ValueError(f"budget group {name} must be non-negative")
    if group.gpu_cell_units < group.cells:
        raise ValueError("budget group GPU-cell units cannot be below its cells")
    _validate_component_rows("budget group component_ms", group.component_ms)
    _validate_status_counts(
        "budget group p99_anchor_status_counts",
        group.p99_anchor_status_counts,
        group.cells,
    )
    for name, milliseconds, hours in (
        ("group compute", group.compute_gpu_ms, group.compute_gpu_hours),
        ("group reserved", group.reserved_gpu_ms, group.reserved_gpu_hours),
        (
            "group fixed-instance billed",
            group.fixed_instance_billed_gpu_ms,
            group.fixed_instance_billed_gpu_hours,
        ),
    ):
        _validate_exact_hours(name, milliseconds, hours)
    if not _scenario_at_least(group.reserved_gpu_ms, group.compute_gpu_ms):
        raise ValueError("budget group reserved GPU time is below compute time")
    if not _scenario_at_least(group.fixed_instance_billed_gpu_ms, group.compute_gpu_ms):
        raise ValueError("budget group billed GPU time is below compute time")


def _validate_budget_report(report: IndustrialBudgetReport) -> None:
    if report.budget_sha256s != tuple(sorted(set(report.budget_sha256s))):
        raise ValueError("budget identities must be canonically sorted and unique")
    for value in report.budget_sha256s:
        _require_sha256("budget report budget identity", value)
    if len(report.budget_sha256s) != report.cells:
        raise ValueError("budget identity count does not match report cells")
    group_keys = tuple(
        (row.experiment, row.method, row.workload_class, row.topology)
        for row in report.groups
    )
    if group_keys != tuple(sorted(set(group_keys))):
        raise ValueError("budget report groups must be canonically sorted and unique")
    for group in report.groups:
        _validate_budget_group(group)
    for name in (
        "cells",
        "gpu_cell_units",
        "minimum_completed_requests",
        "retry_allowance",
    ):
        if getattr(report, name) < 0:
            raise ValueError(f"budget report {name} must be non-negative")
    if report.gpu_cell_units < report.cells:
        raise ValueError("budget report GPU-cell units cannot be below its cells")
    _validate_component_rows("budget report component_ms", report.component_ms)
    _validate_status_counts(
        "budget report p99_anchor_status_counts",
        report.p99_anchor_status_counts,
        report.cells,
    )
    if sum(group.cells for group in report.groups) != report.cells:
        raise ValueError("budget group cell totals do not match the report")
    if sum(group.gpu_cell_units for group in report.groups) != report.gpu_cell_units:
        raise ValueError("budget group GPU-cell totals do not match the report")
    group_components = tuple(
        (
            component,
            _sum_scenarios(
                tuple(group.component_ms[index][1] for group in report.groups)
            ),
        )
        for index, component in enumerate(_BUDGET_COMPONENT_FIELDS)
    )
    if group_components != report.component_ms:
        raise ValueError("budget group components do not sum to the report")
    if _sum_scenarios(tuple(group.request_deadline_ms for group in report.groups)) != (
        report.request_deadline_ms
    ):
        raise ValueError("budget group deadlines do not sum to the report")
    if _sum_counts(
        tuple(group.excluded_warmup_requests for group in report.groups)
    ) != (report.excluded_warmup_requests):
        raise ValueError("budget group warm-up requests do not sum to the report")
    if (
        _sum_counts(tuple(group.output_tokens for group in report.groups))
        != report.output_tokens
    ):
        raise ValueError("budget group output tokens do not sum to the report")
    if sum(group.minimum_completed_requests for group in report.groups) != (
        report.minimum_completed_requests
    ):
        raise ValueError("budget group completions do not sum to the report")
    if sum(group.retry_allowance for group in report.groups) != report.retry_allowance:
        raise ValueError("budget group retry allowances do not sum to the report")
    aggregate_status = tuple(
        (
            status,
            sum(
                dict(group.p99_anchor_status_counts)[status] for group in report.groups
            ),
        )
        for status in _P99_STATUS_VALUES
    )
    if aggregate_status != report.p99_anchor_status_counts:
        raise ValueError("budget group p99 statuses do not sum to the report")
    for name, group_values, total, hours in (
        (
            "report compute",
            tuple(group.compute_gpu_ms for group in report.groups),
            report.compute_gpu_ms,
            report.compute_gpu_hours,
        ),
        (
            "report reserved",
            tuple(group.reserved_gpu_ms for group in report.groups),
            report.reserved_gpu_ms,
            report.reserved_gpu_hours,
        ),
        (
            "report fixed-instance billed",
            tuple(group.fixed_instance_billed_gpu_ms for group in report.groups),
            report.fixed_instance_billed_gpu_ms,
            report.fixed_instance_billed_gpu_hours,
        ),
    ):
        if _sum_scenarios(group_values) != total:
            raise ValueError(f"{name} group total mismatch")
        _validate_exact_hours(name, total, hours)
    if not _scenario_at_least(report.reserved_gpu_ms, report.compute_gpu_ms):
        raise ValueError("budget report reserved GPU time is below compute time")
    if not _scenario_at_least(
        report.fixed_instance_billed_gpu_ms, report.compute_gpu_ms
    ):
        raise ValueError("budget report billed GPU time is below compute time")
    if report.estimated_wall_ms is not None:
        if report.estimated_wall_hours is None:
            raise ValueError("estimated wall hours are missing")
        _validate_exact_hours(
            "estimated wall", report.estimated_wall_ms, report.estimated_wall_hours
        )
    if report.schedule_fixed_instance_billed_gpu_ms is not None:
        if report.schedule_fixed_instance_billed_gpu_hours is None:
            raise ValueError("scheduled fixed-instance billed hours are missing")
        _validate_exact_hours(
            "scheduled fixed-instance billed",
            report.schedule_fixed_instance_billed_gpu_ms,
            report.schedule_fixed_instance_billed_gpu_hours,
        )
        if report.estimated_wall_ms is None or (
            report.schedule_fixed_instance_billed_gpu_ms
            != report.estimated_wall_ms.scale(report.inventory.gpu_count)
        ):
            raise ValueError("scheduled billed time does not match inventory gang time")


def _validate_stage_plan(plan: StageActivationPlan) -> None:
    for name in (
        "activated_cell_ids",
        "not_applicable_cell_ids",
        "blocked_cell_ids",
        "deferred_cell_ids",
    ):
        values = getattr(plan, name)
        if values != tuple(sorted(set(values))):
            raise ValueError(f"stage plan {name} must be canonically sorted and unique")


def _validate_loaded_artifact(value: object) -> None:
    if isinstance(value, IndustrialBudgetReport):
        _validate_budget_report(value)
    elif isinstance(value, ReducerActivationArtifact):
        _validate_stage_plan(value.plan)
    elif isinstance(value, E1ParetoArtifact):
        for geometry in value.surviving_geometries:
            if geometry.parameterization == "lora" and (
                type(geometry.rank) is not int or geometry.rank < 1
            ):
                raise ValueError("E1 LoRA geometry rank must be a positive integer")
    elif isinstance(value, E2StageEvidenceArtifact):
        for evaluation in value.evaluations:
            for reason in evaluation.safety_reason_codes:
                _require_single_line_text("E2 safety reason", reason)
    elif isinstance(value, E2FinalRecipeArtifact):
        if (
            value.candidate.sha256 != value.candidate_id
            or value.recipe.sha256 != value.recipe_sha256
        ):
            raise ValueError("E2 final recipe changed candidate or recipe identity")
    elif isinstance(value, E2StageReductionArtifact):
        if (
            value.activation.sha256 != value.stage_evidence.activation_sha256
            or value.survivor_receipt.tuning_evidence_sha256
            != value.stage_evidence.sha256
        ):
            raise ValueError("E2 reduction changed its raw-evidence binding")
    elif isinstance(value, ConfirmationFamilyPowerReductionArtifact):
        if value.raw_evidence_manifest_sha256 != value.plan.pilot_evidence_sha256:
            raise ValueError("family power reduction changed its raw-evidence binding")
    elif isinstance(value, EvidenceDependenceMap):
        for unit in value.units:
            for cell_id in unit.member_cell_ids:
                _require_sha256("dependence member cell_id", cell_id)


def _make_to_dict(model: type[Any], artifact_kind: str):
    def encode(value: Any) -> dict[str, Any]:
        return _artifact_to_dict(value, model=model, artifact_kind=artifact_kind)

    return encode


def _make_from_dict(model: type[Any], artifact_kind: str):
    def decode(value: object, *, sidecar: SidecarInput = None) -> Any:
        return _artifact_from_dict(
            value,
            model=model,
            artifact_kind=artifact_kind,
            sidecar=sidecar,
        )

    return decode


experiment_budget_to_dict = _make_to_dict(ExperimentBudget, "experiment_budget")
experiment_budget_from_dict = _make_from_dict(ExperimentBudget, "experiment_budget")


def production_load_plan_to_dict(value: ProductionLoadPlan) -> dict[str, Any]:
    """Serialize a complete load plan under its paired-replay identity."""

    if type(value) is not ProductionLoadPlan:
        raise TypeError("production_load_plan must be an exact ProductionLoadPlan")
    value.validate()
    content = _encode_value(value, path="production_load_plan")
    rebuilt = _decode_dataclass(
        ProductionLoadPlan,
        content,
        path="production_load_plan",
    )
    rebuilt.validate()
    if rebuilt != value:
        raise ValueError("production_load_plan is not canonically representable")
    artifact_sha256 = _require_sha256(
        "production_load_plan.paired_replay_sha256",
        value.paired_replay_sha256,
    )
    return {
        _ARTIFACT_KIND_FIELD: "production_load_plan",
        _ARTIFACT_SHA256_FIELD: artifact_sha256,
        **content,
    }


def production_load_plan_from_dict(
    value: object,
    *,
    sidecar: SidecarInput = None,
) -> ProductionLoadPlan:
    """Strictly reconstruct and sidecar-check a complete load plan."""

    expected = frozenset(field.name for field in fields(ProductionLoadPlan)) | {
        _ARTIFACT_KIND_FIELD,
        _ARTIFACT_SHA256_FIELD,
    }
    row = _strict_object("production_load_plan", value, frozenset(expected))
    if (
        _strict_text(
            "production_load_plan.artifact_kind",
            row[_ARTIFACT_KIND_FIELD],
        )
        != "production_load_plan"
    ):
        raise ValueError("production_load_plan wire kind mismatch")
    declared_sha256 = _require_sha256(
        "production_load_plan.artifact_sha256",
        row[_ARTIFACT_SHA256_FIELD],
    )
    content = {field.name: row[field.name] for field in fields(ProductionLoadPlan)}
    artifact = _decode_dataclass(
        ProductionLoadPlan,
        content,
        path="production_load_plan",
    )
    artifact.validate()
    actual_sha256 = _require_sha256(
        "production_load_plan.computed_sha256",
        artifact.paired_replay_sha256,
    )
    if declared_sha256 != actual_sha256:
        raise ValueError("production_load_plan redundant artifact SHA-256 mismatch")
    _validate_sidecar(
        artifact_kind="production_load_plan",
        artifact_sha256=actual_sha256,
        sidecar=sidecar,
    )
    return artifact


budget_inventory_identity_to_dict = _make_to_dict(
    BudgetInventoryIdentity, "budget_inventory_identity"
)
budget_inventory_identity_from_dict = _make_from_dict(
    BudgetInventoryIdentity, "budget_inventory_identity"
)
budget_load_binding_to_dict = _make_to_dict(BudgetLoadBinding, "budget_load_binding")
budget_load_binding_from_dict = _make_from_dict(
    BudgetLoadBinding, "budget_load_binding"
)
budget_materialization_authority_binding_to_dict = _make_to_dict(
    BudgetMaterializationAuthorityBinding,
    "budget_materialization_authority_binding",
)
budget_materialization_authority_binding_from_dict = _make_from_dict(
    BudgetMaterializationAuthorityBinding,
    "budget_materialization_authority_binding",
)
e1_activation_authority_binding_to_dict = _make_to_dict(
    E1ActivationAuthorityBinding,
    "e1_activation_authority_binding",
)
e1_activation_authority_binding_from_dict = _make_from_dict(
    E1ActivationAuthorityBinding,
    "e1_activation_authority_binding",
)
e2_activation_authority_binding_to_dict = _make_to_dict(
    E2ActivationAuthorityBinding,
    "e2_activation_authority_binding",
)
e2_activation_authority_binding_from_dict = _make_from_dict(
    E2ActivationAuthorityBinding,
    "e2_activation_authority_binding",
)
e2_stage_completion_authority_binding_to_dict = _make_to_dict(
    E2StageCompletionAuthorityBinding,
    "e2_stage_completion_authority_binding",
)
e2_stage_completion_authority_binding_from_dict = _make_from_dict(
    E2StageCompletionAuthorityBinding,
    "e2_stage_completion_authority_binding",
)
confirmation_pilot_activation_authority_binding_to_dict = _make_to_dict(
    ConfirmationPilotActivationAuthorityBinding,
    "confirmation_pilot_activation_authority_binding",
)
confirmation_pilot_activation_authority_binding_from_dict = _make_from_dict(
    ConfirmationPilotActivationAuthorityBinding,
    "confirmation_pilot_activation_authority_binding",
)
confirmation_auxiliary_activation_authority_binding_to_dict = _make_to_dict(
    ConfirmationAuxiliaryActivationAuthorityBinding,
    "confirmation_auxiliary_activation_authority_binding",
)
confirmation_auxiliary_activation_authority_binding_from_dict = _make_from_dict(
    ConfirmationAuxiliaryActivationAuthorityBinding,
    "confirmation_auxiliary_activation_authority_binding",
)
confirmation_auxiliary_completion_authority_binding_to_dict = _make_to_dict(
    ConfirmationAuxiliaryCompletionAuthorityBinding,
    "confirmation_auxiliary_completion_authority_binding",
)
confirmation_auxiliary_completion_authority_binding_from_dict = _make_from_dict(
    ConfirmationAuxiliaryCompletionAuthorityBinding,
    "confirmation_auxiliary_completion_authority_binding",
)
family_pilot_completion_authority_binding_to_dict = _make_to_dict(
    FamilyPilotCompletionAuthorityBinding,
    "family_pilot_completion_authority_binding",
)
family_pilot_completion_authority_binding_from_dict = _make_from_dict(
    FamilyPilotCompletionAuthorityBinding,
    "family_pilot_completion_authority_binding",
)
confirmation_final_activation_authority_binding_to_dict = _make_to_dict(
    ConfirmationFinalActivationAuthorityBinding,
    "confirmation_final_activation_authority_binding",
)
confirmation_final_activation_authority_binding_from_dict = _make_from_dict(
    ConfirmationFinalActivationAuthorityBinding,
    "confirmation_final_activation_authority_binding",
)
confirmation_family_completion_authority_binding_to_dict = _make_to_dict(
    ConfirmationFamilyCompletionAuthorityBinding,
    "confirmation_family_completion_authority_binding",
)
confirmation_family_completion_authority_binding_from_dict = _make_from_dict(
    ConfirmationFamilyCompletionAuthorityBinding,
    "confirmation_family_completion_authority_binding",
)
confirmation_stage_family_authority_binding_to_dict = _make_to_dict(
    ConfirmationStageFamilyAuthorityBinding,
    "confirmation_stage_family_authority_binding",
)
confirmation_stage_family_authority_binding_from_dict = _make_from_dict(
    ConfirmationStageFamilyAuthorityBinding,
    "confirmation_stage_family_authority_binding",
)
confirmation_stage_aggregate_authority_binding_to_dict = _make_to_dict(
    ConfirmationStageAggregateAuthorityBinding,
    "confirmation_stage_aggregate_authority_binding",
)
confirmation_stage_aggregate_authority_binding_from_dict = _make_from_dict(
    ConfirmationStageAggregateAuthorityBinding,
    "confirmation_stage_aggregate_authority_binding",
)
registry_stage_activation_authority_binding_to_dict = _make_to_dict(
    RegistryStageActivationAuthorityBinding,
    "registry_stage_activation_authority_binding",
)
registry_stage_activation_authority_binding_from_dict = _make_from_dict(
    RegistryStageActivationAuthorityBinding,
    "registry_stage_activation_authority_binding",
)
budget_policy_to_dict = _make_to_dict(BudgetPolicy, "budget_policy")
budget_policy_from_dict = _make_from_dict(BudgetPolicy, "budget_policy")
budget_plan_to_dict = _make_to_dict(BudgetPlan, "budget_plan")
budget_plan_from_dict = _make_from_dict(BudgetPlan, "budget_plan")
capacity_authority_binding_to_dict = _make_to_dict(
    CapacityAuthorityBinding, "capacity_authority_binding"
)
capacity_authority_binding_from_dict = _make_from_dict(
    CapacityAuthorityBinding, "capacity_authority_binding"
)
capacity_envelope_to_dict = _make_to_dict(CapacityEnvelope, "capacity_envelope")
capacity_envelope_from_dict = _make_from_dict(CapacityEnvelope, "capacity_envelope")
industrial_budget_report_to_dict = _make_to_dict(
    IndustrialBudgetReport, "industrial_budget_report"
)
industrial_budget_report_from_dict = _make_from_dict(
    IndustrialBudgetReport, "industrial_budget_report"
)
sealed_e3a_selection_to_dict = _make_to_dict(SealedE3aSelection, "sealed_e3a_selection")
sealed_e3a_selection_from_dict = _make_from_dict(
    SealedE3aSelection, "sealed_e3a_selection"
)
reducer_activation_artifact_to_dict = _make_to_dict(
    ReducerActivationArtifact, "reducer_activation_artifact"
)
reducer_activation_artifact_from_dict = _make_from_dict(
    ReducerActivationArtifact, "reducer_activation_artifact"
)
e1_pareto_artifact_to_dict = _make_to_dict(E1ParetoArtifact, "e1_pareto_artifact")
e1_pareto_artifact_from_dict = _make_from_dict(E1ParetoArtifact, "e1_pareto_artifact")
e2_final_recipe_artifact_to_dict = _make_to_dict(
    E2FinalRecipeArtifact, "e2_final_recipe_artifact"
)
e2_final_recipe_artifact_from_dict = _make_from_dict(
    E2FinalRecipeArtifact, "e2_final_recipe_artifact"
)
e2_stage_evidence_artifact_to_dict = _make_to_dict(
    E2StageEvidenceArtifact, "e2_stage_evidence_artifact"
)
e2_stage_evidence_artifact_from_dict = _make_from_dict(
    E2StageEvidenceArtifact, "e2_stage_evidence_artifact"
)
e2_stage_reduction_artifact_to_dict = _make_to_dict(
    E2StageReductionArtifact, "e2_stage_reduction_artifact"
)
e2_stage_reduction_artifact_from_dict = _make_from_dict(
    E2StageReductionArtifact, "e2_stage_reduction_artifact"
)
e2_survivor_receipt_to_dict = _make_to_dict(E2SurvivorReceipt, "e2_survivor_receipt")
e2_survivor_receipt_from_dict = _make_from_dict(
    E2SurvivorReceipt, "e2_survivor_receipt"
)
confirmation_family_identity_to_dict = _make_to_dict(
    ConfirmationFamilyIdentity, "confirmation_family_identity"
)
confirmation_family_identity_from_dict = _make_from_dict(
    ConfirmationFamilyIdentity, "confirmation_family_identity"
)
family_activation_artifact_to_dict = _make_to_dict(
    FamilyActivationArtifact, "family_activation_artifact"
)
family_activation_artifact_from_dict = _make_from_dict(
    FamilyActivationArtifact, "family_activation_artifact"
)
confirmation_family_power_plan_to_dict = _make_to_dict(
    ConfirmationFamilyPowerPlan, "confirmation_family_power_plan"
)
confirmation_family_power_plan_from_dict = _make_from_dict(
    ConfirmationFamilyPowerPlan, "confirmation_family_power_plan"
)
confirmation_family_power_reduction_artifact_to_dict = _make_to_dict(
    ConfirmationFamilyPowerReductionArtifact,
    "confirmation_family_power_reduction_artifact",
)
confirmation_family_power_reduction_artifact_from_dict = _make_from_dict(
    ConfirmationFamilyPowerReductionArtifact,
    "confirmation_family_power_reduction_artifact",
)
evidence_alias_receipt_to_dict = _make_to_dict(
    EvidenceAliasReceipt, "evidence_alias_receipt"
)
evidence_alias_receipt_from_dict = _make_from_dict(
    EvidenceAliasReceipt, "evidence_alias_receipt"
)
evidence_alias_reduction_artifact_to_dict = _make_to_dict(
    EvidenceAliasReductionArtifact, "evidence_alias_reduction_artifact"
)
evidence_alias_reduction_artifact_from_dict = _make_from_dict(
    EvidenceAliasReductionArtifact, "evidence_alias_reduction_artifact"
)
evidence_dependence_map_to_dict = _make_to_dict(
    EvidenceDependenceMap, "evidence_dependence_map"
)
evidence_dependence_map_from_dict = _make_from_dict(
    EvidenceDependenceMap, "evidence_dependence_map"
)


def experiment_budget_sequence_to_dict(
    budgets: Sequence[ExperimentBudget],
) -> dict[str, Any]:
    """Encode a canonical cell-sorted immutable budget set."""

    if isinstance(budgets, (str, bytes)) or not isinstance(budgets, Sequence):
        raise TypeError("experiment budgets must be a sequence")
    rows = tuple(budgets)
    if any(type(row) is not ExperimentBudget for row in rows):
        raise TypeError("every experiment budget must be an exact ExperimentBudget")
    ordered = tuple(sorted(rows, key=lambda row: row.cell_id))
    if len({row.cell_id for row in ordered}) != len(ordered):
        raise ValueError("experiment budget sequence contains duplicate cell IDs")
    encoded = [experiment_budget_to_dict(row) for row in ordered]
    return {
        "schema_version": 1,
        _ARTIFACT_KIND_FIELD: "experiment_budget_sequence",
        _ARTIFACT_SHA256_FIELD: content_sha256(ordered),
        "budgets": encoded,
    }


def experiment_budget_sequence_from_dict(
    value: object, *, sidecar: SidecarInput = None
) -> tuple[ExperimentBudget, ...]:
    row = _strict_object(
        "experiment_budget_sequence",
        value,
        frozenset(
            {
                "schema_version",
                _ARTIFACT_KIND_FIELD,
                _ARTIFACT_SHA256_FIELD,
                "budgets",
            }
        ),
    )
    if (
        _strict_int("experiment_budget_sequence.schema_version", row["schema_version"])
        != 1
    ):
        raise ValueError(
            "only experiment budget sequence schema version 1 is supported"
        )
    if (
        _strict_text(
            "experiment_budget_sequence.artifact_kind", row[_ARTIFACT_KIND_FIELD]
        )
        != "experiment_budget_sequence"
    ):
        raise ValueError("experiment budget sequence wire kind mismatch")
    if type(row["budgets"]) is not list:
        raise TypeError("experiment_budget_sequence.budgets must be a JSON array")
    budgets = tuple(experiment_budget_from_dict(item) for item in row["budgets"])
    cell_ids = tuple(budget.cell_id for budget in budgets)
    if cell_ids != tuple(sorted(set(cell_ids))):
        raise ValueError("experiment budgets must be cell-sorted and unique")
    actual_sha256 = content_sha256(budgets)
    declared_sha256 = _require_sha256(
        "experiment_budget_sequence.artifact_sha256",
        row[_ARTIFACT_SHA256_FIELD],
    )
    if actual_sha256 != declared_sha256:
        raise ValueError("experiment budget sequence redundant SHA-256 mismatch")
    _validate_sidecar(
        artifact_kind="experiment_budget_sequence",
        artifact_sha256=actual_sha256,
        sidecar=sidecar,
    )
    return budgets


__all__ = [
    "PlanningArtifactSidecar",
    "budget_inventory_identity_from_dict",
    "budget_inventory_identity_to_dict",
    "budget_load_binding_from_dict",
    "budget_load_binding_to_dict",
    "budget_materialization_authority_binding_from_dict",
    "budget_materialization_authority_binding_to_dict",
    "budget_plan_from_dict",
    "budget_plan_to_dict",
    "budget_policy_from_dict",
    "budget_policy_to_dict",
    "capacity_authority_binding_from_dict",
    "capacity_authority_binding_to_dict",
    "capacity_envelope_from_dict",
    "capacity_envelope_to_dict",
    "confirmation_auxiliary_activation_authority_binding_from_dict",
    "confirmation_auxiliary_activation_authority_binding_to_dict",
    "confirmation_auxiliary_completion_authority_binding_from_dict",
    "confirmation_auxiliary_completion_authority_binding_to_dict",
    "confirmation_family_completion_authority_binding_from_dict",
    "confirmation_family_completion_authority_binding_to_dict",
    "confirmation_family_identity_from_dict",
    "confirmation_family_identity_to_dict",
    "confirmation_family_power_plan_from_dict",
    "confirmation_family_power_plan_to_dict",
    "confirmation_family_power_reduction_artifact_from_dict",
    "confirmation_family_power_reduction_artifact_to_dict",
    "confirmation_final_activation_authority_binding_from_dict",
    "confirmation_final_activation_authority_binding_to_dict",
    "confirmation_pilot_activation_authority_binding_from_dict",
    "confirmation_pilot_activation_authority_binding_to_dict",
    "confirmation_stage_aggregate_authority_binding_from_dict",
    "confirmation_stage_aggregate_authority_binding_to_dict",
    "confirmation_stage_family_authority_binding_from_dict",
    "confirmation_stage_family_authority_binding_to_dict",
    "e1_activation_authority_binding_from_dict",
    "e1_activation_authority_binding_to_dict",
    "e1_pareto_artifact_from_dict",
    "e1_pareto_artifact_to_dict",
    "e2_activation_authority_binding_from_dict",
    "e2_activation_authority_binding_to_dict",
    "e2_stage_completion_authority_binding_from_dict",
    "e2_stage_completion_authority_binding_to_dict",
    "e2_stage_evidence_artifact_from_dict",
    "e2_stage_evidence_artifact_to_dict",
    "e2_stage_reduction_artifact_from_dict",
    "e2_stage_reduction_artifact_to_dict",
    "e2_survivor_receipt_from_dict",
    "e2_survivor_receipt_to_dict",
    "evidence_alias_receipt_from_dict",
    "evidence_alias_receipt_to_dict",
    "evidence_alias_reduction_artifact_from_dict",
    "evidence_alias_reduction_artifact_to_dict",
    "evidence_dependence_map_from_dict",
    "evidence_dependence_map_to_dict",
    "experiment_budget_from_dict",
    "experiment_budget_sequence_from_dict",
    "experiment_budget_sequence_to_dict",
    "experiment_budget_to_dict",
    "family_activation_artifact_from_dict",
    "family_activation_artifact_to_dict",
    "family_pilot_completion_authority_binding_from_dict",
    "family_pilot_completion_authority_binding_to_dict",
    "industrial_budget_report_from_dict",
    "industrial_budget_report_to_dict",
    "production_load_plan_from_dict",
    "production_load_plan_to_dict",
    "reducer_activation_artifact_from_dict",
    "reducer_activation_artifact_to_dict",
    "registry_stage_activation_authority_binding_from_dict",
    "registry_stage_activation_authority_binding_to_dict",
    "sealed_e3a_selection_from_dict",
    "sealed_e3a_selection_to_dict",
]
