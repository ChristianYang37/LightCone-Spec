"""Canonical figure/table specifications from existing typed reductions.

This is a presentation boundary, not another reducer.  It never opens raw
evidence or recomputes a statistic.  Missing production artifacts become a
named ``BLOCKED`` output, and an ``UNRESOLVED`` formal source keeps every
publication measurement as JSON ``null``.  In particular, diagnostic E3b
observations are not silently promoted into paper values.

The family-power artifact is preregistered planning evidence derived from four
excluded pilots.  Its complete 12--20 block grid remains renderable while an
``UNDERPOWERED`` decision stays visible.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from itertools import pairwise
from typing import Any, Literal

from lightcone_spec.experiments.industrial_analysis import (
    E3bLongContextStageArtifact,
    E3bNamedLongContextReduction,
    IndustrialReducerArtifact,
    MethodReduction,
)
from lightcone_spec.experiments.long_context_analysis import (
    E3B_CONTEXT_GRID,
    E3bCrossoverOutcome,
    E3bCrossoverReduction,
    E3bCurvePoint,
    E3bIntervalEstimate,
    E3bLongContextReduction,
    E3bMethod,
    E3bMetric,
    E3bReductionStatus,
)
from lightcone_spec.experiments.planning import (
    ConfirmationFamilyIdentity,
    ConfirmationFamilyPowerPlan,
    ConfirmationFamilyPowerReductionArtifact,
    RawEvidenceRunBinding,
)
from lightcone_spec.experiments.registry import (
    CONFIRMATION_METHOD_ROLES,
    PILOT_BLOCKS,
    content_sha256,
)
from lightcone_spec.experiments.statistics import (
    P99_MINIMUM_COMPLETIONS,
    PRIMARY_CONTRASTS,
    PRIMARY_TARGET_POWER,
    REGISTERED_CONFIDENCE,
    SECONDARY_CONTRASTS,
    ContrastPower,
    MultiplicityDecision,
    P99ClaimGuard,
    PairedBcaContrast,
    PowerSizingPlan,
    guard_p99_claim,
)

type SourceKind = Literal[
    "e3b_long_context_stage_reducer",
    "industrial_schema_v3_reducer",
    "confirmation_family_power_reduction",
]
type SpecificationKind = Literal[
    "e3b_long_context_spline_crossover",
    "confirmation_family_power_grid",
    "formal_claim_status",
]


class OutputStatus(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"


CROSS_FAMILY_INTERACTION_AXES = (
    "method_by_model",
    "method_by_context",
    "method_by_load",
)


CROSS_FAMILY_INTERACTION_REDUCER_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_cross_family_interaction_input_shape_contract",
        "roles": CONFIRMATION_METHOD_ROLES,
        "axes": CROSS_FAMILY_INTERACTION_AXES,
        "e0_repetition_shape": "at_least_two_levels_per_declared_axis",
        "formal_authorities": (
            "registry_owned_cells_typed_e2_seal_and_native_completion_absent"
        ),
        "lightcone": "sealed_e2_recipe_receipt_required",
        "lightcone_template": "structural_slot_not_formal_interaction_input",
        "input": "self_consistent_content_bound_structural_bindings_only",
        "numeric_policy": "unresolved_until_complete_formal_reduction",
        "serialization": "canonical_content_sha256",
    }
)


PRODUCTION_OUTPUT_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "lightcone_figure_table_production_output_protocol",
        "inputs": (
            "exact_e3b_long_context_stage_reducer",
            "exact_industrial_schema_v3_reducer",
            "exact_confirmation_family_power_reduction",
        ),
        "raw_evidence_loading": "forbidden",
        "statistical_reduction": "forbidden",
        "diagnostic_numeric_publication": "forbidden",
        "unresolved_numeric_policy": "json_null",
        "missing_source_policy": "named_block",
        "e3b_uncertainty": (
            "paired_hierarchical_block_then_request_95_percentile_refit"
        ),
        "p99": "count_gate_separate_from_formal_claim_status",
        "power": "four_excluded_pilots_registered_12_through_20_grid",
        "power_output_role": "preregistered_planning_only_not_formal_result",
        "scientific_roles": CONFIRMATION_METHOD_ROLES,
        "primary_contrasts": ("lightcone_vs_tts", "lightcone_vs_static"),
        "secondary_contrasts": (
            "l0_naive_vs_tts",
            "lightcone_vs_l0_naive",
        ),
        "interaction_axes": ("method_by_model", "method_by_context", "method_by_load"),
        "interaction_input": "nonformal_cross_family_input_shape_contract",
        "serialization": "canonical_json_sort_keys_ascii_no_nan",
    }
)

_E3B_PANELS = tuple(
    sorted(
        (
            (
                E3bMetric.ACCEPTED_LENGTH,
                E3bMethod.LIGHTCONE,
                E3bMethod.STATIC,
            ),
            (E3bMetric.ACCEPTED_LENGTH, E3bMethod.LIGHTCONE, E3bMethod.TTS),
            (E3bMetric.ACCEPTED_LENGTH, E3bMethod.L0_NAIVE, E3bMethod.TTS),
            (
                E3bMetric.ACCEPTED_LENGTH,
                E3bMethod.LIGHTCONE,
                E3bMethod.L0_NAIVE,
            ),
            (
                E3bMetric.COMMITTED_TOKEN_GOODPUT,
                E3bMethod.LIGHTCONE,
                E3bMethod.STATIC,
            ),
            (
                E3bMetric.COMMITTED_TOKEN_GOODPUT,
                E3bMethod.LIGHTCONE,
                E3bMethod.TTS,
            ),
            (
                E3bMetric.COMMITTED_TOKEN_GOODPUT,
                E3bMethod.L0_NAIVE,
                E3bMethod.TTS,
            ),
            (
                E3bMetric.COMMITTED_TOKEN_GOODPUT,
                E3bMethod.LIGHTCONE,
                E3bMethod.L0_NAIVE,
            ),
        ),
        key=lambda value: ":".join(item.value for item in value),
    )
)
_SOURCE_KINDS: tuple[SourceKind, ...] = (
    "e3b_long_context_stage_reducer",
    "industrial_schema_v3_reducer",
    "confirmation_family_power_reduction",
)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_text(label: str, value: object) -> str:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be exact non-empty single-line text")
    return value


def _canonical_reasons(values: tuple[str, ...]) -> tuple[str, ...]:
    for value in values:
        _require_text("reason code", value)
    return tuple(sorted(set(values)))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(body: bytes) -> dict[str, Any]:
    if type(body) is not bytes:
        raise TypeError("machine-readable specification payload must be bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("machine-readable specification has duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(body, object_pairs_hook=reject_duplicates)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("machine-readable specification is not strict JSON") from error
    if type(value) is not dict:
        raise TypeError("machine-readable specification must be a JSON object")
    try:
        canonical = _canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "machine-readable specification contains a non-finite or invalid value"
        ) from error
    if body != canonical:
        raise ValueError("machine-readable specification is not canonical JSON")
    return value


def _exact_fields(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a JSON object")
    if set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return value


def _exact_list(value: object, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON array")
    return value


def _reason_list(value: object, *, label: str) -> tuple[str, ...]:
    rows = _exact_list(value, label=label)
    if any(type(row) is not str for row in rows):
        raise TypeError(f"{label} must contain strings")
    reasons = tuple(rows)
    if reasons != _canonical_reasons(reasons):
        raise ValueError(f"{label} must be sorted and unique")
    return reasons


_E3B_INTERVAL_FIELDS = (
    "candidate_fitted_metric",
    "baseline_fitted_metric",
    "candidate_elasticity",
    "baseline_elasticity",
    "paired_elasticity_difference",
    "candidate_curvature",
    "baseline_curvature",
    "paired_curvature_difference",
)


def _require_current_sha256(
    label: str,
    *,
    sealed_sha256: str,
    current_sha256: str,
) -> None:
    if sealed_sha256 != current_sha256:
        raise ValueError(f"{label} changed after its SHA-256 was sealed")


def _revalidate_e3b_source(source: E3bLongContextStageArtifact) -> None:
    """Replay the exact typed E3b source, including cached digest identities."""

    if type(source.reductions) is not tuple or any(
        type(row) is not E3bNamedLongContextReduction for row in source.reductions
    ):
        raise TypeError("E3b stage reductions must be exact")
    for row in source.reductions:
        row.__post_init__()
        reduction = row.reduction
        if type(reduction) is not E3bLongContextReduction:
            raise TypeError("E3b stage reduction payloads must be exact")
        crossover = reduction.crossover
        if (
            type(crossover) is not E3bCrossoverReduction
            or type(crossover.outcome) is not E3bCrossoverOutcome
        ):
            raise TypeError("E3b crossover reductions must be exact")
        crossover.__post_init__()
        if reduction.curve_points is not None:
            if type(reduction.curve_points) is not tuple:
                raise TypeError("E3b curve point container must be exact")
            for point in reduction.curve_points:
                if type(point) is not E3bCurvePoint:
                    raise TypeError("E3b curve points must be exact")
                for field in _E3B_INTERVAL_FIELDS:
                    interval = getattr(point, field)
                    if type(interval) is not E3bIntervalEstimate:
                        raise TypeError("E3b curve intervals must be exact")
                    interval.__post_init__()
        reduction.__post_init__()
        _require_current_sha256(
            "E3b reduction",
            sealed_sha256=reduction.sha256,
            current_sha256=content_sha256(reduction.to_dict()),
        )
    source.__post_init__()
    _require_current_sha256(
        "E3b stage source",
        sealed_sha256=source.sha256,
        current_sha256=content_sha256(source.to_dict()),
    )


def _revalidate_power_source(
    source: ConfirmationFamilyPowerReductionArtifact,
) -> None:
    """Replay the family decision and reject stale cached source identities."""

    plan = source.plan
    if type(plan) is not ConfirmationFamilyPowerPlan:
        raise TypeError("family-power plan must be exact")
    if type(plan.family) is not ConfirmationFamilyIdentity:
        raise TypeError("family-power identity must be exact")
    if (
        type(plan.power_sizing) is not PowerSizingPlan
        or type(plan.power_sizing.power_grid) is not tuple
        or type(plan.power_sizing.pilot_log_standard_deviations) is not tuple
        or len(plan.power_sizing.pilot_log_standard_deviations)
        != len(PRIMARY_CONTRASTS)
        or any(
            type(row) is not tuple or len(row) != 2
            for row in plan.power_sizing.pilot_log_standard_deviations
        )
        or any(type(row) is not ContrastPower for row in plan.power_sizing.power_grid)
    ):
        raise TypeError("family-power sizing must be exact")
    if type(source.run_bindings) is not tuple or any(
        type(row) is not RawEvidenceRunBinding for row in source.run_bindings
    ):
        raise TypeError("family-power run bindings must be exact")
    plan.family.__post_init__()
    _require_current_sha256(
        "confirmation family",
        sealed_sha256=plan.family.sha256,
        current_sha256=content_sha256(plan.family),
    )
    plan.__post_init__()
    expected_reason = (
        "registered_family_power_target_met"
        if plan.status == "POWERED"
        else "registered_family_underpowered"
    )
    if plan.reason_code != expected_reason:
        raise ValueError("family-power reason differs from its sealed decision")
    _require_current_sha256(
        "family-power plan",
        sealed_sha256=plan.sha256,
        current_sha256=content_sha256(plan),
    )
    for binding in source.run_bindings:
        binding.__post_init__()
        _require_current_sha256(
            "family-power run binding",
            sealed_sha256=binding.sha256,
            current_sha256=content_sha256(binding),
        )
    source.__post_init__()
    _require_current_sha256(
        "family-power source",
        sealed_sha256=source.sha256,
        current_sha256=content_sha256(source),
    )


@dataclass(frozen=True)
class CrossFamilyInteractionBinding:
    """One receipt-bound formal role observation for an interaction reducer.

    This is deliberately an input binding rather than a metric row: the
    presentation layer must not invent an interaction estimate before the
    registered reducer has complete formal evidence.  A LightCone binding is
    valid only after the E2 recipe seal; ``lightcone_template`` is not a role
    accepted by this type.
    """

    schema_version: int
    cell_id: str
    scientific_role: str
    role_authority_sha256: str | None
    sealed_e2_recipe_receipt_sha256: str | None
    model: str
    context: int
    load: str
    block: int
    block_phase: str
    paired_block_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("interaction binding schema is unsupported")
        if not _is_sha256(self.cell_id):
            raise ValueError("interaction cell ID must be a SHA-256")
        for label, value in (
            ("interaction model", self.model),
            ("interaction load", self.load),
        ):
            _require_text(label, value)
        if self.scientific_role not in CONFIRMATION_METHOD_ROLES:
            raise ValueError("interaction binding has no formal scientific role")
        if type(self.context) is not int or self.context < 1:
            raise ValueError("interaction context must be positive")
        if type(self.block) is not int or self.block in PILOT_BLOCKS or self.block < 0:
            raise ValueError("formal interaction bindings require a final paired block")
        if self.block_phase != "final_candidate":
            raise ValueError(
                "formal interaction bindings require final_candidate phase"
            )
        for label, value in (
            ("interaction paired-block", self.paired_block_sha256),
            ("interaction evidence", self.evidence_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{label} SHA-256 is invalid")
        adaptive = self.scientific_role in {"tts", "l0_naive", "lightcone"}
        if adaptive != (self.role_authority_sha256 is not None):
            raise ValueError("interaction role authority coverage is invalid")
        if self.role_authority_sha256 is not None and not _is_sha256(
            self.role_authority_sha256
        ):
            raise ValueError("interaction role authority SHA-256 is invalid")
        if self.scientific_role == "lightcone":
            if not _is_sha256(self.sealed_e2_recipe_receipt_sha256):
                raise ValueError(
                    "LightCone interaction input requires sealed E2 receipt"
                )
        elif self.sealed_e2_recipe_receipt_sha256 is not None:
            raise ValueError("only LightCone interaction inputs carry an E2 receipt")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class CrossFamilyInteractionReducerArtifact:
    """Content-bound, non-formal input-shape contract for interactions.

    It records a bounded structural example of role/axis bindings, but does not
    prove registry-owned final-block coverage, a typed E2 seal, native
    completion evidence, or full factorial coverage.  A later formal binder
    must establish those authorities.  Its only truthful status is
    ``UNRESOLVED`` and it carries no numerical estimate.
    """

    schema_version: int
    protocol_sha256: str
    status: str
    registry_sha256: str
    e0_repetition_authority_sha256: str
    runtime_sha256: str
    split_sha256: str
    bindings: tuple[CrossFamilyInteractionBinding, ...]
    input_manifest_sha256: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("interaction reducer schema is unsupported")
        if self.protocol_sha256 != CROSS_FAMILY_INTERACTION_REDUCER_PROTOCOL_SHA256:
            raise ValueError("interaction reducer protocol differs from registration")
        if self.status != "UNRESOLVED":
            raise ValueError("interaction reducer cannot claim a formal result")
        for label, value in (
            ("interaction registry", self.registry_sha256),
            ("E0 repetition authority", self.e0_repetition_authority_sha256),
            ("interaction runtime", self.runtime_sha256),
            ("interaction split", self.split_sha256),
            ("interaction input manifest", self.input_manifest_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{label} SHA-256 is invalid")
        if self.e0_repetition_authority_sha256 != self.registry_sha256:
            raise ValueError("E0 repetition authority must bind the exact registry")
        if (
            type(self.bindings) is not tuple
            or not self.bindings
            or any(
                type(row) is not CrossFamilyInteractionBinding for row in self.bindings
            )
        ):
            raise TypeError("interaction bindings must be a non-empty exact tuple")
        for row in self.bindings:
            row.__post_init__()
        if tuple(sorted(row.sha256 for row in self.bindings)) != tuple(
            row.sha256 for row in self.bindings
        ):
            raise ValueError(
                "interaction bindings must be canonical by content SHA-256"
            )
        if len({row.cell_id for row in self.bindings}) != len(self.bindings):
            raise ValueError("interaction bindings cannot reuse a cell")
        if self.input_manifest_sha256 != content_sha256(
            tuple(row.sha256 for row in self.bindings)
        ):
            raise ValueError("interaction input manifest differs from bindings")
        if self.reason_codes != _canonical_reasons(self.reason_codes):
            raise ValueError("interaction reducer reasons are not canonical")
        if not self.reason_codes:
            raise ValueError("unresolved interaction reducer needs a named reason")

        for level in (
            lambda row: row.model,
            lambda row: str(row.context),
            lambda row: row.load,
        ):
            groups: dict[tuple[str, int], list[CrossFamilyInteractionBinding]] = {}
            for row in self.bindings:
                groups.setdefault((level(row), row.block), []).append(row)
            if len({key[0] for key in groups}) < 2:
                raise ValueError(
                    "interaction reducer needs at least two registered levels"
                )
            for rows in groups.values():
                if {row.scientific_role for row in rows} != set(
                    CONFIRMATION_METHOD_ROLES
                ) or len(rows) != len(CONFIRMATION_METHOD_ROLES):
                    raise ValueError(
                        "interaction reducer requires one formal role per paired block"
                    )
                if len({row.paired_block_sha256 for row in rows}) != 1:
                    raise ValueError(
                        "interaction roles must share one paired-block receipt"
                    )
                by_role = {row.scientific_role: row for row in rows}
                if (
                    by_role["tts"].role_authority_sha256
                    != by_role["l0_naive"].role_authority_sha256
                ):
                    raise ValueError(
                        "TTS and L0-naive interaction inputs must share one frozen recipe authority"
                    )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def _revalidate_interaction_source(
    source: CrossFamilyInteractionReducerArtifact,
) -> None:
    source.__post_init__()
    for binding in source.bindings:
        _require_current_sha256(
            "interaction binding",
            sealed_sha256=binding.sha256,
            current_sha256=content_sha256(binding),
        )
    _require_current_sha256(
        "interaction reducer source",
        sealed_sha256=source.sha256,
        current_sha256=content_sha256(source),
    )


@dataclass(frozen=True)
class ProductionSourceBinding:
    kind: SourceKind
    artifact_sha256: str | None
    status: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in _SOURCE_KINDS:
            raise ValueError("production source kind is unsupported")
        _require_text("production source status", self.status)
        allowed_statuses = {
            "e3b_long_context_stage_reducer": {"MISSING", "UNRESOLVED"},
            "industrial_schema_v3_reducer": {"MISSING", "UNRESOLVED"},
            "confirmation_family_power_reduction": {
                "MISSING",
                "POWERED",
                "UNDERPOWERED",
            },
        }
        if self.status not in allowed_statuses[self.kind]:
            raise ValueError("production source status is unsupported")
        if self.status == "MISSING":
            if self.artifact_sha256 is not None:
                raise ValueError("missing production sources cannot carry a digest")
        elif not _is_sha256(self.artifact_sha256):
            raise ValueError("present production sources require a SHA-256")
        if self.reason_codes != _canonical_reasons(self.reason_codes):
            raise ValueError("production source reasons must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "artifact_sha256": self.artifact_sha256,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
        }


def _validate_e3b_specification(
    value: dict[str, Any],
    *,
    status: OutputStatus,
    source_sha256: str | None,
) -> None:
    fields = frozenset(
        {
            "id",
            "kind",
            "status",
            "source_sha256",
            "reason_codes",
            "render_mode",
            "source_stage_status",
            "source_evidence_level",
            "claim",
            "axes",
            "uncertainty",
            "value_fields",
            "panels",
        }
    )
    _exact_fields(value, fields=fields, label="E3b figure specification")
    if status is not OutputStatus.BLOCKED:
        raise ValueError("current E3b production figure must remain BLOCKED")
    stage_status = value["source_stage_status"]
    if stage_status not in {"MISSING", "UNRESOLVED"}:
        raise ValueError("E3b figure source status is unsupported")
    if (stage_status == "MISSING") != (source_sha256 is None):
        raise ValueError("E3b figure source identity is partial")
    if stage_status == "MISSING":
        if value["source_evidence_level"] is not None:
            raise ValueError("missing E3b source cannot claim an evidence level")
    elif value["source_evidence_level"] not in {
        "RAW_DIAGNOSTIC_OBSERVED_UNATTESTED",
        "RAW_UNRESOLVED",
    }:
        raise ValueError("E3b figure evidence level is unsupported")
    if value["render_mode"] != "vector-native" or value["axes"] != {
        "x": {
            "field": "context_tokens",
            "unit": "tokens",
            "scale": "log",
            "measured_grid": list(E3B_CONTEXT_GRID),
        },
        "y": {
            "transform": "metric_specific",
            "interpolation": "registered_natural_cubic_log_context_spline",
        },
    }:
        raise ValueError("E3b figure changes its registered rendering axes")
    if value["uncertainty"] != {
        "confidence": REGISTERED_CONFIDENCE,
        "method": "paired_hierarchical_block_then_request_percentile",
        "refit_spline_each_sample": True,
        "interval_fields": ["estimate", "lower", "upper"],
    }:
        raise ValueError("E3b figure changes its registered uncertainty")
    if value["value_fields"] != [
        "candidate_fitted_metric",
        "baseline_fitted_metric",
        "candidate_elasticity",
        "baseline_elasticity",
        "paired_elasticity_difference",
        "candidate_curvature",
        "baseline_curvature",
        "paired_curvature_difference",
    ]:
        raise ValueError("E3b figure value fields differ")
    panels = _exact_list(value["panels"], label="E3b panels")
    expected_ids = tuple(":".join(item.value for item in row) for row in _E3B_PANELS)
    if len(panels) != len(expected_ids):
        raise ValueError("E3b figure does not cover the registered panels")
    for raw_panel, (panel_id, identity) in zip(
        panels,
        zip(expected_ids, _E3B_PANELS, strict=True),
        strict=True,
    ):
        panel = _exact_fields(
            raw_panel,
            fields=frozenset(
                {
                    "panel_id",
                    "metric",
                    "candidate_method",
                    "baseline_method",
                    "status",
                    "source_reduction_sha256",
                    "source_reduction_status",
                    "reason_codes",
                    "points",
                    "crossover",
                }
            ),
            label="E3b panel",
        )
        metric, candidate, baseline = identity
        if (
            panel["panel_id"] != panel_id
            or panel["metric"] != metric.value
            or panel["candidate_method"] != candidate.value
            or panel["baseline_method"] != baseline.value
            or panel["status"] != "BLOCKED"
        ):
            raise ValueError("E3b panel identity differs from the registered set")
        if not _reason_list(panel["reason_codes"], label="E3b panel reasons"):
            raise ValueError("blocked E3b panel requires a named reason")
        reduction_sha256 = panel["source_reduction_sha256"]
        reduction_status = panel["source_reduction_status"]
        if (reduction_sha256 is None) != (reduction_status is None):
            raise ValueError("E3b panel source reduction identity is partial")
        if reduction_sha256 is not None and (
            not _is_sha256(reduction_sha256)
            or reduction_status not in {"OBSERVED", "UNRESOLVED"}
        ):
            raise ValueError("E3b panel source reduction is invalid")
        if stage_status == "MISSING" and reduction_sha256 is not None:
            raise ValueError("missing E3b stage cannot contain a reduction")
        points = _exact_list(panel["points"], label="E3b panel points")
        if len(points) != len(E3B_CONTEXT_GRID):
            raise ValueError("E3b panel changes the registered context grid")
        for raw_point, context in zip(points, E3B_CONTEXT_GRID, strict=True):
            point = _exact_fields(
                raw_point,
                fields=frozenset({"context_tokens", "values"}),
                label="E3b figure point",
            )
            if point != {"context_tokens": context, "values": None}:
                raise ValueError(
                    "unresolved E3b publication measurements must remain null"
                )
        crossover = _exact_fields(
            panel["crossover"],
            fields=frozenset(
                {
                    "status",
                    "reason_code",
                    "first_bracket_tokens",
                    "root_tokens",
                    "root_interval_tokens",
                }
            ),
            label="E3b crossover",
        )
        _require_text("E3b crossover reason", crossover["reason_code"])
        if crossover["status"] != "UNRESOLVED" or any(
            crossover[field] is not None
            for field in (
                "first_bracket_tokens",
                "root_tokens",
                "root_interval_tokens",
            )
        ):
            raise ValueError("unresolved E3b crossover must remain null")


def _validate_power_specification(
    value: dict[str, Any],
    *,
    status: OutputStatus,
    source_sha256: str | None,
) -> None:
    _exact_fields(
        value,
        fields=frozenset(
            {
                "id",
                "kind",
                "status",
                "source_sha256",
                "reason_codes",
                "scientific_status",
                "family_sha256",
                "selected_final_blocks",
                "target_power",
                "evidence_role",
                "formal_result_eligible",
                "independent_unit",
                "pilot_blocks",
                "metric_directions",
                "rows",
            }
        ),
        label="power table specification",
    )
    if (
        value["evidence_role"] != "preregistered_power_planning_only"
        or value["formal_result_eligible"] is not False
        or value["independent_unit"] != "excluded_paired_pilot_block"
        or value["pilot_blocks"] != 4
        or value["metric_directions"]
        != {
            "power": "higher",
            "final_blocks": "resource_axis",
            "pilot_log_standard_deviation": "descriptive",
        }
    ):
        raise ValueError("power table changes its registered semantics")
    scientific_status = value["scientific_status"]
    rows = _exact_list(value["rows"], label="power table rows")
    if scientific_status == "MISSING":
        if (
            status is not OutputStatus.BLOCKED
            or source_sha256 is not None
            or value["family_sha256"] is not None
            or value["selected_final_blocks"] is not None
            or value["target_power"] is not None
            or rows
        ):
            raise ValueError("missing power source must keep its payload null")
        return
    if scientific_status not in {"POWERED", "UNDERPOWERED"}:
        raise ValueError("power table scientific status is unsupported")
    expected_status = (
        OutputStatus.READY if scientific_status == "POWERED" else OutputStatus.BLOCKED
    )
    if status is not expected_status or not _is_sha256(source_sha256):
        raise ValueError("power table readiness differs from its scientific status")
    if not _is_sha256(value["family_sha256"]):
        raise ValueError("power table family SHA-256 is invalid")
    if value["target_power"] != PRIMARY_TARGET_POWER:
        raise ValueError("power table changes the preregistered target")
    selected = value["selected_final_blocks"]
    if scientific_status == "POWERED":
        if type(selected) is not int or not 12 <= selected <= 20:
            raise ValueError("POWERED table requires 12--20 final blocks")
    elif selected is not None:
        raise ValueError("UNDERPOWERED table cannot select final blocks")
    expected_keys = tuple(
        (contrast, blocks) for blocks in range(12, 21) for contrast in PRIMARY_CONTRASTS
    )
    if len(rows) != len(expected_keys):
        raise ValueError("power table changes the registered grid")
    powers: dict[tuple[str, int], float] = {}
    deviations: dict[str, float] = {}
    for raw_row, (contrast, blocks) in zip(rows, expected_keys, strict=True):
        row = _exact_fields(
            raw_row,
            fields=frozenset(
                {
                    "contrast",
                    "final_blocks",
                    "power",
                    "pilot_log_standard_deviation",
                }
            ),
            label="power table row",
        )
        power = row["power"]
        deviation = row["pilot_log_standard_deviation"]
        if (
            row["contrast"] != contrast
            or row["final_blocks"] != blocks
            or not isinstance(power, (int, float))
            or isinstance(power, bool)
            or not math.isfinite(float(power))
            or not 0.0 <= float(power) <= 1.0
            or not isinstance(deviation, (int, float))
            or isinstance(deviation, bool)
            or not math.isfinite(float(deviation))
            or float(deviation) <= 0.0
        ):
            raise ValueError("power table row is invalid")
        checked_power = float(power)
        checked_deviation = float(deviation)
        prior_deviation = deviations.setdefault(contrast, checked_deviation)
        if checked_deviation != prior_deviation:
            raise ValueError("power table changes pilot variance across block counts")
        powers[(contrast, blocks)] = checked_power
    for contrast in PRIMARY_CONTRASTS:
        contrast_powers = tuple(powers[(contrast, blocks)] for blocks in range(12, 21))
        if any(later < earlier for earlier, later in pairwise(contrast_powers)):
            raise ValueError("power table must be monotone in final block count")
    first_powered = next(
        (
            blocks
            for blocks in range(12, 21)
            if all(
                powers[(contrast, blocks)] >= PRIMARY_TARGET_POWER
                for contrast in PRIMARY_CONTRASTS
            )
        ),
        None,
    )
    if (
        scientific_status == "POWERED"
        and selected != first_powered
        or scientific_status == "UNDERPOWERED"
        and first_powered is not None
    ):
        raise ValueError("power table decision differs from its registered grid")


def _validate_claim_specification(
    value: dict[str, Any],
    *,
    status: OutputStatus,
    source_sha256: str | None,
) -> None:
    _exact_fields(
        value,
        fields=frozenset(
            {
                "id",
                "kind",
                "status",
                "source_sha256",
                "reason_codes",
                "reducer_status",
                "gpu_evidence",
                "metric_directions",
                "uncertainty",
                "p99_rows",
                "primary_rows",
                "secondary_rows",
                "interaction_reducer_status",
                "interaction_rows",
            }
        ),
        label="formal claim table specification",
    )
    if status is not OutputStatus.BLOCKED:
        raise ValueError("current formal claim table must remain BLOCKED")
    reducer_status = value["reducer_status"]
    if reducer_status not in {"MISSING", "UNRESOLVED"}:
        raise ValueError("formal claim reducer status is unsupported")
    if (reducer_status == "MISSING") != (source_sha256 is None):
        raise ValueError("formal claim source identity is partial")
    expected_gpu_evidence = None if reducer_status == "MISSING" else "UNMEASURED"
    if value["gpu_evidence"] != expected_gpu_evidence:
        raise ValueError("formal claim GPU evidence status is invalid")
    if value["metric_directions"] != {
        "aggregate_latency_p99_ms": "lower",
        "mean_relative_gain": "higher",
        "adjusted_p_value": "lower",
    } or value["uncertainty"] != {
        "primary": "paired_BCa_95_percent",
        "secondary": "paired_BCa_95_percent_descriptive",
        "interactions": "registered_cross_family_reducer_required",
        "p99": "time_block_bootstrap_95_percent_when_registered",
    }:
        raise ValueError("formal claim table changes registered semantics")

    p99_rows = _exact_list(value["p99_rows"], label="p99 claim rows")
    if len(p99_rows) != len(CONFIRMATION_METHOD_ROLES):
        raise ValueError("p99 claim rows do not cover core methods")
    for raw_row, method in zip(p99_rows, CONFIRMATION_METHOD_ROLES, strict=True):
        row = _exact_fields(
            raw_row,
            fields=frozenset(
                {
                    "method",
                    "formal_claim_status",
                    "request_count_gate_status",
                    "anchor_id",
                    "completed_requests",
                    "minimum_completions",
                    "observed_p99_ms",
                    "reason_codes",
                }
            ),
            label="p99 claim row",
        )
        minimum = row["minimum_completions"]
        if (
            row["method"] != method
            or row["formal_claim_status"] != "UNRESOLVED"
            or row["request_count_gate_status"] not in {"CLAIMABLE", "UNRESOLVED"}
            or row["completed_requests"] is not None
            or row["observed_p99_ms"] is not None
            or (minimum is not None and (type(minimum) is not int or minimum < 0))
            or (minimum is not None and 0 < minimum < P99_MINIMUM_COMPLETIONS)
        ):
            raise ValueError("unresolved p99 publication measurements must remain null")
        if row["anchor_id"] is not None:
            _require_text("p99 anchor", row["anchor_id"])
        if not _reason_list(row["reason_codes"], label="p99 reasons"):
            raise ValueError("unresolved p99 row requires a named reason")
        if row["request_count_gate_status"] == "CLAIMABLE" and (
            row["anchor_id"] is None
            or minimum is None
            or minimum < P99_MINIMUM_COMPLETIONS
        ):
            raise ValueError("claimable p99 count gate lacks registered authority")
        if reducer_status == "MISSING" and (
            row["request_count_gate_status"] != "UNRESOLVED"
            or row["anchor_id"] is not None
            or minimum is not None
        ):
            raise ValueError("missing reducer cannot expose p99 source metadata")

    primary_rows = _exact_list(value["primary_rows"], label="primary claim rows")
    if len(primary_rows) != len(PRIMARY_CONTRASTS):
        raise ValueError("primary claim rows do not cover the registered family")
    numeric_fields = (
        "mean_relative_gain",
        "ci_lower_relative_gain",
        "ci_upper_relative_gain",
        "adjusted_p_value",
        "rejected",
    )
    for raw_row, contrast in zip(primary_rows, PRIMARY_CONTRASTS, strict=True):
        row = _exact_fields(
            raw_row,
            fields=frozenset(
                {
                    "contrast",
                    "formal_claim_status",
                    *numeric_fields,
                    "independent_unit",
                    "adjustment_procedure",
                    "reason_codes",
                }
            ),
            label="primary claim row",
        )
        if (
            row["contrast"] != contrast
            or row["formal_claim_status"] != "UNRESOLVED"
            or any(row[field] is not None for field in numeric_fields)
        ):
            raise ValueError("unresolved primary statistics must remain null")
        for field in ("independent_unit", "adjustment_procedure"):
            if row[field] is not None:
                _require_text(f"primary {field}", row[field])
        if row["independent_unit"] not in {None, "paired_block"}:
            raise ValueError("primary independent unit is not registered")
        if row["adjustment_procedure"] not in {None, "holm"}:
            raise ValueError("primary adjustment procedure is not registered")
        if not _reason_list(row["reason_codes"], label="primary claim reasons"):
            raise ValueError("unresolved primary row requires a named reason")
        if reducer_status == "MISSING" and (
            row["independent_unit"] is not None
            or row["adjustment_procedure"] is not None
        ):
            raise ValueError("missing reducer cannot expose primary source metadata")

    secondary_rows = _exact_list(value["secondary_rows"], label="secondary claim rows")
    if len(secondary_rows) != len(SECONDARY_CONTRASTS):
        raise ValueError("secondary claim rows do not cover the registered family")
    secondary_numeric_fields = (
        "mean_relative_gain",
        "ci_lower_relative_gain",
        "ci_upper_relative_gain",
        "raw_p_value",
    )
    for raw_row, contrast in zip(secondary_rows, SECONDARY_CONTRASTS, strict=True):
        row = _exact_fields(
            raw_row,
            fields=frozenset(
                {
                    "contrast",
                    "formal_claim_status",
                    *secondary_numeric_fields,
                    "independent_unit",
                    "multiplicity_role",
                    "reason_codes",
                }
            ),
            label="secondary claim row",
        )
        if (
            row["contrast"] != contrast
            or row["formal_claim_status"] != "UNRESOLVED"
            or any(row[field] is not None for field in secondary_numeric_fields)
            or row["independent_unit"] not in {None, "paired_block"}
            or row["multiplicity_role"] != "descriptive_secondary_not_in_holm_family"
            or not _reason_list(row["reason_codes"], label="secondary claim reasons")
        ):
            raise ValueError("unresolved secondary statistics are invalid")
        if reducer_status == "MISSING" and row["independent_unit"] is not None:
            raise ValueError("missing reducer cannot expose secondary source metadata")

    interaction_reducer_status = value["interaction_reducer_status"]
    if interaction_reducer_status not in {"MISSING", "UNRESOLVED"}:
        raise ValueError("interaction reducer status is unsupported")
    interaction_rows = _exact_list(
        value["interaction_rows"], label="interaction claim rows"
    )
    expected_interactions = CROSS_FAMILY_INTERACTION_AXES
    if len(interaction_rows) != len(expected_interactions):
        raise ValueError("interaction rows do not cover the registered axes")
    for raw_row, axis in zip(interaction_rows, expected_interactions, strict=True):
        row = _exact_fields(
            raw_row,
            fields=frozenset(
                {
                    "axis",
                    "formal_claim_status",
                    "estimate",
                    "ci_lower",
                    "ci_upper",
                    "independent_unit",
                    "reducer_protocol_sha256",
                    "source_sha256",
                    "input_manifest_sha256",
                    "reason_codes",
                }
            ),
            label="interaction claim row",
        )
        if (
            row["axis"] != axis
            or row["formal_claim_status"] != "UNRESOLVED"
            or any(
                row[field] is not None for field in ("estimate", "ci_lower", "ci_upper")
            )
            or row["reducer_protocol_sha256"]
            != CROSS_FAMILY_INTERACTION_REDUCER_PROTOCOL_SHA256
            or not _reason_list(row["reason_codes"], label="interaction claim reasons")
        ):
            raise ValueError("unresolved interaction statistics must remain null")
        if interaction_reducer_status == "MISSING":
            if (
                row["independent_unit"] is not None
                or row["source_sha256"] is not None
                or row["input_manifest_sha256"] is not None
            ):
                raise ValueError(
                    "missing interaction reducer cannot expose source metadata"
                )
        elif (
            row["independent_unit"] != "paired_block"
            or not _is_sha256(row["source_sha256"])
            or not _is_sha256(row["input_manifest_sha256"])
        ):
            raise ValueError("interaction reducer source metadata is invalid")


def _validate_specification_payload(
    value: dict[str, Any],
    *,
    kind: SpecificationKind,
    status: OutputStatus,
    source_sha256: str | None,
) -> None:
    if kind == "e3b_long_context_spline_crossover":
        _validate_e3b_specification(
            value,
            status=status,
            source_sha256=source_sha256,
        )
    elif kind == "confirmation_family_power_grid":
        _validate_power_specification(
            value,
            status=status,
            source_sha256=source_sha256,
        )
    else:
        _validate_claim_specification(
            value,
            status=status,
            source_sha256=source_sha256,
        )


@dataclass(frozen=True)
class MachineReadableSpecification:
    """An immutable, already-canonical JSON figure or table specification."""

    spec_id: str
    kind: SpecificationKind
    status: OutputStatus
    source_sha256: str | None
    reason_codes: tuple[str, ...]
    canonical_payload: bytes

    def __post_init__(self) -> None:
        _require_text("production specification ID", self.spec_id)
        if type(self.kind) is not str or self.kind not in {
            "e3b_long_context_spline_crossover",
            "confirmation_family_power_grid",
            "formal_claim_status",
        }:
            raise ValueError("production specification kind is unsupported")
        if type(self.status) is not OutputStatus:
            raise TypeError("production specification status must be exact")
        if self.source_sha256 is not None and not _is_sha256(self.source_sha256):
            raise ValueError("production specification source SHA-256 is invalid")
        if self.reason_codes != _canonical_reasons(self.reason_codes):
            raise ValueError("production specification reasons are not canonical")
        if self.status is OutputStatus.BLOCKED and not self.reason_codes:
            raise ValueError("blocked production specification needs a named reason")
        payload = _strict_json_object(self.canonical_payload)
        expected_header = {
            "id": self.spec_id,
            "kind": self.kind,
            "status": self.status.value,
            "source_sha256": self.source_sha256,
            "reason_codes": list(self.reason_codes),
        }
        if any(payload.get(key) != value for key, value in expected_header.items()):
            raise ValueError("production specification header differs from its payload")
        _validate_specification_payload(
            payload,
            kind=self.kind,
            status=self.status,
            source_sha256=self.source_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return _strict_json_object(self.canonical_payload)

    @cached_property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_payload).hexdigest()


@dataclass(frozen=True)
class ProductionOutputArtifact:
    schema_version: int
    protocol_sha256: str
    status: OutputStatus
    blocker_codes: tuple[str, ...]
    sources: tuple[ProductionSourceBinding, ...]
    figure: MachineReadableSpecification
    tables: tuple[MachineReadableSpecification, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 3:
            raise ValueError("production output artifact schema is unsupported")
        if self.protocol_sha256 != PRODUCTION_OUTPUT_PROTOCOL_SHA256:
            raise ValueError("production output artifact changes its protocol")
        if type(self.status) is not OutputStatus:
            raise TypeError("production output status must be exact")
        if type(self.sources) is not tuple or type(self.tables) is not tuple:
            raise TypeError("production source and table containers must be tuples")
        if self.blocker_codes != _canonical_reasons(self.blocker_codes):
            raise ValueError("production blockers must be sorted and unique")
        if any(type(source) is not ProductionSourceBinding for source in self.sources):
            raise TypeError("production source binding must be exact")
        if tuple(source.kind for source in self.sources) != _SOURCE_KINDS:
            raise ValueError("production source coverage is not canonical")
        if (
            type(self.figure) is not MachineReadableSpecification
            or self.figure.kind != "e3b_long_context_spline_crossover"
        ):
            raise TypeError("production output requires the exact E3b figure spec")
        if any(
            type(table) is not MachineReadableSpecification for table in self.tables
        ):
            raise TypeError("production output table specification must be exact")
        if tuple(table.kind for table in self.tables) != (
            "confirmation_family_power_grid",
            "formal_claim_status",
        ):
            raise TypeError("production output table coverage is not canonical")
        for source in self.sources:
            source.__post_init__()
        self.figure.__post_init__()
        for table in self.tables:
            table.__post_init__()

        e3b_source, industrial_source, power_source = self.sources
        power_table, claim_table = self.tables
        if (
            self.figure.source_sha256 != e3b_source.artifact_sha256
            or claim_table.source_sha256 != industrial_source.artifact_sha256
            or power_table.source_sha256 != power_source.artifact_sha256
        ):
            raise ValueError("production specifications differ from source bindings")
        figure_payload = self.figure.to_dict()
        power_payload = power_table.to_dict()
        claim_payload = claim_table.to_dict()
        if (
            figure_payload["source_stage_status"] != e3b_source.status
            or claim_payload["reducer_status"] != industrial_source.status
            or power_payload["scientific_status"] != power_source.status
        ):
            raise ValueError("production specification statuses differ from sources")
        expected_blockers = {
            (
                "e3b_long_context_stage_artifact_missing"
                if e3b_source.status == "MISSING"
                else "e3b_formal_production_status_unresolved"
            ),
            (
                "industrial_reducer_artifact_missing"
                if industrial_source.status == "MISSING"
                else "industrial_formal_production_status_unresolved"
            ),
        }
        if power_source.status == "MISSING":
            expected_blockers.add(
                "confirmation_family_power_reduction_artifact_missing"
            )
        elif power_source.status == "UNDERPOWERED":
            expected_blockers.add("confirmation_family_underpowered")
        if self.blocker_codes != tuple(sorted(expected_blockers)):
            raise ValueError("production blockers differ from source statuses")
        expected_status = (
            OutputStatus.BLOCKED if self.blocker_codes else OutputStatus.READY
        )
        if self.status is not expected_status:
            raise ValueError("production output status differs from its blockers")
        if (
            self.figure.status is not OutputStatus.BLOCKED
            or claim_table.status is not OutputStatus.BLOCKED
            or power_table.status
            is not (
                OutputStatus.READY
                if power_source.status == "POWERED"
                else OutputStatus.BLOCKED
            )
        ):
            raise ValueError("production spec readiness differs from source status")
        if self.status is OutputStatus.READY and any(
            value.status is not OutputStatus.READY
            for value in (self.figure, *self.tables)
        ):
            raise ValueError("ready production output contains a blocked spec")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "lightcone_figure_table_production_output",
            "protocol_sha256": self.protocol_sha256,
            "status": self.status.value,
            "blocker_codes": list(self.blocker_codes),
            "sources": [source.to_dict() for source in self.sources],
            "figures": [self.figure.to_dict()],
            "tables": [table.to_dict() for table in self.tables],
        }

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict()) + b"\n"

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def _output_status(value: object, *, label: str) -> OutputStatus:
    if type(value) is not str:
        raise TypeError(f"{label} must be text")
    try:
        return OutputStatus(value)
    except ValueError as error:
        raise ValueError(f"{label} is unsupported") from error


def _source_binding_from_dict(value: object) -> ProductionSourceBinding:
    raw = _exact_fields(
        value,
        fields=frozenset({"kind", "artifact_sha256", "status", "reason_codes"}),
        label="production source binding",
    )
    kind = raw["kind"]
    if kind not in _SOURCE_KINDS:
        raise ValueError("production source kind is unsupported")
    artifact_sha256 = raw["artifact_sha256"]
    if artifact_sha256 is not None and type(artifact_sha256) is not str:
        raise TypeError("production source SHA-256 must be text or null")
    if type(raw["status"]) is not str:
        raise TypeError("production source status must be text")
    return ProductionSourceBinding(
        kind=kind,
        artifact_sha256=artifact_sha256,
        status=raw["status"],
        reason_codes=_reason_list(
            raw["reason_codes"],
            label="production source reasons",
        ),
    )


def _specification_from_dict(value: object) -> MachineReadableSpecification:
    if type(value) is not dict:
        raise TypeError("production specification must be a JSON object")
    for field in ("id", "kind", "status", "source_sha256", "reason_codes"):
        if field not in value:
            raise ValueError("production specification lacks its canonical header")
    spec_id = value["id"]
    kind = value["kind"]
    source_sha256 = value["source_sha256"]
    if type(spec_id) is not str or kind not in {
        "e3b_long_context_spline_crossover",
        "confirmation_family_power_grid",
        "formal_claim_status",
    }:
        raise ValueError("production specification identity is unsupported")
    if source_sha256 is not None and type(source_sha256) is not str:
        raise TypeError("production specification source SHA-256 must be text or null")
    return MachineReadableSpecification(
        spec_id=spec_id,
        kind=kind,
        status=_output_status(value["status"], label="production spec status"),
        source_sha256=source_sha256,
        reason_codes=_reason_list(
            value["reason_codes"],
            label="production specification reasons",
        ),
        canonical_payload=_canonical_json_bytes(value),
    )


def production_output_artifact_from_json_bytes(
    body: bytes,
    *,
    e3b_stage: E3bLongContextStageArtifact | None,
    industrial_reduction: IndustrialReducerArtifact | None,
    family_power_reduction: ConfirmationFamilyPowerReductionArtifact | None,
    interaction_reduction: CrossFamilyInteractionReducerArtifact | None = None,
) -> ProductionOutputArtifact:
    """Strictly reopen the canonical output document.

    Duplicate keys, non-finite JSON constants, unknown fields, noncanonical
    whitespace/key order, missing final newline, and joint status/payload
    mutations all fail closed.  The exact typed sources are mandatory so a
    content-rehashed presentation document cannot replace reducer output while
    retaining the reducer's source SHA-256.
    """

    if type(body) is not bytes:
        raise TypeError("production output document must be bytes")
    if not body.endswith(b"\n") or body.endswith(b"\n\n"):
        raise ValueError("production output document needs one final newline")
    raw = _strict_json_object(body[:-1])
    _exact_fields(
        raw,
        fields=frozenset(
            {
                "schema_version",
                "kind",
                "protocol_sha256",
                "status",
                "blocker_codes",
                "sources",
                "figures",
                "tables",
            }
        ),
        label="production output document",
    )
    if raw["kind"] != "lightcone_figure_table_production_output":
        raise ValueError("production output document kind is invalid")
    sources = _exact_list(raw["sources"], label="production sources")
    figures = _exact_list(raw["figures"], label="production figures")
    tables = _exact_list(raw["tables"], label="production tables")
    if len(sources) != 3 or len(figures) != 1 or len(tables) != 2:
        raise ValueError("production output document coverage is incomplete")
    artifact = ProductionOutputArtifact(
        schema_version=raw["schema_version"],
        protocol_sha256=raw["protocol_sha256"],
        status=_output_status(raw["status"], label="production output status"),
        blocker_codes=_reason_list(
            raw["blocker_codes"],
            label="production blocker codes",
        ),
        sources=tuple(_source_binding_from_dict(value) for value in sources),
        figure=_specification_from_dict(figures[0]),
        tables=tuple(_specification_from_dict(value) for value in tables),
    )
    if artifact.canonical_json_bytes() != body:
        raise ValueError("production output document differs after strict replay")
    expected = build_production_output_artifact(
        e3b_stage=e3b_stage,
        industrial_reduction=industrial_reduction,
        family_power_reduction=family_power_reduction,
        interaction_reduction=interaction_reduction,
    )
    if artifact.to_dict() != expected.to_dict():
        raise ValueError("production output differs from exact typed source replay")
    return artifact


def _specification(
    *,
    spec_id: str,
    kind: SpecificationKind,
    status: OutputStatus,
    source_sha256: str | None,
    reason_codes: tuple[str, ...],
    payload: dict[str, object],
) -> MachineReadableSpecification:
    reasons = _canonical_reasons(reason_codes)
    value = {
        "id": spec_id,
        "kind": kind,
        "status": status.value,
        "source_sha256": source_sha256,
        "reason_codes": list(reasons),
        **payload,
    }
    return MachineReadableSpecification(
        spec_id=spec_id,
        kind=kind,
        status=status,
        source_sha256=source_sha256,
        reason_codes=reasons,
        canonical_payload=_canonical_json_bytes(value),
    )


def _e3b_figure(
    source: E3bLongContextStageArtifact | None,
) -> MachineReadableSpecification:
    source_by_name = (
        {} if source is None else {value.name: value for value in source.reductions}
    )
    reasons = (
        ("e3b_long_context_stage_artifact_missing",)
        if source is None
        else _canonical_reasons(
            ("e3b_formal_production_status_unresolved", *source.reasons)
        )
    )
    panels: list[dict[str, object]] = []
    for metric, candidate, baseline in _E3B_PANELS:
        panel_id = f"{metric.value}:{candidate.value}:{baseline.value}"
        named = source_by_name.get(panel_id)
        panel_reasons = set(reasons)
        if source is not None and named is None:
            panel_reasons.add("e3b_registered_reduction_missing")
        elif named is not None and (
            named.reduction.status is E3bReductionStatus.UNRESOLVED
        ):
            panel_reasons.add(named.reduction.reason_code)
        points = [
            {"context_tokens": context, "values": None} for context in E3B_CONTEXT_GRID
        ]
        panels.append(
            {
                "panel_id": panel_id,
                "metric": metric.value,
                "candidate_method": candidate.value,
                "baseline_method": baseline.value,
                "status": OutputStatus.BLOCKED.value,
                "source_reduction_sha256": (
                    None if named is None else named.reduction.sha256
                ),
                "source_reduction_status": (
                    None if named is None else named.reduction.status.value
                ),
                "reason_codes": sorted(panel_reasons),
                "points": points,
                "crossover": {
                    "status": "UNRESOLVED",
                    "reason_code": (
                        "e3b_long_context_stage_artifact_missing"
                        if source is None
                        else "e3b_formal_production_status_unresolved"
                    ),
                    "first_bracket_tokens": None,
                    "root_tokens": None,
                    "root_interval_tokens": None,
                },
            }
        )
    return _specification(
        spec_id="fig:e3b-long-context-spline-crossover",
        kind="e3b_long_context_spline_crossover",
        status=OutputStatus.BLOCKED,
        source_sha256=None if source is None else source.sha256,
        reason_codes=reasons,
        payload={
            "render_mode": "vector-native",
            "source_stage_status": "MISSING" if source is None else source.status,
            "source_evidence_level": (
                None if source is None else source.evidence_level
            ),
            "claim": "registered E3b long-context spline and crossover",
            "axes": {
                "x": {
                    "field": "context_tokens",
                    "unit": "tokens",
                    "scale": "log",
                    "measured_grid": list(E3B_CONTEXT_GRID),
                },
                "y": {
                    "transform": "metric_specific",
                    "interpolation": "registered_natural_cubic_log_context_spline",
                },
            },
            "uncertainty": {
                "confidence": REGISTERED_CONFIDENCE,
                "method": "paired_hierarchical_block_then_request_percentile",
                "refit_spline_each_sample": True,
                "interval_fields": ["estimate", "lower", "upper"],
            },
            "value_fields": [
                "candidate_fitted_metric",
                "baseline_fitted_metric",
                "candidate_elasticity",
                "baseline_elasticity",
                "paired_elasticity_difference",
                "candidate_curvature",
                "baseline_curvature",
                "paired_curvature_difference",
            ],
            "panels": panels,
        },
    )


def _power_table(
    source: ConfirmationFamilyPowerReductionArtifact | None,
) -> MachineReadableSpecification:
    if source is None:
        return _specification(
            spec_id="tab:confirmation-family-power",
            kind="confirmation_family_power_grid",
            status=OutputStatus.BLOCKED,
            source_sha256=None,
            reason_codes=("confirmation_family_power_reduction_artifact_missing",),
            payload={
                "scientific_status": "MISSING",
                "family_sha256": None,
                "selected_final_blocks": None,
                "target_power": None,
                "evidence_role": "preregistered_power_planning_only",
                "formal_result_eligible": False,
                "independent_unit": "excluded_paired_pilot_block",
                "pilot_blocks": 4,
                "metric_directions": {
                    "power": "higher",
                    "final_blocks": "resource_axis",
                    "pilot_log_standard_deviation": "descriptive",
                },
                "rows": [],
            },
        )
    plan = source.plan.power_sizing
    deviations = dict(plan.pilot_log_standard_deviations)
    grid = {(row.contrast, row.final_blocks): row.power for row in plan.power_grid}
    rows = [
        {
            "contrast": contrast,
            "final_blocks": blocks,
            "power": grid[(contrast, blocks)],
            "pilot_log_standard_deviation": deviations[contrast],
        }
        for blocks in range(12, 21)
        for contrast in PRIMARY_CONTRASTS
    ]
    return _specification(
        spec_id="tab:confirmation-family-power",
        kind="confirmation_family_power_grid",
        status=(
            OutputStatus.READY if source.status == "POWERED" else OutputStatus.BLOCKED
        ),
        source_sha256=source.sha256,
        reason_codes=(source.plan.reason_code,),
        payload={
            "scientific_status": source.status,
            "family_sha256": source.family.sha256,
            "selected_final_blocks": source.selected_final_blocks,
            "target_power": plan.target_power,
            "evidence_role": "preregistered_power_planning_only",
            "formal_result_eligible": False,
            "independent_unit": "excluded_paired_pilot_block",
            "pilot_blocks": 4,
            "metric_directions": {
                "power": "higher",
                "final_blocks": "resource_axis",
                "pilot_log_standard_deviation": "descriptive",
            },
            "rows": rows,
        },
    )


def _claim_table(
    source: IndustrialReducerArtifact | None,
    interaction_source: CrossFamilyInteractionReducerArtifact | None,
) -> MachineReadableSpecification:
    if source is None:
        reasons = ("industrial_reducer_artifact_missing",)
        methods: dict[str, Any] = {}
        contrasts: dict[str, Any] = {}
        secondary_contrasts: dict[str, Any] = {}
        decisions: dict[str, Any] = {}
    else:
        reasons = _canonical_reasons(
            ("industrial_formal_production_status_unresolved", *source.reasons)
        )
        methods = {value.method: value for value in source.methods}
        contrasts = {value.name: value for value in source.primary_contrasts}
        secondary_contrasts = {
            value.name: value for value in source.secondary_contrasts
        }
        decisions = {value.name: value for value in source.holm_family}

    p99_rows: list[dict[str, object]] = []
    for method in CONFIRMATION_METHOD_ROLES:
        reduction = methods.get(method)
        guard = None if reduction is None else reduction.aggregate_latency_p99
        row_reasons = set(reasons)
        if guard is None:
            row_reasons.add("industrial_method_reduction_unavailable")
        p99_rows.append(
            {
                "method": method,
                "formal_claim_status": "UNRESOLVED",
                "request_count_gate_status": (
                    "UNRESOLVED" if guard is None else guard.status
                ),
                "anchor_id": None if guard is None else guard.anchor_id,
                "completed_requests": None,
                "minimum_completions": (
                    None if guard is None else guard.minimum_completions
                ),
                "observed_p99_ms": None,
                "reason_codes": sorted(row_reasons),
            }
        )

    primary_rows: list[dict[str, object]] = []
    for name in PRIMARY_CONTRASTS:
        contrast = contrasts.get(name)
        decision = decisions.get(name)
        row_reasons = set(reasons)
        if contrast is None or decision is None:
            row_reasons.add("industrial_primary_contrast_unavailable")
        primary_rows.append(
            {
                "contrast": name,
                "formal_claim_status": "UNRESOLVED",
                "mean_relative_gain": None,
                "ci_lower_relative_gain": None,
                "ci_upper_relative_gain": None,
                "adjusted_p_value": None,
                "rejected": None,
                "independent_unit": (
                    None if contrast is None else contrast.independent_unit
                ),
                "adjustment_procedure": (
                    None if decision is None else decision.procedure
                ),
                "reason_codes": sorted(row_reasons),
            }
        )
    secondary_rows: list[dict[str, object]] = []
    for name in SECONDARY_CONTRASTS:
        contrast = secondary_contrasts.get(name)
        row_reasons = set(reasons)
        if contrast is None:
            row_reasons.add("industrial_secondary_contrast_unavailable")
        secondary_rows.append(
            {
                "contrast": name,
                "formal_claim_status": "UNRESOLVED",
                "mean_relative_gain": None,
                "ci_lower_relative_gain": None,
                "ci_upper_relative_gain": None,
                "raw_p_value": None,
                "independent_unit": (
                    None if contrast is None else contrast.independent_unit
                ),
                "multiplicity_role": "descriptive_secondary_not_in_holm_family",
                "reason_codes": sorted(row_reasons),
            }
        )
    interaction_reasons = (
        ("cross_family_interaction_reducer_artifact_missing",)
        if interaction_source is None
        else _canonical_reasons(
            (
                "cross_family_interaction_formal_status_unresolved",
                *interaction_source.reason_codes,
            )
        )
    )
    interaction_rows = [
        {
            "axis": axis,
            "formal_claim_status": "UNRESOLVED",
            "estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "independent_unit": (
                None if interaction_source is None else "paired_block"
            ),
            "reducer_protocol_sha256": CROSS_FAMILY_INTERACTION_REDUCER_PROTOCOL_SHA256,
            "source_sha256": None
            if interaction_source is None
            else interaction_source.sha256,
            "input_manifest_sha256": (
                None
                if interaction_source is None
                else interaction_source.input_manifest_sha256
            ),
            "reason_codes": sorted(
                {
                    *reasons,
                    *interaction_reasons,
                }
            ),
        }
        for axis in CROSS_FAMILY_INTERACTION_AXES
    ]
    return _specification(
        spec_id="tab:formal-claim-status",
        kind="formal_claim_status",
        status=OutputStatus.BLOCKED,
        source_sha256=None if source is None else source.sha256,
        reason_codes=reasons,
        payload={
            "reducer_status": "MISSING" if source is None else source.status,
            "gpu_evidence": None if source is None else source.gpu_evidence,
            "metric_directions": {
                "aggregate_latency_p99_ms": "lower",
                "mean_relative_gain": "higher",
                "adjusted_p_value": "lower",
            },
            "uncertainty": {
                "primary": "paired_BCa_95_percent",
                "secondary": "paired_BCa_95_percent_descriptive",
                "interactions": "registered_cross_family_reducer_required",
                "p99": "time_block_bootstrap_95_percent_when_registered",
            },
            "p99_rows": p99_rows,
            "primary_rows": primary_rows,
            "secondary_rows": secondary_rows,
            "interaction_reducer_status": (
                "MISSING" if interaction_source is None else interaction_source.status
            ),
            "interaction_rows": interaction_rows,
        },
    )


def build_production_output_artifact(
    *,
    e3b_stage: E3bLongContextStageArtifact | None,
    industrial_reduction: IndustrialReducerArtifact | None,
    family_power_reduction: ConfirmationFamilyPowerReductionArtifact | None,
    interaction_reduction: CrossFamilyInteractionReducerArtifact | None = None,
) -> ProductionOutputArtifact:
    """Build paper-output specs without accepting summaries or raw rows.

    ``None`` is legal only to materialize an explicit named ``BLOCKED`` result
    before a production reducer artifact exists.  Present inputs are exact
    existing artifact types and are revalidated onsite.
    """

    if e3b_stage is not None:
        if type(e3b_stage) is not E3bLongContextStageArtifact:
            raise TypeError("E3b output requires an exact stage artifact")
        _revalidate_e3b_source(e3b_stage)
        if e3b_stage.status != "UNRESOLVED":
            raise ValueError("unsupported E3b formal production status")
    if industrial_reduction is not None:
        if type(industrial_reduction) is not IndustrialReducerArtifact:
            raise TypeError("claim output requires an exact industrial reducer")
        industrial_reduction.__post_init__()
        if (
            industrial_reduction.status != "UNRESOLVED"
            or industrial_reduction.gpu_evidence != "UNMEASURED"
        ):
            raise ValueError("unsupported industrial formal production status")
        if any(
            type(values) is not tuple
            for values in (
                industrial_reduction.methods,
                industrial_reduction.primary_contrasts,
                industrial_reduction.secondary_contrasts,
                industrial_reduction.holm_family,
            )
        ):
            raise TypeError("industrial reduction containers must be exact")
        if any(
            type(value) is not MethodReduction for value in industrial_reduction.methods
        ):
            raise TypeError("industrial method reductions must be exact")
        if any(
            type(value.aggregate_latency_p99) is not P99ClaimGuard
            for value in industrial_reduction.methods
        ):
            raise TypeError("industrial p99 claim guards must be exact")
        for value in industrial_reduction.methods:
            guard = value.aggregate_latency_p99
            replayed_guard = guard_p99_claim(
                guard.anchor_id,
                completed_requests=guard.completed_requests,
                observed_p99_ms=guard.observed_p99_ms,
                minimum_completions=guard.minimum_completions,
                preregistered_anchor_locked=guard.status == "CLAIMABLE",
            )
            if (
                replayed_guard != guard
                or (
                    guard.status == "CLAIMABLE"
                    and guard.minimum_completions < P99_MINIMUM_COMPLETIONS
                )
                or 0 < guard.minimum_completions < P99_MINIMUM_COMPLETIONS
            ):
                raise ValueError("industrial p99 claim guard is not canonical")
        if any(
            type(value) is not PairedBcaContrast
            for value in industrial_reduction.primary_contrasts
        ):
            raise TypeError("industrial primary contrasts must be exact")
        if any(
            type(value) is not PairedBcaContrast
            for value in industrial_reduction.secondary_contrasts
        ):
            raise TypeError("industrial secondary contrasts must be exact")
        if any(
            type(value) is not MultiplicityDecision
            for value in industrial_reduction.holm_family
        ):
            raise TypeError("industrial multiplicity decisions must be exact")
        if any(
            value.confidence != REGISTERED_CONFIDENCE
            or value.independent_unit != "paired_block"
            for value in industrial_reduction.primary_contrasts
        ):
            raise ValueError("industrial primary contrast semantics are not canonical")
        if any(
            value.confidence != REGISTERED_CONFIDENCE
            or value.independent_unit != "paired_block"
            for value in industrial_reduction.secondary_contrasts
        ):
            raise ValueError(
                "industrial secondary contrast semantics are not canonical"
            )
        if any(value.procedure != "holm" for value in industrial_reduction.holm_family):
            raise ValueError("industrial multiplicity semantics are not canonical")
        if type(
            industrial_reduction.reasons
        ) is not tuple or industrial_reduction.reasons != _canonical_reasons(
            industrial_reduction.reasons
        ):
            raise ValueError("industrial reducer reasons are not canonical")
        methods = tuple(value.method for value in industrial_reduction.methods)
        contrasts = tuple(
            value.name for value in industrial_reduction.primary_contrasts
        )
        secondary_contrasts = tuple(
            value.name for value in industrial_reduction.secondary_contrasts
        )
        decisions = tuple(value.name for value in industrial_reduction.holm_family)
        if methods and methods != CONFIRMATION_METHOD_ROLES:
            raise ValueError("industrial method reductions are not canonical")
        if contrasts and contrasts != PRIMARY_CONTRASTS:
            raise ValueError("industrial primary contrasts are not canonical")
        if secondary_contrasts and secondary_contrasts != SECONDARY_CONTRASTS:
            raise ValueError("industrial secondary contrasts are not canonical")
        if decisions and decisions != PRIMARY_CONTRASTS:
            raise ValueError("industrial Holm family is not canonical")
    if family_power_reduction is not None:
        if type(family_power_reduction) is not ConfirmationFamilyPowerReductionArtifact:
            raise TypeError("power output requires an exact family-power reduction")
        _revalidate_power_source(family_power_reduction)
    if interaction_reduction is not None:
        if type(interaction_reduction) is not CrossFamilyInteractionReducerArtifact:
            raise TypeError("interaction output requires an exact interaction reducer")
        _revalidate_interaction_source(interaction_reduction)

    registry_sha256s = {
        value
        for value in (
            None if e3b_stage is None else e3b_stage.registry_sha256,
            (
                None
                if industrial_reduction is None
                else industrial_reduction.registry_sha256
            ),
            (
                None
                if family_power_reduction is None
                else family_power_reduction.family.registry_sha256
            ),
            (
                None
                if interaction_reduction is None
                else interaction_reduction.registry_sha256
            ),
        )
        if value is not None
    }
    if len(registry_sha256s) > 1:
        raise ValueError("production output sources differ in registry identity")

    if industrial_reduction is not None and family_power_reduction is not None:
        family = family_power_reduction.family
        if (
            industrial_reduction.registry_sha256 != family.registry_sha256
            or industrial_reduction.experiment != family.experiment
            or industrial_reduction.runtime_sha256 != family.runtime_sha256
            or industrial_reduction.split_sha256 != family.split_sha256
            or industrial_reduction.inventory_sha256
            != family_power_reduction.inventory_sha256
            or industrial_reduction.inventory_source_receipt_sha256
            != family_power_reduction.inventory_source_receipt_sha256
            or industrial_reduction.fixed_instance_gpu_count
            != family_power_reduction.fixed_instance_gpu_count
            or industrial_reduction.inventory_host_id
            != family_power_reduction.inventory_host_id
            or industrial_reduction.confirmation_family_sha256 != family.sha256
            or industrial_reduction.pilot_activation_sha256
            != family_power_reduction.plan.pilot_activation_sha256
            or industrial_reduction.confirmation_plan_sha256
            != family_power_reduction.sha256
            or industrial_reduction.pilot_evidence_sha256
            != family_power_reduction.plan.pilot_evidence_sha256
            or industrial_reduction.completed_pilot_cells_sha256
            != family_power_reduction.plan.completed_pilot_cells_sha256
            or industrial_reduction.hardware_envelope_sha256
            != family.hardware_envelope_sha256
            or industrial_reduction.power_plan
            != family_power_reduction.plan.power_sizing
        ):
            raise ValueError("industrial and family-power output sources differ")
    if (
        interaction_reduction is not None
        and industrial_reduction is not None
        and (
            interaction_reduction.runtime_sha256 != industrial_reduction.runtime_sha256
            or interaction_reduction.split_sha256 != industrial_reduction.split_sha256
        )
    ):
        raise ValueError("interaction and industrial output sources differ")
    if (
        interaction_reduction is not None
        and family_power_reduction is not None
        and (
            interaction_reduction.runtime_sha256
            != family_power_reduction.family.runtime_sha256
            or interaction_reduction.split_sha256
            != family_power_reduction.family.split_sha256
        )
    ):
        raise ValueError("interaction and family-power output sources differ")
    if (
        e3b_stage is not None
        and family_power_reduction is not None
        and family_power_reduction.family.experiment == "E3b"
        and e3b_stage.final_block_ids is not None
        and e3b_stage.final_block_ids != family_power_reduction.selected_final_prefix
    ):
        raise ValueError("E3b and family-power output sources differ")

    e3b_source = ProductionSourceBinding(
        kind="e3b_long_context_stage_reducer",
        artifact_sha256=None if e3b_stage is None else e3b_stage.sha256,
        status="MISSING" if e3b_stage is None else e3b_stage.status,
        reason_codes=(
            ("e3b_long_context_stage_artifact_missing",)
            if e3b_stage is None
            else e3b_stage.reasons
        ),
    )
    industrial_source = ProductionSourceBinding(
        kind="industrial_schema_v3_reducer",
        artifact_sha256=(
            None if industrial_reduction is None else industrial_reduction.sha256
        ),
        status=(
            "MISSING" if industrial_reduction is None else industrial_reduction.status
        ),
        reason_codes=(
            ("industrial_reducer_artifact_missing",)
            if industrial_reduction is None
            else _canonical_reasons(industrial_reduction.reasons)
        ),
    )
    power_source = ProductionSourceBinding(
        kind="confirmation_family_power_reduction",
        artifact_sha256=(
            None if family_power_reduction is None else family_power_reduction.sha256
        ),
        status=(
            "MISSING"
            if family_power_reduction is None
            else family_power_reduction.status
        ),
        reason_codes=(
            ("confirmation_family_power_reduction_artifact_missing",)
            if family_power_reduction is None
            else (family_power_reduction.plan.reason_code,)
        ),
    )

    blockers = {
        (
            "e3b_long_context_stage_artifact_missing"
            if e3b_stage is None
            else "e3b_formal_production_status_unresolved"
        ),
        (
            "industrial_reducer_artifact_missing"
            if industrial_reduction is None
            else "industrial_formal_production_status_unresolved"
        ),
    }
    if family_power_reduction is None:
        blockers.add("confirmation_family_power_reduction_artifact_missing")
    elif family_power_reduction.status == "UNDERPOWERED":
        blockers.add("confirmation_family_underpowered")

    return ProductionOutputArtifact(
        schema_version=3,
        protocol_sha256=PRODUCTION_OUTPUT_PROTOCOL_SHA256,
        status=OutputStatus.BLOCKED,
        blocker_codes=tuple(sorted(blockers)),
        sources=(e3b_source, industrial_source, power_source),
        figure=_e3b_figure(e3b_stage),
        tables=(
            _power_table(family_power_reduction),
            _claim_table(industrial_reduction, interaction_reduction),
        ),
    )


__all__ = [
    "CROSS_FAMILY_INTERACTION_AXES",
    "CROSS_FAMILY_INTERACTION_REDUCER_PROTOCOL_SHA256",
    "PRODUCTION_OUTPUT_PROTOCOL_SHA256",
    "CrossFamilyInteractionBinding",
    "CrossFamilyInteractionReducerArtifact",
    "MachineReadableSpecification",
    "OutputStatus",
    "ProductionOutputArtifact",
    "build_production_output_artifact",
    "production_output_artifact_from_json_bytes",
]
