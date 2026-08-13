"""Fail-closed planning boundary for paper figures and tables.

The repository does not own the paper template.  This module therefore plans
the eventual vector-native presentation of an exact
``ProductionOutputArtifact`` but cannot render TeX, PDF, or PNG artifacts.  A
``BLOCKED`` production output contributes no empirical values to the plan.
Only source identities, registered uncertainty semantics, panel membership,
and table roles cross this boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, NoReturn

from lightcone_spec.experiments.production_output import (
    MachineReadableSpecification,
    OutputStatus,
    ProductionOutputArtifact,
)
from lightcone_spec.experiments.registry import content_sha256

_PAPER_BLOCKERS = (
    "paper_main_tex_unregistered",
    "paper_visual_style_unregistered",
)
_FIGURE_GROUPS = (
    (
        "fig:e3b-accepted-length",
        "accepted_length",
        "Which registered methods change accepted length across context?",
        3,
    ),
    (
        "fig:e3b-committed-token-goodput",
        "committed_token_goodput",
        "Which registered methods change committed-token goodput across context?",
        2,
    ),
)

PRODUCTION_VISUAL_PLAN_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_production_visual_plan_protocol",
        "source": "exact_production_output_artifact",
        "raw_evidence": "forbidden",
        "blocked_formal_numeric": "omitted",
        "lifecycle": "planned_only_until_full_paper_audit",
        "paper_template": "source_owned_registration_required",
        "style_manifest": "source_owned_registration_required",
        "data_figure_render_mode": "vector_native_only",
        "main_panel_groups": {
            "accepted_length": 3,
            "committed_token_goodput": 2,
        },
        "empirical_curve": "forbidden_without_registered_uncertainty",
        "power_table": "preregistered_planning_only_not_formal_result",
        "render_outputs": "null_while_blocked",
        "reviewed_gate": (
            "requires_hash_matched_full_paper_compile_pdf_png_visual_audit"
        ),
        "serialization": "canonical_json_sort_keys_ascii_no_nan",
    }
)


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
        raise TypeError("production visual plan payload must be bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("production visual plan has duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(body, object_pairs_hook=reject_duplicates)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("production visual plan is not strict JSON") from error
    if type(value) is not dict:
        raise TypeError("production visual plan must be a JSON object")
    try:
        canonical = _canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError("production visual plan contains an invalid value") from error
    if canonical != body:
        raise ValueError("production visual plan is not canonical JSON")
    return value


def _canonical_codes(values: tuple[str, ...]) -> tuple[str, ...]:
    if type(values) is not tuple or any(
        type(value) is not str or not value or "\n" in value or "\r" in value
        for value in values
    ):
        raise TypeError("production visual blocker codes must be exact text")
    return tuple(sorted(set(values)))


def _revalidate_specification(specification: MachineReadableSpecification) -> None:
    if type(specification) is not MachineReadableSpecification:
        raise TypeError("production visual source specifications must be exact")
    specification.__post_init__()
    current = hashlib.sha256(specification.canonical_payload).hexdigest()
    if specification.sha256 != current:
        raise ValueError("production visual source specification changed after sealing")


def _revalidate_source(source: ProductionOutputArtifact) -> dict[str, object]:
    if type(source) is not ProductionOutputArtifact:
        raise TypeError("production visual planning requires an exact output artifact")
    source.__post_init__()
    _revalidate_specification(source.figure)
    for table in source.tables:
        _revalidate_specification(table)
    value = source.to_dict()
    if source.sha256 != content_sha256(value):
        raise ValueError("production visual source changed after sealing")
    if source.canonical_json_bytes() != _canonical_json_bytes(value) + b"\n":
        raise ValueError("production visual source serialization changed")
    return value


def _paper_plan() -> dict[str, object]:
    return {
        "target_visual_manifest_schema_version": 2,
        "main_tex": None,
        "body_font_pt": None,
        "textwidth_pt": None,
        "style_manifest": None,
        "style_sha256": None,
        "legend_location": None,
        "compiled_pdf": None,
        "compiled_pdf_sha256": None,
        "compile_log": None,
        "visual_gate_report": None,
        "review_status": "NOT_RUN",
    }


def _figure_plans(
    specification: MachineReadableSpecification,
) -> list[dict[str, object]]:
    source = specification.to_dict()
    uncertainty = source["uncertainty"]
    panels = source["panels"]
    if type(uncertainty) is not dict or type(panels) is not list:
        raise TypeError("E3b visual plan requires exact uncertainty and panels")

    plans: list[dict[str, object]] = []
    for plan_id, metric, reader_question, expected_count in _FIGURE_GROUPS:
        selected = [panel for panel in panels if panel["metric"] == metric]
        if len(selected) != expected_count:
            raise ValueError("E3b visual panel grouping differs from the protocol")
        reasons = _canonical_codes(
            (
                *_PAPER_BLOCKERS,
                *specification.reason_codes,
                "formal_figure_spec_blocked",
            )
        )
        plans.append(
            {
                "id": plan_id,
                "source_spec_id": specification.spec_id,
                "source_spec_sha256": specification.sha256,
                "source_status": specification.status.value,
                "status": OutputStatus.BLOCKED.value,
                "blocker_codes": list(reasons),
                "placement": "main",
                "render_mode": "vector-native",
                "routes": ["curves-scaling"],
                "claim": source["claim"],
                "reader_question": reader_question,
                "panel_ids": [panel["panel_id"] for panel in selected],
                "panels": expected_count,
                "uncertainty": {
                    "status": "REQUIRED_NOT_EMITTED",
                    "confidence": "registered_95_percent",
                    "method": uncertainty["method"],
                    "interval_fields": uncertainty["interval_fields"],
                },
                "empirical_curves": [],
                "final_width_pt": None,
                "inclusion_scale": None,
                "legend_location": None,
                "title_present": False,
                "renderer_source": None,
                "pdf": None,
                "text_collisions": None,
                "clipped_elements": None,
                "visual_review": {
                    "status": "NOT_RUN",
                    "png": None,
                    "pdf_sha256": None,
                },
            }
        )
    return plans


def _table_plan(
    specification: MachineReadableSpecification,
    *,
    placement: str,
    evidence_role: str,
) -> dict[str, object]:
    source = specification.to_dict()
    directions = source["metric_directions"]
    if type(directions) is not dict:
        raise TypeError("production visual metric directions must be exact")
    source_blockers = (
        specification.reason_codes
        if specification.status is OutputStatus.BLOCKED
        else ()
    )
    reasons = _canonical_codes(
        (
            *_PAPER_BLOCKERS,
            *source_blockers,
            "formal_table_render_blocked",
        )
    )
    return {
        "id": specification.spec_id,
        "source_spec_sha256": specification.sha256,
        "source_status": specification.status.value,
        "status": OutputStatus.BLOCKED.value,
        "blocker_codes": list(reasons),
        "placement": placement,
        "evidence_role": evidence_role,
        "formal_result_eligible": False,
        "metric_directions": directions,
        "rows": None,
        "columns": None,
        "natural_width_pt": None,
        "linewidth_pt": None,
        "conditional_resize": None,
        "average": {"status": "BLOCKED", "recomputed_from": None},
        "uncertainty_complete": False,
        "ranking_checked": False,
        "compile_status": "NOT_RUN",
        "tex": None,
        "clipped_cells": None,
    }


def _manifest(source: ProductionOutputArtifact) -> dict[str, object]:
    source_value = _revalidate_source(source)
    blockers = set(_PAPER_BLOCKERS)
    if source.status is OutputStatus.BLOCKED:
        blockers.update(source.blocker_codes)
        blockers.add("production_output_blocked")
    power_table, claim_table = source.tables
    return {
        "schema_version": 1,
        "kind": "lightcone_production_visual_plan",
        "protocol_sha256": PRODUCTION_VISUAL_PLAN_PROTOCOL_SHA256,
        "lifecycle": "planned",
        "status": OutputStatus.BLOCKED.value,
        "source": {
            "kind": source_value["kind"],
            "sha256": source.sha256,
            "status": source.status.value,
            "blocker_codes": list(source.blocker_codes),
        },
        "blocker_codes": sorted(blockers),
        "paper": _paper_plan(),
        "figures": _figure_plans(source.figure),
        "tables": [
            _table_plan(
                power_table,
                placement="appendix",
                evidence_role="preregistered_power_planning_only",
            ),
            _table_plan(
                claim_table,
                placement="main",
                evidence_role="formal_result_blocked",
            ),
        ],
    }


@dataclass(frozen=True)
class ProductionVisualPlanArtifact:
    """Canonical data-free plan that must replay from its exact source."""

    source: ProductionOutputArtifact = field(repr=False)
    canonical_payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        value = _strict_json_object(self.canonical_payload)
        if value != _manifest(self.source):
            raise ValueError("production visual plan differs from exact source replay")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return _strict_json_object(self.canonical_payload)

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict()) + b"\n"

    @property
    def schema_version(self) -> int:
        return 1

    @property
    def protocol_sha256(self) -> str:
        return PRODUCTION_VISUAL_PLAN_PROTOCOL_SHA256

    @property
    def lifecycle(self) -> str:
        return "planned"

    @property
    def status(self) -> OutputStatus:
        return OutputStatus.BLOCKED

    @property
    def source_sha256(self) -> str:
        _revalidate_source(self.source)
        return self.source.sha256

    @property
    def blocker_codes(self) -> tuple[str, ...]:
        value = self.to_dict()["blocker_codes"]
        if type(value) is not list or any(type(code) is not str for code in value):
            raise TypeError("production visual plan blockers must be exact")
        return tuple(value)

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


class ProductionVisualRenderBlocked(RuntimeError):
    """Named refusal to render a planned or otherwise blocked visual plan."""

    def __init__(self, blocker_codes: tuple[str, ...]) -> None:
        self.blocker_codes = blocker_codes
        super().__init__(
            "production visual render is BLOCKED: " + ",".join(blocker_codes)
        )


def build_production_visual_plan(
    source: ProductionOutputArtifact,
) -> ProductionVisualPlanArtifact:
    """Build a data-free planned manifest from one exact output artifact."""

    value = _manifest(source)
    return ProductionVisualPlanArtifact(
        source=source,
        canonical_payload=_canonical_json_bytes(value),
    )


def require_production_visual_renderable(
    plan: ProductionVisualPlanArtifact,
) -> NoReturn:
    """Fail before any renderer or filesystem output can be selected."""

    if type(plan) is not ProductionVisualPlanArtifact:
        raise TypeError("production visual render requires an exact plan artifact")
    plan.__post_init__()
    raise ProductionVisualRenderBlocked(plan.blocker_codes)


__all__ = [
    "PRODUCTION_VISUAL_PLAN_PROTOCOL_SHA256",
    "ProductionVisualPlanArtifact",
    "ProductionVisualRenderBlocked",
    "build_production_visual_plan",
    "require_production_visual_renderable",
]
