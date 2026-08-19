"""Registered E0 secondary-family authority and Benjamini--Hochberg reducer.

E0 breadth findings are secondary hypotheses.  Their family membership is
derived from the complete E0 registry rather than supplied by a caller: core
L0 comparisons form one family and isolated OnlineSPEC comparisons form a
second.  The raw input must cover both families and every registered
hypothesis exactly once.  Missing rows, post-hoc regrouping, foreign cells, or
non-finite p-values fail closed.

The legacy binder remains diagnostic-only.  The formal entry point instead
deep-rebuilds the signed E0 final result DAG, derives paired block p-values
from verifier-owned terminal/timestamp proofs, and records every compatibility
N/A as an explicit excluded preregistered hypothesis.  No caller-supplied
p-value or release digest allowlist is accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import t as student_t

from lightcone_spec.experiments.e0_authority_artifact import (
    E0FinalAnalysisProjection,
    rebuild_e0_final_analysis_projection_from_artifact,
)
from lightcone_spec.experiments.formal_protocol import verify_signed_payload
from lightcone_spec.experiments.registry import (
    E0_BACKENDS,
    E0_METHOD_ROLES,
    E0_MODELS,
    E0_TASKS,
    ExperimentCell,
    ExperimentRegistry,
    ScientificMethodRole,
    content_sha256,
    scientific_role_for_cell,
)
from lightcone_spec.experiments.statistics import (
    bca_mean_interval,
    benjamini_hochberg,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)

E0BreadthFamilyId = Literal[
    "e0_core_breadth",
    "e0_isolated_onlinespec_breadth",
]

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RAW_BYTES = 32 * 1024 * 1024

E0_BREADTH_FALSE_DISCOVERY_RATE = 0.05
E0_BREADTH_RAW_SOURCE_MISSING_REASON = "e0_breadth_raw_p_values_missing"
E0_BREADTH_FORMAL_SOURCE_UNAVAILABLE_REASON = "e0_final_proof_artifact_required"
E0_BREADTH_BOOTSTRAP_REPETITIONS = 10_000
E0_BREADTH_BOOTSTRAP_SEED = 0

_CORE_CONTRASTS = (
    ("lightcone_vs_tts", "lightcone", "tts"),
    ("lightcone_vs_static", "lightcone", "static"),
    ("l0_naive_vs_tts", "l0_naive", "tts"),
    ("lightcone_vs_l0_naive", "lightcone", "l0_naive"),
)
_ONLINESPEC_CONTRASTS = (
    ("onlinespec_ens_vs_static", "onlinespec_ens", "static"),
    ("onlinespec_ogd_vs_static", "onlinespec_ogd", "static"),
    ("onlinespec_opt_vs_static", "onlinespec_opt", "static"),
)

_E0_BREADTH_FAMILY_PROTOCOL = {
    "families": {
        "e0_core_breadth": [row[0] for row in _CORE_CONTRASTS],
        "e0_isolated_onlinespec_breadth": [row[0] for row in _ONLINESPEC_CONTRASTS],
    },
    "panels": {
        "models": list(E0_MODELS),
        "backends": list(E0_BACKENDS),
        "tasks": list(E0_TASKS),
    },
    "procedure": "benjamini-hochberg",
    "false_discovery_rate": E0_BREADTH_FALSE_DISCOVERY_RATE,
}

E0_BREADTH_LEGACY_DIAGNOSTIC_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e0_legacy_diagnostic_breadth_fdr",
        **_E0_BREADTH_FAMILY_PROTOCOL,
        "universe_source": "legacy_structural_compatibility_templates",
        "raw_source": "caller_file_diagnostic_only_never_formal",
        "primary_family": "forbidden",
    }
)

E0_BREADTH_FDR_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 5,
        "kind": "e0_registered_secondary_breadth_fdr",
        **_E0_BREADTH_FAMILY_PROTOCOL,
        "coverage": (
            "all_VALID_hypotheses_exactly_once_and_all_NA_hypotheses_"
            "explicitly_excluded"
        ),
        "universe_source": (
            "deep_rebuilt_signed_108_compatibility_decisions_plus_fixed_contrasts"
        ),
        "formal_source": (
            "deep_rebuilt_signed_E0_final_materialization_coverage_terminal_"
            "and_native_ITL_proofs"
        ),
        "independent_unit": "paired_final_block",
        "reporting_load": "common_slo_load",
        "bootstrap_repetitions": E0_BREADTH_BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": E0_BREADTH_BOOTSTRAP_SEED,
        "contrast_scale": (
            "paired_absolute_slo_goodput_tps_difference_zero_is_measured"
        ),
        "goodput": (
            "individually_slo_qualified_output_tokens_per_native_scored_window"
        ),
        "slo_policy": (
            "all_task_native_scored_requests_eligible;prompt_bucket_from_input_"
            "tokens;native_TTFT_and_within_request_p99_ITL"
        ),
        "formal_lightcone": "seal_bound_materialization_only",
        "primary_family": "forbidden",
    }
)

_FORMAL_ROLE_TO_METHOD = {
    "Target-only": "target_only",
    "Static": "static",
    "TTS": "tts",
    "L0-naive": "l0_naive",
    "LightCone": "lightcone",
    "OnlineSPEC-OGD": "onlinespec_ogd",
    "OnlineSPEC-OPT": "onlinespec_opt",
    "OnlineSPEC-ENS": "onlinespec_ens",
    "OnlineSPEC-Optimistic-OGD": "onlinespec_opt",
    "OnlineSPEC-Hedge": "onlinespec_ens",
}


class E0BreadthFdrAuthorityBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"E0 breadth FDR authority is BLOCKED: {reason}")
        self.reason = reason


def _require_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if not isinstance(value, str) or not value or "\n" in value:
        raise ValueError(f"{label} must be non-empty single-line text")
    return value


def _strict_mapping(label: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _strict_sequence(label: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def _strict_keys(label: str, value: Mapping[str, object], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


@dataclass(frozen=True)
class E0BreadthHypothesis:
    family_id: E0BreadthFamilyId
    hypothesis_id: str
    model: str
    backend: str
    task: str
    contrast: str
    numerator_method: str
    denominator_method: str
    numerator_cell_id: str
    numerator_cell_sha256: str
    denominator_cell_id: str
    denominator_cell_sha256: str

    def __post_init__(self) -> None:
        if self.family_id not in {
            "e0_core_breadth",
            "e0_isolated_onlinespec_breadth",
        }:
            raise ValueError("E0 breadth family is not registered")
        for label, value in (
            ("hypothesis", self.hypothesis_id),
            ("model", self.model),
            ("backend", self.backend),
            ("task", self.task),
            ("contrast", self.contrast),
            ("numerator method", self.numerator_method),
            ("denominator method", self.denominator_method),
        ):
            _require_text(label, value)
        for label, value in (
            ("numerator cell", self.numerator_cell_id),
            ("numerator declaration", self.numerator_cell_sha256),
            ("denominator cell", self.denominator_cell_id),
            ("denominator declaration", self.denominator_cell_sha256),
        ):
            _require_sha256(label, value)
        if self.numerator_cell_id == self.denominator_cell_id:
            raise ValueError("E0 breadth contrast cannot compare one cell to itself")
        expected_id = content_sha256(
            {
                "schema_version": 1,
                "family_id": self.family_id,
                "model": self.model,
                "backend": self.backend,
                "task": self.task,
                "contrast": self.contrast,
                "numerator_method": self.numerator_method,
                "denominator_method": self.denominator_method,
                "numerator_cell_id": self.numerator_cell_id,
                "numerator_cell_sha256": self.numerator_cell_sha256,
                "denominator_cell_id": self.denominator_cell_id,
                "denominator_cell_sha256": self.denominator_cell_sha256,
            }
        )
        if self.hypothesis_id != expected_id:
            raise ValueError("E0 breadth hypothesis identity is not canonical")

    def to_dict(self) -> dict[str, str]:
        return {
            "family_id": self.family_id,
            "hypothesis_id": self.hypothesis_id,
            "model": self.model,
            "backend": self.backend,
            "task": self.task,
            "contrast": self.contrast,
            "numerator_method": self.numerator_method,
            "denominator_method": self.denominator_method,
            "numerator_cell_id": self.numerator_cell_id,
            "numerator_cell_sha256": self.numerator_cell_sha256,
            "denominator_cell_id": self.denominator_cell_id,
            "denominator_cell_sha256": self.denominator_cell_sha256,
        }


@dataclass(frozen=True)
class E0FormalBreadthHypothesis:
    """E0 hypothesis derived from signed compatibility, not legacy templates."""

    family_id: E0BreadthFamilyId
    hypothesis_id: str
    compatibility_decision_id: str
    compatibility_decision_sha256: str
    model: str
    backend: str
    task: str
    contrast: str
    numerator_method: str
    denominator_method: str

    def __post_init__(self) -> None:
        if self.family_id not in {
            "e0_core_breadth",
            "e0_isolated_onlinespec_breadth",
        }:
            raise ValueError("formal E0 breadth family is not registered")
        for label, value in (
            ("compatibility decision", self.compatibility_decision_id),
            ("compatibility declaration", self.compatibility_decision_sha256),
        ):
            _require_sha256(f"formal E0 breadth {label}", value)
        for label, value in (
            ("model", self.model),
            ("backend", self.backend),
            ("task", self.task),
            ("contrast", self.contrast),
            ("numerator method", self.numerator_method),
            ("denominator method", self.denominator_method),
        ):
            _require_text(f"formal E0 breadth {label}", value)
        expected = content_sha256(
            {
                "schema_version": 1,
                "family_id": self.family_id,
                "compatibility_decision_id": self.compatibility_decision_id,
                "compatibility_decision_sha256": (self.compatibility_decision_sha256),
                "model": self.model,
                "backend": self.backend,
                "task": self.task,
                "contrast": self.contrast,
                "numerator_method": self.numerator_method,
                "denominator_method": self.denominator_method,
            }
        )
        if self.hypothesis_id != expected:
            raise ValueError("formal E0 breadth hypothesis identity is not canonical")

    def to_dict(self) -> dict[str, str]:
        return {
            "family_id": self.family_id,
            "hypothesis_id": self.hypothesis_id,
            "compatibility_decision_id": self.compatibility_decision_id,
            "compatibility_decision_sha256": self.compatibility_decision_sha256,
            "model": self.model,
            "backend": self.backend,
            "task": self.task,
            "contrast": self.contrast,
            "numerator_method": self.numerator_method,
            "denominator_method": self.denominator_method,
        }


def registered_formal_e0_breadth_hypotheses(
    projection: E0FinalAnalysisProjection,
) -> tuple[E0FormalBreadthHypothesis, ...]:
    """Derive all 756 hypotheses from the signed 108-row E0 decision set."""

    if type(projection) is not E0FinalAnalysisProjection:
        raise TypeError("formal E0 breadth universe requires an exact projection")
    projection.__post_init__()
    hypotheses: list[E0FormalBreadthHypothesis] = []
    families: tuple[tuple[E0BreadthFamilyId, tuple[tuple[str, str, str], ...]], ...] = (
        ("e0_core_breadth", _CORE_CONTRASTS),
        ("e0_isolated_onlinespec_breadth", _ONLINESPEC_CONTRASTS),
    )
    for decision in projection.compatibility_decisions:
        decision_sha256 = content_sha256(decision)
        for family_id, contrasts in families:
            for contrast, numerator_method, denominator_method in contrasts:
                payload = {
                    "schema_version": 1,
                    "family_id": family_id,
                    "compatibility_decision_id": decision.decision_id,
                    "compatibility_decision_sha256": decision_sha256,
                    "model": decision.model,
                    "backend": decision.backend,
                    "task": decision.task,
                    "contrast": contrast,
                    "numerator_method": numerator_method,
                    "denominator_method": denominator_method,
                }
                hypotheses.append(
                    E0FormalBreadthHypothesis(
                        family_id=family_id,
                        hypothesis_id=content_sha256(payload),
                        compatibility_decision_id=decision.decision_id,
                        compatibility_decision_sha256=decision_sha256,
                        model=decision.model,
                        backend=decision.backend,
                        task=decision.task,
                        contrast=contrast,
                        numerator_method=numerator_method,
                        denominator_method=denominator_method,
                    )
                )
    result = tuple(sorted(hypotheses, key=lambda row: row.hypothesis_id))
    if len(result) != 756 or len({row.hypothesis_id for row in result}) != 756:
        raise AssertionError("formal E0 breadth hypothesis count changed")
    return result


def _hypothesis(
    *,
    family_id: E0BreadthFamilyId,
    model: str,
    backend: str,
    task: str,
    contrast: str,
    numerator_method: str,
    denominator_method: str,
    numerator: ExperimentCell,
    denominator: ExperimentCell,
) -> E0BreadthHypothesis:
    payload = {
        "schema_version": 1,
        "family_id": family_id,
        "model": model,
        "backend": backend,
        "task": task,
        "contrast": contrast,
        "numerator_method": numerator_method,
        "denominator_method": denominator_method,
        "numerator_cell_id": numerator.cell_id,
        "numerator_cell_sha256": numerator.sha256,
        "denominator_cell_id": denominator.cell_id,
        "denominator_cell_sha256": denominator.sha256,
    }
    return E0BreadthHypothesis(
        family_id=family_id,
        hypothesis_id=content_sha256(payload),
        model=model,
        backend=backend,
        task=task,
        contrast=contrast,
        numerator_method=numerator_method,
        denominator_method=denominator_method,
        numerator_cell_id=numerator.cell_id,
        numerator_cell_sha256=numerator.sha256,
        denominator_cell_id=denominator.cell_id,
        denominator_cell_sha256=denominator.sha256,
    )


def registered_e0_breadth_hypotheses(
    registry: ExperimentRegistry,
) -> tuple[E0BreadthHypothesis, ...]:
    """Derive the complete secondary universe from exact E0 registry cells."""

    if type(registry) is not ExperimentRegistry:
        raise TypeError("E0 breadth universe requires an exact registry")
    cells = tuple(
        cell
        for cell in registry.cells_for("E0")
        if cell.identity.variant.startswith("compatibility_template:role=")
    )
    expected_cell_count = (
        len(E0_MODELS) * len(E0_BACKENDS) * len(E0_TASKS) * len(E0_METHOD_ROLES)
    )
    if len(cells) != expected_cell_count:
        raise ValueError("E0 structural breadth template universe is incomplete")
    by_key: dict[tuple[str, str, str, str], ExperimentCell] = {}
    for cell in cells:
        role = scientific_role_for_cell(registry, cell)
        # A sealed-sentinel registry row is only a planned slot. It may enter
        # the preregistered universe under the intended contrast name, but the
        # empty release raw-source allowlist below prevents it from becoming a
        # formal LightCone finding before seal-bound materialization.
        if role == ScientificMethodRole.LIGHTCONE_TEMPLATE.value:
            role = ScientificMethodRole.LIGHTCONE.value
        key = (
            cell.identity.model,
            cell.identity.backend,
            cell.identity.task,
            role,
        )
        if key in by_key:
            raise ValueError("E0 registry duplicates a breadth method cell")
        by_key[key] = cell
    hypotheses: list[E0BreadthHypothesis] = []
    families: tuple[tuple[E0BreadthFamilyId, tuple[tuple[str, str, str], ...]], ...] = (
        ("e0_core_breadth", _CORE_CONTRASTS),
        ("e0_isolated_onlinespec_breadth", _ONLINESPEC_CONTRASTS),
    )
    for family_id, contrasts in families:
        for model in E0_MODELS:
            for backend in E0_BACKENDS:
                for task in E0_TASKS:
                    for contrast, numerator_method, denominator_method in contrasts:
                        try:
                            numerator = by_key[(model, backend, task, numerator_method)]
                            denominator = by_key[
                                (model, backend, task, denominator_method)
                            ]
                        except KeyError as error:
                            raise ValueError(
                                "E0 registry lacks a registered breadth contrast cell"
                            ) from error
                        hypotheses.append(
                            _hypothesis(
                                family_id=family_id,
                                model=model,
                                backend=backend,
                                task=task,
                                contrast=contrast,
                                numerator_method=numerator_method,
                                denominator_method=denominator_method,
                                numerator=numerator,
                                denominator=denominator,
                            )
                        )
    result = tuple(sorted(hypotheses, key=lambda value: value.hypothesis_id))
    if len(result) != 756 or len({row.hypothesis_id for row in result}) != 756:
        raise AssertionError("registered E0 breadth hypothesis count changed")
    return result


@dataclass(frozen=True)
class E0BreadthRawPValue:
    hypothesis_id: str
    numerator_terminal_sha256: str
    denominator_terminal_sha256: str
    contrast_artifact_sha256: str
    raw_p_value: float

    def __post_init__(self) -> None:
        for label, value in (
            ("breadth hypothesis", self.hypothesis_id),
            ("breadth numerator terminal", self.numerator_terminal_sha256),
            ("breadth denominator terminal", self.denominator_terminal_sha256),
            ("breadth contrast artifact", self.contrast_artifact_sha256),
        ):
            _require_sha256(label, value)
        if self.numerator_terminal_sha256 == self.denominator_terminal_sha256:
            raise ValueError("breadth numerator and denominator sources must differ")
        if (
            isinstance(self.raw_p_value, bool)
            or not isinstance(self.raw_p_value, (int, float))
            or not math.isfinite(float(self.raw_p_value))
            or not 0.0 <= float(self.raw_p_value) <= 1.0
        ):
            raise ValueError("breadth raw p-value must be a finite probability")
        expected = content_sha256(
            {
                "schema_version": 1,
                "hypothesis_id": self.hypothesis_id,
                "numerator_terminal_sha256": self.numerator_terminal_sha256,
                "denominator_terminal_sha256": self.denominator_terminal_sha256,
                "raw_p_value": float(self.raw_p_value),
            }
        )
        if self.contrast_artifact_sha256 != expected:
            raise ValueError("breadth contrast source identity is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "numerator_terminal_sha256": self.numerator_terminal_sha256,
            "denominator_terminal_sha256": self.denominator_terminal_sha256,
            "contrast_artifact_sha256": self.contrast_artifact_sha256,
            "raw_p_value": float(self.raw_p_value),
        }


@dataclass(frozen=True)
class E0BreadthFdrAuthority:
    schema_version: int
    kind: str
    registry_sha256: str
    protocol_sha256: str
    raw_source_path: str
    raw_source_sha256: str
    false_discovery_rate: float
    hypotheses_sha256: str
    p_values: tuple[E0BreadthRawPValue, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "e0_breadth_fdr_authority":
            raise ValueError("E0 breadth FDR authority schema is unsupported")
        for label, value in (
            ("breadth registry", self.registry_sha256),
            ("breadth protocol", self.protocol_sha256),
            ("breadth raw source", self.raw_source_sha256),
            ("breadth hypotheses", self.hypotheses_sha256),
        ):
            _require_sha256(label, value)
        if self.protocol_sha256 != E0_BREADTH_LEGACY_DIAGNOSTIC_PROTOCOL_SHA256:
            raise ValueError("E0 breadth authority uses another protocol")
        if self.false_discovery_rate != E0_BREADTH_FALSE_DISCOVERY_RATE:
            raise ValueError("E0 breadth false-discovery rate is preregistered")
        path = Path(self.raw_source_path)
        if not path.is_absolute() or path != path.resolve():
            raise ValueError("E0 breadth raw source path must be absolute and resolved")
        ids = tuple(value.hypothesis_id for value in self.p_values)
        if not self.p_values or ids != tuple(sorted(set(ids))):
            raise ValueError("E0 breadth raw p-values must be sorted and unique")
        for value in self.p_values:
            value.__post_init__()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "registry_sha256": self.registry_sha256,
            "protocol_sha256": self.protocol_sha256,
            "raw_source_path": self.raw_source_path,
            "raw_source_sha256": self.raw_source_sha256,
            "false_discovery_rate": self.false_discovery_rate,
            "hypotheses_sha256": self.hypotheses_sha256,
            "p_values": [value.to_dict() for value in self.p_values],
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class E0BreadthFdrDecision:
    family_id: E0BreadthFamilyId
    hypothesis_id: str
    raw_p_value: float
    q_value: float
    rejected: bool
    procedure: Literal["benjamini-hochberg"]
    false_discovery_rate: float
    contrast_artifact_sha256: str

    def __post_init__(self) -> None:
        if self.family_id not in {
            "e0_core_breadth",
            "e0_isolated_onlinespec_breadth",
        }:
            raise ValueError("E0 breadth decision family is unsupported")
        _require_sha256("breadth decision hypothesis", self.hypothesis_id)
        _require_sha256("breadth decision source", self.contrast_artifact_sha256)
        for label, value in (
            ("raw p-value", self.raw_p_value),
            ("q-value", self.q_value),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"breadth {label} must be a finite probability")
        if type(self.rejected) is not bool:
            raise TypeError("breadth FDR decision must be boolean")
        if self.procedure != "benjamini-hochberg":
            raise ValueError("E0 breadth decision uses another procedure")
        if self.false_discovery_rate != E0_BREADTH_FALSE_DISCOVERY_RATE:
            raise ValueError("E0 breadth decision uses another FDR")
        if self.rejected != (self.q_value <= self.false_discovery_rate):
            raise ValueError("E0 breadth rejection disagrees with its q-value")

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "hypothesis_id": self.hypothesis_id,
            "raw_p_value": self.raw_p_value,
            "q_value": self.q_value,
            "rejected": self.rejected,
            "procedure": self.procedure,
            "false_discovery_rate": self.false_discovery_rate,
            "contrast_artifact_sha256": self.contrast_artifact_sha256,
        }


@dataclass(frozen=True)
class E0BreadthFdrReduction:
    schema_version: int
    kind: str
    registry_sha256: str
    protocol_sha256: str
    authority_sha256: str
    raw_source_sha256: str
    hypotheses_sha256: str
    families: tuple[E0BreadthFamilyId, ...]
    decisions: tuple[E0BreadthFdrDecision, ...]
    primary_family_eligible: bool
    formal_execution_authorized: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "e0_breadth_fdr_reduction":
            raise ValueError("E0 breadth FDR reduction schema is unsupported")
        for label, value in (
            ("breadth reduction registry", self.registry_sha256),
            ("breadth reduction protocol", self.protocol_sha256),
            ("breadth reduction authority", self.authority_sha256),
            ("breadth reduction raw source", self.raw_source_sha256),
            ("breadth reduction hypotheses", self.hypotheses_sha256),
        ):
            _require_sha256(label, value)
        if self.protocol_sha256 != E0_BREADTH_LEGACY_DIAGNOSTIC_PROTOCOL_SHA256:
            raise ValueError("E0 breadth reduction uses another protocol")
        if self.families != (
            "e0_core_breadth",
            "e0_isolated_onlinespec_breadth",
        ):
            raise ValueError("E0 breadth reduction changed its registered families")
        ids = tuple(value.hypothesis_id for value in self.decisions)
        if len(ids) != 756 or ids != tuple(sorted(set(ids))):
            raise ValueError("E0 breadth decision coverage is incomplete")
        for value in self.decisions:
            value.__post_init__()
        if self.primary_family_eligible is not False:
            raise ValueError("E0 breadth findings cannot enter the primary family")
        if self.formal_execution_authorized is not False:
            raise ValueError("an FDR reduction cannot authorize formal execution")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "registry_sha256": self.registry_sha256,
            "protocol_sha256": self.protocol_sha256,
            "authority_sha256": self.authority_sha256,
            "raw_source_sha256": self.raw_source_sha256,
            "hypotheses_sha256": self.hypotheses_sha256,
            "families": list(self.families),
            "decisions": [value.to_dict() for value in self.decisions],
            "primary_family_eligible": self.primary_family_eligible,
            "formal_execution_authorized": self.formal_execution_authorized,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class E0PairedSloGoodputContrast:
    """Paired absolute SLO-goodput effect; measured zero remains observable."""

    name: str
    block_ids: tuple[str, ...]
    mean_difference_tps: float
    ci_lower_difference_tps: float
    ci_upper_difference_tps: float
    raw_p_value: float
    confidence: Literal[0.95]
    independent_unit: Literal["paired_block"] = "paired_block"
    metric: Literal["slo_goodput_tps"] = "slo_goodput_tps"

    def __post_init__(self) -> None:
        _require_sha256("formal E0 contrast hypothesis", self.name)
        if (
            type(self.block_ids) is not tuple
            or len(self.block_ids) < 12
            or self.block_ids != tuple(sorted(set(self.block_ids)))
        ):
            raise ValueError("formal E0 contrast block coverage is not exact")
        for label, value in (
            ("mean", self.mean_difference_tps),
            ("lower", self.ci_lower_difference_tps),
            ("upper", self.ci_upper_difference_tps),
            ("p-value", self.raw_p_value),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"formal E0 contrast {label} is not finite")
        if (
            not 0.0 <= self.raw_p_value <= 1.0
            or self.ci_lower_difference_tps > self.ci_upper_difference_tps
            or self.confidence != 0.95
            or self.independent_unit != "paired_block"
            or self.metric != "slo_goodput_tps"
        ):
            raise ValueError("formal E0 contrast protocol differs")


def _paired_slo_goodput_contrast(
    name: str,
    paired_goodput: Mapping[str, tuple[float, float]],
) -> E0PairedSloGoodputContrast:
    block_ids = tuple(sorted(paired_goodput))
    if len(block_ids) < 12:
        raise ValueError("formal E0 contrast requires the powered final prefix")
    differences = []
    for block_id in block_ids:
        pair = paired_goodput[block_id]
        if len(pair) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in pair
        ):
            raise ValueError("formal E0 SLO-goodput pair is invalid")
        differences.append(float(pair[0]) - float(pair[1]))
    cluster_values = {
        block_id: np.asarray([difference], dtype=np.float64)
        for block_id, difference in zip(block_ids, differences, strict=True)
    }
    mean, lower, upper = bca_mean_interval(
        cluster_values,
        confidence=0.95,
        repetitions=E0_BREADTH_BOOTSTRAP_REPETITIONS,
        seed=E0_BREADTH_BOOTSTRAP_SEED,
    )
    values = np.asarray(differences, dtype=np.float64)
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation <= np.finfo(np.float64).tiny:
        raw_p_value = 1.0 if abs(mean) <= np.finfo(np.float64).eps else 0.0
    else:
        statistic = mean / (standard_deviation / math.sqrt(values.size))
        raw_p_value = float(2.0 * student_t.sf(abs(statistic), values.size - 1))
    return E0PairedSloGoodputContrast(
        name=name,
        block_ids=block_ids,
        mean_difference_tps=float(mean),
        ci_lower_difference_tps=float(lower),
        ci_upper_difference_tps=float(upper),
        raw_p_value=raw_p_value,
        confidence=0.95,
    )


@dataclass(frozen=True)
class E0FormalBreadthHypothesisResult:
    """One preregistered E0 hypothesis after proof-derived eligibility."""

    hypothesis: E0FormalBreadthHypothesis
    compatibility_decision_id: str
    status: Literal["TESTED", "EXCLUDED_NA"]
    exclusion_reason: str | None
    numerator_terminal_sha256s: tuple[str, ...]
    denominator_terminal_sha256s: tuple[str, ...]
    contrast: E0PairedSloGoodputContrast | None
    contrast_artifact_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.hypothesis) is not E0FormalBreadthHypothesis:
            raise TypeError("formal E0 breadth hypothesis must be exact")
        self.hypothesis.__post_init__()
        _require_sha256(
            "formal E0 breadth compatibility decision",
            self.compatibility_decision_id,
        )
        if self.status == "EXCLUDED_NA":
            if (
                not isinstance(self.exclusion_reason, str)
                or not self.exclusion_reason
                or self.numerator_terminal_sha256s
                or self.denominator_terminal_sha256s
                or self.contrast is not None
                or self.contrast_artifact_sha256 is not None
            ):
                raise ValueError("excluded E0 breadth hypothesis is not explicit")
            return
        if self.status != "TESTED":
            raise ValueError("formal E0 breadth hypothesis status is unsupported")
        if self.exclusion_reason is not None:
            raise ValueError("tested E0 breadth hypothesis cannot be excluded")
        if (
            type(self.numerator_terminal_sha256s) is not tuple
            or type(self.denominator_terminal_sha256s) is not tuple
            or len(self.numerator_terminal_sha256s) < 12
            or len(self.numerator_terminal_sha256s)
            != len(self.denominator_terminal_sha256s)
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    *self.numerator_terminal_sha256s,
                    *self.denominator_terminal_sha256s,
                )
            )
            or any(
                numerator == denominator
                for numerator, denominator in zip(
                    self.numerator_terminal_sha256s,
                    self.denominator_terminal_sha256s,
                    strict=True,
                )
            )
            or type(self.contrast) is not E0PairedSloGoodputContrast
        ):
            raise ValueError("tested E0 breadth proof coverage is not exact")
        assert self.contrast is not None
        if (
            self.contrast.name != self.hypothesis.hypothesis_id
            or self.contrast.block_ids
            != tuple(
                sorted(
                    f"block:{index}"
                    for index in range(
                        4,
                        4 + len(self.numerator_terminal_sha256s),
                    )
                )
            )
            or self.contrast.independent_unit != "paired_block"
            or self.contrast.confidence != 0.95
        ):
            raise ValueError("formal E0 breadth contrast changed its protocol")
        expected = content_sha256(
            {
                "schema_version": 1,
                "hypothesis_id": self.hypothesis.hypothesis_id,
                "compatibility_decision_id": self.compatibility_decision_id,
                "numerator_terminal_sha256s": self.numerator_terminal_sha256s,
                "denominator_terminal_sha256s": self.denominator_terminal_sha256s,
                "contrast": asdict(self.contrast),
            }
        )
        if self.contrast_artifact_sha256 != expected:
            raise ValueError("formal E0 breadth contrast identity differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E0FormalBreadthFdrReceipt:
    """Formal BH result derived only from a complete E0 proof projection."""

    schema_version: Literal[1]
    kind: Literal["e0_formal_breadth_fdr_receipt"]
    registry_sha256: str
    protocol_sha256: str
    final_completion_receipt_sha256: str
    final_analysis_projection_sha256: str
    hypotheses_sha256: str
    hypotheses: tuple[E0FormalBreadthHypothesisResult, ...]
    decisions: tuple[E0BreadthFdrDecision, ...]
    primary_family_eligible: Literal[False]
    formal_result_authorized: Literal[True]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "e0_formal_breadth_fdr_receipt":
            raise ValueError("formal E0 breadth FDR receipt schema is unsupported")
        for label, digest in (
            ("registry", self.registry_sha256),
            ("protocol", self.protocol_sha256),
            ("final completion", self.final_completion_receipt_sha256),
            ("final projection", self.final_analysis_projection_sha256),
            ("hypothesis universe", self.hypotheses_sha256),
        ):
            _require_sha256(f"formal E0 breadth {label}", digest)
        if self.protocol_sha256 != E0_BREADTH_FDR_PROTOCOL_SHA256:
            raise ValueError("formal E0 breadth FDR protocol differs")
        ids = tuple(row.hypothesis.hypothesis_id for row in self.hypotheses)
        if (
            type(self.hypotheses) is not tuple
            or len(self.hypotheses) != 756
            or ids != tuple(sorted(set(ids)))
            or any(
                type(row) is not E0FormalBreadthHypothesisResult
                for row in self.hypotheses
            )
        ):
            raise ValueError("formal E0 breadth hypothesis coverage is incomplete")
        if self.hypotheses_sha256 != content_sha256(
            [row.hypothesis.to_dict() for row in self.hypotheses]
        ):
            raise ValueError("formal E0 breadth hypothesis universe digest differs")
        tested = {
            row.hypothesis.hypothesis_id: row
            for row in self.hypotheses
            if row.status == "TESTED"
        }
        decision_ids = tuple(row.hypothesis_id for row in self.decisions)
        if (
            decision_ids != tuple(sorted(tested))
            or any(type(row) is not E0BreadthFdrDecision for row in self.decisions)
            or any(
                row.contrast_artifact_sha256
                != tested[row.hypothesis_id].contrast_artifact_sha256
                for row in self.decisions
            )
        ):
            raise ValueError("formal E0 breadth FDR decision coverage differs")
        if self.primary_family_eligible is not False:
            raise ValueError("E0 breadth findings cannot enter the primary family")
        if self.formal_result_authorized is not True:
            raise ValueError(
                "proof-derived E0 breadth receipt must authorize its result"
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE0FormalBreadthFdrReceipt:
    payload: E0FormalBreadthFdrReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        registry: ExperimentRegistry,
        final_result_rebuild_artifact_path: str | Path,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E0FormalBreadthFdrReceipt:
        if type(self.payload) is not E0FormalBreadthFdrReceipt:
            raise TypeError("signed formal E0 breadth payload must be exact")
        expected = reduce_formal_e0_breadth_fdr_from_artifact(
            registry,
            final_result_rebuild_artifact_path,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError(
                "signed formal E0 breadth result differs from proof reducer"
            )
        verify_signed_payload(
            self.payload,
            payload_sha256=self.payload_sha256,
            challenge=self.challenge,
            attestation=self.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
        )
        return self.payload

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "payload": asdict(self.payload),
                "payload_sha256": self.payload_sha256,
                "challenge": asdict(self.challenge),
                "attestation": asdict(self.attestation),
            }
        )


def formal_e0_breadth_fdr_receipt_to_dict(
    value: E0FormalBreadthFdrReceipt,
) -> dict[str, object]:
    if type(value) is not E0FormalBreadthFdrReceipt:
        raise TypeError("formal E0 breadth codec requires an exact receipt")
    value.__post_init__()
    return {**asdict(value), "receipt_sha256": value.sha256}


def formal_e0_breadth_fdr_receipt_from_dict(
    value: object,
) -> E0FormalBreadthFdrReceipt:
    row = dict(_strict_mapping("formal E0 breadth receipt", value))
    _strict_keys(
        "formal E0 breadth receipt",
        row,
        {*E0FormalBreadthFdrReceipt.__dataclass_fields__, "receipt_sha256"},
    )
    declared = _require_sha256("formal E0 breadth receipt", row.pop("receipt_sha256"))
    hypotheses = []
    for value_row in _strict_sequence(
        "formal E0 breadth hypothesis results", row["hypotheses"]
    ):
        result_row = dict(
            _strict_mapping("formal E0 breadth hypothesis result", value_row)
        )
        _strict_keys(
            "formal E0 breadth hypothesis result",
            result_row,
            set(E0FormalBreadthHypothesisResult.__dataclass_fields__),
        )
        hypothesis_row = dict(
            _strict_mapping("formal E0 breadth hypothesis", result_row["hypothesis"])
        )
        _strict_keys(
            "formal E0 breadth hypothesis",
            hypothesis_row,
            set(E0FormalBreadthHypothesis.__dataclass_fields__),
        )
        result_row["hypothesis"] = E0FormalBreadthHypothesis(**hypothesis_row)
        for name in (
            "numerator_terminal_sha256s",
            "denominator_terminal_sha256s",
        ):
            result_row[name] = tuple(
                _strict_sequence(f"formal E0 breadth {name}", result_row[name])
            )
        if result_row["contrast"] is not None:
            contrast_row = dict(
                _strict_mapping("formal E0 breadth contrast", result_row["contrast"])
            )
            _strict_keys(
                "formal E0 breadth contrast",
                contrast_row,
                set(E0PairedSloGoodputContrast.__dataclass_fields__),
            )
            contrast_row["block_ids"] = tuple(
                _strict_sequence(
                    "formal E0 breadth contrast blocks",
                    contrast_row["block_ids"],
                )
            )
            result_row["contrast"] = E0PairedSloGoodputContrast(**contrast_row)
        hypotheses.append(E0FormalBreadthHypothesisResult(**result_row))
    decisions = []
    for value_row in _strict_sequence("formal E0 breadth decisions", row["decisions"]):
        decision_row = dict(_strict_mapping("formal E0 breadth decision", value_row))
        _strict_keys(
            "formal E0 breadth decision",
            decision_row,
            set(E0BreadthFdrDecision.__dataclass_fields__),
        )
        decisions.append(E0BreadthFdrDecision(**decision_row))
    row["hypotheses"] = tuple(hypotheses)
    row["decisions"] = tuple(decisions)
    receipt = E0FormalBreadthFdrReceipt(**row)  # type: ignore[arg-type]
    if receipt.sha256 != declared:
        raise ValueError("formal E0 breadth receipt digest differs from content")
    return receipt


def signed_formal_e0_breadth_fdr_to_dict(
    value: SignedE0FormalBreadthFdrReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE0FormalBreadthFdrReceipt:
        raise TypeError("signed formal E0 breadth codec requires an exact wrapper")
    return {
        "payload": formal_e0_breadth_fdr_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_formal_e0_breadth_fdr_from_dict(
    value: object,
) -> SignedE0FormalBreadthFdrReceipt:
    row = dict(_strict_mapping("signed formal E0 breadth receipt", value))
    _strict_keys(
        "signed formal E0 breadth receipt",
        row,
        {
            "payload",
            "payload_sha256",
            "challenge",
            "attestation",
            "signed_receipt_sha256",
        },
    )
    declared = _require_sha256(
        "signed formal E0 breadth receipt",
        row.pop("signed_receipt_sha256"),
    )
    challenge_row = dict(
        _strict_mapping("formal E0 breadth challenge", row["challenge"])
    )
    attestation_row = dict(
        _strict_mapping("formal E0 breadth attestation", row["attestation"])
    )
    _strict_keys(
        "formal E0 breadth challenge",
        challenge_row,
        set(AttestationChallenge.__dataclass_fields__),
    )
    _strict_keys(
        "formal E0 breadth attestation",
        attestation_row,
        set(SignedAttestation.__dataclass_fields__),
    )
    signed = SignedE0FormalBreadthFdrReceipt(
        payload=formal_e0_breadth_fdr_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],  # type: ignore[arg-type]
        challenge=AttestationChallenge(**challenge_row),
        attestation=SignedAttestation(**attestation_row),
    )
    if signed.sha256 != declared:
        raise ValueError("signed formal E0 breadth digest differs from content")
    return signed


def _read_stable_raw(path_value: str | Path) -> tuple[Path, bytes]:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError("E0 breadth raw source path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise E0BreadthFdrAuthorityBlocked(
            E0_BREADTH_RAW_SOURCE_MISSING_REASON
        ) from error
    if resolved != path:
        raise ValueError("E0 breadth raw source path must be resolved and non-symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("E0 breadth raw source cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_RAW_BYTES
        ):
            raise ValueError("E0 breadth raw source must be a bounded regular file")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("E0 breadth raw source changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("E0 breadth raw source grew while being read")
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
            not stat.S_ISREG(current.st_mode)
            or identity(before) != identity(after)
            or identity(after) != identity(current)
        ):
            raise ValueError("E0 breadth raw source changed during coordinated read")
    finally:
        os.close(descriptor)
    return path, b"".join(chunks)


def _load_strict_json(raw: bytes) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"E0 breadth raw source duplicates key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"E0 breadth raw source contains non-finite {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("E0 breadth raw source is not strict UTF-8 JSON") from error
    return _strict_mapping("E0 breadth raw source", value)


def _parse_raw_p_value(value: object) -> E0BreadthRawPValue:
    row = _strict_mapping("E0 breadth p-value row", value)
    _strict_keys(
        "E0 breadth p-value row",
        row,
        {
            "hypothesis_id",
            "numerator_terminal_sha256",
            "denominator_terminal_sha256",
            "contrast_artifact_sha256",
            "raw_p_value",
        },
    )
    raw_p_value = row["raw_p_value"]
    if isinstance(raw_p_value, bool) or not isinstance(raw_p_value, (int, float)):
        raise TypeError("E0 breadth raw p-value must be numeric")
    return E0BreadthRawPValue(
        hypothesis_id=row["hypothesis_id"],  # type: ignore[arg-type]
        numerator_terminal_sha256=row["numerator_terminal_sha256"],  # type: ignore[arg-type]
        denominator_terminal_sha256=row["denominator_terminal_sha256"],  # type: ignore[arg-type]
        contrast_artifact_sha256=row["contrast_artifact_sha256"],  # type: ignore[arg-type]
        raw_p_value=float(raw_p_value),
    )


def bind_e0_breadth_fdr_authority(
    registry: ExperimentRegistry,
    raw_source_path: str | Path,
) -> E0BreadthFdrAuthority:
    """Bind an exact all-family raw p-value artifact to the E0 registry."""

    hypotheses = registered_e0_breadth_hypotheses(registry)
    hypotheses_sha256 = content_sha256([value.to_dict() for value in hypotheses])
    path, raw = _read_stable_raw(raw_source_path)
    source = _load_strict_json(raw)
    _strict_keys(
        "E0 breadth raw source",
        source,
        {
            "schema_version",
            "kind",
            "registry_sha256",
            "protocol_sha256",
            "false_discovery_rate",
            "hypotheses_sha256",
            "families",
        },
    )
    if (
        source["schema_version"] != 1
        or source["kind"] != "e0_breadth_raw_p_values"
        or source["registry_sha256"] != registry.sha256
        or source["protocol_sha256"] != E0_BREADTH_LEGACY_DIAGNOSTIC_PROTOCOL_SHA256
        or source["false_discovery_rate"] != E0_BREADTH_FALSE_DISCOVERY_RATE
        or source["hypotheses_sha256"] != hypotheses_sha256
    ):
        raise ValueError("E0 breadth raw source differs from the registered protocol")
    family_values = _strict_sequence("E0 breadth families", source["families"])
    expected_family_ids = (
        "e0_core_breadth",
        "e0_isolated_onlinespec_breadth",
    )
    if len(family_values) != len(expected_family_ids):
        raise ValueError("E0 breadth raw source changes the registered families")
    rows: list[E0BreadthRawPValue] = []
    expected_by_family = {
        family_id: tuple(
            value.hypothesis_id for value in hypotheses if value.family_id == family_id
        )
        for family_id in expected_family_ids
    }
    for index, family_value in enumerate(family_values):
        family = _strict_mapping("E0 breadth family", family_value)
        _strict_keys("E0 breadth family", family, {"family_id", "p_values"})
        family_id = family["family_id"]
        if family_id != expected_family_ids[index]:
            raise ValueError("E0 breadth families were reordered or regrouped")
        p_values = tuple(
            _parse_raw_p_value(value)
            for value in _strict_sequence(
                "E0 breadth family p-values", family["p_values"]
            )
        )
        ids = tuple(value.hypothesis_id for value in p_values)
        if ids != expected_by_family[family_id]:
            raise ValueError("E0 breadth family hypothesis coverage is not exact")
        rows.extend(p_values)
    ordered = tuple(sorted(rows, key=lambda value: value.hypothesis_id))
    if tuple(value.hypothesis_id for value in ordered) != tuple(
        value.hypothesis_id for value in hypotheses
    ):
        raise ValueError("E0 breadth raw source does not cover the registered universe")
    return E0BreadthFdrAuthority(
        schema_version=1,
        kind="e0_breadth_fdr_authority",
        registry_sha256=registry.sha256,
        protocol_sha256=E0_BREADTH_LEGACY_DIAGNOSTIC_PROTOCOL_SHA256,
        raw_source_path=str(path),
        raw_source_sha256=hashlib.sha256(raw).hexdigest(),
        false_discovery_rate=E0_BREADTH_FALSE_DISCOVERY_RATE,
        hypotheses_sha256=hypotheses_sha256,
        p_values=ordered,
    )


def revalidate_e0_breadth_fdr_authority(
    registry: ExperimentRegistry,
    authority: E0BreadthFdrAuthority,
) -> E0BreadthFdrAuthority:
    if type(authority) is not E0BreadthFdrAuthority:
        raise TypeError("E0 breadth revalidation requires an exact authority")
    authority.__post_init__()
    rebound = bind_e0_breadth_fdr_authority(registry, authority.raw_source_path)
    if rebound != authority or rebound.sha256 != authority.sha256:
        raise ValueError("E0 breadth FDR authority changed during revalidation")
    return rebound


def _formal_breadth_result(
    *,
    hypothesis: E0FormalBreadthHypothesis,
    projection: E0FinalAnalysisProjection,
    decision_id: str,
    decision_reason: str | None,
) -> E0FormalBreadthHypothesisResult:
    if decision_reason is not None:
        return E0FormalBreadthHypothesisResult(
            hypothesis=hypothesis,
            compatibility_decision_id=decision_id,
            status="EXCLUDED_NA",
            exclusion_reason=decision_reason,
            numerator_terminal_sha256s=(),
            denominator_terminal_sha256s=(),
            contrast=None,
            contrast_artifact_sha256=None,
        )
    by_key = {
        (row.method_role, row.block, row.load): row
        for row in projection.cells
        if row.compatibility_decision_id == decision_id
    }
    numerator_role = next(
        role
        for role, method in _FORMAL_ROLE_TO_METHOD.items()
        if method == hypothesis.numerator_method
    )
    denominator_role = next(
        role
        for role, method in _FORMAL_ROLE_TO_METHOD.items()
        if method == hypothesis.denominator_method
    )
    blocks = projection.completion_receipt.selected_final_prefix
    numerator_rows = tuple(
        by_key[(numerator_role, block, "common_slo_load")] for block in blocks
    )
    denominator_rows = tuple(
        by_key[(denominator_role, block, "common_slo_load")] for block in blocks
    )
    paired_goodput = {
        f"block:{block}": (
            numerator.slo_goodput_tps,
            denominator.slo_goodput_tps,
        )
        for block, numerator, denominator in zip(
            blocks,
            numerator_rows,
            denominator_rows,
            strict=True,
        )
    }
    contrast = _paired_slo_goodput_contrast(
        hypothesis.hypothesis_id,
        paired_goodput,
    )
    numerator_terminals = tuple(row.terminal_receipt_sha256 for row in numerator_rows)
    denominator_terminals = tuple(
        row.terminal_receipt_sha256 for row in denominator_rows
    )
    contrast_artifact_sha256 = content_sha256(
        {
            "schema_version": 1,
            "hypothesis_id": hypothesis.hypothesis_id,
            "compatibility_decision_id": decision_id,
            "numerator_terminal_sha256s": numerator_terminals,
            "denominator_terminal_sha256s": denominator_terminals,
            "contrast": asdict(contrast),
        }
    )
    return E0FormalBreadthHypothesisResult(
        hypothesis=hypothesis,
        compatibility_decision_id=decision_id,
        status="TESTED",
        exclusion_reason=None,
        numerator_terminal_sha256s=numerator_terminals,
        denominator_terminal_sha256s=denominator_terminals,
        contrast=contrast,
        contrast_artifact_sha256=contrast_artifact_sha256,
    )


def reduce_formal_e0_breadth_fdr_from_projection(
    registry: ExperimentRegistry,
    projection: E0FinalAnalysisProjection,
) -> E0FormalBreadthFdrReceipt:
    """Reduce one already deep-validated E0 final projection."""

    if type(projection) is not E0FinalAnalysisProjection:
        raise TypeError("formal E0 breadth reducer requires an exact projection")
    projection.__post_init__()
    if projection.completion_receipt.registry_sha256 != registry.sha256:
        raise ValueError("formal E0 breadth projection uses a foreign registry")
    hypotheses = registered_formal_e0_breadth_hypotheses(projection)
    compatibility = {
        (row.model, row.backend, row.task): row
        for row in projection.compatibility_decisions
    }
    results = []
    for hypothesis in hypotheses:
        decision = compatibility[
            (hypothesis.model, hypothesis.backend, hypothesis.task)
        ]
        if (
            decision.decision_id != hypothesis.compatibility_decision_id
            or content_sha256(decision) != hypothesis.compatibility_decision_sha256
        ):
            raise ValueError("formal E0 breadth compatibility identity changed")
        results.append(
            _formal_breadth_result(
                hypothesis=hypothesis,
                projection=projection,
                decision_id=decision.decision_id,
                decision_reason=(
                    None if decision.disposition == "VALID" else decision.reason_code
                ),
            )
        )
    ordered_results = tuple(
        sorted(results, key=lambda row: row.hypothesis.hypothesis_id)
    )
    fdr_decisions = []
    for family_id in (
        "e0_core_breadth",
        "e0_isolated_onlinespec_breadth",
    ):
        family = tuple(
            row
            for row in ordered_results
            if row.hypothesis.family_id == family_id and row.status == "TESTED"
        )
        if not family:
            continue
        adjusted = benjamini_hochberg(
            {
                row.hypothesis.hypothesis_id: row.contrast.raw_p_value
                for row in family
                if row.contrast is not None
            },
            false_discovery_rate=E0_BREADTH_FALSE_DISCOVERY_RATE,
        )
        by_id = {row.hypothesis.hypothesis_id: row for row in family}
        for decision in adjusted:
            source = by_id[decision.name]
            assert source.contrast_artifact_sha256 is not None
            fdr_decisions.append(
                E0BreadthFdrDecision(
                    family_id=source.hypothesis.family_id,
                    hypothesis_id=decision.name,
                    raw_p_value=decision.raw_p_value,
                    q_value=decision.adjusted_p_value,
                    rejected=decision.rejected,
                    procedure="benjamini-hochberg",
                    false_discovery_rate=E0_BREADTH_FALSE_DISCOVERY_RATE,
                    contrast_artifact_sha256=source.contrast_artifact_sha256,
                )
            )
    receipt = E0FormalBreadthFdrReceipt(
        schema_version=1,
        kind="e0_formal_breadth_fdr_receipt",
        registry_sha256=registry.sha256,
        protocol_sha256=E0_BREADTH_FDR_PROTOCOL_SHA256,
        final_completion_receipt_sha256=projection.completion_receipt.sha256,
        final_analysis_projection_sha256=projection.sha256,
        hypotheses_sha256=content_sha256(
            [hypothesis.to_dict() for hypothesis in hypotheses]
        ),
        hypotheses=ordered_results,
        decisions=tuple(
            sorted(fdr_decisions, key=lambda decision: decision.hypothesis_id)
        ),
        primary_family_eligible=False,
        formal_result_authorized=True,
    )
    receipt.__post_init__()
    return receipt


def reduce_formal_e0_breadth_fdr_from_artifact(
    registry: ExperimentRegistry,
    final_result_rebuild_artifact_path: str | Path,
    *,
    now_ns: int,
) -> E0FormalBreadthFdrReceipt:
    """Deep-rebuild the final proof DAG before deriving any p-value."""

    projection = rebuild_e0_final_analysis_projection_from_artifact(
        final_result_rebuild_artifact_path,
        now_ns=now_ns,
    )
    return reduce_formal_e0_breadth_fdr_from_projection(registry, projection)


def require_formal_e0_breadth_fdr_authority(
    registry: ExperimentRegistry,
    final_result_rebuild_artifact_path: str | Path,
    *,
    now_ns: int,
) -> E0FormalBreadthFdrReceipt:
    """Deep-rebuild E0 final proofs and derive the registered FDR receipt.

    The formal path deliberately does not accept raw p-values.  The only
    numerical source is the path-bound E0 final proof DAG, which is reopened by
    the same reducer used for the completion receipt.
    """

    return reduce_formal_e0_breadth_fdr_from_artifact(
        registry,
        final_result_rebuild_artifact_path,
        now_ns=now_ns,
    )


def reduce_e0_breadth_fdr(
    registry: ExperimentRegistry,
    authority: E0BreadthFdrAuthority,
) -> E0BreadthFdrReduction:
    """Apply fixed BH FDR separately to both complete secondary families."""

    bound = revalidate_e0_breadth_fdr_authority(registry, authority)
    hypotheses = registered_e0_breadth_hypotheses(registry)
    hypothesis_by_id = {value.hypothesis_id: value for value in hypotheses}
    raw_by_id = {value.hypothesis_id: value for value in bound.p_values}
    decisions: list[E0BreadthFdrDecision] = []
    for family_id in (
        "e0_core_breadth",
        "e0_isolated_onlinespec_breadth",
    ):
        family_hypotheses = tuple(
            value for value in hypotheses if value.family_id == family_id
        )
        adjusted = benjamini_hochberg(
            {
                value.hypothesis_id: raw_by_id[value.hypothesis_id].raw_p_value
                for value in family_hypotheses
            },
            false_discovery_rate=E0_BREADTH_FALSE_DISCOVERY_RATE,
        )
        for decision in adjusted:
            hypothesis = hypothesis_by_id[decision.name]
            raw_value = raw_by_id[decision.name]
            decisions.append(
                E0BreadthFdrDecision(
                    family_id=hypothesis.family_id,
                    hypothesis_id=hypothesis.hypothesis_id,
                    raw_p_value=decision.raw_p_value,
                    q_value=decision.adjusted_p_value,
                    rejected=decision.rejected,
                    procedure="benjamini-hochberg",
                    false_discovery_rate=E0_BREADTH_FALSE_DISCOVERY_RATE,
                    contrast_artifact_sha256=(raw_value.contrast_artifact_sha256),
                )
            )
    return E0BreadthFdrReduction(
        schema_version=1,
        kind="e0_breadth_fdr_reduction",
        registry_sha256=registry.sha256,
        protocol_sha256=E0_BREADTH_LEGACY_DIAGNOSTIC_PROTOCOL_SHA256,
        authority_sha256=bound.sha256,
        raw_source_sha256=bound.raw_source_sha256,
        hypotheses_sha256=bound.hypotheses_sha256,
        families=(
            "e0_core_breadth",
            "e0_isolated_onlinespec_breadth",
        ),
        decisions=tuple(sorted(decisions, key=lambda value: value.hypothesis_id)),
        primary_family_eligible=False,
        formal_execution_authorized=False,
    )


__all__ = [
    "E0_BREADTH_BOOTSTRAP_REPETITIONS",
    "E0_BREADTH_BOOTSTRAP_SEED",
    "E0_BREADTH_FALSE_DISCOVERY_RATE",
    "E0_BREADTH_FDR_PROTOCOL_SHA256",
    "E0_BREADTH_FORMAL_SOURCE_UNAVAILABLE_REASON",
    "E0_BREADTH_LEGACY_DIAGNOSTIC_PROTOCOL_SHA256",
    "E0_BREADTH_RAW_SOURCE_MISSING_REASON",
    "E0BreadthFdrAuthority",
    "E0BreadthFdrAuthorityBlocked",
    "E0BreadthFdrDecision",
    "E0BreadthFdrReduction",
    "E0BreadthHypothesis",
    "E0BreadthRawPValue",
    "E0FormalBreadthFdrReceipt",
    "E0FormalBreadthHypothesis",
    "E0FormalBreadthHypothesisResult",
    "E0PairedSloGoodputContrast",
    "SignedE0FormalBreadthFdrReceipt",
    "bind_e0_breadth_fdr_authority",
    "formal_e0_breadth_fdr_receipt_from_dict",
    "formal_e0_breadth_fdr_receipt_to_dict",
    "reduce_e0_breadth_fdr",
    "reduce_formal_e0_breadth_fdr_from_artifact",
    "reduce_formal_e0_breadth_fdr_from_projection",
    "registered_e0_breadth_hypotheses",
    "registered_formal_e0_breadth_hypotheses",
    "require_formal_e0_breadth_fdr_authority",
    "revalidate_e0_breadth_fdr_authority",
    "signed_formal_e0_breadth_fdr_from_dict",
    "signed_formal_e0_breadth_fdr_to_dict",
]
