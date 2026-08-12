"""Registered E0 secondary-family authority and Benjamini--Hochberg reducer.

E0 breadth findings are secondary hypotheses.  Their family membership is
derived from the complete E0 registry rather than supplied by a caller: core
L0 comparisons form one family and isolated OnlineSPEC comparisons form a
second.  The raw input must cover both families and every registered
hypothesis exactly once.  Missing rows, post-hoc regrouping, foreign cells, or
non-finite p-values fail closed.

This module adjusts already-derived raw p-values.  It does not derive p-values
from GPU evidence, authorize E0 execution, or promote a secondary finding into
the E3b primary Holm family.
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

from lightcone_spec.experiments.registry import (
    E0_BACKENDS,
    E0_MODELS,
    E0_TASKS,
    ExperimentCell,
    ExperimentRegistry,
    content_sha256,
)
from lightcone_spec.experiments.statistics import benjamini_hochberg

E0BreadthFamilyId = Literal[
    "e0_core_breadth",
    "e0_isolated_onlinespec_breadth",
]

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RAW_BYTES = 32 * 1024 * 1024

E0_BREADTH_FALSE_DISCOVERY_RATE = 0.05
E0_BREADTH_RAW_SOURCE_MISSING_REASON = "e0_breadth_raw_p_values_missing"
E0_BREADTH_FORMAL_SOURCE_UNAVAILABLE_REASON = (
    "e0_breadth_release_trusted_raw_source_unavailable"
)

# Empty until a release-owned raw-source digest is locked.  Keeping the
# allowlist in source makes formal trust a release decision, never a caller
# assertion.  The diagnostic binder below remains useful for schema and
# reducer tests, but its output explicitly cannot authorize formal execution.
E0_BREADTH_RELEASE_TRUSTED_RAW_SOURCE_SHA256: tuple[str, ...] = ()

_CORE_CONTRASTS = (
    ("l0_vs_static", "l0", "static"),
    ("l0_vs_tts", "l0", "tts"),
)
_ONLINESPEC_CONTRASTS = (
    ("onlinespec_ens_vs_static", "onlinespec_ens", "static"),
    ("onlinespec_ogd_vs_static", "onlinespec_ogd", "static"),
    ("onlinespec_opt_vs_static", "onlinespec_opt", "static"),
)

E0_BREADTH_FDR_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e0_registered_secondary_breadth_fdr",
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
        "coverage": "all_registered_hypotheses_exactly_once",
        "primary_family": "forbidden",
    }
)


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


def _hypothesis(
    *,
    family_id: E0BreadthFamilyId,
    model: str,
    backend: str,
    task: str,
    contrast: str,
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
        "numerator_method": numerator.identity.method,
        "denominator_method": denominator.identity.method,
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
        numerator_method=numerator.identity.method,
        denominator_method=denominator.identity.method,
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
    cells = registry.cells_for("E0")
    expected_cell_count = len(E0_MODELS) * len(E0_BACKENDS) * len(E0_TASKS) * 7
    if len(cells) != expected_cell_count:
        raise ValueError("E0 registry cell universe is incomplete")
    by_key: dict[tuple[str, str, str, str], ExperimentCell] = {}
    for cell in cells:
        key = (
            cell.identity.model,
            cell.identity.backend,
            cell.identity.task,
            cell.identity.method,
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
                                numerator=numerator,
                                denominator=denominator,
                            )
                        )
    result = tuple(sorted(hypotheses, key=lambda value: value.hypothesis_id))
    if len(result) != 540 or len({row.hypothesis_id for row in result}) != 540:
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
        if self.protocol_sha256 != E0_BREADTH_FDR_PROTOCOL_SHA256:
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
        if self.protocol_sha256 != E0_BREADTH_FDR_PROTOCOL_SHA256:
            raise ValueError("E0 breadth reduction uses another protocol")
        if self.families != (
            "e0_core_breadth",
            "e0_isolated_onlinespec_breadth",
        ):
            raise ValueError("E0 breadth reduction changed its registered families")
        ids = tuple(value.hypothesis_id for value in self.decisions)
        if len(ids) != 540 or ids != tuple(sorted(set(ids))):
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
        or source["protocol_sha256"] != E0_BREADTH_FDR_PROTOCOL_SHA256
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
        protocol_sha256=E0_BREADTH_FDR_PROTOCOL_SHA256,
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


def require_formal_e0_breadth_fdr_authority(
    registry: ExperimentRegistry,
    raw_source_path: str | Path,
) -> E0BreadthFdrAuthority:
    """Require a release-owned E0 raw source before any file-system read.

    The release allowlist is intentionally empty today.  This entry point is
    therefore a named, pre-side-effect BLOCK rather than a way for a caller to
    bless a self-reported p-value artifact.
    """

    if not E0_BREADTH_RELEASE_TRUSTED_RAW_SOURCE_SHA256:
        raise E0BreadthFdrAuthorityBlocked(E0_BREADTH_FORMAL_SOURCE_UNAVAILABLE_REASON)
    authority = bind_e0_breadth_fdr_authority(registry, raw_source_path)
    if authority.raw_source_sha256 not in E0_BREADTH_RELEASE_TRUSTED_RAW_SOURCE_SHA256:
        raise E0BreadthFdrAuthorityBlocked("e0_breadth_raw_source_not_release_trusted")
    return authority


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
        protocol_sha256=E0_BREADTH_FDR_PROTOCOL_SHA256,
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
    "E0_BREADTH_FALSE_DISCOVERY_RATE",
    "E0_BREADTH_FDR_PROTOCOL_SHA256",
    "E0_BREADTH_FORMAL_SOURCE_UNAVAILABLE_REASON",
    "E0_BREADTH_RAW_SOURCE_MISSING_REASON",
    "E0_BREADTH_RELEASE_TRUSTED_RAW_SOURCE_SHA256",
    "E0BreadthFdrAuthority",
    "E0BreadthFdrAuthorityBlocked",
    "E0BreadthFdrDecision",
    "E0BreadthFdrReduction",
    "E0BreadthHypothesis",
    "E0BreadthRawPValue",
    "bind_e0_breadth_fdr_authority",
    "reduce_e0_breadth_fdr",
    "registered_e0_breadth_hypotheses",
    "require_formal_e0_breadth_fdr_authority",
    "revalidate_e0_breadth_fdr_authority",
]
