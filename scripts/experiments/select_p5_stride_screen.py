#!/usr/bin/env python3
"""Select deterministic TTS/L0 stride-screen winners from attested P5 analyses."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from lightcone_spec.orchestration.manifest import ExperimentManifest


CONTEXTS = (4096, 16384)
STRIDES = (1, 4, 8, 16)
METHODS = {"tts": "tts", "l0": "naive_async"}
OPTIONAL_SAFETY_COLUMNS = {
    "nonfinite_update_count",
    "adaptation_fallback_count",
}
TIE_FRACTION = 0.02
SELECTION_SCHEMA_VERSION = 2
SELECTION_RULE_ID = "p5_stride_screen_selection_v2"
CONTEXT_RESOLVED_PERFORMANCE_SCOPES = {
    "checkpoint_request",
    "context_resolved",
}
WORKLOAD_GLOBAL_PERFORMANCE_SCOPES = {
    "mixed_workload_global",
    "workload_global",
}


class SelectionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_receipt(
    path: Path, payload: dict[str, Any], evidence: Iterable[Path]
) -> dict[str, Any]:
    rows = [
        {"path": str(item.resolve()), "sha256": _sha256(item)}
        for item in sorted({Path(value).resolve() for value in evidence})
    ]
    body = {**payload, "evidence": rows}
    text = json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_text(path, text)
    _atomic_text(Path(str(path) + ".sha256"), _sha256(path) + "\n")
    return body


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectionError(f"JSON must be an object: {path}")
    return value


def _verify_exact_sidecar(path: Path, sidecar: Path | None = None) -> Path:
    sidecar = sidecar or Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise SelectionError(f"missing SHA-256 sidecar: {sidecar}")
    if sidecar.read_text(encoding="utf-8").strip() != _sha256(path):
        raise SelectionError(f"SHA-256 sidecar mismatch: {path}")
    return sidecar


def _load_analysis(
    root: Path,
    *,
    baseline: str,
    expected_manifest_sha256: str,
    expected_unit_ids: set[str],
) -> tuple[pd.DataFrame, list[Path]]:
    table_path = root / "p5_long_context_acceptance.parquet"
    manifest_path = root / "analysis-manifest.json"
    hashes_path = root / "analysis-hashes.json"
    for path in (table_path, manifest_path, hashes_path):
        if not path.is_file():
            raise SelectionError(f"analysis evidence is missing: {path}")
    # ``analyze`` publishes this provenance sidecar without the JSON suffix.
    manifest_sidecar = _verify_exact_sidecar(
        manifest_path, root / "analysis-manifest.sha256"
    )
    analysis_manifest = _load_json(manifest_path)
    analysis = analysis_manifest.get("analysis", {})
    if analysis.get("baseline") != baseline:
        raise SelectionError(
            f"{root}: baseline is {analysis.get('baseline')!r}, expected {baseline!r}"
        )
    if analysis.get("expected_manifest_sha256") != expected_manifest_sha256:
        raise SelectionError(f"{root}: expected manifest identity mismatch")
    observed_units = {
        str(row.get("unit_id"))
        for row in analysis_manifest.get("input_runs", [])
        if isinstance(row, dict)
    }
    if observed_units != expected_unit_ids:
        raise SelectionError(
            f"{root}: analyzed unit coverage differs from the stride manifest"
        )
    ledger = _load_json(hashes_path)
    for path in (table_path, manifest_path, manifest_sidecar):
        entry = ledger.get(path.name)
        if not isinstance(entry, dict) or entry.get("sha256") != _sha256(path):
            raise SelectionError(f"{root}: analysis hash ledger mismatch for {path.name}")
    try:
        frame = pd.read_parquet(table_path)
    except Exception as exc:
        raise SelectionError(f"cannot read {table_path}: {exc}") from exc
    return frame, [table_path, manifest_path, manifest_sidecar, hashes_path]


def _required_columns(frame: pd.DataFrame, *, label: str) -> None:
    required = {
        "method",
        "update_stride",
        "context_length",
        "survival_weighted_accepted_prefix",
        "target_calls_per_output_token",
        "decode_goodput_tps",
        "exactness_violations",
        "version_mismatch_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SelectionError(f"{label} analysis lacks required columns: {missing}")


def _performance_evidence(
    vs_static: pd.DataFrame, vs_tts: pd.DataFrame
) -> dict[str, Any]:
    """Resolve whether goodput is measured per context or for the whole unit.

    Frozen v7 analyses predate the scope column and copied one mixed 4K/16K
    unit wall-time denominator into both context rows.  They remain valid for
    algorithmic screening and one workload-level tie-break, but cannot support
    a context-specific performance claim.  New analyses must carry an explicit
    scope marker before their per-context values are exposed.
    """

    has_static = "performance_scope" in vs_static.columns
    has_tts = "performance_scope" in vs_tts.columns
    if has_static != has_tts:
        raise SelectionError(
            "vs-static/vs-tts performance scope evidence differs"
        )
    if not has_static:
        return {
            "scope": "mixed_context_workload_global",
            "source_marker": "legacy_unmarked_v7",
            "context_specific_goodput_available": False,
            "allowed_use": "candidate_workload_tiebreak_only",
            "engineering_pass_evaluated": False,
        }

    observed: list[set[str]] = []
    for label, frame in (("vs-static", vs_static), ("vs-tts", vs_tts)):
        if frame["performance_scope"].isna().any():
            raise SelectionError(f"{label} has missing performance_scope values")
        values = set(frame["performance_scope"].astype(str))
        if len(values) != 1:
            raise SelectionError(
                f"{label} mixes performance scopes: {sorted(values)}"
            )
        observed.append(values)
    if observed[0] != observed[1]:
        raise SelectionError("vs-static/vs-tts performance scope mismatch")
    marker = next(iter(observed[0]))
    if marker in CONTEXT_RESOLVED_PERFORMANCE_SCOPES:
        return {
            "scope": "context_resolved",
            "source_marker": marker,
            "context_specific_goodput_available": True,
            "allowed_use": "candidate_tiebreak_and_context_diagnostic_only",
            "engineering_pass_evaluated": False,
        }
    if marker in WORKLOAD_GLOBAL_PERFORMANCE_SCOPES:
        return {
            "scope": "mixed_context_workload_global",
            "source_marker": marker,
            "context_specific_goodput_available": False,
            "allowed_use": "candidate_workload_tiebreak_only",
            "engineering_pass_evaluated": False,
        }
    raise SelectionError(f"unsupported performance_scope marker: {marker!r}")


def _number(value: Any, *, field: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(output):
        raise SelectionError(f"{field} is not finite")
    return output


def _metric(row: pd.Series, name: str) -> float:
    return _number(row[name], field=name)


def _one_row(
    frame: pd.DataFrame, *, method: str, stride: int, context: int, label: str
) -> pd.Series:
    selected = frame[
        (frame["method"] == method)
        & (pd.to_numeric(frame["update_stride"], errors="coerce") == stride)
        & (pd.to_numeric(frame["context_length"], errors="coerce") == context)
    ]
    if len(selected) != 1:
        raise SelectionError(
            f"{label}: expected one ({method}, stride={stride}, context={context}) "
            f"row, got {len(selected)}"
        )
    return selected.iloc[0]


def _static_metrics(frame: pd.DataFrame, context: int) -> dict[str, float]:
    selected = frame[
        (frame["method"] == "static")
        & (pd.to_numeric(frame["context_length"], errors="coerce") == context)
    ]
    if selected.empty:
        raise SelectionError(f"missing Static row at context={context}")
    columns = (
        "survival_weighted_accepted_prefix",
        "target_calls_per_output_token",
        "decode_goodput_tps",
    )
    values = {name: [_number(value, field=name) for value in selected[name]] for name in columns}
    for name, observed in values.items():
        consistent = all(
            math.isclose(observed[0], item, rel_tol=1e-12, abs_tol=1e-12)
            for item in observed[1:]
        )
        if not consistent:
            raise SelectionError(f"conflicting duplicated Static {name} at context={context}")
    output = {name: observed[0] for name, observed in values.items()}
    return output


def _workload_goodput(
    frame: pd.DataFrame, *, method: str, stride: int, label: str
) -> float:
    selected = frame[
        (frame["method"] == method)
        & (pd.to_numeric(frame["update_stride"], errors="coerce") == stride)
        & pd.to_numeric(frame["context_length"], errors="coerce").isin(CONTEXTS)
    ]
    if len(selected) != len(CONTEXTS):
        raise SelectionError(
            f"{label}: expected one workload goodput row per context for "
            f"({method}, stride={stride}), got {len(selected)}"
        )
    values = [
        _number(value, field="decode_goodput_tps")
        for value in selected["decode_goodput_tps"]
    ]
    if not all(
        math.isclose(values[0], value, rel_tol=1e-12, abs_tol=1e-12)
        for value in values[1:]
    ):
        raise SelectionError(
            f"{label}: unmarked/global performance differs by context for "
            f"({method}, stride={stride}); context claims are unavailable and "
            "no single workload value can be attested"
        )
    return values[0]


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _validate_analysis_pair(vs_static: pd.DataFrame, vs_tts: pd.DataFrame) -> None:
    """Require both whole-root analyses to preserve every stride cell."""

    optional_static = OPTIONAL_SAFETY_COLUMNS & set(vs_static.columns)
    optional_tts = OPTIONAL_SAFETY_COLUMNS & set(vs_tts.columns)
    if optional_static != optional_tts:
        raise SelectionError(
            "vs-static/vs-tts optional safety evidence columns differ"
        )
    absolute = {
        "survival_weighted_accepted_prefix",
        "target_calls_per_output_token",
        "decode_goodput_tps",
        "exactness_violations",
        "version_mismatch_count",
        *optional_static,
    }
    if "performance_scope" in vs_static.columns:
        absolute.add("performance_scope")
    for method in METHODS.values():
        for stride in STRIDES:
            for context in CONTEXTS:
                left = _one_row(
                    vs_static,
                    method=method,
                    stride=stride,
                    context=context,
                    label="vs-static",
                )
                right = _one_row(
                    vs_tts,
                    method=method,
                    stride=stride,
                    context=context,
                    label="vs-tts",
                )
                for name in absolute:
                    if name == "performance_scope":
                        equal = str(left[name]) == str(right[name])
                    else:
                        equal = math.isclose(
                            _metric(left, name),
                            _metric(right, name),
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    if not equal:
                        raise SelectionError(
                            f"analysis metric drift for {method}/stride={stride}/"
                            f"context={context}: {name}"
                        )


def _unit_index(raw_manifest: dict[str, Any]) -> tuple[dict[tuple[str, int], str], set[str]]:
    index: dict[tuple[str, int], str] = {}
    all_ids: set[str] = set()
    for raw in raw_manifest.get("units", []):
        method, stride, unit_id = raw.get("method"), raw.get("stride"), raw.get("unit_id")
        if not isinstance(unit_id, str) or not unit_id:
            raise SelectionError("manifest unit lacks unit_id")
        all_ids.add(unit_id)
        if method in {"static", *METHODS.values()}:
            key = (str(method), int(stride))
            if key in index:
                raise SelectionError(f"duplicate manifest method/stride: {key}")
            index[key] = unit_id
    if set(index) != {
        ("static", 1),
        *((method, stride) for method in METHODS.values() for stride in STRIDES),
    }:
        raise SelectionError("manifest is not the expected Static + TTS/L0 stride screen")
    if len(all_ids) != 9:
        raise SelectionError("stride screen manifest must contain exactly nine units")
    return index, all_ids


def _coverage_status(coverage: dict[str, Any], expected: set[str]) -> dict[str, str]:
    cells = coverage.get("cells")
    if not isinstance(cells, dict) or set(cells) != expected:
        raise SelectionError("coverage unit set differs from the stride manifest")
    return {unit_id: str(cell.get("status")) for unit_id, cell in cells.items()}


def _candidate(
    *,
    family: str,
    stride: int,
    unit_id: str,
    status: str,
    vs_static: pd.DataFrame,
    required_safety: set[str],
    performance_evidence: dict[str, Any],
) -> dict[str, Any]:
    method = METHODS[family]
    reasons = [] if status == "complete_valid" else [f"unit_status:{status}"]
    contexts: dict[str, Any] = {}
    for context in CONTEXTS:
        row = _one_row(
            vs_static,
            method=method,
            stride=stride,
            context=context,
            label="vs-static",
        )
        base = _static_metrics(vs_static, context)
        acceptance = _metric(row, "survival_weighted_accepted_prefix")
        target_calls = _metric(row, "target_calls_per_output_token")
        goodput = _metric(row, "decode_goodput_tps")
        exactness = _metric(row, "exactness_violations")
        mismatch = _metric(row, "version_mismatch_count")
        if exactness != 0:
            reasons.append(f"context_{context}:exactness_violations")
        if mismatch != 0:
            reasons.append(f"context_{context}:version_mismatch")
        safety = {}
        for name in sorted(OPTIONAL_SAFETY_COLUMNS & set(vs_static.columns)):
            value = _metric(row, name)
            safety[name] = value
            if value != 0:
                reasons.append(f"context_{context}:{name}")
        context_performance = (
            {
                "decode_goodput_tps": goodput,
                "goodput_ratio_vs_static": _ratio(
                    goodput, base["decode_goodput_tps"]
                ),
                "goodput_claim_status": (
                    "available_context_resolved_candidate_screen_only"
                ),
            }
            if performance_evidence["context_specific_goodput_available"]
            else {
                "decode_goodput_tps": None,
                "goodput_ratio_vs_static": None,
                "goodput_claim_status": (
                    "unavailable_mixed_context_workload_global"
                ),
            }
        )
        contexts[str(context)] = {
            "survival_weighted_accepted_prefix": acceptance,
            "acceptance_gain_vs_static": acceptance
            - base["survival_weighted_accepted_prefix"],
            "target_calls_per_output_token": target_calls,
            "target_call_reduction_vs_static": base[
                "target_calls_per_output_token"
            ]
            - target_calls,
            **context_performance,
            "exactness_violations": exactness,
            "version_mismatch_count": mismatch,
            **safety,
        }
    absent_required = sorted(required_safety - set(vs_static.columns))
    reasons.extend(f"missing_required_safety_column:{name}" for name in absent_required)
    workload_performance = None
    if not performance_evidence["context_specific_goodput_available"]:
        workload_goodput = _workload_goodput(
            vs_static, method=method, stride=stride, label="vs-static"
        )
        static_goodput = _workload_goodput(
            vs_static, method="static", stride=1, label="vs-static"
        )
        workload_performance = {
            "scope": "mixed_context_workload_global",
            "context_lengths": list(CONTEXTS),
            "decode_goodput_tps": workload_goodput,
            "goodput_ratio_vs_static": _ratio(workload_goodput, static_goodput),
            "allowed_use": "candidate_workload_tiebreak_only",
            "engineering_claim_available": False,
        }
    return {
        "family": family,
        "method": method,
        "stride": stride,
        "unit_id": unit_id,
        "unit_status": status,
        "eligible": not reasons,
        "rejection_reasons": sorted(set(reasons)),
        "metrics_by_context": contexts,
        "workload_performance": workload_performance,
    }


def _tiebreak_goodput(candidate: dict[str, Any]) -> float:
    workload = candidate.get("workload_performance")
    if workload is not None:
        return _number(workload["decode_goodput_tps"], field="decode_goodput_tps")
    return _number(
        candidate["metrics_by_context"]["16384"]["decode_goodput_tps"],
        field="decode_goodput_tps",
    )


def _choose(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        return None
    metric = lambda row: row["metrics_by_context"]["16384"]
    best_gain = max(metric(row)["acceptance_gain_vs_static"] for row in eligible)
    gain_tolerance = TIE_FRACTION * max(abs(best_gain), 1e-12)
    acceptance_ties = [
        row
        for row in eligible
        if metric(row)["acceptance_gain_vs_static"] >= best_gain - gain_tolerance
    ]
    best_target_calls = min(
        metric(row)["target_calls_per_output_token"] for row in acceptance_ties
    )
    target_ties = [
        row
        for row in acceptance_ties
        if metric(row)["target_calls_per_output_token"]
        <= best_target_calls * (1.0 + TIE_FRACTION)
    ]
    return sorted(
        target_ties,
        key=lambda row: (
            -_tiebreak_goodput(row),
            int(row["stride"]),
            str(row["unit_id"]),
        ),
    )[0]


def _engineering_gate(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate.get("eligible") is not True:
        reasons.append("hard_safety")
    gains: list[float] = []
    reductions: list[float] = []
    for context in CONTEXTS:
        metrics = candidate["metrics_by_context"][str(context)]
        gain = _number(
            metrics["acceptance_gain_vs_static"],
            field="acceptance_gain_vs_static",
        )
        reduction = _number(
            metrics["target_call_reduction_vs_static"],
            field="target_call_reduction_vs_static",
        )
        gains.append(gain)
        reductions.append(reduction)
        if gain < 0:
            reasons.append(f"context_{context}:negative_acceptance_gain")
        if reduction < 0:
            reasons.append(f"context_{context}:negative_target_call_reduction")
    if not any(value > 0 for value in (*gains, *reductions)):
        reasons.append("no_strict_engineering_improvement")
    return not reasons, reasons


def _choose_engineering(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    eligible = []
    for row in candidates:
        passed, reasons = _engineering_gate(row)
        row["engineering_eligible"] = passed
        row["engineering_rejection_reasons"] = reasons
        if passed:
            eligible.append(row)
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            -_tiebreak_goodput(row),
            -sum(
                _number(
                    row["metrics_by_context"][str(context)][
                        "acceptance_gain_vs_static"
                    ],
                    field="acceptance_gain_vs_static",
                )
                for context in CONTEXTS
            ),
            int(row["stride"]),
            str(row["unit_id"]),
        ),
    )[0]


def _comparison(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for context in CONTEXTS:
        left = current["metrics_by_context"][str(context)]
        right = baseline["metrics_by_context"][str(context)]
        context_goodput_available = left["decode_goodput_tps"] is not None
        output[str(context)] = {
            "acceptance_gain": left["survival_weighted_accepted_prefix"]
            - right["survival_weighted_accepted_prefix"],
            "target_call_reduction": right["target_calls_per_output_token"]
            - left["target_calls_per_output_token"],
            "goodput_ratio": (
                _ratio(left["decode_goodput_tps"], right["decode_goodput_tps"])
                if context_goodput_available
                else None
            ),
            "goodput_claim_status": (
                "available_context_resolved_candidate_screen_only"
                if context_goodput_available
                else "unavailable_mixed_context_workload_global"
            ),
        }
    left_workload = current.get("workload_performance")
    right_workload = baseline.get("workload_performance")
    if left_workload is not None and right_workload is not None:
        output["workload_performance"] = {
            "scope": "mixed_context_workload_global",
            "goodput_ratio": _ratio(
                _number(
                    left_workload["decode_goodput_tps"],
                    field="decode_goodput_tps",
                ),
                _number(
                    right_workload["decode_goodput_tps"],
                    field="decode_goodput_tps",
                ),
            ),
            "allowed_use": "candidate_workload_tiebreak_only",
            "engineering_claim_available": False,
        }
    return output


def canonicalize_selection(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical v2 view, migrating immutable v1 receipts in memory."""

    selection = copy.deepcopy(payload)
    version = selection.get("schema_version")
    if version == SELECTION_SCHEMA_VERSION:
        winners = selection.get("winners")
        if not isinstance(winners, dict) or "tts_best" in winners:
            raise SelectionError("schema-v2 winners are not canonical")
        expected = {
            "tts_acceptance_best",
            "tts_engineering_best",
            "l0_best",
            "same_stride_tts_for_l0",
        }
        if set(winners) != expected:
            raise SelectionError("schema-v2 winner roles are incomplete")
        return selection
    if version != 1:
        raise SelectionError(f"unsupported selection schema: {version!r}")

    winners = selection.get("winners")
    candidates = selection.get("candidates")
    if (
        not isinstance(winners, dict)
        or set(winners) != {"tts_best", "l0_best", "same_stride_tts_for_l0"}
        or not isinstance(candidates, dict)
        or not isinstance(candidates.get("tts"), list)
    ):
        raise SelectionError("legacy selection winner/candidate schema is invalid")
    legacy_acceptance = winners.pop("tts_best")
    engineering = _choose_engineering(candidates["tts"])
    acceptance = next(
        (
            row
            for row in candidates["tts"]
            if isinstance(legacy_acceptance, dict)
            and row.get("unit_id") == legacy_acceptance.get("unit_id")
        ),
        legacy_acceptance,
    )
    if engineering is None and isinstance(acceptance, dict):
        engineering = copy.deepcopy(acceptance)
        engineering["engineering_eligible"] = False
        engineering["engineering_fallback_reason"] = (
            "no_eligible_engineering_candidate_used_tts_acceptance_best"
        )
    winners["tts_acceptance_best"] = acceptance
    winners["tts_engineering_best"] = engineering
    # Reinsert in the canonical serialization order used by new writes.
    selection["winners"] = {
        "tts_acceptance_best": winners["tts_acceptance_best"],
        "tts_engineering_best": winners["tts_engineering_best"],
        "l0_best": winners["l0_best"],
        "same_stride_tts_for_l0": winners["same_stride_tts_for_l0"],
    }
    comparisons = selection.get("comparisons")
    if isinstance(comparisons, dict) and "l0_best_vs_tts_best" in comparisons:
        comparisons["l0_best_vs_tts_acceptance_best"] = comparisons.pop(
            "l0_best_vs_tts_best"
        )
    rule = selection.get("selection_rule")
    if not isinstance(rule, dict):
        raise SelectionError("legacy selection rule is invalid")
    rule["id"] = SELECTION_RULE_ID
    rule["engineering_order"] = [
        "hard_safety",
        "nonnegative_4k_16k_acceptance_gain",
        "nonnegative_4k_16k_target_call_reduction",
        "at_least_one_strict_improvement",
        "candidate_only_goodput_desc",
        "acceptance_gain_4k_16k_sum_desc",
        "stride_asc",
        "unit_id_asc",
    ]
    confirmation = selection.get("confirmation_unit_ids")
    if isinstance(confirmation, list) and confirmation:
        ordered = [confirmation[0]]
        for row in (
            acceptance,
            engineering,
            winners["l0_best"],
            winners["same_stride_tts_for_l0"],
        ):
            if isinstance(row, dict) and row.get("unit_id") not in ordered:
                ordered.append(row["unit_id"])
        selection["confirmation_unit_ids"] = ordered
    selection["schema_version"] = SELECTION_SCHEMA_VERSION
    return selection


def _selection_payload(
    *,
    manifest_path: Path,
    coverage_path: Path,
    vs_static_root: Path,
    vs_tts_root: Path,
) -> tuple[dict[str, Any], list[Path]]:
    manifest = ExperimentManifest.load(manifest_path)
    raw_manifest = _load_json(manifest_path)
    unit_index, unit_ids = _unit_index(raw_manifest)
    coverage_sidecar = _verify_exact_sidecar(coverage_path)
    statuses = _coverage_status(_load_json(coverage_path), unit_ids)
    required_safety_raw = manifest.engine_params.get(
        "p5_stride_screen_required_safety_columns", []
    )
    if not isinstance(required_safety_raw, list) or not all(
        isinstance(value, str) for value in required_safety_raw
    ):
        raise SelectionError(
            "p5_stride_screen_required_safety_columns must be a string list"
        )
    required_safety = set(required_safety_raw)
    unknown_safety = sorted(required_safety - OPTIONAL_SAFETY_COLUMNS)
    if unknown_safety:
        raise SelectionError(f"unsupported required safety columns: {unknown_safety}")

    vs_static, static_evidence = _load_analysis(
        vs_static_root,
        baseline="static",
        expected_manifest_sha256=manifest.content_sha256(),
        expected_unit_ids=unit_ids,
    )
    vs_tts, tts_evidence = _load_analysis(
        vs_tts_root,
        baseline="tts",
        expected_manifest_sha256=manifest.content_sha256(),
        expected_unit_ids=unit_ids,
    )
    _required_columns(vs_static, label="vs-static")
    _required_columns(vs_tts, label="vs-tts")
    performance_evidence = _performance_evidence(vs_static, vs_tts)
    _validate_analysis_pair(vs_static, vs_tts)

    candidates = {}
    winners = {}
    for family, method in METHODS.items():
        rows = [
            _candidate(
                family=family,
                stride=stride,
                unit_id=unit_index[(method, stride)],
                status=statuses[unit_index[(method, stride)]],
                vs_static=vs_static,
                required_safety=required_safety,
                performance_evidence=performance_evidence,
            )
            for stride in STRIDES
        ]
        candidates[family] = rows
        winners[family] = _choose(rows)

    tts_acceptance_best, l0_best = winners["tts"], winners["l0"]
    tts_engineering_best = _choose_engineering(candidates["tts"])
    if tts_engineering_best is None and tts_acceptance_best is not None:
        tts_engineering_best = copy.deepcopy(tts_acceptance_best)
        tts_engineering_best["engineering_eligible"] = False
        tts_engineering_best["engineering_fallback_reason"] = (
            "no_eligible_engineering_candidate_used_tts_acceptance_best"
        )
    comparisons: dict[str, Any] = {}
    same_stride_tts = None
    if l0_best is not None:
        same_stride_tts = next(
            row for row in candidates["tts"] if row["stride"] == l0_best["stride"]
        )
        # This row-level assertion proves the vs-TTS analysis did not match L0
        # against a pooled or different-stride baseline.
        for context in CONTEXTS:
            row = _one_row(
                vs_tts,
                method="naive_async",
                stride=int(l0_best["stride"]),
                context=context,
                label="vs-tts",
            )
            declared = _number(
                row.get("acceptance_gain_vs_baseline"),
                field="acceptance_gain_vs_baseline",
            )
            computed = _comparison(l0_best, same_stride_tts)[str(context)][
                "acceptance_gain"
            ]
            if not math.isclose(declared, computed, rel_tol=1e-9, abs_tol=1e-9):
                raise SelectionError(
                    f"vs-tts same-stride acceptance pairing mismatch at {context}"
                )
        comparisons["l0_best_vs_same_stride_tts"] = _comparison(
            l0_best, same_stride_tts
        )
    if l0_best is not None and tts_acceptance_best is not None:
        comparisons["l0_best_vs_tts_acceptance_best"] = _comparison(
            l0_best, tts_acceptance_best
        )

    objective_pass = bool(
        tts_acceptance_best
        and l0_best
        and tts_acceptance_best["metrics_by_context"]["16384"][
            "acceptance_gain_vs_static"
        ]
        > 0
        and tts_acceptance_best["metrics_by_context"]["16384"][
            "target_call_reduction_vs_static"
        ]
        > 0
        and comparisons["l0_best_vs_tts_acceptance_best"]["16384"][
            "acceptance_gain"
        ]
        > 0
        and comparisons["l0_best_vs_tts_acceptance_best"]["16384"][
            "target_call_reduction"
        ]
        > 0
    )
    confirmation = [unit_index[("static", 1)]]
    for row in (
        tts_acceptance_best,
        tts_engineering_best,
        l0_best,
        same_stride_tts,
    ):
        if row is not None and row["unit_id"] not in confirmation:
            confirmation.append(row["unit_id"])
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "winner_selected" if objective_pass else "scientifically_blocked",
        "scope": "candidate_screen_only_no_claim",
        "objective_screen_pass": objective_pass,
        "selection_rule": {
            "id": SELECTION_RULE_ID,
            "ranking_context": 16384,
            "performance_tiebreak_scope": performance_evidence["scope"],
            "performance_tiebreak_allowed_use": performance_evidence[
                "allowed_use"
            ],
            "acceptance_tie_fraction": TIE_FRACTION,
            "target_calls_tie_fraction": TIE_FRACTION,
            "order": [
                "hard_safety",
                "acceptance_gain_vs_static_desc",
                "target_calls_per_output_token_asc",
                "candidate_only_goodput_tiebreak_desc",
                "stride_asc",
                "unit_id_asc",
            ],
            "engineering_order": [
                "hard_safety",
                "nonnegative_4k_16k_acceptance_gain",
                "nonnegative_4k_16k_target_call_reduction",
                "at_least_one_strict_improvement",
                "candidate_only_goodput_desc",
                "acceptance_gain_4k_16k_sum_desc",
                "stride_asc",
                "unit_id_asc",
            ],
            "required_optional_safety_columns": sorted(required_safety),
        },
        "cardinality": {
            "contexts": list(CONTEXTS),
            "strides": list(STRIDES),
            "no_stride_pooling": True,
        },
        "performance_evidence": performance_evidence,
        "candidates": candidates,
        "winners": {
            "tts_acceptance_best": tts_acceptance_best,
            "tts_engineering_best": tts_engineering_best,
            "l0_best": l0_best,
            "same_stride_tts_for_l0": same_stride_tts,
        },
        "comparisons": comparisons,
        "confirmation_unit_ids": confirmation,
        "source_inputs": {
            "manifest": str(manifest_path.resolve()),
            "coverage": str(coverage_path.resolve()),
            "vs_static_analysis": str(vs_static_root.resolve()),
            "vs_tts_analysis": str(vs_tts_root.resolve()),
        },
    }
    evidence = [
        Path(__file__).resolve(),
        manifest_path,
        coverage_path,
        coverage_sidecar,
        *static_evidence,
        *tts_evidence,
    ]
    return payload, evidence


def select(
    *,
    manifest_path: Path,
    coverage_path: Path,
    vs_static_root: Path,
    vs_tts_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    payload, evidence = _selection_payload(
        manifest_path=manifest_path,
        coverage_path=coverage_path,
        vs_static_root=vs_static_root,
        vs_tts_root=vs_tts_root,
    )
    return _write_receipt(output_path, payload, evidence)


def validate_selection_receipt(
    *, selector_path: Path, manifest_path: Path
) -> dict[str, Any]:
    """Recompute candidates, winners and comparisons from attested inputs."""

    _verify_exact_sidecar(selector_path)
    raw_receipt = _load_json(selector_path)
    receipt = canonicalize_selection(raw_receipt)
    raw_evidence = receipt.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise SelectionError("selection receipt has no evidence")
    observed_evidence: dict[Path, str] = {}
    for row in raw_evidence:
        if not isinstance(row, dict):
            raise SelectionError("selection receipt evidence row is invalid")
        path = Path(str(row.get("path", ""))).resolve()
        if path in observed_evidence:
            raise SelectionError(f"selection receipt repeats evidence: {path}")
        if not path.is_file() or row.get("sha256") != _sha256(path):
            raise SelectionError(f"selection receipt evidence mismatch: {path}")
        observed_evidence[path] = str(row["sha256"])

    inputs = receipt.get("source_inputs")
    required = {"manifest", "coverage", "vs_static_analysis", "vs_tts_analysis"}
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise SelectionError("selection receipt source_inputs are incomplete")
    recorded_manifest = Path(str(inputs["manifest"])).resolve()
    if recorded_manifest != manifest_path.resolve():
        raise SelectionError("selection receipt source manifest path mismatch")
    recomputed, evidence = _selection_payload(
        manifest_path=recorded_manifest,
        coverage_path=Path(str(inputs["coverage"])).resolve(),
        vs_static_root=Path(str(inputs["vs_static_analysis"])).resolve(),
        vs_tts_root=Path(str(inputs["vs_tts_analysis"])).resolve(),
    )
    declared = {key: value for key, value in receipt.items() if key != "evidence"}
    if declared != recomputed:
        raise SelectionError(
            "selection receipt semantics differ from recomputed candidates/winners"
        )
    expected_evidence = {path.resolve(): _sha256(path) for path in evidence}
    if observed_evidence != expected_evidence:
        raise SelectionError("selection receipt evidence set differs from recomputation")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--vs-static-analysis", type=Path, required=True)
    parser.add_argument("--vs-tts-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        select(
            manifest_path=args.manifest,
            coverage_path=args.coverage,
            vs_static_root=args.vs_static_analysis,
            vs_tts_root=args.vs_tts_analysis,
            output_path=args.output,
        )
    except SelectionError as exc:
        raise SystemExit(f"stride selection failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
