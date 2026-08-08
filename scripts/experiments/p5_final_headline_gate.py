#!/usr/bin/env python3
"""Publish the final, hash-closed P5 0--40K scientific verdict.

This helper is deliberately downstream of experiment generation and analysis.
It never mutates a manifest, run, controller, or queue.  A CONFIRMED receipt
means that the raw prompt/seed evidence proves the algorithmic claim; the
engineering verdict is separate and can optionally be made terminal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from lightcone_spec.artifacts.rundir import REQUIRED_FILES
from lightcone_spec.config.schema import canonical_weight_update_mode
from lightcone_spec.locking.hashing import sha256_file, sha256_json
from lightcone_spec.statistics.bootstrap import cluster_bca
from lightcone_spec.statistics.tables import (
    P5_IDENTITY_COLUMNS,
    expand_static_p5_identities,
    paired_cross_stride_acceptance_table,
)


CONTEXTS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 40000)
LC_METHODS = {
    "naive_async": "l0",
    "lc_gate": "l1",
    "lc_damp": "l2",
    "lc_transport": "l3",
}
ROLE_ALIASES = {
    "same_stride": (
        "same_stride_tts_for_l0",
        "same_stride_tts",
        "same_stride",
    ),
    "acceptance_best": (
        "tts_acceptance_best",
        "acceptance_best_tts",
        "acceptance_best",
    ),
    "engineering_best": (
        "tts_engineering_best",
        "engineering_best_tts",
        "engineering_best",
    ),
}
_REQUEST_GROUP = re.compile(r"^lightcone-g([0-9a-f]{64})-")


class FinalGateError(RuntimeError):
    pass


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


def _json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise FinalGateError(f"duplicate JSON key {key!r}: {path}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise FinalGateError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalGateError(f"JSON must be an object: {path}")
    return value


def _sidecar(path: Path, sidecar: Path | None = None) -> Path:
    sidecar = (sidecar or Path(str(path) + ".sha256")).resolve()
    path = path.resolve()
    if not path.is_file() or not sidecar.is_file():
        raise FinalGateError(f"artifact or SHA-256 sidecar is missing: {path}")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise FinalGateError(f"SHA-256 sidecar mismatch: {path}")
    return sidecar


def _receipt_tree(path: Path) -> tuple[dict[str, Any], list[Path]]:
    """Verify every nested receipt, sidecar and declared evidence hash."""

    visited: set[Path] = set()
    active: set[Path] = set()
    evidence: set[Path] = set()

    def visit(current: Path) -> dict[str, Any]:
        current = current.resolve()
        if current in active:
            raise FinalGateError(f"receipt evidence cycle: {current}")
        if current in visited:
            return _json(current)
        sidecar = _sidecar(current)
        payload = _json(current)
        rows = payload.get("evidence")
        if not isinstance(rows, list) or not rows:
            raise FinalGateError(f"receipt has no evidence: {current}")
        active.add(current)
        seen: set[Path] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise FinalGateError(f"receipt evidence row is invalid: {current}")
            raw_path = row.get("path")
            if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
                raise FinalGateError(f"receipt evidence path is not absolute: {current}")
            item = Path(raw_path).resolve()
            if item in seen:
                raise FinalGateError(f"receipt repeats direct evidence: {item}")
            if not item.is_file() or row.get("sha256") != sha256_file(item):
                raise FinalGateError(f"receipt evidence hash mismatch: {item}")
            seen.add(item)
            evidence.add(item)
            if item.suffix == ".json":
                nested = _json(item)
                if {"schema_version", "status", "scope", "evidence"}.issubset(
                    nested
                ):
                    visit(item)
        active.remove(current)
        visited.add(current)
        evidence.update((current, sidecar))
        return payload

    root = visit(path)
    return root, sorted(evidence)


def verify_resume_receipt(path: Path) -> dict[str, Any]:
    """Validate the only receipt allowed to unlock the legacy ablations."""

    payload, _evidence = _receipt_tree(path)
    if payload.get("status") != "CONFIRMED":
        raise FinalGateError("resume receipt is not CONFIRMED")
    if payload.get("scope") != "final_p5_0_40k_evidence_gate":
        raise FinalGateError("resume receipt has the wrong scientific scope")
    if payload.get("resume_old_ablations_allowed") is not True:
        raise FinalGateError("resume receipt does not authorize old ablations")
    if payload.get("all_declared_l0123_evaluated") is not True:
        raise FinalGateError("resume receipt lacks complete L0/L1/L2/L3 coverage")
    if payload.get("all_algorithmic_pass") is not True:
        raise FinalGateError("resume receipt lacks the complete algorithmic pass")
    methods = payload.get("eligible_lc_methods")
    if not isinstance(methods, list) or set(methods) != set(LC_METHODS):
        raise FinalGateError("resume receipt does not enumerate exactly L0/L1/L2/L3")
    status = payload.get("controller_method_status")
    if not isinstance(status, Mapping) or set(status) != set(LC_METHODS):
        raise FinalGateError("resume receipt has incomplete controller method status")
    if any(status[method] != "EVALUATED" for method in LC_METHODS):
        raise FinalGateError("resume receipt contains a non-evaluated LightCone method")

    declared_hash = payload.get("decision_sha256")
    decision = dict(payload)
    decision.pop("evidence", None)
    decision.pop("decision_sha256", None)
    if declared_hash != sha256_json(decision):
        raise FinalGateError("resume receipt decision hash mismatch")
    return payload


def _evidence_hashes(payload: Mapping[str, Any]) -> dict[Path, str]:
    output: dict[Path, str] = {}
    for row in payload.get("evidence", []):
        if isinstance(row, Mapping):
            output[Path(str(row.get("path", ""))).resolve()] = str(
                row.get("sha256", "")
            )
    return output


def _role_strides(payload: Mapping[str, Any]) -> dict[str, int]:
    raw = payload.get("roles", payload.get("role_strides"))
    if not isinstance(raw, Mapping):
        raise FinalGateError("TTS foundation terminal lacks three role strides")
    result = {}
    for role, aliases in ROLE_ALIASES.items():
        matches = [raw[name] for name in aliases if name in raw]
        if len(matches) != 1:
            raise FinalGateError(f"TTS foundation role is missing or ambiguous: {role}")
        value = matches[0]
        if isinstance(value, Mapping):
            if value.get("method", "tts") != "tts":
                raise FinalGateError(f"TTS foundation {role} role is not TTS")
            value = value.get("update_stride", value.get("stride"))
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise FinalGateError(f"TTS foundation {role} stride is invalid")
        result[role] = value
    return result


def _controller_methods(
    terminal: Mapping[str, Any], generation: Mapping[str, Any]
) -> list[str]:
    eligible = terminal.get("eligible")
    if (
        not isinstance(eligible, Mapping)
        or set(eligible) != {"l1", "l2", "l3"}
        or not all(isinstance(value, bool) for value in eligible.values())
    ):
        raise FinalGateError("controller terminal eligibility is malformed")
    methods = ["naive_async"]
    for method, gate in LC_METHODS.items():
        if method != "naive_async" and eligible[gate]:
            methods.append(method)
    declared = generation.get("methods")
    expected = {"static", "tts", *methods}
    if not isinstance(declared, list) or set(map(str, declared)) != expected:
        raise FinalGateError("headline generation/controller method coverage differs")
    return methods


def _all_lc_methods_pass(
    methods: Iterable[str], results: Mapping[str, Mapping[str, Any]], field: str
) -> bool:
    """Require an explicit passing row for L0, L1, L2 and L3."""

    expected = set(LC_METHODS)
    return bool(
        set(methods) == expected
        and set(results) == expected
        and all(results[method].get(field) is True for method in expected)
    )


def _verify_manifest(path: Path) -> tuple[dict[str, Any], list[Path], set[str]]:
    sidecar = _sidecar(path)
    payload = _json(path)
    units = payload.get("units")
    if not isinstance(units, list) or not units:
        raise FinalGateError(f"experiment manifest has no units: {path}")
    unit_ids = [str(row.get("unit_id", "")) for row in units if isinstance(row, Mapping)]
    if len(unit_ids) != len(units) or any(not item for item in unit_ids):
        raise FinalGateError(f"experiment manifest unit identity is invalid: {path}")
    if len(set(unit_ids)) != len(unit_ids):
        raise FinalGateError(f"experiment manifest repeats a unit: {path}")
    return payload, [path.resolve(), sidecar], set(unit_ids)


def _verify_analysis(
    *, root: Path, artifact_root: Path, manifest_path: Path
) -> tuple[dict[str, Any], list[Path], list[Path]]:
    """Verify derived-output ledger and every immutable input run ledger."""

    root = root.resolve()
    artifact_root = artifact_root.resolve()
    manifest_path = manifest_path.resolve()
    _, manifest_evidence, expected_units = _verify_manifest(manifest_path)
    analysis_path = root / "analysis-manifest.json"
    analysis_sidecar = _sidecar(analysis_path, root / "analysis-manifest.sha256")
    analysis = _json(analysis_path)
    hashes_path = root / "analysis-hashes.json"
    hashes = _json(hashes_path)
    derived = analysis.get("derived_outputs")
    if not isinstance(derived, Mapping) or not derived:
        raise FinalGateError(f"analysis has no derived outputs: {root}")
    for relative, row in derived.items():
        candidate = (root / str(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise FinalGateError(f"analysis output escapes its root: {relative}") from exc
        if (
            not isinstance(row, Mapping)
            or not candidate.is_file()
            or row.get("sha256") != sha256_file(candidate)
            or row.get("bytes") != candidate.stat().st_size
            or hashes.get(relative) != row
        ):
            raise FinalGateError(f"analysis derived-output hash mismatch: {candidate}")
    expected_hash_keys = {
        *map(str, derived),
        "analysis-manifest.json",
        "analysis-manifest.sha256",
    }
    if set(hashes) != expected_hash_keys:
        raise FinalGateError("analysis hash ledger has missing or unexpected outputs")
    for name, candidate in (
        ("analysis-manifest.json", analysis_path),
        ("analysis-manifest.sha256", analysis_sidecar),
    ):
        row = hashes.get(name)
        if not isinstance(row, Mapping) or row != {
            "sha256": sha256_file(candidate),
            "bytes": candidate.stat().st_size,
        }:
            raise FinalGateError(f"analysis provenance hash mismatch: {candidate}")
    expected_sha = analysis.get("analysis", {}).get("expected_manifest_sha256")
    if expected_sha not in {sha256_file(manifest_path), sha256_json(_json(manifest_path))}:
        raise FinalGateError("analysis is not bound to the supplied experiment manifest")
    input_runs = analysis.get("input_runs")
    if not isinstance(input_runs, list) or not input_runs:
        raise FinalGateError("analysis has no immutable input runs")
    observed_units: set[str] = set()
    run_dirs: list[Path] = []
    evidence = [*manifest_evidence, analysis_path, analysis_sidecar, hashes_path]
    for row in input_runs:
        if not isinstance(row, Mapping):
            raise FinalGateError("analysis input-run row is invalid")
        run = (artifact_root / str(row.get("run_id", ""))).resolve()
        try:
            run.relative_to(artifact_root)
        except ValueError as exc:
            raise FinalGateError("analysis input run escapes its artifact root") from exc
        manifest = run / "manifest.json"
        ledger_path = run / "hashes.json"
        if (
            not run.is_dir()
            or row.get("manifest_sha256") != sha256_file(manifest)
            or row.get("hashes_sha256") != sha256_file(ledger_path)
        ):
            raise FinalGateError(f"analysis input-run binding drift: {run}")
        run_manifest = _json(manifest)
        unit_id = str(run_manifest.get("unit_id", ""))
        if unit_id != str(row.get("unit_id", "")) or unit_id in observed_units:
            raise FinalGateError(f"analysis run has duplicate/mismatched unit: {run}")
        ledger = _json(ledger_path)
        required = set(REQUIRED_FILES) - {"hashes.json"}
        if not required.issubset(ledger):
            raise FinalGateError(f"run ledger omits required files: {run}")
        for relative, entry in ledger.items():
            item = (run / relative).resolve()
            try:
                item.relative_to(run.resolve())
            except ValueError as exc:
                raise FinalGateError(f"run ledger path escapes its root: {relative}") from exc
            if (
                not isinstance(entry, Mapping)
                or not item.is_file()
                or entry.get("sha256") != sha256_file(item)
                or entry.get("bytes") != item.stat().st_size
            ):
                raise FinalGateError(f"run hash ledger mismatch: {item}")
            evidence.append(item)
        if _json(run / "exit.json").get("status") != "complete_valid":
            raise FinalGateError(f"analysis contains a non-valid run: {run}")
        observed_units.add(unit_id)
        run_dirs.append(run)
        evidence.extend((manifest, ledger_path))
    if observed_units != expected_units:
        raise FinalGateError(
            f"manifest/run coverage mismatch: missing={sorted(expected_units-observed_units)}, "
            f"extra={sorted(observed_units-expected_units)}"
        )
    return analysis, sorted(set(evidence)), run_dirs


def _snapshot(paths: Iterable[Path]) -> dict[Path, str]:
    return {Path(path).resolve(): sha256_file(path) for path in set(paths)}


def _verify_snapshot(snapshot: Mapping[Path, str]) -> None:
    for path, digest in snapshot.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise FinalGateError(f"evidence changed during final-gate evaluation: {path}")


def _raw_prompt_table(run_dirs: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in run_dirs:
        manifest = _json(run / "manifest.json")
        checkpoints = _json(run / "prefix-checkpoints.json").get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise FinalGateError(f"P5 run lacks exact prefix checkpoints: {run}")
        context: dict[str, int] = {}
        cluster: dict[str, str] = {}
        for item in checkpoints:
            group = hashlib.sha256(str(item["sample_id"]).encode()).hexdigest()
            context[group] = int(item["context_length"])
            cluster[group] = hashlib.sha256(
                str(item["source_sample_id"]).encode()
            ).hexdigest()
        contexts = sorted(set(context.values()))
        frame = pd.read_parquet(run / "rounds.parquet")
        if frame.empty:
            raise FinalGateError(f"P5 run has no raw rounds: {run}")
        aggregates: dict[tuple[str, int], list[float]] = {}
        for record in frame.to_dict("records"):
            request_id = str(record.get("request_id", ""))
            match = _REQUEST_GROUP.match(request_id)
            group = match.group(1) if match else None
            if group not in context:
                raise FinalGateError(f"round has no locked prefix checkpoint: {request_id}")
            prefix_exact = record.get("prefix_feature_exact")
            if not isinstance(prefix_exact, (bool, np.bool_)) or not bool(prefix_exact):
                raise FinalGateError(f"round lacks an exact prefix feature: {request_id}")
            prefix = int(record.get("prefix_len_before"))
            bucket = context[group]
            next_context = next((item for item in contexts if item > bucket), None)
            if prefix < bucket or (next_context is not None and prefix >= next_context):
                raise FinalGateError(f"round escaped its exact context bucket: {request_id}")
            censored = record.get("algorithmic_censored")
            if not isinstance(censored, (bool, np.bool_)):
                raise FinalGateError(
                    f"round lacks an explicit censoring state: {request_id}"
                )
            if bool(censored):
                continue
            key = (group, bucket)
            cell = aggregates.setdefault(key, [0.0, 0.0])
            cell[0] += float(record.get("accepted_drafts", 0))
            cell[1] += 1.0
            version_ok = record.get("version_canary_ok")
            if not isinstance(version_ok, (bool, np.bool_)) or not bool(version_ok):
                raise FinalGateError(f"raw round version canary failed: {request_id}")
            cache_canary = record.get("cache_version_canary_ok")
            if cache_canary is not None and not bool(pd.isna(cache_canary)):
                if not isinstance(cache_canary, (bool, np.bool_)) or not bool(
                    cache_canary
                ):
                    raise FinalGateError(
                        f"raw round cache version canary failed: {request_id}"
                    )
        stride = manifest.get("stride", manifest.get("update_stride", 4))
        mode = canonical_weight_update_mode(
            manifest.get("weight_update_mode", manifest.get("trainable_scope", "output_residual"))
        )
        repetitions = int(manifest.get("engine_params", {}).get("benchmark_repetitions", 1))
        for (group, bucket), (accepted, count) in aggregates.items():
            rows.append(
                {
                    "method": str(manifest["method"]),
                    "model_pair": str(manifest["model_pair"]),
                    "weight_update_mode": str(mode),
                    "update_stride": int(stride),
                    "dataset": str(manifest["dataset"]),
                    "lifecycle": str(manifest["lifecycle"]),
                    "offered_concurrency": int(manifest["concurrency"]),
                    "context_length": int(bucket),
                    "prompt_cluster": cluster[group],
                    "seed": int(manifest["seed"]),
                    "accepted_sum": accepted,
                    "round_count": count,
                    "benchmark_repetitions": repetitions,
                }
            )
    frame = pd.DataFrame(rows)
    group_columns = [
        "method",
        *P5_IDENTITY_COLUMNS,
        "context_length",
        "prompt_cluster",
        "seed",
    ]
    if not frame.empty:
        repetition_span = frame.groupby(group_columns, dropna=False)[
            "benchmark_repetitions"
        ].agg(["min", "max"])
        if (repetition_span["min"] != repetition_span["max"]).any():
            raise FinalGateError("raw prompt groups disagree on repetition count")
        frame = frame.groupby(group_columns, as_index=False, dropna=False).agg(
            accepted_sum=("accepted_sum", "sum"),
            round_count=("round_count", "sum"),
            benchmark_repetitions=("benchmark_repetitions", "min"),
        )
        frame["acceptance"] = frame["accepted_sum"] / frame["round_count"]
        frame = frame.drop(columns=["accepted_sum", "round_count"])
        frame = expand_static_p5_identities(frame)
    required = {"method", *P5_IDENTITY_COLUMNS, "context_length", "prompt_cluster", "seed", "acceptance"}
    if frame.empty or not required.issubset(frame):
        raise FinalGateError("raw prompt acceptance reconstruction is empty")
    duplicate = ["method", *P5_IDENTITY_COLUMNS, "context_length", "prompt_cluster", "seed"]
    if frame.duplicated(duplicate).any():
        raise FinalGateError("raw runs duplicate a prompt/seed/identity cell")
    return frame.sort_values(duplicate).reset_index(drop=True)


def _verify_prompt_derivation(raw: pd.DataFrame, analysis_root: Path) -> None:
    derived = pd.read_parquet(analysis_root / "p5_prompt_acceptance.parquet")
    columns = [
        "method", *P5_IDENTITY_COLUMNS, "context_length", "prompt_cluster", "seed",
        "acceptance", "benchmark_repetitions",
    ]
    if any(name not in derived for name in columns):
        raise FinalGateError("derived prompt table lacks required evidence columns")
    left = raw[columns].sort_values(columns[:-2]).reset_index(drop=True)
    right = derived[columns].sort_values(columns[:-2]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=1e-12, atol=1e-12)
    except AssertionError as exc:
        raise FinalGateError("derived prompt acceptance differs from raw rounds") from exc


def _comparison(
    prompt: pd.DataFrame,
    *,
    method: str,
    stride: int,
    baseline_method: str,
    baseline_stride: int,
    b: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        table = paired_cross_stride_acceptance_table(
            prompt,
            candidate_method=method,
            candidate_stride=stride,
            baseline_method=baseline_method,
            baseline_stride=baseline_stride,
            b=b,
        )
    except ValueError as exc:
        raise FinalGateError(f"cross-stride comparison failed: {exc}") from exc
    observed = set(pd.to_numeric(table["context_length"]).astype(int))
    if observed != set(CONTEXTS) or len(table) != len(CONTEXTS):
        raise FinalGateError(f"comparison context coverage differs: {sorted(observed)}")
    if (table["paired_prompt_clusters"] < 2).any():
        raise FinalGateError("comparison has fewer than two prompt clusters")
    selected_candidate = prompt[
        (prompt.method == method) & (prompt.update_stride == stride)
    ]
    selected_baseline = prompt[
        (prompt.method == baseline_method)
        & (prompt.update_stride == baseline_stride)
    ]
    if (
        selected_candidate["benchmark_repetitions"].min() < 5
        or selected_baseline["benchmark_repetitions"].min() < 5
    ):
        raise FinalGateError("headline evidence requires at least five repetitions")

    long_table = table[table.context_length >= 4096]
    values = []
    clusters = []
    identity_no_stride = [name for name in P5_IDENTITY_COLUMNS if name != "update_stride"]
    pair_keys = [*identity_no_stride, "context_length", "prompt_cluster", "seed"]
    candidate = selected_candidate.set_index(pair_keys)["acceptance"]
    baseline = selected_baseline.set_index(pair_keys)["acceptance"]
    if set(candidate.index) != set(baseline.index):
        raise FinalGateError("LCAG prompt/seed coverage is not exactly paired")
    for key in sorted(candidate.index):
        if int(key[len(identity_no_stride)]) >= 4096:
            values.append(float(candidate[key] - baseline[key]))
            clusters.append(str(key[len(identity_no_stride) + 1]))
    lcag = cluster_bca(np.asarray(values), np.asarray(clusters), np.mean, b=b)

    ordered = table.sort_values("context_length")
    contexts = ordered.context_length.to_numpy(dtype=float)
    candidate_a = ordered.candidate_acceptance.to_numpy(dtype=float)
    baseline_a = ordered.baseline_acceptance.to_numpy(dtype=float)
    if (candidate_a <= 0).any() or (baseline_a <= 0).any():
        raise FinalGateError("elasticity requires strictly positive acceptance")
    candidate_e = -np.diff(np.log(candidate_a)) / np.diff(np.log(contexts))
    baseline_e = -np.diff(np.log(baseline_a)) / np.diff(np.log(contexts))
    interval_left = contexts[:-1]
    long_delta = candidate_e[interval_left >= 4096] - baseline_e[interval_left >= 4096]
    if not len(long_delta):
        raise FinalGateError("no long-context elasticity intervals")
    summary = {
        "candidate_method": method,
        "candidate_stride": stride,
        "baseline_method": baseline_method,
        "baseline_stride": baseline_stride,
        "lcag": lcag.estimate,
        "lcag_ci_low": lcag.ci_low,
        "lcag_ci_high": lcag.ci_high,
        "lcag_prompt_clusters": lcag.n_clusters,
        "mean_delta_acceptance_elasticity": float(long_delta.mean()),
        "per_context": json.loads(
            ordered[[
                "context_length", "candidate_acceptance", "baseline_acceptance",
                "acceptance_gain", "acceptance_gain_ci_low", "acceptance_gain_ci_high",
                "paired_prompt_clusters",
            ]].to_json(orient="records", double_precision=15)
        ),
    }
    summary["algorithmic_pass"] = bool(
        summary["lcag_ci_low"] > 0
        and summary["mean_delta_acceptance_elasticity"] < 0
    )
    return table, summary


def _standard_same_stride_gate(
    *, analysis_root: Path, method: str, stride: int
) -> dict[str, Any]:
    payload = json.loads((analysis_root / "p5_claim_gates.json").read_text())
    if not isinstance(payload, list):
        raise FinalGateError("standard p5_claim_gates must be a list")
    matches = [
        row for row in payload
        if isinstance(row, Mapping)
        and row.get("method") == method
        and row.get("baseline_method") == "tts"
        and row.get("update_stride") == stride
    ]
    if len(matches) != 1:
        raise FinalGateError(f"expected one standard same-stride gate for {method}@{stride}")
    row = dict(matches[0])
    if row.get("benchmark_repetitions", 0) < 5 or row.get("exactness_pass") is not True:
        raise FinalGateError(f"same-stride gate has insufficient/unsafe evidence: {method}")
    return row


def _safety(analysis_root: Path, methods: Iterable[str]) -> dict[str, Any]:
    frame = pd.read_parquet(analysis_root / "p5_long_context_acceptance.parquet")
    required = {
        "method", "context_length", "adaptation_fallback_count",
        "exactness_violations", "version_mismatch_count",
    }
    if not required.issubset(frame):
        raise FinalGateError("P5 safety table lacks required columns")
    output = {}
    for method in methods:
        selected = frame[frame.method == method]
        if set(selected.context_length.astype(int)) != set(CONTEXTS):
            raise FinalGateError(f"safety context coverage differs for {method}")
        counts = {
            name: int(pd.to_numeric(selected[name], errors="raise").sum())
            for name in (
                "adaptation_fallback_count", "exactness_violations",
                "version_mismatch_count",
            )
        }
        counts["pass"] = all(value == 0 for value in counts.values())
        output[method] = counts
    return output


def _engineering(
    *, analysis_root: Path, method: str, stride: int, baseline_stride: int
) -> dict[str, Any]:
    frame = pd.read_parquet(analysis_root / "p5_long_context_acceptance.parquet")
    required = {
        "method", "update_stride", "context_length", "offered_concurrency",
        "decode_goodput_tps", "target_calls_per_output_token",
    }
    if not required.issubset(frame):
        raise FinalGateError("MFU analysis lacks context-specific engineering columns")
    rows = []
    for context in CONTEXTS:
        candidate = frame[
            (frame.method == method)
            & (frame.update_stride == stride)
            & (frame.context_length == context)
        ]
        baseline = frame[
            (frame.method == "tts")
            & (frame.update_stride == baseline_stride)
            & (frame.context_length == context)
        ]
        if len(candidate) != 1 or len(baseline) != 1:
            raise FinalGateError(f"MFU role cell is missing/ambiguous: {method}@{context}")
        left, right = candidate.iloc[0], baseline.iloc[0]
        if int(left.offered_concurrency) != int(right.offered_concurrency):
            raise FinalGateError(f"MFU load differs at context {context}")
        goodput = float(left.decode_goodput_tps)
        baseline_goodput = float(right.decode_goodput_tps)
        calls = float(left.target_calls_per_output_token)
        baseline_calls = float(right.target_calls_per_output_token)
        if not all(math.isfinite(item) and item >= 0 for item in (goodput, baseline_goodput, calls, baseline_calls)):
            raise FinalGateError("MFU engineering evidence is non-finite")
        rows.append(
            {
                "context_length": context,
                "offered_concurrency": int(left.offered_concurrency),
                "candidate_goodput_tps": goodput,
                "baseline_goodput_tps": baseline_goodput,
                "goodput_ratio": goodput / max(baseline_goodput, 1e-12),
                "candidate_target_calls_per_output_token": calls,
                "baseline_target_calls_per_output_token": baseline_calls,
                "goodput_non_decrease": goodput >= baseline_goodput,
                "target_calls_reduced": calls < baseline_calls,
            }
        )
    return {
        "baseline_role": "engineering_best",
        "baseline_stride": baseline_stride,
        "per_context_load": rows,
        "engineering_pass": all(
            row["goodput_non_decrease"] and row["target_calls_reduced"] for row in rows
        ),
        "claim_semantics": "context_load_goodput_and_target_calls_not_acceptance",
    }


def _verify_engineering_pairing(
    prompt: pd.DataFrame,
    *,
    method: str,
    stride: int,
    baseline_stride: int,
) -> None:
    """Prove identical MFU prompt/seed/load cells without pooling throughput."""

    identity_no_stride = [
        name for name in P5_IDENTITY_COLUMNS if name != "update_stride"
    ]
    keys = [*identity_no_stride, "context_length", "prompt_cluster", "seed"]
    candidate = prompt[
        (prompt.method == method) & (prompt.update_stride == stride)
    ]
    baseline = prompt[
        (prompt.method == "tts") & (prompt.update_stride == baseline_stride)
    ]
    if candidate.empty or baseline.empty or set(map(tuple, candidate[keys].to_numpy())) != set(
        map(tuple, baseline[keys].to_numpy())
    ):
        raise FinalGateError(f"MFU prompt/seed/load pairing differs for {method}")
    if (
        candidate["benchmark_repetitions"].min() < 5
        or baseline["benchmark_repetitions"].min() < 5
    ):
        raise FinalGateError("MFU evidence requires at least five repetitions")


def evaluate(
    *,
    headline_generation: Path,
    controller_terminal: Path,
    tts_foundation_terminal: Path,
    algorithmic_artifact_root: Path,
    algorithmic_manifest: Path,
    algorithmic_analysis_root: Path,
    mfu_artifact_root: Path,
    mfu_manifest: Path,
    mfu_analysis_root: Path,
    output: Path,
    bootstrap_replicates: int = 5000,
    require_all_engineering: bool = False,
) -> dict[str, Any]:
    generation, generation_evidence = _receipt_tree(headline_generation)
    controller, controller_evidence = _receipt_tree(controller_terminal)
    foundation, foundation_evidence = _receipt_tree(tts_foundation_terminal)
    receipt_snapshot = _snapshot(
        [*generation_evidence, *controller_evidence, *foundation_evidence]
    )
    if not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise FinalGateError("bootstrap_replicates must be positive")
    if (
        generation.get("schema_version") != 1
        or generation.get("status") != "final_0_40k_manifests_generated"
        or generation.get("scope") != "evidence_bound_context_specific_p5"
    ):
        raise FinalGateError("headline generation receipt identity is invalid")
    if (
        controller.get("schema_version") != 1
        or controller.get("scope") != "matched_dflash_l1_l2_l3_evidence"
        or controller.get("status")
        not in {"matched_controller_selected", "matched_controller_blocked"}
    ):
        raise FinalGateError("controller terminal identity is invalid")
    controller_flags = controller.get("eligible")
    controller_selected = controller.get("status") == "matched_controller_selected"
    if not isinstance(controller_flags, Mapping) or controller_selected != all(
        value is True for value in controller_flags.values()
    ):
        raise FinalGateError("controller terminal status/eligibility conflicts")
    conflicting_controller = controller_terminal.with_name(
        "CONTROLLER_BLOCKED.json"
        if controller_terminal.name == "CONTROLLER_SELECTED.json"
        else "CONTROLLER_SELECTED.json"
    )
    if conflicting_controller.exists() or Path(
        str(conflicting_controller) + ".sha256"
    ).exists():
        raise FinalGateError("conflicting controller terminals coexist")
    if (
        foundation.get("schema_version") != 2
        or foundation.get("scope") != "tts_0_40k_foundation"
        or foundation.get("status") != "TTS_0_40K_CONFIRMED"
    ):
        raise FinalGateError("TTS foundation terminal identity is invalid")
    if foundation.get("formal_acceptance_foundation_pass") is not True:
        raise FinalGateError("TTS foundation terminal did not pass its formal gate")
    foundation_identity = foundation.get("identity")
    if (
        not isinstance(foundation_identity, Mapping)
        or foundation.get("identity_sha256") != sha256_json(foundation_identity)
    ):
        raise FinalGateError("TTS foundation terminal identity hash is invalid")
    roles = _role_strides(foundation)
    methods = _controller_methods(controller, generation)

    direct_generation = _evidence_hashes(generation)
    for terminal in (controller_terminal.resolve(), tts_foundation_terminal.resolve()):
        if direct_generation.get(terminal) != sha256_file(terminal):
            raise FinalGateError("headline generation does not directly bind both terminals")

    algo_analysis, algo_evidence, algo_runs = _verify_analysis(
        root=algorithmic_analysis_root,
        artifact_root=algorithmic_artifact_root,
        manifest_path=algorithmic_manifest,
    )
    mfu_analysis, mfu_evidence, mfu_runs = _verify_analysis(
        root=mfu_analysis_root,
        artifact_root=mfu_artifact_root,
        manifest_path=mfu_manifest,
    )
    analysis_snapshot = _snapshot([*algo_evidence, *mfu_evidence])
    if algo_analysis.get("analysis", {}).get("baseline") != "tts":
        raise FinalGateError("algorithmic analysis must use TTS as its baseline")
    if mfu_analysis.get("analysis", {}).get("baseline") != "tts":
        raise FinalGateError("MFU analysis must use TTS as its baseline")
    artifacts = generation.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FinalGateError("headline generation lacks artifact manifest bindings")
    for role, path in (("ALGORITHMIC_C4", algorithmic_manifest), ("MFU_CONTEXT_LOAD", mfu_manifest)):
        row = artifacts.get(role)
        if not isinstance(row, Mapping) or Path(str(row.get("path", ""))).resolve() != path.resolve() or row.get("sha256") != sha256_file(path):
            raise FinalGateError(f"headline generation {role} manifest binding differs")

    raw_prompt = _raw_prompt_table(algo_runs)
    _verify_prompt_derivation(raw_prompt, algorithmic_analysis_root)
    mfu_prompt = _raw_prompt_table(mfu_runs)
    _verify_prompt_derivation(mfu_prompt, mfu_analysis_root)
    if set(raw_prompt.context_length.astype(int)) != set(CONTEXTS):
        raise FinalGateError("algorithmic raw evidence does not cover exactly 0--40K")
    optimizer = generation.get("optimizer_identity")
    if not isinstance(optimizer, Mapping):
        raise FinalGateError("headline generation lacks optimizer identity")
    adaptation_stride = optimizer.get("adaptation_stride")
    if isinstance(adaptation_stride, bool) or not isinstance(adaptation_stride, int):
        raise FinalGateError("headline generation adaptation stride is invalid")
    if roles["same_stride"] != adaptation_stride:
        raise FinalGateError("TTS same-stride role differs from LC adaptation stride")

    safety_methods = ["tts", *methods]
    algorithmic_safety = _safety(algorithmic_analysis_root, safety_methods)
    mfu_safety = _safety(mfu_analysis_root, safety_methods)
    safety = {
        method: {
            "algorithmic": algorithmic_safety[method],
            "mfu": mfu_safety[method],
            "pass": bool(
                algorithmic_safety[method]["pass"] and mfu_safety[method]["pass"]
            ),
        }
        for method in safety_methods
    }
    _, tts_foundation_comparison = _comparison(
        raw_prompt,
        method="tts",
        stride=roles["acceptance_best"],
        baseline_method="static",
        baseline_stride=roles["acceptance_best"],
        b=bootstrap_replicates,
    )
    tts_foundation_comparison["safety_pass"] = safety["tts"]["pass"]
    tts_foundation_comparison["algorithmic_pass"] = bool(
        tts_foundation_comparison["algorithmic_pass"]
        and safety["tts"]["pass"]
    )
    comparisons = {}
    engineering_role_acceptance = {}
    same_stride = {}
    engineering = {}
    for method in methods:
        same_stride[method] = _standard_same_stride_gate(
            analysis_root=algorithmic_analysis_root,
            method=method,
            stride=adaptation_stride,
        )
        _, comparisons[method] = _comparison(
            raw_prompt,
            method=method,
            stride=adaptation_stride,
            baseline_method="tts",
            baseline_stride=roles["acceptance_best"],
            b=bootstrap_replicates,
        )
        comparisons[method]["same_stride_standard_pass"] = bool(
            same_stride[method].get("algorithmic_pass") is True
        )
        comparisons[method]["safety_pass"] = safety[method]["pass"]
        comparisons[method]["algorithmic_pass"] = bool(
            comparisons[method]["algorithmic_pass"]
            and comparisons[method]["same_stride_standard_pass"]
            and safety[method]["pass"]
        )
        _, engineering_role_acceptance[method] = _comparison(
            raw_prompt,
            method=method,
            stride=adaptation_stride,
            baseline_method="tts",
            baseline_stride=roles["engineering_best"],
            b=bootstrap_replicates,
        )
        engineering_role_acceptance[method]["safety_pass"] = safety[method][
            "pass"
        ]
        engineering_role_acceptance[method]["algorithmic_pass"] = bool(
            engineering_role_acceptance[method]["algorithmic_pass"]
            and safety[method]["pass"]
        )
        _verify_engineering_pairing(
            mfu_prompt,
            method=method,
            stride=adaptation_stride,
            baseline_stride=roles["engineering_best"],
        )
        engineering[method] = _engineering(
            analysis_root=mfu_analysis_root,
            method=method,
            stride=adaptation_stride,
            baseline_stride=roles["engineering_best"],
        )

    eligible_algorithmic_all = bool(
        tts_foundation_comparison["algorithmic_pass"]
        and all(row["algorithmic_pass"] for row in comparisons.values())
    )
    eligible_engineering_all = all(
        row["engineering_pass"] for row in engineering.values()
    )
    full_l0123_coverage = set(methods) == set(LC_METHODS)
    # A final 0--40K claim is about all four LightCone levels, not merely the
    # controller methods that happened to clear their upstream enable gate.
    # Keep the eligible-only diagnostic, but never let an empty/partial
    # ``comparisons`` mapping make the terminal or old-ablation resume gate
    # pass vacuously.
    algorithmic_all = bool(
        tts_foundation_comparison["algorithmic_pass"]
        and _all_lc_methods_pass(methods, comparisons, "algorithmic_pass")
    )
    engineering_all = _all_lc_methods_pass(
        methods, engineering, "engineering_pass"
    )
    terminal_confirmed = algorithmic_all and (
        engineering_all if require_all_engineering else True
    )
    decision = {
        "schema_version": 1,
        "status": "CONFIRMED" if terminal_confirmed else "BLOCKED",
        "scope": "final_p5_0_40k_evidence_gate",
        "algorithmic_status": "CONFIRMED" if algorithmic_all else "BLOCKED",
        "engineering_status": "CONFIRMED" if engineering_all else "BLOCKED",
        "all_algorithmic_pass": algorithmic_all,
        "all_eligible_algorithmic_pass": eligible_algorithmic_all,
        "all_engineering_pass": engineering_all,
        "all_eligible_engineering_pass": eligible_engineering_all,
        "all_declared_l0123_evaluated": full_l0123_coverage,
        "require_all_engineering": require_all_engineering,
        "resume_old_ablations_allowed": algorithmic_all,
        "eligible_lc_methods": methods,
        "missing_controller_methods_are_not_passes": True,
        "controller_method_status": {
            method: (
                "EVALUATED" if method in methods else "NOT_ELIGIBLE_NOT_PASS"
            )
            for method in LC_METHODS
        },
        "context_contract": list(CONTEXTS),
        "tts_role_strides": roles,
        "tts_foundation_comparison_vs_static": tts_foundation_comparison,
        "algorithmic_comparisons_vs_acceptance_best_tts": comparisons,
        "diagnostic_acceptance_comparisons_vs_engineering_best_tts": (
            engineering_role_acceptance
        ),
        "same_stride_standard_p5_claim_gates": same_stride,
        "engineering_comparisons_vs_engineering_best_tts": engineering,
        "safety": safety,
        "analysis_bindings": {
            "algorithmic_analysis_manifest_sha256": sha256_file(
                algorithmic_analysis_root / "analysis-manifest.json"
            ),
            "mfu_analysis_manifest_sha256": sha256_file(
                mfu_analysis_root / "analysis-manifest.json"
            ),
            "algorithmic_manifest_sha256": sha256_file(algorithmic_manifest),
            "mfu_manifest_sha256": sha256_file(mfu_manifest),
            "raw_prompt_table_sha256": sha256_json(
                json.loads(raw_prompt.to_json(orient="records", double_precision=15))
            ),
            "algorithmic_analysis_identity": algo_analysis.get("analysis"),
        },
    }
    decision["decision_sha256"] = sha256_json(decision)
    evidence = sorted(
        {
            *generation_evidence,
            *controller_evidence,
            *foundation_evidence,
            *algo_evidence,
            *mfu_evidence,
            Path(__file__).resolve(),
        }
    )
    body = {
        **decision,
        "evidence": [
            {"path": str(path), "sha256": sha256_file(path)} for path in evidence
        ],
    }
    text = json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _verify_snapshot(receipt_snapshot)
    _verify_snapshot(analysis_snapshot)
    if output.is_file() and output.read_text(encoding="utf-8") != text:
        raise FinalGateError(f"final gate receipt already exists with different content: {output}")
    _atomic_text(output, text)
    _atomic_text(Path(str(output) + ".sha256"), sha256_file(output) + "\n")
    return body


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline-generation", type=Path, required=True)
    parser.add_argument("--controller-terminal", type=Path, required=True)
    parser.add_argument("--tts-foundation-terminal", type=Path, required=True)
    parser.add_argument("--algorithmic-artifact-root", type=Path, required=True)
    parser.add_argument("--algorithmic-manifest", type=Path, required=True)
    parser.add_argument("--algorithmic-analysis-root", type=Path, required=True)
    parser.add_argument("--mfu-artifact-root", type=Path, required=True)
    parser.add_argument("--mfu-manifest", type=Path, required=True)
    parser.add_argument("--mfu-analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--require-all-engineering", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(os.sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "verify-resume":
        parser = argparse.ArgumentParser(
            description="Verify a final receipt before resuming old ablations."
        )
        parser.add_argument("verify-resume")
        parser.add_argument("--receipt", type=Path, required=True)
        args = parser.parse_args(argv)
        try:
            verify_resume_receipt(args.receipt)
        except FinalGateError as exc:
            print(f"BLOCKED: {exc}", file=os.sys.stderr)
            return 2
        print("RESUME_OLD_ABLATIONS_ALLOWED")
        return 0
    args = build_parser().parse_args(argv)
    try:
        result = evaluate(**vars(args))
    except FinalGateError as exc:
        print(f"BLOCKED: {exc}", file=os.sys.stderr)
        return 2
    print(result["status"])
    return 0 if result["status"] == "CONFIRMED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
