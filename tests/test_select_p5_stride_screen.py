from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from lightcone_spec.orchestration.manifest import ExperimentManifest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiments/select_p5_stride_screen.py"
SOURCE_MANIFEST = (
    ROOT
    / "manifests/p5/p5_priority_dflash_stride_screen_v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("select_p5_stride_screen", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_hashed(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    Path(str(path) + ".sha256").write_text(_sha(path) + "\n", encoding="utf-8")


def _manifest(tmp_path: Path, *, required_safety=()) -> Path:
    source = ExperimentManifest.load(SOURCE_MANIFEST)
    engine = dict(source.engine_params)
    if required_safety:
        engine["p5_stride_screen_required_safety_columns"] = list(required_safety)
    manifest = dataclasses.replace(source, engine_params=engine)
    path = tmp_path / "screen.json"
    manifest.write(path)
    return path


def _coverage(manifest_path: Path, tmp_path: Path, statuses=None) -> Path:
    manifest = ExperimentManifest.load(manifest_path)
    statuses = statuses or {}
    cells = {
        unit.unit_id: {
            "unit_id": unit.unit_id,
            "method": unit.method,
            "stride": unit.stride,
            "status": statuses.get(unit.unit_id, "complete_valid"),
        }
        for unit in manifest.units
    }
    path = tmp_path / "coverage.json"
    _write_hashed(
        path,
        json.dumps({"cells": cells, "summary": {}}, sort_keys=True),
    )
    return path


TTS = {
    1: (0.80, 0.200, 130.0),
    4: (1.00, 0.180, 110.0),
    8: (0.99, 0.170, 115.0),
    16: (0.99, 0.171, 125.0),
}
L0 = {
    1: (0.90, 0.190, 120.0),
    4: (1.10, 0.160, 130.0),
    8: (1.08, 0.155, 140.0),
    16: (1.07, 0.150, 150.0),
}


def _frame(*, optional_safety=False) -> pd.DataFrame:
    rows = []
    for context in (4096, 16384):
        static_a = 3.0 if context == 16384 else 4.0
        rows.append(
            {
                "method": "static",
                "update_stride": 1,
                "context_length": context,
                "survival_weighted_accepted_prefix": static_a,
                "acceptance_gain_vs_baseline": 0.0,
                "target_calls_per_output_token": 0.25,
                "decode_goodput_tps": 100.0,
                "exactness_violations": 0,
                "version_mismatch_count": 0,
                "adaptation_fallback_count": 0,
            }
        )
        for method, values in (("tts", TTS), ("naive_async", L0)):
            for stride, (gain, target_calls, goodput) in values.items():
                row = {
                    "method": method,
                    "update_stride": stride,
                    "context_length": context,
                    "survival_weighted_accepted_prefix": static_a + gain,
                    "acceptance_gain_vs_baseline": gain,
                    "target_calls_per_output_token": target_calls,
                    "decode_goodput_tps": goodput,
                    "exactness_violations": 0,
                    "version_mismatch_count": 0,
                    "adaptation_fallback_count": 0,
                }
                if optional_safety:
                    row.update(
                        nonfinite_update_count=0,
                    )
                rows.append(row)
    frame = pd.DataFrame(rows)
    if optional_safety:
        frame[["nonfinite_update_count"]] = frame[
            ["nonfinite_update_count"]
        ].fillna(0)
    return frame


def _analysis(
    root: Path,
    *,
    baseline: str,
    manifest_path: Path,
    frame: pd.DataFrame,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = ExperimentManifest.load(manifest_path)
    if baseline == "tts":
        frame = frame.copy()
        for context in (4096, 16384):
            for stride in (1, 4, 8, 16):
                tts_a = float(
                    frame[
                        (frame.method == "tts")
                        & (frame.update_stride == stride)
                        & (frame.context_length == context)
                    ].iloc[0].survival_weighted_accepted_prefix
                )
                mask = (
                    (frame.method == "naive_async")
                    & (frame.update_stride == stride)
                    & (frame.context_length == context)
                )
                frame.loc[mask, "acceptance_gain_vs_baseline"] = (
                    frame.loc[mask, "survival_weighted_accepted_prefix"] - tts_a
                )
    table = root / "p5_long_context_acceptance.parquet"
    frame.to_parquet(table, index=False)
    analysis_manifest = root / "analysis-manifest.json"
    analysis_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "analysis": {
                    "baseline": baseline,
                    "expected_manifest_sha256": manifest.content_sha256(),
                },
                "input_runs": [
                    {"run_id": unit.unit_id, "unit_id": unit.unit_id}
                    for unit in manifest.units
                ],
                "derived_outputs": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    sidecar = root / "analysis-manifest.sha256"
    sidecar.write_text(_sha(analysis_manifest) + "\n", encoding="utf-8")
    ledger = {
        path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in (table, analysis_manifest, sidecar)
    }
    (root / "analysis-hashes.json").write_text(
        json.dumps(ledger, sort_keys=True), encoding="utf-8"
    )
    return root


def _inputs(tmp_path: Path, *, required_safety=(), optional_safety=False):
    manifest = _manifest(tmp_path, required_safety=required_safety)
    coverage = _coverage(manifest, tmp_path)
    frame = _frame(optional_safety=optional_safety)
    static = _analysis(
        tmp_path / "vs-static",
        baseline="static",
        manifest_path=manifest,
        frame=frame,
    )
    tts = _analysis(
        tmp_path / "vs-tts",
        baseline="tts",
        manifest_path=manifest,
        frame=frame,
    )
    return manifest, coverage, static, tts


def test_selects_with_two_percent_ties_and_preserves_same_stride_tts(tmp_path):
    module = _module()
    manifest, coverage, static, tts = _inputs(tmp_path)
    source_table = pd.read_parquet(
        static / "p5_long_context_acceptance.parquet"
    )
    assert "update_stride" in source_table
    assert "stride" not in source_table
    output = tmp_path / "selection.json"
    payload = module.select(
        manifest_path=manifest,
        coverage_path=coverage,
        vs_static_root=static,
        vs_tts_root=tts,
        output_path=output,
    )

    assert payload["status"] == "winner_selected"
    assert payload["objective_screen_pass"] is True
    assert payload["schema_version"] == 2
    assert "tts_best" not in payload["winners"]
    assert payload["winners"]["tts_acceptance_best"]["stride"] == 16
    assert payload["winners"]["tts_engineering_best"]["stride"] == 1
    assert payload["winners"]["l0_best"]["stride"] == 8
    assert payload["winners"]["same_stride_tts_for_l0"]["stride"] == 8
    assert payload["comparisons"]["l0_best_vs_tts_acceptance_best"]["16384"][
        "acceptance_gain"
    ] == pytest.approx(0.09)
    assert payload["performance_evidence"] == {
        "scope": "mixed_context_workload_global",
        "source_marker": "legacy_unmarked_v7",
        "context_specific_goodput_available": False,
        "allowed_use": "candidate_workload_tiebreak_only",
        "engineering_pass_evaluated": False,
    }
    winner = payload["winners"]["l0_best"]
    assert winner["metrics_by_context"]["4096"]["decode_goodput_tps"] is None
    assert winner["metrics_by_context"]["16384"]["goodput_ratio_vs_static"] is None
    assert winner["workload_performance"]["decode_goodput_tps"] == 140.0
    assert payload["comparisons"]["l0_best_vs_tts_acceptance_best"]["16384"][
        "goodput_ratio"
    ] is None
    assert payload["comparisons"]["l0_best_vs_tts_acceptance_best"][
        "workload_performance"
    ]["goodput_ratio"] == pytest.approx(140.0 / 125.0)
    assert output.is_file()
    assert Path(str(output) + ".sha256").read_text().strip() == _sha(output)
    selector_evidence = next(
        row for row in payload["evidence"] if row["path"] == str(SCRIPT.resolve())
    )
    assert selector_evidence["sha256"] == _sha(SCRIPT)
    assert module.validate_selection_receipt(
        selector_path=output, manifest_path=manifest
    ) == payload


def test_explicit_context_resolved_scope_exposes_per_context_goodput(tmp_path):
    module = _module()
    manifest, coverage, static, tts = _inputs(tmp_path)
    frame = pd.read_parquet(static / "p5_long_context_acceptance.parquet")
    frame["performance_scope"] = "checkpoint_request"
    frame.loc[frame.context_length == 4096, "decode_goodput_tps"] *= 0.5
    _analysis(static, baseline="static", manifest_path=manifest, frame=frame)
    _analysis(tts, baseline="tts", manifest_path=manifest, frame=frame)

    payload = module.select(
        manifest_path=manifest,
        coverage_path=coverage,
        vs_static_root=static,
        vs_tts_root=tts,
        output_path=tmp_path / "selection.json",
    )

    assert payload["performance_evidence"]["scope"] == "context_resolved"
    assert payload["performance_evidence"][
        "context_specific_goodput_available"
    ] is True
    winner = payload["winners"]["l0_best"]
    assert winner["workload_performance"] is None
    assert winner["metrics_by_context"]["4096"]["decode_goodput_tps"] == 70.0
    assert winner["metrics_by_context"]["16384"]["decode_goodput_tps"] == 140.0
    assert payload["comparisons"]["l0_best_vs_tts_acceptance_best"]["4096"][
        "goodput_ratio"
    ] == pytest.approx(140.0 / 125.0)


def test_engineering_winner_requires_nonnegative_both_contexts(tmp_path):
    module = _module()
    manifest, coverage, static, tts = _inputs(tmp_path)
    frame = pd.read_parquet(static / "p5_long_context_acceptance.parquet")
    frame["performance_scope"] = "checkpoint_request"
    mask = (
        (frame.method == "tts")
        & (frame.update_stride == 1)
        & (frame.context_length == 4096)
    )
    frame.loc[mask, "survival_weighted_accepted_prefix"] = 3.9
    frame.loc[mask, "target_calls_per_output_token"] = 0.26
    _analysis(static, baseline="static", manifest_path=manifest, frame=frame)
    _analysis(tts, baseline="tts", manifest_path=manifest, frame=frame)

    payload = module.select(
        manifest_path=manifest,
        coverage_path=coverage,
        vs_static_root=static,
        vs_tts_root=tts,
        output_path=tmp_path / "selection.json",
    )

    rejected = next(
        row for row in payload["candidates"]["tts"] if row["stride"] == 1
    )
    assert rejected["eligible"] is True
    assert rejected["engineering_eligible"] is False
    assert set(rejected["engineering_rejection_reasons"]) == {
        "context_4096:negative_acceptance_gain",
        "context_4096:negative_target_call_reduction",
    }
    assert payload["winners"]["tts_acceptance_best"]["stride"] == 16
    assert payload["winners"]["tts_engineering_best"]["stride"] == 16


def test_legacy_v1_selection_migrates_in_memory_without_dual_truth(tmp_path):
    module = _module()
    manifest, coverage, static, tts = _inputs(tmp_path)
    current = module.select(
        manifest_path=manifest,
        coverage_path=coverage,
        vs_static_root=static,
        vs_tts_root=tts,
        output_path=tmp_path / "selection.json",
    )
    legacy = copy.deepcopy(current)
    legacy["schema_version"] = 1
    legacy["selection_rule"]["id"] = "p5_stride_screen_selection_v1"
    legacy["selection_rule"].pop("engineering_order")
    legacy["winners"] = {
        "tts_best": legacy["winners"]["tts_acceptance_best"],
        "l0_best": legacy["winners"]["l0_best"],
        "same_stride_tts_for_l0": legacy["winners"][
            "same_stride_tts_for_l0"
        ],
    }
    legacy["comparisons"]["l0_best_vs_tts_best"] = legacy["comparisons"].pop(
        "l0_best_vs_tts_acceptance_best"
    )

    migrated = module.canonicalize_selection(legacy)

    assert migrated["schema_version"] == 2
    assert set(migrated["winners"]) == {
        "tts_acceptance_best",
        "tts_engineering_best",
        "l0_best",
        "same_stride_tts_for_l0",
    }
    assert "tts_best" not in migrated["winners"]


def test_missing_engineering_candidate_falls_back_without_blocking_objective(
    tmp_path,
):
    module = _module()
    manifest, coverage, static, tts = _inputs(tmp_path)
    frame = pd.read_parquet(static / "p5_long_context_acceptance.parquet")
    for stride in (1, 4, 8, 16):
        mask = (
            (frame.method == "tts")
            & (frame.update_stride == stride)
            & (frame.context_length == 4096)
        )
        frame.loc[mask, "target_calls_per_output_token"] = 0.26
    _analysis(static, baseline="static", manifest_path=manifest, frame=frame)
    _analysis(tts, baseline="tts", manifest_path=manifest, frame=frame)

    payload = module.select(
        manifest_path=manifest,
        coverage_path=coverage,
        vs_static_root=static,
        vs_tts_root=tts,
        output_path=tmp_path / "selection.json",
    )

    assert payload["objective_screen_pass"] is True
    assert payload["status"] == "winner_selected"
    acceptance = payload["winners"]["tts_acceptance_best"]
    engineering = payload["winners"]["tts_engineering_best"]
    assert engineering["unit_id"] == acceptance["unit_id"]
    assert engineering["engineering_eligible"] is False
    assert engineering["engineering_fallback_reason"] == (
        "no_eligible_engineering_candidate_used_tts_acceptance_best"
    )


def test_unmarked_global_goodput_must_be_one_attestable_workload_value(tmp_path):
    module = _module()
    manifest, coverage, static, tts = _inputs(tmp_path)
    frame = pd.read_parquet(static / "p5_long_context_acceptance.parquet")
    frame.loc[
        (frame.method == "naive_async")
        & (frame.update_stride == 8)
        & (frame.context_length == 4096),
        "decode_goodput_tps",
    ] = 139.0
    _analysis(static, baseline="static", manifest_path=manifest, frame=frame)
    _analysis(tts, baseline="tts", manifest_path=manifest, frame=frame)

    with pytest.raises(module.SelectionError, match="no single workload value"):
        module.select(
            manifest_path=manifest,
            coverage_path=coverage,
            vs_static_root=static,
            vs_tts_root=tts,
            output_path=tmp_path / "selection.json",
        )


def test_goodput_does_not_enter_algorithmic_objective_or_safety(tmp_path):
    module = _module()
    manifest, coverage, static, tts = _inputs(tmp_path)
    frame = pd.read_parquet(static / "p5_long_context_acceptance.parquet")
    frame["decode_goodput_tps"] = 0.0
    _analysis(static, baseline="static", manifest_path=manifest, frame=frame)
    _analysis(tts, baseline="tts", manifest_path=manifest, frame=frame)

    payload = module.select(
        manifest_path=manifest,
        coverage_path=coverage,
        vs_static_root=static,
        vs_tts_root=tts,
        output_path=tmp_path / "selection.json",
    )

    assert payload["objective_screen_pass"] is True
    assert payload["status"] == "winner_selected"
    assert payload["winners"]["tts_acceptance_best"]["eligible"] is True
    assert payload["winners"]["l0_best"]["eligible"] is True


def test_unknown_performance_scope_fails_closed(tmp_path):
    module = _module()
    manifest, coverage, static, tts = _inputs(tmp_path)
    frame = pd.read_parquet(static / "p5_long_context_acceptance.parquet")
    frame["performance_scope"] = "trust_me_contextish"
    _analysis(static, baseline="static", manifest_path=manifest, frame=frame)
    _analysis(tts, baseline="tts", manifest_path=manifest, frame=frame)

    with pytest.raises(module.SelectionError, match="unsupported performance_scope"):
        module.select(
            manifest_path=manifest,
            coverage_path=coverage,
            vs_static_root=static,
            vs_tts_root=tts,
            output_path=tmp_path / "selection.json",
        )


def test_optional_safety_is_enforced_when_present(tmp_path):
    module = _module()
    manifest, coverage, static, tts = _inputs(tmp_path, optional_safety=True)
    frame = pd.read_parquet(static / "p5_long_context_acceptance.parquet")
    frame.loc[
        (frame.method == "tts") & (frame.update_stride == 16),
        "adaptation_fallback_count",
    ] = 1
    _analysis(
        static,
        baseline="static",
        manifest_path=manifest,
        frame=frame,
    )
    _analysis(
        tts,
        baseline="tts",
        manifest_path=manifest,
        frame=frame,
    )
    payload = module.select(
        manifest_path=manifest,
        coverage_path=coverage,
        vs_static_root=static,
        vs_tts_root=tts,
        output_path=tmp_path / "selection.json",
    )
    rejected = next(
        row for row in payload["candidates"]["tts"] if row["stride"] == 16
    )
    assert rejected["eligible"] is False
    assert any("adaptation_fallback_count" in reason for reason in rejected["rejection_reasons"])
    assert payload["winners"]["tts_acceptance_best"]["stride"] == 8


def test_required_missing_safety_blocks_all_candidates(tmp_path):
    module = _module()
    manifest, coverage, static, tts = _inputs(
        tmp_path,
        required_safety=("nonfinite_update_count",),
    )
    payload = module.select(
        manifest_path=manifest,
        coverage_path=coverage,
        vs_static_root=static,
        vs_tts_root=tts,
        output_path=tmp_path / "selection.json",
    )
    assert payload["status"] == "scientifically_blocked"
    assert payload["winners"]["tts_acceptance_best"] is None
    assert payload["winners"]["tts_engineering_best"] is None
    assert payload["winners"]["l0_best"] is None


def test_missing_stride_context_row_fails_closed(tmp_path):
    module = _module()
    manifest, coverage, static, tts = _inputs(tmp_path)
    frame = pd.read_parquet(static / "p5_long_context_acceptance.parquet")
    frame = frame[
        ~(
            (frame.method == "tts")
            & (frame.update_stride == 4)
            & (frame.context_length == 16384)
        )
    ]
    _analysis(
        static,
        baseline="static",
        manifest_path=manifest,
        frame=frame,
    )
    with pytest.raises(module.SelectionError, match="expected one"):
        module.select(
            manifest_path=manifest,
            coverage_path=coverage,
            vs_static_root=static,
            vs_tts_root=tts,
            output_path=tmp_path / "selection.json",
        )
