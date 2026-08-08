from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from lightcone_spec.locking.hashing import sha256_file, sha256_json
from lightcone_spec.orchestration.catalog import (
    P5_PRIORITY_FINAL_CONTEXTS,
    p5_priority_dflash_stride_screen_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiments/p5_tts_foundation.py"
MATCHED_FIXTURE = (
    ROOT / "tests/test_build_p5_matched_controller_manifests.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("p5_tts_foundation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _matched_fixture_module():
    spec = importlib.util.spec_from_file_location(
        "matched_fixture_for_tts_foundation", MATCHED_FIXTURE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _context(module, *, acceptance=4, engineering=8, same=4):
    source = p5_priority_dflash_stride_screen_manifest()
    tts = {(unit.stride): unit for unit in source.units if unit.method == "tts"}
    roles = {
        "tts_acceptance_best": {
            "method": "tts",
            "stride": acceptance,
            "source_unit_id": tts[acceptance].unit_id,
        },
        "tts_engineering_best": {
            "method": "tts",
            "stride": engineering,
            "source_unit_id": tts[engineering].unit_id,
        },
        "same_stride_tts_for_l0": {
            "method": "tts",
            "stride": same,
            "source_unit_id": tts[same].unit_id,
        },
    }
    runtime_body = {
        "schema_version": 1,
        "files": {},
        "locked_reference": {},
    }
    runtime = {**runtime_body, "sha256": sha256_json(runtime_body)}
    return {
        "source": source,
        "roles": roles,
        "bindings": {
            "schema_version": 2,
            "selection_sha256": "1" * 64,
            "selected_terminal_sha256": "2" * 64,
            "source_manifest_file_sha256": "3" * 64,
            "source_manifest_sha256": source.content_sha256(),
            "lockfile_sha256": "4" * 64,
            "model_roots_sha256": "5" * 64,
            "model_revisions": {
                "target": "6" * 40,
                "drafter": "7" * 40,
                "tokenizer": "6" * 40,
            },
            "runtime_implementation_fingerprint": runtime,
            "role_source_unit_ids": {
                role: row["source_unit_id"] for role, row in roles.items()
            },
        },
    }


def test_foundation_manifest_is_exact_safe_and_deduplicates_role_strides():
    module = _module()
    manifest, roles = module._manifest(
        _context(module, acceptance=4, engineering=8, same=4)
    )

    assert [unit.method for unit in manifest.units] == ["static", "tts", "tts"]
    assert [unit.stride for unit in manifest.units] == [1, 4, 8]
    assert len({unit.unit_id for unit in manifest.units}) == 3
    assert roles["tts_acceptance_best"]["foundation_unit_id"] == roles[
        "same_stride_tts_for_l0"
    ]["foundation_unit_id"]
    assert manifest.engine_params["p5_context_lengths"] == list(
        P5_PRIORITY_FINAL_CONTEXTS
    )
    assert manifest.engine_params["p5_context_timing_contract"] == (
        "independent_exact_context_group_v1"
    )
    assert manifest.engine_params["max_running_requests"] == 4
    assert manifest.engine_params["prompt_limit"] == 48
    assert manifest.engine_params["prompt_offset"] == 40
    assert manifest.engine_params["benchmark_repetitions"] == 5
    assert manifest.engine_params["optimizer"] == "adamw"
    assert manifest.engine_params["lr"] == pytest.approx(1e-4)
    assert manifest.engine_params["weight_decay"] == pytest.approx(1e-2)
    assert 40000 + manifest.engine_params["max_new_tokens"] <= 40960
    assert all(unit.trainable_scope == "tail_lora" for unit in manifest.units)


def test_build_consumes_recursive_terminal_and_writes_hash_bound_manifest(tmp_path):
    module = _module()
    chain = _matched_fixture_module()._chain(tmp_path)
    runtime_body = {
        "schema_version": 1,
        "files": {},
        "locked_reference": {},
    }
    runtime = {**runtime_body, "sha256": sha256_json(runtime_body)}
    runtime_path = tmp_path / "runtime-fingerprint.json"
    runtime_path.write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(str(runtime_path) + ".sha256").write_text(
        sha256_file(runtime_path) + "\n", encoding="utf-8"
    )
    manifest_path = tmp_path / "foundation.json"
    receipt_path = tmp_path / "foundation-generation.json"
    receipt = module.build(
        selected_or_terminal=chain["terminal"],
        source_manifest_path=chain["source"],
        lockfile_path=chain["lockfile"],
        model_roots_path=chain["model_roots"],
        runtime_fingerprint_path=runtime_path,
        artifact_root=tmp_path / "foundation-runs",
        output_manifest_path=manifest_path,
        output_receipt_path=receipt_path,
    )
    assert receipt["schema_version"] == 2
    assert receipt["status"] == "ready_for_execution"
    assert receipt["identity_sha256"] == sha256_json(receipt["identity"])
    assert receipt["identity"]["bindings"]["selection_sha256"] == sha256_file(
        chain["selection"]
    )
    assert len(receipt["identity"]["contexts"]) == 8
    assert receipt["identity"]["prompt_windows"] == {
        "selection": {"offset": 0, "limit": 40, "half_open": [0, 40]},
        "foundation": {"offset": 40, "limit": 48, "half_open": [40, 88]},
    }
    assert Path(str(receipt_path) + ".sha256").read_text().strip() == sha256_file(
        receipt_path
    )


def _curve(stride: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "tts",
                "update_stride": stride,
                "offered_concurrency": 4,
                "context_length": context,
                "version_mismatch_count": 0,
                "exactness_violations": 0,
                "adaptation_fallback_count": 0,
                "lcag": 0.1,
                "lcag_ci_low": 0.01,
                "lcag_ci_high": 0.2,
                "lcag_prompt_clusters": 2,
                "benchmark_repetitions": 5,
                "survival_weighted_accepted_prefix": 2.0,
                "acceptance_gain_vs_baseline": 0.1,
                "acceptance_gain_ci_low": -0.01,
                "acceptance_gain_ci_high": 0.2,
                "throughput_speedup_vs_baseline": 0.9,
            }
            for context in P5_PRIORITY_FINAL_CONTEXTS
        ]
    )


def _gate(*, stride=4, algorithmic=True, engineering=False, clusters=2, reps=5):
    return {
        "method": "tts",
        "baseline_method": "static",
        "weight_update_mode": "lora",
        "update_stride": stride,
        "offered_concurrency": 4,
        "algorithmic_pass": algorithmic,
        "engineering_pass": engineering,
        "exactness_pass": True,
        "lcag_ci_low": 0.01,
        "mean_delta_acceptance_elasticity": -0.02,
        "paired_prompt_clusters": clusters,
        "benchmark_repetitions": reps,
        # Formal foundation intentionally does not require every bucket to win.
        "window_dominance_pass": False,
    }


def test_acceptance_foundation_uses_lcag_elasticity_exactness_not_bucket_dominance():
    module = _module()
    definition = _context(module)["roles"]["tts_acceptance_best"]
    result = module._role_result(
        role="tts_acceptance_best",
        definition=definition,
        gates=[_gate()],
        curve=_curve(4),
    )
    assert result["gate"]["algorithmic_pass"] is True
    assert result["gate"]["engineering_pass"] is False
    assert result["gate"]["window_dominance_pass"] is False
    assert len(result["window_curve"]) == 8


@pytest.mark.parametrize(
    "gate_change",
    [
        {"lcag_ci_low": 0.0},
        {"mean_delta_acceptance_elasticity": 0.0},
        {"exactness_pass": False},
        {"paired_prompt_clusters": 1},
        {"benchmark_repetitions": 4},
    ],
)
def test_acceptance_foundation_fails_closed_on_each_scientific_gate(gate_change):
    module = _module()
    gate = _gate()
    gate.update(gate_change)
    curve = _curve(4)
    if "lcag_ci_low" in gate_change:
        curve["lcag_ci_low"] = gate_change["lcag_ci_low"]
    if "paired_prompt_clusters" in gate_change:
        curve["lcag_prompt_clusters"] = gate_change["paired_prompt_clusters"]
    if "benchmark_repetitions" in gate_change:
        curve["benchmark_repetitions"] = gate_change["benchmark_repetitions"]
    definition = _context(module)["roles"]["tts_acceptance_best"]
    result = module._role_result(
        role="tts_acceptance_best",
        definition=definition,
        gates=[gate],
        curve=curve,
    )
    assert result["gate"]["algorithmic_pass"] is False


def test_claim_gate_lookup_is_exact_on_method_stride_and_concurrency():
    module = _module()
    rows = [
        _gate(stride=4),
        {**_gate(stride=8), "offered_concurrency": 1},
        {**_gate(stride=8), "method": "naive_async"},
    ]
    assert module._one_gate(rows, stride=4)["update_stride"] == 4
    with pytest.raises(module.FoundationError, match="found 0 rows"):
        module._one_gate(rows, stride=8)


def test_curve_identity_requires_static_and_every_unique_role_stride_at_all_buckets():
    module = _module()
    manifest, _ = module._manifest(
        _context(module, acceptance=4, engineering=8, same=4)
    )
    curve = pd.concat(
        [
            _curve(4),
            _curve(8),
            _curve(1).assign(method="static"),
        ],
        ignore_index=True,
    )
    module._validate_curve_identity(curve, manifest)
    with pytest.raises(module.FoundationError, match="eight-bucket coverage"):
        module._validate_curve_identity(
            curve[~((curve.method == "tts") & (curve.context_length == 40000))],
            manifest,
        )


def test_legacy_selection_is_rejected_instead_of_silently_migrated():
    module = _module()
    source = p5_priority_dflash_stride_screen_manifest()
    with pytest.raises(module.FoundationError, match="schema-v2"):
        module._selected_roles(
            {
                "schema_version": 1,
                "winners": {"tts_best": {}, "l0_best": {}},
            },
            source,
        )
