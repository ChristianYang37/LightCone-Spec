from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lightcone_spec.locking.hashing import sha256_file


SCRIPT = Path(__file__).parents[1] / "scripts/experiments/p5_final_headline_gate.py"
SPEC = importlib.util.spec_from_file_location("p5_final_headline_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _attest(path: Path) -> Path:
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(sha256_file(path) + "\n", encoding="utf-8")
    return sidecar


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _receipt(path: Path, payload: dict, evidence: list[Path]) -> Path:
    body = {
        **payload,
        "evidence": [
            {"path": str(item.resolve()), "sha256": sha256_file(item)}
            for item in evidence
        ],
    }
    _json(path, body)
    _attest(path)
    return path


def _resume_receipt(tmp_path: Path, **overrides) -> Path:
    evidence = _json(tmp_path / "bound-input.json", {"kind": "input"})
    decision = {
        "schema_version": 1,
        "status": "CONFIRMED",
        "scope": "final_p5_0_40k_evidence_gate",
        "resume_old_ablations_allowed": True,
        "all_declared_l0123_evaluated": True,
        "all_algorithmic_pass": True,
        "eligible_lc_methods": list(gate.LC_METHODS),
        "controller_method_status": {
            method: "EVALUATED" for method in gate.LC_METHODS
        },
        **overrides,
    }
    decision["decision_sha256"] = gate.sha256_json(decision)
    return _receipt(tmp_path / "FINAL_0_40K_CONFIRMED.json", decision, [evidence])


def test_resume_verifier_requires_complete_confirmed_l0123_receipt(tmp_path: Path):
    receipt = _resume_receipt(tmp_path)
    assert gate.verify_resume_receipt(receipt)["resume_old_ablations_allowed"] is True
    assert gate.main(["verify-resume", "--receipt", str(receipt)]) == 0


def test_resume_verifier_rejects_partial_controller_coverage(tmp_path: Path):
    receipt = _resume_receipt(
        tmp_path,
        eligible_lc_methods=["naive_async", "lc_gate", "lc_damp"],
    )
    with pytest.raises(gate.FinalGateError, match="exactly L0/L1/L2/L3"):
        gate.verify_resume_receipt(receipt)


def _prompt(*, method_gain: float = 0.35, bad: bool = False) -> pd.DataFrame:
    rows = []
    methods = (
        ("static", 2),
        ("static", 4),
        ("static", 8),
        ("tts", 2),
        ("tts", 4),
        ("tts", 8),
        ("naive_async", 4),
    )
    for method, stride in methods:
        for context in gate.CONTEXTS:
            scale = context / 512.0
            for cluster in range(8):
                noise = (cluster - 3.5) * 0.002
                static = 3.25 * scale ** -0.31
                if method == "static":
                    value = static
                elif method == "tts":
                    # Stride 2 is acceptance-best; stride 8 represents the
                    # independently selected engineering role.
                    offset = {2: 0.28, 4: 0.22, 8: 0.12}[stride]
                    value = static + offset
                else:
                    baseline = static + 0.28
                    gain = method_gain * (scale ** 0.14)
                    if bad:
                        gain = (-0.04 if cluster < 4 else 0.04)
                    value = baseline + gain
                rows.append(
                    {
                        "method": method,
                        "model_pair": "dflash",
                        "weight_update_mode": "tail_lora",
                        "update_stride": stride,
                        "dataset": "livecodebench",
                        "lifecycle": "stream",
                        "offered_concurrency": 4,
                        "context_length": context,
                        "prompt_cluster": f"prompt-{cluster}",
                        "seed": 0,
                        "acceptance": value + noise,
                        "benchmark_repetitions": 5,
                    }
                )
    return pd.DataFrame(rows)


def test_cross_stride_uses_acceptance_role_without_pooling_and_passes_ci():
    table, summary = gate._comparison(
        _prompt(),
        method="naive_async",
        stride=4,
        baseline_method="tts",
        baseline_stride=2,
        b=300,
    )
    assert set(table.context_length) == set(gate.CONTEXTS)
    assert set(table.baseline_update_stride) == {2}
    assert summary["lcag_ci_low"] > 0
    assert summary["mean_delta_acceptance_elasticity"] < 0
    assert summary["algorithmic_pass"] is True
    assert len(summary["per_context"]) == 8

    _, foundation = gate._comparison(
        _prompt(),
        method="tts",
        stride=2,
        baseline_method="static",
        baseline_stride=2,
        b=300,
    )
    assert foundation["algorithmic_pass"] is True


def test_cross_stride_ci_failure_is_not_hidden_by_positive_cells():
    _, summary = gate._comparison(
        _prompt(bad=True),
        method="naive_async",
        stride=4,
        baseline_method="tts",
        baseline_stride=2,
        b=300,
    )
    assert summary["lcag_ci_low"] <= 0
    assert summary["algorithmic_pass"] is False


def test_cross_stride_missing_or_ood_role_fails_closed():
    prompt = _prompt()
    prompt = prompt[~((prompt.method == "tts") & (prompt.update_stride == 2) & (prompt.context_length == 40000))]
    with pytest.raises(gate.FinalGateError, match="coverage|comparison"):
        gate._comparison(
            prompt,
            method="naive_async",
            stride=4,
            baseline_method="tts",
            baseline_stride=2,
            b=20,
        )


def test_three_tts_roles_are_explicit_and_unambiguous():
    payload = {
        "roles": {
            "same_stride_tts": {"method": "tts", "update_stride": 4},
            "acceptance_best_tts": {"method": "tts", "stride": 2},
            "engineering_best_tts": 8,
        }
    }
    assert gate._role_strides(payload) == {
        "same_stride": 4,
        "acceptance_best": 2,
        "engineering_best": 8,
    }
    payload["roles"]["acceptance_best"] = 4
    with pytest.raises(gate.FinalGateError, match="ambiguous"):
        gate._role_strides(payload)

    canonical = {
        "roles": {
            "same_stride_tts_for_l0": {"method": "tts", "stride": 4},
            "tts_acceptance_best": {"method": "tts", "stride": 2},
            "tts_engineering_best": {"method": "tts", "stride": 8},
        }
    }
    assert gate._role_strides(canonical) == {
        "same_stride": 4,
        "acceptance_best": 2,
        "engineering_best": 8,
    }


def test_recursive_receipt_rejects_nested_hash_drift(tmp_path):
    leaf = _json(tmp_path / "leaf.txt", {"value": 1})
    nested = _receipt(
        tmp_path / "nested.json",
        {"schema_version": 1, "status": "nested", "scope": "test"},
        [leaf],
    )
    root = _receipt(
        tmp_path / "root.json",
        {"schema_version": 1, "status": "root", "scope": "test"},
        [nested],
    )
    gate._receipt_tree(root)
    leaf.write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(gate.FinalGateError, match="hash mismatch"):
        gate._receipt_tree(root)


def test_raw_round_reconstruction_groups_repetitions_and_expands_static(tmp_path):
    run_dirs = []
    for method, stride in (("static", 1), ("tts", 4)):
        run = tmp_path / method
        run.mkdir()
        _json(
            run / "manifest.json",
            {
                "unit_id": method,
                "method": method,
                "model_pair": "dflash",
                "trainable_scope": "lora",
                "stride": stride,
                "dataset": "livecodebench",
                "lifecycle": "stream",
                "concurrency": 4,
                "seed": 0,
                "engine_params": {"benchmark_repetitions": 5},
            },
        )
        checkpoints = []
        rounds = []
        for repeat, accepted in ((0, 2), (1, 4)):
            sample = f"sample-{repeat}"
            group = __import__("hashlib").sha256(sample.encode()).hexdigest()
            checkpoints.append(
                {
                    "sample_id": sample,
                    "source_sample_id": "same-prompt",
                    "context_length": 512,
                }
            )
            rounds.append(
                {
                    "request_id": f"lightcone-g{group}-p{repeat}",
                    "prefix_feature_exact": True,
                    "prefix_len_before": 512,
                    "algorithmic_censored": False,
                    "accepted_drafts": accepted,
                    "version_canary_ok": True,
                    "cache_version_canary_ok": True,
                }
            )
        _json(run / "prefix-checkpoints.json", {"checkpoints": checkpoints})
        pd.DataFrame(rounds).to_parquet(run / "rounds.parquet", index=False)
        run_dirs.append(run)

    frame = gate._raw_prompt_table(run_dirs)
    assert set(frame.method) == {"static", "tts"}
    # Static receives the observed adaptive (mode, stride) identity, exactly
    # like the standard analyzer, and two repetitions become one prompt cell.
    assert set(frame.update_stride) == {4}
    assert len(frame) == 2
    assert np.allclose(frame.acceptance, 3.0)
    frame.to_parquet(tmp_path / "p5_prompt_acceptance.parquet", index=False)
    gate._verify_prompt_derivation(frame, tmp_path)


def test_standard_gate_selects_exact_method_stride(tmp_path):
    _json(
        tmp_path / "p5_claim_gates.json",
        [
            {
                "method": "naive_async",
                "baseline_method": "tts",
                "update_stride": 4,
                "benchmark_repetitions": 5,
                "exactness_pass": True,
                "algorithmic_pass": True,
            },
            {
                "method": "naive_async",
                "baseline_method": "tts",
                "update_stride": 8,
                "benchmark_repetitions": 5,
                "exactness_pass": True,
                "algorithmic_pass": False,
            },
        ],
    )
    row = gate._standard_same_stride_gate(
        analysis_root=tmp_path, method="naive_async", stride=4
    )
    assert row["algorithmic_pass"] is True
    with pytest.raises(gate.FinalGateError, match="expected one"):
        gate._standard_same_stride_gate(
            analysis_root=tmp_path, method="lc_gate", stride=4
        )


def _engineering_table(*, regress_at: int | None = None) -> pd.DataFrame:
    rows = []
    for context in gate.CONTEXTS:
        load = 20 if context <= 16384 else 8
        rows.extend(
            [
                {
                    "method": "tts",
                    "update_stride": 8,
                    "context_length": context,
                    "offered_concurrency": load,
                    "decode_goodput_tps": 100.0,
                    "target_calls_per_output_token": 0.30,
                },
                {
                    "method": "naive_async",
                    "update_stride": 4,
                    "context_length": context,
                    "offered_concurrency": load,
                    "decode_goodput_tps": 99.0 if context == regress_at else 102.0,
                    "target_calls_per_output_token": 0.28,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_engineering_gate_is_context_load_specific_and_separate(tmp_path):
    _engineering_table().to_parquet(tmp_path / "p5_long_context_acceptance.parquet")
    result = gate._engineering(
        analysis_root=tmp_path,
        method="naive_async",
        stride=4,
        baseline_stride=8,
    )
    assert result["engineering_pass"] is True
    assert {row["offered_concurrency"] for row in result["per_context_load"]} == {8, 20}
    _engineering_table(regress_at=16384).to_parquet(
        tmp_path / "p5_long_context_acceptance.parquet"
    )
    result = gate._engineering(
        analysis_root=tmp_path,
        method="naive_async",
        stride=4,
        baseline_stride=8,
    )
    assert result["engineering_pass"] is False
    assert result["claim_semantics"].endswith("not_acceptance")


def test_controller_ineligible_methods_are_not_counted_as_passes():
    controller = {"eligible": {"l1": False, "l2": True, "l3": False}}
    generation = {"methods": ["static", "tts", "naive_async", "lc_damp"]}
    assert gate._controller_methods(controller, generation) == [
        "naive_async",
        "lc_damp",
    ]
    generation["methods"].append("lc_gate")
    with pytest.raises(gate.FinalGateError, match="coverage differs"):
        gate._controller_methods(controller, generation)


def test_final_method_aggregation_requires_every_explicit_l0_l1_l2_l3_pass():
    methods = list(gate.LC_METHODS)
    passing = {method: {"algorithmic_pass": True} for method in methods}
    assert gate._all_lc_methods_pass(methods, passing, "algorithmic_pass") is True

    for missing in methods:
        partial = {method: row for method, row in passing.items() if method != missing}
        assert gate._all_lc_methods_pass(
            list(partial), partial, "algorithmic_pass"
        ) is False

    for failed in methods:
        rows = {method: dict(row) for method, row in passing.items()}
        rows[failed]["algorithmic_pass"] = False
        assert gate._all_lc_methods_pass(methods, rows, "algorithmic_pass") is False


def test_terminal_levels_do_not_turn_missing_controllers_into_passes(
    tmp_path, monkeypatch
):
    anchor = _json(tmp_path / "anchor.json", {"frozen": True})
    controller = _receipt(
        tmp_path / "CONTROLLER_BLOCKED.json",
        {
            "schema_version": 1,
            "status": "matched_controller_blocked",
            "scope": "matched_dflash_l1_l2_l3_evidence",
            "eligible": {"l1": False, "l2": False, "l3": False},
        },
        [anchor],
    )
    foundation_identity = {"roles": "frozen"}
    foundation = _receipt(
        tmp_path / "TTS_FOUNDATION_SELECTED.json",
        {
            "schema_version": 2,
            "status": "TTS_0_40K_CONFIRMED",
            "scope": "tts_0_40k_foundation",
            "formal_acceptance_foundation_pass": True,
            "identity": foundation_identity,
            "identity_sha256": gate.sha256_json(foundation_identity),
            "roles": {
                "same_stride_tts": 4,
                "acceptance_best_tts": 2,
                "engineering_best_tts": 8,
            },
        },
        [anchor],
    )
    algorithmic_manifest = _json(
        tmp_path / "algorithmic.json", {"units": [{"unit_id": "algorithmic"}]}
    )
    mfu_manifest = _json(tmp_path / "mfu.json", {"units": [{"unit_id": "mfu"}]})
    _attest(algorithmic_manifest)
    _attest(mfu_manifest)
    generation = _receipt(
        tmp_path / "HEADLINE_GENERATED.json",
        {
            "schema_version": 1,
            "status": "final_0_40k_manifests_generated",
            "scope": "evidence_bound_context_specific_p5",
            "methods": ["static", "tts", "naive_async"],
            "optimizer_identity": {"adaptation_stride": 4},
            "artifacts": {
                "ALGORITHMIC_C4": {
                    "path": str(algorithmic_manifest),
                    "sha256": sha256_file(algorithmic_manifest),
                },
                "MFU_CONTEXT_LOAD": {
                    "path": str(mfu_manifest),
                    "sha256": sha256_file(mfu_manifest),
                },
            },
        },
        [controller, foundation],
    )
    algorithmic_analysis = tmp_path / "algorithmic-analysis"
    mfu_analysis = tmp_path / "mfu-analysis"
    algorithmic_analysis.mkdir()
    mfu_analysis.mkdir()
    algo_provenance = _json(
        algorithmic_analysis / "analysis-manifest.json", {"analysis": {"baseline": "tts"}}
    )
    mfu_provenance = _json(
        mfu_analysis / "analysis-manifest.json", {"analysis": {"baseline": "tts"}}
    )

    def fake_analysis(*, root, artifact_root, manifest_path):
        provenance = algo_provenance if root == algorithmic_analysis else mfu_provenance
        return {"analysis": {"baseline": "tts"}}, [provenance], [tmp_path / "unused"]

    monkeypatch.setattr(gate, "_verify_analysis", fake_analysis)
    monkeypatch.setattr(gate, "_raw_prompt_table", lambda _: _prompt())
    monkeypatch.setattr(gate, "_verify_prompt_derivation", lambda *args: None)
    monkeypatch.setattr(gate, "_verify_engineering_pairing", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gate,
        "_standard_same_stride_gate",
        lambda **kwargs: {"algorithmic_pass": True},
    )
    monkeypatch.setattr(
        gate,
        "_safety",
        lambda root, methods: {
            method: {
                "adaptation_fallback_count": 0,
                "exactness_violations": 0,
                "version_mismatch_count": 0,
                "pass": True,
            }
            for method in methods
        },
    )
    monkeypatch.setattr(
        gate,
        "_engineering",
        lambda **kwargs: {"engineering_pass": True, "per_context_load": []},
    )
    common = dict(
        headline_generation=generation,
        controller_terminal=controller,
        tts_foundation_terminal=foundation,
        algorithmic_artifact_root=tmp_path / "algorithmic-artifacts",
        algorithmic_manifest=algorithmic_manifest,
        algorithmic_analysis_root=algorithmic_analysis,
        mfu_artifact_root=tmp_path / "mfu-artifacts",
        mfu_manifest=mfu_manifest,
        mfu_analysis_root=mfu_analysis,
        bootstrap_replicates=50,
    )
    result = gate.evaluate(
        **common,
        output=tmp_path / "default.json",
    )
    assert result["status"] == "BLOCKED"
    assert result["algorithmic_status"] == "BLOCKED"
    assert result["all_algorithmic_pass"] is False
    assert result["all_eligible_algorithmic_pass"] is True
    assert result["all_declared_l0123_evaluated"] is False
    assert result["resume_old_ablations_allowed"] is False
    assert result["all_eligible_engineering_pass"] is True
    assert result["all_engineering_pass"] is False
    assert result["controller_method_status"]["lc_gate"] == "NOT_ELIGIBLE_NOT_PASS"
    assert (
        result["diagnostic_acceptance_comparisons_vs_engineering_best_tts"]
        ["naive_async"]["baseline_stride"]
        == 8
    )

    strict = gate.evaluate(
        **common,
        output=tmp_path / "strict.json",
        require_all_engineering=True,
    )
    assert strict["status"] == "BLOCKED"
    assert strict["algorithmic_status"] == "BLOCKED"
    assert strict["engineering_status"] == "BLOCKED"
