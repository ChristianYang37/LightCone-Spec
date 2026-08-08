from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from lightcone_spec.orchestration.manifest import ExperimentManifest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiments/run_priority_dflash_calibration_queue.py"
SPEC = ROOT / "scripts/experiments/priority_calibration_candidates_v1.json"
CALIBRATION = (
    ROOT
    / "manifests/p5/p5_priority_dflash_calibration_v1.json"
)
EVALUATION = (
    ROOT / "manifests/p5/p5_priority_dflash_0_40k_v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("priority_calibration_queue", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load(*, lcag=0.1, ci=0.02, throughput=0.9, gains=None):
    gains = gains or {512: 0.0, 4096: 0.03, 16384: 0.05, 40000: 0.08}
    return {
        "acceptance_gain_by_context": gains,
        "lcag": lcag,
        "lcag_ci_low": ci,
        "positive_contexts": sum(value > 0 for value in gains.values()),
        "worst_context_gain": min(gains.values()),
        "min_throughput_ratio": throughput,
        "peak_hbm_bytes": 80 * (1 << 30),
        "max_hbm_delta_bytes": 1 << 30,
    }


def _summary(**load_kwargs):
    return {
        "loads": {
            str(concurrency): _load(**load_kwargs)
            for concurrency in (1, 4, 8)
        },
        "hard_safety_pass": True,
        "engineering_pass": load_kwargs.get("throughput", 0.9) >= 0.8,
    }


def test_calibration_manifest_is_disjoint_and_candidate_spec_is_not_a_manifest():
    module = _module()
    calibration = ExperimentManifest.load(CALIBRATION)
    evaluation = ExperimentManifest.load(EVALUATION)
    assert {unit.method for unit in calibration.units} == {"static", "tts"}
    assert {unit.dataset for unit in calibration.units} == {"math500"}
    assert {unit.dataset for unit in evaluation.units} == {"livecodebench"}
    assert calibration.engine_params["p5_context_lengths"] == [4096, 16384, 40000]
    assert calibration.engine_params["max_consecutive_exactness_failures"] == 1
    assert not SPEC.is_relative_to(ROOT / "manifests")
    candidates, calibration_gate, evaluation_gate = module.load_candidate_spec(SPEC)
    assert [candidate.candidate_id for candidate in candidates] == [
        "tail-lora-r16-lr1e-4",
        "tail-lora-r16-lr3e-4",
        "output-residual-r16-lr3e-4",
    ]
    assert calibration_gate["min_positive_contexts"] == 2
    assert evaluation_gate["expected_contexts"] == [512, 4096, 16384, 40000]
    assert evaluation_gate["min_lcag_ci_low"] == 0.0


def test_selection_requires_more_than_one_positive_calibration_bucket():
    module = _module()
    gate = json.loads(SPEC.read_text())["calibration_gate"]
    summary = {
        "loads": {
            "1": _load(gains={4096: 0.0, 16384: 0.0, 40000: 0.1})
        },
        "hard_safety_pass": True,
        "engineering_pass": True,
    }
    verdict = module.calibration_verdict(summary, gate)
    assert verdict["algorithmic_screen_pass"] is False
    assert verdict["eligible"] is False


def test_winner_ranks_acceptance_before_independent_engineering_cost():
    module = _module()
    rows = []
    for candidate_id, lcag, throughput in (("a", 0.2, 0.79), ("b", 0.1, 0.95)):
        summary = {
            "loads": {"1": _load(lcag=lcag, ci=0.01, throughput=throughput)},
            "hard_safety_pass": True,
            "engineering_pass": throughput >= 0.8,
        }
        rows.append(
            {
                "candidate": {
                    "candidate_id": candidate_id,
                    "weight_update_mode": "lora",
                    "learning_rate": 1e-4,
                },
                "summary": summary,
                "verdict": {"eligible": True},
            }
        )
    winner = module.choose_winner(rows)
    assert winner["candidate"]["candidate_id"] == "a"
    assert winner["summary"]["engineering_pass"] is False


def test_heldout_gate_requires_every_bucket_and_strict_positive_lcag_ci():
    module = _module()
    gate = json.loads(SPEC.read_text())["evaluation_gate"]
    verdict = module.heldout_verdict(_summary(throughput=0.7), gate)
    assert verdict["algorithmic_pass"] is True
    assert verdict["engineering_pass"] is False
    assert verdict["downstream_ready"] is True

    zero_ci = module.heldout_verdict(_summary(ci=0.0), gate)
    assert zero_ci["algorithmic_pass"] is False
    assert zero_ci["downstream_ready"] is False

    negative = module.heldout_verdict(
        _summary(gains={512: 0.0, 4096: 0.03, 16384: -1e-6, 40000: 0.08}),
        gate,
    )
    assert negative["algorithmic_pass"] is False
    assert negative["downstream_ready"] is False


def test_heldout_gate_fails_closed_on_version_or_exactness_safety():
    module = _module()
    gate = json.loads(SPEC.read_text())["evaluation_gate"]
    summary = _summary()
    summary["hard_safety_pass"] = False
    verdict = module.heldout_verdict(summary, gate)
    assert verdict["algorithmic_pass"] is True
    assert verdict["hard_safety_pass"] is False
    assert verdict["downstream_ready"] is False


def test_attested_receipt_detects_evidence_drift(tmp_path):
    module = _module()
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("v1", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    module.write_attested_json(
        receipt, {"schema_version": 1, "status": "complete"}, evidence=[evidence]
    )
    assert module.load_attested_json(receipt)["status"] == "complete"
    evidence.write_text("v2", encoding="utf-8")
    with pytest.raises(module.QueueError, match="evidence drift"):
        module.load_attested_json(receipt)


def test_source_tree_fingerprint_is_deterministic_and_detects_drift(tmp_path):
    module = _module()
    tree = tmp_path / "runtime"
    tree.mkdir()
    source = tree / "worker.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    ignored = tree / "__pycache__"
    ignored.mkdir()
    (ignored / "worker.py").write_text("generated", encoding="utf-8")

    first = module.source_tree_fingerprint(tree)
    assert first == module.source_tree_fingerprint(tree)
    assert first["file_count"] == 1
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert module.source_tree_fingerprint(tree)["merkle_sha256"] != first[
        "merkle_sha256"
    ]


def test_queue_subprocess_drops_reference_cuda_debug_environment(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    module = _module()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    receipt = tmp_path / "complete.json"
    assert module._run_commands(
        tmp_path,
        "headline",
        [["lightcone-spec", "run-manifest"]],
        receipt,
        lambda: [],
    )
    assert "CUBLAS_WORKSPACE_CONFIG" not in observed["env"]
    assert "CUDA_LAUNCH_BLOCKING" not in observed["env"]
    assert module.load_attested_json(receipt)["headline_env_sanitized"] is True


def test_combined_view_reuses_one_static_and_isolates_candidate(tmp_path):
    module = _module()
    roots = []
    for label in ("static", "candidate"):
        root = tmp_path / label
        run = root / f"run-{label}"
        run.mkdir(parents=True)
        (run / "exit.json").write_text(
            json.dumps({"status": "complete_valid", "exit_code": 0}),
            encoding="utf-8",
        )
        (run / "hashes.json").write_text("{}", encoding="utf-8")
        roots.append(root)
    combined = tmp_path / "combined"
    module._combined_view(roots[0], roots[1], combined)
    links = sorted(combined.iterdir())
    assert len(links) == 2
    assert all(link.is_symlink() for link in links)
    assert {link.name.split("--", 1)[0] for link in links} == {
        "static",
        "candidate",
    }


def test_nonfinite_update_scan_distinguishes_parquet_nulls_from_failures(
    tmp_path, monkeypatch
):
    module = _module()
    run = tmp_path / "runs" / "complete-run"
    run.mkdir(parents=True)
    (run / "exit.json").write_text(
        json.dumps({"status": "complete_valid", "exit_code": 0}),
        encoding="utf-8",
    )
    (run / "hashes.json").write_text("{}", encoding="utf-8")
    (run / "updates.parquet").touch()
    rows = [
        {
            "failure_reason": float("nan"),
            "grad_norm": 1.0,
            "candidate_delta_norm": 0.1,
        },
        {
            "failure_reason": "request_ended",
            "grad_norm": float("nan"),
            "candidate_delta_norm": float("nan"),
        },
        {
            "failure_reason": "nonfinite_gradient",
            "grad_norm": float("nan"),
            "candidate_delta_norm": None,
        },
        {
            "failure_reason": None,
            "grad_norm": math.inf,
            "candidate_delta_norm": 0.2,
        },
    ]

    class Frame:
        def to_dict(self, orient):
            assert orient == "records"
            return rows

    monkeypatch.setattr(module, "_read_table", lambda _path: Frame())
    assert module._nonfinite_updates(tmp_path / "runs") == 2


def test_mode_agnostic_static_pairs_with_residual_acceptance_and_throughput():
    import pandas as pd

    from lightcone_spec.statistics.tables import (
        expand_static_weight_update_modes,
        long_context_acceptance_table,
    )

    rounds = []
    for context in (4096, 16384):
        for prompt in range(4):
            common = {
                "model_pair": "qwen3_4b_dflash16",
                "dataset": "math500",
                "lifecycle": "stream",
                "offered_concurrency": 1,
                "context_length": context,
                "request_id": f"prompt-{prompt}:ctx-{context}",
                "prompt_cluster": f"prompt-{prompt}",
                "seed": 0,
                "draft_tokens": 7,
                "verify_len": 8,
                "committed_per_verify": 2,
                "target_calls": 1,
                "draft_cuda_us": 1.0,
                "verify_cuda_us": 2.0,
                "accept_cuda_us": 1.0,
                "batch_size": 1,
                "version_canary_ok": True,
                "prefix_len_before": context,
            }
            rounds.extend(
                (
                    {
                        **common,
                        "method": "static",
                        "weight_update_mode": "lora",
                        "accepted_drafts": 1,
                    },
                    {
                        **common,
                        "method": "tts",
                        "weight_update_mode": "residual",
                        "accepted_drafts": 2,
                        "committed_per_verify": 3,
                    },
                )
            )
    acceptance = long_context_acceptance_table(pd.DataFrame(rounds), b=40)
    tts = acceptance[acceptance.method == "tts"]
    assert set(tts.weight_update_mode) == {"residual"}
    assert tts.acceptance_gain_vs_static.notna().all()
    assert set(tts.acceptance_gain_vs_static) == {1.0}

    summaries = pd.DataFrame(
        [
            {
                "model_pair": "qwen3_4b_dflash16",
                "method": "static",
                "weight_update_mode": "lora",
                "decode_tps": 100.0,
            },
            {
                "model_pair": "qwen3_4b_dflash16",
                "method": "tts",
                "weight_update_mode": "residual",
                "decode_tps": 90.0,
            },
        ]
    )
    expanded = expand_static_weight_update_modes(summaries)
    residual = expanded[expanded.weight_update_mode == "residual"]
    assert residual.set_index("method").decode_tps.to_dict() == {
        "static": 100.0,
        "tts": 90.0,
    }

    static_only = summaries[summaries.method == "static"]
    assert expand_static_weight_update_modes(static_only).equals(
        static_only.reset_index(drop=True)
    )
