from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "experiments" / "analyze_dflash_tts_pair.py"
SPEC = importlib.util.spec_from_file_location("analyze_dflash_tts_pair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def _write_run(
    root: Path,
    *,
    mode: str,
    acceptance: list[int],
    tokens: list[int],
    seed: int = 0,
) -> Path:
    root.mkdir(parents=True)
    input_tokens = 2
    prefix = input_tokens
    rows = []
    for index, length in enumerate(acceptance):
        committed = list(range(prefix, prefix + length))
        draft_block = list(range(prefix, prefix + 16))
        target_posterior = list(range(prefix + 1, prefix + 17))
        rows.append(
            {
                "schema_version": 1,
                "sample_id": "sample-7",
                "round_index": index,
                "seed": seed,
                "mode": mode,
                "prefix_length_before": prefix,
                "acceptance_length": length,
                "accepted_draft_tokens": length - 1,
                "committed_token_ids": committed,
                "bonus_token_id": target_posterior[length - 1],
                "draft_block_token_ids": draft_block,
                "target_posterior_token_ids": target_posterior,
                "update": {
                    "applied": mode != "static",
                    "optimizer_step": index + 1 if mode != "static" else None,
                    "loss": 1.0 if mode != "static" else None,
                    "distillation_kl": 0.8 if mode != "static" else None,
                    "proximal_kl": 0.2 if mode != "static" else None,
                    "grad_norm": 2.0 if mode != "static" else None,
                    "parameters_with_grad": 1 if mode != "static" else 0,
                    "parameters_without_grad": [],
                    "backward_cuda_us": 10.0 if mode != "static" else None,
                    "optimizer_cuda_us": 11.0 if mode != "static" else None,
                    "update_cuda_us": 21.0 if mode != "static" else None,
                    "parameter_delta_l2": None,
                    "parameter_displacement_l2": None,
                    "parameter_l2": None,
                    "relative_parameter_delta": None,
                    "parameter_audit_interval_steps": None,
                },
                "timing_seconds": {
                    "draft_forward": 0.001,
                    "target_verify": 0.002,
                    "update": 0.003 if mode != "static" else 0.0,
                    "round_total": 0.006,
                },
                "cache_lengths": {
                    "draft_before": prefix,
                    "draft_after_forward": prefix + 15,
                    "draft_after_crop": prefix,
                    "target_after_crop": prefix + length,
                    "draft_tensors_detached_after_update": 0,
                },
                "hbm_bytes": {
                    "allocated_end": 100 + index,
                    "reserved_end": 200 + index,
                    "running_peak_allocated": 300 + index,
                    "running_peak_reserved": 400 + index,
                },
            }
        )
        prefix += length
    (root / "rounds.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    output_count = sum(acceptance)
    summary = {
        "schema_version": 1,
        "status": "complete_reference_run",
        "mode": mode,
        "trainable_scope": "none_static" if mode == "static" else "candidate",
        "reference": {"source_sha256": "reference"},
        "models": {
            "target": {"declared_revision": "target"},
            "draft": {"declared_revision": "draft"},
        },
        "dataset": {
            "sha256": "dataset",
            "declared_revision": "dataset-revision",
            "sample_index": 7,
            "sample_id": "sample-7",
        },
        "parameters": {
            "seed": seed,
            "temperature": 0.0,
            "block_size": 16,
            "mask_token_id": 9,
            "stop_token_ids": None,
            "max_new_tokens": output_count,
            "draft_cache_policy": "stale",
            "dtype": "bfloat16",
            "enable_thinking": True,
            "optimizer": None if mode == "static" else "ADAM",
            "lr": None if mode == "static" else 1e-4,
            "weight_decay": 0.0,
        },
        "trainable_layout": {
            "parameter_count": 0 if mode == "static" else 1,
            "parameter_tensors": 0 if mode == "static" else 1,
            "layout_sha256": f"layout-{mode}",
        },
        "generation": {
            "num_input_tokens": input_tokens,
            "num_output_tokens": output_count,
            "rounds": len(rows),
            "optimizer_steps": 0 if mode == "static" else len(rows),
            "trainable_parameter_count": 0 if mode == "static" else 1,
            "decode_seconds": float(len(rows) * (2 if mode != "static" else 1)),
        },
        "output": {"token_ids": tokens},
    }
    (root / "summary.json").write_text(json.dumps(summary))
    return root


def _upgrade_to_verified_v3(run: Path) -> None:
    summary_path = run / "summary.json"
    rounds_path = run / "rounds.jsonl"
    summary = json.loads(summary_path.read_text())
    rounds = [json.loads(line) for line in rounds_path.read_text().splitlines()]
    summary["schema_version"] = 3
    summary["harness"] = {
        "source_sha256": "b" * 64,
        "artifact_schema_version": 3,
    }
    summary["artifact_identity"] = {
        "verification_status": "fully_verified_content_sha256_v1"
    }
    for model in summary["models"].values():
        model["weight_files"] = [
            {"name": "model.safetensors", "bytes": 4, "sha256": "c" * 64}
        ]
    summary["tokenizer"] = {
        "files": [
            {"name": "tokenizer.json", "bytes": 4, "sha256": "d" * 64}
        ],
        "content_identity_sha256": "f" * 64,
    }
    summary["dataset"]["rendered_input_token_ids"] = {
        "serialization": "int64_le_c_order_v1",
        "shape": [1, summary["generation"]["num_input_tokens"]],
        "sha256": "e" * 64,
    }
    summary["runtime_fingerprint"] = {
        "schema_version": 1,
        "python_version": "3.12.11",
        "python_implementation": "CPython",
        "platform": "Linux-x86_64",
        "torch_version": "2.8.0",
        "cuda_runtime_version": "12.8",
        "cuda_driver_version": 12080,
        "attention_implementation": "sdpa",
        "dtype": "bfloat16",
        "device": "cuda:0",
        "resolved_device": "cuda:0",
        "allocator_config": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_ALLOC_CONF": None,
        },
        "cuda_visible_devices": None,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "allow_tf32": {"matmul": False, "cudnn": False},
        "float32_matmul_precision": "highest",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
        "sdpa_backends": {
            "flash": False,
            "memory_efficient": False,
            "math": True,
            "cudnn": False,
        },
        "gpu": {
            "name": "RTX PRO 6000 Blackwell",
            "total_memory_bytes": 96 << 30,
            "compute_capability": "12.0",
            "device_index": 0,
        },
    }
    summary["parameters"]["required_prefix_plus_block"] = (
        summary["generation"]["num_input_tokens"]
        + summary["parameters"]["max_new_tokens"]
        + summary["parameters"]["block_size"]
        - 1
    )
    runtime_layout = f"runtime-{summary['mode']}"
    summary["generation"]["parameter_layout_sha256"] = runtime_layout
    summary["output"]["rounds_jsonl"] = "rounds.jsonl"
    for row in rounds:
        row["schema_version"] = 3
        row["prefix_len_before"] = row.pop("prefix_length_before")
        row["trainable_parameter_count"] = summary["generation"][
            "trainable_parameter_count"
        ]
        row["parameter_layout_sha256"] = runtime_layout
        row["draft_cache_policy"] = "stale"
        row["provenance"] = {
            "reference_source_sha256": "reference",
            "target_declared_revision": "target",
            "draft_declared_revision": "draft",
            "dataset_declared_revision": "dataset-revision",
            "dataset_sha256": "dataset",
            "harness_source_sha256": "b" * 64,
        }
    rounds_path.write_text("".join(json.dumps(row) + "\n" for row in rounds))
    summary["output"]["rounds_sha256"] = hashlib.sha256(
        rounds_path.read_bytes()
    ).hexdigest()
    summary_path.write_text(json.dumps(summary))


def _rewrite_rounds(run: Path, mutate) -> None:
    rounds_path = run / "rounds.jsonl"
    summary_path = run / "summary.json"
    rows = [json.loads(line) for line in rounds_path.read_text().splitlines()]
    mutate(rows)
    rounds_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = json.loads(summary_path.read_text())
    summary["output"]["rounds_sha256"] = hashlib.sha256(
        rounds_path.read_bytes()
    ).hexdigest()
    summary_path.write_text(json.dumps(summary))


def test_pair_analysis_separates_algorithmic_and_reference_speed(tmp_path):
    baseline = _write_run(
        tmp_path / "baseline",
        mode="static",
        acceptance=[2, 2, 2],
        tokens=[1, 2, 3, 4, 5, 6],
    )
    candidate = _write_run(
        tmp_path / "candidate",
        mode="full-drafter",
        acceptance=[3, 3],
        tokens=[1, 2, 3, 4, 5, 6],
    )

    result = analysis.compare_runs(baseline, candidate, bucket_size=4)

    assert result["exact_output_token_ids"] is True
    assert result["baseline"]["accepted_drafts_per_verify"] == 1.0
    assert result["candidate"]["accepted_drafts_per_verify"] == 2.0
    assert result["gain"]["target_calls_per_output_token_relative"] < 0.0
    assert result["algorithmic_pass_exploratory"] is True
    assert result["engineering_pass_reference"] is False
    assert result["classification"] == "single-prompt-legacy-unverified-pilot"
    assert "paired_buckets" not in result
    assert all(
        row["comparison_status"].startswith("descriptive_")
        for row in result["descriptive_bucket_comparisons"]
    )


def test_pair_analysis_rejects_seed_mismatch(tmp_path):
    baseline = _write_run(
        tmp_path / "baseline",
        mode="static",
        acceptance=[2],
        tokens=[1, 2],
    )
    candidate = _write_run(
        tmp_path / "candidate",
        mode="full-drafter",
        acceptance=[2],
        tokens=[1, 2],
        seed=1,
    )

    with pytest.raises(ValueError, match="paired identity mismatch"):
        analysis.compare_runs(baseline, candidate, bucket_size=4)


def test_pair_analysis_accepts_schema_v2_prefix_and_clips_final_round(tmp_path):
    baseline = _write_run(
        tmp_path / "baseline-v2",
        mode="static",
        acceptance=[4, 4],
        tokens=[1, 2, 3, 4, 5, 6],
    )
    candidate = _write_run(
        tmp_path / "candidate-v2",
        mode="full-drafter",
        acceptance=[3, 3],
        tokens=[1, 2, 3, 4, 5, 6],
    )
    for run in (baseline, candidate):
        summary_path = run / "summary.json"
        rounds_path = run / "rounds.jsonl"
        summary = json.loads(summary_path.read_text())
        rounds = [json.loads(line) for line in rounds_path.read_text().splitlines()]
        summary["schema_version"] = 2
        summary["harness"] = {
            "source_sha256": "harness",
            "artifact_schema_version": 2,
        }
        summary["generation"]["num_output_tokens"] = 6
        summary["generation"]["parameter_layout_sha256"] = f"runtime-{summary['mode']}"
        summary["parameters"]["max_new_tokens"] = 6
        summary["output"]["rounds_jsonl"] = "rounds.jsonl"
        for row in rounds:
            row["schema_version"] = 2
            row["prefix_len_before"] = row.pop("prefix_length_before")
            row["trainable_parameter_count"] = summary["generation"][
                "trainable_parameter_count"
            ]
            row["parameter_layout_sha256"] = summary["generation"][
                "parameter_layout_sha256"
            ]
            row["draft_cache_policy"] = "stale"
            row["provenance"] = {
                "reference_source_sha256": "reference",
                "target_declared_revision": "target",
                "draft_declared_revision": "draft",
                "dataset_declared_revision": "dataset-revision",
                "dataset_sha256": "dataset",
                "harness_source_sha256": "harness",
            }
        rounds_path.write_text("".join(json.dumps(row) + "\n" for row in rounds))
        import hashlib

        summary["output"]["rounds_sha256"] = hashlib.sha256(
            rounds_path.read_bytes()
        ).hexdigest()
        summary_path.write_text(json.dumps(summary))

    result = analysis.compare_runs(baseline, candidate, bucket_size=8)
    assert result["baseline"]["algorithmic_committed_tokens_per_verify"] == 4.0
    assert result["baseline"]["committed_tokens_per_verify"] == 3.0
    assert result["baseline"]["target_calls_per_output_token"] == pytest.approx(
        2 / 6
    )
    assert result["baseline"]["buckets"][0]["committed_tokens_per_verify"] == 3.0


def test_common_prefix_trajectory_certifies_exact_schema_v3_rounds(tmp_path):
    shorter = _write_run(
        tmp_path / "shorter",
        mode="full-drafter",
        acceptance=[2, 2],
        tokens=[1, 2, 3, 4],
    )
    longer = _write_run(
        tmp_path / "longer",
        mode="full-drafter",
        acceptance=[2, 2, 2],
        tokens=[1, 2, 3, 4, 5, 6],
    )
    _upgrade_to_verified_v3(shorter)
    _upgrade_to_verified_v3(longer)

    result = analysis.compare_common_prefix_trajectory(longer, shorter)

    assert result["comparison_kind"] == "common_prefix"
    assert result["status"] == "exact_common_prefix"
    assert result["max_new_tokens"] == {"shorter": 4, "longer": 6}
    assert result["round_counts"] == {"shorter": 2, "longer": 3}
    assert result["exact_common_rounds"] == 2
    assert result["output_token_prefix_exact"] is True
    assert result["first_mismatch"] is None
    assert len(result["comparison_identity_sha256"]) == 64
    assert len(result["exact_common_trajectory_sha256"]) == 64
    assert len(result["artifact_set_sha256"]) == 64
    for artifact in result["artifact_sha256"].values():
        assert len(artifact["summary_sha256"]) == 64
        assert len(artifact["rounds_sha256"]) == 64


def test_exact_repeat_trajectory_certifies_all_rounds_and_full_identity(
    tmp_path,
):
    run_a = _write_run(
        tmp_path / "repeat-a",
        mode="full-drafter",
        acceptance=[2, 2],
        tokens=[1, 2, 3, 4],
    )
    run_b = _write_run(
        tmp_path / "repeat-b",
        mode="full-drafter",
        acceptance=[2, 2],
        tokens=[1, 2, 3, 4],
    )
    _upgrade_to_verified_v3(run_a)
    _upgrade_to_verified_v3(run_b)

    def mutate_performance(rows):
        rows[0]["timing_seconds"]["round_total"] += 3.0
        rows[1]["hbm_bytes"]["allocated_end"] += 1024

    _rewrite_rounds(run_b, mutate_performance)
    result = analysis.compare_common_prefix_trajectory(run_a, run_b)

    assert result["comparison_kind"] == "exact_repeat"
    assert result["status"] == "exact_repeat"
    assert result["max_new_tokens"] == {"run_a": 4, "run_b": 4}
    assert result["round_counts"] == {"run_a": 2, "run_b": 2}
    assert result["exact_common_rounds"] == 2
    assert result["output_token_ids_exact"] is True
    assert result["output_token_prefix_exact"] is None
    assert result["first_mismatch"] is None
    assert (
        result["artifact_sha256"]["run_a"]["rounds_sha256"]
        != result["artifact_sha256"]["run_b"]["rounds_sha256"]
    )

    summary_path = run_b / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["parameters"]["required_prefix_plus_block"] += 1
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(
        ValueError,
        match=r"trajectory identity mismatch.*required_prefix_plus_block",
    ):
        analysis.compare_common_prefix_trajectory(run_a, run_b)


def test_exact_repeat_trajectory_reports_round_mismatch(tmp_path):
    run_a = _write_run(
        tmp_path / "repeat-round-a",
        mode="full-drafter",
        acceptance=[2, 2],
        tokens=[1, 2, 3, 4],
    )
    run_b = _write_run(
        tmp_path / "repeat-round-b",
        mode="full-drafter",
        acceptance=[2, 2],
        tokens=[1, 2, 3, 4],
    )
    _upgrade_to_verified_v3(run_a)
    _upgrade_to_verified_v3(run_b)
    _rewrite_rounds(
        run_b,
        lambda rows: rows[1]["update"].__setitem__("loss", 1.25),
    )

    result = analysis.compare_common_prefix_trajectory(run_a, run_b)

    assert result["comparison_kind"] == "exact_repeat"
    assert result["status"] == "trajectory_mismatch"
    assert result["exact_common_rounds"] == 1
    assert result["output_token_ids_exact"] is True
    assert result["first_mismatch"]["round_index"] == 1
    assert result["first_mismatch"]["field"] == "update.loss"
    assert result["first_mismatch"]["run_a_value"] == 1.0
    assert result["first_mismatch"]["run_b_value"] == 1.25


def test_exact_repeat_trajectory_compares_complete_output(tmp_path):
    run_a = _write_run(
        tmp_path / "repeat-output-a",
        mode="full-drafter",
        acceptance=[2, 2],
        tokens=[1, 2, 3, 4],
    )
    run_b = _write_run(
        tmp_path / "repeat-output-b",
        mode="full-drafter",
        acceptance=[2, 2],
        tokens=[1, 2, 3, 9],
    )
    _upgrade_to_verified_v3(run_a)
    _upgrade_to_verified_v3(run_b)

    result = analysis.compare_common_prefix_trajectory(run_a, run_b)

    assert result["comparison_kind"] == "exact_repeat"
    assert result["status"] == "trajectory_mismatch"
    assert result["exact_common_rounds"] == 2
    assert result["output_token_ids_exact"] is False
    assert result["first_mismatch"] == {
        "round_index": None,
        "field": "output.token_ids[3]",
        "run_a_value": 4,
        "run_b_value": 9,
    }


def test_common_prefix_trajectory_reports_first_semantic_mismatch(
    tmp_path, capsys
):
    shorter = _write_run(
        tmp_path / "shorter-mismatch",
        mode="full-drafter",
        acceptance=[2, 2],
        tokens=[1, 2, 3, 4],
    )
    longer = _write_run(
        tmp_path / "longer-mismatch",
        mode="full-drafter",
        acceptance=[2, 2, 2],
        tokens=[1, 2, 3, 4, 5, 6],
    )
    _upgrade_to_verified_v3(shorter)
    _upgrade_to_verified_v3(longer)
    _rewrite_rounds(
        longer,
        lambda rows: rows[1]["update"].__setitem__("loss", 1.25),
    )

    result = analysis.compare_common_prefix_trajectory(shorter, longer)

    assert result["status"] == "trajectory_mismatch"
    assert result["exact_common_rounds"] == 1
    assert result["output_token_prefix_exact"] is None
    assert result["first_mismatch"]["round_index"] == 1
    assert result["first_mismatch"]["field"] == "update.loss"
    assert result["first_mismatch"]["shorter_value"] == 1.0
    assert result["first_mismatch"]["longer_value"] == 1.25
    assert len(
        result["first_mismatch"]["shorter_round_semantics_sha256"]
    ) == 64
    assert analysis.main(
        ["--common-prefix", str(shorter), str(longer)]
    ) == 1
    assert json.loads(capsys.readouterr().out)["status"] == (
        "trajectory_mismatch"
    )


def test_common_prefix_trajectory_excludes_only_performance_measurements(
    tmp_path,
):
    shorter = _write_run(
        tmp_path / "shorter-performance",
        mode="full-drafter",
        acceptance=[2, 2],
        tokens=[1, 2, 3, 4],
    )
    longer = _write_run(
        tmp_path / "longer-performance",
        mode="full-drafter",
        acceptance=[2, 2, 2],
        tokens=[1, 2, 3, 4, 5, 6],
    )
    _upgrade_to_verified_v3(shorter)
    _upgrade_to_verified_v3(longer)

    def mutate_performance(rows):
        for row in rows[:2]:
            row["timing_seconds"]["round_total"] += 9.0
            row["hbm_bytes"]["allocated_end"] += 1 << 20
            row["update"]["backward_cuda_us"] = 9999.0
            row["update"]["optimizer_cuda_us"] = 9998.0
            row["update"]["update_cuda_us"] = 19997.0

    _rewrite_rounds(longer, mutate_performance)
    result = analysis.compare_common_prefix_trajectory(shorter, longer)
    assert result["status"] == "exact_common_prefix"
    assert result["exact_common_rounds"] == 2


def test_common_prefix_trajectory_rejects_optimizer_or_schema_drift(tmp_path):
    shorter = _write_run(
        tmp_path / "shorter-identity",
        mode="full-drafter",
        acceptance=[2, 2],
        tokens=[1, 2, 3, 4],
    )
    longer = _write_run(
        tmp_path / "longer-identity",
        mode="full-drafter",
        acceptance=[2, 2, 2],
        tokens=[1, 2, 3, 4, 5, 6],
    )
    _upgrade_to_verified_v3(shorter)
    _upgrade_to_verified_v3(longer)
    longer_summary_path = longer / "summary.json"
    longer_summary = json.loads(longer_summary_path.read_text())
    longer_summary["parameters"]["lr"] = 2e-4
    longer_summary_path.write_text(json.dumps(longer_summary))

    with pytest.raises(ValueError, match=r"trajectory identity mismatch.*lr"):
        analysis.compare_common_prefix_trajectory(shorter, longer)

    legacy = _write_run(
        tmp_path / "legacy",
        mode="full-drafter",
        acceptance=[2, 2, 2, 2],
        tokens=list(range(8)),
    )
    with pytest.raises(ValueError, match="requires schema_version=3"):
        analysis.compare_common_prefix_trajectory(shorter, legacy)


def test_common_prefix_cli_writes_auditable_result(tmp_path, capsys):
    shorter_run = tmp_path / "shorter-cli"
    longer_run = tmp_path / "longer-cli"
    shorter = _write_run(
        shorter_run / "artifact",
        mode="full-drafter",
        acceptance=[2, 2],
        tokens=[1, 2, 3, 4],
    )
    longer = _write_run(
        longer_run / "artifact",
        mode="full-drafter",
        acceptance=[2, 2, 2],
        tokens=[1, 2, 3, 4, 5, 6],
    )
    _upgrade_to_verified_v3(shorter)
    _upgrade_to_verified_v3(longer)
    output = tmp_path / "trajectory.json"

    assert analysis.main(
        [
            "--common-prefix",
            str(shorter_run),
            str(longer_run),
            "--output",
            str(output),
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed == written
    assert written["status"] == "exact_common_prefix"
