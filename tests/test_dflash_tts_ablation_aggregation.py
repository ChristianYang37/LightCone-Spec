from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "experiments"
    / "aggregate_dflash_tts_ablations.py"
)
SPEC = importlib.util.spec_from_file_location(
    "aggregate_dflash_tts_ablations", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
aggregation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aggregation
SPEC.loader.exec_module(aggregation)


def _write_run(
    root: Path,
    *,
    mode: str,
    acceptance: list[int],
    rank: int | None,
    learning_rate: float | None,
    peak_hbm_bytes: int,
    trainable_parameters: int,
    losses: list[float] | None = None,
    legacy_loss_alias: bool = False,
    draft_cache_policy: str = "stale",
) -> Path:
    root.mkdir()
    input_tokens = 2
    prefix = input_tokens
    rounds = []
    losses = losses or []
    for index, length in enumerate(acceptance):
        update = {
            "applied": mode != "static",
            "optimizer_step": index + 1 if mode != "static" else None,
            "loss": None,
            "distillation_kl": None,
            "proximal_kl": None,
            "grad_norm": None,
            "parameters_with_grad": 0 if mode == "static" else 2,
        }
        row = {
            "schema_version": 1,
            "sample_id": "sample-7",
            "round_index": index,
            "mode": mode,
            "prefix_length_before": prefix,
            "acceptance_length": length,
            "accepted_draft_tokens": length - 1,
            "committed_token_ids": list(range(prefix, prefix + length)),
            "update": update,
        }
        if mode != "static":
            loss = losses[index]
            if legacy_loss_alias:
                row["tts_loss"] = loss
            else:
                update.update(
                    {
                        "loss": loss,
                        "distillation_kl": loss * 0.8,
                        "proximal_kl": loss * 0.2,
                        "grad_norm": loss * 2.0,
                    }
                )
        rounds.append(row)
        prefix += length
    (root / "rounds.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rounds)
    )

    memory = None
    if mode != "static":
        memory = {
            "forward_parameter_bytes": trainable_parameters * 2,
            "master_parameter_bytes": trainable_parameters * 4,
            "master_gradient_bytes": trainable_parameters * 4,
            "optimizer_moment_bytes": trainable_parameters * 8,
        }
    parameters = {
        "seed": 0,
        "temperature": 0.0,
        "block_size": 16,
        "mask_token_id": 9,
        "stop_token_ids": None,
        "max_new_tokens": sum(acceptance),
        "draft_cache_policy": draft_cache_policy,
        "dtype": "bfloat16",
        "enable_thinking": True,
        "optimizer": None if mode == "static" else "ADAM",
        "lr": learning_rate,
        "weight_decay": 0.0,
        "rank": rank,
        "adapter_seed": 3 if rank is not None else None,
        "optimizer_memory_bytes": memory,
    }
    if legacy_loss_alias:
        parameters["adam_weight_decay"] = parameters.pop("weight_decay")
    summary = {
        "schema_version": 1,
        "status": "complete_reference_run",
        "mode": mode,
        "trainable_scope": "none_static" if mode == "static" else mode,
        "trainable_layout": {
            "parameter_count": trainable_parameters,
            "parameter_tensors": 0 if mode == "static" else 2,
            "layout_sha256": f"layout-{mode}-{rank}",
            "algorithm": "DFLASH",
            "mode": mode,
            "rank": rank,
            "adapter_seed": 3 if rank is not None else None,
            "parameters": (
                []
                if mode == "static"
                else [
                    {
                        "name": "tail.lora_a",
                        "shape": [8, rank or 8],
                        "numel": trainable_parameters // 2,
                        "forward_dtype": "torch.bfloat16",
                        "master_dtype": "torch.float32",
                    },
                    {
                        "name": "tail.lora_b",
                        "shape": [rank or 8, 8],
                        "numel": trainable_parameters
                        - trainable_parameters // 2,
                        "forward_dtype": "torch.bfloat16",
                        "master_dtype": "torch.float32",
                    },
                ]
            ),
        },
        "reference": {"source_sha256": "reference-source"},
        "models": {
            "target": {
                "declared_revision": "target-revision",
                "config.json_sha256": "target-config",
            },
            "draft": {
                "declared_revision": "draft-revision",
                "config.json_sha256": "draft-config",
            },
        },
        "dataset": {
            "sha256": "dataset-sha",
            "declared_revision": "dataset-revision",
            "sample_index": 7,
            "sample_id": "sample-7",
        },
        "parameters": parameters,
        "generation": {
            "num_input_tokens": input_tokens,
            "num_output_tokens": sum(acceptance),
            "rounds": len(rounds),
            "optimizer_steps": 0 if mode == "static" else len(rounds),
            "decode_seconds": float(len(rounds)),
            "peak_hbm_bytes": peak_hbm_bytes,
            "trainable_parameter_count": trainable_parameters,
            "optimizer_memory_bytes": memory,
        },
        "output": {"token_ids": [10, 11, 12, 13]},
    }
    (root / "summary.json").write_text(json.dumps(summary))
    return root


def test_aggregation_records_loss_memory_parameters_and_exact_pair(tmp_path):
    static = _write_run(
        tmp_path / "static",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    rank2 = _write_run(
        tmp_path / "rank2",
        mode="tail-lora",
        acceptance=[4],
        rank=2,
        learning_rate=1e-3,
        peak_hbm_bytes=120,
        trainable_parameters=10,
        losses=[1.5],
    )
    rank4 = _write_run(
        tmp_path / "rank4",
        mode="tail-lora",
        acceptance=[2, 2],
        rank=4,
        learning_rate=1e-3,
        peak_hbm_bytes=140,
        trainable_parameters=20,
        losses=[0.5, 0.25],
        legacy_loss_alias=True,
    )

    rows = aggregation.build_long_table(
        [rank4, static, rank2], bucket_size=8
    )
    by_rank = {
        row["rank"]: row for row in rows if row["mode"] == "tail-lora"
    }

    assert by_rank[2]["prefix_len_observed_min"] == 2
    assert by_rank[2]["prefix_bucket_start"] == 0
    assert by_rank[2]["paper_acceptance_length"] == 4.0
    assert by_rank[2]["paper_acceptance_length_gain_vs_static"] == 2.0
    assert by_rank[2]["exact_output_token_match_static"] is True
    assert by_rank[2]["loss_mean"] == 1.5
    assert by_rank[2]["loss_median"] == 1.5
    assert by_rank[2]["loss_finite_count"] == 1
    assert by_rank[2]["peak_hbm_bytes"] == 120
    assert by_rank[2]["peak_hbm_over_static_bytes"] is None
    assert by_rank[2]["hbm_pairing_status"].startswith("pilot_descriptive")
    assert by_rank[2]["optimizer_resident_bytes"] == 140
    assert by_rank[2]["optimizer_update_peak_bytes"] == 180
    assert (
        by_rank[2]["optimizer_memory_evidence"]
        == "legacy_lower_bound_missing_forward_gradient"
    )
    assert by_rank[2]["optimizer_bytes_per_trainable_parameter"] == 14.0
    assert by_rank[2]["trainable_parameter_count"] == 10
    assert by_rank[2]["gradient_parameter_tensors_min"] == 2
    assert by_rank[2]["gradient_parameter_tensors_max"] == 2
    assert len(by_rank[2]["summary_sha256"]) == 64
    assert len(by_rank[2]["rounds_sha256"]) == 64
    assert len(by_rank[2]["source_identity_sha256"]) == 64

    assert by_rank[4]["optimizer"] == "adam"
    assert by_rank[4]["weight_decay"] == 0.0
    assert by_rank[4]["loss_mean"] == pytest.approx(0.375)
    assert by_rank[4]["loss_median"] == pytest.approx(0.375)
    assert by_rank[4]["loss_source_fields"] == "tts_loss"

    tables = aggregation.build_ablation_tables(rows)
    assert len(tables["lr_ablation"]) == 2
    assert {row["rank"] for row in tables["lora_rank_ablation"]} == {2, 4}
    assert len(
        {row["rank_comparison_key"] for row in tables["lora_rank_ablation"]}
    ) == 1
    pareto = {row["rank"]: row for row in tables["memory_al_pareto"]}
    assert pareto[2]["pareto_optimal"] is None
    assert pareto[4]["pareto_optimal"] is None
    assert pareto[2]["pareto_eligible"] is False
    assert pareto[2]["pareto_acceptance_basis"].startswith("run_level_exact")
    assert "runtime_fingerprint_not_comparable" in pareto[2]["pareto_status"]


def test_writer_emits_deterministic_evidence_tables(tmp_path):
    static = _write_run(
        tmp_path / "static",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    candidate = _write_run(
        tmp_path / "candidate",
        mode="drafter-lora",
        acceptance=[4],
        rank=8,
        learning_rate=1e-3,
        peak_hbm_bytes=130,
        trainable_parameters=30,
        losses=[0.75],
    )
    rows = aggregation.build_long_table([static, candidate], bucket_size=8)
    tables = aggregation.build_ablation_tables(rows)

    manifest = aggregation.write_ablation_tables(
        tmp_path / "tables", tables, bucket_size=8
    )
    reversed_tables = aggregation.build_ablation_tables(list(reversed(rows)))
    aggregation.write_ablation_tables(
        tmp_path / "tables-reversed", reversed_tables, bucket_size=8
    )

    output = Path(manifest["output_dir"])
    assert (output / "dflash_tts_long.csv").is_file()
    assert (output / "dflash_tts_lr_ablation.json").is_file()
    assert (output / "dflash_tts_memory_al_pareto.csv").is_file()
    assert (output / "dflash_tts_lora_rank_ablation.json").is_file()
    payload = json.loads((output / "dflash_tts_long.json").read_text())
    assert payload["bucket_size"] == 8
    assert len(payload["rows"]) == 2
    manifest_payload = json.loads(
        (output / "dflash_tts_ablation_manifest.json").read_text()
    )
    assert manifest_payload["bucket_semantics"] == "observed_prefix_length_before"
    assert manifest_payload["tables"]["long"]["rows"] == 2
    assert all(len(value) == 64 for value in manifest_payload["files"].values())
    for name in (
        "long",
        "lr_ablation",
        "memory_al_pareto",
        "lora_rank_ablation",
    ):
        assert (output / f"dflash_tts_{name}.json").read_bytes() == (
            Path(tmp_path / "tables-reversed") / f"dflash_tts_{name}.json"
        ).read_bytes()


def test_aggregation_fails_closed_on_missing_core_identity(tmp_path):
    run = _write_run(
        tmp_path / "candidate",
        mode="full-drafter",
        acceptance=[2, 2],
        rank=None,
        learning_rate=1e-4,
        peak_hbm_bytes=200,
        trainable_parameters=100,
        losses=[1.0, 0.5],
    )
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    del summary["dataset"]["sha256"]
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="dataset.sha256"):
        aggregation.build_long_table([run], bucket_size=8)


def test_schema_v2_hbm_parameter_drift_and_round_hash_are_preserved(tmp_path):
    run = _write_run(
        tmp_path / "candidate",
        mode="full-drafter",
        acceptance=[2, 2],
        rank=None,
        learning_rate=1e-4,
        peak_hbm_bytes=240,
        trainable_parameters=100,
        losses=[1.0, 0.5],
    )
    rounds_path = run / "rounds.jsonl"
    rounds = [json.loads(line) for line in rounds_path.read_text().splitlines()]
    for index, row in enumerate(rounds):
        row["schema_version"] = 2
        row["prefix_len_before"] = row.pop("prefix_length_before")
        row["hbm_bytes"] = {
            "allocated_end": 200 + index,
            "reserved_end": 300 + index,
            "running_peak_allocated": 220 + index,
            "running_peak_reserved": 320 + index,
        }
        row["update"].update(
            {
                "backward_cuda_us": 10.0 + index,
                "optimizer_cuda_us": 20.0 + index,
                "update_cuda_us": 30.0 + index,
                "parameter_delta_l2": 0.1 + index * 0.1,
                "parameter_displacement_l2": 0.2 + index * 0.2,
                "parameter_l2": 5.0 + index,
                "relative_parameter_delta": 0.01 + index * 0.01,
                "parameter_audit_interval_steps": 1,
            }
        )
    rounds_path.write_text("".join(json.dumps(row) + "\n" for row in rounds))
    rounds_hash = hashlib.sha256(rounds_path.read_bytes()).hexdigest()
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = 2
    summary["harness"] = {
        "source_sha256": "harness-source",
        "artifact_schema_version": 2,
    }
    summary["generation"].update(
        {
            "peak_hbm_reserved_bytes": 340,
            "hbm_bytes": {
                "allocated_end": 210,
                "reserved_end": 310,
                "running_peak_allocated": 240,
                "running_peak_reserved": 340,
            },
            "parameter_layout_sha256": "runtime-layout",
        }
    )
    summary["hbm_bytes"] = {
        phase: {
            "allocated_end": allocated,
            "reserved_end": reserved,
            "running_peak_allocated": allocated + 5,
            "running_peak_reserved": reserved + 5,
        }
        for phase, allocated, reserved in (
            ("after_model_load", 100, 150),
            ("after_adapter", 120, 170),
            ("after_optimizer", 180, 230),
            ("after_run", 210, 310),
        )
    }
    memory = summary["generation"]["optimizer_memory_bytes"]
    memory.update(
        {
            "forward_gradient_bytes": 200,
            "persistent_bytes": 1400,
            "estimated_update_peak_bytes": 2000,
            "total_bytes": 2000,
            "parameter_audit_cpu_snapshot_bytes": 800,
        }
    )
    summary["parameters"]["optimizer_memory_bytes"] = dict(memory)
    summary["output"]["rounds_jsonl"] = "rounds.jsonl"
    summary["output"]["rounds_sha256"] = rounds_hash
    for row in rounds:
        row["trainable_parameter_count"] = 100
        row["parameter_layout_sha256"] = "runtime-layout"
        row["draft_cache_policy"] = "stale"
        row["provenance"] = {
            "reference_source_sha256": "reference-source",
            "target_declared_revision": "target-revision",
            "draft_declared_revision": "draft-revision",
            "dataset_declared_revision": "dataset-revision",
            "dataset_sha256": "dataset-sha",
            "harness_source_sha256": "harness-source",
        }
    rounds_path.write_text("".join(json.dumps(row) + "\n" for row in rounds))
    rounds_hash = hashlib.sha256(rounds_path.read_bytes()).hexdigest()
    summary["output"]["rounds_sha256"] = rounds_hash
    summary_path.write_text(json.dumps(summary))

    [row] = aggregation.build_long_table([run], bucket_size=8)

    assert row["harness_source_sha256"] == "harness-source"
    assert row["peak_hbm_bytes"] == 240
    assert row["peak_hbm_reserved_bytes"] == 340
    assert row["hbm_allocated_end_max_bytes"] == 201
    assert row["hbm_running_peak_reserved_max_bytes"] == 321
    assert row["hbm_after_model_load_allocated_end_bytes"] == 100
    assert row["hbm_adapter_allocated_end_delta_bytes"] == 20
    assert row["hbm_optimizer_allocated_end_delta_bytes"] == 60
    assert row["hbm_run_reserved_end_delta_bytes"] == 80
    assert row["optimizer_resident_bytes"] == 1400
    assert row["optimizer_update_peak_bytes"] == 2000
    assert row["optimizer_memory_evidence"] == (
        "exact_declared_optimizer_tensor_ledger"
    )
    assert row["whole_process_peak_hbm_bytes"] == 215
    assert row["backward_cuda_us_mean"] == pytest.approx(10.5)
    assert row["parameter_delta_l2_mean"] == pytest.approx(0.15)
    assert row["parameter_displacement_l2_last"] == pytest.approx(0.4)
    assert row["relative_parameter_delta_max"] == pytest.approx(0.02)
    assert row["parameter_audit_interval_steps_sum"] == 2.0
    [pareto] = aggregation.build_ablation_tables([row])["memory_al_pareto"]
    assert pareto["pareto_eligible"] is False
    assert pareto["pareto_optimal"] is None


def test_cache_policy_runs_use_policy_matched_static_baselines(tmp_path):
    static_stale = _write_run(
        tmp_path / "static-stale",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
        draft_cache_policy="stale",
    )
    static_rebuild = _write_run(
        tmp_path / "static-rebuild",
        mode="static",
        acceptance=[1, 3],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=101,
        trainable_parameters=0,
        draft_cache_policy="rebuild",
    )
    adaptive_stale = _write_run(
        tmp_path / "adaptive-stale",
        mode="tail-lora",
        acceptance=[4],
        rank=2,
        learning_rate=1e-3,
        peak_hbm_bytes=120,
        trainable_parameters=10,
        losses=[1.0],
        draft_cache_policy="stale",
    )
    adaptive_rebuild = _write_run(
        tmp_path / "adaptive-rebuild",
        mode="tail-lora",
        acceptance=[2, 2],
        rank=2,
        learning_rate=1e-3,
        peak_hbm_bytes=121,
        trainable_parameters=10,
        losses=[1.0, 0.9],
        draft_cache_policy="rebuild",
    )

    rows = aggregation.build_long_table(
        [static_stale, static_rebuild, adaptive_stale, adaptive_rebuild],
        bucket_size=8,
    )
    adapted = {
        row["draft_cache_policy"]: row
        for row in rows
        if row["mode"] == "tail-lora"
    }
    assert adapted["stale"]["paired_static_summary_sha256"] == hashlib.sha256(
        (static_stale / "summary.json").read_bytes()
    ).hexdigest()
    assert adapted["rebuild"]["paired_static_summary_sha256"] == hashlib.sha256(
        (static_rebuild / "summary.json").read_bytes()
    ).hexdigest()
    assert adapted["stale"]["run_paper_acceptance_length_gain_vs_static"] == 2.0
    assert adapted["rebuild"]["run_paper_acceptance_length_gain_vs_static"] == 0.0


def test_final_round_is_clipped_to_effective_output_tokens(tmp_path):
    run = _write_run(
        tmp_path / "static",
        mode="static",
        acceptance=[4, 4],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["generation"]["num_output_tokens"] = 6
    summary["parameters"]["max_new_tokens"] = 6
    summary["output"]["token_ids"] = [10, 11, 12, 13, 14, 15]
    summary_path.write_text(json.dumps(summary))

    [row] = aggregation.build_long_table([run], bucket_size=8)

    assert row["algorithmic_committed_tokens"] == 8
    assert row["committed_output_tokens"] == 6
    assert row["target_calls_per_output_token"] == pytest.approx(2 / 6)
    assert row["logical_block_target_calls_per_output_token"] == pytest.approx(
        2 / 6
    )


def test_reference_audit_physical_target_calls_are_separate_from_logical_calls(
    tmp_path,
):
    run = _write_run(
        tmp_path / "canonical-audit-calls",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["generation"]["target_calls"] = {
        "block_prefill": 1,
        "canonical_prefill": 1,
        "block_verify_decode": 2,
        "canonical_commit_verify_decode": 4,
        "physical_total": 8,
    }
    summary_path.write_text(json.dumps(summary))
    rounds_path = run / "rounds.jsonl"
    rounds = [json.loads(line) for line in rounds_path.read_text().splitlines()]
    for row in rounds:
        row["target_calls"] = {
            "block_verify": 1,
            "canonical_commit_verify": 2,
            "physical_total": 3,
        }
    rounds_path.write_text("\n".join(json.dumps(row) for row in rounds) + "\n")

    [row] = aggregation.build_long_table([run], bucket_size=8)

    assert row["target_calls_per_output_token"] == pytest.approx(0.5)
    assert row["logical_block_target_calls_per_output_token"] == pytest.approx(
        0.5
    )
    assert row["physical_target_calls_per_output_token"] == pytest.approx(1.5)
    assert row["canonical_audit_target_calls_per_output_token"] == pytest.approx(
        1.0
    )
    assert row["run_physical_target_calls_per_output_token"] == pytest.approx(2.0)
    assert row[
        "run_canonical_audit_target_calls_per_output_token"
    ] == pytest.approx(1.25)
    assert "headline_algorithmic_metric" in row["target_call_metric_scope"]


def test_pairing_binds_model_artifact_and_exactness(tmp_path):
    static = _write_run(
        tmp_path / "static",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    candidate = _write_run(
        tmp_path / "candidate",
        mode="tail-lora",
        acceptance=[4],
        rank=2,
        learning_rate=1e-3,
        peak_hbm_bytes=120,
        trainable_parameters=10,
        losses=[1.0],
    )
    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["models"]["draft"]["config.json_sha256"] = "different-config"
    summary_path.write_text(json.dumps(summary))

    rows = aggregation.build_long_table([static, candidate], bucket_size=8)
    adapted = next(row for row in rows if row["mode"] == "tail-lora")
    assert adapted["exact_output_token_match_static"] is None
    assert adapted["paper_acceptance_length_gain_vs_static"] is None
    [pareto] = aggregation.build_ablation_tables(rows)["memory_al_pareto"]
    assert pareto["pareto_eligible"] is False


def test_pairing_ignores_injected_runtime_layout_but_binds_checkpoint_hashes(
    tmp_path,
):
    static = _write_run(
        tmp_path / "static-runtime-layout",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    candidate = _write_run(
        tmp_path / "drafter-lora-runtime-layout",
        mode="drafter-lora",
        acceptance=[4],
        rank=8,
        learning_rate=1e-3,
        peak_hbm_bytes=120,
        trainable_parameters=10,
        losses=[1.0],
    )
    for run, runtime_layout in (
        (static, "checkpoint-module-layout"),
        (candidate, "lora-injected-module-layout"),
    ):
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["models"]["draft"]["layout_sha256"] = runtime_layout
        summary["models"]["draft"]["content_identity_sha256"] = (
            "a" * 64
        )
        summary_path.write_text(json.dumps(summary))

    rows = aggregation.build_long_table([static, candidate], bucket_size=8)
    adapted = next(row for row in rows if row["mode"] == "drafter-lora")
    assert adapted["exact_output_token_match_static"] is True
    assert adapted["paper_acceptance_length_gain_vs_static"] == 2.0

    candidate_summary_path = candidate / "summary.json"
    candidate_summary = json.loads(candidate_summary_path.read_text())
    candidate_summary["models"]["draft"]["content_identity_sha256"] = (
        "b" * 64
    )
    candidate_summary_path.write_text(json.dumps(candidate_summary))
    changed_rows = aggregation.build_long_table(
        [static, candidate], bucket_size=8
    )
    changed = next(
        row for row in changed_rows if row["mode"] == "drafter-lora"
    )
    assert changed["exact_output_token_match_static"] is None
    assert changed["paper_acceptance_length_gain_vs_static"] is None


def test_non_exact_diagnostic_run_is_rejected_from_selection(tmp_path):
    diagnostic = _write_run(
        tmp_path / "non-exact-diagnostic",
        mode="tail-lora",
        acceptance=[2, 2],
        rank=8,
        learning_rate=1e-3,
        peak_hbm_bytes=120,
        trainable_parameters=10,
        losses=[1.0, 0.9],
    )
    summary_path = diagnostic / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "complete_non_exact_diagnostic_run"
    summary["parameters"]["canonical_greedy_verifier"] = False
    summary["exactness"] = {
        "classification": "non_exact_diagnostic",
        "selection_eligible": False,
        "canonical_commit_verifier": False,
    }
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="run is not complete_reference_run"):
        aggregation.build_long_table([diagnostic], bucket_size=8)


def test_rejects_false_static_and_inconsistent_memory(tmp_path):
    static = _write_run(
        tmp_path / "static",
        mode="static",
        acceptance=[2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    rounds_path = static / "rounds.jsonl"
    row = json.loads(rounds_path.read_text())
    row["update"].update(
        {"applied": True, "optimizer_step": 1, "loss": 1.0, "parameters_with_grad": 1}
    )
    rounds_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="static update evidence"):
        aggregation.build_long_table([static], bucket_size=8)

    candidate = _write_run(
        tmp_path / "candidate",
        mode="full-drafter",
        acceptance=[2],
        rank=None,
        learning_rate=1e-4,
        peak_hbm_bytes=200,
        trainable_parameters=100,
        losses=[1.0],
    )
    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text())
    memory = summary["generation"]["optimizer_memory_bytes"]
    memory.update(
        {
            "forward_gradient_bytes": 200,
            "persistent_bytes": 999,
            "estimated_update_peak_bytes": 1999,
            "total_bytes": 1999,
            "parameter_audit_cpu_snapshot_bytes": 0,
        }
    )
    summary["parameters"]["optimizer_memory_bytes"] = dict(memory)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="optimizer memory identity"):
        aggregation.build_long_table([candidate], bucket_size=8)


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
        ]
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
        "resolved_device": "cuda:0",
        "allocator_config": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_ALLOC_CONF": None,
        },
        "cuda_visible_devices": None,
        "deterministic_algorithms": False,
        "deterministic_warn_only": False,
        "allow_tf32": {"matmul": True, "cudnn": True},
        "float32_matmul_precision": "high",
        "cudnn_benchmark": False,
        "gpu": {
            "name": "RTX PRO 6000 Blackwell",
            "total_memory_bytes": 96 << 30,
            "compute_capability": "12.0",
            "device_index": 0,
        },
    }
    run_peak = summary["generation"]["peak_hbm_bytes"]
    summary["hbm_bytes"] = {
        "after_model_load": {
            "allocated_end": max(run_peak - 30, 0),
            "reserved_end": max(run_peak - 20, 0),
            "running_peak_allocated": max(run_peak - 20, 0),
            "running_peak_reserved": max(run_peak - 10, 0),
        },
        "after_adapter": {
            "allocated_end": max(run_peak - 20, 0),
            "reserved_end": max(run_peak - 10, 0),
            "running_peak_allocated": max(run_peak - 10, 0),
            "running_peak_reserved": run_peak,
        },
        "after_optimizer": {
            "allocated_end": max(run_peak - 10, 0),
            "reserved_end": run_peak,
            "running_peak_allocated": run_peak,
            "running_peak_reserved": run_peak + 10,
        },
        "after_run": {
            "allocated_end": run_peak,
            "reserved_end": run_peak + 10,
            "running_peak_allocated": run_peak,
            "running_peak_reserved": run_peak + 10,
        },
    }
    runtime_layout = f"runtime-{summary['mode']}-{summary['parameters']['rank']}"
    summary["generation"]["parameter_layout_sha256"] = runtime_layout
    summary["output"]["rounds_jsonl"] = "rounds.jsonl"
    for row in rounds:
        row["schema_version"] = 3
        row["trainable_parameter_count"] = summary["generation"][
            "trainable_parameter_count"
        ]
        row["parameter_layout_sha256"] = runtime_layout
        row["draft_cache_policy"] = "stale"
        row["provenance"] = {
            "reference_source_sha256": "reference-source",
            "target_declared_revision": "target-revision",
            "draft_declared_revision": "draft-revision",
            "dataset_declared_revision": "dataset-revision",
            "dataset_sha256": "dataset-sha",
            "harness_source_sha256": "b" * 64,
        }
    rounds_path.write_text("".join(json.dumps(row) + "\n" for row in rounds))
    summary["output"]["rounds_sha256"] = hashlib.sha256(
        rounds_path.read_bytes()
    ).hexdigest()
    summary_path.write_text(json.dumps(summary))


def test_formal_pareto_requires_verified_identity_and_runtime_fingerprint(tmp_path):
    static = _write_run(
        tmp_path / "static-v3",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    efficient = _write_run(
        tmp_path / "efficient-v3",
        mode="tail-lora",
        acceptance=[4],
        rank=2,
        learning_rate=1e-3,
        peak_hbm_bytes=120,
        trainable_parameters=10,
        losses=[1.0],
    )
    dominated = _write_run(
        tmp_path / "dominated-v3",
        mode="tail-lora",
        acceptance=[2, 2],
        rank=4,
        learning_rate=1e-3,
        peak_hbm_bytes=140,
        trainable_parameters=20,
        losses=[1.0, 0.5],
    )
    for run in (static, efficient, dominated):
        _upgrade_to_verified_v3(run)

    tables = aggregation.build_ablation_tables(
        aggregation.build_long_table([static, efficient, dominated], bucket_size=8)
    )
    pareto = {row["rank"]: row for row in tables["memory_al_pareto"]}
    assert pareto[2]["pareto_eligible"] is True
    assert pareto[2]["pareto_optimal"] is True
    assert pareto[2]["peak_hbm_over_static_bytes"] == 20
    assert pareto[4]["pareto_optimal"] is False
    assert pareto[2]["prefix_bucket_start"] is None
    assert pareto[2]["pareto_scope"] == "run_level_exact_output_pair"


def test_hbm_pairing_fails_closed_on_runtime_fingerprint_mismatch(tmp_path):
    static = _write_run(
        tmp_path / "static-runtime",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    candidate = _write_run(
        tmp_path / "candidate-runtime",
        mode="tail-lora",
        acceptance=[4],
        rank=2,
        learning_rate=1e-3,
        peak_hbm_bytes=120,
        trainable_parameters=10,
        losses=[1.0],
    )
    for run in (static, candidate):
        _upgrade_to_verified_v3(run)
    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["runtime_fingerprint"]["gpu"]["total_memory_bytes"] -= 1
    summary_path.write_text(json.dumps(summary))

    rows = aggregation.build_long_table([static, candidate], bucket_size=8)
    adapted = next(row for row in rows if row["mode"] == "tail-lora")
    assert adapted["paper_acceptance_length_gain_vs_static"] == 2.0
    assert adapted["peak_hbm_over_static_bytes"] is None
    assert adapted["hbm_pairing_status"] == (
        "unavailable_runtime_fingerprint_mismatch"
    )
    [pareto] = aggregation.build_ablation_tables(rows)["memory_al_pareto"]
    assert pareto["pareto_optimal"] is None
    assert "runtime_fingerprint_not_comparable" in pareto["pareto_status"]


def test_numeric_runtime_mismatch_prevents_acceptance_pairing(tmp_path):
    static = _write_run(
        tmp_path / "static-numeric-runtime",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    candidate = _write_run(
        tmp_path / "candidate-numeric-runtime",
        mode="tail-lora",
        acceptance=[4],
        rank=2,
        learning_rate=1e-3,
        peak_hbm_bytes=120,
        trainable_parameters=10,
        losses=[1.0],
    )
    for run in (static, candidate):
        _upgrade_to_verified_v3(run)
    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["runtime_fingerprint"]["allow_tf32"]["matmul"] = False
    summary_path.write_text(json.dumps(summary))

    rows = aggregation.build_long_table([static, candidate], bucket_size=8)
    adapted = next(row for row in rows if row["mode"] == "tail-lora")
    assert adapted["numeric_runtime_status"] == "complete_numeric_runtime_v1"
    assert adapted["exact_output_token_match_static"] is None
    assert adapted["paper_acceptance_length_gain_vs_static"] is None
    assert adapted["run_gain_status"] == "unavailable_no_static_baseline"


def test_unknown_cuda_driver_keeps_v3_readable_but_unpaired(tmp_path):
    static = _write_run(
        tmp_path / "static-null-driver",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    candidate = _write_run(
        tmp_path / "candidate-null-driver",
        mode="tail-lora",
        acceptance=[4],
        rank=2,
        learning_rate=1e-3,
        peak_hbm_bytes=120,
        trainable_parameters=10,
        losses=[1.0],
    )
    for run in (static, candidate):
        _upgrade_to_verified_v3(run)
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["runtime_fingerprint"]["cuda_driver_version"] = None
        summary_path.write_text(json.dumps(summary))

    rows = aggregation.build_long_table([static, candidate], bucket_size=8)
    adapted = next(row for row in rows if row["mode"] == "tail-lora")
    assert adapted["numeric_runtime_status"].startswith(
        "incomplete_numeric_runtime_missing_"
    )
    assert adapted["runtime_fingerprint_status"].startswith(
        "incomplete_missing_"
    )
    assert adapted["paper_acceptance_length_gain_vs_static"] is None
    assert adapted["run_gain_status"] == "unavailable_no_static_baseline"


def test_lr_and_rank_comparison_keys_bind_correct_layout_identity(tmp_path):
    run2 = _write_run(
        tmp_path / "rank2-layout",
        mode="tail-lora",
        acceptance=[4],
        rank=2,
        learning_rate=1e-3,
        peak_hbm_bytes=120,
        trainable_parameters=10,
        losses=[1.0],
    )
    run4 = _write_run(
        tmp_path / "rank4-layout",
        mode="tail-lora",
        acceptance=[4],
        rank=4,
        learning_rate=1e-3,
        peak_hbm_bytes=130,
        trainable_parameters=20,
        losses=[1.0],
    )
    for run, runtime_layout in ((run2, "runtime-r2"), (run4, "runtime-r4")):
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["generation"]["parameter_layout_sha256"] = runtime_layout
        summary_path.write_text(json.dumps(summary))
    rows = aggregation.build_long_table([run2, run4], bucket_size=8)
    tables = aggregation.build_ablation_tables(rows)
    lr_keys = {row["learning_rate_comparison_key"] for row in tables["lr_ablation"]}
    rank_keys = {row["rank_comparison_key"] for row in tables["lora_rank_ablation"]}
    assert len(lr_keys) == 2
    assert None not in lr_keys
    assert len(rank_keys) == 1
    assert None not in rank_keys

    changed = dict(rows[0])
    changed["projection_artifact_sha256"] = "f" * 64
    changed["mode"] = "output-residual"
    first, first_status = aggregation._ablation_comparison_key(
        changed, axis="learning_rate"
    )
    changed["projection_artifact_sha256"] = "a" * 64
    second, second_status = aggregation._ablation_comparison_key(
        changed, axis="learning_rate"
    )
    assert first_status == second_status == "comparable_exact_parameter_layout"
    assert first != second


def test_rendered_input_token_change_prevents_pairing_and_v1_is_unverified(tmp_path):
    legacy = _write_run(
        tmp_path / "legacy-static",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    [legacy_row] = aggregation.build_long_table([legacy], bucket_size=8)
    assert legacy_row["identity_verification_status"] == "legacy_unverified"

    static = _write_run(
        tmp_path / "verified-static",
        mode="static",
        acceptance=[2, 2],
        rank=None,
        learning_rate=None,
        peak_hbm_bytes=100,
        trainable_parameters=0,
    )
    candidate = _write_run(
        tmp_path / "verified-candidate",
        mode="tail-lora",
        acceptance=[4],
        rank=2,
        learning_rate=1e-3,
        peak_hbm_bytes=120,
        trainable_parameters=10,
        losses=[1.0],
    )
    for run in (static, candidate):
        _upgrade_to_verified_v3(run)
    candidate_summary_path = candidate / "summary.json"
    candidate_summary = json.loads(candidate_summary_path.read_text())
    candidate_summary["dataset"]["rendered_input_token_ids"]["sha256"] = (
        "f" * 64
    )
    candidate_summary_path.write_text(json.dumps(candidate_summary))

    rows = aggregation.build_long_table([static, candidate], bucket_size=8)
    adapted = next(row for row in rows if row["mode"] == "tail-lora")
    assert adapted["exact_output_token_match_static"] is None
    assert adapted["paper_acceptance_length_gain_vs_static"] is None
