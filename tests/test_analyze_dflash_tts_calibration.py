from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts" / "experiments"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner_test = _load(
    "_schema_v3_calibration_runner_test_helpers",
    Path(__file__).with_name("test_run_dflash_tts_calibration_sweep.py"),
)
analysis = _load(
    "analyze_dflash_tts_calibration",
    SCRIPTS / "analyze_dflash_tts_calibration.py",
)


def _candidates() -> list[dict]:
    return [
        {
            "candidate_id": "static",
            "mode": "static",
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "rank": None,
            "draft_cache_policy": "stale",
            "diagnostic_kind": "selection",
            "parameter_audit_stride": 0,
        },
        {
            "candidate_id": "tail-r16-adam-lr1e-4",
            "mode": "tail-lora",
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "rank": 16,
            "draft_cache_policy": "stale",
            "diagnostic_kind": "selection",
            "parameter_audit_stride": 0,
        },
        {
            "candidate_id": "tail-r16-adam-lr2e-4-unsafe",
            "mode": "tail-lora",
            "optimizer": "adam",
            "learning_rate": 2e-4,
            "weight_decay": 0.0,
            "rank": 16,
            "draft_cache_policy": "stale",
            "diagnostic_kind": "selection",
            "parameter_audit_stride": 0,
        },
        {
            "candidate_id": "tail-r16-adamw-lr3e-4",
            "mode": "tail-lora",
            "optimizer": "adamw",
            "learning_rate": 3e-4,
            "weight_decay": 1e-2,
            "rank": 16,
            "draft_cache_policy": "stale",
            "diagnostic_kind": "selection",
            "parameter_audit_stride": 0,
        },
    ]


def _plans(tmp_path: Path, *, candidate_newline: bytes = b"\n"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    argv, candidate_spec = runner_test._base_argv(tmp_path)
    payload = runner_test._candidate_spec()
    payload["candidates"] = _candidates()
    candidate_spec.write_bytes(
        json.dumps(payload, sort_keys=True).encode("utf-8") + candidate_newline
    )
    args = analysis.calibration.build_parser().parse_args(argv)
    plans = analysis.calibration.build_run_plans(args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True)
    candidate_copy = output_root / "candidate-specification.json"
    candidate_copy.write_bytes(candidate_spec.read_bytes())
    return candidate_copy, output_root, plans


def _partition(total: int, rounds: int) -> list[int]:
    quotient, remainder = divmod(total, rounds)
    assert 1 <= quotient <= 16 and quotient + bool(remainder) <= 16
    return [quotient + 1] * remainder + [quotient] * (rounds - remainder)


def _write_artifact(
    plan,
    *,
    verification_calls: int,
    peak_hbm: int,
    trainable_parameters: int,
    output_suffix: int = 0,
) -> None:
    runner_test._write_completed_run(plan)
    frozen = analysis.calibration.frozen
    identity = plan.identity
    mode = identity["mode"]
    generation_identity = identity["generation"]
    input_tokens = generation_identity["input_tokens"]
    acceptance = _partition(2048, verification_calls)
    layout_sha256 = hashlib.sha256(
        f"{mode}:{identity['optimization']['rank']}".encode()
    ).hexdigest()
    sample_id = f"math500:sample_index={identity['dataset']['sample_index']}"
    prefix = input_tokens
    rounds = []
    for index, length in enumerate(acceptance):
        applied = mode != "static"
        update = {
            "applied": applied,
            "optimizer_step": index + 1 if applied else None,
            "loss": 1.0 / (index + 1) if applied else None,
            "distillation_kl": 0.8 / (index + 1) if applied else None,
            "proximal_kl": 0.2 / (index + 1) if applied else None,
            "grad_norm": 2.0 / (index + 1) if applied else None,
            "parameters_with_grad": 2 if applied else 0,
            "parameters_without_grad": [],
            "backward_cuda_us": None,
            "optimizer_cuda_us": None,
            "update_cuda_us": None,
            "parameter_delta_l2": None,
            "parameter_displacement_l2": None,
            "parameter_l2": None,
            "relative_parameter_delta": None,
            "parameter_audit_interval_steps": None,
        }
        rounds.append(
            {
                "schema_version": 3,
                "sample_id": sample_id,
                "round_index": index,
                "seed": generation_identity["seed"],
                "provenance": {
                    "reference_source_sha256": identity["reference"]["source_sha256"],
                    "target_declared_revision": identity["target"]["revision"],
                    "draft_declared_revision": identity["draft"]["revision"],
                    "dataset_declared_revision": identity["dataset"]["revision"],
                    "dataset_sha256": identity["dataset"]["sha256"],
                    "harness_source_sha256": identity["runtime"]["harness"]["sha256"],
                },
                "mode": mode,
                "trainable_scope": "none_static" if mode == "static" else "tail_lora",
                "trainable_parameter_count": trainable_parameters,
                "parameter_layout_sha256": layout_sha256,
                "optimizer": (
                    None
                    if mode == "static"
                    else identity["optimization"]["optimizer"].upper()
                ),
                "draft_cache_policy": "stale",
                "prefix_len_before": prefix,
                "accepted_draft_tokens": length - 1,
                "acceptance_length": length,
                "committed_token_ids": [7] * length,
                "update": update,
            }
        )
        prefix += length

    rounds_path = plan.artifact_dir / "rounds.jsonl"
    rounds_path.write_text("".join(json.dumps(row) + "\n" for row in rounds))
    rounds_sha256 = hashlib.sha256(rounds_path.read_bytes()).hexdigest()
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    memory = None
    if mode != "static":
        memory = {
            "forward_parameter_bytes": trainable_parameters * 2,
            "master_parameter_bytes": trainable_parameters * 4,
            "forward_gradient_bytes": 0,
            "master_gradient_bytes": trainable_parameters * 4,
            "optimizer_moment_bytes": trainable_parameters * 8,
            "persistent_bytes": trainable_parameters * 14,
            "estimated_update_peak_bytes": trainable_parameters * 18,
            "total_bytes": trainable_parameters * 18,
            "parameter_audit_cpu_snapshot_bytes": 0,
        }
    summary.update(
        {
            "trainable_scope": "none_static" if mode == "static" else "tail_lora",
            "trainable_layout": {
                "parameter_count": trainable_parameters,
                "parameter_tensors": 0 if mode == "static" else 2,
                "layout_sha256": layout_sha256,
                "algorithm": "DFLASH",
                "mode": mode,
                "rank": identity["optimization"]["rank"],
                "adapter_seed": identity["optimization"]["adapter_seed"],
                "initialization": "synthetic_zero_b",
                "parameters": (
                    []
                    if mode == "static"
                    else [
                        {
                            "name": "tail.lora_a",
                            "shape": [8, 16],
                            "numel": trainable_parameters // 2,
                            "forward_dtype": "torch.bfloat16",
                            "master_dtype": "torch.float32",
                        },
                        {
                            "name": "tail.lora_b",
                            "shape": [16, 8],
                            "numel": trainable_parameters - trainable_parameters // 2,
                            "forward_dtype": "torch.bfloat16",
                            "master_dtype": "torch.float32",
                        },
                    ]
                ),
            },
            "reconstruction_status": {
                "gradient_semantics": "current_round_cache_safe_tail_only"
            },
            "hbm_bytes": {
                "after_model_load": {
                    "allocated_end": 80,
                    "reserved_end": 85,
                    "running_peak_allocated": 85,
                    "running_peak_reserved": 90,
                },
                "after_adapter": {
                    "allocated_end": 90,
                    "reserved_end": 95,
                    "running_peak_allocated": 95,
                    "running_peak_reserved": 100,
                },
                "after_optimizer": {
                    "allocated_end": peak_hbm - 10,
                    "reserved_end": peak_hbm,
                    "running_peak_allocated": peak_hbm,
                    "running_peak_reserved": peak_hbm + 10,
                },
                "after_run": {
                    "allocated_end": peak_hbm - 5,
                    "reserved_end": peak_hbm,
                    "running_peak_allocated": peak_hbm,
                    "running_peak_reserved": peak_hbm + 10,
                },
            },
        }
    )
    summary["dataset"].update(
        {
            "sample_id": sample_id,
            "input_format": "turns",
            "thinking_effective_via_chat_template": True,
        }
    )
    summary["generation"].update(
        {
            "rounds": len(rounds),
            "optimizer_steps": 0 if mode == "static" else len(rounds),
            "acceptance_lengths": acceptance,
            "trainable_parameter_count": trainable_parameters,
            "parameter_layout_sha256": layout_sha256,
            "peak_hbm_bytes": peak_hbm,
            "peak_hbm_reserved_bytes": peak_hbm + 10,
            "optimizer_memory_bytes": memory,
            "decode_seconds": float(len(rounds)),
            "target_calls": {
                "block_prefill": 1,
                "canonical_prefill": 1,
                "block_verify_decode": len(rounds),
                "canonical_commit_verify_decode": sum(acceptance),
                "physical_total": 2 + len(rounds) + sum(acceptance),
            },
        }
    )
    summary["parameters"]["optimizer_memory_bytes"] = memory
    output_tokens = [
        identity["dataset"]["sample_index"] % 1000
    ] * (input_tokens + 2048)
    output_tokens[-1] += output_suffix
    summary["output"].update(
        {
            "token_ids": output_tokens,
            "rounds_jsonl": "rounds.jsonl",
            "rounds_sha256": rounds_sha256,
        }
    )
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    completion = frozen._completion(
        plan,
        hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        rounds_sha256,
        identity["dataset"]["rendered_input_token_ids_sha256"],
        frozen._sha256_json(summary["runtime_fingerprint"]),
    )
    plan.completion_path.write_text(json.dumps(completion, sort_keys=True) + "\n")


def _rewrite_summary_and_completion(plan, summary: dict) -> None:
    frozen = analysis.calibration.frozen
    summary_path = plan.artifact_dir / "summary.json"
    summary_path.write_bytes(
        (json.dumps(summary, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )
    completion = frozen._completion(
        plan,
        hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        summary["output"]["rounds_sha256"],
        plan.identity["dataset"]["rendered_input_token_ids_sha256"],
        frozen._sha256_json(summary["runtime_fingerprint"]),
    )
    plan.completion_path.write_bytes(
        (json.dumps(completion, sort_keys=True) + "\n").encode("utf-8")
    )


def _complete_sweep(
    tmp_path: Path,
    *,
    inexact_candidate: str | None = None,
    all_unsafe: bool = False,
    candidate_newline: bytes = b"\n",
):
    candidate_spec, output_root, plans = _plans(
        tmp_path,
        candidate_newline=candidate_newline,
    )
    calls = {
        "static": {0: 1024, 419: 1024},
        "tail-r16-adam-lr1e-4": {0: 850, 419: 900},
        "tail-r16-adam-lr2e-4-unsafe": {0: 700, 419: 1100},
        "tail-r16-adamw-lr3e-4": {0: 750, 419: 800},
    }
    if all_unsafe:
        for candidate_id in calls:
            if candidate_id != "static":
                calls[candidate_id] = {0: 1100, 419: 1200}
    hbm = {
        "static": 100,
        "tail-r16-adam-lr1e-4": 120,
        "tail-r16-adam-lr2e-4-unsafe": 125,
        "tail-r16-adamw-lr3e-4": 130,
    }
    analysis.calibration.frozen._ensure_artifact_identity_lock(plans[0])
    for plan in plans:
        candidate_id = plan.identity["calibration_candidate"]["candidate_id"]
        sample = plan.identity["dataset"]["sample_index"]
        _write_artifact(
            plan,
            verification_calls=calls[candidate_id][sample],
            peak_hbm=hbm[candidate_id],
            trainable_parameters=0 if candidate_id == "static" else 128,
            output_suffix=int(candidate_id == inexact_candidate),
        )
    return candidate_spec, output_root, plans


def test_safe_local_selection_boundary_pareto_and_hashes(tmp_path: Path):
    candidate_spec, output_root, _plans_ = _complete_sweep(tmp_path)
    payload = analysis.build_analysis(
        candidate_spec=candidate_spec, output_root=output_root
    )

    [decision] = payload["selection_decisions"]
    assert decision["status"] == "local_grid_winner"
    assert decision["winner"]["candidate_id"] == "tail-r16-adamw-lr3e-4"
    assert decision["winner"]["learning_rate_boundary"]["at_group_boundary"] is True
    assert decision["global_optimum_claim"] is False

    rows = {row["candidate_id"]: row for row in payload["candidate_rows"]}
    static_samples = rows["static"]["sample_results"]
    assert all(
        result["accepted_drafts_per_verify_A"] == 1.0
        for result in static_samples
    )
    assert all(result["paper_acceptance_length"] == 2.0 for result in static_samples)
    unsafe = rows["tail-r16-adam-lr2e-4-unsafe"]
    assert unsafe["aggregate"]["evidence_eligible"] is True
    assert unsafe["aggregate"]["safe_for_selection"] is False
    assert unsafe["aggregate"]["worst_sample_paired_delta_A"] < 0
    assert unsafe["aggregate"]["all_losses_and_gradients_finite"] is True
    assert unsafe["aggregate"]["trainable_parameter_count"] == 128
    assert any(
        row["candidate_id"] == unsafe["candidate_id"]
        for row in payload["pareto"]["rows"]
    )
    assert payload["source_run_count"] == 8
    assert payload["source_artifact_count"] == 33
    assert "output_root" not in payload
    assert payload["candidate_specification"]["path"] == (
        "candidate-specification.json"
    )
    assert payload["artifact_identity_lock"]["path"] == (
        "artifact_identity_lock.json"
    )
    assert payload["source_artifact_set_sha256"] == (
        analysis.aggregation._sha256_json(
            {
                "artifact_identity_lock": payload["artifact_identity_lock"],
                "runs": payload["source_runs"],
            }
        )
    )
    selected = rows["tail-r16-adamw-lr3e-4"]
    selected_sample = selected["sample_results"][0]
    assert "run_root" not in selected_sample
    assert selected_sample["rounds_path"] == (
        "sample-0000/tail-r16-adamw-lr3e-4/artifact/rounds.jsonl"
    )
    assert analysis.aggregation._is_sha256(selected_sample["rounds_sha256"])
    assert selected_sample["adapter_seed"] == 0
    assert selected_sample["adapter_seed_valid_for_parameterization"] is True
    assert selected_sample["target_calls_per_output_token"] == pytest.approx(
        750 / 2048
    )
    assert selected_sample[
        "logical_block_target_calls_per_output_token"
    ] == pytest.approx(750 / 2048)
    assert selected_sample[
        "physical_target_calls_per_output_token"
    ] == pytest.approx((750 + 2050) / 2048)
    assert selected_sample[
        "canonical_audit_target_calls_per_output_token"
    ] == pytest.approx(2049 / 2048)
    assert selected_sample["target_call_metric_scope"] == (
        analysis.TARGET_CALL_METRIC_SCOPE
    )
    assert selected["adapter_seed"] == 0
    assert selected["aggregate"]["adapter_seed"] == 0
    assert rows["static"]["adapter_seed"] is None
    assert selected_sample["whole_process_peak_hbm_bytes"] == 130
    assert selected_sample["whole_process_peak_hbm_reserved_bytes"] == 140
    assert selected_sample["whole_process_peak_hbm_over_static_bytes"] == 30
    assert (
        selected_sample["whole_process_peak_hbm_reserved_over_static_bytes"]
        == 30
    )
    assert selected_sample["optimizer_persistent_bytes"] == 128 * 14
    assert selected_sample["optimizer_update_peak_bytes"] == 128 * 18
    assert (
        selected_sample["optimizer_persistent_bytes_per_trainable_parameter"]
        == 14.0
    )
    assert (
        selected_sample["optimizer_update_peak_bytes_per_trainable_parameter"]
        == 18.0
    )
    assert selected["aggregate"]["max_whole_process_peak_hbm_bytes"] == 130
    assert (
        selected["aggregate"]["max_whole_process_peak_hbm_reserved_bytes"]
        == 140
    )
    assert selected["aggregate"]["max_optimizer_persistent_bytes"] == 128 * 14
    assert selected["aggregate"]["max_optimizer_update_peak_bytes"] == 128 * 18
    assert selected["aggregate"][
        "mean_logical_block_target_calls_per_output_token"
    ] == pytest.approx(775 / 2048)
    assert selected["aggregate"][
        "mean_physical_target_calls_per_output_token"
    ] == pytest.approx((775 + 2050) / 2048)
    assert selected["aggregate"][
        "mean_canonical_audit_target_calls_per_output_token"
    ] == pytest.approx(2049 / 2048)
    assert payload["target_call_metric_scope"] == analysis.TARGET_CALL_METRIC_SCOPE
    selected_pareto = next(
        row
        for row in payload["pareto"]["rows"]
        if row["candidate_id"] == selected["candidate_id"]
    )
    assert selected_pareto[
        "mean_logical_block_target_calls_per_output_token"
    ] == pytest.approx(775 / 2048)
    assert selected_pareto["target_call_metric_scope"] == (
        analysis.TARGET_CALL_METRIC_SCOPE
    )
    assert set(payload["analysis_implementation"]) == {
        "analyzer",
        "metric_aggregator",
        "calibration_orchestrator",
        "frozen_run_validator",
    }
    assert all(
        analysis.aggregation._is_sha256(item["sha256"])
        for item in payload["analysis_implementation"].values()
    )
    assert set(payload["execution_implementation"]) == {
        "calibration_orchestrator",
        "frozen_run_validator",
        "harness",
    }
    assert all(
        analysis.aggregation._is_sha256(item["sha256"])
        for item in payload["execution_implementation"].values()
    )
    assert payload["execution_implementation_sha256"] == (
        analysis.aggregation._sha256_json(payload["execution_implementation"])
    )
    unsigned = dict(payload)
    observed = unsigned.pop("analysis_sha256")
    assert analysis.aggregation._sha256_json(unsigned) == observed


def test_candidate_aggregate_rejects_prompt_mixed_adapter_seed(tmp_path: Path):
    candidate_spec, output_root, _plans_ = _complete_sweep(tmp_path)
    payload = analysis.build_analysis(
        candidate_spec=candidate_spec,
        output_root=output_root,
    )
    original = next(
        row
        for row in payload["candidate_rows"]
        if row["candidate_id"] == "tail-r16-adamw-lr3e-4"
    )
    sample_results = deepcopy(original["sample_results"])
    sample_results[1]["adapter_seed"] = 7
    sweep = analysis.calibration.load_candidate_sweep(candidate_spec)
    candidate = next(
        item
        for item in sweep.candidates
        if item.candidate_id == "tail-r16-adamw-lr3e-4"
    )
    rebuilt = analysis._candidate_row(candidate, sample_results)
    assert rebuilt["adapter_seed"] is None
    assert rebuilt["aggregate"]["adapter_seed"] is None
    assert rebuilt["aggregate"]["evidence_eligible"] is False
    assert "adapter_seed_differs_by_sample" in rebuilt["aggregate"][
        "ineligibility_reasons"
    ]


def test_exact_output_is_a_selection_gate_but_row_is_preserved(tmp_path: Path):
    candidate_spec, output_root, _plans_ = _complete_sweep(
        tmp_path, inexact_candidate="tail-r16-adamw-lr3e-4"
    )
    payload = analysis.build_analysis(
        candidate_spec=candidate_spec, output_root=output_root
    )
    rows = {row["candidate_id"]: row for row in payload["candidate_rows"]}
    row = rows["tail-r16-adamw-lr3e-4"]
    assert row["aggregate"]["evidence_eligible"] is False
    assert any(
        reason.endswith("output_not_exact_static")
        for reason in row["aggregate"]["ineligibility_reasons"]
    )
    assert payload["selection_decisions"][0]["winner"]["candidate_id"] == (
        "tail-r16-adam-lr1e-4"
    )


def test_no_safe_selection_does_not_drop_descriptive_candidates(tmp_path: Path):
    candidate_spec, output_root, _plans_ = _complete_sweep(
        tmp_path, all_unsafe=True
    )
    payload = analysis.build_analysis(
        candidate_spec=candidate_spec, output_root=output_root
    )
    [decision] = payload["selection_decisions"]
    assert decision["status"] == "no_safe_selection"
    assert decision["winner"] is None
    assert decision["evidence_eligible_count"] == 3
    assert decision["safe_candidate_count"] == 0
    assert len(payload["pareto"]["rows"]) == 3


def test_nonfinite_gradient_is_recorded_and_excluded(tmp_path: Path):
    candidate_spec, output_root, plans = _complete_sweep(tmp_path)
    plan = next(
        item
        for item in plans
        if item.identity["dataset"]["sample_index"] == 0
        and item.identity["calibration_candidate"]["candidate_id"]
        == "tail-r16-adamw-lr3e-4"
    )
    rounds_path = plan.artifact_dir / "rounds.jsonl"
    rounds = [json.loads(line) for line in rounds_path.read_text().splitlines()]
    rounds[0]["update"]["grad_norm"] = float("nan")
    rounds_path.write_text("".join(json.dumps(row) + "\n" for row in rounds))
    rounds_sha256 = hashlib.sha256(rounds_path.read_bytes()).hexdigest()
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["output"]["rounds_sha256"] = rounds_sha256
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    frozen = analysis.calibration.frozen
    plan.completion_path.write_text(
        json.dumps(
            frozen._completion(
                plan,
                hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                rounds_sha256,
                plan.identity["dataset"]["rendered_input_token_ids_sha256"],
                frozen._sha256_json(summary["runtime_fingerprint"]),
            ),
            sort_keys=True,
        )
        + "\n"
    )

    payload = analysis.build_analysis(
        candidate_spec=candidate_spec, output_root=output_root
    )
    row = next(
        item
        for item in payload["candidate_rows"]
        if item["candidate_id"] == "tail-r16-adamw-lr3e-4"
    )
    sample = next(
        item for item in row["sample_results"] if item["sample_index"] == 0
    )
    assert sample["update"]["nonfinite_grad_norm_count"] == 1
    assert row["aggregate"]["evidence_eligible"] is False
    assert "sample_0_loss_or_grad_not_finite" in row["aggregate"][
        "ineligibility_reasons"
    ]


@pytest.mark.parametrize("tamper", ("completion", "spec_binding"))
def test_attestation_and_spec_tamper_fail_closed(tmp_path: Path, tamper: str):
    candidate_spec, output_root, plans = _complete_sweep(tmp_path)
    plan = plans[-1]
    if tamper == "completion":
        completion = json.loads(plan.completion_path.read_text())
        completion["summary_sha256"] = "0" * 64
        plan.completion_path.write_text(json.dumps(completion) + "\n")
        message = "completion record mismatch"
    else:
        identity = json.loads(plan.identity_path.read_text())
        identity["identity"]["calibration_candidate"]["candidate_index"] = 999
        plan.identity_path.write_text(json.dumps(identity) + "\n")
        message = "calibration candidate mismatch"
    with pytest.raises(ValueError, match=message):
        analysis.build_analysis(candidate_spec=candidate_spec, output_root=output_root)


def test_analysis_publication_is_no_clobber_and_checkable(tmp_path: Path):
    candidate_spec, output_root, _plans = _complete_sweep(tmp_path)
    output = output_root / "selection.json"
    argv = [
        "--candidate-spec",
        str(candidate_spec),
        "--output-root",
        str(output_root),
        "--output",
        str(output),
    ]
    assert analysis.main(argv) == 0
    original = output.read_bytes()
    rebuilt = analysis.build_analysis(
        candidate_spec=candidate_spec,
        output_root=output_root,
    )
    assert original == analysis._render_bytes(rebuilt)
    with pytest.raises(FileExistsError):
        analysis.main(argv)
    assert output.read_bytes() == original
    assert analysis.main(
        [
            "--candidate-spec",
            str(candidate_spec),
            "--output-root",
            str(output_root),
            "--check",
            str(output),
        ]
    ) == 0


def test_published_bundle_is_byte_stable_after_nonascii_relocation_and_crlf(
    tmp_path: Path,
):
    source = tmp_path / "source"
    candidate_spec, output_root, _plans = _complete_sweep(
        source,
        candidate_newline=b"\r\n",
    )
    assert candidate_spec.read_bytes().endswith(b"\r\n")
    output = output_root / "selection-analysis.json"
    assert analysis.main(
        [
            "--candidate-spec",
            str(candidate_spec),
            "--output-root",
            str(output_root),
            "--output",
            str(output),
        ]
    ) == 0
    original_analysis = output.read_bytes()
    original_spec = candidate_spec.read_bytes()

    relocated_root = tmp_path / "relocated-证据包" / "runs"
    relocated_root.parent.mkdir()
    shutil.copytree(output_root, relocated_root)
    shutil.rmtree(output_root)
    relocated_spec = relocated_root / candidate_spec.name
    relocated_analysis = relocated_root / output.name
    assert relocated_spec.read_bytes() == original_spec
    assert relocated_analysis.read_bytes() == original_analysis
    assert analysis.main(
        [
            "--candidate-spec",
            str(relocated_spec),
            "--output-root",
            str(relocated_root),
            "--check",
            str(relocated_analysis),
        ]
    ) == 0
    assert relocated_analysis.read_bytes() == original_analysis


def test_candidate_copy_and_analysis_must_be_inside_portable_bundle(tmp_path: Path):
    candidate_spec, output_root, _plans = _complete_sweep(tmp_path)
    source_spec = tmp_path / "candidates.json"
    assert source_spec.read_bytes() == candidate_spec.read_bytes()
    with pytest.raises(ValueError, match="must live under output_root"):
        analysis.build_analysis(
            candidate_spec=source_spec,
            output_root=output_root,
        )
    with pytest.raises(ValueError, match="must live directly under output_root"):
        analysis.main(
            [
                "--candidate-spec",
                str(candidate_spec),
                "--output-root",
                str(output_root),
                "--output",
                str(output_root / "nested" / "selection.json"),
            ]
        )


@pytest.mark.parametrize("missing", ("reserved_hbm", "optimizer_ledger"))
def test_incomplete_reserved_or_optimizer_memory_is_preserved_and_ineligible(
    tmp_path: Path,
    missing: str,
):
    candidate_spec, output_root, plans = _complete_sweep(tmp_path)
    plan = next(
        item
        for item in plans
        if item.identity["dataset"]["sample_index"] == 0
        and item.identity["calibration_candidate"]["candidate_id"]
        == "tail-r16-adamw-lr3e-4"
    )
    summary_path = plan.artifact_dir / "summary.json"
    summary = analysis.calibration.frozen._read_json(summary_path)
    if missing == "reserved_hbm":
        for snapshot in summary["hbm_bytes"].values():
            snapshot.pop("running_peak_reserved")
    else:
        summary["generation"]["optimizer_memory_bytes"] = None
        summary["parameters"]["optimizer_memory_bytes"] = None
    _rewrite_summary_and_completion(plan, summary)

    payload = analysis.build_analysis(
        candidate_spec=candidate_spec,
        output_root=output_root,
    )
    row = next(
        item
        for item in payload["candidate_rows"]
        if item["candidate_id"] == "tail-r16-adamw-lr3e-4"
    )
    sample = next(
        item for item in row["sample_results"] if item["sample_index"] == 0
    )
    if missing == "reserved_hbm":
        assert sample["whole_process_peak_hbm_reserved_bytes"] is None
        assert sample["memory_complete_and_comparable"] is False
        reason = "sample_0_hbm_not_comparable"
    else:
        assert sample["optimizer_persistent_bytes"] is None
        assert sample["optimizer_update_peak_bytes"] is None
        assert sample["optimizer_memory_complete"] is False
        reason = "sample_0_optimizer_memory_incomplete"
    assert row["aggregate"]["evidence_eligible"] is False
    assert reason in row["aggregate"]["ineligibility_reasons"]


@pytest.mark.parametrize("tamper", ("missing", "file_bytes", "content_binding"))
def test_artifact_identity_lock_is_a_global_fail_closed_source(
    tmp_path: Path,
    tamper: str,
):
    candidate_spec, output_root, plans = _complete_sweep(tmp_path)
    lock_path = output_root / "artifact_identity_lock.json"
    if tamper == "missing":
        lock_path.unlink()
        message = "missing artifact identity lock"
    elif tamper == "file_bytes":
        lock_path.write_bytes(lock_path.read_bytes() + b"\n")
        message = "artifact identity lock file hash mismatch"
    else:
        lock_payload = analysis.calibration.frozen._read_json(lock_path)
        lock_payload["target"]["revision"] = "tampered-revision"
        lock_path.write_bytes(
            (
                json.dumps(lock_payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            ).encode("utf-8")
        )
        stored = analysis.calibration.frozen._read_json(plans[0].identity_path)
        stored["identity"]["runtime"]["artifact_identity_lock"]["sha256"] = (
            analysis.aggregation._sha256_file(lock_path)
        )
        plans[0].identity_path.write_bytes(
            (
                json.dumps(stored, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            ).encode("utf-8")
        )
        message = "artifact identity lock target content mismatch"
    with pytest.raises(ValueError, match=message):
        analysis.build_analysis(
            candidate_spec=candidate_spec,
            output_root=output_root,
        )


def test_artifact_identity_lock_duplicate_json_key_is_rejected(tmp_path: Path):
    candidate_spec, output_root, _plans = _complete_sweep(tmp_path)
    lock_path = output_root / "artifact_identity_lock.json"
    original = lock_path.read_bytes()
    marker = b'  "kind": "dflash_tts_artifact_identity_lock",\n'
    assert marker in original
    lock_path.write_bytes(original.replace(marker, marker + marker, 1))
    with pytest.raises(ValueError, match="duplicate JSON key 'kind'"):
        analysis.build_analysis(
            candidate_spec=candidate_spec,
            output_root=output_root,
        )
