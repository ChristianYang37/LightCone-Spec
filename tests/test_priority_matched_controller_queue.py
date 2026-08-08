from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lightcone_spec.orchestration.manifest import ExperimentManifest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/experiments/p5_controller_queue.py"
QUEUE = ROOT / "scripts/experiments/run_priority_matched_controller_queue.sh"
OLD_QUEUE = ROOT / "scripts/experiments/run_remote_experiment_queue.sh"


def _module():
    spec = importlib.util.spec_from_file_location("_p5_controller_queue", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar(path: Path) -> Path:
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(_sha(path) + "\n", encoding="utf-8")
    return sidecar


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _receipt(path: Path, payload: dict, evidence: list[Path]) -> Path:
    body = {
        **payload,
        "evidence": [
            {"path": str(item.resolve()), "sha256": _sha(item)}
            for item in evidence
        ],
    }
    _json(path, body)
    _sidecar(path)
    return path


def _comparison(tmp_path: Path, *, passed: bool = True) -> Path:
    raw = _json(tmp_path / "raw.json", {"value": 1})
    analysis_identity = {
        "id": "p5_prompt_paired_cross_stride_v1",
        "comparisons": {
            "tts_best_vs_static": {
                "candidate_method": "tts",
                "candidate_update_stride": 16,
                "baseline_method": "static",
                "baseline_update_stride": 1,
            },
            "l0_best_vs_tts_best": {
                "candidate_method": "naive_async",
                "candidate_update_stride": 4,
                "baseline_method": "tts",
                "baseline_update_stride": 16,
            },
        },
    }
    return _receipt(
        tmp_path / "comparison.json",
        {
            "schema_version": 1,
            "status": "comparison_complete",
            "scope": "paired_stride_confirmation",
            "scientific_sample_pass": True,
            "all_cells_ci_low_positive": passed,
            "raw_provenance_pass": True,
            "formal_acceptance_claim_pass": passed,
            "ci_gates": {
                "tts_best_vs_static": passed,
                "l0_best_vs_tts_best": passed,
            },
            "analysis_identity": analysis_identity,
            "analysis_identity_sha256": hashlib.sha256(
                json.dumps(
                    analysis_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
        },
        [raw],
    )


def _generation(
    tmp_path: Path,
    *,
    model_pair: str = "qwen3_4b_dflash16",
    adapter_rank: int = 16,
    weight_update_mode: str = "lora",
    tail_layout_mode: str = "tail_lora",
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
) -> Path:
    windows = {
        "phase1_trace": {"offset": 88, "limit": 48, "half_open": [88, 136]},
        "phase2_l3": {"offset": 136, "limit": 48, "half_open": [136, 184]},
    }
    trace = _json(
        tmp_path / "trace.json",
        {
            "name": "trace",
            "engine_params": {"prompt_offset": 88},
            "units": [{"method": "naive_async"}, {"method": "tts"}],
        },
    )
    trace_sidecar = _sidecar(trace)
    l3 = _json(
        tmp_path / "l3.json",
        {
            "name": "l3",
            "engine_params": {"prompt_offset": 136, "l3_evaluation_only": True},
            "units": [{"method": "lc_transport"}],
        },
    )
    l3_sidecar = _sidecar(l3)
    l3_tts = _json(
        tmp_path / "l3-tts.json",
        {
            "name": "l3-tts",
            "engine_params": {
                "prompt_offset": 136,
                "phase2_tts_reference_only": True,
            },
            "units": [{"method": "tts"}],
        },
    )
    l3_tts_sidecar = _sidecar(l3_tts)
    runtime_body = {"schema_version": 1, "files": {}, "locked_reference": {}}
    runtime_fingerprint = {
        **runtime_body,
        "sha256": hashlib.sha256(
            json.dumps(
                runtime_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    controller_identity_body = {
        "model_pair": model_pair,
        "weight_update_mode": weight_update_mode,
        "tail_layout_mode": tail_layout_mode,
        "parameter_scope": "tail",
        "adapter_rank": adapter_rank,
        "optimizer": "adamw",
        "lr": lr,
        "weight_decay": weight_decay,
        "update_stride": 4,
        "lifecycle": "stream",
        "prompt_windows": windows,
        "pytorch_cuda_alloc_conf": "backend:native,expandable_segments:True",
        "bindings": {
            "lockfile_sha256": "4" * 64,
            "model_roots_sha256": "5" * 64,
            "model_revisions": {
                "target": "1" * 40,
                "drafter": "2" * 40,
                "tokenizer": "1" * 40,
            },
            "runtime_implementation_fingerprint": runtime_fingerprint,
        },
    }
    identity = hashlib.sha256(
        json.dumps(
            controller_identity_body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    controller_identity = {**controller_identity_body, "sha256": identity}
    return _receipt(
        tmp_path / "generation.json",
        {
            "schema_version": 2,
            "status": "matched_controller_manifests_generated",
            "controller_identity": controller_identity,
            "controller_identity_sha256": identity,
            "locked_inputs": {
                "lockfile": str((tmp_path / "lock.json").resolve()),
                "lockfile_sha256": "4" * 64,
                "model_roots": str((tmp_path / "roots.json").resolve()),
                "model_roots_sha256": "5" * 64,
                "model_revisions": {
                    "target": "1" * 40,
                    "drafter": "2" * 40,
                    "tokenizer": "1" * 40,
                },
            },
            "mirror_contract": {
                "exact": True,
                "prompt_windows": windows,
                "prompt_windows_disjoint": True,
            },
            "artifacts": {
                "TRACE_MATCHED": {
                    "path": str(trace.resolve()),
                    "sha256": _sha(trace),
                    "sidecar_path": str(trace_sidecar.resolve()),
                    "sidecar_sha256": _sha(trace_sidecar),
                },
                "L3_PHASE2_MATCHED": {
                    "path": str(l3.resolve()),
                    "sha256": _sha(l3),
                    "sidecar_path": str(l3_sidecar.resolve()),
                    "sidecar_sha256": _sha(l3_sidecar),
                },
                "L3_PHASE2_TTS_REFERENCE": {
                    "path": str(l3_tts.resolve()),
                    "sha256": _sha(l3_tts),
                    "sidecar_path": str(l3_tts_sidecar.resolve()),
                    "sidecar_sha256": _sha(l3_tts_sidecar),
                },
            },
        },
        [trace, trace_sidecar, l3, l3_sidecar, l3_tts, l3_tts_sidecar],
    )


def _foundation(tmp_path: Path) -> Path:
    raw = _json(tmp_path / "foundation-raw.json", {"value": 1})
    identity = {"schema_version": 2, "manifest_sha256": "a" * 64}
    return _receipt(
        tmp_path / "TTS_0_40K_FOUNDATION.json",
        {
            "schema_version": 2,
            "status": "TTS_0_40K_CONFIRMED",
            "scope": "tts_0_40k_foundation",
            "formal_acceptance_foundation_pass": True,
            "identity": identity,
            "identity_sha256": hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "roles": {
                "tts_acceptance_best": {"method": "tts", "stride": 16},
                "tts_engineering_best": {"method": "tts", "stride": 8},
                "same_stride_tts_for_l0": {"method": "tts", "stride": 4},
            },
        },
        [raw],
    )


def _report(tmp_path: Path, *, final: bool = False, l2: bool = True) -> Path:
    trace_exactness = {
        "verified": True,
        "violation_count": 0,
        "rounds_checked": 10,
    }
    oracle = {
        "complete": True,
        "reference": "same_arrival_full_candidate_l0",
        "n_test_groups": 8,
        "l1_eligible": True,
        "l1_ci95": [0.1, 0.2],
        "l2_eligible": l2,
        "l2_ci95": [0.1, 0.2] if l2 else [-0.1, 0.2],
    }
    paired = {
        "complete": True,
        "reference": "same_candidate_actual_tts_barrier",
        "n_test_groups": 8,
        "incomplete_pairs": 0,
        "l1_eligible": True,
        "l1_ci95": [0.1, 0.2],
        "l2_eligible": l2,
        "l2_ci95": [0.1, 0.2] if l2 else [-0.1, 0.2],
    }
    learned = {
        "complete": True,
        "reference": "learned_policy_same_candidate_actual_tts_barrier",
        "n_test_groups": 8,
        "incomplete_pairs": 0,
        "l1_eligible": True,
        "l1_ci95": [0.1, 0.2],
        "l2_eligible": l2,
        "l2_ci95": [0.1, 0.2] if l2 else [-0.1, 0.2],
        "l1_zero_delay_fastpath_fraction": 0.1,
        "l1_constant_apply_fastpath_fraction": 0.2,
        "l1_constant_discard_fastpath_fraction": 0.1,
        "l1_predictor_path_fraction": 0.6,
        "l2_zero_delay_fastpath_fraction": 0.1,
        "l2_constant_profile_fastpath_fraction": 0.3,
        "l2_unit_kappa_fastpath_fraction": 0.2,
        "l2_predictor_path_fraction": 0.6,
    }
    transport_map = {"rank": 16, "basis": [[1.0]], "coef": [[1.0]]}
    transport_map_sha = hashlib.sha256(
        json.dumps(
            transport_map,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    l3_gate = {
            "evaluation_ready": True,
            "enabled": final,
            "exactness": {
                "verified": final,
                "violation_count": 0 if final else -1,
                "rounds_checked": 10 if final else 0,
            },
            "heldout_transported_utility_gate": {
                "complete": final,
                "eligible": final,
                "utility_metric": "survival_weighted_accepted_prefix_v1",
                "l3_contract": "joint_fisher_transport_adamw_damping_v1",
                "n_test_groups": 8 if final else 0,
                "ci95_vs_tts": [0.1, 0.2] if final else None,
                "ci95_vs_l2": [0.1, 0.2] if final else None,
                "transport_map_sha256": transport_map_sha if final else None,
                "pairing_contract": (
                    "exact_request_seed_concurrency_trace_stage_v1"
                    if final
                    else None
                ),
            },
        }
    runtime_identity = {
        "schema_version": 3,
        "model": {
            "pair_id": "qwen3_4b_dflash16",
            "target_revision": "1" * 40,
            "drafter_revision": "2" * 40,
            "tokenizer_revision": "1" * 40,
        },
        "candidate": {
            "weight_update_mode": "tail_lora",
            "adapter_rank": 16,
            "optimizer": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-2,
            "update_stride": 4,
            "lifecycle": "stream",
        },
        "sampling": {"temperature": 0.0, "top_p": 1.0},
    }
    runtime_sha = hashlib.sha256(
        json.dumps(
            runtime_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    artifact = _json(
        tmp_path
        / ("final-controller" if final else "phase1-controller")
        / (
            "qwen3_4b_dflash16.tail_lora."
            + "3" * 64
            + ".controller.json"
        ),
        {
            "schema_version": 1,
            "model_pair_id": "qwen3_4b_dflash16",
            "transport_map": transport_map,
            "extra": {
                "controller_runtime_identity": runtime_identity,
                "controller_runtime_identity_sha256": runtime_sha,
                "parameter_layout_sha256": "3" * 64,
                "transport_map_sha256": transport_map_sha,
                "constant_fast_path_source": "calibration_only_v1",
                "constant_fast_path_calibration_coverage": {
                    "records": 24,
                    "l1_constant_apply_fraction": 0.25,
                    "l1_constant_discard_fraction": 0.25,
                    "l2_constant_profile_fraction": 0.5,
                },
                "trace_exactness": trace_exactness,
                "oracle_replay_gate": oracle,
                "tts_paired_gate": paired,
                "learned_policy_gate": learned,
                "l3_gate": l3_gate,
            },
        },
    )
    _sidecar(artifact)
    payload = {
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": _sha(artifact),
        "trace_exactness": trace_exactness,
        "oracle_replay_gate": oracle,
        "tts_paired_gate": paired,
        "learned_policy_gate": learned,
        "l3_gate": l3_gate,
    }
    return _json(tmp_path / ("final-report.json" if final else "phase1-report.json"), payload)


def _trace_tree(tmp_path: Path) -> Path:
    run = tmp_path / "traces/phase1/run-a"
    artifact = _json(run / "run-manifest.json", {"complete": True})
    _json(
        run / "hashes.json",
        {"run-manifest.json": {"sha256": _sha(artifact), "bytes": artifact.stat().st_size}},
    )
    return tmp_path / "traces"


def _controller_terminal(
    tmp_path: Path,
    generation: Path,
    *,
    eligible: dict[str, bool],
) -> Path:
    generation_payload = json.loads(generation.read_text(encoding="utf-8"))
    selected = all(eligible.values())
    return _receipt(
        tmp_path
        / "queue"
        / ("CONTROLLER_SELECTED.json" if selected else "CONTROLLER_BLOCKED.json"),
        {
            "schema_version": 1,
            "status": (
                "matched_controller_selected"
                if selected
                else "matched_controller_blocked"
            ),
            "scope": "matched_dflash_l1_l2_l3_evidence",
            "controller_identity_sha256": generation_payload[
                "controller_identity_sha256"
            ],
            "eligible": eligible,
        },
        [generation],
    )


def test_queue_shell_is_two_pass_fail_closed_and_resumable():
    subprocess.run(["bash", "-n", str(QUEUE)], check=True)
    source = QUEUE.read_text(encoding="utf-8")
    assert '"$FOUNDATION_TOOL" compare' in source
    assert '--tts-foundation-terminal "$FOUNDATION_TERMINAL"' in source
    assert '"$FINAL_GATE_TOOL"' in source
    assert 'gate_rc" -ne 0' in source
    assert 'gate_rc" -ne 3' in source
    assert 'L3_PHASE2_TTS_REFERENCE' in source
    assert "TTS_0_40K_CONFIRMED" in source
    assert "phase1-gate" in source
    assert "block-confirmation" not in source
    assert 'if [ "$PHASE2_ALLOWED" = 1 ]' in source
    assert "--controller-root" in source
    assert "CONTROLLER_SELECTED.json" in source
    assert "CONTROLLER_BLOCKED.json" in source
    assert "write-failure" in source
    assert "build-headline" in source
    assert "FINAL_0_40K_MANIFESTS.json" in source
    assert "p5-dflash4b-final-0-40k-v10" in source
    assert "algorithmic-c4" in source
    assert "mfu-context-load" in source
    assert "exit 42" in source
    assert "flock -n 9" in source
    assert 'PY_BIN_DIR=$(dirname -- "$PY")' in source
    assert 'PATH="$PY_BIN_DIR:$CUDA_TOOLKIT/bin:$PATH"' in source
    assert "HEADLINE_ATTEMPT_ROOT" in source
    assert 'archive-failure --root "$HEADLINE_ATTEMPT_ROOT"' in source
    assert '--root "$HEADLINE_ATTEMPT_ROOT" --phase "$phase"' in source
    assert "queue_process_control_init" in source
    assert "queue_run_managed" in source
    blocked_resume = source.split("  blocked)", 1)[1].split("  failed)", 1)[0]
    assert "run_final_headline" not in blocked_resume
    final_block = source.split('if [ "$FINAL_STATUS" = selected ]', 1)[1]
    assert "LIGHTCONE_SKIP_LEGACY_PRIORITY=1" in final_block
    assert 'LIGHTCONE_RESUME_RECEIPT="$HEADLINE_CONFIRMED"' in final_block


def test_old_queue_can_skip_priority_only_with_verified_final_receipt():
    subprocess.run(["bash", "-n", str(OLD_QUEUE)], check=True)
    source = OLD_QUEUE.read_text(encoding="utf-8")
    assert 'if [ "$SKIP_LEGACY_PRIORITY" = 1 ]' in source
    assert 'if [ -z "$RESUME_RECEIPT" ]' in source
    assert 'verify-resume --receipt "$RESUME_RECEIPT"' in source
    skip_block = source.split('if [ "$SKIP_LEGACY_PRIORITY" = 1 ]', 1)[1]
    skip_block = skip_block.split("else", 1)[0]
    assert "run_priority_chain" not in skip_block


def test_phase1_gate_and_selected_terminal_close_all_three_methods(tmp_path: Path):
    module = _module()
    comparison = _comparison(tmp_path)
    generation = _generation(tmp_path)
    phase1_report = _report(tmp_path)
    gate_path = tmp_path / module.PHASE1_GATE
    gate = module.phase1_gate(
        comparison=comparison,
        generation=generation,
        report=phase1_report,
        output=gate_path,
        queue_source=QUEUE,
    )
    assert gate["l1_eligible"] is True
    assert gate["l2_eligible"] is True
    assert gate["l3_phase2_allowed"] is True
    assert gate["fast_path_coverage"]["source"] == "calibration_only_v1"
    assert (
        gate["fast_path_coverage"]["heldout"][
            "l1_constant_apply_fastpath_fraction"
        ]
        == 0.2
    )

    result = module.finalize(
        root=tmp_path / "queue",
        comparison=comparison,
        generation=generation,
        phase1_gate_path=gate_path,
        trace_root=_trace_tree(tmp_path),
        final_report=_report(tmp_path, final=True),
        queue_source=QUEUE,
    )
    assert result["status"] == "matched_controller_selected"
    assert result["eligible"] == {"l1": True, "l2": True, "l3": True}
    assert result["fast_path_coverage"] == gate["fast_path_coverage"]
    assert module.terminal_status(tmp_path / "queue") == "selected"


def test_failed_l2_or_missing_phase2_is_scientific_block_not_selection(tmp_path: Path):
    module = _module()
    comparison = _comparison(tmp_path)
    generation = _generation(tmp_path)
    gate_path = tmp_path / module.PHASE1_GATE
    gate = module.phase1_gate(
        comparison=comparison,
        generation=generation,
        report=_report(tmp_path, l2=False),
        output=gate_path,
        queue_source=QUEUE,
    )
    assert gate["l1_eligible"] is True
    assert gate["l2_eligible"] is False

    result = module.finalize(
        root=tmp_path / "queue",
        comparison=comparison,
        generation=generation,
        phase1_gate_path=gate_path,
        trace_root=_trace_tree(tmp_path),
        final_report=None,
        queue_source=QUEUE,
    )
    assert result["status"] == "matched_controller_blocked"
    assert result["eligible"] == {"l1": True, "l2": False, "l3": False}
    assert module.terminal_status(tmp_path / "queue") == "blocked"


def test_resumable_failure_is_archived_before_retry(tmp_path: Path):
    module = _module()
    root = tmp_path / "queue"
    state = _json(root / "queue-state.jsonl", {"phase": "phase1"})
    module.write_failure(root=root, phase="phase1", return_code=7, evidence=[state])
    assert module.terminal_status(root) == "failed"
    module.archive_failure(root)
    assert module.terminal_status(root) == "none"
    archived = list((root / "attempts").glob("CONTROLLER_FAILED.*.json"))
    assert len(archived) == 1
    assert Path(str(archived[0]) + ".sha256").is_file()


def test_negative_formal_confirmation_publishes_scientific_block(tmp_path: Path):
    module = _module()
    root = tmp_path / "queue"
    result = module.block_confirmation(
        root=root,
        comparison=_comparison(tmp_path, passed=False),
        queue_source=QUEUE,
    )

    assert result["status"] == "matched_controller_blocked"
    assert result["blocked_reasons"] == ["formal_confirmation_gate_failed"]
    assert module.terminal_status(root) == "blocked"


def test_confirmation_or_recursive_evidence_drift_fails_closed(tmp_path: Path):
    module = _module()
    comparison = _comparison(tmp_path)
    generation = _generation(tmp_path)
    raw = tmp_path / "raw.json"
    raw.write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(module.QueueEvidenceError, match="evidence drift"):
        module.phase1_gate(
            comparison=comparison,
            generation=generation,
            report=_report(tmp_path),
            output=tmp_path / module.PHASE1_GATE,
            queue_source=QUEUE,
        )


def test_final_0_40k_manifests_bind_winners_gates_and_separate_loads(
    tmp_path: Path,
):
    module = _module()
    comparison = _comparison(tmp_path)
    generation = _generation(tmp_path)
    terminal = _controller_terminal(
        tmp_path,
        generation,
        eligible={"l1": True, "l2": True, "l3": True},
    )
    report = _report(tmp_path, final=True)
    receipt = module.build_headline_manifests(
        comparison=comparison,
        generation=generation,
        terminal=terminal,
        tts_foundation_terminal=_foundation(tmp_path),
        controller_report=report,
        output_dir=tmp_path / "headline/manifests",
        output=tmp_path / "headline/FINAL_0_40K_MANIFESTS.json",
    )

    assert receipt["status"] == "final_0_40k_manifests_generated"
    assert receipt["methods"] == [
        "static",
        "tts",
        "naive_async",
        "lc_gate",
        "lc_damp",
        "lc_transport",
    ]
    identity = receipt["optimizer_identity"]
    assert identity == {
        "model_pair": "qwen3_4b_dflash16",
        "weight_update_mode": "lora",
        "tail_layout_mode": "tail_lora",
        "adapter_rank": 16,
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "tts_role_strides": {
            "acceptance_best": 16,
            "engineering_best": 8,
            "same_stride": 4,
        },
        "adaptation_stride": 4,
    }
    assert receipt["context_contract"]["contexts"] == [
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        40000,
    ]
    assert receipt["context_contract"]["load_profiles_are_not_pooled"] is True
    assert max(
        row["required_kv_token_slots"]
        for row in receipt["context_contract"]["capacity"]
    ) <= 400000

    algorithmic = ExperimentManifest.load(
        receipt["artifacts"]["ALGORITHMIC_C4"]["path"]
    )
    mfu = ExperimentManifest.load(receipt["artifacts"]["MFU_CONTEXT_LOAD"]["path"])
    assert algorithmic.engine_params["p5_context_lengths"] == receipt[
        "context_contract"
    ]["contexts"]
    assert algorithmic.engine_params["max_new_tokens"] == 512
    assert algorithmic.engine_params["checkpoint_max_context_length"] == 40960
    assert algorithmic.engine_params["p5_context_timing_contract"] == (
        "independent_exact_context_group_v1"
    )
    assert algorithmic.engine_params["model_roots_sha256"] == "5" * 64
    assert algorithmic.engine_params["locked_model_revisions"] == {
        "target": "1" * 40,
        "drafter": "2" * 40,
        "tokenizer": "1" * 40,
    }
    assert algorithmic.engine_params["runtime_implementation_fingerprint"][
        "sha256"
    ]
    assert {unit.concurrency for unit in algorithmic.units} == {4}
    assert {unit.prompt_subset for unit in algorithmic.units} == {
        "p5_ctx_512-40000"
    }
    assert {
        unit.stride for unit in algorithmic.units if unit.method == "tts"
    } == {4, 8, 16}
    assert {
        unit.stride
        for unit in algorithmic.units
        if unit.method not in {"static", "tts"}
    } == {4}
    assert {
        (unit.prompt_subset, unit.concurrency) for unit in mfu.units
    } == {
        ("p5_ctx_512-4096", 48),
        ("p5_ctx_8192-16384", 20),
        ("p5_ctx_32768-40000", 8),
    }
    assert algorithmic.engine_params["prompt_offset"] == 184
    assert receipt["tts_role_strides"] == {
        "acceptance_best": 16,
        "engineering_best": 8,
        "same_stride": 4,
    }


def test_final_manifest_injects_only_individually_gated_controller_methods(
    tmp_path: Path,
):
    module = _module()
    comparison = _comparison(tmp_path)
    generation = _generation(
        tmp_path,
        model_pair="qwen3_8b_dflash16",
        adapter_rank=8,
        weight_update_mode="residual",
        tail_layout_mode="output_residual",
        lr=2e-4,
        weight_decay=5e-3,
    )
    terminal = _controller_terminal(
        tmp_path,
        generation,
        eligible={"l1": False, "l2": False, "l3": False},
    )
    receipt = module.build_headline_manifests(
        comparison=comparison,
        generation=generation,
        terminal=terminal,
        tts_foundation_terminal=_foundation(tmp_path),
        output_dir=tmp_path / "headline/manifests",
        output=tmp_path / "headline/FINAL_0_40K_MANIFESTS.json",
    )
    assert receipt["methods"] == ["static", "tts", "naive_async"]
    assert receipt["controller_artifact_path"] is None
    assert receipt["optimizer_identity"]["model_pair"] == "qwen3_8b_dflash16"
    assert receipt["optimizer_identity"]["weight_update_mode"] == "residual"
    assert receipt["optimizer_identity"]["tail_layout_mode"] == "output_residual"
    assert receipt["optimizer_identity"]["adapter_rank"] == 8
    assert receipt["optimizer_identity"]["lr"] == 2e-4
    assert receipt["optimizer_identity"]["weight_decay"] == 5e-3
    for row in receipt["artifacts"].values():
        manifest = ExperimentManifest.load(row["path"])
        assert {unit.method for unit in manifest.units} == {
            "static",
            "tts",
            "naive_async",
        }
        assert {unit.model_pair for unit in manifest.units} == {
            "qwen3_8b_dflash16"
        }
        assert {unit.adapter_rank for unit in manifest.units} == {8}
        assert {unit.trainable_scope for unit in manifest.units} == {
            "output_residual"
        }
        assert manifest.engine_params["lr"] == 2e-4
        assert manifest.engine_params["weight_decay"] == 5e-3


def test_final_manifest_refuses_eligible_controller_without_artifact(
    tmp_path: Path,
):
    module = _module()
    comparison = _comparison(tmp_path)
    generation = _generation(tmp_path)
    terminal = _controller_terminal(
        tmp_path,
        generation,
        eligible={"l1": True, "l2": False, "l3": False},
    )
    with pytest.raises(module.QueueEvidenceError, match="frozen replay report"):
        module.build_headline_manifests(
            comparison=comparison,
            generation=generation,
            terminal=terminal,
            tts_foundation_terminal=_foundation(tmp_path),
            output_dir=tmp_path / "headline/manifests",
            output=tmp_path / "headline/FINAL_0_40K_MANIFESTS.json",
        )


def test_final_manifest_supports_partial_l1_gate_without_l2_or_l3(
    tmp_path: Path,
):
    module = _module()
    comparison = _comparison(tmp_path)
    generation = _generation(tmp_path)
    terminal = _controller_terminal(
        tmp_path,
        generation,
        eligible={"l1": True, "l2": False, "l3": False},
    )
    receipt = module.build_headline_manifests(
        comparison=comparison,
        generation=generation,
        terminal=terminal,
        tts_foundation_terminal=_foundation(tmp_path),
        controller_report=_report(tmp_path, l2=False),
        output_dir=tmp_path / "headline/manifests",
        output=tmp_path / "headline/FINAL_0_40K_MANIFESTS.json",
    )
    assert receipt["methods"] == ["static", "tts", "naive_async", "lc_gate"]
    for row in receipt["artifacts"].values():
        manifest = ExperimentManifest.load(row["path"])
        assert {unit.method for unit in manifest.units} == {
            "static",
            "tts",
            "naive_async",
            "lc_gate",
        }
