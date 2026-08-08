from __future__ import annotations

import errno
import hashlib
import importlib.util
import io
import json
import os
import sys
import threading
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "experiments" / "run_dflash_tts_frozen_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_dflash_tts_frozen_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_selection(tmp_path: Path) -> tuple[Path, Path]:
    fixture_path = Path(__file__).with_name(
        "test_build_dflash_tts_optimizer_selection.py"
    )
    fixture_spec = importlib.util.spec_from_file_location(
        "synthetic_optimizer_selection_fixture", fixture_path
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture = importlib.util.module_from_spec(fixture_spec)
    sys.modules[fixture_spec.name] = fixture
    fixture_spec.loader.exec_module(fixture)
    project, calibration = fixture._synthetic_calibration(tmp_path / "selection")
    summary_path = tmp_path / "selection_summary.json"
    with redirect_stdout(io.StringIO()):
        assert fixture.builder.main(
            [
                "--project-root",
                str(project),
                "--calibration-root",
                str(calibration),
                "--output",
                str(summary_path),
            ]
        ) == 0
    summary = json.loads(summary_path.read_text())
    selection = {
        "schema_version": 1,
        "status": "locked",
        "study_id": summary["study_id"],
        "evidence_scope": summary["evidence_classification"],
        "calibration": summary["calibration"],
        "candidate_grid": summary["candidate_grid"],
        "selection_rule": summary["selection_rule"],
        "locked_configs": summary["selected_configs"],
        "evidence_artifacts": [
            {
                "kind": "deterministic_selection_summary",
                "path": summary_path.name,
                "sha256": _sha256(summary_path),
            }
        ],
    }
    selection_path = tmp_path / "selected_optimizer_config.json"
    selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n")
    return selection_path, summary_path


def _base_argv(tmp_path: Path, *, modes: tuple[str, ...]) -> list[str]:
    harness = tmp_path / "dflash_tts_reference.py"
    harness.write_text("# frozen fake harness\n")
    reference = tmp_path / "reference"
    (reference / "dflash").mkdir(parents=True)
    (reference / "dflash" / "model.py").write_text("# official source\n")
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    for model, label in ((target, "target"), (draft, "draft")):
        (model / "config.json").write_text(json.dumps({"model": label}) + "\n")
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "model.safetensors"}}) + "\n"
        )
        (model / "model.safetensors").write_bytes(
            f"{label}-weights".encode()
        )
    (target / "tokenizer_config.json").write_text('{"chat_template":"x"}\n')
    (target / "tokenizer.json").write_text('{"version":"1.0"}\n')
    dataset = tmp_path / "math500.jsonl"
    dataset.write_text('{"turns":["question"]}\n')
    projection = tmp_path / "projection.npz"
    projection.write_bytes(b"locked projection")
    target_weight = target / "model.safetensors"
    (tmp_path / "projection.npz.meta.json").write_text(
        json.dumps(
            {
                "binding": {
                    "target_head_artifact": {
                        "weight_files": [
                            {
                                "name": target_weight.name,
                                "bytes": target_weight.stat().st_size,
                                "sha256": _sha256(target_weight),
                            }
                        ]
                    }
                }
            }
        )
        + "\n"
    )
    selection_path, _calibration_evidence = _synthetic_selection(tmp_path)
    return [
        "--python",
        sys.executable,
        "--harness",
        str(harness),
        "--selected-optimizer-config",
        str(selection_path),
        "--reference-root",
        str(reference),
        "--target-model",
        str(target),
        "--target-revision",
        "target-rev",
        "--draft-model",
        str(draft),
        "--draft-revision",
        "draft-rev",
        "--dataset",
        str(dataset),
        "--dataset-revision",
        "dataset-rev",
        "--sample-index",
        "419",
        "--input-tokens",
        "109",
        "--mask-token-id",
        "151669",
        "--output-root",
        str(tmp_path / "runs"),
        "--total-contexts",
        "8192",
        "--modes",
        *modes,
        "--projection-artifact",
        str(projection),
    ]


def _flag(command: tuple[str, ...], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def test_context_arithmetic_closes_on_pending_block_boundary():
    assert (
        runner.max_new_tokens_for_total_context(
            total_context=40960,
            input_tokens=109,
            draft_block_size=16,
        )
        == 40836
    )
    assert 109 + 40836 + (16 - 1) == 40960
    with pytest.raises(ValueError, match="no generation budget"):
        runner.max_new_tokens_for_total_context(
            total_context=124,
            input_tokens=109,
            draft_block_size=16,
        )
    with pytest.raises(ValueError, match="greater than one"):
        runner.max_new_tokens_for_total_context(
            total_context=8192,
            input_tokens=109,
            draft_block_size=1,
        )


def test_exclusive_json_publish_never_exposes_partial_final(
    tmp_path: Path, monkeypatch
):
    destination = tmp_path / "identity.json"
    real_dump = runner.json.dump

    def observing_dump(value, handle, **kwargs):
        assert not destination.exists()
        return real_dump(value, handle, **kwargs)

    monkeypatch.setattr(runner.json, "dump", observing_dump)
    runner._write_json_exclusive(destination, {"identity": "complete"})
    assert json.loads(destination.read_text()) == {"identity": "complete"}
    assert not list(tmp_path.glob(".identity.json.*.tmp"))

    monkeypatch.setattr(runner.json, "dump", real_dump)
    with pytest.raises(FileExistsError):
        runner._write_json_exclusive(destination, {"identity": "replacement"})
    assert json.loads(destination.read_text()) == {"identity": "complete"}
    assert not list(tmp_path.glob(".identity.json.*.tmp"))


def test_exclusive_json_write_failure_leaves_no_final_or_temp(
    tmp_path: Path, monkeypatch
):
    destination = tmp_path / "identity.json"

    def failing_dump(_value, handle, **_kwargs):
        handle.write('{"partial":')
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(runner.json, "dump", failing_dump)
    with pytest.raises(RuntimeError, match="injected serialization failure"):
        runner._write_json_exclusive(destination, {"identity": "complete"})
    assert not destination.exists()
    assert not list(tmp_path.glob(".identity.json.*.tmp"))


def test_exclusive_json_concurrent_writers_have_one_complete_winner(
    tmp_path: Path,
):
    destination = tmp_path / "identity.json"
    barrier = threading.Barrier(2)

    def write(candidate: int) -> str:
        barrier.wait()
        try:
            runner._write_json_exclusive(
                destination, {"candidate": candidate, "payload": "x" * 4096}
            )
        except FileExistsError:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(write, (1, 2)))
    assert sorted(outcomes) == ["lost", "won"]
    payload = json.loads(destination.read_text())
    assert payload["candidate"] in {1, 2}
    assert payload["payload"] == "x" * 4096
    assert not list(tmp_path.glob(".identity.json.*.tmp"))


def test_exclusive_json_hardlink_failure_fails_closed(
    tmp_path: Path, monkeypatch
):
    destination = tmp_path / "identity.json"

    def reject_link(_source, _destination):
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    monkeypatch.setattr(runner.os, "link", reject_link)
    with pytest.raises(OSError, match="hard links unavailable"):
        runner._write_json_exclusive(destination, {"identity": "complete"})
    assert not destination.exists()
    assert not list(tmp_path.glob(".identity.json.*.tmp"))


def test_commands_use_exact_context_and_frozen_mode_table(tmp_path: Path):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=runner.MODE_ORDER)
    )
    runner._validate_args(args)
    plans = runner.build_run_plans(args)
    assert [plan.identity["mode"] for plan in plans] == list(runner.MODE_ORDER)
    assert all(plan.command[0] == os.path.abspath(sys.executable) for plan in plans)
    assert all(_flag(plan.command, "--max-new-tokens") == "8068" for plan in plans)
    assert all(
        plan.identity["generation"]["required_prefix_plus_block"] == 8192
        for plan in plans
    )
    assert all(
        plan.identity["optimizer_selection"]["status"] == "locked"
        for plan in plans
    )
    assert all(
        plan.identity["optimizer_selection"]["sha256"]
        == _sha256(Path(args.selected_optimizer_config))
        for plan in plans
    )
    assert all(
        _flag(plan.command, "--run-identity-sha256")
        == plan.identity_sha256
        for plan in plans
    )
    assert all(
        _flag(plan.command, "--command-sha256")
        == runner._signed_command_sha256(plan.command)
        for plan in plans
    )
    by_mode = {plan.identity["mode"]: plan for plan in plans}

    full = by_mode["full-drafter"]
    assert _flag(full.command, "--optimizer") == "adam"
    assert float(_flag(full.command, "--lr")) == pytest.approx(1e-5)
    assert float(_flag(full.command, "--weight-decay")) == pytest.approx(0.0)
    assert "--rank" not in full.command

    wide_lora = by_mode["drafter-lora"]
    assert _flag(wide_lora.command, "--optimizer") == "adam"
    assert float(_flag(wide_lora.command, "--lr")) == pytest.approx(3e-4)
    assert _flag(wide_lora.command, "--rank") == "8"

    full_tail = by_mode["full-rank-tail"]
    assert float(_flag(full_tail.command, "--lr")) == pytest.approx(3e-6)
    assert "--rank" not in full_tail.command

    tail_lora = by_mode["tail-lora"]
    assert _flag(tail_lora.command, "--optimizer") == "adamw"
    assert float(_flag(tail_lora.command, "--lr")) == pytest.approx(1e-4)
    assert float(_flag(tail_lora.command, "--weight-decay")) == pytest.approx(1e-2)
    assert _flag(tail_lora.command, "--rank") == "16"

    residual = by_mode["output-residual"]
    assert float(_flag(residual.command, "--lr")) == pytest.approx(3e-4)
    assert _flag(residual.command, "--rank") == "16"
    assert _flag(residual.command, "--projection-artifact").endswith(
        "projection.npz"
    )


def test_deterministic_contract_is_default_and_identity_locked(tmp_path: Path):
    argv = _base_argv(tmp_path, modes=("static",))
    deterministic = runner.build_run_plans(
        runner.build_parser().parse_args(argv)
    )[0]
    contract = runner.determinism_contract(True)

    assert deterministic.identity["runtime"]["determinism"] == contract
    assert "--deterministic" in deterministic.command
    assert "--no-deterministic" not in deterministic.command
    assert deterministic.plan_payload()["environment"][
        "CUBLAS_WORKSPACE_CONFIG"
    ] == runner.DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG

    nondeterministic = runner.build_run_plans(
        runner.build_parser().parse_args([*argv, "--no-deterministic"])
    )[0]
    assert nondeterministic.identity["runtime"]["determinism"] == (
        runner.determinism_contract(False)
    )
    assert "--no-deterministic" in nondeterministic.command
    assert "--deterministic" not in nondeterministic.command
    assert nondeterministic.plan_payload()["environment"][
        "CUBLAS_WORKSPACE_CONFIG"
    ] is None
    assert deterministic.identity_sha256 != nondeterministic.identity_sha256


def test_frozen_source_drift_makes_old_completion_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    argv = _base_argv(tmp_path, modes=("static",))
    frozen_source = tmp_path / "run_dflash_tts_frozen_sweep.py"
    frozen_source.write_text("# frozen orchestrator v1\n")
    monkeypatch.setattr(runner, "__file__", str(frozen_source))

    first = runner.build_run_plans(runner.build_parser().parse_args(argv))[0]
    implementation = first.identity["runtime"]
    assert implementation["frozen_orchestrator"] == {
        "path": str(frozen_source.resolve()),
        "sha256": _sha256(frozen_source),
    }
    assert implementation["frozen_run_validator"] == implementation[
        "frozen_orchestrator"
    ]
    _write_completed_run(first)

    frozen_source.write_text("# frozen orchestrator v2\n")
    second = runner.build_run_plans(runner.build_parser().parse_args(argv))[0]
    assert second.identity_sha256 != first.identity_sha256
    assert second.identity["runtime"]["frozen_orchestrator"]["sha256"] == (
        _sha256(frozen_source)
    )
    with pytest.raises(ValueError, match="stored run plan mismatch"):
        runner.completed_run_matches(second)


def test_subprocess_environment_applies_planned_cublas_contract(
    tmp_path: Path, monkeypatch
):
    argv = _base_argv(tmp_path, modes=("static",))
    deterministic = runner.build_run_plans(
        runner.build_parser().parse_args(argv)
    )[0]
    nondeterministic = runner.build_run_plans(
        runner.build_parser().parse_args([*argv, "--no-deterministic"])
    )[0]
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    assert runner._subprocess_environment(deterministic)[
        "CUBLAS_WORKSPACE_CONFIG"
    ] == runner.DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    assert "CUBLAS_WORKSPACE_CONFIG" not in runner._subprocess_environment(
        nondeterministic
    )


def _write_completed_run(plan: runner.RunPlan) -> None:
    plan.run_dir.mkdir(parents=True)
    runner._write_json_exclusive(plan.identity_path, plan.plan_payload())
    plan.artifact_dir.mkdir()
    rounds_path = plan.artifact_dir / "rounds.jsonl"
    rounds_path.write_text('{"round_index":0}\n')
    identity = plan.identity
    generation = identity["generation"]
    optimization = identity["optimization"]
    effective_optimizer = (
        None
        if identity["mode"] == "static"
        else optimization["optimizer"].upper()
    )
    determinism = identity["runtime"]["determinism"]
    runtime_fingerprint = {
        "schema_version": 1,
        "python_version": "3.12.0",
        "python_implementation": "CPython",
        "platform": "Linux-test",
        "torch_version": "2.8.0",
        "cuda_runtime_version": "12.8",
        "cuda_driver_version": 12080,
        "attention_implementation": identity["runtime"][
            "attention_implementation"
        ],
        "dtype": identity["runtime"]["dtype"],
        "device": identity["runtime"]["device"],
        "resolved_device": identity["runtime"]["device"],
        "allocator_config": identity["runtime"]["allocator_config"],
        "cuda_visible_devices": None,
        "deterministic_algorithms": determinism[
            "torch_deterministic_algorithms"
        ],
        "deterministic_warn_only": determinism[
            "torch_deterministic_warn_only"
        ],
        "allow_tf32": {
            "matmul": determinism["cuda_matmul_allow_tf32"],
            "cudnn": determinism["cudnn_allow_tf32"],
        },
        "float32_matmul_precision": determinism[
            "float32_matmul_precision"
        ],
        "cudnn_benchmark": determinism["cudnn_benchmark"],
        "cudnn_deterministic": determinism["cudnn_deterministic"],
        "cublas_workspace_config": determinism[
            "cublas_workspace_config"
        ],
        "sdpa_backends": determinism["sdpa_backends"],
        "determinism_contract": determinism,
        "gpu": {
            "name": "Test GPU",
            "total_memory_bytes": 96 << 30,
            "compute_capability": "12.0",
            "device_index": 0,
        },
    }
    summary = {
        "schema_version": runner.HARNESS_ARTIFACT_SCHEMA_VERSION,
        "status": "complete_reference_run",
        "mode": identity["mode"],
        "run_attestation": runner._expected_run_attestation(plan),
        "harness": {
            "source_sha256": identity["runtime"]["harness"]["sha256"],
            "artifact_schema_version": runner.HARNESS_ARTIFACT_SCHEMA_VERSION,
        },
        "artifact_identity": {
            "verification_status": "fully_verified_content_sha256_v1",
            "lock": identity["runtime"]["artifact_identity_lock"],
        },
        "runtime_fingerprint": runtime_fingerprint,
        "reference": {
            "declared_revision": identity["reference"]["revision"],
            "source_sha256": identity["reference"]["source_sha256"],
            "official_static_parity": (
                {"status": "skipped_by_explicit_cli"}
                if identity["audit"]["skip_static_parity_preflight"]
                else {
                    "status": "passed",
                    "max_new_tokens": min(
                        generation["max_new_tokens"],
                        identity["audit"]["parity_max_new_tokens"],
                    ),
                    "official_acceptance_lengths": [1],
                    "policies": {
                        policy: {
                            "output_ids_match": True,
                            "acceptance_lengths_match": True,
                            "acceptance_lengths": [1],
                        }
                        for policy in ("stale", "rebuild")
                    },
                }
            ),
        },
        "models": {
            "target": {
                **{
                    key: value
                    for key, value in identity["target"].items()
                    if key != "revision"
                },
                "declared_revision": identity["target"]["revision"],
            },
            "draft": {
                **{
                    key: value
                    for key, value in identity["draft"].items()
                    if key != "revision"
                },
                "declared_revision": identity["draft"]["revision"],
            },
        },
        "tokenizer": identity["tokenizer"],
        "dataset": {
            "declared_revision": identity["dataset"]["revision"],
            "sha256": identity["dataset"]["sha256"],
            "sample_index": identity["dataset"]["sample_index"],
            "rendered_input_token_ids": {
                "serialization": "int64_le_c_order_v1",
                "shape": [1, generation["input_tokens"]],
                "sha256": "1" * 64,
            },
        },
        "generation": {
            "num_input_tokens": generation["input_tokens"],
            "num_output_tokens": generation["max_new_tokens"],
        },
        "parameters": {
            "max_new_tokens": generation["max_new_tokens"],
            "required_prefix_plus_block": generation["requested_total_context"],
            "stop_token_ids": None,
            "seed": generation["seed"],
            "deterministic": determinism["enabled"],
            "temperature": generation["temperature"],
            "block_size": generation["draft_block_size"],
            "mask_token_id": generation["mask_token_id"],
            "lr": optimization["learning_rate"],
            "optimizer": effective_optimizer,
            "weight_decay": optimization["weight_decay"],
            "rank": optimization["rank"],
            "adapter_seed": optimization["adapter_seed"],
            "proximal_lambda": optimization["proximal_lambda"],
            "update_stride": optimization["update_stride"],
            "position_weighting": optimization["position_weighting"],
            "position_decay_gamma": optimization["position_decay_gamma"],
            "loss_reduction": optimization["loss_reduction"],
            "adam_betas": optimization["adam_betas"],
            "adam_eps": optimization["adam_eps"],
            "draft_cache_policy": optimization["draft_cache_policy"],
            "dtype": identity["runtime"]["dtype"],
            "device": identity["runtime"]["device"],
            "enable_thinking": identity["dataset"]["enable_thinking"],
            "prompt_field": identity["dataset"]["prompt_field"],
            "messages_field": identity["dataset"]["messages_field"],
            "turns_field": identity["dataset"]["turns_field"],
            "projection_artifact": (
                None
                if identity["projection"] is None
                else identity["projection"]["path"]
            ),
            "audit_cuda_timing": identity["audit"]["cuda_timing"],
            "parameter_audit_stride": identity["audit"][
                "parameter_audit_stride"
            ],
            "parity_max_new_tokens": identity["audit"][
                "parity_max_new_tokens"
            ],
            "skip_static_parity_preflight": identity["audit"][
                "skip_static_parity_preflight"
            ],
        },
        "output": {"rounds_sha256": _sha256(rounds_path)},
    }
    if identity["projection"] is not None:
        summary["trainable_layout"] = {
            "projection_identity": {
                "artifact_file_sha256": identity["projection"]["sha256"]
            }
        }
    summary_path = plan.artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    runner._write_json_exclusive(
        plan.completion_path,
        {
            "schema_version": runner.SCHEMA_VERSION,
            "run_identity_sha256": plan.identity_sha256,
            "summary_path": "artifact/summary.json",
            "summary_sha256": _sha256(summary_path),
            "rounds_path": "artifact/rounds.jsonl",
            "rounds_sha256": _sha256(rounds_path),
            "rendered_input_token_ids_sha256": "1" * 64,
            "runtime_fingerprint_sha256": runner._sha256_json(
                runtime_fingerprint
            ),
        },
    )


def test_literal_model_keys_resume_without_completion_and_mismatch_fails_closed(
    tmp_path: Path,
):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("output-residual",))
    )
    plan = runner.build_run_plans(args)[0]
    for role in ("target", "draft"):
        assert "config.json_sha256" in plan.identity[role]
        assert "model.safetensors.index.json_sha256" in plan.identity[role]
    assert runner.completed_run_matches(plan) is False
    _write_completed_run(plan)
    assert runner.completed_run_matches(plan) is True
    assert runner.execute_plan(plan) == "resumed_complete"
    plan.completion_path.unlink()
    assert runner.completed_run_matches(plan) is True
    assert runner.execute_plan(plan) == "resumed_complete"
    assert plan.completion_path.is_file()

    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["parameters"]["rank"] = 32
    summary_path.write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match=r"summary parameters\.rank mismatch"):
        runner.completed_run_matches(plan)


def test_partial_directory_is_never_overwritten(tmp_path: Path):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    plan.run_dir.mkdir(parents=True)
    runner._write_json_exclusive(plan.identity_path, plan.plan_payload())
    with pytest.raises(ValueError, match="partial run directory"):
        runner.completed_run_matches(plan)


def test_partial_directory_is_archived_before_same_logical_run_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    plan.run_dir.mkdir(parents=True)
    runner._write_json_exclusive(plan.identity_path, plan.plan_payload())
    (plan.run_dir / "partial-marker.txt").write_text("attempt zero\n")

    calls = 0

    def fake_child(*_args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["pass_fds"]
        if calls == 1:
            return SimpleNamespace(returncode=7)
        plan.artifact_dir.mkdir(parents=True)
        (plan.artifact_dir / "summary.json").write_text("{}\n")
        (plan.artifact_dir / "rounds.jsonl").write_text("{}\n")
        return SimpleNamespace(returncode=0)

    hashes = ("1" * 64, "2" * 64, "3" * 64, "4" * 64)
    monkeypatch.setattr(runner.subprocess, "run", fake_child)
    monkeypatch.setattr(runner, "_validate_summary", lambda _plan: hashes)

    with pytest.raises(
        runner.RetryableRunError,
        match=r"exit code 7.*attempt-0002/run.*rerun the same logical plan",
    ):
        runner.execute_plan(plan)

    quarantine = runner._quarantine_root(plan)
    first = quarantine / "attempt-0001" / "run"
    second = quarantine / "attempt-0002" / "run"
    assert (first / "partial-marker.txt").read_text() == "attempt zero\n"
    assert (second / "run.log").is_file()
    first_record = json.loads(
        (quarantine / "attempt-0001" / "attempt.json").read_text()
    )
    second_record = json.loads(
        (quarantine / "attempt-0002" / "attempt.json").read_text()
    )
    assert first_record["reason"] == (
        "incomplete_or_invalid_attempt_detected_before_retry"
    )
    assert second_record["reason"] == "child_exit_7"
    assert first_record["expected_run_identity_sha256"] == plan.identity_sha256
    first_bytes = (first / "run_identity.json").read_bytes()

    assert runner.execute_plan(plan) == "completed"
    completion_bytes = plan.completion_path.read_bytes()
    assert runner.execute_plan(plan) == "resumed_complete"
    assert calls == 2
    assert plan.completion_path.read_bytes() == completion_bytes
    assert (first / "run_identity.json").read_bytes() == first_bytes
    assert sorted(path.name for path in quarantine.iterdir()) == [
        "attempt-0001",
        "attempt-0002",
    ]


def test_active_logical_run_lock_prevents_partial_quarantine(tmp_path: Path):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    plan.run_dir.mkdir(parents=True)
    runner._write_json_exclusive(plan.identity_path, plan.plan_payload())

    with runner._exclusive_run_lock(plan):
        with pytest.raises(RuntimeError, match="logical run is already active"):
            runner._execute_resumable_plan(plan)

    assert plan.run_dir.is_dir()
    assert not runner._quarantine_root(plan).exists()


def test_dry_run_creates_no_output_directories(tmp_path: Path, capsys):
    argv = _base_argv(tmp_path, modes=("static",)) + ["--dry-run"]
    assert runner.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry_run"
    assert payload["optimizer_selection"]["status"] == "locked"
    assert len(payload["runs"]) == 1
    assert not (tmp_path / "runs").exists()


def test_prepare_identity_lock_is_atomic_idempotent_and_runs_no_model(
    tmp_path: Path, capsys, monkeypatch
):
    argv = _base_argv(tmp_path, modes=("static",)) + [
        "--prepare-identity-lock"
    ]

    def reject_execution(_plan):
        raise AssertionError("identity-lock preparation launched a model run")

    monkeypatch.setattr(runner, "execute_plan", reject_execution)
    assert runner.main(argv) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "identity_lock_ready"
    lock_path = Path(first["path"])
    assert lock_path == tmp_path / "runs" / "artifact_identity_lock.json"
    assert _sha256(lock_path) == first["sha256"]
    assert sorted(item.name for item in lock_path.parent.iterdir()) == [
        "artifact_identity_lock.json"
    ]

    assert runner.main(argv) == 0
    second = json.loads(capsys.readouterr().out)
    assert second == first
    assert sorted(item.name for item in lock_path.parent.iterdir()) == [
        "artifact_identity_lock.json"
    ]


def test_prepare_identity_lock_fails_closed_on_ambiguous_modes_or_dry_run(
    tmp_path: Path,
):
    argv = _base_argv(tmp_path, modes=("static", "tail-lora")) + [
        "--prepare-identity-lock"
    ]
    with pytest.raises(ValueError, match="requires exactly --modes static"):
        runner.main(argv)
    assert not (tmp_path / "runs").exists()

    dry_run_root = tmp_path / "dry-run-conflict"
    dry_run_root.mkdir()
    argv = _base_argv(dry_run_root, modes=("static",)) + [
        "--prepare-identity-lock",
        "--dry-run",
    ]
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.main(argv)
    assert not (dry_run_root / "runs").exists()


def test_prepare_identity_lock_requires_locked_selection(tmp_path: Path):
    argv = _base_argv(tmp_path, modes=("static",)) + [
        "--prepare-identity-lock"
    ]
    selection_path = Path(argv[argv.index("--selected-optimizer-config") + 1])
    selection = json.loads(selection_path.read_text())
    selection["status"] = "provisional_pilot"
    selection["evidence_artifacts"] = []
    selection_path.write_text(json.dumps(selection) + "\n")
    with pytest.raises(ValueError, match="provisional_pilot"):
        runner.main(argv)
    assert not (tmp_path / "runs").exists()


def test_provisional_optimizer_selection_is_auditable_but_not_executable(
    tmp_path: Path,
):
    argv = _base_argv(tmp_path, modes=("static",))
    selection_path = Path(argv[argv.index("--selected-optimizer-config") + 1])
    selection = json.loads(selection_path.read_text())
    selection["status"] = "provisional_pilot"
    selection["evidence_artifacts"] = []
    selection_path.write_text(json.dumps(selection) + "\n")
    args = runner.build_parser().parse_args(argv)
    plan = runner.build_run_plans(args)[0]
    assert plan.identity["optimizer_selection"]["status"] == "provisional_pilot"
    with pytest.raises(ValueError, match="optimizer selection is provisional"):
        runner.execute_plan(plan)
    assert not plan.run_dir.exists()


def test_optimizer_selection_requires_two_samples_and_locks_grid_membership(
    tmp_path: Path,
):
    argv = _base_argv(tmp_path, modes=("static",))
    selection_path = Path(argv[argv.index("--selected-optimizer-config") + 1])
    selection = json.loads(selection_path.read_text())
    selection["calibration"]["sample_ids"] = ["only-one"]
    selection_path.write_text(json.dumps(selection) + "\n")
    with pytest.raises(ValueError, match="exactly two distinct"):
        runner.build_run_plans(runner.build_parser().parse_args(argv))

    selection["calibration"]["sample_ids"] = ["sample-0", "sample-419"]
    selection["locked_configs"]["tail-lora"]["rank"] = 64
    selection_path.write_text(json.dumps(selection) + "\n")
    with pytest.raises(ValueError, match="outside candidate_grid"):
        runner.build_run_plans(runner.build_parser().parse_args(argv))


def test_optimizer_selection_evidence_cannot_change_after_plan(tmp_path: Path):
    argv = _base_argv(tmp_path, modes=("static",))
    plan = runner.build_run_plans(runner.build_parser().parse_args(argv))[0]
    evidence_path = Path(
        plan.identity["optimizer_selection"]["evidence_artifacts"][0]["path"]
    )
    evidence_path.write_text('{"samples":[419,0]}\n')
    with pytest.raises(ValueError, match="evidence artifact changed"):
        runner.execute_plan(plan)
    assert not plan.run_dir.exists()


def test_locked_optimizer_table_must_match_deterministic_summary(tmp_path: Path):
    argv = _base_argv(tmp_path, modes=("tail-lora",))
    selection_path = Path(argv[argv.index("--selected-optimizer-config") + 1])
    selection = json.loads(selection_path.read_text())
    selection["locked_configs"]["tail-lora"].update(
        {"optimizer": "adam", "weight_decay": 0.0}
    )
    selection_path.write_text(json.dumps(selection) + "\n")
    with pytest.raises(
        ValueError,
        match=r"selection summary selected_configs\.tail-lora mismatch",
    ):
        runner.build_run_plans(runner.build_parser().parse_args(argv))


def test_input_tokens_can_be_bound_to_one_validated_preflight(tmp_path: Path):
    argv = _base_argv(tmp_path, modes=("static",))
    input_index = argv.index("--input-tokens")
    del argv[input_index : input_index + 2]
    dataset = Path(argv[argv.index("--dataset") + 1])
    target = Path(argv[argv.index("--target-model") + 1])
    preflight = tmp_path / "preflight-summary.json"
    preflight.write_text(
        json.dumps(
            {
                "reference": {"declared_revision": "94e4abc"},
                "models": {
                    "target": {"declared_revision": "target-rev"},
                    "draft": {"declared_revision": "draft-rev"},
                },
                "tokenizer": runner._tokenizer_identity(target),
                "dataset": {
                    "sample_index": 419,
                    "sha256": _sha256(dataset),
                    "declared_revision": "dataset-rev",
                    "rendered_input_token_ids": {
                        "sha256": "2" * 64,
                    },
                },
                "parameters": {
                    "block_size": 16,
                    "enable_thinking": True,
                    "prompt_field": "prompt",
                    "messages_field": "messages",
                    "turns_field": "turns",
                },
                "generation": {"num_input_tokens": 109},
            }
        )
        + "\n"
    )
    argv.extend(("--preflight-summary", str(preflight)))
    plan = runner.build_run_plans(runner.build_parser().parse_args(argv))[0]
    source = plan.identity["dataset"]["input_token_source"]
    assert source["kind"] == "validated_preflight_summary"
    assert source["sha256"] == _sha256(preflight)
    assert plan.identity["generation"]["max_new_tokens"] == 8068
    assert plan.identity["dataset"]["rendered_input_token_ids_sha256"] == (
        "2" * 64
    )


def test_weight_shard_content_hash_is_locked_once_and_edit_is_rejected(
    tmp_path: Path, monkeypatch
):
    argv = _base_argv(tmp_path, modes=("static",))
    args = runner.build_parser().parse_args(argv)
    plan = runner.build_run_plans(args)[0]
    target = Path(args.target_model)
    shard = target / "model.safetensors"
    before = plan.identity["target"]["weight_files"][0]["sha256"]
    runner._ensure_artifact_identity_lock(plan)

    original_sha256_file = runner._sha256_file

    def reject_weight_rehash(path: Path) -> str:
        if path.suffix == ".safetensors":
            raise AssertionError("completed sweep resume re-hashed a model shard")
        return original_sha256_file(path)

    monkeypatch.setattr(runner, "_sha256_file", reject_weight_rehash)
    resumed = runner.build_run_plans(args)[0]
    assert resumed.identity["target"]["weight_files"][0]["sha256"] == before

    monkeypatch.setattr(runner, "_sha256_file", original_sha256_file)
    shard.write_bytes(b"target-weightt")  # same byte count, different content
    assert shard.stat().st_size == plan.identity["target"]["weight_files"][0]["bytes"]
    with pytest.raises(ValueError, match="changed since the immutable"):
        runner.build_run_plans(args)


def test_tokenizer_same_size_edit_rejects_locked_sweep(tmp_path: Path):
    argv = _base_argv(tmp_path, modes=("static",))
    args = runner.build_parser().parse_args(argv)
    plan = runner.build_run_plans(args)[0]
    runner._ensure_artifact_identity_lock(plan)
    tokenizer = Path(args.target_model) / "tokenizer.json"
    original = tokenizer.read_bytes()
    replacement = bytearray(original)
    replacement[-2] = ord("1") if replacement[-2] != ord("1") else ord("2")
    tokenizer.write_bytes(bytes(replacement))
    assert tokenizer.stat().st_size == len(original)
    with pytest.raises(ValueError, match="changed since the immutable"):
        runner.build_run_plans(args)


def test_completion_binds_rendered_input_token_ids(tmp_path: Path):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    _write_completed_run(plan)
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["dataset"]["rendered_input_token_ids"]["sha256"] = "3" * 64
    summary_path.write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="completion record mismatch"):
        runner.completed_run_matches(plan)


def test_runtime_fingerprint_must_match_planned_numerical_runtime(tmp_path: Path):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    _write_completed_run(plan)
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["runtime_fingerprint"]["attention_implementation"] = "flash_attention_2"
    summary_path.write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="runtime fingerprint attention_implementation"):
        runner.completed_run_matches(plan)


def test_run_identity_binds_allocator_configuration(
    tmp_path: Path, monkeypatch
):
    allocator = (
        "backend:native,expandable_segments:True,"
        "garbage_collection_threshold:0.70,roundup_power2_divisions:8"
    )
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", allocator)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    assert plan.identity["runtime"]["allocator_config"] == {
        "PYTORCH_CUDA_ALLOC_CONF": allocator,
        "PYTORCH_ALLOC_CONF": None,
    }
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
    child_environment = runner._subprocess_environment(plan)
    assert child_environment["PYTORCH_CUDA_ALLOC_CONF"] == allocator
    assert "PYTORCH_ALLOC_CONF" not in child_environment

    _write_completed_run(plan)
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["runtime_fingerprint"]["allocator_config"][
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = None
    summary_path.write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="runtime fingerprint allocator config"):
        runner._validate_summary(plan)


def test_runtime_fingerprint_must_match_planned_determinism(tmp_path: Path):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    _write_completed_run(plan)
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["runtime_fingerprint"]["allow_tf32"]["matmul"] = True
    summary_path.write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="runtime fingerprint TF32"):
        runner._validate_summary(plan)


def test_mask_and_parity_are_behavior_bound(tmp_path: Path):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    _write_completed_run(plan)
    summary_path = plan.artifact_dir / "summary.json"
    original = json.loads(summary_path.read_text())

    changed_mask = json.loads(json.dumps(original))
    changed_mask["parameters"]["mask_token_id"] += 1
    summary_path.write_text(json.dumps(changed_mask) + "\n")
    with pytest.raises(ValueError, match=r"parameters\.mask_token_id mismatch"):
        runner._validate_summary(plan)

    changed_parity = json.loads(json.dumps(original))
    changed_parity["reference"]["official_static_parity"]["policies"][
        "stale"
    ]["output_ids_match"] = False
    summary_path.write_text(json.dumps(changed_parity) + "\n")
    with pytest.raises(ValueError, match="static parity stale did not pass"):
        runner._validate_summary(plan)


def test_current_parity_schema_requires_only_the_official_stale_policy(
    tmp_path: Path,
):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    _write_completed_run(plan)
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    parity = summary["reference"]["official_static_parity"]
    parity["classification"] = (
        "official_stale_cache_block_verifier_reconstruction"
    )
    parity["official_policy"] = "stale"
    parity["policies"].pop("rebuild")
    summary_path.write_text(json.dumps(summary) + "\n")

    runner._validate_summary(plan)


def test_current_parity_schema_rejects_rebuild_as_official(tmp_path: Path):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    _write_completed_run(plan)
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    parity = summary["reference"]["official_static_parity"]
    parity["classification"] = (
        "official_stale_cache_block_verifier_reconstruction"
    )
    parity["official_policy"] = "stale"
    summary_path.write_text(json.dumps(summary) + "\n")

    with pytest.raises(ValueError, match="static parity policies are incomplete"):
        runner._validate_summary(plan)


def test_legacy_parity_schema_still_requires_both_policies(tmp_path: Path):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    _write_completed_run(plan)
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["reference"]["official_static_parity"]["policies"].pop("rebuild")
    summary_path.write_text(json.dumps(summary) + "\n")

    with pytest.raises(ValueError, match="static parity policies are incomplete"):
        runner._validate_summary(plan)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("classification", "official_rebuild_cache_block_verifier_reconstruction"),
        ("official_policy", "rebuild"),
    ),
)
def test_current_parity_schema_rejects_wrong_discriminator(
    tmp_path: Path, field: str, value: str
):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    _write_completed_run(plan)
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    parity = summary["reference"]["official_static_parity"]
    parity["classification"] = (
        "official_stale_cache_block_verifier_reconstruction"
    )
    parity["official_policy"] = "stale"
    parity["policies"].pop("rebuild")
    parity[field] = value
    summary_path.write_text(json.dumps(summary) + "\n")

    with pytest.raises(ValueError, match="static parity classification is invalid"):
        runner._validate_summary(plan)


def test_tampered_completionless_attempt_is_archived_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    _write_completed_run(plan)
    plan.completion_path.unlink()
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["run_attestation"]["command_sha256"] = "2" * 64
    summary_path.write_text(json.dumps(summary) + "\n")

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=9),
    )
    with pytest.raises(runner.RetryableRunError, match="exit code 9"):
        runner.execute_plan(plan)
    assert not plan.completion_path.exists()
    quarantine = runner._quarantine_root(plan)
    archived_summary = (
        quarantine / "attempt-0001" / "run" / "artifact" / "summary.json"
    )
    assert (
        json.loads(archived_summary.read_text())["run_attestation"][
            "command_sha256"
        ]
        == "2" * 64
    )
    assert (quarantine / "attempt-0002" / "run" / "run.log").is_file()


def test_invalid_completed_evidence_is_never_archived_or_overwritten(
    tmp_path: Path,
):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    _write_completed_run(plan)
    completion_bytes = plan.completion_path.read_bytes()
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["run_attestation"]["command_sha256"] = "5" * 64
    summary_path.write_text(json.dumps(summary) + "\n")

    with pytest.raises(
        runner.ImmutableCompletionError,
        match="immutable completion evidence.*refusing to archive or overwrite",
    ):
        runner.execute_plan(plan)
    assert plan.run_dir.is_dir()
    assert plan.completion_path.read_bytes() == completion_bytes
    assert not runner._quarantine_root(plan).exists()


def test_cuda_driver_query_failure_is_explicitly_allowed(tmp_path: Path):
    args = runner.build_parser().parse_args(
        _base_argv(tmp_path, modes=("static",))
    )
    plan = runner.build_run_plans(args)[0]
    _write_completed_run(plan)
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["runtime_fingerprint"]["cuda_driver_version"] = None
    summary_path.write_text(json.dumps(summary) + "\n")
    # Completion still binds the original summary, so validate the harness
    # artifact directly to isolate the null-driver contract.
    _summary_hash, _rounds_hash, _input_hash, runtime_hash = (
        runner._validate_summary(plan)
    )
    assert len(runtime_hash) == 64


def _safe_aggregate() -> dict[str, object]:
    return {
        "evidence_eligible": True,
        "safe_for_selection": True,
        "all_outputs_exact_static": True,
        "all_losses_and_gradients_finite": True,
        "ineligibility_reasons": [],
    }


def _analysis_digest(payload: dict[str, object]) -> dict[str, object]:
    value = dict(payload)
    value["analysis_sha256"] = runner._sha256_json(value)
    return value


def _stage1_analysis(*, unsafe_mode: str | None = None, boundary_mode: str | None = None):
    rows = [
        {
            "candidate_id": "static",
            "mode": "static",
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "rank": None,
            "adapter_seed": None,
            "aggregate": {},
        }
    ]
    configs = {
        "full-drafter": ("adamw", 3e-6, 0.0, None),
        "full-rank-tail": ("adamw", 1e-5, 0.0, None),
        "output-residual": ("adam", 3e-4, 0.0, 16),
    }
    decisions = []
    for mode, (optimizer, lr, weight_decay, rank) in configs.items():
        candidate_id = f"{mode}-safe"
        aggregate = _safe_aggregate()
        rows.append(
            {
                "candidate_id": candidate_id,
                "mode": mode,
                "optimizer": optimizer,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "rank": rank,
                "adapter_seed": 0 if rank is not None else None,
                "aggregate": aggregate,
            }
        )
        boundary = {
            "at_group_boundary": mode == boundary_mode,
            "at_optimizer_weight_decay_boundary": mode == boundary_mode,
            "optimizer_weight_decay_bounds": {
                "minimum": lr / 3.0,
                "maximum": lr * 3.0,
            },
            "requires_grid_extension_before_optimum_claim": mode
            == boundary_mode,
        }
        decisions.append(
            {
                "mode": mode,
                "rank": rank,
                "status": (
                    "no_safe_selection"
                    if mode == unsafe_mode
                    else "local_grid_winner"
                ),
                "safe_candidate_count": 0 if mode == unsafe_mode else 1,
                "winner": (
                    None
                    if mode == unsafe_mode
                    else {
                        "candidate_id": candidate_id,
                        "optimizer": optimizer,
                        "learning_rate": lr,
                        "weight_decay": weight_decay,
                        "rank": rank,
                        "aggregate": aggregate,
                        "learning_rate_boundary": boundary,
                    }
                ),
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": runner.SCHEMA_V3_STAGE1_KIND,
        "status": "complete",
        "sample_indices": [0, 419],
        "candidate_specification": {
            "path": "stage1-candidates.json",
            "file_sha256": "1" * 64,
            "content_sha256": "2" * 64,
            "study_id": "stage1-study",
        },
        "artifact_identity_lock": {
            "path": "artifact_identity_lock.json",
            "file_sha256": "3" * 64,
            "content_sha256": "4" * 64,
        },
        "candidate_rows": rows,
        "candidate_rows_sha256": runner._sha256_json(rows),
        "selection_decisions": decisions,
        "selection_decisions_sha256": runner._sha256_json(decisions),
        "selection_rule_sha256": "5" * 64,
        "source_artifact_set_sha256": "6" * 64,
        "analysis_hash_scheme": "canonical_json_without_analysis_sha256_v1",
    }
    return _analysis_digest(payload)


def _stage2_analysis(
    stage1: dict[str, object],
    stage1_path: Path,
    *,
    unsafe_mode: str | None = None,
    rank_boundary_mode: str | None = None,
    lr_boundary_mode: str | None = None,
):
    rows = []
    tuned = []
    for mode, optimizer, lr, rank in (
        ("drafter-lora", "adamw", 1e-4, 16),
        ("tail-lora", "adam", 3e-4, 32),
    ):
        candidate_id = f"{mode}-r{rank}-safe"
        aggregate = _safe_aggregate()
        rows.append(
            {
                "candidate_id": candidate_id,
                "mode": mode,
                "optimizer": optimizer,
                "learning_rate": lr,
                "weight_decay": 0.0,
                "rank": rank,
                "adapter_seed": 0,
                "aggregate": aggregate,
            }
        )
        tuned.append(
            {
                "mode": mode,
                "status": (
                    "no_safe_selection"
                    if mode == unsafe_mode
                    else "bounded_rank_winner"
                ),
                "winner": (
                    None
                    if mode == unsafe_mode
                    else {
                        "candidate_id": candidate_id,
                        "mode": mode,
                        "optimizer": optimizer,
                        "learning_rate": lr,
                        "weight_decay": 0.0,
                        "rank": rank,
                        "adapter_seed": 0,
                        "prompt_safety": [
                            {"sample_index": 0, "safe_nonnegative": True},
                            {"sample_index": 419, "safe_nonnegative": True},
                        ],
                        "aggregate": aggregate,
                    }
                ),
                "rank_boundary": {
                    "requires_rank_grid_extension_before_optimum_claim": mode
                    == rank_boundary_mode,
                },
                "requires_lr_grid_extension_before_optimum_claim": mode
                == lr_boundary_mode,
                "winner_learning_rate_boundary": {
                    "requires_lr_grid_extension_before_optimum_claim": mode
                    == lr_boundary_mode,
                },
            }
        )
    comparisons = {"tuned_envelope": tuned, "fixed_center_control": []}
    binding = {
        "analysis_file_sha256": _sha256(stage1_path),
        "analysis_sha256": stage1["analysis_sha256"],
        "selection_decisions_sha256": stage1[
            "selection_decisions_sha256"
        ],
        "source_artifact_set_sha256": stage1[
            "source_artifact_set_sha256"
        ],
        "candidate_specification_file_sha256": stage1[
            "candidate_specification"
        ]["file_sha256"],
        "candidate_specification_content_sha256": stage1[
            "candidate_specification"
        ]["content_sha256"],
        "source_study_id": stage1["candidate_specification"]["study_id"],
        "artifact_identity_lock_file_sha256": stage1[
            "artifact_identity_lock"
        ]["file_sha256"],
        "artifact_identity_lock_content_sha256": stage1[
            "artifact_identity_lock"
        ]["content_sha256"],
    }
    source_attestation = {
        "portable_evidence_core": {
            "stage2_candidate_specification": {
                "file_sha256": "7" * 64,
                "content_sha256": "8" * 64,
            }
        },
        "locator_bound_provenance": {"stage1_binding": binding},
    }
    omissions: list[object] = []
    pareto: dict[str, object] = {}
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": runner.SCHEMA_V3_STAGE2_KIND,
        "status": "complete",
        "sample_indices": [0, 419],
        "candidate_rows": rows,
        "candidate_rows_sha256": runner._sha256_json(rows),
        "comparisons": comparisons,
        "comparisons_sha256": runner._sha256_json(comparisons),
        "mode_omissions": omissions,
        "mode_omissions_sha256": runner._sha256_json(omissions),
        "pareto": pareto,
        "pareto_sha256": runner._sha256_json(pareto),
        "source_attestation": source_attestation,
        "source_attestation_sha256": runner._sha256_json(source_attestation),
        "selection_rule_sha256": "9" * 64,
        "analysis_hash_scheme": "canonical_json_without_analysis_sha256_v1",
    }
    return _analysis_digest(payload)


def _write_schema_v3_analyses(
    tmp_path: Path,
    *,
    unsafe_stage1: str | None = None,
    stage1_boundary: str | None = None,
    unsafe_stage2: str | None = None,
    stage2_rank_boundary: str | None = None,
    stage2_lr_boundary: str | None = None,
) -> tuple[Path, Path]:
    stage1 = _stage1_analysis(
        unsafe_mode=unsafe_stage1, boundary_mode=stage1_boundary
    )
    stage1_path = tmp_path / "selection-analysis.json"
    stage1_path.write_text(json.dumps(stage1, indent=2, sort_keys=True) + "\n")
    stage2 = _stage2_analysis(
        stage1,
        stage1_path,
        unsafe_mode=unsafe_stage2,
        rank_boundary_mode=stage2_rank_boundary,
        lr_boundary_mode=stage2_lr_boundary,
    )
    stage2_path = tmp_path / "rank-analysis.json"
    stage2_path.write_text(json.dumps(stage2, indent=2, sort_keys=True) + "\n")
    return stage1_path, stage2_path


def _fake_schema_v3_closure(**_kwargs):
    return {
        "stage1_candidate_specification": {
            "path": "/attested/stage1.json",
            "file_sha256": "1" * 64,
            "content_sha256": "2" * 64,
        },
        "stage2_candidate_specification": {
            "path": "/attested/stage2.json",
            "file_sha256": "7" * 64,
            "content_sha256": "8" * 64,
        },
    }


def test_schema_v3_selection_builds_six_modes_and_binds_run_identity(
    tmp_path: Path, monkeypatch
):
    stage1_path, stage2_path = _write_schema_v3_analyses(tmp_path)
    monkeypatch.setattr(
        runner, "_verify_schema_v3_analysis_closure", _fake_schema_v3_closure
    )
    argv = _base_argv(tmp_path, modes=runner.MODE_ORDER)
    index = argv.index("--selected-optimizer-config")
    del argv[index : index + 2]
    argv.extend(
        [
            "--stage1-analysis",
            str(stage1_path),
            "--stage2-analysis",
            str(stage2_path),
        ]
    )
    plans = runner.build_run_plans(runner.build_parser().parse_args(argv))
    assert [plan.identity["mode"] for plan in plans] == list(runner.MODE_ORDER)
    selection = plans[0].identity["optimizer_selection"]
    assert selection["kind"] == runner.SCHEMA_V3_SELECTION_KIND
    assert selection["schema_v3_selection"]["boundary_gate"]["status"] == "passed"
    assert selection["schema_v3_selection"]["mode_parameter_scopes"][
        "full-rank-tail"
    ] == "cache_safe_full_rank_tail_only"
    assert selection["schema_v3_selection"]["mode_parameter_scopes"][
        "full-drafter"
    ] == "drafter_all_trainable_parameters"
    assert Path(selection["schema_v3_selection"]["stage1_analysis"]["path"]) == (
        stage1_path.resolve()
    )
    assert Path(selection["schema_v3_selection"]["stage2_analysis"]["path"]) == (
        stage2_path.resolve()
    )
    by_mode = {plan.identity["mode"]: plan for plan in plans}
    assert by_mode["drafter-lora"].identity["optimization"]["rank"] == 16
    assert by_mode["tail-lora"].identity["optimization"]["rank"] == 32
    assert by_mode["output-residual"].identity["optimization"][
        "adapter_seed"
    ] == 0


def test_schema_v3_selection_rejects_stage1_file_hash_mismatch(
    tmp_path: Path, monkeypatch
):
    stage1_path, stage2_path = _write_schema_v3_analyses(tmp_path)
    monkeypatch.setattr(
        runner, "_verify_schema_v3_analysis_closure", _fake_schema_v3_closure
    )
    stage1_path.write_text(stage1_path.read_text() + "\n")
    with pytest.raises(ValueError, match="analysis_file_sha256 mismatch"):
        runner._load_schema_v3_selection(str(stage1_path), str(stage2_path))


@pytest.mark.parametrize(
    ("stage1_mode", "stage2_mode"),
    [("full-drafter", None), (None, "tail-lora")],
)
def test_schema_v3_selection_rejects_missing_safe_winner(
    tmp_path: Path, monkeypatch, stage1_mode: str | None, stage2_mode: str | None
):
    stage1_path, stage2_path = _write_schema_v3_analyses(
        tmp_path, unsafe_stage1=stage1_mode, unsafe_stage2=stage2_mode
    )
    monkeypatch.setattr(
        runner, "_verify_schema_v3_analysis_closure", _fake_schema_v3_closure
    )
    with pytest.raises(ValueError, match="no safe winner"):
        runner._load_schema_v3_selection(str(stage1_path), str(stage2_path))


@pytest.mark.parametrize(
    ("stage1_mode", "stage2_rank_mode", "stage2_lr_mode", "message"),
    [
        ("output-residual", None, None, "learning-rate boundary"),
        (None, "drafter-lora", None, "rank boundary"),
        (None, None, "tail-lora", "learning-rate boundary"),
    ],
)
def test_schema_v3_selection_rejects_unresolved_boundary_winner(
    tmp_path: Path,
    monkeypatch,
    stage1_mode: str | None,
    stage2_rank_mode: str | None,
    stage2_lr_mode: str | None,
    message: str,
):
    stage1_path, stage2_path = _write_schema_v3_analyses(
        tmp_path,
        stage1_boundary=stage1_mode,
        stage2_rank_boundary=stage2_rank_mode,
        stage2_lr_boundary=stage2_lr_mode,
    )
    monkeypatch.setattr(
        runner, "_verify_schema_v3_analysis_closure", _fake_schema_v3_closure
    )
    with pytest.raises(ValueError, match=message):
        runner._load_schema_v3_selection(str(stage1_path), str(stage2_path))
