from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "experiments"
    / "run_dflash_tts_calibration_sweep.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_dflash_tts_calibration_sweep", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_spec() -> dict:
    common_tail = {
        "mode": "tail-lora",
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "rank": 16,
    }
    return {
        "schema_version": 3,
        "kind": calibration.SPEC_KIND,
        "study_id": "mock-rank-lr-calibration-v3",
        "evidence_scope": calibration.EVIDENCE_SCOPE,
        "max_new_tokens": 2048,
        "samples": [
            {
                "sample_index": 0,
                "input_tokens": 71,
                "rendered_input_token_ids_sha256": "1" * 64,
            },
            {
                "sample_index": 419,
                "input_tokens": 109,
                "rendered_input_token_ids_sha256": "2" * 64,
            },
        ],
        "candidates": [
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
                "candidate_id": "tail-lora-r8-adam-lr3e-4",
                "mode": "tail-lora",
                "optimizer": "adam",
                "learning_rate": 3e-4,
                "weight_decay": 0.0,
                "rank": 8,
                "draft_cache_policy": "stale",
                "diagnostic_kind": "selection",
                "parameter_audit_stride": 0,
            },
            {
                "candidate_id": "tail-lora-r16-adamw-lr1e-4-wd1e-2",
                **common_tail,
                "draft_cache_policy": "stale",
                "diagnostic_kind": "selection",
                "parameter_audit_stride": 0,
            },
            {
                "candidate_id": "tail-lora-cache-stale-diagnostic",
                **common_tail,
                "draft_cache_policy": "stale",
                "diagnostic_kind": "cache-policy-diagnostic",
                "parameter_audit_stride": 0,
            },
            {
                "candidate_id": "tail-lora-cache-rebuild-diagnostic",
                **common_tail,
                "draft_cache_policy": "rebuild",
                "diagnostic_kind": "cache-policy-diagnostic",
                "parameter_audit_stride": 0,
            },
            {
                "candidate_id": "tail-lora-parameter-audit",
                **common_tail,
                "draft_cache_policy": "stale",
                "diagnostic_kind": "parameter-audit",
                "parameter_audit_stride": 7,
            },
        ],
    }


def _base_argv(tmp_path: Path) -> tuple[list[str], Path]:
    harness = tmp_path / "dflash_tts_reference.py"
    harness.write_text("# fake harness\n")
    reference = tmp_path / "reference"
    (reference / "dflash").mkdir(parents=True)
    (reference / "dflash" / "model.py").write_text("# reference\n")
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    for model, label in ((target, "target"), (draft, "draft")):
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model": label}) + "\n")
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "model.safetensors"}}) + "\n"
        )
        (model / "model.safetensors").write_bytes(label.encode())
    (target / "tokenizer_config.json").write_text('{"chat_template":"x"}\n')
    (target / "tokenizer.json").write_text('{"version":"1"}\n')
    dataset = tmp_path / "math500.jsonl"
    dataset.write_text('{"turns":["question"]}\n')
    candidate_spec = tmp_path / "candidates.json"
    candidate_spec.write_text(json.dumps(_candidate_spec(), sort_keys=True) + "\n")
    argv = [
        "--candidate-spec",
        str(candidate_spec),
        "--python",
        sys.executable,
        "--harness",
        str(harness),
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
        "--output-root",
        str(tmp_path / "runs"),
        "--mask-token-id",
        "151669",
    ]
    return argv, candidate_spec


def _flag(command: tuple[str, ...], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def _write_completed_run(plan: calibration.frozen.RunPlan) -> None:
    """Create a real schema-v3 artifact accepted by the shared validator."""

    frozen = calibration.frozen
    plan.run_dir.mkdir(parents=True)
    frozen._write_json_exclusive(plan.identity_path, plan.plan_payload())
    plan.artifact_dir.mkdir()
    rounds_path = plan.artifact_dir / "rounds.jsonl"
    rounds_path.write_text('{"round_index":0}\n')
    identity = plan.identity
    generation = identity["generation"]
    optimization = identity["optimization"]
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
        "allocator_config": {
            "PYTORCH_CUDA_ALLOC_CONF": None,
            "PYTORCH_ALLOC_CONF": None,
        },
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
        "cublas_workspace_config": determinism["cublas_workspace_config"],
        "sdpa_backends": determinism["sdpa_backends"],
        "determinism_contract": determinism,
        "gpu": {
            "name": "Test GPU",
            "total_memory_bytes": 96 << 30,
            "compute_capability": "12.0",
            "device_index": 0,
        },
    }
    rendered_sha256 = identity["dataset"][
        "rendered_input_token_ids_sha256"
    ]
    parity = {
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
    summary = {
        "schema_version": frozen.HARNESS_ARTIFACT_SCHEMA_VERSION,
        "status": "complete_reference_run",
        "mode": identity["mode"],
        "run_attestation": frozen._expected_run_attestation(plan),
        "harness": {
            "source_sha256": identity["runtime"]["harness"]["sha256"],
            "artifact_schema_version": frozen.HARNESS_ARTIFACT_SCHEMA_VERSION,
        },
        "artifact_identity": {
            "verification_status": "fully_verified_content_sha256_v1",
            "lock": identity["runtime"]["artifact_identity_lock"],
        },
        "runtime_fingerprint": runtime_fingerprint,
        "reference": {
            "declared_revision": identity["reference"]["revision"],
            "source_sha256": identity["reference"]["source_sha256"],
            "official_static_parity": parity,
        },
        "models": {
            role: {
                **{
                    key: value
                    for key, value in identity[role].items()
                    if key != "revision"
                },
                "declared_revision": identity[role]["revision"],
            }
            for role in ("target", "draft")
        },
        "tokenizer": identity["tokenizer"],
        "dataset": {
            "declared_revision": identity["dataset"]["revision"],
            "sha256": identity["dataset"]["sha256"],
            "sample_index": identity["dataset"]["sample_index"],
            "rendered_input_token_ids": {
                "serialization": "int64_le_c_order_v1",
                "shape": [1, generation["input_tokens"]],
                "sha256": rendered_sha256,
            },
        },
        "generation": {
            "num_input_tokens": generation["input_tokens"],
            "num_output_tokens": generation["max_new_tokens"],
        },
        "parameters": {
            "max_new_tokens": generation["max_new_tokens"],
            "required_prefix_plus_block": generation[
                "requested_total_context"
            ],
            "stop_token_ids": None,
            "seed": generation["seed"],
            "deterministic": determinism["enabled"],
            "temperature": generation["temperature"],
            "block_size": generation["draft_block_size"],
            "mask_token_id": generation["mask_token_id"],
            "lr": optimization["learning_rate"],
            "optimizer": (
                None
                if identity["mode"] == "static"
                else optimization["optimizer"].upper()
            ),
            "weight_decay": optimization["weight_decay"],
            "rank": optimization["rank"],
            "adapter_seed": optimization["adapter_seed"],
            "proximal_lambda": optimization["proximal_lambda"],
            "update_stride": optimization["update_stride"],
            "position_weighting": optimization["position_weighting"],
            "position_decay_gamma": optimization[
                "position_decay_gamma"
            ],
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
            "projection_artifact": None,
            "audit_cuda_timing": identity["audit"]["cuda_timing"],
            "parameter_audit_stride": identity["audit"][
                "parameter_audit_stride"
            ],
            "parity_max_new_tokens": identity["audit"][
                "parity_max_new_tokens"
            ],
            "skip_static_parity_preflight": False,
        },
        "output": {"rounds_sha256": _sha256(rounds_path)},
    }
    summary_path = plan.artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    frozen._write_json_exclusive(
        plan.completion_path,
        frozen._completion(
            plan,
            _sha256(summary_path),
            _sha256(rounds_path),
            rendered_sha256,
            frozen._sha256_json(runtime_fingerprint),
        ),
    )


def test_explicit_candidates_build_in_sample_then_list_order(tmp_path: Path):
    argv, candidate_spec = _base_argv(tmp_path)
    args = calibration.build_parser().parse_args(argv)
    calibration._validate_args(args)
    plans = calibration.build_run_plans(args)

    assert len(plans) == 12
    assert [
        (
            plan.identity["dataset"]["sample_index"],
            plan.identity["calibration_candidate"]["candidate_id"],
        )
        for plan in plans
    ] == [
        (sample, candidate["candidate_id"])
        for sample in (0, 419)
        for candidate in _candidate_spec()["candidates"]
    ]
    assert all(
        plan.identity["schema_version"] == 3
        and plan.identity["generation"]["max_new_tokens"] == 2048
        and _flag(plan.command, "--max-new-tokens") == "2048"
        and plan.identity["candidate_specification"]["file_sha256"]
        == _sha256(candidate_spec)
        for plan in plans
    )
    assert plans[0].identity["generation"]["required_prefix_plus_block"] == 2134
    assert plans[6].identity["generation"]["required_prefix_plus_block"] == 2172
    runtime = plans[0].identity["runtime"]
    assert runtime["calibration_orchestrator"]["sha256"] == _sha256(
        Path(calibration.__file__)
    )
    assert runtime["frozen_run_validator"]["sha256"] == _sha256(
        Path(calibration.frozen.__file__)
    )

    by_id = {
        plan.identity["calibration_candidate"]["candidate_id"]: plan
        for plan in plans[:6]
    }
    primary = by_id["tail-lora-r16-adamw-lr1e-4-wd1e-2"]
    assert primary.identity["calibration_candidate"]["selection_eligible"] is True
    assert _flag(primary.command, "--optimizer") == "adamw"
    assert float(_flag(primary.command, "--weight-decay")) == pytest.approx(1e-2)
    assert _flag(primary.command, "--rank") == "16"

    rebuild = by_id["tail-lora-cache-rebuild-diagnostic"]
    assert rebuild.identity["calibration_candidate"]["selection_eligible"] is False
    assert _flag(rebuild.command, "--draft-cache-policy") == "rebuild"
    audit = by_id["tail-lora-parameter-audit"]
    assert audit.identity["calibration_candidate"]["diagnostic_kind"] == (
        "parameter-audit"
    )
    assert _flag(audit.command, "--parameter-audit-stride") == "7"


def test_candidate_schema_rejects_grids_duplicates_and_unpaired_diagnostics(
    tmp_path: Path,
):
    path = tmp_path / "candidates.json"
    payload = _candidate_spec()
    payload["candidate_grid"] = {"rank": [8, 16]}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="keys must be exactly"):
        calibration.load_candidate_sweep(path)

    payload = _candidate_spec()
    payload["max_new_tokens"] = 2047
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="max_new_tokens must be 2048"):
        calibration.load_candidate_sweep(path)

    payload = _candidate_spec()
    payload["samples"].reverse()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fixed order"):
        calibration.load_candidate_sweep(path)

    payload = _candidate_spec()
    payload["candidates"][1] = dict(payload["candidates"][0])
    payload["candidates"][1]["candidate_id"] = "static-duplicate"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate semantic candidates"):
        calibration.load_candidate_sweep(path)

    payload = _candidate_spec()
    payload["candidates"] = [
        candidate
        for candidate in payload["candidates"]
        if candidate["candidate_id"] != "tail-lora-cache-rebuild-diagnostic"
    ]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="explicit stale and rebuild"):
        calibration.load_candidate_sweep(path)


def test_selection_and_parameter_audit_markers_fail_closed(tmp_path: Path):
    path = tmp_path / "candidates.json"
    payload = _candidate_spec()
    payload["candidates"][1]["draft_cache_policy"] = "rebuild"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="selection candidates require stale"):
        calibration.load_candidate_sweep(path)

    payload = _candidate_spec()
    payload["candidates"][-1]["parameter_audit_stride"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="require a positive audit stride"):
        calibration.load_candidate_sweep(path)


@pytest.mark.parametrize(
    ("flag", "message"),
    (
        ("--no-deterministic", "requires --deterministic"),
        ("--skip-static-parity-preflight", "cannot skip Static parity"),
        ("--audit-cuda-timing", "cannot enable synchronized CUDA timing"),
    ),
)
def test_selection_entry_rejects_global_diagnostic_shortcuts(
    tmp_path: Path, flag: str, message: str
):
    argv, _ = _base_argv(tmp_path)
    args = calibration.build_parser().parse_args([*argv, flag])
    with pytest.raises(ValueError, match=message):
        calibration.build_run_plans(args)


def test_partial_run_validator_remains_non_destructive(
    tmp_path: Path,
):
    argv, _ = _base_argv(tmp_path)
    plan = calibration.build_run_plans(
        calibration.build_parser().parse_args(argv)
    )[0]
    plan.run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match=r"partial run; refusing to overwrite"):
        calibration._completed_run_matches(plan)
    assert plan.run_dir.is_dir()
    assert not calibration.frozen._quarantine_root(plan).exists()


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("attestation", "summary run_attestation mismatch"),
        ("mask", r"summary parameters\.mask_token_id mismatch"),
        ("parity", "summary static parity stale did not pass"),
    ),
)
def test_real_schema_v3_resume_repairs_completion_and_rejects_tamper(
    tmp_path: Path, tamper: str, message: str
):
    argv, candidate_spec = _base_argv(tmp_path)
    plan = calibration.build_run_plans(
        calibration.build_parser().parse_args(argv)
    )[0]
    _write_completed_run(plan)
    assert calibration.frozen.completed_run_matches(plan) is True
    assert calibration.execute_plan(plan) == "resumed_complete"

    plan.completion_path.unlink()
    assert calibration.execute_plan(plan) == "resumed_complete"
    assert plan.completion_path.is_file()

    plan.completion_path.unlink()
    summary_path = plan.artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if tamper == "attestation":
        summary["run_attestation"]["command_sha256"] = "0" * 64
    elif tamper == "mask":
        summary["parameters"]["mask_token_id"] += 1
    else:
        summary["reference"]["official_static_parity"]["policies"]["stale"][
            "output_ids_match"
        ] = False
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match=message):
        calibration._completed_run_matches(plan)
    assert calibration.frozen._prepare_existing_run(plan) is False
    assert not plan.completion_path.exists()
    archived = (
        calibration.frozen._quarantine_root(plan)
        / "attempt-0001"
        / "run"
        / "artifact"
        / "summary.json"
    )
    assert archived.is_file()


def test_spec_mutation_check_precedes_new_run(tmp_path: Path):
    argv, candidate_spec = _base_argv(tmp_path)
    other = calibration.build_run_plans(
        calibration.build_parser().parse_args(argv)
    )[1]
    candidate_spec.write_text(candidate_spec.read_text() + "\n")
    with pytest.raises(ValueError, match="candidate specification changed"):
        calibration.execute_plan(other)
    assert not other.run_dir.exists()


def test_identity_creation_failure_preserves_retryable_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    argv, _ = _base_argv(tmp_path)
    plan = calibration.build_run_plans(
        calibration.build_parser().parse_args(argv)
    )[0]
    calibration.frozen._ensure_artifact_identity_lock(plan)

    def fail_exclusive_write(*_args, **_kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(
        calibration.frozen, "_write_json_exclusive", fail_exclusive_write
    )
    with pytest.raises(
        calibration.frozen.RetryableRunError,
        match=r"simulated disk full.*failed evidence preserved.*attempt-0001/run",
    ):
        calibration.execute_plan(plan)
    assert not plan.run_dir.exists()
    archived = (
        calibration.frozen._quarantine_root(plan)
        / "attempt-0001"
        / "run"
    )
    assert archived.is_dir()
    assert not (archived / "run_identity.json").exists()


def test_failed_child_is_archived_and_same_logical_run_can_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    argv, _ = _base_argv(tmp_path)
    plan = calibration.build_run_plans(
        calibration.build_parser().parse_args(argv)
    )[0]
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
    monkeypatch.setattr(calibration.frozen.subprocess, "run", fake_child)
    monkeypatch.setattr(
        calibration.frozen, "_validate_summary", lambda _plan: hashes
    )
    with pytest.raises(
        calibration.frozen.RetryableRunError,
        match=r"exit code 7.*attempt-0001/run.*rerun the same logical plan",
    ):
        calibration.execute_plan(plan)
    archived = (
        calibration.frozen._quarantine_root(plan)
        / "attempt-0001"
        / "run"
    )
    assert archived.is_dir()
    assert (archived / "run_identity.json").is_file()
    assert (archived / "run.log").is_file()
    archived_identity = (archived / "run_identity.json").read_bytes()

    assert calibration.execute_plan(plan) == "completed"
    completion = plan.completion_path.read_bytes()
    assert calibration.execute_plan(plan) == "resumed_complete"
    assert calls == 2
    assert plan.completion_path.read_bytes() == completion
    assert (archived / "run_identity.json").read_bytes() == archived_identity


def test_dry_run_is_non_mutating_and_lists_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    argv, _ = _base_argv(tmp_path)
    output_root = Path(argv[argv.index("--output-root") + 1])
    assert calibration.main([*argv, "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 3
    assert len(payload["runs"]) == 12
    assert any(
        run["candidate"]["diagnostic_kind"] == "parameter-audit"
        and run["candidate"]["selection_eligible"] is False
        for run in payload["runs"]
    )
    assert not output_root.exists()
