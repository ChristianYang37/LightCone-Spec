from __future__ import annotations

import hashlib
import csv
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


def _watchdog_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "providers"
        / "autodl"
        / "autodl_poweroff_watchdog.py"
    )
    spec = importlib.util.spec_from_file_location("autodl_poweroff_watchdog", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_watchdog_terminal_markers_are_bound_to_queue_run(tmp_path):
    watchdog = _watchdog_module()
    marker = tmp_path / "QUEUE_COMPLETE"

    assert not watchdog.marker_matches(marker, "run-new", 2)
    marker.write_text("run-old\t2\n", encoding="utf-8")
    assert not watchdog.marker_matches(marker, "run-new", 2)
    marker.write_text("run-new\t1\n", encoding="utf-8")
    assert not watchdog.marker_matches(marker, "run-new", 2)
    marker.write_text("run-new\t2\n", encoding="utf-8")
    assert watchdog.marker_matches(marker, "run-new", 2)


def test_watchdog_disarms_when_queue_run_is_superseded(tmp_path):
    watchdog = _watchdog_module()
    current = tmp_path / "CURRENT_RUN"

    assert not watchdog.current_run_matches(tmp_path, "run-a")
    current.write_text("run-a\n", encoding="utf-8")
    assert watchdog.current_run_matches(tmp_path, "run-a")
    current.write_text("run-b\n", encoding="utf-8")
    assert not watchdog.current_run_matches(tmp_path, "run-a")


def test_watchdog_waits_for_attempt_publication_and_detects_supersession(tmp_path):
    watchdog = _watchdog_module()

    assert (
        watchdog.wait_for_attempt_binding(
            tmp_path, "run-a", 1, timeout_seconds=0
        )
        == "timeout"
    )
    (tmp_path / "CURRENT_RUN").write_text("run-a\n", encoding="utf-8")
    (tmp_path / "CURRENT_ATTEMPT").write_text("run-a\t1\n", encoding="utf-8")
    assert (
        watchdog.wait_for_attempt_binding(
            tmp_path, "run-a", 1, timeout_seconds=0
        )
        == "match"
    )
    (tmp_path / "CURRENT_RUN").write_text("run-b\n", encoding="utf-8")
    assert watchdog.attempt_binding_state(tmp_path, "run-a", 1) == "superseded"


def test_watchdog_validates_attempt_bound_pid_and_heartbeat(tmp_path):
    watchdog = _watchdog_module()
    attempt = tmp_path / "sessions" / "run-a" / "attempts" / "3"
    attempt.mkdir(parents=True)
    attempt.joinpath("queue.pid").write_text(
        f"run-a\t3\t{os.getpid()}\n", encoding="utf-8"
    )
    attempt.joinpath("heartbeat").write_text(
        f"run-a\t3\t{os.getpid()}\t1000\n", encoding="utf-8"
    )

    assert watchdog.launcher_health(
        tmp_path,
        "run-a",
        3,
        heartbeat_timeout_seconds=60,
        now_epoch=1020,
    ) == (True, "healthy")
    assert watchdog.launcher_health(
        tmp_path,
        "run-a",
        3,
        heartbeat_timeout_seconds=60,
        now_epoch=1100,
    ) == (False, "heartbeat_stale")
    attempt.joinpath("heartbeat").write_text(
        f"run-a\t2\t{os.getpid()}\t1100\n", encoding="utf-8"
    )
    assert watchdog.launcher_health(
        tmp_path,
        "run-a",
        3,
        heartbeat_timeout_seconds=60,
        now_epoch=1100,
    ) == (False, "heartbeat_missing_or_unbound")


def test_watchdog_marker_grace_resets_when_failed_marker_is_archived():
    watchdog = _watchdog_module()

    seen = watchdog.marker_seen_since(None, True, 10.0)
    assert seen == 10.0
    assert watchdog.marker_seen_since(seen, True, 20.0) == 10.0
    assert watchdog.marker_seen_since(seen, False, 20.0) is None
    assert watchdog.marker_seen_since(None, True, 30.0) == 30.0


def test_watchdog_requires_autodl_success_response(monkeypatch):
    watchdog = _watchdog_module()

    def response(payload):
        monkeypatch.setattr(
            watchdog.urllib.request,
            "urlopen",
            lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
        )
        log = io.StringIO()
        watchdog.power_off("secret-not-logged", "pro-test", log)
        return log.getvalue()

    output = response({"code": "Success", "msg": "", "request_id": "req-1"})
    assert "Success" in output
    assert "secret-not-logged" not in output

    with pytest.raises(RuntimeError, match="code=Success"):
        response({"code": "Busy", "msg": "retry", "request_id": "req-2"})


def test_watchdog_rechecks_attempt_under_publisher_lock_before_power_off(
    tmp_path, monkeypatch
):
    watchdog = _watchdog_module()
    calls = []
    monkeypatch.setattr(
        watchdog,
        "power_off",
        lambda token, instance_uuid, log: calls.append((token, instance_uuid)),
    )
    (tmp_path / "CURRENT_RUN").write_text("run-a\n", encoding="utf-8")
    (tmp_path / "CURRENT_ATTEMPT").write_text("run-a\t1\n", encoding="utf-8")

    assert watchdog.power_off_if_attempt_current(
        "secret",
        "instance",
        io.StringIO(),
        queue_root=tmp_path,
        run_id="run-a",
        attempt_generation=1,
    )
    assert calls == [("secret", "instance")]

    (tmp_path / "CURRENT_ATTEMPT").write_text("run-a\t2\n", encoding="utf-8")
    assert not watchdog.power_off_if_attempt_current(
        "secret",
        "instance",
        io.StringIO(),
        queue_root=tmp_path,
        run_id="run-a",
        attempt_generation=1,
    )
    assert calls == [("secret", "instance")]


def test_watchdog_singleton_lock_is_exclusive(tmp_path):
    watchdog = _watchdog_module()
    first = watchdog._acquire_singleton_lock(tmp_path, "run-a", 1)
    try:
        with pytest.raises(SystemExit, match="already active"):
            watchdog._acquire_singleton_lock(tmp_path, "run-a", 1)
        next_attempt = watchdog._acquire_singleton_lock(tmp_path, "run-a", 2)
        next_attempt.close()
    finally:
        first.close()


def test_remote_queue_shell_is_syntax_valid():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_remote_queue_preserves_stale_state_and_binds_resume_markers(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_flock = fake_bin / "flock"
    fake_flock.write_text(
        """#!/bin/sh
while [ \"$#\" -gt 0 ]; do
  case \"$1\" in
    -n|-o) shift ;;
    -E) shift 2 ;;
    *) shift; break ;;
  esac
done
exec \"$@\"
""",
        encoding="utf-8",
    )
    fake_flock.chmod(0o755)
    fake_ninja = fake_bin / "ninja"
    fake_ninja.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_ninja.chmod(0o755)
    fake_nvcc = fake_bin / "nvcc"
    fake_nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_nvcc.chmod(0o755)
    queue_root = tmp_path / "queue"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LIGHTCONE_QUEUE_ROOT": str(queue_root),
        "LIGHTCONE_STAGE1_ROOT": str(tmp_path / "stage1"),
        "LIGHTCONE_STAGE1_SPEC": str(tmp_path / "missing-spec.json"),
        "LIGHTCONE_P5_ROOT": str(tmp_path / "p5"),
        "LIGHTCONE_P5_ANALYSIS": str(tmp_path / "p5-analysis"),
        # Keep this test hermetic on the actual AutoDL host, where the default
        # absolute runtime paths exist and would otherwise launch real work.
        "LIGHTCONE_PYTHON": "/bin/false",
            "LIGHTCONE_CLI": "/bin/false",
            "LIGHTCONE_CUDA_TOOLKIT_ROOT": str(tmp_path),
            "LIGHTCONE_QUEUE_HEARTBEAT_SECONDS": "1",
    }

    first = subprocess.run(
        ["bash", str(script), "--run-id", "run-a"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 1
    assert (queue_root / "QUEUE_FAILED").read_text().strip() == "run-a\t1"
    assert (queue_root / "QUEUE_FINISHED").read_text().strip() == "run-a\t1"
    assert (queue_root / "CURRENT_ATTEMPT").read_text().strip() == "run-a\t1"
    attempt = queue_root / "sessions" / "run-a" / "attempts" / "1"
    assert attempt.joinpath("queue.pid").is_file()
    assert attempt.joinpath("heartbeat").is_file()
    state_link = queue_root / "queue-state.tsv"
    assert state_link.is_symlink()
    assert Path(os.readlink(state_link)).is_absolute()

    second = subprocess.run(
        ["bash", str(script), "--run-id", "run-b"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 1
    histories = list((queue_root / "history").iterdir())
    assert len(histories) == 1
    history = histories[0]
    assert (history / "CURRENT_RUN").read_text().strip() == "run-a"
    assert (history / "CURRENT_ATTEMPT").read_text().strip() == "run-a\t1"
    assert (history / "QUEUE_FAILED").read_text().strip() == "run-a\t1"
    assert (history / "queue-state.tsv").is_symlink()
    assert (history / "queue-state.tsv").resolve().is_file()

    # Resume validates every terminal marker before moving any of them.
    (queue_root / "QUEUE_FAILED").write_text("foreign-run\t1\n", encoding="utf-8")
    resume = subprocess.run(
        ["bash", str(script), "--resume", "run-b"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert resume.returncode == 2
    assert (queue_root / "QUEUE_FAILED").read_text().strip() == "foreign-run\t1"
    assert (queue_root / "QUEUE_FINISHED").read_text().strip() == "run-b\t1"

    # A valid resume gets a fresh attempt generation, so an old watchdog cannot
    # carry its original deadline into the resumed work.
    (queue_root / "QUEUE_FAILED").write_text("run-b\t1\n", encoding="utf-8")
    resumed = subprocess.run(
        ["bash", str(script), "--resume", "run-b"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 1
    assert (queue_root / "CURRENT_ATTEMPT").read_text().strip() == "run-b\t2"
    assert (queue_root / "QUEUE_FAILED").read_text().strip() == "run-b\t2"
    assert (queue_root / "QUEUE_FINISHED").read_text().strip() == "run-b\t2"


def test_remote_queue_rejects_known_precanonical_root(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    result = subprocess.run(
        ["bash", str(script), "--run-id", "must-not-start"],
        env={
            **os.environ,
            "LIGHTCONE_QUEUE_ROOT": str(tmp_path / "queue"),
            "LIGHTCONE_STAGE1_ROOT": (
                "/srv/lightcone-runtime/reference/tts-dflash/runs/"
                "calibration-schema-v3-optlr-stage1"
            ),
            "LIGHTCONE_RUNTIME_ROOT": "/srv/lightcone-runtime",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "pre-canonical schema-v3 evidence" in result.stderr
    assert not (tmp_path / "queue" / "CURRENT_RUN").exists()


def test_remote_queue_rejects_historical_runner_unbound_stage1_root(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    result = subprocess.run(
        ["bash", str(script), "--run-id", "must-not-start"],
        env={
            **os.environ,
            "LIGHTCONE_QUEUE_ROOT": str(tmp_path / "queue"),
            "LIGHTCONE_STAGE1_ROOT": (
                "/srv/lightcone-runtime/reference/tts-dflash/runs/"
                "calibration-schema-v4-canonical-exact-stage1"
            ),
            "LIGHTCONE_RUNTIME_ROOT": "/srv/lightcone-runtime",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "historical runner-unbound evidence" in result.stderr
    assert "do not migrate old artifacts" in result.stderr
    assert not (tmp_path / "queue" / "CURRENT_RUN").exists()


def test_remote_queue_uses_canonical_roots_and_deduplicated_p5_methods():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    source = script.read_text(encoding="utf-8")
    assert "$RUNTIME_ROOT/experiments/optimizer-screen" in source
    assert "$RUNTIME_ROOT/experiments/rank-screen" in source
    assert "$RUNTIME_ROOT/experiments/diagnostics" in source
    assert "run_p5_tts_mode_screen residual static tts" in source
    assert "run_p5_tts_mode_screen lora static tts" in source
    assert "run_p5_tts_mode_screen full static tts" in source
    assert "run_p5_residual_l0" in source
    assert "run_p5_controller_matched" not in source
    assert "canonical_stream_controller_artifact_not_available" in source
    assert "--methods naive_async lc_gate lc_damp" not in source
    assert '--projection-artifact "$PROJECTION"' in source
    common_args = source.split("common_calibration_args() {", 1)[1].split(
        "spec_uses_output_residual() {", 1
    )[0]
    assert "--projection-artifact" not in common_args
    stage1 = source.split("run_stage1() {", 1)[1].split(
        "publish_stage1_analysis() {", 1
    )[0]
    stage2 = source.split("run_stage2() {", 1)[1].split(
        "run_diagnostics() {", 1
    )[0]
    diagnostics = source.split("run_diagnostics() {", 1)[1].split(
        "dflash_preflight_complete() {", 1
    )[0]
    preflight_validator = source.split("dflash_preflight_complete() {", 1)[
        1
    ].split("run_dflash_long_context() {", 1)[0]
    long_context = source.split("run_dflash_long_context() {", 1)[1].split(
        "run_p5_residual_l0() {", 1
    )[0]
    assert 'append_projection_arg_if_required "$STAGE1_BUNDLE_SPEC"' in stage1
    assert "append_projection_arg_if_required" not in stage2
    assert 'append_projection_arg_if_required "$DIAGNOSTIC_SPEC"' in diagnostics
    assert "frozen._read_json(summary_path)" in preflight_validator
    assert "object_pairs_hook=frozen._reject_duplicate_keys" in preflight_validator
    assert '("status", "complete_reference_run")' in preflight_validator
    assert 'frozen._model_identity(root, revision)' in preflight_validator
    assert 'frozen._tokenizer_identity(target)' in preflight_validator
    assert 'frozen._get(summary, "output.rounds_sha256")' in preflight_validator
    assert 'struct.pack(f"<{input_tokens}q"' in preflight_validator
    assert '"round rendered input identity"' in preflight_validator
    assert long_context.count(
        'if ! dflash_preflight_complete "$preflight_summary" "$preflight_rounds"'
    ) == 2
    for option in (
        "--block-size 16",
        "--seed 0",
        "--lr 0.0001",
        "--proximal-lambda 0",
        "--update-stride 1",
        "--position-weighting uniform",
        "--loss-reduction sum",
        "--draft-cache-policy stale",
        "--enable-thinking",
    ):
        assert option in long_context
    validation_call, archive_branch = long_context.split(
        'if ! dflash_preflight_complete "$preflight_summary" "$preflight_rounds"',
        1,
    )
    assert "preflight_root=" in validation_call
    assert 'mv "$preflight_root" "$preflight_history/preflight-partial"' in (
        archive_branch
    )
    assert 'export PATH="$PY_BIN_DIR:$CUDA_TOOLKIT_ROOT/bin:$PATH"' in source
    assert "ninja is required for FlashInfer JIT" in source
    assert "archived_partial_preflight" in source
    assert '"$SESSION_ROOT/attempt-history"' in source
    assert "record q07_dflash_eagle_l0_l3 blocked" in source
    assert "remote_backend_sync_and_checkpoint_gate\noverall=1" in source


def test_priority_continuous40k_queue_is_resume_safe_and_uses_true_prefix_bins():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_priority_continuous40k_queue.sh"
    )
    source = script.read_text(encoding="utf-8")

    assert "TOTAL_CONTEXT=${LIGHTCONE_TOTAL_CONTEXT:-40960}" in source
    assert "--modes static drafter-lora" in source
    assert "--update-stride 1" in source
    assert "--selected-optimizer-config \"$SELECTION\"" in source
    assert "--lr 0.0001" in source
    assert "--lr 0 \\" not in source
    assert "--bucket-size 4096" in source
    assert 'mv "$PREFLIGHT_ROOT" "$archive"' in source
    assert '"status": "failed_resumable"' in source
    assert '"trajectory": "one_request_continuous_true_prefix"' in source
    assert '"status": "execution_complete_exploratory"' in source
    assert '"claim_scope": "single_held_out_prompt_candidate_screen_no_ci"' in source
    assert "range(0, 40960, 4096)" in source


def test_priority_sglang_continuous_queue_pairs_tts_and_l0_before_analysis():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_priority_sglang_continuous_queue.sh"
    )
    source = script.read_text(encoding="utf-8")

    assert "p5_priority_dflash_continuous40k_calibration_v3.json" in source
    assert "continuous-prefix.lock.json" in source
    assert source.count("--methods static tts naive_async") == 4
    assert "--weight-update-mode lora" in source
    assert "--baseline static" in source
    assert "--baseline tts" in source
    assert "env -u CUBLAS_WORKSPACE_CONFIG -u CUDA_LAUNCH_BLOCKING" in source
    assert 'PY_BIN_DIR=$(dirname -- "$PY")' in source
    assert 'PATH="$PY_BIN_DIR:$CUDA_TOOLKIT/bin:$PATH"' in source
    assert '"status": "failed_resumable"' in source
    assert "prepare-datasets" in source
    assert 'get("prompt_limit")' in source
    assert '--limit "$PROMPT_LIMIT"' in source
    assert "--limit 1" not in source
    assert "validate-artifacts" in source
    assert "EXECUTION_COMPLETE.json" in source
    assert "OBJECTIVE_BLOCKED.json" in source
    assert '"tts_over_static"' in source
    assert '"l0_over_tts"' in source
    assert 'gate.get("scientific_sample_pass") is True' in source
    assert 'gate.get("algorithmic_pass") is True' in source
    assert 'gate.get("window_dominance_pass") is True' in source
    assert '"engineering_pass_evaluated": False' in source


def test_remote_queue_runs_hash_closed_priority_chain_before_old_q00():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    source = script.read_text(encoding="utf-8")

    defaults = {
        "PRIORITY_MANIFEST": (
            "$WORKSPACE/manifests/p5/"
            "p5_priority_dflash_0_40k_v1.json"
        ),
        "PRIORITY_TRACE_MANIFEST": (
            "$WORKSPACE/manifests/p5/"
            "p5_priority_dflash_paired_trace_v1.json"
        ),
        "PRIORITY_L3_EVALUATION_MANIFEST": (
            "$WORKSPACE/manifests/p5/"
            "p5_priority_dflash_l3_evaluation_v1.json"
        ),
        "PRIORITY_CALIBRATION_MANIFEST": (
            "$WORKSPACE/manifests/p5/"
            "p5_priority_dflash_calibration_v1.json"
        ),
        "PRIORITY_SMOKE_MANIFEST": (
            "$WORKSPACE/manifests/p5/"
            "p5_priority_dflash_smoke_v1.json"
        ),
        "PRIORITY_LOCKFILE": (
            "$RUNTIME_ROOT/priority/p5-dflash4b-v1.lock.json"
        ),
        "PRIORITY_MODEL_ROOTS": (
            "$RUNTIME_ROOT/priority/"
            "p5-dflash4b-v1.model-roots.json"
        ),
    }
    for variable, expected in defaults.items():
        match = re.search(
            rf"^{variable}=\$\{{[^:}}]+:-([^}}]+)\}}$", source, re.MULTILINE
        )
        assert match is not None
        assert match.group(1) == expected

    assert "PRIORITY_PAIR=qwen3_4b_dflash16" in source
    assert "PRIORITY_MODE=${LIGHTCONE_PRIORITY_MODE:-lora}" in source
    assert "PRIORITY_LR=${LIGHTCONE_PRIORITY_LR:-0.00003}" in source
    for leaf in ("eval", "trace", "controller", "analysis"):
        assert f"p5-dflash4b-v1/{leaf}" in source

    chain = source.split("run_priority_chain() {", 1)[1].split(
        "\noverall=0", 1
    )[0]
    ordered = [
        "p00_priority_inputs",
        "p00a_priority_calibration",
        "p00b_priority_calibration_winner",
        "p00c_priority_evidence_contract",
        "p00d_priority_smoke",
        "p01_priority_static_tts",
        "p02_priority_l0",
        "p03_priority_paired_trace",
        "p03a_priority_trace_contract",
        "p04_priority_phase1_replay",
        "p04a_priority_phase1_controller_contract",
        "p05_priority_l3_evaluation_readiness",
        "p06_priority_l3_evaluation",
        "p06a_priority_l3_evaluation_contract",
        "p07_priority_final_replay",
        "p07a_priority_final_controller_contract",
        "p08_priority_gate_read",
        "select_priority_methods",
        "p09_priority_eligible_methods",
        "p10_priority_final_analysis",
        "write_priority_terminal",
    ]
    positions = [chain.index(item) for item in ordered]
    assert positions == sorted(positions)
    assert positions[3] < chain.index("priority_terminal_closed") < positions[4]

    execution = source.split("overall=0", 1)[1]
    priority = execution.index("run_priority_chain || priority_rc=$?")
    old_q00 = execution.index("run_task q00_stage1_sweep")
    assert priority < old_q00
    before_q00 = execution[priority:old_q00]
    assert (
        'if [ "$priority_rc" -eq "$PRIORITY_SCIENTIFIC_BLOCKED_RC" ]'
        in before_q00
    )
    assert "record priority scientifically_blocked stop_before_q00" in before_q00
    assert (
        "record q00_stage1_sweep skipped priority_scientifically_blocked"
        in before_q00
    )
    assert "record queue failed priority_scientifically_blocked" in before_q00
    assert "record priority engineering_failed fail_closed_before_q00" in before_q00
    assert "write_marker QUEUE_FAILED" in before_q00
    assert "write_marker QUEUE_FINISHED" in before_q00
    assert "exit 1" in before_q00

    terminal = source.split("priority_terminal_closed() {", 1)[1].split(
        "priority_archive_partial_pair() {", 1
    )[0]
    assert '"scientifically_blocked"' in terminal
    assert "hashlib.sha256(path.read_bytes()).hexdigest()" in terminal
    assert 'record priority skipped terminal_hash_closed' in chain
    assert chain.count('return "$PRIORITY_SCIENTIFIC_BLOCKED_RC"') == 3
    assert "priority_terminal_status" in chain

    epoch = source.split("archive_stale_priority_evidence() {", 1)[1].split(
        "quarantine_invalid_priority_runs() {", 1
    )[0]
    assert "priority artifact contract changed" in epoch
    assert "legacy priority evidence has no artifact contract" in epoch
    assert "artifact_epoch_sha256" in source
    assert "production_files" in epoch
    assert "runtime_trees" in epoch
    assert "runtime_environment" in epoch
    assert "PRIORITY_DATASET_RECEIPT" in epoch
    for label in ("smoke", "eval", "trace", "controller", "analysis"):
        assert f'("{label}",' in epoch
    assert '"phase1-controller",' in epoch
    assert "calibration_queue_script" in epoch
    assert "calibration_candidate_spec" in epoch
    assert "calibration_ready" in epoch
    assert "priority-terminal.json" in epoch
    assert "for destination, original in reversed(moved)" in epoch
    assert "created.append(root)" in epoch
    assert 'archive / "archive-reason.json"' in epoch

    trace_contract = source.split(
        "validate_priority_trace_contract() {", 1
    )[1].split("prepare_priority_inputs() {", 1)[0]
    assert "runtime_implementation_fingerprint" in trace_contract
    assert "experiment manifest hash mismatch" in trace_contract
    assert "controller/trace layout mismatch" in trace_contract
    assert "priority trace allowlist is incomplete or ambiguous" in trace_contract
    assert 'completion.get("status") != "complete_valid"' in trace_contract
    assert "trace replay shard drift" in trace_contract
    assert "trace hash drift" in trace_contract
    assert "priority-trace-quarantine" in trace_contract
    assert "priority-controller-quarantine" in trace_contract


def test_remote_priority_calibration_gate_controls_epoch_and_old_queue():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    source = script.read_text(encoding="utf-8")
    chain = source.split("run_priority_chain() {", 1)[1].split(
        "\noverall=0", 1
    )[0]

    assert chain.index("prepare_priority_inputs") < chain.index(
        "run_priority_calibration"
    )
    assert chain.index("run_priority_calibration") < chain.index(
        "load_priority_calibration_winner"
    )
    assert chain.index("load_priority_calibration_winner") < chain.index(
        "archive_stale_priority_evidence"
    )
    assert 'if [ "$calibration_rc" -eq 3 ]' in chain
    assert 'return "$PRIORITY_SCIENTIFIC_BLOCKED_RC"' in chain
    assert 'elif [ "$calibration_rc" -ne 0 ]' in chain
    assert 'return "$calibration_rc"' in chain
    assert '"mode=$PRIORITY_MODE;lr=$PRIORITY_LR"' in chain

    execution = source.split("overall=0", 1)[1]
    before_q00 = execution.split("run_task q00_stage1_sweep", 1)[0]
    assert "priority_scientifically_blocked" in before_q00
    assert "priority_engineering_failure" in before_q00
    assert before_q00.count("exit 1") >= 2


def test_remote_priority_calibration_ready_is_transitively_hash_verified(
    tmp_path,
):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    source = script.read_text(encoding="utf-8")
    match = re.search(
        r"load_priority_calibration_winner\(\) \{.*?<<'PY'\n"
        r"(?P<body>.*?)\nPY\n",
        source,
        re.DOTALL,
    )
    assert match is not None

    def attested(path, payload, evidence=()):
        rows = [
            {
                "path": str(item.resolve()),
                "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
            }
            for item in evidence
        ]
        path.write_text(
            json.dumps({**payload, "evidence": rows}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(str(path) + ".sha256").write_text(
            hashlib.sha256(path.read_bytes()).hexdigest() + "\n",
            encoding="utf-8",
        )

    winner = {
        "candidate_id": "tail-lora-r16-lr1e-4",
        "weight_update_mode": "lora",
        "learning_rate": 1e-4,
    }
    selection = tmp_path / "calibration-selection.json"
    attested(selection, {"schema_version": 1, "winner": winner})
    gate = tmp_path / "heldout-gate.json"
    attested(
        gate,
        {
            "schema_version": 1,
            "winner": winner,
            "verdict": {"downstream_ready": True},
        },
        [selection],
    )
    ready = tmp_path / "DOWNSTREAM_READY.json"
    attested(
        ready,
        {
            "schema_version": 1,
            "status": "ready",
            "winner": winner,
            "heldout_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
        },
        [gate, selection],
    )
    valid = subprocess.run(
        [sys.executable, "-", str(ready)],
        input=match.group("body"),
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0, valid.stderr
    assert valid.stdout.strip() == "lora\t0.0001"

    selection.write_text("tampered\n", encoding="utf-8")
    tampered = subprocess.run(
        [sys.executable, "-", str(ready)],
        input=match.group("body"),
        capture_output=True,
        text=True,
    )
    assert tampered.returncode != 0
    assert "evidence drift" in tampered.stderr

    missing = subprocess.run(
        [sys.executable, "-", str(tmp_path / "missing.json")],
        input=match.group("body"),
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0
    assert "incomplete attested calibration receipt" in missing.stderr


def test_priority_artifact_epoch_inventory_is_explicit_and_secret_free():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    source = script.read_text(encoding="utf-8")
    block = source.split("PRIORITY_PRODUCTION_FILES=(", 1)[1].split("\n)", 1)[0]
    selected = [line.strip() for line in block.splitlines() if line.strip()]
    assert selected == [
        "src/lightcone_spec/adapters/adapter_params.py",
        "src/lightcone_spec/adapters/losses.py",
        "src/lightcone_spec/methods/base.py",
        "src/lightcone_spec/methods/registry.py",
        "src/lightcone_spec/replay/real.py",
        "src/lightcone_spec/config/schema.py",
        "src/lightcone_spec/orchestration/runtime_config.py",
        "src/lightcone_spec/orchestration/catalog.py",
        "src/lightcone_spec/sglang_bridge/client.py",
        "src/lightcone_spec/sglang_bridge/hooks.py",
        "src/lightcone_spec/sglang_bridge/runtime.py",
        "src/lightcone_spec/sglang_bridge/static_observer.py",
        "src/lightcone_spec/sglang_bridge/telemetry.py",
        "src/lightcone_spec/artifacts/schemas.py",
        "src/lightcone_spec/statistics/tables.py",
        "src/lightcone_spec/cli/main.py",
        "src/lightcone_spec/runtime/engine.py",
        "sglang/python/sglang/srt/speculative/dflash_info_v2.py",
        "sglang/python/sglang/srt/speculative/dflash_worker_v2.py",
        "sglang/python/sglang/srt/speculative/dspark_components/dspark_adaptation.py",
        "sglang/python/sglang/srt/speculative/eagle_worker_v2.py",
        "sglang/python/sglang/srt/speculative/tail_adaptation.py",
    ]
    epoch = source.split("archive_stale_priority_evidence() {", 1)[1].split(
        "quarantine_invalid_priority_runs() {", 1
    )[0]
    assert "HF_TOKEN" not in epoch
    assert "AUTODL" not in epoch
    assert "PASSWORD" not in epoch.upper()


def test_priority_deployment_contract_sidecars_hash_exact_source_bytes():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "manifests/p5/p5_priority_dflash_0_40k_v1.json",
        root
        / "manifests/p5/p5_priority_dflash_calibration_v1.json",
        root
        / "manifests/p5/"
        "p5_priority_dflash_continuous40k_calibration_v1.json",
        root
        / "manifests/p5/p5_priority_dflash_paired_trace_v1.json",
        root
        / "manifests/p5/p5_priority_dflash_l3_evaluation_v1.json",
        root / "manifests/p5/p5_priority_dflash_smoke_v1.json",
        root / "scripts/experiments/priority_calibration_candidates_v1.json",
    ]
    for path in paths:
        assert Path(str(path) + ".sha256").read_text(encoding="utf-8").strip() == (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_priority_artifact_epoch_archives_legacy_reuses_current_and_rolls_epoch(
    tmp_path,
):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    source = script.read_text(encoding="utf-8")
    match = re.search(
        r"archive_stale_priority_evidence\(\) \{.*?<<'PY'\n(?P<body>.*?)\nPY\n\}",
        source,
        re.DOTALL,
    )
    assert match is not None
    production_block = source.split("PRIORITY_PRODUCTION_FILES=(", 1)[1].split(
        "\n)", 1
    )[0]
    production = [
        line.strip() for line in production_block.splitlines() if line.strip()
    ]

    workspace = tmp_path / "workspace"
    for relative in production:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    queue_script = tmp_path / "queue.sh"
    queue_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    def paired(name):
        path = tmp_path / name
        path.write_text(f'{{"name":"{name}"}}\n', encoding="utf-8")
        Path(str(path) + ".sha256").write_text(
            hashlib.sha256(path.read_bytes()).hexdigest() + "\n",
            encoding="utf-8",
        )
        return path

    manifests = [paired(f"manifest-{index}.json") for index in range(5)]
    calibration_spec = paired("calibration-spec.json")
    calibration_ready = paired("DOWNSTREAM_READY.json")
    calibration_script = tmp_path / "calibration-queue.py"
    calibration_script.write_text("# calibration queue\n", encoding="utf-8")
    lock = paired("lock.json")
    model_roots = paired("model-roots.json")
    receipt = paired("dataset-receipt.json")
    contract = tmp_path / "artifact-contract.json"
    roots = [
        tmp_path / name
        for name in (
            "smoke",
            "eval",
            "trace",
            "phase1-controller",
            "controller",
            "analysis",
        )
    ]
    for root in roots:
        root.mkdir()
        (root / "legacy.txt").write_text(root.name, encoding="utf-8")
    terminal = tmp_path / "priority-terminal.json"
    terminal.write_text("legacy terminal\n", encoding="utf-8")
    archive_parent = tmp_path / "history"

    command = [
        sys.executable,
        "-",
        str(contract),
        *(str(root) for root in roots),
        str(terminal),
        str(archive_parent),
        "9",
        "qwen3_4b_dflash16",
        "lora",
        "0.00003",
        str(lock),
        str(model_roots),
        str(receipt),
        str(workspace),
        str(queue_script),
        str(calibration_script),
        *(str(path) for path in manifests),
        str(calibration_spec),
        str(calibration_ready),
        *production,
    ]

    first = subprocess.run(
        command,
        input=match.group("body"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "archived stale priority artifact epoch" in first.stdout
    assert contract.is_file() and Path(str(contract) + ".sha256").is_file()
    assert not terminal.exists()
    archives = sorted(archive_parent.iterdir())
    assert len(archives) == 1
    assert (archives[0] / "priority-terminal.json").is_file()
    for root in roots:
        assert root.is_dir() and not any(root.iterdir())
        assert (archives[0] / root.name / "legacy.txt").is_file()

    for root in roots:
        (root / "breakpoint.txt").write_text(root.name, encoding="utf-8")
    current_sha = Path(str(contract) + ".sha256").read_text().strip()
    second = subprocess.run(
        command,
        input=match.group("body"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "contract is current" in second.stdout
    assert Path(str(contract) + ".sha256").read_text().strip() == current_sha
    assert all((root / "breakpoint.txt").is_file() for root in roots)
    assert len(list(archive_parent.iterdir())) == 1

    changed = workspace / production[0]
    changed.write_text(changed.read_text(encoding="utf-8") + "# changed\n")
    third = subprocess.run(
        command,
        input=match.group("body"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "artifact contract changed" in third.stdout
    archives = sorted(archive_parent.iterdir())
    assert len(archives) == 2
    newest = next(
        path for path in archives if (path / "smoke" / "breakpoint.txt").is_file()
    )
    assert (newest / "artifact-contract.json").is_file()
    assert (newest / "artifact-contract.json.sha256").is_file()
    assert all(root.is_dir() and not any(root.iterdir()) for root in roots)


def test_priority_run_quarantine_preserves_valid_breakpoints(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    source = script.read_text(encoding="utf-8")
    match = re.search(
        r"quarantine_invalid_priority_runs\(\) \{.*?<<'PY'\n"
        r"(?P<body>.*?)\nPY\n\}",
        source,
        re.DOTALL,
    )
    assert match is not None
    root = tmp_path / "eval"
    valid = root / "valid-run"
    invalid = root / "partial-run"
    valid.mkdir(parents=True)
    invalid.mkdir()
    invalid.joinpath("manifest.json").write_text("partial\n", encoding="utf-8")
    files = {
        "manifest.json": '{"unit_id":"u0"}\n',
        "environment.json": "{}\n",
        "lock-reference.json": "{}\n",
        "stdout.log": "ok\n",
        "stderr.log": "",
        "exit.json": '{"status":"complete_valid","exit_code":0}\n',
        "rounds.parquet": "rounds",
        "updates.parquet": "updates",
        "decisions.parquet": "decisions",
        "system_samples.parquet": "systems",
        "request_summary.parquet": "requests",
    }
    for name, body in files.items():
        valid.joinpath(name).write_text(body, encoding="utf-8")
    valid.joinpath("manifest.sha256").write_text(
        hashlib.sha256(valid.joinpath("manifest.json").read_bytes()).hexdigest()
        + "\n",
        encoding="utf-8",
    )
    ledger = {}
    for path in valid.iterdir():
        ledger[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    valid.joinpath("hashes.json").write_text(
        json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8"
    )
    history = tmp_path / "history"
    result = subprocess.run(
        [sys.executable, "-", str(root), str(history), "9", "eval"],
        input=match.group("body"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "quarantined invalid priority eval runs" in result.stdout
    assert valid.is_dir()
    assert not invalid.exists()
    archives = list(history.iterdir())
    assert len(archives) == 1
    assert (archives[0] / "partial-run" / "manifest.json").is_file()
    assert (archives[0] / "archive-reason.json").is_file()


def test_remote_priority_trace_and_controller_gates_fail_closed_independently(
    tmp_path,
):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    source = script.read_text(encoding="utf-8")

    prepare = source.split("prepare_priority_inputs() {", 1)[1].split(
        "run_priority_smoke() {", 1
    )[0]
    assert '--reuse-inputs-from "$P5_LOCKFILE"' in prepare
    assert '--pairs "$PRIORITY_PAIR"' in prepare
    assert "prepare-models" in prepare
    assert "prepare-datasets" in prepare
    assert "--limit 96" in prepare
    assert 'local receipt="$PRIORITY_DATASET_RECEIPT"' in prepare
    assert 'PRIORITY_ANALYSIS_ROOT/dataset-preflight.json' not in prepare

    smoke = source.split("run_priority_smoke() {", 1)[1].split(
        "run_priority_static_tts() {", 1
    )[0]
    assert '--manifest "$PRIORITY_SMOKE_MANIFEST"' in smoke
    assert '--artifact-root "$PRIORITY_SMOKE_ROOT"' in smoke
    assert smoke.count("--methods static tts naive_async") == 2
    assert smoke.count('--learning-rate "$PRIORITY_LR"') == 2
    assert smoke.count('--weight-update-mode "$PRIORITY_MODE"') == 2

    trace = source.split("run_priority_paired_trace() {", 1)[1].split(
        "fit_priority_phase1_controller() {", 1
    )[0]
    assert trace.count("--methods tts naive_async") == 2
    assert trace.count('--learning-rate "$PRIORITY_LR"') == 2
    assert trace.count('--weight-update-mode "$PRIORITY_MODE"') == 2
    assert "--lifecycles" not in trace
    assert "--logical-delay" not in trace
    assert "Do not filter logical delay" in trace
    assert '--artifact-root "$PRIORITY_PHASE1_TRACE_ROOT"' in trace

    phase1_replay = source.split(
        "fit_priority_phase1_controller() {", 1
    )[1].split("read_priority_l3_evaluation_ready() {", 1)[0]
    assert '--trace-root "$PRIORITY_PHASE1_TRACE_ROOT"' in phase1_replay
    assert '--output-dir "$PRIORITY_PHASE1_CONTROLLER_ROOT"' in phase1_replay

    l3_evaluation = source.split("run_priority_l3_evaluation() {", 1)[1].split(
        "fit_priority_controller() {", 1
    )[0]
    assert '--manifest "$PRIORITY_L3_EVALUATION_MANIFEST"' in l3_evaluation
    assert '--artifact-root "$PRIORITY_L3_EVALUATION_TRACE_ROOT"' in l3_evaluation
    assert '--controller-root "$PRIORITY_PHASE1_CONTROLLER_ROOT"' in l3_evaluation
    assert l3_evaluation.count("--methods lc_transport") == 2

    replay = source.split("fit_priority_controller() {", 1)[1].split(
        "read_priority_gates() {", 1
    )[0]
    assert '--trace-root "$PRIORITY_TRACE_ROOT"' in replay
    assert '--pair "$PRIORITY_PAIR"' in replay

    match = re.search(
        r'read_priority_gates\(\) \{.*?<<\'PY\'\n(?P<body>.*?)\nPY\n',
        source,
        re.DOTALL,
    )
    assert match is not None
    report = tmp_path / "replay_report.json"
    report.write_text(
        json.dumps(
            {
                "oracle_replay_gate": {
                    "l1_eligible": True,
                    "l2_eligible": False,
                },
                "tts_paired_gate": {
                    "l1_eligible": False,
                    "l2_eligible": True,
                },
                "learned_policy_gate": {
                    "l1_eligible": True,
                    "l2_eligible": False,
                },
                "trace_exactness": {"verified": True},
                "l3_gate": {
                    "enabled": True,
                    "exactness": {"verified": True},
                    "heldout_transported_utility_gate": {"eligible": False},
                },
            }
        ),
        encoding="utf-8",
    )
    parsed = subprocess.run(
        [sys.executable, "-", str(report)],
        input=match.group("body"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert parsed.stdout.strip().split("\t") == [
        "1",
        "0",
        "1",
        "0",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1",
        "0",
    ]

    selection = source.split("select_priority_methods() {", 1)[1].split(
        "run_priority_eligible_methods() {", 1
    )[0]
    assert (
        '"$PRIORITY_ORACLE_L1" -eq 1 ] && [ "$PRIORITY_TTS_L1" -eq 1'
        in selection
    )
    assert (
        '"$PRIORITY_ORACLE_L2" -eq 1 ] && [ "$PRIORITY_TTS_L2" -eq 1'
        in selection
    )
    assert '"$PRIORITY_LEARNED_L1" -eq 1' in selection
    assert '"$PRIORITY_LEARNED_L2" -eq 1' in selection
    assert selection.count('"$PRIORITY_TRACE_EXACT" -eq 1') == 3
    assert '"$PRIORITY_L3_EXACT" -eq 1' in selection
    assert '"$PRIORITY_L3_PAIRING" -eq 1' in selection
    assert (
        'if [ "$PRIORITY_L3" -eq 1 ] && [ "$PRIORITY_L3_UTILITY" -eq 1 ] &&'
        in selection
    )
    assert "evidence_insufficient" in selection
    assert selection.count("PRIORITY_BLOCKED_METHODS+=(") == 3
    assert "overall=" not in selection
    assert "return 1" not in selection

    eligible = source.split("run_priority_eligible_methods() {", 1)[1].split(
        "finalize_priority_analysis() {", 1
    )[0]
    assert '--controller-root "$PRIORITY_CONTROLLER_ROOT"' in eligible
    assert '--methods "${eligible[@]}"' in eligible


def test_remote_priority_terminal_writer_and_hash_validator_round_trip(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    source = script.read_text(encoding="utf-8")
    writer_section = source.split("write_priority_terminal() {", 1)[1].split(
        "run_priority_chain() {", 1
    )[0]
    writer_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", writer_section, re.DOTALL)
    assert len(writer_blocks) == 3
    writer = writer_blocks[2]
    validator = re.search(
        r"priority_terminal_closed\(\) \{.*?<<'PY'\n(?P<body>.*?)\nPY\n\}",
        source,
        re.DOTALL,
    )
    assert validator is not None

    report = tmp_path / "replay_report.json"
    report.write_text(
        json.dumps(
            {
                "oracle_replay_gate": {"l1_eligible": False},
                "tts_paired_gate": {"l1_eligible": False},
                "learned_policy_gate": {"l1_eligible": False},
                "trace_exactness": {"verified": True},
                "l3_gate": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "smoke-manifest.json"
    evidence.write_text("immutable\n", encoding="utf-8")
    static_claims = tmp_path / "static-claims.json"
    final_claims = tmp_path / "final-claims.json"
    static_acceptance = tmp_path / "static-acceptance.csv"
    final_acceptance = tmp_path / "final-acceptance.csv"
    manifest = tmp_path / "priority-manifest.json"
    contexts = [512, 4096, 16384, 40000]
    concurrencies = [1, 4]
    adaptation_methods = [
        "naive_async", "lc_gate", "lc_damp", "lc_transport"
    ]
    all_methods = ["static", "tts", *adaptation_methods]
    manifest.write_text(
        json.dumps(
            {
                "engine_params": {"p5_context_lengths": contexts},
                "units": [
                    {"method": method, "concurrency": concurrency}
                    for method in all_methods
                    for concurrency in concurrencies
                ],
            }
        ),
        encoding="utf-8",
    )

    def claim_rows(methods, baseline, passing):
        return [
            {
                "method": method,
                "baseline_method": baseline,
                "offered_concurrency": concurrency,
                "algorithmic_pass": passing,
                "engineering_pass": False,
                "exactness_pass": True,
                "lcag_ci_low": 0.2 if passing else -0.2,
                "mean_delta_acceptance_elasticity": -0.1 if passing else 0.1,
            }
            for method in methods
            for concurrency in concurrencies
        ]

    def acceptance_rows(methods, baseline, passing):
        return [
            {
                "method": method,
                "baseline_method": baseline,
                "offered_concurrency": concurrency,
                "context_length": context,
                "acceptance_gain_vs_baseline": 0.1 if passing else -0.1,
                "gain_prompt_clusters": 8,
                "version_mismatch_count": 0,
                "exactness_violations": 0,
            }
            for method in methods
            for concurrency in concurrencies
            for context in contexts
        ]

    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer_csv.writeheader()
            writer_csv.writerows(rows)

    def write_analysis(passing):
        static_claims.write_text(
            json.dumps(claim_rows(["tts"], "static", passing)),
            encoding="utf-8",
        )
        final_claims.write_text(
            json.dumps(claim_rows(adaptation_methods, "tts", passing)),
            encoding="utf-8",
        )
        write_csv(
            static_acceptance,
            acceptance_rows(["tts"], "static", passing),
        )
        write_csv(
            final_acceptance,
            acceptance_rows(adaptation_methods, "tts", passing),
        )

    write_analysis(False)
    epoch = tmp_path / "artifact-contract.json"
    epoch.write_text('{"schema_version": 1}\n', encoding="utf-8")
    epoch.with_suffix(".json.sha256").write_text(
        hashlib.sha256(epoch.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    roots = [tmp_path / name for name in ("smoke", "eval", "trace", "analysis")]
    for root in roots:
        root.mkdir()
    terminal = tmp_path / "priority-terminal.json"
    def invoke_writer():
        subprocess.run(
            [
                sys.executable,
                "-",
                str(terminal),
                "run-a",
                "complete",
                "qwen3_4b_dflash16",
                "lora",
                "0.00003",
                ",".join(all_methods),
                "",
                str(report),
                str(epoch),
                *(str(root) for root in roots),
                str(static_claims),
                str(final_claims),
                str(manifest),
                str(static_acceptance),
                str(final_acceptance),
                str(report),
                str(evidence),
            ],
            input=writer,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(terminal.read_text(encoding="utf-8"))

    payload = invoke_writer()
    assert payload["actual_methods"] == all_methods
    assert payload["status"] == "scientifically_blocked"
    assert "aggregate_gate_failed:tts:c1" in payload[
        "scientific_block_reasons"
    ]
    assert "bucket_gate_failed:lc_transport:c4:L40000" in payload[
        "scientific_block_reasons"
    ]
    assert payload["gates"]["trace_exactness"] == {"verified": True}
    assert payload["contract"]["artifact_epoch_sha256"] == hashlib.sha256(
        epoch.read_bytes()
    ).hexdigest()
    assert {row["path"] for row in payload["evidence"]} == {
        str(report.resolve()),
        str(evidence.resolve()),
        str(static_claims.resolve()),
        str(final_claims.resolve()),
        str(manifest.resolve()),
        str(static_acceptance.resolve()),
        str(final_acceptance.resolve()),
    }

    # Every frozen concurrency/context cell and every aggregate gate must pass.
    write_analysis(True)
    payload = invoke_writer()
    assert payload["status"] == "complete"
    assert payload["scientific_block_reasons"] == []
    assert payload["coverage_contract"]["contexts"] == contexts

    # Missing one 40K cell cannot be hidden by a positive aggregate headline.
    rows = acceptance_rows(adaptation_methods, "tts", True)
    rows = [
        row for row in rows
        if not (
            row["method"] == "lc_transport"
            and row["offered_concurrency"] == 4
            and row["context_length"] == 40000
        )
    ]
    write_csv(final_acceptance, rows)
    payload = invoke_writer()
    assert payload["status"] == "scientifically_blocked"
    assert "bucket_coverage:lc_transport:c4:L40000:found=0" in payload[
        "scientific_block_reasons"
    ]

    # Exactness and adapter-version mismatches fail the affected bucket even
    # when its acceptance point estimate and aggregate headline are positive.
    rows = acceptance_rows(adaptation_methods, "tts", True)
    bad = next(
        row for row in rows
        if row["method"] == "lc_damp"
        and row["offered_concurrency"] == 1
        and row["context_length"] == 512
    )
    bad["version_mismatch_count"] = 1
    bad["exactness_violations"] = 1
    write_csv(final_acceptance, rows)
    payload = invoke_writer()
    assert payload["status"] == "scientifically_blocked"
    assert "bucket_gate_failed:lc_damp:c1:L512" in payload[
        "scientific_block_reasons"
    ]

    # Restore a fully passing closure before testing its immutable hash fence.
    write_analysis(True)
    payload = invoke_writer()
    assert payload["status"] == "complete"

    command = [
        sys.executable,
        "-",
        str(terminal),
        "run-a",
        "qwen3_4b_dflash16",
        "lora",
        "0.00003",
        str(epoch),
    ]
    valid = subprocess.run(
        command,
        input=validator.group("body"),
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0
    evidence.write_text("drift\n", encoding="utf-8")
    invalid = subprocess.run(
        command,
        input=validator.group("body"),
        capture_output=True,
        text=True,
    )
    assert invalid.returncode != 0
    assert "hash drift" in invalid.stderr


def test_remote_queue_uses_engine_reuse_p5_with_isolated_overlay_identities():
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "experiments" / "run_remote_experiment_queue.sh"
    source = script.read_text(encoding="utf-8")
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "manifests"
        / "p5"
        / "p5_long_context_acceptance_engine_reuse.json"
    )
    manifest = ExperimentManifest.load(manifest_path)

    assert re.search(
        r"^P5_MANIFEST=\$\{LIGHTCONE_P5_MANIFEST:-.*/"
        r"p5_long_context_acceptance_engine_reuse\.json\}$",
        source,
        re.MULTILINE,
    )
    assert len(manifest.units) == 75
    assert {
        method: sum(unit.method == method for unit in manifest.units)
        for method in ("static", "tts", "naive_async", "lc_gate", "lc_damp")
    } == {
        "static": 15,
        "tts": 15,
        "naive_async": 15,
        "lc_gate": 15,
        "lc_damp": 15,
    }

    effective = {
        mode: manifest.with_methods(("static", "tts")).with_weight_update_mode(
            mode
        )
        for mode in ("residual", "lora", "full")
    }
    static_ids = {
        mode: {
            unit.unit_id
            for unit in selected.units
            if unit.method == "static"
        }
        for mode, selected in effective.items()
    }
    adapted_ids = {
        mode: {
            unit.unit_id
            for unit in selected.units
            if unit.method == "tts"
        }
        for mode, selected in effective.items()
    }
    assert len(set(map(frozenset, static_ids.values()))) == 1
    assert all(len(ids) == 15 for ids in adapted_ids.values())
    assert not (adapted_ids["residual"] & adapted_ids["lora"])
    assert not (adapted_ids["residual"] & adapted_ids["full"])
    assert not (adapted_ids["lora"] & adapted_ids["full"])
    assert len({selected.content_sha256() for selected in effective.values()}) == 3

    guard = source.index("if require_p5_engine_reuse_transition_safe; then")
    first_q05 = source.index("run_task q05a_dspark_tts_residual", guard)
    assert guard < first_q05
    guard_body = source.split(
        "require_p5_engine_reuse_transition_safe() {", 1
    )[1].split("run_p5_residual_l0() {", 1)[0]
    assert 'source.name != "p5_long_context_acceptance_engine_reuse"' in guard_body
    assert "claimed != unit.unit_id or claimed not in allowed" in guard_body
    assert "experiment_digest not in allowed[claimed]" in guard_body
    assert "unsafe_p5_manifest_transition" in source


def test_remote_queue_p5_transition_guard_rejects_foreign_or_cross_overlay_runs(
    tmp_path,
):
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_remote_experiment_queue.sh"
    )
    source = script.read_text(encoding="utf-8")
    match = re.search(
        r"require_p5_engine_reuse_transition_safe\(\) \{.*?<<'PY'\n"
        r"(?P<body>.*?)\nPY\n\}",
        source,
        re.DOTALL,
    )
    assert match is not None
    guard = match.group("body")
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "manifests"
        / "p5"
        / "p5_long_context_acceptance_engine_reuse.json"
    )
    manifest = ExperimentManifest.load(manifest_path)
    artifact_root = tmp_path / "p5"
    artifact_root.mkdir()

    def run_guard():
        return subprocess.run(
            [sys.executable, "-", str(manifest_path), str(artifact_root)],
            input=guard,
            capture_output=True,
            text=True,
        )

    assert run_guard().returncode == 0

    effective = manifest.with_methods(("static", "tts")).with_weight_update_mode(
        "residual"
    )
    unit = next(unit for unit in effective.units if unit.method == "tts")
    run_dir = artifact_root / "p5-tts-test"
    run_dir.mkdir()
    payload = {
        **unit.to_manifest_dict(),
        "experiment_manifest_sha256": effective.content_sha256(),
    }
    run_dir.joinpath("manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert run_guard().returncode == 0

    payload["experiment_manifest_sha256"] = "0" * 64
    run_dir.joinpath("manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    rejected = run_guard()
    assert rejected.returncode != 0
    assert "cross-overlay manifest identity" in rejected.stderr

    foreign = manifest.with_methods(("lc_gate",)).with_weight_update_mode(
        "residual"
    ).units[0]
    payload = {
        **foreign.to_manifest_dict(),
        "experiment_manifest_sha256": "0" * 64,
    }
    run_dir.joinpath("manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    rejected = run_guard()
    assert rejected.returncode != 0
    assert "legacy or foreign unit" in rejected.stderr


def test_remote_queue_runs_controller_free_load_saturation_before_p5():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "experiments" / "run_remote_experiment_queue.sh"
    source = script.read_text(encoding="utf-8")
    block = source.split("run_load_saturation() {", 1)[1].split(
        "require_p5_engine_reuse_transition_safe() {", 1
    )[0]

    assert "load_tune_gpu_qwen3_4b.json" in source
    assert "LOAD_TUNE_LOCKFILE=${LIGHTCONE_LOAD_TUNE_LOCKFILE:-" in source
    assert "LOAD_TUNE_ROOT=${LIGHTCONE_LOAD_TUNE_ROOT:-" in source
    assert "LOAD_TUNE_ANALYSIS=${LIGHTCONE_LOAD_TUNE_ANALYSIS:-" in source
    assert block.count('--lockfile "$LOAD_TUNE_LOCKFILE"') == 2
    assert '--lockfile "$P5_LOCKFILE"' not in block
    assert block.count("--methods static tts naive_async") == 3
    assert "--methods static tts naive_async lc_gate" not in block
    assert block.count("--weight-update-mode residual") == 3
    assert "run_sglang_headline run-manifest" in block
    assert "env -u CUBLAS_WORKSPACE_CONFIG -u CUDA_LAUNCH_BLOCKING" in source
    assert "LIGHTCONE_CUDA_TOOLKIT_ROOT:-$RUNTIME_ROOT/cuda-12.9" in source
    assert 'export CUDA_HOME="$CUDA_TOOLKIT_ROOT"' in source
    assert 'CUDA_HOME="$CUDA_HOME" PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH"' in source
    saturation = source.index(
        "run_task q04b_sglang_load_saturation run_load_saturation"
    )
    p5_guard = source.index("if require_p5_engine_reuse_transition_safe; then")
    assert saturation < p5_guard

    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "manifests"
            / "load_tune"
            / "load_tune_gpu_qwen3_4b.json"
        ).read_text(encoding="utf-8")
    )
    selected = [
        unit
        for unit in manifest["units"]
        if unit["method"] in {"static", "tts", "naive_async"}
    ]
    assert len(selected) == 21
    assert {unit["concurrency"] for unit in selected} == {
        1,
        2,
        4,
        8,
        16,
        32,
        48,
    }

    p5_l0 = source.split("run_p5_residual_l0() {", 1)[1].split(
        "run_p5_tts_mode_screen() {", 1
    )[0]
    p5_tts = source.split("run_p5_tts_mode_screen() {", 1)[1].split(
        "priority_terminal_closed() {", 1
    )[0]
    assert p5_l0.count("run_sglang_headline run-manifest") == 1
    assert p5_tts.count("run_sglang_headline run-manifest") == 1


def test_p5_manifest_excludes_only_the_infeasible_32k_concurrency_16_cells():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "manifests"
        / "p5"
        / "p5_long_context_acceptance.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = payload["units"]

    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == Path(
        f"{manifest_path}.sha256"
    ).read_text(encoding="utf-8").strip()

    assert len(units) == 345
    assert {
        method: sum(unit["method"] == method for unit in units)
        for method in ("static", "tts", "naive_async", "lc_gate", "lc_damp")
    } == {
        "static": 69,
        "tts": 69,
        "naive_async": 69,
        "lc_gate": 69,
        "lc_damp": 69,
    }
    assert not any(unit["allow_resource_skip"] for unit in units)
    assert not any(
        unit["lifecycle"] == "stream"
        and unit["concurrency"] == 16
        and unit["prompt_subset"] == "p5_ctx_32768"
        for unit in units
    )

    methods = {unit["method"] for unit in units}
    datasets = {unit["dataset"] for unit in units}
    for dataset in datasets:
        for method in methods:
            assert {
                unit["concurrency"]
                for unit in units
                if unit["dataset"] == dataset
                and unit["method"] == method
                and unit["lifecycle"] == "stream"
                and unit["prompt_subset"] == "p5_ctx_32768"
            } == {1, 4}

    engine = payload["engine_params"]
    infeasible_tokens = (
        32768
        + engine["max_new_tokens"]
        + engine["speculative_num_draft_tokens"]
    ) * 16
    assert infeasible_tokens == 528512
    assert infeasible_tokens > engine["max_total_tokens"]
