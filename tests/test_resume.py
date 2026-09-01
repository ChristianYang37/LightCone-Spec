import json
import signal
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from lightcone_spec.config import ExperimentConfig, ProtocolConfig, ServerConfig
from lightcone_spec.protocol import E0_ONLINESPEC_RECIPES, Job, materialize
from lightcone_spec.runner import (
    _cleanup_interrupted_servers,
    _complete_blocked_profiler,
    _e2_keep_count,
    _e2_missing_dependency_jobs,
    _exclude_redundant_e2_dependency_jobs,
    _ncu_permission_block_reason,
    _repair_completed_s10_downstream_resume,
    _repair_e0_e6_partial_resume_v1,
    _resume_materialization,
    _save_or_validate_run_config,
    _segment_jobs,
    _select_e0_recipes,
    _selection_for_job,
    _set_e2_expected_evidence,
    _skip_satisfied_e2_dependency_jobs,
    _upgrade_legacy_e0_materialization,
)
from lightcone_spec.state import StateStore


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        source=tmp_path / "paper.yaml",
        run_name="paper-v2-test",
        sglang_root=tmp_path / "sglang",
        results_root=tmp_path,
        models={},
        drafts={},
        datasets={},
        gpu_ids=(0, 1),
        server=ServerConfig(python=tmp_path / "python"),
        protocol=ProtocolConfig(),
    )


def test_interrupt_retry_skip_and_resume(tmp_path: Path):
    state = StateStore(tmp_path)
    jobs = materialize("preflight")[:3]
    state.add_jobs("preflight", jobs)
    first = jobs[0]
    attempt = state.start(first, (0, 1), tmp_path / "attempt-1")
    assert attempt == 1
    assert state.recover_interrupted() == 1
    assert state.next_attempt(first.job_id) == 2
    attempt = state.start(first, (0, 1), tmp_path / "attempt-2")
    state.fail(first.job_id, attempt, "network", retry=True)
    attempt = state.start(first, (0, 1), tmp_path / "attempt-3")
    state.complete(first.job_id, attempt)
    assert state.status_counts("preflight") == {"completed": 1, "pending": 2}


def test_interrupted_server_cleanup_reads_proc_without_spawning_ps(
    monkeypatch, tmp_path: Path
):
    run_dir = tmp_path / "run"
    proc_root = tmp_path / "proc"
    active = run_dir / "jobs" / "job-a" / "attempt-01"
    unrelated = run_dir / "jobs" / "job-b" / "attempt-01"
    stopped = run_dir / "sessions" / "session-c"
    for path, pid in ((active, 101), (unrelated, 102), (stopped, 103)):
        path.mkdir(parents=True)
        (path / "server.pid").write_text(str(pid), encoding="utf-8")
        (proc_root / str(pid)).mkdir(parents=True)
    (proc_root / "101" / "cmdline").write_bytes(
        b"python\0-m\0sglang.launch_server\0"
    )
    (proc_root / "102" / "cmdline").write_bytes(b"python\0worker.py\0")
    (proc_root / "103" / "cmdline").write_bytes(
        b"python\0-m\0sglang.launch_server\0"
    )
    (stopped / "server.stopped").touch()
    killed = []
    monkeypatch.setattr(
        "lightcone_spec.runner.os.killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )

    _cleanup_interrupted_servers(run_dir, proc_root=proc_root)

    assert killed == [(101, signal.SIGTERM)]


def test_e2_halving_floor_never_invents_infeasible_finalists():
    assert _e2_keep_count(53, 53, 1) == 21
    assert _e2_keep_count(21, 20, 2) == 20
    assert _e2_keep_count(20, 20, 2) == 20
    assert _e2_keep_count(20, 20, 3) == 1
    assert _e2_keep_count(21, 0, 2) == 0
    with pytest.raises(ValueError, match="invalid E2 candidate cardinality"):
        _e2_keep_count(20, 21, 2)


def test_explicit_interruption_returns_job_to_pending(tmp_path: Path):
    state = StateStore(tmp_path)
    job = materialize("preflight")[0]
    state.add_jobs("preflight", (job,))
    attempt = state.start(job, (0, 1), tmp_path / "attempt")
    state.interrupt(job.job_id, attempt, "runner cancellation")
    assert state.status_counts("preflight") == {"pending": 1}
    assert state.failed_attempts(job.job_id) == 0
    with state.connect() as connection:
        row = connection.execute(
            "SELECT status,error FROM attempts WHERE job_id=?", (job.job_id,)
        ).fetchone()
    assert (row["status"], row["error"]) == ("interrupted", "runner cancellation")


def test_segment_jobs_resume_without_expanding_paper_stage(tmp_path: Path):
    state = StateStore(tmp_path)
    parent = materialize("E3a")[0]
    state.add_jobs("E3a", (parent,))
    children = _segment_jobs(parent)
    state.add_internal_jobs(children, storage_node="E3a-segments")
    assert len(children) == 2
    assert state.status_counts("E3a") == {"pending": 1}
    assert state.status_counts("E3a-segments") == {"pending": 2}
    attempt = state.start(children[0], (0,), tmp_path / "child")
    state.complete(children[0].job_id, attempt)
    assert state.completed_attempt_dir(children[0].job_id) == tmp_path / "child"


def test_e2_dependency_identity_survives_candidate_reordering(tmp_path: Path):
    state = StateStore(tmp_path)
    first = {
        "parameterization": "lora",
        "rank": 1,
        "scope": "last1",
        "optimizer": "nag",
        "learning_rate": 3e-5,
        "schedule": "constant",
    }
    changed = {
        **first,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
    }

    first_jobs = _e2_missing_dependency_jobs(state, "E2-r2", [first])
    changed_jobs = _e2_missing_dependency_jobs(state, "E2-r2", [changed])
    assert len(first_jobs) == len(changed_jobs) == 1
    assert first_jobs[0].ordinal == changed_jobs[0].ordinal == 0
    assert first_jobs[0].job_id != changed_jobs[0].job_id
    assert "nag__lr-3em05" in first_jobs[0].job_id
    assert "adamw__lr-0p001" in changed_jobs[0].job_id

    state.add_internal_jobs(first_jobs, storage_node="S10-e2-dependency-repair")
    state.add_internal_jobs(changed_jobs, storage_node="S10-e2-dependency-repair")
    state.add_internal_jobs(changed_jobs, storage_node="S10-e2-dependency-repair")
    assert state.status_counts("S10-e2-dependency-repair") == {"pending": 2}


def test_e2_dependency_reuses_equivalent_completed_evidence(tmp_path: Path):
    state = StateStore(tmp_path)
    selection = {
        "parameterization": "lora",
        "rank": 8,
        "scope": "last1",
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "schedule": "constant",
        "registered_request_count": 16,
        "stride": 10,
    }
    source = materialize("E2-r1", e2_rows=[selection])[0]
    completed = replace(
        source,
        job_id="legacy-dependency",
        node="S10-e2-dependency-repair",
        parameters={
            **source.parameters,
            "source_node": "E2-r1",
            "reconciliation_kind": "e2_dependency_closure",
        },
    )
    pending = replace(completed, job_id="recipe-identity-dependency")
    duplicate = replace(
        completed,
        job_id=(
            "s10-e2-dependency-v2__E2-r1__"
            "lora-r8__last1__adamw__lr-0p001__constant"
        ),
    )
    state.add_internal_jobs(
        (completed, pending, duplicate),
        storage_node="S10-e2-dependency-repair",
    )
    attempt_dir = tmp_path / "legacy" / "attempt-01"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "config.json").write_text(json.dumps(completed.to_dict()))
    (attempt_dir / "metrics.json").write_text(json.dumps({"finite": True}))
    attempt = state.start(completed, (0,), attempt_dir)
    state.complete(completed.job_id, attempt)
    duplicate_dir = tmp_path / "duplicate" / "attempt-01"
    duplicate_dir.mkdir(parents=True)
    (duplicate_dir / "config.json").write_text(json.dumps(duplicate.to_dict()))
    (duplicate_dir / "metrics.json").write_text(json.dumps({"finite": True}))
    duplicate_attempt = state.start(duplicate, (1,), duplicate_dir)
    state.complete(duplicate.job_id, duplicate_attempt)

    assert _set_e2_expected_evidence(state, "E2-r1", [selection]) == 0
    assert _exclude_redundant_e2_dependency_jobs(state, "E2-r1") == 1
    assert _skip_satisfied_e2_dependency_jobs(state, "E2-r1") == 1
    assert state.status_counts("S10-e2-dependency-repair") == {
        "completed": 2,
        "skipped": 1,
    }
    assert completed.job_id not in state.selection("formal_evidence_exclusions", [])
    assert duplicate.job_id in state.selection("formal_evidence_exclusions", [])


def test_plain_config_resume_rejects_different_values(tmp_path: Path):
    config = _config(tmp_path)
    config.run_dir.mkdir()
    _save_or_validate_run_config(config)
    _save_or_validate_run_config(config)
    changed = replace(config, server=replace(config.server, max_new_tokens=128))
    with pytest.raises(RuntimeError, match="different experiment config"):
        _save_or_validate_run_config(changed)
    saved = yaml.safe_load((config.run_dir / "paper.yaml").read_text())
    assert saved["protocol"]["preset"] == "paper-v2"
    assert "final_blocks" not in saved["protocol"]


def test_plain_config_resume_allows_new_sglang_path_and_dataset_key(tmp_path: Path):
    config = _config(tmp_path)
    config.run_dir.mkdir()
    _save_or_validate_run_config(config)
    changed = replace(
        config,
        sglang_root=tmp_path / "sglang-v14",
        datasets={"AIME-2024": tmp_path / "aime-2024.jsonl"},
    )
    _save_or_validate_run_config(changed)
    saved = yaml.safe_load((config.run_dir / "paper.yaml").read_text())
    assert saved["paths"]["sglang_root"].endswith("sglang-v14")
    assert set(saved["paths"]["datasets"]) == {"AIME-2024"}


def test_sqlite_records_actual_gpu_pair(tmp_path: Path):
    state = StateStore(tmp_path)
    job = materialize("E6-final")[0]
    state.add_jobs("E6-final", (job,))
    state.start(job, (4, 5), tmp_path / "attempt")
    with state.connect() as connection:
        row = connection.execute(
            "SELECT assigned_gpus FROM jobs WHERE job_id=?", (job.job_id,)
        ).fetchone()
    assert row["assigned_gpus"] == "4,5"


def test_protocol_repair_requeues_without_deleting_attempt_history(tmp_path: Path):
    state = StateStore(tmp_path)
    failed = materialize("E4-profile")[0]
    skipped = materialize("E3b-pilot")[0]
    state.add_jobs("E4-profile", (failed,))
    state.add_jobs("E3b-pilot", (skipped,))
    attempt = state.start(failed, (0, 1), tmp_path / "failed-attempt")
    state.fail(failed.job_id, attempt, "diagnostic", retry=False)
    state.skip_job(skipped.job_id, "upstream failure")

    assert state.retry_failed("E4-profile") == 1
    assert state.reopen_skipped(("E3b-pilot",)) == 1
    assert state.status_counts("E4-profile") == {"pending": 1}
    assert state.status_counts("E3b-pilot") == {"pending": 1}
    with state.connect() as connection:
        attempts = connection.execute(
            "SELECT status,error FROM attempts WHERE job_id=?", (failed.job_id,)
        ).fetchall()
    assert [(row["status"], row["error"]) for row in attempts] == [
        ("failed", "diagnostic")
    ]


def test_targeted_registered_load_retry_preserves_other_failures(tmp_path: Path):
    state = StateStore(tmp_path)
    first, second = materialize("E4-profile")[:2]
    state.add_jobs("E4-profile", (first, second))
    first_attempt = state.start(first, (0, 1), tmp_path / "first-attempt")
    state.fail(
        first.job_id,
        first_attempt,
        "RuntimeError: 9 requests did not complete in a measured cell",
        retry=False,
    )
    second_attempt = state.start(second, (0, 1), tmp_path / "second-attempt")
    state.fail(second.job_id, second_attempt, "RuntimeError: connection refused", retry=False)

    assert (
        state.retry_failed_errors(
            "E4-profile",
            "requests did not complete in a measured cell",
            reason="registered-load timeout classification repair",
        )
        == 1
    )
    assert state.status_counts("E4-profile") == {"failed": 1, "pending": 1}
    with state.connect() as connection:
        attempts = connection.execute(
            "SELECT job_id,status,error FROM attempts ORDER BY job_id"
        ).fetchall()
    assert [(row["job_id"], row["status"], row["error"]) for row in attempts] == [
        (first.job_id, "failed", "RuntimeError: 9 requests did not complete in a measured cell"),
        (second.job_id, "failed", "RuntimeError: connection refused"),
    ]


def test_completed_stage_resume_preserves_immutable_materialization(tmp_path: Path):
    state = StateStore(tmp_path)
    original = materialize("E2-r2")[0]
    state.add_jobs("E2-r2", (original,))
    attempt = state.start(original, (0,), tmp_path / "attempt")
    state.complete(original.job_id, attempt)
    assert state.finish_stage("E2-r2") == "completed"

    changed = replace(original, parameters={**original.parameters, "lr": 9e-4})
    resumed = _resume_materialization(state, "E2-r2", (changed,))
    assert resumed == (original,)
    state.add_jobs("E2-r2", resumed)


def test_pending_stage_resume_uses_current_materialization(tmp_path: Path):
    state = StateStore(tmp_path)
    original = materialize("E2-r2")[0]
    state.add_jobs("E2-r2", (original,))
    changed = replace(original, parameters={**original.parameters, "lr": 9e-4})
    assert _resume_materialization(state, "E2-r2", (changed,)) == (changed,)


def test_reopened_stage_preserves_completed_rows_but_checks_pending_rows(
    tmp_path: Path,
):
    state = StateStore(tmp_path)
    completed, pending = materialize("E1a")[116:118]
    state.add_jobs("E1a", (completed, pending))
    attempt = state.start(completed, (0,), tmp_path / "completed-attempt")
    state.complete(completed.job_id, attempt)

    changed_completed = replace(
        completed,
        parameters={**completed.parameters, "generation_tokens": 16_384},
    )
    changed_pending = replace(
        pending,
        parameters={**pending.parameters, "generation_tokens": 16_384},
    )
    resumed = _resume_materialization(
        state,
        "E1a",
        (changed_completed, changed_pending),
    )

    assert resumed[0].job_id == completed.job_id
    assert resumed[0].parameters["generation_tokens"] == 8192
    assert resumed[1] == changed_pending
    with pytest.raises(RuntimeError, match="row 1 changed after materialization"):
        state.add_jobs("E1a", resumed)


def test_e0_e6_partial_resume_is_scoped_and_idempotent(tmp_path: Path):
    state = StateStore(tmp_path)
    e0 = Job("e0-unsupported", "E0-tune", 0, "static", "m", "DFLASH", "t")
    e0_sibling = Job("e0-sibling", "E0-tune", 1, "static", "m", "DFLASH", "t")
    state.add_jobs("E0-tune", (e0, e0_sibling))
    attempt = state.start(e0, (0, 1), tmp_path / "e0-attempt")
    state.fail(
        e0.job_id,
        attempt,
        "RuntimeError: specialized variants fail closed",
        retry=False,
    )
    state.skip_job(e0_sibling.job_id, "stopped after sibling failure")
    e6_rows = tuple(
        Job(
            f"e6-{index}",
            "E6-common-load-segments",
            index,
            "target_only",
            "m",
            "NONE",
            "t",
        )
        for index in range(2)
    )
    state.add_internal_jobs(e6_rows, storage_node="E6-common-load-segments")
    for index, job in enumerate(e6_rows):
        attempt = state.start(job, (0, 1), tmp_path / f"e6-attempt-{index}")
        state.fail(
            job.job_id,
            attempt,
            "ValueError: dataset supplied 175 prompts; 256 required",
            retry=False,
        )
    downstream = ("E5-pilot", "E5-final", "E6-pilot", "E6-final", "E0-pilot", "E0-final")
    for index, node in enumerate(downstream):
        job = Job(f"downstream-{index}", node, 0, "static", "m", "DFLASH", "t")
        state.add_jobs(node, (job,))
        state.skip_pending(node, "old dependency failure")

    _repair_e0_e6_partial_resume_v1(state)

    assert state.status_counts("E0-tune") == {"pending": 2}
    assert state.status_counts("E6-common-load-segments") == {"pending": 2}
    assert all(state.status_counts(node) == {"pending": 1} for node in downstream)
    audit = state.selection("formal_e0_e6_partial_resume_version")
    assert audit == {"version": 1, "e0_retried": 1, "e6_retried": 2, "reopened": 7}
    _repair_e0_e6_partial_resume_v1(state)
    assert state.selection("formal_e0_e6_partial_resume_version") == audit


def test_e0_source_transfer_upgrade_is_idempotent_and_preserves_old_evidence(
    tmp_path: Path,
):
    state = StateStore(tmp_path)
    planned = materialize("E0-tune")
    legacy = Job(
        job_id="E0-tune__legacy-grid-row",
        node="E0-tune",
        ordinal=12,
        method="onlinespec_ogd",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="CalibrationMix",
        parameters={"stride": 20, "learning_rate": 1e-4},
    )
    state.add_jobs("E0-tune", (*planned[:12], legacy))
    attempt_dir = tmp_path / "legacy-attempt"
    attempt_dir.mkdir()
    (attempt_dir / "config.json").write_text(json.dumps(legacy.to_dict()))
    (attempt_dir / "metrics.json").write_text(json.dumps({"goodput": 1.0}))
    attempt = state.start(legacy, (0, 1), attempt_dir)
    state.complete(legacy.job_id, attempt)

    upgraded = _upgrade_legacy_e0_materialization(state, planned)
    assert upgraded is not None
    assert state.selection("formal_e0_source_transfer_upgrade_version") == 1
    assert legacy.job_id in state.selection("formal_evidence_exclusions")
    assert state.completed_attempt_dir(legacy.job_id) == attempt_dir
    assert len([job for job in state.jobs("E0-tune") if job.job_id in {row.job_id for row in planned}]) == 54

    _upgrade_legacy_e0_materialization(state, planned)
    assert len([job for job in state.jobs("E0-tune") if job.job_id in {row.job_id for row in planned}]) == 54


def test_e0_recipe_selection_injects_only_feasible_validated_methods(tmp_path: Path):
    state = StateStore(tmp_path)
    validations = tuple(
        job
        for job in materialize("E0-tune")
        if job.parameters.get("recipe_validation")
    )
    state.add_jobs("E0-tune", validations)
    for job in validations:
        attempt_dir = tmp_path / job.method / "attempt-01"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "config.json").write_text(json.dumps(job.to_dict()))
        feasible = job.method != "onlinespec_opt"
        (attempt_dir / "metrics.json").write_text(
            json.dumps({"feasible": feasible, "slo_pass": feasible})
        )
        attempt = state.start(job, (0, 1), attempt_dir)
        state.complete(job.job_id, attempt)

    recipes = _select_e0_recipes(state)
    assert set(recipes) == {
        "Qwen/Qwen3-8B|DFLASH|onlinespec_ogd",
        "Qwen/Qwen3-8B|DFLASH|onlinespec_ens",
    }
    state.set_selection("e0_recipes", recipes)
    downstream = materialize("E0-pilot", e0_recipes=recipes)
    assert {job.method for job in downstream if job.method.startswith("onlinespec")} == {
        "onlinespec_ogd",
        "onlinespec_ens",
    }
    job = next(job for job in downstream if job.method == "onlinespec_ens")
    assert json.dumps(_selection_for_job(state, job), sort_keys=True) == json.dumps(
        E0_ONLINESPEC_RECIPES["onlinespec_ens"], sort_keys=True
    )


def test_ncu_permission_probe_reports_provider_block(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 1, "", "==ERROR== ERR_NVGPUCTRPERM")

    monkeypatch.setattr("lightcone_spec.runner.subprocess.run", fake_run)
    reason = _ncu_permission_block_reason(tmp_path / "ncu", tmp_path / "python", 1)
    assert reason == "Nsight Compute counters blocked by provider (ERR_NVGPUCTRPERM)"
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["kwargs"]["timeout"] == 120


def test_blocked_profiler_is_auditable_completed_outcome(tmp_path: Path):
    state = StateStore(tmp_path)
    job = materialize("E4-profile")[2]
    state.add_jobs("E4-profile", (job,))
    _complete_blocked_profiler(
        state,
        job,
        tmp_path,
        (0, 1),
        "Nsight Compute counters blocked by provider (ERR_NVGPUCTRPERM)",
    )
    assert state.status_counts("E4-profile") == {"completed": 1}
    metrics_path = state.completed_attempt_dir(job.job_id) / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["scientific_outcome"] == "blocked"
    assert metrics["feasible"] is False
    assert metrics["profiler"] == "ncu"


def test_completed_s10_repair_requeues_bundled_segments_and_downstream(
    tmp_path: Path,
):
    state = StateStore(tmp_path)
    width = materialize("E3b-pilot")[0]
    failed_segment = replace(
        width,
        job_id="e3-width-test__segment-000",
        node="E3-width-calibration-segments",
    )
    skipped_pilot = materialize("E3b-pilot")[1]
    state.add_internal_jobs((failed_segment,))
    state.add_jobs("E3b-pilot", (skipped_pilot,))
    attempt = state.start(failed_segment, (0,), tmp_path / "failed-segment")
    state.fail(failed_segment.job_id, attempt, "scientific rejection", retry=False)
    state.skip_job(skipped_pilot.job_id, "width calibration incomplete")
    state.mark_stage_failed("E3b-pilot")
    state.set_selection("formal_s10_reconciliation_complete", True)

    _repair_completed_s10_downstream_resume(state)

    assert state.status_counts("E3-width-calibration-segments") == {"pending": 1}
    assert state.status_counts("E3b-pilot") == {"pending": 1}
    assert state.stage_status("E3b-pilot") == "pending"
    assert state.selection("formal_s10_downstream_resume_version") == 2
    audit = json.loads(
        (
            tmp_path
            / "stages"
            / "S10-reconciliation"
            / "downstream-resume-v2.json"
        ).read_text()
    )
    assert audit["width_calibration_segment_retries"] == 1
    assert audit["future_jobs_reopened"] == 1

    _repair_completed_s10_downstream_resume(state)
    assert state.status_counts("E3-width-calibration-segments") == {"pending": 1}
