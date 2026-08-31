from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from lightcone_spec.config import ExperimentConfig, ProtocolConfig, ServerConfig
from lightcone_spec.protocol import materialize
from lightcone_spec.runner import (
    _resume_materialization,
    _save_or_validate_run_config,
    _segment_jobs,
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
