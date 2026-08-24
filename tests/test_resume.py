import json
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.config import ExperimentConfig, ProtocolConfig, ServerConfig
from lightcone_spec.protocol import materialize
from lightcone_spec.runner import (
    _complete_infeasible_startup,
    _node_final_blocks,
    _save_or_validate_run_config,
)
from lightcone_spec.state import StateStore


def test_interrupt_retry_skip_and_resume(tmp_path: Path):
    state = StateStore(tmp_path)
    jobs = materialize("preflight")[:3]
    state.add_jobs("preflight", jobs)

    first = jobs[0]
    attempt = state.start(first, (0, 1), tmp_path / "a1")
    assert attempt == 1
    assert state.recover_interrupted() == 1
    assert state.status_counts("preflight")["pending"] == 3
    assert state.next_attempt(first.job_id) == 2
    assert state.failed_attempts(first.job_id) == 0

    attempt = state.start(first, (0, 1), tmp_path / "a2")
    state.fail(first.job_id, attempt, "network", retry=True)
    assert state.failed_attempts(first.job_id) == 1
    assert state.next_attempt(first.job_id) == 3
    attempt = state.start(first, (0, 1), tmp_path / "a3")
    state.complete(first.job_id, attempt)
    assert state.jobs("preflight") == jobs

    second = jobs[1]
    attempt = state.start(second, (0,), tmp_path / "b1")
    state.fail(second.job_id, attempt, "nonfinite", retry=False)
    assert state.skip_pending("preflight", "dependency unavailable") == 1
    assert state.status_counts("preflight") == {"completed": 1, "failed": 1, "skipped": 1}
    assert state.finish_stage("preflight") == "failed"


def test_node_can_expand_without_changing_existing_rows(tmp_path: Path):
    state = StateStore(tmp_path)
    probes = materialize("E0-tune", valid_e0=[])
    expanded = materialize("E0-tune", valid_e0=[("m", "DFLASH", "task")])
    state.add_jobs("E0-tune", probes)
    state.add_jobs("E0-tune", expanded)
    assert state.status_counts("E0-tune") == {"pending": 108 + 239}


def test_screening_startup_oom_is_a_completed_infeasible_row(tmp_path: Path):
    state = StateStore(tmp_path)
    job = replace(
        materialize("E3a")[0],
        job_id="screening-startup-oom",
        node="E6-interface",
    )
    state.add_internal_jobs((job,))
    _complete_infeasible_startup(
        state, job, tmp_path, (0, 1), RuntimeError("CUDA out of memory")
    )
    assert state.status_counts("E6-interface") == {"completed": 1}
    directory = state.completed_attempt_dirs("E6-interface")[0]
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["scientific_outcome"] == "infeasible"
    assert metrics["feasible"] is False


def test_internal_jobs_resume_without_adding_a_paper_stage(tmp_path: Path):
    state = StateStore(tmp_path)
    job = replace(
        materialize("E1a")[0],
        job_id="e1a-confidence-weight-0p1",
        node="E1a-confidence-calibration",
    )
    state.add_internal_jobs((job,))
    assert state.pending_jobs(job.node) == (job,)
    assert all(row["node"] != job.node for row in state.stage_rows())


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        source=tmp_path / "source.yaml",
        run_name="run",
        sglang_root=tmp_path / "sglang",
        results_root=tmp_path,
        models={},
        drafts={},
        datasets={},
        gpu_ids=(0, 1),
        server=ServerConfig(python=tmp_path / "python"),
        protocol=ProtocolConfig(),
    )


def test_plain_config_resume_and_global_final_n(tmp_path: Path):
    config = _config(tmp_path)
    config.run_dir.mkdir()
    _save_or_validate_run_config(config)
    _save_or_validate_run_config(config)
    changed = replace(config, server=replace(config.server, max_new_tokens=128))
    with pytest.raises(RuntimeError, match="different experiment config"):
        _save_or_validate_run_config(changed)

    state = StateStore(config.run_dir)
    state.set_selection("global_final_blocks", 17)
    assert _node_final_blocks(config, state, "E3b-final") == 17
    assert _node_final_blocks(config, state, "E5-final") == 17
    assert _node_final_blocks(config, state, "E6-final") == 17
    assert _node_final_blocks(config, state, "E0-final") == 17


def test_sqlite_records_actual_pair_not_the_whole_pool(tmp_path: Path):
    state = StateStore(tmp_path)
    job = materialize("E6-final", final_blocks=12)[0]
    state.add_jobs("E6-final", (job,))
    state.start(job, (4, 5), tmp_path / "attempt")
    with state.connect() as connection:
        row = connection.execute(
            "SELECT assigned_gpus FROM jobs WHERE job_id=?", (job.job_id,)
        ).fetchone()
    assert row["assigned_gpus"] == "4,5"
