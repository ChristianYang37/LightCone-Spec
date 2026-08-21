from pathlib import Path

from lightcone_spec.protocol import materialize
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

    second = jobs[1]
    attempt = state.start(second, (0,), tmp_path / "b1")
    state.fail(second.job_id, attempt, "nonfinite", retry=False)
    assert state.skip_pending("preflight", "dependency unavailable") == 1
    assert state.status_counts("preflight") == {"completed": 1, "failed": 1, "skipped": 1}
    assert state.finish_stage("preflight") == "failed"
    assert state.finish_stage("preflight", allow_failed=True) == "completed"


def test_node_can_expand_without_changing_existing_rows(tmp_path: Path):
    state = StateStore(tmp_path)
    probes = materialize("E0-tune", valid_e0=[])
    expanded = materialize("E0-tune", valid_e0=[("m", "DFLASH", "task")])
    state.add_jobs("E0-tune", probes)
    state.add_jobs("E0-tune", expanded)
    assert state.status_counts("E0-tune") == {"pending": 108 + 239}
