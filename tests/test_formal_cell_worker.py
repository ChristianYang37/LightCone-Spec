from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import formal_single_operator_stages as stages
from lightcone_spec.orchestration.experiment_operator import QueuedCommandSpec
from lightcone_spec.orchestration.formal_cell_worker import (
    FormalCellWorkerError,
    FormalCellWorkerSpec,
    load_formal_cell_worker_spec,
    publish_formal_cell_worker_spec,
    revalidate_formal_cell_worker_terminal,
    run_formal_cell_worker,
)


def _command(
    tmp_path: Path,
    *,
    spec_path: Path,
    spec_sha256: str,
    cell_id: str = "cell-001",
    attempt: int = 1,
) -> QueuedCommandSpec:
    evidence = tmp_path / "evidence"
    control = tmp_path / "control"
    return QueuedCommandSpec(
        cell_id=cell_id,
        attempt=attempt,
        argv=(
            sys.executable,
            "-m",
            "lightcone_spec.orchestration.formal_cell_worker",
            "--spec",
            str(spec_path),
        ),
        launch_compatibility_key="model/backend/tp1",
        required_gpu_count=1,
        timing_class="HEADLINE",
        predicted_high_water_bytes=1,
        monitored_path=str(tmp_path),
        log_path=str(control / "command.log"),
        expected_terminal_path=str(evidence / "operator-terminal.json"),
        expected_junit_path=str(evidence / "operator-junit.xml"),
        expected_raw_log_path=str(evidence / "physical-command.log"),
        atomic_pointer_path=str(evidence / "operator-pointer.json"),
        child_exit_receipt_path=str(control / "child-exit.json"),
        environment=(("LIGHTCONE_CELL_WORKER_SPEC_SHA256", spec_sha256),),
    )


def _environment(command: QueuedCommandSpec) -> dict[str, str]:
    return {
        **dict(command.environment),
        "LIGHTCONE_OPERATOR_CELL_ID": command.cell_id,
        "LIGHTCONE_OPERATOR_ATTEMPT": str(command.attempt),
        "LIGHTCONE_OPERATOR_COMMAND_SHA256": command.command_sha256,
        "LIGHTCONE_OPERATOR_TERMINAL_PATH": command.expected_terminal_path,
        "LIGHTCONE_OPERATOR_JUNIT_PATH": command.expected_junit_path,
        "LIGHTCONE_OPERATOR_RAW_LOG_PATH": command.expected_raw_log_path,
        "LIGHTCONE_OPERATOR_POINTER_PATH": command.atomic_pointer_path,
        "LIGHTCONE_OPERATOR_COMMAND_LOG_PATH": command.log_path,
        "LIGHTCONE_OPERATOR_CHILD_EXIT_RECEIPT_PATH": (command.child_exit_receipt_path),
        "LIGHTCONE_OPERATOR_CHILD_START_RECEIPT_PATH": (
            f"{command.child_exit_receipt_path}.start.json"
        ),
        "LIGHTCONE_OPERATOR_HEARTBEAT_PATH": (
            f"{command.child_exit_receipt_path}.heartbeat.json"
        ),
    }


def _spec(tmp_path: Path, *, included: bool = True) -> FormalCellWorkerSpec:
    node = tmp_path / "node-materialization.json"
    node.write_text("{}\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    return FormalCellWorkerSpec(
        schema_version=1,
        kind="formal_single_operator_cell_worker",
        cell_id="cell-001",
        attempt=1,
        repository_root=str(tmp_path),
        node_materialization_path=str(node),
        actual_result_path=str(evidence / "actual.json"),
        evidence_root=str(evidence),
        evidence_manifest_path=str(evidence / "sha256-manifest.json"),
        job_argv=(sys.executable, "-c", "pass"),
        failure_class_on_nonzero="SCIENTIFIC",
        included_in_analysis_on_complete=included,
        complete_exclusion_reason=None if included else "descriptive_only",
    )


def _validation() -> SimpleNamespace:
    return SimpleNamespace(
        status="COMPLETE",
        result_identity_sha256="1" * 64,
        validator_kind="test_actual_validator",
        validator_protocol_sha256="2" * 64,
    )


def test_worker_validates_then_publishes_atomic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    spec_path = tmp_path / "worker-spec.json"
    spec_sha = publish_formal_cell_worker_spec(spec, spec_path)
    command = _command(tmp_path, spec_path=spec_path, spec_sha256=spec_sha)
    monkeypatch.setattr(
        stages,
        "validate_formal_single_operator_cell_actual",
        lambda **_kwargs: _validation(),
    )

    def runner(_argv, **kwargs):
        Path(spec.actual_result_path).write_text(
            '{"kind":"actual"}\n', encoding="utf-8"
        )
        kwargs["stdout"].write(b"physical output\n")
        return SimpleNamespace(returncode=0)

    assert (
        run_formal_cell_worker(
            spec_path,
            environment=_environment(command),
            runner=runner,
        )
        == 0
    )
    terminal = json.loads(Path(command.expected_terminal_path).read_text())
    assert terminal["schema_version"] == 2
    assert terminal["status"] == "COMPLETE"
    assert terminal["result_identity_sha256"] == "1" * 64
    evidence = revalidate_formal_cell_worker_terminal(terminal, command=command)
    assert spec.actual_result_path in evidence
    assert spec.evidence_manifest_path in evidence
    assert command.expected_junit_path in evidence
    assert command.expected_raw_log_path in evidence
    heartbeat = json.loads(
        Path(f"{command.child_exit_receipt_path}.heartbeat.json").read_text()
    )
    assert heartbeat["kind"] == "formal_experiment_child_heartbeat"
    assert heartbeat["command_sha256"] == command.command_sha256
    assert heartbeat["phase"] == "FINALIZING"
    assert heartbeat["sequence"] >= 2


def test_worker_nonzero_is_failed_and_never_claims_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    spec_path = tmp_path / "worker-spec.json"
    spec_sha = publish_formal_cell_worker_spec(spec, spec_path)
    command = _command(tmp_path, spec_path=spec_path, spec_sha256=spec_sha)
    monkeypatch.setattr(
        stages,
        "validate_formal_single_operator_cell_actual",
        lambda **_kwargs: pytest.fail("nonzero job must not validate an actual"),
    )

    def runner(_argv, **kwargs):
        kwargs["stdout"].write(b"physical failure\n")
        return SimpleNamespace(returncode=42)

    assert (
        run_formal_cell_worker(
            spec_path,
            environment=_environment(command),
            runner=runner,
        )
        == 42
    )
    terminal = json.loads(Path(command.expected_terminal_path).read_text())
    assert terminal["status"] == "FAILED"
    assert terminal["failure_class"] == "SCIENTIFIC"
    assert terminal["result_identity_sha256"] is None
    evidence = revalidate_formal_cell_worker_terminal(terminal, command=command)
    assert spec.actual_result_path not in evidence


def test_worker_forwards_term_and_gracefully_seals_infrastructure_terminal(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "physical-ready"
    base = _spec(tmp_path)
    spec = FormalCellWorkerSpec(
        **{
            **vars(base),
            "job_argv": (
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import time; "
                    f"Path({str(ready)!r}).write_text('ready'); time.sleep(60)"
                ),
            ),
        }
    )
    spec_path = tmp_path / "worker-spec.json"
    spec_sha = publish_formal_cell_worker_spec(spec, spec_path)
    command = _command(tmp_path, spec_path=spec_path, spec_sha256=spec_sha)
    environment = {**os.environ, **_environment(command)}
    process = subprocess.Popen(
        list(command.argv),
        cwd=tmp_path,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        os.kill(process.pid, signal.SIGTERM)
        assert process.wait(timeout=10) == 128 + signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    terminal = json.loads(Path(command.expected_terminal_path).read_text())
    assert terminal["status"] == "FAILED"
    assert terminal["failure_class"] == "INFRASTRUCTURE"
    assert terminal["failure_code"] == f"OPERATOR_SIGNAL_{signal.SIGTERM}"
    assert terminal["included_in_analysis"] is False
    assert Path(command.atomic_pointer_path).is_file()


def test_worker_revalidation_detects_result_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    spec_path = tmp_path / "worker-spec.json"
    spec_sha = publish_formal_cell_worker_spec(spec, spec_path)
    command = _command(tmp_path, spec_path=spec_path, spec_sha256=spec_sha)
    monkeypatch.setattr(
        stages,
        "validate_formal_single_operator_cell_actual",
        lambda **_kwargs: _validation(),
    )

    def runner(_argv, **_kwargs):
        Path(spec.actual_result_path).write_text(
            '{"kind":"actual"}\n', encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    run_formal_cell_worker(
        spec_path,
        environment=_environment(command),
        runner=runner,
    )
    terminal = json.loads(Path(command.expected_terminal_path).read_text())
    Path(spec.actual_result_path).write_text('{"kind":"mutated"}\n', encoding="utf-8")
    with pytest.raises(FormalCellWorkerError, match="identity differs"):
        revalidate_formal_cell_worker_terminal(terminal, command=command)


def test_worker_spec_is_canonical_and_no_replace(tmp_path: Path) -> None:
    spec = _spec(tmp_path, included=False)
    path = tmp_path / "worker-spec.json"
    digest = publish_formal_cell_worker_spec(spec, path)
    loaded, observed = load_formal_cell_worker_spec(path)
    assert loaded == spec
    assert observed == digest == spec.sha256
    with pytest.raises(FormalCellWorkerError, match="already exists"):
        publish_formal_cell_worker_spec(spec, path)
