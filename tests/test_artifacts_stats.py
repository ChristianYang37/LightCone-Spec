"""Artifact writer/validator roundtrip via a real CPU smoke unit, plus
statistics conformance (cluster BCa, BH FDR)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lightcone_spec.orchestration.executor import execute_unit
from lightcone_spec.orchestration.units import RunUnit


def _toy_unit(method: str = "tts") -> RunUnit:
    return RunUnit(
        phase="smoke",
        model_pair="toy_markov4",
        method=method,
        dataset="markov4_world",
        prompt_subset="full",
        seed=0,
        lifecycle="request",
        sampling_profile="main_t1_p1",
        trainable_scope="adapter",
        stride=4,
        logical_delay=2,
        concurrency=1,
        contention_condition="none",
        adapter_rank=16,
    )


def test_cpu_unit_writes_valid_run_dir(tmp_path):
    from lightcone_spec.artifacts.rundir import RunDirectory
    from lightcone_spec.artifacts.validator import validate_artifact_root

    unit = _toy_unit()
    outcome = execute_unit(
        unit,
        {"num_requests": 2, "max_rounds": 10, "max_new_tokens": 24},
        tmp_path,
        lockfile_sha256=None,
    )
    assert outcome.status == "complete_valid", outcome.detail
    report = validate_artifact_root(tmp_path, expected_units=[unit.to_manifest_dict()])
    assert report.ok, report.errors
    assert report.unit_status.get(unit.unit_id) == "complete_valid"
    updates = RunDirectory(tmp_path, outcome.run_id).read_table("updates").to_pylist()
    assert updates
    assert all(row["source_training_loss"] is not None for row in updates)
    assert all(
        row["source_expected_accepted_prefix"] is not None for row in updates
    )
    assert all(row["source_prefix_len"] >= 0 for row in updates)


def test_validator_rejects_forged_declared_unit_id(tmp_path):
    import json

    from lightcone_spec.artifacts.validator import validate_artifact_root
    from lightcone_spec.locking.hashing import canonical_json, sha256_bytes, sha256_file

    unit = _toy_unit()
    outcome = execute_unit(
        unit,
        {"num_requests": 1, "max_rounds": 8, "max_new_tokens": 16},
        tmp_path,
        lockfile_sha256=None,
    )
    run_path = tmp_path / outcome.run_id
    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prompt_subset"] = "forged-but-claimed-as-original"
    body = canonical_json(manifest)
    manifest_path.write_text(body)
    manifest_sha_path = run_path / "manifest.sha256"
    manifest_sha_path.write_text(sha256_bytes(body.encode("utf-8")) + "\n")
    hashes_path = run_path / "hashes.json"
    hashes = json.loads(hashes_path.read_text())
    for path in (manifest_path, manifest_sha_path):
        hashes[path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True))

    report = validate_artifact_root(
        tmp_path, expected_units=[unit.to_manifest_dict()]
    )

    assert not report.ok
    assert report.unit_status[unit.unit_id] == "invalid_artifact"
    assert any("manifest unit identity is invalid" in error for error in report.errors)


def test_coverage_detects_missing_unit(tmp_path):
    """The coverage generator must flag never-executed required units."""
    from lightcone_spec.artifacts.validator import validate_artifact_root

    executed = _toy_unit("tts")
    missing = _toy_unit("naive_async")
    execute_unit(
        executed,
        {"num_requests": 1, "max_rounds": 8, "max_new_tokens": 16},
        tmp_path,
        lockfile_sha256=None,
    )
    report = validate_artifact_root(
        tmp_path,
        expected_units=[executed.to_manifest_dict(), missing.to_manifest_dict()],
    )
    assert report.unit_status.get(missing.unit_id) == "missing"


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_exit", "source_exit"),
    (
        ("exactness", "failed_exactness", 5, 5),
        ("numerical", "failed_runtime", 7, 6),
    ),
)
def test_declared_failure_after_run_creation_is_finalized_immutably(
    tmp_path, monkeypatch, error_type, expected_status, expected_exit, source_exit
):
    import json

    from lightcone_spec.artifacts.rundir import RunDirectory
    from lightcone_spec.exit_codes import ExactnessViolation, NumericalFailure
    from lightcone_spec.orchestration import executor

    error = (
        ExactnessViolation("version mismatch")
        if error_type == "exactness"
        else NumericalFailure("nonfinite update")
    )
    monkeypatch.setattr(
        executor,
        "_run_cpu_unit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    outcome = executor.execute_unit(
        _toy_unit(), {}, tmp_path, lockfile_sha256=None
    )

    assert outcome.status == expected_status
    rd = RunDirectory(tmp_path, outcome.run_id)
    assert rd.is_complete
    exit_info = json.loads((rd.path / "exit.json").read_text())
    assert exit_info["status"] == expected_status
    assert exit_info["exit_code"] == expected_exit
    assert exit_info["source_exit_code"] == source_exit
    assert exit_info["error_type"] == type(error).__name__


@pytest.mark.parametrize(
    ("failure_status", "limit", "executed"),
    (("failed_runtime", 2, 2), ("failed_exactness", 1, 1)),
)
def test_manifest_failure_circuit_breaker_stops_repeated_gpu_work(
    tmp_path, monkeypatch, failure_status, limit, executed
):
    from lightcone_spec.orchestration import executor
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    units = [replace(_toy_unit(), seed=seed) for seed in range(4)]
    calls = []

    def fail(unit, *_args, **_kwargs):
        calls.append(unit.unit_id)
        return executor.ExecutionOutcome(
            unit.unit_id,
            f"failed-{len(calls)}",
            failure_status,
            "representative integration failure",
        )

    monkeypatch.setattr(executor, "execute_unit", fail)
    params = {
        "max_consecutive_runtime_failures": 2,
        "max_consecutive_exactness_failures": 1,
    }
    manifest = ExperimentManifest(
        name="failure-breaker",
        phase="smoke",
        description="stop repeated hard failures",
        units=units,
        engine_params=params,
    )

    report = executor.execute_manifest(manifest, tmp_path)

    assert len(calls) == executed == limit
    assert report.counts() == {
        failure_status: executed,
        "not_run_circuit_breaker": len(units) - executed,
    }
    assert not report.ok
    assert all(
        "circuit breaker opened" in outcome.detail
        for outcome in report.outcomes[executed:]
    )


@pytest.mark.parametrize("interleaved_status", ("complete_valid", "skipped_existing"))
def test_failure_circuit_breaker_isolated_from_other_execution_domains(
    tmp_path, monkeypatch, interleaved_status
):
    from lightcone_spec.orchestration import executor
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    units = []
    for seed in range(3):
        units.extend(
            (
                replace(_toy_unit("static"), seed=seed),
                replace(_toy_unit("tts"), seed=seed),
            )
        )
    calls = []

    def execute(unit, *_args, **_kwargs):
        calls.append(unit.method)
        status = "complete_valid" if unit.method == "static" else "failed_runtime"
        return executor.ExecutionOutcome(unit.unit_id, "run", status, status)

    def existing(_root, unit_id, _execution_hash):
        if interleaved_status != "skipped_existing":
            return None
        unit = next(candidate for candidate in units if candidate.unit_id == unit_id)
        return "existing-static" if unit.method == "static" else None

    monkeypatch.setattr(executor, "execute_unit", execute)
    monkeypatch.setattr(executor, "_existing_complete_unit", existing)
    manifest = ExperimentManifest(
        name="domain-isolated-breaker",
        phase="smoke",
        description="Static success cannot mask repeated TTS failures",
        units=units,
        engine_params={"max_consecutive_runtime_failures": 2},
    )

    report = executor.execute_manifest(manifest, tmp_path)

    assert report.counts()["failed_runtime"] == 2
    assert report.counts()["not_run_circuit_breaker"] == 2
    if interleaved_status == "complete_valid":
        assert calls == ["static", "tts", "static", "tts"]
    else:
        assert calls == ["tts", "tts"]
        assert report.counts()["skipped_existing"] == 2


def test_failure_circuit_breaker_success_resets_the_same_domain(
    tmp_path, monkeypatch
):
    from lightcone_spec.orchestration import executor
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    units = [replace(_toy_unit("tts"), seed=seed) for seed in range(5)]
    statuses = iter(
        ("failed_runtime", "complete_valid", "failed_runtime", "failed_runtime")
    )
    calls = []

    def execute(unit, *_args, **_kwargs):
        calls.append(unit.unit_id)
        status = next(statuses)
        return executor.ExecutionOutcome(unit.unit_id, "run", status, status)

    monkeypatch.setattr(executor, "execute_unit", execute)
    manifest = ExperimentManifest(
        name="same-domain-reset",
        phase="smoke",
        description="A same-domain success resets consecutive failures",
        units=units,
        engine_params={"max_consecutive_runtime_failures": 2},
    )

    report = executor.execute_manifest(manifest, tmp_path, resume=False)

    assert len(calls) == 4
    assert report.counts() == {
        "failed_runtime": 3,
        "complete_valid": 1,
        "not_run_circuit_breaker": 1,
    }


@pytest.mark.parametrize("value", (True, -1, 1.5, "2"))
def test_manifest_failure_circuit_breaker_rejects_invalid_limits(
    tmp_path, value
):
    from lightcone_spec.exit_codes import ConfigError
    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    manifest = ExperimentManifest(
        name="invalid-failure-breaker",
        phase="smoke",
        description="invalid limit must fail before a unit starts",
        units=[_toy_unit()],
        engine_params={"max_consecutive_runtime_failures": value},
    )
    with pytest.raises(ConfigError, match="non-negative integer"):
        execute_manifest(manifest, tmp_path)
    assert not list(tmp_path.iterdir())


def _write_empty_attempt(root, manifest, status: str, exit_code: int):
    from lightcone_spec.artifacts.rundir import RunDirectory
    from lightcone_spec.artifacts.schemas import TABLES
    from lightcone_spec.orchestration.executor import _unit_execution_sha256

    unit = manifest.units[0]
    rd = RunDirectory(root, f"attempt-{status}")
    run_manifest = unit.to_manifest_dict()
    run_manifest.update(
        {
            "run_id": rd.run_id,
            "engine_params": manifest.engine_params,
            "experiment_manifest_sha256": manifest.content_sha256(),
            "unit_execution_sha256": _unit_execution_sha256(
                unit, manifest.engine_params, manifest.lockfile_sha256
            ),
        }
    )
    rd.create(run_manifest)
    for table_name in TABLES:
        rd.write_table(table_name, [])
    rd.finalize(exit_code=exit_code, status=status)
    return rd


@pytest.mark.parametrize(
    ("status", "exit_code"),
    (("failed_runtime", 7), ("failed_exactness", 5), ("complete_valid", 0)),
)
def test_resume_rejects_failures_and_empty_normative_telemetry(
    tmp_path, status, exit_code
):
    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    unit = _toy_unit()
    manifest = ExperimentManifest(
        name="resume-fail-closed",
        phase="smoke",
        description="failed or empty attempts are not successful checkpoints",
        units=[unit],
        engine_params={"num_requests": 1, "max_rounds": 4},
    )
    _write_empty_attempt(tmp_path, manifest, status, exit_code)
    report = execute_manifest(manifest, tmp_path)
    assert report.counts() == {"complete_valid": 1}
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("failed_status,exit_code", (("failed_runtime", 7), ("failed_exactness", 5)))
def test_successful_retry_supersedes_structurally_valid_failed_attempt(
    tmp_path, failed_status, exit_code
):
    from lightcone_spec.artifacts.validator import validate_artifact_root
    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    unit = _toy_unit()
    manifest = ExperimentManifest(
        name="retry-history",
        phase="smoke",
        description="failed attempt remains evidence after a successful retry",
        units=[unit],
        engine_params={"num_requests": 1, "max_rounds": 4},
    )
    failed = _write_empty_attempt(tmp_path, manifest, failed_status, exit_code)
    execution = execute_manifest(manifest, tmp_path)
    assert execution.counts() == {"complete_valid": 1}

    validation = validate_artifact_root(
        tmp_path, expected_units=[unit.to_manifest_dict()]
    )
    assert validation.ok, validation.errors
    assert validation.run_status[failed.run_id] == failed_status
    assert validation.unit_status[unit.unit_id] == "complete_valid"


@pytest.mark.parametrize("status", ("failed_runtime", "failed_exactness"))
def test_required_coverage_rejects_failed_units(status):
    from lightcone_spec.artifacts.coverage import build_coverage

    unit = _toy_unit()
    coverage = build_coverage(
        [unit.to_manifest_dict()], {unit.unit_id: status}
    )
    assert coverage.missing_required() == [unit.unit_id]
    assert coverage.summary()["final_validation_ok"] is False


def test_validate_artifacts_returns_incomplete_coverage_for_runtime_failure(
    tmp_path,
):
    from types import SimpleNamespace

    from lightcone_spec import exit_codes
    from lightcone_spec.cli.main import cmd_validate_artifacts
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    unit = _toy_unit()
    manifest = ExperimentManifest(
        name="failed-coverage",
        phase="smoke",
        description="required failed unit",
        units=[unit],
    )
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    _write_empty_attempt(artifact_root, manifest, "failed_runtime", 7)
    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)
    coverage_path = tmp_path / "coverage.json"
    rc = cmd_validate_artifacts(
        SimpleNamespace(
            artifact_root=str(artifact_root),
            manifest=str(manifest_path),
            coverage_output=str(coverage_path),
        )
    )
    assert rc == exit_codes.INCOMPLETE_COVERAGE
    assert coverage_path.is_file()
    assert Path(str(coverage_path) + ".sha256").is_file()


def test_run_dir_immutability(tmp_path):
    unit = _toy_unit()
    outcome = execute_unit(
        unit,
        {"num_requests": 1, "max_rounds": 8, "max_new_tokens": 16},
        tmp_path,
        lockfile_sha256=None,
    )
    from lightcone_spec.artifacts.rundir import RunDirectory

    rd = RunDirectory(tmp_path, outcome.run_id)
    assert rd.is_complete
    marker = (rd.path / "hashes.json").read_bytes()
    with pytest.raises(Exception):
        rd.write_table("rounds", [])  # finalized dirs refuse writes
    with pytest.raises(Exception):
        rd.append_log("stdout", "late write\n")
    with pytest.raises(Exception):
        rd.finalize(0, "complete_valid")
    assert (rd.path / "hashes.json").read_bytes() == marker


def _run_finalize_sigkill_child(
    artifact_root: Path, manifest_path: Path, window: str
):
    import os
    import signal
    import subprocess
    import sys
    import textwrap

    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )
    script = textwrap.dedent(
        """
        import json
        import os
        import signal
        import sys
        from pathlib import Path

        import lightcone_spec.artifacts.rundir as rundir_module
        from lightcone_spec.artifacts.rundir import RunDirectory
        from lightcone_spec.artifacts.schemas import TABLES

        root = Path(sys.argv[1])
        manifest = json.loads(Path(sys.argv[2]).read_text())
        window = sys.argv[3]
        rd = RunDirectory(root, "sigkill-attempt")
        rd.create(manifest)
        for table_name in TABLES:
            rd.write_table(table_name, [])
        runtime = rd.path / "runtime"
        runtime.mkdir()
        (runtime / "adaptation-telemetry-p1-r0.jsonl").write_text("{}\\n")

        real_replace = rundir_module.os.replace
        def replace_then_crash(source, destination):
            if window == "before-publish":
                os.kill(os.getpid(), signal.SIGKILL)
            real_replace(source, destination)
            os.kill(os.getpid(), signal.SIGKILL)

        rundir_module.os.replace = replace_then_crash
        rd.finalize(7, "failed_runtime", {"reason": "fault injection"})
        """
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(artifact_root),
            str(manifest_path),
            window,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
    )
    assert completed.returncode == -signal.SIGKILL


def test_sigkill_before_marker_publish_is_incomplete_and_safely_retried(tmp_path):
    import json

    from lightcone_spec.artifacts.rundir import RunDirectory
    from lightcone_spec.artifacts.validator import validate_artifact_root
    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    unit = _toy_unit()
    manifest = ExperimentManifest(
        name="atomic-finalize-retry",
        phase="smoke",
        description="crash before marker publication is not completion",
        units=[unit],
        engine_params={"num_requests": 1, "max_rounds": 4},
    )
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    manifest_path = tmp_path / "unit-manifest.json"
    manifest_path.write_text(json.dumps(unit.to_manifest_dict()))

    _run_finalize_sigkill_child(artifact_root, manifest_path, "before-publish")
    crashed = artifact_root / "sigkill-attempt"
    assert not (crashed / "hashes.json").exists()
    assert not RunDirectory(artifact_root, crashed.name).is_complete

    execution = execute_manifest(manifest, artifact_root)
    assert execution.counts() == {"complete_valid": 1}
    report = validate_artifact_root(
        artifact_root, expected_units=[unit.to_manifest_dict()]
    )
    assert report.ok, report.errors
    assert any("sigkill-attempt" in warning for warning in report.warnings)


def test_sigkill_after_marker_publish_leaves_complete_immutable_ledger(tmp_path):
    import json

    from lightcone_spec.artifacts.rundir import RunDirectory
    from lightcone_spec.artifacts.validator import validate_artifact_root

    unit = _toy_unit()
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    manifest_path = tmp_path / "unit-manifest.json"
    manifest_path.write_text(json.dumps(unit.to_manifest_dict()))

    _run_finalize_sigkill_child(artifact_root, manifest_path, "after-publish")
    rd = RunDirectory(artifact_root, "sigkill-attempt")
    assert rd.is_complete
    hashes = json.loads((rd.path / "hashes.json").read_text())
    assert "runtime/adaptation-telemetry-p1-r0.jsonl" in hashes
    marker = (rd.path / "hashes.json").read_bytes()
    report = validate_artifact_root(artifact_root)
    assert report.ok, report.errors
    assert report.run_status[rd.run_id] == "failed_runtime"
    with pytest.raises(Exception):
        rd.append_log("stderr", "late write\n")
    assert (rd.path / "hashes.json").read_bytes() == marker


def test_legacy_pre_auxiliary_marker_is_ignored_and_retried(tmp_path):
    from lightcone_spec.artifacts.rundir import RunDirectory
    from lightcone_spec.artifacts.validator import validate_artifact_root
    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    unit = _toy_unit()
    manifest = ExperimentManifest(
        name="legacy-crash-window",
        phase="smoke",
        description="old marker published before auxiliary provenance binding",
        units=[unit],
        engine_params={"num_requests": 1, "max_rounds": 4},
    )
    first = execute_manifest(manifest, tmp_path)
    old = RunDirectory(tmp_path, first.outcomes[0].run_id)
    runtime = old.path / "runtime"
    runtime.mkdir()
    (runtime / "late-old-writer.jsonl").write_text("{}\n")
    assert not old.is_complete

    retry = execute_manifest(manifest, tmp_path)
    assert retry.counts() == {"complete_valid": 1}
    report = validate_artifact_root(
        tmp_path, expected_units=[unit.to_manifest_dict()]
    )
    assert report.ok, report.errors
    assert any(old.run_id in warning for warning in report.warnings)


def test_corrupt_legacy_marker_is_ignored_and_retried(tmp_path):
    from lightcone_spec.artifacts.rundir import RunDirectory
    from lightcone_spec.artifacts.validator import validate_artifact_root
    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    unit = _toy_unit()
    manifest = ExperimentManifest(
        name="corrupt-completion-marker",
        phase="smoke",
        description="a torn legacy marker is an incomplete attempt",
        units=[unit],
        engine_params={"num_requests": 1, "max_rounds": 4},
    )
    first = execute_manifest(manifest, tmp_path)
    old = RunDirectory(tmp_path, first.outcomes[0].run_id)
    (old.path / "hashes.json").write_text('{"manifest.json":')
    assert not old.is_complete

    retry = execute_manifest(manifest, tmp_path)
    assert retry.counts() == {"complete_valid": 1}
    report = validate_artifact_root(
        tmp_path, expected_units=[unit.to_manifest_dict()]
    )
    assert report.ok, report.errors
    assert any(old.run_id in warning for warning in report.warnings)


def test_auxiliary_runtime_and_checkpoint_provenance_is_hash_bound(
    tmp_path, monkeypatch
):
    import json

    from lightcone_spec.artifacts.validator import validate_artifact_root
    from lightcone_spec.orchestration import executor

    unit = RunUnit(**{**_toy_unit().__dict__, "phase": "p5-smoke"})
    original = executor._run_cpu_unit

    def run_with_raw_evidence(run_unit, engine_params, run_id):
        rows = original(run_unit, engine_params, run_id)
        run_path = tmp_path / run_id
        runtime = run_path / "runtime"
        runtime.mkdir()
        (runtime / "adaptation-telemetry-p1-r0.jsonl").write_text(
            json.dumps({"kind": "round", "request_id": "measured"}) + "\n"
        )
        (run_path / "prefix-checkpoints.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "checkpoints": [
                        {
                            "sample_id": "sample:ctx-16",
                            "source_sample_id": "sample",
                            "context_length": 16,
                        }
                    ],
                }
            )
        )
        return rows

    monkeypatch.setattr(executor, "_run_cpu_unit", run_with_raw_evidence)
    outcome = executor.execute_unit(unit, {}, tmp_path, lockfile_sha256=None)
    run_path = tmp_path / outcome.run_id
    hashes = json.loads((run_path / "hashes.json").read_text())
    assert "runtime/adaptation-telemetry-p1-r0.jsonl" in hashes
    assert "prefix-checkpoints.json" in hashes
    assert validate_artifact_root(tmp_path).ok

    telemetry = run_path / "runtime/adaptation-telemetry-p1-r0.jsonl"
    telemetry.write_text(telemetry.read_text() + "{}\n")
    report = validate_artifact_root(tmp_path)
    assert not report.ok
    assert report.unit_status[unit.unit_id] == "invalid_artifact"
    assert any("hash drift: runtime/" in error for error in report.errors)


@pytest.mark.parametrize(
    ("status", "exit_code"),
    (("failed_runtime", 7), ("failed_exactness", 5), ("complete_valid", 0)),
)
def test_analyze_rejects_failed_or_empty_artifacts(
    tmp_path, status, exit_code
):
    from types import SimpleNamespace

    from lightcone_spec.cli.main import cmd_analyze
    from lightcone_spec.exit_codes import ArtifactValidationFailure
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    unit = _toy_unit()
    manifest = ExperimentManifest(
        name="analysis-fail-closed",
        phase="smoke",
        description="do not analyze failed attempts",
        units=[unit],
    )
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    _write_empty_attempt(artifact_root, manifest, status, exit_code)
    with pytest.raises(ArtifactValidationFailure):
        cmd_analyze(
            SimpleNamespace(
                artifact_root=str(artifact_root),
                output_dir=str(tmp_path / "analysis"),
                manifest=None,
                baseline="static",
                itl_slo_ms=50.0,
            )
        )


def test_analysis_outputs_are_transitively_hash_bound(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace

    from lightcone_spec.cli.main import cmd_analyze

    artifact_root = tmp_path / "runs"
    outcome = execute_unit(
        _toy_unit("static"),
        {"num_requests": 1, "max_rounds": 8, "max_new_tokens": 16},
        artifact_root,
        lockfile_sha256=None,
    )
    output_dir = tmp_path / "analysis"
    assert (
        cmd_analyze(
            SimpleNamespace(
                artifact_root=str(artifact_root),
                output_dir=str(output_dir),
                manifest=None,
                baseline="static",
                itl_slo_ms=50.0,
            )
        )
        == 0
    )
    manifest_path = output_dir / "analysis-manifest.json"
    manifest_sha = (output_dir / "analysis-manifest.sha256").read_text().strip()
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_sha
    manifest = json.loads(manifest_path.read_text())
    assert manifest["input_runs"] == [
        {
            "run_id": outcome.run_id,
            "unit_id": _toy_unit("static").unit_id,
            "manifest_sha256": manifest["input_runs"][0]["manifest_sha256"],
            "hashes_sha256": manifest["input_runs"][0]["hashes_sha256"],
        }
    ]
    hashes = json.loads((output_dir / "analysis-hashes.json").read_text())
    for relative, entry in hashes.items():
        path = output_dir / relative
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert set(manifest["derived_outputs"]) <= set(hashes)


def test_p5_context_performance_requires_an_explicit_timing_contract():
    from lightcone_spec.cli.main import _p5_checkpoint_performance_scope
    from lightcone_spec.exit_codes import ArtifactValidationFailure

    assert _p5_checkpoint_performance_scope({}) == "mixed_workload_global"
    assert _p5_checkpoint_performance_scope(
        {"p5_context_timing_contract": "independent_exact_context_group_v1"}
    ) == "checkpoint_request"
    with pytest.raises(ArtifactValidationFailure, match="timing contract"):
        _p5_checkpoint_performance_scope(
            {"p5_context_timing_contract": "unverified_context_timing"}
        )


def test_validate_and_analyze_apply_the_execution_mode_overlay(tmp_path):
    """A shared root must expose only the requested effective experiment."""
    import json
    from types import SimpleNamespace

    from lightcone_spec.cli.main import cmd_analyze, cmd_validate_artifacts
    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    source = ExperimentManifest(
        name="mode-overlay-evidence",
        phase="smoke",
        description="validate the same in-memory overlay used for execution",
        units=[_toy_unit("static"), _toy_unit("tts"), _toy_unit("naive_async")],
        engine_params={"num_requests": 1, "max_rounds": 4, "max_new_tokens": 8},
    )
    source_path = tmp_path / "source-manifest.json"
    source.write(source_path)
    methods = ["static", "tts"]
    lifecycles = ["request"]
    learning_rate = 2e-4
    effective = (
        source.with_methods(methods)
        .with_lifecycles(lifecycles)
        .with_weight_update_mode("lora")
        .with_learning_rate(learning_rate)
    )
    artifact_root = tmp_path / "runs"
    residual = (
        source.with_methods(methods)
        .with_lifecycles(lifecycles)
        .with_weight_update_mode("residual")
        .with_learning_rate(learning_rate)
    )
    first = execute_manifest(residual, artifact_root)
    assert first.counts() == {"complete_valid": 2}
    execution = execute_manifest(effective, artifact_root)
    assert execution.counts() == {
        "skipped_existing": 1,
        "complete_valid": 1,
    }

    # A failed unit outside the effective method/mode overlay remains immutable
    # evidence in the shared root, but cannot poison this analysis.
    unrelated = ExperimentManifest(
        name="unrelated-failure",
        phase=source.phase,
        description="outside the requested effective unit set",
        units=[source.units[2]],
        engine_params=source.engine_params,
    )
    unrelated_attempt = _write_empty_attempt(
        artifact_root, unrelated, "failed_runtime", 7
    )
    # Corrupt an immutable attempt outside the requested effective manifest.
    # Whole-root validation must still catch it, while manifest-scoped
    # validation and analysis must neither fail nor attest it.
    (unrelated_attempt.path / "rounds.parquet").write_bytes(b"corrupt parquet")
    from lightcone_spec.artifacts.validator import validate_artifact_root

    strict = validate_artifact_root(artifact_root)
    assert not strict.ok
    assert strict.run_status[unrelated_attempt.run_id] == "invalid_artifact"
    requested_corrupt = validate_artifact_root(
        artifact_root, expected_units=unrelated.expected_units()
    )
    assert not requested_corrupt.ok
    assert unrelated_attempt.run_id in requested_corrupt.checked_runs
    assert requested_corrupt.unit_status[unrelated.units[0].unit_id] == (
        "invalid_artifact"
    )
    scoped = validate_artifact_root(
        artifact_root, expected_units=effective.expected_units()
    )
    assert scoped.ok, scoped.errors
    assert unrelated_attempt.run_id not in scoped.checked_runs
    assert unrelated_attempt.run_id not in scoped.run_status
    assert unrelated.units[0].unit_id not in scoped.unit_status
    assert any(
        unrelated_attempt.run_id in warning and "unrelated unit" in warning
        for warning in scoped.warnings
    )

    coverage_path = tmp_path / "coverage.json"
    assert cmd_validate_artifacts(
        SimpleNamespace(
            artifact_root=str(artifact_root),
            manifest=str(source_path),
            weight_update_mode="lora",
            methods=methods,
            lifecycles=lifecycles,
            learning_rate=learning_rate,
            coverage_output=str(coverage_path),
        )
    ) == 0
    coverage = json.loads(coverage_path.read_text())
    assert set(coverage["cells"]) == {unit.unit_id for unit in effective.units}
    assert coverage["summary"]["total_units"] == 2

    analysis_root = tmp_path / "analysis"
    assert cmd_analyze(
        SimpleNamespace(
            artifact_root=str(artifact_root),
            output_dir=str(analysis_root),
            manifest=str(source_path),
            weight_update_mode="lora",
            methods=methods,
            lifecycles=lifecycles,
            learning_rate=learning_rate,
            baseline="static",
            itl_slo_ms=50.0,
        )
    ) == 0
    analysis_manifest = json.loads(
        (analysis_root / "analysis-manifest.json").read_text()
    )
    provenance = analysis_manifest["analysis"]
    assert provenance["weight_update_mode_overlay"] == "lora"
    assert provenance["methods_overlay"] == methods
    assert provenance["lifecycles_overlay"] == lifecycles
    assert provenance["learning_rate_overlay"] == pytest.approx(learning_rate)
    assert provenance["expected_manifest_sha256"] == effective.content_sha256()
    assert {row["unit_id"] for row in analysis_manifest["input_runs"]} == {
        unit.unit_id for unit in effective.units
    }


def test_additive_telemetry_schema_keeps_legacy_v1_artifacts_valid(tmp_path):
    import json

    import pyarrow.parquet as pq

    from lightcone_spec.artifacts.rundir import RunDirectory
    from lightcone_spec.artifacts.schemas import (
        SCHEMA_COMPAT_OPTIONAL_FIELDS,
        TABLES,
    )
    from lightcone_spec.artifacts.validator import validate_artifact_root
    from lightcone_spec.locking.hashing import sha256_file

    unit = _toy_unit()
    outcome = execute_unit(
        unit,
        {"num_requests": 1, "max_rounds": 8, "max_new_tokens": 16},
        tmp_path,
        lockfile_sha256=None,
    )
    rd = RunDirectory(tmp_path, outcome.run_id)

    # Simulate schema-v1 Parquet produced before the additive telemetry fields
    # existed. The compatibility allow-list is intentionally narrow: original
    # schema columns remain mandatory.
    for table_name, optional_fields in SCHEMA_COMPAT_OPTIONAL_FIELDS.items():
        assert optional_fields <= set(TABLES[table_name].names)
        assert all(TABLES[table_name].field(name).nullable for name in optional_fields)
        path = rd.path / f"{table_name}.parquet"
        table = pq.read_table(path)
        legacy_columns = [
            name for name in table.schema.names if name not in optional_fields
        ]
        pq.write_table(table.select(legacy_columns), path)

    hashes_path = rd.path / "hashes.json"
    hashes = json.loads(hashes_path.read_text())
    for table_name in SCHEMA_COMPAT_OPTIONAL_FIELDS:
        relative = f"{table_name}.parquet"
        path = rd.path / relative
        hashes[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True))
    report = validate_artifact_root(tmp_path)
    assert report.ok, report.errors
    assert report.unit_status == {unit.unit_id: "complete_valid"}


def test_resume_requires_full_manifest_identity(tmp_path):
    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    unit = _toy_unit()
    common = dict(
        name="resume-identity",
        phase="smoke",
        description="resume identity test",
        units=[unit],
    )
    first = ExperimentManifest(
        **common,
        engine_params={"num_requests": 1, "max_rounds": 4, "lr": 1e-3},
    )
    changed = ExperimentManifest(
        **common,
        engine_params={"num_requests": 1, "max_rounds": 4, "lr": 1e-2},
    )

    assert execute_manifest(first, tmp_path).counts() == {"complete_valid": 1}
    assert execute_manifest(first, tmp_path).counts() == {"skipped_existing": 1}
    assert execute_manifest(changed, tmp_path).counts() == {"complete_valid": 1}
    assert sum(path.is_dir() for path in tmp_path.iterdir()) == 2


def test_runtime_implementation_fingerprint_is_bound_to_resume_identity(tmp_path):
    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    unit = _toy_unit("static")
    common = dict(
        name="implementation-identity",
        phase="smoke",
        description="runtime source changes must invalidate completed units",
        units=[unit],
    )
    first = ExperimentManifest(
        **common,
        engine_params={
            "num_requests": 1,
            "max_rounds": 4,
            "runtime_implementation_fingerprint": {"sha256": "a" * 64},
        },
    )
    changed = ExperimentManifest(
        **common,
        engine_params={
            "num_requests": 1,
            "max_rounds": 4,
            "runtime_implementation_fingerprint": {"sha256": "b" * 64},
        },
    )

    assert execute_manifest(first, tmp_path).counts() == {"complete_valid": 1}
    assert execute_manifest(first, tmp_path).counts() == {"skipped_existing": 1}
    assert execute_manifest(changed, tmp_path).counts() == {"complete_valid": 1}


def test_static_resume_reuses_across_only_mode_and_controller_root(tmp_path):
    import json

    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    unit = _toy_unit("static")
    common = dict(
        name="static-tier-reuse",
        phase="smoke",
        description="same static execution across tail tier sweeps",
        units=[unit],
        lockfile_sha256="a" * 64,
    )
    first = ExperimentManifest(
        **common,
        engine_params={
            "num_requests": 1,
            "max_rounds": 4,
            "lr": 1e-3,
            "weight_update_mode_override": "output_residual",
            "controller_root": "/controllers/output",
        },
    )
    second = ExperimentManifest(
        **common,
        engine_params={
            "num_requests": 1,
            "max_rounds": 4,
            "lr": 1e-3,
            "weight_update_mode_override": "tail_lora",
            "controller_root": "/controllers/lora",
        },
    )
    assert first.content_sha256() != second.content_sha256()
    first_report = execute_manifest(first, tmp_path)
    second_report = execute_manifest(second, tmp_path)
    assert first_report.counts() == {"complete_valid": 1}
    assert second_report.counts() == {"skipped_existing": 1}
    run_manifest = json.loads(
        (tmp_path / first_report.outcomes[0].run_id / "manifest.json").read_text()
    )
    assert run_manifest["experiment_manifest_sha256"] == first.content_sha256()
    assert len(run_manifest["unit_execution_sha256"]) == 64


def test_execution_resume_does_not_ignore_real_inputs_or_adapted_controller(tmp_path):
    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    static = _toy_unit("static")
    base = dict(num_requests=1, max_rounds=4, lr=1e-3)
    first = ExperimentManifest(
        name="strict-static",
        phase="smoke",
        description="bind non-tier input",
        units=[static],
        engine_params=base,
        lockfile_sha256="a" * 64,
    )
    changed_lock = ExperimentManifest(
        name="strict-static",
        phase="smoke",
        description="bind non-tier input",
        units=[static],
        engine_params=base,
        lockfile_sha256="b" * 64,
    )
    assert execute_manifest(first, tmp_path / "static").counts() == {
        "complete_valid": 1
    }
    assert execute_manifest(changed_lock, tmp_path / "static").counts() == {
        "complete_valid": 1
    }
    changed_engine = ExperimentManifest(
        name="strict-static",
        phase="smoke",
        description="bind non-tier input",
        units=[static],
        engine_params={**base, "lr": 1e-2},
        lockfile_sha256="a" * 64,
    )
    assert execute_manifest(changed_engine, tmp_path / "static").counts() == {
        "complete_valid": 1
    }

    adapted = _toy_unit("tts")
    adapted_a = ExperimentManifest(
        name="strict-adapted",
        phase="smoke",
        description="controller root remains bound for adapted units",
        units=[adapted],
        engine_params={**base, "controller_root": "/controllers/a"},
    )
    adapted_b = ExperimentManifest(
        name="strict-adapted",
        phase="smoke",
        description="controller root remains bound for adapted units",
        units=[adapted],
        engine_params={**base, "controller_root": "/controllers/b"},
    )
    assert execute_manifest(adapted_a, tmp_path / "adapted").counts() == {
        "complete_valid": 1
    }
    assert execute_manifest(adapted_b, tmp_path / "adapted").counts() == {
        "complete_valid": 1
    }


def test_cluster_bca_covers_known_shift():
    from lightcone_spec.statistics.bootstrap import cluster_bca

    rng = np.random.Generator(np.random.PCG64(0))
    values = rng.standard_normal(400) + 0.5
    clusters = np.repeat(np.arange(40), 10)
    res = cluster_bca(values, clusters, b=800, seed=0)
    assert res.ci_low < 0.5 < res.ci_high
    assert res.excludes_zero, "CI should exclude zero for a 0.5 shift"


def test_cluster_bca_rejects_misaligned_or_nonfinite_evidence():
    from lightcone_spec.statistics.bootstrap import cluster_bca

    with pytest.raises(ValueError, match="differ in length"):
        cluster_bca(np.asarray([1.0, 2.0]), np.asarray(["a"]), b=10)
    with pytest.raises(ValueError, match="finite"):
        cluster_bca(
            np.asarray([1.0, float("nan")]),
            np.asarray(["a", "b"]),
            b=10,
        )


def test_controller_artifact_rejects_malformed_hash_sidecar(tmp_path):
    from lightcone_spec.controller.artifact import ControllerArtifact
    from lightcone_spec.exit_codes import ConfigError

    path = tmp_path / "controller.json"
    path.write_text("{}")
    Path(str(path) + ".sha256").write_text("not-a-sha256\n")

    with pytest.raises(ConfigError, match="hash sidecar is invalid"):
        ControllerArtifact.load(path)


def test_bh_fdr_monotone():
    from lightcone_spec.statistics.fdr import benjamini_hochberg

    p = [0.001, 0.011, 0.02, 0.8]
    rejected = benjamini_hochberg(p, q=0.05)
    assert rejected[0] and not rejected[3]
    # monotone: any p smaller than a rejected one is rejected too
    for i, r in enumerate(rejected):
        if r:
            assert all(rejected[j] for j in range(len(p)) if p[j] <= p[i])


def test_claim_gates():
    from lightcone_spec.statistics.hypotheses import exactness_gate, h1_gate

    assert exactness_gate(0, True)["pass"]
    assert not exactness_gate(1, True)["pass"]
    assert not exactness_gate(0, False)["pass"]
    # H1: >=15% MAE improvement with CI excluding zero
    assert h1_gate(delay_mae=1.0, path_mae=0.8, ci95=(0.05, 0.3))["pass"]
    assert not h1_gate(delay_mae=1.0, path_mae=0.95, ci95=(0.01, 0.09))["pass"]


def test_dual_load_profile_selection_respects_itl_slo():
    import pandas as pd

    from lightcone_spec.statistics.tables import select_load_profiles

    rows = []
    for concurrency, tps, p99 in ((1, 10.0, 10.0), (4, 32.0, 25.0), (8, 40.0, 80.0)):
        rows.append(
            {
                "request_id": f"r{concurrency}",
                "model_pair_id": "pair",
                "method": "static",
                "dataset": "data",
                "lifecycle": "request",
                "status": "complete_valid",
                "offered_concurrency": concurrency,
                "decode_tps": tps,
                "p99_itl_ms": p99,
            }
        )
    selected = select_load_profiles(pd.DataFrame(rows), itl_slo_ms=50.0)
    by_name = {row["profile"]: row for row in selected}
    assert by_name["throughput"]["selected_concurrency"] == 8
    assert by_name["throughput"]["saturation_confirmed"] is False
    assert by_name["throughput"]["recommended_next_concurrency"] == 16
    assert by_name["latency_slo"]["selected_concurrency"] == 4
    assert by_name["latency_slo"]["slo_met"] is True


def test_load_profile_confirms_a_flat_boundary():
    import pandas as pd

    from lightcone_spec.statistics.tables import select_load_profiles

    rows = [
        {
            "request_id": f"r{concurrency}",
            "model_pair_id": "pair",
            "method": "static",
            "dataset": "data",
            "lifecycle": "request",
            "status": "complete_valid",
            "offered_concurrency": concurrency,
            "decode_tps": tps,
            "p99_itl_ms": 10.0,
        }
        for concurrency, tps in ((16, 100.0), (32, 102.0))
    ]
    selected = select_load_profiles(pd.DataFrame(rows), itl_slo_ms=50.0)
    throughput = next(row for row in selected if row["profile"] == "throughput")
    assert throughput["selected_concurrency"] == 32
    assert throughput["last_step_goodput_gain_fraction"] == pytest.approx(0.02)
    assert throughput["saturation_confirmed"] is True
    assert throughput["recommended_next_concurrency"] is None


def test_load_profile_excludes_retracted_capacity_points():
    import pandas as pd

    from lightcone_spec.statistics.tables import select_load_profiles

    rows = [
        {
            "request_id": f"r{concurrency}",
            "model_pair_id": "pair",
            "method": "static",
            "dataset": "data",
            "lifecycle": "request",
            "status": "complete_valid",
            "offered_concurrency": concurrency,
            "decode_tps": tps,
            "p99_itl_ms": 10.0,
            "kv_retracted_requests": retractions,
        }
        for concurrency, tps, retractions in (
            (16, 100.0, 0),
            (32, 120.0, 0),
            (48, 140.0, 2),
        )
    ]
    selected = select_load_profiles(pd.DataFrame(rows), itl_slo_ms=50.0)
    throughput = next(row for row in selected if row["profile"] == "throughput")
    assert throughput["selected_concurrency"] == 32
    assert throughput["capacity_limited"] is True
    assert throughput["saturation_confirmed"] is False
    assert throughput["recommended_next_concurrency"] is None


def test_load_profile_never_pools_weight_update_modes():
    import pandas as pd

    from lightcone_spec.statistics.tables import select_load_profiles

    rows = []
    for mode, best_concurrency in (("tail_lora", 1), ("full_rank_tail", 4)):
        for concurrency in (1, 4):
            rows.append(
                {
                    "request_id": f"{mode}-{concurrency}",
                    "model_pair_id": "pair",
                    "method": "naive_async",
                    "weight_update_mode": mode,
                    "dataset": "data",
                    "lifecycle": "stream",
                    "status": "complete_valid",
                    "offered_concurrency": concurrency,
                    "decode_tps": 20.0
                    if concurrency == best_concurrency
                    else 1.0,
                    "p99_itl_ms": 10.0,
                }
            )
    selected = select_load_profiles(pd.DataFrame(rows), itl_slo_ms=50.0)
    throughput = {
        row["weight_update_mode"]: row["selected_concurrency"]
        for row in selected
        if row["profile"] == "throughput"
    }
    assert throughput == {"tail_lora": 1, "full_rank_tail": 4}


def test_load_profile_never_pools_update_strides():
    import pandas as pd

    from lightcone_spec.statistics.tables import select_load_profiles

    rows = []
    for stride, best_concurrency in ((1, 1), (4, 4)):
        for concurrency in (1, 4):
            rows.append(
                {
                    "request_id": f"stride-{stride}-{concurrency}",
                    "model_pair_id": "pair",
                    "method": "tts",
                    "weight_update_mode": "tail_lora",
                    "update_stride": stride,
                    "dataset": "data",
                    "lifecycle": "stream",
                    "status": "complete_valid",
                    "offered_concurrency": concurrency,
                    "decode_tps": 20.0
                    if concurrency == best_concurrency
                    else 1.0,
                    "p99_itl_ms": 10.0,
                }
            )
    selected = select_load_profiles(pd.DataFrame(rows), itl_slo_ms=50.0)
    throughput = {
        row["update_stride"]: row["selected_concurrency"]
        for row in selected
        if row["profile"] == "throughput"
    }
    assert throughput == {1: 1, 4: 4}


def test_method_table_uses_repetitions_as_speedup_clusters():
    import pandas as pd

    from lightcone_spec.statistics.tables import method_table

    rows = []
    static_walls = (2.0, 2.2, 1.8, 2.1, 1.9)
    adapted_walls = (2.5, 2.6, 2.4, 2.8, 2.3)
    for method, walls in (("static", static_walls), ("naive_async", adapted_walls)):
        for repetition, wall in enumerate(walls):
            for prompt in range(2):
                rows.append(
                    {
                        "method": method,
                        "lifecycle": "request",
                        "status": "complete_valid",
                        "prompt_id_hash": f"prompt-{prompt}:repeat-{repetition}",
                        "seed": 0,
                        "offered_concurrency": 1,
                        "output_tokens": 100,
                        "decode_wall_s": wall,
                        "e2e_wall_s": wall,
                        "decode_tps": 1.0,
                        "e2e_tps": 1.0,
                        "mean_accepted_drafts": 1.0,
                        "mean_committed_per_verify": 2.0,
                        "target_calls_per_output_token": 0.5,
                        "quality_value": None,
                        "p95_round_ms": 1.0,
                        "p50_itl_ms": 1.0,
                        "p95_itl_ms": 1.0,
                        "p99_itl_ms": 1.0,
                        "version_mismatch_count": 0,
                    }
                )
    table = method_table(pd.DataFrame(rows), "request", b=400)
    adapted = table[table["method"] == "naive_async"].iloc[0]
    assert adapted["speedup_n_clusters"] == 5
    assert adapted["speedup_ci_low"] < adapted["speedup_ci_high"]
    assert adapted["speedup_vs_static"] < 1.0


def test_method_table_does_not_pseudoreplicate_copied_run_throughput():
    import pandas as pd

    from lightcone_spec.statistics.tables import method_table

    rows = []
    for method, throughput in (("static", 50.0), ("naive_async", 51.0)):
        for prompt in range(8):
            rows.append(
                {
                    "method": method,
                    "lifecycle": "request",
                    "status": "complete_valid",
                    "prompt_id_hash": f"prompt-{prompt}",
                    "seed": 0,
                    "offered_concurrency": 4,
                    "output_tokens": 100,
                    "decode_wall_s": 2.0,
                    "e2e_wall_s": 2.0,
                    # SGLang copies this run-level value to each request.
                    "decode_tps": throughput,
                    "e2e_tps": throughput,
                    "mean_accepted_drafts": 1.0,
                    "mean_committed_per_verify": 2.0,
                    "target_calls_per_output_token": 0.5,
                    "quality_value": None,
                    "p95_round_ms": 1.0,
                    "p50_itl_ms": 1.0,
                    "p95_itl_ms": 1.0,
                    "p99_itl_ms": 1.0,
                    "version_mismatch_count": 0,
                }
            )
    table = method_table(pd.DataFrame(rows), "request", b=100)
    adapted = table[table["method"] == "naive_async"].iloc[0]
    assert adapted["speedup_vs_static"] == 1.02
    assert adapted["speedup_vs_baseline"] == adapted["speedup_vs_static"]
    assert adapted["baseline_method"] == "static"
    assert adapted["speedup_n_clusters"] == 1
    assert np.isnan(adapted["speedup_ci_low"])
    assert np.isnan(adapted["speedup_ci_high"])

    versus_tts = method_table(
        pd.DataFrame(rows), "request", baseline_method="naive_async", b=100
    )
    assert set(versus_tts["baseline_method"]) == {"naive_async"}
    assert "speedup_vs_baseline" in versus_tts
    assert "speedup_vs_static" not in versus_tts
    baseline_row = versus_tts[versus_tts["method"] == "naive_async"].iloc[0]
    assert baseline_row["speedup_vs_baseline"] == 1.0

    from lightcone_spec.statistics.tables import paired_method_delta

    delta = paired_method_delta(
        pd.DataFrame(rows), "naive_async", "static", b=100
    )
    assert delta.estimate == 1.0
    assert delta.n_clusters == 1
    assert delta.method == "insufficient_runs"
    assert np.isnan(delta.ci_low)
    assert np.isnan(delta.ci_high)


def test_p5_survival_lcag_elasticity_and_onset_are_prompt_paired(tmp_path):
    import pandas as pd

    from lightcone_spec.statistics.tables import (
        acceptance_elasticity_table,
        long_context_acceptance_table,
    )

    static = {
        512: (3, 3, 2, 2),
        1024: (3, 2, 2, 1),
        4096: (2, 2, 1, 1),
        8192: (2, 1, 1, 0),
    }
    adapted = {
        512: (3, 3, 2, 2),
        1024: (3, 3, 2, 1),
        4096: (3, 2, 2, 1),
        8192: (3, 2, 2, 1),
    }
    rows = []
    for method, curve in (("static", static), ("lc_damp", adapted)):
        for context, accepted in curve.items():
            for prompt in range(8):
                for round_id, value in enumerate(accepted):
                    rows.append(
                        {
                            "method": method,
                            "dataset": "math500",
                            "lifecycle": "stream",
                            "offered_concurrency": 1,
                            "context_length": context,
                            "request_id": f"prompt-{prompt}",
                            "prompt_cluster": f"prompt-{prompt}",
                            "seed": 0,
                            "round_id": round_id,
                            "prefix_len_before": context + round_id,
                            "draft_tokens": 7,
                            "verify_len": 8,
                            "accepted_drafts": value,
                            "committed_per_verify": value + 1,
                            "target_calls": 1,
                            "draft_cuda_us": 10.0,
                            "verify_cuda_us": 20.0,
                            "accept_cuda_us": 2.0,
                            "batch_size": 1,
                            "version_canary_ok": True,
                        }
                    )
    rounds = pd.DataFrame(rows)
    table = long_context_acceptance_table(rounds, b=200)
    row = table[(table.method == "lc_damp") & (table.context_length == 4096)].iloc[0]
    survival_sum = sum(row[f"survival_k{k}"] for k in range(1, 8))
    assert survival_sum == pytest.approx(row.survival_weighted_accepted_prefix)
    assert row.lcag_ci_low > 0
    assert row.benefit_onset_status == "confirmed"
    assert row.verified_drafts_per_verify == 7
    assert row.verification_waste > 0

    aggregates = []
    for key, group in rounds.groupby(
        ["method", "dataset", "lifecycle", "offered_concurrency", "context_length", "request_id", "seed"]
    ):
        record = dict(
            zip(
                ["method", "dataset", "lifecycle", "offered_concurrency", "context_length", "request_id", "seed"],
                key,
            )
        )
        record.update(
            {
                "round_count": len(group),
                "accepted_sum": group.accepted_drafts.sum(),
                "committed_sum": group.committed_per_verify.sum(),
                "verified_sum": (group.verify_len - 1).sum(),
                "waste_sum": ((group.verify_len - 1) - group.accepted_drafts).sum(),
                "target_calls_sum": group.target_calls.sum(),
                "draft_cuda_us_sum": group.draft_cuda_us.sum(),
                "verify_cuda_us_sum": group.verify_cuda_us.sum(),
                "accept_cuda_us_sum": group.accept_cuda_us.sum(),
                "batch_size_sum": group.batch_size.sum(),
                "version_mismatch_count": 0,
                "observed_prefix_min": group.prefix_len_before.min(),
                "observed_prefix_max": group.prefix_len_before.max(),
                "prefix_len_before": group.prefix_len_before.min(),
                "draft_tokens": 7,
                "verify_len": 8,
                "accepted_drafts": group.accepted_drafts.mean(),
                "committed_per_verify": group.committed_per_verify.mean(),
                "target_calls": 1,
                "draft_cuda_us": 10.0,
                "verify_cuda_us": 20.0,
                "accept_cuda_us": 2.0,
                "batch_size": 1,
                "version_canary_ok": True,
            }
        )
        for k in range(1, 8):
            record[f"survival_count_k{k}"] = int((group.accepted_drafts >= k).sum())
        aggregates.append(record)
    aggregate_table = long_context_acceptance_table(pd.DataFrame(aggregates), b=200)
    aggregate_row = aggregate_table[
        (aggregate_table.method == "lc_damp")
        & (aggregate_table.context_length == 4096)
    ].iloc[0]
    assert aggregate_row.survival_weighted_accepted_prefix == pytest.approx(
        row.survival_weighted_accepted_prefix
    )

    shape = acceptance_elasticity_table(rounds, b=200)
    assert set(shape.shape_semantics) == {"pointwise_context_shape"}
    long_elasticity = shape[
        (shape.metric == "elasticity")
        & (shape.method == "lc_damp")
        & (shape.context_left == 4096)
    ].iloc[0]
    assert long_elasticity.delta_vs_static < 0
    assert long_elasticity.delta_vs_baseline == long_elasticity.delta_vs_static
    assert {"elasticity", "curvature"} == set(shape.metric)
    from lightcone_spec.plots.figures import (
        acceptance_cost_pareto_figure,
        acceptance_shape_figure,
        long_context_acceptance_figure,
    )

    table["throughput_speedup_vs_baseline"] = 1.0
    table["throughput_speedup_vs_static"] = 1.0
    table["round_cuda_us"] = (
        table.draft_cuda_us + table.verify_cuda_us + table.accept_cuda_us
    )
    shape_nan = shape[shape.metric == "elasticity"].copy()
    shape_nan[["estimate", "ci_low", "ci_high"]] = np.nan
    paths = (
        long_context_acceptance_figure(table, tmp_path / "acceptance.png"),
        acceptance_shape_figure(shape, tmp_path / "shape.png"),
        acceptance_shape_figure(
            shape[shape.metric == "elasticity"], tmp_path / "shape-no-curvature.png"
        ),
        acceptance_shape_figure(shape_nan, tmp_path / "shape-nan.png"),
        acceptance_cost_pareto_figure(table, tmp_path / "pareto.png"),
    )
    assert all(path.stat().st_size > 0 for path in paths)

    # A non-Static baseline must never be serialized or plotted under a
    # ``vs_static`` label.  Re-labeling the paired baseline is sufficient to
    # exercise the identity-aware P5 estimators without changing the curves.
    tts_rounds = rounds.copy()
    tts_rounds.loc[tts_rounds.method == "static", "method"] = "tts"
    tts_table = long_context_acceptance_table(
        tts_rounds, baseline_method="tts", b=200
    )
    assert set(tts_table.baseline_method) == {"tts"}
    assert "acceptance_gain_vs_baseline" in tts_table
    assert "acceptance_gain_vs_static" not in tts_table
    tts_shape = acceptance_elasticity_table(
        tts_rounds, baseline_method="tts", b=200
    )
    assert set(tts_shape.baseline_method) == {"tts"}
    assert "delta_vs_baseline" in tts_shape
    assert "delta_vs_static" not in tts_shape
    tts_table["throughput_speedup_vs_baseline"] = 1.0
    tts_table["round_cuda_us"] = (
        tts_table.draft_cuda_us
        + tts_table.verify_cuda_us
        + tts_table.accept_cuda_us
    )
    tts_pareto = acceptance_cost_pareto_figure(
        tts_table, tmp_path / "pareto-vs-tts.png"
    )
    assert tts_pareto.stat().st_size > 0


def test_p5_batch_mean_is_scheduler_step_weighted():
    import pandas as pd

    from lightcone_spec.statistics.tables import long_context_acceptance_table

    rows = []
    # Two scheduler steps: B=2 then B=4.  Per-request telemetry repeats the
    # batch size B times, so a naive row mean would be 20/6 rather than 3.
    for round_id, batch_size in enumerate((2, 4)):
        for request in range(batch_size):
            rows.append(
                {
                    "method": "static",
                    "model_pair": "qwen3_4b_dspark7",
                    "weight_update_mode": "output_residual",
                    "dataset": "math500",
                    "lifecycle": "stream",
                    "offered_concurrency": 4,
                    "context_length": 4096,
                    "request_id": f"step-{round_id}-request-{request}",
                    "prompt_cluster": f"prompt-{request}",
                    "seed": 0,
                    "round_id": round_id,
                    "prefix_len_before": 4096 + round_id,
                    "draft_tokens": 3,
                    "verify_len": 4,
                    "accepted_drafts": 1,
                    "committed_per_verify": 2,
                    "target_calls": 1,
                    "draft_cuda_us": 1.0,
                    "verify_cuda_us": 2.0,
                    "accept_cuda_us": 1.0,
                    "batch_size": batch_size,
                    "version_canary_ok": True,
                }
            )
    row = long_context_acceptance_table(pd.DataFrame(rows), b=20).iloc[0]
    assert row.update_stride == 4
    assert row.mean_batch_size == pytest.approx(3.0)
    assert row.request_weighted_mean_batch_size == pytest.approx(20.0 / 6.0)
    assert row.decode_step_count_estimate == pytest.approx(2.0)


def test_p5_adaptation_cost_is_complete_and_does_not_double_count_breakdown():
    import pandas as pd

    from lightcone_spec.statistics.tables import long_context_acceptance_table

    rows = []
    for method in ("static", "tts"):
        rows.append(
            {
                "method": method,
                "model_pair": "qwen3_4b_dflash16",
                "weight_update_mode": "tail_lora",
                "update_stride": 4,
                "dataset": "livecodebench",
                "lifecycle": "stream",
                "offered_concurrency": 8,
                "context_length": 4096,
                "request_id": f"prompt-{method}",
                "prompt_cluster": "prompt-0",
                "seed": 0,
                "round_count": 10,
                "accepted_sum": 20,
                "committed_sum": 30,
                "verified_sum": 40,
                "waste_sum": 20,
                "target_calls_sum": 10,
                "draft_tokens": 4,
                "verify_len": 5,
                "accepted_drafts": 2,
                "committed_per_verify": 3,
                "target_calls": 1,
                "draft_cuda_us_sum": 100.0,
                "verify_cuda_us_sum": 200.0,
                "accept_cuda_us_sum": 30.0,
                "draft_cuda_us": 10.0,
                "verify_cuda_us": 20.0,
                "accept_cuda_us": 3.0,
                "signal_prep_cuda_us_sum": 100.0 if method == "tts" else 0.0,
                "signal_prep_timed_rounds": 10,
                "signal_prep_unknown_rounds": 0,
                "update_count": 2 if method == "tts" else 0,
                "update_cost_evidence_complete": True,
                "side_queue_cuda_us_sum": 40.0 if method == "tts" else 0.0,
                "candidate_cuda_us_sum": 200.0 if method == "tts" else 0.0,
                "backward_cuda_us_sum": 80.0 if method == "tts" else 0.0,
                "optimizer_cuda_us_sum": 20.0 if method == "tts" else 0.0,
                "controller_cuda_us_sum": 10.0 if method == "tts" else 0.0,
                "publish_cuda_us_sum": 5.0 if method == "tts" else 0.0,
                "barrier_wait_cpu_us_sum": 50.0 if method == "tts" else 0.0,
                "batch_size": 8,
                "batch_size_sum": 80.0,
                "batch_reciprocal_sum": 1.25,
                "version_canary_ok": True,
                "version_mismatch_count": 0,
                "observed_prefix_min": 4096,
                "observed_prefix_max": 4600,
                "prefix_len_before": 4096,
            }
        )
    table = long_context_acceptance_table(pd.DataFrame(rows), b=20)
    row = table[table.method == "tts"].iloc[0]
    assert bool(row.adaptation_cost_complete)
    assert row.signal_prep_cuda_us == pytest.approx(10.0)
    assert row.candidate_cuda_us == pytest.approx(20.0)
    assert row.backward_cuda_us == pytest.approx(8.0)
    assert row.optimizer_cuda_us == pytest.approx(2.0)
    assert row.controller_cuda_us == pytest.approx(1.0)
    assert row.publish_cuda_us == pytest.approx(0.5)
    # Candidate is inclusive of backward/optimizer; the total adds it once.
    assert row.adaptation_cuda_us == pytest.approx(31.5)


def test_p5_excludes_censored_algorithmic_gain_but_keeps_physical_cost():
    import pandas as pd

    from lightcone_spec.statistics.tables import long_context_acceptance_table

    base = {
        "method": "static",
        "model_pair": "qwen3_4b_dflash16",
        "weight_update_mode": "tail_lora",
        "dataset": "livecodebench",
        "lifecycle": "stream",
        "offered_concurrency": 1,
        "context_length": 4096,
        "request_id": "prompt-0",
        "prompt_cluster": "prompt-0",
        "seed": 0,
        "draft_tokens": 4,
        "verify_len": 5,
        "committed_per_verify": 2,
        "target_calls": 1,
        "verify_cuda_us": 10.0,
        "draft_cuda_us": 2.0,
        "accept_cuda_us": 1.0,
        "batch_size": 1,
        "version_canary_ok": True,
    }
    rounds = pd.DataFrame(
        [
            {
                **base,
                "round_id": 0,
                "prefix_len_before": 4096,
                "accepted_drafts": 1,
                "algorithmic_censored": False,
            },
            {
                **base,
                "round_id": 1,
                "prefix_len_before": 4098,
                "accepted_drafts": 4,
                "committed_per_verify": 5,
                "verify_cuda_us": 30.0,
                "algorithmic_censored": True,
            },
        ]
    )
    row = long_context_acceptance_table(rounds, b=20).iloc[0]
    assert row.rounds == 1
    assert row.physical_rounds == 2
    assert row.algorithmic_censored_rounds == 1
    assert row.survival_weighted_accepted_prefix == pytest.approx(1.0)
    assert row.committed_tokens_per_verify == pytest.approx(2.0)
    assert row.verify_cuda_us == pytest.approx(20.0)

    aggregated = pd.DataFrame(
        [
            {
                **base,
                "prefix_len_before": 4096,
                "round_count": 2,
                "semantic_round_count": 1,
                "accepted_sum": 5,
                "semantic_accepted_sum": 1,
                "committed_sum": 7,
                "semantic_committed_sum": 2,
                "verified_sum": 8,
                "semantic_verified_sum": 4,
                "waste_sum": 3,
                "semantic_waste_sum": 3,
                "target_calls_sum": 2,
                "semantic_target_calls_sum": 1,
                "draft_cuda_us_sum": 4.0,
                "verify_cuda_us_sum": 40.0,
                "accept_cuda_us_sum": 2.0,
                "batch_size_sum": 2.0,
                "batch_reciprocal_sum": 2.0,
                "algorithmic_censored_count": 1,
                "semantic_survival_count_k1": 1,
                "semantic_survival_count_k2": 0,
                "semantic_survival_count_k3": 0,
                "semantic_survival_count_k4": 0,
                "version_mismatch_count": 0,
                "observed_prefix_min": 4096,
                "observed_prefix_max": 4098,
                "accepted_drafts": 2.5,
            }
        ]
    )
    aggregate_row = long_context_acceptance_table(aggregated, b=20).iloc[0]
    for field in (
        "rounds",
        "physical_rounds",
        "algorithmic_censored_rounds",
        "survival_weighted_accepted_prefix",
        "committed_tokens_per_verify",
        "verify_cuda_us",
    ):
        assert aggregate_row[field] == pytest.approx(row[field])


def test_p5_single_context_has_schema_valid_empty_shape_table():
    import pandas as pd

    from lightcone_spec.statistics.tables import acceptance_elasticity_table

    rounds = pd.DataFrame(
        [
            {
                "method": method,
                "dataset": "livecodebench",
                "lifecycle": "stream",
                "offered_concurrency": 1,
                "context_length": 16384,
                "request_id": f"{method}-prompt",
                "seed": 0,
                "draft_tokens": 3,
                "accepted_drafts": 2,
                "committed_per_verify": 3,
                "target_calls": 1,
                "draft_cuda_us": 1,
                "verify_cuda_us": 2,
                "accept_cuda_us": 1,
                "version_canary_ok": True,
            }
            for method in ("static", "tts")
        ]
    )
    shape = acceptance_elasticity_table(rounds, b=20)
    assert shape.empty
    assert {"metric", "context_center", "delta_vs_static"}.issubset(shape.columns)


def test_p5_never_pools_backends_or_weight_update_modes():
    import pandas as pd

    from lightcone_spec.statistics.tables import (
        acceptance_elasticity_table,
        long_context_acceptance_table,
    )

    rows = []
    for pair, mode, gain in (
        ("qwen3_8b_dflash16", "tail_lora", 1),
        ("qwen3_8b_dflash16", "full_rank_tail", 2),
        ("qwen3_8b_eagle3", "tail_lora", 3),
    ):
        for context in (4096, 8192):
            for prompt in range(3):
                common = {
                    "model_pair": pair,
                    "dataset": "math500",
                    "lifecycle": "stream",
                    "offered_concurrency": 1,
                    "context_length": context,
                    "request_id": f"{pair}-prompt-{prompt}",
                    "prompt_cluster": f"prompt-{prompt}",
                    "seed": 0,
                    "draft_tokens": 7,
                    "verify_len": 8,
                    "committed_per_verify": 2,
                    "target_calls": 1,
                    "draft_cuda_us": 1.0,
                    "verify_cuda_us": 2.0,
                    "accept_cuda_us": 1.0,
                    "batch_size": 1,
                    "version_canary_ok": True,
                    "prefix_len_before": context,
                }
                # One shared static row is deliberately labelled residual;
                # normalization must reuse it for each adapted mode of the
                # same pair, never for a different backend.
                rows.append(
                    {
                        **common,
                        "method": "static",
                        "weight_update_mode": "output_residual",
                        "accepted_drafts": 1,
                    }
                )
                rows.append(
                    {
                        **common,
                        "method": "lc_damp",
                        "weight_update_mode": mode,
                        "accepted_drafts": 1 + gain,
                        "committed_per_verify": 2 + gain,
                    }
                )

    rounds = pd.DataFrame(rows).drop_duplicates(
        [
            "model_pair",
            "method",
            "context_length",
            "request_id",
            "weight_update_mode",
        ]
    )
    table = long_context_acceptance_table(rounds, b=40)
    adapted = table[table.method == "lc_damp"]
    assert len(adapted) == 6
    assert set(zip(adapted.model_pair, adapted.weight_update_mode)) == {
        ("qwen3_8b_dflash16", "lora"),
        ("qwen3_8b_dflash16", "full"),
        ("qwen3_8b_eagle3", "lora"),
    }
    assert adapted.groupby(["model_pair", "weight_update_mode"])[
        "acceptance_gain_vs_static"
    ].first().to_dict() == {
        ("qwen3_8b_dflash16", "lora"): 1.0,
        ("qwen3_8b_dflash16", "full"): 2.0,
        ("qwen3_8b_eagle3", "lora"): 3.0,
    }
    shape = acceptance_elasticity_table(rounds, b=40)
    assert set(zip(shape.model_pair, shape.weight_update_mode)) >= {
        ("qwen3_8b_dflash16", "lora"),
        ("qwen3_8b_dflash16", "full"),
        ("qwen3_8b_eagle3", "lora"),
    }


def test_p5_never_pools_update_strides_and_reuses_static_once():
    import pandas as pd

    from lightcone_spec.statistics.tables import (
        acceptance_elasticity_table,
        expand_static_p5_identities,
        long_context_acceptance_table,
    )

    rows = []
    for context in (4096, 8192):
        for prompt in range(3):
            common = {
                "model_pair": "qwen3_4b_dflash16",
                "weight_update_mode": "lora",
                "dataset": "math500",
                "lifecycle": "stream",
                "offered_concurrency": 20,
                "context_length": context,
                "prompt_cluster": f"prompt-{prompt}",
                "seed": 0,
                "draft_tokens": 7,
                "verify_len": 8,
                "target_calls": 1,
                "draft_cuda_us": 1.0,
                "verify_cuda_us": 2.0,
                "accept_cuda_us": 1.0,
                "batch_size": 1,
                "version_canary_ok": True,
                "prefix_len_before": context,
            }
            rows.append(
                {
                    **common,
                    "method": "static",
                    "update_stride": 1,
                    "request_id": f"static-{context}-{prompt}",
                    "accepted_drafts": 1,
                    "committed_per_verify": 2,
                }
            )
            for stride, gain in ((1, 1), (4, 2)):
                rows.append(
                    {
                        **common,
                        "method": "tts",
                        "update_stride": stride,
                        "request_id": f"tts-s{stride}-{context}-{prompt}",
                        "accepted_drafts": 1 + gain,
                        "committed_per_verify": 2 + gain,
                    }
                )

    rounds = pd.DataFrame(rows)
    table = long_context_acceptance_table(rounds, b=40)
    tts = table[table.method == "tts"]
    assert set(tts.update_stride) == {1, 4}
    assert tts.groupby("update_stride")[
        "acceptance_gain_vs_static"
    ].first().to_dict() == {1: 1.0, 4: 2.0}
    assert set(tts.gain_prompt_clusters) == {3}

    # The one Static run is copied to both observed adaptive strides, but each
    # table cell still contains exactly the original three prompt rounds.
    static = table[table.method == "static"]
    assert set(static.update_stride) == {1, 4}
    assert set(static.physical_rounds) == {3}
    assert len(static) == 4

    shape = acceptance_elasticity_table(rounds, b=40)
    tts_shape = shape[shape.method == "tts"]
    assert set(tts_shape.update_stride) == {1, 4}
    assert set(tts_shape.paired_prompt_clusters) == {3}

    # Performance summaries use the same joint expansion, so their baseline
    # cannot be pooled or multiplied independently of the acceptance path.
    summaries = pd.DataFrame(
        [
            {
                "model_pair": "qwen3_4b_dflash16",
                "method": "static",
                "weight_update_mode": "lora",
                "update_stride": 1,
                "dataset": "math500",
                "lifecycle": "stream",
                "offered_concurrency": 20,
                "context_length": 4096,
                "decode_tps": 100.0,
            },
            *(
                {
                    "model_pair": "qwen3_4b_dflash16",
                    "method": "tts",
                    "weight_update_mode": "lora",
                    "update_stride": stride,
                    "dataset": "math500",
                    "lifecycle": "stream",
                    "offered_concurrency": 20,
                    "context_length": 4096,
                    "decode_tps": tps,
                }
                for stride, tps in ((1, 90.0), (4, 110.0))
            ),
        ]
    )
    expanded = expand_static_p5_identities(summaries)
    expanded_static = expanded[expanded.method == "static"]
    assert expanded_static.groupby("update_stride").decode_tps.apply(
        list
    ).to_dict() == {1: [100.0], 4: [100.0]}


def test_p5_prefix_checkpoints_are_exact_and_resource_gated(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from lightcone_spec.exit_codes import ResourceSkip
    from lightcone_spec.orchestration.units import RunUnit
    from lightcone_spec.sglang_bridge.client import (
        _p5_context_lengths,
        _p5_prefix_prompts,
    )

    class Tokenizer:
        eos_token_id = 0

        @staticmethod
        def encode(text, add_special_tokens=False):
            assert add_special_tokens is False
            return [1 + ord(char) % 251 for char in text]

        @staticmethod
        def apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
            chat_template_kwargs=None,
            **kwargs,
        ):
            assert tokenize is True
            assert add_generation_prompt is True
            thinking = bool(enable_thinking) or bool(
                (chat_template_kwargs or {}).get("enable_thinking")
            )
            assert thinking is True
            text = "".join(message["content"] for message in messages)
            # Distinct from raw encode so thinking path is observable.
            return [200 + ord(char) % 50 for char in text] + [999]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: Tokenizer()
            )
        ),
    )
    unit = RunUnit(
        phase="p5",
        model_pair="qwen3_4b_dspark7",
        method="static",
        dataset="math500",
        prompt_subset="p5_ctx_32",
        seed=0,
        lifecycle="stream",
        sampling_profile="main_t1_p1",
        trainable_scope="adapter",
        stride=4,
        logical_delay=0,
        concurrency=1,
        contention_condition="none",
        adapter_rank=16,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    params = {
        "model_roots": {"Qwen/Qwen3-4B": "/verified/qwen"},
        "adaptation_config_path": str(run_dir / "adaptation.runtime.yaml"),
        "max_new_tokens": 4,
        "max_total_tokens": 100,
        "speculative_num_draft_tokens": 8,
    }
    samples = [
        SimpleNamespace(sample_id="a", prompt="alpha"),
        SimpleNamespace(sample_id="b", prompt="beta"),
    ]
    prompts = _p5_prefix_prompts(unit, params, samples, 32)
    assert all(len(prompt["input_ids"]) == 32 for prompt in prompts)
    evidence = (run_dir / "prefix-checkpoints.json").read_text()
    assert "input_ids_sha256" in evidence
    grouped = RunUnit(
        **{**unit.__dict__, "prompt_subset": "p5_ctx_32-64"}
    )
    params["p5_context_lengths"] = [32, 64]
    assert _p5_context_lengths(grouped, params) == (32, 64)
    grouped_prompts = _p5_prefix_prompts(
        grouped, params, samples, _p5_context_lengths(grouped, params)
    )
    assert [len(prompt["input_ids"]) for prompt in grouped_prompts] == [32, 32, 64, 64]
    constrained = RunUnit(**{**unit.__dict__, "concurrency": 4})
    with pytest.raises(ResourceSkip, match="KV token slots"):
        _p5_prefix_prompts(constrained, params, samples, 32)
