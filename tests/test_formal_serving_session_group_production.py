from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.orchestration.experiment_operator import (
    ProcessObservation,
    QueuedCommandSpec,
)
from lightcone_spec.orchestration.formal_cell_worker import (
    FormalCellWorkerSpec,
    load_formal_cell_worker_spec,
    publish_formal_cell_worker_spec,
)
from lightcone_spec.orchestration.formal_serving_session_group_production import (
    FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV,
    FormalServingSessionGroupChildHeartbeatPublisher,
    build_formal_serving_session_group_production_spec,
    ensure_formal_serving_session_group_production_outputs_unoccupied,
    execute_formal_serving_session_group_production,
    formal_serving_session_group_production_environment,
    formal_serving_session_group_production_spec_path_from_command,
    formal_serving_session_group_shared_evidence_bound_bytes,
    publish_formal_serving_session_group_production_spec,
    revalidate_formal_serving_session_group_production_publication,
    revalidate_formal_serving_session_group_production_spec,
    revalidate_formal_serving_session_group_production_terminal,
    revalidate_formal_serving_session_group_production_terminals,
)
from lightcone_spec.orchestration.formal_serving_session_group_worker import (
    FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
    FormalServingSessionGroupCellArtifact,
    FormalServingSessionGroupExecutionResult,
    revalidate_formal_serving_session_group_execution,
)
from lightcone_spec.orchestration.live_sglang import PinnedNvidiaSmiTool
from lightcone_spec.orchestration.native_terminal import canonical_sha256
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_RESIDENT_FIXTURE_PATH = Path(__file__).with_name(
    "test_formal_serving_session_group_resident.py"
)
_RESIDENT_SPEC = importlib.util.spec_from_file_location(
    "_production_resident_fixture", _RESIDENT_FIXTURE_PATH
)
assert _RESIDENT_SPEC is not None and _RESIDENT_SPEC.loader is not None
_RESIDENT = importlib.util.module_from_spec(_RESIDENT_SPEC)
_RESIDENT_SPEC.loader.exec_module(_RESIDENT)

_REPOSITORY_ROOT = Path(__file__).parents[1].resolve()


def _sha(value: str) -> str:
    return canonical_sha256(value)


def _publish(path: Path, value: object) -> CanonicalJsonProofBinding:
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


class _Runtime:
    def __init__(self) -> None:
        self.force_close_calls = 0

    async def force_close_active(self):
        self.force_close_calls += 1


def _production_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    member_count: int = 2,
):
    execution = _RESIDENT._execution(
        tmp_path / "resident", monkeypatch, member_count=member_count
    )
    node_path = (tmp_path / "node-materialization.json").resolve()
    _publish(node_path, {"kind": "test-node-materialization"})
    production_spec_path = (tmp_path / "production-spec.json").resolve()
    environment = formal_serving_session_group_production_environment(
        production_spec_path
    )
    commands: list[QueuedCommandSpec] = []
    worker_paths: list[Path] = []
    for index, member in enumerate(execution.plan.members):
        cell_root = (tmp_path / "cells" / str(index)).resolve()
        cell_root.mkdir(parents=True)
        worker = FormalCellWorkerSpec(
            schema_version=1,
            kind="formal_single_operator_cell_worker",
            cell_id=member.materialized_cell_id,
            attempt=member.attempt,
            repository_root=str(_REPOSITORY_ROOT),
            node_materialization_path=str(node_path),
            actual_result_path=str(cell_root / "actual.json"),
            evidence_root=str(cell_root),
            evidence_manifest_path=str(cell_root / "evidence-manifest.json"),
            job_argv=(str(Path(sys.executable).resolve()), "-c", "pass"),
            failure_class_on_nonzero="SCIENTIFIC",
            included_in_analysis_on_complete=True,
            complete_exclusion_reason=None,
        )
        worker_path = (cell_root / "worker-spec.json").resolve()
        publish_formal_cell_worker_spec(worker, worker_path)
        worker_paths.append(worker_path)
        controls = (tmp_path / "controls" / str(index)).resolve()
        command = QueuedCommandSpec(
            cell_id=member.materialized_cell_id,
            attempt=member.attempt,
            argv=(str(Path(sys.executable).resolve()), "-c", "pass"),
            launch_compatibility_key="resident-production-test",
            required_gpu_count=1,
            timing_class="HEADLINE",
            predicted_high_water_bytes=0,
            monitored_path=str(controls / "monitor"),
            log_path=str(controls / "wrapper.log"),
            expected_terminal_path=str(controls / "terminal.json"),
            expected_junit_path=str(controls / "junit.xml"),
            expected_raw_log_path=str(controls / "raw.json"),
            atomic_pointer_path=str(controls / "pointer.json"),
            child_exit_receipt_path=str(controls / "exit.json"),
            environment=environment,
        )
        commands.append(command)
    spec = build_formal_serving_session_group_production_spec(
        production_spec_path=production_spec_path,
        group_execution_spec_path=execution.spec_binding.absolute_path,
        cell_worker_spec_paths=worker_paths,
        commands=commands,
        nvidia_smi_tool=PinnedNvidiaSmiTool.bind(sys.executable),
        resident_evidence_root=(tmp_path / "resident-evidence").resolve(),
        shared_publication_path=(tmp_path / "shared-publication.json").resolve(),
        server_watch_target_path=(tmp_path / "server-watch-target.json").resolve(),
    )
    binding = publish_formal_serving_session_group_production_spec(spec=spec)
    return execution, binding, spec, tuple(commands), tuple(worker_paths)


def _fake_dependencies(
    *,
    production_spec_path: Path,
    worker_paths: tuple[Path, ...],
    close_gate: dict[str, bool],
    publish_close: bool = True,
):
    validation_calls: list[str] = []

    def actual_validator(**kwargs):
        assert close_gate["validated"]
        validation_calls.append(kwargs["cell_id"])
        return SimpleNamespace(
            status="COMPLETE",
            result_identity_sha256=_sha(f"actual:{kwargs['cell_id']}"),
            validator_kind="production_test_actual",
            validator_protocol_sha256=_sha("production-test-validator"),
        )

    async def execution_runner(*, execution_spec_path, runtime):
        del runtime
        execution = revalidate_formal_serving_session_group_execution(
            execution_spec_path
        )
        output = Path(execution.spec.output_directory)
        output.mkdir(parents=True)
        artifacts = []
        for index, (member, worker_path) in enumerate(
            zip(execution.plan.members, worker_paths, strict=True)
        ):
            worker, _digest = load_formal_cell_worker_spec(worker_path)
            actual = _publish(
                Path(worker.actual_result_path),
                {
                    "schema_version": 1,
                    "kind": "production-test-actual",
                    "cell_id": member.materialized_cell_id,
                },
            )
            reset = _publish(
                output / f"reset-{index}.json",
                {"schema_version": 1, "kind": "production-test-reset"},
            )
            artifact = FormalServingSessionGroupCellArtifact(
                schema_version=1,
                kind="formal_serving_session_group_cell_artifact",
                protocol_sha256=(
                    FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256
                ),
                group_id=execution.plan.group_id,
                materialized_cell_id=member.materialized_cell_id,
                attempt=member.attempt,
                member_index=index,
                execution_mode="shared_session_tp1",
                status="COMPLETE",
                process_id=9000,
                session_epoch=index + 1,
                reset_boundary=reset,
                result_pointer=actual,
                fallback_reason=None,
                failure_class=None,
                failure_code=None,
                started_ns=1_000_000 + index * 100,
                finished_ns=1_000_050 + index * 100,
                exit_code=None,
                evidence_level=("trusted_single_operator_empirical_no_signature"),
                formal_measured=False,
            )
            artifacts.append(
                _publish(
                    output / f"cell-{index}.json",
                    artifact.to_dict(),
                )
            )
        result = FormalServingSessionGroupExecutionResult(
            schema_version=1,
            kind="formal_serving_session_group_execution_result",
            protocol_sha256=FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
            group_id=execution.plan.group_id,
            status="COMPLETE",
            execution_spec=execution.spec_binding,
            group_plan=execution.plan_binding,
            reset_authority=execution.authority_binding,
            cell_artifacts=tuple(artifacts),
            shared_completed=len(artifacts),
            fresh_fallback_completed=0,
            failed=0,
            fallback_reason=None,
            fallback_evidence=None,
            evidence_level="trusted_single_operator_empirical_no_signature",
            formal_measured=False,
        )
        _publish(output / "result.json", result.to_dict())
        if publish_close:
            production = revalidate_formal_serving_session_group_production_spec(
                production_spec_path
            )
            _publish(
                Path(production.spec.shared_close_path),
                {"schema_version": 1, "kind": "production-test-close"},
            )
        return result

    def close_revalidator(path):
        binding = CanonicalJsonProofBinding.bind(path)
        production = revalidate_formal_serving_session_group_production_spec(
            production_spec_path
        )
        close_gate["validated"] = True
        return binding, SimpleNamespace(
            group_plan=production.execution.plan_binding,
            process_group_empty=True,
            member_trace_receipts=tuple(
                f"trace-{index}" for index in range(len(worker_paths))
            ),
        )

    async def watch_publisher(production):
        return _publish(
            Path(production.spec.server_watch_target_path),
            {"schema_version": 1, "kind": "production-test-watch-target"},
        )

    def watch_revalidator(path):
        return CanonicalJsonProofBinding.bind(path), SimpleNamespace()

    return (
        actual_validator,
        execution_runner,
        close_revalidator,
        watch_publisher,
        watch_revalidator,
        validation_calls,
    )


@pytest.mark.parametrize("member_count", (2, 32))
def test_production_group_closes_then_fans_out_one_shared_terminal_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_count: int,
) -> None:
    _execution, binding, spec, commands, worker_paths = _production_fixture(
        tmp_path, monkeypatch, member_count=member_count
    )
    close_gate = {"validated": False}
    (
        actual_validator,
        execution_runner,
        close_revalidator,
        watch_publisher,
        watch_revalidator,
        validation_calls,
    ) = _fake_dependencies(
        production_spec_path=Path(binding.absolute_path),
        worker_paths=worker_paths,
        close_gate=close_gate,
    )
    runtime = _Runtime()
    publication = asyncio.run(
        execute_formal_serving_session_group_production(
            binding.absolute_path,
            runtime=runtime,  # type: ignore[arg-type]
            execution_runner=execution_runner,
            actual_validator=actual_validator,
            close_revalidator=close_revalidator,
            watch_target_publisher=watch_publisher,
            watch_target_revalidator=watch_revalidator,
            heartbeat_interval_seconds=60.0,
        )
    )
    assert publication.formal_measured is False
    assert runtime.force_close_calls == 1
    assert close_gate["validated"]
    assert len(validation_calls) >= len(commands)
    assert Path(spec.shared_publication_path).is_file()
    assert all(Path(row.atomic_pointer_path).is_file() for row in spec.members)

    reopened = revalidate_formal_serving_session_group_production_publication(
        binding.absolute_path,
        actual_validator=actual_validator,
        close_revalidator=close_revalidator,
        watch_target_revalidator=watch_revalidator,
    )
    terminals = revalidate_formal_serving_session_group_production_terminals(
        binding.absolute_path,
        commands,
        actual_validator=actual_validator,
        close_revalidator=close_revalidator,
        watch_target_revalidator=watch_revalidator,
    )
    shared_sha = CanonicalJsonProofBinding.bind(spec.shared_publication_path).raw_sha256
    assert reopened.binding.raw_sha256 == shared_sha
    assert set(terminals) == {command.cell_id for command in commands}
    assert len(terminals) == member_count
    assert {terminal.atomic_publication_sha256 for terminal in terminals.values()} == {
        shared_sha
    }
    assert all(terminal.status == "COMPLETE" for terminal in terminals.values())
    assert formal_serving_session_group_production_spec_path_from_command(
        commands[0]
    ) == Path(binding.absolute_path)


def test_missing_close_never_publishes_member_controls_or_shared_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _execution, binding, spec, _commands, worker_paths = _production_fixture(
        tmp_path, monkeypatch
    )
    close_gate = {"validated": False}
    dependencies = _fake_dependencies(
        production_spec_path=Path(binding.absolute_path),
        worker_paths=worker_paths,
        close_gate=close_gate,
        publish_close=False,
    )
    runtime = _Runtime()
    with pytest.raises((FileNotFoundError, ValueError)):
        asyncio.run(
            execute_formal_serving_session_group_production(
                binding.absolute_path,
                runtime=runtime,  # type: ignore[arg-type]
                execution_runner=dependencies[1],
                actual_validator=dependencies[0],
                close_revalidator=dependencies[2],
                watch_target_publisher=dependencies[3],
                watch_target_revalidator=dependencies[4],
                heartbeat_interval_seconds=60.0,
            )
        )
    assert runtime.force_close_calls == 1
    assert not Path(spec.shared_publication_path).exists()
    assert all(not Path(row.expected_terminal_path).exists() for row in spec.members)
    assert all(not Path(row.atomic_pointer_path).exists() for row in spec.members)


def test_trace_sealed_is_only_a_heartbeat_phase_and_terminal_stays_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _execution, _binding, spec, commands, _worker_paths = _production_fixture(
        tmp_path, monkeypatch
    )
    clock = iter((100, 200, 300, 400))
    heartbeat = FormalServingSessionGroupChildHeartbeatPublisher(
        path=spec.heartbeat_path,
        cell_id=spec.members[0].cell_id,
        attempt=spec.members[0].attempt,
        command_sha256=spec.members[0].command_sha256,
        clock_ns=lambda: next(clock),
        interval_seconds=60.0,
    )
    heartbeat.start("RUNNING")
    heartbeat.set_phase("TRACE_SEALED")
    value = CanonicalJsonProofBinding.bind(spec.heartbeat_path).reopen()
    assert value["phase"] == "TRACE_SEALED"
    assert "status" not in value
    observation = ProcessObservation(
        pid=999,
        alive=False,
        observed_pgid=999,
        reason="test wrapper exited",
        exit_code=0,
    )
    assert (
        revalidate_formal_serving_session_group_production_terminal(
            commands[0], {}, observation
        )
        is None
    )
    assert not Path(spec.shared_publication_path).exists()
    heartbeat.stop("FINALIZING")


def test_production_spec_enforces_bounds_and_unoccupied_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _execution, binding, spec, _commands, _worker_paths = _production_fixture(
        tmp_path, monkeypatch
    )
    reopened = revalidate_formal_serving_session_group_production_spec(
        binding.absolute_path
    )
    assert len(reopened.spec.members) == 2
    with pytest.raises(ValueError, match="2-32"):
        replace(spec, members=(spec.members[0],))
    oversized = tuple(
        replace(
            spec.members[0],
            cell_id=_sha(f"oversized-cell-{index}"),
            attempt=index + 1,
            expected_terminal_path=str(
                (tmp_path / "oversized" / str(index) / "terminal.json").resolve()
            ),
            expected_junit_path=str(
                (tmp_path / "oversized" / str(index) / "junit.xml").resolve()
            ),
            expected_raw_log_path=str(
                (tmp_path / "oversized" / str(index) / "raw.json").resolve()
            ),
            atomic_pointer_path=str(
                (tmp_path / "oversized" / str(index) / "pointer.json").resolve()
            ),
        )
        for index in range(33)
    )
    with pytest.raises(ValueError, match="2-32"):
        replace(spec, members=oversized)
    ensure_formal_serving_session_group_production_outputs_unoccupied(
        binding.absolute_path
    )
    Path(spec.shared_publication_path).write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="occupied"):
        ensure_formal_serving_session_group_production_outputs_unoccupied(
            binding.absolute_path
        )
    assert dict(spec.members[0].__dict__)["command_sha256"] == _sha256_command(
        spec.members[0].command_sha256
    )


@pytest.mark.parametrize("member_count", (2, 32))
def test_shared_evidence_bound_is_group_scoped_and_protocol_bounded(
    member_count: int,
) -> None:
    assert (
        formal_serving_session_group_shared_evidence_bound_bytes(member_count)
        == 1024**3 + member_count * 64 * 1024**2
    )


def _sha256_command(value: str) -> str:
    assert len(value) == 64
    return value


def test_environment_is_path_only_and_has_no_digest_cycle(tmp_path: Path) -> None:
    path = (tmp_path / "future-production-spec.json").resolve()
    environment = formal_serving_session_group_production_environment(path)
    assert environment == (
        (FORMAL_SERVING_SESSION_GROUP_PRODUCTION_SPEC_ENV, str(path)),
    )
    assert all("SHA" not in name for name, _value in environment)
