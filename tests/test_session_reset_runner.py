from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.runtime import session_reset_runner as runner_module
from lightcone_spec.runtime.readiness import NATIVE_RUNTIME_QUALIFICATION_TESTS
from lightcone_spec.runtime.session_reset_runner import (
    SESSION_RESET_GPU_TEST_FILE,
    SESSION_RESET_GPU_TEST_NAMES,
    SESSION_RESET_RUNNER_PROTOCOL_SHA256,
    SessionResetRawRankTerminal,
)


def _sha(character: str) -> str:
    return character * 64


def _terminal() -> SessionResetRawRankTerminal:
    return SessionResetRawRankTerminal(
        schema_version=1,
        kind="formal_session_reset_raw_rank_terminal",
        runner_protocol_sha256=SESSION_RESET_RUNNER_PROTOCOL_SHA256,
        assignment_sha256=_sha("a"),
        global_rank=0,
        gpu_uuid="GPU-qualification-0",
        status="PASSED",
        started_ns=10,
        finished_ns=20,
        process_id=101,
        completed_test_names=SESSION_RESET_GPU_TEST_NAMES,
        native_terminal_sha256=_sha("b"),
        reset_receipt_sha256=_sha("c"),
        close_receipt_sha256=_sha("d"),
        cold_output_ids_sha256=_sha("e"),
        reused_output_ids_sha256=_sha("e"),
        observation_sha256=_sha("f"),
    )


def test_session_reset_suite_is_exact_non_skippable_live_server_contract() -> None:
    assert (
        "timeout_seconds"
        not in inspect.signature(
            runner_module.execute_session_reset_qualification
        ).parameters
    )
    assert SESSION_RESET_GPU_TEST_NAMES == tuple(
        f"test_{name}"
        for name in NATIVE_RUNTIME_QUALIFICATION_TESTS["session_reset_tp1"]
    )
    source = Path(
        "patches/sglang/0007-feat-spec-bind-distributed-runtime-readiness.patch"
    ).read_text(encoding="utf-8")
    assert SESSION_RESET_GPU_TEST_FILE in source
    file_marker = (
        f"diff --git a/{SESSION_RESET_GPU_TEST_FILE} b/{SESSION_RESET_GPU_TEST_FILE}"
    )
    file_section = source.split(file_marker, maxsplit=1)[1].split(
        "\ndiff --git ", maxsplit=1
    )[0]
    added = "\n".join(
        line[1:] for line in file_section.splitlines() if line.startswith("+")
    )
    for name in SESSION_RESET_GPU_TEST_NAMES:
        assert f"def {name}()" in added
    for endpoint in (
        "/generate",
        "/v1/lightcone-spec/session-reset/capability",
        "/v1/lightcone-spec/session-reset/initial-state",
        "/v1/lightcone-spec/session-reset",
        "/v1/lightcone-spec/session-reset/trace/begin",
        "/v1/lightcone-spec/session-reset/trace/reset",
        "/v1/lightcone-spec/session-reset/trace/finalize",
        "/v1/lightcone-spec/session-reset/close-terminal",
    ):
        assert endpoint in added
    assert "pytest.skip" not in added
    assert "@pytest.mark.skip" not in added
    assert "torch.cuda.device_count() != 1" in added
    assert "formal_session_reset_raw_rank_terminal" in added
    assert "LIGHTCONE_SESSION_RANK0_TERMINAL_PATH" in added


def test_session_reset_raw_rank_terminal_rejects_partial_or_foreign_claims() -> None:
    terminal = _terminal()
    assert SessionResetRawRankTerminal.from_dict(terminal.to_dict()) == terminal
    with pytest.raises(ValueError, match="incomplete"):
        replace(
            terminal,
            completed_test_names=SESSION_RESET_GPU_TEST_NAMES[:-1],
        )
    with pytest.raises(ValueError, match="invalid"):
        replace(terminal, reused_output_ids_sha256=_sha("0"))
    with pytest.raises(ValueError, match="invalid"):
        replace(terminal, process_id=0)


def test_session_reset_source_protocol_binds_exact_identity_and_external_trust() -> (
    None
):
    source = Path("src/lightcone_spec/runtime/session_reset_runner.py").read_text(
        encoding="utf-8"
    )
    for required in (
        "CompileLaunchManifest.load",
        "run_config.method != self.method",
        'run_config.model.algorithm != "DFLASH"',
        "authorize_session_reset_dispatch",
        "verify_and_reserve_release_control_artifact_attestations",
        "NativeRuntimeGpuProofReceipt",
        "verify_native_runtime_gpu_proof",
        "build_native_runtime_gpu_proof_artifact",
        "SessionResetQualificationProofPointer",
        "SessionResetQualificationFailurePointer",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "before_gpu_snapshot",
        "after_gpu_snapshot",
        "formal_session_reset_qualification_fatal_terminal",
        "_stop_process_group",
    ):
        assert required in source
    assert len(SESSION_RESET_RUNNER_PROTOCOL_SHA256) == 64


def test_session_reset_gpu_snapshot_rejects_foreign_or_residual_processes() -> None:
    assignment = SimpleNamespace(
        sha256=_sha("a"),
        inventory_sha256=_sha("b"),
        gpu_uuid="GPU-qualification-0",
        gpu_model="RTX PRO 6000 Blackwell Server Edition",
    )
    snapshot = {
        "schema_version": 1,
        "kind": "formal_session_reset_gpu_snapshot",
        "assignment_sha256": assignment.sha256,
        "inventory_sha256": assignment.inventory_sha256,
        "captured_ns": 10,
        "status": "AVAILABLE",
        "gpu": {
            "uuid": assignment.gpu_uuid,
            "name": assignment.gpu_model,
            "memory_used_mib": 0,
        },
        "compute_process_rows": [],
        "error_code": None,
    }
    assert (
        runner_module._validate_gpu_snapshot(snapshot, assignment=assignment)
        == snapshot
    )
    with pytest.raises(ValueError, match="assigned GPU"):
        runner_module._validate_gpu_snapshot(
            {
                **snapshot,
                "gpu": {**snapshot["gpu"], "uuid": "GPU-foreign"},
            },
            assignment=assignment,
        )
    with pytest.raises(ValueError, match="compute-process"):
        runner_module._validate_gpu_snapshot(
            {
                **snapshot,
                "compute_process_rows": [
                    {
                        "gpu_uuid": assignment.gpu_uuid,
                        "pid": 0,
                        "used_gpu_memory_mib": 1,
                    }
                ],
            },
            assignment=assignment,
        )
