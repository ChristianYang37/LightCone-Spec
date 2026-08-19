from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import lightcone_spec.runtime.native_qualification_runner as runner
from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.runtime.native_qualification_runner import (
    NATIVE_RUNTIME_GPU_TEST_FILES,
    NATIVE_RUNTIME_GPU_TEST_NAMES,
    NativeRuntimeQualificationAssignment,
    NativeRuntimeQualificationError,
    NativeRuntimeQualificationFailurePointer,
    NativeRuntimeQualificationObservation,
    NativeRuntimeQualificationResultPointer,
    execute_native_runtime_qualification,
)
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.sglang_bridge import launch as launch_module


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _binding(path: Path, value: object) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _file_binding(path: Path, body: bytes) -> EvidenceFileBinding:
    path.write_bytes(body)
    path.chmod(0o700)
    return EvidenceFileBinding.bind(path, label=path.name)


def _resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    suite_id: runner.SourceOwnedRuntimeQualificationSuite = "native_hot_path_tp1",
) -> tuple[
    NativeRuntimeQualificationAssignment,
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    SimpleNamespace,
]:
    checkout = (tmp_path / "patched-sglang").resolve()
    test_file = checkout / NATIVE_RUNTIME_GPU_TEST_FILES[suite_id]
    test_file.parent.mkdir(parents=True)
    test_file.write_text("# source-owned qualification fixture\n", encoding="utf-8")
    evidence = (tmp_path / "evidence").resolve()
    evidence.mkdir(mode=0o700)
    launch_binding = _binding(
        evidence / "launch.json",
        {"schema_version": 1, "kind": "test_compile_launch"},
    )
    python = _file_binding((tmp_path / "python").resolve(), b"#!/bin/sh\nexit 1\n")
    nvidia_smi = _file_binding(
        (tmp_path / "nvidia-smi").resolve(), b"#!/bin/sh\nexit 1\n"
    )
    expected_tp, expected_dp, expected_gpu_count = runner._expected_topology(suite_id)
    gpu_uuids = tuple(f"GPU-{index}" for index in range(expected_gpu_count))
    gpu_models = tuple("Test GPU" for _index in range(expected_gpu_count))
    launch = SimpleNamespace(
        sha256=launch_binding.semantic_sha256,
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        patched_sglang_checkout=str(checkout),
        run_config_path=str((tmp_path / "run-config.json").resolve()),
        run_config_semantic_sha256=_sha("run-config"),
        prepared_model_content_manifest_sha256=_sha("prepared-content"),
        inventory_sha256=_sha("inventory"),
        gpu_uuids=gpu_uuids,
        child_environment=lambda: {
            "PATH": "/usr/bin",
            "LD_LIBRARY_PATH": "/usr/lib",
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    run_config = SimpleNamespace(
        model=SimpleNamespace(algorithm=runner._SUITE_ALGORITHM[suite_id]),
        adaptation=(
            SimpleNamespace(
                eagle3_qualification_compatibility_authority_sha256=_sha(
                    "eagle3-compatibility"
                ),
                eagle3_qualification_model_selector_sha256=_sha("eagle3-selector"),
                eagle3_e0_execution_authority_sha256=None,
                eagle3_compatibility_authority_sha256=None,
                eagle3_model_selector_sha256=None,
                eagle3_native_gpu_proof_sha256=None,
            )
            if suite_id == "eagle3_tp1"
            else None
        ),
        runtime=SimpleNamespace(
            tensor_parallel_size=expected_tp,
            data_parallel_size=expected_dp,
            speculation_enabled=True,
        ),
    )
    monkeypatch.setattr(runner.CompileLaunchManifest, "load", lambda _path: launch)
    monkeypatch.setattr(runner, "load_run_config", lambda _path: run_config)
    base_binding = None
    if suite_id in runner.DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS:
        base_assignment_path = (evidence / "base-assignment.json").resolve()
        base_assignment_path.write_text("{}\n", encoding="utf-8")
        base_assignment_file = EvidenceFileBinding.bind(
            base_assignment_path, label="base assignment"
        )
        base_junit_path = (evidence / "base-junit.xml").resolve()
        base_junit_path.write_bytes(b"<testsuite/>\n")
        base_junit = EvidenceFileBinding.bind(base_junit_path, label="base JUnit")
        base_binding = _binding(
            evidence / "base-exactness-result.json",
            {"schema_version": 4, "kind": "test_base_exactness"},
        )
        base_pointer = SimpleNamespace(
            schema_version=4,
            sha256=base_binding.semantic_sha256,
            assignment=base_assignment_file,
            junit_xml=base_junit,
            qualification_proof_artifact=object(),
        )
        base_assignment = SimpleNamespace(
            registry_sha256=_sha("registry"),
            runtime_sha256=_sha("runtime"),
            inventory_sha256=launch.inventory_sha256,
            hardware_envelope_sha256=_sha("hardware"),
            gpu_uuids=gpu_uuids,
        )
        monkeypatch.setattr(
            runner.ExactnessPreflightResultPointer,
            "load",
            lambda _path: base_pointer,
        )
        monkeypatch.setattr(
            runner.ExactnessPreflightAssignment,
            "load",
            lambda _path: base_assignment,
        )
    assignment = NativeRuntimeQualificationAssignment(
        schema_version=1,
        kind="formal_native_runtime_gpu_qualification_assignment",
        suite_id=suite_id,
        runner_protocol_sha256=runner._runner_protocol_sha256(suite_id),
        registry_sha256=_sha("registry"),
        runtime_sha256=_sha("runtime"),
        topology_sha256=_sha("topology"),
        inventory_sha256=launch.inventory_sha256,
        hardware_envelope_sha256=_sha("hardware"),
        run_nonce_sha256=_sha("run-nonce"),
        gpu_uuids=gpu_uuids,
        gpu_models=gpu_models,
        launch_manifest=launch_binding,
        base_exactness_result_pointer=base_binding,
        eagle3_selector_status=("COMPATIBLE" if suite_id == "eagle3_tp1" else None),
        eagle3_compatibility_authority_sha256=(
            _sha("eagle3-compatibility") if suite_id == "eagle3_tp1" else None
        ),
        eagle3_model_selector_sha256=(
            _sha("eagle3-selector") if suite_id == "eagle3_tp1" else None
        ),
        python_executable=python.absolute_path,
        python_executable_raw_sha256=python.raw_sha256,
        python_executable_size=python.size,
        nvidia_smi_executable=nvidia_smi.absolute_path,
        nvidia_smi_raw_sha256=nvidia_smi.raw_sha256,
        nvidia_smi_size=nvidia_smi.size,
        evidence_directory=str(evidence),
    )
    assignment_binding = assignment.write(evidence / "assignment.json")
    dispatch_binding = _binding(
        evidence / "dispatch.json",
        {"schema_version": 1, "kind": "test_dispatch"},
    )

    class Dispatch:
        def revalidate(self, *, assignment):
            assert assignment.sha256 == assignment_binding.semantic_sha256
            return object()

    dispatch = Dispatch()
    monkeypatch.setattr(
        runner.NativeRuntimeQualificationDispatchAuthority,
        "from_dict",
        classmethod(lambda _cls, _value: dispatch),
    )
    return assignment, assignment_binding, dispatch_binding, launch


def _snapshot(
    assignment: NativeRuntimeQualificationAssignment,
    *,
    phase: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "formal_native_runtime_gpu_qualification_snapshot",
        "runner_protocol_sha256": assignment.runner_protocol_sha256,
        "assignment_sha256": assignment.sha256,
        "inventory_sha256": assignment.inventory_sha256,
        "phase": phase,
        "captured_ns": 100 if phase == "before" else 300,
        "gpu_uuids": list(assignment.gpu_uuids),
        "status": "AVAILABLE",
        "gpu_rows": [
            {"uuid": uuid, "name": model, "memory_used_mib": 1}
            for uuid, model in zip(
                assignment.gpu_uuids, assignment.gpu_models, strict=True
            )
        ],
        "compute_process_rows": [],
        "error_code": None,
    }


def _junit(names: tuple[str, ...]) -> bytes:
    cases = "".join(f'<testcase name="{name}"/>' for name in names)
    return f'<testsuite tests="8">{cases}</testsuite>'.encode()


def _live_observation(
    assignment: NativeRuntimeQualificationAssignment,
    launch: SimpleNamespace,
) -> NativeRuntimeQualificationObservation:
    return NativeRuntimeQualificationObservation(
        schema_version=1,
        kind="source_owned_native_runtime_live_observation",
        suite_id=assignment.suite_id,
        runner_protocol_sha256=assignment.runner_protocol_sha256,
        assignment_sha256=assignment.sha256,
        source_capability_sha256=runner._source_capability_sha256(assignment.suite_id),
        launch_manifest_sha256=assignment.launch_manifest.semantic_sha256,
        prepared_model_content_manifest_sha256=(
            launch.prepared_model_content_manifest_sha256
        ),
        run_config_sha256=launch.run_config_semantic_sha256,
        inventory_sha256=assignment.inventory_sha256,
        gpu_uuids=assignment.gpu_uuids,
        completed_test_names=NATIVE_RUNTIME_GPU_TEST_NAMES[assignment.suite_id],
        server_process_ids=tuple(
            990 + index for index in range(len(assignment.gpu_uuids))
        ),
        rank_terminal_sha256s=tuple(
            _sha(f"rank-{index}") for index in range(len(assignment.gpu_uuids))
        ),
        live_server_receipt_sha256=_sha("live-server"),
        native_terminal_sha256=_sha("native-terminal"),
        native_itl_pointer_sha256=_sha("native-itl"),
        graph_observation_sha256=_sha("graph"),
        worker_hook_observation_sha256=_sha("worker-hook"),
        scored_request_inputs_sha256=_sha("requests"),
        completed_request_count=2,
        worker_hook_invocation_count=2,
        graph_replay_count=2,
        native_timestamp_count=4,
        started_ns=100,
        finished_ns=200,
        actual_sglang_server=True,
        component_only=False,
    )


def test_source_owned_native_hot_path_runner_publishes_exact_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assignment, assignment_binding, dispatch_binding, launch = _resources(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(runner, "_gpu_snapshot", _snapshot)
    monkeypatch.setattr(runner, "_process_group_exists", lambda _pid: False)
    observed: dict[str, object] = {}

    class Process:
        pid = 48123
        returncode = 0

        def __init__(self, command, **kwargs):
            observed["command"] = command
            observed["environment"] = kwargs["env"]
            self.environment = kwargs["env"]

        def wait(self, *, timeout):
            assert 1 <= timeout <= 3_600
            Path(self.environment["LIGHTCONE_NATIVE_QUALIFICATION_OBSERVATION_PATH"])
            observation = _live_observation(assignment, launch)
            publish_canonical_json_no_replace(
                self.environment["LIGHTCONE_NATIVE_QUALIFICATION_OBSERVATION_PATH"],
                observation.to_dict(),
            )
            junit_flag = next(
                value
                for value in observed["command"]
                if value.startswith("--junitxml=")
            )
            Path(junit_flag.split("=", 1)[1]).write_bytes(
                _junit(NATIVE_RUNTIME_GPU_TEST_NAMES[assignment.suite_id])
            )
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    result_binding = execute_native_runtime_qualification(
        assignment_binding.absolute_path,
        dispatch_binding.absolute_path,
    )
    result = NativeRuntimeQualificationResultPointer.load(result_binding.absolute_path)
    receipt = result.unsigned_gpu_proof.reopen()
    assert receipt["suite_id"] == "native_hot_path_tp1"
    assert receipt["assignment_sha256"] == assignment.sha256
    assert receipt["qualification_observation_sha256"] == (
        result.live_observation.semantic_sha256
    )
    command = observed["command"]
    assert command[:4] == (assignment.python_executable, "-m", "pytest", "-q")
    assert (
        tuple(value.split("::")[-1] for value in command[4:12])
        == (NATIVE_RUNTIME_GPU_TEST_NAMES[assignment.suite_id])
    )
    assert (
        "callback"
        not in inspect.signature(execute_native_runtime_qualification).parameters
    )
    assert (
        "timeout_seconds"
        not in inspect.signature(execute_native_runtime_qualification).parameters
    )
    environment = observed["environment"]
    assert environment["LIGHTCONE_NATIVE_QUALIFICATION_SOURCE_CAPABILITY_SHA256"] == (
        runner._source_capability_sha256(assignment.suite_id)
    )
    assert environment["LIGHTCONE_NATIVE_QUALIFICATION_DISPATCH_PATH"] == (
        dispatch_binding.absolute_path
    )
    assert environment["LIGHTCONE_NATIVE_QUALIFICATION_DISPATCH_SHA256"] == (
        dispatch_binding.semantic_sha256
    )


def test_source_owned_dp2_runner_binds_separate_distributed_artifact_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assignment, assignment_binding, dispatch_binding, launch = _resources(
        monkeypatch, tmp_path, suite_id="tp1_dp2"
    )
    monkeypatch.setattr(runner, "_gpu_snapshot", _snapshot)
    monkeypatch.setattr(runner, "_process_group_exists", lambda _pid: False)

    class Process:
        pid = 48125
        returncode = 0

        def __init__(self, command, **kwargs):
            self.command = command
            self.environment = kwargs["env"]

        def wait(self, *, timeout):
            assert timeout == runner._SOURCE_OWNED_QUALIFICATION_TIMEOUT_SECONDS
            publish_canonical_json_no_replace(
                self.environment["LIGHTCONE_NATIVE_QUALIFICATION_OBSERVATION_PATH"],
                _live_observation(assignment, launch).to_dict(),
            )
            junit_flag = next(
                value for value in self.command if value.startswith("--junitxml=")
            )
            Path(junit_flag.split("=", 1)[1]).write_bytes(
                _junit(NATIVE_RUNTIME_GPU_TEST_NAMES[assignment.suite_id])
            )
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    result = NativeRuntimeQualificationResultPointer.load(
        execute_native_runtime_qualification(
            assignment_binding.absolute_path,
            dispatch_binding.absolute_path,
        ).absolute_path
    )
    receipt = result.unsigned_gpu_proof.reopen()
    assert receipt["kind"] == "lightcone_distributed_runtime_gpu_proof"
    assert receipt["topology_mode"] == "tp1_dp2"
    assert receipt["assignment_sha256"] == assignment.sha256
    assert assignment.base_exactness_result_pointer is not None
    assert receipt["base_exactness_result_pointer_sha256"] == (
        assignment.base_exactness_result_pointer.semantic_sha256
    )
    assert receipt["qualification_junit_xml_sha256"] == result.junit_xml.raw_sha256


def test_missing_live_observation_publishes_non_authorizing_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assignment, assignment_binding, dispatch_binding, _launch = _resources(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(runner, "_gpu_snapshot", _snapshot)
    monkeypatch.setattr(runner, "_process_group_exists", lambda _pid: False)

    class Process:
        pid = 48124
        returncode = 0

        def __init__(self, command, **_kwargs):
            self.command = command

        def wait(self, *, timeout):
            assert timeout == runner._SOURCE_OWNED_QUALIFICATION_TIMEOUT_SECONDS
            junit_flag = next(
                value for value in self.command if value.startswith("--junitxml=")
            )
            Path(junit_flag.split("=", 1)[1]).write_bytes(
                _junit(NATIVE_RUNTIME_GPU_TEST_NAMES[assignment.suite_id])
            )
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    with pytest.raises(
        NativeRuntimeQualificationError,
        match="qualification_evidence_validation_failed",
    ) as raised:
        execute_native_runtime_qualification(
            assignment_binding.absolute_path,
            dispatch_binding.absolute_path,
        )
    failure = NativeRuntimeQualificationFailurePointer.load(
        raised.value.failure_pointer.absolute_path
    )
    assert failure.fatal_terminal.reopen()["status"] == "ERROR"
    assert not assignment.evidence_path("unsigned-proof.json").exists()


def test_suite_algorithm_and_topology_are_source_owned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assignment, _assignment_binding, _dispatch_binding, _launch = _resources(
        monkeypatch, tmp_path
    )
    with pytest.raises(ValueError, match="schema is unsupported"):
        replace(assignment, suite_id="session_reset_tp1")


def test_eagle3_assignment_requires_compatible_official_selector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assignment, _assignment_binding, _dispatch_binding, _launch = _resources(
        monkeypatch, tmp_path, suite_id="eagle3_tp1"
    )
    assert assignment.eagle3_selector_status == "COMPATIBLE"
    with pytest.raises(ValueError, match="COMPATIBLE official selector"):
        replace(assignment, eagle3_selector_status=None)


def test_non_eagle_assignment_rejects_selector_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assignment, _assignment_binding, _dispatch_binding, _launch = _resources(
        monkeypatch, tmp_path
    )
    with pytest.raises(ValueError, match="restricted to eagle3_tp1"):
        replace(
            assignment,
            eagle3_selector_status="COMPATIBLE",
            eagle3_compatibility_authority_sha256=_sha("foreign-compatibility"),
            eagle3_model_selector_sha256=_sha("foreign-selector"),
        )


def test_ordinary_launch_never_installs_qualification_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LIGHTCONE_NATIVE_QUALIFICATION_MODE", raising=False)
    launch_module._install_qualification_runtime_bridge(
        python_root="/path/that/must/not/be-opened",
        raw_config={},
    )


def test_qualification_bridge_requires_dispatch_bound_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIGHTCONE_NATIVE_QUALIFICATION_MODE", "1")
    monkeypatch.delenv("LIGHTCONE_NATIVE_QUALIFICATION_ASSIGNMENT_PATH", raising=False)
    monkeypatch.delenv("LIGHTCONE_NATIVE_QUALIFICATION_DISPATCH_PATH", raising=False)
    with pytest.raises(RuntimeError, match="dispatch-bound inputs"):
        launch_module._install_qualification_runtime_bridge(
            python_root="/path/that/must/not/be-opened",
            raw_config={},
        )


def test_qualification_bridge_rejects_foreign_environment_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assignment = _binding(tmp_path / "assignment.json", {"assignment": 1})
    dispatch = _binding(tmp_path / "dispatch.json", {"dispatch": 1})
    monkeypatch.setenv("LIGHTCONE_NATIVE_QUALIFICATION_MODE", "1")
    monkeypatch.setenv(
        "LIGHTCONE_NATIVE_QUALIFICATION_ASSIGNMENT_PATH", assignment.absolute_path
    )
    monkeypatch.setenv(
        "LIGHTCONE_NATIVE_QUALIFICATION_DISPATCH_PATH", dispatch.absolute_path
    )
    monkeypatch.setenv("LIGHTCONE_NATIVE_QUALIFICATION_ASSIGNMENT_SHA256", "0" * 64)
    monkeypatch.setenv(
        "LIGHTCONE_NATIVE_QUALIFICATION_DISPATCH_SHA256",
        dispatch.semantic_sha256,
    )
    with pytest.raises(ValueError, match="environment identity differs"):
        launch_module._install_qualification_runtime_bridge(
            python_root=str(tmp_path),
            raw_config={},
        )
