from __future__ import annotations

import base64
import hashlib
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import lightcone_spec.runtime.preflight_runner as runner
import lightcone_spec.runtime.release_trust_root as root_module
from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.experiments import formal_preflight_execution
from lightcone_spec.experiments.formal_dispatch import (
    FormalPreflightExecutionBinding,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAssignment,
    registry_pool_work_item,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
    attestation_message,
)
from lightcone_spec.runtime.attester_bundle import (
    AttestationNoncePolicy,
    TrustedAttesterPolicyBundle,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayStore,
    ControlArtifactAttestation,
    ControlArtifactSubject,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
)
from lightcone_spec.runtime.preflight_runner import (
    PREFLIGHT_EXACTNESS_QUALIFICATION_PROTOCOL_SHA256,
    PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256,
    PREFLIGHT_EXACTNESS_TEST_NAMES,
    BurstGptShapeAuthority,
    BurstGptSourcePin,
    ExactnessControlVerificationReceipt,
    ExactnessLoaderEnvironment,
    ExactnessPreflightAssignment,
    ExactnessPreflightResultPointer,
    ExactnessPreflightTerminal,
    ExactnessQualificationProofArtifact,
    ExactnessRankTerminal,
    PreflightInputLocks,
)
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    SourceReleaseEd25519Root,
    SourceReleaseRootBinding,
    deployment_policy_subject_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _burst() -> BurstGptShapeAuthority:
    return BurstGptShapeAuthority(
        schema_version=1,
        kind="burstgpt_six_source_shape_authority",
        sources=tuple(
            BurstGptSourcePin(f"official-source-{index}", _sha(f"source-{index}"))
            for index in range(6)
        ),
        rows_sha256=_sha("rows"),
        row_count=128,
    )


def _locks() -> PreflightInputLocks:
    return PreflightInputLocks(
        prepared_model_set_sha256=_sha("prepared"),
        prepared_model_content_authority_sha256=_sha("prepared-content"),
        formal_workload_lock_sha256=_sha("workload"),
        burstgpt_shape_authority=_burst(),
    )


def _assignment(tmp_path: Path) -> ExactnessPreflightAssignment:
    checkout = tmp_path / "sglang"
    evidence = tmp_path / "evidence"
    checkout.mkdir()
    evidence.mkdir()
    python = tmp_path / "python"
    nvidia_smi = tmp_path / "nvidia-smi"
    binary_directory = tmp_path / "bin"
    library_directory = tmp_path / "lib"
    cuda_home = tmp_path / "cuda"
    binary_directory.mkdir()
    library_directory.mkdir()
    cuda_home.mkdir()
    python.write_bytes(b"python-executable")
    nvidia_smi.write_bytes(b"nvidia-smi-executable")
    return ExactnessPreflightAssignment(
        schema_version=3,
        kind="formal_exactness_preflight_assignment",
        protocol_sha256=PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256,
        registry_sha256=_sha("registry"),
        cell_id=_sha("cell"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        inventory_sha256=_sha("inventory"),
        hardware_envelope_sha256=_sha("hardware"),
        physical_assignment_sha256=_sha("physical-assignment"),
        experiment_budget_sha256=_sha("experiment-budget"),
        gpu_uuids=("GPU-0", "GPU-1"),
        gpu_model="RTX PRO 6000 Blackwell Server Edition",
        patched_sglang_checkout=str(checkout.resolve()),
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        python_executable=str(python.resolve()),
        python_raw_sha256=hashlib.sha256(python.read_bytes()).hexdigest(),
        nvidia_smi_executable=str(nvidia_smi.resolve()),
        nvidia_smi_raw_sha256=hashlib.sha256(nvidia_smi.read_bytes()).hexdigest(),
        python_version="3.12.11",
        torch_version="2.11.0+cu130",
        cuda_version="13.0",
        driver_version="580.65.06",
        input_locks=_locks(),
        loader_environment=ExactnessLoaderEnvironment(
            path_entries=(str(binary_directory.resolve()),),
            library_path_entries=(str(library_directory.resolve()),),
            cuda_home=str(cuda_home.resolve()),
        ),
        evidence_directory=str(evidence.resolve()),
    )


def test_sealed_preflight_exactness_assignment_binds_physical_budget_and_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = build_industrial_registry()
    cell = next(
        row
        for row in registry.cells_for("preflight")
        if row.identity.task == "exactness_memory_telemetry_preflight"
    )
    item = registry_pool_work_item(cell, estimated_duration_seconds=1.0)
    gpu_uuids = ("GPU-0", "GPU-1")
    assignment = GpuAssignment(
        work_item=item,
        gpu_uuids=gpu_uuids,
        rank_groups=(gpu_uuids,),
        ports=tuple(range(24_000, 24_000 + item.claim.port_count)),
    )
    locks = _locks()
    source_values = {
        "burstgpt_shape": locks.burstgpt_shape_authority.sha256,
        "compile_qualification": _sha("compile-qualification"),
        "exactness_qualification": _sha("exactness-qualification"),
        "formal_workload_e0": _sha("workload-e0"),
        "formal_workload_e3a": locks.formal_workload_lock_sha256,
        "native_runtime_qualification": _sha("native-qualification"),
        "offline_release_trust_root": _sha("release-root"),
        "prepared_model_content": (locks.prepared_model_content_authority_sha256),
    }
    binding = FormalPreflightExecutionBinding(
        materialized_cell_id=_sha("materialized-exactness"),
        registry_cell_id=cell.cell_id,
        runner_kind="first_party_exactness",
        work_item_sha256=item.sha256,
        assignment_sha256=assignment.sha256,
        experiment_budget_sha256=_sha("sealed-budget"),
        source_authority_bindings=tuple(sorted(source_values.items())),
        cell=cell,
        assignment=assignment,
        gpu_uuids=gpu_uuids,
        rank_groups=(gpu_uuids,),
    )
    runtime_sha256 = _sha("sealed-runtime")
    split_sha256 = _sha("sealed-split")
    inventory_sha256 = _sha("sealed-inventory")
    token = SimpleNamespace(
        subject=SimpleNamespace(
            execution_bindings=(binding,),
            inventory_sha256=inventory_sha256,
        ),
        dispatch_context=SimpleNamespace(registry=registry),
    )
    monkeypatch.setattr(
        formal_preflight_execution,
        "_verified",
        lambda _value: token,
    )
    monkeypatch.setattr(
        formal_preflight_execution,
        "_activation",
        lambda _value: SimpleNamespace(
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
        ),
    )
    monkeypatch.setattr(
        formal_preflight_execution,
        "_one_binding",
        lambda _value, _kind: binding,
    )
    exact = replace(
        _assignment(tmp_path),
        registry_sha256=registry.sha256,
        cell_id=cell.cell_id,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        inventory_sha256=inventory_sha256,
        physical_assignment_sha256=assignment.sha256,
        experiment_budget_sha256=binding.experiment_budget_sha256,
        gpu_uuids=gpu_uuids,
        input_locks=locks,
    )
    exact_path = exact.write((tmp_path / "sealed-exactness.json").resolve())
    assert (
        formal_preflight_execution.require_formal_preflight_exactness_assignment(
            token,
            assignment_path=exact_path,
        )
        == exact
    )

    wrong = replace(exact, physical_assignment_sha256=_sha("wrong-physical"))
    wrong_path = wrong.write((tmp_path / "wrong-exactness.json").resolve())
    with pytest.raises(ValueError, match="sealed preflight dispatch"):
        formal_preflight_execution.require_formal_preflight_exactness_assignment(
            token,
            assignment_path=wrong_path,
        )


def _public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _control_authority(
    monkeypatch: pytest.MonkeyPatch,
    assignment: ExactnessPreflightAssignment,
) -> tuple[
    Ed25519PrivateKey,
    SourceReleaseRootBinding,
    TrustedAttesterPolicyBundle,
    callable,
]:
    root_private = Ed25519PrivateKey.generate()
    controller_private = Ed25519PrivateKey.generate()
    root_public = _public_bytes(root_private)
    root_spki = root_private.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    root = SourceReleaseEd25519Root(
        schema_version=1,
        kind="lightcone_source_release_ed25519_root",
        root_id="lightcone-release-root-2026q3",
        key_id="lightcone-release-root-key-2026q3",
        algorithm="Ed25519",
        public_key_base64=base64.b64encode(root_public).decode("ascii"),
        public_key_sha256=hashlib.sha256(root_public).hexdigest(),
        spki_sha256=hashlib.sha256(root_spki).hexdigest(),
    )
    root_path = "/validation/release-root.json"
    binding = SourceReleaseRootBinding(
        root=root,
        path=root_path,
        sidecar_path=f"{root_path}.sha256",
        semantic_sha256=root.sha256,
        file_sha256=_sha("root-file"),
        sidecar_file_sha256=_sha("root-sidecar"),
    )
    controller_public = _public_bytes(controller_private)
    fingerprint = hashlib.sha256(controller_public).hexdigest()
    bundle = TrustedAttesterPolicyBundle(
        schema_version=1,
        kind="lightcone_trusted_attester_policy_bundle",
        bundle_id="preflight-exactness-test-bundle-v1",
        valid_from_ns=1,
        expires_ns=10_000_000_000,
        nonce_policy=AttestationNoncePolicy(
            schema_version=1,
            kind="lightcone_attestation_nonce_policy",
            nonce_bytes=32,
            minimum_lifetime_ns=100_000_000,
            maximum_lifetime_ns=2_000_000_000,
            maximum_clock_skew_ns=100_000_000,
            replay_policy="external_single_use_store",
            subject_binding_required=True,
        ),
        hardware_envelope_sha256_allowlist=(assignment.hardware_envelope_sha256,),
        trusted_attester_policy=TrustedAttesterPolicy(
            policy_id="preflight-exactness-test-policy-v1",
            trusted_attesters=(
                ("validation-signer", "validation-signer-key", fingerprint),
            ),
            public_keys=(
                (fingerprint, base64.b64encode(controller_public).decode("ascii")),
            ),
        ),
    )

    def authorize(label: str, nonce: bytes) -> DeploymentPolicyAuthorization:
        subject_sha256 = deployment_policy_subject_sha256(
            root_manifest_sha256=binding.semantic_sha256,
            inventory_sha256=assignment.inventory_sha256,
            bundle_sha256=bundle.sha256,
        )
        challenge = AttestationChallenge(
            schema_version=1,
            kind="lightcone_attestation_challenge",
            challenge_id=f"preflight-deployment-{label}",
            nonce_base64=base64.b64encode(nonce * 32).decode("ascii"),
            subject_sha256=subject_sha256,
            issued_ns=1_500_000_000,
            expires_ns=3_000_000_000,
        )
        signature = root_private.sign(
            attestation_message(challenge, payload_sha256=bundle.sha256)
        )
        return DeploymentPolicyAuthorization(
            schema_version=1,
            kind="lightcone_deployment_policy_authorization",
            root_manifest_sha256=binding.semantic_sha256,
            inventory_sha256=assignment.inventory_sha256,
            bundle=bundle,
            challenge=challenge,
            signature_base64=base64.b64encode(signature).decode("ascii"),
        )

    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: binding,
    )
    return controller_private, binding, bundle, authorize


def _control_envelope(
    *,
    private: Ed25519PrivateKey,
    binding: SourceReleaseRootBinding,
    bundle: TrustedAttesterPolicyBundle,
    authorization: DeploymentPolicyAuthorization,
    subject: ControlArtifactSubject,
    hardware_envelope_sha256: str,
    label: str,
    nonce: bytes,
) -> ControlArtifactAttestation:
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=f"preflight-control-{label}",
        nonce_base64=base64.b64encode(nonce * 32).decode("ascii"),
        subject_sha256=subject.sha256,
        issued_ns=1_600_000_000,
        expires_ns=2_600_000_000,
    )
    public = _public_bytes(private)
    signature = private.sign(
        attestation_message(challenge, payload_sha256=subject.artifact_sha256)
    )
    return ControlArtifactAttestation(
        schema_version=1,
        kind="lightcone_control_artifact_attestation",
        subject=subject,
        hardware_envelope_sha256=hardware_envelope_sha256,
        trust_anchor_sha256=binding.sha256,
        trust_bundle_sha256=bundle.sha256,
        trusted_attester_policy_sha256=bundle.trusted_attester_policy.sha256,
        deployment_policy_authorization=authorization,
        challenge=challenge,
        attestation=SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="validation-signer",
            key_id="validation-signer-key",
            environment="release",
            public_key_base64=base64.b64encode(public).decode("ascii"),
            challenge_sha256=challenge.sha256,
            payload_sha256=subject.artifact_sha256,
            signature_base64=base64.b64encode(signature).decode("ascii"),
        ),
    )


def _raw_schema3_pointer(
    tmp_path: Path,
    *,
    assignment: ExactnessPreflightAssignment,
    dispatch: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> Path:
    assignment_path = assignment.write((tmp_path / "assignment.json").resolve())
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    snapshot = {
        "schema_version": 2,
        "kind": "formal_exactness_gpu_snapshot",
        "assignment_sha256": assignment.sha256,
        "inventory_sha256": assignment.inventory_sha256,
        "status": "AVAILABLE",
        "gpu_rows": [
            {"uuid": gpu_uuid, "name": assignment.gpu_model, "memory_used_mib": 0}
            for gpu_uuid in assignment.gpu_uuids
        ],
        "compute_process_rows": [],
        "error_code": None,
        "captured_ns": 1,
    }
    runner._publish_json(before, snapshot)
    runner._publish_json(after, {**snapshot, "captured_ns": 2})
    log = tmp_path / "log.txt"
    log.write_text("exact qualification\n", encoding="utf-8")
    junit = tmp_path / "junit.xml"
    cases = "".join(
        f'<testcase classname="formal" name="{name}"/>'
        for name in PREFLIGHT_EXACTNESS_TEST_NAMES
    )
    junit.write_text(f'<testsuite tests="8">{cases}</testsuite>', encoding="utf-8")
    rank_paths = (tmp_path / "rank-0.json", tmp_path / "rank-1.json")
    ranks = tuple(
        ExactnessRankTerminal(
            schema_version=1,
            kind="formal_exactness_raw_rank_terminal",
            runner_protocol_sha256=PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256,
            assignment_sha256=assignment.sha256,
            global_rank=rank,
            gpu_uuid=assignment.gpu_uuids[rank],
            status="PASSED",
            started_ns=1,
            finished_ns=2,
            process_id=100 + rank,
            process_started_ns=1,
            completed_test_names=PREFLIGHT_EXACTNESS_TEST_NAMES,
            native_terminal_sha256=_sha(f"formal-native-terminal-{rank}"),
            observation_sha256=_sha(f"formal-observation-{rank}"),
        )
        for rank in range(2)
    )
    for path, rank in zip(rank_paths, ranks, strict=True):
        runner._publish_json(path, rank.to_dict())
    verified = verify_and_reserve_release_control_artifact_attestations(
        (dispatch,),
        expected_inventory_sha256=assignment.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified, reserved_ns=now_ns
    )
    reservation_path = (
        Path(replay_store.root) / f"reservation-{reservation_sha256}.json"
    )
    control_receipt = ExactnessControlVerificationReceipt(
        schema_version=1,
        kind="formal_exactness_control_verification_receipt",
        verified_ns=now_ns,
        assignment_sha256=assignment.sha256,
        inventory_sha256=assignment.inventory_sha256,
        hardware_envelope_sha256=assignment.hardware_envelope_sha256,
        control_envelope=dispatch,
        verified_control=verified[0],
        reservation_sha256=reservation_sha256,
        reservation_record=runner.EvidenceFileBinding.bind(
            reservation_path, label="exactness dispatch reservation"
        ),
    )
    control_path = tmp_path / "dispatch-control.json"
    runner._publish_json(control_path, control_receipt.to_dict())
    files = {
        "before": runner.EvidenceFileBinding.bind(before, label="before"),
        "after": runner.EvidenceFileBinding.bind(after, label="after"),
        "log": runner.EvidenceFileBinding.bind(log, label="log"),
        "junit": runner.EvidenceFileBinding.bind(junit, label="junit"),
    }
    terminal = ExactnessPreflightTerminal(
        schema_version=2,
        kind="formal_exactness_preflight_terminal",
        assignment_sha256=assignment.sha256,
        dispatch_attestation_sha256=dispatch.sha256,
        replay_reservation_sha256=reservation_sha256,
        status="PASSED",
        started_ns=1,
        finished_ns=2,
        process_exit_code=0,
        test_names=PREFLIGHT_EXACTNESS_TEST_NAMES,
        tests_collected=8,
        tests_passed=8,
        tests_failed=0,
        tests_errored=0,
        tests_skipped=0,
        terminal_rank_count=2,
        rank_terminal_sha256s=tuple(rank.sha256 for rank in ranks),
        before_gpu_snapshot_sha256=files["before"].raw_sha256,
        after_gpu_snapshot_sha256=files["after"].raw_sha256,
        before_gpu_snapshot_status="AVAILABLE",
        after_gpu_snapshot_status="AVAILABLE",
        before_compute_process_count=0,
        after_compute_process_count=0,
        process_group_cleanup="CLEAN",
        process_group_id=123,
        runner_error_code=None,
        junit_xml_sha256=files["junit"].raw_sha256,
        log_sha256=files["log"].raw_sha256,
    )
    terminal_path = tmp_path / "terminal.json"
    runner._publish_json(terminal_path, terminal.to_dict())
    raw_pointer = ExactnessPreflightResultPointer(
        schema_version=3,
        kind="formal_exactness_preflight_result_pointer",
        assignment=runner.EvidenceFileBinding.bind(assignment_path, label="assignment"),
        before_gpu_snapshot=files["before"],
        after_gpu_snapshot=files["after"],
        log=files["log"],
        terminal=runner.EvidenceFileBinding.bind(terminal_path, label="terminal"),
        rank_terminals=tuple(
            runner.EvidenceFileBinding.bind(path, label=f"rank {rank}")
            for rank, path in enumerate(rank_paths)
        ),
        junit_xml=files["junit"],
        control_verification_receipt=runner.EvidenceFileBinding.bind(
            control_path, label="dispatch control"
        ),
    )
    raw_path = tmp_path / "raw-result.json"
    runner._publish_json(raw_path, raw_pointer.to_dict())
    return raw_path.resolve()


def test_input_locks_require_six_official_sources() -> None:
    value = _burst()
    assert BurstGptShapeAuthority.from_dict(value.to_dict()) == value
    assert PreflightInputLocks.from_dict(_locks().to_dict()) == _locks()
    with pytest.raises(ValueError, match="six sorted official sources"):
        replace(value, sources=value.sources[:-1])


def test_assignment_is_path_bound_and_reopens_toolchain(tmp_path: Path) -> None:
    assignment = _assignment(tmp_path)
    path = assignment.write((tmp_path / "assignment.json").resolve())
    assert ExactnessPreflightAssignment.load(path) == assignment
    assert assignment.dispatch_lineage_sha256 != assignment.sha256
    Path(assignment.python_executable).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Python executable changed"):
        ExactnessPreflightAssignment.load(path)


def test_junit_requires_exact_eight_zero_skip(tmp_path: Path) -> None:
    good = tmp_path / "good.xml"
    cases = "".join(
        f'<testcase classname="formal" name="{name}"/>'
        for name in PREFLIGHT_EXACTNESS_TEST_NAMES
    )
    good.write_text(f'<testsuite tests="8">{cases}</testsuite>', encoding="utf-8")
    assert runner._junit_summary(good) == (
        PREFLIGHT_EXACTNESS_TEST_NAMES,
        8,
        8,
        0,
        0,
        0,
    )
    skipped = tmp_path / "skipped.xml"
    skipped.write_text(
        '<testsuite tests="1"><testcase classname="formal" '
        f'name="{PREFLIGHT_EXACTNESS_TEST_NAMES[0]}"><skipped/>'
        "</testcase></testsuite>",
        encoding="utf-8",
    )
    assert runner._junit_summary(skipped)[-1] == 1


def test_terminal_and_pointer_fail_closed_on_partial_or_tampered_evidence(
    tmp_path: Path,
) -> None:
    assignment = _assignment(tmp_path)
    assignment_path = assignment.write((tmp_path / "assignment.json").resolve())
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    snapshot = {
        "schema_version": 2,
        "kind": "formal_exactness_gpu_snapshot",
        "assignment_sha256": assignment.sha256,
        "inventory_sha256": assignment.inventory_sha256,
        "status": "AVAILABLE",
        "gpu_rows": [
            {
                "uuid": gpu_uuid,
                "name": assignment.gpu_model,
                "memory_used_mib": 0,
            }
            for gpu_uuid in assignment.gpu_uuids
        ],
        "compute_process_rows": [],
        "error_code": None,
        "captured_ns": 1,
    }
    runner._publish_json(before, snapshot)
    runner._publish_json(after, {**snapshot, "captured_ns": 2})
    log = tmp_path / "log"
    log.write_text("log\n", encoding="utf-8")
    junit = tmp_path / "junit.xml"
    cases = "".join(
        f'<testcase classname="formal" name="{name}"/>'
        for name in PREFLIGHT_EXACTNESS_TEST_NAMES
    )
    junit.write_text(f'<testsuite tests="8">{cases}</testsuite>', encoding="utf-8")
    rank_paths = (tmp_path / "rank-0.json", tmp_path / "rank-1.json")
    ranks = tuple(
        ExactnessRankTerminal(
            schema_version=1,
            kind="formal_exactness_raw_rank_terminal",
            runner_protocol_sha256=PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256,
            assignment_sha256=assignment.sha256,
            global_rank=rank,
            gpu_uuid=assignment.gpu_uuids[rank],
            status="PASSED",
            started_ns=1,
            finished_ns=2,
            process_id=100 + rank,
            process_started_ns=1,
            completed_test_names=PREFLIGHT_EXACTNESS_TEST_NAMES,
            native_terminal_sha256=_sha(f"native-rank-{rank}"),
            observation_sha256=_sha(f"observation-rank-{rank}"),
        )
        for rank in range(2)
    )
    for path, rank in zip(rank_paths, ranks, strict=True):
        runner._publish_json(path, rank.to_dict())
    files = {
        "before": runner.EvidenceFileBinding.bind(before, label="before"),
        "after": runner.EvidenceFileBinding.bind(after, label="after"),
        "log": runner.EvidenceFileBinding.bind(log, label="log"),
        "junit": runner.EvidenceFileBinding.bind(junit, label="junit"),
    }
    terminal = ExactnessPreflightTerminal(
        schema_version=2,
        kind="formal_exactness_preflight_terminal",
        assignment_sha256=assignment.sha256,
        dispatch_attestation_sha256=_sha("dispatch"),
        replay_reservation_sha256=_sha("reservation"),
        status="PASSED",
        started_ns=1,
        finished_ns=2,
        process_exit_code=0,
        test_names=PREFLIGHT_EXACTNESS_TEST_NAMES,
        tests_collected=8,
        tests_passed=8,
        tests_failed=0,
        tests_errored=0,
        tests_skipped=0,
        terminal_rank_count=2,
        rank_terminal_sha256s=tuple(rank.sha256 for rank in ranks),
        before_gpu_snapshot_sha256=files["before"].raw_sha256,
        after_gpu_snapshot_sha256=files["after"].raw_sha256,
        before_gpu_snapshot_status="AVAILABLE",
        after_gpu_snapshot_status="AVAILABLE",
        before_compute_process_count=0,
        after_compute_process_count=0,
        process_group_cleanup="CLEAN",
        process_group_id=123,
        runner_error_code=None,
        junit_xml_sha256=files["junit"].raw_sha256,
        log_sha256=files["log"].raw_sha256,
    )
    with pytest.raises(ValueError, match="status differs"):
        replace(terminal, tests_skipped=1)
    terminal_path = tmp_path / "terminal.json"
    runner._publish_json(terminal_path, terminal.to_dict())
    pointer = ExactnessPreflightResultPointer(
        schema_version=2,
        kind="formal_exactness_preflight_result_pointer",
        assignment=runner.EvidenceFileBinding.bind(assignment_path, label="assignment"),
        before_gpu_snapshot=files["before"],
        after_gpu_snapshot=files["after"],
        log=files["log"],
        terminal=runner.EvidenceFileBinding.bind(terminal_path, label="terminal"),
        rank_terminals=tuple(
            runner.EvidenceFileBinding.bind(path, label=f"rank {rank}")
            for rank, path in enumerate(rank_paths)
        ),
        junit_xml=files["junit"],
    )
    pointer_path = tmp_path / "pointer.json"
    runner._publish_json(pointer_path, pointer.to_dict())
    assert ExactnessPreflightResultPointer.load(pointer_path) == pointer
    log.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        ExactnessPreflightResultPointer.load(pointer_path)


def test_pointer_rejects_synthetic_rank_count_without_raw_rank_file(
    tmp_path: Path,
) -> None:
    assignment = _assignment(tmp_path)
    with pytest.raises(ValueError, match="rank aggregate"):
        ExactnessPreflightTerminal(
            schema_version=2,
            kind="formal_exactness_preflight_terminal",
            assignment_sha256=assignment.sha256,
            dispatch_attestation_sha256=_sha("dispatch"),
            replay_reservation_sha256=_sha("reservation"),
            status="ERROR",
            started_ns=1,
            finished_ns=2,
            process_exit_code=1,
            test_names=(),
            tests_collected=0,
            tests_passed=0,
            tests_failed=0,
            tests_errored=1,
            tests_skipped=0,
            terminal_rank_count=2,
            rank_terminal_sha256s=(),
            before_gpu_snapshot_sha256=_sha("before"),
            after_gpu_snapshot_sha256=_sha("after"),
            before_gpu_snapshot_status="ERROR",
            after_gpu_snapshot_status="ERROR",
            before_compute_process_count=0,
            after_compute_process_count=0,
            process_group_cleanup="ERROR",
            process_group_id=None,
            runner_error_code="missing_raw_rank_terminal",
            junit_xml_sha256=None,
            log_sha256=_sha("log"),
        )


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups required")
def test_timeout_terminates_entire_qualification_process_group(
    tmp_path: Path,
) -> None:
    log = tmp_path / "process.log"
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        outcome = runner._run_logged_process_group(
            (
                sys.executable,
                "-c",
                (
                    "import subprocess,sys,time;"
                    "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                    "time.sleep(60)"
                ),
            ),
            cwd=str(tmp_path),
            environment={"PATH": os.environ.get("PATH", "")},
            log_fd=descriptor,
            timeout_seconds=0.1,
        )
    finally:
        os.close(descriptor)
    assert outcome.error_code == "qualification_process_timeout"
    assert outcome.cleanup in {"TERM_CLEAN", "KILL_CLEAN"}
    assert outcome.process_group_id is not None
    assert not runner._process_group_exists(outcome.process_group_id)


def test_post_run_rank_aggregate_qualifies_schema4_and_rejects_cross_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_ns = 2_000_000_000
    assignment = _assignment(tmp_path)
    private, binding, bundle, authorize = _control_authority(monkeypatch, assignment)
    dispatch_subject = ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=assignment.sha256,
        protocol_sha256=PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256,
        registry_sha256=assignment.registry_sha256,
        lineage_sha256=assignment.dispatch_lineage_sha256,
    )
    dispatch = _control_envelope(
        private=private,
        binding=binding,
        bundle=bundle,
        authorization=authorize("dispatch", b"d"),
        subject=dispatch_subject,
        hardware_envelope_sha256=assignment.hardware_envelope_sha256,
        label="dispatch",
        nonce=b"1",
    )
    replay_root = tmp_path / "replay"
    replay_root.mkdir(mode=0o700)
    replay_store = ChallengeReplayStore(str(replay_root.resolve()))
    raw_path = _raw_schema3_pointer(
        tmp_path,
        assignment=assignment,
        dispatch=dispatch,
        replay_store=replay_store,
        now_ns=now_ns,
    )
    payload = runner.derive_exactness_qualification_payload(raw_path)
    aggregate_subject = ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="rank_aggregate",
        artifact_sha256=payload.sha256,
        protocol_sha256=PREFLIGHT_EXACTNESS_QUALIFICATION_PROTOCOL_SHA256,
        registry_sha256=payload.registry_sha256,
        lineage_sha256=payload.lineage_sha256,
    )
    aggregate = _control_envelope(
        private=private,
        binding=binding,
        bundle=bundle,
        authorization=authorize("aggregate", b"a"),
        subject=aggregate_subject,
        hardware_envelope_sha256=assignment.hardware_envelope_sha256,
        label="aggregate",
        nonce=b"2",
    )
    proof_path = (tmp_path / "qualification-proof.json").resolve()
    proof = runner.publish_release_exactness_qualification_proof(
        raw_path,
        control_attestation=aggregate,
        replay_store=replay_store,
        now_ns=now_ns,
        proof_artifact_path=proof_path,
    )
    assert isinstance(proof, ExactnessQualificationProofArtifact)
    proof.revalidate(now_ns=now_ns + 20_000_000_000)
    qualified_path = (tmp_path / "qualified-result.json").resolve()
    qualified = runner.finalize_release_exactness_preflight(
        raw_path,
        proof_artifact_path=proof_path,
        qualified_result_pointer_path=qualified_path,
        now_ns=now_ns + 20_000_000_000,
    )
    assert qualified.schema_version == 4
    assert qualified.qualification_proof_artifact is not None
    assert ExactnessPreflightResultPointer.load(qualified_path) == qualified

    wrong_suite_subject = replace(
        aggregate_subject,
        protocol_sha256=_sha("native-hot-path-tp1-suite"),
    )
    wrong_suite = _control_envelope(
        private=private,
        binding=binding,
        bundle=bundle,
        authorization=authorize("wrong-suite", b"w"),
        subject=wrong_suite_subject,
        hardware_envelope_sha256=assignment.hardware_envelope_sha256,
        label="wrong-suite",
        nonce=b"3",
    )
    with pytest.raises(ValueError, match="control differs"):
        runner.publish_release_exactness_qualification_proof(
            raw_path,
            control_attestation=wrong_suite,
            replay_store=replay_store,
            now_ns=now_ns,
            proof_artifact_path=(tmp_path / "wrong-suite.json").resolve(),
        )
    with pytest.raises(ValueError, match="fields differ|schema"):
        ExactnessQualificationProofArtifact.from_dict(
            {
                "schema_version": 1,
                "kind": "lightcone_native_runtime_gpu_proof_artifact",
            }
        )
    with pytest.raises(ValueError, match="already consumed"):
        runner.publish_release_exactness_qualification_proof(
            raw_path,
            control_attestation=aggregate,
            replay_store=replay_store,
            now_ns=now_ns,
            proof_artifact_path=(tmp_path / "replayed-proof.json").resolve(),
        )

    Path(qualified.rank_terminals[0].absolute_path).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        ExactnessPreflightResultPointer.load(qualified_path)


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups required")
def test_exactness_runner_executes_fake_child_and_reduces_exact_eight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real subprocess/snapshot/JUnit/rank-terminal reducer on CPU."""

    assignment = _assignment(tmp_path)
    python_path = Path(assignment.python_executable)
    nvidia_smi_path = Path(assignment.nvidia_smi_executable)
    python_path.write_text(
        f"""#!{sys.executable}
import hashlib
import json
import os
import sys

def publish(path, value):
    canonical = json.dumps(value, sort_keys=True, separators=(\",\", \":\"), allow_nan=False)
    with open(path, \"x\", encoding=\"utf-8\") as handle:
        handle.write(canonical + \"\\n\")
    semantic = hashlib.sha256(canonical.encode(\"utf-8\")).hexdigest()
    with open(path + \".sha256\", \"x\", encoding=\"ascii\") as handle:
        handle.write(semantic + \"\\n\")

names = sorted(arg.rsplit(\"::\", 1)[1] for arg in sys.argv if \"::\" in arg)
junit_path = next(arg.split(\"=\", 1)[1] for arg in sys.argv if arg.startswith(\"--junitxml=\"))
cases = \"\".join(f'<testcase classname=\"formal\" name=\"{{name}}\"/>' for name in names)
with open(junit_path, \"x\", encoding=\"utf-8\") as handle:
    handle.write(f'<testsuite tests=\"8\">{{cases}}</testsuite>')
for rank in (0, 1):
    publish(
        os.environ[f\"LIGHTCONE_PREFLIGHT_RANK{{rank}}_TERMINAL_PATH\"],
        {{
            \"schema_version\": 1,
            \"kind\": \"formal_exactness_raw_rank_terminal\",
            \"runner_protocol_sha256\": os.environ[\"LIGHTCONE_PREFLIGHT_RUNNER_PROTOCOL_SHA256\"],
            \"assignment_sha256\": os.environ[\"LIGHTCONE_PREFLIGHT_ASSIGNMENT_SHA256\"],
            \"global_rank\": rank,
            \"gpu_uuid\": f\"GPU-{{rank}}\",
            \"status\": \"PASSED\",
            \"started_ns\": 1,
            \"finished_ns\": 2,
            \"process_id\": 4000 + rank,
            \"process_started_ns\": 1,
            \"completed_test_names\": names,
            \"native_terminal_sha256\": hashlib.sha256(f\"native-{{rank}}\".encode()).hexdigest(),
            \"observation_sha256\": hashlib.sha256(f\"observation-{{rank}}\".encode()).hexdigest(),
        }},
    )
print(\"fake exactness child completed\")
""",
        encoding="utf-8",
    )
    nvidia_smi_path.write_text(
        """#!/bin/sh
case "$1" in
  --query-gpu=*)
    printf '%s\n' 'GPU-0, RTX PRO 6000 Blackwell Server Edition, 0' 'GPU-1, RTX PRO 6000 Blackwell Server Edition, 0'
    ;;
  --query-compute-apps=*)
    ;;
  *)
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    python_path.chmod(0o700)
    nvidia_smi_path.chmod(0o700)
    assignment = replace(
        assignment,
        python_raw_sha256=hashlib.sha256(python_path.read_bytes()).hexdigest(),
        nvidia_smi_raw_sha256=hashlib.sha256(nvidia_smi_path.read_bytes()).hexdigest(),
    )
    now_ns = 2_000_000_000
    private, binding, bundle, authorize = _control_authority(
        monkeypatch,
        assignment,
    )
    subject = ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=assignment.sha256,
        protocol_sha256=PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256,
        registry_sha256=assignment.registry_sha256,
        lineage_sha256=assignment.dispatch_lineage_sha256,
    )
    dispatch = _control_envelope(
        private=private,
        binding=binding,
        bundle=bundle,
        authorization=authorize("fake-child-dispatch", b"f"),
        subject=subject,
        hardware_envelope_sha256=assignment.hardware_envelope_sha256,
        label="fake-child-dispatch",
        nonce=b"4",
    )
    replay_root = tmp_path / "fake-child-replay"
    replay_root.mkdir(mode=0o700)
    assignment_path = assignment.write(
        (tmp_path / "fake-child-assignment.json").resolve()
    )

    pointer = runner.execute_release_exactness_preflight(
        assignment_path,
        dispatch_attestation=dispatch,
        replay_store=ChallengeReplayStore(str(replay_root.resolve())),
        now_ns=now_ns,
        timeout_seconds=10.0,
    )

    terminal = ExactnessPreflightTerminal.load(pointer.terminal.absolute_path)
    assert pointer.schema_version == 3
    assert terminal.status == "PASSED"
    assert (
        terminal.tests_collected,
        terminal.tests_passed,
        terminal.tests_failed,
        terminal.tests_errored,
        terminal.tests_skipped,
        terminal.terminal_rank_count,
    ) == (8, 8, 0, 0, 0, 2)
    assert terminal.before_compute_process_count == 0
    assert terminal.after_compute_process_count == 0
    assert terminal.process_group_cleanup == "CLEAN"
    runner.derive_exactness_qualification_payload(
        Path(assignment.evidence_directory)
        / f"exactness-{assignment.sha256}.result.json"
    )

    # The trusted single-operator branch runs the same real child and raw
    # reducers, but publishes a diagnostic pointer bound to its current-source
    # marker instead of reserving signed control/replay state.
    trusted_evidence = (tmp_path / "trusted-exactness-evidence").resolve()
    trusted_evidence.mkdir(mode=0o700)
    trusted_assignment = replace(
        assignment,
        evidence_directory=str(trusted_evidence),
    )
    trusted_assignment_path = trusted_assignment.write(
        (tmp_path / "trusted-exactness-assignment.json").resolve()
    )
    trusted_authority = _sha("trusted-exactness-authority")
    trusted_pointer = runner._execute_exactness_preflight(
        trusted_assignment_path,
        dispatch_attestation=None,
        replay_store=None,
        now_ns=now_ns,
        timeout_seconds=10.0,
        single_operator_authority_sha256=trusted_authority,
    )
    trusted_terminal = ExactnessPreflightTerminal.load(
        trusted_pointer.terminal.absolute_path
    )
    assert trusted_pointer.schema_version == 2
    assert trusted_pointer.control_verification_receipt is None
    assert trusted_terminal.status == "PASSED"
    assert trusted_terminal.dispatch_attestation_sha256 == trusted_authority
    assert trusted_terminal.replay_reservation_sha256 == runner._sha(
        {
            "schema_version": 1,
            "kind": "formal_single_operator_exactness_execution_marker",
            "execution_authority_sha256": trusted_authority,
            "assignment_sha256": trusted_assignment.sha256,
        }
    )


def test_local_exactness_qualification_consumes_only_stable_pull_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = object()
    raw_path = "/validation/pulled/raw-exactness.json"
    raw = SimpleNamespace(
        schema_version=3,
        assignment=SimpleNamespace(absolute_path="/validation/assignment.json"),
    )
    assignment = SimpleNamespace(sha256=_sha("stable-pull-assignment"))
    qualified = object()
    published: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        formal_preflight_execution,
        "_verified",
        lambda value: token if value is token else None,
    )
    monkeypatch.setattr(
        formal_preflight_execution,
        "_load_formal_preflight_remote_raw_evidence_receipt",
        lambda dispatch, path: (
            object(),
            SimpleNamespace(exactness_result=SimpleNamespace(absolute_path=raw_path)),
        ),
    )
    monkeypatch.setattr(
        formal_preflight_execution.ExactnessPreflightResultPointer,
        "load",
        classmethod(lambda cls, path: raw),
    )
    monkeypatch.setattr(
        formal_preflight_execution.ExactnessPreflightAssignment,
        "load",
        classmethod(lambda cls, path: assignment),
    )
    monkeypatch.setattr(
        formal_preflight_execution,
        "require_formal_preflight_exactness_assignment",
        lambda dispatch, assignment_path: assignment,
    )
    monkeypatch.setattr(
        formal_preflight_execution,
        "publish_release_exactness_qualification_proof",
        lambda path, **kwargs: published.append((path, kwargs)),
    )
    monkeypatch.setattr(
        formal_preflight_execution,
        "finalize_release_exactness_preflight",
        lambda path, **kwargs: qualified,
    )

    result = formal_preflight_execution.qualify_formal_preflight_exactness_locally(
        token,
        remote_raw_receipt_path="/validation/pulled/raw-receipt.json",
        rank_aggregate_control_attestation=object(),
        replay_store=object(),
        now_ns=1,
        proof_artifact_path="/validation/pulled/proof.json",
        qualified_result_pointer_path="/validation/pulled/qualified.json",
    )

    assert result is qualified
    assert len(published) == 1
    assert published[0][0] == raw_path
