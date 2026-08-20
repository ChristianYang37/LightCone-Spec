from __future__ import annotations

import base64
from itertools import pairwise
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_control_attestation import (
    HARDWARE_SHA256,
    INVENTORY_SHA256,
    NOW_NS,
    _bundle,
    _public_bytes,
    _root_binding,
)
from test_formal_physical_dispatch import _run_tp1_operator_fixture

import lightcone_spec.runtime.release_trust_root as root_module
from lightcone_spec.experiments.itl_authority import (
    STAGE_ITL_TIMESTAMP_PROOF_PROTOCOL_SHA256,
    StageItlExecutionIdentity,
    build_stage_itl_external_control_binding,
    publish_stage_itl_timestamp_proof_artifact,
    publish_stage_itl_timestamp_raw_receipt,
    validate_stage_itl_timestamp_proof_artifact,
)
from lightcone_spec.orchestration.formal_single_operator_admission import (
    FormalSingleOperatorAdmission,
)
from lightcone_spec.orchestration.formal_terminal_result import (
    build_formal_tp1_terminal_control_subject,
    publish_formal_tp1_terminal_result_proof_artifact,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    attestation_message,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayStore,
    ControlArtifactAttestation,
    ControlArtifactSubject,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256,
    NATIVE_RUNTIME_QUALIFICATION_TESTS,
    NATIVE_RUNTIME_RELEASE_CAPABILITY,
    NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
    NativeRuntimeGpuProofReceipt,
    build_native_runtime_gpu_proof_artifact,
    verify_native_runtime_gpu_proof,
)
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    deployment_policy_subject_sha256,
)

REGISTRY_SHA256 = "e" * 64


def _authorization(
    *,
    root_private: Ed25519PrivateKey,
    root_binding,
    bundle,
    suffix: str,
    nonce_byte: bytes,
    inventory_sha256: str = INVENTORY_SHA256,
) -> DeploymentPolicyAuthorization:
    subject_sha256 = deployment_policy_subject_sha256(
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=inventory_sha256,
        bundle_sha256=bundle.sha256,
    )
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=f"stage-itl-deployment-{suffix}",
        nonce_base64=base64.b64encode(nonce_byte * 32).decode("ascii"),
        subject_sha256=subject_sha256,
        issued_ns=1_500_000_000,
        expires_ns=3_000_000_000,
    )
    return DeploymentPolicyAuthorization(
        schema_version=1,
        kind="lightcone_deployment_policy_authorization",
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=inventory_sha256,
        bundle=bundle,
        challenge=challenge,
        signature_base64=base64.b64encode(
            root_private.sign(
                attestation_message(challenge, payload_sha256=bundle.sha256)
            )
        ).decode("ascii"),
    )


def _control(
    *,
    artifact_private: Ed25519PrivateKey,
    root_binding,
    bundle,
    authorization: DeploymentPolicyAuthorization,
    subject: ControlArtifactSubject,
    suffix: str,
    nonce_byte: bytes,
) -> ControlArtifactAttestation:
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=f"stage-itl-control-{suffix}",
        nonce_base64=base64.b64encode(nonce_byte * 32).decode("ascii"),
        subject_sha256=subject.sha256,
        issued_ns=1_600_000_000,
        expires_ns=2_600_000_000,
    )
    payload = subject.artifact_sha256
    return ControlArtifactAttestation(
        schema_version=1,
        kind="lightcone_control_artifact_attestation",
        subject=subject,
        hardware_envelope_sha256=HARDWARE_SHA256,
        trust_anchor_sha256=root_binding.sha256,
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
            public_key_base64=base64.b64encode(_public_bytes(artifact_private)).decode(
                "ascii"
            ),
            challenge_sha256=challenge.sha256,
            payload_sha256=payload,
            signature_base64=base64.b64encode(
                artifact_private.sign(
                    attestation_message(challenge, payload_sha256=payload)
                )
            ).decode("ascii"),
        ),
    )


def _subject(
    *,
    artifact_sha256: str,
    protocol_sha256: str,
    registry_sha256: str,
    lineage_sha256: str,
) -> ControlArtifactSubject:
    return ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=artifact_sha256,
        protocol_sha256=protocol_sha256,
        registry_sha256=registry_sha256,
        lineage_sha256=lineage_sha256,
    )


def test_stage_itl_proof_deep_reopens_terminal_gpu_timing_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_private = Ed25519PrivateKey.generate()
    artifact_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    bundle = _bundle(artifact_private)
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    replay_root = (tmp_path / "replay").resolve()
    replay_root.mkdir(mode=0o700)
    replay_store = ChallengeReplayStore(str(replay_root))

    plan, plan_path, _verified, _inventory_path, admission_binding, _live = (
        _run_tp1_operator_fixture(monkeypatch, tmp_path)
    )
    admission = FormalSingleOperatorAdmission.from_dict(admission_binding.reopen())
    inventory_sha256 = plan.inventory_sha256
    registry_sha256 = admission.registry_sha256
    gpu_uuids = plan.gpu_uuids
    run_binding = plan.native_terminal_binding
    terminal_subject = build_formal_tp1_terminal_control_subject(
        plan_path=str(plan_path),
        expected_inventory_sha256=inventory_sha256,
        expected_registry_sha256=registry_sha256,
    )
    terminal_control = _control(
        artifact_private=artifact_private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=_authorization(
            root_private=root_private,
            root_binding=root_binding,
            bundle=bundle,
            suffix="terminal",
            nonce_byte=b"1",
            inventory_sha256=inventory_sha256,
        ),
        subject=terminal_subject,
        suffix="terminal",
        nonce_byte=b"2",
    )
    result_proof_path = (tmp_path / "result-proof.json").resolve()
    publish_formal_tp1_terminal_result_proof_artifact(
        plan_path=str(plan_path),
        control_attestation=terminal_control,
        replay_store=replay_store,
        expected_inventory_sha256=inventory_sha256,
        expected_registry_sha256=registry_sha256,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        now_ns=NOW_NS,
        proof_artifact_path=str(result_proof_path),
    )

    capability = NATIVE_RUNTIME_RELEASE_CAPABILITY
    gpu_receipt = NativeRuntimeGpuProofReceipt(
        schema_version=1,
        kind="lightcone_native_runtime_gpu_proof",
        suite_id="native_hot_path_tp1",
        topology_mode="tp1_dp1",
        topology_sha256="4" * 64,
        runner_protocol_sha256=NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[
            "native_hot_path_tp1"
        ],
        assignment_sha256="8" * 64,
        qualification_observation_sha256="9" * 64,
        source_capability_sha256=capability.sha256,
        pinned_sglang_commit=capability.pinned_sglang_commit,
        patched_sglang_tree=capability.patched_sglang_tree,
        semantic_patch_sha256=capability.semantic_patch_sha256,
        run_nonce_sha256="5" * 64,
        qualification_authority_sha256=(NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256),
        source_identity_sha256="6" * 64,
        inventory_sha256=inventory_sha256,
        gpu_uuids=gpu_uuids,
        hardware_envelope_sha256=HARDWARE_SHA256,
        junit_xml_sha256="7" * 64,
        test_names=NATIVE_RUNTIME_QUALIFICATION_TESTS["native_hot_path_tp1"],
        tests_collected=8,
        tests_passed=8,
        tests_failed=0,
        tests_errored=0,
        tests_skipped=0,
    )
    gpu_receipt_path = (tmp_path / "gpu-receipt.json").resolve()
    gpu_receipt_binding = gpu_receipt.write_unsigned(str(gpu_receipt_path))
    gpu_control = _control(
        artifact_private=artifact_private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=_authorization(
            root_private=root_private,
            root_binding=root_binding,
            bundle=bundle,
            suffix="gpu",
            nonce_byte=b"3",
            inventory_sha256=inventory_sha256,
        ),
        subject=_subject(
            artifact_sha256=gpu_receipt_binding.raw_sha256,
            protocol_sha256=capability.suite_protocol_sha256,
            registry_sha256=gpu_receipt.source_identity_sha256,
            lineage_sha256=gpu_receipt.control_lineage_sha256,
        ),
        suffix="gpu",
        nonce_byte=b"4",
    )
    verified_gpu = verify_native_runtime_gpu_proof(
        str(gpu_receipt_path),
        control_attestation=gpu_control,
        replay_store=replay_store,
        expected_suite_id="native_hot_path_tp1",
        expected_topology_sha256=gpu_receipt.topology_sha256,
        expected_source_identity_sha256=gpu_receipt.source_identity_sha256,
        expected_inventory_sha256=inventory_sha256,
        expected_gpu_uuids=gpu_uuids,
        expected_hardware_envelope_sha256=HARDWARE_SHA256,
        expected_run_nonce_sha256=gpu_receipt.run_nonce_sha256,
        now_ns=NOW_NS,
    )
    gpu_artifact = build_native_runtime_gpu_proof_artifact(
        receipt_path=str(gpu_receipt_path),
        control_attestation=gpu_control,
        replay_store=replay_store,
        verified_proof=verified_gpu,
    )
    gpu_proof_path = (tmp_path / "gpu-proof.json").resolve()
    publish_canonical_json_no_replace(str(gpu_proof_path), gpu_artifact.to_dict())
    assert (
        CanonicalJsonProofBinding.bind(str(gpu_proof_path)).semantic_sha256
        == gpu_artifact.sha256
    )

    execution = StageItlExecutionIdentity(
        schema_version=1,
        kind="stage_itl_execution_identity",
        materialized_cell_id=plan.materialized_cell_id,
        inventory_sha256=inventory_sha256,
        registry_sha256=registry_sha256,
        execution_plan_sha256=run_binding.execution_plan_sha256,
        rank_config_sha256=run_binding.rank_config_sha256,
        run_id=run_binding.run_id,
        run_nonce_sha256=run_binding.run_nonce_sha256,
        attempt_id=run_binding.attempt_id,
        method=run_binding.method,
        runtime_trust_mode=run_binding.runtime_trust_mode,
        formal_measurement=run_binding.formal_measurement,
    )
    pointer_bundle = CanonicalJsonProofBinding.bind(
        plan.native_itl_pointer_output_path
    ).reopen()
    pointer = pointer_bundle["native_result_pointers"][0]
    raw_itl_path = (tmp_path / "raw-itl.json").resolve()
    publish_stage_itl_timestamp_raw_receipt(
        str(raw_itl_path),
        native_result_proof_path=str(result_proof_path),
        native_gpu_proof_path=str(gpu_proof_path),
        execution_identity=execution,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        native_result_pointers=(pointer,),
        now_ns=NOW_NS,
    )
    control_binding = build_stage_itl_external_control_binding(
        str(raw_itl_path),
        native_result_proof_path=str(result_proof_path),
        native_gpu_proof_path=str(gpu_proof_path),
        execution_identity=execution,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        now_ns=NOW_NS,
    )
    timing_control = _control(
        artifact_private=artifact_private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=_authorization(
            root_private=root_private,
            root_binding=root_binding,
            bundle=bundle,
            suffix="timing",
            nonce_byte=b"6",
            inventory_sha256=inventory_sha256,
        ),
        subject=_subject(
            artifact_sha256=control_binding.sha256,
            protocol_sha256=STAGE_ITL_TIMESTAMP_PROOF_PROTOCOL_SHA256,
            registry_sha256=registry_sha256,
            lineage_sha256=control_binding.lineage_sha256,
        ),
        suffix="timing",
        nonce_byte=b"7",
    )
    timing_proof_path = (tmp_path / "timing-proof.json").resolve()
    publish_stage_itl_timestamp_proof_artifact(
        str(raw_itl_path),
        native_result_proof_path=str(result_proof_path),
        native_gpu_proof_path=str(gpu_proof_path),
        execution_identity=execution,
        control_attestation=timing_control,
        replay_store=replay_store,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        now_ns=NOW_NS,
        proof_artifact_path=str(timing_proof_path),
    )
    reservation_files = tuple(replay_root.glob("reservation-*.json"))
    authority = validate_stage_itl_timestamp_proof_artifact(
        str(timing_proof_path),
        expected_execution_identity=execution,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        now_ns=NOW_NS + 1,
    )
    event_times = [row["observed_ns"] for row in pointer["events"]]
    token_ids = [row["token_id"] for row in pointer["events"]]
    inter_token = [later - earlier for earlier, later in pairwise(event_times)]
    assert authority.client_timing_inputs == (
        {
            "request_id": pointer["request_id"],
            "arrival_ns": pointer["request_started_ns"],
            "first_token_ns": event_times[0],
            "completion_ns": pointer["request_terminal_ns"],
            "output_token_ids": token_ids,
            "native_per_token_observed_ns": event_times,
            "inter_token_ns": inter_token,
        },
    )
    assert authority.throughput_numerator_tokens == len(token_ids)
    assert authority.throughput_window_ns == (
        pointer["request_terminal_ns"] - pointer["request_started_ns"]
    )
    assert authority.p99_itl_input_ns == tuple(inter_token)
    assert tuple(replay_root.glob("reservation-*.json")) == reservation_files

    forged = StageItlExecutionIdentity(
        **{
            **execution.to_dict(),
            "materialized_cell_id": "9" * 64,
        }
    )
    with pytest.raises(ValueError, match="file identity differs"):
        validate_stage_itl_timestamp_proof_artifact(
            str(timing_proof_path),
            expected_execution_identity=forged,
            expected_root_manifest_sha256=root_binding.semantic_sha256,
            now_ns=NOW_NS + 1,
        )
