from __future__ import annotations

import base64
from dataclasses import replace

import pytest
import torch
from test_control_attestation import (
    HARDWARE_SHA256,
    INVENTORY_SHA256,
    NOW_NS,
    _public_bytes,
)
from test_control_attestation import (
    _authority as _control_authority,
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
from lightcone_spec.runtime.proof_artifact import publish_canonical_json_no_replace
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256,
    NATIVE_RUNTIME_QUALIFICATION_TESTS,
    NATIVE_RUNTIME_RELEASE_CAPABILITY,
    NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
    GraphFixedAddressContract,
    GraphHotPathStateMachine,
    GraphTensorBinding,
    NativeItlBufferBinding,
    NativeItlTimestampStateMachine,
    NativeReadinessBlocked,
    NativeRuntimeGpuProofArtifact,
    NativeRuntimeGpuProofReceipt,
    NativeTokenTimestampEvent,
    build_native_runtime_gpu_proof_artifact,
    require_fixed_address_graph_gpu_proof,
    validate_native_runtime_gpu_proof_artifact,
    verify_native_runtime_gpu_proof,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _itl_binding() -> NativeItlBufferBinding:
    return NativeItlBufferBinding(
        schema_version=1,
        run_sha256=SHA_A,
        producer_sha256=SHA_B,
        clock_identity="monotonic_ns",
        producer_stream_id="decode-stream-0",
        buffer_pointer=4096,
        buffer_capacity_tokens=8,
        buffer_generation=2,
    )


def _event(
    binding: NativeItlBufferBinding,
    index: int,
    token_id: int,
    observed_ns: int,
) -> NativeTokenTimestampEvent:
    return NativeTokenTimestampEvent(
        binding_sha256=binding.sha256,
        request_id="request-0",
        token_index=index,
        token_id=token_id,
        observed_ns=observed_ns,
        buffer_pointer=binding.buffer_pointer,
        buffer_generation=binding.buffer_generation,
    )


def test_native_itl_pointer_state_machine_requires_order_and_full_coverage() -> None:
    binding = _itl_binding()
    recorder = NativeItlTimestampStateMachine(
        binding,
        request_id="request-0",
        output_token_ids=(7, 8, 9),
        request_started_ns=100,
    )
    recorder.record(_event(binding, 0, 7, 110))
    recorder.record(_event(binding, 1, 8, 120))
    with pytest.raises(ValueError, match="full per-token coverage"):
        recorder.finalize(request_terminal_ns=130)
    recorder.record(_event(binding, 2, 9, 125))
    pointer = recorder.finalize(request_terminal_ns=130)
    assert pointer.token_observed_ns == (110, 120, 125)
    assert pointer.evidence_level == "CPU_CONTRACT_ONLY"
    assert not pointer.formal_authorized
    with pytest.raises(
        NativeReadinessBlocked,
        match="native_itl_gpu_pointer_proof_unavailable",
    ):
        recorder.require_formal_authority()
    self_described = replace(
        binding,
        native_patch_capability_sha256=SHA_C,
        gpu_proof_sha256=SHA_C,
    )
    with pytest.raises(NativeReadinessBlocked):
        NativeItlTimestampStateMachine(
            self_described,
            request_id="request-0",
            output_token_ids=(7, 8),
            request_started_ns=100,
        ).require_formal_authority()


def test_native_itl_pointer_drift_and_reordered_tokens_fail_closed() -> None:
    binding = _itl_binding()
    recorder = NativeItlTimestampStateMachine(
        binding,
        request_id="request-0",
        output_token_ids=(7, 8),
        request_started_ns=100,
    )
    with pytest.raises(ValueError, match="ordered token identity"):
        recorder.record(_event(binding, 1, 8, 110))
    with pytest.raises(ValueError, match="pointer changed"):
        recorder.record(replace(_event(binding, 0, 7, 110), buffer_pointer=8192))


def _graph_contract(tensor: torch.Tensor) -> GraphFixedAddressContract:
    return GraphFixedAddressContract(
        schema_version=1,
        graph_identity_sha256=SHA_C,
        capture_generation=1,
        tensors=(GraphTensorBinding.from_tensor("weight", tensor),),
    )


def test_graph_state_machine_preserves_address_and_rejects_host_operations() -> None:
    tensor = torch.zeros(4)
    tensors = (("weight", tensor),)
    machine = GraphHotPathStateMachine(_graph_contract(tensor))
    machine.observe("cuda_event_record", tensors=tensors)
    machine.observe("device_to_device_copy", tensors=tensors)
    with pytest.raises(NativeReadinessBlocked, match="blocking_d2h"):
        machine.observe("blocking_d2h", tensors=tensors)
    machine.observe("graph_replay", tensors=tensors)
    receipt = machine.finalize()
    assert receipt.evidence_level == "CPU_CONTRACT_ONLY"
    assert not receipt.formal_authorized
    with pytest.raises(
        NativeReadinessBlocked,
        match="graph_hot_path_gpu_proof_unavailable",
    ):
        machine.require_formal_authority()
    with pytest.raises(NativeReadinessBlocked):
        GraphHotPathStateMachine(
            replace(_graph_contract(tensor), gpu_proof_sha256=SHA_A)
        ).require_formal_authority()


def test_graph_state_machine_rejects_reallocated_tensor_and_incomplete_trace() -> None:
    tensor = torch.zeros(4)
    contract = _graph_contract(tensor)
    machine = GraphHotPathStateMachine(contract)
    with pytest.raises(RuntimeError, match="identity changed"):
        machine.observe(
            "device_to_device_copy",
            tensors=(("weight", tensor.clone()),),
        )
    incomplete = GraphHotPathStateMachine(contract)
    incomplete.observe("cuda_event_record", tensors=(("weight", tensor),))
    with pytest.raises(ValueError, match="required safe operations"):
        incomplete.finalize()


def test_native_gpu_proof_artifact_reopens_signatures_and_reservation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private, root_binding, bundle, authorization = _control_authority(monkeypatch)
    capability = NATIVE_RUNTIME_RELEASE_CAPABILITY
    payload = {
        "schema_version": 1,
        "kind": "lightcone_native_runtime_gpu_proof",
        "suite_id": "native_hot_path_tp1",
        "topology_mode": "tp1_dp1",
        "topology_sha256": SHA_A,
        "runner_protocol_sha256": NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[
            "native_hot_path_tp1"
        ],
        "assignment_sha256": "8" * 64,
        "qualification_observation_sha256": "9" * 64,
        "source_capability_sha256": capability.sha256,
        "pinned_sglang_commit": capability.pinned_sglang_commit,
        "patched_sglang_tree": capability.patched_sglang_tree,
        "semantic_patch_sha256": capability.semantic_patch_sha256,
        "run_nonce_sha256": "d" * 64,
        "qualification_authority_sha256": (
            NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256
        ),
        "source_identity_sha256": SHA_B,
        "inventory_sha256": INVENTORY_SHA256,
        "gpu_uuids": ["GPU-A"],
        "hardware_envelope_sha256": HARDWARE_SHA256,
        "junit_xml_sha256": SHA_C,
        "test_names": list(NATIVE_RUNTIME_QUALIFICATION_TESTS["native_hot_path_tp1"]),
        "tests_collected": 8,
        "tests_passed": 8,
        "tests_failed": 0,
        "tests_errored": 0,
        "tests_skipped": 0,
    }
    receipt = NativeRuntimeGpuProofReceipt(
        **{
            **payload,
            "gpu_uuids": ("GPU-A",),
            "test_names": tuple(payload["test_names"]),
        }
    )
    with pytest.raises(ValueError, match="qualification authority differs"):
        replace(receipt, qualification_authority_sha256="0" * 64)
    receipt_path = tmp_path / "native-runtime-proof.json"
    receipt_binding = receipt.write_unsigned(str(receipt_path.resolve()))
    receipt_raw_sha256 = receipt_binding.raw_sha256
    subject = ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=receipt_raw_sha256,
        protocol_sha256=capability.suite_protocol_sha256,
        registry_sha256=SHA_B,
        lineage_sha256=receipt.control_lineage_sha256,
    )
    control_challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id="native-hot-path-control-1",
        nonce_base64=base64.b64encode(b"c" * 32).decode("ascii"),
        subject_sha256=subject.sha256,
        issued_ns=1_600_000_000,
        expires_ns=2_600_000_000,
    )
    control = ControlArtifactAttestation(
        schema_version=1,
        kind="lightcone_control_artifact_attestation",
        subject=subject,
        hardware_envelope_sha256=HARDWARE_SHA256,
        trust_anchor_sha256=root_binding.sha256,
        trust_bundle_sha256=bundle.sha256,
        trusted_attester_policy_sha256=bundle.trusted_attester_policy.sha256,
        deployment_policy_authorization=authorization,
        challenge=control_challenge,
        attestation=SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="validation-signer",
            key_id="validation-signer-key",
            environment="release",
            public_key_base64=base64.b64encode(_public_bytes(private)).decode("ascii"),
            challenge_sha256=control_challenge.sha256,
            payload_sha256=receipt_raw_sha256,
            signature_base64=base64.b64encode(
                private.sign(
                    attestation_message(
                        control_challenge, payload_sha256=receipt_raw_sha256
                    )
                )
            ).decode("ascii"),
        ),
    )
    replay_root = tmp_path / "native-proof-replay"
    replay_root.mkdir(mode=0o700)
    replay_store = ChallengeReplayStore(str(replay_root.resolve()))
    verified = verify_native_runtime_gpu_proof(
        str(receipt_path.resolve()),
        control_attestation=control,
        replay_store=replay_store,
        expected_suite_id="native_hot_path_tp1",
        expected_topology_sha256=SHA_A,
        expected_source_identity_sha256=SHA_B,
        expected_inventory_sha256=INVENTORY_SHA256,
        expected_gpu_uuids=("GPU-A",),
        expected_hardware_envelope_sha256=HARDWARE_SHA256,
        expected_run_nonce_sha256=receipt.run_nonce_sha256,
        now_ns=NOW_NS,
    )
    assert (
        verified.qualification_authority_sha256
        == NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256
    )
    assert (
        require_fixed_address_graph_gpu_proof(
            claimed_source_capability_sha256=capability.sha256,
            verified_gpu_proof=verified,
            expected_source_identity_sha256=SHA_B,
            expected_inventory_sha256=INVENTORY_SHA256,
            expected_gpu_uuids=("GPU-A",),
        )
        is verified
    )
    for changed in (
        {
            "expected_source_identity_sha256": SHA_C,
            "expected_inventory_sha256": INVENTORY_SHA256,
            "expected_gpu_uuids": ("GPU-A",),
        },
        {
            "expected_source_identity_sha256": SHA_B,
            "expected_inventory_sha256": SHA_C,
            "expected_gpu_uuids": ("GPU-A",),
        },
        {
            "expected_source_identity_sha256": SHA_B,
            "expected_inventory_sha256": INVENTORY_SHA256,
            "expected_gpu_uuids": ("GPU-B",),
        },
    ):
        with pytest.raises(NativeReadinessBlocked, match="gpu_proof_unavailable"):
            require_fixed_address_graph_gpu_proof(
                claimed_source_capability_sha256=capability.sha256,
                verified_gpu_proof=verified,
                **changed,
            )
    artifact = build_native_runtime_gpu_proof_artifact(
        receipt_path=str(receipt_path.resolve()),
        control_attestation=control,
        replay_store=replay_store,
        verified_proof=verified,
    )
    reopened = NativeRuntimeGpuProofArtifact.from_dict(artifact.to_dict())
    assert reopened.revalidate(now_ns=NOW_NS).sha256 == verified.sha256
    assert receipt.run_nonce_sha256 in reopened.replay_reservation.challenge_sha256s
    artifact_path = (tmp_path / "native-runtime-proof-artifact.json").resolve()
    publish_canonical_json_no_replace(artifact_path, artifact.to_dict())
    assert (
        validate_native_runtime_gpu_proof_artifact(
            str(artifact_path),
            expected_suite_id="native_hot_path_tp1",
            expected_topology_sha256=SHA_A,
            expected_source_identity_sha256=SHA_B,
            expected_inventory_sha256=INVENTORY_SHA256,
            expected_gpu_uuids=("GPU-A",),
            expected_hardware_envelope_sha256=HARDWARE_SHA256,
            expected_assignment_sha256=receipt.assignment_sha256,
            expected_qualification_observation_sha256=(
                receipt.qualification_observation_sha256
            ),
            expected_root_manifest_sha256=root_binding.semantic_sha256,
            now_ns=NOW_NS,
        ).sha256
        == verified.sha256
    )
    with pytest.raises(ValueError, match="expected suite identity"):
        validate_native_runtime_gpu_proof_artifact(
            str(artifact_path),
            expected_suite_id="native_hot_path_tp1",
            expected_topology_sha256=SHA_A,
            expected_source_identity_sha256=SHA_B,
            expected_inventory_sha256=INVENTORY_SHA256,
            expected_gpu_uuids=("GPU-A",),
            expected_hardware_envelope_sha256=HARDWARE_SHA256,
            expected_assignment_sha256="e" * 64,
            expected_qualification_observation_sha256=(
                receipt.qualification_observation_sha256
            ),
            expected_root_manifest_sha256=root_binding.semantic_sha256,
            now_ns=NOW_NS,
        )
