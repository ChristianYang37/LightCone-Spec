from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import lightcone_spec.runtime.release_trust_root as release_root_module
from lightcone_spec.adaptation.governor import (
    BoundedCohortStateManager,
    CohortAdmissionReason,
    CohortOffloadMode,
    CohortStateKey,
    HBMAdmissionReason,
    HBMAdmissionRequest,
    HBMGovernor,
    HBMLedger,
    MemoryPressureAction,
    RankMemoryState,
)
from lightcone_spec.doctor import _mode_gates
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
)
from lightcone_spec.runtime.distributed import (
    DISTRIBUTED_RUNTIME_GPU_PROOF_PROTOCOL_SHA256,
    DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS,
    DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
    DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
    AllRankPublicationCoordinator,
    CohortRouteIdentity,
    DistributedRuntimeGpuProofArtifact,
    DistributedRuntimeGpuProofReceipt,
    GlooPublicationTransport,
    InferenceParameterOwnership,
    ParameterOwnership,
    PrepareDisposition,
    PublicationCandidate,
    PublicationOutcome,
    RankDecisionReceipt,
    RankPrepare,
    RankTopologyReceipt,
    ReplicaLocalRouter,
    TopologyIdentity,
    TopologyReceiptSet,
    UpdateIdentity,
    VerifiedDistributedRuntimeGpuProof,
    build_distributed_runtime_gpu_proof_artifact,
    validate_decision_receipts,
    verify_distributed_runtime_gpu_proof,
)
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256,
)
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    SourceReleaseEd25519Root,
    SourceReleaseRootBinding,
    deployment_policy_subject_sha256,
)


def _gloo_publication_worker(
    init_file: str,
    rank: int,
    queue: multiprocessing.Queue,
    nonfinite_rank: int | None,
) -> None:
    from torch import distributed

    try:
        distributed.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=20),
        )
        topology = topology_receipts(tp=2, dp=1)
        update = candidate()
        vote = prepare_votes(topology, update)[rank]
        if rank == nonfinite_rank:
            vote = replace(vote, finite=False)
        transport = GlooPublicationTransport(topology, local_rank=rank)
        decision = transport.prepare_and_decide(update, vote)
        receipts = transport.finalize(
            decision,
            applied=decision.outcome is PublicationOutcome.COMMIT,
        )
        queue.put(
            (
                "ok",
                rank,
                decision.sha256,
                decision.outcome.value,
                tuple(receipt.rank for receipt in receipts),
            )
        )
    except Exception as error:  # noqa: BLE001  # pragma: no cover - process boundary
        queue.put(("error", rank, type(error).__name__, str(error)))
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def topology_receipts(
    *,
    tp: int = 2,
    dp: int = 1,
    process_prefix: str = "process",
) -> TopologyReceiptSet:
    world = tp * dp
    receipts = []
    for rank in range(world):
        topology = TopologyIdentity(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            node_count=1,
            node_id="node-0",
            node_rank=0,
            global_rank=rank,
            local_rank=rank,
            tensor_parallel_rank=rank % tp,
            data_parallel_rank=rank // tp,
            device_id=f"gpu-{rank}",
            rendezvous_id="rdzv-locked",
            router_id="router-locked",
            clock_id="clock-ptp-locked",
        )
        receipts.append(
            RankTopologyReceipt(
                topology=topology,
                process_id=f"{process_prefix}-{rank}",
                observed_world_size=world,
            )
        )
    return TopologyReceiptSet(tuple(receipts))


def candidate(*, sequence: int = 7) -> PublicationCandidate:
    return PublicationCandidate(
        update=UpdateIdentity(
            cohort_sha256="a" * 64,
            source_version=3,
            cohort_epoch=2,
            sequence_number=sequence,
            source_rows_sha256="b" * 64,
        ),
        buffer_generation=5,
        optimizer_generation=11,
    )


def prepare_votes(
    topology: TopologyReceiptSet,
    value: PublicationCandidate,
) -> tuple[RankPrepare, ...]:
    return tuple(
        RankPrepare(
            rank=rank,
            topology_receipt_sha256=topology.receipt_for_rank(rank).sha256,
            candidate_sha256=value.sha256,
            source_version=value.update.source_version,
            cohort_epoch=value.update.cohort_epoch,
            buffer_generation=value.buffer_generation,
            optimizer_generation=value.optimizer_generation,
            ready=True,
            finite=True,
            memory_reserved=True,
            safe_boundary=True,
        )
        for rank in range(topology.world_size)
    )


def decision_receipts(
    topology: TopologyReceiptSet,
    decision_sha256: str,
    *,
    applied: bool,
) -> tuple[RankDecisionReceipt, ...]:
    return tuple(
        RankDecisionReceipt(
            rank=rank,
            topology_receipt_sha256=topology.receipt_for_rank(rank).sha256,
            decision_sha256=decision_sha256,
            applied=applied,
        )
        for rank in range(topology.world_size)
    )


def _unsigned_gpu_proof(
    *,
    mode: str = "tp2_dp1",
    now_ns: int = 1_000,
) -> tuple[
    DistributedRuntimeGpuProofReceipt,
    TrustedAttesterPolicy,
    Ed25519PrivateKey,
]:
    capability = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES[mode]
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_base64 = base64.b64encode(public_key).decode()
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    policy = TrustedAttesterPolicy(
        policy_id="release-distributed-gpu-proof-v1",
        trusted_attesters=(("release-root", "release-root-key", public_key_sha256),),
        public_keys=((public_key_sha256, public_key_base64),),
    )
    receipt = DistributedRuntimeGpuProofReceipt(
        schema_version=1,
        kind="lightcone_distributed_runtime_gpu_proof",
        topology_mode=mode,
        topology_sha256="5" * 64,
        runner_protocol_sha256=DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[mode],
        assignment_sha256="7" * 64,
        qualification_observation_sha256="8" * 64,
        base_exactness_result_pointer_sha256="9" * 64,
        source_capability_sha256=capability.sha256,
        pinned_sglang_commit=capability.pinned_sglang_commit,
        patched_sglang_tree=capability.patched_sglang_tree,
        semantic_patch_sha256=capability.semantic_patch_sha256,
        run_nonce_sha256=hashlib.sha256(
            f"distributed-qualification:{mode}:{now_ns}".encode()
        ).hexdigest(),
        qualification_authority_sha256=(NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256),
        source_identity_sha256="1" * 64,
        inventory_sha256="2" * 64,
        gpu_uuids=("GPU-A", "GPU-B"),
        hardware_envelope_sha256="3" * 64,
        junit_xml_sha256="4" * 64,
        tests_collected=8,
        tests_passed=8,
        tests_failed=0,
        tests_errored=0,
        tests_skipped=0,
        qualification_junit_xml_sha256="6" * 64,
        qualification_test_names=DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS[mode],
        qualification_tests_collected=8,
        qualification_tests_passed=8,
        qualification_tests_failed=0,
        qualification_tests_errored=0,
        qualification_tests_skipped=0,
    )
    with pytest.raises(ValueError, match="qualification authority differs"):
        replace(receipt, qualification_authority_sha256="0" * 64)
    return receipt, policy, private_key


def _write_unsigned_gpu_proof(
    root: Path,
    receipt: DistributedRuntimeGpuProofReceipt,
) -> tuple[str, str]:
    path = root / "distributed-runtime-gpu-proof.json"
    binding = receipt.write_unsigned(str(path.resolve()))
    return binding.absolute_path, binding.raw_sha256


def _gpu_proof_control_attestation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    receipt: DistributedRuntimeGpuProofReceipt,
    receipt_raw_sha256: str,
    policy: TrustedAttesterPolicy,
    artifact_private_key: Ed25519PrivateKey,
    now_ns: int,
) -> ControlArtifactAttestation:
    root_private_key = Ed25519PrivateKey.generate()
    root_public_key = root_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    root_spki = root_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    root = SourceReleaseEd25519Root(
        schema_version=1,
        kind="lightcone_source_release_ed25519_root",
        root_id="lightcone-release-root-2026q3",
        key_id="lightcone-release-root-key-2026q3",
        algorithm="Ed25519",
        public_key_base64=base64.b64encode(root_public_key).decode("ascii"),
        public_key_sha256=hashlib.sha256(root_public_key).hexdigest(),
        spki_sha256=hashlib.sha256(root_spki).hexdigest(),
    )
    root_binding = SourceReleaseRootBinding(
        root=root,
        path="/validation/release-root.json",
        sidecar_path="/validation/release-root.json.sha256",
        semantic_sha256=root.sha256,
        file_sha256="7" * 64,
        sidecar_file_sha256="8" * 64,
    )
    bundle = TrustedAttesterPolicyBundle(
        schema_version=1,
        kind="lightcone_trusted_attester_policy_bundle",
        bundle_id="release-distributed-gpu-proof-bundle-v1",
        valid_from_ns=now_ns - 1_000_000_000,
        expires_ns=now_ns + 2_000_000_000,
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
        hardware_envelope_sha256_allowlist=(receipt.hardware_envelope_sha256,),
        trusted_attester_policy=policy,
    )
    deployment_subject = deployment_policy_subject_sha256(
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=receipt.inventory_sha256,
        bundle_sha256=bundle.sha256,
    )
    deployment_challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id="release-distributed-deployment-policy",
        nonce_base64=base64.b64encode(b"d" * 32).decode("ascii"),
        subject_sha256=deployment_subject,
        issued_ns=now_ns - 500_000_000,
        expires_ns=now_ns + 1_500_000_000,
    )
    deployment_signature = root_private_key.sign(
        attestation_message(deployment_challenge, payload_sha256=bundle.sha256)
    )
    authorization = DeploymentPolicyAuthorization(
        schema_version=1,
        kind="lightcone_deployment_policy_authorization",
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=receipt.inventory_sha256,
        bundle=bundle,
        challenge=deployment_challenge,
        signature_base64=base64.b64encode(deployment_signature).decode("ascii"),
    )
    subject = ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=receipt_raw_sha256,
        protocol_sha256=DISTRIBUTED_RUNTIME_GPU_PROOF_PROTOCOL_SHA256,
        registry_sha256=receipt.source_identity_sha256,
        lineage_sha256=receipt.control_lineage_sha256,
    )
    control_challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id="release-distributed-proof-control",
        nonce_base64=base64.b64encode(b"c" * 32).decode("ascii"),
        subject_sha256=subject.sha256,
        issued_ns=now_ns - 100_000_000,
        expires_ns=now_ns + 900_000_000,
    )
    control_signature = artifact_private_key.sign(
        attestation_message(
            control_challenge,
            payload_sha256=subject.artifact_sha256,
        )
    )
    public_key_base64 = policy.public_keys[0][1]
    control = ControlArtifactAttestation(
        schema_version=1,
        kind="lightcone_control_artifact_attestation",
        subject=subject,
        hardware_envelope_sha256=receipt.hardware_envelope_sha256,
        trust_anchor_sha256=root_binding.sha256,
        trust_bundle_sha256=bundle.sha256,
        trusted_attester_policy_sha256=policy.sha256,
        deployment_policy_authorization=authorization,
        challenge=control_challenge,
        attestation=SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="release-root",
            key_id="release-root-key",
            environment="release",
            public_key_base64=public_key_base64,
            challenge_sha256=control_challenge.sha256,
            payload_sha256=subject.artifact_sha256,
            signature_base64=base64.b64encode(control_signature).decode("ascii"),
        ),
    )
    monkeypatch.setattr(
        release_root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    return control


def test_distributed_gpu_proof_is_root_signed_identity_bound_and_one_shot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now_ns = 2_000_000_000
    receipt, policy, private_key = _unsigned_gpu_proof(now_ns=now_ns)
    receipt_path, receipt_raw_sha256 = _write_unsigned_gpu_proof(tmp_path, receipt)
    control = _gpu_proof_control_attestation(
        monkeypatch,
        receipt=receipt,
        receipt_raw_sha256=receipt_raw_sha256,
        policy=policy,
        artifact_private_key=private_key,
        now_ns=now_ns,
    )
    replay_root = tmp_path / "distributed-proof-replay"
    replay_root.mkdir()
    replay_store = ChallengeReplayStore(str(replay_root.resolve()))
    verified = verify_distributed_runtime_gpu_proof(
        receipt_path,
        control_attestation=control,
        replay_store=replay_store,
        expected_topology_mode="tp2_dp1",
        expected_topology_sha256="5" * 64,
        expected_source_capability_sha256=receipt.source_capability_sha256,
        expected_source_identity_sha256="1" * 64,
        expected_inventory_sha256="2" * 64,
        expected_gpu_uuids=("GPU-A", "GPU-B"),
        expected_hardware_envelope_sha256="3" * 64,
        expected_run_nonce_sha256=receipt.run_nonce_sha256,
        now_ns=now_ns,
    )
    assert verified.receipt_sha256 == receipt.sha256
    assert verified.topology_mode == "tp2_dp1"
    assert verified.control_envelope_sha256 == control.sha256
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "manifests/runtime/industrial_compatibility_v1.json"
    )
    projection = _mode_gates(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        distributed_gpu_proofs=(verified,),
    )
    assert projection["available_after_dynamic_proof"]["tp2"] == {
        "status": "AVAILABLE",
        "qualification_receipt_sha256": verified.receipt_sha256,
    }
    assert "tp2" not in projection["pending_dynamic_gpu_proof"]
    assert "dp2" in projection["pending_dynamic_gpu_proof"]
    assert tuple(replay_root.glob("reservation-*.json"))
    with pytest.raises(ValueError, match="already consumed"):
        verify_distributed_runtime_gpu_proof(
            receipt_path,
            control_attestation=control,
            replay_store=replay_store,
            expected_topology_mode="tp2_dp1",
            expected_topology_sha256="5" * 64,
            expected_source_capability_sha256=receipt.source_capability_sha256,
            expected_source_identity_sha256="1" * 64,
            expected_inventory_sha256="2" * 64,
            expected_gpu_uuids=("GPU-A", "GPU-B"),
            expected_hardware_envelope_sha256="3" * 64,
            expected_run_nonce_sha256=receipt.run_nonce_sha256,
            now_ns=now_ns,
        )
    with pytest.raises(TypeError, match="only come from signature verification"):
        VerifiedDistributedRuntimeGpuProof(
            receipt_sha256=receipt.sha256,
            receipt_raw_sha256=receipt_raw_sha256,
            runner_protocol_sha256=receipt.runner_protocol_sha256,
            assignment_sha256=receipt.assignment_sha256,
            qualification_observation_sha256=(receipt.qualification_observation_sha256),
            base_exactness_result_pointer_sha256=(
                receipt.base_exactness_result_pointer_sha256
            ),
            source_capability_sha256=receipt.source_capability_sha256,
            qualification_authority_sha256=(receipt.qualification_authority_sha256),
            trusted_policy_sha256=policy.sha256,
            challenge_sha256=control.challenge.sha256,
            source_identity_sha256="1" * 64,
            inventory_sha256="2" * 64,
            hardware_envelope_sha256="3" * 64,
            topology_mode="tp2_dp1",
            topology_sha256="5" * 64,
            gpu_uuids=("GPU-A", "GPU-B"),
            control_envelope_sha256=control.sha256,
            challenge_reservation_sha256=verified.challenge_reservation_sha256,
            _verification_tag=object(),
        )
    artifact = build_distributed_runtime_gpu_proof_artifact(
        receipt_path=receipt_path,
        control_attestation=control,
        replay_store=replay_store,
        verified_proof=verified,
    )
    reopened = DistributedRuntimeGpuProofArtifact.from_dict(artifact.to_dict())
    assert reopened.revalidate(now_ns=now_ns).sha256 == verified.sha256


def test_distributed_gpu_proof_rejects_expiry_identity_drift_and_skips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now_ns = 2_000_000_000
    receipt, policy, private_key = _unsigned_gpu_proof(now_ns=now_ns)
    receipt_path, receipt_raw_sha256 = _write_unsigned_gpu_proof(tmp_path, receipt)
    control = _gpu_proof_control_attestation(
        monkeypatch,
        receipt=receipt,
        receipt_raw_sha256=receipt_raw_sha256,
        policy=policy,
        artifact_private_key=private_key,
        now_ns=now_ns,
    )
    replay_root = tmp_path / "distributed-proof-replay"
    replay_root.mkdir()
    replay_store = ChallengeReplayStore(str(replay_root.resolve()))
    with pytest.raises(ValueError, match="expected identity"):
        verify_distributed_runtime_gpu_proof(
            receipt_path,
            control_attestation=control,
            replay_store=replay_store,
            expected_topology_mode="tp2_dp1",
            expected_topology_sha256="5" * 64,
            expected_source_capability_sha256=receipt.source_capability_sha256,
            expected_source_identity_sha256="1" * 64,
            expected_inventory_sha256="9" * 64,
            expected_gpu_uuids=("GPU-A", "GPU-B"),
            expected_hardware_envelope_sha256="3" * 64,
            expected_run_nonce_sha256=receipt.run_nonce_sha256,
            now_ns=now_ns,
        )
    with pytest.raises(ValueError, match="expired"):
        verify_distributed_runtime_gpu_proof(
            receipt_path,
            control_attestation=control,
            replay_store=replay_store,
            expected_topology_mode="tp2_dp1",
            expected_topology_sha256="5" * 64,
            expected_source_capability_sha256=receipt.source_capability_sha256,
            expected_source_identity_sha256="1" * 64,
            expected_inventory_sha256="2" * 64,
            expected_gpu_uuids=("GPU-A", "GPU-B"),
            expected_hardware_envelope_sha256="3" * 64,
            expected_run_nonce_sha256=receipt.run_nonce_sha256,
            now_ns=control.challenge.expires_ns + 1,
        )
    with pytest.raises(ValueError, match="challenge subject"):
        replace(
            control,
            subject=replace(control.subject, registry_sha256="9" * 64),
        )
    with pytest.raises(ValueError, match="zero skip"):
        replace(receipt, tests_passed=7, tests_skipped=1)
    with pytest.raises(ValueError, match="named qualification"):
        replace(
            receipt,
            qualification_test_names=(
                *receipt.qualification_test_names[:-1],
                "caller_substituted_qualification",
            ),
        )
    with pytest.raises(ValueError, match="every named qualification"):
        replace(
            receipt,
            qualification_tests_passed=7,
            qualification_tests_skipped=1,
        )


def test_topology_identity_is_complete_stable_and_fail_closed() -> None:
    receipts = topology_receipts()
    rebuilt = topology_receipts()
    assert receipts.topology_sha256 == rebuilt.topology_sha256
    assert receipts.receipt_sha256 == rebuilt.receipt_sha256
    assert receipts.tensor_parallel_group(0) == (0, 1)
    assert receipts.mode == "tp2_dp1"
    with pytest.raises(ValueError, match="global rank"):
        replace(
            receipts.receipt_for_rank(0).topology,
            global_rank=1,
        )
    with pytest.raises(ValueError, match="cover every declared rank"):
        TopologyReceiptSet(receipts.receipts[:-1])
    with pytest.raises(ValueError, match="tp1_dp1, tp2_dp1, or tp1_dp2"):
        replace(receipts.receipt_for_rank(0).topology, data_parallel_size=2)
    with pytest.raises(ValueError, match="multi-node"):
        replace(receipts.receipt_for_rank(0).topology, node_count=2)


def test_inference_ownership_never_averages_across_dp_replicas() -> None:
    topology = topology_receipts(tp=2, dp=1)
    sharded = InferenceParameterOwnership(
        "layers.0.q_proj.weight",
        ParameterOwnership.SHARDED,
        (0, 1),
        shard_axis=0,
    )
    replicated = InferenceParameterOwnership(
        "acceptance_projection",
        ParameterOwnership.REPLICATED,
        (0, 1),
    )
    assert sharded.gradient_reduction_ranks(0, topology) == (0,)
    assert replicated.gradient_reduction_ranks(0, topology) == (0, 1)
    with pytest.raises(ValueError, match="partially covers"):
        InferenceParameterOwnership(
            "partial",
            ParameterOwnership.REPLICATED,
            (0,),
        ).validate(topology)

    replicas = topology_receipts(tp=1, dp=2)
    replica_local = InferenceParameterOwnership(
        "replica_local_adapter",
        ParameterOwnership.REPLICATED,
        (0, 1),
    )
    assert replica_local.gradient_reduction_ranks(0, replicas) == (0,)
    assert replica_local.gradient_reduction_ranks(1, replicas) == (1,)


def test_two_phase_publication_commits_one_all_rank_decision() -> None:
    topology = topology_receipts()
    update = candidate()
    coordinator = AllRankPublicationCoordinator(topology)
    prepared = coordinator.prepare(update, prepare_votes(topology, update))
    assert prepared.disposition is PrepareDisposition.COMMIT_READY
    decision = coordinator.decide(prepared)
    assert decision.outcome is PublicationOutcome.COMMIT
    assert not decision.service_ready
    assert not decision.admission_allowed
    assert not coordinator.service_ready
    assert not coordinator.admission_allowed
    receipts = decision_receipts(topology, decision.sha256, applied=True)
    coordinator.finalize(decision, receipts)
    assert coordinator.service_ready
    assert coordinator.admission_allowed
    with pytest.raises(ValueError, match="partial model"):
        validate_decision_receipts(
            decision,
            (replace(receipts[0], applied=False), *receipts[1:]),
            topology,
        )


def test_partial_rank_copy_never_reopens_service() -> None:
    topology = topology_receipts()
    update = candidate()
    coordinator = AllRankPublicationCoordinator(topology)
    decision = coordinator.decide(
        coordinator.prepare(update, prepare_votes(topology, update))
    )
    receipts = decision_receipts(topology, decision.sha256, applied=True)
    with pytest.raises(ValueError, match="partial model"):
        coordinator.finalize(
            decision,
            (replace(receipts[0], applied=False), *receipts[1:]),
        )
    assert not coordinator.service_ready
    assert not coordinator.admission_allowed
    assert coordinator.restart_required


@pytest.mark.parametrize(
    ("nonfinite_rank", "expected"),
    [(None, "commit"), (1, "abort_static")],
)
def test_real_gloo_processes_reach_one_two_phase_decision(
    tmp_path, nonfinite_rank: int | None, expected: str
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / f"gloo-{expected}")
    processes = [
        context.Process(
            target=_gloo_publication_worker,
            args=(init_file, rank, queue, nonfinite_rank),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive(), "gloo publication worker hung"
        assert process.exitcode == 0
    rows = [queue.get(timeout=5) for _ in processes]
    assert {row[0] for row in rows} == {"ok"}
    assert {row[2] for row in rows} == {rows[0][2]}
    assert {row[3] for row in rows} == {expected}
    assert {row[4] for row in rows} == {(0, 1)}


def test_tp_candidate_failure_collectively_aborts_to_static() -> None:
    topology = topology_receipts()
    update = candidate()
    votes = list(prepare_votes(topology, update))
    votes[1] = replace(votes[1], finite=False)
    coordinator = AllRankPublicationCoordinator(topology)
    prepared = coordinator.prepare(update, tuple(votes))
    assert prepared.disposition is PrepareDisposition.ABORT_STATIC
    assert prepared.reasons == ("rank_1:finiteness",)
    decision = coordinator.decide(prepared)
    assert decision.outcome is PublicationOutcome.ABORT_STATIC
    assert not decision.service_ready
    assert not decision.admission_allowed
    assert not decision.restart_required
    with pytest.raises(ValueError, match="service state"):
        replace(decision, service_ready=True, admission_allowed=True)
    validate_decision_receipts(
        decision,
        decision_receipts(topology, decision.sha256, applied=False),
        topology,
    )
    coordinator.finalize(
        decision,
        decision_receipts(topology, decision.sha256, applied=False),
    )
    assert coordinator.service_ready
    assert coordinator.admission_allowed


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("source_version", 4, "source_version"),
        ("cohort_epoch", 3, "cohort_epoch"),
        ("buffer_generation", 6, "buffer_generation"),
        ("optimizer_generation", 12, "optimizer_generation"),
        ("ready", False, "readiness"),
        ("finite", False, "finiteness"),
        ("memory_reserved", False, "memory_reservation"),
        ("safe_boundary", False, "safe_boundary"),
    ],
)
def test_prepare_checks_every_candidate_generation_and_gate(
    field: str,
    value: int | bool,
    reason: str,
) -> None:
    topology = topology_receipts()
    update = candidate()
    votes = list(prepare_votes(topology, update))
    votes[0] = replace(votes[0], **{field: value})
    prepared = AllRankPublicationCoordinator(topology).prepare(update, tuple(votes))
    assert prepared.disposition is PrepareDisposition.ABORT_STATIC
    assert prepared.reasons == (f"rank_0:{reason}",)


def test_process_group_failure_stops_admission_until_clean_restart() -> None:
    topology = topology_receipts()
    update = candidate()
    votes = list(prepare_votes(topology, update))
    votes[1] = replace(votes[1], process_group_healthy=False)
    coordinator = AllRankPublicationCoordinator(topology)
    decision = coordinator.decide(coordinator.prepare(update, tuple(votes)))
    assert decision.outcome is PublicationOutcome.PROCESS_GROUP_FAILURE
    assert not coordinator.service_ready
    assert not coordinator.admission_allowed
    assert coordinator.restart_required
    with pytest.raises(RuntimeError, match="restart"):
        coordinator.prepare(
            candidate(sequence=8), prepare_votes(topology, candidate(sequence=8))
        )
    restarted = topology_receipts(process_prefix="restarted")
    assert restarted.topology_sha256 == topology.topology_sha256
    assert restarted.receipt_sha256 != topology.receipt_sha256
    coordinator.mark_process_group_restarted(restarted)
    assert coordinator.service_ready
    assert coordinator.admission_allowed
    assert not coordinator.restart_required


def test_missing_rank_is_process_failure_and_retry_identity_is_deduplicated() -> None:
    topology = topology_receipts()
    first = candidate()
    coordinator = AllRankPublicationCoordinator(topology)
    first_decision = coordinator.decide(
        coordinator.prepare(first, prepare_votes(topology, first))
    )
    coordinator.finalize(
        first_decision,
        decision_receipts(topology, first_decision.sha256, applied=True),
    )
    duplicate = coordinator.prepare(first, prepare_votes(topology, first))
    assert duplicate.disposition is PrepareDisposition.ABORT_STATIC
    assert duplicate.reasons == ("duplicate_update_identity",)
    duplicate_decision = coordinator.decide(duplicate)
    coordinator.finalize(
        duplicate_decision,
        decision_receipts(
            topology,
            duplicate_decision.sha256,
            applied=False,
        ),
    )

    second = candidate(sequence=8)
    missing = coordinator.prepare(second, prepare_votes(topology, second)[:-1])
    assert missing.disposition is PrepareDisposition.PROCESS_GROUP_FAILURE
    assert missing.reasons == ("missing_ranks:1",)


def test_replica_local_routing_is_sticky_and_topology_bound() -> None:
    topology = topology_receipts(tp=1, dp=2)
    router = ReplicaLocalRouter(topology)
    identity = CohortRouteIdentity(
        tenant_id="tenant-a",
        cohort_sha256="c" * 64,
        router_id="router-locked",
        topology_sha256=topology.topology_sha256,
    )
    replica = router.route(identity)
    assert router.route(identity) == replica
    assert router.ranks_for(identity) == topology.tensor_parallel_group(replica)
    assert not router.data_parallel_gradient_averaging
    with pytest.raises(ValueError, match="another router"):
        router.route(replace(identity, router_id="other-router"))


def test_dp2_cohort_versions_and_optimizer_slabs_never_cross_replicas() -> None:
    manager = BoundedCohortStateManager(
        capacity=2,
        slab_bytes=4096,
        tenant_quotas={"tenant-a": 2},
        ttl_seconds=60,
    )
    replica_zero = cohort_key("f", replica=0)
    replica_one = cohort_key("f", replica=1)
    assert manager.admit(replica_zero, now=0, version=3).admitted
    assert manager.admit(replica_one, now=0, version=11).admitted
    zero = manager.publish_version(
        replica_zero,
        source_version=3,
        new_version=4,
        now=1,
    )
    one = manager.snapshot(replica_one)
    assert one is not None
    assert zero.slab_id != one.slab_id
    assert zero.version == 4
    assert one.version == 11


def full_ledger(*, global_peak: int) -> HBMLedger:
    return HBMLedger(
        target_weights_bytes=10,
        drafter_weights_bytes=10,
        target_kv_bytes=10,
        drafter_kv_bytes=10,
        active_merged_parameters_bytes=10,
        fp32_masters_bytes=10,
        gradients_bytes=10,
        optimizer_tensors_bytes=10,
        candidate_bytes=10,
        staging_bytes=10,
        merge_scratch_bytes=10,
        differentiable_activations_bytes=10,
        graph_private_pools_bytes=10,
        library_workspace_bytes=10,
        nccl_buffers_bytes=10,
        kv_gather_scratch_bytes=10,
        backend_scratch_bytes=10,
        telemetry_staging_bytes=10,
        fragmentation_margin_bytes=20,
        allocator_allocated_peak_bytes=250,
        allocator_reserved_peak_bytes=300,
        nvml_process_peak_bytes=280,
        nvml_global_peak_bytes=global_peak,
    )


def test_hbm_ledger_keeps_predictions_and_observations_separate() -> None:
    ledger = full_ledger(global_peak=320)
    assert ledger.predicted_resident_bytes == 130
    assert ledger.predicted_peak_bytes == 200
    assert ledger.observed_process_peak_bytes == 300
    assert ledger.prediction_error_bytes == 100
    assert RankMemoryState(0, 1000, ledger).charged_peak_bytes == 320
    with pytest.raises(ValueError, match="reserved peak"):
        replace(ledger, allocator_reserved_peak_bytes=200)


def test_hbm_admission_uses_least_feasible_rank_and_reserves_before_kv() -> None:
    governor = HBMGovernor(
        (
            RankMemoryState(0, 1000, full_ledger(global_peak=320)),
            RankMemoryState(1, 800, full_ledger(global_peak=400)),
        ),
        expected_ranks=2,
    )
    admitted = governor.assess(
        HBMAdmissionRequest(
            adaptation_reserve_bytes=100,
            target_kv_bytes=100,
            drafter_kv_bytes=50,
            safety_margin_bytes=50,
        )
    )
    assert admitted.admitted
    assert admitted.limiting_rank == 1
    assert admitted.ranks[1].headroom_bytes == 100

    kv_blocked = governor.assess(
        HBMAdmissionRequest(
            adaptation_reserve_bytes=100,
            target_kv_bytes=200,
            drafter_kv_bytes=100,
            safety_margin_bytes=50,
        )
    )
    assert not kv_blocked.admitted
    assert kv_blocked.reason is HBMAdmissionReason.KV_ADMISSION_EXCEEDS_LEAST_RANK
    adaptation_blocked = governor.assess(
        HBMAdmissionRequest(
            adaptation_reserve_bytes=400,
            target_kv_bytes=1,
            drafter_kv_bytes=1,
            safety_margin_bytes=50,
        )
    )
    assert adaptation_blocked.reason is (
        HBMAdmissionReason.ADAPTATION_RESERVE_EXCEEDS_LEAST_RANK
    )


def test_memory_pressure_order_never_sacrifices_active_correctness() -> None:
    base = HBMGovernor.pressure_plan()
    assert [step.action for step in base] == list(MemoryPressureAction)[:5]
    assert all(
        step.action is not MemoryPressureAction.OFFLOAD_COLD_INACTIVE_COHORT
        for step in base
    )
    with_offload = HBMGovernor.pressure_plan(allow_cold_offload=True)
    assert with_offload[-1].action is (
        MemoryPressureAction.OFFLOAD_COLD_INACTIVE_COHORT
    )


def cohort_key(
    suffix: str,
    *,
    tenant: str = "tenant-a",
    replica: int = 0,
) -> CohortStateKey:
    return CohortStateKey(
        tenant_id=tenant,
        cohort_sha256=suffix * 64,
        replica_id=replica,
    )


def test_cohort_manager_enforces_quota_ttl_and_privacy_isolation() -> None:
    manager = BoundedCohortStateManager(
        capacity=2,
        slab_bytes=4096,
        tenant_quotas={"tenant-a": 1, "tenant-b": 1},
        ttl_seconds=10,
    )
    a = cohort_key("a")
    a_other_tenant = cohort_key("a", tenant="tenant-b")
    assert manager.admit(a, now=0).reason is CohortAdmissionReason.ADMITTED
    assert manager.admit(cohort_key("b"), now=0).reason is (
        CohortAdmissionReason.TENANT_QUOTA
    )
    assert manager.admit(a_other_tenant, now=0).admitted
    assert manager.snapshot(a) != manager.snapshot(a_other_tenant)

    manager.acquire(a, now=5)
    expired = manager.reclaim_expired(now=11)
    assert [receipt.key for receipt in expired] == [a_other_tenant]
    assert manager.snapshot(a) is not None
    manager.release(a, now=12)
    assert manager.reclaim_expired(now=21) == ()
    final = manager.reclaim_expired(now=22)
    assert [receipt.key for receipt in final] == [a]
    assert manager.state_count == 0


def test_cohort_manager_requires_explicit_lru_reclamation() -> None:
    manager = BoundedCohortStateManager(
        capacity=2,
        slab_bytes=1024,
        tenant_quotas={"tenant-a": 2},
        ttl_seconds=100,
    )
    old = cohort_key("a")
    recent = cohort_key("b")
    assert manager.admit(old, now=0).admitted
    assert manager.admit(recent, now=1).admitted
    manager.acquire(old, now=5)
    manager.release(old, now=5)
    blocked = manager.admit(cohort_key("c"), now=6)
    assert blocked.reason is CohortAdmissionReason.TENANT_QUOTA
    reclaimed = manager.reclaim_lru(count=1, now=6)
    assert [receipt.key for receipt in reclaimed] == [recent]
    assert manager.admit(cohort_key("c"), now=6).admitted

    capacity_limited = BoundedCohortStateManager(
        capacity=1,
        slab_bytes=1024,
        tenant_quotas={"tenant-a": 1, "tenant-b": 1},
        ttl_seconds=100,
    )
    assert capacity_limited.admit(cohort_key("d"), now=0).admitted
    assert (
        capacity_limited.admit(cohort_key("e", tenant="tenant-b"), now=0).reason
        is CohortAdmissionReason.CAPACITY
    )


def test_cold_offload_is_separately_enabled_timed_and_inactive_only() -> None:
    key = cohort_key("d")
    disabled = BoundedCohortStateManager(
        capacity=1,
        slab_bytes=2048,
        tenant_quotas={"tenant-a": 1},
        ttl_seconds=100,
    )
    disabled.admit(key, now=0)
    with pytest.raises(RuntimeError, match="not explicitly enabled"):
        disabled.offload_cold(key, started_at=1, completed_at=2)

    manager = BoundedCohortStateManager(
        capacity=1,
        slab_bytes=2048,
        tenant_quotas={"tenant-a": 1},
        ttl_seconds=100,
        offload_mode=CohortOffloadMode.COLD_INACTIVE_TIMED,
        offloaded_capacity=1,
    )
    manager.admit(key, now=0)
    manager.acquire(key, now=1)
    with pytest.raises(RuntimeError, match="active"):
        manager.offload_cold(key, started_at=1, completed_at=2)
    manager.release(key, now=2)
    offload = manager.offload_cold(key, started_at=3, completed_at=4)
    assert offload.operation == "cold_offload"
    assert offload.bytes_transferred == 2048
    assert manager.resident_count == 0
    assert manager.admit(key, now=5).reason is (
        CohortAdmissionReason.OFFLOADED_RESTORE_REQUIRED
    )
    replacement = cohort_key("e")
    assert manager.admit(replacement, now=5).admitted
    assert manager.resident_count == 1
    assert manager.offloaded_count == 1
    manager.acquire(replacement, now=5)
    manager.release(replacement, now=5)
    with pytest.raises(MemoryError, match="host cohort tier is full"):
        manager.offload_cold(replacement, started_at=5, completed_at=6)
    with pytest.raises(MemoryError, match="no fixed cohort slab"):
        manager.restore_cold(key, started_at=6, completed_at=7)
    manager.reclaim(replacement, now=7, reason_code="test_reclaim")
    restore = manager.restore_cold(key, started_at=6, completed_at=7)
    assert restore.operation == "cold_restore"
    assert manager.resident_count == 1


def test_cold_offload_requires_an_explicit_bounded_host_tier() -> None:
    with pytest.raises(ValueError, match="bounded host tier"):
        BoundedCohortStateManager(
            capacity=1,
            slab_bytes=2048,
            tenant_quotas={"tenant-a": 1},
            ttl_seconds=100,
            offload_mode=CohortOffloadMode.COLD_INACTIVE_TIMED,
        )
