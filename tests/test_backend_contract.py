from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.runtime import backend as backend_module
from lightcone_spec.runtime import readiness as readiness_module
from lightcone_spec.runtime.attestation import TrustedAttesterPolicy
from lightcone_spec.runtime.backend import (
    DSPARK_NATIVE_HEAD_NAMES,
    BackendNotApplicable,
    BackendPayload,
    BackendRegistry,
    DFlashBackendContract,
    DSparkBackendContract,
    DSparkSelectorAuthority,
    Eagle3CompatibilityAuthority,
    EagleBackendContract,
    NextNBackendContract,
    NextNTp2DynamicAuthorityArtifact,
    NextNTwoModelTp2Authority,
    ProposalEvidence,
    Reconstruction,
    VerifiedEagle3CompatibilityAuthority,
    VerifiedEagle3E0ExecutionAuthority,
    VerifiedEagle3OfficialSelectorContentAuthority,
    VerifiedNextNTp2Authority,
    bind_eagle3_e0_execution_authority,
    dspark_composite_loss,
    dspark_conditional_survival_target,
    dspark_selector_candidate_ids,
    nextn_tp2_dynamic_proof_sha256,
    publish_nextn_tp2_dynamic_authority_artifact,
    require_eagle3_e0_execution_authority,
    validate_nextn_tp2_dynamic_authority_artifact,
    verify_formal_eagle3_compatibility_authority,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayStore,
    ControlArtifactAttestation,
)
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256,
    NATIVE_RUNTIME_QUALIFICATION_TESTS,
    NATIVE_RUNTIME_RELEASE_CAPABILITY,
    NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
    NativeRuntimeGpuProofReceipt,
    VerifiedNativeRuntimeGpuProof,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _verified_eagle3_selector_content(
    *, status: str = "COMPATIBLE"
) -> VerifiedEagle3OfficialSelectorContentAuthority:
    authority = object.__new__(VerifiedEagle3OfficialSelectorContentAuthority)
    root = Path.cwd().resolve()
    values = {
        "protocol_sha256": backend_module.EAGLE3_OFFICIAL_SELECTOR_CONTENT_PROTOCOL_SHA256,
        "stage": "E0",
        "backend": "EAGLE3",
        "task": "LiveCodeBench",
        "status": status,
        "reason_code": (
            "official_interface_match"
            if status == "COMPATIBLE"
            else "official_selector_unavailable"
        ),
        "target_member_id": "e0:eagle3:target",
        "drafter_member_id": "e0:eagle3:drafter",
        "target_model_id": "Qwen/Qwen3-4B",
        "drafter_model_id": "official/eagle3-qwen3-4b",
        "target_revision": "a" * 40,
        "drafter_revision": "b" * 40,
        "interface_sha256": "c" * 64,
        "source_repository": "https://example.invalid/official/eagle3",
        "source_commit": "d" * 40,
        "model_selector_sha256": "e" * 64,
        "selector_asset_path": str(root / "eagle3-selector.json"),
        "selector_asset_raw_sha256": "f" * 64,
        "selector_asset_semantic_sha256": "0" * 64,
        "target_snapshot_raw_sha256": "1" * 64,
        "target_snapshot_semantic_sha256": "2" * 64,
        "drafter_snapshot_raw_sha256": "3" * 64,
        "drafter_snapshot_semantic_sha256": "4" * 64,
        "prepared_content_receipt_path": str(root / "content-receipt.json"),
        "prepared_content_receipt_raw_sha256": "5" * 64,
        "prepared_content_receipt_sha256": "6" * 64,
        "prepared_content_authorization_sha256": "7" * 64,
        "root_manifest_sha256": "8" * 64,
    }
    for name, value in values.items():
        object.__setattr__(authority, name, value)
    authority.__post_init__()
    return authority


def _selector() -> DSparkSelectorAuthority:
    interface_sha256 = _sha("dspark-model-interface")
    parameter_names = tuple(
        sorted((*DSPARK_NATIVE_HEAD_NAMES, "layers.0.mlp.weight", "lm_head.weight"))
    )
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            {
                "schema_version": 1,
                "model_interface_sha256": interface_sha256,
                "parameter_names": parameter_names,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return DSparkSelectorAuthority(
        model_interface_sha256=interface_sha256,
        parameter_names=parameter_names,
        supplied_candidate_ids=dspark_selector_candidate_ids(
            model_interface_sha256=interface_sha256,
            parameter_inventory_sha256=inventory_sha256,
        ),
    )


def evidence() -> ProposalEvidence:
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    proposal = logits.softmax(dim=-1)
    return ProposalEvidence(
        backend="DSPARK",
        adapter_free_logits=logits.clone(),
        proposal_logits=logits,
        corrected_distribution=proposal,
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        teacher_rows=torch.tensor([[[0.75, 0.25], [0.25, 0.75]]]),
        predecessor_token_ids=torch.tensor([[7, 8]]),
        predecessor_embeddings=torch.randn(1, 2, 4),
        confidence=torch.zeros(1, 2),
        request_ids=("request-a",),
        cohort_sha256="a" * 64,
        source_adapter_version=3,
        payload=BackendPayload(
            schema="dspark-native-v2",
            values={
                "markov_w1_feature": torch.randn(1, 2, 4),
                "markov_w2_feature": torch.randn(1, 2, 4),
                "markov_w1_source": "inference_native",
                "markov_w2_source": "inference_native",
                "predecessor_source": "sampled_token",
                "predecessor_embedding_source": "embedding_lookup_of_sampled_token",
                "confidence_head_source": "inference_native",
                "selector_authority": _selector(),
                "selector_candidate_id": _selector().candidate_ids[0],
                "model_interface_sha256": _selector().model_interface_sha256,
                "parameter_inventory_sha256": (_selector().parameter_inventory_sha256),
                "native_head_mode": "full_w1_w2_confidence",
                "scheduler_mode": "native_scheduler",
                "fixed_verification_budget": None,
                "proposal_correction": "frozen_at_sampling",
            },
        ),
    )


def reconstruct(
    value: ProposalEvidence,
    delta: dict[str, torch.Tensor],
    already_applied: bool,
) -> Reconstruction:
    offset = delta.get("offset", torch.zeros_like(value.proposal_logits))
    logits = (
        value.proposal_logits if already_applied else value.proposal_logits + offset
    )
    return Reconstruction(
        proposal_logits=logits,
        corrected_distribution=logits.softmax(dim=-1),
        confidence=value.confidence,
    )


def test_dspark_contract_binds_native_features_and_reconstructs_once() -> None:
    value = evidence()
    registry = BackendRegistry((DSparkBackendContract(reconstruct),))
    result = registry.reconstruct(
        value,
        adapter_delta={"offset": torch.full_like(value.proposal_logits, 0.25)},
    )
    assert result.proposal_logits.shape == value.proposal_logits.shape
    assert bool(value.numerical_predicate())
    assert len(value.identity_sha256) == 64
    with pytest.raises(ValueError, match="double-count"):
        registry.reconstruct(
            value,
            adapter_delta={"offset": torch.ones_like(value.proposal_logits)},
            adapter_already_applied=True,
        )


def test_dspark_contract_rejects_placeholder_provenance() -> None:
    value = evidence()
    bad = ProposalEvidence(
        **{
            **value.__dict__,
            "payload": BackendPayload(
                schema="dspark-native-v2",
                values={
                    **value.payload.values,
                    "markov_w1_source": "placeholder",
                },
            ),
        }
    )
    with pytest.raises(ValueError, match="real inference"):
        DSparkBackendContract(reconstruct).validate_payload(bad)


def test_dspark_contract_requires_exact_56_selector_and_fixed_budget_pairing() -> None:
    value = evidence()
    fixed = ProposalEvidence(
        **{
            **value.__dict__,
            "payload": BackendPayload(
                schema="dspark-native-v2",
                values={
                    **value.payload.values,
                    "scheduler_mode": "fixed_budget",
                    "fixed_verification_budget": 8,
                },
            ),
        }
    )
    DSparkBackendContract(reconstruct).validate_payload(fixed)
    wrong = ProposalEvidence(
        **{
            **fixed.__dict__,
            "payload": BackendPayload(
                schema="dspark-native-v2",
                values={
                    **fixed.payload.values,
                    "selector_candidate_id": _sha("outside-selector"),
                },
            ),
        }
    )
    with pytest.raises(ValueError, match="outside the 56-candidate selector"):
        DSparkBackendContract(reconstruct).validate_payload(wrong)
    with pytest.raises(ValueError, match="exact 56-cell grid"):
        DSparkSelectorAuthority(
            model_interface_sha256=_selector().model_interface_sha256,
            parameter_names=tuple(
                sorted(
                    (
                        *DSPARK_NATIVE_HEAD_NAMES,
                        "layers.0.mlp.weight",
                        "lm_head.weight",
                    )
                )
            ),
            supplied_candidate_ids=_selector().candidate_ids[:-1],
        )
    with pytest.raises(ValueError, match="native confidence head"):
        DSparkBackendContract(reconstruct).validate_payload(
            ProposalEvidence(**{**value.__dict__, "confidence": None})
        )


def test_dspark_selector_reducer_rejects_foreign_grid_and_head_inventory() -> None:
    selector = _selector()
    parameters = tuple(
        sorted((*DSPARK_NATIVE_HEAD_NAMES, "layers.0.mlp.weight", "lm_head.weight"))
    )
    foreign = tuple(sorted((*selector.candidate_ids[:-1], _sha("foreign-rank-128"))))
    with pytest.raises(ValueError, match="exact 56-cell grid"):
        DSparkSelectorAuthority(
            model_interface_sha256=selector.model_interface_sha256,
            parameter_names=parameters,
            supplied_candidate_ids=foreign,
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        DSparkSelectorAuthority(
            model_interface_sha256=selector.model_interface_sha256,
            parameter_names=(*parameters, parameters[-1]),
            supplied_candidate_ids=selector.candidate_ids,
        )
    with pytest.raises(ValueError, match="lacks exact native heads"):
        DSparkSelectorAuthority(
            model_interface_sha256=selector.model_interface_sha256,
            parameter_names=tuple(
                name for name in parameters if name != "acceptance.confidence"
            ),
            supplied_candidate_ids=selector.candidate_ids,
        )
    with pytest.raises(ValueError, match="W1/W2/confidence heads"):
        DSparkSelectorAuthority(
            model_interface_sha256=selector.model_interface_sha256,
            parameter_names=parameters,
            supplied_candidate_ids=selector.candidate_ids,
            native_head_names=("markov.w1", "markov.w2", "foreign.confidence"),
        )


def test_dspark_confidence_target_is_detached_and_composite_loss_is_finite() -> None:
    teacher = torch.tensor([[[0.8, 0.2], [0.3, 0.7]]])
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    proposal = logits.softmax(dim=-1)
    target = dspark_conditional_survival_target(teacher, proposal)
    assert not target.requires_grad
    loss = dspark_composite_loss(
        teacher_distribution=teacher,
        proposal_distribution=proposal,
        confidence_logits=torch.zeros((1, 2), requires_grad=True),
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        confidence_weight=0.25,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


@pytest.mark.parametrize(
    ("backend", "schema", "payload", "contract"),
    [
        (
            "DFLASH",
            "dflash-native-v1",
            {
                "canvas_state": torch.zeros(1),
                "proposal_correction": "frozen_at_sampling",
            },
            lambda: DFlashBackendContract(reconstruct),
        ),
        (
            "EAGLE",
            "eagle-native-v1",
            {
                "tree_state": torch.zeros(1),
                "topk": 1,
                "proposal_correction": "frozen_at_sampling",
            },
            lambda: EagleBackendContract("EAGLE", reconstruct),
        ),
        (
            "NEXTN",
            "nextn-native-v2",
            {
                "mtp_hidden_state": torch.zeros(1, 2, 4),
                "mtp_teacher_rows": None,
                "mtp_valid_mask": None,
                "interface_sha256": "f" * 64,
                "source_adapter_version": 3,
                "target_revision": "a" * 40,
                "drafter_revision": "b" * 40,
                "topology_mode": "tp1_dp1",
                "tp2_model_authority": None,
                "proposal_correction": "frozen_at_sampling",
            },
            lambda: NextNBackendContract(reconstruct),
        ),
    ],
)
def test_registered_native_backends_use_one_common_evidence_envelope(
    backend: str,
    schema: str,
    payload: dict,
    contract,
) -> None:
    source = evidence()
    if backend == "NEXTN":
        payload = {
            **payload,
            "mtp_teacher_rows": source.teacher_rows,
            "mtp_valid_mask": source.valid_mask,
        }
    value = ProposalEvidence(
        **{
            **source.__dict__,
            "backend": backend,
            "confidence": None,
            "payload": BackendPayload(schema=schema, values=payload),
        }
    )
    registry = BackendRegistry((contract(),))
    result = registry.reconstruct(value, adapter_delta={})
    assert result.proposal_logits.shape == source.proposal_logits.shape


def _signed_eagle3_authority(
    private_key: Ed25519PrivateKey,
    *,
    status: str = "COMPATIBLE",
    reason_code: str = "official_interface_match",
) -> Eagle3CompatibilityAuthority:
    seed = Eagle3CompatibilityAuthority(
        schema_version=1,
        status=status,
        target_revision="a" * 40,
        drafter_revision="b" * 40,
        interface_sha256="c" * 64,
        source_commit="d" * 40,
        model_selector_sha256="e" * 64,
        reason_code=reason_code,
        signer_key_id="official-eagle3",
        signature_hex="0" * 128,
    )
    return replace(seed, signature_hex=private_key.sign(seed.message).hex())


def test_eagle3_requires_external_trusted_signature_and_preserves_na() -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    source = evidence()
    authority = _signed_eagle3_authority(private_key)
    payload = {
        "tree_state": torch.zeros(1),
        "topk": 1,
        "proposal_correction": "frozen_at_sampling",
        "compatibility_authority": authority,
        "target_revision": authority.target_revision,
        "drafter_revision": authority.drafter_revision,
    }
    value = ProposalEvidence(
        **{
            **source.__dict__,
            "backend": "EAGLE3",
            "confidence": None,
            "payload": BackendPayload(schema="eagle3-native-v2", values=payload),
        }
    )
    with pytest.raises(BackendNotApplicable, match="formal.*unavailable"):
        EagleBackendContract("EAGLE3", reconstruct).validate_payload(value)
    contract = EagleBackendContract(
        "EAGLE3",
        reconstruct,
        diagnostic_trusted_public_keys={"official-eagle3": public_key},
    )
    with pytest.raises(BackendNotApplicable, match="formal.*unavailable"):
        contract.validate_payload(value)
    tampered = replace(authority, reason_code="caller_changed_compatibility")
    tampered_value = ProposalEvidence(
        **{
            **value.__dict__,
            "payload": BackendPayload(
                schema="eagle3-native-v2",
                values={**payload, "compatibility_authority": tampered},
            ),
        }
    )
    with pytest.raises(ValueError, match="signature is invalid"):
        contract.validate_payload(tampered_value)

    unavailable = _signed_eagle3_authority(
        private_key,
        status="N/A",
        reason_code="official_interface_not_supported",
    )
    na_value = ProposalEvidence(
        **{
            **value.__dict__,
            "payload": BackendPayload(
                schema="eagle3-native-v2",
                values={**payload, "compatibility_authority": unavailable},
            ),
        }
    )
    with pytest.raises(BackendNotApplicable) as error:
        contract.validate_payload(na_value)
    assert error.value.reason_code == (
        "eagle3_formal_compatibility_authority_unavailable"
    )

    with pytest.raises(TypeError, match="only come from release verification"):
        VerifiedEagle3CompatibilityAuthority(
            authority=authority,
            official_selector_content=_verified_eagle3_selector_content(),
            trusted_policy_sha256="1" * 64,
            control_envelope_sha256="2" * 64,
            challenge_reservation_sha256="3" * 64,
            _verification_tag=object(),
        )


def test_formal_eagle3_rejects_invalid_inner_signature_before_control(
    tmp_path,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    policy = TrustedAttesterPolicy(
        policy_id="release-eagle3-compatibility-v1",
        trusted_attesters=(("official-eagle3", "official-eagle3", public_key_sha256),),
        public_keys=(
            (public_key_sha256, base64.b64encode(public_key).decode("ascii")),
        ),
    )
    policy.validate()
    authority = Eagle3CompatibilityAuthority(
        schema_version=1,
        status="N/A",
        target_revision="a" * 40,
        drafter_revision="b" * 40,
        interface_sha256="c" * 64,
        source_commit="d" * 40,
        model_selector_sha256="e" * 64,
        reason_code="official_selector_unavailable",
        signer_key_id="official-eagle3",
        signature_hex="0" * 128,
    )
    control = object.__new__(ControlArtifactAttestation)
    object.__setattr__(
        control,
        "deployment_policy_authorization",
        SimpleNamespace(bundle=SimpleNamespace(trusted_attester_policy=policy)),
    )

    with pytest.raises(ValueError, match="signature is invalid"):
        verify_formal_eagle3_compatibility_authority(
            authority,
            control_attestation=control,
            replay_store=ChallengeReplayStore(str(tmp_path.resolve())),
            expected_inventory_sha256="1" * 64,
            expected_hardware_envelope_sha256="2" * 64,
            expected_target_revision=authority.target_revision,
            expected_drafter_revision=authority.drafter_revision,
            expected_interface_sha256=authority.interface_sha256,
            expected_source_commit=authority.source_commit,
            expected_model_selector_sha256=authority.model_selector_sha256,
            now_ns=1,
        )


def _verified_eagle3_e0_inputs() -> tuple[
    VerifiedEagle3CompatibilityAuthority,
    VerifiedNativeRuntimeGpuProof,
]:
    compatibility_raw = Eagle3CompatibilityAuthority(
        schema_version=1,
        status="COMPATIBLE",
        target_revision="a" * 40,
        drafter_revision="b" * 40,
        interface_sha256="c" * 64,
        source_commit="d" * 40,
        model_selector_sha256="e" * 64,
        reason_code="official_interface_match",
        signer_key_id="official-eagle3",
        signature_hex="0" * 128,
    )
    compatibility = VerifiedEagle3CompatibilityAuthority(
        authority=compatibility_raw,
        official_selector_content=_verified_eagle3_selector_content(),
        trusted_policy_sha256="1" * 64,
        control_envelope_sha256="2" * 64,
        challenge_reservation_sha256="3" * 64,
        _verification_tag=backend_module._VERIFIED_EAGLE3_COMPATIBILITY_SENTINEL,
    )
    capability = NATIVE_RUNTIME_RELEASE_CAPABILITY
    receipt = NativeRuntimeGpuProofReceipt(
        schema_version=1,
        kind="lightcone_native_runtime_gpu_proof",
        suite_id="eagle3_tp1",
        topology_mode="tp1_dp1",
        topology_sha256="4" * 64,
        runner_protocol_sha256=NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[
            "eagle3_tp1"
        ],
        assignment_sha256="5" * 64,
        qualification_observation_sha256="6" * 64,
        source_capability_sha256=capability.sha256,
        pinned_sglang_commit=capability.pinned_sglang_commit,
        patched_sglang_tree=capability.patched_sglang_tree,
        semantic_patch_sha256=capability.semantic_patch_sha256,
        run_nonce_sha256="7" * 64,
        qualification_authority_sha256=(NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256),
        source_identity_sha256="8" * 64,
        inventory_sha256="9" * 64,
        gpu_uuids=("GPU-eagle3",),
        hardware_envelope_sha256="a" * 64,
        junit_xml_sha256="b" * 64,
        test_names=NATIVE_RUNTIME_QUALIFICATION_TESTS["eagle3_tp1"],
        tests_collected=8,
        tests_passed=8,
        tests_failed=0,
        tests_errored=0,
        tests_skipped=0,
    )
    proof = VerifiedNativeRuntimeGpuProof(
        receipt=receipt,
        receipt_raw_sha256="c" * 64,
        trusted_policy_sha256="d" * 64,
        challenge_sha256="e" * 64,
        control_envelope_sha256="f" * 64,
        challenge_reservation_sha256="1" * 64,
        _verification_tag=readiness_module._VERIFIED_NATIVE_GPU_PROOF_SENTINEL,
    )
    return compatibility, proof


def test_eagle3_e0_bridge_requires_exact_compatible_suite_and_selector() -> None:
    compatibility, proof = _verified_eagle3_e0_inputs()
    authority = bind_eagle3_e0_execution_authority(
        method="l0",
        verified_compatibility_authority=compatibility,
        verified_native_gpu_proof=proof,
        expected_target_revision=compatibility.target_revision,
        expected_drafter_revision=compatibility.drafter_revision,
        expected_task=compatibility.task,
        expected_target_member_id=compatibility.target_member_id,
        expected_drafter_member_id=compatibility.drafter_member_id,
        expected_source_identity_sha256=proof.source_identity_sha256,
        expected_inventory_sha256=proof.inventory_sha256,
        expected_gpu_uuids=proof.gpu_uuids,
    )
    assert type(authority) is VerifiedEagle3E0ExecutionAuthority
    assert (
        require_eagle3_e0_execution_authority(
            claimed_execution_authority_sha256=authority.sha256,
            claimed_compatibility_authority_sha256=(
                authority.compatibility_authority_sha256
            ),
            claimed_model_selector_sha256=authority.model_selector_sha256,
            claimed_native_gpu_proof_sha256=authority.native_gpu_receipt_sha256,
            verified_execution_authority=authority,
            expected_method="l0",
            expected_target_revision=authority.target_revision,
            expected_drafter_revision=authority.drafter_revision,
            expected_source_identity_sha256=authority.native_source_identity_sha256,
            expected_inventory_sha256=authority.inventory_sha256,
            expected_gpu_uuids=authority.gpu_uuids,
        )
        is authority
    )
    with pytest.raises(BackendNotApplicable, match="unavailable"):
        require_eagle3_e0_execution_authority(
            claimed_execution_authority_sha256=authority.sha256,
            claimed_compatibility_authority_sha256=(
                authority.compatibility_authority_sha256
            ),
            claimed_model_selector_sha256="0" * 64,
            claimed_native_gpu_proof_sha256=authority.native_gpu_receipt_sha256,
            verified_execution_authority=authority,
            expected_method="l0",
            expected_target_revision=authority.target_revision,
            expected_drafter_revision=authority.drafter_revision,
            expected_source_identity_sha256=authority.native_source_identity_sha256,
            expected_inventory_sha256=authority.inventory_sha256,
            expected_gpu_uuids=authority.gpu_uuids,
        )
    with pytest.raises(BackendNotApplicable, match="unavailable"):
        require_eagle3_e0_execution_authority(
            claimed_execution_authority_sha256=authority.sha256,
            claimed_compatibility_authority_sha256=(
                authority.compatibility_authority_sha256
            ),
            claimed_model_selector_sha256=authority.model_selector_sha256,
            claimed_native_gpu_proof_sha256=authority.native_gpu_receipt_sha256,
            verified_execution_authority=None,
            expected_method="l0",
            expected_target_revision=authority.target_revision,
            expected_drafter_revision=authority.drafter_revision,
            expected_source_identity_sha256=authority.native_source_identity_sha256,
            expected_inventory_sha256=authority.inventory_sha256,
            expected_gpu_uuids=authority.gpu_uuids,
        )


def test_nextn_tp2_two_model_authority_cannot_self_enable_release() -> None:
    source = evidence()
    authority = NextNTwoModelTp2Authority(
        schema_version=1,
        interface_sha256="f" * 64,
        target_revision="a" * 40,
        drafter_revision="b" * 40,
        target_shard_manifest_sha256="c" * 64,
        drafter_shard_manifest_sha256="d" * 64,
        topology_sha256="e" * 64,
        source_adapter_version=source.source_adapter_version,
        status="GPU_VERIFIED",
        gpu_proof_sha256=_sha("caller-gpu-proof"),
    )
    value = ProposalEvidence(
        **{
            **source.__dict__,
            "backend": "NEXTN",
            "confidence": None,
            "payload": BackendPayload(
                schema="nextn-native-v2",
                values={
                    "mtp_hidden_state": torch.zeros(1, 2, 4),
                    "mtp_teacher_rows": source.teacher_rows,
                    "mtp_valid_mask": source.valid_mask,
                    "interface_sha256": authority.interface_sha256,
                    "source_adapter_version": source.source_adapter_version,
                    "target_revision": authority.target_revision,
                    "drafter_revision": authority.drafter_revision,
                    "topology_mode": "tp2_dp1",
                    "tp2_model_authority": authority,
                    "proposal_correction": "frozen_at_sampling",
                },
            ),
        }
    )
    with pytest.raises(BackendNotApplicable) as error:
        NextNBackendContract(reconstruct).validate_payload(value)
    assert error.value.reason_code == "nextn_tp2_native_gpu_authority_unavailable"


def _verified_nextn_tp2(
    authority: NextNTwoModelTp2Authority,
) -> VerifiedNextNTp2Authority:
    verified = object.__new__(VerifiedNextNTp2Authority)
    values = {
        "artifact_sha256": _sha("nextn-artifact"),
        "authority_sha256": authority.sha256,
        "interface_sha256": authority.interface_sha256,
        "target_revision": authority.target_revision,
        "drafter_revision": authority.drafter_revision,
        "target_shard_manifest_sha256": authority.target_shard_manifest_sha256,
        "drafter_shard_manifest_sha256": authority.drafter_shard_manifest_sha256,
        "topology_sha256": authority.topology_sha256,
        "source_adapter_version": authority.source_adapter_version,
        "native_gpu_proof_sha256": _sha("native-nextn-proof"),
        "distributed_gpu_proof_sha256": _sha("distributed-nextn-proof"),
        "content_verification_receipt_sha256": _sha("nextn-content-proof"),
        "inventory_sha256": _sha("nextn-inventory"),
        "registry_sha256": _sha("nextn-registry"),
        "root_manifest_sha256": _sha("nextn-root"),
        "gpu_uuids": ("GPU-A", "GPU-B"),
    }
    for name, value in values.items():
        object.__setattr__(verified, name, value)
    return verified


def _nextn_tp2_evidence(
    authority: NextNTwoModelTp2Authority,
) -> ProposalEvidence:
    source = evidence()
    return ProposalEvidence(
        **{
            **source.__dict__,
            "backend": "NEXTN",
            "confidence": None,
            "payload": BackendPayload(
                schema="nextn-native-v2",
                values={
                    "mtp_hidden_state": torch.zeros(1, 2, 4),
                    "mtp_teacher_rows": source.teacher_rows,
                    "mtp_valid_mask": source.valid_mask,
                    "interface_sha256": authority.interface_sha256,
                    "source_adapter_version": source.source_adapter_version,
                    "target_revision": authority.target_revision,
                    "drafter_revision": authority.drafter_revision,
                    "topology_mode": "tp2_dp1",
                    "tp2_model_authority": authority,
                    "proposal_correction": "frozen_at_sampling",
                },
            ),
        }
    )


def test_nextn_tp2_verified_authority_reaches_live_reconstruction() -> None:
    source = evidence()
    authority = NextNTwoModelTp2Authority(
        schema_version=1,
        interface_sha256="f" * 64,
        target_revision="a" * 40,
        drafter_revision="b" * 40,
        target_shard_manifest_sha256="c" * 64,
        drafter_shard_manifest_sha256="d" * 64,
        topology_sha256="e" * 64,
        source_adapter_version=source.source_adapter_version,
        status="GPU_VERIFIED",
        gpu_proof_sha256=_sha("dynamic-proof-dag"),
    )
    calls: list[str] = []

    def live_reconstruct(
        value: ProposalEvidence,
        delta: dict[str, torch.Tensor],
        already_applied: bool,
    ) -> Reconstruction:
        calls.append(value.identity_sha256)
        return reconstruct(value, delta, already_applied)

    registry = BackendRegistry(
        (
            NextNBackendContract(
                live_reconstruct,
                verified_tp2_authority=_verified_nextn_tp2(authority),
            ),
        )
    )
    result = registry.reconstruct(
        _nextn_tp2_evidence(authority),
        adapter_delta={},
    )
    assert result.proposal_logits.shape == (1, 2, 2)
    assert len(calls) == 1


def test_nextn_tp2_rejects_foreign_verified_authority_before_live_hook() -> None:
    source = evidence()
    authority = NextNTwoModelTp2Authority(
        schema_version=1,
        interface_sha256="f" * 64,
        target_revision="a" * 40,
        drafter_revision="b" * 40,
        target_shard_manifest_sha256="c" * 64,
        drafter_shard_manifest_sha256="d" * 64,
        topology_sha256="e" * 64,
        source_adapter_version=source.source_adapter_version,
        status="GPU_VERIFIED",
        gpu_proof_sha256=_sha("dynamic-proof-dag"),
    )
    foreign = replace(authority, target_shard_manifest_sha256="9" * 64)
    calls: list[bool] = []

    def must_not_run(
        value: ProposalEvidence,
        delta: dict[str, torch.Tensor],
        already_applied: bool,
    ) -> Reconstruction:
        calls.append(True)
        return reconstruct(value, delta, already_applied)

    contract = NextNBackendContract(
        must_not_run,
        verified_tp2_authority=_verified_nextn_tp2(foreign),
    )
    with pytest.raises(ValueError, match="verified two-model authority"):
        contract.reconstruct(
            _nextn_tp2_evidence(authority),
            adapter_delta={},
            adapter_already_applied=False,
        )
    assert calls == []


def test_nextn_tp2_verified_token_rejects_direct_construction() -> None:
    source = evidence()
    authority = NextNTwoModelTp2Authority(
        schema_version=1,
        interface_sha256="f" * 64,
        target_revision="a" * 40,
        drafter_revision="b" * 40,
        target_shard_manifest_sha256="c" * 64,
        drafter_shard_manifest_sha256="d" * 64,
        topology_sha256="e" * 64,
        source_adapter_version=source.source_adapter_version,
        status="GPU_VERIFIED",
        gpu_proof_sha256=_sha("dynamic-proof-dag"),
    )
    with pytest.raises(TypeError, match="verifier-owned"):
        VerifiedNextNTp2Authority(
            artifact_sha256=_sha("nextn-artifact"),
            authority=authority,
            target_model_id="target/model",
            drafter_model_id="drafter/model",
            native_gpu_proof_sha256=_sha("native-nextn-proof"),
            distributed_gpu_proof_sha256=_sha("distributed-nextn-proof"),
            content_verification_receipt_sha256=_sha("nextn-content-proof"),
            inventory_sha256=_sha("nextn-inventory"),
            registry_sha256=_sha("nextn-registry"),
            root_manifest_sha256=_sha("nextn-root"),
            gpu_uuids=("GPU-A", "GPU-B"),
            _verification_tag=object(),
        )


def test_nextn_tp2_dynamic_artifact_cannot_promote_missing_gpu_proofs(
    tmp_path,
) -> None:
    nested_paths = tuple(
        (tmp_path / name).resolve()
        for name in ("native.json", "distributed.json", "content.json")
    )
    for path in nested_paths:
        path.write_text("{}\n", encoding="utf-8")
    from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

    bindings = tuple(CanonicalJsonProofBinding.bind(path) for path in nested_paths)
    proof_sha256 = nextn_tp2_dynamic_proof_sha256(
        native_gpu_proof=bindings[0],
        distributed_gpu_proof=bindings[1],
        content_verification_receipt=bindings[2],
        target_member_id="nextn:target",
        drafter_member_id="nextn:drafter",
    )
    authority = NextNTwoModelTp2Authority(
        schema_version=1,
        interface_sha256="f" * 64,
        target_revision="a" * 40,
        drafter_revision="b" * 40,
        target_shard_manifest_sha256="c" * 64,
        drafter_shard_manifest_sha256="d" * 64,
        topology_sha256="e" * 64,
        source_adapter_version=3,
        status="GPU_VERIFIED",
        gpu_proof_sha256=proof_sha256,
    )
    artifact_path = (tmp_path / "nextn-authority.json").resolve()
    binding = publish_nextn_tp2_dynamic_authority_artifact(
        str(artifact_path),
        authority=authority,
        native_gpu_proof_artifact_path=str(nested_paths[0]),
        distributed_gpu_proof_artifact_path=str(nested_paths[1]),
        content_verification_receipt_path=str(nested_paths[2]),
        target_member_id="nextn:target",
        drafter_member_id="nextn:drafter",
    )
    reopened = NextNTp2DynamicAuthorityArtifact.from_dict(binding.reopen())
    assert reopened.authority.sha256 == authority.sha256
    with pytest.raises(ValueError, match="native runtime proof artifact fields"):
        validate_nextn_tp2_dynamic_authority_artifact(
            str(artifact_path),
            expected_inventory_sha256=_sha("nextn-inventory"),
            expected_registry_sha256=_sha("nextn-registry"),
            expected_root_manifest_sha256=_sha("nextn-root"),
            expected_interface_sha256=authority.interface_sha256,
            expected_topology_sha256=authority.topology_sha256,
            expected_source_adapter_version=3,
            expected_target_member_id="nextn:target",
            expected_drafter_member_id="nextn:drafter",
            now_ns=1,
        )


def test_backend_payload_identity_binds_scalar_provenance() -> None:
    first = BackendPayload(
        schema="dflash-native-v1",
        values={
            "canvas_state": torch.zeros(1),
            "proposal_correction": "frozen_at_sampling",
        },
    )
    second = BackendPayload(
        schema="dflash-native-v1",
        values={
            "canvas_state": torch.zeros(1),
            "proposal_correction": "changed",
        },
    )
    assert first.sha256 != second.sha256
