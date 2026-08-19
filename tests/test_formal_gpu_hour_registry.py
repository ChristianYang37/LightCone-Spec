from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_control_attestation import _root_binding
from test_formal_registry_integration import _candidate_coverage, _complete_coverage
from test_gpu_hour_authority import (
    _case,
    _inventory,
    _materialization,
    _prospective_projection_case,
    _reservation,
    _runtime_manifest,
    _subject_and_binding,
)
from test_native_terminal_provider import (
    FakeAdminTransport,
    _server_request,
)
from test_native_terminal_provider import (
    _binding as _native_binding,
)
from test_native_terminal_provider import (
    _run as _run_native_terminal,
)
from test_preflight_authority import _formal_source

from lightcone_spec.cli.main import _load_bound_json, _write_json, main
from lightcone_spec.experiments import formal_gpu_hour_registry, gpu_hour_authority
from lightcone_spec.experiments.downstream_stage_authority import (
    E3B_POWER_PREFIX_PROTOCOL_SHA256,
    E3bPowerPrefixReceipt,
    FormalFamilyPowerCommitment,
    SignedE3bPowerPrefixReceipt,
)
from lightcone_spec.experiments.formal_gpu_hour_registry import (
    FORMAL_GPU_HOUR_REGISTRY_PROTOCOL_SHA256,
    FormalStageGpuHourVerificationReceipt,
    FormalStudyGpuHourEstimate,
    aggregate_formal_study_gpu_hours,
    reserve_formal_stage_gpu_hour_verification_receipt,
)
from lightcone_spec.experiments.formal_protocol import (
    SignedProtocolLock,
    content_sha256,
)
from lightcone_spec.experiments.formal_registry import (
    FormalRegistryVerificationReceipt,
    extend_formal_registry_verification_receipt,
    formal_registry_verification_receipt_to_dict,
    formal_runtime_authority_manifest_to_dict,
    reserve_formal_registry_verification_receipt,
    signed_stage_gpu_hour_to_dict,
    stage_gpu_hour_envelope_from_dict,
    stage_gpu_hour_envelope_to_dict,
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.formal_registry_layers import (
    bind_formal_registry_layer_artifact,
    load_formal_registry_verification_receipt_path,
    publish_formal_registry_layer_artifact,
    publish_formal_registry_replay_proof_shards,
)
from lightcone_spec.experiments.formal_slo_metrics import (
    FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
)
from lightcone_spec.experiments.gpu_hour_authority import (
    FORMAL_SERVING_EXECUTION_PROOF_PROTOCOL_SHA256,
    PREFLIGHT_GPU_HOUR_PROTOCOL_SHA256,
    LifecycleGpuHourSourceManifest,
    PreflightGpuHourObservation,
    PreflightGpuHourSourceManifest,
    publish_formal_serving_execution_proof_artifact,
    validate_formal_serving_execution_proof_artifact,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    SignedStageCoverageReceipt,
    SignedStageGpuHourEnvelope,
    SignedStageMaterializationReceipt,
    StageGpuHourEnvelope,
    StageMaterializationReceipt,
    materialize_e3a,
    materialize_preflight,
)
from lightcone_spec.experiments.statistics import PowerSizingPlan
from lightcone_spec.orchestration.native_terminal import (
    NATIVE_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256,
    NativeTerminalProvider,
    build_native_terminal_external_control_binding,
    publish_candidate_state_replay_proof_artifacts,
    validate_candidate_state_replay_proof_artifact,
)
from lightcone_spec.runtime import release_trust_root as root_module
from lightcone_spec.runtime.attestation import (
    NO_TRUSTED_ATTESTERS,
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
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    ControlArtifactSubject,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    deployment_policy_subject_sha256,
)
from lightcone_spec.runtime.relocatable_evidence import (
    materialize_relocatable_evidence_bundle,
)
from lightcone_spec.runtime.scientific_source_validation import (
    publish_scientific_source_validation_artifact,
)

NOW_NS = 2_000_000_000
HARDWARE_SHA256 = hashlib.sha256(b"formal-gpu-hour-hardware").hexdigest()


def _public_bytes(private: Ed25519PrivateKey) -> bytes:
    return private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _bundle(private: Ed25519PrivateKey) -> TrustedAttesterPolicyBundle:
    public = _public_bytes(private)
    fingerprint = hashlib.sha256(public).hexdigest()
    return TrustedAttesterPolicyBundle(
        schema_version=1,
        kind="lightcone_trusted_attester_policy_bundle",
        bundle_id="formal-gpu-hour-registry-test-v1",
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
        hardware_envelope_sha256_allowlist=(HARDWARE_SHA256,),
        trusted_attester_policy=TrustedAttesterPolicy(
            policy_id="formal-gpu-hour-registry-test-v1",
            trusted_attesters=(
                ("validation-signer", "validation-signer-key", fingerprint),
            ),
            public_keys=((fingerprint, base64.b64encode(public).decode("ascii")),),
        ),
    )


def _challenge(subject_sha256: str, nonce: int) -> AttestationChallenge:
    return AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=f"formal-gpu-hour-{nonce}",
        nonce_base64=base64.b64encode(bytes([nonce]) * 32).decode("ascii"),
        subject_sha256=subject_sha256,
        issued_ns=1_600_000_000,
        expires_ns=2_600_000_000,
    )


def _sign(private: Ed25519PrivateKey, payload: object, nonce: int):
    digest = content_sha256(payload)
    challenge = _challenge(digest, nonce)
    public = _public_bytes(private)
    attestation = SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id="validation-signer",
        key_id="validation-signer-key",
        environment="release",
        public_key_base64=base64.b64encode(public).decode("ascii"),
        challenge_sha256=challenge.sha256,
        payload_sha256=digest,
        signature_base64=base64.b64encode(
            private.sign(attestation_message(challenge, payload_sha256=digest))
        ).decode("ascii"),
    )
    return digest, challenge, attestation


def _e3b_family_commitments(
    power_sizing: PowerSizingPlan,
) -> tuple[FormalFamilyPowerCommitment, ...]:
    rows = []
    for index in range(96):
        dimensions = (("family", f"family-{index:03d}"),)
        family_sha256 = content_sha256(
            {
                "stage": "E3b",
                "model": "model",
                "task": "task",
                "dimensions": list(dimensions),
            }
        )
        rows.append(
            FormalFamilyPowerCommitment(
                schema_version=1,
                stage="E3b",
                model="model",
                task="task",
                family_dimensions=dimensions,
                family_sha256=family_sha256,
                slo_goodput_protocol_sha256=FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
                pilot_goodput_observation_sha256s=tuple(
                    sorted(
                        (
                            block,
                            role,
                            content_sha256(
                                {
                                    "family": index,
                                    "block": block,
                                    "role": role,
                                }
                            ),
                        )
                        for block in range(4)
                        for role in ("Static", "TTS", "LightCone")
                    )
                ),
                power_sizing=power_sizing,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.family_sha256))


def _proof_wrapped_signed_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    label: str,
    artifact_type: str,
    signed: (
        SignedProtocolLock
        | SignedStageMaterializationReceipt
        | SignedStageCoverageReceipt
    ),
) -> Path:
    """Publish a compact proof wrapper for registry-layer transport tests.

    The synthetic reducer is deliberately confined to tests.  Production
    registry code still calls the real source-validation dispatcher and never
    accepts the raw signed row used to construct this fixture.
    """

    from lightcone_spec.runtime import scientific_source_validation

    remote_root = (tmp_path / f"{label}-remote-proof").resolve()
    remote_root.mkdir(mode=0o700)
    proof_path = (remote_root / "proof.json").resolve()
    publish_canonical_json_no_replace(
        proof_path,
        {"schema_version": 1, "kind": f"test_{label}_reducer_proof"},
    )
    previous = scientific_source_validation._expected_payload

    def expected(kind: str, path: str | Path, now_ns: int):
        if kind == artifact_type and Path(path) == proof_path:
            return signed.payload
        return previous(kind, path, now_ns)

    monkeypatch.setattr(
        scientific_source_validation,
        "_expected_payload",
        expected,
    )
    local_root = (tmp_path / f"{label}-local-proof").resolve()
    local_root.mkdir(mode=0o700)
    bundle = materialize_relocatable_evidence_bundle(
        remote_root=remote_root,
        entry_paths=(proof_path,),
        local_root=local_root,
    )
    source_path = (tmp_path / f"{label}-source-validation.json").resolve()
    publish_scientific_source_validation_artifact(
        artifact_type=artifact_type,
        proof_bundle_path=bundle.absolute_path,
        now_ns=NOW_NS,
        output_path=source_path,
    )
    source = CanonicalJsonProofBinding.bind(source_path)
    body = {
        "schema_version": 1,
        "kind": "lightcone_scientific_signed_proof_wrapper",
        "artifact_type": artifact_type,
        "source_validation_artifact": source.to_dict(),
        "payload_sha256": signed.payload_sha256,
        "challenge": asdict(signed.challenge),
        "attestation": asdict(signed.attestation),
        "signed_artifact_sha256": signed.sha256,
    }
    wrapper_path = (tmp_path / f"{label}-signed-proof-wrapper.json").resolve()
    publish_canonical_json_no_replace(
        wrapper_path,
        {**body, "wrapper_sha256": content_sha256(body)},
    )
    return wrapper_path


def _reservation_for_challenges(
    tmp_path: Path,
    *,
    label: str,
    challenges: tuple[str, ...],
    reserved_ns: int = NOW_NS,
) -> ChallengeReplayReservationBinding:
    canonical_challenges = tuple(sorted(set(challenges)))
    canonical = {
        "schema_version": 2,
        "kind": "lightcone_control_challenge_reservation",
        "reserved_ns": reserved_ns,
        "challenge_sha256s": list(canonical_challenges),
    }
    identity = content_sha256(canonical)
    body = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    directory = tmp_path / label
    directory.mkdir()
    path = (directory / f"reservation-{identity}.json").resolve()
    path.write_bytes(body)
    path.chmod(0o600)
    return ChallengeReplayReservationBinding(
        schema_version=1,
        kind="lightcone_challenge_replay_reservation_binding",
        path=str(path),
        reservation_sha256=identity,
        raw_sha256=hashlib.sha256(body).hexdigest(),
        size=len(body),
        reserved_ns=reserved_ns,
        challenge_sha256s=canonical_challenges,
    )


def _deployment(
    root_private: Ed25519PrivateKey,
    *,
    root_binding,
    bundle: TrustedAttesterPolicyBundle,
    inventory_sha256: str,
    nonce: int,
) -> DeploymentPolicyAuthorization:
    subject = deployment_policy_subject_sha256(
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=inventory_sha256,
        bundle_sha256=bundle.sha256,
    )
    challenge = _challenge(subject, nonce)
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
    private: Ed25519PrivateKey,
    *,
    root_binding,
    bundle: TrustedAttesterPolicyBundle,
    authorization: DeploymentPolicyAuthorization,
    artifact_type: str,
    artifact_sha256: str,
    protocol_sha256: str,
    lineage_sha256: str,
    nonce: int,
) -> ControlArtifactAttestation:
    subject = ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type=artifact_type,
        artifact_sha256=artifact_sha256,
        protocol_sha256=protocol_sha256,
        registry_sha256=build_industrial_registry().sha256,
        lineage_sha256=lineage_sha256,
    )
    challenge = _challenge(subject.sha256, nonce)
    public = _public_bytes(private)
    signature = private.sign(
        attestation_message(challenge, payload_sha256=artifact_sha256)
    )
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
            public_key_base64=base64.b64encode(public).decode("ascii"),
            challenge_sha256=challenge.sha256,
            payload_sha256=artifact_sha256,
            signature_base64=base64.b64encode(signature).decode("ascii"),
        ),
    )


def _registry_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    inventory,
    runtime_manifest,
    root_private: Ed25519PrivateKey,
    controller_private: Ed25519PrivateKey,
    root_binding,
    bundle,
    tts_calibration_authority_sha256: str | None = None,
):
    from test_formal_dispatch import _protocol_lock

    lock = replace(
        _protocol_lock(),
        offline_release_trust_root_sha256=root_binding.semantic_sha256,
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
        **(
            {}
            if tts_calibration_authority_sha256 is None
            else {
                "tts_calibration_authority_sha256": (tts_calibration_authority_sha256)
            }
        ),
    )
    signed_lock = SignedProtocolLock(lock, *_sign(controller_private, lock, 1))
    root_lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": ((signed_lock.sha256, "dispatch"),),
        }
    )
    authorization = _deployment(
        root_private,
        root_binding=root_binding,
        bundle=bundle,
        inventory_sha256=inventory.sha256,
        nonce=2,
    )
    root_control = _control(
        controller_private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=authorization,
        artifact_type="dispatch",
        artifact_sha256=signed_lock.sha256,
        protocol_sha256=lock.sha256,
        lineage_sha256=root_lineage,
        nonce=3,
    )
    replay_directory = tmp_path / "registry-replay"
    replay_directory.mkdir()
    store = ChallengeReplayStore(str(replay_directory.resolve()))
    root = reserve_formal_registry_verification_receipt(
        signed_lock,
        control_attestation=root_control,
        expected_inventory_sha256=inventory.sha256,
        replay_store=store,
        now_ns=NOW_NS,
    )
    materialization = materialize_preflight(
        protocol_lock_sha256=lock.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    candidate_authorization = _deployment(
        root_private,
        root_binding=root_binding,
        bundle=bundle,
        inventory_sha256=inventory.sha256,
        nonce=20,
    )
    raw_paths: list[str] = []
    proof_paths: list[str] = []
    candidate_controls: list[ControlArtifactAttestation] = []
    native_bindings = []
    for index, method in enumerate(("tts", "l0")):
        binding = _native_binding(
            method=method,
            warmup=(),
            identity_suffix=f"gpu-hour-{method}",
            run_nonce_sha256=hashlib.sha256(
                f"gpu-hour-{method}-nonce".encode()
            ).hexdigest(),
        )
        request = _server_request("score-0", inputs=(1,), outputs=(2, 3))
        transport = FakeAdminTransport(
            binding=binding,
            warmup=(),
            scored=(request,),
        )
        provider = NativeTerminalProvider(
            transport,
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
        )
        _, _, _, terminal = asyncio.run(
            _run_native_terminal(transport, provider=provider)
        )
        artifact = terminal.to_artifact(warmup_requests=())
        raw_path = (tmp_path / f"candidate-{method}-raw.json").resolve()
        raw_path.write_text(
            json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        external = build_native_terminal_external_control_binding(
            artifact,
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
            inventory_sha256=inventory.sha256,
            expected_binding=binding,
        )
        candidate_controls.append(
            _control(
                controller_private,
                root_binding=root_binding,
                bundle=bundle,
                authorization=candidate_authorization,
                artifact_type="non_serving_terminal",
                artifact_sha256=external.sha256,
                protocol_sha256=NATIVE_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256,
                lineage_sha256=external.lineage_sha256,
                nonce=21 + index,
            )
        )
        raw_paths.append(str(raw_path))
        proof_paths.append(str((tmp_path / f"candidate-{method}-proof.json").resolve()))
        native_bindings.append(binding)
    candidate_replay_directory = tmp_path / "candidate-replay"
    candidate_replay_directory.mkdir()
    publish_candidate_state_replay_proof_artifacts(
        tuple(raw_paths),
        control_attestations=tuple(candidate_controls),
        replay_store=ChallengeReplayStore(str(candidate_replay_directory.resolve())),
        expected_inventory_sha256=inventory.sha256,
        expected_registry_sha256=build_industrial_registry().sha256,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        now_ns=NOW_NS,
        proof_artifact_paths=tuple(proof_paths),
        expected_bindings=tuple(native_bindings),
    )
    pointers = {
        pointer.method: pointer
        for pointer in (
            validate_candidate_state_replay_proof_artifact(
                path,
                expected_inventory_sha256=inventory.sha256,
                expected_registry_sha256=build_industrial_registry().sha256,
                expected_root_manifest_sha256=root_binding.semantic_sha256,
                now_ns=NOW_NS,
            )
            for path in proof_paths
        )
    }
    base_candidate = _candidate_coverage(materialization)
    tts_pointer = pointers["tts"]
    l0_pointer = pointers["l0"]
    tts_update = tts_pointer.updates[0]
    l0_update = l0_pointer.updates[0]
    tts_observation = replace(
        base_candidate.tts_observations[0],
        run_id=tts_pointer.run_id,
        native_replay_pointer_sha256=tts_pointer.semantic_commitment_sha256,
        source_state_sha256=tts_update.source_state_sha256,
        candidate_bytes_sha256=tts_update.candidate_bytes_sha256,
        optimizer_state_bytes_sha256=tts_update.optimizer_state_bytes_sha256,
        proposal_evidence_sha256=tts_update.proposal_evidence_sha256,
    )
    l0_observation = replace(
        base_candidate.l0_naive_observations[0],
        run_id=l0_pointer.run_id,
        native_replay_pointer_sha256=l0_pointer.semantic_commitment_sha256,
        source_state_sha256=l0_update.source_state_sha256,
        candidate_bytes_sha256=l0_update.candidate_bytes_sha256,
        optimizer_state_bytes_sha256=l0_update.optimizer_state_bytes_sha256,
        proposal_evidence_sha256=l0_update.proposal_evidence_sha256,
    )
    terminal_pair = replace(
        base_candidate.terminal_pairs[0],
        tts_run_id=tts_pointer.run_id,
        l0_naive_run_id=l0_pointer.run_id,
        tts_native_replay_pointer_sha256=(tts_pointer.semantic_commitment_sha256),
        l0_naive_native_replay_pointer_sha256=(l0_pointer.semantic_commitment_sha256),
        proposal_evidence_sha256=tts_update.proposal_evidence_sha256,
        tts_terminal_receipt_sha256=tts_pointer.terminal_sha256,
        l0_naive_terminal_receipt_sha256=l0_pointer.terminal_sha256,
    )
    candidate_coverage = replace(
        base_candidate,
        tts_native_replay_pointer_sha256=tts_pointer.semantic_commitment_sha256,
        l0_naive_native_replay_pointer_sha256=(l0_pointer.semantic_commitment_sha256),
        tts_observations=(tts_observation,),
        l0_naive_observations=(l0_observation,),
        terminal_pairs=(terminal_pair,),
    )
    coverage = _complete_coverage(
        materialization,
        candidate_coverages=(candidate_coverage,),
    )
    signed_materialization = SignedStageMaterializationReceipt(
        materialization, *_sign(controller_private, materialization, 4)
    )
    signed_coverage = SignedStageCoverageReceipt(
        coverage, *_sign(controller_private, coverage, 5)
    )
    rows = tuple(
        sorted(
            (
                (signed_materialization.sha256, "dispatch"),
                (signed_coverage.sha256, "rank_aggregate"),
            )
        )
    )
    lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": rows,
            "prior_registry_verification_receipt_sha256": root.sha256,
        }
    )
    extension_authorization = _deployment(
        root_private,
        root_binding=root_binding,
        bundle=bundle,
        inventory_sha256=inventory.sha256,
        nonce=6,
    )
    controls = tuple(
        _control(
            controller_private,
            root_binding=root_binding,
            bundle=bundle,
            authorization=extension_authorization,
            artifact_type=artifact_type,
            artifact_sha256=digest,
            protocol_sha256=lock.sha256,
            lineage_sha256=lineage,
            nonce=nonce,
        )
        for nonce, (digest, artifact_type) in enumerate(rows, start=7)
    )
    receipt = extend_formal_registry_verification_receipt(
        root,
        appended_signed_materializations=(signed_materialization,),
        appended_signed_coverage=(signed_coverage,),
        control_attestations=controls,
        candidate_replay_proof_artifact_paths=tuple(proof_paths),
        replay_store=store,
        now_ns=NOW_NS,
    )
    root_layer_path = (tmp_path / "registry-root-layer.json").resolve()
    materialization_path = _proof_wrapped_signed_row(
        tmp_path,
        monkeypatch,
        label="preflight-materialization",
        artifact_type="stage-materialization",
        signed=signed_materialization,
    )
    coverage_path = _proof_wrapped_signed_row(
        tmp_path,
        monkeypatch,
        label="preflight-coverage",
        artifact_type="stage-coverage",
        signed=signed_coverage,
    )
    replay_shard_path = (tmp_path / "registry-replay-shard-0.json").resolve()
    layer_path = (tmp_path / "registry-layer.json").resolve()
    protocol_lock_path = _proof_wrapped_signed_row(
        tmp_path,
        monkeypatch,
        label="protocol-lock",
        artifact_type="protocol-lock",
        signed=signed_lock,
    )
    root_layer = bind_formal_registry_layer_artifact(
        root,
        prior_layer_path=None,
        signed_protocol_lock_path=protocol_lock_path,
        signed_materialization_paths=(),
        signed_coverage_paths=(),
        formal_stage_prefix_paths=(),
    )
    publish_formal_registry_layer_artifact(root_layer, root_layer_path)
    publish_formal_registry_replay_proof_shards(
        receipt,
        prior_receipt=root,
        candidate_replay_proof_paths=tuple(proof_paths),
        shard_output_paths=(replay_shard_path,),
    )
    layer = bind_formal_registry_layer_artifact(
        receipt,
        prior_layer_path=root_layer_path,
        signed_materialization_paths=(materialization_path,),
        signed_coverage_paths=(coverage_path,),
        formal_stage_prefix_paths=(),
        candidate_replay_proof_shard_paths=(replay_shard_path,),
    )
    publish_formal_registry_layer_artifact(layer, layer_path)
    assert (
        load_formal_registry_verification_receipt_path(
            layer_path,
            now_ns=NOW_NS,
        )
        == receipt
    )
    return lock, receipt, materialization, layer_path


def _preflight_gpu_hour_source(
    tmp_path: Path,
    *,
    lock,
    materialization,
    coverage,
    inventory,
    runtime_manifest,
) -> tuple[PreflightGpuHourSourceManifest, StageGpuHourEnvelope, Path]:
    source_authority = replace(
        _formal_source(lock.registry_sha256),
        inventory_sha256=inventory.sha256,
        release_root_manifest_sha256=lock.offline_release_trust_root_sha256,
    )
    remote_path = (tmp_path / "preflight-remote-raw.json").resolve()
    publish_canonical_json_no_replace(
        remote_path,
        {"schema_version": 1, "kind": "preflight_gpu_hour_test_raw"},
    )
    observations = []
    phase_counts = {"compile": 0, "exactness": 0, "interference": 0}
    for index, cell in enumerate(materialization.cells):
        if cell.task == "environment_and_patch_preflight":
            phase_kind = "compile"
        elif cell.task == "exactness_memory_telemetry_preflight":
            phase_kind = "exactness"
        else:
            phase_kind = "interference"
        phase_counts[phase_kind] += 1
        timing_path = (tmp_path / f"preflight-timing-{index}.json").resolve()
        timing_payload = {
            "schema_version": 1,
            "kind": f"preflight_{phase_kind}_timing_test_receipt",
            "materialized_cell_id": cell.cell_id,
        }
        publish_canonical_json_no_replace(timing_path, timing_payload)
        start = 10_000_000_000 + index * 10_000_000_000
        dimensions = dict(cell.dimensions)
        observations.append(
            PreflightGpuHourObservation(
                materialized_cell_id=cell.cell_id,
                registry_cell_id=dimensions["registry_cell_id"],
                phase_kind=phase_kind,
                timing_proof=CanonicalJsonProofBinding.bind(timing_path),
                timing_authority_sha256=content_sha256(timing_payload),
                execution_identity_sha256=content_sha256(
                    {
                        "kind": "preflight_test_execution_identity",
                        "cell_id": cell.cell_id,
                    }
                ),
                control_envelope_sha256=content_sha256(
                    {"kind": "preflight_test_control", "cell_id": cell.cell_id}
                ),
                replay_reservation=_reservation(
                    tmp_path,
                    label=f"preflight-timing-{index}",
                    reserved_ns=NOW_NS,
                ),
                gpu_uuids=(inventory.devices[index % 2].uuid,),
                process_started_ns=start,
                process_finished_ns=start + 3_000_000_000,
                gpu_released_ns=start + 4_000_000_000,
                evidence_finished_ns=start + 5_000_000_000,
                wave_index=index,
            )
        )
    assert phase_counts == {"compile": 1, "exactness": 1, "interference": 8}
    ordered = tuple(
        sorted(observations, key=lambda row: (row.wave_index, row.materialized_cell_id))
    )
    manifest = PreflightGpuHourSourceManifest(
        schema_version=1,
        kind="preflight_gpu_hour_source_manifest",
        protocol_sha256=PREFLIGHT_GPU_HOUR_PROTOCOL_SHA256,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_member_sha256=runtime_manifest.member(
            "gpu_hour_budget_reducer"
        ).sha256,
        materialization_receipt_sha256=materialization.sha256,
        stage_coverage_receipt_sha256=coverage.sha256,
        final_evidence_sha256=content_sha256(
            {"kind": "preflight_gpu_hour_test_final_evidence"}
        ),
        remote_raw_receipt=CanonicalJsonProofBinding.bind(remote_path),
        source_authority=source_authority,
        activation_sha256=content_sha256(
            {"kind": "preflight_gpu_hour_test_activation"}
        ),
        pointer_coverage_sha256=content_sha256(
            {"kind": "preflight_gpu_hour_test_pointer_coverage"}
        ),
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=HARDWARE_SHA256,
        observations=ordered,
        schedule_sha256=gpu_hour_authority._preflight_schedule_sha256(ordered),
    )
    source_path = (tmp_path / "preflight-gpu-hour-source.json").resolve()
    publish_canonical_json_no_replace(source_path, manifest.to_dict())
    envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        signed_pilot_receipt_sha256=manifest.sha256,
        schedule_sha256=manifest.schedule_sha256,
        estimate=gpu_hour_authority._estimate_preflight_manifest(manifest),
    )
    return manifest, envelope, source_path


def _extend_registry_with_e3a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    lock,
    prior_receipt: FormalRegistryVerificationReceipt,
    prior_layer_path: Path,
    preflight_materialization,
    inventory,
    root_private: Ed25519PrivateKey,
    controller_private: Ed25519PrivateKey,
    root_binding,
    bundle: TrustedAttesterPolicyBundle,
) -> tuple[FormalRegistryVerificationReceipt, StageMaterializationReceipt, Path]:
    preflight_coverage = next(
        row.payload
        for row in prior_receipt.cumulative_signed_coverage
        if row.payload.materialization_receipt_sha256
        == preflight_materialization.sha256
    )
    e3a = materialize_e3a(
        registry_verification_receipt=prior_receipt,
        protocol_lock=lock,
        preflight_materialization=preflight_materialization,
        preflight_coverage=preflight_coverage,
        now_ns=NOW_NS,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    signed_e3a = SignedStageMaterializationReceipt(
        e3a,
        *_sign(controller_private, e3a, 30),
    )
    signed_rows = ((signed_e3a.sha256, "dispatch"),)
    lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": signed_rows,
            "prior_registry_verification_receipt_sha256": prior_receipt.sha256,
        }
    )
    authorization = _deployment(
        root_private,
        root_binding=root_binding,
        bundle=bundle,
        inventory_sha256=inventory.sha256,
        nonce=31,
    )
    control = _control(
        controller_private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=authorization,
        artifact_type="dispatch",
        artifact_sha256=signed_e3a.sha256,
        protocol_sha256=lock.sha256,
        lineage_sha256=lineage,
        nonce=32,
    )
    replay_directory = tmp_path / "e3a-registry-replay"
    replay_directory.mkdir()
    receipt = extend_formal_registry_verification_receipt(
        prior_receipt,
        appended_signed_materializations=(signed_e3a,),
        control_attestations=(control,),
        replay_store=ChallengeReplayStore(str(replay_directory.resolve())),
        now_ns=NOW_NS,
    )
    signed_path = _proof_wrapped_signed_row(
        tmp_path,
        monkeypatch,
        label="e3a-materialization",
        artifact_type="stage-materialization",
        signed=signed_e3a,
    )
    layer_path = (tmp_path / "registry-e3a-layer.json").resolve()
    layer = bind_formal_registry_layer_artifact(
        receipt,
        prior_layer_path=prior_layer_path,
        signed_materialization_paths=(signed_path,),
        signed_coverage_paths=(),
        formal_stage_prefix_paths=(),
    )
    publish_formal_registry_layer_artifact(layer, layer_path)
    assert (
        load_formal_registry_verification_receipt_path(
            layer_path,
            now_ns=NOW_NS,
        )
        == receipt
    )
    return receipt, e3a, layer_path


def _available_envelope(
    *,
    protocol_lock_sha256: str,
    materialization_receipt_sha256: str,
    source_sha256: str,
    schedule_sha256: str,
    compute_gpu_hours: float = 1.0,
    wall_hours: float = 1.0,
) -> StageGpuHourEnvelope:
    values = {
        "source_pilot_receipt_sha256": source_sha256,
        "source_schedule_sha256": schedule_sha256,
        "source_materialization_receipt_sha256": (materialization_receipt_sha256),
        "source_inventory_gpu_count": 2,
        "compute_gpu_hours": float(compute_gpu_hours),
        "reserved_gpu_hours": float(2 * wall_hours + 0.1 * compute_gpu_hours),
        "estimated_wall_hours": float(wall_hours),
        "retry_reserve_gpu_hours": float(0.1 * compute_gpu_hours),
        "profile_reserve_gpu_hours": 0.0,
        "evidence_reserve_gpu_hours": 0.0,
    }
    estimate = GpuHourEstimate(
        status="AVAILABLE",
        derivation_sha256=content_sha256(values),
        **values,
    )
    return StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=protocol_lock_sha256,
        materialization_receipt_sha256=materialization_receipt_sha256,
        signed_pilot_receipt_sha256=source_sha256,
        schedule_sha256=schedule_sha256,
        estimate=estimate,
    )


def _stage_gpu_hour_receipt(
    tmp_path: Path,
    *,
    label: str,
    nonce: int,
    lock,
    registry_receipt: FormalRegistryVerificationReceipt,
    registry_layer_path: Path,
    materialization: StageMaterializationReceipt,
    source_path: Path,
    source,
    envelope: StageGpuHourEnvelope,
    runtime_manifest,
    inventory,
    root_private: Ed25519PrivateKey,
    controller_private: Ed25519PrivateKey,
    root_binding,
    bundle: TrustedAttesterPolicyBundle,
    prospective_pilot_materialization_path: Path | None = None,
) -> FormalStageGpuHourVerificationReceipt:
    signed_envelope = SignedStageGpuHourEnvelope(
        envelope,
        *_sign(controller_private, envelope, nonce),
    )
    source_binding = CanonicalJsonProofBinding.bind(source_path)
    pilot_binding = (
        None
        if prospective_pilot_materialization_path is None
        else CanonicalJsonProofBinding.bind(prospective_pilot_materialization_path)
    )
    lineage = formal_gpu_hour_registry._control_lineage_sha256(
        registry_receipt_sha256=registry_receipt.sha256,
        signed_envelope_sha256=signed_envelope.sha256,
        source_manifest=source_binding,
        runtime_authority_manifest_sha256=runtime_manifest.sha256,
        inventory_sha256=inventory.sha256,
        prospective_pilot_materialization=pilot_binding,
    )
    authorization = _deployment(
        root_private,
        root_binding=root_binding,
        bundle=bundle,
        inventory_sha256=inventory.sha256,
        nonce=nonce + 1,
    )
    control = _control(
        controller_private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=authorization,
        artifact_type="rank_aggregate",
        artifact_sha256=signed_envelope.sha256,
        protocol_sha256=FORMAL_GPU_HOUR_REGISTRY_PROTOCOL_SHA256,
        lineage_sha256=lineage,
        nonce=nonce + 2,
    )
    reservation = _reservation_for_challenges(
        tmp_path,
        label=f"stage-receipt-{label}",
        challenges=(
            signed_envelope.challenge.sha256,
            control.challenge.sha256,
            control.deployment_policy_authorization.challenge.sha256,
        ),
    )
    receipt = FormalStageGpuHourVerificationReceipt(
        schema_version=3,
        kind="lightcone_formal_stage_gpu_hour_verification_receipt",
        verified_ns=NOW_NS,
        stage=materialization.stage,
        registry_receipt_source=CanonicalJsonProofBinding.bind(registry_layer_path),
        signed_envelope=signed_envelope,
        source_manifest=source_binding,
        formal_runtime_authority_manifest=runtime_manifest,
        inventory=inventory,
        prospective_pilot_materialization=pilot_binding,
        control_attestation=control,
        reservation=reservation,
    )
    assert receipt.registry_receipt == registry_receipt
    assert type(source).from_dict(receipt.source_manifest.reopen()) == source
    return receipt


def test_aggregate_rejects_replay_shared_by_distinct_real_source_wrappers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_private = Ed25519PrivateKey.generate()
    controller_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    bundle = _bundle(controller_private)
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    runtime_manifest = _runtime_manifest()
    inventory = _inventory()
    lock, initial_registry, first_materialization, initial_layer_path = (
        _registry_receipt(
            tmp_path,
            monkeypatch,
            inventory=inventory,
            runtime_manifest=runtime_manifest,
            root_private=root_private,
            controller_private=controller_private,
            root_binding=root_binding,
            bundle=bundle,
        )
    )
    registry_receipt, second_materialization, registry_layer_path = (
        _extend_registry_with_e3a(
            tmp_path,
            monkeypatch,
            lock=lock,
            prior_receipt=initial_registry,
            prior_layer_path=initial_layer_path,
            preflight_materialization=first_materialization,
            inventory=inventory,
            root_private=root_private,
            controller_private=controller_private,
            root_binding=root_binding,
            bundle=bundle,
        )
    )
    (tmp_path / "first-source").mkdir()
    (tmp_path / "second-source").mkdir()
    first_source, first_envelope, first_path = _preflight_gpu_hour_source(
        tmp_path / "first-source",
        lock=lock,
        materialization=first_materialization,
        coverage=SimpleNamespace(
            sha256=content_sha256({"kind": "first_source_coverage"})
        ),
        inventory=inventory,
        runtime_manifest=runtime_manifest,
    )
    *_, nested_source_path, _nested_envelope = _case(
        tmp_path / "second-source" / "nested",
        gangs=(("GPU-0",),),
        starts=(10_000_000_000,),
        monkeypatch=monkeypatch,
    )
    nested = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(nested_source_path).reopen()
    )
    shared_challenge = first_source.observations[
        0
    ].replay_reservation.challenge_sha256s[0]
    nested_observation = replace(
        nested.observations[0],
        materialized_cell_id=second_materialization.cells[0].cell_id,
        execution_replay_reservation=_reservation_for_challenges(
            tmp_path,
            label="cross-stage-second-real-wrapper",
            challenges=(
                shared_challenge,
                content_sha256({"kind": "second_wrapper_unique_challenge"}),
            ),
        ),
    )
    nested = replace(
        nested,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_member_sha256=runtime_manifest.member(
            "gpu_hour_budget_reducer"
        ).sha256,
        materialization_receipt_sha256=second_materialization.sha256,
        inventory_sha256=inventory.sha256,
        observations=(nested_observation,),
        schedule_sha256=gpu_hour_authority._schedule_sha256((nested_observation,)),
    )
    rebound_nested_path = (tmp_path / "second-source" / "nested-rebound.json").resolve()
    publish_canonical_json_no_replace(rebound_nested_path, nested.to_dict())
    second_source = gpu_hour_authority._derive_staged_prospective_gpu_hour_source(
        protocol_lock=lock,
        runtime_authority_member_sha256=runtime_manifest.member(
            "gpu_hour_budget_reducer"
        ).sha256,
        materialization=second_materialization,
        inventory=inventory,
        completed_source_binding=CanonicalJsonProofBinding.bind(rebound_nested_path),
        completed_source=nested,
    )
    second_path = (tmp_path / "second-source-rebound.json").resolve()
    publish_canonical_json_no_replace(second_path, second_source.to_dict())
    second_envelope = _available_envelope(
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=second_materialization.sha256,
        source_sha256=second_source.sha256,
        schedule_sha256=second_source.mapping_sha256,
    )
    first_receipt = _stage_gpu_hour_receipt(
        tmp_path,
        label="first",
        nonce=50,
        lock=lock,
        registry_receipt=registry_receipt,
        registry_layer_path=registry_layer_path,
        materialization=first_materialization,
        source_path=first_path,
        source=first_source,
        envelope=first_envelope,
        runtime_manifest=runtime_manifest,
        inventory=inventory,
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=bundle,
    )
    second_receipt = _stage_gpu_hour_receipt(
        tmp_path,
        label="second",
        nonce=60,
        lock=lock,
        registry_receipt=registry_receipt,
        registry_layer_path=registry_layer_path,
        materialization=second_materialization,
        source_path=second_path,
        source=second_source,
        envelope=second_envelope,
        runtime_manifest=runtime_manifest,
        inventory=inventory,
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=bundle,
    )
    source_by_receipt = {
        id(first_receipt): first_source,
        id(second_receipt): second_source,
    }
    monkeypatch.setattr(
        FormalStageGpuHourVerificationReceipt,
        "revalidate",
        lambda self, *, current_ns: source_by_receipt[id(self)],
    )
    with pytest.raises(ValueError, match="reuses replay challenge"):
        aggregate_formal_study_gpu_hours(
            registry_receipt=registry_receipt,
            stage_receipts=(first_receipt, second_receipt),
            current_ns=NOW_NS + 1,
        )


def test_registry_receipt_revalidates_candidate_replay_proof_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_private = Ed25519PrivateKey.generate()
    controller_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    bundle = _bundle(controller_private)
    monkeypatch.setattr(
        root_module, "load_source_release_ed25519_root", lambda: root_binding
    )
    inventory = _inventory()
    lock, receipt, materialization, _registry_layer_path = _registry_receipt(
        tmp_path,
        monkeypatch,
        inventory=inventory,
        runtime_manifest=_runtime_manifest(),
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=bundle,
    )

    receipt.revalidate(current_ns=NOW_NS + 1)

    assert receipt.signed_protocol_lock.payload == lock
    assert receipt.cumulative_signed_materializations[-1].payload == materialization
    assert receipt.manifest.candidate_replay_proofs
    assert all(
        row.proof_artifact.absolute_path
        for row in receipt.manifest.candidate_replay_proofs
    )


def test_cross_stage_evidence_cannot_rewrap_one_replay_challenge() -> None:
    labels = formal_gpu_hour_registry._FORMAL_STUDY_EVIDENCE_IDENTITY_LABELS
    observed = {label: set() for label in labels}
    first = {
        label: {hashlib.sha256(f"first:{label}".encode()).hexdigest()}
        for label in labels
    }
    second = {
        label: {hashlib.sha256(f"second:{label}".encode()).hexdigest()}
        for label in labels
    }
    second["replay challenge"] = set(first["replay challenge"])
    formal_gpu_hour_registry._reserve_unique_evidence_identities(observed, first)
    with pytest.raises(ValueError, match="reuses replay challenge"):
        formal_gpu_hour_registry._reserve_unique_evidence_identities(observed, second)


def test_registry_identity_reducer_deep_accounts_staged_projection_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_private = Ed25519PrivateKey.generate()
    controller_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    bundle = _bundle(controller_private)
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    runtime_manifest = _runtime_manifest()
    inventory = _inventory()
    lock, initial_registry, preflight, initial_layer_path = _registry_receipt(
        tmp_path,
        monkeypatch,
        inventory=inventory,
        runtime_manifest=runtime_manifest,
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=bundle,
    )
    registry_receipt, materialization, registry_layer_path = _extend_registry_with_e3a(
        tmp_path,
        monkeypatch,
        lock=lock,
        prior_receipt=initial_registry,
        prior_layer_path=initial_layer_path,
        preflight_materialization=preflight,
        inventory=inventory,
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=bundle,
    )
    (
        _fixture_lock,
        _fixture_runtime_manifest,
        _fixture_inventory,
        _fixture_materialization,
        _proof_inputs,
        _verified,
        completed_source_path,
        _actual_envelope,
    ) = _case(
        tmp_path / "registry-staged-source",
        gangs=(("GPU-0",), ("GPU-1",)),
        starts=(10_000_000_000, 10_000_000_000),
        monkeypatch=monkeypatch,
    )
    completed = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(completed_source_path).reopen()
    )
    completed_observations = tuple(
        sorted(
            (
                replace(row, materialized_cell_id=cell.cell_id)
                for row, cell in zip(
                    completed.observations,
                    materialization.cells[:2],
                    strict=True,
                )
            ),
            key=lambda row: (row.wave_index, row.materialized_cell_id),
        )
    )
    completed = replace(
        completed,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_member_sha256=runtime_manifest.member(
            "gpu_hour_budget_reducer"
        ).sha256,
        materialization_receipt_sha256=materialization.sha256,
        inventory_sha256=inventory.sha256,
        observations=completed_observations,
        schedule_sha256=gpu_hour_authority._schedule_sha256(completed_observations),
    )
    rebound_completed_path = (tmp_path / "registry-staged-completed.json").resolve()
    publish_canonical_json_no_replace(rebound_completed_path, completed.to_dict())
    staged_source_path = (tmp_path / "registry-staged-projection.json").resolve()
    source = gpu_hour_authority._derive_staged_prospective_gpu_hour_source(
        protocol_lock=lock,
        runtime_authority_member_sha256=runtime_manifest.member(
            "gpu_hour_budget_reducer"
        ).sha256,
        materialization=materialization,
        inventory=inventory,
        completed_source_binding=CanonicalJsonProofBinding.bind(rebound_completed_path),
        completed_source=completed,
    )
    publish_canonical_json_no_replace(staged_source_path, source.to_dict())
    envelope = _available_envelope(
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        source_sha256=source.sha256,
        schedule_sha256=source.mapping_sha256,
    )
    receipt = _stage_gpu_hour_receipt(
        tmp_path,
        label="staged-identity",
        nonce=70,
        lock=lock,
        registry_receipt=registry_receipt,
        registry_layer_path=registry_layer_path,
        materialization=materialization,
        source_path=staged_source_path,
        source=source,
        envelope=envelope,
        runtime_manifest=runtime_manifest,
        inventory=inventory,
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=bundle,
    )
    identities = formal_gpu_hour_registry._formal_stage_evidence_identities(
        receipt,
        source,
    )
    assert identities["materialized cell"] == {
        cell.cell_id for cell in materialization.cells
    }
    assert identities["actual source materialized cell"] == {
        row.materialized_cell_id for row in completed_observations
    }
    assert identities["prospective mapping"] == {source.mapping_sha256}
    assert len(identities["source manifest path"]) == 2


def test_study_aggregate_charges_prospective_pilot_and_final_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_private = Ed25519PrivateKey.generate()
    controller_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    bundle = _bundle(controller_private)
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    runtime_manifest = _runtime_manifest()
    inventory = _inventory()
    lock, registry_receipt, final, registry_layer_path = _registry_receipt(
        tmp_path,
        monkeypatch,
        inventory=inventory,
        runtime_manifest=runtime_manifest,
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=bundle,
    )
    (
        _pilot_lock,
        _pilot_runtime_manifest,
        _pilot_inventory,
        pilot,
        pilot_source,
        _pilot_source_path,
        _pilot_envelope,
        _preliminary_final,
    ) = _prospective_projection_case(tmp_path / "pilot", monkeypatch)
    pilot = replace(pilot, protocol_lock_sha256=lock.sha256)
    pilot_source = replace(
        pilot_source,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_member_sha256=runtime_manifest.member(
            "gpu_hour_budget_reducer"
        ).sha256,
        materialization_receipt_sha256=pilot.sha256,
        inventory_sha256=inventory.sha256,
    )
    pilot_materialization_path = (tmp_path / "pilot-materialization.json").resolve()
    publish_canonical_json_no_replace(
        pilot_materialization_path,
        stage_materialization_receipt_to_dict(pilot),
    )
    pilot_source_path = (tmp_path / "pilot-lifecycle-source.json").resolve()
    publish_canonical_json_no_replace(
        pilot_source_path,
        pilot_source.to_dict(),
    )

    power_sizing = PowerSizingPlan(
        status="READY",
        pilot_block_ids=tuple(f"excluded-pilot-{index}" for index in range(4)),
        selected_final_blocks=12,
        minimum_final_blocks=12,
        maximum_final_blocks=20,
        target_power=0.8,
        family_alpha=0.05,
        adjusted_alpha=0.025,
        minimum_relative_effect=0.03,
        minimum_log_effect=0.029558802241544398,
        pilot_log_standard_deviations=(
            ("LightCone-Static", 0.01),
            ("LightCone-TTS", 0.01),
        ),
        power_grid=(),
    )
    power = E3bPowerPrefixReceipt(
        schema_version=2,
        protocol_lock_sha256=lock.sha256,
        registry_sha256=lock.registry_sha256,
        pilot_materialization_receipt_sha256=pilot.sha256,
        pilot_coverage_receipt_sha256=content_sha256(
            {"kind": "aggregate_pilot_coverage"}
        ),
        evidence_manifest_sha256=content_sha256({"kind": "aggregate_pilot_evidence"}),
        inventory_sha256=inventory.sha256,
        protocol_sha256=E3B_POWER_PREFIX_PROTOCOL_SHA256,
        family_power_commitments=_e3b_family_commitments(power_sizing),
        selected_final_blocks=12,
        selected_final_prefix=tuple(range(4, 16)),
    )
    signed_power = SignedE3bPowerPrefixReceipt(
        power,
        *_sign(controller_private, power, 80),
    )

    def cost(category: str) -> gpu_hour_authority.ProspectiveGpuHourCost:
        return gpu_hour_authority.ProspectiveGpuHourCost(
            category=category,  # type: ignore[arg-type]
            cell_count=1,
            compute_gpu_ns=3_600_000_000_000,
            provider_base_reserved_gpu_ns=7_200_000_000_000,
            wall_ns=3_600_000_000_000,
            retry_reserve_gpu_ns=360_000_000_000,
            profile_reserve_gpu_ns=0,
            evidence_reserve_gpu_ns=0,
        )

    source = gpu_hour_authority.ProspectiveGpuHourSourceManifest(
        schema_version=1,
        kind="prospective_gpu_hour_source_manifest",
        protocol_sha256=gpu_hour_authority.PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_member_sha256=runtime_manifest.member(
            "gpu_hour_budget_reducer"
        ).sha256,
        stage="E3b",
        final_materialization_receipt_sha256=final.sha256,
        pilot_materialization_receipt_sha256=pilot.sha256,
        pilot_source_manifest=CanonicalJsonProofBinding.bind(pilot_source_path),
        one_shot_source_manifest=None,
        prospective_authority_sha256=power.sha256,
        signed_power_authority_sha256=signed_power.sha256,
        signed_power_challenge_sha256=signed_power.challenge.sha256,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=pilot_source.hardware_envelope_sha256,
        mapping_sha256=content_sha256({"kind": "projection-mapping"}),
        costs=(cost("actual_tuning"), cost("projected_final")),
    )
    estimate = gpu_hour_authority._estimate_prospective_manifest(source)
    source_path = (tmp_path / "prospective-source.json").resolve()
    publish_canonical_json_no_replace(source_path, source.to_dict())
    envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=final.sha256,
        signed_pilot_receipt_sha256=source.sha256,
        schedule_sha256=source.mapping_sha256,
        estimate=estimate,
    )
    receipt = _stage_gpu_hour_receipt(
        tmp_path,
        label="prospective",
        nonce=90,
        lock=lock,
        registry_receipt=registry_receipt,
        registry_layer_path=registry_layer_path,
        materialization=final,
        source_path=source_path,
        source=source,
        envelope=envelope,
        runtime_manifest=runtime_manifest,
        inventory=inventory,
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=bundle,
        prospective_pilot_materialization_path=pilot_materialization_path,
    )
    monkeypatch.setattr(
        FormalStageGpuHourVerificationReceipt,
        "revalidate",
        lambda _self, *, current_ns: source,
    )
    aggregate = aggregate_formal_study_gpu_hours(
        registry_receipt=registry_receipt,
        stage_receipts=(receipt,),
        current_ns=NOW_NS,
    )
    assert aggregate.status == "COMPLETE"
    assert aggregate.covered_materialization_sha256s == (final.sha256,)
    assert pilot.sha256 not in aggregate.covered_materialization_sha256s
    assert aggregate.compute_gpu_hours == estimate.compute_gpu_hours
    assert aggregate.compute_gpu_hours == pytest.approx(2.0)


def test_formal_serving_execution_proof_binds_exact_e3a_cell_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_formal_dispatch import _protocol_lock

    root_private = Ed25519PrivateKey.generate()
    controller_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    bundle = _bundle(controller_private)
    monkeypatch.setattr(
        root_module, "load_source_release_ed25519_root", lambda: root_binding
    )
    inventory = _inventory()
    runtime_manifest = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        offline_release_trust_root_sha256=root_binding.semantic_sha256,
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    materialization = _materialization(lock.sha256, 2)
    cell, foreign_cell = materialization.cells
    binding, _native, _lifecycle_path = _subject_and_binding(
        tmp_path=tmp_path,
        lock=lock,
        runtime_manifest=runtime_manifest,
        materialization=materialization,
        inventory=inventory,
        cell=cell,
        gpu_uuids=("GPU-0",),
        suffix="authentic-execution",
    )
    object.__setattr__(binding, "hardware_envelope_sha256", HARDWARE_SHA256)
    payload = gpu_hour_authority._execution_proof_payload(binding)
    lineage = gpu_hour_authority._execution_proof_lineage_sha256(
        protocol_lock_sha256=payload.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(payload.runtime_authority_manifest_sha256),
        materialization_receipt_sha256=payload.materialization_receipt_sha256,
        materialized_cell_id=payload.materialized_cell_id,
        inventory_sha256=payload.inventory_sha256,
        execution_binding_sha256=payload.execution_binding_sha256,
    )
    control = _control(
        controller_private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=_deployment(
            root_private,
            root_binding=root_binding,
            bundle=bundle,
            inventory_sha256=inventory.sha256,
            nonce=100,
        ),
        artifact_type="dispatch",
        artifact_sha256=payload.sha256,
        protocol_sha256=FORMAL_SERVING_EXECUTION_PROOF_PROTOCOL_SHA256,
        lineage_sha256=lineage,
        nonce=101,
    )
    replay_directory = (tmp_path / "authentic-execution-replay").resolve()
    replay_directory.mkdir()
    store = ChallengeReplayStore(str(replay_directory))
    artifact_path = (tmp_path / "authentic-execution-proof.json").resolve()
    publish_formal_serving_execution_proof_artifact(
        binding,
        control_attestation=control,
        replay_store=store,
        output_path=str(artifact_path),
        now_ns=NOW_NS,
    )

    validated = validate_formal_serving_execution_proof_artifact(
        str(artifact_path),
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        inventory=inventory,
        expected_cell_id=cell.cell_id,
        now_ns=NOW_NS + 1,
    )
    assert validated.execution_binding_sha256 == binding.sha256
    with pytest.raises(ValueError, match="lineage differs"):
        validate_formal_serving_execution_proof_artifact(
            str(artifact_path),
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            expected_cell_id=foreign_cell.cell_id,
            now_ns=NOW_NS + 1,
        )
    with pytest.raises(ValueError, match="already consumed"):
        publish_formal_serving_execution_proof_artifact(
            binding,
            control_attestation=control,
            replay_store=store,
            output_path=str(tmp_path / "replayed-execution-proof.json"),
            now_ns=NOW_NS,
        )


def test_operator_cli_materializes_registered_prospective_stage_without_scalars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        lock,
        runtime_manifest,
        inventory,
        pilot_materialization,
        pilot_source,
        pilot_source_path,
        pilot_envelope,
        preliminary_final,
    ) = _prospective_projection_case(tmp_path / "prospective", monkeypatch)
    power_sizing = PowerSizingPlan(
        status="READY",
        pilot_block_ids=tuple(f"excluded-pilot-{index}" for index in range(4)),
        selected_final_blocks=12,
        minimum_final_blocks=12,
        maximum_final_blocks=20,
        target_power=0.8,
        family_alpha=0.05,
        adjusted_alpha=0.025,
        minimum_relative_effect=0.03,
        minimum_log_effect=0.029558802241544398,
        pilot_log_standard_deviations=(
            ("LightCone-Static", 0.01),
            ("LightCone-TTS", 0.01),
        ),
        power_grid=(),
    )
    power = E3bPowerPrefixReceipt(
        schema_version=2,
        protocol_lock_sha256=lock.sha256,
        registry_sha256=lock.registry_sha256,
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_coverage_receipt_sha256=content_sha256(
            {"kind": "prospective_test_pilot_coverage"}
        ),
        evidence_manifest_sha256=content_sha256(
            {"kind": "prospective_test_pilot_evidence"}
        ),
        inventory_sha256=inventory.sha256,
        protocol_sha256=E3B_POWER_PREFIX_PROTOCOL_SHA256,
        family_power_commitments=_e3b_family_commitments(power_sizing),
        selected_final_blocks=12,
        selected_final_prefix=tuple(range(4, 16)),
    )
    signer = Ed25519PrivateKey.generate()
    signed_power = SignedE3bPowerPrefixReceipt(power, *_sign(signer, power, 131))
    final_materialization = replace(
        preliminary_final,
        source_decision_sha256=signed_power.sha256,
    )
    signed_materialization = SignedStageMaterializationReceipt(
        final_materialization,
        *_sign(signer, final_materialization, 132),
    )
    registry_receipt = object.__new__(FormalRegistryVerificationReceipt)
    for name, value in (
        ("prior_receipt", None),
        ("inventory_sha256", inventory.sha256),
        ("signed_protocol_lock", SimpleNamespace(payload=lock)),
        ("appended_signed_materializations", (signed_materialization,)),
        ("appended_signed_coverage", ()),
        ("appended_signed_e3b_power_prefixes", (signed_power,)),
    ):
        object.__setattr__(registry_receipt, name, value)
    monkeypatch.setattr(
        FormalRegistryVerificationReceipt,
        "revalidate",
        lambda self, *, current_ns: SimpleNamespace(
            inventory_sha256=self.inventory_sha256
        ),
    )
    authority = gpu_hour_authority.verify_registered_prospective_gpu_hour_authority(
        registry_receipt=registry_receipt,
        pilot_materialization=pilot_materialization,
        final_materialization=final_materialization,
        current_ns=NOW_NS,
    )
    assert authority.signed_authority_sha256 == signed_power.sha256
    assert authority.selected_final_prefix == tuple(range(4, 16))

    monkeypatch.setattr(
        "lightcone_spec.cli.main._load_formal_registry_receipt_path",
        lambda _path, *, now_ns: registry_receipt,
    )
    monkeypatch.setattr(
        gpu_hour_authority,
        "revalidate_persisted_stage_gpu_hour_source_manifest",
        lambda _path, **_kwargs: pilot_source,
    )
    registry_path = tmp_path / "prospective-registry.json"
    pilot_materialization_path = tmp_path / "prospective-pilots.json"
    pilot_envelope_path = tmp_path / "prospective-pilot-envelope.json"
    runtime_path = tmp_path / "prospective-runtime.json"
    inventory_path = tmp_path / "prospective-inventory.json"
    source_output = tmp_path / "prospective-source.json"
    envelope_output = tmp_path / "prospective-envelope.json"
    for path, value in (
        (registry_path, {"kind": "typed_registry_loader_fixture"}),
        (
            pilot_materialization_path,
            stage_materialization_receipt_to_dict(pilot_materialization),
        ),
        (pilot_envelope_path, stage_gpu_hour_envelope_to_dict(pilot_envelope)),
        (runtime_path, formal_runtime_authority_manifest_to_dict(runtime_manifest)),
        (inventory_path, inventory.to_dict()),
    ):
        _write_json(path, value)
    assert (
        main(
            [
                "materialize-prospective-stage-gpu-hours",
                "--stage",
                "E3b",
                "--registry-verification-receipt",
                str(registry_path),
                "--pilot-materialization",
                str(pilot_materialization_path),
                "--pilot-envelope",
                str(pilot_envelope_path),
                "--pilot-source-manifest",
                str(pilot_source_path),
                "--formal-runtime-authority-manifest",
                str(runtime_path),
                "--inventory",
                str(inventory_path),
                "--source-output",
                str(source_output),
                "--now-ns",
                str(NOW_NS),
                "--output",
                str(envelope_output),
            ]
        )
        == 0
    )
    prospective_source = gpu_hour_authority.ProspectiveGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(source_output).reopen()
    )
    envelope = stage_gpu_hour_envelope_from_dict(_load_bound_json(envelope_output))
    assert prospective_source.signed_power_authority_sha256 == signed_power.sha256
    assert prospective_source.costs[0].category == "actual_tuning"
    assert prospective_source.costs[1].category == "projected_final"
    assert envelope.signed_pilot_receipt_sha256 == prospective_source.sha256
    assert envelope.estimate.profile_reserve_gpu_hours == 0.0


def test_stage_gpu_hour_receipt_reopens_sources_reserves_once_and_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_private = Ed25519PrivateKey.generate()
    controller_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    bundle = _bundle(controller_private)
    monkeypatch.setattr(
        root_module, "load_source_release_ed25519_root", lambda: root_binding
    )
    inventory = _inventory()
    runtime_manifest = _runtime_manifest()
    lock, registry_receipt, materialization, registry_path = _registry_receipt(
        tmp_path,
        monkeypatch,
        inventory=inventory,
        runtime_manifest=runtime_manifest,
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=bundle,
    )
    coverage = next(
        row.payload
        for row in registry_receipt.cumulative_signed_coverage
        if row.payload.materialization_receipt_sha256 == materialization.sha256
    )
    source, envelope, source_path = _preflight_gpu_hour_source(
        tmp_path,
        lock=lock,
        materialization=materialization,
        coverage=coverage,
        inventory=inventory,
        runtime_manifest=runtime_manifest,
    )

    def revalidate_typed_preflight_source(path, **kwargs):
        binding = CanonicalJsonProofBinding.bind(path)
        reopened = PreflightGpuHourSourceManifest.from_dict(binding.reopen())
        assert reopened == source
        assert kwargs["materialization"] == materialization
        assert kwargs["stage_coverage"] == coverage
        assert kwargs["inventory"] == inventory
        for observation in reopened.observations:
            if (
                CanonicalJsonProofBinding.bind(observation.timing_proof.absolute_path)
                != observation.timing_proof
            ):
                raise ValueError("preflight timing proof identity changed")
            observation.replay_reservation.revalidate()
        return reopened

    monkeypatch.setattr(
        formal_gpu_hour_registry,
        "revalidate_persisted_preflight_gpu_hour_source_manifest",
        revalidate_typed_preflight_source,
    )
    signed_envelope = SignedStageGpuHourEnvelope(
        envelope, *_sign(controller_private, envelope, 10)
    )
    source_binding = CanonicalJsonProofBinding.bind(source_path)
    lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_stage_gpu_hour_registry_lineage",
            "registry_verification_receipt_sha256": registry_receipt.sha256,
            "signed_stage_gpu_hour_envelope_sha256": signed_envelope.sha256,
            "source_manifest": source_binding.to_dict(),
            "runtime_authority_manifest_sha256": runtime_manifest.sha256,
            "inventory_sha256": inventory.sha256,
        }
    )
    authorization = _deployment(
        root_private,
        root_binding=root_binding,
        bundle=bundle,
        inventory_sha256=inventory.sha256,
        nonce=11,
    )
    control = _control(
        controller_private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=authorization,
        artifact_type="rank_aggregate",
        artifact_sha256=signed_envelope.sha256,
        protocol_sha256=FORMAL_GPU_HOUR_REGISTRY_PROTOCOL_SHA256,
        lineage_sha256=lineage,
        nonce=12,
    )
    replay_directory = tmp_path / "gpu-hour-replay"
    replay_directory.mkdir()
    store = ChallengeReplayStore(str(replay_directory.resolve()))
    envelope_path = tmp_path / "signed-gpu-hours.json"
    runtime_manifest_path = tmp_path / "runtime-authority.json"
    inventory_path = tmp_path / "inventory.json"
    control_path = tmp_path / "gpu-hour-control.json"
    receipt_path = tmp_path / "formal-stage-gpu-hours.json"
    for path, value in (
        (envelope_path, signed_stage_gpu_hour_to_dict(signed_envelope)),
        (
            runtime_manifest_path,
            formal_runtime_authority_manifest_to_dict(runtime_manifest),
        ),
        (inventory_path, inventory.to_dict()),
        (control_path, control.to_dict()),
    ):
        _write_json(path, value)
    assert (
        main(
            [
                "reserve-formal-stage-gpu-hours",
                "--registry-verification-receipt",
                str(registry_path),
                "--signed-envelope",
                str(envelope_path),
                "--source-manifest",
                str(source_path),
                "--formal-runtime-authority-manifest",
                str(runtime_manifest_path),
                "--inventory",
                str(inventory_path),
                "--control-attestation",
                str(control_path),
                "--control-replay-store",
                str(replay_directory),
                "--now-ns",
                str(NOW_NS),
                "--output",
                str(receipt_path),
            ]
        )
        == 0
    )
    receipt = FormalStageGpuHourVerificationReceipt.from_dict(
        _load_bound_json(receipt_path)
    )
    receipt.revalidate(current_ns=NOW_NS + 1)
    decoded = FormalStageGpuHourVerificationReceipt.from_dict(receipt.to_dict())
    assert decoded == receipt
    decoded.revalidate(current_ns=NOW_NS + 1)
    aggregate_path = tmp_path / "formal-study-gpu-hours.json"
    assert (
        main(
            [
                "aggregate-formal-study-gpu-hours",
                "--registry-verification-receipt",
                str(registry_path),
                "--stage-receipt",
                str(receipt_path),
                "--now-ns",
                str(NOW_NS + 1),
                "--output",
                str(aggregate_path),
            ]
        )
        == 0
    )
    estimate = _load_bound_json(aggregate_path)
    assert isinstance(estimate, dict)
    assert estimate["status"] == "COMPLETE"
    assert estimate["schema_version"] == 2
    assert tuple(estimate["covered_materialized_cell_ids"]) == tuple(
        row.cell_id for row in materialization.cells
    )
    assert estimate["compute_gpu_hours"] == pytest.approx(
        envelope.estimate.compute_gpu_hours
    )
    assert estimate["reserved_gpu_hours"] == pytest.approx(
        envelope.estimate.reserved_gpu_hours
    )
    assert FormalStudyGpuHourEstimate.from_dict(estimate).to_dict() == estimate
    direct_estimate = aggregate_formal_study_gpu_hours(
        registry_receipt=registry_receipt,
        stage_receipts=(decoded,),
        current_ns=NOW_NS + 1,
    )
    assert direct_estimate.to_dict() == estimate
    legacy_direct_path = (tmp_path / "legacy-direct-registry.json").resolve()
    publish_canonical_json_no_replace(
        legacy_direct_path,
        formal_registry_verification_receipt_to_dict(registry_receipt),
    )
    legacy_rewrapped = replace(
        decoded,
        registry_receipt_source=CanonicalJsonProofBinding.bind(legacy_direct_path),
    )
    with pytest.raises(ValueError, match="proof-carrying schema-5"):
        legacy_rewrapped.revalidate(current_ns=NOW_NS + 1)
    *_, lifecycle_source_path, _lifecycle_envelope = _case(
        tmp_path / "lifecycle-wrapper-extraction",
        gangs=(("GPU-0",),),
        starts=(20_000_000_000,),
        monkeypatch=monkeypatch,
    )
    lifecycle_source = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(lifecycle_source_path).reopen()
    )
    shared_challenge = lifecycle_source.observations[
        0
    ].execution_replay_reservation.challenge_sha256s[0]
    conflicting_stage_reservation = _reservation_for_challenges(
        tmp_path,
        label="stage-wrapper-reuses-execution-challenge",
        challenges=(
            shared_challenge,
            content_sha256({"kind": "foreign_stage_wrapper_challenge"}),
        ),
    )
    lifecycle_receipt = replace(
        decoded,
        source_manifest=CanonicalJsonProofBinding.bind(lifecycle_source_path),
        reservation=conflicting_stage_reservation,
    )
    with pytest.raises(ValueError, match="repeats replay challenge"):
        formal_gpu_hour_registry._formal_stage_evidence_identities(
            lifecycle_receipt,
            lifecycle_source,
        )
    with pytest.raises(ValueError, match="repeats a materialization"):
        aggregate_formal_study_gpu_hours(
            registry_receipt=registry_receipt,
            stage_receipts=(decoded, decoded),
            current_ns=NOW_NS + 1,
        )
    with pytest.raises(ValueError, match="already consumed"):
        reserve_formal_stage_gpu_hour_verification_receipt(
            registry_receipt=registry_receipt,
            registry_receipt_path=str(registry_path),
            signed_envelope=signed_envelope,
            source_manifest_path=str(source_path),
            formal_runtime_authority_manifest=runtime_manifest,
            inventory=inventory,
            control_attestation=control,
            replay_store=store,
            now_ns=NOW_NS,
        )
    tampered = copy.deepcopy(receipt.to_dict())
    tampered["signed_envelope"]["payload"]["estimate"]["compute_gpu_hours"] = 11.0
    with pytest.raises(ValueError):
        FormalStageGpuHourVerificationReceipt.from_dict(tampered)
    Path(source.observations[0].timing_proof.absolute_path).write_text(
        '{"kind":"tampered-preflight-timing-proof"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="proof differs|identity changed"):
        receipt.revalidate(current_ns=NOW_NS + 2)
