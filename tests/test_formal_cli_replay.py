from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_control_attestation import (
    HARDWARE_SHA256,
    INVENTORY_SHA256,
    NOW_NS,
    _authority,
    _bundle,
    _root_binding,
)

from lightcone_spec.cli.main import (
    _formal_single_control_lineage_sha256,
    _reserve_formal_registry_verification,
    _verify_signed_protocol_lock,
    _verify_signed_stage_coverage,
    _verify_signed_stage_materialization,
    _write_json,
)
from lightcone_spec.experiments.formal_protocol import (
    CandidateStateReplay,
    CandidateStateTerminalPair,
    ProtocolLock,
    SignedProtocolLock,
    TtsL0CandidateStateCoverage,
    content_sha256,
)
from lightcone_spec.experiments.formal_registry import (
    assemble_and_reserve_formal_registry_manifest,
    extend_formal_registry_verification_receipt,
    formal_registry_verification_receipt_from_dict,
    formal_registry_verification_receipt_to_dict,
    reserve_formal_registry_verification_receipt,
    signed_protocol_lock_to_dict,
    signed_stage_coverage_to_dict,
    signed_stage_materialization_to_dict,
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.formal_registry_layers import (
    bind_formal_registry_layer_artifact,
    load_formal_registry_verification_receipt_path,
    load_formal_signed_coverage_path,
    load_formal_signed_materialization_path,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    SignedStageCoverageReceipt,
    SignedStageMaterializationReceipt,
    StageCellDisposition,
    StageCoverageReceipt,
    materialize_preflight,
)
from lightcone_spec.runtime import release_trust_root as root_module
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
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    deployment_policy_subject_sha256,
)


def _sha(label: str) -> str:
    return content_sha256({"formal-cli-replay": label})


def _sign(
    private: Ed25519PrivateKey,
    payload: object,
    *,
    nonce_byte: bytes,
):
    digest = content_sha256(payload)
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=f"formal-wrapper-{nonce_byte.hex()}",
        nonce_base64=base64.b64encode(nonce_byte * 32).decode("ascii"),
        subject_sha256=digest,
        issued_ns=1_600_000_000,
        expires_ns=2_600_000_000,
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
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


def _control(
    private: Ed25519PrivateKey,
    *,
    binding,
    bundle,
    authorization,
    artifact_type: str,
    artifact_sha256: str,
    protocol_sha256: str,
    lineage_sha256: str,
    nonce_byte: bytes,
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
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=f"formal-control-{nonce_byte.hex()}",
        nonce_base64=base64.b64encode(nonce_byte * 32).decode("ascii"),
        subject_sha256=subject.sha256,
        issued_ns=1_600_000_000,
        expires_ns=2_600_000_000,
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    attestation = SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id="validation-signer",
        key_id="validation-signer-key",
        environment="release",
        public_key_base64=base64.b64encode(public).decode("ascii"),
        challenge_sha256=challenge.sha256,
        payload_sha256=artifact_sha256,
        signature_base64=base64.b64encode(
            private.sign(attestation_message(challenge, payload_sha256=artifact_sha256))
        ).decode("ascii"),
    )
    return ControlArtifactAttestation(
        schema_version=1,
        kind="lightcone_control_artifact_attestation",
        subject=subject,
        hardware_envelope_sha256=HARDWARE_SHA256,
        trust_anchor_sha256=binding.sha256,
        trust_bundle_sha256=bundle.sha256,
        trusted_attester_policy_sha256=bundle.trusted_attester_policy.sha256,
        deployment_policy_authorization=authorization,
        challenge=challenge,
        attestation=attestation,
    )


def _deployment_authorization(
    root_private: Ed25519PrivateKey,
    *,
    binding,
    bundle,
    nonce_byte: bytes,
) -> DeploymentPolicyAuthorization:
    subject_sha256 = deployment_policy_subject_sha256(
        root_manifest_sha256=binding.semantic_sha256,
        inventory_sha256=INVENTORY_SHA256,
        bundle_sha256=bundle.sha256,
    )
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=f"formal-deployment-{nonce_byte.hex()}",
        nonce_base64=base64.b64encode(nonce_byte * 32).decode("ascii"),
        subject_sha256=subject_sha256,
        issued_ns=1_500_000_000,
        expires_ns=3_000_000_000,
    )
    return DeploymentPolicyAuthorization(
        schema_version=1,
        kind="lightcone_deployment_policy_authorization",
        root_manifest_sha256=binding.semantic_sha256,
        inventory_sha256=INVENTORY_SHA256,
        bundle=bundle,
        challenge=challenge,
        signature_base64=base64.b64encode(
            root_private.sign(
                attestation_message(challenge, payload_sha256=bundle.sha256)
            )
        ).decode("ascii"),
    )


def _lock(root_manifest_sha256: str) -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="lightcone-formal-cli-replay-v2",
        code_git_head="1" * 40,
        code_git_tree="2" * 40,
        patch_manifest_sha256=_sha("patch"),
        registry_sha256=build_industrial_registry().sha256,
        english_protocol_sha256=_sha("english"),
        chinese_protocol_sha256=_sha("chinese"),
        tts_calibration_authority_sha256=_sha("tts"),
        chronobelief_authority_sha256=_sha("chronobelief"),
        e1_recipe_anchor_authority_sha256=_sha("e1"),
        e2_recipe_grid_authority_sha256=_sha("e2"),
        formal_runtime_authority_manifest_sha256=_sha("formal-runtime"),
        offline_release_trust_root_sha256=root_manifest_sha256,
        prepared_model_content_authorization_sha256=_sha("prepared"),
        formal_workload_e3a_authorization_sha256=_sha("e3a-workload"),
        formal_workload_e0_authorization_sha256=_sha("e0-workload"),
        burstgpt_shape_authorization_sha256=_sha("burstgpt"),
        native_runtime_qualification_protocol_sha256=_sha("native-protocol"),
        native_runtime_qualification_runner_sha256=_sha("native-runner"),
        native_runtime_qualification_test_set_sha256=_sha("native-tests"),
        compile_qualification_protocol_sha256=_sha("compile-protocol"),
        compile_qualification_runner_sha256=_sha("compile-runner"),
        compile_qualification_test_set_sha256=_sha("compile-tests"),
        exactness_qualification_protocol_sha256=_sha("exactness-protocol"),
        exactness_qualification_runner_sha256=_sha("exactness-runner"),
        exactness_qualification_test_set_sha256=_sha("exactness-tests"),
    )


def _candidate_coverage(materialization) -> TtsL0CandidateStateCoverage:
    exactness = next(
        cell
        for cell in materialization.cells
        if cell.task == "exactness_memory_telemetry_preflight"
    )
    plan = _sha("candidate-plan")
    shared = {
        "source_round": 1,
        "source_version": 0,
        "source_state_sha256": _sha("candidate-source"),
        "trainable_plan_sha256": plan,
        "candidate_bytes_sha256": _sha("candidate-bytes"),
        "optimizer_state_bytes_sha256": _sha("candidate-optimizer-state"),
        "proposal_evidence_sha256": _sha("candidate-proposal"),
    }
    return TtsL0CandidateStateCoverage(
        schema_version=1,
        stage="preflight",
        scope="preflight_exactness_qualification",
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        pair_id=_sha("candidate-pair"),
        tts_cell_id=_sha("candidate-tts-fixture"),
        l0_naive_cell_id=_sha("candidate-l0-fixture"),
        tts_native_replay_pointer_sha256=_sha("candidate-tts-pointer"),
        l0_naive_native_replay_pointer_sha256=_sha("candidate-l0-pointer"),
        qualification_cell_id=exactness.cell_id,
        source_round_plan_sha256=_sha("candidate-round-plan"),
        trainable_plan_sha256=plan,
        expected_source_rounds=(1,),
        tts_observations=(
            CandidateStateReplay(
                method_role="TTS",
                cell_id=_sha("candidate-tts-fixture"),
                native_replay_pointer_sha256=_sha("candidate-tts-pointer"),
                run_id="preflight-tts-round-0",
                publication_policy="fixed_barrier",
                **shared,
            ),
        ),
        l0_naive_observations=(
            CandidateStateReplay(
                method_role="L0-naive",
                cell_id=_sha("candidate-l0-fixture"),
                native_replay_pointer_sha256=_sha("candidate-l0-pointer"),
                run_id="preflight-l0-round-0",
                publication_policy="first_ready",
                **shared,
            ),
        ),
        terminal_pairs=(
            CandidateStateTerminalPair(
                source_round=1,
                tts_cell_id=_sha("candidate-tts-fixture"),
                l0_naive_cell_id=_sha("candidate-l0-fixture"),
                tts_run_id="preflight-tts-round-0",
                l0_naive_run_id="preflight-l0-round-0",
                tts_native_replay_pointer_sha256=_sha("candidate-tts-pointer"),
                l0_naive_native_replay_pointer_sha256=_sha("candidate-l0-pointer"),
                proposal_evidence_sha256=_sha("candidate-proposal"),
                tts_terminal_receipt_sha256=_sha("candidate-tts-terminal"),
                l0_naive_terminal_receipt_sha256=_sha("candidate-l0-terminal"),
            ),
        ),
    )


def test_diagnostic_cli_verification_does_not_consume_formal_batch_challenges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private, binding, bundle, authorization = _authority(monkeypatch)
    lock = _lock(binding.semantic_sha256)
    lock_digest, lock_challenge, lock_attestation = _sign(
        private, lock, nonce_byte=b"l"
    )
    signed_lock = SignedProtocolLock(
        lock, lock_digest, lock_challenge, lock_attestation
    )
    materialization = materialize_preflight(
        protocol_lock_sha256=lock.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    mat_digest, mat_challenge, mat_attestation = _sign(
        private, materialization, nonce_byte=b"m"
    )
    signed_materialization = SignedStageMaterializationReceipt(
        materialization, mat_digest, mat_challenge, mat_attestation
    )
    coverage = StageCoverageReceipt(
        schema_version=2,
        stage="preflight",
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(
            StageCellDisposition(
                stage="preflight",
                cell_id=cell.cell_id,
                status="COMPLETE",
                reason_code="terminal_complete",
                terminal_receipt_sha256=_sha(f"terminal-{cell.cell_id}"),
            )
            for cell in materialization.cells
        ),
        tts_l0_candidate_state_coverages=(_candidate_coverage(materialization),),
    )
    coverage_digest, coverage_challenge, coverage_attestation = _sign(
        private, coverage, nonce_byte=b"c"
    )
    signed_coverage = SignedStageCoverageReceipt(
        coverage, coverage_digest, coverage_challenge, coverage_attestation
    )

    signed_rows = (
        (signed_lock, "dispatch"),
        (signed_materialization, "dispatch"),
        (signed_coverage, "rank_aggregate"),
    )
    single_controls = []
    for index, (signed, artifact_type) in enumerate(signed_rows):
        single_controls.append(
            _control(
                private,
                binding=binding,
                bundle=bundle,
                authorization=authorization,
                artifact_type=artifact_type,
                artifact_sha256=signed.sha256,
                protocol_sha256=lock.sha256,
                lineage_sha256=_formal_single_control_lineage_sha256(
                    signed_artifact_sha256=signed.sha256,
                    protocol_lock_sha256=lock.sha256,
                ),
                nonce_byte=bytes((ord("d") + index,)),
            )
        )

    lock_path = tmp_path / "signed-lock.json"
    materialization_path = tmp_path / "signed-materialization.json"
    coverage_path = tmp_path / "signed-coverage.json"
    raw_materialization_path = tmp_path / "materialization.json"
    paths = (lock_path, materialization_path, coverage_path)
    values = (
        signed_protocol_lock_to_dict(signed_lock),
        signed_stage_materialization_to_dict(signed_materialization),
        signed_stage_coverage_to_dict(signed_coverage),
    )
    control_paths = []
    for path, value, control in zip(paths, values, single_controls, strict=True):
        _write_json(path, value)
        control_path = path.with_name(f"{path.stem}-control.json")
        _write_json(control_path, control.to_dict())
        control_paths.append(control_path)
    _write_json(
        raw_materialization_path,
        stage_materialization_receipt_to_dict(materialization),
    )
    replay_path = tmp_path / "replay"
    common = {
        "inventory_sha256": INVENTORY_SHA256,
        "control_replay_store": str(replay_path),
        "now_ns": NOW_NS,
    }
    _verify_signed_protocol_lock(
        argparse.Namespace(
            signed_lock=str(lock_path),
            control_attestation=str(control_paths[0]),
            output=str(tmp_path / "verified-lock.json"),
            **common,
        )
    )
    _verify_signed_stage_materialization(
        argparse.Namespace(
            signed_receipt=str(materialization_path),
            control_attestation=str(control_paths[1]),
            output=str(tmp_path / "verified-materialization.json"),
            **common,
        )
    )
    _verify_signed_stage_coverage(
        argparse.Namespace(
            signed_receipt=str(coverage_path),
            materialization=str(raw_materialization_path),
            control_attestation=str(control_paths[2]),
            output=str(tmp_path / "verified-coverage.json"),
            **common,
        )
    )
    assert not replay_path.exists()

    raw_signed_materialization = tmp_path / "formal-raw-signed-materialization.json"
    raw_signed_coverage = tmp_path / "formal-raw-signed-coverage.json"
    publish_canonical_json_no_replace(
        raw_signed_materialization,
        signed_stage_materialization_to_dict(signed_materialization),
    )
    publish_canonical_json_no_replace(
        raw_signed_coverage,
        signed_stage_coverage_to_dict(signed_coverage),
    )
    with pytest.raises(ValueError, match="typed predecessor reducer proof"):
        load_formal_signed_materialization_path(
            raw_signed_materialization,
            now_ns=NOW_NS,
        )
    with pytest.raises(ValueError, match="registry proof wrapper"):
        load_formal_signed_coverage_path(
            raw_signed_coverage,
            formal_stage_prefix_paths=(),
            now_ns=NOW_NS,
        )

    coverage_artifact_types = {signed.sha256: kind for signed, kind in signed_rows}
    coverage_ordered_digests = tuple(sorted(coverage_artifact_types))
    coverage_batch_lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": tuple(
                (digest, coverage_artifact_types[digest])
                for digest in coverage_ordered_digests
            ),
        }
    )
    coverage_batch_controls = tuple(
        _control(
            private,
            binding=binding,
            bundle=bundle,
            authorization=authorization,
            artifact_type=coverage_artifact_types[digest],
            artifact_sha256=digest,
            protocol_sha256=lock.sha256,
            lineage_sha256=coverage_batch_lineage,
            nonce_byte=bytes((ord("j") + index,)),
        )
        for index, digest in enumerate(coverage_ordered_digests)
    )
    replay_path.mkdir()
    with pytest.raises(ValueError, match="candidate replay proof artifacts"):
        assemble_and_reserve_formal_registry_manifest(
            signed_lock,
            signed_materializations=(signed_materialization,),
            signed_coverage=(signed_coverage,),
            control_attestations=coverage_batch_controls,
            expected_inventory_sha256=INVENTORY_SHA256,
            replay_store=ChallengeReplayStore(str(replay_path.resolve())),
            now_ns=NOW_NS,
        )
    assert not tuple(replay_path.glob("reservation-*.json"))

    formal_signed_rows = signed_rows[:2]
    artifact_types = {signed.sha256: kind for signed, kind in formal_signed_rows}
    ordered_digests = tuple(sorted(artifact_types))
    batch_lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": tuple(
                (digest, artifact_types[digest]) for digest in ordered_digests
            ),
        }
    )
    batch_controls = tuple(
        _control(
            private,
            binding=binding,
            bundle=bundle,
            authorization=authorization,
            artifact_type=artifact_types[digest],
            artifact_sha256=digest,
            protocol_sha256=lock.sha256,
            lineage_sha256=batch_lineage,
            nonce_byte=bytes((ord("g") + index,)),
        )
        for index, digest in enumerate(ordered_digests)
    )
    invalid_control = replace(
        batch_controls[-1],
        attestation=replace(
            batch_controls[-1].attestation,
            signature_base64=base64.b64encode(b"\x00" * 64).decode("ascii"),
        ),
    )
    with pytest.raises(ValueError, match="signature"):
        assemble_and_reserve_formal_registry_manifest(
            signed_lock,
            signed_materializations=(signed_materialization,),
            signed_coverage=(),
            control_attestations=(*batch_controls[:-1], invalid_control),
            expected_inventory_sha256=INVENTORY_SHA256,
            replay_store=ChallengeReplayStore(str(replay_path.resolve())),
            now_ns=NOW_NS,
        )
    assert not tuple(replay_path.glob("reservation-*.json"))
    manifest = assemble_and_reserve_formal_registry_manifest(
        signed_lock,
        signed_materializations=(signed_materialization,),
        signed_coverage=(),
        control_attestations=batch_controls,
        expected_inventory_sha256=INVENTORY_SHA256,
        replay_store=ChallengeReplayStore(str(replay_path.resolve())),
        now_ns=NOW_NS,
    )
    assert manifest.status == "MATERIALIZED_PENDING_COVERAGE"
    with pytest.raises(ValueError, match="already consumed"):
        assemble_and_reserve_formal_registry_manifest(
            signed_lock,
            signed_materializations=(signed_materialization,),
            signed_coverage=(),
            control_attestations=batch_controls,
            expected_inventory_sha256=INVENTORY_SHA256,
            replay_store=ChallengeReplayStore(str(replay_path.resolve())),
            now_ns=NOW_NS,
        )


def test_durable_registry_root_allows_preflight_append_without_lock_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root_private = Ed25519PrivateKey.generate()
    artifact_private = Ed25519PrivateKey.generate()
    binding = _root_binding(root_private)
    bundle = _bundle(artifact_private)
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: binding,
    )
    lock = _lock(binding.semantic_sha256)
    lock_digest, lock_challenge, lock_attestation = _sign(
        artifact_private, lock, nonce_byte=b"r"
    )
    signed_lock = SignedProtocolLock(
        lock, lock_digest, lock_challenge, lock_attestation
    )
    root_lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": ((signed_lock.sha256, "dispatch"),),
        }
    )
    root_authorization = _deployment_authorization(
        root_private,
        binding=binding,
        bundle=bundle,
        nonce_byte=b"1",
    )
    root_control = _control(
        artifact_private,
        binding=binding,
        bundle=bundle,
        authorization=root_authorization,
        artifact_type="dispatch",
        artifact_sha256=signed_lock.sha256,
        protocol_sha256=lock.sha256,
        lineage_sha256=root_lineage,
        nonce_byte=b"2",
    )
    replay_path = tmp_path / "durable-replay"
    replay_path.mkdir()
    store = ChallengeReplayStore(str(replay_path.resolve()))
    root_receipt = reserve_formal_registry_verification_receipt(
        signed_lock,
        control_attestation=root_control,
        expected_inventory_sha256=INVENTORY_SHA256,
        replay_store=store,
        now_ns=NOW_NS,
    )
    assert root_receipt.manifest.status == "LOCKED"

    materialization = materialize_preflight(
        protocol_lock_sha256=lock.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    mat_digest, mat_challenge, mat_attestation = _sign(
        artifact_private, materialization, nonce_byte=b"3"
    )
    signed_materialization = SignedStageMaterializationReceipt(
        materialization, mat_digest, mat_challenge, mat_attestation
    )
    extension_lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": ((signed_materialization.sha256, "dispatch"),),
            "prior_registry_verification_receipt_sha256": root_receipt.sha256,
        }
    )
    extension_authorization = _deployment_authorization(
        root_private,
        binding=binding,
        bundle=bundle,
        nonce_byte=b"4",
    )
    extension_control = _control(
        artifact_private,
        binding=binding,
        bundle=bundle,
        authorization=extension_authorization,
        artifact_type="dispatch",
        artifact_sha256=signed_materialization.sha256,
        protocol_sha256=lock.sha256,
        lineage_sha256=extension_lineage,
        nonce_byte=b"5",
    )
    appended = extend_formal_registry_verification_receipt(
        root_receipt,
        appended_signed_materializations=(signed_materialization,),
        control_attestations=(extension_control,),
        replay_store=store,
        now_ns=NOW_NS,
    )
    assert appended.manifest.status == "MATERIALIZED_PENDING_COVERAGE"
    assert appended.manifest.prior_registry_verification_receipt_sha256 == (
        root_receipt.sha256
    )
    assert root_receipt.reservation.reservation_sha256 != (
        appended.reservation.reservation_sha256
    )
    assert len(tuple(replay_path.glob("reservation-*.json"))) == 2
    appended.revalidate(current_ns=NOW_NS + 1)
    encoded = formal_registry_verification_receipt_to_dict(appended)
    decoded = formal_registry_verification_receipt_from_dict(encoded)
    assert decoded == appended
    decoded.revalidate(current_ns=NOW_NS + 1)
    direct_appended_path = tmp_path / "forbidden-direct-appended-receipt.json"
    publish_canonical_json_no_replace(
        direct_appended_path,
        formal_registry_verification_receipt_to_dict(appended),
    )
    with pytest.raises(ValueError, match="bounded proof-replay layer"):
        load_formal_registry_verification_receipt_path(
            direct_appended_path,
            now_ns=NOW_NS + 1,
        )

    with pytest.raises(ValueError, match="ProtocolLock proof-replay wrapper"):
        bind_formal_registry_layer_artifact(
            root_receipt,
            prior_layer_path=None,
            signed_materialization_paths=(),
            signed_coverage_paths=(),
            formal_stage_prefix_paths=(),
        )

    tampered = copy.deepcopy(encoded)
    tampered["manifest"]["status"] = "COVERED"
    with pytest.raises(ValueError):
        formal_registry_verification_receipt_from_dict(tampered)

    cli_replay_path = tmp_path / "durable-cli-replay"
    cli_replay_path.mkdir()
    signed_lock_path = tmp_path / "durable-cli-signed-lock.json"
    root_control_path = tmp_path / "durable-cli-root-control.json"
    publish_canonical_json_no_replace(
        signed_lock_path,
        signed_protocol_lock_to_dict(signed_lock),
    )
    publish_canonical_json_no_replace(root_control_path, root_control.to_dict())
    with pytest.raises(ValueError, match="proof-replay wrapper"):
        _reserve_formal_registry_verification(
            argparse.Namespace(
                signed_protocol_lock=str(signed_lock_path),
                control_attestation=str(root_control_path),
                inventory_sha256=INVENTORY_SHA256,
                control_replay_store=str(cli_replay_path),
                now_ns=NOW_NS,
                output=str(tmp_path / "forbidden-root-layer.json"),
            )
        )
    assert not tuple(cli_replay_path.iterdir())
